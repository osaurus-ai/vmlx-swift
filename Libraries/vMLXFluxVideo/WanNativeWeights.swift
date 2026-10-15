// Wan 2.1 native inference. Architecture follows ml-explore/mlx-examples
// video/wan2.1 (Copyright 2026 Apple Inc., MIT license).
import Foundation
@preconcurrency import MLX
import MLXNN
import vMLXFluxKit

/// Raw Wan safetensors component layout. Missing weights are errors; no layer
/// creates random parameters. PyTorch convolution axes are converted once.
final class WanNativeWeights {
    let arrays: [String: MLXArray]
    let transformerCompute: Bool
    private var convolutions: [String: MLXArray] = [:]
    init(_ arrays: [String: MLXArray], transformerCompute: Bool = false) {
        self.arrays = arrays; self.transformerCompute = transformerCompute
    }
    func optional(_ name: String) -> MLXArray? { arrays[name] ?? arrays["model." + name] }
    func tensor(_ name: String) throws -> MLXArray {
        guard let value = optional(name) else { throw FluxError.invalidRequest("missing Wan tensor \(name)") }
        return value
    }
    func linear(_ name: String, _ x: MLXArray) throws -> MLXArray {
        // Original Wan checkpoints store fp32 weights, but the released runtime
        // uses bf16 autocast for transformer attention/FFN. Keep time embedding,
        // modulation/residual arithmetic and the final head in fp32.
        let x = transformerCompute && !name.hasPrefix("time_") && name != "head.head"
            ? x.asType(.bfloat16) : x
        let weight = try tensor(name + ".weight")
        let output: MLXArray
        if let scales = optional(name + ".scales") {
            guard let bits = [2, 3, 4, 5, 6, 8].first(where: { weight.dim(1) * 32 / $0 == x.dim(-1) }),
                  scales.dim(1) > 0, x.dim(-1) % scales.dim(1) == 0 else {
                throw FluxError.invalidRequest("invalid Wan quantization for \(name)")
            }
            output = quantizedMM(x, weight, scales: scales, biases: optional(name + ".biases"),
                transpose: true, groupSize: x.dim(-1) / scales.dim(1), bits: bits)
        } else {
            guard weight.ndim == 2, weight.dim(1) == x.dim(-1) else {
                throw FluxError.invalidRequest("invalid Wan linear shape for \(name)")
            }
            output = matmul(x, weight.asType(x.dtype).T)
        }
        return optional(name + ".bias").map { output + $0.asType(output.dtype) } ?? output
    }
    func rms(_ name: String, _ x: MLXArray, eps: Float = 1e-6) throws -> MLXArray {
        let weight = try optional(name + ".gamma") ?? tensor(name + ".weight")
        return MLXFast.rmsNorm(x, weight: weight.reshaped([-1]).asType(x.dtype), eps: eps)
    }
    func norm(_ name: String?, _ x: MLXArray, eps: Float = 1e-6) throws -> MLXArray {
        let weight = try name.map { try tensor($0 + ".weight").asType(x.dtype) }
        let bias = name.flatMap { optional($0 + ".bias")?.asType(x.dtype) }
        return MLXFast.layerNorm(x, weight: weight, bias: bias, eps: eps)
    }
    func conv(_ name: String, _ input: MLXArray, stride: [Int] = [1, 1, 1],
              causal: Int = 0, spatial: Int = 0, cache: MLXArray? = nil) throws -> MLXArray {
        var x = input
        var left = causal
        if let cache, causal > 0 { x = concatenated([cache, x], axis: 1); left = max(0, causal - cache.dim(1)) }
        if left > 0 || spatial > 0 {
            x = padded(x, widths: [.init(0), .init((left, 0)), .init(spatial), .init(spatial), .init(0)])
        }
        let weight: MLXArray
        if let cached = convolutions[name] { weight = cached }
        else {
            let raw = try tensor(name + ".weight")
            guard raw.ndim == 5 else { throw FluxError.invalidRequest("Wan requires a real 3D kernel at \(name)") }
            weight = raw.transposed(0, 2, 3, 4, 1)
            convolutions[name] = weight
        }
        var result = conv3d(x, weight.asType(x.dtype), stride: IntOrTriple((stride[0], stride[1], stride[2])))
        if let bias = optional(name + ".bias") { result = result + bias.asType(result.dtype) }
        return result
    }
    func conv2(_ name: String, _ x: MLXArray, stride: Int = 1, padding: Int = 0) throws -> MLXArray {
        let weight: MLXArray
        if let cached = convolutions[name] { weight = cached }
        else {
            let raw = try tensor(name + ".weight")
            weight = raw.transposed(0, 2, 3, 1); convolutions[name] = weight
        }
        var result = conv2d(x, weight, stride: IntOrPair(stride), padding: IntOrPair(padding))
        if let bias = optional(name + ".bias") { result = result + bias }
        return result
    }
    func pointwise(_ name: String, _ x: MLXArray) throws -> MLXArray {
        let raw = try tensor(name + ".weight")
        let weight = raw.reshaped([raw.dim(0), -1])
        var result = matmul(x, weight.T)
        if let bias = optional(name + ".bias") { result = result + bias }
        return result
    }
}
