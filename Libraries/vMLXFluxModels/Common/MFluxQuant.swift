//
//  MFluxQuant.swift
//  vMLXFluxModels
//
//  Shared building blocks for native mFLUX model ports (flux1, flux2, qwen).
//  These mirror the MLX group-quantization format that mflux's `*-mflux-{4,8}bit`
//  bundles use for every Linear/Embedding (`weight` + `scales` + `biases`,
//  group-quantized, bits inferred from shapes). The proven Z-Image native
//  pipeline uses an equivalent private store; this is the reusable extraction so
//  flux/qwen don't re-implement quant handling. Plain (non-quantized) params
//  (layer norms, conv bias) are loaded as-is.
//

import Foundation
@preconcurrency import MLX
import MLXNN
import vMLXFluxKit

// MARK: - Weight store

/// Wraps a `LoadedWeights` (component-organized safetensors) and builds
/// quant-aware layers from explicit checkpoint keys. Callers pass the EXACT
/// key path from the checkpoint (e.g. "t5_blocks.0.attention.SelfAttention.q").
final class MFluxStore {
    let loaded: LoadedWeights

    init(_ loaded: LoadedWeights) {
        self.loaded = loaded
    }

    /// Resolve a tensor by component + key, trying `component.key`, the raw
    /// per-component dict, and a bare top-level key. `text_encoder*` components
    /// also try a `model.` prefix (HF layout drift).
    func optionalTensor(_ component: String, _ key: String) -> MLXArray? {
        var candidates = [key]
        if component.hasPrefix("text_encoder") {
            candidates.append("model.\(key)")
        }
        for candidate in candidates {
            if let value = loaded.componentWeights[component]?[candidate] { return value }
            if let value = loaded.weights["\(component).\(candidate)"] { return value }
            if let value = loaded.weights[candidate] { return value }
        }
        return nil
    }

    func tensor(_ component: String, _ key: String) throws -> MLXArray {
        if let value = optionalTensor(component, key) { return value }
        throw FluxError.invalidRequest("missing \(component) weight \(key)")
    }

    func hasKey(_ component: String, _ key: String) -> Bool {
        optionalTensor(component, key) != nil
    }

    func linear(
        _ component: String,
        _ prefix: String,
        inputDimensions: Int,
        outputDimensions: Int,
        bias: Bool = false
    ) throws -> MFluxLinear {
        try MFluxLinear(
            weight: tensor(component, "\(prefix).weight"),
            scales: optionalTensor(component, "\(prefix).scales"),
            biases: optionalTensor(component, "\(prefix).biases"),
            bias: bias ? optionalTensor(component, "\(prefix).bias") : nil,
            inputDimensions: inputDimensions,
            outputDimensions: outputDimensions,
            name: "\(component).\(prefix)")
    }

    func linear(
        _ component: String,
        prefixes: [String],
        inputDimensions: Int,
        outputDimensions: Int,
        bias: Bool = false
    ) throws -> MFluxLinear {
        for prefix in prefixes where hasKey(component, "\(prefix).weight") {
            return try linear(
                component,
                prefix,
                inputDimensions: inputDimensions,
                outputDimensions: outputDimensions,
                bias: bias)
        }
        let names = prefixes.map { "\(component).\($0).weight" }.joined(separator: ", ")
        throw FluxError.invalidRequest("missing any of weights [\(names)]")
    }

    func embedding(
        _ component: String,
        _ prefix: String,
        dimensions: Int
    ) throws -> MFluxEmbedding {
        try MFluxEmbedding(
            weight: tensor(component, "\(prefix).weight"),
            scales: optionalTensor(component, "\(prefix).scales"),
            biases: optionalTensor(component, "\(prefix).biases"),
            dimensions: dimensions,
            name: "\(component).\(prefix)")
    }

    /// T5-style RMSNorm (no mean-subtract, no bias). `weight * x * rsqrt(mean(x^2)+eps)`.
    func rmsNorm(_ component: String, _ prefix: String, eps: Float = 1e-6) throws -> MFluxRMSNorm {
        MFluxRMSNorm(weight: try tensor(component, "\(prefix).weight"), eps: eps)
    }

    /// Standard LayerNorm with weight + bias.
    func layerNorm(_ component: String, _ prefix: String, eps: Float = 1e-5) throws -> MFluxLayerNorm {
        MFluxLayerNorm(
            weight: try tensor(component, "\(prefix).weight"),
            bias: try tensor(component, "\(prefix).bias"),
            eps: eps)
    }

    func groupNorm(_ component: String, _ prefix: String, groups: Int = 32, eps: Float = 1e-6) throws -> MFluxGroupNorm {
        MFluxGroupNorm(
            weight: try tensor(component, "\(prefix).weight"),
            bias: try tensor(component, "\(prefix).bias"),
            groups: groups, eps: eps)
    }

    func conv2d(_ component: String, _ prefix: String, stride: Int = 1, padding: Int = 0) throws -> MFluxConv2D {
        MFluxConv2D(
            weight: try tensor(component, "\(prefix).weight"),
            bias: optionalTensor(component, "\(prefix).bias"),
            stride: stride, padding: padding)
    }
}

// MARK: - Layers

/// Linear that runs quantized matmul when `scales` is present, else dense matmul.
/// Bits + group size are inferred from the packed `weight`/`scales` shapes.
final class MFluxLinear {
    private let weight: MLXArray
    private let scales: MLXArray?
    private let biases: MLXArray?
    private let bias: MLXArray?
    private let groupSize: Int
    private let bits: Int

    init(
        weight: MLXArray,
        scales: MLXArray?,
        biases: MLXArray?,
        bias: MLXArray?,
        inputDimensions: Int,
        outputDimensions: Int,
        name: String
    ) throws {
        guard weight.dim(0) == outputDimensions else {
            throw FluxError.invalidRequest("\(name) output mismatch: weight=\(weight.shape), expected output \(outputDimensions)")
        }
        self.weight = weight
        self.scales = scales
        self.biases = biases
        self.bias = bias
        if scales != nil {
            var inferred: Int?
            for candidate in [2, 3, 4, 5, 6, 8] where weight.dim(1) * 32 / candidate == inputDimensions {
                inferred = candidate; break
            }
            guard let inferred else {
                throw FluxError.invalidRequest("\(name) quantized input mismatch: weight=\(weight.shape), expected input \(inputDimensions)")
            }
            let scaleColumns = scales?.dim(1) ?? 0
            guard scaleColumns > 0, inputDimensions % scaleColumns == 0 else {
                throw FluxError.invalidRequest("\(name) invalid quantization scales \(scales?.shape ?? [])")
            }
            self.bits = inferred
            self.groupSize = inputDimensions / scaleColumns
        } else {
            guard weight.dim(1) == inputDimensions else {
                throw FluxError.invalidRequest("\(name) input mismatch: weight=\(weight.shape), expected input \(inputDimensions)")
            }
            self.bits = 0
            self.groupSize = 0
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y: MLXArray
        if let scales {
            y = quantizedMM(x, weight, scales: scales, biases: biases,
                            transpose: true, groupSize: groupSize, bits: bits, mode: .affine)
        } else {
            y = matmul(x, weight.T)
        }
        if let bias { y = y + bias }
        return y
    }
}

/// Embedding lookup that dequantizes the selected rows when quantized.
final class MFluxEmbedding {
    private let weight: MLXArray
    private let scales: MLXArray?
    private let biases: MLXArray?
    private let groupSize: Int
    private let bits: Int

    init(weight: MLXArray, scales: MLXArray?, biases: MLXArray?, dimensions: Int, name: String) throws {
        self.weight = weight
        self.scales = scales
        self.biases = biases
        if scales != nil {
            var inferred: Int?
            for candidate in [2, 3, 4, 5, 6, 8] where weight.dim(1) * 32 / candidate == dimensions {
                inferred = candidate; break
            }
            guard let inferred else {
                throw FluxError.invalidRequest("\(name) quantized dim mismatch: weight=\(weight.shape), expected \(dimensions)")
            }
            let scaleColumns = scales?.dim(1) ?? 0
            guard scaleColumns > 0, dimensions % scaleColumns == 0 else {
                throw FluxError.invalidRequest("\(name) invalid quantization scales \(scales?.shape ?? [])")
            }
            self.bits = inferred
            self.groupSize = dimensions / scaleColumns
        } else {
            guard weight.dim(1) == dimensions else {
                throw FluxError.invalidRequest("\(name) dim mismatch: weight=\(weight.shape), expected \(dimensions)")
            }
            self.bits = 0
            self.groupSize = 0
        }
    }

    func callAsFunction(_ ids: MLXArray) -> MLXArray {
        let selected = weight[ids]
        guard let scales else { return selected }
        let selectedScales = scales[ids]
        let selectedBiases = biases == nil ? nil : biases![ids]
        return dequantized(selected, scales: selectedScales, biases: selectedBiases,
                           groupSize: groupSize, bits: bits, mode: .affine)
    }
}

final class MFluxRMSNorm {
    private let weight: MLXArray
    private let eps: Float
    init(weight: MLXArray, eps: Float) { self.weight = weight; self.eps = eps }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // T5 RMSNorm computed in fp32 for stability, scaled by weight.
        let h = x.asType(.float32)
        let variance = mean(h * h, axis: -1, keepDims: true)
        let normed = h * rsqrt(variance + MLXArray(eps))
        return (weight * normed.asType(x.dtype))
    }
}

final class MFluxLayerNorm {
    private let weight: MLXArray
    private let bias: MLXArray
    private let eps: Float
    init(weight: MLXArray, bias: MLXArray, eps: Float) { self.weight = weight; self.bias = bias; self.eps = eps }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let meanValue = mean(x, axis: -1, keepDims: true)
        let centered = x - meanValue
        let variance = mean(centered * centered, axis: -1, keepDims: true)
        return weight * (centered * rsqrt(variance + MLXArray(eps))) + bias
    }
}

final class MFluxGroupNorm {
    private let weight: MLXArray
    private let bias: MLXArray
    private let groups: Int
    private let eps: Float
    init(weight: MLXArray, bias: MLXArray, groups: Int, eps: Float) {
        self.weight = weight; self.bias = bias; self.groups = groups; self.eps = eps
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let batch = x.dim(0)
        let dims = x.dim(-1)
        let rest = Array(x.shape.dropFirst().dropLast())
        let groupSize = dims / groups
        var y = x.reshaped([batch, -1, groups, groupSize])
        y = y.transposed(0, 2, 1, 3).reshaped([batch, groups, -1])
        let meanValue = mean(y, axis: -1, keepDims: true)
        let centered = y - meanValue
        let variance = mean(centered * centered, axis: -1, keepDims: true)
        y = centered * rsqrt(variance + MLXArray(eps))
        y = y.reshaped([batch, groups, -1, groupSize]).transposed(0, 2, 1, 3).reshaped([batch] + rest + [dims])
        return y * weight + bias
    }
}

final class MFluxConv2D {
    private let weight: MLXArray
    private let bias: MLXArray?
    private let stride: Int
    private let padding: Int
    init(weight: MLXArray, bias: MLXArray?, stride: Int, padding: Int) {
        // safetensors conv weight is (O, C, H, W); MLX conv2d wants (O, H, W, C).
        if weight.ndim == 4, weight.dim(2) == weight.dim(3), weight.dim(1) != weight.dim(2) {
            self.weight = weight.transposed(0, 2, 3, 1)
        } else {
            self.weight = weight
        }
        self.bias = bias
        self.stride = stride
        self.padding = padding
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = conv2d(x, weight, stride: IntOrPair(stride), padding: IntOrPair(padding))
        if let bias { y = y + bias }
        return y
    }
}
