// Architecture reference: ml-explore/mlx-examples/video/wan2.1/vae*.py.
import Foundation
@preconcurrency import MLX
import MLXNN
import vMLXFluxKit

final class WanNativeVAE {
    let weights: WanNativeWeights
    private let mean = MLXArray([-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653,
        -0.1517, 1.5508, 0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921] as [Float])
    private let std = MLXArray([2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052,
        2.0743, 3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160] as [Float])
    init(weights: [String: MLXArray]) throws {
        self.weights = WanNativeWeights(weights)
        _ = try self.weights.tensor("decoder.conv1.weight")
        _ = try self.weights.tensor("conv2.weight")
    }

    private func history(_ x: MLXArray, previous: MLXArray?) -> MLXArray {
        if x.dim(1) >= 2 { return x[0..., (x.dim(1) - 2)..., 0..., 0..., 0...] }
        let old = previous.map { $0[0..., ($0.dim(1) - 1)..., 0..., 0..., 0...] }
            ?? MLXArray.zeros([x.dim(0), 1, x.dim(2), x.dim(3), x.dim(4)], dtype: x.dtype)
        return concatenated([old, x], axis: 1)
    }
    private func causal(_ name: String, _ x: MLXArray, cache: inout [String: MLXArray]) throws -> MLXArray {
        let old = cache[name]
        let result = try weights.conv(name, x, causal: 2, spatial: 1, cache: old)
        cache[name] = history(x, previous: old)
        return result
    }
    private func residual(_ name: String, _ x: MLXArray, cache: inout [String: MLXArray],
                          interrupt: FluxInterrupt?) throws -> MLXArray {
        try interrupt?.check()
        let shortcut = try weights.optional(name + ".shortcut.weight") == nil ? x : weights.conv(name + ".shortcut", x)
        var y = try causal(name + ".residual.2", silu(weights.rms(name + ".residual.0", x, eps: 1e-12)), cache: &cache)
        y = try causal(name + ".residual.6", silu(weights.rms(name + ".residual.3", y, eps: 1e-12)), cache: &cache)
        return shortcut + y
    }
    private func attention(_ name: String, _ x: MLXArray) throws -> MLXArray {
        let b = x.dim(0), t = x.dim(1), h = x.dim(2), w = x.dim(3), c = x.dim(4)
        let normalized = try weights.rms(name + ".norm", x, eps: 1e-12)
        let qkv = try weights.pointwise(name + ".to_qkv", normalized).reshaped([b * t, h * w, 3, c])
        let q = qkv[0..., 0..., 0, 0...].reshaped([b * t, 1, h * w, c])
        let k = qkv[0..., 0..., 1, 0...].reshaped([b * t, 1, h * w, c])
        let v = qkv[0..., 0..., 2, 0...].reshaped([b * t, 1, h * w, c])
        let result = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v,
            scale: 1 / sqrt(Float(c)), mask: nil).reshaped([b, t, h, w, c])
        return x + (try weights.pointwise(name + ".proj", result))
    }
    private func upsample(_ name: String, _ input: MLXArray, temporal: Bool,
                          cache: inout [String: MLXArray]) throws -> MLXArray {
        var x = input
        let b = x.dim(0), t = x.dim(1), h = x.dim(2), w = x.dim(3), c = x.dim(4)
        if temporal {
            if let previous = cache[name] {
                x = try weights.conv(name + ".time_conv", x, causal: 2, cache: previous)
                cache[name] = history(input, previous: previous)
                x = x.reshaped([b, t, h, w, 2, c]).transposed(0, 1, 4, 2, 3, 5).reshaped([b, t * 2, h, w, c])
            } else {
                cache[name] = MLXArray.zeros([b, 2, h, w, c], dtype: x.dtype)
            }
        }
        let count = x.dim(1)
        x = x.reshaped([b * count, h, w, c])
        x = Upsample(scaleFactor: .init(2.0), mode: .nearest)(x)
        x = try weights.conv2(name + ".resample.1", x, padding: 1)
        return x.reshaped([b, count, x.dim(1), x.dim(2), x.dim(3)])
    }

    func decode(_ latent: MLXArray, interrupt: FluxInterrupt?,
                frame: (MLXArray) throws -> Void) throws {
        var cache: [String: MLXArray] = [:]
        let z = try weights.conv("conv2", (latent * std + mean).expandedDimensions(axis: 0))
        for index in 0..<z.dim(1) {
            try interrupt?.check()
            var x = try causal("decoder.conv1", z[0..., index..<(index + 1), 0..., 0..., 0...], cache: &cache)
            x = try residual("decoder.middle.0", x, cache: &cache, interrupt: interrupt)
            x = try attention("decoder.middle.1", x)
            x = try residual("decoder.middle.2", x, cache: &cache, interrupt: interrupt)
            for stage in 0..<4 {
                for block in 0..<3 {
                    x = try residual("decoder.upsamples.\(stage * 4 + block)", x, cache: &cache, interrupt: interrupt)
                }
                if stage < 3 { x = try upsample("decoder.upsamples.\(stage * 4 + 3)", x, temporal: stage < 2, cache: &cache) }
            }
            x = try causal("decoder.head.2", silu(weights.rms("decoder.head.0", x, eps: 1e-12)), cache: &cache)
            x = clip(x[0], min: -1, max: 1)
            eval(x, Array(cache.values))
            try interrupt?.check()
            try frame(x)
        }
    }

    private func downsample(_ name: String, _ input: MLXArray, temporal: Bool,
                            cache: inout [String: MLXArray]) throws -> MLXArray {
        let b = input.dim(0), t = input.dim(1)
        var x = input.reshaped([b * t, input.dim(2), input.dim(3), input.dim(4)])
        x = padded(x, widths: [.init(0), .init((0, 1)), .init((0, 1)), .init(0)])
        x = try weights.conv2(name + ".resample.1", x, stride: 2)
        x = x.reshaped([b, t, x.dim(1), x.dim(2), x.dim(3)])
        if temporal {
            if let previous = cache[name] {
                let combined = concatenated([previous[0..., (previous.dim(1) - 1)..., 0..., 0..., 0...], x], axis: 1)
                cache[name] = x[0..., (x.dim(1) - 1)..., 0..., 0..., 0...]
                x = try weights.conv(name + ".time_conv", combined, stride: [2, 1, 1])
            } else { cache[name] = x }
        }
        return x
    }

    func encode(_ video: MLXArray, interrupt: FluxInterrupt?) throws -> MLXArray {
        var cache: [String: MLXArray] = [:]
        var outputs: [MLXArray] = []
        var index = 0
        while index < video.dim(0) {
            try interrupt?.check()
            let end = min(video.dim(0), index + (index == 0 ? 1 : 4))
            var x = video[index..<end].expandedDimensions(axis: 0)
            x = try causal("encoder.conv1", x, cache: &cache)
            for stage in 0..<4 {
                for block in 0..<2 {
                    x = try residual("encoder.downsamples.\(stage * 3 + block)", x, cache: &cache, interrupt: interrupt)
                }
                if stage < 3 { x = try downsample("encoder.downsamples.\(stage * 3 + 2)", x, temporal: stage > 0, cache: &cache) }
            }
            x = try residual("encoder.middle.0", x, cache: &cache, interrupt: interrupt)
            x = try attention("encoder.middle.1", x)
            x = try residual("encoder.middle.2", x, cache: &cache, interrupt: interrupt)
            x = try causal("encoder.head.2", silu(weights.rms("encoder.head.0", x, eps: 1e-12)), cache: &cache)
            eval(x, Array(cache.values))
            outputs.append(x)
            index = end
        }
        let encoded = try weights.conv("conv1", concatenated(outputs, axis: 1))
        return (encoded[0, 0..., 0..., 0..., 0..<16] - mean) / std
    }
}
