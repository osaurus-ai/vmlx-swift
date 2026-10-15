// Architecture reference: ml-explore/mlx-examples/video/wan2.1.
import Foundation
@preconcurrency import MLX
import MLXNN
import vMLXFluxKit

final class WanNativeTransformer {
    let weights: WanNativeWeights
    let dim: Int, heads: Int, layers: Int, frequencyDim: Int
    let imageConditioned: Bool
    init(weights: [String: MLXArray], config: [String: Any]) throws {
        self.weights = WanNativeWeights(weights)
        dim = config["dim"] as? Int ?? 1536
        heads = config["num_heads"] as? Int ?? 12
        layers = config["num_layers"] as? Int ?? 30
        frequencyDim = config["freq_dim"] as? Int ?? 256
        imageConditioned = (config["model_type"] as? String) == "i2v"
        guard dim > 0, heads > 0, dim % heads == 0, layers > 0,
              frequencyDim > 0, frequencyDim % 2 == 0, dim / heads % 2 == 0 else {
            throw FluxError.invalidRequest("invalid Wan transformer configuration")
        }
        _ = try self.weights.tensor("patch_embedding.weight")
        _ = try self.weights.tensor("head.head.weight")
        for i in 0..<layers { _ = try self.weights.tensor("blocks.\(i).modulation") }
    }

    private func rope(_ x: MLXArray, t: Int, h: Int, w: Int) -> MLXArray {
        let d = dim / heads, dh = 2 * (d / 6), dt = d - 2 * dh
        let parts = [(0, dt, 0), (dt, dh, 1), (dt + dh, dh, 2)]
        return concatenated(parts.map { start, count, axis in
            var positions: [Float] = []
            for frame in 0..<t { for row in 0..<h { for column in 0..<w {
                positions.append(Float(axis == 0 ? frame : (axis == 1 ? row : column)))
            } } }
            let frequencies = exp(-log(Float(10000)) * MLXArray(0..<(count / 2)).asType(.float32) * (2 / Float(count)))
            let angles = MLXArray(positions).reshaped([1, t * h * w, 1, 1]) * frequencies
            let v = x[.ellipsis, start..<(start + count)].reshaped([1, t * h * w, heads, count / 2, 2])
            let even = v[.ellipsis, 0], odd = v[.ellipsis, 1]
            return stacked([even * cos(angles) - odd * sin(angles), even * sin(angles) + odd * cos(angles)], axis: -1)
                .reshaped([1, t * h * w, heads, count]).asType(x.dtype)
        }, axis: -1)
    }

    private func attention(_ name: String, x: MLXArray, context: MLXArray?, grid: (Int, Int, Int), imageContext: MLXArray? = nil) throws -> MLXArray {
        let source = context ?? x
        let q = try weights.rms(name + ".norm_q", weights.linear(name + ".q", x))
        let k = try weights.rms(name + ".norm_k", weights.linear(name + ".k", source))
        let v = try weights.linear(name + ".v", source)
        var qh = q.reshaped([1, q.dim(1), heads, dim / heads])
        var kh = k.reshaped([1, k.dim(1), heads, dim / heads])
        if context == nil {
            qh = rope(qh, t: grid.0, h: grid.1, w: grid.2)
            kh = rope(kh, t: grid.0, h: grid.1, w: grid.2)
        }
        let vh = v.reshaped([1, v.dim(1), heads, dim / heads]).transposed(0, 2, 1, 3)
        var attended = MLXFast.scaledDotProductAttention(queries: qh.transposed(0, 2, 1, 3),
            keys: kh.transposed(0, 2, 1, 3), values: vh, scale: 1 / sqrt(Float(dim / heads)), mask: nil)
            .transposed(0, 2, 1, 3).reshaped([1, q.dim(1), dim])
        if let imageContext {
            let ki = try weights.rms(name + ".norm_k_img", weights.linear(name + ".k_img", imageContext))
                .reshaped([1, imageContext.dim(1), heads, dim / heads]).transposed(0, 2, 1, 3)
            let vi = try weights.linear(name + ".v_img", imageContext)
                .reshaped([1, imageContext.dim(1), heads, dim / heads]).transposed(0, 2, 1, 3)
            let imageAttention = MLXFast.scaledDotProductAttention(queries: qh.transposed(0, 2, 1, 3),
                keys: ki, values: vi, scale: 1 / sqrt(Float(dim / heads)), mask: nil)
                .transposed(0, 2, 1, 3).reshaped([1, q.dim(1), dim])
            attended = attended + imageAttention
        }
        return try weights.linear(name + ".o", attended)
    }

    func predict(_ latent: MLXArray, time: Float, context: MLXArray, imageFeatures: MLXArray? = nil, imageLatent: MLXArray? = nil, interrupt: FluxInterrupt?) throws -> MLXArray {
        guard imageConditioned == (imageFeatures != nil && imageLatent != nil) else {
            throw FluxError.invalidRequest("Wan image conditioning does not match this bundle")
        }
        let input = imageLatent.map { concatenated([latent, $0], axis: -1) } ?? latent
        var x = try weights.conv("patch_embedding", input.expandedDimensions(axis: 0), stride: [1, 2, 2])
        let imageContext = try imageFeatures.map { features in
            try weights.norm("img_emb.proj.4", weights.linear("img_emb.proj.3",
                gelu(weights.linear("img_emb.proj.1", weights.norm("img_emb.proj.0", features, eps: 1e-5)))), eps: 1e-5)
        }
        let t = x.dim(1), h = x.dim(2), w = x.dim(3)
        x = x.reshaped([1, t * h * w, dim])
        let text = try weights.linear("text_embedding.2", geluApproximate(weights.linear("text_embedding.0", context)))
        let frequencies = exp(-log(Float(10000)) * MLXArray(0..<(frequencyDim / 2)).asType(.float32) / Float(frequencyDim / 2))
        let phases = frequencies * time
        let sinusoid = concatenated([cos(phases), sin(phases)], axis: 0).reshaped([1, frequencyDim])
        let timeEmbedding = try weights.linear("time_embedding.2", silu(weights.linear("time_embedding.0", sinusoid)))
        let modulation = try weights.linear("time_projection.1", silu(timeEmbedding)).reshaped([1, 6, dim])
        for index in 0..<layers {
            try interrupt?.check()
            let prefix = "blocks.\(index)"
            let e = try weights.tensor(prefix + ".modulation") + modulation
            let normalized = try weights.norm(nil, x)
            let y = normalized * (1 + e[0..., 1, 0...]) + e[0..., 0, 0...]
            x = x + (try attention(prefix + ".self_attn", x: y, context: nil, grid: (t, h, w))) * e[0..., 2, 0...]
            let crossInput = try weights.optional(prefix + ".norm3.weight") == nil ? x : weights.norm(prefix + ".norm3", x)
            x = x + (try attention(prefix + ".cross_attn", x: crossInput, context: text, grid: (t, h, w), imageContext: imageContext))
            let z = (try weights.norm(nil, x)) * (1 + e[0..., 4, 0...]) + e[0..., 3, 0...]
            x = x + (try weights.linear(prefix + ".ffn.2", geluApproximate(weights.linear(prefix + ".ffn.0", z)))) * e[0..., 5, 0...]
            eval(x)
        }
        let e = try weights.tensor("head.modulation") + timeEmbedding.expandedDimensions(axis: 1)
        x = try weights.linear("head.head", weights.norm(nil, x) * (1 + e[0..., 1, 0...]) + e[0..., 0, 0...])
        return x.reshaped([t, h, w, 1, 2, 2, 16]).transposed(0, 3, 1, 4, 2, 5, 6)
            .reshaped([t, h * 2, w * 2, 16])
    }
}
