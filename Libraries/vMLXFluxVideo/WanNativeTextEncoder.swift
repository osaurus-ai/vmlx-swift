// Architecture reference: ml-explore/mlx-examples/video/wan2.1/t5.py.
import Foundation
@preconcurrency import MLX
import MLXNN
import VMLXTokenizers
import vMLXFluxKit

final class WanNativeTextEncoder {
    let weights: WanNativeWeights
    let tokenizer: any VMLXTokenizers.Tokenizer
    init(weights: [String: MLXArray], modelURL: URL) async throws {
        self.weights = WanNativeWeights(weights)
        tokenizer = try await AutoTokenizer.from(modelFolder: modelURL.appendingPathComponent("tokenizer"), strict: false)
        _ = try self.weights.tensor("token_embedding.weight")
        _ = try self.weights.tensor("norm.weight")
    }

    func encode(_ prompt: String, interrupt: FluxInterrupt?) throws -> MLXArray {
        var ids = Array(tokenizer.encode(text: prompt).prefix(512)).map(Int32.init)
        guard !ids.isEmpty else { throw FluxError.invalidRequest("Wan tokenizer returned no tokens") }
        let count = ids.count
        ids += Array(repeating: 0, count: 512 - count)
        let embedding = try weights.tensor("token_embedding.weight")
        var x = embedding.take(MLXArray(ids), axis: 0).expandedDimensions(axis: 0)
        guard x.dim(-1) == 4096 else { throw FluxError.invalidRequest("Wan requires the bundled UMT5-XXL encoder") }
        var buckets: [Int32] = []
        for q in 0..<512 { for k in 0..<512 {
            let distance = abs(k - q)
            let magnitude = distance < 8 ? distance
                : min(15, 8 + Int(log(Double(distance) / 8) / log(16) * 8))
            buckets.append(Int32((k > q ? 16 : 0) + magnitude))
        } }
        let indices = MLXArray(buckets).reshaped([512, 512])
        let valid = MLXArray((0..<512).map { $0 < count }).reshaped([1, 1, 1, 512])
        for layer in 0..<24 {
            try interrupt?.check()
            let p = "blocks.\(layer)"
            let y = try weights.rms(p + ".norm1", x)
            let q = try weights.linear(p + ".attn.q", y).reshaped([1, 512, 64, 64]).transposed(0, 2, 1, 3)
            let k = try weights.linear(p + ".attn.k", y).reshaped([1, 512, 64, 64]).transposed(0, 2, 1, 3)
            let v = try weights.linear(p + ".attn.v", y).reshaped([1, 512, 64, 64]).transposed(0, 2, 1, 3)
            let relative = try weights.tensor(p + ".pos_embedding.embedding.weight").take(indices, axis: 0)
                .transposed(2, 0, 1).expandedDimensions(axis: 0)
            let bias = which(valid, relative, MLXArray(Float(-1e9))).asType(x.dtype)
            let attention = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: 1, mask: bias)
                .transposed(0, 2, 1, 3).reshaped([1, 512, 4096])
            x = x + (try weights.linear(p + ".attn.o", attention))
            let normalized = try weights.rms(p + ".norm2", x)
            // Original Wan checkpoints store the gate Sequential at gate.0.
            let gate = weights.optional(p + ".ffn.gate.weight") == nil ? p + ".ffn.gate.0" : p + ".ffn.gate"
            x = x + (try weights.linear(p + ".ffn.fc2",
                weights.linear(p + ".ffn.fc1", normalized) * geluApproximate(weights.linear(gate, normalized))))
            eval(x)
        }
        x = try weights.rms("norm", x)
        if count < 512 {
            x = concatenated([x[0..., 0..<count, 0...], MLXArray.zeros([1, 512 - count, 4096], dtype: x.dtype)], axis: 1)
        }
        eval(x)
        return x
    }
}
