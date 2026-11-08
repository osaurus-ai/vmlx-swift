// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXNN
import MLXRandom
import Testing

@testable import MLXLLM

/// `DiffusionGemmaModel.softEmbeddings`: self-conditioning's `probs @ W` over the tied embedding
/// table. A quantized table is used packed, through `quantizedMatmul(transpose: false)`, and must
/// give the product of the dequantized table.
@Suite("DiffusionGemma soft embeddings")
struct DiffusionGemmaSoftEmbeddingTests {

    static let vocab = 512
    static let hidden = 256

    /// Rows of a softmax over the vocabulary, shaped as the denoising step builds them.
    static func probs() -> MLXArray {
        let logits = MLXRandom.normal([1, 8, vocab], key: MLXRandom.key(7)) * 4
        return softmax(logits, axis: -1, precise: true)
    }

    /// A table in float16, so that affine quantization gives float16 scales, as bundles ship them.
    static func table() -> MLXArray {
        (MLXRandom.normal([vocab, hidden], key: MLXRandom.key(11)) * 0.05).asType(.float16)
    }

    /// The product of the dequantized table, in float32.
    static func reference(_ probs: MLXArray, _ q: QuantizedEmbedding) -> MLXArray {
        let w = dequantized(
            q.weight, scales: q.scales, biases: q.biases,
            groupSize: q.groupSize, bits: q.bits, mode: q.mode, dtype: .float32)
        return matmul(probs.asType(.float32), w)
    }

    static func check(_ q: QuantizedEmbedding) {
        let p = probs()
        let out = DiffusionGemmaModel.softEmbeddings(p, table: q, dtype: .bfloat16)
        #expect(out.dtype == .bfloat16)
        #expect(out.shape == [1, 8, hidden])
        let ref = reference(p, q)
        let err = abs(out.asType(.float32) - ref).max().item(Float.self)
        let scale = abs(ref).max().item(Float.self)
        #expect(scale > 0)
        #expect(err <= scale * 1e-2, "max error \(err) against max magnitude \(scale)")
    }

    @Test(
        "affine tables give the dequantized table's product",
        arguments: [(64, 4), (64, 6), (64, 8), (32, 4)])
    func affine(groupSize: Int, bits: Int) throws {
        let q = QuantizedEmbedding(
            weight: Self.table(), groupSize: groupSize, bits: bits, mode: .affine)
        try #require(q.scales.dtype == .float16)
        Self.check(q)
    }

    @Test("an mxfp4 table gives the dequantized table's product")
    func mxfp4() {
        Self.check(QuantizedEmbedding(weight: Self.table(), groupSize: 32, bits: 4, mode: .mxfp4))
    }

    @Test("a dense table is multiplied as it is")
    func dense() {
        let table = Embedding(weight: Self.table())
        let p = Self.probs()
        let out = DiffusionGemmaModel.softEmbeddings(p, table: table, dtype: .bfloat16)
        #expect(out.dtype == .bfloat16)
        let ref = matmul(p.asType(.bfloat16), table.weight.asType(.bfloat16))
        #expect(allClose(out, ref).item(Bool.self))
    }
}
