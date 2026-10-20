// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// The fused router tail (softmax + top-k + normalize) against the op chain it replaces in
// `SparseMoeBlock.routerSelection`: same indices in the same slot order and the same score bits,
// including tied probabilities (BF16 softmax outputs tie often) and the normalize-off variant.

import Foundation
import MLX
import Testing

@testable import MLXVLM

@Suite("Qwen3.5 fused router tail", .serialized)
struct Qwen35RouterTailTests {
    func reference(_ logits: MLXArray, topK: Int, normalize: Bool) -> (MLXArray, MLXArray) {
        let gates = MLX.softmax(logits, axis: -1, precise: true)
        let kth = gates.dim(-1) - topK
        let inds = MLX.argPartition(gates, kth: kth, axis: -1)[.ellipsis, kth...]
        var selected = MLX.takeAlong(gates, inds, axis: -1)
        if normalize {
            selected = selected / selected.sum(axis: -1, keepDims: true)
        }
        return (inds, selected)
    }

    @Test(
        "indices, slot order and scores match the op chain bit for bit",
        arguments: [(256, 8, true), (256, 8, false), (128, 4, true), (512, 10, true)])
    func matchesOpChain(experts: Int, topK: Int, normalize: Bool) throws {
        MLXRandom.seed(UInt64(experts * 31 + topK))
        // Narrow logits produce many tied BF16 probabilities; a few rows are fully tied.
        var logits = (MLXRandom.normal([64, experts]) * 1.5).asType(.bfloat16)
        let tiedRows = MLXArray.zeros([4, experts], dtype: .bfloat16)
        let coarse = (MLXRandom.randInt(low: 0, high: 4, [8, experts]).asType(.float32) * 0.5)
            .asType(.bfloat16)
        logits = concatenated([logits, tiedRows, coarse], axis: 0).reshaped(1, 76, experts)
        let (wantIdx, wantScores) = reference(logits, topK: topK, normalize: normalize)
        let fused = try #require(Qwen35RouterTail.call(logits, topK: topK, normalize: normalize))
        MLX.eval(wantIdx, wantScores, fused.0, fused.1)
        #expect(fused.0.shape == wantIdx.shape)
        #expect(fused.0.asType(.int32).asArray(Int32.self) == wantIdx.asType(.int32).asArray(Int32.self),
                "indices / slot order")
        #expect(fused.1.asData(access: .copy).data == wantScores.asData(access: .copy).data, "scores")
    }
}
