// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import Testing

@testable import MLXVLM

/// Property tests for the QSA block-selection mask; each pins an invariant
/// of the reference recurrence rather than a specific score value.
@Suite("qwen4_exp QSA block selection", .serialized)
struct Qwen4ExpQSATests {

    private func makeMask(
        pastLen: Int, seqLen: Int, keyLen: Int,
        compressRatio: Int = 4, blockTopK: Int = 2, heads: Int = 2, dim: Int = 8
    ) -> MLXArray? {
        MLXRandom.seed(3)
        let numBlocks = keyLen / compressRatio
        let q = MLXRandom.normal([1, heads, seqLen, dim])
        let pooled = MLXRandom.normal([1, 1, numBlocks, dim])
        return Qwen4ExpQSA.selectedTokenMask(
            query: q, pooledKeys: pooled, pastLen: pastLen,
            compressRatio: compressRatio, blockTopK: blockTopK, keyLen: keyLen)
    }

    private func makeReferenceMask(
        query: MLXArray, pooled: MLXArray,
        pastLen: Int, compressRatio: Int, blockTopK: Int, keyLen: Int
    ) -> MLXArray? {
        let seqLen = query.dim(2)
        let maxCompleteBlocks = keyLen / compressRatio
        guard maxCompleteBlocks > blockTopK else { return nil }

        var scores = matmul(query, pooled.transposed(0, 1, 3, 2))
        scores = maximum(scores.asType(.float32), MLXArray(Float(0))).sum(axis: 1)
        scores = scores / sqrt(Float(query.dim(3)))
        let queryEnds = MLXArray((0 ..< seqLen).map { Int32(pastLen + $0 + 1) })
        let completeCounts = floorDivide(queryEnds, MLXArray(Int32(compressRatio)))
        let blockIdx = MLXArray((0 ..< maxCompleteBlocks).map(Int32.init))
        let validBlocks = MLX.less(
            blockIdx.reshaped(1, 1, maxCompleteBlocks),
            completeCounts.reshaped(1, seqLen, 1))
        scores = MLX.where(validBlocks, scores, MLXArray(-Float.infinity))
        let selectedBlocks = argPartition(scores, kth: -blockTopK, axis: -1)[
            .ellipsis, (-blockTopK)...]
        let tokenIdx = MLXArray((0 ..< keyLen).map(Int32.init))
        let tokenBlocks = floorDivide(tokenIdx, MLXArray(Int32(compressRatio)))
        let selectedTokens = MLX.any(
            MLX.equal(
                tokenBlocks.reshaped(1, 1, 1, keyLen),
                expandedDimensions(selectedBlocks, axis: -1)),
            axis: 2)
        let tailStarts = completeCounts * Int32(compressRatio)
        let tokenRow = tokenIdx.reshaped(1, 1, keyLen)
        let beforeQueryEnd = MLX.less(tokenRow, queryEnds.reshaped(1, seqLen, 1))
        let tail = MLX.logicalAnd(
            MLX.greaterEqual(tokenRow, tailStarts.reshaped(1, seqLen, 1)),
            beforeQueryEnd)
        let useSparse = MLX.greater(completeCounts, MLXArray(Int32(blockTopK)))
        let mask = MLX.where(
            useSparse.reshaped(1, seqLen, 1),
            MLX.logicalOr(selectedTokens, tail),
            beforeQueryEnd)
        return expandedDimensions(mask, axis: 1)
    }

    @Test("below the sparse threshold returns nil (dense fallback)")
    func denseFallback() throws {
        try MLXMetalTestLock.withLock {
            // 8 keys / ratio 4 = 2 complete blocks, not > topk 2.
            #expect(makeMask(pastLen: 4, seqLen: 4, keyLen: 8) == nil)
        }
    }

    @Test("mask is causal: nothing beyond each query position attends")
    func causality() throws {
        try MLXMetalTestLock.withLock {
            let mask = try #require(makeMask(pastLen: 28, seqLen: 4, keyLen: 32))
            for t in 0 ..< 4 {
                let queryEnd = 28 + t + 1
                for j in queryEnd ..< 32 {
                    #expect(
                        mask[0, 0, t, j].item(Bool.self) == false,
                        "t=\(t) attends future token \(j)")
                }
            }
        }
    }

    @Test("the incomplete tail up to the query always attends when sparse")
    func tailInclusion() throws {
        try MLXMetalTestLock.withLock {
            // pastLen 29 → query 0 ends at 30: blocks 0..6 complete (28 tokens),
            // tail 28..29 must be included.
            let mask = try #require(makeMask(pastLen: 29, seqLen: 2, keyLen: 32))
            for t in 0 ..< 2 {
                let queryEnd = 29 + t + 1
                let tailStart = (queryEnd / 4) * 4
                for j in tailStart ..< queryEnd {
                    #expect(
                        mask[0, 0, t, j].item(Bool.self) == true,
                        "t=\(t) tail token \(j) excluded")
                }
            }
        }
    }

    @Test("sparse rows keep exactly blockTopK complete blocks")
    func budget() throws {
        try MLXMetalTestLock.withLock {
            let compressRatio = 4, blockTopK = 2
            let mask = try #require(
                makeMask(
                    pastLen: 31, seqLen: 1, keyLen: 32,
                    compressRatio: compressRatio, blockTopK: blockTopK))
            let queryEnd = 32
            let completeCount = queryEnd / compressRatio  // 8 complete blocks
            var selectedComplete = 0
            for b in 0 ..< completeCount {
                let start = b * compressRatio
                // skip the tail block (== completeCount*ratio.. none here since 32%4==0)
                var allOn = true
                for j in start ..< min(start + compressRatio, queryEnd) {
                    if mask[0, 0, 0, j].item(Bool.self) == false { allOn = false; break }
                }
                if allOn { selectedComplete += 1 }
            }
            // 32 % 4 == 0 → no tail; exactly blockTopK complete blocks attend.
            #expect(selectedComplete == blockTopK, "selected \(selectedComplete)")
        }
    }

    @Test("block-axis scatter is exactly equivalent to the reference membership mask")
    func blockAxisScatterParity() throws {
        try MLXMetalTestLock.withLock {
            MLXRandom.seed(17)
            let seqLen = 7, keyLen = 35, compressRatio = 4, blockTopK = 2
            let query = MLXRandom.normal([1, 3, seqLen, 8])
            let pooled = MLXRandom.normal([1, 1, keyLen / compressRatio, 8])
            let actual = try #require(Qwen4ExpQSA.selectedTokenMask(
                query: query, pooledKeys: pooled, pastLen: keyLen - seqLen,
                compressRatio: compressRatio, blockTopK: blockTopK, keyLen: keyLen))
            let reference = try #require(makeReferenceMask(
                query: query, pooled: pooled, pastLen: keyLen - seqLen,
                compressRatio: compressRatio, blockTopK: blockTopK, keyLen: keyLen))

            #expect(actual.shape == reference.shape)
            #expect(actual.asArray(Bool.self) == reference.asArray(Bool.self))
        }
    }

    @Test("4K sparse prefill materializes without a topK-expanded token tensor")
    func longPrefillMembershipIsBounded() throws {
        try MLXMetalTestLock.withLock {
            let seqLen = 4_096, compressRatio = 4, blockTopK = 512
            MLXRandom.seed(23)
            let query = MLXRandom.normal([1, 4, seqLen, 8])
            let pooled = MLXRandom.normal([1, 1, seqLen / compressRatio, 8])
            let mask = try #require(Qwen4ExpQSA.selectedTokenMask(
                query: query, pooledKeys: pooled, pastLen: 0,
                compressRatio: compressRatio, blockTopK: blockTopK, keyLen: seqLen))

            #expect(mask.shape == [1, 1, seqLen, seqLen])
            MLX.eval(mask)
            #expect(mask[0, 0, seqLen - 1].sum().item(Int.self) >= blockTopK * compressRatio)
        }
    }
}
