// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import Testing

@testable import MLXVLM

/// Property tests for the QSA block-selection mask; each pins an invariant
/// of the reference recurrence rather than a specific score value.
@Suite("qwen4_exp QSA block selection", .serialized)
struct Qwen4ExpQSATests {

    @Test("complete QSA masks match the frozen original across prefill, budget crossings and ties")
    func completeMaskParity() throws {
        try MLXMetalTestLock.withLock {
            for (past, length) in [(0, 128), (2040, 16), (2047, 4),
                                   (2048, 1), (2048, 128), (8192, 4)] {
                for tied in [false, true] {
                    MLXRandom.seed(19)
                    let keyLen = past + length
                    let q = tied ? MLXArray.zeros([1, 4, length, 8])
                        : MLXRandom.normal([1, 4, length, 8])
                    let pooled = MLXRandom.normal([1, 1, keyLen / 4, 8])
                    let original = Qwen4ExpQSAOriginalReference.selectedTokenMask(
                        query: q, pooledKeys: pooled, pastLen: past,
                        compressRatio: 4, blockTopK: 512, keyLen: keyLen)
                    let candidate = Qwen4ExpQSA.selectedTokenMask(
                        query: q, pooledKeys: pooled, pastLen: past,
                        compressRatio: 4, blockTopK: 512, keyLen: keyLen)
                    #expect((original == nil) == (candidate == nil))
                    if let original, let candidate {
                        #expect(candidate.asArray(Bool.self) == original.asArray(Bool.self),
                            "past=\(past) length=\(length) tied=\(tied)")
                    }
                }
            }
        }
    }

    private func originalMembership(_ selected: MLXArray, keyLen: Int, ratio: Int) -> MLXArray {
        let tokenBlocks = floorDivide(
            MLXArray((0..<keyLen).map(Int32.init)), MLXArray(Int32(ratio)))
        return MLX.any(MLX.equal(
            tokenBlocks.reshaped(1, 1, 1, keyLen),
            expandedDimensions(selected, axis: -1)), axis: 2)
    }

    @Test("block expansion exactly matches original token membership at partial tails and verify widths")
    func blockMembershipParity() throws {
        try MLXMetalTestLock.withLock {
            for ratio in [1, 4, 8] {
                for keyLen in [1, 7, 8, 9, 31, 32, 33, 257] {
                    for queries in [1, 2, 3, 4, 6] {
                        let blocks = (keyLen + ratio - 1) / ratio
                        let selected = MLXArray((0..<(2 * queries * 5)).map {
                            Int32(($0 * 7) % (blocks + 2) - 1)
                        }).reshaped(2, queries, 5)
                        let expected = originalMembership(selected, keyLen: keyLen, ratio: ratio)
                        let actual = Qwen4ExpQSA.tokenMembership(
                            selectedBlocks: selected, keyLen: keyLen, compressRatio: ratio)
                        #expect(actual.shape == expected.shape)
                        #expect(actual.asArray(Bool.self) == expected.asArray(Bool.self))
                    }
                }
            }
        }
    }

    @Test("QSA membership production-budget diagnostic with synchronized timings")
    func membershipTiming() throws {
        guard ProcessInfo.processInfo.environment["VMLX_QSA_MEMBERSHIP_BENCH"] == "1" else { return }
        try MLXMetalTestLock.withLock {
            for keyLen in [8193, 32769] {
                for queries in [1, 4] {
                    let selected = MLXArray((0..<(queries * 512)).map {
                        Int32(($0 * 7) % (keyLen / 4))
                    }).reshaped(1, queries, 512)
                    let reference = originalMembership(selected, keyLen: keyLen, ratio: 4).asArray(Bool.self)
                    for round in 0..<3 {
                        for candidate in [false, true] {
                            let start = DispatchTime.now().uptimeNanoseconds
                            for _ in 0..<8 {
                                let result = candidate
                                    ? Qwen4ExpQSA.tokenMembership(selectedBlocks: selected, keyLen: keyLen, compressRatio: 4)
                                    : originalMembership(selected, keyLen: keyLen, ratio: 4)
                                MLX.eval(result)
                            }
                            let elapsed = DispatchTime.now().uptimeNanoseconds - start
                            let actual = Qwen4ExpQSA.tokenMembership(
                                selectedBlocks: selected, keyLen: keyLen, compressRatio: 4)
                            #expect(actual.asArray(Bool.self) == reference)
                            print("QSA_MEMBERSHIP_BENCH round=\(round) candidate=\(candidate) keys=\(keyLen) queries=\(queries) iterations=8 ns=\(elapsed)")
                        }
                    }
                }
            }
        }
    }

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
}
