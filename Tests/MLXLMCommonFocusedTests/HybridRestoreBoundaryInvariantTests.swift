// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Hybrid restore boundary invariant.
//
// A coordinator cache hit restores TWO independently derived lengths:
//
//   * attention layers — `restoreLayerData(from:into:)` sets each KV layer's
//     offset from the restored tensor's sequence dim and returns that as
//     `restoredTokens`;
//   * recurrent layers — `restoreSSMStates(_:into:boundary:)` sets every
//     MambaCache / ArraysCache offset to the coordinator's `matchedTokens`.
//
// The apply sites (Evaluate / BatchEngine / NativeMTPTokenIterator, plus the
// disk-array paths right after each) only checked `restoredTokens > 0`. When
// the two lengths disagree, attention runs over M tokens while KDA / GDN
// state summarises N != M — the hybrid's two halves desynchronise and the
// logits degenerate (token id 0 `!` on Raptor / Ling 3, verbatim loops on
// Ornith / qwen3_5).
//
// The invariant is: after a restore, `restoredTokens == matchedTokens` AND
// every layer's `offset == matchedTokens`. `validateRestoredCacheBoundary`
// is the fail-closed check; a violation makes the caller rebuild the cache
// and full-prefill.
//
// 37 is used throughout on purpose: it is not a multiple of 16 (the block
// size used here) nor of 64 / 256 (KV step sizes), so a padded/aligned
// buffer cannot masquerade as the logical offset.

import Foundation
import MLX
@testable import MLXLMCommon
import Testing

@Suite("Hybrid restore boundary invariant", .serialized)
struct HybridRestoreBoundaryInvariantTests {

    // MARK: - Fixtures

    private static let kvHeads = 2
    private static let headDim = 8

    /// A KVCacheSimple whose logical offset is exactly `offset`.
    ///
    /// Goes through the `state` setter (the same seam `restoreLayerData`
    /// uses), which derives `offset` from `keys.dim(2)`.
    private func makeKV(offset: Int, fill: Float = 1) -> KVCacheSimple {
        let cache = KVCacheSimple()
        guard offset > 0 else { return cache }
        let keys = MLXArray.ones([1, Self.kvHeads, offset, Self.headDim], dtype: .bfloat16) * fill
        let values = MLXArray.ones([1, Self.kvHeads, offset, Self.headDim], dtype: .bfloat16) * (fill / 2)
        MLX.eval(keys, values)
        cache.state = [keys, values]
        return cache
    }

    /// A MambaCache with populated conv/ssm slots and logical offset `offset`.
    private func makeMamba(offset: Int) -> MambaCache {
        let mamba = MambaCache()
        let conv = MLXArray.ones([1, 4, 8], dtype: .float32)
        let ssm = MLXArray.ones([1, 2, 8, 8], dtype: .float32) * Float(2)
        MLX.eval(conv, ssm)
        mamba.state = [conv, ssm]
        mamba.offset = offset
        return mamba
    }

    /// `[KVCacheSimple, MambaCache, KVCacheSimple]` — the smallest hybrid
    /// shape with a recurrent layer sandwiched between two attention layers,
    /// so a per-layer walk must visit both kinds on either side of it.
    private func makeHybridCache(kv0: Int, mamba: Int, kv2: Int) -> [any KVCache] {
        [makeKV(offset: kv0), makeMamba(offset: mamba), makeKV(offset: kv2)]
    }

    /// Paged blocks carrying `tokenCount` tokens in total for the 2 KV-bearing
    /// layers of `makeHybridCache`, split into `blockSize`-token blocks (last
    /// block partial). Mirrors what `CacheCoordinator.fetch` hands back.
    private func makeBlocks(tokenCount: Int, blockSize: Int) -> [CacheBlock] {
        var blocks: [CacheBlock] = []
        var start = 0
        var blockId = 0
        while start < tokenCount {
            let n = min(blockSize, tokenCount - start)
            let block = CacheBlock(blockId: blockId, blockSize: blockSize)
            block.tokenIds = Array(start ..< (start + n))
            var layers: [(keys: MLXArray, values: MLXArray)?] = []
            for layer in 0 ..< 2 {
                let keys = MLXArray.ones([1, Self.kvHeads, n, Self.headDim], dtype: .bfloat16)
                    * Float(layer + 1)
                let values = MLXArray.ones([1, Self.kvHeads, n, Self.headDim], dtype: .bfloat16)
                    * Float(start + 1)
                MLX.eval(keys, values)
                layers.append((keys: keys, values: values))
            }
            block.cacheData = layers
            blocks.append(block)
            start += n
            blockId += 1
        }
        return blocks
    }

    private func offsets(_ cache: [any KVCache]) -> [Int] {
        cache.map(\.offset)
    }

    // MARK: - 1. Pure helper contract

    @Test("all layers at 37 with restoredTokens 37 is a valid restore")
    func consistentRestoreIsValid() {
        FocusedMLXTestSupport.withLock {
            let cache = makeHybridCache(kv0: 37, mamba: 37, kv2: 37)
            // Fixture sanity: the check below must not pass vacuously.
            #expect(offsets(cache) == [37, 37, 37])
            #expect(
                validateRestoredCacheBoundary(cache, matchedTokens: 37, restoredTokens: 37),
                "a fully consistent restore must be accepted")
        }
    }

    @Test("one KV layer at 36 while matched is 37 is refused")
    func shortKVLayerIsRefused() {
        FocusedMLXTestSupport.withLock {
            let cache = makeHybridCache(kv0: 37, mamba: 37, kv2: 36)
            #expect(offsets(cache) == [37, 37, 36])
            #expect(
                !validateRestoredCacheBoundary(cache, matchedTokens: 37, restoredTokens: 37),
                "an attention layer one token short of the boundary must refuse")

            // Same defect on the FIRST KV layer — the one whose tensor seeds
            // `restoredTokens` in the disk path — must also refuse, even when
            // restoredTokens happens to agree with matchedTokens.
            let cacheFirst = makeHybridCache(kv0: 36, mamba: 37, kv2: 37)
            #expect(
                !validateRestoredCacheBoundary(cacheFirst, matchedTokens: 37, restoredTokens: 37))
        }
    }

    @Test("MambaCache at 38 while attention is at 37 is refused (the production desync shape)")
    func recurrentLayerAheadIsRefused() {
        FocusedMLXTestSupport.withLock {
            let cache = makeHybridCache(kv0: 37, mamba: 38, kv2: 37)
            #expect(offsets(cache) == [37, 38, 37])
            #expect(
                !validateRestoredCacheBoundary(cache, matchedTokens: 37, restoredTokens: 37),
                "a recurrent layer ahead of the attention boundary must refuse")

            // And the mirror image: attention at the boundary, mamba behind.
            let behind = makeHybridCache(kv0: 37, mamba: 36, kv2: 37)
            #expect(
                !validateRestoredCacheBoundary(behind, matchedTokens: 37, restoredTokens: 37),
                "a recurrent layer behind the attention boundary must refuse")
        }
    }

    @Test("restoredTokens 36 with every offset at 37 is refused")
    func restoredCountDisagreeingWithOffsetsIsRefused() {
        FocusedMLXTestSupport.withLock {
            let cache = makeHybridCache(kv0: 37, mamba: 37, kv2: 37)
            #expect(offsets(cache) == [37, 37, 37])
            #expect(
                !validateRestoredCacheBoundary(cache, matchedTokens: 37, restoredTokens: 36),
                "restoredTokens must equal matchedTokens even when every layer offset agrees")
            // matchedTokens itself disagreeing with a self-consistent cache
            // is the same refusal from the other side.
            #expect(
                !validateRestoredCacheBoundary(cache, matchedTokens: 36, restoredTokens: 36),
                "a cache at 37 must not be accepted as a 36-token restore")
        }
    }

    @Test("a zero-token restore is never valid")
    func zeroTokenRestoreIsRefused() {
        FocusedMLXTestSupport.withLock {
            let cache = makeHybridCache(kv0: 0, mamba: 0, kv2: 0)
            #expect(offsets(cache) == [0, 0, 0])
            // `restoredTokens > 0` was the old liveness check; the invariant
            // helper must not turn "nothing restored" into an accepted hit.
            #expect(!validateRestoredCacheBoundary(cache, matchedTokens: 0, restoredTokens: 0))
        }
    }

    // MARK: - 2. Desync reproduced through the real restore helpers

    @Test("restoreLayerData(37 tokens) + restoreSSMStates(boundary: 40) is exactly the production desync and is refused")
    func realHelpersProduceDesyncThatIsRefused() {
        FocusedMLXTestSupport.withLock {
            let blockSize = 16
            let blocks = makeBlocks(tokenCount: 37, blockSize: blockSize)
            #expect(blocks.count == 3, "37 tokens at block size 16 must be 16+16+5")
            #expect(blocks.last?.tokenCount == 5, "last block must be PARTIAL — 37 % 16 != 0")

            let target = makeHybridCache(kv0: 0, mamba: 0, kv2: 0)
            let restoredTokens = restoreLayerData(from: blocks, into: target)
            #expect(restoredTokens == 37, "block restore must report the block token total")
            #expect(target[0].offset == 37)
            #expect(target[2].offset == 37)
            #expect(target[1].offset == 0, "restoreLayerData must not touch the recurrent layer")

            // The SSM companion is applied at a DIFFERENT boundary than the
            // blocks covered. This is the coordinator-side shape of the bug:
            // `matchedTokens` drives the recurrent offsets, the block tensors
            // drive the attention offsets, and nothing reconciled them.
            let companion = extractSSMStates(from: [makeMamba(offset: 40)])
            #expect(companion.count == 2)
            restoreSSMStates(companion, into: target, boundary: 40)
            #expect(target[1].offset == 40, "restoreSSMStates(boundary:) seats the recurrent offset from the boundary")
            #expect(offsets(target) == [37, 40, 37], "this IS the desync: attention at 37, recurrent at 40")

            // Both views of the boundary must refuse this cache.
            #expect(
                !validateRestoredCacheBoundary(target, matchedTokens: 40, restoredTokens: restoredTokens),
                "matched=40 vs restored=37 must be refused")
            #expect(
                !validateRestoredCacheBoundary(target, matchedTokens: 37, restoredTokens: restoredTokens),
                "matched=37 with a recurrent layer at 40 must be refused")
        }
    }

    @Test("restoreLayerData(37 tokens) + restoreSSMStates(boundary: 37) is accepted")
    func realHelpersAtTheSameBoundaryAreAccepted() {
        FocusedMLXTestSupport.withLock {
            // Control for the previous test: identical path, boundary agrees.
            let blocks = makeBlocks(tokenCount: 37, blockSize: 16)
            let target = makeHybridCache(kv0: 0, mamba: 0, kv2: 0)
            let restoredTokens = restoreLayerData(from: blocks, into: target)
            #expect(restoredTokens == 37)
            let companion = extractSSMStates(from: [makeMamba(offset: 37)])
            restoreSSMStates(companion, into: target, boundary: 37)
            #expect(offsets(target) == [37, 37, 37])
            #expect(
                validateRestoredCacheBoundary(target, matchedTokens: 37, restoredTokens: restoredTokens),
                "the same helpers at an agreeing boundary must be accepted — otherwise every hybrid hit would be refused")
        }
    }

    @Test("restoreLayerData over blocks whose tensors are one token short of tokenIds is refused")
    func blockTensorShorterThanTokenIdsIsRefused() {
        FocusedMLXTestSupport.withLock {
            // `restoreLayerData` returns SUM(tokenIds.count) but sets the
            // attention offset from the TENSOR length. A block whose KV
            // tensor lost a token (the disk-array analogue is a truncated
            // `kv_{i}_keys`) reports 37 restored while the layers sit at 36.
            let blocks = makeBlocks(tokenCount: 37, blockSize: 16)
            let last = blocks[blocks.count - 1]
            let n = last.tokenCount
            last.cacheData = last.cacheData?.map { entry -> (keys: MLXArray, values: MLXArray)? in
                guard let entry else { return nil }
                return (
                    keys: entry.keys[.ellipsis, ..<(n - 1), 0...],
                    values: entry.values[.ellipsis, ..<(n - 1), 0...]
                )
            }

            let target = makeHybridCache(kv0: 0, mamba: 0, kv2: 0)
            let restoredTokens = restoreLayerData(from: blocks, into: target)
            #expect(restoredTokens == 37, "tokenIds still total 37")
            #expect(target[0].offset == 36, "tensor length drives the attention offset")
            #expect(target[2].offset == 36)
            restoreSSMStates(extractSSMStates(from: [makeMamba(offset: 37)]), into: target, boundary: 37)
            #expect(offsets(target) == [36, 37, 36])
            #expect(
                !validateRestoredCacheBoundary(target, matchedTokens: 37, restoredTokens: restoredTokens),
                "attention offsets 36 under a 37-token boundary must be refused")
        }
    }

    // MARK: - 3. The invariant is wired at every apply site

    /// The helper existing is not the fix; the fix is every coordinator-hit
    /// apply site consulting it. This is the one test in the file that fails
    /// BEFORE the product change (the sites only checked `restoredTokens > 0`)
    /// and passes after.
    @Test("every coordinator-hit apply site consults validateRestoredCacheBoundary")
    func applySitesConsultTheInvariant() throws {
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let libraries = root.appendingPathComponent("Libraries/MLXLMCommon")

        let helpers = try String(
            contentsOf: libraries.appendingPathComponent("Cache/CacheHelpers.swift"),
            encoding: .utf8)
        #expect(
            helpers.contains("public func validateRestoredCacheBoundary("),
            "CacheHelpers.swift must declare the public fail-closed helper")

        let applySites = [
            "Evaluate.swift",
            "BatchEngine/BatchEngine.swift",
            "SpecDec/NativeMTPTokenIterator.swift",
        ]
        for relative in applySites {
            let source = try String(
                contentsOf: libraries.appendingPathComponent(relative), encoding: .utf8)
            // Each of these files restores blocks and/or disk arrays on a
            // coordinator hit; a site that never names the helper still
            // accepts a desynchronised restore on `restoredTokens > 0` alone.
            #expect(
                source.contains("validateRestoredCacheBoundary("),
                "\(relative) applies a coordinator cache hit but never validates the restored boundary")
        }
    }
}
