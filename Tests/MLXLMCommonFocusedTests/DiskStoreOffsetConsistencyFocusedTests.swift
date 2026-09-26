// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// The disk tier must refuse a snapshot whose rotating offset does not match
// the boundary token count it is being stored under.
//
// Observed live (2026-08-02, VMLX_CACHE_FETCH_TRACE=1):
//
//   [vmlx][cache/paged-store] tokens=1831 requiredCompanion=true
//       effectiveKVLayers=0 blocks=0 payload=false companion=false
//       rotatingOffsets=[1832]
//
// A snapshot keyed at 1831 tokens carried a rotating cache at offset 1832.
// The PAGED tier caught it — `serializePagedRotatingCompanion(expectedOffset:)`
// returned nil, so `companion=false` and the paged store was refused. The DISK
// tier had no equivalent guard: `TQDiskSerializer.serialize(cache:)` ran
// unconditionally and the poisoned entry was persisted to SSD.
//
// What a later fetch does with it: content-addressing matches the 1831-token
// key, restore reinstates offset 1832, the engine believes 1831 tokens are
// covered and re-feeds token index 1831 — which the cache already contains.
// One duplicated position shifts every subsequent token's position by one, so
// attention is subtly wrong for the whole continuation while generation stays
// fluent. Each following turn captures its N-1 seed FROM that corrupted state
// and stores it, compounding per turn.
//
// This is the DSV4 agent-loop degeneration: it appears only with disk L2
// enabled (RAM prefix reuse shares the live object and never round-trips), it
// worsens across tool rounds, and DSV4 is hit hardest because it is
// paged-incompatible, so the disk tier is its only reuse path — every turn
// round-trips through exactly this seam.

import Foundation
import MLX
@testable import MLXLMCommon
import Testing

@Suite("Disk store offset consistency", .serialized)
struct DiskStoreOffsetConsistencyFocusedTests {

    private func makeCoordinator(diskDir: URL) -> CacheCoordinator {
        CacheCoordinator(config: CacheCoordinatorConfig(
            usePagedCache: false,
            enableDiskCache: true,
            pagedBlockSize: 4,
            maxCacheBlocks: 16,
            diskCacheMaxGB: 1.0,
            diskCacheDir: diskDir,
            modelKey: "disk-offset-consistency"))
    }

    private func rotatingCache(fedTokens: Int, window: Int = 8) -> RotatingKVCache {
        let rot = RotatingKVCache(maxSize: window, keep: 0)
        for position in 0 ..< fedTokens {
            _ = rot.update(
                keys: MLXArray.ones([1, 1, 1, 4]) * Float(position + 1),
                values: MLXArray.ones([1, 1, 1, 4]) * Float(-(position + 1)))
        }
        MLX.eval(rot)
        return rot
    }

    private func tempDir() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("disk-offset-consistency-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    @Test("a snapshot whose offset exceeds the boundary must not reach disk")
    func inconsistentSnapshotIsRefused() throws {
        let dir = try tempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        let coordinator = makeCoordinator(diskDir: dir)

        // The live shape: keyed at N tokens, cache actually fed N+1.
        let tokens = Array(0 ..< 11)
        let cache = rotatingCache(fedTokens: 12)
        #expect(cache.offset == 12)

        coordinator.storeAfterGeneration(
            promptTokens: tokens,
            perLayerData: [],
            ssmStates: nil,
            cache: [cache])

        // A fetch at the same key must MISS: restoring this entry would hand
        // back one more token of state than the engine believes is covered,
        // duplicating the boundary token on re-feed.
        let result = coordinator.fetch(tokens: tokens)
        guard case .miss = result else {
            Issue.record(
                """
                poisoned entry was stored and served: keyed=\(tokens.count) offset=12 — \
                the boundary token will be fed twice and every later position shifts
                """)
            return
        }
    }

    @Test("a consistent snapshot still stores and hits")
    func consistentSnapshotStillStores() throws {
        let dir = try tempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        let coordinator = makeCoordinator(diskDir: dir)

        let tokens = Array(0 ..< 12)
        let cache = rotatingCache(fedTokens: 12)
        #expect(cache.offset == 12)

        coordinator.storeAfterGeneration(
            promptTokens: tokens,
            perLayerData: [],
            ssmStates: nil,
            cache: [cache])

        // Guard must not be over-broad: the healthy path keeps working. The
        // exact boundary is skippable by policy, so probe via a longer prompt
        // that can reuse this entry as a prefix.
        let longer = tokens + [12, 13]
        let result = coordinator.fetch(tokens: longer)
        guard case .hit(let matched, _, _, _, _, _) = result else {
            Issue.record("consistent entry did not round-trip — guard is over-broad")
            return
        }
        #expect(matched == 12)
    }

    @Test("composite snapshots store by leaf boundaries and restore from disk")
    func compositeSnapshotStoresAndRestores() throws {
        let dir = try tempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        let coordinator = makeCoordinator(diskDir: dir)
        let tokens = Array(0..<12)
        let recurrent = MambaCache()
        recurrent[0] = MLXArray.ones([1, 2, 4])
        recurrent[1] = MLXArray.ones([1, 4, 4])
        recurrent.offset = tokens.count
        let attention = KVCacheSimple()
        _ = attention.update(keys: MLXArray.ones([1, 1, 12, 4]), values: MLXArray.ones([1, 1, 12, 4]))
        MLX.eval(attention)
        let nested = CacheList(recurrent, attention)
        #expect(nested.offset == 0)
        coordinator.storeAfterGeneration(
            promptTokens: tokens, perLayerData: [], ssmStates: nil, cache: [nested])
        guard case .hit(let matched, _, _, _, _, let arrays) =
            coordinator.fetch(tokens: tokens + [12, 13]), let arrays
        else {
            Issue.record("valid composite was not stored and fetched through the coordinator")
            return
        }
        var restored: [any KVCache] = [CacheList(MambaCache(), KVCacheSimple())]
        let count = restoreFromDiskArrays(arrays, into: &restored, requirePromptBoundary: true)
        #expect(matched == tokens.count)
        #expect(count == tokens.count, "Stored payload: \(arrays.keys.sorted().map { "\($0):\(arrays[$0]!.dtype)" })")
        #expect(validateRestoredCacheBoundary(restored, matchedTokens: matched, restoredTokens: count))
    }

    @Test("unsupported deeper composite serialization does not claim a restored boundary")
    func unsupportedNestedSerializationFailsClosed() {
        let original = CacheList(CacheList(rotatingCache(fedTokens: 12)))
        let arrays = TQDiskSerializer.serialize(cache: [original])
        var target: [any KVCache] = [CacheList(CacheList(RotatingKVCache(maxSize: 8, keep: 0)))]
        #expect(restoreFromDiskArrays(arrays, into: &target, requirePromptBoundary: true) == 0)
        #expect(cacheBoundaryLeafOffsets(target) == [0])
    }

    @Test("composite store rejects a mismatched child or empty subtree")
    func invalidCompositeStoreIsRefused() throws {
        for empty in [false, true] {
            let dir = try tempDir()
            defer { try? FileManager.default.removeItem(at: dir) }
            let coordinator = makeCoordinator(diskDir: dir)
            let good = rotatingCache(fedTokens: 12)
            let bad: any KVCache = empty ? CacheList([]) : rotatingCache(fedTokens: 13)
            coordinator.storeAfterGeneration(
                promptTokens: Array(0..<12), perLayerData: [], ssmStates: nil,
                cache: [CacheList(good, CacheList(bad))])
            guard case .miss = coordinator.fetch(tokens: Array(0..<14)) else {
                Issue.record("invalid composite reached disk (empty=\(empty))")
                continue
            }
        }
    }

    // MARK: - Post-answer boundary key alignment (consumed stop token)

    /// The async decode pipeline forwards the consumed stop token while
    /// computing the never-consumed next step, so at store time every cache
    /// layer sits ONE token past `prompt + generated`. Keying the store at
    /// `prompt + generated` desynchronizes key and cache, and the boundary
    /// guard above (correctly) refuses it — observed live as
    /// `REFUSED offset/key mismatch tokens=3627 offsets=[3628]`, silently
    /// costing the post-answer boundary every DSV4 turn. The aligned key
    /// extends by the pending drained token exactly when the cache is
    /// uniformly one ahead and the iterator still holds that token.
    @Test("post-answer boundary key extends by the consumed stop token")
    func generatedBoundaryKeyExtendsByConsumedStop() {
        // Exact match: key unchanged.
        #expect(
            TokenIterator.generatedBoundaryTokensAligned(
                promptTokenIds: [1, 2, 3],
                generatedTokenIds: [4, 5],
                cacheOffsets: [5, 5],
                pendingDrainedTokenId: 99) == [1, 2, 3, 4, 5])
        // Uniformly one ahead with the drained stop token available:
        // key extends.
        #expect(
            TokenIterator.generatedBoundaryTokensAligned(
                promptTokenIds: [1, 2, 3],
                generatedTokenIds: [4, 5],
                cacheOffsets: [6, 6],
                pendingDrainedTokenId: 99) == [1, 2, 3, 4, 5, 99])
        // One ahead but the drained token is unavailable: fall back to the
        // base key so the store guard decides (fail-closed).
        #expect(
            TokenIterator.generatedBoundaryTokensAligned(
                promptTokenIds: [1, 2, 3],
                generatedTokenIds: [4, 5],
                cacheOffsets: [6, 6],
                pendingDrainedTokenId: nil) == [1, 2, 3, 4, 5])
        // Mixed offsets: no consistent extension; base key for the guard.
        #expect(
            TokenIterator.generatedBoundaryTokensAligned(
                promptTokenIds: [1, 2, 3],
                generatedTokenIds: [4, 5],
                cacheOffsets: [5, 6],
                pendingDrainedTokenId: 99) == [1, 2, 3, 4, 5])
        // Cache BEHIND the key: no valid store exists.
        #expect(
            TokenIterator.generatedBoundaryTokensAligned(
                promptTokenIds: [1, 2, 3],
                generatedTokenIds: [4, 5],
                cacheOffsets: [4, 4],
                pendingDrainedTokenId: 99) == nil)
    }
}
