// MiniMaxM3SparseCache contract tests — the 3-lane (keys, values, idx_keys)
// reuse primitive that every osaurus cache tier (prefix/SSD/snapshot/trim) leans
// on. No model load required: pure cache logic. These prove the lanes grow,
// trim, copy, and serialize in lockstep so a reuse path can NEVER silently drop
// idx_keys (the root cause of the historical repetition-loop saga).

import Foundation
import MLX
@testable import MLXLMCommon
import Testing

@Suite("MiniMaxM3SparseCache contract", .serialized)
struct MiniMaxM3SparseCacheTests {
    private static let nKV = 4
    private static let headDim = 128
    private static let indexDim = 128

    private func kv(_ s: Int) -> (MLXArray, MLXArray) {
        (
            MLXArray.zeros([1, Self.nKV, s, Self.headDim], dtype: .bfloat16),
            MLXArray.zeros([1, Self.nKV, s, Self.headDim], dtype: .bfloat16)
        )
    }
    private func idx(_ s: Int) -> MLXArray {
        MLXArray.zeros([1, 1, s, Self.indexDim], dtype: .bfloat16)
    }

    @Test("Append grows all 3 lanes in lockstep; idx history == offset")
    func appendLockstep() {
        let c = MiniMaxM3SparseCache(indexDim: Self.indexDim)
        let (k, v) = kv(4)
        _ = c.update(keys: k, values: v)            // K/V first (upstream order)
        #expect(c.offset == 4)
        let hist = c.updateIndex(idx(4))             // then indexer lane
        #expect(hist.dim(2) == 4)                    // sliced to offset

        let (k2, v2) = kv(1)
        _ = c.update(keys: k2, values: v2)
        #expect(c.offset == 5)
        let hist2 = c.updateIndex(idx(1))
        #expect(hist2.dim(2) == 5)
        #expect(c.readIndex().dim(2) == 5)
    }

    @Test("Trim slices K/V AND idx_keys to the same length")
    func trimLockstep() {
        let c = MiniMaxM3SparseCache(indexDim: Self.indexDim)
        let (k, v) = kv(10)
        _ = c.update(keys: k, values: v)
        _ = c.updateIndex(idx(10))
        #expect(c.offset == 10)

        let trimmed = c.trim(3)
        #expect(trimmed == 3)
        #expect(c.offset == 7)
        #expect(c.readIndex().dim(2) == 7)           // idx lane trimmed lockstep
    }

    @Test("copy() preserves the sparse TYPE and all 3 lanes")
    func copyPreservesTypeAndLanes() {
        let c = MiniMaxM3SparseCache(indexDim: Self.indexDim)
        let (k, v) = kv(6)
        _ = c.update(keys: k, values: v)
        _ = c.updateIndex(idx(6))

        let dup = c.copy()
        #expect(dup is MiniMaxM3SparseCache)         // never a plain KVCache
        #expect(dup.offset == 6)
        let dupSparse = try! #require(dup as? MiniMaxM3SparseCache)
        #expect(dupSparse.readIndex().dim(2) == 6)
    }

    @Test("state round-trip carries the idx_keys lane")
    func stateRoundTrip() {
        let c = MiniMaxM3SparseCache(indexDim: Self.indexDim)
        let (k, v) = kv(5)
        _ = c.update(keys: k, values: v)
        _ = c.updateIndex(idx(5))

        let s = c.state
        #expect(s.count == 3)                        // [keys, values, idx_keys]
        let m = c.metaState

        let restored = MiniMaxM3SparseCache(indexDim: Self.indexDim)
        restored.state = s.map { $0[.ellipsis] }
        restored.metaState = m
        #expect(restored.offset == 5)
        #expect(restored.readIndex().dim(2) == 5)
    }

    @Test("Topology classifies M3 sparse layers as composite (not plain KV)")
    func topologyClassification() {
        var caches: [any KVCache] = []
        for _ in 0 ..< 3 { caches.append(KVCacheSimple()) }                     // dense 0-2
        for _ in 0 ..< 57 { caches.append(MiniMaxM3SparseCache(indexDim: Self.indexDim)) }  // sparse 3-59
        let topo = ModelCacheTopologySnapshot(cache: caches)
        #expect(topo.layerCount == 60)
        #expect(topo.kvLayerCount == 3)                       // only the 3 dense layers
        #expect(topo.miniMaxM3SparseLayerCount == 57)         // NOT folded into kv
        #expect(topo.requiresMiniMaxM3SparseState)
        #expect(!topo.requiresSSMCompanionState)              // M3 is not an SSM hybrid
        #expect(topo.requiresDiskBackedCoordinatorRestore)    // composite must disk-restore
        #expect(topo.topologyTags.contains("minimaxM3SparseLayers=57"))
    }

    @Test("M3 cache list flips paged-incompatible (disk restore) but is NOT SSM path-dependent")
    func autoFlipHelpers() {
        var caches: [any KVCache] = [KVCacheSimple(), KVCacheSimple(), KVCacheSimple()]
        for _ in 0 ..< 57 { caches.append(MiniMaxM3SparseCache(indexDim: Self.indexDim)) }
        // Must restore through the disk serializer (idx_keys can't ride the paged tier)…
        #expect(cacheRequiresDiskBackedCoordinatorRestore(caches))
        // …but must NOT be treated as SSM/conv path-dependent (no companion extraction).
        #expect(!cacheContainsPathDependentState(caches))

        // A purely dense KV list triggers neither.
        let dense: [any KVCache] = [KVCacheSimple(), KVCacheSimple()]
        #expect(!cacheRequiresDiskBackedCoordinatorRestore(dense))
        #expect(!cacheContainsPathDependentState(dense))
    }

    @Test("TQDiskSerializer round-trips M3 sparse layers with idx_keys intact")
    func diskSerializerRoundTrip() {
        // Source: 3 dense + 2 M3 sparse, all populated to 7 tokens.
        var src: [any KVCache] = [KVCacheSimple(), KVCacheSimple(), KVCacheSimple()]
        for i in 0 ..< 3 { let (k, v) = kv(7); _ = src[i].update(keys: k, values: v) }
        for _ in 0 ..< 2 {
            let c = MiniMaxM3SparseCache(indexDim: Self.indexDim)
            let (k, v) = kv(7)
            _ = c.update(keys: k, values: v)
            _ = c.updateIndex(idx(7))
            src.append(c)
        }

        let arrays = TQDiskSerializer.serialize(cache: src)
        // M3 layers must serialize as their own kind, not skip.
        #expect(arrays["mm3_3_idx_keys"] != nil)
        #expect(arrays["mm3_4_idx_keys"] != nil)

        // Restore into a fresh typed list (as `newCache()` would build).
        var dst: [any KVCache] = [
            KVCacheSimple(), KVCacheSimple(), KVCacheSimple(),
            MiniMaxM3SparseCache(indexDim: Self.indexDim),
            MiniMaxM3SparseCache(indexDim: Self.indexDim),
        ]
        _ = restoreFromDiskArrays(arrays, into: &dst)

        for i in 3 ..< 5 {
            let m3 = try! #require(dst[i] as? MiniMaxM3SparseCache)
            #expect(m3.offset == 7)
            #expect(m3.readIndex().dim(2) == 7)   // idx_keys survived the round-trip
        }
    }

    @Test("maybeQuantizeKVCache never TQ/affine-encodes an M3 sparse layer (B6)")
    func tqKvSkipsM3Sparse() {
        // Dense KVCacheSimple + M3 sparse, both populated well past any threshold.
        let dense = KVCacheSimple()
        let (dk, dv) = kv(2048)
        _ = dense.update(keys: dk, values: dv)
        let m3 = MiniMaxM3SparseCache(indexDim: Self.indexDim)
        let (mk, mv) = kv(2048)
        _ = m3.update(keys: mk, values: mv)
        _ = m3.updateIndex(idx(2048))

        var cache: [KVCache] = [dense, m3]
        maybeQuantizeKVCache(
            cache: &cache, kvBits: nil, quantizedKVStart: 8,
            kvMode: .turboQuant(keyBits: 3, valueBits: 3))
        // The MSA layer is never converted — its idx_keys lane can't be encoded.
        #expect(cache[1] is MiniMaxM3SparseCache)
        #expect(!(cache[1] is TurboQuantKVCache))

        // Same for affine mode.
        var cache2: [KVCache] = [KVCacheSimple(), m3]
        let (k0, v0) = kv(2048); _ = (cache2[0] as! KVCacheSimple).update(keys: k0, values: v0)
        maybeQuantizeKVCache(cache: &cache2, kvBits: 4, quantizedKVStart: 8)
        #expect(cache2[1] is MiniMaxM3SparseCache)
    }

    @Test("innerState() returns all 3 lanes so eval(cache) materializes idx_keys (B7)")
    func innerStateCoversIdxKeys() {
        let c = MiniMaxM3SparseCache(indexDim: Self.indexDim)
        let (k, v) = kv(5)
        _ = c.update(keys: k, values: v)
        _ = c.updateIndex(idx(5))
        // eval(cache) drives Evaluatable.innerState(); 3 lanes ⇒ idx_keys can't lag K/V.
        #expect(c.innerState().count == 3)
        #expect(c.innerState()[2].dim(2) == 5)
    }

    @Test("empty cache state has a zero-length idx sentinel, round-trips to empty")
    func emptyRoundTrip() {
        let c = MiniMaxM3SparseCache(indexDim: Self.indexDim)
        let s = c.state
        #expect(s.count == 3)
        #expect(s[2].dim(2) == 0)                     // empty idx sentinel
        let restored = MiniMaxM3SparseCache(indexDim: Self.indexDim)
        restored.state = s.map { $0[.ellipsis] }
        #expect(restored.offset == 0)
    }
}
