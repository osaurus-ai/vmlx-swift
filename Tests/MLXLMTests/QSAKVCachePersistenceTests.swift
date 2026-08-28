// Copyright © 2026 Apple Inc.

// Regression tests for osaurus#2525: the qwen4_exp QSA indexer-keys lane
// must survive the disk round-trip, and any restore that would leave the
// lane out of sync with `offset` must fail CLOSED (fresh prefill), never
// seat a divergent cache. The shipped crash was a restore that kept the
// main K/V (offset ~2700) while the lane restarted at the post-restore
// suffix (2052 rows) — the sparse selector then emitted a [1,1,T,2052]
// mask against a ~4770-long KV and SDPA fataled on the broadcast.

import Foundation
import MLX
import Testing

@testable import MLXLMCommon

private func makeQSACache(
    tokens: Int, kvHeads: Int = 2, headDim: Int = 8, indexerDim: Int = 4
) -> QSAKVCache {
    let cache = QSAKVCache()
    let keys = MLXArray.ones([1, kvHeads, tokens, headDim]).asType(.bfloat16)
    let values = MLXArray.ones([1, kvHeads, tokens, headDim]).asType(.bfloat16)
    // The layer calls updateIndexerKeys BEFORE update() advances offset,
    // matching Qwen4ExpQSAIndexer's forward order.
    _ = cache.updateIndexerKeys(MLXArray.ones([1, tokens, indexerDim]).asType(.bfloat16))
    _ = cache.update(keys: keys, values: values)
    return cache
}

/// The full round-trip: serialize → deserialize (kind `.qsaKV`) → restore
/// into a fresh QSAKVCache → the lane covers exactly `offset`, and a
/// follow-up segment keeps `total == offset + S` — the invariant whose
/// violation was the shipped broadcast crash. Uses a restored offset just
/// UNDER the 2048 indexer budget and a suffix that crosses it: the hard
/// shape, because the sparse mask only materializes past the budget.
@Test func qsaRoundTripPreservesIndexerLaneAcrossBudgetCrossing() {
    MLXMetalTestLock.withLock {
        let restoredTokens = 2040
        let cache = makeQSACache(tokens: restoredTokens)
        let dict = TQDiskSerializer.serialize(cache: [cache])

        let layers = TQDiskSerializer.deserializeIndexed(dict)
        #expect(layers.count == 1)
        guard case .qsaKV(let comp) = layers[0].data else {
            Issue.record("expected .qsaKV, got \(layers[0].data)")
            return
        }
        #expect(comp.indexerKeys.dim(1) == restoredTokens)

        var fresh: [any KVCache] = [QSAKVCache()]
        let restored = restoreFromDiskArrays(dict, into: &fresh)
        #expect(restored == restoredTokens)
        guard let qsa = fresh[0] as? QSAKVCache else {
            Issue.record("restore replaced the QSAKVCache type")
            return
        }
        #expect(qsa.offset == restoredTokens)
        #expect(qsa.indexerKeys?.dim(1) == restoredTokens)

        // Continuation segment crossing the 2048 budget: the lane total MUST
        // track offset + S or the selector mask and the KV length diverge.
        let suffix = 16
        let all = qsa.updateIndexerKeys(
            MLXArray.ones([1, suffix, 4]).asType(.bfloat16))
        #expect(all.dim(1) == restoredTokens + suffix)
        #expect(all.dim(1) > 2048)
    }
}

/// A pre-fix disk entry (plain `.kv`, no indexer lane) restored into a QSA
/// layer must be a WHOLE-RECORD miss: zero restored tokens and an untouched
/// cache, so the caller re-prefills instead of decoding against a cache
/// whose lane restarts at zero.
@Test func legacyTwoArrayRecordForQSALayerFailsClosed() {
    MLXMetalTestLock.withLock {
        // Build the legacy record by serializing a plain KVCacheSimple.
        let simple = KVCacheSimple()
        _ = simple.update(
            keys: MLXArray.ones([1, 2, 64, 8]).asType(.bfloat16),
            values: MLXArray.ones([1, 2, 64, 8]).asType(.bfloat16))
        let legacyDict = TQDiskSerializer.serialize(cache: [simple])

        var qsaCache: [any KVCache] = [QSAKVCache()]
        let restored = restoreFromDiskArrays(legacyDict, into: &qsaCache)
        #expect(restored == 0)
        #expect((qsaCache[0] as? QSAKVCache)?.offset == 0)
        #expect((qsaCache[0] as? QSAKVCache)?.indexerKeys == nil)
    }
}

/// A QSA cache that somehow lost its lane (2-array state) must serialize
/// as `.skip` — a lossy `.kv` record would recreate the divergence on the
/// NEXT process's restore.
@Test func qsaWithoutIndexerLaneSerializesAsSkip() {
    MLXMetalTestLock.withLock {
        let cache = QSAKVCache()
        cache.state = [
            MLXArray.ones([1, 2, 32, 8]).asType(.bfloat16),
            MLXArray.ones([1, 2, 32, 8]).asType(.bfloat16),
        ]
        #expect(cache.indexerKeys == nil)
        let dict = TQDiskSerializer.serialize(cache: [cache])
        let layers = TQDiskSerializer.deserializeIndexed(dict)
        #expect(layers.count == 1)
        guard case .skip = layers[0].data else {
            Issue.record("expected .skip for a lane-less QSA cache, got \(layers[0].data)")
            return
        }
    }
}

/// A stored QSA record whose lane length disagrees with the KV offset is
/// damage — restore must refuse it rather than seat a divergent state.
@Test func qsaRecordWithShortLaneFailsClosed() {
    MLXMetalTestLock.withLock {
        let cache = makeQSACache(tokens: 64)
        var dict = TQDiskSerializer.serialize(cache: [cache])
        dict["kv_0_indexer_keys"] = MLXArray.ones([1, 40, 4]).asType(.bfloat16)

        var fresh: [any KVCache] = [QSAKVCache()]
        let restored = restoreFromDiskArrays(dict, into: &fresh)
        #expect(restored == 0)
        #expect((fresh[0] as? QSAKVCache)?.offset == 0)
    }
}

/// KV-cache quantization must leave QSA layers untouched: TurboQuant's
/// `fromSimpleCache` refuses 3-array state and would return an EMPTY
/// cache, and the affine path would replace the type the indexer needs.
@Test func kvQuantizationSkipsQSALayers() {
    MLXMetalTestLock.withLock {
        let qsa = makeQSACache(tokens: 6000)
        var cache: [KVCache] = [qsa]
        maybeQuantizeKVCache(cache: &cache, kvBits: 4, quantizedKVStart: 128)
        #expect((cache[0] as? QSAKVCache) === qsa)
        #expect((cache[0] as? QSAKVCache)?.indexerKeys != nil)
    }
}
