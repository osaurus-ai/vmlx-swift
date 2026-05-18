// Copyright © 2025 JANG. All rights reserved.
//
// Round-trip tests for `TQDiskSerializer` covering every layer kind the
// L2 disk cache supports: standard KV, Mamba SSM, QuantizedKVCache, and
// the SSM companion state fold. These tests are synthetic — they do not
// load any model and use small MLXArrays only.

import Foundation
import MLX
@testable import MLXLMCommon
import Testing

// MARK: - Helpers

/// Build a tiny K/V tensor pair the cache can ingest.
private func smallKV(seqLen: Int = 8) -> (MLXArray, MLXArray) {
    let keys = MLXArray.ones([1, 4, seqLen, 16], dtype: .bfloat16)
    let values = MLXArray.ones([1, 4, seqLen, 16], dtype: .bfloat16) * Float(0.5)
    return (keys, values)
}

// MARK: - Layer kind: KV (KVCacheSimple)

@Test
func testRoundTripKVCacheSimple() async throws {
    let mlxTestLock = lockSerializedMLXTest()
    defer { mlxTestLock.unlock() }

    let cache: [any KVCache] = (0..<3).map { _ in KVCacheSimple() }
    let (k, v) = smallKV()
    for layer in cache { _ = layer.update(keys: k, values: v) }

    let arrays = TQDiskSerializer.serialize(cache: cache)

    #expect(TQDiskSerializer.formatVersion(of: arrays) == 2)
    #expect(arrays.keys.contains("__layer_kind_0__"))
    #expect(arrays.keys.contains("kv_0_keys"))
    #expect(arrays.keys.contains("kv_2_values"))

    // Round-trip into a fresh cache of the same shape.
    var restored: [any KVCache] = (0..<3).map { _ in KVCacheSimple() }
    let n = restoreFromDiskArrays(arrays, into: &restored)
    #expect(n == 8)
    for i in 0..<3 {
        #expect(restored[i].state.count == 2)
        #expect(restored[i].state[0].dim(2) == 8)
    }
}

// MARK: - Layer kind: Mamba

@Test
func testRoundTripMambaCache() async throws {
    let mlxTestLock = lockSerializedMLXTest()
    defer { mlxTestLock.unlock() }

    let cache: [any KVCache] = (0..<2).map { _ in MambaCache() }
    let convState = MLXArray.ones([1, 32, 4], dtype: .float32) * Float(0.25)
    let ssmState = MLXArray.ones([1, 32, 16], dtype: .float32) * Float(0.75)
    for layer in cache {
        if let arrays = layer as? ArraysCache {
            arrays[0] = convState
            arrays[1] = ssmState
        }
    }

    let dict = TQDiskSerializer.serialize(cache: cache)

    #expect(dict.keys.contains("__layer_kind_0__"))
    #expect(dict.keys.contains("mamba_0_state0"))
    #expect(dict.keys.contains("mamba_1_state1"))
    #expect(dict.keys.contains("__mamba_0_offset__"))
    // Mamba layers must NOT also be written as kv_*.
    #expect(!dict.keys.contains("kv_0_keys"))

    // Restore into fresh caches and confirm state was set.
    var restored: [any KVCache] = (0..<2).map { _ in MambaCache() }
    _ = restoreFromDiskArrays(dict, into: &restored)
    for layer in restored {
        #expect(layer.state.count == 2)
        // Both state arrays should have a non-zero element count.
        #expect(layer.state[0].size > 0)
        #expect(layer.state[1].size > 0)
    }
}

// MARK: - Layer kind: QuantizedKVCache

@Test
func testRoundTripQuantizedKVCache() async throws {
    let mlxTestLock = lockSerializedMLXTest()
    defer { mlxTestLock.unlock() }

    // group_size 32 is the smallest MLX supports; the tensor's last dim
    // must be a multiple of the group size (32 → 32 ✓).
    let cache: [any KVCache] = (0..<2).map { _ in
        QuantizedKVCache(groupSize: 32, bits: 8)
    }
    let keys = MLXArray.ones([1, 4, 8, 32], dtype: .float16)
    let values = MLXArray.ones([1, 4, 8, 32], dtype: .float16) * Float(0.5)
    for layer in cache {
        if let qkv = layer as? QuantizedKVCache {
            _ = qkv.updateQuantized(keys: keys, values: values)
        }
    }

    let dict = TQDiskSerializer.serialize(cache: cache)

    #expect(dict.keys.contains("__layer_kind_0__"))
    #expect(dict.keys.contains("__qkv_0_count__"))
    #expect(dict.keys.contains("__qkv_0_offset__"))
    #expect(dict.keys.contains("__qkv_0_group_size__"))
    #expect(dict.keys.contains("__qkv_0_bits__"))
    #expect(dict.keys.contains("qkv_0_0"))   // first state array
    #expect(dict.keys.contains("qkv_1_0"))   // second layer's first state array

    // Restore into fresh quantized caches and verify offset round-trips.
    var restored: [any KVCache] = (0..<2).map { _ in
        QuantizedKVCache(groupSize: 32, bits: 8)
    }
    _ = restoreFromDiskArrays(dict, into: &restored)
    for layer in restored {
        #expect(layer.offset == cache[0].offset)
        #expect(layer.state.count == cache[0].state.count)
    }
}

@Test
func testQKVDiskRestoreMaterializesFreshSimpleCache() async throws {
    let mlxTestLock = lockSerializedMLXTest()
    defer { mlxTestLock.unlock() }

    let source = QuantizedKVCache(groupSize: 32, bits: 8)
    let keys = MLXArray.ones([1, 4, 8, 32], dtype: .float16)
    let values = MLXArray.ones([1, 4, 8, 32], dtype: .float16) * Float(0.5)
    _ = source.updateQuantized(keys: keys, values: values)

    let dict = TQDiskSerializer.serialize(cache: [source])
    var restored: [any KVCache] = [KVCacheSimple()]
    let restoredTokens = restoreFromDiskArrays(dict, into: &restored)

    #expect(restoredTokens == source.offset)
    #expect(restored[0] is QuantizedKVCache)
    guard let qkv = restored[0] as? QuantizedKVCache else { return }
    #expect(qkv.groupSize == 32)
    #expect(qkv.bits == 8)
    #expect(qkv.offset == source.offset)
    #expect(qkv.state.count == source.state.count)
}

@Test
func testTQDiskRestoreMaterializesFreshSimpleCache() async throws {
    let mlxTestLock = lockSerializedMLXTest()
    defer { mlxTestLock.unlock() }

    let sourceSimple = KVCacheSimple()
    let (keys, values) = smallKV(seqLen: 96)
    _ = sourceSimple.update(keys: keys, values: values)
    let sourceTQ = TurboQuantKVCache.fromSimpleCache(
        sourceSimple,
        keyBits: 3,
        valueBits: 3,
        sinkTokens: 4)
    #expect(TQDiskSerializer.isTQCompressed(sourceTQ))

    let dict = TQDiskSerializer.serialize(cache: [sourceTQ])
    var restored: [any KVCache] = [KVCacheSimple()]
    let restoredTokens = restoreFromDiskArrays(dict, into: &restored)

    #expect(restoredTokens == sourceSimple.offset)
    #expect(restored[0] is TurboQuantKVCache)
    guard let tq = restored[0] as? TurboQuantKVCache else { return }
    #expect(TQDiskSerializer.isTQCompressed(tq))
    #expect(tq.offset == sourceSimple.offset)
    #expect(tq.compressedKeys?.tailCount == TurboQuantKVCache.defaultResidualTokens)
}

// MARK: - Hybrid mix (KV + Mamba) + SSM companion state

@Test
func testRoundTripHybridWithSSMState() async throws {
    let mlxTestLock = lockSerializedMLXTest()
    defer { mlxTestLock.unlock() }

    // Layer 0: attention KV. Layer 1: Mamba. Layer 2: attention KV.
    let cache: [any KVCache] = [
        KVCacheSimple(),
        MambaCache(),
        KVCacheSimple(),
    ]
    let (k, v) = smallKV(seqLen: 12)
    _ = cache[0].update(keys: k, values: v)
    if let m = cache[1] as? ArraysCache {
        m[0] = MLXArray.ones([1, 8, 4], dtype: .float32)
        m[1] = MLXArray.ones([1, 8, 16], dtype: .float32) * Float(0.25)
    }
    _ = cache[2].update(keys: k, values: v)

    // Companion SSM state — three arrays, simulating a hybrid runtime
    // capturing per-mamba-layer SSM snapshots at the prefill boundary.
    let ssm: [MLXArray] = [
        MLXArray.ones([1, 8, 16], dtype: .float32) * Float(0.1),
        MLXArray.ones([1, 8, 16], dtype: .float32) * Float(0.2),
        MLXArray.ones([1, 8, 16], dtype: .float32) * Float(0.3),
    ]

    let dict = TQDiskSerializer.serialize(cache: cache, ssmStates: ssm)

    // Sanity: every layer has a kind tag, kinds match the cache layout.
    #expect(dict.keys.contains("__layer_kind_0__"))
    #expect(dict.keys.contains("__layer_kind_1__"))
    #expect(dict.keys.contains("__layer_kind_2__"))
    let kind0 = dict["__layer_kind_0__"]!.item(Int32.self)
    let kind1 = dict["__layer_kind_1__"]!.item(Int32.self)
    let kind2 = dict["__layer_kind_2__"]!.item(Int32.self)
    #expect(kind0 == TQDiskSerializer.LayerKind.kv.rawValue)
    #expect(kind1 == TQDiskSerializer.LayerKind.mamba.rawValue)
    #expect(kind2 == TQDiskSerializer.LayerKind.kv.rawValue)

    // SSM companion state was folded into the same dict.
    #expect(dict["__ssm_count__"]?.item(Int32.self) == 3)
    #expect(dict.keys.contains("ssm_0"))
    #expect(dict.keys.contains("ssm_2"))
    let recoveredSSM = TQDiskSerializer.ssmStates(from: dict)
    #expect(recoveredSSM?.count == 3)

    // Restore round-trip: KV layers and Mamba layers all populated by index.
    var restored: [any KVCache] = [
        KVCacheSimple(),
        MambaCache(),
        KVCacheSimple(),
    ]
    let n = restoreFromDiskArrays(dict, into: &restored)
    #expect(n == 12)
    #expect(restored[0].state.count == 2)
    #expect(restored[1].state.count == 2)
    #expect(restored[2].state.count == 2)
}

// MARK: - Format version detection

@Test
func testFormatVersionTaggingV2() async throws {
    let mlxTestLock = lockSerializedMLXTest()
    defer { mlxTestLock.unlock() }

    let cache: [any KVCache] = [KVCacheSimple()]
    let (k, v) = smallKV()
    _ = cache[0].update(keys: k, values: v)
    let dict = TQDiskSerializer.serialize(cache: cache)
    #expect(TQDiskSerializer.formatVersion(of: dict) == 2)
    #expect(TQDiskSerializer.isTQNative(dict))
}

@Test
func testFormatVersionLegacyV1Detection() async throws {
    let mlxTestLock = lockSerializedMLXTest()
    defer { mlxTestLock.unlock() }

    // Hand-craft a v1 dict: just the legacy marker + a bare kv pair.
    let dict: [String: MLXArray] = [
        "__tq_native_marker__": MLXArray([Int32(1)]),
        "kv_0_keys": MLXArray.ones([1, 4, 8, 16], dtype: .bfloat16),
        "kv_0_values": MLXArray.ones([1, 4, 8, 16], dtype: .bfloat16),
    ]
    #expect(TQDiskSerializer.formatVersion(of: dict) == 1)
    #expect(TQDiskSerializer.isTQNative(dict))

    // V1 round-trip into an attention-only cache should still work via the
    // legacy fallback path inside restoreFromDiskArrays.
    var cache: [any KVCache] = [KVCacheSimple()]
    let n = restoreFromDiskArrays(dict, into: &cache)
    #expect(n == 8)
}

@Test
func testFormatVersionForeignDict() async throws {
    let mlxTestLock = lockSerializedMLXTest()
    defer { mlxTestLock.unlock() }

    let dict: [String: MLXArray] = [
        "random_key": MLXArray([Int32(0)])
    ]
    #expect(TQDiskSerializer.formatVersion(of: dict) == 0)
    #expect(!TQDiskSerializer.isTQNative(dict))
}

// MARK: - Skip path

@Test
func testRoundTripRotatingKVCache() async throws {
    let mlxTestLock = lockSerializedMLXTest()
    defer { mlxTestLock.unlock() }

    // SLIDING-1 (2026-04-15): RotatingKVCache must round-trip via the
    // new `.rotating` LayerKind. Verifies that ring buffer keys/values,
    // wrap state (offset, idx), and config (keep, maxSize, step) all
    // survive serialize → deserialize → restore.
    let original = RotatingKVCache(maxSize: 32, keep: 4, step: 8)
    let (k, v) = smallKV(seqLen: 12)
    _ = original.update(keys: k, values: v)
    // Run another small update to advance offset past `keep`.
    let (k2, v2) = smallKV(seqLen: 1)
    _ = original.update(keys: k2, values: v2)

    let dict = TQDiskSerializer.serialize(cache: [original])

    // Tag check.
    #expect(dict.keys.contains("__layer_kind_0__"))
    let kind = dict["__layer_kind_0__"]!.item(Int32.self)
    #expect(kind == TQDiskSerializer.LayerKind.rotating.rawValue)

    // Payload + meta keys present.
    #expect(dict.keys.contains("rot_0_keys"))
    #expect(dict.keys.contains("rot_0_values"))
    #expect(dict.keys.contains("__rot_0_meta__"))
    let meta = dict["__rot_0_meta__"]!.asArray(Int32.self)
    #expect(meta.count == 5)
    // metaState[0]=keep, [1]=maxSize, [2]=step, [3]=offset, [4]=idx
    #expect(meta[0] == 4)   // keep
    #expect(meta[1] == 32)  // maxSize
    #expect(meta[2] == 8)   // step
    #expect(meta[3] == 13)  // offset = 12 + 1
    #expect(meta[4] >= 13)  // idx tracks ring head

    // Restore into a fresh layer with the SAME config and verify.
    let restored = RotatingKVCache(maxSize: 32, keep: 4, step: 8)
    var restoreTarget: [any KVCache] = [restored]
    let n = restoreFromDiskArrays(dict, into: &restoreTarget)
    #expect(n == 13)
    // Wrap state survived.
    #expect(restored.offset == 13)
    let restoredMeta = restored.metaState
    #expect(restoredMeta.count == 5)
    #expect(restoredMeta[0] == "4")
    #expect(restoredMeta[1] == "32")
    #expect(restoredMeta[3] == "13")
}

@Test
func testUnknownCacheTypeIsTaggedSkip() async throws {
    let mlxTestLock = lockSerializedMLXTest()
    defer { mlxTestLock.unlock() }

    // Sentinel test for an empty RotatingKVCache (pre-prefill). The
    // serializer should tag it `.skip` since `state.count == 0`.
    // After SLIDING-1 a populated RotatingKVCache lands on `.rotating`
    // (covered by testRoundTripRotatingKVCache).
    let cache: [any KVCache] = [RotatingKVCache(maxSize: 32)]
    // No update — state is empty.
    let dict = TQDiskSerializer.serialize(cache: cache)
    #expect(dict.keys.contains("__layer_kind_0__"))
    let kind = dict["__layer_kind_0__"]!.item(Int32.self)
    #expect(kind == TQDiskSerializer.LayerKind.skip.rawValue)
}

// MARK: - Layer kind: CacheList composite (BaichuanM1, FalconH1)

// Serialize them as a serialized suite so MLXArray operations don't
// race the Metal command-encoder coalescer (same approach as
// DSV4ModelSmokeTests.swift's `.serialized` suite).
@Suite("TQDiskSerializer CacheList", .serialized)
struct TQDiskSerializerCacheListTests {

@Test
func testRoundTripCacheListMambaPlusKV() async throws {
    let mlxTestLock = lockSerializedMLXTest()
    defer { mlxTestLock.unlock() }

    // FalconH1 layout: CacheList(MambaCache, KVCacheSimple). Pre-fix this
    // landed on .skip and dropped multi-turn disk-cache reuse for
    // FalconH1. Verify both sub-caches round-trip through the new
    // .cacheList LayerKind.
    let mamba = MambaCache()
    let mambaArrays: ArraysCache = mamba
    mambaArrays[0] = MLXArray.ones([1, 8, 4], dtype: .float32)
    mambaArrays[1] = MLXArray.ones([1, 8, 16], dtype: .float32) * Float(0.25)
    let kv = KVCacheSimple()
    let (k, v) = smallKV(seqLen: 12)
    _ = kv.update(keys: k, values: v)

    let composite = CacheList(mamba, kv)
    let cache: [any KVCache] = [composite]

    let dict = TQDiskSerializer.serialize(cache: cache)

    // Composite kind tag.
    #expect(dict.keys.contains("__layer_kind_0__"))
    let kind = dict["__layer_kind_0__"]!.item(Int32.self)
    #expect(kind == TQDiskSerializer.LayerKind.cacheList.rawValue)

    // Sub-cache count.
    #expect(dict.keys.contains("__cache_list_0_count__"))
    #expect(dict["__cache_list_0_count__"]!.item(Int32.self) == 2)

    // Per-sub kind tags.
    #expect(dict.keys.contains("__cache_list_0_sub_0_kind__"))
    #expect(dict.keys.contains("__cache_list_0_sub_1_kind__"))
    #expect(dict["__cache_list_0_sub_0_kind__"]!.item(Int32.self)
        == TQDiskSerializer.LayerKind.mamba.rawValue)
    #expect(dict["__cache_list_0_sub_1_kind__"]!.item(Int32.self)
        == TQDiskSerializer.LayerKind.kv.rawValue)

    // Sub-keyed payload keys present.
    #expect(dict.keys.contains("mamba_0_sub_0_state0"))
    #expect(dict.keys.contains("mamba_0_sub_0_state1"))
    #expect(dict.keys.contains("__mamba_0_sub_0_offset__"))
    #expect(dict.keys.contains("kv_0_sub_1_keys"))
    #expect(dict.keys.contains("kv_0_sub_1_values"))

    // No collision with top-level layer-0 keys.
    #expect(!dict.keys.contains("mamba_0_state0"))
    #expect(!dict.keys.contains("kv_0_keys"))

    // Restore round-trip: build a fresh composite and verify both
    // sub-caches received their data.
    let restoredMamba = MambaCache()
    let restoredKV = KVCacheSimple()
    var restored: [any KVCache] = [CacheList(restoredMamba, restoredKV)]
    _ = restoreFromDiskArrays(dict, into: &restored)

    #expect(restoredMamba.state.count == 2)
    #expect(restoredMamba.state[0].size > 0)
    #expect(restoredMamba.state[1].size > 0)
    #expect(restoredKV.state.count == 2)
    #expect(restoredKV.offset == kv.offset)
}

@Test
func testRoundTripCacheListRotatingPlusMamba() async throws {
    let mlxTestLock = lockSerializedMLXTest()
    defer { mlxTestLock.unlock() }

    // BaichuanM1 layout: CacheList(RotatingKVCache, MambaCache). Sub-0
    // is a sliding-window attention; sub-1 is SSM. Verify both
    // round-trip through the new .cacheList LayerKind.
    let rot = RotatingKVCache(maxSize: 32, keep: 0)
    let (k, v) = smallKV(seqLen: 12)
    _ = rot.update(keys: k, values: v)

    let mamba = MambaCache()
    let mambaArrays: ArraysCache = mamba
    mambaArrays[0] = MLXArray.ones([1, 8, 4], dtype: .float32) * Float(0.5)
    mambaArrays[1] = MLXArray.ones([1, 8, 16], dtype: .float32) * Float(0.75)

    let composite = CacheList(rot, mamba)
    let cache: [any KVCache] = [composite]

    let dict = TQDiskSerializer.serialize(cache: cache)

    #expect(dict["__layer_kind_0__"]!.item(Int32.self)
        == TQDiskSerializer.LayerKind.cacheList.rawValue)
    #expect(dict["__cache_list_0_sub_0_kind__"]!.item(Int32.self)
        == TQDiskSerializer.LayerKind.rotating.rawValue)
    #expect(dict["__cache_list_0_sub_1_kind__"]!.item(Int32.self)
        == TQDiskSerializer.LayerKind.mamba.rawValue)

    // Rotating sub-cache: keys, values, and 5-tuple meta.
    #expect(dict.keys.contains("rot_0_sub_0_keys"))
    #expect(dict.keys.contains("rot_0_sub_0_values"))
    #expect(dict.keys.contains("__rot_0_sub_0_meta__"))

    // Mamba sub-cache.
    #expect(dict.keys.contains("mamba_0_sub_1_state0"))
    #expect(dict.keys.contains("mamba_0_sub_1_state1"))

    // Restore.
    let restoredRot = RotatingKVCache(maxSize: 32, keep: 0)
    let restoredMamba = MambaCache()
    var restored: [any KVCache] = [CacheList(restoredRot, restoredMamba)]
    _ = restoreFromDiskArrays(dict, into: &restored)

    #expect(restoredRot.state.count == 2)
    #expect(restoredMamba.state.count == 2)
    #expect(restoredMamba.state[0].size > 0)
}

@Test
func testCacheListEmptyTagsAsSkip() async throws {
    let mlxTestLock = lockSerializedMLXTest()
    defer { mlxTestLock.unlock() }

    // CacheList with no populated sub-caches → composite tags as .skip.
    // Restore is a no-op (no per-sub data was written).
    let composite = CacheList(MambaCache(), KVCacheSimple())
    let cache: [any KVCache] = [composite]
    // Don't update either sub-cache — both have empty state.

    let dict = TQDiskSerializer.serialize(cache: cache)

    #expect(dict.keys.contains("__layer_kind_0__"))
    let kind = dict["__layer_kind_0__"]!.item(Int32.self)
    #expect(kind == TQDiskSerializer.LayerKind.skip.rawValue)
    // Composite count should not have been written when nothing persisted.
    #expect(!dict.keys.contains("__cache_list_0_count__"))
}

@Test
func testCacheListSurvivesAlongsidePlainLayers() async throws {
    let mlxTestLock = lockSerializedMLXTest()
    defer { mlxTestLock.unlock() }

    // Mixed layout: layer 0 = plain KV, layer 1 = CacheList(Mamba, KV),
    // layer 2 = plain Mamba. Verify all three round-trip independently
    // and the kind tags don't collide.
    let plainKV = KVCacheSimple()
    let (k0, v0) = smallKV(seqLen: 6)
    _ = plainKV.update(keys: k0, values: v0)

    let listMamba = MambaCache()
    let listMambaArrays: ArraysCache = listMamba
    listMambaArrays[0] = MLXArray.ones([1, 8, 4], dtype: .float32)
    listMambaArrays[1] = MLXArray.ones([1, 8, 16], dtype: .float32) * Float(0.5)
    let listKV = KVCacheSimple()
    let (k1, v1) = smallKV(seqLen: 8)
    _ = listKV.update(keys: k1, values: v1)
    let composite = CacheList(listMamba, listKV)

    let plainMamba = MambaCache()
    let plainArrays: ArraysCache = plainMamba
    plainArrays[0] = MLXArray.ones([1, 8, 4], dtype: .float32) * Float(0.25)
    plainArrays[1] = MLXArray.ones([1, 8, 16], dtype: .float32) * Float(0.75)

    let cache: [any KVCache] = [plainKV, composite, plainMamba]
    let dict = TQDiskSerializer.serialize(cache: cache)

    #expect(dict["__layer_kind_0__"]!.item(Int32.self)
        == TQDiskSerializer.LayerKind.kv.rawValue)
    #expect(dict["__layer_kind_1__"]!.item(Int32.self)
        == TQDiskSerializer.LayerKind.cacheList.rawValue)
    #expect(dict["__layer_kind_2__"]!.item(Int32.self)
        == TQDiskSerializer.LayerKind.mamba.rawValue)

    // Sub-keyed keys for layer 1's composite must NOT collide with
    // top-level layer 0's `kv_0_*` or layer 2's `mamba_2_*`.
    #expect(dict.keys.contains("kv_0_keys"))
    #expect(dict.keys.contains("kv_1_sub_1_keys"))
    #expect(dict.keys.contains("mamba_2_state0"))
    #expect(dict.keys.contains("mamba_1_sub_0_state0"))

    // Restore into matching topology.
    var restored: [any KVCache] = [
        KVCacheSimple(),
        CacheList(MambaCache(), KVCacheSimple()),
        MambaCache(),
    ]
    let totalTokens = restoreFromDiskArrays(dict, into: &restored)
    #expect(totalTokens > 0)  // attention layers contributed
}

}
