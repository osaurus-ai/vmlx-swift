// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// L2 disk round-trip tests for DSV4 per-layer cache types.
//
// Verifies (post-2026-05-04 pure-long-context pass):
//   - plain `RotatingKVCache` (cr=0 layers only) encodes via
//     `TQDiskSerializer` and decodes via `restoreRotatingLayer`
//   - `DeepseekV4Cache` (every cr>0 layer — there is no fallback
//     anymore) conforms to `RotatingKVCacheWrapper` so its inner
//     rotating state round-trips, AND its compressor + indexer pool
//     tensors plus per-branch incomplete-window buffers ROUND-TRIP
//     through `state` / `metaState` so multi-turn prefix-cache reuse
//     doesn't have to re-derive the pool from prompt tokens every turn.
//   - A mixed per-layer array of both types encodes and restores
//     without kind-tag drift.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import Testing

@Suite("DSV4 L2 disk round-trip", .serialized)
struct DeepseekV4CacheDiskRoundTripTests {

    /// Fill a RotatingKVCache with two (keys, values) steps so its
    /// state has shape-preserving content for the roundtrip check.
    static func fillRotating(
        _ rot: RotatingKVCache,
        B: Int = 1, H: Int = 1, headDim: Int = 8
    ) {
        let step1Keys = MLXArray.ones([B, H, 3, headDim])
        let step1Vals = MLXArray.ones([B, H, 3, headDim]) * 2.0
        _ = rot.update(keys: step1Keys, values: step1Vals)
        let step2Keys = MLXArray.ones([B, H, 2, headDim]) * 3.0
        let step2Vals = MLXArray.ones([B, H, 2, headDim]) * 4.0
        _ = rot.update(keys: step2Keys, values: step2Vals)
    }

    @Test("RotatingKVCache (default DSV4 path) disk round-trips")
    func rotatingRoundTrip() {
        let rot = RotatingKVCache(maxSize: 16, keep: 0)
        Self.fillRotating(rot)
        let originalState = rot.state
        let originalMeta = rot.metaState
        let originalOffset = rot.offset

        // Encode via TQDiskSerializer.
        let encoded = TQDiskSerializer.serialize(cache: [rot])
        #expect(encoded["__layer_kind_0__"] != nil,
            "encode must tag layer 0 kind")
        #expect(encoded["rot_0_keys"] != nil)
        #expect(encoded["rot_0_values"] != nil)
        #expect(encoded["__rot_0_meta__"] != nil)

        // Decode into a fresh cache via restoreFromDiskArrays.
        let target = RotatingKVCache(maxSize: 16, keep: 0)
        var restoreTarget: [any KVCache] = [target]
        _ = restoreFromDiskArrays(encoded, into: &restoreTarget)
        #expect(target.state.count == originalState.count)
        #expect(target.metaState == originalMeta)
        #expect(target.offset == originalOffset,
            "offset must survive disk round-trip")
    }

    @Test("DeepseekV4Cache disk round-trip: rotating + pool + buffers all survive")
    func deepseekV4CacheRoundTrip() {
        let v4 = DeepseekV4Cache(slidingWindow: 16, compressRatio: 4)
        Self.fillRotating(v4.local)
        // Populate pool + per-branch buffer state. New (post-2026-05-04)
        // contract: ALL of this round-trips so multi-turn chat doesn't
        // re-derive the pool from prompt tokens every turn.
        let pool = MLXArray.ones([1, 5, 8]) * 7.0
        let bufKV = MLXArray.ones([1, 3, 8])
        let bufGate = MLXArray.ones([1, 3, 8]) * 2.0
        v4.setPooled(.compressor, value: pool)
        v4.setBuffers(.compressor, kv: bufKV, gate: bufGate)
        v4.setPooled(.indexer, value: pool * 3.0)

        let originalOffset = v4.offset

        // Encode — DSV4 path now uses dedicated `dsv4_*` keys (not the
        // `rot_*` rotating-only keys) so the pool tensors can round-trip.
        let encoded = TQDiskSerializer.serialize(cache: [v4])
        #expect(encoded["dsv4_0_keys"] != nil,
            "DeepseekV4Cache must serialize via the dsv4 layer kind")
        #expect(encoded["dsv4_0_values"] != nil)
        #expect(encoded["__dsv4_0_meta__"] != nil,
            "dsv4 layer must persist 7-element meta tuple")
        #expect(encoded["dsv4_0_pool_comp"] != nil,
            "compressor pool must be in the encoded dict")
        #expect(encoded["dsv4_0_pool_idx"] != nil,
            "indexer pool must be in the encoded dict")

        // Decode into a fresh v4 cache.
        let target = DeepseekV4Cache(slidingWindow: 16, compressRatio: 4)
        var restoreTarget: [any KVCache] = [target]
        _ = restoreFromDiskArrays(encoded, into: &restoreTarget)
        #expect(target.offset == originalOffset,
            "inner offset must survive round-trip")
        // Pool state survives (the new contract).
        let restoredPool = target.getPooled(.compressor)
        #expect(restoredPool != nil,
            "compressor pool must survive disk round-trip (multi-turn prefix-cache reuse)")
        let (rbufKV, rbufGate) = target.getBuffers(.compressor)
        #expect(rbufKV != nil && rbufGate != nil,
            "incomplete-window buffer state must survive disk round-trip")
        #expect(target.getPooled(.indexer) != nil,
            "indexer pool must survive disk round-trip")
    }

    @Test("Mixed per-layer array: RotatingKVCache + DeepseekV4Cache round-trip together")
    func mixedPerLayerRoundTrip() {
        let layer0 = RotatingKVCache(maxSize: 16, keep: 0)
        Self.fillRotating(layer0)
        let layer1 = DeepseekV4Cache(slidingWindow: 16, compressRatio: 4)
        Self.fillRotating(layer1.local)

        let caches: [any KVCache] = [layer0, layer1]
        let encoded = TQDiskSerializer.serialize(cache: caches)
        #expect(encoded["rot_0_keys"] != nil,
            "layer 0 (plain RotatingKVCache) keeps the rot_* keys")
        #expect(encoded["dsv4_1_keys"] != nil,
            "layer 1 (DeepseekV4Cache) goes through the dsv4_* keys")

        var target: [any KVCache] = [
            RotatingKVCache(maxSize: 16, keep: 0),
            DeepseekV4Cache(slidingWindow: 16, compressRatio: 4),
        ]
        _ = restoreFromDiskArrays(encoded, into: &target)
        #expect(target[0].offset == layer0.offset)
        #expect(target[1].offset == layer1.offset)
    }

    @Test("DeepseekV4Cache conforms to RotatingKVCacheWrapper protocol")
    func wrapperProtocolConformance() {
        let v4 = DeepseekV4Cache(slidingWindow: 16, compressRatio: 4)
        let wrapper: RotatingKVCacheWrapper? = v4
        #expect(wrapper != nil,
            "DeepseekV4Cache must conform to RotatingKVCacheWrapper")
        #expect(wrapper?.rotating === v4.local,
            "wrapper.rotating must return the exact inner RotatingKVCache")
    }

    @Test("Gemma-style mixed rotating/simple cache disk round-trips tensor state")
    func gemmaStyleMixedRotatingSimpleRoundTripPreservesTensorState() {
        let mlxTestLock = lockSerializedMLXTest()
        defer { mlxTestLock.unlock() }

        let sliding = RotatingKVCache(maxSize: 16, keep: 0)
        Self.fillRotating(sliding)

        let full = KVCacheSimple()
        let fullKeys = MLXArray.ones([1, 1, 5, 8]) * 5.0
        let fullValues = MLXArray.ones([1, 1, 5, 8]) * 6.0
        full.state = [fullKeys, fullValues]

        let sink = RotatingKVCache(maxSize: 32, keep: 4)
        Self.fillRotating(sink)

        let caches: [any KVCache] = [sliding, full, sink]
        eval(caches.flatMap(\.state))
        let encoded = TQDiskSerializer.serialize(cache: caches)

        var restored: [any KVCache] = [
            RotatingKVCache(maxSize: 16, keep: 0),
            KVCacheSimple(),
            RotatingKVCache(maxSize: 32, keep: 4),
        ]
        let restoredTokens = restoreFromDiskArrays(encoded, into: &restored)
        eval(restored.flatMap(\.state))
        #expect(restoredTokens == sliding.offset)

        guard let restoredSliding = restored[0] as? RotatingKVCache,
              let restoredFull = restored[1] as? KVCacheSimple,
              let restoredSink = restored[2] as? RotatingKVCache
        else {
            Issue.record("Restored cache types changed unexpectedly")
            return
        }

        assertStateEqual(restoredSliding.state, sliding.state, label: "sliding")
        #expect(restoredSliding.metaState == sliding.metaState)
        assertStateEqual(restoredFull.state, full.state, label: "full")
        assertStateEqual(restoredSink.state, sink.state, label: "sink")
        #expect(restoredSink.metaState == sink.metaState)
    }

    @Test("short-prompt DSV4 cache with an empty pool round-trips restorably")
    func emptyPoolRoundTripsRestorably() {
        // A prompt shorter than one compress window leaves the pool with
        // zero rows. That checkpoint must round-trip: the canonical
        // on-disk form of an empty pool is absence, and the restore
        // validator accepts a nil pool. Writing the zero-row tensor made
        // every short-prompt DSV4 checkpoint permanently unrestorable.
        let v4 = DeepseekV4Cache(slidingWindow: 16, compressRatio: 4)
        Self.fillRotating(v4.local)
        v4.setPooled(.compressor, value: MLXArray.zeros([1, 0, 8]))
        let originalOffset = v4.offset

        let encoded = TQDiskSerializer.serialize(cache: [v4])
        let target = DeepseekV4Cache(slidingWindow: 16, compressRatio: 4)
        var restoreTarget: [any KVCache] = [target]
        let restored = restoreFromDiskArrays(encoded, into: &restoreTarget)

        #expect(restored > 0, "empty-pool DSV4 record must restore, not miss")
        #expect(target.offset == originalOffset)
    }

    @Test("qkv group-size mismatch refuses the whole record atomically")
    func qkvMismatchRefusesWholeRecord() {
        // A quantized-KV layer whose stored group size / bit width no
        // longer matches the runtime cache must refuse the entire record
        // BEFORE any sibling layer is seated: `totalTokens` seeds from
        // sibling layers, so a per-layer skip reported a "hit" with this
        // layer left empty.
        let plain = KVCacheSimple()
        let keys = MLXArray.ones([1, 1, 6, 64])
        let values = MLXArray.ones([1, 1, 6, 64]) * 2.0
        _ = plain.update(keys: keys, values: values)
        let qkv = QuantizedKVCache(groupSize: 64, bits: 8)
        _ = qkv.updateQuantized(keys: keys, values: values)

        let encoded = TQDiskSerializer.serialize(cache: [plain, qkv])

        let targetPlain = KVCacheSimple()
        let targetQKV = QuantizedKVCache(groupSize: 32, bits: 8)
        var restoreTarget: [any KVCache] = [targetPlain, targetQKV]
        let restored = restoreFromDiskArrays(encoded, into: &restoreTarget)

        #expect(restored == 0, "mismatched qkv layer must be an atomic whole-record miss")
        #expect(targetPlain.offset == 0, "no sibling layer may be seated before the refusal")

        // Control: a matching target restores normally.
        let okPlain = KVCacheSimple()
        let okQKV = QuantizedKVCache(groupSize: 64, bits: 8)
        var okTarget: [any KVCache] = [okPlain, okQKV]
        let okRestored = restoreFromDiskArrays(encoded, into: &okTarget)
        #expect(okRestored == 6)
        #expect(okQKV.offset == 6)
    }

    @Test("non-q8 quantized pool segment poisons the record as an atomic miss")
    func nonQ8QuantizedPoolPoisonsRecord() {
        // The disk schema for quantized pools is defined for 8-bit affine
        // segments. Any other width must not degrade to a silently
        // skipped layer (sibling .deepseekV4 layers would restore while
        // this one stayed empty) — the record poisons to a required miss.
        let v4 = DeepseekV4Cache(slidingWindow: 16, compressRatio: 4)
        Self.fillRotating(v4.local)
        let bogus = HybridPoolQuantizedSegment(
            codes: MLXArray.zeros([1, 64, 8], dtype: .uint32),
            scales: MLXArray.zeros([1, 64, 2]),
            biases: MLXArray.zeros([1, 64, 2]),
            originalShape: [1, 64, 8],
            groupSize: 32,
            bits: 4,
            originalDType: .float16)
        v4.setHybridPoolQuantizedSegments(branch: .compressor, segments: [bogus])

        let encoded = TQDiskSerializer.serialize(cache: [v4])
        #expect(encoded["dsv4_0_keys"] == nil,
            "poisoned record must not carry restorable keys")

        let target = DeepseekV4Cache(slidingWindow: 16, compressRatio: 4)
        var restoreTarget: [any KVCache] = [target]
        let restored = restoreFromDiskArrays(encoded, into: &restoreTarget)
        #expect(restored == 0, "non-q8 pool record must be an atomic miss")
        #expect(target.offset == 0)
    }

    @Test("serializer q8 contract matches DeepseekV4PoolStorage bits")
    func serializerBitsMatchPoolStorage() {
        // The serializer accepts exactly 8-bit pool segments; the pool
        // storage produces exactly 8-bit segments. Tie the two constants
        // so one cannot drift without this test going red.
        #expect(DeepseekV4Cache.poolQuantizationBits == 8)
    }

    private func assertStateEqual(
        _ actual: [MLXArray],
        _ expected: [MLXArray],
        label: String,
        sourceLocation: SourceLocation = #_sourceLocation
    ) {
        #expect(actual.count == expected.count, "\(label) state array count", sourceLocation: sourceLocation)
        guard actual.count == expected.count else { return }
        for (index, pair) in zip(actual, expected).enumerated() {
            #expect(pair.0.shape == pair.1.shape,
                "\(label) state \(index) shape", sourceLocation: sourceLocation)
            #expect(allClose(pair.0, pair.1).item(Bool.self),
                "\(label) state \(index) values", sourceLocation: sourceLocation)
        }
    }
}
