// ZayaCCACache disk round-trip — TQDiskSerializer.serialize emits
// LayerKind.zayaCCA + 4 state arrays + 4-element meta, and
// restoreFromDiskArrays restores all four arrays byte-identical.

import Foundation
import MLX
@testable import MLXLMCommon
import Testing

@Suite("ZayaCCACache disk round-trip", .serialized)
struct ZayaCCACacheDiskRoundTripTests {

    @Test("LayerKind.zayaCCA exists with rawValue 9")
    func layerKindEnumIsRegistered() {
        let mlxTestLock = lockSerializedMLXTest()
        defer { mlxTestLock.unlock() }

        #expect(TQDiskSerializer.LayerKind.zayaCCA.rawValue == 9)
    }

    @Test("serialize emits zaya_{i}_{keys,values,conv_state,prev_hs} + meta + kind tag")
    func serializeEmitsAllFourArrays() {
        let mlxTestLock = lockSerializedMLXTest()
        defer { mlxTestLock.unlock() }

        let z = ZayaCCACache(batchSize: 1, convChannels: 4, hiddenSize: 8)
        _ = z.update(
            keys: MLXArray.ones([1, 1, 3, 8], dtype: .bfloat16),
            values: MLXArray.ones([1, 1, 3, 8], dtype: .bfloat16) * 2)
        z.writeCCA(
            conv: MLXArray.ones([1, 4, 2], dtype: .float32) * 3,
            prev: MLXArray.ones([1, 8], dtype: .float32) * 4)

        let encoded = TQDiskSerializer.serialize(cache: [z])

        #expect(encoded["zaya_0_keys"] != nil)
        #expect(encoded["zaya_0_values"] != nil)
        #expect(encoded["zaya_0_conv_state"] != nil)
        #expect(encoded["zaya_0_prev_hs"] != nil)
        #expect(encoded["__zaya_0_meta__"] != nil)
        #expect(encoded["__layer_kind_0__"] != nil)
        // Meta tuple shape sanity check.
        let meta = encoded["__zaya_0_meta__"]!.asArray(Int32.self)
        #expect(meta.count == 4)
        #expect(meta[0] == 3)   // offset
        #expect(meta[1] == 4)   // convChannels
        #expect(meta[2] == 8)   // hiddenSize
        #expect(meta[3] == 1)   // batchSize
    }

    @Test("Round-trip preserves all 4 state arrays byte-identical with offset")
    func roundTripPreservesByteIdentity() {
        let mlxTestLock = lockSerializedMLXTest()
        defer { mlxTestLock.unlock() }

        let src = ZayaCCACache(batchSize: 1, convChannels: 4, hiddenSize: 8)
        _ = src.update(
            keys: MLXArray.ones([1, 1, 3, 8], dtype: .bfloat16),
            values: MLXArray.ones([1, 1, 3, 8], dtype: .bfloat16) * 2)
        src.writeCCA(
            conv: MLXArray.ones([1, 4, 2], dtype: .float32) * 3,
            prev: MLXArray.ones([1, 8], dtype: .float32) * 4)
        let originalOffset = src.offset

        let encoded = TQDiskSerializer.serialize(cache: [src])

        let dst = ZayaCCACache(batchSize: 1, convChannels: 4, hiddenSize: 8)
        var restoreTarget: [any KVCache] = [dst]
        let restored = restoreFromDiskArrays(encoded, into: &restoreTarget)
        #expect(restored == originalOffset)
        #expect(dst.offset == originalOffset)

        // Use simple delta-norms to keep this dtype-agnostic
        // (.bfloat16 keys, .float32 conv state, .float32 prev_hs).
        let kDelta = (dst.state[0].asType(.float32) - src.state[0].asType(.float32))
            .abs().sum().item(Float.self)
        let vDelta = (dst.state[1].asType(.float32) - src.state[1].asType(.float32))
            .abs().sum().item(Float.self)
        let cDelta = (dst.readCCA().conv - src.readCCA().conv)
            .abs().sum().item(Float.self)
        let pDelta = (dst.readCCA().prev - src.readCCA().prev)
            .abs().sum().item(Float.self)
        #expect(kDelta < 1e-2)
        #expect(vDelta < 1e-2)
        #expect(cDelta < 1e-3)
        #expect(pDelta < 1e-3)
    }

    @Test("Corrupted CCA disk metadata misses without mutating live cache")
    func corruptedMetadataMissesWithoutMutation() {
        let mlxTestLock = lockSerializedMLXTest()
        defer { mlxTestLock.unlock() }

        let src = ZayaCCACache(batchSize: 1, convChannels: 4, hiddenSize: 8)
        _ = src.update(
            keys: MLXArray.ones([1, 1, 3, 8], dtype: .bfloat16),
            values: MLXArray.ones([1, 1, 3, 8], dtype: .bfloat16) * 2)
        src.writeCCA(
            conv: MLXArray.ones([1, 4, 2], dtype: .float32) * 3,
            prev: MLXArray.ones([1, 8], dtype: .float32) * 4)

        var encoded = TQDiskSerializer.serialize(cache: [src])
        encoded["__zaya_0_meta__"] = MLXArray([Int32(src.offset), 4, 16, 1])

        let dst = ZayaCCACache(batchSize: 1, convChannels: 4, hiddenSize: 8)
        let before = dst.readCCA()
        var restoreTarget: [any KVCache] = [dst]

        let restored = restoreFromDiskArrays(encoded, into: &restoreTarget)

        #expect(restored == 0)
        #expect(dst.offset == 0)
        let convDelta = (dst.readCCA().conv - before.conv).abs().sum().item(Float.self)
        let prevDelta = (dst.readCCA().prev - before.prev).abs().sum().item(Float.self)
        #expect(convDelta < 1e-6)
        #expect(prevDelta < 1e-6)
    }

    @Test("Empty source cache round-trips cleanly via zero-seq sentinel")
    func emptyKVRoundTrip() {
        let mlxTestLock = lockSerializedMLXTest()
        defer { mlxTestLock.unlock() }

        // Source has no prefill — only the always-populated CCA state.
        let src = ZayaCCACache(batchSize: 1, convChannels: 4, hiddenSize: 8)
        src.writeCCA(
            conv: MLXArray.ones([1, 4, 2], dtype: .float32) * 7,
            prev: MLXArray.ones([1, 8], dtype: .float32) * 11)
        let encoded = TQDiskSerializer.serialize(cache: [src])
        // Sentinel keys must be present (axis-2 dim 0).
        #expect(encoded["zaya_0_keys"] != nil)
        #expect(encoded["zaya_0_keys"]!.dim(2) == 0)

        let dst = ZayaCCACache(batchSize: 1, convChannels: 4, hiddenSize: 8)
        var restoreTarget: [any KVCache] = [dst]
        _ = restoreFromDiskArrays(encoded, into: &restoreTarget)
        // Restored cache must report offset 0 (KV stayed empty) but CCA state populated.
        #expect(dst.offset == 0)
        let cDelta = (dst.readCCA().conv - src.readCCA().conv).abs().sum().item(Float.self)
        let pDelta = (dst.readCCA().prev - src.readCCA().prev).abs().sum().item(Float.self)
        #expect(cDelta < 1e-3)
        #expect(pDelta < 1e-3)
    }

    @Test("Mixed per-layer round-trip: KVCacheSimple + ZayaCCACache + RotatingKVCache")
    func mixedPerLayerArrayRoundTrip() {
        let mlxTestLock = lockSerializedMLXTest()
        defer { mlxTestLock.unlock() }

        let l0 = KVCacheSimple()
        _ = l0.update(
            keys: MLXArray.ones([1, 1, 2, 8], dtype: .bfloat16),
            values: MLXArray.ones([1, 1, 2, 8], dtype: .bfloat16) * 2)

        let l1 = ZayaCCACache(batchSize: 1, convChannels: 4, hiddenSize: 8)
        _ = l1.update(
            keys: MLXArray.ones([1, 1, 3, 8], dtype: .bfloat16) * 5,
            values: MLXArray.ones([1, 1, 3, 8], dtype: .bfloat16) * 6)
        l1.writeCCA(
            conv: MLXArray.ones([1, 4, 2], dtype: .float32) * 7,
            prev: MLXArray.ones([1, 8], dtype: .float32) * 8)

        let l2 = RotatingKVCache(maxSize: 16, keep: 0)
        _ = l2.update(
            keys: MLXArray.ones([1, 1, 4, 8], dtype: .bfloat16) * 9,
            values: MLXArray.ones([1, 1, 4, 8], dtype: .bfloat16) * 10)

        let caches: [any KVCache] = [l0, l1, l2]
        let encoded = TQDiskSerializer.serialize(cache: caches)

        let r0 = KVCacheSimple()
        let r1 = ZayaCCACache(batchSize: 1, convChannels: 4, hiddenSize: 8)
        let r2 = RotatingKVCache(maxSize: 16, keep: 0)
        var restored: [any KVCache] = [r0, r1, r2]
        _ = restoreFromDiskArrays(encoded, into: &restored)

        #expect(r0.offset == l0.offset)
        #expect(r1.offset == l1.offset)
        #expect(r2.offset == l2.offset)
        let zayaDelta = (r1.readCCA().conv - l1.readCCA().conv).abs().sum().item(Float.self)
        #expect(zayaDelta < 1e-3)
    }
}
