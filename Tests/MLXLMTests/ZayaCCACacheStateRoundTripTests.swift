// ZayaCCACache state contract — getter/setter shape, metaState round-trip,
// and copy() deep-clone semantics.

import Foundation
import MLX
@testable import MLXLMCommon
import Testing

@Suite("ZayaCCACache state contract", .serialized)
struct ZayaCCACacheStateRoundTripTests {

    @Test("Empty cache reports zero offset and four-slot state with shapes")
    func emptyCacheShape() {
        let mlxTestLock = lockSerializedMLXTest()
        defer { mlxTestLock.unlock() }

        let c = ZayaCCACache(batchSize: 1, convChannels: 1280, hiddenSize: 2048)
        #expect(c.offset == 0)
        #expect(c.state.count == 4)
        // Sentinels for empty KV (axis-2 dim of 0).
        #expect(c.state[0].dim(2) == 0)
        #expect(c.state[1].dim(2) == 0)
        // Real CCA state, populated to zeros at init.
        #expect(c.state[2].shape == [1, 1280, 2])
        #expect(c.state[3].shape == [1, 2048])
    }

    @Test("State setter accepts four-slot array and round-trips all values")
    func stateRoundTripPreservesAll4() {
        let mlxTestLock = lockSerializedMLXTest()
        defer { mlxTestLock.unlock() }

        let src = ZayaCCACache(batchSize: 1, convChannels: 4, hiddenSize: 8)
        _ = src.update(
            keys: MLXArray.ones([1, 1, 3, 8], dtype: .bfloat16),
            values: MLXArray.ones([1, 1, 3, 8], dtype: .bfloat16) * 2)
        src.writeCCA(
            conv: MLXArray.ones([1, 4, 2], dtype: .float32) * 3,
            prev: MLXArray.ones([1, 8], dtype: .float32) * 4)

        let snapshot = src.state
        #expect(snapshot.count == 4)
        #expect(snapshot[0].shape == [1, 1, 3, 8])
        #expect(snapshot[2].shape == [1, 4, 2])

        let dst = ZayaCCACache(batchSize: 1, convChannels: 4, hiddenSize: 8)
        dst.state = snapshot

        #expect(dst.offset == src.offset)
        let (dConv, dPrev) = dst.readCCA()
        #expect(dConv.shape == [1, 4, 2])
        #expect(dPrev.shape == [1, 8])
        let convDelta = (dConv - src.readCCA().conv).abs().sum().item(Float.self)
        let prevDelta = (dPrev - src.readCCA().prev).abs().sum().item(Float.self)
        #expect(convDelta < 1e-3)
        #expect(prevDelta < 1e-3)
    }

    @Test("metaState round-trips kv tag plus ZAYA trailer")
    func metaStateRoundTrip() {
        let mlxTestLock = lockSerializedMLXTest()
        defer { mlxTestLock.unlock() }

        let src = ZayaCCACache(batchSize: 1, convChannels: 4, hiddenSize: 8)
        let meta = src.metaState
        // Trailer: "zaya_cca_v1", convChannels, hiddenSize, batchSize
        #expect(meta.contains("zaya_cca_v1"))
        #expect(meta.contains("4"))
        #expect(meta.contains("8"))
        #expect(meta.contains("1"))

        let dst = ZayaCCACache(batchSize: 1, convChannels: 4, hiddenSize: 8)
        dst.metaState = meta
        // Setter must accept the legitimate trailer without throwing.
        #expect(dst.metaState.contains("zaya_cca_v1"))
    }

    @Test("copy() produces a deep clone — mutating src does not touch dst")
    func copyDoesNotShareState() {
        let mlxTestLock = lockSerializedMLXTest()
        defer { mlxTestLock.unlock() }

        let src = ZayaCCACache(batchSize: 1, convChannels: 4, hiddenSize: 8)
        src.writeCCA(
            conv: MLXArray.ones([1, 4, 2], dtype: .float32) * 3,
            prev: MLXArray.ones([1, 8], dtype: .float32) * 4)

        let cp = src.copy() as! ZayaCCACache

        // Mutate src after the snapshot; cp must be unaffected.
        src.writeCCA(
            conv: MLXArray.zeros([1, 4, 2], dtype: .float32),
            prev: MLXArray.zeros([1, 8], dtype: .float32))

        let (cConv, cPrev) = cp.readCCA()
        #expect((cConv.sum().item(Float.self) - 24.0).magnitude < 1e-3)  // 4*2*3 = 24
        #expect((cPrev.sum().item(Float.self) - 32.0).magnitude < 1e-3)  // 8*4 = 32
    }

    @Test("update() returns the full accumulated keys/values")
    func updateReturnsFullSlice() {
        let mlxTestLock = lockSerializedMLXTest()
        defer { mlxTestLock.unlock() }

        let c = ZayaCCACache(batchSize: 1, convChannels: 4, hiddenSize: 8)
        let (k1, v1) = c.update(
            keys: MLXArray.ones([1, 1, 2, 8], dtype: .bfloat16),
            values: MLXArray.ones([1, 1, 2, 8], dtype: .bfloat16) * 2)
        #expect(k1.dim(2) == 2)
        #expect(c.offset == 2)

        let (k2, _) = c.update(
            keys: MLXArray.ones([1, 1, 3, 8], dtype: .bfloat16) * 3,
            values: MLXArray.ones([1, 1, 3, 8], dtype: .bfloat16) * 4)
        #expect(k2.dim(2) == 5)         // 2 + 3
        #expect(c.offset == 5)
        // The first two timesteps stayed value 1, the next three are value 3.
        let firstSum = k2[.ellipsis, 0..<2, 0...].sum().item(Float.self)
        let lastSum = k2[.ellipsis, 2..<5, 0...].sum().item(Float.self)
        #expect((firstSum - Float(1 * 1 * 2 * 8)).magnitude < 1e-3)
        #expect((lastSum - Float(3 * 1 * 3 * 8)).magnitude < 1e-3)
        _ = v1
    }

    @Test("TurboQuant promotion encodes only attention KV and preserves native CCA state")
    func turboQuantPromotionPreservesCCAState() {
        let mlxTestLock = lockSerializedMLXTest()
        defer { mlxTestLock.unlock() }

        let cache = ZayaCCACache(batchSize: 1, convChannels: 4, hiddenSize: 8)
        _ = cache.update(
            keys: MLXArray.ones([1, 1, 96, 32], dtype: .bfloat16),
            values: MLXArray.ones([1, 1, 96, 32], dtype: .bfloat16) * 2)
        let conv = MLXArray.ones([1, 4, 2], dtype: .float32) * 3
        let prev = MLXArray.ones([1, 8], dtype: .float32) * 4
        cache.writeCCA(conv: conv, prev: prev)

        #expect(cache.promoteAttentionKVToTurboQuant(keyBits: 3, valueBits: 3))
        #expect(cache.usesTurboQuantKV)
        #expect(cache.turboQuantKVCache?.offset == 96)
        #expect(cache.turboQuantKVCache?.compressedKeys != nil)
        #expect(cache.turboQuantKVCache?.compressedValues != nil)
        #expect(containsUnprovenZayaTurboQuantDiskState([cache]))
        #expect(cache.readCCA().conv.dtype == .float32)
        #expect(cache.readCCA().prev.dtype == .float32)
        #expect((cache.readCCA().conv - conv).abs().sum().item(Float.self) < 1e-3)
        #expect((cache.readCCA().prev - prev).abs().sum().item(Float.self) < 1e-3)

        let (keys, _) = cache.update(
            keys: MLXArray.ones([1, 1, 1, 32], dtype: .bfloat16) * 5,
            values: MLXArray.ones([1, 1, 1, 32], dtype: .bfloat16) * 6)
        #expect(cache.offset == 97)
        #expect(keys.dim(2) == 97)
        #expect(cache.usesTurboQuantKV)

        let fourBit = ZayaCCACache(batchSize: 1, convChannels: 4, hiddenSize: 8)
        _ = fourBit.update(
            keys: MLXArray.ones([1, 1, 96, 32], dtype: .bfloat16),
            values: MLXArray.ones([1, 1, 96, 32], dtype: .bfloat16))
        #expect(fourBit.promoteAttentionKVToTurboQuant(keyBits: 4, valueBits: 4))
        #expect(!containsUnprovenZayaTurboQuantDiskState([fourBit]))
    }

    @Test("generic TQ conversion ignores ZAYA MoE placeholders")
    func genericTurboQuantConversionTargetsZayaAttentionOnly() {
        let mlxTestLock = lockSerializedMLXTest()
        defer { mlxTestLock.unlock() }

        let zaya = ZayaCCACache(batchSize: 1, convChannels: 4, hiddenSize: 8)
        _ = zaya.update(
            keys: MLXArray.ones([1, 1, 96, 32], dtype: .bfloat16),
            values: MLXArray.ones([1, 1, 96, 32], dtype: .bfloat16))
        var cache: [any KVCache] = [zaya, ZayaMoEPlaceholderCache()]

        let before = ModelCacheTopologySnapshot(cache: cache)
        maybeQuantizeKVCache(
            cache: &cache,
            kvBits: nil,
            kvMode: .turboQuant(keyBits: 3, valueBits: 3))
        let after = ModelCacheTopologySnapshot(cache: cache)

        #expect(before.layerCount == 2)
        #expect(before.kvLayerCount == 1)
        #expect(before.zayaCCALayerCount == 1)
        #expect(before.turboQuantKVLayerCount == 0)
        #expect(after.kvLayerCount == 0)
        #expect(after.zayaCCALayerCount == 1)
        #expect(after.turboQuantKVLayerCount == 1)
        #expect(cache[1] is ZayaMoEPlaceholderCache)
        #expect(!(cache[1] is TurboQuantKVCache))
    }

    @Test("disk prompt policy requires typed ZAYA and at least four TQ bits")
    func selectivePromptDiskPolicyIsZayaOnly() {
        let mlxTestLock = lockSerializedMLXTest()
        defer { mlxTestLock.unlock() }

        let requested = KVQuantizationMode.turboQuant(keyBits: 4, valueBits: 4)
        let subFourBit = KVQuantizationMode.turboQuant(keyBits: 3, valueBits: 3)
        let zaya: [any KVCache] = [
            ZayaCCACache(batchSize: 1, convChannels: 4, hiddenSize: 8),
            ZayaMoEPlaceholderCache(),
        ]
        let dense: [any KVCache] = [KVCacheSimple()]

        #expect(
            selectivePromptBoundaryDiskKVMode(cache: zaya, requested: requested)
                == requested)
        #expect(
            selectivePromptBoundaryDiskKVMode(cache: dense, requested: requested)
                == .none)
        #expect(
            selectivePromptBoundaryDiskKVMode(cache: zaya, requested: subFourBit)
                == .none)
        #expect(
            selectivePromptBoundaryDiskKVMode(cache: zaya, requested: .none)
                == .none)
        #expect(
            selectivePromptBoundaryDiskKVMode(
                cache: zaya,
                requested: .affine(bits: 4, groupSize: 64))
                == .none)
    }
}
