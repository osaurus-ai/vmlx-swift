// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// DeepSeek-V4.1 compressed-KV pools are encoded at 4 bits (packed two codes per
// byte) so 1M context fits; DSV4-0731 stays 8-bit. Runs on the MLX CPU device
// only (no model weights, no GPU).

import Foundation
import MLX
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

@Suite("DSV4 pool code width (4-bit V4.1, 8-bit 0731)")
struct DeepseekV41PoolBitsFocusedTests {

    static func randomPool(rows: Int, features: Int = 512, seed: UInt64 = 1) -> MLXArray {
        var rng = SplitMix64(seed: seed)
        let values = (0..<(rows * features)).map { _ in
            Float(Double(rng.next() % 20_001) / 10_000.0 - 1.0)
        }
        return MLXArray(values, [1, rows, features])
    }

    @Test("nibble packing is an exact inverse")
    func nibbleRoundTrip() {
        Device.withDefaultDevice(.cpu) {
            let codes = MLXArray((0..<64).map { UInt8($0 % 16) }, [1, 2, 1, 32])
            let packed = DeepseekV4PoolStorage.packNibbles(codes)
            #expect(packed.shape == [1, 2, 1, 16])
            #expect(packed.dtype == .uint8)
            let back = DeepseekV4PoolStorage.unpackNibbles(packed)
            #expect(back.asArray(UInt8.self) == codes.asArray(UInt8.self))
        }
    }

    @Test("4-bit segments halve code bytes and stay within half a quantization step")
    func fourBitAccuracyAndSize() {
        Device.withDefaultDevice(.cpu) {
            let pool = Self.randomPool(rows: 64)
            let s8 = DeepseekV4PoolStorage.quantize(pool, bits: 8)
            let s4 = DeepseekV4PoolStorage.quantize(pool, bits: 4)
            #expect(s8.bits == 8 && s8.codes.shape == [1, 64, 16, 32])
            #expect(s4.bits == 4 && s4.codes.shape == [1, 64, 16, 16])
            #expect(s4.codes.nbytes * 2 == s8.codes.nbytes)
            let d4 = DeepseekV4PoolStorage.dequantize(s4)
            let err = MLX.abs(d4 - pool).max().item(Float.self)
            let maxStep = s4.scales.asType(.float32).max().item(Float.self)
            #expect(err <= maxStep * 0.5 + 1e-3, "err \(err) step \(maxStep)")
            let d8 = DeepseekV4PoolStorage.dequantize(s8)
            #expect(MLX.abs(d8 - pool).max().item(Float.self) < err)
        }
    }

    @Test("4-bit storage: tiles, selected-row gathers and copies agree with full dequant")
    func fourBitStorageViews() {
        Device.withDefaultDevice(.cpu) {
            let storage = DeepseekV4PoolStorage(quantizationEnabled: true, hotByteLimit: 0, bits: 4)
            storage.append(Self.randomPool(rows: 100, seed: 2))
            storage.append(Self.randomPool(rows: 37, seed: 3))
            #expect(storage.rowCount == 137)
            #expect(storage.quantizedSegments.allSatisfy { $0.bits == 4 })
            let full = storage.materialized()!
            let idx = MLXArray([Int32(0), 5, 99, 100, 136, 64], [1, 6])
            let gathered = storage.gatherDequantizedRows(idx)
            let expected = take(full, MLXArray([Int32(0), 5, 99, 100, 136, 64]), axis: 1)
            #expect(MLX.abs(gathered - expected).max().item(Float.self) == 0)
            let copy = storage.copy()
            #expect(copy.bits == 4)
            #expect(MLX.abs(copy.materialized()! - full).max().item(Float.self) == 0)
            #expect(storage.retainedByteCount < DeepseekV4PoolStorage(
                quantizationEnabled: true, hotByteLimit: 0, bits: 8).retainedByteCount + 137 * 512)
        }
    }

    @Test("disk serializer round-trips both 4-bit and 8-bit pools and refuses cross-width restores")
    func serializerBothWidths() {
        Device.withDefaultDevice(.cpu) {
            for bits in [4, 8] {
                let src = DeepseekV4Cache(
                    slidingWindow: 16, compressRatio: 2,
                    poolQuantizationEnabled: true, poolBF16HotByteLimit: 0,
                    poolQuantizationBits: bits)
                _ = src.local.update(
                    keys: MLXArray.ones([1, 1, 4, 8]), values: MLXArray.ones([1, 1, 4, 8]))
                _ = src.appendPooled(.compressor, value: Self.randomPool(rows: 70, seed: 9), isPrefill: true)
                let original = src.getPooled(.compressor)!
                let encoded = TQDiskSerializer.serialize(cache: [src])
                #expect(encoded["dsv4_0_keys"] != nil, "bits \(bits) must serialize")

                let dst = DeepseekV4Cache(
                    slidingWindow: 16, compressRatio: 2,
                    poolQuantizationEnabled: true, poolBF16HotByteLimit: 0,
                    poolQuantizationBits: bits)
                var target: [any KVCache] = [dst]
                #expect(restoreFromDiskArrays(encoded, into: &target) == 4)
                #expect(dst.hybridPoolQuantizedSegments(branch: .compressor)?
                    .allSatisfy { $0.bits == bits } == true)
                #expect(MLX.abs(dst.getPooled(.compressor)! - original).max().item(Float.self) == 0)

                let other = DeepseekV4Cache(
                    slidingWindow: 16, compressRatio: 2,
                    poolQuantizationEnabled: true, poolBF16HotByteLimit: 0,
                    poolQuantizationBits: bits == 4 ? 8 : 4)
                var wrong: [any KVCache] = [other]
                #expect(restoreFromDiskArrays(encoded, into: &wrong) == 0, "cross-width must miss")
            }
        }
    }

    @Test("width contract: V4.1 caches are 4-bit, 0731 default stays 8-bit")
    func widthContract() {
        #expect(DeepseekV4Cache.poolQuantizationBits == 8)
        #expect(DeepseekV4Cache.supportedPoolQuantizationBits == HybridPoolQuantizedSchema.supportedBits)
        #expect(DeepseekV4Cache(slidingWindow: 128, compressRatio: 4).hybridPoolQuantizationBits == 8)
        var v41 = DeepseekV4Configuration()
        v41.kvSourceLayerIds = [2]
        #expect(v41.isDeepseekV41)
        let fourBit = DeepseekV4Cache(
            slidingWindow: 128, compressRatio: 2, requiresAuxiliaryTokenHistory: true,
            poolQuantizationBits: 4)
        #expect((fourBit.copy() as! DeepseekV4Cache).hybridPoolQuantizationBits == 4)
        let v4 = DeepseekV4Configuration()
        #expect(!v4.isDeepseekV41)
        let projected4 = DeepseekV4Cache.projectedQuantizedCacheUpperBoundBytes(
            contextLength: 1_000_000, compressRatios: [1], headDim: 512, indexerHeadDim: 128,
            slidingWindow: 128, poolBF16HotByteLimit: 0, poolQuantizationBits: 4)
        let projected8 = DeepseekV4Cache.projectedQuantizedCacheUpperBoundBytes(
            contextLength: 1_000_000, compressRatios: [1], headDim: 512, indexerHeadDim: 128,
            slidingWindow: 128, poolBF16HotByteLimit: 0)
        #expect(projected4 < projected8 * 6 / 10)
    }
}
