// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// DeepSeek-V4.1 keeps each pool as one contiguous 4-bit slab, and its
// compress-ratio-1 layers hold one row per token. The disk reader bounds a
// pool segment at 16K rows, so a record past 16K tokens has to be written in
// bounded slabs or it can never be restored. Runs on the MLX CPU device only
// (no model weights, no GPU).

import Foundation
import MLX
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

@Suite("DSV4.1 contiguous pools past 16K rows round-trip through the disk record")
struct DeepseekV41LongPoolDiskRestoreFocusedTests {

    static func pool(rows: Int, features: Int = 64) -> MLXArray {
        MLX.sin(MLXArray(Int32(0) ..< Int32(rows * features)).asType(.float32) * 0.37)
            .reshaped([1, rows, features])
    }

    static func contiguousCache() -> DeepseekV4Cache {
        DeepseekV4Cache(
            slidingWindow: 16, compressRatio: 1,
            poolQuantizationEnabled: true, poolBF16HotByteLimit: 0,
            poolQuantizationBits: 4, contiguousPools: true)
    }

    @Test("a ratio-1 pool longer than one segment restores exactly", arguments: [
        HybridPoolQuantizedSchema.maxSegmentRows,
        HybridPoolQuantizedSchema.maxSegmentRows + 1,
        2 * HybridPoolQuantizedSchema.maxSegmentRows + 277,
    ])
    func longPoolRoundTrip(rows: Int) {
        Device.withDefaultDevice(.cpu) {
            let src = Self.contiguousCache()
            _ = src.local.update(
                keys: MLXArray.ones([1, 1, 4, 8]), values: MLXArray.ones([1, 1, 4, 8]))
            _ = src.appendPooled(.compressor, value: Self.pool(rows: rows), isPrefill: true)
            _ = src.appendPooled(.indexer, value: Self.pool(rows: rows), isPrefill: true)
            #expect(src.hybridPoolQuantizedSegments(branch: .compressor)?.count == 1)
            let compressor = src.getPooled(.compressor)!
            let indexer = src.getPooled(.indexer)!

            let encoded = TQDiskSerializer.serialize(cache: [src])
            let maxRows = HybridPoolQuantizedSchema.maxSegmentRows
            let expectedSegments = (rows + maxRows - 1) / maxRows
            #expect(
                encoded["__dsv4_0_pool_comp_qcount__"]?.asArray(Int32.self)
                    == [Int32(expectedSegments)])

            let dst = Self.contiguousCache()
            var target: [any KVCache] = [dst]
            #expect(restoreFromDiskArrays(encoded, into: &target, requirePromptBoundary: true) == 4)
            #expect(dst.hybridPoolQuantizedSegments(branch: .compressor)?.count == 1)
            #expect(dst.hybridPoolQuantizedSegments(branch: .compressor)?.first?.rowCount == rows)
            #expect(dst.getPooled(.compressor).map {
                MLX.abs($0 - compressor).max().item(Float.self) } == 0)
            #expect(dst.getPooled(.indexer).map {
                MLX.abs($0 - indexer).max().item(Float.self) } == 0)
        }
    }

    @Test("pool storage compaction and the disk reader share one segment bound")
    func sharedBound() {
        #expect(DeepseekV4PoolStorage.slabMaxRows == HybridPoolQuantizedSchema.maxSegmentRows)
    }
}
