// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// DSV4 compressor/indexer POOL + BUFFER survival across a partial disk restore.
//
// `DeepseekV4PartialRestoreContinuationTests` drives the cache through
// `update(keys:values:)`, which only forwards to the inner rotating window:
//
//     public func update(keys:values:) -> (MLXArray, MLXArray) {
//         local.update(keys: keys, values: values)
//     }
//
// The compressor/indexer pool is fed by the attention module through a separate
// API (`appendPooled` / `setBuffers`). So that suite left both pools nil in the
// cold AND warm caches and compared empty to empty — it never proved the pool
// survives a restore. The pool is the whole-history-dependent state, so that is
// precisely where a "fluent but cannot converge" failure would originate.
//
// `DeepseekV4Cache` documents the design assumption on its buffer accessors:
//
//     "verify the ephemeral buffers are cleared on restore
//      (they recompute from prompt tokens on the next prefill)"
//
// That holds for a FULL re-prefill. The engine does partial restores —
// `HIT disk boundary=8092 remaining=2004` — where only tokens B..N are fed
// afterwards, so anything the buffers held from tokens 0..B is never
// reconstructed. These tests pin what actually survives, so the assumption is
// checked rather than assumed.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import Testing

@Suite("DSV4 pool/buffer survival across disk restore", .serialized)
struct DeepseekV4PoolBufferRestoreTests {

    private static func pooledRow(_ value: Float, dim: Int = 4) -> MLXArray {
        MLXArray.ones([1, 1, dim]) * value
    }

    /// Populate the branch the way the attention module does: several pooled
    /// rows, plus a partial-window buffer that has not been flushed into the
    /// pool yet.
    private static func fillBranch(
        _ cache: DeepseekV4Cache, _ key: DeepseekV4Cache.BranchKey, base: Float
    ) {
        for index in 0 ..< 3 {
            _ = cache.appendPooled(key, value: pooledRow(base + Float(index)))
        }
        cache.setBuffers(
            key,
            kv: MLXArray.ones([1, 2, 4]) * (base + 100),
            gate: MLXArray.ones([1, 2, 4]) * (base + 200))
    }

    @Test("pooled rows survive a disk round trip")
    func pooledRowsSurviveDiskRoundTrip() throws {
        let source = DeepseekV4Cache(slidingWindow: 16, compressRatio: 4)
        Self.fillBranch(source, .compressor, base: 1)
        Self.fillBranch(source, .indexer, base: 10)

        let sourceComp = try #require(
            source.getPooled(.compressor), "compressor pool was not populated by the test")
        let sourceIdx = try #require(source.getPooled(.indexer))

        let encoded = TQDiskSerializer.serialize(cache: [source])
        let target = DeepseekV4Cache(slidingWindow: 16, compressRatio: 4)
        var restoreTarget: [any KVCache] = [target]
        _ = restoreFromDiskArrays(encoded, into: &restoreTarget)

        let restoredComp = try #require(
            target.getPooled(.compressor),
            "compressor POOL did not survive the disk round trip")
        let restoredIdx = try #require(target.getPooled(.indexer), "indexer pool lost")

        #expect(restoredComp.shape == sourceComp.shape)
        #expect(MLX.allClose(restoredComp, sourceComp).item(Bool.self))
        #expect(restoredIdx.shape == sourceIdx.shape)
        #expect(MLX.allClose(restoredIdx, sourceIdx).item(Bool.self))
    }

    /// Characterize the documented buffer-clearing behaviour, and state plainly
    /// what it costs on a PARTIAL restore.
    ///
    /// If the buffers come back nil, the contract is "the next prefill rebuilds
    /// them". A partial restore only feeds tokens B..N, so a buffer holding the
    /// unflushed window from before B cannot be rebuilt — the pool is then
    /// missing that fragment for the rest of the conversation.
    @Test("buffer clearing on restore is only safe if the boundary is pool-aligned")
    func bufferClearingIsOnlySafeWhenPoolAligned() throws {
        let source = DeepseekV4Cache(slidingWindow: 16, compressRatio: 4)
        Self.fillBranch(source, .compressor, base: 1)

        let (sourceKV, sourceGate) = source.getBuffers(.compressor)
        #expect(sourceKV != nil, "test did not populate the compressor buffer")
        #expect(sourceGate != nil)

        let encoded = TQDiskSerializer.serialize(cache: [source])
        let target = DeepseekV4Cache(slidingWindow: 16, compressRatio: 4)
        var restoreTarget: [any KVCache] = [target]
        _ = restoreFromDiskArrays(encoded, into: &restoreTarget)

        let (restoredKV, restoredGate) = target.getBuffers(.compressor)

        // Whichever way this lands, record it explicitly — the engine's partial
        // restore depends on the answer, and it was previously untested.
        if restoredKV == nil || restoredGate == nil {
            Issue.record(
                """
                DSV4 compressor buffers are DROPPED by the disk round trip \
                (kv=\(restoredKV == nil ? "nil" : "present"), \
                gate=\(restoredGate == nil ? "nil" : "present")).

                That is safe only when the restored boundary is aligned to \
                compressRatio. The engine restores arbitrary boundaries — live: \
                `HIT disk boundary=8092 remaining=2004` — and then feeds only the \
                suffix, so an unflushed window from before the boundary is never \
                rebuilt. The pool is permanently missing that fragment, which \
                degrades attention over long context while leaving generation \
                fluent.
                """)
        } else {
            #expect(MLX.allClose(restoredKV!, sourceKV!).item(Bool.self))
            #expect(MLX.allClose(restoredGate!, sourceGate!).item(Bool.self))
        }
    }

    /// The pool must also survive when the cache is restored and then KEEPS
    /// receiving tokens, which is what a partial restore actually does.
    @Test("pool is intact after restore + continued decoding")
    func poolIntactAfterRestoreAndContinue() throws {
        let source = DeepseekV4Cache(slidingWindow: 16, compressRatio: 4)
        Self.fillBranch(source, .compressor, base: 1)
        for position in 0 ..< 20 {
            _ = source.update(
                keys: MLXArray.ones([1, 1, 1, 8]) * Float(position + 1),
                values: MLXArray.ones([1, 1, 1, 8]) * Float(-(position + 1)))
        }
        let expectedRows = try #require(source.getPooled(.compressor)).dim(1)

        let encoded = TQDiskSerializer.serialize(cache: [source])
        let target = DeepseekV4Cache(slidingWindow: 16, compressRatio: 4)
        var restoreTarget: [any KVCache] = [target]
        _ = restoreFromDiskArrays(encoded, into: &restoreTarget)

        // Continue decoding, as the engine does after `remaining=N`.
        for position in 20 ..< 30 {
            _ = target.update(
                keys: MLXArray.ones([1, 1, 1, 8]) * Float(position + 1),
                values: MLXArray.ones([1, 1, 1, 8]) * Float(-(position + 1)))
        }

        let after = try #require(
            target.getPooled(.compressor),
            "compressor pool was LOST once decoding continued after a restore")
        #expect(
            after.dim(1) >= expectedRows,
            "pool shrank across restore+continue: \(after.dim(1)) < \(expectedRows)")
    }
}
