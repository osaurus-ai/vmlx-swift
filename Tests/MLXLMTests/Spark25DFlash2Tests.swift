// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Spark2.5 as a DFlash 2 target: hidden-state capture, and the rollback of
// its sliding-window layers once the window has filled.
//
// Spark2.5 runs 3 of every 4 layers on a `RotatingKVCache` of the sliding
// window. A speculative verify appends a whole drafted block and then drops
// the rejected suffix. `RotatingKVCache.isTrimmable` goes false the moment
// the window fills, and the iterator used to read "not trimmable" as
// "recurrent state" — the sliding layers kept every rejected row while the
// full-attention layers dropped them. These tests drive the real model
// through verify/rollback cycles and compare every accepted row against a
// cache-free forward over the accepted history.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

@Suite(.serialized)
struct Spark25DFlash2Tests {

    /// Deterministic source so a failure reproduces.
    struct Rng {
        var state: UInt64
        mutating func next(_ range: ClosedRange<Int>) -> Int {
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            let span = UInt64(range.upperBound - range.lowerBound + 1)
            return range.lowerBound + Int((state >> 33) % span)
        }
    }

    static func model(window: Int = 4) throws -> Spark25Model {
        MLXRandom.seed(7)
        let model = Spark25Model(try Spark25Tests.config(["sliding_window": window]))
        eval(model)
        return model
    }

    /// Reference logits for every position of `history`, with no cache —
    /// the sliding mask comes straight from `createAttentionMask(cache: nil)`.
    static func reference(_ model: Spark25Model, _ history: [Int]) -> MLXArray {
        model(MLXArray(history.map(Int32.init)).reshaped(1, -1), cache: nil)
    }

    /// Runs `cycles` verify blocks of uneven size with random acceptance and
    /// returns the largest difference between an accepted row's logits and
    /// the cache-free reference. `rollback` is how the rejected rows leave
    /// the cache.
    static func maxDivergence(
        _ model: Spark25Model, promptLength: Int, cycles: Int, seed: UInt64,
        rollback: ([KVCache], Int) -> Void
    ) -> Float {
        var rng = Rng(state: seed)
        var history = (0 ..< promptLength).map { _ in rng.next(2 ... 127) }
        let cache = model.newCache()
        eval(model(MLXArray(history.map(Int32.init)).reshaped(1, -1), cache: cache))
        var worst: Float = 0
        for cycle in 0 ..< cycles {
            // Every seventh cycle is a single-token step, the iterator's
            // autoregressive fallback, which takes RotatingKVCache's
            // in-place path right after a rollback.
            let rows = cycle % 7 == 6 ? 1 : rng.next(2 ... 5)
            let block = (0 ..< rows).map { _ in rng.next(2 ... 127) }
            let logits = model(MLXArray(block.map(Int32.init)).reshaped(1, -1), cache: cache)
            let accepted = rows == 1 ? 1 : rng.next(1 ... rows)
            history += block[..<accepted]
            let expected = reference(model, history)
            let start = history.count - accepted
            let diff = abs(
                logits[0..., ..<accepted, 0...] - expected[0..., start..., 0...]
            ).max().item(Float.self)
            worst = Swift.max(worst, diff)
            rollback(cache, rows - accepted)
            let offsets = Set(cache.map(\.offset))
            #expect(offsets == [history.count], "cycle \(cycle): layer offsets \(offsets)")
        }
        return worst
    }

    @Test func slidingLayersRollBackExactlyAfterTheWindowFills() throws {
        try MLXMetalTestLock.withLock {
            let model = try Self.model(window: 4)
            // Prompt lengths on both sides of the window, none a multiple of it.
            for (promptLength, seed) in [(3, UInt64(1)), (9, 2), (13, 3)] {
                let worst = Self.maxDivergence(
                    model, promptLength: promptLength, cycles: 40, seed: seed
                ) { cache, rejected in
                    #expect(DFlash2TokenIterator.trimVerifiedRows(cache, rejected))
                }
                // < 1e-3 with MLX_ENABLE_TF32=0; ~1e-2 under the default
                // TF32 matmuls. The legacy rollback measures ~3.9.
                #expect(worst < 5e-2, "prompt \(promptLength): max divergence \(worst)")
            }
        }
    }

    /// Control: the rollback the iterator used before — trim only the
    /// layers that still report `isTrimmable` — must be caught by the same
    /// measurement, or the passing test above proves nothing.
    @Test func legacyRollbackIsCaughtByTheSameMeasurement() throws {
        try MLXMetalTestLock.withLock {
            let model = try Self.model(window: 4)
            var rng = Rng(state: 11)
            var history = (0 ..< 9).map { _ in rng.next(2 ... 127) }
            let cache = model.newCache()
            eval(model(MLXArray(history.map(Int32.init)).reshaped(1, -1), cache: cache))
            var worst: Float = 0
            for _ in 0 ..< 12 {
                let block = (0 ..< 4).map { _ in rng.next(2 ... 127) }
                let logits = model(MLXArray(block.map(Int32.init)).reshaped(1, -1), cache: cache)
                history += block[..<2]
                let expected = Self.reference(model, history)
                let diff = abs(
                    logits[0..., ..<2, 0...] - expected[0..., (history.count - 2)..., 0...]
                ).max().item(Float.self)
                worst = Swift.max(worst, diff)
                for layer in cache where layer.isTrimmable { layer.trim(2) }
            }
            print("[control] legacy rollback max divergence \(worst)")
            #expect(worst > 0.5, "legacy rollback went undetected (max divergence \(worst))")
        }
    }

    @Test func rotatingCacheRollbackInvariant() throws {
        try MLXMetalTestLock.withLock {
            // Window 7, blocks of 3: nothing divides evenly.
            let cache = RotatingKVCache(maxSize: 7, keep: 0)
            func rows(_ from: Int, _ count: Int) -> MLXArray {
                MLXArray((from ..< from + count).map(Float.init)).reshaped(1, 1, count, 1)
            }
            var position = 0
            for _ in 0 ..< 9 {
                _ = cache.update(keys: rows(position, 3), values: rows(position, 3))
                position += 3
                #expect(cache.canRollBackVerifiedRows(1))
                cache.trim(1)
                position -= 1
                // The buffer is the accepted history's tail, in order.
                let keys = cache.state[0].reshaped(-1).asArray(Float.self)
                let tail = (Swift.max(0, position - keys.count) ..< position).map(Float.init)
                #expect(keys == tail)
                #expect(keys.count >= Swift.min(6, position))
            }
            // A ring rotated by single-token writes cannot give rows back.
            for _ in 0 ..< 3 {
                _ = cache.update(keys: rows(position, 1), values: rows(position, 1))
                position += 1
            }
            #expect(!cache.canRollBackVerifiedRows(2))
            #expect(!DFlash2TokenIterator.trimVerifiedRows([cache], 2))
            #expect(cache.offset == position, "a refused rollback must not move the cache")
        }
    }

    @Test func captureIsASideEffectOnly() throws {
        try MLXMetalTestLock.withLock {
            let model = try Self.model()
            let tokens = MLXArray((0 ..< 11).map { Int32(3 + $0 * 7 % 120) }).reshaped(1, -1)
            let plain = model(tokens, cache: model.newCache())
            let (logits, captured) = model(
                tokens, cache: model.newCache(), captureLayerIDs: [0, 2, 3])
            #expect(abs(plain - logits).max().item(Float.self) == 0)
            #expect(Set(captured.keys) == [0, 2, 3])
            for (_, hidden) in captured {
                #expect(hidden.shape == [1, 11, 64])
                #expect(hidden.dtype == model.model.embedding.weight.dtype)
            }
            let (_, none) = model(tokens, cache: nil, captureLayerIDs: [])
            #expect(none.isEmpty)
            // The shared head is the model's own projection.
            let hidden = model.model(tokens, cache: nil)
            #expect(abs(model.projectToLogits(hidden) - plain).max().item(Float.self) == 0)
            #expect(model.embed(tokens).shape == [1, 11, 64])
        }
    }

    @Test func conformsAsADFlash2Target() throws {
        let model: any LanguageModel = Spark25Model(try Spark25Tests.config())
        #expect(model is any DFlash2Target)
    }

    @Test func defaultWidthIsCappedOnlyWhereItWasProven() {
        let attention: [KVCache] = [KVCacheSimple(), RotatingKVCache(maxSize: 512)]
        let hybrid: [KVCache] = [KVCacheSimple(), MambaCache()]
        #expect(
            DFlash2TokenIterator.defaultBlockSize(requested: nil, trained: 8, cache: attention) == 5
        )
        #expect(
            DFlash2TokenIterator.defaultBlockSize(requested: nil, trained: 4, cache: attention) == 4
        )
        #expect(
            DFlash2TokenIterator.defaultBlockSize(requested: nil, trained: 8, cache: hybrid) == 8)
        #expect(
            DFlash2TokenIterator.defaultBlockSize(requested: 8, trained: 8, cache: attention) == 8)
        #expect(DFlash2TokenIterator.defaultBlockSize(requested: 3, trained: 8, cache: hybrid) == 3)
    }
}
