// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// DSV4 PARTIAL disk restore, then continue.
//
// `DeepseekV4CacheDiskRoundTripTests` proves serialize -> deserialize is
// lossless for a cache that is then left alone. That is not what the engine
// does. Live DSV4 (2026-08-02) restores an EARLIER boundary and keeps feeding:
//
//   [vmlx][cache/fetch] HIT disk boundary=8092  remaining=2004
//   [vmlx][cache/fetch] HIT disk boundary=10095 remaining=2498
//
// DSV4's per-layer cache is `RotatingKVCache(maxSize: slidingWindow, keep: 0)`
// for cr=0 layers, i.e. a ring. `CacheHelpers` already warns that once that
// ring wraps, the cache "cannot produce a lossless N-1 checkpoint by trimming
// the completed prompt". A restore at an arbitrary earlier boundary followed
// by thousands more tokens is that hazard in its worst form.
//
// The invariant the prefix cache depends on is stronger than round-tripping:
//
//     restore(B) + feed(N-B)   ==   feed(N)
//
// If it does not hold, attention reads a subtly wrong window and the model
// stays fluent while losing the plot — which is exactly the symptom that
// appeared the day disk L2 started working on this machine (a coherent
// reasoning loop repeating verbatim at ~12.5k tokens, at temperature 1.0 /
// top_p 0.95, where sampling cannot explain verbatim repetition).

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import Testing

@Suite("DSV4 partial disk restore then continue", .serialized)
struct DeepseekV4PartialRestoreContinuationTests {

    /// One decode step of deterministic, position-dependent content, so a
    /// misaligned ring shows up as a value mismatch rather than coincidentally
    /// matching uniform data.
    private static func step(
        _ cache: any KVCache, position: Int, B: Int = 1, H: Int = 1, headDim: Int = 8
    ) {
        let keys = MLXArray.ones([B, H, 1, headDim]) * Float(position + 1)
        let values = MLXArray.ones([B, H, 1, headDim]) * Float(-(position + 1))
        _ = cache.update(keys: keys, values: values)
    }

    private static func feed(_ cache: any KVCache, from: Int, to: Int) {
        for position in from ..< to { step(cache, position: position) }
    }

    private static func assertSameState(
        _ warm: any KVCache, _ cold: any KVCache, label: String
    ) {
        #expect(warm.offset == cold.offset, "\(label): offset diverged")
        let warmState = warm.state
        let coldState = cold.state
        guard warmState.count == coldState.count else {
            Issue.record("\(label): state tensor count \(warmState.count) vs \(coldState.count)")
            return
        }
        for (index, pair) in zip(warmState, coldState).enumerated() {
            let (w, c) = pair
            guard w.shape == c.shape else {
                Issue.record("\(label): tensor \(index) shape \(w.shape) vs \(c.shape)")
                continue
            }
            let equal = MLX.allClose(w, c).item(Bool.self)
            #expect(equal, "\(label): tensor \(index) differs after restore+continue")
        }
    }

    /// The ring has wrapped several times before the boundary, and thousands
    /// more tokens arrive after it — the live shape, scaled down.
    @Test("RotatingKVCache: restore(B) + feed(N-B) == feed(N) after the ring wraps")
    func rotatingPartialRestoreMatchesColdPrefill() {
        let window = 16
        let boundary = 40   // ring already wrapped twice
        let total = 72      // and wraps twice more after the restore

        // Cold: one continuous run.
        let cold = RotatingKVCache(maxSize: window, keep: 0)
        Self.feed(cold, from: 0, to: total)

        // Warm: run to the boundary, persist, restore, continue.
        let source = RotatingKVCache(maxSize: window, keep: 0)
        Self.feed(source, from: 0, to: boundary)
        let encoded = TQDiskSerializer.serialize(cache: [source])

        let warm = RotatingKVCache(maxSize: window, keep: 0)
        var target: [any KVCache] = [warm]
        _ = restoreFromDiskArrays(encoded, into: &target)
        #expect(warm.offset == boundary, "restore did not reinstate the boundary offset")
        Self.feed(warm, from: boundary, to: total)

        Self.assertSameState(warm, cold, label: "rotating restore(\(boundary))+\(total - boundary)")
    }

    /// Same invariant for the composite type every cr>0 layer uses.
    @Test("DeepseekV4Cache: restore(B) + feed(N-B) == feed(N) after the ring wraps")
    func deepseekV4CachePartialRestoreMatchesColdPrefill() {
        let window = 16
        let boundary = 40
        let total = 72

        let cold = DeepseekV4Cache(slidingWindow: window, compressRatio: 4)
        Self.feed(cold, from: 0, to: total)

        let source = DeepseekV4Cache(slidingWindow: window, compressRatio: 4)
        Self.feed(source, from: 0, to: boundary)
        let encoded = TQDiskSerializer.serialize(cache: [source])

        let warm = DeepseekV4Cache(slidingWindow: window, compressRatio: 4)
        var target: [any KVCache] = [warm]
        _ = restoreFromDiskArrays(encoded, into: &target)
        Self.feed(warm, from: boundary, to: total)

        Self.assertSameState(warm, cold, label: "dsv4 restore(\(boundary))+\(total - boundary)")
    }

    /// Edge case: boundary exactly on a ring wrap. Off-by-one in the rotation
    /// index is invisible mid-window and maximal here.
    @Test("restore exactly on a ring-wrap boundary continues correctly")
    func restoreOnExactWrapBoundary() {
        let window = 16
        for boundary in [window, window * 2, window * 3] {
            let total = boundary + window + 3

            let cold = RotatingKVCache(maxSize: window, keep: 0)
            Self.feed(cold, from: 0, to: total)

            let source = RotatingKVCache(maxSize: window, keep: 0)
            Self.feed(source, from: 0, to: boundary)
            let encoded = TQDiskSerializer.serialize(cache: [source])

            let warm = RotatingKVCache(maxSize: window, keep: 0)
            var target: [any KVCache] = [warm]
            _ = restoreFromDiskArrays(encoded, into: &target)
            Self.feed(warm, from: boundary, to: total)

            Self.assertSameState(warm, cold, label: "wrap-boundary restore(\(boundary))")
        }
    }

    /// Edge case: restore, then feed only ONE token — the `skipExactDisk`
    /// N-1 seed path the engine prefers for path-dependent topologies.
    @Test("restore(N-1) + one token == feed(N)")
    func restoreMinusOneSeedMatches() {
        let window = 16
        let total = 51

        let cold = RotatingKVCache(maxSize: window, keep: 0)
        Self.feed(cold, from: 0, to: total)

        let source = RotatingKVCache(maxSize: window, keep: 0)
        Self.feed(source, from: 0, to: total - 1)
        let encoded = TQDiskSerializer.serialize(cache: [source])

        let warm = RotatingKVCache(maxSize: window, keep: 0)
        var target: [any KVCache] = [warm]
        _ = restoreFromDiskArrays(encoded, into: &target)
        Self.feed(warm, from: total - 1, to: total)

        Self.assertSameState(warm, cold, label: "N-1 seed restore")
    }
}
