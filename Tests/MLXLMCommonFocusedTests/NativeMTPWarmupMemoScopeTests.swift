// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// The hybrid warmup memo's contract is "a property of the model, not the
// prompt" — yet the warmup verdict it stores is an acceptance criterion,
// and acceptance is content. Live repro (Qwen3.8-27B-JANG_4D, temp 0): a
// prose turn warms up at averageAccepted 1.00 and a code turn at 3.04 on
// the SAME loaded model. Before the scope fix, the prose turn memoized a
// permanent negative verdict and every later code turn ran pure AR
// (verifyCalls=0, reason=hybrid_warmup_memo) until the model reloaded.
//
// These tests pin the boundary: only a catastrophic miss — below even the
// depth-1 floor, which no depth could ever clear — may be blamed on the
// model and memoized. Everything above is per-request content variance.

import Testing

@testable import MLXLMCommon

@Suite("NativeMTP hybrid warmup memo scope")
struct NativeMTPWarmupMemoScopeTests {

    @Test("prose-grade warmup miss is content, not a model property")
    func contentMissIsNotMemoizable() {
        // The live prose repro: 1.00 average accepted per verify at depth 3
        // (floor 1.65). Failing warmup is correct; memoizing it is not.
        #expect(!NativeMTPTokenIterator.warmupFailureIsModelProperty(averageAccepted: 1.00))
        // Just below a depth-2 floor (1.10) but healthy for depth 1.
        #expect(!NativeMTPTokenIterator.warmupFailureIsModelProperty(averageAccepted: 0.80))
        // Exactly the depth-1 floor still clears depth 1, so it stays
        // per-request.
        #expect(!NativeMTPTokenIterator.warmupFailureIsModelProperty(averageAccepted: 0.55))
    }

    @Test("catastrophic miss below the depth-1 floor is the model")
    func catastrophicMissIsMemoizable() {
        // A broken or mismatched MTP head accepts near zero; no depth could
        // ever clear warmup, so re-probing every request only burns the
        // 16-cycle probe the memo census flagged (84% of requests).
        #expect(NativeMTPTokenIterator.warmupFailureIsModelProperty(averageAccepted: 0.0))
        #expect(NativeMTPTokenIterator.warmupFailureIsModelProperty(averageAccepted: 0.30))
        #expect(NativeMTPTokenIterator.warmupFailureIsModelProperty(averageAccepted: 0.54))
    }

    @Test("memo verdict stays per model instance and passing runs still memoize")
    func memoStillWorksForItsRealCase() {
        final class Marker {}
        let healthy = Marker()
        let broken = Marker()
        #expect(NativeMTPHybridWarmupMemo.verdict(for: healthy) == nil)
        NativeMTPHybridWarmupMemo.record(true, for: healthy)
        NativeMTPHybridWarmupMemo.record(false, for: broken)
        #expect(NativeMTPHybridWarmupMemo.verdict(for: healthy) == true)
        #expect(NativeMTPHybridWarmupMemo.verdict(for: broken) == false)
    }
}
