// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// The hybrid safety-warmup memo records a verdict "for the model, not the
// prompt" — so only a CATASTROPHIC probe failure (broken/mismatched head,
// genuinely depth-independent) may be memoized. The pass floor scales with
// depth (0.55 * depth) but the catastrophic line was the bare constant 0.55,
// so at depth 1 the two coincided: every ordinary depth-1 content miss was
// recorded as a failed model and hard-disabled MTP at ALL depths for the
// rest of the residency. Live signature: `adaptiveFallback=hybrid_warmup_memo`,
// `verifyCalls=0`, `avgCommittedPerVerify=0.00`, on every turn after one
// depth-1 warmup at ~0.44 acceptance.

import Testing

@testable import MLXLMCommon

@Suite("Hybrid warmup memo scope")
struct HybridWarmupMemoScopeTests {

    /// The live failure: 0.44 at depth 1 is a content miss, not a broken head.
    @Test func anOrdinaryDepthOneMissIsNotAModelProperty() {
        #expect(
            !NativeMTPTokenIterator.warmupFailureIsModelProperty(
                averageAccepted: 0.44, depth: 1))
    }

    /// Truly broken heads accept essentially nothing at any depth.
    @Test func aNearZeroAcceptanceIsCatastrophicAtEveryDepth() {
        #expect(NativeMTPTokenIterator.warmupFailureIsModelProperty(
            averageAccepted: 0.05, depth: 1))
        #expect(NativeMTPTokenIterator.warmupFailureIsModelProperty(
            averageAccepted: 0.05, depth: 2))
    }

    /// Depth 2 keeps its historical line exactly: below 0.55 is catastrophic.
    @Test func depthTwoLineIsUnchanged() {
        #expect(NativeMTPTokenIterator.warmupFailureIsModelProperty(
            averageAccepted: 0.54, depth: 2))
        #expect(!NativeMTPTokenIterator.warmupFailureIsModelProperty(
            averageAccepted: 0.56, depth: 2))
    }

    /// A marginal depth-2 miss (accepted >= the depth-1 floor 0.55) must NOT
    /// be catastrophic: the iterator downshifts to depth 1 and keeps
    /// speculating. This is the exact live case — acceptance dips to
    /// ~0.94-1.06 after a plain-decoded span and self-recovers.
    @Test func aMarginalDepthTwoMissIsNeitherCatastrophicNorFatal() {
        #expect(!NativeMTPTokenIterator.warmupFailureIsModelProperty(
            averageAccepted: 0.94, depth: 2))
        #expect(!NativeMTPTokenIterator.warmupFailureIsModelProperty(
            averageAccepted: 1.06, depth: 2))
    }

    /// The line must sit strictly BELOW each depth's pass floor, or a plain
    /// failure and a catastrophic one are the same thing again.
    @Test func catastrophicIsStrictlyBelowThePassFloorAtEveryDepth() {
        for depth in 1...3 {
            let floor = 0.55 * Double(depth)
            #expect(
                !NativeMTPTokenIterator.warmupFailureIsModelProperty(
                    averageAccepted: floor - 0.01, depth: depth),
                "depth \(depth): a just-under-floor miss must stay per-request")
        }
    }
}
