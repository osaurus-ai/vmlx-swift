//
//  NativeMTPARSafetyTests.swift
//  MLXLMCommonFocusedTests
//
//  Pins legacy Swift context-scaling arithmetic and the correctness-updated
//  Python policy for independently confirmed mean losses. No model execution.
//

import XCTest

@testable import MLXLMCommon

final class NativeMTPARSafetyTests: XCTestCase {

    private typealias V = NativeMTPTokenIterator

    func testFastMTPHolds() {
        // MTP at 5ms/tok, AR seed 10ms, no context growth -> MTP is 2x faster.
        XCTAssertNil(
            V.windowedARVerdict(
                arStepMs: 10, firstVerifyMs: 12, windowCycles: 16,
                deltaEmitted: 32, deltaWallMs: 160, deltaVerifyMs: 12 * 16, margin: 1.25))
    }

    func testSlowMTPTrips() {
        // MTP at 20ms/tok vs AR 10ms (flat context) -> 2x slower, must trip.
        let v = V.windowedARVerdict(
            arStepMs: 10, firstVerifyMs: 12, windowCycles: 16,
            deltaEmitted: 16, deltaWallMs: 320, deltaVerifyMs: 12 * 16, margin: 1.25)
        XCTAssertNotNil(v)
        XCTAssertEqual(v?.mtpMsPerToken ?? 0, 20, accuracy: 1e-6)
        XCTAssertEqual(v?.arBaselineMs ?? 0, 10, accuracy: 1e-6)
    }

    func testLongContextDoesNotFalseTrip() {
        // Verify doubled (24 vs 12) -> AR baseline scales to 20; MTP at 18
        // is still worth it (18 < 20 * 1.25). A stale short-context baseline
        // WOULD have tripped (18 > 10 * 1.25) — the context-fairness fix.
        XCTAssertNil(
            V.windowedARVerdict(
                arStepMs: 10, firstVerifyMs: 12, windowCycles: 16,
                deltaEmitted: 16, deltaWallMs: 288, deltaVerifyMs: 24 * 16, margin: 1.25))
        XCTAssertNotNil(
            V.windowedARVerdict(
                arStepMs: 10, firstVerifyMs: 0, windowCycles: 16,
                deltaEmitted: 16, deltaWallMs: 288, deltaVerifyMs: 24 * 16, margin: 1.25),
            "no scaling -> stale baseline -> trips")
    }

    func testContextScaledSlowStillTrips() {
        // Even with context growth (baseline 20), MTP at 30ms/tok loses.
        let v = V.windowedARVerdict(
            arStepMs: 10, firstVerifyMs: 12, windowCycles: 16,
            deltaEmitted: 16, deltaWallMs: 480, deltaVerifyMs: 24 * 16, margin: 1.25)
        XCTAssertEqual(v?.mtpMsPerToken ?? 0, 30, accuracy: 1e-6)
        XCTAssertEqual(v?.arBaselineMs ?? 0, 20, accuracy: 1e-6)
    }

    func testGuardsNeverDivide() {
        for (emitted, wall, ar, cycles) in [(0, 100.0, 10.0, 16), (16, 0.0, 10.0, 16),
                                            (16, 100.0, 0.0, 16), (16, 100.0, 10.0, 0)] {
            XCTAssertNil(
                V.windowedARVerdict(
                    arStepMs: ar, firstVerifyMs: 12, windowCycles: cycles,
                    deltaEmitted: emitted, deltaWallMs: wall, deltaVerifyMs: 192, margin: 1.25),
                "emitted=\(emitted) wall=\(wall) ar=\(ar) cycles=\(cycles) must not judge")
        }
    }

    func testContextScaleNeverMakesBaselineCheaper() {
        // Verify got FASTER than the first cycle (cache warmed): scale clamps
        // at 1.0 so the baseline never drops below the seed AR step.
        let v = V.windowedARVerdict(
            arStepMs: 10, firstVerifyMs: 12, windowCycles: 16,
            deltaEmitted: 16, deltaWallMs: 320, deltaVerifyMs: 6 * 16, margin: 1.25)
        XCTAssertEqual(v?.arBaselineMs ?? 0, 10, accuracy: 1e-6)
    }

    // MARK: median guard (one stalled cycle must not trip the window)

    private func ring(_ cycleMs: [Double], emittedPerCycle: Int = 4) -> [V.ARSafetySample] {
        var out = [V.ARSafetySample(emitted: 0, wall: 0, verifyTotal: 0)]
        var wall = 0.0, emitted = 0
        for ms in cycleMs {
            wall += ms / 1000; emitted += emittedPerCycle
            out.append(V.ARSafetySample(emitted: emitted, wall: wall, verifyTotal: 0))
        }
        return out
    }

    func testMedianIgnoresSingleStallCycle() {
        // 7 healthy cycles at 10 ms/tok, one 400 ms/tok stall: the MEAN is
        // ~59 ms/tok (would trip against a 25 ms AR step × 1.25), the median
        // stays at 10 ms/tok → no trip.
        let r = ring([40, 40, 40, 40, 1600, 40, 40, 40])
        XCTAssertEqual(V.medianCycleMsPerToken(r) ?? -1, 10, accuracy: 1e-9)
    }

    func testMedianTripsOnSustainedLoss() {
        // Every cycle costs 45 ms/tok against a 25 ms AR step → median agrees.
        let r = ring(Array(repeating: 180, count: 8))
        XCTAssertEqual(V.medianCycleMsPerToken(r) ?? -1, 45, accuracy: 1e-9)
        XCTAssertGreaterThan(V.medianCycleMsPerToken(r)!, 25 * 1.25)
    }

    func testMedianGuardsEmptyAndZeroEmission() {
        XCTAssertNil(V.medianCycleMsPerToken([]))
        XCTAssertNil(V.medianCycleMsPerToken([V.ARSafetySample(emitted: 0, wall: 0, verifyTotal: 0)]))
        let flat = [V.ARSafetySample(emitted: 3, wall: 0, verifyTotal: 0), V.ARSafetySample(emitted: 3, wall: 1, verifyTotal: 0)]
        XCTAssertNil(V.medianCycleMsPerToken(flat))
    }

    // MARK: independently confirmed mean loss (measured, unscaled AR only)

    private func meanLossWindow(
        _ state: inout V.RepeatedMeanLossState, end: Int,
        wall: Double = 280, emitted: Int = 8, median: Double = 20,
        baseline: Double = 25, margin: Double = 1, depth: Int = 1,
        measured: Bool = true, scaled: Bool = false, probe: Bool = false
    ) -> Bool {
        state.confirm(
            endCycle: end, windowCycles: 8, emitted: emitted, wallMs: wall,
            medianMs: median, arMs: baseline, margin: margin, depth: depth,
            measured: measured, scaled: scaled, probe: probe)
    }

    func testRepeatedMinorityStallsRequireTwoDisjointWindows() {
        // Six 20ms + two 80ms one-token cycles: mean35 loses to AR25,
        // median20 wins. The old conjunction rejects EVERY such window.
        var state = V.RepeatedMeanLossState()
        XCTAssertFalse(meanLossWindow(&state, end: 16))
        XCTAssertEqual(state.pendingEndCycle, 16)
        for end in 17..<24 {
            XCTAssertFalse(meanLossWindow(&state, end: end), "overlap is not confirmation")
            XCTAssertEqual(state.pendingEndCycle, 16)
        }
        XCTAssertTrue(meanLossWindow(&state, end: 24))
        XCTAssertNil(state.pendingEndCycle)
    }

    func testOneOffStallThenIndependentWinOrEqualityClearsConfirmation() {
        for settledCost in [160.0, 200.0] {
            var state = V.RepeatedMeanLossState()
            XCTAssertFalse(meanLossWindow(&state, end: 16))
            // A cheap overlapping observation must not move the pending fence.
            XCTAssertFalse(meanLossWindow(&state, end: 20, wall: settledCost))
            XCTAssertEqual(state.pendingEndCycle, 16)
            XCTAssertFalse(meanLossWindow(&state, end: 24, wall: settledCost))
            XCTAssertNil(state.pendingEndCycle)
            XCTAssertFalse(meanLossWindow(&state, end: 32))
            XCTAssertEqual(state.pendingEndCycle, 32, "new loss needs fresh confirmation")
        }
    }

    func testReferenceChangesCannotReuseLossConfirmation() {
        for change in ["baseline", "margin", "depth"] {
            var state = V.RepeatedMeanLossState()
            XCTAssertFalse(meanLossWindow(&state, end: 16))
            XCTAssertFalse(meanLossWindow(
                &state, end: 24, baseline: change == "baseline" ? 30 : 25,
                margin: change == "margin" ? 1.1 : 1, depth: change == "depth" ? 2 : 1))
            XCTAssertEqual(state.pendingEndCycle, 24)
        }
    }

    func testDepthResetExcludesMixedDepthWindowAndRequiresTwoNewWindows() {
        var state = V.RepeatedMeanLossState()
        XCTAssertFalse(meanLossWindow(&state, end: 16, depth: 3))
        state.reset(afterCycle: 18)
        XCTAssertFalse(meanLossWindow(&state, end: 24, depth: 2))
        XCTAssertNil(state.pendingEndCycle, "window includes cycles before the depth switch")
        XCTAssertFalse(meanLossWindow(&state, end: 26, depth: 2))
        XCTAssertEqual(state.pendingEndCycle, 26)
        XCTAssertTrue(meanLossWindow(&state, end: 34, depth: 2))
    }

    func testUnmeasuredScaledAndProbeWindowsCannotConfirm() {
        for excluded in ["unmeasured", "scaled", "probe"] {
            var state = V.RepeatedMeanLossState()
            XCTAssertFalse(meanLossWindow(&state, end: 16))
            XCTAssertFalse(meanLossWindow(
                &state, end: 24, measured: excluded != "unmeasured",
                scaled: excluded == "scaled", probe: excluded == "probe"))
            XCTAssertNil(state.pendingEndCycle)
            XCTAssertFalse(meanLossWindow(&state, end: 32))
            XCTAssertEqual(state.pendingEndCycle, 32)
        }
    }

    func testPauseOrReentryResetDoesNotCarryLoss() {
        var state = V.RepeatedMeanLossState()
        XCTAssertFalse(meanLossWindow(&state, end: 16))
        state.reset(afterCycle: 16)
        XCTAssertFalse(meanLossWindow(&state, end: 24))
        XCTAssertTrue(meanLossWindow(&state, end: 32))
    }

    func testTokenWeightedWinningCostDoesNotArmDespiteLosingMedian() {
        var state = V.RepeatedMeanLossState()
        // Five 30ms single-token cycles + three 100ms ten-token cycles:
        // median30 loses, but total450ms /35 tokens wins against AR25.
        for end in 16...40 {
            XCTAssertFalse(meanLossWindow(&state, end: end, wall: 450, emitted: 35, median: 30))
            XCTAssertNil(state.pendingEndCycle)
        }
    }

    func testUniformLossKeepsImmediateMedianDecisionWithoutConfirmation() {
        var state = V.RepeatedMeanLossState()
        XCTAssertFalse(meanLossWindow(&state, end: 16, wall: 240, median: 30))
        XCTAssertNil(state.pendingEndCycle)
        XCTAssertNotNil(V.windowedARVerdict(
            arStepMs: 25, firstVerifyMs: 0, windowCycles: 8,
            deltaEmitted: 8, deltaWallMs: 240, deltaVerifyMs: 0, margin: 1))
        XCTAssertGreaterThan(V.medianCycleMsPerToken(ring(Array(repeating: 30, count: 8),
                                                        emittedPerCycle: 1)) ?? 0, 25,
                             "existing median conjunction owns immediate loss")
    }

    func testInvalidMeanLossSamplesNeverArmOrConfirm() {
        for (wall, emitted, baseline) in [(0.0, 8, 25.0), (280, 0, 25),
                                           (280, 8, 0), (.infinity, 8, 25)] {
            var state = V.RepeatedMeanLossState()
            XCTAssertFalse(meanLossWindow(&state, end: 16))
            XCTAssertFalse(meanLossWindow(&state, end: 24, wall: wall, emitted: emitted,
                                           baseline: baseline))
            XCTAssertNil(state.pendingEndCycle)
        }
    }
}
