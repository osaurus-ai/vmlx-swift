// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import XCTest

@testable import MLXLMCommon

/// The governor chooses between two lossless paths by measured speed. These
/// drive it with a synthetic clock: drafting at a fixed cost per cycle and
/// tokens per cycle, plain decoding at a fixed cost per token.
final class DFlash2ThroughputGovernorTests: XCTestCase {

    /// Runs `tokens` through the governor and returns it with the number of
    /// tokens each mode produced.
    private func run(
        tokens: Int, draftCycle: Double, tokensPerCycle: (Int) -> Int, plainStep: Double
    ) -> (governor: DFlash2ThroughputGovernor, drafted: Int, plain: Int) {
        var g = DFlash2ThroughputGovernor()
        var clock = 100.0
        var drafted = 0
        var plain = 0
        var cycle = 0
        while drafted + plain < tokens {
            if g.shouldDraft(now: clock) {
                let n = tokensPerCycle(cycle)
                cycle += 1
                clock += draftCycle
                drafted += n
                g.record(tokens: n, now: clock)
            } else {
                clock += plainStep
                plain += 1
                g.record(tokens: 1, now: clock)
            }
        }
        return (g, drafted, plain)
    }

    func testDraftingThatPaysKeepsDrafting() {
        // 2.6 tokens per 17.6 ms cycle (148 tok/s) against 9.2 ms plain.
        let r = run(
            tokens: 2000, draftCycle: 0.0176, tokensPerCycle: { $0 % 5 < 3 ? 3 : 2 },
            plainStep: 0.0092)
        XCTAssertEqual(r.governor.pauses, 0)
        XCTAssertEqual(r.plain, DFlash2ThroughputGovernor.calibrationTokens)
        XCTAssertGreaterThan(r.governor.draftRate ?? 0, r.governor.plainRate ?? .infinity)
    }

    func testDraftingThatLosesStepsAsideWithGrowingBackoff() {
        // 1.8 tokens per 20 ms cycle (90 tok/s) against 9.1 ms plain (110 tok/s).
        let r = run(
            tokens: 3000, draftCycle: 0.020, tokensPerCycle: { $0 % 5 < 4 ? 2 : 1 },
            plainStep: 0.0091)
        XCTAssertGreaterThan(r.governor.pauses, 2)
        XCTAssertGreaterThan(
            r.plain, r.drafted * 3, "a losing drafter must spend most tokens plain")
        XCTAssertEqual(
            r.governor.pausedTokens, r.plain - DFlash2ThroughputGovernor.calibrationTokens)
        // Paused time is measured, not assumed: it recovers the plain step.
        XCTAssertEqual(
            r.governor.pausedSeconds / Double(r.governor.pausedTokens), 0.0091, accuracy: 1e-9)
    }

    func testAWinAfterLosingResumesDrafting() {
        // Loses for the first 300 tokens' worth of cycles, then content turns
        // structured and it wins; the governor must find its way back.
        var g = DFlash2ThroughputGovernor()
        var clock = 0.0
        var emitted = 0
        var lateDrafted = 0
        while emitted < 4000 {
            if g.shouldDraft(now: clock) {
                let winning = emitted > 300
                let n = winning ? 4 : 1
                clock += winning ? 0.018 : 0.020
                emitted += n
                if emitted > 1500 { lateDrafted += n }
                g.record(tokens: n, now: clock)
            } else {
                clock += 0.0091
                emitted += 1
                g.record(tokens: 1, now: clock)
            }
        }
        XCTAssertGreaterThan(g.pauses, 0)
        XCTAssertEqual(g.mode, .drafting)
        XCTAssertGreaterThan(lateDrafted, 2000)
    }

    func testNothingIsDecidedBeforeAWindowCompletes() {
        var g = DFlash2ThroughputGovernor()
        var clock = 0.0
        for _ in 0 ..< DFlash2ThroughputGovernor.windowCycles - 1 {
            XCTAssertTrue(g.shouldDraft(now: clock))
            clock += 1
            g.record(tokens: 1, now: clock)
        }
        XCTAssertEqual(g.mode, .drafting)
        XCTAssertNil(g.plainRate)
    }

    func testOneMarginallyLosingWindowDoesNotPause() {
        // Drafting 5% slower than plain for exactly one window, then winning:
        // noise, not a trend — no pause.
        var g = DFlash2ThroughputGovernor()
        var clock = 0.0
        var cycle = 0
        var emitted = 0
        while emitted < 600 {
            if g.shouldDraft(now: clock) {
                cycle += 1
                let slow = (13 ... 24).contains(cycle)  // the second window
                clock += slow ? 0.0216 : 0.015
                emitted += 2
                g.record(tokens: 2, now: clock)
            } else {
                clock += 0.0098
                emitted += 1
                g.record(tokens: 1, now: clock)
            }
        }
        XCTAssertEqual(g.pauses, 0)
    }
}
