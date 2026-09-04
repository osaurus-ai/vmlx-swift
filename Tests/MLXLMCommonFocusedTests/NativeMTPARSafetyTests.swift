//
//  NativeMTPARSafetyTests.swift
//  MLXLMCommonFocusedTests
//
//  Pins the AR-safety governor's pure decision arithmetic — a 1:1 port of
//  the Python engine's tests/test_native_mtp_ar_safety.py so both runtimes
//  trip on the same numbers: fast MTP holds, slow MTP trips, long-context
//  growth does NOT false-trip, and the div-by-small/empty guards.
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
}
