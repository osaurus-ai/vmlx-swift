// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// DFlash2WidthChooser: AR cycles carry no acceptance evidence, so they must not age it.

import Foundation
import XCTest

@testable import MLXLMCommon

final class DFlash2WidthChooserTests: XCTestCase {

    /// Run `cycles` of the chooser against a fixed world: `seconds[w]` per cycle at width w, and
    /// `acceptedAt[w]` accepted drafts per verify cycle. Returns how often each width ran.
    private func simulate(
        cycles: Int, seconds: [Int: Double], accepted: (Int) -> Int
    ) -> [Int: Int] {
        let costs = DFlash2WidthCostTable()
        var chooser = DFlash2WidthChooser(widths: [1, 8, 16], costs: costs)
        var counts: [Int: Int] = [:]
        for _ in 0 ..< cycles {
            let w = chooser.width
            counts[w, default: 0] += 1
            let tokens = w == 1 ? 1 : 1 + min(accepted(w), w - 1)
            chooser.observe(verifyWidth: w, tokens: tokens, seconds: seconds[w]!)
        }
        return counts
    }

    /// Low-acceptance drafter (Diplodocus preview 3 on reasoning prose: ~0.8 accepted per cycle).
    /// AR is cheaper per token, so the chooser settles on width 1 and re-tries wide widths only on
    /// the periodic probe, not every few cycles.
    func testLowAcceptanceSettlesOnARAndProbesOnlyPeriodically() {
        let counts = simulate(
            cycles: 640, seconds: [1: 0.0128, 8: 0.031, 16: 0.033], accepted: { _ in 0 })
        let wide = (counts[8] ?? 0) + (counts[16] ?? 0)
        XCTAssertGreaterThan(counts[1] ?? 0, 560, "\(counts)")
        // Confirming re-times back off 32, 64, 128, 256, 512: a handful of probes plus the two
        // warm-up/first-timing cycles per width, not one every 32 cycles (20 here) or every ~6.
        XCTAssertLessThanOrEqual(wide, 14, "wide widths re-tried too often: \(counts)")
    }

    /// High-acceptance drafter (predictable text): the wide verify pays and stays chosen.
    func testHighAcceptanceKeepsWideVerify() {
        let counts = simulate(
            cycles: 320, seconds: [1: 0.0128, 8: 0.031, 16: 0.033], accepted: { w in w - 2 })
        XCTAssertLessThan(counts[1] ?? 0, 40, "\(counts)")
    }

    /// The evidence an AR stretch leaves untouched: after AR-only observations, the expected tokens
    /// for width 8 equal what they were before the stretch.
    func testARCyclesDoNotAgeAcceptanceEvidence() {
        let costs = DFlash2WidthCostTable()
        var chooser = DFlash2WidthChooser(widths: [1, 8], costs: costs)
        for _ in 0 ..< 6 { chooser.observe(verifyWidth: 8, tokens: 1, seconds: 0.03) }
        let before = chooser.expectedTokens(8)
        for _ in 0 ..< 30 { chooser.observe(verifyWidth: 1, tokens: 1, seconds: 0.0128) }
        XCTAssertEqual(chooser.expectedTokens(8), before, accuracy: 1e-12)
    }

    /// Backoff must not hide a shift in the text: after a long losing stretch, predictable text
    /// is found by the next re-time and the wide verify then stays chosen.
    func testRegimeChangeIsFoundAfterBackoff() {
        let costs = DFlash2WidthCostTable()
        var chooser = DFlash2WidthChooser(widths: [1, 8, 16], costs: costs)
        let seconds: [Int: Double] = [1: 0.0128, 8: 0.031, 16: 0.033]
        var firstWideAfterShift: Int?
        var wideAfterSettling = 0
        for cycle in 0 ..< 2000 {
            let w = chooser.width
            let predictable = cycle >= 1000
            let accepted = predictable ? w - 2 : 0
            let tokens = w == 1 ? 1 : 1 + min(max(accepted, 0), w - 1)
            if predictable, w > 1, firstWideAfterShift == nil { firstWideAfterShift = cycle }
            if cycle >= 1600, w > 1 { wideAfterSettling += 1 }
            chooser.observe(verifyWidth: w, tokens: tokens, seconds: seconds[w]!)
        }
        let found = try? XCTUnwrap(firstWideAfterShift)
        XCTAssertNotNil(found)
        XCTAssertLessThanOrEqual((found ?? 9999) - 1000, DFlash2WidthChooser.maxProbeInterval + 1)
        XCTAssertGreaterThan(wideAfterSettling, 360, "wide verify not kept on predictable text")
    }

    /// Uno (`UnoWidthPlan`): a cycle commits the seed row's token on top of the accepted drafts, so
    /// a width that accepts nothing still emits 2 tokens. With that bonus, a width costing 1.7 AR
    /// steps beats AR even at zero acceptance; without it the chooser would wrongly pick AR.
    func testBonusTokenCountsTheUnoSeedRow() {
        func settle(bonus: Double) -> Int {
            var chooser = DFlash2WidthChooser(
                widths: [1, 5], costs: DFlash2WidthCostTable(), bonusTokens: bonus)
            let seconds: [Int: Double] = [1: 0.013, 5: 0.022]
            for _ in 0 ..< 40 {
                let w = chooser.width
                chooser.observe(verifyWidth: w, tokens: 1, seconds: seconds[w]!)
            }
            return chooser.width
        }
        XCTAssertEqual(settle(bonus: 1), 5)
        XCTAssertEqual(settle(bonus: 0), 1)
    }

    /// Costs are kept per context bucket: a long context makes wide blocks expensive (vector
    /// attention ends at 8 rows), a short one does not. Each bucket settles on its own width, and a
    /// return to a timed bucket uses its costs at once instead of re-timing.
    func testCostsArePerContextBucket() {
        var chooser = DFlash2WidthChooser(
            widths: [1, 5, 12], costs: DFlash2WidthCostTable(), bonusTokens: 1)
        // Short context: 12 rows cost about what 5 do. Long: 12 rows cost far more.
        let seconds: [Int: [Int: Double]] = [
            0: [1: 0.0122, 5: 0.040, 12: 0.042],
            2: [1: 0.0130, 5: 0.046, 12: 0.100],
        ]
        func run(bucket: Int, cycles: Int) -> [Int: Int] {
            chooser.setCostBucket(bucket)
            var counts: [Int: Int] = [:]
            for _ in 0 ..< cycles {
                let w = chooser.width
                counts[w, default: 0] += 1
                // Nine drafts land, the tenth misses: 5 rows emit 6 tokens, 12 rows emit 11.
                chooser.observe(
                    verifyWidth: w, tokens: w == 1 ? 1 : min(w - 1, 9) + 1,
                    seconds: seconds[bucket]![w]!)
            }
            return counts
        }
        let short = run(bucket: 0, cycles: 200)
        XCTAssertGreaterThan(short[12] ?? 0, 150, "short context: \(short)")
        let long = run(bucket: 2, cycles: 200)
        XCTAssertGreaterThan(long[5] ?? 0, 150, "long context: \(long)")
        chooser.setCostBucket(0)
        XCTAssertEqual(chooser.width, 12, "a timed bucket's costs apply immediately")
    }

    /// Uno shares acceptance evidence across requests on one target: after a request that accepted
    /// well, the next one keeps speculating through an opening cycle that accepts nothing, where a
    /// fresh prior would price speculation below AR.
    func testSharedAcceptanceSurvivesOneMissAtTheNextRequest() {
        let costs = DFlash2WidthCostTable()
        let seconds: [Int: Double] = [1: 0.0122, 5: 0.040]
        func request(_ accepted: (Int) -> Int, cycles: Int, shares: Bool) -> Int {
            var chooser = DFlash2WidthChooser(
                widths: [1, 5], costs: costs, bonusTokens: 1, initial: 5, sharesAcceptance: shares)
            var speculative = 0
            for cycle in 0 ..< cycles {
                let w = chooser.width
                if w > 1 { speculative += 1 }
                chooser.observe(
                    verifyWidth: w, tokens: w == 1 ? 1 : min(w - 1, accepted(cycle)) + 1,
                    seconds: seconds[w]!)
            }
            return speculative
        }
        _ = request({ _ in 4 }, cycles: 60, shares: true)
        // Next request: the first cycle misses, the rest accept fully.
        let shared = request({ $0 == 0 ? 0 : 4 }, cycles: 40, shares: true)
        XCTAssertGreaterThan(shared, 30, "shared evidence should keep speculating")
        let fresh = request({ $0 == 0 ? 0 : 4 }, cycles: 40, shares: false)
        XCTAssertLessThan(fresh, shared, "the fresh prior is what flipped to AR")
    }
}
