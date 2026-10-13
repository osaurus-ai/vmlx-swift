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

    /// Uno over a drifting GPU clock (M5: 1.5 GHz to ~600 MHz within one response). Each width's cost
    /// is refreshed only when that width runs, so speculative costs timed in a slow patch were
    /// compared with AR steps timed after the clock recovered, and the chooser parked on AR
    /// (K2 Uno sampled prose, live trace). World: base costs AR 12.2 ms, width 16 57 ms, a cycle
    /// accepts 5 of 15 → 7 tokens per cycle with the seed row, 1.5x AR at ANY clock. The clock runs
    /// 2x slow for the first 60 decisions, during which acceptance dips to zero for 8 cycles
    /// (correctly sending the chooser to AR); then the clock and acceptance recover.
    /// AR stretches are 8 steps observed together, as `UnoDecoder.observeAR` does.
    private func driftingClock(tracksClock: Bool, rewards: Bool) -> (spec: Int, ar: Int) {
        let costs = DFlash2WidthCostTable(tracksClock: tracksClock)
        var chooser = DFlash2WidthChooser(
            widths: [1, 16], costs: costs, bonusTokens: 1, initial: 16, rewardsWinningProbes: rewards)
        let base: [Int: Double] = [1: 0.0122, 16: 0.057]
        var decision = 0
        var specAfter = 0
        var arAfter = 0
        var specRun = 0
        while decision < 460 {
            let w = chooser.width
            let slow = decision < 60
            let clock = slow ? 2.0 : 1.0
            if w == 1 {
                if !slow { arAfter += 8 }
                chooser.observe(
                    verifyWidth: 1, tokens: 1, seconds: base[1]! * clock, count: 8)
            } else {
                specRun += 1
                let dip = slow && specRun > 20
                if !slow { specAfter += 1 }
                chooser.observe(
                    verifyWidth: 16, tokens: dip ? 1 : 6, seconds: base[16]! * clock)
            }
            decision += 1
        }
        return (specAfter, arAfter)
    }

    func testClockDriftDoesNotParkUnoOnAR() {
        var runs: [String: (spec: Int, ar: Int)] = [:]
        for track in [false, true] {
            for reward in [false, true] {
                runs["track=\(track) reward=\(reward)"] = driftingClock(tracksClock: track, rewards: reward)
            }
        }
        print("UNO_CLOCK \(runs.sorted { $0.key < $1.key })")
        let shipped = runs["track=true reward=true"]!
        let old = runs["track=false reward=false"]!
        // After recovery speculation is 1.5x AR: nearly every decision must speculate.
        XCTAssertGreaterThan(shipped.spec, 360, "shipped: \(shipped)")
        XCTAssertLessThan(old.spec, 100, "control: the old chooser parks on AR: \(old)")
    }

    /// The scale is shared: timing only AR at a new clock moves every width's cost with it, while
    /// each width keeps its cost relative to AR.
    func testClockScaleMovesEveryWidth() {
        let costs = DFlash2WidthCostTable(tracksClock: true)
        for _ in 0 ..< 2 { costs.observe(width: 1, seconds: 0.012) }
        for _ in 0 ..< 2 { costs.observe(width: 16, seconds: 0.057) }
        let ratio = costs.cost(of: 16)! / costs.cost(of: 1)!
        for _ in 0 ..< 40 { costs.observe(width: 1, seconds: 0.024) }
        XCTAssertEqual(costs.cost(of: 1)!, 0.024, accuracy: 0.0015)
        // AR-only timings cannot tell "AR got slower" from "the clock did": most of the change goes
        // to the shared scale (rate 0.25 vs 0.05), a bounded share to AR's own base.
        XCTAssertEqual(costs.cost(of: 16)! / costs.cost(of: 1)!, ratio, accuracy: 0.2 * ratio)
        XCTAssertGreaterThan(costs.cost(of: 16)!, 0.09)
    }

    /// A machine where Uno loses (an older chip: a speculative cycle costs 6.5 AR steps and commits ~4
    /// tokens). Context grows through six cost buckets. With lazy exploration every bucket opens on AR
    /// and untimed widths are priced from estimates, so speculative cycles are only the sparse
    /// re-time probes; the old chooser timed every width in every bucket (plus warm-up cycles).
    private func losingMachine(lazy: Bool) -> (spec: Int, specTokens: Int, arTokens: Int, firstInBucket: [Int]) {
        let costs = DFlash2WidthCostTable(tracksClock: true)
        var chooser = DFlash2WidthChooser(
            widths: [1, 5, 8, 12, 16], costs: costs, bonusTokens: 1, initial: 5,
            sharesAcceptance: true, rewardsWinningProbes: true, decay: 0.95, lazyExploration: lazy)
        var spec = 0, specTokens = 0, arTokens = 0
        var firstInBucket: [Int] = []
        var tokens = 0
        var bucket = 0
        while tokens < 12_000 {
            let b = min(5, tokens / 2_000)
            if b != bucket || tokens == 0 {
                bucket = b
                chooser.setCostBucket(b)
                firstInBucket.append(chooser.width)
            }
            let ar = 0.0125 * (1 + 0.1 * Double(b))
            let w = chooser.width
            if w == 1 {
                chooser.observe(verifyWidth: 1, tokens: 1, seconds: ar, count: 8)
                tokens += 8
                arTokens += 8
            } else {
                let accepted = min(w - 1, 2)
                chooser.observe(verifyWidth: w, tokens: accepted + 1, seconds: ar * (6 + 0.1 * Double(w)))
                tokens += accepted + 2
                spec += 1
                specTokens += accepted + 2
            }
        }
        return (spec, specTokens, arTokens, firstInBucket)
    }

    func testLazyExplorationKeepsALosingMachineOnAR() {
        let lazy = losingMachine(lazy: true)
        let eager = losingMachine(lazy: false)
        print("UNO_LOSING lazy spec=\(lazy.spec) first=\(lazy.firstInBucket) eager spec=\(eager.spec) first=\(eager.firstInBucket)")
        // Every bucket after the first opens on AR (the first opens on AR too: AR is timed first).
        XCTAssertEqual(lazy.firstInBucket, Array(repeating: 1, count: lazy.firstInBucket.count))
        // Speculative cycles: only probes, under 2 % of tokens.
        XCTAssertLessThan(Double(lazy.specTokens) / Double(lazy.specTokens + lazy.arTokens), 0.02, "\(lazy)")
        XCTAssertLessThan(lazy.spec, eager.spec, "control: eager exploration times every width per bucket")
    }

    /// A target-declared ladder replaces the default candidates (AR stays; widths outside 2...2*trained drop).
    func testPlanUsesTargetLadder() {
        guard ProcessInfo.processInfo.environment["VMLX_DFLASH2_BLOCK"] == nil,
            ProcessInfo.processInfo.environment["VMLX_DFLASH2_WIDTHS"] == nil,
            ProcessInfo.processInfo.environment["VMLX_DFLASH2_AR_OPTION"] == nil
        else { return }
        XCTAssertEqual(DFlash2WidthChooser.plan(trained: 8, laneFlat: false), [1, 5, 8, 16])
        XCTAssertEqual(DFlash2WidthChooser.plan(trained: 8, laneFlat: true), [1, 8, 16])
        XCTAssertEqual(
            DFlash2WidthChooser.plan(trained: 8, laneFlat: false, targetLadder: [3, 4, 6, 8, 16, 32, 1]),
            [1, 3, 4, 6, 8, 16])
        XCTAssertEqual(DFlash2WidthChooser.plan(trained: 8, laneFlat: false, targetLadder: []), [1, 5, 8, 16])
    }

    /// AR stretch length is a per-target choice: only a target that declares `DFlash2ARBurstPolicy` gets a pipelined
    /// stretch (GLM-5.3: 4). Everything else runs the serial step; a global default of 4 cost Qwen3.8-27B ~20 % decode.
    func testARBurstIsPerTargetAndOtherwiseAdaptive() {
        struct Declares: DFlash2ARBurstPolicy { var dflash2ARBurstSteps: Int { 4 } }
        struct Silent {}
        unsetenv("VMLX_DFLASH2_AR_BURST")
        XCTAssertEqual(DFlash2TokenIterator.arBurstSteps(for: Declares()), 4)
        XCTAssertNil(DFlash2TokenIterator.arBurstSteps(for: Silent()), "no policy: adaptive stretch")
        setenv("VMLX_DFLASH2_AR_BURST", "3", 1)
        defer { unsetenv("VMLX_DFLASH2_AR_BURST") }
        XCTAssertEqual(DFlash2TokenIterator.arBurstSteps(for: Silent()), 3, "the env switch still overrides for A/B")
        XCTAssertEqual(DFlash2TokenIterator.arBurstSteps(for: Declares()), 3)
    }
}
