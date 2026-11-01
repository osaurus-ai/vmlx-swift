// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// Measured width pricing (`DSparkWidthPlanner`, `VMLX_DSPARK_MEASURED_WIDTH=1`): Adaptive picks the width with the most
/// expected tokens per measured second, any width from 1 to the ladder top (no cap), and greedy text never changes.
@Suite("DSpark measured width pricing")
struct DSparkWidthPlannerTests {
    typealias P = DSparkWidthPlanner

    static func planner(p: Double, costs: [Int: Double], top: Int = 14) -> P {
        var w = P(ladderTop: top, alpha: 1, prior: p)
        for (d, c) in costs { w.observeCost(drafts: d, seconds: c) }
        return w
    }

    @Test("Best width == brute force over the same estimates; estimates are monotone and pass through the measured points")
    func bestIsTheArgmax() throws {
        // A convex verify curve (AR 30 ms; rows 10 ms, growing): measured at 3, 6, 9.
        let costs = [3: 0.068, 6: 0.112, 9: 0.170]
        for p in [0.5, 0.7, 0.85, 0.95] {
            let w = Self.planner(p: p, costs: costs)
            let est = (1 ... 14).map { w.estimatedCost($0, arStep: 0.030)! }
            #expect(zip(est, est.dropFirst()).allSatisfy { $0 <= $1 }, "p \(p): estimates not monotone \(est)")
            for (d, c) in costs { #expect(abs(w.estimatedCost(d, arStep: 0.030)! - c) < 1e-12) }
            let rates = (1 ... 14).map { w.expectedTokens($0) / est[$0 - 1] }
            let brute = rates.indices.max { rates[$0] < rates[$1] }! + 1
            let best = try #require(w.best(arStep: 0.030))
            // Ties within 2 % go to fewer rows, so the pick is the brute-force argmax or a cheaper width within 2 %.
            #expect(best <= brute && rates[best - 1] * 1.02 >= rates[brute - 1], "p \(p): best \(best), brute \(brute)")
            print("[width planner] p \(p): best \(best) drafts, tokens/s \(String(format: "%.1f", rates[best - 1]))")
        }
    }

    @Test("No cap: flat cost and high acceptance reach the ladder top; expensive rows stay narrow")
    func reachesTheTopWhenItPays() throws {
        let flat = Self.planner(p: 0.97, costs: [2: 0.050, 10: 0.050])
        #expect(flat.best(arStep: 0.030) == 14)
        let steep = Self.planner(p: 0.6, costs: [2: 0.050, 4: 0.090])
        #expect(try #require(steep.best(arStep: 0.030)) <= 2)
        // Nothing measured: no opinion (the decoder keeps the acceptance rule).
        #expect(P(ladderTop: 14).best(arStep: 0.03) == nil)
    }

    @Test("Acceptance per position: accepted positions rise, the first rejected falls, later ones are unobserved")
    func acceptanceUpdates() {
        var w = P(ladderTop: 6, alpha: 0.5, prior: 0.5)
        w.observe(drafts: 5, accepted: 2)
        #expect(w.acceptance == [0.75, 0.75, 0.25, 0.5, 0.5, 0.5])
        w.observe(drafts: 3, accepted: 3)
        #expect(w.acceptance[0 ..< 3] == [0.875, 0.875, 0.625] && w.acceptance[3] == 0.5)
    }

    @Test("Exploration: every 8th planned round tries a neighbour, alternating sides; clamped samples")
    func exploration() {
        // 30 ms + 6 ms per draft at p 0.8: the best width is in the middle of the ladder (3).
        var w = Self.planner(p: 0.8, costs: [3: 0.048, 6: 0.066, 9: 0.084])
        let b = w.best(arStep: 0.03)!
        #expect(b > 1 && b < 14)
        let plans = (0 ..< 32).map { _ in w.plan(arStep: 0.03)! }
        #expect(plans.enumerated().filter { ($0.offset + 1) % 8 != 0 }.allSatisfy { $0.element == b })
        let probes = plans.enumerated().filter { ($0.offset + 1) % 8 == 0 }.map(\.element)
        #expect(probes == [b + 1, b - 1, b + 1, b - 1], "probes \(probes)")
        var c = P(ladderTop: 4, alpha: 1, prior: 0.7)
        c.observeCost(drafts: 2, seconds: 0.05)
        c.observeCost(drafts: 2, seconds: 0.5)  // a stall: clamped to 1.5x
        #expect(abs(c.cost[2]! - 0.075) < 1e-12)
    }

    @Test("Greedy DSpark == AR with measured widths (text never depends on the width), and the widths vary")
    func greedyIdentityWithMeasuredWidths() throws {
        typealias DS = MiniMaxM3DSparkTests
        let directory = DS.temporaryDirectory("planner")
        defer { try? FileManager.default.removeItem(at: directory) }
        try DS.writeReleaseDrafter(directory)
        try Device.withDefaultDevice(.gpu) {
            let model = try DS.attachedModel(directory)
            let prompt = DS.ids(773, seed: 11)
            let (reference, _) = DS.greedyAR(model, prompt, count: 45)
            let inner = try #require(model.makeDSparkDraftSession() as? MiniMaxM3DSparkSession)
            let scripted = DS.ScriptedSession(inner: inner, reference: prompt + reference) { $0 % 5 == 4 }
            // A synthetic clock: a round costs 30 ms + 12 ms per draft, so widths have a price.
            final class Clock: @unchecked Sendable { var t = 0.0 }
            let clock = Clock()
            var decoder = DSparkDecoder(
                session: scripted, blockSize: DS.draftBlock, stopTokenIDs: [], governor: DS.alwaysSpeculate,
                ladderTop: model.dsparkLadderTop, pinnedWidth: nil, measuredWidth: true,
                clock: { clock.t += 0.001; return clock.t })
            var rows: [Int] = []
            var paths: [MiniMaxM3MSA.Path] = []
            let (out, _) = DS.dsparkRun(model, decoder: &decoder, prompt: prompt, count: 45, rows: &rows, msaPaths: &paths)
            #expect(out == reference, "DSpark with measured widths != AR")
            #expect(decoder.planner != nil && !(decoder.planner?.cost.isEmpty ?? true))
            print("[width planner] greedy == AR \(out == reference); widths \(Dictionary(grouping: scripted.widths, by: { $0 }).mapValues(\.count))")
        }
    }
}
