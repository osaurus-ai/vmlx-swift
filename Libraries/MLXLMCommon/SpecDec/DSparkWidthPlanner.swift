// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation

/// Measured width pricing for Adaptive (`VMLX_DSPARK_MEASURED_WIDTH=1`, default off; the same lever name as vMLX
/// Python's opt-in measured planner).
///
/// The default width rule is acceptance-only: W = round(ladderTop x acceptance EMA), so a verify row is priced at
/// nothing. On MiniMax-M3.1 a verified row costs 0.33-0.62 of an AR step (10-ENGINE-DELTA-2026-10-29 §0), and the
/// Osaurus GUI proof's short-turn rounds ran at 52-246 ms / token against AR ~30: wide and slow. This planner prices
/// every width from what rounds actually cost and still lets Adaptive choose any width from 1 to the ladder top:
///
/// - expected tokens of a round with d drafts: 1 + sum over j <= d of p_1 ... p_j, with p_j the per-position
///   conditional acceptance (EMA; a position is observed when every earlier draft was accepted);
/// - seconds per round by d: EMA of measured rounds (start to the next step's start, the same span the governor times),
///   each sample clamped to [0.7x, 1.5x] of the previous value (a one-off stall moves it at most 50 %);
/// - unmeasured widths: piecewise-linear between measured neighbours, beyond them along the nearest measured slope
///   (never decreasing), or, with one measured width, a slope of 0.4 AR steps per row (the middle of the measured
///   0.33-0.62);
/// - the width with the most tokens per second, ties within 2 % to fewer rows; every 8th planned round one neighbour of
///   it (alternating +1 / -1) so the curve keeps being measured where it matters.
///
/// No width is capped or pinned: the ladder top stays reachable whenever its price pays.
public struct DSparkWidthPlanner: Sendable, Equatable {
    public static let environmentKey = "VMLX_DSPARK_MEASURED_WIDTH"

    public static var enabledByEnvironment: Bool {
        let raw = (ProcessInfo.processInfo.environment[environmentKey] ?? "0").trimmingCharacters(in: .whitespaces)
            .lowercased()
        return ["1", "true", "on", "yes"].contains(raw)
    }

    public let ladderTop: Int
    public let alpha: Double
    public let prior: Double
    public let exploreEvery: Int
    /// p_j for j = 1 ... ladderTop (index j - 1).
    public private(set) var acceptance: [Double]
    /// Seconds per round by drafts d.
    public private(set) var cost: [Int: Double] = [:]
    private var rounds = 0
    private var exploreUp = true

    public init(ladderTop: Int, alpha: Double = 0.15, prior: Double = 0.7, exploreEvery: Int = 8) {
        self.ladderTop = Swift.max(1, ladderTop)
        self.alpha = alpha
        self.prior = prior
        self.exploreEvery = Swift.max(2, exploreEvery)
        acceptance = Array(repeating: prior, count: Swift.max(1, ladderTop))
    }

    /// A round drafted `drafts` tokens and `accepted` of them matched: positions 1 ... accepted were accepted, position
    /// accepted + 1 (when drafted) was rejected; later positions were not observed.
    public mutating func observe(drafts: Int, accepted: Int) {
        guard drafts > 0 else { return }
        let k = Swift.min(Swift.max(0, accepted), drafts)
        for j in 0 ..< Swift.min(k, ladderTop) { acceptance[j] = (1 - alpha) * acceptance[j] + alpha }
        if k < drafts, k < ladderTop { acceptance[k] = (1 - alpha) * acceptance[k] }
    }

    /// Wall-clock seconds of one round with `drafts` drafts.
    public mutating func observeCost(drafts: Int, seconds: Double) {
        guard drafts > 0, seconds.isFinite, seconds > 0 else { return }
        if let old = cost[drafts] {
            let s = Swift.min(Swift.max(seconds, 0.7 * old), 1.5 * old)
            cost[drafts] = (1 - alpha) * old + alpha * s
        } else {
            cost[drafts] = seconds
        }
    }

    public func expectedTokens(_ drafts: Int) -> Double {
        var total = 1.0, run = 1.0
        for j in 0 ..< Swift.min(drafts, ladderTop) {
            run *= acceptance[j]
            total += run
        }
        return total
    }

    /// Measured or estimated seconds of a round with `drafts` drafts; nil before any round was measured.
    public func estimatedCost(_ drafts: Int, arStep: Double?) -> Double? {
        if let c = cost[drafts] { return c }
        let measured = cost.keys.sorted()
        guard let first = measured.first, let last = measured.last else { return nil }
        if measured.count == 1 {
            let slope = 0.4 * (arStep ?? cost[first]! / Double(first + 1))
            return Swift.max(cost[first]! / 4, cost[first]! + slope * Double(drafts - first))
        }
        if drafts > last {
            let prev = measured[measured.count - 2]
            let slope = Swift.max(0, (cost[last]! - cost[prev]!) / Double(last - prev))
            return cost[last]! + slope * Double(drafts - last)
        }
        if drafts < first {
            let next = measured[1]
            let slope = Swift.max(0, (cost[next]! - cost[first]!) / Double(next - first))
            return Swift.max(cost[first]! / 4, cost[first]! - slope * Double(first - drafts))
        }
        let lo = measured.last { $0 < drafts }!, hi = measured.first { $0 > drafts }!
        let t = Double(drafts - lo) / Double(hi - lo)
        return cost[lo]! + t * (cost[hi]! - cost[lo]!)
    }

    /// The drafts with the most expected tokens per second (ties within 2 % to fewer); nil before any round was
    /// measured (the caller keeps its own rule until then).
    public func best(arStep: Double?) -> Int? {
        var bestD: Int?
        var bestRate = 0.0
        for d in 1 ... ladderTop {
            guard let c = estimatedCost(d, arStep: arStep), c > 0 else { return nil }
            let rate = expectedTokens(d) / c
            if bestD == nil || rate > bestRate * 1.02 {
                bestD = d
                bestRate = rate
            }
        }
        return bestD
    }

    /// The next round's drafts: the best width, or every `exploreEvery`-th round a neighbour of it.
    public mutating func plan(arStep: Double?) -> Int? {
        guard let b = best(arStep: arStep) else { return nil }
        rounds += 1
        guard rounds % exploreEvery == 0 else { return b }
        defer { exploreUp.toggle() }
        let d = exploreUp ? b + 1 : b - 1
        return (1 ... ladderTop).contains(d) ? d : (exploreUp ? b - 1 : b + 1).clamped(1, ladderTop)
    }
}

extension Int {
    fileprivate func clamped(_ lo: Int, _ hi: Int) -> Int { Swift.max(lo, Swift.min(hi, self)) }
}
