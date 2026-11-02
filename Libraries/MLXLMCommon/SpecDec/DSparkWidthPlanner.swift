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

/// Adaptive width priced per verify row: vMLX Python's default rule for a laddered drafter (`dspark/generate.py`
/// `ARFallback.planned_width` + `choose_width`).
///
/// The acceptance rule, W = round(ladderTop x acceptance EMA), charges nothing for a verify row, so a ladder of 14 at
/// 0.5 acceptance drafts 7. In the Osaurus app on MiniMax-M3.1 that drafted 6.8-8.65 at 0.40-0.59 acceptance, where
/// MLX Studio on the same machine and sampling drafted 3.9-5.7 at 0.52-0.78.
///
/// Here each position keeps a conditional acceptance EMA (prior 0.7, alpha 0.3; a position is observed when every
/// earlier draft was accepted) and a round drafts the k maximizing E[tokens | k] / (1 + costPerRow x k), with
/// E[tokens | k] = 1 + sum over j <= k of p_1 ... p_j. Every width up to the ladder top stays reachable, none is capped
/// or pinned, and the output doesn't depend on it. `VMLX_DSPARK_PRICED_WIDTH=0` keeps the acceptance rule.
public struct DSparkPricedWidth: Sendable, Equatable {
    public static let environmentKey = "VMLX_DSPARK_PRICED_WIDTH"
    /// Python's default when a bundle stamps no `cost_per_row`.
    public static let defaultCostPerRow = 0.12

    public static var enabledByEnvironment: Bool {
        let raw = (ProcessInfo.processInfo.environment[environmentKey] ?? "1").trimmingCharacters(in: .whitespaces)
            .lowercased()
        return !["0", "false", "off", "no"].contains(raw)
    }

    public let ladderTop: Int
    public let costPerRow: Double
    public let alpha: Double
    /// p_j for j = 1 ... ladderTop (index j - 1).
    public private(set) var acceptance: [Double]

    public init(ladderTop: Int, costPerRow: Double = defaultCostPerRow, alpha: Double = 0.3, prior: Double = 0.7) {
        self.ladderTop = Swift.max(1, ladderTop)
        self.costPerRow = costPerRow
        self.alpha = alpha
        acceptance = Array(repeating: prior, count: Swift.max(1, ladderTop))
    }

    /// A round drafted `drafts` tokens and `accepted` of them matched: positions 1 ... accepted were accepted, position
    /// accepted + 1 (when drafted) was rejected, and later positions were not observed.
    public mutating func observe(drafts: Int, accepted: Int) {
        guard drafts > 0 else { return }
        let k = Swift.min(Swift.max(0, accepted), drafts)
        for j in 0 ..< Swift.min(k + 1, Swift.min(drafts, ladderTop)) {
            let hit = j < k ? 1.0 : 0.0
            acceptance[j] += alpha * (hit - acceptance[j])
        }
    }

    /// The next round's drafts.
    public var width: Int { Self.choose(acceptance, ladderTop: ladderTop, costPerRow: costPerRow) }

    /// The first k with the most expected tokens per priced round.
    public static func choose(_ acceptance: [Double], ladderTop: Int, costPerRow: Double) -> Int {
        var best = 0.0, bestK = 1, run = 1.0, expected = 1.0
        for k in stride(from: 1, through: Swift.min(ladderTop, acceptance.count), by: 1) {
            run *= acceptance[k - 1]
            expected += run
            let value = expected / (1 + costPerRow * Double(k))
            if value > best {
                best = value
                bestK = k
            }
        }
        return bestK
    }
}
