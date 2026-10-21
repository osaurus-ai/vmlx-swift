// Port of vMLX Python `dflash2_runtime._BlockChooser` / `_dflash2_block_plan` (commits 39166fc02, b299ab6dc;
// audit R2-20 §3 and R2-21 §5c). Chooses the DFlash2 verify width per cycle.
//
// WHY: a wider verify drafts more tokens but costs one wider target forward. With the lane matmul
// (LaneQMM.swift) 8 and 16 verify rows cost about the same, so width 16 pays on predictable text
// (+51-60 % easy text in Python) while prose, where deep draft positions rarely land, wants 8.
// Fixed widths lose somewhere; the rate-EMA and acceptance-only rules were measured and rejected in Python.
//
// MODEL (expected tokens per measured second, TensorFold's depth objective):
//   E[tokens | w] = 1 + sum_{j=1}^{w-1} prod_{k<=j} p_k, p_k = P(draft position k accepted | 1..k-1 were).
//   p_k is learned from EVERY cycle at ANY width (positions 1..accepted hit, accepted+1 missed if it was
//   verified), counts decay x0.85 per cycle; a position with little data leans on the previous estimate
//   (one pseudo-trial, prior 0.7), so width-8 cycles teach width 16's first 7 positions.
//   Cost per width = EMA (0.25) of measured cycle seconds, shared process-wide per target (a model+kernel
//   property, not a text property); the first cycle at each width in the process is discarded (kernel
//   compile / warm-up). The non-current width measured longest ago is re-timed after 32 cycles; a re-time
//   that leaves the decision unchanged doubles that interval (up to 512), one that changes it resets it to 32.
//   Acceptance evidence decays only on cycles that verified a draft position (AR cycles carry none).
// Correctness: width only changes how many drafts are verified, never the acceptance rule.

import Foundation

final class DFlash2WidthCostTable: @unchecked Sendable {
    private struct Key: Hashable {
        let width: Int
        let bucket: Int
    }

    private let lock = NSLock()
    private var cost: [Key: Double] = [:]
    private var warm: Set<Int> = []

    /// Clock-tracking mode (Uno). The M5 GPU clock drifts by 2x within one response (1.5 GHz to
    /// ~600 MHz under sustained load), and each width's cost was refreshed only when that width ran,
    /// so the chooser compared an AR step timed in a slow moment against a cycle timed in a fast one
    /// (or the reverse) and parked on the wrong side: K2 Uno, sampled prose, AR priced at 23.9 ms
    /// while it ran at 16.8; once on AR, the speculative costs stayed frozen. Here
    /// `cost = base[width] × scale`: `scale` is one shared slowdown factor that EVERY observation
    /// moves (fast EMA), `base` is each width's cost relative to it (slow EMA). Decisions compare
    /// `base` ratios, which the clock does not move, and an AR re-time is paired with the
    /// speculative cycles just before it.
    let tracksClock: Bool
    private var scale = 1.0
    static let scaleRate = 0.25
    static let baseRate = 0.05

    init(tracksClock: Bool = false) {
        self.tracksClock = tracksClock
    }

    /// `bucket` separates costs that depend on more than the width (Uno: context length, see
    /// `UnoWidthPlan.contextBucket`). The warm-up discard is per width: it is kernel compilation.
    func observe(width: Int, seconds: Double, bucket: Int = 0) {
        lock.lock()
        defer { lock.unlock() }
        // AR (width 1) runs the ordinary decode kernels the request already compiled.
        if !warm.contains(width), !(tracksClock && width == 1) {
            warm.insert(width)
            return
        }
        let key = Key(width: width, bucket: bucket)
        if tracksClock {
            guard let base = cost[key] else {
                cost[key] = seconds / scale
                return
            }
            // One outlier (a compile, a page fault) moves the scale by at most 2x.
            let ratio = Swift.min(Swift.max(seconds / (base * scale), 0.5), 2)
            scale *= 1 + Self.scaleRate * (ratio - 1)
            cost[key] = base + Self.baseRate * (seconds / scale - base)
            return
        }
        if let c = cost[key] {
            cost[key] = c + 0.25 * (seconds - c)
        } else {
            cost[key] = seconds
        }
    }

    private var warmupSteps: Set<String> = []

    /// True exactly once per warm-up step per table (per living target and namespace): the caller
    /// runs that step then (`UnoWarmup.prepare(step:...)`).
    func claimWarmupStep(_ step: String) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return warmupSteps.insert(step).inserted
    }

    /// Widths whose kernels are already compiled (`UnoDecoder.prewarm` at model load): their
    /// first timing is kept instead of discarded.
    func markWarm(_ widths: [Int]) {
        lock.lock()
        defer { lock.unlock() }
        warm.formUnion(widths)
    }

    func isMeasured(width: Int, bucket: Int) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return cost[Key(width: width, bucket: bucket)] != nil
    }

    /// Cost of `width` in `bucket` for a chooser that explores lazily (clock-tracking tables only):
    /// measured; else the same width measured in the nearest bucket where AR was measured too,
    /// scaled by AR's cost ratio between the two buckets (attention grows with context for every
    /// width alike); else the nearest measured speculative width in this bucket (optimistic for a
    /// wider one, so a width that might win still gets tried). Nil: nothing to estimate from.
    func estimatedCost(of width: Int, bucket: Int) -> Double? {
        lock.lock()
        defer { lock.unlock() }
        let s = tracksClock ? scale : 1
        if let c = cost[Key(width: width, bucket: bucket)] { return c * s }
        if let arHere = cost[Key(width: 1, bucket: bucket)] {
            let others = cost.keys.filter { $0.width == width && $0.bucket != bucket }
                .filter { cost[Key(width: 1, bucket: $0.bucket)] != nil }
                .sorted { abs($0.bucket - bucket) < abs($1.bucket - bucket) }
            if let near = others.first, let c = cost[near], let arThere = cost[Key(width: 1, bucket: near.bucket)],
                arThere > 0
            {
                return c * (arHere / arThere) * s
            }
        }
        guard width > 1 else { return nil }
        let sameBucket = cost.keys.filter { $0.bucket == bucket && $0.width > 1 }
            .sorted { abs($0.width - width) < abs($1.width - width) }
        return sameBucket.first.flatMap { cost[$0] }.map { $0 * s }
    }

    func cost(of width: Int, bucket: Int = 0) -> Double? {
        lock.lock()
        defer { lock.unlock() }
        return cost[Key(width: width, bucket: bucket)].map { tracksClock ? $0 * scale : $0 }
    }

    /// Per-position acceptance evidence (decayed trials and hits) left by the latest chooser that
    /// shares it (`DFlash2WidthChooser(sharesAcceptance:)`).
    private var acceptance: (trials: [Double], hits: [Double])?

    func storeAcceptance(trials: [Double], hits: [Double]) {
        lock.lock()
        defer { lock.unlock() }
        acceptance = (trials, hits)
    }

    func storedAcceptance(count: Int) -> (trials: [Double], hits: [Double])? {
        lock.lock()
        defer { lock.unlock() }
        guard let acceptance, acceptance.trials.count == count else { return nil }
        return acceptance
    }

    private final class Entry {
        weak var target: AnyObject?
        let table: DFlash2WidthCostTable

        init(target: AnyObject, table: DFlash2WidthCostTable) {
            self.target = target
            self.table = table
        }
    }

    private struct TableKey: Hashable {
        let target: ObjectIdentifier
        let namespace: String
    }

    nonisolated(unsafe) private static var tables: [TableKey: Entry] = [:]
    private static let tablesLock = NSLock()

    /// One table per living target and decoding method (`namespace`): a DFlash2 cycle and an Uno
    /// cycle of the same width are different work.
    static func shared(
        for target: AnyObject, widths: [Int], namespace: String = "dflash2", tracksClock: Bool = false
    ) -> DFlash2WidthCostTable {
        tablesLock.lock()
        defer { tablesLock.unlock() }
        let key = TableKey(target: ObjectIdentifier(target), namespace: namespace)
        // ObjectIdentifier can be reused after a model unloads. Preserve learned costs only
        // for the same living target, without keeping its weights resident.
        if let entry = tables[key], entry.target === target { return entry.table }
        let t = DFlash2WidthCostTable(tracksClock: tracksClock)
        tables[key] = Entry(target: target, table: t)
        return t
    }
}

/// A DFlash 2 target that measured a better candidate ladder than the default (min(5, trained), trained,
/// 2*trained) declares it here (speculative widths only; AR is added unless `VMLX_DFLASH2_AR_OPTION=0`).
public protocol DFlash2WidthLadderPolicy {
    var dflash2WidthLadder: [Int] { get }
}

/// A DFlash 2 target whose AR stretches measured faster pipelined declares the steps per stretch here. Targets that
/// do not declare it run the serial single step: the pipelined stretch was measured on GLM-5.3 (+4 % prose) and cost
/// Qwen3.8-27B ~20 % decode on low-acceptance text (JANGH2 and JANG_4D, alternating A/B at matched GPU clock).
public protocol DFlash2ARBurstPolicy {
    var dflash2ARBurstSteps: Int { get }
}

struct DFlash2WidthChooser {
    static let decay = 0.85
    static let probeEvery = 32
    /// Upper bound for the re-time interval after repeated confirming probes.
    static let maxProbeInterval = 512
    static let priorP = 0.7

    let widths: [Int]
    private(set) var width: Int
    private var trials: [Double]
    private var hits: [Double]
    private var cycles = 0
    private var lastTimed: [Int: Int] = [:]
    /// Cycles between re-times. A probe that leaves the decision unchanged doubles it (a drafter
    /// that loses keeps losing; each probe of a losing width costs several AR steps); a probe that
    /// changes the decision resets it, so a shift in the text is still found within 32 cycles of
    /// the next probe.
    private var probeInterval = DFlash2WidthChooser.probeEvery
    private var nextProbe = DFlash2WidthChooser.probeEvery
    private var probing: (width: Int, from: Int)?
    private let costs: DFlash2WidthCostTable
    /// Tokens a speculative cycle commits beyond `expectedTokens` (Uno: the seed row's own token).
    private let bonusTokens: Double
    /// Cost bucket of the coming cycle (`DFlash2WidthCostTable.observe(width:seconds:bucket:)`).
    private(set) var costBucket = 0
    /// Start from, and keep updating, the acceptance evidence stored in `costs`, so a request opens
    /// with what this target has been accepting rather than the prior.
    private let sharesAcceptance: Bool
    /// A probe that beats the width it left earns the next decision (Uno; see `observe`).
    private let rewardsWinningProbes: Bool
    /// Per-cycle decay of acceptance evidence. Uno uses 0.95 (~20 cycles of memory): its
    /// acceptance is bursty (runs of 0-2 then 15/15 on the same text), and with 0.85 (~7 cycles) a
    /// few low cycles priced speculation under AR on text where every width beats AR. Replay of
    /// recorded K2 width-16 acceptance (any width w accepts min(a16, w-1): the noise rows are
    /// causal), spec costs 1.0/1.2/1.4x: on_invoice 109/91/84 tok/s at 0.85, 112/94/82 at 0.95
    /// (best fixed width 106/88/76, AR 78).
    private let decay: Double
    /// Lazy exploration (Uno): AR is timed first in every cost bucket, and untimed speculative
    /// widths are priced by `DFlash2WidthCostTable.estimatedCost` instead of each being forced
    /// through a cycle. A width that cannot beat AR on its estimate is never tried outside the
    /// periodic re-time, so on a machine or text where speculation loses, entering a new context
    /// bucket costs nothing over ordinary decoding (before: one cycle per width per bucket, plus a
    /// discarded warm-up cycle per width).
    private let lazyExploration: Bool
    /// The speculative width timed first when no speculative cost exists anywhere yet.
    private let initialSpeculative: Int

    /// Candidate widths: (trained, 2*trained) when the target's verify cost is lane-flat, else
    /// (min(5, trained), trained, 2*trained). `VMLX_DFLASH2_BLOCK=<n>` pins one width (A/B tool).
    /// DSpark candidate widths: narrow-to-trained only. The drafter is trained for exactly `trained − 1` drafts, so
    /// wider blocks are out of distribution; and on MoE targets the verify cost grows fast with rows, so narrow widths
    /// must stay available. MiniCPM-V 4.7 35B-A3B (fixed-width sweep, 1.6 GHz): prose AR 91 / w3 101 / w5 83 / w8 57 /
    /// w16 52; code 91 / 135 / 133 / 99 / 102; easy prose 90 / 148 / 177 / 169 / 258 tok/s. Width 12 never won.
    static func dsparkLadder(trained: Int) -> [Int] {
        Array(Set([3, 5, 8, trained].filter { $0 >= 2 && $0 <= trained })).sorted()
    }

    static func plan(trained: Int, laneFlat: Bool, targetLadder: [Int]? = nil) -> [Int] {
        let t = max(2, trained)
        if let raw = ProcessInfo.processInfo.environment["VMLX_DFLASH2_BLOCK"], let n = Int(raw) {
            return [max(2, min(n, 2 * t))]
        }
        // `VMLX_DFLASH2_WIDTHS=1,4,8,16`: an explicit candidate ladder (A/B tool; 1 = AR; `0` disables the chooser).
        if let raw = ProcessInfo.processInfo.environment["VMLX_DFLASH2_WIDTHS"] {
            let list = Array(Set(raw.split(separator: ",").compactMap { Int($0) }.filter { $0 >= 1 && $0 <= 2 * t }))
                .sorted()
            if !list.isEmpty { return list }
        }
        // Width 1 = a plain AR step, measured like any other width: when speculation emits fewer
        // tokens per second than AR (27B JANGH2 sampled prose: 24 vs 29 tok/s), the chooser
        // picks AR; decayed acceptance drifting back to the prior plus the periodic re-time
        // re-probes speculation. `VMLX_DFLASH2_AR_OPTION=0` removes it.
        let ar = ProcessInfo.processInfo.environment["VMLX_DFLASH2_AR_OPTION"] == "0" ? [] : [1]
        if let ladder = targetLadder?.filter({ $0 >= 2 && $0 <= 2 * t }), !ladder.isEmpty {
            return ar + Array(Set(ladder)).sorted()
        }
        if laneFlat { return ar + [t, 2 * t] }
        return ar + Array(Set([min(5, t), t, 2 * t])).sorted()
    }

    init(
        widths: [Int], costs: DFlash2WidthCostTable, bonusTokens: Double = 0, initial: Int? = nil,
        sharesAcceptance: Bool = false, rewardsWinningProbes: Bool = false,
        decay: Double = DFlash2WidthChooser.decay, lazyExploration: Bool = false
    ) {
        self.decay = decay
        self.lazyExploration = lazyExploration
        self.widths = widths.sorted()
        self.width = initial.flatMap { w in widths.contains(w) ? w : nil }
            ?? self.widths.first(where: { $0 > 1 }) ?? self.widths[0]
        self.bonusTokens = bonusTokens
        self.sharesAcceptance = sharesAcceptance
        self.rewardsWinningProbes = rewardsWinningProbes
        let top = self.widths.last ?? 2
        if sharesAcceptance, let stored = costs.storedAcceptance(count: top) {
            self.trials = stored.trials
            self.hits = stored.hits
        } else {
            self.trials = Array(repeating: 0, count: top)
            self.hits = Array(repeating: 0, count: top)
        }
        self.costs = costs
        self.initialSpeculative = initial.flatMap { w in widths.contains(w) && w > 1 ? w : nil }
            ?? self.widths.first(where: { $0 > 1 }) ?? self.widths[0]
        if lazyExploration, let first = untimedWidth() { width = first }
    }

    /// Cost the decision uses for width `w` in the current bucket.
    private func costOf(_ w: Int) -> Double? {
        lazyExploration
            ? costs.estimatedCost(of: w, bucket: costBucket) : costs.cost(of: w, bucket: costBucket)
    }

    /// The width that must be timed before rates can be compared, if any.
    private func untimedWidth() -> Int? {
        if lazyExploration {
            if widths.contains(1), !costs.isMeasured(width: 1, bucket: costBucket) { return 1 }
            if widths.contains(where: { costOf($0) == nil }) {
                return costOf(initialSpeculative) == nil
                    ? initialSpeculative : widths.first { costOf($0) == nil }
            }
            return nil
        }
        return widths.first(where: { costs.cost(of: $0, bucket: costBucket) == nil })
    }

    private func bestWidth() -> Int {
        widths.filter { rate($0) != nil }.max { rate($0)! < rate($1)! } ?? width
    }

    func expectedTokens(_ w: Int) -> Double {
        var e = 1.0
        var run = 1.0
        var prev = Self.priorP
        if w > 1 {
            for k in 1 ..< w {
                prev = (hits[k] + prev) / (trials[k] + 1.0)
                run *= prev
                e += run
            }
        }
        return e
    }

    /// Expected tokens per second at width `w` in the current cost bucket (nil: not yet timed).
    func rate(_ w: Int) -> Double? {
        guard let c = costOf(w), c > 0 else { return nil }
        return (w > 1 ? expectedTokens(w) + bonusTokens : 1) / c
    }

    /// Trace: "w:rate/expected/cost" per width (rate nil = untimed in this bucket).
    var traceDescription: String {
        widths.map { w in
            let c = costOf(w)
            let measured = costs.isMeasured(width: w, bucket: costBucket) ? "" : "~"
            // CVarArg context otherwise leaves the AR literal as an Int for a %f conversion.
            let expected: Double = w > 1 ? expectedTokens(w) + bonusTokens : 1
            return String(
                format: "%d:%@/%.2f/%@%@", w, rate(w).map { String(format: "%.0f", $0) } ?? "-",
                expected, c == nil ? "" : measured,
                c.map { String(format: "%.1f", $0 * 1000) } ?? "-")
        }.joined(separator: " ") + " next=\(width) bucket=\(costBucket)"
    }

    /// Move to cost bucket `bucket` before a cycle. Widths not yet timed there are timed first;
    /// otherwise the best width by that bucket's costs is chosen.
    mutating func setCostBucket(_ bucket: Int) {
        guard bucket != costBucket else { return }
        costBucket = bucket
        probing = nil
        width = untimedWidth() ?? bestWidth()
    }

    /// One finished cycle: `tokens` emitted (accepted + 1) in `seconds` wall time at `verifyWidth` rows.
    /// `count` > 1: that many identical AR steps timed together (one cost observation, `count` cycles
    /// toward the next re-time).
    mutating func observe(verifyWidth: Int, tokens: Int, seconds: Double, count: Int = 1) {
        guard seconds > 0, widths.contains(verifyWidth) else { return }  // clipped final block
        let accepted = tokens - 1
        let lastVerified = min(accepted + 1, verifyWidth - 1)
        // Decay ages evidence by the cycles that produced new evidence. An AR cycle (width 1)
        // verifies no draft position; decaying on it erased what was learned and pulled the
        // estimate back to the 0.7 prior after ~15 AR steps, so a low-acceptance drafter was
        // re-tried every ~6 cycles instead of at the `probeEvery` re-time (K2/Diplodocus prose:
        // 93 DFlash cycles among 548). Re-exploration still comes from the periodic probe.
        if lastVerified >= 1 {
            for k in 1 ..< trials.count {
                trials[k] *= decay
                hits[k] *= decay
            }
        }
        if lastVerified >= 1 {
            for k in 1 ... lastVerified where k < trials.count {
                trials[k] += 1
                if k <= accepted { hits[k] += 1 }
            }
        }
        if sharesAcceptance, lastVerified >= 1 { costs.storeAcceptance(trials: trials, hits: hits) }
        costs.observe(width: verifyWidth, seconds: seconds, bucket: costBucket)
        cycles += count
        lastTimed[verifyWidth] = cycles
        guard widths.count > 1 else { return }
        if let unseen = untimedWidth() {
            width = unseen
            return
        }
        let best = bestWidth()
        if let probe = probing, probe.width == verifyWidth {
            // A probe that itself ran faster than the width it left earns the next decision too.
            // While the chooser sits on AR, a probe is the only source of acceptance evidence, and
            // one good cycle moves a decayed estimate only part of the way: with the backoff alone,
            // recovered acceptance after a dip was found 1-2 probes per 100+ tokens (K2 Uno,
            // sampled prose). A losing drafter's probes lose, so its backoff is unchanged.
            let realized = Double(verifyWidth > 1 ? tokens : 1)
                + (verifyWidth > 1 ? bonusTokens : 0)
            let won = best != probe.from
                || (rewardsWinningProbes
                    && (rate(probe.from).map { realized / seconds > $0 } ?? false))
            probeInterval = won ? Self.probeEvery : min(probeInterval * 2, Self.maxProbeInterval)
            probing = nil
            nextProbe = best == probe.from && won ? cycles : cycles + probeInterval
        }
        if probing == nil, cycles >= nextProbe {
            let others = widths.filter { $0 != best }
            let probe: Int
            if lazyExploration, best == 1,
                let likeliest = others.filter({ rate($0) != nil }).max(by: { rate($0)! < rate($1)! })
            {
                // On AR, re-time the speculative width most likely to win (by its estimate), not
                // the one timed longest ago: at long context that was often a 12-16-row block that
                // cost 230-455 ms per probe on K2 and could not win anyway.
                probe = likeliest
            } else {
                probe = others.min { (lastTimed[$0] ?? -1) < (lastTimed[$1] ?? -1) } ?? best
            }
            probing = (probe, best)
            width = probe
        } else {
            width = best
        }
    }
}
