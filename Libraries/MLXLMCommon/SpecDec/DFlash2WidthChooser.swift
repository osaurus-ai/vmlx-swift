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

    /// `bucket` separates costs that depend on more than the width (Uno: context length, see
    /// `UnoWidthPlan.contextBucket`). The warm-up discard is per width: it is kernel compilation.
    func observe(width: Int, seconds: Double, bucket: Int = 0) {
        lock.lock()
        defer { lock.unlock() }
        if !warm.contains(width) {
            warm.insert(width)
            return
        }
        let key = Key(width: width, bucket: bucket)
        if let c = cost[key] {
            cost[key] = c + 0.25 * (seconds - c)
        } else {
            cost[key] = seconds
        }
    }

    func cost(of width: Int, bucket: Int = 0) -> Double? {
        lock.lock()
        defer { lock.unlock() }
        return cost[Key(width: width, bucket: bucket)]
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
    static func shared(for target: AnyObject, widths: [Int], namespace: String = "dflash2")
        -> DFlash2WidthCostTable
    {
        tablesLock.lock()
        defer { tablesLock.unlock() }
        let key = TableKey(target: ObjectIdentifier(target), namespace: namespace)
        // ObjectIdentifier can be reused after a model unloads. Preserve learned costs only
        // for the same living target, without keeping its weights resident.
        if let entry = tables[key], entry.target === target { return entry.table }
        let t = DFlash2WidthCostTable()
        tables[key] = Entry(target: target, table: t)
        return t
    }
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

    /// Candidate widths: (trained, 2*trained) when the target's verify cost is lane-flat, else
    /// (min(5, trained), trained, 2*trained). `VMLX_DFLASH2_BLOCK=<n>` pins one width (A/B tool).
    static func plan(trained: Int, laneFlat: Bool) -> [Int] {
        let t = max(2, trained)
        if let raw = ProcessInfo.processInfo.environment["VMLX_DFLASH2_BLOCK"], let n = Int(raw) {
            return [max(2, min(n, 2 * t))]
        }
        // Width 1 = a plain AR step, measured like any other width: when speculation emits fewer
        // tokens per second than AR (27B JANGH2 sampled prose: 24 vs 29 tok/s), the chooser
        // picks AR; decayed acceptance drifting back to the prior plus the periodic re-time
        // re-probes speculation. `VMLX_DFLASH2_AR_OPTION=0` removes it.
        let ar = ProcessInfo.processInfo.environment["VMLX_DFLASH2_AR_OPTION"] == "0" ? [] : [1]
        if laneFlat { return ar + [t, 2 * t] }
        return ar + Array(Set([min(5, t), t, 2 * t])).sorted()
    }

    init(
        widths: [Int], costs: DFlash2WidthCostTable, bonusTokens: Double = 0, initial: Int? = nil,
        sharesAcceptance: Bool = false
    ) {
        self.widths = widths.sorted()
        self.width = initial.flatMap { w in widths.contains(w) ? w : nil }
            ?? self.widths.first(where: { $0 > 1 }) ?? self.widths[0]
        self.bonusTokens = bonusTokens
        self.sharesAcceptance = sharesAcceptance
        let top = self.widths.last ?? 2
        if sharesAcceptance, let stored = costs.storedAcceptance(count: top) {
            self.trials = stored.trials
            self.hits = stored.hits
        } else {
            self.trials = Array(repeating: 0, count: top)
            self.hits = Array(repeating: 0, count: top)
        }
        self.costs = costs
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
        guard let c = costs.cost(of: w, bucket: costBucket), c > 0 else { return nil }
        return (w > 1 ? expectedTokens(w) + bonusTokens : 1) / c
    }

    /// Move to cost bucket `bucket` before a cycle. Widths not yet timed there are timed first;
    /// otherwise the best width by that bucket's costs is chosen.
    mutating func setCostBucket(_ bucket: Int) {
        guard bucket != costBucket else { return }
        costBucket = bucket
        probing = nil
        if let unseen = widths.first(where: { costs.cost(of: $0, bucket: bucket) == nil }) {
            width = unseen
        } else {
            width = widths.max { rate($0)! < rate($1)! } ?? width
        }
    }

    /// One finished cycle: `tokens` emitted (accepted + 1) in `seconds` wall time at `verifyWidth` rows.
    mutating func observe(verifyWidth: Int, tokens: Int, seconds: Double) {
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
                trials[k] *= Self.decay
                hits[k] *= Self.decay
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
        cycles += 1
        lastTimed[verifyWidth] = cycles
        guard widths.count > 1 else { return }
        if let unseen = widths.first(where: { costs.cost(of: $0, bucket: costBucket) == nil }) {
            width = unseen
            return
        }
        let best = widths.max { rate($0)! < rate($1)! } ?? width
        if let probe = probing, probe.width == verifyWidth {
            probeInterval = best == probe.from
                ? min(probeInterval * 2, Self.maxProbeInterval) : Self.probeEvery
            probing = nil
            nextProbe = cycles + probeInterval
        }
        if probing == nil, cycles >= nextProbe {
            let others = widths.filter { $0 != best }
            let probe = others.min { (lastTimed[$0] ?? -1) < (lastTimed[$1] ?? -1) } ?? best
            probing = (probe, best)
            width = probe
        } else {
            width = best
        }
    }
}
