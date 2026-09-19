// The disk prefix cache's eviction POLICY, tested as a pure function.
//
// Today's order is "oldest recency first, on every store, to exactly the cap". It
// is blind to conversations: once the cache is full, an old chat's resume point
// (its longest snapshot) is pushed out by other chats' superseded snapshots, and
// every single store is an evicting pass.
//
// `DiskQuotaPlanner` orders by what a row is worth: rows nobody will resume from
// go first and pay for the hysteresis; a conversation's tip goes only while the
// total is still over the cap. These tests pin every sentence of that policy,
// its no-regression guarantee for caches migrated from the old schema (all rows
// chain-less -> the old oldest-first answer), and a scenario replay against
// today's order as the control.
//
// No MLX, no files, no SQLite: nothing here needs `MLXMetalTestLock`.

import Foundation
import Testing

@testable import MLXLMCommon

// MARK: - Helpers

/// SplitMix64 — a seeded generator so every random corpus is reproducible.
private struct SeededRNG: RandomNumberGenerator {
    var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}

private func row(
    _ id: String, tokens: Int, bytes: Int64? = nil, recency: Double,
    chain: String? = nil, stable: Bool = false, legacy: Bool = false
) -> QuotaRow {
    QuotaRow(
        id: id, tokenCount: tokens, bytes: bytes ?? Int64(tokens), recency: recency,
        isStableRoot: stable, chainId: chain, isLegacyCompanion: legacy)
}

private func legacy(_ key: String, bytes: Int64, recency: Double) -> QuotaRow {
    row("legacy:\(key)", tokens: 0, bytes: bytes, recency: recency, legacy: true)
}

private func total(_ rows: [QuotaRow]) -> Int64 { rows.reduce(0) { $0 + $1.bytes } }

/// Today's order, as `CacheCoordinator.groupsToEvict` has it: every row that can
/// never fit, then legacy companions, then oldest recency, to exactly the cap.
/// Ties on recency break on the sort key (the id), in the planner too, so the
/// two answers agree even when a tie straddles the stopping point.
private func todaysEviction(_ rows: [QuotaRow], cap: Int64) -> [QuotaRow] {
    var remaining = total(rows)
    guard remaining > cap else { return [] }
    var evicted = rows.filter { $0.bytes > cap }
    remaining -= total(evicted)
    let oversized = Set(evicted.map(\.id))
    var lru: [QuotaRow] = []
    for r in rows.sorted(by: {
        if $0.isLegacyCompanion != $1.isLegacyCompanion { return $0.isLegacyCompanion }
        if $0.recency != $1.recency { return $0.recency < $1.recency }
        return $0.id < $1.id
    }) where remaining > cap && !oversized.contains(r.id) {
        lru.append(r)
        remaining -= r.bytes
    }
    return evicted + lru
}

@Suite("Disk quota planner")
struct DiskQuotaPlannerTests {

    // MARK: - Requirement 1: the trigger

    @Test func emptyInput() {
        for cap: Int64 in [0, 1, 4099] {
            let plan = DiskQuotaPlanner.plan(rows: [], capBytes: cap, activeChain: "chat")
            #expect(plan == QuotaPlan(
                evict: [], evictedBytes: 0, totalBefore: 0, totalAfter: 0, event: nil))
        }
    }

    @Test func belowCapPlansNothing() {
        let rows = [
            row("a", tokens: 333, recency: 1, chain: "A"),
            row("b", tokens: 777, recency: 2, chain: "A"),
            row("c", tokens: 1291, recency: 3, chain: "B"),
        ]
        // 2401 == cap: "fits" is `<=`, not `<`.
        for cap: Int64 in [2401, 2402, 99_991] {
            let plan = DiskQuotaPlanner.plan(rows: rows, capBytes: cap, activeChain: "A")
            #expect(plan.evict.isEmpty)
            #expect(plan.evictedBytes == 0)
            #expect(plan.totalBefore == 2401)
            #expect(plan.totalAfter == 2401)
            #expect(plan.event == nil)
        }
    }

    @Test func triggerIsTheCapNotTheLowWatermark() {
        // cap 2600 -> low 2340. Total 2401 sits between them, and there is
        // plenty that a watermark-triggered pass would happily delete.
        let rows = [
            legacy("old", bytes: 337, recency: 0),
            row("a1", tokens: 101, recency: 1, chain: "A"),
            row("a2", tokens: 211, recency: 2, chain: "A"),
            row("a3", tokens: 461, recency: 3, chain: "A"),
            row("b", tokens: 1291, recency: 4, chain: "B"),
        ]
        #expect(total(rows) == 2401)
        let plan = DiskQuotaPlanner.plan(rows: rows, capBytes: 2600, activeChain: nil)
        #expect(plan.evict.isEmpty)
        #expect(plan.totalBefore == 2401)
        #expect(plan.totalAfter == 2401)
        #expect(plan.event == nil)
    }

    // MARK: - Requirement 3: the soft phase

    @Test func softPhaseStopsAtTheLowWatermark() {
        // cap 2700 -> low 2430. 2814 -> 2713 (still over the cap) -> 2502 (under
        // the cap, over low: keep going) -> 2195 (under low: stop; a401 stays).
        let rows = [
            row("a101", tokens: 101, recency: 1, chain: "A"),
            row("a211", tokens: 211, recency: 2, chain: "A"),
            row("a307", tokens: 307, recency: 3, chain: "A"),
            row("a401", tokens: 401, recency: 4, chain: "A"),
            row("a503", tokens: 503, recency: 5, chain: "A"),
            row("a1291", tokens: 1291, recency: 6, chain: "A"),
        ]
        let plan = DiskQuotaPlanner.plan(rows: rows, capBytes: 2700, activeChain: nil)
        #expect(plan.evict == ["a101", "a211", "a307"])
        #expect(plan.totalBefore == 2814)
        #expect(plan.totalAfter == 2195)
        #expect(plan.evictedBytes == 619)

        // The same chain as the conversation IN PROGRESS stops at the cap:
        // 2814 -> 2713 -> 2502, under 2700, and a307 (which a cold chain loses
        // to the watermark) stays.
        let active = DiskQuotaPlanner.plan(rows: rows, capBytes: 2700, activeChain: "A")
        #expect(active.evict == ["a101", "a211"])
        #expect(active.totalAfter == 2502)
        #expect(active.totalAfter > 2430)
    }

    @Test func legacyCompanionsGoFirst() {
        // The legacy companions are the MOST recently used rows here, so only
        // their class can put them first; within the class, oldest first.
        // cap 1650 -> low 1485: 2258 -> 1839 -> 1502 -> (cold non-tip) 1291.
        let rows = [
            row("a211", tokens: 211, recency: 1, chain: "A"),
            row("a1291", tokens: 1291, recency: 2, chain: "A"),
            legacy("L1", bytes: 337, recency: 50),
            legacy("L2", bytes: 419, recency: 5),
        ]
        let plan = DiskQuotaPlanner.plan(rows: rows, capBytes: 1650, activeChain: nil)
        #expect(plan.evict == ["legacy:L2", "legacy:L1", "a211"])
        #expect(plan.totalAfter == 1291)
        #expect(plan.event == nil)
    }

    @Test func coldNonTipRowsGoBeforeActiveNonTipRows() {
        // The active chain holds the OLDEST rows in the cache, so plain LRU
        // would eat it first. Order must be: the coldest chain's non-tip
        // (cold2), then the warmer chain's non-tips by token count (cold1: 311
        // before 353, although 311 is the fresher row), then the active
        // chain's non-tip.
        let rows = [
            row("cold1:353", tokens: 353, recency: 9, chain: "cold1"),
            row("cold1:311", tokens: 311, recency: 10, chain: "cold1"),
            row("cold1:1009", tokens: 1009, recency: 8, chain: "cold1"),
            row("cold2:409", tokens: 409, recency: 3, chain: "cold2"),
            row("cold2:907", tokens: 907, recency: 2.5, chain: "cold2"),
            row("act:331", tokens: 331, recency: 1, chain: "act"),
            row("act:1103", tokens: 1103, recency: 0.5, chain: "act"),
        ]
        #expect(total(rows) == 4423)
        // cap 3300: 4423 -> 4014 -> 3703 -> 3350, still over the cap with no
        // cold non-tip left -> (act:331) 3019.
        let plan = DiskQuotaPlanner.plan(rows: rows, capBytes: 3300, activeChain: "act")
        #expect(plan.evict == ["cold2:409", "cold1:311", "cold1:353", "act:331"])
        #expect(plan.totalAfter == 3019)

        // cap 3700 -> low 3330: the cold non-tips run out at 3350, over low but
        // under the cap. The active chain does not pay for the watermark.
        let lighter = DiskQuotaPlanner.plan(rows: rows, capBytes: 3700, activeChain: "act")
        #expect(lighter.evict == ["cold2:409", "cold1:311", "cold1:353"])
        #expect(lighter.totalAfter == 3350)
        #expect(lighter.event == nil)

        // cap 4100 -> low 3690: the same three, for the watermark alone (the
        // first of them already restores the cap: 4014).
        let lightest = DiskQuotaPlanner.plan(rows: rows, capBytes: 4100, activeChain: "act")
        #expect(lightest.evict == ["cold2:409", "cold1:311", "cold1:353"])
    }

    @Test func nonTipRowsGoBeforeAnyTip() {
        // X's tip is by far the oldest row. It is still a resume point; Y's
        // superseded snapshot is not, however fresh.
        let rows = [
            row("x701", tokens: 701, recency: 1, chain: "X"),
            row("y223", tokens: 223, recency: 100, chain: "Y"),
            row("y1291", tokens: 1291, recency: 101, chain: "Y"),
        ]
        // cap 2100 -> low 1890: 2215 -> 1992, under the cap, no non-tip left.
        let plan = DiskQuotaPlanner.plan(rows: rows, capBytes: 2100, activeChain: nil)
        #expect(plan.evict == ["y223"])
        #expect(plan.totalAfter == 1992)
    }

    @Test func hysteresisIsNeverPaidWithATip() {
        let root = row("root", tokens: 3137, recency: 9, stable: true)
        let tip = row("act:12089", tokens: 12089, recency: 8, chain: "act")
        // cap 16832 -> low 15148.
        // (a) root + tip = 15226: under the cap, over low. Nothing.
        let idle = DiskQuotaPlanner.plan(
            rows: [root, tip], capBytes: 16832, activeChain: "act")
        #expect(idle.evict.isEmpty)

        // (b) The pass is genuinely triggered (16927 > cap), the one non-tip row
        // goes, and the total lands at 15226 — under the cap, still over low.
        // The pass must stop there: neither the tip nor the root pays for low.
        for active in ["act", "someone-else"] {
            let superseded = row("act:1701", tokens: 1701, recency: 7, chain: "act")
            let plan = DiskQuotaPlanner.plan(
                rows: [root, tip, superseded], capBytes: 16832, activeChain: active)
            #expect(plan.evict == ["act:1701"])
            #expect(plan.totalBefore == 16927)
            #expect(plan.totalAfter == 15226)
            #expect(plan.totalAfter > 15_148)  // low
        }
    }

    @Test func activeChainNeverPaysForHysteresis() {
        // The end of a turn in a regenerate-heavy chat. Root + exact-prompt row
        // + tip = 30 402 fits the cap (31 561) but not low (28 404). The prompt
        // row is what the NEXT regenerate restores from (13 438 tokens against
        // the root's 3 137), so it must not be spent on hysteresis: the active
        // chain's non-tip rows drain to the CAP.
        let rows = [
            row("root", tokens: 3137, recency: 20, stable: true),
            row("act:9286", tokens: 9286, recency: 1, chain: "act"),
            row("act:9675", tokens: 9675, recency: 2, chain: "act"),
            row("act:10778", tokens: 10778, recency: 3, chain: "act"),
            row("act:11167", tokens: 11167, recency: 4, chain: "act"),
            row("act:11749", tokens: 11749, recency: 5, chain: "act"),
            row("act:prompt", tokens: 13438, recency: 6, chain: "act"),
            row("act:tip", tokens: 13827, recency: 7, chain: "act"),
        ]
        #expect(total(rows) == 83_057)
        let plan = DiskQuotaPlanner.plan(rows: rows, capBytes: 31_561, activeChain: "act")
        #expect(plan.evict == ["act:9286", "act:9675", "act:10778", "act:11167", "act:11749"])
        #expect(plan.totalAfter == 30_402)
        #expect(plan.totalAfter > 28_404)  // over low, and that is fine
        #expect(plan.event == DiskCachePressureEvent(
            kind: .activeChainTrimmed, chainId: "act", tipBytes: 13_827, capBytes: 31_561))

        // The control: the very same rows as a COLD chain do pay for low.
        let cold = DiskQuotaPlanner.plan(rows: rows, capBytes: 31_561, activeChain: "other")
        #expect(cold.evict == [
            "act:9286", "act:9675", "act:10778", "act:11167", "act:11749", "act:prompt",
        ])
        #expect(cold.totalAfter == 16_964)
        #expect(cold.event == nil)
    }

    // MARK: - Requirement 4: the hard phase

    @Test func coldestChainsTipGoesFirstUnderHardPressure() {
        // Chain recency is the MAX over the chain's rows: R's tip is the oldest
        // row in the cache (recency 1) but R was touched at 9, so R is the
        // warmest cold chain and its tip outlives Q's and P's. The active tip
        // is older than everything and is not a candidate at all.
        let rows = [
            row("r131", tokens: 131, recency: 9, chain: "R"),
            row("r883", tokens: 883, recency: 1, chain: "R"),
            row("q997", tokens: 997, recency: 2, chain: "Q"),
            row("p1291", tokens: 1291, recency: 5, chain: "P"),
            row("a1103", tokens: 1103, recency: 0.5, chain: "A"),
        ]
        #expect(total(rows) == 4405)
        // cap 3300 -> low 2970: soft 4274; hard q997 -> 3277 <= cap: STOP, even
        // though 3277 > low. Tips never pay for the watermark.
        let one = DiskQuotaPlanner.plan(rows: rows, capBytes: 3300, activeChain: "A")
        #expect(one.evict == ["r131", "q997"])
        #expect(one.totalAfter == 3277)

        // cap 2300: 4274 -> 3277 -> 1986.
        let two = DiskQuotaPlanner.plan(rows: rows, capBytes: 2300, activeChain: "A")
        #expect(two.evict == ["r131", "q997", "p1291"])
        #expect(two.totalAfter == 1986)
        #expect(two.event == nil)
    }

    @Test func stableRootYieldsBeforeTheActiveTip() {
        // The root is the most recently used row; the tip is a superset prefix
        // of it, so for the chat in progress the root saves nothing extra.
        let rows = [
            row("root", tokens: 3137, recency: 100, stable: true),
            row("act:12089", tokens: 12089, recency: 1, chain: "act"),
        ]
        let plan = DiskQuotaPlanner.plan(rows: rows, capBytes: 14000, activeChain: "act")
        #expect(plan.evict == ["root"])
        #expect(plan.totalAfter == 12089)
        #expect(plan.event == nil)
    }

    @Test func activeTipIsLast() {
        // The active tip is the OLDEST row. Cold tips go (coldest first), then
        // stable roots (oldest first), and the pass is under the cap before it
        // ever reaches the active tip.
        let rows = [
            row("act:4931", tokens: 4931, recency: 1, chain: "act"),
            row("c701", tokens: 701, recency: 50, chain: "C"),
            row("d709", tokens: 709, recency: 40, chain: "D"),
            row("rootB", tokens: 3137, recency: 61, stable: true),
            row("rootA", tokens: 2003, recency: 60, stable: true),
        ]
        let plan = DiskQuotaPlanner.plan(rows: rows, capBytes: 5000, activeChain: "act")
        #expect(plan.evict == ["d709", "c701", "rootA", "rootB"])
        #expect(plan.totalAfter == 4931)
        #expect(plan.event == nil)

        // With less pressure the newer root survives too.
        let lighter = DiskQuotaPlanner.plan(rows: rows, capBytes: 8100, activeChain: "act")
        #expect(lighter.evict == ["d709", "c701", "rootA"])
        #expect(lighter.totalAfter == 8068)
    }

    // MARK: - Requirement 2: oversized rows

    @Test func oversizedRowIsDroppedFirst() {
        // z5003 can never fit under 5000 and is the NEWEST row. It goes first;
        // Z's prior fitting prefix (z1291) becomes Z's resume point and stays.
        let rows = [
            legacy("L", bytes: 337, recency: 1),
            row("y2203", tokens: 2203, recency: 3, chain: "Y"),
            row("y2909", tokens: 2909, recency: 4, chain: "Y"),
            row("z1291", tokens: 1291, recency: 98, chain: "Z"),
            row("z5003", tokens: 5003, recency: 99, chain: "Z"),
        ]
        #expect(total(rows) == 11743)
        // cap 5000 -> low 4500: 6740 -> 6403 -> 4200.
        let plan = DiskQuotaPlanner.plan(rows: rows, capBytes: 5000, activeChain: nil)
        #expect(plan.evict == ["z5003", "legacy:L", "y2203"])
        #expect(plan.totalAfter == 4200)
        #expect(plan.event == nil)
    }

    @Test func oversizedDropThatRestoresCapStopsThePass() {
        // After the oversized row is gone the total is 4801: under the cap,
        // over low (4500), with a legacy companion and a non-tip row on offer.
        // An oversized row is no reason to chase the low watermark.
        let rows = [
            legacy("L", bytes: 337, recency: 1),
            row("y2203", tokens: 2203, recency: 3, chain: "Y"),
            row("y2261", tokens: 2261, recency: 4, chain: "Y"),
            row("z5003", tokens: 5003, recency: 99, chain: "Z"),
        ]
        let plan = DiskQuotaPlanner.plan(rows: rows, capBytes: 5000, activeChain: nil)
        #expect(plan.evict == ["z5003"])
        #expect(plan.totalBefore == 9804)
        #expect(plan.totalAfter == 4801)
    }

    @Test func oversizedActiveTipEmitsTipDropped() {
        // Port of `combinedQuotaRejectsOversizedNewestBoundaryButPreservesPrior
        // FittingPrefix`: the newest boundary cannot fit; the prior one must
        // survive as the resume point, so the root yields instead.
        let rows = [
            row("root", tokens: 3137, recency: 3, stable: true),
            row("act:10597", tokens: 10597, recency: 1, chain: "act"),
            row("act:12089", tokens: 12089, recency: 2, chain: "act"),
        ]
        let plan = DiskQuotaPlanner.plan(rows: rows, capBytes: 12000, activeChain: "act")
        #expect(plan.evict == ["act:12089", "root"])
        #expect(plan.totalAfter == 10597)
        #expect(plan.event == DiskCachePressureEvent(
            kind: .activeTipDropped, chainId: "act", tipBytes: 12089, capBytes: 12000))

        // The same rows seen from another conversation. The chain is cold now:
        // its oversized tip is nobody's event, and its fitting tip is a cold
        // tip, which yields BEFORE the stable root (4d before 4e).
        let cold = DiskQuotaPlanner.plan(rows: rows, capBytes: 12000, activeChain: "other")
        #expect(cold.evict == ["act:12089", "act:10597"])
        #expect(cold.totalAfter == 3137)
        #expect(cold.event == nil)
    }

    @Test func oversizedActiveRowThatIsNotTheTipEmitsNoEvent() {
        // Bytes are not monotone in tokens (a companion payload rides on some
        // boundaries only). The oversized row is not the resume point.
        let rows = [
            row("act:500", tokens: 500, bytes: 7001, recency: 1, chain: "act"),
            row("act:900", tokens: 900, bytes: 907, recency: 2, chain: "act"),
        ]
        let plan = DiskQuotaPlanner.plan(rows: rows, capBytes: 7000, activeChain: "act")
        #expect(plan.evict == ["act:500"])
        #expect(plan.event == nil)
    }

    // MARK: - Requirement 5: the pressure event

    @Test func trimmingTheActiveChainEmitsTrimmedOnce() {
        // Three active rows go; ONE event, carrying the tip's BYTES (4099), not
        // its token count (1291), and not the bytes that were evicted.
        let rows = [
            row("act:101", tokens: 101, recency: 1, chain: "act"),
            row("act:211", tokens: 211, recency: 2, chain: "act"),
            row("act:307", tokens: 307, recency: 3, chain: "act"),
            row("act:1291", tokens: 1291, bytes: 4099, recency: 4, chain: "act"),
        ]
        // cap 4400: 4718 -> 4617 -> 4406 -> 4099.
        let plan = DiskQuotaPlanner.plan(rows: rows, capBytes: 4400, activeChain: "act")
        #expect(plan.evict == ["act:101", "act:211", "act:307"])
        #expect(plan.totalAfter == 4099)
        #expect(plan.event == DiskCachePressureEvent(
            kind: .activeChainTrimmed, chainId: "act", tipBytes: 4099, capBytes: 4400))

        // cap 4500 -> low 4050: the active chain stops at the cap (4406), one
        // row earlier than the watermark would have it. Still one event.
        let lighter = DiskQuotaPlanner.plan(rows: rows, capBytes: 4500, activeChain: "act")
        #expect(lighter.evict == ["act:101", "act:211"])
        #expect(lighter.totalAfter == 4406)
        #expect(lighter.event == DiskCachePressureEvent(
            kind: .activeChainTrimmed, chainId: "act", tipBytes: 4099, capBytes: 4500))
    }

    @Test func tipDroppedTakesPrecedenceOverTrimmed() {
        // ONE pass both drops the oversized active tip (a3) and trims the
        // active chain (a1). The event is the first one: `.activeTipDropped`,
        // carrying a3's bytes — not `.activeChainTrimmed`.
        //
        // 10115 -> (a3, oversized) 5112 -> (3c: a1, to the cap) 5011, still over
        // 5000, no non-tip row left -> (4d: cold tip c) 211. The eviction is
        // [a3, a1, c], not [a3, a1]: a2 (the chain's fitting resume point) + c
        // is 5011, eleven bytes over the cap, so the cold tip has to go too.
        let rows = [
            row("a1", tokens: 100, bytes: 101, recency: 1, chain: "act"),
            row("a2", tokens: 200, bytes: 211, recency: 2, chain: "act"),
            row("a3", tokens: 300, bytes: 5003, recency: 3, chain: "act"),
            row("c", tokens: 50, bytes: 4800, recency: 4, chain: "cold"),
        ]
        let plan = DiskQuotaPlanner.plan(rows: rows, capBytes: 5000, activeChain: "act")
        #expect(plan.evict == ["a3", "a1", "c"])
        #expect(plan.totalAfter == 211)
        #expect(plan.event?.kind == .activeTipDropped)
        #expect(plan.event?.tipBytes == 5003)
    }

    @Test func evictingOnlyColdRowsEmitsNoEvent() {
        // Soft pressure absorbed by a legacy companion and a cold non-tip...
        let rows = [
            legacy("L", bytes: 337, recency: 90),
            row("c409", tokens: 409, recency: 3, chain: "C"),
            row("c907", tokens: 907, recency: 4, chain: "C"),
            row("root", tokens: 3137, recency: 80, stable: true),
            row("act:331", tokens: 331, recency: 1, chain: "act"),
            row("act:1103", tokens: 1103, recency: 2, chain: "act"),
        ]
        #expect(total(rows) == 6224)
        // cap 6200 -> low 5580: 6224 -> 5887 -> 5478.
        let soft = DiskQuotaPlanner.plan(rows: rows, capBytes: 6200, activeChain: "act")
        #expect(soft.evict == ["legacy:L", "c409"])
        #expect(soft.event == nil)

        // ...and hard pressure absorbed by a cold tip and the stable root, with
        // the active chain's own rows gone in between: THAT is an event, and
        // the control for the nil above.
        // cap 1200 -> low 1080: the root (3137) cannot fit at all -> 3087;
        // soft 2750 -> 2341 -> (act:331) 2010; hard (c907) 1103.
        let hard = DiskQuotaPlanner.plan(rows: rows, capBytes: 1200, activeChain: "act")
        #expect(hard.evict == ["root", "legacy:L", "c409", "act:331", "c907"])
        #expect(hard.event?.kind == .activeChainTrimmed)

        // Same pass with nobody active: "act" is just the coldest chain. Its
        // rows go first in each class — its tip instead of C's — and the very
        // same evictions of act:331 are no event.
        // 3087 -> 2750 -> (act:331) 2419 -> (c409) 2010; hard (act:1103) 907.
        let nobody = DiskQuotaPlanner.plan(rows: rows, capBytes: 1200, activeChain: nil)
        #expect(nobody.evict == ["root", "legacy:L", "act:331", "c409", "act:1103"])
        #expect(nobody.event == nil)
    }

    // MARK: - Chains

    @Test func nilChainRowsAreChainsOfOne() {
        // If nil == nil made these one chain, n1 (most tokens) would be its tip
        // and n2 would be a superseded row: the soft phase would take n2 (870
        // <= low 900). As chains of one they are all tips: oldest first, to
        // the cap.
        let rows = [
            row("n1", tokens: 5000, bytes: 431, recency: 1),
            row("n2", tokens: 100, bytes: 433, recency: 2),
            row("n3", tokens: 200, bytes: 439, recency: 3),
        ]
        let plan = DiskQuotaPlanner.plan(rows: rows, capBytes: 1000, activeChain: nil)
        #expect(plan.evict == ["n1"])
        #expect(plan.totalAfter == 872)
        #expect(plan.event == nil)

        // A chain-less row is never "the active chain", not even when its id
        // happens to spell the active chain's name.
        let named = [
            row("act", tokens: 100, bytes: 433, recency: 1),
            row("act:1", tokens: 50, bytes: 211, recency: 2, chain: "act"),
            row("act:2", tokens: 5000, bytes: 431, recency: 3, chain: "act"),
        ]
        // cap 1000: 1075 -> the active non-tip -> 864, under the cap: stop.
        let planNamed = DiskQuotaPlanner.plan(rows: named, capBytes: 1000, activeChain: "act")
        #expect(planNamed.evict == ["act:1"])
        #expect(planNamed.event?.kind == .activeChainTrimmed)
        #expect(planNamed.event?.tipBytes == 431)
    }

    @Test func stableRootCarryingAChainIdIsStillAStableRoot() {
        // `kind` decides, not `chain_id`. If the root (most tokens) counted as a
        // row of "act" it would be the chain's tip, act:907 would be a
        // superseded row, and the root would be the last row standing.
        let rows = [
            row("root", tokens: 3137, recency: 9, chain: "act", stable: true),
            row("act:409", tokens: 409, recency: 1, chain: "act"),
            row("act:907", tokens: 907, recency: 2, chain: "act"),
        ]
        // cap 1000: 4453 -> (root, oversized) 1316 -> (act:409) 907.
        let plan = DiskQuotaPlanner.plan(rows: rows, capBytes: 1000, activeChain: "act")
        #expect(plan.evict == ["root", "act:409"])
        #expect(plan.event == DiskCachePressureEvent(
            kind: .activeChainTrimmed, chainId: "act", tipBytes: 907, capBytes: 1000))

        // cap 4100: nothing is oversized. 4453 -> (act:409) 4044: the root, as a
        // stable root, is not touched; as the chain's tip it would not be
        // either, but then tipBytes would be 3137.
        let roomy = DiskQuotaPlanner.plan(rows: rows, capBytes: 4100, activeChain: "act")
        #expect(roomy.evict == ["act:409"])
        #expect(roomy.event?.tipBytes == 907)

        // cap 3300: 4453 -> (act:409) 4044 -> no cold tip -> (root) 907.
        let hard = DiskQuotaPlanner.plan(rows: rows, capBytes: 3300, activeChain: "act")
        #expect(hard.evict == ["act:409", "root"])
        #expect(hard.totalAfter == 907)
    }

    @Test func legacyCompanionCarryingAChainIdIsStillALegacyCompanion() {
        // It goes first, as a legacy companion, although its chain is active;
        // it is never the chain's tip (tipBytes is act:907's), and losing it
        // alone is not an event.
        let rows = [
            row("legacy:L", tokens: 5000, bytes: 337, recency: 99, chain: "act", legacy: true),
            row("act:409", tokens: 409, recency: 1, chain: "act"),
            row("act:907", tokens: 907, recency: 2, chain: "act"),
        ]
        // cap 1400 -> low 1260: 1653 -> (legacy) 1316, under the cap: stop.
        let plan = DiskQuotaPlanner.plan(rows: rows, capBytes: 1400, activeChain: "act")
        #expect(plan.evict == ["legacy:L"])
        #expect(plan.event == nil)

        // cap 1300: 1653 -> 1316 -> (act:409) 907.
        let tighter = DiskQuotaPlanner.plan(rows: rows, capBytes: 1300, activeChain: "act")
        #expect(tighter.evict == ["legacy:L", "act:409"])
        #expect(tighter.event?.tipBytes == 907)
    }

    @Test func capZeroEvictsEveryRowThatHasBytes() {
        // Every row with bytes is oversized against a cap of 0: oldest first,
        // whatever its class, and the active tip's loss is reported.
        let rows = [
            legacy("L", bytes: 337, recency: 5),
            row("root", tokens: 3137, recency: 4, stable: true),
            row("act:409", tokens: 409, recency: 1, chain: "act"),
            row("act:907", tokens: 907, recency: 2, chain: "act"),
            row("c701", tokens: 701, recency: 3, chain: "C"),
        ]
        let plan = DiskQuotaPlanner.plan(rows: rows, capBytes: 0, activeChain: "act")
        #expect(plan.evict == ["act:409", "act:907", "c701", "root", "legacy:L"])
        #expect(plan.totalBefore == 5491)
        #expect(plan.totalAfter == 0)
        #expect(plan.event == DiskCachePressureEvent(
            kind: .activeTipDropped, chainId: "act", tipBytes: 907, capBytes: 0))
    }

    @Test func zeroByteRowsCostNothingAndSaveNothing() {
        // A zero-byte row fits any cap, so it is never oversized; at cap 0 it
        // is all that survives.
        let rows = [
            row("z0", tokens: 50, bytes: 0, recency: 1, chain: "Z"),
            row("z907", tokens: 907, recency: 2, chain: "Z"),
        ]
        let zero = DiskQuotaPlanner.plan(rows: rows, capBytes: 0, activeChain: nil)
        #expect(zero.evict == ["z907"])
        #expect(zero.totalAfter == 0)

        // In an ordered phase it is taken in its turn like any other row: it is
        // a superseded row, the total is over the goal, it goes — for 0 bytes.
        // cap 1500 -> low 1350: 1608 -> (z0) 1608 -> (y211) 1397 -> (y223) 1174.
        let more = rows + [
            row("y211", tokens: 211, recency: 3, chain: "Y"),
            row("y223", tokens: 223, recency: 4, chain: "Y"),
            row("y490", tokens: 490, bytes: 267, recency: 5, chain: "Y"),
        ]
        #expect(total(more) == 1608)
        let plan = DiskQuotaPlanner.plan(rows: more, capBytes: 1500, activeChain: nil)
        #expect(plan.evict == ["z0", "y211", "y223"])
        #expect(plan.evictedBytes == 434)
        #expect(plan.totalAfter == 1174)
    }

    @Test func activeChainWithNoRowsIsHarmless() {
        let rows = [
            legacy("L", bytes: 337, recency: 90),
            row("c409", tokens: 409, recency: 3, chain: "C"),
            row("c907", tokens: 907, recency: 4, chain: "C"),
            row("d1291", tokens: 1291, recency: 1, chain: "D"),
            row("root", tokens: 3137, recency: 80, stable: true),
        ]
        for cap: Int64 in [6000, 5000, 3000, 1000, 100] {
            let ghost = DiskQuotaPlanner.plan(rows: rows, capBytes: cap, activeChain: "ghost")
            let nobody = DiskQuotaPlanner.plan(rows: rows, capBytes: cap, activeChain: nil)
            #expect(ghost == nobody)
            #expect(ghost.event == nil)
            #expect(ghost.totalBefore == 6081)
            #expect(ghost.totalAfter <= cap)
        }
    }

    // MARK: - Requirement 7: migrated caches

    @Test func migratedCacheMatchesOldestFirstOrder() {
        var rng = SeededRNG(seed: 0x5EED_0007)
        let cases = 500
        var compared = 0, evicting = 0, withOversized = 0, tieAtTheStoppingPoint = 0
        for _ in 0..<cases {
            let n = Int.random(in: 1...40, using: &rng)
            var ids = (0..<n).map { String(format: "h%04d", $0) }
            ids.shuffle(using: &rng)
            var rows = ids.map { id in
                row(
                    id, tokens: Int.random(in: 1...50_000, using: &rng),
                    bytes: Int64.random(in: 333...1291, using: &rng),
                    // A narrow recency range on purpose: ties happen, and some
                    // straddle the stopping point, where only the shared id
                    // tie-break keeps the two answers equal.
                    recency: Double(Int.random(in: 0..<(n * 3), using: &rng)))
            }
            if Int.random(in: 0..<5, using: &rng) == 0 {
                rows.append(row(
                    "hbig", tokens: 7, bytes: Int64.random(in: 3001...9001, using: &rng),
                    recency: Double(n * 3 + 1)))
            }
            let sum = total(rows)
            let cap = Int64(Double(sum) * Double.random(in: 0.05...1.15, using: &rng))

            let classic = todaysEviction(rows, cap: cap)
            let plan = DiskQuotaPlanner.plan(rows: rows, capBytes: cap, activeChain: nil)
            compared += 1
            if !classic.isEmpty { evicting += 1 }
            if rows.contains(where: { $0.bytes > cap }) { withOversized += 1 }
            let gone = Set(classic.map(\.id))
            if classic.contains(where: { e in
                e.bytes <= cap && rows.contains { !gone.contains($0.id) && $0.recency == e.recency }
            }) { tieAtTheStoppingPoint += 1 }
            #expect(Set(plan.evict) == gone)
            #expect(plan.totalBefore == sum)
            #expect(plan.totalAfter == sum - total(classic))
            #expect(plan.event == nil)
        }
        print(
            "PLANNER_PROPERTY cases=\(cases) compared=\(compared)"
                + " tie_at_the_stopping_point=\(tieAtTheStoppingPoint)"
                + " evicting=\(evicting) with_oversized=\(withOversized)")
        // Fail closed: a corpus that never evicts, or never ties, proves nothing.
        #expect(compared == cases)
        #expect(evicting >= 300)
        #expect(withOversized >= 20)
        #expect(tieAtTheStoppingPoint >= 50)
    }

    @Test func legacyCompanionsPayForLowWhereTheOldOrderStoppedAtTheCap() {
        // The one place a migrated cache differs from the old order: both take
        // legacy companions first, oldest first; the old order stopped at the
        // cap, the planner goes on to low — with legacy companions only.
        let rows = [
            legacy("L1", bytes: 337, recency: 1),
            legacy("L2", bytes: 419, recency: 2),
            legacy("L3", bytes: 431, recency: 3),
            row("h1", tokens: 1291, recency: 4),
            row("h2", tokens: 1301, recency: 5),
        ]
        #expect(total(rows) == 3779)
        // cap 3500 -> low 3150: old 3779 -> 3442, stop. Planner -> 3442 -> 3023.
        #expect(todaysEviction(rows, cap: 3500).map(\.id) == ["legacy:L1"])
        let plan = DiskQuotaPlanner.plan(rows: rows, capBytes: 3500, activeChain: nil)
        #expect(plan.evict == ["legacy:L1", "legacy:L2"])
        #expect(plan.totalAfter == 3023)

        // Out of legacy companions and still over low: no KV row pays for it.
        // cap 2650 -> low 2385: 3779 -> 3442 -> 3023 -> 2592, under the cap and
        // over low with no legacy companion left: h1 stays, as in the old order.
        let tight = DiskQuotaPlanner.plan(rows: rows, capBytes: 2650, activeChain: nil)
        #expect(tight.evict == ["legacy:L1", "legacy:L2", "legacy:L3"])
        #expect(tight.totalAfter == 2592)
        #expect(todaysEviction(rows, cap: 2650).map(\.id) == tight.evict)
    }

    // MARK: - Requirements 6 and 8: determinism and accounting

    /// A corpus with every row class in it, including recency and token ties.
    ///
    /// `tieHeavy` squeezes recency and token counts into three values each, so
    /// that nearly every ordering decision the planner makes is a tie and only
    /// the final `id` tie-break stands between it and input order.
    private static func mixedCorpus(
        _ rng: inout SeededRNG, tieHeavy: Bool = false
    ) -> (rows: [QuotaRow], active: String?) {
        let n = Int.random(in: 1...48, using: &rng)
        var ids = (0..<n).map { String(format: "r%04d", $0) }
        ids.shuffle(using: &rng)
        let rows = ids.map { id -> QuotaRow in
            let kind = Int.random(in: 0..<20, using: &rng)
            let chainPick = Int.random(in: 0..<7, using: &rng)
            return QuotaRow(
                id: kind < 2 ? "legacy:\(id)" : id,
                tokenCount: kind < 2 ? 0 : Int.random(in: 1...(tieHeavy ? 3 : 12), using: &rng) * 997,
                bytes: Int64.random(in: 333...1291, using: &rng)
                    * (Int.random(in: 0..<25, using: &rng) == 0 ? 9 : 1),
                recency: Double(Int.random(in: 0..<(tieHeavy ? 3 : n * 2), using: &rng)),
                isStableRoot: kind == 2,
                chainId: chainPick < 5 ? "c\(chainPick)" : nil,
                isLegacyCompanion: kind < 2)
        }
        let activePick = Int.random(in: 0..<7, using: &rng)
        return (rows, activePick < 6 ? "c\(activePick)" : nil)  // c5 never has rows
    }

    @Test func planIsIndependentOfInputOrder() {
        var rng = SeededRNG(seed: 0x5EED_0006)
        var nonTrivial = 0, corpora = 0
        // One ordinary corpus, then tie-heavy ones: a stable sort with a
        // missing tie-break reproduces INPUT order, and only ties can show it.
        for tieHeavy in [false, true, true, true] {
            var corpus: (rows: [QuotaRow], active: String?)
            repeat { corpus = Self.mixedCorpus(&rng, tieHeavy: tieHeavy) }
            while corpus.rows.count < 40 || corpus.active == nil
            corpora += 1
            let sum = total(corpus.rows)
            for fraction in [0.97, 0.71, 0.43, 0.13] {
                let cap = Int64(Double(sum) * fraction)
                let baseline = DiskQuotaPlanner.plan(
                    rows: corpus.rows, capBytes: cap, activeChain: corpus.active)
                if baseline.evict.count >= 2 { nonTrivial += 1 }
                for _ in 0..<200 {
                    let shuffled = corpus.rows.shuffled(using: &rng)
                    let again = DiskQuotaPlanner.plan(
                        rows: shuffled, capBytes: cap, activeChain: corpus.active)
                    #expect(again == baseline)
                    if again != baseline { return }  // one report is enough
                }
            }
        }
        #expect(corpora == 4)
        #expect(nonTrivial == 16)  // fail closed: equal EMPTY plans prove nothing
    }

    /// The planner's doc comment says step 4f never actually runs: once every
    /// other row is gone, a tip that fits the cap fits it alone. Hold it to that.
    @Test func activeTipThatFitsIsNeverEvicted() {
        var rng = SeededRNG(seed: 0x5EED_0004)
        var evictingWithActiveTip = 0, downToTheTipAlone = 0
        for _ in 0..<500 {
            let (rows, active) = Self.mixedCorpus(&rng)
            guard let active else { continue }
            let tip = rows
                .filter { !$0.isStableRoot && !$0.isLegacyCompanion && $0.chainId == active }
                .max {
                    ($0.tokenCount, $0.recency, $0.id) < ($1.tokenCount, $1.recency, $1.id)
                }
            guard let tip else { continue }
            // Caps from "barely fits the tip" upward: the hardest pressure that
            // still leaves the tip a legal survivor.
            let cap = tip.bytes + Int64.random(in: 0...1291, using: &rng)
            guard total(rows) > cap else { continue }
            let plan = DiskQuotaPlanner.plan(rows: rows, capBytes: cap, activeChain: active)
            evictingWithActiveTip += 1
            if plan.totalAfter == tip.bytes { downToTheTipAlone += 1 }
            #expect(!plan.evict.contains(tip.id))
            #expect(plan.totalAfter <= cap)
            #expect(plan.event?.kind != .activeTipDropped)
        }
        #expect(evictingWithActiveTip >= 200)
        #expect(downToTheTipAlone >= 50)
    }

    @Test func accountingIsExact() {
        var rng = SeededRNG(seed: 0x5EED_0008)
        var evicting = 0, withEvent = 0
        for _ in 0..<500 {
            let (rows, active) = Self.mixedCorpus(&rng)
            let sum = total(rows)
            let cap = Int64(Double(sum) * Double.random(in: 0.05...1.15, using: &rng))
            let plan = DiskQuotaPlanner.plan(rows: rows, capBytes: cap, activeChain: active)

            let byId = Dictionary(uniqueKeysWithValues: rows.map { ($0.id, $0) })
            #expect(plan.totalBefore == sum)
            #expect(Set(plan.evict).count == plan.evict.count)
            #expect(plan.evict.allSatisfy { byId[$0] != nil })
            #expect(plan.evictedBytes == plan.evict.reduce(Int64(0)) { $0 + (byId[$1]?.bytes ?? 0) })
            #expect(plan.totalAfter == plan.totalBefore - plan.evictedBytes)
            if sum <= cap {
                #expect(plan.evict.isEmpty)
                #expect(plan.event == nil)
            } else {
                evicting += 1
                #expect(!plan.evict.isEmpty)
                #expect(plan.totalAfter <= cap)
                // Every row that can never fit is gone.
                #expect(rows.allSatisfy { $0.bytes <= cap || plan.evict.contains($0.id) })
            }
            if let event = plan.event {
                withEvent += 1
                #expect(event.chainId == active)
                #expect(event.capBytes == cap)
            }
        }
        #expect(evicting >= 300)
        #expect(withEvent >= 20)
    }

    // MARK: - The scenario replay (regression guard against today's order)
    //
    // A port of the Python model this policy was reviewed with. MODEL ONLY:
    // bytes == tokens, one shared stable root, and per turn the history
    // top, up to 3 ladder rungs, the prompt and the post-answer boundary. Rows
    // are keyed by the MESSAGE IDS they cover, so the two answers of a
    // regenerate are distinct prefixes even where their token counts collide.

    private enum ReplayPolicy: String {
        /// The control: today's order, inline on every store.
        case today
        /// Printed control: today's order, once at the end of the turn.
        case todayEndOfTurn
        /// Printed control: the planner, once at the end of the turn.
        case plannerEndOfTurn
        /// The production design: the planner, inline after every store.
        case plannerPerStore
    }

    private struct ReplayMessage { let id: String; let tokens: Int }

    private final class ReplayConversation {
        let name: String
        var messages: [ReplayMessage] = []
        var counter = 0
        var suffix = ""
        init(_ name: String) { self.name = name }

        /// A branch that shares this conversation's first `keep` messages.
        func fork(keep: Int, suffix: String) -> ReplayConversation {
            let c = ReplayConversation(name)
            c.messages = Array(messages[..<keep])
            c.counter = counter
            c.suffix = suffix
            return c
        }

        func newId(_ tag: String) -> String {
            counter += 1
            return "\(name).\(tag)\(counter)\(suffix)"
        }
    }

    private struct ReplayTurn { let conversation: String; let tag: String; let prompt: Int; let hit: Int }

    private struct ReplayResult {
        var turns: [ReplayTurn] = []
        var passes = 0
        /// Greatest total ÷ cap seen right after a store (before a per-store
        /// planner pass; after today's inline pass).
        var peak = 0.0
        /// Fail-closed counters: a policy that does not really evict has
        /// perfect reuse, and must not be able to win that way.
        var turnsEndedOverCap = 0, passesThatLeftTheCacheOverCap = 0
        var trace: [String] = []

        var meanReuse: Double {
            turns.reduce(0.0) { $0 + Double($1.hit) / Double($1.prompt) } / Double(turns.count)
        }
        var tokenWeightedReuse: Double {
            Double(turns.reduce(0) { $0 + $1.hit }) / Double(turns.reduce(0) { $0 + $1.prompt })
        }
    }

    private final class ReplayCache {
        struct Row {
            let key: [String]  // the message ids covered; [] = the stable root
            let tokens: Int
            var recency: Int
            let chain: String?
            var id: String { key.isEmpty ? "root" : key.joined(separator: "/") }
        }
        static let sys = 3137, user = 1103, ans = 389, regeneratedAns = 211
        static let minGap = 512, maxRungs = 3, drops = [1, 2, 4, 8, 16]

        let cap: Int64
        let policy: ReplayPolicy
        var rows: [Row] = []  // insertion order; a re-stored row goes to the end
        var clock = 0
        var chainSeq = 0
        var result = ReplayResult()

        init(cap: Int64, policy: ReplayPolicy) { self.cap = cap; self.policy = policy }

        /// Cumulative (key, tokens) for every message end; index 0 = system only.
        static func prefixRows(_ messages: [ReplayMessage]) -> [(key: [String], tokens: Int)] {
            var out: [(key: [String], tokens: Int)] = [([], sys)]
            var total = sys
            for (i, m) in messages.enumerated() {
                total += m.tokens
                out.append((messages[...i].map(\.id), total))
            }
            return out
        }

        /// `messages` = the history plus the new user message.
        static func boundaries(_ messages: [ReplayMessage]) -> (
            rungs: [(key: [String], tokens: Int)], top: (key: [String], tokens: Int),
            prompt: (key: [String], tokens: Int)
        ) {
            let ends = prefixRows(Array(messages.dropLast()))
            let prompt = prefixRows(messages).last!
            let top = ends.last!
            var rungs: [(key: [String], tokens: Int)] = []
            var last = top.tokens
            for d in drops {
                if rungs.count >= maxRungs { break }
                let i = ends.count - 1 - d
                if i <= 0 { break }
                if last - ends[i].tokens >= minGap { rungs.append(ends[i]); last = ends[i].tokens }
            }
            return (rungs, top, prompt)
        }

        var total: Int64 { rows.reduce(0) { $0 + Int64($1.tokens) } }
        func tick() -> Int { clock += 1; return clock }

        /// The longest stored prefix of `key`. Touches it, and the root.
        func fetch(_ key: [String]) -> (tokens: Int, chain: String?) {
            var best: Int?
            for (i, r) in rows.enumerated()
            where r.key.count <= key.count && r.key.elementsEqual(key.prefix(r.key.count)) {
                if best == nil || r.tokens > rows[best!].tokens { best = i }
            }
            guard let best else { return (0, nil) }
            let now = tick()
            rows[best].recency = now
            if let root = rows.firstIndex(where: { $0.key.isEmpty }) { rows[root].recency = now }
            return (rows[best].tokens, rows[best].key.isEmpty ? nil : rows[best].chain)
        }

        func store(_ key: [String], tokens: Int, chain: String?) {
            guard !rows.contains(where: { $0.key == key }) else { return }
            rows.append(Row(
                key: key, tokens: tokens, recency: tick(), chain: key.isEmpty ? nil : chain))
            if policy == .today { evictTodaysWay() }
            result.peak = max(result.peak, Double(total) / Double(cap))
            if policy == .plannerPerStore, !key.isEmpty, let chain { plannerPass(active: chain) }
        }

        /// Oversized first, then oldest recency, to the cap.
        func evictTodaysWay() {
            guard total > cap else { return }
            result.passes += 1
            rows.removeAll { Int64($0.tokens) > cap }
            let order = rows.enumerated()
                .sorted { ($0.element.recency, $0.offset) < ($1.element.recency, $1.offset) }
                .map(\.element.key)
            for key in order {
                if total <= cap { break }
                rows.removeAll { $0.key == key }
            }
            if total > cap { result.passesThatLeftTheCacheOverCap += 1 }
        }

        func plannerPass(active: String) {
            guard total > cap else { return }
            result.passes += 1
            let plan = DiskQuotaPlanner.plan(
                rows: rows.map {
                    QuotaRow(
                        id: $0.id, tokenCount: $0.tokens, bytes: Int64($0.tokens),
                        recency: Double($0.recency), isStableRoot: $0.key.isEmpty,
                        chainId: $0.chain, isLegacyCompanion: false)
                },
                capBytes: cap, activeChain: active)
            let gone = Set(plan.evict)
            rows.removeAll { gone.contains($0.id) }
            if total > cap { result.passesThatLeftTheCacheOverCap += 1 }
        }

        func endTurn(active: String) {
            switch policy {
            case .todayEndOfTurn: evictTodaysWay()
            case .plannerEndOfTurn: plannerPass(active: active)
            case .today, .plannerPerStore: break
            }
        }

        /// One request: restore, then store this turn's boundary set.
        func turn(
            _ conversation: ReplayConversation, tag: String = "", ans: Int = ReplayCache.ans,
            userMessage: ReplayMessage? = nil
        ) {
            let user = userMessage
                ?? ReplayMessage(id: conversation.newId("u"), tokens: ReplayCache.user)
            let messages = conversation.messages + [user]
            let b = Self.boundaries(messages)
            let before = describe()
            let hit = fetch(b.prompt.key)
            // The chain of the row that was hit, unless that is the shared root
            // (or nothing): then this is a new conversation.
            let chain: String
            if let hitChain = hit.chain {
                chain = hitChain
            } else {
                chainSeq += 1
                chain = "chain\(chainSeq)"
            }
            let answer = ReplayMessage(id: conversation.newId("a"), tokens: ans)
            let post = Self.prefixRows(messages + [answer]).last!
            store([], tokens: Self.sys, chain: nil)
            var seen = Set<[String]>()
            for row in (b.rungs + [b.top, b.prompt, post]).sorted(by: { $0.tokens < $1.tokens })
            where !row.key.isEmpty && seen.insert(row.key).inserted {
                store(row.key, tokens: row.tokens, chain: chain)
            }
            endTurn(active: chain)
            if total > cap { result.turnsEndedOverCap += 1 }
            conversation.messages = messages + [answer]
            result.turns.append(ReplayTurn(
                conversation: conversation.name, tag: tag, prompt: b.prompt.tokens, hit: hit.tokens))
            result.trace.append(
                "  #\(result.turns.count - 1) \(conversation.name) \(tag) chain=\(chain)"
                    + " prompt=\(b.prompt.tokens) hit=\(hit.tokens) total=\(total)"
                    + "\n    before: \(before)\n    after:  \(describe())")
        }

        func describe() -> String {
            rows.map { "\($0.key.last ?? "root")#\($0.tokens)[\($0.chain ?? "-")]@\($0.recency)" }
                .joined(separator: " ")
        }
    }

    private enum ReplayShape: String, CaseIterable {
        /// 3 chats one after another, 12 turns each, then chat 0 resumes.
        case baseline
        /// The same 3 chats round-robin, then chat 0 once more.
        case interleaved
        /// 24 chats x 2 turns, then turn 3 of the first 6 and of the last 6.
        case manyShort
        /// One chat, 12 turns; every answer is regenerated once, the
        /// regenerated answer is SHORTER (211 against 389) and the conversation
        /// goes on from it. By token count the dead answer's row is the
        /// chain's tip and the live one is a superseded row.
        case regenerate

        var expectedTurns: Int {
            switch self {
            case .baseline, .interleaved: return 37
            case .manyShort: return 60
            case .regenerate: return 24
            }
        }

        func run(on cache: ReplayCache) {
            typealias C = ReplayConversation
            switch self {
            case .baseline:
                let chats = (0..<3).map { C("c\($0)") }
                for chat in chats { for _ in 1...12 { cache.turn(chat) } }
                cache.turn(chats[0], tag: "resume")
            case .interleaved:
                let chats = (0..<3).map { C("c\($0)") }
                for _ in 1...12 { for chat in chats { cache.turn(chat) } }
                cache.turn(chats[0], tag: "resume")
            case .manyShort:
                let chats = (0..<24).map { C("m\($0)") }
                for chat in chats { for _ in 1...2 { cache.turn(chat) } }
                for chat in chats[..<6] + chats[18...] { cache.turn(chat, tag: "resume") }
            case .regenerate:
                var chat = C("a")
                for _ in 1...12 {
                    let history = chat.messages.count
                    cache.turn(chat, tag: "first")
                    let again = chat.fork(keep: history, suffix: "r")
                    cache.turn(
                        again, tag: "regen", ans: ReplayCache.regeneratedAns,
                        userMessage: chat.messages[history])
                    chat = again
                    chat.suffix = ""
                }
            }
        }
    }

    private static func replay(
        _ shape: ReplayShape, _ policy: ReplayPolicy, capTips: Double
    ) -> ReplayResult {
        let tip = ReplayCache.sys + 11 * 1492 + ReplayCache.user + ReplayCache.ans  // 21 041
        let cache = ReplayCache(cap: Int64(capTips * Double(tip)), policy: policy)
        shape.run(on: cache)
        return cache.result
    }

    @Test func replaySimulatorShapesMatchTheModel() {
        // Pin the scenario itself, so the replay cannot silently drift into a
        // different (easier) workload.
        func boundarySet(turn: Int) -> (set: [Int], prompt: Int) {
            var messages: [ReplayMessage] = []
            for t in 1..<turn {
                messages += [
                    ReplayMessage(id: "u\(t)", tokens: ReplayCache.user),
                    ReplayMessage(id: "a\(t)", tokens: ReplayCache.ans),
                ]
            }
            messages.append(ReplayMessage(id: "u\(turn)", tokens: ReplayCache.user))
            let b = ReplayCache.boundaries(messages)
            let set = Set((b.rungs + [b.top, b.prompt]).map(\.tokens) + [b.prompt.tokens + ReplayCache.ans])
            return (set.sorted(), b.prompt.tokens)
        }
        #expect(boundarySet(turn: 1).set == [3137, 4240, 4629])
        #expect(boundarySet(turn: 1).prompt == 4240)
        // Turn 2: the only candidate rung is 389 tokens back, under the 512 gap.
        #expect(boundarySet(turn: 2).set == [4629, 5732, 6121])
        let t12 = boundarySet(turn: 12)
        #expect(t12.prompt == 3137 + 11 * 1492 + 1103)
        #expect(t12.prompt + ReplayCache.ans == 21_041)
        // History top, 3 rungs (drop 1 is 389 back: under the gap; drops 2, 4, 8
        // are kept), prompt, post-answer.
        #expect(t12.set.count == 6)

        // A regenerate is a second request for the SAME prompt: it restores the
        // exact-prompt row, stays on its chain, and its shorter answer is a
        // distinct row from the first one.
        let roomy = Self.replay(.regenerate, .plannerPerStore, capTips: 1000)
        #expect(roomy.turns.count == 24)
        #expect(roomy.passes == 0)
        #expect(roomy.turns[0].hit == 0)
        #expect(roomy.turns[1].tag == "regen")
        #expect(roomy.turns[1].hit == roomy.turns[1].prompt)
        // Turn 2 continues from the REGENERATED answer: 3137 + 1103 + 211.
        #expect(roomy.turns[2].hit == 4451)
        #expect(roomy.turns[2].prompt == 5554)
    }

    /// The planner, run the way production runs it (inline, after every store,
    /// with the storing conversation as the active chain), against today's
    /// order on the same workload.
    ///
    /// - `baseline`, `interleaved`, `manyShort`: the planner's restore is at
    ///   least today's on EVERY turn, at every cap.
    /// - `regenerate`: equal or better on every turn at caps >= 2 tips. Below
    ///   that there is a known, accepted residual loss (model: mean -0.0117 at
    ///   0.8 tips, -0.0167 at 1.2, -0.0102 at 1.5; up to 8 987 tokens on an
    ///   affected turn). It is the window where prompt row + tip fit the cap
    ///   but root + prompt row + tip do not: the planner gives up the active
    ///   chain's prompt row before the shared stable root (3c before 4e), and
    ///   today's order keeps whatever was touched last, which is the prompt
    ///   row. Keeping the root is what makes every OTHER conversation's first
    ///   turn cheap, so the trade is accepted; it is bounded here at 0.02 on
    ///   the mean and on the token-weighted reuse. The bound is a property of
    ///   THIS shape: how many turns fall inside that window depends on the
    ///   conversation's length against the cap (the same chat cut to 8 turns
    ///   loses 0.072 at 1.2 tips in the model).
    /// - Per store == end of turn: the inline design rests on the per-store
    ///   planner reusing exactly what an end-of-turn planner would. FAIL CLOSED
    ///   if the two means differ at all in any baseline / interleaved /
    ///   manyShort cell. The end-of-turn arms are otherwise printed controls:
    ///   they let the cache sit several times over the cap inside a turn
    ///   (`peak_eot`), which is why the pass stays inline.
    @Test func tinyCapReuseIsNotWorseThanTodaysOrder() {
        func f(_ x: Double) -> String { String(format: "%.4f", x) }
        func pair(_ r: ReplayResult) -> String { "\(f(r.meanReuse))/\(f(r.tokenWeightedReuse))" }
        var lines = 0
        for shape in ReplayShape.allCases {
            let caps: [Double] = shape == .regenerate
                ? [0.8, 1.2, 1.5, 2.0, 2.5, 5, 10] : [0.8, 1.5, 2.5, 5, 10]
            for capTips in caps {
                let cell = "shape=\(shape.rawValue) cap=\(capTips)"
                let today = Self.replay(shape, .today, capTips: capTips)
                let perStore = Self.replay(shape, .plannerPerStore, capTips: capTips)
                let endOfTurn = Self.replay(shape, .plannerEndOfTurn, capTips: capTips)
                let todayEOT = Self.replay(shape, .todayEndOfTurn, capTips: capTips)
                print(
                    "PLANNER_REPLAY \(cell) today=\(pair(today)) planner_ps=\(pair(perStore))"
                        + " planner_eot=\(pair(endOfTurn)) today_eot=\(pair(todayEOT))"
                        + " peak_ps=\(String(format: "%.2f", perStore.peak))"
                        + " peak_eot=\(String(format: "%.2f", endOfTurn.peak))"
                        + " passes_today=\(today.passes) passes_ps=\(perStore.passes)")
                lines += 1

                // Fail closed: the workload ran, something was reused, every
                // arm really evicted, and no pass or turn left the cache over
                // its cap. 0 == 0 and "never evicts" must not pass.
                var valid = true
                for arm in [today, perStore, endOfTurn, todayEOT] {
                    let ok = arm.turns.count == shape.expectedTurns
                        && arm.turns.contains { $0.hit > 0 } && arm.passes > 0
                        && arm.turnsEndedOverCap == 0 && arm.passesThatLeftTheCacheOverCap == 0
                    #expect(ok, "INVALID \(cell)")
                    valid = valid && ok
                }
                #expect(perStore.peak > 1, "INVALID \(cell)")

                var ok = true
                func check(_ condition: Bool, _ what: String) {
                    #expect(condition, "\(cell): \(what)")
                    ok = ok && condition
                }
                let perTurn = zip(perStore.turns, today.turns).enumerated()
                    .filter { $0.element.0.hit < $0.element.1.hit }
                    .map { "#\($0.offset) planner=\($0.element.0.hit) today=\($0.element.1.hit)" }
                if shape == .regenerate {
                    check(
                        perStore.meanReuse >= today.meanReuse - 0.02, "mean reuse")
                    check(
                        perStore.tokenWeightedReuse >= today.tokenWeightedReuse - 0.02,
                        "token-weighted reuse")
                    if capTips >= 2.0 { check(perTurn.isEmpty, "per-turn \(perTurn)") }
                } else {
                    check(perTurn.isEmpty, "per-turn \(perTurn)")
                    check(
                        abs(perStore.meanReuse - endOfTurn.meanReuse) <= 1e-9,
                        "per-store planner != end-of-turn planner")
                }
                if !ok || !valid {
                    for (name, arm) in [
                        ("planner_ps", perStore), ("planner_eot", endOfTurn), ("today", today),
                    ] {
                        print("PLANNER_REPLAY_TRACE \(cell) policy=\(name)")
                        arm.trace.forEach { print($0) }
                    }
                }
            }
        }
        #expect(lines == 3 * 5 + 7)
    }
}
