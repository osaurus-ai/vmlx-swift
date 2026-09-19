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
/// Ties on recency break on the sort key, which is what makes the answer
/// non-unique when a tie straddles the stopping point.
private func todaysEviction(_ rows: [QuotaRow], cap: Int64) -> (evicted: [QuotaRow], unique: Bool) {
    var remaining = total(rows)
    guard remaining > cap else { return ([], true) }
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
    evicted += lru
    let gone = Set(evicted.map(\.id))
    let kept = rows.filter { !gone.contains($0.id) }
    let unique = !lru.contains { e in
        kept.contains { $0.recency == e.recency && $0.isLegacyCompanion == e.isLegacyCompanion }
    }
    return (evicted, unique)
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
        // cap 3700 -> low 3330: 4423 -> 4014 -> 3703 -> 3350 -> 3019.
        let plan = DiskQuotaPlanner.plan(rows: rows, capBytes: 3700, activeChain: "act")
        #expect(plan.evict == ["cold2:409", "cold1:311", "cold1:353", "act:331"])
        #expect(plan.totalAfter == 3019)

        // A little less pressure and the active chain is not touched at all.
        // cap 4100 -> low 3690: 4423 -> 4014 -> 3703 -> 3350.
        let lighter = DiskQuotaPlanner.plan(rows: rows, capBytes: 4100, activeChain: "act")
        #expect(lighter.evict == ["cold2:409", "cold1:311", "cold1:353"])
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
            #expect(plan.totalAfter > Int64(Double(16832) * DiskQuotaPlanner.lowWatermarkFraction))
        }
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
        // cap 4500 -> low 4050: 4718 -> 4617 -> 4406 -> 4099, nothing left.
        let plan = DiskQuotaPlanner.plan(rows: rows, capBytes: 4500, activeChain: "act")
        #expect(plan.evict == ["act:101", "act:211", "act:307"])
        #expect(plan.totalAfter == 4099)
        #expect(plan.event == DiskCachePressureEvent(
            kind: .activeChainTrimmed, chainId: "act", tipBytes: 4099, capBytes: 4500))
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
        // cap 1000 -> low 900: soft takes the active non-tip (864) and stops.
        let planNamed = DiskQuotaPlanner.plan(rows: named, capBytes: 1000, activeChain: "act")
        #expect(planNamed.evict == ["act:1"])
        #expect(planNamed.event?.kind == .activeChainTrimmed)
        #expect(planNamed.event?.tipBytes == 431)
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
        var compared = 0, skipped = 0, evicting = 0, withOversized = 0
        for _ in 0..<cases {
            let n = Int.random(in: 1...40, using: &rng)
            var ids = (0..<n).map { String(format: "h%04d", $0) }
            ids.shuffle(using: &rng)
            var rows = ids.map { id in
                row(
                    id, tokens: Int.random(in: 1...50_000, using: &rng),
                    bytes: Int64.random(in: 333...1291, using: &rng),
                    // A narrow recency range on purpose: ties happen, and some
                    // straddle the stopping point (the non-unique cases).
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
            guard classic.unique else { skipped += 1; continue }
            let plan = DiskQuotaPlanner.plan(rows: rows, capBytes: cap, activeChain: nil)
            compared += 1
            if !classic.evicted.isEmpty { evicting += 1 }
            if rows.contains(where: { $0.bytes > cap }) { withOversized += 1 }
            #expect(Set(plan.evict) == Set(classic.evicted.map(\.id)))
            #expect(plan.totalBefore == sum)
            #expect(plan.totalAfter == sum - total(classic.evicted))
            #expect(plan.event == nil)
        }
        print(
            "PLANNER_PROPERTY cases=\(cases) compared=\(compared) skipped_non_unique=\(skipped)"
                + " evicting=\(evicting) with_oversized=\(withOversized)")
        // Fail closed: a corpus that mostly skips, or never evicts, proves nothing.
        #expect(compared + skipped == cases)
        #expect(compared >= 300)
        #expect(evicting >= 200)
        #expect(withOversized >= 20)
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

    private enum ReplayPolicy: String { case today, planner }

    private struct ReplayResult {
        var meanReuse = 0.0, resumeReuse = 0.0
        var passes = 0
        var written: Int64 = 0
        /// Turns that ended with the cache over its cap. A policy that never
        /// evicts has perfect reuse; it must not be able to win that way.
        var turnsEndedOverCap = 0
        var trace: [String] = []
    }

    /// A model of a growing conversation cache, the same one as
    /// `harness/sim_policy.py`: bytes == tokens, one shared stable root, and
    /// per turn the history top, up to 3 ladder rungs, the prompt and the
    /// post-answer boundary.
    private final class ReplayCache {
        struct Row { let id: String; let chat: Int?; let tokens: Int; var recency: Int }
        static let sys = 3137, user = 1103, ans = 389
        static let minGap = 512, maxRungs = 3, drops = [1, 2, 4, 8, 16]

        let cap: Int64
        let policy: ReplayPolicy
        var rows: [Row] = []  // insertion order; a re-stored row goes to the end
        var clock = 0
        var result = ReplayResult()

        init(cap: Int64, policy: ReplayPolicy) { self.cap = cap; self.policy = policy }

        static func boundaries(turn: Int) -> (set: [Int], prompt: Int) {
            var ends = [sys]
            for _ in 1..<turn {
                let last = ends[ends.count - 1]
                ends += [last + user, last + user + ans]
            }
            let top = ends[ends.count - 1]
            let prompt = top + user
            var rungs: [Int] = []
            var last = top
            for d in drops {
                if rungs.count >= maxRungs { break }
                let i = ends.count - 1 - d
                if i <= 0 { break }
                if last - ends[i] >= minGap { rungs.append(ends[i]); last = ends[i] }
            }
            return (Set([top, prompt, prompt + ans] + rungs).sorted(), prompt)
        }

        var total: Int64 { rows.reduce(0) { $0 + Int64($1.tokens) } }
        func tick() -> Int { clock += 1; return clock }

        func fetch(chat: Int, prompt: Int) -> Int {
            var best: Int?
            for (i, r) in rows.enumerated() where (r.chat == chat || r.chat == nil) && r.tokens <= prompt {
                if best == nil || r.tokens > rows[best!].tokens { best = i }
            }
            guard let best else { return 0 }
            let now = tick()
            rows[best].recency = now
            if let root = rows.firstIndex(where: { $0.chat == nil }) { rows[root].recency = now }
            return rows[best].tokens
        }

        func store(chat: Int?, tokens: Int) {
            let id = chat.map { "chat\($0):\(tokens)" } ?? "root:\(tokens)"
            guard !rows.contains(where: { $0.id == id }) else { return }
            rows.append(Row(id: id, chat: chat, tokens: tokens, recency: tick()))
            result.written += Int64(tokens)
            if policy == .today { evictTodaysWay() }
        }

        /// The control: per store, oversized first, then oldest recency, to the cap.
        func evictTodaysWay() {
            guard total > cap else { return }
            result.passes += 1
            rows.removeAll { Int64($0.tokens) > cap }
            let order = rows.enumerated()
                .sorted { ($0.element.recency, $0.offset) < ($1.element.recency, $1.offset) }
                .map(\.element.id)
            for id in order {
                if total <= cap { break }
                rows.removeAll { $0.id == id }
            }
        }

        func endTurn(chat: Int) {
            guard policy == .planner else { return }
            if total > cap { result.passes += 1 }
            let plan = DiskQuotaPlanner.plan(
                rows: rows.map {
                    QuotaRow(
                        id: $0.id, tokenCount: $0.tokens, bytes: Int64($0.tokens),
                        recency: Double($0.recency), isStableRoot: $0.chat == nil,
                        chainId: $0.chat.map { "chat\($0)" }, isLegacyCompanion: false)
                },
                capBytes: cap, activeChain: "chat\(chat)")
            let gone = Set(plan.evict)
            rows.removeAll { gone.contains($0.id) }
        }

        func describe() -> String {
            rows.sorted { ($0.chat ?? -1, $0.tokens) < ($1.chat ?? -1, $1.tokens) }
                .map { "\($0.id)@\($0.recency)" }.joined(separator: " ")
        }
    }

    private static func replay(_ policy: ReplayPolicy, capTips: Double) -> ReplayResult {
        let chats = 3, turns = 12
        let tip = ReplayCache.boundaries(turn: turns).prompt + ReplayCache.ans
        let cache = ReplayCache(cap: Int64(capTips * Double(tip)), policy: policy)
        var reuse: [Double] = []
        func oneTurn(chat: Int, turn: Int) -> Double {
            let b = ReplayCache.boundaries(turn: turn)
            let before = cache.describe()
            let hit = cache.fetch(chat: chat, prompt: b.prompt)
            let r = Double(hit) / Double(b.prompt)
            reuse.append(r)
            cache.store(chat: nil, tokens: ReplayCache.sys)
            for n in b.set { cache.store(chat: chat, tokens: n) }
            cache.endTurn(chat: chat)
            if cache.total > cache.cap { cache.result.turnsEndedOverCap += 1 }
            cache.result.trace.append(
                "  chat\(chat) turn \(turn) prompt=\(b.prompt) hit=\(hit) total=\(cache.total)"
                    + "\n    before: \(before)\n    after:  \(cache.describe())")
            return r
        }
        for chat in 0..<chats { for turn in 1...turns { _ = oneTurn(chat: chat, turn: turn) } }
        let resume = oneTurn(chat: 0, turn: turns + 1)
        cache.result.meanReuse = reuse.reduce(0, +) / Double(reuse.count)
        cache.result.resumeReuse = resume
        return cache.result
    }

    @Test func replaySimulatorShapesMatchTheModel() {
        // Pin the scenario itself, so the replay cannot silently drift into a
        // different (easier) workload.
        #expect(ReplayCache.boundaries(turn: 1).set == [3137, 4240, 4629])
        #expect(ReplayCache.boundaries(turn: 1).prompt == 4240)
        // Turn 2: the only candidate rung is 389 tokens back, under the 512 gap.
        #expect(ReplayCache.boundaries(turn: 2).set == [4629, 5732, 6121])
        let t12 = ReplayCache.boundaries(turn: 12)
        #expect(t12.prompt == 3137 + 11 * 1492 + 1103)
        #expect(t12.prompt + ReplayCache.ans == 21_041)
        // History top, 3 rungs (drop 1 is 389 back: under the gap; drops 2, 4, 8
        // are kept), prompt, post-answer.
        #expect(t12.set.count == 6)
    }

    @Test func tinyCapReuseIsNotWorseThanTodaysOrder() {
        func f(_ x: Double) -> String { String(format: "%.4f", x) }
        let tip = Double(ReplayCache.boundaries(turn: 12).prompt + ReplayCache.ans)
        var lines = 0
        for capTips in [0.8, 1.5, 2.5, 5, 10] {
            let today = Self.replay(.today, capTips: capTips)
            let planner = Self.replay(.planner, capTips: capTips)
            print(
                "PLANNER_REPLAY cap=\(capTips) today_reuse=\(f(today.meanReuse))"
                    + " planner_reuse=\(f(planner.meanReuse)) today_resume=\(f(today.resumeReuse))"
                    + " planner_resume=\(f(planner.resumeReuse)) today_passes=\(today.passes)"
                    + " planner_passes=\(planner.passes)"
                    + " today_written=\(f(Double(today.written) / tip))"
                    + " planner_written=\(f(Double(planner.written) / tip))")
            lines += 1
            // Fail closed: a replay in which nothing is ever reused compares 0 with 0.
            #expect(today.meanReuse > 0.5)
            #expect(today.passes > 0)
            #expect(planner.passes > 0)
            #expect(today.turnsEndedOverCap == 0, "cap=\(capTips)")
            #expect(planner.turnsEndedOverCap == 0, "cap=\(capTips)")
            let ok = planner.meanReuse >= today.meanReuse - 0.005
                && planner.resumeReuse >= today.resumeReuse - 0.005
            #expect(planner.meanReuse >= today.meanReuse - 0.005, "cap=\(capTips)")
            #expect(planner.resumeReuse >= today.resumeReuse - 0.005, "cap=\(capTips)")
            if !ok {
                print("PLANNER_REPLAY_TRACE cap=\(capTips) policy=planner")
                planner.trace.forEach { print($0) }
                print("PLANNER_REPLAY_TRACE cap=\(capTips) policy=today")
                today.trace.forEach { print($0) }
            }
        }
        #expect(lines == 5)
    }
}
