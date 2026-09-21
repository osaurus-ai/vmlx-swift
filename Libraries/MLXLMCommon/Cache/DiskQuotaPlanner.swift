import Foundation

/// Arithmetic on byte counts that came out of `cache_index.db`.
///
/// The index is data. `file_size = 1e19` stays REAL in its INTEGER column and
/// reads back as `Int64.max`, and a negative size is as easy to write: a
/// plain `+` over the first TRAPS, and the second hides the bytes of every
/// other row. So a count is clamped to `0...` where it is read, and every
/// sum of counts saturates at `Int64.max` instead of overflowing. A saturated
/// total is a lower bound, never a wrapped number, and a row that claims more
/// than any volume holds is an ordinary row: oversized, so it goes first.
enum IndexedBytes {
    static func clamped(_ value: Int64) -> Int64 { max(0, value) }

    static func sum(_ a: Int64, _ b: Int64) -> Int64 {
        let (total, overflow) = clamped(a).addingReportingOverflow(clamped(b))
        return overflow ? .max : total
    }

    static func total<Counts: Sequence>(_ counts: Counts) -> Int64 where Counts.Element == Int64 {
        counts.reduce(0, sum)
    }

    /// `a - b`, not below 0. Both are counts, so this cannot overflow.
    static func difference(_ a: Int64, _ b: Int64) -> Int64 {
        max(0, clamped(a) - clamped(b))
    }

    /// For ``DiskCacheStats``, whose usage figure is an `Int`.
    static func asInt(_ value: Int64) -> Int { Int(clamping: clamped(value)) }
}

/// One evictable unit as the planner sees it. Pure data.
///
/// `id` must be unique within one `plan` call (it is the KV hash, a primary
/// key, or `"legacy:<key>"` for a legacy companion).
struct QuotaRow: Equatable, Sendable {
    /// KV hash, or `"legacy:<key>"` for a legacy companion.
    let id: String
    /// Prefix length of the snapshot. 0 for legacy companions.
    let tokenCount: Int
    /// `file_size + companion_bytes`, or the legacy companion's bytes. A
    /// negative count is 0 bytes, and every sum saturates (``IndexedBytes``).
    let bytes: Int64
    /// Larger = more recently used.
    let recency: Double
    /// `kind == 1`: system prompt + tools, shared across conversations.
    let isStableRoot: Bool
    /// The conversation. `nil` = a row written before chains existed: it is
    /// its own single-row chain.
    let chainId: String?
    /// A companion payload with no KV row.
    let isLegacyCompanion: Bool
}

public enum DiskCachePressureKind: String, Sendable, Equatable {
    /// The active conversation's newest snapshot does not fit under the cap at all.
    case activeTipDropped
    /// Rows of the active conversation had to be evicted to get under the cap.
    case activeChainTrimmed
}

/// The disk cap is too small for the conversation in progress. Advisory: the
/// plan it rides on is carried out either way.
public struct DiskCachePressureEvent: Sendable, Equatable {
    public let kind: DiskCachePressureKind
    /// The active conversation.
    public let chainId: String?
    /// Bytes of the active conversation's newest snapshot, before this pass.
    public let tipBytes: Int64
    public let capBytes: Int64
}

struct QuotaPlan: Equatable {
    /// Row ids, in eviction order. No id appears twice.
    let evict: [String]
    let evictedBytes: Int64
    let totalBefore: Int64
    /// Always `totalBefore - evictedBytes`.
    let totalAfter: Int64
    let event: DiskCachePressureEvent?

    /// A selected victim is not proof of lost progress: its file deletion can fail.
    /// `lostRows` includes a KV payload or required companion actually removed.
    func confirmedEvent(rows: [QuotaRow], lostRows: Set<String>) -> DiskCachePressureEvent? {
        guard let event else { return nil }
        let active = rows.filter {
            !$0.isStableRoot && !$0.isLegacyCompanion && $0.chainId == event.chainId
        }
        if event.kind == .activeTipDropped {
            let tip = active.max {
                if $0.tokenCount != $1.tokenCount { return $0.tokenCount < $1.tokenCount }
                if $0.recency != $1.recency { return $0.recency < $1.recency }
                return $0.id < $1.id
            }
            return tip.map { lostRows.contains($0.id) } == true ? event : nil
        }
        return active.contains { lostRows.contains($0.id) } ? event : nil
    }
}

/// The disk prefix cache's eviction policy as a pure function: rows + cap +
/// the conversation in progress in, an ordered eviction list out. No IO, no
/// SQLite, no locks, no clock.
///
/// Vocabulary. A **chain** is every non-stable, non-legacy row sharing a
/// `chainId`; a row with `chainId == nil` is a chain of one. A chain's **tip**
/// is its row with the greatest `tokenCount` (ties: greatest recency, then
/// greatest `id`). **Chain recency** is the greatest recency among its rows.
/// The **active** chain is the one named by `activeChain`; every other chain
/// is **cold**. `low = Int64(Double(cap) * lowWatermarkFraction)`.
///
/// The policy:
///
/// 1. `total <= cap`: nothing. The trigger is the CAP, not the low watermark.
/// 2. **Oversized first.** A row with `bytes > cap` can never fit and goes
///    before anything else, so the best prior prefix that does fit is not
///    sacrificed to make room for it. If it is the active chain's tip the
///    plan carries `.activeTipDropped`. If the total now fits the cap the pass
///    STOPS: an oversized row is no reason to chase the low watermark. Tips
///    and chain recency below are taken over the rows that survive this step,
///    so a chain whose newest snapshot was oversized keeps its best fitting
///    one as its resume point.
/// 3. **Soft phase,** paid only with rows that are not a resume point:
///    (a) legacy companions, oldest first, down to `low`; (b) non-tip rows of
///    cold chains, coldest chain first, smallest `tokenCount` first within a
///    chain, down to `low`; (c) non-tip rows of the active chain, smallest
///    `tokenCount` first, down to the CAP only. Each step stops as soon as
///    its goal is met, or when it runs out of such rows.
///
///    Why (c) stops at the cap: the conversation in progress is about to read
///    its own non-tip rows again. A regenerate restores from the exact-prompt
///    row, one boundary below the tip; an edit restores from further down.
///    Spending those on hysteresis trades a certain re-prefill now for fewer
///    evicting passes later. Cold chains' superseded rows have no such reader.
/// 4. **Hard phase, only while `total > cap`:** (d) cold chains' tips, coldest
///    chain first; (e) stable roots, oldest first; (f) the active chain's tip,
///    last. Stops as soon as `total <= cap`.
///
/// Why tips never pay for hysteresis: a tip is a conversation's resume point,
/// and losing it costs a full cold prefill of that conversation. The low
/// watermark only buys fewer evicting passes; that is worth legacy companions
/// and cold conversations' superseded snapshots, and nothing more. And why the stable root yields before the
/// active tip: for the conversation in progress the tip is a superset prefix
/// of the stable root, so the root saves nothing the tip does not.
///
/// (f) is the end of the order rather than a step that runs: by then every
/// other row is gone and a tip that survived step 2 fits the cap on its own.
///
/// 5. If a row of the active chain went in (c) or (f) and the tip was not
///    already reported dropped, the plan carries `.activeChainTrimmed` with
///    the bytes of the active tip as it was before the pass. Evicting only
///    cold, legacy or stable rows is not an event.
/// 6. The plan does not depend on the order of `rows`: every ordering below
///    is total and ends on `id`.
/// 7. A cache migrated from the old schema — every `chainId` nil, no stable
///    roots, no active chain — has no superseded rows to find: every row is a
///    tip, the soft phase has no KV row to take, and (d) is oldest recency
///    first to exactly the cap. With no legacy companions that is the old
///    order exactly, oversized-first included. With legacy companions it is
///    the old order plus hysteresis on them alone: both policies take legacy
///    companions first, oldest first, but the old one stopped at the cap and
///    (a) goes on to `low`. The KV rows evicted are never more than the old
///    policy's; the legacy companions evicted are never fewer.
enum DiskQuotaPlanner {
    static let lowWatermarkFraction = 0.90

    private enum ChainKey: Hashable {
        case named(String)
        /// A chain-less row, keyed by its own id so it can never collide with
        /// a named chain or with another chain-less row.
        case solo(String)
    }

    private struct Chain {
        let tip: QuotaRow
        /// Smallest `tokenCount` first.
        let superseded: [QuotaRow]
        let recency: Double
    }

    static func plan(rows: [QuotaRow], capBytes: Int64, activeChain: String?) -> QuotaPlan {
        assert(Set(rows.map(\.id)).count == rows.count, "QuotaRow ids must be unique")
        let totalBefore = IndexedBytes.total(rows.lazy.map(\.bytes))
        var total = totalBefore
        var evict: [String] = []
        var event: DiskCachePressureEvent?

        func finish() -> QuotaPlan {
            QuotaPlan(
                evict: evict, evictedBytes: totalBefore - total,
                totalBefore: totalBefore, totalAfter: total, event: event)
        }
        /// Evicts from `candidates`, in order, until `total <= goal`.
        /// Returns whether it took anything.
        func drain(_ candidates: [QuotaRow], to goal: Int64) -> Bool {
            var took = false
            for row in candidates {
                if total <= goal { break }
                evict.append(row.id)
                total = IndexedBytes.difference(total, row.bytes)
                took = true
            }
            return took
        }

        // 1. The trigger is the cap.
        guard total > capBytes else { return finish() }

        let activeKey = activeChain.map(ChainKey.named)
        let activeTipBefore =
            rows
            .filter { isChainRow($0) && chainKey($0) == activeKey }
            .max(by: tipOrder)

        // 2. Rows that can never fit.
        let oversized = rows.filter { $0.bytes > capBytes }.sorted(by: oldestFirst)
        if let tip = activeTipBefore, oversized.contains(where: { $0.id == tip.id }) {
            event = DiskCachePressureEvent(
                kind: .activeTipDropped, chainId: activeChain,
                tipBytes: tip.bytes, capBytes: capBytes)
        }
        _ = drain(oversized, to: .min)
        if totalBefore == .max {
            // A total that saturated on the way up is a lower bound, and what
            // was taken off it says nothing about what is left: count that.
            total = IndexedBytes.total(rows.lazy.filter { $0.bytes <= capBytes }.map(\.bytes))
        }
        guard total > capBytes else { return finish() }

        let fitting = rows.filter { $0.bytes <= capBytes }
        let chains = Dictionary(grouping: fitting.filter(isChainRow), by: chainKey)
            .mapValues { members -> Chain in
                let tip = members.max(by: tipOrder)!
                return Chain(
                    tip: tip,
                    superseded: members.filter { $0.id != tip.id }.sorted(by: shortestFirst),
                    recency: members.reduce(-Double.infinity) { max($0, $1.recency) })
            }
        let active = activeKey.flatMap { chains[$0] }
        let cold = chains.filter { $0.key != activeKey }.values.sorted {
            if $0.recency != $1.recency { return $0.recency < $1.recency }
            return $0.tip.id < $1.tip.id
        }

        // 3. Soft phase: never with a resume point. Legacy and cold rows pay
        // for the low watermark; the active conversation's rows only for the cap.
        let low = Int64(Double(capBytes) * lowWatermarkFraction)
        _ = drain(fitting.filter(\.isLegacyCompanion).sorted(by: oldestFirst), to: low)
        _ = drain(cold.flatMap(\.superseded), to: low)
        var trimmedActive = drain(active?.superseded ?? [], to: capBytes)

        // 4. Hard phase: only while still over the cap.
        _ = drain(cold.map(\.tip), to: capBytes)
        _ = drain(
            fitting.filter { $0.isStableRoot && !$0.isLegacyCompanion }.sorted(by: oldestFirst),
            to: capBytes)
        if let active, drain([active.tip], to: capBytes) { trimmedActive = true }

        // 5. The pressure event.
        if trimmedActive, event == nil, let tip = activeTipBefore {
            event = DiskCachePressureEvent(
                kind: .activeChainTrimmed, chainId: activeChain,
                tipBytes: tip.bytes, capBytes: capBytes)
        }
        return finish()
    }

    private static func isChainRow(_ row: QuotaRow) -> Bool {
        !row.isStableRoot && !row.isLegacyCompanion
    }

    private static func chainKey(_ row: QuotaRow) -> ChainKey {
        row.chainId.map(ChainKey.named) ?? .solo(row.id)
    }

    /// `max(by:)` order for a chain's tip: tokens, then recency, then id.
    private static func tipOrder(_ a: QuotaRow, _ b: QuotaRow) -> Bool {
        if a.tokenCount != b.tokenCount { return a.tokenCount < b.tokenCount }
        if a.recency != b.recency { return a.recency < b.recency }
        return a.id < b.id
    }

    private static func oldestFirst(_ a: QuotaRow, _ b: QuotaRow) -> Bool {
        if a.recency != b.recency { return a.recency < b.recency }
        return a.id < b.id
    }

    private static func shortestFirst(_ a: QuotaRow, _ b: QuotaRow) -> Bool {
        if a.tokenCount != b.tokenCount { return a.tokenCount < b.tokenCount }
        return oldestFirst(a, b)
    }
}
