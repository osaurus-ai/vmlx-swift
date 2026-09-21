import Foundation
import Testing

@testable import MLXLMCommon

/// Which row of a conversation survives pressure decides whether its next
/// turn is warm. On templates that re-render the assistant turn (Qwen,
/// Nanbeige, Ling) only the history boundary — the prompt cut just before
/// the last user message — ever matches the next prompt; the exact-prompt and
/// post-answer rows never do. Measured live: a chat kept its two useless rows
/// and lost the one useful one, and reused 3 308 of 6 258 tokens instead of
/// 4 164. These numbers are that run's.
@Suite("Disk quota planner: resume boundaries")
struct DiskQuotaResumeBoundaryTests {

    private func row(
        _ id: String, tokens: Int, bytes: Int64? = nil, recency: Double, chain: String? = "A",
        stable: Bool = false, resume: Bool = false
    ) -> QuotaRow {
        QuotaRow(
            id: id, tokenCount: tokens, bytes: bytes ?? Int64(tokens), recency: recency,
            isStableRoot: stable, isResumeBoundary: resume, chainId: chain,
            isLegacyCompanion: false)
    }

    /// The live turn: roots 2 074 and 3 308, history boundary 4 164, exact
    /// prompt 4 171, post-answer 4 191; the cap holds four of the five.
    @Test
    func theHistoryBoundaryOutlivesTheExactPromptAndThePostAnswerRow() throws {
        let rows = [
            row("root-a", tokens: 2_074, recency: 1, chain: nil, stable: true),
            row("root-b", tokens: 3_308, recency: 2, chain: nil, stable: true),
            row("history", tokens: 4_164, recency: 3, resume: true),
            row("exact", tokens: 4_171, recency: 4),
            row("post", tokens: 4_191, recency: 5),
        ]
        let total = rows.reduce(Int64(0)) { $0 + $1.bytes }
        let plan = DiskQuotaPlanner.plan(rows: rows, capBytes: total - 1, activeChain: "A")
        try #require(!plan.evict.isEmpty, "the cap must have forced an eviction")
        #expect(!plan.evict.contains("history"), "the one row the next turn hits survived")
        #expect(plan.evict.first == "exact", "the smallest row with no future goes first")
        #expect(plan.event == nil, "spending rows the next turn never reads is not pressure")
    }

    /// Under harder pressure the order within the chat is: rows with no
    /// future (shortest first), then older history boundaries, and the newest
    /// history boundary last — it is the chat's resume point, not the largest row.
    @Test
    func theNewestHistoryBoundaryIsTheResumePointNotTheLargestRow() throws {
        let rows = [
            row("h1", tokens: 4_164, recency: 1, resume: true),
            row("exact1", tokens: 4_171, recency: 2),
            row("h2", tokens: 6_251, recency: 3, resume: true),
            row("exact2", tokens: 6_258, recency: 4),
            row("post2", tokens: 6_277, recency: 5),
        ]
        // Room for the resume point and nothing else.
        let plan = DiskQuotaPlanner.plan(rows: rows, capBytes: 6_300, activeChain: "A")
        try #require(plan.evict.count == 4)
        #expect(plan.evict == ["exact1", "exact2", "post2", "h1"])
        #expect(!plan.evict.contains("h2"))
        #expect(plan.event?.kind == .activeChainTrimmed)
    }

    /// A cold conversation keeps its history boundary as the row that goes
    /// last too, so coming back to it is warm.
    @Test
    func aColdChatsResumePointIsItsHistoryBoundary() throws {
        let rows = [
            row("cold-h", tokens: 4_164, recency: 1, chain: "C", resume: true),
            row("cold-post", tokens: 4_191, recency: 2, chain: "C"),
            row("active-h", tokens: 8_001, recency: 9, chain: "A", resume: true),
        ]
        let plan = DiskQuotaPlanner.plan(rows: rows, capBytes: 12_500, activeChain: "A")
        try #require(plan.evict == ["cold-post"], "\(plan.evict)")
    }

    /// Without any resume boundary marked, nothing changes: the largest row is
    /// still the tip and the order is what it was.
    @Test
    func chainsWithoutResumeBoundariesOrderExactlyAsBefore() throws {
        let rows = [
            row("a1", tokens: 37, recency: 1),
            row("a2", tokens: 101, recency: 2),
            row("a3", tokens: 149, recency: 3),
        ]
        let plan = DiskQuotaPlanner.plan(rows: rows, capBytes: 200, activeChain: "A")
        #expect(plan.evict == ["a1", "a2"])
    }
}
