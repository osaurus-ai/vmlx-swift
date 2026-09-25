import Testing

@testable import MLXLMCommon

@Suite struct CanonicalCheckpointQuotaTests {
    private func row(
        _ id: String, tokens: Int, recency: Double = 1,
        checkpoint: Bool = false, resume: Bool = false,
        chain: String = "active"
    ) -> QuotaRow {
        QuotaRow(
            id: id, tokenCount: tokens, bytes: 100, recency: recency,
            isStableRoot: false, isResumeBoundary: resume,
            isCanonicalCheckpoint: checkpoint, chainId: chain,
            isLegacyCompanion: false)
    }

    @Test func currentCheckpointSurvivesDiscardableSnapshots() {
        let rows = [
            row("resume", tokens: 10057, resume: true),
            row("checkpoint", tokens: 9728, checkpoint: true),
            row("old-checkpoint", tokens: 9216, recency: 0, checkpoint: true),
            row("ordinary", tokens: 10058),
        ]
        let plan = DiskQuotaPlanner.plan(rows: rows, capBytes: 200, activeChain: "active")
        #expect(Set(plan.evict) == ["old-checkpoint", "ordinary"])
        #expect(plan.totalAfter == 200)
        #expect(plan.event == nil)
        #expect(DiskQuotaPlanner.resumePoint(of: rows)?.id == "resume")
    }

    @Test func capAlwaysWinsButActualResumePointOutlivesCheckpoint() {
        let rows = [
            row("resume", tokens: 10057, resume: true),
            row("checkpoint", tokens: 9728, checkpoint: true),
        ]
        let plan = DiskQuotaPlanner.plan(rows: rows, capBytes: 100, activeChain: "active")
        #expect(plan.evict == ["checkpoint"])
        #expect(plan.totalAfter == 100)
        #expect(plan.event == nil)
    }

    @Test func coldCheckpointYieldsBeforeOtherChatsResumePoint() {
        let rows = [
            row("active-resume", tokens: 10057, resume: true),
            row("active-chunk", tokens: 9728, checkpoint: true),
            row("cold-resume", tokens: 11000, resume: true, chain: "cold"),
            row("cold-chunk", tokens: 10752, checkpoint: true, chain: "cold"),
        ]
        let plan = DiskQuotaPlanner.plan(rows: rows, capBytes: 300, activeChain: "active")
        #expect(plan.evict == ["cold-chunk"])
        #expect(plan.totalAfter == 300)
    }

    @Test func editedShorterCheckpointReplacesOlderLongerOne() {
        let rows = [
            row("resume", tokens: 10057, resume: true),
            row("latest", tokens: 512, recency: 2, checkpoint: true),
            row("obsolete", tokens: 9728, recency: 1, checkpoint: true),
        ]
        let plan = DiskQuotaPlanner.plan(rows: rows, capBytes: 200, activeChain: "active")
        #expect(plan.evict == ["obsolete"])
    }

    @Test func checkpointAloneNeverReportsLostChatProgress() {
        let rows = [row("chunk", tokens: 512, checkpoint: true)]
        let plan = DiskQuotaPlanner.plan(rows: rows, capBytes: 50, activeChain: "active")
        #expect(plan.evict == ["chunk"])
        #expect(plan.event == nil)
        #expect(DiskQuotaPlanner.resumePoint(of: rows) == nil)
    }

    @Test func coldCheckpointYieldsBeforeActiveEditBoundary() {
        let rows = [
            row("current", tokens: 10057, resume: true),
            row("edit", tokens: 10028, resume: true),
            row("cold-resume", tokens: 11000, resume: true, chain: "cold"),
            row("cold-chunk", tokens: 10752, checkpoint: true, chain: "cold"),
        ]
        let plan = DiskQuotaPlanner.plan(rows: rows, capBytes: 300, activeChain: "active")
        #expect(plan.evict == ["cold-chunk"])
        #expect(plan.event == nil)
    }

    @Test func checkpointOnlyColdChainDoesNotDisplaceActiveCheckpoint() {
        let rows = [
            row("current", tokens: 10057, resume: true),
            row("active-chunk", tokens: 9728, checkpoint: true),
            row("cold-chunk", tokens: 10752, checkpoint: true, chain: "cold"),
        ]
        let plan = DiskQuotaPlanner.plan(rows: rows, capBytes: 200, activeChain: "active")
        #expect(plan.evict == ["cold-chunk"])
    }

    @Test func checkpointOnlyActiveChainDoesNotDisplaceRealResumePoint() {
        let rows = [
            row("active-chunk", tokens: 9728, checkpoint: true),
            row("cold-resume", tokens: 11000, resume: true, chain: "cold"),
        ]
        let plan = DiskQuotaPlanner.plan(rows: rows, capBytes: 100, activeChain: "active")
        #expect(plan.evict == ["active-chunk"])
        #expect(plan.event == nil)
    }

    @Test func everyCapIsBoundedAndIndependentOfInputOrder() {
        let rows = [
            row("resume", tokens: 10057, resume: true),
            row("latest", tokens: 9728, recency: 3, checkpoint: true),
            row("older", tokens: 9216, checkpoint: true),
            row("cold", tokens: 11000, resume: true, chain: "cold"),
        ]
        for cap in stride(from: Int64(0), through: 500, by: 25) {
            let plan = DiskQuotaPlanner.plan(rows: rows, capBytes: cap, activeChain: "active")
            #expect(
                plan
                    == DiskQuotaPlanner.plan(
                        rows: rows.reversed(), capBytes: cap, activeChain: "active"))
            #expect(plan.totalAfter <= cap)
            #expect(Set(plan.evict).count == plan.evict.count)
            #expect(
                plan.totalAfter
                    == rows.filter { !plan.evict.contains($0.id) }.reduce(0) { $0 + $1.bytes })
        }
    }
}
