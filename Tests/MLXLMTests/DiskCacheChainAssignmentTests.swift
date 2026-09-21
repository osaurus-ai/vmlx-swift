import Foundation
import MLX
import Testing

@testable import MLXLMCommon

/// A conversation id travels with every store so the quota pass can tell the
/// chat in progress from cold ones. Rows written before the id existed are
/// adopted by the first chat that hits them.
@Suite(.serialized)
struct DiskCacheChainAssignmentTests {

    private typealias Support = DiskCacheAccountingTestSupport

    private func makeRoot(_ label: String) -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("vmlx-chain-\(label)-\(UUID().uuidString)")
    }

    private func tokens(_ count: Int, seed: Int) -> [Int] {
        (0 ..< count).map { seed * 100_000 + $0 }
    }

    private func kv(_ elements: Int = 1_031) -> [String: MLXArray] {
        ["data": MLXArray.ones([elements], dtype: .float32)]
    }

    private func coordinator(root: URL, capBytes: Int64 = 1 << 30, modelKey: String = "m")
        throws -> CacheCoordinator
    {
        try #require(capBytes == 1 << 30 || capBytes < 1 << 24)
        let config = CacheCoordinatorConfig(
            usePagedCache: false,
            enableDiskCache: true,
            diskCacheMaxGB: Float(capBytes) / 1_073_741_824,
            diskCacheDir: root,
            modelKey: modelKey)
        let coordinator = CacheCoordinator(config: config)
        try #require(Int64(try #require(coordinator.diskCache).maxSizeBytes) == capBytes)
        try #require(try #require(coordinator.diskCache).indexHasV2Columns)
        return coordinator
    }

    private struct Row {
        let hash: String
        let kind: Int
        let chain: String?
        let tokens: Int
    }

    private func rows(_ root: URL) throws -> [Row] {
        let raw = try Support.RawDB(root: root)
        return try raw.rows(
            "SELECT hash, kind, chain_id, token_count FROM cache_entries ORDER BY token_count"
        ).map { r in
            Row(
                hash: r[0] ?? "", kind: Int(r[1] ?? "0") ?? -1, chain: r[2],
                tokens: Int(r[3] ?? "0") ?? -1)
        }
    }

    private func hash(_ tokens: [Int], _ modelKey: String = "m") -> String {
        DiskCache.hashTokens(tokens, modelKey: modelKey, mediaSalt: nil)
    }

    @Test
    func storeWritesTheChainAndTheKind() throws {
        let root = makeRoot("write")
        defer { try? FileManager.default.removeItem(at: root) }
        let c = try coordinator(root: root)
        let history = tokens(37, seed: 1)
        let stable = tokens(11, seed: 1)
        c.storePersistentBoundary(
            tokens: history, diskArrays: kv(), ssmStates: nil, chainId: "A", isStableRoot: false)
        c.storePersistentBoundary(
            tokens: stable, diskArrays: kv(), ssmStates: nil, chainId: "A", isStableRoot: true)
        let after = try rows(root)
        try #require(after.count == 2)
        #expect(after[0].tokens == 11 && after[0].kind == 1)
        #expect(after[1].tokens == 37 && after[1].kind == 0 && after[1].chain == "A")

        // A second chat re-storing the same history row takes it over; a
        // re-store without an id keeps the id already there; a stable root
        // never loses its kind.
        c.storePersistentBoundary(
            tokens: history, diskArrays: kv(), ssmStates: nil, chainId: "B", isStableRoot: false)
        c.storePersistentBoundary(
            tokens: stable, diskArrays: kv(), ssmStates: nil, chainId: nil, isStableRoot: false)
        let again = try rows(root)
        #expect(again[1].chain == "B")
        #expect(again[0].kind == 1 && again[0].chain == "A")
    }

    @Test
    func aV1IndexAcceptsAChainWithoutColumns() throws {
        let root = makeRoot("v1")
        defer { try? FileManager.default.removeItem(at: root) }
        try Support.makeV1OnlyIndex(in: root)
        let disk = DiskCache(cacheDir: root, maxSizeBytes: 1 << 30, modelKey: "m")
        try #require(!disk.indexHasV2Columns)
        disk.store(
            tokens: tokens(37, seed: 2), arrays: kv(), enforceQuota: false, chainId: "A",
            isStableRoot: true)
        #expect(try Support.RawDB(root: root).rows("SELECT hash FROM cache_entries").count == 1)
    }

    @Test
    func theChatInProgressLosesNothingWhileColdChatsHaveRowsToGive() throws {
        let root = makeRoot("protect")
        defer { try? FileManager.default.removeItem(at: root) }
        // Each row is ~4.1 KB; cap for about four of them.
        let c = try coordinator(root: root, capBytes: 17_000)
        for i in 1 ... 3 {
            c.storePersistentBoundary(
                tokens: tokens(20 + i * 7, seed: 10), diskArrays: kv(), ssmStates: nil,
                chainId: "A", isStableRoot: false)
        }
        try #require(try rows(root).count == 3)
        for i in 1 ... 3 {
            c.storePersistentBoundary(
                tokens: tokens(20 + i * 7, seed: 20), diskArrays: kv(), ssmStates: nil,
                chainId: "B", isStableRoot: false)
        }
        let survivors = try rows(root)
        try #require(!survivors.isEmpty)
        #expect(survivors.filter { $0.chain == "B" }.count == 3, "the active chain kept every row")
        #expect(survivors.filter { $0.chain == "A" }.count < 3, "cold chain A paid")
        let stats = try #require(c.diskCache).snapshotStats()
        #expect(stats.evictions > 0)
        #expect(stats.lastPressureEvent == nil, "no pressure on the active chain")
    }

    @Test
    func trimmingTheChatInProgressIsReportedWithItsId() throws {
        let root = makeRoot("trim")
        defer { try? FileManager.default.removeItem(at: root) }
        // Room for two rows only, all of them the active chain's.
        let c = try coordinator(root: root, capBytes: 9_000)
        for i in 1 ... 3 {
            c.storePersistentBoundary(
                tokens: tokens(20 + i * 7, seed: 30), diskArrays: kv(), ssmStates: nil,
                chainId: "B", isStableRoot: false)
        }
        let stats = try #require(c.diskCache).snapshotStats()
        let event = try #require(stats.lastPressureEvent)
        #expect(event.kind == .activeChainTrimmed)
        #expect(event.chainId == "B")
        let survivors = try rows(root)
        #expect(survivors.map(\.tokens).max() == 41, "the tip survives")
    }

    @Test
    func aTipThatCannotFitIsReportedAsDropped() throws {
        let root = makeRoot("drop")
        defer { try? FileManager.default.removeItem(at: root) }
        let c = try coordinator(root: root, capBytes: 3_000)
        c.storePersistentBoundary(
            tokens: tokens(37, seed: 40), diskArrays: kv(), ssmStates: nil,
            chainId: "B", isStableRoot: false)
        let stats = try #require(c.diskCache).snapshotStats()
        let event = try #require(stats.lastPressureEvent)
        #expect(event.kind == .activeTipDropped)
        #expect(event.chainId == "B")
        #expect(try rows(root).isEmpty)
    }

    @Test
    func aHitAdoptsAnUnassignedRowIntoTheChatThatUsedIt() throws {
        let root = makeRoot("adopt")
        defer { try? FileManager.default.removeItem(at: root) }
        let c = try coordinator(root: root)
        let history = tokens(37, seed: 50)
        let stable = tokens(11, seed: 50)
        c.storePersistentBoundary(tokens: history, diskArrays: kv(), ssmStates: nil)
        c.storePersistentBoundary(
            tokens: stable, diskArrays: kv(), ssmStates: nil, chainId: nil, isStableRoot: true)
        try #require(try rows(root).allSatisfy { $0.chain == nil })
        let prompt = history + tokens(5, seed: 51)
        let result = c.fetch(tokens: prompt, chainId: "C")
        guard case .hit(let matched, _, _, _, _, _) = result else {
            Issue.record("expected a hit, got \(result)")
            return
        }
        #expect(matched == 37)
        let after = try rows(root)
        #expect(after.first { $0.tokens == 37 }?.chain == "C")
        #expect(after.first { $0.tokens == 11 }?.chain == nil, "a shared stable root stays unowned")
        // A row that already belongs to another chat is not taken by a read.
        _ = c.fetch(tokens: prompt, chainId: "D")
        #expect(try rows(root).first { $0.tokens == 37 }?.chain == "C")
    }

    /// A hybrid model's own FETCH writes the recurrent state it found folded
    /// into the payload back out as a companion, and that store runs a quota
    /// pass. If the pass does not know which conversation is reading, that
    /// pass treats every chain as cold and spends the COLDEST chain's
    /// superseded rows first — and the conversation being read is the coldest
    /// one precisely when the user has just come back to an older chat. Its
    /// regenerate/edit boundaries are then gone, mid-conversation, caused by a
    /// read. Here the reader's rows are the oldest, so the two orders
    /// disagree and the test can tell them apart.
    @Test
    func aFetchThatWritesBackACompanionDoesNotSpendTheReadingChatsRows() throws {
        let root = makeRoot("readback")
        defer { try? FileManager.default.removeItem(at: root) }
        let c = try coordinator(root: root, capBytes: 18_000)
        c.setHybrid(true, requiresRecurrentSSMCompanion: true)
        let disk = try #require(c.diskCache)

        // The reader: three boundaries of one conversation, the oldest rows in
        // the cache, its tip carrying recurrent state folded into the payload
        // and no companion file — so the fetch has to write one back.
        var mine: [Int] = []
        for i in 1 ... 3 {
            let t = tokens(16 + i * 7, seed: 60)
            var payload = kv()
            if i == 3 {
                mine = t
                payload["__ssm_count__"] = MLXArray([Int32(1)])
                payload["ssm_0"] = MLXArray.ones([257], dtype: .float32)
            }
            disk.store(tokens: t, arrays: payload, enforceQuota: false, chainId: "B")
            try #require(
                disk.touchRecency(tokens: t, at: Date(timeIntervalSince1970: 1_000 + Double(i))))
        }
        // A conversation that is not reading, with newer rows.
        for i in 1 ... 3 {
            let t = tokens(16 + i * 7, seed: 70)
            disk.store(tokens: t, arrays: kv(), enforceQuota: false, chainId: "A")
            try #require(
                disk.touchRecency(tokens: t, at: Date(timeIntervalSince1970: 9_000 + Double(i))))
        }
        try #require(try rows(root).count == 6)

        let result = c.fetch(tokens: mine + tokens(5, seed: 61), chainId: "B")
        guard case .hit(let matched, _, _, _, _, _) = result else {
            Issue.record("expected the folded row to be served, got \(result)")
            return
        }
        #expect(matched == 37)

        let after = try rows(root)
        try #require(after.count < 6, "the write-back must have run a pass that evicted")
        let kept = { (chain: String) in after.filter { $0.chain == chain }.map(\.tokens).sorted() }
        let mineKept = kept("B")
        let othersKept = kept("A")
        let state: Comment = "reader kept \(mineKept), the other chat kept \(othersKept)"
        // The row just read is the reader's tip and must always survive.
        #expect(mineKept.contains(37), state)
        // The policy: the reader's superseded boundaries are spent only after
        // the chat that is not reading has no superseded row left. Without the
        // reading chat named, this pass ate the reader's rows FIRST (measured:
        // it kept only its tip while the other chat kept all three).
        if mineKept.count < 3 {
            #expect(othersKept == [37], "the other chat's superseded rows go first; \(state)")
        }
    }
}
