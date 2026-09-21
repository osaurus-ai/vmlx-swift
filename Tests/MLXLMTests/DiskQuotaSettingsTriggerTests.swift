import Foundation
import MLX
import Testing

@testable import MLXLMCommon

/// How a disk-cache size setting reaches the quota pass, and whose cap that
/// pass enforces. These pin what the code does; two of them pin a policy that
/// is known to be a problem, so that a change to it is a decision and not an
/// accident.
///
/// No payload size is round, and no token length is a multiple of 64.
@Suite(.serialized)
struct DiskQuotaSettingsTriggerTests {

    private typealias Support = DiskCacheAccountingTestSupport

    private static func makeRoot(_ label: String) -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("vmlx-quota-settings-\(label)-\(UUID().uuidString)")
    }

    private static func tokens(_ count: Int, seed: Int) -> [Int] {
        (0 ..< count).map { seed * 100_000 + $0 }
    }

    private static func payload(_ elements: Int) -> [String: MLXArray] {
        ["data": MLXArray.ones([elements], dtype: .float32)]
    }

    private static func coordinator(
        root: URL, modelKey: String, capBytes: Int64
    ) -> CacheCoordinator {
        CacheCoordinator(
            config: CacheCoordinatorConfig(
                usePagedCache: false,
                enableDiskCache: true,
                diskCacheMaxGB: Float(capBytes) / 1_073_741_824,
                diskCacheDir: root,
                modelKey: modelKey))
    }

    // MARK: - 9. One root, two caps

    /// The cap belongs to a coordinator; the root is shared. Whichever
    /// coordinator stores last runs the pass, over every model's rows, with
    /// its own cap: a model loaded with a small cap evicts what a model with
    /// a generous cap was entitled to keep. The evictions are counted on the
    /// coordinator that ran the pass, not on the one that lost the rows.
    @Test func twoCapsOneRootTheLastWriterEnforcesItsCapOverEveryModel() throws {
        try MLXMetalTestLock.withLock {
            let root = Self.makeRoot("two-caps")
            defer { try? FileManager.default.removeItem(at: root) }
            let generousCap: Int64 = 250_007
            let smallCap: Int64 = 50_003
            let generous = Self.coordinator(root: root, modelKey: "A", capBytes: generousCap)
            let generousDisk = try #require(generous.diskCache)

            let base = Date(timeIntervalSinceNow: -7_200)
            var hashesOldestFirst: [String] = []
            for index in 0 ..< 7 {
                let tokens = Self.tokens(11 + index, seed: index + 1)
                generous.storePersistentBoundary(
                    tokens: tokens, diskArrays: Self.payload(3_001), ssmStates: nil)
                // Stores within one millisecond tie on recency; spread them.
                try #require(
                    generousDisk.touchRecency(
                        tokens: tokens, mediaSalt: nil,
                        at: base.addingTimeInterval(Double(index) * 60)))
                hashesOldestFirst.append(DiskCache.hashTokens(tokens, modelKey: "A"))
            }
            let filled = try #require(generous.snapshotStats().diskStats)
            try #require(filled.currentEntryCount == 7)
            try #require(filled.evictions == 0)
            try #require(Int64(filled.currentPayloadBytes) > smallCap)
            try #require(Int64(filled.currentPayloadBytes) <= generousCap)

            let small = Self.coordinator(root: root, modelKey: "B", capBytes: smallCap)
            // Opening runs a pass too, with the same effect: the generous
            // model's rows start to go before this model has stored anything.
            let opened = try #require(small.snapshotStats().diskStats)
            #expect(opened.evictions > 0)
            #expect(Int64(opened.currentPayloadBytes) <= smallCap)
            let smallTokens = Self.tokens(37, seed: 99)
            small.storePersistentBoundary(
                tokens: smallTokens, diskArrays: Self.payload(3_001), ssmStates: nil)

            let remaining = Set(try Support.indexedRows(root).map(\.hash))
            let survivors = hashesOldestFirst.map(remaining.contains)
            try #require(survivors.count == 7)
            #expect(survivors.contains(false), "none of the generous model's rows was evicted")
            // Oldest first: once one row survives, every newer one does.
            let firstSurvivor = survivors.firstIndex(of: true) ?? survivors.count
            #expect(survivors[firstSurvivor...].allSatisfy { $0 })
            #expect(remaining.contains(DiskCache.hashTokens(smallTokens, modelKey: "B")))

            let after = try #require(small.snapshotStats().diskStats)
            #expect(Int64(after.currentPayloadBytes) <= smallCap)
            #expect(after.currentPayloadBytes > 0)
            #expect(after.evictions > opened.evictions, "the store's own pass evicted nothing")
            #expect(after.evictions == survivors.filter { !$0 }.count)
            #expect(try #require(generous.snapshotStats().diskStats).evictions == 0)
        }
    }

    // MARK: - 10. The resolver

    /// The shared arithmetic is deterministic; the public resolver's file IO
    /// path is checked separately with an explicit cap on an unknown volume.
    @Test func resolveDiskCacheMaxGBTable() {
        let gb: Int64 = 1_073_741_824
        struct Row {
            let percent: Double?
            let legacyGB: Double?
            let capGB: Double
        }
        let table = [
            Row(percent: nil, legacyGB: nil, capGB: 30),
            Row(percent: nil, legacyGB: 50, capGB: 25),
            Row(percent: 10, legacyGB: nil, capGB: 25),
            Row(percent: 10, legacyGB: 50, capGB: 25),
            Row(percent: 0.01, legacyGB: nil, capGB: 0.1),
            Row(percent: 0.01, legacyGB: 50, capGB: 0.1),
            Row(percent: 0, legacyGB: nil, capGB: 30),
            Row(percent: -5, legacyGB: 50, capGB: 25),
            Row(percent: 100, legacyGB: nil, capGB: 25),
            Row(percent: 1000, legacyGB: nil, capGB: 30),
        ]
        for row in table {
            let result = DiskCacheCapPolicy.resolve(
                percent: row.percent, legacyGB: row.legacyGB,
                totalBytes: 1000 * gb, freeBytes: 80 * gb, ownBytes: 20 * gb)
            #expect(abs(result.capGB - row.capGB) < 1e-8)
        }
        #expect(VMLXServerRuntimeSettings.resolveDiskCacheMaxGB(
            percent: 1, legacyGB: 50, directory: nil) == 10)
    }

    // MARK: - 11. The companion store's own cap

    /// A companion written directly to the store (not through the
    /// coordinator's linked store) is followed by the store's OWN pass. That
    /// pass compares companion bytes alone with a cap the coordinator sets to
    /// the WHOLE cache's cap: the root can be over its cap by everything the
    /// KV payloads hold and this pass evicts nothing. It only acts once the
    /// companions by themselves exceed the whole cap, and then only on
    /// companions.
    @Test func standaloneCompanionEvictionComparesCompanionBytesAloneToTheFullCap() throws {
        try MLXMetalTestLock.withLock {
            let root = Self.makeRoot("companion-cap")
            defer { try? FileManager.default.removeItem(at: root) }
            let cap: Int64 = 100_003
            let coordinator = Self.coordinator(root: root, modelKey: "companion-cap", capBytes: cap)
            coordinator.setHybrid(true, requiresRecurrentSSMCompanion: true)
            let disk = try #require(coordinator.diskCache)
            let companions = try #require(coordinator.ssmStateCache.diskStore)
            try #require(disk.indexHasV2Columns)

            // KV payloads for 80 % of the cap, written with the pass deferred.
            var boundaries: [[Int]] = []
            for index in 0 ..< 4 {
                let tokens = Self.tokens(11 + index, seed: index + 1)
                boundaries.append(tokens)
                disk.store(tokens: tokens, arrays: Self.payload(5_003), enforceQuota: false)
            }
            let kvBytes = disk.usageBytes()
            try #require(kvBytes > cap * 3 / 4 && kvBytes < cap)

            // Direct companion writes, each followed by the store's own pass.
            let state = [MLXArray.ones([3_001], dtype: .float32)]
            for tokens in boundaries {
                try companions.store(ssmStates: state, tokens: tokens, boundary: tokens.count)
            }
            let companionBytes = disk.companionUsageBytes()
            try #require(companionBytes > 0 && companionBytes < cap)
            #expect(companions.quotaEntries().count == 4)
            #expect(disk.quotaEntries().count == 4)
            // Over the whole cap, and the pass that just ran four times left
            // it there.
            #expect(disk.usageBytes() == kvBytes + companionBytes)
            #expect(disk.usageBytes() > cap)

            // Companions alone past the whole cap: now it acts, on companions.
            var written = 4
            while disk.companionUsageBytes() <= cap, written < 40 {
                let tokens = Self.tokens(37 + written, seed: 50 + written)
                try companions.store(ssmStates: state, tokens: tokens, boundary: tokens.count)
                written += 1
                if companions.quotaEntries().count < written { break }
            }
            try #require(written < 40, "the companion store's own pass never ran")
            #expect(companions.quotaEntries().count < written)
            #expect(disk.companionUsageBytes() <= cap)
            #expect(disk.quotaEntries().count == 4, "a companion pass evicted a KV payload")
            #expect(disk.usageBytes() > cap)
            #expect(try #require(coordinator.snapshotStats().diskStats).evictions == 0)
        }
    }
}
