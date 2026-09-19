import Foundation
import MLX
@testable import MLXLMCommon
import SQLite3
import Testing

/// Recurrent companion payloads are files outside `cache_index.db`. With a v2
/// index their bytes are accounted in it, so the combined quota and the stats
/// poll are SQL aggregates instead of a walk of `ssm_companion/`.
///
/// The invariant every storing test asserts directly, not through an output,
/// has two halves:
///
/// - accuracy: `usageBytes()` equals the bytes on disk of every payload the
///   index names plus the companion files it names;
/// - completeness: the index names every published payload and every
///   published companion file that is on disk. Accuracy alone cannot see the
///   serious direction — a file the index never heard of is simply not summed
///   on either side — so completeness is checked by walking the directories.
///
/// Token counts are deliberately not multiples of 64 or 256.
@Suite(.serialized)
struct DiskCacheCompanionAccountingTests {

    // MARK: - Fixtures

    private static func makeRoot(_ label: String) -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("vmlx-companion-accounting-\(label)-\(UUID().uuidString)")
    }

    private static func tokens(_ count: Int, seed: Int) -> [Int] {
        (0..<count).map { seed * 100_000 + $0 }
    }

    private static func kv(_ elements: Int = 1_024) -> [String: MLXArray] {
        ["data": MLXArray.ones([elements], dtype: .float32)]
    }

    private static func recurrent(_ elements: Int = 1_024, states: Int = 1) -> [MLXArray] {
        (0..<states).map { _ in MLXArray.ones([elements], dtype: .float32) }
    }

    /// A clock a test moves by hand, so "a minute later" costs nothing.
    private final class TestClock: @unchecked Sendable {
        private let lock = NSLock()
        private var value = Date()

        var now: Date {
            lock.lock()
            defer { lock.unlock() }
            return value
        }

        func advance(_ seconds: TimeInterval) {
            lock.lock()
            value += seconds
            lock.unlock()
        }
    }

    private static func coordinator(
        root: URL, capBytes: Int64 = 1 << 30, modelKey: String, hybrid: Bool = true,
        busyTimeoutMs: Int32 = DiskCache.defaultIndexBusyTimeoutMs,
        clock: TestClock? = nil
    ) -> CacheCoordinator {
        let config = CacheCoordinatorConfig(
            usePagedCache: false,
            enableDiskCache: true,
            diskCacheMaxGB: Float(capBytes) / 1_073_741_824,
            diskCacheDir: root,
            modelKey: modelKey)
        let coordinator = clock.map { clock in
            CacheCoordinator(
                config: config, diskIndexBusyTimeoutMs: busyTimeoutMs,
                importRetryInterval: 60, now: { clock.now })
        } ?? CacheCoordinator(config: config, diskIndexBusyTimeoutMs: busyTimeoutMs)
        if hybrid {
            coordinator.setHybrid(true, requiresRecurrentSSMCompanion: true)
        }
        return coordinator
    }

    // The file layout, the raw index connection and the accounting invariant
    // live in `DiskCacheAccountingTestSupport`, shared with the quota-planner
    // wiring suite. These forwards keep every call site below as it was.
    private typealias Support = DiskCacheAccountingTestSupport
    private typealias RawDB = Support.RawDB
    private typealias IndexedRow = Support.IndexedRow

    private static func companionDir(_ root: URL) -> URL { Support.companionDir(root) }
    private static func payloadURL(_ root: URL, _ hash: String) -> URL { Support.payloadURL(root, hash) }
    private static func companionURLs(_ root: URL, _ key: String) -> [URL] { Support.companionURLs(root, key) }
    private static func fileBytes(_ url: URL) -> Int64 { Support.fileBytes(url) }
    private static func companionBytes(_ root: URL, _ key: String) -> Int64 { Support.companionBytes(root, key) }
    private static func indexedRows(_ root: URL) throws -> [IndexedRow] { try Support.indexedRows(root) }
    private static func legacyRows(_ root: URL) throws -> [String: Int64] { try Support.legacyRows(root) }
    private static func onDiskBytesOfIndexedFiles(_ root: URL) throws -> Int64 {
        try Support.onDiskBytesOfIndexedFiles(root)
    }

    private static func expectUsageMatchesDisk(
        _ disk: DiskCache, root: URL, atLeast: Int64 = 1, checkCompleteness: Bool = true,
        sourceLocation: SourceLocation = #_sourceLocation
    ) throws {
        try Support.expectUsageMatchesDisk(
            disk, root: root, atLeast: atLeast, checkCompleteness: checkCompleteness,
            sourceLocation: sourceLocation)
    }

    private static func expectIndexNamesEveryPublishedFile(
        _ root: URL, sourceLocation: SourceLocation = #_sourceLocation
    ) throws {
        try Support.expectIndexNamesEveryPublishedFile(root, sourceLocation: sourceLocation)
    }

    private static func requireUnlistable(
        _ dir: URL, sourceLocation: SourceLocation = #_sourceLocation
    ) throws {
        try Support.requireUnlistable(dir, sourceLocation: sourceLocation)
    }

    private struct InjectedFault: Error {}

    private static func kvHash(_ tokens: [Int], _ modelKey: String) -> String {
        DiskCache.hashTokens(tokens, modelKey: modelKey)
    }

    private static func ssmKey(_ tokens: [Int], _ modelKey: String) -> String {
        SSMCompanionDiskStore.keyFor(tokens: tokens, boundary: tokens.count, modelKey: modelKey)
    }

    private static func makeV1OnlyIndex(in root: URL) throws { try Support.makeV1OnlyIndex(in: root) }

    // MARK: - 1

    @Test func companionBytesAreCountedFromTheIndex() throws {
        try MLXMetalTestLock.withLock {
            let root = Self.makeRoot("counted")
            defer { try? FileManager.default.removeItem(at: root) }
            let modelKey = "accounting-counted"
            let coordinator = Self.coordinator(root: root, modelKey: modelKey)
            let disk = try #require(coordinator.diskCache)
            try #require(disk.indexHasV2Columns)

            let boundaries = [Self.tokens(301, seed: 1), Self.tokens(517, seed: 2), Self.tokens(1_003, seed: 3)]
            for tokens in boundaries {
                coordinator.storePersistentBoundary(
                    tokens: tokens, diskArrays: Self.kv(), ssmStates: Self.recurrent())
            }

            let rows = try Self.indexedRows(root)
            #expect(rows.count == 3)
            var kvTotal: Int64 = 0
            var companionTotal: Int64 = 0
            for tokens in boundaries {
                let row = try #require(rows.first { $0.hash == Self.kvHash(tokens, modelKey) })
                let key = Self.ssmKey(tokens, modelKey)
                #expect(row.companionKey == key)
                #expect(row.companionBytes == Self.companionBytes(root, key))
                #expect(row.companionBytes > 0)
                kvTotal += row.fileSize
                companionTotal += Self.companionBytes(root, key)
            }
            #expect(try Self.legacyRows(root).isEmpty)
            #expect(disk.usageBytes() == kvTotal + companionTotal)
            try Self.expectUsageMatchesDisk(disk, root: root, atLeast: kvTotal + 1)

            // A file the index does not know about, named like an entry so a
            // directory walk WOULD count it.
            let before = disk.usageBytes()
            let statsBefore = try #require(coordinator.snapshotStats().diskStats)
            try Data(repeating: 0xAB, count: 70_001).write(
                to: Self.companionDir(root).appendingPathComponent("ssm-unindexedjunk.safetensors"))
            #expect(disk.usageBytes() == before)
            let statsAfter = try #require(coordinator.snapshotStats().diskStats)
            #expect(statsAfter.currentPayloadBytes == statsBefore.currentPayloadBytes)
            #expect(statsAfter.currentPayloadBytes == Int(before))
            #expect(statsAfter.currentEntryCount == 3)
            // Completeness is switched off here and only here: the junk file
            // above is on disk and unindexed on purpose, to prove the usage
            // comes from the index and not from a walk.
            try Self.expectUsageMatchesDisk(disk, root: root, checkCompleteness: false)
        }
    }

    // MARK: - 2

    @Test func restoringACompanionReplacesItsBytes() throws {
        try MLXMetalTestLock.withLock {
            let root = Self.makeRoot("replace")
            defer { try? FileManager.default.removeItem(at: root) }
            let modelKey = "accounting-replace"
            let coordinator = Self.coordinator(root: root, modelKey: modelKey)
            let disk = try #require(coordinator.diskCache)
            let tokens = Self.tokens(517, seed: 4)
            let hash = Self.kvHash(tokens, modelKey)
            let key = Self.ssmKey(tokens, modelKey)

            coordinator.storePersistentBoundary(
                tokens: tokens, diskArrays: Self.kv(), ssmStates: Self.recurrent(64, states: 1))
            let first = try #require(try Self.indexedRows(root).first { $0.hash == hash })
            #expect(first.companionBytes == Self.companionBytes(root, key))
            try Self.expectUsageMatchesDisk(disk, root: root)

            // A different state count is not the validated entry, so this is a
            // real rewrite with a different size, not the touch-only skip.
            coordinator.storePersistentBoundary(
                tokens: tokens, diskArrays: Self.kv(), ssmStates: Self.recurrent(4_099, states: 2))
            let second = try #require(try Self.indexedRows(root).first { $0.hash == hash })
            let onDisk = Self.companionBytes(root, key)
            #expect(onDisk > first.companionBytes + 30_000)
            #expect(second.companionKey == key)
            #expect(second.companionBytes == onDisk, "bytes must be replaced, not added")
            #expect(try Self.indexedRows(root).count == 1)
            try Self.expectUsageMatchesDisk(disk, root: root)

            // The touch-only skip reports the same bytes again: still replaced.
            coordinator.storePersistentBoundary(
                tokens: tokens, diskArrays: Self.kv(), ssmStates: Self.recurrent(4_099, states: 2))
            #expect(try Self.indexedRows(root).first?.companionBytes == onDisk)
            try Self.expectUsageMatchesDisk(disk, root: root)
        }
    }

    // MARK: - 3

    @Test func reinsertingAKVRowKeepsItsCompanionLink() throws {
        try MLXMetalTestLock.withLock {
            let root = Self.makeRoot("reinsert")
            defer { try? FileManager.default.removeItem(at: root) }
            let modelKey = "accounting-reinsert"
            let coordinator = Self.coordinator(root: root, modelKey: modelKey)
            let disk = try #require(coordinator.diskCache)
            let tokens = Self.tokens(1_003, seed: 5)
            let hash = Self.kvHash(tokens, modelKey)

            coordinator.storePersistentBoundary(
                tokens: tokens, diskArrays: Self.kv(1_024), ssmStates: Self.recurrent())
            let before = try #require(try Self.indexedRows(root).first { $0.hash == hash })
            #expect(before.companionKey == Self.ssmKey(tokens, modelKey))
            #expect(before.companionBytes > 0)

            // A different layout forces the full write path and a second
            // INSERT for the same hash. The companion files are untouched.
            disk.store(tokens: tokens, arrays: Self.kv(3_001), enforceQuota: false)
            let after = try #require(try Self.indexedRows(root).first { $0.hash == hash })
            #expect(after.fileSize != before.fileSize)
            #expect(after.fileSize == Self.fileBytes(Self.payloadURL(root, hash)))
            #expect(after.companionKey == before.companionKey)
            #expect(after.companionBytes == before.companionBytes)
            try Self.expectUsageMatchesDisk(disk, root: root)

            let modelKeys = try RawDB(root: root)
                .rows("SELECT model_key FROM cache_entries").map { $0[0] }
            #expect(modelKeys == [modelKey])
        }
    }

    // MARK: - 4

    @Test func companionOnlyStoreLinksToExistingRowElseLegacy() throws {
        try MLXMetalTestLock.withLock {
            let root = Self.makeRoot("companion-only")
            defer { try? FileManager.default.removeItem(at: root) }
            let modelKey = "accounting-companion-only"
            let coordinator = Self.coordinator(root: root, modelKey: modelKey)
            let disk = try #require(coordinator.diskCache)
            let withRow = Self.tokens(301, seed: 6)
            let withoutRow = Self.tokens(1_291, seed: 7)

            // Row first, companion later: what `resolveSSMStates` does when it
            // rehydrates folded recurrent state after a disk hit.
            disk.store(tokens: withRow, arrays: Self.kv(), enforceQuota: false)
            #expect(try Self.indexedRows(root).first?.companionKey == nil)
            coordinator.storePersistentBoundary(
                tokens: withRow, diskArrays: nil, ssmStates: Self.recurrent())
            let linked = try #require(try Self.indexedRows(root).first)
            #expect(linked.companionKey == Self.ssmKey(withRow, modelKey))
            #expect(linked.companionBytes == Self.companionBytes(root, Self.ssmKey(withRow, modelKey)))
            #expect(try Self.legacyRows(root).isEmpty)
            try Self.expectUsageMatchesDisk(disk, root: root)

            // No row to hang from: counted as unlinked.
            coordinator.storePersistentBoundary(
                tokens: withoutRow, diskArrays: nil, ssmStates: Self.recurrent())
            let orphanKey = Self.ssmKey(withoutRow, modelKey)
            #expect(try Self.indexedRows(root).count == 1)
            #expect(try Self.legacyRows(root) == [orphanKey: Self.companionBytes(root, orphanKey)])
            #expect(Self.companionBytes(root, orphanKey) > 0)
            try Self.expectUsageMatchesDisk(disk, root: root)
            let stats = try #require(coordinator.snapshotStats().diskStats)
            #expect(stats.currentEntryCount == 2)
            #expect(stats.currentPayloadBytes == Int(disk.usageBytes()))

            // Its KV row arrives with the next full store: it becomes linked
            // and is no longer counted twice.
            coordinator.storePersistentBoundary(
                tokens: withoutRow, diskArrays: Self.kv(), ssmStates: Self.recurrent())
            #expect(try Self.legacyRows(root).isEmpty)
            #expect(try Self.indexedRows(root).allSatisfy { $0.companionKey != nil })
            try Self.expectUsageMatchesDisk(disk, root: root)
        }
    }

    // MARK: - 5

    @Test func importRunsOncePerRootAndIsIdempotent() throws {
        try MLXMetalTestLock.withLock {
            let root = Self.makeRoot("import")
            defer { try? FileManager.default.removeItem(at: root) }
            let modelKey = "accounting-import"
            let boundaries = [Self.tokens(301, seed: 8), Self.tokens(517, seed: 9), Self.tokens(1_291, seed: 10)]

            let populated: [IndexedRow]
            do {
                let writer = Self.coordinator(root: root, modelKey: modelKey)
                for tokens in boundaries {
                    writer.storePersistentBoundary(
                        tokens: tokens, diskArrays: Self.kv(), ssmStates: Self.recurrent())
                }
                populated = try Self.indexedRows(root)
                try #require(populated.count == 3)
                try #require(populated.allSatisfy { $0.companionKey != nil && $0.companionBytes > 0 })
            }

            // An older build's three-column INSERT OR REPLACE leaves exactly
            // this behind: rows present, companion columns at their defaults.
            func wipeCompanionColumns() throws {
                try RawDB(root: root).require(
                    "UPDATE cache_entries SET companion_key = NULL, companion_bytes = 0")
                try #require(try Self.indexedRows(root).allSatisfy {
                    $0.companionKey == nil && $0.companionBytes == 0
                })
            }
            try wipeCompanionColumns()

            // A new process opening the same directory.
            CacheCoordinator.resetImportedRootsForTesting()
            let reopened = Self.coordinator(root: root, modelKey: modelKey)
            let disk = try #require(reopened.diskCache)
            #expect(try Self.indexedRows(root) == populated)
            try Self.expectUsageMatchesDisk(disk, root: root)

            // Idempotent: the same walk again changes nothing.
            let companions = try #require(reopened.ssmStateCache.diskStore).quotaEntries()
            try #require(companions.count == 3)
            let again = disk.reconcileCompanionAccounting(companions: companions)
            #expect(again == DiskCacheCompanionImportSummary())
            CacheCoordinator.resetImportedRootsForTesting()
            _ = Self.coordinator(root: root, modelKey: modelKey)
            #expect(try Self.indexedRows(root) == populated)
            #expect(try Self.legacyRows(root).isEmpty)

            // Once per root: without a new process, another coordinator on the
            // same root does not walk the directory again.
            try wipeCompanionColumns()
            _ = Self.coordinator(root: root, modelKey: modelKey)
            #expect(try Self.indexedRows(root).allSatisfy { $0.companionKey == nil })

            CacheCoordinator.resetImportedRootsForTesting()
            _ = Self.coordinator(root: root, modelKey: modelKey)
            #expect(try Self.indexedRows(root) == populated)
        }
    }

    // MARK: - 6

    @Test func orphanRowIsReconciledAtOpen() throws {
        try MLXMetalTestLock.withLock {
            let root = Self.makeRoot("orphan-row")
            defer { try? FileManager.default.removeItem(at: root) }
            let modelKey = "accounting-orphan-row"
            let kept = Self.tokens(517, seed: 11)
            let lost = Self.tokens(1_003, seed: 12)
            do {
                let writer = Self.coordinator(root: root, modelKey: modelKey)
                for tokens in [kept, lost] {
                    writer.storePersistentBoundary(
                        tokens: tokens, diskArrays: Self.kv(), ssmStates: Self.recurrent())
                }
                try #require(try Self.indexedRows(root).count == 2)
            }
            try FileManager.default.removeItem(at: Self.payloadURL(root, Self.kvHash(lost, modelKey)))

            CacheCoordinator.resetImportedRootsForTesting()
            let reopened = Self.coordinator(root: root, modelKey: modelKey)
            let disk = try #require(reopened.diskCache)

            #expect(try Self.indexedRows(root).map(\.hash) == [Self.kvHash(kept, modelKey)])
            // Its companion files are still on disk, so they are still counted.
            let lostKey = Self.ssmKey(lost, modelKey)
            #expect(try Self.legacyRows(root) == [lostKey: Self.companionBytes(root, lostKey)])
            try Self.expectUsageMatchesDisk(disk, root: root)
        }
    }

    @Test func missingCompanionFilesClearTheLink() throws {
        try MLXMetalTestLock.withLock {
            let root = Self.makeRoot("missing-companion")
            defer { try? FileManager.default.removeItem(at: root) }
            let modelKey = "accounting-missing-companion"
            let intact = Self.tokens(301, seed: 13)
            let stripped = Self.tokens(1_291, seed: 14)
            let unlinked = Self.tokens(517, seed: 15)
            do {
                let writer = Self.coordinator(root: root, modelKey: modelKey)
                for tokens in [intact, stripped] {
                    writer.storePersistentBoundary(
                        tokens: tokens, diskArrays: Self.kv(), ssmStates: Self.recurrent())
                }
                writer.storePersistentBoundary(
                    tokens: unlinked, diskArrays: nil, ssmStates: Self.recurrent())
                try #require(try Self.legacyRows(root).count == 1)
            }
            for key in [Self.ssmKey(stripped, modelKey), Self.ssmKey(unlinked, modelKey)] {
                for url in Self.companionURLs(root, key) {
                    try FileManager.default.removeItem(at: url)
                }
            }

            CacheCoordinator.resetImportedRootsForTesting()
            let reopened = Self.coordinator(root: root, modelKey: modelKey)
            let disk = try #require(reopened.diskCache)

            let rows = try Self.indexedRows(root)
            let strippedRow = try #require(rows.first { $0.hash == Self.kvHash(stripped, modelKey) })
            #expect(strippedRow.companionKey == nil)
            #expect(strippedRow.companionBytes == 0)
            let intactRow = try #require(rows.first { $0.hash == Self.kvHash(intact, modelKey) })
            #expect(intactRow.companionKey == Self.ssmKey(intact, modelKey))
            #expect(try Self.legacyRows(root).isEmpty)
            try Self.expectUsageMatchesDisk(disk, root: root)
        }
    }

    // MARK: - 7

    @Test func missingPayloadOnFetchDeletesItsRow() throws {
        try MLXMetalTestLock.withLock {
            let root = Self.makeRoot("missing-payload")
            defer { try? FileManager.default.removeItem(at: root) }
            let modelKey = "accounting-missing-payload"
            let coordinator = Self.coordinator(root: root, modelKey: modelKey)
            let disk = try #require(coordinator.diskCache)
            let dense = Self.tokens(301, seed: 16)
            let hybrid = Self.tokens(1_003, seed: 17)
            let kept = Self.tokens(517, seed: 18)

            disk.store(tokens: dense, arrays: Self.kv(2_003), enforceQuota: false)
            for tokens in [hybrid, kept] {
                coordinator.storePersistentBoundary(
                    tokens: tokens, diskArrays: Self.kv(), ssmStates: Self.recurrent())
            }
            try Self.expectUsageMatchesDisk(disk, root: root)

            func fileSize(_ tokens: [Int]) throws -> Int64 {
                try #require(try Self.indexedRows(root).first {
                    $0.hash == Self.kvHash(tokens, modelKey)
                }).fileSize
            }

            // A row with no companion: usage drops by exactly its bytes.
            let denseBytes = try fileSize(dense)
            var before = disk.usageBytes()
            try FileManager.default.removeItem(at: Self.payloadURL(root, Self.kvHash(dense, modelKey)))
            #expect(disk.fetch(tokens: dense) == nil)
            #expect(try Self.indexedRows(root).count == 2)
            #expect(disk.usageBytes() == before - denseBytes)
            try Self.expectUsageMatchesDisk(disk, root: root)

            // A row with a companion: the row's bytes go, the companion files
            // are still on disk and stay counted, now unlinked.
            let hybridBytes = try fileSize(hybrid)
            before = disk.usageBytes()
            try FileManager.default.removeItem(at: Self.payloadURL(root, Self.kvHash(hybrid, modelKey)))
            #expect(disk.fetch(tokens: hybrid) == nil)
            #expect(try Self.indexedRows(root).map(\.hash) == [Self.kvHash(kept, modelKey)])
            #expect(disk.usageBytes() == before - hybridBytes)
            let hybridKey = Self.ssmKey(hybrid, modelKey)
            #expect(try Self.legacyRows(root) == [hybridKey: Self.companionBytes(root, hybridKey)])
            try Self.expectUsageMatchesDisk(disk, root: root)

            // A miss for a prefix that never had a row writes nothing.
            #expect(disk.fetch(tokens: Self.tokens(1_291, seed: 19)) == nil)
            try Self.expectUsageMatchesDisk(disk, root: root)
        }
    }

    // MARK: - 8

    @Test func tornCompanionWriteLeavesNoFinalNamedFile() throws {
        try MLXMetalTestLock.withLock {
            let root = Self.makeRoot("torn")
            defer { try? FileManager.default.removeItem(at: root) }
            let modelKey = "accounting-torn"
            let tokens = Self.tokens(517, seed: 20)
            let key = Self.ssmKey(tokens, modelKey)
            let dir = Self.companionDir(root)

            let partial = dir.appendingPathComponent("ssm-\(key).partial-1a2b3c4d.safetensors")
            let usageBefore: Int64
            do {
                let coordinator = Self.coordinator(root: root, modelKey: modelKey)
                let disk = try #require(coordinator.diskCache)
                let companion = try #require(coordinator.ssmStateCache.diskStore)
                coordinator.storePersistentBoundary(
                    tokens: tokens, diskArrays: Self.kv(), ssmStates: Self.recurrent())

                // A completed store publishes by rename: nothing unpublished
                // is left, and the final-named file holds every declared byte.
                let names = try FileManager.default.contentsOfDirectory(atPath: dir.path)
                #expect(names.sorted() == ["ssm-\(key).json", "ssm-\(key).safetensors"])
                #expect(DiskCache.isCompleteSafetensors(url: Self.companionURLs(root, key)[0]))
                let entriesBefore = companion.quotaEntries()
                try #require(entriesBefore.count == 1)
                usageBefore = disk.usageBytes()

                // What a write that died before its rename leaves behind.
                try Data(repeating: 0xEE, count: 40_003).write(to: partial)
                let entriesAfter = companion.quotaEntries()
                #expect(entriesAfter.map(\.hash) == [key])
                #expect(entriesAfter.first?.bytes == entriesBefore.first?.bytes)
                #expect(disk.usageBytes() == usageBefore)
                try Self.expectUsageMatchesDisk(disk, root: root)
            }

            // The next open sweeps it; the published entry is untouched.
            CacheCoordinator.resetImportedRootsForTesting()
            let reopened = Self.coordinator(root: root, modelKey: modelKey)
            #expect(!FileManager.default.fileExists(atPath: partial.path))
            let names = try FileManager.default.contentsOfDirectory(atPath: dir.path)
            #expect(names.sorted() == ["ssm-\(key).json", "ssm-\(key).safetensors"])
            let disk = try #require(reopened.diskCache)
            #expect(disk.usageBytes() == usageBefore)
            try Self.expectUsageMatchesDisk(disk, root: root)
        }
    }

    // MARK: - 9

    @Test func snapshotStatsDoesNoDirectoryIO() throws {
        try MLXMetalTestLock.withLock {
            let root = Self.makeRoot("stats-no-io")
            let dir = Self.companionDir(root)
            defer {
                try? FileManager.default.setAttributes(
                    [.posixPermissions: 0o755], ofItemAtPath: dir.path)
                try? FileManager.default.removeItem(at: root)
            }
            let modelKey = "accounting-stats-no-io"
            let entries = 1_000
            let coordinator = Self.coordinator(root: root, capBytes: 64 << 30, modelKey: modelKey)
            let disk = try #require(coordinator.diskCache)
            let companion = try #require(coordinator.ssmStateCache.diskStore)

            // The two stores' own write paths, as `storePersistentBoundary`
            // drives them, without a quota pass per entry.
            let kv = Self.kv(16)
            let recurrent = Self.recurrent(16)
            for index in 0..<entries {
                let tokens = [2_000_000 + index, 1, 2, 3, 5]
                disk.store(tokens: tokens, arrays: kv, enforceQuota: false)
                try companion.store(
                    ssmStates: recurrent, tokens: tokens, boundary: tokens.count,
                    enforceQuota: false)
            }
            let expectedBytes = try Self.onDiskBytesOfIndexedFiles(root)
            let rows = try Self.indexedRows(root)
            try #require(rows.count == entries)
            try #require(rows.allSatisfy { $0.companionKey != nil && $0.companionBytes > 0 })
            let kvOnly = rows.reduce(Int64(0)) { $0 + $1.fileSize }
            try #require(expectedBytes > kvOnly)
            #expect(disk.usageBytes() == expectedBytes)

            // Unreadable: any listing or stat of the companions now fails.
            try FileManager.default.setAttributes(
                [.posixPermissions: 0o000], ofItemAtPath: dir.path)
            try Self.requireUnlistable(dir)

            var samples: [UInt64] = []
            for _ in 0..<100 {
                let start = DispatchTime.now().uptimeNanoseconds
                let stats = coordinator.snapshotStats().diskStats
                samples.append(DispatchTime.now().uptimeNanoseconds - start)
                #expect(stats?.currentPayloadBytes == Int(expectedBytes))
                #expect(stats?.currentEntryCount == entries)
            }
            samples.sort()
            let medianMs = Double(samples[samples.count / 2]) / 1_000_000
            let maxMs = Double(samples[samples.count - 1]) / 1_000_000
            #if DEBUG
                let build = "debug"
            #else
                let build = "release"
            #endif
            print(
                "COMPANION_ACCOUNTING snapshotStats rows=\(entries) build=\(build) "
                    + "median_ms=\(String(format: "%.3f", medianMs)) "
                    + "max_ms=\(String(format: "%.3f", maxMs)) samples=\(samples.count)")
            #expect(medianMs < 1.0)

            try FileManager.default.setAttributes(
                [.posixPermissions: 0o755], ofItemAtPath: dir.path)
            try Self.expectUsageMatchesDisk(disk, root: root, atLeast: expectedBytes)
        }
    }

    // MARK: - 10

    @Test func storeBelowCapDoesNoCompanionDirectoryIO() throws {
        try MLXMetalTestLock.withLock {
            let root = Self.makeRoot("store-no-io")
            let dir = Self.companionDir(root)
            defer {
                try? FileManager.default.setAttributes(
                    [.posixPermissions: 0o755], ofItemAtPath: dir.path)
                try? FileManager.default.removeItem(at: root)
            }
            let modelKey = "accounting-store-no-io"
            let coordinator = Self.coordinator(root: root, modelKey: modelKey)
            let disk = try #require(coordinator.diskCache)
            for seed in [21, 22, 23] {
                coordinator.storePersistentBoundary(
                    tokens: Self.tokens(301 + seed, seed: seed),
                    diskArrays: Self.kv(), ssmStates: Self.recurrent())
            }
            let before = disk.usageBytes()
            try Self.expectUsageMatchesDisk(disk, root: root)
            let companionTotal = try Self.indexedRows(root).reduce(Int64(0)) { $0 + $1.companionBytes }
            try #require(companionTotal > 0)

            try FileManager.default.setAttributes(
                [.posixPermissions: 0o000], ofItemAtPath: dir.path)
            try Self.requireUnlistable(dir)

            // A dense-shaped store: a KV payload, no recurrent state.
            let dense = Self.tokens(1_291, seed: 24)
            coordinator.storePersistentBoundary(
                tokens: dense, diskArrays: Self.kv(2_003), ssmStates: nil)

            let denseHash = Self.kvHash(dense, modelKey)
            let denseBytes = Self.fileBytes(Self.payloadURL(root, denseHash))
            #expect(denseBytes > 0)
            #expect(try Self.indexedRows(root).count == 4)
            #expect(disk.usageBytes() == before + denseBytes)
            let stats = try #require(coordinator.snapshotStats().diskStats)
            #expect(stats.currentPayloadBytes == Int(before + denseBytes))
            #expect(stats.currentEntryCount == 4)
            #expect(stats.evictions == 0)
            #expect(disk.fetch(tokens: dense) != nil)

            try FileManager.default.setAttributes(
                [.posixPermissions: 0o755], ofItemAtPath: dir.path)
            try Self.expectUsageMatchesDisk(disk, root: root)
        }
    }

    // MARK: - 11

    /// The no-regression control for "policy unchanged": one fixture, built
    /// file for file the same way, evicted once by the index-sourced pass and
    /// once by the directory-walk pass. Both must equal the set the documented
    /// order produces: every group that can never fit, then unlinked legacy
    /// companions, then oldest recency, stopping as soon as the total fits.
    @Test func quotaOrderIsUnchanged() throws {
        try MLXMetalTestLock.withLock {
            let modelKey = "accounting-order"
            let g1 = Self.tokens(301, seed: 31)
            let g2 = Self.tokens(517, seed: 32)
            let g3 = Self.tokens(1_003, seed: 33)
            let g4 = Self.tokens(307, seed: 34)
            let oversized = Self.tokens(1_291, seed: 35)
            let legacy = Self.tokens(311, seed: 36)

            // Recency is deliberately not insertion order, and the two groups
            // that go first regardless of recency are the two NEWEST.
            let recency: [(tokens: [Int], at: TimeInterval)] = [
                (g1, 40_000), (g2, 10_000), (g3, 20_000), (g4, 30_000),
                (oversized, 60_000), (legacy, 50_000),
            ]

            struct Outcome: Equatable {
                var survivingKV: Set<String>
                var survivingCompanions: Set<String>
                var evictions: Int
            }

            func run(v2: Bool) throws -> Outcome {
                let root = Self.makeRoot(v2 ? "order-v2" : "order-v1")
                defer { try? FileManager.default.removeItem(at: root) }
                if !v2 { try Self.makeV1OnlyIndex(in: root) }

                // Built with the standalone stores so both runs get the same
                // files; the coordinator only ever sees a finished directory.
                var groupBytes: [Int64] = []
                do {
                    let disk = DiskCache(cacheDir: root, maxSizeBytes: 1 << 30, modelKey: modelKey)
                    try #require(disk.indexHasV2Columns == v2)
                    let companion = try SSMCompanionDiskStore(
                        cacheDir: Self.companionDir(root), modelKey: modelKey, maxBytes: 0)
                    for tokens in [g1, g2, g3, g4] {
                        disk.store(tokens: tokens, arrays: Self.kv(), enforceQuota: false)
                        try companion.store(
                            ssmStates: Self.recurrent(), tokens: tokens, boundary: tokens.count,
                            enforceQuota: false)
                    }
                    disk.store(tokens: oversized, arrays: Self.kv(65_537), enforceQuota: false)
                    try companion.store(
                        ssmStates: Self.recurrent(), tokens: oversized, boundary: oversized.count,
                        enforceQuota: false)
                    // A companion from before sidecars carried `kv_hash`, with
                    // no KV payload of its own.
                    try companion.store(
                        ssmStates: Self.recurrent(), tokens: legacy, boundary: legacy.count,
                        enforceQuota: false)
                    let sidecarURL = Self.companionURLs(root, Self.ssmKey(legacy, modelKey))[1]
                    var sidecar = try #require(
                        JSONSerialization.jsonObject(with: Data(contentsOf: sidecarURL)) as? [String: Any])
                    sidecar.removeValue(forKey: "kv_hash")
                    sidecar.removeValue(forKey: "boundary")
                    try JSONSerialization.data(withJSONObject: sidecar, options: [.sortedKeys])
                        .write(to: sidecarURL, options: [.atomic])

                    for (tokens, at) in recency {
                        let date = Date(timeIntervalSince1970: at)
                        if tokens != legacy {
                            try #require(disk.touchRecency(tokens: tokens, at: date))
                        }
                        for url in Self.companionURLs(root, Self.ssmKey(tokens, modelKey)) {
                            try FileManager.default.setAttributes(
                                [.modificationDate: date], ofItemAtPath: url.path)
                        }
                    }
                    for tokens in [g1, g2, g3, g4] {
                        groupBytes.append(
                            Self.fileBytes(Self.payloadURL(root, Self.kvHash(tokens, modelKey)))
                                + Self.companionBytes(root, Self.ssmKey(tokens, modelKey)))
                    }
                }
                let oversizedBytes =
                    Self.fileBytes(Self.payloadURL(root, Self.kvHash(oversized, modelKey)))
                    + Self.companionBytes(root, Self.ssmKey(oversized, modelKey))

                // Room for the two newest ordinary groups and half of another.
                let cap = groupBytes[0] + groupBytes[3] + groupBytes.min()! / 2
                try #require(oversizedBytes > cap)
                try #require(groupBytes[0] + groupBytes[2] + groupBytes[3] > cap)
                try #require(cap < 1 << 24, "cap must survive the Float GiB round trip exactly")

                CacheCoordinator.resetImportedRootsForTesting()
                let coordinator = Self.coordinator(root: root, capBytes: cap, modelKey: modelKey)
                let disk = try #require(coordinator.diskCache)
                try #require(disk.indexHasV2Columns == v2)
                try #require(Int64(disk.maxSizeBytes) == cap)

                let survivingKV = Set(
                    try RawDB(root: root).rows("SELECT hash FROM cache_entries").compactMap { $0[0] })
                for hash in survivingKV {
                    #expect(FileManager.default.fileExists(atPath: Self.payloadURL(root, hash).path))
                }
                let payloadsOnDisk = try FileManager.default.contentsOfDirectory(atPath: root.path)
                    .filter { $0.hasSuffix(".safetensors") }
                #expect(Set(payloadsOnDisk) == Set(survivingKV.map { "\($0).safetensors" }))
                let companionNames = try FileManager.default
                    .contentsOfDirectory(atPath: Self.companionDir(root).path)
                let survivingCompanions = Set(companionNames.compactMap { name -> String? in
                    guard name.hasPrefix("ssm-"), name.hasSuffix(".safetensors") else { return nil }
                    return String(name.dropFirst(4).dropLast(".safetensors".count))
                })
                #expect(companionNames.count == survivingCompanions.count * 2)
                // A v1 index names no companion, so neither half of the
                // invariant applies to that run; the file-for-file comparison
                // with the v2 run below is its check.
                if v2 { try Self.expectUsageMatchesDisk(disk, root: root) }
                return Outcome(
                    survivingKV: survivingKV,
                    survivingCompanions: survivingCompanions,
                    evictions: disk.snapshotStats().evictions)
            }

            // Evicted, in order: `oversized` (can never fit, newest of all),
            // `legacy` (unlinked, second newest), then g2 (t=10 000) and g3
            // (t=20 000). g4 + g1 fit, so the pass stops there.
            let expected = Outcome(
                survivingKV: [Self.kvHash(g1, modelKey), Self.kvHash(g4, modelKey)],
                survivingCompanions: [Self.ssmKey(g1, modelKey), Self.ssmKey(g4, modelKey)],
                evictions: 4)

            let indexed = try run(v2: true)
            let walked = try run(v2: false)
            #expect(walked == expected, "the baseline pass no longer matches the documented order")
            #expect(indexed == expected)
            #expect(indexed == walked)
        }
    }

    // MARK: - 12

    @Test func v1IndexFallsBackToTheDirectoryWalk() throws {
        try MLXMetalTestLock.withLock {
            let root = Self.makeRoot("v1-fallback")
            defer { try? FileManager.default.removeItem(at: root) }
            try Self.makeV1OnlyIndex(in: root)
            let modelKey = "accounting-v1-fallback"
            let boundaries = [Self.tokens(301, seed: 41), Self.tokens(517, seed: 42), Self.tokens(1_003, seed: 43)]

            func directoryBytes() throws -> Int64 {
                var total: Int64 = 0
                for name in try FileManager.default.contentsOfDirectory(atPath: root.path)
                where name.hasSuffix(".safetensors") {
                    total += Self.fileBytes(root.appendingPathComponent(name))
                }
                for name in try FileManager.default
                    .contentsOfDirectory(atPath: Self.companionDir(root).path)
                {
                    total += Self.fileBytes(Self.companionDir(root).appendingPathComponent(name))
                }
                return total
            }

            var groupBytes: [Int64] = []
            do {
                let coordinator = Self.coordinator(root: root, modelKey: modelKey)
                let disk = try #require(coordinator.diskCache)
                #expect(!disk.indexHasV2Columns)
                #expect(disk.indexSchemaVersion == 99)

                for (index, tokens) in boundaries.enumerated() {
                    coordinator.storePersistentBoundary(
                        tokens: tokens, diskArrays: Self.kv(), ssmStates: Self.recurrent())
                    #expect(disk.touchRecency(
                        tokens: tokens, at: Date(timeIntervalSince1970: 10_000 * Double(index + 1))))
                    #expect(try #require(coordinator.ssmStateCache.diskStore).touchRecency(
                        tokens: tokens, boundary: tokens.count,
                        at: Date(timeIntervalSince1970: 10_000 * Double(index + 1))))
                    groupBytes.append(
                        Self.fileBytes(Self.payloadURL(root, Self.kvHash(tokens, modelKey)))
                            + Self.companionBytes(root, Self.ssmKey(tokens, modelKey)))
                }

                // Stats still count companions, by walking the directory.
                let stats = try #require(coordinator.snapshotStats().diskStats)
                #expect(stats.currentPayloadBytes == Int(try directoryBytes()))
                #expect(stats.currentPayloadBytes == Int(groupBytes.reduce(0, +)))
                #expect(stats.currentEntryCount == 3)
                // The index was not touched beyond the three v1 columns.
                let columns = try RawDB(root: root)
                    .rows("SELECT name FROM pragma_table_info('cache_entries')").compactMap { $0[0] }
                #expect(columns == ["hash", "token_count", "file_size", "created_at"])
            }

            // Quota still evicts a linked group as a unit, oldest first.
            let cap = groupBytes[1] + groupBytes[2] + groupBytes[0] / 2
            let reopened = Self.coordinator(root: root, capBytes: cap, modelKey: modelKey)
            let stats = try #require(reopened.snapshotStats().diskStats)
            #expect(stats.evictions == 1)
            #expect(stats.currentEntryCount == 2)
            #expect(stats.currentPayloadBytes == Int(groupBytes[1] + groupBytes[2]))
            #expect(stats.currentPayloadBytes == Int(try directoryBytes()))
            #expect(!FileManager.default.fileExists(
                atPath: Self.payloadURL(root, Self.kvHash(boundaries[0], modelKey)).path))
            #expect(Self.companionBytes(root, Self.ssmKey(boundaries[0], modelKey)) == 0)
            #expect(try RawDB(root: root).rows("PRAGMA user_version").first?.first == "99")
            // No index invariant here: a v1 index has no companion columns to
            // be complete about. `directoryBytes()` above is the walk itself.
        }
    }

    // MARK: - Failure modes this change introduces

    /// The index is only right if EVERY removal reaches it. The companion
    /// store still evicts on its own when it is written to directly (not
    /// through `storePersistentBoundary`), and `clear()` empties it.
    @Test func companionStoreEvictionAndClearKeepTheIndexInStep() throws {
        try MLXMetalTestLock.withLock {
            let sizingRoot = Self.makeRoot("standalone-sizing")
            let root = Self.makeRoot("standalone-evict")
            defer {
                try? FileManager.default.removeItem(at: sizingRoot)
                try? FileManager.default.removeItem(at: root)
            }
            let modelKey = "accounting-standalone-evict"
            let boundaries = [Self.tokens(301, seed: 61), Self.tokens(517, seed: 62), Self.tokens(1_003, seed: 63)]

            let oneCompanion: Int64
            do {
                let sizing = Self.coordinator(root: sizingRoot, modelKey: modelKey)
                sizing.storePersistentBoundary(
                    tokens: boundaries[0], diskArrays: Self.kv(16), ssmStates: Self.recurrent())
                oneCompanion = Self.companionBytes(sizingRoot, Self.ssmKey(boundaries[0], modelKey))
                try #require(oneCompanion > 0)
            }

            // Room for two companions and a half; the KV payloads are tiny.
            let coordinator = Self.coordinator(
                root: root, capBytes: oneCompanion * 5 / 2, modelKey: modelKey)
            let disk = try #require(coordinator.diskCache)
            for (index, tokens) in boundaries.enumerated() {
                disk.store(tokens: tokens, arrays: Self.kv(16), enforceQuota: false)
                // Direct write-through, with the companion store's own quota.
                coordinator.ssmStateCache.store(
                    ssmStates: Self.recurrent(), tokens: tokens, boundary: tokens.count)
                for url in Self.companionURLs(root, Self.ssmKey(tokens, modelKey))
                where FileManager.default.fileExists(atPath: url.path) {
                    try FileManager.default.setAttributes(
                        [.modificationDate: Date(timeIntervalSince1970: 10_000 * Double(index + 1))],
                        ofItemAtPath: url.path)
                }
            }

            // The third write pushed the companions over the cap and the store
            // evicted the oldest by itself.
            #expect(Self.companionBytes(root, Self.ssmKey(boundaries[0], modelKey)) == 0)
            let rows = try Self.indexedRows(root)
            #expect(rows.count == 3)
            let first = try #require(rows.first { $0.hash == Self.kvHash(boundaries[0], modelKey) })
            #expect(first.companionKey == nil)
            #expect(first.companionBytes == 0)
            #expect(rows.filter { $0.companionKey != nil }.count == 2)
            try Self.expectUsageMatchesDisk(disk, root: root)

            coordinator.clear()
            #expect(try Self.indexedRows(root).isEmpty)
            #expect(try Self.legacyRows(root).isEmpty)
            #expect(disk.usageBytes() == 0)
            #expect(coordinator.snapshotStats().diskStats?.currentPayloadBytes == 0)
            let left = try FileManager.default.contentsOfDirectory(atPath: Self.companionDir(root).path)
            #expect(left.isEmpty)
            try Self.expectIndexNamesEveryPublishedFile(root)
        }
    }

    /// An older build sharing the directory writes rows with the three-column
    /// INSERT OR REPLACE. Such a row is a KV-only group: counted, evictable,
    /// never a crash, and the row it replaced loses its link until the next
    /// import finds the companion files again.
    @Test func rowsWrittenByAnOlderBuildAreKVOnlyGroups() throws {
        try MLXMetalTestLock.withLock {
            let root = Self.makeRoot("older-build")
            defer { try? FileManager.default.removeItem(at: root) }
            let modelKey = "accounting-older-build"
            let tokens = Self.tokens(1_291, seed: 64)
            let hash = Self.kvHash(tokens, modelKey)
            let key = Self.ssmKey(tokens, modelKey)

            let groupBytes: Int64
            do {
                let coordinator = Self.coordinator(root: root, modelKey: modelKey)
                coordinator.storePersistentBoundary(
                    tokens: tokens, diskArrays: Self.kv(), ssmStates: Self.recurrent())
                let disk = try #require(coordinator.diskCache)
                groupBytes = disk.usageBytes()

                let fileSize = Self.fileBytes(Self.payloadURL(root, hash))
                try RawDB(root: root).require(
                    "INSERT OR REPLACE INTO cache_entries (hash, token_count, file_size) "
                        + "VALUES ('\(hash)', \(tokens.count), \(fileSize))")
                let row = try #require(try Self.indexedRows(root).first)
                #expect(row.companionKey == nil)
                #expect(row.companionBytes == 0)
                // Under-counted, not wrong in a way that evicts anything.
                #expect(disk.usageBytes() == fileSize)
                coordinator.enforceCombinedDiskQuota()
                #expect(coordinator.snapshotStats().diskStats?.currentEntryCount == 1)
                #expect(coordinator.snapshotStats().diskStats?.evictions == 0)
            }

            CacheCoordinator.resetImportedRootsForTesting()
            let reopened = Self.coordinator(root: root, modelKey: modelKey)
            let disk = try #require(reopened.diskCache)
            let repaired = try #require(try Self.indexedRows(root).first)
            #expect(repaired.companionKey == key)
            #expect(disk.usageBytes() == groupBytes)
            try Self.expectUsageMatchesDisk(disk, root: root)
        }
    }

    // MARK: - 13

    @Test func lostInsertRemovesThePublishedPayload() throws {
        try MLXMetalTestLock.withLock {
            let root = Self.makeRoot("lost-insert")
            defer { try? FileManager.default.removeItem(at: root) }
            let modelKey = "accounting-lost-insert"
            // The production wait is 1 s; 50 ms keeps this test honest about
            // the mechanism without paying for it.
            let disk = DiskCache(
                cacheDir: root, maxSizeBytes: 1 << 30, modelKey: modelKey, indexBusyTimeoutMs: 50)
            try #require(disk.indexHasV2Columns)
            let tokens = Self.tokens(1_291, seed: 51)
            let payload = Self.payloadURL(root, Self.kvHash(tokens, modelKey))

            // Another model's connection holds the index write lock for longer
            // than this connection is willing to wait.
            let blocker = try RawDB(root: root)
            try blocker.require("BEGIN IMMEDIATE")

            let start = DispatchTime.now().uptimeNanoseconds
            disk.store(tokens: tokens, arrays: Self.kv(), enforceQuota: false)
            let waitedMs = Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000

            #expect(!FileManager.default.fileExists(atPath: payload.path))
            let names = try FileManager.default.contentsOfDirectory(atPath: root.path)
            #expect(names.filter { $0.hasSuffix(".safetensors") }.isEmpty)
            // Nothing was reused: it is a failed index write, not a skip.
            #expect(disk.snapshotStats().storeSkips == 0)
            #expect(disk.snapshotStats().failedIndexWrites == 1)
            #expect(disk.snapshotStats().currentEntryCount == 0)
            #expect(waitedMs >= 50, "the insert did not wait for the lock at all")
            #expect(waitedMs < 900, "the injected timeout was not honoured")
            #expect(disk.fetch(tokens: tokens) == nil)

            // Once the lock is released the same store lands normally.
            try blocker.require("COMMIT")
            disk.store(tokens: tokens, arrays: Self.kv(), enforceQuota: false)
            #expect(FileManager.default.fileExists(atPath: payload.path))
            #expect(try Self.indexedRows(root).map(\.hash) == [Self.kvHash(tokens, modelKey)])
            #expect(disk.snapshotStats().storeSkips == 0)
            #expect(disk.snapshotStats().failedIndexWrites == 1)
            try Self.expectUsageMatchesDisk(disk, root: root)
        }
    }

    // MARK: - Never under-count

    /// R1. An import that could not take the write lock must not be recorded
    /// as done: on an upgraded directory that would leave every companion
    /// uncounted until the next launch.
    @Test func skippedImportIsRetriedByTheNextCoordinator() throws {
        try MLXMetalTestLock.withLock {
            let root = Self.makeRoot("skipped-import")
            defer { try? FileManager.default.removeItem(at: root) }
            let modelKey = "accounting-skipped-import"
            let boundaries = [Self.tokens(301, seed: 71), Self.tokens(1_003, seed: 72)]

            let populated: [IndexedRow]
            do {
                let writer = Self.coordinator(root: root, modelKey: modelKey)
                for tokens in boundaries {
                    writer.storePersistentBoundary(
                        tokens: tokens, diskArrays: Self.kv(), ssmStates: Self.recurrent())
                }
                populated = try Self.indexedRows(root)
                try #require(populated.count == 2)
                try #require(populated.allSatisfy { $0.companionKey != nil && $0.companionBytes > 0 })
            }

            // An upgraded directory: the companions are on disk, none counted.
            try RawDB(root: root).require(
                "UPDATE cache_entries SET companion_key = NULL, companion_bytes = 0")
            CacheCoordinator.resetImportedRootsForTesting()

            // The purge tool (or another model's connection) holds the write
            // lock for longer than the first model load is willing to wait.
            let blocker = try RawDB(root: root)
            try blocker.require("BEGIN IMMEDIATE")
            let start = DispatchTime.now().uptimeNanoseconds
            let blocked = Self.coordinator(root: root, modelKey: modelKey, busyTimeoutMs: 50)
            let waitedMs = Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000
            try #require(blocked.diskCache?.indexHasV2Columns == true)
            // Fail closed: the import really was skipped.
            try #require(
                try Self.indexedRows(root).allSatisfy { $0.companionKey == nil },
                "INVALID: the import was not blocked")
            #expect(waitedMs < 900, "the injected timeout was not honoured")
            try blocker.require("COMMIT")

            // Same process, next model load on the same root: the import runs.
            let next = Self.coordinator(root: root, modelKey: modelKey)
            let disk = try #require(next.diskCache)
            #expect(try Self.indexedRows(root) == populated)
            try Self.expectUsageMatchesDisk(disk, root: root)

            // And having committed, it is not run a third time.
            try RawDB(root: root).require(
                "UPDATE cache_entries SET companion_key = NULL, companion_bytes = 0")
            _ = Self.coordinator(root: root, modelKey: modelKey)
            #expect(try Self.indexedRows(root).allSatisfy { $0.companionKey == nil })
            withExtendedLifetime(blocked) {}
        }
    }

    /// R2. The row is there for the SELECT and gone for the UPDATE: another
    /// connection (the purge tool, another model's quota pass) deleted it in
    /// between. The companion files are already on disk.
    @Test func companionLinkToAVanishedRowFallsBackToLegacy() throws {
        try MLXMetalTestLock.withLock {
            let root = Self.makeRoot("vanished-row")
            defer { try? FileManager.default.removeItem(at: root) }
            let modelKey = "accounting-vanished-row"
            let coordinator = Self.coordinator(root: root, modelKey: modelKey)
            let disk = try #require(coordinator.diskCache)
            let companion = try #require(coordinator.ssmStateCache.diskStore)
            let tokens = Self.tokens(1_003, seed: 73)
            let key = Self.ssmKey(tokens, modelKey)

            disk.store(tokens: tokens, arrays: Self.kv(), enforceQuota: false)
            try #require(try Self.indexedRows(root).count == 1)

            // Landing inside that window needs either a hook in the product
            // or this: a trigger that removes the row as the UPDATE reaches
            // it, so the UPDATE changes 0 rows. The other connection takes
            // the payload with it, as the purge tool does.
            let raw = try RawDB(root: root)
            try raw.require(
                """
                CREATE TRIGGER vanish_on_link BEFORE UPDATE OF companion_key ON cache_entries
                BEGIN DELETE FROM cache_entries WHERE hash = OLD.hash; END
                """)
            try FileManager.default.removeItem(
                at: Self.payloadURL(root, Self.kvHash(tokens, modelKey)))

            try companion.store(
                ssmStates: Self.recurrent(), tokens: tokens, boundary: tokens.count,
                enforceQuota: false)
            try raw.require("DROP TRIGGER vanish_on_link")

            try #require(
                try Self.indexedRows(root).isEmpty, "INVALID: the trigger did not remove the row")
            let onDisk = Self.companionBytes(root, key)
            try #require(onDisk > 0)
            #expect(try Self.legacyRows(root) == [key: onDisk])
            #expect(disk.snapshotStats().failedIndexWrites == 0)
            try Self.expectUsageMatchesDisk(disk, root: root)
        }
    }

    /// R2. The index cannot be written at all (another connection holds the
    /// write lock past the timeout). Files in neither table would never be
    /// evicted, so they are taken back: a miss is the safe outcome.
    @Test func companionIndexWriteFailureRemovesTheFiles() throws {
        try MLXMetalTestLock.withLock {
            let root = Self.makeRoot("record-failure")
            defer { try? FileManager.default.removeItem(at: root) }
            let modelKey = "accounting-record-failure"
            let coordinator = Self.coordinator(root: root, modelKey: modelKey, busyTimeoutMs: 50)
            let disk = try #require(coordinator.diskCache)
            let companion = try #require(coordinator.ssmStateCache.diskStore)
            let withRow = Self.tokens(517, seed: 74)  // fails in the link UPDATE
            let withoutRow = Self.tokens(1_291, seed: 75)  // fails in the legacy upsert
            disk.store(tokens: withRow, arrays: Self.kv(), enforceQuota: false)

            let blocker = try RawDB(root: root)
            try blocker.require("BEGIN IMMEDIATE")
            for tokens in [withRow, withoutRow] {
                let record = try companion.store(
                    ssmStates: Self.recurrent(), tokens: tokens, boundary: tokens.count,
                    enforceQuota: false)
                #expect(record == nil)
                for url in Self.companionURLs(root, Self.ssmKey(tokens, modelKey)) {
                    #expect(!FileManager.default.fileExists(atPath: url.path))
                }
                // Locals, so a failure does not print the token array.
                let stillValidated = companion.hasValidatedCompleteEntry(
                    tokens: tokens, boundary: tokens.count)
                #expect(!stillValidated)
                let restorable = companion.fetch(tokens: tokens, boundary: tokens.count) != nil
                #expect(!restorable)
            }
            #expect(disk.snapshotStats().failedIndexWrites == 2)
            #expect(coordinator.snapshotStats().diskStats?.failedIndexWrites == 2)
            #expect(disk.snapshotStats().storeSkips == 0)
            try blocker.require("COMMIT")
            try Self.expectUsageMatchesDisk(disk, root: root)

            // With the lock released the same writes land normally.
            for tokens in [withRow, withoutRow] {
                let record = try companion.store(
                    ssmStates: Self.recurrent(), tokens: tokens, boundary: tokens.count,
                    enforceQuota: false)
                #expect(record?.bytes == Self.companionBytes(root, Self.ssmKey(tokens, modelKey)))
            }
            #expect(try Self.indexedRows(root).first?.companionKey == Self.ssmKey(withRow, modelKey))
            #expect(try Self.legacyRows(root).keys.sorted() == [Self.ssmKey(withoutRow, modelKey)])
            #expect(disk.snapshotStats().failedIndexWrites == 2)
            try Self.expectUsageMatchesDisk(disk, root: root)
        }
    }

    /// R3, the under-count direction: a fresh key whose sidecar cannot be
    /// written. The tensor is already under its final name. No hook: a real
    /// directory sits where the sidecar goes, so the atomic write's rename
    /// fails.
    @Test func sidecarFailureStillCountsThePublishedTensor() throws {
        try MLXMetalTestLock.withLock {
            let root = Self.makeRoot("sidecar-failure")
            defer { try? FileManager.default.removeItem(at: root) }
            let modelKey = "accounting-sidecar-failure"
            let coordinator = Self.coordinator(root: root, modelKey: modelKey)
            let disk = try #require(coordinator.diskCache)
            let companion = try #require(coordinator.ssmStateCache.diskStore)
            let tokens = Self.tokens(1_003, seed: 76)
            let hash = Self.kvHash(tokens, modelKey)
            let key = Self.ssmKey(tokens, modelKey)
            let tensorURL = Self.companionURLs(root, key)[0]
            let sidecarURL = Self.companionURLs(root, key)[1]

            disk.store(tokens: tokens, arrays: Self.kv(), enforceQuota: false)
            try FileManager.default.createDirectory(at: sidecarURL, withIntermediateDirectories: true)
            try Data("in the way".utf8).write(to: sidecarURL.appendingPathComponent("occupant"))

            #expect(throws: (any Error).self) {
                _ = try companion.store(
                    ssmStates: Self.recurrent(), tokens: tokens, boundary: tokens.count,
                    enforceQuota: false)
            }
            let tensorBytes = Self.fileBytes(tensorURL)
            try #require(
                tensorBytes > 0, "INVALID: the tensor was not published before the sidecar failed")

            let row = try #require(try Self.indexedRows(root).first { $0.hash == hash })
            #expect(row.companionKey == key)
            #expect(row.companionBytes == tensorBytes)
            let stillValidated = companion.hasValidatedCompleteEntry(
                tokens: tokens, boundary: tokens.count)
            #expect(!stillValidated)
            try Self.expectUsageMatchesDisk(disk, root: root)

            // Once the obstacle is gone the same store heals the entry.
            try FileManager.default.removeItem(at: sidecarURL)
            try companion.store(
                ssmStates: Self.recurrent(), tokens: tokens, boundary: tokens.count,
                enforceQuota: false)
            let healed = try #require(try Self.indexedRows(root).first { $0.hash == hash })
            #expect(healed.companionBytes == Self.companionBytes(root, key))
            #expect(healed.companionBytes > tensorBytes)
            #expect(companion.fetch(tokens: tokens, boundary: tokens.count) != nil)
            try Self.expectUsageMatchesDisk(disk, root: root)
        }
    }

    /// R3, the over-count direction: a rewrite removes the old tensor and the
    /// rename of the new one then fails. That interleaving cannot be arranged
    /// from outside the process, so this uses the store's internal
    /// `writeFaultForTesting` seam.
    @Test func moveFailureStopsCountingTheRemovedTensor() throws {
        try MLXMetalTestLock.withLock {
            let root = Self.makeRoot("move-failure")
            defer { try? FileManager.default.removeItem(at: root) }
            let modelKey = "accounting-move-failure"
            let coordinator = Self.coordinator(root: root, modelKey: modelKey)
            let disk = try #require(coordinator.diskCache)
            let companion = try #require(coordinator.ssmStateCache.diskStore)
            let tokens = Self.tokens(517, seed: 77)
            let fresh = Self.tokens(1_291, seed: 78)
            let hash = Self.kvHash(tokens, modelKey)
            let key = Self.ssmKey(tokens, modelKey)

            coordinator.storePersistentBoundary(
                tokens: tokens, diskArrays: Self.kv(), ssmStates: Self.recurrent(64, states: 1))
            let before = try #require(try Self.indexedRows(root).first { $0.hash == hash })
            try #require(before.companionBytes == Self.companionBytes(root, key))

            companion.writeFaultForTesting = { stage in
                if stage == .moveTensorIntoPlace { throw InjectedFault() }
            }
            // A different state count: a real rewrite, not the touch-only skip.
            #expect(throws: InjectedFault.self) {
                _ = try companion.store(
                    ssmStates: Self.recurrent(4_099, states: 2), tokens: tokens,
                    boundary: tokens.count, enforceQuota: false)
            }
            // A fresh key that fails the same way leaves nothing at all.
            #expect(throws: InjectedFault.self) {
                _ = try companion.store(
                    ssmStates: Self.recurrent(), tokens: fresh, boundary: fresh.count,
                    enforceQuota: false)
            }
            companion.writeFaultForTesting = nil

            try #require(
                Self.fileBytes(Self.companionURLs(root, key)[0]) == 0,
                "INVALID: the old tensor survived the failed rewrite")
            let sidecarBytes = Self.fileBytes(Self.companionURLs(root, key)[1])
            try #require(sidecarBytes > 0)

            let after = try #require(try Self.indexedRows(root).first { $0.hash == hash })
            #expect(after.companionKey == key)
            #expect(after.companionBytes == sidecarBytes, "the removed tensor is still counted")
            #expect(try Self.legacyRows(root).isEmpty)
            #expect(Self.companionBytes(root, Self.ssmKey(fresh, modelKey)) == 0)
            let stillValidated = companion.hasValidatedCompleteEntry(
                tokens: tokens, boundary: tokens.count)
            #expect(!stillValidated)
            let names = try FileManager.default.contentsOfDirectory(atPath: Self.companionDir(root).path)
            #expect(names == ["ssm-\(key).json"], "an unpublished file was left behind")
            try Self.expectUsageMatchesDisk(disk, root: root)
        }
    }

    /// R4. The direct writers (`maybeReDeriveSSMState`, `SSMStateCache.store`
    /// with its default `persistToDisk`) can put a companion on disk before
    /// its KV row exists. When the row arrives in a call that writes no
    /// companion, the companion must join it, or the quota retires the
    /// hottest companion first and leaves its KV payload useless.
    @Test func companionStoredBeforeItsRowIsAdoptedWhenTheRowArrives() throws {
        try MLXMetalTestLock.withLock {
            let root = Self.makeRoot("adopt")
            defer { try? FileManager.default.removeItem(at: root) }
            let modelKey = "accounting-adopt"
            let older = Self.tokens(517, seed: 81)
            let early = Self.tokens(1_003, seed: 82)
            let earlyKey = Self.ssmKey(early, modelKey)

            let cap: Int64
            do {
                let writer = Self.coordinator(root: root, modelKey: modelKey)
                let disk = try #require(writer.diskCache)
                let companion = try #require(writer.ssmStateCache.diskStore)

                writer.storePersistentBoundary(
                    tokens: older, diskArrays: Self.kv(), ssmStates: Self.recurrent())
                let past = Date(timeIntervalSince1970: 10_000)
                try #require(disk.touchRecency(tokens: older, at: past))
                try #require(companion.touchRecency(tokens: older, boundary: older.count, at: past))

                // Companion first, through the in-memory cache's write-through.
                writer.ssmStateCache.store(
                    ssmStates: Self.recurrent(), tokens: early, boundary: early.count)
                try #require(
                    try Self.legacyRows(root) == [earlyKey: Self.companionBytes(root, earlyKey)])

                // Its KV row, in a call that writes no companion.
                writer.storePersistentBoundary(tokens: early, diskArrays: Self.kv(), ssmStates: nil)
                let row = try #require(
                    try Self.indexedRows(root).first { $0.hash == Self.kvHash(early, modelKey) })
                #expect(row.companionKey == earlyKey)
                #expect(row.companionBytes == Self.companionBytes(root, earlyKey))
                #expect(try Self.legacyRows(root).isEmpty)
                try Self.expectUsageMatchesDisk(disk, root: root)

                // Removing the early companion ALONE would relieve this cap,
                // which is what a legacy-first pass does.
                cap = try Self.onDiskBytesOfIndexedFiles(root) - Self.companionBytes(root, earlyKey) / 2
                try #require(cap < 1 << 24, "cap must survive the Float GiB round trip exactly")

                // What R4 costs a store that has nothing to adopt: one
                // primary-key SELECT on an empty table, plus the key hash.
                var samples: [UInt64] = []
                for index in 0..<1_000 {
                    let start = DispatchTime.now().uptimeNanoseconds
                    _ = disk.adoptLegacyCompanion(kvHash: row.hash, companionKey: "absent-\(index)")
                    samples.append(DispatchTime.now().uptimeNanoseconds - start)
                }
                samples.sort()
                let long = Self.tokens(32_003, seed: 83)
                var hashSamples: [UInt64] = []
                for _ in 0..<21 {
                    let start = DispatchTime.now().uptimeNanoseconds
                    _ = SSMCompanionDiskStore.keyFor(tokens: long, boundary: long.count, modelKey: modelKey)
                    hashSamples.append(DispatchTime.now().uptimeNanoseconds - start)
                }
                hashSamples.sort()
                print(
                    "COMPANION_ACCOUNTING adoptLegacyCompanion_empty "
                        + "median_ms=\(String(format: "%.4f", Double(samples[500]) / 1_000_000)) "
                        + "max_ms=\(String(format: "%.4f", Double(samples[999]) / 1_000_000)) samples=1000 "
                        + "keyFor_32003_tokens_median_ms=\(String(format: "%.3f", Double(hashSamples[10]) / 1_000_000))")
            }

            // Same process, so no import runs that could re-link anything.
            let pressed = Self.coordinator(root: root, capBytes: cap, modelKey: modelKey)
            let disk = try #require(pressed.diskCache)
            try #require(Int64(disk.maxSizeBytes) == cap)
            #expect(disk.snapshotStats().evictions == 1)
            #expect(
                Self.companionBytes(root, earlyKey) > 0,
                "the just-written companion was evicted ahead of an older group")
            #expect(FileManager.default.fileExists(
                atPath: Self.payloadURL(root, Self.kvHash(early, modelKey)).path))
            #expect(!FileManager.default.fileExists(
                atPath: Self.payloadURL(root, Self.kvHash(older, modelKey)).path))
            #expect(Self.companionBytes(root, Self.ssmKey(older, modelKey)) == 0)
            try Self.expectUsageMatchesDisk(disk, root: root)
        }
    }

    /// O1. The host's "Clear SSD Cache" deletes files and rows with its own
    /// SQL. It does not know about `legacy_companions`.
    @Test func reconcileDiskAccountingAfterAnExternalPurge() throws {
        try MLXMetalTestLock.withLock {
            let root = Self.makeRoot("purge")
            defer { try? FileManager.default.removeItem(at: root) }
            let modelKey = "accounting-purge"
            let coordinator = Self.coordinator(root: root, modelKey: modelKey)
            let disk = try #require(coordinator.diskCache)
            let companion = try #require(coordinator.ssmStateCache.diskStore)
            let linked = [Self.tokens(301, seed: 91), Self.tokens(1_003, seed: 92)]
            let loose = Self.tokens(517, seed: 93)
            let phantom = Self.tokens(307, seed: 94)
            let looseKey = Self.ssmKey(loose, modelKey)
            let phantomKey = Self.ssmKey(phantom, modelKey)

            for tokens in linked {
                coordinator.storePersistentBoundary(
                    tokens: tokens, diskArrays: Self.kv(), ssmStates: Self.recurrent())
            }
            for tokens in [loose, phantom] {
                coordinator.storePersistentBoundary(
                    tokens: tokens, diskArrays: nil, ssmStates: Self.recurrent())
            }
            try #require(try Self.legacyRows(root).count == 2)
            try Self.expectUsageMatchesDisk(disk, root: root)

            // The purge tool: raw SQL and file deletion, none of this package.
            do {
                let raw = try RawDB(root: root)
                let hashes = Set(try raw.rows("SELECT hash FROM cache_entries").compactMap { $0[0] })
                try #require(hashes.count == 2)
                for hash in hashes {
                    try FileManager.default.removeItem(at: Self.payloadURL(root, hash))
                }
                let dir = Self.companionDir(root)
                for name in try FileManager.default.contentsOfDirectory(atPath: dir.path)
                where name.hasPrefix("ssm-") && name.hasSuffix(".json") {
                    let sidecarURL = dir.appendingPathComponent(name)
                    guard let sidecar = try JSONSerialization.jsonObject(
                        with: Data(contentsOf: sidecarURL)) as? [String: Any],
                        let kvHash = sidecar["kv_hash"] as? String, hashes.contains(kvHash)
                    else { continue }
                    try FileManager.default.removeItem(at: sidecarURL)
                    try FileManager.default.removeItem(
                        at: sidecarURL.deletingPathExtension().appendingPathExtension("safetensors"))
                }
                try raw.require("DELETE FROM cache_entries")
            }
            // One unlinked companion's files went as well; its row names nothing.
            for url in Self.companionURLs(root, phantomKey) {
                try FileManager.default.removeItem(at: url)
            }
            let looseBytes = Self.companionBytes(root, looseKey)
            try #require(looseBytes > 0, "INVALID: the purge was not supposed to reach this one")
            try #require(
                disk.usageBytes() > looseBytes, "INVALID: the purge left nothing stale to reconcile")

            #expect(coordinator.reconcileDiskAccounting())
            #expect(disk.usageBytes() == looseBytes)
            #expect(try Self.legacyRows(root) == [looseKey: looseBytes])
            #expect(try Self.indexedRows(root).isEmpty)
            let stats = try #require(coordinator.snapshotStats().diskStats)
            #expect(stats.currentPayloadBytes == Int(looseBytes))
            #expect(stats.currentEntryCount == 1)
            try Self.expectUsageMatchesDisk(disk, root: root, atLeast: looseBytes)

            // The cache carries on: a purged boundary stores and restores.
            coordinator.storePersistentBoundary(
                tokens: linked[0], diskArrays: Self.kv(), ssmStates: Self.recurrent())
            #expect(disk.fetch(tokens: linked[0]) != nil)
            #expect(companion.fetch(tokens: linked[0], boundary: linked[0].count) != nil)
            #expect(try Self.indexedRows(root).first?.companionKey == Self.ssmKey(linked[0], modelKey))
            try Self.expectUsageMatchesDisk(disk, root: root, atLeast: looseBytes + 1)
        }
    }

    /// O5. Two loaded models share one root through separate connections,
    /// each with its own in-memory view.
    @Test func twoCoordinatorsOnOneRootStayCoherent() throws {
        try MLXMetalTestLock.withLock {
            let root = Self.makeRoot("two-models")
            defer { try? FileManager.default.removeItem(at: root) }
            let keyA = "accounting-two-models-a"
            let keyB = "accounting-two-models-b"
            let b1 = Self.tokens(301, seed: 101)
            let b2 = Self.tokens(517, seed: 102)
            let a1 = Self.tokens(1_003, seed: 103)

            let modelB = Self.coordinator(root: root, modelKey: keyB)
            let diskB = try #require(modelB.diskCache)
            let companionB = try #require(modelB.ssmStateCache.diskStore)
            for (index, tokens) in [b1, b2].enumerated() {
                modelB.storePersistentBoundary(
                    tokens: tokens, diskArrays: Self.kv(), ssmStates: Self.recurrent())
                let at = Date(timeIntervalSince1970: 10_000 * Double(index + 1))
                try #require(diskB.touchRecency(tokens: tokens, at: at))
                try #require(companionB.touchRecency(tokens: tokens, boundary: tokens.count, at: at))
            }
            let b1Bytes = Self.fileBytes(Self.payloadURL(root, Self.kvHash(b1, keyB)))
                + Self.companionBytes(root, Self.ssmKey(b1, keyB))
            let cap = diskB.usageBytes() + b1Bytes / 2
            try #require(cap < 1 << 24, "cap must survive the Float GiB round trip exactly")

            // Model A has the smaller cap; its own store pushes the root over.
            let modelA = Self.coordinator(root: root, capBytes: cap, modelKey: keyA)
            let diskA = try #require(modelA.diskCache)
            try #require(Int64(diskA.maxSizeBytes) == cap)
            #expect(diskA.snapshotStats().evictions == 0)
            modelA.storePersistentBoundary(
                tokens: a1, diskArrays: Self.kv(), ssmStates: Self.recurrent())

            // A's pass retired B's oldest group: files and row.
            #expect(diskA.snapshotStats().evictions == 1)
            #expect(!FileManager.default.fileExists(
                atPath: Self.payloadURL(root, Self.kvHash(b1, keyB)).path))
            #expect(Self.companionBytes(root, Self.ssmKey(b1, keyB)) == 0)
            #expect(try Self.indexedRows(root).map(\.hash).sorted()
                == [Self.kvHash(b2, keyB), Self.kvHash(a1, keyA)].sorted())
            try Self.expectUsageMatchesDisk(diskA, root: root)
            try Self.expectUsageMatchesDisk(diskB, root: root)

            // B still believes it validated that boundary. It must find a
            // miss, not a crash and not a stale hit.
            if case .hit = modelB.fetch(tokens: b1) {
                Issue.record("model B restored a boundary model A had evicted")
            }
            #expect(diskB.fetch(tokens: b1) == nil)
            let stillValidated = modelB.hasValidatedDiskEntry(tokens: b1)
            #expect(!stillValidated)
            try Self.expectUsageMatchesDisk(diskB, root: root)

            // B stores it again (B's own cap is large, so nothing is evicted).
            modelB.storePersistentBoundary(
                tokens: b1, diskArrays: Self.kv(), ssmStates: Self.recurrent())
            let restored = try #require(
                try Self.indexedRows(root).first { $0.hash == Self.kvHash(b1, keyB) })
            #expect(restored.companionKey == Self.ssmKey(b1, keyB))
            #expect(diskB.fetch(tokens: b1) != nil)
            #expect(try Self.indexedRows(root).count == 3)
            #expect(diskA.usageBytes() == diskB.usageBytes())
            try Self.expectUsageMatchesDisk(diskA, root: root)
            try Self.expectUsageMatchesDisk(diskB, root: root)
        }
    }

    /// O7. With a ledger attached, the companion store's own quota (direct
    /// writes) reads the index. The directory here can be written to but not
    /// listed, so a walk finds nothing and evicts nothing.
    @Test func directCompanionWriteEvictsWithoutListingTheDirectory() throws {
        try MLXMetalTestLock.withLock {
            let sizingRoot = Self.makeRoot("no-walk-sizing")
            let root = Self.makeRoot("no-walk")
            let dir = Self.companionDir(root)
            defer {
                try? FileManager.default.setAttributes(
                    [.posixPermissions: 0o755], ofItemAtPath: dir.path)
                try? FileManager.default.removeItem(at: sizingRoot)
                try? FileManager.default.removeItem(at: root)
            }
            let modelKey = "accounting-no-walk"
            let boundaries = [Self.tokens(301, seed: 111), Self.tokens(517, seed: 112), Self.tokens(1_003, seed: 113)]

            let oneCompanion: Int64
            do {
                let sizing = Self.coordinator(root: sizingRoot, modelKey: modelKey)
                sizing.storePersistentBoundary(
                    tokens: boundaries[0], diskArrays: Self.kv(16), ssmStates: Self.recurrent())
                oneCompanion = Self.companionBytes(sizingRoot, Self.ssmKey(boundaries[0], modelKey))
                try #require(oneCompanion > 0)
            }

            // Room for two companions and a half; the KV payloads are tiny.
            let coordinator = Self.coordinator(
                root: root, capBytes: oneCompanion * 5 / 2, modelKey: modelKey)
            let disk = try #require(coordinator.diskCache)
            for tokens in boundaries.prefix(2) {
                disk.store(tokens: tokens, arrays: Self.kv(16), enforceQuota: false)
                coordinator.ssmStateCache.store(
                    ssmStates: Self.recurrent(), tokens: tokens, boundary: tokens.count)
            }
            disk.store(tokens: boundaries[2], arrays: Self.kv(16), enforceQuota: false)
            try #require(try Self.indexedRows(root).filter { $0.companionKey != nil }.count == 2)

            // Write + search, no read: files can be created, renamed, stat'ed
            // and removed, but the directory cannot be listed.
            try FileManager.default.setAttributes(
                [.posixPermissions: 0o300], ofItemAtPath: dir.path)
            try #require(
                (try? FileManager.default.contentsOfDirectory(atPath: dir.path)) == nil,
                "INVALID: chmod 300 had no effect (root / filesystem)")

            coordinator.ssmStateCache.store(
                ssmStates: Self.recurrent(), tokens: boundaries[2], boundary: boundaries[2].count)
            try #require(
                Self.companionBytes(root, Self.ssmKey(boundaries[2], modelKey)) > 0,
                "INVALID: the write itself failed in the unlistable directory")

            // The third write pushed the companions over the cap, and the
            // oldest went — decided from the index.
            #expect(Self.companionBytes(root, Self.ssmKey(boundaries[0], modelKey)) == 0)
            let rows = try Self.indexedRows(root)
            #expect(rows.count == 3)
            #expect(rows.first { $0.hash == Self.kvHash(boundaries[0], modelKey) }?.companionKey == nil)
            #expect(rows.filter { $0.companionKey != nil }.count == 2)

            try FileManager.default.setAttributes(
                [.posixPermissions: 0o755], ofItemAtPath: dir.path)
            try Self.expectUsageMatchesDisk(disk, root: root)
        }
    }

    // MARK: - Review follow-ups

    /// An upgraded directory (companions on disk, none counted) whose first
    /// coordinator could not import it: the index write lock was held past
    /// the busy timeout. Returns that coordinator and the rows a committed
    /// import must reproduce. The lock has been released on return.
    private static func coordinatorWhoseImportWasBlocked(
        root: URL, modelKey: String, seeds: (Int, Int), clock: TestClock
    ) throws -> (coordinator: CacheCoordinator, populated: [IndexedRow]) {
        let populated: [IndexedRow]
        do {
            let writer = coordinator(root: root, modelKey: modelKey)
            for boundary in [tokens(301, seed: seeds.0), tokens(1_003, seed: seeds.1)] {
                writer.storePersistentBoundary(
                    tokens: boundary, diskArrays: kv(), ssmStates: recurrent())
            }
            populated = try indexedRows(root)
            try #require(populated.count == 2)
            try #require(populated.allSatisfy { $0.companionKey != nil && $0.companionBytes > 0 })
        }
        try RawDB(root: root).require(
            "UPDATE cache_entries SET companion_key = NULL, companion_bytes = 0")
        CacheCoordinator.resetImportedRootsForTesting()

        let blocker = try RawDB(root: root)
        try blocker.require("BEGIN IMMEDIATE")
        let blocked = coordinator(root: root, modelKey: modelKey, busyTimeoutMs: 50, clock: clock)
        try #require(blocked.diskCache?.indexHasV2Columns == true)
        try #require(
            try indexedRows(root).allSatisfy { $0.companionKey == nil },
            "INVALID: the import was not blocked")
        try blocker.require("COMMIT")
        return (blocked, populated)
    }

    /// S1. One model, loaded for hours: no second coordinator ever opens the
    /// root, so the coordinator whose import was skipped has to retry it.
    @Test func failedImportIsRetriedFromTheQuotaPassOfTheSameCoordinator() throws {
        try MLXMetalTestLock.withLock {
            let root = Self.makeRoot("import-retry")
            defer { try? FileManager.default.removeItem(at: root) }
            let modelKey = "accounting-import-retry"
            let clock = TestClock()
            let (coordinator, populated) = try Self.coordinatorWhoseImportWasBlocked(
                root: root, modelKey: modelKey, seeds: (121, 122), clock: clock)
            let disk = try #require(coordinator.diskCache)
            let populatedHashes = Set(populated.map(\.hash))

            clock.advance(61)
            coordinator.storePersistentBoundary(
                tokens: Self.tokens(517, seed: 123), diskArrays: Self.kv(),
                ssmStates: Self.recurrent())

            #expect(try Self.indexedRows(root).filter { populatedHashes.contains($0.hash) } == populated)
            try Self.expectUsageMatchesDisk(disk, root: root)

            // Having committed, the pass does not import again.
            let hashList = populatedHashes.map { "'\($0)'" }.joined(separator: ",")
            try RawDB(root: root).require(
                "UPDATE cache_entries SET companion_key = NULL, companion_bytes = 0 WHERE hash IN (\(hashList))")
            clock.advance(61)
            coordinator.storePersistentBoundary(
                tokens: Self.tokens(307, seed: 124), diskArrays: Self.kv(), ssmStates: nil)
            #expect(
                try Self.indexedRows(root).filter { populatedHashes.contains($0.hash) }
                    .allSatisfy { $0.companionKey == nil })
        }
    }

    /// S1. A permanently locked index must cost one walk and one busy wait
    /// per interval, not per store. The companion directory cannot be listed
    /// here, so a retry inside the interval would see no companions, commit
    /// an empty import and mark the root done — and the retry that IS due,
    /// once the directory is readable again, would then never run.
    @Test func failedImportIsNotRetriedBeforeTheIntervalHasPassed() throws {
        try MLXMetalTestLock.withLock {
            let root = Self.makeRoot("import-retry-paced")
            let dir = Self.companionDir(root)
            defer {
                try? FileManager.default.setAttributes(
                    [.posixPermissions: 0o755], ofItemAtPath: dir.path)
                try? FileManager.default.removeItem(at: root)
            }
            let modelKey = "accounting-import-retry-paced"
            let clock = TestClock()
            let (coordinator, populated) = try Self.coordinatorWhoseImportWasBlocked(
                root: root, modelKey: modelKey, seeds: (125, 126), clock: clock)
            let disk = try #require(coordinator.diskCache)
            let populatedHashes = Set(populated.map(\.hash))

            try FileManager.default.setAttributes(
                [.posixPermissions: 0o000], ofItemAtPath: dir.path)
            try Self.requireUnlistable(dir)
            clock.advance(59)
            coordinator.storePersistentBoundary(
                tokens: Self.tokens(1_291, seed: 127), diskArrays: Self.kv(), ssmStates: nil)
            try FileManager.default.setAttributes(
                [.posixPermissions: 0o755], ofItemAtPath: dir.path)
            #expect(
                try Self.indexedRows(root).filter { populatedHashes.contains($0.hash) }
                    .allSatisfy { $0.companionKey == nil })

            clock.advance(2)
            coordinator.storePersistentBoundary(
                tokens: Self.tokens(307, seed: 128), diskArrays: Self.kv(), ssmStates: nil)
            #expect(try Self.indexedRows(root).filter { populatedHashes.contains($0.hash) } == populated)
            try Self.expectUsageMatchesDisk(disk, root: root)
        }
    }

    /// (g) An on-demand reconcile that could not commit leaves the index as
    /// the purge left it. The root must stop counting as imported, so the
    /// quota pass retries on the same schedule as a skipped import at open.
    @Test func uncommittedOnDemandReconcileIsRetriedFromTheQuotaPass() throws {
        try MLXMetalTestLock.withLock {
            let root = Self.makeRoot("reconcile-retry")
            defer { try? FileManager.default.removeItem(at: root) }
            let modelKey = "accounting-reconcile-retry"
            let clock = TestClock()
            CacheCoordinator.resetImportedRootsForTesting()
            let coordinator = Self.coordinator(
                root: root, modelKey: modelKey, busyTimeoutMs: 50, clock: clock)
            let disk = try #require(coordinator.diskCache)
            for tokens in [Self.tokens(301, seed: 129), Self.tokens(1_003, seed: 130)] {
                coordinator.storePersistentBoundary(
                    tokens: tokens, diskArrays: Self.kv(), ssmStates: Self.recurrent())
            }
            let populated = try Self.indexedRows(root)
            try #require(populated.count == 2)
            try #require(populated.allSatisfy { $0.companionKey != nil && $0.companionBytes > 0 })
            let populatedHashes = Set(populated.map(\.hash))

            // Something outside the package rewrote the rows.
            try RawDB(root: root).require(
                "UPDATE cache_entries SET companion_key = NULL, companion_bytes = 0")
            let blocker = try RawDB(root: root)
            try blocker.require("BEGIN IMMEDIATE")
            try #require(
                !coordinator.reconcileDiskAccounting(), "INVALID: the reconcile was not blocked")
            try blocker.require("COMMIT")

            clock.advance(61)
            coordinator.storePersistentBoundary(
                tokens: Self.tokens(307, seed: 134), diskArrays: Self.kv(), ssmStates: nil)
            #expect(try Self.indexedRows(root).filter { populatedHashes.contains($0.hash) } == populated)
            try Self.expectUsageMatchesDisk(disk, root: root)
        }
    }

    /// S2. The import walks the companion directory before it takes the
    /// index write lock. A companion written and recorded in between (a
    /// second process on the root, or a direct writer racing an on-demand
    /// reconcile) is in the index and not in the walk.
    @Test func companionRecordedBetweenTheWalkAndTheTransactionIsKept() throws {
        try MLXMetalTestLock.withLock {
            let root = Self.makeRoot("walk-gap")
            defer { try? FileManager.default.removeItem(at: root) }
            let modelKey = "accounting-walk-gap"
            let coordinator = Self.coordinator(root: root, modelKey: modelKey)
            let disk = try #require(coordinator.diskCache)
            let companion = try #require(coordinator.ssmStateCache.diskStore)
            let before = Self.tokens(301, seed: 141)
            let linkedInGap = Self.tokens(517, seed: 142)
            let looseInGap = Self.tokens(1_003, seed: 143)
            let staleBytesInGap = Self.tokens(1_291, seed: 144)
            let goneInGap = Self.tokens(307, seed: 145)

            coordinator.storePersistentBoundary(
                tokens: before, diskArrays: Self.kv(), ssmStates: Self.recurrent())
            let walked = companion.quotaEntries()
            try #require(walked.map(\.hash) == [Self.ssmKey(before, modelKey)])

            // The gap.
            for tokens in [linkedInGap, staleBytesInGap, goneInGap] {
                coordinator.storePersistentBoundary(
                    tokens: tokens, diskArrays: Self.kv(), ssmStates: Self.recurrent())
            }
            coordinator.storePersistentBoundary(
                tokens: looseInGap, diskArrays: nil, ssmStates: Self.recurrent())
            try RawDB(root: root).require(
                "UPDATE cache_entries SET companion_bytes = 7 WHERE hash = '\(Self.kvHash(staleBytesInGap, modelKey))'")
            // The control: a record whose files really are gone is still dropped.
            for url in Self.companionURLs(root, Self.ssmKey(goneInGap, modelKey)) {
                try FileManager.default.removeItem(at: url)
            }
            let looseKey = Self.ssmKey(looseInGap, modelKey)
            try #require(try Self.legacyRows(root) == [looseKey: Self.companionBytes(root, looseKey)])

            let summary = try #require(disk.reconcileCompanionAccounting(companions: walked))
            #expect(summary.linksCleared == 1)
            #expect(summary.legacyDeleted == 0)

            let rows = try Self.indexedRows(root)
            for tokens in [before, linkedInGap, staleBytesInGap] {
                let row = try #require(rows.first { $0.hash == Self.kvHash(tokens, modelKey) })
                let key = Self.ssmKey(tokens, modelKey)
                #expect(row.companionKey == key)
                #expect(row.companionBytes == Self.companionBytes(root, key))
                #expect(row.companionBytes > 7)
            }
            let gone = try #require(rows.first { $0.hash == Self.kvHash(goneInGap, modelKey) })
            #expect(gone.companionKey == nil)
            #expect(gone.companionBytes == 0)
            #expect(try Self.legacyRows(root) == [looseKey: Self.companionBytes(root, looseKey)])
            try Self.expectUsageMatchesDisk(disk, root: root)
        }
    }

    // MARK: Undeletable files

    private static func setImmutable(_ url: URL, _ immutable: Bool) throws {
        try FileManager.default.setAttributes([.immutable: immutable], ofItemAtPath: url.path)
    }

    private static func clearImmutableFlags(under root: URL) {
        guard let enumerator = FileManager.default.enumerator(at: root, includingPropertiesForKeys: nil)
        else { return }
        for case let url as URL in enumerator {
            try? FileManager.default.setAttributes([.immutable: false], ofItemAtPath: url.path)
        }
    }

    /// The immutable flag is what makes these deletes fail. On a file system
    /// without it the tests below would prove nothing.
    private static func requireImmutableBlocksDeletion(
        in root: URL, sourceLocation: SourceLocation = #_sourceLocation
    ) throws {
        let sacrificial = root.appendingPathComponent("immutable-check")
        try Data([1]).write(to: sacrificial)
        try setImmutable(sacrificial, true)
        let removed = (try? FileManager.default.removeItem(at: sacrificial)) != nil
        try? setImmutable(sacrificial, false)
        try? FileManager.default.removeItem(at: sacrificial)
        try #require(
            !removed, "INVALID: an immutable file could be deleted (filesystem)",
            sourceLocation: sourceLocation)
    }

    /// Three linked groups, oldest first, each touched to a fixed recency.
    private static func storeGroups(
        _ groups: [[Int]], through coordinator: CacheCoordinator
    ) throws {
        let disk = try #require(coordinator.diskCache)
        let companion = try #require(coordinator.ssmStateCache.diskStore)
        for (index, tokens) in groups.enumerated() {
            coordinator.storePersistentBoundary(
                tokens: tokens, diskArrays: kv(), ssmStates: recurrent())
            let at = Date(timeIntervalSince1970: 10_000 * Double(index + 1))
            try #require(disk.touchRecency(tokens: tokens, at: at))
            try #require(companion.touchRecency(tokens: tokens, boundary: tokens.count, at: at))
        }
    }

    /// Q18.2. A payload the quota pass cannot delete keeps its row: it is
    /// still on disk, so it is still counted, and it is tried again on the
    /// next pass — once per pass, and without taking the rest of the cache
    /// with it.
    @Test func undeletablePayloadKeepsItsRowUntilItCanBeDeleted() throws {
        try MLXMetalTestLock.withLock {
            let root = Self.makeRoot("undeletable-kv")
            defer {
                Self.clearImmutableFlags(under: root)
                try? FileManager.default.removeItem(at: root)
            }
            let modelKey = "accounting-undeletable-kv"
            let groups = [Self.tokens(301, seed: 151), Self.tokens(517, seed: 152), Self.tokens(1_003, seed: 153)]
            let writer = Self.coordinator(root: root, modelKey: modelKey)
            try Self.storeGroups(groups, through: writer)
            func groupBytes(_ tokens: [Int]) -> Int64 {
                Self.fileBytes(Self.payloadURL(root, Self.kvHash(tokens, modelKey)))
                    + Self.companionBytes(root, Self.ssmKey(tokens, modelKey))
            }
            let newestBytes = groupBytes(groups[2])
            let stuckPayload = Self.payloadURL(root, Self.kvHash(groups[0], modelKey))
            let stuckBytes = Self.fileBytes(stuckPayload)
            try #require(stuckBytes > 0)

            try Self.requireImmutableBlocksDeletion(in: root)
            try Self.setImmutable(stuckPayload, true)

            // Room for one group and a quarter: the two oldest must go. Opening
            // a coordinator with that cap runs the pass.
            let cap = newestBytes + newestBytes / 4
            try #require(cap < 1 << 24, "cap must survive the Float GiB round trip exactly")
            let small = Self.coordinator(root: root, capBytes: cap, modelKey: modelKey)
            let disk = try #require(small.diskCache)
            try #require(Int64(disk.maxSizeBytes) == cap)

            func expectStuckRowAndNewestGroup(_ sourceLocation: SourceLocation = #_sourceLocation) throws {
                let rows = try Self.indexedRows(root)
                #expect(
                    rows.map(\.hash).sorted()
                        == [Self.kvHash(groups[0], modelKey), Self.kvHash(groups[2], modelKey)].sorted(),
                    sourceLocation: sourceLocation)
                let stuck = rows.first { $0.hash == Self.kvHash(groups[0], modelKey) }
                #expect(stuck?.fileSize == stuckBytes, sourceLocation: sourceLocation)
                // Its companion was deletable and went: no longer counted.
                #expect(stuck?.companionKey == nil, sourceLocation: sourceLocation)
                #expect(stuck?.companionBytes == 0, sourceLocation: sourceLocation)
                #expect(
                    rows.first { $0.hash == Self.kvHash(groups[2], modelKey) }?.companionKey
                        == Self.ssmKey(groups[2], modelKey),
                    sourceLocation: sourceLocation)
                #expect(FileManager.default.fileExists(atPath: stuckPayload.path), sourceLocation: sourceLocation)
                #expect(groupBytes(groups[2]) == newestBytes, sourceLocation: sourceLocation)
                #expect(disk.usageBytes() == stuckBytes + newestBytes, sourceLocation: sourceLocation)
                try Self.expectUsageMatchesDisk(disk, root: root, sourceLocation: sourceLocation)
            }

            try expectStuckRowAndNewestGroup()
            #expect(Self.companionBytes(root, Self.ssmKey(groups[0], modelKey)) == 0)
            #expect(groupBytes(groups[1]) == 0)
            #expect(disk.snapshotStats().evictions == 1)

            // Still over the cap, still undeletable: the pass tries the stuck
            // payload again and leaves the newest group alone.
            try #require(disk.usageBytes() > cap, "INVALID: nothing left for a second pass to do")
            small.enforceCombinedDiskQuota()
            try expectStuckRowAndNewestGroup()
            #expect(disk.snapshotStats().evictions == 1)

            try Self.setImmutable(stuckPayload, false)
            small.enforceCombinedDiskQuota()
            #expect(!FileManager.default.fileExists(atPath: stuckPayload.path))
            #expect(try Self.indexedRows(root).map(\.hash) == [Self.kvHash(groups[2], modelKey)])
            #expect(disk.usageBytes() == newestBytes)
            #expect(disk.snapshotStats().evictions == 2)
            try Self.expectUsageMatchesDisk(disk, root: root)
        }
    }

    /// Q18.2, the companion half: the row goes with its payload, and the
    /// companion file that could not be deleted stays counted, unlinked, with
    /// the bytes that are really left.
    @Test func undeletableCompanionStaysCountedAfterItsRowIsEvicted() throws {
        try MLXMetalTestLock.withLock {
            let root = Self.makeRoot("undeletable-companion")
            defer {
                Self.clearImmutableFlags(under: root)
                try? FileManager.default.removeItem(at: root)
            }
            let modelKey = "accounting-undeletable-companion"
            let groups = [Self.tokens(301, seed: 154), Self.tokens(1_003, seed: 155)]
            let writer = Self.coordinator(root: root, modelKey: modelKey)
            try Self.storeGroups(groups, through: writer)
            let stuckKey = Self.ssmKey(groups[0], modelKey)
            let stuckTensor = Self.companionURLs(root, stuckKey)[0]
            let stuckBytes = Self.fileBytes(stuckTensor)
            try #require(stuckBytes > 0)
            let newestBytes = Self.fileBytes(Self.payloadURL(root, Self.kvHash(groups[1], modelKey)))
                + Self.companionBytes(root, Self.ssmKey(groups[1], modelKey))

            try Self.requireImmutableBlocksDeletion(in: root)
            try Self.setImmutable(stuckTensor, true)

            let cap = newestBytes + newestBytes / 4
            try #require(cap < 1 << 24, "cap must survive the Float GiB round trip exactly")
            let small = Self.coordinator(root: root, capBytes: cap, modelKey: modelKey)
            let disk = try #require(small.diskCache)

            for _ in 0..<2 {  // the pass at open, then one more with the flag still set
                #expect(try Self.indexedRows(root).map(\.hash) == [Self.kvHash(groups[1], modelKey)])
                #expect(!FileManager.default.fileExists(
                    atPath: Self.payloadURL(root, Self.kvHash(groups[0], modelKey)).path))
                #expect(try Self.legacyRows(root) == [stuckKey: stuckBytes])
                #expect(disk.usageBytes() == stuckBytes + newestBytes)
                #expect(disk.snapshotStats().evictions == 0)
                try Self.expectUsageMatchesDisk(disk, root: root)
                try #require(disk.usageBytes() > cap, "INVALID: nothing left for a second pass to do")
                small.enforceCombinedDiskQuota()
            }

            try Self.setImmutable(stuckTensor, false)
            small.enforceCombinedDiskQuota()
            #expect(Self.companionBytes(root, stuckKey) == 0)
            #expect(try Self.legacyRows(root).isEmpty)
            #expect(disk.usageBytes() == newestBytes)
            #expect(disk.snapshotStats().evictions == 1)
            try Self.expectUsageMatchesDisk(disk, root: root)
        }
    }

    /// Q18.2, the companion store's own cap on a direct write.
    @Test func directCompanionEvictionKeepsTheRecordOfAnUndeletableCompanion() throws {
        try MLXMetalTestLock.withLock {
            let sizingRoot = Self.makeRoot("undeletable-direct-sizing")
            let root = Self.makeRoot("undeletable-direct")
            defer {
                Self.clearImmutableFlags(under: root)
                try? FileManager.default.removeItem(at: sizingRoot)
                try? FileManager.default.removeItem(at: root)
            }
            let modelKey = "accounting-undeletable-direct"
            let boundaries = [
                Self.tokens(301, seed: 156), Self.tokens(517, seed: 157),
                Self.tokens(1_003, seed: 158), Self.tokens(1_291, seed: 159),
            ]
            let oneCompanion: Int64
            do {
                let sizing = Self.coordinator(root: sizingRoot, modelKey: modelKey)
                sizing.storePersistentBoundary(
                    tokens: boundaries[0], diskArrays: Self.kv(16), ssmStates: Self.recurrent())
                oneCompanion = Self.companionBytes(sizingRoot, Self.ssmKey(boundaries[0], modelKey))
                try #require(oneCompanion > 0)
            }

            // Room for two companions and a half; the KV payloads are tiny.
            let coordinator = Self.coordinator(
                root: root, capBytes: oneCompanion * 5 / 2, modelKey: modelKey)
            let disk = try #require(coordinator.diskCache)
            for tokens in boundaries {
                disk.store(tokens: tokens, arrays: Self.kv(16), enforceQuota: false)
            }
            for tokens in boundaries.prefix(2) {
                coordinator.ssmStateCache.store(
                    ssmStates: Self.recurrent(), tokens: tokens, boundary: tokens.count)
            }
            let stuckKey = Self.ssmKey(boundaries[0], modelKey)
            let stuckTensor = Self.companionURLs(root, stuckKey)[0]
            let stuckBytes = Self.fileBytes(stuckTensor)
            try Self.requireImmutableBlocksDeletion(in: root)
            try Self.setImmutable(stuckTensor, true)

            // Third companion: over the cap, the oldest is chosen, and its
            // tensor cannot be deleted.
            coordinator.ssmStateCache.store(
                ssmStates: Self.recurrent(), tokens: boundaries[2], boundary: boundaries[2].count)
            var rows = try Self.indexedRows(root)
            let stuckRow = try #require(rows.first { $0.hash == Self.kvHash(boundaries[0], modelKey) })
            #expect(stuckRow.companionKey == stuckKey)
            #expect(stuckRow.companionBytes == stuckBytes)
            #expect(rows.filter { $0.companionKey != nil }.count == 3)
            try Self.expectUsageMatchesDisk(disk, root: root)

            try Self.setImmutable(stuckTensor, false)
            coordinator.ssmStateCache.store(
                ssmStates: Self.recurrent(), tokens: boundaries[3], boundary: boundaries[3].count)
            rows = try Self.indexedRows(root)
            #expect(Self.companionBytes(root, stuckKey) == 0)
            #expect(rows.first { $0.hash == Self.kvHash(boundaries[0], modelKey) }?.companionKey == nil)
            try Self.expectUsageMatchesDisk(disk, root: root)
        }
    }

    // MARK: Payloads without a row

    /// Q18.1. A payload with no row is counted by nothing and evicted by
    /// nothing, so it must not be served either. `fetch` leaves the file
    /// alone: the insert may be in flight on another connection.
    @Test func payloadWithoutARowIsNotServed() throws {
        try MLXMetalTestLock.withLock {
            let root = Self.makeRoot("rowless-fetch")
            defer { try? FileManager.default.removeItem(at: root) }
            let modelKey = "accounting-rowless-fetch"
            let disk = DiskCache(cacheDir: root, maxSizeBytes: 1 << 30, modelKey: modelKey)
            try #require(disk.indexHasV2Columns)
            let tokens = Self.tokens(1_003, seed: 161)
            let hash = Self.kvHash(tokens, modelKey)
            let payload = Self.payloadURL(root, hash)
            disk.store(tokens: tokens, arrays: Self.kv(), enforceQuota: false)
            try #require(disk.fetch(tokens: tokens) != nil, "INVALID: the entry was never restorable")
            let bytes = Self.fileBytes(payload)

            try RawDB(root: root).require("DELETE FROM cache_entries")
            let missesBefore = disk.snapshotStats().misses
            #expect(disk.fetch(tokens: tokens) == nil)
            #expect(disk.snapshotStats().misses == missesBefore + 1)
            #expect(Self.fileBytes(payload) == bytes)

            // The row arrives (the other connection's insert): served again.
            try RawDB(root: root).require(
                "INSERT INTO cache_entries (hash, token_count, file_size) VALUES ('\(hash)', \(tokens.count), \(bytes))")
            #expect(disk.fetch(tokens: tokens) != nil)
            try Self.expectUsageMatchesDisk(disk, root: root)
        }
    }

    /// Q18.1. The import removes a payload with no row once it is older than
    /// the guard age, and only then: a younger one may be another process's
    /// store between its publish and its insert.
    @Test func oldUnindexedPayloadIsRemovedAndAYoungOneIsLeftAlone() throws {
        try MLXMetalTestLock.withLock {
            let root = Self.makeRoot("rowless-sweep")
            defer { try? FileManager.default.removeItem(at: root) }
            let modelKey = "accounting-rowless-sweep"
            let coordinator = Self.coordinator(root: root, modelKey: modelKey)
            let disk = try #require(coordinator.diskCache)
            let companion = try #require(coordinator.ssmStateCache.diskStore)
            let indexed = Self.tokens(301, seed: 162)
            let old = Self.tokens(517, seed: 163)
            let nineMinutes = Self.tokens(1_003, seed: 164)
            let young = Self.tokens(1_291, seed: 165)

            coordinator.storePersistentBoundary(
                tokens: indexed, diskArrays: Self.kv(), ssmStates: Self.recurrent())
            for tokens in [old, nineMinutes, young] {
                coordinator.storePersistentBoundary(
                    tokens: tokens, diskArrays: Self.kv(), ssmStates: nil)
            }
            func payload(_ tokens: [Int]) -> URL { Self.payloadURL(root, Self.kvHash(tokens, modelKey)) }
            // Takes a name, not the tokens, so a failure does not print them.
            let payloads = ["indexed": indexed, "old": old, "nineMinutes": nineMinutes, "young": young]
            func exists(_ name: String) -> Bool {
                // Force-unwrapped: a mistyped name must not read as "absent".
                FileManager.default.fileExists(atPath: payload(payloads[name]!).path)
            }
            func age(_ tokens: [Int], minutes: Double) throws {
                try FileManager.default.setAttributes(
                    [.modificationDate: Date().addingTimeInterval(-minutes * 60)],
                    ofItemAtPath: payload(tokens).path)
            }
            let rowless = [old, nineMinutes, young].map { "'\(Self.kvHash($0, modelKey))'" }
            try RawDB(root: root).require(
                "DELETE FROM cache_entries WHERE hash IN (\(rowless.joined(separator: ",")))")
            // The control: as old as the one that goes, but it has a row.
            try age(indexed, minutes: 11)
            try age(old, minutes: 11)
            try age(nineMinutes, minutes: 9)

            // The production path and the production guard age (10 minutes).
            #expect(coordinator.reconcileDiskAccounting())
            #expect(!exists("old"))
            #expect(exists("nineMinutes"))
            #expect(exists("young"))
            #expect(exists("indexed"))
            #expect(try Self.indexedRows(root).map(\.hash) == [Self.kvHash(indexed, modelKey)])
            try Self.expectUsageMatchesDisk(disk, root: root, checkCompleteness: false)

            // The guard age is a parameter: five minutes reaches the second.
            let summary = try #require(
                disk.reconcileCompanionAccounting(
                    companions: companion.quotaEntries(), unindexedPayloadGuardAge: 300))
            #expect(summary.unindexedPayloadsRemoved == 1)
            #expect(!exists("nineMinutes"))
            #expect(exists("young"))

            // Nothing is left to remove: the young one stays, run after run.
            let again = try #require(
                disk.reconcileCompanionAccounting(companions: companion.quotaEntries()))
            #expect(!again.changedAnything)
            #expect(exists("young"))

            try age(young, minutes: 11)
            #expect(coordinator.reconcileDiskAccounting())
            #expect(!exists("young"))
            #expect(exists("indexed"))
            #expect(disk.fetch(tokens: indexed) != nil)
            try Self.expectUsageMatchesDisk(disk, root: root)
        }
    }

    /// Q18.1. The sweep deletes what the index does not name, so it must
    /// know that it read the index. An index that cannot be read is not an
    /// empty index.
    @Test func unreadableIndexNeverTriggersThePayloadSweep() throws {
        try MLXMetalTestLock.withLock {
            let root = Self.makeRoot("rowless-unreadable")
            defer { try? FileManager.default.removeItem(at: root) }
            let modelKey = "accounting-rowless-unreadable"
            let disk = DiskCache(cacheDir: root, maxSizeBytes: 1 << 30, modelKey: modelKey)
            try #require(disk.indexHasV2Columns)
            let tokens = Self.tokens(1_003, seed: 166)
            let payload = Self.payloadURL(root, Self.kvHash(tokens, modelKey))
            disk.store(tokens: tokens, arrays: Self.kv(), enforceQuota: false)
            try FileManager.default.setAttributes(
                [.modificationDate: Date().addingTimeInterval(-11 * 60)], ofItemAtPath: payload.path)

            let raw = try RawDB(root: root)
            try raw.require("ALTER TABLE cache_entries RENAME TO cache_entries_moved")
            let unreadable = disk.reconcileCompanionAccounting(companions: [])
            try raw.require("ALTER TABLE cache_entries_moved RENAME TO cache_entries")
            #expect(unreadable == nil)
            #expect(FileManager.default.fileExists(atPath: payload.path))

            let readable = try #require(disk.reconcileCompanionAccounting(companions: []))
            #expect(!readable.changedAnything)
            #expect(disk.fetch(tokens: tokens) != nil)
            try Self.expectUsageMatchesDisk(disk, root: root)
        }
    }

    // MARK: A failed record of an entry that is already counted

    /// (c) Re-storing a validated companion only touches its files and
    /// re-records it. When that write loses to another connection's lock the
    /// index still counts the pair, so deleting it would throw away a valid,
    /// counted entry.
    @Test func busyIndexDoesNotDeleteAnAlreadyCountedCompanion() throws {
        try MLXMetalTestLock.withLock {
            let root = Self.makeRoot("counted-busy")
            defer { try? FileManager.default.removeItem(at: root) }
            let modelKey = "accounting-counted-busy"
            let coordinator = Self.coordinator(root: root, modelKey: modelKey, busyTimeoutMs: 50)
            let disk = try #require(coordinator.diskCache)
            let companion = try #require(coordinator.ssmStateCache.diskStore)
            let tokens = Self.tokens(517, seed: 171)  // no KV row: counted as unlinked
            let key = Self.ssmKey(tokens, modelKey)

            try companion.store(
                ssmStates: Self.recurrent(), tokens: tokens, boundary: tokens.count,
                enforceQuota: false)
            let bytes = Self.companionBytes(root, key)
            try #require(try Self.legacyRows(root) == [key: bytes])

            let blocker = try RawDB(root: root)
            try blocker.require("BEGIN IMMEDIATE")
            let skipsBefore = companion.snapshotStoreSkips()
            let record = try companion.store(
                ssmStates: Self.recurrent(), tokens: tokens, boundary: tokens.count,
                enforceQuota: false)
            try blocker.require("COMMIT")
            try #require(
                companion.snapshotStoreSkips() == skipsBefore + 1,
                "INVALID: the second store was not the touch-only path")
            try #require(
                disk.snapshotStats().failedIndexWrites == 1,
                "INVALID: the index write was not refused")

            #expect(Self.companionBytes(root, key) == bytes)
            #expect(record?.bytes == bytes)
            #expect(try Self.legacyRows(root) == [key: bytes])
            let restorable = companion.fetch(tokens: tokens, boundary: tokens.count) != nil
            #expect(restorable)
            try Self.expectUsageMatchesDisk(disk, root: root)
        }
    }

    /// (c), the other side: the index names the key, but with fewer bytes
    /// than the rewrite left on disk. Keeping those files would under-count,
    /// so they still go; the stale record over-counts until it is reconciled.
    @Test func busyIndexStillRemovesARewriteLargerThanItsRecord() throws {
        try MLXMetalTestLock.withLock {
            let root = Self.makeRoot("counted-busy-larger")
            defer { try? FileManager.default.removeItem(at: root) }
            let modelKey = "accounting-counted-busy-larger"
            let coordinator = Self.coordinator(root: root, modelKey: modelKey, busyTimeoutMs: 50)
            let disk = try #require(coordinator.diskCache)
            let companion = try #require(coordinator.ssmStateCache.diskStore)
            let tokens = Self.tokens(1_003, seed: 172)
            let key = Self.ssmKey(tokens, modelKey)
            coordinator.storePersistentBoundary(
                tokens: tokens, diskArrays: Self.kv(), ssmStates: Self.recurrent())
            let recorded = Self.companionBytes(root, key)
            try #require(try Self.indexedRows(root).first?.companionBytes == recorded)

            let blocker = try RawDB(root: root)
            try blocker.require("BEGIN IMMEDIATE")
            let record = try companion.store(
                ssmStates: Self.recurrent(states: 2), tokens: tokens, boundary: tokens.count,
                enforceQuota: false)
            try blocker.require("COMMIT")
            try #require(
                disk.snapshotStats().failedIndexWrites == 1,
                "INVALID: the index write was not refused")

            #expect(record == nil)
            #expect(Self.companionBytes(root, key) == 0)
            // Over-counted, never under-counted.
            #expect(disk.usageBytes() == Self.fileBytes(Self.payloadURL(root, Self.kvHash(tokens, modelKey))) + recorded)
            #expect(coordinator.reconcileDiskAccounting())
            try Self.expectUsageMatchesDisk(disk, root: root)
        }
    }
}
