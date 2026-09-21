import Foundation
import MLX
import SQLite3
import Testing

@testable import MLXLMCommon

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
        (0 ..< count).map { seed * 100_000 + $0 }
    }

    private static func kv(_ elements: Int = 1_024) -> [String: MLXArray] {
        ["data": MLXArray.ones([elements], dtype: .float32)]
    }

    private static func recurrent(_ elements: Int = 1_024, states: Int = 1) -> [MLXArray] {
        (0 ..< states).map { _ in MLXArray.ones([elements], dtype: .float32) }
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
        let coordinator =
            clock.map { clock in
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
    private static func payloadURL(_ root: URL, _ hash: String) -> URL {
        Support.payloadURL(root, hash)
    }
    private static func companionURLs(_ root: URL, _ key: String) -> [URL] {
        Support.companionURLs(root, key)
    }
    private static func fileBytes(_ url: URL) -> Int64 { Support.fileBytes(url) }
    private static func companionBytes(_ root: URL, _ key: String) -> Int64 {
        Support.companionBytes(root, key)
    }
    private static func indexedRows(_ root: URL) throws -> [IndexedRow] {
        try Support.indexedRows(root)
    }
    private static func legacyRows(_ root: URL) throws -> [String: Int64] {
        try Support.legacyRows(root)
    }
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

    private static func makeV1OnlyIndex(in root: URL) throws {
        try Support.makeV1OnlyIndex(in: root)
    }

    // MARK: - 1

    @Test func companionBytesAreCountedFromTheIndex() throws {
        try MLXMetalTestLock.withLock {
            let root = Self.makeRoot("counted")
            defer { try? FileManager.default.removeItem(at: root) }
            let modelKey = "accounting-counted"
            let coordinator = Self.coordinator(root: root, modelKey: modelKey)
            let disk = try #require(coordinator.diskCache)
            try #require(disk.indexHasV2Columns)

            let boundaries = [
                Self.tokens(301, seed: 1), Self.tokens(517, seed: 2), Self.tokens(1_003, seed: 3),
            ]
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
            #expect(
                linked.companionBytes == Self.companionBytes(root, Self.ssmKey(withRow, modelKey)))
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
            let boundaries = [
                Self.tokens(301, seed: 8), Self.tokens(517, seed: 9), Self.tokens(1_291, seed: 10),
            ]

            let populated: [IndexedRow]
            do {
                let writer = Self.coordinator(root: root, modelKey: modelKey)
                for tokens in boundaries {
                    writer.storePersistentBoundary(
                        tokens: tokens, diskArrays: Self.kv(), ssmStates: Self.recurrent())
                }
                populated = try Self.indexedRows(root)
                try #require(populated.count == 3)
                try #require(
                    populated.allSatisfy { $0.companionKey != nil && $0.companionBytes > 0 })
            }

            // An older build's three-column INSERT OR REPLACE leaves exactly
            // this behind: rows present, companion columns at their defaults.
            func wipeCompanionColumns() throws {
                try RawDB(root: root).require(
                    "UPDATE cache_entries SET companion_key = NULL, companion_bytes = 0")
                try #require(
                    try Self.indexedRows(root).allSatisfy {
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
            try FileManager.default.removeItem(
                at: Self.payloadURL(root, Self.kvHash(lost, modelKey)))

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
            let strippedRow = try #require(
                rows.first { $0.hash == Self.kvHash(stripped, modelKey) })
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
                try #require(
                    try Self.indexedRows(root).first {
                        $0.hash == Self.kvHash(tokens, modelKey)
                    }
                ).fileSize
            }

            // A row with no companion: usage drops by exactly its bytes.
            let denseBytes = try fileSize(dense)
            var before = disk.usageBytes()
            try FileManager.default.removeItem(
                at: Self.payloadURL(root, Self.kvHash(dense, modelKey)))
            #expect(disk.fetch(tokens: dense) == nil)
            #expect(try Self.indexedRows(root).count == 2)
            #expect(disk.usageBytes() == before - denseBytes)
            try Self.expectUsageMatchesDisk(disk, root: root)

            // A row with a companion: the row's bytes go, the companion files
            // are still on disk and stay counted, now unlinked.
            let hybridBytes = try fileSize(hybrid)
            before = disk.usageBytes()
            try FileManager.default.removeItem(
                at: Self.payloadURL(root, Self.kvHash(hybrid, modelKey)))
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

            // The next open sweeps it — once it is old enough that no write
            // can still be producing it; the published entry is untouched.
            try Self.age(partial, by: Self.elevenMinutes)
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
            for index in 0 ..< entries {
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
            for _ in 0 ..< 100 {
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
            let companionTotal = try Self.indexedRows(root).reduce(Int64(0)) {
                $0 + $1.companionBytes
            }
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
                        JSONSerialization.jsonObject(with: Data(contentsOf: sidecarURL))
                            as? [String: Any])
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
                    try RawDB(root: root).rows("SELECT hash FROM cache_entries").compactMap {
                        $0[0]
                    })
                for hash in survivingKV {
                    #expect(
                        FileManager.default.fileExists(atPath: Self.payloadURL(root, hash).path))
                }
                let payloadsOnDisk = try FileManager.default.contentsOfDirectory(atPath: root.path)
                    .filter { $0.hasSuffix(".safetensors") }
                #expect(Set(payloadsOnDisk) == Set(survivingKV.map { "\($0).safetensors" }))
                let companionNames = try FileManager.default
                    .contentsOfDirectory(atPath: Self.companionDir(root).path)
                let survivingCompanions = Set(
                    companionNames.compactMap { name -> String? in
                        guard name.hasPrefix("ssm-"), name.hasSuffix(".safetensors") else {
                            return nil
                        }
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
            let boundaries = [
                Self.tokens(301, seed: 41), Self.tokens(517, seed: 42),
                Self.tokens(1_003, seed: 43),
            ]

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
                    #expect(
                        disk.touchRecency(
                            tokens: tokens,
                            at: Date(timeIntervalSince1970: 10_000 * Double(index + 1))))
                    #expect(
                        try #require(coordinator.ssmStateCache.diskStore).touchRecency(
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
                    .rows("SELECT name FROM pragma_table_info('cache_entries')").compactMap {
                        $0[0]
                    }
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
            #expect(
                !FileManager.default.fileExists(
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
            let boundaries = [
                Self.tokens(301, seed: 61), Self.tokens(517, seed: 62),
                Self.tokens(1_003, seed: 63),
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
            for (index, tokens) in boundaries.enumerated() {
                disk.store(tokens: tokens, arrays: Self.kv(16), enforceQuota: false)
                // Direct write-through, with the companion store's own quota.
                coordinator.ssmStateCache.store(
                    ssmStates: Self.recurrent(), tokens: tokens, boundary: tokens.count)
                for url in Self.companionURLs(root, Self.ssmKey(tokens, modelKey))
                where FileManager.default.fileExists(atPath: url.path) {
                    try FileManager.default.setAttributes(
                        [
                            .modificationDate: Date(
                                timeIntervalSince1970: 10_000 * Double(index + 1))
                        ],
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
            let left = try FileManager.default.contentsOfDirectory(
                atPath: Self.companionDir(root).path)
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
                try #require(
                    populated.allSatisfy { $0.companionKey != nil && $0.companionBytes > 0 })
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
            #expect(
                try Self.indexedRows(root).first?.companionKey == Self.ssmKey(withRow, modelKey))
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
            try FileManager.default.createDirectory(
                at: sidecarURL, withIntermediateDirectories: true)
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

    /// R3, the over-count direction: the rename that publishes a rewritten
    /// tensor fails. The write used to unlink the old tensor first, so the
    /// failure left a sidecar alone and a record that over-counted it; the
    /// publish is now one `rename(2)` over the old tensor, and a failure
    /// leaves the old pair, and its count, exactly as they were. A failing
    /// rename cannot be arranged from outside the process, so this uses the
    /// store's internal `writeFaultForTesting` seam.
    @Test func moveFailureKeepsTheOldTensorAndItsCount() throws {
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
            let oldPair = try Self.companionURLs(root, key).map { try Data(contentsOf: $0) }

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

            // Publishing is one rename over the old tensor, so a rename that
            // does not happen has cost nothing: the old pair is still there,
            // byte for byte, and still counted for what it is.
            let pairNow = Self.companionURLs(root, key).map { try? Data(contentsOf: $0) }
            #expect(pairNow == oldPair, "the old valid pair did not survive the failed rewrite")

            let after = try #require(try Self.indexedRows(root).first { $0.hash == hash })
            #expect(after.companionKey == key)
            #expect(after.companionBytes == before.companionBytes)
            #expect(try Self.legacyRows(root).isEmpty)
            #expect(Self.companionBytes(root, Self.ssmKey(fresh, modelKey)) == 0)
            let stillValidated = companion.hasValidatedCompleteEntry(
                tokens: tokens, boundary: tokens.count)
            #expect(!stillValidated)
            let names = try FileManager.default.contentsOfDirectory(
                atPath: Self.companionDir(root).path)
            #expect(
                names.sorted() == ["ssm-\(key).json", "ssm-\(key).safetensors"],
                "an unpublished file was left behind")
            #expect(companion.fetch(tokens: tokens, boundary: tokens.count)?.states.count == 1)
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
                cap =
                    try Self.onDiskBytesOfIndexedFiles(root) - Self.companionBytes(root, earlyKey)
                    / 2
                try #require(cap < 1 << 24, "cap must survive the Float GiB round trip exactly")

                // What R4 costs a store that has nothing to adopt: one
                // primary-key SELECT on an empty table, plus the key hash.
                var samples: [UInt64] = []
                for index in 0 ..< 1_000 {
                    let start = DispatchTime.now().uptimeNanoseconds
                    _ = disk.adoptLegacyCompanion(kvHash: row.hash, companionKey: "absent-\(index)")
                    samples.append(DispatchTime.now().uptimeNanoseconds - start)
                }
                samples.sort()
                let long = Self.tokens(32_003, seed: 83)
                var hashSamples: [UInt64] = []
                for _ in 0 ..< 21 {
                    let start = DispatchTime.now().uptimeNanoseconds
                    _ = SSMCompanionDiskStore.keyFor(
                        tokens: long, boundary: long.count, modelKey: modelKey)
                    hashSamples.append(DispatchTime.now().uptimeNanoseconds - start)
                }
                hashSamples.sort()
                print(
                    "COMPANION_ACCOUNTING adoptLegacyCompanion_empty "
                        + "median_ms=\(String(format: "%.4f", Double(samples[500]) / 1_000_000)) "
                        + "max_ms=\(String(format: "%.4f", Double(samples[999]) / 1_000_000)) samples=1000 "
                        + "keyFor_32003_tokens_median_ms=\(String(format: "%.3f", Double(hashSamples[10]) / 1_000_000))"
                )
            }

            // Same process, so no import runs that could re-link anything.
            let pressed = Self.coordinator(root: root, capBytes: cap, modelKey: modelKey)
            let disk = try #require(pressed.diskCache)
            try #require(Int64(disk.maxSizeBytes) == cap)
            #expect(disk.snapshotStats().evictions == 1)
            #expect(
                Self.companionBytes(root, earlyKey) > 0,
                "the just-written companion was evicted ahead of an older group")
            #expect(
                FileManager.default.fileExists(
                    atPath: Self.payloadURL(root, Self.kvHash(early, modelKey)).path))
            #expect(
                !FileManager.default.fileExists(
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
                let hashes = Set(
                    try raw.rows("SELECT hash FROM cache_entries").compactMap { $0[0] })
                try #require(hashes.count == 2)
                for hash in hashes {
                    try FileManager.default.removeItem(at: Self.payloadURL(root, hash))
                }
                let dir = Self.companionDir(root)
                for name in try FileManager.default.contentsOfDirectory(atPath: dir.path)
                where name.hasPrefix("ssm-") && name.hasSuffix(".json") {
                    let sidecarURL = dir.appendingPathComponent(name)
                    guard
                        let sidecar = try JSONSerialization.jsonObject(
                            with: Data(contentsOf: sidecarURL)) as? [String: Any],
                        let kvHash = sidecar["kv_hash"] as? String, hashes.contains(kvHash)
                    else { continue }
                    try FileManager.default.removeItem(at: sidecarURL)
                    try FileManager.default.removeItem(
                        at: sidecarURL.deletingPathExtension().appendingPathExtension("safetensors")
                    )
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
                disk.usageBytes() > looseBytes, "INVALID: the purge left nothing stale to reconcile"
            )

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
            #expect(
                try Self.indexedRows(root).first?.companionKey == Self.ssmKey(linked[0], modelKey))
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
                try #require(
                    companionB.touchRecency(tokens: tokens, boundary: tokens.count, at: at))
            }
            let b1Bytes =
                Self.fileBytes(Self.payloadURL(root, Self.kvHash(b1, keyB)))
                + Self.companionBytes(root, Self.ssmKey(b1, keyB))
            let cap = diskB.usageBytes() + b1Bytes / 2
            try #require(cap < 1 << 24, "cap must survive the Float GiB round trip exactly")

            // Loading model A updates the root's shared cap; its store pushes it over.
            let modelA = Self.coordinator(root: root, capBytes: cap, modelKey: keyA)
            let diskA = try #require(modelA.diskCache)
            try #require(Int64(diskA.maxSizeBytes) == cap)
            #expect(diskB.maxSizeBytes == diskA.maxSizeBytes)
            #expect(diskA.snapshotStats().evictions == 0)
            modelA.storePersistentBoundary(
                tokens: a1, diskArrays: Self.kv(), ssmStates: Self.recurrent())

            // A's pass retired B's oldest group: files and row.
            #expect(diskA.snapshotStats().evictions == 1)
            #expect(
                !FileManager.default.fileExists(
                    atPath: Self.payloadURL(root, Self.kvHash(b1, keyB)).path))
            #expect(Self.companionBytes(root, Self.ssmKey(b1, keyB)) == 0)
            #expect(
                try Self.indexedRows(root).map(\.hash).sorted()
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

            // B stores it again under the SAME root cap. It retires the next
            // oldest group, rather than silently restoring its old larger cap.
            modelB.storePersistentBoundary(
                tokens: b1, diskArrays: Self.kv(), ssmStates: Self.recurrent())
            let restored = try #require(
                try Self.indexedRows(root).first { $0.hash == Self.kvHash(b1, keyB) })
            #expect(restored.companionKey == Self.ssmKey(b1, keyB))
            #expect(diskB.fetch(tokens: b1) != nil)
            #expect(try Self.indexedRows(root).count == 2)
            #expect(diskB.fetch(tokens: b2) == nil)
            #expect(diskB.snapshotStats().evictions == 1)
            #expect(diskA.usageBytes() <= cap)
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
            let boundaries = [
                Self.tokens(301, seed: 111), Self.tokens(517, seed: 112),
                Self.tokens(1_003, seed: 113),
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
            #expect(
                rows.first { $0.hash == Self.kvHash(boundaries[0], modelKey) }?.companionKey == nil)
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

            #expect(
                try Self.indexedRows(root).filter { populatedHashes.contains($0.hash) } == populated
            )
            try Self.expectUsageMatchesDisk(disk, root: root)

            // Having committed, the pass does not import again.
            let hashList = populatedHashes.map { "'\($0)'" }.joined(separator: ",")
            try RawDB(root: root).require(
                "UPDATE cache_entries SET companion_key = NULL, companion_bytes = 0 WHERE hash IN (\(hashList))"
            )
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
    /// during the store at 59 s, so an attempt made then cannot commit — an
    /// unlistable directory is not an empty one — and a failed attempt
    /// restarts the pacing timer. That is how a premature retry shows: the
    /// retry that IS due, at 61 s, would find only 2 s on the timer and not
    /// run, and the links would still be missing at the end. That catches a
    /// retry that comes too early; it cannot catch pacing that is missing
    /// altogether, so the store at 59 s is also checked for the line every
    /// uncommitted attempt prints.
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
            let (_, early) = try Self.capturingStandardError {
                coordinator.storePersistentBoundary(
                    tokens: Self.tokens(1_291, seed: 127), diskArrays: Self.kv(), ssmStates: nil)
            }
            try FileManager.default.setAttributes(
                [.posixPermissions: 0o755], ofItemAtPath: dir.path)
            // An attempt that does not commit is always reported, and this
            // one could not have committed. With no pacing at all the timer
            // argument above sees nothing — the attempt at 61 s still runs —
            // so the attempt itself is what is looked for.
            #expect(
                !early.contains("companion import reason=retry"),
                "the import was retried 59 s after it failed")
            #expect(
                try Self.indexedRows(root).filter { populatedHashes.contains($0.hash) }
                    .allSatisfy { $0.companionKey == nil })

            clock.advance(2)
            coordinator.storePersistentBoundary(
                tokens: Self.tokens(307, seed: 128), diskArrays: Self.kv(), ssmStates: nil)
            #expect(
                try Self.indexedRows(root).filter { populatedHashes.contains($0.hash) } == populated
            )
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
            #expect(
                try Self.indexedRows(root).filter { populatedHashes.contains($0.hash) } == populated
            )
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
                "UPDATE cache_entries SET companion_bytes = 7 WHERE hash = '\(Self.kvHash(staleBytesInGap, modelKey))'"
            )
            // The control: a record whose files really are gone is still dropped.
            for url in Self.companionURLs(root, Self.ssmKey(goneInGap, modelKey)) {
                try FileManager.default.removeItem(at: url)
            }
            let looseKey = Self.ssmKey(looseInGap, modelKey)
            try #require(
                try Self.legacyRows(root) == [looseKey: Self.companionBytes(root, looseKey)])

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
        guard
            let enumerator = FileManager.default.enumerator(
                at: root, includingPropertiesForKeys: nil)
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
            let groups = [
                Self.tokens(301, seed: 151), Self.tokens(517, seed: 152),
                Self.tokens(1_003, seed: 153),
            ]
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

            func expectStuckRowAndNewestGroup(_ sourceLocation: SourceLocation = #_sourceLocation)
                throws
            {
                let rows = try Self.indexedRows(root)
                #expect(
                    rows.map(\.hash).sorted()
                        == [Self.kvHash(groups[0], modelKey), Self.kvHash(groups[2], modelKey)]
                        .sorted(),
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
                #expect(
                    FileManager.default.fileExists(atPath: stuckPayload.path),
                    sourceLocation: sourceLocation)
                #expect(groupBytes(groups[2]) == newestBytes, sourceLocation: sourceLocation)
                #expect(
                    disk.usageBytes() == stuckBytes + newestBytes, sourceLocation: sourceLocation)
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
            let newestBytes =
                Self.fileBytes(Self.payloadURL(root, Self.kvHash(groups[1], modelKey)))
                + Self.companionBytes(root, Self.ssmKey(groups[1], modelKey))

            try Self.requireImmutableBlocksDeletion(in: root)
            try Self.setImmutable(stuckTensor, true)

            let cap = newestBytes + newestBytes / 4
            try #require(cap < 1 << 24, "cap must survive the Float GiB round trip exactly")
            let small = Self.coordinator(root: root, capBytes: cap, modelKey: modelKey)
            let disk = try #require(small.diskCache)

            for _ in 0 ..< 2 {  // the pass at open, then one more with the flag still set
                #expect(
                    try Self.indexedRows(root).map(\.hash) == [Self.kvHash(groups[1], modelKey)])
                #expect(
                    !FileManager.default.fileExists(
                        atPath: Self.payloadURL(root, Self.kvHash(groups[0], modelKey)).path))
                #expect(try Self.legacyRows(root) == [stuckKey: stuckBytes])
                #expect(disk.usageBytes() == stuckBytes + newestBytes)
                #expect(disk.snapshotStats().evictions == 0)
                try Self.expectUsageMatchesDisk(disk, root: root)
                try #require(
                    disk.usageBytes() > cap, "INVALID: nothing left for a second pass to do")
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
            let stuckRow = try #require(
                rows.first { $0.hash == Self.kvHash(boundaries[0], modelKey) })
            #expect(stuckRow.companionKey == stuckKey)
            #expect(stuckRow.companionBytes == stuckBytes)
            #expect(rows.filter { $0.companionKey != nil }.count == 3)
            try Self.expectUsageMatchesDisk(disk, root: root)

            try Self.setImmutable(stuckTensor, false)
            coordinator.ssmStateCache.store(
                ssmStates: Self.recurrent(), tokens: boundaries[3], boundary: boundaries[3].count)
            rows = try Self.indexedRows(root)
            #expect(Self.companionBytes(root, stuckKey) == 0)
            #expect(
                rows.first { $0.hash == Self.kvHash(boundaries[0], modelKey) }?.companionKey == nil)
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
            try #require(
                disk.fetch(tokens: tokens) != nil, "INVALID: the entry was never restorable")
            let bytes = Self.fileBytes(payload)

            try RawDB(root: root).require("DELETE FROM cache_entries")
            let missesBefore = disk.snapshotStats().misses
            #expect(disk.fetch(tokens: tokens) == nil)
            #expect(disk.snapshotStats().misses == missesBefore + 1)
            #expect(Self.fileBytes(payload) == bytes)

            // The row arrives (the other connection's insert): served again.
            try RawDB(root: root).require(
                "INSERT INTO cache_entries (hash, token_count, file_size) VALUES ('\(hash)', \(tokens.count), \(bytes))"
            )
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
            func payload(_ tokens: [Int]) -> URL {
                Self.payloadURL(root, Self.kvHash(tokens, modelKey))
            }
            // Takes a name, not the tokens, so a failure does not print them.
            let payloads = [
                "indexed": indexed, "old": old, "nineMinutes": nineMinutes, "young": young,
            ]
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
                [.modificationDate: Date().addingTimeInterval(-11 * 60)], ofItemAtPath: payload.path
            )

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
            #expect(
                disk.usageBytes() == Self.fileBytes(
                    Self.payloadURL(root, Self.kvHash(tokens, modelKey))) + recorded)
            #expect(coordinator.reconcileDiskAccounting())
            try Self.expectUsageMatchesDisk(disk, root: root)
        }
    }

    // MARK: The payload sweep only ever removes our own regular files

    /// Everything written to standard error while `body` runs. The suite is
    /// serialized and these tests hold `MLXMetalTestLock`, so nothing else in
    /// the process is expected to write meanwhile.
    private static func capturingStandardError<T>(_ body: () throws -> T) throws -> (T, String) {
        try Support.capturingStandardError(body)
    }

    private static let elevenMinutes: TimeInterval = 11 * 60

    private static func age(_ url: URL, by seconds: TimeInterval) throws {
        try Support.age(url, by: seconds)
    }

    /// An ACL entry that makes every stat of `url` fail with EACCES while the
    /// directory that holds it stays listable; `false` removes the ACL.
    private static func setStatDenied(_ url: URL, _ denied: Bool) throws {
        let chmod = Process()
        chmod.executableURL = URL(fileURLWithPath: "/bin/chmod")
        chmod.arguments = denied ? ["+a", "everyone deny readattr", url.path] : ["-N", url.path]
        try chmod.run()
        chmod.waitUntilExit()
        try #require(chmod.terminationStatus == 0, "INVALID: chmod \(chmod.arguments ?? []) failed")
    }

    /// Deny, and prove that this is a stat failure and nothing else.
    private static func denyStat(
        of url: URL, sourceLocation: SourceLocation = #_sourceLocation
    ) throws {
        try setStatDenied(url, true)
        try #require(
            lstatErrno(url) == EACCES,
            "INVALID: lstat did not fail with EACCES (root / filesystem)",
            sourceLocation: sourceLocation)
        try #require(
            (try? FileManager.default.contentsOfDirectory(
                atPath: url.deletingLastPathComponent().path))?
                .contains(url.lastPathComponent) == true,
            "INVALID: the directory cannot be listed, so this is not a stat failure",
            sourceLocation: sourceLocation)
    }

    private static func lstatErrno(_ url: URL) -> Int32 {
        var info = stat()
        return lstat(url.path, &info) == 0 ? 0 : errno
    }

    /// R1. The cache root is a user setting. Whatever else lives in it — a
    /// model's shards, a directory, a link — has no row and is old, which is
    /// exactly what the sweep looks for. Only a regular file named the way
    /// this cache names payloads may go.
    @Test func sweepNeverTouchesForeignSafetensors() throws {
        try MLXMetalTestLock.withLock {
            let root = Self.makeRoot("foreign-sweep")
            let outside = Self.makeRoot("foreign-sweep-outside")
            defer {
                try? FileManager.default.removeItem(at: root)
                try? FileManager.default.removeItem(at: outside)
            }
            let modelKey = "accounting-foreign-sweep"
            let coordinator = Self.coordinator(root: root, modelKey: modelKey)
            let disk = try #require(coordinator.diskCache)
            let indexed = Self.tokens(301, seed: 181)
            let control = Self.tokens(517, seed: 182)
            coordinator.storePersistentBoundary(
                tokens: indexed, diskArrays: Self.kv(), ssmStates: Self.recurrent())
            coordinator.storePersistentBoundary(
                tokens: control, diskArrays: Self.kv(), ssmStates: nil)
            let indexedName = "\(Self.kvHash(indexed, modelKey)).safetensors"
            let controlName = "\(Self.kvHash(control, modelKey)).safetensors"
            try RawDB(root: root).require(
                "DELETE FROM cache_entries WHERE hash = '\(Self.kvHash(control, modelKey))'")

            // Not ours, each for one reason.
            let foreignFiles: [String: Data] = [
                "foo.safetensors": Data(repeating: 0x11, count: 4_099),
                "model-00001-of-00002.safetensors": Data(repeating: 0x22, count: 70_001),
                "ABCDEF0123456789ABCDEF0123456789.safetensors": Data(repeating: 0x33, count: 1_031),
                "0123456789abcdef0123456789abcde.safetensors": Data(repeating: 0x44, count: 1_033),
                "0123456789abcdef0123456789abcdef0.safetensors": Data(
                    repeating: 0x55, count: 1_039),
            ]
            for (name, data) in foreignFiles {
                try #require(name.hasSuffix(".safetensors"))
                try data.write(to: root.appendingPathComponent(name))
            }
            let directory = root.appendingPathComponent(
                "0123456789abcdef0123456789abcdef.safetensors")
            let inner = directory.appendingPathComponent("inner.bin")
            let innerData = Data(repeating: 0x66, count: 2_053)
            try FileManager.default.createDirectory(
                at: directory, withIntermediateDirectories: true)
            try innerData.write(to: inner)

            try FileManager.default.createDirectory(at: outside, withIntermediateDirectories: true)
            let target = outside.appendingPathComponent("weights.safetensors")
            let targetData = Data(repeating: 0x77, count: 3_001)
            try targetData.write(to: target)
            let link = root.appendingPathComponent("fedcba9876543210fedcba9876543210.safetensors")
            try FileManager.default.createSymbolicLink(at: link, withDestinationURL: target)

            // All of it as old as the control, whichever date the sweep reads.
            for name in Array(foreignFiles.keys) + [indexedName, controlName] {
                try Self.age(root.appendingPathComponent(name), by: Self.elevenMinutes)
            }
            try Self.age(target, by: Self.elevenMinutes)
            try Self.age(link, by: Self.elevenMinutes)
            try Self.age(directory, by: Self.elevenMinutes)

            func listing() throws -> Set<String> {
                Set(try FileManager.default.contentsOfDirectory(atPath: root.path))
            }
            let before = try listing()
            try #require(
                before.contains(controlName), "INVALID: the control payload is not on disk")

            // The production path and the production guard age.
            #expect(coordinator.reconcileDiskAccounting())

            let removed = before.subtracting(try listing())
            #expect(
                removed == [controlName],
                "the sweep removed \(removed.sorted()); only \(controlName) was its to remove")
            for (name, data) in foreignFiles.sorted(by: { $0.key < $1.key }) {
                let now = try? Data(contentsOf: root.appendingPathComponent(name))
                #expect(now == data, "foreign file \(name) did not survive the sweep byte for byte")
            }
            #expect(
                (try? Data(contentsOf: inner)) == innerData,
                "a DIRECTORY named like a payload was removed with its contents")
            #expect(
                (try? FileManager.default.destinationOfSymbolicLink(atPath: link.path))
                    == target.path,
                "a symlink named like a payload was removed")
            #expect(
                (try? Data(contentsOf: target)) == targetData, "the symlink's target was touched")
            #expect(disk.fetch(tokens: indexed) != nil)
            try Self.expectUsageMatchesDisk(disk, root: root, checkCompleteness: false)
        }
    }

    /// R1. A root that holds `config.json` or `jang_config.json` looks like a
    /// model bundle. Nothing is swept there, not even a payload that is ours.
    @Test func sweepIsSkippedForAModelBundleRoot() throws {
        try MLXMetalTestLock.withLock {
            for marker in ["config.json", "jang_config.json"] {
                let root = Self.makeRoot("bundle-root")
                defer { try? FileManager.default.removeItem(at: root) }
                let modelKey = "accounting-bundle-root-\(marker)"
                let coordinator = Self.coordinator(root: root, modelKey: modelKey)
                let disk = try #require(coordinator.diskCache)
                let kept = Self.tokens(301, seed: 183)
                let rowless = Self.tokens(1_003, seed: 184)
                for tokens in [kept, rowless] {
                    coordinator.storePersistentBoundary(
                        tokens: tokens, diskArrays: Self.kv(), ssmStates: nil)
                }
                let payload = Self.payloadURL(root, Self.kvHash(rowless, modelKey))
                try RawDB(root: root).require(
                    "DELETE FROM cache_entries WHERE hash = '\(Self.kvHash(rowless, modelKey))'")
                try Self.age(payload, by: Self.elevenMinutes)
                let bytes = try Data(contentsOf: payload)
                let markerURL = root.appendingPathComponent(marker)
                try Data("{}".utf8).write(to: markerURL)

                let (committed, log) = try Self.capturingStandardError {
                    coordinator.reconcileDiskAccounting()
                }
                #expect(committed, "a skipped sweep is not a failed import")
                #expect(
                    (try? Data(contentsOf: payload)) == bytes,
                    "\(marker): the sweep ran in a model bundle root")
                let skipLines = log.split(separator: "\n").filter {
                    $0.hasPrefix("[vmlx][cache/disk-index] payload sweep skipped: ")
                }
                #expect(skipLines.count == 1, "\(marker): expected one skip line, got \(skipLines)")
                #expect(skipLines.first?.contains(marker) == true)

                // The control: without the marker the very same file goes.
                try FileManager.default.removeItem(at: markerURL)
                #expect(coordinator.reconcileDiskAccounting())
                #expect(!FileManager.default.fileExists(atPath: payload.path))
                try Self.expectUsageMatchesDisk(disk, root: root)
            }
        }
    }

    /// R1. An index a newer build has claimed may name its payloads another
    /// way. This build does not know which files are that build's, so it
    /// removes none.
    @Test func sweepIsSkippedOnANewerSchema() throws {
        try MLXMetalTestLock.withLock {
            let root = Self.makeRoot("newer-schema")
            defer { try? FileManager.default.removeItem(at: root) }
            let modelKey = "accounting-newer-schema"
            let tokens = Self.tokens(1_003, seed: 185)
            let payload = Self.payloadURL(root, Self.kvHash(tokens, modelKey))
            do {
                let writer = DiskCache(cacheDir: root, maxSizeBytes: 1 << 30, modelKey: modelKey)
                writer.store(tokens: tokens, arrays: Self.kv(), enforceQuota: false)
            }
            try RawDB(root: root).require("DELETE FROM cache_entries")
            try Self.age(payload, by: Self.elevenMinutes)
            let bytes = try Data(contentsOf: payload)

            try RawDB(root: root).require("PRAGMA user_version = 7")
            let newer = DiskCache(cacheDir: root, maxSizeBytes: 1 << 30, modelKey: modelKey)
            try #require(newer.indexSchemaVersion == 7, "INVALID: the newer version was not read")
            try #require(newer.indexHasV2Columns, "INVALID: the import would not run at all")
            let (summary, log) = try Self.capturingStandardError {
                newer.reconcileCompanionAccounting(companions: [])
            }
            #expect(summary != nil, "the rest of the import still commits")
            #expect(summary?.unindexedPayloadsRemoved == 0)
            #expect(
                (try? Data(contentsOf: payload)) == bytes,
                "swept under a schema this build does not know")
            #expect(log.contains("[vmlx][cache/disk-index] payload sweep skipped: "))

            // The control: the same file under the current schema goes.
            try RawDB(root: root).require(
                "PRAGMA user_version = \(DiskCacheIndexSchema.currentVersion)")
            let current = DiskCache(cacheDir: root, maxSizeBytes: 1 << 30, modelKey: modelKey)
            try #require(current.indexSchemaVersion == DiskCacheIndexSchema.currentVersion)
            let swept = try #require(current.reconcileCompanionAccounting(companions: []))
            #expect(swept.unindexedPayloadsRemoved == 1)
            #expect(!FileManager.default.fileExists(atPath: payload.path))
        }
    }

    /// R1. A modification date in the future says nothing about how long a
    /// file has been there, so it is kept. (Pinned, not a regression: a
    /// negative age was already below the guard age.)
    @Test func futureMtimeIsKept() throws {
        try MLXMetalTestLock.withLock {
            let root = Self.makeRoot("future-mtime")
            defer { try? FileManager.default.removeItem(at: root) }
            let modelKey = "accounting-future-mtime"
            let disk = DiskCache(cacheDir: root, maxSizeBytes: 1 << 30, modelKey: modelKey)
            try #require(disk.indexHasV2Columns)
            let tokens = Self.tokens(517, seed: 186)
            let payload = Self.payloadURL(root, Self.kvHash(tokens, modelKey))
            disk.store(tokens: tokens, arrays: Self.kv(), enforceQuota: false)
            try RawDB(root: root).require("DELETE FROM cache_entries")

            try Self.age(payload, by: -86_400)
            let kept = try #require(disk.reconcileCompanionAccounting(companions: []))
            #expect(kept.unindexedPayloadsRemoved == 0)
            #expect(FileManager.default.fileExists(atPath: payload.path))
            // Even against a guard age of zero.
            let zero = try #require(
                disk.reconcileCompanionAccounting(companions: [], unindexedPayloadGuardAge: 0))
            #expect(zero.unindexedPayloadsRemoved == 0)
            #expect(FileManager.default.fileExists(atPath: payload.path))

            // The control: the same file, old, goes.
            try Self.age(payload, by: Self.elevenMinutes)
            let swept = try #require(disk.reconcileCompanionAccounting(companions: []))
            #expect(swept.unindexedPayloadsRemoved == 1)
            #expect(!FileManager.default.fileExists(atPath: payload.path))
        }
    }

    /// R1, the sweep at open. It removes what cannot be read as a complete
    /// safetensors file — which a directory never can, and which a model
    /// shard that is still downloading cannot either.
    @Test func openSweepLeavesDirectoriesLinksAndModelBundleRootsAlone() throws {
        try MLXMetalTestLock.withLock {
            let root = Self.makeRoot("open-sweep-foreign")
            let bundle = Self.makeRoot("open-sweep-bundle")
            let outside = Self.makeRoot("open-sweep-outside")
            defer {
                for url in [root, bundle, outside] { try? FileManager.default.removeItem(at: url) }
            }
            for url in [root, bundle, outside] {
                try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
            }
            let junk = Data(repeating: 0x5A, count: 4_099)  // no safetensors header

            var innerFiles: [URL] = []
            for name in ["0123456789abcdef0123456789abcdef.safetensors", "weights.safetensors"] {
                let directory = root.appendingPathComponent(name)
                try FileManager.default.createDirectory(
                    at: directory, withIntermediateDirectories: true)
                innerFiles.append(directory.appendingPathComponent("inner.bin"))
                try junk.write(to: innerFiles.last!)
            }
            let target = outside.appendingPathComponent("unfinished.safetensors")
            try junk.write(to: target)
            let link = root.appendingPathComponent("fedcba9876543210fedcba9876543210.safetensors")
            try FileManager.default.createSymbolicLink(at: link, withDestinationURL: target)
            // The control: a regular file of ours that is incomplete still goes.
            let ours = root.appendingPathComponent("00112233445566778899aabbccddeeff.safetensors")
            try junk.write(to: ours)

            _ = DiskCache(cacheDir: root, maxSizeBytes: 1 << 30, modelKey: "open-sweep")
            for file in innerFiles {
                #expect(
                    (try? Data(contentsOf: file)) == junk,
                    "the open sweep removed the directory \(file.deletingLastPathComponent().lastPathComponent)"
                )
            }
            #expect(
                (try? FileManager.default.destinationOfSymbolicLink(atPath: link.path))
                    == target.path)
            #expect((try? Data(contentsOf: target)) == junk)
            #expect(
                !FileManager.default.fileExists(atPath: ours.path),
                "INVALID: the open sweep did not run")

            let shard = bundle.appendingPathComponent("model-00001-of-00002.safetensors")
            let oursInBundle = bundle.appendingPathComponent(
                "00112233445566778899aabbccddeeff.safetensors")
            try junk.write(to: shard)
            try junk.write(to: oursInBundle)
            try Data("{}".utf8).write(to: bundle.appendingPathComponent("config.json"))
            _ = DiskCache(cacheDir: bundle, maxSizeBytes: 1 << 30, modelKey: "open-sweep")
            #expect(
                (try? Data(contentsOf: shard)) == junk, "the open sweep ran in a model bundle root")
            #expect((try? Data(contentsOf: oursInBundle)) == junk)
        }
    }

    /// `hash` is a TEXT primary key, which SQLite lets be NULL. Such a row
    /// names no file: it neither hides a payload from the sweep nor exposes
    /// one to it, and it must not stop the import for good.
    @Test func nullHashRowNeitherBlocksTheImportNorExposesAPayload() throws {
        try MLXMetalTestLock.withLock {
            let root = Self.makeRoot("null-hash")
            defer { try? FileManager.default.removeItem(at: root) }
            let modelKey = "accounting-null-hash"
            let disk = DiskCache(cacheDir: root, maxSizeBytes: 1 << 30, modelKey: modelKey)
            try #require(disk.indexHasV2Columns)
            let tokens = Self.tokens(1_291, seed: 187)
            let payload = Self.payloadURL(root, Self.kvHash(tokens, modelKey))
            disk.store(tokens: tokens, arrays: Self.kv(), enforceQuota: false)
            try Self.age(payload, by: Self.elevenMinutes)
            try RawDB(root: root).require(
                "INSERT INTO cache_entries (hash, token_count, file_size) VALUES (NULL, 1, 0)")

            let summary = try #require(disk.reconcileCompanionAccounting(companions: []))
            #expect(summary.unindexedPayloadsRemoved == 0)
            #expect(disk.fetch(tokens: tokens) != nil)
            try Self.expectUsageMatchesDisk(disk, root: root)
        }
    }

    // MARK: "Could not look" is not "not there"

    /// R2. A stat that fails for any reason other than "no such file" says
    /// nothing about whether the file is there. Reading it as "gone" clears
    /// a link (under-count) or deletes a row — and a payload without a row
    /// is what the sweep removes at the next import.
    ///
    /// The error is produced without a product hook: an ACL entry denying
    /// `readattr` on a file leaves its directory listable and makes `lstat`
    /// of the file fail with EACCES — and `FileManager.fileExists` say false.
    @Test func statErrorOtherThanMissingAbandonsTheImport() throws {
        try MLXMetalTestLock.withLock {
            let root = Self.makeRoot("stat-error")
            var denied: [URL] = []
            func restorePermissions() {
                for url in denied { try? Self.setStatDenied(url, false) }
                denied = []
            }
            defer {
                restorePermissions()
                try? FileManager.default.removeItem(at: root)
            }
            let modelKey = "accounting-stat-error"
            let clock = TestClock()
            CacheCoordinator.resetImportedRootsForTesting()
            let coordinator = Self.coordinator(root: root, modelKey: modelKey, clock: clock)
            let disk = try #require(coordinator.diskCache)
            for tokens in [Self.tokens(301, seed: 191), Self.tokens(1_003, seed: 192)] {
                coordinator.storePersistentBoundary(
                    tokens: tokens, diskArrays: Self.kv(), ssmStates: Self.recurrent())
            }
            let populated = try Self.indexedRows(root)
            try #require(populated.count == 2)
            try #require(populated.allSatisfy { $0.companionKey != nil && $0.companionBytes > 0 })
            let populatedHashes = Set(populated.map(\.hash))
            let companionFiles = populated.flatMap { Self.companionURLs(root, $0.companionKey!) }

            // The walk is stale (empty), so every companion the index names
            // is looked at again inside the transaction — and cannot be.
            for url in companionFiles {
                denied.append(url)
                try Self.denyStat(of: url)
            }
            let (direct, directLog) = try Self.capturingStandardError {
                disk.reconcileCompanionAccounting(companions: [])
            }
            let viaCoordinator = coordinator.reconcileDiskAccounting()
            restorePermissions()
            #expect(direct == nil, "an import that could not look at a companion committed")
            #expect(directLog.contains("companion import abandoned"))
            #expect(!viaCoordinator)
            #expect(
                try Self.indexedRows(root) == populated,
                "a link was cleared for a companion that is on disk")
            #expect(try Self.legacyRows(root).isEmpty)
            for url in companionFiles {
                #expect(Self.fileBytes(url) > 0, "\(url.lastPathComponent) is gone")
            }

            // Not marked imported: the quota pass retries, and only an
            // import puts these links back.
            try RawDB(root: root).require(
                "UPDATE cache_entries SET companion_key = NULL, companion_bytes = 0")
            clock.advance(61)
            coordinator.storePersistentBoundary(
                tokens: Self.tokens(307, seed: 193), diskArrays: Self.kv(), ssmStates: nil)
            #expect(
                try Self.indexedRows(root).filter { populatedHashes.contains($0.hash) } == populated
            )
            try Self.expectUsageMatchesDisk(disk, root: root)
        }
    }

    /// R2. The other half: a row whose payload cannot be examined. Since the
    /// sweep exists, deleting that row is worse than an over-count — the
    /// payload is then row-less, and the NEXT import removes it.
    @Test func unexaminablePayloadKeepsItsRowAndAbandonsTheImport() throws {
        try MLXMetalTestLock.withLock {
            let root = Self.makeRoot("payload-stat-error")
            var denied: [URL] = []
            func restorePermissions() {
                for url in denied { try? Self.setStatDenied(url, false) }
                denied = []
            }
            defer {
                restorePermissions()
                try? FileManager.default.removeItem(at: root)
            }
            let modelKey = "accounting-payload-stat-error"
            let clock = TestClock()
            CacheCoordinator.resetImportedRootsForTesting()
            let coordinator = Self.coordinator(root: root, modelKey: modelKey, clock: clock)
            let disk = try #require(coordinator.diskCache)
            for tokens in [Self.tokens(517, seed: 198), Self.tokens(1_291, seed: 199)] {
                coordinator.storePersistentBoundary(
                    tokens: tokens, diskArrays: Self.kv(), ssmStates: Self.recurrent())
            }
            let populated = try Self.indexedRows(root)
            try #require(populated.count == 2)
            let populatedHashes = Set(populated.map(\.hash))
            let payloads = populated.map { Self.payloadURL(root, $0.hash) }
            for url in payloads {
                try Self.age(url, by: Self.elevenMinutes)
                denied.append(url)
                try Self.denyStat(of: url)
            }

            // `fileExists` says false for these payloads. A fetch is a miss,
            // and must not take that for a lost payload either.
            let misses = disk.snapshotStats().misses
            #expect(disk.fetch(tokens: Self.tokens(517, seed: 198)) == nil)
            try #require(disk.snapshotStats().misses == misses + 1)
            #expect(
                try Self.indexedRows(root) == populated,
                "fetch deleted the row of a payload that is on disk")

            let (committed, log) = try Self.capturingStandardError {
                coordinator.reconcileDiskAccounting()
            }
            restorePermissions()
            #expect(!committed, "an import that could not look at a payload committed")
            #expect(log.contains("companion import abandoned"))
            #expect(
                try Self.indexedRows(root) == populated,
                "a row was deleted for a payload that is on disk")

            // The retry that is due commits, and sweeps nothing: the rows
            // were kept, so the old payloads are still indexed.
            try RawDB(root: root).require(
                "UPDATE cache_entries SET companion_key = NULL, companion_bytes = 0")
            clock.advance(61)
            coordinator.storePersistentBoundary(
                tokens: Self.tokens(307, seed: 200), diskArrays: Self.kv(), ssmStates: nil)
            #expect(
                try Self.indexedRows(root).filter { populatedHashes.contains($0.hash) } == populated
            )
            for url in payloads {
                #expect(
                    Self.fileBytes(url) > 0,
                    "\(url.lastPathComponent) was swept after losing its row")
            }
            try Self.expectUsageMatchesDisk(disk, root: root)
        }
    }

    /// R2. The sweep already deletes nothing when the root cannot be listed;
    /// the import must not count as done either, or the sweep it skipped is
    /// never run again in this process.
    @Test func unreadablePayloadListingMakesTheImportNotCommit() throws {
        try MLXMetalTestLock.withLock {
            let root = Self.makeRoot("listing-error")
            func restorePermissions() {
                try? FileManager.default.setAttributes(
                    [.posixPermissions: 0o755], ofItemAtPath: root.path)
            }
            defer {
                restorePermissions()
                try? FileManager.default.removeItem(at: root)
            }
            let modelKey = "accounting-listing-error"
            let clock = TestClock()
            CacheCoordinator.resetImportedRootsForTesting()
            let coordinator = Self.coordinator(root: root, modelKey: modelKey, clock: clock)
            let disk = try #require(coordinator.diskCache)
            let linked = [Self.tokens(301, seed: 194), Self.tokens(1_003, seed: 195)]
            let rowless = Self.tokens(517, seed: 196)
            for tokens in linked {
                coordinator.storePersistentBoundary(
                    tokens: tokens, diskArrays: Self.kv(), ssmStates: Self.recurrent())
            }
            coordinator.storePersistentBoundary(
                tokens: rowless, diskArrays: Self.kv(), ssmStates: nil)
            let rowlessPayload = Self.payloadURL(root, Self.kvHash(rowless, modelKey))
            try RawDB(root: root).require(
                "DELETE FROM cache_entries WHERE hash = '\(Self.kvHash(rowless, modelKey))'")
            try Self.age(rowlessPayload, by: Self.elevenMinutes)
            let populated = try Self.indexedRows(root)
            try #require(populated.count == 2)

            // Search but no read permission: every file can be examined,
            // the directory cannot be listed.
            try FileManager.default.setAttributes(
                [.posixPermissions: 0o300], ofItemAtPath: root.path)
            try Self.requireUnlistable(root)
            try #require(
                Self.lstatErrno(rowlessPayload) == 0, "INVALID: this is a stat failure too")
            let committed = coordinator.reconcileDiskAccounting()
            restorePermissions()
            #expect(
                !committed, "an import whose payload sweep could not list the root counted as done")
            #expect(try Self.indexedRows(root) == populated)
            #expect(Self.fileBytes(rowlessPayload) > 0)

            // Not marked imported: the quota pass retries, and this time the
            // sweep runs.
            clock.advance(61)
            coordinator.storePersistentBoundary(
                tokens: Self.tokens(307, seed: 197), diskArrays: Self.kv(), ssmStates: nil)
            #expect(
                !FileManager.default.fileExists(atPath: rowlessPayload.path),
                "the import was never retried")
            let populatedHashes = Set(populated.map(\.hash))
            #expect(
                try Self.indexedRows(root).filter { populatedHashes.contains($0.hash) } == populated
            )
            try Self.expectUsageMatchesDisk(disk, root: root)
        }
    }

    // MARK: Log tags

    /// R3. Every `[vmlx][cache/disk-quota]` line is a pass summary that
    /// starts `before= after= max=`; a parser relies on it. A failed delete
    /// has its own tag, once per path: a file that can never be deleted is
    /// tried again by every over-cap store.
    @Test func failedDeleteIsReportedOncePerPathUnderItsOwnTag() throws {
        let root = Self.makeRoot("delete-tag")
        defer {
            Self.clearImmutableFlags(under: root)
            try? FileManager.default.removeItem(at: root)
        }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try Self.requireImmutableBlocksDeletion(in: root)
        let stuck = ["stuck-one.safetensors", "stuck-two.safetensors"].map {
            root.appendingPathComponent($0)
        }
        for url in stuck {
            try Data([1, 2, 3]).write(to: url)
            try Self.setImmutable(url, true)
        }

        let (results, log) = try Self.capturingStandardError {
            [stuck[0], stuck[0], stuck[1], stuck[0]].map { DiskCache.removeCacheFile(at: $0) }
        }
        #expect(results == [false, false, false, false])
        let lines = log.split(separator: "\n").map(String.init)
        #expect(
            !lines.contains { $0.hasPrefix("[vmlx][cache/disk-quota]") && !$0.contains(" before=") }
        )
        for url in stuck {
            let mine = lines.filter { $0.contains("path=\(url.path) ") }
            #expect(mine.count == 1, "\(url.lastPathComponent): \(mine)")
            #expect(mine.first?.hasPrefix("[vmlx][cache/disk-delete] failed path=") == true)
            #expect(mine.first?.hasSuffix("— row kept") == true)
        }
    }

    // MARK: Only our own names, only regular files

    /// A safetensors file whose header declares 16 payload bytes it does not
    /// have: what a crashed write, or a download in progress, looks like.
    private static func truncatedSafetensors() -> Data {
        let header = #"{"kv_0_keys":{"dtype":"F32","shape":[4],"data_offsets":[0,16]}}"#
        var data = Data()
        var length = UInt64(header.utf8.count).littleEndian
        data.append(Data(bytes: &length, count: 8))
        data.append(Data(header.utf8))
        return data
    }

    /// Entries that are not this cache's, each for one reason, written into
    /// `dir`. `validName` is a name this cache WOULD use, given to a
    /// directory and to a symlink that points outside.
    private struct ForeignEntries {
        var files: [String: Data] = [:]
        var inner: [URL: Data] = [:]
        var links: [URL: URL] = [:]
        var targets: [URL: Data] = [:]

        var names: Set<String> {
            Set(files.keys)
                .union(inner.keys.map { $0.deletingLastPathComponent().lastPathComponent })
                .union(links.keys.map(\.lastPathComponent))
        }

        mutating func addFile(_ name: String, _ data: Data, in dir: URL) throws {
            try data.write(to: dir.appendingPathComponent(name))
            files[name] = data
        }

        mutating func addDirectory(_ name: String, in dir: URL) throws {
            let directory = dir.appendingPathComponent(name)
            try FileManager.default.createDirectory(
                at: directory, withIntermediateDirectories: true)
            let file = directory.appendingPathComponent("inner.bin")
            let data = Data(repeating: 0x66, count: 2_053)
            try data.write(to: file)
            inner[file] = data
        }

        mutating func addLink(_ name: String, in dir: URL, toNewFileIn outside: URL) throws {
            try FileManager.default.createDirectory(at: outside, withIntermediateDirectories: true)
            let target = outside.appendingPathComponent("target-of-\(name)")
            let data = Data(repeating: 0x77, count: 3_001)
            try data.write(to: target)
            let link = dir.appendingPathComponent(name)
            try FileManager.default.createSymbolicLink(at: link, withDestinationURL: target)
            links[link] = target
            targets[target] = data
        }

        func expectIntact(
            in dir: URL, after what: String, sourceLocation: SourceLocation = #_sourceLocation
        ) {
            for (name, data) in files.sorted(by: { $0.key < $1.key }) {
                let now = try? Data(contentsOf: dir.appendingPathComponent(name))
                #expect(
                    now == data, "foreign file \(name) did not survive \(what) byte for byte",
                    sourceLocation: sourceLocation)
            }
            for (file, data) in inner {
                #expect(
                    (try? Data(contentsOf: file)) == data,
                    "the DIRECTORY \(file.deletingLastPathComponent().lastPathComponent) was removed by \(what)",
                    sourceLocation: sourceLocation)
            }
            for (link, target) in links {
                #expect(
                    (try? FileManager.default.destinationOfSymbolicLink(atPath: link.path))
                        == target.path,
                    "the symlink \(link.lastPathComponent) was removed by \(what)",
                    sourceLocation: sourceLocation)
                #expect(
                    (try? Data(contentsOf: target)) == targets[target],
                    "the target of \(link.lastPathComponent) was touched by \(what)",
                    sourceLocation: sourceLocation)
            }
        }
    }

    private static func listing(_ dir: URL) throws -> Set<String> {
        Set(try FileManager.default.contentsOfDirectory(atPath: dir.path))
    }

    /// The name predicates are checked against what the production name
    /// builders really produce, not against a literal that merely looks right.
    @Test func namePredicatesAcceptExactlyWhatTheCacheProduces() throws {
        let hash = DiskCache.hashTokens(Self.tokens(307, seed: 300), modelKey: "names")
        let final = URL(fileURLWithPath: "/nonexistent/\(hash).safetensors")
        #expect(DiskCache.isPublishedPayloadName(final.lastPathComponent))
        for _ in 0 ..< 8 {
            let partial = DiskCache.temporaryURL(for: final).lastPathComponent
            #expect(DiskCache.isUnpublishedPayloadName(partial), "\(partial)")
            #expect(!DiskCache.isPublishedPayloadName(partial))
        }
        #expect(!DiskCache.isUnpublishedPayloadName(final.lastPathComponent))
        for foreign in [
            "random.partial-abcdefgh.safetensors",
            "something.partial-xyz.safetensors",
            "\(hash).partial-xyz.safetensors",
            "\(hash).partial-1A2B3C4.safetensors",
            "\(hash).partial-1A2B3C4D5.safetensors",
            "\(hash).partial-1A2B3C4G.safetensors",
            "\(hash.dropLast()).partial-1A2B3C4D.safetensors",
            "\(hash)0.partial-1A2B3C4D.safetensors",
            "\(hash.uppercased()).partial-1A2B3C4D.safetensors",
            "\(hash).partial-1A2B3C4D.safetensors.bak",
            "\(hash).partial-1A2B3C4D.partial-1A2B3C4D.safetensors",
            "\(hash).tmp-1A2B3C4D.safetensors",
        ] {
            #expect(!DiskCache.isUnpublishedPayloadName(foreign), "\(foreign)")
        }

        let key = SSMCompanionDiskStore.keyFor(
            tokens: Self.tokens(307, seed: 300), boundary: 307, modelKey: "names")
        try #require(key.utf8.count == 64, "INVALID: the companion key is not 64 characters")
        let tensor = URL(fileURLWithPath: "/nonexistent/ssm-\(key).safetensors")
        #expect(SSMCompanionDiskStore.publishedEntryKey(fromName: tensor.lastPathComponent) == key)
        #expect(SSMCompanionDiskStore.publishedEntryKey(fromName: "ssm-\(key).json") == key)
        let partial = DiskCache.temporaryURL(for: tensor).lastPathComponent
        #expect(SSMCompanionDiskStore.isUnpublishedTensorName(partial), "\(partial)")
        #expect(SSMCompanionDiskStore.publishedEntryKey(fromName: partial) == nil)
        for foreign in [
            "ssm-notes.txt", "ssm-notes.safetensors", "ssm-notes.json", "ssm-\(key).txt",
            "ssm-\(key.uppercased()).json", "ssm-\(key.dropLast()).json", "ssm-\(key)0.json",
            "\(key).json", "ssm-\(key).json.bak",
        ] {
            #expect(SSMCompanionDiskStore.publishedEntryKey(fromName: foreign) == nil, "\(foreign)")
            #expect(!SSMCompanionDiskStore.isUnpublishedTensorName(foreign), "\(foreign)")
        }
        for foreign in [
            "ssm-notes.partial-1A2B3C4D.safetensors", "ssm-\(key).partial-xyz.safetensors",
            "ssm-\(key).partial-1A2B3C4D.json", "\(key).partial-1A2B3C4D.safetensors",
        ] {
            #expect(!SSMCompanionDiskStore.isUnpublishedTensorName(foreign), "\(foreign)")
        }
    }

    /// `clear()` is public, and the root is a user setting. It removes what
    /// this cache wrote — indexed payloads, a payload a crash left without a
    /// row, a dead partial — and nothing else, however it is named.
    @Test func clearNeverTouchesForeignFiles() throws {
        try MLXMetalTestLock.withLock {
            let root = Self.makeRoot("clear-foreign")
            let outside = Self.makeRoot("clear-foreign-outside")
            defer {
                try? FileManager.default.removeItem(at: root)
                try? FileManager.default.removeItem(at: outside)
            }
            let modelKey = "accounting-clear-foreign"
            let coordinator = Self.coordinator(root: root, modelKey: modelKey, hybrid: false)
            let disk = try #require(coordinator.diskCache)

            let indexed = [Self.tokens(517, seed: 301), Self.tokens(1_003, seed: 302)]
            let rowless = Self.tokens(307, seed: 303)
            for tokens in indexed + [rowless] {
                coordinator.storePersistentBoundary(
                    tokens: tokens, diskArrays: Self.kv(), ssmStates: nil)
            }
            try RawDB(root: root).require(
                "DELETE FROM cache_entries WHERE hash = '\(Self.kvHash(rowless, modelKey))'")
            let rowlessURL = Self.payloadURL(root, Self.kvHash(rowless, modelKey))
            let partialURL = DiskCache.temporaryURL(for: rowlessURL)
            try Data(repeating: 0xEE, count: 40_003).write(to: partialURL)
            let genuine = Set(
                (indexed + [rowless]).map { "\(Self.kvHash($0, modelKey)).safetensors" }
                    + [partialURL.lastPathComponent])
            try #require(try Self.indexedRows(root).count == 2)

            var foreign = ForeignEntries()
            try foreign.addFile("foo.safetensors", Data(repeating: 0x11, count: 4_099), in: root)
            try foreign.addFile(
                "model-00001-of-00002.safetensors", Data(repeating: 0x22, count: 70_001), in: root)
            try foreign.addFile(
                "ABCDEF0123456789ABCDEF0123456789.safetensors", Data(repeating: 0x33, count: 1_031),
                in: root)
            try foreign.addFile(
                "0123456789abcdef0123456789abcde.safetensors", Data(repeating: 0x44, count: 1_033),
                in: root)
            try foreign.addFile(
                "0123456789abcdef0123456789abcdef0.safetensors",
                Data(repeating: 0x55, count: 1_039),
                in: root)
            try foreign.addFile(
                "random.partial-abcdefgh.safetensors", Data(repeating: 0x56, count: 1_049), in: root
            )
            try foreign.addFile(
                "0123456789abcdef0123456789abcdef.partial-xyz.safetensors",
                Data(repeating: 0x57, count: 1_051), in: root)
            try foreign.addDirectory("0123456789abcdef0123456789abcdef.safetensors", in: root)
            try foreign.addDirectory(
                "00112233445566778899aabbccddeeff.partial-1A2B3C4D.safetensors", in: root)
            try foreign.addLink(
                "fedcba9876543210fedcba9876543210.safetensors", in: root, toNewFileIn: outside)
            try foreign.addLink(
                "fedcba9876543210fedcba9876543210.partial-1A2B3C4D.safetensors", in: root,
                toNewFileIn: outside)

            let before = try Self.listing(root)
            try #require(before.isSuperset(of: genuine), "INVALID: a genuine file is not on disk")
            try #require(
                before.isSuperset(of: foreign.names), "INVALID: a foreign entry is not on disk")

            coordinator.clear()

            let removed = before.subtracting(try Self.listing(root))
            #expect(
                removed == genuine,
                "clear removed \(removed.sorted()); its own were \(genuine.sorted())")
            foreign.expectIntact(in: root, after: "clear()")
            #expect(try Self.indexedRows(root).isEmpty)
            let stats = disk.snapshotStats()
            #expect(stats.currentPayloadBytes == 0)
            #expect(stats.currentEntryCount == 0)
            for tokens in indexed { #expect(disk.fetch(tokens: tokens) == nil) }
        }
    }

    /// A root that holds `config.json` or `jang_config.json` looks like a
    /// model bundle, and nothing is removed from a LISTING of it: a payload
    /// without a row and a dead partial stay, like every shard. The payloads
    /// the index names are this cache's by construction — it wrote each one
    /// under the hash in its row, and the quota pass removes the same files
    /// in the same root — so they go, by the path built from the row. The
    /// index is emptied either way.
    @Test func clearIsSkippedForAModelBundleRoot() throws {
        try MLXMetalTestLock.withLock {
            for marker in ["config.json", "jang_config.json"] {
                let root = Self.makeRoot("clear-bundle-root")
                defer { try? FileManager.default.removeItem(at: root) }
                let modelKey = "accounting-clear-bundle-\(marker)"
                let coordinator = Self.coordinator(root: root, modelKey: modelKey, hybrid: false)
                let disk = try #require(coordinator.diskCache)
                let indexed = [Self.tokens(301, seed: 311), Self.tokens(517, seed: 312)]
                let rowless = Self.tokens(1_003, seed: 313)
                for tokens in indexed + [rowless] {
                    coordinator.storePersistentBoundary(
                        tokens: tokens, diskArrays: Self.kv(), ssmStates: nil)
                }
                try RawDB(root: root).require(
                    "DELETE FROM cache_entries WHERE hash = '\(Self.kvHash(rowless, modelKey))'")
                let rowlessURL = Self.payloadURL(root, Self.kvHash(rowless, modelKey))
                let rowlessData = try Data(contentsOf: rowlessURL)
                let partialURL = DiskCache.temporaryURL(for: rowlessURL)
                let partialData = Data(repeating: 0xEE, count: 40_003)
                try partialData.write(to: partialURL)

                var foreign = ForeignEntries()
                try foreign.addFile(marker, Data("{}".utf8), in: root)
                try foreign.addFile(
                    "model-00001-of-00002.safetensors", Data(repeating: 0x22, count: 70_001),
                    in: root)
                try foreign.addFile(
                    "model-00002-of-00002.safetensors", Self.truncatedSafetensors(), in: root)
                // A shard that happens to carry a name this cache would use.
                try foreign.addFile(
                    "0123456789abcdef0123456789abcdef.safetensors",
                    Data(repeating: 0x23, count: 8_209), in: root)

                let before = try Self.listing(root)
                let (_, log) = try Self.capturingStandardError { coordinator.clear() }

                let skipped = log.split(separator: "\n").filter {
                    $0.hasPrefix("[vmlx][cache/disk-index] clear skipped: ")
                }
                #expect(skipped.count == 1, "\(marker): \(skipped)")
                #expect(skipped.first?.contains(marker) == true)
                foreign.expectIntact(in: root, after: "clear() in a \(marker) root")
                #expect((try? Data(contentsOf: rowlessURL)) == rowlessData)
                #expect((try? Data(contentsOf: partialURL)) == partialData)
                let removed = before.subtracting(try Self.listing(root))
                #expect(
                    removed == Set(indexed.map { "\(Self.kvHash($0, modelKey)).safetensors" }),
                    "\(marker): clear removed \(removed.sorted())")
                #expect(try Self.indexedRows(root).isEmpty)
                #expect(disk.snapshotStats().currentPayloadBytes == 0)
                #expect(disk.snapshotStats().currentEntryCount == 0)
            }
        }
    }

    /// The sweep at open removes this cache's dead partials and its own
    /// payloads that are short of their declared bytes. A partial or a
    /// truncated shard under any other name is somebody else's download.
    @Test func openSweepNeverTouchesForeignPartialsOrIncompleteForeignShards() throws {
        try MLXMetalTestLock.withLock {
            let root = Self.makeRoot("open-sweep-names")
            let outside = Self.makeRoot("open-sweep-names-outside")
            defer {
                try? FileManager.default.removeItem(at: root)
                try? FileManager.default.removeItem(at: outside)
            }
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
            let truncated = Self.truncatedSafetensors()

            var foreign = ForeignEntries()
            try foreign.addFile(
                "something.partial-xyz.safetensors", Data(repeating: 0x31, count: 4_099), in: root)
            try foreign.addFile("random.partial-abcdefgh.safetensors", truncated, in: root)
            try foreign.addFile(
                "0123456789abcdef0123456789abcdef.partial-xyz.safetensors", truncated, in: root)
            try foreign.addFile("model-00001-of-00002.safetensors", truncated, in: root)
            try foreign.addFile("0123456789abcdef.safetensors", truncated, in: root)
            try foreign.addFile("ABCDEF0123456789ABCDEF0123456789.safetensors", truncated, in: root)
            try foreign.addDirectory(
                "00112233445566778899aabbccddeeff.partial-1A2B3C4D.safetensors", in: root)
            try foreign.addLink(
                "fedcba9876543210fedcba9876543210.partial-1A2B3C4D.safetensors", in: root,
                toNewFileIn: outside)
            try #require(
                !DiskCache.isCompleteSafetensors(
                    url: root.appendingPathComponent("model-00001-of-00002.safetensors")),
                "INVALID: the foreign shard is not incomplete")

            // The controls: the same bytes under names that ARE this cache's.
            let oursTruncated = root.appendingPathComponent(
                "00112233445566778899aabbccddeeff.safetensors")
            try truncated.write(to: oursTruncated)
            let oursPartial = DiskCache.temporaryURL(
                for: root.appendingPathComponent("8899aabbccddeeff0011223344556677.safetensors"))
            try Data(repeating: 0xEE, count: 40_003).write(to: oursPartial)
            try Self.age(oursPartial, by: Self.elevenMinutes)

            _ = DiskCache(cacheDir: root, maxSizeBytes: 1 << 30, modelKey: "open-sweep-names")

            #expect(
                !FileManager.default.fileExists(atPath: oursTruncated.path),
                "INVALID: the open sweep did not remove this cache's incomplete payload")
            #expect(
                !FileManager.default.fileExists(atPath: oursPartial.path),
                "INVALID: the open sweep did not remove this cache's dead partial")
            foreign.expectIntact(in: root, after: "the open sweep")
        }
    }

    /// The companion directory: `clear()`, the sweep at open and the quota
    /// listing act on regular files named `ssm-<key>.…` with a real key, and
    /// on nothing else that happens to start with `ssm-`.
    @Test func companionClearAndSweepNeverTouchForeignEntries() throws {
        try MLXMetalTestLock.withLock {
            let root = Self.makeRoot("companion-foreign")
            let outside = Self.makeRoot("companion-foreign-outside")
            defer {
                try? FileManager.default.removeItem(at: root)
                try? FileManager.default.removeItem(at: outside)
            }
            let modelKey = "accounting-companion-foreign"
            let tokens = Self.tokens(517, seed: 321)
            let key = Self.ssmKey(tokens, modelKey)
            let dir = Self.companionDir(root)
            let otherKeys = (322 ... 325).map { Self.ssmKey(Self.tokens(307, seed: $0), modelKey) }

            var foreign = ForeignEntries()
            let oursPartial: URL
            do {
                let coordinator = Self.coordinator(root: root, modelKey: modelKey)
                coordinator.storePersistentBoundary(
                    tokens: tokens, diskArrays: Self.kv(), ssmStates: Self.recurrent())
                try #require(
                    try Self.listing(dir) == ["ssm-\(key).json", "ssm-\(key).safetensors"],
                    "INVALID: the genuine companion pair is not on disk")

                try foreign.addFile("ssm-notes.txt", Data("mine".utf8), in: dir)
                try foreign.addFile(
                    "ssm-notes.safetensors", Data(repeating: 0x41, count: 4_099), in: dir)
                try foreign.addFile("ssm-notes.json", Data("{}".utf8), in: dir)
                try foreign.addFile(
                    "ssm-notes.partial-1A2B3C4D.safetensors", Data(repeating: 0x42, count: 1_031),
                    in: dir)
                try foreign.addFile(
                    "ssm-\(otherKeys[0]).partial-xyz.safetensors",
                    Data(repeating: 0x43, count: 1_033),
                    in: dir)
                try foreign.addDirectory("ssm-\(otherKeys[0]).safetensors", in: dir)
                try foreign.addDirectory("ssm-\(otherKeys[1]).json", in: dir)
                try foreign.addDirectory(
                    "ssm-\(otherKeys[1]).partial-1A2B3C4D.safetensors", in: dir)
                try foreign.addLink(
                    "ssm-\(otherKeys[2]).safetensors", in: dir, toNewFileIn: outside)
                try foreign.addLink("ssm-\(otherKeys[2]).json", in: dir, toNewFileIn: outside)
                try foreign.addLink(
                    "ssm-\(otherKeys[3]).partial-1A2B3C4D.safetensors", in: dir,
                    toNewFileIn: outside)

                oursPartial = DiskCache.temporaryURL(for: Self.companionURLs(root, key)[0])
                try Data(repeating: 0xEE, count: 40_003).write(to: oursPartial)
                try Self.age(oursPartial, by: Self.elevenMinutes)
            }

            // The sweep at open.
            CacheCoordinator.resetImportedRootsForTesting()
            let reopened = Self.coordinator(root: root, modelKey: modelKey)
            let companion = try #require(reopened.ssmStateCache.diskStore)
            #expect(
                !FileManager.default.fileExists(atPath: oursPartial.path),
                "INVALID: the open sweep did not remove this store's dead partial")
            foreign.expectIntact(in: dir, after: "the companion sweep at open")

            // What the quota would be offered for deletion.
            #expect(companion.quotaEntries().map(\.hash) == [key])

            let before = try Self.listing(dir)
            reopened.clear()
            let removed = before.subtracting(try Self.listing(dir))
            #expect(
                removed == ["ssm-\(key).json", "ssm-\(key).safetensors"],
                "clear removed \(removed.sorted())")
            foreign.expectIntact(in: dir, after: "the companion clear()")
        }
    }

    /// A store publishes by renaming over its final name. Whatever sits there
    /// that is not a regular file is not an older copy of the payload, and
    /// making room for the rename must not descend into it.
    @Test func storeRefusesToReplaceADirectoryNamedLikeItsPayload() throws {
        try MLXMetalTestLock.withLock {
            let root = Self.makeRoot("store-over-directory")
            defer { try? FileManager.default.removeItem(at: root) }
            let modelKey = "accounting-store-over-directory"
            let disk = DiskCache(cacheDir: root, maxSizeBytes: 1 << 30, modelKey: modelKey)
            let tokens = Self.tokens(517, seed: 331)
            let control = Self.tokens(1_003, seed: 332)
            let hash = Self.kvHash(tokens, modelKey)

            var foreign = ForeignEntries()
            try foreign.addDirectory("\(hash).safetensors", in: root)

            let (_, log) = try Self.capturingStandardError {
                disk.store(tokens: tokens, arrays: Self.kv(), enforceQuota: false)
            }
            foreign.expectIntact(in: root, after: "a store of the same hash")
            #expect(try Self.indexedRows(root).isEmpty, "a refused store left a row")
            #expect(disk.refusedOccupiedStores == 1)
            #expect(disk.fetch(tokens: tokens) == nil)
            #expect(
                log.split(separator: "\n").filter {
                    $0.hasPrefix("[vmlx][cache/disk-store] REFUSED ")
                        && $0.contains(hash.prefix(12))
                }.count == 1, "\(log)")
            #expect(
                try Self.listing(root).allSatisfy { !$0.contains(".partial-") },
                "a refused store left its partial behind")

            // The control: the cache still stores, and an ordinary re-store
            // over its own regular file still replaces it.
            disk.store(tokens: control, arrays: Self.kv(), enforceQuota: false)
            disk.forgetValidatedFiles()
            disk.store(tokens: control, arrays: Self.kv(2_048), enforceQuota: false)
            let row = try #require(try Self.indexedRows(root).first)
            #expect(row.hash == Self.kvHash(control, modelKey))
            #expect(row.fileSize == Self.fileBytes(Self.payloadURL(root, row.hash)))
            #expect(disk.fetch(tokens: control) != nil)
            #expect(disk.refusedOccupiedStores == 1)
        }
    }

    /// The same, for the companion tensor.
    @Test func companionStoreRefusesToReplaceADirectoryNamedLikeItsTensor() throws {
        try MLXMetalTestLock.withLock {
            let root = Self.makeRoot("companion-store-over-directory")
            defer { try? FileManager.default.removeItem(at: root) }
            let modelKey = "accounting-companion-store-over-directory"
            let coordinator = Self.coordinator(root: root, modelKey: modelKey)
            let companion = try #require(coordinator.ssmStateCache.diskStore)
            let tokens = Self.tokens(517, seed: 341)
            let key = Self.ssmKey(tokens, modelKey)
            let dir = Self.companionDir(root)

            var foreign = ForeignEntries()
            try foreign.addDirectory("ssm-\(key).safetensors", in: dir)

            #expect(throws: (any Error).self) {
                _ = try companion.store(
                    ssmStates: Self.recurrent(), tokens: tokens, boundary: tokens.count,
                    enforceQuota: false)
            }
            foreign.expectIntact(in: dir, after: "a companion store of the same key")
            #expect(
                try Self.listing(dir) == ["ssm-\(key).safetensors"],
                "a refused companion store left files behind")
        }
    }

    /// Eviction and the corrupt-entry path remove by a path built from a
    /// hash. If a directory has taken that name, it is not the cache's file.
    @Test func removeCacheFileLeavesADirectoryAlone() throws {
        let root = Self.makeRoot("remove-directory")
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        var foreign = ForeignEntries()
        try foreign.addDirectory("0123456789abcdef0123456789abcdef.safetensors", in: root)
        let ours = root.appendingPathComponent("00112233445566778899aabbccddeeff.safetensors")
        try Data([1, 2, 3]).write(to: ours)

        #expect(
            !DiskCache.removeCacheFile(
                at: root.appendingPathComponent("0123456789abcdef0123456789abcdef.safetensors")))
        foreign.expectIntact(in: root, after: "removeCacheFile")
        #expect(DiskCache.removeCacheFile(at: ours))
        #expect(!FileManager.default.fileExists(atPath: ours.path))
        #expect(DiskCache.removeCacheFile(at: ours), "a file that is not there is gone")
    }
}
