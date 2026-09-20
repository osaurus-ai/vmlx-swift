import Foundation
import MLX
import SQLite3
import Testing

@testable import MLXLMCommon

/// `cache_index.db` is data. Besides names (see `IndexRowIdentity`) it holds
/// NUMBERS, and a number an older build, another tool or corruption left
/// there must never take the process down or wedge the quota:
///
/// - `created_at = NULL` (the v1 DDL allows it) made the hand-over of a
///   dropped row's companion fail its NOT NULL constraint, which rolled the
///   whole retirement back, on every pass, without a word;
/// - `file_size = 1e19` stays REAL in the INTEGER column, reads back as
///   `Int64.max`, and one `+` later the process TRAPPED; `2^63 - 1` made the
///   SQL aggregate itself fail, and a negative size hid the other rows;
/// - a retirement that cannot be written is logged and counted, is not tried
///   again by every store, and is never started by a stats poll;
/// - under a newer build's index, opaque bytes that reach the cap evict every
///   store — which is at least said, once, and reported.
///
/// Token counts are deliberately not multiples of 64 or 256.
extension DiskCacheCompanionAccountingTests {

    @Suite(.serialized)
    struct IndexArithmetic {

        private typealias Support = DiskCacheAccountingTestSupport
        private typealias RawDB = Support.RawDB

        // MARK: - Fixtures

        private static func makeRoot(_ label: String) -> URL {
            FileManager.default.temporaryDirectory
                .appendingPathComponent("vmlx-index-arith-\(label)-\(UUID().uuidString)")
        }

        private static func tokens(_ count: Int, seed: Int) -> [Int] {
            (0 ..< count).map { seed * 100_000 + $0 }
        }

        private static func kv(_ elements: Int = 1_024) -> [String: MLXArray] {
            ["data": MLXArray.ones([elements], dtype: .float32)]
        }

        private static func recurrent(_ elements: Int = 1_024) -> [MLXArray] {
            [MLXArray.ones([elements], dtype: .float32)]
        }

        private static let cap: Int64 = 1 << 20

        /// A clock a test moves by hand.
        final class TestClock: @unchecked Sendable {
            private let lock = NSLock()
            private var value = Date()
            var now: Date {
                lock.lock()
                defer { lock.unlock() }
                return value
            }
            func advance(by seconds: TimeInterval) {
                lock.lock()
                value = value.addingTimeInterval(seconds)
                lock.unlock()
            }
        }

        private static func coordinator(
            root: URL, capBytes: Int64 = cap, modelKey: String,
            busyTimeoutMs: Int32 = DiskCache.defaultIndexBusyTimeoutMs,
            migrationBusyTimeoutMs: Int32 = DiskCacheIndexSchema.defaultBusyTimeoutMs,
            clock: TestClock? = nil
        ) -> CacheCoordinator {
            let coordinator = CacheCoordinator(
                config: CacheCoordinatorConfig(
                    usePagedCache: false,
                    enableDiskCache: true,
                    diskCacheMaxGB: Float(capBytes) / 1_073_741_824,
                    diskCacheDir: root,
                    modelKey: modelKey),
                diskIndexBusyTimeoutMs: busyTimeoutMs,
                diskIndexMigrationBusyTimeoutMs: migrationBusyTimeoutMs,
                now: { clock?.now ?? Date() })
            coordinator.setHybrid(true, requiresRecurrentSSMCompanion: true)
            return coordinator
        }

        private static func invalidValueLines(_ log: String) -> [Substring] {
            log.split(separator: "\n").filter {
                $0.hasPrefix("[vmlx][cache/disk-index] dropped row with an invalid hash")
                    || $0.hasPrefix("[vmlx][cache/disk-index] forgot companion with an invalid key")
            }
        }

        private static func retireFailureLines(_ log: String) -> [Substring] {
            log.split(separator: "\n").filter {
                $0.hasPrefix("[vmlx][cache/disk-index] could not retire")
            }
        }

        private static func opaqueCapLines(_ log: String) -> [Substring] {
            log.split(separator: "\n").filter {
                $0.hasPrefix("[vmlx][cache/disk-index] rows of a newer build hold")
            }
        }

        /// `typeof(column)` of every row that matches, so a fixture that did
        /// not store what it meant to is INVALID rather than a pass.
        private static func requireStored(
            _ raw: RawDB, column: String, as type: String, where clause: String
        ) throws {
            let shape = try raw.rows(
                "SELECT typeof(\(column)) FROM cache_entries WHERE \(clause)")
            try #require(shape == [[type]], "INVALID: \(column) was stored as \(shape)")
        }

        /// Rows whose hash is not 32 lowercase hex digits of TEXT. Counted in
        /// characters, so the answer is the same in a UTF-16 index.
        private static func hostileRowCount(_ root: URL) throws -> Int {
            try RawDB(root: root).rows(
                """
                SELECT rowid FROM cache_entries
                WHERE typeof(hash) != 'text' OR length(hash) != 32 OR hash GLOB '*[^0-9a-f]*'
                """
            ).count
        }

        private struct LegacyRecord: Equatable {
            let key: String
            let bytes: Int64
            let modifiedType: String
            let modified: Double
        }

        private static func legacyRecords(_ root: URL) throws -> [LegacyRecord] {
            try RawDB(root: root).rows(
                "SELECT key, bytes, typeof(modified), modified FROM legacy_companions ORDER BY key"
            ).map {
                LegacyRecord(
                    key: $0[0] ?? "", bytes: Int64($0[1] ?? "") ?? -1, modifiedType: $0[2] ?? "?",
                    modified: Double($0[3] ?? "") ?? -1)
            }
        }

        private static func expectHandedOver(
            _ root: URL, key: String, bytes: Int64,
            sourceLocation: SourceLocation = #_sourceLocation
        ) throws {
            let legacy = try legacyRecords(root)
            try #require(
                legacy.count == 1, "the unlinked companions are \(legacy)",
                sourceLocation: sourceLocation)
            #expect(legacy[0].key == key, sourceLocation: sourceLocation)
            #expect(legacy[0].bytes == bytes, sourceLocation: sourceLocation)
            #expect(
                legacy[0].modifiedType == "real", "modified is \(legacy[0].modifiedType)",
                sourceLocation: sourceLocation)
            #expect(
                abs(legacy[0].modified - Date().timeIntervalSince1970) < 86_400,
                "a NULL recency must become NOW, not 0 or NULL: \(legacy[0].modified)",
                sourceLocation: sourceLocation)
        }

        // MARK: - R1: `created_at = NULL`

        /// An invalid hash, a VALID companion key, `created_at = NULL`. The
        /// hand-over computed `modified = NULL` for a NOT NULL column, the
        /// statement failed, the transaction rolled back, and every record
        /// found by that read stayed — on every pass.
        @Test func aHostileRowWithANullTimestampIsRetiredAndItsCompanionHandedOver() throws {
            try MLXMetalTestLock.withLock {
                let root = Self.makeRoot("null-created-at")
                defer { try? FileManager.default.removeItem(at: root) }
                let modelKey = "index-arith-null-created-at"
                let coordinator = Self.coordinator(root: root, modelKey: modelKey)
                let disk = try #require(coordinator.diskCache)
                try #require(disk.indexHasV2Columns)

                let linked = Self.tokens(523, seed: 9_001)
                coordinator.storePersistentBoundary(
                    tokens: linked, diskArrays: Self.kv(), ssmStates: Self.recurrent())
                let key = SSMCompanionDiskStore.keyFor(
                    tokens: linked, boundary: linked.count, modelKey: modelKey)
                let companionBytes = Support.companionBytes(root, key)
                try #require(companionBytes > 0, "INVALID: no linked companion")
                let controls = [Self.tokens(301, seed: 9_002), Self.tokens(517, seed: 9_003)]
                for tokens in controls {
                    coordinator.storePersistentBoundary(
                        tokens: tokens, diskArrays: Self.kv(), ssmStates: nil)
                }
                // A second hostile row, found by the same read: it must not
                // be held back by the first.
                let raw = try RawDB(root: root)
                try raw.require(
                    """
                    INSERT INTO cache_entries (hash, token_count, file_size, created_at)
                    VALUES (x'ff', 307, \(2 * Self.cap), 2440587.5)
                    """)

                let hash = DiskCache.hashTokens(linked, modelKey: modelKey)
                try raw.require(
                    """
                    UPDATE cache_entries SET hash = NULL, created_at = NULL, file_size = \(2 * Self.cap)
                    WHERE hash = '\(hash)'
                    """)
                try #require(sqlite3_changes(raw.handle) == 1, "INVALID: no row to make hostile")
                let shape = try raw.rows(
                    """
                    SELECT typeof(hash), typeof(created_at), companion_key FROM cache_entries
                    WHERE hash IS NULL
                    """)
                try #require(shape == [["null", "null", key]], "INVALID: \(shape)")
                // Its payload has no row now; the row never named it anyway.
                try FileManager.default.removeItem(at: Support.payloadURL(root, hash))
                try #require(try Self.hostileRowCount(root) == 2)
                try #require(disk.usageBytes() > Self.cap, "INVALID: not over the cap")

                DiskCache.resetRateLimitedReportsForTesting()
                let (_, log) = try Support.capturingStandardError {
                    coordinator.enforceCombinedDiskQuota()
                }
                #expect(try Self.hostileRowCount(root) == 0, "the retirement rolled back: \(log)")
                try Self.expectHandedOver(root, key: key, bytes: companionBytes)
                #expect(Support.companionBytes(root, key) == companionBytes, "the companion went")
                try Support.expectUsageMatchesDisk(disk, root: root, atLeast: companionBytes)
                #expect(Self.invalidValueLines(log).count == 2, "\(log)")
                #expect(Self.retireFailureLines(log).isEmpty, "\(log)")
                for tokens in controls {
                    #expect(disk.fetch(tokens: tokens, touchRecency: false, countHit: false) != nil)
                }
                let stats = disk.snapshotStats()
                #expect(stats.evictions == 0)
                #expect(stats.failedIndexWrites == 0)

                // A second pass finds nothing to do.
                let legacyBefore = try Self.legacyRecords(root)
                let attempts = disk.retireAttemptsForTesting
                DiskCache.resetRateLimitedReportsForTesting()
                let (_, again) = try Support.capturingStandardError {
                    coordinator.enforceCombinedDiskQuota()
                }
                #expect(Self.invalidValueLines(again).isEmpty, "\(again)")
                #expect(disk.retireAttemptsForTesting == attempts, "a second pass wrote again")
                #expect(try Self.legacyRecords(root) == legacyBefore)
                try Support.expectUsageMatchesDisk(disk, root: root, atLeast: companionBytes)
            }
        }

        /// The same statement runs when a VALID row is dropped because its
        /// payload went missing. There the failure was silent AND the row
        /// went anyway: the companion's files stayed on disk, counted by
        /// nothing, for good.
        @Test func aValidRowWithANullTimestampKeepsItsCompanionCounted() throws {
            try MLXMetalTestLock.withLock {
                let root = Self.makeRoot("null-created-at-valid")
                defer { try? FileManager.default.removeItem(at: root) }
                let modelKey = "index-arith-null-created-at-valid"
                let coordinator = Self.coordinator(root: root, modelKey: modelKey)
                let disk = try #require(coordinator.diskCache)
                try #require(disk.indexHasV2Columns)

                let linked = Self.tokens(523, seed: 9_011)
                coordinator.storePersistentBoundary(
                    tokens: linked, diskArrays: Self.kv(), ssmStates: Self.recurrent())
                let key = SSMCompanionDiskStore.keyFor(
                    tokens: linked, boundary: linked.count, modelKey: modelKey)
                let companionBytes = Support.companionBytes(root, key)
                try #require(companionBytes > 0, "INVALID: no linked companion")

                let hash = DiskCache.hashTokens(linked, modelKey: modelKey)
                let raw = try RawDB(root: root)
                try raw.require(
                    "UPDATE cache_entries SET created_at = NULL WHERE hash = '\(hash)'")
                try #require(sqlite3_changes(raw.handle) == 1, "INVALID: no row")
                try Self.requireStored(
                    raw, column: "created_at", as: "null", where: "hash = '\(hash)'")
                try FileManager.default.removeItem(at: Support.payloadURL(root, hash))

                #expect(disk.fetch(tokens: linked) == nil)
                #expect(try Support.indexedRows(root).isEmpty, "the row of a missing payload stays")
                try Self.expectHandedOver(root, key: key, bytes: companionBytes)
                try Support.expectUsageMatchesDisk(disk, root: root, atLeast: companionBytes)
            }
        }

        /// A retirement that cannot be written: said once, counted every
        /// time, and not tried again by the very next pass.
        @Test func aRetirementThatFailsIsLoggedOnceCountedAndNotRetriedAtOnce() throws {
            try MLXMetalTestLock.withLock {
                let root = Self.makeRoot("retire-fails")
                defer { try? FileManager.default.removeItem(at: root) }
                let modelKey = "index-arith-retire-fails"
                let clock = TestClock()
                let coordinator = Self.coordinator(
                    root: root, modelKey: modelKey, busyTimeoutMs: 50, clock: clock)
                let disk = try #require(coordinator.diskCache)
                let controls = [Self.tokens(301, seed: 9_021), Self.tokens(517, seed: 9_022)]
                for tokens in controls {
                    coordinator.storePersistentBoundary(
                        tokens: tokens, diskArrays: Self.kv(), ssmStates: nil)
                }
                try RawDB(root: root).require(
                    """
                    INSERT INTO cache_entries (hash, token_count, file_size, created_at)
                    VALUES (NULL, 307, \(2 * Self.cap), 2440587.5)
                    """)
                try #require(disk.usageBytes() > Self.cap, "INVALID: not over the cap")
                try #require(disk.retireAttemptsForTesting == 0, "INVALID: already attempted")

                let blocker = try RawDB(root: root)
                try blocker.require("BEGIN IMMEDIATE")
                DiskCache.resetRateLimitedReportsForTesting()
                let (_, log) = try Support.capturingStandardError {
                    for _ in 0 ..< 3 {
                        coordinator.enforceCombinedDiskQuota()
                        clock.advance(by: DiskCache.defaultRetireRetryInterval + 1)
                    }
                }
                try #require(
                    try Self.hostileRowCount(root) == 1,
                    "INVALID: the retirement was written despite the held write lock")
                #expect(disk.retireAttemptsForTesting == 3)
                #expect(disk.snapshotStats().failedIndexWrites == 3, "a failed retire is counted")
                let lines = Self.retireFailureLines(log)
                #expect(lines.count == 1, "said once, not per pass: \(log)")
                #expect(lines.first?.contains("database is locked") == true, "\(log)")

                // Within the interval the next passes read, offer nothing
                // hostile, and open no write transaction at all.
                for _ in 0 ..< 3 { coordinator.enforceCombinedDiskQuota() }
                #expect(disk.retireAttemptsForTesting == 4, "one attempt per interval")
                #expect(disk.snapshotStats().failedIndexWrites == 4)
                #expect(disk.snapshotStats().evictions == 0, "something real paid for it")
                for tokens in controls {
                    #expect(disk.fetch(tokens: tokens, touchRecency: false, countHit: false) != nil)
                }

                try blocker.require("ROLLBACK")
                coordinator.enforceCombinedDiskQuota()
                #expect(try Self.hostileRowCount(root) == 1, "retried inside the interval")
                clock.advance(by: DiskCache.defaultRetireRetryInterval + 1)
                coordinator.enforceCombinedDiskQuota()
                #expect(try Self.hostileRowCount(root) == 0, "not retried after the interval")
                #expect(disk.usageBytes() < Self.cap)
            }
        }

        // MARK: - R2: absurd byte counts
        //
        // The first two TRAPPED. Each is on its own so that a regression
        // takes down one named test.

        /// `file_size = 1e19` on a valid row, through `DiskCache`'s own quota.
        @Test func anAbsurdFileSizeNeverTrapsTheStandaloneEviction() throws {
            try MLXMetalTestLock.withLock {
                let root = Self.makeRoot("absurd-standalone")
                defer { try? FileManager.default.removeItem(at: root) }
                let modelKey = "index-arith-absurd-standalone"
                let disk = DiskCache(
                    cacheDir: root, maxSizeBytes: Int(Self.cap), modelKey: modelKey)
                let normal = Self.tokens(301, seed: 9_031)
                let absurd = Self.tokens(517, seed: 9_032)
                let later = Self.tokens(1_003, seed: 9_033)
                disk.store(tokens: normal, arrays: Self.kv())
                Thread.sleep(forTimeInterval: 0.02)
                disk.store(tokens: absurd, arrays: Self.kv())
                let absurdHash = DiskCache.hashTokens(absurd, modelKey: modelKey)
                let raw = try RawDB(root: root)
                try raw.require(
                    "UPDATE cache_entries SET file_size = 1e19 WHERE hash = '\(absurdHash)'")
                try Self.requireStored(
                    raw, column: "file_size", as: "real", where: "hash = '\(absurdHash)'")

                // The NEWER row is the absurd one: oldest-first would take
                // the normal row before it.
                Thread.sleep(forTimeInterval: 0.02)
                disk.store(tokens: later, arrays: Self.kv())

                #expect(Support.fileBytes(Support.payloadURL(root, absurdHash)) == 0)
                #expect(
                    try raw.rows("SELECT 1 FROM cache_entries WHERE hash = '\(absurdHash)'")
                        .isEmpty, "the absurd row is an ordinary victim")
                #expect(disk.fetch(tokens: normal) != nil, "the normal row fits and must survive")
                #expect(disk.fetch(tokens: later) != nil)
                #expect(disk.snapshotStats().evictions == 1)
                try Support.expectUsageMatchesDisk(disk, root: root)
            }
        }

        /// The same through the coordinator's index pass — and with the
        /// companion bytes absurd as well, which is a second `+`.
        @Test(arguments: [false, true])
        func anAbsurdFileSizeNeverTrapsTheCoordinatorPass(companionBytesToo: Bool) throws {
            try MLXMetalTestLock.withLock {
                let root = Self.makeRoot("absurd-coordinator-\(companionBytesToo)")
                defer { try? FileManager.default.removeItem(at: root) }
                let modelKey = "index-arith-absurd-coordinator-\(companionBytesToo)"
                let coordinator = Self.coordinator(root: root, modelKey: modelKey)
                let disk = try #require(coordinator.diskCache)
                try #require(disk.indexHasV2Columns)
                let normal = Self.tokens(301, seed: 9_041)
                let absurd = Self.tokens(523, seed: 9_042)
                coordinator.storePersistentBoundary(
                    tokens: normal, diskArrays: Self.kv(), ssmStates: nil)
                Thread.sleep(forTimeInterval: 0.02)
                coordinator.storePersistentBoundary(
                    tokens: absurd, diskArrays: Self.kv(), ssmStates: Self.recurrent())
                let absurdHash = DiskCache.hashTokens(absurd, modelKey: modelKey)
                let key = SSMCompanionDiskStore.keyFor(
                    tokens: absurd, boundary: absurd.count, modelKey: modelKey)
                try #require(Support.companionBytes(root, key) > 0, "INVALID: no companion")
                let raw = try RawDB(root: root)
                try raw.require(
                    """
                    UPDATE cache_entries SET file_size = 1e19\(companionBytesToo ? ", companion_bytes = 1e19" : "")
                    WHERE hash = '\(absurdHash)'
                    """)
                try Self.requireStored(
                    raw, column: "file_size", as: "real", where: "hash = '\(absurdHash)'")
                if companionBytesToo {
                    try Self.requireStored(
                        raw, column: "companion_bytes", as: "real", where: "hash = '\(absurdHash)'")
                }
                #expect(disk.usageBytes() == .max, "usage saturates; it does not wrap or trap")
                #expect(coordinator.snapshotStats().diskStats?.currentPayloadBytes == Int.max)

                coordinator.enforceCombinedDiskQuota()

                #expect(Support.fileBytes(Support.payloadURL(root, absurdHash)) == 0)
                #expect(Support.companionBytes(root, key) == 0, "a group goes as a unit")
                #expect(disk.fetch(tokens: normal) != nil, "the normal row fits and must survive")
                let stats = disk.snapshotStats()
                #expect(stats.evictions == 1)
                #expect(stats.evictedBytes == .max)
                try Support.expectUsageMatchesDisk(disk, root: root)
            }
        }

        enum Pass: String, CaseIterable, Sendable {
            case standalone, coordinator
        }

        /// Three real rows of `size` bytes each under a cap of two and a
        /// half; `mutate` makes the MIDDLE one hostile; the pass runs.
        private static func threeRows(
            _ pass: Pass, label: String, fileSizeSQL: String, storedAs type: String
        ) throws -> (
            disk: DiskCache, coordinator: CacheCoordinator?, root: URL, tokens: [[Int]],
            hashes: [String]
        ) {
            let root = Self.makeRoot("\(label)-\(pass.rawValue)")
            let modelKey = "index-arith-\(label)-\(pass.rawValue)"
            let tokens = [
                Self.tokens(301, seed: 9_051), Self.tokens(517, seed: 9_052),
                Self.tokens(1_003, seed: 9_053),
            ]
            let hashes = tokens.map { DiskCache.hashTokens($0, modelKey: modelKey) }
            // Populate under a large cap; the pass under test gets a small one.
            do {
                let writer = DiskCache(cacheDir: root, maxSizeBytes: 1 << 30, modelKey: modelKey)
                for entry in tokens {
                    writer.store(tokens: entry, arrays: Self.kv())
                    Thread.sleep(forTimeInterval: 0.02)
                }
            }
            let size = Support.fileBytes(Support.payloadURL(root, hashes[0]))
            try #require(size > 0, "INVALID: no payload")
            let raw = try RawDB(root: root)
            try raw.require(
                "UPDATE cache_entries SET file_size = \(fileSizeSQL) WHERE hash = '\(hashes[1])'")
            try Self.requireStored(
                raw, column: "file_size", as: type, where: "hash = '\(hashes[1])'")

            let capBytes = size * 5 / 2
            switch pass {
            case .standalone:
                let disk = DiskCache(
                    cacheDir: root, maxSizeBytes: Int(capBytes), modelKey: modelKey)
                // A store is what runs the standalone pass; re-storing the
                // newest row adds no bytes.
                disk.store(tokens: tokens[2], arrays: Self.kv())
                return (disk, nil, root, tokens, hashes)
            case .coordinator:
                CacheCoordinator.resetImportedRootsForTesting()
                let coordinator = Self.coordinator(
                    root: root, capBytes: capBytes, modelKey: modelKey)
                let disk = try #require(coordinator.diskCache)
                coordinator.enforceCombinedDiskQuota()
                return (disk, coordinator, root, tokens, hashes)
            }
        }

        /// `2^63 - 1` is an INTEGER, so `SUM(file_size)` over it and one
        /// more row is an integer overflow: the aggregate FAILED, usage read
        /// as 0, and the cache neither evicted nor noticed.
        @Test(arguments: Pass.allCases)
        func aFileSizeThatOverflowsTheAggregateIsStillEvicted(_ pass: Pass) throws {
            try MLXMetalTestLock.withLock {
                let fixture = try Self.threeRows(
                    pass, label: "int64-max", fileSizeSQL: "9223372036854775807",
                    storedAs: "integer")
                defer { try? FileManager.default.removeItem(at: fixture.root) }
                let (disk, root) = (fixture.disk, fixture.root)
                #expect(
                    Support.fileBytes(Support.payloadURL(root, fixture.hashes[1])) == 0,
                    "the aggregate failed, so nothing was evicted")
                #expect(disk.fetch(tokens: fixture.tokens[0]) != nil, "it fits and must survive")
                #expect(disk.fetch(tokens: fixture.tokens[2]) != nil, "it fits and must survive")
                #expect(disk.snapshotStats().evictions == 1)
                #expect(disk.usageBytes() <= Int64(disk.maxSizeBytes))
                try Support.expectUsageMatchesDisk(disk, root: root)
            }
        }

        /// A negative size made the aggregate small enough to hide the rest.
        @Test(arguments: Pass.allCases)
        func aNegativeFileSizeIsZeroBytesAndHidesNothing(_ pass: Pass) throws {
            try MLXMetalTestLock.withLock {
                let fixture = try Self.threeRows(
                    pass, label: "negative", fileSizeSQL: "-5000000000", storedAs: "integer")
                defer { try? FileManager.default.removeItem(at: fixture.root) }
                let disk = fixture.disk
                let size = Support.fileBytes(
                    Support.payloadURL(fixture.root, fixture.hashes[2]))
                try #require(size > 0, "INVALID: the newest row is gone")
                // Counted: 0 for the negative row, `size` for the other two —
                // under the cap of two and a half, so nothing is evicted, and
                // the usage is what those two hold, not a negative number
                // clamped to 0.
                #expect(disk.usageBytes() == 2 * size)
                #expect(disk.snapshotStats().currentPayloadBytes == Int(2 * size))
                #expect(disk.snapshotStats().evictions == 0)

                // One more row: three counted rows are over the cap, and the
                // negative one must not hide that.
                let extra = Self.tokens(1_291, seed: 9_054)
                if let coordinator = fixture.coordinator {
                    coordinator.storePersistentBoundary(
                        tokens: extra, diskArrays: Self.kv(), ssmStates: nil)
                } else {
                    disk.store(tokens: extra, arrays: Self.kv())
                }
                #expect(disk.snapshotStats().evictions == 1)
                #expect(
                    Support.fileBytes(Support.payloadURL(fixture.root, fixture.hashes[0])) == 0,
                    "three counted rows over a cap of two and a half: the oldest goes")
                #expect(disk.usageBytes() <= Int64(disk.maxSizeBytes))
                #expect(disk.fetch(tokens: extra) != nil)
            }
        }

        /// A count can be TEXT or a BLOB: `'-9000000000abc'` is not a number,
        /// so the INTEGER column keeps it as it is. Read as a number it is
        /// -9 000 000 000 — by `+`, by `TOTAL`, by `sqlite3_column_int64` —
        /// while `MAX('-9000000000abc', 0)` compares a TEXT with an INTEGER
        /// and answers the TEXT. Both aggregates were nine gigabytes short,
        /// for good. Found at open, and found by a row read while running.
        @Test(arguments: [false, true])
        func aNegativeCountStoredAsTextOrBlobIsZeroBytesAndHidesNothing(
            plantedWhileRunning: Bool
        ) throws {
            try MLXMetalTestLock.withLock {
                let root = Self.makeRoot("text-blob-\(plantedWhileRunning)")
                defer { try? FileManager.default.removeItem(at: root) }
                let modelKey = "index-arith-text-blob-\(plantedWhileRunning)"
                let entries = [
                    Self.tokens(301, seed: 9_061), Self.tokens(517, seed: 9_062),
                    Self.tokens(1_003, seed: 9_063), Self.tokens(1_291, seed: 9_064),
                    Self.tokens(2_003, seed: 9_065),
                ]
                let elements = [1_009, 2_003, 3_001, 4_099, 5_003]
                let hashes = entries.map { DiskCache.hashTokens($0, modelKey: modelKey) }
                let writer = DiskCache(cacheDir: root, maxSizeBytes: 1 << 30, modelKey: modelKey)
                for (tokens, count) in zip(entries, elements) {
                    writer.store(tokens: tokens, arrays: Self.kv(count))
                }
                let sizes = hashes.map { Support.fileBytes(Support.payloadURL(root, $0)) }
                try #require(sizes.allSatisfy { $0 > 0 }, "INVALID: a payload is missing")
                try #require(writer.usageBytes() == sizes.reduce(0, +), "INVALID fixture")
                let real = sizes[0] + sizes[2] + sizes[4]
                try #require(real > 0)

                let raw = try RawDB(root: root)
                try raw.require(
                    "UPDATE cache_entries SET file_size = '-9000000000abc' WHERE hash = '\(hashes[1])'")
                try raw.require(
                    """
                    UPDATE cache_entries SET file_size = CAST('-9000000000' AS BLOB)
                    WHERE hash = '\(hashes[3])'
                    """)
                try Self.requireStored(
                    raw, column: "file_size", as: "text", where: "hash = '\(hashes[1])'")
                try Self.requireStored(
                    raw, column: "file_size", as: "blob", where: "hash = '\(hashes[3])'")

                let disk: DiskCache
                if plantedWhileRunning {
                    disk = writer
                    // Any read of the rows sees the counts for what they are.
                    #expect(
                        disk.quotaEntries(retiringInvalidRecords: false).map(\.bytes).sorted()
                            == [0, 0, sizes[0], sizes[2], sizes[4]])
                } else {
                    disk = DiskCache(cacheDir: root, maxSizeBytes: 1 << 30, modelKey: modelKey)
                }
                #expect(disk.usageBytes() == real)
                #expect(disk.snapshotStats().currentPayloadBytes == Int(real))
                #expect(disk.snapshotStats().currentEntryCount == 5, "nothing was dropped")
            }
        }

        /// The clamp is not free (a third of the aggregate, on every store
        /// and every stats poll), so the plain aggregate is used until a
        /// count that needs clamping has been SEEN. At open that is one scan
        /// (the tests above); while running it is a total that comes back
        /// negative, or any read of the rows.
        ///
        /// Until then there is a window, and it is accepted: a negative count
        /// another writer plants while this process runs, too small to turn
        /// the total negative, takes that much off the usage until the next
        /// read of the rows (an over-cap pass, a stats poll of the
        /// coordinator) or the next launch. The first assertion of each arm
        /// states what usage reads inside the window.
        @Test func aNegativeCountPlantedWhileRunningIsClampedOnceItIsSeen() throws {
            try MLXMetalTestLock.withLock {
                for (label, planted) in [("huge", -5_000_000_000), ("small", -1_009)] {
                    let root = Self.makeRoot("negative-live-\(label)")
                    defer { try? FileManager.default.removeItem(at: root) }
                    let modelKey = "index-arith-negative-live-\(label)"
                    let disk = DiskCache(cacheDir: root, maxSizeBytes: 1 << 30, modelKey: modelKey)
                    let entries = [
                        Self.tokens(301, seed: 9_055), Self.tokens(517, seed: 9_056),
                        Self.tokens(1_003, seed: 9_057),
                    ]
                    for tokens in entries { disk.store(tokens: tokens, arrays: Self.kv()) }
                    let hashes = entries.map { DiskCache.hashTokens($0, modelKey: modelKey) }
                    let size = Support.fileBytes(Support.payloadURL(root, hashes[0]))
                    try #require(size > 1_009 && disk.usageBytes() == 3 * size, "INVALID fixture")
                    try RawDB(root: root).require(
                        "UPDATE cache_entries SET file_size = \(planted) WHERE hash = '\(hashes[1])'"
                    )

                    // Before any row is read. A total that comes back
                    // negative is seen by that alone; a small one is not,
                    // and is short by what was planted.
                    #expect(
                        disk.usageBytes() == (label == "huge" ? 2 * size : 2 * size - 1_009),
                        "\(label): inside the window")
                    #expect(disk.quotaEntries().map(\.bytes).sorted() == [0, size, size])
                    #expect(disk.usageBytes() == 2 * size, "\(label): still hiding other rows")
                    #expect(disk.snapshotStats().currentPayloadBytes == Int(2 * size), "\(label)")
                }
            }
        }

        /// The same on a row that is not even offered: a hash that is not a
        /// payload hash, read by a poll that may not retire it. Its negative
        /// count is seen all the same, and stops hiding the real rows' bytes.
        @Test func aNegativeCountOnARowThatNamesNothingIsSeenToo() throws {
            try MLXMetalTestLock.withLock {
                let root = Self.makeRoot("negative-live-unnamed")
                defer { try? FileManager.default.removeItem(at: root) }
                let modelKey = "index-arith-negative-live-unnamed"
                let disk = DiskCache(cacheDir: root, maxSizeBytes: 1 << 30, modelKey: modelKey)
                let entries = [Self.tokens(301, seed: 9_058), Self.tokens(517, seed: 9_059)]
                for tokens in entries { disk.store(tokens: tokens, arrays: Self.kv(1_009)) }
                let real = disk.usageBytes()
                try #require(real > 2 * 1_009, "INVALID fixture")
                try RawDB(root: root).require(
                    """
                    INSERT INTO cache_entries (hash, token_count, file_size, created_at)
                    VALUES ('not-a-hash', 307, -1009, 2440587.5)
                    """)
                try #require(disk.usageBytes() == real - 1_009, "INVALID: not inside the window")

                let offered = disk.quotaEntries(retiringInvalidRecords: false)
                try #require(offered.count == 2, "INVALID: the unnamed row was offered")
                try #require(try Self.hostileRowCount(root) == 1, "INVALID: the row was retired")
                #expect(disk.usageBytes() == real)
                #expect(disk.snapshotStats().currentPayloadBytes == Int(real))
            }
        }

        // MARK: - R4: opaque bytes that reach the cap

        /// `user_version = 7` and opaque rows that hold the whole cap: every
        /// understood row is evicted by the pass that follows its store.
        /// That decision stands — but it is said, once, and reported.
        @Test func opaqueBytesThatReachTheCapAreSaidOnceAndReported() throws {
            try MLXMetalTestLock.withLock {
                let root = Self.makeRoot("opaque-cap")
                defer { try? FileManager.default.removeItem(at: root) }
                let modelKey = "index-arith-opaque-cap"
                _ = Self.coordinator(root: root, modelKey: modelKey)

                let raw = try RawDB(root: root)
                let futureHashes = (1 ... 3).map { String(repeating: "\($0)a", count: 32) }
                for hash in futureHashes {
                    try #require(!DiskCache.isPayloadHash(hash))
                    try raw.require(
                        """
                        INSERT INTO cache_entries (hash, token_count, file_size, created_at)
                        VALUES ('\(hash)', 307, \(Self.cap / 2), 2440587.5)
                        """)
                }
                let opaqueTotal = 3 * (Self.cap / 2)
                try raw.require("PRAGMA user_version = 7")

                CacheCoordinator.resetImportedRootsForTesting()
                DiskCache.resetRateLimitedReportsForTesting()
                let stored = [
                    Self.tokens(301, seed: 9_061), Self.tokens(517, seed: 9_062),
                    Self.tokens(1_003, seed: 9_063),
                ]
                let ((coordinator, disk), log) = try Support.capturingStandardError {
                    let coordinator = Self.coordinator(root: root, modelKey: modelKey)
                    let disk = try #require(coordinator.diskCache)
                    try #require(disk.indexIsFromANewerBuild, "INVALID: not a newer schema")
                    for tokens in stored {
                        coordinator.storePersistentBoundary(
                            tokens: tokens, diskArrays: Self.kv(), ssmStates: nil)
                        #expect(
                            disk.fetch(tokens: tokens) == nil,
                            "the decision changed: an understood row outlived a cap the opaque rows hold"
                        )
                    }
                    return (coordinator, disk)
                }
                #expect(disk.snapshotStats().evictions == stored.count)
                let lines = Self.opaqueCapLines(log)
                #expect(lines.count == 1, "exactly once across three stores: \(log)")
                #expect(lines.first?.contains("\(opaqueTotal)") == true, "\(log)")
                #expect(lines.first?.contains("leaving 0 of") == true, "\(log)")
                #expect(disk.snapshotStats().opaqueBytes == opaqueTotal)
                #expect(coordinator.snapshotStats().diskStats?.opaqueBytes == opaqueTotal)
                #expect(
                    try raw.rows("SELECT COUNT(*) FROM cache_entries") == [["3"]],
                    "an opaque row was touched")

                // The standalone pass says and reports the same.
                DiskCache.resetRateLimitedReportsForTesting()
                let standalone = DiskCache(
                    cacheDir: root, maxSizeBytes: Int(Self.cap), modelKey: modelKey)
                let (_, standaloneLog) = try Support.capturingStandardError {
                    for tokens in stored { standalone.store(tokens: tokens, arrays: Self.kv()) }
                }
                #expect(Self.opaqueCapLines(standaloneLog).count == 1, "\(standaloneLog)")
                #expect(standalone.snapshotStats().opaqueBytes == opaqueTotal)
                #expect(standalone.snapshotStats().evictions == stored.count)
            }
        }

        // MARK: - O1: a stats poll opens no write transaction

        /// On an index without the v2 columns the stats poll reads every
        /// row. It read them through the retiring reader, so with a hostile
        /// row and a held write lock every poll — every 2-3 s per window —
        /// was a write transaction that waited out the busy timeout.
        @Test func aStatsPollNeverTriesToRetire() throws {
            try MLXMetalTestLock.withLock {
                let root = Self.makeRoot("stats-poll")
                defer { try? FileManager.default.removeItem(at: root) }
                try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
                let raw = try RawDB(root: root)
                try raw.require("PRAGMA journal_mode=WAL")
                for statement in DiskCacheIndexSchema.v1Statements { try raw.require(statement) }
                try raw.require(
                    """
                    INSERT INTO cache_entries (hash, token_count, file_size, created_at)
                    VALUES (NULL, 307, 4099, 2440587.5)
                    """)

                // The write lock is held from before the open: the migration
                // cannot run, so this is a CURRENT build on a v1 index.
                let blocker = try RawDB(root: root)
                try blocker.require("BEGIN IMMEDIATE")
                let clock = TestClock()
                let coordinator = Self.coordinator(
                    root: root, modelKey: "index-arith-stats-poll", busyTimeoutMs: 50,
                    migrationBusyTimeoutMs: 50, clock: clock)
                let disk = try #require(coordinator.diskCache)
                try #require(!disk.indexHasV2Columns, "INVALID: the migration ran")
                try #require(!disk.indexIsFromANewerBuild, "INVALID: nothing would be retired")
                try #require(coordinator.ssmStateCache.diskStore != nil)

                let attempts = disk.retireAttemptsForTesting
                for _ in 0 ..< 5 {
                    let stats = try #require(coordinator.snapshotStats().diskStats)
                    #expect(stats.currentEntryCount == 1)
                    #expect(stats.currentPayloadBytes == 4_099)
                    // Not the back-off either: a poll is a read, whenever.
                    clock.advance(by: DiskCache.defaultRetireRetryInterval + 1)
                }
                #expect(
                    disk.retireAttemptsForTesting == attempts,
                    "a stats poll opened \(disk.retireAttemptsForTesting - attempts) write transaction(s)"
                )
                try blocker.require("ROLLBACK")
            }
        }

        // MARK: - O2: bytes with no key, under a newer schema

        /// `companion_key IS NULL` with `companion_bytes != 0`: the reader
        /// for the coordinator calls that link opaque under a newer schema
        /// and leaves the row alone. The standalone eviction took the row.
        @Test func bytesWithNoKeyAreAnOpaqueLinkForTheStandaloneEvictionToo() throws {
            try MLXMetalTestLock.withLock {
                let root = Self.makeRoot("null-key-bytes")
                defer { try? FileManager.default.removeItem(at: root) }
                let modelKey = "index-arith-null-key-bytes"
                let entries = [
                    Self.tokens(301, seed: 9_071), Self.tokens(517, seed: 9_072),
                    Self.tokens(1_003, seed: 9_073),
                ]
                let hashes = entries.map { DiskCache.hashTokens($0, modelKey: modelKey) }
                do {
                    let writer = DiskCache(
                        cacheDir: root, maxSizeBytes: 1 << 30, modelKey: modelKey)
                    try #require(writer.indexHasV2Columns)
                    for tokens in entries.prefix(2) {
                        writer.store(tokens: tokens, arrays: Self.kv())
                        Thread.sleep(forTimeInterval: 0.02)
                    }
                }
                let size = Support.fileBytes(Support.payloadURL(root, hashes[0]))
                try #require(size > 0)
                let raw = try RawDB(root: root)
                // The OLDEST row: first in line for an oldest-first pass.
                try raw.require(
                    "UPDATE cache_entries SET companion_bytes = 5003 WHERE hash = '\(hashes[0])'")
                try raw.require("PRAGMA user_version = 7")

                let disk = DiskCache(
                    cacheDir: root, maxSizeBytes: Int(size * 5 / 2), modelKey: modelKey)
                try #require(disk.indexIsFromANewerBuild)
                #expect(
                    !disk.quotaEntries().contains { $0.hash == hashes[0] },
                    "INVALID: the coordinator's reader offers the row")
                disk.store(tokens: entries[2], arrays: Self.kv())

                #expect(disk.snapshotStats().evictions == 1, "INVALID: the pass evicted nothing")
                #expect(
                    Support.fileBytes(Support.payloadURL(root, hashes[0])) == size,
                    "VICTIM: a row whose link this build cannot read was evicted")
                #expect(
                    try raw.rows(
                        "SELECT companion_bytes FROM cache_entries WHERE hash = '\(hashes[0])'")
                        == [["5003"]], "the opaque row was touched")
                #expect(Support.fileBytes(Support.payloadURL(root, hashes[1])) == 0)
                #expect(disk.fetch(tokens: entries[2]) != nil)
            }
        }

        // MARK: - O3: the import reports what it deleted

        /// A DELETE can fail inside the import's own transaction too (here:
        /// a trigger in the index). The summary counted the row as dropped
        /// and the log said so; neither was true.
        @Test func theImportCountsOnlyTheRowsItReallyDropped() throws {
            try MLXMetalTestLock.withLock {
                let root = Self.makeRoot("import-delete-fails")
                defer { try? FileManager.default.removeItem(at: root) }
                let modelKey = "index-arith-import-delete-fails"
                let coordinator = Self.coordinator(
                    root: root, capBytes: 1 << 30, modelKey: modelKey)
                let disk = try #require(coordinator.diskCache)
                let control = Self.tokens(517, seed: 9_081)
                coordinator.storePersistentBoundary(
                    tokens: control, diskArrays: Self.kv(), ssmStates: nil)
                let raw = try RawDB(root: root)
                for value in ["NULL", "x'ff'"] {
                    try raw.require(
                        """
                        INSERT INTO cache_entries (hash, token_count, file_size, created_at)
                        VALUES (\(value), 307, 4099, 2440587.5)
                        """)
                }
                // Only the NULL row is protected; the BLOB row goes.
                try raw.require(
                    """
                    CREATE TRIGGER keep_null_hash BEFORE DELETE ON cache_entries
                    WHEN old.hash IS NULL
                    BEGIN SELECT RAISE(ABORT, 'kept by a trigger'); END
                    """)

                DiskCache.resetRateLimitedReportsForTesting()
                let (summary, log) = try Support.capturingStandardError {
                    disk.reconcileCompanionAccounting(companions: [])
                }
                let counts = try #require(summary, "the import did not commit")
                #expect(counts.rowsDroppedForInvalidHash == 1, "\(counts)")
                #expect(Self.invalidValueLines(log).count == 1, "\(log)")
                #expect(
                    !Self.invalidValueLines(log).contains { $0.contains(" NULL ") }, "\(log)")
                let failures = Self.retireFailureLines(log)
                #expect(failures.count == 1, "\(log)")
                #expect(failures.first?.contains("kept by a trigger") == true, "\(log)")
                #expect(disk.snapshotStats().failedIndexWrites == 1)
                #expect(
                    try raw.rows("SELECT typeof(hash) FROM cache_entries ORDER BY rowid")
                        == [["text"], ["null"]])
                #expect(disk.fetch(tokens: control) != nil)
            }
        }

        // MARK: - O4: what the review listed as untested

        /// In a UTF-16 index a valid hash is stored as 64 bytes. The reader
        /// judges a value on its UTF-8 bytes — type, then
        /// `sqlite3_column_text`, then `sqlite3_column_bytes` — and must find
        /// 32 of them; judged in the index's own encoding (the stored bytes,
        /// or `sqlite3_column_bytes16`) every VALID hash is 64 bytes long and
        /// the whole cache is retired. That mutant fails here. The call ORDER
        /// cannot be pinned by behaviour: `sqlite3_column_bytes` converts to
        /// UTF-8 by itself, so both orders answer 32 (measured, not assumed).
        @Test func validRowsOfAUTF16IndexAreRecognised() throws {
            try MLXMetalTestLock.withLock {
                let root = Self.makeRoot("utf16")
                defer { try? FileManager.default.removeItem(at: root) }
                try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
                do {
                    let raw = try RawDB(root: root)
                    try raw.require("PRAGMA encoding = 'UTF-16le'")
                    for statement in DiskCacheIndexSchema.v1Statements {
                        try raw.require(statement)
                    }
                    try #require(
                        try raw.rows("PRAGMA encoding") == [["UTF-16le"]],
                        "INVALID: the index is not UTF-16")
                }
                let modelKey = "index-arith-utf16"
                let coordinator = Self.coordinator(root: root, modelKey: modelKey)
                let disk = try #require(coordinator.diskCache)
                try #require(disk.indexHasV2Columns, "INVALID: the migration did not run")
                let linked = Self.tokens(523, seed: 9_091)
                coordinator.storePersistentBoundary(
                    tokens: linked, diskArrays: Self.kv(), ssmStates: Self.recurrent())
                let plain = Self.tokens(301, seed: 9_092)
                coordinator.storePersistentBoundary(
                    tokens: plain, diskArrays: Self.kv(), ssmStates: nil)
                let key = SSMCompanionDiskStore.keyFor(
                    tokens: linked, boundary: linked.count, modelKey: modelKey)
                let raw = try RawDB(root: root)
                try #require(
                    try raw.rows("PRAGMA encoding") == [["UTF-16le"]],
                    "INVALID: the index is not UTF-16 any more")
                try #require(
                    try raw.rows("SELECT length(CAST(hash AS BLOB)) FROM cache_entries")
                        == [["64"], ["64"]], "INVALID: the hashes are not stored as UTF-16")
                for value in [
                    "NULL", "x'ff'", "'not-a-hash'", "'\(String(repeating: "z", count: 32))'",
                ] {
                    try raw.require(
                        """
                        INSERT INTO cache_entries (hash, token_count, file_size, created_at)
                        VALUES (\(value), 307, \(2 * Self.cap), 2440587.5)
                        """)
                }
                try raw.require(
                    """
                    INSERT INTO legacy_companions (key, bytes, modified)
                    VALUES ('not-a-key', \(2 * Self.cap), 0)
                    """)
                try #require(disk.usageBytes() > Self.cap, "INVALID: not over the cap")

                coordinator.enforceCombinedDiskQuota()

                let offered = disk.quotaEntries()
                #expect(
                    Set(offered.map(\.hash))
                        == Set(
                            [linked, plain].map { DiskCache.hashTokens($0, modelKey: modelKey) }),
                    "a valid UTF-16 row was not recognised")
                #expect(offered.compactMap(\.companionKey) == [key])
                #expect(try Self.hostileRowCount(root) == 0)
                #expect(try Self.legacyRecords(root).isEmpty)
                #expect(disk.snapshotStats().evictions == 0)
                #expect(disk.fetch(tokens: linked) != nil)
                #expect(disk.fetch(tokens: plain) != nil)
                try Support.expectUsageMatchesDisk(disk, root: root)
            }
        }

        // MARK: - O5: what the second review found

        /// The companion store's own cap read its records oldest first. One
        /// unlinked record that claims `Int64.max` bytes — the NEWEST one —
        /// kept the total saturated while every real companion was evicted
        /// ahead of it. A record that can never fit goes first, as in both
        /// other passes, and what is left is counted again.
        @Test func anAbsurdCompanionRecordGoesFirstAndCostsNoRealCompanion() throws {
            try MLXMetalTestLock.withLock {
                let root = Self.makeRoot("absurd-companion")
                defer { try? FileManager.default.removeItem(at: root) }
                let modelKey = "index-arith-absurd-companion"
                let ledger = DiskCache(cacheDir: root, maxSizeBytes: 1 << 30, modelKey: modelKey)
                try #require(ledger.indexHasV2Columns)
                let directory = Support.companionDir(root)
                let entries = (0 ..< 10).map { Self.tokens(301, seed: 9_071 + $0) }
                let keys = entries.map {
                    SSMCompanionDiskStore.keyFor(tokens: $0, boundary: $0.count, modelKey: modelKey)
                }

                let sizing = try SSMCompanionDiskStore(
                    cacheDir: directory, modelKey: modelKey, maxBytes: 0)
                sizing.attachLedger(ledger)
                try sizing.store(
                    ssmStates: Self.recurrent(4_099), tokens: entries[0], boundary: 301)
                let one = Support.companionBytes(root, keys[0])
                try #require(one > 4 * 4_099, "INVALID: no companion")

                // Room for the ten, and not for an eleventh.
                let store = try SSMCompanionDiskStore(
                    cacheDir: directory, modelKey: modelKey, maxBytes: Int(10 * one + one / 2))
                store.attachLedger(ledger)
                for tokens in entries.dropFirst() {
                    try store.store(
                        ssmStates: Self.recurrent(4_099), tokens: tokens, boundary: 301)
                }
                let real = keys.map { Support.companionBytes(root, $0) }
                try #require(real.allSatisfy { $0 > 0 }, "INVALID: the cap already evicted")
                try #require(ledger.companionUsageBytes() == real.reduce(0, +), "INVALID fixture")

                let absurd = SSMCompanionDiskStore.keyFor(
                    tokens: Self.tokens(307, seed: 9_099), boundary: 307, modelKey: modelKey)
                try RawDB(root: root).require(
                    """
                    INSERT INTO legacy_companions (key, bytes, modified)
                    VALUES ('\(absurd)', 9223372036854775807, \(Date().timeIntervalSince1970 + 3_600))
                    """)
                try #require(ledger.companionUsageBytes() == .max, "INVALID: not over the cap")

                // A direct write is what applies this cap; re-storing the
                // newest companion adds no bytes.
                try store.store(
                    ssmStates: Self.recurrent(4_099), tokens: entries[9], boundary: 301)

                #expect(keys.map { Support.companionBytes(root, $0) } == real)
                #expect(try Self.legacyRecords(root).map(\.key).sorted() == keys.sorted())
                #expect(ledger.companionUsageBytes() == real.reduce(0, +))
            }
        }

        /// Dropping a row hands its companion to the unlinked list first, so
        /// the companion's bytes stay counted. When that hand-over cannot be
        /// written the row stays: the index may over-count what is on disk,
        /// and never under-counts it.
        @Test func aRowWhoseCompanionCannotBeHandedOverIsNotDropped() throws {
            try MLXMetalTestLock.withLock {
                let root = Self.makeRoot("handover-fails")
                defer { try? FileManager.default.removeItem(at: root) }
                let modelKey = "index-arith-handover-fails"
                let coordinator = Self.coordinator(root: root, modelKey: modelKey)
                let disk = try #require(coordinator.diskCache)
                let tokens = Self.tokens(301, seed: 9_081)
                coordinator.storePersistentBoundary(
                    tokens: tokens, diskArrays: Self.kv(1_009), ssmStates: Self.recurrent(4_099))
                let hash = DiskCache.hashTokens(tokens, modelKey: modelKey)
                let key = SSMCompanionDiskStore.keyFor(
                    tokens: tokens, boundary: tokens.count, modelKey: modelKey)
                let companionBytes = Support.companionBytes(root, key)
                try #require(companionBytes > 0, "INVALID: no companion")
                let raw = try RawDB(root: root)
                try #require(
                    try raw.rows("SELECT companion_key FROM cache_entries WHERE hash = '\(hash)'")
                        == [[key]], "INVALID: the row carries no link")

                // The payload goes; the fetch that finds it missing drops
                // the row — handing the companion over first.
                try FileManager.default.removeItem(at: Support.payloadURL(root, hash))
                try raw.require(
                    """
                    CREATE TRIGGER refuse_handover BEFORE INSERT ON legacy_companions
                    BEGIN SELECT RAISE(ABORT, 'refused by a trigger'); END
                    """)
                try #require(disk.fetch(tokens: tokens) == nil)
                #expect(
                    try raw.rows("SELECT companion_key FROM cache_entries WHERE hash = '\(hash)'")
                        == [[key]], "the row went and took the companion's bytes with it")
                #expect(try Self.legacyRecords(root).isEmpty)
                #expect(disk.companionUsageBytes() == companionBytes)

                // Control: once the hand-over can be written, the same fetch
                // drops the row and the companion stays counted.
                try raw.require("DROP TRIGGER refuse_handover")
                try #require(disk.fetch(tokens: tokens) == nil)
                #expect(try raw.rows("SELECT 1 FROM cache_entries WHERE hash = '\(hash)'").isEmpty)
                #expect(try Self.legacyRecords(root).map(\.key) == [key])
                #expect(disk.companionUsageBytes() == companionBytes)
            }
        }

        /// The back-off is a wall-clock time. A clock that is set BACKWARDS
        /// must not stretch it: a wait longer than the interval is not one
        /// this cache asked for.
        @Test func aClockSetBackwardsDoesNotStretchTheRetireBackOff() throws {
            try MLXMetalTestLock.withLock {
                let root = Self.makeRoot("retire-clock")
                defer { try? FileManager.default.removeItem(at: root) }
                let modelKey = "index-arith-retire-clock"
                let clock = TestClock()
                let coordinator = Self.coordinator(
                    root: root, modelKey: modelKey, busyTimeoutMs: 50, clock: clock)
                let disk = try #require(coordinator.diskCache)
                coordinator.storePersistentBoundary(
                    tokens: Self.tokens(301, seed: 9_085), diskArrays: Self.kv(1_009),
                    ssmStates: nil)
                try RawDB(root: root).require(
                    """
                    INSERT INTO cache_entries (hash, token_count, file_size, created_at)
                    VALUES (NULL, 307, \(2 * Self.cap), 2440587.5)
                    """)
                try #require(disk.usageBytes() > Self.cap, "INVALID: not over the cap")

                let blocker = try RawDB(root: root)
                try blocker.require("BEGIN IMMEDIATE")
                DiskCache.resetRateLimitedReportsForTesting()
                _ = try Support.capturingStandardError { coordinator.enforceCombinedDiskQuota() }
                try blocker.require("ROLLBACK")
                try #require(disk.retireAttemptsForTesting == 1, "INVALID: no failed attempt")
                try #require(try Self.hostileRowCount(root) == 1)

                // Control: inside the interval nothing is tried.
                clock.advance(by: 11)
                coordinator.enforceCombinedDiskQuota()
                try #require(disk.retireAttemptsForTesting == 1, "INVALID: no back-off to stretch")

                // A day backwards: the back-off now ends a day and 49 s away.
                clock.advance(by: -86_411)
                coordinator.enforceCombinedDiskQuota()
                #expect(disk.retireAttemptsForTesting == 2)
                #expect(try Self.hostileRowCount(root) == 0)
                #expect(disk.usageBytes() < Self.cap)
            }
        }

        /// Why a retirement failed is SQLite's message, and a trigger in the
        /// index writes that message: it is data, and is shown bounded and
        /// escaped like every other value from the index.
        @Test func theReasonARetirementFailedIsShownBoundedAndEscaped() throws {
            try MLXMetalTestLock.withLock {
                let root = Self.makeRoot("retire-reason")
                defer { try? FileManager.default.removeItem(at: root) }
                let modelKey = "index-arith-retire-reason"
                let coordinator = Self.coordinator(root: root, modelKey: modelKey)
                let disk = try #require(coordinator.diskCache)
                coordinator.storePersistentBoundary(
                    tokens: Self.tokens(301, seed: 9_087), diskArrays: Self.kv(1_009),
                    ssmStates: nil)
                let raw = try RawDB(root: root)
                try raw.require(
                    """
                    INSERT INTO cache_entries (hash, token_count, file_size, created_at)
                    VALUES (NULL, 307, \(2 * Self.cap), 2440587.5)
                    """)
                let forged = "[vmlx][cache/forged] a line of its own"
                let message = "kept\n\(forged) " + String(repeating: "A", count: 301)
                try raw.require(
                    """
                    CREATE TRIGGER keep_null_hash BEFORE DELETE ON cache_entries
                    WHEN OLD.hash IS NULL
                    BEGIN SELECT RAISE(ABORT, '\(message)'); END
                    """)
                try #require(disk.usageBytes() > Self.cap, "INVALID: not over the cap")

                DiskCache.resetRateLimitedReportsForTesting()
                let (_, log) = try Support.capturingStandardError {
                    coordinator.enforceCombinedDiskQuota()
                }
                try #require(try Self.hostileRowCount(root) == 1, "INVALID: the trigger did not hold")
                let lines = Self.retireFailureLines(log)
                try #require(lines.count == 1, "INVALID: no retire failure was logged: \(log)")
                let line = String(lines[0])
                #expect(line.contains("kept\\n"), "the newline was not escaped: \(line)")
                #expect(!log.split(separator: "\n").contains { $0.hasPrefix(forged) }, "\(log)")
                #expect(line.contains("… (\(message.utf8.count) bytes)"), "\(line)")
                #expect(!line.contains(String(repeating: "A", count: 97)), "\(line)")
            }
        }

        /// The retirement cannot be written (a trigger refuses it) WHILE the
        /// real rows are over the cap on their own. The pass still evicts —
        /// on the totals it offers, so exactly as much as the real rows ask
        /// for and not one byte for the hostile record.
        @Test func aBlockedRetirementStillEvictsRealRowsOnOfferedTotals() throws {
            try MLXMetalTestLock.withLock {
                let root = Self.makeRoot("blocked-over-cap")
                defer { try? FileManager.default.removeItem(at: root) }
                let modelKey = "index-arith-blocked-over-cap"
                let entries = [
                    Self.tokens(301, seed: 9_101), Self.tokens(517, seed: 9_102),
                    Self.tokens(1_003, seed: 9_103),
                ]
                let hashes = entries.map { DiskCache.hashTokens($0, modelKey: modelKey) }
                do {
                    let writer = DiskCache(
                        cacheDir: root, maxSizeBytes: 1 << 30, modelKey: modelKey)
                    for tokens in entries {
                        writer.store(tokens: tokens, arrays: Self.kv())
                        Thread.sleep(forTimeInterval: 0.02)
                    }
                }
                let size = Support.fileBytes(Support.payloadURL(root, hashes[0]))
                try #require(size > 0)
                let raw = try RawDB(root: root)
                try raw.require(
                    """
                    INSERT INTO cache_entries (hash, token_count, file_size, created_at)
                    VALUES (NULL, 307, \(1 << 40), 2440587.5)
                    """)
                try raw.require(
                    """
                    CREATE TRIGGER keep_null_hash BEFORE DELETE ON cache_entries
                    WHEN old.hash IS NULL
                    BEGIN SELECT RAISE(ABORT, 'kept by a trigger'); END
                    """)

                CacheCoordinator.resetImportedRootsForTesting()
                DiskCache.resetRateLimitedReportsForTesting()
                let clock = TestClock()
                let ((coordinator, disk), log) = try Support.capturingStandardError {
                    let coordinator = Self.coordinator(
                        root: root, capBytes: size * 5 / 2, modelKey: modelKey, clock: clock)
                    return (coordinator, try #require(coordinator.diskCache))
                }
                try #require(try Self.hostileRowCount(root) == 1, "INVALID: it was retired")
                #expect(Self.retireFailureLines(log).count == 1, "\(log)")
                #expect(Self.retireFailureLines(log).first?.contains("kept by a trigger") == true)
                #expect(Self.invalidValueLines(log).isEmpty, "reported as dropped: \(log)")
                let stats = disk.snapshotStats()
                #expect(stats.evictions == 1, "three real rows over a cap of two and a half")
                #expect(stats.evictedBytes == size, "hostile bytes were charged to a real row")
                #expect(stats.failedIndexWrites >= 1)
                #expect(Support.fileBytes(Support.payloadURL(root, hashes[0])) == 0)
                #expect(disk.fetch(tokens: entries[1]) != nil)
                #expect(disk.fetch(tokens: entries[2]) != nil)

                try raw.require("DROP TRIGGER keep_null_hash")
                clock.advance(by: DiskCache.defaultRetireRetryInterval + 1)
                coordinator.enforceCombinedDiskQuota()
                #expect(try Self.hostileRowCount(root) == 0)
                #expect(disk.snapshotStats().evictions == 1)
                try Support.expectUsageMatchesDisk(disk, root: root)
            }
        }
    }
}
