import Foundation
import MLX
import SQLite3
import Testing

@testable import MLXLMCommon

/// A value read from `cache_index.db` is judged on its BYTES and its storage
/// class, and the record that carries it is addressed by `rowid` — never by
/// the value. `String(cString:)` stops at the first NUL and repairs invalid
/// UTF-8, so a statement that binds what it read back matches nothing for a
/// TEXT with an embedded NUL, for a BLOB and for invalid UTF-8, and a NULL
/// hash has no value to bind at all. Such a row could never be dropped: its
/// bytes stayed counted, and every pass evicted every real entry to make up
/// for them. A `<valid hash>\0junk` row is worse: it reads back as the valid
/// hash, so the real entry of that name pays for it.
///
/// Under an index a NEWER build has claimed the same rows are opaque: counted,
/// never offered for eviction, never dropped, never touched.
///
/// Token counts are deliberately not multiples of 64 or 256.
extension DiskCacheCompanionAccountingTests {

    @Suite(.serialized)
    struct IndexRowIdentity {

        private typealias Support = DiskCacheAccountingTestSupport
        private typealias RawDB = Support.RawDB

        // MARK: - Fixtures

        private static func makeRoot(_ label: String) -> URL {
            FileManager.default.temporaryDirectory
                .appendingPathComponent("vmlx-index-rowid-\(label)-\(UUID().uuidString)")
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

        private static func coordinator(
            root: URL, capBytes: Int64 = cap, modelKey: String
        ) -> CacheCoordinator {
            let coordinator = CacheCoordinator(
                config: CacheCoordinatorConfig(
                    usePagedCache: false,
                    enableDiskCache: true,
                    diskCacheMaxGB: Float(capBytes) / 1_073_741_824,
                    diskCacheDir: root,
                    modelKey: modelKey))
            coordinator.setHybrid(true, requiresRecurrentSSMCompanion: true)
            return coordinator
        }

        private static func hex(_ text: some StringProtocol) -> String {
            text.utf8.map { String(format: "%02x", $0) }.joined()
        }

        /// One hostile column value, as the SQL expression that stores it,
        /// with the storage class and byte length SQLite must report for it —
        /// a fixture that did not store what it meant to is INVALID, not a
        /// pass.
        struct HostileValue: Sendable, CustomStringConvertible {
            let label: String
            let sql: String
            let storedType: String
            let storedBytes: Int?
            var description: String { label }
        }

        /// Every kind, built on `valid`: a value this cache really computed
        /// (32 or 64 lowercase hex digits) and that names a REAL entry.
        private static func hostileValues(basedOn valid: String) -> [HostileValue] {
            let count = valid.utf8.count
            return [
                HostileValue(
                    label: "embedded NUL", sql: "CAST(x'7a7a0078' AS TEXT)",
                    storedType: "text", storedBytes: 4),
                HostileValue(
                    label: "a valid value, NUL, junk",
                    sql: "CAST(x'\(hex(valid))006a756e6b' AS TEXT)",
                    storedType: "text", storedBytes: count + 5),
                HostileValue(
                    label: "one-byte BLOB", sql: "x'ff'", storedType: "blob", storedBytes: 1),
                HostileValue(
                    label: "the valid value as a BLOB", sql: "x'\(hex(valid))'",
                    storedType: "blob", storedBytes: count),
                HostileValue(label: "NULL", sql: "NULL", storedType: "null", storedBytes: nil),
                HostileValue(
                    label: "invalid UTF-8", sql: "CAST(x'\(hex(valid.dropLast()))ff' AS TEXT)",
                    storedType: "text", storedBytes: count),
                HostileValue(
                    label: "trailing newline", sql: "'\(valid)' || char(10)",
                    storedType: "text", storedBytes: count + 1),
                HostileValue(
                    label: "trailing space", sql: "'\(valid) '",
                    storedType: "text", storedBytes: count + 1),
                HostileValue(
                    label: "one digit short", sql: "'\(valid.dropLast())'",
                    storedType: "text", storedBytes: count - 1),
                HostileValue(
                    label: "one digit long", sql: "'\(valid)0'",
                    storedType: "text", storedBytes: count + 1),
                // U+0660 is two bytes: the BYTE count is right, the digits are not.
                HostileValue(
                    label: "non-ASCII digit", sql: "'\(valid.dropLast(2))\u{0660}'",
                    storedType: "text", storedBytes: count),
            ]
        }

        /// The kinds a drop that binds the value it read back can never match.
        private static let undroppableLabels: Set<String> = [
            "embedded NUL", "one-byte BLOB", "NULL", "invalid UTF-8",
        ]

        /// `(rowid, storage class, bytes as hex)` of every record: identity
        /// that does not go through a C string.
        struct Record: Hashable, CustomStringConvertible {
            let rowid: Int64
            let type: String
            let bytesHex: String?
            var description: String { "#\(rowid) \(type) \(bytesHex ?? "NULL")" }
        }

        private static func records(_ root: URL, table: String, column: String) throws -> Set<
            Record
        > {
            Set(
                try RawDB(root: root).rows(
                    "SELECT rowid, typeof(\(column)), hex(CAST(\(column) AS BLOB)) FROM \(table)"
                ).map { row in
                    Record(
                        rowid: Int64(row[0] ?? "") ?? -1, type: row[1] ?? "?",
                        bytesHex: row[1] == "null" ? nil : row[2])
                })
        }

        private static func requireStored(
            _ raw: RawDB, _ value: HostileValue, table: String, column: String, rowid: Int64
        ) throws {
            let shape = try raw.rows(
                "SELECT typeof(\(column)), length(CAST(\(column) AS BLOB)) FROM \(table) WHERE rowid = \(rowid)"
            )
            try #require(
                shape == [[value.storedType, value.storedBytes.map(String.init)]],
                "INVALID: \(value.label) was stored as \(shape)")
        }

        /// Hostile `cache_entries` rows, the oldest in the index and each
        /// larger than the whole cap. Returns their rowids.
        private static func plantHostileHashRows(
            root: URL, values: [HostileValue]
        ) throws -> Set<Int64> {
            let raw = try RawDB(root: root)
            var rowids = Set<Int64>()
            for value in values {
                try raw.require(
                    """
                    INSERT INTO cache_entries (hash, token_count, file_size, created_at)
                    VALUES (\(value.sql), 307, \(2 * cap), 2440587.5)
                    """)
                let rowid = sqlite3_last_insert_rowid(raw.handle)
                try requireStored(
                    raw, value, table: "cache_entries", column: "hash", rowid: rowid)
                rowids.insert(rowid)
            }
            try #require(rowids.count == values.count, "INVALID: a hostile row was not planted")
            return rowids
        }

        /// The real entries of a root: their records, and their files byte
        /// for byte.
        private struct Real {
            var records: Set<Record>
            var legacy: Set<Record>
            var files: [URL: Data] = [:]

            init(root: URL) throws {
                records = try IndexRowIdentity.records(
                    root, table: "cache_entries", column: "hash")
                legacy = try IndexRowIdentity.records(
                    root, table: "legacy_companions", column: "key")
                for dir in [root, Support.companionDir(root)] {
                    let names =
                        (try? FileManager.default.contentsOfDirectory(atPath: dir.path)) ?? []
                    for name in names
                    where name.hasSuffix(".safetensors") || name.hasSuffix(".json") {
                        let url = dir.appendingPathComponent(name)
                        files[url] = try Data(contentsOf: url)
                    }
                }
                try #require(!records.isEmpty, "INVALID: no real row")
                try #require(!files.isEmpty, "INVALID: no real file")
            }

            var bytesOnDisk: Int64 { files.values.reduce(0) { $0 + Int64($1.count) } }
        }

        private static func invalidValueLines(_ log: String) -> [Substring] {
            log.split(separator: "\n").filter {
                $0.hasPrefix("[vmlx][cache/disk-index] dropped row with an invalid hash")
                    || $0.hasPrefix("[vmlx][cache/disk-index] forgot companion with an invalid key")
            }
        }

        /// Exactly the hostile records are gone; every real record and file
        /// is as it was; nothing hostile is counted; nothing was "evicted".
        private static func expectOnlyHostileRecordsGone(
            disk: DiskCache, root: URL, real: Real, controls: [[Int]], after what: String,
            sourceLocation: SourceLocation = #_sourceLocation
        ) throws {
            let rows = try records(root, table: "cache_entries", column: "hash")
            #expect(
                rows == real.records,
                "cache_entries after \(what): unexpected \(rows.subtracting(real.records).sorted { $0.rowid < $1.rowid }), missing \(real.records.subtracting(rows).sorted { $0.rowid < $1.rowid })",
                sourceLocation: sourceLocation)
            let legacy = try records(root, table: "legacy_companions", column: "key")
            #expect(
                legacy == real.legacy,
                "legacy_companions after \(what): unexpected \(legacy.subtracting(real.legacy).sorted { $0.rowid < $1.rowid }), missing \(real.legacy.subtracting(legacy).sorted { $0.rowid < $1.rowid })",
                sourceLocation: sourceLocation)
            let links = try RawDB(root: root).rows(
                """
                SELECT rowid FROM cache_entries
                WHERE companion_bytes != 0 AND (typeof(companion_key) != 'text'
                      OR length(CAST(companion_key AS BLOB)) != 64)
                """)
            #expect(
                links.isEmpty, "hostile links after \(what): \(links)",
                sourceLocation: sourceLocation)
            for (url, data) in real.files.sorted(by: { $0.key.path < $1.key.path }) {
                #expect(
                    (try? Data(contentsOf: url)) == data,
                    "REAL FILE \(url.lastPathComponent) did not survive \(what) byte for byte",
                    sourceLocation: sourceLocation)
            }
            #expect(
                disk.usageBytes() == real.bytesOnDisk,
                "usage after \(what) is \(disk.usageBytes()), the real files hold \(real.bytesOnDisk)",
                sourceLocation: sourceLocation)
            let stats = disk.snapshotStats()
            #expect(
                stats.evictions == 0, "a hostile record was counted as an eviction by \(what)",
                sourceLocation: sourceLocation)
            #expect(
                stats.evictedBytes == 0, "hostile bytes were reported as reclaimed by \(what)",
                sourceLocation: sourceLocation)
            for tokens in controls {
                #expect(
                    disk.fetch(tokens: tokens, touchRecency: false, countHit: false) != nil,
                    "a real entry no longer fetches after \(what)",
                    sourceLocation: sourceLocation)
            }
        }

        // MARK: - R1: hostile `cache_entries.hash`

        /// The coordinator's index quota pass, twice.
        @Test func unbindableHashRowsAreDroppedByRowidInTheCoordinatorQuotaPass() throws {
            try MLXMetalTestLock.withLock {
                let root = Self.makeRoot("hash-quota")
                defer { try? FileManager.default.removeItem(at: root) }
                let modelKey = "index-rowid-hash-quota"
                let coordinator = Self.coordinator(root: root, modelKey: modelKey)
                let disk = try #require(coordinator.diskCache)
                try #require(disk.indexHasV2Columns)
                let controls = [Self.tokens(301, seed: 8_001), Self.tokens(517, seed: 8_002)]
                for tokens in controls {
                    coordinator.storePersistentBoundary(
                        tokens: tokens, diskArrays: Self.kv(), ssmStates: nil)
                }
                let real = try Real(root: root)
                let values = Self.hostileValues(
                    basedOn: DiskCache.hashTokens(controls[0], modelKey: modelKey))
                let hostile = try Self.plantHostileHashRows(root: root, values: values)
                try #require(
                    disk.usageBytes() > Self.cap, "INVALID: the fixture is not over the cap")

                DiskCache.resetRateLimitedReportsForTesting()
                let (_, log) = try Support.capturingStandardError {
                    coordinator.enforceCombinedDiskQuota()
                }
                try Self.expectOnlyHostileRecordsGone(
                    disk: disk, root: root, real: real, controls: controls,
                    after: "the coordinator's index quota pass")
                #expect(!Self.invalidValueLines(log).isEmpty, "\(log)")
                let left = try Self.records(root, table: "cache_entries", column: "hash")
                #expect(Set(left.map(\.rowid)).isDisjoint(with: hostile))

                // A second pass finds nothing to do.
                DiskCache.resetRateLimitedReportsForTesting()
                let (_, again) = try Support.capturingStandardError {
                    coordinator.enforceCombinedDiskQuota()
                }
                try Self.expectOnlyHostileRecordsGone(
                    disk: disk, root: root, real: real, controls: controls,
                    after: "a second quota pass")
                #expect(Self.invalidValueLines(again).isEmpty, "\(again)")
            }
        }

        /// `DiskCache`'s own quota on a direct store: `_evictIfNeededLocked`.
        @Test func unbindableHashRowsAreDroppedByRowidInTheStandaloneEviction() throws {
            try MLXMetalTestLock.withLock {
                let root = Self.makeRoot("hash-standalone")
                defer { try? FileManager.default.removeItem(at: root) }
                let modelKey = "index-rowid-hash-standalone"
                let disk = DiskCache(
                    cacheDir: root, maxSizeBytes: Int(Self.cap), modelKey: modelKey)
                try #require(disk.indexHasV2Columns)
                let controls = [
                    Self.tokens(301, seed: 8_011), Self.tokens(517, seed: 8_012),
                    Self.tokens(1_003, seed: 8_013),
                ]
                disk.store(tokens: controls[0], arrays: Self.kv())
                let values = Self.hostileValues(
                    basedOn: DiskCache.hashTokens(controls[0], modelKey: modelKey))
                let hostile = try Self.plantHostileHashRows(root: root, values: values)

                // The store whose standalone pass finds the index over its cap.
                disk.store(tokens: controls[1], arrays: Self.kv())
                // Read after the pass, so what it must hold is pinned here:
                // the two real rows (the second took a rowid above every
                // hostile one) and their two payloads.
                var real = try Real(root: root)
                real.records = real.records.filter { !hostile.contains($0.rowid) }
                #expect(
                    real.files.count == 2,
                    "a real payload was deleted to pay for a hostile row: \(real.files.keys.map(\.lastPathComponent))"
                )
                try Self.expectOnlyHostileRecordsGone(
                    disk: disk, root: root, real: real,
                    controls: Array(controls.prefix(2)), after: "DiskCache's standalone eviction")
                #expect(real.records.count == 2, "\(real.records)")

                // The next store's pass is an ordinary one.
                disk.store(tokens: controls[2], arrays: Self.kv())
                real = try Real(root: root)
                real.records = real.records.filter { !hostile.contains($0.rowid) }
                #expect(real.files.count == 3)
                #expect(real.records.count == 3, "\(real.records)")
                try Self.expectOnlyHostileRecordsGone(
                    disk: disk, root: root, real: real, controls: controls,
                    after: "a second standalone pass")
            }
        }

        /// The import: a hostile row is dropped once, and the import is then
        /// idempotent — it used to find `<valid hash>\0junk` "without a
        /// payload" on every run.
        @Test func unbindableHashRowsAreDroppedByRowidInTheImport() throws {
            try MLXMetalTestLock.withLock {
                let root = Self.makeRoot("hash-import")
                defer { try? FileManager.default.removeItem(at: root) }
                let modelKey = "index-rowid-hash-import"
                let coordinator = Self.coordinator(
                    root: root, capBytes: 1 << 30, modelKey: modelKey)
                let disk = try #require(coordinator.diskCache)
                let controls = [Self.tokens(301, seed: 8_021), Self.tokens(1_003, seed: 8_022)]
                for tokens in controls {
                    coordinator.storePersistentBoundary(
                        tokens: tokens, diskArrays: Self.kv(), ssmStates: nil)
                }
                let real = try Real(root: root)
                // A hash with NO payload, so the prefix kind is not rescued
                // by a file that happens to exist.
                let absent = DiskCache.hashTokens(Self.tokens(307, seed: 8_029), modelKey: modelKey)
                let values =
                    Self.hostileValues(
                        basedOn: DiskCache.hashTokens(controls[0], modelKey: modelKey))
                    + Self.hostileValues(basedOn: absent).filter {
                        $0.label == "a valid value, NUL, junk"
                            || $0.label == "the valid value as a BLOB"
                    }
                _ = try Self.plantHostileHashRows(root: root, values: values)

                let first = try #require(
                    disk.reconcileCompanionAccounting(
                        companions: coordinator.ssmStateCache.diskStore?.quotaEntries() ?? []))
                #expect(first.rowsDroppedForInvalidHash == values.count, "\(first)")
                #expect(first.rowsDeletedForMissingPayload == 0, "\(first)")
                try Self.expectOnlyHostileRecordsGone(
                    disk: disk, root: root, real: real, controls: controls, after: "the import")

                let again = try #require(disk.reconcileCompanionAccounting(companions: []))
                #expect(!again.changedAnything, "the import is not idempotent: \(again)")
            }
        }

        /// The cache-availability control. One row that cannot be dropped
        /// used to hold the index over its cap for good, and every store then
        /// evicted every real entry.
        @Test(arguments: ["embedded NUL", "one-byte BLOB", "NULL", "invalid UTF-8"])
        func oneHostileRowDoesNotEvictTheRealEntries(_ label: String) throws {
            try MLXMetalTestLock.withLock {
                try #require(Self.undroppableLabels.contains(label))
                for standalone in [true, false] {
                    let root = Self.makeRoot(
                        "available-\(standalone ? "standalone" : "coordinator")")
                    defer { try? FileManager.default.removeItem(at: root) }
                    let modelKey = "index-rowid-available-\(standalone)"
                    let coordinator =
                        standalone ? nil : Self.coordinator(root: root, modelKey: modelKey)
                    let disk: DiskCache
                    if let coordinator {
                        disk = try #require(coordinator.diskCache)
                    } else {
                        disk = DiskCache(
                            cacheDir: root, maxSizeBytes: Int(Self.cap), modelKey: modelKey)
                    }
                    func store(_ tokens: [Int]) {
                        if let coordinator {
                            coordinator.storePersistentBoundary(
                                tokens: tokens, diskArrays: Self.kv(), ssmStates: nil)
                        } else {
                            disk.store(tokens: tokens, arrays: Self.kv())
                        }
                    }
                    let controls = [Self.tokens(301, seed: 8_031), Self.tokens(517, seed: 8_032)]
                    controls.forEach(store)
                    let value = try #require(
                        Self.hostileValues(
                            basedOn: DiskCache.hashTokens(controls[0], modelKey: modelKey)
                        ).first { $0.label == label })
                    _ = try Self.plantHostileHashRows(root: root, values: [value])

                    let later = [Self.tokens(1_003, seed: 8_033), Self.tokens(1_291, seed: 8_034)]
                    later.forEach(store)

                    for tokens in controls + later {
                        #expect(
                            disk.fetch(tokens: tokens) != nil,
                            "\(label), \(standalone ? "standalone" : "coordinator"): a real entry was evicted to pay for a row that names nothing"
                        )
                    }
                    #expect(disk.snapshotStats().evictions == 0)
                    #expect(disk.usageBytes() < Self.cap, "the hostile bytes are still counted")
                }
            }
        }

        /// The retirement is a write, and a write can lose to another
        /// connection. A record that names nothing is then still there and
        /// still counted — and still nothing real pays for it: it is not
        /// offered, and the pass works from what it offers.
        @Test func aRetirementThatCannotBeWrittenStillEvictsNothingReal() throws {
            try MLXMetalTestLock.withLock {
                let root = Self.makeRoot("busy")
                defer { try? FileManager.default.removeItem(at: root) }
                let modelKey = "index-rowid-busy"
                let clock = IndexArithmetic.TestClock()
                let coordinator = CacheCoordinator(
                    config: CacheCoordinatorConfig(
                        usePagedCache: false, enableDiskCache: true,
                        diskCacheMaxGB: Float(Self.cap) / 1_073_741_824, diskCacheDir: root,
                        modelKey: modelKey),
                    diskIndexBusyTimeoutMs: 50, now: { clock.now })
                coordinator.setHybrid(true, requiresRecurrentSSMCompanion: true)
                let disk = try #require(coordinator.diskCache)
                let controls = [Self.tokens(301, seed: 8_041), Self.tokens(517, seed: 8_042)]
                for tokens in controls {
                    coordinator.storePersistentBoundary(
                        tokens: tokens, diskArrays: Self.kv(), ssmStates: nil)
                }
                let real = try Real(root: root)
                let values = Self.hostileValues(
                    basedOn: DiskCache.hashTokens(controls[0], modelKey: modelKey))
                let hostile = try Self.plantHostileHashRows(root: root, values: values)

                let blocker = try RawDB(root: root)
                try blocker.require("BEGIN IMMEDIATE")
                coordinator.enforceCombinedDiskQuota()
                let during = try Self.records(root, table: "cache_entries", column: "hash")
                try #require(
                    Set(during.map(\.rowid)).isSuperset(of: hostile),
                    "INVALID: the retirement was written despite the held write lock")
                #expect(during.isSuperset(of: real.records), "a real row was lost")
                #expect(disk.snapshotStats().evictions == 0)
                for (url, data) in real.files {
                    #expect(
                        (try? Data(contentsOf: url)) == data,
                        "REAL FILE \(url.lastPathComponent) paid for a row that names nothing")
                }
                try blocker.require("ROLLBACK")

                // A failed retirement is not tried again straight away (see
                // `IndexArithmetic`); this pass is the first one that is due.
                clock.advance(by: DiskCache.defaultRetireRetryInterval + 1)
                coordinator.enforceCombinedDiskQuota()
                try Self.expectOnlyHostileRecordsGone(
                    disk: disk, root: root, real: real, controls: controls,
                    after: "the pass that could take the write lock")
            }
        }

        // MARK: - R1: hostile `companion_key` / `legacy_companions.key`

        enum Placement: String, CaseIterable, Sendable {
            case linked, legacy
        }

        enum Pass: String, CaseIterable, Sendable {
            case coordinatorQuota, companionOverCap, importer
        }

        private struct CompanionFixture {
            let coordinator: CacheCoordinator
            let disk: DiskCache
            let controls: [[Int]]
            let real: Real
            let hostileLegacy: Set<Int64>
        }

        /// A real linked companion, a real unlinked one, and the hostile
        /// kinds built on the LINKED one's key, each claiming twice the cap.
        private static func companionFixture(
            root: URL, placement: Placement, capBytes: Int64, modelKey: String
        ) throws -> CompanionFixture {
            let coordinator = Self.coordinator(root: root, capBytes: capBytes, modelKey: modelKey)
            let disk = try #require(coordinator.diskCache)
            try #require(disk.indexHasV2Columns)
            let store = try #require(coordinator.ssmStateCache.diskStore)

            let linked = Self.tokens(523, seed: 8_100)
            coordinator.storePersistentBoundary(
                tokens: linked, diskArrays: Self.kv(), ssmStates: Self.recurrent())
            let key = SSMCompanionDiskStore.keyFor(
                tokens: linked, boundary: linked.count, modelKey: modelKey)
            try #require(Support.companionBytes(root, key) > 0, "INVALID: no linked companion")
            let values = Self.hostileValues(basedOn: key)

            // One plain row per hostile value, to carry it as a link.
            var controls = [linked]
            for index in values.indices {
                let tokens = Self.tokens(301 + index, seed: 8_101 + index)
                coordinator.storePersistentBoundary(
                    tokens: tokens, diskArrays: Self.kv(), ssmStates: nil)
                controls.append(tokens)
            }
            let unlinked = Self.tokens(1_003, seed: 8_190)
            try store.store(ssmStates: Self.recurrent(), tokens: unlinked, boundary: unlinked.count)
            let real = try Real(root: root)
            try #require(real.legacy.count == 1, "INVALID: no real unlinked companion")

            let raw = try RawDB(root: root)
            var hostileLegacy = Set<Int64>()
            for (value, tokens) in zip(values, controls.dropFirst()) {
                switch placement {
                case .linked:
                    let hash = DiskCache.hashTokens(tokens, modelKey: modelKey)
                    try raw.require(
                        """
                        UPDATE cache_entries SET companion_key = \(value.sql), companion_bytes = \(2 * cap)
                        WHERE hash = '\(hash)'
                        """)
                    try #require(sqlite3_changes(raw.handle) == 1, "INVALID: no row to link")
                    let rowid = try #require(
                        try raw.rows("SELECT rowid FROM cache_entries WHERE hash = '\(hash)'")
                            .first?[0].flatMap { Int64($0) })
                    try requireStored(
                        raw, value, table: "cache_entries", column: "companion_key", rowid: rowid)
                case .legacy:
                    try raw.require(
                        """
                        INSERT INTO legacy_companions (key, bytes, modified)
                        VALUES (\(value.sql), \(2 * cap), 0)
                        """)
                    let rowid = sqlite3_last_insert_rowid(raw.handle)
                    try requireStored(
                        raw, value, table: "legacy_companions", column: "key", rowid: rowid)
                    hostileLegacy.insert(rowid)
                }
            }
            return CompanionFixture(
                coordinator: coordinator, disk: disk, controls: controls, real: real,
                hostileLegacy: hostileLegacy)
        }

        @Test(arguments: Placement.allCases, Pass.allCases)
        func unbindableCompanionKeysAreForgottenByRowid(_ placement: Placement, _ pass: Pass) throws
        {
            try MLXMetalTestLock.withLock {
                let root = Self.makeRoot("key-\(placement.rawValue)-\(pass.rawValue)")
                defer { try? FileManager.default.removeItem(at: root) }
                let fixture = try Self.companionFixture(
                    root: root, placement: placement,
                    capBytes: pass == .importer ? 1 << 30 : Self.cap,
                    modelKey: "index-rowid-key-\(placement.rawValue)-\(pass.rawValue)")
                let disk = fixture.disk
                if pass != .importer {
                    try #require(
                        disk.companionUsageBytes() > Self.cap, "INVALID: not over the cap")
                }
                var real = fixture.real
                var directWrites = 0

                /// After a direct write: its two files and its unlinked
                /// record are real too. Pinned by count, so a hostile record
                /// that survived cannot slip in as "real".
                func admitDirectWrite() throws {
                    let dir = Support.companionDir(root)
                    var added = 0
                    for name in try FileManager.default.contentsOfDirectory(atPath: dir.path) {
                        let url = dir.appendingPathComponent(name)
                        guard real.files[url] == nil else { continue }
                        real.files[url] = try Data(contentsOf: url)
                        added += 1
                    }
                    try #require(added == 2, "INVALID: the direct write left \(added) new file(s)")
                    real.legacy = try Self.records(root, table: "legacy_companions", column: "key")
                        .filter { !fixture.hostileLegacy.contains($0.rowid) && $0.type == "text" }
                    try #require(
                        real.legacy.count == 1 + directWrites
                            && real.legacy.allSatisfy { $0.bytesHex?.count == 128 },
                        "the real unlinked companions are \(real.legacy)")
                }

                func run() throws {
                    switch pass {
                    case .coordinatorQuota:
                        fixture.coordinator.enforceCombinedDiskQuota()
                    case .companionOverCap:
                        // A direct companion write applies the store's own cap
                        // from the index.
                        let store = try #require(fixture.coordinator.ssmStateCache.diskStore)
                        directWrites += 1
                        let direct = Self.tokens(1_291, seed: 8_190 + directWrites)
                        try store.store(
                            ssmStates: Self.recurrent(), tokens: direct, boundary: direct.count)
                    case .importer:
                        #expect(
                            fixture.coordinator.reconcileDiskAccounting(),
                            "the import did not commit")
                    }
                }

                DiskCache.resetRateLimitedReportsForTesting()
                let (_, log) = try Support.capturingStandardError { try run() }
                if pass == .companionOverCap { try admitDirectWrite() }
                try Self.expectOnlyHostileRecordsGone(
                    disk: disk, root: root, real: real, controls: fixture.controls,
                    after: "\(pass.rawValue) (\(placement.rawValue))")
                #expect(!Self.invalidValueLines(log).isEmpty, "\(log)")
                let legacyLeft = try Self.records(root, table: "legacy_companions", column: "key")
                #expect(Set(legacyLeft.map(\.rowid)).isDisjoint(with: fixture.hostileLegacy))

                DiskCache.resetRateLimitedReportsForTesting()
                let (_, again) = try Support.capturingStandardError { try run() }
                if pass == .companionOverCap { try admitDirectWrite() }
                try Self.expectOnlyHostileRecordsGone(
                    disk: disk, root: root, real: real, controls: fixture.controls,
                    after: "a second \(pass.rawValue) (\(placement.rawValue))")
                #expect(Self.invalidValueLines(again).isEmpty, "\(again)")
                if pass == .importer {
                    let summary = try #require(
                        disk.reconcileCompanionAccounting(
                            companions: fixture.coordinator.ssmStateCache.diskStore?.quotaEntries()
                                ?? []))
                    #expect(!summary.changedAnything, "the import is not idempotent: \(summary)")
                }
            }
        }

        // MARK: - R3: a newer build's rows are opaque

        /// `user_version = 7`, the v2 columns, three rows shaped as a newer
        /// build might shape them, and two ordinary rows. The opaque rows
        /// count, so the index is over its cap — and only rows this build
        /// understands pay for that.
        @Test func rowsOfANewerSchemaAreCountedButNeverOfferedDroppedOrTouched() throws {
            try MLXMetalTestLock.withLock {
                let root = Self.makeRoot("newer-schema")
                defer { try? FileManager.default.removeItem(at: root) }
                let modelKey = "index-rowid-newer"
                let normal = [Self.tokens(301, seed: 8_201), Self.tokens(517, seed: 8_202)]
                var normalBytes: Int64 = 0
                do {
                    let writer = Self.coordinator(root: root, modelKey: modelKey)
                    for tokens in normal {
                        writer.storePersistentBoundary(
                            tokens: tokens, diskArrays: Self.kv(), ssmStates: nil)
                        // Recency must order the two.
                        Thread.sleep(forTimeInterval: 0.02)
                    }
                    normalBytes = Support.fileBytes(
                        Support.payloadURL(
                            root, DiskCache.hashTokens(normal[0], modelKey: modelKey)))
                    try #require(normalBytes > 0)
                }

                // Opaque bytes leave room for exactly one ordinary row.
                let futureHashes = (1 ... 3).map { String(repeating: "\($0)a", count: 32) }
                let futureKey = "v3:" + String(repeating: "c", count: 40)
                let futureLegacy = "v3:" + String(repeating: "d", count: 40)
                let opaqueTotal = Self.cap - normalBytes - normalBytes / 2
                let quarter = opaqueTotal / 4
                var futureFiles: [URL: Data] = [:]
                let raw = try RawDB(root: root)
                for (index, hash) in futureHashes.enumerated() {
                    try #require(!DiskCache.isPayloadHash(hash))
                    let bytes = quarter
                    try raw.require(
                        """
                        INSERT INTO cache_entries
                            (hash, token_count, file_size, created_at, companion_key, companion_bytes)
                        VALUES ('\(hash)', 307, \(bytes), 2440587.5,
                                \(index == 0 ? "'\(futureKey)'" : "NULL"), \(index == 0 ? quarter / 2 : 0))
                        """)
                    let url = root.appendingPathComponent("\(hash).safetensors")
                    futureFiles[url] = Data(repeating: UInt8(0x31 + index), count: 4_099 + index)
                }
                try raw.require(
                    """
                    INSERT INTO legacy_companions (key, bytes, modified)
                    VALUES ('\(futureLegacy)', \(opaqueTotal - 3 * quarter - quarter / 2), 0)
                    """)
                let companionDir = Support.companionDir(root)
                try FileManager.default.createDirectory(
                    at: companionDir, withIntermediateDirectories: true)
                for key in [futureKey, futureLegacy] {
                    futureFiles[companionDir.appendingPathComponent("ssm-\(key).safetensors")] =
                        Data(repeating: 0x41, count: 4_111)
                }
                for (url, data) in futureFiles { try data.write(to: url) }
                try raw.require("PRAGMA user_version = 7")

                func opaqueRecords() throws -> [[String?]] {
                    try RawDB(root: root).rows(
                        """
                        SELECT rowid, quote(hash), file_size, quote(companion_key), companion_bytes,
                               created_at
                        FROM cache_entries WHERE length(hash) = 64 ORDER BY rowid
                        """)
                        + RawDB(root: root).rows(
                            "SELECT rowid, quote(key), bytes, modified FROM legacy_companions WHERE key LIKE 'v3:%'"
                        )
                }
                let opaqueBefore = try opaqueRecords()
                try #require(opaqueBefore.count == 4, "INVALID: \(opaqueBefore)")

                CacheCoordinator.resetImportedRootsForTesting()
                DiskCache.resetRateLimitedReportsForTesting()
                let (coordinator, log) = try Support.capturingStandardError {
                    Self.coordinator(root: root, modelKey: modelKey)
                }
                let disk = try #require(coordinator.diskCache)
                try #require(
                    disk.indexSchemaVersion == 7, "INVALID: the newer version was not read")
                try #require(disk.indexHasV2Columns)

                func expectOpaqueUntouched(_ what: String) throws {
                    #expect(
                        try opaqueRecords() == opaqueBefore, "an opaque record changed in \(what)")
                    for (url, data) in futureFiles {
                        #expect(
                            (try? Data(contentsOf: url)) == data,
                            "VICTIM: \(url.lastPathComponent) did not survive \(what)")
                    }
                }
                try expectOpaqueUntouched("the open (import and quota pass)")
                #expect(Self.invalidValueLines(log).isEmpty, "\(log)")
                // Counted: the older ordinary row paid. Understood: only it did.
                #expect(disk.fetch(tokens: normal[0]) == nil, "the opaque bytes were not counted")
                #expect(disk.fetch(tokens: normal[1]) != nil, "more was evicted than the cap asks")
                #expect(disk.snapshotStats().evictions == 1)
                #expect(disk.usageBytes() <= Self.cap)
                #expect(disk.usageBytes() > opaqueTotal - 8, "the opaque bytes left the accounting")

                // The import, run directly.
                DiskCache.resetRateLimitedReportsForTesting()
                let (summary, importLog) = try Support.capturingStandardError {
                    disk.reconcileCompanionAccounting(companions: [])
                }
                let counts = try #require(summary)
                #expect(counts.rowsDroppedForInvalidHash == 0, "\(counts)")
                #expect(counts.legacyDroppedForInvalidKey == 0, "\(counts)")
                #expect(counts.linksCleared == 0, "\(counts)")
                #expect(Self.invalidValueLines(importLog).isEmpty, "\(importLog)")
                try expectOpaqueUntouched("the import")

                // `DiskCache`'s standalone pass, on a second connection.
                // Its cap counts payloads only: room for the opaque ones and
                // one and a half ordinary rows.
                let standalone = DiskCache(
                    cacheDir: root, maxSizeBytes: Int(3 * quarter + normalBytes + normalBytes / 2),
                    modelKey: modelKey)
                try #require(standalone.indexIsFromANewerBuild)
                let extra = [Self.tokens(1_003, seed: 8_203), Self.tokens(1_291, seed: 8_204)]
                DiskCache.resetRateLimitedReportsForTesting()
                let (_, storeLog) = try Support.capturingStandardError {
                    for tokens in extra {
                        Thread.sleep(forTimeInterval: 0.02)
                        standalone.store(tokens: tokens, arrays: Self.kv())
                    }
                }
                #expect(Self.invalidValueLines(storeLog).isEmpty, "\(storeLog)")
                try expectOpaqueUntouched("the standalone eviction")
                #expect(
                    standalone.snapshotStats().evictions > 0, "INVALID: the pass evicted nothing")
                #expect(standalone.fetch(tokens: extra[1]) != nil)

                // The companion store's own cap, and both readers of the
                // unlinked list.
                DiskCache.resetRateLimitedReportsForTesting()
                let (_, readLog) = try Support.capturingStandardError {
                    _ = disk.companionsOldestFirst()
                    _ = disk.legacyCompanions()
                    _ = disk.quotaEntries()
                    coordinator.enforceCombinedDiskQuota()
                }
                #expect(Self.invalidValueLines(readLog).isEmpty, "\(readLog)")
                try expectOpaqueUntouched("the index readers")
                #expect(
                    !disk.quotaEntries().contains { $0.hash.utf8.count != 32 },
                    "an opaque row was offered to the planner")
            }
        }

        // MARK: - O6: the second layer, called directly

        @Test func removeQuotaEntriesNeverBuildsAPathFromAnInvalidValue() throws {
            try MLXMetalTestLock.withLock {
                // `../x` leaves the root: keep where it lands inside `base`.
                let base = Self.makeRoot("second-layer")
                let root = base.appendingPathComponent("root")
                defer { try? FileManager.default.removeItem(at: base) }
                let modelKey = "index-rowid-second-layer"
                let coordinator = Self.coordinator(
                    root: root, capBytes: 1 << 30, modelKey: modelKey)
                let disk = try #require(coordinator.diskCache)
                let store = try #require(coordinator.ssmStateCache.diskStore)
                let kept = Self.tokens(301, seed: 8_301)
                let doomed = Self.tokens(517, seed: 8_302)
                for tokens in [kept, doomed] {
                    coordinator.storePersistentBoundary(
                        tokens: tokens, diskArrays: Self.kv(), ssmStates: Self.recurrent())
                }
                let doomedHash = DiskCache.hashTokens(doomed, modelKey: modelKey)
                let doomedKey = SSMCompanionDiskStore.keyFor(
                    tokens: doomed, boundary: doomed.count, modelKey: modelKey)
                try #require(Support.fileBytes(Support.payloadURL(root, doomedHash)) > 0)
                try #require(Support.companionBytes(root, doomedKey) > 0)

                let hostile = ["../x", "notes", "model-00001-of-00008"]
                var victims: [URL: Data] = [:]
                let dir = Support.companionDir(root)
                for (index, name) in hostile.enumerated() {
                    victims[
                        root.appendingPathComponent("\(name).safetensors").standardizedFileURL] =
                        Data(repeating: UInt8(0x51 + index), count: 4_099 + index)
                    for suffix in [".safetensors", ".json"] {
                        victims[
                            dir.appendingPathComponent("ssm-\(name)\(suffix)").standardizedFileURL] =
                            Data(repeating: UInt8(0x61 + index), count: 4_111 + index)
                    }
                }
                // `ssm-../x` needs a directory named `ssm-..`.
                try FileManager.default.createDirectory(
                    at: dir.appendingPathComponent("ssm-.."), withIntermediateDirectories: true)
                for (url, data) in victims {
                    try FileManager.default.createDirectory(
                        at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
                    try data.write(to: url)
                }
                func expectVictimsIntact(_ what: String) {
                    for (url, data) in victims {
                        #expect(
                            (try? Data(contentsOf: url)) == data,
                            "VICTIM \(url.path) did not survive \(what)")
                    }
                }

                let stillOnDisk = store.removeQuotaEntries(hashes: Set(hostile + [doomedKey]))
                expectVictimsIntact("SSMCompanionDiskStore.removeQuotaEntries")
                #expect(stillOnDisk.isEmpty, "\(stillOnDisk)")
                #expect(
                    Support.companionBytes(root, doomedKey) == 0,
                    "INVALID: the positive control — a real key — was not removed")

                let removed = disk.removeQuotaEntries(
                    hashes: Set(hostile + [doomedHash]), removedCompanions: [doomedKey])
                expectVictimsIntact("DiskCache.removeQuotaEntries")
                #expect(
                    removed == [doomedHash], "only the real hash is reported as removed: \(removed)"
                )
                #expect(
                    Support.fileBytes(Support.payloadURL(root, doomedHash)) == 0,
                    "INVALID: the positive control — a real hash — was not removed")
                #expect(disk.fetch(tokens: kept) != nil)
                try Support.expectUsageMatchesDisk(
                    disk, root: root,
                    foreign: Set(victims.keys.map(\.lastPathComponent)))
            }
        }

        // MARK: - O7: the report is bounded in BYTES

        @Test func invalidValueReportIsBoundedInBytes() throws {
            try MLXMetalTestLock.withLock {
                let root = Self.makeRoot("report-bytes")
                defer { try? FileManager.default.removeItem(at: root) }
                let disk = DiskCache(
                    cacheDir: root, maxSizeBytes: 1 << 30, modelKey: "index-rowid-report")
                try #require(disk.indexHasV2Columns)
                // One grapheme, ten thousand bytes.
                let grapheme = "e" + String(repeating: "\u{0301}", count: 5_000)
                try #require(grapheme.count == 1 && grapheme.utf8.count == 10_001)
                try RawDB(root: root).require(
                    "INSERT INTO cache_entries (hash, token_count, file_size) VALUES ('\(grapheme)', 7, 11)"
                )

                DiskCache.resetRateLimitedReportsForTesting()
                let (_, log) = try Support.capturingStandardError {
                    disk.reconcileCompanionAccounting(companions: [])
                }
                let lines = Self.invalidValueLines(log)
                try #require(
                    lines.count == 1, "INVALID: the row was not reported: \(log.prefix(400))")
                #expect(
                    lines[0].utf8.count < 1_024,
                    "one report line is \(lines[0].utf8.count) bytes")
                #expect(try Support.indexedRows(root).isEmpty)
            }
        }
    }
}
