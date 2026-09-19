import Foundation
import MLX
@testable import MLXLMCommon
import SQLite3
import Testing

/// Schema versioning of `cache_index.db`.
///
/// The index is shared: every loaded model opens its own connection to the
/// same file, and an older app build may read and write the same directory.
/// These tests pin the properties that follow from that: a v1 index migrates
/// without losing a row, racing or interrupted migrations converge, a
/// migration that fails gives the write lock back, and the literal v1
/// statements keep working against a v2 index.
@Suite struct DiskCacheIndexMigrationTests {

    // MARK: - Fixtures

    /// The v1 DDL exactly as shipped, including WAL mode.
    private static let v1DDL: [String] = [
        "PRAGMA journal_mode=WAL",
        """
        CREATE TABLE IF NOT EXISTS cache_entries (
            hash TEXT PRIMARY KEY,
            token_count INTEGER,
            file_size INTEGER,
            created_at REAL DEFAULT (julianday('now'))
        )
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_cache_entries_token_count
        ON cache_entries(token_count DESC)
        """,
    ]

    private struct Row: Equatable {
        let hash: String
        let tokenCount: Int64
        let fileSize: Int64
        let createdAt: Double
    }

    private static let seedRows: [Row] = [
        Row(hash: "aaaa", tokenCount: 17, fileSize: 1_001, createdAt: 2_460_000.125),
        Row(hash: "bbbb", tokenCount: 333, fileSize: 20_002, createdAt: 2_460_001.5),
        Row(hash: "cccc", tokenCount: 4_099, fileSize: 300_003, createdAt: 2_460_002.875),
    ]

    private static let v2ColumnNames = [
        "model_key", "kind", "chain_id", "companion_key", "companion_bytes",
    ]

    private static func makeTempDir(_ label: String) throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("vmlx-index-migration-\(label)-\(UUID())")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    private static func dbPath(_ dir: URL) -> String {
        dir.appendingPathComponent("cache_index.db").path
    }

    /// A raw connection, independent of any `DiskCache`. Used by one thread
    /// at a time.
    private final class RawDB: @unchecked Sendable {
        enum Bind {
            case int(Int64)
            case real(Double)
            case text(String)
        }

        let handle: OpaquePointer

        init(_ path: String) throws {
            var db: OpaquePointer?
            guard sqlite3_open(path, &db) == SQLITE_OK, let db else {
                throw RawDBError.open(path)
            }
            handle = db
        }

        deinit { sqlite3_close(handle) }

        /// Runs one statement to completion; returns the SQLite result code of
        /// the first failing call, or `SQLITE_OK`.
        @discardableResult
        func exec(_ sql: String) -> Int32 {
            sqlite3_exec(handle, sql, nil, nil, nil)
        }

        func require(_ sql: String) throws {
            let rc = exec(sql)
            guard rc == SQLITE_OK else {
                throw RawDBError.statement(sql, rc, String(cString: sqlite3_errmsg(handle)))
            }
        }

        func int(_ sql: String) throws -> Int64 {
            var stmt: OpaquePointer?
            guard sqlite3_prepare_v2(handle, sql, -1, &stmt, nil) == SQLITE_OK else {
                throw RawDBError.statement(sql, -1, String(cString: sqlite3_errmsg(handle)))
            }
            defer { sqlite3_finalize(stmt) }
            guard sqlite3_step(stmt) == SQLITE_ROW else {
                throw RawDBError.statement(sql, -2, "no row")
            }
            return sqlite3_column_int64(stmt, 0)
        }

        func strings(_ sql: String) throws -> [String] {
            var stmt: OpaquePointer?
            guard sqlite3_prepare_v2(handle, sql, -1, &stmt, nil) == SQLITE_OK else {
                throw RawDBError.statement(sql, -1, String(cString: sqlite3_errmsg(handle)))
            }
            defer { sqlite3_finalize(stmt) }
            var out: [String] = []
            while sqlite3_step(stmt) == SQLITE_ROW {
                out.append(sqlite3_column_text(stmt, 0).map { String(cString: $0) } ?? "<null>")
            }
            return out
        }

        func rows() throws -> [Row] {
            let sql = """
                SELECT hash, token_count, file_size, created_at
                FROM cache_entries ORDER BY hash
                """
            var stmt: OpaquePointer?
            guard sqlite3_prepare_v2(handle, sql, -1, &stmt, nil) == SQLITE_OK else {
                throw RawDBError.statement(sql, -1, String(cString: sqlite3_errmsg(handle)))
            }
            defer { sqlite3_finalize(stmt) }
            var out: [Row] = []
            while sqlite3_step(stmt) == SQLITE_ROW {
                out.append(
                    Row(
                        hash: String(cString: sqlite3_column_text(stmt, 0)),
                        tokenCount: sqlite3_column_int64(stmt, 1),
                        fileSize: sqlite3_column_int64(stmt, 2),
                        createdAt: sqlite3_column_double(stmt, 3)))
            }
            return out
        }

        func columnNames() throws -> [String] {
            try strings("SELECT name FROM pragma_table_info('cache_entries')")
        }

        /// Prepares `sql` exactly as given, binds, and steps to completion.
        /// Every row comes back as the text of its columns.
        @discardableResult
        func run(_ sql: String, _ binds: [Bind] = []) throws -> [[String]] {
            var stmt: OpaquePointer?
            guard sqlite3_prepare_v2(handle, sql, -1, &stmt, nil) == SQLITE_OK else {
                throw RawDBError.statement(sql, -1, String(cString: sqlite3_errmsg(handle)))
            }
            defer { sqlite3_finalize(stmt) }
            let transient = unsafeBitCast(-1, to: sqlite3_destructor_type.self)
            for (offset, bind) in binds.enumerated() {
                let index = Int32(offset + 1)
                switch bind {
                case .int(let value): sqlite3_bind_int64(stmt, index, value)
                case .real(let value): sqlite3_bind_double(stmt, index, value)
                case .text(let value): sqlite3_bind_text(stmt, index, value, -1, transient)
                }
            }
            var out: [[String]] = []
            while true {
                let rc = sqlite3_step(stmt)
                if rc == SQLITE_DONE { return out }
                guard rc == SQLITE_ROW else {
                    throw RawDBError.statement(sql, rc, String(cString: sqlite3_errmsg(handle)))
                }
                out.append(
                    (0 ..< sqlite3_column_count(stmt)).map { column in
                        sqlite3_column_text(stmt, column).map { String(cString: $0) } ?? "<null>"
                    })
            }
        }

        /// Watches, and optionally fails, this connection's statements from
        /// here on. `log` must outlive the connection or be removed first.
        func install(_ log: StatementLog) {
            sqlite3_set_authorizer(
                handle,
                { context, action, first, second, _, _ in
                    guard let context else { return SQLITE_OK }
                    return Unmanaged<StatementLog>.fromOpaque(context).takeUnretainedValue()
                        .authorize(action, first, second)
                },
                Unmanaged.passUnretained(log).toOpaque())
        }

        func removeStatementLog() {
            sqlite3_set_authorizer(handle, nil, nil)
        }
    }

    private enum RawDBError: Error {
        case open(String)
        case statement(String, Int32, String)
    }

    /// Builds a v1 index holding `seedRows`. The connection is closed before
    /// this returns, so the file is exactly what an old build leaves behind.
    private static func buildV1Index(in dir: URL, extra: [String] = []) throws {
        let raw = try RawDB(dbPath(dir))
        for sql in v1DDL { try raw.require(sql) }
        for row in seedRows {
            try raw.require(
                """
                INSERT INTO cache_entries (hash, token_count, file_size, created_at)
                VALUES ('\(row.hash)', \(row.tokenCount), \(row.fileSize), \(row.createdAt))
                """)
        }
        for sql in extra { try raw.require(sql) }
    }

    /// The whole v2 shape in one place: version, one copy of every column,
    /// the side table, and the seed rows untouched with v2 defaults.
    private static func expectMigratedSeedIndex(
        _ dir: URL, sourceLocation: SourceLocation = #_sourceLocation
    ) throws {
        let raw = try RawDB(dbPath(dir))
        #expect(try raw.int("PRAGMA user_version") == 2, sourceLocation: sourceLocation)
        let columns = try raw.columnNames()
        for name in ["hash", "token_count", "file_size", "created_at"] + v2ColumnNames {
            #expect(
                columns.filter { $0 == name }.count == 1,
                "column \(name) in \(columns)", sourceLocation: sourceLocation)
        }
        #expect(
            try raw.int(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='legacy_companions'")
                == 1, sourceLocation: sourceLocation)
        #expect(
            try raw.int(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='index' AND name='idx_cache_entries_chain'")
                == 1, sourceLocation: sourceLocation)
        #expect(try raw.rows() == seedRows, sourceLocation: sourceLocation)
        #expect(
            try raw.int(
                """
                SELECT COUNT(*) FROM cache_entries
                WHERE chain_id IS NULL AND kind = 0 AND companion_bytes = 0
                  AND model_key IS NULL AND companion_key IS NULL
                """) == Int64(seedRows.count), sourceLocation: sourceLocation)
    }

    private static let v1ColumnNames = ["hash", "token_count", "file_size", "created_at"]

    /// What a migration that failed inside its transaction has to leave
    /// behind: nothing, and no lock.
    ///
    /// The migrating connection is asked first. A transaction left open
    /// still shows that connection its own ALTERs and its own version stamp,
    /// which no other connection can see.
    private static func expectRolledBackAndUnlocked(
        migrator: RawDB, returned: Int32, dir: URL, _ label: String,
        sourceLocation: SourceLocation = #_sourceLocation
    ) throws {
        #expect(returned == 0, "\(label): returned version", sourceLocation: sourceLocation)
        #expect(
            sqlite3_get_autocommit(migrator.handle) != 0,
            "\(label): the migrating connection is still inside its transaction",
            sourceLocation: sourceLocation)
        #expect(
            try migrator.int("PRAGMA user_version") == 0, "\(label): migrator",
            sourceLocation: sourceLocation)
        #expect(
            try migrator.columnNames() == v1ColumnNames, "\(label): migrator",
            sourceLocation: sourceLocation)

        let other = try RawDB(dbPath(dir))
        sqlite3_busy_timeout(other.handle, 0)
        #expect(
            other.exec("BEGIN IMMEDIATE") == SQLITE_OK,
            "\(label): a second connection cannot take the write lock",
            sourceLocation: sourceLocation)
        other.exec("ROLLBACK")
        #expect(
            try other.int("PRAGMA user_version") == 0, "\(label): second connection",
            sourceLocation: sourceLocation)
        #expect(
            try other.columnNames() == v1ColumnNames, "\(label): second connection",
            sourceLocation: sourceLocation)
        #expect(
            try other.int(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='legacy_companions'")
                == 0, "\(label)", sourceLocation: sourceLocation)
        #expect(
            try other.int(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='index' AND name='idx_cache_entries_chain'")
                == 0, "\(label)", sourceLocation: sourceLocation)
        #expect(try other.rows() == seedRows, "\(label)", sourceLocation: sourceLocation)
    }

    /// What one round of racing opens observed.
    private struct RaceRound {
        let versions: [Int32]
        let hasColumns: [Bool]
        /// At least two workers were inside `DiskCache.init` at the same time.
        let overlapped: Bool
    }

    /// Opens `workers` caches on `dir` at once. Every worker is a thread of
    /// its own (`concurrentPerform` may legally run its iterations one after
    /// another), parks on a barrier until all of them exist, and records when
    /// its open started and ended.
    private static func raceOpens(workers: Int, dir: URL) -> RaceRound {
        let results = RaceResults(count: workers)
        let ready = DispatchSemaphore(value: 0)
        let go = DispatchSemaphore(value: 0)
        let done = DispatchGroup()
        for slot in 0 ..< workers {
            done.enter()
            Thread {
                ready.signal()
                go.wait()
                // Scoped: the connection is closed before this worker reports
                // done. A last close checkpoints the WAL under an exclusive
                // lock, and the caller's verifying connection does not wait.
                do {
                    let start = DispatchTime.now().uptimeNanoseconds
                    let cache = DiskCache(
                        cacheDir: dir, maxSizeBytes: 1 << 30, modelKey: "m\(slot)")
                    let end = DispatchTime.now().uptimeNanoseconds
                    results.record(
                        slot: slot, version: cache.indexSchemaVersion,
                        hasColumns: cache.indexHasV2Columns, openStart: start, openEnd: end)
                }
                done.leave()
            }.start()
        }
        for _ in 0 ..< workers { ready.wait() }
        for _ in 0 ..< workers { go.signal() }
        done.wait()
        return RaceRound(
            versions: results.versions, hasColumns: results.hasColumns,
            overlapped: results.anyOpensOverlapped)
    }

    /// A race test in which no two opens ever overlapped has proven nothing,
    /// so it fails rather than passes.
    private static func reportRace(
        rounds: Int, overlapped: Int, workers: Int, index: String,
        sourceLocation: SourceLocation = #_sourceLocation
    ) {
        print(
            "MIGRATION_RACE rounds=\(rounds) overlapped=\(overlapped) workers=\(workers) index=\(index)")
        if overlapped == 0 {
            Issue.record(
                "INVALID: no round had overlapping opens — this run proves nothing about the race",
                sourceLocation: sourceLocation)
        }
    }

    // MARK: - Tests

    @Test func v1IndexOpensAndKeepsEveryRow() throws {
        let dir = try Self.makeTempDir("v1-open")
        defer { try? FileManager.default.removeItem(at: dir) }
        try Self.buildV1Index(in: dir)

        let cache = DiskCache(cacheDir: dir, maxSizeBytes: 1 << 30, modelKey: "m")

        #expect(cache.indexSchemaVersion == 2)
        #expect(cache.indexHasV2Columns)
        try Self.expectMigratedSeedIndex(dir)
    }

    @Test func freshIndexIsCreatedAtV2() throws {
        let dir = try Self.makeTempDir("fresh")
        defer { try? FileManager.default.removeItem(at: dir) }

        let cache = DiskCache(cacheDir: dir, maxSizeBytes: 1 << 30, modelKey: "m")

        #expect(cache.indexSchemaVersion == 2)
        #expect(cache.indexHasV2Columns)
        let raw = try RawDB(Self.dbPath(dir))
        #expect(try raw.int("PRAGMA user_version") == 2)
        let columns = try raw.columnNames()
        for name in Self.v2ColumnNames {
            #expect(columns.filter { $0 == name }.count == 1, "column \(name) in \(columns)")
        }
        #expect(
            try raw.int(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='legacy_companions'")
                == 1)
        #expect(try raw.int("SELECT COUNT(*) FROM cache_entries") == 0)
    }

    @Test func migrationIsIdempotent() throws {
        let dir = try Self.makeTempDir("idempotent")
        defer { try? FileManager.default.removeItem(at: dir) }
        try Self.buildV1Index(in: dir)

        for _ in 0 ..< 3 {
            // Scoped so the connection closes before the next open.
            do {
                let cache = DiskCache(cacheDir: dir, maxSizeBytes: 1 << 30, modelKey: "m")
                #expect(cache.indexSchemaVersion == 2)
                #expect(cache.indexHasV2Columns)
            }
            try Self.expectMigratedSeedIndex(dir)
        }
    }

    /// Two models opening the same v1 index at once. What this pins is the
    /// waiting: both connections take `BEGIN IMMEDIATE` and the loser sits in
    /// the busy handler until the winner commits. The re-read of
    /// `user_version` under the lock and the per-column check each cover for
    /// the other here, so neither is pinned by this test; they have their own
    /// (`versionMovedWhileWaitingForTheLockIsLeftAlone`,
    /// `partiallyAppliedMigrationCompletes`).
    @Test func twoConnectionsRacingTheMigrationBothSucceed() throws {
        var completed = 0
        var overlapped = 0
        defer {
            Self.reportRace(rounds: completed, overlapped: overlapped, workers: 2, index: "v1")
        }
        for round in 0 ..< 20 {
            let dir = try Self.makeTempDir("race-\(round)")
            defer { try? FileManager.default.removeItem(at: dir) }
            try Self.buildV1Index(in: dir)

            let result = Self.raceOpens(workers: 2, dir: dir)
            if result.overlapped { overlapped += 1 }

            #expect(result.versions == [2, 2], "round \(round)")
            #expect(result.hasColumns == [true, true], "round \(round)")
            try Self.expectMigratedSeedIndex(dir)
            completed += 1
        }
    }

    /// The same race with no index on disk at all: several models loading at
    /// first launch. `DiskCache.init` issues its CREATE TABLE with no busy
    /// timeout, so every connection's CREATE can lose to another's lock; the
    /// migration has to create the table itself, and the column probe has to
    /// wait rather than read a busy database as "no v2 columns".
    ///
    /// Two connections for twenty rounds passes without the first fix, so
    /// this runs the shape that does not: eight connections, a hundred
    /// rounds. The waiting probe is not pinned here (this race passes without
    /// it); `columnProbeWaitsForABusyIndexInsteadOfReportingAbsent` pins it.
    @Test func connectionsRacingOnAFreshDirectoryAllReachV2() throws {
        let connections = 8
        var completed = 0
        var overlapped = 0
        defer {
            Self.reportRace(
                rounds: completed, overlapped: overlapped, workers: connections, index: "fresh")
        }
        for round in 0 ..< 100 {
            let dir = try Self.makeTempDir("fresh-race-\(round)")
            defer { try? FileManager.default.removeItem(at: dir) }

            let result = Self.raceOpens(workers: connections, dir: dir)
            if result.overlapped { overlapped += 1 }

            #expect(
                result.versions == Array(repeating: 2, count: connections), "round \(round)")
            #expect(
                result.hasColumns == Array(repeating: true, count: connections),
                "round \(round)")
            let raw = try RawDB(Self.dbPath(dir))
            #expect(try raw.int("PRAGMA user_version") == 2, "round \(round)")
            let columns = try raw.columnNames()
            for name in ["hash", "token_count", "file_size", "created_at"] + Self.v2ColumnNames {
                #expect(
                    columns.filter { $0 == name }.count == 1,
                    "round \(round) column \(name) in \(columns)")
            }
            completed += 1
        }
    }

    @Test func partiallyAppliedMigrationCompletes() throws {
        let dir = try Self.makeTempDir("partial")
        defer { try? FileManager.default.removeItem(at: dir) }
        // A crash after the first two ALTERs: two columns exist, user_version
        // is still 0.
        try Self.buildV1Index(
            in: dir,
            extra: [
                "ALTER TABLE cache_entries ADD COLUMN model_key TEXT",
                "ALTER TABLE cache_entries ADD COLUMN kind INTEGER NOT NULL DEFAULT 0",
            ])
        do {
            let raw = try RawDB(Self.dbPath(dir))
            #expect(try raw.int("PRAGMA user_version") == 0)
            #expect(try raw.columnNames().count == 6)
        }

        let cache = DiskCache(cacheDir: dir, maxSizeBytes: 1 << 30, modelKey: "m")

        #expect(cache.indexSchemaVersion == 2)
        #expect(cache.indexHasV2Columns)
        try Self.expectMigratedSeedIndex(dir)
    }

    // The statements of the last v1 build, copied verbatim from
    // `DiskCache.swift` at 60263594.
    private static let baselineInsert = """
        INSERT OR REPLACE INTO cache_entries (hash, token_count, file_size)
        VALUES (?, ?, ?)
        """
    private static let baselineCandidateCounts = """
        SELECT DISTINCT token_count
        FROM cache_entries
        WHERE token_count > 0 AND token_count <= ?
        ORDER BY token_count DESC
        LIMIT ?
        """
    private static let baselineAllEntries = "SELECT hash, file_size, created_at FROM cache_entries"
    private static let baselineEntryMetadata =
        "SELECT token_count, file_size FROM cache_entries WHERE hash = ?"
    private static let baselinePayloadUsage =
        "SELECT COALESCE(SUM(file_size), 0), COUNT(*) FROM cache_entries"
    private static let baselineTouch = "UPDATE cache_entries SET created_at = ? WHERE hash = ?"
    private static let baselineDelete = "DELETE FROM cache_entries WHERE hash = ?"
    private static let baselineTotalSize = "SELECT COALESCE(SUM(file_size), 0) FROM cache_entries"
    private static let baselineEvictionOrder =
        "SELECT hash, file_size FROM cache_entries ORDER BY created_at ASC"

    /// The downgrade guarantee: an older build, and the osaurus purge tool,
    /// run these exact statements against whatever index they find.
    @Test func v1WriterAndReaderStillWorkOnV2Index() throws {
        let dir = try Self.makeTempDir("downgrade")
        defer { try? FileManager.default.removeItem(at: dir) }
        try Self.buildV1Index(in: dir)
        do {
            let cache = DiskCache(cacheDir: dir, maxSizeBytes: 1 << 30, modelKey: "m")
            #expect(cache.indexSchemaVersion == 2)
        }
        let raw = try RawDB(Self.dbPath(dir))
        // Fail closed: everything below is vacuous against a v1 index.
        try #require(try raw.int("PRAGMA user_version") == 2)
        try #require(try raw.columnNames().contains("companion_bytes"))

        // Old writer.
        try raw.run(Self.baselineInsert, [.text("oldwriter"), .int(77), .int(7_007)])
        #expect(
            try raw.int(
                """
                SELECT COUNT(*) FROM cache_entries
                WHERE hash = 'oldwriter' AND token_count = 77 AND file_size = 7007
                  AND created_at IS NOT NULL
                  AND kind = 0 AND chain_id IS NULL AND companion_bytes = 0
                """) == 1)

        // Old readers.
        #expect(try raw.run(Self.baselineAllEntries).count == 4)
        // Bound and limit both cut: 4099 is over the bound, 17 is past the limit.
        #expect(
            try raw.run(Self.baselineCandidateCounts, [.int(333), .int(2)]) == [["333"], ["77"]])
        #expect(
            try raw.run(Self.baselineEntryMetadata, [.text("oldwriter")]) == [["77", "7007"]])
        let totalBytes = 1_001 + 20_002 + 300_003 + 7_007
        #expect(try raw.run(Self.baselinePayloadUsage) == [["\(totalBytes)", "4"]])
        #expect(try raw.run(Self.baselineTotalSize) == [["\(totalBytes)"]])

        // Old touch, then the old eviction order: the touched row moves from
        // oldest to newest. `oldwriter` carries julianday('now').
        try raw.run(Self.baselineTouch, [.real(2_470_000.5), .text("aaaa")])
        #expect(sqlite3_changes(raw.handle) == 1)
        #expect(
            try raw.run(Self.baselineEvictionOrder) == [
                ["bbbb", "20002"], ["cccc", "300003"], ["oldwriter", "7007"], ["aaaa", "1001"],
            ])

        // Old delete-by-hash.
        try raw.run(Self.baselineDelete, [.text("aaaa")])
        #expect(sqlite3_changes(raw.handle) == 1)

        // osaurus purge tool.
        #expect(
            try raw.strings("SELECT hash FROM cache_entries").sorted()
                == ["bbbb", "cccc", "oldwriter"])
        try raw.require("DELETE FROM cache_entries")
        #expect(try raw.int("SELECT COUNT(*) FROM cache_entries") == 0)
    }

    @Test func newerSchemaIsLeftAlone() throws {
        let dir = try Self.makeTempDir("newer")
        defer { try? FileManager.default.removeItem(at: dir) }
        try Self.buildV1Index(in: dir, extra: ["PRAGMA user_version = 7"])

        let cache = DiskCache(cacheDir: dir, maxSizeBytes: 1 << 30, modelKey: "m")

        #expect(cache.indexSchemaVersion == 7)
        // Whatever v7 looks like, this build did not write to it.
        #expect(!cache.indexHasV2Columns)
        let raw = try RawDB(Self.dbPath(dir))
        #expect(try raw.int("PRAGMA user_version") == 7)
        #expect(try raw.columnNames() == ["hash", "token_count", "file_size", "created_at"])
        #expect(
            try raw.int(
                "SELECT COUNT(*) FROM sqlite_master WHERE name='legacy_companions'") == 0)
        #expect(try raw.rows() == Self.seedRows)
    }

    /// The positive half of the same contract: the probe goes by the columns,
    /// not by the version, so a newer build's index that still carries them
    /// is usable.
    @Test func newerSchemaWithV2ColumnsReportsTrue() throws {
        let dir = try Self.makeTempDir("newer-v2")
        defer { try? FileManager.default.removeItem(at: dir) }
        try Self.buildV1Index(in: dir)
        do {
            let raw = try RawDB(Self.dbPath(dir))
            try #require(DiskCacheIndexSchema.migrate(raw.handle) == 2)
            try raw.require("PRAGMA user_version = 7")
        }

        do {
            let raw = try RawDB(Self.dbPath(dir))
            #expect(DiskCacheIndexSchema.migrate(raw.handle) == 7)
            #expect(DiskCacheIndexSchema.hasV2Columns(raw.handle))
        }
        let cache = DiskCache(cacheDir: dir, maxSizeBytes: 1 << 30, modelKey: "m")
        #expect(cache.indexSchemaVersion == 7)
        #expect(cache.indexHasV2Columns)

        let raw = try RawDB(Self.dbPath(dir))
        #expect(try raw.int("PRAGMA user_version") == 7)
        #expect(try raw.columnNames() == Self.v1ColumnNames + Self.v2ColumnNames)
        #expect(try raw.rows() == Self.seedRows)
    }

    /// A newer build migrates the index while this connection is waiting for
    /// the write lock. The version read before the lock (0) is stale by the
    /// time the lock arrives; stamping 2 over the newer build's 7 would be a
    /// downgrade. The interleaving is forced, not hoped for: the blocker
    /// commits only once the migrating connection has prepared its BEGIN,
    /// which is after its pre-lock read.
    @Test func versionMovedWhileWaitingForTheLockIsLeftAlone() throws {
        let dir = try Self.makeTempDir("moved")
        defer { try? FileManager.default.removeItem(at: dir) }
        try Self.buildV1Index(in: dir)

        let blocker = try RawDB(Self.dbPath(dir))
        try blocker.require("BEGIN IMMEDIATE")
        try blocker.require("PRAGMA user_version = 7")

        let migrator = try RawDB(Self.dbPath(dir))
        // Fail closed: the blocker's uncommitted 7 is not visible yet.
        try #require(try migrator.int("PRAGMA user_version") == 0)
        let log = StatementLog(fault: .none)
        migrator.install(log)

        let result = RaceResults(count: 1)
        let finished = DispatchSemaphore(value: 0)
        Thread {
            // The connection's authorizer points at `log`: keep it alive for
            // as long as this thread can run a statement.
            defer { withExtendedLifetime(log) {} }
            let version = DiskCacheIndexSchema.migrate(migrator.handle)
            result.record(slot: 0, version: version, hasColumns: false, openStart: 0, openEnd: 0)
            finished.signal()
        }.start()

        try #require(log.sawBegin.wait(timeout: .now() + 10) == .success)
        try blocker.require("COMMIT")
        try #require(finished.wait(timeout: .now() + 10) == .success)
        migrator.removeStatementLog()

        #expect(result.versions == [7])
        // Nothing was written, so there is nothing to commit.
        #expect(log.transactionEvents == ["BEGIN", "ROLLBACK"])
        #expect(log.events.filter { $0 == "ALTER" }.isEmpty)
        #expect(sqlite3_get_autocommit(migrator.handle) != 0)
        #expect(try migrator.int("PRAGMA user_version") == 7)
        #expect(try migrator.columnNames() == Self.v1ColumnNames)

        let other = try RawDB(Self.dbPath(dir))
        sqlite3_busy_timeout(other.handle, 0)
        #expect(other.exec("BEGIN IMMEDIATE") == SQLITE_OK)
        other.exec("ROLLBACK")
        #expect(try other.int("PRAGMA user_version") == 7)
        #expect(try other.columnNames() == Self.v1ColumnNames)
        #expect(
            try other.int("SELECT COUNT(*) FROM sqlite_master WHERE name='legacy_companions'")
                == 0)
        #expect(try other.rows() == Self.seedRows)
    }

    /// A v2 statement fails with the transaction open and five ALTERs already
    /// applied. A table squatting on the v2 index's name does it: `CREATE
    /// INDEX IF NOT EXISTS` still fails with "there is already a table named
    /// idx_cache_entries_chain". Without the ROLLBACK this connection keeps
    /// the write lock and every other model's insert comes back BUSY.
    @Test func failedStatementRollsBackAndReleasesTheWriteLock() throws {
        try MLXMetalTestLock.withLock {
            let dir = try Self.makeTempDir("stmt-fail")
            defer { try? FileManager.default.removeItem(at: dir) }
            try Self.buildV1Index(
                in: dir, extra: ["CREATE TABLE idx_cache_entries_chain (x INTEGER)"])

            do {
                let migrator = try RawDB(Self.dbPath(dir))
                // Fail closed: a v1 index, and the squatter is in place.
                try #require(try migrator.int("PRAGMA user_version") == 0)
                try #require(try migrator.columnNames() == Self.v1ColumnNames)
                try #require(
                    try migrator.int(
                        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='idx_cache_entries_chain'")
                        == 1)
                let log = StatementLog(fault: .none)
                migrator.install(log)

                let returned = DiskCacheIndexSchema.migrate(migrator.handle)
                migrator.removeStatementLog()

                // The failure came after the ALTERs, inside the transaction.
                #expect(log.events.filter { $0 == "ALTER" }.count == 5)
                #expect(log.transactionEvents == ["BEGIN", "ROLLBACK"])
                try Self.expectRolledBackAndUnlocked(
                    migrator: migrator, returned: returned, dir: dir, "failed statement")
                #expect(!DiskCacheIndexSchema.hasV2Columns(migrator.handle))
            }

            // The cache comes up on that index, as v1, and works.
            let modelKey = "failed-statement-model"
            let cache = DiskCache(cacheDir: dir, maxSizeBytes: 1 << 30, modelKey: modelKey)
            #expect(cache.indexSchemaVersion == 0)
            #expect(!cache.indexHasV2Columns)

            // 5 tokens, 7 elements: nothing here is a round number.
            let tokens = [21, 22, 23, 24, 25]
            let arrays = ["data": MLXArray(Array(0 ..< 7).map { Float($0) + 0.25 })]
            cache.store(tokens: tokens, arrays: arrays)

            let fetched = try #require(cache.fetch(tokens: tokens))
            #expect(
                fetched["data"]?.asArray(Float.self) == [0.25, 1.25, 2.25, 3.25, 4.25, 5.25, 6.25])
            #expect(cache.hits == 1)

            let raw = try RawDB(Self.dbPath(dir))
            let hash = DiskCache.hashTokens(tokens, modelKey: modelKey)
            #expect(
                try raw.int(
                    "SELECT COUNT(*) FROM cache_entries WHERE hash = '\(hash)' AND token_count = 5")
                    == 1)
            #expect(try raw.int("SELECT COUNT(*) FROM cache_entries") == 4)
            #expect(try raw.int("PRAGMA user_version") == 0)
            #expect(try raw.columnNames() == Self.v1ColumnNames)
        }
    }

    /// `user_version` cannot be read once the lock is held. The failure is
    /// injected through SQLite's authorizer, so the product code is unchanged.
    @Test func unreadableVersionUnderTheLockRollsBackAndReleasesTheWriteLock() throws {
        let dir = try Self.makeTempDir("version-unreadable")
        defer { try? FileManager.default.removeItem(at: dir) }
        try Self.buildV1Index(in: dir)

        let migrator = try RawDB(Self.dbPath(dir))
        let log = StatementLog(fault: .userVersionRead)
        migrator.install(log)

        let returned = DiskCacheIndexSchema.migrate(migrator.handle)
        migrator.removeStatementLog()

        // The lock was taken, the read under it failed, nothing was attempted.
        #expect(
            log.events == [
                "read user_version denied", "BEGIN", "read user_version denied", "ROLLBACK",
            ])
        try Self.expectRolledBackAndUnlocked(
            migrator: migrator, returned: returned, dir: dir, "unreadable version")
    }

    /// The last exit: every statement ran, then the version stamp or the
    /// COMMIT itself fails.
    @Test func failedStampOrCommitRollsBackAndReleasesTheWriteLock() throws {
        for fault in [StatementLog.Fault.userVersionWrite, .commit] {
            let dir = try Self.makeTempDir("commit-fail")
            defer { try? FileManager.default.removeItem(at: dir) }
            try Self.buildV1Index(in: dir)

            let migrator = try RawDB(Self.dbPath(dir))
            let log = StatementLog(fault: fault)
            migrator.install(log)

            let returned = DiskCacheIndexSchema.migrate(migrator.handle)
            migrator.removeStatementLog()

            #expect(log.events.filter { $0 == "ALTER" }.count == 5, "\(fault)")
            switch fault {
            case .userVersionWrite:
                #expect(log.events.contains("write user_version denied"), "\(fault)")
                #expect(log.transactionEvents == ["BEGIN", "ROLLBACK"], "\(fault)")
            case .commit:
                #expect(log.events.contains("write user_version"), "\(fault)")
                #expect(
                    log.transactionEvents == ["BEGIN", "COMMIT denied", "ROLLBACK"], "\(fault)")
            default:
                Issue.record("unexpected fault \(fault)")
            }
            try Self.expectRolledBackAndUnlocked(
                migrator: migrator, returned: returned, dir: dir, "\(fault)")
        }
    }

    /// The schema helpers borrow the connection's busy timeout; they do not
    /// get to keep it, and they do not get to replace one the caller set.
    @Test func busyTimeoutIsRestoredAndACallersTimeoutIsKept() throws {
        for callerTimeout: Int32 in [0, 1234] {
            let dir = try Self.makeTempDir("busy-timeout-\(callerTimeout)")
            defer { try? FileManager.default.removeItem(at: dir) }
            try Self.buildV1Index(in: dir)

            let raw = try RawDB(Self.dbPath(dir))
            if callerTimeout > 0 { sqlite3_busy_timeout(raw.handle, callerTimeout) }
            try #require(try raw.int("PRAGMA busy_timeout") == Int64(callerTimeout))

            let version = DiskCacheIndexSchema.migrate(raw.handle)
            #expect(
                try raw.int("PRAGMA busy_timeout") == Int64(callerTimeout),
                "after migrate, caller timeout \(callerTimeout)")
            let hasColumns = DiskCacheIndexSchema.hasV2Columns(raw.handle)
            #expect(
                try raw.int("PRAGMA busy_timeout") == Int64(callerTimeout),
                "after migrate and hasV2Columns, caller timeout \(callerTimeout)")

            // The probe on a connection `migrate` never touched.
            let probe = try RawDB(Self.dbPath(dir))
            if callerTimeout > 0 { sqlite3_busy_timeout(probe.handle, callerTimeout) }
            let probed = DiskCacheIndexSchema.hasV2Columns(probe.handle)
            #expect(
                try probe.int("PRAGMA busy_timeout") == Int64(callerTimeout),
                "after hasV2Columns alone, caller timeout \(callerTimeout)")

            // Fail closed: both helpers ran their full path, not an early exit.
            #expect(version == 2)
            #expect(hasColumns)
            #expect(probed)
        }
    }

    /// The column probe against an index it cannot read yet. Before WAL mode
    /// takes hold (a fresh directory), a writer's exclusive lock turns a plain
    /// read into SQLITE_BUSY. The probe has to wait that out; only an index
    /// that stays unreadable for the whole timeout reports `false`.
    @Test func columnProbeWaitsForABusyIndexInsteadOfReportingAbsent() throws {
        let dir = try Self.makeTempDir("probe-busy")
        defer { try? FileManager.default.removeItem(at: dir) }
        // A v2 index in rollback-journal mode: no `PRAGMA journal_mode=WAL`.
        do {
            let raw = try RawDB(Self.dbPath(dir))
            for sql in Self.v1DDL.dropFirst() { try raw.require(sql) }
            try #require(DiskCacheIndexSchema.migrate(raw.handle) == 2)
            try #require(try raw.strings("PRAGMA journal_mode") == ["delete"])
        }

        let probe = try RawDB(Self.dbPath(dir))
        try #require(DiskCacheIndexSchema.hasV2Columns(probe.handle))

        let blocker = try RawDB(Self.dbPath(dir))
        try blocker.require("BEGIN EXCLUSIVE")
        // Fail closed: a plain read on the probe's connection is BUSY now.
        try #require(probe.exec("SELECT COUNT(*) FROM cache_entries") == SQLITE_BUSY)

        // Unreadable for the whole timeout: `false`, the conservative answer.
        #expect(!DiskCacheIndexSchema.hasV2Columns(probe.handle, busyTimeoutMs: 50))

        // Released while the probe is waiting: the columns are there.
        let releasedAt = Stamp()
        let releaserDone = DispatchSemaphore(value: 0)
        let releaser = Thread {
            Thread.sleep(forTimeInterval: 0.3)
            releasedAt.set(DispatchTime.now().uptimeNanoseconds)
            blocker.exec("ROLLBACK")
            releaserDone.signal()
        }
        let started = DispatchTime.now().uptimeNanoseconds
        releaser.start()
        let hasColumns = DiskCacheIndexSchema.hasV2Columns(probe.handle)
        let returned = DispatchTime.now().uptimeNanoseconds
        try #require(releaserDone.wait(timeout: .now() + 10) == .success)

        #expect(hasColumns)
        // The answer came from a read made after the lock was gone, by a
        // call that began while it was still held.
        let release = try #require(releasedAt.value)
        #expect(started < release)
        #expect(returned >= release)
    }

    @Test func failedMigrationLeavesAWorkingV1Index() throws {
        try MLXMetalTestLock.withLock {
            let dir = try Self.makeTempDir("failed")
            defer { try? FileManager.default.removeItem(at: dir) }
            try Self.buildV1Index(in: dir)

            // Another connection holds the write lock for longer than the
            // migration is willing to wait.
            let blocker = try RawDB(Self.dbPath(dir))
            try blocker.require("BEGIN IMMEDIATE")

            // Directly: the function reports v1 and does not throw or trap.
            do {
                let raw = try RawDB(Self.dbPath(dir))
                let version = DiskCacheIndexSchema.migrate(raw.handle, busyTimeoutMs: 50)
                #expect(version < 2)
                #expect(!DiskCacheIndexSchema.hasV2Columns(raw.handle))
            }

            // Through DiskCache: the cache comes up on the v1 index.
            let modelKey = "failed-migration-model"
            let cache = DiskCache(
                cacheDir: dir, maxSizeBytes: 1 << 30, modelKey: modelKey,
                indexMigrationBusyTimeoutMs: 50)
            #expect(cache.indexSchemaVersion < 2)
            #expect(!cache.indexHasV2Columns)

            try blocker.require("ROLLBACK")

            // Fail closed: prove the index this cache is about to use is v1.
            do {
                let raw = try RawDB(Self.dbPath(dir))
                try #require(try raw.int("PRAGMA user_version") == 0)
                try #require(
                    try raw.columnNames() == ["hash", "token_count", "file_size", "created_at"])
                #expect(try raw.rows() == Self.seedRows)
            }

            // 5 tokens, 7 elements: nothing here is a round number.
            let tokens = [11, 12, 13, 14, 15]
            let arrays = ["data": MLXArray(Array(0 ..< 7).map { Float($0) + 0.5 })]
            cache.store(tokens: tokens, arrays: arrays)

            let fetched = try #require(cache.fetch(tokens: tokens))
            #expect(fetched["data"]?.asArray(Float.self) == [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5])
            #expect(cache.hits == 1)

            let raw = try RawDB(Self.dbPath(dir))
            let hash = DiskCache.hashTokens(tokens, modelKey: modelKey)
            #expect(
                try raw.int(
                    "SELECT COUNT(*) FROM cache_entries WHERE hash = '\(hash)' AND token_count = 5")
                    == 1)
            #expect(try raw.int("SELECT COUNT(*) FROM cache_entries") == 4)
            #expect(try raw.int("PRAGMA user_version") == 0)
        }
    }
}

/// Results written from racing worker threads, with the window each worker
/// spent opening its cache.
private final class RaceResults: @unchecked Sendable {
    private let lock = NSLock()
    private var _versions: [Int32]
    private var _hasColumns: [Bool]
    private var _opens: [(start: UInt64, end: UInt64)]

    init(count: Int) {
        _versions = Array(repeating: -1, count: count)
        _hasColumns = Array(repeating: false, count: count)
        _opens = Array(repeating: (0, 0), count: count)
    }

    func record(slot: Int, version: Int32, hasColumns: Bool, openStart: UInt64, openEnd: UInt64) {
        lock.lock()
        defer { lock.unlock() }
        _versions[slot] = version
        _hasColumns[slot] = hasColumns
        _opens[slot] = (openStart, openEnd)
    }

    var versions: [Int32] {
        lock.lock()
        defer { lock.unlock() }
        return _versions
    }

    var hasColumns: [Bool] {
        lock.lock()
        defer { lock.unlock() }
        return _hasColumns
    }

    /// Whether some open started before an earlier-starting one had ended.
    var anyOpensOverlapped: Bool {
        lock.lock()
        defer { lock.unlock() }
        var latestEnd: UInt64 = 0
        for (index, open) in _opens.sorted(by: { $0.start < $1.start }).enumerated() {
            if index > 0, open.start < latestEnd { return true }
            latestEnd = max(latestEnd, open.end)
        }
        return false
    }
}

/// A time written by one thread and read by another.
private final class Stamp: @unchecked Sendable {
    private let lock = NSLock()
    private var _value: UInt64?

    func set(_ value: UInt64) {
        lock.lock()
        defer { lock.unlock() }
        _value = value
    }

    var value: UInt64? {
        lock.lock()
        defer { lock.unlock() }
        return _value
    }
}

/// A SQLite authorizer that records what a connection prepares — transaction
/// control, reads and writes of `user_version`, ALTERs — and can refuse one
/// kind of statement, which the caller sees as that statement failing.
private final class StatementLog: @unchecked Sendable {
    enum Fault { case none, userVersionRead, userVersionWrite, commit }

    let fault: Fault
    /// Signalled when the connection prepares a BEGIN.
    let sawBegin = DispatchSemaphore(value: 0)
    private let lock = NSLock()
    private var _events: [String] = []

    init(fault: Fault) { self.fault = fault }

    var events: [String] {
        lock.lock()
        defer { lock.unlock() }
        return _events
    }

    var transactionEvents: [String] {
        events.filter { event in
            ["BEGIN", "COMMIT", "ROLLBACK"].contains { event.hasPrefix($0) }
        }
    }

    func authorize(_ action: Int32, _ first: UnsafePointer<CChar>?, _ second: UnsafePointer<CChar>?)
        -> Int32
    {
        switch action {
        case SQLITE_TRANSACTION:
            let operation = first.map { String(cString: $0) } ?? "?"
            if operation == "COMMIT", fault == .commit { return deny("COMMIT") }
            append(operation)
            if operation == "BEGIN" { sawBegin.signal() }
        case SQLITE_PRAGMA:
            guard let first, String(cString: first) == "user_version" else { break }
            if second == nil {
                if fault == .userVersionRead { return deny("read user_version") }
                append("read user_version")
            } else {
                if fault == .userVersionWrite { return deny("write user_version") }
                append("write user_version")
            }
        case SQLITE_ALTER_TABLE:
            append("ALTER")
        default:
            break
        }
        return SQLITE_OK
    }

    private func deny(_ event: String) -> Int32 {
        append("\(event) denied")
        return SQLITE_DENY
    }

    private func append(_ event: String) {
        lock.lock()
        defer { lock.unlock() }
        _events.append(event)
    }
}
