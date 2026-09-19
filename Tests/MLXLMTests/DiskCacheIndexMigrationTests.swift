import Foundation
import MLX
@testable import MLXLMCommon
import SQLite3
import Testing

/// Schema versioning of `cache_index.db`.
///
/// The index is shared: every loaded model opens its own connection to the
/// same file, and an older app build may read and write the same directory.
/// These tests pin the three properties that follow from that: a v1 index
/// migrates without losing a row, racing or interrupted migrations converge,
/// and the literal v1 statements keep working against a v2 index.
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

    /// A raw connection, independent of any `DiskCache`.
    private final class RawDB {
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

    @Test func twoConnectionsRacingTheMigrationBothSucceed() throws {
        for round in 0 ..< 20 {
            let dir = try Self.makeTempDir("race-\(round)")
            defer { try? FileManager.default.removeItem(at: dir) }
            try Self.buildV1Index(in: dir)

            let results = RaceResults(count: 2)
            DispatchQueue.concurrentPerform(iterations: 2) { slot in
                let cache = DiskCache(cacheDir: dir, maxSizeBytes: 1 << 30, modelKey: "m\(slot)")
                results.record(
                    slot: slot, version: cache.indexSchemaVersion,
                    hasColumns: cache.indexHasV2Columns)
            }

            #expect(results.versions == [2, 2], "round \(round)")
            #expect(results.hasColumns == [true, true], "round \(round)")
            try Self.expectMigratedSeedIndex(dir)
        }
    }

    /// The same race with no index on disk at all: several models loading at
    /// first launch. `DiskCache.init` issues its CREATE TABLE with no busy
    /// timeout, so every connection's CREATE can lose to another's lock; the
    /// migration has to create the table itself, and the column probe has to
    /// wait rather than read a busy database as "no v2 columns".
    ///
    /// Two connections for twenty rounds passes without either fix, so this
    /// runs the shape that does not: eight connections, a hundred rounds.
    @Test func connectionsRacingOnAFreshDirectoryAllReachV2() throws {
        let connections = 8
        for round in 0 ..< 100 {
            let dir = try Self.makeTempDir("fresh-race-\(round)")
            defer { try? FileManager.default.removeItem(at: dir) }

            let results = RaceResults(count: connections)
            DispatchQueue.concurrentPerform(iterations: connections) { slot in
                let cache = DiskCache(cacheDir: dir, maxSizeBytes: 1 << 30, modelKey: "m\(slot)")
                results.record(
                    slot: slot, version: cache.indexSchemaVersion,
                    hasColumns: cache.indexHasV2Columns)
            }

            #expect(
                results.versions == Array(repeating: 2, count: connections), "round \(round)")
            #expect(
                results.hasColumns == Array(repeating: true, count: connections),
                "round \(round)")
            let raw = try RawDB(Self.dbPath(dir))
            #expect(try raw.int("PRAGMA user_version") == 2, "round \(round)")
            let columns = try raw.columnNames()
            for name in ["hash", "token_count", "file_size", "created_at"] + Self.v2ColumnNames {
                #expect(
                    columns.filter { $0 == name }.count == 1,
                    "round \(round) column \(name) in \(columns)")
            }
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
        var stmt: OpaquePointer?
        try #require(
            sqlite3_prepare_v2(
                raw.handle,
                "INSERT OR REPLACE INTO cache_entries (hash, token_count, file_size) VALUES (?,?,?)",
                -1, &stmt, nil) == SQLITE_OK)
        sqlite3_bind_int64(stmt, 2, 77)
        sqlite3_bind_int64(stmt, 3, 7_007)
        "oldwriter".withCString { cStr in
            sqlite3_bind_text(stmt, 1, cStr, -1, nil)
            #expect(sqlite3_step(stmt) == SQLITE_DONE)
        }
        sqlite3_finalize(stmt)

        #expect(
            try raw.int(
                """
                SELECT COUNT(*) FROM cache_entries
                WHERE hash = 'oldwriter' AND token_count = 77 AND file_size = 7007
                  AND created_at IS NOT NULL
                  AND kind = 0 AND chain_id IS NULL AND companion_bytes = 0
                """) == 1)

        // Old readers.
        #expect(
            try raw.strings("SELECT hash, file_size, created_at FROM cache_entries").count == 4)
        #expect(
            try raw.strings(
                "SELECT DISTINCT token_count FROM cache_entries ORDER BY token_count DESC"
            ) == ["4099", "333", "77", "17"])
        #expect(
            try raw.int("SELECT COALESCE(SUM(file_size),0) FROM cache_entries")
                == 1_001 + 20_002 + 300_003 + 7_007)

        // Old delete-by-hash.
        try #require(
            sqlite3_prepare_v2(
                raw.handle, "DELETE FROM cache_entries WHERE hash = ?", -1, &stmt, nil)
                == SQLITE_OK)
        "aaaa".withCString { cStr in
            sqlite3_bind_text(stmt, 1, cStr, -1, nil)
            #expect(sqlite3_step(stmt) == SQLITE_DONE)
        }
        sqlite3_finalize(stmt)
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

/// Results written from `concurrentPerform` workers.
private final class RaceResults: @unchecked Sendable {
    private let lock = NSLock()
    private var _versions: [Int32]
    private var _hasColumns: [Bool]

    init(count: Int) {
        _versions = Array(repeating: -1, count: count)
        _hasColumns = Array(repeating: false, count: count)
    }

    func record(slot: Int, version: Int32, hasColumns: Bool) {
        lock.lock()
        defer { lock.unlock() }
        _versions[slot] = version
        _hasColumns[slot] = hasColumns
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
}
