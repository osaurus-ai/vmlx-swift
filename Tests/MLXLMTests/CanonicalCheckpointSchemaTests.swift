import Foundation
import SQLite3
import Testing

@testable import MLXLMCommon

@Suite struct CanonicalCheckpointSchemaTests {
    private final class Database {
        let handle: OpaquePointer

        init(_ path: String = ":memory:", readOnly: Bool = false) throws {
            var pointer: OpaquePointer?
            let flags = readOnly ? SQLITE_OPEN_READONLY : SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE
            let result = sqlite3_open_v2(path, &pointer, flags, nil)
            handle = try #require(pointer)
            #expect(result == SQLITE_OK)
        }

        deinit { sqlite3_close(handle) }

        func execute(_ sql: String) throws {
            try #require(sqlite3_exec(handle, sql, nil, nil, nil) == SQLITE_OK)
        }

        func integer(_ sql: String) throws -> Int64 {
            var statement: OpaquePointer?
            try #require(sqlite3_prepare_v2(handle, sql, -1, &statement, nil) == SQLITE_OK)
            defer { sqlite3_finalize(statement) }
            try #require(sqlite3_step(statement) == SQLITE_ROW)
            return sqlite3_column_int64(statement, 0)
        }
    }

    @Test func legacyWritesAndReopenPreserveCompatibility() throws {
        let path = FileManager.default.temporaryDirectory
            .appendingPathComponent("canonical-schema-\(UUID().uuidString).db")
        defer { try? FileManager.default.removeItem(at: path) }
        do {
            let db = try Database(path.path)
            #expect(DiskCacheIndexSchema.migrate(db.handle) == 2)
            try db.execute(
                "INSERT INTO cache_entries(hash, token_count, file_size) VALUES ('old', 16, 100)")
            #expect(DiskCacheIndexSchema.ensureReplayChunkColumn(db.handle))
            #expect(
                try db.integer(
                    "SELECT replay_chunk_size IS NULL FROM cache_entries WHERE hash='old'") == 1)
            try db.execute(
                "INSERT INTO cache_entries(hash, token_count, file_size, replay_chunk_size) VALUES ('canonical', 512, 200, 512)"
            )
            // Literal legacy replacement clears optional metadata rather than
            // accidentally inheriting a claim about a replacement payload.
            try db.execute(
                "INSERT OR REPLACE INTO cache_entries(hash, token_count, file_size) VALUES ('canonical', 512, 201)"
            )
            #expect(
                try db.integer(
                    "SELECT replay_chunk_size IS NULL FROM cache_entries WHERE hash='canonical'")
                    == 1)
            #expect(try db.integer("SELECT SUM(file_size) FROM cache_entries") == 301)
            #expect(try db.integer("PRAGMA user_version") == 2)
        }
        let reopened = try Database(path.path)
        #expect(DiskCacheIndexSchema.ensureReplayChunkColumn(reopened.handle))
        #expect(try reopened.integer("SELECT COUNT(*) FROM cache_entries") == 2)
    }

    @Test func futureAndIncompleteSchemasAreUntouched() throws {
        for future in [false, true] {
            let db = try Database()
            if future {
                #expect(DiskCacheIndexSchema.migrate(db.handle) == 2)
                try db.execute("PRAGMA user_version=3")
            } else {
                try db.execute(DiskCacheIndexSchema.v1Statements[0])
                try db.execute("PRAGMA user_version=2")
            }
            #expect(!DiskCacheIndexSchema.ensureReplayChunkColumn(db.handle))
            #expect(
                try db.integer(
                    "SELECT COUNT(*) FROM pragma_table_info('cache_entries') WHERE name='replay_chunk_size'"
                ) == 0)
        }
        #expect(!DiskCacheIndexSchema.ensureReplayChunkColumn(nil))
    }

    @Test func contentionDisablesFeatureThenRetriesWithoutLosingRows() throws {
        let path = FileManager.default.temporaryDirectory
            .appendingPathComponent("canonical-lock-\(UUID().uuidString).db")
        defer { try? FileManager.default.removeItem(at: path) }
        let owner = try Database(path.path)
        #expect(DiskCacheIndexSchema.migrate(owner.handle) == 2)
        try owner.execute(
            "INSERT INTO cache_entries(hash, token_count, file_size) VALUES ('retained', 16, 100)")
        let contender = try Database(path.path)
        try owner.execute("BEGIN IMMEDIATE")
        #expect(!DiskCacheIndexSchema.ensureReplayChunkColumn(contender.handle, busyTimeoutMs: 1))
        #expect(try contender.integer("PRAGMA busy_timeout") == 0)
        #expect(
            try contender.integer("SELECT file_size FROM cache_entries WHERE hash='retained'")
                == 100)
        try owner.execute("ROLLBACK")
        #expect(DiskCacheIndexSchema.ensureReplayChunkColumn(contender.handle, busyTimeoutMs: 1))
        // Successful extension leaves no transaction or write lock behind.
        try owner.execute("BEGIN IMMEDIATE")
        try owner.execute("ROLLBACK")
    }

    @Test func readOnlyLegacyIndexRemainsReadable() throws {
        let path = FileManager.default.temporaryDirectory
            .appendingPathComponent("canonical-readonly-\(UUID().uuidString).db")
        defer { try? FileManager.default.removeItem(at: path) }
        do {
            let db = try Database(path.path)
            #expect(DiskCacheIndexSchema.migrate(db.handle) == 2)
            try db.execute(
                "INSERT INTO cache_entries(hash, token_count, file_size) VALUES ('retained', 16, 100)"
            )
        }
        let db = try Database(path.path, readOnly: true)
        #expect(!DiskCacheIndexSchema.ensureReplayChunkColumn(db.handle, busyTimeoutMs: 1))
        #expect(try db.integer("SELECT COUNT(*) FROM cache_entries") == 1)
        #expect(try db.integer("PRAGMA user_version") == 2)
    }
}
