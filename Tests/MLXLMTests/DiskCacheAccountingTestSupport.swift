import Foundation
import SQLite3
import Testing

@testable import MLXLMCommon

/// What the disk-cache accounting suites share: where the files of a cache
/// root live, a raw connection to its index, and the accounting invariant
/// (`usageBytes()` equals the bytes on disk of everything the index names,
/// and the index names every published file).
enum DiskCacheAccountingTestSupport {

    static func companionDir(_ root: URL) -> URL {
        root.appendingPathComponent("ssm_companion")
    }

    static func payloadURL(_ root: URL, _ hash: String) -> URL {
        root.appendingPathComponent("\(hash).safetensors")
    }

    static func companionURLs(_ root: URL, _ key: String) -> [URL] {
        [
            companionDir(root).appendingPathComponent("ssm-\(key).safetensors"),
            companionDir(root).appendingPathComponent("ssm-\(key).json"),
        ]
    }

    /// Bytes of a regular file; 0 for anything else (missing, a directory).
    static func fileBytes(_ url: URL) -> Int64 {
        let attributes = try? FileManager.default.attributesOfItem(atPath: url.path)
        guard attributes?[.type] as? FileAttributeType == .typeRegular else { return 0 }
        return (attributes?[.size] as? NSNumber)?.int64Value ?? 0
    }

    static func companionBytes(_ root: URL, _ key: String) -> Int64 {
        companionURLs(root, key).reduce(0) { $0 + fileBytes($1) }
    }

    /// A raw connection, independent of any `DiskCache`.
    final class RawDB {
        let handle: OpaquePointer

        init(root: URL) throws {
            var db: OpaquePointer?
            let path = root.appendingPathComponent("cache_index.db").path
            guard sqlite3_open(path, &db) == SQLITE_OK, let db else {
                throw RawDBError.open(path)
            }
            handle = db
            sqlite3_busy_timeout(db, 2_000)
        }

        deinit { sqlite3_close(handle) }

        func require(_ sql: String) throws {
            guard sqlite3_exec(handle, sql, nil, nil, nil) == SQLITE_OK else {
                throw RawDBError.statement(sql, String(cString: sqlite3_errmsg(handle)))
            }
        }

        /// Every row of `sql` as optional strings (NULL → nil).
        func rows(_ sql: String) throws -> [[String?]] {
            var stmt: OpaquePointer?
            guard sqlite3_prepare_v2(handle, sql, -1, &stmt, nil) == SQLITE_OK else {
                throw RawDBError.statement(sql, String(cString: sqlite3_errmsg(handle)))
            }
            defer { sqlite3_finalize(stmt) }
            var out: [[String?]] = []
            while sqlite3_step(stmt) == SQLITE_ROW {
                out.append(
                    (0 ..< sqlite3_column_count(stmt)).map { column in
                        sqlite3_column_text(stmt, column).map { String(cString: $0) }
                    })
            }
            return out
        }
    }

    enum RawDBError: Error {
        case open(String)
        case statement(String, String)
    }

    struct IndexedRow: Equatable {
        let hash: String
        let fileSize: Int64
        let companionKey: String?
        let companionBytes: Int64
    }

    static func indexedRows(_ root: URL) throws -> [IndexedRow] {
        try RawDB(root: root)
            .rows(
                "SELECT hash, file_size, companion_key, companion_bytes FROM cache_entries ORDER BY hash"
            )
            .map { row in
                IndexedRow(
                    hash: row[0] ?? "", fileSize: Int64(row[1] ?? "") ?? -1,
                    companionKey: row[2], companionBytes: Int64(row[3] ?? "") ?? -1)
            }
    }

    static func legacyRows(_ root: URL) throws -> [String: Int64] {
        var out: [String: Int64] = [:]
        for row in try RawDB(root: root).rows("SELECT key, bytes FROM legacy_companions") {
            out[row[0] ?? ""] = Int64(row[1] ?? "") ?? -1
        }
        return out
    }

    /// Bytes on disk of everything a v2 index names: each row's payload, each
    /// row's companion files, and each unlinked companion's files.
    static func onDiskBytesOfIndexedFiles(_ root: URL) throws -> Int64 {
        var total: Int64 = 0
        for row in try indexedRows(root) {
            total += fileBytes(payloadURL(root, row.hash))
            if let key = row.companionKey {
                total += companionBytes(root, key)
            }
        }
        for key in try legacyRows(root).keys {
            total += companionBytes(root, key)
        }
        return total
    }

    /// The invariant. `atLeast` keeps it from passing on an empty fixture.
    /// `checkCompleteness: false` is for the one test that drops a file the
    /// index must NOT know about.
    ///
    /// `foreign` are the names (in the root or in the companion directory) of
    /// files the TEST planted as somebody else's: the completeness walk is
    /// deliberately loose about names — anything that ends in `.safetensors`
    /// is taken for a payload — so a planted victim has to be named, exactly,
    /// to be left out of it.
    static func expectUsageMatchesDisk(
        _ disk: DiskCache, root: URL, atLeast: Int64 = 1, checkCompleteness: Bool = true,
        foreign: Set<String> = [],
        sourceLocation: SourceLocation = #_sourceLocation
    ) throws {
        let onDisk = try onDiskBytesOfIndexedFiles(root)
        #expect(onDisk >= atLeast, "fixture is empty", sourceLocation: sourceLocation)
        #expect(disk.usageBytes() == onDisk, sourceLocation: sourceLocation)
        if checkCompleteness {
            try expectIndexNamesEveryPublishedFile(
                root, foreign: foreign, sourceLocation: sourceLocation)
        }
    }

    /// Completeness, by a test-only walk: every published `<hash>.safetensors`
    /// has a row, and every published `ssm-<key>.{safetensors,json}` is named
    /// by a row's `companion_key` or by `legacy_companions`. Unpublished
    /// (`.partial-`) names are not entries. Throws when a directory cannot be
    /// listed, so an unreadable directory is never a silent pass.
    static func expectIndexNamesEveryPublishedFile(
        _ root: URL, foreign: Set<String> = [],
        sourceLocation: SourceLocation = #_sourceLocation
    ) throws {
        let rows = try indexedRows(root)
        let rowHashes = Set(rows.map(\.hash))
        let namedCompanions = Set(rows.compactMap(\.companionKey))
            .union(try legacyRows(root).keys)

        for name in try FileManager.default.contentsOfDirectory(atPath: root.path)
        where name.hasSuffix(".safetensors") && !DiskCache.isUnpublishedName(name)
            && !foreign.contains(name)
        {
            guard fileBytes(root.appendingPathComponent(name)) > 0 else { continue }
            let hash = String(name.dropLast(".safetensors".count))
            #expect(
                rowHashes.contains(hash), "payload \(name) is on disk with no row",
                sourceLocation: sourceLocation)
        }

        let dir = companionDir(root)
        guard FileManager.default.fileExists(atPath: dir.path) else { return }
        for name in try FileManager.default.contentsOfDirectory(atPath: dir.path)
        where name.hasPrefix("ssm-") && !DiskCache.isUnpublishedName(name)
            && !foreign.contains(name)
        {
            guard name.hasSuffix(".safetensors") || name.hasSuffix(".json"),
                fileBytes(dir.appendingPathComponent(name)) > 0,
                let dot = name.lastIndex(of: ".")
            else { continue }
            let key = String(name[name.index(name.startIndex, offsetBy: 4) ..< dot])
            #expect(
                namedCompanions.contains(key),
                "companion \(name) is on disk but the index does not name it",
                sourceLocation: sourceLocation)
        }
    }

    /// `chmod` does nothing for root and on some file systems; a test that
    /// relies on it must not pass because of that.
    static func requireUnlistable(
        _ dir: URL, sourceLocation: SourceLocation = #_sourceLocation
    ) throws {
        try #require(
            (try? FileManager.default.contentsOfDirectory(atPath: dir.path)) == nil,
            "INVALID: chmod 000 had no effect (root / filesystem)",
            sourceLocation: sourceLocation)
    }

    /// Run `body` with standard error redirected to a file, and return what
    /// it wrote. The callers are serialized and hold `MLXMetalTestLock`.
    static func capturingStandardError<T>(_ body: () throws -> T) throws -> (T, String) {
        let sink = FileManager.default.temporaryDirectory
            .appendingPathComponent("vmlx-stderr-\(UUID().uuidString).log")
        defer { try? FileManager.default.removeItem(at: sink) }
        fflush(stderr)
        let saved = dup(2)
        try #require(saved >= 0, "INVALID: could not save standard error")
        let fd = open(sink.path, O_WRONLY | O_CREAT | O_TRUNC, 0o600)
        try #require(fd >= 0, "INVALID: could not open the capture file")
        dup2(fd, 2)
        close(fd)
        defer {
            fflush(stderr)
            dup2(saved, 2)
            close(saved)
        }
        let value = try body()
        fflush(stderr)
        let text = (try? String(contentsOf: sink, encoding: .utf8)) ?? ""
        // Still visible in the test log.
        FileHandle(fileDescriptor: saved).write(Data(text.utf8))
        return (value, text)
    }

    /// Set a path's own modification date — a symlink's, not its target's.
    static func age(_ url: URL, by seconds: TimeInterval) throws {
        let when = Date().addingTimeInterval(-seconds).timeIntervalSince1970
        var times = [
            timeval(tv_sec: Int(when), tv_usec: 0), timeval(tv_sec: Int(when), tv_usec: 0),
        ]
        try #require(
            lutimes(url.path, &times) == 0, "INVALID: lutimes failed for \(url.lastPathComponent)")
    }

    /// A v1 index a newer build has claimed: this build leaves it alone, so
    /// `indexHasV2Columns` is false and the directory-walk path stays in force.
    static func makeV1OnlyIndex(in root: URL) throws {
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let raw = try RawDB(root: root)
        try raw.require("PRAGMA journal_mode=WAL")
        try raw.require(
            """
            CREATE TABLE IF NOT EXISTS cache_entries (
                hash TEXT PRIMARY KEY,
                token_count INTEGER,
                file_size INTEGER,
                created_at REAL DEFAULT (julianday('now'))
            )
            """)
        try raw.require(
            "CREATE INDEX IF NOT EXISTS idx_cache_entries_token_count ON cache_entries(token_count DESC)"
        )
        try raw.require("PRAGMA user_version = 99")
    }
}
