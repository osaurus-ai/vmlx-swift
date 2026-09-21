import Foundation
import SQLite3

/// Read-only measurements for cap resolution. No directory walk, cache
/// construction, migration or payload loading is performed to paint settings.
public struct DiskCacheVolumeSnapshot: Equatable, Sendable {
    public let totalBytes: Int64?
    public let freeBytes: Int64?
    public let ownBytes: Int64?

    public static func read(directory: URL?) -> Self {
        guard let directory else {
            return Self(totalBytes: nil, freeBytes: nil, ownBytes: nil)
        }
        var ancestor = directory.standardizedFileURL
        while !FileManager.default.fileExists(atPath: ancestor.path) {
            let parent = ancestor.deletingLastPathComponent()
            guard parent.path != ancestor.path else { break }
            ancestor = parent
        }
        let attributes = try? FileManager.default.attributesOfFileSystem(forPath: ancestor.path)
        return Self(
            totalBytes: (attributes?[.systemSize] as? NSNumber)?.int64Value,
            freeBytes: (attributes?[.systemFreeSize] as? NSNumber)?.int64Value,
            ownBytes: indexedPayloadBytes(directory: directory))
    }

    /// Missing index means no indexed payloads. Failure to read an existing
    /// index means unknown, not zero; otherwise a transient lock could ratchet
    /// Automatic down as though all of this cache's bytes belonged elsewhere.
    static func indexedPayloadBytes(directory: URL) -> Int64? {
        let path = directory.appendingPathComponent("cache_index.db").path
        guard FileManager.default.fileExists(atPath: path) else { return 0 }
        guard let attributes = try? FileManager.default.attributesOfItem(atPath: path),
            attributes[.type] as? FileAttributeType == .typeRegular
        else { return nil }
        // A closed WAL index may have no -shm file. SQLite's read-only
        // connection then cannot establish its snapshot on macOS. Allow it
        // to create coordination files, but never create a missing database
        // or execute a data write. A genuinely read-only volume still uses
        // the first path when its existing WAL state is readable.
        return measureIndex(path: path, flags: SQLITE_OPEN_READONLY)
            ?? measureIndex(path: path, flags: SQLITE_OPEN_READWRITE)
    }

    private static func measureIndex(path: String, flags: Int32) -> Int64? {
        var db: OpaquePointer?
        guard
            sqlite3_open_v2(path, &db, flags | SQLITE_OPEN_FULLMUTEX, nil)
                == SQLITE_OK
        else {
            if let db { sqlite3_close(db) }
            return nil
        }
        defer { sqlite3_close(db) }
        sqlite3_busy_timeout(db, 100)
        guard sqlite3_exec(db, "PRAGMA query_only=ON", nil, nil, nil) == SQLITE_OK else {
            return nil
        }

        func read(_ sql: String) -> Int64? {
            var statement: OpaquePointer?
            guard sqlite3_prepare_v2(db, sql, -1, &statement, nil) == SQLITE_OK else { return nil }
            defer { sqlite3_finalize(statement) }
            guard sqlite3_step(statement) == SQLITE_ROW else { return nil }
            return max(0, sqlite3_column_int64(statement, 0))
        }
        // One snapshot for schema checks and both aggregates. TOTAL avoids
        // integer-overflow failures; clamping agrees with quota row decoding.
        guard sqlite3_exec(db, "BEGIN", nil, nil, nil) == SQLITE_OK else { return nil }
        defer { sqlite3_exec(db, "ROLLBACK", nil, nil, nil) }
        guard
            let hasCompanions = read(
                "SELECT COUNT(*) FROM pragma_table_info('cache_entries') WHERE name='companion_bytes'"
            )
        else { return nil }
        let columns = hasCompanions > 0 ? ["file_size", "companion_bytes"] : ["file_size"]
        let aggregate = columns.map { "TOTAL(MAX(CAST(\($0) AS INTEGER), 0))" }.joined(
            separator: " + ")
        guard let kv = read("SELECT \(aggregate) FROM cache_entries"),
            let hasLegacy = read(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='legacy_companions'"
            )
        else { return nil }
        guard hasLegacy > 0 else { return kv }
        guard
            let legacy = read("SELECT TOTAL(MAX(CAST(bytes AS INTEGER), 0)) FROM legacy_companions")
        else {
            return nil
        }
        return IndexedBytes.sum(kv, legacy)
    }
}

extension DiskCacheCapPolicy {
    public static func resolve(
        percent: Double?, legacyGB: Double?, directory: URL?, previousCapBytes: Int64? = nil
    ) -> Resolution {
        let volume = DiskCacheVolumeSnapshot.read(directory: directory)
        return resolve(
            percent: percent, legacyGB: legacyGB,
            totalBytes: volume.totalBytes, freeBytes: volume.freeBytes, ownBytes: volume.ownBytes,
            previousCapBytes: previousCapBytes
                ?? directory.flatMap { SharedDiskCacheLimit.currentBytes(for: $0).map(Int64.init) })
    }
}
