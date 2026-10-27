import Foundation
import SQLite3

/// Read-only measurements for cap resolution. No directory walk, cache
/// construction, migration or payload loading is performed to paint settings.
public struct DiskCacheVolumeSnapshot: Equatable, Sendable {
    public let totalBytes: Int64?
    public let freeBytes: Int64?
    public let ownBytes: Int64?

    /// INVARIANT: cap resolution uses the greater of strict and host-supplied
    /// purgeable-inclusive free space. The host provider must be nonblocking;
    /// the runtime never performs the potentially slow important-usage query.
    /// Guarded by DiskCacheCapPolicyTests' resolution and provider cases.
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
            freeBytes: resolvedFreeBytes(
                strict: (attributes?[.systemFreeSize] as? NSNumber)?.int64Value,
                purgeableInclusive: purgeableInclusiveFreeBytes.provider?(ancestor.path)),
            ownBytes: indexedPayloadBytes(directory: directory))
    }

    /// Lets the host supply free space that counts purgeable space (APFS
    /// local snapshots, purgeable caches), the figure Finder and System
    /// Settings show. `.systemFreeSize` excludes it: one 8 TB Mac read 18 GB
    /// strict with 1.44 TB available, so Automatic sized the cache to ~5 GB
    /// and every explicit share above 0.2% was flagged as exceeding free
    /// space. The purgeable-inclusive query can stall in CacheDelete, so the
    /// runtime never issues it; the provider must answer without blocking
    /// (nil until it has a value). Pass nil to remove it.
    public static func setPurgeableInclusiveFreeBytesProvider(
        _ provider: (@Sendable (String) -> Int64?)?
    ) {
        purgeableInclusiveFreeBytes.provider = provider
    }

    private static let purgeableInclusiveFreeBytes = FreeBytesProviderBox()

    /// The larger of the two figures; either one alone when the other is
    /// missing or zero.
    static func resolvedFreeBytes(strict: Int64?, purgeableInclusive: Int64?) -> Int64? {
        guard let purgeableInclusive, purgeableInclusive > 0 else { return strict }
        guard let strict, strict > 0 else { return purgeableInclusive }
        return max(strict, purgeableInclusive)
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

private final class FreeBytesProviderBox: @unchecked Sendable {
    private let lock = NSLock()
    private var stored: (@Sendable (String) -> Int64?)?

    var provider: (@Sendable (String) -> Int64?)? {
        get { lock.withLock { stored } }
        set { lock.withLock { stored = newValue } }
    }
}
