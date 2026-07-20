// Copyright © 2025 Apple Inc. All rights reserved.

import CryptoKit
import Foundation
import MLX
import SQLite3
import os

/// Thread-safe snapshot of ``DiskCache`` counters.
public struct DiskCacheStats: Sendable {
    public let hits: Int
    public let misses: Int
    public let stores: Int
    public let maxSizeBytes: Int
}

/// One indexed KV payload used by the coordinator's shared disk-quota pass.
/// `createdAt` mirrors the existing oldest-entry eviction order.
struct DiskCacheQuotaEntry: Sendable {
    let hash: String
    let bytes: Int64
    let createdAt: Date
}

/// Process-wide guard for MLX safetensors disk-cache IO.
///
/// Each model owns its own ``DiskCache`` instance, so an instance-local lock
/// cannot prevent this crash class:
///
/// - model A finishes generation and calls `save_safetensors`
/// - model B starts a following request and calls `loadArraysAndMetadata`
///
/// Both paths can submit/evaluate Metal work while touching safetensors-backed
/// arrays. Keep them globally serialized until MLX's safetensors IO is proven
/// safe for cross-thread, cross-model overlap.
enum MLXDiskCacheIOLock {
    static let shared = OSAllocatedUnfairLock()
}

/// Public bridge for callers that need to serialize MLX materialization with
/// vMLX disk/cache tensor I/O.
///
/// This is intentionally narrower than a general inference lock. It protects
/// operations such as `MLXArray.asArray(...)` that submit/evaluate Metal work
/// while cache stores or safetensors I/O may also be draining command buffers.
/// Live Ling/Nemotron-family rows reproduced Metal command-buffer assertions
/// when a post-tool request tokenized while the previous turn's SSM companion
/// cache write-through was still saving.
public enum MLXCacheIOLock {
    public static func withSerializedMLXCacheIO<T>(_ body: () throws -> T) rethrows -> T {
        MLXDiskCacheIOLock.shared.lock()
        defer {
            Stream.gpu.synchronize()
            MLXDiskCacheIOLock.shared.unlock()
        }
        Stream.gpu.synchronize()
        return try body()
    }
}

/// L2 SSD cache with SQLite index and safetensors file storage.
///
/// `DiskCache` provides persistent KV cache storage on disk using safetensors
/// files for tensor data and a SQLite database for indexing. Writes are
/// synchronous and serialized under a lock — the comment here previously claimed
/// they were dispatched to a background task, which they are not (see `store`);
/// that mattered, because it implies the caller's arrays are retained past the
/// call, and callers reasoning about copy lifetimes were misled by it. Reads are
/// likewise synchronous since they typically feed directly into model inference.
public final class DiskCache: @unchecked Sendable {

    // MARK: - Properties

    /// Root directory for cache files and the SQLite index.
    public let cacheDir: URL

    /// Maximum total cache size in bytes.
    public let maxSizeBytes: Int

    /// Model key for cache isolation (prevents cross-model hash collisions).
    public let modelKey: String?

    /// SQLite database handle.
    private var db: OpaquePointer?

    /// Lock for thread-safe access to mutable state.
    private let lock = OSAllocatedUnfairLock()

    /// Number of successful cache hits.
    public private(set) var hits: Int = 0

    /// Number of cache misses.
    public private(set) var misses: Int = 0

    /// Number of store operations initiated.
    public private(set) var stores: Int = 0

    /// Thread-safe copy of current disk-cache counters.
    public func snapshotStats() -> DiskCacheStats {
        lock.lock()
        defer { lock.unlock() }
        return DiskCacheStats(
            hits: hits,
            misses: misses,
            stores: stores,
            maxSizeBytes: maxSizeBytes)
    }

    // MARK: - Initialization

    /// Creates a new disk cache.
    ///
    /// - Parameters:
    ///   - cacheDir: Directory where safetensors files and the SQLite index are stored.
    ///   - maxSizeGB: Maximum cache size in gigabytes. Defaults to 10 GB.
    public init(cacheDir: URL, maxSizeGB: Float = 10.0, modelKey: String? = nil) {
        self.cacheDir = cacheDir
        self.maxSizeBytes = Int(maxSizeGB * 1_073_741_824)
        self.modelKey = modelKey

        // Create cache directory if needed
        try? FileManager.default.createDirectory(at: cacheDir, withIntermediateDirectories: true)

        // Open SQLite database
        let dbPath = cacheDir.appendingPathComponent("cache_index.db").path
        if sqlite3_open(dbPath, &db) != SQLITE_OK {
            db = nil
            return
        }

        // Enable WAL mode for better concurrent read performance
        executeSQL("PRAGMA journal_mode=WAL")

        // Create the index table
        executeSQL("""
            CREATE TABLE IF NOT EXISTS cache_entries (
                hash TEXT PRIMARY KEY,
                token_count INTEGER,
                file_size INTEGER,
                created_at REAL DEFAULT (julianday('now'))
            )
            """)
    }

    deinit {
        if let db {
            sqlite3_close(db)
        }
    }

    // MARK: - Public API

    /// Store token arrays to disk as a safetensors file.
    ///
    /// Arrays are evaluated on the calling thread, then the file write and
    /// SQLite insert are dispatched to a background task.
    ///
    /// - Parameters:
    ///   - tokens: Token IDs used to compute the cache key hash.
    ///   - arrays: Dictionary of named MLX arrays to persist.
    public func store(tokens: [Int], arrays: [String: MLXArray], mediaSalt: String? = nil) {
        let hash = DiskCache.hashTokens(tokens, modelKey: modelKey, mediaSalt: mediaSalt)
        let url = safetensorsURL(for: hash)
        let tokenCount = tokens.count
        if ProcessInfo.processInfo.environment["VMLX_CACHE_FETCH_TRACE"] == "1" {
            FileHandle.standardError.write(Data(
                "[vmlx][cache/disk-store] count=\(tokenCount) keys=\(arrays.keys.sorted().prefix(6))\n".utf8))
        }

        // Iter 61: the full write path (realize + save + SQLite insert)
        // must be serialized. MLX.eval AND the safetensors save both
        // submit Metal command-buffer work, and two threads overlapping
        // those calls crash with
        //   "failed assertion _status < MTLCommandBufferStatusCommitted"
        // even when each individual `save()` is held by a lock. So the
        // lock has to cover the realize step too.
        //
        // Iter 174: make that serialization process-wide. Osaurus can keep
        // multiple models resident, therefore multiple CacheCoordinator /
        // DiskCache instances can overlap. A MiniMax post-answer save raced a
        // ZAYA restore in the next request and crashed in MLX safetensors IO.
        // Instance locks are not enough for that topology.
        //
        // BatchEngine's actor serializes per-engine, but the coordinator
        // is reachable from non-actor callers (TokenIterator path,
        // external cache warmers), so thread-safety has to live here,
        // not rely on the caller.
        //
        // SYNCHRONOUS write (not dispatched to background) because prior
        // Darwin dispatch-to-background races with process termination on
        // short sessions would leave 0-byte safetensors files on disk.
        //
        // Use manual lock/unlock rather than `withLock` because MLXArray
        // is not `Sendable` and `OSAllocatedUnfairLock.withLock` needs
        // `@Sendable` closures under Swift 6 strict concurrency. The
        // unfair-lock primitive doesn't require Sendable — we just need
        // `defer { unlock() }` to cover every exit path.
        MLXDiskCacheIOLock.shared.lock()
        defer { MLXDiskCacheIOLock.shared.unlock() }
        lock.lock()
        defer { lock.unlock() }
        stores += 1
        // Pre-realize arrays under the lock so Metal work completes
        // before the writer hits the C++ save path AND no other thread
        // can interleave MLX ops on the same device during this window.
        // The explicit stream syncs are required for post-generation cache
        // stores: the decode loop uses asyncEval, and MLX's eval/safetensors
        // paths add command-buffer completion handlers. Entering those paths
        // while the default GPU stream still has a committed command buffer
        // can trip Metal's `_status < MTLCommandBufferStatusCommitted`
        // assertion. Sync before materializing, then again before/after save.
        Stream.gpu.synchronize()
        MLX.eval(Array(arrays.values))
        Stream.gpu.synchronize()
        do {
            try save(arrays: arrays, metadata: ["format": "mlx"], url: url)
            Stream.gpu.synchronize()

            let fileSize: Int
            if let attrs = try? FileManager.default.attributesOfItem(atPath: url.path),
                let size = attrs[.size] as? Int
            {
                fileSize = size
            } else {
                fileSize = 0
            }

            _insertEntryLocked(hash: hash, tokenCount: tokenCount, fileSize: fileSize)
            _evictIfNeededLocked()
        } catch {
            // Best-effort: swallow so a write failure doesn't fail
            // the caller's request — the model output is already
            // produced. But LOG to stderr so operational failures
            // surface instead of hiding silently.
            FileHandle.standardError.write(Data(
                "[vmlx][cache/disk] store failed for hash \(hash): \(error)\n"
                .utf8))
        }
    }

    /// Fetch cached arrays for the given token sequence.
    ///
    /// - Parameter tokens: Token IDs to look up.
    /// - Returns: The cached arrays if found, or `nil` on a miss.
    public func fetch(tokens: [Int], mediaSalt: String? = nil) -> [String: MLXArray]? {
        let hash = DiskCache.hashTokens(tokens, modelKey: modelKey, mediaSalt: mediaSalt)
        let url = safetensorsURL(for: hash)

        MLXDiskCacheIOLock.shared.lock()
        defer { MLXDiskCacheIOLock.shared.unlock() }
        lock.lock()
        defer { lock.unlock() }

        guard FileManager.default.fileExists(atPath: url.path) else {
            misses += 1
            return nil
        }

        do {
            let (arrays, _) = try loadArraysAndMetadata(url: url)
            hits += 1
            return arrays
        } catch {
            misses += 1
            // A failed deserialize is almost always a corrupt safetensors
            // file — a 0-byte leftover from the pre-synchronous-store
            // bug, a partial write from an earlier crash, disk full
            // during flush, or a format-version mismatch after upgrade.
            // Log the specific error so operators can see the reason
            // instead of silently counting a cache miss, and delete the
            // corrupt file so the next turn doesn't retry and log the
            // same error on every fetch.
            FileHandle.standardError.write(Data(
                "[vmlx][cache/disk] fetch corrupt entry at \(url.lastPathComponent): \(error) — removing\n"
                .utf8))
            try? FileManager.default.removeItem(at: url)
            // Drop the SQLite row too. Removing only the file orphans the
            // `cache_entries` row, whose `file_size` then permanently inflates
            // the `SUM(file_size)` eviction quota (unbounded on-disk growth and
            // premature eviction of live entries). The fetch path already holds
            // `lock`, so delete in-place.
            _deleteEntryLocked(hash: hash)
            return nil
        }
    }

    /// Candidate prompt-boundary lengths currently present in the disk index.
    ///
    /// The disk tier is content-addressed by the full token prefix hash, so a
    /// caller still has to probe `fetch(tokens: tokens.prefix(n))` to prove a
    /// candidate is for the same model/media/token prefix. Returning lengths
    /// from the SQLite index lets higher layers find cross-session growing-chat
    /// prefix hits without walking every possible token count.
    public func candidateTokenCounts(maxTokens: Int, limit: Int = 128) -> [Int] {
        guard let db, maxTokens > 0, limit > 0 else { return [] }
        lock.lock()
        defer { lock.unlock() }

        var counts: [Int] = []
        let sql = """
            SELECT DISTINCT token_count
            FROM cache_entries
            WHERE token_count > 0 AND token_count <= ?
            ORDER BY token_count DESC
            LIMIT ?
            """
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            return []
        }
        sqlite3_bind_int64(stmt, 1, Int64(maxTokens))
        sqlite3_bind_int(stmt, 2, Int32(limit))
        while sqlite3_step(stmt) == SQLITE_ROW {
            counts.append(Int(sqlite3_column_int64(stmt, 0)))
        }
        sqlite3_finalize(stmt)
        return counts
    }

    /// Snapshot indexed KV payloads for the coordinator's combined KV +
    /// recurrent-companion quota. Database/WAL bookkeeping is intentionally
    /// excluded, matching this cache's existing `SUM(file_size)` contract.
    func quotaEntries() -> [DiskCacheQuotaEntry] {
        guard let db else { return [] }
        lock.lock()
        defer { lock.unlock() }

        var entries: [DiskCacheQuotaEntry] = []
        var stmt: OpaquePointer?
        let sql = "SELECT hash, file_size, created_at FROM cache_entries"
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            return []
        }
        defer { sqlite3_finalize(stmt) }

        while sqlite3_step(stmt) == SQLITE_ROW {
            guard let cHash = sqlite3_column_text(stmt, 0) else { continue }
            let hash = String(cString: cHash)
            let bytes = max(0, sqlite3_column_int64(stmt, 1))
            let julianDay = sqlite3_column_double(stmt, 2)
            let unixTime = (julianDay - 2_440_587.5) * 86_400
            entries.append(DiskCacheQuotaEntry(
                hash: hash,
                bytes: bytes,
                createdAt: Date(timeIntervalSince1970: unixTime)))
        }
        return entries
    }

    /// Remove indexed KV payloads selected by the combined quota pass.
    /// The process-wide IO lock prevents another cache instance from loading
    /// a file while it is removed; the SQLite row is deleted atomically with
    /// respect to this instance's fetch/candidate queries.
    func removeQuotaEntries(hashes: Set<String>) {
        guard !hashes.isEmpty else { return }
        MLXDiskCacheIOLock.shared.lock()
        defer { MLXDiskCacheIOLock.shared.unlock() }
        lock.lock()
        defer { lock.unlock() }

        for hash in hashes {
            try? FileManager.default.removeItem(at: safetensorsURL(for: hash))
            _deleteEntryLocked(hash: hash)
        }
    }

    /// Remove all cached entries and safetensors files.
    public func clear() {
        MLXDiskCacheIOLock.shared.lock()
        defer { MLXDiskCacheIOLock.shared.unlock() }

        // Delete all SQLite entries
        lock.lock()
        defer { lock.unlock() }

        executeSQL("DELETE FROM cache_entries")

        // Remove all .safetensors files in the cache directory
        if let enumerator = FileManager.default.enumerator(
            at: cacheDir,
            includingPropertiesForKeys: nil,
            options: [.skipsSubdirectoryDescendants]
        ) {
            for case let fileURL as URL in enumerator {
                if fileURL.pathExtension == "safetensors" {
                    try? FileManager.default.removeItem(at: fileURL)
                }
            }
        }

        // Reset stats
        hits = 0
        misses = 0
        stores = 0
    }

    // MARK: - Hashing

    /// Compute a deterministic hash from a token sequence.
    ///
    /// Uses SHA-256 over the raw byte representation of the token array
    /// and returns the first 32 hex characters. When `modelKey` is provided,
    /// it is hashed first to prevent cross-model cache collisions.
    ///
    /// - Parameters:
    ///   - tokens: The token IDs to hash.
    ///   - modelKey: Optional model identifier for cache isolation.
    /// - Returns: A 32-character lowercase hex string.
    public static func hashTokens(
        _ tokens: [Int],
        modelKey: String? = nil,
        mediaSalt: String? = nil
    ) -> String {
        var hasher = SHA256()
        if let modelKey {
            hasher.update(data: Data(modelKey.utf8))
        }
        // Mix the VLM media salt after modelKey so VLM inputs with the same
        // text prefix but different images/videos land at different hashes.
        // Passing `nil` preserves the exact pre-existing text-only hash.
        if let mediaSalt {
            hasher.update(data: Data("|media:".utf8))
            hasher.update(data: Data(mediaSalt.utf8))
        }
        tokens.withUnsafeBufferPointer { buffer in
            let rawBuffer = UnsafeRawBufferPointer(buffer)
            hasher.update(bufferPointer: rawBuffer)
        }
        let digest = hasher.finalize()
        let fullHex = digest.map { String(format: "%02x", $0) }.joined()
        return String(fullHex.prefix(32))
    }

    // MARK: - Private Helpers

    /// Build the file URL for a given hash.
    private func safetensorsURL(for hash: String) -> URL {
        cacheDir.appendingPathComponent("\(hash).safetensors")
    }

    /// Execute a simple SQL statement with no bindings.
    private func executeSQL(_ sql: String) {
        guard let db else { return }
        var stmt: OpaquePointer?
        if sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK {
            sqlite3_step(stmt)
        }
        sqlite3_finalize(stmt)
    }

    /// Insert or replace a cache entry in the SQLite index.
    /// Caller MUST hold `lock` — the `_Locked` suffix is the convention
    /// for helpers that assume serialized access.
    private func _insertEntryLocked(hash: String, tokenCount: Int, fileSize: Int) {
        guard let db else { return }

        let sql = """
            INSERT OR REPLACE INTO cache_entries (hash, token_count, file_size)
            VALUES (?, ?, ?)
            """
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { return }

        hash.withCString { cStr in
            sqlite3_bind_text(stmt, 1, cStr, -1, nil)
            sqlite3_bind_int64(stmt, 2, Int64(tokenCount))
            sqlite3_bind_int64(stmt, 3, Int64(fileSize))
            sqlite3_step(stmt)
        }
        sqlite3_finalize(stmt)
    }

    /// Delete a single `cache_entries` row by hash. Caller MUST hold `lock`.
    ///
    /// Used when the on-disk file for an entry is removed (corrupt/truncated
    /// payload) so the row's `file_size` stops counting toward the eviction
    /// quota. Removing only the file would orphan the row and permanently
    /// inflate `SUM(file_size)`.
    private func _deleteEntryLocked(hash: String) {
        guard let db else { return }
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, "DELETE FROM cache_entries WHERE hash = ?", -1, &stmt, nil)
            == SQLITE_OK
        else { return }
        hash.withCString { cStr in
            sqlite3_bind_text(stmt, 1, cStr, -1, nil)
            sqlite3_step(stmt)
        }
        sqlite3_finalize(stmt)
    }

    /// Evict oldest entries until total cache size is under `maxSizeBytes`.
    /// Caller MUST hold `lock`.
    private func _evictIfNeededLocked() {
        guard let db else { return }

        // Query total size
        var totalSize: Int64 = 0
        var stmt: OpaquePointer?
        if sqlite3_prepare_v2(db, "SELECT COALESCE(SUM(file_size), 0) FROM cache_entries", -1, &stmt, nil) == SQLITE_OK {
            if sqlite3_step(stmt) == SQLITE_ROW {
                totalSize = sqlite3_column_int64(stmt, 0)
            }
        }
        sqlite3_finalize(stmt)

        guard totalSize > Int64(maxSizeBytes) else { return }

        // Fetch oldest entries (by creation time) to evict
        var toEvict: [(hash: String, fileSize: Int64)] = []
        var accumulated: Int64 = 0
        let excess = totalSize - Int64(maxSizeBytes)

        if sqlite3_prepare_v2(db, "SELECT hash, file_size FROM cache_entries ORDER BY created_at ASC", -1, &stmt, nil) == SQLITE_OK {
            while sqlite3_step(stmt) == SQLITE_ROW, accumulated < excess {
                if let cStr = sqlite3_column_text(stmt, 0) {
                    let hash = String(cString: cStr)
                    let size = sqlite3_column_int64(stmt, 1)
                    toEvict.append((hash: hash, fileSize: size))
                    accumulated += size
                }
            }
        }
        sqlite3_finalize(stmt)

        // Delete evicted entries and their files
        for entry in toEvict {
            let url = safetensorsURL(for: entry.hash)
            try? FileManager.default.removeItem(at: url)

            entry.hash.withCString { cStr in
                if sqlite3_prepare_v2(db, "DELETE FROM cache_entries WHERE hash = ?", -1, &stmt, nil) == SQLITE_OK {
                    sqlite3_bind_text(stmt, 1, cStr, -1, nil)
                    sqlite3_step(stmt)
                }
                sqlite3_finalize(stmt)
            }
        }
    }
}
