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
    public let storeSkips: Int
    /// Payload bytes currently counted against the configured disk quota.
    /// SQLite/WAL bookkeeping is intentionally excluded.
    public let currentPayloadBytes: Int
    /// Current logical cache-boundary count. A coordinator snapshot counts a
    /// linked KV + recurrent-companion pair as one entry.
    public let currentEntryCount: Int
    /// Logical cache boundaries removed by quota enforcement in this process.
    /// A linked KV + recurrent-companion pair increments this once.
    public let evictions: Int
    public let maxSizeBytes: Int
}

/// One indexed KV payload used by the coordinator's shared disk-quota pass.
/// `createdAt` is the entry's eviction recency timestamp. The SQLite column
/// retains its historical `created_at` name for on-disk schema compatibility.
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
enum DiskCacheIntegrityError: Error {
    case incompleteFile(String)
    case incompleteWrite(String)
    /// A record whose payload carries NaN/Inf. A cache row is only worth
    /// restoring when it reproduces a finite forward; a poisoned row restores
    /// a non-finite recurrent state or KV and every generation built on it is
    /// token 0 forever (osaurus#2652: 14 such rows, written once by a broken
    /// build, kept serving "!" on every later build until removed).
    case nonFinitePayload(String)
}

public final class DiskCache: @unchecked Sendable {

    private struct ValidatedFileFingerprint: Equatable {
        let size: Int
        let modificationDate: Date
    }

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

    /// Number of store operations that reused an already validated file.
    public private(set) var storeSkips: Int = 0
    /// Stores refused because the payload carried NaN/Inf (never persisted).
    public private(set) var refusedNonFiniteStores: Int = 0
    /// Fetches that found a NaN/Inf record on disk (removed, reported as a miss).
    public private(set) var refusedNonFiniteFetches: Int = 0

    /// The names of the float tensors in `arrays` that carry a non-finite
    /// value (at most `limit`), in key order. Integer and boolean tensors are
    /// skipped. Used on both sides of the disk boundary: a record is neither
    /// written nor restored when it is not entirely finite.
    static func nonFiniteTensorNames(in arrays: [String: MLXArray], limit: Int = 4) -> [String] {
        var names: [String] = []
        for key in arrays.keys.sorted() {
            guard let array = arrays[key], array.dtype.isFloatingPoint, array.size > 0 else { continue }
            let nonFinite = (1 - MLX.isFinite(array).asType(.int32)).sum().item(Int32.self)
            if nonFinite > 0 {
                names.append("\(key)(\(nonFinite))")
                if names.count >= limit { break }
            }
        }
        return names
    }

    /// Number of logical cache boundaries removed by quota enforcement.
    public private(set) var evictions: Int = 0

    /// Files successfully written or deserialized in this process. A matching
    /// fingerprint lets `store` avoid realizing and rewriting the same large
    /// prompt boundary after a cache hit, while a fresh process still validates
    /// an inherited file before it can take the fast path.
    private var validatedFiles: [String: ValidatedFileFingerprint] = [:]

    /// Trace-only identity of the most recent boundary written by this cache
    /// instance. Growing agent loops can store N tokens and immediately probe N
    /// tokens under a different hash on the next turn; counts alone hide where
    /// the prompt stopped being a prefix. Retain the IDs only while explicit
    /// cache tracing is enabled so the miss log can report the first divergent
    /// token without changing cache selection or normal-process memory use.
    private var traceLastStoredTokens: [Int]?
    private var traceLastStoredHash: String?

    /// Thread-safe copy of current disk-cache counters.
    public func snapshotStats() -> DiskCacheStats {
        lock.lock()
        defer { lock.unlock() }
        let usage = _payloadUsageLocked()
        return DiskCacheStats(
            hits: hits,
            misses: misses,
            stores: stores,
            storeSkips: storeSkips,
            currentPayloadBytes: usage.bytes,
            currentEntryCount: usage.entryCount,
            evictions: evictions,
            maxSizeBytes: maxSizeBytes)
    }

    // MARK: - Initialization

    /// Creates a new disk cache.
    ///
    /// - Parameters:
    ///   - cacheDir: Directory where safetensors files and the SQLite index are stored.
    ///   - maxSizeGB: Maximum cache size in gigabytes. Defaults to 10 GB.
    public convenience init(
        cacheDir: URL,
        maxSizeGB: Float = 10.0,
        modelKey: String? = nil
    ) {
        self.init(
            cacheDir: cacheDir,
            maxSizeBytes: Int(maxSizeGB * 1_073_741_824),
            modelKey: modelKey)
    }

    /// Exact-byte initializer used by deterministic quota tests and callers
    /// that already resolved a user-facing GiB limit to bytes.
    init(cacheDir: URL, maxSizeBytes: Int, modelKey: String? = nil) {
        self.cacheDir = cacheDir
        self.maxSizeBytes = maxSizeBytes
        self.modelKey = modelKey

        // Create cache directory if needed
        try? FileManager.default.createDirectory(at: cacheDir, withIntermediateDirectories: true)
        // Storage integrity at open: an interrupted store (crash, force-quit,
        // disk full) used to leave a partial `<hash>.safetensors` under its
        // FINAL name. `fetch` only checked existence, `loadArraysAndMetadata`
        // maps lazily, and the MLX reader's short-read exception is dropped
        // on the stream (`Load::eval_cpu` waits on the future without
        // `get()`), so the row restored as zero-filled KV / recurrent state
        // at a valid offset — silently. Stores now publish atomically
        // (temp → rename), so at open anything still named `*.tmp` is a dead
        // write, and any final-named file whose size is short of the
        // payload its own header declares is removed together with its row.
        Self.sweepUnpublishedAndIncompleteFiles(in: cacheDir)

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
        executeSQL("""
            CREATE INDEX IF NOT EXISTS idx_cache_entries_token_count
            ON cache_entries(token_count DESC)
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
    /// SQLite insert complete synchronously under the process-wide IO lock.
    ///
    /// - Parameters:
    ///   - tokens: Token IDs used to compute the cache key hash.
    ///   - arrays: Dictionary of named MLX arrays to persist.
    public func store(tokens: [Int], arrays: [String: MLXArray], mediaSalt: String? = nil) {
        store(
            tokens: tokens,
            arrays: arrays,
            mediaSalt: mediaSalt,
            enforceQuota: true)
    }

    /// Coordinator-only transactional store. The unified coordinator writes
    /// KV and recurrent companion payloads under one combined quota lock, so
    /// it defers this cache's standalone quota pass until the linked group is
    /// complete. Direct callers retain the historical per-cache quota above.
    func store(
        tokens: [Int],
        arrays: [String: MLXArray],
        mediaSalt: String? = nil,
        enforceQuota: Bool
    ) {
        let hash = DiskCache.hashTokens(tokens, modelKey: modelKey, mediaSalt: mediaSalt)
        let url = safetensorsURL(for: hash)
        let tokenCount = tokens.count
        if ProcessInfo.processInfo.environment["VMLX_CACHE_FETCH_TRACE"] == "1" {
            FileHandle.standardError.write(Data(
                ("[vmlx][cache/disk-store] count=\(tokenCount) hash=\(hash.prefix(12)) "
                    + "modelKey=\(modelKey ?? "nil") salt=\(mediaSalt.map { String($0.prefix(12)) } ?? "nil") "
                    + "keys=\(arrays.keys.sorted().prefix(6))\n").utf8))
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
        // A payload with NaN/Inf is not a cache entry, it is the failure the
        // cache would replay: refuse before touching the disk or the index.
        let nonFinite = Self.nonFiniteTensorNames(in: arrays)
        if !nonFinite.isEmpty {
            lock.lock()
            refusedNonFiniteStores += 1
            lock.unlock()
            FileHandle.standardError.write(Data(
                ("[vmlx][cache/disk-store] REFUSED non-finite payload count=\(tokenCount) "
                    + "hash=\(hash.prefix(12)) modelKey=\(modelKey ?? "nil") tensors=\(nonFinite)\n").utf8))
            return
        }

        MLXDiskCacheIOLock.shared.lock()
        defer { MLXDiskCacheIOLock.shared.unlock() }
        lock.lock()
        defer { lock.unlock() }
        stores += 1
        if ProcessInfo.processInfo.environment["VMLX_CACHE_FETCH_TRACE"] == "1" {
            traceLastStoredTokens = tokens
            traceLastStoredHash = hash
        }

        // A normal warm request fetches an L2 entry and then publishes the
        // same prompt boundary again at completion. Rewriting it used to
        // synchronize Metal, realize every cache tensor, write hundreds of MB,
        // and churn quota eviction even though the content-addressed key had
        // just been validated. Only skip files successfully loaded or written
        // by this process, and only while their size + mtime and SQLite row
        // still match. A fresh process, changed/corrupt file, missing index, or
        // format migration therefore takes the full write path and heals the
        // entry instead of preserving an assumption.
        if let validated = validatedFiles[hash],
           let current = _fileFingerprint(url: url),
           current == validated,
           let indexed = _entryMetadataLocked(hash: hash),
           indexed.tokenCount == tokenCount,
           indexed.fileSize == current.size,
           current.size > 0
        {
            storeSkips += 1
            _touchEntryLocked(hash: hash)
            if ProcessInfo.processInfo.environment["VMLX_CACHE_FETCH_TRACE"] == "1" {
                FileHandle.standardError.write(Data(
                    "[vmlx][cache/disk-store] SKIP validated hash=\(hash) count=\(tokenCount) bytes=\(current.size)\n".utf8))
            }
            return
        }
        // Pre-realize arrays under the lock so Metal work completes
        // before the writer hits the C++ save path AND no other thread
        // can interleave MLX ops on the same device during this window.
        // The explicit stream syncs are required for post-generation cache
        // stores: the decode loop uses asyncEval, and MLX's eval/safetensors
        // paths add command-buffer completion handlers. Entering those paths
        // while the default GPU stream still has a committed command buffer
        // can trip Metal's `_status < MTLCommandBufferStatusCommitted`
        // assertion. Sync before materializing, then again before/after save.
        let phaseTrace =
            ProcessInfo.processInfo.environment["VMLX_CACHE_FETCH_TRACE"] == "1"
        let tStart = Date()
        Stream.gpu.synchronize()
        MLX.eval(Array(arrays.values))
        Stream.gpu.synchronize()
        let tEval = Date()
        do {
            // Atomic publication: the row becomes visible under its content
            // hash only after every byte is on disk. A reader that races the
            // write, or a process that dies mid-write, never sees a partial
            // file under the final name (it sees a miss, or a `.tmp` swept
            // at the next open).
            let finalURL = url
            let url = Self.temporaryURL(for: finalURL)
            try? FileManager.default.removeItem(at: url)
            try save(arrays: arrays, metadata: ["format": "mlx"], url: url)
            Stream.gpu.synchronize()
            guard Self.isCompleteSafetensors(url: url) else {
                try? FileManager.default.removeItem(at: url)
                throw DiskCacheIntegrityError.incompleteWrite(finalURL.lastPathComponent)
            }
            try? FileManager.default.removeItem(at: finalURL)
            try FileManager.default.moveItem(at: url, to: finalURL)
            if phaseTrace {
                // A 27B ternary model spent ~25 s storing a single ~357 MB
                // boundary — about 14 MB/s, which is far too slow to be the
                // write itself, so the cost is either materializing the cache
                // or serializing it. Splitting the two says which, instead of
                // leaving it to inference.
                let tSave = Date()
                FileHandle.standardError.write(Data(
                    ("[vmlx][cache/store-phase] count=\(tokenCount) "
                        + "eval=\(tEval.timeIntervalSince(tStart))s "
                        + "save=\(tSave.timeIntervalSince(tEval))s\n").utf8))
            }

            let fileSize: Int
            if let attrs = try? FileManager.default.attributesOfItem(atPath: finalURL.path),
                let size = attrs[.size] as? Int
            {
                fileSize = size
            } else {
                fileSize = 0
            }

            _insertEntryLocked(hash: hash, tokenCount: tokenCount, fileSize: fileSize)
            if let fingerprint = _fileFingerprint(url: finalURL), fingerprint.size > 0 {
                validatedFiles[hash] = fingerprint
            } else {
                validatedFiles.removeValue(forKey: hash)
            }
            if enforceQuota {
                _evictIfNeededLocked()
            }
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
    /// - Parameters:
    ///   - tokens: Token IDs to look up.
    ///   - mediaSalt: Optional media fingerprint mixed into the cache key.
    ///   - touchRecency: Whether a successful fetch refreshes eviction
    ///     recency. Defaults to `true` for direct callers. CacheCoordinator
    ///     disables it while validating architecture-specific companion state,
    ///     then touches only a restore it actually accepts.
    ///   - countHit: Whether a successful fetch increments hit telemetry.
    ///     Defaults to `true` for direct callers. CacheCoordinator disables it
    ///     for candidate reads and records only an accepted restore.
    /// - Returns: The cached arrays if found, or `nil` on a miss.
    public func fetch(
        tokens: [Int],
        mediaSalt: String? = nil,
        touchRecency: Bool = true,
        countHit: Bool = true
    ) -> [String: MLXArray]? {
        let hash = DiskCache.hashTokens(tokens, modelKey: modelKey, mediaSalt: mediaSalt)
        let url = safetensorsURL(for: hash)

        MLXDiskCacheIOLock.shared.lock()
        defer { MLXDiskCacheIOLock.shared.unlock() }
        lock.lock()
        defer { lock.unlock() }

        guard FileManager.default.fileExists(atPath: url.path) else {
            validatedFiles.removeValue(forKey: hash)
            misses += 1
            if ProcessInfo.processInfo.environment["VMLX_CACHE_FETCH_TRACE"] == "1" {
                // A miss with a row/file present under a DIFFERENT hash is a
                // key-input mismatch (modelKey or salt), invisible without
                // printing what this lookup actually hashed.
                FileHandle.standardError.write(Data(
                    ("[vmlx][cache/disk-fetch] noFile count=\(tokens.count) "
                        + "hash=\(hash.prefix(12)) modelKey=\(modelKey ?? "nil") "
                        + "salt=\(mediaSalt.map { String($0.prefix(12)) } ?? "nil")\n").utf8))
                if let stored = traceLastStoredTokens,
                   stored.count == tokens.count,
                   let storedHash = traceLastStoredHash,
                   storedHash != hash
                {
                    var index = 0
                    while index < tokens.count, stored[index] == tokens[index] {
                        index += 1
                    }
                    let storedID = index < stored.count ? String(stored[index]) : "end"
                    let requestedID = index < tokens.count ? String(tokens[index]) : "end"
                    FileHandle.standardError.write(Data(
                        ("[vmlx][cache/key-divergence] count=\(tokens.count) "
                            + "storedHash=\(storedHash.prefix(12)) fetchHash=\(hash.prefix(12)) "
                            + "firstDiff=\(index) storedToken=\(storedID) fetchToken=\(requestedID)\n")
                            .utf8))
                }
            }
            return nil
        }

        do {
            // Fail closed on a short file BEFORE the lazy map: the reader's
            // short-read error never reaches the caller, so a truncated row
            // would otherwise restore as zeros at a valid offset.
            guard Self.isCompleteSafetensors(url: url) else {
                throw DiskCacheIntegrityError.incompleteFile(url.lastPathComponent)
            }
            let (arrays, _) = try loadArraysAndMetadata(url: url)
            // A record written before the store-side check (or by a build that
            // computed NaN) must never be restored: it is removed on first touch
            // so an installed user recovers on the next prefill without clearing
            // anything by hand.
            let nonFinite = Self.nonFiniteTensorNames(in: arrays)
            if !nonFinite.isEmpty {
                refusedNonFiniteFetches += 1
                throw DiskCacheIntegrityError.nonFinitePayload(nonFinite.joined(separator: ","))
            }
            if let fingerprint = _fileFingerprint(url: url), fingerprint.size > 0 {
                validatedFiles[hash] = fingerprint
            }
            if touchRecency {
                _touchEntryLocked(hash: hash)
            }
            if countHit {
                hits += 1
            }
            return arrays
        } catch {
            misses += 1
            validatedFiles.removeValue(forKey: hash)
            // A failed deserialize is almost always a corrupt safetensors
            // file — a 0-byte leftover from the pre-synchronous-store
            // bug, a partial write from an earlier crash, disk full
            // during flush, or a format-version mismatch after upgrade.
            // Log the specific error so operators can see the reason
            // instead of silently counting a cache miss, and delete the
            // corrupt file so the next turn doesn't retry and log the
            // same error on every fetch.
            FileHandle.standardError.write(Data(
                "[vmlx][cache/disk] fetch REFUSED entry at \(url.lastPathComponent) count=\(tokens.count): \(error) — removing\n"
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

    /// Record a deserialized candidate that CacheCoordinator accepted after
    /// validating any architecture-specific companion state.
    func recordAcceptedHit() {
        lock.lock()
        hits += 1
        lock.unlock()
    }

    /// Whether this process has already proved that the content-addressed
    /// entry is intact and matches its SQLite metadata.
    ///
    /// This is intentionally stricter than a filename/index existence check.
    /// A fresh process returns `false` until `fetch` deserializes the payload;
    /// a successful store also validates it. Stable system/tool boundaries can
    /// use this to avoid a second architecture rederive and serialization at
    /// the end of every warm request without trusting inherited or stale files.
    public func hasValidatedEntry(tokens: [Int], mediaSalt: String? = nil) -> Bool {
        let hash = DiskCache.hashTokens(tokens, modelKey: modelKey, mediaSalt: mediaSalt)
        let url = safetensorsURL(for: hash)
        lock.lock()
        defer { lock.unlock() }

        guard let validated = validatedFiles[hash],
              let current = _fileFingerprint(url: url),
              current == validated,
              current.size > 0,
              let indexed = _entryMetadataLocked(hash: hash),
              indexed.tokenCount == tokens.count,
              indexed.fileSize == current.size
        else {
            return false
        }
        return true
    }

    /// Whether a complete, self-consistent entry for exactly these tokens is on
    /// disk, regardless of which process wrote it.
    ///
    /// `hasValidatedEntry` deliberately trusts only what this process wrote or
    /// read, which is right for skipping a rewrite it can vouch for. It is too
    /// strict for deciding whether a boundary needs producing at all: after a
    /// restart, or on any turn that restored from cache, the entry is on disk
    /// but unvalidated, so the store path tries to rebuild it — and rebuilding
    /// means replaying the prefix through the model, which is cancellable and
    /// was observed dying as `rederive-failed ... CancellationError()` on a
    /// user Stop. The key is content-addressed over exactly these tokens, so an
    /// indexed row whose size matches the file on disk is the same bytes a
    /// rebuild would produce.
    public func hasDurableEntry(tokens: [Int], mediaSalt: String? = nil) -> Bool {
        let hash = DiskCache.hashTokens(tokens, modelKey: modelKey, mediaSalt: mediaSalt)
        let url = safetensorsURL(for: hash)
        lock.lock()
        defer { lock.unlock() }
        guard let current = _fileFingerprint(url: url), current.size > 0,
              let indexed = _entryMetadataLocked(hash: hash),
              indexed.tokenCount == tokens.count,
              indexed.fileSize == current.size
        else {
            return false
        }
        return true
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

    /// Refresh one indexed payload's eviction recency without decoding or
    /// rewriting its safetensors file. The file must still exist and its size
    /// and token count must match the index; an orphan or stale row is not
    /// allowed to become hot merely because its content hash still exists in
    /// SQLite. CacheCoordinator uses the explicit timestamp form to touch a
    /// linked KV + recurrent-companion group under one combined-quota critical
    /// section.
    @discardableResult
    func touchRecency(
        tokens: [Int],
        mediaSalt: String? = nil,
        at date: Date
    ) -> Bool {
        let hash = DiskCache.hashTokens(
            tokens, modelKey: modelKey, mediaSalt: mediaSalt)
        let url = safetensorsURL(for: hash)
        MLXDiskCacheIOLock.shared.lock()
        defer { MLXDiskCacheIOLock.shared.unlock() }
        lock.lock()
        defer { lock.unlock() }
        guard let current = _fileFingerprint(url: url),
              current.size > 0,
              let indexed = _entryMetadataLocked(hash: hash),
              indexed.tokenCount == tokens.count,
              indexed.fileSize == current.size
        else {
            validatedFiles.removeValue(forKey: hash)
            return false
        }
        return _touchEntryLocked(hash: hash, at: date)
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
            validatedFiles.removeValue(forKey: hash)
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
        storeSkips = 0
        evictions = 0
        validatedFiles.removeAll(keepingCapacity: true)
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

    /// Sibling temp name used while a row is being written. MLX's `save`
    /// chooses the container format from the extension, so the temp name
    /// must still end in `.safetensors`; the `.partial-` infix marks it as
    /// unpublished (never a content-hash filename, never fetched).
    static func temporaryURL(for finalURL: URL) -> URL {
        let stem = finalURL.deletingPathExtension().lastPathComponent
        return finalURL.deletingLastPathComponent()
            .appendingPathComponent("\(stem).partial-\(UUID().uuidString.prefix(8)).safetensors")
    }

    static func isUnpublishedName(_ name: String) -> Bool {
        name.contains(".partial-") && name.hasSuffix(".safetensors")
    }

    /// The byte offset one past the last tensor payload the file's own
    /// safetensors header declares (8-byte little-endian header length, JSON
    /// header, `data_offsets: [begin, end]` per tensor relative to the end
    /// of the header), or nil when the header itself cannot be read.
    static func declaredPayloadEnd(url: URL) -> Int? {
        guard let handle = try? FileHandle(forReadingFrom: url) else { return nil }
        defer { try? handle.close() }
        guard let lengthData = try? handle.read(upToCount: 8), lengthData.count == 8 else { return nil }
        let headerLength = lengthData.withUnsafeBytes { Int($0.load(as: UInt64.self).littleEndian) }
        guard headerLength > 0, headerLength < 256 * 1024 * 1024 else { return nil }
        guard let headerData = try? handle.read(upToCount: headerLength), headerData.count == headerLength,
            let header = try? JSONSerialization.jsonObject(with: headerData) as? [String: Any]
        else { return nil }
        var end = 0
        for (key, value) in header where key != "__metadata__" {
            guard let tensor = value as? [String: Any],
                let offsets = tensor["data_offsets"] as? [Any], offsets.count == 2,
                let last = (offsets[1] as? NSNumber)?.intValue
            else { return nil }
            end = max(end, last)
        }
        return 8 + headerLength + end
    }

    /// True when the file on disk holds every byte its header declares.
    static func isCompleteSafetensors(url: URL) -> Bool {
        guard let declared = declaredPayloadEnd(url: url),
            let attributes = try? FileManager.default.attributesOfItem(atPath: url.path),
            let size = (attributes[.size] as? NSNumber)?.intValue
        else { return false }
        return size >= declared
    }

    /// Remove dead temp files and incomplete final-named rows (with their
    /// index rows) from `cacheDir`. Header-only reads: cheap even for a
    /// multi-hundred-GB cache.
    static func sweepUnpublishedAndIncompleteFiles(in cacheDir: URL) {
        guard let names = try? FileManager.default.contentsOfDirectory(atPath: cacheDir.path) else { return }
        var removed = 0
        for name in names {
            let url = cacheDir.appendingPathComponent(name)
            if isUnpublishedName(name) {
                try? FileManager.default.removeItem(at: url); removed += 1
            } else if name.hasSuffix(".safetensors"), !isCompleteSafetensors(url: url) {
                try? FileManager.default.removeItem(at: url); removed += 1
                FileHandle.standardError.write(Data(
                    "[vmlx][cache/disk] removed incomplete row \(name) at open (short of its declared payload)\n".utf8))
            }
        }
        if removed > 0 {
            FileHandle.standardError.write(Data("[vmlx][cache/disk] integrity sweep removed \(removed) file(s)\n".utf8))
        }
    }

    private func _fileFingerprint(url: URL) -> ValidatedFileFingerprint? {
        guard let attributes = try? FileManager.default.attributesOfItem(atPath: url.path),
              let sizeNumber = attributes[.size] as? NSNumber,
              let modificationDate = attributes[.modificationDate] as? Date
        else { return nil }
        return ValidatedFileFingerprint(
            size: sizeNumber.intValue,
            modificationDate: modificationDate)
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

    private func _entryMetadataLocked(hash: String) -> (tokenCount: Int, fileSize: Int)? {
        guard let db else { return nil }
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(
            db,
            "SELECT token_count, file_size FROM cache_entries WHERE hash = ?",
            -1,
            &stmt,
            nil) == SQLITE_OK
        else { return nil }
        defer { sqlite3_finalize(stmt) }
        _ = hash.withCString { cStr in
            sqlite3_bind_text(stmt, 1, cStr, -1, nil)
        }
        guard sqlite3_step(stmt) == SQLITE_ROW else { return nil }
        return (
            tokenCount: Int(sqlite3_column_int64(stmt, 0)),
            fileSize: Int(sqlite3_column_int64(stmt, 1)))
    }

    /// Current indexed payload usage. Caller MUST hold `lock`.
    private func _payloadUsageLocked() -> (bytes: Int, entryCount: Int) {
        guard let db else { return (0, 0) }
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(
            db,
            "SELECT COALESCE(SUM(file_size), 0), COUNT(*) FROM cache_entries",
            -1,
            &stmt,
            nil) == SQLITE_OK
        else { return (0, 0) }
        defer { sqlite3_finalize(stmt) }
        guard sqlite3_step(stmt) == SQLITE_ROW else { return (0, 0) }
        return (
            bytes: max(0, Int(sqlite3_column_int64(stmt, 0))),
            entryCount: max(0, Int(sqlite3_column_int64(stmt, 1))))
    }

    /// Record logical evictions selected by the coordinator's linked KV +
    /// recurrent-companion quota pass. The coordinator counts groups before it
    /// removes either half, so one atomic pair increments this counter once.
    func recordQuotaEvictions(_ count: Int) {
        guard count > 0 else { return }
        lock.lock()
        evictions += count
        lock.unlock()
    }

    /// Refresh the existing eviction timestamp without replacing the row or
    /// rewriting the payload. Caller MUST hold `lock`.
    @discardableResult
    private func _touchEntryLocked(
        hash: String,
        at date: Date = Date()
    ) -> Bool {
        guard let db else { return false }
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(
            db,
            "UPDATE cache_entries SET created_at = ? WHERE hash = ?",
            -1,
            &stmt,
            nil) == SQLITE_OK
        else { return false }
        defer { sqlite3_finalize(stmt) }
        let julianDay = date.timeIntervalSince1970 / 86_400 + 2_440_587.5
        sqlite3_bind_double(stmt, 1, julianDay)
        _ = hash.withCString { cStr in
            sqlite3_bind_text(stmt, 2, cStr, -1, nil)
        }
        guard sqlite3_step(stmt) == SQLITE_DONE else { return false }
        return sqlite3_changes(db) > 0
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
            validatedFiles.removeValue(forKey: entry.hash)

            entry.hash.withCString { cStr in
                if sqlite3_prepare_v2(db, "DELETE FROM cache_entries WHERE hash = ?", -1, &stmt, nil) == SQLITE_OK {
                    sqlite3_bind_text(stmt, 1, cStr, -1, nil)
                    sqlite3_step(stmt)
                }
                sqlite3_finalize(stmt)
            }
        }
        evictions += toEvict.count
    }
}
