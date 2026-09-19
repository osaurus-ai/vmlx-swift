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
    /// Writes to the index that failed in this process after the files they
    /// describe were already on disk: a KV insert that lost to another
    /// connection's write lock, or a companion the index could not record.
    /// In both cases the files were removed again — unless an earlier record
    /// of the same companion already counts them — so nothing is left on disk
    /// uncounted; at worst the boundary is simply not cached.
    public let failedIndexWrites: Int
    /// Bytes of the logical cache boundaries counted in ``evictions``: what
    /// quota enforcement has really removed from disk in this process.
    public let evictedBytes: Int64
    /// Quota passes in this process that removed at least one boundary. A
    /// pass runs inline on every store; below the cap it is one SQL aggregate
    /// and is not counted here.
    public let quotaPasses: Int
    /// Wall time of the most recent quota pass that found the cache over its
    /// cap: reading the index rows, selecting victims and deleting them. 0
    /// until there has been one. Every store waits for its own pass.
    public let lastQuotaPassMs: Double
    /// Incremented once per quota pass that produced a pressure event, so a
    /// poller can tell a new event from the one it has already shown.
    public let pressureEventSeq: UInt64
    /// The most recent pressure event: the cap is too small for the
    /// conversation in progress. Advisory; nothing was refused.
    public let lastPressureEvent: DiskCachePressureEvent?

    init(
        hits: Int, misses: Int, stores: Int, storeSkips: Int,
        currentPayloadBytes: Int, currentEntryCount: Int,
        evictions: Int, maxSizeBytes: Int, failedIndexWrites: Int = 0,
        evictedBytes: Int64 = 0, quotaPasses: Int = 0, lastQuotaPassMs: Double = 0,
        pressureEventSeq: UInt64 = 0, lastPressureEvent: DiskCachePressureEvent? = nil
    ) {
        self.hits = hits
        self.misses = misses
        self.stores = stores
        self.storeSkips = storeSkips
        self.currentPayloadBytes = currentPayloadBytes
        self.currentEntryCount = currentEntryCount
        self.evictions = evictions
        self.maxSizeBytes = maxSizeBytes
        self.failedIndexWrites = failedIndexWrites
        self.evictedBytes = evictedBytes
        self.quotaPasses = quotaPasses
        self.lastQuotaPassMs = lastQuotaPassMs
        self.pressureEventSeq = pressureEventSeq
        self.lastPressureEvent = lastPressureEvent
    }

    /// The same counters over a different usage figure (the coordinator's
    /// directory-walk total on an index without the companion columns).
    func replacingUsage(currentPayloadBytes: Int, currentEntryCount: Int) -> DiskCacheStats {
        DiskCacheStats(
            hits: hits, misses: misses, stores: stores, storeSkips: storeSkips,
            currentPayloadBytes: currentPayloadBytes, currentEntryCount: currentEntryCount,
            evictions: evictions, maxSizeBytes: maxSizeBytes,
            failedIndexWrites: failedIndexWrites, evictedBytes: evictedBytes,
            quotaPasses: quotaPasses, lastQuotaPassMs: lastQuotaPassMs,
            pressureEventSeq: pressureEventSeq, lastPressureEvent: lastPressureEvent)
    }
}

/// One indexed KV payload used by the coordinator's shared disk-quota pass.
/// `createdAt` is the entry's eviction recency timestamp. The SQLite column
/// retains its historical `created_at` name for on-disk schema compatibility.
struct DiskCacheQuotaEntry: Sendable {
    let hash: String
    let bytes: Int64
    let createdAt: Date
    /// The recurrent companion linked to this row in a v2 index: its store
    /// key and the bytes of its payload + sidecar. `nil` / 0 when the row has
    /// none, when an older build wrote the row, or on a v1 index.
    var companionKey: String? = nil
    var companionBytes: Int64 = 0
    /// What the conversation-aware quota planner orders by, from a v2 index:
    /// the prefix length, whether the row is a stable root (`kind == 1`), and
    /// the conversation it belongs to (`chain_id`, NULL until one is assigned).
    var tokenCount: Int = 0
    var isStableRoot: Bool = false
    var chainId: String? = nil
}

/// A recurrent companion the v2 index counts but cannot attach to a KV row:
/// a sidecar from before `kv_hash` existed, or one whose KV row is absent.
struct DiskCacheLegacyCompanion: Sendable, Equatable {
    let key: String
    let bytes: Int64
    let modifiedAt: Date
}

/// What one COMMITTED import changed: all zero when the index already agreed
/// with the directory. An import that could not take the write lock or could
/// not commit changed nothing either, but is not this value —
/// ``DiskCache/reconcileCompanionAccounting(companions:companionDirectory:unindexedPayloadGuardAge:now:)`` returns nil for it,
/// so "nothing to do" and "did not run" cannot be mistaken for each other.
struct DiskCacheCompanionImportSummary: Sendable, Equatable {
    var rowsDeletedForMissingPayload = 0
    var linksWritten = 0
    var linksCleared = 0
    var legacyUpserted = 0
    var legacyDeleted = 0
    var unindexedPayloadsRemoved = 0

    var changedAnything: Bool { self != DiskCacheCompanionImportSummary() }
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

    private struct ValidatedRecord {
        let file: ValidatedFileFingerprint
        let layout: [String]
        let recurrentGeometry: RecurrentGeometry

        var hasRecurrentGeometry: Bool { recurrentGeometry != .incomplete }
    }

    private enum RecurrentGeometry {
        case absent, native, incomplete
    }

    /// Validate declared Mamba occupancy without loading state tensors. The
    /// same checks run on already-realized metadata at store/fetch and bounded
    /// integer reads on a cold disk query. Presence of `_state0` alone cannot
    /// certify missing PLE/GDN slots or an entirely missing declared layer.
    private static func recurrentGeometry(
        _ names: Set<String>, readInts: (String) -> [Int32]?
    ) -> RecurrentGeometry {
        var prefixes = Set<String>()
        for name in names {
            if name.hasPrefix("__layer_kind_"), name.hasSuffix("__"),
               readInts(name) == [TQDiskSerializer.LayerKind.mamba.rawValue] {
                prefixes.insert("mamba_" + name.dropFirst("__layer_kind_".count).dropLast(2))
            } else if name.hasPrefix("__cache_list_"), name.hasSuffix("_kind__"),
                      readInts(name) == [TQDiskSerializer.LayerKind.mamba.rawValue] {
                prefixes.insert("mamba_" + name.dropFirst("__cache_list_".count).dropLast("_kind__".count))
            } else if name.hasPrefix("mamba_"), let range = name.range(of: "_state") {
                prefixes.insert(String(name[..<range.lowerBound]))
            }
        }
        guard !prefixes.isEmpty else { return .absent }
        guard readInts(TQDiskSerializer.formatVersionKey) == [TQDiskSerializer.currentFormatVersion]
        else { return .incomplete }
        for prefix in prefixes {
            let suffix = String(prefix.dropFirst("mamba_".count))
            let kind = suffix.contains("_sub_")
                ? "__cache_list_\(suffix)_kind__" : "__layer_kind_\(suffix)__"
            guard readInts(kind) == [TQDiskSerializer.LayerKind.mamba.rawValue],
                  let slots = readInts("__\(prefix)_slots__"), slots.count == 1, slots[0] > 0,
                  let occupied = readInts("__\(prefix)_occupied__"), !occupied.isEmpty,
                  occupied.count <= Int(slots[0]), Set(occupied).count == occupied.count,
                  occupied.contains(0), occupied.allSatisfy({ $0 >= 0 && $0 < slots[0] }),
                  let offset = readInts("__\(prefix)_offset__"), offset.count == 1, offset[0] >= 0,
                  Set(names.filter { $0.hasPrefix("\(prefix)_state") })
                    == Set(occupied.map { "\(prefix)_state\($0)" })
            else { return .incomplete }
        }
        return .native
    }

    private static func recurrentGeometry(_ arrays: [String: MLXArray]) -> RecurrentGeometry {
        recurrentGeometry(Set(arrays.keys)) { name in
            guard let value = arrays[name], value.dtype == .int32,
                  value.ndim <= 1, value.size <= arrays.count
            else { return nil }
            return value.asArray(Int32.self)
        }
    }

    /// Header and tiny integer metadata only; never mmap/evaluate the cache
    /// tensors just to decide whether another full prefill is necessary.
    private static func recurrentGeometry(
        url: URL, header: (length: Int, tensors: [String: Any])
    ) -> RecurrentGeometry {
        guard let handle = try? FileHandle(forReadingFrom: url) else { return .incomplete }
        defer { try? handle.close() }
        var metadataBudget = 64 * 1024
        return recurrentGeometry(Set(header.tensors.keys)) { name in
            guard let tensor = header.tensors[name] as? [String: Any],
                  tensor["dtype"] as? String == "I32",
                  let shape = tensor["shape"] as? [Int], shape.count <= 1,
                  let offsets = tensor["data_offsets"] as? [Int], offsets.count == 2
            else { return nil }
            let count = shape.first ?? 1
            guard count >= 0, count <= header.tensors.count,
                  count <= metadataBudget / 4, offsets[0] >= 0,
                  offsets[1] >= offsets[0], offsets[1] - offsets[0] == count * 4,
                  offsets[0] <= Int.max - 8 - header.length
            else { return nil }
            metadataBudget -= count * 4
            do {
                try handle.seek(toOffset: UInt64(8 + header.length + offsets[0]))
                guard let bytes = try handle.read(upToCount: count * 4), bytes.count == count * 4
                else { return nil }
                return bytes.withUnsafeBytes { raw in
                    (0..<count).map {
                        Int32(littleEndian: raw.loadUnaligned(fromByteOffset: $0 * 4, as: Int32.self))
                    }
                }
            } catch { return nil }
        }
    }

    /// Token identity alone cannot justify retaining an older representation.
    /// Include tensor geometry and typed serializer metadata, without reading
    /// the large state tensors back to the CPU or retaining mapped arrays.
    private static func payloadLayout(_ arrays: [String: MLXArray]) -> [String] {
        arrays.keys.sorted().map { key in
            let array = arrays[key]!
            let metadata = key.hasPrefix("__") && array.dtype == .int32
                ? String(describing: array.asArray(Int32.self)) : ""
            return "\(key)|\(array.dtype)|\(array.shape)|\(metadata)"
        }
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

    /// `PRAGMA user_version` of `cache_index.db` after this connection's
    /// migration attempt. Below `DiskCacheIndexSchema.currentVersion` when the
    /// migration could not run (the index then keeps working as v1); above it
    /// when a newer build owns the schema. It is also 0 when the version could
    /// not be read at all — the database did not open, or `user_version` was
    /// unreadable under the migration's lock — so 0 means "treat as v1", not
    /// "the file says 0". Nothing branches on it; see `indexHasV2Columns`.
    let indexSchemaVersion: Int32

    /// Whether the v2 columns and `legacy_companions` are really present on
    /// this index. Callers that use them must check this, not the version.
    let indexHasV2Columns: Bool

    /// How long an ordinary index statement waits for another connection's
    /// write lock. Without a wait, an insert that loses to another model's
    /// connection fails after its payload is already published.
    static let defaultIndexBusyTimeoutMs: Int32 = 1000

    /// A published payload with no index row is removed by the import only
    /// once it is at least this old. A younger one may be another
    /// connection's store between its publish and its insert.
    static let defaultUnindexedPayloadGuardAge: TimeInterval = 600

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
    /// Index writes that failed after their files were on disk (see
    /// ``DiskCacheStats/failedIndexWrites``).
    public private(set) var failedIndexWrites: Int = 0
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
    /// The coordinator's quota pass, as ``DiskCacheStats`` reports it. Written
    /// by ``recordQuotaPass(evictedGroups:evictedBytes:milliseconds:event:)``.
    private var quotaEvictedBytes: Int64 = 0
    private var quotaPasses: Int = 0
    private var lastQuotaPassMs: Double = 0
    private var pressureEventSeq: UInt64 = 0
    private var lastPressureEvent: DiskCachePressureEvent?

    /// Files successfully written or deserialized in this process. A matching
    /// fingerprint lets `store` avoid realizing and rewriting the same large
    /// prompt boundary after a cache hit, while a fresh process still validates
    /// an inherited file before it can take the fast path.
    private var validatedFiles: [String: ValidatedRecord] = [:]

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
        return _statsLocked(bytes: usage.bytes, entryCount: usage.entryCount)
    }

    /// Caller MUST hold `lock`.
    private func _statsLocked(bytes: Int, entryCount: Int) -> DiskCacheStats {
        DiskCacheStats(
            hits: hits,
            misses: misses,
            stores: stores,
            storeSkips: storeSkips,
            currentPayloadBytes: bytes,
            currentEntryCount: entryCount,
            evictions: evictions,
            maxSizeBytes: maxSizeBytes,
            failedIndexWrites: failedIndexWrites,
            evictedBytes: quotaEvictedBytes,
            quotaPasses: quotaPasses,
            lastQuotaPassMs: lastQuotaPassMs,
            pressureEventSeq: pressureEventSeq,
            lastPressureEvent: lastPressureEvent)
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
    init(
        cacheDir: URL, maxSizeBytes: Int, modelKey: String? = nil,
        indexMigrationBusyTimeoutMs: Int32 = DiskCacheIndexSchema.defaultBusyTimeoutMs,
        indexBusyTimeoutMs: Int32 = DiskCache.defaultIndexBusyTimeoutMs
    ) {
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
            indexSchemaVersion = 0
            indexHasV2Columns = false
            return
        }

        // Enable WAL mode for better concurrent read performance
        Self.executeSQL(db, "PRAGMA journal_mode=WAL")

        // Create the index table
        for statement in DiskCacheIndexSchema.v1Statements {
            Self.executeSQL(db, statement)
        }

        // Bring the index to the current schema. A migration that cannot run
        // leaves a working v1 index; nothing below depends on the v2 columns.
        indexSchemaVersion = DiskCacheIndexSchema.migrate(
            db, busyTimeoutMs: indexMigrationBusyTimeoutMs)
        indexHasV2Columns = DiskCacheIndexSchema.hasV2Columns(
            db, busyTimeoutMs: indexMigrationBusyTimeoutMs)

        // The schema helpers put the connection back to "no wait" when they
        // finish. Every statement from here on waits a bounded time instead.
        sqlite3_busy_timeout(db, max(0, indexBusyTimeoutMs))
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
           current == validated.file,
           validated.hasRecurrentGeometry,
           Self.payloadLayout(arrays) == validated.layout,
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

            let insertResult = _insertEntryLocked(
                hash: hash, tokenCount: tokenCount, fileSize: fileSize)
            guard insertResult == SQLITE_DONE else {
                // The payload is published but has no row, so no quota pass
                // could ever see or evict it. Take it back rather than leak it.
                try? FileManager.default.removeItem(at: finalURL)
                validatedFiles.removeValue(forKey: hash)
                failedIndexWrites += 1
                FileHandle.standardError.write(Data(
                    ("[vmlx][cache/disk-store] index insert failed rc=\(insertResult) "
                        + "hash=\(hash.prefix(12)) — payload removed\n").utf8))
                return
            }
            if let fingerprint = _fileFingerprint(url: finalURL), fingerprint.size > 0 {
                validatedFiles[hash] = ValidatedRecord(
                    file: fingerprint, layout: Self.payloadLayout(arrays),
                    recurrentGeometry: Self.recurrentGeometry(arrays))
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
            // A row that outlives its payload keeps counting toward the quota
            // and keeps being offered as a candidate boundary. Most misses
            // have no row at all; look first (a read never waits on another
            // connection's write lock) and only then write.
            //
            // Only a definite "no such file" drops the row. `fileExists` is
            // also false for a payload that could not be examined, and a row
            // dropped for a payload that is still there hands that payload
            // to the import's sweep.
            if Self.pathState(at: url) == .missing, _entryMetadataLocked(hash: hash) != nil {
                _deleteEntryLocked(hash: hash)
            }
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

        // A payload the index does not name is counted by nothing and evicted
        // by nothing (a crash between publish and insert, or an external purge
        // that deleted the row and could not delete the file), so it is not
        // served either. It is NOT removed here: the insert may be in flight
        // on another connection. The import removes it once it is old enough
        // that no insert can still be pending. Without a database there is no
        // index to be missing from, and the file alone decides as it always has.
        if db != nil, _entryMetadataLocked(hash: hash) == nil {
            validatedFiles.removeValue(forKey: hash)
            misses += 1
            if ProcessInfo.processInfo.environment["VMLX_CACHE_FETCH_TRACE"] == "1" {
                FileHandle.standardError.write(Data(
                    ("[vmlx][cache/disk-fetch] noRow count=\(tokens.count) "
                        + "hash=\(hash.prefix(12)) — payload present, not indexed\n").utf8))
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
                validatedFiles[hash] = ValidatedRecord(
                    file: fingerprint, layout: Self.payloadLayout(arrays),
                    recurrentGeometry: Self.recurrentGeometry(arrays))
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
            // Drop the SQLite row too. Removing only the file orphans the
            // `cache_entries` row, whose `file_size` then permanently inflates
            // the `SUM(file_size)` eviction quota (unbounded on-disk growth and
            // premature eviction of live entries). The fetch path already holds
            // `lock`, so delete in-place — but only once the file really is
            // gone: a row is what keeps an undeletable file counted.
            if Self.removeCacheFile(at: url) {
                _deleteEntryLocked(hash: hash)
            }
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
    public func hasValidatedEntry(
        tokens: [Int], mediaSalt: String? = nil, requireNativeRecurrent: Bool = false
    ) -> Bool {
        let hash = DiskCache.hashTokens(tokens, modelKey: modelKey, mediaSalt: mediaSalt)
        let url = safetensorsURL(for: hash)
        lock.lock()
        defer { lock.unlock() }

        guard let validated = validatedFiles[hash],
              let current = _fileFingerprint(url: url),
              current == validated.file,
              validated.hasRecurrentGeometry,
              !requireNativeRecurrent || validated.recurrentGeometry == .native,
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
    public func hasDurableEntry(
        tokens: [Int], mediaSalt: String? = nil, requireNativeRecurrent: Bool = false
    ) -> Bool {
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
        // A legacy recurrent payload is readable for ordinary two-slot caches,
        // but cannot suppress producing the current declared representation.
        // This also covers a cold process before its first fetch. Read only the
        // safetensors header; do not map or realize model state to decide.
        if let validated = validatedFiles[hash], validated.file == current {
            return validated.hasRecurrentGeometry
                && (!requireNativeRecurrent || validated.recurrentGeometry == .native)
        }
        guard let header = Self.tensorHeader(url: url) else { return false }
        let geometry = Self.recurrentGeometry(url: url, header: header)
        return geometry != .incomplete && (!requireNativeRecurrent || geometry == .native)
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
        let sql = indexHasV2Columns
            ? """
                SELECT hash, file_size, created_at, companion_key, companion_bytes,
                       token_count, kind, chain_id
                FROM cache_entries
                """
            : "SELECT hash, file_size, created_at FROM cache_entries"
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            return []
        }
        defer { sqlite3_finalize(stmt) }

        while sqlite3_step(stmt) == SQLITE_ROW {
            guard let cHash = sqlite3_column_text(stmt, 0) else { continue }
            let hash = String(cString: cHash)
            let bytes = max(0, sqlite3_column_int64(stmt, 1))
            var entry = DiskCacheQuotaEntry(
                hash: hash,
                bytes: bytes,
                createdAt: Self.date(julianDay: sqlite3_column_double(stmt, 2)))
            // A row an older build wrote has NULL / 0 here: it is simply a
            // KV-only group.
            if indexHasV2Columns {
                entry.companionKey = sqlite3_column_text(stmt, 3).map { String(cString: $0) }
                entry.companionBytes = max(0, sqlite3_column_int64(stmt, 4))
                entry.tokenCount = max(0, Int(sqlite3_column_int64(stmt, 5)))
                entry.isStableRoot = sqlite3_column_int64(stmt, 6) == 1
                entry.chainId = sqlite3_column_text(stmt, 7).map { String(cString: $0) }
            }
            entries.append(entry)
        }
        return entries
    }

    // MARK: - Companion accounting (index schema v2)
    //
    // Recurrent companions live in `<cacheDir>/ssm_companion/`, outside this
    // cache, but their bytes count against the same quota. With a v2 index
    // they are accounted here so the quota and the stats poll are SQL
    // aggregates instead of a directory walk. Every method below is a no-op
    // (or reports "nothing") on an index without the v2 columns.

    /// Stop counting unlinked companions whose files the combined quota pass
    /// has removed. (`forgetCompanions` is the general form: it also clears
    /// a link.)
    func forgetLegacyCompanions(keys: Set<String>) {
        guard indexHasV2Columns, !keys.isEmpty else { return }
        lock.lock()
        defer { lock.unlock() }
        for key in keys {
            _runLocked("DELETE FROM legacy_companions WHERE key = ?", [.text(key)])
        }
    }

    func legacyCompanions() -> [DiskCacheLegacyCompanion] {
        guard indexHasV2Columns else { return [] }
        lock.lock()
        defer { lock.unlock() }
        return _legacyCompanionsLocked()
    }

    /// Bytes counted against the quota: KV payloads plus, on a v2 index, the
    /// companions linked to them and the unlinked ones.
    func usageBytes() -> Int64 {
        lock.lock()
        defer { lock.unlock() }
        return _combinedUsageLocked().bytes
    }

    /// A companion store finished writing (or re-validated) one entry. Link it
    /// to its KV row; when the row is absent — or another connection deletes
    /// it between this method's SELECT and its UPDATE — count it as unlinked.
    /// Writing the same entry again replaces its bytes. A row already linked
    /// to this key with these bytes is left alone, with no statement written;
    /// an unlinked companion is always written again, because that write is
    /// also what refreshes its recency.
    ///
    /// Returns nil on success, otherwise the SQLite result code of the
    /// statement that failed (in practice: another connection held the write
    /// lock past the busy timeout). The index then does not count these files
    /// as they are now, and the caller must not leave them on disk unless
    /// ``countedCompanionBytes(key:)`` shows an earlier record still covers
    /// them: files in neither table are never evicted.
    func recordCompanionFailureCode(
        kvHash: String, companionKey: String, bytes: Int64, modified: Date
    ) -> Int32? {
        guard indexHasV2Columns else { return nil }
        lock.lock()
        defer { lock.unlock() }
        let rc = _recordCompanionLocked(
            kvHash: kvHash, companionKey: companionKey, bytes: bytes, modified: modified)
        guard rc != SQLITE_DONE else { return nil }
        failedIndexWrites += 1
        return rc
    }

    private func _recordCompanionLocked(
        kvHash: String, companionKey: String, bytes: Int64, modified: Date
    ) -> Int32 {
        var rowExists = false
        var alreadyLinked = false
        _queryLocked(
            "SELECT companion_key, companion_bytes FROM cache_entries WHERE hash = ?",
            [.text(kvHash)]
        ) { stmt in
            rowExists = true
            if let cKey = sqlite3_column_text(stmt, 0) {
                alreadyLinked = String(cString: cKey) == companionKey
                    && sqlite3_column_int64(stmt, 1) == bytes
            }
        }

        if rowExists {
            var linked = alreadyLinked
            if !linked {
                let link = _linkCompanionLocked(
                    kvHash: kvHash, companionKey: companionKey, bytes: bytes)
                guard link.rc == SQLITE_DONE else { return link.rc }
                linked = link.changed
            }
            if linked {
                // It may have been counted as unlinked before its row existed.
                // If this DELETE fails the companion is counted twice until
                // the next import: an over-count, so not a failed record.
                var wasLegacy = false
                _queryLocked(
                    "SELECT 1 FROM legacy_companions WHERE key = ?", [.text(companionKey)]
                ) { _ in wasLegacy = true }
                if wasLegacy {
                    _runLocked("DELETE FROM legacy_companions WHERE key = ?", [.text(companionKey)])
                }
                return SQLITE_DONE
            }
            // The UPDATE changed no row: the row went away after the SELECT.
        }
        return _upsertLegacyCompanionLocked(key: companionKey, bytes: bytes, modified: modified)
    }

    /// Whether any companion is counted as unlinked. One statement; lets a
    /// caller skip hashing a companion key it would only use to look in that
    /// table. The table is often empty but not reliably so: an unlinked
    /// companion stays in it, on disk and counted, for as long as the total
    /// fits under the cap.
    func hasLegacyCompanions() -> Bool {
        guard indexHasV2Columns else { return false }
        lock.lock()
        defer { lock.unlock() }
        var found = false
        _queryLocked("SELECT 1 FROM legacy_companions LIMIT 1") { _ in found = true }
        return found
    }

    /// A companion that was recorded before its KV row existed is counted as
    /// unlinked, and unlinked companions are evicted first. When the row has
    /// arrived, move the companion onto it: one primary-key SELECT when there
    /// is nothing to adopt; otherwise the link and the removal of the
    /// unlinked entry in one transaction, with the bytes read inside it.
    /// Returns whether a companion was adopted. On any failure the companion
    /// simply stays unlinked, still counted.
    @discardableResult
    func adoptLegacyCompanion(kvHash: String, companionKey: String) -> Bool {
        guard indexHasV2Columns, let db else { return false }
        lock.lock()
        defer { lock.unlock() }

        var isLegacy = false
        _queryLocked("SELECT 1 FROM legacy_companions WHERE key = ?", [.text(companionKey)]) { _ in
            isLegacy = true
        }
        guard isLegacy else { return false }

        guard sqlite3_exec(db, "BEGIN IMMEDIATE", nil, nil, nil) == SQLITE_OK else { return false }
        let linkRC = _runLocked(
            """
            UPDATE cache_entries
            SET companion_key = ?1,
                companion_bytes = (SELECT bytes FROM legacy_companions WHERE key = ?1)
            WHERE hash = ?2 AND EXISTS (SELECT 1 FROM legacy_companions WHERE key = ?1)
            """,
            [.text(companionKey), .text(kvHash)])
        if linkRC == SQLITE_DONE, sqlite3_changes(db) > 0,
           _runLocked("DELETE FROM legacy_companions WHERE key = ?", [.text(companionKey)])
               == SQLITE_DONE,
           sqlite3_exec(db, "COMMIT", nil, nil, nil) == SQLITE_OK
        {
            return true
        }
        sqlite3_exec(db, "ROLLBACK", nil, nil, nil)
        return false
    }

    /// Companion bytes alone, linked and unlinked: what the companion store's
    /// own cap is compared with when it is written to directly.
    func companionUsageBytes() -> Int64 {
        guard indexHasV2Columns else { return 0 }
        lock.lock()
        defer { lock.unlock() }
        var bytes: Int64 = 0
        _queryLocked(
            """
            SELECT (SELECT COALESCE(SUM(companion_bytes), 0) FROM cache_entries)
                 + (SELECT COALESCE(SUM(bytes), 0) FROM legacy_companions)
            """
        ) { stmt in bytes = max(0, sqlite3_column_int64(stmt, 0)) }
        return bytes
    }

    /// Every counted companion, least recent first. A linked companion has
    /// its row's recency, an unlinked one its own; insertion order breaks
    /// ties (`julianday('now')` has millisecond resolution).
    func companionsOldestFirst() -> [DiskCacheLegacyCompanion] {
        guard indexHasV2Columns else { return [] }
        lock.lock()
        defer { lock.unlock() }
        var result: [DiskCacheLegacyCompanion] = []
        _queryLocked(
            """
            SELECT companion_key, companion_bytes,
                   (created_at - 2440587.5) * 86400.0 AS recency, rowid AS seq
            FROM cache_entries WHERE companion_key IS NOT NULL
            UNION ALL
            SELECT key, bytes, modified, rowid FROM legacy_companions
            ORDER BY recency ASC, seq ASC
            """
        ) { stmt in
            guard let cKey = sqlite3_column_text(stmt, 0) else { return }
            result.append(DiskCacheLegacyCompanion(
                key: String(cString: cKey),
                bytes: max(0, sqlite3_column_int64(stmt, 1)),
                modifiedAt: Date(timeIntervalSince1970: sqlite3_column_double(stmt, 2))))
        }
        return result
    }

    /// The bytes the index counts for this companion, linked or unlinked;
    /// nil when it names no such companion. A read, so it answers while
    /// another connection holds the write lock.
    func countedCompanionBytes(key: String) -> Int64? {
        guard indexHasV2Columns else { return nil }
        lock.lock()
        defer { lock.unlock() }
        var counted: Int64?
        _queryLocked(
            """
            SELECT companion_bytes FROM cache_entries WHERE companion_key = ?1
            UNION ALL
            SELECT bytes FROM legacy_companions WHERE key = ?1
            """,
            [.text(key)]
        ) { stmt in counted = max(counted ?? 0, sqlite3_column_int64(stmt, 0)) }
        return counted
    }

    /// Part of a companion could not be deleted: count what is left of it,
    /// wherever the index names it. If this write fails the record keeps its
    /// old, larger figure — an over-count.
    func correctCompanionBytes(key: String, bytes: Int64) {
        guard indexHasV2Columns else { return }
        lock.lock()
        defer { lock.unlock() }
        _runLocked(
            "UPDATE cache_entries SET companion_bytes = ? WHERE companion_key = ?",
            [.int(max(0, bytes)), .text(key)])
        _runLocked(
            "UPDATE legacy_companions SET bytes = ? WHERE key = ?",
            [.int(max(0, bytes)), .text(key)])
    }

    /// Forget which payloads this process has validated. After something
    /// outside this package deleted files, the fingerprints describe files
    /// that may be gone or replaced; the next store or fetch validates again.
    func forgetValidatedFiles() {
        lock.lock()
        validatedFiles.removeAll(keepingCapacity: true)
        lock.unlock()
    }

    /// Companion files are gone (the companion store's own eviction, or a
    /// failed write that left nothing). Stop counting them, linked or not.
    func forgetCompanions(keys: Set<String>) {
        guard indexHasV2Columns, !keys.isEmpty else { return }
        lock.lock()
        defer { lock.unlock() }
        for key in keys {
            _runLocked(
                "UPDATE cache_entries SET companion_key = NULL, companion_bytes = 0 WHERE companion_key = ?",
                [.text(key)])
            _runLocked("DELETE FROM legacy_companions WHERE key = ?", [.text(key)])
        }
    }

    /// The companion directory was emptied.
    func forgetAllCompanions() {
        guard indexHasV2Columns else { return }
        lock.lock()
        defer { lock.unlock() }
        _runLocked(
            "UPDATE cache_entries SET companion_key = NULL, companion_bytes = 0 WHERE companion_key IS NOT NULL")
        _runLocked("DELETE FROM legacy_companions")
    }

    /// Counters plus the whole-root usage in one critical section. A linked
    /// KV + companion pair is one logical entry; each unlinked companion is
    /// one more.
    func snapshotStatsIncludingCompanions() -> DiskCacheStats {
        lock.lock()
        defer { lock.unlock() }
        let usage = _combinedUsageLocked()
        return _statsLocked(bytes: Int(usage.bytes), entryCount: usage.entryCount)
    }

    /// Bring the companion columns and `legacy_companions` in line with what
    /// one walk of the companion directory found, and drop rows whose payload
    /// is gone. This is what adopts a directory written before the v2 index,
    /// and what repairs an index an older build wrote to since (its
    /// three-column INSERT OR REPLACE resets the companion columns).
    ///
    /// `companions` was listed BEFORE this method takes the index write lock,
    /// so it can be out of date by the time the transaction starts: a
    /// companion written and recorded in between is in the index and not in
    /// the list. Wherever the two disagree about a companion the index names,
    /// its two files are looked at again inside the transaction, and what is
    /// on disk then decides — a record is only dropped when its files are
    /// gone, and its bytes are corrected when they differ.
    ///
    /// It also removes payloads the index does not name, once they are older
    /// than `unindexedPayloadGuardAge` (as of `now`). Such a file cannot be
    /// adopted instead: a row needs the token count its writer hashed, and
    /// neither the content hash nor the payload carries it. `fetch` never
    /// serves it, so removing it loses nothing. A younger one is left alone:
    /// it may be another connection's store, between publishing the file and
    /// inserting the row. The cache root is a user setting and may hold files
    /// that are not this cache's, so the sweep works from an allow-list — see
    /// ``removeUnindexedPayloadsLocked(names:indexed:olderThan:now:summary:)``
    /// — and does not run at all under an index a newer build has claimed,
    /// or in a root that looks like a model bundle.
    ///
    /// One consequence worth knowing: deleting `cache_index.db` by hand
    /// leaves every payload without a row. None is served from then on, and
    /// each is removed by the next import once it is older than the guard
    /// age — consistent (they are unreachable), but not what "I only deleted
    /// the index" suggests.
    ///
    /// "Could not look" is never read as "not there". Everything is examined
    /// before anything is deleted or written, and a row is dropped, a link
    /// cleared or an unlinked record forgotten only on a definite "no such
    /// file". If the rows, the payload listing, a row's payload or a named
    /// companion cannot be examined for any other reason, the import abandons
    /// itself: nothing is deleted, nothing is committed, nil is returned.
    ///
    /// `companionDirectory` is the directory `companions` was listed from;
    /// the re-check looks there, so the two cannot disagree about where a
    /// companion lives. Defaults to this root's own companion directory.
    ///
    /// Idempotent: a second committed run over the same directory returns an
    /// all-zero summary. Returns nil when nothing was committed — no v2
    /// index, the write lock could not be taken within the busy timeout,
    /// something could not be examined, or the COMMIT failed — and the
    /// caller must then treat the import as not done and try again later.
    @discardableResult
    func reconcileCompanionAccounting(
        companions: [SSMCompanionQuotaEntry],
        companionDirectory walkedDirectory: URL? = nil,
        unindexedPayloadGuardAge: TimeInterval = DiskCache.defaultUnindexedPayloadGuardAge,
        now: Date = Date()
    ) -> DiskCacheCompanionImportSummary? {
        var summary = DiskCacheCompanionImportSummary()
        guard indexHasV2Columns, let db else { return nil }
        lock.lock()
        defer { lock.unlock() }

        guard sqlite3_exec(db, "BEGIN IMMEDIATE", nil, nil, nil) == SQLITE_OK else {
            FileHandle.standardError.write(Data(
                ("[vmlx][cache/disk-index] companion import skipped: "
                    + "\(String(cString: sqlite3_errmsg(db)))\n").utf8))
            return nil
        }
        // Nothing has been deleted or written when this is called.
        func abandon(_ reason: String) -> DiskCacheCompanionImportSummary? {
            FileHandle.standardError.write(Data(
                "[vmlx][cache/disk-index] companion import abandoned, \(reason)\n".utf8))
            sqlite3_exec(db, "ROLLBACK", nil, nil, nil)
            return nil
        }

        struct Row {
            let hash: String
            let companionKey: String?
            let companionBytes: Int64
        }
        var rows: [Row] = []
        var sawUnreadableColumn = false
        let rowsWereRead = _queryLocked(
            "SELECT hash, companion_key, companion_bytes FROM cache_entries"
        ) { stmt in
            // `sqlite3_column_text` returns NULL for SQL NULL and for an
            // allocation failure; only the type, read first, tells them
            // apart. A failed read of a hash would make its payload look
            // unindexed. A row whose hash IS NULL (the column is a TEXT
            // primary key, which SQLite lets be NULL) names no file: it can
            // neither hide a payload from the sweep nor expose one to it.
            let hashIsNull = sqlite3_column_type(stmt, 0) == SQLITE_NULL
            let keyIsNull = sqlite3_column_type(stmt, 1) == SQLITE_NULL
            let cHash = sqlite3_column_text(stmt, 0)
            let cKey = sqlite3_column_text(stmt, 1)
            if (cHash == nil && !hashIsNull) || (cKey == nil && !keyIsNull) {
                sawUnreadableColumn = true
                return
            }
            guard let cHash else { return }
            rows.append(Row(
                hash: String(cString: cHash),
                companionKey: cKey.map { String(cString: $0) },
                companionBytes: sqlite3_column_int64(stmt, 2)))
        }
        // Everything below treats "not in `rows`" as "not indexed", and the
        // payload sweep deletes on it. A read that failed part-way must not
        // be mistaken for a short index.
        guard rowsWereRead, !sawUnreadableColumn else {
            return abandon("rows unreadable: \(String(cString: sqlite3_errmsg(db)))")
        }

        // The payload listing, complete, before anything is deleted — or the
        // reason there is no sweep this time.
        var payloadNames: [String] = []
        var sweepSkipped: String?
        if indexSchemaVersion > DiskCacheIndexSchema.currentVersion {
            sweepSkipped =
                "index schema version \(indexSchemaVersion) is newer than this build's "
                + "\(DiskCacheIndexSchema.currentVersion); its payloads may be named another way"
        } else {
            do {
                payloadNames = try FileManager.default.contentsOfDirectory(atPath: cacheDir.path)
            } catch {
                return abandon(
                    "cache root could not be listed: \(error.localizedDescription)")
            }
            if let marker = Self.modelBundleMarker(in: payloadNames) {
                sweepSkipped = "cache root holds \(marker), so it looks like a model bundle"
            }
        }

        // Which rows have lost their payload. `fileExists` cannot answer
        // that: it is false for EACCES and every other failure too, and a
        // row dropped for a payload that is still there leaves that payload
        // for the sweep of the next import.
        var liveRows: [Row] = []
        var rowsWithoutPayload: [Row] = []
        for row in rows {
            let url = safetensorsURL(for: row.hash)
            switch Self.pathState(at: url) {
            case .missing:
                rowsWithoutPayload.append(row)
            case .regularFile, .notRegularFile:
                liveRows.append(row)
            case .unreadable(let code):
                return abandon(
                    "payload \(url.lastPathComponent) could not be examined: "
                        + String(cString: strerror(code)))
            }
        }

        // Close the gap between the caller's walk and this transaction (see
        // the doc comment). In the steady state the walk and the index agree
        // and nothing is looked at twice.
        let recordedLegacy = _legacyCompanionsLocked()
        var onDisk = Dictionary(
            companions.map { ($0.hash, $0) }, uniquingKeysWith: { first, _ in first })
        struct Named {
            let key: String
            let kvHash: String?
            let bytes: Int64
        }
        var named: [Named] = rows.compactMap { row in
            row.companionKey.map { Named(key: $0, kvHash: row.hash, bytes: row.companionBytes) }
        }
        named += recordedLegacy.map { Named(key: $0.key, kvHash: nil, bytes: $0.bytes) }
        let companionDirectory = walkedDirectory ?? self.companionDirectory
        for record in named {
            let walked = onDisk[record.key]
            if let walked, walked.bytes == record.bytes { continue }
            switch SSMCompanionDiskStore.publishedEntryState(
                key: record.key, in: companionDirectory)
            {
            case .present(let bytes, let modifiedAt):
                onDisk[record.key] = SSMCompanionQuotaEntry(
                    hash: record.key, kvHash: walked?.kvHash ?? record.kvHash,
                    bytes: bytes, modifiedAt: modifiedAt)
            case .absent:
                onDisk[record.key] = nil
            case .unreadable(let code):
                return abandon(
                    "companion \(record.key.prefix(12)) could not be examined: "
                        + String(cString: strerror(code)))
            }
        }
        let reconciled = Array(onDisk.values)

        // Everything has been looked at. From here on things are removed.
        if let sweepSkipped {
            FileHandle.standardError.write(Data(
                "[vmlx][cache/disk-index] payload sweep skipped: \(sweepSkipped)\n".utf8))
        } else {
            removeUnindexedPayloadsLocked(
                names: payloadNames, indexed: Set(rows.map(\.hash)),
                olderThan: unindexedPayloadGuardAge, now: now, summary: &summary)
        }

        for row in rowsWithoutPayload {
            _runLocked("DELETE FROM cache_entries WHERE hash = ?", [.text(row.hash)])
            validatedFiles.removeValue(forKey: row.hash)
            summary.rowsDeletedForMissingPayload += 1
        }

        // One companion per row. Keys are content-addressed over the same
        // tokens as the row hash, so a second claimant cannot arise from this
        // build; if one exists anyway it is counted as unlinked.
        let liveHashes = Set(liveRows.map(\.hash))
        var linkByHash: [String: SSMCompanionQuotaEntry] = [:]
        var unlinked: [SSMCompanionQuotaEntry] = []
        for companion in reconciled.sorted(by: { $0.hash < $1.hash }) {
            if let kvHash = companion.kvHash, liveHashes.contains(kvHash),
               linkByHash[kvHash] == nil
            {
                linkByHash[kvHash] = companion
            } else {
                unlinked.append(companion)
            }
        }

        for row in liveRows {
            if let companion = linkByHash[row.hash] {
                let bytes = max(0, companion.bytes)
                if row.companionKey != companion.hash || row.companionBytes != bytes {
                    _linkCompanionLocked(
                        kvHash: row.hash, companionKey: companion.hash, bytes: bytes)
                    summary.linksWritten += 1
                }
            } else if row.companionKey != nil || row.companionBytes != 0 {
                _runLocked(
                    "UPDATE cache_entries SET companion_key = NULL, companion_bytes = 0 WHERE hash = ?",
                    [.text(row.hash)])
                summary.linksCleared += 1
            }
        }

        let recorded = Dictionary(
            recordedLegacy.map { ($0.key, $0) }, uniquingKeysWith: { first, _ in first })
        let unlinkedKeys = Set(unlinked.map(\.hash))
        for key in recorded.keys where !unlinkedKeys.contains(key) {
            _runLocked("DELETE FROM legacy_companions WHERE key = ?", [.text(key)])
            summary.legacyDeleted += 1
        }
        for companion in unlinked {
            let bytes = max(0, companion.bytes)
            if let existing = recorded[companion.hash], existing.bytes == bytes { continue }
            _upsertLegacyCompanionLocked(
                key: companion.hash, bytes: bytes, modified: companion.modifiedAt)
            summary.legacyUpserted += 1
        }

        if sqlite3_exec(db, "COMMIT", nil, nil, nil) != SQLITE_OK {
            FileHandle.standardError.write(Data(
                ("[vmlx][cache/disk-index] companion import could not commit: "
                    + "\(String(cString: sqlite3_errmsg(db)))\n").utf8))
            sqlite3_exec(db, "ROLLBACK", nil, nil, nil)
            return nil
        }
        return summary
    }

    /// Where the coordinator keeps this root's recurrent companions.
    static let companionDirectoryName = "ssm_companion"

    var companionDirectory: URL {
        cacheDir.appendingPathComponent(Self.companionDirectoryName)
    }

    /// The payload half of ``reconcileCompanionAccounting``. Caller holds
    /// `lock` and the index write lock, so no row can appear between reading
    /// `indexed` and the last removal here. `names` is the caller's complete
    /// listing of the root.
    ///
    /// The root may hold files that are not this cache's — the directory is
    /// a user setting — and a wrong removal there is somebody's model. So
    /// this is an allow-list, and whatever is in doubt stays:
    ///
    /// - the name is exactly a published payload's
    ///   (``isPublishedPayloadName(_:)``); `.partial-` names never are;
    /// - the entry is a regular file by `lstat` — not a directory, and not a
    ///   symlink, which is never followed;
    /// - its modification date could be read, is not in the future, and is
    ///   at least the guard age back;
    /// - it goes by `unlink`, which removes that one name and cannot descend
    ///   into a directory that took the name since the `lstat`.
    ///
    /// The process-wide IO lock is not taken (it orders before `lock`).
    /// This instance's `fetch` refuses a payload without a row, but an older
    /// build, or an instance that could not open the index, may have one of
    /// these files mapped. That is harmless: unlinking a mapped file is safe,
    /// the mapping keeps its pages until it is released. What remains is a
    /// store re-publishing the very same hash between the age check and the
    /// removal; its insert then names a missing file — an over-count the
    /// next fetch of that hash clears.
    private func removeUnindexedPayloadsLocked(
        names: [String], indexed: Set<String>, olderThan guardAge: TimeInterval, now: Date,
        summary: inout DiskCacheCompanionImportSummary
    ) {
        for name in names where Self.isPublishedPayloadName(name) {
            let hash = String(name.dropLast(Self.payloadSuffix.count))
            guard !indexed.contains(hash) else { continue }
            let url = cacheDir.appendingPathComponent(name)
            // Stat now, not at listing time: the guard is about this instant.
            guard case .regularFile(_, let modified) = Self.pathState(at: url),
                  modified <= now
            else { continue }
            let age = now.timeIntervalSince(modified)
            guard age >= guardAge else { continue }
            validatedFiles.removeValue(forKey: hash)
            let failure = Self.unlinkFile(at: url)
            if failure == 0 {
                summary.unindexedPayloadsRemoved += 1
                FileHandle.standardError.write(Data(
                    ("[vmlx][cache/disk-index] removed unindexed payload \(name) "
                        + "ageSeconds=\(Int(age))\n").utf8))
            } else {
                FileHandle.standardError.write(Data(
                    ("[vmlx][cache/disk-index] could not remove unindexed payload \(name): "
                        + "\(String(cString: strerror(failure)))\n").utf8))
            }
        }
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
    ///
    /// `removedCompanions` are the companion keys whose files the pass has
    /// already removed (files before rows). Returns the hashes that are gone.
    ///
    /// A row is only dropped once its payload is: a payload that could not be
    /// deleted keeps its row, so it stays counted and is tried again by the
    /// next pass (once per pass — nothing here loops). Either way the row's
    /// companion is counted exactly while its files exist: a removed one
    /// leaves the accounting with the row or is unlinked from a row that
    /// stays, and one that was not removed outlives its row as an unlinked
    /// companion.
    @discardableResult
    func removeQuotaEntries(
        hashes: Set<String>, removedCompanions: Set<String> = []
    ) -> Set<String> {
        guard !hashes.isEmpty else { return [] }
        MLXDiskCacheIOLock.shared.lock()
        defer { MLXDiskCacheIOLock.shared.unlock() }
        lock.lock()
        defer { lock.unlock() }

        var removed = Set<String>()
        for hash in hashes {
            let payloadGone = Self.removeCacheFile(at: safetensorsURL(for: hash))
            validatedFiles.removeValue(forKey: hash)

            var companionRemoved = false
            if indexHasV2Columns {
                _queryLocked(
                    "SELECT companion_key FROM cache_entries WHERE hash = ?", [.text(hash)]
                ) { stmt in
                    if let cKey = sqlite3_column_text(stmt, 0) {
                        companionRemoved = removedCompanions.contains(String(cString: cKey))
                    }
                }
            }
            if payloadGone {
                _deleteEntryLocked(hash: hash, keepCompanionCounted: !companionRemoved)
                removed.insert(hash)
            } else if companionRemoved {
                _runLocked(
                    "UPDATE cache_entries SET companion_key = NULL, companion_bytes = 0 WHERE hash = ?",
                    [.text(hash)])
            }
        }
        return removed
    }

    /// Remove one cache file. Returns whether it is gone afterwards (a file
    /// that was never there is gone). "Gone" is a definite "no such file": a
    /// path that cannot be examined counts as still there, because its row
    /// is what keeps it counted, and the caller keeps it.
    ///
    /// A file that is still there is reported under its own tag — every
    /// `[vmlx][cache/disk-quota]` line is a pass summary beginning
    /// `before= after= max=`, and a log parser relies on that — and once per
    /// path per process: a file that can never be deleted is tried again by
    /// every over-cap store.
    static func removeCacheFile(at url: URL) -> Bool {
        do {
            try FileManager.default.removeItem(at: url)
            return true
        } catch {
            if pathState(at: url) == .missing { return true }
            let firstReport = reportedDeleteFailures.withLock { $0.insert(url.path).inserted }
            if firstReport {
                FileHandle.standardError.write(Data(
                    ("[vmlx][cache/disk-delete] failed path=\(url.path) "
                        + "error=\(error.localizedDescription) — row kept\n").utf8))
            }
            return false
        }
    }

    /// Paths ``removeCacheFile(at:)`` has already reported in this process.
    private static let reportedDeleteFailures = OSAllocatedUnfairLock(initialState: Set<String>())

    /// Remove all cached entries and safetensors files.
    public func clear() {
        MLXDiskCacheIOLock.shared.lock()
        defer { MLXDiskCacheIOLock.shared.unlock() }

        // Delete all SQLite entries
        lock.lock()
        defer { lock.unlock() }

        // This cache does not own the companion files. Rows that carried one
        // hand it to the unlinked list so its bytes stay counted until the
        // companion store clears or evicts it.
        if indexHasV2Columns {
            _runLocked(Self.moveLinkedCompanionsToLegacySQL + " WHERE companion_key IS NOT NULL")
        }
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
        failedIndexWrites = 0
        evictions = 0
        quotaEvictedBytes = 0
        quotaPasses = 0
        lastQuotaPassMs = 0
        pressureEventSeq = 0
        lastPressureEvent = nil
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

    static let payloadSuffix = ".safetensors"

    /// Whether `name` is exactly what ``safetensorsURL(for:)`` produces for a
    /// hash from ``hashTokens(_:modelKey:mediaSalt:)``: 32 lowercase hex
    /// digits and the suffix, nothing else. This is the only test of "is
    /// this file ours" that anything deleting from a listing of the root may
    /// use. Uppercase hex, another length, a `.partial-` name, a model's
    /// `model-00001-of-00008.safetensors` — none of them is ours.
    static func isPublishedPayloadName(_ name: String) -> Bool {
        guard name.hasSuffix(payloadSuffix) else { return false }
        let stem = name.utf8.dropLast(payloadSuffix.utf8.count)
        return stem.count == 32
            && stem.allSatisfy { (UInt8(ascii: "0")...UInt8(ascii: "9")).contains($0)
                || (UInt8(ascii: "a")...UInt8(ascii: "f")).contains($0) }
    }

    /// Files that mark a directory as a model bundle. The host's own purge
    /// tool refuses such a root; so does every sweep here.
    static let modelBundleMarkers = ["config.json", "jang_config.json"]

    /// The first bundle marker among a directory's entry names, if any.
    static func modelBundleMarker(in names: [String]) -> String? {
        modelBundleMarkers.first(where: names.contains)
    }

    /// What one `lstat` says about a path. `missing` is a definite ENOENT;
    /// every other failure is `unreadable`, which says nothing about whether
    /// the file is there. (`FileManager.fileExists` folds the two together.)
    /// A symlink is `notRegularFile`: it is never followed.
    enum PathState: Equatable {
        case regularFile(size: Int64, modified: Date)
        case notRegularFile
        case missing
        case unreadable(errno: Int32)
    }

    static func pathState(at url: URL) -> PathState {
        var info = stat()
        let rc = url.withUnsafeFileSystemRepresentation { path -> Int32 in
            guard let path else {
                errno = EINVAL
                return -1
            }
            return lstat(path, &info)
        }
        guard rc == 0 else {
            let code = errno
            return code == ENOENT ? .missing : .unreadable(errno: code)
        }
        guard (info.st_mode & S_IFMT) == S_IFREG else { return .notRegularFile }
        #if canImport(Darwin)
        let time = info.st_mtimespec
        #else
        let time = info.st_mtim
        #endif
        return .regularFile(
            size: Int64(info.st_size),
            modified: Date(
                timeIntervalSince1970: TimeInterval(time.tv_sec)
                    + TimeInterval(time.tv_nsec) / 1_000_000_000))
    }

    /// `unlink(2)`: removes that one name, never the target of a link, and
    /// fails on a directory instead of descending into it. Returns 0, or
    /// the errno.
    static func unlinkFile(at url: URL) -> Int32 {
        let rc = url.withUnsafeFileSystemRepresentation { path -> Int32 in
            guard let path else {
                errno = EINVAL
                return -1
            }
            return unlink(path)
        }
        return rc == 0 ? 0 : errno
    }

    /// The byte offset one past the last tensor payload the file's own
    /// safetensors header declares (8-byte little-endian header length, JSON
    /// header, `data_offsets: [begin, end]` per tensor relative to the end
    /// of the header), or nil when the header itself cannot be read.
    static func declaredPayloadEnd(url: URL) -> Int? {
        guard let header = tensorHeader(url: url) else { return nil }
        var end = 0
        for (key, value) in header.tensors where key != "__metadata__" {
            guard let tensor = value as? [String: Any],
                let offsets = tensor["data_offsets"] as? [Any], offsets.count == 2,
                let last = (offsets[1] as? NSNumber)?.intValue
            else { return nil }
            end = max(end, last)
        }
        return 8 + header.length + end
    }

    private static func tensorHeader(url: URL) -> (length: Int, tensors: [String: Any])? {
        guard let handle = try? FileHandle(forReadingFrom: url) else { return nil }
        defer { try? handle.close() }
        guard let lengthData = try? handle.read(upToCount: 8), lengthData.count == 8 else { return nil }
        let headerLength = lengthData.withUnsafeBytes { Int($0.load(as: UInt64.self).littleEndian) }
        guard headerLength > 0, headerLength < 256 * 1024 * 1024 else { return nil }
        guard let headerData = try? handle.read(upToCount: headerLength), headerData.count == headerLength,
            let header = try? JSONSerialization.jsonObject(with: headerData) as? [String: Any]
        else { return nil }
        return (headerLength, header)
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
    ///
    /// The root may hold files that are not this cache's. Only regular files
    /// are considered (by `lstat`: never a directory, which cannot be read
    /// as a safetensors file and used to go recursively, and never a
    /// symlink), removal is by `unlink`, and a root that looks like a model
    /// bundle is left alone entirely — a shard that is still downloading is
    /// an "incomplete safetensors file" too.
    static func sweepUnpublishedAndIncompleteFiles(in cacheDir: URL) {
        guard let names = try? FileManager.default.contentsOfDirectory(atPath: cacheDir.path) else { return }
        if let marker = modelBundleMarker(in: names) {
            FileHandle.standardError.write(Data(
                ("[vmlx][cache/disk] integrity sweep skipped: cache root holds \(marker), "
                    + "so it looks like a model bundle\n").utf8))
            return
        }
        var removed = 0
        for name in names where name.hasSuffix(payloadSuffix) {
            let url = cacheDir.appendingPathComponent(name)
            guard case .regularFile = pathState(at: url) else { continue }
            if isUnpublishedName(name) {
                if unlinkFile(at: url) == 0 { removed += 1 }
            } else if !isCompleteSafetensors(url: url) {
                guard unlinkFile(at: url) == 0 else { continue }
                removed += 1
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
        Self.executeSQL(db, sql)
    }

    /// Static form for `init`, which runs before every stored property is
    /// set and so cannot call an instance method.
    private static func executeSQL(_ db: OpaquePointer?, _ sql: String) {
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
    ///
    /// Returns the SQLite result: `SQLITE_DONE` when the row is written. With
    /// no database at all there is no index to fall out of step with, and the
    /// call reports success as it always has.
    ///
    /// On a v2 index the row is upserted, not replaced: REPLACE deletes the
    /// old row first, which would reset `companion_key` / `companion_bytes`
    /// to their defaults while the companion files are still on disk.
    @discardableResult
    private func _insertEntryLocked(hash: String, tokenCount: Int, fileSize: Int) -> Int32 {
        guard db != nil else { return SQLITE_DONE }
        if indexHasV2Columns {
            return _runLocked(
                """
                INSERT INTO cache_entries (hash, token_count, file_size, model_key)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(hash) DO UPDATE SET
                    token_count = excluded.token_count,
                    file_size = excluded.file_size,
                    created_at = julianday('now'),
                    model_key = excluded.model_key
                """,
                [
                    .text(hash), .int(Int64(tokenCount)), .int(Int64(fileSize)),
                    modelKey.map(SQLValue.text) ?? .null,
                ])
        }
        return _runLocked(
            """
            INSERT OR REPLACE INTO cache_entries (hash, token_count, file_size)
            VALUES (?, ?, ?)
            """,
            [.text(hash), .int(Int64(tokenCount)), .int(Int64(fileSize))])
    }

    // MARK: - SQLite statement helpers

    private enum SQLValue {
        case text(String)
        case int(Int64)
        case real(Double)
        case null
    }

    /// SQLite copies the bytes before `sqlite3_bind_text` returns.
    private static let sqliteTransient = unsafeBitCast(-1, to: sqlite3_destructor_type.self)

    private func _prepareLocked(_ sql: String, _ values: [SQLValue]) -> OpaquePointer? {
        guard let db else { return nil }
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            sqlite3_finalize(stmt)
            return nil
        }
        for (offset, value) in values.enumerated() {
            let index = Int32(offset + 1)
            switch value {
            case .text(let text): sqlite3_bind_text(stmt, index, text, -1, Self.sqliteTransient)
            case .int(let int): sqlite3_bind_int64(stmt, index, int)
            case .real(let real): sqlite3_bind_double(stmt, index, real)
            case .null: sqlite3_bind_null(stmt, index)
            }
        }
        return stmt
    }

    /// Run one statement to completion. Returns `SQLITE_DONE` on success, the
    /// failing result code otherwise. Caller MUST hold `lock`.
    @discardableResult
    private func _runLocked(_ sql: String, _ values: [SQLValue] = []) -> Int32 {
        guard let db else { return SQLITE_MISUSE }
        guard let stmt = _prepareLocked(sql, values) else { return sqlite3_errcode(db) }
        defer { sqlite3_finalize(stmt) }
        var rc = sqlite3_step(stmt)
        while rc == SQLITE_ROW { rc = sqlite3_step(stmt) }
        return rc
    }

    /// Visit every result row. Returns whether the statement ran to its end;
    /// false means `row` may have seen only some of the rows, or none.
    /// Caller MUST hold `lock`.
    @discardableResult
    private func _queryLocked(
        _ sql: String, _ values: [SQLValue] = [], _ row: (OpaquePointer) -> Void
    ) -> Bool {
        guard let stmt = _prepareLocked(sql, values) else { return false }
        defer { sqlite3_finalize(stmt) }
        var rc = sqlite3_step(stmt)
        while rc == SQLITE_ROW {
            row(stmt)
            rc = sqlite3_step(stmt)
        }
        return rc == SQLITE_DONE
    }

    private static func date(julianDay: Double) -> Date {
        Date(timeIntervalSince1970: (julianDay - 2_440_587.5) * 86_400)
    }

    // MARK: - Companion accounting helpers (caller holds `lock`)

    /// `INSERT … SELECT` that turns a row's companion link into an unlinked
    /// entry, keeping the row's recency. Callers append the WHERE clause.
    private static let moveLinkedCompanionsToLegacySQL = """
        INSERT OR REPLACE INTO legacy_companions (key, bytes, modified)
        SELECT companion_key, companion_bytes, (created_at - 2440587.5) * 86400.0
        FROM cache_entries
        """

    /// `rc` is the statement's result; `changed` is whether a row with that
    /// hash was there to update. `SQLITE_DONE` with `changed == false` means
    /// the row does not exist (any more).
    @discardableResult
    private func _linkCompanionLocked(
        kvHash: String, companionKey: String, bytes: Int64
    ) -> (rc: Int32, changed: Bool) {
        guard let db else { return (SQLITE_MISUSE, false) }
        let rc = _runLocked(
            "UPDATE cache_entries SET companion_key = ?, companion_bytes = ? WHERE hash = ?",
            [.text(companionKey), .int(max(0, bytes)), .text(kvHash)])
        return (rc, rc == SQLITE_DONE && sqlite3_changes(db) > 0)
    }

    @discardableResult
    private func _upsertLegacyCompanionLocked(key: String, bytes: Int64, modified: Date) -> Int32 {
        _runLocked(
            "INSERT OR REPLACE INTO legacy_companions (key, bytes, modified) VALUES (?, ?, ?)",
            [.text(key), .int(max(0, bytes)), .real(modified.timeIntervalSince1970)])
    }

    private func _legacyCompanionsLocked() -> [DiskCacheLegacyCompanion] {
        var result: [DiskCacheLegacyCompanion] = []
        _queryLocked("SELECT key, bytes, modified FROM legacy_companions") { stmt in
            guard let cKey = sqlite3_column_text(stmt, 0) else { return }
            result.append(DiskCacheLegacyCompanion(
                key: String(cString: cKey),
                bytes: max(0, sqlite3_column_int64(stmt, 1)),
                modifiedAt: Date(timeIntervalSince1970: sqlite3_column_double(stmt, 2))))
        }
        return result
    }

    /// Whole-root usage: two aggregates, no row materialization.
    private func _combinedUsageLocked() -> (bytes: Int64, entryCount: Int) {
        guard indexHasV2Columns else {
            let usage = _payloadUsageLocked()
            return (Int64(usage.bytes), usage.entryCount)
        }
        var bytes: Int64 = 0
        var count = 0
        _queryLocked(
            "SELECT COALESCE(SUM(file_size + companion_bytes), 0), COUNT(*) FROM cache_entries"
        ) { stmt in
            bytes += max(0, sqlite3_column_int64(stmt, 0))
            count += Int(sqlite3_column_int64(stmt, 1))
        }
        _queryLocked("SELECT COALESCE(SUM(bytes), 0), COUNT(*) FROM legacy_companions") { stmt in
            bytes += max(0, sqlite3_column_int64(stmt, 0))
            count += Int(sqlite3_column_int64(stmt, 1))
        }
        return (bytes, count)
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

    /// Record one over-cap pass of the coordinator's linked KV +
    /// recurrent-companion quota. `evictedGroups` / `evictedBytes` count the
    /// logical boundaries whose every file is really gone, so one atomic pair
    /// increments `evictions` once; a pass that removed none is timed but is
    /// not a counted pass. `pressureEventSeq` moves once per pass that
    /// produced an event — the plan's, whether or not every delete succeeded.
    func recordQuotaPass(
        evictedGroups: Int, evictedBytes: Int64, milliseconds: Double,
        event: DiskCachePressureEvent?
    ) {
        lock.lock()
        defer { lock.unlock() }
        if evictedGroups > 0 {
            evictions += evictedGroups
            quotaEvictedBytes += max(0, evictedBytes)
            quotaPasses += 1
        }
        lastQuotaPassMs = milliseconds
        if let event {
            pressureEventSeq += 1
            lastPressureEvent = event
        }
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
    ///
    /// `keepCompanionCounted`: this cache does not own the companion files,
    /// so when it drops a row on its own (payload missing or refused, or its
    /// standalone eviction) the row's companion is handed to the unlinked
    /// list. Its bytes stay counted and the combined quota retires it first,
    /// instead of the files silently leaving the accounting.
    private func _deleteEntryLocked(hash: String, keepCompanionCounted: Bool = true) {
        guard db != nil else { return }
        if indexHasV2Columns, keepCompanionCounted {
            _runLocked(
                Self.moveLinkedCompanionsToLegacySQL
                    + " WHERE hash = ? AND companion_key IS NOT NULL",
                [.text(hash)])
        }
        _runLocked("DELETE FROM cache_entries WHERE hash = ?", [.text(hash)])
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

        // Delete evicted entries and their files. A payload that could not
        // be deleted keeps its row (see `removeQuotaEntries`).
        for entry in toEvict {
            validatedFiles.removeValue(forKey: entry.hash)
            guard Self.removeCacheFile(at: safetensorsURL(for: entry.hash)) else { continue }
            _deleteEntryLocked(hash: entry.hash)
            evictions += 1
        }
    }
}
