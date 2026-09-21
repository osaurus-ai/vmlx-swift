// Copyright © 2025 Apple Inc. All rights reserved.

import CryptoKit
import Foundation
import MLX
import SQLite3
import os

/// Thread-safe snapshot of ``DiskCache`` counters.
public struct DiskCacheStats: Sendable {
    /// Disk entries served to an engine. One that the engine then reported
    /// as not restorable is taken out again (``rejectedDiskRestores``). One
    /// that it abandoned for a reason of the REQUEST's — a media placeholder
    /// left in the suffix, a missing seed state for an exact hit — is still
    /// counted here although the turn prefilled the whole prompt: the entry
    /// was fine, and nothing is reported for it.
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
    ///
    /// Also counted: a retirement of records that name nothing which could
    /// not be written. Those records stay counted until one can; no file is
    /// involved, and nothing real is evicted to make up for them. And a row
    /// that was to be dropped and stayed, because its companion could not be
    /// handed to the unlinked list: it goes the next time that can be written.
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
    /// Fetches in this process that found an indexed payload they could not
    /// READ (EACCES, EMFILE, EIO, a loader that could not open it). That says
    /// nothing about the payload being corrupt: it is reported as a miss and
    /// the file and its row are left exactly as they are.
    public let unreadablePayloadFetches: Int
    /// Bytes held by records of an index a NEWER build has claimed that this
    /// build counts but can neither read nor evict, as the most recent
    /// over-cap quota pass found them; 0 until there has been one, and always
    /// 0 under the current schema. They come off the cap the rows this build
    /// understands share: once they reach it, every store is evicted again
    /// by the pass that follows it.
    public let opaqueBytes: Int64
    /// Disk hits in this process that the engine then could not restore into
    /// the running model's cache and reported back
    /// (``CacheCoordinator/reportDiskRestoreRejected(tokens:boundary:mediaSalt:reason:)``).
    /// Each one was first counted in ``hits`` and has been taken out of it
    /// again. ``hits`` still includes hits that were abandoned for reasons
    /// of the request's, which are not rejections.
    public let rejectedDiskRestores: Int
    /// Rejections counted in ``rejectedDiskRestores`` whose payload this
    /// process had itself written after an earlier rejection of the same
    /// entry: what it stores for that boundary does not restore either. The
    /// entry stays passed over by fetch and is NOT written a second time — a
    /// boundary payload can be hundreds of megabytes, and a store → restore
    /// round trip that fails once fails every turn.
    public let rejectedRewritesSuppressed: Int

    init(
        hits: Int, misses: Int, stores: Int, storeSkips: Int,
        currentPayloadBytes: Int, currentEntryCount: Int,
        evictions: Int, maxSizeBytes: Int, failedIndexWrites: Int = 0,
        evictedBytes: Int64 = 0, quotaPasses: Int = 0, lastQuotaPassMs: Double = 0,
        pressureEventSeq: UInt64 = 0, lastPressureEvent: DiskCachePressureEvent? = nil,
        unreadablePayloadFetches: Int = 0, opaqueBytes: Int64 = 0,
        rejectedDiskRestores: Int = 0, rejectedRewritesSuppressed: Int = 0
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
        self.unreadablePayloadFetches = unreadablePayloadFetches
        self.opaqueBytes = opaqueBytes
        self.rejectedDiskRestores = rejectedDiskRestores
        self.rejectedRewritesSuppressed = rejectedRewritesSuppressed
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
            pressureEventSeq: pressureEventSeq, lastPressureEvent: lastPressureEvent,
            unreadablePayloadFetches: unreadablePayloadFetches, opaqueBytes: opaqueBytes,
            rejectedDiskRestores: rejectedDiskRestores,
            rejectedRewritesSuppressed: rejectedRewritesSuppressed)
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
    /// Rows whose `hash` is not a payload hash: they name no file, so none
    /// was looked at, and they are simply dropped.
    var rowsDroppedForInvalidHash = 0
    var linksWritten = 0
    var linksCleared = 0
    var legacyUpserted = 0
    var legacyDeleted = 0
    /// Unlinked records whose `key` is not a companion key: forgotten
    /// without looking at any file.
    var legacyDroppedForInvalidKey = 0
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
    /// The unpublished name a store was about to write under is held by
    /// something that is not a regular file (a link, a directory) or could
    /// not be examined. `save` would follow a link; nothing is written.
    case occupiedTemporaryName(String)
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
    /// "the file says 0". Which statements are used never branches on it
    /// (see `indexHasV2Columns`); only ``indexIsFromANewerBuild`` does, to
    /// keep every listing-driven removal away from a newer build's root.
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

    /// After a retirement of records that name nothing could not be written,
    /// how long this cache leaves them alone before it tries again. Until
    /// then they stay counted and are still never offered to a quota pass,
    /// so nothing real pays for them; what the wait saves is a write
    /// transaction (and its busy timeout) on every over-cap store.
    static let defaultRetireRetryInterval: TimeInterval = 60
    private let retireRetryInterval: TimeInterval
    /// The clock that paces that retry; a test passes its own.
    private let now: @Sendable () -> Date

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
    /// Stores refused because the payload's final name is held by something
    /// that is not a regular file (a directory, a link). It is not an older
    /// copy of the payload, so it is not replaced; nothing is published.
    public private(set) var refusedOccupiedStores: Int = 0
    /// See ``DiskCacheStats/unreadablePayloadFetches``.
    public private(set) var unreadablePayloadFetches: Int = 0
    /// See ``DiskCacheStats/rejectedDiskRestores``.
    public private(set) var rejectedDiskRestores: Int = 0
    /// See ``DiskCacheStats/rejectedRewritesSuppressed``.
    public private(set) var rejectedRewritesSuppressed: Int = 0

    /// Test seams, never set in production. `temporaryURLForTesting` names
    /// the unpublished file of the next store (the real name carries a random
    /// tag, so nothing can be planted under it in advance). An error thrown
    /// from `publishFaultForTesting` stands in for the publishing rename
    /// failing, and one from `loadFaultForTesting` for the safetensors loader
    /// failing on a payload that passed the header inspection.
    var temporaryURLForTesting: (@Sendable (URL) -> URL)?
    var publishFaultForTesting: (@Sendable () throws -> Void)?
    var loadFaultForTesting: (@Sendable (URL) throws -> Void)?
    /// Write transactions this cache has opened to retire records that name
    /// nothing. Only a test reads it: a stats poll must open none, and a
    /// retirement that failed must not be tried again by every store.
    private(set) var retireAttemptsForTesting = 0

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
    /// See ``DiskCacheStats/opaqueBytes``.
    private var lastOpaqueBytes: Int64 = 0
    /// Set when a retirement could not be written: none is tried before it.
    private var retireNotBefore: Date?
    /// Whether usage is summed with the clamping aggregate; see
    /// ``_usageSQLLocked(_:)``. Never goes back to false.
    private var indexNeedsClampedUsage = false

    /// Files successfully written or deserialized in this process. A matching
    /// fingerprint lets `store` avoid realizing and rewriting the same large
    /// prompt boundary after a cache hit, while a fresh process still validates
    /// an inherited file before it can take the fast path.
    private var validatedFiles: [String: ValidatedRecord] = [:]

    /// Entries an engine could not restore into the running model's cache
    /// (``markRestoreRejected(tokens:mediaSalt:)``), with the fingerprint of
    /// the payload that was refused. While that very file is under the name,
    /// the entry is not offered to the coordinator again — so a shorter entry
    /// can win — and does not count as validated or durable, so the next
    /// store of the boundary writes it again; that store clears the mark. A
    /// payload somebody else has replaced since is a different file, and the
    /// mark goes the first time that is seen. In memory only, per instance:
    /// nothing is deleted, and a new process finds out again at the cost of
    /// one refused restore.
    private var rejectedRestores: [String: ValidatedFileFingerprint] = [:]

    /// Entries this cache has already written again after a rejection. When
    /// the payload it wrote is rejected too, the store → restore round trip
    /// itself is what fails (a serializer that does not round-trip, a
    /// caller-supplied cache of another topology), and it fails the same way
    /// every turn: fetch, refuse, write a boundary of hundreds of megabytes,
    /// lift the mark, again. So the second mark still makes fetch pass over
    /// the entry and no longer makes it non-durable
    /// (``_awaitsRewriteAfterRejectionLocked(hash:current:)``): at most one
    /// rewrite per entry per process. No store follows to lift that mark. It
    /// goes, with this memory, when the payload under the name is not the one
    /// that was refused — somebody else has replaced it — or with the process.
    private var rewrittenAfterRejection: Set<String> = []

    /// The payloads ``fetchCandidate(tokens:mediaSalt:)`` has handed out,
    /// oldest first. A rejection is about the file that was served, and the
    /// engine reports it some time after the fetch: a payload that another
    /// writer has put under the name in between was refused by nobody
    /// (``markRestoreRejected(tokens:mediaSalt:)``). Bounded: only the most
    /// recent ``servedCandidateLimit`` entries are kept, least recently
    /// served dropped first. A report that comes later than that finds
    /// nothing to compare with and is dropped — the entry is served, and
    /// refused, once more.
    private var servedCandidates: [(hash: String, file: ValidatedFileFingerprint)] = []
    static let servedCandidateLimit = 64

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
            lastPressureEvent: lastPressureEvent,
            unreadablePayloadFetches: unreadablePayloadFetches,
            opaqueBytes: lastOpaqueBytes,
            rejectedDiskRestores: rejectedDiskRestores,
            rejectedRewritesSuppressed: rejectedRewritesSuppressed)
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
        indexBusyTimeoutMs: Int32 = DiskCache.defaultIndexBusyTimeoutMs,
        retireRetryInterval: TimeInterval = DiskCache.defaultRetireRetryInterval,
        now: @escaping @Sendable () -> Date = { Date() }
    ) {
        self.cacheDir = cacheDir
        self.maxSizeBytes = maxSizeBytes
        self.modelKey = modelKey
        self.retireRetryInterval = retireRetryInterval
        self.now = now

        // Create cache directory if needed
        try? FileManager.default.createDirectory(at: cacheDir, withIntermediateDirectories: true)
        // Open SQLite database
        let dbPath = cacheDir.appendingPathComponent("cache_index.db").path
        if sqlite3_open(dbPath, &db) != SQLITE_OK {
            db = nil
            indexSchemaVersion = 0
            indexHasV2Columns = false
            Self.sweepUnpublishedAndIncompleteFiles(in: cacheDir)
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
        if let failure = DiskCacheIndexSchema.ensureV2Indexes(
            db, version: indexSchemaVersion, hasV2Columns: indexHasV2Columns,
            busyTimeoutMs: indexMigrationBusyTimeoutMs),
            Self.isFirstReport(cacheDir.path, in: Self.reportedIndexCreationFailures)
        {
            // Every open tries again, so every open would say so again.
            FileHandle.standardError.write(Data(
                ("[vmlx][cache/disk-index] \(DiskCacheIndexSchema.modelTokensIndexStatement) "
                    + "failed: \(Self.boundedRendering(of: failure)); carrying on without it\n")
                    .utf8))
        }

        // The schema helpers put the connection back to "no wait" when they
        // finish. Every statement from here on waits a bounded time instead.
        sqlite3_busy_timeout(db, max(0, indexBusyTimeoutMs))

        indexNeedsClampedUsage = _indexHoldsCountsToClampLocked()

        // Storage integrity at open: an interrupted store (crash, force-quit,
        // disk full) used to leave a partial `<hash>.safetensors` under its
        // FINAL name. `fetch` only checked existence, `loadArraysAndMetadata`
        // maps lazily, and the MLX reader's short-read exception is dropped
        // on the stream (`Load::eval_cpu` waits on the future without
        // `get()`), so the row restored as zero-filled KV / recurrent state
        // at a valid offset — silently. Stores now publish atomically
        // (temp → rename), so at open a `.partial-` file that is old enough
        // is a dead write, and a final-named file that was read and found
        // short of the payload its own header declares is removed.
        //
        // After the index is open, not before: under a schema a newer build
        // has claimed, this build does not know what the files in the root
        // mean, and removes none of them from a listing — here, in the
        // import's sweep and in `clear()` alike.
        if indexIsFromANewerBuild {
            FileHandle.standardError.write(Data(
                ("[vmlx][cache/disk] integrity sweep skipped: index schema version "
                    + "\(indexSchemaVersion) is newer than this build's "
                    + "\(DiskCacheIndexSchema.currentVersion)\n").utf8))
        } else {
            Self.sweepUnpublishedAndIncompleteFiles(in: cacheDir)
        }
    }

    /// Whether a newer build has claimed this root's index. Nothing is then
    /// removed from a listing of the root or of its companion directory.
    var indexIsFromANewerBuild: Bool {
        indexSchemaVersion > DiskCacheIndexSchema.currentVersion
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
    /// `chainId` names the conversation storing the row and `isStableRoot`
    /// marks a system/tool prefix shared by many conversations; both are index
    /// metadata for the quota planner only (a v1 index ignores them) and never
    /// touch the content key. Re-storing a row updates its owner to the
    /// storing conversation, keeps the previous owner when none is given, and
    /// never demotes a stable root.
    func store(
        tokens: [Int],
        arrays: [String: MLXArray],
        mediaSalt: String? = nil,
        enforceQuota: Bool,
        chainId: String? = nil,
        isStableRoot: Bool = false
    ) {
        guard let (hash, url) = entryKey(tokens: tokens, mediaSalt: mediaSalt) else { return }
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
           !_awaitsRewriteAfterRejectionLocked(hash: hash, current: current),
           validated.hasRecurrentGeometry,
           Self.payloadLayout(arrays) == validated.layout,
           let indexed = _entryMetadataLocked(hash: hash),
           indexed.tokenCount == tokenCount,
           indexed.fileSize == current.size,
           current.size > 0
        {
            storeSkips += 1
            _touchEntryLocked(hash: hash)
            _claimOwnershipLocked(hash: hash, chainId: chainId, isStableRoot: isStableRoot)
            if ProcessInfo.processInfo.environment["VMLX_CACHE_FETCH_TRACE"] == "1" {
                FileHandle.standardError.write(Data(
                    "[vmlx][cache/disk-store] SKIP validated hash=\(hash) count=\(tokenCount) bytes=\(current.size)\n".utf8))
            }
            return
        }
        // Refuse before the expensive part: see `_refuseOccupiedStoreLocked`.
        if _refuseOccupiedStoreLocked(finalURL: url, hash: hash, tokenCount: tokenCount) {
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
            let url = temporaryURLForTesting?(finalURL) ?? Self.temporaryURL(for: finalURL)
            // A leftover of ours under that name goes. Anything else there
            // is not written through: `save` would follow a link.
            switch Self.removeRegularFile(at: url) {
            case .removed, .missing:
                break
            case .notRegularFile, .failed:
                throw DiskCacheIntegrityError.occupiedTemporaryName(url.lastPathComponent)
            }
            try save(arrays: arrays, metadata: ["format": "mlx"], url: url)
            Stream.gpu.synchronize()
            guard Self.isCompleteSafetensors(url: url) else {
                _ = Self.removeRegularFile(at: url)
                throw DiskCacheIntegrityError.incompleteWrite(finalURL.lastPathComponent)
            }
            // Publish with one `rename(2)`. It replaces an older regular file
            // of the same hash atomically, so there is no moment at which
            // the old valid payload is gone and the new one not yet there,
            // and a rename that fails has cost nothing. Anything else under
            // the final name is not an older copy of this payload: a
            // directory or a link found there refuses the store, and a
            // directory that takes the name after that look makes the
            // rename itself fail (EISDIR) instead of being descended into.
            if _refuseOccupiedStoreLocked(finalURL: finalURL, hash: hash, tokenCount: tokenCount) {
                _ = Self.removeRegularFile(at: url)
                return
            }
            do {
                try publishFaultForTesting?()
                let code = Self.renameFile(from: url, to: finalURL)
                if code == EISDIR || code == ENOTDIR {
                    _ = Self.removeRegularFile(at: url)
                    _recordRefusedOccupiedStoreLocked(
                        finalURL: finalURL, hash: hash, tokenCount: tokenCount)
                    return
                }
                guard code == 0 else { throw POSIXError(POSIXErrorCode(rawValue: code) ?? .EIO) }
            } catch {
                _ = Self.removeRegularFile(at: url)
                throw error
            }
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
                hash: hash, tokenCount: tokenCount, fileSize: fileSize,
                chainId: chainId, isStableRoot: isStableRoot)
            guard insertResult == SQLITE_DONE else {
                // The payload is published but has no row, so no quota pass
                // could ever see or evict it. Take it back rather than leak it.
                _ = Self.removeRegularFile(at: finalURL)
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
            // What was refused has just been replaced — the one time this
            // process does that for the entry.
            if rejectedRestores.removeValue(forKey: hash) != nil {
                rewrittenAfterRejection.insert(hash)
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

    /// A store is refused — counted, logged, nothing published and no row
    /// written — when its final name is held by something that is not a
    /// regular file. The root is a user setting: a directory or a link that
    /// happens to carry this hash is not an older copy of the payload and is
    /// never replaced. The boundary is simply not cached; an ordinary miss.
    /// Returns whether the store was refused. Caller holds `lock`.
    private func _refuseOccupiedStoreLocked(finalURL: URL, hash: String, tokenCount: Int) -> Bool {
        guard case .notRegularFile = Self.pathState(at: finalURL) else { return false }
        _recordRefusedOccupiedStoreLocked(finalURL: finalURL, hash: hash, tokenCount: tokenCount)
        return true
    }

    private func _recordRefusedOccupiedStoreLocked(finalURL: URL, hash: String, tokenCount: Int) {
        refusedOccupiedStores += 1
        validatedFiles.removeValue(forKey: hash)
        FileHandle.standardError.write(Data(
            ("[vmlx][cache/disk-store] REFUSED occupied path count=\(tokenCount) "
                + "hash=\(hash.prefix(12)) — \(finalURL.lastPathComponent) is not a regular file "
                + "and is left alone; nothing published\n").utf8))
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
        guard let (hash, url) = entryKey(tokens: tokens, mediaSalt: mediaSalt) else { return nil }

        MLXDiskCacheIOLock.shared.lock()
        defer { MLXDiskCacheIOLock.shared.unlock() }
        lock.lock()
        defer { lock.unlock() }
        return _fetchLocked(
            hash: hash, url: url, tokens: tokens, mediaSalt: mediaSalt,
            touchRecency: touchRecency, countHit: countHit)
    }

    /// What ``fetchCandidate(tokens:mediaSalt:)`` found.
    enum CandidateFetch {
        case arrays([String: MLXArray])
        case miss
        /// The entry is there, and an engine has already refused to restore
        /// this very payload. Nothing was read; not counted as a miss.
        case restoreRejectedEarlier
    }

    /// The coordinator's candidate read: ``fetch(tokens:mediaSalt:touchRecency:countHit:)``
    /// with no recency touch and no hit counted — the coordinator does both
    /// for the candidate it accepts — that also passes over an entry whose
    /// restore was refused (``markRestoreRejected(tokens:mediaSalt:)``).
    func fetchCandidate(tokens: [Int], mediaSalt: String?) -> CandidateFetch {
        guard let (hash, url) = entryKey(tokens: tokens, mediaSalt: mediaSalt) else { return .miss }

        MLXDiskCacheIOLock.shared.lock()
        defer { MLXDiskCacheIOLock.shared.unlock() }
        lock.lock()
        defer { lock.unlock() }

        if let refused = rejectedRestores[hash] {
            if _fileFingerprint(url: url) == refused { return .restoreRejectedEarlier }
            // Another writer has replaced the payload, or it is gone: what
            // this process knew about the entry was about the other file.
            rejectedRestores.removeValue(forKey: hash)
            rewrittenAfterRejection.remove(hash)
        }
        let before = _fileFingerprint(url: url)
        let arrays = _fetchLocked(
            hash: hash, url: url, tokens: tokens, mediaSalt: mediaSalt,
            touchRecency: false, countHit: false)
        // Remember what was served only when the file was the same one on
        // both sides of the read (`_fetchLocked` fingerprints what it loaded).
        servedCandidates.removeAll { $0.hash == hash }
        if arrays != nil, let before, validatedFiles[hash]?.file == before {
            servedCandidates.append((hash, before))
            if servedCandidates.count > Self.servedCandidateLimit {
                servedCandidates.removeFirst(servedCandidates.count - Self.servedCandidateLimit)
            }
        }
        return arrays.map(CandidateFetch.arrays) ?? .miss
    }

    /// An engine fetched this entry through the coordinator and could not
    /// restore it into the running model's cache. See ``rejectedRestores``
    /// for what follows from that. Returns false, and changes nothing, when
    /// there is no such entry (no row, or no payload under the name), when
    /// the payload under the name is not the one the candidate fetch served
    /// (see ``servedCandidates``), or when this payload is already marked — a
    /// rejection reported twice takes one hit back, not two. A payload this
    /// cache wrote after an earlier rejection is marked for fetch only; see
    /// ``rewrittenAfterRejection``.
    ///
    /// Takes `lock` only, like the other predicates: it reads one row and
    /// stats one file.
    @discardableResult
    func markRestoreRejected(tokens: [Int], mediaSalt: String?) -> Bool {
        guard let (hash, url) = entryKey(tokens: tokens, mediaSalt: mediaSalt) else { return false }
        lock.lock()
        defer { lock.unlock() }
        guard let current = _fileFingerprint(url: url),
              servedCandidates.last(where: { $0.hash == hash })?.file == current,
              _entryMetadataLocked(hash: hash) != nil,
              rejectedRestores[hash] != current
        else { return false }
        rejectedRestores[hash] = current
        if rewrittenAfterRejection.contains(hash) {
            rejectedRewritesSuppressed += 1
        } else {
            validatedFiles.removeValue(forKey: hash)
        }
        hits = max(0, hits - 1)
        rejectedDiskRestores += 1
        return true
    }

    /// Whether the payload under the name is one an engine refused and this
    /// cache has yet to write again: such an entry is neither validated nor
    /// durable, which is what makes the next store of the boundary a real
    /// write. False once that write has happened, whatever is refused after
    /// it; see ``rewrittenAfterRejection``. Caller holds `lock`.
    private func _awaitsRewriteAfterRejectionLocked(
        hash: String, current: ValidatedFileFingerprint
    ) -> Bool {
        rejectedRestores[hash] == current && !rewrittenAfterRejection.contains(hash)
    }

    /// Caller holds ``MLXDiskCacheIOLock`` and `lock`.
    private func _fetchLocked(
        hash: String, url: URL, tokens: [Int], mediaSalt: String?,
        touchRecency: Bool, countHit: Bool
    ) -> [String: MLXArray]? {
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

        // Only a regular file is a payload. The name is ours, but the root is
        // a user setting: `open` on a FIFO of that name never returns, and
        // this thread holds the process-wide IO lock. Whatever else holds
        // the name is not this cache's file — a miss, and it and the row
        // stay (the same answer the sweep, the import and a store give).
        switch Self.pathState(at: url) {
        case .regularFile:
            break
        case .notRegularFile:
            return _unreadablePayloadMissLocked(
                hash: hash, url: url, tokenCount: tokens.count, reason: "not a regular file")
        case .unreadable(let code):
            return _unreadablePayloadMissLocked(
                hash: hash, url: url, tokenCount: tokens.count,
                reason: String(cString: strerror(code)))
        case .missing:
            // Gone since `fileExists`: the next fetch settles the row.
            validatedFiles.removeValue(forKey: hash)
            misses += 1
            return nil
        }

        do {
            // Fail closed on a short file BEFORE the lazy map: the reader's
            // short-read error never reaches the caller, so a truncated row
            // would otherwise restore as zeros at a valid offset.
            switch Self.inspectSafetensors(url: url) {
            case .complete:
                break
            case .shortOrMalformed:
                throw DiskCacheIntegrityError.incompleteFile(url.lastPathComponent)
            case .unreadable(let code):
                return _unreadablePayloadMissLocked(
                    hash: hash, url: url, tokenCount: tokens.count,
                    reason: String(cString: strerror(code)))
            }
            try loadFaultForTesting?(url)
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
            // Only what has been positively identified as unusable is
            // removed: a file short of its declared bytes, a non-finite
            // payload, or a decode error on a file that opens and whose
            // header parses. A loader that could not OPEN the file, or a
            // file that can no longer be inspected, may be a transient
            // condition (EMFILE, EIO, EACCES), and deleting a valid payload
            // for it would be reading "could not look" as "corrupt".
            if !Self.isPositiveCorruption(error, url: url) {
                return _unreadablePayloadMissLocked(
                    hash: hash, url: url, tokenCount: tokens.count, reason: "\(error)")
            }
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

    /// What the vendored MLX safetensors loader's message contains when it
    /// could not OPEN the file (`[load_safetensors] Failed to open …`). The
    /// loader has no error codes, so this text is the only thing that tells
    /// "could not look" from "looked, and it is corrupt" — and the second
    /// deletes. `theLoaderStillSaysItCouldNotOpenAFile` runs the real loader
    /// against an unreadable payload, so a reworded message fails a test
    /// instead of deleting caches on a transient EACCES.
    static let loaderCouldNotOpenMarker = "Failed to open"

    /// Whether a failed fetch has shown the payload itself to be unusable.
    /// The two integrity errors are this cache's own findings. Anything else
    /// came from the loader: it counts only when the loader got as far as
    /// the file's contents — it did not fail to open it — and the file can
    /// still be opened and inspected now.
    private static func isPositiveCorruption(_ error: Error, url: URL) -> Bool {
        switch error {
        case DiskCacheIntegrityError.incompleteFile, DiskCacheIntegrityError.nonFinitePayload:
            return true
        default:
            if "\(error)".contains(loaderCouldNotOpenMarker) { return false }
            if case .unreadable = inspectSafetensors(url: url) { return false }
            return true
        }
    }

    /// A payload that could not be read is a miss and nothing else: the file
    /// and its row stay exactly as they are. Caller holds `lock`.
    private func _unreadablePayloadMissLocked(
        hash: String, url: URL, tokenCount: Int, reason: String
    ) -> [String: MLXArray]? {
        misses += 1
        unreadablePayloadFetches += 1
        validatedFiles.removeValue(forKey: hash)
        if Self.isFirstReport(url.path, in: Self.reportedUnreadablePayloads) {
            FileHandle.standardError.write(Data(
                ("[vmlx][cache/disk] fetch could not read \(url.lastPathComponent) "
                    + "count=\(tokenCount): \(reason) — a miss; file and row left alone\n").utf8))
        }
        return nil
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
        guard let (hash, url) = entryKey(tokens: tokens, mediaSalt: mediaSalt) else { return false }
        lock.lock()
        defer { lock.unlock() }

        guard let validated = validatedFiles[hash],
              let current = _fileFingerprint(url: url),
              current == validated.file,
              !_awaitsRewriteAfterRejectionLocked(hash: hash, current: current),
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
        guard let (hash, url) = entryKey(tokens: tokens, mediaSalt: mediaSalt) else { return false }
        lock.lock()
        defer { lock.unlock() }
        guard let current = _fileFingerprint(url: url), current.size > 0,
              let indexed = _entryMetadataLocked(hash: hash),
              indexed.tokenCount == tokens.count,
              indexed.fileSize == current.size,
              // Complete and self-consistent, and of no use to the model
              // that is running: it has to be produced again to be replaced.
              !_awaitsRewriteAfterRejectionLocked(hash: hash, current: current)
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
    ///
    /// On an index with the v2 columns only the lengths of rows this model
    /// could have written are returned: rows that carry this cache's
    /// `modelKey`, and rows that carry none. Every other model's row hashes
    /// to a different key by construction, so probing its length can only
    /// miss — and in a shared root those probes were most of what a fetch did
    /// (each one a prefix copy, a SHA-256 of the prefix and an index lookup
    /// under the process-wide IO lock).
    ///
    /// The filter errs towards returning a length. A row with a NULL
    /// `model_key` was written by an older build (its three-column
    /// `INSERT OR REPLACE` also resets the key of a row this build wrote), or
    /// by a cache with no model key, and stays a candidate for everyone. The
    /// key compared is the very string ``_insertEntryLocked`` binds, bound
    /// the same way — except that a cache with NO model key also asks for
    /// the empty one: ``hashTokens(_:modelKey:mediaSalt:)`` gives nil and ""
    /// the same hashes, so those two share their entries, while the column
    /// holds NULL for one and '' for the other. Under a newer build's schema
    /// the column may mean something else, so nothing is filtered there.
    /// Rows of this model from other conversations or other media salts are
    /// NOT removed: neither is a column.
    public func candidateTokenCounts(maxTokens: Int, limit: Int = 128) -> [Int] {
        guard db != nil, maxTokens > 0, limit > 0 else { return [] }
        lock.lock()
        defer { lock.unlock() }

        var counts: [Int] = []
        let read: (OpaquePointer) -> Void = { stmt in
            counts.append(Int(sqlite3_column_int64(stmt, 0)))
        }
        if indexHasV2Columns, !indexIsFromANewerBuild,
            _queryLocked(
                Self.modelCandidateTokenCountsSQL,
                [.int(Int64(maxTokens)), .int(Int64(limit)), .text(modelKey ?? "")],
                read)
        {
            return counts
        }
        // No model column to filter on — or the filtered statement did not
        // run to its end, and a length too many costs a probe where a length
        // too few costs the hit.
        counts.removeAll(keepingCapacity: true)
        _queryLocked(
            Self.candidateTokenCountsSQL, [.int(Int64(maxTokens)), .int(Int64(limit))], read)
        return counts
    }

    static let candidateTokenCountsSQL = """
        SELECT DISTINCT token_count
        FROM cache_entries
        WHERE token_count > 0 AND token_count <= ?
        ORDER BY token_count DESC
        LIMIT ?
        """

    /// ``candidateTokenCountsSQL`` restricted to `model_key = ?3 OR model_key
    /// IS NULL`, written as two arms so that each is one bounded range scan
    /// of `idx_cache_entries_model_tokens`, newest length first, that stops
    /// at the LIMIT. The one-statement `OR` form reads and sorts EVERY row of
    /// the model below `maxTokens` for every page (the LIMIT cannot be pushed
    /// under the DISTINCT), which makes paging through a large cache
    /// quadratic; and once the index has been ANALYZEd the planner answers it
    /// from the token-count index instead, reading every other model's rows
    /// again. The top `limit` of the union is within the union of each arm's
    /// top `limit`.
    static let modelCandidateTokenCountsSQL = """
        SELECT token_count FROM (
            SELECT * FROM (
                SELECT DISTINCT token_count FROM cache_entries
                WHERE model_key = ?3 AND token_count > 0 AND token_count <= ?1
                ORDER BY token_count DESC LIMIT ?2)
            UNION
            SELECT * FROM (
                SELECT DISTINCT token_count FROM cache_entries
                WHERE model_key IS NULL AND token_count > 0 AND token_count <= ?1
                ORDER BY token_count DESC LIMIT ?2)
        )
        ORDER BY token_count DESC
        LIMIT ?2
        """

    /// Snapshot indexed KV payloads for the coordinator's combined KV +
    /// recurrent-companion quota. Database/WAL bookkeeping is intentionally
    /// excluded, matching this cache's existing `SUM(file_size)` contract.
    ///
    /// `retiringInvalidRecords: false` is for a reader that must not write —
    /// the stats poll: what names nothing is left out all the same, and is
    /// retired by the next quota pass instead.
    func quotaEntries(retiringInvalidRecords: Bool = true) -> [DiskCacheQuotaEntry] {
        guard let db else { return [] }
        lock.lock()
        defer { lock.unlock() }

        var entries: [DiskCacheQuotaEntry] = []
        // What is returned here is what a quota pass may delete, by a path
        // built from it. A row whose hash is not a payload hash, or a link
        // whose key is not a companion key, names no file: it is not offered
        // — and, once the statement is finished, retired BY ROWID, so that
        // its bytes stop counting towards a cap they could otherwise hold
        // over for good. Under a newer build's index it is not offered and
        // not retired either (``_retireInvalidRecordsLocked(_:reconciling:)``).
        var invalid = InvalidRecords()
        defer { if retiringInvalidRecords { _retireInvalidRecordsLocked(invalid) } }
        let rowsAreOpaque = indexIsFromANewerBuild
        var stmt: OpaquePointer?
        let sql = indexHasV2Columns
            ? """
                SELECT hash, file_size, created_at, companion_key, companion_bytes,
                       token_count, kind, chain_id, rowid
                FROM cache_entries
                """
            : "SELECT hash, file_size, created_at, rowid FROM cache_entries"
        let rowidColumn: Int32 = indexHasV2Columns ? 8 : 3
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK, let stmt else {
            sqlite3_finalize(stmt)
            return []
        }
        defer { sqlite3_finalize(stmt) }

        while sqlite3_step(stmt) == SQLITE_ROW {
            // Read before anything can skip the row: a negative count on a
            // row that is not offered hides other rows' bytes all the same,
            // and reading it is what switches the aggregate over.
            let bytes = _indexedBytesLocked(stmt, 1)
            let hash: String
            switch Self.indexValue(stmt, column: 0, hexDigits: Self.hashLength) {
            case .valid(let value):
                hash = value
            case .unreadable:
                continue
            case .null, .invalid:
                invalid.entries.append(sqlite3_column_int64(stmt, rowidColumn))
                continue
            }
            var entry = DiskCacheQuotaEntry(
                hash: hash,
                bytes: bytes,
                createdAt: Self.date(julianDay: sqlite3_column_double(stmt, 2)))
            // A row an older build wrote has NULL / 0 here: it is simply a
            // KV-only group.
            if indexHasV2Columns {
                switch Self.indexValue(
                    stmt, column: 3, hexDigits: SSMCompanionDiskStore.keyLength)
                {
                case .valid(let key):
                    entry.companionKey = key
                    entry.companionBytes = _indexedBytesLocked(stmt, 4)
                case .unreadable:
                    continue
                case .null where sqlite3_column_int64(stmt, 4) == 0:
                    break
                case .null, .invalid:
                    // A link that names nothing (or bytes with no key at
                    // all). Cleared, and the row is offered as KV-only —
                    // unless a newer build wrote it: evicting the row would
                    // hand that key on, so the whole row is left alone.
                    invalid.links.append(sqlite3_column_int64(stmt, rowidColumn))
                    if rowsAreOpaque { continue }
                }
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
        return _countingAgainIfClampingBecameNecessary {
            var bytes: Int64 = 0
            _queryLocked(
                """
                SELECT (SELECT \(_usageSQLLocked("companion_bytes")) FROM cache_entries)
                     + (SELECT \(_usageSQLLocked("bytes")) FROM legacy_companions)
                """
            ) { stmt in bytes = _indexedBytesLocked(stmt, 0) }
            return bytes
        }
    }

    /// Every counted companion, least recent first. A linked companion has
    /// its row's recency, an unlinked one its own; insertion order breaks
    /// ties (`julianday('now')` has millisecond resolution).
    func companionsOldestFirst() -> [DiskCacheLegacyCompanion] {
        guard indexHasV2Columns else { return [] }
        lock.lock()
        defer { lock.unlock() }
        var result: [DiskCacheLegacyCompanion] = []
        // As in `quotaEntries()`: what is returned may be deleted by a path
        // built from its key, so a key that is not one names nothing here,
        // and its record is retired by rowid.
        var invalid = InvalidRecords()
        defer { _retireInvalidRecordsLocked(invalid) }
        _queryLocked(
            """
            SELECT companion_key, companion_bytes,
                   (created_at - 2440587.5) * 86400.0 AS recency, rowid AS seq, 0 AS unlinked
            FROM cache_entries WHERE companion_key IS NOT NULL OR companion_bytes != 0
            UNION ALL
            SELECT key, bytes, modified, rowid, 1 FROM legacy_companions
            ORDER BY recency ASC, seq ASC
            """
        ) { stmt in
            let key: String
            switch Self.indexValue(stmt, column: 0, hexDigits: SSMCompanionDiskStore.keyLength) {
            case .valid(let value):
                key = value
            case .unreadable:
                return
            case .null, .invalid:
                let rowid = sqlite3_column_int64(stmt, 3)
                if sqlite3_column_int64(stmt, 4) == 1 {
                    invalid.legacy.append(rowid)
                } else {
                    invalid.links.append(rowid)
                }
                return
            }
            result.append(DiskCacheLegacyCompanion(
                key: key,
                bytes: _indexedBytesLocked(stmt, 1),
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
        keys.forEach(_forgetCompanionLocked)
    }

    /// Stop counting one companion, linked or not. Caller holds `lock`.
    private func _forgetCompanionLocked(key: String) {
        _runLocked(
            "UPDATE cache_entries SET companion_key = NULL, companion_bytes = 0 WHERE companion_key = ?",
            [.text(key)])
        _runLocked("DELETE FROM legacy_companions WHERE key = ?", [.text(key)])
    }

    // MARK: - Index values are data, never path components
    //
    // `cache_index.db` is a plain SQLite file: an older build, another tool
    // or corruption can put anything in it, and the root it sits in is a
    // user setting that may be a model folder. A `hash` of
    // `../../models/foo/model` would address a file outside the root, and
    // `model-00001-of-00008` a shard inside it with no traversal at all. So
    // a value read from the index becomes a path only through
    // ``safetensorsURL(for:)`` / ``SSMCompanionDiskStore/entryURLs(key:in:)``,
    // which answer nil for anything this cache does not itself compute. Such
    // a value names no file: none is stat'ed, none is deleted, and the record
    // that carries it is retired — by rowid, the one handle that works for
    // every value — so that it stops counting. Under an index a newer build
    // has claimed the record is opaque instead: counted, and left alone.

    /// One index column that is about to be used as a name, judged on its
    /// storage class and its BYTES — never through `String(cString:)`, which
    /// stops at the first NUL and repairs invalid UTF-8. `<32 hex>\0junk`, a
    /// BLOB of 32 hex bytes and a TEXT with a broken sequence all READ as
    /// something they are not, and a statement that binds the String back
    /// matches no row for any of them.
    enum IndexValue: Equatable {
        /// TEXT of exactly the expected number of lowercase hex digits.
        case valid(String)
        case null
        /// Anything else. `shown` is a bounded rendering, for the log only.
        case invalid(shown: String)
        /// SQLite could not produce the column (out of memory). Says nothing
        /// about the row, which is neither offered nor retired.
        case unreadable
    }

    static func indexValue(_ stmt: OpaquePointer, column: Int32, hexDigits: Int) -> IndexValue {
        // The type first: it is undefined once a conversion has happened.
        switch sqlite3_column_type(stmt, column) {
        case SQLITE_NULL:
            return .null
        case SQLITE_TEXT:
            guard let text = sqlite3_column_text(stmt, column) else { return .unreadable }
            let bytes = Int(sqlite3_column_bytes(stmt, column))
            // `bytes` counts every byte, embedded NULs included, so the
            // C-string test below sees the whole value or is not run.
            if bytes == hexDigits, isLowercaseHex(cString: text, count: hexDigits) {
                return .valid(String(cString: text))
            }
            return .invalid(
                shown: boundedRendering(
                    of: UnsafeRawBufferPointer(start: text, count: bytes), asText: true))
        case SQLITE_BLOB:
            let blob = sqlite3_column_blob(stmt, column)
            let bytes = Int(sqlite3_column_bytes(stmt, column))
            return .invalid(
                shown: boundedRendering(
                    of: UnsafeRawBufferPointer(start: blob, count: blob == nil ? 0 : bytes),
                    asText: false))
        case SQLITE_INTEGER:
            return .invalid(shown: "INTEGER \(sqlite3_column_int64(stmt, column))")
        default:
            return .invalid(shown: "REAL \(sqlite3_column_double(stmt, column))")
        }
    }

    /// How many bytes of a hostile value a log line shows.
    static let reportedValueByteLimit = 96

    /// At most ``reportedValueByteLimit`` BYTES of the value, escaped.
    /// Counting Characters bounds nothing: one grapheme can carry thousands
    /// of combining marks.
    static func boundedRendering(of bytes: UnsafeRawBufferPointer, asText: Bool) -> String {
        let shownBytes = bytes.prefix(reportedValueByteLimit)
        let body = asText
            ? String(decoding: shownBytes, as: UTF8.self).debugDescription
            : "x'" + shownBytes.map { String(format: "%02x", $0) }.joined() + "'"
        return bytes.count > shownBytes.count ? "\(body)… (\(bytes.count) bytes)" : body
    }

    static func boundedRendering(of value: String) -> String {
        var copy = value
        return copy.withUTF8 { boundedRendering(of: UnsafeRawBufferPointer($0), asText: true) }
    }

    /// Records one read of the index found to name nothing, by `rowid`.
    private struct InvalidRecords {
        /// `cache_entries` rows whose `hash` is not a payload hash.
        var entries: [Int64] = []
        /// `cache_entries` rows whose hash is fine and whose companion link
        /// is not: a key that is not a companion key, or bytes with no key.
        var links: [Int64] = []
        /// `legacy_companions` rows whose `key` is not a companion key.
        var legacy: [Int64] = []

        var isEmpty: Bool { entries.isEmpty && links.isEmpty && legacy.isEmpty }
    }

    /// Drop the rows, clear the links and forget the unlinked records that
    /// name nothing — each BY ROWID, which is the only handle that works for
    /// every value (see ``IndexValue``). Returns how many of each went.
    ///
    /// One transaction, all or nothing: a row's drop and the hand-over of
    /// its companion (if that names a real one, it stays counted as
    /// unlinked) cannot come apart. The rowids were read before the write
    /// lock was taken, so each record is read again under it and retired
    /// only if it still names nothing; a rowid another connection has
    /// reused for a real row is left alone.
    ///
    /// Under an index a NEWER build has claimed nothing is retired: a value
    /// this build cannot read may be how that build spells its names, and a
    /// row dropped here is a payload it would later sweep as unindexed. The
    /// callers do not offer such a record either, so it is counted and
    /// otherwise left exactly as it is.
    ///
    /// A retirement that cannot be written — the write lock refused, a
    /// statement or the COMMIT failing — changes nothing, and is never
    /// silent: ``_retirementFailedLocked(_:backingOff:)`` counts it, says why
    /// once, and keeps the next ``retireRetryInterval`` free of attempts, so
    /// that a failure that lasts does not cost every over-cap store a write
    /// transaction and its busy timeout. The records stay counted meanwhile,
    /// and are still offered to no quota pass.
    ///
    /// `reconciling` is the import, which already holds the write lock in
    /// its own transaction, and which settles every companion from what is
    /// on disk — so nothing is handed over here. Its transaction is not this
    /// method's to roll back: a record whose statement fails is skipped, the
    /// others still go, and only what really went is counted and reported.
    ///
    /// Caller holds `lock`, and no statement is still stepping.
    @discardableResult
    private func _retireInvalidRecordsLocked(
        _ found: InvalidRecords, reconciling: Bool = false
    ) -> (entries: Int, links: Int, legacy: Int) {
        guard let db, !found.isEmpty, !indexIsFromANewerBuild else { return (0, 0, 0) }
        if !reconciling {
            // A wait longer than the interval is not one this cache set: the
            // wall clock has gone backwards since, and the back-off is over.
            if let retireNotBefore {
                let wait = retireNotBefore.timeIntervalSince(now())
                if wait > 0, wait <= retireRetryInterval { return (0, 0, 0) }
            }
            retireAttemptsForTesting += 1
            guard sqlite3_exec(db, "BEGIN IMMEDIATE", nil, nil, nil) == SQLITE_OK else {
                _retirementFailedLocked(String(cString: sqlite3_errmsg(db)), backingOff: true)
                return (0, 0, 0)
            }
        }
        var retired = (entries: 0, links: 0, legacy: 0)
        var reports: [(shown: String, kind: InvalidIndexValueKind)] = []
        // The first failure, in SQLite's words. It ends a retirement of this
        // method's own; the import goes on to the next record.
        var failure: String?
        var stopped: Bool { failure != nil && !reconciling }
        func succeeded(_ ok: Bool) -> Bool {
            if !ok, failure == nil { failure = String(cString: sqlite3_errmsg(db)) }
            return ok
        }
        func run(_ sql: String, _ rowid: Int64) -> Bool {
            succeeded(_runLocked(sql, [.int(rowid)]) == SQLITE_DONE)
        }

        for rowid in found.entries where !stopped {
            var hash = IndexValue.unreadable
            var key = IndexValue.null
            let wasRead = _queryLocked(
                indexHasV2Columns
                    ? "SELECT hash, companion_key FROM cache_entries WHERE rowid = ?"
                    : "SELECT hash FROM cache_entries WHERE rowid = ?",
                [.int(rowid)]
            ) { stmt in
                hash = Self.indexValue(stmt, column: 0, hexDigits: Self.hashLength)
                if self.indexHasV2Columns {
                    key = Self.indexValue(
                        stmt, column: 1, hexDigits: SSMCompanionDiskStore.keyLength)
                }
            }
            guard succeeded(wasRead) else { continue }
            var rowReports: [(shown: String, kind: InvalidIndexValueKind)]
            switch hash {
            case .null: rowReports = [("NULL", .hash)]
            case .invalid(let rendering): rowReports = [(rendering, .hash)]
            case .valid, .unreadable: continue
            }
            switch key {
            case .valid where !reconciling:
                guard run(Self.moveLinkedCompanionsToLegacySQL + " WHERE rowid = ?", rowid)
                else { continue }
            case .invalid(let rendering):
                rowReports.append((rendering, .companionKey))
            default:
                break
            }
            guard run("DELETE FROM cache_entries WHERE rowid = ?", rowid) else { continue }
            reports += rowReports
            retired.entries += 1
        }
        for rowid in found.links where !stopped && indexHasV2Columns {
            var shown: String?
            let wasRead = _queryLocked(
                "SELECT hash, companion_key, companion_bytes FROM cache_entries WHERE rowid = ?",
                [.int(rowid)]
            ) { stmt in
                guard case .valid = Self.indexValue(stmt, column: 0, hexDigits: Self.hashLength)
                else { return }
                switch Self.indexValue(
                    stmt, column: 1, hexDigits: SSMCompanionDiskStore.keyLength)
                {
                case .invalid(let rendering): shown = rendering
                case .null where sqlite3_column_int64(stmt, 2) != 0: shown = "NULL"
                default: break
                }
            }
            guard succeeded(wasRead), let shown else { continue }
            guard
                run(
                    "UPDATE cache_entries SET companion_key = NULL, companion_bytes = 0 WHERE rowid = ?",
                    rowid)
            else { continue }
            reports.append((shown, .companionKey))
            retired.links += 1
        }
        for rowid in found.legacy where !stopped && indexHasV2Columns {
            var shown: String?
            let wasRead = _queryLocked(
                "SELECT key FROM legacy_companions WHERE rowid = ?", [.int(rowid)]
            ) { stmt in
                switch Self.indexValue(
                    stmt, column: 0, hexDigits: SSMCompanionDiskStore.keyLength)
                {
                case .invalid(let rendering): shown = rendering
                case .null: shown = "NULL"
                default: break
                }
            }
            guard succeeded(wasRead), let shown else { continue }
            guard run("DELETE FROM legacy_companions WHERE rowid = ?", rowid) else { continue }
            reports.append((shown, .companionKey))
            retired.legacy += 1
        }

        if !reconciling {
            if failure == nil {
                _ = succeeded(sqlite3_exec(db, "COMMIT", nil, nil, nil) == SQLITE_OK)
            }
            if let failure {
                sqlite3_exec(db, "ROLLBACK", nil, nil, nil)
                _retirementFailedLocked(failure, backingOff: true)
                return (0, 0, 0)
            }
            retireNotBefore = nil
        } else if let failure {
            _retirementFailedLocked(failure, backingOff: false)
        }
        // Reported once it is true.
        for report in reports {
            Self.reportInvalidIndexValue(shown: report.shown, kind: report.kind)
        }
        return retired
    }

    /// A retirement changed nothing (or, in the import, less than it meant
    /// to). Counted every time; said once per root and reason per process —
    /// a failure that lasts would otherwise be a line per store.
    ///
    /// `reason` is SQLite's message, which a trigger in the index can word:
    /// it is shown — and remembered — bounded and escaped, like every other
    /// value that comes out of the index.
    private func _retirementFailedLocked(_ reason: String, backingOff: Bool) {
        let reason = Self.boundedRendering(of: reason)
        failedIndexWrites += 1
        if backingOff { retireNotBefore = now().addingTimeInterval(retireRetryInterval) }
        guard Self.isFirstReport("\(cacheDir.path)\u{0}\(reason)", in: Self.reportedRetireFailures)
        else { return }
        FileHandle.standardError.write(Data(
            ("[vmlx][cache/disk-index] could not retire records that name nothing: \(reason) — "
                + "they stay counted, no quota pass is offered them, and nothing real is "
                + "evicted for them\n").utf8))
    }

    /// The bytes a newer build's records hold that this build can neither
    /// read nor evict, as an over-cap pass found them (0 when it found
    /// none). Kept for ``DiskCacheStats/opaqueBytes``, and said once per
    /// root per process: they come off the cap this build's own rows share,
    /// and once they reach it every store is evicted again straight away —
    /// which nothing else would ever explain.
    func noteOpaqueBytes(_ bytes: Int64, capBytes: Int64) {
        lock.lock()
        defer { lock.unlock() }
        _noteOpaqueBytesLocked(bytes, capBytes: capBytes)
    }

    private func _noteOpaqueBytesLocked(_ bytes: Int64, capBytes: Int64) {
        lastOpaqueBytes = IndexedBytes.clamped(bytes)
        reportOpaqueBytes(bytes, capBytes: capBytes, of: "cap")
    }

    /// The line alone: for the companion store, whose own cap is not the
    /// figure the stats report. Takes no lock.
    func reportOpaqueBytes(_ bytes: Int64, capBytes: Int64, of what: String) {
        guard bytes > 0,
            Self.reportedOpaqueRoots.withLock({ $0.insert(cacheDir.path).inserted })
        else { return }
        FileHandle.standardError.write(Data(
            ("[vmlx][cache/disk-index] rows of a newer build hold \(bytes) bytes that this build "
                + "counts but cannot evict, leaving \(IndexedBytes.difference(capBytes, bytes)) of "
                + "the \(capBytes)-byte \(what) for its own (index schema version "
                + "\(indexSchemaVersion), this build's \(DiskCacheIndexSchema.currentVersion))\n")
                .utf8))
    }

    enum InvalidIndexValueKind: String {
        case hash = "dropped row with an invalid hash"
        case companionKey = "forgot companion with an invalid key"
    }

    /// One line per distinct value per process, and only for the first
    /// ``rateLimitedReportLimit`` distinct values: a damaged index with a
    /// million bad rows is not a million lines. `shown` is already bounded
    /// and escaped (``boundedRendering(of:asText:)``): the value is hostile
    /// by definition.
    static func reportInvalidIndexValue(shown: String, kind: InvalidIndexValueKind) {
        guard isFirstReport("\(kind.rawValue)\u{0}\(shown)", in: reportedInvalidIndexValues)
        else { return }
        FileHandle.standardError.write(Data(
            ("[vmlx][cache/disk-index] \(kind.rawValue) \(shown) — it names no file of this "
                + "cache; no file was looked at or removed\n").utf8))
    }

    static func reportInvalidIndexValue(_ value: String, kind: InvalidIndexValueKind) {
        reportInvalidIndexValue(shown: boundedRendering(of: value), kind: kind)
    }

    static let rateLimitedReportLimit = 8
    private static let reportedInvalidIndexValues = OSAllocatedUnfairLock(initialState: Set<String>())
    private static let reportedUnreadablePayloads = OSAllocatedUnfairLock(initialState: Set<String>())
    private static let reportedRetireFailures = OSAllocatedUnfairLock(initialState: Set<String>())
    /// Roots whose opaque bytes ``reportOpaqueBytes(_:capBytes:of:)`` has said.
    private static let reportedOpaqueRoots = OSAllocatedUnfairLock(initialState: Set<String>())
    /// Roots whose model/tokens index could not be created at open.
    private static let reportedIndexCreationFailures = OSAllocatedUnfairLock(
        initialState: Set<String>())

    private static func isFirstReport(
        _ value: String, in reported: OSAllocatedUnfairLock<Set<String>>
    ) -> Bool {
        reported.withLock { seen in
            guard seen.count < rateLimitedReportLimit else { return false }
            return seen.insert(value).inserted
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
        return _statsLocked(bytes: IndexedBytes.asInt(usage.bytes), entryCount: usage.entryCount)
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
            let rowid: Int64
            let hash: String
            /// nil when the row has none — or a link that names nothing (a
            /// key that is not a companion key, or bytes with no key at
            /// all), which `invalidCompanionKey` then renders.
            var companionKey: String? = nil
            let companionBytes: Int64
            var invalidCompanionKey: String? = nil
        }
        var rows: [Row] = []
        // Rows whose hash is not a payload hash — TEXT of exactly 32 hex
        // BYTES (``IndexValue``): NULL, a BLOB, an embedded NUL. They name no
        // file, so none is looked at; they go by rowid below, or, under a
        // newer build's index, stay exactly as they are.
        var invalid = InvalidRecords()
        // Real companions those rows link to. Under the current schema they
        // are looked at like any other the index names; under a newer one
        // they are the opaque row's, and are not counted a second time.
        var companionsOfInvalidRows: [(key: String, bytes: Int64)] = []
        var sawUnreadableColumn = false
        let rowsWereRead = _queryLocked(
            "SELECT hash, companion_key, companion_bytes, rowid FROM cache_entries"
        ) { stmt in
            // `.unreadable` is an allocation failure, not a value. A failed
            // read of a hash would make its payload look unindexed.
            let hash = Self.indexValue(stmt, column: 0, hexDigits: Self.hashLength)
            let key = Self.indexValue(
                stmt, column: 1, hexDigits: SSMCompanionDiskStore.keyLength)
            if hash == .unreadable || key == .unreadable {
                sawUnreadableColumn = true
                return
            }
            let rowid = sqlite3_column_int64(stmt, 3)
            let companionBytes = sqlite3_column_int64(stmt, 2)
            guard case .valid(let validHash) = hash else {
                invalid.entries.append(rowid)
                if case .valid(let validKey) = key {
                    companionsOfInvalidRows.append((validKey, companionBytes))
                }
                return
            }
            var row = Row(rowid: rowid, hash: validHash, companionBytes: companionBytes)
            switch key {
            case .valid(let validKey): row.companionKey = validKey
            case .invalid(let shown): row.invalidCompanionKey = shown
            case .null where companionBytes != 0: row.invalidCompanionKey = "NULL"
            case .null, .unreadable: break
            }
            rows.append(row)
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
        let rowsAreOpaque = indexIsFromANewerBuild
        if rowsAreOpaque {
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
            // Every hash in `rows` is a payload hash; this cannot fail.
            guard let url = safetensorsURL(for: row.hash) else { continue }
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
        let legacyRead = _readLegacyCompanionsLocked()
        let recordedLegacy = legacyRead.valid
        invalid.legacy = legacyRead.invalidRowids
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
        if !rowsAreOpaque {
            named += companionsOfInvalidRows.map { Named(key: $0.key, kvHash: nil, bytes: $0.bytes) }
        }
        let companionDirectory = walkedDirectory ?? self.companionDirectory
        // Every key in `named` is a companion key: one that is not was never
        // read into a String, is never looked for on disk, and what follows
        // clears the link that carries it.
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
            _runLocked("DELETE FROM cache_entries WHERE rowid = ?", [.int(row.rowid)])
            validatedFiles.removeValue(forKey: row.hash)
            summary.rowsDeletedForMissingPayload += 1
        }
        // By rowid, inside this transaction; nothing under a newer build's
        // index. A dropped row's companion, if real, is on disk and in
        // `reconciled` with no live row to join, so it is counted as
        // unlinked below.
        let retired = _retireInvalidRecordsLocked(invalid, reconciling: true)
        summary.rowsDroppedForInvalidHash += retired.entries
        summary.legacyDroppedForInvalidKey += retired.legacy

        // One companion per row. Keys are content-addressed over the same
        // tokens as the row hash, so a second claimant cannot arise from this
        // build; if one exists anyway it is counted as unlinked.
        //
        // Under a newer build's index a link this build cannot read stays as
        // it is: its row takes no other companion, and a companion an opaque
        // row links to is that row's — not an unlinked one to count again
        // and evict from under it.
        let opaqueLinkHashes = rowsAreOpaque
            ? Set(liveRows.filter { $0.invalidCompanionKey != nil }.map(\.hash)) : []
        let opaquelyLinkedKeys = rowsAreOpaque ? Set(companionsOfInvalidRows.map(\.key)) : []
        let liveHashes = Set(liveRows.map(\.hash)).subtracting(opaqueLinkHashes)
        var linkByHash: [String: SSMCompanionQuotaEntry] = [:]
        var unlinked: [SSMCompanionQuotaEntry] = []
        for companion in reconciled.sorted(by: { $0.hash < $1.hash })
        where !opaquelyLinkedKeys.contains(companion.hash) {
            if let kvHash = companion.kvHash, liveHashes.contains(kvHash),
               linkByHash[kvHash] == nil
            {
                linkByHash[kvHash] = companion
            } else {
                unlinked.append(companion)
            }
        }

        for row in liveRows where !opaqueLinkHashes.contains(row.hash) {
            if let shown = row.invalidCompanionKey {
                Self.reportInvalidIndexValue(shown: shown, kind: .companionKey)
            }
            if let companion = linkByHash[row.hash] {
                let bytes = max(0, companion.bytes)
                if row.companionKey != companion.hash || row.companionBytes != bytes {
                    _linkCompanionLocked(
                        kvHash: row.hash, companionKey: companion.hash, bytes: bytes)
                    summary.linksWritten += 1
                }
            } else if row.companionKey != nil || row.invalidCompanionKey != nil {
                _runLocked(
                    "UPDATE cache_entries SET companion_key = NULL, companion_bytes = 0 WHERE rowid = ?",
                    [.int(row.rowid)])
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
        guard let (hash, url) = entryKey(tokens: tokens, mediaSalt: mediaSalt) else { return false }
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
            // Not a payload hash: no path, no file, and nothing "evicted"
            // — it is not in the returned set, so no bytes are reported as
            // reclaimed for it. `quotaEntries()` never offers such a value
            // (it retires the row that carries it, by rowid); this is the
            // second layer, and a String cannot address that row anyway.
            guard let url = safetensorsURL(for: hash) else {
                Self.reportInvalidIndexValue(hash, kind: .hash)
                continue
            }
            let payloadGone = Self.removeCacheFile(at: url)
            validatedFiles.removeValue(forKey: hash)

            var companionRemoved = false
            if indexHasV2Columns {
                _queryLocked(
                    "SELECT companion_key FROM cache_entries WHERE hash = ?", [.text(hash)]
                ) { stmt in
                    if case .valid(let key) = Self.indexValue(
                        stmt, column: 0, hexDigits: SSMCompanionDiskStore.keyLength)
                    {
                        companionRemoved = removedCompanions.contains(key)
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
    /// Only a regular file is removed (``removeRegularFile(at:)``). The
    /// paths given here are built from a hash or a key, so the name is ours;
    /// a directory or a link that has taken that name is not the cache's
    /// file. It is left alone and reported as still there, like a file that
    /// could not be deleted — the same answer the import gives for such a
    /// row.
    ///
    /// A file that is still there is reported under its own tag — every
    /// `[vmlx][cache/disk-quota]` line is a pass summary beginning
    /// `before= after= max=`, and a log parser relies on that — and once per
    /// path per process: a file that can never be deleted is tried again by
    /// every over-cap store.
    static func removeCacheFile(at url: URL) -> Bool {
        let reason: String
        switch removeRegularFile(at: url) {
        case .removed, .missing:
            return true
        case .notRegularFile:
            reason = "not a regular file, left alone"
        case .failed(let code):
            reason = String(cString: strerror(code))
        }
        let firstReport = reportedDeleteFailures.withLock { $0.insert(url.path).inserted }
        if firstReport {
            FileHandle.standardError.write(Data(
                ("[vmlx][cache/disk-delete] failed path=\(url.path) "
                    + "error=\(reason) — row kept\n").utf8))
        }
        return false
    }

    /// Forget what the once-per-process log lines have already reported, so
    /// a test can see a line that an earlier test in the same process caused.
    static func resetRateLimitedReportsForTesting() {
        reportedInvalidIndexValues.withLock { $0.removeAll() }
        reportedUnreadablePayloads.withLock { $0.removeAll() }
        reportedDeleteFailures.withLock { $0.removeAll() }
        reportedRetireFailures.withLock { $0.removeAll() }
        reportedOpaqueRoots.withLock { $0.removeAll() }
        reportedIndexCreationFailures.withLock { $0.removeAll() }
    }

    /// Paths ``removeCacheFile(at:)`` has already reported in this process.
    private static let reportedDeleteFailures = OSAllocatedUnfairLock(initialState: Set<String>())

    /// Forget every entry and remove this cache's payload files.
    ///
    /// The root is a user setting and may hold files that are not this
    /// cache's — a model's shards, if it was pointed at a model folder — so
    /// this removes by allow-list, and whatever is in doubt stays:
    ///
    /// - a name must be exactly a published payload's
    ///   (``isPublishedPayloadName(_:)``: also what a crash between publish
    ///   and insert leaves) or exactly a dead write's
    ///   (``isUnpublishedPayloadName(_:)``);
    /// - the entry must be a regular file by `lstat` — never a directory,
    ///   never a symlink — and goes by `unlink`.
    ///
    /// In a root that looks like a model bundle (``modelBundleMarkers``),
    /// or whose index a newer build has claimed
    /// (``indexIsFromANewerBuild``),
    /// nothing is removed from a LISTING of it, as in every other sweep: a
    /// payload without a row and a dead partial stay, and one
    /// `[vmlx][cache/disk-index] clear skipped:` line says so. The payloads
    /// the index names still go, by the path built from each row's hash:
    /// this cache wrote each of them under that name (a store replaces
    /// whatever regular file held it), and the quota pass removes the same
    /// files in the same root. Keeping them would leak them for good —
    /// their rows are dropped here, and the unindexed-payload sweep skips a
    /// bundle root too. The same applies when the root cannot be listed.
    /// A row's hash becomes a path only when it is a payload hash
    /// (``safetensorsURL(for:)``'s rule); any other row names no file and is
    /// simply deleted with the rest.
    ///
    /// The index is emptied and the counters reset in every case, so the
    /// cache references nothing afterwards. A payload of ours that could not
    /// be unlinked is left without a row; the import's unindexed-payload
    /// sweep takes it once it is old enough (not in a bundle root).
    public func clear() {
        MLXDiskCacheIOLock.shared.lock()
        defer { MLXDiskCacheIOLock.shared.unlock() }

        lock.lock()
        defer { lock.unlock() }

        // Files first, rows after: dying in between leaves rows that name
        // missing files, which the next fetch or import clears — never
        // files the index has stopped counting.
        var doomed = Set<String>()
        _ = _queryLocked("SELECT hash FROM cache_entries") { stmt in
            // A row's hash is data; only one that is a payload hash, byte
            // for byte, becomes a name. Every row is deleted below either way.
            switch Self.indexValue(stmt, column: 0, hexDigits: Self.hashLength) {
            case .valid(let hash): doomed.insert(hash + Self.payloadSuffix)
            case .invalid(let shown): Self.reportInvalidIndexValue(shown: shown, kind: .hash)
            case .null: Self.reportInvalidIndexValue(shown: "NULL", kind: .hash)
            case .unreadable: break
            }
        }
        var skipped: String?
        if indexIsFromANewerBuild {
            skipped =
                "index schema version \(indexSchemaVersion) is newer than this build's "
                + "\(DiskCacheIndexSchema.currentVersion)"
        } else {
            do {
                let names = try FileManager.default.contentsOfDirectory(atPath: cacheDir.path)
                if let marker = Self.modelBundleMarker(in: names) {
                    skipped = "cache root holds \(marker), so it looks like a model bundle"
                } else {
                    doomed.formUnion(names.filter {
                        Self.isPublishedPayloadName($0) || Self.isUnpublishedPayloadName($0)
                    })
                }
            } catch {
                skipped = "cache root could not be listed: \(error.localizedDescription)"
            }
        }
        if let skipped {
            FileHandle.standardError.write(Data(
                ("[vmlx][cache/disk-index] clear skipped: \(skipped) — nothing is removed from "
                    + "a listing of it, only the \(doomed.count) payload(s) the index names\n").utf8))
        }
        var leftBehind = 0
        for name in doomed {
            if case .failed = Self.removeRegularFile(at: cacheDir.appendingPathComponent(name)) {
                leftBehind += 1
            }
        }
        if leftBehind > 0 {
            FileHandle.standardError.write(Data(
                "[vmlx][cache/disk-index] clear could not remove \(leftBehind) payload(s)\n".utf8))
        }

        // This cache does not own the companion files. Rows that carried one
        // hand it to the unlinked list so its bytes stay counted until the
        // companion store clears or evicts it.
        if indexHasV2Columns {
            _runLocked(Self.moveLinkedCompanionsToLegacySQL + " WHERE companion_key IS NOT NULL")
        }
        executeSQL("DELETE FROM cache_entries")

        // Reset stats
        hits = 0
        misses = 0
        stores = 0
        storeSkips = 0
        failedIndexWrites = 0
        unreadablePayloadFetches = 0
        evictions = 0
        quotaEvictedBytes = 0
        quotaPasses = 0
        lastQuotaPassMs = 0
        pressureEventSeq = 0
        lastPressureEvent = nil
        lastOpaqueBytes = 0
        retireNotBefore = nil
        validatedFiles.removeAll(keepingCapacity: true)
        rejectedRestores.removeAll()
        rewrittenAfterRejection.removeAll()
        servedCandidates.removeAll()
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

    /// ``hashTokens(_:modelKey:mediaSalt:)`` returns exactly this many
    /// lowercase hex digits.
    static let hashLength = 32

    /// Whether `text` could be a hash this cache computed.
    static func isPayloadHash(_ text: String) -> Bool {
        isLowercaseHex(text, count: hashLength)
    }

    /// The payload path for `hash`, or nil when `hash` is not a payload
    /// hash. Every path that `fetch`, a store, a quota pass or the import
    /// builds from a hash comes from here, which is why a hash read from the
    /// index can be handed to it: `../../x`, an empty string or
    /// `model-00001-of-00008` gets no path at all. (`clear()` spells the
    /// name itself, to put it in one set with the names it lists — behind
    /// the same test, ``IndexValue/valid(_:)``.)
    private func safetensorsURL(for hash: String) -> URL? {
        guard Self.isPayloadHash(hash) else { return nil }
        return cacheDir.appendingPathComponent(hash + Self.payloadSuffix)
    }

    /// The hash of a token prefix and the path of its payload.
    /// ``hashTokens(_:modelKey:mediaSalt:)`` produces a payload hash by
    /// construction, which DEBUG builds assert; a build in which that
    /// stopped being true would miss, not build a path from the value.
    private func entryKey(tokens: [Int], mediaSalt: String?) -> (hash: String, url: URL)? {
        let hash = DiskCache.hashTokens(tokens, modelKey: modelKey, mediaSalt: mediaSalt)
        let url = safetensorsURL(for: hash)
        assert(url != nil, "hashTokens produced \(hash), which is not a payload hash")
        return url.map { (hash, $0) }
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

    /// Whether `name` carries the `.partial-` infix at all. Loose on purpose:
    /// it answers "this is not a published entry" for a walk that counts,
    /// and is never a reason to delete. Deleters use
    /// ``isUnpublishedPayloadName(_:)`` (or the companion store's
    /// equivalent), which also require that the name be one of ours.
    static func isUnpublishedName(_ name: String) -> Bool {
        name.contains(".partial-") && name.hasSuffix(".safetensors")
    }

    static let payloadSuffix = ".safetensors"
    static let partialInfix = ".partial-"
    /// `UUID().uuidString.prefix(8)` in ``temporaryURL(for:)``.
    static let partialTagLength = 8

    private static func isLowerHexDigit(_ byte: UInt8) -> Bool {
        (byte >= 0x30 && byte <= 0x39) || (byte >= 0x61 && byte <= 0x66)
    }

    /// `count` lowercase hex digits and nothing else: a content hash as
    /// ``hashTokens(_:modelKey:mediaSalt:)`` (32) and the companion store's
    /// key (64) spell it.
    static func isLowercaseHex<S: StringProtocol>(_ text: S, count: Int) -> Bool {
        guard text.utf8.count == count else { return false }
        return text.withCString { isLowercaseHex(cString: $0, count: count) }
    }

    private static let lowercaseHexDigits: StaticString = "0123456789abcdef"

    /// The same test on a NUL-terminated value, such as a column as SQLite
    /// hands it out. `strspn`, not a Swift loop: every row of an over-cap
    /// quota pass goes through this, and an unoptimized build runs a
    /// byte-by-byte Swift loop some fifty times slower than libc does.
    static func isLowercaseHex(cString: UnsafePointer<CChar>, count: Int) -> Bool {
        let accept = UnsafeRawPointer(lowercaseHexDigits.utf8Start)
            .assumingMemoryBound(to: CChar.self)
        // `count` leading hex digits, none of them the terminator, so the
        // byte at `count` is still inside the string.
        return strspn(cString, accept) == count && cString[count] == 0
    }

    /// `sqlite3_column_text` is `unsigned char *`.
    static func isLowercaseHex(cString: UnsafePointer<UInt8>, count: Int) -> Bool {
        isLowercaseHex(
            cString: UnsafeRawPointer(cString).assumingMemoryBound(to: CChar.self), count: count)
    }

    /// Splits `<stem>.partial-<tag>.safetensors` — the shape
    /// ``temporaryURL(for:)`` gives a file that is still being written — and
    /// returns the stem, or nil when `name` is not exactly that: one
    /// `.partial-` infix, then a tag of ``partialTagLength`` hex digits (a
    /// UUID's first eight; uppercase as Foundation prints them, either case
    /// accepted), then the suffix. The caller decides whether the stem is
    /// one of its own.
    static func unpublishedStem(ofName name: String) -> Substring? {
        guard name.hasSuffix(payloadSuffix) else { return nil }
        let body = name.dropLast(payloadSuffix.count)
        guard let infix = body.range(of: partialInfix) else { return nil }
        let tag = body[infix.upperBound...]
        guard tag.utf8.count == partialTagLength,
              tag.utf8.allSatisfy({
                  isLowerHexDigit($0) || (UInt8(ascii: "A")...UInt8(ascii: "F")).contains($0)
              })
        else { return nil }
        return body[..<infix.lowerBound]
    }

    /// Whether `name` is exactly what ``temporaryURL(for:)`` produces for one
    /// of this cache's payloads: `<32 lowercase hex>.partial-<8 hex>.safetensors`.
    /// `random.partial-abcdefgh.safetensors` in the same root is somebody
    /// else's download, not a dead write of ours.
    static func isUnpublishedPayloadName(_ name: String) -> Bool {
        guard let stem = unpublishedStem(ofName: name) else { return false }
        return isLowercaseHex(stem, count: 32)
    }

    /// Whether `name` is exactly what ``safetensorsURL(for:)`` produces for a
    /// hash from ``hashTokens(_:modelKey:mediaSalt:)``: 32 lowercase hex
    /// digits and the suffix, nothing else. This is the only test of "is
    /// this file ours" that anything deleting from a listing of the root may
    /// use. Uppercase hex, another length, a `.partial-` name, a model's
    /// `model-00001-of-00008.safetensors` — none of them is ours.
    static func isPublishedPayloadName(_ name: String) -> Bool {
        guard name.hasSuffix(payloadSuffix) else { return false }
        return isLowercaseHex(name.dropLast(payloadSuffix.count), count: 32)
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
        // `errno` is read inside the closure, next to the call that set it:
        // by the time the closure has returned, the path buffer has been
        // released, and that may have overwritten it.
        let code = url.withUnsafeFileSystemRepresentation { path -> Int32 in
            guard let path else { return EINVAL }
            return lstat(path, &info) == 0 ? 0 : errno
        }
        guard code == 0 else {
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
        url.withUnsafeFileSystemRepresentation { path -> Int32 in
            guard let path else { return EINVAL }
            return unlink(path) == 0 ? 0 : errno
        }
    }

    /// `rename(2)`: atomic, replaces a regular file, replaces a symlink
    /// ITSELF rather than its target, and fails with EISDIR / ENOTDIR on a
    /// directory instead of descending into it. (`FileManager.moveItem`
    /// refuses an existing destination, and `replaceItemAt` is not this
    /// call.) Returns 0, or the errno.
    static func renameFile(from source: URL, to destination: URL) -> Int32 {
        source.withUnsafeFileSystemRepresentation { from -> Int32 in
            destination.withUnsafeFileSystemRepresentation { to -> Int32 in
                guard let from, let to else { return EINVAL }
                return rename(from, to) == 0 ? 0 : errno
            }
        }
    }

    /// What became of one path that carries one of this cache's names.
    enum OwnFileRemoval: Equatable {
        case removed
        /// A definite "no such file".
        case missing
        /// A directory, a symlink or anything else that is not a regular
        /// file holds the name. It is not ours, and it was left alone.
        case notRegularFile
        /// It could not be examined, or `unlink` failed.
        case failed(errno: Int32)
    }

    /// Remove `url` if, and only if, a regular file holds that name: `lstat`
    /// (a link is never followed), then `unlink`, which cannot descend into
    /// a directory that took the name in between. Every removal in this
    /// cache and in the companion store goes through here; which NAMES may
    /// be removed is the caller's business.
    static func removeRegularFile(at url: URL) -> OwnFileRemoval {
        switch pathState(at: url) {
        case .missing:
            return .missing
        case .notRegularFile:
            return .notRegularFile
        case .unreadable(let code):
            return .failed(errno: code)
        case .regularFile:
            let code = unlinkFile(at: url)
            if code == 0 { return .removed }
            return code == ENOENT ? .missing : .failed(errno: code)
        }
    }

    /// What reading a file's safetensors header established.
    enum PayloadInspection: Equatable {
        /// The header parsed and the file holds every byte it declares.
        case complete
        /// The file was opened and read, and it is positively not a whole
        /// safetensors file: too short for its header, a header that is not
        /// the JSON it should be, or fewer bytes than the header declares.
        case shortOrMalformed
        /// The file could not be opened or read (EACCES, EMFILE, EIO, …).
        /// That says nothing about its contents, and is never a reason to
        /// delete it.
        case unreadable(errno: Int32)
    }

    private enum HeaderRead {
        case header(length: Int, tensors: [String: Any], fileSize: Int)
        case malformed
        case unreadable(errno: Int32)
    }

    /// POSIX calls rather than `FileHandle`, so that the reason a read failed
    /// is an errno and "could not read" stays apart from "read, and wrong".
    ///
    /// Only a regular file is read. `O_NONBLOCK` is there for what is not
    /// one: without it, `open` on a FIFO waits for a writer that never comes
    /// (it changes nothing for a regular file).
    private static func readHeader(url: URL) -> HeaderRead {
        let opened = url.withUnsafeFileSystemRepresentation { path -> (fd: Int32, errno: Int32) in
            guard let path else { return (-1, EINVAL) }
            let fd = open(path, O_RDONLY | O_CLOEXEC | O_NONBLOCK)
            return (fd, fd < 0 ? errno : 0)
        }
        guard opened.fd >= 0 else { return .unreadable(errno: opened.errno) }
        let fd = opened.fd
        defer { close(fd) }

        var info = stat()
        guard fstat(fd, &info) == 0 else { return .unreadable(errno: errno) }
        guard (info.st_mode & S_IFMT) == S_IFREG else {
            #if canImport(Darwin)
            return .unreadable(errno: EFTYPE)
            #else
            return .unreadable(errno: EINVAL)
            #endif
        }

        // nil: the read failed, and `readFailure` says why. Fewer bytes than
        // asked for: end of file.
        var readFailure: Int32 = 0
        func readBytes(_ count: Int) -> Data? {
            var data = Data(count: count)
            var filled = 0
            while filled < count {
                let (got, code) = data.withUnsafeMutableBytes { buffer -> (Int, Int32) in
                    let got = read(fd, buffer.baseAddress! + filled, count - filled)
                    return (got, got < 0 ? errno : 0)
                }
                if got < 0 {
                    if code == EINTR { continue }
                    readFailure = code
                    return nil
                }
                if got == 0 { break }
                filled += got
            }
            return data.prefix(filled)
        }

        guard let lengthData = readBytes(8) else { return .unreadable(errno: readFailure) }
        guard lengthData.count == 8 else { return .malformed }
        let declared = lengthData.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self).littleEndian }
        // Against the file's own size BEFORE anything is allocated for it: a
        // length field is eight hostile bytes, and the buffer below is
        // zero-filled. A header that does not fit in its file is malformed.
        let available = Int64(info.st_size) - 8
        guard declared > 0, declared < 256 * 1024 * 1024, available > 0,
            declared <= UInt64(available)
        else { return .malformed }
        let headerLength = Int(declared)
        guard let headerData = readBytes(headerLength) else {
            return .unreadable(errno: readFailure)
        }
        guard headerData.count == headerLength,
            let tensors = try? JSONSerialization.jsonObject(with: headerData) as? [String: Any]
        else { return .malformed }
        return .header(length: headerLength, tensors: tensors, fileSize: Int(info.st_size))
    }

    /// One past the last tensor byte the header declares, relative to the
    /// start of the file; nil when an entry is not a tensor description —
    /// which includes offsets that are not whole numbers, are negative, run
    /// backwards, or do not fit in an `Int` once the header is added. The
    /// header is a file's contents, and the file is whatever sits in a
    /// directory the user chose: `[0, 9223372036854775807]` made the sum
    /// below TRAP, in a sweep that runs at every open.
    static func declaredPayloadEnd(headerLength: Int, tensors: [String: Any]) -> Int? {
        func offset(_ value: Any) -> Int? {
            guard let number = value as? NSNumber else { return nil }
            // Exactly an Int64: 15.5, 1e300 and 2^64-1 all have an
            // `int64Value`, and none of them round-trips.
            let whole = number.int64Value
            guard NSNumber(value: whole) == number, whole >= 0 else { return nil }
            return Int(exactly: whole)
        }
        guard headerLength >= 0 else { return nil }
        var end = 0
        for (key, value) in tensors where key != "__metadata__" {
            guard let tensor = value as? [String: Any],
                let offsets = tensor["data_offsets"] as? [Any], offsets.count == 2,
                let first = offset(offsets[0]), let last = offset(offsets[1]), last >= first
            else { return nil }
            end = max(end, last)
        }
        let (prefix, prefixOverflow) = headerLength.addingReportingOverflow(8)
        let (total, totalOverflow) = prefix.addingReportingOverflow(end)
        return prefixOverflow || totalOverflow ? nil : total
    }

    /// The byte offset one past the last tensor payload the file's own
    /// safetensors header declares (8-byte little-endian header length, JSON
    /// header, `data_offsets: [begin, end]` per tensor relative to the end
    /// of the header), or nil when the header itself cannot be read.
    static func declaredPayloadEnd(url: URL) -> Int? {
        guard case .header(let length, let tensors, _) = readHeader(url: url) else { return nil }
        return declaredPayloadEnd(headerLength: length, tensors: tensors)
    }

    private static func tensorHeader(url: URL) -> (length: Int, tensors: [String: Any])? {
        guard case .header(let length, let tensors, _) = readHeader(url: url) else { return nil }
        return (length, tensors)
    }

    /// Header-only inspection; see ``PayloadInspection``. Only
    /// `.shortOrMalformed` permits a caller to delete the file.
    static func inspectSafetensors(url: URL) -> PayloadInspection {
        switch readHeader(url: url) {
        case .unreadable(let code):
            return .unreadable(errno: code)
        case .malformed:
            return .shortOrMalformed
        case .header(let length, let tensors, let fileSize):
            guard let declared = declaredPayloadEnd(headerLength: length, tensors: tensors),
                fileSize >= declared
            else { return .shortOrMalformed }
            return .complete
        }
    }

    /// True when the file on disk holds every byte its header declares.
    /// False says only "not shown to be complete" — it may not have been
    /// readable at all. Anything that deletes asks
    /// ``inspectSafetensors(url:)`` instead.
    static func isCompleteSafetensors(url: URL) -> Bool {
        inspectSafetensors(url: url) == .complete
    }

    /// How old a `.partial-` file must be before a sweep at open takes it
    /// for a dead write. A second coordinator can open a root while another
    /// model's store into it is still running — a large boundary takes
    /// seconds to write — and its partial is then a store in flight, not a
    /// dead write. Same figure, and same reasoning, as
    /// ``defaultUnindexedPayloadGuardAge``.
    static let defaultUnpublishedGuardAge: TimeInterval = 600

    /// Whether a sweep at open may take the `.partial-` file at `url` for a
    /// dead write: a regular file (by `lstat`) whose modification date is
    /// not in the future and is at least `guardAge` back.
    static func isDeadWrite(at url: URL, olderThan guardAge: TimeInterval, now: Date) -> Bool {
        guard case .regularFile(_, let modified) = pathState(at: url), modified <= now
        else { return false }
        return now.timeIntervalSince(modified) >= guardAge
    }

    /// Remove dead temp files and incomplete final-named rows from
    /// `cacheDir` (a row that names one is dropped by the next fetch of it,
    /// or by the import). Header-only reads: cheap even for a
    /// multi-hundred-GB cache.
    ///
    /// The root may hold files that are not this cache's. A dead write must
    /// carry exactly the name this cache gives one
    /// (``isUnpublishedPayloadName(_:)``) and an incomplete file exactly a
    /// published payload's (``isPublishedPayloadName(_:)``): somebody else's
    /// `x.partial-y.safetensors`, or a shard that is still downloading, is
    /// an "unpublished or incomplete safetensors file" too. Only regular
    /// files are considered (by `lstat`: never a directory, which cannot be
    /// read as a safetensors file and used to go recursively, and never a
    /// symlink), removal is by `unlink`, and a root that looks like a model
    /// bundle is left alone entirely.
    ///
    /// A `.partial-` file goes only once it is old enough to be dead
    /// (``isDeadWrite(at:olderThan:now:)``). A final-named file goes only
    /// when it was READ and found short of its own header
    /// (``PayloadInspection/shortOrMalformed``); one that could not be read
    /// has not been shown to be anything, and stays.
    static func sweepUnpublishedAndIncompleteFiles(
        in cacheDir: URL,
        unpublishedGuardAge: TimeInterval = DiskCache.defaultUnpublishedGuardAge,
        now: Date = Date()
    ) {
        guard let names = try? FileManager.default.contentsOfDirectory(atPath: cacheDir.path) else { return }
        if let marker = modelBundleMarker(in: names) {
            FileHandle.standardError.write(Data(
                ("[vmlx][cache/disk] integrity sweep skipped: cache root holds \(marker), "
                    + "so it looks like a model bundle\n").utf8))
            return
        }
        var removed = 0
        var unreadable = 0
        for name in names {
            let isUnpublished = isUnpublishedPayloadName(name)
            guard isUnpublished || isPublishedPayloadName(name) else { continue }
            let url = cacheDir.appendingPathComponent(name)
            if isUnpublished {
                guard isDeadWrite(at: url, olderThan: unpublishedGuardAge, now: now) else { continue }
                if unlinkFile(at: url) == 0 { removed += 1 }
                continue
            }
            guard case .regularFile = pathState(at: url) else { continue }
            switch inspectSafetensors(url: url) {
            case .complete:
                continue
            case .unreadable:
                unreadable += 1
            case .shortOrMalformed:
                guard unlinkFile(at: url) == 0 else { continue }
                removed += 1
                FileHandle.standardError.write(Data(
                    "[vmlx][cache/disk] removed incomplete row \(name) at open (short of its declared payload)\n".utf8))
            }
        }
        if removed > 0 {
            FileHandle.standardError.write(Data("[vmlx][cache/disk] integrity sweep removed \(removed) file(s)\n".utf8))
        }
        if unreadable > 0 {
            FileHandle.standardError.write(Data(
                ("[vmlx][cache/disk] integrity sweep could not read \(unreadable) payload(s); "
                    + "left alone\n").utf8))
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
    private func _insertEntryLocked(
        hash: String, tokenCount: Int, fileSize: Int,
        chainId: String? = nil, isStableRoot: Bool = false
    ) -> Int32 {
        guard db != nil else { return SQLITE_DONE }
        if indexHasV2Columns {
            // `kind` only ever rises (a root stays a root); `chain_id` follows
            // the latest owner but is never cleared by an ownerless re-store.
            return _runLocked(
                """
                INSERT INTO cache_entries (hash, token_count, file_size, model_key, kind, chain_id)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(hash) DO UPDATE SET
                    token_count = excluded.token_count,
                    file_size = excluded.file_size,
                    created_at = julianday('now'),
                    model_key = excluded.model_key,
                    kind = MAX(kind, excluded.kind),
                    chain_id = COALESCE(excluded.chain_id, chain_id)
                """,
                [
                    .text(hash), .int(Int64(tokenCount)), .int(Int64(fileSize)),
                    modelKey.map(SQLValue.text) ?? .null,
                    .int(isStableRoot ? 1 : 0),
                    chainId.map(SQLValue.text) ?? .null,
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

    // MARK: - Byte counts are data too
    //
    // A `file_size` is whatever is in the index. `1e19` stays REAL in the
    // INTEGER column; `2^63 - 1` and one more row made `SUM(file_size)` an
    // integer overflow — the statement FAILED, usage read as 0, and the cache
    // neither evicted nor retired; a negative size hid every other row.
    // So: a count is read through ``_indexedBytesLocked(_:_:)`` and summed
    // through ``IndexedBytes``, and a usage aggregate is ``_usageSQLLocked(_:)``.

    /// One byte count from the index: `sqlite3_column_int64` saturates a REAL,
    /// answers 0 for NULL, and reads TEXT or a BLOB as the integer it starts
    /// with (`'-9000000000abc'` is -9 000 000 000; no digits, 0). A negative
    /// count is 0 bytes — and from then on this cache sums with the clamping
    /// aggregate. Caller holds `lock`.
    private func _indexedBytesLocked(_ stmt: OpaquePointer, _ column: Int32) -> Int64 {
        let raw = sqlite3_column_int64(stmt, column)
        if raw < 0 { indexNeedsClampedUsage = true }
        return IndexedBytes.clamped(raw)
    }

    /// The usage aggregate over `columns`, which agrees with
    /// ``_indexedBytesLocked(_:_:)`` row by row. `TOTAL` sums in floating
    /// point, so it cannot overflow (it is exact below 2^53 bytes), and read
    /// back as an integer it saturates at `Int64.max`.
    ///
    /// Two forms, because this runs on every store and every stats poll. A
    /// clamp per row — `MAX(x, 0)`, a CASE, any spelling of it — costs a
    /// third of the whole statement (measured: 0.149 ms → 0.200 ms at 5 003
    /// rows), and buys nothing on an index that holds no negative or NULL
    /// count, which is every index this build writes. So the plain form is
    /// used until such a count has been SEEN: by the one check at open
    /// (``_indexHoldsCountsToClampLocked()``), or by any later read of the
    /// rows. A negative count that another writer plants while this process
    /// runs is therefore clamped from the next such read, or the next
    /// launch, and hides other rows' bytes until then.
    ///
    /// The clamping form is `MAX(CAST(x AS INTEGER), 0)`. The cast is what
    /// makes it agree with the row reads: a count can be TEXT or a BLOB
    /// (`'-9000000000abc'` is no number, so the INTEGER column keeps it),
    /// `MAX` of a TEXT and an INTEGER is the TEXT, whatever it says, and
    /// `TOTAL` then reads that as -9 000 000 000. Cast first, it is the same
    /// integer `sqlite3_column_int64` reads — a REAL saturates here too —
    /// and a negative one is 0. NULL stays NULL through both, which `TOTAL`
    /// skips; the columns are summed apart, so a NULL in one cannot swallow
    /// the other.
    private func _usageSQLLocked(_ columns: String...) -> String {
        indexNeedsClampedUsage
            ? columns.map { "TOTAL(MAX(CAST(\($0) AS INTEGER), 0))" }.joined(separator: " + ")
            : "TOTAL(\(columns.joined(separator: " + ")))"
    }

    /// A plain aggregate that comes back NEGATIVE has just shown that the
    /// index needs the clamping one: count again, once, with that.
    private func _countingAgainIfClampingBecameNecessary<T>(_ count: () -> T) -> T {
        let wasClamping = indexNeedsClampedUsage
        let result = count()
        return !wasClamping && indexNeedsClampedUsage ? count() : result
    }

    /// Whether the index holds a byte count the plain aggregate would get
    /// wrong: a negative one, a NULL (`x + NULL` is NULL, and takes the
    /// row's other count with it), or one that is not stored as an INTEGER
    /// (TEXT and BLOB compare greater than every number, so `x < 0` alone
    /// never finds `'-9000000000abc'`). One scan, at open; a read, so it
    /// answers while another connection holds the write lock. A check that
    /// could not be run has found nothing out, and answers true: the
    /// clamping form is right for every index, only slower.
    private func _indexHoldsCountsToClampLocked() -> Bool {
        func suspect(_ column: String) -> String {
            "CAST(\(column) AS INTEGER) < 0 OR \(column) IS NULL "
                + "OR typeof(\(column)) NOT IN ('integer')"
        }
        var found = false
        let answered = _queryLocked(
            indexHasV2Columns
                ? """
                    SELECT EXISTS(SELECT 1 FROM cache_entries
                                  WHERE \(suspect("file_size")) OR \(suspect("companion_bytes")))
                        OR EXISTS(SELECT 1 FROM legacy_companions WHERE \(suspect("bytes")))
                    """
                : "SELECT EXISTS(SELECT 1 FROM cache_entries WHERE \(suspect("file_size")))"
        ) { stmt in found = sqlite3_column_int64(stmt, 0) != 0 }
        return found || !answered
    }

    // MARK: - Companion accounting helpers (caller holds `lock`)

    /// `INSERT … SELECT` that turns a row's companion link into an unlinked
    /// entry, keeping the row's recency. Callers append the WHERE clause.
    ///
    /// Both target columns are NOT NULL and the v1 DDL lets `created_at` be
    /// NULL (`OR REPLACE` does not rescue a NOT NULL column that has no
    /// default): a row without a recency is handed over as of now, and one
    /// without a byte count as 0 bytes. Without that the statement fails —
    /// and takes a whole retirement down with it, on every pass.
    private static let moveLinkedCompanionsToLegacySQL = """
        INSERT OR REPLACE INTO legacy_companions (key, bytes, modified)
        SELECT companion_key, COALESCE(companion_bytes, 0),
               (COALESCE(created_at, julianday('now')) - 2440587.5) * 86400.0
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
        // See `quotaEntries()`: a key that is not a companion key names
        // nothing, is not offered to anyone, and is forgotten by rowid.
        let read = _readLegacyCompanionsLocked()
        _retireInvalidRecordsLocked(InvalidRecords(legacy: read.invalidRowids))
        return read.valid
    }

    private func _readLegacyCompanionsLocked()
        -> (valid: [DiskCacheLegacyCompanion], invalidRowids: [Int64])
    {
        var result: [DiskCacheLegacyCompanion] = []
        var invalidRowids: [Int64] = []
        _queryLocked("SELECT key, bytes, modified, rowid FROM legacy_companions") { stmt in
            let key: String
            switch Self.indexValue(stmt, column: 0, hexDigits: SSMCompanionDiskStore.keyLength) {
            case .valid(let value):
                key = value
            case .unreadable:
                return
            case .null, .invalid:
                invalidRowids.append(sqlite3_column_int64(stmt, 3))
                return
            }
            result.append(DiskCacheLegacyCompanion(
                key: key,
                bytes: _indexedBytesLocked(stmt, 1),
                modifiedAt: Date(timeIntervalSince1970: sqlite3_column_double(stmt, 2))))
        }
        return (result, invalidRowids)
    }

    /// Whole-root usage: two aggregates, no row materialization.
    private func _combinedUsageLocked() -> (bytes: Int64, entryCount: Int) {
        guard indexHasV2Columns else {
            let usage = _payloadUsageLocked()
            return (Int64(usage.bytes), usage.entryCount)
        }
        return _countingAgainIfClampingBecameNecessary {
            var bytes: Int64 = 0
            var count = 0
            _queryLocked(
                """
                SELECT \(_usageSQLLocked("file_size", "companion_bytes")), COUNT(*)
                FROM cache_entries
                """
            ) { stmt in
                bytes = IndexedBytes.sum(bytes, _indexedBytesLocked(stmt, 0))
                count += Int(sqlite3_column_int64(stmt, 1))
            }
            _queryLocked(
                "SELECT \(_usageSQLLocked("bytes")), COUNT(*) FROM legacy_companions"
            ) { stmt in
                bytes = IndexedBytes.sum(bytes, _indexedBytesLocked(stmt, 0))
                count += Int(sqlite3_column_int64(stmt, 1))
            }
            return (bytes, count)
        }
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
        // TRANSIENT: the statement is stepped after this String's buffer
        // may be gone (a nil destructor is SQLITE_STATIC).
        sqlite3_bind_text(stmt, 1, hash, -1, Self.sqliteTransient)
        guard sqlite3_step(stmt) == SQLITE_ROW else { return nil }
        return (
            tokenCount: Int(sqlite3_column_int64(stmt, 0)),
            fileSize: Int(sqlite3_column_int64(stmt, 1)))
    }

    /// Current indexed payload usage. Caller MUST hold `lock`.
    private func _payloadUsageLocked() -> (bytes: Int, entryCount: Int) {
        guard db != nil else { return (0, 0) }
        return _countingAgainIfClampingBecameNecessary {
            var usage = (bytes: 0, entryCount: 0)
            _queryLocked("SELECT \(_usageSQLLocked("file_size")), COUNT(*) FROM cache_entries") {
                stmt in
                usage = (
                    IndexedBytes.asInt(_indexedBytesLocked(stmt, 0)),
                    max(0, Int(sqlite3_column_int64(stmt, 1))))
            }
            return usage
        }
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
            quotaEvictedBytes = IndexedBytes.sum(quotaEvictedBytes, evictedBytes)
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
    /// A skipped store is still this conversation storing this boundary: the
    /// row's owner follows it exactly as a rewrite would have made it, and a
    /// root stays a root. Caller holds `lock`. No-op on a v1 index.
    private func _claimOwnershipLocked(hash: String, chainId: String?, isStableRoot: Bool) {
        guard indexHasV2Columns, !indexIsFromANewerBuild,
            chainId != nil || isStableRoot
        else { return }
        _ = _runLocked(
            """
            UPDATE cache_entries SET chain_id = COALESCE(?, chain_id), kind = MAX(kind, ?)
            WHERE hash = ?
            """,
            [chainId.map(SQLValue.text) ?? .null, .int(isStableRoot ? 1 : 0), .text(hash)])
    }

    /// Give an unowned history row (no `chain_id`, not a stable root) to the
    /// conversation that just restored from it, so rows written before owners
    /// existed — or by a request without one — come under the planner's
    /// protection the first time a chat resumes through them. Rows that
    /// already have an owner keep it: a read is not a claim. No-op on a v1
    /// index.
    func assignChain(tokens: [Int], mediaSalt: String?, chainId: String) {
        guard indexHasV2Columns, !indexIsFromANewerBuild,
            let (hash, _) = entryKey(tokens: tokens, mediaSalt: mediaSalt)
        else { return }
        lock.lock()
        defer { lock.unlock() }
        _ = _runLocked(
            """
            UPDATE cache_entries SET chain_id = ?
            WHERE hash = ? AND chain_id IS NULL AND kind = 0
            """,
            [.text(chainId), .text(hash)])
    }

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
        sqlite3_bind_text(stmt, 2, hash, -1, Self.sqliteTransient)
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
    ///
    /// A hand-over that could not be written leaves the row where it is, for
    /// the next caller to try again: the row is all that still counts the
    /// companion's bytes, and the index may over-count the disk but never
    /// under-counts it.
    private func _deleteEntryLocked(hash: String, keepCompanionCounted: Bool = true) {
        guard db != nil else { return }
        if indexHasV2Columns, keepCompanionCounted {
            let rc = _runLocked(
                Self.moveLinkedCompanionsToLegacySQL
                    + " WHERE hash = ? AND companion_key IS NOT NULL",
                [.text(hash)])
            guard rc == SQLITE_DONE else {
                failedIndexWrites += 1
                return
            }
        }
        _runLocked("DELETE FROM cache_entries WHERE hash = ?", [.text(hash)])
    }

    /// Evict entries until the total cache size is under `maxSizeBytes`:
    /// first every row that could never fit on its own, then oldest first.
    /// Caller MUST hold `lock`.
    ///
    /// Below the cap this is one SQL aggregate. Over it, every row is read:
    /// a row whose hash is not a payload hash names no file, so it is
    /// retired by rowid — not evicted — and nothing real pays for the bytes
    /// it claims, whether or not the retirement could be written. Under a
    /// newer build's index such a row is opaque instead: its bytes count,
    /// it is never a victim, and only rows this build understands go — and
    /// so is a row whose hash is fine and whose companion link this build
    /// cannot read, exactly as ``quotaEntries(retiringInvalidRecords:)``
    /// has it: evicting the row would hand that link on.
    ///
    /// A `file_size` is data (``IndexedBytes``). A row that claims more than
    /// the cap — `1e19` reads back as `Int64.max` — is an ordinary victim,
    /// and goes FIRST, as in the coordinator's pass: taking older rows that
    /// do fit to make room for one that never will would empty the cache and
    /// still end with that row's eviction. What is left is then counted, not
    /// subtracted from a total that may have saturated.
    private func _evictIfNeededLocked() {
        guard let db else { return }
        let capBytes = Int64(maxSizeBytes)

        let indexedBytes = _countingAgainIfClampingBecameNecessary { () -> Int64 in
            var bytes: Int64 = 0
            _queryLocked("SELECT \(_usageSQLLocked("file_size")) FROM cache_entries") { stmt in
                bytes = _indexedBytesLocked(stmt, 0)
            }
            return bytes
        }
        guard indexedBytes > capBytes else { return }

        let rowsAreOpaque = indexIsFromANewerBuild
        var oldestFirst: [(hash: String, url: URL, fileSize: Int64)] = []
        var invalid = InvalidRecords()
        var unofferedBytes: Int64 = 0
        var stmt: OpaquePointer?
        let sql = indexHasV2Columns
            ? """
                SELECT hash, file_size, rowid, companion_key, companion_bytes
                FROM cache_entries ORDER BY created_at ASC
                """
            : "SELECT hash, file_size, rowid FROM cache_entries ORDER BY created_at ASC"
        if sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK, let stmt {
            while sqlite3_step(stmt) == SQLITE_ROW {
                let fileSize = _indexedBytesLocked(stmt, 1)
                var victim: (hash: String, url: URL)?
                switch Self.indexValue(stmt, column: 0, hexDigits: Self.hashLength) {
                case .valid(let hash):
                    // Dropping a row hands its companion key on; one a newer
                    // build wrote in a shape this build cannot read — or
                    // bytes with no key at all — stays where it is, with
                    // its row.
                    var linkIsOpaque = false
                    if rowsAreOpaque, indexHasV2Columns {
                        switch Self.indexValue(
                            stmt, column: 3, hexDigits: SSMCompanionDiskStore.keyLength)
                        {
                        case .invalid, .unreadable: linkIsOpaque = true
                        case .null: linkIsOpaque = sqlite3_column_int64(stmt, 4) != 0
                        case .valid: break
                        }
                    }
                    if !linkIsOpaque, let url = safetensorsURL(for: hash) {
                        victim = (hash, url)
                    }
                case .null, .invalid:
                    invalid.entries.append(sqlite3_column_int64(stmt, 2))
                case .unreadable:
                    break
                }
                if let victim {
                    oldestFirst.append((victim.hash, victim.url, fileSize))
                } else {
                    unofferedBytes = IndexedBytes.sum(unofferedBytes, fileSize)
                }
            }
        }
        sqlite3_finalize(stmt)
        _retireInvalidRecordsLocked(invalid)

        // What the offered rows share. Under the current schema a record
        // that is not offered names nothing and costs them nothing; under a
        // newer build's it is opaque, and its bytes come off the cap.
        let opaqueBytes = rowsAreOpaque ? unofferedBytes : 0
        if rowsAreOpaque { _noteOpaqueBytesLocked(opaqueBytes, capBytes: capBytes) }
        let offeredCap = IndexedBytes.difference(capBytes, opaqueBytes)

        // Delete evicted entries and their files. A payload that could not
        // be deleted keeps its row (see `removeQuotaEntries`); it is tried
        // once per pass and its bytes are not made up for by evicting more.
        func evict(_ entry: (hash: String, url: URL, fileSize: Int64)) {
            validatedFiles.removeValue(forKey: entry.hash)
            guard Self.removeCacheFile(at: entry.url) else { return }
            _deleteEntryLocked(hash: entry.hash)
            evictions += 1
        }
        var remaining: Int64 = 0
        for entry in oldestFirst {
            if entry.fileSize > offeredCap {
                evict(entry)
            } else {
                remaining = IndexedBytes.sum(remaining, entry.fileSize)
            }
        }
        for entry in oldestFirst where entry.fileSize <= offeredCap {
            guard remaining > offeredCap else { break }
            remaining = IndexedBytes.difference(remaining, entry.fileSize)
            evict(entry)
        }
    }
}
