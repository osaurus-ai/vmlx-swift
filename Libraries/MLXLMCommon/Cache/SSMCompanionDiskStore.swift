// Copyright © 2024 Apple Inc.
//
// §441 — SSMCompanionDiskStore (native port of Python vmlx_engine #110).
//
// In-memory `SSMStateCache` (companion cache for hybrid Mamba+attention
// models — NemotronH / Cascade-2 / Nemotron-Omni / Qwen3.5-A3B / Jamba)
// is fast but volatile: a process restart re-prefills the prompt from
// scratch even if the user's system prompt + first turn haven't
// changed. For stable-system-prompt workloads (Terminal mode with a
// fixed scope-flagged agent prompt, server-side chat with one canonical
// system message) this re-prefill costs O(prompt_len) on every cold
// start.
//
// This store mirrors `DiskCache.swift`'s pattern: hash-keyed
// safetensors files under a flat directory, JSON sidecar for
// `is_complete` flag (parity with Python's `(states, is_complete)`
// tuple semantics from `vmlx_engine/utils/ssm_companion_cache.py`).
//
// Storage format per entry:
//   <cacheDir>/ssm-<sha>.safetensors    — N MLX arrays keyed `state_0`…`state_N-1`
//   <cacheDir>/ssm-<sha>.json           — metadata { is_complete, num_states, model_key }
//
// Cache key derivation delegates to the in-memory `SSMStateCache.makeKey`
// implementation so model key and media salt isolation cannot drift between
// memory and disk companion caches.
//
// Concurrency: store/fetch/clear are serialized with an
// `OSAllocatedUnfairLock`, and MLX safetensors IO also takes
// `MLXDiskCacheIOLock.shared` so companion-state reads/writes cannot overlap
// KV-cache safetensors reads/writes from another resident model. MLX tensor
// realization and safetensors IO should not overlap, and the metadata sidecar
// must stay paired with the tensor file.
//
// Wired by `CacheCoordinator` when `CacheCoordinatorConfig.enableDiskCache`
// is true. `SSMStateCache.store` write-throughs here and `fetchEntry`
// falls through on memory miss, using the same model key and media salt
// isolation as the KV tiers.

import CryptoKit
import Foundation
import MLX
import os

/// One recurrent companion payload used by the coordinator's shared quota.
/// New sidecars carry the hash of their matching KV payload so eviction can
/// remove an old hybrid entry as one unit instead of orphaning half of it.
struct SSMCompanionQuotaEntry: Sendable {
    let hash: String
    let kvHash: String?
    let bytes: Int64
    let modifiedAt: Date
}

/// What one `store` left on disk: the entry's key, the KV payload it belongs
/// to, and the bytes of its tensor file + sidecar.
struct SSMCompanionStoreRecord: Sendable, Equatable {
    let key: String
    let kvHash: String
    let bytes: Int64
    let modifiedAt: Date
}

/// Disk-backed extension to the in-memory `SSMStateCache`. See header
/// comment for storage format + concurrency model.
public final class SSMCompanionDiskStore: @unchecked Sendable {

    private struct FileFingerprint: Equatable {
        let size: Int
        let modificationDate: Date
    }

    private struct ValidatedEntry: Equatable {
        let safetensors: FileFingerprint
        let sidecar: FileFingerprint
        let isComplete: Bool
        let numStates: Int
        let boundary: Int
        let kvHash: String
    }

    // MARK: - Properties

    private let lock = OSAllocatedUnfairLock()
    private let cacheDir: URL
    private let modelKey: String?
    /// Maximum total disk bytes before oldest-entry eviction. 0 = unlimited.
    private let maxBytes: Int

    /// Companion pairs successfully written or deserialized by this process.
    /// The process-local validation requirement prevents an inherited corrupt
    /// pair from being trusted merely because both pathnames exist.
    private var validatedEntries: [String: ValidatedEntry] = [:]

    /// Number of full companion rewrites avoided after current-process
    /// validation. Exposed as a locked snapshot for tests and telemetry.
    private var storeSkips: Int = 0

    /// The KV index that counts this store's bytes against the shared quota,
    /// when the coordinator has one with the v2 columns, so that the quota
    /// does not have to walk this directory — whichever caller did the
    /// writing. Exactly these are reported to it, all from `store` and
    /// `clear`, after this store's own locks are released:
    ///
    /// - a completed write, and the touch-only skip of a validated entry
    ///   (`recordCompanionFailureCode`; if the index cannot record it, the
    ///   files are removed again unless an earlier record still covers them);
    /// - a write that threw, or that left no readable pair: whatever is on
    ///   disk for that key afterwards (`recordCompanionFailureCode` with the
    ///   real bytes, or `forgetCompanions` when nothing is);
    /// - this store's own eviction on a direct write (`forgetCompanions` for
    ///   what it removed, `correctCompanionBytes` for what it could not);
    /// - `clear()` (`forgetAllCompanions`).
    ///
    /// NOT reported: `removeQuotaEntries` — its caller, the coordinator's
    /// combined quota pass, removes the rows itself; `fetch` and
    /// `touchRecency`, which remove nothing (a pair that fails to decode
    /// stays on disk and stays counted); and anything that deletes files
    /// without going through this type, which is what
    /// `CacheCoordinator.reconcileDiskAccounting()` is for.
    private var ledger: DiskCache?

    /// The two steps of a write that publish a file under its final name.
    enum WriteStage: Sendable { case moveTensorIntoPlace, writeSidecar }

    /// Test seam, never set in production: called just before each publishing
    /// step of a write, and an error thrown from it stands in for the file
    /// system failing at that step. The sidecar failure can be provoked with
    /// a real directory in the way; "the old tensor was removed and the
    /// rename then failed" cannot be arranged from outside the process.
    var writeFaultForTesting: (@Sendable (WriteStage) throws -> Void)?

    // MARK: - Initialization

    public init(cacheDir: URL, modelKey: String? = nil, maxBytes: Int = 0) throws {
        self.cacheDir = cacheDir
        self.modelKey = modelKey
        self.maxBytes = maxBytes
        try FileManager.default.createDirectory(
            at: cacheDir, withIntermediateDirectories: true)
        Self.sweepUnpublishedFiles(in: cacheDir)
    }

    /// Tensor files are written under a `.partial-` name and renamed into
    /// place, so one that still carries that name at open is a dead write —
    /// if the name is exactly this store's (``isUnpublishedTensorName(_:)``)
    /// and a regular file holds it. Anything else that starts with `ssm-` is
    /// not this store's to remove.
    static func sweepUnpublishedFiles(in cacheDir: URL) {
        guard let names = try? FileManager.default.contentsOfDirectory(atPath: cacheDir.path)
        else { return }
        for name in names where isUnpublishedTensorName(name) {
            _ = DiskCache.removeRegularFile(at: cacheDir.appendingPathComponent(name))
        }
    }

    static let namePrefix = "ssm-"
    /// ``keyFor(tokens:boundary:mediaSalt:modelKey:)`` is a whole SHA-256
    /// digest in lowercase hex, and has been in every revision of this
    /// store in this repository; a shorter `ssm-<32 hex>` name is not one
    /// this code has written.
    static let keyLength = 64

    /// The key in `ssm-<key>.safetensors` / `ssm-<key>.json`, or nil when
    /// `name` is not exactly one of those two with a real key. This is the
    /// only test of "is this entry ours" that anything listing the directory
    /// may use: `ssm-notes.txt` is not, nor is an `.partial-` name.
    static func publishedEntryKey(fromName name: String) -> String? {
        guard name.hasPrefix(namePrefix) else { return nil }
        for suffix in [DiskCache.payloadSuffix, ".json"] where name.hasSuffix(suffix) {
            let key = name.dropFirst(namePrefix.count).dropLast(suffix.count)
            return DiskCache.isLowercaseHex(key, count: keyLength) ? String(key) : nil
        }
        return nil
    }

    /// Whether `name` is exactly what `DiskCache.temporaryURL(for:)` makes of
    /// one of this store's tensor files:
    /// `ssm-<key>.partial-<8 hex>.safetensors`.
    static func isUnpublishedTensorName(_ name: String) -> Bool {
        guard let stem = DiskCache.unpublishedStem(ofName: name), stem.hasPrefix(namePrefix)
        else { return false }
        return DiskCache.isLowercaseHex(stem.dropFirst(namePrefix.count), count: keyLength)
    }

    /// Where this store's files live.
    var directory: URL { cacheDir }

    func attachLedger(_ ledger: DiskCache?) {
        lock.lock()
        self.ledger = ledger
        lock.unlock()
    }

    // MARK: - Public API

    func snapshotStoreSkips() -> Int {
        lock.lock()
        defer { lock.unlock() }
        return storeSkips
    }

    /// Whether the current process has validated a complete companion pair for
    /// this exact content-addressed boundary. Used with the KV validation gate
    /// so a valid KV file cannot suppress healing a missing/corrupt recurrent
    /// sidecar after a hybrid cache miss.
    func hasValidatedCompleteEntry(
        tokens: [Int],
        boundary: Int,
        mediaSalt: String? = nil
    ) -> Bool {
        guard boundary > 0, boundary <= tokens.count else { return false }
        let key = Self.keyFor(
            tokens: tokens, boundary: boundary,
            mediaSalt: mediaSalt, modelKey: modelKey)
        let expectedKVHash = DiskCache.hashTokens(
            Array(tokens.prefix(boundary)),
            modelKey: modelKey,
            mediaSalt: mediaSalt)
        let safetensorsURL = self.safetensorsURL(for: key)
        let sidecarURL = self.sidecarURL(for: key)

        lock.lock()
        defer { lock.unlock() }
        guard let validated = validatedEntries[key],
              validated.isComplete,
              validated.numStates > 0,
              validated.boundary == boundary,
              validated.kvHash == expectedKVHash,
              let currentSafetensors = fileFingerprint(at: safetensorsURL),
              let currentSidecar = fileFingerprint(at: sidecarURL),
              currentSafetensors == validated.safetensors,
              currentSidecar == validated.sidecar
        else {
            return false
        }
        return true
    }

    /// Persist SSM layer states for a given token prefix. Mirrors
    /// `SSMStateCache.store(ssmStates:tokens:boundary:)` with the
    /// addition of an `isComplete` flag (parity with Python tuple).
    ///
    /// Iter 143: `mediaSalt` is now threaded through to the disk key
    /// so VL/Omni paths don't collide with text-only prefixes that
    /// happen to share a token prefix. Previously hardcoded to `nil`
    /// here, which silently aliased text-only and audio/image variants
    /// of the same prefix on disk → wrong SSM state restored on cold
    /// start for hybrid-VL or Nemotron-Omni audio sessions.
    public func store(
        ssmStates: [MLXArray],
        tokens: [Int],
        boundary: Int,
        mediaSalt: String? = nil,
        isComplete: Bool = true
    ) throws {
        try store(
            ssmStates: ssmStates,
            tokens: tokens,
            boundary: boundary,
            mediaSalt: mediaSalt,
            isComplete: isComplete,
            enforceQuota: true)
    }

    /// Coordinator-only transactional store. See ``DiskCache/store``: linked
    /// KV + recurrent state must be admitted or evicted as one group.
    ///
    /// Returns what is now on disk for this entry, or nil when nothing was
    /// stored — including when the ledger could not record the entry, in
    /// which case the files just written have been removed again.
    @discardableResult
    func store(
        ssmStates: [MLXArray],
        tokens: [Int],
        boundary: Int,
        mediaSalt: String? = nil,
        isComplete: Bool = true,
        enforceQuota: Bool
    ) throws -> SSMCompanionStoreRecord? {
        guard !ssmStates.isEmpty, boundary > 0, boundary <= tokens.count else { return nil }
        let key = Self.keyFor(
            tokens: tokens, boundary: boundary,
            mediaSalt: mediaSalt, modelKey: modelKey)
        let kvHash = DiskCache.hashTokens(
            Array(tokens.prefix(boundary)),
            modelKey: modelKey,
            mediaSalt: mediaSalt)

        let outcome: WriteOutcome
        do {
            outcome = try writeEntry(
                ssmStates: ssmStates, key: key, kvHash: kvHash, boundary: boundary,
                isComplete: isComplete, enforceQuota: enforceQuota)
        } catch {
            // The write can fail after it has already changed the directory:
            // the old tensor is removed before the new one is renamed in, and
            // the tensor is published before the sidecar is written. The
            // index must describe what is there now, not what was intended.
            reportDiskTruth(key: key, kvHash: kvHash)
            throw error
        }

        // Reported after this store's locks are released: the index takes its
        // own lock, and nothing here needs the two held together.
        guard let ledger = outcome.ledger else { return outcome.record }
        var record = outcome.record
        if let written = record {
            if let rc = ledger.recordCompanionFailureCode(
                kvHash: written.kvHash, companionKey: written.key,
                bytes: written.bytes, modified: written.modifiedAt),
                removeUnrecordedEntry(key: key, bytes: written.bytes, rc: rc, countedBy: ledger)
            {
                record = nil
            }
        } else {
            // No throw, but no readable pair either.
            reportDiskTruth(key: key, kvHash: kvHash)
        }

        var evicted = outcome.evictedKeys
        if enforceQuota, ledger.indexHasV2Columns {
            evicted.formUnion(evictOverCap(countedBy: ledger))
        }
        ledger.forgetCompanions(keys: evicted)
        return record
    }

    /// After a write that did not produce a record: drop this process's
    /// validation of the key and tell the ledger what the two final-named
    /// files actually hold now.
    private func reportDiskTruth(key: String, kvHash: String) {
        lock.lock()
        validatedEntries.removeValue(forKey: key)
        let ledger = self.ledger
        let state = Self.publishedEntryState(key: key, in: cacheDir)
        lock.unlock()

        guard let ledger else { return }
        let current: (bytes: Int64, modifiedAt: Date)
        switch state {
        case .present(let bytes, let modifiedAt):
            current = (bytes, modifiedAt)
        case .absent:
            ledger.forgetCompanions(keys: [key])
            return
        case .unreadable:
            // Could not look: whatever the ledger records for this key
            // stays. An over-count at worst, which the next import settles.
            return
        }
        if let rc = ledger.recordCompanionFailureCode(
            kvHash: kvHash, companionKey: key, bytes: current.bytes, modified: current.modifiedAt)
        {
            _ = removeUnrecordedEntry(key: key, bytes: current.bytes, rc: rc, countedBy: ledger)
        }
    }

    /// What is under one key's two final names right now.
    enum PublishedEntryState {
        /// The bytes of whichever of the tensor file and the sidecar are
        /// regular files, and the older of their modification dates.
        case present(bytes: Int64, modifiedAt: Date)
        /// Neither name holds anything: each is a definite "no such file",
        /// not a regular file, or empty.
        case absent
        /// One of the two could not be examined (any `lstat` failure other
        /// than ENOENT). That says nothing about whether it is there, and
        /// must not be read as `absent`: clearing a link or forgetting a
        /// record on it under-counts files that are still on disk.
        case unreadable(errno: Int32)
    }

    /// Stats only, no lock: also used by the index while it holds its own.
    static func publishedEntryState(key: String, in cacheDir: URL) -> PublishedEntryState {
        var bytes: Int64 = 0
        var modified: Date?
        for name in ["ssm-\(key).safetensors", "ssm-\(key).json"] {
            switch DiskCache.pathState(at: cacheDir.appendingPathComponent(name)) {
            case .regularFile(let size, let date):
                bytes += size
                modified = modified.map { min($0, date) } ?? date
            case .missing, .notRegularFile:
                continue
            case .unreadable(let code):
                return .unreadable(errno: code)
            }
        }
        return bytes > 0 ? .present(bytes: bytes, modifiedAt: modified ?? Date()) : .absent
    }

    /// ``publishedEntryState(key:in:)`` for callers that only correct bytes
    /// upwards from what they find: nil when nothing is there OR it could
    /// not be examined. Not for deciding that files are gone.
    static func publishedEntry(
        key: String, in cacheDir: URL
    ) -> (bytes: Int64, modifiedAt: Date)? {
        if case .present(let bytes, let modifiedAt) = publishedEntryState(key: key, in: cacheDir) {
            return (bytes, modifiedAt)
        }
        return nil
    }

    /// The index could not be made to count files that are already on disk.
    /// Files in neither table would never be evicted, so they are taken back;
    /// the boundary is a miss, which is the safe outcome. If another thread
    /// rewrote the same key in the meantime its files go too and its record
    /// stays: an over-count until that group is evicted or re-stored.
    ///
    /// Unless the index already counts this key for at least these bytes:
    /// the touch-only skip re-records an entry that is already counted, and
    /// a rewrite normally lands on the same size. The failed write then cost
    /// a recency refresh, not the accounting, and the pair is valid and
    /// stays. A record for FEWER bytes than are now on disk does not cover
    /// them, and the files still go.
    ///
    /// Returns whether the files were removed.
    private func removeUnrecordedEntry(
        key: String, bytes: Int64, rc: Int32, countedBy ledger: DiskCache
    ) -> Bool {
        if let counted = ledger.countedCompanionBytes(key: key), counted >= bytes {
            FileHandle.standardError.write(Data(
                ("[vmlx][cache/ssm-store] index record failed rc=\(rc) key=\(key.prefix(12)) "
                    + "— already counted (\(counted) bytes), companion kept\n").utf8))
            return false
        }
        MLXDiskCacheIOLock.shared.lock()
        lock.lock()
        _ = DiskCache.removeRegularFile(at: safetensorsURL(for: key))
        _ = DiskCache.removeRegularFile(at: sidecarURL(for: key))
        validatedEntries.removeValue(forKey: key)
        lock.unlock()
        MLXDiskCacheIOLock.shared.unlock()
        FileHandle.standardError.write(Data(
            "[vmlx][cache/ssm-store] index record failed rc=\(rc) key=\(key.prefix(12)) — companion removed\n"
                .utf8))
        return true
    }

    /// This store's own cap on a direct write, decided from the ledger: one
    /// aggregate below the cap, and no listing of the directory above it.
    /// Returns the keys whose files it removed; the caller reports them.
    private func evictOverCap(countedBy ledger: DiskCache) -> Set<String> {
        guard maxBytes > 0 else { return [] }
        var total = ledger.companionUsageBytes()
        guard total > Int64(maxBytes) else { return [] }

        var evicted = Set<String>()
        for companion in ledger.companionsOldestFirst() {
            evicted.insert(companion.key)
            total -= companion.bytes
            if total <= Int64(maxBytes) { break }
        }
        // Files first, rows after (by the caller): dying in between leaves
        // rows naming files that are gone — an over-count the next import
        // clears — never files the index has stopped counting. For the same
        // reason a companion that could not be deleted keeps its record, for
        // the bytes that are left; the next over-cap write tries it again.
        let stillOnDisk = removeQuotaEntries(hashes: evicted)
        for key in stillOnDisk {
            if let current = Self.publishedEntry(key: key, in: cacheDir) {
                ledger.correctCompanionBytes(key: key, bytes: current.bytes)
            }
        }
        return evicted.subtracting(stillOnDisk)
    }

    /// A write was refused because one of the entry's final names is held
    /// by something that is not a regular file. It is not an older copy of
    /// the entry and is never replaced.
    struct OccupiedNameError: Error, CustomStringConvertible {
        let name: String
        var description: String { "\(name) is not a regular file and is left alone" }
    }

    private static func requireReplaceable(_ url: URL) throws {
        if case .notRegularFile = DiskCache.pathState(at: url) {
            throw OccupiedNameError(name: url.lastPathComponent)
        }
    }

    private struct WriteOutcome {
        var record: SSMCompanionStoreRecord?
        var evictedKeys: Set<String> = []
        var ledger: DiskCache?
    }

    private func writeEntry(
        ssmStates: [MLXArray],
        key: String,
        kvHash: String,
        boundary: Int,
        isComplete: Bool,
        enforceQuota: Bool
    ) throws -> WriteOutcome {
        let safetensorsURL = self.safetensorsURL(for: key)
        let sidecarURL = self.sidecarURL(for: key)

        MLXDiskCacheIOLock.shared.lock()
        defer { MLXDiskCacheIOLock.shared.unlock() }
        lock.lock()
        defer { lock.unlock() }
        var outcome = WriteOutcome(ledger: ledger)

        // A normal warm hybrid request fetches a companion and publishes the
        // same prompt boundary again after generation. Avoid synchronizing the
        // GPU and rewriting the full recurrent-state payload when this process
        // has already validated the exact tensor/metadata pair. Completeness,
        // state count, boundary, and linked KV hash must all still match; a
        // changed file or metadata contract falls through to a healing write.
        if let validated = validatedEntries[key],
           validated.isComplete == isComplete,
           validated.numStates == ssmStates.count,
           validated.boundary == boundary,
           validated.kvHash == kvHash,
           let currentSafetensors = fileFingerprint(at: safetensorsURL),
           let currentSidecar = fileFingerprint(at: sidecarURL),
           currentSafetensors == validated.safetensors,
           currentSidecar == validated.sidecar
        {
            let touchedAt = Date()
            if let touched = touchEntryFilesLocked(
                safetensorsURL: safetensorsURL,
                sidecarURL: sidecarURL,
                at: touchedAt)
            {
                validatedEntries[key] = ValidatedEntry(
                    safetensors: touched.safetensors,
                    sidecar: touched.sidecar,
                    isComplete: isComplete,
                    numStates: ssmStates.count,
                    boundary: boundary,
                    kvHash: kvHash)
                storeSkips += 1
                if ProcessInfo.processInfo.environment["VMLX_CACHE_FETCH_TRACE"] == "1" {
                    FileHandle.standardError.write(Data(
                        "[vmlx][cache/ssm-store] SKIP validated key=\(key) boundary=\(boundary) states=\(ssmStates.count)\n".utf8))
                }
                outcome.record = SSMCompanionStoreRecord(
                    key: key, kvHash: kvHash,
                    bytes: Int64(touched.safetensors.size + touched.sidecar.size),
                    modifiedAt: touchedAt)
                return outcome
            }
            validatedEntries.removeValue(forKey: key)
        }

        // Refuse before the expensive part; checked again at the rename.
        try Self.requireReplaceable(safetensorsURL)

        // Pre-realize on calling thread — same rationale as
        // DiskCache.swift:148-157. GPU work must complete before the
        // safetensors writer can read the storage. MLX's tensor
        // realization (NOT script eval — this is `mlx.core.eval`).
        Stream.gpu.synchronize()
        MLX.eval(ssmStates)
        Stream.gpu.synchronize()

        // Materialize key→array dict expected by `save(arrays:metadata:url:)`.
        // Ordering preserved by `state_<idx>` keys; `extractSSMStates`
        // returns layers in cache order, so the round-trip is positional.
        var arrays: [String: MLXArray] = [:]
        for (i, arr) in ssmStates.enumerated() {
            arrays["state_\(i)"] = arr
        }

        // Sync write — same rationale as DiskCache.swift:122-130.
        // Async dispatch races with SIGTERM on short-lived sessions,
        // leaving zero-byte files. Costs ~ms on already-realized arrays.
        //
        // Atomic publication, as in `DiskCache.store`: a process that dies
        // mid-write must not leave a short tensor file under the final name,
        // where the next fetch would map it.
        let partialURL = DiskCache.temporaryURL(for: safetensorsURL)
        _ = DiskCache.removeRegularFile(at: partialURL)
        do {
            try save(arrays: arrays, metadata: ["format": "mlx"], url: partialURL)
            Stream.gpu.synchronize()
            // Make room for the rename only by unlinking an older regular
            // file of the same key; `removeItem` would have descended into a
            // directory that carries the name.
            try Self.requireReplaceable(safetensorsURL)
            _ = DiskCache.removeRegularFile(at: safetensorsURL)
            try writeFaultForTesting?(.moveTensorIntoPlace)
            try FileManager.default.moveItem(at: partialURL, to: safetensorsURL)
        } catch {
            _ = DiskCache.removeRegularFile(at: partialURL)
            throw error
        }

        // JSON sidecar for is_complete flag + num_states.
        let sidecar: [String: Any] = [
            "is_complete": isComplete,
            "num_states": ssmStates.count,
            "model_key": modelKey ?? "",
            "boundary": boundary,
            "kv_hash": kvHash,
        ]
        let sidecarData = try JSONSerialization.data(
            withJSONObject: sidecar, options: [.sortedKeys])
        try writeFaultForTesting?(.writeSidecar)
        // The atomic write renames over the final name; it must not land on
        // (or through) a directory or a link that carries it.
        try Self.requireReplaceable(sidecarURL)
        try sidecarData.write(to: sidecarURL, options: [.atomic])

        if let writtenSafetensors = fileFingerprint(at: safetensorsURL),
           let writtenSidecar = fileFingerprint(at: sidecarURL)
        {
            validatedEntries[key] = ValidatedEntry(
                safetensors: writtenSafetensors,
                sidecar: writtenSidecar,
                isComplete: isComplete,
                numStates: ssmStates.count,
                boundary: boundary,
                kvHash: kvHash)
            outcome.record = SSMCompanionStoreRecord(
                key: key, kvHash: kvHash,
                bytes: Int64(writtenSafetensors.size + writtenSidecar.size),
                modifiedAt: min(
                    writtenSafetensors.modificationDate, writtenSidecar.modificationDate))
        } else {
            validatedEntries.removeValue(forKey: key)
        }

        // With a v2 ledger the cap is applied by `store`, from the index,
        // once this entry has been recorded in it.
        if enforceQuota, !(ledger?.indexHasV2Columns ?? false) {
            outcome.evictedKeys = evictIfNeededLocked()
        }
        return outcome
    }

    /// Look up SSM layer states for a given token prefix + boundary.
    /// Returns nil on miss / corruption / decode failure. Direct reads refresh
    /// recency by default; CacheCoordinator disables that while it validates a
    /// candidate and touches the companion only after accepting the restore.
    /// Reusable-boundary callers can require a complete snapshot; incomplete
    /// entries are then rejected from sidecar metadata before tensor decode or
    /// recency mutation.
    ///
    /// Iter 143: `mediaSalt` mirror of the store-side change. Pass the
    /// same salt the L1 store consumed (typically derived from
    /// `computeMediaSalt(images:videos:audios:)`).
    public func fetch(
        tokens: [Int],
        boundary: Int,
        mediaSalt: String? = nil,
        touchRecency: Bool = true,
        requireComplete: Bool = false
    ) -> SSMStateCache.FetchResult? {
        guard boundary > 0, boundary <= tokens.count else { return nil }
        let key = Self.keyFor(
            tokens: tokens, boundary: boundary,
            mediaSalt: mediaSalt, modelKey: modelKey)
        let safetensorsURL = self.safetensorsURL(for: key)
        let sidecarURL = self.sidecarURL(for: key)

        MLXDiskCacheIOLock.shared.lock()
        defer { MLXDiskCacheIOLock.shared.unlock() }
        lock.lock()
        defer { lock.unlock() }

        guard FileManager.default.fileExists(atPath: safetensorsURL.path),
              FileManager.default.fileExists(atPath: sidecarURL.path)
        else {
            validatedEntries.removeValue(forKey: key)
            return nil
        }

        // Decode sidecar first — cheap, validates the entry shape.
        guard let sidecarData = try? Data(contentsOf: sidecarURL),
              let sidecar = try? JSONSerialization.jsonObject(with: sidecarData)
                as? [String: Any],
              let isComplete = sidecar["is_complete"] as? Bool,
              let numStates = sidecar["num_states"] as? Int,
              numStates > 0
        else {
            validatedEntries.removeValue(forKey: key)
            return nil
        }

        guard !requireComplete || isComplete else {
            return nil
        }

        // Decode safetensors. A failed deserialize is most often a
        // truncated file (process killed mid-write, rare on sync IO
        // but possible). Treat as miss.
        guard let arraysAndMeta = try? loadArraysAndMetadata(url: safetensorsURL)
        else {
            validatedEntries.removeValue(forKey: key)
            return nil
        }
        let arrays = arraysAndMeta.0

        // Reassemble in positional order. Bail if any `state_<idx>` is
        // missing — partial entries are unsafe to extend per the Python
        // `(states, is_complete)` contract.
        var states: [MLXArray] = []
        states.reserveCapacity(numStates)
        for i in 0 ..< numStates {
            guard let arr = arrays["state_\(i)"] else {
                validatedEntries.removeValue(forKey: key)
                return nil
            }
            states.append(arr)
        }

        // Accepted/direct reads, not only repeated stores, make an entry hot.
        // Refresh both files to the same timestamp so quota observes the pair
        // as one recency unit. Candidate validation disables this and lets the
        // coordinator touch only an accepted KV + companion group. This
        // changes filesystem metadata only; tensor and JSON payload bytes
        // remain untouched.
        let touchedFingerprints = touchRecency
            ? touchEntryFilesLocked(
                safetensorsURL: safetensorsURL,
                sidecarURL: sidecarURL,
                at: Date())
            : nil

        // Legacy sidecars remain readable, but only a complete current-format
        // metadata match is eligible to suppress the next write-through.
        let expectedKVHash = DiskCache.hashTokens(
            Array(tokens.prefix(boundary)),
            modelKey: modelKey,
            mediaSalt: mediaSalt)
        if let storedBoundary = sidecar["boundary"] as? Int,
           let storedKVHash = sidecar["kv_hash"] as? String,
           let storedModelKey = sidecar["model_key"] as? String,
           storedBoundary == boundary,
           storedKVHash == expectedKVHash,
           storedModelKey == (modelKey ?? ""),
           let fetchedSafetensors = touchedFingerprints?.safetensors
                ?? fileFingerprint(at: safetensorsURL),
           let fetchedSidecar = touchedFingerprints?.sidecar
                ?? fileFingerprint(at: sidecarURL)
        {
            validatedEntries[key] = ValidatedEntry(
                safetensors: fetchedSafetensors,
                sidecar: fetchedSidecar,
                isComplete: isComplete,
                numStates: numStates,
                boundary: boundary,
                kvHash: storedKVHash)
        } else {
            validatedEntries.removeValue(forKey: key)
        }

        return SSMStateCache.FetchResult(states: states, isComplete: isComplete)
    }

    /// Refresh one linked recurrent companion's eviction recency without
    /// decoding or rewriting either payload. CacheCoordinator calls this with
    /// the same timestamp as the matching KV row, including when recurrent
    /// state was satisfied from the in-memory L1 or folded disk payload.
    @discardableResult
    func touchRecency(
        tokens: [Int],
        boundary: Int,
        mediaSalt: String? = nil,
        at date: Date
    ) -> Bool {
        guard boundary > 0, boundary <= tokens.count else { return false }
        let key = Self.keyFor(
            tokens: tokens,
            boundary: boundary,
            mediaSalt: mediaSalt,
            modelKey: modelKey)
        let safetensorsURL = self.safetensorsURL(for: key)
        let sidecarURL = self.sidecarURL(for: key)

        MLXDiskCacheIOLock.shared.lock()
        defer { MLXDiskCacheIOLock.shared.unlock() }
        lock.lock()
        defer { lock.unlock() }

        let expectedKVHash = DiskCache.hashTokens(
            Array(tokens.prefix(boundary)),
            modelKey: modelKey,
            mediaSalt: mediaSalt)
        guard let sidecarData = try? Data(contentsOf: sidecarURL),
              let sidecar = try? JSONSerialization.jsonObject(with: sidecarData)
                as? [String: Any],
              sidecar["is_complete"] as? Bool == true,
              (sidecar["num_states"] as? Int ?? 0) > 0,
              sidecar["boundary"] as? Int == boundary,
              sidecar["kv_hash"] as? String == expectedKVHash,
              sidecar["model_key"] as? String == (modelKey ?? "")
        else {
            validatedEntries.removeValue(forKey: key)
            return false
        }

        guard let touched = touchEntryFilesLocked(
            safetensorsURL: safetensorsURL,
            sidecarURL: sidecarURL,
            at: date)
        else {
            validatedEntries.removeValue(forKey: key)
            return false
        }

        if let validated = validatedEntries[key] {
            validatedEntries[key] = ValidatedEntry(
                safetensors: touched.safetensors,
                sidecar: touched.sidecar,
                isComplete: validated.isComplete,
                numStates: validated.numStates,
                boundary: validated.boundary,
                kvHash: validated.kvHash)
        }
        return true
    }

    /// Remove all entries for a given model key. Called on model
    /// unload so subsequent loads don't see stale state. No-op if the
    /// directory is empty.
    public func clear() {
        let ledger = removeEveryEntry()
        ledger?.forgetAllCompanions()
    }

    /// Forget which pairs this process has validated; see
    /// ``DiskCache/forgetValidatedFiles()``.
    func forgetValidatedEntries() {
        lock.lock()
        validatedEntries.removeAll(keepingCapacity: true)
        lock.unlock()
    }

    private func removeEveryEntry() -> DiskCache? {
        MLXDiskCacheIOLock.shared.lock()
        defer { MLXDiskCacheIOLock.shared.unlock() }
        lock.lock()
        defer { lock.unlock() }

        // Only regular files that carry exactly one of this store's names:
        // a published pair's, or a dead write's. `ssm-notes.txt`, a
        // directory, a link — whatever else is here is not this store's.
        guard let names = try? FileManager.default.contentsOfDirectory(atPath: cacheDir.path)
        else { return nil }
        for name in names
        where Self.publishedEntryKey(fromName: name) != nil || Self.isUnpublishedTensorName(name) {
            _ = DiskCache.removeRegularFile(at: cacheDir.appendingPathComponent(name))
        }
        validatedEntries.removeAll(keepingCapacity: true)
        return ledger
    }

    /// Snapshot recurrent payloads for the coordinator's combined KV +
    /// companion quota. Legacy sidecars have no `kv_hash`; they remain valid
    /// for reads, but quota pressure retires them before indexed KV because
    /// they cannot prove which durable KV payload can still reach them.
    ///
    /// A directory that cannot be listed reads as empty here. That is right
    /// for a quota walk (nothing it could delete) and wrong for an import,
    /// which uses ``listedQuotaEntries()``.
    func quotaEntries() -> [SSMCompanionQuotaEntry] {
        listedQuotaEntries() ?? []
    }

    /// ``quotaEntries()``, or nil when the directory exists but could not be
    /// listed — which is not "no companions": whoever reconciles an index
    /// against this list must not take nil for an empty directory. An absent
    /// directory is empty, not an error.
    func listedQuotaEntries() -> [SSMCompanionQuotaEntry]? {
        lock.lock()
        defer { lock.unlock() }
        return listedDiskEntriesLocked()?.map { hash, entry in
            SSMCompanionQuotaEntry(
                hash: hash,
                kvHash: entry.kvHash,
                bytes: Int64(entry.bytes),
                modifiedAt: entry.modified)
        }
    }

    /// Remove recurrent payloads selected by a quota pass. Files only: the
    /// caller owns the index rows and removes them afterwards — except for
    /// the keys returned here, which still have a file on disk because it
    /// could not be deleted. Those must stay counted (for the bytes that are
    /// left), or nothing would ever evict them.
    @discardableResult
    func removeQuotaEntries(hashes: Set<String>) -> Set<String> {
        guard !hashes.isEmpty else { return [] }
        MLXDiskCacheIOLock.shared.lock()
        defer { MLXDiskCacheIOLock.shared.unlock() }
        lock.lock()
        defer { lock.unlock() }

        var stillOnDisk = Set<String>()
        for hash in hashes {
            let tensorGone = DiskCache.removeCacheFile(at: safetensorsURL(for: hash))
            let sidecarGone = DiskCache.removeCacheFile(at: sidecarURL(for: hash))
            validatedEntries.removeValue(forKey: hash)
            if !tensorGone || !sidecarGone { stillOnDisk.insert(hash) }
        }
        return stillOnDisk
    }

    // MARK: - Helpers

    private func safetensorsURL(for hash: String) -> URL {
        cacheDir.appendingPathComponent("ssm-\(hash).safetensors")
    }

    private func sidecarURL(for hash: String) -> URL {
        cacheDir.appendingPathComponent("ssm-\(hash).json")
    }

    private func fileFingerprint(at url: URL) -> FileFingerprint? {
        guard let values = try? url.resourceValues(forKeys: [
            .contentModificationDateKey, .fileSizeKey,
        ]),
            let size = values.fileSize,
            size > 0,
            let modificationDate = values.contentModificationDate
        else { return nil }
        return FileFingerprint(size: size, modificationDate: modificationDate)
    }

    /// Touch a companion pair without changing either payload. Caller MUST
    /// hold `lock`; cross-model safetensors serialization is owned by the
    /// public caller or the already-locked store/fetch path.
    private func touchEntryFilesLocked(
        safetensorsURL: URL,
        sidecarURL: URL,
        at date: Date
    ) -> (safetensors: FileFingerprint, sidecar: FileFingerprint)? {
        guard FileManager.default.fileExists(atPath: safetensorsURL.path),
              FileManager.default.fileExists(atPath: sidecarURL.path)
        else { return nil }
        do {
            try FileManager.default.setAttributes(
                [.modificationDate: date],
                ofItemAtPath: safetensorsURL.path)
            try FileManager.default.setAttributes(
                [.modificationDate: date],
                ofItemAtPath: sidecarURL.path)
            guard let safetensors = fileFingerprint(at: safetensorsURL),
                  let sidecar = fileFingerprint(at: sidecarURL)
            else { return nil }
            return (safetensors: safetensors, sidecar: sidecar)
        } catch {
            return nil
        }
    }

    private struct DiskEntry {
        var urls: [URL] = []
        var bytes: Int = 0
        var modified: Date = .distantPast
        var kvHash: String?
    }

    private func diskEntriesLocked() -> [String: DiskEntry] {
        listedDiskEntriesLocked() ?? [:]
    }

    /// nil when the directory is there but cannot be listed, or one of its
    /// entries cannot be examined.
    private func listedDiskEntriesLocked() -> [String: DiskEntry]? {
        let urls: [URL]
        do {
            urls = try FileManager.default.contentsOfDirectory(
                at: cacheDir,
                includingPropertiesForKeys: [.contentModificationDateKey, .fileSizeKey],
                options: [.skipsHiddenFiles])
        } catch let error as CocoaError where error.code == .fileReadNoSuchFile {
            return [:]
        } catch {
            return nil
        }

        var entries: [String: DiskEntry] = [:]
        for url in urls {
            guard let hash = entryHash(for: url) else { continue }
            // An entry that vanished since the listing is not an entry. One
            // that cannot be examined is there with unknown bytes, and a
            // walk that reported it as zero bytes would under-count it: the
            // walk as a whole has then failed.
            let bytes: Int
            let modified: Date
            switch DiskCache.pathState(at: url) {
            case .regularFile(let size, let date):
                bytes = Int(size)
                modified = date
            case .missing:
                continue
            case .notRegularFile:
                // A directory or a link under one of our names is not an
                // entry: it is not counted, and never offered for deletion.
                continue
            case .unreadable:
                return nil
            }

            var entry = entries[hash] ?? DiskEntry()
            entry.urls.append(url)
            entry.bytes += bytes
            if entry.modified == .distantPast || modified < entry.modified {
                entry.modified = modified
            }
            if url.pathExtension == "json",
               let data = try? Data(contentsOf: url),
               let sidecar = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let kvHash = sidecar["kv_hash"] as? String,
               !kvHash.isEmpty
            {
                entry.kvHash = kvHash
            }
            entries[hash] = entry
        }
        return entries
    }

    /// Standalone quota by directory walk, for a store with no ledger or a
    /// ledger without the v2 columns. With a v2 ledger `evictOverCap` applies
    /// the same cap from the index instead. Returns the keys it removed.
    @discardableResult
    private func evictIfNeededLocked() -> Set<String> {
        guard maxBytes > 0 else { return [] }
        let entries = diskEntriesLocked()
        var totalBytes = entries.values.reduce(0) { $0 + $1.bytes }

        guard totalBytes > maxBytes else { return [] }

        var evicted = Set<String>()
        for (hash, entry) in entries.sorted(by: { $0.value.modified < $1.value.modified }) {
            // `allSatisfy` would stop at the first failure; try every file.
            let gone = entry.urls.map { DiskCache.removeCacheFile(at: $0) }.allSatisfy { $0 }
            validatedEntries.removeValue(forKey: hash)
            if gone { evicted.insert(hash) }
            totalBytes -= entry.bytes
            if totalBytes <= maxBytes { break }
        }
        return evicted
    }

    /// An unpublished tensor file is not an entry (it is not counted, and
    /// its name must not be mistaken for a key), and neither is anything
    /// whose name is not exactly a published one: what this walk lists is
    /// what the quota may delete.
    private func entryHash(for url: URL) -> String? {
        Self.publishedEntryKey(fromName: url.lastPathComponent)
    }

    /// SHA-256 hash. P0-2 (2026-04-30): converged with `SSMStateCache.makeKey`
    /// AND with Python's `ssm_companion_cache._key`. Previous formula used
    /// `:` separator + Int32 LE bytes, which collided with NEITHER. Result
    /// was a write-only L2: every disk fetch missed L1's hash, so backfill
    /// silently failed (`AUDIT-SSM-WARMPASS-PARITY.md` §1). New formula
    /// delegates to `SSMStateCache.makeKey` so the two sites cannot drift.
    ///
    /// Iter 143: `mediaSalt` is now a real parameter (was hardcoded to
    /// nil — flagged "P1 follow-up" in the prior comment). Threading it
    /// through closes the L2 disk collision class for VL/Omni hybrid
    /// sessions: text-only and audio/image variants of the same token
    /// prefix used to share a key on disk → wrong SSM state restored.
    /// The 3-arg form (no mediaSalt) is preserved as a thin wrapper so
    /// existing tests + text-only callers don't need updating.
    public static func keyFor(
        tokens: [Int], boundary: Int,
        mediaSalt: String? = nil, modelKey: String?
    ) -> String {
        SSMStateCache.makeKey(
            tokens: tokens, boundary: boundary,
            mediaSalt: mediaSalt, modelKey: modelKey
        )
    }
}
