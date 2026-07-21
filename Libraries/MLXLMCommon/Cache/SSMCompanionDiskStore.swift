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

    // MARK: - Initialization

    public init(cacheDir: URL, modelKey: String? = nil, maxBytes: Int = 0) throws {
        self.cacheDir = cacheDir
        self.modelKey = modelKey
        self.maxBytes = maxBytes
        try FileManager.default.createDirectory(
            at: cacheDir, withIntermediateDirectories: true)
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
        guard !ssmStates.isEmpty, boundary > 0, boundary <= tokens.count else { return }
        let key = Self.keyFor(
            tokens: tokens, boundary: boundary,
            mediaSalt: mediaSalt, modelKey: modelKey)
        let safetensorsURL = self.safetensorsURL(for: key)
        let sidecarURL = self.sidecarURL(for: key)
        let kvHash = DiskCache.hashTokens(
            Array(tokens.prefix(boundary)),
            modelKey: modelKey,
            mediaSalt: mediaSalt)

        MLXDiskCacheIOLock.shared.lock()
        defer { MLXDiskCacheIOLock.shared.unlock() }
        lock.lock()
        defer { lock.unlock() }

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
            let now = Date()
            do {
                try FileManager.default.setAttributes(
                    [.modificationDate: now], ofItemAtPath: safetensorsURL.path)
                try FileManager.default.setAttributes(
                    [.modificationDate: now], ofItemAtPath: sidecarURL.path)
                if let touchedSafetensors = fileFingerprint(at: safetensorsURL),
                   let touchedSidecar = fileFingerprint(at: sidecarURL)
                {
                    validatedEntries[key] = ValidatedEntry(
                        safetensors: touchedSafetensors,
                        sidecar: touchedSidecar,
                        isComplete: isComplete,
                        numStates: ssmStates.count,
                        boundary: boundary,
                        kvHash: kvHash)
                } else {
                    validatedEntries.removeValue(forKey: key)
                }
                storeSkips += 1
                if ProcessInfo.processInfo.environment["VMLX_CACHE_FETCH_TRACE"] == "1" {
                    FileHandle.standardError.write(Data(
                        "[vmlx][cache/ssm-store] SKIP validated key=\(key) boundary=\(boundary) states=\(ssmStates.count)\n".utf8))
                }
                return
            } catch {
                validatedEntries.removeValue(forKey: key)
            }
        }

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
        try save(arrays: arrays, metadata: ["format": "mlx"], url: safetensorsURL)
        Stream.gpu.synchronize()

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
        } else {
            validatedEntries.removeValue(forKey: key)
        }

        evictIfNeededLocked()
    }

    /// Look up SSM layer states for a given token prefix + boundary.
    /// Returns nil on miss / corruption / decode failure.
    ///
    /// Iter 143: `mediaSalt` mirror of the store-side change. Pass the
    /// same salt the L1 store consumed (typically derived from
    /// `computeMediaSalt(images:videos:audios:)`).
    public func fetch(
        tokens: [Int],
        boundary: Int,
        mediaSalt: String? = nil
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
           let fetchedSafetensors = fileFingerprint(at: safetensorsURL),
           let fetchedSidecar = fileFingerprint(at: sidecarURL)
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

    /// Remove all entries for a given model key. Called on model
    /// unload so subsequent loads don't see stale state. No-op if the
    /// directory is empty.
    public func clear() {
        MLXDiskCacheIOLock.shared.lock()
        defer { MLXDiskCacheIOLock.shared.unlock() }
        lock.lock()
        defer { lock.unlock() }

        guard let entries = try? FileManager.default.contentsOfDirectory(
            at: cacheDir, includingPropertiesForKeys: nil) else { return }
        for url in entries {
            let name = url.lastPathComponent
            if name.hasPrefix("ssm-") {
                try? FileManager.default.removeItem(at: url)
            }
        }
        validatedEntries.removeAll(keepingCapacity: true)
    }

    /// Snapshot recurrent payloads for the coordinator's combined KV +
    /// companion quota. Legacy sidecars have no `kv_hash`; they remain valid
    /// for reads, but quota pressure retires them before indexed KV because
    /// they cannot prove which durable KV payload can still reach them.
    func quotaEntries() -> [SSMCompanionQuotaEntry] {
        lock.lock()
        defer { lock.unlock() }
        return diskEntriesLocked().map { hash, entry in
            SSMCompanionQuotaEntry(
                hash: hash,
                kvHash: entry.kvHash,
                bytes: Int64(entry.bytes),
                modifiedAt: entry.modified)
        }
    }

    /// Remove recurrent payloads selected by the combined quota pass.
    func removeQuotaEntries(hashes: Set<String>) {
        guard !hashes.isEmpty else { return }
        MLXDiskCacheIOLock.shared.lock()
        defer { MLXDiskCacheIOLock.shared.unlock() }
        lock.lock()
        defer { lock.unlock() }

        for hash in hashes {
            try? FileManager.default.removeItem(at: safetensorsURL(for: hash))
            try? FileManager.default.removeItem(at: sidecarURL(for: hash))
            validatedEntries.removeValue(forKey: hash)
        }
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

    private struct DiskEntry {
        var urls: [URL] = []
        var bytes: Int = 0
        var modified: Date = .distantPast
        var kvHash: String?
    }

    private func diskEntriesLocked() -> [String: DiskEntry] {
        guard let urls = try? FileManager.default.contentsOfDirectory(
            at: cacheDir,
            includingPropertiesForKeys: [.contentModificationDateKey, .fileSizeKey],
            options: [.skipsHiddenFiles])
        else { return [:] }

        var entries: [String: DiskEntry] = [:]
        for url in urls {
            guard let hash = entryHash(for: url) else { continue }
            let values = try? url.resourceValues(forKeys: [
                .contentModificationDateKey, .fileSizeKey,
            ])
            let bytes = values?.fileSize ?? 0
            let modified = values?.contentModificationDate ?? .distantPast

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

    private func evictIfNeededLocked() {
        guard maxBytes > 0 else { return }
        let entries = diskEntriesLocked()
        var totalBytes = entries.values.reduce(0) { $0 + $1.bytes }

        guard totalBytes > maxBytes else { return }

        for (hash, entry) in entries.sorted(by: { $0.value.modified < $1.value.modified }) {
            for url in entry.urls {
                try? FileManager.default.removeItem(at: url)
            }
            validatedEntries.removeValue(forKey: hash)
            totalBytes -= entry.bytes
            if totalBytes <= maxBytes { break }
        }
    }

    private func entryHash(for url: URL) -> String? {
        let name = url.lastPathComponent
        guard name.hasPrefix("ssm-"),
              (name.hasSuffix(".safetensors") || name.hasSuffix(".json")),
              let dot = name.lastIndex(of: ".")
        else { return nil }
        let start = name.index(name.startIndex, offsetBy: 4)
        guard start < dot else { return nil }
        return String(name[start..<dot])
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
