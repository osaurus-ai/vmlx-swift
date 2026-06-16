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
// Cache key derivation delegates to `SSMStateCache.makeKey`:
//   SHA-256(modelKey blob || mediaSalt blob || "|tokens:" || compact JSON tokens)
// This keeps the in-memory and disk companion tiers byte-identical while
// preserving model and media/request-salt isolation.
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

/// Disk-backed extension to the in-memory `SSMStateCache`. See header
/// comment for storage format + concurrency model.
public final class SSMCompanionDiskStore: @unchecked Sendable {

    // MARK: - Properties

    private let lock = OSAllocatedUnfairLock()
    private let cacheDir: URL
    private let modelKey: String?
    /// Maximum total disk bytes before oldest-entry eviction. 0 = unlimited.
    private let maxBytes: Int

    // MARK: - Initialization

    public init(cacheDir: URL, modelKey: String? = nil, maxBytes: Int = 0) throws {
        self.cacheDir = cacheDir
        self.modelKey = modelKey
        self.maxBytes = maxBytes
        try FileManager.default.createDirectory(
            at: cacheDir, withIntermediateDirectories: true)
    }

    // MARK: - Public API

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

        MLXDiskCacheIOLock.shared.lock()
        defer { MLXDiskCacheIOLock.shared.unlock() }
        lock.lock()
        defer { lock.unlock() }

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

        let safetensorsURL = self.safetensorsURL(for: key)
        let sidecarURL = self.sidecarURL(for: key)

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
        ]
        let sidecarData = try JSONSerialization.data(
            withJSONObject: sidecar, options: [.sortedKeys])
        try sidecarData.write(to: sidecarURL, options: [.atomic])

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
        else { return nil }

        // Decode sidecar first — cheap, validates the entry shape.
        guard let sidecarData = try? Data(contentsOf: sidecarURL),
              let sidecar = try? JSONSerialization.jsonObject(with: sidecarData)
                as? [String: Any],
              let isComplete = sidecar["is_complete"] as? Bool,
              let numStates = sidecar["num_states"] as? Int,
              numStates > 0
        else { return nil }

        // Decode safetensors. A failed deserialize is most often a
        // truncated file (process killed mid-write, rare on sync IO
        // but possible). Treat as miss.
        guard let arraysAndMeta = try? loadArraysAndMetadata(url: safetensorsURL)
        else { return nil }
        let arrays = arraysAndMeta.0

        // Reassemble in positional order. Bail if any `state_<idx>` is
        // missing — partial entries are unsafe to extend per the Python
        // `(states, is_complete)` contract.
        var states: [MLXArray] = []
        states.reserveCapacity(numStates)
        for i in 0 ..< numStates {
            guard let arr = arrays["state_\(i)"] else { return nil }
            states.append(arr)
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
    }

    // MARK: - Helpers

    private func safetensorsURL(for hash: String) -> URL {
        cacheDir.appendingPathComponent("ssm-\(hash).safetensors")
    }

    private func sidecarURL(for hash: String) -> URL {
        cacheDir.appendingPathComponent("ssm-\(hash).json")
    }

    private struct DiskEntry {
        var urls: [URL] = []
        var bytes: Int = 0
        var modified: Date = .distantPast
    }

    private func evictIfNeededLocked() {
        guard maxBytes > 0 else { return }
        guard let urls = try? FileManager.default.contentsOfDirectory(
            at: cacheDir,
            includingPropertiesForKeys: [.contentModificationDateKey, .fileSizeKey],
            options: [.skipsHiddenFiles])
        else { return }

        var entries: [String: DiskEntry] = [:]
        var totalBytes = 0
        for url in urls {
            guard let hash = entryHash(for: url) else { continue }
            let values = try? url.resourceValues(forKeys: [
                .contentModificationDateKey, .fileSizeKey,
            ])
            let bytes = values?.fileSize ?? 0
            let modified = values?.contentModificationDate ?? .distantPast
            totalBytes += bytes

            var entry = entries[hash] ?? DiskEntry()
            entry.urls.append(url)
            entry.bytes += bytes
            if entry.modified == .distantPast || modified < entry.modified {
                entry.modified = modified
            }
            entries[hash] = entry
        }

        guard totalBytes > maxBytes else { return }

        for entry in entries.values.sorted(by: { $0.modified < $1.modified }) {
            for url in entry.urls {
                try? FileManager.default.removeItem(at: url)
            }
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
