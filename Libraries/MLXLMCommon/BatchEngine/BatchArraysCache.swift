// Copyright 2025 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import MLX

// MARK: - BatchArraysCache

/// A MambaCache wrapper that merges N per-sequence SSM states along the batch
/// dimension for batched decode through hybrid SSM models.
///
/// ## How It Works
///
/// Hybrid models (Qwen3.5, Jamba, FalconH1, etc.) use `MambaCache` for SSM layers.
/// Each MambaCache stores 2 state arrays: conv state `[1, H, D, K]` and hidden
/// state `[1, H, Dv, Dk]`. For batched decode with B sequences, these states
/// need to be concatenated along dim 0 to `[B, ...]`, run through the model,
/// then split back.
///
/// `BatchArraysCache` inherits from `MambaCache` so it passes the `as? MambaCache`
/// type checks used by models like `Qwen35GatedDeltaNet`. It:
/// 1. On `init`: merges N slot MambaCache states into `[B, ...]` batched states
/// 2. The model reads/writes via `cache[0]`, `cache[1]` — works normally on the merged state
/// 3. After the forward pass: `splitBack()` extracts each sequence's state and
///    writes it back to the original per-sequence MambaCaches
///
/// ## Usage (inside BatchEngine)
///
/// ```swift
/// // Before batched forward pass:
/// let batchSSMCache = BatchArraysCache(slotCaches: [slot0.cache[layer], slot1.cache[layer]])
///
/// // Model uses batchSSMCache as a MambaCache — type check passes
/// gatedDeltaNet(x, cache: batchSSMCache)  // reads/writes [B, ...] states
///
/// // After forward pass:
/// batchSSMCache.splitBack()  // writes [1, ...] states back to each slot
/// ```
public final class BatchArraysCache: MambaCache {

    /// The original per-sequence caches this batch was merged from.
    private let slotCaches: [ArraysCache]

    /// Per-sequence logical positions before/after this batched step.
    private var offsets: [Int]

    /// Number of sequences in this batch.
    public let batchSize: Int

    /// Number of state slots in the wrapped per-sequence cache.
    private let numSlots: Int

    /// Per-sequence position offsets as `[B]`.
    ///
    /// Ling/Bailing linear attention applies RoPE before recurrent state update.
    /// Mixed-length B>1 decode must use these per-slot positions; the scalar
    /// `offset` remains the maximum only for legacy sizing/fallback paths.
    public private(set) var offsetArray: MLXArray

    /// Create a batched SSM cache by merging N per-sequence caches.
    ///
    /// - Parameter slotCaches: One cache per active sequence, all for the same
    ///   model layer. Must not be empty. All must have the same number of state slots.
    public init(slotCaches: [ArraysCache]) {
        precondition(!slotCaches.isEmpty, "BatchArraysCache requires at least one slot cache")
        let numSlots = slotCaches[0].slotCount
        precondition(
            slotCaches.allSatisfy { $0.slotCount == numSlots },
            "BatchArraysCache requires all slot caches to have the same slot count")
        precondition(
            numSlots <= 2,
            "BatchArraysCache currently supports ArraysCache/MambaCache layouts with at most 2 slots")
        self.slotCaches = slotCaches
        self.batchSize = slotCaches.count
        self.numSlots = numSlots
        self.offsets = slotCaches.map(\.offset)
        self.offsetArray = MLXArray(offsets.map { Int32($0) })

        super.init()

        // Merge states: concatenate each slot along batch dim 0
        mergeStates()

        // Use max offset across sequences for scalar compatibility only.
        self.offset = offsets.max() ?? 0
    }

    /// Merge per-sequence states into batched states.
    private func mergeStates() {
        for i in 0 ..< numSlots {
            let states = slotCaches.compactMap { $0[i] }
            if !states.isEmpty && states.count == batchSize {
                // All sequences have state for this slot — concatenate along batch dim
                self[i] = concatenated(states, axis: 0)
            }
            // If some sequences have nil state (first step), leave as nil —
            // the model will initialize it from the input shape
        }
    }

    /// Split batched states back to per-sequence caches.
    ///
    /// Call this AFTER the model forward pass to write updated states back
    /// to each sequence's original MambaCache.
    public func splitBack() {
        // The model mutates `offset` on this BatchArraysCache wrapper after
        // recurrent update. Push the same decode-step advance back to every
        // wrapped per-sequence cache; otherwise B>1 hybrid SSM decode updates
        // the state arrays while leaving each slot's logical position stale.
        let previousMaxOffset = offsets.max() ?? 0
        let advancedBy = max(0, offset - previousMaxOffset)
        if advancedBy > 0 {
            offsets = offsets.map { $0 + advancedBy }
            offsetArray = MLXArray(offsets.map { Int32($0) })
        }

        for i in 0 ..< numSlots {
            guard let merged = self[i] else { continue }
            for (j, slotCache) in slotCaches.enumerated() {
                slotCache[i] = merged[j ..< j + 1]
                slotCache.offset = offsets[j]
            }
        }
        offset = offsets.max() ?? 0
    }

    /// Advance every wrapped sequence by the same number of processed tokens.
    ///
    /// Recurrent layers do not call `update(keys:values:)`, so the batch wrapper
    /// needs an explicit offset advance after the model writes the merged state.
    public func advance(by tokenCount: Int) {
        guard tokenCount != 0 else { return }
        offsets = offsets.map { $0 + tokenCount }
        offsetArray = MLXArray(offsets.map { Int32($0) })
        offset = offsets.max() ?? 0
    }
}
