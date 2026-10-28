// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXLMCommon

/// Cache of one MiniMax-M3 sparse-attention (MSA) layer: keys and values `[B, kvHeads, S, headDim]` plus the
/// Lightning indexer's keys `[B, 1, S, indexHeadDim]` (normed and rotated).
///
/// The block selection is recomputed every step from the indexer keys and is never stored, so these three lanes are
/// the layer's whole state. They grow in lockstep: the attention layer appends K/V, then the indexer appends its keys
/// for the same rows. Blocks are anchored to absolute positions, so the cache only appends and trims; it never
/// rotates or evicts.
///
/// Composed around `KVCacheSimple` (public, not open: it cannot be subclassed from this module), so `trim`, `copy`
/// and restore visibly cover both buffers. Generic code that understands only plain K/V caches therefore cannot
/// mistake this cache for one and drop the indexer lane; disk persistence goes through `DiskCacheStateProviding`,
/// which restores all three lanes or nothing.
public final class MiniMaxM3SparseCache: DiskCacheStateProviding {
    /// The K/V lane. Readable in-module (tests poison its unused capacity to prove nothing reads past `offset`).
    private(set) var kv = KVCacheSimple()

    /// Capacity buffer for the indexer keys; rows past `indexCount` are unused capacity, never visible. Readable
    /// in-module (tests poison its unused capacity to prove nothing reads past `offset`).
    private(set) var indexStorage: MLXArray?
    private var indexCount = 0
    private static let step = 256

    /// DSpark (`MiniMaxM3DSpark`): the drafter's tap rows (target-layer outputs) of the last committed positions,
    /// recorded by every forward of this sequence while a drafter is attached, keyed by absolute position. Speed-only:
    /// a cache without it is exact, the drafter just has less context. Trims, copies and checkpoints carry it; a
    /// restored state does not.
    var dsparkTail: HybridPoolHiddenTail?
    /// DSpark: the next multi-row forward is a verify (every row on decode arithmetic), and the offset it starts at.
    var dsparkVerifyArmed = false
    var dsparkVerifyStart: Int?

    public init() {}

    public var offset: Int { kv.offset }
    /// Rows of the indexer-key lane (equal to `offset` whenever the lanes are in lockstep).
    var indexRows: Int { indexCount }
    public var maxSize: Int? { nil }

    public func innerState() -> [MLXArray] {
        kv.innerState() + [indexStorage].compactMap { $0 }
    }

    public func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        kv.update(keys: keys, values: values)
    }

    /// Appends this step's indexer keys `[B, 1, T, D]` for the rows whose K/V were just appended and returns the
    /// whole history `[B, 1, offset, D]`.
    public func updateIndexKeys(_ keys: MLXArray) -> MLXArray {
        let previous = indexCount
        let end = previous + keys.dim(2)
        precondition(
            end == offset,
            "MiniMax-M3 indexer keys out of step with K/V: \(end) indexer rows for \(offset) K/V rows")
        if let storage = indexStorage,
            storage.dim(0) != keys.dim(0) || storage.dim(1) != keys.dim(1)
                || storage.dim(3) != keys.dim(3) || storage.dtype != keys.dtype
        {
            preconditionFailure(
                "MiniMax-M3 indexer key geometry changed: cached \(storage.shape) \(storage.dtype), "
                    + "new \(keys.shape) \(keys.dtype)")
        }
        if indexStorage == nil || end > indexStorage!.dim(2) {
            let capacity = (end + Self.step - 1) / Self.step * Self.step
            var grown = MLXArray.zeros(
                [keys.dim(0), keys.dim(1), capacity, keys.dim(3)], dtype: keys.dtype)
            if previous > 0, let storage = indexStorage {
                grown[.ellipsis, ..<previous, 0...] = storage[.ellipsis, ..<previous, 0...]
            }
            indexStorage = grown
        }
        indexStorage![.ellipsis, previous ..< end, 0...] = keys
        indexCount = end
        return indexStorage![.ellipsis, ..<end, 0...]
    }

    /// `[keys, values, indexKeys]`, each sliced to `offset` rows; empty before the first update.
    public var state: [MLXArray] {
        get {
            let kvState = kv.state
            guard !kvState.isEmpty, let storage = indexStorage else { return kvState }
            return kvState + [storage[.ellipsis, ..<offset, 0...]]
        }
        set {
            dsparkTail = nil
            if newValue.isEmpty {
                kv = KVCacheSimple()
                indexStorage = nil
                indexCount = 0
                return
            }
            precondition(
                Self.isValid(newValue, offset: newValue[0].dim(2)),
                "MiniMax-M3 sparse cache state must be [keys, values, indexKeys] with equal sequence lengths")
            // Adopt views, not the caller's array objects: an in-place append after a trim must not write into
            // arrays the caller still holds (a snapshot, a disk record).
            kv.state = newValue[0 ..< 2].map { $0[0..., 0..., 0..., 0...] }
            indexStorage = newValue[2][0..., 0..., 0..., 0...]
            indexCount = newValue[2].dim(2)
        }
    }

    public var metaState: [String] {
        get { kv.metaState }
        set { kv.metaState = newValue }
    }

    public var isTrimmable: Bool { true }

    @discardableResult
    public func trim(_ n: Int) -> Int {
        let trimmed = kv.trim(n)
        indexCount = min(indexCount, offset)
        dsparkTail?.truncate(toEnd: offset)
        if dsparkTail?.isEmpty == true { dsparkTail = nil }
        return trimmed
    }

    public func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        kv.makeMask(n: n, windowSize: windowSize, returnArray: returnArray)
    }

    public func copy() -> any KVCache {
        let copy = MiniMaxM3SparseCache()
        let current = state
        if !current.isEmpty {
            // `* 1` materializes owned buffers: a slice would share the live cache's storage and block donation.
            copy.state = current.map { $0 * 1 }
            copy.dsparkTail = dsparkTail
        }
        return copy
    }

    // MARK: DiskCacheStateProviding

    public var diskCacheStateIdentifier: String { "minimax-m3-msa-kv-index-v1" }

    public func restoreDiskCacheState(_ state: [MLXArray], metadata: [String], offset: Int) -> Bool {
        guard offset > 0, metadata == kv.metaState, Self.isValid(state, offset: offset) else { return false }
        self.state = state
        return true
    }

    private static func isValid(_ state: [MLXArray], offset: Int) -> Bool {
        guard state.count == 3 else { return false }
        let (keys, values, index) = (state[0], state[1], state[2])
        return keys.ndim == 4 && values.ndim == 4 && index.ndim == 4
            && keys.dim(0) == values.dim(0) && keys.dim(0) == index.dim(0)
            && keys.dim(1) == values.dim(1) && keys.dim(1) > 0 && index.dim(1) == 1
            && keys.dim(2) == offset && values.dim(2) == offset && index.dim(2) == offset
            && keys.dim(3) > 0 && values.dim(3) > 0 && index.dim(3) > 0
            && keys.dtype.isFloatingPoint && keys.dtype == values.dtype && index.dtype.isFloatingPoint
    }
}

// MARK: - Speculative checkpoint (DSpark verify)

extension MiniMaxM3SparseCache: SpeculativeCheckpointCache {
    private struct Checkpoint {
        let kv: Any
        let indexStorage: MLXArray?
        let indexCount: Int
        let dsparkTail: HybridPoolHiddenTail?
    }

    public var speculativeCheckpointIsExact: Bool { kv.speculativeCheckpointIsExact }

    /// References only (O(1)): the K / V lane's own checkpoint, the index buffer (a separate wrapper, so a later slice
    /// update into the live buffer cannot change it) and its row count, the DSpark tap tail.
    public func speculativeCheckpoint() -> Any {
        Checkpoint(
            kv: kv.speculativeCheckpoint(), indexStorage: indexStorage.map { $0[.ellipsis] }, indexCount: indexCount,
            dsparkTail: dsparkTail)
    }

    public func restoreSpeculativeCheckpoint(_ checkpoint: Any) {
        guard let c = checkpoint as? Checkpoint else {
            preconditionFailure("MiniMaxM3SparseCache: foreign speculative checkpoint")
        }
        kv.restoreSpeculativeCheckpoint(c.kv)
        indexStorage = c.indexStorage
        indexCount = c.indexCount
        dsparkTail = c.dsparkTail
    }
}
