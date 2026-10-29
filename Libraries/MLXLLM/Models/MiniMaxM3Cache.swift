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
    /// Bumped whenever the lanes are replaced wholesale (`state` set, a disk restore): a row checkpoint taken before
    /// it is meaningless after it.
    private(set) var stateGeneration = 0

    /// DSpark (`MiniMaxM3DSpark`): the drafter's tap rows (target-layer outputs) of the last committed positions,
    /// recorded by every forward of this sequence while a drafter is attached, keyed by absolute position. Speed-only:
    /// a cache without it is exact, the drafter just has less context. Trims, copies and checkpoints carry it; a
    /// restored state does not.
    var dsparkTail: HybridPoolHiddenTail?
    /// DSpark: the next multi-row forward is a verify (every row on decode arithmetic), and the offset it starts at.
    var dsparkVerifyArmed = false
    var dsparkVerifyStart: Int?

    /// Aligned prefill (`AlignedPrefillPrefix.swift`): the current request's plan, set by the engine before its prompt
    /// prefill and read by `MiniMaxM3Model.prepare`. Request-scoped: never persisted or copied.
    public var alignedPrefillPlan: AlignedPrefillPlan?
    /// The geometry the rows of this cache were produced in (prefill step + producing store boundary); nil when not
    /// produced by an aligned prefill. Persisted in `metaState` (so a disk entry keeps it), kept by `copy` and `trim`.
    public var alignedPrefillTag: AlignedPrefillTag?

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
            stateGeneration += 1
            // New rows of unknown provenance: whoever sets a state that has a geometry (`copy`, a disk restore) sets
            // the tag after it.
            alignedPrefillTag = nil
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

    /// The K/V lane's metadata, then the aligned-prefill tag when the rows have one.
    public var metaState: [String] {
        get { kv.metaState + (alignedPrefillTag?.metaState ?? []) }
        set {
            let own = kv.metaState.count
            kv.metaState = Array(newValue.prefix(own))
            alignedPrefillTag = newValue.count > own ? AlignedPrefillTag(metaState: Array(newValue.dropFirst(own))) : nil
        }
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
            copy.alignedPrefillTag = alignedPrefillTag
        }
        return copy
    }

    // MARK: DiskCacheStateProviding

    public var diskCacheStateIdentifier: String { "minimax-m3-msa-kv-index-v1" }

    /// Restores all three lanes or nothing. The metadata is the K/V lane's, optionally followed by the aligned-prefill
    /// tag of the stored rows (entries written before aligned prefill have none, and restore without a tag).
    public func restoreDiskCacheState(_ state: [MLXArray], metadata: [String], offset: Int) -> Bool {
        let own = kv.metaState
        guard offset > 0, metadata.count >= own.count, Array(metadata.prefix(own.count)) == own,
            Self.isValid(state, offset: offset)
        else { return false }
        let extra = Array(metadata.dropFirst(own.count))
        let tag = extra.isEmpty ? nil : AlignedPrefillTag(metaState: extra)
        guard extra.isEmpty || tag != nil else { return false }
        self.state = state
        alignedPrefillTag = tag
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

extension MiniMaxM3SparseCache: AlignedPrefillCache {
    /// K/V rows and indexer-key rows.
    public var alignedPrefillLaneRows: [Int] { [offset, indexCount] }
}

// MARK: - Speculative checkpoint (DSpark verify)

extension MiniMaxM3SparseCache: SpeculativeCheckpointCache {
    /// The reference form (before 2026-10-29): saved array references of every lane.
    private struct ReferenceCheckpoint {
        let kv: Any
        let indexStorage: MLXArray?
        let indexCount: Int
        let dsparkTail: HybridPoolHiddenTail?
    }

    /// Row counts only. Every lane is append-only between a checkpoint and its restore (a verify appends rows at the
    /// end, growth copies the old rows into a larger buffer unchanged, a rollback trims), so the rows below the saved
    /// counts never change and a restore is a trim back to them.
    private struct RowCheckpoint {
        let offset: Int
        let indexCount: Int
        let dsparkTail: HybridPoolHiddenTail?
        let generation: Int
    }

    /// `VMLX_M3_REFERENCE_CHECKPOINT=1` restores the reference checkpoint (A/B only). A saved reference is a second
    /// owner of each lane's buffer, so MLX cannot donate it to the verify's first slice update: every DSpark round
    /// copied every layer's full-capacity K, V and index-key buffers (at 35k context 4.6 GiB per round on M3.1).
    static let referenceCheckpointEnvironment =
        ProcessInfo.processInfo.environment["VMLX_M3_REFERENCE_CHECKPOINT"] == "1"
    /// Tests: the checkpoint form for the current task (nil = the environment's).
    @TaskLocal static var referenceCheckpointOverride: Bool?
    static var referenceCheckpoint: Bool { referenceCheckpointOverride ?? referenceCheckpointEnvironment }

    public var speculativeCheckpointIsExact: Bool { kv.speculativeCheckpointIsExact }

    public func speculativeCheckpoint() -> Any {
        if Self.referenceCheckpoint {
            return ReferenceCheckpoint(
                kv: kv.speculativeCheckpoint(), indexStorage: indexStorage.map { $0[.ellipsis] },
                indexCount: indexCount, dsparkTail: dsparkTail)
        }
        return RowCheckpoint(offset: offset, indexCount: indexCount, dsparkTail: dsparkTail, generation: stateGeneration)
    }

    public func restoreSpeculativeCheckpoint(_ checkpoint: Any) {
        if let c = checkpoint as? RowCheckpoint {
            precondition(
                c.generation == stateGeneration && offset >= c.offset && indexCount >= c.indexCount,
                "MiniMaxM3SparseCache: speculative checkpoint restored after the lanes were replaced or trimmed below it")
            kv.trim(offset - c.offset)
            indexCount = c.indexCount
            dsparkTail = c.dsparkTail
            return
        }
        guard let c = checkpoint as? ReferenceCheckpoint else {
            preconditionFailure("MiniMaxM3SparseCache: foreign speculative checkpoint")
        }
        kv.restoreSpeculativeCheckpoint(c.kv)
        indexStorage = c.indexStorage
        indexCount = c.indexCount
        dsparkTail = c.dsparkTail
    }
}
