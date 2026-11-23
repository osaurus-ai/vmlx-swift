// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX

/// KV cache for a sliding-window attention layer whose rows must stay in temporal order (a relative position bias,
/// a position-aware mask): a logical `offset` over a physical buffer that only holds the recent rows.
///
/// A query at position p reads keys p - window + 1 ... p, so once the context passes the window every older row is
/// dead weight in RAM and in every stored prefix entry. This cache keeps the last `window - 1 + rollbackRows` rows
/// (`keep`) plus growth headroom; a copy, a stored entry and `state` hold exactly `min(offset, keep)` rows.
///
/// Positions: the held rows are positions `offset - heldRows ..< offset`, oldest first. `update` returns views of
/// every held row; attention takes the last `window - 1 + T` of them (`S = keys.dim(2)` physical, first key at
/// `offset_before + T - S`), so callers that already compute key positions from the returned row count are exact.
///
/// Rollback: compaction runs only at the start of an `update`, before the new rows are written, so the rows an
/// update appended are always held and a speculative verify can drop any number of them. `rollbackRows` only covers
/// trims across requests (the exact-hit `trim(1)`, the post-answer stop row). It never bounds a draft width.
public final class TemporalWindowKVCache: DiskCacheStateProviding, NonQuantizableKVCache, ExactPrefixRestoreCache,
    CacheRetainedByteCountProviding
{
    public static let defaultRollbackRows = 64
    /// Growth headroom: a compaction copies `keep` rows once per this many appended rows.
    static let growth = 256

    /// Positions a query attends to, including its own.
    public let window: Int
    /// Rows held past `window - 1` so a stored entry can still be trimmed a little.
    public let rollbackRows: Int
    /// Optional shape expectations checked on a disk restore.
    public let expectedHeads: Int?
    public let expectedHeadDim: Int?

    public private(set) var offset = 0
    public var maxSize: Int? { nil }

    private var keys: MLXArray?
    private var values: MLXArray?
    /// Valid physical rows, `[0, count)` of the buffers, temporal; the last is position `offset - 1`.
    private var count = 0

    public init(window: Int, rollbackRows: Int = defaultRollbackRows, heads: Int? = nil, headDim: Int? = nil) {
        precondition(window > 0 && rollbackRows >= 0)
        self.window = window
        self.rollbackRows = rollbackRows
        self.expectedHeads = heads
        self.expectedHeadDim = headDim
    }

    /// Rows any copy, stored entry or `state` holds at most.
    public var keep: Int { window - 1 + rollbackRows }
    /// Physical rows held now (live buffer; may exceed `keep` by up to the growth headroom).
    public var heldRows: Int { count }
    /// Position of the oldest held row.
    public var firstPosition: Int { offset - count }
    /// Allocated rows of the live buffer.
    public var capacity: Int { keys?.dim(2) ?? 0 }

    public func innerState() -> [MLXArray] { [keys, values].compactMap { $0 } }

    public func update(keys newKeys: MLXArray, values newValues: MLXArray) -> (MLXArray, MLXArray) {
        let rows = newKeys.dim(2)
        if keys == nil || count + rows > capacity {
            // Compact into a fresh buffer (never an overlapping in-place shift: a checkpoint or a stored snapshot
            // may still reference the old one): the last `keep` held rows, then room for this update + headroom.
            let retained = Swift.min(count, keep)
            let newCapacity = (retained + rows + Self.growth + Self.growth - 1) / Self.growth * Self.growth
            func rebuilt(_ old: MLXArray?, like new: MLXArray) -> MLXArray {
                var shape = new.shape
                shape[2] = newCapacity - retained
                let fresh = MLXArray.zeros(shape, dtype: new.dtype)
                guard let old, retained > 0 else { return fresh }
                return concatenated([old[0..., 0..., (count - retained) ..< count, 0...], fresh], axis: 2)
            }
            keys = rebuilt(keys, like: newKeys)
            values = rebuilt(values, like: newValues)
            count = retained
        }
        keys![0..., 0..., count ..< (count + rows), 0...] = newKeys
        values![0..., 0..., count ..< (count + rows), 0...] = newValues
        count += rows
        offset += rows
        return (keys![0..., 0..., ..<count, 0...], values![0..., 0..., ..<count, 0...])
    }

    /// The last `min(count, keep)` held rows (views).
    public var state: [MLXArray] {
        get {
            guard let keys, let values, count > 0 else { return [] }
            let start = Swift.max(0, count - keep)
            return [keys[0..., 0..., start ..< count, 0...], values[0..., 0..., start ..< count, 0...]]
        }
        set {
            // Generic seat (prompt-cache files): rows only; `metaState` sets the position afterwards.
            precondition(newValue.isEmpty || newValue.count == 2, "windowed cache state is keys and values")
            keys = newValue.first
            values = newValue.last
            count = newValue.first?.dim(2) ?? 0
        }
    }

    public var diskCacheStateIdentifier: String { "temporal-window-kv-v1-window-\(window)" }

    public var metaState: [String] {
        get { [diskCacheStateIdentifier, String(offset)] }
        set {
            guard newValue.count == 2, newValue[0] == diskCacheStateIdentifier, let restored = Int(newValue[1]),
                restored >= count, count >= Swift.min(restored, window - 1)
            else {
                preconditionFailure("windowed cache metadata \(newValue) does not fit \(count) held rows")
            }
            offset = restored
        }
    }

    /// Accepts only a pair that holds the full window of position `offset - 1`; anything else is refused whole.
    public func restoreDiskCacheState(_ state: [MLXArray], metadata: [String], offset: Int) -> Bool {
        guard metadata == [diskCacheStateIdentifier, String(offset)], offset > 0, state.count == 2 else { return false }
        let (k, v) = (state[0], state[1])
        guard k.ndim == 4, v.ndim == 4, k.dim(0) == v.dim(0), k.dim(1) == v.dim(1), k.dim(2) == v.dim(2),
            k.dtype.isFloatingPoint, k.dtype == v.dtype,
            k.dim(2) <= offset, k.dim(2) >= Swift.min(offset, window - 1),
            expectedHeads.map({ k.dim(1) == $0 }) ?? true,
            expectedHeadDim.map({ k.dim(3) == $0 && v.dim(3) == $0 }) ?? true
        else { return false }
        // Restored arrays may be shared (resident disk entries): capacity == count, so the first update reallocates
        // instead of writing into them.
        keys = k
        values = v
        count = k.dim(2)
        self.offset = offset
        return true
    }

    public var isTrimmable: Bool { true }

    /// Whether `trim(n)` would remove exactly `min(n, offset)` rows.
    public func canTrim(_ n: Int) -> Bool {
        let wanted = Swift.min(n, offset)
        return count - wanted >= Swift.min(offset - wanted, window - 1)
    }

    /// All or nothing: removes `min(n, offset)` rows when the remaining position keeps its full window, otherwise
    /// changes nothing and returns 0 (a caller that needs the rows gone treats the cache as unusable).
    @discardableResult
    public func trim(_ n: Int) -> Int {
        let wanted = Swift.min(n, offset)
        guard wanted > 0, canTrim(wanted) else { return 0 }
        count -= wanted
        offset -= wanted
        return wanted
    }

    /// Position-aware mask over the rows `update` returns.
    public func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        if n == 1 { return .none }
        let queries = MLXArray(Int32(offset) ..< Int32(offset + n))[0..., .newAxis]
        let held = MLXArray(Int32(offset - count) ..< Int32(offset + n))[.newAxis]
        return .array((held .<= queries) .&& (held .> (queries - Int32(windowSize ?? window))))
    }

    /// Owned copy of the last `min(count, keep)` rows, capacity == rows.
    public func copy() -> any KVCache {
        let new = TemporalWindowKVCache(
            window: window, rollbackRows: rollbackRows, heads: expectedHeads, headDim: expectedHeadDim)
        if let keys, let values, count > 0 {
            let start = Swift.max(0, count - keep)
            new.keys = keys[0..., 0..., start ..< count, 0...] + MLXArray(0, dtype: keys.dtype)
            new.values = values[0..., 0..., start ..< count, 0...] + MLXArray(0, dtype: values.dtype)
            new.count = count - start
        }
        new.offset = offset
        return new
    }

    public var retainedCacheByteCount: Int { (keys?.nbytes ?? 0) + (values?.nbytes ?? 0) }
}

extension TemporalWindowKVCache: SpeculativeCheckpointCache {
    private struct Checkpoint {
        let keys: MLXArray?
        let values: MLXArray?
        let offset: Int
        let count: Int
    }

    public var speculativeCheckpointIsExact: Bool { true }

    public func speculativeCheckpoint() -> Any {
        Checkpoint(keys: speculativeReference(keys), values: speculativeReference(values), offset: offset, count: count)
    }

    public func restoreSpeculativeCheckpoint(_ checkpoint: Any) {
        guard let c = checkpoint as? Checkpoint else {
            preconditionFailure("TemporalWindowKVCache: foreign speculative checkpoint")
        }
        keys = c.keys
        values = c.values
        offset = c.offset
        count = c.count
    }
}
