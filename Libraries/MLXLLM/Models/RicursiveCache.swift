import Foundation
import MLX
import MLXFast
import MLXLMCommon

/// KV cache of a Ripple sliding-attention layer. A query reads at most the newest `window` positions, so older rows
/// can never be read again: they are not persisted, and a restored cache holds fewer physical rows than its logical
/// `offset`. Rows stay in temporal order in an append-only capacity buffer (in-place writes, no per-step copy), so
/// `trim` is an exact rollback as long as `window - 1` rows remain before the kept position, including past the
/// window (a ring buffer cannot do that, and speculative rollback needs it).
///
/// A cold prefill keeps every row until the request ends: boundaries stored at the end of the turn (a stable system
/// prompt, the prompt, the answer) need the rows before them, and those rows are never recomputed. Only what is
/// written out, and what a later request restores, is bounded: at 15.5k tokens an entry is about 1.3 GB instead of
/// 5 GB, since 64 of the 80 layers are sliding.
public final class RicursiveSlidingKVCache: DiskCacheStateProviding, NonQuantizableKVCache,
    AppendOnlyRowsCache, ExactPrefixRestoreCache
{
    /// Rows beyond the window kept in an entry, so a restored cache can still roll back (a speculative verify is at
    /// most two trained blocks) and take the exact-hit `trim(1)`.
    public static let rollbackRows = 64
    private static let step = 256

    public let window: Int
    public private(set) var offset = 0
    public var maxSize: Int? { nil }
    private var keys: MLXArray?
    private var values: MLXArray?
    /// Valid physical rows; capacity past it is unused. The first physical row is at position `offset - count`.
    private var count = 0
    public private(set) var isPrefixView = false

    public init(window: Int) {
        precondition(window > 0)
        self.window = window
    }

    /// Rows an entry stores at most.
    private var storedRows: Int { window - 1 + Self.rollbackRows }

    public func innerState() -> [MLXArray] { [keys, values].compactMap { $0 } }

    public func update(keys newKeys: MLXArray, values newValues: MLXArray) -> (MLXArray, MLXArray) {
        let rows = newKeys.dim(2)
        let needed = count + rows
        if keys == nil || needed > keys!.dim(2) {
            let capacity = (needed + Self.step - 1) / Self.step * Self.step
            func grown(_ old: MLXArray?, like new: MLXArray) -> MLXArray {
                var shape = new.shape
                shape[2] = capacity - count
                let fresh = MLXArray.zeros(shape, dtype: new.dtype)
                guard let old, count > 0 else { return fresh }
                return concatenated([old[0..., 0..., ..<count, 0...], fresh], axis: 2)
            }
            keys = grown(keys, like: newKeys)
            values = grown(values, like: newValues)
        }
        keys![0..., 0..., count ..< needed, 0...] = newKeys
        values![0..., 0..., count ..< needed, 0...] = newValues
        count = needed
        offset += rows
        return (keys![0..., 0..., ..<count, 0...], values![0..., 0..., ..<count, 0...])
    }

    public var state: [MLXArray] {
        get {
            guard let keys, let values, count > 0 else { return [] }
            let start = Swift.max(0, count - storedRows)
            return [keys[0..., 0..., start ..< count, 0...], values[0..., 0..., start ..< count, 0...]]
        }
        set {
            precondition(newValue.isEmpty || newValue.count == 2, "sliding cache state is keys and values")
            keys = newValue.first
            values = newValue.last
            count = newValue.first?.dim(2) ?? 0
        }
    }

    public var diskCacheStateIdentifier: String { "ricursive-sliding-v1-window-\(window)" }

    public var metaState: [String] {
        get { [diskCacheStateIdentifier, String(offset)] }
        set {
            guard newValue.count == 2, newValue[0] == diskCacheStateIdentifier, let restored = Int(newValue[1]) else {
                preconditionFailure("foreign sliding cache metadata")
            }
            offset = restored
        }
    }

    /// Validates the pair and refuses an entry that lacks the window for its position.
    public func restoreDiskCacheState(_ state: [MLXArray], metadata: [String], offset: Int) -> Bool {
        guard metadata == [diskCacheStateIdentifier, String(offset)], offset > 0, state.count == 2 else { return false }
        let (k, v) = (state[0], state[1])
        guard k.ndim == 4, v.ndim == 4, k.dim(0) == v.dim(0), k.dim(1) == v.dim(1), k.dim(2) == v.dim(2),
            k.dtype.isFloatingPoint, k.dtype == v.dtype,
            k.dim(2) <= offset, k.dim(2) >= Swift.min(offset, window - 1)
        else { return false }
        keys = k
        values = v
        count = k.dim(2)
        self.offset = offset
        return true
    }

    public var isTrimmable: Bool { true }

    /// Exact rollback, limited to what leaves the kept position its full window. Reports what it did; a caller that
    /// asked for more treats the entry as unusable, it never gets a shortened window silently.
    @discardableResult
    public func trim(_ n: Int) -> Int {
        let wanted = Swift.min(n, offset)
        let remainingPositions = offset - wanted
        let removable = count - Swift.min(remainingPositions, window - 1)
        let trimmed = Swift.max(0, Swift.min(wanted, removable))
        count -= trimmed
        offset -= trimmed
        return trimmed
    }

    public var supportsPrefixViews: Bool { true }

    /// Views (no copy) of the rows up to position `rows`. Only valid while the rows are physically held, which
    /// `cacheSupportsPrefixViews` cannot see for a restored cache, so `rows` below the held range traps as for any
    /// other view: callers ask for boundaries at or past the restored position.
    public func prefixView(rows: Int) -> any KVCache {
        let physical = rows - (offset - count)
        precondition(rows > 0 && rows <= offset && physical >= Swift.min(rows, window - 1), "sliding view of \(rows) rows")
        let view = RicursiveSlidingKVCache(window: window)
        if let keys, let values {
            view.keys = keys[0..., 0..., ..<physical, 0...]
            view.values = values[0..., 0..., ..<physical, 0...]
        }
        view.count = physical
        view.offset = rows
        view.isPrefixView = true
        return view
    }

    /// Position-aware mask over the rows this cache returns (Ripple's attention builds its own).
    public func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        if n == 1 { return .none }
        let queries = MLXArray(Int32(offset) ..< Int32(offset + n))[0..., .newAxis]
        let first = offset - count
        let held = MLXArray(Int32(first) ..< Int32(offset + n))[.newAxis]
        return .array((held .<= queries) .&& (held .> (queries - Int32(windowSize ?? window))))
    }

    public func copy() -> any KVCache {
        let new = RicursiveSlidingKVCache(window: window)
        if let keys, let values, count > 0 {
            // Only what any later read or store can use, in owned buffers.
            let start = Swift.max(0, count - storedRows)
            let k = concatenated([keys[0..., 0..., start ..< count, 0...]], axis: 2) + MLXArray(0, dtype: keys.dtype)
            let v = concatenated([values[0..., 0..., start ..< count, 0...]], axis: 2) + MLXArray(0, dtype: values.dtype)
            new.keys = k
            new.values = v
            new.count = count - start
        }
        new.offset = offset
        return new
    }
}
