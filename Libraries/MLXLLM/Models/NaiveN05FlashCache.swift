import Foundation
import MLX
import MLXLMCommon

/// Model-owned transactional pair: sparse indexer rows cannot be restored or
/// trimmed independently of attention KV. Sliding caches retain window-1 rows.
/// Batch conversion and speculative rollback after window wrap are not enabled.
final class NaiveN05FlashCache: DiskCacheStateProviding {
    enum Failure: Error { case invalidGeometry, invalidCompanion }
    private(set) var offset: Int = 0
    var maxSize: Int? { window }
    func innerState() -> [MLXArray] { rows }
    let window: Int?
    let requiresIndexer: Bool
    /// Full-attention layers (no window) keep capacity buffers grown in ``step``-row steps and write new rows in
    /// place, so a decode step does not copy the whole history (the concatenating append copied K, V and the fp32
    /// indexer keys of every DSA layer every token: tens of MB per layer per token at 20k context). `rows` exposes
    /// only the valid prefix. Sliding layers keep at most window - 1 rows and stay on the concatenating append.
    private static let step = 256
    private var storage: [MLXArray] = []
    private var count = 0
    private var rows: [MLXArray] {
        // Always views, never the buffer objects: an in-place write swaps new contents into the buffer object, and
        // a caller holding it (a snapshot, a lazy store) must keep the rows it was handed.
        get { storage.map { $0[0..., 0..., ..<count, 0...] } }
        set {
            storage = newValue
            count = newValue.first?.dim(2) ?? 0
        }
    }
    var keyOffset: Int { offset - count }
    init(window: Int?, requiresIndexer: Bool) {
        precondition(window == nil || window! > 0)
        self.window = window
        self.requiresIndexer = requiresIndexer
    }
    func makeMask(n: Int, windowSize: Int?, returnArray: Bool) -> MLXFast.ScaledDotProductAttentionMaskMode {
        let length = (rows.first?.dim(2) ?? 0) + n
        let padding = MLXArray.ones([1, offset + n], dtype: .bool)
        return .array(NaiveN05FlashMath.allowedMask(padding: padding, queryOffset: offset,
            length: n, keyOffset: keyOffset, keyLength: length, window: window ?? windowSize))
    }
    var diskCacheStateIdentifier: String { "naive-n05-paired-v1-window-\(window ?? 0)-indexer-\(requiresIndexer)" }
    var state: [MLXArray] {
        get { rows }
        set {
            // Generic restore must use the validated DiskCacheStateProviding API.
            precondition(newValue.isEmpty || valid(newValue, offset: offset))
            rows = newValue
        }
    }
    var metaState: [String] {
        get { [diskCacheStateIdentifier, String(offset)] }
        set {
            precondition(newValue.count == 2 && newValue[0] == diskCacheStateIdentifier)
            offset = Int(newValue[1])!
        }
    }
    private func valid(_ arrays: [MLXArray], offset: Int) -> Bool {
        if offset == 0 && arrays.isEmpty { return true }
        guard offset >= 0, arrays.count == (requiresIndexer ? 3 : 2),
            arrays.allSatisfy({ $0.ndim == 4 }), let key = arrays.first,
            key.dim(2) <= offset, arrays[1].dim(0) == key.dim(0),
            arrays[1].dim(1) == key.dim(1), arrays[1].dim(2) == key.dim(2)
        else { return false }
        if let window, key.dim(2) != min(offset, window - 1) { return false }
        if window == nil && key.dim(2) != offset { return false }
        if requiresIndexer {
            let index = arrays[2]
            guard index.dim(0) == key.dim(0), index.dim(1) == 1,
                index.dim(2) == key.dim(2), index.dtype == .float32
            else { return false }
        }
        return true
    }
    func restoreDiskCacheState(_ state: [MLXArray], metadata: [String], offset: Int) -> Bool {
        guard metadata == [diskCacheStateIdentifier, String(offset)], valid(state, offset: offset) else { return false }
        // Validate the entire pair before touching either leaf.
        rows = state
        self.offset = offset
        return true
    }
    func append(keys: MLXArray, values: MLXArray, indexer: MLXArray?) throws -> (MLXArray, MLXArray, MLXArray?) {
        guard keys.ndim == 4, values.ndim == 4, keys.dim(2) > 0,
            keys.shape.dropLast() == values.shape.dropLast() else { throw Failure.invalidGeometry }
        guard requiresIndexer == (indexer != nil) else { throw Failure.invalidCompanion }
        if let indexer {
            guard indexer.ndim == 4, indexer.dim(0) == keys.dim(0), indexer.dim(1) == 1,
                indexer.dim(2) == keys.dim(2), indexer.dtype == .float32 else { throw Failure.invalidCompanion }
        }
        let incoming = [keys, values] + (indexer.map { [$0] } ?? [])
        if !rows.isEmpty {
            guard zip(rows, incoming).allSatisfy({ old, new in
                old.dim(0) == new.dim(0) && old.dim(1) == new.dim(1)
                && old.dim(3) == new.dim(3) && old.dtype == new.dtype
            }) else { throw Failure.invalidGeometry }
        }
        let n = keys.dim(2)
        if window == nil {
            let needed = count + n
            if storage.isEmpty || needed > storage[0].dim(2) {
                let capacity = (needed + Self.step - 1) / Self.step * Self.step
                storage = (0 ..< incoming.count).map { i in
                    var shape = incoming[i].shape
                    shape[2] = capacity - count
                    let fresh = MLXArray.zeros(shape, dtype: incoming[i].dtype)
                    return count == 0 ? fresh : concatenated([rows[i], fresh], axis: 2)
                }
            }
            for i in incoming.indices {
                storage[i][0..., 0..., count ..< needed, 0...] = incoming[i]
            }
            count = needed
            offset += n
            let full = rows
            return (full[0], full[1], requiresIndexer ? full[2] : nil)
        }
        let full = rows.isEmpty ? incoming : zip(rows, incoming).map { concatenated([$0, $1], axis: 2) }
        let start = max(0, full[0].dim(2) - (window! - 1))
        rows = full.map { $0[0..., 0..., start..., 0...] }
        offset += n
        return (full[0], full[1], requiresIndexer ? full[2] : nil)
    }
    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        precondition(!requiresIndexer, "Sparse Naive cache requires atomic companion append")
        let result = try! append(keys: keys, values: values, indexer: nil)
        return (result.0, result.1)
    }
    var isTrimmable: Bool { window == nil }
    func trim(_ n: Int) -> Int {
        guard n > 0, window == nil, !rows.isEmpty else { return 0 }
        let removed = min(n, offset)
        count -= removed
        offset -= removed
        return removed
    }
    func copy() -> any KVCache {
        let result = NaiveN05FlashCache(window: window, requiresIndexer: requiresIndexer)
        // The copy owns its rows: a lazy `+ 0` over the valid prefix, independent of later in-place appends here.
        result.rows = rows.map { $0 + MLXArray(0, dtype: $0.dtype) }
        result.offset = offset
        return result
    }
}

/// Exact-prefix restore (``ExactPrefixRestoreCache``): a stored boundary holds exactly the K, V and fp32 indexer rows
/// the forward produced for those tokens (sliding layers: exactly the last window - 1 rows; no recurrent, pooled or
/// quantized state), so forced tool selection may restore it like a plain KV cache. Without this every tool-named
/// osaurus turn (tool_choice=required) re-prefilled the whole history: 40 s at 21k tokens, 167 s at 48k.
extension NaiveN05FlashCache: ExactPrefixRestoreCache {}
