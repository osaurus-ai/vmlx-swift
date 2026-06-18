// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// MiniMax-M3 (MSA / Lightning-Indexer) sparse-attention cache.
//
// M3 attention is GQA on every layer. Layers 0-2 are full attention and use a
// stock `KVCacheSimple`. Layers 3-59 are sparse MSA: they carry TWO append-only
// caches in lockstep —
//   * the standard GQA KV cache         keys/values  [B, n_kv(=4), S, head_dim(=128)]
//   * the Lightning-Indexer key cache   idx_keys      [B, 1, S, index_dim(=128)]
//
// The indexer scores the current step's idx_q against ALL cached idx_keys,
// max-pools per 128-token block, and selects top-k blocks; the main branch then
// attends the selected K/V blocks. SELECTION IS RECOMPUTED EVERY STEP from
// idx_keys — it is never cached. Blocks are anchored to ABSOLUTE position
// (block = pos / 128), so the cache is append-only / trim-and-replay only: never
// shift, rotate, or evict mid-stream (that moves block boundaries and corrupts
// selection). Trimming to N slices BOTH lanes on the sequence axis in lockstep.
//
// CONTRACT (mirrors vllm-mlx `models/minimax_m3/cache.py`; lesson from the v1.5.62
// repetition-loop postmortem): this type must stay FIRST-CLASS through every
// reuse path — copy / fetch / trim / store / snapshot. `keys`, `values`,
// `idx_keys`, and `offset` move together. Never downcast to a plain KVCache: the
// generic helpers copy only (keys,values) and drop idx_keys → the indexer scores
// against corrupt keys → loops. `copy()` returns a `MiniMaxM3SparseCache`, and
// `state` carries all three tensors so the disk/prefix tiers round-trip the lane.
//
// Composite-cache precedent: `ZayaCCACache` (one inner `KVCacheSimple` + extra
// persistent lanes serialized through `state`/`metaState`). M3 is the simplest of
// the composite family — one extra tensor, no conv/SSM state, no compressor pool.

import Foundation
import MLX
import MLXFast

public final class MiniMaxM3SparseCache: KVCache {

    /// Standard GQA K/V half. Inherits the exact update / trim / step-growth
    /// semantics the runtime already relies on.
    private let kv: KVCacheSimple

    /// Lightning-Indexer key history `[B, 1, S, indexDim]`, append-only, grown in
    /// lockstep with `kv` so both share `offset`. `nil` until the first append.
    private var idxKeys: MLXArray?

    /// Indexer key width (e.g. 128). Carried in `metaState` for round-trip.
    public let indexDim: Int
    public let batchSize: Int

    public init(indexDim: Int = 128, batchSize: Int = 1) {
        self.kv = KVCacheSimple()
        self.indexDim = indexDim
        self.batchSize = batchSize
    }

    // MARK: KVCache conformance

    public var offset: Int { kv.offset }
    public var maxSize: Int? { nil }
    public var isTrimmable: Bool { true }

    /// Standard GQA K/V append. The attention forward calls this BEFORE
    /// `updateIndex` (upstream ordering), so after both appends `idxKeys`'
    /// sequence length equals `offset`.
    public func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        kv.update(keys: keys, values: values)
    }

    public func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        kv.makeMask(n: n, windowSize: windowSize, returnArray: returnArray)
    }

    /// Append-only trim: slice BOTH lanes to the same logical length. Returns the
    /// number of tokens actually trimmed.
    @discardableResult
    public func trim(_ n: Int) -> Int {
        let trimmed = kv.trim(n)
        if trimmed > 0, idxKeys != nil {
            idxKeys = sliceSeq(idxKeys!, to: kv.offset)
        }
        assertLanesAligned()
        return trimmed
    }

    // MARK: Lightning-Indexer lane (called by the MSA attention each step)

    /// Append this step's indexer keys `idx_k [B, 1, T, indexDim]` and return the
    /// full indexer history sliced to the current KV `offset` so `Sk` matches the
    /// SDPA K length. Grows in lockstep with the K/V side.
    @discardableResult
    public func updateIndex(_ idxK: MLXArray) -> MLXArray {
        if let prev = idxKeys {
            idxKeys = concatenated([prev, idxK], axis: 2)
        } else {
            idxKeys = idxK
        }
        assertLanesAligned()
        return readIndex()
    }

    /// The indexer history sliced to the current KV `offset`.
    public func readIndex() -> MLXArray {
        guard let idx = idxKeys else {
            return MLXArray.zeros([batchSize, 1, 0, indexDim], dtype: .bfloat16)
        }
        return offset > 0 ? sliceSeq(idx, to: offset) : idx
    }

    // MARK: Serialization — [keys, values, idx_keys]

    public var state: [MLXArray] {
        get {
            let kvState = kv.state
            let k: MLXArray
            let v: MLXArray
            if kvState.count == 2 {
                k = kvState[0]
                v = kvState[1]
            } else {
                // Empty KV — sentinels with a zero-length sequence axis.
                k = MLXArray.zeros([batchSize, 1, 0, 1], dtype: .bfloat16)
                v = MLXArray.zeros([batchSize, 1, 0, 1], dtype: .bfloat16)
            }
            let idx = idxKeys ?? MLXArray.zeros([batchSize, 1, 0, indexDim], dtype: .bfloat16)
            return [k, v, idx]
        }
        set {
            precondition(
                newValue.count == 3,
                "MiniMaxM3SparseCache.state requires [keys, values, idx_keys]")
            let k = newValue[0]
            let v = newValue[1]
            let idx = newValue[2]
            if k.ndim >= 3 && k.dim(2) == 0 {
                // Leave kv empty — offset stays 0.
            } else {
                kv.state = [k, v]
            }
            // Zero-length sequence axis ⇒ no indexer history yet.
            idxKeys = (idx.ndim >= 3 && idx.dim(2) == 0) ? nil : idx
            assertLanesAligned()
        }
    }

    /// metaState: kv's trailer + an M3 sentinel carrying `indexDim`.
    public var metaState: [String] {
        get { kv.metaState + ["minimax_m3_v1", String(indexDim), String(batchSize)] }
        set {
            let trailer = 3
            if newValue.count >= trailer,
                newValue[newValue.count - trailer] == "minimax_m3_v1"
            {
                kv.metaState = Array(newValue[0 ..< (newValue.count - trailer)])
            } else {
                kv.metaState = newValue
            }
        }
    }

    /// Deep copy that PRESERVES the sparse type AND all three lanes — the single
    /// safe path for prefix/disk/snapshot reuse. Never returns a plain KVCache.
    public func copy() -> any KVCache {
        let dup = MiniMaxM3SparseCache(indexDim: indexDim, batchSize: batchSize)
        dup.state = self.state.map { $0[.ellipsis] }
        dup.metaState = self.metaState
        return dup
    }

    // MARK: helpers

    /// Slice a `[B, H, S, D]` tensor to the first `n` positions on the sequence axis.
    private func sliceSeq(_ a: MLXArray, to n: Int) -> MLXArray {
        a[0..., 0..., 0 ..< n, 0...]
    }

    /// DEBUG-only invariant: a wrong restored offset corrupts attention even when
    /// shapes look right (postmortem). idx_keys length must equal `offset`.
    private func assertLanesAligned() {
        #if DEBUG
        if let idx = idxKeys {
            assert(
                idx.dim(2) == kv.offset,
                "MiniMaxM3SparseCache lanes desynced: idx_keys S=\(idx.dim(2)) != kv.offset=\(kv.offset)")
        }
        #endif
    }
}
