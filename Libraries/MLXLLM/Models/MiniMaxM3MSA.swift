// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXLMCommon

// MiniMax-M3 MSA (block-sparse attention): the block selection and the attention paths that consume it.
//
// Port of the vMLX Python runtime's `vmlx_engine/models/minimax_m3/msa.py` (perf/minimax-m3-block-sparse-msa,
// 885355f0): the selection as block ids, the dense mask path (the reference) and the op-level gather path, with the
// same env lever. The block-sparse Metal kernels are separate.
//
// The Lightning indexer picks, for every query row and KV group, `keep` (16) key blocks of `block` (128) keys: the top
// block scores, the query's own (local) block forced in. Attention then sees only those blocks, causally.
//
// `MiniMaxM3MSASelection` carries that pick as block ids `[B, G, Sq, keep]` plus the positions it was made at. Paths
// (`choosePath` routes by the measured crossovers; `VMLX_MINIMAX_MSA` forces one for comparisons):
//
// - `dense`: the additive `[B, G, Sq, Sk]` float32 key mask (`denseBias`) and one SDPA over all Sk keys per KV group.
//   The reference path.
// - `gather`: MLX ops only (gather the selected keys, SDPA over them). Never routed to; an op-level reference.
//
// Positions: query row r sits at `offset + r`; key j is allowed when its block is selected and `j <= offset + r`.
// Keys at or past `keyLength` (the logical extent, i.e. the cache offset) are never read: the kernels stop at the
// row's position (< keyLength) and the gather clamps to `keyLength - 1` and masks the clamped lanes. A partial last
// block, a query's own block (causal inside it) and selected blocks that lie entirely in the future (possible when
// fewer than `keep` blocks precede a prefill row) are all handled by that one positional rule.

/// Top-k key blocks per (query row, KV group) for queries at `offset ..< offset + Sq` over keys `0 ..< keyLength`.
///
/// Positions are `arange` ops (int32, built on the device), not host arrays: a host `MLXArray(0 ..< Sk)` costs
/// milliseconds per call at long context.
struct MiniMaxM3MSASelection {
    /// `[B, G, Sq, keep]`: distinct block ids per row, in `argPartition` order.
    let topIndices: MLXArray
    let offset: Int
    let keyLength: Int
    let blockSize: Int

    var groups: Int { topIndices.dim(1) }
    var rows: Int { topIndices.dim(2) }
    var keep: Int { topIndices.dim(3) }
    var blockCount: Int { (keyLength + blockSize - 1) / blockSize }

    /// The additive key mask `[B, G, Sq, keyLength]` (float32 0 / -inf, causal): block selected and key at or before
    /// the query's position.
    func denseBias() -> MLXArray {
        let (B, G, Sq) = (topIndices.dim(0), topIndices.dim(1), topIndices.dim(2))
        let queryPositions = arange(offset, offset + Sq).reshaped(1, 1, Sq, 1)
        let keyPositions = arange(0, keyLength).reshaped(1, 1, 1, keyLength)
        var blockKeep = MLXArray.full([B, G, Sq, blockCount], values: MLXArray(-Float.infinity), dtype: .float32)
        blockKeep = putAlong(blockKeep, topIndices, values: MLXArray(Float(0)), axis: -1)
        let keyBias = repeated(blockKeep, count: blockSize, axis: -1)[.ellipsis, ..<keyLength]
        return MLX.where(keyPositions .> queryPositions, MLXArray(-Float.infinity), keyBias)
    }
}

enum MiniMaxM3MSA {
    // MARK: routing

    enum Path: String, Sendable, CaseIterable {
        case dense, gather
    }

    /// `VMLX_MINIMAX_MSA`: `auto` (unset) routes by the measured crossovers; `dense` forces the mask path, `sparse` the
    /// block-sparse path for the call's size (decode kernel for decode-sized calls, chunk kernel for longer ones),
    /// `gather` / `decode` / `kernel` that path wherever its shapes are supported.
    enum Request: String, Sendable {
        case auto, dense, sparse, gather, decode, kernel
    }

    static let environmentVariable = "VMLX_MINIMAX_MSA"

    /// Test / diagnostics hooks: a forced request that wins over the environment, and an observer of every routed
    /// pick (the Python tests' router spy).
    final class Hooks: @unchecked Sendable {
        private let lock = NSLock()
        private var forced: Request?
        private var observer: ((Path) -> Void)?

        var request: Request? {
            get { lock.withLock { forced } }
            set { lock.withLock { forced = newValue } }
        }

        var pathObserver: ((Path) -> Void)? {
            get { lock.withLock { observer } }
            set { lock.withLock { observer = newValue } }
        }
    }

    static let hooks = Hooks()

    static func requestedPath() -> Request {
        if let forced = hooks.request { return forced }
        let raw = RuntimeEnvironment.value(environmentVariable)?
            .trimmingCharacters(in: .whitespaces).lowercased() ?? ""
        guard let request = Request(rawValue: raw), request != .auto else { return .auto }
        return request
    }

    /// `dense` or `gather` for this call. Only picks a path; every path computes the same
    /// attention (the sparse ones over the selected blocks only). The pick depends on (rows, keys, dtype, shapes)
    /// only, so a one-row verify call takes exactly the path a one-row decode step takes.
    static func choosePath(
        _ selection: MiniMaxM3MSASelection, queries: MLXArray, keys: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode, kvHeads: Int
    ) -> Path {
        let path = route(selection, queries: queries, keys: keys, mask: mask, kvHeads: kvHeads)
        hooks.pathObserver?(path)
        return path
    }

    private static func route(
        _ selection: MiniMaxM3MSASelection, queries: MLXArray, keys: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode, kvHeads: Int
    ) -> Path {
        // Indexer heads that do not line up with KV groups, or an extra array mask (batch padding): the dense mask
        // path covers these.
        if selection.groups != kvHeads { return .dense }
        switch mask {
        case .array, .arrays: return .dense
        case .none, .causal: break
        }
        // The block-sparse kernels land separately; until then every other request takes the mask path.
        return requestedPath() == .gather ? .gather : .dense
    }

    // MARK: selection

    /// Block selection for queries `[B, G, Sq, d]` at positions `offset...` over index keys `[B, 1, Sk, d]`; nil when
    /// every block is visible (`keep >= blocks`, i.e. at most `topK * blockSize` keys: plain causal attention).
    ///
    /// Each indexer head is one KV group and keeps its own selection. Scores are float32; each block scores the max
    /// over its keys (future keys -inf, the last block padded with -inf); the query's own block and the
    /// `localBlocks - 1` before it are forced in; the top `topK` blocks per (row, group) are kept. One query row over
    /// its own key prefix is exactly a decode step's selection.
    static func select(
        queries: MLXArray, keys: MLXArray, offset: Int, blockSize: Int, topK: Int, localBlocks: Int
    ) -> MiniMaxM3MSASelection? {
        let (B, heads, queryLength) = (queries.dim(0), queries.dim(1), queries.dim(2))
        let keyLength = keys.dim(2)
        let blockCount = (keyLength + blockSize - 1) / blockSize
        let keep = min(topK, blockCount)
        if keep >= blockCount {
            return nil
        }
        var scores = matmul(queries.asType(.float32), keys.asType(.float32).transposed(0, 1, 3, 2))
        let queryPositions = arange(offset, offset + queryLength).reshaped(1, 1, queryLength, 1)
        let keyPositions = arange(0, keyLength).reshaped(1, 1, 1, keyLength)
        scores = MLX.where(keyPositions .> queryPositions, MLXArray(-Float.infinity), scores)
        let padding = blockCount * blockSize - keyLength
        if padding > 0 {
            scores = concatenated(
                [scores, MLXArray.full([B, heads, queryLength, padding], values: MLXArray(-Float.infinity))],
                axis: -1)
        }
        var blockScores = scores.reshaped(B, heads, queryLength, blockCount, blockSize).max(axis: -1)
        if localBlocks > 0 {
            let queryBlocks = blockIndex(queryPositions.reshaped(1, queryLength), blockSize)
            let lags = arange(0, localBlocks).reshaped(1, 1, 1, localBlocks)
            let local = broadcast(
                maximum(queryBlocks.reshaped(1, 1, queryLength, 1) - lags, MLXArray(Int32(0))),
                to: [B, heads, queryLength, localBlocks])
            blockScores = putAlong(blockScores, local, values: MLXArray(Float.infinity), axis: -1)
        }
        let top = argPartition(-blockScores, kth: keep - 1, axis: -1)[.ellipsis, ..<keep]
        return MiniMaxM3MSASelection(topIndices: top, offset: offset, keyLength: keyLength, blockSize: blockSize)
    }

    /// `floor(positions / block)` for non-negative int32 positions: a right shift when the block is a power of two.
    /// (MLX's integer `floorDivide` is a chain of ops: it measured ~0.4 ms per call in this build, more than the rest
    /// of a decode step's selection; the shift is ~0.02 ms and exact for non-negative values.)
    static func blockIndex(_ positions: MLXArray, _ block: Int) -> MLXArray {
        if block > 0 && block & (block - 1) == 0 {
            return rightShift(positions, MLXArray(Int32(block.trailingZeroBitCount)))
        }
        return floorDivide(positions, MLXArray(Int32(block)))
    }

    /// The selection covers more keys than K / V hold (the index lane and the K / V lane have desynchronised), or the
    /// query rows end past the selected keys: reading on would gather rows that were never written.
    static func checkExtent(_ keys: MLXArray, _ values: MLXArray, _ selection: MiniMaxM3MSASelection) {
        precondition(
            keys.dim(2) >= selection.keyLength && values.dim(2) >= selection.keyLength,
            "MiniMax-M3 MSA selection covers \(selection.keyLength) keys but K / V hold \(keys.dim(2)) / "
                + "\(values.dim(2))")
        precondition(
            selection.offset + selection.rows <= selection.keyLength,
            "MiniMax-M3 MSA query rows end at \(selection.offset + selection.rows) past the \(selection.keyLength) "
                + "selected keys")
    }

    // MARK: gather (MLX ops)

    /// (source index `[B, G, Sq, N]` clamped below keyLength, allowed `[B, G, Sq, N]`) for the N = keep * block
    /// selected keys of every row, blocks in ascending order.
    static func keyPositions(_ selection: MiniMaxM3MSASelection) -> (MLXArray, MLXArray) {
        let (B, G, Sq, keep) = (
            selection.topIndices.dim(0), selection.groups, selection.rows, selection.keep
        )
        let block = selection.blockSize
        let blocks = sorted(selection.topIndices.asType(.int32), axis: -1)
        var positions =
            expandedDimensions(blocks, axis: -1) * MLXArray(Int32(block))
            + arange(0, block)
        positions = positions.reshaped(B, G, Sq, keep * block)
        let queryPositions = arange(selection.offset, selection.offset + Sq)
            .reshaped(1, 1, Sq, 1)
        let allowed = positions .<= queryPositions
        return (minimum(positions, MLXArray(Int32(selection.keyLength - 1))), allowed)
    }

    /// Attention `[B, H, L, D]` of q over only the selected key blocks of each (row, KV group), from MLX ops. Every
    /// row is one SDPA batch element over its own keep * block gathered keys.
    static func gatheredAttention(
        queries q: MLXArray, keys k: MLXArray, values v: MLXArray, selection: MiniMaxM3MSASelection, scale: Float
    ) -> MLXArray {
        let (B, H, L, D) = (q.dim(0), q.dim(1), q.dim(2), q.dim(3))
        let G = selection.groups
        let hg = H / G
        checkExtent(k, v, selection)
        let (source, allowed) = keyPositions(selection)
        let N = source.dim(-1)
        let bias = MLX.where(allowed, MLXArray(Float(0)).asType(q.dtype), MLXArray(-Float.infinity).asType(q.dtype))
        if L == 1 {
            let index = source.reshaped(B, G, N, 1)
            let ks = takeAlong(k, index, axis: 2)
            let vs = takeAlong(v, index, axis: 2)
            let o = MLXFast.scaledDotProductAttention(
                queries: q.reshaped(B * G, hg, 1, D), keys: ks.reshaped(B * G, 1, N, D),
                values: vs.reshaped(B * G, 1, N, v.dim(3)), scale: scale,
                mask: .array(bias.reshaped(B * G, 1, 1, N)))
            return o.reshaped(B, H, 1, -1)
        }
        let index = expandedDimensions(source, axis: -1)
        let ks = takeAlong(expandedDimensions(k, axis: 2), index, axis: 3)
        let vs = takeAlong(expandedDimensions(v, axis: 2), index, axis: 3)
        let qr = q.reshaped(B, G, hg, L, D).transposed(0, 1, 3, 2, 4).reshaped(B * G * L, hg, 1, D)
        let o = MLXFast.scaledDotProductAttention(
            queries: qr, keys: ks.reshaped(B * G * L, 1, N, D), values: vs.reshaped(B * G * L, 1, N, v.dim(3)),
            scale: scale, mask: .array(bias.reshaped(B * G * L, 1, 1, N)))
        return o.reshaped(B, G, L, hg, -1).transposed(0, 1, 3, 2, 4).reshaped(B, H, L, -1)
    }
}
