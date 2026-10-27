// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// DeepSeek-V4.1 shared lightning indexer (port of jang_tools/dsv41/model.py `Indexer`,
// `select_candidate_blocks`, `topk_indices`, `Attention._compressed`).
//
// V4.1 differs from the DSV4-0731 indexer in three ways:
//   * index_k is SHARED: only kv-source layers ("owners", [2, 8, 14, 20]) project their compressor
//     latents through `wk` + `k_norm` (+ RoPE at the group positions) into an index-key pool; the
//     other index-source layers ([24, 28, 32, 36]) score against the most recent owner's pool.
//   * every index-source layer has its own `wq_b` / `weights_proj` and selects its own top-k
//     (512) compressed rows; the compressed layers between two index sources reuse the previous
//     source's selection.
//   * two-level selection: the candidate-source layer (20) keeps the top `candidate_topk_blocks`
//     blocks of `candidate_block_size` rows (plus each query's last visible block), and the later
//     index sources only choose among those candidate rows.
//
// Without this every V4.1 compressed layer attended ALL visible pool rows: identical to the
// reference only while <= 512 rows are visible, divergent beyond, and quadratic in prefill.
//
// Selection contract shared with the DSV4 attention paths: the returned top-k indices are unique
// row ids in [0, T); rows a query cannot see yet are NOT filtered here (the reference maps them to
// -1) because every attention path already applies the block-causal visibility
// `(idx + 1) * ratio <= qpos + 1`, which is the reference's `idx < lens` test. nil means "every
// visible row" (T <= top-k), which is what the reference's arange top-k reduces to.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Per-forward state the V4.1 attention layers hand down the stack (reference `Shared`).
final class DSV41CompressShare {
    /// Most recent kv-source layer's compressed-KV pool.
    var pool: DeepseekV4PoolView?
    /// Most recent owner layer's index-key pool.
    var indexK: DeepseekV4PoolView?
    /// Most recent index-source layer's selection per query sub-chunk (keyed by the sub-chunk's
    /// start position: attention runs in sub-chunks INSIDE a layer, and every layer uses the same
    /// partition). Value `[1, L, k]` int32, or nil = every visible row. Read through
    /// ``selection(offset:rows:)``.
    private(set) var selections: [Int: (rows: Int, value: MLXArray?)] = [:]
    /// Candidate-source layer's row mask `[L, T]` bool, per query sub-chunk.
    private(set) var candidates: [Int: (rows: Int, value: MLXArray)] = [:]
    /// dSpark verify capture for this forward (nil unless the sequence asked for one).
    var capture: DSV41VerifyCapture?

    func setSelection(_ topK: MLXArray?, offset: Int, rows: Int) {
        selections[offset] = (rows, topK)
    }

    func setCandidates(_ mask: MLXArray, offset: Int, rows: Int) {
        candidates[offset] = (rows, mask)
    }

    /// The selection for the queries at `offset ..< offset + rows`: the entry stored for exactly
    /// these rows, or (prompt-suffix pruning, where a late layer runs only the trailing rows of a
    /// sub-chunk) the trailing rows of the latest entry that ends at the same position. Selections
    /// are per query row, so a trailing slice is what the layer would have selected for those rows.
    /// `.none` when nothing covers the rows; `.some(nil)` = every visible row.
    func selection(offset: Int, rows: Int) -> MLXArray?? {
        guard let (key, entry) = Self.covering(selections, offset: offset, rows: rows) else {
            return .none
        }
        guard let value = entry.value, key != offset else { return .some(entry.value) }
        return .some(value[0..., (offset - key)...])
    }

    /// The candidate-row mask for the queries at `offset ..< offset + rows` (same rule as
    /// ``selection(offset:rows:)``).
    func candidates(offset: Int, rows: Int) -> MLXArray? {
        guard let (key, entry) = Self.covering(candidates, offset: offset, rows: rows) else {
            return nil
        }
        return key == offset ? entry.value : entry.value[(offset - key)...]
    }

    /// Exact key first (every unpruned forward). Otherwise the entry with the largest start before
    /// `offset` that ends at `offset + rows`: within one forward the later writer of a sub-chunk
    /// always starts at or after the earlier one (a deeper layer keeps fewer rows).
    private static func covering<V>(
        _ map: [Int: (rows: Int, value: V)], offset: Int, rows: Int
    ) -> (Int, (rows: Int, value: V))? {
        if let entry = map[offset], entry.rows == rows { return (offset, entry) }
        let end = offset + rows
        var best: Int? = nil
        for (key, entry) in map where key < offset && key + entry.rows == end {
            if best.map({ key > $0 }) ?? true { best = key }
        }
        guard let best, let entry = map[best] else { return nil }
        return (best, entry)
    }
}

/// Index keys produced without a cache (cache-less prefill), collected by the latent sink.
final class DSV41IndexKeyBox {
    var rows: [MLXArray] = []
}

/// Prefill segments longer than one query chunk evaluate each chunk's top-k (and candidate blocks) before the next
/// chunk is built (Python `INDEX_CHUNK_EVAL`, jangq #32: 1M prefill at 1024-token chunks, 786k-1M transient median
/// 8.4 -> 6.2 GiB, max 11.6 -> 8.3 GiB, ~5 % faster). Same arithmetic. `VMLX_DSV41_INDEX_CHUNK_EVAL=0` keeps the
/// asynchronous hand-off.
private let dsv41IndexChunkEval: Bool = {
    let raw = ProcessInfo.processInfo.environment["VMLX_DSV41_INDEX_CHUNK_EVAL"] ?? "1"
    return !["0", "false", "off", "no"].contains(raw.trimmingCharacters(in: .whitespaces).lowercased())
}()

/// Pure selection math (CPU-testable; no module state).
enum DeepseekV41IndexerMath {
    /// Query rows per scoring chunk (reference: `max(16, min(512, 2**26 // (h*T)))`).
    static func queryChunk(heads: Int, rows: Int) -> Int {
        max(16, min(512, (1 << 26) / max(1, heads * rows)))
    }

    /// Head-weighted ReLU scores `[C, T]` f32 for one query chunk and one key tile, with the
    /// reference's bf16 rounding points: `sum_h(max(q·k, 0) * w).bf16`.
    /// - q: `[C, h, d]` (roped), k: `[T, d]`, w: `[C, h]` (already scaled by d^-0.5 * h^-0.5).
    static func scores(q: MLXArray, k: MLXArray, w: MLXArray) -> MLXArray {
        let qb = q.asType(.bfloat16)
        let kb = k.asType(.bfloat16)
        let sc = maximum(matmul(qb, kb.transposed()), MLXArray(Float(0)).asType(.bfloat16))  // [C,h,T]
        let weighted = (sc * w.asType(.bfloat16).expandedDimensions(axis: -1)).asType(.float32)
        return weighted.sum(axis: 1).asType(.bfloat16).asType(.float32)  // [C,T]
    }

    /// `lens[c] = (startPos + c0 + c + 1) / ratio` — rows visible to each query.
    static func visibleLengths(startPos: Int, chunkStart: Int, count: Int, ratio: Int) -> MLXArray {
        let pos = MLXArray(Int32(startPos + chunkStart)..<Int32(startPos + chunkStart + count))
        return floorDivide(pos + Int32(1), Int32(ratio)).expandedDimensions(axis: -1)  // [C,1] int32
    }

    /// Mask rows at or beyond each query's visible length to -inf.
    static func maskInvisible(_ scores: MLXArray, lens: MLXArray) -> MLXArray {
        let T = scores.dim(-1)
        let cols = MLXArray(Int32(0)..<Int32(T)).expandedDimensions(axis: 0)
        return MLX.where(cols .>= lens, MLXArray(-Float.infinity), scores)
    }

    /// Reference `select_candidate_blocks`: `[C, T]` bool rows kept for the later index sources.
    static func candidateBlocks(
        _ logits: MLXArray, lens: MLXArray, topkBlocks: Int, blockSize: Int
    ) -> MLXArray {
        let C = logits.dim(0)
        let T = logits.dim(1)
        let pad = (blockSize - T % blockSize) % blockSize
        var sc = logits
        if pad > 0 {
            sc = concatenated(
                [sc, MLXArray.full([C, pad], values: MLXArray(-Float.infinity))], axis: -1)
        }
        sc = sc.reshaped(C, -1, blockSize).max(axis: -1)  // [C, nb]
        let nb = sc.dim(-1)
        let last = floorDivide(lens - Int32(1), Int32(blockSize))  // [C,1]
        let blockIds = MLXArray(Int32(0)..<Int32(nb)).expandedDimensions(axis: 0)
        sc = MLX.where(blockIds .== last, MLXArray(Float.infinity), sc)
        var keep: MLXArray
        if topkBlocks >= nb {
            keep = sc .> MLXArray(-Float.infinity)
        } else {
            let top = argPartition(-sc, kth: topkBlocks - 1, axis: -1)[0..., 0..<topkBlocks]
            let vals = takeAlong(sc, top, axis: -1)
            // put_along_axis(zeros, top, vals > -inf). A one-hot over blocks here was
            // [C, topkBlocks, nb] int32: ~4 GB per query chunk at 42k rows, and
            // 1 GB per decode token at 1M.
            keep = putAlong(
                MLXArray.zeros([C, nb], dtype: .bool), top,
                values: vals .> MLXArray(-Float.infinity), axis: -1)  // [C, nb]
        }
        let rows = repeated(keep.expandedDimensions(axis: -1), count: blockSize, axis: -1)
            .reshaped(C, nb * blockSize)
        return rows[0..., 0..<T]
    }

    /// Reference `topk_indices`: unique ids of the k largest (unordered); arange when k >= T.
    /// Prefill rows (more than 8 queries) at least this long take the block top-k: MLX's GPU
    /// `argPartition` sorts the whole row, which at long context dominates the indexer (python
    /// reference: 2048 x 100k 32.7 -> 13.5 ms, 256 x 1M 51.9 -> 3.2 ms). Decode and dSpark verify
    /// keep the direct partition (faster for one row; their scores are the bf16 path's).
    static let blockTopKMinRows = 65536
    static let blockTopKBlock = 64

    /// Indices of the k largest along the last axis (unordered). Long 2-D rows: the k largest
    /// values lie in the k blocks with the largest maxima, so select those blocks first and run
    /// the exact selection over their `k * blockTopKBlock` entries. Same values selected; only the
    /// pick among exactly tied values can differ. Indices past the row end (block padding) hold
    /// -inf; callers mask indices past the visible length.
    static func topK(_ scores: MLXArray, k: Int) -> MLXArray {
        let T = scores.dim(-1)
        if k >= T {
            return broadcast(MLXArray(Int32(0)..<Int32(T)), to: scores.shape)
        }
        let b = blockTopKBlock
        if scores.ndim == 2, scores.dim(0) > 8, T >= blockTopKMinRows, T / b > k {
            let C = scores.dim(0)
            let pad = (b - T % b) % b
            let s = pad > 0
                ? concatenated(
                    [scores, MLXArray.full([C, pad], values: MLXArray(-Float.infinity)).asType(scores.dtype)],
                    axis: -1)
                : scores
            let nb = s.dim(1) / b
            let blockMax = s.reshaped(C, nb, b).max(axis: -1)
            let topBlocks = argPartition(-blockMax, kth: k - 1, axis: -1)[0..., 0..<k]
            let cand = (topBlocks.expandedDimensions(axis: -1) * Int32(b)
                + MLXArray(Int32(0)..<Int32(b)).reshaped(1, 1, b)).reshaped(C, k * b)
            let sel = argPartition(-takeAlong(s, cand, axis: -1), kth: k - 1, axis: -1)[0..., 0..<k]
            return takeAlong(cand, sel, axis: -1).asType(.int32)
        }
        return argPartition(-scores, kth: k - 1, axis: -1)[.ellipsis, 0..<k].asType(.int32)
    }
}

final class DeepseekV41Indexer: Module {
    let nHeads: Int
    let headDim: Int
    let topK: Int
    let ratio: Int
    let ownsK: Bool
    let isCandidateSource: Bool
    let usesCandidates: Bool
    let candidateBlocks: Int
    let candidateBlockSize: Int

    @ModuleInfo(key: "wq_b") var wqB: Linear
    @ModuleInfo(key: "weights_proj") var weightsProj: Linear
    @ModuleInfo(key: "wk") var wk: Linear?
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm?

    init(config: DeepseekV4Configuration, layerIdx: Int, compressRatio: Int) {
        self.nHeads = config.indexNHeads
        self.headDim = config.indexHeadDim
        self.topK = config.indexTopk
        self.ratio = compressRatio
        self.ownsK = config.kvSourceLayerIds.contains(layerIdx)
        self.isCandidateSource = layerIdx == config.candidateSourceLayerId
        self.usesCandidates =
            config.candidateSourceLayerId >= 0 && config.candidateSourceLayerId < layerIdx
        self.candidateBlocks = config.candidateTopkBlocks
        self.candidateBlockSize = max(1, config.candidateBlockSize)
        self._wqB.wrappedValue = Linear(config.qLoraRank, nHeads * headDim, bias: false)
        self._weightsProj.wrappedValue = Linear(config.hiddenSize, nHeads, bias: false)
        if ownsK {
            self._wk.wrappedValue = Linear(config.headDim, headDim, bias: false)
            self._kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        }
    }

    /// Index keys for freshly completed compressor groups: `k_norm(wk(latent))`, roped at the
    /// groups' first-token positions `base + i * ratio` (reference `Indexer.__call__` owner branch).
    func indexKeys(latent: MLXArray, base: Int, rope: DeepseekV4RoPE) -> MLXArray {
        guard let wk, let kNorm else {
            preconditionFailure("DeepSeek-V4.1 indexer: non-owner layer asked for index keys")
        }
        var k = kNorm(wk(latent))
        let (cosT, sinT) = rope.cosSin(base: base, count: k.dim(1), stride: ratio)
        k = DeepseekV4Math.applyPartialRoPE(
            k, cos: cosT.expandedDimensions(axis: 0), sin: sinT.expandedDimensions(axis: 0),
            ropeDim: rope.dim)
        return k
    }

    /// Select compressed rows for `L` queries at `startPos ..< startPos + L`.
    /// - Returns: top-k ids `[1, L, k]` (nil when every visible row fits the budget) and, on the
    ///   candidate-source layer, the candidate row mask `[L, T]`.
    func select(
        x: MLXArray,
        qResidual: MLXArray,
        indexK: DeepseekV4PoolView,
        startPos: Int,
        rope: DeepseekV4RoPE,
        candidates: MLXArray?
    ) -> (topK: MLXArray?, candidates: MLXArray?) {
        let B = x.dim(0)
        let L = x.dim(1)
        precondition(B == 1, "DeepSeek-V4.1 indexer is batch-1 per sequence")
        let T = (startPos + L) / ratio
        let available = indexK.rowCount
        guard T > 0, available > 0 else { return (nil, nil) }
        let rows = min(T, available)
        // Every visible row fits the budget: the reference's top-k is arange(T) -> all rows.
        if rows <= topK { return (nil, nil) }

        var q = DeepseekV41FP8Rows.apply(wqB, qResidual).reshaped(B, L, nHeads, headDim).transposed(0, 2, 1, 3)
        let (cosT, sinT) = rope.cosSin(offset: startPos, length: L)
        q = DeepseekV4Math.applyPartialRoPE(
            q, cos: cosT.expandedDimensions(axes: [0, 1]), sin: sinT.expandedDimensions(axes: [0, 1]),
            ropeDim: rope.dim)
        let qs = q.transposed(0, 2, 1, 3)[0]  // [L, h, d]
        let w = (weightsProj(x) * MLXArray(
            pow(Float(headDim), -0.5) * pow(Float(nHeads), -0.5)))[0]  // [L, h]

        // Prefill (> 8 query rows, >= 32 keys, on the GPU): a fused kernel scores each query chunk in one
        // pass without the [C, h, T] intermediate; its chunk keeps [C, T] fp32 near 1 GB. Preferred: the
        // bf16 operands on the M5 matrix units (`DeepseekV41MPPIndexScores`, `VMLX_DSV41_INDEX_MPP=0` off);
        // then the fp32 simdgroup-MMA kernel on the same values upcast; then the stock path. Decode and
        // dSpark verify (<= 8 rows) keep the stock bf16 path below, unchanged.
        let path = DeepseekV41MPPIndexScores.prefillPath(
            queries: L, keys: rows, heads: nHeads, dim: headDim)
        let fused = path != .stock
        let chunk = fused
            ? DeepseekV41FusedIndexScores.queryChunk(queries: L, keys: rows)
            : DeepseekV41IndexerMath.queryChunk(heads: nHeads, rows: rows)
        let tileRows = max(1024, min(rows, (1 << 26) / max(1, nHeads * chunk)))
        // Fused operands, built once per call: the stock path's bf16 rounding of q / k / w (the reference
        // scores in bf16), passed as bf16 to the matrix units or held in fp32 for the fp32 kernel
        // (reference: `q.astype(f32)`, `index_k.astype(f32)`, `w.astype(f32)` of the bf16 values).
        let operandType: DType = path == .matrixUnits ? .bfloat16 : .float32
        var fusedQ: MLXArray? = nil
        var fusedK: MLXArray? = nil
        var fusedW: MLXArray? = nil
        if fused {
            fusedQ = qs.asType(.bfloat16).asType(operandType).expandedDimensions(axis: 0)  // [1, L, h, d]
            fusedW = w.asType(.bfloat16).asType(operandType).expandedDimensions(axis: 0)  // [1, L, h]
            var keyTiles: [MLXArray] = []
            indexK.forEachDequantizedTile(maxRows: rows) { start, tile in
                guard start < rows else { return }
                keyTiles.append(tile[0..., 0 ..< (min(rows, start + tile.dim(1)) - start)])
            }
            let keys = keyTiles.count == 1 ? keyTiles[0] : concatenated(keyTiles, axis: 1)
            fusedK = keys.asType(.bfloat16).asType(operandType)  // [1, rows, d]
        }
        var idxParts: [MLXArray] = []
        var candParts: [MLXArray] = []
        var c = 0
        while c < L {
            let e = min(L, c + chunk)
            let qi = qs[c ..< e]
            let wi = w[c ..< e]
            var fusedScores: MLXArray? = nil
            if let fusedQ, let fusedK, let fusedW {
                // Scale 1.0: `w` already carries d^-0.5 * h^-0.5.
                let qc = fusedQ[0..., c ..< e]
                let wc = fusedW[0..., c ..< e]
                if path == .matrixUnits {
                    fusedScores = DeepseekV41MPPIndexScores.poolIndexScores(
                        q: qc, poolKeys: fusedK, headW: wc, scale: 1.0)?[0]  // [C, rows]
                }
                if fusedScores == nil {
                    // fp32 kernel; bf16 operands upcast exactly (no-ops on the fp32 path).
                    fusedScores = DeepseekV41FusedIndexScores.poolIndexScores(
                        q: qc.asType(.float32), poolKeys: fusedK.asType(.float32),
                        headW: wc.asType(.float32), scale: 1.0)?[0]  // [C, rows]
                }
            }
            var tiles: [MLXArray] = []
            if fusedScores == nil {
                indexK.forEachDequantizedTile(maxRows: tileRows) { start, tile in
                    guard start < rows else { return }
                    let end = min(rows, start + tile.dim(1))
                    tiles.append(DeepseekV41IndexerMath.scores(
                        q: qi, k: tile[0, 0 ..< (end - start)], w: wi))
                }
            }
            var sc = fusedScores
                ?? (tiles.count == 1 ? tiles[0] : concatenated(tiles, axis: -1))  // [C, rows]
            let lens = DeepseekV41IndexerMath.visibleLengths(
                startPos: startPos, chunkStart: c, count: e - c, ratio: ratio)
            sc = DeepseekV41IndexerMath.maskInvisible(sc, lens: lens)
            if isCandidateSource {
                candParts.append(DeepseekV41IndexerMath.candidateBlocks(
                    sc, lens: lens, topkBlocks: candidateBlocks, blockSize: candidateBlockSize))
            } else if usesCandidates, let candidates, candidates.dim(-1) == rows {
                sc = MLX.where(candidates[c ..< e], sc, MLXArray(-Float.infinity))
            }
            idxParts.append(DeepseekV41IndexerMath.topK(sc, k: min(topK, rows)))
            if e < L {
                if dsv41IndexChunkEval {
                    eval([idxParts.last!] + (isCandidateSource ? [candParts.last!] : []))
                } else {
                    asyncEval(idxParts.last!)
                }
            }
            c = e
        }
        let idx = (idxParts.count == 1 ? idxParts[0] : concatenated(idxParts, axis: 0))
            .expandedDimensions(axis: 0)
        let cand: MLXArray? = isCandidateSource
            ? (candParts.count == 1 ? candParts[0] : concatenated(candParts, axis: 0)) : nil
        return (idx, cand)
    }
}
