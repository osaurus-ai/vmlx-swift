// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXLMCommon

// MiniMax-M3 MSA (block-sparse attention): the block selection and the attention paths that consume it.
//
// Port of the vMLX Python runtime's `vmlx_engine/models/minimax_m3/msa.py` (perf/minimax-m3-block-sparse-msa,
// 885355f0). The Metal sources below are that file's kernel sources; the routing thresholds and the env lever are
// the same.
//
// The Lightning indexer picks, for every query row and KV group, `keep` (16) key blocks of `block` (128) keys: the top
// block scores, the query's own (local) block forced in. Attention then sees only those blocks, causally.
//
// `MiniMaxM3MSASelection` carries that pick as block ids `[B, G, Sq, keep]` plus the positions it was made at. Paths
// (`choosePath` routes by the measured crossovers; `VMLX_MINIMAX_MSA` forces one for comparisons):
//
// - `dense`: the additive `[B, G, Sq, Sk]` float32 key mask (`denseBias`) and one SDPA over all Sk keys per KV group.
//   The reference path.
// - `decode`: decode-sized calls (decode steps, verify rows). A Metal kernel splits each (row, group)'s keep * block
//   selected keys into 32-key chunks across threadgroups and merges the partial softmaxes; keys are read in place
//   from the cache. Rows are independent: a row's output is the same alone or with other rows.
// - `kernel`: prefill chunks. One simdgroup per (row, group) walks the row's selected blocks with an online softmax,
//   on the matrix units (Metal 4 tensor ops) for bf16 / fp16, else on 8x8 simdgroup matrices. No mask, no copy of
//   K / V.
// - `gather`: MLX ops only (gather the selected keys, SDPA over them). Never routed to; an op-level reference.
//
// The selection itself (`blockScores`): the reference ops build a `[B, G, Sq, Sk]` float32 score sheet; the selection
// kernels (msa.py at 5fc31aca) compute the same block scores bit for bit without it (`selectArithmetic` admits the
// kernel whose dot-product arithmetic is the one MLX's float32 matmul runs in this process), so the top-k picks the
// same blocks in the same order. `VMLX_MINIMAX_MSA_SELECT=reference` (or `VMLX_MINIMAX_MSA=dense`) keeps the
// reference ops.
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
        case dense, gather, decode, kernel
    }

    /// `VMLX_MINIMAX_MSA`: `auto` (unset) routes by the measured crossovers; `dense` forces the mask path, `sparse` the
    /// block-sparse path for the call's size (decode kernel for decode-sized calls, chunk kernel for longer ones),
    /// `gather` / `decode` / `kernel` that path wherever its shapes are supported.
    enum Request: String, Sendable {
        case auto, dense, sparse, gather, decode, kernel
    }

    static let environmentVariable = "VMLX_MINIMAX_MSA"

    // Routing constants. They only pick a path; measured by the Python runtime on an M5 Max, bf16, M3.1 geometry (64
    // heads, 4 KV groups, head dim 128, blocks of 128, top-16), latency of dependent calls, medians of 7:
    // - one row (decode step / verify row): the decode kernel beat the dense mask path at every sparse length.
    // - 2 ..< 16 rows: up to 8 rows the decode kernel won everywhere; at 16 rows dense was 6 % faster at 2.5k keys and
    //   the decode kernel 2.9x faster at 8k.
    // - longer chunks: the chunk kernel's cost is flat in context, dense grows with it; dense won below ~5k keys.
    //   fp32 chunks (simdgroup kernel) lost to dense, so fp32 chunks stay dense; the fp32 decode kernel still won.
    static let decodeMaxRows = 16
    static let decodeAlwaysRows = 8
    static let decodeMinKeys = 4096
    static let kernelMinKeys = 5120

    /// Test / diagnostics hooks: a forced request that wins over the environment, and an observer of every routed
    /// pick (the Python tests' router spy).
    final class Hooks: @unchecked Sendable {
        private let lock = NSLock()
        private var forced: Request?
        private var observer: ((Path) -> Void)?
        private var forcedSelect: SelectRequest?
        private var selectPathObserver: ((SelectPath) -> Void)?

        var request: Request? {
            get { lock.withLock { forced } }
            set { lock.withLock { forced = newValue } }
        }

        var pathObserver: ((Path) -> Void)? {
            get { lock.withLock { observer } }
            set { lock.withLock { observer = newValue } }
        }

        /// A forced selection request (wins over `VMLX_MINIMAX_MSA_SELECT`).
        var selectRequest: SelectRequest? {
            get { lock.withLock { forcedSelect } }
            set { lock.withLock { forcedSelect = newValue } }
        }

        /// Observer of every routed selection pick (the Python tests' selection spy).
        var selectObserver: ((SelectPath) -> Void)? {
            get { lock.withLock { selectPathObserver } }
            set { lock.withLock { selectPathObserver = newValue } }
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

    /// The custom kernels run on the Metal GPU only (on a CPU default device every call takes the dense path).
    static var onGPU: Bool { Device.defaultDevice().deviceType == .gpu }

    /// `dense`, `gather`, `decode` or `kernel` for this call. Only picks a path; every path computes the same
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
        let want = requestedPath()
        switch want {
        case .dense: return .dense
        case .gather: return .gather
        case .decode:
            return decodeKernelSupported(queries: queries, keys: keys, selection: selection) ? .decode : .dense
        case .kernel:
            return kernelSupported(queries: queries, keys: keys, selection: selection) ? .kernel : .dense
        case .auto, .sparse: break
        }
        let rows = queries.dim(2)
        if rows <= decodeMaxRows {
            guard decodeKernelSupported(queries: queries, keys: keys, selection: selection) else { return .dense }
            if want == .sparse || rows <= decodeAlwaysRows || selection.keyLength >= decodeMinKeys {
                return .decode
            }
            return .dense
        }
        guard kernelSupported(queries: queries, keys: keys, selection: selection) else { return .dense }
        if want == .sparse { return .kernel }
        // Auto: the chunk kernel only on the matrix units (fp32 / no Metal 4: the simdgroup version measured slower).
        return selection.keyLength >= kernelMinKeys && matrixUnitsSupported(queries: queries, selection: selection)
            ? .kernel : .dense
    }

    // MARK: selection

    /// Block selection for queries `[B, G, Sq, d]` at positions `offset...` over index keys `[B, 1, Sk, d]`; nil when
    /// every block is visible (`keep >= blocks`, i.e. at most `topK * blockSize` keys: plain causal attention).
    ///
    /// Each indexer head is one KV group and keeps its own selection. The block scores (`blockScores`: the score-sheet
    /// ops or a selection kernel that reproduces them bit for bit) are float32; each block scores the max over its keys
    /// (future keys -inf, the last block padded with -inf); the query's own block and the `localBlocks - 1` before it
    /// are forced in; the top `topK` blocks per (row, group) are kept. One query row over its own key prefix is exactly
    /// a decode step's selection.
    static func select(
        queries: MLXArray, keys: MLXArray, offset: Int, blockSize: Int, topK: Int, localBlocks: Int
    ) -> MiniMaxM3MSASelection? {
        let keyLength = keys.dim(2)
        let blockCount = (keyLength + blockSize - 1) / blockSize
        let keep = min(topK, blockCount)
        if keep >= blockCount {
            return nil
        }
        let scores = blockScores(
            queries: queries, keys: keys, offset: offset, blockSize: blockSize, localBlocks: localBlocks)
        let top = argPartition(-scores, kth: keep - 1, axis: -1)[.ellipsis, ..<keep]
        return MiniMaxM3MSASelection(topIndices: top, offset: offset, keyLength: keyLength, blockSize: blockSize)
    }

    /// The indexer's block scores `[B, G, Sq, blocks]` float32 with the runtime's ops (the reference): queries at
    /// `offset...` over keys `0 ..< Sk`, through the `[B, G, Sq, Sk]` float32 score sheet (cast q / k to float32, one
    /// matmul, causal -inf, -inf padding to whole blocks, max per block, local block(s) +inf).
    static func referenceBlockScores(
        queries: MLXArray, keys: MLXArray, offset: Int, blockSize: Int, localBlocks: Int
    ) -> MLXArray {
        let (B, heads, queryLength) = (queries.dim(0), queries.dim(1), queries.dim(2))
        let keyLength = keys.dim(2)
        let blockCount = (keyLength + blockSize - 1) / blockSize
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
        return blockScores
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

    // MARK: kernel plumbing

    /// Metal element type of each activation dtype the kernels take.
    static func elementType(_ dtype: DType) -> String? {
        switch dtype {
        case .bfloat16: return "bfloat"
        case .float16: return "half"
        case .float32: return "float"
        default: return nil
        }
    }

    /// `using elem_t = ...;` plus one `constexpr int` line per constant, in order.
    static func constantsHeader(_ element: String, _ constants: [(String, Int)]) -> String {
        "  using elem_t = \(element);\n" + constants.map { "  constexpr int \($0.0) = \($0.1);\n" }.joined()
    }

    /// Kernel name carrying a hash of its full source: MLX caches compiled kernels by name.
    static func kernelName(_ prefix: String, _ source: String) -> String {
        var hash: UInt64 = 0xcbf2_9ce4_8422_2325
        for byte in source.utf8 {
            hash ^= UInt64(byte)
            hash = hash &* 0x0000_0100_0000_01b3
        }
        return prefix + "_" + String(hash, radix: 16)
    }

    private static let lock = NSLock()
    nonisolated(unsafe) private static var compiled: [String: MLXFast.MLXFastKernel] = [:]

    private static func cachedKernel(
        _ name: String, inputs: [String], outputs: [String], source: String, header: String = "",
        ensureRowContiguous: Bool
    ) -> MLXFast.MLXFastKernel {
        lock.lock()
        defer { lock.unlock() }
        if let kernel = compiled[name] { return kernel }
        let kernel = MLXFast.metalKernel(
            name: name, inputNames: inputs, outputNames: outputs, source: source, header: header,
            ensureRowContiguous: ensureRowContiguous)
        compiled[name] = kernel
        return kernel
    }

    // MARK: decode kernel

    // Few query rows (decode steps, verify rows): latency matters, so the keep * block selected keys of every
    // (row, KV group) are split into 32-key chunks, one threadgroup per chunk, and the group's query heads into slices
    // of HS heads, one simdgroup per slice. Lane j owns key j of the chunk: it scores that key against the slice's
    // heads (queries pre-scaled, in threadgroup memory), the simdgroup takes each head's max / sum over the chunk, then
    // lane j accumulates P V for head dims [j * DPL, (j + 1) * DPL). Partials (max, sum, output) per chunk and head go
    // to a second launch that merges a row's chunks (log-sum-exp). All fp32. Keys past the row's position are masked
    // and never read (so nothing at or past keyLength is read). Every threadgroup touches one row only: a row's result
    // does not depend on how many rows share the launch.
    static let decodePartialSource = #"""
          const uint lane = thread_index_in_simdgroup;
          const int sg = int(simdgroup_index_in_threadgroup);            // head slice
          const uint tid = thread_position_in_threadgroup.x;
          const int chunk = int(threadgroup_position_in_grid.x);         // 32-key chunk of the row's selection
          const int rg = int(threadgroup_position_in_grid.y);            // (b * G + g) * L + l
          const int G = int(blocks_shape[1]);
          const int L = int(blocks_shape[2]);
          const int H = int(q_shape[2]);
          const int l = rg % L;
          const int bgi = rg / L;
          const int b = bgi / G;
          const int g = bgi - b * G;
          const int q_pos = meta[0] + l;
          const float qscale = scale[0] * 1.4426950408889634f;
          const float NEG = -1.0e30f;
          constexpr int CPB = BLOCK / 32;                                // chunks per block

          const int start = int(blocks[(size_t)rg * KEEP + chunk / CPB]) * BLOCK + (chunk % CPB) * 32;
          const size_t pbase = ((size_t)rg * NCHUNK + chunk) * HG + sg * HS;
          if (start > q_pos) {                                           // whole chunk in the future: empty partial
            for (int h = 0; h < HS; ++h) {
              for (int e = 0; e < DPL; ++e) po[(pbase + h) * D + lane * DPL + e] = 0.0f;
              if (lane == 0) { pm[pbase + h] = NEG; pl[pbase + h] = 0.0f; }
            }
            return;
          }
          threadgroup float qs[HG * D];
          threadgroup float ps[HG * 32];
          device const elem_t* qrow = q + ((size_t)(b * L + l) * H + g * HG) * D;
          for (int i = int(tid); i < HG * D; i += (HG / HS) * 32) qs[i] = float(qrow[i]) * qscale;
          threadgroup_barrier(mem_flags::mem_threadgroup);

          device const elem_t* kb = k + (size_t)b * k_strides[0] + (size_t)g * k_strides[1];
          device const elem_t* vb = v + (size_t)b * v_strides[0] + (size_t)g * v_strides[1];
          const size_t kss = k_strides[2], vss = v_strides[2], kds = k_strides[3], vds = v_strides[3];
          threadgroup const float* qh = qs + sg * HS * D;
          threadgroup float* pp = ps + sg * HS * 32;

          const int kp = start + int(lane);
          const bool valid = kp <= q_pos;                                // q_pos < sk, so valid keys are inside the extent
          float s[HS];
          #pragma unroll
          for (int h = 0; h < HS; ++h) s[h] = 0.0f;
          // 4-wide loads when the head dim is contiguous and rows stay 4-element aligned (always true for a cache view).
          const bool vec_ok = kds == 1 && vds == 1 && (kss % 4) == 0 && (vss % 4) == 0
              && ((ulong)kb % (4 * sizeof(elem_t))) == 0 && ((ulong)vb % (4 * sizeof(elem_t))) == 0;
          if (valid) {
            device const elem_t* krow = kb + (size_t)kp * kss;
            if (vec_ok) {
              for (int d = 0; d < D; d += 4) {
                const vec<elem_t, 4> k4 = *((device const vec<elem_t, 4>*)(krow + d));
                #pragma unroll
                for (int e = 0; e < 4; ++e) {
                  const float kd = float(k4[e]);
                  #pragma unroll
                  for (int h = 0; h < HS; ++h) s[h] += qh[h * D + d + e] * kd;
                }
              }
            } else {
              for (int d = 0; d < D; ++d) {
                const float kd = float(krow[(size_t)d * kds]);
                #pragma unroll
                for (int h = 0; h < HS; ++h) s[h] += qh[h * D + d] * kd;
              }
            }
          }
          #pragma unroll
          for (int h = 0; h < HS; ++h) {
            const float x = valid ? s[h] : NEG;
            const float mh = simd_max(x);
            const float p = valid ? fast::exp2(x - mh) : 0.0f;
            const float lh = simd_sum(p);
            pp[h * 32 + lane] = p;
            if (lane == 0) {
              pm[pbase + h] = mh;
              pl[pbase + h] = lh;
            }
          }
          simdgroup_barrier(mem_flags::mem_threadgroup);
          float o[HS][DPL];
          #pragma unroll
          for (int h = 0; h < HS; ++h) {
            #pragma unroll
            for (int e = 0; e < DPL; ++e) o[h][e] = 0.0f;
          }
          const int nvalid = min(32, q_pos - start + 1);
          for (int j = 0; j < nvalid; ++j) {
            device const elem_t* vrow = vb + (size_t)(start + j) * vss + (size_t)(lane * DPL) * vds;
            float vv[DPL];
            if (vec_ok && DPL % 4 == 0) {
              #pragma unroll
              for (int e = 0; e < DPL; e += 4) {
                const vec<elem_t, 4> v4 = *((device const vec<elem_t, 4>*)(vrow + e));
                vv[e] = float(v4[0]); vv[e + 1] = float(v4[1]); vv[e + 2] = float(v4[2]); vv[e + 3] = float(v4[3]);
              }
            } else {
              #pragma unroll
              for (int e = 0; e < DPL; ++e) vv[e] = float(vrow[(size_t)e * vds]);
            }
            #pragma unroll
            for (int h = 0; h < HS; ++h) {
              const float p = pp[h * 32 + j];
              #pragma unroll
              for (int e = 0; e < DPL; ++e) o[h][e] += p * vv[e];
            }
          }
          #pragma unroll
          for (int h = 0; h < HS; ++h) {
            #pragma unroll
            for (int e = 0; e < DPL; ++e) po[(pbase + h) * D + lane * DPL + e] = o[h][e];
          }
        """#

    static let decodeCombineSource = #"""
          const uint lane = thread_index_in_simdgroup;
          const int h = int(threadgroup_position_in_grid.y);
          const int rg = int(threadgroup_position_in_grid.z);
          const int G = int(blocks_shape[1]);
          const int L = int(blocks_shape[2]);
          const int H = G * HG;
          const int l = rg % L;
          const int bgi = rg / L;
          const int b = bgi / G;
          const int g = bgi - b * G;
          float mmax = -1.0e30f;
          for (int c = 0; c < NCHUNK; ++c) mmax = max(mmax, pm[((size_t)rg * NCHUNK + c) * HG + h]);
          float lsum = 0.0f;
          float o[DPL];
          for (int e = 0; e < DPL; ++e) o[e] = 0.0f;
          for (int c = 0; c < NCHUNK; ++c) {
            const size_t pidx = ((size_t)rg * NCHUNK + c) * HG + h;
            const float w = fast::exp2(pm[pidx] - mmax);
            lsum += w * pl[pidx];
            for (int e = 0; e < DPL; ++e) o[e] += w * po[pidx * D + lane * DPL + e];
          }
          const float inv = 1.0f / lsum;
          device elem_t* orow = out + ((size_t)(b * L + l) * H + g * HG + h) * D + lane * DPL;
          for (int e = 0; e < DPL; ++e) orow[e] = elem_t(o[e] * inv);
        """#

    struct DecodeKernels {
        let partial: MLXFast.MLXFastKernel
        let combine: MLXFast.MLXFastKernel
        let chunks: Int
        let slices: Int
    }

    /// Query heads per simdgroup in the decode kernel (4, or all of them when the group has fewer / not a multiple).
    static func decodeHeadSlice(_ hg: Int) -> Int { hg % 4 == 0 ? 4 : hg }

    static func decodeKernels(
        element: String, hg: Int, d: Int, keep: Int, block: Int,
        partialSource: String = decodePartialSource, combineSource: String = decodeCombineSource
    ) -> DecodeKernels {
        let hs = decodeHeadSlice(hg)
        let chunks = keep * (block / 32)
        let head = constantsHeader(
            element,
            [("HG", hg), ("HS", hs), ("D", d), ("DPL", d / 32), ("KEEP", keep), ("BLOCK", block), ("NCHUNK", chunks)])
        let tag = "\(element)_h\(hg)_d\(d)_k\(keep)_b\(block)"
        let partial = cachedKernel(
            kernelName("vmlx_minimax_msa_decode_partial_\(tag)", head + partialSource),
            inputs: ["q", "k", "v", "blocks", "meta", "scale"], outputs: ["po", "pm", "pl"],
            source: head + partialSource, ensureRowContiguous: false)
        let combine = cachedKernel(
            kernelName("vmlx_minimax_msa_decode_combine_\(tag)", head + combineSource),
            inputs: ["po", "pm", "pl", "blocks"], outputs: ["out"], source: head + combineSource,
            ensureRowContiguous: true)
        return DecodeKernels(partial: partial, combine: combine, chunks: chunks, slices: hg / hs)
    }

    /// Head dim a multiple of 32 (lane j owns D / 32 output dims), at most 32 query heads per KV group, blocks a
    /// multiple of the 32-key chunk, and the group's queries plus the chunk probabilities in threadgroup memory.
    static func decodeKernelSupported(queries q: MLXArray, keys k: MLXArray, selection: MiniMaxM3MSASelection) -> Bool
    {
        let (H, D) = (q.dim(1), q.dim(3))
        let G = selection.groups
        guard onGPU, elementType(q.dtype) != nil, k.dtype == q.dtype, H % G == 0, D % 32 == 0,
            selection.blockSize % 32 == 0, selection.blockSize <= 1024
        else { return false }
        let hg = H / G
        return hg <= 32 && 4 * (hg * D + hg * 32) <= 32768
    }

    /// Attention `[B, H, L, D]` over each (row, KV group)'s selected blocks for decode-sized calls (split keys +
    /// merge). Rows may come in any block order (the merge does not depend on it), so the selection is used as picked.
    static func decodeKernelAttention(
        queries q: MLXArray, keys k: MLXArray, values v: MLXArray, selection: MiniMaxM3MSASelection, scale: Float,
        kernels override: DecodeKernels? = nil
    ) -> MLXArray {
        let (B, H, L, D) = (q.dim(0), q.dim(1), q.dim(2), q.dim(3))
        let G = selection.groups
        let hg = H / G
        checkExtent(k, v, selection)
        // argPartition's [..., :keep] is a strided view; the kernel indexes it densely.
        let blocks = contiguous(selection.topIndices)
        let qc = contiguous(q.transposed(0, 2, 1, 3))  // [B, L, H, D]
        let meta = MLXArray([Int32(selection.offset), Int32(selection.keyLength)])
        let kernels =
            override
            ?? decodeKernels(
                element: elementType(q.dtype)!, hg: hg, d: D, keep: selection.keep, block: selection.blockSize)
        let rows = B * G * L
        let partials = kernels.partial(
            [qc, k, v, blocks, meta, MLXArray([scale])],
            grid: (32 * kernels.slices * kernels.chunks, rows, 1), threadGroup: (32 * kernels.slices, 1, 1),
            outputShapes: [[rows, kernels.chunks, hg, D], [rows, kernels.chunks, hg], [rows, kernels.chunks, hg]],
            outputDTypes: [.float32, .float32, .float32])
        let out = kernels.combine(
            [partials[0], partials[1], partials[2], blocks],
            grid: (32, hg, rows), threadGroup: (32, 1, 1),
            outputShapes: [[B, L, H, D]], outputDTypes: [q.dtype])[0]
        return out.transposed(0, 2, 1, 3)
    }

    // MARK: chunk kernel (simdgroup matrices)

    // One threadgroup per (batch, KV group, query row): the HG query heads of the group (simdgroups of 8 heads) attend
    // that row's selected blocks, read in place from K / V (any sequence / group / batch stride; the head dim must be
    // contiguous for the vector loads). Online softmax in fp32 over 32-key tiles (16 for fp32) staged in threadgroup
    // memory; S = Q K^T and O += P V on 8x8 simdgroup matrices with fp32 accumulators. Blocks are visited in
    // ascending order; tiles that start past the row's position are skipped (never read), keys past the row's position
    // are masked. Ints that vary per call (offset, keyLength) ride an input buffer, so one compiled kernel serves every
    // chunk.
    static let prefillSource = #"""
          const uint lane = thread_index_in_simdgroup;
          const uint sg = simdgroup_index_in_threadgroup;
          const uint tid = thread_position_in_threadgroup.x;
          const int l = int(threadgroup_position_in_grid.y);
          const int bg = int(threadgroup_position_in_grid.z);
          const int G = int(blocks_shape[1]);
          const int L = int(blocks_shape[2]);
          const int H = int(q_shape[2]);
          const int b = bg / G;
          const int g = bg - b * G;
          const int offset = meta[0];
          const int sk = meta[1];
          const int q_pos = offset + l;
          const float qscale = scale[0] * 1.4426950408889634f;   // softmax in base 2
          const float NEG = -1.0e30f;

          threadgroup elem_t ks[TK * KPAD];
          threadgroup elem_t vs[TK * KPAD];

          // Lane layout of an 8x8 simdgroup matrix: this lane holds (row fm, cols fn and fn + 1).
          const short qid = short(lane / 4);
          const short fm = (qid & 4) + short((lane / 2) % 4);
          const short fn = (qid & 2) * 2 + short(lane % 2) * 2;

          const int head = g * HG + int(sg) * 8 + fm;
          device const elem_t* qrow = q + ((size_t)(b * L + l) * H + head) * D;
          simdgroup_matrix<float, 8, 8> Qf[D / 8];
          simdgroup_matrix<float, 8, 8> Of[D / 8];
          for (int d = 0; d < D / 8; ++d) {
            Qf[d].thread_elements()[0] = float(qrow[d * 8 + fn]) * qscale;
            Qf[d].thread_elements()[1] = float(qrow[d * 8 + fn + 1]) * qscale;
            Of[d] = simdgroup_matrix<float, 8, 8>(0.0f);
          }
          float m_run = NEG;
          float l_run = 0.0f;

          device const int* blk = blocks + ((size_t)(b * G + g) * L + l) * KEEP;
          device const elem_t* kbase = k + (size_t)b * k_strides[0] + (size_t)g * k_strides[1];
          device const elem_t* vbase = v + (size_t)b * v_strides[0] + (size_t)g * v_strides[1];
          const size_t kss = k_strides[2];
          const size_t vss = v_strides[2];
          constexpr int CE = 16 / sizeof(elem_t);                 // elements per 16-byte chunk
          const bool vec_ok = k_strides[3] == 1 && v_strides[3] == 1 && (kss % CE) == 0 && (vss % CE) == 0
              && ((ulong)kbase % 16) == 0 && ((ulong)vbase % 16) == 0;

          for (int i = 0; i < KEEP; ++i) {
            const int start = blk[i] * BLOCK;
            if (start > q_pos) break;                               // ascending: every later block is in the future too
            for (int t0 = 0; t0 < BLOCK; t0 += TK) {
              const int ts = start + t0;
              if (ts > q_pos) break;
              threadgroup_barrier(mem_flags::mem_threadgroup);
              if (vec_ok) {
                for (int c = int(tid); c < TK * (D / CE); c += NT) {
                  const int r = c / (D / CE);
                  const int cc = c - r * (D / CE);
                  const int kp = ts + r;
                  uint4 kv = uint4(0), vv = uint4(0);
                  if (kp < sk) {
                    kv = *((device const uint4*)(kbase + (size_t)kp * kss) + cc);
                    vv = *((device const uint4*)(vbase + (size_t)kp * vss) + cc);
                  }
                  *((threadgroup uint4*)(ks + r * KPAD) + cc) = kv;
                  *((threadgroup uint4*)(vs + r * KPAD) + cc) = vv;
                }
              } else {
                for (int c = int(tid); c < TK * D; c += NT) {
                  const int r = c / D;
                  const int dd = c - r * D;
                  const int kp = ts + r;
                  ks[r * KPAD + dd] = kp < sk ? kbase[(size_t)kp * kss + (size_t)dd * k_strides[3]] : elem_t(0);
                  vs[r * KPAD + dd] = kp < sk ? vbase[(size_t)kp * vss + (size_t)dd * v_strides[3]] : elem_t(0);
                }
              }
              threadgroup_barrier(mem_flags::mem_threadgroup);

              simdgroup_matrix<float, 8, 8> Sf[TK / 8];
              for (int f = 0; f < TK / 8; ++f) {
                Sf[f] = simdgroup_matrix<float, 8, 8>(0.0f);
                for (int d = 0; d < D / 8; ++d) {
                  simdgroup_matrix<float, 8, 8> Kt;
                  Kt.thread_elements()[0] = float(ks[(f * 8 + fn) * KPAD + d * 8 + fm]);
                  Kt.thread_elements()[1] = float(ks[(f * 8 + fn + 1) * KPAD + d * 8 + fm]);
                  simdgroup_multiply_accumulate(Sf[f], Qf[d], Kt, Sf[f]);
                }
              }
              float tmax = NEG;
              for (int f = 0; f < TK / 8; ++f) {
                for (int e = 0; e < 2; ++e) {
                  const int kp = ts + f * 8 + fn + e;
                  float s = Sf[f].thread_elements()[e];
                  s = kp <= q_pos ? s : NEG;
                  Sf[f].thread_elements()[e] = s;
                  tmax = max(tmax, s);
                }
              }
              tmax = max(tmax, simd_shuffle_xor(tmax, 1));
              tmax = max(tmax, simd_shuffle_xor(tmax, 8));
              const float m_new = max(m_run, tmax);
              const float factor = fast::exp2(m_run - m_new);
              float psum = 0.0f;
              for (int f = 0; f < TK / 8; ++f) {
                for (int e = 0; e < 2; ++e) {
                  const float p = fast::exp2(Sf[f].thread_elements()[e] - m_new);
                  Sf[f].thread_elements()[e] = p;
                  psum += p;
                }
              }
              psum += simd_shuffle_xor(psum, 1);
              psum += simd_shuffle_xor(psum, 8);
              l_run = l_run * factor + psum;
              m_run = m_new;
              for (int d = 0; d < D / 8; ++d) {
                Of[d].thread_elements()[0] *= factor;
                Of[d].thread_elements()[1] *= factor;
              }
              for (int f = 0; f < TK / 8; ++f) {
                for (int d = 0; d < D / 8; ++d) {
                  simdgroup_matrix<float, 8, 8> Vf;
                  Vf.thread_elements()[0] = float(vs[(f * 8 + fm) * KPAD + d * 8 + fn]);
                  Vf.thread_elements()[1] = float(vs[(f * 8 + fm) * KPAD + d * 8 + fn + 1]);
                  simdgroup_multiply_accumulate(Of[d], Sf[f], Vf, Of[d]);
                }
              }
            }
          }
          const float inv = 1.0f / l_run;
          device elem_t* orow = out + ((size_t)(b * L + l) * H + head) * D;
          for (int d = 0; d < D / 8; ++d) {
            orow[d * 8 + fn] = elem_t(Of[d].thread_elements()[0] * inv);
            orow[d * 8 + fn + 1] = elem_t(Of[d].thread_elements()[1] * inv);
          }
        """#

    /// Keys per staged K / V tile: two [TK][D + 8] tiles fit the 32 KB of threadgroup memory.
    static func prefillTile(_ element: String) -> Int { element == "float" ? 16 : 32 }

    static func prefillKernel(
        element: String, hg: Int, d: Int, keep: Int, block: Int, source body: String = prefillSource
    ) -> MLXFast.MLXFastKernel {
        let source =
            constantsHeader(
                element,
                [
                    ("HG", hg), ("D", d), ("KEEP", keep), ("BLOCK", block), ("TK", prefillTile(element)),
                    ("KPAD", d + 8), ("NT", 32 * (hg / 8)),
                ]) + body
        return cachedKernel(
            kernelName("vmlx_minimax_msa_prefill_\(element)_h\(hg)_d\(d)_k\(keep)_b\(block)", source),
            inputs: ["q", "k", "v", "blocks", "meta", "scale"], outputs: ["out"], source: source,
            ensureRowContiguous: false)
    }

    // MARK: chunk kernel (matrix units)

    // Same contract as the kernel above, on the M5 matrix units (Metal 4 tensor ops, per-simdgroup 16x32x16 matmul2d
    // on register fragments). One simdgroup owns one (batch, KV group, query row): the group's query heads are the 16
    // rows of every fragment (fewer heads leave the remaining rows zero), so S = Q K^T is [16 heads x 32 keys] per step
    // and O is [16 x D] in registers. Fragment layout of a 16x16 tile: this lane holds rows fm and fm + 8, columns
    // fn .. fn + 3. K / V are read straight from device memory (no staging); NSG consecutive query rows share a
    // threadgroup so blocks the rows have in common are read while still in cache. Q K^T multiplies the stored
    // (bf16 / fp16) values exactly with fp32 accumulation; P (fp32) times V runs at the op's relaxed precision.
    static let matrixUnitHeader = #"""
        #include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

        template <typename AT, typename BT, bool TB>
        METAL_FUNC void msa_mma_16x32(thread float* c, thread const AT* a, thread const BT* b0, thread const BT* b1) {
          constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
              16, 32, 16, false, TB, true, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
          mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
          auto ca = op.template get_left_input_cooperative_tensor<AT, BT, float>();
          auto cb = op.template get_right_input_cooperative_tensor<AT, BT, float>();
          auto cc = op.template get_destination_cooperative_tensor<
              metal::remove_addrspace_t<decltype(ca)>, metal::remove_addrspace_t<decltype(cb)>, float>();
          #pragma unroll
          for (short i = 0; i < 8; ++i) { ca[i] = a[i]; cb[i] = b0[i]; cb[8 + i] = b1[i]; }
          #pragma unroll
          for (short i = 0; i < 16; ++i) cc[i] = c[i];
          op.run(ca, cb, cc);
          #pragma unroll
          for (short i = 0; i < 16; ++i) c[i] = cc[i];
        }

        // dst[0..3] = row[col .. col + 3] (head-dim stride ds); zeros when the row is past the logical extent.
        template <typename T>
        METAL_FUNC void msa_load4(thread T* dst, device const T* row, int col, size_t ds, bool vec, bool valid) {
          if (!valid) {
            dst[0] = T(0); dst[1] = T(0); dst[2] = T(0); dst[3] = T(0);
          } else if (vec) {
            const metal::vec<T, 4> x = *((device const metal::vec<T, 4>*)(row + col));
            dst[0] = x[0]; dst[1] = x[1]; dst[2] = x[2]; dst[3] = x[3];
          } else {
            for (short j = 0; j < 4; ++j) dst[j] = row[(size_t)(col + j) * ds];
          }
        }
        """#

    static let matrixUnitSource = #"""
          const ushort lane = thread_index_in_simdgroup;
          const int sg = int(simdgroup_index_in_threadgroup);
          const int L = int(blocks_shape[2]);
          const int l = int(threadgroup_position_in_grid.y) * NSG + sg;
          if (l >= L) return;                                      // per-simdgroup ops only: no threadgroup barrier below
          const int G = int(blocks_shape[1]);
          const int H = int(q_shape[2]);
          const int bg = int(threadgroup_position_in_grid.z);
          const int b = bg / G;
          const int g = bg - b * G;
          const int offset = meta[0];
          const int sk = meta[1];
          const int q_pos = offset + l;
          const float qscale = scale[0] * 1.4426950408889634f;
          const float NEG = -1.0e30f;

          const short qid = short(lane >> 2);
          const short fm = (qid & 4) | short((lane >> 1) & 3);
          const short fn = short(((qid & 2) | (lane & 1)) * 4);

          // Q fragments: rows = heads of the group, columns = 16-wide head-dim chunks.
          elem_t qf[D / 16][8];
          device const elem_t* qrow = q + ((size_t)(b * L + l) * H + g * HG) * D;
          for (int c = 0; c < D / 16; ++c) {
            for (short i = 0; i < 2; ++i) {
              const short r = fm + 8 * i;
              for (short j = 0; j < 4; ++j) {
                qf[c][i * 4 + j] = r < HG ? qrow[(size_t)r * D + c * 16 + fn + j] : elem_t(0);
              }
            }
          }
          float o[D / 32][16];
          for (int dp = 0; dp < D / 32; ++dp) {
            for (short e = 0; e < 16; ++e) o[dp][e] = 0.0f;
          }
          float m_run[2] = {NEG, NEG};
          float l_run[2] = {0.0f, 0.0f};

          device const int* blk = blocks + ((size_t)(b * G + g) * L + l) * KEEP;
          device const elem_t* kbase = k + (size_t)b * k_strides[0] + (size_t)g * k_strides[1];
          device const elem_t* vbase = v + (size_t)b * v_strides[0] + (size_t)g * v_strides[1];
          const size_t kss = k_strides[2];
          const size_t vss = v_strides[2];
          const size_t kds = k_strides[3];
          const size_t vds = v_strides[3];
          // 8-byte vector loads when the head dim is contiguous and every 4-element chunk is 8-byte aligned.
          const bool vec_ok = kds == 1 && vds == 1 && (kss % 4) == 0 && (vss % 4) == 0
              && ((ulong)kbase % 8) == 0 && ((ulong)vbase % 8) == 0;

          for (int bi = 0; bi < KEEP; ++bi) {
            const int start = blk[bi] * BLOCK;
            if (start > q_pos) break;                              // ascending: every later block is in the future too
            for (int t0 = 0; t0 < BLOCK; t0 += 32) {
              const int ts = start + t0;
              if (ts > q_pos) break;
              float s[16];
              for (short e = 0; e < 16; ++e) s[e] = 0.0f;
              for (int c = 0; c < D / 16; ++c) {
                elem_t kb0[8], kb1[8];
                for (short i = 0; i < 2; ++i) {
                  const int kp0 = ts + fm + 8 * i;
                  const int kp1 = kp0 + 16;
                  msa_load4(kb0 + i * 4, kbase + (size_t)kp0 * kss, c * 16 + fn, kds, vec_ok, kp0 < sk);
                  msa_load4(kb1 + i * 4, kbase + (size_t)kp1 * kss, c * 16 + fn, kds, vec_ok, kp1 < sk);
                }
                msa_mma_16x32<elem_t, elem_t, true>(s, qf[c], kb0, kb1);
              }
              // s[n * 8 + i * 4 + j] = S[head fm + 8i][key ts + n * 16 + fn + j]
              float rmax[2] = {NEG, NEG};
              for (short n = 0; n < 2; ++n) {
                for (short i = 0; i < 2; ++i) {
                  for (short j = 0; j < 4; ++j) {
                    const int kp = ts + n * 16 + fn + j;
                    const short e = n * 8 + i * 4 + j;
                    const float x = kp <= q_pos ? s[e] * qscale : NEG;
                    s[e] = x;
                    rmax[i] = max(rmax[i], x);
                  }
                }
              }
              float factor[2];
              for (short i = 0; i < 2; ++i) {
                rmax[i] = max(rmax[i], simd_shuffle_xor(rmax[i], ushort(1)));
                rmax[i] = max(rmax[i], simd_shuffle_xor(rmax[i], ushort(8)));
                const float m_new = max(m_run[i], rmax[i]);
                factor[i] = fast::exp2(m_run[i] - m_new);
                m_run[i] = m_new;
              }
              float rsum[2] = {0.0f, 0.0f};
              for (short n = 0; n < 2; ++n) {
                for (short i = 0; i < 2; ++i) {
                  for (short j = 0; j < 4; ++j) {
                    const short e = n * 8 + i * 4 + j;
                    const float p = fast::exp2(s[e] - m_run[i]);
                    s[e] = p;
                    rsum[i] += p;
                  }
                }
              }
              for (short i = 0; i < 2; ++i) {
                rsum[i] += simd_shuffle_xor(rsum[i], ushort(1));
                rsum[i] += simd_shuffle_xor(rsum[i], ushort(8));
                l_run[i] = l_run[i] * factor[i] + rsum[i];
              }
              for (int dp = 0; dp < D / 32; ++dp) {
                for (short h = 0; h < 2; ++h) {
                  for (short i = 0; i < 2; ++i) {
                    for (short j = 0; j < 4; ++j) o[dp][h * 8 + i * 4 + j] *= factor[i];
                  }
                }
              }
              for (short n = 0; n < 2; ++n) {
                for (int dp = 0; dp < D / 32; ++dp) {
                  elem_t vb0[8], vb1[8];
                  for (short i = 0; i < 2; ++i) {
                    const int kp = ts + n * 16 + fm + 8 * i;
                    msa_load4(vb0 + i * 4, vbase + (size_t)kp * vss, dp * 32 + fn, vds, vec_ok, kp < sk);
                    msa_load4(vb1 + i * 4, vbase + (size_t)kp * vss, dp * 32 + 16 + fn, vds, vec_ok, kp < sk);
                  }
                  msa_mma_16x32<float, elem_t, false>(o[dp], s + n * 8, vb0, vb1);
                }
              }
            }
          }
          device elem_t* orow = out + ((size_t)(b * L + l) * H + g * HG) * D;
          for (short i = 0; i < 2; ++i) {
            const short r = fm + 8 * i;
            if (r >= HG) continue;
            const float inv = 1.0f / l_run[i];
            for (int dp = 0; dp < D / 32; ++dp) {
              for (short h = 0; h < 2; ++h) {
                for (short j = 0; j < 4; ++j) {
                  orow[(size_t)r * D + dp * 32 + h * 16 + fn + j] = elem_t(o[dp][h * 8 + i * 4 + j] * inv);
                }
              }
            }
          }
        """#

    /// Query rows per threadgroup of the matrix-unit kernel (one simdgroup per row).
    static let matrixUnitSimdgroups = 4

    static func matrixUnitKernel(
        element: String, hg: Int, d: Int, keep: Int, block: Int, source body: String = matrixUnitSource
    ) -> MLXFast.MLXFastKernel {
        let source =
            constantsHeader(
                element, [("HG", hg), ("D", d), ("KEEP", keep), ("BLOCK", block), ("NSG", matrixUnitSimdgroups)])
            + body
        return cachedKernel(
            kernelName(
                "vmlx_minimax_msa_prefill_nax_\(element)_h\(hg)_d\(d)_k\(keep)_b\(block)",
                matrixUnitHeader + source),
            inputs: ["q", "k", "v", "blocks", "meta", "scale"], outputs: ["out"], source: source,
            header: matrixUnitHeader, ensureRowContiguous: false)
    }

    static func matrixUnitLaunch(
        queries q: MLXArray, keys k: MLXArray, values v: MLXArray, selection: MiniMaxM3MSASelection, scale: Float,
        kernel override: MLXFast.MLXFastKernel? = nil
    ) -> MLXArray {
        let (B, H, L, D) = (q.dim(0), q.dim(1), q.dim(2), q.dim(3))
        let G = selection.groups
        let blocks = sorted(selection.topIndices.asType(.int32), axis: -1)
        let qc = contiguous(q.transposed(0, 2, 1, 3))  // [B, L, H, D]
        let meta = MLXArray([Int32(selection.offset), Int32(selection.keyLength)])
        let kernel =
            override
            ?? matrixUnitKernel(
                element: elementType(q.dtype)!, hg: H / G, d: D, keep: selection.keep, block: selection.blockSize)
        let nsg = matrixUnitSimdgroups
        let out = kernel(
            [qc, k, v, blocks, meta, MLXArray([scale])],
            grid: (32 * nsg, (L + nsg - 1) / nsg, B * G), threadGroup: (32 * nsg, 1, 1),
            outputShapes: [[B, L, H, D]], outputDTypes: [q.dtype])[0]
        return out.transposed(0, 2, 1, 3)
    }

    nonisolated(unsafe) private static var matrixUnitsChecked: Bool?

    /// Metal 4 tensor ops compile here and the matrix-unit kernel matches the fp32 simdgroup kernel on a small case
    /// with a partial last block and causal rows inside it (checked once per process, on the GPU). The case is the
    /// Python runtime's, from the same MLX random keys.
    static func matrixUnitsAvailable() -> Bool {
        lock.lock()
        if let checked = matrixUnitsChecked {
            lock.unlock()
            return checked
        }
        lock.unlock()
        let ok = matrixUnitSelfCheck() < 2e-2
        lock.lock()
        matrixUnitsChecked = ok
        lock.unlock()
        return ok
    }

    /// The self-check's relative error (max |matrix units - fp32 kernel| / max |fp32 kernel|); infinity when the
    /// kernel does not compile or run here.
    static func matrixUnitSelfCheck(kernel override: MLXFast.MLXFastKernel? = nil) -> Float {
        do {
            return try withError { error in
                try Device.withDefaultDevice(.gpu) {
                    let keys = MLXRandom.split(key: MLXRandom.key(11), into: 4)
                    let (G, HG, D, sk, L, block, keep) = (2, 16, 64, 300, 40, 32, 4)
                    let q = MLXRandom.normal([1, G * HG, L, D], key: keys[0]).asType(.bfloat16)
                    let kk = MLXRandom.normal([1, G, sk, D], key: keys[1]).asType(.bfloat16)
                    let vv = MLXRandom.normal([1, G, sk, D], key: keys[2]).asType(.bfloat16)
                    let blockCount = (sk + block - 1) / block
                    var pick = argSort(
                        MLXRandom.uniform(0 ..< 1, [1, G, L, blockCount], key: keys[3]), axis: -1
                    )[.ellipsis, ..<(keep - 1)]
                    let local = broadcast(
                        floorDivide(
                            arange(sk - L, sk).reshaped(1, 1, L, 1), MLXArray(Int32(block))),
                        to: [1, G, L, 1])
                    pick = MLX.where(
                        pick .== local, remainder(local + MLXArray(Int32(1)), MLXArray(Int32(blockCount))), pick)
                    let selection = MiniMaxM3MSASelection(
                        topIndices: concatenated([local, pick], axis: -1).asType(.uint32), offset: sk - L,
                        keyLength: sk, blockSize: block)
                    let scale = pow(Float(D), -0.5)
                    let got = matrixUnitLaunch(
                        queries: q, keys: kk, values: vv, selection: selection, scale: scale, kernel: override
                    ).asType(.float32)
                    let reference = kernelAttention(
                        queries: q.asType(.float32), keys: kk.asType(.float32), values: vv.asType(.float32),
                        selection: selection, scale: scale, matrixUnits: false)
                    let err = (abs(got - reference).max() / abs(reference).max()).item(Float.self)
                    try error.check()
                    return err.isFinite ? err : .infinity
                }
            }
        } catch {
            return .infinity
        }
    }

    /// Shapes the simdgroup chunk kernel is written for: head dim a multiple of 8, whole simdgroups of 8 query heads
    /// per KV group (at most 32), blocks a multiple of the staged tile, two tiles in threadgroup memory.
    static func kernelSupported(queries q: MLXArray, keys k: MLXArray, selection: MiniMaxM3MSASelection) -> Bool {
        let (H, D) = (q.dim(1), q.dim(3))
        let G = selection.groups
        guard onGPU, let element = elementType(q.dtype), k.dtype == q.dtype, H % G == 0 else { return false }
        let hg = H / G
        let tile = prefillTile(element)
        return D % 8 == 0 && 2 * tile * (D + 8) * q.dtype.size <= 32768 && hg % 8 == 0 && hg <= 32
            && selection.blockSize % tile == 0
    }

    /// The matrix-unit kernel's shapes: bf16 / fp16, at most 16 query heads per KV group (one fragment of rows), head
    /// dim a multiple of 32 (at most 256), blocks a multiple of 32 keys, Metal 4 tensor ops present and self-checked.
    static func matrixUnitsSupported(queries q: MLXArray, selection: MiniMaxM3MSASelection) -> Bool {
        let (H, D) = (q.dim(1), q.dim(3))
        let G = selection.groups
        return (q.dtype == .bfloat16 || q.dtype == .float16) && H % G == 0 && H / G <= 16 && D % 32 == 0
            && D <= 256 && selection.blockSize % 32 == 0 && matrixUnitsAvailable()
    }

    /// Block-sparse attention `[B, H, L, D]` for an L-row chunk, reading only the selected blocks (no key mask).
    /// bf16 / fp16 run on the matrix units when admitted (`matrixUnitsSupported`), everything else (fp32 included) on
    /// the simdgroup kernel. `matrixUnits` forces one or the other.
    static func kernelAttention(
        queries q: MLXArray, keys k: MLXArray, values v: MLXArray, selection: MiniMaxM3MSASelection, scale: Float,
        matrixUnits: Bool? = nil
    ) -> MLXArray {
        checkExtent(k, v, selection)
        if matrixUnits ?? matrixUnitsSupported(queries: q, selection: selection) {
            return matrixUnitLaunch(queries: q, keys: k, values: v, selection: selection, scale: scale)
        }
        return simdgroupLaunch(queries: q, keys: k, values: v, selection: selection, scale: scale)
    }

    static func simdgroupLaunch(
        queries q: MLXArray, keys k: MLXArray, values v: MLXArray, selection: MiniMaxM3MSASelection, scale: Float,
        kernel override: MLXFast.MLXFastKernel? = nil
    ) -> MLXArray {
        let (B, H, L, D) = (q.dim(0), q.dim(1), q.dim(2), q.dim(3))
        let G = selection.groups
        let hg = H / G
        let blocks = sorted(selection.topIndices.asType(.int32), axis: -1)
        let qc = contiguous(q.transposed(0, 2, 1, 3))  // [B, L, H, D]
        let meta = MLXArray([Int32(selection.offset), Int32(selection.keyLength)])
        let kernel =
            override
            ?? prefillKernel(
                element: elementType(q.dtype)!, hg: hg, d: D, keep: selection.keep, block: selection.blockSize)
        let out = kernel(
            [qc, k, v, blocks, meta, MLXArray([scale])],
            grid: (32 * (hg / 8), L, B * G), threadGroup: (32 * (hg / 8), 1, 1),
            outputShapes: [[B, L, H, D]], outputDTypes: [q.dtype])[0]
        return out.transposed(0, 2, 1, 3)
    }

    // MARK: selection kernels (block scores without the score sheet)

    // The indexer's block scores `[B, G, Sq, blocks]` without the `[B, G, Sq, Sk]` float32 score sheet. Port of the
    // vMLX Python runtime's selection kernels (`msa.py` at 5fc31aca); the Metal sources below are its sources.
    //
    // Reference (`referenceBlockScores`, the runtime's ops): score(r, g, j) = q[r, g] . k[j] as one MLX float32 matmul
    // of the float32-cast operands; keys past the row's position -inf; the key axis padded with -inf to whole blocks;
    // max over each block (NaN-propagating); the row's local block(s) +inf. The top-k over these scores (`select`:
    // argPartition) is the selection.
    //
    // The kernels compute the same numbers in place: the scores of one block live in registers only until its maximum
    // is taken, the index keys are read in their stored dtype (no float32 copy), keys past the row's position are never
    // read and the forced local block is written as +inf. Identical block scores give an identical top-k (same op,
    // same input), so the selected blocks AND their order are the reference's.
    //
    // Bit-identical scores need the reference's dot-product arithmetic, i.e. whatever MLX's float32 matmul runs for
    // these shapes (measured on an M5 Max by the Python runtime, MLX 0.32.3):
    // - `nax`: matrix units with TF32 on (MLX's default): 16x32x16 matmul2d ops (relaxed precision) accumulating 16
    //   head-dim elements per op in ascending order; the stored bf16 / fp16 values give the same bits as their float32
    //   casts.
    // - `fma`: TF32 off or no matrix units: one fused multiply-add per head-dim element, ascending.
    // - A one-row call with batch > 1 is neither (MLX runs a matrix-vector kernel there): such calls, and anything else
    //   `selectKernelSupported` does not cover, take the reference ops.
    // Which arithmetic this process's matmul runs is not read from MLX internals: `selectArithmetic` runs both kernels
    // against the reference ops on probe shapes once per (dtype, head dim) and admits the one that reproduces it bit
    // for bit; when neither does, every call takes the reference ops.

    /// `VMLX_MINIMAX_MSA_SELECT`: `reference` (the score-sheet ops), `kernel` (the admitted kernel), `nax` / `fma`
    /// (that kernel arithmetic even when it does not reproduce the reference: comparisons only); unset: `auto`.
    enum SelectRequest: String, Sendable {
        case auto, reference, kernel, nax, fma
    }

    /// The selection path a call takes: the reference ops or one kernel family.
    enum SelectPath: String, Sendable {
        case reference, nax, fma
    }

    static let selectEnvironmentVariable = "VMLX_MINIMAX_MSA_SELECT"

    // Selection kernel sizes (they only shape the work; every admitted kernel reproduces the reference bit for bit).
    // Measured by the Python runtime on the M5 Max, bf16, 4 groups (2048-row chunk at 32k: fragments per simdgroup
    // 1 / 2 / 4 = 4.4 / 2.8 / 3.6 ms; fma chunk tile of 32 / 64 pairs = 7.6 / 10.2 ms).
    static let selectSimdgroups = 4  // nax: simdgroups per threadgroup
    static let selectTargetSimdgroups = 8192  // nax: simdgroups per launch the key ranges aim for
    static let selectNaxFragments = 2  // nax: 16-pair fragments per simdgroup
    static let selectFmaPairs = 16  // fma: (group, row) pairs per threadgroup of the scalar kernel; more: tiled kernel
    static let selectSimdPairs = 32  // fma: pairs per threadgroup of the tiled simdgroup-matrix kernel

    static func requestedSelectPath() -> SelectRequest {
        if let forced = hooks.selectRequest { return forced }
        let raw = RuntimeEnvironment.value(selectEnvironmentVariable)?
            .trimmingCharacters(in: .whitespaces).lowercased() ?? ""
        if let request = SelectRequest(rawValue: raw), request != .auto { return request }
        // `VMLX_MINIMAX_MSA=dense` is the pre-kernel runtime end to end: the reference selection too.
        return requestedPath() == .dense ? .reference : .auto
    }

    /// `reference`, `nax` or `fma` for this selection call. Only picks a path: every kernel admitted here reproduces
    /// the reference's block scores bit for bit (`selectArithmetic`). Auto takes the admitted kernel for every
    /// supported call: measured by the Python runtime on the M5 Max (bf16, 4 groups, scores + top-k, dependent calls,
    /// medians of 7) it was faster than the reference ops at every shape tried, 1 ..< 2049 rows x 2049 ... 131072 keys.
    static func chooseSelectPath(queries: MLXArray, keys: MLXArray, blockSize: Int) -> SelectPath {
        let path = routeSelect(queries: queries, keys: keys, blockSize: blockSize)
        hooks.selectObserver?(path)
        return path
    }

    private static func routeSelect(queries: MLXArray, keys: MLXArray, blockSize: Int) -> SelectPath {
        let want = requestedSelectPath()
        if want == .reference || !selectKernelSupported(queries: queries, keys: keys, blockSize: blockSize) {
            return .reference
        }
        switch want {
        case .fma: return .fma
        case .nax: return matrixUnitsAvailable() ? .nax : .reference
        default: return selectArithmetic(queries.dtype, headDim: queries.dim(3)) ?? .reference
        }
    }

    /// The indexer's block scores `[B, G, Sq, blocks]` float32 (`referenceBlockScores`), by the routed path.
    static func blockScores(
        queries: MLXArray, keys: MLXArray, offset: Int, blockSize: Int, localBlocks: Int
    ) -> MLXArray {
        switch chooseSelectPath(queries: queries, keys: keys, blockSize: blockSize) {
        case .reference:
            return referenceBlockScores(
                queries: queries, keys: keys, offset: offset, blockSize: blockSize, localBlocks: localBlocks)
        case let arithmetic:
            return kernelBlockScores(
                queries: queries, keys: keys, offset: offset, blockSize: blockSize, localBlocks: localBlocks,
                arithmetic: arithmetic)
        }
    }

    /// Calls the selection kernels reproduce: Metal GPU, one index-key head, q and k in one float dtype, head dim a
    /// multiple of 16, blocks a multiple of 32 keys (<= 1024), and the reference's matmul a matrix product (batch 1
    /// with more than one (group, row) pair, or more than one row): a one-row call with batch > 1 runs MLX's
    /// matrix-vector kernel, whose arithmetic neither kernel has.
    static func selectKernelSupported(queries q: MLXArray, keys k: MLXArray, blockSize: Int) -> Bool {
        guard onGPU, q.ndim == 4, elementType(q.dtype) != nil, k.dtype == q.dtype, k.ndim == 4 else { return false }
        let (B, G, L, D) = (q.dim(0), q.dim(1), q.dim(2), q.dim(3))
        guard k.dim(0) == B, k.dim(1) == 1, k.dim(3) == D, D % 16 == 0, blockSize % 32 == 0, blockSize <= 1024
        else { return false }
        return (B == 1 && G * L > 1) || L > 1
    }

    /// Mutated kernel sources (tests: controls that must fail); nil keeps the runtime's.
    struct SelectSources {
        var nax = selectNaxSource
        var naxSplit = selectNaxSplitSource
        var fma = selectFmaSource
        var simd = selectSimdSource
    }

    private enum SelectKernelKind: String {
        case nax, naxSplit = "nax_split", fma, simd
    }

    private static func selectKernel(
        _ kind: SelectKernelKind, element: String, d: Int, block: Int, pt: Int, sources: SelectSources?
    ) -> MLXFast.MLXFastKernel {
        let s = sources ?? SelectSources()
        let constants: [(String, Int)]
        let body: String
        var header = ""
        switch kind {
        case .nax:
            constants = [("D", d), ("BLOCK", block), ("NSG", selectSimdgroups), ("FPS", pt)]
            body = s.nax
            header = matrixUnitHeader
        case .simd:
            constants = [("D", d), ("BLOCK", block), ("TP", pt)]
            body = s.simd
        case .naxSplit:
            constants = [("D", d), ("BLOCK", block), ("FPS", pt)]
            body = s.naxSplit
            header = matrixUnitHeader
        case .fma:
            constants = [("D", d), ("BLOCK", block), ("PT", pt), ("NSGB", block / 32)]
            body = s.fma
        }
        let source = constantsHeader(element, constants) + body
        let tag = "\(kind.rawValue)_\(element)_d\(d)_b\(block)" + (pt != 0 ? "_p\(pt)" : "")
        return cachedKernel(
            kernelName("vmlx_minimax_msa_select_\(tag)", header + source),
            inputs: ["q", "k", "meta"], outputs: ["out"], source: source, header: header,
            ensureRowContiguous: false)
    }

    /// `referenceBlockScores` computed by the `arithmetic` (`nax` / `fma`) kernel family: `[B, G, Sq, blocks]` float32,
    /// no score sheet. q `[B, G, Sq, D]` and k `[B, 1, Sk, D]` in one dtype, any strides.
    static func kernelBlockScores(
        queries q: MLXArray, keys k: MLXArray, offset: Int, blockSize block: Int, localBlocks local: Int,
        arithmetic: SelectPath, sources: SelectSources? = nil
    ) -> MLXArray {
        let (B, G, L, D) = (q.dim(0), q.dim(1), q.dim(2), q.dim(3))
        let sk = k.dim(2)
        let nb = (sk + block - 1) / block
        let P = G * L
        let element = elementType(q.dtype)!
        func meta(_ bps: Int) -> MLXArray {
            MLXArray([Int32(offset), Int32(sk), Int32(nb), Int32(bps), Int32(local)])
        }
        func launch(
            _ kernel: MLXFast.MLXFastKernel, grid: (Int, Int, Int), threadGroup: (Int, Int, Int), bps: Int = 0
        ) -> MLXArray {
            kernel(
                [q, k, meta(bps)], grid: grid, threadGroup: threadGroup, outputShapes: [[B, G, L, nb]],
                outputDTypes: [.float32])[0]
        }
        switch arithmetic {
        case .nax where P <= 32:
            // Decode-sized: every pair in one or two fragments, the block's tiles spread over a threadgroup.
            let fps = (P + 15) / 16
            let kernel = selectKernel(.naxSplit, element: element, d: D, block: block, pt: fps, sources: sources)
            return launch(kernel, grid: (block * nb, B, 1), threadGroup: (block, 1, 1))
        case .nax:
            let fps = max(1, min(selectNaxFragments, (P + 15) / 16))
            let fragmentGroups = (P + 16 * fps - 1) / (16 * fps)
            let ranges = max(1, (selectTargetSimdgroups + max(1, fragmentGroups) - 1) / max(1, fragmentGroups))
            let bps = max(1, (nb + ranges - 1) / ranges)
            let simdgroups = (nb + bps - 1) / bps * fragmentGroups
            let nsg = selectSimdgroups
            let kernel = selectKernel(.nax, element: element, d: D, block: block, pt: fps, sources: sources)
            return launch(
                kernel, grid: (32 * nsg * ((simdgroups + nsg - 1) / nsg), B, 1), threadGroup: (32 * nsg, 1, 1),
                bps: bps)
        case .fma where P > selectFmaPairs && block == 128:
            // Chunks: the tiled simdgroup-matrix kernel (same fma chain per element).
            let tp = selectSimdPairs
            let kernel = selectKernel(.simd, element: element, d: D, block: block, pt: tp, sources: sources)
            return launch(kernel, grid: (128 * nb, (P + tp - 1) / tp, B), threadGroup: (128, 1, 1))
        case .fma:
            var pt = 1
            while pt < min(P, selectFmaPairs) { pt *= 2 }
            let kernel = selectKernel(.fma, element: element, d: D, block: block, pt: pt, sources: sources)
            return launch(kernel, grid: (nb * block, (P + pt - 1) / pt, B), threadGroup: (block, 1, 1))
        case .reference:
            preconditionFailure("kernelBlockScores needs a kernel arithmetic (nax / fma)")
        }
    }

    nonisolated(unsafe) private static var selectArithmetics: [String: SelectPath?] = [:]

    /// `nax` or `fma`: the kernel family whose block scores equal `referenceBlockScores` bit for bit in this process
    /// for index tensors of `dtype` / head dim `headDim` (checked once on probe shapes, matrix units first), or nil
    /// when neither does (every call then takes the reference ops).
    static func selectArithmetic(_ dtype: DType, headDim d: Int = 128) -> SelectPath? {
        let key = "\(dtype)/\(d)"
        lock.lock()
        if let known = selectArithmetics[key] {
            lock.unlock()
            return known
        }
        lock.unlock()
        let result = selectArithmeticCheck(dtype, headDim: d)
        lock.lock()
        selectArithmetics[key] = result
        lock.unlock()
        return result
    }

    /// Probe shapes for the arithmetic check (the Python runtime's case): 4 groups, 2853 keys (not a multiple of the
    /// 128-key block); one row, five rows and forty rows ending at the last key.
    static func selectProbe(_ dtype: DType, headDim d: Int) -> [(MLXArray, MLXArray, Int)] {
        let keys = MLXRandom.split(key: MLXRandom.key(17), into: 4)
        let sk = 2853
        let k = (MLXRandom.normal([1, 1, sk, d], key: keys[0]) * Float(3)).asType(dtype)
        return [1, 5, 40].enumerated().map { i, rows in
            (MLXRandom.normal([1, 4, rows, d], key: keys[i + 1]).asType(dtype), k, sk - rows)
        }
    }

    private static func selectArithmeticCheck(_ dtype: DType, headDim d: Int) -> SelectPath? {
        guard onGPU, elementType(dtype) != nil, d % 16 == 0 else { return nil }
        let probes = selectProbe(dtype, headDim: d)
        guard
            let references = try? withError({ error -> [MLXArray] in
                let references = probes.map {
                    referenceBlockScores(queries: $0.0, keys: $0.1, offset: $0.2, blockSize: 128, localBlocks: 1)
                }
                eval(references)
                try error.check()
                return references
            })
        else { return nil }
        for arithmetic in matrixUnitsAvailable() ? [SelectPath.nax, .fma] : [.fma] {
            let equal =
                (try? withError { error -> Bool in
                    let got = probes.map {
                        kernelBlockScores(
                            queries: $0.0, keys: $0.1, offset: $0.2, blockSize: 128, localBlocks: 1,
                            arithmetic: arithmetic)
                    }
                    eval(got)
                    try error.check()
                    return zip(got, references).allSatisfy { arrayEqual($0, $1).item(Bool.self) }
                }) ?? false
            if equal { return arithmetic }
        }
        return nil
    }

    static let selectNaxSource = #"""
          const ushort lane = thread_index_in_simdgroup;
          const int sgid = int(threadgroup_position_in_grid.x) * NSG + int(simdgroup_index_in_threadgroup);
          const int b = int(threadgroup_position_in_grid.y);
          const int G = int(q_shape[1]);
          const int L = int(q_shape[2]);
          const int P = G * L;                                     // (group, row) pairs, p = g * L + l
          const int FG = (P + 16 * FPS - 1) / (16 * FPS);          // groups of FPS 16-pair fragments
          const int offset = meta[0];
          const int sk = meta[1];
          const int nb = meta[2];
          const int bps = meta[3];
          const int local = meta[4];
          const int nr = (nb + bps - 1) / bps;
          if (sgid >= nr * FG) return;                             // per-simdgroup ops only: no threadgroup barrier below
          const int fg = sgid % FG;                                // simdgroups of a threadgroup share the key range
          const int n_lo = (sgid / FG) * bps;
          const int n_hi = min(nb, n_lo + bps);

          // Fragment layout of a 16x16 tile: this lane holds rows fm and fm + 8, columns fn .. fn + 3.
          const short qid = short(lane >> 2);
          const short fm = (qid & 4) | short((lane >> 1) & 3);
          const short fn = short(((qid & 2) | (lane & 1)) * 4);

          int qp[FPS][2];                                          // position of the lane's pairs (-1: no pair)
          elem_t qf[FPS][D / 16][8];
          int qmax = -1;
          for (short u = 0; u < FPS; ++u) {
            for (short i = 0; i < 2; ++i) {
              const int p = (fg * FPS + u) * 16 + fm + 8 * i;
              const bool pv = p < P;
              const int g = pv ? p / L : 0;
              const int l = pv ? p - g * L : 0;
              qp[u][i] = pv ? offset + l : -1;
              qmax = max(qmax, qp[u][i]);
              device const elem_t* qrow = q + (size_t)b * q_strides[0] + (size_t)g * q_strides[1] + (size_t)l * q_strides[2];
              for (int c = 0; c < D / 16; ++c) {
                for (short j = 0; j < 4; ++j) {
                  qf[u][c][i * 4 + j] = pv ? qrow[(size_t)(c * 16 + fn + j) * q_strides[3]] : elem_t(0);
                }
              }
            }
          }
          qmax = simd_max(qmax);                                   // last key any pair of the simdgroup sees
          // first position no pair of the simdgroup sees past (lanes without a pair do not count)
          int qmin = INT_MAX;
          for (short u = 0; u < FPS; ++u) {
            for (short i = 0; i < 2; ++i) qmin = qp[u][i] >= 0 ? min(qmin, qp[u][i]) : qmin;
          }
          qmin = min(simd_min(qmin), sk - 1);
          device const elem_t* kbase = k + (size_t)b * k_strides[0];
          const size_t kss = k_strides[2];
          const size_t kds = k_strides[3];
          const bool vec_ok = kds == 1 && (kss % 4) == 0 && ((ulong)kbase % (4 * sizeof(elem_t))) == 0;
          const bool writer = (lane & 9) == 0;                     // one of the four lanes holding a row

          for (int n = n_lo; n < n_hi; ++n) {
            const int start = n * BLOCK;
            float m[FPS][2];
            bool bad[FPS][2];
            for (short u = 0; u < FPS; ++u) {
              m[u][0] = m[u][1] = -INFINITY;
              bad[u][0] = bad[u][1] = false;
            }
            for (int t0 = 0; t0 < BLOCK && start + t0 <= qmax; t0 += 32) {   // uniform across the simdgroup
              const int ts = start + t0;
              float s[FPS][16];
              for (short u = 0; u < FPS; ++u) {
                for (short e = 0; e < 16; ++e) s[u][e] = 0.0f;
              }
              for (int c = 0; c < D / 16; ++c) {
                elem_t kb0[8], kb1[8];
                for (short i = 0; i < 2; ++i) {
                  const int kp0 = ts + fm + 8 * i;
                  const int kp1 = kp0 + 16;
                  msa_load4(kb0 + i * 4, kbase + (size_t)kp0 * kss, c * 16 + fn, kds, vec_ok, kp0 < sk);
                  msa_load4(kb1 + i * 4, kbase + (size_t)kp1 * kss, c * 16 + fn, kds, vec_ok, kp1 < sk);
                }
                for (short u = 0; u < FPS; ++u) msa_mma_16x32<elem_t, elem_t, true>(s[u], qf[u][c], kb0, kb1);
              }
              // s[u][h * 8 + i * 4 + j] = score(pair row fm + 8i of fragment u, key ts + h * 16 + fn + j)
              if (ts + 31 <= qmin) {                               // every pair sees the whole tile (qmin < sk)
                for (short u = 0; u < FPS; ++u) {
                  for (short i = 0; i < 2; ++i) {
                    float tm = s[u][i * 4];
                    bool tb = isnan(tm);
                    for (short h = 0; h < 2; ++h) {
                      for (short j = 0; j < 4; ++j) {
                        const float x = s[u][h * 8 + i * 4 + j];
                        tb = tb || isnan(x);
                        tm = max(tm, x);
                      }
                    }
                    bad[u][i] = bad[u][i] || tb;
                    m[u][i] = max(m[u][i], tm);
                  }
                }
              } else {
                for (short u = 0; u < FPS; ++u) {
                  for (short h = 0; h < 2; ++h) {
                    for (short i = 0; i < 2; ++i) {
                      for (short j = 0; j < 4; ++j) {
                        const int kp = ts + h * 16 + fn + j;
                        if (kp <= qp[u][i] && kp < sk) {
                          const float x = s[u][h * 8 + i * 4 + j];
                          bad[u][i] = bad[u][i] || isnan(x);
                          m[u][i] = max(m[u][i], x);
                        }
                      }
                    }
                  }
                }
              }
            }
            for (short u = 0; u < FPS; ++u) {
              for (short i = 0; i < 2; ++i) {
                float mm = max(m[u][i], simd_shuffle_xor(m[u][i], ushort(1)));
                mm = max(mm, simd_shuffle_xor(mm, ushort(8)));
                int bi = int(bad[u][i]);
                bi |= simd_shuffle_xor(bi, ushort(1));
                bi |= simd_shuffle_xor(bi, ushort(8));
                if (writer && qp[u][i] >= 0) {
                  const int qb = qp[u][i] / BLOCK;
                  const bool forced = n <= qb && n >= max(0, qb - local + 1);
                  out[((size_t)b * P + (fg * FPS + u) * 16 + fm + 8 * i) * nb + n] = forced ? INFINITY : (bi ? NAN : mm);
                }
              }
            }
          }
        """#

    static let selectNaxSplitSource = #"""
          const ushort lane = thread_index_in_simdgroup;
          const int t = int(simdgroup_index_in_threadgroup);       // 32-key tile of the block
          const int tid = int(thread_position_in_threadgroup.x);
          const int n = int(threadgroup_position_in_grid.x);       // key block
          const int b = int(threadgroup_position_in_grid.y);
          const int G = int(q_shape[1]);
          const int L = int(q_shape[2]);
          const int P = G * L;
          const int offset = meta[0];
          const int sk = meta[1];
          const int nb = meta[2];
          const int local = meta[4];
          constexpr int NT = BLOCK / 32;
          threadgroup float red[NT * 16 * FPS];
          threadgroup int redn[NT * 16 * FPS];

          const short qid = short(lane >> 2);
          const short fm = (qid & 4) | short((lane >> 1) & 3);
          const short fn = short(((qid & 2) | (lane & 1)) * 4);
          int qp[FPS][2];
          int qmax = -1;
          for (short u = 0; u < FPS; ++u) {
            for (short i = 0; i < 2; ++i) {
              const int p = u * 16 + fm + 8 * i;
              qp[u][i] = p < P ? offset + p % L : -1;
              qmax = max(qmax, qp[u][i]);
            }
          }
          qmax = simd_max(qmax);
          float m[FPS][2];
          bool bad[FPS][2];
          for (short u = 0; u < FPS; ++u) {
            m[u][0] = m[u][1] = -INFINITY;
            bad[u][0] = bad[u][1] = false;
          }
          const int ts = n * BLOCK + t * 32;
          if (ts <= qmax) {                                        // uniform across the simdgroup
            elem_t qf[FPS][D / 16][8];
            for (short u = 0; u < FPS; ++u) {
              for (short i = 0; i < 2; ++i) {
                const int p = u * 16 + fm + 8 * i;
                const bool pv = p < P;
                const int g = pv ? p / L : 0;
                const int l = pv ? p - g * L : 0;
                device const elem_t* qrow = q + (size_t)b * q_strides[0] + (size_t)g * q_strides[1]
                    + (size_t)l * q_strides[2];
                for (int c = 0; c < D / 16; ++c) {
                  for (short j = 0; j < 4; ++j) {
                    qf[u][c][i * 4 + j] = pv ? qrow[(size_t)(c * 16 + fn + j) * q_strides[3]] : elem_t(0);
                  }
                }
              }
            }
            device const elem_t* kbase = k + (size_t)b * k_strides[0];
            const size_t kss = k_strides[2];
            const size_t kds = k_strides[3];
            const bool vec_ok = kds == 1 && (kss % 4) == 0 && ((ulong)kbase % (4 * sizeof(elem_t))) == 0;
            float s[FPS][16];
            for (short u = 0; u < FPS; ++u) {
              for (short e = 0; e < 16; ++e) s[u][e] = 0.0f;
            }
            for (int c = 0; c < D / 16; ++c) {
              elem_t kb0[8], kb1[8];
              for (short i = 0; i < 2; ++i) {
                const int kp0 = ts + fm + 8 * i;
                const int kp1 = kp0 + 16;
                msa_load4(kb0 + i * 4, kbase + (size_t)kp0 * kss, c * 16 + fn, kds, vec_ok, kp0 < sk);
                msa_load4(kb1 + i * 4, kbase + (size_t)kp1 * kss, c * 16 + fn, kds, vec_ok, kp1 < sk);
              }
              for (short u = 0; u < FPS; ++u) msa_mma_16x32<elem_t, elem_t, true>(s[u], qf[u][c], kb0, kb1);
            }
            for (short u = 0; u < FPS; ++u) {
              for (short h = 0; h < 2; ++h) {
                for (short i = 0; i < 2; ++i) {
                  for (short j = 0; j < 4; ++j) {
                    const int kp = ts + h * 16 + fn + j;
                    if (kp <= qp[u][i] && kp < sk) {
                      const float x = s[u][h * 8 + i * 4 + j];
                      bad[u][i] = bad[u][i] || isnan(x);
                      m[u][i] = max(m[u][i], x);
                    }
                  }
                }
              }
            }
          }
          for (short u = 0; u < FPS; ++u) {
            for (short i = 0; i < 2; ++i) {
              float mm = max(m[u][i], simd_shuffle_xor(m[u][i], ushort(1)));
              mm = max(mm, simd_shuffle_xor(mm, ushort(8)));
              int bi = int(bad[u][i]);
              bi |= simd_shuffle_xor(bi, ushort(1));
              bi |= simd_shuffle_xor(bi, ushort(8));
              if ((lane & 9) == 0) {
                red[(t * FPS + u) * 16 + fm + 8 * i] = mm;
                redn[(t * FPS + u) * 16 + fm + 8 * i] = bi;
              }
            }
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
          if (tid < 16 * FPS && tid < P) {
            float mm = -INFINITY;
            int bi = 0;
            for (int tt = 0; tt < NT; ++tt) {
              mm = max(mm, red[(tt * FPS + tid / 16) * 16 + tid % 16]);
              bi |= redn[(tt * FPS + tid / 16) * 16 + tid % 16];
            }
            const int qb = (offset + tid % L) / BLOCK;
            const bool forced = n <= qb && n >= max(0, qb - local + 1);
            out[((size_t)b * P + tid) * nb + n] = forced ? INFINITY : (bi ? NAN : mm);
          }
        """#

    static let selectFmaSource = #"""
          const int tid = int(thread_position_in_threadgroup.x);
          const int n = int(threadgroup_position_in_grid.x);
          const int p0 = int(threadgroup_position_in_grid.y) * PT;
          const int b = int(threadgroup_position_in_grid.z);
          const int G = int(q_shape[1]);
          const int L = int(q_shape[2]);
          const int P = G * L;
          const int offset = meta[0];
          const int sk = meta[1];
          const int nb = meta[2];
          const int local = meta[4];
          threadgroup float qs[PT * D];
          threadgroup float red[PT * NSGB];
          threadgroup int redn[PT * NSGB];
          for (int i = tid; i < PT * D; i += BLOCK) {
            const int pp = i / D;
            const int d = i - pp * D;
            const int p = p0 + pp;
            float x = 0.0f;
            if (p < P) {
              const int g = p / L;
              const int l = p - g * L;
              x = float(q[(size_t)b * q_strides[0] + (size_t)g * q_strides[1] + (size_t)l * q_strides[2]
                          + (size_t)d * q_strides[3]]);
            }
            qs[i] = x;
          }
          int qmax = -1;
          for (int pp = 0; pp < PT; ++pp) {
            if (p0 + pp < P) qmax = max(qmax, offset + (p0 + pp) % L);
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);

          const int kp = n * BLOCK + tid;
          float s[PT];
          for (int pp = 0; pp < PT; ++pp) s[pp] = 0.0f;
          if (kp < sk && kp <= qmax) {
            device const elem_t* krow = k + (size_t)b * k_strides[0] + (size_t)kp * k_strides[2];
            const size_t kds = k_strides[3];
            if (kds == 1 && ((ulong)krow % (4 * sizeof(elem_t))) == 0) {
              for (int d = 0; d < D; d += 4) {
                const vec<elem_t, 4> k4 = *((device const vec<elem_t, 4>*)(krow + d));
                for (int e = 0; e < 4; ++e) {
                  const float kd = float(k4[e]);
                  for (int pp = 0; pp < PT; ++pp) s[pp] = fma(qs[pp * D + d + e], kd, s[pp]);
                }
              }
            } else {
              for (int d = 0; d < D; ++d) {
                const float kd = float(krow[(size_t)d * kds]);
                for (int pp = 0; pp < PT; ++pp) s[pp] = fma(qs[pp * D + d], kd, s[pp]);
              }
            }
          }
          const int sg = tid / 32;
          const bool lead = (tid % 32) == 0;
          for (int pp = 0; pp < PT; ++pp) {
            const int p = p0 + pp;
            const int qpos = p < P ? offset + p % L : -1;
            const bool valid = kp < sk && kp <= qpos;
            const bool isn = valid && isnan(s[pp]);
            const float x = valid && !isn ? s[pp] : -INFINITY;
            const float bm = simd_max(x);
            const bool bn = simd_any(isn);
            if (lead) {
              red[pp * NSGB + sg] = bm;
              redn[pp * NSGB + sg] = int(bn);
            }
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
          if (tid < PT && p0 + tid < P) {
            float m = -INFINITY;
            int bad = 0;
            for (int i = 0; i < NSGB; ++i) {
              m = max(m, red[tid * NSGB + i]);
              bad |= redn[tid * NSGB + i];
            }
            const int p = p0 + tid;
            const int qb = (offset + p % L) / BLOCK;
            const bool forced = n <= qb && n >= max(0, qb - local + 1);
            out[((size_t)b * P + p) * nb + n] = forced ? INFINITY : (bad ? NAN : m);
          }
        """#

    static let selectSimdSource = #"""
          const uint lane = thread_index_in_simdgroup;
          const int sg = int(simdgroup_index_in_threadgroup);
          const int tid = int(thread_position_in_threadgroup.x);
          const int n = int(threadgroup_position_in_grid.x);
          const int p0 = int(threadgroup_position_in_grid.y) * TP;
          const int b = int(threadgroup_position_in_grid.z);
          const int G = int(q_shape[1]);
          const int L = int(q_shape[2]);
          const int P = G * L;
          const int offset = meta[0];
          const int sk = meta[1];
          const int nb = meta[2];
          const int local = meta[4];
          constexpr int BK = 16;
          constexpr int LDS = BK + 4;
          constexpr int RF = TP / 16;                              // 8-row fragments per simdgroup (TP / 2 rows)
          threadgroup float qs[TP * LDS];
          threadgroup float ks[BLOCK * LDS];
          threadgroup float red[2 * TP];
          threadgroup int redn[2 * TP];

          int qmax = -1;
          for (int pp = 0; pp < TP; ++pp) {
            if (p0 + pp < P) qmax = max(qmax, offset + (p0 + pp) % L);
          }
          const int start = n * BLOCK;
          const short qid = short(lane / 4);
          const short fm = (qid & 4) + short((lane / 2) % 4);      // 8x8 layout: row fm, columns fn, fn + 1
          const short fn = (qid & 2) * 2 + short(lane % 2) * 2;
          const int rbase = (sg / 2) * (TP / 2);
          const int cbase = (sg % 2) * 64;
          // Fragments as float2 registers (row fm, columns fn, fn + 1), wrapped in simdgroup matrices per op only: arrays of
          // simdgroup_matrix were 6x slower here.
          float2 Cf[RF][8];
          #pragma unroll
          for (int r = 0; r < RF; ++r) {
            #pragma unroll
            for (int c = 0; c < 8; ++c) Cf[r][c] = float2(0.0f);
          }
          if (start <= qmax) {                                     // uniform across the threadgroup
            device const elem_t* kb = k + (size_t)b * k_strides[0];
            const bool vec_ok = k_strides[3] == 1 && (k_strides[2] % 4) == 0 && ((ulong)kb % (4 * sizeof(elem_t))) == 0;
            for (int d0 = 0; d0 < D; d0 += BK) {
              for (int i = tid; i < TP * BK; i += 128) {
                const int pp = i / BK;
                const int d = d0 + i - pp * BK;
                const int p = p0 + pp;
                float x = 0.0f;
                if (p < P) {
                  const int g = p / L;
                  const int l = p - g * L;
                  x = float(q[(size_t)b * q_strides[0] + (size_t)g * q_strides[1] + (size_t)l * q_strides[2]
                              + (size_t)d * q_strides[3]]);
                }
                qs[pp * LDS + i - pp * BK] = x;
              }
              for (int i = tid; i < BLOCK * (BK / 4); i += 128) {
                const int kk = i / (BK / 4);
                const int dc = (i - kk * (BK / 4)) * 4;
                const int kp = start + kk;
                float4 x = float4(0.0f);
                if (kp < sk && kp <= qmax) {
                  device const elem_t* krow = kb + (size_t)kp * k_strides[2];
                  if (vec_ok) {
                    x = float4(*((device const vec<elem_t, 4>*)(krow + d0 + dc)));
                  } else {
                    for (int e = 0; e < 4; ++e) x[e] = float(krow[(size_t)(d0 + dc + e) * k_strides[3]]);
                  }
                }
                for (int e = 0; e < 4; ++e) ks[kk * LDS + dc + e] = x[e];
              }
              threadgroup_barrier(mem_flags::mem_threadgroup);
              #pragma unroll
              for (int kk = 0; kk < BK; kk += 8) {
                float2 Af[RF];
                #pragma unroll
                for (int r = 0; r < RF; ++r) {
                  const int row = rbase + r * 8 + fm;            // A[m][k] = qs[row m][kk + k]
                  Af[r] = float2(qs[row * LDS + kk + fn], qs[row * LDS + kk + fn + 1]);
                }
                #pragma unroll
                for (int c = 0; c < 8; ++c) {
                  const int col = cbase + c * 8 + fn;            // B[k][n] = ks[key n][kk + k]
                  const float2 Bf = float2(ks[col * LDS + kk + fm], ks[(col + 1) * LDS + kk + fm]);
                  #pragma unroll
                  for (int r = 0; r < RF; ++r) {
                    simdgroup_matrix<float, 8, 8> Am, Bm, Cm, Dm;
                    reinterpret_cast<thread float2&>(Am.thread_elements()) = Af[r];
                    reinterpret_cast<thread float2&>(Bm.thread_elements()) = Bf;
                    reinterpret_cast<thread float2&>(Cm.thread_elements()) = Cf[r][c];
                    simdgroup_multiply_accumulate(Dm, Am, Bm, Cm);
                    Cf[r][c] = reinterpret_cast<thread float2&>(Dm.thread_elements());
                  }
                }
              }
              threadgroup_barrier(mem_flags::mem_threadgroup);
            }
          }
          // this lane: pair rows rbase + r * 8 + fm, keys start + cbase + c * 8 + fn + e
          #pragma unroll
          for (int r = 0; r < RF; ++r) {
            const int p = p0 + rbase + r * 8 + fm;
            const int qpos = p < P ? offset + p % L : -1;
            float m = -INFINITY;
            bool bad = false;
            #pragma unroll
            for (int c = 0; c < 8; ++c) {
              #pragma unroll
              for (int e = 0; e < 2; ++e) {
                const int kp = start + cbase + c * 8 + fn + e;
                if (kp <= qpos && kp < sk) {
                  const float x = Cf[r][c][e];
                  bad = bad || isnan(x);
                  m = max(m, x);
                }
              }
            }
            m = max(m, simd_shuffle_xor(m, ushort(1)));
            m = max(m, simd_shuffle_xor(m, ushort(8)));
            int bi = int(bad);
            bi |= simd_shuffle_xor(bi, ushort(1));
            bi |= simd_shuffle_xor(bi, ushort(8));
            if ((lane & 9) == 0) {
              red[(sg % 2) * TP + rbase + r * 8 + fm] = m;
              redn[(sg % 2) * TP + rbase + r * 8 + fm] = bi;
            }
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
          if (tid < TP && p0 + tid < P) {
            const float m = max(red[tid], red[TP + tid]);
            const int bi = redn[tid] | redn[TP + tid];
            const int p = p0 + tid;
            const int qb = (offset + p % L) / BLOCK;
            const bool forced = n <= qb && n >= max(0, qb - local + 1);
            out[((size_t)b * P + p) * nb + n] = forced ? INFINITY : (bi ? NAN : m);
          }
        """#
}
