// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXLMCommon

// MiniMax-M3 MSA (block-sparse attention): the block selection and the attention paths that consume it.
//
// Port of the vMLX Python runtime's `vmlx_engine/models/minimax_m3/msa.py` (perf/minimax-m3-block-sparse-msa,
// 885355f0). The Metal sources below are that file's kernel sources; the routing thresholds and the env lever are
// the same. (Chunks of more than 16 rows take the dense path here; the chunk kernel is separate.)
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
        case dense, gather, decode
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
    // Longer chunks take the dense path until the chunk kernel lands.
    static let decodeMaxRows = 16
    static let decodeAlwaysRows = 8
    static let decodeMinKeys = 4096

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

    /// The custom kernels run on the Metal GPU only (on a CPU default device every call takes the dense path).
    static var onGPU: Bool { Device.defaultDevice().deviceType == .gpu }

    /// `dense`, `gather` or `decode` for this call. Only picks a path; every path computes the same
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
        case .kernel: return .dense
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
        return .dense
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
}
