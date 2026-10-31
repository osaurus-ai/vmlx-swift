// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXFast

/// Expert-sorted JANGT prefill: the JANGH NAX gather matmul (`JANGHPrefillSource.nax`) with a trellis tile loader.
/// Port of vMLX Python `jangt/prefill.py` (2026-10-09/10; N0.5 prefill +18 % over the decode kernels there, GLM
/// cold prefill tracks it). Only the weight pointer (uint32 rows of W words), the fp32 row scales and the loader
/// change; tiling, expert segmentation, the fused SwiGLU (clamped at `limit`) and the optional H32 output epilogue
/// are the JANGH kernel's.
///
/// Loader: 128 threads fill a 64-row x 64-column tile, thread = (row, half): 32 columns = two 16-column runs, each
/// decoded with the decode kernels' window logic and normalized as (raw - mu) * isd * row_scale in the tile dtype.
/// Input rows are the ROTATED basis (H128 with the projection's signs), sorted by expert. Outlier columns are not
/// included (the caller adds them). NAX (M5-class GPUs, macOS 26.2+) only: elsewhere the caller keeps the decode
/// kernels, which are correct on every GPU (the vMLX Python NAX path computed garbage on an M4).
enum JANGTPrefillKernel {
    static var available: Bool {
        JANGHPrefillKernel.nativeBackend == .nax
    }

    private static let lock = NSLock()
    nonisolated(unsafe) private static var kernels: [String: MLXFast.MLXFastKernel] = [:]

    /// The tile loader for one code width and row stride (W words).
    static func loaderSource(k: Double, words: Int) -> String {
        let (mu0, mu1, isd0, isd1) = JANGTKernelSource.statsLiterals(k)
        let so = JANGTCode.stepOffsets(k, 9)
        let mult = JANGTCode.mult[k]!
        let mask = (1 << JANGTCode.stateBits) - 1
        let pl16 = JANGTCode.laneBits(k)
        // Sub-chunk starts are multiples of PL16 bits, the same shift set as the decode lanes: one window rule for
        // every width (K=1/1.5: 2 words realigned into a0; K=2.5/3: 3 words into a0/a1; K=2: static). Python
        // `prefill._loader` (2026-10-16).
        let load = JANGTKernelSource.windowLoad(k: k, l: JANGTCode.stateBits) { $0 == 0 ? "r[wi]" : "r[wi + \($0)]" }
        var steps: [String] = []
        for j in 0 ..< 8 {
            let ex = JANGTKernelSource.stepExtract(so[j], l: JANGTCode.stateBits)
            steps.append(
                "{ uint s = (\(ex)) & \(mask)u; half2 v = as_type<half2>(((s * \(mult)u) & \(JANGTCode.mask2)u) | \(JANGTCode.orb2)u);"
                    + " d[\(2 * j)] = T((float(v.x) - \(mu0)) * \(isd0) * s_row);"
                    + " d[\(2 * j + 1)] = T((float(v.y) - \(mu1)) * \(isd1) * s_row); }")
        }
        return """

            template <typename T, short BROWS, short BCOLS, short dst_ld, short tgp_size>
            struct JTBlockLoader {
              static_assert(BCOLS == 64 && tgp_size == 128, "JANGT loader: 64-column tiles, 128 threads");
              const short thread_idx; const short bi; const short hf;
              threadgroup T* dst; const device uint32_t* src; float s_row; uint bitpos;
              JTBlockLoader(const device uint32_t* src_, const device float* scales_, const int src_ld_,
                            threadgroup T* dst_, ushort simd_group_id, ushort simd_lane_id, short valid_rows)
                  : thread_idx(simd_group_id * 32 + simd_lane_id), bi(thread_idx / 2), hf(thread_idx % 2),
                    dst(dst_ + bi * dst_ld + hf * 32), src(src_ + size_t(bi) * \(words)), bitpos(uint(hf) * \(2 * pl16)u) {
                s_row = (bi < valid_rows) ? scales_[bi] : 0.0f;
              }
              METAL_FUNC void sub_(uint b, threadgroup T* d) const {
                const device uint32_t* r = src; uint wi = b >> 5; uint sh = b & 31u; (void)sh;
                \(load)
                \(steps.joined(separator: " "))
              }
              void load_unsafe() const { if (bi >= BROWS) return; sub_(bitpos, dst); sub_(bitpos + \(pl16)u, dst + 16); }
              void load_safe(short2 src_tile_dim) const {
                if (bi >= BROWS) return;
                if (bi >= src_tile_dim.y) { for (short i = 0; i < 32; i++) dst[i] = T(0); return; }
                sub_(bitpos, dst); sub_(bitpos + \(pl16)u, dst + 16);
              }
              void next() { bitpos += \(4 * pl16)u; }
            };

            """
    }

    /// Fused gate/up with the layer-wide OUTLIER COLUMNS added to both accumulators before the activation (vMLX
    /// Python `_EPI`, 2026-10-09): one pass instead of a gate pass, an up pass and two outlier matmuls. `xo` (M, NO)
    /// fp32 = raw-basis x at the columns, for the sorted rows; `wog`/`wou` are the bundle's fp16 (E, N, NO) outlier
    /// weights read in place (the reference keeps an fp32 copy of them; ~0.6 GB over Naive-N0.5's layers).
    static func outlierEpilogue(_ count: Int) -> String {
        """

            {
              for (short f = 0; f < decltype(Gt)::kNumFrags; ++f)
                for (short e = 0; e < decltype(Gt)::kElemsPerFrag; ++e) {
                  const short2 pos = BaseNAXFrag::get_coord(e);
                  const int row = tm + (f / TN) * 16 + pos.y; const int col = tn + (f % TN) * 16 + pos.x;
                  if (row < tgp_bm && col < tgp_bn) {
                    const device float* xr_ = xo + size_t(y_row + row) * \(count);
                    const size_t wb = (size_t(index) * N + y_col + col) * \(count);
                    float ag = 0.0f, au = 0.0f;
                    for (int j = 0; j < \(count); ++j) { ag += xr_[j] * float(wog[wb + j]); au += xr_[j] * float(wou[wb + j]); }
                    Gt.val_frags[f][e] += ag; Ut.val_frags[f][e] += au;
                  }
                }
            }

        """
    }

    /// `JANGHPrefillSource.nax` with the trellis loader, fp32 scales and word-strided rows (and, with
    /// `outliers` > 0, the fused outlier epilogue). The fused epilogue applies `activation` after the source's clamp
    /// at `lim`; `.silu` keeps the source's expression unchanged.
    static func naxSource(words: Int, outliers: Int = 0, activation: JANGTActivation = .silu,
                          expertAligned: Bool = false) -> String {
        var s = JANGHPrefillSource.nax
        if activation != .silu {
            let silu = "return \(JANGTActivation.silu.metalValue());"
            precondition(s.components(separatedBy: silu).count == 2, "JANGT prefill: activation anchor missing")
            s = s.replacingOccurrences(of: silu, with: "return \(activation.metalValue());")
        }
        if outliers > 0 {
            let params = "const device uint32_t* indices, device T* y, const int M, const int N, const int K, const int EXPERTS, const float lim,"
            // The activation loop (whitespace-insensitive: the source is an indented multi-line literal).
            let pattern = #"if \(FUSED\) \{\s*for \(short i = 0; i < decltype\(Gt\)::kNumFrags; i\+\+\)"#
            guard s.contains(params), let act = s.range(of: pattern, options: .regularExpression) else {
                preconditionFailure("JANGT prefill: outlier anchors missing")
            }
            s.insert(contentsOf: outlierEpilogue(outliers), at: act.lowerBound)
            s = s.replacingOccurrences(
                of: params, with: params + " const device float* xo, const device half* wog, const device half* wou,")
        }
        if expertAligned { s = expertAlignedSource(s) }
        let replacements: [(String, String)] = [
            ("METAL_FUNC float tq_act(", "METAL_FUNC float jt_act("),
            ("METAL_FUNC void tq_gather_qmm_nax(", "METAL_FUNC void jt_gather_qmm_nax("),
            ("const device uint32_t* wg, const device half* sg, const device uint32_t* wu, const device half* su,",
             "const device uint32_t* wg, const device float* sg, const device uint32_t* wu, const device float* su,"),
            ("using loader_w_t = TQBlockLoader<T, BN, BK, BK_padded, WM * WN * SIMD_SIZE, bits>;",
             "using loader_w_t = JTBlockLoader<T, BN, BK, BK_padded, WM * WN * SIMD_SIZE>;"),
            ("using loader_u_t = TQBlockLoader<T, BN, BK, BK_padded, WM * WN * SIMD_SIZE, bits_u>;",
             "using loader_u_t = JTBlockLoader<T, BN, BK, BK_padded, WM * WN * SIMD_SIZE>;"),
            ("const int K_w = K * bytes_per_pack / pack_factor; const int K_it = K / BK;",
             "const int K_w = \(words); const int K_it = K / BK;"),
            ("const int K_wu = K * get_bytes_per_pack<bits_u>() / get_pack_factor<bits_u, 8>();",
             "const int K_wu = \(words);"),
            ("auto wgl = (const device uint8_t*)wg; auto wul = (const device uint8_t*)wu;",
             "auto wgl = wg; auto wul = wu;"),
            ("= tq_act(Gt.val_frags[i][e], Ut.val_frags[i][e], lim);",
             "= jt_act(Gt.val_frags[i][e], Ut.val_frags[i][e], lim);"),
        ]
        for (from, to) in replacements {
            precondition(s.contains(from), "JANGT prefill: NAX source anchor missing: \(from)")
            s = s.replacingOccurrences(of: from, with: to)
        }
        // Only the gather routine: the steel/split-K helpers are other kernels' sources.
        return s
    }

    /// The expert-aligned schedule (vMLX Python `jangh.kernels` EXPERT_ALIGNED, edf7e36e8): threadgroup row `tid.y`
    /// finds its segment in `tile_offsets` (binary search, empty segments skipped) and covers 64 rows of ONE segment,
    /// [expert_offsets[s] + 64 (tid.y - tile_offsets[s]), expert_offsets[s + 1]). The row-tiled schedule cut the sorted
    /// rows every 64 regardless of experts, so a tile straddling a boundary ran its segments one after the other, each
    /// decoding a full weight column block for a part of the tile's rows with the other simdgroups idle. Here no tile
    /// mixes two experts. A row's arithmetic is unchanged (same weight tile values, same K order, same epilogue), only
    /// which threadgroup computes it. Threadgroups past the last tile return before any barrier (the grid is an upper
    /// bound: rows / 64 + segments). The last segment holds the rows whose index is not a valid expert, so they still
    /// take the source's NaN path instead of being skipped.
    static func expertAlignedSource(_ source: String) -> String {
        var s = source
        let params = "const device uint32_t* indices, device T* y, const int M, const int N, const int K, const int EXPERTS, const float lim,"
        let replacements: [(String, String)] = [
            (params, params + " const device int* expert_offsets, const device int* tile_offsets, const int SEGMENTS,"),
            ("const int y_row = tid.y * BM; const int y_col = tid.x * BN;",
             """
             if (int(tid.y) >= tile_offsets[SEGMENTS]) return;  // uniform: the whole threadgroup, before any barrier
                       int seg_lo = 0, seg_hi = SEGMENTS;
                       while (seg_lo < seg_hi) {  // the segment s with tile_offsets[s] <= tid.y < tile_offsets[s + 1]
                         const int mid = (seg_lo + seg_hi) / 2;
                         if (tile_offsets[mid + 1] <= int(tid.y)) seg_lo = mid + 1; else seg_hi = mid;
                       }
                       const int y_row = expert_offsets[seg_lo] + (int(tid.y) - tile_offsets[seg_lo]) * BM;
                       const int row_end = expert_offsets[seg_lo + 1];
                       const int y_col = tid.x * BN;
             """),
            ("const short tgp_bm = short(min(BM, M - y_row));", "const short tgp_bm = short(min(BM, row_end - y_row));"),
            ("const short sgp_sm = short(min(int(SM), max(0, M - (y_row + tm))));",
             "const short sgp_sm = short(min(int(SM), max(0, row_end - (y_row + tm))));"),
        ]
        for (from, to) in replacements {
            precondition(s.components(separatedBy: from).count == 2, "JANGT prefill: expert-tile anchor missing: \(from)")
            s = s.replacingOccurrences(of: from, with: to)
        }
        return s
    }

    /// The expert-aligned tile plan of sorted expert ids (vMLX Python `expert_tile_plan`), on the GPU, no host
    /// synchronization: `starts` (E + 2,) int32 = the first sorted row of each expert 0 ... E - 1, then of the rows whose
    /// id is not a valid expert (>= E), then M; `tiles` (E + 2,) int32 = the exclusive prefix sum of each segment's
    /// 64-row tile count. Shared by every projection of a layer that runs on the same sorted ids.
    public struct ExpertTilePlan {
        public let starts: MLXArray
        public let tiles: MLXArray
        /// E + 1 (the experts and the invalid-id segment).
        public let segments: Int
        public let rows: Int
    }

    public static func expertTilePlan(sortedIndices: MLXArray, experts: Int) -> ExpertTilePlan {
        let ids = sortedIndices.reshaped(-1).asType(.uint32)
        let m = ids.dim(0)
        // starts[e] = #ids < e for e = 0 ... E (sorted ids: the first row of expert e), then M.
        let bounds = MLXArray(0 ..< UInt32(experts + 1)).reshaped(experts + 1, 1)
        let below = (ids.reshaped(1, m) .< bounds).asType(.int32).sum(axis: 1)
        var starts = concatenated([below, MLXArray([Int32(m)])]).asType(.int32)
        // Tiles per segment, ceil(rows / 64): a shift (non-negative), not MLX's composed integer floorDivide.
        let counts = rightShift(starts[1...] - starts[..<(experts + 1)] + MLXArray(Int32(63)), MLXArray(Int32(6)))
        var tiles = concatenated([MLXArray([Int32(0)]), cumsum(counts, axis: 0)]).asType(.int32)
        // MLX passes inputs of < 8 elements through constant pointers; the tile reads device pointers. Entries past
        // `segments` are never read.
        if experts + 2 < 8 {
            starts = concatenated([starts, broadcast(starts[(experts + 1)...], to: [6 - experts])])
            tiles = concatenated([tiles, broadcast(tiles[(experts + 1)...], to: [6 - experts])])
        }
        return ExpertTilePlan(starts: starts, tiles: tiles, segments: experts + 1, rows: m)
    }

    /// `VMLX_JANGT_EXPERT_TILES` (default on; `0` = the row-tiled schedule). Read per call (getenv, no allocation) so
    /// one process can A/B both schedules, as vMLX Python's `JANGT_EXPERT_TILES`.
    public static var expertTilesEnabled: Bool {
        guard let raw = getenv("VMLX_JANGT_EXPERT_TILES") else { return true }
        return String(cString: raw) != "0"
    }

    private static func kernel(k: Double, words: Int, dtype: DType, fused: Bool, rotate: Bool, outliers: Int = 0,
                               activation: JANGTActivation = .silu, expertAligned: Bool = false)
        -> MLXFast.MLXFastKernel
    {
        let type = dtype == .bfloat16 ? "bfloat16_t" : "half"
        // Only the fused epilogue applies the activation: single-weight tiles share the SiLU source and name.
        let activation = fused ? activation : .silu
        let name = "jangt_prefill_K\(JANGTKernels.kname(k))_w\(words)_\(type)_f\(fused ? 1 : 0)_r\(rotate ? 1 : 0)_o\(outliers)"
            + activation.nameTag + (expertAligned ? "_et" : "")
        lock.lock()
        defer { lock.unlock() }
        if let cached = kernels[name] { return cached }
        let source = """
            constexpr int PAD = 64 + 16 / sizeof(\(type));
            threadgroup \(type) Wg[64 * PAD];
            threadgroup \(type) Wu[\(fused ? 64 : 1) * PAD];
            jt_gather_qmm_nax<\(type), 2, 2, \(fused), \(rotate), 64>(
                x, wg, sg, wu, su, indices, y, meta[0], meta[1], meta[2], meta[3], lim[0],\(expertAligned ? " expert_offsets, tile_offsets, meta[4]," : "")\(outliers > 0 ? " xo, wog, wou," : "") Wg, Wu,
                threadgroup_position_in_grid, simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
            """
        let built = MLXFast.metalKernel(
            name: name,
            inputNames: ["x", "wg", "sg", "wu", "su", "indices", "meta", "lim"] + (outliers > 0 ? ["xo", "wog", "wou"] : [])
                + (expertAligned ? ["expert_offsets", "tile_offsets"] : []),
            outputNames: ["y"],
            source: source,
            header: JANGHPrefillHeaders.nax + loaderSource(k: k, words: words)
                + naxSource(words: words, outliers: outliers, activation: activation, expertAligned: expertAligned),
            ensureRowContiguous: false)
        kernels[name] = built
        return built
    }

    /// `x` (M, K_in) in the projection's ROTATED basis, rows sorted by expert (`indices` (M,) uint32, sorted).
    /// Returns (M, N) in `x.dtype`: `x . W[e]^T` (or `activation`(gate, up) clamped at `limit` when `up` is given;
    /// with `rotateOutput`, the result is Hadamard-32 rotated — the input basis of a JANGH down unit). No outlier
    /// columns.
    /// `plan` (`expertTilePlan` of `indices`): the expert-aligned schedule; nil = row-tiled. Same output either way.
    static func project(
        _ x: MLXArray, _ projection: JANGTProjection, up: JANGTProjection? = nil, indices: MLXArray,
        limit: Float = 0, activation: JANGTActivation = .silu, rotateOutput: Bool = false,
        plan: ExpertTilePlan? = nil
    ) -> MLXArray {
        let m = x.dim(0), kin = x.dim(1), n = projection.outputs
        precondition(available && [.bfloat16, .float16].contains(x.dtype) && kin == projection.inputs
                     && kin % 64 == 0 && indices.size == m && (up == nil || up!.k == projection.k)
                     && (!rotateOutput || n % 32 == 0), "JANGT prefill: unsupported geometry")
        // MLX passes inputs of < 8 elements through constant pointers; the tile reads device pointers.
        let idx = indices.size < 8
            ? concatenated([indices, broadcast(indices[(indices.size - 1)...], to: [8 - indices.size])])
            : indices
        precondition(plan.map { $0.rows == m } ?? true, "JANGT prefill: tile plan of other rows")
        let meta = MLXArray([Int32(m), Int32(n), Int32(kin), Int32(projection.experts)]
            + (plan.map { [Int32($0.segments)] } ?? []))
        let u = up ?? projection
        return kernel(k: projection.k, words: projection.words, dtype: x.dtype, fused: up != nil, rotate: rotateOutput,
                      activation: activation, expertAligned: plan != nil)(
            [contiguous(x), projection.packed, projection.scale, u.packed, u.scale, contiguous(idx.asType(.uint32)),
             meta, MLXArray([limit])] + (plan.map { [$0.starts, $0.tiles] } ?? []),
            grid: (((n + 63) / 64) * 128, (m + 63) / 64 + (plan?.segments ?? 0), 1), threadGroup: (128, 1, 1),
            outputShapes: [[m, n]], outputDTypes: [x.dtype])[0]
    }

    /// Fused gate/up for expert-sorted rows with the layer-wide outlier columns `columns` (NO,) added before the
    /// (clamped) `activation`: `xr` rotated rows, `rawRows` raw-basis rows (M, K_in). Same output as two `project`
    /// passes plus `outliers` on each, in one tile pass.
    static func projectGateUpWithOutliers(
        _ xr: MLXArray, rawRows: MLXArray, gate: JANGTProjection, up: JANGTProjection, columns: MLXArray,
        indices: MLXArray, limit: Float = 0, activation: JANGTActivation = .silu, plan: ExpertTilePlan? = nil
    ) -> MLXArray {
        projectGateUpWithOutliers(
            xr, outlierRows: take(rawRows, columns.asType(.int32), axis: 1).asType(.float32), gate: gate, up: up,
            columns: columns.size, indices: indices, limit: limit, activation: activation, plan: plan)
    }

    /// As above with the outlier activations already gathered: `outlierRows` (M, NO) float32 = the raw-basis input
    /// at the layer's outlier columns, for the sorted rows (gathering those few columns before the rows avoids
    /// copying every raw row).
    static func projectGateUpWithOutliers(
        _ xr: MLXArray, outlierRows: MLXArray, gate: JANGTProjection, up: JANGTProjection, columns count: Int,
        indices: MLXArray, limit: Float = 0, activation: JANGTActivation = .silu, plan: ExpertTilePlan? = nil
    ) -> MLXArray {
        let m = xr.dim(0), kin = xr.dim(1), n = gate.outputs
        precondition(available && [.bfloat16, .float16].contains(xr.dtype) && kin == gate.inputs && kin % 64 == 0
                     && indices.size == m && up.k == gate.k && count == gate.outlierColumns
                     && count == up.outlierColumns && outlierRows.shape == [m, count] && outlierRows.dtype == .float32
                     && (plan.map { $0.rows == m } ?? true), "JANGT prefill: unsupported outlier geometry")
        let idx = indices.size < 8
            ? concatenated([indices, broadcast(indices[(indices.size - 1)...], to: [8 - indices.size])])
            : indices
        let meta = MLXArray([Int32(m), Int32(n), Int32(kin), Int32(gate.experts)]
            + (plan.map { [Int32($0.segments)] } ?? []))
        return kernel(k: gate.k, words: gate.words, dtype: xr.dtype, fused: true, rotate: false, outliers: count,
                      activation: activation, expertAligned: plan != nil)(
            [contiguous(xr), gate.packed, gate.scale, up.packed, up.scale, contiguous(idx.asType(.uint32)),
             meta, MLXArray([limit]), contiguous(outlierRows), gate.outlierWeights, up.outlierWeights]
                + (plan.map { [$0.starts, $0.tiles] } ?? []),
            grid: (((n + 63) / 64) * 128, (m + 63) / 64 + (plan?.segments ?? 0), 1), threadGroup: (128, 1, 1),
            outputShapes: [[m, n]], outputDTypes: [xr.dtype])[0]
    }

    /// The outlier-column term of a projection for expert-sorted rows: rawRows (M, K_in) in the RAW basis.
    /// Returns (M, N) float32 = sum_j raw[ocol[e, j]] * wout[e, :, j] (padding columns contribute nothing).
    static func outliers(_ rawRows: MLXArray, _ projection: JANGTProjection, indices: MLXArray) -> MLXArray? {
        guard projection.outlierColumns > 0 else { return nil }
        let rows = indices.asType(.int32)
        let cols = take(projection.outlierColumnIndices, rows, axis: 0)  // (M, n_out)
        let valid = (cols .>= MLXArray(Int32(0))).asType(.float32)
        // Gather the few columns, then widen (exact either way; widening every raw row first copied (M, K_in) fp32).
        let xo = takeAlong(rawRows, maximum(cols, MLXArray(Int32(0))), axis: 1).asType(.float32) * valid
        // (E, N, n_out) -> (E, n_out, N), transient fp32 for this call only (never cached: 37 MB per GLM down bank).
        let wT = projection.outlierWeights.swappedAxes(-1, -2).asType(.float32)
        return gatherMM(xo.expandedDimensions(axis: 1), wT, rhsIndices: indices.asType(.uint32),
                        sortedIndices: true).squeezed(axis: 1)
    }
}
