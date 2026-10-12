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
        let load = k == 2
            ? "uint a0 = r[wi], a1 = r[wi + 1];"
            : "uint w0 = r[wi], w1 = r[wi + 1], w2 = r[wi + 2]; "
                + "uint a0 = sh == 0u ? w0 : ((w0 >> sh) | (w1 << (32u - sh))); "
                + "uint a1 = sh == 0u ? w1 : ((w1 >> sh) | (w2 << (32u - sh)));"
        var steps: [String] = []
        for j in 0 ..< 8 {
            let o = so[j]
            let ex: String
            if o + JANGTCode.stateBits <= 32 {
                ex = o == 0 ? "a0" : "(a0 >> \(o)u)"
            } else if o >= 32 {
                ex = o == 32 ? "a1" : "(a1 >> \(o - 32)u)"
            } else {
                ex = "((a0 >> \(o)u) | (a1 << \(32 - o)u))"
            }
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

    /// `JANGHPrefillSource.nax` with the trellis loader, fp32 scales and word-strided rows.
    static func naxSource(words: Int) -> String {
        var s = JANGHPrefillSource.nax
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

    private static func kernel(k: Double, words: Int, dtype: DType, fused: Bool, rotate: Bool) -> MLXFast.MLXFastKernel {
        let type = dtype == .bfloat16 ? "bfloat16_t" : "half"
        let name = "jangt_prefill_K\(JANGTKernels.kname(k))_w\(words)_\(type)_f\(fused ? 1 : 0)_r\(rotate ? 1 : 0)"
        lock.lock()
        defer { lock.unlock() }
        if let cached = kernels[name] { return cached }
        let source = """
            constexpr int PAD = 64 + 16 / sizeof(\(type));
            threadgroup \(type) Wg[64 * PAD];
            threadgroup \(type) Wu[\(fused ? 64 : 1) * PAD];
            jt_gather_qmm_nax<\(type), 2, 2, \(fused), \(rotate), 64>(
                x, wg, sg, wu, su, indices, y, meta[0], meta[1], meta[2], meta[3], lim[0], Wg, Wu,
                threadgroup_position_in_grid, simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
            """
        let built = MLXFast.metalKernel(
            name: name, inputNames: ["x", "wg", "sg", "wu", "su", "indices", "meta", "lim"], outputNames: ["y"],
            source: source,
            header: JANGHPrefillHeaders.nax + loaderSource(k: k, words: words) + naxSource(words: words),
            ensureRowContiguous: false)
        kernels[name] = built
        return built
    }

    /// `x` (M, K_in) in the projection's ROTATED basis, rows sorted by expert (`indices` (M,) uint32, sorted).
    /// Returns (M, N) in `x.dtype`: `x . W[e]^T` (or SwiGLU(gate, up) clamped at `limit` when `up` is given; with
    /// `rotateOutput`, the result is Hadamard-32 rotated — the input basis of a JANGH down unit). No outlier columns.
    static func project(
        _ x: MLXArray, _ projection: JANGTProjection, up: JANGTProjection? = nil, indices: MLXArray,
        limit: Float = 0, rotateOutput: Bool = false
    ) -> MLXArray {
        let m = x.dim(0), kin = x.dim(1), n = projection.outputs
        precondition(available && [.bfloat16, .float16].contains(x.dtype) && kin == projection.inputs
                     && kin % 64 == 0 && indices.size == m && (up == nil || up!.k == projection.k)
                     && (!rotateOutput || n % 32 == 0), "JANGT prefill: unsupported geometry")
        // MLX passes inputs of < 8 elements through constant pointers; the tile reads device pointers.
        let idx = indices.size < 8
            ? concatenated([indices, broadcast(indices[(indices.size - 1)...], to: [8 - indices.size])])
            : indices
        let meta = MLXArray([Int32(m), Int32(n), Int32(kin), Int32(projection.experts)])
        let u = up ?? projection
        return kernel(k: projection.k, words: projection.words, dtype: x.dtype, fused: up != nil, rotate: rotateOutput)(
            [contiguous(x), projection.packed, projection.scale, u.packed, u.scale, contiguous(idx.asType(.uint32)),
             meta, MLXArray([limit])],
            grid: (((n + 63) / 64) * 128, (m + 63) / 64, 1), threadGroup: (128, 1, 1),
            outputShapes: [[m, n]], outputDTypes: [x.dtype])[0]
    }

    /// The outlier-column term of a projection for expert-sorted rows: rawRows (M, K_in) in the RAW basis.
    /// Returns (M, N) float32 = sum_j raw[ocol[e, j]] * wout[e, :, j] (padding columns contribute nothing).
    static func outliers(_ rawRows: MLXArray, _ projection: JANGTProjection, indices: MLXArray) -> MLXArray? {
        guard projection.outlierColumns > 0 else { return nil }
        let rows = indices.asType(.int32)
        let cols = take(projection.outlierColumnIndices, rows, axis: 0)  // (M, n_out)
        let valid = (cols .>= MLXArray(Int32(0))).asType(.float32)
        let xo = takeAlong(rawRows.asType(.float32), maximum(cols, MLXArray(Int32(0))), axis: 1) * valid
        // (E, N, n_out) -> (E, n_out, N), transient fp32 for this call only (never cached: 37 MB per GLM down bank).
        let wT = projection.outlierWeights.swappedAxes(-1, -2).asType(.float32)
        return gatherMM(xo.expandedDimensions(axis: 1), wT, rhsIndices: indices.asType(.uint32),
                        sortedIndices: true).squeezed(axis: 1)
    }
}
