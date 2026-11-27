// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN

/// Fused Metal kernels for ERNIE 5.5 Flash Preview (decode, multi-row DSpark verify, and exact FP32 compressor).
enum Ernie55Kernels {
    private static let scalarLock = NSLock()
    nonisolated(unsafe) private static var scalars: [Float: MLXArray] = [:]

    static func scalar(_ value: Float) -> MLXArray {
        scalarLock.lock()
        defer { scalarLock.unlock() }
        if let existing = scalars[value] { return existing }
        let array = MLXArray([value])
        array.eval()
        scalars[value] = array
        return array
    }

    // MARK: - 1. Exact IEEE FP32 Dual Compressor Projection (wkv + wgate)
    //
    // Computes keys = x.f32 @ keyWeight.f32.T and gates = x.f32 @ gateWeight.f32.T in full
    // 23-bit IEEE FP32 (never TF32) directly from BF16 or FP32 weights, avoiding both the
    // M=1 vs batched TF32 discrepancy and per-token FP32 weight casts (~1.24 GB/tok saved).

    #if canImport(Metal)
    private static let compressorTiledHeader = "#include <metal_simdgroup_matrix>\n"

    private static let compressorDualTiledKernel = MLXFast.metalKernel(
        name: "ernie55_compressor_dual_f32_tiled",
        inputNames: ["x", "wk", "wg"],
        outputNames: ["keys", "gates"],
        source: """
            const uint tid = thread_position_in_threadgroup.x;
            const uint sg = simdgroup_index_in_threadgroup;
            const uint m0 = threadgroup_position_in_grid.y * 32u;
            const uint n0 = threadgroup_position_in_grid.x * 32u;
            const uint sg_m = (sg >> 1u) * 16u;
            const uint sg_n = (sg & 1u) * 16u;

            threadgroup float As[32][33];
            threadgroup float Bs[32][33];

            simdgroup_float8x8 accK[2][2];
            simdgroup_float8x8 accG[2][2];
            #pragma clang loop unroll(full)
            for (uint i = 0u; i < 2u; ++i) {
                #pragma clang loop unroll(full)
                for (uint j = 0u; j < 2u; ++j) {
                    accK[i][j] = simdgroup_float8x8(0.0f);
                    accG[i][j] = simdgroup_float8x8(0.0f);
                }
            }

            const uint lr = tid >> 2u;
            const uint lc = (tid & 3u) * 8u;

            for (uint k0 = 0u; k0 < uint(K); k0 += 32u) {
                if (m0 + lr < uint(M)) {
                    const auto xr = x + size_t(m0 + lr) * uint(K) + k0 + lc;
                    #pragma clang loop unroll(full)
                    for (uint j = 0u; j < 8u; ++j) As[lr][lc + j] = float(xr[j]);
                } else {
                    #pragma clang loop unroll(full)
                    for (uint j = 0u; j < 8u; ++j) As[lr][lc + j] = 0.0f;
                }
                {
                    const auto wkr = wk + size_t(n0 + lr) * uint(K) + k0 + lc;
                    #pragma clang loop unroll(full)
                    for (uint j = 0u; j < 8u; ++j) Bs[lc + j][lr] = float(wkr[j]);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                #pragma clang loop unroll(full)
                for (uint ks = 0u; ks < 32u; ks += 8u) {
                    simdgroup_float8x8 a0, a1, b0, b1;
                    simdgroup_load(a0, &As[sg_m][ks], 33);
                    simdgroup_load(a1, &As[sg_m + 8u][ks], 33);
                    simdgroup_load(b0, &Bs[ks][sg_n], 33);
                    simdgroup_load(b1, &Bs[ks][sg_n + 8u], 33);
                    simdgroup_multiply_accumulate(accK[0][0], a0, b0, accK[0][0]);
                    simdgroup_multiply_accumulate(accK[0][1], a0, b1, accK[0][1]);
                    simdgroup_multiply_accumulate(accK[1][0], a1, b0, accK[1][0]);
                    simdgroup_multiply_accumulate(accK[1][1], a1, b1, accK[1][1]);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                {
                    const auto wgr = wg + size_t(n0 + lr) * uint(K) + k0 + lc;
                    #pragma clang loop unroll(full)
                    for (uint j = 0u; j < 8u; ++j) Bs[lc + j][lr] = float(wgr[j]);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                #pragma clang loop unroll(full)
                for (uint ks = 0u; ks < 32u; ks += 8u) {
                    simdgroup_float8x8 a0, a1, b0, b1;
                    simdgroup_load(a0, &As[sg_m][ks], 33);
                    simdgroup_load(a1, &As[sg_m + 8u][ks], 33);
                    simdgroup_load(b0, &Bs[ks][sg_n], 33);
                    simdgroup_load(b1, &Bs[ks][sg_n + 8u], 33);
                    simdgroup_multiply_accumulate(accG[0][0], a0, b0, accG[0][0]);
                    simdgroup_multiply_accumulate(accG[0][1], a0, b1, accG[0][1]);
                    simdgroup_multiply_accumulate(accG[1][0], a1, b0, accG[1][0]);
                    simdgroup_multiply_accumulate(accG[1][1], a1, b1, accG[1][1]);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            simdgroup_store(accK[0][0], &As[sg_m][sg_n], 33);
            simdgroup_store(accK[0][1], &As[sg_m][sg_n + 8u], 33);
            simdgroup_store(accK[1][0], &As[sg_m + 8u][sg_n], 33);
            simdgroup_store(accK[1][1], &As[sg_m + 8u][sg_n + 8u], 33);
            simdgroup_store(accG[0][0], &Bs[sg_m][sg_n], 33);
            simdgroup_store(accG[0][1], &Bs[sg_m][sg_n + 8u], 33);
            simdgroup_store(accG[1][0], &Bs[sg_m + 8u][sg_n], 33);
            simdgroup_store(accG[1][1], &Bs[sg_m + 8u][sg_n + 8u], 33);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (m0 + lr < uint(M)) {
                const size_t outOff = size_t(m0 + lr) * uint(N) + n0 + lc;
                #pragma clang loop unroll(full)
                for (uint j = 0u; j < 8u; ++j) {
                    keys[outOff + j] = As[lr][lc + j];
                    gates[outOff + j] = Bs[lr][lc + j];
                }
            }
            """,
        header: compressorTiledHeader)

    private static let compressorDualFastKernel = MLXFast.metalKernel(
        name: "ernie55_compressor_dual_f32_fast",
        inputNames: ["x", "wk", "wg"],
        outputNames: ["keys", "gates"],
        source: """
            const uint lane = thread_index_in_simdgroup;
            const uint row = threadgroup_position_in_grid.x * 16u + simdgroup_index_in_threadgroup * 4u;
            const uint m = threadgroup_position_in_grid.y;
            const auto xm = x + size_t(m) * uint(K);
            float ak[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            float ag[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (uint base = 0u; base < uint(K); base += 128u) {
                const uint k = base + lane * 4u;
                float xv[4];
                #pragma clang loop unroll(full)
                for (uint j = 0u; j < 4u; ++j) xv[j] = float(xm[k + j]);
                #pragma clang loop unroll(full)
                for (uint r = 0u; r < 4u; ++r) {
                    const size_t woff = size_t(row + r) * uint(K) + k;
                    #pragma clang loop unroll(full)
                    for (uint j = 0u; j < 4u; ++j) {
                        ak[r] += xv[j] * float(wk[woff + j]);
                        ag[r] += xv[j] * float(wg[woff + j]);
                    }
                }
            }
            #pragma clang loop unroll(full)
            for (uint r = 0u; r < 4u; ++r) {
                #pragma clang loop unroll(full)
                for (ushort offset = 16u; offset >= 1u; offset >>= 1u) {
                    ak[r] += simd_shuffle_down(ak[r], offset);
                    ag[r] += simd_shuffle_down(ag[r], offset);
                }
                if (lane == 0u) {
                    const size_t outOff = size_t(m) * uint(N) + row + r;
                    keys[outOff] = ak[r];
                    gates[outOff] = ag[r];
                }
            }
            """)

    private static let compressorDualGeneralKernel = MLXFast.metalKernel(
        name: "ernie55_compressor_dual_f32_gen",
        inputNames: ["x", "wk", "wg"],
        outputNames: ["keys", "gates"],
        source: """
            const uint n = thread_position_in_grid.x;
            const uint m = thread_position_in_grid.y;
            if (n >= uint(N) || m >= uint(M)) return;
            const auto xm = x + size_t(m) * uint(K);
            const auto wkn = wk + size_t(n) * uint(K);
            const auto wgn = wg + size_t(n) * uint(K);
            float ak = 0.0f;
            float ag = 0.0f;
            for (uint k = 0u; k < uint(K); ++k) {
                const float xv = float(xm[k]);
                ak += xv * float(wkn[k]);
                ag += xv * float(wgn[k]);
            }
            const size_t outOff = size_t(m) * uint(N) + n;
            keys[outOff] = ak;
            gates[outOff] = ag;
            """)
    #endif

    static func compressorProject(
        _ input: MLXArray, keyWeight: MLXArray, gateWeight: MLXArray
    ) -> (keys: MLXArray, gates: MLXArray) {
        let batch = input.dim(0)
        let length = input.dim(1)
        let k = input.dim(2)
        let n = keyWeight.dim(0)
        #if canImport(Metal)
        if Device.defaultDevice().deviceType == .gpu,
            length > 0,
            keyWeight.dtype == gateWeight.dtype,
            [.bfloat16, .float16, .float32].contains(input.dtype),
            [.bfloat16, .float16, .float32].contains(keyWeight.dtype)
        {
            let m = batch * length
            let x2 = contiguous(input.reshaped(m, k))
            if m > 8 && k % 32 == 0 && n % 32 == 0 {
                let gridM = (m + 31) / 32
                let gridN = n / 32
                let outs = compressorDualTiledKernel(
                    [x2, keyWeight, gateWeight],
                    template: [("M", m), ("K", k), ("N", n)],
                    grid: (gridN * 128, gridM, 1),
                    threadGroup: (128, 1, 1),
                    outputShapes: [[batch, length, n], [batch, length, n]],
                    outputDTypes: [.float32, .float32])
                return (outs[0], outs[1])
            }
            if k % 128 == 0 && n % 16 == 0 {
                let outs = compressorDualFastKernel(
                    [x2, keyWeight, gateWeight],
                    template: [("K", k), ("N", n)],
                    grid: (n / 16 * 128, m, 1),
                    threadGroup: (128, 1, 1),
                    outputShapes: [[batch, length, n], [batch, length, n]],
                    outputDTypes: [.float32, .float32])
                return (outs[0], outs[1])
            }
            let tgX = min(256, max(1, n))
            let gridX = ((n + tgX - 1) / tgX) * tgX
            let outs = compressorDualGeneralKernel(
                [x2, keyWeight, gateWeight],
                template: [("M", m), ("K", k), ("N", n)],
                grid: (gridX, m, 1),
                threadGroup: (tgX, 1, 1),
                outputShapes: [[batch, length, n], [batch, length, n]],
                outputDTypes: [.float32, .float32])
            return (outs[0], outs[1])
        }
        #endif
        let x = input.asType(.float32)
        return (
            matmul(x, keyWeight.asType(.float32).transposed()),
            matmul(x, gateWeight.asType(.float32).transposed())
        )
    }

    // MARK: - 2. Fused ERNIE mHC (Optional Prev Expand + Mix + 20-iter Sinkhorn + Collapse + RMSNorm)
    //
    // Handles 1...8 rows (AR decode and DSpark verify), 100% row-invariant.
    // ERNIE places normEps OUTSIDE sqrt: rms = sqrt(mean(flat^2)) + normEps.

    #if canImport(Metal)
    private static let hcHeader = """
        template <typename T, int HC, int D, bool X, typename PH, typename PY, typename PP, typename PC>
        inline float ernie_hcs_stream(PH h, PY y, PP post, PC comb, int i) {
          if (!X) return static_cast<float>(h[i]);
          const uint d = uint(i) % D;
          const uint target_hc = uint(i) / D;
          float residual_mix = static_cast<float>(comb[target_hc]) * static_cast<float>(h[d]);
          for (uint source_hc = 1; source_hc < HC; ++source_hc) {
            residual_mix += static_cast<float>(comb[source_hc * HC + target_hc])
                * static_cast<float>(h[source_hc * D + d]);
          }
          const float direct = static_cast<float>(post[target_hc]) * static_cast<float>(y[d]);
          return static_cast<float>(static_cast<T>(residual_mix + direct));
        }
        """

    private static let hcExpandMixKernel = MLXFast.metalKernel(
        name: "ernie55_hc_expand_mix_rows",
        inputNames: ["h", "y", "post", "comb", "fn"],
        outputNames: ["xo", "part"],
        source: """
            constexpr int N = HC * D;
            constexpr int MIX = (2 + HC) * HC;
            const uint j = threadgroup_position_in_grid.x;
            const uint r = threadgroup_position_in_grid.y;
            const uint t = thread_position_in_threadgroup.x;
            const uint lane = thread_index_in_simdgroup;
            const uint sg = simdgroup_index_in_threadgroup;
            threadgroup float red[8][MIX + 1];
            auto hr = h + size_t(r) * N;
            auto yr = y + size_t(r) * D;
            auto pr = post + size_t(r) * HC;
            auto cr = comb + size_t(r) * HC * HC;
            const uint col = 1024u * j + 4u * t;
            float v[4];
            #pragma clang loop unroll(full)
            for (int i = 0; i < 4; i++) {
              v[i] = ernie_hcs_stream<T, HC, D, X>(hr, yr, pr, cr, int(col) + i);
              if (X) xo[size_t(r) * N + col + i] = static_cast<T>(v[i]);
            }
            float acc[MIX + 1];
            acc[MIX] = v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3];
            for (int m = 0; m < MIX; m++) {
              const device float* f = fn + size_t(m) * N + col;
              acc[m] = f[0] * v[0] + f[1] * v[1] + f[2] * v[2] + f[3] * v[3];
            }
            for (int m = 0; m <= MIX; m++) {
              const float s = simd_sum(acc[m]);
              if (lane == 0) red[sg][m] = s;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (t <= uint(MIX)) {
              float s = 0.0f;
              for (int q = 0; q < 8; q++) s += red[q][t];
              part[(size_t(r) * SLICES + j) * (MIX + 1) + t] = s;
            }
            """,
        header: hcHeader)

    private static let hcSinkhornCollapseNormKernel = MLXFast.metalKernel(
        name: "ernie55_hc_sinkhorn_collapse_norm",
        inputNames: ["x", "part", "w", "reps", "invn", "neps", "scale", "base", "eps"],
        outputNames: ["normed", "post_out", "comb"],
        source: """
            constexpr int N = HC * D;
            constexpr int MIX = (2 + HC) * HC;
            const uint r = threadgroup_position_in_grid.y;
            const uint lid = thread_position_in_threadgroup.x;
            const uint lane = thread_index_in_simdgroup;
            const uint sg = simdgroup_index_in_threadgroup;
            threadgroup float tot[MIX + 1];
            threadgroup float t_pre[HC];
            threadgroup float t_sums[32];
            threadgroup float t_inv[1];

            if (lid <= uint(MIX)) {
              float s = 0.0f;
              for (int j = 0; j < SLICES; j++) s += part[(size_t(r) * SLICES + j) * (MIX + 1) + lid];
              tot[lid] = s;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (sg == 0) {
              // ERNIE places normEps OUTSIDE sqrt: rms = sqrt(mean(flat^2)) + neps.
              const float inv_rms = 1.0f / (metal::precise::sqrt(tot[MIX] * invn[0]) + neps[0]);
              const float epsv = static_cast<float>(eps[0]);
              const float s0 = static_cast<float>(scale[0]);
              const float s1 = static_cast<float>(scale[1]);
              const float s2 = static_cast<float>(scale[2]);
              if (lane < uint(HC)) {
                const float zp = (tot[lane] * inv_rms) * s0 + static_cast<float>(base[lane]);
                t_pre[lane] = 1.0f / (1.0f + metal::fast::exp(-zp)) + epsv;
                const float zo = (tot[HC + lane] * inv_rms) * s1 + static_cast<float>(base[HC + lane]);
                post_out[r * HC + lane] = 2.0f / (1.0f + metal::fast::exp(-zo));
              }
              const ushort cell = ushort(lane & 15u);
              const ushort row = cell & ~ushort(3);
              const ushort colk = cell & ushort(3);
              float c = (tot[2 * HC + cell] * inv_rms) * s2 + static_cast<float>(base[2 * HC + cell]);
              float row_max = -INFINITY;
              for (ushort k = 0; k < 4; k++) row_max = metal::max(row_max, simd_shuffle(c, ushort(row + k)));
              c = metal::fast::exp(c - row_max);
              float row_sum = 0.0f;
              for (ushort k = 0; k < 4; k++) row_sum += simd_shuffle(c, ushort(row + k));
              c = c * (1.0f / row_sum) + epsv;
              float col_sum = 0.0f;
              for (ushort k = 0; k < 4; k++) col_sum += simd_shuffle(c, ushort(colk + 4 * k));
              c *= 1.0f / (col_sum + epsv);
              for (int iter = 1; iter < ITERS; ++iter) {
                float rsum = 0.0f;
                for (ushort k = 0; k < 4; k++) rsum += simd_shuffle(c, ushort(row + k));
                c *= 1.0f / (rsum + epsv);
                float csum = 0.0f;
                for (ushort k = 0; k < 4; k++) csum += simd_shuffle(c, ushort(colk + 4 * k));
                c *= 1.0f / (csum + epsv);
              }
              if (lane < 16u) comb[r * 16 + lane] = c;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Collapse 4 streams with t_pre and apply RMSNorm across D = 4096 (1024 threads x 4).
            const device T* xr = x + size_t(r) * N;
            float p0 = t_pre[0], p1 = t_pre[1], p2 = t_pre[2], p3 = t_pre[3];
            T yv[4];
            float acc = 0.0f;
            #pragma clang loop unroll(full)
            for (int i = 0; i < 4; i++) {
              const int d = int(lid) * 4 + i;
              float val = p0 * float(xr[d]) + p1 * float(xr[D + d])
                        + p2 * float(xr[2 * D + d]) + p3 * float(xr[3 * D + d]);
              T y = static_cast<T>(val);
              yv[i] = y;
              float fy = float(y);
              acc += fy * fy;
            }
            acc = simd_sum(acc);
            if (sg == 0) t_sums[lane] = 0.0f;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (lane == 0) t_sums[sg] = acc;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sg == 0) {
              acc = simd_sum(t_sums[lane]);
              if (lane == 0) t_inv[0] = metal::precise::rsqrt(acc / float(D) + reps[0]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const float inv = t_inv[0];
            #pragma clang loop unroll(full)
            for (int i = 0; i < 4; i++) {
              const int d = int(lid) * 4 + i;
              normed[size_t(r) * D + d] = w[d] * static_cast<T>(float(yv[i]) * inv);
            }
            """)
    #endif

    struct FusedHCOutput {
        let normed: MLXArray
        let post: MLXArray
        let comb: MLXArray
        let stream: MLXArray
    }

    static func fusedHCCollapseNorm(
        h: MLXArray, prevOut: MLXArray? = nil, prevPost: MLXArray? = nil, prevComb: MLXArray? = nil,
        fn: MLXArray, scale: MLXArray, base: MLXArray, normWeight: MLXArray,
        iterations: Int, mixingEpsilon: Float, normEpsilon: Float, rmsEpsilon: Float
    ) -> FusedHCOutput? {
        #if canImport(Metal)
        guard Device.defaultDevice().deviceType == .gpu,
            h.ndim == 4, h.dim(0) == 1, (1...16).contains(h.dim(1)),
            h.dim(2) == 4, h.dim(3) == 4096,
            h.dtype == .bfloat16 || h.dtype == .float16,
            fn.dtype == .float32, scale.dtype == .float32, base.dtype == .float32,
            normWeight.dtype == h.dtype
        else { return nil }
        let rows = h.dim(1)
        let hc = 4, d = 4096, mix = 24, slices = (hc * d) / 1024
        let expand = prevOut != nil
        let hf = h.reshaped(rows * hc * d)
        let mixed = hcExpandMixKernel(
            [hf, (prevOut ?? hf).reshaped(-1),
             (prevPost ?? scale).reshaped(-1).asType(.float32),
             (prevComb ?? scale).reshaped(-1).asType(.float32), fn],
            template: [("T", h.dtype), ("HC", hc), ("D", d), ("X", expand), ("SLICES", slices)],
            grid: (256 * slices, rows, 1),
            threadGroup: (256, 1, 1),
            outputShapes: [expand ? [rows, hc, d] : [1], [rows, slices, mix + 1]],
            outputDTypes: [h.dtype, .float32])
        let stream = expand ? mixed[0].reshaped(1, rows, hc, d) : h
        let outs = hcSinkhornCollapseNormKernel(
            [stream.reshaped(-1), mixed[1].reshaped(-1), normWeight,
             scalar(rmsEpsilon), scalar(Float(1.0 / Double(hc * d))), scalar(normEpsilon),
             scale, base, scalar(mixingEpsilon)],
            template: [("T", h.dtype), ("HC", hc), ("D", d), ("SLICES", slices), ("ITERS", iterations)],
            grid: (1024, rows, 1),
            threadGroup: (1024, 1, 1),
            outputShapes: [[1, rows, d], [1, rows, hc], [1, rows, hc, hc]],
            outputDTypes: [h.dtype, .float32, .float32])
        return FusedHCOutput(normed: outs[0], post: outs[1], comb: outs[2], stream: stream)
        #else
        return nil
        #endif
    }

    // MARK: - 2b. Batched Prefill Fused mHC (Sinkhorn + Collapse + RMSNorm in 1 pass over h)
    //
    // Replaces ~90 MLX ops and 4x FP32 casts of [1, L, 16384] during prefill with 1 BF16 GEMM
    // + 1 Metal kernel that reads h[r] once into registers for RMS, 20-iter Sinkhorn, collapse,
    // and output RMSNorm.

    #if canImport(Metal)
    private static let hcBatchedSinkhornCollapseNormKernel = MLXFast.metalKernel(
        name: "ernie55_hc_batched_sinkhorn_collapse_norm",
        inputNames: ["x", "proj", "w", "reps", "invn", "neps", "scale", "base", "eps"],
        outputNames: ["normed", "post_out", "comb"],
        source: """
            constexpr int N = HC * D;
            constexpr int MIX = (2 + HC) * HC;
            const uint r = threadgroup_position_in_grid.y;
            const uint lid = thread_position_in_threadgroup.x;
            const uint lane = thread_index_in_simdgroup;
            const uint sg = simdgroup_index_in_threadgroup;
            threadgroup float t_pre[HC];
            threadgroup float t_sums[32];
            threadgroup float t_inv[1];

            // Load all 4 streams x 4 cols per thread (16 elements/thread x 1024 = 16384) once.
            const device T* xr = x + size_t(r) * N;
            float s0[4], s1[4], s2[4], s3[4];
            float sq = 0.0f;
            #pragma clang loop unroll(full)
            for (int i = 0; i < 4; ++i) {
                const int d = int(lid) * 4 + i;
                s0[i] = float(xr[d]);
                s1[i] = float(xr[D + d]);
                s2[i] = float(xr[2 * D + d]);
                s3[i] = float(xr[3 * D + d]);
                sq += s0[i] * s0[i] + s1[i] * s1[i] + s2[i] * s2[i] + s3[i] * s3[i];
            }
            sq = simd_sum(sq);
            if (sg == 0) t_sums[lane] = 0.0f;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (lane == 0) t_sums[sg] = sq;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (sg == 0) {
                const float tot_sq = simd_sum(t_sums[lane]);
                const float inv_rms = 1.0f / (metal::precise::sqrt(tot_sq * invn[0]) + neps[0]);
                const float epsv = static_cast<float>(eps[0]);
                const float sc0 = static_cast<float>(scale[0]);
                const float sc1 = static_cast<float>(scale[1]);
                const float sc2 = static_cast<float>(scale[2]);
                const device float* pr = proj + size_t(r) * MIX;
                if (lane < uint(HC)) {
                    const float zp = (pr[lane] * inv_rms) * sc0 + static_cast<float>(base[lane]);
                    t_pre[lane] = 1.0f / (1.0f + metal::fast::exp(-zp)) + epsv;
                    const float zo = (pr[HC + lane] * inv_rms) * sc1 + static_cast<float>(base[HC + lane]);
                    post_out[r * HC + lane] = 2.0f / (1.0f + metal::fast::exp(-zo));
                }
                const ushort cell = ushort(lane & 15u);
                const ushort row = cell & ~ushort(3);
                const ushort colk = cell & ushort(3);
                float c = (pr[2 * HC + cell] * inv_rms) * sc2 + static_cast<float>(base[2 * HC + cell]);
                float row_max = -INFINITY;
                for (ushort k = 0; k < 4; k++) row_max = metal::max(row_max, simd_shuffle(c, ushort(row + k)));
                c = metal::fast::exp(c - row_max);
                float row_sum = 0.0f;
                for (ushort k = 0; k < 4; k++) row_sum += simd_shuffle(c, ushort(row + k));
                c = c * (1.0f / row_sum) + epsv;
                float col_sum = 0.0f;
                for (ushort k = 0; k < 4; k++) col_sum += simd_shuffle(c, ushort(colk + 4 * k));
                c *= 1.0f / (col_sum + epsv);
                for (int iter = 1; iter < ITERS; ++iter) {
                    float rsum = 0.0f;
                    for (ushort k = 0; k < 4; k++) rsum += simd_shuffle(c, ushort(row + k));
                    c *= 1.0f / (rsum + epsv);
                    float csum = 0.0f;
                    for (ushort k = 0; k < 4; k++) csum += simd_shuffle(c, ushort(colk + 4 * k));
                    c *= 1.0f / (csum + epsv);
                }
                if (lane < 16u) comb[r * 16 + lane] = c;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            const float p0 = t_pre[0], p1 = t_pre[1], p2 = t_pre[2], p3 = t_pre[3];
            T yv[4];
            float acc = 0.0f;
            #pragma clang loop unroll(full)
            for (int i = 0; i < 4; ++i) {
                T y = static_cast<T>(p0 * s0[i] + p1 * s1[i] + p2 * s2[i] + p3 * s3[i]);
                yv[i] = y;
                float fy = float(y);
                acc += fy * fy;
            }
            acc = simd_sum(acc);
            if (sg == 0) t_sums[lane] = 0.0f;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (lane == 0) t_sums[sg] = acc;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sg == 0) {
                acc = simd_sum(t_sums[lane]);
                if (lane == 0) t_inv[0] = metal::precise::rsqrt(acc / float(D) + reps[0]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const float inv = t_inv[0];
            #pragma clang loop unroll(full)
            for (int i = 0; i < 4; ++i) {
                const int d = int(lid) * 4 + i;
                normed[size_t(r) * D + d] = w[d] * static_cast<T>(float(yv[i]) * inv);
            }
            """)
    #endif

    static func batchedHCCollapseNorm(
        h: MLXArray, fn: MLXArray, scale: MLXArray, base: MLXArray, normWeight: MLXArray,
        iterations: Int, mixingEpsilon: Float, normEpsilon: Float, rmsEpsilon: Float
    ) -> (normed: MLXArray, post: MLXArray, comb: MLXArray)? {
        #if canImport(Metal)
        guard Device.defaultDevice().deviceType == .gpu,
            h.ndim == 4, h.dim(0) == 1, h.dim(1) > 0,
            h.dim(2) == 4, h.dim(3) == 4096,
            h.dtype == .bfloat16 || h.dtype == .float16,
            scale.dtype == .float32, base.dtype == .float32,
            normWeight.dtype == h.dtype
        else { return nil }
        let rows = h.dim(1)
        let hc = 4, d = 4096
        let hFlat = contiguous(h).reshaped(rows, hc * d)
        let proj = matmul(hFlat, fn.asType(h.dtype).transposed()).asType(.float32)
        let outs = hcBatchedSinkhornCollapseNormKernel(
            [hFlat.reshaped(-1), contiguous(proj).reshaped(-1), normWeight,
             scalar(rmsEpsilon), scalar(Float(1.0 / Double(hc * d))), scalar(normEpsilon),
             scale, base, scalar(mixingEpsilon)],
            template: [("T", h.dtype), ("HC", hc), ("D", d), ("ITERS", iterations)],
            grid: (1024, rows, 1),
            threadGroup: (1024, 1, 1),
            outputShapes: [[1, rows, d], [1, rows, hc], [1, rows, hc, hc]],
            outputDTypes: [h.dtype, .float32, .float32])
        return (outs[0], outs[1], outs[2])
        #else
        return nil
        #endif
    }

    // MARK: - 3. Fused ERNIE Dual-Weight Router GEMV + Top-K (1...8 rows)
    //
    // Reads x once and both BF16 [512, 4096] router weights directly without FP32 casts
    // (-1.41 GB/tok traffic), then selects exact top-10 in a single 512-thread kernel.

    #if canImport(Metal)
    private static let routerDualGemvKernel = MLXFast.metalKernel(
        name: "ernie55_router_dual_bf16_gemv",
        inputNames: ["x", "w1", "w2"],
        outputNames: ["scores"],
        source: """
            const uint lane = thread_index_in_simdgroup;
            const uint row = threadgroup_position_in_grid.x * 16u + simdgroup_index_in_threadgroup * 4u;
            const uint r_in = threadgroup_position_in_grid.y;
            const auto xr = x + size_t(r_in) * uint(K);
            float a1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            float a2[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (uint base = 0u; base < uint(K); base += 128u) {
                const uint k = base + lane * 4u;
                float v[4];
                #pragma clang loop unroll(full)
                for (uint j = 0u; j < 4u; ++j) v[j] = float(xr[k + j]);
                #pragma clang loop unroll(full)
                for (uint r = 0u; r < 4u; ++r) {
                    const size_t woff = size_t(row + r) * uint(K) + k;
                    #pragma clang loop unroll(full)
                    for (uint j = 0u; j < 4u; ++j) {
                        a1[r] += float(w1[woff + j]) * v[j];
                        a2[r] += float(w2[woff + j]) * v[j];
                    }
                }
            }
            #pragma clang loop unroll(full)
            for (uint r = 0u; r < 4u; ++r) {
                #pragma clang loop unroll(full)
                for (ushort offset = 16u; offset >= 1u; offset >>= 1u) {
                    a1[r] += simd_shuffle_down(a1[r], offset);
                    a2[r] += simd_shuffle_down(a2[r], offset);
                }
                if (lane == 0u) {
                    const float s1 = 1.0f / (1.0f + metal::precise::exp(-a1[r]));
                    const float s2 = 1.0f / (1.0f + metal::precise::exp(-a2[r]));
                    scores[size_t(r_in) * uint(E) + row + r] = s1 + s2;
                }
            }
            """)

    private static let routerTopKKernel = MLXFast.metalKernel(
        name: "ernie55_router_topk",
        inputNames: ["scores", "correction", "expertScale"],
        outputNames: ["indices", "weights"],
        source: """
            const uint tid = thread_position_in_threadgroup.x;
            const uint lane = thread_index_in_simdgroup;
            const uint sg = simdgroup_index_in_threadgroup;
            const uint r = threadgroup_position_in_grid.y;
            constexpr uint NSG = E / 32;
            threadgroup float sg_max[NSG];
            threadgroup uint sg_idx[NSG];
            threadgroup uint best_idx_shared[TOPK];
            threadgroup float best_score_shared[TOPK];

            const float raw_score = scores[size_t(r) * uint(E) + tid];
            float biased = raw_score + float(correction[tid]);
            bool alive = true;

            for (uint k = 0u; k < uint(TOPK); ++k) {
                float val = alive ? biased : -INFINITY;
                uint idx = tid;
                #pragma clang loop unroll(full)
                for (ushort off = 16u; off >= 1u; off >>= 1u) {
                    float ov = simd_shuffle_down(val, off);
                    uint oi = simd_shuffle_down(idx, off);
                    if (ov > val || (ov == val && oi < idx)) { val = ov; idx = oi; }
                }
                if (lane == 0u) { sg_max[sg] = val; sg_idx[sg] = idx; }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (sg == 0u) {
                    float v = lane < NSG ? sg_max[lane] : -INFINITY;
                    uint i = lane < NSG ? sg_idx[lane] : uint(E);
                    #pragma clang loop unroll(full)
                    for (ushort off = 16u; off >= 1u; off >>= 1u) {
                        float ov = simd_shuffle_down(v, off);
                        uint oi = simd_shuffle_down(i, off);
                        if (ov > v || (ov == v && oi < i)) { v = ov; i = oi; }
                    }
                    if (lane == 0u) best_idx_shared[k] = i;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                const uint winner = best_idx_shared[k];
                if (tid == winner) {
                    alive = false;
                    best_score_shared[k] = raw_score;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (tid < uint(TOPK)) {
                float sum = 0.0f;
                if (NORMALIZE) {
                    for (uint k = 0u; k < uint(TOPK); ++k) sum += best_score_shared[k];
                    sum = metal::max(sum, 1e-20f);
                } else {
                    sum = 1.0f;
                }
                const uint e = best_idx_shared[tid];
                indices[size_t(r) * uint(TOPK) + tid] = e;
                weights[size_t(r) * uint(TOPK) + tid] = (best_score_shared[tid] / sum) * float(expertScale[e]);
            }
            """)
    #endif

    static func routeFused(
        _ input: MLXArray, first: MLXArray, second: MLXArray, correction: MLXArray,
        expertScale: MLXArray, count: Int, normalize: Bool
    ) -> (indices: MLXArray, weights: MLXArray)? {
        #if canImport(Metal)
        guard Device.defaultDevice().deviceType == .gpu,
            input.ndim == 3, input.dim(0) == 1, (1...16).contains(input.dim(1)),
            first.dtype == .bfloat16, second.dtype == .bfloat16,
            first.shape == second.shape,
            first.dim(0) == 512, first.dim(1) % 128 == 0,
            input.dim(2) == first.dim(1), count > 0, count <= 32
        else { return nil }
        let rows = input.dim(1)
        let experts = first.dim(0)
        let hidden = first.dim(1)
        let scores = routerDualGemvKernel(
            [input.reshaped(rows, hidden), first, second],
            template: [("K", hidden), ("E", experts)],
            grid: (experts / 16 * 128, rows, 1),
            threadGroup: (128, 1, 1),
            outputShapes: [[rows, experts]],
            outputDTypes: [.float32])[0]
        let picked = routeTopK(
            scores, correction: correction, expertScale: expertScale, count: count, normalize: normalize)!
        return (picked.indices.reshaped(1, rows, count), picked.weights.reshaped(1, rows, count))
        #else
        return nil
        #endif
    }

    #if canImport(Metal)
    /// Single-dispatch exact top-K over 512 experts (256 threads / 8 simdgroups, 2 experts per thread).
    /// Handles both decode/verify (T=1..16) and prefill (T up to 65535).
    private static let multiRouterTopKKernel = MLXFast.metalKernel(
        name: "ernie55_multi_router_topk_256",
        inputNames: ["scores", "correction", "expertScale"],
        outputNames: ["indices", "weights"],
        source: """
            const uint tid  = thread_position_in_threadgroup.x;
            const uint sg   = tid >> 5;
            const uint lane = tid & 31u;
            const uint row  = threadgroup_position_in_grid.y;

            const device float* s_row = scores + size_t(row) * 512u;
            float r0 = s_row[tid];
            float r1 = s_row[tid + 256u];
            float v0 = r0 + float(correction[tid]);
            float v1 = r1 + float(correction[tid + 256u]);

            threadgroup float sg_val[8];
            threadgroup uint  sg_idx[8];
            threadgroup uint  win_idx_sh;
            threadgroup uint  picked_idx[TOPK];
            threadgroup float picked_raw[TOPK];

            for (uint k = 0u; k < uint(TOPK); ++k) {
                float bv = (v0 >= v1) ? v0 : v1;
                uint  bi = (v0 >= v1) ? tid : (tid + 256u);
                float sm = simd_max(bv);
                uint  si = simd_min((bv == sm) ? bi : 0xFFFFFFFFu);
                if (lane == 0u) {
                    sg_val[sg] = sm;
                    sg_idx[sg] = si;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (tid == 0u) {
                    float best_v = sg_val[0];
                    uint  best_i = sg_idx[0];
                    for (uint s = 1u; s < 8u; ++s) {
                        if (sg_val[s] > best_v || (sg_val[s] == best_v && sg_idx[s] < best_i)) {
                            best_v = sg_val[s];
                            best_i = sg_idx[s];
                        }
                    }
                    win_idx_sh = best_i;
                    picked_idx[k] = best_i;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                uint w = win_idx_sh;
                if (tid == w) {
                    picked_raw[k] = r0;
                    v0 = -INFINITY;
                } else if (tid + 256u == w) {
                    picked_raw[k] = r1;
                    v1 = -INFINITY;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sg == 0u) {
                float raw = (lane < uint(TOPK)) ? picked_raw[lane] : 0.0f;
                float denom = metal::max(simd_sum(raw), 1e-20f);
                if (lane < uint(TOPK)) {
                    uint id = picked_idx[lane];
                    float w = NORMALIZE ? (raw / denom) : raw;
                    indices[size_t(row) * uint(TOPK) + lane] = id;
                    weights[size_t(row) * uint(TOPK) + lane] = w * float(expertScale[id]);
                }
            }
            """)
    #endif

    static func routeTopK(
        _ scores: MLXArray, correction: MLXArray, expertScale: MLXArray,
        count: Int, normalize: Bool
    ) -> (indices: MLXArray, weights: MLXArray)? {
        #if canImport(Metal)
        guard Device.defaultDevice().deviceType == .gpu,
            scores.dim(-1) == 512, count > 0, count <= 32, scores.dtype == .float32
        else { return nil }
        let lead = Array(scores.shape.dropLast())
        let rows = scores.size / 512
        guard rows >= 1 && rows <= 65535 else { return nil }
        let picked = multiRouterTopKKernel(
            [contiguous(scores.reshaped(rows, 512)),
             contiguous(correction.asType(.float32)),
             contiguous(expertScale.asType(.float32))],
            template: [("TOPK", count), ("NORMALIZE", normalize)],
            grid: (256, rows, 1),
            threadGroup: (256, 1, 1),
            outputShapes: [[rows, count], [rows, count]],
            outputDTypes: [.uint32, .float32])
        return (picked[0].reshaped(lead + [count]), picked[1].reshaped(lead + [count]))
        #else
        return nil
        #endif
    }

    // MARK: - 3b. Fused LatentMoE: Shared SiTU + Latent RMSNorm + Concat

    #if canImport(Metal)
    private static let sharedSiTULatentNormConcatKernel = MLXFast.metalKernel(
        name: "ernie55_shared_situ_latent_norm_concat",
        inputNames: ["proj", "comb", "norm_w", "floats"],
        outputNames: ["out"],
        source: """
            const uint tid  = thread_position_in_threadgroup.x;
            const uint sg   = tid >> 5;
            const uint lane = tid & 31u;
            const uint row  = threadgroup_position_in_grid.y;

            const float beta = floats[0];
            const float lbeta = floats[1];
            const float norm_eps = floats[2];

            const device T* p_row = proj + size_t(row) * uint(P_DIM);
            const device T* c_row = comb + size_t(row) * uint(L_DIM);
            device T* o_row = out + size_t(row) * uint(S_DIM + L_DIM);

            for (uint i = tid; i < uint(S_DIM); i += 256u) {
                float g = float(p_row[i]);
                float u = float(p_row[uint(S_DIM) + i]);
                float sig = 1.0f / (1.0f + metal::precise::exp(-g));
                float g_act = beta * metal::precise::tanh(g / beta) * sig;
                float u_act = lbeta * metal::precise::tanh(u / lbeta);
                o_row[i] = T(g_act * u_act);
            }

            threadgroup float sg_sq[8];
            float sq = 0.0f;
            for (uint j = tid; j < uint(L_DIM); j += 256u) {
                float v = float(c_row[j]);
                sq += v * v;
            }
            float sg_s = simd_sum(sq);
            if (lane == 0u) sg_sq[sg] = sg_s;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float total_sq = sg_sq[0] + sg_sq[1] + sg_sq[2] + sg_sq[3]
                           + sg_sq[4] + sg_sq[5] + sg_sq[6] + sg_sq[7];
            float inv_rms = metal::precise::rsqrt(total_sq / float(L_DIM) + norm_eps);
            for (uint j = tid; j < uint(L_DIM); j += 256u) {
                float v = float(c_row[j]);
                o_row[uint(S_DIM) + j] = T(v * inv_rms * float(norm_w[j]));
            }
            """)
    #endif

    static func sharedSiTULatentNormConcat(
        proj: MLXArray, combined: MLXArray, normWeight: MLXArray,
        sharedWidth: Int, gateCap: Float, upCap: Float, normEps: Float
    ) -> MLXArray? {
        #if canImport(Metal)
        guard Device.defaultDevice().deviceType == .gpu,
            proj.dtype == combined.dtype,
            [.bfloat16, .float16, .float32].contains(combined.dtype)
        else { return nil }
        let pDim = proj.dim(-1)
        let lDim = combined.dim(-1)
        guard pDim >= 2 * sharedWidth, sharedWidth > 0, lDim > 0 else { return nil }
        let rows = combined.size / lDim
        guard rows >= 1 && rows <= 65535 else { return nil }
        let lead = Array(combined.shape.dropLast())
        let nw = normWeight.dtype == combined.dtype ? normWeight : normWeight.asType(combined.dtype)
        let floats = MLXArray([gateCap, upCap, normEps])
        let out = sharedSiTULatentNormConcatKernel(
            [contiguous(proj.reshaped(rows, pDim)), contiguous(combined.reshaped(rows, lDim)),
             contiguous(nw), floats],
            template: [("T", combined.dtype), ("P_DIM", pDim), ("S_DIM", sharedWidth), ("L_DIM", lDim)],
            grid: (256, rows, 1),
            threadGroup: (256, 1, 1),
            outputShapes: [[rows, sharedWidth + lDim]],
            outputDTypes: [combined.dtype])[0]
        return out.reshaped(lead + [sharedWidth + lDim])
        #else
        return nil
        #endif
    }

    // MARK: - 3c. Native MXFP8 Linear / Row Projection

    static func releaseMXFP8(_ linear: QuantizedLinear) {
        linear.update(parameters: ModuleParameters.unflattened([
            "weight": MLXArray.zeros([1, 1], dtype: .uint32),
            "scales": MLXArray.zeros([1, 1], dtype: .uint8),
        ]))
    }

    static func mxfp8Linear(_ linear: Linear, _ x: MLXArray) -> MLXArray {
        if let q = linear as? QuantizedLinear, q.mode == .mxfp8, q.bias == nil {
            return quantizedMM(
                x, q.weight, scales: q.scales, biases: nil,
                transpose: true, groupSize: 32, bits: 8, mode: .mxfp8)
        }
        return linear(x)
    }

    static func mxfp8Rows(_ x: MLXArray, weight: MLXArray, scales: MLXArray) -> MLXArray? {
        guard weight.dtype == .uint32, scales.dtype == .uint8 else { return nil }
        return quantizedMM(
            x, weight, scales: scales, biases: nil,
            transpose: true, groupSize: 32, bits: 8, mode: .mxfp8)
    }

    // MARK: - 4. Fused BF16 LM Head + MultiMax (1...8 rows)
    //
    // Keeps lm_head.weight in BF16 (1.65 GB vs 3.30 GB in FP32), reads weight ONCE across
    // all R verify rows, accumulates in exact FP32, and fuses MultiMax in the epilogue.

    #if canImport(Metal)
    private static let lmHeadBF16MultiMaxKernel = MLXFast.metalKernel(
        name: "ernie55_lm_head_bf16_multimax",
        inputNames: ["x", "w", "ranges", "slopes"],
        outputNames: ["logits"],
        source: """
            const uint lane = thread_index_in_simdgroup;
            const uint v0 = threadgroup_position_in_grid.x * 16u + simdgroup_index_in_threadgroup * 4u;
            float acc[ROWS][4];
            #pragma clang loop unroll(full)
            for (uint r = 0u; r < uint(ROWS); ++r)
                for (uint o = 0u; o < 4u; ++o) acc[r][o] = 0.0f;

            for (uint base = 0u; base < uint(K); base += 128u) {
                const uint k = base + lane * 4u;
                float xv[ROWS][4];
                #pragma clang loop unroll(full)
                for (uint r = 0u; r < uint(ROWS); ++r) {
                    const auto xr = x + size_t(r) * uint(K) + k;
                    #pragma clang loop unroll(full)
                    for (uint j = 0u; j < 4u; ++j) xv[r][j] = float(xr[j]);
                }
                #pragma clang loop unroll(full)
                for (uint o = 0u; o < 4u; ++o) {
                    if (v0 + o >= uint(V)) continue;
                    const auto wo = w + size_t(v0 + o) * uint(K) + k;
                    float wv[4];
                    #pragma clang loop unroll(full)
                    for (uint j = 0u; j < 4u; ++j) wv[j] = float(wo[j]);
                    #pragma clang loop unroll(full)
                    for (uint r = 0u; r < uint(ROWS); ++r) {
                        #pragma clang loop unroll(full)
                        for (uint j = 0u; j < 4u; ++j) acc[r][o] += xv[r][j] * wv[j];
                    }
                }
            }
            const float r0 = float(ranges[0]), r1 = float(ranges[1]);
            const float r2 = float(ranges[2]), r3 = float(ranges[3]);
            const float s0 = float(slopes[0]), s1 = float(slopes[1]);
            const float s2 = float(slopes[2]), s3 = float(slopes[3]);
            #pragma clang loop unroll(full)
            for (uint r = 0u; r < uint(ROWS); ++r) {
                #pragma clang loop unroll(full)
                for (uint o = 0u; o < 4u; ++o) {
                    float val = acc[r][o];
                    #pragma clang loop unroll(full)
                    for (ushort off = 16u; off >= 1u; off >>= 1u) val += simd_shuffle_down(val, off);
                    if (lane == 0u && v0 + o < uint(V)) {
                        if (MULTIMAX) {
                            const float low = metal::max(r2 - val, 0.0f);
                            const float high = metal::max(val - r3, 0.0f);
                            const float mm = val
                                + s0 * metal::max(r0 - val, 0.0f)
                                + s1 * metal::max(val - r1, 0.0f)
                                + s2 * (low * low)
                                + s3 * (high * high);
                            logits[size_t(r) * uint(V) + v0 + o] = mm;
                        } else {
                            logits[size_t(r) * uint(V) + v0 + o] = val;
                        }
                    }
                }
            }
            """)
    #endif

    static func lmHeadMultiMax(
        _ hidden: MLXArray, weight: MLXArray, ranges: MLXArray, slopes: MLXArray,
        applyMultiMax: Bool = true
    ) -> MLXArray? {
        #if canImport(Metal)
        guard Device.defaultDevice().deviceType == .gpu,
            hidden.ndim == 3, hidden.dim(0) == 1, (1...16).contains(hidden.dim(1)),
            weight.ndim == 2, weight.dtype == .bfloat16,
            hidden.dim(2) == weight.dim(1), weight.dim(1) % 128 == 0
        else { return nil }
        let rows = hidden.dim(1)
        let vocab = weight.dim(0)
        let hiddenDim = weight.dim(1)
        let blocks = (vocab + 15) / 16
        if rows <= 8 {
            return lmHeadBF16MultiMaxKernel(
                [contiguous(hidden.reshaped(rows, hiddenDim)), weight, ranges, slopes],
                template: [("ROWS", rows), ("K", hiddenDim), ("V", vocab), ("MULTIMAX", applyMultiMax)],
                grid: (blocks * 128, 1, 1),
                threadGroup: (128, 1, 1),
                outputShapes: [[1, rows, vocab]],
                outputDTypes: [.float32])[0]
        }
        let first = lmHeadBF16MultiMaxKernel(
            [contiguous(hidden[0, ..<8, 0...]), weight, ranges, slopes],
            template: [("ROWS", 8), ("K", hiddenDim), ("V", vocab), ("MULTIMAX", applyMultiMax)],
            grid: (blocks * 128, 1, 1),
            threadGroup: (128, 1, 1),
            outputShapes: [[1, 8, vocab]],
            outputDTypes: [.float32])[0]
        let rest = rows - 8
        let second = lmHeadBF16MultiMaxKernel(
            [contiguous(hidden[0, 8..., 0...]), weight, ranges, slopes],
            template: [("ROWS", rest), ("K", hiddenDim), ("V", vocab), ("MULTIMAX", applyMultiMax)],
            grid: (blocks * 128, 1, 1),
            threadGroup: (128, 1, 1),
            outputShapes: [[1, rest, vocab]],
            outputDTypes: [.float32])[0]
        return concatenated([first, second], axis: 1)
        #else
        return nil
        #endif
    }

    // MARK: - 5. Fused Inverse Partial RoPE + Precomputed VHA Mix + Group Transpose (HCA, 1...8 rows)
    //
    // Replaces inverse RoPE + transpose + VHA matmul + reshape + transpose into [O_GROUPS, R, 4096]
    // in one Metal launch (saves ~300 MLX ops/token across 37 HCA layers).

    #if canImport(Metal)
    private static let invRoPEVHAGroupKernel = MLXFast.metalKernel(
        name: "ernie55_inv_rope_vha_group",
        inputNames: ["attended", "cosine", "sine", "vhaMix"],
        outputNames: ["grouped"],
        source: """
            // attended: [1, 64, R, 512] -> grouped: [8, R, 4096]
            // Grid: (256, 16, R), TG: (256, 1, 1). Each TG owns 32 feature cols d0..d0+31 of one row r.
            const uint tid = thread_position_in_threadgroup.x;
            const uint d0 = threadgroup_position_in_grid.y * 32u;
            const uint r = threadgroup_position_in_grid.z;
            threadgroup float buf[64][32];
            const bool is_rope = d0 >= (512u - uint(ROPE));

            // Phase 1: 256 threads load 64 x 32 = 2048 elements (8 per thread: 4 pairs if rope).
            for (uint idx = tid; idx < 1024u; idx += 256u) {
                const uint h = idx / 16u;
                const uint pair = idx % 16u;
                const uint c = pair * 2u;
                const size_t base = (size_t(h) * uint(ROWS) + r) * 512u + d0 + c;
                const float x0 = float(attended[base]);
                const float x1 = float(attended[base + 1u]);
                if (is_rope) {
                    const uint phase = (d0 + c - (512u - uint(ROPE))) / 2u;
                    const float co = float(cosine[size_t(r) * (uint(ROPE) / 2u) + phase]);
                    const float si = -float(sine[size_t(r) * (uint(ROPE) / 2u) + phase]);
                    const float r0 = float(T(x0 * co)) - float(T(x1 * si));
                    const float r1 = float(T(x0 * si)) + float(T(x1 * co));
                    buf[h][c] = float(T(r0));
                    buf[h][c + 1u] = float(T(r1));
                } else {
                    buf[h][c] = x0;
                    buf[h][c + 1u] = x1;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Phase 2: 256 threads compute 64 x 32 = 2048 outputs of vhaMix[64, 64] @ buf[64, 32]
            // and scatter directly to grouped[g, r, local_h * 512 + d0 + c].
            for (uint idx = tid; idx < 2048u; idx += 256u) {
                const uint h_out = idx / 32u;
                const uint c = idx % 32u;
                const auto mrow = vhaMix + size_t(h_out) * 64u;
                float sum = 0.0f;
                for (uint h_in = 0u; h_in < 64u; ++h_in) {
                    sum += float(mrow[h_in]) * buf[h_in][c];
                }
                const uint g = h_out / 8u;
                const uint lh = h_out % 8u;
                grouped[(size_t(g) * uint(ROWS) + r) * 4096u + lh * 512u + d0 + c] = T(sum);
            }
            """)
    #endif

    static func invRoPEVHAGroup(
        _ attended: MLXArray, cos: MLXArray, sin: MLXArray, vhaMix: MLXArray, ropeWidth: Int
    ) -> MLXArray? {
        #if canImport(Metal)
        guard Device.defaultDevice().deviceType == .gpu,
            attended.ndim == 4, attended.dim(0) == 1, attended.dim(1) == 64,
            (1...16).contains(attended.dim(2)), attended.dim(3) == 512,
            ropeWidth > 0, ropeWidth <= 512, ropeWidth % 32 == 0,
            attended.dtype == .bfloat16 || attended.dtype == .float16
        else { return nil }
        let rows = attended.dim(2)
        return invRoPEVHAGroupKernel(
            [contiguous(attended), contiguous(cos.asType(attended.dtype)),
             contiguous(sin.asType(attended.dtype)), contiguous(vhaMix.asType(attended.dtype))],
            template: [("T", attended.dtype), ("ROWS", rows), ("ROPE", ropeWidth)],
            grid: (256, 16, rows),
            threadGroup: (256, 1, 1),
            outputShapes: [[8, rows, 4096]],
            outputDTypes: [attended.dtype])[0]
        #else
        return nil
        #endif
    }

    // MARK: - 6. Fused SiTU Activation (Shared Experts + Dense Layer 0)

    #if canImport(Metal)
    private static let situKernel = MLXFast.metalKernel(
        name: "ernie55_fused_situ",
        inputNames: ["gate", "up", "gateCap", "upCap"],
        outputNames: ["out"],
        source: """
            const uint i = thread_position_in_grid.x;
            if (i >= uint(N)) return;
            const float g = float(gate[i]);
            const float u = float(up[i]);
            const float gc = gateCap[0];
            const float uc = upCap[0];
            const float gv = gc * metal::precise::tanh(g / gc) * (1.0f / (1.0f + metal::precise::exp(-g)));
            const float uv = uc * metal::precise::tanh(u / uc);
            out[i] = T(gv * uv);
            """)
    #endif

    static func fusedSiTU(_ gate: MLXArray, _ up: MLXArray, gateCap: Float, upCap: Float) -> MLXArray? {
        #if canImport(Metal)
        guard Device.defaultDevice().deviceType == .gpu,
            gate.shape == up.shape, gate.dtype == up.dtype, gate.size > 0,
            [.bfloat16, .float16, .float32].contains(gate.dtype)
        else { return nil }
        let n = gate.size
        let tg = min(256, n)
        let grid = ((n + tg - 1) / tg) * tg
        return situKernel(
            [contiguous(gate), contiguous(up), scalar(gateCap), scalar(upCap)],
            template: [("T", gate.dtype), ("N", n)],
            grid: (grid, 1, 1),
            threadGroup: (tg, 1, 1),
            outputShapes: [gate.shape],
            outputDTypes: [gate.dtype])[0]
        #else
        return nil
        #endif
    }

    // MARK: - 7. Fused HCA Decode Q/K Multi-Row Norm + Partial RoPE (1...8 rows)

    #if canImport(Metal)
    private static let decodeQKRosKernel = MLXFast.metalKernel(
        name: "ernie55_hca_decode_qk_rows",
        inputNames: ["query", "key", "key_weight", "cosine", "sine", "eps"],
        outputNames: ["q_out", "k_out"],
        source: """
            const uint row = threadgroup_position_in_grid.x;
            const uint r = threadgroup_position_in_grid.y;
            const uint lane = thread_index_in_simdgroup;
            const uint simd_row = simdgroup_index_in_threadgroup;
            const uint col = 4u * thread_position_in_threadgroup.x;
            const bool is_key = row == Heads;
            const device DType* values = is_key
                ? (key + size_t(r) * 512u)
                : (query + (size_t(r) * Heads + row) * 512u);
            float x[4];
            float sq = 0.0f;
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 4; ++i) {
                x[i] = float(values[col + i]);
                sq += x[i] * x[i];
            }
            threadgroup float sums[4];
            threadgroup float inv;
            const float partial = simd_sum(sq);
            if (lane == 0) sums[simd_row] = partial;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simd_row == 0) {
                const float total = simd_sum(lane < 4 ? sums[lane] : 0.0f);
                if (lane == 0) inv = metal::precise::rsqrt(total / 512.0f + eps[0]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            device DType* out = is_key
                ? (k_out + size_t(r) * 512u)
                : (q_out + (size_t(row) * uint(ROWS) + r) * 512u);
            DType norm[4];
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 4; ++i) {
                const DType n = DType(x[i] * inv);
                norm[i] = is_key ? DType(float(n) * float(key_weight[col + i])) : n;
            }
            if (col >= 512u - Rope) {
                const uint ropeHalf = Rope / 2u;
                #pragma clang loop unroll(full)
                for (uint k = 0; k < 4; k += 2) {
                    const uint phase = (col + k - (512u - Rope)) / 2u;
                    const float c = float(cosine[size_t(r) * ropeHalf + phase]);
                    const float s = float(sine[size_t(r) * ropeHalf + phase]);
                    const float x0 = float(norm[k]);
                    const float x1 = float(norm[k + 1]);
                    out[col + k] = DType(float(DType(x0 * c)) - float(DType(x1 * s)));
                    out[col + k + 1] = DType(float(DType(x0 * s)) + float(DType(x1 * c)));
                }
            } else {
                #pragma clang loop unroll(full)
                for (uint i = 0; i < 4; ++i) out[col + i] = norm[i];
            }
            """)
    #endif

    static func prepareHCARows(
        _ query: MLXArray, rawKV: MLXArray, keyWeight: MLXArray,
        cos: MLXArray, sin: MLXArray, heads: Int, ropeWidth: Int, epsilon: Float
    ) -> (MLXArray, MLXArray)? {
        #if canImport(Metal)
        guard Device.defaultDevice().deviceType == .gpu,
            query.dim(0) == 1, (1...16).contains(query.dim(1)),
            query.dim(3) == 512, ropeWidth > 0, ropeWidth <= 512, ropeWidth.isMultiple(of: 4),
            query.dtype == rawKV.dtype
        else { return nil }
        let rows = query.dim(1)
        let out = decodeQKRosKernel(
            [contiguous(query), contiguous(rawKV), keyWeight.asType(rawKV.dtype),
             contiguous(cos.asType(query.dtype)), contiguous(sin.asType(query.dtype)), scalar(epsilon)],
            template: [("DType", query.dtype), ("Heads", heads), ("Rope", ropeWidth), ("ROWS", rows)],
            grid: ((heads + 1) * 128, rows, 1),
            threadGroup: (128, 1, 1),
            outputShapes: [[1, heads, rows, 512], [1, 1, rows, 512]],
            outputDTypes: [query.dtype, rawKV.dtype])
        return (out[0], out[1])
        #else
        return nil
        #endif
    }

    #if canImport(Metal)
    private static let dsparkMarkovStage1Kernel = MLXFast.metalKernel(
        name: "ernie55_dspark_markov_s1",
        inputNames: ["base_row", "w2", "trans", "ranges", "ts"],
        outputNames: ["blk_val", "blk_idx"],
        source: """
            const uint tid = thread_position_in_threadgroup.x;
            const uint lane = thread_index_in_simdgroup;
            const uint sg = simdgroup_index_in_threadgroup;
            const uint tg = threadgroup_position_in_grid.x;
            const uint n_tg = threadgroups_per_grid.x;

            threadgroup float t_smem[256];
            t_smem[tid] = float(trans[tid]);
            threadgroup float sg_val[8];
            threadgroup uint  sg_idx[8];
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float tv0 = t_smem[lane * 8u + 0u];
            float tv1 = t_smem[lane * 8u + 1u];
            float tv2 = t_smem[lane * 8u + 2u];
            float tv3 = t_smem[lane * 8u + 3u];
            float tv4 = t_smem[lane * 8u + 4u];
            float tv5 = t_smem[lane * 8u + 5u];
            float tv6 = t_smem[lane * 8u + 6u];
            float tv7 = t_smem[lane * 8u + 7u];

            const float r0 = ranges[0], r1 = ranges[1], r2 = ranges[2], r3 = ranges[3];
            const float t0 = ts[0], t1 = ts[1], t2 = ts[2], t3 = ts[3];

            float best_v = -1e38f;
            uint best_i = 0u;
            const uint global_sg = tg * 8u + sg;
            const uint stride_sg = n_tg * 8u;

            for (uint v = global_sg; v < uint(V); v += stride_sg) {
                const device bfloat16_t* wp = w2 + size_t(v) * 256u + lane * 8u;
                float4 w_lo = float4(*(const device vec<bfloat16_t, 4>*)(wp));
                float4 w_hi = float4(*(const device vec<bfloat16_t, 4>*)(wp + 4u));
                float d = w_lo.x * tv0 + w_lo.y * tv1 + w_lo.z * tv2 + w_lo.w * tv3
                        + w_hi.x * tv4 + w_hi.y * tv5 + w_hi.z * tv6 + w_hi.w * tv7;
                float dot_v = simd_sum(d);
                if (lane == 0u) {
                    float val = base_row[v] + float(bfloat16_t(dot_v));
                    float a = metal::max(r0 - val, 0.0f);
                    float b = metal::max(val - r1, 0.0f);
                    float c = metal::max(r2 - val, 0.0f);
                    float e = metal::max(val - r3, 0.0f);
                    val = val + t0 * a + t1 * b + t2 * (c * c) + t3 * (e * e);
                    if (val > best_v) {
                        best_v = val;
                        best_i = v;
                    }
                }
            }
            if (lane == 0u) {
                sg_val[sg] = best_v;
                sg_idx[sg] = best_i;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid == 0u) {
                float bv = sg_val[0];
                uint bi = sg_idx[0];
                for (uint s = 1u; s < 8u; s++) {
                    if (sg_val[s] > bv || (sg_val[s] == bv && sg_idx[s] < bi)) {
                        bv = sg_val[s];
                        bi = sg_idx[s];
                    }
                }
                blk_val[tg] = bv;
                blk_idx[tg] = bi;
            }
            """)
    #endif

    /// Single-pass fused argmax_v MultiMax(baseRow[v] + (markovHead @ transition)[v]).
    static func dsparkMarkovArgmax(
        baseRow: MLXArray, markovHead: MLXArray, transition: MLXArray,
        ranges: MLXArray, ts: MLXArray
    ) -> MLXArray? {
        #if canImport(Metal)
        guard Device.defaultDevice().deviceType == .gpu,
            markovHead.ndim == 2, markovHead.dim(1) == 256,
            markovHead.dtype == .bfloat16,
            baseRow.ndim == 1, baseRow.dim(0) == markovHead.dim(0)
        else { return nil }
        let vocab = markovHead.dim(0)
        let nTG = 512
        let out = dsparkMarkovStage1Kernel(
            [contiguous(baseRow.asType(.float32)), contiguous(markovHead),
             contiguous(transition.asType(.bfloat16)),
             contiguous(ranges.asType(.float32)), contiguous(ts.asType(.float32))],
            template: [("V", vocab)],
            grid: (nTG * 256, 1, 1),
            threadGroup: (256, 1, 1),
            outputShapes: [[nTG], [nTG]],
            outputDTypes: [.float32, .uint32])
        let winner = argMax(out[0])
        return out[1][winner].asType(.int32)
        #else
        return nil
        #endif
    }

    // MARK: Sparse MLA Long-Context Prefill (O(1) memory indexer + sparse absorbed MLA up to 1M+)

    #if canImport(Metal)
    /// Fused MLA indexer scores: `scores[b, 0, t, c] = sum_h w[b, t, h] * max(0, dot(q[b, h, t], k[b, 0, c]))`.
    /// Keeps `q[b, :, t, :]` in threadgroup memory (8 KB at H=32, D=128) and streams keys in registers,
    /// avoiding the O(H * T * S) FP32 intermediate (26.2 GB at 100K, 262 GB at 1M).
    private static let mlaIndexScoresKernel = MLXFast.metalKernel(
        name: "ernie55_mla_index_scores",
        inputNames: ["q", "k", "w"],
        outputNames: ["scores"],
        source: """
            constexpr int D = 128;
            constexpr int KEYS_PER_TG = 64;
            uint cBlock = threadgroup_position_in_grid.x;
            uint t = threadgroup_position_in_grid.y;
            uint b = threadgroup_position_in_grid.z;
            uint tid = thread_position_in_threadgroup.x;
            uint sg = simdgroup_index_in_threadgroup;
            uint lane = thread_index_in_simdgroup;
            int T = q_shape[2];
            int S = k_shape[2];

            threadgroup bfloat16_t qTile[H][D];
            threadgroup float wTile[H];
            for (int idx = int(tid); idx < H * D; idx += 256) {
                int h = idx / D;
                int d = idx % D;
                qTile[h][d] = q[(((size_t)b * H + h) * T + t) * D + d];
            }
            if (int(tid) < H) {
                wTile[tid] = w[((size_t)b * T + t) * H + tid];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            int cBase = int(cBlock) * KEYS_PER_TG;
            size_t kHeadBase = (size_t)b * S * D;
            size_t outBase = ((size_t)b * T + t) * S;

            for (int kk = int(sg); kk < KEYS_PER_TG; kk += 8) {
                int c = cBase + kk;
                if (c >= S) break;
                size_t kOff = kHeadBase + (size_t)c * D;
                float k0 = float(k[kOff + lane]);
                float k1 = float(k[kOff + lane + 32]);
                float k2 = float(k[kOff + lane + 64]);
                float k3 = float(k[kOff + lane + 96]);
                float total = 0.f;
                for (int h = 0; h < H; ++h) {
                    float dot = k0 * float(qTile[h][lane])
                              + k1 * float(qTile[h][lane + 32])
                              + k2 * float(qTile[h][lane + 64])
                              + k3 * float(qTile[h][lane + 96]);
                    float s = simd_sum(dot);
                    if (s > 0.f) total += wTile[h] * s;
                }
                if (lane == 0) {
                    scores[outBase + c] = total;
                }
            }
            """)

    /// Sparse absorbed MLA prefill over per-query selected key indices `[B, 1, T, M]`.
    /// Each threadgroup handles HEADS_PER_TG query heads for one `(b, t)`, sharing BK=4 gathered
    /// latent `kv[idx]` + `kPos[idx]` rows in 4.5 KB threadgroup memory. Zero 16x KV expansion
    /// and zero O(T * S) positional bias allocation even at 1M context.
    private static let sparseMLAPrefillKernel = MLXFast.metalKernel(
        name: "ernie55_sparse_mla_prefill",
        inputNames: ["qAbs", "qPos", "kv", "kPos", "indices", "sinks", "scaleArr"],
        outputNames: ["out"],
        source: """
            constexpr int EL = LATENT / 32;
            constexpr int RL = ROPE / 32;
            constexpr int BK = 4;
            constexpr uint TG_THREADS = 32 * HEADS_PER_TG;
            uint hg = threadgroup_position_in_grid.x;
            uint t = threadgroup_position_in_grid.y;
            uint b = threadgroup_position_in_grid.z;
            uint tid = thread_position_in_threadgroup.x;
            uint sg = simdgroup_index_in_threadgroup;
            uint lane = thread_index_in_simdgroup;
            int NQ = qAbs_shape[1];
            int T = qAbs_shape[2];
            int S = kv_shape[2];
            int M = indices_shape[3];
            float sc = scaleArr[0];
            int h = int(hg * HEADS_PER_TG + sg);

            float qa[EL];
            float qp[RL];
            float acc[EL];
            size_t qAbsBase = (((size_t)b * NQ + h) * T + t) * LATENT;
            size_t qPosBase = (((size_t)b * NQ + h) * T + t) * ROPE;
            for (int e = 0; e < EL; ++e) {
                qa[e] = float(qAbs[qAbsBase + lane + 32 * e]) * sc;
                acc[e] = 0.f;
            }
            for (int r = 0; r < RL; ++r) {
                qp[r] = float(qPos[qPosBase + lane + 32 * r]) * sc;
            }
            float m = float(sinks[h]);
            float l = 1.0f;

            threadgroup bfloat16_t kvTile[BK][LATENT];
            threadgroup bfloat16_t kpTile[BK][ROPE];
            threadgroup int idxTile[BK];
            size_t idxBase = ((size_t)b * T + t) * M;
            size_t kvBatchBase = (size_t)b * S * LATENT;
            size_t kpBatchBase = (size_t)b * S * ROPE;

            for (int mBase = 0; mBase < M; mBase += BK) {
                int tileLen = min(BK, M - mBase);
                if (int(tid) < tileLen) {
                    idxTile[tid] = indices[idxBase + mBase + tid];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                for (int i = int(tid); i < tileLen * LATENT; i += int(TG_THREADS)) {
                    int kk = i / LATENT;
                    int d = i % LATENT;
                    int kIdx = idxTile[kk];
                    kvTile[kk][d] = (kIdx >= 0 && kIdx < S)
                        ? kv[kvBatchBase + (size_t)kIdx * LATENT + d] : bfloat16_t(0.f);
                }
                for (int i = int(tid); i < tileLen * ROPE; i += int(TG_THREADS)) {
                    int kk = i / ROPE;
                    int d = i % ROPE;
                    int kIdx = idxTile[kk];
                    kpTile[kk][d] = (kIdx >= 0 && kIdx < S)
                        ? kPos[kpBatchBase + (size_t)kIdx * ROPE + d] : bfloat16_t(0.f);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                for (int kk = 0; kk < tileLen; ++kk) {
                    int kIdx = idxTile[kk];
                    if (kIdx < 0 || kIdx >= S) continue;
                    float dot = 0.f;
                    for (int e = 0; e < EL; ++e) {
                        dot += qa[e] * float(kvTile[kk][lane + 32 * e]);
                    }
                    for (int r = 0; r < RL; ++r) {
                        dot += qp[r] * float(kpTile[kk][lane + 32 * r]);
                    }
                    float score = simd_sum(dot);
                    float mNew = max(m, score);
                    float alpha = exp(m - mNew);
                    float p = exp(score - mNew);
                    l = l * alpha + p;
                    for (int e = 0; e < EL; ++e) {
                        acc[e] = acc[e] * alpha + p * float(kvTile[kk][lane + 32 * e]);
                    }
                    m = mNew;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            float invL = 1.0f / l;
            for (int e = 0; e < EL; ++e) {
                out[qAbsBase + lane + 32 * e] = bfloat16_t(acc[e] * invL);
            }
            """)
    #endif

    static func mlaIndexScores(q: MLXArray, k: MLXArray, weights: MLXArray) -> MLXArray? {
        #if canImport(Metal)
        guard Device.defaultDevice().deviceType == .gpu,
            q.ndim == 4, k.ndim == 4, weights.ndim == 3,
            q.dim(3) == 128, k.dim(1) == 1, k.dim(3) == 128,
            q.dim(1) > 0, q.dim(1) <= 64
        else { return nil }
        let (B, H, T) = (q.dim(0), q.dim(1), q.dim(2))
        let S = k.dim(2)
        let cBlocks = (S + 63) / 64
        return mlaIndexScoresKernel(
            [contiguous(q.asType(.bfloat16)), contiguous(k.asType(.bfloat16)), contiguous(weights.asType(.float32))],
            template: [("H", H)],
            grid: (cBlocks * 256, T, B),
            threadGroup: (256, 1, 1),
            outputShapes: [[B, 1, T, S]],
            outputDTypes: [.float32])[0]
        #else
        return nil
        #endif
    }

    static func sparseMLAPrefill(
        qAbs: MLXArray, qPos: MLXArray, kv: MLXArray, kPos: MLXArray,
        indices: MLXArray, sinks: MLXArray, scale: Float
    ) -> MLXArray? {
        #if canImport(Metal)
        guard Device.defaultDevice().deviceType == .gpu,
            qAbs.ndim == 4, qPos.ndim == 4, kv.ndim == 4, kPos.ndim == 4, indices.ndim == 4,
            kv.dim(1) == 1, kPos.dim(1) == 1
        else { return nil }
        let (B, NQ, T, latent) = (qAbs.dim(0), qAbs.dim(1), qAbs.dim(2), qAbs.dim(3))
        let rope = qPos.dim(3)
        guard latent > 0, latent <= 512, latent % 32 == 0,
            rope > 0, rope <= 128, rope % 32 == 0
        else { return nil }
        let headsPerTG = NQ % 8 == 0 ? 8 : (NQ % 4 == 0 ? 4 : 1)
        let tgThreads = 32 * headsPerTG
        let headGroups = NQ / headsPerTG
        let scaleArr = MLXArray([scale])
        return sparseMLAPrefillKernel(
            [
                contiguous(qAbs.asType(.bfloat16)), contiguous(qPos.asType(.bfloat16)),
                contiguous(kv.asType(.bfloat16)), contiguous(kPos.asType(.bfloat16)),
                contiguous(indices.asType(.int32)), contiguous(sinks.asType(.bfloat16)), scaleArr,
            ],
            template: [("LATENT", latent), ("ROPE", rope), ("HEADS_PER_TG", headsPerTG)],
            grid: (headGroups * tgThreads, T, B),
            threadGroup: (tgThreads, 1, 1),
            outputShapes: [[B, NQ, T, latent]],
            outputDTypes: [.bfloat16])[0].asType(qAbs.dtype)
        #else
        return nil
        #endif
    }
}
