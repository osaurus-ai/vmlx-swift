// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// DeepSeek-V4.1 staggered hyper-connections for dSpark verify rows (2-8) in two Metal dispatches a sublayer.
//
// The stock path is about 14 dispatches a sublayer (see `DeepseekV41HCStage`), two sublayers a layer. Two
// fused forms, both taking the attention branch's expand into the FFN boundary so a layer runs 5 dispatches:
//
//   stage + split: `DeepseekV41HCStage`'s two kernels with a grid row per verify row. Every row runs the 1-row
//     statements, so it is bit-identical to the 1-row kernels and to the stock graph run one row at a time.
//   mix + boundary: the 24 mix dot products and the sum of squares as partial sums over 1024-value slices, so
//     `fn` is read by 20 threadgroups a row instead of 6; then the partials' totals, the collapse, the RMSNorm
//     and the split in one threadgroup a row. The normed input and the expanded stream are bit-identical to
//     stage + split; the partial sums run in another order than MLX's GEMV and row reduction, so pre / post /
//     comb agree to f32 rounding (each row still equal to its own row run alone through these kernels).
//
// Each row count is admitted once per process on its first eager call: stage + split bitwise against the stock
// graph run one row at a time, mix + boundary against stage + split (rtol 1e-5, atol 1e-6 on pre / post / comb,
// bitwise on the rest). A difference or a kernel that fails to run keeps the next path down for that shape.
//
// `VMLX_DSV41_HC_STAGE=0` keeps the stock graph; stage + split is the default and `VMLX_DSV41_HC_BOUNDARY=1` opts in to
// mix + boundary. The Python runtime
// trusts its fused hc only where f32 GEMMs round to TF32, so these rows do too (`gemmRoundsToTF32`), although
// the row reference here is not a GEMM: a 1-row Swift expand is `deepseek_v4_hc_post_decode`, in full fp32.

import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN

/// Helpers the verify-row kernels share.
enum DeepseekV41RowKernels {
    static func envOn(_ names: [String], default fallback: String = "1") -> Bool {
        let env = ProcessInfo.processInfo.environment
        let raw = names.lazy.compactMap { env[$0] }.first ?? fallback
        return !["0", "false", "off", "no"].contains(
            raw.trimmingCharacters(in: .whitespaces).lowercased())
    }

    /// Evaluates `arrays` and runs `compare` inside MLX's error scope: a kernel that fails to compile or launch
    /// reports false rather than taking the process down.
    static func check(_ arrays: [MLXArray], compare: () -> Bool) -> Bool {
        do {
            return try withError { error in
                eval(arrays)
                try error.check()
                let ok = compare()
                try error.check()
                return ok
            }
        } catch {
            NSLog("VMLX DSV4.1 verify-row kernel failed to run: \(error)")
            return false
        }
    }
}

enum DeepseekV41HCRows {
    typealias Output = DeepseekV41HCStage.Output

    /// On while `DeepseekV41HCStage.enabled` is (`VMLX_DSV41_HC_STAGE=0` turns both off). Tests clear this alone to
    /// keep verify rows on the stock graph.
    nonisolated(unsafe) static var enabled = true
    /// `VMLX_DSV41_HC_BOUNDARY=1` serves mix + boundary. Off by default: Swift's 1-row decode is the stage (the Python
    /// runtime's 1-row decode is its boundary), so here only stage + split keeps a verify row equal to its 1-row result,
    /// and at 2-8 rows the dispatch count is the same (5 a layer).
    nonisolated(unsafe) static var boundaryEnabled = DeepseekV41RowKernels.envOn(["VMLX_DSV41_HC_BOUNDARY"], default: "0")

    static let maxRows = 8

    enum Path: Hashable { case stage, boundary }

    struct Key: Hashable {
        let path: Path, rows: Int, hc: Int, hidden: Int, expand: Bool
    }

    private static let lock = NSLock()
    nonisolated(unsafe) private static var verdicts: [Key: Bool] = [:]
    nonisolated(unsafe) private static var runs: [Path: Int] = [:]

    static func verdict(_ key: Key) -> Bool? {
        lock.lock(); defer { lock.unlock() }
        return verdicts[key]
    }

    static func record(_ key: Key, _ ok: Bool) {
        lock.lock(); defer { lock.unlock() }
        verdicts[key] = ok
    }

    static func resetVerdicts() {
        lock.lock(); defer { lock.unlock() }
        verdicts.removeAll()
    }

    /// Calls served by a path (test and diagnostic counter).
    static func runCount(_ path: Path) -> Int {
        lock.lock(); defer { lock.unlock() }
        return runs[path] ?? 0
    }

    private static func countRun(_ path: Path) {
        lock.lock(); defer { lock.unlock() }
        runs[path, default: 0] += 1
    }

    // MARK: - TF32

    /// Whether f32 GEMMs on this device round their operands to TF32 (M5 matrix units with MLX's default
    /// `MLX_ENABLE_TF32`), probed with the batched expand's own shape.
    static let gemmRoundsToTF32: Bool = {
        let raw = ProcessInfo.processInfo.environment["MLX_ENABLE_TF32"] ?? "1"
        if ["0", "false", "False"].contains(raw.trimmingCharacters(in: .whitespaces)) { return false }
        // METAL-ONLY: case 1. The rows kernels are Metal kernels; other devices keep the stock graph.
        guard Device.defaultDevice().deviceType == .gpu else { return false }
        // 1 + 2^-12 is an fp32 value TF32 cannot hold: the sum of four is 4 only if each was truncated to 1.
        let comb = MLXArray.full([1, 1, 4, 4], values: MLXArray(Float(1.0 + 1.0 / 4096.0)))
        let y = DeepseekV4Math.hcExpandResidual(comb: comb, residual: MLXArray.ones([1, 1, 4, 256], dtype: .float32))
        return y[0, 0, 0, 0].item(Float.self) == 4.0
    }()

    /// Tests set this to exercise the kernels with `MLX_ENABLE_TF32=0`, and to check the fallback.
    nonisolated(unsafe) static var tf32Override: Bool? = nil

    static var tf32Trusted: Bool { tf32Override ?? gemmRoundsToTF32 }

    // MARK: - Kernels

    /// The collapse and RMSNorm of one row at 1024 threads, as the 1-row stage kernel (`hr`, `yr`, `pr`, `cr`, `cp`, `r`,
    /// `lid`, `lane`, `sg`, `t_sums`, `t_inv` in scope).
    private static let collapseNorm = """
                float acc = 0;
                T yv[2 * 4];
                for (int rr = 0; rr < 2; rr++) {
                  const int rb = rr * 4096;
                  for (int i = 0; i < 4; i++) {
                    const int d = rb + int(lid) * 4 + i;
                    if (rb + int(lid) * 4 + i < D) {
                      float t = cp[0] * hcs_stream<T, HC, D, X>(hr, yr, pr, cr, d) + 0;
                      for (int j = 1; j < HC; j++) {
                        t = (cp[j] * hcs_stream<T, HC, D, X>(hr, yr, pr, cr, j * D + d) + 0) + t;
                      }
                      yv[rr * 4 + i] = static_cast<T>(t);
                    }
                  }
                }
                for (int rr = 0; rr < 2; rr++) {
                  const int rb = rr * 4096;
                  if (rb + int(lid) * 4 + 4 <= D) {
                    for (int i = 0; i < 4; i++) {
                      float xi = yv[rr * 4 + i];
                      acc += xi * xi;
                    }
                  } else {
                    for (int i = 0; i < 4; i++) {
                      if ((rb + int(lid) * 4 + i) < D) {
                        float xi = yv[rr * 4 + i];
                        acc += xi * xi;
                      }
                    }
                  }
                }
                acc = simd_sum(acc);
                if (sg == 0) t_sums[lane] = 0;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (lane == 0) t_sums[sg] = acc;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (sg == 0) {
                  acc = simd_sum(t_sums[lane]);
                  if (lane == 0) t_inv[0] = metal::precise::rsqrt(acc / uint(D) + reps[0]);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (int rr = 0; rr < 2; rr++) {
                  const int rb = rr * 4096;
                  for (int i = 0; i < 4; i++) {
                    const int d = rb + int(lid) * 4 + i;
                    if (d < D) normed[size_t(r) * D + d] = w[d] * static_cast<T>(float(yv[rr * 4 + i]) * t_inv[0]);
                  }
                }
        """

    // METAL-ONLY: case 2. Metal kernel; without a Metal device the caller keeps the stock hyper-connection ops.
    static let stageKernel = MLXFast.metalKernel(
        name: "vmlx_dsv41_hc_stage_rows",
        inputNames: ["h", "y", "post", "comb", "cpre", "w", "fn", "reps"],
        outputNames: ["normed", "z", "ss", "hx"],
        source: """
            constexpr int N = HC * D;
            constexpr int MIX = (2 + HC) * HC;
            const uint tg = threadgroup_position_in_grid.x;
            const uint r = threadgroup_position_in_grid.y;
            const uint lid = thread_position_in_threadgroup.x;
            const uint lane = thread_index_in_simdgroup;
            const uint sg = simdgroup_index_in_threadgroup;
            threadgroup float t_gemv[8 * 8];
            threadgroup float t_sums[32];
            threadgroup float t_inv[1];
            auto hr = h + size_t(r) * N;
            auto yr = y + size_t(r) * D;
            auto pr = post + size_t(r) * HC;
            auto cr = comb + size_t(r) * HC * HC;

            if (tg < MIX / 4) {
              // gemv_float32_bm1_bn8_sm1_sn32_tm4_tn4 for this row.
              float result[4] = {0, 0, 0, 0};
              const int out_row = int(tg) * 4;
              if (sg < 8) {
                int bn = (32 * int(sg) + int(lane)) * 4;
                for (int i = 0; i < N / 1024; ++i) {
                  float v_coeff[4];
                  for (int tn = 0; tn < 4; tn++) v_coeff[tn] = hcs_stream<T, HC, D, X>(hr, yr, pr, cr, bn + tn);
                  for (int tm = 0; tm < 4; tm++) {
                    float inter[4];
                    for (int tn = 0; tn < 4; tn++) inter[tn] = fn[size_t(out_row + tm) * N + bn + tn];
                    for (int tn = 0; tn < 4; tn++) {
                      result[tm] += inter[tn] * v_coeff[tn];
                    }
                  }
                  bn += 1024;
                }
                for (int tm = 0; tm < 4; tm++) {
                  for (ushort sn = 16; sn >= 1; sn >>= 1) {
                    result[tm] += simd_shuffle_down(result[tm], sn);
                  }
                }
                if (lane == 0) {
                  for (int tm = 0; tm < 4; tm++) t_gemv[sg * 8 + tm] = result[tm];
                }
              }
              threadgroup_barrier(mem_flags::mem_threadgroup);
              if (sg == 0 && lane == 0) {
                for (int sgn = 1; sgn < 8; sgn++) {
                  for (int tm = 0; tm < 4; tm++) {
                    result[tm] += t_gemv[sgn * 8 + tm];
                  }
                }
                for (int tm = 0; tm < 4; tm++) z[size_t(r) * MIX + out_row + tm] = result[tm];
              }
            } else if (tg == MIX / 4) {
              // row_reduce_looped (sum) over x * x for this row: 1024 threads, 4 reads each, N / 4096 blocks.
              float row_total = 0;
              int base = int(lid) * 4;
              for (int i = 0; i < N / 4096; i++) {
                for (int j = 0; j < 4; j++) {
                  const float v = hcs_stream<T, HC, D, X>(hr, yr, pr, cr, base + j);
                  const float sq = v * v;
                  row_total = sq + row_total;
                }
                base += 4096;
              }
              float total = 0;
              total = total + row_total;
              total = simd_sum(total);
              if (lane == 0) t_sums[sg] = total;
              threadgroup_barrier(mem_flags::mem_threadgroup);
              const float value = lid < 32 ? t_sums[lid] : 0;
              total = simd_sum(value);
              if (lid == 0) ss[r] = total;
            } else {
              float cp[HC];
              for (int j = 0; j < HC; j++) cp[j] = cpre[size_t(r) * HC + j];
              if (X) {
                for (int i = int(lid); i < N; i += 1024) {
                  hx[size_t(r) * N + i] = static_cast<T>(hcs_stream<T, HC, D, X>(hr, yr, pr, cr, i));
                }
              }
            """ + collapseNorm + """

            }
            """,
        header: DeepseekV41HCStage.header)

    // METAL-ONLY: case 2. Metal kernel; without a Metal device the caller keeps the stock hyper-connection ops.
    static let splitKernel = MLXFast.metalKernel(
        name: "vmlx_dsv41_hc_split_rows",
        inputNames: ["z", "ss", "invn", "neps", "scale", "base", "eps"],
        outputNames: ["pre", "post", "comb"],
        source: """
            constexpr int MIX = (2 + HC) * HC;
            const uint row = thread_position_in_grid.x;
            const float r = metal::precise::rsqrt(ss[row] * invn[0] + neps[0]);
            float mix[MIX];
            for (int i = 0; i < MIX; i++) mix[i] = z[size_t(row) * MIX + i] * r;
            float epsv = static_cast<float>(eps[0]);
            auto pre_out = pre + row * HC;
            auto post_out = post + row * HC;
            auto comb_out = comb + row * HC * HC;
            """ + DeepseekV41HCStage.splitBody)

    // METAL-ONLY: case 2. Metal kernel; without a Metal device the caller keeps the stock hyper-connection ops.
    static let mixKernel = MLXFast.metalKernel(
        name: "vmlx_dsv41_hc_mix_rows",
        inputNames: ["h", "y", "post", "comb", "fn"],
        outputNames: ["xo", "part"],
        source: """
            constexpr int N = HC * D;
            constexpr int MIX = (2 + HC) * HC;
            const uint j = threadgroup_position_in_grid.x, r = threadgroup_position_in_grid.y;
            const uint t = thread_position_in_threadgroup.x;
            const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
            threadgroup float red[8][MIX + 1];
            auto hr = h + size_t(r) * N;
            auto yr = y + size_t(r) * D;
            auto pr = post + size_t(r) * HC;
            auto cr = comb + size_t(r) * HC * HC;
            const uint col = 1024u * j + 4u * t;
            float v[4];
            for (int i = 0; i < 4; i++) {
              v[i] = hcs_stream<T, HC, D, X>(hr, yr, pr, cr, int(col) + i);
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
        header: DeepseekV41HCStage.header)

    // METAL-ONLY: case 2. Metal kernel; without a Metal device the caller keeps the stock hyper-connection ops.
    static let boundaryKernel = MLXFast.metalKernel(
        name: "vmlx_dsv41_hc_boundary_rows",
        inputNames: ["x", "part", "cpre", "w", "reps", "invn", "neps", "scale", "base", "eps"],
        outputNames: ["normed", "pre_out", "post_out", "comb"],
        source: """
            constexpr int N = HC * D;
            constexpr int MIX = (2 + HC) * HC;
            const uint r = threadgroup_position_in_grid.y;
            const uint lid = thread_position_in_threadgroup.x;
            const uint lane = thread_index_in_simdgroup;
            const uint sg = simdgroup_index_in_threadgroup;
            threadgroup float t_sums[32];
            threadgroup float t_inv[1];
            threadgroup float tot[MIX + 1];
            auto hr = x + size_t(r) * N;
            auto yr = hr;
            auto pr = hr;
            auto cr = hr;
            if (lid <= uint(MIX)) {
              float s = 0.0f;
              for (int j = 0; j < SLICES; j++) s += part[(size_t(r) * SLICES + j) * (MIX + 1) + lid];
              tot[lid] = s;
            }
            float cp[HC];
            for (int j = 0; j < HC; j++) cp[j] = cpre[size_t(r) * HC + j];
            """ + collapseNorm + """

            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sg == 0) {
              // The split's statements a comb cell a lane; shuffles feed each lane its row and column in order.
              const float rs = metal::precise::rsqrt(tot[MIX] * invn[0] + neps[0]);
              const float epsv = static_cast<float>(eps[0]);
              const float s0 = static_cast<float>(scale[0]), s1 = static_cast<float>(scale[1]);
              const float s2 = static_cast<float>(scale[2]);
              if (lane < uint(HC)) {
                const float zp = (tot[lane] * rs) * s0 + static_cast<float>(base[lane]);
                pre_out[r * HC + lane] = 1.0f / (1.0f + metal::fast::exp(-zp)) + epsv;
                const float zo = (tot[HC + lane] * rs) * s1 + static_cast<float>(base[HC + lane]);
                post_out[r * HC + lane] = 2.0f / (1.0f + metal::fast::exp(-zo));
              }
              const ushort cell = ushort(lane & 15u), row = cell & ~ushort(3), colk = cell & ushort(3);
              float c = (tot[2 * HC + cell] * rs) * s2 + static_cast<float>(base[2 * HC + cell]);
              float row_max = -INFINITY;
              for (ushort k = 0; k < 4; k++) row_max = metal::max(row_max, simd_shuffle(c, ushort(row + k)));
              c = metal::fast::exp(c - row_max);
              float row_sum = 0.0f;
              for (ushort k = 0; k < 4; k++) row_sum += simd_shuffle(c, ushort(row + k));
              const float inv_sum = 1.0f / row_sum;
              c = c * inv_sum + epsv;
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
            """,
        header: DeepseekV41HCStage.header)

    // MARK: - Runs

    /// `h` [1, R, HC, D]; on the FFN side `y` [1, R, D] with `post` / `comb` its expand's split. Both run the kernels
    /// unconditionally (tests call them directly).
    static func runStage(
        h: MLXArray, y: MLXArray?, post: MLXArray?, comb: MLXArray?, cpre: MLXArray,
        hc: DeepseekV4HyperConnection, norm: RMSNorm
    ) -> Output {
        let R = h.dim(1), HC = hc.hcMult, D = hc.hiddenSize, mix = hc.mixHc
        let expand = y != nil
        let dt = h.dtype
        let hf = h.reshaped(R * HC * D)
        let pre32 = cpre.reshaped(-1).asType(.float32)
        let outs = stageKernel(
            [hf, (y ?? hf).reshaped(-1), (post ?? cpre).reshaped(-1).asType(.float32),
             (comb ?? cpre).reshaped(-1).asType(.float32), pre32,
             norm.weight, hc.fn, DeepseekV41HCStage.scalar(norm.eps)],
            template: [("T", dt), ("HC", HC), ("D", D), ("X", expand)],
            grid: (DeepseekV41HCStage.threads * (mix / 4 + 2), R, 1),
            threadGroup: (DeepseekV41HCStage.threads, 1, 1),
            outputShapes: [[R, D], [R, mix], [R], expand ? [R, HC, D] : [1]],
            outputDTypes: [dt, .float32, .float32, dt])
        let split = splitKernel(
            [outs[1].reshaped(-1), outs[2], DeepseekV41HCStage.scalar(Float(1.0 / Double(HC * D))),
             DeepseekV41HCStage.scalar(hc.hcNormEps), hc.scale, hc.base, DeepseekV41HCStage.scalar(hc.hcEps)],
            template: [("HC", HC), ("ITERS", hc.hcIters)],
            grid: (R, 1, 1),
            threadGroup: (R, 1, 1),
            outputShapes: [[1, R, HC], [1, R, HC], [1, R, HC, HC]],
            outputDTypes: [.float32, .float32, .float32])
        return Output(
            normed: outs[0].reshaped(1, R, D), pre: split[0], post: split[1], comb: split[2],
            stream: expand ? outs[3].reshaped(1, R, HC, D) : nil)
    }

    static func runBoundary(
        h: MLXArray, y: MLXArray?, post: MLXArray?, comb: MLXArray?, cpre: MLXArray,
        hc: DeepseekV4HyperConnection, norm: RMSNorm
    ) -> Output {
        let R = h.dim(1), HC = hc.hcMult, D = hc.hiddenSize, mix = hc.mixHc
        let expand = y != nil
        let dt = h.dtype
        let slices = HC * D / 1024
        let hf = h.reshaped(R * HC * D)
        let pre32 = cpre.reshaped(-1).asType(.float32)
        let mixed = mixKernel(
            [hf, (y ?? hf).reshaped(-1), (post ?? cpre).reshaped(-1).asType(.float32),
             (comb ?? cpre).reshaped(-1).asType(.float32), hc.fn],
            template: [("T", dt), ("HC", HC), ("D", D), ("X", expand), ("SLICES", slices)],
            grid: (256 * slices, R, 1),
            threadGroup: (256, 1, 1),
            outputShapes: [expand ? [R, HC, D] : [1], [R, slices, mix + 1]],
            outputDTypes: [dt, .float32])
        let stream = expand ? mixed[0] : h.reshaped(R, HC, D)
        let outs = boundaryKernel(
            [stream.reshaped(-1), mixed[1].reshaped(-1), pre32, norm.weight,
             DeepseekV41HCStage.scalar(norm.eps), DeepseekV41HCStage.scalar(Float(1.0 / Double(HC * D))),
             DeepseekV41HCStage.scalar(hc.hcNormEps), hc.scale, hc.base, DeepseekV41HCStage.scalar(hc.hcEps)],
            template: [("T", dt), ("HC", HC), ("D", D), ("X", false), ("SLICES", slices), ("ITERS", hc.hcIters)],
            grid: (DeepseekV41HCStage.threads, R, 1),
            threadGroup: (DeepseekV41HCStage.threads, 1, 1),
            outputShapes: [[R, D], [1, R, HC], [1, R, HC], [1, R, HC, HC]],
            outputDTypes: [dt, .float32, .float32, .float32])
        return Output(
            normed: outs[0].reshaped(1, R, D), pre: outs[1], post: outs[2], comb: outs[3],
            stream: expand ? mixed[0].reshaped(1, R, HC, D) : nil)
    }

    /// The stock graph run one row at a time, each row's output at that row.
    static func stockRows(
        h: MLXArray, y: MLXArray?, post: MLXArray?, comb: MLXArray?, cpre: MLXArray,
        hc: DeepseekV4HyperConnection, norm: RMSNorm
    ) -> Output {
        func row(_ a: MLXArray, _ r: Int) -> MLXArray { a[0..., r ..< (r + 1)] }
        let rows = (0 ..< h.dim(1)).map { r in
            DeepseekV41HCStage.stock(
                h: row(h, r), y: y.map { row($0, r) }, post: post.map { row($0, r) },
                comb: comb.map { row($0, r) }, cpre: row(cpre, r), hc: hc, norm: norm)
        }
        func cat(_ f: (Output) -> MLXArray) -> MLXArray { concatenated(rows.map(f), axis: 1) }
        return Output(
            normed: cat(\.normed), pre: cat(\.pre), post: cat(\.post), comb: cat(\.comb),
            stream: y == nil ? nil : cat { $0.stream! })
    }

    // MARK: - Admission

    /// Same shape and dtype; exactly equal when `rtol` is nil, else within `rtol` / `atol` (MLX allclose).
    private static func same(_ a: MLXArray, _ b: MLXArray, rtol: Float? = nil, atol: Float = 0) -> Bool {
        guard a.shape == b.shape, a.dtype == b.dtype else { return false }
        guard let rtol else { return arrayEqual(a, b).item(Bool.self) }
        let limit = abs(b) * rtol + atol
        return all(abs(a - b) .<= limit).item(Bool.self)
    }

    /// Bitwise on every output: what stage + split must meet against the stock graph run a row at a time.
    static func bitEqual(_ a: Output, _ b: Output) -> Bool {
        guard (a.stream == nil) == (b.stream == nil) else { return false }
        return same(a.normed, b.normed) && same(a.pre, b.pre) && same(a.post, b.post) && same(a.comb, b.comb)
            && (a.stream.map { same($0, b.stream!) } ?? true)
    }

    /// What mix + boundary must meet against stage + split: normed input and stream bitwise, the split within f32 rounding.
    static func withinRounding(_ a: Output, _ b: Output) -> Bool {
        guard (a.stream == nil) == (b.stream == nil) else { return false }
        return same(a.normed, b.normed) && (a.stream.map { same($0, b.stream!) } ?? true)
            && same(a.pre, b.pre, rtol: 1e-5, atol: 1e-6) && same(a.post, b.post, rtol: 1e-5, atol: 1e-6)
            && same(a.comb, b.comb, rtol: 1e-5, atol: 1e-6)
    }

    private static func arrays(_ o: Output) -> [MLXArray] {
        [o.normed, o.pre, o.post, o.comb] + (o.stream.map { [$0] } ?? [])
    }

    /// 2-8 rows of a bf16 / fp16 stream under the 1-row stage's own shape conditions.
    static func admits(_ h: MLXArray, hc: DeepseekV4HyperConnection, norm: RMSNorm) -> Bool {
        let n = hc.hcMult * hc.hiddenSize
        return h.ndim == 4 && h.dim(0) == 1 && (2 ... maxRows).contains(h.dim(1)) && h.dim(2) == hc.hcMult
            && h.dim(3) == hc.hiddenSize && (h.dtype == .bfloat16 || h.dtype == .float16)
            && hc.mixHc % 4 == 0 && hc.mixHc / 4 <= 8 && n % 4096 == 0
            && hc.hiddenSize > 4096 && hc.hiddenSize <= 8192
            && hc.fn.dtype == .float32 && hc.fn.shape == [hc.mixHc, n]
            && hc.scale.dtype == .float32 && hc.base.dtype == .float32
            && norm.weight.dtype == h.dtype && norm.weight.shape == [hc.hiddenSize]
            // METAL-ONLY: case 1. Guards the Metal kernels; other devices keep the stock graph.
            && Device.defaultDevice().deviceType == .gpu
    }

    /// The fused sublayer boundary for 2-8 rows, or nil for the stock graph. Stage + split, or with
    /// `boundaryEnabled` mix + boundary, admitted only over an admitted stage + split.
    static func apply(
        h: MLXArray, y: MLXArray? = nil, post: MLXArray? = nil, comb: MLXArray? = nil, cpre: MLXArray,
        hc: DeepseekV4HyperConnection, norm: RMSNorm
    ) -> Output? {
        guard DeepseekV41HCStage.enabled, enabled, admits(h, hc: hc, norm: norm),
            cpre.size == h.dim(1) * hc.hcMult, cpre.dim(1) == h.dim(1)
        else { return nil }
        if let y {
            guard y.shape == [1, h.dim(1), hc.hiddenSize], y.dtype == h.dtype, post?.size == cpre.size,
                comb?.size == cpre.size * hc.hcMult
            else { return nil }
        }
        // A compile trace must not run the checks below (nor the probe), and verify rows never trace.
        if CompiledDecodeTrace.isActive { return nil }
        guard tf32Trusted else { return nil }
        let rows = h.dim(1), expand = y != nil
        let stageKey = Key(path: .stage, rows: rows, hc: hc.hcMult, hidden: hc.hiddenSize, expand: expand)
        var stage: Output?
        switch verdict(stageKey) {
        case .some(false):
            return nil
        case .some(true):
            break
        case .none:
            let fused = runStage(h: h, y: y, post: post, comb: comb, cpre: cpre, hc: hc, norm: norm)
            let ref = stockRows(h: h, y: y, post: post, comb: comb, cpre: cpre, hc: hc, norm: norm)
            let ok = DeepseekV41RowKernels.check(arrays(fused) + arrays(ref)) { bitEqual(fused, ref) }
            record(stageKey, ok)
            if !ok {
                NSLog(
                    "VMLX DSV4.1 hc rows=\(rows) (expand=\(expand)) differ from the stock graph run a row at a time; "
                        + "keeping the stock mHC for this shape.")
                return nil
            }
            stage = fused
        }
        if boundaryEnabled, hc.hcMult == 4, hc.mixHc == 24, hc.hcMult * hc.hiddenSize % 1024 == 0 {
            let key = Key(path: .boundary, rows: rows, hc: hc.hcMult, hidden: hc.hiddenSize, expand: expand)
            switch verdict(key) {
            case .some(true):
                countRun(.boundary)
                return runBoundary(h: h, y: y, post: post, comb: comb, cpre: cpre, hc: hc, norm: norm)
            case .some(false):
                break
            case .none:
                let got = runBoundary(h: h, y: y, post: post, comb: comb, cpre: cpre, hc: hc, norm: norm)
                let ref = stage ?? runStage(h: h, y: y, post: post, comb: comb, cpre: cpre, hc: hc, norm: norm)
                let ok = DeepseekV41RowKernels.check(arrays(got) + arrays(ref)) { withinRounding(got, ref) }
                record(key, ok)
                if ok {
                    countRun(.boundary)
                    return got
                }
                NSLog(
                    "VMLX DSV4.1 hc boundary rows=\(rows) (expand=\(expand)) differs from the hc stage; "
                        + "keeping the hc stage for this shape.")
                stage = ref
            }
        }
        countRun(.stage)
        return stage ?? runStage(h: h, y: y, post: post, comb: comb, cpre: cpre, hc: hc, norm: norm)
    }
}
