// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// DeepSeek-V4.1 staggered hyper-connection stage at decode (one token) in two Metal dispatches.
//
// Port of the mlx-serve hc stage (`HC_STAGE_HEAD/TAIL` + `HC_STAGE_LOAD_EXPAND` + `HC_SPLIT_SOURCE`,
// src/deepseek_v41.zig L1429-1553 at 191da71de32a; oMLX `decode_layer.py` K1/K2 is the same split). One
// sublayer of the V4.1 staggered mHC is
//   mixes(h)   : x = f32(h); r = rsqrt(mean(x * x) + eps); z = (x @ fn.T) * r; split-Sinkhorn(z)
//   collapse   : y = bf16(sum_j pre_prev[j] * f32(h[j])), then RMSNorm(y)
// and, on the FFN side, h is itself the attention expand hcPost(attnOut, h, post, comb). The stock path is
// 13 dispatches per sublayer (cast, square, mean, add, rsqrt, GEMV, multiply, Sinkhorn; cast, multiply,
// sum, cast; RMSNorm) plus the expand. Here the stage kernel runs every part that does not need this
// stage's mixes in one grid of 8 threadgroups (6 for the 24 mix dot products, 1 for the sum of squares,
// 1 for the collapse + RMSNorm, which only needs the PREVIOUS sublayer's pre), rebuilding the expanded
// stream in place on the FFN side; the split kernel finishes the mixes and runs the Sinkhorn split.
//
// Exactness: bit-identical to the stock graph. Each part is a transcription of the kernel MLX dispatches
// for it at one row (fork mlx 169ba206): the mix dot products are `gemv` (BM1 BN8 SM1 SN32 TM4 TN4 for
// 24 outputs over K = 20480: 4 outputs a threadgroup, 8 simdgroups over K, shuffle-down ladder, then the
// threadgroup partials in simdgroup order); the sum of squares is `row_reduce_looped` (1024 threads x 4
// reads, simd_sum, threadgroup simd_sum); the mean multiplies by fp32(1 / N) as `mean` does; the collapse
// is `col_reduce_small` over the 4 copies; the norm is `rms_looped` at 1024 threads; the expand and the
// Sinkhorn split are the existing `deepseek_v4_hc_post_decode` / `deepseek_v4_hc_split_sinkhorn` bodies.
// Every shape is still admitted once per process only after a live bitwise comparison against the stock
// graph (the first decode step), so a compiler or MLX change can only fall back. Verify rows (2-8) keep the
// stock graph: MLX runs their mix projection as a GEMM. `VMLX_DSV41_HC_STAGE=0` disables.

import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN

enum DeepseekV41HCStage {
    /// `VMLX_DSV41_HC_STAGE=0` (or false/off/no) keeps the stock staggered-mHC graph. Tests flip it.
    nonisolated(unsafe) static var enabled: Bool = {
        let raw = ProcessInfo.processInfo.environment["VMLX_DSV41_HC_STAGE"] ?? "1"
        return !["0", "false", "off", "no"].contains(
            raw.trimmingCharacters(in: .whitespaces).lowercased())
    }()

    struct Key: Hashable {
        let hc: Int, hidden: Int, expand: Bool
    }

    private static let lock = NSLock()
    nonisolated(unsafe) private static var verdicts: [Key: Bool] = [:]

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

    /// Threads of the stage kernel's threadgroups: `row_reduce_looped` and `rms_looped` both run 1024.
    static let threads = 1024

    static let header = """
        // The stream element i of the (expanded) hc stream, as f32. X: hcPost(y, res, post, comb) in the
        // verbatim `deepseek_v4_hc_post_decode` order, rounded to T; else the stream itself.
        // Pointer types are generic: MLX binds small inputs in the constant address space.
        template <typename T, int HC, int D, bool X, typename PH, typename PY, typename PP, typename PC>
        inline float hcs_stream(PH h, PY y, PP post, PC comb, int i) {
          if (!X) return static_cast<float>(h[i]);
          const uint d = uint(i) % D;
          const uint target_hc = uint(i) / D;
          float residual_mix =
              static_cast<float>(comb[target_hc])
              * static_cast<float>(h[d]);
          for (uint source_hc = 1; source_hc < HC; ++source_hc) {
            const float term =
                static_cast<float>(comb[source_hc * HC + target_hc])
                * static_cast<float>(h[source_hc * D + d]);
            residual_mix = residual_mix + term;
          }
          const float direct =
              static_cast<float>(post[target_hc])
              * static_cast<float>(y[d]);
          return static_cast<float>(static_cast<T>(direct + residual_mix));
        }
        """

    static let stageKernel = MLXFast.metalKernel(
        name: "vmlx_dsv41_hc_stage",
        inputNames: ["h", "y", "post", "comb", "cpre", "w", "fn", "reps"],
        outputNames: ["normed", "z", "ss", "hx"],
        source: """
            constexpr int N = HC * D;
            constexpr int MIX = (2 + HC) * HC;
            const uint tg = threadgroup_position_in_grid.x;
            const uint lid = thread_position_in_threadgroup.x;
            const uint lane = thread_index_in_simdgroup;
            const uint sg = simdgroup_index_in_threadgroup;
            threadgroup float t_gemv[8 * 8];
            threadgroup float t_sums[32];
            threadgroup float t_inv[1];

            if (tg < MIX / 4) {
              // gemv_float32_bm1_bn8_sm1_sn32_tm4_tn4: out rows tg*4..+3, simdgroup sg owns K columns
              // (32 sg + lane) * 4 + 1024 i.
              float result[4] = {0, 0, 0, 0};
              const int out_row = int(tg) * 4;
              if (sg < 8) {
                int bn = (32 * int(sg) + int(lane)) * 4;
                for (int i = 0; i < N / 1024; ++i) {
                  float v_coeff[4];
                  for (int tn = 0; tn < 4; tn++) v_coeff[tn] = hcs_stream<T, HC, D, X>(h, y, post, comb, bn + tn);
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
                for (int tm = 0; tm < 4; tm++) z[out_row + tm] = result[tm];
              }
            } else if (tg == MIX / 4) {
              // row_reduce_looped (sum) over x * x: 1024 threads, 4 reads each, N / 4096 blocks.
              float row_total = 0;
              int base = int(lid) * 4;
              for (int i = 0; i < N / 4096; i++) {
                for (int j = 0; j < 4; j++) {
                  const float v = hcs_stream<T, HC, D, X>(h, y, post, comb, base + j);
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
              if (lid == 0) ss[0] = total;
            } else {
              // Collapse with the previous sublayer's pre (col_reduce_small over the HC copies, then the
              // cast), then rms_looped at 1024 threads. On the FFN side also store the expanded stream.
              float cp[HC];
              for (int j = 0; j < HC; j++) cp[j] = cpre[j];
              if (X) {
                for (int i = int(lid); i < N; i += 1024) hx[i] = static_cast<T>(hcs_stream<T, HC, D, X>(h, y, post, comb, i));
              }
              float acc = 0;
              T yv[2 * 4];
              for (int rr = 0; rr < 2; rr++) {
                const int r = rr * 4096;
                for (int i = 0; i < 4; i++) {
                  const int d = r + int(lid) * 4 + i;
                  if (r + int(lid) * 4 + i < D) {
                    float t = cp[0] * hcs_stream<T, HC, D, X>(h, y, post, comb, d) + 0;
                    for (int j = 1; j < HC; j++) {
                      t = (cp[j] * hcs_stream<T, HC, D, X>(h, y, post, comb, j * D + d) + 0) + t;
                    }
                    yv[rr * 4 + i] = static_cast<T>(t);
                  }
                }
              }
              for (int rr = 0; rr < 2; rr++) {
                const int r = rr * 4096;
                if (r + int(lid) * 4 + 4 <= D) {
                  for (int i = 0; i < 4; i++) {
                    float xi = yv[rr * 4 + i];
                    acc += xi * xi;
                  }
                } else {
                  for (int i = 0; i < 4; i++) {
                    if ((r + int(lid) * 4 + i) < D) {
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
                const int r = rr * 4096;
                for (int i = 0; i < 4; i++) {
                  const int d = r + int(lid) * 4 + i;
                  if (d < D) normed[d] = w[d] * static_cast<T>(float(yv[rr * 4 + i]) * t_inv[0]);
                }
              }
            }
            """,
        header: header)

    static let splitKernel = MLXFast.metalKernel(
        name: "vmlx_dsv41_hc_split",
        inputNames: ["z", "ss", "invn", "neps", "scale", "base", "eps"],
        outputNames: ["pre", "post", "comb"],
        source: """
            constexpr int MIX = (2 + HC) * HC;
            // mean (sum * fp32(1/N)), + eps, precise rsqrt, then the mixes as `z * r`.
            const float r = metal::precise::rsqrt(ss[0] * invn[0] + neps[0]);
            float mix[MIX];
            for (int i = 0; i < MIX; i++) mix[i] = z[i] * r;
            float epsv = static_cast<float>(eps[0]);
            auto pre_out = pre;
            auto post_out = post;
            auto comb_out = comb;

            float pre_scale = static_cast<float>(scale[0]);
            float post_scale = static_cast<float>(scale[1]);
            float comb_scale = static_cast<float>(scale[2]);

            for (int i = 0; i < HC; ++i) {
                float zz = static_cast<float>(mix[i]) * pre_scale
                    + static_cast<float>(base[i]);
                pre_out[i] = 1.0f / (1.0f + metal::fast::exp(-zz)) + epsv;
            }
            for (int i = 0; i < HC; ++i) {
                int off = HC + i;
                float zz = static_cast<float>(mix[off]) * post_scale
                    + static_cast<float>(base[off]);
                post_out[i] = 2.0f / (1.0f + metal::fast::exp(-zz));
            }

            float c[HC * HC];
            for (int i = 0; i < HC; ++i) {
                float row_max = -INFINITY;
                for (int j = 0; j < HC; ++j) {
                    int cidx = i * HC + j;
                    int off = 2 * HC + cidx;
                    float v = static_cast<float>(mix[off]) * comb_scale
                        + static_cast<float>(base[off]);
                    c[cidx] = v;
                    row_max = metal::max(row_max, v);
                }
                float row_sum = 0.0f;
                for (int j = 0; j < HC; ++j) {
                    int cidx = i * HC + j;
                    float v = metal::fast::exp(c[cidx] - row_max);
                    c[cidx] = v;
                    row_sum += v;
                }
                float inv_sum = 1.0f / row_sum;
                for (int j = 0; j < HC; ++j) {
                    int cidx = i * HC + j;
                    c[cidx] = c[cidx] * inv_sum + epsv;
                }
            }

            for (int j = 0; j < HC; ++j) {
                float col_sum = 0.0f;
                for (int i = 0; i < HC; ++i) {
                    col_sum += c[i * HC + j];
                }
                float inv_denom = 1.0f / (col_sum + epsv);
                for (int i = 0; i < HC; ++i) {
                    c[i * HC + j] *= inv_denom;
                }
            }

            for (int iter = 1; iter < ITERS; ++iter) {
                for (int i = 0; i < HC; ++i) {
                    float row_sum = 0.0f;
                    for (int j = 0; j < HC; ++j) {
                        row_sum += c[i * HC + j];
                    }
                    float inv_denom = 1.0f / (row_sum + epsv);
                    for (int j = 0; j < HC; ++j) {
                        c[i * HC + j] *= inv_denom;
                    }
                }
                for (int j = 0; j < HC; ++j) {
                    float col_sum = 0.0f;
                    for (int i = 0; i < HC; ++i) {
                        col_sum += c[i * HC + j];
                    }
                    float inv_denom = 1.0f / (col_sum + epsv);
                    for (int i = 0; i < HC; ++i) {
                        c[i * HC + j] *= inv_denom;
                    }
                }
            }

            for (int i = 0; i < HC * HC; ++i) {
                comb_out[i] = c[i];
            }
            """)

    /// One sublayer's stage outputs: the normed sublayer input, this sublayer's (pre, post, comb), and on
    /// the FFN side the expanded stream it collapsed (`hA`).
    struct Output {
        let normed: MLXArray, pre: MLXArray, post: MLXArray, comb: MLXArray
        let stream: MLXArray?
    }

    /// The kernels' admission conditions: one token, `HC * D` a multiple of the GEMV / reduce blocks, the
    /// norm row inside two rms_looped passes, fp32 hc parameters, a bf16/fp16 stream and norm weight.
    static func admits(_ h: MLXArray, hc: DeepseekV4HyperConnection, norm: RMSNorm) -> Bool {
        let n = hc.hcMult * hc.hiddenSize
        return h.ndim == 4 && h.dim(0) == 1 && h.dim(1) == 1 && h.dim(2) == hc.hcMult
            && h.dim(3) == hc.hiddenSize && (h.dtype == .bfloat16 || h.dtype == .float16)
            && hc.mixHc % 4 == 0 && hc.mixHc / 4 <= 8 && n % 4096 == 0
            && hc.hiddenSize > 4096 && hc.hiddenSize <= 8192
            && hc.fn.dtype == .float32 && hc.fn.shape == [hc.mixHc, n]
            && hc.scale.dtype == .float32 && hc.base.dtype == .float32
            && norm.weight.dtype == h.dtype && norm.weight.shape == [hc.hiddenSize]
            // METAL-ONLY: case 1. Guards the Metal kernels; other devices keep the stock graph.
            && Device.defaultDevice().deviceType == .gpu
    }

    private static let scalarLock = NSLock()
    nonisolated(unsafe) private static var scalars: [Float: MLXArray] = [:]
    private static func scalar(_ v: Float) -> MLXArray {
        scalarLock.lock(); defer { scalarLock.unlock() }
        if let a = scalars[v] { return a }
        let a = MLXArray([v]); a.eval(); scalars[v] = a
        return a
    }

    /// Run the two kernels. `h` [1, 1, HC, D] is the stream (attention side) or the expand's residual
    /// (FFN side, with `y` the attention output [1, 1, D] and `post` / `comb` its split).
    static func run(
        h: MLXArray, y: MLXArray?, post: MLXArray?, comb: MLXArray?, cpre: MLXArray,
        hc: DeepseekV4HyperConnection, norm: RMSNorm
    ) -> Output {
        let HC = hc.hcMult, D = hc.hiddenSize, mix = hc.mixHc
        let expand = y != nil
        let dt = h.dtype
        let hf = h.reshaped(HC * D)
        let outs = stageKernel(
            [hf, (y ?? hf).reshaped(-1), (post ?? cpre).reshaped(-1).asType(.float32),
             (comb ?? cpre).reshaped(-1).asType(.float32), cpre.reshaped(-1).asType(.float32),
             norm.weight, hc.fn, scalar(norm.eps)],
            template: [("T", dt), ("HC", HC), ("D", D), ("X", expand)],
            grid: (threads * (mix / 4 + 2), 1, 1),
            threadGroup: (threads, 1, 1),
            outputShapes: [[D], [mix], [1], expand ? [HC * D] : [1]],
            outputDTypes: [dt, .float32, .float32, dt])
        // mean multiplies by fp32(1 / N), the constant `number_of_elements(inverted: true)` builds.
        let split = splitKernel(
            [outs[1], outs[2], scalar(Float(1.0 / Double(HC * D))), scalar(hc.hcNormEps),
             hc.scale, hc.base, scalar(hc.hcEps)],
            template: [("HC", HC), ("ITERS", hc.hcIters)],
            grid: (1, 1, 1),
            threadGroup: (1, 1, 1),
            outputShapes: [[1, 1, HC], [1, 1, HC], [1, 1, HC, HC]],
            outputDTypes: [.float32, .float32, .float32])
        return Output(
            normed: outs[0].reshaped(1, 1, D), pre: split[0], post: split[1], comb: split[2],
            stream: expand ? outs[3].reshaped(1, 1, HC, D) : nil)
    }

    /// The stock graph the kernels reproduce (`callStaggered`'s statements).
    static func stock(
        h: MLXArray, y: MLXArray?, post: MLXArray?, comb: MLXArray?, cpre: MLXArray,
        hc: DeepseekV4HyperConnection, norm: RMSNorm
    ) -> Output {
        let stream = y.map { hc.expand(blockOut: $0, residual: h, post: post!, comb: comb!) } ?? h
        let (pre, p, c) = hc.mixes(stream)
        let normed = norm(DeepseekV4Math.hcCollapse(stream, pre: cpre))
        return Output(normed: normed, pre: pre, post: p, comb: c, stream: y == nil ? nil : stream)
    }

    /// The fused stage, or nil for the stock graph (admitted per shape on the first eager call).
    static func apply(
        h: MLXArray, y: MLXArray? = nil, post: MLXArray? = nil, comb: MLXArray? = nil, cpre: MLXArray,
        hc: DeepseekV4HyperConnection, norm: RMSNorm
    ) -> Output? {
        guard enabled, admits(h, hc: hc, norm: norm), cpre.size == hc.hcMult else { return nil }
        if let y {
            guard y.size == hc.hiddenSize, y.dtype == h.dtype, post?.size == hc.hcMult,
                comb?.size == hc.hcMult * hc.hcMult
            else { return nil }
        }
        let key = Key(hc: hc.hcMult, hidden: hc.hiddenSize, expand: y != nil)
        switch verdict(key) {
        case .some(true):
            return run(h: h, y: y, post: post, comb: comb, cpre: cpre, hc: hc, norm: norm)
        case .some(false):
            return nil
        case .none:
            if CompiledDecodeTrace.isActive { return nil }
            let fused = run(h: h, y: y, post: post, comb: comb, cpre: cpre, hc: hc, norm: norm)
            let ref = stock(h: h, y: y, post: post, comb: comb, cpre: cpre, hc: hc, norm: norm)
            let pairs: [(MLXArray, MLXArray)] =
                [(fused.normed, ref.normed), (fused.pre, ref.pre), (fused.post, ref.post),
                 (fused.comb, ref.comb)] + (fused.stream.map { [($0, ref.stream!)] } ?? [])
            eval(pairs.flatMap { [$0.0, $0.1] })
            let ok = pairs.allSatisfy { a, b in
                a.shape == b.shape && a.dtype == b.dtype && arrayEqual(a, b).item(Bool.self)
            }
            record(key, ok)
            if !ok {
                NSLog(
                    "VMLX DSV4.1 fused hc stage (expand=\(y != nil)) differs from the stock graph; "
                        + "keeping the stock mHC for this shape.")
            }
            return ok ? fused : ref
        }
    }
}
