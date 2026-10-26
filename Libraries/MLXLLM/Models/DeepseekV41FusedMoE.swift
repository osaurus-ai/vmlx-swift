// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// DeepSeek-V4.1 MoE at decode and dSpark-verify sizes (1-8 rows) in two Metal dispatches.
//
// Port of the mlx-serve fused MoE decode (`MOE_GU_SOURCE` / `MOE_DN_SOURCE`, src/deepseek_v41.zig
// L1612-1693 at 191da71de32a; oMLX `decode_moe.py` is the same idea): kernel 1 runs the six routed
// experts' gate and up projections plus the shared expert's w1/w3 as slot 6 in one grid and applies the
// clamped SwiGLU (times the route weight on the routed slots); kernel 2 runs the six routed down
// projections, sums them in fp32 in slot order and adds the shared expert's down projection.
// Default: 1-6 rows (see `maxRows`).
//
// The stock path is ~14 dispatches per layer (fp32 casts aside): two gather_qmv for gate/up, the scored
// SwiGLU, a gather_qmv for down, a cast + column sum, three shared-expert qmv, the shared SwiGLU and the
// fp32 add.
//
// Exactness. mlx-serve's own dot products are "every rounding where the composed path takes it" with a
// different fp32 accumulation order. This port goes further and is bit-identical to the stock MLX path:
// each dot product is a VERBATIM transcription of the kernel MLX itself dispatches for that projection
// (fork kernels/quantized.h, mlx 169ba206):
//   - routed gate/up (`gather_qmv_fast`, K = 5120): `qmv_fast_impl` with `load_vector` / `qdot`;
//   - routed down (`gather_qmv`, K = 2304 is not a multiple of the 512 fast step): `qmv_impl`;
//   - shared expert at one row (`qmv_fast`): `qmv_fast_impl`;
//   - shared expert at 2+ rows (`qmv_wide`, gen >= 15, 8 k-lanes): `qmv_wide_impl` with `dequantize`
//     (each row's sum is independent of the vector tile, so one row per lane group reproduces it);
// every result is rounded to the activation dtype where gather_qmm / quantized_matmul writes it; the
// SwiGLU is the compiled DSV4 body's fp32 arithmetic in its order (MLX's `Minimum` / `Maximum` /
// `Sigmoid` functors); the routed sum is `col_reduce_small`'s order for a 6-long column
// (((y0 + 0) + y1) + ... + y5); the shared add and the final cast are `addSharedExpertFP32`'s.
//
// Because exactness rests on the Metal compiler treating the transcription like MLX's own build, every
// (rows, bit plan, shape) is admitted once per process only after its output equals the stock path bit
// for bit on the live layer (eagerly, on the first decode / verify call that meets it). A mismatch
// keeps the stock path for that shape and logs once. `VMLX_DSV41_FUSED_MOE=0` disables.

import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN

enum DeepseekV41FusedMoE {
    /// `VMLX_DSV41_FUSED_MOE=0` (or false/off/no) keeps the stock MoE graph. Tests flip it.
    nonisolated(unsafe) static var enabled: Bool = {
        let raw = ProcessInfo.processInfo.environment["VMLX_DSV41_FUSED_MOE"] ?? "1"
        return !["0", "false", "off", "no"].contains(
            raw.trimmingCharacters(in: .whitespaces).lowercased())
    }()

    /// Rows the kernels serve by default. Interleaved min-of-30 microbenchmarks (M5 Ultra, 40 layers):
    /// 1 row 1.22-1.27x the stock graph, 2 rows 1.03-1.07x, 4 rows 0.96-1.05x (1.10-1.14x when rows share
    /// 2-4 of their 6 experts, as adjacent tokens do), 6 rows 0.95-1.00x, 8 rows 0.94-0.97x: an 8-row
    /// block is bandwidth-bound on ~48 distinct experts and the stock gather is marginally better there.
    /// `VMLX_DSV41_FUSED_MOE_MAX_ROWS` (1-8) overrides; tests raise it to 8.
    nonisolated(unsafe) static var maxRows: Int = {
        let raw = ProcessInfo.processInfo.environment["VMLX_DSV41_FUSED_MOE_MAX_ROWS"].flatMap(Int.init)
        return min(8, max(1, raw ?? 6))
    }()

    /// The largest block the kernels can serve (`VMLX_DSV41_FUSED_MOE_MAX_ROWS` upper bound).
    static let maxSupportedRows = 8

    // MARK: - Admission

    struct Key: Hashable {
        let rows: Int, hidden: Int, inter: Int, topK: Int
        let gateBits: Int, downBits: Int, sharedBits: Int, wide: Bool
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

    /// Tests: forget every verdict so the next call re-admits.
    static func resetVerdicts() {
        lock.lock(); defer { lock.unlock() }
        verdicts.removeAll()
    }

    // MARK: - MLX's own dispatch decisions (fork quantized.cpp), so the transcription matches

    /// `qmv_fast_k_alignment`: `get_pack_factor<bits, 32>() * (bits <= 2 ? 1 : 2) * 32`.
    static func fastAlignment(bits: Int) -> Int {
        let packFactor = (bits == 3 || bits == 5) ? 8 : (bits == 6 ? 4 : 32 / bits)
        return packFactor * (bits <= 2 ? 1 : 2) * 32
    }

    /// `qmv` / `gather_qmv` take the fast kernel when `N % 8 == 0 && K % alignment == 0`.
    static func usesFast(k: Int, n: Int, bits: Int) -> Bool {
        n % 8 == 0 && k % fastAlignment(bits: bits) == 0
    }

    /// GPU architecture generation and size letter (`applegpu_g17d` -> (17, "d")), as MLX parses it.
    static let architecture: (gen: Int, size: Character) = {
        #if canImport(Metal)
            let name = GPU.deviceInfo().architecture
        #else
            let name = ""  // no Metal device: the fused kernels are never admitted
        #endif
        let digits = name.split(separator: "_").last.map { s in
            s.drop(while: { !$0.isNumber }).prefix(while: { $0.isNumber })
        }
        return (Int(digits ?? "") ?? 0, name.last ?? " ")
    }()

    /// `get_qmv_batch_limit(D, O)`: at or above it QuantizedMatmul runs a GEMM instead of a qmv.
    static func qmvBatchLimit(d: Int, o: Int) -> Int {
        let (gen, size) = architecture
        if gen >= 17 && size != "d" {
            return d <= 2048 && o <= 2048 ? 33 : (d <= 4096 && o <= 4096 ? 25 : 13)
        } else if gen >= 15 && size != "d" {
            return d <= 2048 && o <= 2048 ? 13 : (d <= 4096 && o <= 4096 ? 15 : 13)
        } else if size == "d" {
            return d <= 2048 && o <= 2048 ? 32 : (d <= 4096 && o <= 4096 ? 18 : 12)
        } else if gen >= 13 {
            return d <= 2048 && o <= 2048 ? 14 : (d <= 4096 && o <= 4096 ? 10 : 6)
        } else {
            return d <= 2048 && o <= 2048 ? 18 : (d <= 4096 && o <= 4096 ? 12 : 10)
        }
    }

    // MARK: - Kernels

    static let header = """
        // ---- verbatim from mlx/backend/metal/kernels/quantized.h (fork 169ba206), bits 2/3/4/8 ----
        template <typename T, typename U, int values_per_thread, int bits>
        inline U dq_load_vector(const device T* x, thread U* x_thread) {
          U sum = 0;
          if (bits == 2) {
            for (int i = 0; i < values_per_thread; i += 4) {
              sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
              x_thread[i] = x[i];
              x_thread[i + 1] = x[i + 1] / 4.0f;
              x_thread[i + 2] = x[i + 2] / 16.0f;
              x_thread[i + 3] = x[i + 3] / 64.0f;
            }
          }
          else if (bits == 3) {
            for (int i = 0; i < values_per_thread; i += 8) {
              sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3] + x[i + 4] + x[i + 5] +
                  x[i + 6] + x[i + 7];
              x_thread[i] = x[i];
              x_thread[i + 1] = x[i + 1] / 8.0f;
              x_thread[i + 2] = x[i + 2] / 64.0f;
              x_thread[i + 3] = x[i + 3] / 2.0f;
              x_thread[i + 4] = x[i + 4] / 16.0f;
              x_thread[i + 5] = x[i + 5] / 128.0f;
              x_thread[i + 6] = x[i + 6] / 4.0f;
              x_thread[i + 7] = x[i + 7] / 32.0f;
            }
          }
          else if (bits == 4) {
            for (int i = 0; i < values_per_thread; i += 4) {
              sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
              x_thread[i] = x[i];
              x_thread[i + 1] = x[i + 1] / 16.0f;
              x_thread[i + 2] = x[i + 2] / 256.0f;
              x_thread[i + 3] = x[i + 3] / 4096.0f;
            }
          }
          else if (bits == 8) {
            for (int i = 0; i < values_per_thread; i++) {
              sum += x[i];
              x_thread[i] = x[i];
            }
          }
          return sum;
        }

        template <typename T, typename U, int values_per_thread, int bits>
        inline U dq_load_vector_safe(const device T* x, thread U* x_thread, int N) {
          U sum = 0;
          if (bits == 2) {
            for (int i = 0; i < N; i += 4) {
              sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
              x_thread[i] = x[i];
              x_thread[i + 1] = x[i + 1] / 4.0f;
              x_thread[i + 2] = x[i + 2] / 16.0f;
              x_thread[i + 3] = x[i + 3] / 64.0f;
            }
          }
          else if (bits == 3) {
            for (int i = 0; i < N; i += 8) {
              sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3] + x[i + 4] + x[i + 5] +
                  x[i + 6] + x[i + 7];

              x_thread[i] = x[i];
              x_thread[i + 1] = x[i + 1] / 8.0f;
              x_thread[i + 2] = x[i + 2] / 64.0f;
              x_thread[i + 3] = x[i + 3] / 2.0f;
              x_thread[i + 4] = x[i + 4] / 16.0f;
              x_thread[i + 5] = x[i + 5] / 128.0f;
              x_thread[i + 6] = x[i + 6] / 4.0f;
              x_thread[i + 7] = x[i + 7] / 32.0f;
            }
          }
          else if (bits == 4) {
            for (int i = 0; i < N; i += 4) {
              sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
              x_thread[i] = x[i];
              x_thread[i + 1] = x[i + 1] / 16.0f;
              x_thread[i + 2] = x[i + 2] / 256.0f;
              x_thread[i + 3] = x[i + 3] / 4096.0f;
            }
          }
          else if (bits == 8) {
            for (int i = 0; i < N; i++) {
              sum += x[i];
              x_thread[i] = x[i];
            }
          }
          for (int i = N; i < values_per_thread; i++) {
            x_thread[i] = 0;
          }
          return sum;
        }

        template <typename U, int values_per_thread, int bits>
        inline U dq_qdot(
            const device uint8_t* w,
            const thread U* x_thread,
            U scale,
            U bias,
            U sum) {
          U accum = 0;
          if (bits == 2) {
            for (int i = 0; i < (values_per_thread / 4); i++) {
              accum +=
                  (x_thread[4 * i] * (w[i] & 0x03) +
                   x_thread[4 * i + 1] * (w[i] & 0x0c) +
                   x_thread[4 * i + 2] * (w[i] & 0x30) +
                   x_thread[4 * i + 3] * (w[i] & 0xc0));
            }
          }
          else if (bits == 3) {
            for (int i = 0; i < (values_per_thread / 8); i++) {
              x_thread += 8 * i;
              w += 3 * i;

              accum += (w[0] & 0x07) * x_thread[0];
              accum += (w[0] & 0x38) * x_thread[1];
              accum += (w[0] & 0xc0) * x_thread[2];
              accum += (w[1] & 0x01) * (x_thread[2] * 256.0f);

              accum += (w[1] & 0x0e) * x_thread[3];
              accum += (w[1] & 0x70) * x_thread[4];
              accum += (w[1] & 0x80) * x_thread[5];
              accum += (w[2] & 0x03) * (x_thread[5] * 256.0f);

              accum += (w[2] & 0x1c) * x_thread[6];
              accum += (w[2] & 0xe0) * x_thread[7];
            }
          }
          else if (bits == 4) {
            const device uint16_t* ws = (const device uint16_t*)w;
            for (int i = 0; i < (values_per_thread / 4); i++) {
              accum +=
                  (x_thread[4 * i] * (ws[i] & 0x000f) +
                   x_thread[4 * i + 1] * (ws[i] & 0x00f0) +
                   x_thread[4 * i + 2] * (ws[i] & 0x0f00) +
                   x_thread[4 * i + 3] * (ws[i] & 0xf000));
            }
          }
          else if (bits == 8) {
            for (int i = 0; i < values_per_thread; i++) {
              accum += x_thread[i] * w[i];
            }
          }
          return scale * accum + sum * bias;
        }

        template <typename U, int values_per_thread, int bits>
        inline U dq_qdot_safe(
            const device uint8_t* w,
            const thread U* x_thread,
            U scale,
            U bias,
            U sum,
            int N) {
          U accum = 0;
          if (bits == 2) {
            for (int i = 0; i < (N / 4); i++) {
              accum +=
                  (x_thread[4 * i] * (w[i] & 0x03) +
                   x_thread[4 * i + 1] * (w[i] & 0x0c) +
                   x_thread[4 * i + 2] * (w[i] & 0x30) +
                   x_thread[4 * i + 3] * (w[i] & 0xc0));
            }
          }
          else if (bits == 3) {
            for (int i = 0; i < (N / 8); i++) {
              x_thread += 8 * i;
              w += 3 * i;

              accum += (w[0] & 0x07) * x_thread[0];
              accum += (w[0] & 0x38) * x_thread[1];
              accum += (w[0] & 0xc0) * x_thread[2];
              accum += (w[1] & 0x01) * (x_thread[2] * 256.0f);

              accum += (w[1] & 0x0e) * x_thread[3];
              accum += (w[1] & 0x70) * x_thread[4];
              accum += (w[1] & 0x80) * x_thread[5];
              accum += (w[2] & 0x03) * (x_thread[5] * 256.0f);

              accum += (w[2] & 0x1c) * x_thread[6];
              accum += (w[2] & 0xe0) * x_thread[7];
            }
          }
          else if (bits == 4) {
            const device uint16_t* ws = (const device uint16_t*)w;
            for (int i = 0; i < (N / 4); i++) {
              accum +=
                  (x_thread[4 * i] * (ws[i] & 0x000f) +
                   x_thread[4 * i + 1] * (ws[i] & 0x00f0) +
                   x_thread[4 * i + 2] * (ws[i] & 0x0f00) +
                   x_thread[4 * i + 3] * (ws[i] & 0xf000));
            }
          }
          else if (bits == 8) {
            for (int i = 0; i < N; i++) {
              accum += x_thread[i] * w[i];
            }
          }
          return scale * accum + sum * bias;
        }

        template <typename U, int N, int bits>
        inline void dq_dequantize(const device uint8_t* w, U scale, U bias, thread U* w_local) {
          const float s = float(scale);
          const float b = float(bias);
          if (bits == 8) {
            for (int i = 0; i < N; i++) {
              w_local[i] = static_cast<U>(s * w[i] + b);
            }
          }
        }
        // ---- end verbatim ----

        // qmv_fast_impl (verbatim body) for the 4 rows of one simdgroup starting at out_row, x one
        // row, K a multiple of block_size; the 4 dots come back reduced (simd_sum) on every lane.
        template <typename T, typename S, int group_size, int bits, int K>
        METAL_FUNC void dq_qmv_fast_rows(
            const device uint32_t* w, const device S* scales, const device S* biases,
            const device T* x, const int out_row, const uint simd_lid, thread float* result) {
          constexpr int packs_per_thread = (bits == 1 || bits == 2) ? 1 : 2;
          constexpr int results_per_simdgroup = 4;
          constexpr int pack_factor = (bits == 3 || bits == 5) ? 8 : (bits == 6 ? 4 : 32 / bits);
          constexpr int bytes_per_pack = ((bits & (bits - 1)) == 0) ? 4 : (bits == 5 ? 5 : 3);
          constexpr int values_per_thread = pack_factor * packs_per_thread;
          constexpr int block_size = values_per_thread * 32;
          constexpr int scale_step_per_thread = group_size / values_per_thread;
          const device uint8_t* ws = (const device uint8_t*)w;
          typedef float U;
          thread U x_thread[values_per_thread];
          for (int row = 0; row < results_per_simdgroup; row++) result[row] = 0;
          const int in_vec_size_w = K * bytes_per_pack / pack_factor;
          const int in_vec_size_g = K / group_size;
          ws += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
          scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
          biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
          x += simd_lid * values_per_thread;
          for (int k = 0; k < K; k += block_size) {
            U sum = dq_load_vector<T, U, values_per_thread, bits>(x, x_thread);
            for (int row = 0; row < results_per_simdgroup; row++) {
              auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
              const device S* sl = scales + row * in_vec_size_g;
              const device S* bl = biases + row * in_vec_size_g;
              U s = sl[0];
              U b = bl[0];
              result[row] += dq_qdot<U, values_per_thread, bits>(wl, x_thread, s, b, sum);
            }
            ws += block_size * bytes_per_pack / pack_factor;
            scales += block_size / group_size;
            biases += block_size / group_size;
            x += block_size;
          }
          for (int row = 0; row < results_per_simdgroup; row++) {
            result[row] = simd_sum(result[row]);
          }
        }

        // qmv_impl (verbatim body, the out_vec_size >= 8 branch, out_row a multiple of 4 below N).
        template <typename T, int group_size, int bits, int K>
        METAL_FUNC void dq_qmv_rows(
            const device uint32_t* w, const device T* scales, const device T* biases,
            const device T* x, const int out_row, const uint simd_lid, thread float* result) {
          constexpr int results_per_simdgroup = 4;
          constexpr int packs_per_thread = 1;
          constexpr int pack_factor = (bits == 3 || bits == 5) ? 8 : (bits == 6 ? 4 : 32 / bits);
          constexpr int bytes_per_pack = ((bits & (bits - 1)) == 0) ? 4 : (bits == 5 ? 5 : 3);
          constexpr int values_per_thread = pack_factor * packs_per_thread;
          constexpr int block_size = values_per_thread * 32;
          constexpr int scale_step_per_thread = group_size / values_per_thread;
          const device uint8_t* ws = (const device uint8_t*)w;
          typedef float U;
          thread U x_thread[values_per_thread];
          for (int row = 0; row < results_per_simdgroup; row++) result[row] = 0;
          const int in_vec_size = K;
          const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
          const int in_vec_size_g = in_vec_size / group_size;
          ws += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
          scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
          biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
          x += simd_lid * values_per_thread;
          int k = 0;
          for (; k < in_vec_size - block_size; k += block_size) {
            U sum = dq_load_vector<T, U, values_per_thread, bits>(x, x_thread);
            for (int row = 0; row < results_per_simdgroup; row++) {
              auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
              const device T* sl = scales + row * in_vec_size_g;
              const device T* bl = biases + row * in_vec_size_g;
              U s = sl[0];
              U b = bl[0];
              result[row] += dq_qdot<U, values_per_thread, bits>(wl, x_thread, s, b, sum);
            }
            ws += block_size * bytes_per_pack / pack_factor;
            scales += block_size / group_size;
            biases += block_size / group_size;
            x += block_size;
          }
          const int remaining = clamp(
              static_cast<int>(in_vec_size - k - simd_lid * values_per_thread),
              0,
              values_per_thread);
          if (remaining > 0) {
            U sum = dq_load_vector_safe<T, U, values_per_thread, bits>(x, x_thread, remaining);
            for (int row = 0; row < results_per_simdgroup; row++) {
              auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
              const device T* sl = scales + row * in_vec_size_g;
              const device T* bl = biases + row * in_vec_size_g;
              U s = sl[0];
              U b = bl[0];
              result[row] += dq_qdot_safe<U, values_per_thread, bits>(
                  wl, x_thread, s, b, sum, remaining);
            }
          }
          for (int row = 0; row < results_per_simdgroup; row++) {
            result[row] = simd_sum(result[row]);
          }
        }

        // qmv_wide_impl (verbatim body) for one vector: lane group simd_lid / k_lanes owns row
        // out_row + simd_lid / k_lanes; the row's dot is valid on its k_lane == 0 lane.
        template <typename T, int group_size, int bits, int k_lanes, int K>
        METAL_FUNC float dq_qmv_wide_row(
            const device uint32_t* w, const device T* scales, const device T* biases,
            const device T* x, const int row, const uint simd_lid) {
          constexpr int sub = 8;
          typedef float U;
          const short k_lane = simd_lid % k_lanes;
          const int in_vec_size_w = K * bits / 8;
          const int in_vec_size_g = K / group_size;
          const device uint8_t* wrow = (const device uint8_t*)w + row * in_vec_size_w;
          const device T* srow = scales + row * in_vec_size_g;
          const device T* brow = biases + row * in_vec_size_g;
          U result = 0;
          for (int g = k_lane; g < in_vec_size_g; g += k_lanes) {
            U scale = srow[g];
            U bias = brow[g];
        #pragma unroll
            for (int sc = 0; sc < group_size / sub; sc++) {
              const int k0 = g * group_size + sc * sub;
              const device uint8_t* wc = wrow + k0 * bits / 8;
              U w_dq[sub];
              dq_dequantize<U, sub, bits>(wc, scale, bias, w_dq);
              const device T* xc = x + k0;
              U acc = 0;
        #pragma unroll
              for (int i = 0; i < sub; i++) {
                acc += static_cast<U>(xc[i]) * w_dq[i];
              }
              result += acc;
            }
          }
          if (k_lanes >= 32) result += simd_shuffle_down(result, 16);
          if (k_lanes >= 16) result += simd_shuffle_down(result, 8);
          if (k_lanes >= 8) result += simd_shuffle_down(result, 4);
          if (k_lanes >= 4) result += simd_shuffle_down(result, 2);
          if (k_lanes >= 2) result += simd_shuffle_down(result, 1);
          return result;
        }

        // MLX Minimum / Maximum / Sigmoid functors (binary_ops.h, unary_ops.h) on float.
        inline float dq_minimum(float x, float y) { if (metal::isnan(x)) return x; return x < y ? x : y; }
        inline float dq_maximum(float x, float y) { if (metal::isnan(x)) return x; return x > y ? x : y; }
        inline float dq_sigmoid(float x) {
          auto y = 1 / (1 + metal::precise::exp(metal::abs(x)));
          return (x < 0) ? y : 1 - y;
        }
        // The DSV4 SwiGLU bodies (DeepseekV4MathHelpers): fp32 clamp, silu(g) * u (* score), one cast.
        inline float dq_swiglu(float gate, float up, float limit, bool clamped) {
          float g = clamped ? dq_minimum(gate, limit) : gate;
          float u = clamped ? dq_minimum(dq_maximum(up, -limit), limit) : up;
          return (g * dq_sigmoid(g)) * u;
        }
        """

    /// Kernel 1: gate + up + SwiGLU for every (row, slot); slot TOPK is the shared expert.
    // METAL-ONLY: case 2. Metal kernel; without a Metal device the caller keeps the stock gather_qmm MoE graph.
    static let gateUpKernel = MLXFast.metalKernel(
        name: "vmlx_dsv41_fused_moe_gate_up",
        inputNames: [
            "x", "inds", "wts", "gw", "gs", "gb", "uw", "us", "ub",
            "sgw", "sgs", "sgb", "suw", "sus", "sub_", "lim",
        ],
        outputNames: ["act"],
        source: """
            const uint lane = thread_index_in_simdgroup;
            const uint sg = simdgroup_index_in_threadgroup;
            // Rows vary fastest so the verify rows of one output block run back to back and the
            // shared expert's weight block is fetched from DRAM once for all of them.
            const uint yb = threadgroup_position_in_grid.y / ROWS, r = threadgroup_position_in_grid.y % ROWS;
            const int out_row = int(yb) * 8 + int(sg) * 4;
            const uint e = threadgroup_position_in_grid.z;
            const device T* xr = x + size_t(r) * D;
            const float limit = lim[0];
            device T* out = act + (size_t(r) * (TOPK + 1) + e) * I;
            if (e < TOPK) {
              const size_t ex = size_t(inds[r * TOPK + e]);
              constexpr int WW = D * GBITS / 32, WG = D / 64;
              float g[4], u[4];
              dq_qmv_fast_rows<T, T, 64, GBITS, D>(
                  gw + ex * I * WW, gs + ex * I * WG, gb + ex * I * WG, xr, out_row, lane, g);
              dq_qmv_fast_rows<T, T, 64, GBITS, D>(
                  uw + ex * I * WW, us + ex * I * WG, ub + ex * I * WG, xr, out_row, lane, u);
              const float s = wts[r * TOPK + e];
              if (lane == 0) {
                for (int i = 0; i < 4; i++) {
                  const float a = dq_swiglu(float(T(g[i])), float(T(u[i])), limit, CLAMP);
                  out[out_row + i] = T(a * s);
                }
              }
            } else if (SWIDE) {
              const int row = out_row + int(lane / 8);
              const float g = dq_qmv_wide_row<T, 64, SBITS, 8, D>(sgw, sgs, sgb, xr, row, lane);
              const float u = dq_qmv_wide_row<T, 64, SBITS, 8, D>(suw, sus, sub_, xr, row, lane);
              if (lane % 8 == 0) out[row] = T(dq_swiglu(float(T(g)), float(T(u)), limit, CLAMP));
            } else {
              float g[4], u[4];
              dq_qmv_fast_rows<T, T, 64, SBITS, D>(sgw, sgs, sgb, xr, out_row, lane, g);
              dq_qmv_fast_rows<T, T, 64, SBITS, D>(suw, sus, sub_, xr, out_row, lane, u);
              if (lane == 0) {
                for (int i = 0; i < 4; i++)
                  out[out_row + i] = T(dq_swiglu(float(T(g[i])), float(T(u[i])), limit, CLAMP));
              }
            }
            """,
        header: header)

    /// Kernel 2: routed downs summed in fp32 (column-reduce order) plus the shared down, one cast.
    // METAL-ONLY: case 2. Metal kernel; without a Metal device the caller keeps the stock gather_qmm MoE graph.
    static let downKernel = MLXFast.metalKernel(
        name: "vmlx_dsv41_fused_moe_down",
        inputNames: ["act", "inds", "dw", "ds", "db", "sdw", "sds", "sdb"],
        outputNames: ["y"],
        source: """
            const uint lane = thread_index_in_simdgroup;
            const uint sg = simdgroup_index_in_threadgroup;
            const uint yb = threadgroup_position_in_grid.y / ROWS, r = threadgroup_position_in_grid.y % ROWS;
            const int out_row = int(yb) * 8 + int(sg) * 4;
            const device T* ar = act + size_t(r) * (TOPK + 1) * I;
            constexpr int WW = I * DBITS / 32, WG = I / 64;
            float tot[4];
            for (int e = 0; e < TOPK; e++) {
              const size_t ex = size_t(inds[r * TOPK + e]);
              float a[4];
              if (DFAST) {
                dq_qmv_fast_rows<T, T, 64, DBITS, I>(
                    dw + ex * D * WW, ds + ex * D * WG, db + ex * D * WG, ar + e * I, out_row, lane, a);
              } else {
                dq_qmv_rows<T, 64, DBITS, I>(
                    dw + ex * D * WW, ds + ex * D * WG, db + ex * D * WG, ar + e * I, out_row, lane, a);
              }
              for (int i = 0; i < 4; i++) {
                const float v = float(T(a[i]));
                tot[i] = e == 0 ? v + 0.0f : (v + 0.0f) + tot[i];
              }
            }
            device T* yr = y + size_t(r) * D;
            if (SWIDE) {
              const int row = out_row + int(lane / 8);
              const float s = dq_qmv_wide_row<T, 64, SBITS, 8, I>(sdw, sds, sdb, ar + TOPK * I, row, lane);
              float t = tot[0];
              for (int i = 1; i < 4; i++) if (int(lane / 8) == i) t = tot[i];
              if (lane % 8 == 0) yr[row] = T(t + float(T(s)));
            } else {
              float s[4];
              dq_qmv_fast_rows<T, T, 64, SBITS, I>(sdw, sds, sdb, ar + TOPK * I, out_row, lane, s);
              if (lane == 0) {
                for (int i = 0; i < 4; i++) yr[out_row + i] = T(tot[i] + float(T(s[i])));
              }
            }
            """,
        header: header)

    // MARK: - Call

    /// The layer's arrays and geometry when the bundle's layout fits the kernels.
    struct Layer {
        let gate: QuantizedSwitchLinear, up: QuantizedSwitchLinear, down: QuantizedSwitchLinear
        let sGate: QuantizedLinear, sUp: QuantizedLinear, sDown: QuantizedLinear
        let hidden: Int, inter: Int
    }

    static func layer(_ switchMLP: SwitchGLU, shared: DeepseekV4MLP) -> Layer? {
        guard let p = switchMLP.quantizedProjections,
            let sg = shared.gateProj as? QuantizedLinear,
            let su = shared.upProj as? QuantizedLinear,
            let sd = shared.downProj as? QuantizedLinear
        else { return nil }
        let hidden = p.gate.inputDimension, inter = p.gate.outputDimension
        let routed = [p.gate, p.up, p.down]
        let dense = [sg, su, sd]
        guard routed.allSatisfy({ $0.mode == .affine && $0.groupSize == 64 && $0.groupBiases != nil
                && $0.projectionBias == nil && [2, 3, 4, 8].contains($0.bits) }),
            dense.allSatisfy({ $0.mode == .affine && $0.groupSize == 64 && $0.biases != nil
                && $0.bias == nil && [4, 8].contains($0.bits) }),
            p.gate.bits == p.up.bits, sg.bits == su.bits, sg.bits == sd.bits,
            p.up.inputDimension == hidden, p.up.outputDimension == inter,
            p.down.inputDimension == inter, p.down.outputDimension == hidden,
            sg.weight.dim(0) == inter, sd.weight.dim(0) == hidden,
            hidden % 8 == 0, inter % 8 == 0, hidden % 64 == 0, inter % 64 == 0,
            // gate/up: MLX takes gather_qmv_fast (the only one transcribed for K = hidden).
            usesFast(k: hidden, n: inter, bits: p.gate.bits),
            // shared expert at one row: qmv_fast (w1/w3 over hidden, w2 over inter).
            usesFast(k: hidden, n: inter, bits: sg.bits), usesFast(k: inter, n: hidden, bits: sd.bits)
        else { return nil }
        let dt = p.gate.groupScales.dtype
        guard dt == .bfloat16 || dt == .float16,
            (routed.flatMap { [$0.groupScales.dtype, $0.groupBiases!.dtype] }
                + dense.flatMap { [$0.scales.dtype, $0.biases!.dtype] }).allSatisfy({ $0 == dt })
        else { return nil }
        return Layer(gate: p.gate, up: p.up, down: p.down, sGate: sg, sUp: su, sDown: sd,
            hidden: hidden, inter: inter)
    }

    /// Run the two kernels. `x` [1, R, D], `indices` [1, R, K] (uint32), `scores` [1, R, K] (fp32).
    static func run(
        _ x: MLXArray, indices: MLXArray, scores: MLXArray, layer l: Layer, limit: Float, wide: Bool
    ) -> MLXArray {
        let rows = x.dim(1)
        let topK = indices.dim(-1)
        let dtype = x.dtype
        let xr = x.reshaped(rows, l.hidden)
        let inds = indices.reshaped(-1).asType(.uint32)
        let wts = scores.reshaped(-1).asType(.float32)
        let lim = MLXArray([limit])
        let common: [(String, any KernelTemplateArg)] = [
            ("T", dtype), ("D", l.hidden), ("I", l.inter), ("TOPK", topK),
            ("SBITS", l.sGate.bits), ("SWIDE", wide), ("CLAMP", limit > 0), ("ROWS", rows),
        ]
        let act = gateUpKernel(
            [xr, inds, wts,
             l.gate.packedWeight, l.gate.groupScales, l.gate.groupBiases!,
             l.up.packedWeight, l.up.groupScales, l.up.groupBiases!,
             l.sGate.weight, l.sGate.scales, l.sGate.biases!, l.sUp.weight, l.sUp.scales, l.sUp.biases!,
             lim],
            template: common + [("GBITS", l.gate.bits)],
            grid: (32, 2 * (l.inter / 8) * rows, topK + 1),
            threadGroup: (32, 2, 1),
            outputShapes: [[rows, topK + 1, l.inter]],
            outputDTypes: [dtype])[0]
        let y = downKernel(
            [act, inds, l.down.packedWeight, l.down.groupScales, l.down.groupBiases!,
             l.sDown.weight, l.sDown.scales, l.sDown.biases!],
            template: common + [
                ("DBITS", l.down.bits), ("DFAST", usesFast(k: l.inter, n: l.hidden, bits: l.down.bits)),
            ],
            grid: (32, 2 * (l.hidden / 8) * rows, 1),
            threadGroup: (32, 2, 1),
            outputShapes: [[rows, l.hidden]],
            outputDTypes: [dtype])[0]
        return y.reshaped(1, rows, l.hidden)
    }

    /// The admission key, layer and shared-expert variant when the kernels can serve this call.
    static func plan(_ x: MLXArray, topK: Int, switchMLP: SwitchGLU, shared: DeepseekV4MLP)
        -> (key: Key, layer: Layer, wide: Bool)?
    {
        guard enabled, x.ndim == 3, x.dim(0) == 1, (1 ... maxRows).contains(x.dim(1)),
            x.dtype == .bfloat16 || x.dtype == .float16,
            // METAL-ONLY: case 1. Guards the Metal kernels; other devices keep the stock graph.
            Device.defaultDevice().deviceType == .gpu,
            let l = layer(switchMLP, shared: shared), x.dim(2) == l.hidden,
            l.gate.groupScales.dtype == x.dtype
        else { return nil }
        let rows = x.dim(1)
        // Shared expert: one row runs qmv_fast; more rows run qmv_wide on gen >= 15 (qmv per row
        // before that) until the batch limit, where MLX switches to a GEMM this does not reproduce.
        let wide = rows >= 2 && architecture.gen >= 15
        guard rows < qmvBatchLimit(d: l.hidden, o: l.inter), rows < qmvBatchLimit(d: l.inter, o: l.hidden),
            // the wide transcription is the 8-bit (sub-chunk) branch only
            !wide || l.sGate.bits == 8
        else { return nil }
        let key = Key(
            rows: rows, hidden: l.hidden, inter: l.inter, topK: topK,
            gateBits: l.gate.bits, downBits: l.down.bits, sharedBits: l.sGate.bits, wide: wide)
        return (key, l, wide)
    }

    /// True when the kernels could serve `x` but this shape has not been admitted yet. The caller then
    /// runs its eager path (where admission happens) instead of recording a compiled decode trace, which
    /// would otherwise bake the stock graph in for good.
    static func awaitingAdmission(_ x: MLXArray, topK: Int, switchMLP: SwitchGLU, shared: DeepseekV4MLP) -> Bool {
        guard let p = plan(x, topK: topK, switchMLP: switchMLP, shared: shared) else { return false }
        return verdict(p.key) == nil
    }

    /// The fused MoE output for `x` [1, R, D] with R <= 8, or nil for the stock path. `stock` builds the
    /// stock output for the admission check (called at most once per shape, outside any compile trace).
    static func apply(
        _ x: MLXArray, indices: MLXArray, scores: MLXArray, switchMLP: SwitchGLU,
        shared: DeepseekV4MLP, limit: Float, stock: () -> MLXArray
    ) -> MLXArray? {
        guard indices.ndim == 3, scores.shape == indices.shape,
            let p = plan(x, topK: indices.dim(-1), switchMLP: switchMLP, shared: shared)
        else { return nil }
        let (key, l, wide) = (p.key, p.layer, p.wide)
        let rows = x.dim(1)
        switch verdict(key) {
        case .some(true):
            return run(x, indices: indices, scores: scores, layer: l, limit: limit, wide: wide)
        case .some(false):
            return nil
        case .none:
            // Admission needs concrete arrays; inside a compile trace keep the stock graph.
            if CompiledDecodeTrace.isActive { return nil }
            let fused = run(x, indices: indices, scores: scores, layer: l, limit: limit, wide: wide)
            let reference = stock()
            eval(fused, reference)
            let ok = fused.shape == reference.shape && fused.dtype == reference.dtype
                && arrayEqual(fused, reference).item(Bool.self)
            record(key, ok)
            if !ok {
                NSLog(
                    "VMLX DSV4.1 fused MoE: rows=\(rows) bits=\(l.gate.bits)/\(l.down.bits)/\(l.sGate.bits) "
                        + "differs from the stock path; keeping the stock MoE for this shape.")
            }
            return ok ? fused : reference
        }
    }
}
