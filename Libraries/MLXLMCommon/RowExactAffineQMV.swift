// Copyright © 2026 Apple Inc.

import Foundation
import MLX

/// Multi-row affine `qmv` that is bitwise equal, row by row, to MLX's single-row `qmv_fast`.
///
/// Native-MTP verification needs every verify row to reproduce the AR step bit for bit. MLX routes
/// M >= 2 to `qmv_wide` (a different reduction), so the row-exact callers ran the M = 1 kernel per
/// row through the batched `[rows, 1, K] x [1, N, K]` form. That form is exact but the batch axis is
/// the slowest grid axis, so each row streams the whole weight matrix again: a 4-row verify of an
/// 8-bit dense backbone reads its dense bytes four times and costs more than the tokens it accepts.
///
/// This kernel keeps MLX's `qmv_fast_impl` structure (2 simdgroups x 4 outputs, 2 packs per thread,
/// one pass over K in `block_size` steps) and its `load_vector` / `qdot` statements, transcribed
/// VERBATIM from the fork's `kernels/quantized.h` (4- and 8-bit branches), but runs all `ROWS` rows
/// inside one threadgroup: every weight word is fetched from DRAM once per k-block for all rows, and
/// each row's own accumulation keeps exactly the single-row statement order.
///
/// Every (dtype, bits, group, K, N, rows) shape is admitted once per process only after its output on
/// the REAL weights equals the caller's exact reference bit for bit; otherwise the caller keeps its
/// own path. `VMLX_ROW_EXACT_QMV_ROWS=0` disables (A/B tool).
public enum RowExactAffineQMV {
    static let enabled = ProcessInfo.processInfo.environment["VMLX_ROW_EXACT_QMV_ROWS"] != "0"

    private static let lock = NSLock()
    nonisolated(unsafe) private static var verdicts: [String: Bool] = [:]

    static let header = """
        // ---- verbatim from mlx/backend/metal/kernels/quantized.h (bits 4 and 8 branches) ----
        // WIDEN: the fork's mixed BF16-input / F16-metadata q4 branch of qmv_fast_impl (widen, then sum).
        template <typename T, typename U, int values_per_thread, int bits, bool WIDEN = false>
        inline U rx_load_vector(const device T* x, thread U* x_thread) {
          U sum = 0;
          if (bits == 4 && WIDEN) {
            for (int i = 0; i < values_per_thread; i += 4) {
              U x0 = x[i], x1 = x[i + 1], x2 = x[i + 2], x3 = x[i + 3];
              sum += x0 + x1 + x2 + x3;
              x_thread[i] = x0;
              x_thread[i + 1] = x1 / 16.0f;
              x_thread[i + 2] = x2 / 256.0f;
              x_thread[i + 3] = x3 / 4096.0f;
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

        template <typename U, int values_per_thread, int bits>
        inline U rx_qdot(
            const device uint8_t* w,
            const thread U* x_thread,
            U scale,
            U bias,
            U sum) {
          U accum = 0;
          if (bits == 4) {
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
        // ---- end verbatim ----

        // rx_qdot with the weight words already widened to float. Integer -> float conversion of the
        // masked words is exact, so every product, sum and rounding is the one rx_qdot performs; the
        // widening just happens once per k-block instead of once per verify row.
        template <int values_per_thread, int bits>
        inline void rx_widen(const device uint8_t* w, thread float* wf, int n) {
          if (bits == 4) {
            const device uint16_t* ws = (const device uint16_t*)w;
            for (int i = 0; i < (n / 4); i++) {
              wf[4 * i] = (ws[i] & 0x000f);
              wf[4 * i + 1] = (ws[i] & 0x00f0);
              wf[4 * i + 2] = (ws[i] & 0x0f00);
              wf[4 * i + 3] = (ws[i] & 0xf000);
            }
          } else if (bits == 8) {
            for (int i = 0; i < n; i++) {
              wf[i] = w[i];
            }
          }
        }

        template <typename U, int values_per_thread, int bits>
        inline U rx_qdot_f(const thread float* wf, const thread U* x_thread, U scale, U bias, U sum, int n) {
          U accum = 0;
          if (bits == 4) {
            for (int i = 0; i < (n / 4); i++) {
              accum +=
                  (x_thread[4 * i] * wf[4 * i] +
                   x_thread[4 * i + 1] * wf[4 * i + 1] +
                   x_thread[4 * i + 2] * wf[4 * i + 2] +
                   x_thread[4 * i + 3] * wf[4 * i + 3]);
            }
          } else if (bits == 8) {
            for (int i = 0; i < n; i++) {
              accum += x_thread[i] * wf[i];
            }
          }
          return scale * accum + sum * bias;
        }

        // 4-bit rows: x vectors of a chunk of up to 4 rows precomputed (rx_load_vector), each weight
        // word widened once and applied to every row of the chunk. Per (row, output) the accumulator
        // sees exactly rx_qdot's sequence: accum += (x0*w0 + x1*w1 + x2*w2 + x3*w3) for i ascending,
        // then scale * accum + sum * bias.
        template <typename T, typename S, int values_per_thread, int RC>
        inline void rx_q4_chunk(
            const device uint8_t* ws, const device S* scales, const device S* biases,
            const int in_vec_size_w, const int in_vec_size_g,
            const thread float (&xt)[RC][values_per_thread], const thread float (&sums)[RC],
            const int rc, thread float (&result)[RC][4]) {
          for (int row = 0; row < 4; row++) {
            const device uint16_t* wsr = (const device uint16_t*)(ws + row * in_vec_size_w);
            float accum[RC];
            for (int m = 0; m < RC; m++) accum[m] = 0;
            for (int i = 0; i < (values_per_thread / 4); i++) {
              const ushort word = wsr[i];
              const float w0 = (word & 0x000f);
              const float w1 = (word & 0x00f0);
              const float w2 = (word & 0x0f00);
              const float w3 = (word & 0xf000);
              for (int m = 0; m < RC; m++) {
                if (m < rc) {
                  accum[m] +=
                      (xt[m][4 * i] * w0 +
                       xt[m][4 * i + 1] * w1 +
                       xt[m][4 * i + 2] * w2 +
                       xt[m][4 * i + 3] * w3);
                }
              }
            }
            const float s = scales[row * in_vec_size_g];
            const float b = biases[row * in_vec_size_g];
            for (int m = 0; m < RC; m++) {
              if (m < rc) result[m][row] += s * accum[m] + sums[m] * b;
            }
          }
        }

        // qmv_fast_impl with ROWS input rows per threadgroup (rows share every weight fetch).
        template <typename T, typename S, int group_size, int bits, int ROWS>
        inline void rx_qmv_fast_rows(
            const device uint32_t* w,
            const device S* scales,
            const device S* biases,
            const device T* x,
            device T* y,
            const int in_vec_size,
            const int out_vec_size,
            uint tid_y,
            uint simd_gid,
            uint simd_lid) {
          constexpr int packs_per_thread = 2;
          constexpr int num_simdgroups = 2;
          constexpr int results_per_simdgroup = 4;
          constexpr int pack_factor = 32 / bits;
          constexpr int bytes_per_pack = 4;
          constexpr int values_per_thread = pack_factor * packs_per_thread;
          constexpr int block_size = values_per_thread * 32;
          constexpr int scale_step_per_thread = group_size / values_per_thread;

          const device uint8_t* ws = (const device uint8_t*)w;
          typedef float U;
          constexpr bool widen = metal::is_same_v<T, bfloat16_t> && metal::is_same_v<S, float16_t>;
          thread U x_thread[values_per_thread];
          thread U result[ROWS][results_per_simdgroup];
          for (int m = 0; m < ROWS; m++)
            for (int r = 0; r < results_per_simdgroup; r++) result[m][r] = 0;

          const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
          const int in_vec_size_g = in_vec_size / group_size;
          const int out_row = tid_y * (num_simdgroups * results_per_simdgroup) +
              simd_gid * results_per_simdgroup;

          ws += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
          scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
          biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
          x += simd_lid * values_per_thread;
          y += out_row;

          thread float wf[results_per_simdgroup][bits == 8 ? values_per_thread : 1];
          thread U s_r[results_per_simdgroup];
          thread U b_r[results_per_simdgroup];
          for (int k = 0; k < in_vec_size; k += block_size) {
            if (bits == 8) {
              // Widen once per k-block for all rows (4 x 8 floats). Not for 4 bits: 4 x 16 floats
              // spill registers and measured 2-4x slower than re-reading the words per row.
              for (int row = 0; row < results_per_simdgroup; row++) {
                rx_widen<values_per_thread, bits>(
                    (const device uint8_t*)(ws + row * in_vec_size_w), wf[row], values_per_thread);
                s_r[row] = scales[row * in_vec_size_g];
                b_r[row] = biases[row * in_vec_size_g];
              }
              for (int m = 0; m < ROWS; m++) {
                U sum = rx_load_vector<T, U, values_per_thread, bits, widen>(x + m * in_vec_size, x_thread);
                for (int row = 0; row < results_per_simdgroup; row++) {
                  result[m][row] += rx_qdot_f<U, values_per_thread, bits>(
                      wf[row], x_thread, s_r[row], b_r[row], sum, values_per_thread);
                }
              }
            } else {
              constexpr int RC = ROWS < 4 ? ROWS : 4;
              for (int m0 = 0; m0 < ROWS; m0 += RC) {
                const int rc = min(RC, ROWS - m0);
                thread float xt[RC][values_per_thread];
                thread float sums[RC];
                thread float part[RC][4];
                for (int m = 0; m < RC; m++) {
                  for (int r = 0; r < 4; r++) part[m][r] = result[min(m0 + m, ROWS - 1)][r];
                  if (m < rc) sums[m] = rx_load_vector<T, U, values_per_thread, bits, widen>(x + (m0 + m) * in_vec_size, xt[m]);
                }
                rx_q4_chunk<T, S, values_per_thread, RC>(ws, scales, biases, in_vec_size_w, in_vec_size_g, xt, sums, rc, part);
                for (int m = 0; m < rc; m++)
                  for (int r = 0; r < 4; r++) result[m0 + m][r] = part[m][r];
              }
            }
            ws += block_size * bytes_per_pack / pack_factor;
            scales += block_size / group_size;
            biases += block_size / group_size;
            x += block_size;
          }

          for (int m = 0; m < ROWS; m++) {
            for (int row = 0; row < results_per_simdgroup; row++) {
              result[m][row] = simd_sum(result[m][row]);
              if (simd_lid == 0) {
                y[m * out_vec_size + row] = static_cast<T>(result[m][row]);
              }
            }
          }
        }
        """

    private static let kernel = MLXFast.metalKernel(
        name: "vmlx_row_exact_affine_qmv_fast_rows",
        inputNames: ["x", "w", "scales", "biases"],
        outputNames: ["out"],
        source: """
            rx_qmv_fast_rows<T, S, GROUP_SIZE, BITS, ROWS>(
                w, scales, biases, x, out,
                int(x_shape[x_ndim - 1]), int(w_shape[0]),
                threadgroup_position_in_grid.y,
                simdgroup_index_in_threadgroup,
                thread_index_in_simdgroup);
            """,
        header: header,
        ensureRowContiguous: true)

    /// Whether the kernel can express this projection (MLX would run `qmv_fast` for one row).
    public static func supports(
        rows: Int, k: Int, n: Int, bits: Int, groupSize: Int, inputDType: DType, scalesDType: DType,
        hasBiases: Bool
    ) -> Bool {
        // 4 bits above 4 rows: the batched per-row form measured faster (scalar dequant per row
        // is compute-bound at 8 rows), so the caller keeps it.
        guard enabled, (2 ... 8).contains(rows), bits == 8 || (bits == 4 && rows <= 4), hasBiases,
            [32, 64, 128].contains(groupSize),
            inputDType == .bfloat16 || inputDType == .float16,
            scalesDType == inputDType || (inputDType == .bfloat16 && scalesDType == .float16)
        else { return false }
        let blockSize = (32 / bits) * 2 * 32
        return k % blockSize == 0 && n % 8 == 0 && groupSize % ((32 / bits) * 2) == 0
    }

    /// `input` [rows, K] against an affine-quantized [N, K] weight; nil when unsupported or not
    /// admitted bitwise against `reference` (the caller's exact single-row-kernel result).
    public static func project(
        _ input: MLXArray, weight: MLXArray, scales: MLXArray, biases: MLXArray?,
        groupSize: Int, bits: Int, reference: (MLXArray) -> MLXArray
    ) -> MLXArray? {
        guard input.ndim == 2, weight.ndim == 2, weight.dtype == .uint32, let biases,
            scales.ndim == 2, biases.dtype == scales.dtype,
            input.dim(1) == weight.dim(1) * (32 / max(bits, 1)), !CompiledDecodeTrace.isActive
        else { return nil }
        let rows = input.dim(0)
        let k = input.dim(1)
        let n = weight.dim(0)
        guard
            supports(
                rows: rows, k: k, n: n, bits: bits, groupSize: groupSize, inputDType: input.dtype,
                scalesDType: scales.dtype, hasBiases: true)
        else { return nil }
        let dtype = input.dtype
        func launch(_ x: MLXArray) -> MLXArray {
            kernel(
                [x, weight, scales, biases],
                template: [
                    ("T", dtype), ("S", scales.dtype), ("BITS", bits), ("GROUP_SIZE", groupSize),
                    ("ROWS", rows),
                ],
                grid: (32, (n / 8) * 2, 1), threadGroup: (32, 2, 1),
                outputShapes: [[rows, n]], outputDTypes: [dtype])[0]
        }
        let key = "\(dtype):\(scales.dtype):\(bits):\(groupSize):\(k):\(n):\(rows)"
        lock.lock()
        let known = verdicts[key]
        lock.unlock()
        if let known { return known ? launch(input) : nil }
        let probe = (MLXRandom.normal([rows, k]) * 0.5).asType(dtype)
        let got = launch(probe)
        let want = reference(probe)
        let equal =
            got.shape == want.shape
            && all(got.asType(.float32) .== want.asType(.float32)).item(Bool.self)
        lock.lock()
        verdicts[key] = equal
        lock.unlock()
        FileHandle.standardError.write(
            Data("[RowExactQMV] rows K=\(k) N=\(n) bits=\(bits) rows=\(rows) dtype=\(dtype) admitted=\(equal)\n".utf8))
        return equal ? launch(input) : nil
    }
}

/// Routed-expert `gather_qmv` for native-MTP verify rows, deduplicated by expert and bitwise equal,
/// pair by pair, to MLX's per-(row, expert) single-row kernels.
///
/// A W-row verify of a top-k MoE runs W x k (row, expert) pairs through `gather_qmv`, one M = 1
/// product per pair, so an expert that several verify rows selected streams its weights once per
/// row. Consecutive tokens share many experts (Super-VL, 512 experts / top-22: unique/total = 0.78 at
/// W = 2, 0.63 at W = 4, 0.50 at W = 8). Here the pairs are sorted by expert and the first pair of
/// each expert's run computes every pair of that run in one threadgroup: the expert's weight words are
/// fetched once, and each pair keeps the exact statements of `qmv_fast_impl` (aligned shapes) or the
/// tiled branch of `qmv_impl` (MLX's `gather_qmv` for unaligned K), transcribed verbatim with their
/// `load_vector(_safe)` / `qdot(_safe)` helpers. Admitted per shape only after a bitwise match
/// against MLX's own `gatherQuantizedMM` on random rows with forced duplicate experts.
public enum RowExactGatherQMV {
    /// OPT-IN (`VMLX_ROW_EXACT_GATHER_DEDUP=1`): on Super-VL (4-bit experts, M5 Max, GPU shared) the
    /// end-to-end verify cost moved by -5 %..+4 % at W = 4 / 8 and 0..+7 % at W = 2 — within noise.
    /// 4-bit per-pair dequantisation is compute-bound once several pairs share a threadgroup, which
    /// cancels most of the saved expert bytes. Re-measure on a quiet GPU before turning it on.
    static let enabled = ProcessInfo.processInfo.environment["VMLX_ROW_EXACT_GATHER_DEDUP"] == "1"
    private static let lock = NSLock()
    nonisolated(unsafe) private static var verdicts: [String: Bool] = [:]

    private static let header = RowExactAffineQMV.header + """

        // ---- verbatim from mlx/backend/metal/kernels/quantized.h (bits 4 and 8 branches) ----
        template <typename T, typename U, int values_per_thread, int bits>
        inline U rx_load_vector_safe(const device T* x, thread U* x_thread, int N) {
          U sum = 0;
          if (bits == 4) {
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
        inline U rx_qdot_safe(
            const device uint8_t* w,
            const thread U* x_thread,
            U scale,
            U bias,
            U sum,
            int N) {
          U accum = 0;
          if (bits == 4) {
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
        // ---- end verbatim ----

        // One expert run (pairs slot..slot+run-1 share the expert), up to MAXR pairs per pass.
        template <typename T, int group_size, int bits, int MAXR, bool FAST>
        inline void rx_gather_rows(
            const device uint32_t* w0, const device T* scales0, const device T* biases0,
            const device T* x, const device uint32_t* order, const device uint32_t* experts,
            device T* y, const int in_vec_size, const int out_vec_size, const int pairs,
            const int x_div, uint slot, uint tid_y, uint simd_gid, uint simd_lid) {
          const uint e = experts[slot];
          if (slot > 0 && experts[slot - 1] == e) return;
          int run = 1;
          while (int(slot) + run < pairs && experts[slot + run] == e) run++;

          constexpr int num_simdgroups = 2;
          constexpr int results_per_simdgroup = 4;
          constexpr int packs_per_thread = FAST ? 2 : 1;
          constexpr int pack_factor = 32 / bits;
          constexpr int bytes_per_pack = 4;
          constexpr int values_per_thread = pack_factor * packs_per_thread;
          constexpr int block_size = values_per_thread * 32;
          constexpr int scale_step_per_thread = group_size / values_per_thread;
          typedef float U;

          const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
          const int in_vec_size_g = in_vec_size / group_size;
          const int out_row = tid_y * (num_simdgroups * results_per_simdgroup) +
              simd_gid * results_per_simdgroup;
          if (out_row >= out_vec_size) return;
          const int used_out_row = FAST ? out_row : min(out_vec_size - results_per_simdgroup, out_row);

          const device uint8_t* wexp = (const device uint8_t*)(w0 + size_t(e) * out_vec_size * in_vec_size_w / 4);
          const device T* sexp = scales0 + size_t(e) * out_vec_size * in_vec_size_g;
          const device T* bexp = biases0 + size_t(e) * out_vec_size * in_vec_size_g;

          for (int base = 0; base < run; base += MAXR) {
            const int rows = min(MAXR, run - base);
            thread U x_thread[values_per_thread];
            thread float wf[results_per_simdgroup][bits == 8 ? values_per_thread : 1];
            thread U s_r[results_per_simdgroup];
            thread U b_r[results_per_simdgroup];
            thread U result[MAXR][results_per_simdgroup];
            const device T* xs[MAXR];
            device T* ys[MAXR];
            for (int m = 0; m < MAXR; m++) {
              for (int r = 0; r < results_per_simdgroup; r++) result[m][r] = 0;
              const uint pair = order[slot + base + min(m, rows - 1)];
              xs[m] = x + size_t(pair / x_div) * in_vec_size + simd_lid * values_per_thread;
              ys[m] = y + size_t(pair) * out_vec_size + used_out_row;
            }
            const device uint8_t* ws = wexp + used_out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
            const device T* scales = sexp + used_out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
            const device T* biases = bexp + used_out_row * in_vec_size_g + simd_lid / scale_step_per_thread;

            int k = 0;
            if (FAST) {
              for (; k < in_vec_size; k += block_size) {
                if (bits == 8) {
                  for (int row = 0; row < results_per_simdgroup; row++) {
                    rx_widen<values_per_thread, bits>(
                        (const device uint8_t*)(ws + row * in_vec_size_w), wf[row], values_per_thread);
                    s_r[row] = scales[row * in_vec_size_g];
                    b_r[row] = biases[row * in_vec_size_g];
                  }
                  for (int m = 0; m < rows; m++) {
                    U sum = rx_load_vector<T, U, values_per_thread, bits>(xs[m] + k, x_thread);
                    for (int row = 0; row < results_per_simdgroup; row++) {
                      result[m][row] += rx_qdot_f<U, values_per_thread, bits>(
                          wf[row], x_thread, s_r[row], b_r[row], sum, values_per_thread);
                    }
                  }
                } else {
                  for (int m = 0; m < rows; m++) {
                    U sum = rx_load_vector<T, U, values_per_thread, bits>(xs[m] + k, x_thread);
                    for (int row = 0; row < results_per_simdgroup; row++) {
                      auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
                      const device T* sl = scales + row * in_vec_size_g;
                      const device T* bl = biases + row * in_vec_size_g;
                      U s = sl[0];
                      U b = bl[0];
                      result[m][row] += rx_qdot<U, values_per_thread, bits>(wl, x_thread, s, b, sum);
                    }
                  }
                }
                ws += block_size * bytes_per_pack / pack_factor;
                scales += block_size / group_size;
                biases += block_size / group_size;
              }
            } else {
              for (; k < in_vec_size - block_size; k += block_size) {
                if (bits == 8) {
                  for (int row = 0; row < results_per_simdgroup; row++) {
                    rx_widen<values_per_thread, bits>(
                        (const device uint8_t*)(ws + row * in_vec_size_w), wf[row], values_per_thread);
                    s_r[row] = scales[row * in_vec_size_g];
                    b_r[row] = biases[row * in_vec_size_g];
                  }
                  for (int m = 0; m < rows; m++) {
                    U sum = rx_load_vector<T, U, values_per_thread, bits>(xs[m] + k, x_thread);
                    for (int row = 0; row < results_per_simdgroup; row++) {
                      result[m][row] += rx_qdot_f<U, values_per_thread, bits>(
                          wf[row], x_thread, s_r[row], b_r[row], sum, values_per_thread);
                    }
                  }
                } else {
                  for (int m = 0; m < rows; m++) {
                    U sum = rx_load_vector<T, U, values_per_thread, bits>(xs[m] + k, x_thread);
                    for (int row = 0; row < results_per_simdgroup; row++) {
                      auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
                      const device T* sl = scales + row * in_vec_size_g;
                      const device T* bl = biases + row * in_vec_size_g;
                      U s = sl[0];
                      U b = bl[0];
                      result[m][row] += rx_qdot<U, values_per_thread, bits>(wl, x_thread, s, b, sum);
                    }
                  }
                }
                ws += block_size * bytes_per_pack / pack_factor;
                scales += block_size / group_size;
                biases += block_size / group_size;
              }
              const int remaining = clamp(
                  static_cast<int>(in_vec_size - k - simd_lid * values_per_thread),
                  0,
                  values_per_thread);
              if (remaining > 0) {
                for (int m = 0; m < rows; m++) {
                  U sum = rx_load_vector_safe<T, U, values_per_thread, bits>(xs[m] + k, x_thread, remaining);
                  for (int row = 0; row < results_per_simdgroup; row++) {
                    auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
                    const device T* sl = scales + row * in_vec_size_g;
                    const device T* bl = biases + row * in_vec_size_g;
                    U s = sl[0];
                    U b = bl[0];
                    result[m][row] += rx_qdot_safe<U, values_per_thread, bits>(
                        wl, x_thread, s, b, sum, remaining);
                  }
                }
              }
            }
            for (int m = 0; m < rows; m++) {
              for (int row = 0; row < results_per_simdgroup; row++) {
                result[m][row] = simd_sum(result[m][row]);
                if (simd_lid == 0) {
                  ys[m][row] = static_cast<T>(result[m][row]);
                }
              }
            }
          }
        }
        """

    private static let kernel = MLXFast.metalKernel(
        name: "vmlx_row_exact_gather_qmv_dedup",
        inputNames: ["x", "w", "scales", "biases", "order", "experts"],
        outputNames: ["out"],
        source: """
            rx_gather_rows<T, GROUP_SIZE, BITS, MAXR, FAST>(
                w, scales, biases, x, order, experts, out,
                IN_VEC, OUT_VEC, PAIRS, X_DIV,
                threadgroup_position_in_grid.x, threadgroup_position_in_grid.y,
                simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
            """,
        header: header,
        ensureRowContiguous: true)

    /// `x` [X, K]; pair p uses row p / xDivisor of `x` and expert `indices[p]` ([P] uint32).
    /// Returns [P, N] or nil (unsupported / not admitted). `weight` [E, N, K * bits / 32] uint32,
    /// `scales` / `biases` [E, N, K / groupSize] in the activation dtype.
    public static func gather(
        _ x: MLXArray, indices: MLXArray, xDivisor: Int, maxRun: Int, weight: MLXArray,
        scales: MLXArray, biases: MLXArray, groupSize: Int, bits: Int
    ) -> MLXArray? {
        guard enabled, x.ndim == 2, indices.ndim == 1, weight.ndim == 3, scales.ndim == 3,
            biases.ndim == 3, bits == 4 || bits == 8, [32, 64, 128].contains(groupSize),
            x.dtype == .bfloat16 || x.dtype == .float16, scales.dtype == x.dtype,
            biases.dtype == x.dtype, weight.dtype == .uint32, (1 ... 8).contains(maxRun),
            !CompiledDecodeTrace.isActive
        else { return nil }
        let k = x.dim(1)
        let n = weight.dim(1)
        let pairs = indices.dim(0)
        guard weight.dim(2) * (32 / bits) == k, scales.dim(2) * groupSize == k, n >= 8,
            k % groupSize == 0, groupSize % (32 / bits) == 0
        else { return nil }
        let fast = n % 8 == 0 && k % ((32 / bits) * 2 * 32) == 0
        let dtype = x.dtype
        func launch(_ x: MLXArray, _ idx: MLXArray, divisor: Int) -> MLXArray {
            let idx = idx.asType(.uint32)
            let order = argSort(idx, axis: 0).asType(.uint32)
            let experts = idx[order]
            return kernel(
                [x, weight, scales, biases, order, experts],
                template: [
                    ("T", dtype), ("GROUP_SIZE", groupSize), ("BITS", bits), ("MAXR", maxRun),
                    ("FAST", fast), ("IN_VEC", k), ("OUT_VEC", n), ("PAIRS", idx.dim(0)),
                    ("X_DIV", divisor),
                ],
                grid: (idx.dim(0) * 32, ((n + 7) / 8) * 2, 1), threadGroup: (32, 2, 1),
                outputShapes: [[idx.dim(0), n]], outputDTypes: [dtype])[0]
        }
        let key = "\(dtype):\(bits):\(groupSize):\(k):\(n):\(pairs):\(xDivisor):\(maxRun)"
        lock.lock()
        let known = verdicts[key]
        lock.unlock()
        if let known { return known ? launch(x, indices, divisor: xDivisor) : nil }
        // Probe: random rows; experts drawn from a small pool so runs up to maxRun occur.
        let xRows = (pairs + xDivisor - 1) / xDivisor
        let probeX = (MLXRandom.normal([xRows, k]) * 0.5).asType(dtype)
        let pool = Swift.min(weight.dim(0), Swift.max(2, pairs / Swift.max(1, maxRun)))
        let probeIdx = MLXRandom.randInt(0 ..< pool, [pairs]).asType(.uint32)
        let got = launch(probeX, probeIdx, divisor: xDivisor)
        let rowOfPair = MLXArray(Array(0 ..< pairs).map { Int32($0 / xDivisor) })
        let want = gatherQuantizedMM(
            probeX[rowOfPair].reshaped(pairs, 1, k), weight, scales: scales, biases: biases,
            rhsIndices: probeIdx, transpose: true, groupSize: groupSize, bits: bits
        ).reshaped(pairs, n).asType(dtype)
        let equal = all(got.asType(.float32) .== want.asType(.float32)).item(Bool.self)
        lock.lock()
        verdicts[key] = equal
        lock.unlock()
        FileHandle.standardError.write(
            Data("[RowExactGather] K=\(k) N=\(n) bits=\(bits) pairs=\(pairs) fast=\(fast) admitted=\(equal)\n".utf8))
        return equal ? launch(x, indices, divisor: xDivisor) : nil
    }
}

extension QuantizedSwitchLinear {
    /// Verify rows through ``RowExactGatherQMV`` (affine, no bias); nil keeps `callAsFunction`.
    public func rowExactDedupGather(
        _ x: MLXArray, indices: MLXArray, xDivisor: Int, maxRun: Int
    ) -> MLXArray? {
        guard mode == .affine, bias == nil, let biases else { return nil }
        return RowExactGatherQMV.gather(
            x, indices: indices, xDivisor: xDivisor, maxRun: maxRun, weight: weight,
            scales: scales, biases: biases, groupSize: groupSize, bits: bits)
    }
}
