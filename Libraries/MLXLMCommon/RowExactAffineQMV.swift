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

    private static let header = """
        // ---- verbatim from mlx/backend/metal/kernels/quantized.h (bits 4 and 8 branches) ----
        template <typename T, typename U, int values_per_thread, int bits>
        inline U rx_load_vector(const device T* x, thread U* x_thread) {
          U sum = 0;
          if (bits == 4) {
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

          for (int k = 0; k < in_vec_size; k += block_size) {
            for (int m = 0; m < ROWS; m++) {
              U sum = rx_load_vector<T, U, values_per_thread, bits>(x + m * in_vec_size, x_thread);
              for (int row = 0; row < results_per_simdgroup; row++) {
                auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
                const device S* sl = scales + row * in_vec_size_g;
                const device S* bl = biases + row * in_vec_size_g;
                U s = sl[0];
                U b = bl[0];
                result[m][row] += rx_qdot<U, values_per_thread, bits>(wl, x_thread, s, b, sum);
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
        guard enabled, (2 ... 8).contains(rows), bits == 4 || bits == 8, hasBiases,
            [32, 64, 128].contains(groupSize),
            inputDType == .bfloat16 || inputDType == .float16,
            scalesDType == inputDType
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
