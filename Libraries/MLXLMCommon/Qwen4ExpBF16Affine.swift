// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN

/// Native BF16 affine kernels for Qwen3.8 Flash Next JANG bundles.
///
/// The bundle's scales and affine zero points are F16 storage payloads. They
/// are read as F16 and converted to the FP32 accumulator inside Metal; the
/// activation and result are BF16. This avoids MLX's generic BF16+F16 => F32
/// tensor promotion without rewriting checkpoint metadata or introducing an
/// F16 activation/output lane.
public enum Qwen4ExpBF16Affine {
    private static let diagnosticLock = NSLock()
    private nonisolated(unsafe) static var reportedSignatures = Set<String>()

    private static func reportDispatch(
        operation: String, input: MLXArray, weight: MLXArray,
        scales: MLXArray, biases: MLXArray?, groupSize: Int, bits: Int,
        native: Bool
    ) {
        let signature = "\(operation):\(input.dtype):\(weight.dtype):\(scales.dtype):\(String(describing: biases?.dtype)):\(groupSize):\(bits):\(native)"
        diagnosticLock.lock()
        let inserted = reportedSignatures.insert(signature).inserted
        diagnosticLock.unlock()
        guard inserted else { return }
        FileHandle.standardError.write(Data(
            ("[Qwen4Exp] bf16_affine_dispatch op=\(operation) native=\(native) "
                + "input=\(input.dtype) weight=\(weight.dtype) scales=\(scales.dtype) "
                + "biases=\(String(describing: biases?.dtype)) gs=\(groupSize) bits=\(bits)\n").utf8))
    }

    private static let qmvHeader = """
        template <int bits>
        inline constexpr short qwen_pack_factor() {
          return 32 / bits;
        }

        template <typename T, int values_per_thread, int bits>
        inline float qwen_load_vector(
            const device T* x, thread float* x_thread, int valid) {
          float sum = 0.0f;
          for (int i = 0; i < valid; ++i) {
            float value = float(x[i]);
            sum += value;
            if (bits == 4) {
              constexpr float divisors[4] = {1.0f, 16.0f, 256.0f, 4096.0f};
              x_thread[i] = value / divisors[i & 3];
            } else {
              x_thread[i] = value;
            }
          }
          for (int i = valid; i < values_per_thread; ++i) x_thread[i] = 0.0f;
          return sum;
        }

        template <int values_per_thread, int bits>
        inline float qwen_qdot(
            const device uchar* w, const thread float* x_thread,
            float scale, float bias, float sum, int valid) {
          float accum = 0.0f;
          if (bits == 4) {
            const device ushort* ws = (const device ushort*)w;
            for (int i = 0; i < (valid + 3) / 4; ++i) {
              ushort word = ws[i];
              int base = 4 * i;
              if (base < valid) accum += x_thread[base] * (word & 0x000f);
              if (base + 1 < valid) accum += x_thread[base + 1] * (word & 0x00f0);
              if (base + 2 < valid) accum += x_thread[base + 2] * (word & 0x0f00);
              if (base + 3 < valid) accum += x_thread[base + 3] * (word & 0xf000);
            }
          } else {
            for (int i = 0; i < valid; ++i) accum += x_thread[i] * w[i];
          }
          return scale * accum + sum * bias;
        }

        template <typename T, typename ScalePtr, typename BiasPtr,
                  int group_size, int bits, int packs_per_thread>
        inline void qwen_mixed_qmv(
            const device uint* w, ScalePtr scales,
            BiasPtr biases, const device T* x, device T* y,
            int in_vec_size, int out_vec_size,
            uint input_row, uint output_row, uint output_block,
            uint simd_gid, uint simd_lid) {
          constexpr int pack_factor = qwen_pack_factor<bits>();
          constexpr int values_per_thread = pack_factor * packs_per_thread;
          constexpr int block_size = values_per_thread * 32;
          constexpr int scale_step = group_size / values_per_thread;
          const int packed_row_bytes = in_vec_size * bits / 8;
          const int groups = in_vec_size / group_size;
          constexpr int results_per_simd = 4;
          uint n0 = output_block * 8 + simd_gid * results_per_simd;
          if (n0 >= out_vec_size) return;

          const device uchar* ws = (const device uchar*)w
              + n0 * packed_row_bytes
              + simd_lid * packs_per_thread * 4;
          ScalePtr ss = scales + n0 * groups + simd_lid / scale_step;
          BiasPtr bs = biases + n0 * groups + simd_lid / scale_step;
          const device T* xs = x + input_row * in_vec_size + simd_lid * values_per_thread;
          device T* ys = y + output_row * out_vec_size + n0;
          float result[results_per_simd] = {0.0f};
          float x_thread[values_per_thread];

          for (int k = 0; k < in_vec_size; k += block_size) {
            int valid = clamp(in_vec_size - k - int(simd_lid) * values_per_thread,
                              0, values_per_thread);
            float sum = qwen_load_vector<T, values_per_thread, bits>(
                xs, x_thread, valid);
            for (int r = 0; r < results_per_simd && n0 + r < out_vec_size; ++r) {
              result[r] += qwen_qdot<values_per_thread, bits>(
                  ws + r * packed_row_bytes, x_thread,
                  float(ss[r * groups]), float(bs[r * groups]), sum, valid);
            }
            ws += block_size * bits / 8;
            ss += block_size / group_size;
            bs += block_size / group_size;
            xs += block_size;
          }
          for (int r = 0; r < results_per_simd && n0 + r < out_vec_size; ++r) {
            result[r] = simd_sum(result[r]);
            if (simd_lid == 0) ys[r] = static_cast<T>(result[r]);
          }
        }
        """

    // METAL-ONLY: case 2. Metal kernel; `quantizedMM` computes the same
    // with MLX ops, but `supports` in `dense` does not pick it for the
    // shapes this kernel handles, so this path needs a Metal device.
    private static let denseKernel = MLXFast.metalKernel(
        name: "qwen4_exp_bf16_f16_affine_qmv_fast",
        inputNames: ["x", "w", "scales", "biases"],
        outputNames: ["out"],
        source: """
            qwen_mixed_qmv<T, decltype(scales), decltype(biases),
                           GROUP_SIZE, BITS, PACKS_PER_THREAD>(
                w, scales, biases, x, out,
                int(x_shape[x_ndim - 1]), int(w_shape[0]),
                threadgroup_position_in_grid.x,
                threadgroup_position_in_grid.x,
                threadgroup_position_in_grid.y,
                simdgroup_index_in_threadgroup,
                thread_index_in_simdgroup);
            """,
        header: qmvHeader,
        ensureRowContiguous: false)

    // METAL-ONLY: case 2. Metal kernel, off unless `Qwen4ExpBF16QuantizedSwitchLinear`
    // is installed, which `loadWeights` does only when it enables
    // `qwen4ExpNativeBF16Affine`; the default path computes the same with MLX ops.
    private static let gatheredKernel = MLXFast.metalKernel(
        name: "qwen4_exp_bf16_f16_affine_gather_qmv_fast",
        inputNames: ["x", "w", "scales", "biases", "indices"],
        outputNames: ["out"],
        source: """
            uint route = threadgroup_position_in_grid.x;
            uint input_row = route / ROUTES_PER_ROW;
            uint expert = uint(indices[route]);
            uint K = uint(x_shape[x_ndim - 1]);
            uint N = uint(w_shape[1]);
            constexpr uint VALUES_PER_WORD = 32u / BITS;
            uint PACKED_K = K / VALUES_PER_WORD;
            uint GROUPS = K / GROUP_SIZE;
            qwen_mixed_qmv<T, decltype(scales), decltype(biases),
                           GROUP_SIZE, BITS, PACKS_PER_THREAD>(
                w + expert * N * PACKED_K,
                scales + expert * N * GROUPS,
                biases + expert * N * GROUPS,
                x, out, int(K), int(N), input_row, route,
                threadgroup_position_in_grid.y,
                simdgroup_index_in_threadgroup,
                thread_index_in_simdgroup);
            """,
        header: qmvHeader,
        ensureRowContiguous: false)


    // METAL-ONLY: case 2. Metal kernel, off unless `Qwen4ExpBF16QuantizedEmbedding`
    // is installed, which `loadWeights` does only when it enables
    // `qwen4ExpNativeBF16Affine`; the default path computes the same with MLX ops.
    private static let embeddingKernel = MLXFast.metalKernel(
        name: "qwen4_exp_bf16_affine_embedding",
        inputNames: ["indices", "w", "scales", "biases"],
        outputNames: ["out"],
        source: """
            uint element = thread_position_in_grid.x;
            if (element >= ROWS * D) return;
            uint row = element / D;
            uint k = element - row * D;
            uint source_row = uint(indices[row]);
            constexpr uint VALUES_PER_WORD = 32u / BITS;
            constexpr uint PACKED_D = D / VALUES_PER_WORD;
            constexpr uint GROUPS = D / GROUP_SIZE;
            constexpr uint MASK = (1u << BITS) - 1u;
            uint word_index = k / VALUES_PER_WORD;
            uint word_lane = k - word_index * VALUES_PER_WORD;
            uint packed = w[source_row * PACKED_D + word_index];
            uint group = k / GROUP_SIZE;
            float scale = float(scales[source_row * GROUPS + group]);
            float zero = float(biases[source_row * GROUPS + group]);
            float code = float((packed >> (word_lane * BITS)) & MASK);
            out[element] = static_cast<T>(scale * code + zero);
            """,
        ensureRowContiguous: false)

    public static func supports(
        input: MLXArray, weight: MLXArray, scales: MLXArray,
        biases: MLXArray?, groupSize: Int, bits: Int,
        mode: QuantizationMode
    ) -> Bool {
        mode == .affine
            && input.dtype == .bfloat16
            && weight.dtype == .uint32
            && scales.dtype == .float16
            && biases?.dtype == .float16
            && groupSize == 64
            && (bits == 4 || bits == 8)
    }

    public static func dense(
        _ input: MLXArray, _ weight: MLXArray,
        scales: MLXArray, biases: MLXArray?,
        groupSize: Int, bits: Int, mode: QuantizationMode
    ) -> MLXArray {
        let native = supports(
            input: input, weight: weight, scales: scales, biases: biases,
            groupSize: groupSize, bits: bits, mode: mode)
        reportDispatch(
            operation: "dense", input: input, weight: weight, scales: scales,
            biases: biases, groupSize: groupSize, bits: bits, native: native)
        guard native,
            let biases,
            input.ndim >= 2, weight.ndim == 2
        else {
            // Stock quantizedMM silently promotes to f32 whenever the
            // activation and affine-metadata dtypes differ (e.g. the JANG_2L
            // 6-bit trunk: bf16 input x f16 scales). The projection contract
            // is that the output returns the incoming activation dtype;
            // without this the promoted result poisons the residual stream
            // and disables every dtype-gated fused kernel downstream
            // (measured 40 -> 25 tok/s on JANG_2L).
            return quantizedMM(
                input, weight, scales: scales, biases: biases,
                transpose: true, groupSize: groupSize, bits: bits, mode: mode
            ).asType(input.dtype)
        }
        let inputDimensions = weight.dim(1) * 32 / bits
        let outputDimensions = weight.dim(0)
        precondition(input.dim(-1) == inputDimensions)
        let rows = input.size / inputDimensions
        var outputShape = input.shape
        outputShape[outputShape.count - 1] = outputDimensions
        return denseKernel(
            [input, weight, scales, biases],
            template: [
                ("T", DType.bfloat16), ("BITS", bits),
                ("GROUP_SIZE", groupSize),
                ("PACKS_PER_THREAD", inputDimensions % 512 == 0 ? 2 : 1),
                ("ROWS", rows),
            ],
            grid: (rows * 32, ((outputDimensions + 7) / 8) * 2, 1),
            threadGroup: (32, 2, 1),
            outputShapes: [outputShape], outputDTypes: [.bfloat16])[0]
    }

    public static func gathered(
        _ input: MLXArray, _ weight: MLXArray,
        scales: MLXArray, biases: MLXArray?, indices: MLXArray,
        groupSize: Int, bits: Int, mode: QuantizationMode
    ) -> MLXArray? {
        let native = supports(
            input: input, weight: weight, scales: scales, biases: biases,
            groupSize: groupSize, bits: bits, mode: mode)
        reportDispatch(
            operation: "gathered", input: input, weight: weight, scales: scales,
            biases: biases, groupSize: groupSize, bits: bits, native: native)
        guard native,
            let biases,
            input.ndim >= 2, weight.ndim == 3
        else { return nil }
        let inputDimensions = weight.dim(2) * 32 / bits
        let outputDimensions = weight.dim(1)
        precondition(input.dim(-1) == inputDimensions)
        let inputRows = input.size / inputDimensions
        let routes = indices.size
        guard inputRows > 0, routes >= inputRows, routes % inputRows == 0 else {
            return nil
        }
        let routesPerRow = routes / inputRows
        let outputShape = indices.shape + [1, outputDimensions]
        return gatheredKernel(
            [input, weight, scales, biases, indices],
            template: [
                ("T", DType.bfloat16), ("BITS", bits),
                ("GROUP_SIZE", groupSize),
                ("PACKS_PER_THREAD", inputDimensions % 512 == 0 ? 2 : 1),
                ("ROUTES", routes),
                ("ROUTES_PER_ROW", routesPerRow),
            ],
            grid: (routes * 32, ((outputDimensions + 7) / 8) * 2, 1),
            threadGroup: (32, 2, 1),
            outputShapes: [outputShape], outputDTypes: [.bfloat16])[0]
    }

    public static func embedding(
        _ indices: MLXArray, _ weight: MLXArray,
        scales: MLXArray, biases: MLXArray?,
        groupSize: Int, bits: Int, mode: QuantizationMode
    ) -> MLXArray? {
        let dtypeProbe = MLXArray.zeros([1, weight.dim(1) * 32 / bits], dtype: .bfloat16)
        guard supports(
            input: dtypeProbe, weight: weight, scales: scales, biases: biases,
            groupSize: groupSize, bits: bits, mode: mode),
            let biases, weight.ndim == 2
        else { return nil }
        let dimensions = weight.dim(1) * 32 / bits
        let rows = indices.size
        return embeddingKernel(
            [indices.flattened(), weight, scales, biases],
            template: [
                ("T", DType.bfloat16), ("BITS", bits),
                ("GROUP_SIZE", groupSize), ("D", dimensions), ("ROWS", rows),
            ],
            grid: (rows * dimensions, 1, 1),
            threadGroup: (256, 1, 1),
            outputShapes: [indices.shape + [dimensions]], outputDTypes: [.bfloat16])[0]
    }
}

public final class Qwen4ExpBF16QuantizedLinear: QuantizedLinear {
    public override func callAsFunction(_ input: MLXArray) -> MLXArray {
        var output = Qwen4ExpBF16Affine.dense(
            input, weight, scales: scales, biases: biases,
            groupSize: groupSize, bits: bits, mode: mode)
        if let bias { output = output + bias.asType(.bfloat16) }
        return output
    }
}

public final class Qwen4ExpBF16QuantizedEmbedding: QuantizedEmbedding {
    public override func callAsFunction(_ input: MLXArray) -> MLXArray {
        Qwen4ExpBF16Affine.embedding(
            input, weight, scales: scales, biases: biases,
            groupSize: groupSize, bits: bits, mode: mode)
            ?? super.callAsFunction(input).asType(.bfloat16)
    }

    public override func asLinear(_ input: MLXArray) -> MLXArray {
        Qwen4ExpBF16Affine.dense(
            input, weight, scales: scales, biases: biases,
            groupSize: groupSize, bits: bits, mode: mode)
    }
}

public final class Qwen4ExpBF16QuantizedSwitchLinear: QuantizedSwitchLinear {
    public override func callAsFunction(
        _ input: MLXArray, _ indices: MLXArray, sortedIndices: Bool = false
    ) -> MLXArray {
        if let output = Qwen4ExpBF16Affine.gathered(
            input, weight, scales: scales, biases: biases, indices: indices,
            groupSize: groupSize, bits: bits, mode: mode)
        {
            if let bias {
                return output + MLX.expandedDimensions(
                    bias[indices].asType(.bfloat16), axis: -2)
            }
            return output
        }
        return super.callAsFunction(input, indices, sortedIndices: sortedIndices)
    }
}
