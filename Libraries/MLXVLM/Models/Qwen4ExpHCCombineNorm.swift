// Copyright © 2026 Apple Inc.
// Copyright © 2026 Osaurus AI. All rights reserved.

import Foundation
import MLX
import MLXLMCommon

/// Producer/consumer fusion: materialize the next residual AND its grouped HC
/// normalization in one dispatch. Each group owns one disjoint residual slice.
/// Arithmetic follows this runtime's MLX rms_single_row (not an FP32 residual
/// formulation).
/// Product, residual sum, normalization, and weight scaling round separately.
/// No live cache array is overwritten, and neither output is retained here.
final class Qwen4ExpHCCombineNorm {
    private static let traceAdmission = RuntimeEnvironment.flag(
        "VMLX_QWEN4_HC_NORM_ADMISSION_TRACE")
    private var didReportInputs = false
    let hiddenSize: Int
    private let epsilon: MLXArray
    private let width: MLXArray

    init(hiddenSize: Int, eps: Float) {
        self.hiddenSize = hiddenSize
        epsilon = MLXArray([eps])
        width = MLXArray([UInt32(max(0, hiddenSize))])
    }

    #if canImport(Metal)
        private static let kernel = MLXFast.metalKernel(
            name: "vmlx_qwen4_hc_combine_next_norm",
            inputNames: ["residual", "block", "inject", "weight", "eps", "width"],
            outputNames: ["next_residual", "normalized"],
            source: """
                uint stream = threadgroup_position_in_grid.x;
                uint lid = thread_position_in_threadgroup.x;
                uint lane = thread_index_in_simdgroup;
                uint sg = simdgroup_index_in_threadgroup;
                threadgroup float partials[32];
                threadgroup float inv_mean[1];
                T values[4];
                for (uint i = 0; i < 4; ++i) {
                    uint feature = lid * 4 + i;
                    values[i] = T(0);
                    if (feature < HIDDEN) {
                        {
                        #pragma clang fp contract(off)
                        #pragma clang fp reassociate(off)
                        // The ordinary graph promotes to the block dtype before
                        // adding the residual, then casts the result back to T.
                        B product = B(float(block[feature]) * float(inject[stream]));
                        values[i] = T(float(residual[stream * HIDDEN + feature]) + float(product));
                        }
                    }
                }
                // Match rms_single_row's runtime full/tail branch and reduction
                // structure, independently of the residual producer's guards.
                float acc = 0.0f;
                if (lid * 4 + 4 <= width[0]) {
                    for (int i = 0; i < 4; ++i) {
                        float xi = float(values[i]);
                        acc += xi * xi;
                    }
                } else {
                    for (int i = 0; i < 4; ++i) {
                        if (lid * 4 + i < width[0]) {
                            float xi = float(values[i]);
                            acc += xi * xi;
                        }
                    }
                }
                acc = simd_sum(acc);
                if (sg == 0) partials[lane] = 0.0f;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (lane == 0) partials[sg] = acc;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (sg == 0) {
                    acc = simd_sum(partials[lane]);
                    if (lane == 0) {
                        // AOT fast RMS contracts its reciprocal multiply and
                        // epsilon addition. A separately rounded mean is NOT
                        // equivalent near BF16/FP16 normalization boundaries.
                        float mean_eps = RECIPROCAL_MEAN
                            ? fma(acc, (1.0f / float(width[0])), eps[0])
                            : (acc / width[0] + eps[0]);
                        inv_mean[0] = metal::precise::rsqrt(mean_eps);
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                {
                #pragma clang fp contract(off)
                #pragma clang fp reassociate(off)
                for (uint i = 0; i < 4; ++i) {
                    uint feature = lid * 4 + i;
                    if (feature < HIDDEN) {
                        uint index = stream * HIDDEN + feature;
                        next_residual[index] = values[i];
                        T normed = T(float(values[i]) * inv_mean[0]);
                        normalized[index] = T(float(normed) * float(weight[index]));
                    }
                }
                }
                """)

        /// Xcode's AOT RMS library and MLX's runtime custom kernels can use
        /// different math flags. Qualify against the library actually loaded,
        /// once per process, rather than infer its arithmetic from build settings.
        /// This uses bounded generated FP32 rows, not model data or cache state.
        /// Neither/ambiguous matches leave the ordinary combine + RMS path intact.
        private static let reciprocalMean: Bool? = {
            let hidden = 96
            let streams = 4
            let values: [Float] = (0 ..< (hidden * streams)).map { (index: Int) -> Float in
                let integer: Int = (index * 17 + 3) % 127 - 63
                return Float(integer) / 16
            }
            let residual = MLXArray(values).reshaped(1, 1, -1)
            let block = MLXArray.zeros([1, 1, hidden])
            let injection = MLXArray.ones([1, 1, streams])
            let weight = MLXArray.ones([hidden * streams])
            let eps = MLXArray([Float(1e-6)])
            let width = MLXArray([UInt32(hidden)])
            let reference = MLXFast.rmsNorm(
                residual.reshaped(streams, hidden), weight: .mlxNone, eps: 1e-6)
            let expected = reference.asArray(Float.self).map(\.bitPattern)
            let matches = [false, true].filter { reciprocal in
                let result = kernel(
                    [residual, block, injection, weight, eps, width],
                    template: [
                        ("T", DType.float32), ("B", DType.float32),
                        ("HIDDEN", hidden), ("RECIPROCAL_MEAN", reciprocal),
                    ],
                    grid: (streams * 32, 1, 1), threadGroup: (32, 1, 1),
                    outputShapes: [residual.shape, residual.shape],
                    outputDTypes: [.float32, .float32])
                return result[1].asArray(Float.self).map(\.bitPattern) == expected
            }
            let mode = matches.count == 1 ? matches[0] : nil
            NSLog(
                "[HCCombineNorm] rms_mean_arithmetic=%@ qualification_rows=4 width=96 source=loaded_rms_library",
                mode.map { $0 ? "reciprocal_fma" : "division" } ?? "unqualified_fallback")
            return mode
        }()
    #endif

    func callAsFunction(
        residual: MLXArray, block: MLXArray, injection: MLXArray, weight: MLXArray
    ) -> (residual: MLXArray, normalized: MLXArray)? {
        #if canImport(Metal)
            guard !CompiledDecodeTrace.isActive,
                Device.defaultDevice().deviceType == .gpu,
                hiddenSize >= 32, hiddenSize <= 4096,
                residual.ndim == 3, residual.dim(0) == 1, residual.dim(1) == 1,
                block.shape == [1, 1, hiddenSize],
                injection.ndim == 3, injection.dim(0) == 1, injection.dim(1) == 1,
                injection.dim(2) > 0,
                residual.dim(2) == injection.dim(2) * hiddenSize,
                weight.shape == [residual.dim(2)],
                residual.dtype == .bfloat16 || residual.dtype == .float16,
                block.dtype == residual.dtype || block.dtype == .float32,
                injection.dtype == residual.dtype,
                weight.dtype == residual.dtype,
                let reciprocalMean = Self.reciprocalMean
            else { return nil }
            if Self.traceAdmission, !didReportInputs {
                didReportInputs = true
                NSLog(
                    "[HCCombineNorm admission] residual=%@ block=%@ injection=%@ weight=%@ hidden=%d streams=%d",
                    String(describing: residual.dtype), String(describing: block.dtype),
                    String(describing: injection.dtype), String(describing: weight.dtype),
                    hiddenSize, injection.dim(2))
            }
            let threads = ((hiddenSize + 127) / 128) * 32
            let result = Self.kernel(
                [residual, block, injection, weight, epsilon, width],
                template: [
                    ("T", residual.dtype), ("B", block.dtype), ("HIDDEN", hiddenSize),
                    ("RECIPROCAL_MEAN", reciprocalMean),
                ],
                grid: (injection.dim(2) * threads, 1, 1), threadGroup: (threads, 1, 1),
                outputShapes: [residual.shape, residual.shape],
                outputDTypes: [residual.dtype, residual.dtype])
            return (result[0], result[1])
        #else
            return nil
        #endif
    }
}
