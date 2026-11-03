// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Cmlx
import CmlxGraphShim
import Foundation
import MLX
import MLXLMCommon

/// Single-row K2 grouped RMSNorm followed by its full-width checkpoint weight.
/// The reduction and BF16 rounding boundary match the pinned native RMSNorm;
/// other layouts, devices, dtypes and transformed graphs retain the reference.
enum K2GroupedNormFusion {
    /// Read once before the first forward, for a reference-only A/B process.
    static let referenceOverride =
        ProcessInfo.processInfo.environment["VMLX_K2_GROUPED_NORM_REFERENCE"] == "1"

    static func apply(
        _ x: MLXArray, weight: MLXArray, groups: Int, eps: Float,
        forceReference: Bool = false
    ) -> MLXArray? {
        #if canImport(Metal)
            guard !forceReference, !referenceOverride, !CompiledDecodeTrace.isActive,
                defaultStreamIsGPU(), groups == 4, eps == 1e-6,
                x.shape == [1, 1, 4096], weight.shape == [4096],
                x.dtype == .bfloat16, weight.dtype == .bfloat16,
                vmlx_graph_array_is_tracer(x.ctx.ctx) == 0,
                vmlx_graph_array_is_tracer(weight.ctx.ctx) == 0,
                hasRowContiguousMetadata(x), hasRowContiguousMetadata(weight)
            else { return nil }

            // Metadata may change when lazy inputs are evaluated. The kernel's
            // ensureRowContiguous also checks the final layout; do not add an
            // eval/readback here merely to stabilize an admission check.
            return kernel(
                [x, weight], template: [("T", DType.bfloat16)],
                grid: (1024, 1, 1), threadGroup: (256, 1, 1),
                outputShapes: [x.shape], outputDTypes: [.bfloat16]
            )[0]
        #else
            return nil
        #endif
    }

    private static func hasRowContiguousMetadata(_ array: MLXArray) -> Bool {
        var contiguous = false
        return _mlx_array_is_row_contiguous(&contiguous, array.ctx) == 0 && contiguous
    }

    #if canImport(Metal)
        // Reduction adapted from MLX rms_norm.metal at native core 3b39b5e:
        // 4 reads/thread, 256 threads/group, two SIMD reductions, precise rsqrt.
        // The checkpoint weight is indexed by group only after normalization
        // has rounded to BF16, as in rmsNorm(..., weight: ones) * weight.
        private static let kernel = MLXFast.metalKernel(
            name: "k2_grouped_bf16_norm_weight", inputNames: ["x", "w"],
            outputNames: ["out"],
            source: """
                constexpr uint axis_size = 1024;
                constexpr uint n_reads = 4;
                constexpr uint simd_size = 32;
                uint gid = threadgroup_position_in_grid.x;
                uint lid = thread_position_in_threadgroup.x;
                uint lane = thread_index_in_simdgroup;
                uint simd_group = simdgroup_index_in_threadgroup;
                uint offset = gid * axis_size + lid * n_reads;

                threadgroup float local_inv_mean[1];
                threadgroup float local_sums[simd_size];
                float acc = 0;
                float thread_x[n_reads];
                for (int i = 0; i < n_reads; i++) {
                    thread_x[i] = x[offset + i];
                    acc += thread_x[i] * thread_x[i];
                }
                acc = simd_sum(acc);
                if (simd_group == 0) {
                    local_sums[lane] = 0;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (lane == 0) {
                    local_sums[simd_group] = acc;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (simd_group == 0) {
                    acc = simd_sum(local_sums[lane]);
                    if (lane == 0) {
                        local_inv_mean[0] = metal::precise::rsqrt(acc / axis_size + 1e-6f);
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (int i = 0; i < n_reads; i++) {
                    T normalized = static_cast<T>(thread_x[i] * local_inv_mean[0]);
                    T unit_weighted = T(1) * normalized;
                    out[offset + i] = unit_weighted * w[offset + i];
                }
                """,
            header: """
                // Copyright © 2024 Apple Inc.
                // Copyright © 2023 Apple Inc.
                //
                // Permission is hereby granted, free of charge, to any person obtaining a copy
                // of this software and associated documentation files (the "Software"), to deal
                // in the Software without restriction, including without limitation the rights
                // to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
                // copies of the Software, and to permit persons to whom the Software is
                // furnished to do so, subject to the following conditions:
                //
                // The above copyright notice and this permission notice shall be included in all
                // copies or substantial portions of the Software.
                //
                // THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
                // IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
                // FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
                // AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
                // LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
                // OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
                // SOFTWARE.
                """ + "\n",
            ensureRowContiguous: true
        )
    #endif
}
