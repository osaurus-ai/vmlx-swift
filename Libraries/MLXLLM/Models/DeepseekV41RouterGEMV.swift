// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// DeepSeek-V4.1 router logits at decode: `x.astype(f32) @ weight.astype(f32).T` for one token, read
// straight from the bf16 router rows.
//
// The stock gate casts the [384, 5120] bf16 router weight to fp32 on every call (an 7.9 MB write and
// read per layer on top of the 3.9 MB bf16 read, ~0.47 GB per token over 40 layers) and then runs
// MLX's fp32 GEMV: three dispatches per layer. This kernel folds both casts into the loads and keeps
// MLX's own GEMV order for this shape (`gemv_float32_bm4_bn1_sm1_sn32_tm4_tn4`, picked by
// `gemv_axbpy` for out 384 < 4096 and K < 16 * out): each simdgroup owns 4 output rows, lane l adds
// k = 128 i + 4 l + {0..3} in that order into one fp32 accumulator per row, then a shuffle-down ladder
// 16, 8, 4, 2, 1. Each logit is therefore bit-identical to the stock fp32 matmul (checked by
// `DeepseekV41DecodeFusionFocusedTests`). Pattern: vMLX `vmlx_engine/metal/glm5_router_matvec.py`
// (GLM 288 x 4096), donor idea mlx-serve `GATE_SOURCE` / ds4 `kernel_dsv41_router_one`.
//
// Decode only (one row). dSpark verify rows (2-8) keep the stock fp32 matmul: MLX runs those as a GEMM
// with a different reduction order, and this kernel must not change their bits.

import Foundation
import MLX
import MLXFast

enum DeepseekV41RouterGEMV {
    /// `VMLX_DSV41_ROUTER_KERNEL=0` (or false/off/no) keeps the stock fp32 cast + matmul. Read once;
    /// tests flip it to time and compare both paths in one process.
    nonisolated(unsafe) static var enabled: Bool = {
        let env = ProcessInfo.processInfo.environment
        // VMLX_DSV41_ROUTER_GEMV is the Python runtime's name for the same switch.
        let raw = env["VMLX_DSV41_ROUTER_KERNEL"] ?? env["VMLX_DSV41_ROUTER_GEMV"] ?? "1"
        return !["0", "false", "off", "no"].contains(
            raw.trimmingCharacters(in: .whitespaces).lowercased())
    }()

    // METAL-ONLY: case 2. Metal kernel; without a Metal device the caller keeps the stock f32 router matmul.
    private static let kernel = MLXFast.metalKernel(
        name: "vmlx_dsv41_router_bf16_gemv_f32",
        inputNames: ["x", "weight"],
        outputNames: ["logits"],
        source: """
            const uint lane = thread_index_in_simdgroup;
            const uint row = threadgroup_position_in_grid.x * 16u + simdgroup_index_in_threadgroup * 4u;
            float accum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (uint base = 0u; base < uint(K); base += 128u) {
                const uint k = base + lane * 4u;
                float v[4];
                #pragma clang loop unroll(full)
                for (uint j = 0u; j < 4u; ++j) v[j] = float(x[k + j]);
                #pragma clang loop unroll(full)
                for (uint r = 0u; r < 4u; ++r) {
                    #pragma clang loop unroll(full)
                    for (uint j = 0u; j < 4u; ++j)
                        accum[r] += float(weight[size_t(row + r) * uint(K) + k + j]) * v[j];
                }
            }
            #pragma clang loop unroll(full)
            for (uint r = 0u; r < 4u; ++r) {
                #pragma clang loop unroll(full)
                for (ushort offset = 16u; offset >= 1u; offset >>= 1u)
                    accum[r] += simd_shuffle_down(accum[r], offset);
                if (lane == 0u) logits[row + r] = accum[r];
            }
            """)

    /// MLX's GEMV picks this tiling only for out < 4096 rows and K < 16 * out; K a multiple of the
    /// 128-wide block (no leftover pass) and out a multiple of 16 (no clamped last tile).
    static func admits(rows: Int, experts: Int, hidden: Int) -> Bool {
        rows == 1 && experts % 16 == 0 && experts < 4096 && hidden % 128 == 0
            && hidden < 16 * experts
    }

    /// fp32 logits `[..., experts]` for one token, or nil to keep the stock path.
    static func logits(_ x: MLXArray, weight: MLXArray) -> MLXArray? {
        guard enabled, weight.ndim == 2, weight.dtype == .bfloat16,
            x.dtype == .bfloat16 || x.dtype == .float32 || x.dtype == .float16
        else { return nil }
        let experts = weight.dim(0)
        let hidden = weight.dim(1)
        guard x.dim(-1) == hidden, admits(rows: x.size / hidden, experts: experts, hidden: hidden),
            // METAL-ONLY: case 1. Guards the Metal kernel; other devices keep the stock matmul.
            Device.defaultDevice().deviceType == .gpu
        else { return nil }
        let out = kernel(
            [x, weight],
            template: [("K", hidden)],
            grid: (experts / 16 * 128, 1, 1),
            threadGroup: (128, 1, 1),
            outputShapes: [Array(x.shape.dropLast()) + [experts]],
            outputDTypes: [.float32])
        return out[0]
    }
}
