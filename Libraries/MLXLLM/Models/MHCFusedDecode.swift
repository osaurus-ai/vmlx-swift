// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXFast

/// Two-dispatch mHC collapse for decode and short verify slabs (4 hyper-connection streams, batch 1). Port of vMLX
/// Python `metal/glm5_mhc_decode.py` v3 (accepted 2026-10-10: GLM decode 27.6 -> 32.1 tok/s, +16.5 %, teacher-forced
/// KL 0.0027; kernel post/comb <= 5e-7).
///
/// The graph form (fp32 cast of the streams AND of the `fn` weight, a 24-row fp32 GEMV, the Sinkhorn kernel, the
/// collapse) runs twice per GLM decoder layer — 90 calls per token, each a short dependent chain whose small kernels
/// cost their latency. Here:
/// 1. projection: one 1024-thread threadgroup per (row, mix entry) computes `rms(streams) * streams . fn[entry]` in
///    fp32, reading `fn` in its stored dtype;
/// 2. epilogue: pre = sigmoid + eps, post = 2 sigmoid, comb = Sinkhorn of the 4 x 4 softmax on 16 SIMD lanes (row sums
///    over lane bits 0-1, column sums over bits 2-3), and the pre-weighted collapse of the streams.
/// Same formulas as `DeepseekV4Math.hcPreGraph` / `hcSplitSinkhorn`; summation order differs (fp32 pairwise).
public enum MHCFusedDecode {
    public static let enabled: Bool = {
        let raw = (ProcessInfo.processInfo.environment["VMLX_MHC_FUSED_DECODE"] ?? "1").lowercased()
        return !["0", "false", "off", "no"].contains(raw)
    }()

    static let maxRows = 16

    private static let projection = MLXFast.metalKernel(
        name: "vmlx_mhc_projection_v3",
        inputNames: ["streams", "hc_fn", "rms_eps"], outputNames: ["mix"],
        source: """
            uint group = thread_position_in_grid.x / PROJECTION_THREADS;
            uint token = group / MIX_SIZE;
            uint row = group % MIX_SIZE;
            uint tid = thread_index_in_threadgroup;
            uint lane = thread_index_in_simdgroup;
            uint simd_id = simdgroup_index_in_threadgroup;
            if (token >= ROWS || row >= MIX_SIZE) return;
            threadgroup float partials[PROJECTION_SIMDGROUPS];
            threadgroup float inv_rms;
            float sumsq = 0.0f;
            for (uint col = tid; col < FEATURES; col += PROJECTION_THREADS) {
                float value = float(streams[(size_t)token * FEATURES + col]);
                sumsq += value * value;
            }
            sumsq = simd_sum(sumsq);
            if (lane == 0u) partials[simd_id] = sumsq;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid == 0u) {
                float total = 0.0f;
                for (uint index = 0u; index < PROJECTION_SIMDGROUPS; ++index) total += partials[index];
                inv_rms = metal::rsqrt(total / float(FEATURES) + float(rms_eps[0]));
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float value = 0.0f;
            size_t weight_base = (size_t)row * (size_t)FEATURES;
            for (uint col = tid; col < FEATURES; col += PROJECTION_THREADS)
                value += float(streams[(size_t)token * FEATURES + col]) * inv_rms * float(hc_fn[weight_base + col]);
            value = simd_sum(value);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (lane == 0u) partials[simd_id] = value;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid == 0u) {
                float total = 0.0f;
                for (uint index = 0u; index < PROJECTION_SIMDGROUPS; ++index) total += partials[index];
                mix[(size_t)token * MIX_SIZE + row] = total;
            }
            """)

    private static let epilogue = MLXFast.metalKernel(
        name: "vmlx_mhc_epilogue_v3",
        inputNames: ["streams", "mix", "hc_base", "hc_scale", "sink_eps"],
        outputNames: ["post", "comb", "collapsed"],
        source: """
            uint tid = thread_index_in_threadgroup;
            uint token = thread_position_in_grid.x / EPILOGUE_THREADS;
            if (token >= ROWS) return;
            size_t streams_base = (size_t)token * H * D;
            size_t mix_base = (size_t)token * MIX_SIZE;
            threadgroup float pre_values[H];
            threadgroup float post_values[H];
            threadgroup float matrix[H * H];
            if (simdgroup_index_in_threadgroup == 0u) {
                uint lane = thread_index_in_simdgroup;
                float scale_pre = float(hc_scale[0]);
                float scale_post = float(hc_scale[1]);
                float scale_comb = float(hc_scale[2]);
                float epsilon = float(sink_eps[0]);
                if (lane < H) {
                    float pre_x = float(mix[mix_base + lane]) * scale_pre + float(hc_base[lane]);
                    pre_values[lane] = 1.0f / (1.0f + metal::exp(-pre_x)) + epsilon;
                    float post_x = float(mix[mix_base + H + lane]) * scale_post + float(hc_base[H + lane]);
                    post_values[lane] = 2.0f / (1.0f + metal::exp(-post_x));
                }
                uint e = lane & 15u;
                float item = float(mix[mix_base + 2u * H + e]) * scale_comb + float(hc_base[2u * H + e]);
                float m = metal::max(item, simd_shuffle_xor(item, 1u));
                m = metal::max(m, simd_shuffle_xor(m, 2u));
                float ex = metal::exp(item - m);
                float rs = ex + simd_shuffle_xor(ex, 1u);
                rs = rs + simd_shuffle_xor(rs, 2u);
                float v = ex / rs + epsilon;
                float cs = v + simd_shuffle_xor(v, 4u);
                cs = cs + simd_shuffle_xor(cs, 8u);
                v = v / (cs + epsilon);
                for (uint iteration = 1u; iteration < ITERS; ++iteration) {
                    rs = v + simd_shuffle_xor(v, 1u);
                    rs = rs + simd_shuffle_xor(rs, 2u);
                    v = v / (rs + epsilon);
                    cs = v + simd_shuffle_xor(v, 4u);
                    cs = cs + simd_shuffle_xor(cs, 8u);
                    v = v / (cs + epsilon);
                }
                if (lane < 16u) matrix[lane] = v;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint dim = tid; dim < D; dim += EPILOGUE_THREADS) {
                float value = 0.0f;
                for (uint stream = 0u; stream < H; ++stream)
                    value += pre_values[stream] * float(streams[streams_base + stream * D + dim]);
                collapsed[(size_t)token * D + dim] = T(value);
            }
            for (uint index = tid; index < H; index += EPILOGUE_THREADS) post[(size_t)token * H + index] = post_values[index];
            for (uint index = tid; index < H * H; index += EPILOGUE_THREADS) comb[(size_t)token * H * H + index] = matrix[index];
            """)

    /// `(collapsed, post, comb)` for `streams` `[1, L, 4, D]`, or nil for any other geometry (caller keeps the graph).
    public static func collapse(
        _ streams: MLXArray, fn: MLXArray, scale: MLXArray, base: MLXArray,
        iters: Int, eps: Float, normEps: Float
    ) -> (x: MLXArray, post: MLXArray, comb: MLXArray)? {
        guard enabled, streams.ndim == 4, streams.dim(0) == 1, streams.dim(2) == 4,
            (1 ... maxRows).contains(streams.dim(1)), iters >= 1, iters <= 64,
            [.bfloat16, .float16, .float32].contains(streams.dtype)
        else { return nil }
        let rows = streams.dim(1), h = 4, d = streams.dim(3)
        let mixSize = (2 + h) * h, features = h * d
        guard fn.shape == [mixSize, features], base.shape == [mixSize], scale.shape == [3] else { return nil }
        let threads = 1024
        let mix = projection(
            [streams, fn, MLXArray([normEps])],
            template: [("FEATURES", features), ("MIX_SIZE", mixSize), ("ROWS", rows),
                       ("PROJECTION_THREADS", threads), ("PROJECTION_SIMDGROUPS", threads / 32)],
            grid: (threads * mixSize * rows, 1, 1), threadGroup: (threads, 1, 1),
            outputShapes: [[rows, mixSize]], outputDTypes: [.float32])[0]
        let out = epilogue(
            [streams, mix, base, scale, MLXArray([eps])],
            template: [("T", streams.dtype), ("H", h), ("D", d), ("ROWS", rows), ("MIX_SIZE", mixSize),
                       ("ITERS", iters), ("EPILOGUE_THREADS", 256)],
            grid: (256 * rows, 1, 1), threadGroup: (256, 1, 1),
            outputShapes: [[1, rows, h], [1, rows, h, h], [1, rows, d]],
            outputDTypes: [.float32, .float32, streams.dtype])
        return (x: out[2], post: out[0], comb: out[1])
    }
}
