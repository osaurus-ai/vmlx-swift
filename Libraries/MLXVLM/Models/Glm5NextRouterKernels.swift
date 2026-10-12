// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXFast

/// GLM-5.3 router for decode and short verify slabs in two launches. Port of vMLX Python `jangt/router.py`
/// (accepted 2026-10-10: router tail +3.4 %, router logits +3.4 % GLM decode; teacher-forced KL 0.0028 / 0.0030).
///
/// - logits: one SIMD group per (token, expert) row, fp32 accumulation of the bf16 hidden state against the bf16
///   router weight (the bundle declares `moe_router_dtype: float32`; the graph form multiplied in bf16).
/// - tail: one threadgroup per token, one thread per expert: sigmoid, + correction bias for the CHOICE only, top-k by
///   repeated threadgroup argmax (ties -> lower expert id), unbiased scores of the winners, normalize, scale. The
///   cross-SIMD-group scratch has one slot per SIMD group up to 1024 experts (the reference's first version sized it
///   for 8 groups = 256 experts and chose wrong experts for GLM's 288).
/// Applies only with one expert group (`n_group` 1); `VMLX_GLM5_ROUTER_KERNELS=0` keeps the graph.
enum Glm5NextRouterKernels {
    static let enabled: Bool = {
        let raw = (ProcessInfo.processInfo.environment["VMLX_GLM5_ROUTER_KERNELS"] ?? "1").lowercased()
        return !["0", "false", "off", "no"].contains(raw)
    }()

    private static let logitsKernel = MLXFast.metalKernel(
        name: "vmlx_glm5_router_logits",
        inputNames: ["x", "w"], outputNames: ["logits"],
        source: """
            uint row = thread_position_in_grid.x / 32u;
            uint lane = thread_index_in_simdgroup;
            if (row >= T_ROWS * NE) return;
            uint t = row / NE, e = row % NE;
            device const T* xr = x + (size_t)t * D;
            device const T* wr = w + (size_t)e * D;
            float acc = 0.0f;
            for (uint k = lane * 4u; k < D; k += 128u) {
                acc += (float)xr[k] * (float)wr[k] + (float)xr[k + 1u] * (float)wr[k + 1u]
                     + (float)xr[k + 2u] * (float)wr[k + 2u] + (float)xr[k + 3u] * (float)wr[k + 3u];
            }
            acc = simd_sum(acc);
            if (lane == 0u) logits[(size_t)t * NE + e] = acc;
            """)

    private static let tailKernel = MLXFast.metalKernel(
        name: "vmlx_glm5_router_tail",
        inputNames: ["logits", "bias", "scaling"], outputNames: ["idx", "w"],
        source: """
            uint t = threadgroup_position_in_grid.x;
            uint e = thread_position_in_threadgroup.x;
            uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
            threadgroup float bv[32]; threadgroup int bi[32]; threadgroup int win[TOPK]; threadgroup float wsc[TOPK];
            float s = 1.0f / (1.0f + metal::exp(-logits[t * NE + e]));
            float c = s + float(bias[e]);
            for (int r = 0; r < TOPK; ++r) {
                float v = c; int i = (int)e;
                for (ushort o = 16; o > 0; o >>= 1) {
                    float v2 = simd_shuffle_down(v, o); int i2 = simd_shuffle_down(i, o);
                    if (v2 > v || (v2 == v && i2 < i)) { v = v2; i = i2; }
                }
                if (lane == 0) { bv[sg] = v; bi[sg] = i; }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (e == 0) {
                    float bb = bv[0]; int ib = bi[0];
                    for (int g = 1; g < NE / 32; ++g) if (bv[g] > bb || (bv[g] == bb && bi[g] < ib)) { bb = bv[g]; ib = bi[g]; }
                    win[r] = ib;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if ((int)e == win[r]) { wsc[r] = s; c = -INFINITY; }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (e < TOPK) {
                float tot = 0.0f;
                for (int r = 0; r < TOPK; ++r) tot += wsc[r];
                idx[t * TOPK + e] = (uint)win[e];
                w[t * TOPK + e] = (NORM ? wsc[e] / (tot + 1e-20f) : wsc[e]) * scaling[0];
            }
            """)

    /// `(indices uint32 [..., k], weights fp32 [..., k])`, or nil when the shapes are not the decode/verify ones.
    static func route(
        _ x: MLXArray, weight: MLXArray, correctionBias: MLXArray, topK: Int, normalize: Bool, scaling: Float
    ) -> (indices: MLXArray, weights: MLXArray)? {
        let d = x.dim(-1), experts = weight.dim(0)
        let rows = x.size / max(d, 1)
        guard enabled, weight.ndim == 2, weight.dim(1) == d, d % 128 == 0, x.dtype == weight.dtype,
            [.bfloat16, .float16].contains(x.dtype), rows >= 1, rows <= 64,
            experts % 32 == 0, experts <= 1024, topK >= 1, topK <= 32, correctionBias.size == experts
        else { return nil }
        let flat = x.reshaped(rows, d)
        let logits = logitsKernel(
            [flat, weight], template: [("T", x.dtype), ("T_ROWS", rows), ("NE", experts), ("D", d)],
            grid: (32 * rows * experts, 1, 1), threadGroup: (256, 1, 1),
            outputShapes: [[rows, experts]], outputDTypes: [.float32])[0]
        let out = tailKernel(
            [logits, correctionBias.asType(.float32), MLXArray([scaling])],
            template: [("NE", experts), ("TOPK", topK), ("NORM", normalize)],
            grid: (rows * experts, 1, 1), threadGroup: (experts, 1, 1),
            outputShapes: [[rows, topK], [rows, topK]], outputDTypes: [.uint32, .float32])
        let lead = Array(x.shape.dropLast())
        return (out[0].reshaped(lead + [topK]), out[1].reshaped(lead + [topK]))
    }
}
