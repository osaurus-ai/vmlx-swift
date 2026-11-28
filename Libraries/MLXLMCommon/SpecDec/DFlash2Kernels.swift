// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Fused Apple Silicon Metal kernels for DFlash 2 drafting:
// 1. `groupedDynamicConvolveK2`: 1-dispatch causal grouped dynamic convolution (K=2)
//    replacing ~8 MLX ops x 20 calls per draft round.
// 2. `topK`: 2-stage exact top-K (K <= 24) over unpadded vocab (~200K) replacing
//    `argPartition` + `takeAlong` (~1.0 ms -> ~0.06 ms).
// 3. `markovSelectGreedy`: single-threadgroup multi-position Markov edge scorer +
//    post-correction softcap + greedy path selector running all P positions in 1 dispatch.
// 4. `markovStepScores`: single-dispatch edge scorer + post-correction softcap for
//    sampled (temperature > 0) candidate selection.

import Foundation
import MLX
import MLXFast

enum DFlash2Kernels {
    #if canImport(Metal)
    private static let metalHeader = """
        #include <metal_stdlib>
        using namespace metal;
        """

    private static let groupedDynConvK2Kernel = MLXFast.metalKernel(
        name: "vmlx_dflash2_gdconv_k2",
        inputNames: ["hidden", "dynamic", "base"],
        outputNames: ["out"],
        source: """
            const uint idx = thread_position_in_grid.x;
            const uint B = uint(hidden_shape[0]);
            const uint L = uint(hidden_shape[1]);
            const uint H = uint(hidden_shape[2]);
            if (idx >= B * L * H) return;
            const uint c = idx % H;
            const uint t = (idx / H) % L;
            const uint b = idx / (H * L);
            const uint g = c / uint(G_SZ);
            const uint G = H / uint(G_SZ);

            const float v0 = float(hidden[idx]);
            const float k0 = float(base[c]) + float(dynamic[((b * L + t) * 2u + 0u) * G + g]);
            float acc = k0 * v0;
            if (t > 0u) {
                const float v1 = float(hidden[((b * L + (t - 1u)) * H) + c]);
                const float k1 = float(base[H + c]) + float(dynamic[((b * L + t) * 2u + 1u) * G + g]);
                acc += k1 * v1;
            }
            out[idx] = T(acc);
            """,
        header: metalHeader,
        ensureRowContiguous: true)

    private static let topKStage1Kernel = MLXFast.metalKernel(
        name: "vmlx_dflash2_topk_s1",
        inputNames: ["logits"],
        outputNames: ["blk_vals", "blk_idxs"],
        source: """
            const uint tid  = thread_position_in_threadgroup.x;
            const uint sg   = tid >> 5;
            const uint lane = tid & 31u;
            const uint blk  = threadgroup_position_in_grid.x;
            const uint pos  = threadgroup_position_in_grid.y;

            const uint V = uint(logits_shape[1]);
            const uint n_blk = threadgroups_per_grid.x;
            const uint chunk = (V + n_blk - 1u) / n_blk;
            const uint base = blk * chunk;
            const uint end  = min(base + chunk, V);

            constexpr uint MAX_E = 6u;
            float vals[MAX_E];
            uint  idxs[MAX_E];
            uint count = 0u;
            const device float* row = logits + size_t(pos) * size_t(V);
            for (uint j = base + tid; j < end && count < MAX_E; j += 256u) {
                vals[count] = row[j];
                idxs[count] = j;
                count++;
            }
            for (uint c = count; c < MAX_E; ++c) {
                vals[c] = -INFINITY;
                idxs[c] = 0xFFFFFFFFu;
            }

            threadgroup float sg_v[8];
            threadgroup uint  sg_i[8];
            threadgroup uint  win_idx_sh;

            const size_t out_off = (size_t(pos) * size_t(n_blk) + size_t(blk)) * uint(TOPK);
            for (uint k = 0u; k < uint(TOPK); ++k) {
                float bv = vals[0];
                uint  bi = idxs[0];
                uint  bc = 0u;
                #pragma clang loop unroll(full)
                for (uint c = 1u; c < MAX_E; ++c) {
                    if (vals[c] > bv || (vals[c] == bv && idxs[c] < bi)) {
                        bv = vals[c];
                        bi = idxs[c];
                        bc = c;
                    }
                }
                float sm = simd_max(bv);
                uint  si = simd_min((bv == sm) ? bi : 0xFFFFFFFFu);
                if (lane == 0u) {
                    sg_v[sg] = sm;
                    sg_i[sg] = si;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (tid == 0u) {
                    float gv = sg_v[0];
                    uint  gi = sg_i[0];
                    for (uint s = 1u; s < 8u; ++s) {
                        if (sg_v[s] > gv || (sg_v[s] == gv && sg_i[s] < gi)) {
                            gv = sg_v[s];
                            gi = sg_i[s];
                        }
                    }
                    win_idx_sh = gi;
                    blk_vals[out_off + k] = gv;
                    blk_idxs[out_off + k] = gi;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (bi == win_idx_sh && bi != 0xFFFFFFFFu) {
                    vals[bc] = -INFINITY;
                    idxs[bc] = 0xFFFFFFFFu;
                }
            }
            """,
        header: metalHeader,
        ensureRowContiguous: true)

    private static let topKStage2Kernel = MLXFast.metalKernel(
        name: "vmlx_dflash2_topk_s2",
        inputNames: ["blk_vals", "blk_idxs"],
        outputNames: ["out_vals", "out_idxs"],
        source: """
            const uint tid  = thread_position_in_threadgroup.x;
            const uint sg   = tid >> 5;
            const uint lane = tid & 31u;
            const uint pos  = threadgroup_position_in_grid.y;

            const uint M = uint(blk_vals_shape[1]);
            const device float* rv = blk_vals + size_t(pos) * size_t(M);
            const device uint*  ri = blk_idxs + size_t(pos) * size_t(M);

            float vals[TOPK];
            uint  idxs[TOPK];
            #pragma clang loop unroll(full)
            for (uint c = 0u; c < uint(TOPK); ++c) {
                uint j = tid + c * 256u;
                if (j < M) {
                    vals[c] = rv[j];
                    idxs[c] = ri[j];
                } else {
                    vals[c] = -INFINITY;
                    idxs[c] = 0xFFFFFFFFu;
                }
            }

            threadgroup float sg_v[8];
            threadgroup uint  sg_i[8];
            threadgroup uint  win_idx_sh;

            const size_t out_off = size_t(pos) * uint(TOPK);
            for (uint k = 0u; k < uint(TOPK); ++k) {
                float bv = vals[0];
                uint  bi = idxs[0];
                uint  bc = 0u;
                #pragma clang loop unroll(full)
                for (uint c = 1u; c < uint(TOPK); ++c) {
                    if (vals[c] > bv || (vals[c] == bv && idxs[c] < bi)) {
                        bv = vals[c];
                        bi = idxs[c];
                        bc = c;
                    }
                }
                float sm = simd_max(bv);
                uint  si = simd_min((bv == sm) ? bi : 0xFFFFFFFFu);
                if (lane == 0u) {
                    sg_v[sg] = sm;
                    sg_i[sg] = si;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (tid == 0u) {
                    float gv = sg_v[0];
                    uint  gi = sg_i[0];
                    for (uint s = 1u; s < 8u; ++s) {
                        if (sg_v[s] > gv || (sg_v[s] == gv && sg_i[s] < gi)) {
                            gv = sg_v[s];
                            gi = sg_i[s];
                        }
                    }
                    win_idx_sh = gi;
                    out_vals[out_off + k] = gv;
                    out_idxs[out_off + k] = gi;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (bi == win_idx_sh && bi != 0xFFFFFFFFu) {
                    vals[bc] = -INFINITY;
                    idxs[bc] = 0xFFFFFFFFu;
                }
            }
            """,
        header: metalHeader,
        ensureRowContiguous: true)

    private static let markovSelectGreedyKernel = MLXFast.metalKernel(
        name: "vmlx_dflash2_mkv_sel_greedy",
        inputNames: ["unary", "cands", "h_proj", "pred_cb", "succ_cb", "anchor", "cap_buf"],
        outputNames: ["path"],
        source: """
            const uint tid  = thread_position_in_threadgroup.x;
            const uint sg   = tid >> 5;
            const uint lane = tid & 31u;

            const uint P = uint(unary_shape[0]);
            const float cap = cap_buf[0];

            threadgroup float ph_sh[256];
            threadgroup float sc_sh[32];
            threadgroup uint  cur_pred_sh;

            if (tid == 0u) {
                cur_pred_sh = uint(anchor[0]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint p = 0u; p < P; ++p) {
                const uint pred_id = cur_pred_sh;
                const float pv = float(pred_cb[size_t(pred_id) * 256u + tid]);
                const float hv = float(h_proj[size_t(p) * 256u + tid]);
                ph_sh[tid] = pv * hv;
                threadgroup_barrier(mem_flags::mem_threadgroup);

                const uint d0 = lane * 8u;
                float4 ph_lo = float4(ph_sh[d0 + 0u], ph_sh[d0 + 1u], ph_sh[d0 + 2u], ph_sh[d0 + 3u]);
                float4 ph_hi = float4(ph_sh[d0 + 4u], ph_sh[d0 + 5u], ph_sh[d0 + 6u], ph_sh[d0 + 7u]);

                for (uint k = sg; k < uint(TOPK); k += 8u) {
                    const uint cid = uint(cands[size_t(p) * uint(TOPK) + k]);
                    const device CT* sp = succ_cb + size_t(cid) * 256u + d0;
                    float4 s_lo = float4(float(sp[0]), float(sp[1]), float(sp[2]), float(sp[3]));
                    float4 s_hi = float4(float(sp[4]), float(sp[5]), float(sp[6]), float(sp[7]));
                    float edge = simd_sum(dot(ph_lo, s_lo) + dot(ph_hi, s_hi));
                    if (lane == 0u) {
                        float sc = float(unary[size_t(p) * uint(TOPK) + k]) + edge;
                        if (HAS_CAP) {
                            sc = cap * metal::tanh(sc / cap);
                        }
                        sc_sh[k] = sc;
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                if (tid == 0u) {
                    float best_s = sc_sh[0];
                    uint  best_k = 0u;
                    for (uint k = 1u; k < uint(TOPK); ++k) {
                        if (sc_sh[k] > best_s) {
                            best_s = sc_sh[k];
                            best_k = k;
                        }
                    }
                    uint winner = uint(cands[size_t(p) * uint(TOPK) + best_k]);
                    path[p] = int(winner);
                    cur_pred_sh = winner;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            """,
        header: metalHeader,
        ensureRowContiguous: true)

    private static let markovStepScoresKernel = MLXFast.metalKernel(
        name: "vmlx_dflash2_mkv_step_scores",
        inputNames: ["unary_row", "cand_row", "h_row", "pred_cb", "succ_cb", "pred_arr", "cap_buf"],
        outputNames: ["scores"],
        source: """
            const uint tid  = thread_position_in_threadgroup.x;
            const uint sg   = tid >> 5;
            const uint lane = tid & 31u;
            const float cap = cap_buf[0];

            threadgroup float ph_sh[256];
            const uint pred_id = uint(pred_arr[0]);
            ph_sh[tid] = float(pred_cb[size_t(pred_id) * 256u + tid]) * float(h_row[tid]);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            const uint d0 = lane * 8u;
            float4 ph_lo = float4(ph_sh[d0 + 0u], ph_sh[d0 + 1u], ph_sh[d0 + 2u], ph_sh[d0 + 3u]);
            float4 ph_hi = float4(ph_sh[d0 + 4u], ph_sh[d0 + 5u], ph_sh[d0 + 6u], ph_sh[d0 + 7u]);

            for (uint k = sg; k < uint(TOPK); k += 8u) {
                const uint cid = uint(cand_row[k]);
                const device CT* sp = succ_cb + size_t(cid) * 256u + d0;
                float4 s_lo = float4(float(sp[0]), float(sp[1]), float(sp[2]), float(sp[3]));
                float4 s_hi = float4(float(sp[4]), float(sp[5]), float(sp[6]), float(sp[7]));
                float edge = simd_sum(dot(ph_lo, s_lo) + dot(ph_hi, s_hi));
                if (lane == 0u) {
                    float sc = float(unary_row[k]) + edge;
                    if (HAS_CAP) {
                        sc = cap * metal::tanh(sc / cap);
                    }
                    scores[k] = sc;
                }
            }
            """,
        header: metalHeader,
        ensureRowContiguous: true)
    #endif

    static func groupedDynamicConvolveK2(
        hidden: MLXArray, dynamic: MLXArray, base: MLXArray, groupSize: Int
    ) -> MLXArray? {
        #if canImport(Metal)
        guard Device.defaultDevice().deviceType == .gpu,
            hidden.ndim == 3, base.ndim == 2, base.dim(0) == 2,
            hidden.size > 0, groupSize > 0, hidden.dim(2) % groupSize == 0,
            [.bfloat16, .float16, .float32].contains(hidden.dtype)
        else { return nil }
        let bCast = base.dtype == hidden.dtype ? base : base.asType(hidden.dtype)
        let dCast = dynamic.dtype == hidden.dtype ? dynamic : dynamic.asType(hidden.dtype)
        let n = hidden.size
        return groupedDynConvK2Kernel(
            [hidden, dCast, bCast],
            template: [("T", hidden.dtype), ("G_SZ", groupSize)],
            grid: (((n + 255) / 256) * 256, 1, 1),
            threadGroup: (256, 1, 1),
            outputShapes: [hidden.shape],
            outputDTypes: [hidden.dtype])[0]
        #else
        return nil
        #endif
    }

    static func topK(logits: MLXArray, topK: Int) -> (unary: MLXArray, candidates: MLXArray)? {
        #if canImport(Metal)
        guard Device.defaultDevice().deviceType == .gpu,
            logits.ndim == 3, logits.dim(0) == 1,
            (1 ... 24).contains(topK)
        else { return nil }
        let p = logits.dim(1)
        let v = logits.dim(2)
        let nBlocks = 256
        guard p >= 1, v > topK, v <= nBlocks * 256 * 6 else { return nil }
        let l2 = logits.reshaped(p, v).asType(.float32)
        let s1 = topKStage1Kernel(
            [l2],
            template: [("TOPK", topK)],
            grid: (nBlocks * 256, p, 1),
            threadGroup: (256, 1, 1),
            outputShapes: [[p, nBlocks * topK], [p, nBlocks * topK]],
            outputDTypes: [.float32, .uint32])
        let s2 = topKStage2Kernel(
            [s1[0], s1[1]],
            template: [("TOPK", topK)],
            grid: (256, p, 1),
            threadGroup: (256, 1, 1),
            outputShapes: [[p, topK], [p, topK]],
            outputDTypes: [.float32, .uint32])
        return (s2[0].reshaped(1, p, topK), s2[1].reshaped(1, p, topK).asType(.int32))
        #else
        return nil
        #endif
    }

    static func markovSelectGreedy(
        unary: MLXArray,
        candidates: MLXArray,
        hiddenProj: MLXArray,
        predWeight: MLXArray,
        succWeight: MLXArray,
        anchorIds: MLXArray,
        cap: Float
    ) -> MLXArray? {
        #if canImport(Metal)
        guard Device.defaultDevice().deviceType == .gpu,
            unary.ndim == 3, unary.dim(0) == 1,
            hiddenProj.dim(-1) == 256,
            predWeight.dtype == succWeight.dtype,
            [.bfloat16, .float16, .float32].contains(predWeight.dtype),
            [.bfloat16, .float16, .float32].contains(hiddenProj.dtype)
        else { return nil }
        let p = unary.dim(1)
        let k = unary.dim(2)
        guard p >= 1, (1 ... 32).contains(k) else { return nil }
        let hasCap = cap > 0
        let capBuf = MLXArray([hasCap ? cap : Float(1.0)])
        let path = markovSelectGreedyKernel(
            [
                unary.reshaped(p, k).asType(.float32),
                candidates.reshaped(p, k).asType(.int32),
                hiddenProj.reshaped(p, 256),
                predWeight,
                succWeight,
                anchorIds.reshaped(1).asType(.int32),
                capBuf,
            ],
            template: [
                ("CT", predWeight.dtype),
                ("TOPK", k),
                ("HAS_CAP", hasCap),
            ],
            grid: (256, 1, 1),
            threadGroup: (256, 1, 1),
            outputShapes: [[p]],
            outputDTypes: [.int32])[0]
        return path.reshaped(1, p)
        #else
        return nil
        #endif
    }

    static func markovStepScores(
        unaryRow: MLXArray,
        candRow: MLXArray,
        hRow: MLXArray,
        predWeight: MLXArray,
        succWeight: MLXArray,
        predId: MLXArray,
        cap: Float
    ) -> MLXArray? {
        #if canImport(Metal)
        let k = unaryRow.dim(-1)
        guard Device.defaultDevice().deviceType == .gpu,
            unaryRow.dim(0) == 1,
            hRow.dim(-1) == 256,
            (1 ... 32).contains(k),
            predWeight.dtype == succWeight.dtype,
            [.bfloat16, .float16, .float32].contains(predWeight.dtype),
            [.bfloat16, .float16, .float32].contains(hRow.dtype)
        else { return nil }
        let hasCap = cap > 0
        let capBuf = MLXArray([hasCap ? cap : Float(1.0)])
        let sc = markovStepScoresKernel(
            [
                unaryRow.reshaped(k).asType(.float32),
                candRow.reshaped(k).asType(.int32),
                hRow.reshaped(256),
                predWeight,
                succWeight,
                predId.reshaped(1).asType(.int32),
                capBuf,
            ],
            template: [
                ("CT", predWeight.dtype),
                ("TOPK", k),
                ("HAS_CAP", hasCap),
            ],
            grid: (256, 1, 1),
            threadGroup: (256, 1, 1),
            outputShapes: [[k]],
            outputDTypes: [.float32])[0]
        return sc.reshaped(1, k)
        #else
        return nil
        #endif
    }
}
