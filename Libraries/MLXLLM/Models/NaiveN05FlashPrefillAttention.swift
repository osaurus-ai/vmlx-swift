// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXFast

/// Naive-N0.5 multi-row (prefill) attention without the O(chunk x context) reference. Ports of vMLX Python
/// `jangt/swa_blocked.py` and `jangt/dsa_sparse.py` (2026-10-09). The reference path repeats K/V over all 64 query
/// heads and materializes fp32 logits for every (query, key) pair, so its cost grows with chunk x context even on the
/// 40 sliding-window layers that see 128 keys per query (measured: attention 4.3 s of a 5k prompt at 512-token chunks,
/// 15.3 s at 2,048). Both rewrites are exact up to summation order; padded batches keep the reference.
enum NaiveN05FlashPrefillAttention {
    static let enabled = ProcessInfo.processInfo.environment["VMLX_NAIVE_PREFILL_ATTENTION"] != "0"
    /// Per-path switches (A/B and diagnosis): `VMLX_NAIVE_BLOCKED_SWA=0`, `VMLX_NAIVE_SPARSE_DSA=0`.
    static let blockedSlidingEnabled = ProcessInfo.processInfo.environment["VMLX_NAIVE_BLOCKED_SWA"] != "0"
    static let sparseEnabled = ProcessInfo.processInfo.environment["VMLX_NAIVE_SPARSE_DSA"] != "0"

    /// Sliding window W: query block b (W rows) only sees keys [bW - W, bW + W) of the front-padded key array, a
    /// 2W-key slab = [previous block | own block]. All blocks run as ONE batched SDPA (batch = block) with the same
    /// (W, 2W) window mask, the sink bias and GQA. `q` (1, H, L, D) are the last L positions of the keys;
    /// `k`/`v` (1, KV, Lk, .) in temporal order, Lk >= L, already value-scaled. Returns (1, H, L, Dv).
    static func blockedSliding(
        q: MLXArray, k: MLXArray, v: MLXArray, scale: Float, sinks: MLXArray?, window w: Int
    ) -> MLXArray {
        let h = q.dim(1), l = q.dim(2), d = q.dim(3), kvHeads = k.dim(1)
        let hist = min(k.dim(2) - l, w - 1)
        let k = k[0..., 0..., (k.dim(2) - (l + hist))..., 0...]
        let v = v[0..., 0..., (v.dim(2) - (l + hist))..., 0...]
        let blocks = (l + w - 1) / w, lp = blocks * w
        let kp = padded(k, widths: [0, 0, .init((w - hist, lp - l)), 0])
        let vp = padded(v, widths: [0, 0, .init((w - hist, lp - l)), 0])
        let qp = lp == l ? q : padded(q, widths: [0, 0, .init((0, lp - l)), 0])
        func slab(_ t: MLXArray) -> MLXArray {  // (1, KV, W + Lp, d) -> (blocks, KV, 2W, d)
            let a = t[0..., 0..., ..<lp, 0...].reshaped(kvHeads, blocks, w, -1)
            let b = t[0..., 0..., w ..< (w + lp), 0...].reshaped(kvHeads, blocks, w, -1)
            return concatenated([a, b], axis: 2).transposed(1, 0, 2, 3)
        }
        let ks = slab(kp), vs = slab(vp)
        let qs = qp.reshaped(h, blocks, w, d).transposed(1, 0, 2, 3)  // (blocks, H, W, D)
        let i = MLXArray(Int32(0) ..< Int32(w)).reshaped(w, 1)
        let t = MLXArray(Int32(0) ..< Int32(2 * w)).reshaped(1, 2 * w)
        let delta = (i + Int32(w)) - t
        let inWindow = (delta .>= Int32(0)) .&& (delta .< Int32(w))  // (W, 2W)
        let rows = MLXArray(Int32(0) ..< Int32(blocks)).reshaped(blocks, 1) * Int32(w)
            + MLXArray(Int32(0) ..< Int32(2 * w)).reshaped(1, 2 * w)
        let real = rows .>= Int32(w - hist)  // (blocks, 2W): front padding is not a key
        let mask = (inWindow.expandedDimensions(axis: 0) .&& real.expandedDimensions(axis: 1))
            .expandedDimensions(axis: 1)  // (blocks, 1, W, 2W)
        let o = MLXFast.scaledDotProductAttention(
            queries: qs, keys: ks, values: vs, scale: scale, mask: .array(mask), sinks: sinks)
        return o.transposed(1, 0, 2, 3).reshaped(1, h, lp, -1)[0..., 0..., ..<l, 0...]
    }

    /// Per-query sparse attention over the indexer's selection (`dsa_sparse.py`): one threadgroup per (query, KV
    /// head) covers the G query heads sharing that KV head; tiles of 32 selected keys, online softmax in fp32, keys
    /// past the query's causal position or < 0 (padding) masked. O(L x topK) instead of O(L x Lk). No sink (the DSA
    /// layers have none). q (1, H, L, DK) bf16 with RoPE; k (1, KV, Lk, DK); v (1, KV, Lk, DV) value-scaled;
    /// `selected` (L, topK) int32 key indices. Returns (1, H, L, DV) bf16.
    static func sparse(
        q: MLXArray, k: MLXArray, v: MLXArray, selected: MLXArray, scale: Float, queryOffset: Int
    ) -> MLXArray {
        let h = q.dim(1), l = q.dim(2), dk = q.dim(3), kvHeads = k.dim(1), keys = k.dim(2), dv = v.dim(3)
        let topK = selected.dim(-1)
        let g = h / kvHeads, groups = 4
        precondition(h % kvHeads == 0 && g % groups == 0 && dv % 32 == 0 && dk % 4 == 0 && topK % 32 == 0,
                     "Naive sparse attention geometry")
        let out = sparseKernel(
            [contiguous(q.asType(.bfloat16)), contiguous(k.asType(.bfloat16)), contiguous(v.asType(.bfloat16)),
             contiguous(selected.reshaped(l, topK).asType(.int32)),
             MLXArray([Int32(l), Int32(keys), Int32(queryOffset)]), MLXArray([scale])],
            template: [("KV", kvHeads), ("G", g), ("NSG", groups), ("HPS", g / groups), ("DK", dk), ("DV", dv),
                       ("TOPK", topK)],
            grid: (l * kvHeads * 32 * groups, 1, 1), threadGroup: (32 * groups, 1, 1),
            outputShapes: [[h, l, dv]], outputDTypes: [.bfloat16])[0]
        return out.expandedDimensions(axis: 0)
    }

    private static let sparseKernel = MLXFast.metalKernel(
        name: "vmlx_naive_dsa_sparse_attn",
        inputNames: ["q", "k", "v", "sel", "meta", "sc"], outputNames: ["o"],
        source: """
            const uint tg = threadgroup_position_in_grid.x;
            const uint qi = tg / KV, kvh = tg % KV;
            const uint sg = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;
            const uint tid = thread_position_in_threadgroup.x;
            const int L = meta[0], Lk = meta[1], qpos = meta[2] + int(qi);
            const float scale = sc[0];
            threadgroup float qs[G * DK];
            for (uint i = tid; i < G * DK; i += 32 * NSG) {
                uint hh = i / DK, d = i % DK;
                qs[i] = float(q[(size_t(kvh * G + hh) * L + qi) * DK + d]) * scale;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float m[HPS], l[HPS], pv[HPS], acc[HPS][DV / 32];
            for (int i = 0; i < HPS; ++i) { m[i] = -INFINITY; l[i] = 0.0f; for (int c = 0; c < DV / 32; ++c) acc[i][c] = 0.0f; }
            const device int* srow = sel + size_t(qi) * TOPK;
            for (int t0 = 0; t0 < TOPK; t0 += 32) {
                const int j = srow[t0 + lane];
                const bool valid = (j >= 0) && (j <= qpos) && (j < Lk);
                float s[HPS];
                for (int i = 0; i < HPS; ++i) s[i] = 0.0f;
                if (valid) {
                    const device bfloat16_t* kr = k + (size_t(kvh) * Lk + j) * DK;
                    for (int d = 0; d < DK; d += 4) {
                        float4 kv4 = float4(float(kr[d]), float(kr[d + 1]), float(kr[d + 2]), float(kr[d + 3]));
                        for (int i = 0; i < HPS; ++i) {
                            threadgroup const float* qh = qs + (sg * HPS + i) * DK + d;
                            s[i] += qh[0] * kv4.x + qh[1] * kv4.y + qh[2] * kv4.z + qh[3] * kv4.w;
                        }
                    }
                }
                for (int i = 0; i < HPS; ++i) {
                    const float si = valid ? s[i] : -INFINITY;
                    const float mn = metal::max(m[i], simd_max(si));
                    if (mn == -INFINITY) { pv[i] = 0.0f; continue; }
                    const float p = valid ? metal::exp(si - mn) : 0.0f;
                    const float corr = (m[i] == -INFINITY) ? 0.0f : metal::exp(m[i] - mn);
                    l[i] = l[i] * corr + simd_sum(p);
                    for (int c = 0; c < DV / 32; ++c) acc[i][c] *= corr;
                    m[i] = mn; pv[i] = p;
                }
                for (int jj = 0; jj < 32; ++jj) {
                    const int jidx = simd_shuffle(j, ushort(jj));
                    if (!((jidx >= 0) && (jidx <= qpos) && (jidx < Lk))) continue;
                    const device bfloat16_t* vr = v + (size_t(kvh) * Lk + jidx) * DV + lane * (DV / 32);
                    float vv[DV / 32];
                    for (int c = 0; c < DV / 32; ++c) vv[c] = float(vr[c]);
                    for (int i = 0; i < HPS; ++i) {
                        const float pp = simd_shuffle(pv[i], ushort(jj));
                        for (int c = 0; c < DV / 32; ++c) acc[i][c] += pp * vv[c];
                    }
                }
            }
            for (int i = 0; i < HPS; ++i) {
                const float inv = l[i] > 0.0f ? 1.0f / l[i] : 0.0f;
                device bfloat16_t* orow = o + (size_t(kvh * G + sg * HPS + i) * L + qi) * DV + lane * (DV / 32);
                for (int c = 0; c < DV / 32; ++c) orow[c] = bfloat16_t(acc[i][c] * inv);
            }
            """)
}
