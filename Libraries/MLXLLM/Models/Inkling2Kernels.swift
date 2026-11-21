// Copyright © 2026 Jinho Jang. All rights reserved.
//
// Fused Metal kernels for Inkling-2-Mini's small per-token blocks. Each replaces a chain of
// 8-20 tiny MLX ops whose launch latency, not bandwidth, dominated decode (measured on M5 Max at
// T=1: short convs ~6 ms/token, message gates ~4 ms, depth attention ~3.9 ms, router ~2 ms of a
// 30 ms step). All arithmetic is fp32 in the reference's order; outputs are rounded once, where
// the reference rounds. They work for any number of rows (prefill and decode).

import Foundation
import MLX

enum Inkling2Kernels {

    /// Typed store: Metal will not narrow float to bfloat implicitly.
    static let storeHeader = """
        template <typename T> inline void ink_store(device T* p, size_t i, float v) { p[i] = static_cast<T>(v); }
        template <typename T> inline float ink_round(const device T* like, float v) { return float(static_cast<T>(v)); }
        """

    // MARK: Short convolutions

    /// Static residual depthwise conv, `(sum_j xp[t+j] * w[j] + x_t) / sqrt(K + 1)`, K = 4.
    /// `x` [B, T, C]; `state` [B, 3, P] fp32, this conv's columns start at `meta[0]`;
    /// `w` [4, C] (tap 3 = current token). Outputs `out` [B, T, C] (x dtype),
    /// `new_state` [B, 3, C] fp32. One thread per (channel, batch row), walking T in order.
    static let shortConv = MLXFast.metalKernel(
        name: "inkling2_short_conv",
        inputNames: ["x", "state", "w", "meta"],
        outputNames: ["out", "new_state"],
        source: """
            uint c = thread_position_in_grid.x;
            uint b = thread_position_in_grid.y;
            int C = x_shape[2];
            int T = x_shape[1];
            int P = state_shape[2];
            if (c >= (uint)C) return;
            int off = meta[0];
            const device float* st = state + (size_t)b * 3 * P + off + c;
            float s0 = st[0], s1 = st[P], s2 = st[2 * P];
            float w0 = float(w[c]), w1 = float(w[C + c]), w2 = float(w[2 * C + c]), w3 = float(w[3 * C + c]);
            const float inv = 2.2360679774997896f;  // sqrt(5)
            for (int t = 0; t < T; ++t) {
                size_t i = ((size_t)b * T + t) * C + c;
                float xt = float(x[i]);
                float y = s0 * w0;
                y = y + s1 * w1;
                y = y + s2 * w2;
                y = y + xt * w3;
                ink_store(out, i, (y + xt) / inv);
                s0 = s1; s1 = s2; s2 = xt;
            }
            device float* ns = new_state + (size_t)b * 3 * C + c;
            ns[0] = s0; ns[C] = s1; ns[2 * C] = s2;
            """,
        header: storeHeader)

    /// Static conv fused with the layer's residual arithmetic (post-attention / post-MLP).
    /// meta = [offset, mode]. The conv output is first rounded to `x`'s dtype (the reference's
    /// `.astype(x.dtype)`), then:
    ///   mode 0: out = fp32(conv)                       (block boundary: residual = a)
    ///   mode 1: out = fp32(res) + fp32(conv)            (residual = h + a; `res` = h)
    ///   mode 2: out = dtype(res + fp32(conv))           (layer output; `res` = residual fp32)
    /// Output dtype is given by the caller (fp32 for modes 0/1, model dtype for mode 2).
    static let residualConv = MLXFast.metalKernel(
        name: "inkling2_residual_conv",
        inputNames: ["x", "state", "w", "res", "meta"],
        outputNames: ["out", "new_state"],
        source: """
            uint c = thread_position_in_grid.x;
            uint b = thread_position_in_grid.y;
            int C = x_shape[2];
            int T = x_shape[1];
            int P = state_shape[2];
            if (c >= (uint)C) return;
            int off = meta[0];
            int mode = meta[1];
            const device float* st = state + (size_t)b * 3 * P + off + c;
            float s0 = st[0], s1 = st[P], s2 = st[2 * P];
            float w0 = float(w[c]), w1 = float(w[C + c]), w2 = float(w[2 * C + c]), w3 = float(w[3 * C + c]);
            const float inv = 2.2360679774997896f;  // sqrt(5)
            for (int t = 0; t < T; ++t) {
                size_t i = ((size_t)b * T + t) * C + c;
                float xt = float(x[i]);
                float y = s0 * w0;
                y = y + s1 * w1;
                y = y + s2 * w2;
                y = y + xt * w3;
                float a = ink_round(x, (y + xt) / inv);
                float o = mode == 0 ? a : float(res[i]) + a;
                ink_store(out, i, o);
                s0 = s1; s1 = s2; s2 = xt;
            }
            device float* ns = new_state + (size_t)b * 3 * C + c;
            ns[0] = s0; ns[C] = s1; ns[2 * C] = s2;
            """,
        header: storeHeader)

    /// `attn * sigmoid(gate)` in fp32, rounded to `attn`'s dtype.
    static let sigmoidGate = MLXFast.metalKernel(
        name: "inkling2_sigmoid_gate",
        inputNames: ["attn", "gate"],
        outputNames: ["out"],
        source: """
            uint i = thread_position_in_grid.x;
            float g = float(gate[i]);
            ink_store(out, i, float(attn[i]) * (1.f / (1.f + exp(-g))));
            """,
        header: storeHeader)

    /// The attention input clips, one simdgroup per head vector:
    ///   q: (q * qscale) L2-clipped to `thr[0]`; r: (r * rscale) clipped to `thr[1]`;
    ///   k: clipped to `thr[0]`; v: unweighted RMS norm (eps `thr[2]`). fp32 math, outputs in the
    /// input dtype. q [R, NQ*D], r [R, NQ*DR], k / v [R, NKV*D]; head layout per row:
    /// NQ q heads, NQ r heads, NKV k heads, NKV v heads.
    static let attnClips = MLXFast.metalKernel(
        name: "inkling2_attn_clips",
        inputNames: ["q", "r", "k", "v", "qscale", "rscale", "thr"],
        outputNames: ["qo", "ro", "ko", "vo"],
        source: """
            uint head = threadgroup_position_in_grid.x;
            uint row = threadgroup_position_in_grid.y;
            uint lane = thread_index_in_simdgroup;
            int NQ = qscale_shape[0];
            int D = qscale_shape[1];
            int DR = rscale_shape[1];
            int NKV = k_shape[1] / D;
            float vals[4];
            int n = 0;
            float sumsq = 0.f;
            int kind; int h; int dim;
            if (head < (uint)NQ) { kind = 0; h = head; dim = D; }
            else if (head < (uint)(2 * NQ)) { kind = 1; h = head - NQ; dim = DR; }
            else if (head < (uint)(2 * NQ + NKV)) { kind = 2; h = head - 2 * NQ; dim = D; }
            else { kind = 3; h = head - 2 * NQ - NKV; dim = D; }
            for (int j = lane; j < dim; j += 32) {
                float x;
                if (kind == 0) x = float(q[(size_t)row * NQ * D + h * D + j]) * qscale[h * D + j];
                else if (kind == 1) x = float(r[(size_t)row * NQ * DR + h * DR + j]) * rscale[h * DR + j];
                else if (kind == 2) x = float(k[(size_t)row * NKV * D + h * D + j]);
                else x = float(v[(size_t)row * NKV * D + h * D + j]);
                vals[n++] = x;
                sumsq += x * x;
            }
            sumsq = simd_sum(sumsq);
            float f;
            if (kind == 3) f = rsqrt(sumsq / float(dim) + thr[2]);
            else f = min((kind == 1 ? thr[1] : thr[0]) / (sqrt(sumsq) + 1e-6f), 1.f);
            n = 0;
            for (int j = lane; j < dim; j += 32) {
                float y = vals[n++] * f;
                if (kind == 0) ink_store(qo, (size_t)row * NQ * D + h * D + j, y);
                else if (kind == 1) ink_store(ro, (size_t)row * NQ * DR + h * DR + j, y);
                else if (kind == 2) ink_store(ko, (size_t)row * NKV * D + h * D + j, y);
                else ink_store(vo, (size_t)row * NKV * D + h * D + j, y);
            }
            """,
        header: storeHeader)

    /// Full-layer dynamic conv: taps `static[lag][c] + dyn[t, head(c), lag]`, lag 0 = current token,
    /// `(sum + x_t) / sqrt(2K + 1)`, K = 4. `dyn` [B, T, NH*4] (any float dtype).
    static let dynamicConv = MLXFast.metalKernel(
        name: "inkling2_dynamic_conv",
        inputNames: ["x", "state", "w", "dyn", "meta"],
        outputNames: ["out", "new_state"],
        source: """
            uint c = thread_position_in_grid.x;
            uint b = thread_position_in_grid.y;
            int C = x_shape[2];
            int T = x_shape[1];
            int P = state_shape[2];
            int NH4 = dyn_shape[2];
            if (c >= (uint)C) return;
            int off = meta[0];
            int head = c / (C / (NH4 / 4));
            const device float* st = state + (size_t)b * 3 * P + off + c;
            float s0 = st[0], s1 = st[P], s2 = st[2 * P];
            float w0 = float(w[c]), w1 = float(w[C + c]), w2 = float(w[2 * C + c]), w3 = float(w[3 * C + c]);
            const float inv = 3.0f;  // sqrt(9)
            for (int t = 0; t < T; ++t) {
                size_t i = ((size_t)b * T + t) * C + c;
                const device auto* d = dyn + ((size_t)b * T + t) * NH4 + head * 4;
                float xt = float(x[i]);
                float y = xt * (w0 + float(d[0]));
                y = y + s2 * (w1 + float(d[1]));
                y = y + s1 * (w2 + float(d[2]));
                y = y + s0 * (w3 + float(d[3]));
                ink_store(out, i, (y + xt) / inv);
                s0 = s1; s1 = s2; s2 = xt;
            }
            device float* ns = new_state + (size_t)b * 3 * C + c;
            ns[0] = s0; ns[C] = s1; ns[2 * C] = s2;
            """,
        header: storeHeader)

    // MARK: Depth attention (block attention residual / loop messages)

    /// `stack` [R, S, H] fp32 (completed states), `partial` [R, H] (any float dtype; may be absent
    /// when `meta[1] == 0`), `query` [NQ, H] fp32. Sources = S stack rows (+ partial). Each source is
    /// unweighted-RMS-normalized (eps 1e-6); scores = normed . q / sqrt(H); softmax over sources;
    /// out [R, NQ, H] = sum_s p * normed_s (fp32). With `meta[3] == 1` each query's output row is
    /// then passed through a weighted RMSNorm (`nw` [NQ, H] fp32, eps `ne[0]`), i.e. the layer's
    /// input_layernorm / post_attention_layernorm fused in, written in `out`'s dtype.
    /// One 256-thread group per row. meta = [S, hasPartial, NQ, doNorm].
    static let depthAttention = MLXFast.metalKernel(
        name: "inkling2_depth_attention",
        inputNames: ["stack", "partial", "query", "nw", "ne", "meta"],
        outputNames: ["out"],
        source: """
            constexpr int MAXS = 12;
            constexpr int MAXQ = 4;
            uint r = threadgroup_position_in_grid.x;
            uint tid = thread_position_in_threadgroup.x;
            uint lane = thread_index_in_simdgroup;
            uint sg = simdgroup_index_in_threadgroup;
            int S = meta[0];
            int hasP = meta[1];
            int NQ = meta[2];
            int doNorm = meta[3];
            int H = query_shape[1];
            int NS = S + hasP;
            float ss[MAXS];
            float dq[MAXS * MAXQ];
            for (int s = 0; s < NS; ++s) { ss[s] = 0.f; for (int q = 0; q < NQ; ++q) dq[s * MAXQ + q] = 0.f; }
            const device float* base = stack + (size_t)r * S * H;
            for (int j = tid; j < H; j += 256) {
                float qv[MAXQ];
                for (int q = 0; q < NQ; ++q) qv[q] = query[q * H + j];
                for (int s = 0; s < NS; ++s) {
                    float v = s < S ? base[(size_t)s * H + j] : float(partial[(size_t)r * H + j]);
                    ss[s] += v * v;
                    for (int q = 0; q < NQ; ++q) dq[s * MAXQ + q] += v * qv[q];
                }
            }
            threadgroup float red[8][MAXS * (MAXQ + 1)];
            for (int s = 0; s < NS; ++s) {
                float a = simd_sum(ss[s]);
                if (lane == 0) red[sg][s * (MAXQ + 1)] = a;
                for (int q = 0; q < NQ; ++q) {
                    float d = simd_sum(dq[s * MAXQ + q]);
                    if (lane == 0) red[sg][s * (MAXQ + 1) + 1 + q] = d;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            threadgroup float inv[MAXS];
            threadgroup float p[MAXS * MAXQ];
            if (tid == 0) {
                float scale = rsqrt(float(H));
                for (int s = 0; s < NS; ++s) {
                    float a = 0.f;
                    for (int g = 0; g < 8; ++g) a += red[g][s * (MAXQ + 1)];
                    inv[s] = rsqrt(a / float(H) + 1e-6f);
                }
                for (int q = 0; q < NQ; ++q) {
                    float m = -INFINITY;
                    for (int s = 0; s < NS; ++s) {
                        float d = 0.f;
                        for (int g = 0; g < 8; ++g) d += red[g][s * (MAXQ + 1) + 1 + q];
                        float sc = d * inv[s] * scale;
                        p[s * MAXQ + q] = sc;
                        m = max(m, sc);
                    }
                    float z = 0.f;
                    for (int s = 0; s < NS; ++s) { float e = exp(p[s * MAXQ + q] - m); p[s * MAXQ + q] = e; z += e; }
                    for (int s = 0; s < NS; ++s) p[s * MAXQ + q] = p[s * MAXQ + q] / z;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            threadgroup float nrm[MAXQ];
            if (doNorm) {
                float sq[MAXQ];
                for (int q = 0; q < NQ; ++q) sq[q] = 0.f;
                for (int j = tid; j < H; j += 256) {
                    float acc[MAXQ];
                    for (int q = 0; q < NQ; ++q) acc[q] = 0.f;
                    for (int s = 0; s < NS; ++s) {
                        float v = (s < S ? base[(size_t)s * H + j] : float(partial[(size_t)r * H + j])) * inv[s];
                        for (int q = 0; q < NQ; ++q) acc[q] += p[s * MAXQ + q] * v;
                    }
                    for (int q = 0; q < NQ; ++q) sq[q] += acc[q] * acc[q];
                }
                threadgroup float red2[8][MAXQ];
                for (int q = 0; q < NQ; ++q) { float a = simd_sum(sq[q]); if (lane == 0) red2[sg][q] = a; }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (tid < (uint)NQ) {
                    float a = 0.f;
                    for (int g = 0; g < 8; ++g) a += red2[g][tid];
                    nrm[tid] = rsqrt(a / float(H) + ne[0]);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            for (int j = tid; j < H; j += 256) {
                float acc[MAXQ];
                for (int q = 0; q < NQ; ++q) acc[q] = 0.f;
                for (int s = 0; s < NS; ++s) {
                    float v = (s < S ? base[(size_t)s * H + j] : float(partial[(size_t)r * H + j])) * inv[s];
                    for (int q = 0; q < NQ; ++q) acc[q] += p[s * MAXQ + q] * v;
                }
                for (int q = 0; q < NQ; ++q) {
                    float o = doNorm ? (acc[q] * nrm[q]) * nw[q * H + j] : acc[q];
                    ink_store(out, ((size_t)r * NQ + q) * H + j, o);
                }
            }
            """,
        header: storeHeader)

    /// Decode relative bias: `bias[h, j] = bf16(r[h] . proj[:, d])` at distance `d = S-1-j`, zero
    /// for `d >= E`. `r` [NQ, DR], `proj` [DR, E]; out [NQ, S] in `r`'s dtype.
    static let decodeBias = MLXFast.metalKernel(
        name: "inkling2_decode_bias",
        inputNames: ["r", "proj", "meta"],
        outputNames: ["out"],
        source: """
            uint j = thread_position_in_grid.x;
            uint h = thread_position_in_grid.y;
            int S = meta[0];
            int DR = proj_shape[0];
            int E = proj_shape[1];
            if (j >= (uint)S) return;
            int d = S - 1 - (int)j;
            float v = 0.f;
            if (d < E) {
                for (int k = 0; k < DR; ++k) v += float(r[h * DR + k]) * float(proj[k * E + d]);
            }
            ink_store(out, (size_t)h * S + j, v);
            """,
        header: storeHeader)

    // MARK: Loop message gate

    /// `[rms(h), rms(m)]` (fp32 statistics, eps from meta as float bits) cast to the down-projection
    /// dtype. `h`, `m` [R, H]; out [R, 2H]. One 256-thread group per row.
    static let gateInput = MLXFast.metalKernel(
        name: "inkling2_gate_input",
        inputNames: ["h", "m", "eps"],
        outputNames: ["out"],
        source: """
            uint r = threadgroup_position_in_grid.x;
            uint tid = thread_position_in_threadgroup.x;
            uint lane = thread_index_in_simdgroup;
            uint sg = simdgroup_index_in_threadgroup;
            int H = h_shape[1];
            float a = 0.f, b = 0.f;
            for (int j = tid; j < H; j += 256) {
                float x = float(h[(size_t)r * H + j]); a += x * x;
                float y = float(m[(size_t)r * H + j]); b += y * y;
            }
            threadgroup float ra[8], rb[8];
            a = simd_sum(a); b = simd_sum(b);
            if (lane == 0) { ra[sg] = a; rb[sg] = b; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float A = 0.f, Bv = 0.f;
            for (int g = 0; g < 8; ++g) { A += ra[g]; Bv += rb[g]; }
            float ia = rsqrt(A / float(H) + eps[0]);
            float ib = rsqrt(Bv / float(H) + eps[0]);
            for (int j = tid; j < H; j += 256) {
                ink_store(out, (size_t)r * 2 * H + j, float(h[(size_t)r * H + j]) * ia);
                ink_store(out, (size_t)r * 2 * H + H + j, float(m[(size_t)r * H + j]) * ib);
            }
            """,
        header: storeHeader)

    /// `(1 - tanh(fg[:, j])) * h[j] + tanh(fg[:, H + j]) * m[j]`, fg = g @ [Wf; Wg]^T in fp32
    /// (computed by one MLX GEMV). out fp32 [R, H].
    static let gateCombine = MLXFast.metalKernel(
        name: "inkling2_gate_combine",
        inputNames: ["fg", "h", "m"],
        outputNames: ["out"],
        source: """
            uint j = thread_position_in_grid.x;
            uint r = thread_position_in_grid.y;
            int H = h_shape[1];
            if (j >= (uint)H) return;
            float f = tanh(fg[(size_t)r * 2 * H + j]);
            float a = tanh(fg[(size_t)r * 2 * H + H + j]);
            float hv = float(h[(size_t)r * H + j]);
            float mv = float(m[(size_t)r * H + j]);
            out[(size_t)r * H + j] = (1.f - f) * hv + a * mv;
            """)

    // MARK: Router

    /// sigmoid + selection bias + top-K + normalized weights * route_scale on precomputed fp32
    /// router logits `[R, E]` (the GEMV runs in MLX across all cores). meta = [K].
    /// Outputs `inds` [R, K] uint32, `weights` [R, K] fp32. One 128..256-thread group per row.
    static let routerSelect = MLXFast.metalKernel(
        name: "inkling2_router_select",
        inputNames: ["logits", "bias", "meta", "scale"],
        outputNames: ["inds", "weights"],
        source: """
            uint r = threadgroup_position_in_grid.x;
            uint tid = thread_position_in_threadgroup.x;
            uint ntg = threads_per_threadgroup.x;
            int E = logits_shape[1];
            int K = meta[0];
            threadgroup float sc[256];
            threadgroup float sel[256];
            for (int e = tid; e < E; e += ntg) {
                float v = 1.f / (1.f + exp(-logits[(size_t)r * E + e]));
                sc[e] = v;
                sel[e] = v + bias[e];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            // Parallel top-K: rank = number of experts that beat this one (ties -> lower index
            // first); ranks < K are the selection, written in descending order.
            threadgroup uint ids[16];
            for (int e = tid; e < E; e += ntg) {
                float v = sel[e];
                int rank = 0;
                for (int o = 0; o < E; ++o) {
                    float u = sel[o];
                    rank += (u > v || (u == v && o < e)) ? 1 : 0;
                }
                if (rank < K) ids[rank] = e;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid < (uint)K) {
                float total = 0.f;
                for (int k = 0; k < K; ++k) total += sc[ids[k]];
                inds[(size_t)r * K + tid] = ids[tid];
                weights[(size_t)r * K + tid] = sc[ids[tid]] / (total + 1e-20f) * scale[0];
            }
            """)

    // MARK: MoE elementwise

    /// De-interleave fused gate/up rows, apply the per-expert factor, and
    /// `softcap(g, gc) * sigmoid(g) * softcap(u, uc)` rounded to the model dtype (kept as fp32 so
    /// the down projection sees exactly the reference's bf16 values).
    /// `gu` [N, 2I] fp32, `idx` [N] expert ids, `es` [E] fp32 factor, caps = [gc, uc].
    static let moeActivation = MLXFast.metalKernel(
        name: "inkling2_moe_activation",
        inputNames: ["gu", "idx", "es", "caps"],
        outputNames: ["out"],
        source: """
            uint i = thread_position_in_grid.x;
            uint n = thread_position_in_grid.y;
            int I = gu_shape[1] / 2;
            if (i >= (uint)I) return;
            float s = es[idx[n]];
            float g = gu[(size_t)n * 2 * I + 2 * i] * s;
            float u = gu[(size_t)n * 2 * I + 2 * i + 1] * s;
            float gc = caps[0], uc = caps[1];
            float a = (tanh(g / gc) * gc) * (1.f / (1.f + exp(-g))) * (tanh(u / uc) * uc);
            out[(size_t)n * I + i] = float(bfloat16_t(a));
            """)

    /// `sum_k down[n, k, :] * es[idx[n, k]] * w[n, k]` in fp32, written in the model dtype.
    /// `down` [N, K, H] fp32, `idx` [N, K], `es` [E], `w` [N, K]. One thread per (h, n).
    static let moeCombine = MLXFast.metalKernel(
        name: "inkling2_moe_combine",
        inputNames: ["down", "idx", "es", "w"],
        outputNames: ["out"],
        source: """
            uint j = thread_position_in_grid.x;
            uint n = thread_position_in_grid.y;
            int K = down_shape[1];
            int H = down_shape[2];
            if (j >= (uint)H) return;
            float acc = 0.f;
            for (int k = 0; k < K; ++k) {
                size_t nk = (size_t)n * K + k;
                acc += (down[nk * H + j] * es[idx[nk]]) * w[nk];
            }
            ink_store(out, (size_t)n * H + j, acc);
            """,
        header: storeHeader)

    // MARK: launch helpers

    static func conv(
        _ x: MLXArray, packed: MLXArray, offset: Int, weight: MLXArray, dyn: MLXArray? = nil
    ) -> (MLXArray, MLXArray) {
        let (B, T, C) = (x.dim(0), x.dim(1), x.dim(2))
        let meta = MLXArray([Int32(offset)])
        let tg = min(256, C)
        let outs: [MLXArray]
        if let dyn {
            outs = dynamicConv(
                [x, packed, weight, dyn, meta], grid: (C, B, 1), threadGroup: (tg, 1, 1),
                outputShapes: [[B, T, C], [B, 3, C]], outputDTypes: [x.dtype, .float32])
        } else {
            outs = shortConv(
                [x, packed, weight, meta], grid: (C, B, 1), threadGroup: (tg, 1, 1),
                outputShapes: [[B, T, C], [B, 3, C]], outputDTypes: [x.dtype, .float32])
        }
        return (outs[0], outs[1])
    }

    /// `stack` [R, S, H] fp32 or nil, `partial` [R, H] or nil, `query` [NQ, H] fp32 → [R, NQ, H].
    /// With `normWeight` ([NQ, H] or [H] fp32) the weighted RMSNorm is fused and the result is
    /// written in `outDType`; otherwise fp32.
    static func depth(
        stack: MLXArray?, partial: MLXArray?, query: MLXArray, rows: Int,
        normWeight: MLXArray? = nil, normEps: MLXArray? = nil, outDType: DType = .float32
    ) -> MLXArray {
        let H = query.dim(-1)
        let q = query.ndim == 1 ? query.reshaped([1, H]) : query
        let S = stack?.dim(1) ?? 0
        let meta = MLXArray([
            Int32(S), Int32(partial == nil ? 0 : 1), Int32(q.dim(0)),
            Int32(normWeight == nil ? 0 : 1),
        ])
        let st = stack ?? MLXArray.zeros([1, 1, H], dtype: .float32)
        let pa = partial ?? MLXArray.zeros([1, H], dtype: .float32)
        let nw = normWeight ?? q
        let ne = normEps ?? MLXArray([Float(1e-6)])
        return depthAttention(
            [st, pa, q, nw, ne, meta], grid: (rows * 256, 1, 1), threadGroup: (256, 1, 1),
            outputShapes: [[rows, q.dim(0), H]],
            outputDTypes: [normWeight == nil ? .float32 : outDType])[0]
    }
}
