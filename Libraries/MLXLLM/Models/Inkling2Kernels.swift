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

    // MARK: Fused zero-mask MMA prefill & Split-K FlashDecoding (AR + DFlash 2 verify)

    static let mmaHeader = "#include <metal_simdgroup_matrix>\n" + storeHeader

    /// Hardware-MMA (`simdgroup_bfloat8x8`) zero-mask FlashAttention prefill for Inkling-2-Mini (D=128, DR=16).
    /// Each 128-thread TG (4 simdgroups) owns 1 Q head x BQ=32 query rows, streaming BK=32 K/V tiles.
    /// For keys strictly beyond `relExtent` (`kPos <= qPosMin - E`), runs a zero-branch, zero-bias
    /// hardware MMA loop (99% of keys at 100K+ context).
    static let prefillAttnMMAKernel = MLXFast.metalKernel(
        name: "inkling2_prefill_attn_mma",
        inputNames: ["q", "k", "v", "r", "projT", "sinks", "meta", "scaleArr"],
        outputNames: ["out"],
        source: """
            constexpr int D = 128;
            constexpr int DR = 16;
            constexpr int BQ = 32;
            constexpr int BK = 32;
            uint h = threadgroup_position_in_grid.x;
            uint tBlock = threadgroup_position_in_grid.y * BQ;
            uint b = threadgroup_position_in_grid.z;
            uint tid = thread_position_in_threadgroup.x;
            uint sg = simdgroup_index_in_threadgroup;
            int NQ = q_shape[1];
            int T = q_shape[2];
            int NKV = k_shape[1];
            int S = k_shape[2];
            int seen = meta[0];
            int kStart = meta[1];
            int W = meta[2];
            int E = meta[3];
            float sc = scaleArr[0];
            uint hkv = h / GQA;

            threadgroup bfloat16_t qTile[BQ][D];
            threadgroup bfloat16_t kvTile[BK][D];
            threadgroup float sBuf[BQ][BK + 2];
            threadgroup bfloat16_t pBuf[BQ][BK];
            threadgroup bfloat16_t rTile[BQ][DR];
            threadgroup float diagMat[4][8][8];
            threadgroup float mState[BQ];
            threadgroup float lState[BQ];

            size_t qHeadBase = ((size_t)b * NQ + h) * T * D;
            for (int idx = int(tid); idx < BQ * (D / 4); idx += 128) {
                int qt = idx / (D / 4);
                int d4 = (idx % (D / 4)) * 4;
                int tIdx = int(tBlock) + qt;
                vec<bfloat16_t, 4> val = (tIdx < T)
                    ? *(const device vec<bfloat16_t, 4>*)(q + qHeadBase + (size_t)tIdx * D + d4)
                    : vec<bfloat16_t, 4>(bfloat16_t(0.f));
                *(threadgroup vec<bfloat16_t, 4>*)(&qTile[qt][d4]) = val;
            }
            for (int idx = int(tid); idx < BQ * (DR / 4); idx += 128) {
                int qt = idx / (DR / 4);
                int d4 = (idx % (DR / 4)) * 4;
                int tIdx = int(tBlock) + qt;
                vec<bfloat16_t, 4> val = (tIdx < T)
                    ? *(const device vec<bfloat16_t, 4>*)(r + (((size_t)b * T + tIdx) * NQ + h) * DR + d4)
                    : vec<bfloat16_t, 4>(bfloat16_t(0.f));
                *(threadgroup vec<bfloat16_t, 4>*)(&rTile[qt][d4]) = val;
            }
            if (tid < uint(BQ)) {
                int tIdx = int(tBlock) + int(tid);
                mState[tid] = (tIdx < T) ? float(sinks[((size_t)b * T + tIdx) * NQ + h]) : -INFINITY;
                lState[tid] = 1.0f;
            }
            for (uint i = tid; i < 256u; i += 128u) {
                diagMat[i / 64u][(i % 64u) / 8u][i % 8u] = 0.0f;
            }

            simdgroup_float8x8 accO[16];
            #pragma clang loop unroll(full)
            for (int j = 0; j < 16; ++j) accO[j] = simdgroup_float8x8(0.0f);

            int tLast = min(T - 1, int(tBlock) + BQ - 1);
            int qPosMin = seen + int(tBlock);
            int qPosMax = seen + tLast;
            int jStart = (W > 0) ? max(0, qPosMin - W + 1 - kStart) : 0;
            int jEnd = min(S, qPosMax - kStart + 1);
            size_t kvHeadBase = ((size_t)b * NKV + hkv) * S * D;
            uint sg_m = sg * 8u;

            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (int jBase = jStart; jBase < jEnd; jBase += BK) {
                int tileLen = min(BK, jEnd - jBase);
                for (int idx = int(tid); idx < BK * (D / 4); idx += 128) {
                    int jj = idx / (D / 4);
                    int d4 = (idx % (D / 4)) * 4;
                    vec<bfloat16_t, 4> val = (jj < tileLen)
                        ? *(const device vec<bfloat16_t, 4>*)(k + kvHeadBase + (size_t)(jBase + jj) * D + d4)
                        : vec<bfloat16_t, 4>(bfloat16_t(0.f));
                    *(threadgroup vec<bfloat16_t, 4>*)(&kvTile[jj][d4]) = val;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                simdgroup_float8x8 accS0(0.0f), accS1(0.0f), accS2(0.0f), accS3(0.0f);
                for (int ks = 0; ks < D; ks += 8) {
                    simdgroup_bfloat8x8 qa, k0, k1, k2, k3;
                    simdgroup_load(qa, &qTile[sg_m][ks], D);
                    simdgroup_load(k0, &kvTile[0][ks], D, ulong2(0, 0), true);
                    simdgroup_load(k1, &kvTile[8][ks], D, ulong2(0, 0), true);
                    simdgroup_load(k2, &kvTile[16][ks], D, ulong2(0, 0), true);
                    simdgroup_load(k3, &kvTile[24][ks], D, ulong2(0, 0), true);
                    simdgroup_multiply_accumulate(accS0, qa, k0, accS0);
                    simdgroup_multiply_accumulate(accS1, qa, k1, accS1);
                    simdgroup_multiply_accumulate(accS2, qa, k2, accS2);
                    simdgroup_multiply_accumulate(accS3, qa, k3, accS3);
                }
                simdgroup_store(accS0, &sBuf[sg_m][0], BK + 2);
                simdgroup_store(accS1, &sBuf[sg_m][8], BK + 2);
                simdgroup_store(accS2, &sBuf[sg_m][16], BK + 2);
                simdgroup_store(accS3, &sBuf[sg_m][24], BK + 2);
                threadgroup_barrier(mem_flags::mem_threadgroup);

                bool pureUnmasked = (tileLen == BK) && (W == 0) && (kStart + jBase + BK - 1 <= qPosMin - E);
                if (!pureUnmasked) {
                    int tr = int(tid / 4u);
                    int kk0 = int(tid % 4u) * 8;
                    int tAbs = int(tBlock) + tr;
                    int qPos = seen + tAbs;
                    for (int kk = kk0; kk < kk0 + 8; ++kk) {
                        if (kk >= tileLen || tAbs >= T) {
                            sBuf[tr][kk] = -INFINITY;
                        } else {
                            int d = qPos - (kStart + jBase + kk);
                            if (d < 0 || (W > 0 && d >= W)) {
                                sBuf[tr][kk] = -INFINITY;
                            } else {
                                float s = sBuf[tr][kk] * sc;
                                if (d < E) {
                                    const device bfloat16_t* pRow = projT + (size_t)d * DR;
                                    float dot = 0.0f;
                                    #pragma clang loop unroll(full)
                                    for (int dr = 0; dr < DR; ++dr) {
                                        dot += float(rTile[tr][dr]) * float(pRow[dr]);
                                    }
                                    s += float(bfloat16_t(dot));
                                }
                                sBuf[tr][kk] = s;
                            }
                        }
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }

                if (tid < uint(BQ)) {
                    int tAbs = int(tBlock) + int(tid);
                    if (tAbs < T) {
                        float mOld = mState[tid];
                        float mNew = mOld;
                        float rowS[BK];
                        if (pureUnmasked) {
                            for (int kk = 0; kk < BK; ++kk) {
                                float s = sBuf[tid][kk] * sc;
                                rowS[kk] = s;
                                mNew = max(mNew, s);
                            }
                        } else {
                            for (int kk = 0; kk < BK; ++kk) {
                                float s = sBuf[tid][kk];
                                rowS[kk] = s;
                                mNew = max(mNew, s);
                            }
                        }
                        float alpha = exp(mOld - mNew);
                        float lNew = lState[tid] * alpha;
                        for (int kk = 0; kk < BK; ++kk) {
                            float p = (rowS[kk] > -1e30f) ? exp(rowS[kk] - mNew) : 0.0f;
                            lNew += p;
                            pBuf[tid][kk] = bfloat16_t(p);
                        }
                        mState[tid] = mNew;
                        lState[tid] = lNew;
                        diagMat[tid / 8u][tid % 8u][tid % 8u] = alpha;
                    } else {
                        for (int kk = 0; kk < BK; ++kk) pBuf[tid][kk] = bfloat16_t(0.0f);
                        diagMat[tid / 8u][tid % 8u][tid % 8u] = 1.0f;
                    }
                }
                // Load V tile concurrently after K scores are done (overwrite kvTile).
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (int idx = int(tid); idx < BK * (D / 4); idx += 128) {
                    int jj = idx / (D / 4);
                    int d4 = (idx % (D / 4)) * 4;
                    vec<bfloat16_t, 4> val = (jj < tileLen)
                        ? *(const device vec<bfloat16_t, 4>*)(v + kvHeadBase + (size_t)(jBase + jj) * D + d4)
                        : vec<bfloat16_t, 4>(bfloat16_t(0.f));
                    *(threadgroup vec<bfloat16_t, 4>*)(&kvTile[jj][d4]) = val;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                simdgroup_float8x8 dA;
                simdgroup_load(dA, &diagMat[sg][0][0], 8);
                simdgroup_bfloat8x8 p0, p1, p2, p3;
                simdgroup_load(p0, &pBuf[sg_m][0], BK);
                simdgroup_load(p1, &pBuf[sg_m][8], BK);
                simdgroup_load(p2, &pBuf[sg_m][16], BK);
                simdgroup_load(p3, &pBuf[sg_m][24], BK);

                #pragma clang loop unroll(full)
                for (int j = 0; j < 16; ++j) {
                    simdgroup_multiply(accO[j], dA, accO[j]);
                    simdgroup_bfloat8x8 v0, v1, v2, v3;
                    simdgroup_load(v0, &kvTile[0][j * 8], D);
                    simdgroup_load(v1, &kvTile[8][j * 8], D);
                    simdgroup_load(v2, &kvTile[16][j * 8], D);
                    simdgroup_load(v3, &kvTile[24][j * 8], D);
                    simdgroup_multiply_accumulate(accO[j], p0, v0, accO[j]);
                    simdgroup_multiply_accumulate(accO[j], p1, v1, accO[j]);
                    simdgroup_multiply_accumulate(accO[j], p2, v2, accO[j]);
                    simdgroup_multiply_accumulate(accO[j], p3, v3, accO[j]);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            if (tid < uint(BQ)) {
                int tAbs = int(tBlock) + int(tid);
                diagMat[tid / 8u][tid % 8u][tid % 8u] = (tAbs < T) ? (1.0f / lState[tid]) : 0.0f;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            simdgroup_float8x8 dInv;
            simdgroup_load(dInv, &diagMat[sg][0][0], 8);
            uint lane = thread_index_in_simdgroup;
            for (int grp = 0; grp < 4; ++grp) {
                #pragma clang loop unroll(full)
                for (int c = 0; c < 4; ++c) {
                    int j = grp * 4 + c;
                    simdgroup_multiply(accO[j], dInv, accO[j]);
                    simdgroup_store(accO[j], &sBuf[sg_m][c * 8], BK + 2);
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
                for (int idx = int(lane); idx < 8 * 32; idx += 32) {
                    int rh = idx / 32;
                    int dc = idx % 32;
                    int tAbs = int(tBlock) + int(sg_m) + rh;
                    if (tAbs < T) {
                        out[qHeadBase + (size_t)tAbs * D + grp * 32 + dc] =
                            bfloat16_t(sBuf[sg_m + rh][dc]);
                    }
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
            """,
        header: mmaHeader)

    /// Zero-mask Split-K FlashDecoding Part 1 for single-token AR decode (T=1, D=128, DR=16).
    /// Reads each K and V row ONCE for all GQA=3 query heads simultaneously (zero mask allocation).
    static let decodeSplitKT1Kernel = MLXFast.metalKernel(
        name: "inkling2_decode_splitk_t1",
        inputNames: ["q", "k", "v", "r", "projT", "meta", "scaleArr"],
        outputNames: ["partOut", "partM", "partL"],
        source: """
            constexpr int D = 128;
            constexpr int DR = 16;
            uint hkv = threadgroup_position_in_grid.x;
            uint sp = threadgroup_position_in_grid.y;
            uint b = threadgroup_position_in_grid.z;
            uint tid = thread_position_in_threadgroup.x;
            uint sg = simdgroup_index_in_threadgroup;
            uint lane = thread_index_in_simdgroup;
            int NQ = q_shape[1];
            int NKV = k_shape[1];
            int S = k_shape[2];
            int seen = meta[0];
            int kStart = meta[1];
            int E = meta[2];
            int numSplits = meta[3];
            float sc = scaleArr[0];

            int chunk = (S + numSplits - 1) / numSplits;
            int jStart = int(sp) * chunk;
            int jEnd = min(S, jStart + chunk);

            float4 qv[GQA];
            float rVal[GQA];
            float4 acc[GQA];
            float m[GQA];
            float l[GQA];
            for (int g = 0; g < GQA; ++g) {
                int h = int(hkv) * GQA + g;
                size_t qOff = (((size_t)b * NQ + h) * D) + lane * 4u;
                qv[g] = float4(*(const device vec<bfloat16_t, 4>*)(q + qOff)) * sc;
                rVal[g] = (lane < uint(DR)) ? float(r[((size_t)b * NQ + h) * DR + lane]) : 0.0f;
                acc[g] = float4(0.0f);
                m[g] = -INFINITY;
                l[g] = 0.0f;
            }

            size_t kvHeadBase = ((size_t)b * NKV + hkv) * S * D;
            bool splitHasRel = (jStart < jEnd) && ((seen - (kStart + jEnd - 1)) < E);

            for (int j = jStart + int(sg); j < jEnd; j += 4) {
                int d = seen - (kStart + j);
                if (d < 0) continue;
                size_t kvOff = kvHeadBase + (size_t)j * D + lane * 4u;
                float4 kv = float4(*(const device vec<bfloat16_t, 4>*)(k + kvOff));
                float4 vv = float4(*(const device vec<bfloat16_t, 4>*)(v + kvOff));
                float pVal = (splitHasRel && d < E && lane < uint(DR))
                    ? float(projT[(size_t)d * DR + lane]) : 0.0f;
                #pragma clang loop unroll(full)
                for (int g = 0; g < GQA; ++g) {
                    float dot = qv[g].x * kv.x + qv[g].y * kv.y + qv[g].z * kv.z + qv[g].w * kv.w;
                    float score = simd_sum(dot);
                    if (splitHasRel && d < E) {
                        float rp = (lane < uint(DR)) ? (rVal[g] * pVal) : 0.0f;
                        score += float(bfloat16_t(simd_sum(rp)));
                    }
                    float mNew = max(m[g], score);
                    float alpha = (l[g] > 0.0f) ? exp(m[g] - mNew) : 0.0f;
                    float p = exp(score - mNew);
                    l[g] = l[g] * alpha + p;
                    acc[g] = acc[g] * alpha + vv * p;
                    m[g] = mNew;
                }
            }

            threadgroup float shM[GQA][4];
            threadgroup float shL[GQA][4];
            threadgroup float shOut[GQA][4][D];
            for (int g = 0; g < GQA; ++g) {
                if (lane == 0u) {
                    shM[g][sg] = m[g];
                    shL[g][sg] = l[g];
                }
                shOut[g][sg][lane * 4u + 0u] = acc[g].x;
                shOut[g][sg][lane * 4u + 1u] = acc[g].y;
                shOut[g][sg][lane * 4u + 2u] = acc[g].z;
                shOut[g][sg][lane * 4u + 3u] = acc[g].w;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (sg < uint(GQA)) {
                int g = int(sg);
                int h = int(hkv) * GQA + g;
                float mTot = max(max(shM[g][0], shM[g][1]), max(shM[g][2], shM[g][3]));
                float4 oTot = float4(0.0f);
                float lTot = 0.0f;
                if (mTot > -1e30f) {
                    for (int s2 = 0; s2 < 4; ++s2) {
                        if (shL[g][s2] > 0.0f) {
                            float w2 = exp(shM[g][s2] - mTot);
                            lTot += shL[g][s2] * w2;
                            oTot.x += shOut[g][s2][lane * 4u + 0u] * w2;
                            oTot.y += shOut[g][s2][lane * 4u + 1u] * w2;
                            oTot.z += shOut[g][s2][lane * 4u + 2u] * w2;
                            oTot.w += shOut[g][s2][lane * 4u + 3u] * w2;
                        }
                    }
                }
                size_t spIdx = ((size_t)b * NQ + h) * numSplits + sp;
                if (lane == 0u) {
                    partM[spIdx] = mTot;
                    partL[spIdx] = lTot;
                }
                size_t outOff = spIdx * D + lane * 4u;
                partOut[outOff + 0u] = oTot.x;
                partOut[outOff + 1u] = oTot.y;
                partOut[outOff + 2u] = oTot.z;
                partOut[outOff + 3u] = oTot.w;
            }
            """)

    /// Zero-mask Split-K Hardware-MMA FlashDecoding Part 1 for multi-row DFlash 2 verify (T = 2..16).
    /// Each 128-thread TG owns 1 KV head x up to 32 flattened (localG, t) queries x 1 key split,
    /// reading K and V ONCE for all GQA*T queries (e.g. 24 queries at T=8) with zero mask buffer.
    static let decodeSplitKMMKernel = MLXFast.metalKernel(
        name: "inkling2_decode_splitk_mma",
        inputNames: ["q", "k", "v", "r", "projT", "meta", "scaleArr"],
        outputNames: ["partOut", "partM", "partL"],
        source: """
            constexpr int D = 128;
            constexpr int DR = 16;
            constexpr int BQ = 32;
            constexpr int BK = 32;
            uint hkv = threadgroup_position_in_grid.x / Q_BLOCKS;
            uint qBlk = threadgroup_position_in_grid.x % Q_BLOCKS;
            uint sp = threadgroup_position_in_grid.y;
            uint b = threadgroup_position_in_grid.z;
            uint tid = thread_position_in_threadgroup.x;
            uint sg = simdgroup_index_in_threadgroup;
            int NQ = q_shape[1];
            int T = q_shape[2];
            int NKV = k_shape[1];
            int S = k_shape[2];
            int seen = meta[0];
            int kStart = meta[1];
            int E = meta[2];
            int numSplits = meta[3];
            float sc = scaleArr[0];

            int totalQ = GQA * T;
            int qBase = int(qBlk) * BQ;
            int chunk = (S + numSplits - 1) / numSplits;
            int jStart = int(sp) * chunk;
            int jEnd = min(S, jStart + chunk);

            threadgroup bfloat16_t qTile[BQ][D];
            threadgroup bfloat16_t kvTile[BK][D];
            threadgroup float sBuf[BQ][BK + 2];
            threadgroup bfloat16_t pBuf[BQ][BK];
            threadgroup bfloat16_t rTile[BQ][DR];
            threadgroup float diagMat[4][8][8];
            threadgroup float mState[BQ];
            threadgroup float lState[BQ];

            for (int idx = int(tid); idx < BQ * (D / 4); idx += 128) {
                int qr = idx / (D / 4);
                int d4 = (idx % (D / 4)) * 4;
                int qIdx = qBase + qr;
                vec<bfloat16_t, 4> val = vec<bfloat16_t, 4>(bfloat16_t(0.f));
                if (qIdx < totalQ) {
                    int g = qIdx / T;
                    int t = qIdx % T;
                    int h = int(hkv) * GQA + g;
                    val = *(const device vec<bfloat16_t, 4>*)(q + (((size_t)b * NQ + h) * T + t) * D + d4);
                }
                *(threadgroup vec<bfloat16_t, 4>*)(&qTile[qr][d4]) = val;
            }
            for (int idx = int(tid); idx < BQ * (DR / 4); idx += 128) {
                int qr = idx / (DR / 4);
                int d4 = (idx % (DR / 4)) * 4;
                int qIdx = qBase + qr;
                vec<bfloat16_t, 4> val = vec<bfloat16_t, 4>(bfloat16_t(0.f));
                if (qIdx < totalQ) {
                    int g = qIdx / T;
                    int t = qIdx % T;
                    int h = int(hkv) * GQA + g;
                    val = *(const device vec<bfloat16_t, 4>*)(r + (((size_t)b * T + t) * NQ + h) * DR + d4);
                }
                *(threadgroup vec<bfloat16_t, 4>*)(&rTile[qr][d4]) = val;
            }
            if (tid < uint(BQ)) {
                mState[tid] = -INFINITY;
                lState[tid] = 0.0f;
            }
            for (uint i = tid; i < 256u; i += 128u) {
                diagMat[i / 64u][(i % 64u) / 8u][i % 8u] = 0.0f;
            }

            simdgroup_float8x8 accO[16];
            #pragma clang loop unroll(full)
            for (int j = 0; j < 16; ++j) accO[j] = simdgroup_float8x8(0.0f);

            size_t kvHeadBase = ((size_t)b * NKV + hkv) * S * D;
            uint sg_m = sg * 8u;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (int jBase = jStart; jBase < jEnd; jBase += BK) {
                int tileLen = min(BK, jEnd - jBase);
                for (int idx = int(tid); idx < BK * (D / 4); idx += 128) {
                    int jj = idx / (D / 4);
                    int d4 = (idx % (D / 4)) * 4;
                    vec<bfloat16_t, 4> val = (jj < tileLen)
                        ? *(const device vec<bfloat16_t, 4>*)(k + kvHeadBase + (size_t)(jBase + jj) * D + d4)
                        : vec<bfloat16_t, 4>(bfloat16_t(0.f));
                    *(threadgroup vec<bfloat16_t, 4>*)(&kvTile[jj][d4]) = val;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                simdgroup_float8x8 accS0(0.0f), accS1(0.0f), accS2(0.0f), accS3(0.0f);
                for (int ks = 0; ks < D; ks += 8) {
                    simdgroup_bfloat8x8 qa, k0, k1, k2, k3;
                    simdgroup_load(qa, &qTile[sg_m][ks], D);
                    simdgroup_load(k0, &kvTile[0][ks], D, ulong2(0, 0), true);
                    simdgroup_load(k1, &kvTile[8][ks], D, ulong2(0, 0), true);
                    simdgroup_load(k2, &kvTile[16][ks], D, ulong2(0, 0), true);
                    simdgroup_load(k3, &kvTile[24][ks], D, ulong2(0, 0), true);
                    simdgroup_multiply_accumulate(accS0, qa, k0, accS0);
                    simdgroup_multiply_accumulate(accS1, qa, k1, accS1);
                    simdgroup_multiply_accumulate(accS2, qa, k2, accS2);
                    simdgroup_multiply_accumulate(accS3, qa, k3, accS3);
                }
                simdgroup_store(accS0, &sBuf[sg_m][0], BK + 2);
                simdgroup_store(accS1, &sBuf[sg_m][8], BK + 2);
                simdgroup_store(accS2, &sBuf[sg_m][16], BK + 2);
                simdgroup_store(accS3, &sBuf[sg_m][24], BK + 2);
                threadgroup_barrier(mem_flags::mem_threadgroup);

                bool pureUnmasked = (tileLen == BK) && (kStart + jBase + BK - 1 <= seen - E);
                if (!pureUnmasked) {
                    int qr = int(tid / 4u);
                    int kk0 = int(tid % 4u) * 8;
                    int qIdx = qBase + qr;
                    int t = (qIdx < totalQ) ? (qIdx % T) : -1;
                    int qPos = seen + t;
                    for (int kk = kk0; kk < kk0 + 8; ++kk) {
                        if (kk >= tileLen || qIdx >= totalQ) {
                            sBuf[qr][kk] = -INFINITY;
                        } else {
                            int d = qPos - (kStart + jBase + kk);
                            if (d < 0) {
                                sBuf[qr][kk] = -INFINITY;
                            } else {
                                float s = sBuf[qr][kk] * sc;
                                if (d < E) {
                                    const device bfloat16_t* pRow = projT + (size_t)d * DR;
                                    float dot = 0.0f;
                                    #pragma clang loop unroll(full)
                                    for (int dr = 0; dr < DR; ++dr) {
                                        dot += float(rTile[qr][dr]) * float(pRow[dr]);
                                    }
                                    s += float(bfloat16_t(dot));
                                }
                                sBuf[qr][kk] = s;
                            }
                        }
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }

                if (tid < uint(BQ)) {
                    int qIdx = qBase + int(tid);
                    if (qIdx < totalQ) {
                        float mOld = mState[tid];
                        float mNew = mOld;
                        float rowS[BK];
                        if (pureUnmasked) {
                            for (int kk = 0; kk < BK; ++kk) {
                                float s = sBuf[tid][kk] * sc;
                                rowS[kk] = s;
                                mNew = max(mNew, s);
                            }
                        } else {
                            for (int kk = 0; kk < BK; ++kk) {
                                float s = sBuf[tid][kk];
                                rowS[kk] = s;
                                mNew = max(mNew, s);
                            }
                        }
                        if (mNew > -1e30f) {
                            float alpha = (lState[tid] > 0.0f) ? exp(mOld - mNew) : 0.0f;
                            float lNew = lState[tid] * alpha;
                            for (int kk = 0; kk < BK; ++kk) {
                                float p = (rowS[kk] > -1e30f) ? exp(rowS[kk] - mNew) : 0.0f;
                                lNew += p;
                                pBuf[tid][kk] = bfloat16_t(p);
                            }
                            mState[tid] = mNew;
                            lState[tid] = lNew;
                            diagMat[tid / 8u][tid % 8u][tid % 8u] = alpha;
                        } else {
                            for (int kk = 0; kk < BK; ++kk) pBuf[tid][kk] = bfloat16_t(0.0f);
                            diagMat[tid / 8u][tid % 8u][tid % 8u] = 1.0f;
                        }
                    } else {
                        for (int kk = 0; kk < BK; ++kk) pBuf[tid][kk] = bfloat16_t(0.0f);
                        diagMat[tid / 8u][tid % 8u][tid % 8u] = 1.0f;
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (int idx = int(tid); idx < BK * (D / 4); idx += 128) {
                    int jj = idx / (D / 4);
                    int d4 = (idx % (D / 4)) * 4;
                    vec<bfloat16_t, 4> val = (jj < tileLen)
                        ? *(const device vec<bfloat16_t, 4>*)(v + kvHeadBase + (size_t)(jBase + jj) * D + d4)
                        : vec<bfloat16_t, 4>(bfloat16_t(0.f));
                    *(threadgroup vec<bfloat16_t, 4>*)(&kvTile[jj][d4]) = val;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                simdgroup_float8x8 dA;
                simdgroup_load(dA, &diagMat[sg][0][0], 8);
                simdgroup_bfloat8x8 p0, p1, p2, p3;
                simdgroup_load(p0, &pBuf[sg_m][0], BK);
                simdgroup_load(p1, &pBuf[sg_m][8], BK);
                simdgroup_load(p2, &pBuf[sg_m][16], BK);
                simdgroup_load(p3, &pBuf[sg_m][24], BK);

                #pragma clang loop unroll(full)
                for (int j = 0; j < 16; ++j) {
                    simdgroup_multiply(accO[j], dA, accO[j]);
                    simdgroup_bfloat8x8 v0, v1, v2, v3;
                    simdgroup_load(v0, &kvTile[0][j * 8], D);
                    simdgroup_load(v1, &kvTile[8][j * 8], D);
                    simdgroup_load(v2, &kvTile[16][j * 8], D);
                    simdgroup_load(v3, &kvTile[24][j * 8], D);
                    simdgroup_multiply_accumulate(accO[j], p0, v0, accO[j]);
                    simdgroup_multiply_accumulate(accO[j], p1, v1, accO[j]);
                    simdgroup_multiply_accumulate(accO[j], p2, v2, accO[j]);
                    simdgroup_multiply_accumulate(accO[j], p3, v3, accO[j]);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            if (tid < uint(BQ)) {
                int qIdx = qBase + int(tid);
                if (qIdx < totalQ) {
                    int g = qIdx / T;
                    int t = qIdx % T;
                    int h = int(hkv) * GQA + g;
                    size_t spIdx = (((size_t)b * NQ + h) * T + t) * numSplits + sp;
                    partM[spIdx] = mState[tid];
                    partL[spIdx] = lState[tid];
                }
            }
            uint lane = thread_index_in_simdgroup;
            for (int grp = 0; grp < 4; ++grp) {
                #pragma clang loop unroll(full)
                for (int c = 0; c < 4; ++c) {
                    simdgroup_store(accO[grp * 4 + c], &sBuf[sg_m][c * 8], BK + 2);
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
                for (int idx = int(lane); idx < 8 * 32; idx += 32) {
                    int rh = idx / 32;
                    int dc = idx % 32;
                    int qIdx = qBase + int(sg_m) + rh;
                    if (qIdx < totalQ) {
                        int g = qIdx / T;
                        int t = qIdx % T;
                        int h = int(hkv) * GQA + g;
                        size_t spIdx = (((size_t)b * NQ + h) * T + t) * numSplits + sp;
                        partOut[spIdx * D + grp * 32 + dc] = sBuf[sg_m + rh][dc];
                    }
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
            """,
        header: mmaHeader)

    /// Split-K FlashDecoding Part 2: exact FP32 online log-sum-exp reduction across all splits + native sink.
    static let decodeSplitKReduceKernel = MLXFast.metalKernel(
        name: "inkling2_decode_splitk_reduce",
        inputNames: ["partOut", "partM", "partL", "sinks"],
        outputNames: ["out"],
        source: """
            constexpr int D = 128;
            uint h = threadgroup_position_in_grid.x;
            uint t = threadgroup_position_in_grid.y;
            uint b = threadgroup_position_in_grid.z;
            uint d = thread_position_in_threadgroup.x;
            int NQ = partM_shape[1];
            int T = partM_shape[2];
            int numSplits = partM_shape[3];

            float m = float(sinks[((size_t)b * T + t) * NQ + h]);
            float l = 1.0f;
            float acc = 0.0f;
            size_t base = (((size_t)b * NQ + h) * T + t) * numSplits;

            for (int sp = 0; sp < numSplits; ++sp) {
                float ls = partL[base + sp];
                if (ls <= 0.0f) continue;
                float ms = partM[base + sp];
                float mNew = max(m, ms);
                float wOld = exp(m - mNew);
                float wNew = exp(ms - mNew);
                l = l * wOld + ls * wNew;
                acc = acc * wOld + partOut[(base + sp) * D + d] * wNew;
                m = mNew;
            }
            out[(((size_t)b * NQ + h) * T + t) * D + d] = bfloat16_t(acc / l);
            """)

    /// Scalar fallback for tiny T < 8 prefill chunks.
    static let prefillAttnKernel = MLXFast.metalKernel(
        name: "inkling2_prefill_attn",
        inputNames: ["q", "k", "v", "r", "projT", "sinks", "meta", "scaleArr"],
        outputNames: ["out"],
        source: """
            constexpr int D = 128;
            constexpr int DR = 16;
            constexpr int BK = 16;
            uint hkv = threadgroup_position_in_grid.x;
            uint tBlock = threadgroup_position_in_grid.y * Q_ROWS;
            uint b = threadgroup_position_in_grid.z;
            uint tid = thread_position_in_threadgroup.x;
            uint sg = simdgroup_index_in_threadgroup;
            uint lane = thread_index_in_simdgroup;
            int NQ = q_shape[1];
            int T = q_shape[2];
            int NKV = k_shape[1];
            int S = k_shape[2];
            int seen = meta[0];
            int kStart = meta[1];
            int W = meta[2];
            int E = meta[3];
            float sc = scaleArr[0];

            uint localT = sg / GQA;
            uint localG = sg % GQA;
            int t = int(tBlock + localT);
            int h = int(hkv * GQA + localG);
            bool activeQ = (t < T) && (h < NQ);

            float q0 = 0.f, q1 = 0.f, q2 = 0.f, q3 = 0.f;
            float rVal = 0.f;
            float m = -INFINITY;
            float l = 1.0f;
            float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;

            if (activeQ) {
                size_t qBase = (((size_t)b * NQ + h) * T + t) * D;
                q0 = float(q[qBase + lane]) * sc;
                q1 = float(q[qBase + lane + 32]) * sc;
                q2 = float(q[qBase + lane + 64]) * sc;
                q3 = float(q[qBase + lane + 96]) * sc;
                if (lane < (uint)DR) {
                    rVal = float(r[(((size_t)b * T + t) * NQ + h) * DR + lane]);
                }
                m = float(sinks[((size_t)b * T + t) * NQ + h]);
            }

            int tFirst = int(tBlock);
            int tLast = min(T - 1, int(tBlock) + Q_ROWS - 1);
            int qPosMin = seen + tFirst;
            int qPosMax = seen + tLast;
            int jStart = (W > 0) ? max(0, qPosMin - W + 1 - kStart) : 0;
            int jEnd = min(S, qPosMax - kStart + 1);

            threadgroup bfloat16_t kTile[BK][D];
            threadgroup bfloat16_t vTile[BK][D];
            constexpr uint TG_THREADS = 32 * GQA * Q_ROWS;
            size_t kvHeadBase = ((size_t)b * NKV + hkv) * S * D;

            for (int jBase = jStart; jBase < jEnd; jBase += BK) {
                int tileLen = min(BK, jEnd - jBase);
                int totalElems = tileLen * D;
                for (int idx = int(tid); idx < totalElems; idx += int(TG_THREADS)) {
                    int jj = idx / D;
                    int dd = idx % D;
                    size_t src = kvHeadBase + (size_t)(jBase + jj) * D + dd;
                    kTile[jj][dd] = k[src];
                    vTile[jj][dd] = v[src];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                if (activeQ) {
                    int qPos = seen + t;
                    int tileMinDist = qPos - (kStart + jBase + tileLen - 1);
                    int tileMaxDist = qPos - (kStart + jBase);
                    bool tileHasRel = (tileMaxDist >= 0) && (tileMinDist < E);
                    for (int jj = 0; jj < tileLen; ++jj) {
                        int d = qPos - (kStart + jBase + jj);
                        if (d < 0 || (W > 0 && d >= W)) continue;
                        float dot = q0 * float(kTile[jj][lane])
                                  + q1 * float(kTile[jj][lane + 32])
                                  + q2 * float(kTile[jj][lane + 64])
                                  + q3 * float(kTile[jj][lane + 96]);
                        float score = simd_sum(dot);
                        if (tileHasRel && d < E) {
                            float rp = (lane < (uint)DR)
                                ? (rVal * float(projT[(size_t)d * DR + lane])) : 0.f;
                            float bias = float(bfloat16_t(simd_sum(rp)));
                            score += bias;
                        }
                        float mNew = max(m, score);
                        float alpha = exp(m - mNew);
                        float p = exp(score - mNew);
                        l = l * alpha + p;
                        a0 = a0 * alpha + p * float(vTile[jj][lane]);
                        a1 = a1 * alpha + p * float(vTile[jj][lane + 32]);
                        a2 = a2 * alpha + p * float(vTile[jj][lane + 64]);
                        a3 = a3 * alpha + p * float(vTile[jj][lane + 96]);
                        m = mNew;
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            if (activeQ) {
                float invL = 1.0f / l;
                size_t oBase = (((size_t)b * NQ + h) * T + t) * D;
                out[oBase + lane] = bfloat16_t(a0 * invL);
                out[oBase + lane + 32] = bfloat16_t(a1 * invL);
                out[oBase + lane + 64] = bfloat16_t(a2 * invL);
                out[oBase + lane + 96] = bfloat16_t(a3 * invL);
            }
            """)

    static func prefillAttention(
        q: MLXArray, k: MLXArray, v: MLXArray, r: MLXArray, projT: MLXArray, sinks: MLXArray,
        seen: Int, kStart: Int, window: Int?, relExtent: Int, scale: Float
    ) -> MLXArray? {
        #if canImport(Metal)
        guard Device.defaultDevice().deviceType == .gpu,
            q.dtype == .bfloat16, k.dtype == .bfloat16, v.dtype == .bfloat16,
            r.dtype == .bfloat16, projT.dtype == .bfloat16,
            q.dim(3) == 128, r.dim(3) == 16
        else { return nil }
        let (B, NQ, T, D) = (q.dim(0), q.dim(1), q.dim(2), q.dim(3))
        let NKV = k.dim(1)
        guard NKV > 0, NQ % NKV == 0 else { return nil }
        let gqa = NQ / NKV
        let meta = MLXArray([Int32(seen), Int32(kStart), Int32(window ?? 0), Int32(relExtent)])
        let scaleArr = MLXArray([scale])
        if T >= 2, B == 1,
            let mppOut = prefillAttentionMPP(
                q: q, k: k, v: v, r: r, projT: projT, sinks: sinks,
                seen: seen, kStart: kStart, window: window ?? 0, relExtent: relExtent, scale: scale)
        {
            return mppOut
        }
        if T >= 8 {
            let tBlocks = (T + 31) / 32
            return prefillAttnMMAKernel(
                [
                    MLX.contiguous(q), MLX.contiguous(k), MLX.contiguous(v),
                    MLX.contiguous(r), MLX.contiguous(projT),
                    MLX.contiguous(sinks.asType(.bfloat16)), meta, scaleArr,
                ],
                template: [("GQA", gqa)],
                grid: (NQ * 128, tBlocks, B),
                threadGroup: (128, 1, 1),
                outputShapes: [[B, NQ, T, D]],
                outputDTypes: [.bfloat16])[0]
        }
        let qRows = gqa <= 3 ? 4 : 2
        guard gqa * qRows <= 16 else { return nil }
        let tgThreads = 32 * gqa * qRows
        let tBlocks = (T + qRows - 1) / qRows
        return prefillAttnKernel(
            [
                MLX.contiguous(q), MLX.contiguous(k), MLX.contiguous(v),
                MLX.contiguous(r), MLX.contiguous(projT),
                MLX.contiguous(sinks.asType(.bfloat16)), meta, scaleArr,
            ],
            template: [("GQA", gqa), ("Q_ROWS", qRows)],
            grid: (NKV * tgThreads, tBlocks, B),
            threadGroup: (tgThreads, 1, 1),
            outputShapes: [[B, NQ, T, D]],
            outputDTypes: [.bfloat16])[0]
        #else
        return nil
        #endif
    }

    static func decodeSplitKAttention(
        q: MLXArray, k: MLXArray, v: MLXArray, r: MLXArray, projT: MLXArray, sinks: MLXArray,
        seen: Int, kStart: Int, relExtent: Int, scale: Float
    ) -> MLXArray? {
        #if canImport(Metal)
        guard Device.defaultDevice().deviceType == .gpu,
            q.dtype == .bfloat16, k.dtype == .bfloat16, v.dtype == .bfloat16,
            r.dtype == .bfloat16, projT.dtype == .bfloat16,
            q.dim(3) == 128, r.dim(3) == 16
        else { return nil }
        let (B, NQ, T, D) = (q.dim(0), q.dim(1), q.dim(2), q.dim(3))
        let NKV = k.dim(1)
        let S = k.dim(2)
        guard NKV > 0, NQ % NKV == 0, T >= 1, T <= 16, S > 0 else { return nil }
        let gqa = NQ / NKV
        let numSplits = max(1, min(64, (S + 255) / 256))
        let meta = MLXArray([Int32(seen), Int32(kStart), Int32(relExtent), Int32(numSplits)])
        let scaleArr = MLXArray([scale])
        let parts: [MLXArray]
        if T == 1 {
            guard gqa <= 4 else { return nil }
            parts = decodeSplitKT1Kernel(
                [
                    MLX.contiguous(q), MLX.contiguous(k), MLX.contiguous(v),
                    MLX.contiguous(r), MLX.contiguous(projT), meta, scaleArr,
                ],
                template: [("GQA", gqa)],
                grid: (NKV * 128, numSplits, B),
                threadGroup: (128, 1, 1),
                outputShapes: [[B, NQ, 1, numSplits, D], [B, NQ, 1, numSplits], [B, NQ, 1, numSplits]],
                outputDTypes: [.float32, .float32, .float32])
        } else {
            let qBlocks = (gqa * T + 31) / 32
            parts = decodeSplitKMMKernel(
                [
                    MLX.contiguous(q), MLX.contiguous(k), MLX.contiguous(v),
                    MLX.contiguous(r), MLX.contiguous(projT), meta, scaleArr,
                ],
                template: [("GQA", gqa), ("Q_BLOCKS", qBlocks)],
                grid: (NKV * qBlocks * 128, numSplits, B),
                threadGroup: (128, 1, 1),
                outputShapes: [[B, NQ, T, numSplits, D], [B, NQ, T, numSplits], [B, NQ, T, numSplits]],
                outputDTypes: [.float32, .float32, .float32])
        }
        return decodeSplitKReduceKernel(
            [parts[0], parts[1], parts[2], MLX.contiguous(sinks.asType(.bfloat16))],
            grid: (NQ * 128, T, B),
            threadGroup: (128, 1, 1),
            outputShapes: [[B, NQ, T, D]],
            outputDTypes: [.bfloat16])[0]
        #else
        return nil
        #endif
    }

    // MARK: Multi-Vector Affine 8-bit g=64 GEMV (fused optional softcap + unpadded vocab slice)

    #if canImport(Metal)
    private static let stdMetalHeader = """
        #include <metal_stdlib>
        using namespace metal;
        """

    private static let mppHeader = """
        #include <metal_stdlib>
        #include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
        using namespace metal;
        using namespace mpp::tensor_ops;
        """

    private static let affine8G64MultiGemvKernel = MLXFast.metalKernel(
        name: "inkling2_aff8g64_multi_gemv",
        inputNames: ["x", "w", "scales", "biases", "cap_buf"],
        outputNames: ["out"],
        source: """
            const uint sg = simdgroup_index_in_threadgroup;
            const uint lane = thread_index_in_simdgroup;
            const uint row0 = threadgroup_position_in_grid.x * 8u + sg * 2u;
            if (row0 >= uint(OUT_N)) return;

            const uint r_valid = metal::min(2u, uint(OUT_N) - row0);
            const uint KW = uint(K) / 4u;
            const uint NG = uint(K) / 64u;
            const uint g_sub = lane >> 3;

            float acc[ROWS][2];
            #pragma clang loop unroll(full)
            for (uint v = 0u; v < uint(ROWS); ++v) {
                acc[v][0] = 0.0f;
                acc[v][1] = 0.0f;
            }

            const device uint* w0 = w + size_t(row0) * KW;
            const device ST*   s0 = scales + size_t(row0) * NG;
            const device ST*   b0 = biases + size_t(row0) * NG;

            for (uint k0 = 0u; k0 < uint(K); k0 += 256u) {
                const uint k_lane = k0 + lane * 8u;
                const uint g = (k0 >> 6) + g_sub;
                float4 x_lo[ROWS], x_hi[ROWS];
                #pragma clang loop unroll(full)
                for (uint v = 0u; v < uint(ROWS); ++v) {
                    const device XT* xp = x + size_t(v) * uint(K) + k_lane;
                    x_lo[v] = float4(*(const device vec<XT, 4>*)(xp));
                    x_hi[v] = float4(*(const device vec<XT, 4>*)(xp + 4u));
                }

                #pragma clang loop unroll(full)
                for (uint r = 0u; r < 2u; ++r) {
                    if (r < r_valid) {
                        const device uint* wr = w0 + size_t(r) * KW + (k_lane >> 2);
                        const uint p0 = wr[0];
                        const uint p1 = wr[1];
                        const float sc = float(s0[size_t(r) * NG + g]);
                        const float bi = float(b0[size_t(r) * NG + g]);
                        const float4 w_lo = float4(
                            float(p0 & 0xFFu),
                            float((p0 >> 8) & 0xFFu),
                            float((p0 >> 16) & 0xFFu),
                            float((p0 >> 24) & 0xFFu)) * sc + bi;
                        const float4 w_hi = float4(
                            float(p1 & 0xFFu),
                            float((p1 >> 8) & 0xFFu),
                            float((p1 >> 16) & 0xFFu),
                            float((p1 >> 24) & 0xFFu)) * sc + bi;
                        #pragma clang loop unroll(full)
                        for (uint v = 0u; v < uint(ROWS); ++v) {
                            acc[v][r] += dot(x_lo[v], w_lo) + dot(x_hi[v], w_hi);
                        }
                    }
                }
            }

            const float cap = cap_buf[0];
            #pragma clang loop unroll(full)
            for (uint v = 0u; v < uint(ROWS); ++v) {
                #pragma clang loop unroll(full)
                for (uint r = 0u; r < 2u; ++r) {
                    float total = simd_sum(acc[v][r]);
                    if (lane == 0u && r < r_valid) {
                        if (HAS_CAP) {
                            total = metal::tanh(total / cap) * cap;
                        }
                        out[size_t(v) * uint(OUT_N) + row0 + r] = OT(total);
                    }
                }
            }
            """,
        header: stdMetalHeader,
        ensureRowContiguous: true)

    // MARK: Fused Inkling-2 MoE Router (E=128, K=2304, T=1..16)

    private static let routerScoresKernel = MLXFast.metalKernel(
        name: "inkling2_router_scores_fused",
        inputNames: ["x", "w"],
        outputNames: ["scores"],
        source: """
            const uint sgi = simdgroup_index_in_threadgroup;
            const uint lane = thread_index_in_simdgroup;
            const uint row = threadgroup_position_in_grid.x;
            if (row >= uint(E)) return;

            threadgroup float sh[ROWS][9];
            const uint k_lane = sgi * 256u + lane * 8u;
            const device WT* wp = w + size_t(row) * uint(K) + k_lane;
            float4 w_lo = float4(*(const device vec<WT, 4>*)(wp));
            float4 w_hi = float4(*(const device vec<WT, 4>*)(wp + 4u));

            #pragma clang loop unroll(full)
            for (uint t = 0u; t < uint(ROWS); ++t) {
                const device XT* xp = x + size_t(t) * uint(K) + k_lane;
                float4 x_lo = float4(*(const device vec<XT, 4>*)(xp));
                float4 x_hi = float4(*(const device vec<XT, 4>*)(xp + 4u));
                float s = simd_sum(dot(x_lo, w_lo) + dot(x_hi, w_hi));
                if (lane == 0u) sh[t][sgi] = s;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sgi == 0u && lane == 0u) {
                #pragma clang loop unroll(full)
                for (uint t = 0u; t < uint(ROWS); ++t) {
                    float tot = sh[t][0] + sh[t][1] + sh[t][2]
                              + sh[t][3] + sh[t][4] + sh[t][5]
                              + sh[t][6] + sh[t][7] + sh[t][8];
                    scores[size_t(t) * uint(E) + row] = 1.0f / (1.0f + metal::exp(-tot));
                }
            }
            """,
        header: stdMetalHeader,
        ensureRowContiguous: true)

    private static let routerTopKKernel = MLXFast.metalKernel(
        name: "inkling2_router_topk_fused",
        inputNames: ["scores", "bias", "scale_buf"],
        outputNames: ["out_ids", "out_w"],
        source: """
            const uint tid = thread_position_in_threadgroup.x;
            const uint sg = tid >> 5;
            const uint lane = tid & 31u;
            const uint row = threadgroup_position_in_grid.y;

            const float raw0 = scores[size_t(row) * 128u + tid];
            float val = raw0 + float(bias[tid]);

            threadgroup float sg_v[4];
            threadgroup uint  sg_i[4];
            threadgroup uint  win_i;
            threadgroup uint  picked_i[TOPK];
            threadgroup float picked_r[TOPK];

            for (uint k = 0u; k < uint(TOPK); ++k) {
                float sm = simd_max(val);
                uint  si = simd_min((val == sm) ? tid : 0xFFFFFFFFu);
                if (lane == 0u) {
                    sg_v[sg] = sm;
                    sg_i[sg] = si;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (tid == 0u) {
                    float bv = sg_v[0];
                    uint  bi = sg_i[0];
                    for (uint s = 1u; s < 4u; ++s) {
                        if (sg_v[s] > bv || (sg_v[s] == bv && sg_i[s] < bi)) {
                            bv = sg_v[s];
                            bi = sg_i[s];
                        }
                    }
                    win_i = bi;
                    picked_i[k] = bi;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (tid == win_i) {
                    picked_r[k] = raw0;
                    val = -INFINITY;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sg == 0u) {
                float r = (lane < uint(TOPK)) ? picked_r[lane] : 0.0f;
                float denom = simd_sum(r) + 1e-20f;
                if (lane < uint(TOPK)) {
                    out_ids[size_t(row) * uint(TOPK) + lane] = picked_i[lane];
                    out_w[size_t(row) * uint(TOPK) + lane] = (r / denom) * scale_buf[0];
                }
            }
            """,
        header: stdMetalHeader,
        ensureRowContiguous: true)

    // MARK: Zero-Mask Split-K Verify Attention (sliding & full, all S, LQ=1..16)

    private static let verifyAttnS1Body = """
            const uint tid = thread_position_in_threadgroup.x;
            const uint sg = tid >> 5;
            const uint lane = tid & 31u;
            const uint split = threadgroup_position_in_grid.y;
            const uint h = threadgroup_position_in_grid.z;
            const uint kv_h = h / uint(GQA);
            const int LQ = int(q_shape[2]);
            const int S  = int(k_shape[2]);
            const int n_splits = int(params[0]);
            const int k_min = int(params[1]);
            const float sc = scale_buf[0];

            const int span = S - k_min;
            const int chunk = (span + n_splits - 1) / n_splits;
            const int s0 = min(k_min + int(split) * chunk, S);
            const int s1 = min(s0 + chunk, S);

            const bool active = int(sg) < LQ;
            const int q_pos = (S - LQ) + int(sg);

            float4 qv = float4(0.0f);
            if (active) {
                qv = float4(*(const device vec<bfloat, 4>*)(q + ((size_t)h * LQ + int(sg)) * 128u + lane * 4u));
            }

            threadgroup float rel_sh[8 * 16];
            if (tid < uint(LQ * 16)) {
                uint qi = tid >> 4;
                uint d  = tid & 15u;
                rel_sh[tid] = float(rel[((size_t)qi * H + h) * 16u + d]);
            }
            threadgroup bfloat k_sh[8 * 128];
            threadgroup bfloat v_sh[8 * 128];
            threadgroup_barrier(mem_flags::mem_threadgroup);

            const size_t kv_base = (size_t)kv_h * (size_t)S * 128u;
            float m = -1e30f;
            float l = 0.0f;
            float4 o = float4(0.0f);

            for (int k0 = s0; k0 < s1; k0 += 8) {
                const int kp_load = k0 + int(sg);
                if (kp_load < s1) {
                    *(threadgroup vec<bfloat, 4>*)(k_sh + sg * 128u + lane * 4u) =
                        *(const device vec<bfloat, 4>*)(k + kv_base + (size_t)kp_load * 128u + lane * 4u);
                    *(threadgroup vec<bfloat, 4>*)(v_sh + sg * 128u + lane * 4u) =
                        *(const device vec<bfloat, 4>*)(v + kv_base + (size_t)kp_load * 128u + lane * 4u);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                if (active) {
                    const int b_lim = min(8, s1 - k0);
                    const threadgroup float* rq = rel_sh + sg * 16u;
                    for (int b = 0; b < b_lim; ++b) {
                        const int kp = k0 + b;
                        const int dist = q_pos - kp;
                        if (dist < 0 || (SLIDING > 0 && dist >= SLIDING)) continue;
                        float4 kv = float4(*(const threadgroup vec<bfloat, 4>*)(k_sh + uint(b) * 128u + lane * 4u));
                        float4 vv = float4(*(const threadgroup vec<bfloat, 4>*)(v_sh + uint(b) * 128u + lane * 4u));
                        float score = simd_sum(dot(qv, kv)) * sc;
                        if (dist < REL_EXTENT) {
                            const device vec<bfloat, 4>* rp = (const device vec<bfloat, 4>*)(rel_proj_t + (size_t)dist * 16u);
                            float4 a0 = float4(rp[0]), a1 = float4(rp[1]), a2 = float4(rp[2]), a3 = float4(rp[3]);
                            float rb = a0.x * rq[0]  + a0.y * rq[1]  + a0.z * rq[2]  + a0.w * rq[3]
                                     + a1.x * rq[4]  + a1.y * rq[5]  + a1.z * rq[6]  + a1.w * rq[7]
                                     + a2.x * rq[8]  + a2.y * rq[9]  + a2.z * rq[10] + a2.w * rq[11]
                                     + a3.x * rq[12] + a3.y * rq[13] + a3.z * rq[14] + a3.w * rq[15];
                            score += float(bfloat(rb));
                        }
                        float m_new = metal::max(m, score);
                        float alpha = (l > 0.0f) ? metal::exp(m - m_new) : 0.0f;
                        float p = metal::exp(score - m_new);
                        o = o * alpha + vv * p;
                        l = l * alpha + p;
                        m = m_new;
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (active) {
        """

    private static let verifyAttnS1SingleKernel = MLXFast.metalKernel(
        name: "inkling2_verify_attn_s1_single",
        inputNames: ["q", "k", "v", "rel", "rel_proj_t", "sinks", "params", "scale_buf"],
        outputNames: ["out"],
        source: verifyAttnS1Body + """
                const float sk = float(sinks[(size_t)int(sg) * H + h]);
                float gm = (l > 0.0f) ? metal::max(m, sk) : sk;
                float w_k = (l > 0.0f) ? metal::exp(m - gm) : 0.0f;
                float w_s = metal::exp(sk - gm);
                float inv_l = 1.0f / (l * w_k + w_s);
                float4 go = o * (w_k * inv_l);
                device vec<bfloat, 4>* dst = (device vec<bfloat, 4>*)(out + ((size_t)h * LQ + int(sg)) * 128u + lane * 4u);
                *dst = vec<bfloat, 4>(bfloat(go.x), bfloat(go.y), bfloat(go.z), bfloat(go.w));
            }
            """,
        header: stdMetalHeader,
        ensureRowContiguous: true)

    private static let verifyAttnS1MultiKernel = MLXFast.metalKernel(
        name: "inkling2_verify_attn_s1_multi",
        inputNames: ["q", "k", "v", "rel", "rel_proj_t", "sinks", "params", "scale_buf"],
        outputNames: ["part_m", "part_l", "part_o"],
        source: verifyAttnS1Body + """
                const size_t p_idx = ((size_t)h * LQ + int(sg)) * uint(n_splits) + split;
                if (lane == 0u) {
                    part_m[p_idx] = m;
                    part_l[p_idx] = l;
                }
                *(device float4*)(part_o + p_idx * 128u + lane * 4u) = o;
            }
            """,
        header: stdMetalHeader,
        ensureRowContiguous: true)

    private static let verifyAttnS2Kernel = MLXFast.metalKernel(
        name: "inkling2_verify_attn_s2",
        inputNames: ["part_m", "part_l", "part_o", "sinks"],
        outputNames: ["out"],
        source: """
            const uint lane = thread_index_in_simdgroup;
            const uint r = threadgroup_position_in_grid.y;
            const uint h = threadgroup_position_in_grid.z;
            const uint H = uint(part_m_shape[0]);
            const uint LQ = uint(part_m_shape[1]);
            const uint n_splits = uint(part_m_shape[2]);

            const size_t base = ((size_t)h * LQ + r) * n_splits;
            const device float* pm = part_m + base;
            const device float* pl = part_l + base;
            const device float* po = part_o + base * 128u;

            const float sk = float(sinks[(size_t)r * H + h]);
            float gm = sk;
            for (uint s = 0u; s < n_splits; ++s) {
                if (pl[s] > 0.0f) gm = metal::max(gm, pm[s]);
            }
            float gl = metal::exp(sk - gm);
            float4 go = float4(0.0f);
            for (uint s = 0u; s < n_splits; ++s) {
                if (pl[s] > 0.0f) {
                    float w = metal::exp(pm[s] - gm);
                    gl += pl[s] * w;
                    go += *(const device float4*)(po + (size_t)s * 128u + lane * 4u) * w;
                }
            }
            float inv_l = 1.0f / gl;
            device vec<bfloat, 4>* dst = (device vec<bfloat, 4>*)(out + ((size_t)h * LQ + r) * 128u + lane * 4u);
            *dst = vec<bfloat, 4>(bfloat(go.x * inv_l), bfloat(go.y * inv_l), bfloat(go.z * inv_l), bfloat(go.w * inv_l));
            """,
        header: stdMetalHeader,
        ensureRowContiguous: true)

    // MARK: Metal 4 MPP Kernels (GQA-grouped Split-K Verify + 32x128x128 FlashAttention Prefill)

    private static let mppLock = NSLock()
    nonisolated(unsafe) private static var mppPrefillCache: [String: MLXFast.MLXFastKernel] = [:]
    nonisolated(unsafe) private static var mppGQACache: [String: MLXFast.MLXFastKernel] = [:]
    nonisolated(unsafe) private static var mppVerified: Bool?

    private static let mppPrefillSource = #"""
          const uint tid = thread_position_in_threadgroup.x;
          const uint sg = tid >> 5;
          const uint lane = tid & 31u;
          const int LQ = int(q_shape[2]);
          const int S  = int(k_shape[2]);
          const int r0 = min(int(threadgroup_position_in_grid.y) * 32, LQ - 32);
          const int h  = int(threadgroup_position_in_grid.z);
          const int kv_h = h / GQA;
          const int q_offset = int(q_off_buf[0]);
          const float sc = scale_buf[0];

          const int k_start = (SLIDING > 0) ? max(0, q_offset + r0 - SLIDING + 1) : 0;
          const int k_end   = min(S, q_offset + r0 + 32);

          threadgroup float S_sh[32 * 128];
          threadgroup bfloat P_sh[32 * 128];
          threadgroup float rel_sh[32 * 16];
          threadgroup float m_sh[32];
          threadgroup float l_sh[32];
          threadgroup float alpha_sh[32];

          if (tid < 32u) {
            m_sh[tid] = float(sinks[(size_t)(r0 + int(tid)) * H + h]);
            l_sh[tid] = 1.0f;
          }
          for (uint i = tid; i < 32u * 16u; i += 128u) {
            uint qi = i >> 4;
            uint d  = i & 15u;
            rel_sh[i] = float(rel[((size_t)(r0 + int(qi)) * H + h) * 16u + d]);
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);

          constexpr auto desc_qk = matmul2d_descriptor(32, 128, 128, false, true, false, matmul2d_descriptor::mode::multiply);
          constexpr auto desc_pv = matmul2d_descriptor(32, 128, 128, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
          matmul2d<desc_qk, execution_simdgroups<4>> op_qk;
          matmul2d<desc_pv, execution_simdgroups<4>> op_pv;

          tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tQ(
              (device bfloat*)q + ((size_t)h * LQ + r0) * 128u, dextents<int32_t, 2>(128, 32));
          auto aQ = tQ.slice<128, 32>(0, 0);

          tensor<threadgroup bfloat, dextents<int32_t, 2>, tensor_inline> tP(
              P_sh, dextents<int32_t, 2>(128, 32));
          auto aP = tP.slice<128, 32>(0, 0);

          const size_t kv_head_base = (size_t)kv_h * (size_t)S * 128u;
          tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tV0(
              (device bfloat*)v + kv_head_base, dextents<int32_t, 2>(128, 128));
          auto bV0 = tV0.slice<128, 128>(0, 0);

          auto O_acc = op_pv.template get_destination_cooperative_tensor<decltype(aP), decltype(bV0), float>();
          #pragma unroll
          for (uint16_t i = 0; i < O_acc.get_capacity(); ++i) {
            if (O_acc.is_valid_element(i)) O_acc[i] = 0.0f;
          }

          for (int k0 = k_start; k0 < k_end; k0 += 128) {
            const int c0 = (k0 + 128 <= S) ? k0 : (S - 128);
            const bool is_clamped = (c0 < k0) || (c0 + 128 > k_end);
            const bool pure_prefix = (!is_clamped) && (SLIDING == 0) && ((q_offset + r0) - (c0 + 127) >= REL_EXTENT);

            tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tK(
                (device bfloat*)k + kv_head_base + (size_t)c0 * 128u, dextents<int32_t, 2>(128, 128));
            auto bK = tK.slice<128, 128>(0, 0);
            auto S_coop = op_qk.template get_destination_cooperative_tensor<decltype(aQ), decltype(bK), float>();
            op_qk.run(aQ, bK, S_coop);

            #pragma unroll
            for (uint16_t i = 0; i < S_coop.get_capacity(); ++i) {
              if (S_coop.is_valid_element(i)) {
                const auto idx = S_coop.get_multidimensional_index(i);
                S_sh[idx[1] * 128 + idx[0]] = S_coop[i] * sc;
              }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (pure_prefix) {
              for (uint r = 0u; r < 8u; ++r) {
                const uint qi = sg * 8u + r;
                const uint rb = qi * 128u;
                const float s0 = S_sh[rb + lane];
                const float s1 = S_sh[rb + lane + 32u];
                const float s2 = S_sh[rb + lane + 64u];
                const float s3 = S_sh[rb + lane + 96u];
                const float rmax = simd_max(metal::max(metal::max(s0, s1), metal::max(s2, s3)));
                const float m_old = m_sh[qi];
                const float m_new = metal::max(m_old, rmax);
                const float alpha = metal::exp(m_old - m_new);
                const float p0 = metal::exp(s0 - m_new);
                const float p1 = metal::exp(s1 - m_new);
                const float p2 = metal::exp(s2 - m_new);
                const float p3 = metal::exp(s3 - m_new);
                const float l_tile = simd_sum((p0 + p1) + (p2 + p3));
                if (lane == 0u) {
                  m_sh[qi] = m_new;
                  l_sh[qi] = l_sh[qi] * alpha + l_tile;
                  alpha_sh[qi] = alpha;
                }
                P_sh[rb + lane]       = bfloat(p0);
                P_sh[rb + lane + 32u] = bfloat(p1);
                P_sh[rb + lane + 64u] = bfloat(p2);
                P_sh[rb + lane + 96u] = bfloat(p3);
              }
            } else {
              for (uint r = 0u; r < 8u; ++r) {
                const uint qi = sg * 8u + r;
                const uint rb = qi * 128u;
                const int q_pos = q_offset + r0 + int(qi);
                const threadgroup float* rq = rel_sh + qi * 16u;
                float sv[4];
                #pragma unroll
                for (uint c = 0u; c < 4u; ++c) {
                  const uint ki = lane + c * 32u;
                  const int kp = c0 + int(ki);
                  float s = S_sh[rb + ki];
                  const int d = q_pos - kp;
                  if (kp < k0 || kp >= k_end || d < 0 || (SLIDING > 0 && d >= SLIDING)) {
                    s = -1e30f;
                  } else if (d < REL_EXTENT) {
                    const device vec<bfloat, 4>* rp = (const device vec<bfloat, 4>*)(rel_proj_t + (size_t)d * 16u);
                    float4 a0 = float4(rp[0]), a1 = float4(rp[1]), a2 = float4(rp[2]), a3 = float4(rp[3]);
                    float rbias = a0.x * rq[0]  + a0.y * rq[1]  + a0.z * rq[2]  + a0.w * rq[3]
                                + a1.x * rq[4]  + a1.y * rq[5]  + a1.z * rq[6]  + a1.w * rq[7]
                                + a2.x * rq[8]  + a2.y * rq[9]  + a2.z * rq[10] + a2.w * rq[11]
                                + a3.x * rq[12] + a3.y * rq[13] + a3.z * rq[14] + a3.w * rq[15];
                    s += float(bfloat(rbias));
                  }
                  sv[c] = s;
                }
                const float rmax = simd_max(metal::max(metal::max(sv[0], sv[1]), metal::max(sv[2], sv[3])));
                const float m_old = m_sh[qi];
                const float m_new = metal::max(m_old, rmax);
                const float alpha = metal::exp(m_old - m_new);
                float l_acc = 0.0f;
                #pragma unroll
                for (uint c = 0u; c < 4u; ++c) {
                  float p = (sv[c] > -1e29f) ? metal::exp(sv[c] - m_new) : 0.0f;
                  l_acc += p;
                  P_sh[rb + lane + c * 32u] = bfloat(p);
                }
                const float l_tile = simd_sum(l_acc);
                if (lane == 0u) {
                  m_sh[qi] = m_new;
                  l_sh[qi] = l_sh[qi] * alpha + l_tile;
                  alpha_sh[qi] = alpha;
                }
              }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            #pragma unroll
            for (uint16_t i = 0; i < O_acc.get_capacity(); ++i) {
              if (O_acc.is_valid_element(i)) {
                O_acc[i] *= alpha_sh[O_acc.get_multidimensional_index(i)[1]];
              }
            }
            tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tV(
                (device bfloat*)v + kv_head_base + (size_t)c0 * 128u, dextents<int32_t, 2>(128, 128));
            auto bV = tV.slice<128, 128>(0, 0);
            op_pv.run(aP, bV, O_acc);
            threadgroup_barrier(mem_flags::mem_threadgroup);
          }

          device bfloat* out_base = (device bfloat*)out + ((size_t)h * LQ + r0) * 128u;
          #pragma unroll
          for (uint16_t i = 0; i < O_acc.get_capacity(); ++i) {
            if (O_acc.is_valid_element(i)) {
              const auto idx = O_acc.get_multidimensional_index(i);
              out_base[idx[1] * 128 + idx[0]] = bfloat(O_acc[i] / l_sh[idx[1]]);
            }
          }
        """#

    private static let mppGQAS1Source = #"""
          const uint tid = thread_position_in_threadgroup.x;
          const uint sg = tid >> 5;
          const uint lane = tid & 31u;
          const int split = int(threadgroup_position_in_grid.y);
          const int kv_h = int(threadgroup_position_in_grid.z);

          const int LQ = int(q_shape[2]);
          const int S  = int(k_shape[2]);
          const int n_splits = int(params[0]);
          const int k_min = int(params[1]);
          const float sc = scale_buf[0];

          const int M = GQA * LQ;
          const int h0 = kv_h * GQA;
          const int span = S - k_min;
          const int chunk = (span + n_splits - 1) / n_splits;
          const int s0 = min(k_min + split * chunk, S);
          const int s1 = min(s0 + chunk, S);

          threadgroup bfloat Q_sh[TQ * 128];
          threadgroup float  S_sh[TQ * TK];
          threadgroup bfloat P_sh[TQ * TK];
          threadgroup bfloat rel_sh[TQ * 16];
          threadgroup float  m_sh[TQ];
          threadgroup float  l_sh[TQ];
          threadgroup float  alpha_sh[TQ];

          for (uint idx4 = tid; idx4 < uint(TQ * 32); idx4 += 128u) {
            const int m = int(idx4 >> 5);
            const uint d4 = (idx4 & 31u) * 4u;
            vec<bfloat, 4> qv = vec<bfloat, 4>(bfloat(0.0f));
            if (m < M) {
              const int g = m / LQ;
              const int qi = m - g * LQ;
              qv = *(const device vec<bfloat, 4>*)(q + ((size_t)(h0 + g) * LQ + qi) * 128u + d4);
            }
            *(threadgroup vec<bfloat, 4>*)(Q_sh + m * 128 + int(d4)) = qv;
          }
          for (uint i = tid; i < uint(TQ * 16); i += 128u) {
            const int m = int(i >> 4);
            const uint d = i & 15u;
            bfloat rv = bfloat(0.0f);
            if (m < M) {
              const int g = m / LQ;
              const int qi = m - g * LQ;
              rv = rel[((size_t)qi * H + (h0 + g)) * 16u + d];
            }
            rel_sh[i] = rv;
          }
          if (int(tid) < TQ) {
            m_sh[tid] = -1e30f;
            l_sh[tid] = 0.0f;
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);

          if (s0 >= s1) {
            for (int m = int(tid); m < M; m += 128) {
              const int g = m / LQ;
              const int qi = m - g * LQ;
              const size_t p_idx = ((size_t)(h0 + g) * LQ + qi) * uint(n_splits) + uint(split);
              part_m[p_idx] = -1e30f;
              part_l[p_idx] = 0.0f;
            }
            for (uint idx4 = tid; idx4 < uint(M * 32); idx4 += 128u) {
              const int m = int(idx4 >> 5);
              const uint d4 = (idx4 & 31u) * 4u;
              const int g = m / LQ;
              const int qi = m - g * LQ;
              const size_t p_idx = ((size_t)(h0 + g) * LQ + qi) * uint(n_splits) + uint(split);
              *(device float4*)(part_o + p_idx * 128u + d4) = float4(0.0f);
            }
            return;
          }

          constexpr auto desc_qk = matmul2d_descriptor(TQ, TK, 128, false, true, false, matmul2d_descriptor::mode::multiply);
          constexpr auto desc_pv = matmul2d_descriptor(TQ, 128, TK, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
          matmul2d<desc_qk, execution_simdgroups<4>> op_qk;
          matmul2d<desc_pv, execution_simdgroups<4>> op_pv;

          tensor<threadgroup bfloat, dextents<int32_t, 2>, tensor_inline> tQ(
              Q_sh, dextents<int32_t, 2>(128, TQ));
          auto aQ = tQ.slice<128, TQ>(0, 0);

          tensor<threadgroup bfloat, dextents<int32_t, 2>, tensor_inline> tP(
              P_sh, dextents<int32_t, 2>(TK, TQ));
          auto aP = tP.slice<TK, TQ>(0, 0);

          const size_t kv_head_base = (size_t)kv_h * (size_t)S * 128u;
          tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tV0(
              (device bfloat*)v + kv_head_base, dextents<int32_t, 2>(128, TK));
          auto bV0 = tV0.slice<128, TK>(0, 0);

          auto O_acc = op_pv.template get_destination_cooperative_tensor<decltype(aP), decltype(bV0), float>();
          #pragma unroll
          for (uint16_t i = 0; i < O_acc.get_capacity(); ++i) {
            if (O_acc.is_valid_element(i)) O_acc[i] = 0.0f;
          }

          const int q_base_pos = S - LQ;
          constexpr uint ROWS_PER_SG = TQ / 4;
          constexpr uint COLS_PER_LANE = TK / 32;

          for (int k0 = s0; k0 < s1; k0 += TK) {
            const int c0 = (k0 + TK <= S) ? k0 : (S - TK);
            const bool is_clamped = (c0 < k0) || (c0 + TK > s1);
            const bool pure_prefix = (!is_clamped) && (SLIDING == 0) && (q_base_pos - (c0 + TK - 1) >= REL_EXTENT);

            tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tK(
                (device bfloat*)k + kv_head_base + (size_t)c0 * 128u, dextents<int32_t, 2>(128, TK));
            auto bK = tK.slice<128, TK>(0, 0);
            auto S_coop = op_qk.template get_destination_cooperative_tensor<decltype(aQ), decltype(bK), float>();
            op_qk.run(aQ, bK, S_coop);

            #pragma unroll
            for (uint16_t i = 0; i < S_coop.get_capacity(); ++i) {
              if (S_coop.is_valid_element(i)) {
                const auto idx = S_coop.get_multidimensional_index(i);
                S_sh[idx[1] * TK + idx[0]] = S_coop[i] * sc;
              }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (pure_prefix) {
              for (uint r = 0u; r < ROWS_PER_SG; ++r) {
                const uint m = sg * ROWS_PER_SG + r;
                const uint rb = m * TK;
                float sv[COLS_PER_LANE];
                float rmax = -1e30f;
                #pragma unroll
                for (uint c = 0u; c < COLS_PER_LANE; ++c) {
                  float s = S_sh[rb + lane + c * 32u];
                  sv[c] = s;
                  rmax = metal::max(rmax, s);
                }
                rmax = simd_max(rmax);
                const float m_old = m_sh[m];
                const float l_old = l_sh[m];
                const float m_new = (l_old > 0.0f) ? metal::max(m_old, rmax) : rmax;
                const float alpha = (l_old > 0.0f) ? metal::exp(m_old - m_new) : 0.0f;
                float l_acc = 0.0f;
                #pragma unroll
                for (uint c = 0u; c < COLS_PER_LANE; ++c) {
                  float p = metal::exp(sv[c] - m_new);
                  l_acc += p;
                  P_sh[rb + lane + c * 32u] = bfloat(p);
                }
                const float l_tile = simd_sum(l_acc);
                if (lane == 0u) {
                  m_sh[m] = m_new;
                  l_sh[m] = l_old * alpha + l_tile;
                  alpha_sh[m] = alpha;
                }
              }
            } else {
              for (uint r = 0u; r < ROWS_PER_SG; ++r) {
                const int m = int(sg * ROWS_PER_SG + r);
                const uint rb = uint(m) * TK;
                const int qi = (m < M) ? (m % LQ) : 0;
                const int q_pos = q_base_pos + qi;
                const threadgroup bfloat* rq = rel_sh + uint(m) * 16u;
                float sv[COLS_PER_LANE];
                float rmax = -1e30f;
                #pragma unroll
                for (uint c = 0u; c < COLS_PER_LANE; ++c) {
                  const uint ki = lane + c * 32u;
                  const int kp = c0 + int(ki);
                  float s = S_sh[rb + ki];
                  const int d = q_pos - kp;
                  if (m >= M || kp < k0 || kp >= s1 || d < 0 || (SLIDING > 0 && d >= SLIDING)) {
                    s = -1e30f;
                  } else if (d < REL_EXTENT) {
                    const device vec<bfloat, 4>* rp = (const device vec<bfloat, 4>*)(rel_proj_t + (size_t)d * 16u);
                    float4 a0 = float4(rp[0]), a1 = float4(rp[1]), a2 = float4(rp[2]), a3 = float4(rp[3]);
                    float rbias = a0.x * float(rq[0])  + a0.y * float(rq[1])  + a0.z * float(rq[2])  + a0.w * float(rq[3])
                                + a1.x * float(rq[4])  + a1.y * float(rq[5])  + a1.z * float(rq[6])  + a1.w * float(rq[7])
                                + a2.x * float(rq[8])  + a2.y * float(rq[9])  + a2.z * float(rq[10]) + a2.w * float(rq[11])
                                + a3.x * float(rq[12]) + a3.y * float(rq[13]) + a3.z * float(rq[14]) + a3.w * float(rq[15]);
                    s += float(bfloat(rbias));
                  }
                  sv[c] = s;
                  rmax = metal::max(rmax, s);
                }
                rmax = simd_max(rmax);
                const float m_old = m_sh[m];
                const float l_old = l_sh[m];
                const bool has_valid = rmax > -1e29f;
                const float m_new = has_valid ? ((l_old > 0.0f) ? metal::max(m_old, rmax) : rmax) : m_old;
                const float alpha = (l_old > 0.0f && has_valid) ? metal::exp(m_old - m_new) : (l_old > 0.0f ? 1.0f : 0.0f);
                float l_acc = 0.0f;
                #pragma unroll
                for (uint c = 0u; c < COLS_PER_LANE; ++c) {
                  float p = (sv[c] > -1e29f) ? metal::exp(sv[c] - m_new) : 0.0f;
                  l_acc += p;
                  P_sh[rb + lane + c * 32u] = bfloat(p);
                }
                const float l_tile = simd_sum(l_acc);
                if (lane == 0u) {
                  m_sh[m] = m_new;
                  l_sh[m] = l_old * alpha + l_tile;
                  alpha_sh[m] = alpha;
                }
              }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            #pragma unroll
            for (uint16_t i = 0; i < O_acc.get_capacity(); ++i) {
              if (O_acc.is_valid_element(i)) {
                O_acc[i] *= alpha_sh[O_acc.get_multidimensional_index(i)[1]];
              }
            }
            tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tV(
                (device bfloat*)v + kv_head_base + (size_t)c0 * 128u, dextents<int32_t, 2>(128, TK));
            auto bV = tV.slice<128, TK>(0, 0);
            op_pv.run(aP, bV, O_acc);
            threadgroup_barrier(mem_flags::mem_threadgroup);
          }

          for (int m = int(tid); m < M; m += 128) {
            const int g = m / LQ;
            const int qi = m - g * LQ;
            const size_t p_idx = ((size_t)(h0 + g) * LQ + qi) * uint(n_splits) + uint(split);
            part_m[p_idx] = m_sh[m];
            part_l[p_idx] = l_sh[m];
          }
          #pragma unroll
          for (uint16_t i = 0; i < O_acc.get_capacity(); ++i) {
            if (O_acc.is_valid_element(i)) {
              const auto idx = O_acc.get_multidimensional_index(i);
              const int m = int(idx[1]);
              if (m < M) {
                const int g = m / LQ;
                const int qi = m - g * LQ;
                const size_t p_idx = ((size_t)(h0 + g) * LQ + qi) * uint(n_splits) + uint(split);
                part_o[p_idx * 128u + uint(idx[0])] = O_acc[i];
              }
            }
          }
        """#

    private static func mppPrefillKernel(h: Int, gqa: Int, sliding: Int, relExtent: Int)
        -> MLXFast.MLXFastKernel
    {
        let key = "h\(h)_g\(gqa)_s\(sliding)_r\(relExtent)"
        mppLock.lock()
        defer { mppLock.unlock() }
        if let k = mppPrefillCache[key] { return k }
        let consts = """
              constexpr int H = \(h);
              constexpr int GQA = \(gqa);
              constexpr int SLIDING = \(sliding);
              constexpr int REL_EXTENT = \(relExtent);
            """
        let k = MLXFast.metalKernel(
            name: "inkling2_mpp_fa128_\(key)",
            inputNames: ["q", "k", "v", "rel", "rel_proj_t", "sinks", "scale_buf", "q_off_buf"],
            outputNames: ["out"],
            source: consts + mppPrefillSource,
            header: mppHeader,
            ensureRowContiguous: true)
        mppPrefillCache[key] = k
        return k
    }

    private static func mppGQAS1Kernel(
        h: Int, gqa: Int, tq: Int, tk: Int, sliding: Int, relExtent: Int
    ) -> MLXFast.MLXFastKernel {
        let key = "h\(h)_g\(gqa)_q\(tq)_k\(tk)_s\(sliding)_r\(relExtent)"
        mppLock.lock()
        defer { mppLock.unlock() }
        if let k = mppGQACache[key] { return k }
        let consts = """
              constexpr int H = \(h);
              constexpr int GQA = \(gqa);
              constexpr int TQ = \(tq);
              constexpr int TK = \(tk);
              constexpr int SLIDING = \(sliding);
              constexpr int REL_EXTENT = \(relExtent);
            """
        let k = MLXFast.metalKernel(
            name: "inkling2_gqa_mpp_s1_\(key)",
            inputNames: ["q", "k", "v", "rel", "rel_proj_t", "params", "scale_buf"],
            outputNames: ["part_m", "part_l", "part_o"],
            source: consts + mppGQAS1Source,
            header: mppHeader,
            ensureRowContiguous: true)
        mppGQACache[key] = k
        return k
    }

    static func mppAvailable() -> Bool {
        if ProcessInfo.processInfo.environment["VMLX_INKLING2_MPP"] == "0" { return false }
        mppLock.lock()
        if let v = mppVerified {
            mppLock.unlock()
            return v
        }
        mppLock.unlock()
        let ok = DeepseekV41MPPIndexScores.available()
        mppLock.lock()
        mppVerified = ok
        mppLock.unlock()
        return ok
    }
    #endif

    static func affine8G64MultiGemv(
        _ x: MLXArray,
        weight: MLXArray,
        scales: MLXArray,
        biases: MLXArray,
        outDType: DType? = nil,
        softcap: Float = 0,
        outCols: Int = 0
    ) -> MLXArray? {
        #if canImport(Metal)
        guard ProcessInfo.processInfo.environment["VMLX_INKLING2_AFF8_MV"] != "0",
            Device.defaultDevice().deviceType == .gpu
        else { return nil }
        let lead = Array(x.shape.dropLast())
        let K = x.dim(-1)
        let N = weight.dim(0)
        let outN = (outCols > 0 && outCols <= N) ? outCols : N
        let T = x.size / K
        let od = outDType ?? x.dtype
        guard T >= 1, K % 256 == 0,
            [.bfloat16, .float16, .float32].contains(x.dtype),
            [.bfloat16, .float16, .float32].contains(scales.dtype),
            biases.dtype == scales.dtype,
            weight.dtype == .uint32
        else { return nil }
        let x2 = x.reshaped(T, K)
        if T > 8 {
            var parts: [MLXArray] = []
            var i = 0
            while i < T {
                let end = min(i + 8, T)
                guard let p = affine8G64MultiGemv(
                    x2[i ..< end], weight: weight, scales: scales, biases: biases,
                    outDType: od, softcap: softcap, outCols: outN)
                else { return nil }
                parts.append(p)
                i = end
            }
            return MLX.concatenated(parts, axis: 0).reshaped(lead + [outN])
        }
        let hasCap = softcap > 0
        let capBuf = MLXArray([hasCap ? softcap : Float(1.0)])
        let nTG = (outN + 7) / 8
        let y = affine8G64MultiGemvKernel(
            [x2, weight, scales, biases, capBuf],
            template: [
                ("XT", x2.dtype),
                ("ST", scales.dtype),
                ("OT", od),
                ("ROWS", T),
                ("K", K),
                ("OUT_N", outN),
                ("HAS_CAP", hasCap),
            ],
            grid: (nTG * 128, 1, 1),
            threadGroup: (128, 1, 1),
            outputShapes: [[T, outN]],
            outputDTypes: [od])[0]
        return y.reshaped(lead + [outN])
        #else
        return nil
        #endif
    }

    static func routerTopK(
        _ x: MLXArray, gateWeight: MLXArray, bias: MLXArray, topK: Int, routeScale: Float
    ) -> (indices: MLXArray, weights: MLXArray)? {
        #if canImport(Metal)
        guard ProcessInfo.processInfo.environment["VMLX_INKLING2_FUSED_ROUTER"] != "0",
            Device.defaultDevice().deviceType == .gpu
        else { return nil }
        let lead = Array(x.shape.dropLast())
        let K = x.dim(-1)
        let E = gateWeight.dim(0)
        let T = x.size / K
        guard E == 128, K == 2304, (1 ... 16).contains(T), (1 ... 16).contains(topK),
            [.bfloat16, .float16, .float32].contains(x.dtype),
            [.bfloat16, .float16, .float32].contains(gateWeight.dtype),
            [.bfloat16, .float16, .float32].contains(bias.dtype)
        else { return nil }
        let x2 = x.reshaped(T, K)
        let scores: MLXArray
        if T <= 8 {
            scores = routerScoresKernel(
                [x2, gateWeight],
                template: [
                    ("XT", x2.dtype), ("WT", gateWeight.dtype),
                    ("ROWS", T), ("E", E), ("K", K),
                ],
                grid: (E * 288, 1, 1),
                threadGroup: (288, 1, 1),
                outputShapes: [[T, E]],
                outputDTypes: [.float32])[0]
        } else {
            let sA = routerScoresKernel(
                [x2[..<8], gateWeight],
                template: [
                    ("XT", x2.dtype), ("WT", gateWeight.dtype),
                    ("ROWS", 8), ("E", E), ("K", K),
                ],
                grid: (E * 288, 1, 1),
                threadGroup: (288, 1, 1),
                outputShapes: [[8, E]],
                outputDTypes: [.float32])[0]
            let rem = T - 8
            let sB = routerScoresKernel(
                [x2[8...], gateWeight],
                template: [
                    ("XT", x2.dtype), ("WT", gateWeight.dtype),
                    ("ROWS", rem), ("E", E), ("K", K),
                ],
                grid: (E * 288, 1, 1),
                threadGroup: (288, 1, 1),
                outputShapes: [[rem, E]],
                outputDTypes: [.float32])[0]
            scores = MLX.concatenated([sA, sB], axis: 0)
        }
        let picked = routerTopKKernel(
            [scores, bias, MLXArray([routeScale])],
            template: [("TOPK", topK)],
            grid: (128, T, 1),
            threadGroup: (128, 1, 1),
            outputShapes: [[T, topK], [T, topK]],
            outputDTypes: [.uint32, .float32])
        return (picked[0].reshaped(lead + [topK]), picked[1].reshaped(lead + [topK]))
        #else
        return nil
        #endif
    }

    static func prefillAttentionMPP(
        q: MLXArray, k: MLXArray, v: MLXArray, r: MLXArray, projT: MLXArray, sinks: MLXArray,
        seen: Int, kStart: Int, window: Int, relExtent: Int, scale: Float
    ) -> MLXArray? {
        #if canImport(Metal)
        guard mppAvailable(),
            Device.defaultDevice().deviceType == .gpu,
            q.dtype == .bfloat16, k.dtype == .bfloat16, v.dtype == .bfloat16,
            r.dtype == .bfloat16, projT.dtype == .bfloat16,
            q.ndim == 4, q.dim(0) == 1, q.dim(3) == 128, r.dim(3) == 16
        else { return nil }
        let H = q.dim(1), LQ = q.dim(2)
        let Hkv = k.dim(1), S = k.dim(2)
        guard LQ >= 2, S >= 128, Hkv > 0, H % Hkv == 0 else { return nil }
        let gqa = H / Hkv
        let sc = MLXArray([scale])
        let qOff = MLXArray([Int32(seen - kStart)])
        let sinks2D = sinks.reshaped(LQ, H).asType(.bfloat16)
        let run = mppPrefillKernel(h: H, gqa: gqa, sliding: window, relExtent: relExtent)
        if LQ < 32 {
            let pad = 32 - LQ
            let qIn = MLX.padded(q, widths: [0, 0, .init((0, pad)), 0])
            let rIn = MLX.padded(r, widths: [0, .init((0, pad)), 0, 0])
            let skIn = MLX.padded(sinks2D, widths: [.init((0, pad)), 0])
            let out = run(
                [qIn, k, v, rIn, projT, skIn, sc, qOff],
                grid: (128, 1, H),
                threadGroup: (128, 1, 1),
                outputShapes: [[1, H, 32, 128]],
                outputDTypes: [.bfloat16])[0]
            return out[0..., 0..., ..<LQ, 0...]
        }
        let nQy = (LQ + 31) / 32
        return run(
            [q, k, v, r, projT, sinks2D, sc, qOff],
            grid: (128, nQy, H),
            threadGroup: (128, 1, 1),
            outputShapes: [[1, H, LQ, 128]],
            outputDTypes: [.bfloat16])[0]
        #else
        return nil
        #endif
    }

    static func gqaMPPAttention(
        q: MLXArray, k: MLXArray, v: MLXArray, r: MLXArray, projT: MLXArray, sinks: MLXArray,
        window: Int, relExtent: Int, scale: Float
    ) -> MLXArray? {
        #if canImport(Metal)
        guard mppAvailable(),
            Device.defaultDevice().deviceType == .gpu,
            q.dtype == .bfloat16, k.dtype == .bfloat16, v.dtype == .bfloat16,
            r.dtype == .bfloat16, projT.dtype == .bfloat16,
            q.ndim == 4, q.dim(0) == 1, q.dim(3) == 128, r.dim(3) == 16
        else { return nil }
        let H = q.dim(1), LQ = q.dim(2)
        let Hkv = k.dim(1), S = k.dim(2)
        guard Hkv > 0, H % Hkv == 0, LQ >= 1 else { return nil }
        let gqa = H / Hkv
        let M = gqa * LQ
        let tq: Int
        let tk: Int
        if M <= 32 {
            tq = 32; tk = 64
        } else if M <= 64 {
            tq = 64; tk = 32
        } else {
            return nil
        }
        guard S >= tk else { return nil }
        let kMin = window > 0 ? max(0, (S - LQ) - window + 1) : 0
        let span = S - kMin
        guard span >= 512 else { return nil }
        let nSplits = min(64, max(8, (span + 511) / 512))
        let params = MLXArray([Int32(nSplits), Int32(kMin)])
        let scBuf = MLXArray([scale])
        let s1 = mppGQAS1Kernel(
            h: H, gqa: gqa, tq: tq, tk: tk, sliding: window, relExtent: relExtent
        )(
            [q, k, v, r, projT, params, scBuf],
            grid: (128, nSplits, Hkv),
            threadGroup: (128, 1, 1),
            outputShapes: [[H, LQ, nSplits], [H, LQ, nSplits], [H, LQ, nSplits, 128]],
            outputDTypes: [.float32, .float32, .float32])
        return verifyAttnS2Kernel(
            [s1[0], s1[1], s1[2], sinks.reshaped(LQ, H).asType(.bfloat16)],
            grid: (32, LQ, H),
            threadGroup: (32, 1, 1),
            outputShapes: [[1, H, LQ, 128]],
            outputDTypes: [.bfloat16])[0]
        #else
        return nil
        #endif
    }

    static func verifyAttention(
        q: MLXArray, k: MLXArray, v: MLXArray, r: MLXArray, projT: MLXArray, sinks: MLXArray,
        window: Int, relExtent: Int, scale: Float
    ) -> MLXArray? {
        #if canImport(Metal)
        guard ProcessInfo.processInfo.environment["VMLX_INKLING2_VERIFY_ATTN"] != "0",
            Device.defaultDevice().deviceType == .gpu,
            q.dtype == .bfloat16, k.dtype == .bfloat16, v.dtype == .bfloat16,
            r.dtype == .bfloat16, projT.dtype == .bfloat16,
            q.ndim == 4, q.dim(0) == 1, q.dim(3) == 128, r.dim(3) == 16
        else { return nil }
        let H = q.dim(1), LQ = q.dim(2)
        let Hkv = k.dim(1), S = k.dim(2)
        guard (1 ... 16).contains(LQ), Hkv > 0, H % Hkv == 0, S >= LQ else { return nil }
        if LQ > 8 {
            let prefix = S - LQ
            let sinks2D = sinks.reshaped(LQ, H)
            guard let o1 = verifyAttention(
                q: q[0..., 0..., ..<8, 0...],
                k: k[0..., 0..., ..<(prefix + 8), 0...],
                v: v[0..., 0..., ..<(prefix + 8), 0...],
                r: r[0..., ..<8, 0..., 0...],
                projT: projT,
                sinks: sinks2D[..<8, 0...],
                window: window, relExtent: relExtent, scale: scale),
                let o2 = verifyAttention(
                    q: q[0..., 0..., 8..., 0...],
                    k: k, v: v,
                    r: r[0..., 8..., 0..., 0...],
                    projT: projT,
                    sinks: sinks2D[8..., 0...],
                    window: window, relExtent: relExtent, scale: scale)
            else { return nil }
            return MLX.concatenated([o1, o2], axis: 2)
        }
        // Prefer GQA-grouped M5 MPP Split-K when span >= 512 (reads each KV head ONCE for all GQA*LQ queries).
        if let mppOut = gqaMPPAttention(
            q: q, k: k, v: v, r: r, projT: projT, sinks: sinks,
            window: window, relExtent: relExtent, scale: scale)
        {
            return mppOut
        }
        let gqa = H / Hkv
        let kMin = window > 0 ? max(0, (S - LQ) - window + 1) : 0
        let span = S - kMin
        let nSplits: Int
        if span <= 128 {
            nSplits = 1
        } else if span <= 1024 {
            nSplits = 4
        } else {
            nSplits = min(64, max(8, (span + 511) / 512))
        }
        let sinks2D = sinks.reshaped(LQ, H).asType(.bfloat16)
        let params = MLXArray([Int32(nSplits), Int32(kMin)])
        let scBuf = MLXArray([scale])
        if nSplits == 1 {
            return verifyAttnS1SingleKernel(
                [q, k, v, r, projT, sinks2D, params, scBuf],
                template: [
                    ("H", H), ("GQA", gqa), ("SLIDING", window), ("REL_EXTENT", relExtent),
                ],
                grid: (256, 1, H),
                threadGroup: (256, 1, 1),
                outputShapes: [[1, H, LQ, 128]],
                outputDTypes: [.bfloat16])[0]
        }
        let s1 = verifyAttnS1MultiKernel(
            [q, k, v, r, projT, sinks2D, params, scBuf],
            template: [
                ("H", H), ("GQA", gqa), ("SLIDING", window), ("REL_EXTENT", relExtent),
            ],
            grid: (256, nSplits, H),
            threadGroup: (256, 1, 1),
            outputShapes: [[H, LQ, nSplits], [H, LQ, nSplits], [H, LQ, nSplits, 128]],
            outputDTypes: [.float32, .float32, .float32])
        return verifyAttnS2Kernel(
            [s1[0], s1[1], s1[2], sinks2D],
            grid: (32, LQ, H),
            threadGroup: (32, 1, 1),
            outputShapes: [[1, H, LQ, 128]],
            outputDTypes: [.bfloat16])[0]
        #else
        return nil
        #endif
    }
}
