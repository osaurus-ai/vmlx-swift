// Copyright © 2026 Jinho Jang. All rights reserved.
//
// Inkling-2-Mini single-token decode path.
//
// On this MLX stack every dependent op costs ~8 us of encode + dispatch regardless of its size
// (measured: a chain of `x + 1` on a 2304-vector), and a looped token runs 48 layer executions.
// The general forward is ~110 ops per layer execution; this path is ~25: each layer is a handful
// of GEMVs / gathers (which MLX does well) joined by four fused kernels:
//
//   depth      block attention residual (+ the layer's input RMSNorm)           -> bf16 inputs
//   qkv        k / v short convs, q / r / k clips, v norm, log scaling, decode relative bias
//   mid        post-attention conv + residual + MLP depth attention (+ RMSNorm)
//   final      expert combine + post-MLP conv + residual + the packed conv-state write
//
// plus, in loop passes 2..3, a gate-input / gate-output kernel pair around two batched GEMVs
// for the q/k/v message gates (and once more for the MLP gate).
//
// Arithmetic follows the reference's rounding points (see Inkling2Loop.swift); the general
// path remains the reference implementation and the prefill path.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

private let f32 = DType.float32

enum Inkling2DecodeKernels {

    static let header = """
        template <typename T> inline void ink_store(device T* p, size_t i, float v) { p[i] = static_cast<T>(v); }
        inline float ink_bf16(float v) { return float(bfloat16_t(v)); }
        template <typename T> inline float ink_round_as(device T* p, float v) { return float(static_cast<T>(v)); }
        """

    // MARK: depth attention, registers sized at compile time

    /// Template: NS sources (S stack rows + optional partial, HASP), NQ queries, NORM, ELEMS per
    /// thread, TG threads (one group per row). Inputs: stack [R, S, H] f32, partial [R, H],
    /// query [NQ, H] f32, nw [NQ, H] f32 (norm weights), ne [1] (norm eps).
    /// Output: [R, NQ, H] — fp32 depth output, or (NORM) the weighted-RMS-normalized row in the
    /// output dtype.
    static let depth = MLXFast.metalKernel(
        name: "inkling2_dec_depth",
        inputNames: ["stack", "partial", "query", "nw", "ne"],
        outputNames: ["out"],
        source: """
            uint r = threadgroup_position_in_grid.x;
            uint tid = thread_position_in_threadgroup.x;
            uint lane = thread_index_in_simdgroup;
            uint sg = simdgroup_index_in_threadgroup;
            constexpr int NSG = TG / 32;
            constexpr int S = NS - HASP;
            int H = query_shape[1];
            float v[NS][ELEMS];
            float ss[NS];
            float dq[NS][NQ];
            for (int s = 0; s < NS; ++s) { ss[s] = 0.f; for (int q = 0; q < NQ; ++q) dq[s][q] = 0.f; }
            for (int e = 0; e < ELEMS; ++e) {
                int j = tid + e * TG;
                bool ok = j < H;
                float qv[NQ];
                for (int q = 0; q < NQ; ++q) qv[q] = ok ? query[q * H + j] : 0.f;
                for (int s = 0; s < NS; ++s) {
                    float x = 0.f;
                    if (ok) x = s < S ? stack[((size_t)r * S + s) * H + j] : float(partial[(size_t)r * H + j]);
                    v[s][e] = x;
                    ss[s] += x * x;
                    for (int q = 0; q < NQ; ++q) dq[s][q] += x * qv[q];
                }
            }
            threadgroup float red[NSG][NS * (NQ + 1)];
            for (int s = 0; s < NS; ++s) {
                float a = simd_sum(ss[s]);
                if (lane == 0) red[sg][s * (NQ + 1)] = a;
                for (int q = 0; q < NQ; ++q) {
                    float d = simd_sum(dq[s][q]);
                    if (lane == 0) red[sg][s * (NQ + 1) + 1 + q] = d;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            threadgroup float inv[NS];
            threadgroup float p[NS][NQ];
            if (tid == 0) {
                float scale = rsqrt(float(H));
                for (int s = 0; s < NS; ++s) {
                    float a = 0.f;
                    for (int g = 0; g < NSG; ++g) a += red[g][s * (NQ + 1)];
                    inv[s] = rsqrt(a / float(H) + 1e-6f);
                }
                for (int q = 0; q < NQ; ++q) {
                    float m = -INFINITY;
                    float sc[NS];
                    for (int s = 0; s < NS; ++s) {
                        float d = 0.f;
                        for (int g = 0; g < NSG; ++g) d += red[g][s * (NQ + 1) + 1 + q];
                        sc[s] = d * inv[s] * scale;
                        m = max(m, sc[s]);
                    }
                    float z = 0.f;
                    for (int s = 0; s < NS; ++s) { sc[s] = exp(sc[s] - m); z += sc[s]; }
                    for (int s = 0; s < NS; ++s) p[s][q] = sc[s] / z;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float acc[NQ][ELEMS];
            float sq[NQ];
            for (int q = 0; q < NQ; ++q) {
                sq[q] = 0.f;
                for (int e = 0; e < ELEMS; ++e) {
                    float a = 0.f;
                    for (int s = 0; s < NS; ++s) a += p[s][q] * (v[s][e] * inv[s]);
                    acc[q][e] = a;
                    sq[q] += a * a;
                }
            }
            threadgroup float red2[NSG][NQ];
            threadgroup float nrm[NQ];
            if (NORM) {
                for (int q = 0; q < NQ; ++q) { float a = simd_sum(sq[q]); if (lane == 0) red2[sg][q] = a; }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (tid < (uint)NQ) {
                    float a = 0.f;
                    for (int g = 0; g < NSG; ++g) a += red2[g][tid];
                    nrm[tid] = rsqrt(a / float(H) + ne[0]);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (int q = 0; q < NQ; ++q)
                    for (int e = 0; e < ELEMS; ++e) {
                        int j = tid + e * TG;
                        if (j < H) ink_store(out, ((size_t)r * NQ + q) * H + j, (acc[q][e] * nrm[q]) * nw[q * H + j]);
                    }
            } else {
                for (int q = 0; q < NQ; ++q)
                    for (int e = 0; e < ELEMS; ++e) {
                        int j = tid + e * TG;
                        if (j < H) ink_store(out, ((size_t)r * NQ + q) * H + j, acc[q][e]);
                    }
            }
            """,
        header: header)

    // MARK: q / k / v preparation

    /// One simdgroup (= threadgroup) per head vector: NQ q heads, NQ r heads, NKV k heads, NKV v
    /// heads. q / r come from `qa` at offsets imeta[0] / imeta[1]; k (+ dyn conv weights) from
    /// `ka` at imeta[2] (/ imeta[3]); v from `va` at imeta[4] (/ imeta[5]). imeta[6] = S visible
    /// keys. fmeta = [qk clip, rel clip, v-norm eps, tau]. k / v run their short conv (static, or
    /// DYN dynamic) against the packed state (k columns 0.., v columns C..), are rounded to bf16,
    /// then clipped / normalized. q / r are clipped, rounded, and (tau != 1) log-scaled and rounded
    /// again, as the reference does. r heads also write the decode bias row for their head.
    /// Template: NQ, NKV, D, DR, DYN.
    static let qkv = MLXFast.metalKernel(
        name: "inkling2_dec_qkv",
        inputNames: ["qa", "ka", "va", "kw", "vw", "packed", "qscale", "rscale", "proj", "fmeta", "imeta"],
        outputNames: ["qo", "ko", "vo", "bias", "kvst"],
        source: """
            uint head = threadgroup_position_in_grid.x;
            uint lane = thread_index_in_simdgroup;
            constexpr int C = NKV * D;
            constexpr int EL = D / 32;
            int P = packed_shape[1];
            int S = imeta[6];
            int E = proj_shape[1];
            float qkc = fmeta[0], rc = fmeta[1], eps = fmeta[2], tau = fmeta[3];
            threadgroup float rr[DR];
            if (head < (uint)NQ) {
                int h = head;
                float x[EL]; float ss = 0.f;
                for (int e = 0; e < EL; ++e) {
                    int j = lane + 32 * e;
                    x[e] = float(qa[imeta[0] + h * D + j]) * qscale[h * D + j];
                    ss += x[e] * x[e];
                }
                ss = simd_sum(ss);
                float f = min(qkc / (sqrt(ss) + 1e-6f), 1.f);
                for (int e = 0; e < EL; ++e) {
                    float y = ink_round_as(qo, x[e] * f);
                    if (tau != 1.f) y = ink_round_as(qo, y * tau);
                    ink_store(qo, h * D + lane + 32 * e, y);
                }
            } else if (head < (uint)(2 * NQ)) {
                int h = head - NQ;
                float x = 0.f;
                if (lane < (uint)DR) x = float(qa[imeta[1] + h * DR + lane]) * rscale[h * DR + lane];
                float ss = simd_sum(x * x);
                float f = min(rc / (sqrt(ss) + 1e-6f), 1.f);
                float y = ink_round_as(bias, x * f);
                if (tau != 1.f) y = ink_round_as(bias, y * tau);
                if (lane < (uint)DR) rr[lane] = y;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (int j = lane; j < S; j += 32) {
                    int d = S - 1 - j;
                    float b = 0.f;
                    if (d < E) {
                        for (int k = 0; k < DR; ++k) b += rr[k] * float(proj[k * E + d]);
                    }
                    ink_store(bias, (size_t)h * S + j, b);
                }
            } else {
                bool isK = head < (uint)(2 * NQ + NKV);
                int h = isK ? head - 2 * NQ : head - 2 * NQ - NKV;
                const device auto* src = isK ? ka : va;
                int off = isK ? imeta[2] : imeta[4];
                int doff = isK ? imeta[3] : imeta[5];
                const device float* w = isK ? kw : vw;
                int col0 = isK ? 0 : C;
                float y[EL]; float ss = 0.f;
                for (int e = 0; e < EL; ++e) {
                    int c = h * D + lane + 32 * e;
                    float xt = float(src[off + c]);
                    float s0 = packed[col0 + c], s1 = packed[P + col0 + c], s2 = packed[2 * P + col0 + c];
                    float o;
                    if (DYN) {
                        float d0 = float(src[doff + h * 4 + 0]), d1 = float(src[doff + h * 4 + 1]);
                        float d2 = float(src[doff + h * 4 + 2]), d3 = float(src[doff + h * 4 + 3]);
                        float a = xt * (w[c] + d0);
                        a = a + s2 * (w[C + c] + d1);
                        a = a + s1 * (w[2 * C + c] + d2);
                        a = a + s0 * (w[3 * C + c] + d3);
                        o = (a + xt) / 3.0f;
                    } else {
                        float a = s0 * w[c];
                        a = a + s1 * w[C + c];
                        a = a + s2 * w[2 * C + c];
                        a = a + xt * w[3 * C + c];
                        o = (a + xt) / 2.2360679774997896f;
                    }
                    y[e] = ink_round_as(ko, o);
                    ss += y[e] * y[e];
                    kvst[col0 + c] = s1; kvst[2 * C + col0 + c] = s2; kvst[4 * C + col0 + c] = xt;
                }
                ss = simd_sum(ss);
                float f = isK ? min(qkc / (sqrt(ss) + 1e-6f), 1.f) : rsqrt(ss / float(D) + eps);
                for (int e = 0; e < EL; ++e) {
                    int c = h * D + lane + 32 * e;
                    if (isK) ink_store(ko, c, y[e] * f); else ink_store(vo, c, y[e] * f);
                }
            }
            """,
        header: header)

    /// `attn * sigmoid(gate)` (fp32, rounded to attn's dtype); gate read from `qa` at meta[0].
    static let sigmoidGate = MLXFast.metalKernel(
        name: "inkling2_dec_sigmoid_gate",
        inputNames: ["attn", "qa", "meta"],
        outputNames: ["out"],
        source: """
            uint i = thread_position_in_grid.x;
            float g = float(qa[meta[0] + i]);
            ink_store(out, i, float(attn[i]) * (1.f / (1.f + exp(-g))));
            """,
        header: header)

    // MARK: mid: post-attention conv + residual + MLP depth attention (+ norm)

    /// Template NS (stack rows S + the residual), ELEMS, TG, NORM, BOUNDARY. Inputs: `a` [H] (o_proj
    /// output), `h` [H] (layer input), `packed` [3, P] f32, `cw` [4, H] f32, `stack` [S, H] f32 (with
    /// the layer input appended at a boundary), `query` [1, H], `nw` [H], `ne` [1], meta [conv
    /// column offset]. Outputs: `xm` [H] (normalized MLP input in the model dtype, NORM),
    /// `xm32` [H] f32 (== fp32(xm) with NORM; the raw depth output without), `res` [H] f32,
    /// `ast` [3, H] f32 (new post-attention conv state).
    static let mid = MLXFast.metalKernel(
        name: "inkling2_dec_mid",
        inputNames: ["a", "h", "packed", "cw", "stack", "query", "nw", "ne", "meta"],
        outputNames: ["xm", "xm32", "res", "ast"],
        source: """
            uint tid = thread_position_in_threadgroup.x;
            uint lane = thread_index_in_simdgroup;
            uint sg = simdgroup_index_in_threadgroup;
            constexpr int NSG = TG / 32;
            constexpr int S = NS - 1;
            int H = query_shape[1];
            int P = packed_shape[1];
            int off = meta[0];
            float v[NS][ELEMS];
            float ss[NS];
            float dq[NS];
            for (int s = 0; s < NS; ++s) { ss[s] = 0.f; dq[s] = 0.f; }
            for (int e = 0; e < ELEMS; ++e) {
                int j = tid + e * TG;
                bool ok = j < H;
                float rv = 0.f;
                if (ok) {
                    float xt = float(a[j]);
                    float s0 = packed[off + j], s1 = packed[P + off + j], s2 = packed[2 * P + off + j];
                    float y = s0 * cw[j];
                    y = y + s1 * cw[H + j];
                    y = y + s2 * cw[2 * H + j];
                    y = y + xt * cw[3 * H + j];
                    float av = ink_round_as(xm, (y + xt) / 2.2360679774997896f);
                    rv = BOUNDARY ? av : float(h[j]) + av;
                    res[j] = rv;
                    ast[j] = s1; ast[H + j] = s2; ast[2 * H + j] = xt;
                }
                float qv = ok ? query[j] : 0.f;
                for (int s = 0; s < NS; ++s) {
                    float x = 0.f;
                    if (ok) x = s < S ? stack[(size_t)s * H + j] : rv;
                    v[s][e] = x;
                    ss[s] += x * x;
                    dq[s] += x * qv;
                }
            }
            threadgroup float red[NSG][NS * 2];
            for (int s = 0; s < NS; ++s) {
                float a0 = simd_sum(ss[s]);
                float d0 = simd_sum(dq[s]);
                if (lane == 0) { red[sg][2 * s] = a0; red[sg][2 * s + 1] = d0; }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            threadgroup float inv[NS];
            threadgroup float p[NS];
            if (tid == 0) {
                float scale = rsqrt(float(H));
                float sc[NS];
                float m = -INFINITY;
                for (int s = 0; s < NS; ++s) {
                    float a0 = 0.f, d0 = 0.f;
                    for (int g = 0; g < NSG; ++g) { a0 += red[g][2 * s]; d0 += red[g][2 * s + 1]; }
                    inv[s] = rsqrt(a0 / float(H) + 1e-6f);
                    sc[s] = d0 * inv[s] * scale;
                    m = max(m, sc[s]);
                }
                float z = 0.f;
                for (int s = 0; s < NS; ++s) { sc[s] = exp(sc[s] - m); z += sc[s]; }
                for (int s = 0; s < NS; ++s) p[s] = sc[s] / z;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float acc[ELEMS];
            float sq = 0.f;
            for (int e = 0; e < ELEMS; ++e) {
                float a0 = 0.f;
                for (int s = 0; s < NS; ++s) a0 += p[s] * (v[s][e] * inv[s]);
                acc[e] = a0;
                sq += a0 * a0;
            }
            threadgroup float red2[NSG];
            threadgroup float nrm;
            if (NORM) {
                float a0 = simd_sum(sq);
                if (lane == 0) red2[sg] = a0;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (tid == 0) {
                    float t = 0.f;
                    for (int g = 0; g < NSG; ++g) t += red2[g];
                    nrm = rsqrt(t / float(H) + ne[0]);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (int e = 0; e < ELEMS; ++e) {
                    int j = tid + e * TG;
                    if (j < H) {
                        float o = (acc[e] * nrm) * nw[j];
                        ink_store(xm, j, o);
                        xm32[j] = float(xm[j]);
                    }
                }
            } else {
                for (int e = 0; e < ELEMS; ++e) {
                    int j = tid + e * TG;
                    if (j < H) xm32[j] = acc[e];
                }
            }
            """,
        header: header)

    // MARK: final: expert combine + post-MLP conv + residual + packed state

    /// Grid = P threads. Columns [0, 2C) copy `kvst`, [2C, 2C+H) copy `ast`, [2C+H, P) compute
    /// `m_j = dtype(sum_k down[k, j] * es[inds[k]] * w[k])`, the post-MLP conv on it, and
    /// `hout_j = dtype(res_j + fp32(dtype(conv)))`. meta = [2C, H].
    static let final = MLXFast.metalKernel(
        name: "inkling2_dec_final",
        inputNames: ["down", "inds", "es", "w", "packed", "cw", "res", "kvst", "ast", "meta"],
        outputNames: ["hout", "np"],
        source: """
            uint c = thread_position_in_grid.x;
            int P = packed_shape[1];
            int C2 = meta[0];
            int H = meta[1];
            int K = inds_shape[1];
            if (c >= (uint)P) return;
            if ((int)c < C2) {
                np[c] = kvst[c]; np[P + c] = kvst[C2 + c]; np[2 * P + c] = kvst[2 * C2 + c];
                return;
            }
            if ((int)c < C2 + H) {
                int j = c - C2;
                np[c] = ast[j]; np[P + c] = ast[H + j]; np[2 * P + c] = ast[2 * H + j];
                return;
            }
            int j = c - C2 - H;
            float acc = 0.f;
            for (int k = 0; k < K; ++k) acc += (down[(size_t)k * H + j] * es[inds[k]]) * w[k];
            float xt = ink_round_as(hout, acc);
            float s0 = packed[c], s1 = packed[P + c], s2 = packed[2 * P + c];
            float y = s0 * cw[j];
            y = y + s1 * cw[H + j];
            y = y + s2 * cw[2 * H + j];
            y = y + xt * cw[3 * H + j];
            float av = ink_round_as(hout, (y + xt) / 2.2360679774997896f);
            ink_store(hout, j, res[j] + av);
            np[c] = s1; np[P + c] = s2; np[2 * P + c] = xt;
            """,
        header: header)

    // MARK: loop message gates

    /// G gate rows: base row `x[NX == 1 ? 0 : g]` (f32), message row `msgs[mrow + g]` (f32).
    /// Out [G, 2H] = [rms(base), rms(msg)] in the down-projection dtype. One group per row.
    /// meta = [mrow, NX]; eps [1]. Template ELEMS, TG.
    static let gateIn = MLXFast.metalKernel(
        name: "inkling2_dec_gate_in",
        inputNames: ["x", "msgs", "eps", "meta"],
        outputNames: ["out"],
        source: """
            uint g = threadgroup_position_in_grid.x;
            uint tid = thread_position_in_threadgroup.x;
            uint lane = thread_index_in_simdgroup;
            uint sg = simdgroup_index_in_threadgroup;
            constexpr int NSG = TG / 32;
            int H = x_shape[1];
            int xr = meta[1] == 1 ? 0 : g;
            int mr = meta[0] + g;
            float hv[ELEMS], mv[ELEMS];
            float a = 0.f, b = 0.f;
            for (int e = 0; e < ELEMS; ++e) {
                int j = tid + e * TG;
                hv[e] = j < H ? x[(size_t)xr * H + j] : 0.f;
                mv[e] = j < H ? msgs[(size_t)mr * H + j] : 0.f;
                a += hv[e] * hv[e]; b += mv[e] * mv[e];
            }
            threadgroup float ra[NSG], rb[NSG];
            a = simd_sum(a); b = simd_sum(b);
            if (lane == 0) { ra[sg] = a; rb[sg] = b; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float A = 0.f, B = 0.f;
            for (int k = 0; k < NSG; ++k) { A += ra[k]; B += rb[k]; }
            float ia = rsqrt(A / float(H) + eps[0]);
            float ib = rsqrt(B / float(H) + eps[0]);
            for (int e = 0; e < ELEMS; ++e) {
                int j = tid + e * TG;
                if (j < H) {
                    ink_store(out, (size_t)g * 2 * H + j, hv[e] * ia);
                    ink_store(out, (size_t)g * 2 * H + H + j, mv[e] * ib);
                }
            }
            """,
        header: header)

    /// G rows: `y = (1 - tanh(fg[g, j])) * base + tanh(fg[g, H + j]) * msg` (fp32), then the
    /// layer's weighted RMSNorm `nw[g]` (eps `ne`), written in the model dtype (`out`) and as
    /// fp32 of that (`out32`). Same base / message row mapping as `gateIn`. Template ELEMS, TG.
    static let gateOut = MLXFast.metalKernel(
        name: "inkling2_dec_gate_out",
        inputNames: ["fg", "x", "msgs", "nw", "ne", "meta"],
        outputNames: ["out", "out32"],
        source: """
            uint g = threadgroup_position_in_grid.x;
            uint tid = thread_position_in_threadgroup.x;
            uint lane = thread_index_in_simdgroup;
            uint sg = simdgroup_index_in_threadgroup;
            constexpr int NSG = TG / 32;
            int H = x_shape[1];
            int xr = meta[1] == 1 ? 0 : g;
            int mr = meta[0] + g;
            float y[ELEMS];
            float sq = 0.f;
            for (int e = 0; e < ELEMS; ++e) {
                int j = tid + e * TG;
                float v = 0.f;
                if (j < H) {
                    float f = tanh(fg[(size_t)g * 2 * H + j]);
                    float a = tanh(fg[(size_t)g * 2 * H + H + j]);
                    v = (1.f - f) * x[(size_t)xr * H + j] + a * msgs[(size_t)mr * H + j];
                }
                y[e] = v;
                sq += v * v;
            }
            threadgroup float red[NSG];
            sq = simd_sum(sq);
            if (lane == 0) red[sg] = sq;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float t = 0.f;
            for (int k = 0; k < NSG; ++k) t += red[k];
            float n = rsqrt(t / float(H) + ne[0]);
            for (int e = 0; e < ELEMS; ++e) {
                int j = tid + e * TG;
                if (j < H) {
                    ink_store(out, (size_t)g * H + j, (y[e] * n) * nw[(size_t)g * H + j]);
                    out32[(size_t)g * H + j] = float(out[(size_t)g * H + j]);
                }
            }
            """,
        header: header)
}

/// Per-layer weights re-laid-out for the decode path (built on first use, outside the module's
/// parameter tree).
final class Inkling2DecodeWeights: @unchecked Sendable {
    /// x-fed fused projection `[q | r | gate | sink | k | v]` (sliding, no messages).
    var xFused: MLXArray?
    /// `[q | r | gate | sink]`, `[k (| k dyn)]`, `[v (| v dyn)]`.
    var qFused: MLXArray?
    var kFused: MLXArray?
    var vFused: MLXArray?
    /// Stacked norm weights for the depth kernel ([1|3, H] fp32) and the MLP norm [H].
    var attnNorm: MLXArray?
    var gateNorm: MLXArray?
    var normEps: MLXArray?
    /// q / k / v message gates: down `[3, 2H, rank]` (transposed views), [Wf; Wg]^T `[3, rank, 2H]`.
    var qkvDown: MLXArray?
    var qkvFG: MLXArray?
    var mlpDown: MLXArray?
    var mlpFG: MLXArray?
    var gateEps: MLXArray?
    var routerT: MLXArray?
    var ones: MLXArray?
    var downOnes: MLXArray?
    var ready = false
}

extension Inkling2DecoderLayer {

    /// True when this layer can run `decodeStep` (plain-weight projections, standard sizes).
    func supportsDecodeStep() -> Bool {
        let a = selfAttn
        let plain: [Linear] = [a.qProj, a.rProj, a.gateProj, a.sinkProj, a.kProj, a.vProj]
        guard plain.allSatisfy({ type(of: $0) == Linear.self && $0.bias == nil }) else { return false }
        if let gates = [qGate, kGate, vGate, mlpGate] as? [Inkling2MessageGate] {
            for g in gates {
                guard type(of: g.downProj) == Linear.self, type(of: g.forgetProj) == Linear.self,
                    type(of: g.gainProj) == Linear.self
                else { return false }
            }
        }
        // Kernel scratch bounds: router selection holds <= 256 experts and <= 16 picks; the
        // depth kernels size their source arrays from the block-state count.
        let router = mlp.gate
        return a.hd % 32 == 0 && a.hd <= 128 && a.dRel <= 32 && a.nq <= 64
            && router.weight.dim(0) <= 256 && router.topK <= 16 && router.topK >= 1
    }

    private func prepareDecodeWeights() {
        let w = decodeWeights
        guard !w.ready else { return }
        let a = selfAttn
        let qParts = [a.qProj.weight, a.rProj.weight, a.gateProj.weight, a.sinkProj.weight]
        if isSliding {
            w.xFused = MLX.concatenated(qParts + [a.kProj.weight, a.vProj.weight], axis: 0)
            w.kFused = a.kProj.weight
            w.vFused = a.vProj.weight
        } else {
            let kd = (a.kSconv as! Inkling2DynamicShortConv).weightProj.weight
            let vd = (a.vSconv as! Inkling2DynamicShortConv).weightProj.weight
            w.kFused = MLX.concatenated([a.kProj.weight, kd.asType(a.kProj.weight.dtype)], axis: 0)
            w.vFused = MLX.concatenated([a.vProj.weight, vd.asType(a.vProj.weight.dtype)], axis: 0)
        }
        w.qFused = MLX.concatenated(qParts, axis: 0)
        let norms: [Inkling2RMSNorm] = isSliding ? [inputLN!] : [qInputLN!, kInputLN!, vInputLN!]
        w.attnNorm = MLX.stacked(norms.map { $0.weight.asType(f32) }, axis: 0)
        // Gate rows always come as q, k, v: sliding layers use input_layernorm for all three.
        w.gateNorm = isSliding
            ? MLX.stacked([inputLN!, inputLN!, inputLN!].map { $0.weight.asType(f32) }, axis: 0)
            : w.attnNorm
        w.normEps = MLXArray([postAttnLN.eps])
        if isLoopLayer, let qg = qGate, let kg = kGate, let vg = vGate, let mg = mlpGate {
            func fg(_ g: Inkling2MessageGate) -> MLXArray {
                MLX.concatenated([g.forgetProj.weight.asType(f32), g.gainProj.weight.asType(f32)], axis: 0)
                    .T  // [rank, 2H]
            }
            w.qkvDown = MLX.contiguous(
                MLX.stacked([qg, kg, vg].map { $0.downProj.weight }, axis: 0).transposed(0, 2, 1))
            w.qkvFG = MLX.contiguous(MLX.stacked([qg, kg, vg].map(fg), axis: 0))
            w.mlpDown = MLX.contiguous(mg.downProj.weight.T.expandedDimensions(axis: 0))
            w.mlpFG = MLX.contiguous(fg(mg).expandedDimensions(axis: 0))
            w.gateEps = MLXArray([qg.eps])
        }
        w.routerT = mlp.gate.weight.asType(f32).T
        w.ones = MLXArray.ones([mlp.switchMLP.gateUp.expertScale.dim(0)], dtype: f32)
        w.downOnes = w.ones
        eval(
            [w.xFused, w.qFused, w.kFused, w.vFused, w.attnNorm, w.gateNorm, w.qkvDown, w.qkvFG,
             w.mlpDown, w.mlpFG, w.routerT, w.ones].compactMap { $0 })
        w.ready = true
    }

    /// One token (B = 1, T = 1). `stack` `[1, 1, S, H]` fp32; `messages` `[1, 1, 4 * loopLayers, H]`
    /// with this layer's rows at `4 * loopIndex`. Returns `(h, stack')`.
    func decodeStep(
        _ h: MLXArray, stack: MLXArray?, kv: KVCacheSimple, conv: MambaCache,
        messages: MLXArray?, loopIndex: Int, ones: MLXArray
    ) -> (MLXArray, MLXArray?) {
        prepareDecodeWeights()
        let w = decodeWeights
        let a = selfAttn
        let H = hidden
        let dtype = h.dtype
        let (nq, nkv, hd, dRel) = (a.nq, a.nkv, a.hd, a.dRel)
        let C = nkv * hd
        let P = 2 * C + 2 * H
        let packed = (conv[0] ?? MLXArray.zeros([1, 3, P], dtype: f32)).reshaped([3, P])
        let h1 = h.reshaped([1, H])
        let tg = 768
        let elems = (H + tg - 1) / tg
        let S0 = stack?.dim(2) ?? 0
        let st0 = stack?.reshaped([S0, H]) ?? MLXArray.zeros([1, H], dtype: f32)
        let msgs = messages?.reshaped([-1, H])
        let nqa = isSliding ? 1 : 3

        // 1. attention inputs: rows [1 | 3, H] in the model dtype
        var rows: MLXArray
        if attentionInput == nil {
            rows = inputLN!(h1)  // layer 0: input_layernorm(h)
        } else if msgs == nil {
            rows = Inkling2DecodeKernels.depth(
                [st0, h1, attentionInput!.query.asType(f32).reshaped([nqa, H]), w.attnNorm!, w.normEps!],
                template: [("NS", S0 + 1), ("HASP", 1), ("NQ", nqa), ("NORM", 1), ("ELEMS", elems), ("TG", tg)],
                grid: (tg, 1, 1), threadGroup: (tg, 1, 1), outputShapes: [[nqa, H]],
                outputDTypes: [dtype])[0]
        } else {
            let xr = Inkling2DecodeKernels.depth(
                [st0, h1, attentionInput!.query.asType(f32).reshaped([nqa, H]), w.attnNorm!, w.normEps!],
                template: [("NS", S0 + 1), ("HASP", 1), ("NQ", nqa), ("NORM", 0), ("ELEMS", elems), ("TG", tg)],
                grid: (tg, 1, 1), threadGroup: (tg, 1, 1), outputShapes: [[nqa, H]],
                outputDTypes: [f32])[0]
            let meta = MLXArray([Int32(4 * loopIndex), Int32(nqa)])
            let gin = Inkling2DecodeKernels.gateIn(
                [xr, msgs!, w.gateEps!, meta], template: [("ELEMS", elems), ("TG", tg)],
                grid: (3 * tg, 1, 1), threadGroup: (tg, 1, 1), outputShapes: [[3, 2 * H]],
                outputDTypes: [w.qkvDown!.dtype])[0]
            let d = MLX.matmul(gin.reshaped([3, 1, 2 * H]), w.qkvDown!).asType(f32)  // [3, 1, rank]
            let fg = MLX.matmul(d, w.qkvFG!).reshaped([3, 2 * H])
            rows = Inkling2DecodeKernels.gateOut(
                [fg, xr, msgs!, w.gateNorm!, w.normEps!, meta], template: [("ELEMS", elems), ("TG", tg)],
                grid: (3 * tg, 1, 1), threadGroup: (tg, 1, 1), outputShapes: [[3, H], [3, H]],
                outputDTypes: [dtype, f32])[0]
        }

        // 2. projections
        let nQ = nq * hd
        let nR = nq * dRel
        let qa: MLXArray
        let ka: MLXArray
        let va: MLXArray
        var imeta: [Int32]
        if rows.dim(0) == 1 {
            let y = MLX.matmul(rows, w.xFused!.T).reshaped([-1])
            qa = y; ka = y; va = y
            let base = 2 * nQ + nR + nq
            imeta = [0, Int32(nQ), Int32(base), 0, Int32(base + C), 0]
        } else {
            qa = MLX.matmul(rows[0 ..< 1], w.qFused!.T).reshaped([-1])
            ka = MLX.matmul(rows[1 ..< 2], w.kFused!.T).reshaped([-1])
            va = MLX.matmul(rows[2 ..< 3], w.vFused!.T).reshaped([-1])
            imeta = [0, Int32(nQ), 0, Int32(C), 0, Int32(C)]
        }
        let seen = kv.offset
        let S = a.window.map { min(seen + 1, $0) } ?? (seen + 1)
        imeta.append(Int32(S))
        var tau: Float = 1
        if !a.isSliding, Float(seen + 1) > a.logFloor {
            tau = 1 + a.logAlpha * Foundation.log(max(Float(seen + 1) / a.logFloor, 1))
        }
        let kwStatic: MLXArray
        let vwStatic: MLXArray
        if let ks = a.kSconv as? Inkling2ShortConv {
            kwStatic = ks.weight
            vwStatic = (a.vSconv as! Inkling2ShortConv).weight
        } else {
            kwStatic = (a.kSconv as! Inkling2DynamicShortConv).staticWeight
            vwStatic = (a.vSconv as! Inkling2DynamicShortConv).staticWeight
        }
        let outs = Inkling2DecodeKernels.qkv(
            [
                qa, ka, va, kwStatic, vwStatic, packed, a.qL2Clip.scale, a.relL2Clip.scale,
                a.relLogits.proj, MLXArray([a.qkClip, a.relClip, a.eps, tau]), MLXArray(imeta),
            ],
            template: [("NQ", nq), ("NKV", nkv), ("D", hd), ("DR", dRel), ("DYN", a.isSliding ? 0 : 1)],
            grid: ((2 * nq + 2 * nkv) * 32, 1, 1), threadGroup: (32, 1, 1),
            outputShapes: [[nQ], [C], [C], [nq * S], [3, 2 * C]],
            outputDTypes: [dtype, dtype, dtype, dtype, f32])

        // 3. attention
        var (keys, values) = kv.update(
            keys: outs[1].reshaped([1, nkv, 1, hd]), values: outs[2].reshaped([1, nkv, 1, hd]))
        let total = keys.dim(2)
        if total > S {
            keys = keys[0..., 0..., (total - S)..., 0...]
            values = values[0..., 0..., (total - S)..., 0...]
        }
        let sinkOff = 2 * nQ + nR
        let attn = MLXFast.scaledDotProductAttention(
            queries: outs[0].reshaped([1, nq, 1, hd]), keys: keys, values: values, scale: a.scale,
            mask: .array(outs[3].reshaped([1, nq, 1, S])),
            sinks: qa[sinkOff ..< (sinkOff + nq)].asType(dtype))
        let gated = Inkling2DecodeKernels.sigmoidGate(
            [attn.reshaped([nQ]), qa, MLXArray([Int32(nQ + nR)])], grid: (nQ, 1, 1),
            threadGroup: (256, 1, 1), outputShapes: [[nQ]], outputDTypes: [dtype])[0]
        let o = a.oProj(gated.reshaped([1, nQ]))

        // 4. residual + MLP input
        let states = isBlockBoundary ? inklingAppendState(stack, h) : stack
        let S1 = states?.dim(2) ?? 0
        let st1 = states?.reshaped([S1, H]) ?? MLXArray.zeros([1, H], dtype: f32)
        let norm = msgs == nil
        let midOut = Inkling2DecodeKernels.mid(
            [o.reshaped([H]), h1.reshaped([H]), packed, postAttnConv.weight, st1,
             mlpInput.query.reshaped([1, H]), postAttnLN.weight.asType(f32), w.normEps!,
             MLXArray([Int32(2 * C)])],
            template: [("NS", S1 + 1), ("ELEMS", elems), ("TG", tg), ("NORM", norm ? 1 : 0),
                       ("BOUNDARY", isBlockBoundary ? 1 : 0)],
            grid: (tg, 1, 1), threadGroup: (tg, 1, 1),
            outputShapes: [[H], [H], [H], [3, H]], outputDTypes: [dtype, f32, f32, f32])
        var xm32 = midOut[1]
        if let msgs {
            let meta = MLXArray([Int32(4 * loopIndex + 3), Int32(1)])
            let mrow = xm32.reshaped([1, H])
            let gin = Inkling2DecodeKernels.gateIn(
                [mrow, msgs, w.gateEps!, meta], template: [("ELEMS", elems), ("TG", tg)],
                grid: (tg, 1, 1), threadGroup: (tg, 1, 1), outputShapes: [[1, 2 * H]],
                outputDTypes: [w.mlpDown!.dtype])[0]
            let d = MLX.matmul(gin.reshaped([1, 1, 2 * H]), w.mlpDown!).asType(f32)
            let fg = MLX.matmul(d, w.mlpFG!).reshaped([1, 2 * H])
            xm32 = Inkling2DecodeKernels.gateOut(
                [fg, mrow, msgs, postAttnLN.weight.asType(f32).reshaped([1, H]), w.normEps!, meta],
                template: [("ELEMS", elems), ("TG", tg)], grid: (tg, 1, 1), threadGroup: (tg, 1, 1),
                outputShapes: [[1, H], [1, H]], outputDTypes: [dtype, f32])[1]
        }

        // 5. MoE
        let x32 = xm32.reshaped([1, H])
        let router = mlp.gate
        let logits = MLX.matmul(x32, w.routerT!)
        let sel = Inkling2Kernels.routerSelect(
            [logits, router.bias, router.meta, router.scaleArray], grid: (128, 1, 1),
            threadGroup: (128, 1, 1), outputShapes: [[1, router.topK], [1, router.topK]],
            outputDTypes: [.uint32, f32])
        let inds = sel[0]
        let sw = mlp.switchMLP
        let k = router.topK
        let gu = sw.gateUp(x32.reshaped([1, 1, 1, H]), inds, sorted: false)  // [1, k, 1, 2I]
        let act = Inkling2Kernels.moeActivation(
            [gu.reshaped([k, 2 * sw.inter]), inds.reshaped([k]),
             sw.gateUp.isNVFP4 ? sw.gateUp.expertScale : w.ones!, sw.caps],
            grid: (sw.inter, k, 1), threadGroup: (min(256, sw.inter), 1, 1),
            outputShapes: [[k, sw.inter]], outputDTypes: [f32])[0]
        let down = sw.down(act.reshaped([1, k, 1, sw.inter]), inds, sorted: false)  // [1, k, 1, H]
        let fin = Inkling2DecodeKernels.final(
            [down.reshaped([k, H]), inds, sw.down.isNVFP4 ? sw.down.expertScale : w.ones!, sel[1],
             packed, postMLPConv.weight, midOut[2], outs[4], midOut[3],
             MLXArray([Int32(2 * C), Int32(H)])],
            grid: (P, 1, 1), threadGroup: (256, 1, 1), outputShapes: [[H], [3, P]],
            outputDTypes: [dtype, f32])
        conv[0] = fin[1].reshaped([1, 3, P])
        conv.offset += 1
        return (fin[0].reshaped([1, 1, H]), states)
    }
}
