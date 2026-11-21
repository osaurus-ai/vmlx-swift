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

    /// One simdgroup (= threadgroup) per head vector: NQ q heads, NKV k heads, NKV v heads, then
    /// NQ * NB r groups (sarr[1] = NB key slices per r head, so the bias rows of long windows
    /// spread over the GPU instead of one simdgroup walking every key). q / r come from `qa` at offsets imeta[0] / imeta[1]; k (+ dyn conv weights) from
    /// `ka` at imeta[2] (/ imeta[3]); v from `va` at imeta[4] (/ imeta[5]); sarr[0] = S visible
    /// keys (one array per token, shared by every visit). fmeta = [qk clip, rel clip, v-norm eps, tau]. k / v run their short conv (static, or
    /// DYN dynamic) against the packed state (k columns 0.., v columns C..), are rounded to bf16,
    /// then clipped / normalized. q / r are clipped, rounded, and (tau != 1) log-scaled and rounded
    /// again, as the reference does. r heads also write the decode bias row for their head.
    /// Template: NQ, NKV, D, DR, DYN.
    static let qkv = MLXFast.metalKernel(
        name: "inkling2_dec_qkv",
        inputNames: [
            "qa", "ka", "va", "kw", "vw", "packed", "qscale", "rscale", "proj", "fmeta", "imeta",
            "sarr",
        ],
        outputNames: ["qo", "ko", "vo", "bias", "kvst"],
        source: """
            uint head = threadgroup_position_in_grid.x;
            uint lane = thread_index_in_simdgroup;
            constexpr int C = NKV * D;
            constexpr int EL = D / 32;
            int P = packed_shape[1];
            int S = sarr[0];
            int NB = sarr[1];
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
            } else if (head >= (uint)(NQ + 2 * NKV)) {
                // r heads: NB groups per head, group b covers keys [b * SB, (b + 1) * SB)
                int rh = head - NQ - 2 * NKV;
                int h = rh / NB, blk = rh % NB;
                int SB = (S + NB - 1) / NB;
                float x = 0.f;
                if (lane < (uint)DR) x = float(qa[imeta[1] + h * DR + lane]) * rscale[h * DR + lane];
                float ss = simd_sum(x * x);
                float f = min(rc / (sqrt(ss) + 1e-6f), 1.f);
                float y = ink_round_as(bias, x * f);
                if (tau != 1.f) y = ink_round_as(bias, y * tau);
                if (lane < (uint)DR) rr[lane] = y;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                int j1 = min(S, (blk + 1) * SB);
                for (int j = blk * SB + lane; j < j1; j += 32) {
                    int d = S - 1 - j;
                    float b = 0.f;
                    if (d < E) {
                        for (int k = 0; k < DR; ++k) b += rr[k] * float(proj[k * E + d]);
                    }
                    ink_store(bias, (size_t)h * S + j, b);
                }
            } else {
                bool isK = head < (uint)(NQ + NKV);
                int h = isK ? head - NQ : head - NQ - NKV;
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

    // MARK: loop message gates (fused: input RMSNorms + down projection | up projections + mix + norm)

    /// Message gate, first half: per gate row r (g-major over T rows: r = g * T + t), the unweighted
    /// RMSNorms of the base row `x[t * NX + (NX == 1 ? 0 : g)]` and the message row
    /// `msgs[t * MS + mrow + g]`, each rounded to the down projection's dtype (the reference's
    /// `rms_norm(...).astype(w)`), then `d = dtype(gin @ down[g]^T)` returned as fp32, one simdgroup
    /// per output. Grid (NCH, G * T): each group recomputes the row norms and covers `NSG * OPS`
    /// of the `rank` outputs. `w` [G, rank, 2H]; meta = [mrow, NX, T, MS]. Template HH, TG, OPS.
    static let gateDown = MLXFast.metalKernel(
        name: "inkling2_gate_down",
        inputNames: ["x", "msgs", "w", "eps", "meta"],
        outputNames: ["d"],
        source: """
            uint chunk = threadgroup_position_in_grid.x;
            uint r = threadgroup_position_in_grid.y;
            uint tid = thread_position_in_threadgroup.x;
            uint lane = thread_index_in_simdgroup;
            uint sg = simdgroup_index_in_threadgroup;
            constexpr int NSG = TG / 32;
            constexpr int H = HH;
            int R = w_shape[1];
            int T = meta[2];
            int g = r / T, t = r % T;
            int xr = t * meta[1] + (meta[1] == 1 ? 0 : g);
            int mr = t * meta[3] + meta[0] + g;
            threadgroup float gin[2 * H];
            float a = 0.f, b = 0.f;
            for (int j = tid; j < H; j += TG) {
                float hv = x[(size_t)xr * H + j], mv = msgs[(size_t)mr * H + j];
                gin[j] = hv; gin[H + j] = mv;
                a += hv * hv; b += mv * mv;
            }
            threadgroup float ra[NSG], rb[NSG];
            a = simd_sum(a); b = simd_sum(b);
            if (lane == 0) { ra[sg] = a; rb[sg] = b; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float A = 0.f, B = 0.f;
            for (int k = 0; k < NSG; ++k) { A += ra[k]; B += rb[k]; }
            float ia = rsqrt(A / float(H) + eps[0]);
            float ib = rsqrt(B / float(H) + eps[0]);
            for (int j = tid; j < H; j += TG) {
                gin[j] = ink_round_as(w, gin[j] * ia);
                gin[H + j] = ink_round_as(w, gin[H + j] * ib);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (int q = 0; q < OPS; ++q) {
                int o = (chunk * NSG + sg) * OPS + q;
                if (o >= R) break;
                const device auto* wr = w + ((size_t)g * R + o) * 2 * H;
                float acc = 0.f;
                for (int j = lane * 4; j < 2 * H; j += 128) {
                    acc += gin[j] * float(wr[j]) + gin[j + 1] * float(wr[j + 1])
                        + gin[j + 2] * float(wr[j + 2]) + gin[j + 3] * float(wr[j + 3]);
                }
                acc = simd_sum(acc);
                if (lane == 0) d[(size_t)r * R + o] = ink_round_as(w, acc);
            }
            """,
        header: header)

    /// Message gate, second half (column-parallel): y[r, j] = (1 - tanh(d[r] . Wf[g][:, j])) * base
    /// + tanh(d[r] . Wg[g][:, j]) * msg, fp32. `fgw` [G, 2, rank, H] (transposed: lanes read
    /// consecutive columns, four per thread). Grid (H / 4, G * T). meta as `gateDown`. Template RANK.
    static let gateMix = MLXFast.metalKernel(
        name: "inkling2_gate_mix",
        inputNames: ["d", "fgw", "x", "msgs", "meta"],
        outputNames: ["y"],
        source: """
            uint j = thread_position_in_grid.x;
            uint r = threadgroup_position_in_grid.y;
            uint tid = thread_position_in_threadgroup.x;
            constexpr int R = RANK;
            int H = x_shape[1];
            int T = meta[2];
            int g = r / T, t = r % T;
            threadgroup float dv[R];
            for (int k = tid; k < R; k += (int)threads_per_threadgroup.x) dv[k] = d[(size_t)r * R + k];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            int j0 = j * 4;
            if (j0 >= H) return;
            int xr = t * meta[1] + (meta[1] == 1 ? 0 : g);
            int mr = t * meta[3] + meta[0] + g;
            const device auto* wf = fgw + (size_t)g * 2 * R * H + j0;
            const device auto* wg = wf + (size_t)R * H;
            float f0 = 0.f, f1 = 0.f, f2 = 0.f, f3 = 0.f, a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
            for (int k = 0; k < R; ++k) {
                float dk = dv[k];
                const device auto* p = wf + (size_t)k * H;
                const device auto* q = wg + (size_t)k * H;
                f0 += dk * float(p[0]); f1 += dk * float(p[1]); f2 += dk * float(p[2]); f3 += dk * float(p[3]);
                a0 += dk * float(q[0]); a1 += dk * float(q[1]); a2 += dk * float(q[2]); a3 += dk * float(q[3]);
            }
            const device float* xb = x + (size_t)xr * H + j0;
            const device float* mb = msgs + (size_t)mr * H + j0;
            device float* yo = y + (size_t)r * H + j0;
            yo[0] = (1.f - tanh(f0)) * xb[0] + tanh(a0) * mb[0];
            yo[1] = (1.f - tanh(f1)) * xb[1] + tanh(a1) * mb[1];
            yo[2] = (1.f - tanh(f2)) * xb[2] + tanh(a2) * mb[2];
            yo[3] = (1.f - tanh(f3)) * xb[3] + tanh(a3) * mb[3];
            """,
        header: header)

    /// The gate row's weighted RMSNorm: out[r] = dtype(y[r] * rsqrt(mean(y^2) + eps) * nw[r / T]),
    /// out32 = fp32(out). One group per row. meta[2] = T. Template ELEMS, TG.
    static let gateNorm = MLXFast.metalKernel(
        name: "inkling2_gate_norm",
        inputNames: ["y", "nw", "ne", "meta"],
        outputNames: ["out", "out32"],
        source: """
            uint r = threadgroup_position_in_grid.x;
            uint tid = thread_position_in_threadgroup.x;
            uint lane = thread_index_in_simdgroup;
            uint sg = simdgroup_index_in_threadgroup;
            constexpr int NSG = TG / 32;
            int H = y_shape[1];
            int g = r / meta[2];
            float v[ELEMS];
            float sq = 0.f;
            for (int e = 0; e < ELEMS; ++e) {
                int j = tid + e * TG;
                v[e] = j < H ? y[(size_t)r * H + j] : 0.f;
                sq += v[e] * v[e];
            }
            threadgroup float red[NSG];
            sq = simd_sum(sq);
            if (lane == 0) red[sg] = sq;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float tt = 0.f;
            for (int k = 0; k < NSG; ++k) tt += red[k];
            float n = rsqrt(tt / float(H) + ne[0]);
            for (int e = 0; e < ELEMS; ++e) {
                int j = tid + e * TG;
                if (j < H) {
                    ink_store(out, (size_t)r * H + j, (v[e] * n) * nw[(size_t)g * H + j]);
                    out32[(size_t)r * H + j] = float(out[(size_t)r * H + j]);
                }
            }
            """,
        header: header)

    // MARK: routed experts (NVFP4), batch-1 gathers

    static let nvfp4Header =
        header + """
            inline float ink_e4m3(uint b) {
                uint e = (b >> 3) & 0xF, m = b & 7;
                float v = e == 0 ? float(m) * 0.001953125f : (1.f + float(m) * 0.125f) * exp2(float(int(e) - 7));
                return (b & 0x80) ? -v : v;
            }
            // one 16-value NVFP4 group (two packed words, even element in the low nibble) . x[0 ..< 16]
            inline float ink_fp4_dot16(uint2 w, const device float* x, threadgroup const float* lut) {
                float4 xa = *(const device float4*)(x), xb = *(const device float4*)(x + 4);
                float4 xc = *(const device float4*)(x + 8), xd = *(const device float4*)(x + 12);
                float p = lut[w.x & 0xF] * xa.x + lut[(w.x >> 4) & 0xF] * xa.y
                    + lut[(w.x >> 8) & 0xF] * xa.z + lut[(w.x >> 12) & 0xF] * xa.w;
                p += lut[(w.x >> 16) & 0xF] * xb.x + lut[(w.x >> 20) & 0xF] * xb.y
                    + lut[(w.x >> 24) & 0xF] * xb.z + lut[(w.x >> 28) & 0xF] * xb.w;
                p += lut[w.y & 0xF] * xc.x + lut[(w.y >> 4) & 0xF] * xc.y
                    + lut[(w.y >> 8) & 0xF] * xc.z + lut[(w.y >> 12) & 0xF] * xc.w;
                p += lut[(w.y >> 16) & 0xF] * xd.x + lut[(w.y >> 20) & 0xF] * xd.y
                    + lut[(w.y >> 24) & 0xF] * xd.z + lut[(w.y >> 28) & 0xF] * xd.w;
                return p;
            }
            """

    /// Routed gate/up (row-interleaved: row 2i = gate_i, 2i + 1 = up_i) for each (row, pick) n:
    /// both rows of pair i in one simdgroup (16 lanes each, one 16-value group per lane step),
    /// then the activation `softcap(g) * sigmoid(g) * softcap(u)` with the per-expert NVFP4 scale,
    /// rounded to bf16 like `moeActivation`. x [T, H] fp32 (row n / KPER); idx [T * KPER].
    /// Grid (I / NSG * 32 * NSG, T * KPER). Output [T * KPER, I] fp32.
    static let moeGateUpAct = MLXFast.metalKernel(
        name: "inkling2_moe_gate_up_act",
        inputNames: ["x", "codes", "scales", "idx", "es", "caps"],
        outputNames: ["out"],
        source: """
            uint lane = thread_index_in_simdgroup;
            uint sg = simdgroup_index_in_threadgroup;
            uint tid = thread_position_in_threadgroup.x;
            uint n = threadgroup_position_in_grid.y;
            threadgroup float lut[16];
            if (tid < 16) {
                uint m_ = tid & 7, e_ = m_ >> 1, f_ = m_ & 1;
                float v_ = e_ == 0 ? 0.5f * float(f_) : (1.f + 0.5f * float(f_)) * float(1 << (e_ - 1));
                lut[tid] = (tid & 8) ? -v_ : v_;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            int N2 = codes_shape[1];
            int I = N2 / 2;
            int Kin = codes_shape[2] * 8;
            int pair = threadgroup_position_in_grid.x * NSG + sg;
            uint sub = lane / 16, sl = lane % 16;
            int row = 2 * pair + sub;
            uint e = idx[n];
            const device uint32_t* cw = codes + ((size_t)e * N2 + row) * (Kin / 8);
            const device uint8_t* sw = scales + ((size_t)e * N2 + row) * (Kin / 16);
            const device float* xr = x + (size_t)(n / KPER) * Kin;
            float acc = 0.f;
            for (int g = sl; g < Kin / 16; g += 16)
                acc += ink_e4m3(sw[g]) * ink_fp4_dot16(*(const device uint2*)(cw + 2 * g), xr + g * 16, lut);
            for (uint o = 8; o > 0; o >>= 1) acc += simd_shuffle_xor(acc, o);
            float up = simd_shuffle(acc, 16);
            if (lane == 0 && pair < I) {
                float s = es[e];
                float gg = acc * s, u = up * s;
                float gc = caps[0], uc = caps[1];
                float a = (tanh(gg / gc) * gc) * (1.f / (1.f + exp(-gg))) * (tanh(u / uc) * uc);
                out[(size_t)n * I + pair] = float(bfloat16_t(a));
            }
            """,
        header: nvfp4Header)

    /// Routed down projection per pick n: out[n, j] = act[n] . W_{idx[n]}[j] (NVFP4, without the
    /// per-expert scale, which `final` applies). 8 lanes per row. Grid (H / (4 * NSG) * 32 * NSG, N).
    static let moeDown = MLXFast.metalKernel(
        name: "inkling2_moe_down",
        inputNames: ["x", "codes", "scales", "idx"],
        outputNames: ["out"],
        source: """
            uint lane = thread_index_in_simdgroup;
            uint sg = simdgroup_index_in_threadgroup;
            uint tid = thread_position_in_threadgroup.x;
            uint n = threadgroup_position_in_grid.y;
            threadgroup float lut[16];
            if (tid < 16) {
                uint m_ = tid & 7, e_ = m_ >> 1, f_ = m_ & 1;
                float v_ = e_ == 0 ? 0.5f * float(f_) : (1.f + 0.5f * float(f_)) * float(1 << (e_ - 1));
                lut[tid] = (tid & 8) ? -v_ : v_;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            int N = codes_shape[1];
            int Kin = codes_shape[2] * 8;
            uint sub = lane / 8, sl = lane % 8;
            int row = (threadgroup_position_in_grid.x * NSG + sg) * 4 + sub;
            uint e = idx[n];
            const device uint32_t* cw = codes + ((size_t)e * N + row) * (Kin / 8);
            const device uint8_t* sw = scales + ((size_t)e * N + row) * (Kin / 16);
            const device float* xr = x + (size_t)n * Kin;
            float acc = 0.f;
            if (row < N) {
                for (int g = sl; g < Kin / 16; g += 8)
                    acc += ink_e4m3(sw[g]) * ink_fp4_dot16(*(const device uint2*)(cw + 2 * g), xr + g * 16, lut);
            }
            for (uint o = 4; o > 0; o >>= 1) acc += simd_shuffle_xor(acc, o);
            if (sl == 0 && row < N) out[(size_t)n * N + row] = acc;
            """,
        header: nvfp4Header)

    /// NVFP4 experts for T rows x k picks: `(act [T*k, I], down [T*k, H])`, both fp32.
    static func nvfp4Experts(
        x32: MLXArray, inds: MLXArray, mlp: Inkling2SwitchMLP, topK: Int
    ) -> (MLXArray, MLXArray) {
        nvfp4Experts(
            x32: x32, inds: inds, gateUpCodes: mlp.gateUp.codes,
            gateUpScales: mlp.gateUp.blockScales, downCodes: mlp.down.codes,
            downScales: mlp.down.blockScales, gateUpExpertScale: mlp.gateUp.expertScale,
            caps: mlp.caps, topK: topK)
    }

    static func nvfp4Experts(
        x32: MLXArray, inds: MLXArray, gateUpCodes: MLXArray, gateUpScales: MLXArray,
        downCodes: MLXArray, downScales: MLXArray, gateUpExpertScale: MLXArray, caps: MLXArray,
        topK: Int
    ) -> (MLXArray, MLXArray) {
        let n = inds.size
        let I = gateUpCodes.dim(1) / 2
        let H = downCodes.dim(1)
        let nsg = 4
        let flat = inds.reshaped([n])
        let act = moeGateUpAct(
            [x32, gateUpCodes, gateUpScales, flat, gateUpExpertScale, caps],
            template: [("NSG", nsg), ("KPER", topK)],
            grid: ((I + nsg - 1) / nsg * 32 * nsg, n, 1), threadGroup: (32 * nsg, 1, 1),
            outputShapes: [[n, I]], outputDTypes: [f32])[0]
        let rowsPerTG = 4 * nsg
        let down = moeDown(
            [act, downCodes, downScales, flat], template: [("NSG", nsg)],
            grid: ((H + rowsPerTG - 1) / rowsPerTG * 32 * nsg, n, 1),
            threadGroup: (32 * nsg, 1, 1),
            outputShapes: [[n, H]], outputDTypes: [f32])[0]
        return (act, down)
    }

    /// The three gate kernels for G gates x T rows. Returns `(out [G*T, H] model dtype, out32 fp32)`.
    static func messageGates(
        x: MLXArray, msgs: MLXArray, down: MLXArray, fg: MLXArray, norm: MLXArray,
        normEps: MLXArray, gateEps: MLXArray, meta: MLXArray, gates: Int, rows: Int,
        dtype: DType
    ) -> (MLXArray, MLXArray) {
        let H = x.dim(-1)
        let rank = down.dim(1)
        let tgD = 256
        let ops = 1
        let nch = (rank + (tgD / 32) * ops - 1) / ((tgD / 32) * ops)
        let n = gates * rows
        let d = gateDown(
            [x, msgs, down, gateEps, meta], template: [("HH", H), ("TG", tgD), ("OPS", ops)],
            grid: (nch * tgD, n, 1), threadGroup: (tgD, 1, 1),
            outputShapes: [[n, rank]], outputDTypes: [f32])[0]
        let tgM = 192
        let y = gateMix(
            [d, fg, x, msgs, meta], template: [("RANK", rank)],
            grid: ((H / 4 + tgM - 1) / tgM * tgM, n, 1), threadGroup: (tgM, 1, 1),
            outputShapes: [[n, H]], outputDTypes: [f32])[0]
        let tg = 768
        let o = gateNorm(
            [y, norm, normEps, meta], template: [("ELEMS", (H + tg - 1) / tg), ("TG", tg)],
            grid: (n * tg, 1, 1), threadGroup: (tg, 1, 1),
            outputShapes: [[n, H], [n, H]], outputDTypes: [dtype, f32])
        return (o[0], o[1])
    }
}

/// Per-token decode scalars shared by every layer visit of one forward (all visits of a token see
/// the same cache offset): small int / float arrays memoized by value instead of one fresh array
/// per kernel call per visit.
final class Inkling2StepScalars {
    private var ints: [[Int32]: MLXArray] = [:]
    private var floats: [[Float]: MLXArray] = [:]
    func ints(_ v: [Int]) -> MLXArray {
        let key = v.map { Int32($0) }
        if let a = ints[key] { return a }
        let a = MLXArray(key)
        ints[key] = a
        return a
    }
    func floats(_ v: [Float]) -> MLXArray {
        if let a = floats[v] { return a }
        let a = MLXArray(v)
        floats[v] = a
        return a
    }
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
    /// Message gates in their checkpoint dtype: down `[G, rank, 2H]`, forget / gain `[G, 2, rank, H]`
    /// (G = 3 for q / k / v, 1 for the MLP gate).
    var qkvDown: MLXArray?
    var qkvFG: MLXArray?
    var mlpDown: MLXArray?
    var mlpFG: MLXArray?
    var gateEps: MLXArray?
    var routerT: MLXArray?
    /// Post-attention (MLP) norm weight `[1, H]` fp32.
    var mlpNorm: MLXArray?
    var ones: MLXArray?
    var downOnes: MLXArray?
    /// Load-time constants of this layer's kernels (one array each instead of one per call).
    var attnQuery: MLXArray?
    var mlpQuery: MLXArray?
    var zeroState: MLXArray?
    var qkvFMeta: MLXArray?
    var qkvIMetaX: MLXArray?
    var qkvIMeta3: MLXArray?
    var gateMetaQKV: MLXArray?
    var gateMetaMLP: MLXArray?
    var sigMeta: MLXArray?
    var midMeta: MLXArray?
    var finalMeta: MLXArray?
    var loopIndex = -1
    /// Both expert projections NVFP4 with the layout the batch-1 kernels read.
    var nativeFP4Experts = false
    var ready = false
    private var zeroSinkCache: MLXArray?

    /// A zero native sink per head: the multi-row step folds each row's sink into its mask.
    func zeroSinks(nq: Int, dtype: DType) -> MLXArray {
        if let z = zeroSinkCache, z.dim(0) == nq, z.dtype == dtype { return z }
        let z = MLXArray.zeros([nq], dtype: dtype)
        zeroSinkCache = z
        return z
    }
}

extension Inkling2DecoderLayer {

    /// True when this layer can run `decodeStep` (plain-weight projections, standard sizes).
    func supportsDecodeStep() -> Bool {
        let a = selfAttn
        let plain: [Linear] = [a.qProj, a.rProj, a.gateProj, a.sinkProj, a.kProj, a.vProj]
        guard plain.allSatisfy({ type(of: $0) == Linear.self && $0.bias == nil }) else {
            return false
        }
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
        // The gate kernels read 4 columns per thread / 4 inputs per lane (hidden % 64 == 0).
        return a.hd % 32 == 0 && a.hd <= 128 && a.dRel <= 32 && a.nq <= 64 && hidden % 64 == 0
            && router.weight.dim(0) <= 256 && router.topK <= 16 && router.topK >= 1
    }

    func prepareDecodeWeights() {
        let w = decodeWeights
        guard !w.ready else { return }
        let a = selfAttn
        let H = hidden
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
        w.gateNorm =
            isSliding
            ? MLX.stacked([inputLN!, inputLN!, inputLN!].map { $0.weight.asType(f32) }, axis: 0)
            : w.attnNorm
        w.normEps = MLXArray([postAttnLN.eps])
        w.mlpNorm = MLX.contiguous(postAttnLN.weight.asType(f32).reshaped([1, H]))
        if isLoopLayer, let qg = qGate, let kg = kGate, let vg = vGate, let mg = mlpGate {
            func fg(_ g: Inkling2MessageGate) -> MLXArray {
                MLX.stacked([g.forgetProj.weight.T, g.gainProj.weight.T], axis: 0)  // [2, rank, H]
            }
            w.qkvDown = MLX.contiguous(
                MLX.stacked([qg, kg, vg].map { $0.downProj.weight }, axis: 0))
            w.qkvFG = MLX.contiguous(MLX.stacked([qg, kg, vg].map(fg), axis: 0))
            w.mlpDown = MLX.contiguous(mg.downProj.weight.expandedDimensions(axis: 0))
            w.mlpFG = MLX.contiguous(fg(mg).expandedDimensions(axis: 0))
            w.gateEps = MLXArray([qg.eps])
        }
        let sw = mlp.switchMLP
        w.nativeFP4Experts =
            sw.gateUp.isNVFP4 && sw.down.isNVFP4 && sw.inter % 16 == 0 && H % 16 == 0
            && sw.gateUp.codes.dim(1) == 2 * sw.inter && sw.down.codes.dim(1) == H
            && ProcessInfo.processInfo.environment["VMLX_INKLING2_MLX_EXPERTS"] != "1"
        w.routerT = mlp.gate.weight.asType(f32).T
        w.ones = MLXArray.ones([mlp.switchMLP.gateUp.expertScale.dim(0)], dtype: f32)
        w.downOnes = w.ones
        let nqa = isSliding ? 1 : 3
        if let attentionInput {
            w.attnQuery = MLX.contiguous(attentionInput.query.asType(f32).reshaped([nqa, H]))
        }
        w.mlpQuery = MLX.contiguous(mlpInput.query.asType(f32).reshaped([1, H]))
        w.zeroState = MLXArray.zeros([1, H], dtype: f32)
        let (nq, nkv, hd, dRel) = (a.nq, a.nkv, a.hd, a.dRel)
        let (nQ, nR, C) = (nq * hd, nq * dRel, nkv * hd)
        let base = 2 * nQ + nR + nq
        w.qkvFMeta = MLXArray([a.qkClip, a.relClip, a.eps, 1])
        w.qkvIMetaX = MLXArray([0, Int32(nQ), Int32(base), 0, Int32(base + C), 0])
        // Three input rows: [q-fused | k-fused | v-fused] outputs laid end to end.
        let nQF = w.qFused!.dim(0)
        let nKF = w.kFused!.dim(0)
        w.qkvIMeta3 = MLXArray([
            0, Int32(nQ), Int32(nQF), Int32(nQF + C), Int32(nQF + nKF), Int32(nQF + nKF + C),
        ])
        w.sigMeta = MLXArray([Int32(nQ + nR)])
        w.midMeta = MLXArray([Int32(2 * C)])
        w.finalMeta = MLXArray([Int32(2 * C), Int32(H)])
        eval(
            [
                w.xFused, w.qFused, w.kFused, w.vFused, w.attnNorm, w.gateNorm,
                w.qkvDown, w.qkvFG, w.mlpDown, w.mlpFG, w.routerT, w.ones, w.mlpNorm, w.attnQuery,
                w.mlpQuery, w.zeroState,
            ].compactMap { $0 })
        w.ready = true
    }

    /// Gate meta rows for this layer's loop index (constant per layer).
    func gateMetas(loopIndex: Int, messageRows: Int) -> (MLXArray, MLXArray) {
        let w = decodeWeights
        if w.loopIndex != loopIndex || w.gateMetaQKV == nil {
            let nqa = isSliding ? 1 : 3
            w.gateMetaQKV = MLXArray([
                Int32(4 * loopIndex), Int32(nqa), 1, Int32(messageRows),
            ])
            w.gateMetaMLP = MLXArray([Int32(4 * loopIndex + 3), 1, 1, Int32(messageRows)])
            w.loopIndex = loopIndex
        }
        return (w.gateMetaQKV!, w.gateMetaMLP!)
    }

    /// One token (B = 1, T = 1). `stack` `[1, 1, S, H]` fp32; `messages` `[1, 1, 4 * loopLayers, H]`
    /// with this layer's rows at `4 * loopIndex`. Returns `(h, stack')`.
    func decodeStep(
        _ h: MLXArray, stack: MLXArray?, kv: KVCacheSimple, conv: MambaCache,
        messages: MLXArray?, loopIndex: Int, ones: MLXArray, scalars: Inkling2StepScalars
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
        let st0 = stack?.reshaped([S0, H]) ?? w.zeroState!
        let msgs = messages?.reshaped([-1, H])
        let nqa = isSliding ? 1 : 3
        let metas = msgs.map { gateMetas(loopIndex: loopIndex, messageRows: $0.dim(0)) }

        // 1. attention inputs: rows [1 | 3, H] in the model dtype
        var rows: MLXArray
        if attentionInput == nil {
            rows = inputLN!(h1)  // layer 0: input_layernorm(h)
        } else if msgs == nil {
            rows =
                Inkling2DecodeKernels.depth(
                    [st0, h1, w.attnQuery!, w.attnNorm!, w.normEps!],
                    template: [
                        ("NS", S0 + 1), ("HASP", 1), ("NQ", nqa), ("NORM", 1), ("ELEMS", elems),
                        ("TG", tg),
                    ],
                    grid: (tg, 1, 1), threadGroup: (tg, 1, 1), outputShapes: [[nqa, H]],
                    outputDTypes: [dtype])[0]
        } else {
            // depth attention (raw) -> fused message gates (+ the q / k / v input RMSNorms)
            let xr = Inkling2DecodeKernels.depth(
                [st0, h1, w.attnQuery!, w.attnNorm!, w.normEps!],
                template: [
                    ("NS", S0 + 1), ("HASP", 1), ("NQ", nqa), ("NORM", 0), ("ELEMS", elems),
                    ("TG", tg),
                ],
                grid: (tg, 1, 1), threadGroup: (tg, 1, 1), outputShapes: [[nqa, H]],
                outputDTypes: [f32])[0]
            rows =
                Inkling2DecodeKernels.messageGates(
                    x: xr, msgs: msgs!, down: w.qkvDown!, fg: w.qkvFG!, norm: w.gateNorm!,
                    normEps: w.normEps!, gateEps: w.gateEps!, meta: metas!.0, gates: 3, rows: 1,
                    dtype: dtype
                ).0
        }

        // 2. projections: one GEMV over [q|r|gate|sink|k|v] (one input row) or three GEMVs
        // (q-fused / k-fused / v-fused for three input rows), read by the kernel at offsets
        let nQ = nq * hd
        let nR = nq * dRel
        let y: MLXArray
        let imeta: MLXArray
        if rows.dim(0) == 1 {
            y = MLX.matmul(rows, w.xFused!.T).reshaped([-1])
            imeta = w.qkvIMetaX!
        } else {
            y = MLX.concatenated([
                MLX.matmul(rows[0 ..< 1], w.qFused!.T).reshaped([-1]),
                MLX.matmul(rows[1 ..< 2], w.kFused!.T).reshaped([-1]),
                MLX.matmul(rows[2 ..< 3], w.vFused!.T).reshaped([-1]),
            ])
            imeta = w.qkvIMeta3!
        }
        let seen = kv.offset
        let S = a.window.map { min(seen + 1, $0) } ?? (seen + 1)
        // relative-bias rows: one simdgroup per 128 keys per head
        let nb = max(1, (S + 127) / 128)
        var fmeta = w.qkvFMeta!
        if !a.isSliding, Float(seen + 1) > a.logFloor {
            let tau = 1 + a.logAlpha * Foundation.log(max(Float(seen + 1) / a.logFloor, 1))
            fmeta = scalars.floats([a.qkClip, a.relClip, a.eps, tau])
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
                y, y, y, kwStatic, vwStatic, packed, a.qL2Clip.scale, a.relL2Clip.scale,
                a.relLogits.proj, fmeta, imeta, scalars.ints([S, nb]),
            ],
            template: [
                ("NQ", nq), ("NKV", nkv), ("D", hd), ("DR", dRel), ("DYN", a.isSliding ? 0 : 1),
            ],
            grid: ((nq + 2 * nkv + nq * nb) * 32, 1, 1), threadGroup: (32, 1, 1),
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
            sinks: y[sinkOff ..< (sinkOff + nq)])
        let gated = Inkling2DecodeKernels.sigmoidGate(
            [attn.reshaped([nQ]), y, w.sigMeta!], grid: (nQ, 1, 1),
            threadGroup: (256, 1, 1), outputShapes: [[nQ]], outputDTypes: [dtype])[0]
        let o = a.oProj(gated.reshaped([1, nQ]))

        // 4. residual + MLP input
        let states = isBlockBoundary ? inklingAppendState(stack, h) : stack
        let S1 = states?.dim(2) ?? 0
        let st1 = states?.reshaped([S1, H]) ?? w.zeroState!
        let norm = msgs == nil
        let midOut = Inkling2DecodeKernels.mid(
            [
                o.reshaped([H]), h1.reshaped([H]), packed, postAttnConv.weight, st1,
                w.mlpQuery!, w.mlpNorm!, w.normEps!, w.midMeta!,
            ],
            template: [
                ("NS", S1 + 1), ("ELEMS", elems), ("TG", tg), ("NORM", norm ? 1 : 0),
                ("BOUNDARY", isBlockBoundary ? 1 : 0),
            ],
            grid: (tg, 1, 1), threadGroup: (tg, 1, 1),
            outputShapes: [[H], [H], [H], [3, H]], outputDTypes: [dtype, f32, f32, f32])
        var xm32 = midOut[1]
        if let msgs {
            xm32 =
                Inkling2DecodeKernels.messageGates(
                    x: xm32.reshaped([1, H]), msgs: msgs, down: w.mlpDown!, fg: w.mlpFG!,
                    norm: w.mlpNorm!, normEps: w.normEps!, gateEps: w.gateEps!, meta: metas!.1,
                    gates: 1, rows: 1, dtype: dtype
                ).1
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
        let down: MLXArray
        if w.nativeFP4Experts {
            down = Inkling2DecodeKernels.nvfp4Experts(x32: x32, inds: inds, mlp: sw, topK: k).1
        } else {
            let gu = sw.gateUp(x32.reshaped([1, 1, 1, H]), inds, sorted: false)  // [1, k, 1, 2I]
            let act = Inkling2Kernels.moeActivation(
                [
                    gu.reshaped([k, 2 * sw.inter]), inds.reshaped([k]),
                    sw.gateUp.isNVFP4 ? sw.gateUp.expertScale : w.ones!, sw.caps,
                ],
                grid: (sw.inter, k, 1), threadGroup: (min(256, sw.inter), 1, 1),
                outputShapes: [[k, sw.inter]], outputDTypes: [f32])[0]
            down = sw.down(act.reshaped([1, k, 1, sw.inter]), inds, sorted: false)  // [1, k, 1, H]
        }
        let fin = Inkling2DecodeKernels.final(
            [
                down.reshaped([k, H]), inds, sw.down.isNVFP4 ? sw.down.expertScale : w.ones!,
                sel[1],
                packed, postMLPConv.weight, midOut[2], outs[4], midOut[3], w.finalMeta!,
            ],
            grid: (P, 1, 1), threadGroup: (256, 1, 1), outputShapes: [[H], [3, P]],
            outputDTypes: [dtype, f32])
        conv[0] = fin[1].reshaped([1, 3, P])
        conv.offset += 1
        return (fin[0].reshaped([1, 1, H]), states)
    }
}
