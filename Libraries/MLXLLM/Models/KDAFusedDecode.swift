// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXFast

/// One-dispatch KDA decode middle (GLM-5.3 / Ling KDA layers, batch 1, one token): short conv + silu, q/k l2norms,
/// decay gate, beta, recurrent update and the gated RMSNorm, one threadgroup per head. Port of vMLX Python
/// `metal/glm5_kda_fused_decode.py` (accepted 2026-10-10: GLM decode 32.1 -> 34.3 tok/s, +7 %, teacher-forced KL 0.0032).
///
/// The stock path issues ~20 dependent dispatches per KDA layer; in GLM's 34-layer dependent decode chain small
/// kernels cost their latency, not their work. Layouts are this engine's, not the reference's:
/// - the conv state is ONE `[1, W-1, 3C]` buffer, channels q | k | v (`cache[0]`);
/// - the recurrent state is `[1, H, Dv, Dk]` (value-major) fp32 (`cache[1]`), so each thread reads its 32-key run of
///   one value row contiguously.
/// Math per head: S' = S * decay(key); kS = S'.k; corr = v - kS; S_new = S' + beta * k (x) corr;
/// o = q.S' + beta * corr * (q.k) (= q.S_new); out = rmsnorm(o) * w * sigmoid(gate). Same formulas as the stock
/// conv / `computeKDADecay` / `kdaRecurrence` / `BailingV3RMSNormGated`, accumulated in fp32 (the stock path rounds
/// q/k to the activation dtype after the l2norm; this keeps them fp32, like the reference kernel).
/// `VMLX_KDA_FUSED_DECODE=0` restores the multi-dispatch path.
enum KDAFusedDecode {
    static let enabled: Bool = {
        let raw = (ProcessInfo.processInfo.environment["VMLX_KDA_FUSED_DECODE"] ?? "1").lowercased()
        return !["0", "false", "off", "no"].contains(raw)
    }()

    private static let lock = NSLock()
    nonisolated(unsafe) private static var kernels: [String: MLXFast.MLXFastKernel] = [:]

    private static func kernel(heads h: Int, keyDim k: Int, width w: Int, lowerBound: Float, eps: Float)
        -> MLXFast.MLXFastKernel
    {
        let name = "vmlx_kda_fused_decode_h\(h)_k\(k)_w\(w)_lb\(lowerBound.bitPattern)_e\(eps.bitPattern)"
        lock.lock()
        defer { lock.unlock() }
        if let cached = kernels[name] { return cached }
        let c = h * k
        let lb = String(format: "%.9e", lowerBound)
        let ep = String(format: "%.9e", eps)
        let invSqrtK = String(format: "%.9e", 1 / Float(k).squareRoot())
        let source = """
            const uint h = threadgroup_position_in_grid.x;
            const uint tid = thread_index_in_threadgroup;
            const uint lane = thread_index_in_simdgroup;
            const uint sg = simdgroup_index_in_threadgroup;
            threadgroup float qv[\(k)], kv[\(k)], vv[\(k)], eg[\(k)];
            threadgroup float red[16][3];
            threadgroup float scal[4];
            threadgroup float pA[4][\(k)], pB[4][\(k)];
            threadgroup float ov[\(k)];

            // ---- phase 1: q/k/v short conv (+silu, rounded to T like the stock conv) and the decay gate
            if (tid < \(3 * k)u) {
                const uint which = tid / \(k)u;            // 0 q, 1 k, 2 v
                const uint i = tid % \(k)u;
                const uint c = h * \(k)u + i;
                const uint ch = which * \(c)u + c;          // channel in the concatenated q|k|v buffers
                device const T* wt = which == 0u ? q_w : (which == 1u ? k_w : v_w);
                float acc = 0.0f;
                for (uint tap = 0; tap < \(w - 1)u; ++tap)
                    acc += (float)conv_state[(size_t)tap * \(3 * c)u + ch] * (float)wt[(size_t)c * \(w)u + tap];
                acc += (float)mixed[ch] * (float)wt[(size_t)c * \(w)u + \(w - 1)u];
                for (uint tap = 0; tap + 1u < \(w - 1)u; ++tap)
                    next_conv[(size_t)tap * \(3 * c)u + ch] = conv_state[(size_t)(tap + 1u) * \(3 * c)u + ch];
                next_conv[(size_t)\(w - 2)u * \(3 * c)u + ch] = mixed[ch];
                float y = (float)(T)(acc / (1.0f + metal::exp(-acc)));
                if (which == 0u) qv[i] = y; else if (which == 1u) kv[i] = y; else vv[i] = y;
            } else {
                const uint i = tid - \(3 * k)u;             // decay gate per key channel
                const uint c = h * \(k)u + i;
                float x = (float)f_raw[c] + (float)dt_bias[c];
                float rate = metal::exp((float)A_log[h]);
                eg[i] = metal::exp(\(lb)f * (1.0f / (1.0f + metal::exp(-(rate * x)))));
                if (i == 0u) scal[0] = 1.0f / (1.0f + metal::exp(-(float)b_raw[h]));   // beta
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid < \(k)u) {
                float a = qv[tid], b = kv[tid];
                float s0 = simd_sum(a * a), s1 = simd_sum(b * b), s2 = simd_sum(a * b);
                if (lane == 0u) { red[sg][0] = s0; red[sg][1] = s1; red[sg][2] = s2; }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid == 0u) {
                float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f;
                for (uint j = 0; j < \(k / 32)u; ++j) { s0 += red[j][0]; s1 += red[j][1]; s2 += red[j][2]; }
                float iq = metal::rsqrt(s0 + 1e-6f), ik = metal::rsqrt(s1 + 1e-6f);
                scal[1] = iq * \(invSqrtK)f;                // q: l2norm * K^-1/2
                scal[2] = ik;                               // k: l2norm
                scal[3] = s2 * iq * \(invSqrtK)f * ik;      // q_scaled . k_normed
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // ---- phase 2: recurrent update; thread = (value row vi, 32-key slice ks)
            const uint vi = tid % \(k)u;
            const uint ks = tid / \(k)u;
            const float qs = scal[1], kn = scal[2], beta = scal[0];
            const size_t rbase = (size_t)h * \(k * k)u + (size_t)vi * \(k)u + ks * 32u;
            float Sd[32];
            float a_part = 0.0f, b_part = 0.0f;
            for (uint j = 0; j < 32u; ++j) {
                const uint kk = ks * 32u + j;
                float s = state[rbase + j] * eg[kk];
                Sd[j] = s;
                a_part += kv[kk] * kn * s;
                b_part += qv[kk] * qs * s;
            }
            pA[ks][vi] = a_part; pB[ks][vi] = b_part;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const float kS = pA[0][vi] + pA[1][vi] + pA[2][vi] + pA[3][vi];
            const float corr = vv[vi] - kS;
            for (uint j = 0; j < 32u; ++j)
                next_state[rbase + j] = Sd[j] + beta * kv[ks * 32u + j] * kn * corr;
            if (ks == 0u) ov[vi] = (pB[0][vi] + pB[1][vi] + pB[2][vi] + pB[3][vi]) + beta * corr * scal[3];
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // ---- phase 3: gated RMSNorm over the head
            if (tid < \(k)u) {
                float o = ov[tid];
                float s = simd_sum(o * o);
                if (lane == 0u) red[sg][0] = s;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid < \(k)u) {
                float s = 0.0f;
                for (uint j = 0; j < \(k / 32)u; ++j) s += red[j][0];
                float inv = metal::rsqrt(s / \(Float(k))f + \(ep)f);
                const uint c = h * \(k)u + tid;
                float gte = (float)gate_raw[c];
                gated[c] = (T)((float)o_norm[tid] * (ov[tid] * inv) * (1.0f / (1.0f + metal::exp(-gte))));
            }
            """
        let built = MLXFast.metalKernel(
            name: name,
            inputNames: ["mixed", "conv_state", "q_w", "k_w", "v_w", "f_raw", "dt_bias", "A_log", "b_raw",
                         "gate_raw", "o_norm", "state"],
            outputNames: ["gated", "next_state", "next_conv"],
            source: source)
        kernels[name] = built
        return built
    }

    /// `VMLX_KDA_FUSED_VERIFY=0` keeps multi-row (speculative verify) forwards on the stock path.
    static let verifyEnabled: Bool = {
        let raw = (ProcessInfo.processInfo.environment["VMLX_KDA_FUSED_VERIFY"] ?? "1").lowercased()
        return !["0", "false", "off", "no"].contains(raw)
    }()
    /// Largest row count the multi-row kernel takes (speculative verify blocks; prefill keeps the chunked path).
    static let maxRows = 16

    /// The decode kernel's math looped over `rows` consecutive tokens inside one threadgroup per head (vMLX Python
    /// `glm5_kda_fused_decode` multi-row verify, c664ba85): each thread keeps its 32-key slice of the recurrent state
    /// in registers across rows, and the conv reads its window from [conv state | earlier rows]. Row r computes
    /// exactly what the one-token kernel computes for that token after rows 0 ..< r, so replaying the first n rows
    /// (verify rollback) reproduces the state after row n bit for bit.
    private static func rowsKernel(rows r: Int, heads h: Int, keyDim k: Int, width w: Int, lowerBound: Float,
                                   eps: Float) -> MLXFast.MLXFastKernel
    {
        let name = "vmlx_kda_fused_rows\(r)_h\(h)_k\(k)_w\(w)_lb\(lowerBound.bitPattern)_e\(eps.bitPattern)"
        lock.lock()
        defer { lock.unlock() }
        if let cached = kernels[name] { return cached }
        let c = h * k
        let lb = String(format: "%.9e", lowerBound)
        let ep = String(format: "%.9e", eps)
        let invSqrtK = String(format: "%.9e", 1 / Float(k).squareRoot())
        let source = """
            const uint h = threadgroup_position_in_grid.x;
            const uint tid = thread_index_in_threadgroup;
            const uint lane = thread_index_in_simdgroup;
            const uint sg = simdgroup_index_in_threadgroup;
            threadgroup float qv[\(k)], kv[\(k)], vv[\(k)], eg[\(k)];
            threadgroup float red[16][3];
            threadgroup float scal[4];
            threadgroup float pA[4][\(k)], pB[4][\(k)];
            threadgroup float ov[\(k)];
            const uint vi = tid % \(k)u;
            const uint ks = tid / \(k)u;
            const size_t rbase = (size_t)h * \(k * k)u + (size_t)vi * \(k)u + ks * 32u;
            float S[32];
            for (uint j = 0; j < 32u; ++j) S[j] = state[rbase + j];

            for (uint r = 0; r < \(r)u; ++r) {
                // ---- phase 1: conv over [conv state | rows], decay gate, beta
                if (tid < \(3 * k)u) {
                    const uint which = tid / \(k)u;
                    const uint i = tid % \(k)u;
                    const uint c = h * \(k)u + i;
                    const uint ch = which * \(c)u + c;
                    device const T* wt = which == 0u ? q_w : (which == 1u ? k_w : v_w);
                    float acc = 0.0f;
                    for (uint tap = 0; tap < \(w)u; ++tap) {
                        const uint idx = r + tap;           // index into [conv state (W-1 rows) | rows]
                        const float xv = idx < \(w - 1)u
                            ? (float)conv_state[(size_t)idx * \(3 * c)u + ch]
                            : (float)mixed[(size_t)(idx - \(w - 1)u) * \(3 * c)u + ch];
                        acc += xv * (float)wt[(size_t)c * \(w)u + tap];
                    }
                    float y = (float)(T)(acc / (1.0f + metal::exp(-acc)));
                    if (which == 0u) qv[i] = y; else if (which == 1u) kv[i] = y; else vv[i] = y;
                } else {
                    const uint i = tid - \(3 * k)u;
                    const uint c = h * \(k)u + i;
                    float x = (float)f_raw[(size_t)r * \(c)u + c] + (float)dt_bias[c];
                    float rate = metal::exp((float)A_log[h]);
                    eg[i] = metal::exp(\(lb)f * (1.0f / (1.0f + metal::exp(-(rate * x)))));
                    if (i == 0u) scal[0] = 1.0f / (1.0f + metal::exp(-(float)b_raw[r * \(h)u + h]));
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (tid < \(k)u) {
                    float a = qv[tid], b = kv[tid];
                    float s0 = simd_sum(a * a), s1 = simd_sum(b * b), s2 = simd_sum(a * b);
                    if (lane == 0u) { red[sg][0] = s0; red[sg][1] = s1; red[sg][2] = s2; }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (tid == 0u) {
                    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f;
                    for (uint j = 0; j < \(k / 32)u; ++j) { s0 += red[j][0]; s1 += red[j][1]; s2 += red[j][2]; }
                    float iq = metal::rsqrt(s0 + 1e-6f), ik = metal::rsqrt(s1 + 1e-6f);
                    scal[1] = iq * \(invSqrtK)f;
                    scal[2] = ik;
                    scal[3] = s2 * iq * \(invSqrtK)f * ik;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                // ---- phase 2: recurrent update, state in registers
                const float qs = scal[1], kn = scal[2], beta = scal[0];
                float a_part = 0.0f, b_part = 0.0f;
                for (uint j = 0; j < 32u; ++j) {
                    const uint kk = ks * 32u + j;
                    float s = S[j] * eg[kk];
                    S[j] = s;
                    a_part += kv[kk] * kn * s;
                    b_part += qv[kk] * qs * s;
                }
                pA[ks][vi] = a_part; pB[ks][vi] = b_part;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                const float kS = pA[0][vi] + pA[1][vi] + pA[2][vi] + pA[3][vi];
                const float corr = vv[vi] - kS;
                for (uint j = 0; j < 32u; ++j)
                    S[j] = S[j] + beta * kv[ks * 32u + j] * kn * corr;
                if (ks == 0u) ov[vi] = (pB[0][vi] + pB[1][vi] + pB[2][vi] + pB[3][vi]) + beta * corr * scal[3];
                threadgroup_barrier(mem_flags::mem_threadgroup);

                // ---- phase 3: gated RMSNorm over the head
                if (tid < \(k)u) {
                    float o = ov[tid];
                    float s = simd_sum(o * o);
                    if (lane == 0u) red[sg][0] = s;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (tid < \(k)u) {
                    float s = 0.0f;
                    for (uint j = 0; j < \(k / 32)u; ++j) s += red[j][0];
                    float inv = metal::rsqrt(s / \(Float(k))f + \(ep)f);
                    const uint c = h * \(k)u + tid;
                    float gte = (float)gate_raw[(size_t)r * \(c)u + c];
                    gated[(size_t)r * \(c)u + c] =
                        (T)((float)o_norm[tid] * (ov[tid] * inv) * (1.0f / (1.0f + metal::exp(-gte))));
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            for (uint j = 0; j < 32u; ++j) next_state[rbase + j] = S[j];
            // conv state after the rows: the last W-1 rows of [conv state | rows]
            if (tid < \(3 * k)u) {
                const uint which = tid / \(k)u;
                const uint ch = which * \(c)u + h * \(k)u + tid % \(k)u;
                for (uint j = 0; j < \(w - 1)u; ++j) {
                    const uint idx = \(r)u + j;
                    next_conv[(size_t)j * \(3 * c)u + ch] = idx < \(w - 1)u
                        ? conv_state[(size_t)idx * \(3 * c)u + ch]
                        : mixed[(size_t)(idx - \(w - 1)u) * \(3 * c)u + ch];
                }
            }
            """
        let built = MLXFast.metalKernel(
            name: name,
            inputNames: ["mixed", "conv_state", "q_w", "k_w", "v_w", "f_raw", "dt_bias", "A_log", "b_raw",
                         "gate_raw", "o_norm", "state"],
            outputNames: ["gated", "next_state", "next_conv"],
            source: source)
        kernels[name] = built
        return built
    }

    /// Multi-row twin of ``run``: `mixed` `[1, R, 3C]`, `fRaw`/`gateRaw` `[1, R, C]`, `bRaw` `[1, R, H]`, 2 <= R <=
    /// ``maxRows``. Returns (gated `[1, R, C]`, state after row R-1, conv state after row R-1) or nil (stock path).
    static func runRows(
        mixed: MLXArray, convState: MLXArray, qWeight: MLXArray, kWeight: MLXArray, vWeight: MLXArray,
        fRaw: MLXArray, dtBias: MLXArray, aLog: MLXArray, bRaw: MLXArray, gateRaw: MLXArray,
        oNormWeight: MLXArray, eps: Float, state: MLXArray, heads: Int, keyDim: Int, lowerBound: Float
    ) -> (gated: MLXArray, state: MLXArray, conv: MLXArray)? {
        let c = heads * keyDim
        guard mixed.ndim == 3 else { return nil }
        let rows = mixed.dim(1)
        guard verifyEnabled, keyDim == 128, rows >= 1, rows <= maxRows, mixed.shape == [1, rows, 3 * c],
            [.bfloat16, .float16].contains(mixed.dtype),
            state.dtype == .float32, state.shape == [1, heads, keyDim, keyDim],
            convState.ndim == 3, convState.dim(0) == 1, convState.dim(2) == 3 * c,
            convState.dtype == mixed.dtype
        else { return nil }
        let width = convState.dim(1) + 1
        func taps(_ w: MLXArray) -> MLXArray? {
            guard w.size == c * width else { return nil }
            return w.reshaped(c, width)
        }
        guard width >= 2, let wq = taps(qWeight), let wk = taps(kWeight), let wv = taps(vWeight),
            fRaw.size == rows * c, gateRaw.size == rows * c, dtBias.size == c, bRaw.size == rows * heads,
            aLog.size == heads, oNormWeight.size == keyDim
        else { return nil }
        let out = rowsKernel(rows: rows, heads: heads, keyDim: keyDim, width: width, lowerBound: lowerBound,
                             eps: eps)(
            [mixed, convState, wq, wk, wv, fRaw.reshaped(rows * c), dtBias.reshaped(c), aLog,
             bRaw.reshaped(rows * heads), gateRaw.reshaped(rows * c), oNormWeight, state],
            template: [("T", mixed.dtype)],
            grid: (heads * 512, 1, 1), threadGroup: (512, 1, 1),
            outputShapes: [[1, rows, c], state.shape, convState.shape],
            outputDTypes: [mixed.dtype, .float32, convState.dtype])
        return (out[0], out[1], out[2])
    }

    /// Returns (gated `[1, 1, C]` in `mixed.dtype`, next recurrent state, next conv state), or nil when the shapes,
    /// dtypes or geometry are not the ones this kernel handles (the caller keeps the stock path).
    static func run(
        mixed: MLXArray, convState: MLXArray, qWeight: MLXArray, kWeight: MLXArray, vWeight: MLXArray,
        fRaw: MLXArray, dtBias: MLXArray, aLog: MLXArray, bRaw: MLXArray, gateRaw: MLXArray,
        oNormWeight: MLXArray, eps: Float, state: MLXArray, heads: Int, keyDim: Int, lowerBound: Float
    ) -> (gated: MLXArray, state: MLXArray, conv: MLXArray)? {
        let c = heads * keyDim
        guard enabled, keyDim == 128, mixed.shape == [1, 1, 3 * c],
            [.bfloat16, .float16].contains(mixed.dtype),
            state.dtype == .float32, state.shape == [1, heads, keyDim, keyDim],
            convState.ndim == 3, convState.dim(0) == 1, convState.dim(2) == 3 * c,
            convState.dtype == mixed.dtype
        else { return nil }
        let width = convState.dim(1) + 1
        func taps(_ w: MLXArray) -> MLXArray? {
            guard w.size == c * width else { return nil }
            return w.reshaped(c, width)
        }
        guard width >= 2, let wq = taps(qWeight), let wk = taps(kWeight), let wv = taps(vWeight),
            fRaw.size == c, gateRaw.size == c, dtBias.size == c, bRaw.size == heads, aLog.size == heads,
            oNormWeight.size == keyDim
        else { return nil }
        let out = kernel(heads: heads, keyDim: keyDim, width: width, lowerBound: lowerBound, eps: eps)(
            [mixed, convState, wq, wk, wv, fRaw.reshaped(c), dtBias.reshaped(c), aLog, bRaw.reshaped(heads),
             gateRaw.reshaped(c), oNormWeight, state],
            template: [("T", mixed.dtype)],
            grid: (heads * 512, 1, 1), threadGroup: (512, 1, 1),
            outputShapes: [[1, 1, c], state.shape, convState.shape],
            outputDTypes: [mixed.dtype, .float32, convState.dtype])
        return (out[0], out[1], out[2])
    }
}
