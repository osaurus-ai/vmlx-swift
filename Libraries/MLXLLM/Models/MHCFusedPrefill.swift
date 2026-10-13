// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXFast

/// mHC collapse and expand for prefill rows (4 hyper-connection streams, batch 1, more than
/// ``MHCFusedDecode/maxRows`` rows). Port of vMLX Python `metal/glm5_mhc_prefill_fused.py` (76f03621 + e7a12a5a:
/// GLM-5.3 4k cold prefill +8.0 %), written to this engine's own stock arithmetic (`DeepseekV4Math.hcPreGraph` /
/// `hcPost`, fp32 throughout, the collapse cast once to the activation dtype) rather than the reference's.
///
/// The stock graph per call (90 per chunk): an fp32 copy of the 4 streams, an fp32 RMS, a (T x 4D) @ (4D x 24)
/// fp32 matmul (TF32 under MLX defaults), the Sinkhorn kernel, the collapse; expand = an fp32 broadcast multiply
/// plus a per-token batched (4 x 4)^T @ (4 x D) matmul. Here:
/// - collapse: one 512-thread threadgroup per TWO tokens (each `fn` element is loaded once for both): RMS, the 24
///   projections in true fp32, the decode kernel's epilogue formulas (sigmoid pre/post, 4 x 4 Sinkhorn on SIMD
///   lanes), the pre-weighted collapse;
/// - expand: one thread per (token, d): `T(post[s] * out[d] + sum_r comb[r, s] * residual[r, d])` in fp32.
/// `VMLX_MHC_FUSED_PREFILL=0` keeps the graph.
public enum MHCFusedPrefill {
    public static let enabled: Bool = {
        let raw = (ProcessInfo.processInfo.environment["VMLX_MHC_FUSED_PREFILL"] ?? "1").lowercased()
        return !["0", "false", "off", "no"].contains(raw)
    }()

    static let threads = 512
    static let tokensPerGroup = 2

    private static let lock = NSLock()
    nonisolated(unsafe) private static var mixKernels: [String: MLXFast.MLXFastKernel] = [:]

    private static func mixKernel(d: Int, iters: Int) -> MLXFast.MLXFastKernel {
        let name = "vmlx_mhc_prefill_mix_d\(d)_i\(iters)"
        lock.lock()
        defer { lock.unlock() }
        if let k = mixKernels[name] { return k }
        let th = threads, tok = tokensPerGroup, f = 4 * d, per = f / th, dper = d / th
        let source = """
            const uint t0 = threadgroup_position_in_grid.x * \(tok)u;
            const uint tid = thread_index_in_threadgroup;
            const uint lane = thread_index_in_simdgroup;
            const uint sg = simdgroup_index_in_threadgroup;
            const uint nr = (uint)n_rows[0];
            threadgroup float red[\(tok)][\(th / 32)][24];
            threadgroup float mixv[\(tok)][24];
            threadgroup float pre_v[\(tok)][4], post_v[\(tok)][4], mat[\(tok)][16];
            threadgroup float scal[\(tok)];
            float v[\(tok)][\(per)];
            for (uint k = 0; k < \(tok)u; ++k) {
                const bool ok = t0 + k < nr;
                device const T* st = streams + (size_t)(ok ? t0 + k : 0u) * \(f)u;
                float ss = 0.0f;
                for (uint j = 0; j < \(per)u; ++j) {
                    v[k][j] = ok ? (float)st[tid + j * \(th)u] : 0.0f;
                    ss += v[k][j] * v[k][j];
                }
                ss = simd_sum(ss);
                if (lane == 0u) red[k][sg][0] = ss;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid < \(tok)u) {
                float tot = 0.0f;
                for (uint g = 0; g < \(th / 32)u; ++g) tot += red[tid][g][0];
                scal[tid] = metal::rsqrt(tot / \(Float(f))f + rms_eps[0]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float acc[\(tok)][24];
            for (uint k = 0; k < \(tok)u; ++k) for (uint r = 0; r < 24u; ++r) acc[k][r] = 0.0f;
            for (uint j = 0; j < \(per)u; ++j) {
                const uint fi = tid + j * \(th)u;
                for (uint r = 0; r < 24u; ++r) {
                    const float w = (float)hc_fn[(size_t)r * \(f)u + fi];
                    for (uint k = 0; k < \(tok)u; ++k) acc[k][r] += v[k][j] * w;
                }
            }
            for (uint k = 0; k < \(tok)u; ++k) for (uint r = 0; r < 24u; ++r) {
                float p = simd_sum(acc[k][r]);
                if (lane == 0u) red[k][sg][r] = p;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid < \(tok * 24)u) {
                const uint k = tid / 24u, r = tid % 24u;
                float tot = 0.0f;
                for (uint g = 0; g < \(th / 32)u; ++g) tot += red[k][g][r];
                mixv[k][r] = tot * scal[k];           // stock: (x @ fn^T) * rsqrt(mean(x^2) + eps)
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sg < \(tok)u) {
                const uint k = sg;
                const float s0 = (float)hc_scale[0], s1 = (float)hc_scale[1], s2 = (float)hc_scale[2];
                const float eps = sink_eps[0];
                if (lane < 4u) {
                    pre_v[k][lane] = 1.0f / (1.0f + metal::exp(-(mixv[k][lane] * s0 + (float)hc_base[lane]))) + eps;
                    post_v[k][lane] = 2.0f / (1.0f + metal::exp(-(mixv[k][4u + lane] * s1 + (float)hc_base[4u + lane])));
                }
                const uint e = lane & 15u;
                float item = mixv[k][8u + e] * s2 + (float)hc_base[8u + e];
                float m = metal::max(item, simd_shuffle_xor(item, 1u)); m = metal::max(m, simd_shuffle_xor(m, 2u));
                float ex = metal::exp(item - m);
                float rs = ex + simd_shuffle_xor(ex, 1u); rs = rs + simd_shuffle_xor(rs, 2u);
                float c = ex / rs + eps;
                float cs = c + simd_shuffle_xor(c, 4u); cs = cs + simd_shuffle_xor(cs, 8u);
                c = c / (cs + eps);
                for (uint it = 1u; it < \(iters)u; ++it) {
                    rs = c + simd_shuffle_xor(c, 1u); rs = rs + simd_shuffle_xor(rs, 2u); c = c / (rs + eps);
                    cs = c + simd_shuffle_xor(c, 4u); cs = cs + simd_shuffle_xor(cs, 8u); c = c / (cs + eps);
                }
                if (lane < 16u) mat[k][lane] = c;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint k = 0; k < \(tok)u; ++k) {
                const uint t = t0 + k;
                if (t >= nr) continue;
                if (tid < 4u) post[(size_t)t * 4u + tid] = post_v[k][tid];
                if (tid < 16u) comb[(size_t)t * 16u + tid] = mat[k][tid];
                device const T* st = streams + (size_t)t * \(f)u;
                for (uint j = 0; j < \(dper)u; ++j) {
                    const uint dd = tid + j * \(th)u;
                    float a = 0.0f;
                    for (uint s = 0; s < 4u; ++s) a += pre_v[k][s] * (float)st[s * \(d)u + dd];
                    collapsed[(size_t)t * \(d)u + dd] = (T)a;
                }
            }
            """
        let k = MLXFast.metalKernel(
            name: name,
            inputNames: ["streams", "hc_fn", "hc_base", "hc_scale", "rms_eps", "sink_eps", "n_rows"],
            outputNames: ["post", "comb", "collapsed"], source: source)
        mixKernels[name] = k
        return k
    }

    private static let placeKernel = MLXFast.metalKernel(
        name: "vmlx_mhc_prefill_place",
        inputNames: ["post", "comb", "out", "residual", "n_rows"], outputNames: ["new_streams"],
        source: """
            const uint gid = thread_position_in_grid.x;
            const uint t = gid / D, d = gid % D;
            if (t >= (uint)n_rows[0]) return;
            const float o = (float)out[(size_t)t * D + d];
            float r[4];
            for (uint q = 0; q < 4u; ++q) r[q] = (float)residual[((size_t)t * 4u + q) * D + d];
            for (uint s = 0; s < 4u; ++s) {
                float m = 0.0f;
                for (uint q = 0; q < 4u; ++q) m += comb[((size_t)t * 4u + q) * 4u + s] * r[q];
                new_streams[((size_t)t * 4u + s) * D + d] = (T)(post[(size_t)t * 4u + s] * o + m);
            }
            """)

    /// `(collapsed, post, comb)` for `streams` `[1, L, 4, D]` with L > ``MHCFusedDecode/maxRows``, or nil.
    public static func collapse(
        _ streams: MLXArray, fn: MLXArray, scale: MLXArray, base: MLXArray, iters: Int, eps: Float, normEps: Float
    ) -> (x: MLXArray, post: MLXArray, comb: MLXArray)? {
        guard enabled, streams.ndim == 4, streams.dim(0) == 1, streams.dim(2) == 4,
            streams.dim(1) > MHCFusedDecode.maxRows, iters >= 1, iters <= 64,
            [.bfloat16, .float16].contains(streams.dtype)
        else { return nil }
        let rows = streams.dim(1), d = streams.dim(3)
        guard d % threads == 0, fn.shape == [24, 4 * d], base.shape == [24], scale.shape == [3] else { return nil }
        let groups = (rows + tokensPerGroup - 1) / tokensPerGroup
        let out = mixKernel(d: d, iters: iters)(
            [streams, fn, base, scale, MLXArray([normEps]), MLXArray([eps]), MLXArray([Int32(rows)])],
            template: [("T", streams.dtype)],
            grid: (groups * threads, 1, 1), threadGroup: (threads, 1, 1),
            outputShapes: [[1, rows, 4], [1, rows, 4, 4], [1, rows, d]],
            outputDTypes: [.float32, .float32, streams.dtype])
        return (x: out[2], post: out[0], comb: out[1])
    }

    /// Expand for prefill rows: `blockOut` `[1, L, D]`, `residual` `[1, L, 4, D]`, `post` `[1, L, 4]` f32,
    /// `comb` `[1, L, 4, 4]` f32. Nil for any other geometry.
    public static func expand(blockOut: MLXArray, residual: MLXArray, post: MLXArray, comb: MLXArray) -> MLXArray? {
        guard enabled, residual.ndim == 4, residual.dim(0) == 1, residual.dim(2) == 4,
            residual.dim(1) > MHCFusedDecode.maxRows, blockOut.dtype == residual.dtype,
            [.bfloat16, .float16].contains(residual.dtype), post.dtype == .float32, comb.dtype == .float32
        else { return nil }
        let rows = residual.dim(1), d = residual.dim(3)
        guard blockOut.shape == [1, rows, d], post.shape == [1, rows, 4], comb.shape == [1, rows, 4, 4] else {
            return nil
        }
        return placeKernel(
            [post, comb, blockOut, residual, MLXArray([Int32(rows)])],
            template: [("T", residual.dtype), ("D", d)],
            grid: (rows * d, 1, 1), threadGroup: (256, 1, 1),
            outputShapes: [[1, rows, 4, d]], outputDTypes: [residual.dtype])[0]
    }
}
