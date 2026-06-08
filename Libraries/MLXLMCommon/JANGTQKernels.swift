//
// JANGTQ Metal kernels — Swift port of jang-tools/jang_tools/turboquant.
// Created by Jinho Jang (eric@jangq.ai).
//
// The kernel source strings here mirror the Python kernels that have
// been validated end-to-end on MiniMax M2.7 JANGTQ_2L:
//
//   ../../../../jang/jang-tools/jang_tools/turboquant/hadamard_kernel.py
//   ../../../../jang/jang-tools/jang_tools/turboquant/fused_gate_up_kernel.py
//   ../../../../jang/jang-tools/jang_tools/turboquant/gather_tq_kernel.py
//
// Because we use `MLXFast.metalKernel(...)` (which calls the same
// `mlx_fast_metal_kernel_*` C++ entry points as Python's
// `mx.fast.metal_kernel`), Swift and Python compile through the same
// MLX fast-kernel path. Runtime parity still has to be proven with
// model-level decode rows, not inferred from source shape alone.
//
// What this gives us:
//   - The JANGTQ hot kernels stay in the same MLX fast-kernel family
//     as the Python runtime, including the <=1024-block SIMD-shuffle
//     Hadamard path used by MiniMax's 1536-dim intermediate rotation.
//   - Source-level kernel parity is a prerequisite only; production
//     claims still require coherent decode telemetry with token/s and
//     Activity Monitor memory pressure.
//
// Sweet-spot tile constants (P17, M3 Ultra sweep):
//   - jangtq_fused_gate_up_swiglu : OPT = 10 outputs per thread
//   - jangtq_gather_tq_matmul     : OPT = 20 outputs per thread
//

import Foundation
import MLX

private struct JANGTQMetaCacheKey: Hashable {
    let kind: String
    let values: [UInt32]
}

private nonisolated(unsafe) var jangtqMetaCache: [JANGTQMetaCacheKey: MLXArray] = [:]
private let jangtqMetaCacheLock = NSLock()

private func cachedJANGTQMeta(kind: String, values: [UInt32]) -> MLXArray {
    let key = JANGTQMetaCacheKey(kind: kind, values: values)
    jangtqMetaCacheLock.lock()
    if let cached = jangtqMetaCache[key] {
        jangtqMetaCacheLock.unlock()
        return cached
    }
    let meta = MLXArray(values)
    jangtqMetaCache[key] = meta
    jangtqMetaCacheLock.unlock()
    return meta
}

// MARK: - Hadamard multiblock

private let kHadamardMultiblockSource = """
    uint batch_idx = thread_position_in_grid.y;
    uint tid = thread_position_in_threadgroup.x;
    uint threads_per_tg = threads_per_threadgroup.x;

    uint total_d = meta[0];
    uint n_blocks = meta[1];

    // Apple Silicon caps threadgroup memory at 32 KB = 8192 floats. The
    // largest single power-of-2 block we ever decompose into is 8192
    // (e.g., Mistral-Medium-3.5 hidden=12288 → [8192, 4096]; GLM-5.1
    // hidden=6144 → [4096, 2048]; Kimi-K2.6 hidden=7168 → [4096, 2048,
    // 1024]). The shmem only needs to hold ONE block at a time —
    // butterflies are independent per block, and the output is written
    // to global memory before the next block is loaded.
    //
    // Earlier versions of this kernel loaded the entire `total_d` slab
    // into shmem up-front (the original Python prototype's design when
    // total_d ≤ 8192). On Mistral 3.5 (total_d=12288 > 8192) that
    // overran the buffer by 4096 entries, silently corrupting half the
    // rotated activations. Diagnosed via VMLX_MISTRAL3_PROJ_PROBE=1:
    // layer-0 V projection L2 was 4.3× the mxfp4 baseline; after this
    // rewrite it sits within 1.5× (residual 2-bit quant noise).
    //
    // Per-block isolation also matches the Python reference's
    // gather_tq_kernel.py (templated `threadgroup float shmem[in_features]`
    // approach) — they similarly never need >8192 floats since they
    // fuse Hadamard+gather and shmem only holds the current block's
    // post-rotation values.
    threadgroup float shmem[8192];

    uint cum_offset = 0;
    for (uint b = 0; b < n_blocks; b++) {
        uint d_b = meta[2u + b * 2u];
        uint log_b = meta[3u + b * 2u];

        // Load this block's slice of (x*signs) into shmem[0..d_b].
        for (uint i = tid; i < d_b; i += threads_per_tg) {
            shmem[i] = static_cast<float>(x[batch_idx * total_d + cum_offset + i])
                       * signs[cum_offset + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint ept = (d_b + threads_per_tg - 1u) / threads_per_tg;
        if (ept == 0u) ept = 1u;

        for (uint stage = 0; stage < log_b; stage++) {
            uint h = 1u << stage;
            uint two_h = 2u * h;

            // Stack buffer per thread for butterfly-stage values. The
            // launcher uses tgSize=min(1024, max(32, maxBlock)), so the
            // maximum in-shmem 8192 block needs at most 8 entries/thread.
            // Keeping this at 8 avoids register pressure on MiniMax-sized
            // JANGTQ decode while preserving the Mistral 3.5 8192-block fix.
            float newv[8];
            for (uint k = 0; k < 8; k++) newv[k] = 0.0f;
            for (uint k = 0; k < ept; k++) {
                uint i_local = tid * ept + k;
                if (i_local < d_b) {
                    uint block_start = (i_local / two_h) * two_h;
                    uint pos = i_local - block_start;
                    float a = shmem[block_start + pos];
                    if (pos < h) {
                        newv[k] = a + shmem[block_start + pos + h];
                    } else {
                        newv[k] = shmem[block_start + pos - h] - a;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint k = 0; k < ept; k++) {
                uint i_local = tid * ept + k;
                if (i_local < d_b) {
                    shmem[i_local] = newv[k];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        float norm = 1.0f / sqrt(static_cast<float>(d_b));
        for (uint i = tid; i < d_b; i += threads_per_tg) {
            out[batch_idx * total_d + cum_offset + i] = shmem[i] * norm;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        cum_offset += d_b;
    }
"""

// MiniMax decode hot path: the intermediate activation rotation is 1536 =
// 1024 + 512 and runs over the active experts. For <=1024 blocks, the
// first five butterfly stages stay inside one SIMD group; lane shuffles
// avoid the threadgroup-memory round trip and barriers for those stages.
private let kHadamardShuffleLE1024Source = """
    uint batch_idx = thread_position_in_grid.y;
    uint tid = thread_position_in_threadgroup.x;

    uint total_d = meta[0];
    uint n_blocks = meta[1];

    threadgroup float shmem[1024];

    uint offset = 0u;
    for (uint b = 0u; b < n_blocks; b++) {
        uint d_b = meta[2u + b * 2u];
        uint log_b = meta[3u + b * 2u];

        float v = 0.0f;
        if (tid < d_b) {
            v = static_cast<float>(x[batch_idx * total_d + offset + tid])
                * signs[offset + tid];
        }

        uint lane = tid & 31u;
        uint simd_stages = log_b < 5u ? log_b : 5u;
        for (uint stage = 0u; stage < simd_stages; stage++) {
            uint h = 1u << stage;
            float other = simd_shuffle_xor(v, h);
            if ((lane & h) == 0u) { v = v + other; }
            else { v = other - v; }
        }

        for (uint stage = simd_stages; stage < log_b; stage++) {
            uint h = 1u << stage;
            if (tid < d_b) { shmem[tid] = v; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid < d_b) {
                float other = shmem[tid ^ h];
                if ((tid & h) == 0u) { v = v + other; }
                else { v = other - v; }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        float norm = 1.0f / sqrt(static_cast<float>(d_b));
        if (tid < d_b) {
            out[batch_idx * total_d + offset + tid] = v * norm;
        }

        offset += d_b;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
"""

// MARK: - Fused gate+up+SwiGLU (P17 OPT=10)

private let kFusedSwiGLUSource = """
    uint global_x = thread_position_in_grid.x;
    uint dispatch_idx = thread_position_in_grid.y;

    uint out_group = global_x / 32u;
    uint lane = global_x % 32u;
    uint out_idx_0 = out_group * 10u;

    uint K = meta[0];
    uint in_features = meta[1];
    uint out_features = meta[2];
    uint packed_cols = meta[3];
    uint bits = meta[4];
    // 2026-05-04 (DSV4 SWA/CSA/HSA correctness pass):
    // `meta[5]` carries the SwiGLU clamp magnitude × 1000 as uint, so
    //   swiglu_limit = float(meta[5]) / 1000.0
    // A value of 0 disables the clamp (ordinary SwiGLU). DeepSeek-V4
    // sets this to 10000 → limit = 10.0, matching the codex_dsv4_fixkit
    // reference. Other models pass 0 → no clamp → byte-identical to the
    // pre-2026-05-04 kernel output.
    uint swiglu_limit_q1000 = meta[5];
    float swiglu_limit = static_cast<float>(swiglu_limit_q1000) * 0.001f;

    if (out_idx_0 >= out_features) return;

    uint token_idx = dispatch_idx / K;
    uint k_idx = dispatch_idx % K;
    uint expert = rhs_indices[token_idx * K + k_idx];

    uint vals_per_u32 = 32u / bits;
    uint mask = (1u << bits) - 1u;

    float acc_g[10];
    float acc_u[10];
    #pragma unroll
    for (uint o = 0; o < 10; o++) { acc_g[o] = 0.0f; acc_u[o] = 0.0f; }

    uint expert_base = expert * out_features * packed_cols;
    uint x_off = token_idx * in_features;

    uint n_outs = 10u;
    if (out_idx_0 + 10u > out_features) n_outs = out_features - out_idx_0;

    for (uint pack_idx = lane; pack_idx < packed_cols; pack_idx += 32u) {
        uint i_base = pack_idx * vals_per_u32;

        uint pvg[10], pvu[10];
        #pragma unroll
        for (uint o = 0; o < 10; o++) {
            if (o < n_outs) {
                uint row_off = expert_base + (out_idx_0 + o) * packed_cols + pack_idx;
                pvg[o] = packed_gate[row_off];
                pvu[o] = packed_up[row_off];
            } else {
                pvg[o] = 0u;
                pvu[o] = 0u;
            }
        }

        // 2026-04-26: loop bound MUST be `vals_per_u32` (= 32 / bits)
        // not the hardcoded 16. Correct for bits=2 (vals_per_u32=16)
        // by coincidence but for bits=4 (vals_per_u32=8) the old
        // hardcoded 16 walked PAST the end of each packed uint32 by
        // 8 iterations: shifts 32-60 are out-of-range for uint32
        // right-shift (Metal undefined behaviour), AND `i = i_base + k`
        // for k=8..15 reads input values that belong to the NEXT
        // pack_idx — corrupting both accumulators. Reproduces as
        // garbage multilingual gibberish on Holo3-35B-A3B-JANGTQ4 and
        // Qwen3.6-35B-A3B-JANGTQ4 with suspiciously fast decode rates
        // (compute is short-circuited / corrupted, not skipped).
        // See research/QWEN36-A3B-JANGTQ4-COHERENCE-BUG-2026-04-25.md.
        for (uint k = 0; k < vals_per_u32; k++) {
            uint i = i_base + k;
            if (i >= in_features) break;
            float xv = static_cast<float>(x_rot[x_off + i]);
            uint shift = k * bits;
            #pragma unroll
            for (uint o = 0; o < 10; o++) {
                float w_g = codebook[(pvg[o] >> shift) & mask];
                float w_u = codebook[(pvu[o] >> shift) & mask];
                acc_g[o] += xv * w_g;
                acc_u[o] += xv * w_u;
            }
        }
    }

    #pragma unroll
    for (uint o = 0; o < 10; o++) {
        acc_g[o] = simd_sum(acc_g[o]);
        acc_u[o] = simd_sum(acc_u[o]);
    }

    if (lane == 0) {
        uint base_off = (token_idx * K + k_idx) * out_features;
        for (uint o = 0; o < n_outs; o++) {
            uint oi = out_idx_0 + o;
            float ng = static_cast<float>(norms_gate[expert * out_features + oi]);
            float nu = static_cast<float>(norms_up[expert * out_features + oi]);
            float gv = acc_g[o] * ng;
            float uv = acc_u[o] * nu;
            // 2026-05-04: optional DSV4-style limited SwiGLU clamp.
            //   gate = min(gate, +limit)        (one-sided)
            //   up   = clamp(up,  -limit, +limit) (two-sided)
            //   y    = silu(gate) * up
            // When `swiglu_limit == 0` (every non-DSV4 caller), this
            // collapses to the original ordinary SwiGLU expression
            // exactly. See codex_dsv4_fixkit/scripts/runtime_dsv4_fixed.py
            // and jang_tools/dsv4/mlx_model.py:_dsv4_swiglu.
            if (swiglu_limit > 0.0f) {
                gv = metal::min(gv, swiglu_limit);
                uv = metal::max(metal::min(uv, swiglu_limit), -swiglu_limit);
            }
            out_act[base_off + oi] = (gv / (1.0f + metal::fast::exp(-gv))) * uv;
        }
    }
"""

private let kFusedSwiGLUOffsetsSource = """
    uint global_x = thread_position_in_grid.x;
    uint dispatch_idx = thread_position_in_grid.y;

    uint out_group = global_x / 32u;
    uint lane = global_x % 32u;
    uint out_idx_0 = out_group * 10u;

    uint K = meta[0];
    uint in_features = meta[1];
    uint out_features = meta[2];
    uint packed_cols = meta[3];
    uint bits = meta[4];
    uint swiglu_limit_q1000 = meta[5];
    float swiglu_limit = static_cast<float>(swiglu_limit_q1000) * 0.001f;

    if (out_idx_0 >= out_features) return;

    uint token_idx = dispatch_idx / K;
    uint k_idx = dispatch_idx % K;
    uint expert = rhs_indices[token_idx * K + k_idx];

    uint vals_per_u32 = 32u / bits;
    uint mask = (1u << bits) - 1u;

    float acc_g[10];
    float acc_u[10];
    #pragma unroll
    for (uint o = 0; o < 10; o++) { acc_g[o] = 0.0f; acc_u[o] = 0.0f; }

    uint gate_base = packed_gate_offsets[expert];
    uint up_base = packed_up_offsets[expert];
    uint x_off = token_idx * in_features;

    uint n_outs = 10u;
    if (out_idx_0 + 10u > out_features) n_outs = out_features - out_idx_0;

    if (gate_base == 0xffffffffu || up_base == 0xffffffffu
        || norms_gate_offsets[expert] == 0xffffffffu
        || norms_up_offsets[expert] == 0xffffffffu) {
        if (lane == 0) {
            uint base_off = (token_idx * K + k_idx) * out_features;
            for (uint o = 0; o < n_outs; o++) {
                out_act[base_off + out_idx_0 + o] = 0.0f;
            }
        }
        return;
    }

    for (uint pack_idx = lane; pack_idx < packed_cols; pack_idx += 32u) {
        uint i_base = pack_idx * vals_per_u32;

        uint pvg[10], pvu[10];
        #pragma unroll
        for (uint o = 0; o < 10; o++) {
            if (o < n_outs) {
                pvg[o] = packed_gate[gate_base + (out_idx_0 + o) * packed_cols + pack_idx];
                pvu[o] = packed_up[up_base + (out_idx_0 + o) * packed_cols + pack_idx];
            } else {
                pvg[o] = 0u;
                pvu[o] = 0u;
            }
        }

        for (uint k = 0; k < vals_per_u32; k++) {
            uint i = i_base + k;
            if (i >= in_features) break;
            float xv = static_cast<float>(x_rot[x_off + i]);
            uint shift = k * bits;
            #pragma unroll
            for (uint o = 0; o < 10; o++) {
                float w_g = codebook[(pvg[o] >> shift) & mask];
                float w_u = codebook[(pvu[o] >> shift) & mask];
                acc_g[o] += xv * w_g;
                acc_u[o] += xv * w_u;
            }
        }
    }

    #pragma unroll
    for (uint o = 0; o < 10; o++) {
        acc_g[o] = simd_sum(acc_g[o]);
        acc_u[o] = simd_sum(acc_u[o]);
    }

    if (lane == 0) {
        uint base_off = (token_idx * K + k_idx) * out_features;
        uint gate_norm_base = norms_gate_offsets[expert];
        uint up_norm_base = norms_up_offsets[expert];
        for (uint o = 0; o < n_outs; o++) {
            uint oi = out_idx_0 + o;
            float ng = static_cast<float>(norms_gate[gate_norm_base + oi]);
            float nu = static_cast<float>(norms_up[up_norm_base + oi]);
            float gv = acc_g[o] * ng;
            float uv = acc_u[o] * nu;
            if (swiglu_limit > 0.0f) {
                gv = metal::min(gv, swiglu_limit);
                uv = metal::max(metal::min(uv, swiglu_limit), -swiglu_limit);
            }
            out_act[base_off + oi] = (gv / (1.0f + metal::fast::exp(-gv))) * uv;
        }
    }
"""

// MARK: - Gather TQ matmul (P17 OPT=20)

private let kGatherTQSource = """
    uint global_x = thread_position_in_grid.x;
    uint dispatch_idx = thread_position_in_grid.y;

    uint out_group = global_x / 32u;
    uint lane = global_x % 32u;
    uint out_idx_0 = out_group * 20u;

    uint K = meta[0];
    uint in_features = meta[1];
    uint out_features = meta[2];
    uint packed_cols = meta[3];
    uint bits = meta[4];

    if (out_idx_0 >= out_features) return;

    uint token_idx = dispatch_idx / K;
    uint k_idx = dispatch_idx % K;
    uint expert = rhs_indices[token_idx * K + k_idx];

    uint vals_per_u32 = 32u / bits;
    uint mask = (1u << bits) - 1u;

    float acc[20];
    #pragma unroll
    for (uint o = 0; o < 20; o++) acc[o] = 0.0f;

    uint expert_base = expert * out_features * packed_cols;
    uint x_offset = token_idx * in_features;

    uint n_outs = 20u;
    if (out_idx_0 + 20u > out_features) n_outs = out_features - out_idx_0;

    for (uint pack_idx = lane; pack_idx < packed_cols; pack_idx += 32u) {
        uint i_base = pack_idx * vals_per_u32;
        uint pv[20];
        #pragma unroll
        for (uint o = 0; o < 20; o++) {
            pv[o] = (o < n_outs) ? packed[expert_base + (out_idx_0 + o) * packed_cols + pack_idx] : 0u;
        }
        // Symmetric fix to the gate/up kernel: loop bound MUST be
        // vals_per_u32 (= 32 / bits), not the hardcoded 16. See the
        // comment in jangtq_fused_gate_up_swiglu_matmul above for the
        // full diagnosis.
        for (uint k = 0; k < vals_per_u32; k++) {
            uint i = i_base + k;
            if (i >= in_features) break;
            float xv = static_cast<float>(x_rot[x_offset + i]);
            uint shift = k * bits;
            #pragma unroll
            for (uint o = 0; o < 20; o++) {
                float w = codebook[(pv[o] >> shift) & mask];
                acc[o] += xv * w;
            }
        }
    }

    #pragma unroll
    for (uint o = 0; o < 20; o++) {
        acc[o] = simd_sum(acc[o]);
    }

    if (lane == 0) {
        uint base_off = (token_idx * K + k_idx) * out_features;
        for (uint o = 0; o < n_outs; o++) {
            uint oi = out_idx_0 + o;
            float n_v = static_cast<float>(norms[expert * out_features + oi]);
            out[base_off + oi] = acc[o] * n_v;
        }
    }
"""

private let kGatherTQOffsetsSource = """
    uint global_x = thread_position_in_grid.x;
    uint dispatch_idx = thread_position_in_grid.y;

    uint out_group = global_x / 32u;
    uint lane = global_x % 32u;
    uint out_idx_0 = out_group * 20u;

    uint K = meta[0];
    uint in_features = meta[1];
    uint out_features = meta[2];
    uint packed_cols = meta[3];
    uint bits = meta[4];

    if (out_idx_0 >= out_features) return;

    uint token_idx = dispatch_idx / K;
    uint k_idx = dispatch_idx % K;
    uint expert = rhs_indices[token_idx * K + k_idx];

    uint vals_per_u32 = 32u / bits;
    uint mask = (1u << bits) - 1u;

    float acc[20];
    #pragma unroll
    for (uint o = 0; o < 20; o++) acc[o] = 0.0f;

    uint packed_base = packed_offsets[expert];
    uint x_offset = token_idx * in_features;

    uint n_outs = 20u;
    if (out_idx_0 + 20u > out_features) n_outs = out_features - out_idx_0;

    if (packed_base == 0xffffffffu || norm_offsets[expert] == 0xffffffffu) {
        if (lane == 0) {
            uint base_off = (token_idx * K + k_idx) * out_features;
            for (uint o = 0; o < n_outs; o++) {
                out[base_off + out_idx_0 + o] = 0.0f;
            }
        }
        return;
    }

    for (uint pack_idx = lane; pack_idx < packed_cols; pack_idx += 32u) {
        uint i_base = pack_idx * vals_per_u32;
        uint pv[20];
        #pragma unroll
        for (uint o = 0; o < 20; o++) {
            pv[o] = (o < n_outs) ? packed[packed_base + (out_idx_0 + o) * packed_cols + pack_idx] : 0u;
        }
        for (uint k = 0; k < vals_per_u32; k++) {
            uint i = i_base + k;
            if (i >= in_features) break;
            float xv = static_cast<float>(x_rot[x_offset + i]);
            uint shift = k * bits;
            #pragma unroll
            for (uint o = 0; o < 20; o++) {
                float w = codebook[(pv[o] >> shift) & mask];
                acc[o] += xv * w;
            }
        }
    }

    #pragma unroll
    for (uint o = 0; o < 20; o++) {
        acc[o] = simd_sum(acc[o]);
    }

    if (lane == 0) {
        uint base_off = (token_idx * K + k_idx) * out_features;
        uint norm_base = norm_offsets[expert];
        for (uint o = 0; o < n_outs; o++) {
            uint oi = out_idx_0 + o;
            float n_v = static_cast<float>(norms[norm_base + oi]);
            out[base_off + oi] = acc[o] * n_v;
        }
    }
"""

private let kGatherTQOffsetsScoredSource = """
    uint global_x = thread_position_in_grid.x;
    uint token_idx = thread_position_in_grid.y;

    uint out_group = global_x / 32u;
    uint lane = global_x % 32u;
    uint out_idx_0 = out_group * 20u;

    uint K = meta[0];
    uint in_features = meta[1];
    uint out_features = meta[2];
    uint packed_cols = meta[3];
    uint bits = meta[4];

    if (out_idx_0 >= out_features) return;

    uint vals_per_u32 = 32u / bits;
    uint mask = (1u << bits) - 1u;

    float acc[20];
    #pragma unroll
    for (uint o = 0; o < 20; o++) acc[o] = 0.0f;

    uint n_outs = 20u;
    if (out_idx_0 + 20u > out_features) n_outs = out_features - out_idx_0;

    for (uint k_idx = 0u; k_idx < K; k_idx++) {
        uint row_idx = token_idx * K + k_idx;
        uint expert = rhs_indices[row_idx];
        uint packed_base = packed_offsets[expert];
        uint norm_base = norm_offsets[expert];

        if (packed_base == 0xffffffffu || norm_base == 0xffffffffu) {
            continue;
        }

        float part[20];
        #pragma unroll
        for (uint o = 0; o < 20; o++) part[o] = 0.0f;

        uint x_offset = row_idx * in_features;
        for (uint pack_idx = lane; pack_idx < packed_cols; pack_idx += 32u) {
            uint i_base = pack_idx * vals_per_u32;
            uint pv[20];
            #pragma unroll
            for (uint o = 0; o < 20; o++) {
                pv[o] = (o < n_outs)
                    ? packed[packed_base + (out_idx_0 + o) * packed_cols + pack_idx]
                    : 0u;
            }
            for (uint k = 0; k < vals_per_u32; k++) {
                uint i = i_base + k;
                if (i >= in_features) break;
                float xv = static_cast<float>(x_rot[x_offset + i]);
                uint shift = k * bits;
                #pragma unroll
                for (uint o = 0; o < 20; o++) {
                    float w = codebook[(pv[o] >> shift) & mask];
                    part[o] += xv * w;
                }
            }
        }

        #pragma unroll
        for (uint o = 0; o < 20; o++) {
            part[o] = simd_sum(part[o]);
        }

        if (lane == 0) {
            float score = static_cast<float>(scores[row_idx]);
            for (uint o = 0; o < n_outs; o++) {
                uint oi = out_idx_0 + o;
                float n_v = static_cast<float>(norms[norm_base + oi]);
                acc[o] += part[o] * n_v * score;
            }
        }
    }

    if (lane == 0) {
        uint base_off = token_idx * out_features;
        for (uint o = 0; o < n_outs; o++) {
            out[base_off + out_idx_0 + o] = acc[o];
        }
    }
"""

private let kGatherTQScoredSource = """
    uint global_x = thread_position_in_grid.x;
    uint token_idx = thread_position_in_grid.y;

    uint out_group = global_x / 32u;
    uint lane = global_x % 32u;
    uint out_idx_0 = out_group * 20u;

    uint K = meta[0];
    uint in_features = meta[1];
    uint out_features = meta[2];
    uint packed_cols = meta[3];
    uint bits = meta[4];

    if (out_idx_0 >= out_features) return;

    uint vals_per_u32 = 32u / bits;
    uint mask = (1u << bits) - 1u;

    float acc[20];
    #pragma unroll
    for (uint o = 0; o < 20; o++) acc[o] = 0.0f;

    uint n_outs = 20u;
    if (out_idx_0 + 20u > out_features) n_outs = out_features - out_idx_0;

    for (uint k_idx = 0u; k_idx < K; k_idx++) {
        uint row_idx = token_idx * K + k_idx;
        uint expert = rhs_indices[row_idx];
        uint packed_base = expert * out_features * packed_cols;
        uint norm_base = expert * out_features;

        float part[20];
        #pragma unroll
        for (uint o = 0; o < 20; o++) part[o] = 0.0f;

        uint x_offset = row_idx * in_features;
        for (uint pack_idx = lane; pack_idx < packed_cols; pack_idx += 32u) {
            uint i_base = pack_idx * vals_per_u32;
            uint pv[20];
            #pragma unroll
            for (uint o = 0; o < 20; o++) {
                pv[o] = (o < n_outs)
                    ? packed[packed_base + (out_idx_0 + o) * packed_cols + pack_idx]
                    : 0u;
            }
            for (uint k = 0; k < vals_per_u32; k++) {
                uint i = i_base + k;
                if (i >= in_features) break;
                float xv = static_cast<float>(x_rot[x_offset + i]);
                uint shift = k * bits;
                #pragma unroll
                for (uint o = 0; o < 20; o++) {
                    float w = codebook[(pv[o] >> shift) & mask];
                    part[o] += xv * w;
                }
            }
        }

        #pragma unroll
        for (uint o = 0; o < 20; o++) {
            part[o] = simd_sum(part[o]);
        }

        if (lane == 0) {
            float score = static_cast<float>(scores[row_idx]);
            for (uint o = 0; o < n_outs; o++) {
                uint oi = out_idx_0 + o;
                float n_v = static_cast<float>(norms[norm_base + oi]);
                acc[o] += part[o] * n_v * score;
            }
        }
    }

    if (lane == 0) {
        uint base_off = token_idx * out_features;
        for (uint o = 0; o < n_outs; o++) {
            out[base_off + out_idx_0 + o] = acc[o];
        }
    }
"""

private let kFusedSwiGLUSlots8Source = """
    #define SLOT8_U32(slot, a0, a1, a2, a3, a4, a5, a6, a7, idx) \\
        ((slot) == 0u ? (a0)[idx] : \\
        ((slot) == 1u ? (a1)[idx] : \\
        ((slot) == 2u ? (a2)[idx] : \\
        ((slot) == 3u ? (a3)[idx] : \\
        ((slot) == 4u ? (a4)[idx] : \\
        ((slot) == 5u ? (a5)[idx] : \\
        ((slot) == 6u ? (a6)[idx] : (a7)[idx])))))))

    uint global_x = thread_position_in_grid.x;
    uint dispatch_idx = thread_position_in_grid.y;

    uint out_group = global_x / 32u;
    uint lane = global_x % 32u;
    uint out_idx_0 = out_group * 10u;

    uint K = meta[0];
    uint in_features = meta[1];
    uint out_features = meta[2];
    uint packed_cols = meta[3];
    uint bits = meta[4];
    uint swiglu_limit_q1000 = meta[5];
    float swiglu_limit = static_cast<float>(swiglu_limit_q1000) * 0.001f;

    if (out_idx_0 >= out_features) return;

    uint token_idx = dispatch_idx / K;
    uint k_idx = dispatch_idx % K;
    if (k_idx >= 8u) return;

    uint vals_per_u32 = 32u / bits;
    uint mask = (1u << bits) - 1u;

    float acc_g[10];
    float acc_u[10];
    #pragma unroll
    for (uint o = 0; o < 10; o++) { acc_g[o] = 0.0f; acc_u[o] = 0.0f; }

    uint x_off = token_idx * in_features;

    uint n_outs = 10u;
    if (out_idx_0 + 10u > out_features) n_outs = out_features - out_idx_0;

    for (uint pack_idx = lane; pack_idx < packed_cols; pack_idx += 32u) {
        uint i_base = pack_idx * vals_per_u32;

        uint pvg[10], pvu[10];
        #pragma unroll
        for (uint o = 0; o < 10; o++) {
            if (o < n_outs) {
                uint weight_idx = (out_idx_0 + o) * packed_cols + pack_idx;
                pvg[o] = SLOT8_U32(
                    k_idx,
                    packed_gate0, packed_gate1, packed_gate2, packed_gate3,
                    packed_gate4, packed_gate5, packed_gate6, packed_gate7,
                    weight_idx);
                pvu[o] = SLOT8_U32(
                    k_idx,
                    packed_up0, packed_up1, packed_up2, packed_up3,
                    packed_up4, packed_up5, packed_up6, packed_up7,
                    weight_idx);
            } else {
                pvg[o] = 0u;
                pvu[o] = 0u;
            }
        }

        for (uint k = 0; k < vals_per_u32; k++) {
            uint i = i_base + k;
            if (i >= in_features) break;
            float xv = static_cast<float>(x_rot[x_off + i]);
            uint shift = k * bits;
            #pragma unroll
            for (uint o = 0; o < 10; o++) {
                float w_g = codebook[(pvg[o] >> shift) & mask];
                float w_u = codebook[(pvu[o] >> shift) & mask];
                acc_g[o] += xv * w_g;
                acc_u[o] += xv * w_u;
            }
        }
    }

    #pragma unroll
    for (uint o = 0; o < 10; o++) {
        acc_g[o] = simd_sum(acc_g[o]);
        acc_u[o] = simd_sum(acc_u[o]);
    }

    if (lane == 0) {
        uint base_off = (token_idx * K + k_idx) * out_features;
        for (uint o = 0; o < n_outs; o++) {
            uint oi = out_idx_0 + o;
            float ng = static_cast<float>(norms_gate_bank[k_idx * out_features + oi]);
            float nu = static_cast<float>(norms_up_bank[k_idx * out_features + oi]);
            float gv = acc_g[o] * ng;
            float uv = acc_u[o] * nu;
            if (swiglu_limit > 0.0f) {
                gv = metal::min(gv, swiglu_limit);
                uv = metal::max(metal::min(uv, swiglu_limit), -swiglu_limit);
            }
            out_act[base_off + oi] = (gv / (1.0f + metal::fast::exp(-gv))) * uv;
        }
    }

    #undef SLOT8_U32
"""

private let kGatherTQSlots8ScoredSource = """
    #define SLOT8_U32(slot, a0, a1, a2, a3, a4, a5, a6, a7, idx) \\
        ((slot) == 0u ? (a0)[idx] : \\
        ((slot) == 1u ? (a1)[idx] : \\
        ((slot) == 2u ? (a2)[idx] : \\
        ((slot) == 3u ? (a3)[idx] : \\
        ((slot) == 4u ? (a4)[idx] : \\
        ((slot) == 5u ? (a5)[idx] : \\
        ((slot) == 6u ? (a6)[idx] : (a7)[idx])))))))

    uint global_x = thread_position_in_grid.x;
    uint token_idx = thread_position_in_grid.y;

    uint out_group = global_x / 32u;
    uint lane = global_x % 32u;
    uint out_idx_0 = out_group * 20u;

    uint K = meta[0];
    uint in_features = meta[1];
    uint out_features = meta[2];
    uint packed_cols = meta[3];
    uint bits = meta[4];

    if (out_idx_0 >= out_features) return;

    uint vals_per_u32 = 32u / bits;
    uint mask = (1u << bits) - 1u;

    float acc[20];
    #pragma unroll
    for (uint o = 0; o < 20; o++) acc[o] = 0.0f;

    uint n_outs = 20u;
    if (out_idx_0 + 20u > out_features) n_outs = out_features - out_idx_0;

    for (uint k_idx = 0u; k_idx < K; k_idx++) {
        if (k_idx >= 8u) continue;
        uint row_idx = token_idx * K + k_idx;

        float part[20];
        #pragma unroll
        for (uint o = 0; o < 20; o++) part[o] = 0.0f;

        uint x_offset = row_idx * in_features;
        for (uint pack_idx = lane; pack_idx < packed_cols; pack_idx += 32u) {
            uint i_base = pack_idx * vals_per_u32;
            uint pv[20];
            #pragma unroll
            for (uint o = 0; o < 20; o++) {
                if (o < n_outs) {
                    uint weight_idx = (out_idx_0 + o) * packed_cols + pack_idx;
                    pv[o] = SLOT8_U32(
                        k_idx,
                        packed0, packed1, packed2, packed3,
                        packed4, packed5, packed6, packed7,
                        weight_idx);
                } else {
                    pv[o] = 0u;
                }
            }
            for (uint k = 0; k < vals_per_u32; k++) {
                uint i = i_base + k;
                if (i >= in_features) break;
                float xv = static_cast<float>(x_rot[x_offset + i]);
                uint shift = k * bits;
                #pragma unroll
                for (uint o = 0; o < 20; o++) {
                    float w = codebook[(pv[o] >> shift) & mask];
                    part[o] += xv * w;
                }
            }
        }

        #pragma unroll
        for (uint o = 0; o < 20; o++) {
            part[o] = simd_sum(part[o]);
        }

        if (lane == 0) {
            float score = static_cast<float>(scores[row_idx]);
            for (uint o = 0; o < n_outs; o++) {
                uint oi = out_idx_0 + o;
                float n_v = static_cast<float>(norms_bank[k_idx * out_features + oi]);
                acc[o] += part[o] * n_v * score;
            }
        }
    }

    if (lane == 0) {
        uint base_off = token_idx * out_features;
        for (uint o = 0; o < n_outs; o++) {
            out[base_off + out_idx_0 + o] = acc[o];
        }
    }

    #undef SLOT8_U32
"""

// MARK: - Public kernel access

/// Lazy-built singleton kernels. Each kernel is compiled once via
/// `MLXFast.metalKernel(...)` and cached for the lifetime of the process.
public enum JANGTQKernelLibrary {

    public static let hadamardMultiblock: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "jangtq_hadamard_multiblock",
        inputNames: ["x", "signs", "meta"],
        outputNames: ["out"],
        source: kHadamardMultiblockSource
    )

    public static let hadamardShuffleLE1024: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "jangtq_hadamard_shuffle_le1024",
        inputNames: ["x", "signs", "meta"],
        outputNames: ["out"],
        source: kHadamardShuffleLE1024Source
    )

    public static let fusedGateUpSwiGLU: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "jangtq_fused_gate_up_swiglu",
        inputNames: [
            "x_rot", "packed_gate", "norms_gate",
            "packed_up", "norms_up",
            "codebook", "rhs_indices", "meta",
        ],
        outputNames: ["out_act"],
        source: kFusedSwiGLUSource
    )

    public static let fusedGateUpSwiGLUOffsets: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "jangtq_fused_gate_up_swiglu_offsets",
        inputNames: [
            "x_rot",
            "packed_gate", "packed_gate_offsets", "norms_gate", "norms_gate_offsets",
            "packed_up", "packed_up_offsets", "norms_up", "norms_up_offsets",
            "codebook", "rhs_indices", "meta",
        ],
        outputNames: ["out_act"],
        source: kFusedSwiGLUOffsetsSource
    )

    public static let gatherTQ: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "jangtq_gather_tq_matmul",
        inputNames: ["x_rot", "packed", "norms", "codebook", "rhs_indices", "meta"],
        outputNames: ["out"],
        source: kGatherTQSource
    )

    public static let gatherTQOffsets: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "jangtq_gather_tq_offsets_matmul",
        inputNames: [
            "x_rot", "packed", "packed_offsets", "norms", "norm_offsets",
            "codebook", "rhs_indices", "meta",
        ],
        outputNames: ["out"],
        source: kGatherTQOffsetsSource
    )

    public static let gatherTQOffsetsScored: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "jangtq_gather_tq_offsets_scored_matmul",
        inputNames: [
            "x_rot", "packed", "packed_offsets", "norms", "norm_offsets",
            "codebook", "rhs_indices", "scores", "meta",
        ],
        outputNames: ["out"],
        source: kGatherTQOffsetsScoredSource
    )

    public static let gatherTQScored: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "jangtq_gather_tq_scored_matmul",
        inputNames: [
            "x_rot", "packed", "norms",
            "codebook", "rhs_indices", "scores", "meta",
        ],
        outputNames: ["out"],
        source: kGatherTQScoredSource
    )

    public static let fusedGateUpSwiGLUSlots8: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "jangtq_fused_gate_up_swiglu_slots8",
        inputNames: [
            "x_rot",
            "packed_gate0", "packed_gate1", "packed_gate2", "packed_gate3",
            "packed_gate4", "packed_gate5", "packed_gate6", "packed_gate7",
            "norms_gate_bank",
            "packed_up0", "packed_up1", "packed_up2", "packed_up3",
            "packed_up4", "packed_up5", "packed_up6", "packed_up7",
            "norms_up_bank",
            "codebook", "meta",
        ],
        outputNames: ["out_act"],
        source: kFusedSwiGLUSlots8Source
    )

    public static let gatherTQSlots8Scored: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "jangtq_gather_tq_slots8_scored_matmul",
        inputNames: [
            "x_rot",
            "packed0", "packed1", "packed2", "packed3",
            "packed4", "packed5", "packed6", "packed7",
            "norms_bank",
            "codebook", "scores", "meta",
        ],
        outputNames: ["out"],
        source: kGatherTQSlots8ScoredSource
    )
}

// MARK: - Codebook + signs cache

/// Sign and codebook arrays are deterministic functions of (in_features, seed/bits)
/// computed at quantization time via NumPy PCG64 + Lloyd-Max iteration. They're
/// loaded once at model load from `jangtq_runtime.safetensors` and cached here
/// keyed on `(in_features, seed)` / `(in_features, bits)`.
public final class JANGTQRuntimeCache: @unchecked Sendable {
    public static let shared = JANGTQRuntimeCache()

    private var signsByKey: [String: MLXArray] = [:]
    private var codebookByKey: [String: MLXArray] = [:]
    private let lock = NSLock()

    private init() {}

    public func loadSidecar(from sidecarPath: URL) throws {
        let loaded = try MLX.loadArrays(url: sidecarPath)
        lock.lock()
        defer { lock.unlock() }
        for (name, arr) in loaded {
            if name.hasPrefix("signs.") {
                signsByKey[name] = arr
            } else if name.hasPrefix("codebook.") {
                codebookByKey[name] = arr
            }
        }
    }

    public func signs(inFeatures: Int, seed: Int) -> MLXArray? {
        let key = "signs.\(inFeatures).\(seed)"
        lock.lock()
        if let hit = signsByKey[key] {
            lock.unlock()
            return hit
        }
        lock.unlock()
        let generated = NumPyPCG64.generateRandomSigns(dim: inFeatures, seed: seed)
        lock.lock()
        signsByKey[key] = generated
        lock.unlock()
        return generated
    }

    public func codebook(inFeatures: Int, bits: Int) -> MLXArray? {
        let key = "codebook.\(inFeatures).\(bits)"
        lock.lock()
        if let hit = codebookByKey[key] {
            lock.unlock()
            return hit
        }
        lock.unlock()
        let centroids = TQCodebook.computeCodebook(dim: inFeatures, bits: bits)
        let arr = MLXArray(centroids)
        lock.lock()
        codebookByKey[key] = arr
        lock.unlock()
        return arr
    }

    /// Sniff the routed-MoE codebook bits directly from a sidecar
    /// safetensors file WITHOUT fully loading it into the runtime
    /// cache. Uses the `codebook.{inFeatures}.{bits}` key naming
    /// convention to read the actual bit width that was used at
    /// quantization time.
    ///
    /// This is the most reliable signal when the bundle's
    /// `jang_config.json` is missing the routed-expert bits field
    /// (e.g. some Qwen3.6-A3B-JANGTQ4 / Kimi-K2.6 bundles ship only
    /// `quantization.bits=8`, which is the affine non-routed setting,
    /// not the codebook bits). Returns the most-frequent `bits` value
    /// among the codebook keys, or `nil` if the file has no codebook
    /// entries (or doesn't exist).
    public static func sniffCodebookBits(at sidecarPath: URL) -> Int? {
        guard FileManager.default.fileExists(atPath: sidecarPath.path),
              let arrays = try? MLX.loadArrays(url: sidecarPath)
        else { return nil }
        var counts = [Int: Int]()
        for name in arrays.keys where name.hasPrefix("codebook.") {
            // Format: `codebook.{inFeatures}.{bits}`
            let parts = name.split(separator: ".")
            guard parts.count == 3, let bits = Int(parts[2]) else { continue }
            counts[bits, default: 0] += 1
        }
        return counts.max(by: { $0.value < $1.value })?.key
    }
}

/// Detect routed-MoE codebook bits from a JANG bundle's `profile`
/// string field (`JANGTQ4` → 4, `JANGTQ2`/`JANGTQ`/`MXTQ` → 2,
/// `JANGTQ1` → 1).
/// Bundle naming convention is empirically reliable: every JANG /
/// JANGTQ converter pre-2026-04 stamped the profile this way.
/// Returns `nil` for unrecognized strings so the caller falls back
/// to the next signal in the resolution chain.
public func jangtqBitsFromProfile(_ profile: String?) -> Int? {
    guard let profile, !profile.isEmpty else { return nil }
    let p = profile.lowercased()
    if p.contains("jangtq1") || p.contains("jangtq_1") || p.contains("jangtq-1") {
        return 1
    }
    if p.contains("jangtq4") || p.contains("jangtq_4") || p.contains("jangtq-4") {
        return 4
    }
    if p.contains("jangtq2") || p.contains("jangtq_2") || p.contains("jangtq-2") {
        return 2
    }
    // Bare "jangtq" / "mxtq" historically meant 2-bit.
    if p == "jangtq" || p == "mxtq" {
        return 2
    }
    return nil
}

// MARK: - High-level kernel wrappers (mirror Python `make_*_decode` factories)

public enum JANGTQKernels {
    private static func padSlots8(_ arrays: [MLXArray]) -> [MLXArray] {
        precondition(!arrays.isEmpty && arrays.count <= 8, "slots8 requires 1...8 arrays")
        var padded = arrays
        while padded.count < 8 {
            padded.append(arrays[0])
        }
        return padded
    }


    /// Decompose a non-pow2 dim into a sum of pow2 blocks (largest first).
    public static func decomposePow2(_ dim: Int) -> [Int] {
        var blocks: [Int] = []
        var rem = dim
        while rem > 0 {
            let p = 1 << (Int.bitWidth - 1 - rem.leadingZeroBitCount)
            blocks.append(p)
            rem -= p
        }
        return blocks
    }

    /// Build the `meta` array the multiblock Hadamard kernel expects:
    /// `[total_d, n_blocks, d_b0, log_b0, d_b1, log_b1, ...]`
    public static func makeHadamardMeta(totalDim: Int) -> MLXArray {
        let blocks = decomposePow2(totalDim)
        var meta: [UInt32] = [UInt32(totalDim), UInt32(blocks.count)]
        for d in blocks {
            meta.append(UInt32(d))
            meta.append(UInt32(d.trailingZeroBitCount))
        }
        return cachedJANGTQMeta(kind: "hadamard", values: meta)
    }

    /// Hadamard rotate `x` (any batch shape with `dim` last). Returns fp32.
    /// `signs` must be shape `(dim,)` fp32.
    ///
    /// Apple Silicon caps threadgroup memory at 32 KB = 8192 floats, so the
    /// per-block Metal kernel can only process blocks up to 8192 elements.
    /// Mistral-Medium-3.5 hits this on `down_proj.in_features=28672` →
    /// `decomposePow2(28672) = [16384, 8192, 4096]`. The 16384-block has
    /// no in-shmem implementation; we instead split it in Swift via the
    /// well-known recursion
    ///     `H_{2n}(u,v) = [H_n((u+v)/√2), H_n((u-v)/√2)]`
    /// applying it once for each "doubling above 8192". The signs are
    /// applied to the original input ONCE before the split (signs are
    /// per-input-coordinate diagonal, so they commute with the split as
    /// long as we don't double-apply), and each leaf-call uses an all-
    /// ones sign vector.
    public static func hadamardRotate(_ x: MLXArray, signs: MLXArray, dim: Int) -> MLXArray {
        let xFlat = x.reshaped([-1, dim]).asType(.float32)
        let batch = xFlat.shape[0]

        let blocks = decomposePow2(dim)
        let maxBlock = blocks.max() ?? dim
        if maxBlock <= 1024 {
            let meta = makeHadamardMeta(totalDim: dim)
            let tgSize = max(32, maxBlock)
            let outArrs = JANGTQKernelLibrary.hadamardShuffleLE1024(
                [xFlat, signs, meta],
                template: nil,
                grid: (tgSize, batch, 1),
                threadGroup: (tgSize, 1, 1),
                outputShapes: [[batch, dim]],
                outputDTypes: [.float32]
            )
            var rot = outArrs[0]
            if x.ndim > 2 || (x.ndim == 2 && x.dim(0) != batch) {
                rot = rot.reshaped(x.shape)
            }
            return rot
        }

        if maxBlock <= 8192 {
            // Fast path: every block fits in shmem — single kernel
            // dispatch processes all blocks back-to-back.
            let meta = makeHadamardMeta(totalDim: dim)
            let tgSize = min(1024, max(32, maxBlock))
            let outArrs = JANGTQKernelLibrary.hadamardMultiblock(
                [xFlat, signs, meta],
                template: nil,
                grid: (tgSize, batch, 1),
                threadGroup: (tgSize, 1, 1),
                outputShapes: [[batch, dim]],
                outputDTypes: [.float32]
            )
            var rot = outArrs[0]
            if x.ndim > 2 || (x.ndim == 2 && x.dim(0) != batch) {
                rot = rot.reshaped(x.shape)
            }
            return rot
        }

        // Slow path (shmem-overflow case): process each pow2 block
        // separately, splitting > 8192 blocks via the H_{2n} recursion.
        var blockOuts: [MLXArray] = []
        var offset = 0
        for d_b in blocks {
            let blockX = xFlat[0..., offset..<(offset + d_b)]
            let blockSigns = signs[offset..<(offset + d_b)]
            blockOuts.append(
                hadamardBlockRecursive(blockX, signs: blockSigns, d_b: d_b))
            offset += d_b
        }
        let merged = MLX.concatenated(blockOuts, axis: -1)
        if x.ndim > 2 || (x.ndim == 2 && x.dim(0) != batch) {
            return merged.reshaped(x.shape)
        }
        return merged
    }

    /// Recursive single-block Hadamard. For `d_b <= 8192` dispatches the
    /// Metal kernel directly. For `d_b > 8192` applies the H_{2n}
    /// recursion in Swift, using all-ones signs in the recursive calls
    /// (signs were consumed by the caller's sign-multiplication step).
    private static func hadamardBlockRecursive(
        _ x: MLXArray, signs: MLXArray, d_b: Int
    ) -> MLXArray {
        if d_b <= 8192 {
            // Use the multi-block kernel with n_blocks=1.
            let meta = makeHadamardMeta(totalDim: d_b)
            let tgSize = min(1024, max(32, d_b))
            let outArrs = JANGTQKernelLibrary.hadamardMultiblock(
                [x, signs, meta],
                template: nil,
                grid: (tgSize, x.shape[0], 1),
                threadGroup: (tgSize, 1, 1),
                outputShapes: [x.shape],
                outputDTypes: [.float32]
            )
            return outArrs[0]
        }
        // Apply signs ONCE to the input. The recursive halves use ones.
        let xSigned = x * signs
        let half = d_b / 2
        let u = xSigned[0..., 0..<half]
        let v = xSigned[0..., half..<d_b]
        let invSqrt2 = MLXArray(Float(1.0 / sqrt(2.0)))
        let a = (u + v) * invSqrt2
        let b = (u - v) * invSqrt2
        let onesSigns = MLXArray.ones([half], dtype: .float32)
        let halfA = hadamardBlockRecursive(a, signs: onesSigns, d_b: half)
        let halfB = hadamardBlockRecursive(b, signs: onesSigns, d_b: half)
        return MLX.concatenated([halfA, halfB], axis: -1)
    }

    /// Fused gate+up+SwiGLU.
    /// - `K` : experts per token (e.g. 8) — becomes `meta[0]` inside the kernel
    ///         so the kernel can compute `token_idx = dispatch_idx / K`.
    /// - `batchTokens` : number of input rows in `xRot` (tokens in the batch).
    ///         Total dispatches in `y` grid = `batchTokens * K`.
    /// - `xRot` shape: `(batchTokens, inFeatures)`
    /// - `rhsIndices` shape: `(batchTokens * K,)` uint32
    /// Returns fp32 of shape `(batchTokens * K, out_features)`.
    public static func fusedGateUpSwiGLU(
        xRot: MLXArray,
        packedGate: MLXArray, normsGate: MLXArray,
        packedUp: MLXArray,   normsUp: MLXArray,
        codebook: MLXArray,
        rhsIndices: MLXArray,
        batchTokens: Int, K: Int,
        inFeatures: Int, outFeatures: Int, bits: Int = 2,
        // 2026-05-04 (DSV4 SWA/CSA/HSA correctness):
        // SwiGLU clamp magnitude. 0.0 (default) preserves the historical
        // ordinary-SwiGLU output bit-for-bit. DSV4 callers must pass 10.0
        // — that activates `silu(min(gate, 10)) * clip(up, -10, 10)` per
        // jang_tools/dsv4/mlx_model.py and codex_dsv4_fixkit/scripts/
        // runtime_dsv4_fixed.py. The kernel encodes this as
        // `meta[5] = round(swigluLimit * 1000)` (uint) and divides by
        // 1000 inside Metal — small enough to fit in a uint32 for the
        // realistic range while being ~1e-3 precise.
        swigluLimit: Float = 0.0
    ) -> MLXArray {
        let valsPerU32 = 32 / bits
        let packedCols = (inFeatures + valsPerU32 - 1) / valsPerU32
        let nDispatches = batchTokens * K
        let limitQ1000 = UInt32(max(0, Int((swigluLimit * 1000.0).rounded())))
        let meta = cachedJANGTQMeta(kind: "fusedGateUpSwiGLU", values: [
            UInt32(K), UInt32(inFeatures), UInt32(outFeatures),
            UInt32(packedCols), UInt32(bits),
            limitQ1000,
        ])
        let opt = 10
        let outGroups = (outFeatures + opt - 1) / opt
        let gridX = outGroups * 32
        let tgX = min(gridX, 256)
        let arr = JANGTQKernelLibrary.fusedGateUpSwiGLU(
            [xRot, packedGate, normsGate, packedUp, normsUp,
             codebook, rhsIndices, meta],
            template: nil,
            grid: (gridX, nDispatches, 1),
            threadGroup: (tgX, 1, 1),
            outputShapes: [[nDispatches, outFeatures]],
            outputDTypes: [.float32]
        )
        return arr[0]
    }

    /// Offset-addressed fused gate+up+SwiGLU. `packed*` and `norms*` are
    /// contiguous backing spans, while the offset arrays contain one element
    /// offset per original expert id. This preserves the standard GPU-side
    /// `rhsIndices` contract without forcing the caller to rebuild a dense
    /// stacked expert bank.
    public static func fusedGateUpSwiGLUOffsets(
        xRot: MLXArray,
        packedGate: MLXArray, packedGateOffsets: MLXArray,
        normsGate: MLXArray, normsGateOffsets: MLXArray,
        packedUp: MLXArray, packedUpOffsets: MLXArray,
        normsUp: MLXArray, normsUpOffsets: MLXArray,
        codebook: MLXArray,
        rhsIndices: MLXArray,
        batchTokens: Int, K: Int,
        inFeatures: Int, outFeatures: Int, bits: Int = 2,
        swigluLimit: Float = 0.0
    ) -> MLXArray {
        let valsPerU32 = 32 / bits
        let packedCols = (inFeatures + valsPerU32 - 1) / valsPerU32
        let nDispatches = batchTokens * K
        let limitQ1000 = UInt32(max(0, Int((swigluLimit * 1000.0).rounded())))
        let meta = cachedJANGTQMeta(kind: "fusedGateUpSwiGLUOffsets", values: [
            UInt32(K), UInt32(inFeatures), UInt32(outFeatures),
            UInt32(packedCols), UInt32(bits),
            limitQ1000,
        ])
        let opt = 10
        let outGroups = (outFeatures + opt - 1) / opt
        let gridX = outGroups * 32
        let tgX = min(gridX, 256)
        let arr = JANGTQKernelLibrary.fusedGateUpSwiGLUOffsets(
            [
                xRot,
                packedGate, packedGateOffsets, normsGate, normsGateOffsets,
                packedUp, packedUpOffsets, normsUp, normsUpOffsets,
                codebook, rhsIndices, meta,
            ],
            template: nil,
            grid: (gridX, nDispatches, 1),
            threadGroup: (tgX, 1, 1),
            outputShapes: [[nDispatches, outFeatures]],
            outputDTypes: [.float32]
        )
        return arr[0]
    }

    /// Decode-specialized fused gate/up kernel for exact active top-k slots.
    /// Each slot is a single expert span starting at offset zero, so the kernel
    /// avoids one Metal launch per routed expert window. Intended for
    /// `batchTokens == 1` Kimi-style decode; prefill stays on the general
    /// offset path unless every token shares the same slot mapping.
    public static func fusedGateUpSwiGLUSlots8(
        xRot: MLXArray,
        packedGate: [MLXArray],
        normsGate: [MLXArray],
        packedUp: [MLXArray],
        normsUp: [MLXArray],
        codebook: MLXArray,
        batchTokens: Int, K: Int,
        inFeatures: Int, outFeatures: Int, bits: Int = 2,
        swigluLimit: Float = 0.0
    ) -> MLXArray {
        precondition(K > 0 && K <= 8, "slots8 requires K in 1...8")
        precondition(
            packedGate.count == K && normsGate.count == K
                && packedUp.count == K && normsUp.count == K,
            "slots8 arrays must match K")
        let packedGate = padSlots8(packedGate)
        let normsGateBank = concatenated(padSlots8(normsGate), axis: 0)
        let packedUp = padSlots8(packedUp)
        let normsUpBank = concatenated(padSlots8(normsUp), axis: 0)
        let valsPerU32 = 32 / bits
        let packedCols = (inFeatures + valsPerU32 - 1) / valsPerU32
        let nDispatches = batchTokens * K
        let limitQ1000 = UInt32(max(0, Int((swigluLimit * 1000.0).rounded())))
        let meta = cachedJANGTQMeta(kind: "fusedGateUpSwiGLUSlots8", values: [
            UInt32(K), UInt32(inFeatures), UInt32(outFeatures),
            UInt32(packedCols), UInt32(bits),
            limitQ1000,
        ])
        let opt = 10
        let outGroups = (outFeatures + opt - 1) / opt
        let gridX = outGroups * 32
        let tgX = min(gridX, 256)
        let arr = JANGTQKernelLibrary.fusedGateUpSwiGLUSlots8(
            [
                xRot,
                packedGate[0], packedGate[1], packedGate[2], packedGate[3],
                packedGate[4], packedGate[5], packedGate[6], packedGate[7],
                normsGateBank,
                packedUp[0], packedUp[1], packedUp[2], packedUp[3],
                packedUp[4], packedUp[5], packedUp[6], packedUp[7],
                normsUpBank,
                codebook, meta,
            ],
            template: nil,
            grid: (gridX, nDispatches, 1),
            threadGroup: (tgX, 1, 1),
            outputShapes: [[nDispatches, outFeatures]],
            outputDTypes: [.float32]
        )
        return arr[0]
    }

    /// Gather TQ matmul in per-row mode (down_proj path).
    /// - `xRot` shape: `(nRows, inFeatures)` — one row per (token, expert) pair.
    /// - `rhsIndices` shape: `(nRows,)` uint32 — expert id for each row.
    /// Returns fp32 of shape `(nRows, outFeatures)`.
    public static func gatherTQ(
        xRot: MLXArray,
        packed: MLXArray, norms: MLXArray,
        codebook: MLXArray, rhsIndices: MLXArray,
        nRows: Int, inFeatures: Int, outFeatures: Int, bits: Int = 2
    ) -> MLXArray {
        let valsPerU32 = 32 / bits
        let packedCols = (inFeatures + valsPerU32 - 1) / valsPerU32
        // Per-row: K_meta = 1, so token_idx = dispatch_idx, k_idx = 0.
        let meta = cachedJANGTQMeta(kind: "gatherTQ", values: [
            UInt32(1), UInt32(inFeatures), UInt32(outFeatures),
            UInt32(packedCols), UInt32(bits),
        ])
        let opt = 20
        let outGroups = (outFeatures + opt - 1) / opt
        let gridX = outGroups * 32
        let tgX = min(gridX, 256)
        let arr = JANGTQKernelLibrary.gatherTQ(
            [xRot, packed, norms, codebook, rhsIndices, meta],
            template: nil,
            grid: (gridX, nRows, 1),
            threadGroup: (tgX, 1, 1),
            outputShapes: [[nRows, outFeatures]],
            outputDTypes: [.float32]
        )
        return arr[0]
    }

    /// Offset-addressed Gather TQ matmul in per-row mode. Offsets are element
    /// offsets into the packed/norm backing spans, indexed by original expert
    /// id from `rhsIndices`.
    public static func gatherTQOffsets(
        xRot: MLXArray,
        packed: MLXArray, packedOffsets: MLXArray,
        norms: MLXArray, normOffsets: MLXArray,
        codebook: MLXArray, rhsIndices: MLXArray,
        nRows: Int, inFeatures: Int, outFeatures: Int, bits: Int = 2
    ) -> MLXArray {
        let valsPerU32 = 32 / bits
        let packedCols = (inFeatures + valsPerU32 - 1) / valsPerU32
        let meta = cachedJANGTQMeta(kind: "gatherTQOffsets", values: [
            UInt32(1), UInt32(inFeatures), UInt32(outFeatures),
            UInt32(packedCols), UInt32(bits),
        ])
        let opt = 20
        let outGroups = (outFeatures + opt - 1) / opt
        let gridX = outGroups * 32
        let tgX = min(gridX, 256)
        let arr = JANGTQKernelLibrary.gatherTQOffsets(
            [xRot, packed, packedOffsets, norms, normOffsets, codebook, rhsIndices, meta],
            template: nil,
            grid: (gridX, nRows, 1),
            threadGroup: (tgX, 1, 1),
            outputShapes: [[nRows, outFeatures]],
            outputDTypes: [.float32]
        )
        return arr[0]
    }

    /// Gather TQ matmul in top-k mode. `xRot` has one row per token while
    /// `rhsIndices` has `batchTokens * K` expert ids. The Metal kernel uses
    /// `token_idx = dispatch_idx / K`, so the same rotated token row is reused
    /// for each selected expert without broadcasting the input K times.
    public static func gatherTQTopK(
        xRot: MLXArray,
        packed: MLXArray, norms: MLXArray,
        codebook: MLXArray, rhsIndices: MLXArray,
        batchTokens: Int, K: Int,
        inFeatures: Int, outFeatures: Int, bits: Int = 2
    ) -> MLXArray {
        let valsPerU32 = 32 / bits
        let packedCols = (inFeatures + valsPerU32 - 1) / valsPerU32
        let nDispatches = batchTokens * K
        let meta = cachedJANGTQMeta(kind: "gatherTQTopK", values: [
            UInt32(K), UInt32(inFeatures), UInt32(outFeatures),
            UInt32(packedCols), UInt32(bits),
        ])
        let opt = 20
        let outGroups = (outFeatures + opt - 1) / opt
        let gridX = outGroups * 32
        let tgX = min(gridX, 256)
        let arr = JANGTQKernelLibrary.gatherTQ(
            [xRot, packed, norms, codebook, rhsIndices, meta],
            template: nil,
            grid: (gridX, nDispatches, 1),
            threadGroup: (tgX, 1, 1),
            outputShapes: [[nDispatches, outFeatures]],
            outputDTypes: [.float32]
        )
        return arr[0]
    }

    /// Gather TQ matmul in top-k mode with router-score reduction fused into
    /// the kernel. `xRot` has one row per selected slot while the output is
    /// reduced to `(batchTokens, outFeatures)`.
    public static func gatherTQTopKScored(
        xRot: MLXArray,
        packed: MLXArray, norms: MLXArray,
        codebook: MLXArray, rhsIndices: MLXArray,
        scores: MLXArray,
        batchTokens: Int, K: Int,
        inFeatures: Int, outFeatures: Int, bits: Int = 2
    ) -> MLXArray {
        let valsPerU32 = 32 / bits
        let packedCols = (inFeatures + valsPerU32 - 1) / valsPerU32
        let meta = cachedJANGTQMeta(kind: "gatherTQTopKScored", values: [
            UInt32(K), UInt32(inFeatures), UInt32(outFeatures),
            UInt32(packedCols), UInt32(bits),
        ])
        let opt = 20
        let outGroups = (outFeatures + opt - 1) / opt
        let gridX = outGroups * 32
        let tgX = min(gridX, 256)
        let arr = JANGTQKernelLibrary.gatherTQScored(
            [xRot, packed, norms, codebook, rhsIndices, scores, meta],
            template: nil,
            grid: (gridX, batchTokens, 1),
            threadGroup: (tgX, 1, 1),
            outputShapes: [[batchTokens, outFeatures]],
            outputDTypes: [.float32]
        )
        return arr[0]
    }

    /// Offset-addressed Gather TQ matmul in top-k mode. This is the down-proj
    /// half of direct active-expert dispatch for expert-major safetensors.
    public static func gatherTQTopKOffsets(
        xRot: MLXArray,
        packed: MLXArray, packedOffsets: MLXArray,
        norms: MLXArray, normOffsets: MLXArray,
        codebook: MLXArray, rhsIndices: MLXArray,
        batchTokens: Int, K: Int,
        inFeatures: Int, outFeatures: Int, bits: Int = 2
    ) -> MLXArray {
        let valsPerU32 = 32 / bits
        let packedCols = (inFeatures + valsPerU32 - 1) / valsPerU32
        let nDispatches = batchTokens * K
        let meta = cachedJANGTQMeta(kind: "gatherTQTopKOffsets", values: [
            UInt32(K), UInt32(inFeatures), UInt32(outFeatures),
            UInt32(packedCols), UInt32(bits),
        ])
        let opt = 20
        let outGroups = (outFeatures + opt - 1) / opt
        let gridX = outGroups * 32
        let tgX = min(gridX, 256)
        let arr = JANGTQKernelLibrary.gatherTQOffsets(
            [xRot, packed, packedOffsets, norms, normOffsets, codebook, rhsIndices, meta],
            template: nil,
            grid: (gridX, nDispatches, 1),
            threadGroup: (tgX, 1, 1),
            outputShapes: [[nDispatches, outFeatures]],
            outputDTypes: [.float32]
        )
        return arr[0]
    }

    /// Offset-addressed Gather TQ matmul with router-score reduction fused
    /// into the down-proj kernel. `xRot` has `batchTokens * K` rows, while
    /// the output is already reduced to `(batchTokens, outFeatures)`.
    public static func gatherTQTopKOffsetsScored(
        xRot: MLXArray,
        packed: MLXArray, packedOffsets: MLXArray,
        norms: MLXArray, normOffsets: MLXArray,
        codebook: MLXArray, rhsIndices: MLXArray,
        scores: MLXArray,
        batchTokens: Int, K: Int,
        inFeatures: Int, outFeatures: Int, bits: Int = 2
    ) -> MLXArray {
        let valsPerU32 = 32 / bits
        let packedCols = (inFeatures + valsPerU32 - 1) / valsPerU32
        let meta = cachedJANGTQMeta(kind: "gatherTQTopKOffsetsScored", values: [
            UInt32(K), UInt32(inFeatures), UInt32(outFeatures),
            UInt32(packedCols), UInt32(bits),
        ])
        let opt = 20
        let outGroups = (outFeatures + opt - 1) / opt
        let gridX = outGroups * 32
        let tgX = min(gridX, 256)
        let arr = JANGTQKernelLibrary.gatherTQOffsetsScored(
            [
                xRot, packed, packedOffsets, norms, normOffsets,
                codebook, rhsIndices, scores, meta,
            ],
            template: nil,
            grid: (gridX, batchTokens, 1),
            threadGroup: (tgX, 1, 1),
            outputShapes: [[batchTokens, outFeatures]],
            outputDTypes: [.float32]
        )
        return arr[0]
    }

    /// Decode-specialized scored down projection for exact active top-k slots.
    /// `xRot` has one row per slot and `scores` has `K` values for the single
    /// token. The output is already reduced to `(batchTokens, outFeatures)`.
    public static func gatherTQTopKSlots8Scored(
        xRot: MLXArray,
        packed: [MLXArray],
        norms: [MLXArray],
        codebook: MLXArray,
        scores: MLXArray,
        batchTokens: Int, K: Int,
        inFeatures: Int, outFeatures: Int, bits: Int = 2
    ) -> MLXArray {
        precondition(K > 0 && K <= 8, "slots8 requires K in 1...8")
        precondition(packed.count == K && norms.count == K, "slots8 arrays must match K")
        let packed = padSlots8(packed)
        let normsBank = concatenated(padSlots8(norms), axis: 0)
        let valsPerU32 = 32 / bits
        let packedCols = (inFeatures + valsPerU32 - 1) / valsPerU32
        let meta = cachedJANGTQMeta(kind: "gatherTQTopKSlots8Scored", values: [
            UInt32(K), UInt32(inFeatures), UInt32(outFeatures),
            UInt32(packedCols), UInt32(bits),
        ])
        let opt = 20
        let outGroups = (outFeatures + opt - 1) / opt
        let gridX = outGroups * 32
        let tgX = min(gridX, 256)
        let arr = JANGTQKernelLibrary.gatherTQSlots8Scored(
            [
                xRot,
                packed[0], packed[1], packed[2], packed[3],
                packed[4], packed[5], packed[6], packed[7],
                normsBank,
                codebook, scores, meta,
            ],
            template: nil,
            grid: (gridX, batchTokens, 1),
            threadGroup: (tgX, 1, 1),
            outputShapes: [[batchTokens, outFeatures]],
            outputDTypes: [.float32]
        )
        return arr[0]
    }
}
