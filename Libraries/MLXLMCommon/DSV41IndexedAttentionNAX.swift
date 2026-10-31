// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXFast

/// DeepSeek-V4.1 indexed prefill attention on the M5 matrix units (opt-in; not bit-identical to the heads-16 kernel).
///
/// DeepSeek-V4.1 has ONE key/value head (keys == values, 512 wide) and 64 query heads, and every query row has its own
/// key list: its 128 window rows, then its top-k compressed pool rows. So per query token the attention is a small
/// GEMM pair with the heads as the M dimension: S = Q_t (heads x 512) . K_t^T (512 x keys), softmax over the keys with
/// the per-head sink, O = P . K_t. This kernel runs that per token on NAX fragments, with the schedule of MLX's own
/// head-dim-split NAX attention (`steel_attention_nax.h`, attention_nax_dsplit: 2 x 4 simdgroups, 16 rows x 128
/// columns each, partial scores exchanged through threadgroup memory and summed in a fixed order, online fp32 softmax,
/// P kept fp32 into P.V): one threadgroup per (token, 32-head block). The key rows are read IN PLACE by index from the
/// window rows and the dense pool (no gather, no fp32 copy): a 32-key block's row table is built once in threadgroup
/// memory, each lane reads its fragment elements from its rows. Invisible keys (window rows before the first, picks
/// that are invalid or whose summarized window has not ended) load zeros and are masked to -inf; a block with no
/// visible key is skipped.
///
/// Q.K^T has bf16 / fp16 operands (exact products) with fp32 accumulation in the matrix unit's order; P.V takes the
/// fp32 probabilities as MLX's fused attention does. The summation order differs from the heads-16 kernel's
/// one-key-at-a-time SIMD loop, so the output is equal within rounding, not bit for bit.
public enum DSV41IndexedAttentionNAX {
    /// NAX (M5-class GPUs) only.
    public static var available: Bool { JANGHPrefillKernel.nativeBackend == .nax }

    /// The model routes calls of more than this many query rows here (prompt chunks), so a dSpark verify (<= 8 rows)
    /// computes exactly what it computes with the switch off.
    public static let minRows = 8

    /// `VMLX_DSV41_ATTN_NAX=1` (default OFF). Read per call (getenv) for in-process A/B.
    public static var enabled: Bool {
        guard let raw = getenv("VMLX_DSV41_ATTN_NAX") else { return false }
        return String(cString: raw) == "1"
    }

    static let source = """
        constexpr short kU = 16, BQ = 32, BK = 32, WN = 4, TK = 2, TDh = 8;  // 128 columns per D group
        constexpr int D = 512;
        using stile_t = NAXTile<float, 1, TK>;
        using otile_t = NAXTile<float, 1, TDh>;
        using frag_f = typename BaseNAXFrag::template dtype_frag_t<float>;
        using frag_t = typename BaseNAXFrag::template dtype_frag_t<T>;
        using vec4_t = metal::vec<T, 4>;
        constexpr short kEPF = BaseNAXFrag::kElemsPerFrag;

        const int t = int(threadgroup_position_in_grid.x);
        const int hb = int(threadgroup_position_in_grid.y);
        const uint sgid = simdgroup_index_in_threadgroup;
        const uint lane = thread_index_in_simdgroup;
        const uint tid = thread_position_in_threadgroup.x;

        const int S = params[0], R = params[1], P = params[2], K = params[3];
        const int offset = params[4], window = params[5], ratio = params[6];
        const int qpos = offset + t;
        const float scale2 = fscale[0] * 1.44269504089f;

        const short row_group = short(sgid) / WN;
        const short d_group = short(sgid) % WN;
        const int h_base = hb * BQ + row_group * kU;
        const int col0 = d_group * (D / WN);
        const size_t hstride = size_t(S) * D;  // between heads of one token

        const short2 sc = BaseNAXFrag::get_coord();
        const short sn = sc.x, sm = sc.y;

        // Q: 16 heads x 128 columns, resident.
        frag_t Qf[TDh];
        {
          const device T* qp = q + (size_t(h_base) * S + t) * D + col0;
          for (short id = 0; id < TDh; id++)
            for (short i = 0; i < 2; i++)
              for (short j = 0; j < 4; j++)
                Qf[id][i * 4 + j] = qp[size_t(sm + i * 8) * hstride + id * kU + sn + j];
        }

        otile_t Otile; Otile.clear();
        float max_score[2], sum_score[2];
        for (short i = 0; i < 2; i++) {
          max_score[i] = 1.44269504089f * sinks[h_base + sm + i * 8];
          sum_score[i] = 1.0f;
        }

        threadgroup int rows_tg[BK];
        threadgroup float s_xchg[2][WN][TK * kEPF * 32];

        const int n_win = (window + BK - 1) / BK;
        const int n_blocks = n_win + (K + BK - 1) / BK;
        const device int* trow = topk + size_t(t) * K;
        for (int kb = 0; kb < n_blocks; kb++) {
          threadgroup_barrier(mem_flags::mem_threadgroup);
          if (tid < uint(BK)) {
            int row = -1;
            const int r = int(tid);
            if (kb < n_win) {
              const int c = kb * BK + r;
              const int j = t + R - S - (window - 1) + c;
              if (c < window && j >= 0) row = j;
            } else {
              const int i = (kb - n_win) * BK + r;
              if (i < K) {
                const int idx = trow[i];
                if (idx >= 0 && idx < P && (idx + 1) * ratio <= qpos + 1) row = R + idx;
              }
            }
            rows_tg[r] = row;
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
          bool any = false;
          for (short r = 0; r < BK; r++) any = any || (rows_tg[r] >= 0);
          if (!any) continue;  // uniform across the threadgroup

          // This lane's 4 key rows of the block: fragment ik = 0 / 1, element row i = 0 / 1 -> key ik * 16 + sm + i * 8.
          const device T* krow[TK][2];
          bool kval[TK][2];
          for (short ik = 0; ik < TK; ik++)
            for (short i = 0; i < 2; i++) {
              const int row = rows_tg[ik * kU + sm + i * 8];
              kval[ik][i] = row >= 0;
              krow[ik][i] = row < 0 ? kv : (row < R ? kv + size_t(row) * D : pool + size_t(row - R) * D);
            }

          // S = Q . K^T over this D group's 128 columns (2 x 16 keys).
          stile_t Stile; Stile.clear();
          for (short id = 0; id < TDh; id++) {
            frag_t k0, k1;
            const int c = col0 + id * kU + sn;
            for (short i = 0; i < 2; i++) {
              // 4 consecutive elements = one 8-byte load (c is a multiple of 4; rows are 1 KiB apart).
              const vec4_t a = kval[0][i] ? *(const device vec4_t*)(krow[0][i] + c) : vec4_t(0);
              const vec4_t b = kval[1][i] ? *(const device vec4_t*)(krow[1][i] + c) : vec4_t(0);
              for (short j = 0; j < 4; j++) { k0[i * 4 + j] = a[j]; k1[i * 4 + j] = b[j]; }
            }
            BaseNAXFrag::mma(Stile.frag_at(0, 0), Stile.frag_at(0, 1), Qf[id], metal::false_type{}, k0, k1,
                             metal::true_type{});
          }

          // Partial scores of the 4 D groups, summed in a fixed order (group 0, 1, 2, 3).
          threadgroup float* slot = s_xchg[row_group][d_group];
          const short base = short(lane) * (TK * kEPF);
          for (short ik = 0; ik < TK; ik++)
            for (short e = 0; e < kEPF; e++) slot[base + ik * kEPF + e] = Stile.frag_at(0, ik)[e];
          threadgroup_barrier(mem_flags::mem_threadgroup);
          for (short ik = 0; ik < TK; ik++) {
            thread auto& s = Stile.frag_at(0, ik);
            for (short e = 0; e < kEPF; e++) s[e] = s_xchg[row_group][0][base + ik * kEPF + e];
            for (short g = 1; g < WN; g++)
              for (short e = 0; e < kEPF; e++) s[e] += s_xchg[row_group][g][base + ik * kEPF + e];
          }

          // Scale (base 2) and mask the invisible keys: column ik * 16 + sn + j of this lane.
          for (short ik = 0; ik < TK; ik++) {
            thread auto& s = Stile.frag_at(0, ik);
            for (short j = 0; j < 4; j++) {
              const bool ok = rows_tg[ik * kU + sn + j] >= 0;
              for (short i = 0; i < 2; i++) {
                const float v = s[i * 4 + j] * scale2;
                s[i * 4 + j] = ok ? v : -INFINITY;
              }
            }
          }

          // Online softmax per head row (rows sm and sm + 8 of this lane).
          float new_max[2] = {max_score[0], max_score[1]};
          for (short ik = 0; ik < TK; ik++) {
            thread auto& s = Stile.frag_at(0, ik);
            for (short i = 0; i < 2; i++) {
              float m = max(max(s[i * 4], s[i * 4 + 1]), max(s[i * 4 + 2], s[i * 4 + 3]));
              m = max(m, simd_shuffle_xor(m, ushort(1)));
              m = max(m, simd_shuffle_xor(m, ushort(8)));
              new_max[i] = max(new_max[i], m);
            }
          }
          float factor[2];
          for (short i = 0; i < 2; i++) {
            factor[i] = fast::exp2(max_score[i] - new_max[i]);
            max_score[i] = new_max[i];
            sum_score[i] *= factor[i];
          }
          for (short ik = 0; ik < TK; ik++) {
            thread auto& s = Stile.frag_at(0, ik);
            for (short i = 0; i < 2; i++) {
              for (short j = 0; j < 4; j++) s[i * 4 + j] = fast::exp2(s[i * 4 + j] - new_max[i]);
              float r = (s[i * 4] + s[i * 4 + 1]) + (s[i * 4 + 2] + s[i * 4 + 3]);
              r += simd_shuffle_xor(r, ushort(1));
              r += simd_shuffle_xor(r, ushort(8));
              sum_score[i] += r;
            }
          }
          for (short f = 0; f < TDh; f++)
            for (short i = 0; i < 2; i++)
              for (short j = 0; j < 4; j++) Otile.frag_at(0, f)[i * 4 + j] *= factor[i];

          // O += P . V (V == K rows), this D group's 128 columns, 16 keys x 32 columns per MMA.
          for (short id = 0; id < TDh; id += 2) {
            for (short ik = 0; ik < TK; ik++) {
              frag_t v0, v1;
              const int c = col0 + id * kU + sn;
              for (short i = 0; i < 2; i++) {
                const vec4_t a = kval[ik][i] ? *(const device vec4_t*)(krow[ik][i] + c) : vec4_t(0);
                const vec4_t b = kval[ik][i] ? *(const device vec4_t*)(krow[ik][i] + c + kU) : vec4_t(0);
                for (short j = 0; j < 4; j++) { v0[i * 4 + j] = a[j]; v1[i * 4 + j] = b[j]; }
              }
              BaseNAXFrag::mma(Otile.frag_at(0, id), Otile.frag_at(0, id + 1), Stile.frag_at(0, ik),
                               metal::false_type{}, v0, v1, metal::false_type{});
            }
          }
        }

        // Normalize and store: 16 heads x 128 columns of token t.
        device T* op = out + (size_t(h_base) * S + t) * D + col0;
        for (short f = 0; f < TDh; f++)
          for (short i = 0; i < 2; i++) {
            const float rcp = 1.0f / sum_score[i];
            for (short j = 0; j < 4; j++)
              op[size_t(sm + i * 8) * hstride + f * kU + sn + j] = T(Otile.frag_at(0, f)[i * 4 + j] * rcp);
          }
        """

    private static let kernel = MLXFast.metalKernel(
        name: "vmlx_dsv41_indexed_attention_nax",
        inputNames: ["q", "kv", "pool", "topk", "sinks", "fscale", "params"],
        outputNames: ["out"],
        source: source,
        header: JANGHPrefillHeaders.nax,
        ensureRowContiguous: true)

    /// q (1, H, S, 512) bf16 / fp16 with H % 32 == 0; window rows `kv2d` (R, 512) whose last S rows are this forward's;
    /// dense pool `pool2d` (P, 512); picks `topk2d` (S, K) int32; `sinks` (H,) fp32. Returns (1, H, S, 512) in q's dtype,
    /// or nil off NAX or for another geometry.
    public static func attend(
        q: MLXArray, kv2d: MLXArray, pool2d: MLXArray, topk2d: MLXArray, sinks: MLXArray,
        offset: Int, window: Int, ratio: Int, scale: Float
    ) -> MLXArray? {
        guard available, q.ndim == 4, q.dim(0) == 1, q.dim(1) % 32 == 0, q.dim(3) == 512,
            [.bfloat16, .float16].contains(q.dtype), kv2d.ndim == 2, kv2d.dim(1) == 512, pool2d.ndim == 2,
            pool2d.dim(1) == 512, topk2d.ndim == 2, topk2d.dim(0) == q.dim(2), topk2d.dim(1) > 0,
            kv2d.dim(0) >= q.dim(2), window > 0, ratio > 0
        else { return nil }
        let heads = q.dim(1), s = q.dim(2)
        let params = MLXArray([
            Int32(s), Int32(kv2d.dim(0)), Int32(pool2d.dim(0)), Int32(topk2d.dim(1)), Int32(offset), Int32(window),
            Int32(ratio),
        ])
        return kernel(
            [q, kv2d.asType(q.dtype), pool2d.asType(q.dtype), topk2d.asType(.int32), sinks.asType(.float32),
             MLXArray([scale]), params],
            template: [("T", q.dtype)],
            grid: (256 * s, heads / 32, 1), threadGroup: (256, 1, 1),
            outputShapes: [[1, heads, s, 512]], outputDTypes: [q.dtype])[0]
    }
}
