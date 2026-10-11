// SPDX-License-Identifier: Apache-2.0
// Portions adapted from TensorFold 0.6.6 (https://github.com/ashhart/TensorFold, Apache-2.0;
// Copyright 2026 TensorFold contributors), src/tensorfold/kernels/qwen/dense/v1/lane_qmm.py and
// lane_widen.py at commit cb2ebf0, via vMLX Python vmlx_engine/metal/lane_qmm.py (commit 4554485c1).
// The Metal kernel sources below are byte-identical to the Python port; the `directScales` variants
// (default since 2026-10-11) differ only in reading scales/biases in place instead of a packed SBt copy.
//
// Row-independent affine quantized matmul for 1...128 rows on the M5 matrix units ("lane matmul").
//
// WHY: MLX's quantized matmul picks a different kernel per row count (qmv at 1 row, qmv variants for a
// few rows, a qmm tile from ~6 rows). On Qwen3.8-27B JANG_4D a DFlash2 verify of 8 rows cost 1.8x one row.
// This kernel always runs a 16-row `matmul2d` (Metal 4 MetalPerformancePrimitives), so 1...16 rows cost
// about the same and each row's bits are independent of how many rows share the call.
// HOW: the matrix unit multiplies bf16 activations by the RAW unsigned 4-bit codes of one 64-value weight
// group (fp32 accumulate); the affine dequant is applied after the dot product:
//   C += s[g,n] * (X_g @ Q_g)[m,n] + b[g,n] * XS[g,m],  XS[g,m] = sum of x over group g.
// 5/6/8-bit codes are widened to bytes in threadgroup memory (uint8 op). K is split over SK simdgroups
// fixed by the weight shape (never by M) and reduced in slice order.
// TRADE-OFFS: not bit-identical to MLX qmv/qmm (different summation order); ~11 % slower than MLX at 1 row;
// affine bf16-scale weights only, group 64 (or 32 for 4-bit), K % 64 == 0, N % 4 == 0; needs an M5-class GPU.

import Foundation
import MLX
import MLXNN

public enum LaneQMM {
    public static let maxRows = 128
    static let rowBlock = 32
    static let nt = 32
    static let readableBits: Set<Int> = [2, 3, 4, 5, 6, 8]

    static let header = #"""

        #include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
        using namespace mpp::tensor_ops;
        """#

    static let xsumSource = #"""

          const int M = mdims[0], MP = mdims[1];
          const uint m = thread_position_in_grid.y;
          const uint g = thread_position_in_grid.x;
          if (g >= K / GS || int(m) >= MP) return;
          float acc = 0.0f;
          if (int(m) < M) for (int i = 0; i < GS; i++) acc += float(X[m * K + g * GS + i]);
          XS[g * MP + m] = acc;
        """#

    static let mainSource = #"""

          const ushort lane = thread_index_in_simdgroup;
          const ushort sg = simdgroup_index_in_threadgroup;     // K slice
          const short qid = lane >> 2;
          const short fm = (qid & 4) | ((lane >> 1) & 3);       // fragment row of this lane (and fm + 8)
          const short fn = ((qid & 2) | (lane & 1)) * 4;        // first of its four fragment columns
          const int M = mdims[0], MP = mdims[1];
          constexpr int KG = K / GS;
          constexpr int NF = NT / 16;
          const int n0 = threadgroup_position_in_grid.x * NT;
          const int rb = threadgroup_position_in_grid.y * 16 * TMR;   // first row of this threadgroup's row block
          const int g_begin = (sg * KG) / SK;
          const int g_end = ((sg + 1) * KG) / SK;

          // one op for all TMR 16-row blocks: each row gets the 16-row op's bits
          constexpr auto desc = matmul2d_descriptor(16 * TMR, NT, GS, false, true, false, matmul2d_descriptor::mode::multiply);
          matmul2d<desc, execution_simdgroup> op;
          tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tA((device bfloat*)X + (int64_t)rb * K, dextents<int32_t, 2>(K, M - rb));
          tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline> tB((device uchar*)Wq, dextents<int32_t, 2>(K, N));

          float C[TMR][NF * 8];
          for (int t = 0; t < TMR; t++) for (int i = 0; i < NF * 8; i++) C[t][i] = 0.0f;
          const device uint4* sbv = (const device uint4*)SBt;   // (s, b) bf16 pairs, [g][n]
          bool colok[NF];
          for (int f = 0; f < NF; f++) colok[f] = n0 + f * 16 + fn < N;
          for (int g = g_begin; g < g_end; g++) {
            float s[NF][4], bb[NF][4];
            for (int f = 0; f < NF; f++) {
              const uint4 q = colok[f] ? sbv[(g * N + n0 + f * 16 + fn) / 4] : uint4(0);
              const vec<bfloat, 8> v = as_type<vec<bfloat, 8>>(q);
              for (int j = 0; j < 4; j++) { s[f][j] = float(v[2 * j]); bb[f][j] = float(v[2 * j + 1]); }
            }
            auto a = tA.slice(g * GS, 0);
            auto b = tB.slice(g * GS, n0);
            auto P = op.template get_destination_cooperative_tensor<decltype(a), decltype(b), float>();
            op.run(a, b, P);
            for (int t = 0; t < TMR; t++) {
              const bool live = !EDGE || rb + t * 16 < MP;     // EDGE: the last 32-row block passes MP, where XS ends
              const float xs0 = live ? XS[g * MP + rb + t * 16 + fm] : 0.0f;
              const float xs1 = live ? XS[g * MP + rb + t * 16 + fm + 8] : 0.0f;
              for (int f = 0; f < NF; f++)
                for (int r = 0; r < 2; r++)
                  for (int j = 0; j < 4; j++) {
                    const int i = f * 8 + r * 4 + j;
                    C[t][i] = fma(s[f][j], P[t * NF * 8 + i], fma(bb[f][j], r ? xs1 : xs0, C[t][i]));
                  }
            }
          }
          // K slices are added in slice order, one 16-row block at a time
          threadgroup float part[(SK > 1 ? SK - 1 : 1) * NF * 8 * 32];
          for (int t = 0; t < TMR; t++) {
            if (SK > 1) {
              if (sg > 0) for (int i = 0; i < NF * 8; i++) part[((sg - 1) * NF * 8 + i) * 32 + lane] = C[t][i];
              threadgroup_barrier(mem_flags::mem_threadgroup);
              if (sg == 0)
                for (int s2 = 1; s2 < SK; s2++) for (int i = 0; i < NF * 8; i++) C[t][i] += part[((s2 - 1) * NF * 8 + i) * 32 + lane];
              threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (sg == 0)
              for (int f = 0; f < NF; f++)
                for (int r = 0; r < 2; r++) {
                  const int m = rb + t * 16 + fm + 8 * r;
                  const int n = n0 + f * 16 + fn;
                  if (m < M && n < N)
                    for (int j = 0; j < 4; j++) Y[m * N + n + j] = static_cast<bfloat>(C[t][f * 8 + r * 4 + j]);
                }
          }
        """#

    static let mainTiledSource = #"""

          const ushort lane = thread_index_in_simdgroup;
          const ushort sg = simdgroup_index_in_threadgroup;     // K slice
          const short qid = lane >> 2;
          const short fm = (qid & 4) | ((lane >> 1) & 3);       // fragment row of this lane (and fm + 8)
          const short fn = ((qid & 2) | (lane & 1)) * 4;        // first of its four fragment columns
          const int M = mdims[0], MP = mdims[1];
          constexpr int KG = K / GS;
          constexpr int NF = NT / 16;
          const int n0 = threadgroup_position_in_grid.x * NT;
          const int rb = threadgroup_position_in_grid.y * 16 * TMR;   // first row of this threadgroup's row block
          const int g_begin = (sg * KG) / SK;
          const int g_end = ((sg + 1) * KG) / SK;

          // one op for all TMR 16-row blocks: each row gets the 16-row op's bits
          constexpr auto desc = matmul2d_descriptor(16 * TMR, NT, GS, false, true, false, matmul2d_descriptor::mode::multiply);
          matmul2d<desc, execution_simdgroup> op;
          tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tA((device bfloat*)X + (int64_t)rb * K, dextents<int32_t, 2>(K, M - rb));
          tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline> tB((device uchar*)Wq, dextents<int32_t, 2>(K, N));

          float C[TMR][NF * 8];
          for (int t = 0; t < TMR; t++) for (int i = 0; i < NF * 8; i++) C[t][i] = 0.0f;
          const device uint4* sbv = (const device uint4*)SBt;   // (s, b) bf16 pairs, [g][n]
          bool colok[NF];
          for (int f = 0; f < NF; f++) colok[f] = n0 + f * 16 + fn < N;
          for (int g = g_begin; g < g_end; g++) {
            float s[NF][4], bb[NF][4];
            for (int f = 0; f < NF; f++) {
              const uint4 q = colok[f] ? sbv[(g * N + n0 + f * 16 + fn) / 4] : uint4(0);
              const vec<bfloat, 8> v = as_type<vec<bfloat, 8>>(q);
              for (int j = 0; j < 4; j++) { s[f][j] = float(v[2 * j]); bb[f][j] = float(v[2 * j + 1]); }
            }
            auto a = tA.slice(g * GS, 0);
            tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline> b(
                (device uchar*)Wq + (int64_t)(threadgroup_position_in_grid.x * KG + g) * (NT * GS / 2), dextents<int32_t, 2>(GS, NT));
            auto P = op.template get_destination_cooperative_tensor<decltype(a), decltype(b), float>();
            op.run(a, b, P);
            for (int t = 0; t < TMR; t++) {
              const bool live = !EDGE || rb + t * 16 < MP;     // EDGE: the last 32-row block passes MP, where XS ends
              const float xs0 = live ? XS[g * MP + rb + t * 16 + fm] : 0.0f;
              const float xs1 = live ? XS[g * MP + rb + t * 16 + fm + 8] : 0.0f;
              for (int f = 0; f < NF; f++)
                for (int r = 0; r < 2; r++)
                  for (int j = 0; j < 4; j++) {
                    const int i = f * 8 + r * 4 + j;
                    C[t][i] = fma(s[f][j], P[t * NF * 8 + i], fma(bb[f][j], r ? xs1 : xs0, C[t][i]));
                  }
            }
          }
          // K slices are added in slice order, one 16-row block at a time
          threadgroup float part[(SK > 1 ? SK - 1 : 1) * NF * 8 * 32];
          for (int t = 0; t < TMR; t++) {
            if (SK > 1) {
              if (sg > 0) for (int i = 0; i < NF * 8; i++) part[((sg - 1) * NF * 8 + i) * 32 + lane] = C[t][i];
              threadgroup_barrier(mem_flags::mem_threadgroup);
              if (sg == 0)
                for (int s2 = 1; s2 < SK; s2++) for (int i = 0; i < NF * 8; i++) C[t][i] += part[((s2 - 1) * NF * 8 + i) * 32 + lane];
              threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (sg == 0)
              for (int f = 0; f < NF; f++)
                for (int r = 0; r < 2; r++) {
                  const int m = rb + t * 16 + fm + 8 * r;
                  const int n = n0 + f * 16 + fn;
                  if (m < M && n < N)
                    for (int j = 0; j < 4; j++) Y[m * N + n + j] = static_cast<bfloat>(C[t][f * 8 + r * 4 + j]);
                }
          }
        """#

    /// 8-bit codes ARE bytes: MLX packs an 8-bit weight as a row-major [N][K] uint8 matrix, so the matrix
    /// unit can take it straight from device memory as a `uint8_t` operand — no bit-stream widening into
    /// threadgroup memory, no staging barriers (what `bytesSource` does for 5/6/8-bit). Same descriptor,
    /// same per-group fma epilogue and K-slice reduction as the 4-bit kernels, so the result is admitted
    /// bitwise against `bytesSource` (LaneQMMDirect8Tests). Untiled: rows of the MLX layout; tiled: a
    /// (tile, group) block is NT columns x 64 contiguous bytes.
    static let main8Source: String = {
        let replaced = mainSource.replacingOccurrences(
            of: "tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline> tB((device uchar*)Wq",
            with: "tensor<device uint8_t, dextents<int32_t, 2>, tensor_inline> tB((device uint8_t*)Wq")
        precondition(!replaced.contains("uint4b_format"), "main8Source: 4-bit operand left")
        return replaced
    }()

    static let main8TiledSource: String = {
        let replaced = mainTiledSource
            .replacingOccurrences(
                of: "tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline> tB((device uchar*)Wq",
                with: "tensor<device uint8_t, dextents<int32_t, 2>, tensor_inline> tB((device uint8_t*)Wq")
            .replacingOccurrences(
                of: "tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline> b(",
                with: "tensor<device uint8_t, dextents<int32_t, 2>, tensor_inline> b(")
            .replacingOccurrences(
                of: "(device uchar*)Wq + (int64_t)(threadgroup_position_in_grid.x * KG + g) * (NT * GS / 2)",
                with: "(device uint8_t*)Wq + (int64_t)(threadgroup_position_in_grid.x * KG + g) * (NT * GS)")
        precondition(!replaced.contains("uint4b_format"), "main8TiledSource: 4-bit operand left")
        return replaced
    }()

    static let bytesSource = #"""

          static_assert(NT == 32, "one column per lane");
          static_assert(BITS == 5 || BITS == 6 || BITS == 8, "bytes for 5-, 6- and 8-bit weights");
          const ushort lane = thread_index_in_simdgroup;
          const ushort sg = simdgroup_index_in_threadgroup;     // K slice
          const short qid = lane >> 2;
          const short fm = (qid & 4) | ((lane >> 1) & 3);
          const short fn = ((qid & 2) | (lane & 1)) * 4;
          const int M = mdims[0], MP = mdims[1];
          constexpr int KG = K / 64;
          constexpr int NF = NT / 16;
          constexpr int WPG = 2 * BITS;                          // words per column per group: 64 values x BITS bits
          constexpr int KW = K * BITS / 32;                      // words per column
          const int n0 = threadgroup_position_in_grid.x * NT;
          const int rb = threadgroup_position_in_grid.y * 16 * TMR;
          const int g_begin = (sg * KG) / SK;
          const int g_end = ((sg + 1) * KG) / SK;
          constexpr auto desc = matmul2d_descriptor(16 * TMR, NT, 64, false, true, false, matmul2d_descriptor::mode::multiply);
          matmul2d<desc, execution_simdgroup> op;
          tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tA((device bfloat*)X + (int64_t)rb * K, dextents<int32_t, 2>(K, M - rb));
          threadgroup uint stage_all[SK * NT * 16];              // per K slice: NT columns x 64 bytes
          threadgroup uint* stage = stage_all + sg * NT * 16;
          tensor<threadgroup uint8_t, dextents<int32_t, 2>, tensor_inline> b((threadgroup uint8_t*)stage, dextents<int32_t, 2>(64, NT));
          const device uint* Wv = (const device uint*)Wq;
          const int n = n0 + lane;

          float C[TMR][NF * 8];
          for (int t = 0; t < TMR; t++) for (int i = 0; i < NF * 8; i++) C[t][i] = 0.0f;
          const device uint4* sbv = (const device uint4*)SBt;
          bool colok[NF];
          for (int f = 0; f < NF; f++) colok[f] = n0 + f * 16 + fn < N;
          for (int g = g_begin; g < g_end; g++) {
            uint w[WPG + 1];
            for (int i = 0; i <= WPG; i++) w[i] = 0;
            if (n < N) {
              const device uint* src = TILED ? Wv + ((int64_t)(threadgroup_position_in_grid.x * KG + g) * NT + lane) * WPG
                                             : Wv + (int64_t)n * KW + g * WPG;
              for (int i = 0; i < WPG; i++) w[i] = src[i];
            }
            for (int c = 0; c < 16; c++) {
              const int bit = 4 * BITS * c, i = bit >> 5, sh = bit & 31;
              uint word = w[i] >> sh;
              if (sh + 4 * BITS > 32) word |= w[i + 1] << (32 - sh);
              word = (word & ((1u << (2 * BITS)) - 1u)) | (((word >> (2 * BITS)) & ((1u << (2 * BITS)) - 1u)) << 16);
              word = (word & ((0x10001u << BITS) - 0x10001u)) | (((word >> BITS) & ((0x10001u << BITS) - 0x10001u)) << 8);
              stage[lane * 16 + c] = word;
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            float s[NF][4], bb[NF][4];
            for (int f = 0; f < NF; f++) {
              const uint4 q = colok[f] ? sbv[(g * N + n0 + f * 16 + fn) / 4] : uint4(0);
              const vec<bfloat, 8> v = as_type<vec<bfloat, 8>>(q);
              for (int j = 0; j < 4; j++) { s[f][j] = float(v[2 * j]); bb[f][j] = float(v[2 * j + 1]); }
            }
            auto a = tA.slice(g * 64, 0);
            auto P = op.template get_destination_cooperative_tensor<decltype(a), decltype(b), float>();
            op.run(a, b, P);
            simdgroup_barrier(mem_flags::mem_threadgroup);   // the op has read the stage before the next group's widening
            for (int t = 0; t < TMR; t++) {
              // the last row block can run past MP (MP % 32 == 16): those rows are never stored, and XS ends at MP
              const bool live = rb + t * 16 < MP;
              const float xs0 = live ? XS[g * MP + rb + t * 16 + fm] : 0.0f;
              const float xs1 = live ? XS[g * MP + rb + t * 16 + fm + 8] : 0.0f;
              for (int f = 0; f < NF; f++)
                for (int r = 0; r < 2; r++)
                  for (int j = 0; j < 4; j++) {
                    const int i = f * 8 + r * 4 + j;
                    C[t][i] = fma(s[f][j], P[t * NF * 8 + i], fma(bb[f][j], r ? xs1 : xs0, C[t][i]));
                  }
            }
          }
          threadgroup float part[(SK > 1 ? SK - 1 : 1) * NF * 8 * 32];
          for (int t = 0; t < TMR; t++) {
            if (SK > 1) {
              if (sg > 0) for (int i = 0; i < NF * 8; i++) part[((sg - 1) * NF * 8 + i) * 32 + lane] = C[t][i];
              threadgroup_barrier(mem_flags::mem_threadgroup);
              if (sg == 0)
                for (int s2 = 1; s2 < SK; s2++) for (int i = 0; i < NF * 8; i++) C[t][i] += part[((s2 - 1) * NF * 8 + i) * 32 + lane];
              threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (sg == 0)
              for (int f = 0; f < NF; f++)
                for (int r = 0; r < 2; r++) {
                  const int m = rb + t * 16 + fm + 8 * r;
                  const int nn = n0 + f * 16 + fn;
                  if (m < M && nn < N)
                    for (int j = 0; j < 4; j++) Y[m * N + nn + j] = static_cast<bfloat>(C[t][f * 8 + r * 4 + j]);
                }
          }
        """#

    // MARK: direct scale variants (no packed SBt copy)

    /// The SBt scale read of every lane kernel, replaced by a read of the module's OWN scales and biases,
    /// (N, SKG) bf16 each, scale group `g / SREP` (SREP = 2 when a group-128 bundle runs the 64-value
    /// bytes kernel). Same values, same arithmetic: bit-identical to the SBt kernel, and no packed
    /// (K/GS, N, 2) copy (1.6 GB on Qwen3.8-27B JANG_4D).
    /// COST, measured 2026-10-11 at 27B shapes, 4-bit g64, 8-16 rows: 0.42-0.54 ms vs 0.18-0.30 ms
    /// packed. It is the per-group instruction count, not memory traffic: a variant with NO scale loads
    /// (constants) still took 0.44 ms, and a coalesced per-lane load + simd_shuffle variant 0.45-0.57 ms.
    /// So this is a per-target choice (``LaneQMMTargetPolicy/laneQMMPackedScales``): the 27B keeps
    /// the packed copy (DFlash2 decode 22-32 vs 13-20 tok/s, i.e. AR speed or below without it);
    /// Diplodocus/K2 Uno reads in place (same speed, -250 MB).
    static func directScales(_ source: String) -> String {
        let edits = [
            ("const device uint4* sbv = (const device uint4*)SBt;",
             "// scales and biases read in place: S, B are (N, SKG) bf16"),
            ("const uint4 q = colok[f] ? sbv[(g * N + n0 + f * 16 + fn) / 4] : uint4(0);", ""),
            ("const vec<bfloat, 8> v = as_type<vec<bfloat, 8>>(q);", ""),
            ("for (int j = 0; j < 4; j++) { s[f][j] = float(v[2 * j]); bb[f][j] = float(v[2 * j + 1]); }",
             "for (int j = 0; j < 4; j++) { const int64_t o = (int64_t)(n0 + f * 16 + fn + j) * SKG + g / SREP; "
                + "s[f][j] = colok[f] ? float(S[o]) : 0.0f; bb[f][j] = colok[f] ? float(B[o]) : 0.0f; }"),
        ]
        var replaced = source
        for (old, new) in edits {
            precondition(replaced.contains(old), "lane source: \(old) not found")
            replaced = replaced.replacingOccurrences(of: old, with: new)
        }
        precondition(!replaced.contains("SBt") && !replaced.contains("sbv"), "lane source: SBt left")
        return replaced
    }
}
