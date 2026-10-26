// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Fused DSA index scores for DeepSeek-V4.1 prefill: out[s, p] = sum_h w[s, h] * relu(q[s, h, :] . k[p, :]).
//
// Port of vMLX `vmlx_engine/metal/dsa_index_scores.py` (`dsa_pool_index_scores`: `_SOURCE`, `_launch`,
// `_rows_frags`, `_admit`), as jangq `jang_tools/dsv41/model.py` `Indexer.__call__` uses it (jangq 3c64d3a).
// The stock indexer materializes the [C, h, T] bf16 scores and runs relu, the head weights and the head sum
// as separate passes, in query chunks that shrink with context (20 rows per chunk at 100k keys). This kernel
// keeps each (query, key) tile in registers: per head an 8x8 simdgroup-MMA dot product over head_dim, then
// relu, weight and accumulation, and writes the fp32 [S, P] result once (Python: 164 -> 97 ms per index
// layer per 2048-token chunk at 100k keys).
//
// Arithmetic: dot products accumulate over head_dim in 8-wide simdgroup-MMA steps, head 0..H-1 sequential
// with the product rounded before the add (no FMA contraction). Every output row is computed independently
// of the tile height and of how many queries share the call. Callers fall back to the stock path for
// anything not admitted.

import Foundation
import MLX
import MLXFast

enum DeepseekV41FusedIndexScores {
    /// `VMLX_DSV41_FUSED_INDEX=0` (or false/off/no) keeps the stock bf16 indexer (Python parity).
    static let enabled: Bool = {
        let raw = ProcessInfo.processInfo.environment["VMLX_DSV41_FUSED_INDEX"] ?? "1"
        return !["0", "false", "off", "no"].contains(
            raw.trimmingCharacters(in: .whitespaces).lowercased())
    }()

    /// Reference `GATHER_MAX_QUERIES`: decode and dSpark verify (<= 8 query rows) keep the stock path.
    static let maxStockQueries = 8

    /// Prefill only: more than 8 query rows and at least one key tile (Python
    /// `S > GATHER_MAX_QUERIES and T >= 32`), on the GPU. Off the GPU the indexer keeps the stock path
    /// with its own chunking.
    static func eligible(queries: Int, keys: Int) -> Bool {
        // METAL-ONLY: case 1. Guards the Metal kernel below; the stock MLX-op path serves other devices.
        enabled && queries > maxStockQueries && keys >= 32 && Device.defaultDevice().deviceType == .gpu
    }

    /// Query rows per fused call: keeps the [C, T] fp32 scores near 1 GB at any context
    /// (Python `max(16, min(S, 2**28 // T))`).
    static func queryChunk(queries: Int, keys: Int) -> Int {
        max(16, min(queries, (1 << 28) / max(1, keys)))
    }

    /// Fragments per simdgroup (8x8 each): 4 x 2 -> a 64 x 32 tile per threadgroup of 4 simdgroups.
    static let fi = 4
    static let fj = 2

    /// Fragments (8 rows each) per simdgroup row band: 64-row tiles, except 16-row tiles for calls under 32
    /// rows (a short tail pads to 16 rows instead of 64). Every output element is the same per-element
    /// arithmetic whatever the tile height.
    static func rowsFrags(_ rows: Int) -> Int { rows >= 32 ? 4 : 1 }

    /// Python `_admit`: D a multiple of 8, 2 <= L, Lk >= one 32-key tile (the last tile overlaps the previous
    /// one instead of reading past the end), fp32 operands, and 32-bit output indexing.
    static func admit(rows L: Int, keys Lk: Int, dim D: Int, arrays: [MLXArray]) -> Bool {
        let tn = 2 * fj * 8
        return !(D % 8 != 0 || L < 2 || Lk < tn || arrays.contains { $0.dtype != .float32 }
            || max(L, 64) * Lk >= (1 << 31))
    }

    static let header = "#include <metal_stdlib>\n#include <metal_simdgroup_matrix>\nusing namespace metal;\n"

    /// Verbatim `_SOURCE` of `vmlx_engine/metal/dsa_index_scores.py`.
    static let source = """
            #pragma clang fp contract(off)
            // Q element (h, s, d) at q + h * QH + s * QS + d: [H, L, D] (QH = L*D, QS = D) or [L, H, D] (QH = D, QS = H*D).
            const int Lq = int(QS_LAYOUT ? q_shape[1] : q_shape[2]);
            const int Lk = int(k_shape[K_AXIS]);
            const size_t QH = QS_LAYOUT ? size_t(D) : size_t(Lq) * D;
            const size_t QS = QS_LAYOUT ? size_t(H) * D : size_t(D);
            const uint sg = simdgroup_index_in_threadgroup;
            const uint lane = thread_index_in_simdgroup;
            const int c0 = min(int(threadgroup_position_in_grid.x) * TN, Lk - TN);
            const int r0 = min(int(threadgroup_position_in_grid.y) * TM, Lq - TM);
            const int sr = r0 + int(sg / 2) * (FI * 8);
            const int sc = c0 + int(sg % 2) * (FJ * 8);
            const short qid = short(lane / 4);
            const short fm = (qid & 4) + short((lane / 2) % 4);
            const short fn = (qid & 2) * 2 + short(lane % 2) * 2;
            float out_acc[FI][FJ][2];
            for (int i = 0; i < FI; ++i) for (int j = 0; j < FJ; ++j) { out_acc[i][j][0] = 0.0f; out_acc[i][j][1] = 0.0f; }
            for (int h = 0; h < H; ++h) {
                simdgroup_float8x8 acc[FI][FJ];
                for (int i = 0; i < FI; ++i) for (int j = 0; j < FJ; ++j) acc[i][j] = simdgroup_float8x8(0.0f);
                const device float* qh = q + (size_t)h * QH;
                for (int d = 0; d < D; d += 8) {
                    simdgroup_float8x8 a[FI];
                    simdgroup_float8x8 b[FJ];
                    for (int i = 0; i < FI; ++i)
                        simdgroup_load(a[i], qh + (size_t)(sr + i * 8) * QS + d, QS);
                    for (int j = 0; j < FJ; ++j)
                        simdgroup_load(b[j], k + (size_t)(sc + j * 8) * D + d, D, ulong2(0, 0), true);
                    for (int i = 0; i < FI; ++i)
                        for (int j = 0; j < FJ; ++j)
                            simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
                }
                for (int i = 0; i < FI; ++i) {
                    const float wv = w[(size_t)(sr + i * 8 + fm) * H + h];
                    for (int j = 0; j < FJ; ++j) {
                        thread auto& e = acc[i][j].thread_elements();
                        float s0 = e[0];
                        float s1 = e[1];
                        if (HAS_SCALE) { s0 = s0 * scale[0]; s1 = s1 * scale[0]; }
                        float p0 = metal::max(s0, 0.0f) * wv;
                        float p1 = metal::max(s1, 0.0f) * wv;
                        out_acc[i][j][0] = out_acc[i][j][0] + p0;
                        out_acc[i][j][1] = out_acc[i][j][1] + p1;
                    }
                }
            }
            for (int i = 0; i < FI; ++i) {
                const size_t row = (size_t)(sr + i * 8 + fm);
                for (int j = 0; j < FJ; ++j) {
                    const size_t col = (size_t)(sc + j * 8 + fn);
                    out[row * (size_t)Lk + col] = out_acc[i][j][0];
                    out[row * (size_t)Lk + col + 1] = out_acc[i][j][1];
                }
            }
        """

    // METAL-ONLY: case 2. Metal kernel; `DeepseekV41IndexerMath.scores` computes the same with MLX ops and
    // `poolIndexScores` returns nil off the GPU, so the indexer takes that path there.
    private static let kernel = MLXFast.metalKernel(
        name: "vmlx_dsa_fused_index_scores_v1",
        inputNames: ["q", "k", "w", "scale"],
        outputNames: ["out"],
        source: source,
        header: header,
        ensureRowContiguous: true)

    /// Python `_launch` with `qs_layout=True, k_axis=1` (the GLM/V4.1 pool layout).
    static func launch(
        q: MLXArray, k: MLXArray, w: MLXArray, heads H: Int, rows L: Int, dim D: Int, keys Lk: Int,
        scale: Float?, fi: Int
    ) -> MLXArray {
        let tm = 2 * fi * 8
        let tn = 2 * fj * 8
        let sc = MLXArray([scale ?? 1.0])
        return kernel(
            [q, k, w, sc],
            template: [
                ("H", H), ("D", D), ("FI", fi), ("FJ", fj), ("TM", tm), ("TN", tn),
                ("QS_LAYOUT", 1), ("K_AXIS", 1), ("HAS_SCALE", scale != nil ? 1 : 0),
            ],
            grid: (((Lk + tn - 1) / tn) * 128, (L + tm - 1) / tm, 1),
            threadGroup: (128, 1, 1),
            outputShapes: [[1, L, Lk]],
            outputDTypes: [.float32])[0]
    }

    /// Python `dsa_pool_index_scores`. q (1, S, H, D) fp32, poolKeys (1, P, D) fp32, headW (1, S, H) fp32
    /// -> (1, S, P) fp32 = sum_h headW * relu((q . poolKey) * scale); nil when not admitted or off the GPU.
    /// `rowsFrags` overrides the tile height (tests only).
    static func poolIndexScores(
        q: MLXArray, poolKeys: MLXArray, headW: MLXArray, scale: Float, rowsFrags override: Int? = nil
    ) -> MLXArray? {
        // METAL-ONLY: case 1. Guards the Metal kernel above; off the GPU the caller takes the MLX-op path.
        guard Device.defaultDevice().deviceType == .gpu else { return nil }
        guard q.ndim == 4, poolKeys.ndim == 3, headW.ndim == 3 else { return nil }
        let B = q.dim(0), S = q.dim(1), H = q.dim(2), D = q.dim(3)
        guard B == 1, poolKeys.dim(0) == 1, poolKeys.dim(2) == D, headW.shape == [1, S, H] else {
            return nil
        }
        let P = poolKeys.dim(1)
        guard admit(rows: S, keys: P, dim: D, arrays: [q, poolKeys, headW]) else { return nil }
        let fi = override ?? rowsFrags(S)
        let tm = 2 * fi * 8
        if S < tm {
            // A short tail takes the same arithmetic as a full chunk: zero-pad the queries to one tile and
            // keep the first S rows (each row is computed independently).
            let qp = padded(q, widths: [0, .init((0, tm - S)), 0, 0])
            let wp = padded(headW, widths: [0, .init((0, tm - S)), 0])
            let out = launch(
                q: qp, k: poolKeys, w: wp, heads: H, rows: tm, dim: D, keys: P, scale: scale, fi: fi)
            return out[0..., 0 ..< S]
        }
        return launch(
            q: q, k: poolKeys, w: headW, heads: H, rows: S, dim: D, keys: P, scale: scale, fi: fi)
    }
}
