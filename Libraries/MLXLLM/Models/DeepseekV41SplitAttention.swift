// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// DeepSeek-V4.1 sparse attention for decode and dSpark verify rows (<= 8), split by keys.
//
// The stock path (`DeepseekV4Math.selectedPoolAttention`) gathers and dequantizes the indexer's picks from the
// 4-bit pool, runs an online softmax over the sliding window, then two fp32 einsums over the picks: about a
// dozen dispatches plus the [S, K, D] gathered copy per layer per step. Here a threadgroup per (16 heads, 16 keys,
// row) loads its keys once into threadgroup memory -- window rows from the local cache, picks decoded in place
// from the contiguous 4-bit slab with the same arithmetic as `gatherDequantizedRows` -- and writes the chunk's
// max, sum and unnormalized output; a merge kernel folds the chunks and the attention sink into one softmax.
// Visibility is the stock path's: `buildWindowMask` for the window, `(row + 1) * ratio <= position + 1` for picks.
//
// The sums run in another order than the stock einsums (and in full fp32 where MLX's fp32 matmuls round to TF32
// on M5), so the result is equal to rounding, not bit-identical. A once-per-process check against the stock path
// keeps the stock path on any build / launch failure or a difference beyond rounding. Python mirror: jangq
// `jang_tools/dsv41/metal_decode.py` `attend_decode` (VMLX_DSV41_ATTN_SPLIT).

import Foundation
import MLX
import MLXFast

enum DeepseekV41SplitAttention {
    /// `VMLX_DSV41_ATTN_SPLIT=0` (or false/off/no) keeps the stock gather + online-softmax path.
    static let enabled: Bool = {
        let raw = ProcessInfo.processInfo.environment["VMLX_DSV41_ATTN_SPLIT"] ?? "1"
        return !["0", "false", "off", "no"].contains(
            raw.trimmingCharacters(in: .whitespaces).lowercased())
    }()

    /// Decode and dSpark verify rows (reference `GATHER_MAX_QUERIES`).
    static let maxRows = 8
    /// Key chunks of 16 per (row, head): the window (<= 128 + verify rows) and the indexer's picks (<= 512).
    static let maxChunks = 48

    static let header = "#include <metal_stdlib>\nusing namespace metal;\n"

    static let splitSource = """
            #pragma clang fp contract(off)
            const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
            const uint t = thread_position_in_threadgroup.x;
            const int h = int(threadgroup_position_in_grid.x) * 16 + int(sg);
            const int chunk = int(threadgroup_position_in_grid.y), s = int(threadgroup_position_in_grid.z);
            const int W = dims[0], KC = dims[1], S = dims[2];
            const int k0 = chunk * 16, k1 = metal::min(W + KC, k0 + 16), n = k1 - k0;
            threadgroup T ks[16][HD];
            threadgroup float sc[16][16];
            threadgroup bool ok[16];
            for (int i = int(t); i < 16 * (HD / 8); i += 512) {
                const int kk = i / (HD / 8), c8 = i % (HD / 8), j = k0 + kk;
                T e8[8];
                bool valid = false;
                if (j < k1 && j < W) {
                    const int ix = widx[s * W + j];
                    valid = ix >= 0;
                    if (valid) {
                        for (int e = 0; e < 8; e++) e8[e] = raw[(size_t)ix * HD + c8 * 8 + e];
                    }
                } else if (j < k1) {
                    const int c = cidx[s * KC + (j - W)];
                    valid = c >= 0;
                    if (valid) {
                        const uint w32 = ((const device uint*)(codes + (size_t)c * (HD / 2)))[c8];
                        const size_t g = (size_t)c * (HD / GS) + (size_t)(c8 * 8) / GS;
                        const float scale = float(scales[g]);
                        const float bias = float(biases[g]);
                        for (int e = 0; e < 8; e++) {
                            const float prod = float((w32 >> (4 * e)) & 0xFu) * scale;
                            e8[e] = static_cast<T>(prod + bias);
                        }
                    }
                }
                if (!valid) {
                    for (int e = 0; e < 8; e++) e8[e] = static_cast<T>(0.0f);
                }
                for (int e = 0; e < 8; e++) ks[kk][c8 * 8 + e] = e8[e];
                if (c8 == 0) ok[kk] = valid;
            }
            float qv[16];
            const device T* qrow = q + ((size_t)h * S + s) * HD;
            for (int i = 0; i < 16; i++) qv[i] = float(qrow[lane * 16 + i]);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (int kk = 0; kk < n; kk++) {
                float d = 0.0f;
                for (int i = 0; i < 16; i++) d += qv[i] * float(ks[kk][lane * 16 + i]);
                d = simd_sum(d);
                if (lane == 0) sc[sg][kk] = ok[kk] ? d * scl[0] : -INFINITY;
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            float m = -1.0e30f;
            for (int j = int(lane); j < n; j += 32) m = metal::max(m, sc[sg][j]);
            m = simd_max(m);
            float l = 0.0f;
            for (int j = int(lane); j < n; j += 32) {
                const float p = metal::precise::exp(sc[sg][j] - m);
                sc[sg][j] = p;
                l += p;
            }
            l = simd_sum(l);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            float o[16];
            for (int i = 0; i < 16; i++) o[i] = 0.0f;
            for (int kk = 0; kk < n; kk++) {
                const float p = sc[sg][kk];
                for (int i = 0; i < 16; i++) o[i] += p * float(ks[kk][lane * 16 + i]);
            }
            const size_t part = ((size_t)s * NH + h) * CHMAX + chunk;
            for (int i = 0; i < 16; i++) po[part * HD + lane * 16 + i] = o[i];
            if (lane == 0) {
                pml[part * 2] = m;
                pml[part * 2 + 1] = l;
            }
        """

    static let mergeSource = """
            #pragma clang fp contract(off)
            const uint t = thread_position_in_threadgroup.x;
            const int h = int(threadgroup_position_in_grid.y), s = int(threadgroup_position_in_grid.z);
            const int S = dims[2];
            const int CH = (dims[0] + dims[1] + 15) / 16;
            threadgroup float wgt[CHMAX];
            threadgroup float lsum[CHMAX];
            const size_t base = ((size_t)s * NH + h) * CHMAX;
            float m = sink[h];
            for (int c = 0; c < CH; c++) m = metal::max(m, pml[(base + c) * 2]);
            for (int c = int(t); c < CH; c += HD) {
                wgt[c] = metal::precise::exp(pml[(base + c) * 2] - m);
                lsum[c] = pml[(base + c) * 2 + 1];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float denom = metal::precise::exp(sink[h] - m);
            for (int c = 0; c < CH; c++) denom += lsum[c] * wgt[c];
            float acc = 0.0f;
            for (int c = 0; c < CH; c++) acc += po[(base + c) * HD + t] * wgt[c];
            out[((size_t)h * S + s) * HD + t] = acc / denom;
        """

    // METAL-ONLY: case 2. Metal kernels; `attend` returns nil off the GPU and the caller keeps the stock path.
    private static let splitKernel = MLXFast.metalKernel(
        name: "vmlx_dsv41_attend_split_v1",
        inputNames: ["q", "raw", "widx", "cidx", "codes", "scales", "biases", "dims", "scl"],
        outputNames: ["po", "pml"],
        source: splitSource,
        header: header,
        ensureRowContiguous: true)

    private static let mergeKernel = MLXFast.metalKernel(
        name: "vmlx_dsv41_attend_merge_v1",
        inputNames: ["po", "pml", "sink", "dims"],
        outputNames: ["out"],
        source: mergeSource,
        header: header,
        ensureRowContiguous: true)

    /// Local + pool attention for `S <= 8` rows, as `DeepseekV4Math.tiledPoolAttention`: q (1, H, S, D) bf16 (RoPE
    /// applied), localKV (1, 1, Lw, D) bf16, the pool as a contiguous 4-bit slab, topK (1, S, K) selected pool rows,
    /// or nil for every pool row (no indexer selection while the pool has at most `index_topk` rows). Returns
    /// (1, H, S, D) in q's dtype, or nil when not admitted (the caller keeps the stock path).
    static func attend(
        queries q: MLXArray, localKV: MLXArray, pooled: DeepseekV4PoolView, topK: MLXArray?,
        offset: Int, window: Int, ratio: Int, scale: Float, sinks: MLXArray?
    ) -> MLXArray? {
        guard enabled, admitted(queries: q, localKV: localKV, pooled: pooled, topK: topK) else { return nil }
        guard selfCheckPassed else { return nil }
        return launch(
            queries: q, localKV: localKV, pooled: pooled, topK: topK, offset: offset, window: window,
            ratio: ratio, scale: scale, sinks: sinks)
    }

    static func admitted(queries q: MLXArray, localKV: MLXArray, pooled: DeepseekV4PoolView, topK: MLXArray?)
        -> Bool
    {
        // METAL-ONLY: case 1. Guards the Metal kernels above; other devices keep the MLX-op path.
        guard Device.defaultDevice().deviceType == .gpu, q.ndim == 4, localKV.ndim == 4, (topK?.ndim ?? 3) == 3 else {
            return false
        }
        let B = q.dim(0), H = q.dim(1), S = q.dim(2), D = q.dim(3)
        guard B == 1, S >= 1, S <= maxRows, D == 512, H % 16 == 0, q.dtype == .bfloat16,
            localKV.dtype == .bfloat16, localKV.dim(0) == 1, localKV.dim(3) == D,
            topK.map({ $0.dim(0) == 1 && $0.dim(1) == S }) ?? true,
            let slab = pooled.nibbleSlab, slab.dtype == .bfloat16, D % slab.groupSize == 0
        else { return false }
        let keys = localKV.dim(2) + (topK?.dim(2) ?? pooled.rowCount)
        return keys > 0 && keys <= 16 * maxChunks
    }

    static func launch(
        queries q: MLXArray, localKV: MLXArray, pooled: DeepseekV4PoolView, topK: MLXArray?,
        offset: Int, window: Int, ratio: Int, scale: Float, sinks: MLXArray?
    ) -> MLXArray? {
        guard let slab = pooled.nibbleSlab else { return nil }
        let H = q.dim(1), S = q.dim(2), D = q.dim(3)
        let Lw = localKV.dim(2), K = topK?.dim(2) ?? pooled.rowCount
        let widx: MLXArray
        let raw: MLXArray
        if Lw > 0 {
            raw = localKV.reshaped(Lw, D)
            let visible = DeepseekV4Math.buildWindowMask(
                batch: 1, queryLen: S, offset: offset, window: window, windowLen: Lw
            ).reshaped(S, Lw)
            widx = MLX.where(
                visible, MLXArray(Int32(0) ..< Int32(Lw)).reshaped(1, Lw), MLXArray(Int32(-1)))
        } else {  // no window rows yet: the kernel never reads these (dims[0] == 0)
            raw = MLXArray.zeros([1, D], dtype: .bfloat16)
            widx = MLXArray.zeros([1, 1], dtype: .int32)
        }
        let cidx: MLXArray
        if K > 0 {
            let rows = topK?.reshaped(S, K).asType(.int32)
                ?? broadcast(MLXArray(Int32(0) ..< Int32(K)).reshaped(1, K), to: [S, K])
            let position = MLXArray(Int32(offset) ..< Int32(offset + S)).reshaped(S, 1)
            let visible = MLX.logicalAnd(
                rows .>= MLXArray(Int32(0)),
                ((rows + MLXArray(Int32(1))) * MLXArray(Int32(ratio))) .<= (position + MLXArray(Int32(1))))
            cidx = MLX.where(visible, rows, MLXArray(Int32(-1)))
        } else {
            cidx = MLXArray.zeros([1, 1], dtype: .int32)
        }
        let dims = MLXArray([Int32(Lw), Int32(K), Int32(S)])
        let sink = sinks?.asType(.float32).reshaped(H)
            ?? MLXArray.full([H], values: MLXArray(Float(-1.0e30)))
        let template: [(String, any KernelTemplateArg)] = [
            ("T", DType.bfloat16), ("HD", D), ("NH", H), ("GS", slab.groupSize), ("CHMAX", maxChunks),
        ]
        let chunks = (Lw + K + 15) / 16
        let parts = splitKernel(
            [q.reshaped(H, S, D), raw, widx, cidx,
             slab.codes.reshaped(-1), slab.scales.reshaped(-1), slab.biases.reshaped(-1), dims,
             MLXArray([scale])],
            template: template,
            grid: (512 * (H / 16), chunks, S),
            threadGroup: (512, 1, 1),
            outputShapes: [[S * H * maxChunks, D], [S * H * maxChunks, 2]],
            outputDTypes: [.float32, .float32])
        let out = mergeKernel(
            [parts[0], parts[1], sink, dims],
            template: template,
            grid: (D, H, S),
            threadGroup: (D, 1, 1),
            outputShapes: [[H * S * D]],
            outputDTypes: [.float32])[0]
        return out.reshaped(1, H, S, D).asType(q.dtype)
    }

    /// Both outputs are bf16, rounded from fp32 sums taken in different orders; an element's fp32 error scales with
    /// the terms it sums (the output's magnitude), not with the element itself, and the final bf16 rounding of the
    /// largest element can then flip by an ulp. Agreement: finite, and max |got - ref| <= max(2e-3 * max |ref|,
    /// 2 bf16 ulps of max |ref|) (Python's check is 2e-3 * max |ref| on fp32 outputs). A wrong mask, key or scale
    /// moves outputs far beyond this.
    static func withinRounding(_ got: MLXArray, _ ref: MLXArray) -> Bool {
        let g = got.asType(.float32), r = ref.asType(.float32)
        let diff = abs(g - r).max().item(Float.self)
        let mag = abs(r).max().item(Float.self)
        guard diff.isFinite, mag.isFinite, mag > 0 else { return false }
        let ulp = Float(pow(2.0, floor(log2(Double(mag))) - 7))
        return diff <= max(2e-3 * mag, 2 * ulp)
    }

    /// Once per process: the kernels agree with the stock path (window rows with a masked slot, pool picks with an
    /// invisible one, with and without a sink) to within the reordered sums' rounding.
    static let selfCheckPassed: Bool = {
        guard Device.defaultDevice().deviceType == .gpu else { return false }
        let H = 64, D = 512, S = 2, Lw = 40, T = 96, K = 24
        let storage = DeepseekV4PoolStorage(
            quantizationEnabled: true, hotByteLimit: 0, bits: 4, contiguous: true)
        storage.append((MLXRandom.normal([1, T, D], key: MLXRandom.key(3)) * 0.3).asType(.bfloat16))
        let pooled = DeepseekV4PoolView(storage: storage)
        let q = (MLXRandom.normal([1, H, S, D], key: MLXRandom.key(1)) * 0.3).asType(.bfloat16)
        let local = (MLXRandom.normal([1, 1, Lw, D], key: MLXRandom.key(2)) * 0.3).asType(.bfloat16)
        let topK = MLXRandom.randInt(Int32(0) ..< Int32(T), [1, S, K], key: MLXRandom.key(4)).asType(.int32)
        let offset = 200, window = 128, ratio = 4
        // Selected picks with and without a sink, and every pool row (no indexer selection yet).
        for (sinks, picks) in [(MLXRandom.normal([H], key: MLXRandom.key(5)), topK), (nil, topK),
                               (MLXRandom.normal([H], key: MLXRandom.key(6)), nil)] as [(MLXArray?, MLXArray?)] {
            guard admitted(queries: q, localKV: local, pooled: pooled, topK: picks),
                let got = launch(
                    queries: q, localKV: local, pooled: pooled, topK: picks, offset: offset, window: window,
                    ratio: ratio, scale: 1 / Float(D).squareRoot(), sinks: sinks)
            else { return false }
            let ref = DeepseekV4Math.tiledPoolAttention(
                queries: q, localKV: local, pooled: pooled, offset: offset, window: window, ratio: ratio,
                scale: 1 / Float(D).squareRoot(), sinks: sinks, topK: picks)
            guard withinRounding(got, ref) else { return false }
        }
        return true
    }()
}
