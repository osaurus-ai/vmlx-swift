// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// DeepSeek-V4.1 routing after the router logits for dSpark verify rows (2-8) in one Metal dispatch.
//
// Replaces the compiled selector's score and top-k glue (scores + bias, sort, slice, take, sum, normalize, scale): one
// threadgroup a row with a thread an expert computes `sqrt(log1p(exp(logit)))` with MLX's Exp, Log1p and Sqrt
// bodies, adds the bias, and picks the top-k in k rounds of a simd-shuffle argmax, lower id on ties. MLX's
// `argPartition` is a stable ascending merge sort on the GPU, so the negated scores come back in this same pick
// order; thread 0 then sums the picked scores in that order (`row_reduce_small` at one row) and normalizes and scales
// them as the selector does.
//
// Each row is compared against the selector run on that row, once per row count per process, on the first eager
// call. A difference or a kernel that fails to run keeps the selector for that shape. `VMLX_DSV41_ROUTE_TOPK=0`
// keeps the selector.

import Foundation
import MLX
import MLXFast
import MLXLMCommon

enum DeepseekV41RouteRows {
    /// `VMLX_DSV41_ROUTE_TOPK=0` (or false/off/no) keeps the compiled selector.
    nonisolated(unsafe) static var enabled = DeepseekV41RowKernels.envOn(["VMLX_DSV41_ROUTE_TOPK"])

    static let maxRows = 8

    struct Key: Hashable {
        let rows: Int, experts: Int, topK: Int, normalize: Bool
    }

    private static let lock = NSLock()
    nonisolated(unsafe) private static var verdicts: [Key: Bool] = [:]
    nonisolated(unsafe) private static var runs = 0

    static func verdict(_ key: Key) -> Bool? {
        lock.lock(); defer { lock.unlock() }
        return verdicts[key]
    }

    static func record(_ key: Key, _ ok: Bool) {
        lock.lock(); defer { lock.unlock() }
        verdicts[key] = ok
    }

    static func resetVerdicts() {
        lock.lock(); defer { lock.unlock() }
        verdicts.removeAll()
    }

    /// Calls served by the kernel (test and diagnostic counter).
    static var runCount: Int {
        lock.lock(); defer { lock.unlock() }
        return runs
    }

    private static func countRun() {
        lock.lock(); defer { lock.unlock() }
        runs += 1
    }

    // METAL-ONLY: case 2. Metal kernel; without a Metal device the caller keeps the compiled selector.
    private static let kernel = MLXFast.metalKernel(
        name: "vmlx_dsv41_route_rows",
        inputNames: ["logits", "bias", "scale"],
        outputNames: ["idx", "w"],
        source: """
            const uint t = thread_position_in_threadgroup.x, lane = thread_index_in_simdgroup;
            const uint sg = simdgroup_index_in_threadgroup, r = threadgroup_position_in_grid.y;
            threadgroup float best_v[32];
            threadgroup int best_i[32];
            threadgroup int sel[16];
            threadgroup float wsel[16];
            constexpr int NSG = (E + 31) / 32;
            float score = 0.0f, key = -INFINITY;
            if (t < uint(E)) {
              const float e = metal::precise::exp(logits[size_t(r) * E + t]);
              const float xp1 = 1.0f + e;
              const float lp = xp1 == INFINITY ? INFINITY
                  : (xp1 == 1.0f ? e : e * (metal::log(xp1) / (xp1 - 1.0f)));
              score = metal::precise::sqrt(lp);
              key = score + bias[t];
            }
            for (int k = 0; k < K; k++) {
              float v = key;
              int id = t < uint(E) ? int(t) : INT_MAX;
              for (ushort o = 16; o > 0; o >>= 1) {
                const float ov = simd_shuffle_down(v, o);
                const int oi = simd_shuffle_down(id, o);
                // Lanes past the simdgroup's end read nothing valid.
                if (lane + o < 32u && (ov > v || (ov == v && oi < id))) {
                  v = ov;
                  id = oi;
                }
              }
              if (lane == 0) {
                best_v[sg] = v;
                best_i[sg] = id;
              }
              threadgroup_barrier(mem_flags::mem_threadgroup);
              if (t == 0) {
                float bv = best_v[0];
                int bi = best_i[0];
                for (int s = 1; s < NSG; s++) {
                  if (best_v[s] > bv || (best_v[s] == bv && best_i[s] < bi)) {
                    bv = best_v[s];
                    bi = best_i[s];
                  }
                }
                sel[k] = bi;
              }
              threadgroup_barrier(mem_flags::mem_threadgroup);
              if (int(t) == sel[k]) {
                wsel[k] = score;
                key = -INFINITY;
              }
              threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (t == 0) {
              float sum = 0.0f;
              if (NORM) {
                for (int k = 0; k < K; k++) sum = wsel[k] + sum;
              }
              for (int k = 0; k < K; k++) {
                idx[size_t(r) * K + k] = sel[k];
                w[size_t(r) * K + k] = (NORM ? wsel[k] / sum : wsel[k]) * scale[0];
              }
            }
            """)

    /// The kernel, unconditionally: logits [1, R, E], bias [E] -> indices [1, R, K] (int32), weights [1, R, K] (fp32).
    static func run(
        _ logits: MLXArray, bias: MLXArray, topK: Int, normalize: Bool, scale: MLXArray
    ) -> (indices: MLXArray, weights: MLXArray) {
        let rows = logits.dim(1), experts = logits.dim(2)
        let threads = (experts + 31) / 32 * 32
        let out = kernel(
            [logits.reshaped(rows, experts), bias.asType(.float32), scale.asType(.float32)],
            template: [("E", experts), ("K", topK), ("NORM", normalize && topK > 1)],
            grid: (threads, rows, 1),
            threadGroup: (threads, 1, 1),
            outputShapes: [[1, rows, topK], [1, rows, topK]],
            outputDTypes: [.int32, .float32])
        return (out[0], out[1])
    }

    /// (indices [1, R, K] uint32, weights [1, R, K] fp32) for 2-8 rows, or nil to keep the selector. `stock` runs
    /// the selector on one row's logits (called at most once per row count, per row).
    static func apply(
        _ logits: MLXArray, bias: MLXArray, topK: Int, normalize: Bool, scale: MLXArray,
        stock: (MLXArray) -> [MLXArray]
    ) -> (indices: MLXArray, weights: MLXArray)? {
        guard enabled, logits.ndim == 3, logits.dim(0) == 1, (2 ... maxRows).contains(logits.dim(1)),
            logits.dtype == .float32, logits.dim(2) >= 32, logits.dim(2) <= 1024,
            topK > 0, topK <= 16, bias.shape == [logits.dim(2)],
            // METAL-ONLY: case 1. Guards the Metal kernel; other devices keep the compiled selector.
            Device.defaultDevice().deviceType == .gpu
        else { return nil }
        let rows = logits.dim(1)
        let key = Key(rows: rows, experts: logits.dim(2), topK: topK, normalize: normalize)
        switch verdict(key) {
        case .some(true):
            countRun()
            let out = run(logits, bias: bias, topK: topK, normalize: normalize, scale: scale)
            return (out.indices.asType(.uint32), out.weights)
        case .some(false):
            return nil
        case .none:
            if CompiledDecodeTrace.isActive { return nil }
            let got = run(logits, bias: bias, topK: topK, normalize: normalize, scale: scale)
            let refs = (0 ..< rows).map { stock(logits[0..., $0 ..< ($0 + 1)]) }
            guard refs.allSatisfy({ $0.count == 2 }) else { return nil }
            let refI = concatenated(refs.map { $0[0] }, axis: 1)
            let refW = concatenated(refs.map { $0[1] }, axis: 1)
            let ok = DeepseekV41RowKernels.check([got.indices, got.weights, refI, refW]) {
                got.indices.shape == refI.shape && arrayEqual(got.indices, refI).item(Bool.self)
                    && got.weights.shape == refW.shape && arrayEqual(got.weights, refW).item(Bool.self)
            }
            record(key, ok)
            if !ok {
                NSLog(
                    "VMLX DSV4.1 route rows=\(rows) differ from the selector run a row at a time; "
                        + "keeping the selector for this shape.")
                return nil
            }
            countRun()
            return (got.indices.asType(.uint32), got.weights)
        }
    }
}
