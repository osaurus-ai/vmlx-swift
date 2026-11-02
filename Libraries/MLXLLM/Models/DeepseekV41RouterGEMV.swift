// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// DeepSeek-V4.1 router logits at decode: `x.astype(f32) @ weight.astype(f32).T` for one token, read
// straight from the bf16 router rows.
//
// The stock gate casts the [384, 5120] bf16 router weight to fp32 on every call (an 7.9 MB write and
// read per layer on top of the 3.9 MB bf16 read, ~0.47 GB per token over 40 layers) and then runs
// MLX's fp32 GEMV: three dispatches per layer. This kernel folds both casts into the loads and keeps
// MLX's own GEMV order for this shape (`gemv_float32_bm4_bn1_sm1_sn32_tm4_tn4`, picked by
// `gemv_axbpy` for out 384 < 4096 and K < 16 * out): each simdgroup owns 4 output rows, lane l adds
// k = 128 i + 4 l + {0..3} in that order into one fp32 accumulator per row, then a shuffle-down ladder
// 16, 8, 4, 2, 1. Each logit is therefore bit-identical to the stock fp32 matmul (checked by
// `DeepseekV41DecodeFusionFocusedTests`). Pattern: vMLX `vmlx_engine/metal/glm5_router_matvec.py`
// (GLM 288 x 4096).
//
// dSpark verify rows (2-8): `rowLogits` runs the same walk once per row, so each row is exactly its own
// 1-row logits. Stock MLX runs those rows as a batched matmul with another reduction order, so they are
// not bit-equal to that (same logits to f32 rounding). Each row count is admitted once per process after a
// live comparison against the stock matmul run one row at a time.

import Foundation
import MLX
import MLXFast
import MLXLMCommon

enum DeepseekV41RouterGEMV {
    /// `VMLX_DSV41_ROUTER_KERNEL=0` (or false/off/no) keeps the stock fp32 cast + matmul. Read once;
    /// tests flip it to time and compare both paths in one process.
    nonisolated(unsafe) static var enabled: Bool = {
        let env = ProcessInfo.processInfo.environment
        // VMLX_DSV41_ROUTER_GEMV is the Python runtime's name for the same switch.
        let raw = env["VMLX_DSV41_ROUTER_KERNEL"] ?? env["VMLX_DSV41_ROUTER_GEMV"] ?? "1"
        return !["0", "false", "off", "no"].contains(
            raw.trimmingCharacters(in: .whitespaces).lowercased())
    }()

    // METAL-ONLY: case 2. Metal kernel; without a Metal device the caller keeps the stock f32 router matmul.
    private static let kernel = MLXFast.metalKernel(
        name: "vmlx_dsv41_router_bf16_gemv_f32",
        inputNames: ["x", "weight"],
        outputNames: ["logits"],
        source: """
            const uint lane = thread_index_in_simdgroup;
            const uint row = threadgroup_position_in_grid.x * 16u + simdgroup_index_in_threadgroup * 4u;
            float accum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (uint base = 0u; base < uint(K); base += 128u) {
                const uint k = base + lane * 4u;
                float v[4];
                #pragma clang loop unroll(full)
                for (uint j = 0u; j < 4u; ++j) v[j] = float(x[k + j]);
                #pragma clang loop unroll(full)
                for (uint r = 0u; r < 4u; ++r) {
                    #pragma clang loop unroll(full)
                    for (uint j = 0u; j < 4u; ++j)
                        accum[r] += float(weight[size_t(row + r) * uint(K) + k + j]) * v[j];
                }
            }
            #pragma clang loop unroll(full)
            for (uint r = 0u; r < 4u; ++r) {
                #pragma clang loop unroll(full)
                for (ushort offset = 16u; offset >= 1u; offset >>= 1u)
                    accum[r] += simd_shuffle_down(accum[r], offset);
                if (lane == 0u) logits[row + r] = accum[r];
            }
            """)

    /// MLX's GEMV picks this tiling only for out < 4096 rows and K < 16 * out; K a multiple of the
    /// 128-wide block (no leftover pass) and out a multiple of 16 (no clamped last tile).
    static func admits(rows: Int, experts: Int, hidden: Int) -> Bool {
        rows == 1 && experts % 16 == 0 && experts < 4096 && hidden % 128 == 0
            && hidden < 16 * experts
    }

    // MARK: - Verify rows

    /// On while `enabled` is (`VMLX_DSV41_ROUTER_KERNEL=0` turns both off). Tests clear this alone to keep verify
    /// rows on the stock matmul.
    nonisolated(unsafe) static var rowsEnabled = true

    static let maxRows = 8

    struct Key: Hashable {
        let rows: Int, experts: Int, hidden: Int, dtype: DType
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

    /// Calls served by the rows kernel (test and diagnostic counter).
    static var rowRuns: Int {
        lock.lock(); defer { lock.unlock() }
        return runs
    }

    private static func countRun() {
        lock.lock(); defer { lock.unlock() }
        runs += 1
    }

    // METAL-ONLY: case 2. Metal kernel; without a Metal device the caller keeps the stock f32 router matmul.
    private static let rowsKernel = MLXFast.metalKernel(
        name: "vmlx_dsv41_router_bf16_gemv_f32_rows",
        inputNames: ["x", "weight"],
        outputNames: ["logits"],
        source: """
            const uint lane = thread_index_in_simdgroup;
            const uint row = threadgroup_position_in_grid.x * 16u + simdgroup_index_in_threadgroup * 4u;
            const uint r_in = threadgroup_position_in_grid.y;
            auto xr = x + size_t(r_in) * uint(K);
            float accum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (uint base = 0u; base < uint(K); base += 128u) {
                const uint k = base + lane * 4u;
                float v[4];
                #pragma clang loop unroll(full)
                for (uint j = 0u; j < 4u; ++j) v[j] = float(xr[k + j]);
                #pragma clang loop unroll(full)
                for (uint r = 0u; r < 4u; ++r) {
                    #pragma clang loop unroll(full)
                    for (uint j = 0u; j < 4u; ++j)
                        accum[r] += float(weight[size_t(row + r) * uint(K) + k + j]) * v[j];
                }
            }
            #pragma clang loop unroll(full)
            for (uint r = 0u; r < 4u; ++r) {
                #pragma clang loop unroll(full)
                for (ushort offset = 16u; offset >= 1u; offset >>= 1u)
                    accum[r] += simd_shuffle_down(accum[r], offset);
                if (lane == 0u) logits[size_t(r_in) * uint(E) + row + r] = accum[r];
            }
            """)

    /// The rows kernel, unconditionally: `x` [1, R, K] -> [1, R, E].
    static func runRows(_ x: MLXArray, weight: MLXArray) -> MLXArray {
        let rows = x.dim(1), experts = weight.dim(0), hidden = weight.dim(1)
        return rowsKernel(
            [x.reshaped(rows, hidden), weight],
            template: [("K", hidden), ("E", experts)],
            grid: (experts / 16 * 128, rows, 1),
            threadGroup: (128, 1, 1),
            outputShapes: [[1, rows, experts]],
            outputDTypes: [.float32])[0]
    }

    /// The stock fp32 matmul run one row at a time: what a 1-row call of each row computes.
    static func stockRows(_ x: MLXArray, weight: MLXArray) -> MLXArray {
        let w = weight.asType(.float32).transposed()
        return concatenated(
            (0 ..< x.dim(1)).map { x[0..., $0 ..< ($0 + 1)].asType(.float32).matmul(w) }, axis: 1)
    }

    /// fp32 logits [1, R, E] for 2-8 rows, or nil to keep the stock path. The first call of each row count
    /// compares against `stockRows` and falls back for good on a difference or a kernel that fails to run.
    static func rowLogits(_ x: MLXArray, weight: MLXArray) -> MLXArray? {
        guard enabled, rowsEnabled, weight.ndim == 2, weight.dtype == .bfloat16, x.ndim == 3, x.dim(0) == 1,
            (2 ... maxRows).contains(x.dim(1)),
            x.dtype == .bfloat16 || x.dtype == .float32 || x.dtype == .float16
        else { return nil }
        let experts = weight.dim(0), hidden = weight.dim(1)
        guard x.dim(2) == hidden, admits(rows: 1, experts: experts, hidden: hidden),
            // METAL-ONLY: case 1. Guards the Metal kernel; other devices keep the stock matmul.
            Device.defaultDevice().deviceType == .gpu
        else { return nil }
        let key = Key(rows: x.dim(1), experts: experts, hidden: hidden, dtype: x.dtype)
        switch verdict(key) {
        case .some(true):
            countRun()
            return runRows(x, weight: weight)
        case .some(false):
            return nil
        case .none:
            if CompiledDecodeTrace.isActive { return nil }
            let fused = runRows(x, weight: weight)
            let ref = stockRows(x, weight: weight)
            let ok = DeepseekV41RowKernels.check([fused, ref]) {
                fused.shape == ref.shape && arrayEqual(fused, ref).item(Bool.self)
            }
            record(key, ok)
            if !ok {
                NSLog(
                    "VMLX DSV4.1 router rows=\(x.dim(1)) differ from the stock matmul run a row at a time; "
                        + "keeping the stock matmul for this shape.")
                return nil
            }
            countRun()
            return fused
        }
    }

    /// fp32 logits `[..., experts]` for one token, or nil to keep the stock path.
    static func logits(_ x: MLXArray, weight: MLXArray) -> MLXArray? {
        guard enabled, weight.ndim == 2, weight.dtype == .bfloat16,
            x.dtype == .bfloat16 || x.dtype == .float32 || x.dtype == .float16
        else { return nil }
        let experts = weight.dim(0)
        let hidden = weight.dim(1)
        guard x.dim(-1) == hidden, admits(rows: x.size / hidden, experts: experts, hidden: hidden),
            // METAL-ONLY: case 1. Guards the Metal kernel; other devices keep the stock matmul.
            Device.defaultDevice().deviceType == .gpu
        else { return nil }
        let out = kernel(
            [x, weight],
            template: [("K", hidden)],
            grid: (experts / 16 * 128, 1, 1),
            threadGroup: (128, 1, 1),
            outputShapes: [Array(x.shape.dropLast()) + [experts]],
            outputDTypes: [.float32])
        return out[0]
    }
}
