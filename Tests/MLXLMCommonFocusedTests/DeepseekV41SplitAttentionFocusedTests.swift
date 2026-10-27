// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXFast
import Testing

@testable import MLXLLM

/// DeepSeek-V4.1 split sparse attention (`DeepseekV41SplitAttention`) against the stock local + selected-pool
/// attention (`DeepseekV4Math.tiledPoolAttention(... topK:)`), on the GPU.
@Suite("DeepSeek-V4.1 split sparse attention")
struct DeepseekV41SplitAttentionFocusedTests {
    struct Case: CustomStringConvertible {
        let rows: Int, window: Int, picks: Int, poolRows: Int, offset: Int, ratio: Int, sink: Bool
        var allRows = false  // no indexer selection: every pool row (topK nil)
        var description: String {
            "S\(rows) Lw\(window) K\(allRows ? "all" : "\(picks)") T\(poolRows) off\(offset) r\(ratio) sink\(sink)"
        }
    }

    static let cases: [Case] = [
        Case(rows: 1, window: 128, picks: 512, poolRows: 3000, offset: 4000, ratio: 1, sink: true),
        Case(rows: 1, window: 128, picks: 64, poolRows: 300, offset: 700, ratio: 2, sink: true),
        Case(rows: 6, window: 133, picks: 512, poolRows: 1500, offset: 3000, ratio: 2, sink: true),
        Case(rows: 3, window: 40, picks: 24, poolRows: 96, offset: 200, ratio: 4, sink: false),
        Case(rows: 2, window: 17, picks: 1, poolRows: 5, offset: 16, ratio: 2, sink: true),
        Case(
            rows: 1, window: 128, picks: 0, poolRows: 300, offset: 600, ratio: 2, sink: true,
            allRows: true),
        Case(
            rows: 6, window: 133, picks: 0, poolRows: 512, offset: 520, ratio: 1, sink: true,
            allRows: true),
        Case(
            rows: 3, window: 20, picks: 0, poolRows: 9, offset: 19, ratio: 2, sink: false,
            allRows: true),
    ]

    static func inputs(_ c: Case, seed: UInt64) -> (
        MLXArray, MLXArray, DeepseekV4PoolView, MLXArray?, MLXArray?
    ) {
        let H = 64
        let D = 512
        let storage = DeepseekV4PoolStorage(
            quantizationEnabled: true, hotByteLimit: 0, bits: 4, contiguous: true)
        storage.append(
            (MLXRandom.normal([1, c.poolRows, D], key: MLXRandom.key(seed)) * 0.3).asType(.bfloat16)
        )
        let q = (MLXRandom.normal([1, H, c.rows, D], key: MLXRandom.key(seed + 1)) * 0.3).asType(
            .bfloat16)
        let local = (MLXRandom.normal([1, 1, c.window, D], key: MLXRandom.key(seed + 2)) * 0.3)
            .asType(.bfloat16)
        let topK: MLXArray? =
            c.allRows
            ? nil
            : MLXRandom.randInt(
                Int32(0) ..< Int32(c.poolRows), [1, c.rows, c.picks], key: MLXRandom.key(seed + 3)
            )
            .asType(.int32)
        let sinks = c.sink ? MLXRandom.normal([H], key: MLXRandom.key(seed + 4)) : nil
        return (q, local, DeepseekV4PoolView(storage: storage), topK, sinks)
    }

    @Test(
        "matches the stock path to rounding, rows 1-6, window and pick masking, with and without a sink",
        arguments: cases)
    func matchesStock(_ c: Case) throws {
        try #require(Device.defaultDevice().deviceType == .gpu)
        let (q, local, pooled, topK, sinks) = Self.inputs(c, seed: 11)
        let scale = 1 / Float(512).squareRoot()
        #expect(
            DeepseekV41SplitAttention.admitted(
                queries: q, localKV: local, pooled: pooled, topK: topK))
        let got = try #require(
            DeepseekV41SplitAttention.launch(
                queries: q, localKV: local, pooled: pooled, topK: topK, offset: c.offset,
                window: 128,
                ratio: c.ratio, scale: scale, sinks: sinks))
        let ref = DeepseekV4Math.tiledPoolAttention(
            queries: q, localKV: local, pooled: pooled, offset: c.offset, window: 128,
            ratio: c.ratio,
            scale: scale, sinks: sinks, topK: topK)
        #expect(got.shape == ref.shape && got.dtype == ref.dtype)
        // Every element within 2 bf16 ulps of the stock element (both round fp32 sums taken in different orders).
        let diff = abs(got.asType(.float32) - ref.asType(.float32)).max().item(Float.self)
        #expect(DeepseekV41SplitAttention.withinRounding(got, ref), "max |diff| \(diff)")
        // Not vacuous: the stock path really is a different computation (some elements differ).
        #expect(diff > 0 || c.picks == 1)
    }

    @Test(
        "not admitted off-shape: 9 rows, non-bf16, no contiguous 4-bit slab; production takes one row"
    )
    func admission() {
        let c = Case(rows: 1, window: 16, picks: 8, poolRows: 32, offset: 64, ratio: 1, sink: true)
        let (q, local, pooled, topK, _) = Self.inputs(c, seed: 31)
        let nine = MLX.concatenated(Array(repeating: q, count: 9), axis: 2)
        #expect(
            !DeepseekV41SplitAttention.admitted(
                queries: nine, localKV: local, pooled: pooled,
                topK: MLX.concatenated(Array(repeating: topK!, count: 9), axis: 1)))
        #expect(
            !DeepseekV41SplitAttention.admitted(
                queries: q.asType(.float16), localKV: local, pooled: pooled, topK: topK))
        // Production takes one decode row; verify-shaped calls keep the dense path.
        let two = MLX.concatenated([q, q], axis: 2)
        #expect(
            DeepseekV41SplitAttention.attend(
                queries: two, localKV: local, pooled: pooled,
                topK: MLX.concatenated([topK!, topK!], axis: 1),
                offset: 64, window: 128, ratio: 1, scale: 0.04, sinks: nil) == nil)
        let dense = DeepseekV4PoolView(dense: MLXRandom.normal([1, 32, 512]).asType(.bfloat16))
        #expect(
            !DeepseekV41SplitAttention.admitted(
                queries: q, localKV: local, pooled: dense, topK: topK))
    }

    @Test("the once-per-process self-check admits the kernels on this GPU")
    func selfCheck() throws {
        try #require(Device.defaultDevice().deviceType == .gpu)
        #expect(DeepseekV41SplitAttention.selfCheckPassed)
    }
}
