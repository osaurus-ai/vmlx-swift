// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Measures how quantized matmul cost scales with the number of input rows M
// on bundle-shaped 4-bit gs64 affine weights. Speculative decoding lives or
// dies on this curve: a verify forward is the plain decode forward with
// M = depth+1 rows, so if M=4 costs ~1x a single row the verify is nearly
// free, and if it costs ~3x the entire speculative gain is consumed.
//
//   VMLX_QMM_BENCH=1 swift test --filter QuantizedSmallMScalingBenchTests
//
// Opt-in via env because it is a timing benchmark, not an assertion suite.

import Foundation
import MLX
import Testing

@testable import MLXLMCommon

@Suite("Quantized small-M scaling bench", .serialized)
struct QuantizedSmallMScalingBenchTests {

    @Test("qmm cost vs M on 4-bit gs64 shapes")
    func smallMScaling() throws {
        guard ProcessInfo.processInfo.environment["VMLX_QMM_BENCH"] == "1" else {
            return
        }
        try FocusedMLXTestSupport.withLock {
            // (name, K, N) — Qwen3.8-27B shapes: hidden 5120, a large FFN
            // projection, and the 248320-row LM head.
            let shapes: [(String, Int, Int)] = [
                ("proj 5120x5120", 5120, 5120),
                ("ffn 5120x17408", 5120, 17408),
                ("lm_head 5120x248320", 5120, 248320),
            ]
            let ms = [1, 2, 4, 8, 9, 16, 32, 64]
            for (name, k, n) in shapes {
                let w = MLXRandom.normal([n, k]).asType(.bfloat16)
                let (wq, scales, biases) = MLX.quantized(w, groupSize: 64, bits: 4)
                MLX.eval(wq, scales)

                // Correctness gate: a multi-row dispatch must produce exactly
                // what the single-row kernel produces for each row.
                let xc = MLXRandom.normal([4, k]).asType(.bfloat16)
                MLX.eval(xc)
                let multi = MLX.quantizedMM(
                    xc, wq, scales: scales, biases: biases,
                    transpose: true, groupSize: 64, bits: 4)
                let rows = (0 ..< 4).map { r in
                    MLX.quantizedMM(
                        xc[r ..< (r + 1), 0...], wq, scales: scales,
                        biases: biases, transpose: true, groupSize: 64, bits: 4)
                }
                let stackedRows = concatenated(rows, axis: 0)
                let maxDiff = MLX.abs(
                    multi.asType(.float32) - stackedRows.asType(.float32)
                ).max().item(Float.self)
                print("[qmm-bench] \(name) M=4 vs per-row maxDiff=\(maxDiff)")
                #expect(maxDiff == 0, "\(name): multi-row result diverged")
                var perM: [Int: Double] = [:]
                for m in ms {
                    let x = MLXRandom.normal([m, k]).asType(.bfloat16)
                    MLX.eval(x)
                    for _ in 0..<5 {
                        MLX.eval(MLX.quantizedMM(
                            x, wq, scales: scales, biases: biases,
                            transpose: true, groupSize: 64, bits: 4))
                    }
                    let iters = 30
                    let start = Date.timeIntervalSinceReferenceDate
                    for _ in 0..<iters {
                        MLX.eval(MLX.quantizedMM(
                            x, wq, scales: scales, biases: biases,
                            transpose: true, groupSize: 64, bits: 4))
                    }
                    let msPerCall =
                        (Date.timeIntervalSinceReferenceDate - start) / Double(iters) * 1000
                    perM[m] = msPerCall
                }
                let base = perM[1] ?? 1
                let weightGB = Double(n * k) * 0.5 / 1e9
                for m in ms {
                    let t = perM[m]!
                    let gbps = weightGB / (t / 1000)
                    print(String(
                        format: "[qmm-bench] %@ M=%-2d  %7.3f ms  %5.2fx vs M=1  %6.1f GB/s",
                        name, m, t, t / base, gbps))
                }
            }
        }
    }
}
