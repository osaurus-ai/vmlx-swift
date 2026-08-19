// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Microbench: vendored-MLX quantized matmul cost vs row count M, at the
// exact projection shapes a Qwen3.8-27B verify forward uses.
//
// Exists to answer one question with the runtime that actually ships:
// does the vendored Metal backend pay a per-row penalty at speculative
// verify shapes (M = 1 + block)? pip mlx 0.32.1 measures M=4 ≈ 1.0× and
// M=8 ≈ 1.9× on the same machine; if the vendored 0.31.1 fork is far
// above that, the fix is a backend update/backport, not a new kernel.
//
//   VMLX_QMM_BENCH=1 swift test --filter QuantizedVerifyShapeBenchTests

import Foundation
import MLX
import MLXNN
import XCTest
@testable import MLXLMCommon

final class QuantizedVerifyShapeBenchTests: XCTestCase {

    func testVerifyShapeCostVsRowCount() throws {
        guard ProcessInfo.processInfo.environment["VMLX_QMM_BENCH"] == "1" else {
            throw XCTSkip("Set VMLX_QMM_BENCH=1 to run the microbench")
        }
        let lock = lockSerializedMLXTest()
        defer { lock.unlock() }

        let d = 5120
        let shapes: [(String, Int, Int)] = [
            ("lm_head", d, 248_320),
            ("up/gate", d, 17_408),
            ("down", 17_408, d),
            ("qkv", d, 6_144),
        ]

        func bench(_ body: () -> MLXArray, n: Int = 50) -> Double {
            MLX.eval(body())
            Stream().synchronize()
            let start = Date.timeIntervalSinceReferenceDate
            for _ in 0 ..< n { MLX.eval(body()) }
            Stream().synchronize()
            return (Date.timeIntervalSinceReferenceDate - start) / Double(n) * 1000
        }

        for (bits, groupSize) in [(4, 64), (4, 128), (6, 64)] {
            print("--- vendored affine \(bits)bit gs\(groupSize) ---")
            for (name, k, n) in shapes {
                let w = MLXRandom.normal([n, k]).asType(.bfloat16)
                let (wq, scales, biases) = quantized(w, groupSize: groupSize, bits: bits)
                var base = 0.0
                for m in [1, 4, 8, 9, 10, 11, 12, 13, 16] {
                    let x = MLXRandom.normal([1, m, k]).asType(.bfloat16)
                    let ms = bench {
                        quantizedMatmul(
                            x, wq, scales: scales, biases: biases,
                            transpose: true, groupSize: groupSize, bits: bits)
                    }
                    if m == 1 { base = ms }
                    print(String(
                        format: "  %-8s M=%2d  %7.3f ms  ratio_vs_M1=%5.2f",
                        (name as NSString).utf8String!, m, ms, ms / base))
                }
            }
        }
    }
}
