// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLMCommon

/// `RowExactAffineQMV` at 6 bits and up to 16 rows: every row equals MLX's own one-row `qmv_fast` call bit for bit
/// (the DSpark / native-MTP verify contract), at the widths a MiniMax-M3.1 verify uses. Opt-in timing
/// (`VMLX_ROW_EXACT_BENCH=1`) on the M3.1 trunk and head shapes: one-row calls vs blocks of 4 vs one launch vs MLX's
/// stock multi-row matmul.
@Suite("Row-exact affine qmv: 6 bits, up to 16 rows", .serialized)
struct RowExactAffineQMVRowsTests {
    static func quantized(n: Int, k: Int, bits: Int, groupSize: Int, scales: DType, seed: UInt64)
        -> (MLXArray, MLXArray, MLXArray)
    {
        let w = MLXRandom.normal([n, k], key: MLXRandom.key(seed)).asType(.float32) * 0.05
        let (wq, s, b) = MLX.quantized(w, groupSize: groupSize, bits: bits)
        eval(wq, s, b!)
        return (wq, s.asType(scales), b!.asType(scales))
    }

    /// MLX's one-row call per row (what a decode step runs), concatenated.
    static func oneRow(_ x: MLXArray, _ w: MLXArray, _ s: MLXArray, _ b: MLXArray, groupSize: Int, bits: Int)
        -> MLXArray
    {
        concatenated(
            (0 ..< x.dim(0)).map {
                quantizedMatmul(x[$0 ..< ($0 + 1)], w, scales: s, biases: b, transpose: true, groupSize: groupSize, bits: bits)
            }, axis: 0)
    }

    @Test("Every row == MLX's one-row qmv, bits 4 / 6 / 8, 2...16 rows, BF16 (and BF16-x-F16 metadata at 4 bits)")
    func rowsMatchOneRowCalls() throws {
        try Device.withDefaultDevice(.gpu) {
            let saved = RowExactAffineQMV.maxRows4Bit
            RowExactAffineQMV.maxRows4Bit = 16
            defer { RowExactAffineQMV.maxRows4Bit = saved }
            var checked = 0
            for bits in [6, 4, 8] {
                for scales in [DType.bfloat16, .float16] where scales == .bfloat16 || bits == 4 {
                    let (n, k) = (264, bits == 6 ? 1024 : 1536)
                    let (w, s, b) = Self.quantized(n: n, k: k, bits: bits, groupSize: 64, scales: scales, seed: UInt64(bits))
                    for rows in [2, 3, 5, 7, 8, 11, 15, 16] {
                        let x = (MLXRandom.normal([rows, k], key: MLXRandom.key(UInt64(100 + rows))) * 2).asType(.bfloat16)
                        let reference = Self.oneRow(x, w, s, b, groupSize: 64, bits: bits)
                        let got = try #require(
                            RowExactAffineQMV.project(
                                x, weight: w, scales: s, biases: b, groupSize: 64, bits: bits,
                                reference: { Self.oneRow($0, w, s, b, groupSize: 64, bits: bits) }),
                            "bits \(bits) scales \(scales) rows \(rows) not admitted")
                        #expect(
                            got.shape == reference.shape && arrayEqual(got, reference).item(Bool.self),
                            "bits \(bits) scales \(scales) rows \(rows)")
                        checked += 1
                    }
                }
            }
            #expect(checked == 32)
        }
    }

    @Test("Diagnosis (opt-in VMLX_ROW_EXACT_DIAG=1): unchecked kernel vs one-row calls")
    func diagnosis() throws {
        guard ProcessInfo.processInfo.environment["VMLX_ROW_EXACT_DIAG"] == "1" else { return }
        try Device.withDefaultDevice(.gpu) {
            for bits in [6, 4] {
                let (w, s, b) = Self.quantized(n: 264, k: 1024, bits: bits, groupSize: 64, scales: .bfloat16, seed: 6)
                let x = (MLXRandom.normal([2, 1024], key: MLXRandom.key(102)) * 2).asType(.bfloat16)
                let ref = Self.oneRow(x, w, s, b, groupSize: 64, bits: bits)
                let got = RowExactAffineQMV.launchUnchecked(x, weight: w, scales: s, biases: b, groupSize: 64, bits: bits)
                let stock = quantizedMatmul(x, w, scales: s, biases: b, transpose: true, groupSize: 64, bits: bits)
                let d = abs(got.asType(.float32) - ref.asType(.float32))
                print("[diag] bits \(bits) max |kernel - one-row| \(d.max().item(Float.self)) differing \((d .> 0).sum().item(Int.self)) / \(d.size); max |stock - one-row| \(abs(stock.asType(.float32) - ref.asType(.float32)).max().item(Float.self)); ref[0,:4] \(ref[0, ..<4].asType(.float32).asArray(Float.self)) got \(got[0, ..<4].asType(.float32).asArray(Float.self))")
            }
        }
    }

    @Test("Control: MLX's stock multi-row matmul is not row-exact at these shapes (so the check can fail)")
    func stockMultiRowDiffers() throws {
        try Device.withDefaultDevice(.gpu) {
            var differing = 0
            for bits in [4, 6, 8] {
                let (w, s, b) = Self.quantized(n: 264, k: 1024, bits: bits, groupSize: 64, scales: .bfloat16, seed: 7)
                let x = (MLXRandom.normal([8, 1024], key: MLXRandom.key(3)) * 2).asType(.bfloat16)
                let stock = quantizedMatmul(x, w, scales: s, biases: b, transpose: true, groupSize: 64, bits: bits)
                if !arrayEqual(stock, Self.oneRow(x, w, s, b, groupSize: 64, bits: bits)).item(Bool.self) {
                    differing += 1
                }
            }
            #expect(differing > 0, "control: the stock multi-row path matched one-row calls at every width")
        }
    }

    /// Median ms of `body` over `probes` (1 discarded), each a chain of `reps` dependent-free calls in one eval.
    static func time(_ probes: Int = 7, reps: Int = 20, _ body: () -> MLXArray) -> Double {
        var t: [Double] = []
        for p in 0 ... probes {
            let t0 = Date()
            var outs: [MLXArray] = []
            for _ in 0 ..< reps { outs.append(body()) }
            eval(outs)
            if p > 0 { t.append(Date().timeIntervalSince(t0) * 1000 / Double(reps)) }
        }
        t.sort()
        return t[t.count / 2]
    }

    @Test("Benchmark (opt-in VMLX_ROW_EXACT_BENCH=1): MiniMax-M3.1 shapes")
    func benchmark() throws {
        guard ProcessInfo.processInfo.environment["VMLX_ROW_EXACT_BENCH"] == "1" else { return }
        try Device.withDefaultDevice(.gpu) {
            let saved = (RowExactAffineQMV.maxRows4Bit, RowExactAffineQMV.launchRowsWide)
            defer { (RowExactAffineQMV.maxRows4Bit, RowExactAffineQMV.launchRowsWide) = saved }
            // (name, N, K, bits): q_proj, k / v, o_proj, shared gate / up, shared down, dense gate / up, lm_head.
            let shapes: [(String, Int, Int, Int)] = [
                ("q_proj", 8192, 6144, 4), ("kv_proj", 512, 6144, 4), ("o_proj", 6144, 8192, 4),
                ("shared_gate_up", 3072, 6144, 4), ("shared_down", 6144, 3072, 4), ("lm_head_q6", 200064, 6144, 6),
                ("dense_q8_probe", 12288, 6144, 8),
            ]
            var lines: [String] = []
            for (name, n, k, bits) in shapes {
                let (w, s, b) = Self.quantized(n: n, k: k, bits: bits, groupSize: 64, scales: .bfloat16, seed: 11)
                for rows in [1, 2, 4, 8, 12, 15] {
                    let x = (MLXRandom.normal([rows, k], key: MLXRandom.key(5)) * 2).asType(.bfloat16)
                    eval(x)
                    let one = Self.time { Self.oneRow(x, w, s, b, groupSize: 64, bits: bits) }
                    let stock = Self.time {
                        quantizedMatmul(x, w, scales: s, biases: b, transpose: true, groupSize: 64, bits: bits)
                    }
                    var cells = "one-row \(String(format: "%.3f", one)) stock \(String(format: "%.3f", stock))"
                    if rows > 1 {
                        for limit in bits == 4 ? [4, 16] : [4, 8, 16] {
                            RowExactAffineQMV.maxRows4Bit = limit
                            RowExactAffineQMV.launchRowsWide = limit
                            let width = RowExactAffineQMV.launchRows(bits: bits)
                            let blocks = stride(from: 0, to: rows, by: width).map { ($0, min(width, rows - $0)) }
                            let f: () -> MLXArray = {
                                concatenated(blocks.map { start, size -> MLXArray in
                                    let xb = x[start ..< (start + size)]
                                    if size == 1 { return Self.oneRow(xb, w, s, b, groupSize: 64, bits: bits) }
                                    return RowExactAffineQMV.project(
                                        xb, weight: w, scales: s, biases: b, groupSize: 64, bits: bits,
                                        reference: { Self.oneRow($0, w, s, b, groupSize: 64, bits: bits) })!
                                }, axis: 0)
                            }
                            _ = f()
                            cells += " exact(width \(width)) \(String(format: "%.3f", Self.time(7, reps: 20, f)))"
                        }
                    }
                    let line = "\(name) N=\(n) K=\(k) q\(bits) rows=\(rows): \(cells)"
                    lines.append(line)
                    print("[row-exact bench] \(line)")
                }
            }
            #expect(!lines.isEmpty)
        }
    }
}
