// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

#if os(Linux)
    import CmlxCPUShim
    import Foundation
    import MLX
    import Testing

    @Suite(.serialized) struct SoftmaxTests {
        /// softmax(x)_i = exp(x_i - max)/Σ_j exp(x_j - max). Each exponential errs by at most
        /// k_exp·ε relative, plus its argument's rounding (range·ε); the sum by (n - 1)·u; the
        /// normalisation by 2u. So |p̂_i - p_i| <= (2·k_exp + n + 2 + range)·ε·p_i. k_exp is measured in
        /// units with an absolute floor, but MLX's exp scales an exact 2^ipart, so its error
        /// is relative throughout. MLX's exp is 0 below about -87.68, which adds at most 1e-38. For bf16
        /// and fp16 (computed in float32, `precise`), one unit of the dtype more.
        static func softmax(_ row: ArraySlice<Double>, kExp: Double, dtype: DType) -> [(
            value: Double, bound: Double
        )] {
            let top = row.max()!
            let range = top - row.min()!
            let w = row.map { Foundation.exp($0 - top) }
            let z = w.reduce(0, +)
            return w.map { wi in
                let p = wi / z
                var bound = (2 * kExp + Double(row.count) + 2 + range) * eps32 * p + 1e-38
                if dtype == .bfloat16 || dtype == .float16 { bound += ulp(p, in: dtype) }
                return (p, bound)
            }
        }

        @Test(arguments: [1, 7, 128, 1000, 4099])
        func softmaxWithinTheBound(n: Int) throws {
            try KernelLock.run {
                let kExp = try polynomialUnits("exp")
                // Row 2 spans [-200, 0], so some exponentials underflow.
                let values =
                    spread(-8 ... 8, count: 2 * n, seed: UInt64(n))
                    + spread(-200 ... 0, count: n, seed: UInt64(n) + 1)
                let (x, xs) = materialize(values, .float32, shape: [3, n])
                let expected = (0 ..< 3).flatMap {
                    Self.softmax(xs[$0 * n ..< ($0 + 1) * n], kExp: kExp, dtype: .float32)
                }
                expectWithin(MLX.softmax(x, axis: -1), expected, "softmax n \(n)")
                for dtype in [DType.bfloat16, .float16] {
                    let (xh, xhs) = materialize(values, dtype, shape: [3, n])
                    let expectedH = (0 ..< 3).flatMap {
                        Self.softmax(xhs[$0 * n ..< ($0 + 1) * n], kExp: kExp, dtype: dtype)
                    }
                    expectWithin(
                        MLX.softmax(xh, axis: -1, precise: true), expectedH,
                        "\(dtype) precise softmax n \(n)")
                }
            }
        }

        /// logsumexp(x) = max + log Σ exp(x_j - max). The sum's relative error, (k_exp + range)·ε
        /// from the exponentials and (n - 1)·u from the additions, becomes an absolute error through
        /// log; log itself adds one unit of log Σ, and the addition of max one of the result.
        @Test(arguments: [1, 7, 128, 4099])
        func logSumExpWithinTheBound(n: Int) throws {
            try KernelLock.run {
                let kExp = try polynomialUnits("exp")
                let (x, xs) = materialize(
                    spread(-30 ... 30, count: 2 * n, seed: UInt64(n) + 2), .float32, shape: [2, n])
                let expected = (0 ..< 2).map { r -> (value: Double, bound: Double) in
                    let row = xs[r * n ..< (r + 1) * n]
                    let top = row.max()!
                    let logSum = Foundation.log(row.reduce(0) { $0 + Foundation.exp($1 - top) })
                    let value = top + logSum
                    let bound =
                        (kExp + Double(n) + (top - row.min()!)) * eps32 + ulp(logSum, in: .float32)
                        + 2 * ulp(value, in: .float32)
                    return (value, bound)
                }
                expectWithin(MLX.logSumExp(x, axis: -1), expected, "logsumexp n \(n)")
                // bf16 and fp16 accumulate in float32 (logsumexp.cpp's AccT) and round once.
                for dtype in [DType.bfloat16, .float16] {
                    let (xh, xhs) = materialize(xs, dtype, shape: [2, n])
                    let expectedH = (0 ..< 2).map { r -> (value: Double, bound: Double) in
                        let row = xhs[r * n ..< (r + 1) * n]
                        let top = row.max()!
                        let logSum = Foundation.log(row.reduce(0) { $0 + Foundation.exp($1 - top) })
                        let value = top + logSum
                        let bound =
                            (kExp + Double(n) + (top - row.min()!)) * eps32
                            + ulp(logSum, in: .float32)
                            + 2 * ulp(value, in: .float32) + ulp(value, in: dtype)
                        return (value, bound)
                    }
                    expectWithin(
                        MLX.logSumExp(xh, axis: -1), expectedH, "\(dtype) logsumexp n \(n)")
                }
            }
        }

        /// [64, 4099]: past MIN_TOTAL_ELEMENTS, so #3019 spreads the rows over the pool.
        @Test func threadedSoftmax() throws {
            try KernelLock.run {
                let kExp = try polynomialUnits("exp")
                let (rows, n) = (64, 4099)
                let (x, xs) = materialize(
                    spread(-8 ... 8, count: rows * n, seed: 100), .float32, shape: [rows, n])
                let expected = (0 ..< rows).flatMap {
                    Self.softmax(xs[$0 * n ..< ($0 + 1) * n], kExp: kExp, dtype: .float32)
                }
                expectWithin(MLX.softmax(x, axis: -1), expected, "threaded softmax")
            }
        }
    }
#endif
