// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

#if os(Linux)
    import CmlxCPUShim
    import Foundation
    import MLX
    import Testing

    @Suite(.serialized) struct ReductionTests {
        static let shape = [8, 33, 17]

        /// Exact float64 reductions of `data` (row-major, `shape`) over `axes`, row-major.
        static func reduce(
            _ data: [Double], axes: [Int], _ combine: (Double, Double) -> Double, _ start: Double
        ) -> [Double] {
            let kept = (0 ..< 3).filter { !axes.contains($0) }
            let outShape = kept.map { shape[$0] }
            var out = [Double](repeating: start, count: outShape.reduce(1, *))
            for i in 0 ..< shape[0] {
                for j in 0 ..< shape[1] {
                    for k in 0 ..< shape[2] {
                        let index = [i, j, k]
                        let position = kept.reduce(0) { $0 * shape[$1] + index[$1] }
                        out[position] = combine(
                            out[position], data[(i * shape[1] + j) * shape[2] + k])
                    }
                }
            }
            return out
        }

        /// Values in {-1, 0, 1}: every partial sum of up to 136 of them is exact even in bf16, so every
        /// order and every accumulator gives the exact sum. The whole array's sum is checked in float32
        /// and float64 only.
        @Test(arguments: floatTypes)
        func sumsOfExactData(dtype: DType) {
            KernelLock.run {
                var random = SeededRandom(seed: 81)
                let data = (0 ..< 8 * 33 * 17).map { _ in Double(Int(random.next() % 3) - 1) }
                let (x, _) = materialize(data, dtype, shape: Self.shape)
                var cases: [[Int]] = [[2], [1], [0], [0, 2]]
                if dtype == .float32 || dtype == .float64 { cases.append([0, 1, 2]) }
                for axes in cases {
                    expectIdentical(
                        MLX.sum(x, axes: axes), Self.reduce(data, axes: axes, +, 0),
                        "sum \(axes) \(dtype)")
                }
                expectIdentical(
                    MLX.sum(x.transposed(2, 0, 1), axis: 1),
                    Self.reduce(data, axes: [0], +, 0).transposed(rows: 33, columns: 17),
                    "sum of a view \(dtype)")
            }
        }

        /// Values in {1, -1, 2, 0.5}: products of up to 33 of them are exact in every dtype but fp16.
        @Test(arguments: [DType.float32, .float64, .bfloat16])
        func productsOfExactData(dtype: DType) {
            KernelLock.run {
                var random = SeededRandom(seed: 82)
                let data = (0 ..< 8 * 33 * 17).map { _ in [1, -1, 2, 0.5][Int(random.next() % 4)] }
                let (x, _) = materialize(data, dtype, shape: Self.shape)
                for axes in [[2], [1]] {
                    expectIdentical(
                        MLX.product(x, axes: axes), Self.reduce(data, axes: axes, *, 1),
                        "prod \(axes) \(dtype)")
                }
            }
        }

        /// The maximum and minimum are elements: exact on any data.
        @Test(arguments: floatTypes)
        func extremes(dtype: DType) {
            KernelLock.run {
                let (x, xs) = materialize(
                    spread(-100 ... 100, count: 8 * 33 * 17, seed: 83), dtype, shape: Self.shape)
                for axes in [[2], [1], [0], [0, 1, 2]] {
                    expectIdentical(
                        MLX.max(x, axes: axes),
                        Self.reduce(xs, axes: axes, { Swift.max($0, $1) }, -.infinity),
                        "max \(axes) \(dtype)")
                    expectIdentical(
                        MLX.min(x, axes: axes),
                        Self.reduce(xs, axes: axes, { Swift.min($0, $1) }, .infinity),
                        "min \(axes) \(dtype)")
                }
            }
        }

        /// A NaN anywhere makes max and min NaN: in any lane of the vector accumulator, or in the
        /// scalar tail. A pin: reduce.cpp returns NaN before Highway's ReduceMax, which on x86-64
        /// can drop one (NEON's FMAXV keeps it); only the x86-64 leg catches a change there. A
        /// failure names the positions that lose the NaN.
        @Test(arguments: floatTypes)
        func extremesOfNaN(dtype: DType) {
            KernelLock.run {
                var lostByMax: [Int] = []
                var lostByMin: [Int] = []
                for p in 0 ..< 67 {
                    var values = (0 ..< 67).map { Double($0 % 9) * 0.5 - 2 }
                    values[p] = .nan
                    let (x, _) = materialize(values, dtype)
                    if !doubles(MLX.max(x))[0].isNaN { lostByMax.append(p) }
                    if !doubles(MLX.min(x))[0].isNaN { lostByMin.append(p) }
                }
                #expect(lostByMax.isEmpty, "max loses a NaN at \(lostByMax), \(dtype)")
                #expect(lostByMin.isEmpty, "min loses a NaN at \(lostByMin), \(dtype)")
            }
        }

        /// A float32 sum of n terms in any order errs by at most (n - 1)·u·Σ|x|; the bound is
        /// (n + 2)·ε·Σ|x|, and the mean's adds its scaling.
        @Test func float32SumsWithinTheBound() {
            KernelLock.run {
                let n = 4099
                let (x, xs) = materialize(
                    spread(-100 ... 100, count: 3 * n, seed: 84), .float32, shape: [3, n])
                let rows = (0 ..< 3).map { Array(xs[$0 * n ..< ($0 + 1) * n]) }
                let sums = rows.map { row in
                    (
                        value: row.reduce(0, +),
                        bound: Double(n + 2) * eps32 * row.reduce(0) { $0 + abs($1) }
                    )
                }
                expectWithin(MLX.sum(x, axis: 1), sums, "sum of 4099")
                expectWithin(
                    MLX.mean(x, axis: 1),
                    sums.map {
                        (
                            $0.value / Double(n),
                            $0.bound / Double(n) + 2 * eps32 * abs($0.value / Double(n))
                        )
                    },
                    "mean of 4099")
                let columns = (0 ..< n).map { j in (0 ..< 3).map { rows[$0][j] } }
                let strided = columns.map { col in
                    (value: col.reduce(0, +), bound: 5 * eps32 * col.reduce(0) { $0 + abs($1) })
                }
                expectWithin(MLX.sum(x, axis: 0), strided, "strided sum")
            }
        }

        /// The Highway build's contiguous reduction accumulates bf16 and fp16 in float32, and adds
        /// its threads' partial sums in float32 too. The scalar code accumulates in the dtype,
        /// where these sums stop at 256 and 2048: upstream's behaviour.
        @Test(
            .disabled(
                if: !Highway.enabled, "the scalar code accumulates bf16 and fp16 in the dtype"),
            arguments: [DType.bfloat16, .float16])
        func halfPrecisionSumsAccumulateInFloat(dtype: DType) {
            KernelLock.run {
                let ones = MLXArray.ones([4096], dtype: dtype)
                #expect(doubles(MLX.sum(ones)) == [4096], "\(dtype) sum of 4096 ones")
                #expect(doubles(MLX.mean(ones)) == [1], "\(dtype) mean of 4096 ones")
                if dtype == .bfloat16 {
                    // Past MIN_TOTAL_ELEMENTS, on the pool: exactly 3. Thread partials rounded to
                    // bf16 give another sum: 0 with a pool of 8 threads, -8192 with 9 or 12.
                    let x = concatenated([
                        MLXArray.ones([(1 << 20) + 3], dtype: dtype),
                        MLXArray([Float(-1_048_576)]).asType(dtype),
                    ])
                    #expect(doubles(MLX.sum(x)) == [3], "bf16 thread partials")
                }
            }
        }

        @Test func allAndAny() {
            KernelLock.run {
                var random = SeededRandom(seed: 98)
                // Mostly true, so that `all` gives both answers along the short axes.
                let data = (0 ..< 8 * 33 * 17).map { _ in random.next() % 50 != 0 }
                let x = MLXArray(data, Self.shape)
                let numbers = data.map { $0 ? 1.0 : 0.0 }
                for axes in [[2], [1], [0], [0, 2], [0, 1, 2]] {
                    let all = Self.reduce(numbers, axes: axes, *, 1).map { $0 != 0 }
                    let any = Self.reduce(numbers, axes: axes, { Swift.max($0, $1) }, 0).map {
                        $0 != 0
                    }
                    #expect(MLX.all(x, axes: axes).asArray(Bool.self) == all, "all \(axes)")
                    #expect(MLX.any(x, axes: axes).asArray(Bool.self) == any, "any \(axes)")
                }
            }
        }

        /// [64, 4099] of {-1, 0, 1}: 262,336 elements, past MIN_TOTAL_ELEMENTS, so #3019 reduces on the
        /// pool. float32 row sums of at most 4099 in magnitude are exact.
        @Test func threadedRows() {
            KernelLock.run {
                var random = SeededRandom(seed: 99)
                let (rows, width) = (64, 4099)
                let data = (0 ..< rows * width).map { _ in Double(Int(random.next() % 3) - 1) }
                let (x, _) = materialize(data, .float32, shape: [rows, width])
                let row = { (r: Int) in data[r * width ..< (r + 1) * width] }
                expectIdentical(
                    MLX.sum(x, axis: 1), (0 ..< rows).map { row($0).reduce(0, +) },
                    "threaded row sums")
                expectIdentical(
                    MLX.max(x, axis: 1), (0 ..< rows).map { row($0).max() ?? 0 },
                    "threaded row maxima")
            }
        }
    }

    extension Array where Element == Double {
        /// This row-major rows × columns matrix, transposed.
        func transposed(rows: Int, columns: Int) -> [Double] {
            (0 ..< columns).flatMap { c in (0 ..< rows).map { r in self[r * columns + c] } }
        }
    }
#endif
