// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

#if os(Linux)
    import CmlxCPUShim
    import Foundation
    import MLX
    import Testing

    @Suite(.serialized) struct NormTests {
        static let widths = [1, 7, 64, 128, 1000, 4096]
        static let eps: Float = 1e-5

        /// RMSNorm of one row in float64, with each element's bound. The float32 work (a sum of n
        /// squares, a square root, a reciprocal, two products) errs by at most (n/2 + 5)·u·|y|; the
        /// bound allows 4 times that. For bf16 and fp16 it adds the two roundings MLX's fallback
        /// makes in the dtype: x·r, then its product with the weight.
        static func rmsNorm(_ x: ArraySlice<Double>, _ w: [Double], dtype: DType)
            -> [(value: Double, bound: Double)]
        {
            let n = Double(x.count)
            let r = 1 / (x.reduce(0) { $0 + $1 * $1 } / n + Double(eps)).squareRoot()
            return zip(x, w).map { xi, wi in
                let normalized = xi * r
                let y = normalized * wi
                var bound = (n + 8) * eps32 * abs(y)
                if dtype == .bfloat16 || dtype == .float16 {
                    bound += abs(wi) * ulp(normalized, in: dtype) + ulp(y, in: dtype)
                }
                return (y, bound)
            }
        }

        /// LayerNorm of one row in float64, with each element's bound. The mean's float32 error, at most
        /// (n + 1)·u·max|x|, reaches every element through x - mean, so the bound carries
        /// |w|·(|mean| + max|x|)/σ; the variance and the products add about (n/2 + 6)·u relative. The
        /// bound, (2n + 10)·ε·(|y| + |w|·(|mean| + max|x|)/σ) + ε·|b|, allows twice that. With
        /// `exactMean` the data make every partial sum of the mean exact, the mean term drops out, and
        /// the bound is (n + 12)·ε·(|y| + |w·(x - mean)/σ|) + ε·|b|, 4 times the analysis. For bf16 and
        /// fp16 it adds the fallback's three roundings in the dtype.
        static func layerNorm(
            _ x: ArraySlice<Double>, _ w: [Double], _ b: [Double], dtype: DType, exactMean: Bool
        ) -> [(value: Double, bound: Double)] {
            let n = Double(x.count)
            let mean = x.reduce(0, +) / n
            let sigma = (x.reduce(0) { $0 + ($1 - mean) * ($1 - mean) } / n + Double(eps))
                .squareRoot()
            let largest = x.map { abs($0) }.max() ?? 0
            return zip(zip(x, w), b).map { pair, bi in
                let (xi, wi) = pair
                let normalized = (xi - mean) / sigma
                let y = normalized * wi + bi
                var bound =
                    exactMean
                    ? (n + 12) * eps32 * (abs(y) + abs(wi * normalized))
                    : (2 * n + 10) * eps32 * (abs(y) + abs(wi) * (abs(mean) + largest) / sigma)
                bound += eps32 * abs(bi)
                if dtype == .bfloat16 || dtype == .float16 {
                    bound +=
                        abs(wi) * ulp(normalized, in: dtype) + ulp(normalized * wi, in: dtype)
                        + ulp(y, in: dtype)
                }
                return (y, bound)
            }
        }

        static func rows<R>(_ values: [Double], width: Int, _ body: (ArraySlice<Double>) -> [R])
            -> [R]
        {
            stride(from: 0, to: values.count, by: width).flatMap { body(values[$0 ..< $0 + width]) }
        }

        @Test(arguments: [DType.float32, .bfloat16, .float16])
        func rmsNormWithinTheBound(dtype: DType) {
            KernelLock.run {
                for n in Self.widths {
                    let x = MLXRandom.normal([3, n], key: MLXRandom.key(UInt64(n))).asType(dtype)
                    let w = (MLXRandom.normal([n], key: MLXRandom.key(UInt64(n) + 1)) * 0.5 + 1)
                        .asType(dtype)
                    let ws = doubles(w)
                    let expected = Self.rows(doubles(x), width: n) {
                        Self.rmsNorm($0, ws, dtype: dtype)
                    }
                    forEachTarget(VMLX_CPU_FAMILY_RMS_NORM) { target in
                        expectWithin(
                            MLXFast.rmsNorm(x, weight: w, eps: Self.eps), expected,
                            "rms_norm \(dtype) n \(n) on \(target)")
                    }
                }
            }
        }

        @Test(arguments: [DType.float32, .bfloat16, .float16])
        func layerNormWithinTheBound(dtype: DType) {
            KernelLock.run {
                for n in Self.widths {
                    let x = (MLXRandom.normal([3, n], key: MLXRandom.key(UInt64(n) + 2)) * 2 + 0.5)
                        .asType(dtype)
                    let w = (MLXRandom.normal([n], key: MLXRandom.key(UInt64(n) + 3)) * 0.5 + 1)
                        .asType(dtype)
                    let b = (MLXRandom.normal([n], key: MLXRandom.key(UInt64(n) + 4)) * 0.1).asType(
                        dtype)
                    let (ws, bs) = (doubles(w), doubles(b))
                    let expected = Self.rows(doubles(x), width: n) {
                        Self.layerNorm($0, ws, bs, dtype: dtype, exactMean: false)
                    }
                    forEachTarget(VMLX_CPU_FAMILY_LAYER_NORM) { target in
                        expectWithin(
                            MLXFast.layerNorm(x, weight: w, bias: b, eps: Self.eps), expected,
                            "layer_norm \(dtype) n \(n) on \(target)")
                    }
                }
            }
        }

        /// A large offset over a small spread: multiples of 1/8 in [998, 1002], and n a power of two up
        /// to 1024, so every partial sum is exact in float32 and any summation order gives the exact
        /// mean. A one-pass variance, E[x²] - mean², cancels about 20 bits here and misses this bound
        /// by orders of magnitude.
        @Test func layerNormWithAnOffset() {
            KernelLock.run {
                for n in [64, 128, 1024] {
                    var random = SeededRandom(seed: UInt64(n))
                    let values = (0 ..< 2 * n).map { _ in
                        1000 + Float(Int(random.next() % 33) - 16) / 8
                    }
                    let x = MLXArray(values, [2, n])
                    let w = MLXRandom.normal([n], key: MLXRandom.key(UInt64(n) + 5)) * 0.5 + 1
                    let b = MLXRandom.normal([n], key: MLXRandom.key(UInt64(n) + 6)) * 0.1
                    let (ws, bs) = (doubles(w), doubles(b))
                    let expected = Self.rows(values.map(Double.init), width: n) {
                        Self.layerNorm($0, ws, bs, dtype: .float32, exactMean: true)
                    }
                    forEachTarget(VMLX_CPU_FAMILY_LAYER_NORM) { target in
                        expectWithin(
                            MLXFast.layerNorm(x, weight: w, bias: b, eps: Self.eps), expected,
                            "layer_norm with an offset, n \(n) on \(target)")
                    }
                }
            }
        }

        /// [80, 4096]: past 262144 elements, where #3019 splits the rows across the pool.
        @Test func threadedRows() {
            KernelLock.run {
                let n = 4096
                let x = MLXRandom.normal([80, n], key: MLXRandom.key(31)) * 2 + 0.5
                let w = MLXRandom.normal([n], key: MLXRandom.key(32)) * 0.5 + 1
                let b = MLXRandom.normal([n], key: MLXRandom.key(33)) * 0.1
                let (xs, ws, bs) = (doubles(x), doubles(w), doubles(b))
                let rms = Self.rows(xs, width: n) { Self.rmsNorm($0, ws, dtype: .float32) }
                let layer = Self.rows(xs, width: n) {
                    Self.layerNorm($0, ws, bs, dtype: .float32, exactMean: false)
                }
                forEachTarget(VMLX_CPU_FAMILY_RMS_NORM) { target in
                    expectWithin(
                        MLXFast.rmsNorm(x, weight: w, eps: Self.eps), rms,
                        "threaded rms_norm on \(target)")
                }
                forEachTarget(VMLX_CPU_FAMILY_LAYER_NORM) { target in
                    expectWithin(
                        MLXFast.layerNorm(x, weight: w, bias: b, eps: Self.eps), layer,
                        "threaded layer_norm on \(target)")
                }
            }
        }

        /// Rows that are not contiguous (a transposed view), and float64, which #3019's kernels do not
        /// handle: Task 17 hands it to MLX's fallback, which computes in float32, and counts that.
        @Test func viewsAndFloat64() {
            KernelLock.run {
                let n = 128
                let w = MLXRandom.normal([n], key: MLXRandom.key(21)) * 0.5 + 1
                let b = MLXRandom.normal([n], key: MLXRandom.key(22)) * 0.1
                let view = MLXRandom.normal([n, 5], key: MLXRandom.key(23)).transposed()
                let (ws, bs) = (doubles(w), doubles(b))
                let rms = Self.rows(doubles(view), width: n) {
                    Self.rmsNorm($0, ws, dtype: .float32)
                }
                let layer = Self.rows(doubles(view), width: n) {
                    Self.layerNorm($0, ws, bs, dtype: .float32, exactMean: false)
                }
                forEachTarget(VMLX_CPU_FAMILY_RMS_NORM) { target in
                    expectWithin(
                        MLXFast.rmsNorm(view, weight: w, eps: Self.eps), rms,
                        "rms_norm view on \(target)")
                }
                forEachTarget(VMLX_CPU_FAMILY_LAYER_NORM) { target in
                    expectWithin(
                        MLXFast.layerNorm(view, weight: w, bias: b, eps: Self.eps), layer,
                        "layer_norm view on \(target)")
                }
                vmlx_cpu_reset_counters()
                let (x64, w64, b64) = (
                    view.asType(.float64), w.asType(.float64), b.asType(.float64)
                )
                expectWithin(
                    MLXFast.rmsNorm(x64, weight: w64, eps: Self.eps), rms, "rms_norm float64")
                expectWithin(
                    MLXFast.layerNorm(x64, weight: w64, bias: b64, eps: Self.eps), layer,
                    "layer_norm float64")
                if Highway.enabled {
                    #expect(vmlx_cpu_fallback_calls(VMLX_CPU_FAMILY_RMS_NORM) > 0)
                    #expect(vmlx_cpu_fallback_calls(VMLX_CPU_FAMILY_LAYER_NORM) > 0)
                }
            }
        }
    }
#endif
