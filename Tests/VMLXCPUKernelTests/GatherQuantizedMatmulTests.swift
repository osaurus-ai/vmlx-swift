// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

#if os(Linux)
    import CmlxCPUShim
    import Foundation
    import MLX
    import Testing

    /// gatherQuantizedMM, the matmul of a mixture-of-experts layer: route i multiplies the rows of
    /// x at lhs[i] by the quantized weights of expert rhs[i].
    @Suite(.serialized) struct GatherQuantizedMatmulTests {
        typealias Route = (lhs: UInt32, rhs: UInt32)

        /// Routes sorted by expert, as a mixture-of-experts layer sorts a prompt's, over 96 rows of
        /// x and three experts. #3019 multiplies consecutive routes to one expert, over rows that
        /// follow on in x, as one matrix: here runs of 40, 5, 33 and 36 routes (its dequantizing
        /// path starts at 32 rows). After the 33, a run of 18 stays with that expert over rows that
        /// do not follow on, and must start a matrix of its own. Then single routes, each to an
        /// expert other than the one before, until the routes outnumber the pool's threads twice
        /// over.
        static func routes(threads: Int) -> [Route] {
            func run(_ rows: Range<Int>, _ expert: UInt32) -> [Route] {
                rows.map { (UInt32($0), expert) }
            }
            var routes =
                run(0 ..< 40, 1) + run(40 ..< 45, 0) + run(45 ..< 78, 2) + run(0 ..< 18, 2)
                + run(60 ..< 96, 0)
            let count = max(routes.count + 4, 2 * threads + 1)
            while routes.count < count {
                routes.append((UInt32(routes.count * 7 % 96), UInt32((routes.count + 1) % 3)))
            }
            return routes
        }

        static func indices(_ routes: [Route]) -> (lhs: MLXArray, rhs: MLXArray) {
            (MLXArray(routes.map { $0.lhs }), MLXArray(routes.map { $0.rhs }))
        }

        /// x[lhs]·w[rhs]ᵀ per route, and the same with |x| and |w|, in Double: x of shape
        /// [96, m, k], w of shape [experts, n, k], the result of shape [routes, m, n].
        static func reference(
            _ x: [Double], _ w: [Double], routes: [Route], m: Int, n: Int, k: Int
        ) -> (value: [Double], magnitude: [Double]) {
            var value = [Double](repeating: 0, count: routes.count * m * n)
            var magnitude = value
            for (i, route) in routes.enumerated() {
                for row in 0 ..< m {
                    let xRow = (Int(route.lhs) * m + row) * k
                    for column in 0 ..< n {
                        let wRow = (Int(route.rhs) * n + column) * k
                        var (sum, size) = (0.0, 0.0)
                        for j in 0 ..< k {
                            let product = x[xRow + j] * w[wRow + j]
                            sum += product
                            size += abs(product)
                        }
                        value[(i * m + row) * n + column] = sum
                        magnitude[(i * m + row) * n + column] = size
                    }
                }
            }
            return (value, magnitude)
        }

        /// Affine experts on exact data (`QuantizedMatmulTests.exactAffine`): with K = 512 every
        /// product and partial sum is exact in float32, so batching routes into matrices, and
        /// summing in any order, must give the float64 result bit for bit. One row of x per route,
        /// and two, with which the run of 18 reaches the dequantizing path too.
        @Test(arguments: [4, 8])
        func affineRoutesBatchedByExpertAreExact(bits: Int) throws {
            try KernelLock.run {
                let (n, k, groupSize) = (128, 512, 64)
                let wValues = QuantizedMatmulTests.exactAffine(
                    n: 3 * n, k: k, groupSize: groupSize, bits: bits, seed: UInt64(bits + 50))
                let (wq, scales, biases) = quantized(
                    MLXArray(wValues, [3, n, k]), groupSize: groupSize, bits: bits)
                let w = hostDequantized(
                    wq, scales: scales, biases: biases, groupSize: groupSize, bits: bits)
                try #require(
                    w == wValues.map(Double.init), "the constructed weights must quantize exactly")
                let routes = Self.routes(threads: Int(vmlx_cpu_thread_count()))
                let (lhs, rhs) = Self.indices(routes)
                for m in [1, 2] {
                    var random = SeededRandom(seed: UInt64(m + 50))
                    let xValues = (0 ..< 96 * m * k).map { _ in Float(Int(random.next() % 17) - 8) }
                    let exact = Self.reference(
                        xValues.map(Double.init), w, routes: routes, m: m, n: n, k: k
                    ).value
                    let x = MLXArray(xValues, [96, m, k])
                    forEachTarget(VMLX_CPU_FAMILY_QMM_AFFINE_DEQUANT) { target in
                        let y = gatherQuantizedMM(
                            x, wq, scales: scales, biases: biases, lhsIndices: lhs,
                            rhsIndices: rhs, transpose: true, groupSize: groupSize, bits: bits)
                        expectIdentical(y, exact, "bits \(bits), \(m) rows a route, on \(target)")
                    }
                }
            }
        }

        /// mxfp4, nvfp4 and mxfp8 experts, dequantized on the host. With one row of x per route and
        /// more routes than pool threads, #3019 hands whole routes to the threads; with two rows it
        /// batches consecutive routes to one expert, as for affine. The kernels sum K products of x
        /// and weights exact in float32, in groups scaled once. In any order that errs by at most
        /// about K·u·Σ|x·ŵ| (u = ε/2); (K + 2)·ε·Σ|x·ŵ| covers it twice, with room for the scales'
        /// roundings. The weights have no bias, so Σ|x·ŵ| is the terms' magnitude. For bf16 and
        /// fp16, one unit of the dtype more, for the output's rounding (`sumBound`).
        static func checkFloatingPointExperts(_ mode: (QuantizationMode, Int, Int), dtype: DType) {
            let (qmode, groupSize, bits) = mode
            let (n, k) = (64, 512)
            let w = MLXRandom.normal([3, n, k], dtype: .float32, key: MLXRandom.key(51))
            let (wq, scales, _) = quantized(w, groupSize: groupSize, bits: bits, mode: qmode)
            let wHat = hostDequantized(
                wq, scales: scales, biases: nil, groupSize: groupSize, bits: bits, mode: qmode)
            let routes = Self.routes(threads: Int(vmlx_cpu_thread_count()))
            let (lhs, rhs) = Self.indices(routes)
            for m in [1, 2] {
                let x = MLXRandom.normal(
                    [96, m, k], dtype: .float32, key: MLXRandom.key(UInt64(51 + m))
                )
                .asType(dtype)
                let reference = sumBound(
                    Self.reference(doubles(x), wHat, routes: routes, m: m, n: n, k: k),
                    c: Double(k + 2), dtype: dtype)
                forEachTarget(VMLX_CPU_FAMILY_QMM_FP) { target in
                    let y = gatherQuantizedMM(
                        x, wq, scales: scales, biases: nil, lhsIndices: lhs, rhsIndices: rhs,
                        transpose: true, groupSize: groupSize, bits: bits, mode: qmode)
                    #expect(y.dtype == dtype)
                    expectWithin(y, reference, "\(qmode) \(dtype), \(m) rows a route, on \(target)")
                }
            }
        }

        @Test(arguments: QuantizedMatmulTests.floatingPointModes)
        func floatingPointExpertsWithinTheFloat32Bound(mode: (QuantizationMode, Int, Int)) {
            KernelLock.run { Self.checkFloatingPointExperts(mode, dtype: .float32) }
        }

        @Test(
            .disabled(
                if: !Highway.enabled, "the scalar fp_qmm_t accumulates bf16 and fp16 in the dtype"),
            arguments: QuantizedMatmulTests.floatingPointModes, [DType.bfloat16, .float16])
        func floatingPointExpertsWithHalfPrecisionActivations(
            mode: (QuantizationMode, Int, Int), dtype: DType
        ) {
            KernelLock.run { Self.checkFloatingPointExperts(mode, dtype: dtype) }
        }
    }
#endif
