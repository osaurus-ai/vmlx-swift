// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

#if os(Linux)
    import CmlxCPUShim
    import Foundation
    import MLX
    import Testing

    @Suite(.serialized) struct QuantizedMatmulTests {
        /// Weights whose affine quantization is exact: each group holds codes 0 and 2^bits - 1 and a
        /// factor of 1 or 2, alternating by group, so that a scale or bias read from the wrong group
        /// shows. Every weight is a multiple of 1/8. MLX may pick a negative scale (8 bits: -1/8 or -1/4,
        /// with codes 255 - c); the quantization is exact either way, which the test checks first. With
        /// integer activations in [-8, 8] and K = 2048, every product and partial sum stays below 2^21
        /// at a resolution of 1/8, so it is exact in float32 in any order, with the bias deferred or not.
        static func exactAffine(n: Int, k: Int, groupSize: Int, bits: Int, seed: UInt64) -> [Float]
        {
            var random = SeededRandom(seed: seed)
            let top = (1 << bits) - 1
            return (0 ..< n * k).map { i in
                let position = i % groupSize
                let factor: Float = (i / groupSize) % 2 == 0 ? 1 : 2
                let code =
                    position == 0 ? 0 : position == 1 ? top : Int(random.next() % UInt64(top + 1))
                return factor * (Float(code) / 8 - 1)
            }
        }

        @Test(arguments: [4, 8])
        func affineOnExactDataIsExact(bits: Int) throws {
            try KernelLock.run {
                let (n, k, groupSize) = (64, 2048, 64)
                let wValues = Self.exactAffine(
                    n: n, k: k, groupSize: groupSize, bits: bits, seed: UInt64(bits))
                let w = MLXArray(wValues, [n, k])
                let (wq, scales, biases) = quantized(w, groupSize: groupSize, bits: bits)
                let restored = dequantized(
                    wq, scales: scales, biases: biases, groupSize: groupSize, bits: bits,
                    dtype: .float32)
                try #require(
                    all(restored .== w).item(Bool.self),
                    "the constructed weights must quantize exactly")
                for rows in [1, 8, 31, 32, 33, 64] {
                    var random = SeededRandom(seed: UInt64(rows))
                    let xValues = (0 ..< rows * k).map { _ in Float(Int(random.next() % 17) - 8) }
                    let x = MLXArray(xValues, [rows, k])
                    var exact = [Double](repeating: 0, count: rows * n)
                    for r in 0 ..< rows {
                        for c in 0 ..< n {
                            var sum = 0.0
                            for i in 0 ..< k {
                                sum += Double(xValues[r * k + i]) * Double(wValues[c * k + i])
                            }
                            exact[r * n + c] = sum
                        }
                    }
                    let check = { (label: String) in
                        let y = quantizedMM(
                            x, wq, scales: scales, biases: biases, transpose: true,
                            groupSize: groupSize,
                            bits: bits)
                        expectIdentical(y, exact, "bits \(bits) rows \(rows) \(label)")
                    }
                    if rows >= 32 {
                        forEachTarget(VMLX_CPU_FAMILY_QMM_AFFINE_DEQUANT) { check("on \($0)") }
                    } else {
                        check("on the static code")
                    }
                }
            }
        }

        @Test(arguments: [4, 8], [32, 64, 128])
        func affineWithinTheFloat32Bound(bits: Int, groupSize: Int) {
            KernelLock.run {
                for k in [512, 4096] {
                    let w = MLXRandom.normal(
                        [256, k], dtype: .float32, key: MLXRandom.key(UInt64(k + bits)))
                    let (wq, scales, biases) = quantized(w, groupSize: groupSize, bits: bits)
                    for rows in [1, 8, 31, 32, 33, 64] {
                        let x = MLXRandom.normal(
                            [rows, k], dtype: .float32, key: MLXRandom.key(UInt64(rows)))
                        let reference = quantizedBound(
                            x, wq, scales: scales, biases: biases, groupSize: groupSize,
                            bits: bits, c: Double(k + 2), dtype: .float32)
                        let run = {
                            let y = quantizedMM(
                                x, wq, scales: scales, biases: biases, transpose: true,
                                groupSize: groupSize, bits: bits)
                            expectWithin(
                                y, reference, "bits \(bits) group \(groupSize) K \(k) rows \(rows)")
                        }
                        if rows >= 32 {
                            forEachTarget(VMLX_CPU_FAMILY_QMM_AFFINE_DEQUANT) { _ in run() }
                        } else {
                            run()
                        }
                    }
                }
            }
        }

        static let floatingPointModes = [
            (QuantizationMode.mxfp4, 32, 4), (.nvfp4, 16, 4), (.mxfp8, 32, 8),
        ]

        /// mxfp4, nvfp4 and mxfp8, whose kernels #3019 dispatches at every M, for each activation dtype
        /// they accept. Their weights carry no bias, so Σ|x·ŵ| is the magnitude of the terms. bf16 and
        /// fp16 run on Highway builds only: the scalar fp_qmm_t accumulates in the dtype.
        static func checkFloatingPointMode(_ mode: (QuantizationMode, Int, Int), dtype: DType) {
            let (qmode, groupSize, bits) = mode
            let k = 1024
            let w = MLXRandom.normal([256, k], dtype: .float32, key: MLXRandom.key(9))
            let (wq, scales, _) = quantized(w, groupSize: groupSize, bits: bits, mode: qmode)
            let wHat = dequantized(
                wq, scales: scales, biases: nil, groupSize: groupSize, bits: bits, mode: qmode,
                dtype: .float32)
            for rows in [1, 8, 32, 64] {
                let x = MLXRandom.normal(
                    [rows, k], dtype: .float32, key: MLXRandom.key(UInt64(rows))
                )
                .asType(dtype)
                let reference = matmulBound(x, wHat.transposed(), c: Double(k + 2), dtype: dtype)
                forEachTarget(VMLX_CPU_FAMILY_QMM_FP) { target in
                    let y = quantizedMM(
                        x, wq, scales: scales, biases: nil, transpose: true,
                        groupSize: groupSize, bits: bits, mode: qmode)
                    #expect(y.dtype == dtype)
                    expectWithin(y, reference, "\(qmode) \(dtype) rows \(rows) on \(target)")
                }
            }
        }

        @Test(arguments: floatingPointModes)
        func floatingPointModesWithinTheFloat32Bound(mode: (QuantizationMode, Int, Int)) {
            KernelLock.run { Self.checkFloatingPointMode(mode, dtype: .float32) }
        }

        @Test(
            .disabled(
                if: !Highway.enabled, "the scalar fp_qmm_t accumulates bf16 and fp16 in the dtype"),
            arguments: floatingPointModes, [DType.bfloat16, .float16])
        func floatingPointModesWithHalfPrecisionActivations(
            mode: (QuantizationMode, Int, Int), dtype: DType
        ) {
            KernelLock.run { Self.checkFloatingPointMode(mode, dtype: dtype) }
        }

        /// bf16 and fp16 activations with scales and biases in the same dtype. #3019 accumulates in
        /// float32 and rounds the output once; the scalar code accumulates in the dtype (the chunk
        /// header), so this runs on Highway builds only.
        @Test(
            .disabled(
                if: !Highway.enabled, "the scalar code accumulates bf16 and fp16 in the dtype"),
            arguments: [DType.bfloat16, .float16],
            [(4, 32), (4, 64), (4, 128), (8, 32), (8, 64), (8, 128)])
        func halfPrecisionActivations(dtype: DType, layout: (bits: Int, groupSize: Int)) {
            KernelLock.run {
                let (bits, groupSize) = layout
                let k = 1024
                let w = MLXRandom.normal([128, k], dtype: .float32, key: MLXRandom.key(21)).asType(
                    dtype)
                let (wq, scales, biases) = quantized(w, groupSize: groupSize, bits: bits)
                for rows in [1, 8, 32] {
                    let x = MLXRandom.normal([rows, k], dtype: .float32, key: MLXRandom.key(22))
                        .asType(dtype)
                    let y = quantizedMM(
                        x, wq, scales: scales, biases: biases, transpose: true,
                        groupSize: groupSize,
                        bits: bits)
                    #expect(y.dtype == dtype)
                    expectWithin(
                        y,
                        quantizedBound(
                            x, wq, scales: scales, biases: biases, groupSize: groupSize,
                            bits: bits, c: Double(k + 2), dtype: dtype),
                        "\(dtype) bits \(bits) group \(groupSize) rows \(rows)")
                }
            }
        }

        /// The int8 path (Highway builds only) runs only when switched on (spec §3.5). One row, affine
        /// 4-bit, group 64, K 256 meets every other condition of it. Switched on, it must leave the
        /// float32 bound, which shows that it ran, and stay within its own: rounding activations to
        /// int8 per group moves each by at most max|x_g|/254, and #3019 multiplies that displacement by
        /// s·q (the bias goes on the unrounded group sum), which |s|·q + |b| bounds. Spec §4 measures
        /// int8 and reports it rather than gating it; this bound only has to hold for a correct kernel.
        @Test(
            .disabled(if: !Highway.enabled, "only builds with Highway kernels have the int8 path"))
        func int8ActivationsOnlyWhenSwitchedOn() {
            KernelLock.run {
                let (n, k, groupSize) = (64, 256, 64)
                let w = MLXRandom.normal([n, k], dtype: .float32, key: MLXRandom.key(31))
                let x = MLXRandom.normal([1, k], dtype: .float32, key: MLXRandom.key(32))
                let (wq, scales, biases) = quantized(w, groupSize: groupSize, bits: 4)
                let multiply = {
                    quantizedMM(
                        x, wq, scales: scales, biases: biases, transpose: true,
                        groupSize: groupSize, bits: 4)
                }
                let fp32Bound = quantizedBound(
                    x, wq, scales: scales, biases: biases, groupSize: groupSize, bits: 4,
                    c: Double(k + 2), dtype: .float32)

                vmlx_cpu_reset_counters()
                #expect(!vmlx_cpu_quantized_int8())
                expectWithin(multiply(), fp32Bound, "int8 off")
                #expect(vmlx_cpu_highway_calls(VMLX_CPU_FAMILY_QMM_AFFINE_INT8) == 0)

                let xs = doubles(x)
                let wHat = doubles(
                    dequantized(
                        wq, scales: scales, biases: biases, groupSize: groupSize, bits: 4,
                        dtype: .float32))
                let wAbs = doubles(
                    dequantized(
                        wq, scales: abs(scales), biases: biases.map { abs($0) },
                        groupSize: groupSize, bits: 4,
                        dtype: .float32))
                let expected: [(value: Double, bound: Double)] = (0 ..< n).map { c in
                    var value = 0.0
                    var rounding = 0.0
                    var magnitude = 0.0
                    for g in stride(from: 0, to: k, by: groupSize) {
                        let largest = (g ..< g + groupSize).map { abs(xs[$0]) }.max() ?? 0
                        var weights = 0.0
                        for i in g ..< g + groupSize {
                            value += xs[i] * wHat[c * k + i]
                            weights += wAbs[c * k + i]
                            magnitude += abs(xs[i]) * wAbs[c * k + i]
                        }
                        rounding += largest / 254 * weights
                    }
                    return (value, rounding + Double(k + 2) * eps32 * magnitude)
                }
                vmlx_cpu_set_quantized_int8(true)
                forEachTarget(VMLX_CPU_FAMILY_QMM_AFFINE_INT8) { target in
                    let y = multiply()
                    #expect(
                        outsideBound(y, fp32Bound) != nil,
                        "int8 on \(target): within the float32 bound, so the int8 path did not run")
                    expectWithin(y, expected, "int8 on \(target)")
                }
            }
        }

        /// A transposed view as activations: non-contiguous input.
        @Test func nonContiguousActivations() {
            KernelLock.run {
                let k = 512
                let w = MLXRandom.normal([128, k], dtype: .float32, key: MLXRandom.key(41))
                let (wq, scales, biases) = quantized(w, groupSize: 64, bits: 8)
                let xT = MLXRandom.normal([k, 40], dtype: .float32, key: MLXRandom.key(42))
                let x = xT.transposed()
                let y = quantizedMM(
                    x, wq, scales: scales, biases: biases, transpose: true, groupSize: 64, bits: 8)
                expectWithin(
                    y,
                    quantizedBound(
                        x, wq, scales: scales, biases: biases, groupSize: 64, bits: 8,
                        c: Double(k + 2), dtype: .float32),
                    "non-contiguous activations")
            }
        }

        /// x·w and |x|·|w| in Double, for row-major x of shape [rows, k] and w of shape [k, n], or
        /// [n, k] when `transposed`.
        static func hostProduct(
            _ x: [Double], _ w: [Double], rows: Int, k: Int, n: Int, transposed: Bool
        ) -> (value: [Double], magnitude: [Double]) {
            var value = [Double](repeating: 0, count: rows * n)
            var magnitude = value
            for r in 0 ..< rows {
                for c in 0 ..< n {
                    var (sum, size) = (0.0, 0.0)
                    for i in 0 ..< k {
                        let product = x[r * k + i] * (transposed ? w[c * k + i] : w[i * n + c])
                        sum += product
                        size += abs(product)
                    }
                    value[r * n + c] = sum
                    magnitude[r * n + c] = size
                }
            }
            return (value, magnitude)
        }

        /// Untransposed weights, x·w with w of shape [K, N], quantized along N. #3019 keeps MLX's
        /// scalar _qmm for them, which adds x_k·(s·q + b) along each row of w. On exact data
        /// (`exactAffine`, its groups along N), with K = 256, every product and partial sum is
        /// exact in float32, so the result must match float64 bit for bit.
        @Test(arguments: [2, 4, 8])
        func untransposedAffineOnExactDataIsExact(bits: Int) throws {
            try KernelLock.run {
                let (k, n, groupSize) = (256, 512, 64)
                let wValues = Self.exactAffine(
                    n: k, k: n, groupSize: groupSize, bits: bits, seed: UInt64(bits + 60))
                let (wq, scales, biases) = quantized(
                    MLXArray(wValues, [k, n]), groupSize: groupSize, bits: bits)
                let w = hostDequantized(
                    wq, scales: scales, biases: biases, groupSize: groupSize, bits: bits)
                try #require(
                    w == wValues.map(Double.init), "the constructed weights must quantize exactly")
                for rows in [1, 5, 33] {
                    var random = SeededRandom(seed: UInt64(rows + 60))
                    let xValues = (0 ..< rows * k).map { _ in Float(Int(random.next() % 17) - 8) }
                    let exact = Self.hostProduct(
                        xValues.map(Double.init), w, rows: rows, k: k, n: n, transposed: false
                    ).value
                    vmlx_cpu_reset_counters()
                    let y = quantizedMM(
                        MLXArray(xValues, [rows, k]), wq, scales: scales, biases: biases,
                        transpose: false, groupSize: groupSize, bits: bits)
                    expectIdentical(y, exact, "bits \(bits) rows \(rows)")
                    #expect(
                        vmlx_cpu_highway_calls(VMLX_CPU_FAMILY_QMM_AFFINE_DEQUANT) == 0
                            && vmlx_cpu_highway_calls(VMLX_CPU_FAMILY_QMM_AFFINE_INT8) == 0,
                        "a dispatched affine kernel ran for untransposed weights")
                }
            }
        }

        /// mxfp4, nvfp4 and mxfp8 weights, untransposed: #3019 keeps MLX's scalar fp_qmm for them,
        /// which adds (x_k·scale)·ŵ in float32 in K steps. That errs by at most about
        /// (K + 2)·u·Σ|x·ŵ| (u = ε/2), which (K + 2)·ε·Σ|x·ŵ| covers twice. float32 only: fp_qmm
        /// accumulates bf16 and fp16 in the dtype, on every build. The weights are dequantized on
        /// the host.
        @Test(arguments: floatingPointModes)
        func untransposedFloatingPointModesWithinTheFloat32Bound(
            mode: (QuantizationMode, Int, Int)
        ) {
            KernelLock.run {
                let (qmode, groupSize, bits) = mode
                let (k, n) = (256, 512)
                let w = MLXRandom.normal([k, n], dtype: .float32, key: MLXRandom.key(61))
                let (wq, scales, _) = quantized(w, groupSize: groupSize, bits: bits, mode: qmode)
                let wHat = hostDequantized(
                    wq, scales: scales, biases: nil, groupSize: groupSize, bits: bits, mode: qmode)
                for rows in [1, 5, 33] {
                    let x = MLXRandom.normal(
                        [rows, k], dtype: .float32, key: MLXRandom.key(UInt64(rows + 61)))
                    let reference = sumBound(
                        Self.hostProduct(
                            doubles(x), wHat, rows: rows, k: k, n: n, transposed: false),
                        c: Double(k + 2), dtype: .float32)
                    vmlx_cpu_reset_counters()
                    let y = quantizedMM(
                        x, wq, scales: scales, biases: nil, transpose: false,
                        groupSize: groupSize, bits: bits, mode: qmode)
                    expectWithin(y, reference, "\(qmode) rows \(rows)")
                    if Highway.enabled {
                        // The family's undispatched fallback, and counted as such.
                        #expect(vmlx_cpu_fallback_calls(VMLX_CPU_FAMILY_QMM_FP) > 0)
                        #expect(vmlx_cpu_highway_calls(VMLX_CPU_FAMILY_QMM_FP) == 0)
                    }
                }
            }
        }

        /// 2-bit weights from 32 rows on. #3019 dequantizes them to float32 with its scalar
        /// dequantizer, since the dispatched one serves 4 and 8 bits, and multiplies with SGEMM,
        /// which it splits by rows across the pool; with N = 128 it splits the dequantization too.
        /// On exact data (`exactAffine`), with K = 1024, the result must match float64 bit for bit.
        @Test(arguments: [32, 64, 128])
        func twoBitAffineFromThirtyTwoRowsIsExact(groupSize: Int) throws {
            try KernelLock.run {
                let (n, k, bits) = (128, 1024, 2)
                let wValues = Self.exactAffine(
                    n: n, k: k, groupSize: groupSize, bits: bits, seed: UInt64(groupSize + 70))
                let (wq, scales, biases) = quantized(
                    MLXArray(wValues, [n, k]), groupSize: groupSize, bits: bits)
                let w = hostDequantized(
                    wq, scales: scales, biases: biases, groupSize: groupSize, bits: bits)
                try #require(
                    w == wValues.map(Double.init), "the constructed weights must quantize exactly")
                for rows in [32, 33, 64] {
                    var random = SeededRandom(seed: UInt64(rows + 70))
                    let xValues = (0 ..< rows * k).map { _ in Float(Int(random.next() % 17) - 8) }
                    let exact = Self.hostProduct(
                        xValues.map(Double.init), w, rows: rows, k: k, n: n, transposed: true
                    ).value
                    vmlx_cpu_reset_counters()
                    let y = quantizedMM(
                        MLXArray(xValues, [rows, k]), wq, scales: scales, biases: biases,
                        transpose: true, groupSize: groupSize, bits: bits)
                    expectIdentical(y, exact, "group \(groupSize) rows \(rows)")
                    #expect(
                        vmlx_cpu_highway_calls(VMLX_CPU_FAMILY_QMM_AFFINE_DEQUANT) == 0,
                        "the dispatched dequantizer ran for 2 bits")
                }
            }
        }
    }
#endif
