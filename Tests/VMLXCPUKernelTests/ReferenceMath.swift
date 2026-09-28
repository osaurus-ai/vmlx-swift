// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

#if os(Linux)
    import Foundation
    import MLX
    import Testing

    /// float32's machine epsilon, 2^-23.
    let eps32 = Double(Float.ulpOfOne)

    /// Rounds a Float to bfloat16 (to nearest, ties to even), as MLX's conversion does, and back.
    func roundToBF16(_ x: Float) -> Float {
        if x.isNaN { return x }
        let bits = x.bitPattern
        return Float(bitPattern: (bits &+ 0x7FFF &+ ((bits >> 16) & 1)) & 0xFFFF_0000)
    }

    /// Rounds a Float to float16, and back.
    func roundToF16(_ x: Float) -> Float { Float(Float16(x)) }

    /// Rounds a Float to `dtype`'s precision, and back.
    func roundTo(_ dtype: DType, _ x: Float) -> Float {
        switch dtype {
        case .bfloat16: return roundToBF16(x)
        case .float16: return roundToF16(x)
        default: return x
        }
    }

    /// One unit in the last place of `value` in `dtype` (bf16 keeps 8 significand bits, float32 24).
    func ulp(_ value: Double, in dtype: DType) -> Double {
        switch dtype {
        case .bfloat16: return Double(Float(value).ulp) * 65536
        case .float16: return Double(Float16(Float(value)).ulp)
        case .float64: return value.ulp
        default: return Double(Float(value).ulp)
        }
    }

    /// `got`'s error against `truth` in units: float32 ULPs at `truth`, but never less than `floor`
    /// absolute, so that zero crossings (sin near a multiple of pi) are judged absolutely.
    func unitsError(_ got: Float, _ truth: Double, floor: Double = 0x1p-33) -> Double {
        if got.isNaN || truth.isNaN { return got.isNaN && truth.isNaN ? 0 : .infinity }
        if got.isInfinite || truth.isInfinite { return Double(got) == truth ? 0 : .infinity }
        return abs(Double(got) - truth) / max(Double(Float(truth).ulp), floor)
    }

    /// erf^-1 in Double, by bisection on libm's erf over [-6, 6]: 80 halvings reach Double's precision.
    func erfInverseReference(_ y: Double) -> Double {
        var low = -6.0
        var high = 6.0
        for _ in 0 ..< 80 {
            let mid = (low + high) / 2
            if Foundation.erf(mid) < y { low = mid } else { high = mid }
        }
        return (low + high) / 2
    }

    /// Evenly spaced points, as many seeded random ones, and `extras`, in [low, high].
    func grid(_ low: Float, _ high: Float, count: Int, seed: UInt64, extras: [Float] = [])
        -> [Float]
    {
        var random = SeededRandom(seed: seed)
        let half = count / 2
        let step = (high - low) / Float(half)
        return (0 ..< half).map { low + Float($0) * step } + random.floats(half, low, high) + extras
    }

    /// The unit roundoff of `dtype`: half the distance from 1 to the next larger value.
    func unitRoundoff(_ dtype: DType) -> Double {
        switch dtype {
        case .bfloat16: return 0x1p-8
        case .float16: return 0x1p-11
        case .float64: return 0x1p-53
        default: return 0x1p-24
        }
    }

    /// A polynomial function's measured scalar error (`ULPBaseline`) plus the 2 units a Highway build
    /// may add: the input to the bounds of functions built on MLX's polynomials.
    func polynomialUnits(_ name: String, sourceLocation: SourceLocation = #_sourceLocation) throws
        -> Double
    {
        try #require(
            ULPBaseline.units[name], "no baseline for \(name): Task 24 measures it",
            sourceLocation: sourceLocation) + 2
    }

    /// Where `got` leaves its float64 reference: nil when every element is within its bound (equal
    /// values, infinities and NaNs included), else the element furthest outside it. On the host.
    func outsideBound(_ got: MLXArray, _ reference: [(value: Double, bound: Double)]) -> String? {
        let values = doubles(got)
        guard values.count == reference.count else {
            return "\(values.count) elements, where float64 gives \(reference.count)"
        }
        var worst = (index: 0, ratio: 0.0)
        for (i, (value, expected)) in zip(values, reference).enumerated() {
            if value == expected.value || (value.isNaN && expected.value.isNaN) {
                continue  // equal, infinities and NaNs included
            }
            let error = abs(value - expected.value)
            let ratio = error.isNaN || expected.bound.isNaN ? .infinity : error / expected.bound
            if ratio > worst.ratio { worst = (i, ratio) }
        }
        if worst.ratio <= 1 { return nil }
        return
            "element \(worst.index) is \(values[worst.index]), float64 gives \(reference[worst.index].value), \(worst.ratio) times its bound"
    }

    /// Checks every element of `got` against its float64 reference value and its bound, and names
    /// the element furthest outside its bound.
    func expectWithin(
        _ got: MLXArray, _ reference: [(value: Double, bound: Double)],
        _ label: @autoclosure () -> String, sourceLocation: SourceLocation = #_sourceLocation
    ) {
        let outside = outsideBound(got, reference)
        #expect(outside == nil, "\(label()): \(outside ?? "")", sourceLocation: sourceLocation)
    }

    /// `count` Doubles over `domain`, alternately evenly spaced and seeded-random.
    func spread(_ domain: ClosedRange<Double>, count: Int = 4099, seed: UInt64) -> [Double] {
        var random = SeededRandom(seed: seed)
        let (low, width) = (domain.lowerBound, domain.upperBound - domain.lowerBound)
        return (0 ..< count).map { i in
            low + width
                * (i % 2 == 0 ? Double(i) / Double(count) : Double(random.next() >> 11) * 0x1p-53)
        }
    }

    /// `values` as an array of `dtype` (rounded to it), and that array's elements as Doubles, exactly.
    func materialize(_ values: [Double], _ dtype: DType, shape: [Int]? = nil) -> (
        MLXArray, [Double]
    ) {
        let array = MLXArray(values, shape ?? [values.count]).asType(dtype)
        return (array, doubles(array))
    }

    /// `got` equals `expected` element for element, bit for bit: +0 and -0 differ, and any NaN
    /// matches any NaN. (`doubles` converts exactly, keeping the sign of a zero.)
    func expectIdentical(
        _ got: MLXArray, _ expected: [Double], _ label: @autoclosure () -> String,
        sourceLocation: SourceLocation = #_sourceLocation
    ) {
        let values = doubles(got)
        #expect(
            values.count == expected.count, "\(label()): element count",
            sourceLocation: sourceLocation)
        let differing = zip(values, expected).enumerated().filter { _, pair in
            !(pair.0.bitPattern == pair.1.bitPattern || (pair.0.isNaN && pair.1.isNaN))
        }
        #expect(
            differing.isEmpty,
            "\(label()): \(differing.count) of \(values.count) differ; element \(differing.first?.offset ?? -1) is \(differing.first?.element.0 ?? 0), not \(differing.first?.element.1 ?? 0)",
            sourceLocation: sourceLocation)
    }

    /// A sum of products computed in float32, element by element: its float64 value, and the bound
    /// c·ε·magnitude, plus one unit of `dtype` at the value when the result is bf16 or fp16.
    func sumBound(_ reference: (value: [Double], magnitude: [Double]), c: Double, dtype: DType)
        -> [(value: Double, bound: Double)]
    {
        let bits: Double? = dtype == .bfloat16 ? 8 : dtype == .float16 ? 11 : nil
        return zip(reference.value, reference.magnitude).map { value, magnitude in
            var bound = magnitude * (c * eps32)
            if let bits {
                let exponent = Foundation.log2(Swift.max(abs(value), 1e-30)).rounded(.down)
                bound += Foundation.exp2(exponent - (bits - 1))
            }
            return (value, bound)
        }
    }

    /// a·b and |a|·|b| in Double, computed on the host, for a of shape [..., m, k] and b of shape
    /// [..., k, n] with the same leading (batch) dimensions. MLX's own float64 matmul would share
    /// the float32 path's partitioning in gemms/cblas.cpp.
    func hostMatmul(_ a: MLXArray, _ b: MLXArray) -> (value: [Double], magnitude: [Double]) {
        let (m, k, n) = (a.dim(-2), a.dim(-1), b.dim(-1))
        precondition(
            b.dim(-2) == k && Array(a.shape.dropLast(2)) == Array(b.shape.dropLast(2)),
            "hostMatmul: \(a.shape) times \(b.shape)")
        let (x, y) = (doubles(a), doubles(b))
        var value = [Double](repeating: 0, count: x.count / k * n)
        var magnitude = value
        for batch in 0 ..< x.count / (m * k) {
            for i in 0 ..< m {
                let row = (batch * m + i) * n
                for p in 0 ..< k {
                    let xip = x[(batch * m + i) * k + p]
                    let column = (batch * k + p) * n
                    for j in 0 ..< n {
                        let product = xip * y[column + j]
                        value[row + j] += product
                        magnitude[row + j] += abs(product)
                    }
                }
            }
        }
        return (value, magnitude)
    }

    /// The matmul bound of a·b, whose result has `dtype` (`sumBound`), with its reference and
    /// magnitude computed on the host (`hostMatmul`).
    func matmulBound(_ a: MLXArray, _ b: MLXArray, c: Double, dtype: DType)
        -> [(value: Double, bound: Double)]
    {
        sumBound(hostMatmul(a, b), c: c, dtype: dtype)
    }

    /// The same bound for an affine quantized matmul x·wᵀ. The kernels compute s·Σ x·q + b·Σ x per
    /// group, so their rounding scales with Σ |x|·(|s|·q + |b|), which exceeds Σ |x·w| where s·q and b
    /// nearly cancel. dequantized() with |s| and |b| gives |s|·q + |b|.
    func quantizedBound(
        _ x: MLXArray, _ wq: MLXArray, scales: MLXArray, biases: MLXArray?, groupSize: Int,
        bits: Int, c: Double, dtype: DType
    ) -> [(value: Double, bound: Double)] {
        let s32 = scales.asType(.float32)
        let b32 = biases?.asType(.float32)
        let w = dequantized(
            wq, scales: s32, biases: b32, groupSize: groupSize, bits: bits, dtype: .float32)
        let wAbs = dequantized(
            wq, scales: abs(s32), biases: b32.map { abs($0) }, groupSize: groupSize, bits: bits,
            dtype: .float32)
        let x64 = x.asType(.float64)
        return sumBound(
            (
                doubles(matmul(x64, w.asType(.float64).transposed())),
                doubles(matmul(abs(x64), wAbs.asType(.float64).transposed()))
            ), c: c, dtype: dtype)
    }
#endif
