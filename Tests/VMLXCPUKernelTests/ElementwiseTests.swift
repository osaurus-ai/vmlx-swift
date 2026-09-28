// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

#if os(Linux)
    import CmlxCPUShim
    import Foundation
    import Glibc
    import MLX
    import Testing

    let floatTypes: [DType] = [.float32, .float64, .bfloat16, .float16]

    /// One correctly rounded operation, or two for rsqrt: bit for bit.
    enum ExactUnary: String, CaseIterable, Sendable {
        case abs, negative, floor, ceil, round, sqrt, square, reciprocal, rsqrt, sign

        var domain: ClosedRange<Double> {
            switch self {
            case .sqrt, .rsqrt: return 0.001 ... 1000
            case .reciprocal: return 0.01 ... 100
            default: return -100 ... 100
            }
        }

        func apply(_ x: MLXArray) -> MLXArray {
            switch self {
            case .abs: return MLX.abs(x)
            case .negative: return MLX.negative(x)
            case .floor: return MLX.floor(x)
            case .ceil: return MLX.ceil(x)
            case .round: return MLX.round(x)
            case .sqrt: return MLX.sqrt(x)
            case .square: return MLX.square(x)
            case .reciprocal: return MLX.reciprocal(x)
            case .rsqrt: return MLX.rsqrt(x)
            case .sign: return MLX.sign(x)
            }
        }

        func exact<T: BinaryFloatingPoint>(_ x: T) -> T {
            switch self {
            case .abs: return Swift.abs(x)
            case .negative: return -x
            case .floor: return x.rounded(.down)
            case .ceil: return x.rounded(.up)
            case .round: return x.rounded(.toNearestOrEven)
            case .sqrt: return x.squareRoot()
            case .square: return x * x
            case .reciprocal: return 1 / x
            case .rsqrt: return 1 / x.squareRoot()
            case .sign: return x > 0 ? 1 : x < 0 ? -1 : 0
            }
        }

        /// MLX's result: computed in the dtype's precision, or for bf16 and fp16 in float32 with each
        /// operation's result rounded to the dtype.
        func expected(_ x: Double, _ dtype: DType) -> Double {
            switch dtype {
            case .float64: return exact(x)
            case .float32: return Double(exact(Float(x)))
            default:
                if self == .rsqrt {
                    return Double(roundTo(dtype, 1 / roundTo(dtype, Float(x).squareRoot())))
                }
                return Double(roundTo(dtype, exact(Float(x))))
            }
        }
    }

    /// Functions both paths take from libm, one element at a time.
    enum LibmUnary: String, CaseIterable, Sendable {
        case acos, acosh, asin, asinh, atan, atanh, cosh, expm1, log, log1p, log2, log10, sinh, tan,
            tanh

        var domain: ClosedRange<Double> {
            switch self {
            case .acos, .asin, .atanh: return -0.999 ... 0.999
            case .acosh: return 1 ... 1000
            case .log, .log2, .log10: return 0.0001 ... 10000
            case .log1p: return -0.999 ... 1000
            default: return -20 ... 20
            }
        }

        func apply(_ x: MLXArray) -> MLXArray {
            switch self {
            case .acos: return MLX.acos(x)
            case .acosh: return MLX.acosh(x)
            case .asin: return MLX.asin(x)
            case .asinh: return MLX.asinh(x)
            case .atan: return MLX.atan(x)
            case .atanh: return MLX.atanh(x)
            case .cosh: return MLX.cosh(x)
            case .expm1: return MLX.expm1(x)
            case .log: return MLX.log(x)
            case .log1p: return MLX.log1p(x)
            case .log2: return MLX.log2(x)
            case .log10: return MLX.log10(x)
            case .sinh: return MLX.sinh(x)
            case .tan: return MLX.tan(x)
            case .tanh: return MLX.tanh(x)
            }
        }

        func float(_ x: Float) -> Float {
            switch self {
            case .acos: return Glibc.acosf(x)
            case .acosh: return Glibc.acoshf(x)
            case .asin: return Glibc.asinf(x)
            case .asinh: return Glibc.asinhf(x)
            case .atan: return Glibc.atanf(x)
            case .atanh: return Glibc.atanhf(x)
            case .cosh: return Glibc.coshf(x)
            case .expm1: return Glibc.expm1f(x)
            case .log: return Glibc.logf(x)
            case .log1p: return Glibc.log1pf(x)
            case .log2: return Glibc.log2f(x)
            case .log10: return Glibc.log10f(x)
            case .sinh: return Glibc.sinhf(x)
            case .tan: return Glibc.tanf(x)
            case .tanh: return Glibc.tanhf(x)
            }
        }

        func double(_ x: Double) -> Double {
            switch self {
            case .acos: return Glibc.acos(x)
            case .acosh: return Glibc.acosh(x)
            case .asin: return Glibc.asin(x)
            case .asinh: return Glibc.asinh(x)
            case .atan: return Glibc.atan(x)
            case .atanh: return Glibc.atanh(x)
            case .cosh: return Glibc.cosh(x)
            case .expm1: return Glibc.expm1(x)
            case .log: return Glibc.log(x)
            case .log1p: return Glibc.log1p(x)
            case .log2: return Glibc.log2(x)
            case .log10: return Glibc.log10(x)
            case .sinh: return Glibc.sinh(x)
            case .tan: return Glibc.tan(x)
            case .tanh: return Glibc.tanh(x)
            }
        }

        func expected(_ x: Double, _ dtype: DType) -> Double {
            switch dtype {
            case .float64: return double(x)
            case .float32: return Double(float(Float(x)))
            default: return Double(roundTo(dtype, float(Float(x))))
            }
        }
    }

    enum ExactBinary: String, CaseIterable, Sendable {
        case add, subtract, multiply, divide, maximum, minimum, arctan2, power, remainder

        /// b's magnitude stays at least 0.5 for divide and remainder, where the test negates every
        /// third b.
        var domains: (ClosedRange<Double>, ClosedRange<Double>) {
            switch self {
            case .divide, .remainder: return (-100 ... 100, 0.5 ... 100)
            case .power: return (0.01 ... 10, -3 ... 3)
            default: return (-100 ... 100, -100 ... 100)
            }
        }

        func apply(_ a: MLXArray, _ b: MLXArray) -> MLXArray {
            switch self {
            case .add: return a + b
            case .subtract: return a - b
            case .multiply: return a * b
            case .divide: return a / b
            case .maximum: return MLX.maximum(a, b)
            case .minimum: return MLX.minimum(a, b)
            case .arctan2: return MLX.atan2(a, b)
            case .power: return MLX.pow(a, b)
            case .remainder: return MLX.remainder(a, b)
            }
        }

        /// As `base_simd.h` computes it: NaN in a wins for maximum and minimum, and remainder moves
        /// std::remainder's result into b's sign (#3019's std::fmod gives the same, bit for bit).
        func float(_ a: Float, _ b: Float) -> Float {
            switch self {
            case .add: return a + b
            case .subtract: return a - b
            case .multiply: return a * b
            case .divide: return a / b
            case .maximum: return a.isNaN ? a : (a > b ? a : b)
            case .minimum: return a.isNaN ? a : (a < b ? a : b)
            case .arctan2: return Glibc.atan2f(a, b)
            case .power: return Glibc.powf(a, b)
            case .remainder:
                var r = Glibc.remainderf(a, b)
                if r != 0 && (r < 0) != (b < 0) { r += b }
                return r
            }
        }

        func double(_ a: Double, _ b: Double) -> Double {
            switch self {
            case .add: return a + b
            case .subtract: return a - b
            case .multiply: return a * b
            case .divide: return a / b
            case .maximum: return a.isNaN ? a : (a > b ? a : b)
            case .minimum: return a.isNaN ? a : (a < b ? a : b)
            case .arctan2: return Glibc.atan2(a, b)
            case .power: return Glibc.pow(a, b)
            case .remainder:
                var r = Glibc.remainder(a, b)
                if r != 0 && (r < 0) != (b < 0) { r += b }
                return r
            }
        }

        func expected(_ a: Double, _ b: Double, _ dtype: DType) -> Double {
            switch dtype {
            case .float64: return double(a, b)
            case .float32: return Double(float(Float(a), Float(b)))
            default: return Double(roundTo(dtype, float(Float(a), Float(b))))
            }
        }
    }

    @Suite(.serialized) struct ElementwiseTests {
        @Test(arguments: ExactUnary.allCases, floatTypes)
        func correctlyRoundedUnary(_ op: ExactUnary, dtype: DType) {
            KernelLock.run {
                let (x, xs) = materialize(spread(op.domain, seed: 71), dtype)
                expectIdentical(op.apply(x), xs.map { op.expected($0, dtype) }, "\(op) \(dtype)")
            }
        }

        @Test(arguments: LibmUnary.allCases, floatTypes)
        func libmUnary(_ op: LibmUnary, dtype: DType) {
            KernelLock.run {
                let (x, xs) = materialize(spread(op.domain, seed: 72), dtype)
                expectIdentical(op.apply(x), xs.map { op.expected($0, dtype) }, "\(op) \(dtype)")
            }
        }

        @Test(arguments: ExactBinary.allCases, floatTypes)
        func binary(_ op: ExactBinary, dtype: DType) {
            KernelLock.run {
                let (a, avalues) = materialize(spread(op.domains.0, seed: 73), dtype)
                let raw = spread(op.domains.1, seed: 74)
                let signed =
                    op == .divide || op == .remainder
                    ? raw.enumerated().map { $0.offset % 3 == 0 ? -$0.element : $0.element } : raw
                let (b, bvalues) = materialize(signed, dtype)
                let expected = zip(avalues, bvalues).map { op.expected($0, $1, dtype) }
                expectIdentical(op.apply(a, b), expected, "\(op) \(dtype)")
            }
        }

        /// A broadcast scalar and a transposed view take other loops than two contiguous arrays.
        @Test(arguments: floatTypes)
        func broadcastsAndViews(dtype: DType) {
            KernelLock.run {
                let (a, avalues) = materialize(
                    spread(-100 ... 100, count: 37 * 111, seed: 75), dtype, shape: [37, 111])
                let (s, svalues) = materialize([2.75], dtype)
                expectIdentical(
                    a * s, avalues.map { ExactBinary.multiply.expected($0, svalues[0], dtype) },
                    "a * scalar \(dtype)")
                let view = a.transposed()
                let viewValues = doubles(view)
                expectIdentical(
                    view + view, viewValues.map { ExactBinary.add.expected($0, $0, dtype) },
                    "view + view \(dtype)")
                // Two column-contiguous views still take the contiguous loop; a column-contiguous view
                // with a row-contiguous array takes the general one.
                let (b, bvalues) = materialize(
                    spread(-100 ... 100, count: 111 * 37, seed: 77), dtype, shape: [111, 37])
                expectIdentical(
                    view + b,
                    zip(viewValues, bvalues).map { ExactBinary.add.expected($0, $1, dtype) },
                    "view + contiguous \(dtype)")
            }
        }

        /// IEEE comparisons: NaN is unordered, and -0 equals +0.
        @Test(arguments: floatTypes)
        func comparisons(dtype: DType) {
            KernelLock.run {
                var a = spread(-4 ... 4, seed: 91)
                var b = spread(-4 ... 4, seed: 92)
                for i in stride(from: 0, to: a.count, by: 7) { b[i] = a[i] }  // ties
                (a[3], b[3]) = (0, -0.0)
                (a[5], b[6]) = (.nan, .nan)
                let (x, xs) = materialize(a, dtype)
                let (y, ys) = materialize(b, dtype)
                let ops: [(String, (MLXArray, MLXArray) -> MLXArray, (Double, Double) -> Bool)] = [
                    ("equal", { MLX.equal($0, $1) }, { $0 == $1 }),
                    ("notEqual", { MLX.notEqual($0, $1) }, { $0 != $1 }),
                    ("less", { MLX.less($0, $1) }, { $0 < $1 }),
                    ("lessEqual", { MLX.lessEqual($0, $1) }, { $0 <= $1 }),
                    ("greater", { MLX.greater($0, $1) }, { $0 > $1 }),
                    ("greaterEqual", { MLX.greaterEqual($0, $1) }, { $0 >= $1 }),
                ]
                for (name, apply, truth) in ops {
                    let got = apply(x, y).asArray(Bool.self)
                    let differing = zip(got, zip(xs, ys).map { truth($0, $1) }).filter { $0 != $1 }
                        .count
                    #expect(differing == 0, "\(name) \(dtype): \(differing) of \(got.count) differ")
                }
            }
        }

        /// int32 arithmetic and bitwise operations, and uint32 shifts. MLX's floor_divide floors an
        /// integer quotient, as a Double quotient rounded down does exactly for these operands, and
        /// remainder takes the divisor's sign.
        @Test func integerAndBitwise() {
            KernelLock.run {
                var random = SeededRandom(seed: 93)
                let count = 4099
                let a = (0 ..< count).map { _ in Int32(Int(random.next() % 2001) - 1000) }
                let b = (0 ..< count).map { _ -> Int32 in
                    let v = Int32(Int(random.next() % 2001) - 1000)
                    return v == 0 ? 7 : v
                }
                let (x, y) = (MLXArray(a), MLXArray(b))
                let remainder = { (p: Int32, q: Int32) -> Int32 in
                    let r = p % q
                    return r != 0 && (r < 0) != (q < 0) ? r + q : r
                }
                let ops: [(String, (MLXArray, MLXArray) -> MLXArray, (Int32, Int32) -> Int32)] = [
                    ("add", { $0 + $1 }, { $0 + $1 }),
                    ("subtract", { $0 - $1 }, { $0 - $1 }),
                    ("multiply", { $0 * $1 }, { $0 * $1 }),
                    (
                        "floorDivide", { MLX.floorDivide($0, $1) },
                        {
                            Int32((Double($0) / Double($1)).rounded(.down))
                        }
                    ),
                    ("remainder", { MLX.remainder($0, $1) }, remainder),
                    ("bitwiseAnd", { MLX.bitwiseAnd($0, $1) }, { $0 & $1 }),
                    ("bitwiseOr", { MLX.bitwiseOr($0, $1) }, { $0 | $1 }),
                    ("bitwiseXOr", { MLX.bitwiseXOr($0, $1) }, { $0 ^ $1 }),
                    ("maximum", { MLX.maximum($0, $1) }, { Swift.max($0, $1) }),
                    ("minimum", { MLX.minimum($0, $1) }, { Swift.min($0, $1) }),
                ]
                for (name, apply, truth) in ops {
                    let got = apply(x, y).asArray(Int32.self)
                    let differing = zip(got, zip(a, b).map { truth($0, $1) }).filter { $0 != $1 }
                        .count
                    #expect(differing == 0, "int32 \(name): \(differing) of \(count) differ")
                }
                let u = (0 ..< count).map { _ in UInt32(truncatingIfNeeded: random.next()) }
                let shift = (0 ..< count).map { _ in UInt32(random.next() % 32) }
                let (ux, us) = (MLXArray(u), MLXArray(shift))
                #expect(
                    MLX.leftShift(ux, us).asArray(UInt32.self) == zip(u, shift).map { $0 << $1 })
                #expect(
                    MLX.rightShift(ux, us).asArray(UInt32.self) == zip(u, shift).map { $0 >> $1 })
            }
        }

        @Test func logical() {
            KernelLock.run {
                var random = SeededRandom(seed: 94)
                let a = (0 ..< 4099).map { _ in random.next() % 2 == 0 }
                let b = (0 ..< 4099).map { _ in random.next() % 3 == 0 }
                let (x, y) = (MLXArray(a), MLXArray(b))
                #expect(MLX.logicalAnd(x, y).asArray(Bool.self) == zip(a, b).map { $0 && $1 })
                #expect(MLX.logicalOr(x, y).asArray(Bool.self) == zip(a, b).map { $0 || $1 })
                #expect(MLX.logicalNot(x).asArray(Bool.self) == a.map { !$0 })
            }
        }

        /// Past MIN_TOTAL_ELEMENTS, so #3019 splits the work across the pool. Bit for bit: contiguous,
        /// with a scalar, a strided unary view (every other column: not contiguous), and a binary
        /// operation whose operands are laid out differently in four outer dimensions (the general
        /// loop, which threads only from four outer dimensions on).
        @Test func threadedElementwise() {
            KernelLock.run {
                let n = 512 * 1024
                let (x, xs) = materialize(
                    spread(0.001 ... 1000, count: n, seed: 95), .float32, shape: [512, 1024])
                let (y, ys) = materialize(
                    spread(-100 ... 100, count: n, seed: 96), .float32, shape: [512, 1024])
                expectIdentical(
                    MLX.sqrt(x), xs.map { Double(Float($0).squareRoot()) }, "threaded sqrt")
                expectIdentical(
                    x + y, zip(xs, ys).map { Double(Float($0) + Float($1)) }, "threaded x + y")
                expectIdentical(
                    x * 2.75, xs.map { Double(Float($0) * 2.75) }, "threaded x * scalar")
                let (wide, _) = materialize(
                    spread(0.001 ... 1000, count: 512 * 2048, seed: 97), .float32,
                    shape: [512, 2048])
                let evens = wide[.ellipsis, .stride(by: 2)]
                expectIdentical(
                    MLX.sqrt(evens), doubles(evens).map { Double(Float($0).squareRoot()) },
                    "threaded strided sqrt")
                let (g, _) = materialize(
                    spread(-100 ... 100, count: n, seed: 98), .float32, shape: [2, 2, 2, 64, 1024])
                let gt = g.transposed(3, 2, 1, 0, 4)
                let (h, hs) = materialize(
                    spread(-100 ... 100, count: n, seed: 99), .float32, shape: [64, 2, 2, 2, 1024])
                expectIdentical(
                    gt + h, zip(doubles(gt), hs).map { Double(Float($0) + Float($1)) },
                    "threaded general binary")
            }
        }

        @Test func maximumAndMinimumPropagateNaN() {
            KernelLock.run {
                for dtype in floatTypes {
                    let a = MLXArray([Float.nan, 1, .nan, -2] + [Float](repeating: 1, count: 13))
                        .asType(dtype)
                    let b = MLXArray([1, Float.nan, .nan, -3] + [Float](repeating: 2, count: 13))
                        .asType(dtype)
                    for (name, result) in [
                        ("maximum", MLX.maximum(a, b)), ("minimum", MLX.minimum(a, b)),
                    ] {
                        let values = doubles(result)
                        #expect(
                            values[0].isNaN && values[1].isNaN && values[2].isNaN,
                            "\(name) \(dtype): \(values.prefix(4))")
                    }
                }
            }
        }

        /// base_simd.h's rule: a negative exponent gives 0, otherwise repeated squaring.
        @Test func integerPower() {
            KernelLock.run {
                var bases: [Int32] = []
                var exponents: [Int32] = []
                for base in Int32(-5) ... 5 {
                    for exponent in Int32(-3) ... 10 {
                        bases.append(base)
                        exponents.append(exponent)
                    }
                }
                let got = MLX.pow(MLXArray(bases), MLXArray(exponents)).asArray(Int32.self)
                for i in bases.indices {
                    var expected: Int32 = exponents[i] < 0 ? 0 : 1
                    for _ in 0 ..< Swift.max(exponents[i], 0) { expected &*= bases[i] }
                    #expect(got[i] == expected, "\(bases[i]) ^ \(exponents[i])")
                }
            }
        }

        /// One polynomial function in bf16 or fp16 on `domain`, against `truth` and the bound of
        /// `polynomialsInHalfPrecision`.
        static func checkInHalfPrecision(
            _ name: String, _ domain: ClosedRange<Double>, _ apply: (MLXArray) -> MLXArray,
            _ truth: (Double) -> Double, dtype: DType
        ) throws {
            let k = try polynomialUnits(name)
            // The floor the baseline was measured with: erf's is 2^-24, the others' 2^-33.
            let unitFloor = polynomialFloor(name)
            let (x, xs) = materialize(spread(domain, seed: 76), dtype)
            let expected = xs.map { v -> (value: Double, bound: Double) in
                let t = truth(v)
                return (t, k * max(Double(Float(t).ulp), unitFloor) + ulp(t, in: dtype))
            }
            expectWithin(apply(x), expected, "\(name) \(dtype)")
        }

        /// MLX's polynomials in bf16 and fp16: the float32 result's error (`polynomialUnits`, the
        /// scalar code's measured error plus 2 units, in units of the float32 result) plus one unit
        /// of the dtype for the final rounding.
        @Test(arguments: [DType.bfloat16, .float16])
        func polynomialsInHalfPrecision(dtype: DType) throws {
            try KernelLock.run {
                let cases:
                    [(String, ClosedRange<Double>, (MLXArray) -> MLXArray, (Double) -> Double)] = [
                        ("exp", -10 ... 10, { MLX.exp($0) }, { Foundation.exp($0) }),
                        ("sin", -300 ... 300, { MLX.sin($0) }, { Foundation.sin($0) }),
                        ("cos", -300 ... 300, { MLX.cos($0) }, { Foundation.cos($0) }),
                        ("erf", -5 ... 5, { MLX.erf($0) }, { Foundation.erf($0) }),
                    ]
                for (name, domain, apply, truth) in cases {
                    try Self.checkInHalfPrecision(name, domain, apply, truth, dtype: dtype)
                }
            }
        }

        /// sigmoid, as `polynomialsInHalfPrecision`.
        @Test(
            .disabled(
                if: !Highway.enabled,
                "the scalar code rounds exp(|x|) to the dtype, which overflows fp16 below x = -11.09"
            ),
            arguments: [DType.bfloat16, .float16])
        func sigmoidInHalfPrecision(dtype: DType) throws {
            try KernelLock.run {
                try Self.checkInHalfPrecision(
                    "sigmoid", -30 ... 30, { MLX.sigmoid($0) }, { 1 / (1 + Foundation.exp(-$0)) },
                    dtype: dtype)
            }
        }
    }
#endif
