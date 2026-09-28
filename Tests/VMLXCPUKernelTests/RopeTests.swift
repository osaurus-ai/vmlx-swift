// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

#if os(Linux)
    import CmlxCPUShim
    import Foundation
    import MLX
    import Testing

    @Suite(.serialized) struct RopeTests {
        struct Case: Sendable, CustomStringConvertible {
            var heads = 2, length = 5, headDim = 64, dims = 64
            var traditional = false
            var base: Float? = 10000
            var scale: Float = 1
            /// One offset, or one per batch entry (MLX's array-offset form).
            var offsets = [0]
            var freqs = false
            var dtype = DType.float32
            var description: String {
                "dims \(dims)/\(headDim) L \(length) offsets \(offsets) traditional \(traditional) "
                    + "\(freqs ? "freqs" : "base \(base ?? 0)") scale \(scale) \(dtype)"
            }
        }

        /// Positions stay at or below 200, so every angle lies in the range the sin and cos baselines
        /// were measured on (Task 24). A rotated half of 10 (dims 20) leaves a tail at 4, 8 and 16
        /// lanes; the others are multiples of 16.
        static let cases: [Case] = [
            Case(),
            Case(length: 33, offsets: [37]),
            Case(length: 1, offsets: [200]),
            Case(traditional: true, offsets: [37]),
            Case(headDim: 64, dims: 32, offsets: [5]),
            Case(headDim: 64, dims: 32, traditional: true, offsets: [5]),
            Case(headDim: 64, dims: 20, offsets: [5]),
            Case(headDim: 64, dims: 20, traditional: true, offsets: [5]),
            Case(headDim: 128, dims: 128, base: 500000, offsets: [100]),
            Case(scale: 0.25, offsets: [64]),
            Case(offsets: [3, 17]),
            Case(base: nil, offsets: [9], freqs: true),
            // 524288 elements, past 262144: #3019 splits the heads of both batch entries across the pool.
            Case(heads: 32, length: 64, headDim: 128, dims: 128, offsets: [3, 17]),
            Case(offsets: [37], dtype: .bfloat16),
            Case(traditional: true, offsets: [37], dtype: .float16),
        ]

        /// RoPE in float64, with each element's bound. An angle θ = position·base^(-j/half) leaves
        /// float32 with relative error at most (k_exp + 2·|ln base| + 4)·ε: exp(-j·ln(base)/half)
        /// magnifies its argument's rounding by up to |ln base|, and MLX's fallback evaluates exp with its
        /// polynomial (k_exp). So cos θ and sin θ err by |θ| times that, plus the trig function's own
        /// error, k_trig·ε (libm in #3019's kernel, MLX's polynomial in the fallback); the rotation adds
        /// its products and sum. For bf16 and fp16 the fallback also rounds cos, sin, both products and
        /// the sum to the dtype. Features past `dims` are copied, so their bound is 0.
        static func reference(
            _ c: Case, x: [Double], freqs: [Double]?, kExp: Double, kTrig: Double
        ) -> [(value: Double, bound: Double)] {
            let half = c.dims / 2
            let logBase = c.base.map { abs(Foundation.log(Double($0))) } ?? 0
            let half16 = c.dtype == .bfloat16 || c.dtype == .float16
            let uT = half16 ? unitRoundoff(c.dtype) : 0
            var out = x.map { (value: $0, bound: 0.0) }
            for b in 0 ..< c.offsets.count {
                for h in 0 ..< c.heads {
                    for t in 0 ..< c.length {
                        let row = ((b * c.heads + h) * c.length + t) * c.headDim
                        let position = Double(t + c.offsets[b]) * Double(c.scale)
                        for j in 0 ..< half {
                            let inverse =
                                freqs.map { 1 / $0[j] }
                                ?? Foundation.pow(Double(c.base ?? 1), -Double(j) / Double(half))
                            let theta = position * inverse
                            let (ia, ib) =
                                c.traditional
                                ? (row + 2 * j, row + 2 * j + 1) : (row + j, row + j + half)
                            let (xa, xb) = (x[ia], x[ib])
                            let (cosine, sine) = (Foundation.cos(theta), Foundation.sin(theta))
                            let ya = xa * cosine - xb * sine
                            let yb = xa * sine + xb * cosine
                            let common =
                                (abs(xa) + abs(xb))
                                * ((abs(theta) * (kExp + 2 * logBase + 4) + kTrig + 4) * eps32 + 4
                                    * uT)
                            out[ia] = (ya, common + (half16 ? ulp(ya, in: c.dtype) : 0))
                            out[ib] = (yb, common + (half16 ? ulp(yb, in: c.dtype) : 0))
                        }
                    }
                }
            }
            return out
        }

        static func rope(_ c: Case, _ x: MLXArray, _ freqs: MLXArray?) -> MLXArray {
            c.offsets.count == 1
                ? MLXFast.RoPE(
                    x, dimensions: c.dims, traditional: c.traditional, base: c.base, scale: c.scale,
                    offset: c.offsets[0], freqs: freqs)
                : MLXFast.RoPE(
                    x, dimensions: c.dims, traditional: c.traditional, base: c.base, scale: c.scale,
                    offset: MLXArray(c.offsets.map(Int32.init)), freqs: freqs)
        }

        @Test(arguments: cases)
        func withinTheBound(_ c: Case) throws {
            try KernelLock.run {
                let kExp = try polynomialUnits("exp")
                let kTrig = try max(polynomialUnits("sin"), polynomialUnits("cos"))
                let x = MLXRandom.normal(
                    [c.offsets.count, c.heads, c.length, c.headDim],
                    key: MLXRandom.key(UInt64(c.length * 131 + c.dims))
                ).asType(c.dtype)
                let freqs = c.freqs ? MLXArray((0 ..< c.dims / 2).map { Float(1 + $0 * $0) }) : nil
                let expected = Self.reference(
                    c, x: doubles(x), freqs: freqs.map(doubles), kExp: kExp, kTrig: kTrig)
                forEachTarget(VMLX_CPU_FAMILY_ROPE) { target in
                    expectWithin(Self.rope(c, x, freqs), expected, "rope \(c) on \(target)")
                }
            }
        }

        /// A view whose heads and positions are swapped, as in a model that transposes [B, L, H, D]
        /// before RoPE: the feature axis stays contiguous. Then float64, which goes to the fallback.
        @Test func viewsAndFloat64() throws {
            try KernelLock.run {
                let kExp = try polynomialUnits("exp")
                let kTrig = try max(polynomialUnits("sin"), polynomialUnits("cos"))
                let c = Case(heads: 4, length: 9, offsets: [11])
                let x = MLXRandom.normal([1, c.length, c.heads, c.headDim], key: MLXRandom.key(41))
                    .transposed(0, 2, 1, 3)
                let expected = Self.reference(
                    c, x: doubles(x), freqs: nil, kExp: kExp, kTrig: kTrig)
                forEachTarget(VMLX_CPU_FAMILY_ROPE) { target in
                    expectWithin(Self.rope(c, x, nil), expected, "rope view on \(target)")
                }
                vmlx_cpu_reset_counters()
                expectWithin(Self.rope(c, x.asType(.float64), nil), expected, "rope float64")
                if Highway.enabled {
                    #expect(vmlx_cpu_fallback_calls(VMLX_CPU_FAMILY_ROPE) > 0)
                }
            }
        }
    }
#endif
