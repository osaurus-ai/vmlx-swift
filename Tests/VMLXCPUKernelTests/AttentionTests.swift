// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

#if os(Linux)
    import CmlxCPUShim
    import Foundation
    import MLX
    import Testing

    @Suite(.serialized) struct AttentionTests {
        enum Mask: String, Sendable { case none, causal, additive, boolean }

        struct Case: Sendable, CustomStringConvertible {
            var batch = 1, heads = 4, kvHeads = 4, queries = 5, keys = 17, headDim = 64
            var valueDim: Int? = nil
            var mask = Mask.none
            var sinks = false
            var dtype = DType.float32
            var dv: Int { valueDim ?? headDim }
            var description: String {
                "B\(batch) H\(heads)/\(kvHeads) M\(queries) L\(keys) D\(headDim)/\(dv) mask \(mask.rawValue)"
                    + "\(sinks ? " sinks" : "") \(dtype)"
            }
        }

        struct Inputs {
            let q: MLXArray, k: MLXArray, v: MLXArray
            let mask: MLXArray?, sinks: MLXArray?
            let scale: Float
        }

        /// Every head grouping, the generation (M = 1) and prompt paths, and head dims up to #3019's
        /// limit of 256. Head dims with a vector tail are in `masked`, `halves` and `eachPath`.
        static let sweep: [Case] = [4, 2, 1].flatMap { kvHeads in
            [(1, 1), (1, 17), (1, 200), (5, 17), (5, 200), (64, 64), (64, 200)].flatMap { m, l in
                [64, 80, 128, 256].map { Case(kvHeads: kvHeads, queries: m, keys: l, headDim: $0) }
            }
        }

        /// 64, 80, 128 and 256 are multiples of every vector width, so the head-dim tails (the scalar
        /// tail of the dot products, the zero-padded value rows, the partial store) run only at 70.
        static let masked: [Case] =
            [Mask.causal, .additive, .boolean].flatMap { mask in
                [(1, 200), (5, 200), (64, 200), (64, 64)].map {
                    Case(kvHeads: 2, queries: $0.0, keys: $0.1, headDim: 128, mask: mask)
                }
            } + [
                Case(kvHeads: 2, queries: 1, keys: 200, headDim: 70, mask: .causal),
                Case(kvHeads: 2, queries: 64, keys: 200, headDim: 70, mask: .additive),
                Case(kvHeads: 2, queries: 5, keys: 200, headDim: 128, sinks: true),
                Case(kvHeads: 2, queries: 64, keys: 200, headDim: 128, mask: .causal, sinks: true),
                Case(batch: 2, kvHeads: 2, queries: 5, keys: 33, headDim: 64, mask: .causal),
            ]

        static let halves: [Case] = [DType.bfloat16, .float16].flatMap { dtype in
            [(1, 200, Mask.none), (64, 200, Mask.causal)].flatMap { m, l, mask in
                [64, 70, 128].map {
                    Case(kvHeads: 2, queries: m, keys: l, headDim: $0, mask: mask, dtype: dtype)
                }
            }
        }

        /// One case per kernel path, for runs under an emulator (Task 31's SDE job), where the sweep
        /// across every target would take hours.
        static let eachPath: [Case] = [
            Case(kvHeads: 2, queries: 1, keys: 200, headDim: 128),
            Case(kvHeads: 2, queries: 64, keys: 200, headDim: 70, mask: .causal, sinks: true),
            Case(kvHeads: 1, queries: 5, keys: 17, headDim: 64, mask: .additive),
            Case(kvHeads: 2, queries: 64, keys: 64, headDim: 128, mask: .causal, dtype: .bfloat16),
            Case(kvHeads: 2, queries: 1, keys: 200, headDim: 70, dtype: .float16),
        ]

        /// Attention in float64, with each output element's bound. For one query row:
        /// - E_j, the error of score j: its D products and sum and the scale ((D + 2)·ε, twice the
        ///   analysis), for bf16 and fp16 the fallback's roundings of q·scale and of the score to the
        ///   dtype, and the mask's addition;
        /// - c, the probabilities' relative error: max E_j (softmax turns an argument's absolute error
        ///   into a relative one), the rounding of each argument s_j - max (range·ε), R + 1
        ///   exponentials at k_exp·ε each, where R counts the positions that can raise the running
        ///   maximum (an online softmax rescales at each, a sink once more), and for bf16 and fp16 the
        ///   probability's rounding to the dtype;
        /// - |o_i - ô_i| <= Σ_j p_j·|v_ji|·(2c + (L + 4)·ε + u_T) + ulp_T(o_i): c twice (the weight
        ///   and the normaliser), the sums over L positions, and the output's rounding; the lone u_T is
        ///   slack. The fallback also rounds `scale` itself to the dtype (fast.cpp's array(scale, dtype):
        ///   1/√128 moves by 1.07e-4 relative, 0.03·u in bf16), which E_j's 2·u_T covers.
        /// k_exp is `polynomialUnits("exp")`: MLX's fallback and #3019's kernels evaluate exp with MLX's
        /// polynomial, whose measured error also covers libm's.
        static func reference(
            _ c: Case, q: [Double], k: [Double], v: [Double], mask: [Double]?, sinks: [Double]?,
            scale: Double, kExp: Double
        ) -> [(value: Double, bound: Double)] {
            let (d, dv, lk) = (c.headDim, c.dv, c.keys)
            let half16 = c.dtype == .bfloat16 || c.dtype == .float16
            let uT = half16 ? unitRoundoff(c.dtype) : 0
            var out: [(value: Double, bound: Double)] = []
            for b in 0 ..< c.batch {
                for h in 0 ..< c.heads {
                    let kvh = h / (c.heads / c.kvHeads)
                    for m in 0 ..< c.queries {
                        let qRow = ((b * c.heads + h) * c.queries + m) * d
                        let limit = c.mask == .causal ? max(0, min(lk, lk - c.queries + m + 1)) : lk
                        var s = [Double](repeating: -.infinity, count: lk)
                        var e = [Double](repeating: 0, count: lk)
                        for j in 0 ..< limit {
                            let added = mask?[m * lk + j] ?? 0
                            if added == -.infinity { continue }
                            let kRow = ((b * c.kvHeads + kvh) * lk + j) * d
                            var dot = 0.0
                            var magnitude = 0.0
                            for i in 0 ..< d {
                                dot += q[qRow + i] * k[kRow + i]
                                magnitude += abs(q[qRow + i] * k[kRow + i])
                            }
                            s[j] = scale * dot + added
                            e[j] =
                                scale * magnitude * (Double(d + 2) * eps32 + 2 * uT) + abs(s[j])
                                * uT
                                + abs(added) * eps32
                        }
                        let live = (0 ..< lk).filter { s[$0] > -.infinity }
                        let sink = sinks?[h]
                        let top = max(live.map { s[$0] }.max() ?? -.infinity, sink ?? -.infinity)
                        let low = min(live.map { s[$0] }.min() ?? .infinity, sink ?? .infinity)
                        let w = s.map { $0 == -.infinity ? 0 : Foundation.exp($0 - top) }
                        let z = w.reduce(0, +) + (sink.map { Foundation.exp($0 - top) } ?? 0)
                        let largest = e.max() ?? 0
                        var records = 0
                        var running = -Double.infinity
                        for j in live {
                            if s[j] > running - 2 * largest { records += 1 }
                            running = max(running, s[j])
                        }
                        let relative =
                            largest + (top - low) * eps32 + Double(records + 1) * kExp * eps32 + uT
                        for i in 0 ..< dv {
                            var value = 0.0
                            var magnitude = 0.0
                            for j in live {
                                let vji = v[((b * c.kvHeads + kvh) * lk + j) * dv + i]
                                value += w[j] / z * vji
                                magnitude += w[j] / z * abs(vji)
                            }
                            let bound =
                                magnitude * (2 * relative + Double(lk + 4) * eps32 + uT)
                                + (half16 ? ulp(value, in: c.dtype) : 0)
                            out.append((value, bound))
                        }
                    }
                }
            }
            return out
        }

        static func prepare(_ c: Case) throws -> (Inputs, [(value: Double, bound: Double)]) {
            let seed = UInt64(c.queries * 1009 + c.keys * 31 + c.headDim + c.kvHeads)
            let q = MLXRandom.normal(
                [c.batch, c.heads, c.queries, c.headDim], key: MLXRandom.key(seed)
            )
            .asType(c.dtype)
            let k = MLXRandom.normal(
                [c.batch, c.kvHeads, c.keys, c.headDim], key: MLXRandom.key(seed + 1)
            )
            .asType(c.dtype)
            let v = MLXRandom.normal(
                [c.batch, c.kvHeads, c.keys, c.dv], key: MLXRandom.key(seed + 2)
            )
            .asType(c.dtype)
            var additive: [Double]? = nil
            var mask: MLXArray? = nil
            if c.mask == .additive || c.mask == .boolean {
                var random = SeededRandom(seed: seed + 3)
                // Position 0 stays visible, so no row is masked entirely.
                let keep = (0 ..< c.queries * c.keys).map {
                    $0 % c.keys == 0 || random.next() % 4 != 0
                }
                if c.mask == .boolean {
                    mask = MLXArray(keep, [c.queries, c.keys])
                    additive = keep.map { $0 ? 0 : -.infinity }
                } else {
                    let values: [Float] = keep.enumerated().map { i, visible in
                        visible ? (i % 3 == 0 ? -1.5 : 0) : -.infinity
                    }
                    mask = MLXArray(values, [c.queries, c.keys]).asType(c.dtype)
                    additive = values.map(Double.init)
                }
            }
            let sinkValues: [Float]? =
                c.sinks ? (0 ..< c.heads).map { [0.5, -1, 1.5, 0.25][$0 % 4] } : nil
            let scale = 1 / Float(c.headDim).squareRoot()
            let expected = reference(
                c, q: doubles(q), k: doubles(k), v: doubles(v), mask: additive,
                sinks: sinkValues?.map(Double.init), scale: Double(scale),
                kExp: try polynomialUnits("exp"))
            let inputs = Inputs(
                q: q, k: k, v: v, mask: mask,
                sinks: sinkValues.map { MLXArray($0).asType(c.dtype) },
                scale: scale)
            return (inputs, expected)
        }

        static func attention(_ c: Case, _ inputs: Inputs) -> MLXArray {
            let mode: MLXFast.ScaledDotProductAttentionMaskMode =
                c.mask == .causal ? .causal : inputs.mask.map { .array($0) } ?? .none
            return MLXFast.scaledDotProductAttention(
                queries: inputs.q, keys: inputs.k, values: inputs.v, scale: inputs.scale,
                mask: mode,
                sinks: inputs.sinks)
        }

        static func check(_ c: Case) throws {
            try KernelLock.run {
                let (inputs, expected) = try prepare(c)
                forEachTarget(VMLX_CPU_FAMILY_SDPA) { target in
                    expectWithin(attention(c, inputs), expected, "sdpa \(c) on \(target)")
                }
            }
        }

        @Test(arguments: sweep + masked + halves)
        func withinTheBound(_ c: Case) throws {
            try Self.check(c)
        }

        @Test(arguments: eachPath)
        func eachPathWithinTheBound(_ c: Case) throws {
            try Self.check(c)
        }

        /// What #3019's kernel does not handle goes to MLX's fallback (Tasks 15 and 17) and is right
        /// there: a value head dim unlike the query's (MLA), head dims over 256, and float64.
        @Test(arguments: [
            Case(headDim: 192, valueDim: 128), Case(headDim: 128, valueDim: 192),
            Case(queries: 1, headDim: 320), Case(headDim: 320), Case(dtype: .float64),
        ])
        func whatTheKernelDeclines(_ c: Case) throws {
            try KernelLock.run {
                let (inputs, expected) = try Self.prepare(c)
                vmlx_cpu_reset_counters()
                expectWithin(Self.attention(c, inputs), expected, "sdpa \(c)")
                #expect(
                    vmlx_cpu_highway_calls(VMLX_CPU_FAMILY_SDPA) == 0,
                    "\(c) must take MLX's fallback")
            }
        }
    }
#endif
