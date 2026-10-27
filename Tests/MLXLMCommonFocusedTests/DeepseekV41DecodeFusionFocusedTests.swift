// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// DeepSeek-V4.1 decode / dSpark-verify fusion (1-8 rows). Synthetic tensors with the real V4.1-Flash
// shapes and bit plan (hidden 5120, 384 routed experts of width 2304, top-6, 2/3-bit group-64 gate/up,
// 3-bit down, 8-bit shared expert, bf16 router); no model loads and no shared MLX test lock, so run this
// suite on its own. The configuration is a real V4.1 config.json (`VMLX_DSV41_CONFIG`, or the bundle's under
// `VMLX_DSV41_BUNDLE`; see DSV41TestBundle); the suite is disabled without one.
//
// Microbenchmarks run only with VMLX_DSV41_DEC_BENCH=1.

import Cmlx
import Foundation
import MLX
import MLXFast
import MLXNN
import MLXRandom
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

enum DSV41Synth {
    static let config: DeepseekV4Configuration? = {
        guard let url = DSV41TestBundle.config, let data = FileManager.default.contents(atPath: url.path)
        else { return nil }
        return try? JSONDecoder().decode(DeepseekV4Configuration.self, from: data)
    }()

    static let benchEnabled = ProcessInfo.processInfo.environment["VMLX_DSV41_DEC_BENCH"] == "1"

    /// Uniformly random packed words (every bit pattern, so every quantized value occurs).
    static func packed(_ shape: [Int], seed: UInt64) -> MLXArray {
        MLXRandom.randInt(Int32.min ..< Int32.max, shape, key: MLXRandom.key(seed))
            .view(dtype: .uint32)
    }

    /// Scales in [0.004, 0.02], biases centring the code range, both bf16.
    static func scalesBiases(_ shape: [Int], bits: Int, seed: UInt64) -> (MLXArray, MLXArray) {
        let s = MLXRandom.uniform(low: 0.004, high: 0.02, shape, key: MLXRandom.key(seed &+ 1))
        let jitter = MLXRandom.uniform(low: -0.5, high: 0.5, shape, key: MLXRandom.key(seed &+ 2))
        let b = -(s * Float((1 << bits) - 1) / 2) + s * jitter
        return (s.asType(.bfloat16), b.asType(.bfloat16))
    }

    static func switchLinear(
        experts: Int, input: Int, output: Int, bits: Int, seed: UInt64
    ) -> QuantizedSwitchLinear {
        let (s, b) = scalesBiases([experts, output, input / 64], bits: bits, seed: seed)
        return QuantizedSwitchLinear(
            inputDims: input, outputDims: output, numExperts: experts,
            weight: packed([experts, output, input * bits / 32], seed: seed),
            scales: s, biases: b, groupSize: 64, bits: bits, mode: .affine)
    }

    static func linear(input: Int, output: Int, bits: Int, seed: UInt64) -> QuantizedLinear {
        let (s, b) = scalesBiases([output, input / 64], bits: bits, seed: seed)
        return QuantizedLinear(
            weight: packed([output, input * bits / 32], seed: seed), scales: s, biases: b,
            groupSize: 64, bits: bits, mode: .affine)
    }

    /// A V4.1 MoE block at `layer` with random weights at the bundle's bit plan (gate/up 3-bit on
    /// layers 0-7, 2-bit after; down 3-bit; shared expert 8-bit; router bf16, bias f32).
    static func moe(
        _ c: DeepseekV4Configuration, layer: Int, gateUpBits: Int? = nil, seed: UInt64 = 1
    ) throws -> DeepseekV4MoE {
        let m = DeepseekV4MoE(config: c, layerIdx: layer)
        let gu = gateUpBits ?? (layer < 8 ? 3 : 2)
        let D = c.hiddenSize, I = c.moeIntermediateSize, E = c.nRoutedExperts
        try m.switchMLP.update(
            modules: ModuleChildren.unflattened([
                ("gate_proj", switchLinear(experts: E, input: D, output: I, bits: gu, seed: seed &+ 10) as Module),
                ("up_proj", switchLinear(experts: E, input: D, output: I, bits: gu, seed: seed &+ 20) as Module),
                ("down_proj", switchLinear(experts: E, input: I, output: D, bits: 3, seed: seed &+ 30) as Module),
            ]), verify: .all)
        try m.sharedExperts.update(
            modules: ModuleChildren.unflattened([
                ("gate_proj", linear(input: D, output: I, bits: 8, seed: seed &+ 40) as Module),
                ("up_proj", linear(input: D, output: I, bits: 8, seed: seed &+ 50) as Module),
                ("down_proj", linear(input: I, output: D, bits: 8, seed: seed &+ 60) as Module),
            ]), verify: .all)
        m.gate.update(parameters: ModuleParameters.unflattened([
            "weight": (MLXRandom.normal([E, D], key: MLXRandom.key(seed &+ 70)) * 0.03).asType(.bfloat16),
            "bias": MLXRandom.normal([E], key: MLXRandom.key(seed &+ 80)) * 0.05,
        ]))
        eval(m)
        return m
    }

    /// Routed banks shared by every synthetic layer (one 4.8 GB bank per bit plan, not one per layer).
    struct Banks {
        let gate: QuantizedSwitchLinear, up: QuantizedSwitchLinear, down: QuantizedSwitchLinear
    }

    static func banks(_ c: DeepseekV4Configuration, gateUpBits: Int, seed: UInt64) -> Banks {
        let D = c.hiddenSize, I = c.moeIntermediateSize, E = c.nRoutedExperts
        let b = Banks(
            gate: switchLinear(experts: E, input: D, output: I, bits: gateUpBits, seed: seed &+ 10),
            up: switchLinear(experts: E, input: D, output: I, bits: gateUpBits, seed: seed &+ 20),
            down: switchLinear(experts: E, input: I, output: D, bits: 3, seed: seed &+ 30))
        eval(b.gate, b.up, b.down)
        return b
    }

    /// A V4.1 decoder layer with random weights in the bundle's formats: attention, shared expert and
    /// indexer wq_b 8-bit affine; compressor, indexer projections, norms bf16; hc / sink / gate bias fp32.
    static func decoderLayer(
        _ c: DeepseekV4Configuration, index: Int, banks: Banks, seed: UInt64
    ) throws -> DeepseekV4DecoderLayer {
        MLXRandom.seed(seed)
        let layer = DeepseekV4DecoderLayer(config: c, layerIdx: index)
        try layer.mlp.switchMLP.update(
            modules: ModuleChildren.unflattened([
                ("gate_proj", QuantizedSwitchLinear(
                    inputDims: c.hiddenSize, outputDims: c.moeIntermediateSize, numExperts: c.nRoutedExperts,
                    weight: banks.gate.packedWeight, scales: banks.gate.groupScales,
                    biases: banks.gate.groupBiases, groupSize: 64, bits: banks.gate.bits) as Module),
                ("up_proj", QuantizedSwitchLinear(
                    inputDims: c.hiddenSize, outputDims: c.moeIntermediateSize, numExperts: c.nRoutedExperts,
                    weight: banks.up.packedWeight, scales: banks.up.groupScales,
                    biases: banks.up.groupBiases, groupSize: 64, bits: banks.up.bits) as Module),
                ("down_proj", QuantizedSwitchLinear(
                    inputDims: c.moeIntermediateSize, outputDims: c.hiddenSize, numExperts: c.nRoutedExperts,
                    weight: banks.down.packedWeight, scales: banks.down.groupScales,
                    biases: banks.down.groupBiases, groupSize: 64, bits: 3) as Module),
            ]), verify: .all)
        // fp parameters to the bundle dtypes before quantizing the 8-bit projections.
        var fp: [(String, MLXArray)] = []
        for (path, value) in layer.parameters().flattened() where value.dtype == .float32 {
            let keepF32 = path.contains("_hc.") || path.hasSuffix("attn_sink") || path.hasSuffix("gate.bias")
            if path.hasSuffix("_hc.fn") {
                fp.append((path, MLXRandom.normal(value.shape, key: MLXRandom.key(seed &+ UInt64(path.count))) * 0.02))
            } else if path.hasSuffix("gate.weight") {
                fp.append((path, (MLXRandom.normal(value.shape) * 0.03).asType(.bfloat16)))
            } else if !keepF32 {
                fp.append((path, value.asType(.bfloat16)))
            }
        }
        layer.update(parameters: ModuleParameters.unflattened(fp))
        quantize(model: layer, groupSize: 64, bits: 8) { path, module in
            guard module is Linear, !(module is QuantizedLinear) else { return false }
            if path.contains("compressor") { return false }
            if path.contains("indexer") { return path.hasSuffix("wq_b") }
            return true
        }
        eval(layer)
        return layer
    }

    static func hidden(rows: Int, _ c: DeepseekV4Configuration, seed: UInt64) -> MLXArray {
        let x = MLXRandom.normal([1, rows, c.hiddenSize], key: MLXRandom.key(seed))
        let out = x.asType(.bfloat16)
        eval(out)
        return out
    }

    /// Median wall time (ms) of building and evaluating `body` (3 warm runs first).
    static func timeMs(reps: Int = 25, _ body: () -> [MLXArray]) -> Double {
        for _ in 0 ..< 3 { eval(body()) }
        var times: [Double] = []
        for _ in 0 ..< reps {
            let t0 = CFAbsoluteTimeGetCurrent()
            eval(body())
            times.append((CFAbsoluteTimeGetCurrent() - t0) * 1_000)
        }
        times.sort()
        return times[times.count / 2]
    }

    /// Interleaved A/B: alternate the bodies rep by rep and keep each one's minimum (robust to other GPU
    /// clients and clock drift).
    static func abMinMs(reps: Int = 30, _ bodies: [() -> [MLXArray]]) -> [Double] {
        for b in bodies { for _ in 0 ..< 2 { eval(b()) } }
        var best = [Double](repeating: .infinity, count: bodies.count)
        for _ in 0 ..< reps {
            for (k, b) in bodies.enumerated() {
                let t0 = CFAbsoluteTimeGetCurrent()
                eval(b())
                best[k] = min(best[k], (CFAbsoluteTimeGetCurrent() - t0) * 1_000)
            }
        }
        return best
    }

    static func bitEqual(_ a: MLXArray, _ b: MLXArray) -> Bool {
        guard a.shape == b.shape, a.dtype == b.dtype else { return false }
        return arrayEqual(a, b).item(Bool.self)
    }

    static func maxAbsDiff(_ a: MLXArray, _ b: MLXArray) -> Float {
        MLX.abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
    }
}

@Suite(
    "DeepSeek-V4.1 decode fusion", .serialized,
    .enabled(if: DSV41Synth.config != nil, "needs VMLX_DSV41_CONFIG or VMLX_DSV41_BUNDLE"))
struct DeepseekV41DecodeFusionFocusedTests {

    // MARK: - Router GEMV (bf16 rows, fp32 math)

    static func stockLogits(_ x: MLXArray, _ w: MLXArray) -> MLXArray {
        x.asType(.float32).matmul(w.asType(.float32).transposed())
    }

    @Test("router GEMV is bit-identical to the stock fp32 cast + matmul")
    func routerBitExact() throws {
        try Device.withDefaultDevice(.gpu) {
            // V4.1 (384 x 5120) plus other admitted shapes; bf16 and fp32 activations.
            for (E, D) in [(384, 5120), (288, 4096), (192, 2048), (384, 3072)] {
                for seed in UInt64(1) ... 4 {
                    let w = (MLXRandom.normal([E, D], key: MLXRandom.key(seed)) * 0.03).asType(.bfloat16)
                    for dtype in [DType.bfloat16, .float32] {
                        let x = (MLXRandom.normal([1, 1, D], key: MLXRandom.key(seed &+ 9)) * 3).asType(dtype)
                        let got = try #require(DeepseekV41RouterGEMV.logits(x, weight: w))
                        let ref = Self.stockLogits(x, w)
                        #expect(got.shape == ref.shape && got.dtype == .float32)
                        #expect(DSV41Synth.bitEqual(got, ref), "E\(E) D\(D) seed \(seed) \(dtype)")
                    }
                }
            }
        }
    }

    @Test("router GEMV declines verify rows, other dtypes, odd shapes and the CPU")
    func routerDeclines() {
        Device.withDefaultDevice(.gpu) {
            let w = MLXArray.zeros([384, 5120], dtype: .bfloat16)
            #expect(DeepseekV41RouterGEMV.logits(MLXArray.zeros([1, 2, 5120], dtype: .bfloat16), weight: w) == nil)
            #expect(DeepseekV41RouterGEMV.logits(MLXArray.zeros([1, 1, 5120], dtype: .bfloat16), weight: w.asType(.float32)) == nil)
            #expect(DeepseekV41RouterGEMV.logits(
                MLXArray.zeros([1, 1, 5000], dtype: .bfloat16), weight: MLXArray.zeros([384, 5000], dtype: .bfloat16)) == nil)
            // K >= 16 * out: MLX picks a different GEMV tiling there.
            #expect(DeepseekV41RouterGEMV.logits(
                MLXArray.zeros([1, 1, 8192], dtype: .bfloat16), weight: MLXArray.zeros([256, 8192], dtype: .bfloat16)) == nil)
        }
        Device.withDefaultDevice(.cpu) {
            #expect(DeepseekV41RouterGEMV.logits(
                MLXArray.zeros([1, 1, 5120], dtype: .bfloat16), weight: MLXArray.zeros([384, 5120], dtype: .bfloat16)) == nil)
        }
    }

    @Test("V4.1 gate selects the same experts and weights with the router kernel on and off")
    func gateParity() throws {
        guard let c = DSV41Synth.config else { return }
        try Device.withDefaultDevice(.gpu) {
            let moe = try DSV41Synth.moe(c, layer: 10)
            let saved = DeepseekV41RouterGEMV.enabled
            defer { DeepseekV41RouterGEMV.enabled = saved }
            for seed in UInt64(1) ... 8 {
                let x = DSV41Synth.hidden(rows: 1, c, seed: seed)
                DeepseekV41RouterGEMV.enabled = true
                let (i1, w1) = moe.gate(x, inputIds: nil)
                DeepseekV41RouterGEMV.enabled = false
                let (i0, w0) = moe.gate(x, inputIds: nil)
                #expect(DSV41Synth.bitEqual(i1, i0) && DSV41Synth.bitEqual(w1, w0), "seed \(seed)")
            }
        }
    }

    @Test("router GEMV microbenchmark (VMLX_DSV41_DEC_BENCH=1)")
    func routerBench() throws {
        guard DSV41Synth.benchEnabled, let c = DSV41Synth.config else { return }
        try Device.withDefaultDevice(.gpu) {
            let layers = 40
            let ws = (0 ..< layers).map {
                (MLXRandom.normal([c.nRoutedExperts, c.hiddenSize], key: MLXRandom.key(UInt64($0))) * 0.03)
                    .asType(.bfloat16)
            }
            let xs = (0 ..< layers).map { DSV41Synth.hidden(rows: 1, c, seed: UInt64(500 + $0)) }
            eval(ws)
            let stock = DSV41Synth.timeMs { zip(xs, ws).map { Self.stockLogits($0, $1) } } / Double(layers)
            let fused = DSV41Synth.timeMs {
                zip(xs, ws).map { DeepseekV41RouterGEMV.logits($0, weight: $1)! }
            } / Double(layers)
            print(String(format: "[dsv41-dec-bench] router logits per layer: stock %.4f ms, kernel %.4f ms", stock, fused))
        }
    }

    // MARK: - Fused MoE (gate/up + SwiGLU, down + sum + shared)

    /// The stock routed + shared graph (what `DeepseekV4MoE.experts` runs without the kernels).
    static func stockExperts(_ moe: DeepseekV4MoE, _ x: MLXArray, _ idx: MLXArray, _ sc: MLXArray) -> MLXArray {
        let routed = (moe.switchMLP as! SwitchGLU)(x, idx, preDownScores: sc)
        return DeepseekV4Math.addSharedExpertFP32(
            DeepseekV4Math.reduceRoutedExpertsFP32(routed), shared: moe.sharedExperts(x), outputDType: x.dtype)
    }

    @Test("fused MoE kernels are bit-identical to the stock graph at 1-8 rows (2- and 3-bit gate/up)")
    func fusedMoEBitExact() throws {
        guard let c = DSV41Synth.config else { return }
        try Device.withDefaultDevice(.gpu) {
            for layer in [3, 10] {
                let moe = try DSV41Synth.moe(c, layer: layer, seed: UInt64(layer))
                let l = try #require(DeepseekV41FusedMoE.layer(moe.switchMLP as! SwitchGLU, shared: moe.sharedExperts))
                for rows in 1 ... 8 {
                    for seed in UInt64(1) ... 3 {
                        let x = DSV41Synth.hidden(rows: rows, c, seed: seed &* 31 &+ UInt64(rows))
                        let (idx, sc) = moe.gate(x, inputIds: nil)
                        let ref = Self.stockExperts(moe, x, idx, sc)
                        let wide = rows >= 2 && DeepseekV41FusedMoE.architecture.gen >= 15
                        let got = DeepseekV41FusedMoE.run(
                            x, indices: idx, scores: sc, layer: l, limit: c.swigluLimit, wide: wide)
                        eval(got, ref)
                        let peak = MLX.abs(ref.asType(.float32)).max().item(Float.self)
                        #expect(peak.isFinite && peak > 1e-3, "non-degenerate reference (peak \(peak))")
                        let same = DSV41Synth.bitEqual(got, ref)
                        #expect(same, "layer \(layer) rows \(rows) seed \(seed): max |d| \(DSV41Synth.maxAbsDiff(got, ref))")
                    }
                }
            }
        }
    }

    @Test("MoE forward with the fused kernels on equals the forward with them off (admission included)")
    func fusedMoEForwardParity() throws {
        guard let c = DSV41Synth.config else { return }
        try Device.withDefaultDevice(.gpu) {
            // Two identical modules: each compiled decode region bakes in the kernel choice active at
            // its first trace, so one module only ever runs with the kernels on, the other off.
            let moeOn = try DSV41Synth.moe(c, layer: 12, seed: 99)
            let moeOff = try DSV41Synth.moe(c, layer: 12, seed: 99)
            let saved = (DeepseekV41FusedMoE.enabled, DeepseekV41FusedMoE.maxRows)
            defer { DeepseekV41FusedMoE.enabled = saved.0; DeepseekV41FusedMoE.maxRows = saved.1 }
            DeepseekV41FusedMoE.maxRows = DeepseekV41FusedMoE.maxSupportedRows
            DeepseekV41FusedMoE.resetVerdicts()
            // A prefill chunk first, as in a real request: it warms the decode region, whose first trace
            // must still wait for the single-row admission (else the stock graph is baked in for good).
            let prefill = DSV41Synth.hidden(rows: 16, c, seed: 1600)
            eval(moeOn(prefill, inputIds: nil), moeOff(prefill, inputIds: nil))
            // Repeated single rows: eager admission first, then the compiled region's trace and replays.
            for (step, rows) in [1, 1, 1, 2, 4, 6, 8, 1, 1].enumerated() {
                let x = DSV41Synth.hidden(rows: rows, c, seed: UInt64(700 + step))
                DeepseekV41FusedMoE.enabled = true
                let on = moeOn(x, inputIds: nil)
                eval(on)
                DeepseekV41FusedMoE.enabled = false
                let off = moeOff(x, inputIds: nil)
                eval(off)
                #expect(DSV41Synth.bitEqual(on, off), "step \(step) rows \(rows)")
            }
            DeepseekV41FusedMoE.enabled = true
            for rows in [1, 2, 4, 6, 8] {
                let wide = rows >= 2 && DeepseekV41FusedMoE.architecture.gen >= 15
                let key = DeepseekV41FusedMoE.Key(
                    rows: rows, hidden: c.hiddenSize, inter: c.moeIntermediateSize, topK: c.numExpertsPerTok,
                    gateBits: 2, downBits: 3, sharedBits: 8, wide: wide)
                #expect(DeepseekV41FusedMoE.verdict(key) == true, "rows \(rows) admitted")
            }
            // 9+ rows (prefill) never take the kernels.
            let x9 = DSV41Synth.hidden(rows: 9, c, seed: 9)
            let (i9, s9) = moeOn.gate(x9, inputIds: nil)
            #expect(DeepseekV41FusedMoE.apply(
                x9, indices: i9, scores: s9, switchMLP: moeOn.switchMLP as! SwitchGLU, shared: moeOn.sharedExperts,
                limit: c.swigluLimit, stock: { Self.stockExperts(moeOn, x9, i9, s9) }) == nil)
        }
    }

    /// Verify-block routing with shared experts: row r takes `shared` experts from a common pool (the
    /// same ids for every row, in rotated slot order) and the rest of its 6 from a row-private range.
    static func overlappingRoutes(rows: Int, topK: Int, shared: Int, experts: Int, seed: UInt64)
        -> (MLXArray, MLXArray)
    {
        var ids: [UInt32] = []
        for r in 0 ..< rows {
            var row: [UInt32] = (0 ..< shared).map { UInt32((7 + $0 * 13) % experts) }
            row += (0 ..< (topK - shared)).map { UInt32((200 + r * topK + $0) % experts) }
            let k = r % topK
            ids += Array(row[k...] + row[..<k])
        }
        let scores = MLXRandom.uniform(low: 0.1, high: 0.5, [1, rows, topK], key: MLXRandom.key(seed))
        return (MLXArray(ids, [1, rows, topK]), scores)
    }

    @Test("fused MoE stays bit-identical when verify rows share experts")
    func fusedMoEOverlapBitExact() throws {
        guard let c = DSV41Synth.config else { return }
        try Device.withDefaultDevice(.gpu) {
            let moe = try DSV41Synth.moe(c, layer: 10, seed: 5)
            let l = try #require(DeepseekV41FusedMoE.layer(moe.switchMLP as! SwitchGLU, shared: moe.sharedExperts))
            for rows in [2, 4, 8] {
                for sharedCount in [0, 2, 4, 6] {
                    let x = DSV41Synth.hidden(rows: rows, c, seed: UInt64(40 + rows))
                    let (idx, sc) = Self.overlappingRoutes(
                        rows: rows, topK: c.numExpertsPerTok, shared: sharedCount, experts: c.nRoutedExperts, seed: 3)
                    let ref = Self.stockExperts(moe, x, idx, sc)
                    let got = DeepseekV41FusedMoE.run(
                        x, indices: idx, scores: sc, layer: l, limit: c.swigluLimit,
                        wide: DeepseekV41FusedMoE.architecture.gen >= 15)
                    eval(got, ref)
                    #expect(DSV41Synth.bitEqual(got, ref), "rows \(rows) shared \(sharedCount)")
                }
            }
        }
    }

    @Test("fused MoE vs stock when verify rows share experts (VMLX_DSV41_DEC_BENCH=1)")
    func fusedMoEOverlapBench() throws {
        guard DSV41Synth.benchEnabled, let c = DSV41Synth.config else { return }
        try Device.withDefaultDevice(.gpu) {
            let moe = try DSV41Synth.moe(c, layer: 10)
            let l = try #require(DeepseekV41FusedMoE.layer(moe.switchMLP as! SwitchGLU, shared: moe.sharedExperts))
            let layers = 40
            for rows in [4, 8] {
                for sharedCount in [0, 2, 4] {
                    let xs = (0 ..< layers).map { DSV41Synth.hidden(rows: rows, c, seed: UInt64(1900 + $0)) }
                    // Different experts per layer so no layer's weights are cache-hot from the previous one.
                    let routes = (0 ..< layers).map { k -> (MLXArray, MLXArray) in
                        let r = Self.overlappingRoutes(rows: rows, topK: c.numExpertsPerTok, shared: sharedCount,
                            experts: c.nRoutedExperts, seed: UInt64(k))
                        return ((r.0 + UInt32(k * 11)) % UInt32(c.nRoutedExperts), r.1)
                    }
                    eval(routes.flatMap { [$0.0, $0.1] })
                    let t = DSV41Synth.abMinMs([
                        { zip(xs, routes).map { Self.stockExperts(moe, $0, $1.0, $1.1) } },
                        { zip(xs, routes).map {
                            DeepseekV41FusedMoE.run($0, indices: $1.0, scores: $1.1, layer: l, limit: c.swigluLimit, wide: true)
                        } },
                    ]).map { $0 / Double(layers) }
                    print(String(format: "[dsv41-dec-bench] experts rows=%d shared=%d/6: stock %.4f, fused %.4f ms/layer",
                        rows, sharedCount, t[0], t[1]))
                }
            }
        }
    }

    @Test("fused MoE microbenchmark at 1-8 rows (VMLX_DSV41_DEC_BENCH=1)")
    func fusedMoEBench() throws {
        guard DSV41Synth.benchEnabled, let c = DSV41Synth.config else { return }
        try Device.withDefaultDevice(.gpu) {
            let moe = try DSV41Synth.moe(c, layer: 10)
            let l = try #require(DeepseekV41FusedMoE.layer(moe.switchMLP as! SwitchGLU, shared: moe.sharedExperts))
            let layers = 40
            for rows in [1, 2, 4, 6, 8] {
                let wide = rows >= 2 && DeepseekV41FusedMoE.architecture.gen >= 15
                let xs = (0 ..< layers).map { DSV41Synth.hidden(rows: rows, c, seed: UInt64(900 + $0)) }
                let routes = xs.map { moe.gate($0, inputIds: nil) }
                eval(routes.flatMap { [$0.0, $0.1] })
                let t = DSV41Synth.abMinMs([
                    { zip(xs, routes).map { Self.stockExperts(moe, $0, $1.0, $1.1) } },
                    { zip(xs, routes).map {
                        DeepseekV41FusedMoE.run($0, indices: $1.0, scores: $1.1, layer: l, limit: c.swigluLimit, wide: wide)
                    } },
                ]).map { $0 / Double(layers) }
                let (stock, fused) = (t[0], t[1])
                print(String(format: "[dsv41-dec-bench] experts per layer rows=%d: stock %.4f ms, fused %.4f ms (%.2fx)",
                    rows, stock, fused, stock / fused))
            }
        }
    }

    // MARK: - Fused hc stage (staggered mHC, one token)

    static func hcModules(_ c: DeepseekV4Configuration, seed: UInt64) -> (DeepseekV4HyperConnection, RMSNorm) {
        let hc = DeepseekV4HyperConnection(config: c)
        let mix = (2 + c.hcMult) * c.hcMult
        hc.update(parameters: ModuleParameters.unflattened([
            "fn": MLXRandom.normal([mix, c.hcMult * c.hiddenSize], key: MLXRandom.key(seed)) * 0.02,
            "scale": MLXRandom.uniform(low: 0.5, high: 2.0, [3], key: MLXRandom.key(seed &+ 1)),
            "base": MLXRandom.normal([mix], key: MLXRandom.key(seed &+ 2)) * 0.5,
        ]))
        let norm = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
        norm.update(parameters: ModuleParameters.unflattened([
            "weight": (MLXRandom.uniform(low: 0.5, high: 1.5, [c.hiddenSize], key: MLXRandom.key(seed &+ 3)))
                .asType(.bfloat16)
        ]))
        eval(hc, norm)
        return (hc, norm)
    }

    static func stageInputs(_ c: DeepseekV4Configuration, seed: UInt64)
        -> (h: MLXArray, y: MLXArray, cpre: MLXArray, post: MLXArray, comb: MLXArray)
    {
        let h = (MLXRandom.normal([1, 1, c.hcMult, c.hiddenSize], key: MLXRandom.key(seed)) * 4).asType(.bfloat16)
        let y = (MLXRandom.normal([1, 1, c.hiddenSize], key: MLXRandom.key(seed &+ 1)) * 2).asType(.bfloat16)
        let cpre = MLXRandom.uniform(low: 0.0, high: 1.0, [1, 1, c.hcMult], key: MLXRandom.key(seed &+ 2))
        let post = MLXRandom.uniform(low: 0.0, high: 2.0, [1, 1, c.hcMult], key: MLXRandom.key(seed &+ 3))
        let comb = softmax(MLXRandom.normal([1, 1, c.hcMult, c.hcMult], key: MLXRandom.key(seed &+ 4)), axis: -1)
        eval(h, y, cpre, post, comb)
        return (h, y, cpre, post, comb)
    }

    @Test("fused hc stage is bit-identical to the stock staggered-mHC graph (with and without the expand)")
    func hcStageBitExact() throws {
        guard let c = DSV41Synth.config else { return }
        try Device.withDefaultDevice(.gpu) {
            for seed in UInt64(1) ... 6 {
                let (hc, norm) = Self.hcModules(c, seed: seed &* 13)
                let i = Self.stageInputs(c, seed: seed &* 7)
                #expect(DeepseekV41HCStage.admits(i.h, hc: hc, norm: norm))
                for expand in [false, true] {
                    let args: (MLXArray?, MLXArray?, MLXArray?) = expand ? (i.y, i.post, i.comb) : (nil, nil, nil)
                    let got = DeepseekV41HCStage.run(
                        h: i.h, y: args.0, post: args.1, comb: args.2, cpre: i.cpre, hc: hc, norm: norm)
                    let ref = DeepseekV41HCStage.stock(
                        h: i.h, y: args.0, post: args.1, comb: args.2, cpre: i.cpre, hc: hc, norm: norm)
                    let names = ["normed", "pre", "post", "comb", "stream"]
                    let pairs = [(got.normed, ref.normed), (got.pre, ref.pre), (got.post, ref.post),
                                 (got.comb, ref.comb)] + (expand ? [(got.stream!, ref.stream!)] : [])
                    for (k, (a, b)) in pairs.enumerated() {
                        eval(a, b)
                        #expect(DSV41Synth.bitEqual(a, b),
                            "seed \(seed) expand \(expand) \(names[k]): \(a.shape) vs \(b.shape), max |d| \(a.shape == b.shape ? DSV41Synth.maxAbsDiff(a, b) : -1)")
                    }
                }
            }
        }
    }

    @Test("hc stage declines verify rows, other dtypes and the CPU")
    func hcStageDeclines() throws {
        guard let c = DSV41Synth.config else { return }
        try Device.withDefaultDevice(.gpu) {
            let (hc, norm) = Self.hcModules(c, seed: 3)
            let h2 = MLXArray.zeros([1, 2, c.hcMult, c.hiddenSize], dtype: .bfloat16)
            #expect(!DeepseekV41HCStage.admits(h2, hc: hc, norm: norm))
            #expect(!DeepseekV41HCStage.admits(h2[0..., 0 ..< 1].asType(.float32), hc: hc, norm: norm))
        }
        Device.withDefaultDevice(.cpu) {
            let (hc, norm) = Self.hcModules(c, seed: 3)
            #expect(!DeepseekV41HCStage.admits(
                MLXArray.zeros([1, 1, c.hcMult, c.hiddenSize], dtype: .bfloat16), hc: hc, norm: norm))
        }
    }

    @Test("hc stage microbenchmark (VMLX_DSV41_DEC_BENCH=1)")
    func hcStageBench() throws {
        guard DSV41Synth.benchEnabled, let c = DSV41Synth.config else { return }
        try Device.withDefaultDevice(.gpu) {
            let layers = 40
            let mods = (0 ..< 8).map { Self.hcModules(c, seed: UInt64(40 + $0)) }
            let ins = (0 ..< layers).map { Self.stageInputs(c, seed: UInt64(300 + $0)) }
            for expand in [false, true] {
                func body(_ fused: Bool) -> [MLXArray] {
                    ins.enumerated().flatMap { k, i -> [MLXArray] in
                        let (hc, norm) = mods[k % mods.count]
                        let a: (MLXArray?, MLXArray?, MLXArray?) = expand ? (i.y, i.post, i.comb) : (nil, nil, nil)
                        let o = fused
                            ? DeepseekV41HCStage.run(h: i.h, y: a.0, post: a.1, comb: a.2, cpre: i.cpre, hc: hc, norm: norm)
                            : DeepseekV41HCStage.stock(h: i.h, y: a.0, post: a.1, comb: a.2, cpre: i.cpre, hc: hc, norm: norm)
                        return [o.normed, o.pre, o.post, o.comb] + (o.stream.map { [$0] } ?? [])
                    }
                }
                let stock = DSV41Synth.timeMs { body(false) } / Double(layers)
                let fused = DSV41Synth.timeMs { body(true) } / Double(layers)
                print(String(format: "[dsv41-dec-bench] hc stage%@ per sublayer: stock %.4f ms, fused %.4f ms (%.2fx)",
                    expand ? " + expand" : "", stock, fused, stock / fused))
            }
        }
    }

    // MARK: - Whole-layer decode (synthetic V4.1 layers with caches)

    /// Synthetic layers 8-11 (ratio 2: kv / index source + readers) and 20-24 (ratio 1: kv / index /
    /// candidate source + readers), `context` tokens of prefill, then single-token decode steps.
    final class LayerStack {
        let c: DeepseekV4Configuration
        let layers: [DeepseekV4DecoderLayer]
        let caches: [KVCache]
        let indices: [Int]

        init(_ c: DeepseekV4Configuration, indices: [Int] = [8, 9, 10, 11, 20, 21, 22, 23, 24], context: Int) throws {
            self.c = c
            self.indices = indices
            let b2 = DSV41Synth.banks(c, gateUpBits: 2, seed: 77)
            layers = try indices.map { try DSV41Synth.decoderLayer(c, index: $0, banks: b2, seed: UInt64(1000 + $0)) }
            let all = DeepseekV4Model.makeCaches(config: c)
            caches = indices.map { all[$0] }
            var done = 0
            while done < context {
                let n = min(512, context - done)
                let h = MLXRandom.normal([1, n, c.hcMult, c.hiddenSize], key: MLXRandom.key(UInt64(done))).asType(.bfloat16)
                eval(forward(h))
                done += n
            }
        }

        func forward(_ hIn: MLXArray) -> MLXArray {
            var h = hIn
            let L = h.dim(1)
            var preMix = concatenated(
                [MLXArray.ones([1, L, 1]), MLXArray.zeros([1, L, c.hcMult - 1])], axis: -1).asType(.float32)
            let mask = createAttentionMask(h: h.reshaped(1, L, -1), cache: caches.first)
            let share = DSV41CompressShare()
            for (i, (layer, cache)) in zip(layers, caches).enumerated() {
                let r = layer.callStaggered(h, mask: mask, cache: cache, inputIds: nil, preMix: preMix, compressShare: share)
                h = r.h
                preMix = r.ffnPre
                // The model loop's per-layer submit (DeepseekV4ModelInner), same statement.
                if DeepseekV4ModelInner.layerSubmit, L <= DeepseekV4ModelInner.layerSubmitMaxRows,
                    i < layers.count - 1
                {
                    asyncEval(h, preMix)
                }
            }
            return h + preMix.sum().asType(h.dtype) * 0
        }

        /// Median ms per layer of one decode step (graph build + eval), over `steps` steps.
        func decodeMsPerLayer(steps: Int = 24) -> Double { decodeTiming(steps: steps).total }

        /// Median per-layer (total, host graph-build) ms of one decode step.
        func decodeTiming(steps: Int = 24) -> (total: Double, build: Double) {
            var times: [Double] = [], builds: [Double] = []
            for s in 0 ..< steps + 3 {
                let h = MLXRandom.normal([1, 1, c.hcMult, c.hiddenSize], key: MLXRandom.key(UInt64(50_000 + s)))
                    .asType(.bfloat16)
                eval(h)
                let t0 = CFAbsoluteTimeGetCurrent()
                let out = forward(h)
                let t1 = CFAbsoluteTimeGetCurrent()
                eval(out)
                if s >= 3 {
                    times.append((CFAbsoluteTimeGetCurrent() - t0) * 1_000)
                    builds.append((t1 - t0) * 1_000)
                }
            }
            times.sort(); builds.sort()
            let n = Double(layers.count)
            return (times[times.count / 2] / n, builds[builds.count / 2] / n)
        }
    }

    @Test("decode and verify through synthetic V4.1 layers: all kernels on == all kernels off, bitwise")
    func layerStackParity() throws {
        guard let c = DSV41Synth.config else { return }
        try Device.withDefaultDevice(.gpu) {
            let flags = (DeepseekV41RouterGEMV.enabled, DeepseekV41FusedMoE.enabled, DeepseekV41HCStage.enabled)
            defer {
                DeepseekV41RouterGEMV.enabled = flags.0; DeepseekV41FusedMoE.enabled = flags.1
                DeepseekV41HCStage.enabled = flags.2
            }
            func set(_ on: Bool) {
                DeepseekV41RouterGEMV.enabled = on; DeepseekV41FusedMoE.enabled = on; DeepseekV41HCStage.enabled = on
            }
            let savedRows = DeepseekV41FusedMoE.maxRows
            defer { DeepseekV41FusedMoE.maxRows = savedRows }
            DeepseekV41FusedMoE.maxRows = DeepseekV41FusedMoE.maxSupportedRows  // exercise every verify size
            DeepseekV41FusedMoE.resetVerdicts(); DeepseekV41HCStage.resetVerdicts()
            let idx = [8, 9, 20, 21, 24]
            set(true); let on = try LayerStack(c, indices: idx, context: 700)
            set(false); let off = try LayerStack(c, indices: idx, context: 700)
            for (step, rows) in [1, 1, 1, 1, 4, 1, 1, 7, 1, 1].enumerated() {
                let h = MLXRandom.normal([1, rows, c.hcMult, c.hiddenSize], key: MLXRandom.key(UInt64(9000 + step)))
                    .asType(.bfloat16)
                set(true); let a = on.forward(h); eval(a)
                set(false); let b = off.forward(h); eval(b)
                #expect(DSV41Synth.bitEqual(a, b), "step \(step) rows \(rows): max |d| \(DSV41Synth.maxAbsDiff(a, b))")
            }
            set(true)
            // Prefill runs before decode, so the decode MoE region's first trace must wait for admission.
            for rows in [1, 4, 7] {
                let key = DeepseekV41FusedMoE.Key(
                    rows: rows, hidden: c.hiddenSize, inter: c.moeIntermediateSize, topK: c.numExpertsPerTok,
                    gateBits: 2, downBits: 3, sharedBits: 8,
                    wide: rows >= 2 && DeepseekV41FusedMoE.architecture.gen >= 15)
                #expect(DeepseekV41FusedMoE.verdict(key) == true, "fused MoE admitted at rows \(rows)")
            }
            #expect(DeepseekV41HCStage.verdict(.init(hc: c.hcMult, hidden: c.hiddenSize, expand: false)) == true)
            #expect(DeepseekV41HCStage.verdict(.init(hc: c.hcMult, hidden: c.hiddenSize, expand: true)) == true)
        }
    }

    /// Writes the lazy graph of one decode step per layer (`mlx_print_graph`) to
    /// $VMLX_DSV41_GRAPH_DIR/layer<k>.txt for a dispatch census.
    @Test("decode graph census (VMLX_DSV41_GRAPH_DIR)")
    func graphCensus() throws {
        guard let dir = ProcessInfo.processInfo.environment["VMLX_DSV41_GRAPH_DIR"], let c = DSV41Synth.config
        else { return }
        try Device.withDefaultDevice(.gpu) {
            let stack = try LayerStack(c, context: 2048)
            for _ in 0 ..< 4 { _ = stack.decodeMsPerLayer(steps: 1) }
            var h = MLXRandom.normal([1, 1, c.hcMult, c.hiddenSize], key: MLXRandom.key(5)).asType(.bfloat16)
            var preMix = concatenated([MLXArray.ones([1, 1, 1]), MLXArray.zeros([1, 1, c.hcMult - 1])], axis: -1)
                .asType(.float32)
            eval(h, preMix)
            let share = DSV41CompressShare()
            for (k, (layer, cache)) in zip(stack.layers, stack.caches).enumerated() {
                let r = layer.callStaggered(h, mask: .none, cache: cache, inputIds: nil, preMix: preMix, compressShare: share)
                let path = "\(dir)/layer\(stack.indices[k]).txt"
                if let f = fopen(path, "w") {
                    let v = mlx_vector_array_new()
                    mlx_vector_array_append_value(v, r.h.ctx)
                    mlx_vector_array_append_value(v, r.ffnPre.ctx)
                    let namer = mlx_node_namer_new()
                    mlx_print_graph(f, namer, v)
                    mlx_node_namer_free(namer)
                    mlx_vector_array_free(v)
                    fclose(f)
                }
                eval(r.h, r.ffnPre)
                h = r.h; preMix = r.ffnPre
            }
        }
    }

    @Test("host graph-build profile of one decode layer (VMLX_DSV41_DEC_BENCH=1)")
    func hostProfile() throws {
        guard DSV41Synth.benchEnabled, let c = DSV41Synth.config else { return }
        try Device.withDefaultDevice(.gpu) {
            let stack = try LayerStack(c, context: 2048)
            var acc: [String: [Double]] = [:]
            for s in 0 ..< 30 {
                var h = MLXRandom.normal([1, 1, c.hcMult, c.hiddenSize], key: MLXRandom.key(UInt64(70_000 + s)))
                    .asType(.bfloat16)
                var preMix = concatenated([MLXArray.ones([1, 1, 1]), MLXArray.zeros([1, 1, c.hcMult - 1])], axis: -1)
                    .asType(.float32)
                eval(h, preMix)
                let share = DSV41CompressShare()
                for (layer, cache) in zip(stack.layers, stack.caches) {
                    var t = CFAbsoluteTimeGetCurrent()
                    func lap(_ k: String) {
                        let n = CFAbsoluteTimeGetCurrent()
                        if s >= 5 { acc[k, default: []].append((n - t) * 1000) }
                        t = n
                    }
                    let a = DeepseekV41HCStage.apply(h: h, cpre: preMix, hc: layer.attnHC, norm: layer.inputLayerNorm)!
                    lap("attn_hc")
                    let attnOut = layer.selfAttn(a.normed, mask: .none, cache: cache, compressShare: share)
                    lap("attention")
                    let f = DeepseekV41HCStage.apply(
                        h: h, y: attnOut, post: a.post, comb: a.comb, cpre: a.pre, hc: layer.ffnHC,
                        norm: layer.postAttentionLayerNorm)!
                    lap("ffn_hc")
                    let ffnOut = layer.mlp(f.normed, inputIds: nil, imageMask: nil)
                    lap("moe")
                    h = layer.ffnHC.expand(blockOut: ffnOut, residual: f.stream!, post: f.post, comb: f.comb)
                    preMix = f.pre
                    lap("expand")
                    asyncEval(h, preMix)
                    lap("submit")
                }
                let te = CFAbsoluteTimeGetCurrent()
                eval(h)
                if s >= 5 { acc["final_eval", default: []].append((CFAbsoluteTimeGetCurrent() - te) * 1000) }
            }
            for k in ["attention", "submit", "final_eval"] {
                print("[dsv41-dec-bench] raw \(k): " + acc[k]!.prefix(40).map { String(format: "%.2f", $0) }.joined(separator: " "))
            }
            let n = Double(stack.layers.count)
            for k in ["attn_hc", "attention", "ffn_hc", "moe", "expand", "submit"] {
                let v = acc[k]!.sorted()
                print(String(format: "[dsv41-dec-bench] host build %@: median %.4f ms per layer call (sum/step/layer %.4f)",
                    k, v[v.count / 2], v.reduce(0, +) / Double(25) / n))
            }
        }
    }

    /// The hc stage and the per-layer submit are read per call, so they can be A/B'd in one process on one
    /// stack, alternating to cancel GPU-clock drift (router / MoE are baked into compiled regions).
    @Test("in-process A/B of the hc stage and the per-layer submit (VMLX_DSV41_DEC_BENCH=1)")
    func layerToggleBench() throws {
        guard DSV41Synth.benchEnabled, let c = DSV41Synth.config else { return }
        try Device.withDefaultDevice(.gpu) {
            let stack = try LayerStack(c, context: 2048)
            let saved = (DeepseekV41HCStage.enabled, DeepseekV4ModelInner.layerSubmit)
            defer { DeepseekV41HCStage.enabled = saved.0; DeepseekV4ModelInner.layerSubmit = saved.1 }
            var r: [String: [Double]] = [:]
            for _ in 0 ..< 6 {
                for (hc, sub) in [(false, false), (true, false), (false, true), (true, true)] {
                    DeepseekV41HCStage.enabled = hc; DeepseekV4ModelInner.layerSubmit = sub
                    r["hc=\(hc ? 1 : 0) submit=\(sub ? 1 : 0)", default: []].append(stack.decodeMsPerLayer(steps: 12))
                }
            }
            for k in r.keys.sorted() {
                let v = r[k]!.sorted()
                print(String(format: "[dsv41-dec-bench] A/B router=%d moe=%d %@: median %.3f ms/layer (min %.3f)",
                    DeepseekV41RouterGEMV.enabled ? 1 : 0, DeepseekV41FusedMoE.enabled ? 1 : 0, k, v[v.count / 2], v[0]))
            }
        }
    }

    /// Compiled decode regions bake the kernel choice in at their first trace, so compare configurations
    /// in separate processes: run with VMLX_DSV41_ROUTER_KERNEL / VMLX_DSV41_FUSED_MOE /
    /// VMLX_DSV41_HC_STAGE set to 0 or 1 and compare the printed lines.
    @Test("whole-layer decode microbenchmark (VMLX_DSV41_DEC_BENCH=1; flags from the environment)")
    func layerDecodeBench() throws {
        guard DSV41Synth.benchEnabled, let c = DSV41Synth.config else { return }
        try Device.withDefaultDevice(.gpu) {
            let stack = try LayerStack(c, context: 2048)
            let a = stack.decodeTiming(), b = stack.decodeTiming()
            let ms = min(a.total, b.total)
            print(String(format: "[dsv41-dec-bench] layer decode host graph build %.3f ms/layer", min(a.build, b.build)))
            print(String(format:
                "[dsv41-dec-bench] layer decode router=%d moe=%d hc=%d submit=%d: %.3f ms/layer (x40 = %.1f ms/token, %.1f tok/s layer-bound)",
                DeepseekV41RouterGEMV.enabled ? 1 : 0, DeepseekV41FusedMoE.enabled ? 1 : 0,
                DeepseekV41HCStage.enabled ? 1 : 0, DeepseekV4ModelInner.layerSubmit ? 1 : 0,
                ms, ms * 40, 1000 / (ms * 40)))
        }
    }

    /// Per-op stock costs at decode / verify sizes, 40 chained layers per timed graph.
    @Test("stock per-op decode costs (VMLX_DSV41_DEC_BENCH=1)")
    func stockCosts() throws {
        guard DSV41Synth.benchEnabled, let c = DSV41Synth.config else { return }
        try Device.withDefaultDevice(.gpu) {
            let moe = try DSV41Synth.moe(c, layer: 10)
            let layers = 40
            for rows in [1, 2, 4, 8] {
                let x = DSV41Synth.hidden(rows: rows, c, seed: 5)
                let full = DSV41Synth.timeMs {
                    var h = x
                    for _ in 0 ..< layers { h = moe(h, inputIds: nil) }
                    return [h]
                } / Double(layers)
                let xs = (0 ..< layers).map { DSV41Synth.hidden(rows: rows, c, seed: UInt64(100 + $0)) }
                let gate = DSV41Synth.timeMs {
                    xs.flatMap { xi in
                        let (a, b) = moe.gate(xi, inputIds: nil)
                        return [a, b]
                    }
                } / Double(layers)
                print(String(format: "[dsv41-dec-bench] stock rows=%d moe_layer_ms=%.3f gate_ms=%.3f", rows, full, gate))
            }
        }
    }
}
