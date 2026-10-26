// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// DeepSeek-V4.1 decode / dSpark-verify fusion (1-8 rows). Synthetic tensors with the real V4.1-Flash
// shapes and bit plan (hidden 5120, 384 routed experts of width 2304, top-6, 2/3-bit group-64 gate/up,
// 3-bit down, 8-bit shared expert, bf16 router); no model loads and no shared MLX test lock, so run this
// suite on its own. The configuration is read from the local V4.1 bundle's config.json when present
// (`VMLX_DSV41_CONFIG` overrides the path); every test is skipped without it.
//
// Microbenchmarks run only with VMLX_DSV41_DEC_BENCH=1.

import Foundation
import MLX
import MLXFast
import MLXNN
import MLXRandom
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

enum DSV41Synth {
    static let configPath =
        ProcessInfo.processInfo.environment["VMLX_DSV41_CONFIG"]
        ?? "/Users/eric/models/DeepSeek-V4.1-Flash-1017-JANG-Q2Q3-IMATRIX/config.json"

    static let config: DeepseekV4Configuration? = {
        guard let data = FileManager.default.contents(atPath: configPath) else { return nil }
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

    static func bitEqual(_ a: MLXArray, _ b: MLXArray) -> Bool {
        guard a.shape == b.shape, a.dtype == b.dtype else { return false }
        return arrayEqual(a, b).item(Bool.self)
    }

    static func maxAbsDiff(_ a: MLXArray, _ b: MLXArray) -> Float {
        MLX.abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
    }
}

@Suite("DeepSeek-V4.1 decode fusion", .serialized)
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
