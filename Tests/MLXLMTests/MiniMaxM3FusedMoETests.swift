// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import CryptoKit
import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// The Swift port of vMLX's MiniMax-M3 fused MoE block (K6 router, K7 gate/up, R rotation, K8 down; `MiniMaxM3FusedMoE`).
///
/// - Python parity: `Fixtures/MiniMaxM3/fused-moe.json` (`scripts/minimax_m3_fixtures/generate_fused_moe_fixtures.py`
///   on vMLX 8990f9bc): the block vMLX's tests build (fp32 router + bias, JANGT routed experts with SwiGLU-OAI, an
///   affine-4 g64 shared expert; 16 experts, top-4, hidden 1024, intermediate 512), two JANGT geometries (1.5 / 2.5 bits
///   with outlier columns, 2 / 3 bits without), rows 1 / 2 / 5 / 16, regenerated here from the same MLX PRNG keys
///   (input digests checked first): every kernel output equals vMLX Python's bit for bit.
/// - Row invariance: rows 1...16 in one call == the same rows one at a time (shared expert in-kernel, the default; and
///   with vMLX's 2-row threshold: the off-kernel shared expert on row-exact projections + a compiled SwiGLU).
/// - Agreement with Swift's unfused MoE block (reported; vMLX's kernels follow vMLX Python's ops, not mlx-swift's).
/// - Model level: a tiny M3 whose MoE layers are JANGT: DSpark verify rows == decode steps bit for bit with both fused
///   halves on.
@Suite("MiniMax-M3 fused MoE (vMLX K6-K8 port)", .serialized)
struct MiniMaxM3FusedMoETests {
    typealias FM = MiniMaxM3FusedMoE
    static let D = 1024, I = 512, E = 16, KS = 4

    struct Fixture: Decodable {
        struct Case: Decodable {
            let rows: Int
            let x: String
            let resid: String
            let outputs: [String: String]
            let unfused: String
            let unfusedEqualsFused: Bool
        }
        struct Geometry: Decodable {
            let name: String
            let kgu: Double
            let kdn: Double
            let nog: Int
            let nod: Int
            let seed: Int
            let arrays: [String: String]
            let cases: [Case]
        }
        let vmlxCommit: String
        let geometries: [Geometry]
    }

    static func fixture() throws -> Fixture {
        let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .appendingPathComponent("Fixtures/MiniMaxM3/fused-moe.json")
        return try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: url))
    }

    static func digest(_ x: MLXArray) -> String { MiniMaxM3FusedAttentionTests.digest(x) }
    static func uniform(_ n: Int, _ shape: [Int]) -> MLXArray { MiniMaxM3FusedAttentionTests.uniform(n, shape) }

    static func configuration(hidden: Int = D, intermediate: Int = I, experts: Int = E, layers: Int = 1,
                              moe: [Int] = [1], sparse: [Int] = [1], heads: Int = 8, kv: Int = 2, headDim: Int = 64)
        throws -> MiniMaxM3Configuration
    {
        let text: [String: Any] = [
            "model_type": "minimax_m3", "hidden_size": hidden, "num_hidden_layers": layers,
            "intermediate_size": intermediate, "dense_intermediate_size": 512, "shared_intermediate_size": intermediate,
            "num_attention_heads": heads,
            "num_key_value_heads": kv, "head_dim": headDim, "rotary_dim": 32, "rope_theta": 10_000.0, "rms_norm_eps": 1e-6,
            "vocab_size": 1024, "num_local_experts": experts, "num_experts_per_tok": KS, "routed_scaling_factor": 2.0,
            "norm_topk_prob": true, "swiglu_alpha": 1.702, "swiglu_limit": 7.0, "moe_layer_freq": moe,
            "tie_word_embeddings": false,
            "sparse_attention_config": [
                "sparse_num_index_heads": kv, "sparse_index_dim": headDim, "sparse_block_size": 32,
                "sparse_topk_blocks": 6, "sparse_local_block": 1, "sparse_attention_freq": sparse,
            ],
        ]
        let json: [String: Any] = ["model_type": "minimax_m3_vl", "text_config": text]
        return try JSONDecoder.json5().decode(
            MiniMaxM3Configuration.self, from: JSONSerialization.data(withJSONObject: json))
    }

    /// The fixture's JANGT routed layer and the digests of its arrays (keys as the generator's).
    static func routed(kgu: Double, kdn: Double, nog: Int, nod: Int, seed s: Int, D: Int = D, I: Int = I, E: Int = E)
        throws -> (JANGTRoutedExpertLayer, [String: String])
    {
        var digests: [String: String] = [:]
        var su: MLXArray?
        var gateCols: MLXArray?
        func projection(_ name: String, _ pi: Int, inputs: Int, outputs: Int, k: Double, nOut: Int) throws
            -> JANGTProjection
        {
            let base = s + 10 * pi
            let words = JANGTCode.wordsPerRow(inputs: inputs, k: k)
            let packed = uniform(base + 1, [E, outputs, words]).view(dtype: .uint32)
            let scale = Float(0.035) + Float(0.0075) * uniform(base + 2, [E, outputs])
            var signs = MLX.where(uniform(base + 3, [inputs]) .>= MLXArray(Float(0)), MLXArray(Float(1)), MLXArray(Float(-1)))
            var cols = MLXArray.full([E, max(nOut, 1)], values: MLXArray(Int32(-1)))
            var wout = MLXArray.zeros([E, outputs, max(nOut, 1)], dtype: .float16)
            if nOut > 0 {
                if name == "down" {
                    var c = (0 ..< E).flatMap { e in [11, 211, 389].prefix(nOut).map { Int32((e * 97 + $0) % inputs) } }
                    c[2 * nOut + nOut - 1] = -1
                    cols = MLXArray(c).reshaped(E, nOut)
                } else {
                    cols = MLXArray([Int32](repeating: 37, count: E * nOut)).reshaped(E, nOut)
                }
                wout = (Float(0.0125) * uniform(base + 4, [E, outputs, nOut])).asType(.float16)
            }
            if name == "up" { signs = su!; cols = gateCols! }
            let cs: MLXArray? = name == "down" ? Float(1) + Float(0.25) * uniform(base + 5, [E, inputs]) : nil
            if name == "gate" { su = signs; gateCols = cols }
            eval(packed, scale, signs, cols, wout)
            digests["\(name).jt_packed"] = digest(packed)
            digests["\(name).jt_scale"] = digest(scale)
            digests["\(name).jt_su"] = digest(signs)
            digests["\(name).jt_ocol"] = digest(cols)
            digests["\(name).jt_wout"] = digest(wout)
            if let cs { digests["\(name).jt_cs"] = digest(cs) }
            return try JANGTProjection(
                k: k, packed: packed, scale: scale, signs: signs, outlierColumnIndices: cols, outlierWeights: wout,
                columnScales: cs)
        }
        let gate = try projection("gate", 0, inputs: D, outputs: I, k: kgu, nOut: nog)
        let up = try projection("up", 1, inputs: D, outputs: I, k: kgu, nOut: nog)
        let down = try projection("down", 2, inputs: I, outputs: D, k: kdn, nOut: nod)
        let layer = try JANGTRoutedExpertLayer(
            gate: gate, up: up, down: down, hiddenSize: D, intermediateSize: I, activationLimit: 7,
            activation: .oai(alpha: 1.702))
        return (layer, digests)
    }

    /// The fixture's MoE block (router, bias, shared expert from the same keys) around `routed`.
    static func block(
        _ routed: JANGTRoutedExpertLayer, seed s: Int, digests: inout [String: String], D: Int = D, I: Int = I,
        E: Int = E
    ) throws -> MiniMaxM3SparseMoeBlock {
        let b = MiniMaxM3SparseMoeBlock(try configuration(hidden: D, intermediate: I, experts: E), routed: routed)
        let gw = Float(0.025) * uniform(s + 50, [E, D])
        let bias = Float(0.025) * uniform(s + 51, [E])
        var update: [(String, MLXArray)] = [("gate.weight", gw), ("e_score_correction_bias", bias)]
        digests["gate.weight"] = digest(gw)
        digests["bias"] = digest(bias)
        quantize(model: b) { path, module in
            path.hasPrefix("shared_experts") && module is Linear ? (64, 4, .affine) : nil
        }
        for (si, (n, o, i)) in [("gate_proj", I, D), ("up_proj", I, D), ("down_proj", D, I)].enumerated() {
            let w = (uniform(s + 60 + si, [o, i]) * Float(0.75 / Double(i).squareRoot())).asType(.bfloat16)
            let (wq, sc, bi) = MLX.quantized(w, groupSize: 64, bits: 4, mode: .affine, stream: .cpu)
            update += [
                ("shared_experts.\(n).weight", wq), ("shared_experts.\(n).scales", sc),
                ("shared_experts.\(n).biases", bi!),
            ]
            digests["shared.\(n).weight"] = digest(wq)
            digests["shared.\(n).scales"] = digest(sc)
            digests["shared.\(n).biases"] = digest(bi!)
        }
        try b.update(parameters: ModuleParameters.unflattened(update), verify: [])
        eval(b)
        return b
    }

    static func rows(seed s: Int) -> (MLXArray, MLXArray) {
        let x = (uniform(s + 70, [1, 16, D]) * Float(0.5)).asType(.bfloat16)
        let r = uniform(s + 71, [1, 16, D]).asType(.bfloat16)
        eval(x, r)
        return (x, r)
    }

    // MARK: Python parity

    @Test("Every kernel output (K6 / K7 / R / K8) == vMLX Python's, bit for bit (identical sources and inputs)")
    func pythonParity() throws {
        try Device.withDefaultDevice(.gpu) {
            let f = try Self.fixture()
            let saved = FM.sharedInKernelRows
            defer { FM.sharedInKernelRows = saved }
            FM.sharedInKernelRows = 16
            var checked = 0
            for g in f.geometries {
                let (routed, routedDigests) = try Self.routed(
                    kgu: g.kgu, kdn: g.kdn, nog: g.nog, nod: g.nod, seed: g.seed)
                var digests = routedDigests
                let block = try Self.block(routed, seed: g.seed, digests: &digests)
                for (name, want) in g.arrays {
                    #expect(digests[name] == want, "\(g.name): input array \(name) differs from the generator's")
                }
                let fm = FM.shared
                let L = try #require(fm.layer(block), "\(g.name): block not covered")
                let (xAll, rAll) = Self.rows(seed: g.seed)
                for c in g.cases {
                    let R = c.rows
                    let x = xAll[0..., ..<R], res = rAll[0..., ..<R]
                    #expect(Self.digest(x) == c.x && Self.digest(res) == c.resid, "\(g.name) rows \(R): inputs")
                    let x2 = contiguous(x.reshaped(R, Self.D)), r2 = contiguous(res.reshaped(R, Self.D))
                    let (sig, xr) = fm.router(L, x2)
                    let gu = fm.gateUp(L, x2, sig: sig, xr: xr, shared: true)
                    let hr = fm.downInput(L, gu.h, inds: gu.inds)
                    let out = fm.down(
                        L, hr: hr, h: gu.h, inds: gu.inds, scores: gu.scores, hs: gu.hs, r2: r2, sdo: nil)
                    let got = [
                        "sig": sig, "xr": xr, "h": gu.h, "hs": gu.hs, "inds": gu.inds, "scores": gu.scores, "hr": hr,
                        "out": out,
                    ]
                    for (n, want) in c.outputs {
                        #expect(Self.digest(got[n]!) == want, "\(g.name) rows \(R): kernel output \(n)")
                    }
                    let block2 = fm.residual(block, xNormed: x, resid: res)
                    #expect(Self.digest(block2.reshaped(R, Self.D)) == c.outputs["out"], "\(g.name) rows \(R): block")
                    checked += 1
                }
            }
            #expect(checked == 8)
        }
    }

    // MARK: row invariance

    @Test("Rows 1...16 in one call == one row at a time (shared expert in-kernel, and off-kernel past 2 rows)")
    func rowsAreBitIdenticalAloneOrTogether() throws {
        try Device.withDefaultDevice(.gpu) {
            let saved = FM.sharedInKernelRows
            defer { FM.sharedInKernelRows = saved }
            var digests: [String: String] = [:]
            let routed = try Self.routed(kgu: 1.5, kdn: 2.5, nog: 1, nod: 3, seed: 100).0
            let block = try Self.block(routed, seed: 100, digests: &digests)
            let (xAll, rAll) = Self.rows(seed: 300)
            for threshold in [16, 2] {
                FM.sharedInKernelRows = threshold
                let alone = concatenated(
                    (0 ..< 16).map {
                        FM.shared.residual(block, xNormed: xAll[0..., $0 ..< ($0 + 1)], resid: rAll[0..., $0 ..< ($0 + 1)])
                    }, axis: 1)
                eval(alone)
                var bad: [Int] = []
                for rows in [2, 3, 5, 8, 11, 16] {
                    let together = FM.shared.residual(block, xNormed: xAll[0..., ..<rows], resid: rAll[0..., ..<rows])
                    if !arrayEqual(together, alone[0..., ..<rows]).item(Bool.self) { bad.append(rows) }
                }
                print("[fused moe invariance] shared in-kernel up to \(threshold) rows: row counts differing from one-row calls \(bad)")
                #expect(bad.isEmpty, "shared threshold \(threshold): rows \(bad) differ from one-row calls")
            }
        }
    }

    // MARK: agreement with Swift's unfused block (reported)

    @Test("Agreement with Swift's unfused MoE block per row (reported: routing, block output)")
    func agreementWithUnfusedBlock() throws {
        try Device.withDefaultDevice(.gpu) {
            for (kgu, kdn, nog, nod, seed) in [(1.5, 2.5, 1, 3, 100), (2.0, 3.0, 0, 0, 200)] {
                var digests: [String: String] = [:]
                let routed = try Self.routed(kgu: kgu, kdn: kdn, nog: nog, nod: nod, seed: seed).0
                let block = try Self.block(routed, seed: seed, digests: &digests)
                let (xAll, rAll) = Self.rows(seed: seed + 5)
                let fused = FM.shared.residual(block, xNormed: xAll, resid: rAll)
                // Swift's stock block, one row at a time (a decode step's MoE).
                let stock = concatenated(
                    (0 ..< 16).map { rAll[0..., $0 ..< ($0 + 1)] + block(xAll[0..., $0 ..< ($0 + 1)]) }, axis: 1)
                // Routing: the stock top-k sets vs K7's.
                let L = FM.shared.layer(block)!
                let x2 = contiguous(xAll.reshaped(16, Self.D))
                let (sig, xr) = FM.shared.router(L, x2)
                let gu = FM.shared.gateUp(L, x2, sig: sig, xr: xr, shared: true)
                // The stock router of a decode step: one row at a time.
                let stockSig = concatenated(
                    (0 ..< 16).map { sigmoid(block.gate(xAll[0..., $0 ..< ($0 + 1)]).asType(.float32)).reshaped(1, -1) },
                    axis: 0)
                let stockChoice = stockSig + block.eScoreCorrectionBias
                let stockInds = argPartition(-stockChoice, kth: Self.KS - 1, axis: -1)[0..., ..<Self.KS]
                var sameSets = 0
                for r in 0 ..< 16 {
                    let a = Set(gu.inds[r].asType(.int32).asArray(Int32.self).map(Int.init))
                    let b = Set(stockInds[r].asType(.int32).asArray(Int32.self).map(Int.init))
                    if a == b { sameSets += 1 }
                }
                let routerBits = arrayEqual(sig, stockSig).item(Bool.self)
                let d = abs(fused.asType(.float32) - stock.asType(.float32))
                let rel = (d.max() / abs(stock.asType(.float32) - rAll.asType(.float32)).max()).item(Float.self)
                let equalRows = (0 ..< 16).filter { arrayEqual(fused[0..., $0], stock[0..., $0]).item(Bool.self) }.count
                print(
                    "[fused moe vs unfused] K\(kgu)/\(kdn) outliers \(nog)/\(nod): router sigmoid bit-equal \(routerBits), "
                        + "top-\(Self.KS) sets equal \(sameSets)/16, block rows bit-equal \(equalRows)/16, "
                        + "max |diff| / max |block delta| \(String(format: "%.2e", rel))")
                #expect(sameSets >= 12, "routing sets differ on \(16 - sameSets) rows")
                #expect(rel.isFinite && rel < 0.5)
            }
        }
    }

    // MARK: model level

    /// A tiny M3 (hidden 1024; layer 0 dense attention + dense MLP, 1 sparse + dense MLP, 2-3 sparse + JANGT MoE) in the
    /// fused formats: attention q / k / v / o affine 4, index MXFP8; shared experts affine 4 g64; fp32 routers.
    static func model() throws -> MiniMaxM3Model {
        let config = try configuration(layers: 4, moe: [0, 0, 1, 1], sparse: [0, 1, 1, 1])
        let routed: [Int: any WeightedRoutedExpertLayer] = [
            2: try Self.routed(kgu: 1.5, kdn: 2.5, nog: 1, nod: 3, seed: 400).0,
            3: try Self.routed(kgu: 2.0, kdn: 3.0, nog: 0, nod: 0, seed: 500).0,
        ]
        let trellis = JANGHTensorIndexPlan.trellisSuffixes.map { String($0.dropFirst()) }
        let excluded = Set(
            routed.keys.flatMap { layer in
                ["gate_proj", "up_proj", "down_proj"].flatMap { role in
                    trellis.map { "model.layers.\(layer).mlp.switch_mlp.\(role).\($0)" }
                }
            })
        let model = try MiniMaxM3Model(config, routedExperts: routed, excludedTensorNames: excluded)
        var key = MLXRandom.key(9)
        var update: [(String, MLXArray)] = []
        for (name, value) in model.parameters().flattened() where !name.contains("switch_mlp") {
            let split = MLXRandom.split(key: key)
            key = split.0
            let r = MLXRandom.normal(value.shape, key: split.1)
            if name.hasSuffix("norm.weight") {
                update.append((name, (r * Float(0.1)).asType(.bfloat16)))
            } else if name.hasSuffix("e_score_correction_bias") {
                update.append((name, (r * Float(0.05)).asType(.float32)))
            } else if name.hasSuffix(".mlp.gate.weight") {
                update.append((name, (r * Float(0.05)).asType(.float32)))
            } else if name.contains("embed_tokens") {
                update.append((name, r.asType(.bfloat16)))
            } else {
                update.append((name, (r * Float(1.5 / Double(value.dim(-1)).squareRoot())).asType(.bfloat16)))
            }
        }
        try model.update(parameters: ModuleParameters.unflattened(update), verify: [])
        quantize(model: model) { path, module in
            if path.hasSuffix("mlp.gate") || path.contains("switch_mlp") { return nil }
            guard module is Linear || module is Embedding else { return nil }
            if path.contains("index_") { return (32, 8, .mxfp8) }
            if path == "lm_head" { return (64, 6, .affine) }
            if path.hasSuffix("embed_tokens") { return (64, 8, .affine) }
            return (64, 4, .affine)
        }
        eval(model)
        return model
    }

    @Test("DSpark verify rows == one-token decode steps with both fused halves on (widths 2...17)",
          arguments: [(773, 2), (765, 8), (771, 16), (300, 17)])
    func verifyRowsEqualDecodeStepsFused(promptLength: Int, width: Int) throws {
        try Device.withDefaultDevice(.gpu) {
            MiniMaxM3FusedDecode.forceEnabled = true
            defer { MiniMaxM3FusedDecode.forceEnabled = nil }
            let model = try Self.model()
            let moeCovered = model.model.layers.map { ($0.mlp as? MiniMaxM3SparseMoeBlock).map { FM.shared.layer($0) != nil } ?? false }
            #expect(moeCovered == [false, false, true, true], "fused MoE coverage \(moeCovered)")
            model.model.dsparkTapLayers = [1, 3]
            model.model.dsparkTailCapacity = 64
            let prompt = MiniMaxM3DSparkTests.ids(promptLength, seed: UInt64(promptLength))
            let rows = MiniMaxM3DSparkTests.ids(width, seed: UInt64(7 * width))
            let cache = model.newCache(parameters: nil)
            MiniMaxM3DSparkTests.prefill(model, prompt, cache: cache)
            FM.stats.reset()
            let logits = try MiniMaxM3DSparkTests.verify(model, rows, cache: cache)
            let verifyCalls = FM.stats.snapshot["calls"] ?? 0
            let reference = model.newCache(parameters: nil)
            MiniMaxM3DSparkTests.prefill(model, prompt, cache: reference)
            FM.stats.reset()
            let steps = rows.map { MiniMaxM3DSparkTests.decode(model, $0, cache: reference) }
            #expect((FM.stats.snapshot["calls"] ?? 0) == 2 * width, "\(FM.stats.snapshot)")
            #expect(verifyCalls == 2 * ((width + 15) / 16), "verify fused MoE calls \(verifyCalls)")
            let bad = (0 ..< width).filter { !MiniMaxM3DSparkTests.equal(logits[$0], steps[$0]) }
            #expect(bad.isEmpty, "verify rows \(bad) differ from decode steps")
            let problems = MiniMaxM3DSparkTests.cacheProblems(cache, reference)
            #expect(problems.isEmpty, "\(problems.prefix(4))")
            print("[fused moe verify] prompt \(promptLength) width \(width): \(width - bad.count)/\(width) rows bit-equal")
        }
    }

    @Test("Fused MoE alone: decode logits == the stock decode, bit for bit (vMLX's kernels follow the unfused ops)")
    func fusedMoEDecodeEqualsStock() throws {
        try Device.withDefaultDevice(.gpu) {
            defer {
                MiniMaxM3FusedDecode.forceEnabled = nil
                MiniMaxM3FusedDecode.path = nil
            }
            let model = try Self.model()
            let prompt = MiniMaxM3DSparkTests.ids(773, seed: 21)
            let next = MiniMaxM3DSparkTests.ids(20, seed: 22)
            // The prompt is prefilled with the seam off in both runs: its last chunk (5 rows) would otherwise be
            // offered to the fused block, whose per-row arithmetic is a decode step's, not the stock multi-row chunk's.
            func run(_ fused: Bool) -> [MLXArray] {
                MiniMaxM3FusedDecode.forceEnabled = false
                let cache = model.newCache(parameters: nil)
                MiniMaxM3DSparkTests.prefill(model, prompt, cache: cache)
                MiniMaxM3FusedDecode.forceEnabled = fused
                MiniMaxM3FusedDecode.path = fused ? MiniMaxM3FusedKernelPath(attention: false, moe: true) : nil
                return next.map { MiniMaxM3DSparkTests.decode(model, $0, cache: cache) }
            }
            let stock = run(false)
            FM.stats.reset()
            let fused = run(true)
            #expect((FM.stats.snapshot["calls"] ?? 0) >= 2 * next.count, "decode steps took the fused MoE")
            let equal = zip(stock, fused).filter { MiniMaxM3DSparkTests.equal($0, $1) }.count
            print("[fused moe decode] \(equal)/\(next.count) decode steps' logits bit-equal to the stock decode")
            #expect(equal == next.count)
        }
    }

    @Test("Benchmark (opt-in VMLX_M3_FUSED_MOE_BENCH=1): MoE block per layer, stock vs fused, M3.1 shapes")
    func benchmark() throws {
        guard ProcessInfo.processInfo.environment["VMLX_M3_FUSED_MOE_BENCH"] == "1" else { return }
        try Device.withDefaultDevice(.gpu) {
            // M3.1 geometry: hidden 6144, intermediate 3072, 128 experts, top-4; JANGT 2-bit gate/up, 3-bit down.
            let (bd, bi, be) = (6144, 3072, 128)
            var blocks: [MiniMaxM3SparseMoeBlock] = []
            for l in 0 ..< 4 {
                var digests: [String: String] = [:]
                let routed = try Self.routed(kgu: 2, kdn: 3, nog: 0, nod: 0, seed: 1000 + 100 * l, D: bd, I: bi, E: be).0
                blocks.append(try Self.block(routed, seed: 1000 + 100 * l, digests: &digests, D: bd, I: bi, E: be))
            }
            for rows in [1, 4, 8, 16] {
                let x = (Self.uniform(7 + rows, [1, rows, bd]) * Float(0.5)).asType(.bfloat16)
                let r = Self.uniform(9 + rows, [1, rows, bd]).asType(.bfloat16)
                eval(x, r)
                func chain(_ step: (MiniMaxM3SparseMoeBlock, MLXArray) -> MLXArray) -> Double {
                    var t: [Double] = []
                    for probe in 0 ... 7 {
                        let t0 = Date()
                        var h = r
                        for b in blocks { h = step(b, h) }
                        eval(h)
                        if probe > 0 { t.append(Date().timeIntervalSince(t0) * 1000 / Double(blocks.count)) }
                    }
                    t.sort()
                    return t[t.count / 2]
                }
                let stock = chain { b, h in rows == 1 ? h + b(x) : h + b.verifyRows(x) }
                let fused = chain { b, h in FM.shared.residual(b, xNormed: x, resid: h) }
                print(
                    "[fused moe bench] rows \(rows): stock \(rows == 1 ? "decode" : "exact verify") "
                        + String(format: "%.3f", stock) + " ms / layer, fused " + String(format: "%.3f", fused)
                        + String(format: " (%.2fx)", stock / fused))
            }
        }
    }
}
