// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import CryptoKit
import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// The Swift port of vMLX's MiniMax-M3 fused attention half (K1-K5, `MiniMaxM3FusedAttention`).
///
/// - Sources: the embedded Metal strings are vMLX's files byte for byte (SHA-256 against the generated table and the
///   fixture).
/// - Python parity: `Fixtures/MiniMaxM3/fused-attn.json` (`scripts/minimax_m3_fixtures/generate_fused_attn_fixtures.py`
///   on vMLX 8990f9bc): one M3.1-shaped layer (hidden 1024; 64 x 128 heads, 4 KV, RoPE 64 / 1e7, index 4 x 128 MXFP8,
///   q / k / v / o affine 4 g64, blocks of 128, top-16) and caches at 700 / 2,040 / 3,001 / 9,000 keys regenerated here
///   from the same MLX PRNG keys (input digests checked first); every kernel output (K1 y, K2 q / iq / new K / V /
///   index rows, K3 scores, K4 attention, K5 residual) and the block output must equal Python's bit for bit.
/// - Row invariance: a block of R rows (1...16) == R one-row blocks (residual rows and appended lane rows), at
///   contexts below, straddling and past the dense threshold; another row's input never changes a row.
/// - Agreement with Swift's unfused path (NOT bit-identical by design: fp32 throughout, fp32 `1 + w`, one rounding):
///   per-kernel relative error printed and bounded loosely.
/// - Model level (tiny M3 with M3.1's attention formats): DSpark verify rows == one-token decode steps bit for bit with
///   the fused path on in both (widths 2...17, 17 chunked), caches equal.
@Suite("MiniMax-M3 fused attention (vMLX K1-K5 port)", .serialized)
struct MiniMaxM3FusedAttentionTests {
    typealias FA = MiniMaxM3FusedAttention
    static let hidden = 1024, heads = 64, kv = 4, hd = 128, rot = 64, block = 128, topK = 16

    // MARK: fixture

    struct Fixture: Decodable {
        struct Case: Decodable {
            let ctx: Int
            let rows: Int
            let sparse: Bool
            let inputs: [String: String]
            let outputs: [String: String]
        }
        let vmlxCommit: String
        let kernelSourcesSha256: [String: String]
        let weights: [String: String]
        let cases: [Case]
    }

    static func fixture() throws -> Fixture {
        let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .appendingPathComponent("Fixtures/MiniMaxM3/fused-attn.json")
        return try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: url))
    }

    static func digest(_ x: MLXArray) -> String {
        let c = contiguous(x)
        eval(c)
        return SHA256.hash(data: c.asData(access: .copy).data).map { String(format: "%02x", $0) }.joined()
    }

    static func uniform(_ n: Int, _ shape: [Int]) -> MLXArray {
        MLXRandom.uniform(Float(-2) ..< Float(2), shape, key: MLXRandom.key(UInt64(n)), stream: .cpu)
    }

    // MARK: the layer

    static func configuration(hidden: Int = hidden) throws -> MiniMaxM3Configuration {
        let text: [String: Any] = [
            "model_type": "minimax_m3", "hidden_size": hidden, "num_hidden_layers": 1, "intermediate_size": 256,
            "dense_intermediate_size": 512, "shared_intermediate_size": 256, "num_attention_heads": heads,
            "num_key_value_heads": kv, "head_dim": hd, "rotary_dim": rot, "rope_theta": 10_000_000.0,
            "rms_norm_eps": 1e-6, "vocab_size": 256, "num_local_experts": 8, "num_experts_per_tok": 4,
            "routed_scaling_factor": 2.0, "norm_topk_prob": true, "swiglu_alpha": 1.702, "swiglu_limit": 7.0,
            "moe_layer_freq": [0], "tie_word_embeddings": false,
            "sparse_attention_config": [
                "sparse_num_index_heads": kv, "sparse_index_dim": hd, "sparse_block_size": block,
                "sparse_topk_blocks": topK, "sparse_local_block": 1, "sparse_attention_freq": [1],
            ],
        ]
        let json: [String: Any] = ["model_type": "minimax_m3_vl", "text_config": text]
        return try JSONDecoder.json5().decode(
            MiniMaxM3Configuration.self, from: JSONSerialization.data(withJSONObject: json))
    }

    /// The fixture's layer: weights from the generator's PRNG keys, quantized on the CPU as vMLX does; returns the
    /// layer and the digests of every weight array.
    static func layer() throws -> (MiniMaxM3DecoderLayer, [String: String]) {
        let layer = MiniMaxM3DecoderLayer(try configuration(), layer: 0)
        let projections: [(String, Int, Int, Bool)] = [
            ("self_attn.q_proj", heads * hd, hidden, false), ("self_attn.k_proj", kv * hd, hidden, false),
            ("self_attn.v_proj", kv * hd, hidden, false), ("self_attn.o_proj", hidden, heads * hd, false),
            ("self_attn.indexer.index_q_proj", kv * hd, hidden, true),
            ("self_attn.indexer.index_k_proj", hd, hidden, true),
        ]
        quantize(model: layer) { path, module in
            guard module is Linear, path.hasPrefix("self_attn") else { return nil }
            return path.contains("index_") ? (32, 8, .mxfp8) : (64, 4, .affine)
        }
        var update: [(String, MLXArray)] = []
        var digests: [String: String] = [:]
        for (seed, (path, o, i, mx8)) in projections.enumerated() {
            let w = (uniform(seed + 1, [o, i]) * Float(0.75 / Double(i).squareRoot())).asType(.bfloat16)
            let (wq, s, b) = MLX.quantized(
                w, groupSize: mx8 ? 32 : 64, bits: mx8 ? 8 : 4, mode: mx8 ? .mxfp8 : .affine, stream: .cpu)
            update.append(("\(path).weight", wq))
            update.append(("\(path).scales", s))
            if let b { update.append(("\(path).biases", b)) }
        }
        let norms: [(String, Int)] = [
            ("input_layernorm", hidden), ("self_attn.q_norm", hd), ("self_attn.k_norm", hd),
            ("self_attn.indexer.index_q_norm", hd), ("self_attn.indexer.index_k_norm", hd),
        ]
        for (seed, (path, n)) in norms.enumerated() {
            update.append(("\(path).weight", (uniform(seed + 11, [n]) * Float(0.1)).asType(.bfloat16)))
        }
        for (k, v) in update { digests[k] = digest(v) }
        try layer.update(parameters: ModuleParameters.unflattened(update), verify: [])
        eval(layer)
        return (layer, digests)
    }

    /// Lanes `[1, KV, ctx, D]` x2, index keys `[1, 1, ctx, D]`, and 16 input rows `[1, 16, hidden]` of fixture case `c`.
    static func lanes(ctx: Int, case c: Int) -> (MLXArray, MLXArray, MLXArray, MLXArray) {
        let base = 1000 * c
        return Device.withDefaultDevice(.cpu) {
            let k = uniform(base + 1, [1, kv, ctx, hd]).asType(.bfloat16)
            let v = uniform(base + 2, [1, kv, ctx, hd]).asType(.bfloat16)
            let drift = cumsum(uniform(base + 4, [1, 1, ctx, hd]) * Float(0.05), axis: 2)
            let ik = (uniform(base + 3, [1, 1, ctx, hd]) + drift).asType(.bfloat16)
            let x = uniform(base + 5, [1, 16, hidden]).asType(.bfloat16)
            eval(k, v, ik, x)
            return (k, v, ik, x)
        }
    }

    static func cache(_ k: MLXArray, _ v: MLXArray, _ ik: MLXArray) -> MiniMaxM3SparseCache {
        let c = MiniMaxM3SparseCache()
        c.state = [k, v, ik]
        return c
    }

    // MARK: sources

    @Test("Embedded Metal sources are vMLX's files byte for byte")
    func sourcesAreVMLXs() throws {
        let f = try Self.fixture()
        for (name, text) in MiniMaxM3FusedKernelSources.byName {
            let got = SHA256.hash(data: Data(text.utf8)).map { String(format: "%02x", $0) }.joined()
            #expect(got == MiniMaxM3FusedKernelSources.sha256[name], "\(name)")
            if let want = f.kernelSourcesSha256[name] { #expect(got == want, "\(name) vs the fixture") }
        }
        #expect(f.kernelSourcesSha256.count == 6)
    }

    // MARK: Python parity

    @Test("Every kernel output and the block output == vMLX Python's, bit for bit (identical sources and inputs)")
    func pythonParity() throws {
        try Device.withDefaultDevice(.gpu) {
            let f = try Self.fixture()
            let (layer, weightDigests) = try Self.layer()
            for (name, want) in f.weights {
                #expect(weightDigests[name] == want, "input weight \(name) differs from the generator's")
            }
            let fa = FA.shared
            guard case .success(let p) = fa.plan(layer) else {
                Issue.record("plan: \(fa.unsupportedReason(layer) ?? "?")")
                return
            }
            var checked = 0
            for (ci, c) in f.cases.enumerated() {
                let (k, v, ik, xall) = Self.lanes(ctx: c.ctx, case: ci + 1)
                let x = xall[0..., ..<c.rows]
                let inputs = ["k": k, "v": v, "ik": ik, "x": x]
                for (n, a) in inputs { #expect(Self.digest(a) == c.inputs[n], "case \(c.ctx)/\(c.rows) input \(n)") }
                let R = c.rows
                let x2 = x.reshaped(R, Self.hidden)
                // Kernel by kernel, the fixture's host order.
                let y = fa.qkv(layer, p, x2)
                let nb = (c.ctx + R + p.block - 1) / p.block
                let meta = MLXArray([Int32(c.ctx), Int32(R), Int32(nb), 0, 0, 0, 0, 0])
                let h2 = fa.heads(layer, p, y, meta, laneDTypes: (.bfloat16, .bfloat16, .bfloat16))
                let kl = concatenated([k, h2.k.transposed(1, 0, 2).expandedDimensions(axis: 0)], axis: 2)
                let vl = concatenated([v, h2.v.transposed(1, 0, 2).expandedDimensions(axis: 0)], axis: 2)
                let il = concatenated([ik, h2.ik.transposed(1, 0, 2).expandedDimensions(axis: 0)], axis: 2)
                let sc: MLXArray? = nb > p.keep ? fa.scores(p, h2.iq, il, meta, nb: nb) : nil
                let o = fa.msa(p, h2.q, kl, vl, sc ?? fa.dummy(), h2.cnt, meta)
                let h = fa.oproj(layer, p, o.reshaped(R, p.H * p.D), x2)
                var outs = ["y": y, "q": h2.q, "iq": h2.iq, "kn": h2.k, "vn": h2.v, "ikn": h2.ik, "o": o, "h": h]
                if let sc { outs["sc"] = sc }
                #expect((sc != nil) == c.sparse)
                for (n, want) in c.outputs {
                    #expect(Self.digest(outs[n]!) == want, "case \(c.ctx)/\(R) kernel output \(n)")
                }
                // The Swift host (cache appends through MiniMaxM3SparseCache) gives the same block output.
                let cache = Self.cache(k, v, ik)
                let hb = fa.attentionBlock(layer, x: x, cache: cache)
                #expect(Self.digest(hb.reshaped(R, Self.hidden)) == c.outputs["h"], "case \(c.ctx)/\(R) block")
                #expect(cache.offset == c.ctx + R && cache.indexRows == c.ctx + R)
                checked += 1
            }
            #expect(checked == f.cases.count && checked == 7)
        }
    }

    // MARK: row invariance

    static func block(_ layer: MiniMaxM3DecoderLayer, _ x: MLXArray, ctx: Int, case c: Int, perRow: Bool)
        -> (MLXArray, [MLXArray])
    {
        let (k, v, ik, _) = lanes(ctx: ctx, case: c)
        let cache = Self.cache(k, v, ik)
        let R = x.dim(1)
        let h: MLXArray
        if perRow {
            h = concatenated(
                (0 ..< R).map { FA.shared.attentionBlock(layer, x: x[0..., $0 ..< ($0 + 1)], cache: cache) }, axis: 1)
        } else {
            h = FA.shared.attentionBlock(layer, x: x, cache: cache)
        }
        let lanes = cache.state.map { $0[.ellipsis, ctx..., 0...] }
        eval([h] + lanes)
        return (h, lanes)
    }

    @Test("Rows 1...16 in one block == the same rows as one-row blocks (residual and K / V / index lane rows)")
    func rowsAreBitIdenticalAloneOrTogether() throws {
        try Device.withDefaultDevice(.gpu) {
            let (layer, _) = try Self.layer()
            var checked = 0
            // 700: dense (every block visible); 2,040: rows cross 2,048 (dense and sparse rows in one launch); 3,001.
            for (ci, ctx) in [700, 2040, 3001].enumerated() {
                let (_, _, _, xall) = Self.lanes(ctx: ctx, case: 50 + ci)
                for rows in [2, 3, 5, 8, 11, 16] {
                    let x = xall[0..., ..<rows]
                    let together = Self.block(layer, x, ctx: ctx, case: 50 + ci, perRow: false)
                    let alone = Self.block(layer, x, ctx: ctx, case: 50 + ci, perRow: true)
                    #expect(
                        arrayEqual(together.0, alone.0).item(Bool.self), "ctx \(ctx) rows \(rows): residual rows")
                    for (a, b) in zip(together.1, alone.1) {
                        #expect(arrayEqual(a, b).item(Bool.self), "ctx \(ctx) rows \(rows): lane rows")
                    }
                    checked += 1
                }
                // Another row's input never changes a row: row 0 alone == row 0 next to a different row 1.
                let (_, _, _, other) = Self.lanes(ctx: ctx, case: 90 + ci)
                let a = Self.block(layer, xall[0..., ..<2], ctx: ctx, case: 50 + ci, perRow: false).0
                let mixed = concatenated([xall[0..., ..<1], other[0..., ..<1]], axis: 1)
                let b = Self.block(layer, mixed, ctx: ctx, case: 50 + ci, perRow: false).0
                #expect(arrayEqual(a[0..., ..<1], b[0..., ..<1]).item(Bool.self), "ctx \(ctx): neighbour changed row 0")
                #expect(!arrayEqual(a[0..., 1...], b[0..., 1...]).item(Bool.self), "control: row 1 differs")
            }
            #expect(checked == 18)
        }
    }

    // MARK: agreement with the unfused path (reported)

    static func rel(_ a: MLXArray, _ ref: MLXArray) -> Float {
        let d = abs(a.asType(.float32) - ref.asType(.float32)).max()
        let m = abs(ref.asType(.float32)).max()
        return (d / maximum(m, MLXArray(Float(1e-30)))).item(Float.self)
    }

    @Test("Per-kernel agreement with Swift's unfused path (reported; not bit-identical by design)")
    func agreementWithUnfusedPath() throws {
        try Device.withDefaultDevice(.gpu) {
            let (layer, _) = try Self.layer()
            let fa = FA.shared
            guard case .success(let p) = fa.plan(layer) else {
                Issue.record("plan")
                return
            }
            let attn = layer.selfAttn
            let ix = attn.indexer!
            for (ci, (ctx, rows)) in [(700, 1), (3001, 1), (3001, 8), (9000, 4)].enumerated() {
                let (k, v, ik, xall) = Self.lanes(ctx: ctx, case: 70 + ci)
                let x = xall[0..., ..<rows]
                let x2 = x.reshaped(rows, Self.hidden)
                // K1 vs the stock norm + projections
                let y = fa.qkv(layer, p, x2)
                let xn = layer.inputLayerNorm(x)
                let stockY = concatenated(
                    [attn.wq(xn), attn.wk(xn), attn.wv(xn), ix.queryProjection(xn), ix.keyProjection(xn)], axis: -1
                ).reshaped(rows, -1)
                // K2 vs the stock head norms + RoPE
                let nb = (ctx + rows + p.block - 1) / p.block
                let meta = MLXArray([Int32(ctx), Int32(rows), Int32(nb), 0, 0, 0, 0, 0])
                let h2 = fa.heads(layer, p, y, meta, laneDTypes: (.bfloat16, .bfloat16, .bfloat16))
                let (sq, sk, sv) = attn.project(xn, offset: ctx)
                let (siq, sik) = ix.project(xn, offset: ctx)
                // Whole block vs the stock decode / multi-row forward on the same cache
                let fused = fa.attentionBlock(layer, x: x, cache: Self.cache(k, v, ik))
                let stockCache = Self.cache(k, v, ik)
                let stock = x + attn(xn, mask: rows > 1 ? .causal : .none, cache: stockCache)
                let report = [
                    ("K1 q|k|v|iq|ik", Self.rel(y, stockY)),
                    ("K2 q", Self.rel(h2.q, sq[0].transposed(1, 0, 2))),
                    ("K2 k", Self.rel(h2.k, sk[0].transposed(1, 0, 2))),
                    ("K2 v", Self.rel(h2.v, sv[0].transposed(1, 0, 2))),
                    ("K2 index q", Self.rel(h2.iq, siq[0].transposed(1, 0, 2))),
                    ("K2 index k", Self.rel(h2.ik, sik[0].transposed(1, 0, 2))),
                    ("block (K1-K5) residual", Self.rel(fused - x, stock - x)),
                ]
                print(
                    "[fused attn vs unfused] ctx \(ctx) rows \(rows): "
                        + report.map { "\($0.0) \(String(format: "%.2e", $0.1))" }.joined(separator: ", "))
                for (name, r) in report {
                    // Not a bit-identity bound: the selection can differ by whole blocks at near-ties (vMLX measured
                    // 3-12 % of (row, group) pairs past 2,048 keys), which moves the block output.
                    #expect(r.isFinite && r < (name.hasPrefix("block") ? 0.5 : 2e-2), "ctx \(ctx) rows \(rows): \(name) \(r)")
                }
            }
        }
    }

    // MARK: model level

    /// The DSpark test model's plan with M3.1's attention formats (q / k / v / o affine 4, index MXFP8): layers 1...3
    /// take the fused path, layer 0 (dense attention) the stock one (vMLX test `_plan`).
    static func plan(_ path: String, _ module: Module) -> (groupSize: Int, bits: Int, mode: QuantizationMode)? {
        if path.hasSuffix("mlp.gate") { return nil }
        if path.contains("switch_mlp") { return module is SwitchLinear ? (64, 4, .affine) : nil }
        guard module is Linear || module is Embedding else { return nil }
        if path.contains("index_") { return (32, 8, .mxfp8) }
        if path.contains("self_attn") { return (64, 4, .affine) }
        if path == "lm_head" { return (64, 6, .affine) }
        if path.hasSuffix("embed_tokens") { return (64, 8, .affine) }
        return (64, 4, .affine)
    }

    static func fusedFormatModel() throws -> MiniMaxM3Model {
        let model = MiniMaxM3Model(try MiniMaxM3DSparkTests.m3Configuration())
        var key = MLXRandom.key(5)
        var update: [(String, MLXArray)] = []
        for (name, value) in model.parameters().flattened() {
            let split = MLXRandom.split(key: key)
            key = split.0
            let r = MLXRandom.normal(value.shape, key: split.1)
            if name.hasSuffix("norm.weight") {
                update.append((name, (r * Float(0.1)).asType(.bfloat16)))
            } else if name.hasSuffix("e_score_correction_bias") {
                update.append((name, (r * Float(0.05)).asType(.float32)))
            } else if name.hasSuffix(".mlp.gate.weight") {
                update.append((name, (r * Float(0.3)).asType(.float32)))
            } else if name.contains("embed_tokens") {
                update.append((name, r.asType(.bfloat16)))
            } else {
                update.append((name, (r * Float(1.5 / Double(value.dim(-1)).squareRoot())).asType(.bfloat16)))
            }
        }
        try model.update(parameters: ModuleParameters.unflattened(update), verify: .all)
        quantize(model: model, filter: plan)
        eval(model)
        return model
    }

    @Test("DSpark verify rows == one-token decode steps with the fused path on in both (widths 2...17)",
          arguments: [(773, 2), (765, 8), (1021, 9), (771, 16), (300, 17)])
    func verifyRowsEqualDecodeStepsFused(promptLength: Int, width: Int) throws {
        try Device.withDefaultDevice(.gpu) {
            MiniMaxM3FusedDecode.forceEnabled = true
            defer { MiniMaxM3FusedDecode.forceEnabled = nil }
            let model = try Self.fusedFormatModel()
            let covered = model.model.layers.map { FA.shared.unsupportedReason($0) == nil }
            #expect(covered == [false, true, true, true], "fused coverage \(covered)")
            model.model.dsparkTapLayers = MiniMaxM3DSparkTests.draftTaps
            model.model.dsparkTailCapacity = MiniMaxM3DSparkTests.draftWindow
            let prompt = MiniMaxM3DSparkTests.ids(promptLength, seed: UInt64(promptLength))
            let rows = MiniMaxM3DSparkTests.ids(width, seed: UInt64(7 * width))
            let cache = model.newCache(parameters: nil)
            MiniMaxM3DSparkTests.prefill(model, prompt, cache: cache)
            FA.stats.reset()
            let logits = try MiniMaxM3DSparkTests.verify(model, rows, cache: cache)
            let verifyCalls = FA.stats.snapshot["calls"] ?? 0
            let reference = model.newCache(parameters: nil)
            MiniMaxM3DSparkTests.prefill(model, prompt, cache: reference)
            FA.stats.reset()
            let steps = rows.map { MiniMaxM3DSparkTests.decode(model, $0, cache: reference) }
            #expect((FA.stats.snapshot["calls"] ?? 0) == 3 * width, "\(FA.stats.snapshot)")
            #expect(verifyCalls == 3 * ((width + 15) / 16), "verify fused calls \(verifyCalls)")
            let bad = (0 ..< width).filter { !MiniMaxM3DSparkTests.equal(logits[$0], steps[$0]) }
            #expect(bad.isEmpty, "verify rows \(bad) differ from decode steps")
            let problems = MiniMaxM3DSparkTests.cacheProblems(cache, reference)
            #expect(problems.isEmpty, "\(problems.prefix(4))")
            print("[fused verify] prompt \(promptLength) width \(width): \(width - bad.count)/\(width) rows bit-equal")
        }
    }

    /// The noise floor: the stock attention block with its output moved by one bf16 rounding step (x (1 + 2^-8)) on
    /// the sparse layers. Measured on this tiny random MoE: such a perturbation alone moves single steps' logits by
    /// whole units (expert routing flips) and flips 2-3 of 30 argmaxes, so it is the reference a non-bit-identical
    /// path has to be compared with (a 1e-3 perturbation rounds away in bf16 and changes nothing).
    final class UlpPerturbedStock: MiniMaxM3FusedPath, @unchecked Sendable {
        func attentionBlock(
            layer: Int, module: MiniMaxM3DecoderLayer, x: MLXArray, cache: KVCache?, verify: Bool
        ) -> MLXArray? {
            guard module.selfAttn.indexer != nil else { return nil }
            let a = module.selfAttn(module.inputLayerNorm(x), mask: x.dim(1) > 1 ? .causal : .none, cache: cache)
            return x + (a.asType(.float32) * (1 + Float(1.0 / 256))).asType(a.dtype)
        }
        func mlpBlock(layer: Int, module: MiniMaxM3DecoderLayer, h: MLXArray, verify: Bool) -> MLXArray? { nil }
    }

    @Test("Fused vs stock logits on the tiny model: within the floor of one bf16 rounding step of the attention output")
    func fusedVsStockLogits() throws {
        try Device.withDefaultDevice(.gpu) {
            defer {
                MiniMaxM3FusedDecode.forceEnabled = nil
                MiniMaxM3FusedDecode.path = nil
            }
            let model = try Self.fusedFormatModel()
            for (promptLength, seed) in [(773, 11), (300, 3)] {
                let prompt = MiniMaxM3DSparkTests.ids(promptLength, seed: UInt64(seed))
                let next = MiniMaxM3DSparkTests.ids(30, seed: UInt64(seed + 1))
                // Teacher-forced: every variant decodes the same tokens after the same prompt.
                func run(_ path: (any MiniMaxM3FusedPath)?, on: Bool) -> [MLXArray] {
                    MiniMaxM3FusedDecode.forceEnabled = on
                    MiniMaxM3FusedDecode.path = path
                    let cache = model.newCache(parameters: nil)
                    MiniMaxM3DSparkTests.prefill(model, prompt, cache: cache)
                    return next.map { MiniMaxM3DSparkTests.decode(model, $0, cache: cache).asType(.float32) }
                }
                let base = run(nil, on: false)
                let floor = run(UlpPerturbedStock(), on: true)
                FA.stats.reset()
                let fused = run(nil, on: true)  // the default path: the fused kernels
                #expect((FA.stats.snapshot["calls"] ?? 0) >= 3 * next.count, "decode steps took the fused path")
                func maxDiff(_ a: [MLXArray], _ b: [MLXArray]) -> [Float] {
                    zip(a, b).map { abs($0 - $1).max().item(Float.self) }
                }
                func flips(_ a: [MLXArray], _ b: [MLXArray]) -> Int {
                    zip(a, b).filter { argMax($0).item(Int.self) != argMax($1).item(Int.self) }.count
                }
                func median(_ v: [Float]) -> Float { v.sorted()[v.count / 2] }
                let dFloor = maxDiff(base, floor), dFused = maxDiff(base, fused)
                let (fFloor, fFused) = (flips(base, floor), flips(base, fused))
                print(
                    "[fused vs stock logits] prompt \(promptLength): max|dlogit| per step fused "
                        + "\(dFused.map { String(format: "%.2f", $0) }) | 1-ulp floor \(dFloor.map { String(format: "%.2f", $0) })")
                print(
                    "[fused vs stock logits] prompt \(promptLength): median fused \(String(format: "%.3f", median(dFused))) "
                        + "vs floor \(String(format: "%.3f", median(dFloor))); argmax flips fused \(fFused) / floor \(fFloor) of \(base.count)")
                #expect(dFloor.contains { $0 > 0 }, "the floor perturbation changed nothing (invalid floor)")
                #expect(median(dFused) <= 3 * max(median(dFloor), 0.0625))
                #expect(fFused <= fFloor + 2, "argmax flips fused \(fFused) vs floor \(fFloor)")
            }
        }
    }

    // MARK: timing (opt-in)

    /// One M3.1-sized attention layer (hidden 6144) with random weights in M3.1's formats.
    static func benchLayer(seed: UInt64) throws -> MiniMaxM3DecoderLayer {
        let layer = MiniMaxM3DecoderLayer(try configuration(hidden: 6144), layer: 0)
        var key = MLXRandom.key(seed)
        var update: [(String, MLXArray)] = []
        for (name, value) in layer.parameters().flattened() where name.hasPrefix("self_attn") || name.hasPrefix("input") {
            let split = MLXRandom.split(key: key)
            key = split.0
            let r = MLXRandom.normal(value.shape, key: split.1)
            let scale = name.hasSuffix("norm.weight") ? Float(0.1) : Float(1.5 / Double(value.dim(-1)).squareRoot())
            update.append((name, (r * scale).asType(.bfloat16)))
        }
        try layer.update(parameters: ModuleParameters.unflattened(update), verify: [])
        quantize(model: layer) { path, module in
            guard module is Linear, path.hasPrefix("self_attn") else { return nil }
            return path.contains("index_") ? (32, 8, .mxfp8) : (64, 4, .affine)
        }
        eval(layer)
        return layer
    }

    @Test("Benchmark (opt-in VMLX_M3_FUSED_ATTN_BENCH=1): attention block per layer, stock vs fused, M3.1 shapes")
    func benchmark() throws {
        guard ProcessInfo.processInfo.environment["VMLX_M3_FUSED_ATTN_BENCH"] == "1" else { return }
        try Device.withDefaultDevice(.gpu) {
            let layers = try (0 ..< 8).map { try Self.benchLayer(seed: UInt64(100 + $0)) }
            for ctx in [1000, 4000, 16000] {
                let caches = layers.indices.map { i -> MiniMaxM3SparseCache in
                    let k = (MLXRandom.normal([1, Self.kv, ctx, Self.hd], key: MLXRandom.key(UInt64(7 * i + 1))) * 0.5)
                        .asType(.bfloat16)
                    let v = MLXRandom.normal([1, Self.kv, ctx, Self.hd], key: MLXRandom.key(UInt64(7 * i + 2)))
                        .asType(.bfloat16)
                    let ik = MLXRandom.normal([1, 1, ctx, Self.hd], key: MLXRandom.key(UInt64(7 * i + 3)))
                        .asType(.bfloat16)
                    eval(k, v, ik)
                    return Self.cache(k, v, ik)
                }
                for rows in [1, 4, 8, 16] {
                    let x = (MLXRandom.normal([1, rows, 6144], key: MLXRandom.key(UInt64(rows))) * 0.5).asType(.bfloat16)
                    eval(x)
                    func chain(_ step: (MiniMaxM3DecoderLayer, MLXArray, MiniMaxM3SparseCache) -> MLXArray) -> Double {
                        var t: [Double] = []
                        for probe in 0 ... 7 {
                            let t0 = Date()
                            var h = x
                            for (layer, cache) in zip(layers, caches) { h = step(layer, h, cache) }
                            eval(h)
                            if probe > 0 { t.append(Date().timeIntervalSince(t0) * 1000 / Double(layers.count)) }
                            for cache in caches { _ = cache.trim(rows) }
                        }
                        t.sort()
                        return t[t.count / 2]
                    }
                    let stock = chain { layer, h, cache in
                        let xn = layer.inputLayerNorm(h)
                        return rows == 1
                            ? h + layer.selfAttn(xn, mask: .none, cache: cache)
                            : h + layer.selfAttn.verifyRows(xn, cache: cache, offset: cache.offset)
                    }
                    let fused = chain { layer, h, cache in FA.shared.attentionBlock(layer, x: h, cache: cache) }
                    print(
                        "[fused attn bench] ctx \(ctx) rows \(rows): stock \(rows == 1 ? "decode" : "exact verify") "
                            + String(format: "%.3f", stock) + " ms / layer, fused " + String(format: "%.3f", fused)
                            + String(format: " (%.2fx)", stock / fused))
                }
            }
        }
    }
}
