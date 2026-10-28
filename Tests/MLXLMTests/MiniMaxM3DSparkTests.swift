// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import CryptoKit
import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// MiniMax-M3.1 DSpark (`MiniMaxM3DSpark.swift`, `MiniMaxM3Verify.swift`): the drafter against the vMLX Python runtime,
/// and the target side (verify, rollback, the speculative loop) against plain decoding.
///
/// Drafter parity: `Fixtures/MiniMaxM3/dspark-parity.json` + `Fixtures/MiniMaxM3/dspark/<variant>/` from
/// `scripts/minimax_m3_fixtures/generate_dspark_drafter_fixtures.py` (Python 630929ed `drafter.py`). Six tiny synthetic
/// drafters (BF16, the release's MXFP8 layout, four pre-quantized copies written by `write_quantized_minimax_drafter`)
/// are loaded with the Swift loader; every loaded parameter, the block hidden, the head logits, the Markov draft
/// (candidate and full-vocabulary paths, tokens and confidences) and a multi-round proposal sequence over a sliding
/// context must equal Python's bit for bit (SHA-256). Inputs are U[-2, 2) from MLX PRNG keys on the CPU, regenerated
/// here and digest-checked first.
///
/// Target (Swift only; no real weights): a tiny synthetic M3 (4 layers: dense attention + dense MLP, sparse attention
/// + dense MLP, two sparse attention + MoE layers of 8 experts / top-4 with a shared expert), quantized per module like
/// an M3.1 JANGHT bundle (MXFP8 q / o, affine 8 k / v, affine 6 indexer and lm_head, affine 8 embeddings, affine 4
/// elsewhere, float32 routers). Index blocks of 32 keys with top-6, so a 773-token context selects (asserted); one
/// test runs the released geometry (blocks of 128, top-16) past 2048 tokens. Lengths do not divide by 32 / 64 / 128 /
/// 256, verify windows straddle those boundaries. Exactness is bit-equality, no tolerance.
@Suite("MiniMax-M3.1 DSpark drafter and target", .serialized)
struct MiniMaxM3DSparkTests {
    typealias Base = MiniMaxM3MSABlockSparseTests

    static var fixtures: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Fixtures/MiniMaxM3")
    }

    static func digest(_ x: MLXArray) -> String { Base.digest(x) }

    struct FixtureTokenizer: Tokenizer {
        var bosToken: String? { nil }
        var eosToken: String? { nil }
        var unknownToken: String? { nil }
        func encode(text: String, addSpecialTokens: Bool) -> [Int] { text.utf8.map { Int($0) } }
        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
        func convertTokenToId(_ token: String) -> Int? { nil }
        func convertIdToToken(_ id: Int) -> String? { nil }
        func applyChatTemplate(
            messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] { [] }
    }

    struct FixtureTokenizerLoader: TokenizerLoader {
        func load(from directory: URL) async throws -> any Tokenizer { FixtureTokenizer() }
    }

    static func dtypeName(_ d: DType) -> String {
        switch d {
        case .bfloat16: "bfloat16"
        case .float16: "float16"
        case .float32: "float32"
        case .uint32: "uint32"
        case .uint8: "uint8"
        case .int32: "int32"
        default: "\(d)"
        }
    }

    // MARK: - Drafter parity with the Python runtime

    static func parityFixture() throws -> [String: Any] {
        let data = try Data(contentsOf: fixtures.appendingPathComponent("dspark-parity.json"))
        return try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    static let variants = ["bf16", "release", "q-default", "q-markov8", "q-mixed", "q-heads8"]

    static func uniform(_ n: Int, _ shape: [Int], _ scale: Float = 1) -> MLXArray {
        Device.withDefaultDevice(.cpu) {
            MLXRandom.uniform(Float(-2) ..< Float(2), shape, key: MLXRandom.key(UInt64(n)), stream: .cpu)
                * scale
        }
    }

    struct ParityInputs {
        let feats: MLXArray, block: MLXArray, head: Linear, table: Embedding, chunks: [MLXArray]
    }

    /// The generator's inputs (hidden 64, two taps, vocabulary 128), digest-checked.
    static func parityInputs(_ fixture: [String: Any]) throws -> ParityInputs {
        let digests = try #require(fixture["inputs"] as? [String: Any])
        let rounds = try #require(fixture["rounds"] as? [[Int]])
        let (feats, block, headW, table, chunks) = Device.withDefaultDevice(.cpu) {
            () -> (MLXArray, MLXArray, MLXArray, MLXArray, [MLXArray]) in
            let feats = uniform(1, [1, 9, 128]).asType(.bfloat16)
            let block = uniform(2, [1, 4, 64]).asType(.bfloat16)
            let headW = uniform(3, [128, 64], 0.3).asType(.bfloat16)
            let table = uniform(4, [128, 64]).asType(.bfloat16)
            let chunks = rounds.enumerated().map { uniform(10 + $0.offset, [1, $0.element[0], 128]).asType(.bfloat16) }
            eval(feats, block, headW, table)
            eval(chunks)
            return (feats, block, headW, table, chunks)
        }
        try #require(digest(feats) == digests["feats"] as? String, "feats differ from Python's")
        try #require(digest(block) == digests["block"] as? String, "block differs from Python's")
        try #require(digest(headW) == digests["head"] as? String, "head differs from Python's")
        try #require(digest(table) == digests["table"] as? String, "table differs from Python's")
        try #require(chunks.map(digest) == digests["chunks"] as? [String], "context chunks differ from Python's")
        return ParityInputs(
            feats: feats, block: block, head: Linear(weight: headW), table: Embedding(weight: table), chunks: chunks)
    }

    @Test("Config: the Python runtime's minimax_config, and what the flavor does not implement is refused")
    func configMatchesPython() throws {
        let fixture = try Self.parityFixture()
        let drafters = try #require(fixture["drafters"] as? [String: Any])
        let release = try #require(drafters["release"] as? [String: Any])
        let py = try #require(release["config"] as? [String: Any])
        let raw = try MiniMaxM3DSparkLoader.rawConfig(Self.fixtures.appendingPathComponent("dspark/release"))
        #expect(MiniMaxM3DSparkConfig.isMiniMax(raw))
        let c = try MiniMaxM3DSparkConfig(raw: raw)
        func int(_ k: String) -> Int? { (py[k] as? NSNumber)?.intValue }
        func dbl(_ k: String) -> Double? { (py[k] as? NSNumber)?.doubleValue }
        let dflash = try #require(py["dflash_config"] as? [String: Any])
        let rope = try #require(py["rope_parameters"] as? [String: Any])
        #expect(c.hiddenSize == int("hidden_size") && c.layers == int("num_hidden_layers"))
        #expect(c.heads == int("num_attention_heads") && c.kvHeads == int("num_key_value_heads"))
        #expect(c.headDim == int("head_dim") && c.rotaryDim == int("rotary_dim"))
        #expect(c.vocabularySize == int("vocab_size") && c.intermediateSize == int("intermediate_size"))
        #expect(c.slidingWindow == int("sliding_window") && c.blockSize == int("block_size"))
        #expect(Double(c.rmsNormEps) == Double(Float(dbl("rms_norm_eps")!)))
        #expect(Double(c.swigluAlpha) == Double(Float(dbl("swiglu_alpha")!)))
        #expect(Double(c.swigluLimit) == Double(Float(dbl("swiglu_limit")!)))
        #expect(Double(c.ropeTheta) == Double(Float((rope["rope_theta"] as! NSNumber).doubleValue)))
        #expect(c.maskTokenId == (dflash["mask_token_id"] as? NSNumber)?.intValue)
        #expect(c.targetLayerIds == (dflash["target_layer_ids"] as? [NSNumber])?.map(\.intValue))
        #expect(c.markovRank == (dflash["markov_rank"] as? NSNumber)?.intValue)
        // Refused, as Python refuses them.
        for (edit, reason) in [
            ({ (t: inout [String: Any]) in t["layer_types"] = ["full_attention", "sliding_attention"] }, "non-sliding"),
            ({ (t: inout [String: Any]) in t["dspark_markov_head_type"] = "gated" }, "markov head"),
            ({ (t: inout [String: Any]) in t["dspark_use_dense_ffn"] = false }, "MoE draft FFN"),
        ] as [((inout [String: Any]) -> Void, String)] {
            var bad = raw
            var text = try #require(bad["text_config"] as? [String: Any])
            edit(&text)
            bad["text_config"] = text
            #expect(throws: MiniMaxM3DSparkConfig.ConfigError.self, "\(reason)") {
                try MiniMaxM3DSparkConfig(raw: bad)
            }
        }
        var noConfidence = raw
        noConfidence["enable_confidence_head"] = false
        #expect(throws: MiniMaxM3DSparkConfig.ConfigError.self) { try MiniMaxM3DSparkConfig(raw: noConfidence) }
        #expect(MiniMaxM3DSparkLoader.tensorName("language_model.model.dspark.layers.3.decoder_layer.mlp.up_proj.weight")
            == "layers.3.mlp.up_proj.weight")
        #expect(MiniMaxM3DSparkLoader.tensorName("language_model.model.dspark.final_norm.weight") == "norm.weight")
        #expect(MiniMaxM3DSparkLoader.tensorName("fc.weight") == "fc.weight")
    }

    @Test("Drafter parity: every variant loads as Python loads it and drafts the same tokens, bit for bit",
          arguments: variants)
    func drafterMatchesPython(variant: String) throws {
        let fixture = try Self.parityFixture()
        let drafters = try #require(fixture["drafters"] as? [String: Any])
        let py = try #require(drafters[variant] as? [String: Any])
        let inputs = try Self.parityInputs(fixture)
        let directory = Self.fixtures.appendingPathComponent("dspark/\(variant)")
        try Device.withDefaultDevice(.gpu) {
            let (config, drafter) = try MiniMaxM3DSparkLoader.load(directory: directory)

            // Loaded parameters: the same names, dtypes, shapes and bytes.
            let pyParams = try #require(py["parameters"] as? [String: [String: Any]])
            let swiftParams = Dictionary(uniqueKeysWithValues: drafter.parameters().flattened())
            #expect(Set(swiftParams.keys) == Set(pyParams.keys), "\(variant): parameter names")
            var sameBytes = 0
            for (name, p) in pyParams {
                guard let a = swiftParams[name] else { continue }
                #expect(Self.dtypeName(a.dtype) == p["dtype"] as? String, "\(variant) \(name) dtype")
                #expect(a.shape == (p["shape"] as? [NSNumber])?.map(\.intValue), "\(variant) \(name) shape")
                if Self.digest(a) == p["sha256"] as? String { sameBytes += 1 }
            }
            #expect(sameBytes == pyParams.count, "\(variant): \(sameBytes)/\(pyParams.count) parameters bit-equal")
            // Module quantization: exactly Python's per module.
            let pyQuant = try #require(py["quantization"] as? [String: Any])
            let modules = Dictionary(uniqueKeysWithValues: drafter.leafModules().flattened())
            for (path, entry) in pyQuant {
                let module = try #require(modules[path], "\(variant): no module \(path)")
                if let e = entry as? [String: Any] {
                    let (mode, g, b): (QuantizationMode?, Int?, Int?) =
                        (module as? QuantizedLinear).map { ($0.mode, $0.groupSize, $0.bits) }
                        ?? (module as? QuantizedEmbedding).map { ($0.mode, $0.groupSize, $0.bits) } ?? (nil, nil, nil)
                    #expect(mode?.rawValue == e["mode"] as? String, "\(variant) \(path) mode")
                    #expect(g == (e["group_size"] as? NSNumber)?.intValue && b == (e["bits"] as? NSNumber)?.intValue)
                } else {
                    #expect(!(module is QuantizedLinear) && !(module is QuantizedEmbedding), "\(variant) \(path)")
                }
            }

            // One context of 9 rows (> window 6), one random block, anchor 5.
            let single = try #require(py["single"] as? [String: Any])
            var context = MiniMaxM3DSparkContext.empty(at: 0)
            context.append(drafter.contextRows(inputs.feats, position: 0))
            let h = drafter.block(inputs.block, start: context.end, context: context)
            #expect(Self.digest(h) == single["h"] as? String, "\(variant): block hidden")
            #expect(Self.digest(inputs.head(h)) == single["logits"] as? String, "\(variant): head logits")
            for k in [16, 32, 0] {
                let (tokens, confidences) = drafter.markovDraft(
                    hidden: h, anchor: 5, head: { inputs.head($0) }, count: config.blockSize, candidates: k,
                    confidences: true)
                #expect(tokens.asArray(UInt32.self).map(Int.init) == (single["tokens\(k)"] as? [Int]), "\(variant) K\(k)")
                #expect(Self.digest(try #require(confidences)) == single["conf\(k)"] as? String, "\(variant) conf K\(k)")
            }

            // A proposal sequence: a context from position 17 grown in chunks (kept to the window, as a session keeps
            // it), blocks of [anchor, mask...] through the toy embedding, span = max(block, count), K = 32.
            let rounds = try #require(py["rounds"] as? [[String: Any]])
            let base = try #require(fixture["context_base"] as? Int)
            let head = MiniMaxM3DSparkHead(
                config: config, drafter: drafter, embed: { inputs.table($0) }, head: { inputs.head($0) },
                candidates: 32)
            context = .empty(at: base)
            for (i, round) in rounds.enumerated() {
                context.append(drafter.contextRows(inputs.chunks[i], position: context.end))
                context.keepLast(config.slidingWindow)
                let anchor = try #require(round["anchor"] as? Int)
                let count = try #require(round["count"] as? Int)
                let span = max(config.blockSize, count)
                #expect(span == round["span"] as? Int && context.end == round["start"] as? Int)
                var ids = [Int32](repeating: Int32(config.maskTokenId), count: span)
                ids[0] = Int32(anchor)
                let hh = drafter.block(inputs.table(MLXArray(ids).reshaped(1, span)), start: context.end, context: context)
                let (tokens, confidences) = drafter.markovDraft(
                    hidden: hh, anchor: anchor, head: { inputs.head($0) }, count: count, candidates: 32,
                    confidences: true)
                let want = round["tokens"] as? [Int]
                #expect(Self.digest(hh) == round["h"] as? String, "\(variant) round \(i): block hidden")
                #expect(tokens.asArray(UInt32.self).map(Int.init) == want, "\(variant) round \(i): tokens")
                #expect(Self.digest(try #require(confidences)) == round["conf"] as? String, "\(variant) round \(i)")
                let proposed = head.propose(anchor: anchor, start: context.end, count: count, context: context)
                #expect(proposed.asArray(UInt32.self).map(Int.init) == want, "\(variant) round \(i): head.propose")
            }
            print("[DSpark parity] \(variant): \(sameBytes)/\(pyParams.count) parameters, single + \(rounds.count) rounds")
        }
    }

    @Test("Sliding window: rows the window never reads do not matter; one more or every row visible, or a shift, do")
    func slidingWindowControls() throws {
        let fixture = try Self.parityFixture()
        let drafters = try #require(fixture["drafters"] as? [String: Any])
        let py = try #require(drafters["release"] as? [String: Any])
        let rounds = try #require(py["rounds"] as? [[String: Any]])
        let inputs = try Self.parityInputs(fixture)
        try Device.withDefaultDevice(.gpu) {
            let (config, drafter) = try MiniMaxM3DSparkLoader.load(
                directory: Self.fixtures.appendingPathComponent("dspark/release"))
            // Round 3 of the proposal sequence (a 7-row block after 13 context rows from position 17).
            func hidden(keep: Int = config.slidingWindow, window: Int? = nil, shift: Int = 0) -> String {
                var context = MiniMaxM3DSparkContext.empty(at: 17)
                for i in 0 ..< 3 {
                    context.append(drafter.contextRows(inputs.chunks[i], position: context.end))
                }
                context.keepLast(keep)
                context.start += shift
                context.end += shift
                var ids = [Int32](repeating: Int32(config.maskTokenId), count: 7)
                ids[0] = 99
                return Self.digest(
                    drafter.block(
                        inputs.table(MLXArray(ids).reshaped(1, 7)), start: context.end, context: context, window: window))
            }
            let want = try #require(rounds[2]["h"] as? String)
            #expect(hidden() == want)  // the session's context (window rows) == Python's (all 13 rows)
            #expect(hidden(keep: 13) == want)  // every row held: the mask still hides the same ones
            #expect(hidden(keep: config.slidingWindow - 1) == want)  // the oldest held row is never read
            // Controls that must change the block: the window one row wider, no window at all, the block one late.
            #expect(hidden(keep: 13, window: config.slidingWindow + 1) != want)
            #expect(hidden(keep: 13, window: 0) != want)
            #expect(hidden(shift: 1) != want)
            // and one row too few held (a row the window reads is missing)
            #expect(hidden(keep: config.slidingWindow - 2) != want)
        }
    }

    // MARK: - Synthetic M3 target

    static let D = 512, vocabulary = 1024, layers = 4, experts = 8, topK = 4
    static let draftTaps = [1, 3], draftBlock = 4, draftWindow = 64

    static func m3Configuration(block: Int = 32, topk: Int = 6) throws -> MiniMaxM3Configuration {
        let text: [String: Any] = [
            "model_type": "minimax_m3", "hidden_size": D, "num_hidden_layers": layers, "intermediate_size": 256,
            "dense_intermediate_size": 512, "shared_intermediate_size": 256, "num_attention_heads": 8,
            "num_key_value_heads": 2, "head_dim": 64, "rotary_dim": 32, "rope_theta": 10_000.0, "rms_norm_eps": 1e-6,
            "vocab_size": vocabulary, "num_local_experts": experts, "num_experts_per_tok": topK,
            "routed_scaling_factor": 2.0, "norm_topk_prob": true, "swiglu_alpha": 1.702, "swiglu_limit": 7.0,
            "moe_layer_freq": [0, 0, 1, 1], "tie_word_embeddings": false,
            "sparse_attention_config": [
                "sparse_num_index_heads": 2, "sparse_index_dim": 64, "sparse_block_size": block,
                "sparse_topk_blocks": topk, "sparse_local_block": 1, "sparse_attention_freq": [0, 1, 1, 1],
            ],
        ]
        let json: [String: Any] = [
            "model_type": "minimax_m3_vl", "architectures": ["MiniMaxM3SparseForConditionalGeneration"],
            "text_config": text,
        ]
        return try JSONDecoder.json5().decode(
            MiniMaxM3Configuration.self, from: JSONSerialization.data(withJSONObject: json))
    }

    /// The per-module plan of an M3.1 JANGHT trunk (the Python test's `_plan`).
    static func plan(_ path: String, _ module: Module) -> (groupSize: Int, bits: Int, mode: QuantizationMode)? {
        if path.hasSuffix("mlp.gate") { return nil }  // float32 router, as released
        if path.contains("switch_mlp") { return module is SwitchLinear ? (64, 4, .affine) : nil }
        guard module is Linear || module is Embedding else { return nil }
        if (path.hasSuffix("q_proj") || path.hasSuffix("o_proj")) && !path.contains("index") { return (32, 8, .mxfp8) }
        if (path.hasSuffix("k_proj") || path.hasSuffix("v_proj")) && !path.contains("index") { return (64, 8, .affine) }
        if path.contains("index_") || path == "lm_head" { return (64, 6, .affine) }
        if path.hasSuffix("embed_tokens") { return (64, 8, .affine) }
        return (64, 4, .affine)
    }

    /// The synthetic target: random weights like the Python test's `_model`, then the per-module plan.
    static func m3Model(seed: UInt64 = 0, block: Int = 32, topk: Int = 6) throws -> MiniMaxM3Model {
        let model = MiniMaxM3Model(try m3Configuration(block: block, topk: topk))
        var key = MLXRandom.key(seed)
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

    static func ids(_ n: Int, seed: UInt64) -> [Int] {
        MLXRandom.randInt(0 ..< Int32(vocabulary - 1), [n], key: MLXRandom.key(seed)).asArray(Int32.self).map(Int.init)
    }

    static func tokens(_ ids: [Int]) -> MLXArray { MLXArray(ids.map { Int32($0) }).reshaped(1, ids.count) }

    /// Prompt prefill in chunks of `step` (the last chunk never a lone token); logits of the last row.
    @discardableResult
    static func prefill(_ model: MiniMaxM3Model, _ prompt: [Int], cache: [KVCache], step: Int = 256) -> MLXArray {
        var position = 0
        var last = MLXArray(0)
        while position < prompt.count {
            var n = min(step, prompt.count - position)
            if prompt.count - (position + n) == 1 && n > 1 { n -= 1 }
            last = model(tokens(Array(prompt[position ..< (position + n)])), cache: cache)
            eval(last)
            eval(cache.flatMap(\.state))
            position += n
        }
        return last[0, -1]
    }

    static func decode(_ model: MiniMaxM3Model, _ token: Int, cache: [KVCache]) -> MLXArray {
        let logits = model(tokens([token]), cache: cache)[0, -1]
        eval(logits)
        return logits
    }

    /// The verify forward of `rows` (armed on the cache's holder): logits `[T, V]`.
    static func verify(_ model: MiniMaxM3Model, _ rows: [Int], cache: [KVCache]) throws -> MLXArray {
        let holder = try #require(MiniMaxM3DSpark.tapHolder(cache))
        holder.dsparkVerifyArmed = true
        holder.dsparkVerifyStart = holder.offset
        defer {
            holder.dsparkVerifyArmed = false
            holder.dsparkVerifyStart = nil
        }
        let logits = model(tokens(rows), cache: cache)[0]
        eval(logits)
        return logits
    }

    static func argmax(_ logits: MLXArray) -> Int { argMax(logits.reshaped(-1)).item(Int.self) }

    static func equal(_ a: MLXArray, _ b: MLXArray) -> Bool { a.shape == b.shape && arrayEqual(a, b).item(Bool.self) }

    /// Every difference between two M3 cache lists: offsets, lane lengths vs offsets, index-lane lockstep, logical
    /// contents of every lane, and the tap tails over the positions both hold. Fails closed (empty lanes are problems).
    static func cacheProblems(_ a: [KVCache], _ b: [KVCache]) -> [String] {
        var problems: [String] = []
        guard a.count == b.count, !a.isEmpty else { return ["layer count \(a.count) vs \(b.count)"] }
        for (li, (x, y)) in zip(a, b).enumerated() {
            guard x.offset == y.offset else {
                problems.append("layer \(li): offset \(x.offset) != \(y.offset)")
                continue
            }
            for (c, tag) in [(x, "dspark"), (y, "reference")] {
                if let s = c as? MiniMaxM3SparseCache, s.indexRows != s.offset {
                    problems.append("layer \(li) \(tag): index lane \(s.indexRows) != K/V lane \(s.offset)")
                }
            }
            let (sa, sb) = (x.state, y.state)
            guard sa.count == sb.count, sa.count == ((x is MiniMaxM3SparseCache) ? 3 : 2) else {
                problems.append("layer \(li): state of \(sa.count) / \(sb.count) lanes")
                continue
            }
            for (name, (p, q)) in zip(["keys", "values", "index keys"], zip(sa, sb)) {
                if p.size == 0 || q.size == 0 {
                    problems.append("layer \(li) \(name): empty")
                } else if p.dim(2) != x.offset || q.dim(2) != y.offset {
                    problems.append("layer \(li) \(name): logical rows \(p.dim(2)) / \(q.dim(2)) != offset \(x.offset)")
                } else if !equal(p, q) {
                    problems.append("layer \(li) \(name): contents differ")
                }
            }
        }
        if let ta = MiniMaxM3DSpark.tapHolder(a)?.dsparkTail, let tb = MiniMaxM3DSpark.tapHolder(b)?.dsparkTail {
            let lo = max(ta.startPosition, tb.startPosition)
            let hi = min(ta.endPosition, tb.endPosition)
            if ta.endPosition != tb.endPosition { problems.append("tap tail ends \(ta.endPosition) / \(tb.endPosition)") }
            if lo < hi, let ra = ta.rows(from: lo, to: hi), let rb = tb.rows(from: lo, to: hi), !equal(ra, rb) {
                problems.append("tap tail rows \(lo) ..< \(hi) differ")
            }
        }
        return problems
    }

    /// True when a decode step at the cache's current length selects a strict subset of index blocks.
    static func sparseActive(_ model: MiniMaxM3Model, cache: [KVCache]) -> Bool {
        guard let sparse = cache[1] as? MiniMaxM3SparseCache, let indexKeys = sparse.state.last,
            let indexer = model.model.layers[1].selfAttn.indexer
        else { return false }
        let x = MLXArray.zeros([1, 1, D], dtype: .bfloat16)
        let (iq, _) = indexer.project(x, offset: sparse.offset)
        return indexer.select(queries: iq, keys: indexKeys, offset: sparse.offset) != nil
    }

    // MARK: - Verify == decode

    @Test("Verify rows == sequential one-token decode steps, bit for bit (logits, taps, every cache lane)",
          arguments: [(773, 2), (765, 8), (509, 5), (1021, 9), (771, 15), (300, 17)])
    func verifyRowsEqualDecodeSteps(promptLength: Int, width: Int) throws {
        try Device.withDefaultDevice(.gpu) {
            let model = try Self.m3Model()
            model.model.dsparkTapLayers = Self.draftTaps
            model.model.dsparkTailCapacity = Self.draftWindow
            let prompt = Self.ids(promptLength, seed: UInt64(promptLength))
            let rows = Self.ids(width, seed: UInt64(7 * width))
            MiniMaxM3RowExact.counters.reset()
            let cache = model.newCache(parameters: nil)
            Self.prefill(model, prompt, cache: cache)
            let logits = try Self.verify(model, rows, cache: cache)
            let counters = MiniMaxM3RowExact.counters.snapshot
            let reference = model.newCache(parameters: nil)
            Self.prefill(model, prompt, cache: reference)
            if promptLength > 64 { #expect(Self.sparseActive(model, cache: reference)) }
            let steps = rows.map { Self.decode(model, $0, cache: reference) }
            #expect(logits.shape == [width, Self.vocabulary] && abs(logits).max().item(Float.self) > 0)
            let bad = (0 ..< width).filter { !Self.equal(logits[$0], steps[$0]) }
            #expect(bad.isEmpty, "verify rows \(bad) differ from decode steps")
            let problems = Self.cacheProblems(cache, reference)
            #expect(problems.isEmpty, "\(problems.prefix(4))")
            // Both projection routes ran (multi-row exact kernels and one-row calls); routed experts batched only
            // below MiniMaxM3SwitchGLU's sort threshold.
            #expect((counters["linear_batched"] ?? 0) > 0 && (counters["linear_rows"] ?? 0) > 0, "\(counters)")
            if width * Self.topK >= 64 {
                #expect((counters["routed_rows"] ?? 0) > 0 && counters["routed_batched"] == nil, "\(counters)")
            } else {
                #expect((counters["routed_batched"] ?? 0) > 0 && counters["routed_rows"] == nil, "\(counters)")
            }
            print("[DSpark verify] prompt \(promptLength) width \(width): \(width - bad.count)/\(width) rows bit-equal, "
                + "cache problems \(problems.count), \(counters)")
        }
    }

    @Test("Control: the model's own multi-row forward is NOT row-exact (so the verify check can fail)")
    func controlStockMultiRowForwardIsNotRowExact() throws {
        try Device.withDefaultDevice(.gpu) {
            let model = try Self.m3Model()
            let prompt = Self.ids(765, seed: 765)
            let rows = Self.ids(8, seed: 56)
            let cache = model.newCache(parameters: nil)
            Self.prefill(model, prompt, cache: cache)
            let stock = model(Self.tokens(rows), cache: cache)[0]
            let reference = model.newCache(parameters: nil)
            Self.prefill(model, prompt, cache: reference)
            let steps = rows.map { Self.decode(model, $0, cache: reference) }
            let differing = (0 ..< 8).filter { !Self.equal(stock[$0], steps[$0]) }.count
            print("[DSpark verify] control: stock 8-row forward differs from decode steps on \(differing)/8 rows")
            #expect(differing >= 4)
        }
    }

    // MARK: - Rollback

    /// (prompt, verify width, kept): kept rows land mid a 32-key block and across 128 / 256 / 512 / 1024.
    static let rollbacks = [(765, 8, 2), (765, 8, 6), (509, 5, 1), (1021, 9, 4), (771, 3, 3)]

    static func rollbackCase(
        _ model: MiniMaxM3Model, prompt promptLength: Int, width: Int, kept: Int,
        rollback: ([KVCache], Int, Int) throws -> Void
    ) throws -> (cache: [KVCache], reference: [KVCache], follow: [Int]) {
        let prompt = ids(promptLength, seed: UInt64(promptLength + 1))
        let rows = ids(width, seed: UInt64(promptLength + 2))
        let follow = ids(3, seed: UInt64(promptLength + 3))
        let cache = model.newCache(parameters: nil)
        prefill(model, prompt, cache: cache)
        _ = try verify(model, rows, cache: cache)
        try rollback(cache, kept, width)
        let reference = model.newCache(parameters: nil)
        prefill(model, prompt, cache: reference)
        for r in 0 ..< kept { _ = decode(model, rows[r], cache: reference) }
        return (cache, reference, follow)
    }

    @Test("Rollback (the session's in-place commit) == a cache that only saw the accepted tokens; next rows exact",
          arguments: rollbacks)
    func rollbackEqualsACacheThatNeverSawTheRejectedTokens(promptLength: Int, width: Int, kept: Int) throws {
        try Device.withDefaultDevice(.gpu) {
            let model = try Self.m3Model()
            model.model.dsparkTapLayers = Self.draftTaps
            model.model.dsparkTailCapacity = Self.draftWindow
            let head = MiniMaxM3DSparkHead(
                config: try Self.drafterConfig(), drafter: MiniMaxM3DSparkDrafter(try Self.drafterConfig()),
                embed: { $0 }, head: { $0 })
            let session = MiniMaxM3DSparkSession(head: head)
            let (cache, reference, follow) = try Self.rollbackCase(
                model, prompt: promptLength, width: width, kept: kept
            ) { cache, kept, total in
                let holder = try #require(MiniMaxM3DSpark.tapHolder(cache))
                holder.dsparkVerifyStart = holder.offset - total
                let checkpoint = try #require(SpeculativeCacheListCheckpoint(cache))
                #expect(session.commitVerifiedPrefix(cache: cache, checkpoint: checkpoint, keep: kept))
                session.endVerifyCapture(cache: cache)
            }
            #expect(cache[0].offset == promptLength + kept)
            #expect((promptLength + kept) % 128 != 0)
            var problems = Self.cacheProblems(cache, reference)
            #expect(problems.isEmpty, "\(problems.prefix(4))")
            for t in follow {
                #expect(Self.equal(Self.decode(model, t, cache: cache), Self.decode(model, t, cache: reference)))
            }
            problems = Self.cacheProblems(cache, reference)
            #expect(problems.isEmpty, "after the follow-up steps: \(problems.prefix(4))")
        }
    }

    @Test("Controls: a rollback that skips the dense layer, or the index lane, is caught")
    func controlBrokenRollbacksAreCaught() throws {
        try Device.withDefaultDevice(.gpu) {
            let model = try Self.m3Model()
            // Layer 0 (plain K / V cache) left un-rolled-back: the invariant reports it and the next steps change
            // (the dense layer attends the stale rows at shifted positions).
            let (cache, reference, follow) = try Self.rollbackCase(model, prompt: 765, width: 8, kept: 2) {
                cache, kept, total in
                for c in cache.dropFirst() { _ = c.trim(total - kept) }
            }
            #expect(Self.cacheProblems(cache, reference).contains { $0.hasPrefix("layer 0") })
            let changed = follow.contains { t in
                !Self.equal(Self.decode(model, t, cache: cache), Self.decode(model, t, cache: reference))
            }
            #expect(changed)
            // K / V rolled back but not the index-key lane of the sparse layers: the invariant reports the lane
            // desync (a decode step on it stops at the cache's lockstep precondition, not run here).
            let (cache2, reference2, _) = try Self.rollbackCase(model, prompt: 765, width: 8, kept: 2) {
                cache, kept, total in
                for c in cache {
                    if let s = c as? MiniMaxM3SparseCache { _ = s.kv.trim(total - kept) } else { _ = c.trim(total - kept) }
                }
            }
            #expect(Self.cacheProblems(cache2, reference2).contains { $0.contains("index lane") })
        }
    }

    // MARK: - Drafter for the synthetic target

    static func drafterRaw(block: Int = draftBlock, window: Int = draftWindow, extra: [String: Any] = [:])
        -> [String: Any]
    {
        var text: [String: Any] = [
            "hidden_size": D, "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
            "head_dim": 64, "rotary_dim": 32, "rms_norm_eps": 1e-6, "vocab_size": vocabulary,
            "dspark_ffn_hidden_size": 256, "swiglu_alpha": 1.702, "swiglu_limit": 7.0, "sliding_window": window,
            "dspark_block_size": block, "rope_theta": 10_000.0, "dspark_noise_token_id": vocabulary - 1,
            "dspark_target_layer_ids": draftTaps, "dspark_markov_rank": 16,
            "layer_types": ["sliding_attention", "sliding_attention"], "dense_intermediate_size": 256,
        ]
        text.merge(extra) { _, new in new }
        return [
            "architectures": ["DSparkMiniMaxDraftModel"], "enable_confidence_head": true,
            "confidence_head_with_markov": true, "text_config": text,
        ]
    }

    static func drafterConfig() throws -> MiniMaxM3DSparkConfig { try MiniMaxM3DSparkConfig(raw: drafterRaw()) }

    /// A drafter directory in the release layout (language_model.model.dspark.* names, decoder linears MXFP8 as e4m3
    /// bytes + e8m0 weight_scale_inv, fc / norms / Markov / confidence heads BF16), random weights.
    static func writeReleaseDrafter(_ directory: URL, raw: [String: Any] = drafterRaw(), seed: UInt64 = 0) throws {
        let t = raw["text_config"] as! [String: Any]
        let (hid, rank, ffn) = (t["hidden_size"] as! Int, t["dspark_markov_rank"] as! Int, t["dspark_ffn_hidden_size"] as! Int)
        let (heads, kv, hd) = (t["num_attention_heads"] as! Int, t["num_key_value_heads"] as! Int, t["head_dim"] as! Int)
        let vocab = t["vocab_size"] as! Int
        let taps = (t["dspark_target_layer_ids"] as! [Int]).count
        var key = MLXRandom.key(seed)
        func next() -> MLXArray {
            let s = MLXRandom.split(key: key)
            key = s.0
            return s.1
        }
        func bf(_ shape: [Int], _ k: Float) -> MLXArray {
            (MLXRandom.normal(shape, key: next()) * k).asType(.bfloat16)
        }
        func mxfp8(_ o: Int, _ i: Int) -> (MLXArray, MLXArray) {
            let sign = MLXRandom.randInt(0 ..< 2, [o, i], key: next()) * 128
            let exponent = MLXRandom.randInt(5 ..< 9, [o, i], key: next()) * 8
            let mantissa = MLXRandom.randInt(0 ..< 8, [o, i], key: next())
            let w = (sign + exponent + mantissa).asType(.uint8)
            return (w, MLXRandom.randInt(118 ..< 121, [o, i / 32], key: next()).asType(.uint8))
        }
        let p = "language_model.model.dspark."
        var w: [String: MLXArray] = [
            p + "fc.weight": bf([hid, taps * hid], 0.05), p + "hidden_norm.weight": bf([hid], 0.1),
            p + "final_norm.weight": bf([hid], 0.1), p + "markov_head.markov_w1.weight": bf([vocab, rank], 0.3),
            p + "markov_head.markov_w2.weight": bf([vocab, rank], 0.3),
            p + "confidence_head.proj.weight": bf([1, hid + rank], 0.3), p + "confidence_head.proj.bias": bf([1], 0.3),
        ]
        for l in 0 ..< (t["num_hidden_layers"] as! Int) {
            let q = "\(p)layers.\(l).decoder_layer."
            for n in ["input_layernorm", "post_attention_layernorm"] { w[q + n + ".weight"] = bf([hid], 0.1) }
            for n in ["self_attn.q_norm", "self_attn.k_norm"] { w[q + n + ".weight"] = bf([hd], 0.1) }
            for (n, o, i) in [
                ("self_attn.q_proj", heads * hd, hid), ("self_attn.k_proj", kv * hd, hid),
                ("self_attn.v_proj", kv * hd, hid), ("self_attn.o_proj", hid, heads * hd), ("mlp.gate_proj", ffn, hid),
                ("mlp.up_proj", ffn, hid), ("mlp.down_proj", hid, ffn),
            ] {
                let (bytes, scales) = mxfp8(o, i)
                w[q + n + ".weight"] = bytes
                w[q + n + ".weight_scale_inv"] = scales
            }
        }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try MLX.save(arrays: w, url: directory.appendingPathComponent("model.safetensors"))
        try JSONSerialization.data(withJSONObject: raw).write(to: directory.appendingPathComponent("config.json"))
    }

    /// A pre-quantized copy in the form `write_quantized_minimax_drafter` writes (default plan): live module names,
    /// decoder-layer linears affine 4-bit (MXFP8 dequantized to BF16 first), fc affine 8-bit, heads as stored, and the
    /// per-module `quantization` block.
    static func writeQuantizedDrafter(from release: URL, to directory: URL) throws {
        let (_, drafter) = try MiniMaxM3DSparkLoader.load(directory: release)
        var raw = try MiniMaxM3DSparkLoader.rawConfig(release)
        var weights: [String: MLXArray] = [:]
        var quant: [String: Any] = ["group_size": 64, "bits": 4]
        for (path, module) in drafter.leafModules().flattened() {
            if let q = module as? QuantizedLinear {
                let dense = dequantized(
                    q.weight, scales: q.scales, biases: q.biases, groupSize: q.groupSize, bits: q.bits, mode: q.mode,
                    dtype: .bfloat16)
                let (wq, s, b) = quantized(dense, groupSize: 64, bits: 4)
                weights[path + ".weight"] = wq
                weights[path + ".scales"] = s
                weights[path + ".biases"] = b
                quant[path] = ["group_size": 64, "bits": 4]
            } else if path == "fc", let l = module as? Linear {
                let (wq, s, b) = quantized(l.weight, groupSize: 64, bits: 8)
                weights["fc.weight"] = wq
                weights["fc.scales"] = s
                weights["fc.biases"] = b
                quant["fc"] = ["group_size": 64, "bits": 8]
            }
        }
        for (name, value) in drafter.parameters().flattened() where weights[name] == nil {
            let module = String(name[..<name.lastIndex(of: ".")!])
            if quant[module] == nil { weights[name] = value }
        }
        raw["quantization"] = quant
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try MLX.save(arrays: weights, url: directory.appendingPathComponent("model.safetensors"))
        try JSONSerialization.data(withJSONObject: raw).write(to: directory.appendingPathComponent("config.json"))
    }

    static func temporaryDirectory(_ tag: String) -> URL {
        FileManager.default.temporaryDirectory.appendingPathComponent("minimax-m3-dspark-\(tag)-\(UUID().uuidString)")
    }

    // MARK: - The speculative loop

    /// A real session whose proposals are replaced by a script over the AR reference sequence: draft at absolute
    /// position p = reference[p] unless `wrong(p)` (then another token). The drafter's own context sync, block and
    /// Markov draft still run every step (Python's `_Scripted`).
    final class ScriptedSession: DSparkDraftSession {
        let inner: MiniMaxM3DSparkSession
        let reference: [Int]
        let wrong: (Int) -> Bool
        var widths: [Int] = []

        init(inner: MiniMaxM3DSparkSession, reference: [Int], wrong: @escaping (Int) -> Bool) {
            self.inner = inner
            self.reference = reference
            self.wrong = wrong
        }

        func script(_ cache: [KVCache], _ count: Int) -> [Int] {
            let start = (MiniMaxM3DSpark.tapHolder(cache)?.offset ?? 0) + 1
            widths.append(count)
            return (start ..< (start + count)).map { p in
                let token = p < reference.count ? reference[p] : 0
                return wrong(p) ? (token + 1) % MiniMaxM3DSparkTests.vocabulary : token
            }
        }

        func draft(seed: Int, cache: [KVCache]) -> [Int] { draft(seed: seed, cache: cache, count: 4) }
        func draft(seed: Int, cache: [KVCache], count: Int) -> [Int] {
            let real = inner.draft(seed: seed, cache: cache, count: count)
            precondition(real.count == count)
            return script(cache, count)
        }
        func draft(seed: Int, cache: [KVCache], count: Int, noise: MLXArray, inverseTemperature: Float) -> [Int] {
            _ = inner.draft(seed: seed, cache: cache, count: count, noise: noise, inverseTemperature: inverseTemperature)
            return script(cache, count)
        }
        var vocabularySize: Int? { inner.vocabularySize }
        var commitsAnyPrefix: Bool { inner.commitsAnyPrefix }
        func beginVerifyCapture(cache: [KVCache]) -> Bool { inner.beginVerifyCapture(cache: cache) }
        func commitVerifiedPrefix(cache: [KVCache], checkpoint: SpeculativeCacheListCheckpoint, keep: Int) -> Bool {
            inner.commitVerifiedPrefix(cache: cache, checkpoint: checkpoint, keep: keep)
        }
        func endVerifyCapture(cache: [KVCache]) { inner.endVerifyCapture(cache: cache) }
    }

    /// Greedy AR: the prompt prefilled, then `count` tokens from one-token decode steps. Returns the tokens and the
    /// cache (holding the prompt and every token but the last).
    static func greedyAR(_ model: MiniMaxM3Model, _ prompt: [Int], count: Int) -> ([Int], [KVCache]) {
        let cache = model.newCache(parameters: nil)
        var token = argmax(prefill(model, prompt, cache: cache))
        var out = [token]
        while out.count < count {
            token = argmax(decode(model, token, cache: cache))
            out.append(token)
        }
        return (out, cache)
    }

    /// Mimics TokenIterator: queue committed tokens, take an ordinary decode step on `.autoregressive`.
    static func dsparkRun(
        _ model: MiniMaxM3Model, decoder: inout DSparkDecoder, prompt: [Int], count: Int,
        rows: inout [Int]
    ) -> ([Int], [KVCache]) {
        let cache = model.newCache(parameters: nil)
        var y = argmax(prefill(model, prompt, cache: cache))
        var queue: [Int] = []
        var out: [Int] = []
        var widths: [Int] = []
        let forward: ([Int]) -> MLXArray = { ids in
            widths.append(ids.count)
            return model(tokens(ids), cache: cache)
        }
        while out.count < count {
            if !queue.isEmpty {
                out.append(queue.removeFirst())
                continue
            }
            let seed = y
            switch decoder.step(seed: seed, cache: cache, remaining: count - out.count, forward: forward) {
            case .autoregressive:
                y = argmax(forward([seed])[0, -1])
                out.append(seed)
            case .committed(let committed, let next):
                y = next.item(Int.self)
                queue = Array(committed.dropFirst())
                out.append(committed[0])
            }
        }
        decoder.finish()
        rows = widths
        // TokenIterator's finishDSpark: the cache never runs ahead of the returned tokens.
        let expected = prompt.count + count - 1
        for layer in cache where layer.offset > expected { _ = layer.trim(layer.offset - expected) }
        return (out, cache)
    }

    static let alwaysSpeculate = DSparkGovernor(probeEvery: 1, arProbeEvery: 1 << 30, warmupSpec: 1 << 30)

    static func attachedModel(_ drafter: URL, seed: UInt64 = 0, block: Int = 32, topk: Int = 6) throws
        -> MiniMaxM3Model
    {
        let model = try m3Model(seed: seed, block: block, topk: topk)
        #expect(try model.attachDSparkDraft(directory: drafter))
        return model
    }

    enum Mode: String, CaseIterable, Sendable {
        case adaptive, adaptiveWithAR, pinned2, pinned7
    }

    @Test("Greedy DSpark == greedy AR token for token, and the caches agree (Adaptive, Adaptive + AR, pinned 2 / 7)",
          arguments: [(773, 11), (517, 12), (1029, 13)])
    func greedyDSparkEqualsAR(promptLength: Int, seed: Int) throws {
        let directory = Self.temporaryDirectory("release")
        defer { try? FileManager.default.removeItem(at: directory) }
        try Self.writeReleaseDrafter(directory)
        try Device.withDefaultDevice(.gpu) {
            let model = try Self.attachedModel(directory)
            #expect(model.dsparkBlockSize == Self.draftBlock && model.dsparkLadderTop == 2 * Self.draftBlock)
            let prompt = Self.ids(promptLength, seed: UInt64(seed))
            let (reference, arCache) = Self.greedyAR(model, prompt, count: 45)
            #expect(reference.count == 45 && Set(reference).count > 5)  // a non-degenerate baseline
            for mode in Mode.allCases {
                let inner = try #require(model.makeDSparkDraftSession() as? MiniMaxM3DSparkSession)
                let scripted = ScriptedSession(inner: inner, reference: prompt + reference) { $0 % 5 == 4 }
                var decoder = DSparkDecoder(
                    session: scripted, blockSize: Self.draftBlock, stopTokenIDs: [],
                    governor: mode == .adaptiveWithAR ? DSparkGovernor() : Self.alwaysSpeculate,
                    ladderTop: model.dsparkLadderTop,
                    pinnedWidth: mode == .pinned2 ? 2 : mode == .pinned7 ? 7 : nil)
                var rows: [Int] = []
                let (out, cache) = Self.dsparkRun(model, decoder: &decoder, prompt: prompt, count: 45, rows: &rows)
                let stats = decoder.stats
                #expect(out == reference, "prompt \(promptLength) \(mode): DSpark output differs from AR")
                let problems = Self.cacheProblems(cache, arCache)
                #expect(problems.isEmpty, "prompt \(promptLength) \(mode): \(problems.prefix(3))")
                if mode != .adaptiveWithAR {
                    #expect(stats.specSteps > 3 && stats.accepted > 0 && stats.accepted < stats.drafted, "\(mode)")
                    #expect(stats.commitForwards == 0, "every rejection commits in place (no second forward)")
                }
                if mode == .pinned2 { #expect(Set(scripted.widths) == [2] || Set(scripted.widths).isSubset(of: [1, 2])) }
                if mode == .pinned7 { #expect(scripted.widths.max() == 7) }
                print("[DSpark greedy] prompt \(promptLength) \(mode): == AR \(out == reference), \(stats.summary), "
                    + "draft widths \(Dictionary(grouping: scripted.widths, by: { $0 }).mapValues(\.count))")
            }
        }
    }

    @Test("Control: the loop with the stock multi-row forward as verify does not reproduce AR")
    func controlStockVerifyBreaksGreedyEquality() throws {
        let directory = Self.temporaryDirectory("release")
        defer { try? FileManager.default.removeItem(at: directory) }
        try Self.writeReleaseDrafter(directory)
        try Device.withDefaultDevice(.gpu) {
            let model = try Self.attachedModel(directory)
            var diverged = 0
            for (promptLength, seed) in [(773, 11), (517, 12), (1029, 13)] {
                let prompt = Self.ids(promptLength, seed: UInt64(seed))
                let (reference, _) = Self.greedyAR(model, prompt, count: 45)
                let inner = try #require(model.makeDSparkDraftSession() as? MiniMaxM3DSparkSession)
                // A session that never arms the verify: the verify forward is the model's ordinary multi-row one.
                final class Unarmed: DSparkDraftSession {
                    let inner: ScriptedSession
                    init(_ inner: ScriptedSession) { self.inner = inner }
                    func draft(seed: Int, cache: [KVCache]) -> [Int] { inner.draft(seed: seed, cache: cache) }
                    func draft(seed: Int, cache: [KVCache], count: Int) -> [Int] {
                        inner.draft(seed: seed, cache: cache, count: count)
                    }
                }
                let scripted = ScriptedSession(inner: inner, reference: prompt + reference) { $0 % 5 == 4 }
                var decoder = DSparkDecoder(
                    session: Unarmed(scripted), blockSize: Self.draftBlock, stopTokenIDs: [],
                    governor: Self.alwaysSpeculate, ladderTop: 8, pinnedWidth: 7)
                var rows: [Int] = []
                let (out, _) = Self.dsparkRun(model, decoder: &decoder, prompt: prompt, count: 45, rows: &rows)
                diverged += out != reference ? 1 : 0
            }
            print("[DSpark greedy] control: stock multi-row verify diverged from AR on \(diverged)/3 prompts")
            #expect(diverged == 3)
        }
    }

    @Test("Adaptive climbs past the trained block toward the ladder top and shrinks to one draft; output == AR")
    func adaptiveWidthClimbsAndShrinks() throws {
        let directory = Self.temporaryDirectory("release")
        defer { try? FileManager.default.removeItem(at: directory) }
        // A bundle stamp that would cap the ladder if it were read: it must not be.
        try Self.writeReleaseDrafter(directory, raw: Self.drafterRaw(extra: ["recommended_num_drafts": 1]))
        try Device.withDefaultDevice(.gpu) {
            let model = try Self.attachedModel(directory)
            #expect(model.dsparkLadderTop == 2 * Self.draftBlock && model.dsparkPinnedWidth == nil)
            let prompt = Self.ids(517, seed: 51)
            let (reference, _) = Self.greedyAR(model, prompt, count: 90)
            let switchAt = prompt.count + 50
            let inner = try #require(model.makeDSparkDraftSession() as? MiniMaxM3DSparkSession)
            let scripted = ScriptedSession(inner: inner, reference: prompt + reference) { $0 >= switchAt }
            var decoder = DSparkDecoder(
                session: scripted, blockSize: Self.draftBlock, stopTokenIDs: [], governor: Self.alwaysSpeculate,
                ladderTop: model.dsparkLadderTop)
            #expect(decoder.width == Self.draftBlock)  // starts at the trained block
            let cache = model.newCache(parameters: nil)
            var y = Self.argmax(Self.prefill(model, prompt, cache: cache))
            var out: [Int] = []
            var widths: [(Int, Int)] = []
            while out.count < 90 {
                let position = cache[0].offset
                let step = decoder.step(seed: y, cache: cache, remaining: 90 - out.count) { ids in
                    if ids.count > 1 { widths.append((position, ids.count)) }
                    return model(Self.tokens(ids), cache: cache)
                }
                switch step {
                case .autoregressive:
                    out.append(y)
                    y = Self.argmax(model(Self.tokens([y]), cache: cache)[0, -1])
                case .committed(let committed, let next):
                    out += committed
                    y = next.item(Int.self)
                }
            }
            #expect(Array(out.prefix(90)) == reference)
            let early = widths.filter { $0.0 < switchAt }.map(\.1)
            let late = widths.filter { $0.0 >= switchAt + 8 }.map(\.1)
            #expect(!early.isEmpty && !late.isEmpty)
            #expect((early.max() ?? 0) - 1 > Self.draftBlock, "never drafted past the trained block: \(early)")
            #expect(Set(widths.map(\.1)).count >= 3, "the width never moved")
            #expect((late.max() ?? 99) <= 2, "did not shrink: \(late)")
            print("[DSpark adaptive] verify widths early \(early), late \(late)")
            // Diagnostic levers only: VMLX_DSPARK_MAX_DRAFT limits the ladder, VMLX_DSPARK_WIDTH pins the width.
            setenv("VMLX_DSPARK_MAX_DRAFT", "3", 1)
            setenv("VMLX_DSPARK_WIDTH", "2", 1)
            #expect(model.dsparkLadderTop == 3 && model.dsparkPinnedWidth == 2)
            unsetenv("VMLX_DSPARK_MAX_DRAFT")
            unsetenv("VMLX_DSPARK_WIDTH")
            #expect(model.dsparkLadderTop == 8 && model.dsparkPinnedWidth == nil)
        }
    }

    @Test("Released geometry (blocks of 128, top-16) past 2048 tokens: verify exact, rollback mid-block, greedy == AR")
    func releasedGeometryPast2048Tokens() throws {
        let directory = Self.temporaryDirectory("release")
        defer { try? FileManager.default.removeItem(at: directory) }
        try Self.writeReleaseDrafter(directory)
        try Device.withDefaultDevice(.gpu) {
            let model = try Self.attachedModel(directory, seed: 4, block: 128, topk: 16)
            let prompt = Self.ids(2429, seed: 31)
            let rows = Self.ids(8, seed: 32)
            let cache = model.newCache(parameters: nil)
            Self.prefill(model, prompt, cache: cache, step: 1024)
            #expect(Self.sparseActive(model, cache: cache))
            let logits = try Self.verify(model, rows, cache: cache)
            let session = try #require(model.makeDSparkDraftSession())
            let holder = try #require(MiniMaxM3DSpark.tapHolder(cache))
            holder.dsparkVerifyStart = 2429
            #expect(session.commitVerifiedPrefix(
                cache: cache, checkpoint: try #require(SpeculativeCacheListCheckpoint(cache)), keep: 2))
            session.endVerifyCapture(cache: cache)
            #expect(cache[0].offset == 2431)  // mid-block, below 2432
            let reference = model.newCache(parameters: nil)
            Self.prefill(model, prompt, cache: reference, step: 1024)
            let steps = (0 ..< 2).map { Self.decode(model, rows[$0], cache: reference) }
            #expect(Self.equal(logits[0], steps[0]) && Self.equal(logits[1], steps[1]))
            #expect(Self.cacheProblems(cache, reference).isEmpty)
            let (ar, _) = Self.greedyAR(model, prompt, count: 24)
            let inner = try #require(model.makeDSparkDraftSession() as? MiniMaxM3DSparkSession)
            let scripted = ScriptedSession(inner: inner, reference: prompt + ar) { $0 % 5 == 4 }
            var decoder = DSparkDecoder(
                session: scripted, blockSize: Self.draftBlock, stopTokenIDs: [], governor: Self.alwaysSpeculate,
                ladderTop: model.dsparkLadderTop)
            var widths: [Int] = []
            let (out, _) = Self.dsparkRun(model, decoder: &decoder, prompt: prompt, count: 24, rows: &widths)
            #expect(out == ar)
            #expect(decoder.stats.accepted > 0 && decoder.stats.accepted < decoder.stats.drafted)
        }
    }

    @Test("Sampled DSpark (coupled noise) gives the position-keyed AR samples, scripted partial accepts")
    func sampledDSparkEqualsPositionKeyedAR() throws {
        let directory = Self.temporaryDirectory("release")
        defer { try? FileManager.default.removeItem(at: directory) }
        try Self.writeReleaseDrafter(directory)
        try Device.withDefaultDevice(.gpu) {
            let model = try Self.attachedModel(directory)
            let prompt = Self.ids(773, seed: 61)
            let sampling = try #require(DSparkSampling(temperature: 0.8, topP: 0.95, seed: 7))
            // AR reference: every token from its own position's noise row.
            let arCache = model.newCache(parameters: nil)
            var token = sampling.sample(Self.prefill(model, prompt, cache: arCache), position: prompt.count).item(Int.self)
            var reference = [token]
            while reference.count < 40 {
                let position = prompt.count + reference.count
                token = sampling.sample(Self.decode(model, token, cache: arCache), position: position).item(Int.self)
                reference.append(token)
            }
            #expect(Set(reference).count > 5)
            let inner = try #require(model.makeDSparkDraftSession() as? MiniMaxM3DSparkSession)
            let scripted = ScriptedSession(inner: inner, reference: prompt + reference) { $0 % 4 == 3 }
            var decoder = DSparkDecoder(
                session: scripted, blockSize: Self.draftBlock, stopTokenIDs: [], sampling: sampling,
                governor: Self.alwaysSpeculate, ladderTop: model.dsparkLadderTop)
            let cache = model.newCache(parameters: nil)
            var y = sampling.sample(Self.prefill(model, prompt, cache: cache), position: prompt.count).item(Int.self)
            var out: [Int] = []
            while out.count < 40 {
                switch decoder.step(seed: y, cache: cache, remaining: 40 - out.count, forward: { ids in
                    model(Self.tokens(ids), cache: cache)
                }) {
                case .autoregressive:
                    out.append(y)
                    let logits = model(Self.tokens([y]), cache: cache)[0, -1]
                    y = sampling.sample(logits, position: cache[0].offset).item(Int.self)
                case .committed(let committed, let next):
                    out += committed
                    y = next.item(Int.self)
                }
            }
            #expect(Array(out.prefix(40)) == reference)
            #expect(decoder.stats.sampled && decoder.stats.accepted > 0 && decoder.stats.accepted < decoder.stats.drafted)
        }
    }

    // MARK: - Serving: bundle wiring, attach, TokenIterator

    @Test("A bundle with dspark/ gets the DSpark runtime; levers only turn it off; a foreign sidecar is not a drafter")
    func bundleWiring() throws {
        let root = Self.temporaryDirectory("bundle")
        defer { try? FileManager.default.removeItem(at: root) }
        try Self.writeReleaseDrafter(root.appendingPathComponent("dspark"))
        let config = try JSONSerialization.data(withJSONObject: [
            "model_type": "minimax_m3_vl", "text_config": ["model_type": "minimax_m3"],
        ])
        let settings = VMLXServerRuntimeSettings()
        #expect(
            settings.bundledDSparkDrafter(configData: config, bundleDirectory: root, environment: [:])?.lastPathComponent
                == "dspark")
        #expect(settings.bundledDSparkDrafter(configData: config, bundleDirectory: root, environment: ["VMLX_DSPARK": "0"]) == nil)
        #expect(
            settings.bundledDSparkDrafter(configData: config, bundleDirectory: root, environment: ["VMLX_DSV41_DSPARK": "off"])
                == nil)
        if case .dspark(let path) = settings.resolvedMTPDraftStrategy(
            configData: config, jangConfig: nil, status: nil, bundleDirectory: root)
        {
            #expect(path.lastPathComponent == "dspark")
        } else {
            Issue.record("a MiniMax-M3 bundle with dspark/ did not resolve to the dSpark strategy")
        }
        // A dspark/ whose config.json does not declare DSpark is not a drafter.
        try JSONSerialization.data(withJSONObject: ["architectures": ["Qwen3ForCausalLM"]])
            .write(to: root.appendingPathComponent("dspark/config.json"))
        #expect(settings.bundledDSparkDrafter(configData: config, bundleDirectory: root, environment: [:]) == nil)
        // Another family's bundle: no MiniMax drafter.
        let other = try JSONSerialization.data(withJSONObject: ["model_type": "qwen3"])
        #expect(settings.bundledDSparkDrafter(configData: other, bundleDirectory: root, environment: [:]) == nil)
    }

    @Test("Attach refuses a drafter that does not fit the target (hidden size, vocabulary, taps); a foreign one is skipped")
    func attachChecksGeometry() throws {
        try Device.withDefaultDevice(.gpu) {
            for (extra, why) in [
                (["hidden_size": 256], "hidden"), (["vocab_size": 512, "dspark_noise_token_id": 511], "vocabulary"),
                (["dspark_target_layer_ids": [1, 4]], "taps"),
            ] as [([String: Any], String)] {
                let directory = Self.temporaryDirectory(why)
                defer { try? FileManager.default.removeItem(at: directory) }
                try Self.writeReleaseDrafter(directory, raw: Self.drafterRaw(extra: extra))
                let model = try Self.m3Model()
                #expect(throws: (any Error).self, "\(why)") { try model.attachDSparkDraft(directory: directory) }
                #expect(model.dsparkBlockSize == nil && model.model.dsparkTapLayers.isEmpty)
            }
            let foreign = Self.temporaryDirectory("foreign")
            defer { try? FileManager.default.removeItem(at: foreign) }
            try FileManager.default.createDirectory(at: foreign, withIntermediateDirectories: true)
            try JSONSerialization.data(withJSONObject: ["architectures": ["Qwen3DSparkModel"]])
                .write(to: foreign.appendingPathComponent("config.json"))
            let model = try Self.m3Model()
            #expect(try model.attachDSparkDraft(directory: foreign) == false)
        }
    }

    @Test("TokenIterator with .dspark: the real drafter (release MXFP8 and pre-quantized) returns exactly the AR tokens")
    func tokenIteratorWithTheRealDrafter() throws {
        let release = Self.temporaryDirectory("release")
        let quantized = Self.temporaryDirectory("quantized")
        defer {
            try? FileManager.default.removeItem(at: release)
            try? FileManager.default.removeItem(at: quantized)
        }
        try Self.writeReleaseDrafter(release)
        try Device.withDefaultDevice(.gpu) {
            try Self.writeQuantizedDrafter(from: release, to: quantized)
            let (_, q) = try MiniMaxM3DSparkLoader.load(directory: quantized)
            #expect((q.layers[0].selfAttn.qProj as? QuantizedLinear)?.bits == 4 && (q.fc as? QuantizedLinear)?.bits == 8)
            let prompt = Self.ids(773, seed: 21)
            let input = LMInput(tokens: MLXArray(prompt.map { Int32($0) }))
            for drafter in [release, quantized] {
                let model = try Self.m3Model()
                var ar = try TokenIterator(
                    input: input, model: model, parameters: GenerateParameters(maxTokens: 30, temperature: 0))
                var reference: [Int] = []
                while let t = ar.next() { reference.append(t) }
                #expect(reference.count == 30 && ar.dsparkGenerationStats == nil)
                var parameters = GenerateParameters(maxTokens: 30, temperature: 0)
                parameters.draftStrategy = .dspark(drafterPath: drafter)
                var it = try TokenIterator(input: input, model: model, parameters: parameters)
                var out: [Int] = []
                while let t = it.next() { out.append(t) }
                let stats = try #require(it.dsparkGenerationStats)
                #expect(out == reference, "\(drafter.lastPathComponent): DSpark output differs from AR")
                #expect(stats.specSteps > 0)
                // The cache ends where AR's does, with the same contents.
                let problems = Self.cacheProblems(it.cache, ar.cache)
                #expect(problems.isEmpty, "\(problems.prefix(3))")
                print("[DSpark TokenIterator] \(drafter.lastPathComponent.prefix(30)): == AR \(out == reference), \(stats.summary)")
            }
        }
    }

    @Test("JANGHT target (JANGT routed experts, mixed MXFP8 / affine trunk): verify rows == decode steps")
    func verifyWithJANGTExpertsEqualsDecodeSteps() async throws {
        let directory = try MiniMaxM3JANGHTTests.bundleCopy()
        defer { try? FileManager.default.removeItem(at: directory) }
        let context = try await LLMModelFactory.shared.load(
            from: directory, using: Self.FixtureTokenizerLoader())
        let model = try #require(context.model as? MiniMaxM3Model)
        #expect(model.model.layers.contains { ($0.mlp as? MiniMaxM3SparseMoeBlock)?.experts is JANGTRoutedExpertLayer })
        try Device.withDefaultDevice(.gpu) {
            let vocab = model.vocabularySize
            let prompt = MLXRandom.randInt(0 ..< Int32(vocab - 1), [2117], key: MLXRandom.key(5)).asArray(Int32.self)
                .map(Int.init)
            for width in [3, 8, 17, 19] {
                let rows = MLXRandom.randInt(0 ..< Int32(vocab - 1), [width], key: MLXRandom.key(UInt64(width)))
                    .asArray(Int32.self).map(Int.init)
                MiniMaxM3RowExact.counters.reset()
                let cache = model.newCache(parameters: nil)
                Self.prefill(model, prompt, cache: cache, step: 512)
                let logits = try Self.verify(model, rows, cache: cache)
                let reference = model.newCache(parameters: nil)
                Self.prefill(model, prompt, cache: reference, step: 512)
                let steps = rows.map { Self.decode(model, $0, cache: reference) }
                let bad = (0 ..< width).filter { !Self.equal(logits[$0], steps[$0]) }
                #expect(bad.isEmpty, "width \(width): rows \(bad) differ")
                #expect(Self.cacheProblems(cache, reference).isEmpty, "width \(width)")
                print("[DSpark verify] JANGHT width \(width): \(width - bad.count)/\(width) rows bit-equal, "
                    + "\(MiniMaxM3RowExact.counters.snapshot)")
            }
        }
    }
}
