// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXHuggingFace
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon
@preconcurrency import VMLXTokenizers

/// LFM3 text runtime: config, rotary layout, weight routing, cache behaviour, parsers, and
/// (opt-in) logit parity with a released bundle.
@Suite("LFM3", .serialized)
struct LFM3Tests {

    // MARK: - Config

    /// The text fields of `LiquidAI/LFM3-1.7B` `config.json`, verbatim.
    static let releasedConfig = #"""
        {"architectures": ["Lfm3ForCausalLM"], "bos_token_id": 124894, "conv_L_cache": 7,
         "conv_bias": false, "dtype": "bfloat16", "eos_token_id": [124900], "head_dim": 128,
         "hidden_size": 2048, "intermediate_size": 7680, "max_position_embeddings": 262144,
         "model_type": "lfm3", "norm_eps": 1e-05, "num_attention_heads": 16,
         "num_hidden_layers": 24, "num_key_value_heads": 4, "pad_token_id": 124893,
         "layer_types": ["conv", "conv", "full_attention", "conv", "sliding_attention", "conv",
           "full_attention", "conv", "sliding_attention", "conv", "full_attention", "conv",
           "sliding_attention", "conv", "full_attention", "conv", "sliding_attention", "conv",
           "full_attention", "conv", "sliding_attention", "conv", "full_attention", "conv"],
         "rope_parameters": {
           "full_attention": {"partial_rotary_factor": 0.5, "rope_theta": 1000000.0, "rope_type": "default"},
           "sliding_attention": {"rope_theta": 10000, "rope_type": "default"}},
         "sliding_window": 256, "tie_word_embeddings": true, "use_sandwich_norm": true,
         "vision_config": {}, "vocab_size": 128000,
         "quantization": {"group_size": 64, "bits": 4, "mode": "affine"}}
        """#

    @Test func releasedConfigDecodes() throws {
        let config = try JSONDecoder().decode(
            LFM3Configuration.self, from: Data(Self.releasedConfig.utf8))
        #expect(config.headDim == 128)
        #expect(config.convLCache == 7)
        #expect(config.slidingWindow == 256)
        #expect(config.useSandwichNorm)
        #expect(config.layerTypes.filter { $0 == "conv" }.count == 13)
        #expect(config.layerTypes.filter { $0 == "full_attention" }.count == 6)
        #expect(config.layerTypes.filter { $0 == "sliding_attention" }.count == 5)
        #expect(config.rope(for: "full_attention") == .init(ropeTheta: 1e6, partialRotaryFactor: 0.5))
        #expect(config.rope(for: "sliding_attention") == .init(ropeTheta: 1e4, partialRotaryFactor: 1))
    }

    @Test func malformedLayerPatternsAreRefused() {
        let short = Self.releasedConfig.replacingOccurrences(
            of: #""num_hidden_layers": 24"#, with: #""num_hidden_layers": 25"#)
        #expect(throws: DecodingError.self) {
            try JSONDecoder().decode(LFM3Configuration.self, from: Data(short.utf8))
        }
        let scaled = Self.releasedConfig.replacingOccurrences(
            of: #""rope_theta": 10000, "rope_type": "default""#,
            with: #""rope_theta": 10000, "rope_type": "yarn""#)
        #expect(throws: DecodingError.self) {
            try JSONDecoder().decode(LFM3Configuration.self, from: Data(scaled.utf8))
        }
    }

    // MARK: - Rotary layout

    /// The reference rotation (mlx-vlm `lfm3/language.py` `RotaryEmbedding`): zero-padded
    /// inverse frequencies, both head halves rotated against each other.
    private func referenceRotation(
        _ x: MLXArray, headDim: Int, theta: Float, partial: Float, offset: Int
    ) -> MLXArray {
        let angles = Int(Float(headDim) * partial) / 2
        let inv = (0 ..< headDim / 2).map { i -> Float in
            i < angles ? Float(pow(Double(theta), -Double(2 * i) / Double(headDim))) : 0
        }
        let positions = MLXArray((0 ..< x.dim(2)).map { Float($0 + offset) })
        var phase = positions.expandedDimensions(axis: -1) * MLXArray(inv)  // [L, d/2]
        phase = concatenated([phase, phase], axis: -1)
        let halves = split(x, parts: 2, axis: -1)
        let rotated = concatenated([-halves[1], halves[0]], axis: -1)
        return x * cos(phase) + rotated * sin(phase)
    }

    @Test(arguments: [(Float(1e6), Float(0.5)), (Float(1e4), Float(1.0))])
    func rotaryMatchesReference(theta: Float, partial: Float) {
        MLXRandom.seed(3)
        let x = MLXRandom.normal([1, 2, 5, 128])
        let rope = LFM3RoPE(
            headDim: 128, parameters: .init(ropeTheta: theta, partialRotaryFactor: partial))
        for offset in [0, 7, 1_000, 4_099] {
            let expected = referenceRotation(
                x, headDim: 128, theta: theta, partial: partial, offset: offset)
            #expect(abs(rope(x, offset: offset) - expected).max().item(Float.self) < 2e-3)
            #expect(
                abs(rope(x, offset: MLXArray(Int32(offset))) - expected).max().item(Float.self)
                    < 2e-3)
        }
    }

    /// Red control for the invariant on `LFM3RoPE`: MLX's own partial RoPE over the same
    /// rotated width pairs the wrong coordinates and is measurably different.
    @Test func mlxPartialRopeIsNotTheLFM3Layout() {
        MLXRandom.seed(4)
        let x = MLXRandom.normal([1, 1, 3, 128])
        let expected = referenceRotation(x, headDim: 128, theta: 1e6, partial: 0.5, offset: 5)
        let naive = RoPE(dimensions: 64, traditional: false, base: 1e6)(x, offset: 5)
        #expect(abs(naive - expected).max().item(Float.self) > 0.1)
    }

    // MARK: - Weights

    private func tinyModel(slidingWindow: Int = 4) throws -> LFM3Model {
        let json = """
            {"model_type": "lfm3", "vocab_size": 64, "hidden_size": 32, "intermediate_size": 48,
             "num_hidden_layers": 5, "num_attention_heads": 4, "num_key_value_heads": 2,
             "head_dim": 8, "conv_L_cache": 7, "sliding_window": \(slidingWindow),
             "layer_types": ["conv", "full_attention", "sliding_attention", "conv", "sliding_attention"],
             "rope_parameters": {"full_attention": {"partial_rotary_factor": 0.5, "rope_theta": 1000000.0},
                                 "sliding_attention": {"rope_theta": 10000}}}
            """
        MLXRandom.seed(11)
        let model = LFM3Model(
            try JSONDecoder().decode(LFM3Configuration.self, from: Data(json.utf8)))
        // Random norm gains so a skipped or misplaced norm changes the logits.
        let norms = model.parameters().flattened().filter { $0.0.hasSuffix("norm.weight") }
        try model.update(
            parameters: ModuleParameters.unflattened(
                norms.map { ($0.0, MLXRandom.uniform(low: 0.5, high: 1.5, $0.1.shape)) }),
            verify: .none)
        eval(model)
        return model
    }

    @Test func sanitizeAcceptsTheReleasedMLXLayout() throws {
        let model = try tinyModel()
        let w = MLXArray.zeros([2])
        let out = model.sanitize(weights: [
            "language_model.model.embed_tokens.weight": w,
            "language_model.model.layers.0.conv.conv.weight": MLXArray.zeros([32, 1, 7]),
            "language_model.model.layers.3.conv.conv.weight": MLXArray.zeros([32, 7, 1]),
            "language_model.lm_head.weight": w,
            "vision_tower.encoder.layers.0.mlp.fc1.weight": w,
            "multi_modal_projector.linear_1.weight": w,
        ])
        #expect(out["model.embed_tokens.weight"] != nil)
        #expect(out["model.layers.0.conv.conv.weight"]?.shape == [32, 7, 1])
        #expect(out["model.layers.3.conv.conv.weight"]?.shape == [32, 7, 1])
        #expect(out["lm_head.weight"] == nil, "tied embeddings: the head is the embedding")
        #expect(out.count == 3, "vision tower and projector are dropped from a text-only load")
    }

    // MARK: - Cache behaviour

    /// Every decode position must reproduce the uncached forward, across the sliding window
    /// (window 4, 13 tokens: not a multiple of the window) and with a multi-token chunk landing
    /// on a wrapped rotating cache.
    ///
    /// Strict with `MLX_ENABLE_TF32=0`. With the default TF32, Metal runs float32 GEMM (M >= 2)
    /// in TF32 while one-token GEMV stays fp32, so single-token steps drift ~1e-3 relative from
    /// the batched forward (measured: max 0.0082 here, 0 issues with TF32 off).
    @Test func cachedDecodeMatchesUncachedForward() throws {
        let model = try tinyModel()
        let tokens = MLXArray((0 ..< 13).map { Int32(($0 * 7 + 3) % 64) }).reshaped(1, 13)
        let full = model(tokens, cache: nil)
        let bound: Float =
            ProcessInfo.processInfo.environment["MLX_ENABLE_TF32"] == "0" ? 1e-3 : 3e-2

        for chunks in [[5, 1, 1, 1, 1, 1, 1, 1, 1], [6, 5, 1, 1], [2, 9, 2]] {
            let cache = model.newCache(parameters: nil)
            var start = 0
            for n in chunks {
                let out = model(tokens[0..., start ..< start + n], cache: cache)
                let diff = abs(out - full[0..., start ..< start + n, 0...]).max().item(Float.self)
                #expect(diff < bound, "chunks \(chunks): positions \(start)..<\(start + n) diff \(diff)")
                start += n
            }
            // INVARIANT (LFM3ShortConv): conv layers advance the offset like attention layers.
            #expect(cache.allSatisfy { $0.offset == 13 }, "offsets \(cache.map(\.offset))")
        }
    }

    @Test func cacheKindsFollowTheLayerPattern() throws {
        let cache = try tinyModel().newCache(parameters: nil)
        #expect(cache[0] is MambaCache)
        #expect(cache[1] is KVCacheSimple)
        #expect((cache[2] as? RotatingKVCache)?.maxSize == 4)
        #expect(cache[3] is MambaCache)
    }

    // MARK: - Parsers

    @Test func toolCallsUseTheXMLFunctionEnvelope() {
        for type in ["lfm3", "lfm3_vl", "lfm3_omni", "lfm3_audio"] {
            #expect(ToolCallFormat.infer(from: type) == .xmlFunction, "\(type)")
            #expect(ToolCallFormat.fromCapabilityName(type) == .xmlFunction, "\(type)")
        }
        #expect(ToolCallFormat.infer(from: "lfm2") == .lfm2)
    }

    @Test func reasoningComesFromTheTemplateAndThePromptTail() throws {
        let template = #"{%- if enable_thinking -%}{{- "<think>" -}}{%- else -%}{{- "<think></think>" -}}{%- endif -%}"#
        let resolved = ParserResolution.reasoning(
            capabilities: nil, modelType: "lfm3", chatTemplate: template)
        let parser = try #require(resolved.parser)
        #expect(parser.startTag == "<think>" && parser.endTag == "</think>")
        // A `<think>` prefill starts inside reasoning; an empty `<think></think>` starts in content.
        var thinking = try #require(
            ReasoningParser.forPrompt(stampName: "qwen3", promptTail: "<|im_start|>assistant\n<think>"))
        #expect(Self.channels(&thinking, "plan</think>answer") == ["r:plan", "c:answer"])
        var plain = try #require(
            ReasoningParser.forPrompt(
                stampName: "qwen3", promptTail: "<|im_start|>assistant\n<think></think>"))
        #expect(Self.channels(&plain, "answer") == ["c:answer"])
    }

    private static func channels(_ parser: inout ReasoningParser, _ text: String) -> [String] {
        // Adjacent segments of one channel are merged: only the routing is under test.
        var out: [String] = []
        for segment in parser.feed(text) + parser.flush() {
            let (tag, body): (String, String)
            switch segment {
            case .reasoning(let r): (tag, body) = ("r:", r)
            case .content(let c): (tag, body) = ("c:", c)
            }
            guard !body.isEmpty else { continue }
            if let last = out.last, last.hasPrefix(tag) { out[out.count - 1] = last + body }
            else { out.append(tag + body) }
        }
        return out
    }

    // MARK: - Released bundle parity (opt-in)

    /// Set `LFM3_TEXT_BUNDLE` to a released LFM3 text bundle and `LFM3_TEXT_REF` to the JSON
    /// written by the reference script (mlx-vlm LFM3 port on the same bundle). Checks the chat
    /// template renders to the reference ids, per-position argmax, final top-20 logits, and the
    /// greedy continuation through `TokenIterator` (chunked prefill across the sliding window).
    @Test func releasedBundleMatchesReference() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let bundle = env["LFM3_TEXT_BUNDLE"], let refPath = env["LFM3_TEXT_REF"] else {
            return
        }
        // bf16 default: two MLX builds round bf16 differently across 24 layers. Measured on
        // LFM3-1.7B bf16 (1096 positions): top-20 diff <= 0.375, argmax flips at gaps <= 0.69.
        // The same run upcast to fp32 on both sides (LFM3_FP32=1, MLX_ENABLE_TF32=0) agrees on
        // 100% of positions with top-20 diff <= 7.8e-5; use LFM3_MAX_LOGIT_DIFF=1e-3 there.
        let maxLogitDiff = Float(env["LFM3_MAX_LOGIT_DIFF"] ?? "") ?? 0.75
        // bf16 runs of the quantized bundles flip 1-2% of prompt positions (gaps up to 3.9 on
        // 8-bit), from bf16 accumulation in the two MLX builds' quantized kernels. Upcasting
        // scales, biases and activations to fp32 on both sides removes every flip (4/8-bit:
        // 100% of 1096 positions, top-20 diff <= 9e-5). Strict runs: LFM3_FP32=1,
        // MLX_ENABLE_TF32=0, LFM3_MAX_LOGIT_DIFF=0.01, LFM3_MIN_ARGMAX_AGREEMENT=1.
        let minArgmaxAgreement = Double(env["LFM3_MIN_ARGMAX_AGREEMENT"] ?? "") ?? 0.95
        let ref = try JSONSerialization.jsonObject(
            with: Data(contentsOf: URL(fileURLWithPath: refPath))) as! [String: Any]
        let cases = try #require(ref["cases"] as? [String: [String: Any]])
        #expect(!cases.isEmpty, "reference has no cases")

        let directory = URL(fileURLWithPath: bundle)
        let context = try await LLMModelFactory.shared.load(
            from: directory, using: #huggingFaceTokenizerLoader())
        let model = try #require(context.model as? LFM3Model)
        if env["LFM3_FP32"] == "1" {
            // Control arm: bf16 weights upcast to fp32 on both sides (run with MLX_ENABLE_TF32=0)
            // separates port differences from bf16 kernel rounding.
            let upcast = model.parameters().flattened().map { key, value in
                (key, value.dtype == .bfloat16 ? value.asType(.float32) : value)
            }
            try model.update(parameters: ModuleParameters.unflattened(upcast), verify: .none)
            eval(model)
        }

        for (name, c) in cases.sorted(by: { $0.key < $1.key }) {
            let ids = try #require(c["ids"] as? [Int])
            let thinking = name.hasSuffix("_think")
            let messages = (name.hasPrefix("short") ? Self.shortMessages : Self.longMessages)
            let rendered = try context.tokenizer.applyChatTemplate(
                messages: messages, tools: nil,
                additionalContext: ["enable_thinking": thinking])
            #expect(rendered == ids, "\(name): template/tokenizer ids differ from reference")

            let logits = model(MLXArray(ids.map(Int32.init)).reshaped(1, -1), cache: nil)
                .asType(.float32)
            let argmax = argMax(logits[0], axis: -1).asArray(Int.self)
            let refArgmax = try #require(c["argmax"] as? [Int])
            let agreement =
                Double(zip(argmax, refArgmax).filter { $0 == $1 }.count) / Double(refArgmax.count)
            // Report each flip with its logit gap so a real divergence (flips clustered past the
            // sliding window, or surviving the fp32 arm) is visible next to rounding noise.
            let flips = zip(argmax, refArgmax).enumerated().filter { $0.element.0 != $0.element.1 }
            let gaps = flips.map { flip -> Float in
                let row = logits[0, flip.offset]
                return (row[flip.element.0] - row[flip.element.1]).item(Float.self)
            }
            if !flips.isEmpty {
                print("[LFM3] \(name) flips (position: gap): \(zip(flips.map(\.offset), gaps).map { "\($0):\($1)" })")
            }
            #expect(
                agreement >= minArgmaxAgreement,
                "\(name): argmax agreement \(agreement), largest flip gap \(gaps.max() ?? 0)")

            let topIds = try #require(c["top20_ids"] as? [Int])
            let topLogits = try #require(c["top20_logits"] as? [Double])
            let last = logits[0, -1]
            let diffs = zip(topIds, topLogits).map { abs(last[$0].item(Float.self) - Float($1)) }
            #expect(diffs.max()! <= maxLogitDiff, "\(name): top-20 logit diff \(diffs.max()!)")

            let refGreedy = try #require(c["greedy"] as? [Int])
            var iterator = try TokenIterator(
                prompt: MLXArray(ids.map(Int32.init)), model: model,
                parameters: GenerateParameters(maxTokens: refGreedy.count, temperature: 0))
            var greedy: [Int] = []
            while greedy.count < refGreedy.count, let token = iterator.next() {
                greedy.append(token)
            }
            let prefix = zip(greedy, refGreedy).prefix { $0 == $1 }.count
            print("[LFM3] \(name): \(ids.count) tok, argmax \(agreement), top20 max diff \(diffs.max()!), greedy prefix \(prefix)/\(refGreedy.count): \(context.tokenizer.decode(tokenIds: greedy))")
            #expect(prefix >= min(refGreedy.count, 12), "\(name): greedy diverged at \(prefix)")
        }
    }

    /// `LFM3_SPEED_BUNDLE`: greedy 256-token decode after a fixed prompt, one warmup and three
    /// timed runs per arm (compiled decode off / on); prints the median. Opt-in measurement,
    /// no assertion: compare with the reference script's numbers on the same bundle.
    @Test func decodeSpeed() async throws {
        guard let bundle = ProcessInfo.processInfo.environment["LFM3_SPEED_BUNDLE"] else { return }
        let context = try await LLMModelFactory.shared.load(
            from: URL(fileURLWithPath: bundle), using: #huggingFaceTokenizerLoader())
        let ids = try context.tokenizer.applyChatTemplate(
            messages: [["role": "user", "content": "Write a long story about a dinosaur who learns to code."]],
            tools: nil, additionalContext: ["enable_thinking": false])
        for compiled in [false, true] {
            func run(_ n: Int) throws -> Double {
                var parameters = GenerateParameters(maxTokens: n + 1, temperature: 0)
                parameters.enableCompiledDecode = compiled
                var iterator = try TokenIterator(
                    input: LMInput(tokens: MLXArray(ids.map(Int32.init))), model: context.model,
                    parameters: parameters)
                _ = iterator.next()  // prefill + first token, not timed
                let start = Date()
                for _ in 0 ..< n { guard iterator.next() != nil else { break } }
                return Double(n) / Date().timeIntervalSince(start)
            }
            _ = try run(32)
            let rates = try (0 ..< 3).map { _ in try run(256) }
            print("[LFM3] swift vmlx \(URL(fileURLWithPath: bundle).lastPathComponent) compiled=\(compiled): decode tok/s runs=\(rates.map { (($0 * 10).rounded() / 10) }) median=\(rates.sorted()[1])")
        }
    }

    static let shortMessages: [[String: any Sendable]] = [
        ["role": "user", "content": "What is the capital of Australia? Answer in one sentence."]
    ]
    static let longMessages: [[String: any Sendable]] = [
        [
            "role": "user",
            "content": String(
                repeating:
                    "The Great Barrier Reef is the world's largest coral reef system, composed of over 2,900 individual reefs and 900 islands stretching for over 2,300 kilometres. ",
                count: 14) + "\nIn one sentence, how long is the reef?",
        ]
    ]
}
