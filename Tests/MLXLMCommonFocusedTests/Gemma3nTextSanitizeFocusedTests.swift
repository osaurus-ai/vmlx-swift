// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import Testing
@testable import MLXLLM

@Suite("Gemma3n text weight sanitizer", .serialized)
struct Gemma3nTextSanitizeFocusedTests {
    @Test("Gemma3n attention applies query RoPE from the pre-update cache offset")
    func attentionAppliesQueryRoPEFromPreUpdateCacheOffset() throws {
        let source = try String(contentsOfFile: Self.repoFile("Libraries/MLXLLM/Models/Gemma3nText.swift"))

        #expect(source.contains("let rotaryOffset = capturedRotaryOffset(for: cache)"))
        #expect(source.contains("queries = applyRotaryPosition(rope, to: queries, offset: rotaryOffset)"))
        #expect(!source.contains("queries = applyRotaryPosition(rope, to: queries, cache: cache)"))
    }

    @Test("conditional-generation config uses the VLM text embedding path")
    func conditionalGenerationConfigUsesVLMTextEmbeddingPath() throws {
        let config = try Self.makeConfig(vocabSize: 4, architectures: ["Gemma3nForConditionalGeneration"])

        #expect(config.usesVLMInputEmbeddingPath)
    }

    @Test("conditional-generation decode tokens keep Gemma3n language-model embedding scale")
    func conditionalGenerationDecodeTokensKeepLanguageModelEmbeddingScale() throws {
        let source = try String(contentsOfFile: Self.repoFile("Libraries/MLXLLM/Models/Gemma3nText.swift"))

        #expect(source.contains("let isSingleCachedTokenDecode ="))
        #expect(source.contains("if !config.usesVLMInputEmbeddingPath || isSingleCachedTokenDecode"))
    }

    @Test("production arithmetic gate avoids Gemma3n reference-failing symbol puzzle prompt")
    func productionArithmeticGateAvoidsReferenceFailingSymbolPuzzlePrompt() throws {
        let source = try String(contentsOfFile: Self.repoFile("RunBench/Bench.swift"))

        #expect(source.contains("What is 5 times 4? Answer with only the number."))
        #expect(!source.contains("Compute 5 * 4. Respond with just the number."))
    }

    @Test("production gate does not accept known wrong Gemma3n one-word answers")
    func productionGateDoesNotAcceptKnownWrongOneWordAnswers() throws {
        let source = try String(contentsOfFile: Self.repoFile("RunBench/Bench.swift"))
        let prodSource = Self.productionMatrixSource(from: source)

        #expect(!source.contains("accepted non-blue"))
        #expect(!prodSource.contains("accepted non-blue"))
        #expect(!prodSource.contains("What color is the sky on a clear day? Answer with one word."))
        #expect(!prodSource.contains("Name a planet. One word."))
        #expect(prodSource.contains("Question: Name the planet Mars. Answer with Mars only:"))
        #expect(prodSource.contains("no 'Mars' in visible output"))
    }

    @Test("production UTF-8 gate validates inclusion rather than fake verbatim success")
    func productionUTF8GateValidatesInclusionRatherThanFakeVerbatimSuccess() throws {
        let source = try String(contentsOfFile: Self.repoFile("RunBench/Bench.swift"))
        let prodSource = Self.productionMatrixSource(from: source)

        #expect(!source.contains("Write exactly this line verbatim:"))
        #expect(!prodSource.contains("S5 utf8 emoji verbatim"))
        #expect(!prodSource.contains("Write exactly this line verbatim:"))
        #expect(prodSource.contains("S5 utf8 inclusion"))
        #expect(prodSource.contains("missing expected UTF-8 words in visible output"))
    }

    @Test("template smoke marks native no-tool templates unsupported instead of forcing a fallback")
    func templateSmokeMarksNativeNoToolTemplatesUnsupported() throws {
        let source = try String(contentsOfFile: Self.repoFile("RunBench/Bench.swift"))
        let templateSource = Self.templateSmokeSource(from: source)

        #expect(templateSource.contains("nativeTemplateMentionsTools"))
        #expect(templateSource.contains("toolsUnsupportedByNativeTemplate"))
        #expect(templateSource.contains("status=\\(ok ? (toolsUnsupportedByNativeTemplate ? \"UNSUPPORTED\" : \"PASS\") : \"FAIL\")"))
        #expect(templateSource.contains("tool schema/name missing despite native tool-capable template"))
        #expect(!templateSource.contains("ChatTemplateFallbacks.gemma4WithTools"))
        #expect(!templateSource.contains("ChatTemplateFallbacks.gemma4Minimal"))
    }

    @Test("Gemma3n cache store skips unsafe history-boundary rederive after trim miss")
    func cacheStoreSkipsUnsafeHistoryBoundaryRederiveAfterTrimMiss() throws {
        let evaluate = try String(contentsOfFile: Self.repoFile("Libraries/MLXLMCommon/Evaluate.swift"))
        let batch = try String(contentsOfFile: Self.repoFile("Libraries/MLXLMCommon/BatchEngine/BatchEngine.swift"))
        let nativeMTP = try String(contentsOfFile: Self.repoFile("Libraries/MLXLMCommon/SpecDec/NativeMTPTokenIterator.swift"))

        for source in [evaluate, batch, nativeMTP] {
            #expect(source.contains("String(describing: Swift.type(of:"))
            #expect(source.contains("contains(\"Gemma3n\")"))
            #expect(!source.contains("Gemma3n") || !source.contains("forced </think>"))
        }
        #expect(evaluate.contains("TokenIterator: skipped Gemma3n history-boundary cache rederive after trim miss"))
        #expect(batch.contains("Skipped Gemma3n history-boundary cache rederive"))
    }

    @Test("text-only config keeps the LM text embedding path")
    func textOnlyConfigKeepsLMTextEmbeddingPath() throws {
        let config = try Self.makeConfig(vocabSize: 4, architectures: nil)

        #expect(!config.usesVLMInputEmbeddingPath)
    }

    @Test("full conditional-generation bundle is reduced to text model keys")
    func fullConditionalGenerationBundleReducesToTextKeys() throws {
        try FocusedMLXTestSupport.withLock {
            let model = try Gemma3nTextModel(config: Self.makeConfig(vocabSize: 4))
            let weights: [String: MLXArray] = [
                "language_model.model.embed_tokens.weight": MLXArray.zeros([6, 2]),
                "language_model.model.layers.0.self_attn.q_proj.weight": MLXArray.ones([2, 2]),
                "language_model.model.norm.weight": MLXArray.ones([2]),
                "model.language_model.layers.0.self_attn.k_proj.weight": MLXArray.ones([2, 2]),
                "audio_tower.conformer.0.attention.attn.q_proj.weight": MLXArray.ones([2, 2]),
                "vision_tower.timm_model.blocks.0.attn.qkv.weight": MLXArray.ones([2, 2]),
                "embed_audio.embedding.weight": MLXArray.ones([2, 2]),
                "embed_vision.embedding.weight": MLXArray.ones([2, 2]),
            ]

            let sanitized = model.sanitize(weights: weights)

            #expect(sanitized["language_model.embed_tokens.weight"]?.dim(0) == 4)
            #expect(sanitized["language_model.layers.0.self_attn.q_proj.weight"] != nil)
            #expect(sanitized["language_model.layers.0.self_attn.k_proj.weight"] != nil)
            #expect(sanitized["language_model.norm.weight"] != nil)
            #expect(sanitized.keys.allSatisfy { !$0.hasPrefix("language_model.model.") })
            #expect(sanitized.keys.allSatisfy { !$0.hasPrefix("audio_tower.") })
            #expect(sanitized.keys.allSatisfy { !$0.hasPrefix("vision_tower.") })
            #expect(sanitized.keys.allSatisfy { !$0.hasPrefix("embed_audio.") })
            #expect(sanitized.keys.allSatisfy { !$0.hasPrefix("embed_vision.") })
        }
    }

    private static func makeConfig(
        vocabSize: Int,
        architectures: [String]? = nil
    ) throws -> Gemma3nTextConfiguration {
        let architectureJSON =
            if let architectures {
                "\"architectures\": \(try String(data: JSONEncoder().encode(architectures), encoding: .utf8)!),"
            } else {
                ""
            }
        let json = """
        {
          \(architectureJSON)
          "text_config": {
            "model_type": "gemma3n_text",
            "hidden_size": 4,
            "num_hidden_layers": 2,
            "intermediate_size": [8, 8],
            "num_attention_heads": 1,
            "head_dim": 4,
            "rms_norm_eps": 0.000001,
            "vocab_size": \(vocabSize),
            "num_key_value_heads": 1,
            "num_kv_shared_layers": 1,
            "vocab_size_per_layer_input": 4,
            "sliding_window": 512,
            "max_position_embeddings": 32768,
            "rope_local_base_freq": 10000.0,
            "rope_theta": 1000000.0,
            "final_logit_softcapping": 30.0,
            "layer_types": ["sliding_attention", "full_attention"],
            "hidden_size_per_layer_input": 2,
            "altup_num_inputs": 2,
            "altup_correct_scale": true,
            "altup_active_idx": 0,
            "laurel_rank": 2
          }
        }
        """
        return try JSONDecoder().decode(Gemma3nTextConfiguration.self, from: Data(json.utf8))
    }

    private static func repoFile(_ relativePath: String) -> String {
        let testFile = URL(fileURLWithPath: #filePath)
        return testFile
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent(relativePath)
            .path
    }

    private static func productionMatrixSource(from source: String) -> String {
        source.components(separatedBy: "func runProdMatrix").dropFirst().joined(separator: "func runProdMatrix")
    }

    private static func templateSmokeSource(from source: String) -> String {
        source.components(separatedBy: "func runTemplateSmoke").dropFirst().joined(separator: "func runTemplateSmoke")
    }
}
