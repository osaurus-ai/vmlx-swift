// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
@testable import MLXLLM
@testable import MLXLMCommon
import XCTest

final class NemotronHJANGTQDispatchFocusedTests: XCTestCase {
    private func minimalConfig(
        weightFormat: String?,
        mxtqBits: Any? = nil,
        routedExpertBits: Int? = nil,
        nRoutedExperts: Int = 4,
        numExpertsPerTok: Int = 2,
        layersBlockType: [String] = ["mamba", "moe", "attention"]
    ) -> Data {
        var dict: [String: Any] = [
            "model_type": "nemotron_h",
            "vocab_size": 32,
            "hidden_size": 8,
            "num_hidden_layers": 3,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "mamba_num_heads": 2,
            "mamba_head_dim": 4,
            "ssm_state_size": 2,
            "conv_kernel": 4,
            "n_groups": 1,
            "intermediate_size": 8,
            "moe_intermediate_size": 6,
            "moe_latent_size": 4,
            "moe_shared_expert_intermediate_size": 6,
            "n_routed_experts": nRoutedExperts,
            "n_shared_experts": 1,
            "num_experts_per_tok": numExpertsPerTok,
            "layers_block_type": layersBlockType,
            "layer_norm_epsilon": 1e-5,
            "n_group": 1,
            "topk_group": 1,
            "norm_topk_prob": true,
            "routed_scaling_factor": 5.0,
            "time_step_limit": [0.0, 1.0e20],
            "tie_word_embeddings": false,
        ]
        if let weightFormat {
            dict["weight_format"] = weightFormat
        }
        if let mxtqBits {
            dict["mxtq_bits"] = mxtqBits
        }
        if let routedExpertBits {
            dict["routed_expert_bits"] = routedExpertBits
        }
        return try! JSONSerialization.data(withJSONObject: dict)
    }

    func testNestedOneBitMxtqBitsRoutesUltraShapeToJANGTQSwitchMLP() async throws {
        let configData = minimalConfig(
            weightFormat: "mxtq",
            mxtqBits: [
                "mamba_projection": 8,
                "routed_expert": [
                    "up_proj": 1,
                    "down_proj": 1,
                ],
                "shared_expert": 8,
            ])

        let model = try await LLMTypeRegistry.shared.createModel(
            configuration: configData,
            modelType: "nemotron_h")

        let modules = model.namedModules()
        let fc1 = modules.compactMap { name, module -> TurboQuantSwitchLinear? in
            name.hasSuffix("mixer.switch_mlp.fc1") ? module as? TurboQuantSwitchLinear : nil
        }
        let fc2 = modules.compactMap { name, module -> TurboQuantSwitchLinear? in
            name.hasSuffix("mixer.switch_mlp.fc2") ? module as? TurboQuantSwitchLinear : nil
        }

        XCTAssertEqual(fc1.count, 1)
        XCTAssertEqual(fc2.count, 1)
        XCTAssertEqual(fc1.first?.bits, 1)
        XCTAssertEqual(fc2.first?.bits, 1)
        XCTAssertEqual(fc1.first?.inFeatures, 4)
        XCTAssertEqual(fc2.first?.inFeatures, 6)
    }

    func testJANGTQ1WeightFormatRoutesNemotronToOneBitJANGTQ() async throws {
        let configData = minimalConfig(weightFormat: "JANGTQ1", routedExpertBits: 1)

        let model = try await LLMTypeRegistry.shared.createModel(
            configuration: configData,
            modelType: "nemotron_h")
        let tqLinears = model.namedModules().compactMap { _, module in
            module as? TurboQuantSwitchLinear
        }

        XCTAssertEqual(tqLinears.count, 2)
        XCTAssertTrue(tqLinears.allSatisfy { $0.bits == 1 })
    }

    func testMissingJANGTQSignalsKeepsNemotronAffineSwitchMLP() async throws {
        let configData = minimalConfig(weightFormat: nil)

        let model = try await LLMTypeRegistry.shared.createModel(
            configuration: configData,
            modelType: "nemotron_h")

        XCTAssertTrue(model.namedModules().contains { _, module in
            String(describing: type(of: module)) == "NemotronHSwitchMLP"
        })
        XCTAssertFalse(model.namedModules().contains { _, module in
            module is TurboQuantSwitchLinear
        })
    }

    func testUltraOneBitJANGTQDoesNotAutoSelectStreamingExperts() async throws {
        let configData = minimalConfig(
            weightFormat: "mxtq",
            mxtqBits: [
                "routed_expert": [
                    "up_proj": 1,
                    "down_proj": 1,
                ]
            ],
            nRoutedExperts: 512,
            numExpertsPerTok: 22)

        let model = try await withEnv("MLXPRESS_STREAMING_EXPERTS", value: nil) {
            try await LLMTypeRegistry.shared.createModel(
                configuration: configData,
                modelType: "nemotron_h")
        }

        XCTAssertFalse(model.namedModules().contains { _, module in
            module is StreamingTurboQuantSwitchReLUSquaredMLP
        })

        let tqLinears = model.namedModules().compactMap { _, module in
            module as? TurboQuantSwitchLinear
        }
        XCTAssertEqual(tqLinears.count, 2)
        XCTAssertTrue(tqLinears.contains { $0.packed.shape == [512, 6, 1] })
        XCTAssertTrue(tqLinears.contains { $0.packed.shape == [512, 4, 1] })
    }

    func testExplicitStreamingOnSelectsUltraStreamingExperts() async throws {
        let configData = minimalConfig(
            weightFormat: "mxtq",
            mxtqBits: [
                "routed_expert": [
                    "up_proj": 1,
                    "down_proj": 1,
                ]
            ],
            nRoutedExperts: 512,
            numExpertsPerTok: 22)

        let model = try await withEnv("MLXPRESS_STREAMING_EXPERTS", value: "1") {
            try await LLMTypeRegistry.shared.createModel(
                configuration: configData,
                modelType: "nemotron_h")
        }

        XCTAssertTrue(model.namedModules().contains { _, module in
            module is StreamingTurboQuantSwitchReLUSquaredMLP
        })

        let tqLinears = model.namedModules().compactMap { _, module in
            module as? TurboQuantSwitchLinear
        }
        XCTAssertEqual(tqLinears.count, 2)
        XCTAssertTrue(tqLinears.allSatisfy { $0.packed.shape == [1, 1, 1] })
        XCTAssertTrue(tqLinears.allSatisfy { $0.norms.shape == [1, 1] })
    }

    func testUltraStreamingMoEUsesWeightedReducedDecodePath() throws {
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let nemotron = try String(
            contentsOf: root.appendingPathComponent("Libraries/MLXLLM/Models/NemotronH.swift"),
            encoding: .utf8)
        let streaming = try String(
            contentsOf: root.appendingPathComponent("Libraries/MLXLMCommon/JANGTQStreamingExperts.swift"),
            encoding: .utf8)

        XCTAssertTrue(nemotron.contains("protocol NemotronHReducedSwitchMLPLayer"))
        XCTAssertTrue(nemotron.contains("reducedLayer.reduced(expertInput, indices: inds, scores: scores)"))
        XCTAssertTrue(nemotron.contains("(y * scores[.ellipsis, .newAxis]).sum(axis: -2)"))
        XCTAssertTrue(streaming.contains("public func reduced(_ x: MLXArray, indices: MLXArray, scores: MLXArray)"))
        XCTAssertTrue(streaming.contains("relu_fc2.offset_scored_build"))
        XCTAssertTrue(streaming.contains("gatherTQTopKOffsetsScored"))
    }

    func testUltraActivationDtypeRetentionIsExplicitInSwiftRuntime() throws {
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let nemotron = try String(
            contentsOf: root.appendingPathComponent("Libraries/MLXLLM/Models/NemotronH.swift"),
            encoding: .utf8)

        XCTAssertTrue(
            nemotron.contains("return (x + output).asType(x.dtype)"),
            "Nemotron block residuals must cast back to the incoming activation dtype.")
        XCTAssertTrue(
            nemotron.contains("out = out.asType(backbone.embeddings.weight.dtype)"),
            "Nemotron final hidden state must match embedding dtype before lm_head/asLinear.")
    }

    func testJANGTQNativeConvertsAffineQuantMetadataWithoutCastingTurboQuantNorms() throws {
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let load = try String(
            contentsOf: root.appendingPathComponent("Libraries/MLXLMCommon/Load.swift"),
            encoding: .utf8)

        XCTAssertTrue(
            load.contains("if isJANGTQNative {"),
            "JANGTQ-native bundles need their own dtype path; a blanket bypass leaves affine QuantizedLinear scales fp16.")
        XCTAssertTrue(load.contains(#"key.hasSuffix(".scales")"#))
        XCTAssertTrue(load.contains(#"key.hasSuffix(".biases")"#))
        XCTAssertTrue(
            load.contains(#"!key.hasSuffix(".tq_norms")"#),
            "TurboQuant norms must stay raw because JANGTQ kernels infer their signature from norm dtype.")
        XCTAssertTrue(
            load.contains("shouldConvert: (String, MLXArray) -> Bool"),
            "The BF16 conversion helper must support filtering so JANGTQ raw tensors are preserved.")
        XCTAssertFalse(
            load.contains("if !isJANGTQNative {\n        convertToBFloat16(model: model)\n    }"),
            "The old blanket JANGTQ bypass skips affine quantized Mamba/shared projections and reintroduces AsType cascades.")
    }

    func testUltraHybridCacheTopologyIsFortyEightMambaPlusTwelveAttentionKV() throws {
        let pattern = Self.ultraLayerBlockTypes
        XCTAssertEqual(pattern.count, 108)
        XCTAssertEqual(pattern.filter { $0 == "mamba" }.count, 48)
        XCTAssertEqual(pattern.filter { $0 == "moe" }.count, 48)
        XCTAssertEqual(pattern.filter { $0 == "attention" }.count, 12)

        let config = try JSONDecoder.json5().decode(
            NemotronHConfiguration.self,
            from: minimalConfig(
                weightFormat: "mxtq",
                mxtqBits: [
                    "routed_expert": [
                        "up_proj": 1,
                        "down_proj": 1,
                    ]
                ],
                nRoutedExperts: 512,
                numExpertsPerTok: 22,
                layersBlockType: pattern))
        let model = NemotronHModel(
            jangtqContext: NemotronHJANGTQContext(bits: 1),
            configuration: config)
        let cache = model.newCache(parameters: nil)

        XCTAssertEqual(cache.count, 60)
        XCTAssertEqual(cache.filter { $0 is MambaCache }.count, 48)
        XCTAssertEqual(cache.filter { $0 is KVCacheSimple }.count, 12)
        XCTAssertFalse(cache.contains { $0 is TurboQuantKVCache })
    }

    func testUltraCapabilityParsersAndGenerationDefaultsStayBundleDriven() throws {
        XCTAssertEqual(ToolCallFormat.fromCapabilityName("nemotron"), .xmlFunction)
        XCTAssertEqual(ToolCallFormat.infer(from: "nemotron_h"), .xmlFunction)
        XCTAssertNotNil(ReasoningParser.fromCapabilityName("deepseek_r1"))

        var parser = try XCTUnwrap(
            ReasoningParser.forPrompt(
                stampName: "deepseek_r1",
                promptTail: "<|im_start|>assistant\n<think></think>"))
        let segments = parser.feed("</think>Visible answer.") + parser.flush()
        let visible = segments.compactMap { segment -> String? in
            if case .content(let value) = segment { return value }
            return nil
        }.joined()
        let reasoning = segments.compactMap { segment -> String? in
            if case .reasoning(let value) = segment { return value }
            return nil
        }.joined()
        XCTAssertEqual(visible, "Visible answer.")
        XCTAssertEqual(reasoning, "")
        XCTAssertFalse(visible.contains("</think>"))

        let toolParser = ToolCallFormat.xmlFunction.createParser()
        let call = try XCTUnwrap(
            toolParser.parse(
                content:
                    "<tool_call><function=search><parameter=query>hybrid ssm cache</parameter></function></tool_call>",
                tools: nil))
        XCTAssertEqual(call.function.name, "search")
        XCTAssertEqual(call.function.arguments["query"], .string("hybrid ssm cache"))

        let generationConfig = try JSONDecoder().decode(
            GenerationConfigFile.self,
            from: Data(
                """
                {
                  "do_sample": true,
                  "temperature": 1.0,
                  "top_p": 0.95,
                  "eos_token_id": [2, 11],
                  "bos_token_id": 1,
                  "pad_token_id": 0
                }
                """.utf8))
        XCTAssertEqual(generationConfig.doSample, true)
        XCTAssertEqual(generationConfig.temperature, 1.0)
        XCTAssertEqual(generationConfig.topP, 0.95)
        XCTAssertNil(
            generationConfig.topK,
            "Ultra generation_config.json omits top_k; vMLX must not invent a Nemotron-specific top-k default")
        XCTAssertEqual(generationConfig.eosTokenIds?.values, [2, 11])
    }

    func testExplicitStreamingOffOverridesUltraAutoStreaming() async throws {
        let configData = minimalConfig(
            weightFormat: "mxtq",
            mxtqBits: [
                "routed_expert": [
                    "up_proj": 1,
                    "down_proj": 1,
                ]
            ],
            nRoutedExperts: 512,
            numExpertsPerTok: 22)

        let model = try await withEnv("MLXPRESS_STREAMING_EXPERTS", value: "0") {
            try await LLMTypeRegistry.shared.createModel(
                configuration: configData,
                modelType: "nemotron_h")
        }

        XCTAssertFalse(model.namedModules().contains { _, module in
            module is StreamingTurboQuantSwitchReLUSquaredMLP
        })

        let tqLinears = model.namedModules().compactMap { _, module in
            module as? TurboQuantSwitchLinear
        }
        XCTAssertEqual(tqLinears.count, 2)
        XCTAssertTrue(tqLinears.contains { $0.packed.shape == [512, 6, 1] })
        XCTAssertTrue(tqLinears.contains { $0.packed.shape == [512, 4, 1] })
    }

    func testNemotronStreamingFastPathRequiresOnlyUpDownProjectionCoverage() throws {
        let sourceURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent("Libraries/MLXLMCommon/JANGTQStreamingExperts.swift")
        let source = try String(contentsOf: sourceURL, encoding: .utf8)

        XCTAssertTrue(source.contains("requiredProjections: [StreamingProjection]"))
        XCTAssertTrue(source.contains("let requiredProjections: [StreamingProjection] = [.up, .down]"))
        XCTAssertTrue(
            source.contains(
                "JANGTQStreamingExpertStore.shared.canUseOffsetDispatch(\n            layerIdx: layerIdx,\n            requiredProjections: requiredProjections)"))
        XCTAssertTrue(
            source.contains(
                "JANGTQStreamingExpertStore.shared.canUseDirectStacked(\n                layerIdx: layerIdx,\n                requiredProjections: requiredProjections)"))
        XCTAssertTrue(
            source.contains(
                "hasOffsetDispatchCoverage(\n                layerIdx: layerIdx,\n                requiredProjections: requiredProjections)"))
        XCTAssertTrue(source.contains("shouldAutoUseOffsetDispatch("))
        XCTAssertTrue(source.contains("shouldAutoFilterOffsetSpans("))
        XCTAssertTrue(source.contains("mlXPressStreamingOffsetActiveShardFilterOverride() == nil"))
        XCTAssertFalse(
            source.contains(
                "let allIndexValues = indicesFlat.reshaped([-1]).asArray(Int32.self).map(Int.init)"))
    }

    private static let ultraLayerBlockTypes: [String] = [
        "mamba", "moe", "mamba", "moe", "mamba", "moe", "mamba", "attention",
        "moe", "mamba", "moe", "mamba", "moe", "mamba", "attention", "moe",
        "mamba", "moe", "mamba", "moe", "mamba", "moe", "mamba", "attention",
        "moe", "mamba", "moe", "mamba", "moe", "mamba", "moe", "mamba",
        "attention", "moe", "mamba", "moe", "mamba", "moe", "mamba",
        "attention", "moe", "mamba", "moe", "mamba", "moe", "mamba",
        "moe", "mamba", "attention", "moe", "mamba", "moe", "mamba",
        "moe", "mamba", "moe", "mamba", "attention", "moe", "mamba",
        "moe", "mamba", "moe", "mamba", "attention", "moe", "mamba",
        "moe", "mamba", "moe", "mamba", "moe", "mamba", "attention",
        "moe", "mamba", "moe", "mamba", "moe", "mamba", "moe", "mamba",
        "attention", "moe", "mamba", "moe", "mamba", "moe", "mamba",
        "attention", "moe", "mamba", "moe", "mamba", "moe", "mamba",
        "moe", "mamba", "attention", "moe", "mamba", "moe", "mamba",
        "moe", "mamba", "moe", "mamba", "moe",
    ]

    private func withEnv<T>(
        _ key: String,
        value: String?,
        _ body: () async throws -> T
    ) async throws -> T {
        let previous = getenv(key).map { String(cString: $0) }
        if let value {
            setenv(key, value, 1)
        } else {
            unsetenv(key)
        }
        defer {
            if let previous {
                setenv(key, previous, 1)
            } else {
                unsetenv(key)
            }
        }
        return try await body()
    }
}
