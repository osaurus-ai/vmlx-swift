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

    func testNestedOneBitMxtqBitsDecodeUltraShape() throws {
        let config = try JSONDecoder.json5().decode(
            NemotronHConfiguration.self,
            from: minimalConfig(
                weightFormat: "mxtq",
                mxtqBits: [
                    "mamba_projection": 8,
                    "routed_expert": [
                        "up_proj": 1,
                        "down_proj": 1,
                    ],
                    "shared_expert": 8,
                ]))

        XCTAssertEqual(config.modelType, "nemotron_h")
        XCTAssertEqual(config.nRoutedExperts, 4)
        XCTAssertEqual(config.numExpertsPerTok, 2)
        XCTAssertEqual(config.moeIntermediateSize, 6)
    }

    func testJANGTQ1WeightFormatSourceRoutesNemotronToOneBitJANGTQ() throws {
        let source = try String(
            contentsOfFile: "Libraries/MLXLLM/Models/NemotronH.swift",
            encoding: .utf8)
        let factorySource = try String(
            contentsOfFile: "Libraries/MLXLLM/LLMModelFactory.swift",
            encoding: .utf8)

        XCTAssertTrue(source.contains("JANGTQStreamingExperts.isEnabled"))
        XCTAssertTrue(source.contains("JANGTQStreamingExperts.shouldAutoEnableNemotronUltra("))
        XCTAssertTrue(source.contains("StreamingTurboQuantSwitchReLUSquaredMLP("))
        XCTAssertTrue(source.contains("NemotronHJANGTQSwitchMLP("))
        XCTAssertTrue(factorySource.contains("routedExpertBits"))
        XCTAssertTrue(factorySource.contains("NemotronHJANGTQContext("))
    }

    func testMissingJANGTQSignalsKeepsNemotronAffineSwitchMLPSourcePath() throws {
        let source = try String(
            contentsOfFile: "Libraries/MLXLLM/Models/NemotronH.swift",
            encoding: .utf8)

        XCTAssertTrue(source.contains("self._switchMLP.wrappedValue = NemotronHSwitchMLP("))
        XCTAssertTrue(source.contains("if let jangtq"))
    }

    func testUltraOneBitJANGTQStreamingStaysExplicitUntilLiveFastPathIsProven() throws {
        let streamingSource = try String(
            contentsOfFile: "Libraries/MLXLMCommon/JANGTQStreamingExperts.swift",
            encoding: .utf8)
        let linearSource = try String(
            contentsOfFile: "Libraries/MLXLMCommon/TurboQuantSwitchLinear.swift",
            encoding: .utf8)

        XCTAssertTrue(streamingSource.contains("shouldAutoEnableNemotronUltra("))
        XCTAssertTrue(streamingSource.contains("return false"))
        XCTAssertTrue(streamingSource.contains("canUseNemotronUltraStreaming(layerIdx: Int)"))
        XCTAssertTrue(streamingSource.contains("diagnostic-only"))
        XCTAssertTrue(linearSource.contains("useStreamingPlaceholders: Bool = false"))
        XCTAssertTrue(linearSource.contains("if useStreamingPlaceholders || JANGTQStreamingExperts.isEnabled"))
    }

    func testUltraRuntimeFastPathControlsAreSourceWired() throws {
        let modelSource = try String(
            contentsOfFile: "Libraries/MLXLLM/Models/NemotronH.swift",
            encoding: .utf8)
        let jangtqSource = try String(
            contentsOfFile: "Libraries/MLXLLM/Models/NemotronHJANGTQ.swift",
            encoding: .utf8)
        let streamingSource = try String(
            contentsOfFile: "Libraries/MLXLMCommon/JANGTQStreamingExperts.swift",
            encoding: .utf8)

        XCTAssertTrue(modelSource.contains("JANGTQ_DISABLE_NEMOTRON_ACTIVATION_BF16"))
        XCTAssertTrue(modelSource.contains("JANGTQ_DISABLE_NEMOTRON_WEIGHTED_MOE_FASTPATH"))
        XCTAssertTrue(modelSource.contains("weightedDecode(expertInput, inds, scores: scores)"))
        XCTAssertTrue(modelSource.contains("residual.asType(x.dtype)"))
        XCTAssertTrue(modelSource.contains("out.asType(lmHead.weight.dtype)"))
        XCTAssertTrue(jangtqSource.contains("func weightedDecode(_ x: MLXArray, _ indices: MLXArray, scores: MLXArray) -> MLXArray?"))
        XCTAssertTrue(jangtqSource.contains("let y = callAsFunction(x, indices)\n        return (y * scores[.ellipsis, .newAxis]).sum(axis: -2)"))
        XCTAssertFalse(jangtqSource.contains("JANGTQKernels.gatherTQTopKScored("))
        XCTAssertFalse(jangtqSource.contains("JANGTQ_ENABLE_NEMOTRON_SWITCHMLP_COMPILE"))
        XCTAssertTrue(jangtqSource.contains("extension StreamingTurboQuantSwitchReLUSquaredMLP: NemotronHSwitchMLPLayer"))
        XCTAssertTrue(jangtqSource.contains("reduced(x, indices: indices, scores: scores)"))
        XCTAssertTrue(streamingSource.contains("public func reduced(_ x: MLXArray, indices: MLXArray, scores: MLXArray) -> MLXArray"))
        XCTAssertTrue(streamingSource.contains("relu_reduce.call_chunk"))
        XCTAssertTrue(streamingSource.contains("relu_reduce.score_sum_build"))
    }

    func testStackedNemotronWeightedDecodeDoesNotUseRejectedScoredDownProjectionKernel() throws {
        let kernelSource = try String(
            contentsOfFile: "Libraries/MLXLMCommon/JANGTQKernels.swift",
            encoding: .utf8)

        XCTAssertFalse(kernelSource.contains("kGatherTQScoredSource"))
        XCTAssertFalse(kernelSource.contains("jangtq_gather_tq_scored_matmul"))
        XCTAssertFalse(kernelSource.contains("public static func gatherTQTopKScored("))
    }

    func testUltraHybridCacheTopologyIsFortyEightMambaPlusTwelveAttentionKV() throws {
        let pattern = Self.ultraLayerBlockTypes
        XCTAssertEqual(pattern.count, 108)
        XCTAssertEqual(pattern.filter { $0 == "mamba" }.count, 48)
        XCTAssertEqual(pattern.filter { $0 == "moe" }.count, 48)
        XCTAssertEqual(pattern.filter { $0 == "attention" }.count, 12)

        let source = try String(
            contentsOfFile: "Libraries/MLXLLM/Models/NemotronH.swift",
            encoding: .utf8)
        XCTAssertTrue(source.contains("return MambaCache()"))
        XCTAssertTrue(source.contains("return KVCacheSimple()"))
        XCTAssertFalse(source.contains("TurboQuantKVCache()"))
    }

    func testUltraCapabilityParsersAndGenerationDefaultsStayBundleDriven() throws {
        XCTAssertEqual(ToolCallFormat.fromCapabilityName("nemotron"), .nemotron)
        XCTAssertEqual(ToolCallFormat.infer(from: "nemotron_h"), .nemotron)
        XCTAssertEqual(ToolCallFormat.fromCapabilityName("qwen3_coder_xml"), .xmlFunction)
        XCTAssertNotNil(ReasoningParser.fromCapabilityName("deepseek_r1"))
        XCTAssertNotNil(ReasoningParser.fromCapabilityName("nemotron_v3"))
        XCTAssertNotNil(ReasoningParser.fromCapabilityName("nemotron_3"))

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
        XCTAssertEqual(generationConfig.eosTokenIds?.values, [2, 11])
    }

    func testExplicitStreamingOffOverridesUltraAutoStreamingSourceContract() throws {
        let source = try String(
            contentsOfFile: "Libraries/MLXLMCommon/JANGTQStreamingExperts.swift",
            encoding: .utf8)

        XCTAssertTrue(source.contains("if let explicit = explicitStreamingEnabled"))
        XCTAssertTrue(source.contains("return explicit"))
        XCTAssertTrue(source.contains("raw == \"0\" || raw == \"false\" || raw == \"no\" || raw == \"off\""))
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

}
