// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import Testing
import VMLX

@Suite("VMLX umbrella product")
struct VMLXUmbrellaProductTests {
    @Test("umbrella re-exports Osaurus runtime modules")
    func reexportsRuntimeModules() {
        let _: MLXArray.Type = MLXArray.self
        let _: ModelContext.Type = ModelContext.self
        let _: UserInput.Type = UserInput.self
        let _: GenerateParameters.Type = GenerateParameters.self
        let _: LLMRegistry.Type = LLMRegistry.self
        let _: MediaProcessing.Type = MediaProcessing.self
        let _: AutoTokenizer.Type = AutoTokenizer.self
        let _: Template.Type = Template.self
        let _: VMLXServerRuntimeSettings.Type = VMLXServerRuntimeSettings.self
        let _: GenerationConfigFile.Type = GenerationConfigFile.self
        let _: JangConfig.Type = JangConfig.self
        let _: JangCapabilities.Type = JangCapabilities.self
        let _: ParserResolution.Type = ParserResolution.self
        let _: ToolCallFormat.Type = ToolCallFormat.self
        let _: ReasoningParser.Type = ReasoningParser.self
        let _: MTPBundleStatus.Type = MTPBundleStatus.self
        let _: MTPBundleStatusSnapshot.Type = MTPBundleStatusSnapshot.self
        let _: NativeMTPTuning.Type = NativeMTPTuning.self
        let _: NativeMTPTuningSnapshot.Type = NativeMTPTuningSnapshot.self
        let _: ModelRuntimeCapabilitySnapshot.Type = ModelRuntimeCapabilitySnapshot.self
        let _: ModelRuntimeDetectionSnapshot.Type = ModelRuntimeDetectionSnapshot.self
        let _: ModelRuntimeBundleFormat.Type = ModelRuntimeBundleFormat.self
        let _: ModelRuntimeCapabilityRequest.Type = ModelRuntimeCapabilityRequest.self
        let _: ModelRuntimeCapabilityValidationResult.Type =
            ModelRuntimeCapabilityValidationResult.self
    }

    @Test("umbrella exposes MTP tuning snapshot JSON for Osaurus")
    func exposesMTPTuningSnapshotJSON() throws {
        let status = MTPBundleStatus(
            bundleHasMTP: true,
            configuredLayers: 1,
            tensorCount: 31,
            visionTensorCount: 333,
            mode: .preservedEnabled,
            tensorSamples: ["model.layers.64.mtp_fc.weight"],
            visionTensorSamples: ["vision_tower.blocks.0.attn.qkv.weight"],
            configEvidence: ["tuning_file=vmlx_mtp_tuning.json"],
            nativeMTPTuning: NativeMTPTuning(
                bestDepth: 2,
                verifierMode: "chunk_commit",
                validated: true,
                outputEquivalent: true,
                blocked: false,
                cacheMode: "paged+ssm",
                artifact: "docs/internal/release-gates/qwen36_mxfp4/result.json",
                baselineTokensPerSecond: 24.65,
                bestTokensPerSecond: 45.71,
                speedupVsBaseline: 1.85))

        let snapshot = status.snapshot
        #expect(snapshot.canAutoLaunch)
        #expect(snapshot.hasUsableNativeMTPTuning)
        #expect(!snapshot.requiresNativeMTPTuningBeforeAutoLaunch)
        #expect(snapshot.tuning?.usableBestDepth == 2)

        let data = try JSONEncoder().encode(snapshot)
        let object = try #require(
            JSONSerialization.jsonObject(with: data) as? [String: Any])
        #expect(object["can_auto_launch"] as? Bool == true)
        #expect(object["has_usable_native_mtp_tuning"] as? Bool == true)
        #expect(object["requires_native_mtp_tuning_before_auto_launch"] as? Bool == false)
        #expect(object["bundle_has_vision"] as? Bool == true)

        let tuning = try #require(object["tuning"] as? [String: Any])
        #expect(tuning["file"] as? String == "vmlx_mtp_tuning.json")
        #expect(tuning["best_depth"] as? Int == 2)
        #expect(tuning["usable_best_depth"] as? Int == 2)
        #expect(tuning["verifier_mode"] as? String == "chunk_commit")
    }

    @Test("umbrella exposes explicit Osaurus runtime capability snapshot")
    func exposesExplicitRuntimeCapabilitySnapshot() throws {
        let capabilities = JangCapabilities(
            reasoningParser: "qwen3_6",
            toolParser: "qwen",
            thinkInTemplate: true,
            supportsTools: true,
            supportsThinking: true,
            supportsText: true,
            supportsVision: true,
            supportsVideo: true,
            supportsAudio: false,
            family: "qwen3_6",
            modality: "vision",
            cacheType: "hybrid")
        let status = MTPBundleStatus(
            bundleHasMTP: true,
            configuredLayers: 1,
            tensorCount: 31,
            visionTensorCount: 333,
            mode: .preservedEnabled,
            tensorSamples: ["model.layers.64.mtp_fc.weight"],
            visionTensorSamples: ["vision_tower.blocks.0.attn.qkv.weight"],
            configEvidence: ["tuning_file=vmlx_mtp_tuning.json"],
            nativeMTPTuning: NativeMTPTuning(
                bestDepth: 3,
                verifierMode: "chunk_commit",
                validated: true,
                outputEquivalent: true,
                blocked: false,
                cacheMode: "paged+ssm",
                artifact: "docs/internal/release-gates/qwen36_mxfp8/result.json",
                baselineTokensPerSecond: 24.0,
                bestTokensPerSecond: 36.0,
                speedupVsBaseline: 1.5))
        let configuration = ModelConfiguration(
            directory: URL(fileURLWithPath: "/tmp/Qwen3.6-27B-MXFP8-MTP"),
            toolCallFormat: .xmlFunction,
            reasoningParserName: "qwen3_6",
            generationDefaults: GenerationConfigFile(
                temperature: 0.6,
                topP: 0.95,
                topK: 20,
                repetitionPenalty: 1.0),
            mtpStatus: status)

        let snapshot = ModelRuntimeCapabilitySnapshot(
            configuration: configuration,
            capabilities: capabilities,
            modelType: "qwen3_5_moe")

        #expect(snapshot.supportsText == .supported)
        #expect(snapshot.supportsVision == .supported)
        #expect(snapshot.supportsVideo == .supported)
        #expect(snapshot.supportsAudio == .unsupported)
        #expect(snapshot.supportsTools == .supported)
        #expect(snapshot.supportsReasoning == .supported)
        #expect(snapshot.supportsNativeMTP == .supported)
        #expect(snapshot.cacheType == "hybrid")
        #expect(snapshot.nativeMTP?.tuning?.usableBestDepth == 3)
        #expect(snapshot.generationDefaults?.topK == 20)

        let data = try JSONEncoder().encode(snapshot)
        let object = try #require(
            JSONSerialization.jsonObject(with: data) as? [String: Any])
        #expect(object["supports_text"] as? String == "supported")
        #expect(object["supports_vision"] as? String == "supported")
        #expect(object["supports_video"] as? String == "supported")
        #expect(object["supports_audio"] as? String == "unsupported")
        #expect(object["supports_native_mtp"] as? String == "supported")
        #expect(object["cache_type"] as? String == "hybrid")

        let resolved = ResolvedModelConfiguration(
            modelDirectory: URL(fileURLWithPath: "/tmp/qwen36/weights"),
            tokenizerDirectory: URL(fileURLWithPath: "/tmp/qwen36/tokenizer"),
            name: "served-qwen36-mtp",
            defaultPrompt: "",
            extraEOSTokens: [],
            eosTokenIds: [],
            toolCallFormat: .xmlFunction,
            reasoningParserName: "qwen3_6",
            generationDefaults: configuration.generationDefaults,
            mtpStatus: status)
        let resolvedSnapshot = ModelRuntimeCapabilitySnapshot(
            resolvedConfiguration: resolved,
            capabilities: capabilities,
            modelType: "qwen3_5_moe")
        #expect(resolvedSnapshot.modelName == "served-qwen36-mtp")
    }

    @Test("runtime detection trace explains JANGTQ VL native-MTP classification")
    func runtimeDetectionTraceExplainsJANGTQVLMTPClassification() throws {
        let root = try Self.makeTemporaryDirectory("vmlx-runtime-trace-jangtq-vl")
        defer { try? FileManager.default.removeItem(at: root) }

        try Data(
            """
            {
              "model_type": "qwen3_5_vl",
              "num_hidden_layers": 64,
              "num_nextn_predict_layers": 3,
              "text_config": {
                "model_type": "qwen3_5_moe",
                "num_hidden_layers": 64,
                "num_nextn_predict_layers": 3
              },
              "vision_config": {
                "model_type": "qwen3_5_vision"
              }
            }
            """.utf8
        )
        .write(to: root.appendingPathComponent("config.json"))

        try Data(
            """
            {
              "format": "jang",
              "format_version": "2.0",
              "weight_format": "mxtq",
              "mxtq_bits": {
                "routed_expert": {
                  "gate_proj": 4,
                  "up_proj": 4,
                  "down_proj": 4
                }
              },
              "quantization": {
                "method": "jang",
                "profile": "JANGTQ4",
                "bit_widths_used": [2, 4, 8]
              },
              "runtime": {
                "bundle_has_mtp": true,
                "mtp_layers": 3,
                "mtp_mode": "preserved_enabled"
              },
              "capabilities": {
                "family": "qwen3_6",
                "modality": "vision"
              }
            }
            """.utf8
        )
        .write(to: root.appendingPathComponent("jang_config.json"))

        try Data("{}".utf8).write(to: root.appendingPathComponent("preprocessor_config.json"))
        try Data(
            """
            {
              "weight_map": {
                "model.layers.64.mtp_fc.weight": "model-00001-of-00001.safetensors",
                "vision_tower.blocks.0.attn.qkv.weight": "model-00001-of-00001.safetensors"
              }
            }
            """.utf8
        )
        .write(to: root.appendingPathComponent("model.safetensors.index.json"))

        let trace = try ModelRuntimeDetectionSnapshot(modelDirectory: root)

        #expect(trace.bundleFormat == .jangtq)
        #expect(trace.configModelType == "qwen3_5_vl")
        #expect(trace.textConfigModelType == "qwen3_5_moe")
        #expect(trace.dispatchModelType == "qwen3_5_moe")
        #expect(trace.hasTextConfig)
        #expect(trace.hasVisionConfig)
        #expect(trace.hasPreprocessorConfig)
        #expect(trace.hasJangConfig)
        #expect(trace.jangWeightFormat == "mxtq")
        #expect(trace.effectiveWeightFormat == "mxtq")
        #expect(trace.mxtqBits == 4)
        #expect(trace.mxtqBitsSource == "jang_config.mxtq_bits.routed_expert.gate_proj")
        #expect(trace.nativeMTPMode == .preservedEnabled)
        #expect(trace.nativeMTPConfiguredLayers == 3)
        #expect(trace.nativeMTPTensorCount == 1)
        #expect(trace.visionTensorCount == 1)
        #expect(trace.evidence.contains("jang_config.weight_format=mxtq"))
        #expect(trace.evidence.contains("native_mtp.mode=preserved_enabled"))

        let snapshot = ModelRuntimeCapabilitySnapshot(
            configuration: ModelConfiguration(directory: root),
            capabilities: JangCapabilities(
                supportsText: true,
                supportsVision: true,
                family: "qwen3_6",
                modality: "vision"))
        #expect(snapshot.detection == trace)

        let encoded = String(decoding: try JSONEncoder().encode(snapshot), as: UTF8.self)
        #expect(encoded.contains("\"bundle_format\":\"jangtq\""))
        #expect(
            encoded.contains(
                "\"mxtq_bits_source\":\"jang_config.mxtq_bits.routed_expert.gate_proj\""))
        #expect(!encoded.contains(root.path))
    }

    @Test("runtime detection trace separates MXFP bundles from JANG affine bundles")
    func runtimeDetectionTraceSeparatesMXFPFromJANGAffine() throws {
        let mxfpRoot = try Self.makeTemporaryDirectory("vmlx-runtime-trace-mxfp")
        let jangRoot = try Self.makeTemporaryDirectory("vmlx-runtime-trace-jang")
        defer {
            try? FileManager.default.removeItem(at: mxfpRoot)
            try? FileManager.default.removeItem(at: jangRoot)
        }

        try Data(
            """
            {
              "model_type": "qwen3_5_moe",
              "quantization": {
                "bits": 4,
                "mode": "mxfp4"
              }
            }
            """.utf8
        )
        .write(to: mxfpRoot.appendingPathComponent("config.json"))

        try Data(
            """
            {
              "model_type": "deepseek_v4",
              "weight_format": "bf16"
            }
            """.utf8
        )
        .write(to: jangRoot.appendingPathComponent("config.json"))
        try Data(
            """
            {
              "format": "jang",
              "format_version": "2.0",
              "weight_format": "bf16",
              "quantization": {
                "method": "jang-affine",
                "profile": "JANG_4M",
                "bit_widths_used": [2, 4, 6, 8]
              }
            }
            """.utf8
        )
        .write(to: jangRoot.appendingPathComponent("jang_config.json"))

        let mxfpTrace = try ModelRuntimeDetectionSnapshot(modelDirectory: mxfpRoot)
        let jangTrace = try ModelRuntimeDetectionSnapshot(modelDirectory: jangRoot)

        #expect(mxfpTrace.bundleFormat == .mxfp)
        #expect(mxfpTrace.effectiveWeightFormat == nil)
        #expect(mxfpTrace.quantizationBits == 4)
        #expect(mxfpTrace.quantizationMode == "mxfp4")
        #expect(mxfpTrace.mxtqBits == nil)
        #expect(mxfpTrace.evidence.contains("config.quantization.mode=mxfp4"))

        #expect(jangTrace.bundleFormat == .jang)
        #expect(jangTrace.jangWeightFormat == "bf16")
        #expect(jangTrace.jangProfile == "JANG_4M")
        #expect(jangTrace.jangQuantizationMethod == "jang-affine")
        #expect(jangTrace.mxtqBits == nil)
    }

    @Test("jang capabilities parse explicit media support booleans")
    func parsesExplicitMediaSupportBooleans() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("vmlx-jang-capabilities-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let data = Data(
            """
            {
              "format": "jang",
              "format_version": "1.0",
              "capabilities": {
                "family": "nemotron_h_omni",
                "modality": "omni",
                "cache_type": "hybrid",
                "supports_tools": true,
                "supports_thinking": true,
                "supports_text": true,
                "supports_vision": true,
                "supports_video": true,
                "supports_audio": true
              }
            }
            """.utf8)
        try data.write(to: root.appendingPathComponent("jang_config.json"))

        let config = try JangLoader.loadConfig(at: root)
        let capabilities = try #require(config.capabilities)
        #expect(capabilities.supportsText == true)
        #expect(capabilities.supportsVision == true)
        #expect(capabilities.supportsVideo == true)
        #expect(capabilities.supportsAudio == true)
    }

    @Test("capability snapshot rejects unsupported modalities with redacted error shape")
    func rejectsUnsupportedModalitiesWithRedactedErrorShape() throws {
        let capabilities = JangCapabilities(
            supportsTools: false,
            supportsThinking: nil,
            supportsText: true,
            supportsVision: true,
            supportsVideo: nil,
            supportsAudio: false,
            family: "gemma4",
            modality: "vision",
            cacheType: "swa")
        let configuration = ModelConfiguration(
            directory: URL(fileURLWithPath: "/tmp/private/Gemma4-E2B"),
            generationDefaults: GenerationConfigFile(temperature: 1.0, topK: 64))
        let snapshot = ModelRuntimeCapabilitySnapshot(
            configuration: configuration,
            capabilities: capabilities,
            modelType: "gemma4")

        let request = ModelRuntimeCapabilityRequest(
            modalities: [.text, .vision, .video, .audio, .tools, .reasoning])
        let result = snapshot.validate(request: request)

        #expect(!result.allowed)
        #expect(result.requestedModalities == [.text, .vision, .video, .audio, .tools, .reasoning])
        #expect(
            result.issues.map(\.code) == [
                "unknown_modality_support",
                "unsupported_modality",
                "unsupported_modality",
                "unknown_modality_support",
            ])
        #expect(result.issues.map(\.modality) == [.video, .audio, .tools, .reasoning])
        #expect(result.issues.first?.redactedLogFields["modality"] == "video")

        let data = try JSONEncoder().encode(result)
        let object = try #require(
            JSONSerialization.jsonObject(with: data) as? [String: Any])
        #expect(object["allowed"] as? Bool == false)
        let encoded = String(decoding: data, as: UTF8.self)
        #expect(encoded.contains("unsupported_modality"))
        #expect(!encoded.contains("/tmp/private"))
        #expect(!encoded.contains("Gemma4-E2B"))
    }

    @Test("Audex model type advertises native audio input")
    func audexModelTypeAdvertisesAudio() {
        let snapshot = ModelRuntimeCapabilitySnapshot(
            configuration: ModelConfiguration(
                directory: URL(fileURLWithPath: "/tmp/Nemotron-Labs-Audex-2B")),
            modelType: "nemotron_dense_audex")

        #expect(snapshot.supportsText == .supported)
        #expect(snapshot.supportsVision == .unknown)
        #expect(snapshot.supportsVideo == .unknown)
        #expect(snapshot.supportsAudio == .supported)
    }

    @Test("capability validator can allow unknown lanes while rejecting explicit unsupported lanes")
    func capabilityValidatorCanAllowUnknownLanes() throws {
        let snapshot = ModelRuntimeCapabilitySnapshot(
            configuration: ModelConfiguration(
                directory: URL(fileURLWithPath: "/tmp/private/TextModel")),
            capabilities: JangCapabilities(
                supportsText: true,
                supportsVision: false,
                supportsVideo: nil,
                supportsAudio: nil,
                family: "unknown",
                modality: nil),
            modelType: "qwen3")
        let request = ModelRuntimeCapabilityRequest(modalities: [.video, .audio, .vision])

        let strict = snapshot.validate(request: request)
        #expect(!strict.allowed)
        #expect(
            strict.issues.map(\.code) == [
                "unsupported_modality",
                "unknown_modality_support",
                "unknown_modality_support",
            ])

        let permissive = snapshot.validate(request: request, unknownPolicy: .allowUnknown)
        #expect(!permissive.allowed)
        #expect(permissive.issues.map(\.code) == ["unsupported_modality"])
        #expect(permissive.issues.first?.modality == .vision)
    }

    @Test("capability request from UserInput records media without prompt content")
    func capabilityRequestFromUserInputRecordsMediaWithoutPromptContent() throws {
        let image = UserInput.Image.array(MLXArray.zeros([1, 1, 3]))
        let video = UserInput.Video.frames([])
        let audio = UserInput.Audio.samples([0.0, 0.1], sampleRate: 16_000)
        let tool: ToolSpec = [
            "type": "function",
            "function": [
                "name": "private_tool_name",
                "parameters": ["type": "object"] as [String: any Sendable],
            ] as [String: any Sendable],
        ]
        let input = UserInput(
            prompt: "private prompt text that must not enter capability logs",
            images: [image],
            videos: [video],
            audios: [audio],
            tools: [tool])

        let request = ModelRuntimeCapabilityRequest(
            input: input,
            usesReasoning: true,
            usesNativeMTP: true)

        #expect(
            request.sortedModalities == [
                .text, .vision, .video, .audio, .tools, .reasoning, .nativeMTP,
            ])
        let data = try JSONEncoder().encode(request)
        let encoded = String(decoding: data, as: UTF8.self)
        #expect(encoded.contains("vision"))
        #expect(encoded.contains("video"))
        #expect(encoded.contains("audio"))
        #expect(encoded.contains("tools"))
        #expect(encoded.contains("native_mtp"))
        #expect(!encoded.contains("private prompt text"))
        #expect(!encoded.contains("private_tool_name"))
    }

    private static func makeTemporaryDirectory(_ prefix: String) throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("\(prefix)-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }
}
