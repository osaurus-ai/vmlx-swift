// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import Testing

@Suite("MTP runtime metadata")
struct MTPRuntimeFocusedTests {
    @Test("Qwen-style preserved MTP bundle is detected but not auto-enabled")
    func qwenPreservedMTPBundleIsDetectedButNotAutoEnabled() throws {
        let root = try makeTemporaryBundle(name: "qwen-mtp-detected")
        defer { try? FileManager.default.removeItem(at: root) }

        try writeJSON([
            "model_type": "qwen3_vl",
            "text_config": [
                "model_type": "qwen3",
                "num_hidden_layers": 48,
                "mtp_num_hidden_layers": 1,
            ] as [String: Any],
        ], to: root.appendingPathComponent("config.json"))
        try writeJSON([
            "runtime": [
                "total_weight_bytes": 17_820_460_160,
                "total_weight_gb": 16.6,
                "bundle_has_mtp": true,
                "mtp_layers": 1,
                "mtp_mode": "preserved_enabled",
            ] as [String: Any],
        ], to: root.appendingPathComponent("jang_config.json"))
        try writeJSON([
            "weight_map": [
                "mtp.fc.weight": "model-00029-of-00029.safetensors",
                "mtp.layers.0.self_attn.q_proj.weight": "model-00029-of-00029.safetensors",
                "mtp.layers.0.mlp.down_proj.weight": "model-00029-of-00029.safetensors",
                "vision_tower.blocks.0.attn.qkv.weight": "model-00001-of-00029.safetensors",
                "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00029.safetensors",
            ] as [String: Any],
        ], to: root.appendingPathComponent("model.safetensors.index.json"))

        let status = try MTPBundleInspector.inspect(modelDirectory: root)

        #expect(status.bundleHasMTP)
        #expect(status.configuredLayers == 1)
        #expect(status.tensorCount == 3)
        #expect(status.visionTensorCount == 1)
        #expect(status.mode == .preservedEnabled)
        #expect(status.hasCompleteMTPArtifact)
        #expect(status.requiresAcceptRejectBeforeEnable)
        #expect(!status.speculativeDecodeEnabled)
        #expect(!status.canAutoLaunchMTP)
        #expect(status.configEvidence.contains("text_config.mtp_num_hidden_layers=1"))
        #expect(status.statusLine.contains("accept/reject required"))
    }

    @Test("MTP config without tensors is reported as metadata-only")
    func configOnlyMTPIsMetadataOnlyMissingWeights() throws {
        let root = try makeTemporaryBundle(name: "qwen-mtp-missing-weights")
        defer { try? FileManager.default.removeItem(at: root) }

        try writeJSON([
            "text_config": [
                "mtp_num_hidden_layers": 1,
            ] as [String: Any],
        ], to: root.appendingPathComponent("config.json"))
        try writeJSON([
            "weight_map": [
                "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00001.safetensors",
            ] as [String: Any],
        ], to: root.appendingPathComponent("model.safetensors.index.json"))

        let status = try MTPBundleInspector.inspect(modelDirectory: root)

        #expect(!status.bundleHasMTP)
        #expect(status.configuredLayers == 1)
        #expect(status.tensorCount == 0)
        #expect(status.mode == .metadataOnlyMissingWeights)
        #expect(!status.hasCompleteMTPArtifact)
        #expect(!status.speculativeDecodeEnabled)
        #expect(!status.canAutoLaunchMTP)
    }

    @Test("inactive native MTP scrub does not touch generic nextn metadata")
    func inactiveNativeMTPScrubDoesNotTouchGenericNextnMetadata() throws {
        let config = """
        {
          "model_type": "deepseek_v4",
          "mtp_num_hidden_layers": 1,
          "num_nextn_predict_layers": 7,
          "text_config": {
            "model_type": "qwen3_5",
            "mtp_num_hidden_layers": 1,
            "num_nextn_predict_layers": 3
          }
        }
        """.data(using: .utf8)!

        let scrubbed = try NativeMTPActivation.scrubInactiveMTPConfig(config)
        let object = try #require(
            JSONSerialization.jsonObject(with: scrubbed) as? [String: Any])
        let textConfig = try #require(object["text_config"] as? [String: Any])

        #expect(object["mtp_num_hidden_layers"] as? Int == 0)
        #expect(object["num_nextn_predict_layers"] as? Int == 7)
        #expect(textConfig["mtp_num_hidden_layers"] as? Int == 0)
        #expect(textConfig["num_nextn_predict_layers"] as? Int == 3)
    }

    @Test("JANG MTP metadata without tensor evidence is not treated as an MTP bundle")
    func jangMTPMetadataWithoutTensorEvidenceIsMissingWeights() throws {
        let root = try makeTemporaryBundle(name: "named-mtp-but-no-mtp-tensors")
        defer { try? FileManager.default.removeItem(at: root) }

        try writeJSON([
            "model_type": "qwen3_5",
            "text_config": [
                "num_hidden_layers": 64,
                "mtp_num_hidden_layers": 1,
            ] as [String: Any],
        ], to: root.appendingPathComponent("config.json"))
        try writeJSON([
            "format": "jang",
            "format_version": "2.0",
            "runtime": [
                "bundle_has_mtp": true,
                "mtp_layers": 1,
                "mtp_mode": "preserved_enabled",
            ] as [String: Any],
        ], to: root.appendingPathComponent("jang_config.json"))
        try writeJSON([
            "weight_map": [
                "model.embed_tokens.weight": "model-00001-of-00001.safetensors",
                "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00001.safetensors",
                "model.layers.63.mlp.down_proj.weight": "model-00001-of-00001.safetensors",
            ] as [String: Any],
        ], to: root.appendingPathComponent("model.safetensors.index.json"))

        let status = try MTPBundleInspector.inspect(modelDirectory: root)

        #expect(!status.bundleHasMTP)
        #expect(status.configuredLayers == 1)
        #expect(status.tensorCount == 0)
        #expect(status.mode == .metadataOnlyMissingWeights)
        #expect(!status.hasCompleteMTPArtifact)
        #expect(!status.canAutoLaunchMTP)
        #expect(status.configEvidence.contains("jang_config.runtime.bundle_has_mtp=true"))
    }

    @Test("JANG runtime parses MTP activation metadata")
    func jangRuntimeParsesMTPActivationMetadata() throws {
        let config = try JangLoader.parseConfig(from: [
            "runtime": [
                "total_weight_bytes": 17_820_460_160,
                "total_weight_gb": 16.6,
                "bundle_has_mtp": true,
                "mtp_layers": 1,
                "mtp_mode": "preserved_enabled",
            ] as [String: Any],
        ])

        #expect(config.runtime.totalWeightBytes == 17_820_460_160)
        #expect(config.runtime.bundleHasMTP)
        #expect(config.runtime.mtpLayers == 1)
        #expect(config.runtime.mtpMode == .preservedEnabled)
    }

    @Test("ModelConfiguration carries MTP status into resolved configuration")
    func modelConfigurationCarriesMTPStatusIntoResolvedConfiguration() {
        let root = URL(fileURLWithPath: "/tmp/qwen-mtp")
        let status = MTPBundleStatus(
            bundleHasMTP: true,
            configuredLayers: 1,
            tensorCount: 31,
            visionTensorCount: 333,
            mode: .preservedEnabled,
            tensorSamples: ["mtp.fc.weight"],
            visionTensorSamples: ["vision_tower.blocks.0.attn.qkv.weight"],
            configEvidence: ["text_config.mtp_num_hidden_layers=1"])
        let configuration = ModelConfiguration(
            directory: root,
            mtpStatus: status)

        let resolved = configuration.resolved(modelDirectory: root, tokenizerDirectory: root)

        #expect(configuration.mtpStatus == status)
        #expect(resolved.mtpStatus == status)
        #expect(resolved.mtpStatus?.requiresAcceptRejectBeforeEnable == true)
    }

    @Test("recursive MTP contract models D3 hidden-state draft verify")
    func recursiveMTPContractModelsD3HiddenStateDraftVerify() {
        let contract = MTPRecursiveDraftContract.mtplxDepth3

        #expect(contract.depth == 3)
        #expect(contract.draftStepReturnsHiddenState)
        #expect(contract.draftCacheIsPrivate)
        #expect(contract.backboneCacheCommitPolicy == .acceptedVerifierTokensOnly)
        #expect(contract.verifierPositionsPerCycle == 4)
        #expect(contract.minAcceptedDraftTokensPerVerify == 0)
        #expect(contract.maxAcceptedDraftTokensPerVerify == 3)
        #expect(contract.requiresVariablePrefixCommit)
        #expect(contract.partialAcceptCommitStrategy == .captureCommit)
        #expect(contract.maxCommittedTokensPerVerify == 4)
        #expect(contract.fullAcceptanceVerifyCycles(forOutputTokens: 256) == 64)
        #expect(contract.speedBenchRequirements.requiresARBaseline)
        #expect(contract.speedBenchRequirements.requiresVerifyCalls)
        #expect(contract.speedBenchRequirements.requiresAcceptedDraftedByDepth)
        #expect(contract.speedBenchRequirements.requiresPhaseTiming)
        #expect(contract.speedBenchRequirements.requiresOutputTailReview)
    }

    @Test("shape-walk quantization preserves MXFP4 mode")
    func shapeWalkQuantizationPreservesMXFP4Mode() {
        let weights: [String: MLXArray] = [
            "model.layers.0.mlp.down_proj.weight": MLXArray.zeros([2, 16], dtype: .uint32),
            "model.layers.0.mlp.down_proj.scales": MLXArray.zeros([2, 4], dtype: .float32),
            "model.layers.1.mlp.down_proj.weight": MLXArray.zeros([2, 32], dtype: .uint32),
            "model.layers.1.mlp.down_proj.scales": MLXArray.zeros([2, 4], dtype: .float32),
        ]

        let inferred = JangLoader.inferPerLayerQuantizationFromShapes(
            weights: weights,
            defaultBits: 4,
            defaultGroupSize: 32,
            defaultMode: .mxfp4)

        #expect(inferred?.quantization?.mode == .mxfp4)
        if case .quantize(let override)? =
            inferred?.perLayerQuantization["model.layers.1.mlp.down_proj"]
        {
            #expect(override.bits == 8)
            #expect(override.groupSize == 32)
            #expect(override.mode == .mxfp4)
        } else {
            Issue.record("Expected 8-bit MXFP4 per-layer override")
        }
    }

    @Test("Qwen3.5 sanitize does not shift base norms just because MTP tensors exist")
    func qwen35SanitizeDoesNotShiftBaseNormsForPreservedMTP() throws {
        let configData = """
        {
          "hidden_size": 4,
          "num_hidden_layers": 1,
          "intermediate_size": 8,
          "num_attention_heads": 1,
          "num_key_value_heads": 1,
          "linear_num_value_heads": 1,
          "linear_num_key_heads": 1,
          "linear_key_head_dim": 4,
          "linear_value_head_dim": 4,
          "linear_conv_kernel_dim": 4,
          "head_dim": 4,
          "vocab_size": 16,
          "tie_word_embeddings": false
        }
        """.data(using: .utf8)!
        let configuration = try JSONDecoder().decode(Qwen35TextConfiguration.self, from: configData)
        let model = Qwen35TextModel(configuration)
        let norm = MLXArray([Float](repeating: 0.5, count: 4))

        let sanitized = model.sanitize(weights: [
            "mtp.layers.0.linear_attn.conv1d.weight": MLXArray.zeros([4, 4, 4], dtype: .float32),
            "mtp.fc.weight": MLXArray.zeros([4, 4], dtype: .float32),
            "model.norm.weight": norm,
        ])

        #expect(sanitized["mtp.fc.weight"] == nil)
        #expect(sanitized["mtp.layers.0.linear_attn.conv1d.weight"] == nil)
        #expect(sanitized["model.norm.weight"]?.asArray(Float.self) == [0.5, 0.5, 0.5, 0.5])
    }

    @Test("Qwen3.5 JANGTQ sanitize also ignores MTP sidecar conv when deciding norm shifts")
    func qwen35JANGTQSanitizeDoesNotShiftBaseNormsForPreservedMTP() throws {
        let configData = """
        {
          "hidden_size": 4,
          "num_hidden_layers": 1,
          "intermediate_size": 8,
          "num_attention_heads": 1,
          "num_key_value_heads": 1,
          "linear_num_value_heads": 1,
          "linear_num_key_heads": 1,
          "linear_key_head_dim": 4,
          "linear_value_head_dim": 4,
          "linear_conv_kernel_dim": 4,
          "head_dim": 4,
          "vocab_size": 16,
          "tie_word_embeddings": false,
          "num_experts": 0,
          "num_experts_per_tok": 0,
          "weight_format": "mxtq",
          "mxtq_bits": 4
        }
        """.data(using: .utf8)!
        let configuration = try JSONDecoder().decode(
            Qwen35JANGTQTextConfiguration.self, from: configData)
        let model = Qwen35JANGTQTextModel(configuration)
        let norm = MLXArray([Float](repeating: 0.5, count: 4))

        let sanitized = model.sanitize(weights: [
            "model.mtp_layers.0.linear_attn.conv1d.weight": MLXArray.zeros(
                [4, 4, 4], dtype: .float32),
            "mtp.fc.weight": MLXArray.zeros([4, 4], dtype: .float32),
            "model.norm.weight": norm,
        ])

        #expect(sanitized["model.mtp_layers.0.linear_attn.conv1d.weight"] == nil)
        #expect(sanitized["mtp.fc.weight"] == nil)
        #expect(sanitized["model.norm.weight"]?.asArray(Float.self) == [0.5, 0.5, 0.5, 0.5])
    }

    @Test("optional real local MTP bundle inspection")
    func optionalRealLocalMTPBundleInspection() throws {
        guard let path = ProcessInfo.processInfo.environment["VMLX_MTP_REAL_BUNDLE"],
            !path.isEmpty
        else {
            return
        }

        let status = try MTPBundleInspector.inspect(
            modelDirectory: URL(fileURLWithPath: path))

        #expect(status.bundleHasMTP)
        #expect(status.configuredLayers > 0)
        #expect(status.tensorCount > 0)
        #expect(status.hasCompleteMTPArtifact)
        #expect(!status.canAutoLaunchMTP)
        if ProcessInfo.processInfo.environment["VMLX_MTP_REAL_BUNDLE_EXPECTS_VL"] == "1" {
            #expect(status.visionTensorCount > 0)
            #expect(status.bundleHasVision)
        }
    }

    private func makeTemporaryBundle(name: String) throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(
            "\(name)-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }

    private func writeJSON(_ object: [String: Any], to url: URL) throws {
        let data = try JSONSerialization.data(
            withJSONObject: object,
            options: [.prettyPrinted, .sortedKeys])
        try data.write(to: url)
    }
}
