// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing

@Suite("MTP runtime metadata", .serialized)
struct MTPRuntimeFocusedTests {
    @Test("cached multi-token verifier mask carries cache offset")
    func cachedMultiTokenVerifierMaskCarriesCacheOffset() {
        FocusedMLXTestSupport.withLock {
            let cache = KVCacheSimple()
            _ = cache.update(
                keys: MLXArray.ones([1, 2, 5, 4]),
                values: MLXArray.ones([1, 2, 5, 4]))

            let mask = cache.makeMask(n: 3, windowSize: nil, returnArray: false)
            guard case .array(let maskArray) = mask else {
                Issue.record("expected explicit offset-aware mask for cached multi-token forward")
                return
            }

            MLX.eval(maskArray)
            #expect(maskArray.shape == [3, 8])
            #expect(maskArray[0, 4].item(Bool.self))
            #expect(maskArray[0, 5].item(Bool.self))
            #expect(!maskArray[0, 6].item(Bool.self))
            #expect(!maskArray[0, 7].item(Bool.self))
            #expect(maskArray[1, 6].item(Bool.self))
            #expect(!maskArray[1, 7].item(Bool.self))
            #expect(maskArray[2, 7].item(Bool.self))
        }
    }

    @Test("Qwen-style preserved MTP bundle is tensor-detected and auto-enabled")
    func qwenPreservedMTPBundleIsTensorDetectedAndAutoEnabled() throws {
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
        try writeJSON([
            "native_mtp": [
                "best_depth": 3,
                "validated": true,
                "output_equivalent": true,
                "artifact": "docs/internal/release-gates/qwen-depth3/result.json",
            ] as [String: Any],
        ], to: root.appendingPathComponent("vmlx_mtp_tuning.json"))

        let status = try MTPBundleInspector.inspect(modelDirectory: root)

        #expect(status.bundleHasMTP)
        #expect(status.configuredLayers == 1)
        #expect(status.tensorCount == 3)
        #expect(status.visionTensorCount == 1)
        #expect(status.mode == .preservedEnabled)
        #expect(status.hasCompleteMTPArtifact)
        #expect(!status.requiresAcceptRejectBeforeEnable)
        #expect(status.speculativeDecodeEnabled)
        #expect(status.configEvidence.contains("text_config.mtp_num_hidden_layers=1"))
        #expect(status.configEvidence.contains("tuning_file=vmlx_mtp_tuning.json"))
        #expect(status.statusLine.contains("speculative=on"))
    }

    @Test("Qwen MTP auto decode uses vmlx_mtp_tuning json")
    func qwenMTPAutoDecodeUsesVMLXTuningJSON() throws {
        let root = try makeTemporaryBundle(name: "qwen-mtp-tuning")
        defer { try? FileManager.default.removeItem(at: root) }

        try writeJSON([
            "model_type": "qwen3_vl",
            "text_config": [
                "model_type": "qwen3_5_moe_text",
                "num_hidden_layers": 48,
                "mtp_num_hidden_layers": 1,
            ] as [String: Any],
            "quantization": [
                "mode": "mxfp4",
                "bits": 4,
            ] as [String: Any],
        ], to: root.appendingPathComponent("config.json"))
        try writeJSON([
            "runtime": [
                "bundle_has_mtp": true,
                "mtp_layers": 1,
                "mtp_mode": "preserved_enabled",
            ] as [String: Any],
        ], to: root.appendingPathComponent("jang_config.json"))
        try writeJSON([
            "weight_map": [
                "mtp.fc.weight": "model-00029-of-00029.safetensors",
                "mtp.layers.0.self_attn.q_proj.weight": "model-00029-of-00029.safetensors",
                "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00029.safetensors",
            ] as [String: Any],
        ], to: root.appendingPathComponent("model.safetensors.index.json"))
        try writeJSON([
            "native_mtp": [
                "best_depth": 2,
                "validated": true,
                "output_equivalent": true,
                "cache_mode": "off",
                "artifact": "docs/internal/release-gates/qwen-depth2/result.json",
                "baseline_tok_s": 24.655,
                "best_tok_s": 45.712,
                "speedup_vs_baseline": 1.854,
            ] as [String: Any],
        ], to: root.appendingPathComponent("vmlx_mtp_tuning.json"))

        let status = try MTPBundleInspector.inspect(modelDirectory: root)
        let configData = try Data(contentsOf: root.appendingPathComponent("config.json"))
        let recommendation = NativeMTPAutoDecodePolicy.recommendation(
            configData: configData,
            jangConfig: try? JangLoader.loadConfig(at: root),
            status: status)

        #expect(recommendation?.depth == 2)
        #expect(recommendation?.verifierMode == "sequential_repair")
        #expect(recommendation?.evidence.contains("tuning_file=vmlx_mtp_tuning.json") == true)
        #expect(recommendation?.reason.contains("vmlx_mtp_tuning.json") == true)
    }

    @Test("complete MTP tensors without tuning stay non auto launch")
    func completeMTPTensorsWithoutTuningStayNonAutoLaunch() throws {
        let root = try makeTemporaryBundle(name: "qwen-mtp-missing-tuning")
        defer { try? FileManager.default.removeItem(at: root) }

        try writeJSON([
            "model_type": "qwen3_vl",
            "text_config": [
                "model_type": "qwen3_5_moe_text",
                "num_hidden_layers": 48,
                "mtp_num_hidden_layers": 1,
            ] as [String: Any],
            "quantization": [
                "mode": "mxfp4",
                "bits": 4,
            ] as [String: Any],
        ], to: root.appendingPathComponent("config.json"))
        try writeJSON([
            "runtime": [
                "bundle_has_mtp": true,
                "mtp_layers": 1,
                "mtp_mode": "preserved_enabled",
            ] as [String: Any],
        ], to: root.appendingPathComponent("jang_config.json"))
        try writeJSON([
            "weight_map": [
                "mtp.fc.weight": "model-00029-of-00029.safetensors",
                "mtp.layers.0.self_attn.q_proj.weight": "model-00029-of-00029.safetensors",
                "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00029.safetensors",
            ] as [String: Any],
        ], to: root.appendingPathComponent("model.safetensors.index.json"))

        let status = try MTPBundleInspector.inspect(modelDirectory: root)

        #expect(status.hasCompleteMTPArtifact)
        #expect(!status.canAutoLaunchMTP)
        #expect(status.statusLine.contains("tuning required"))
        #expect(status.configEvidence.contains("tuning_file_missing=vmlx_mtp_tuning.json"))
    }

    @Test("MTP status snapshot exposes tuning gate fields for Osaurus")
    func mtpStatusSnapshotExposesTuningGateFieldsForOsaurus() throws {
        let status = MTPBundleStatus(
            bundleHasMTP: true,
            configuredLayers: 1,
            tensorCount: 31,
            visionTensorCount: 333,
            mode: .preservedEnabled,
            tensorSamples: ["mtp.fc.weight"],
            visionTensorSamples: ["vision_tower.blocks.0.attn.qkv.weight"],
            configEvidence: ["tuning_file=vmlx_mtp_tuning.json"],
            nativeMTPTuning: NativeMTPTuning(
                bestDepth: 3,
                verifierMode: "sequential-repair",
                validated: true,
                outputEquivalent: true,
                cacheMode: "paged+ssm",
                artifact: "docs/internal/qwen36-mtp.json",
                baselineTokensPerSecond: 24.0,
                bestTokensPerSecond: 45.0,
                speedupVsBaseline: 1.875))

        let snapshot = status.snapshot

        #expect(snapshot.hasCompleteArtifact)
        #expect(snapshot.hasUsableNativeMTPTuning)
        #expect(snapshot.canAutoLaunch)
        #expect(snapshot.speculativeDecodeEnabled)
        #expect(snapshot.statusLine.contains("tuning=d3"))
        #expect(snapshot.tuning?.file == "vmlx_mtp_tuning.json")
        #expect(snapshot.tuning?.bestDepth == 3)
        #expect(snapshot.tuning?.verifierMode == "sequential_repair")

        let encoded = try JSONEncoder().encode(snapshot)
        let object = try #require(JSONSerialization.jsonObject(with: encoded) as? [String: Any])
        #expect(object["has_complete_artifact"] as? Bool == true)
        #expect(object["has_usable_native_mtp_tuning"] as? Bool == true)
        #expect(object["can_auto_launch"] as? Bool == true)
        let tuning = try #require(object["tuning"] as? [String: Any])
        #expect(tuning["file"] as? String == "vmlx_mtp_tuning.json")
        #expect(tuning["best_depth"] as? Int == 3)
    }

    @Test("MTP status snapshot reports missing tuning before auto launch")
    func mtpStatusSnapshotReportsMissingTuningBeforeAutoLaunch() throws {
        let status = MTPBundleStatus(
            bundleHasMTP: true,
            configuredLayers: 1,
            tensorCount: 31,
            visionTensorCount: 0,
            mode: .preservedEnabled)

        let snapshot = status.snapshot

        #expect(snapshot.hasCompleteArtifact)
        #expect(!snapshot.hasUsableNativeMTPTuning)
        #expect(!snapshot.canAutoLaunch)
        #expect(!snapshot.speculativeDecodeEnabled)
        #expect(snapshot.requiresNativeMTPTuningBeforeAutoLaunch)
        #expect(snapshot.statusLine.contains("vmlx_mtp_tuning.json"))
        #expect(snapshot.tuning == nil)
    }

    @Test("MTP status snapshot reports blocked tuning as non auto launch")
    func mtpStatusSnapshotReportsBlockedTuningAsNonAutoLaunch() throws {
        let status = MTPBundleStatus(
            bundleHasMTP: true,
            configuredLayers: 1,
            tensorCount: 31,
            mode: .speculativeVerified,
            nativeMTPTuning: NativeMTPTuning(
                bestDepth: 3,
                validated: false,
                outputEquivalent: false,
                blocked: true,
                reason: "diagnostic row was not production-valid"))

        let snapshot = status.snapshot

        #expect(!snapshot.hasUsableNativeMTPTuning)
        #expect(!snapshot.canAutoLaunch)
        #expect(snapshot.requiresNativeMTPTuningBeforeAutoLaunch)
        #expect(snapshot.tuning?.blocked == true)
        #expect(snapshot.tuning?.reason == "diagnostic row was not production-valid")
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

    @Test("valid tuning sidecar without MTP tensors stays non auto launch")
    func validTuningWithoutMTPTensorsStaysNonAutoLaunch() throws {
        let root = try makeTemporaryBundle(name: "qwen-valid-tuning-missing-weights")
        defer { try? FileManager.default.removeItem(at: root) }

        try writeJSON([
            "model_type": "qwen3_vl",
            "text_config": [
                "model_type": "qwen3_5_moe_text",
                "num_hidden_layers": 48,
                "mtp_num_hidden_layers": 1,
            ] as [String: Any],
        ], to: root.appendingPathComponent("config.json"))
        try writeJSON([
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
                "vision_tower.blocks.0.attn.qkv.weight": "model-00001-of-00001.safetensors",
            ] as [String: Any],
        ], to: root.appendingPathComponent("model.safetensors.index.json"))
        try writeJSON([
            "native_mtp": [
                "best_depth": 3,
                "validated": true,
                "output_equivalent": true,
                "artifact": "docs/internal/release-gates/qwen-depth3/result.json",
            ] as [String: Any],
        ], to: root.appendingPathComponent("vmlx_mtp_tuning.json"))

        let status = try MTPBundleInspector.inspect(modelDirectory: root)
        let configData = try Data(contentsOf: root.appendingPathComponent("config.json"))
        let jangConfig = try JangLoader.loadConfig(at: root)
        let settings = VMLXServerRuntimeSettings()

        #expect(!status.bundleHasMTP)
        #expect(status.bundleHasVision)
        #expect(status.nativeMTPTuning?.usableBestDepth == 3)
        #expect(status.configEvidence.contains("tuning_file=vmlx_mtp_tuning.json"))
        #expect(status.mode == .metadataOnlyMissingWeights)
        #expect(!status.hasCompleteMTPArtifact)
        #expect(!status.canAutoLaunchMTP)
        #expect(NativeMTPAutoDecodePolicy.recommendation(
            configData: configData,
            jangConfig: jangConfig,
            status: status,
            requireVerifiedRuntime: false) == nil)
        #expect(settings.resolvedLoadConfiguration(
            configData: configData,
            jangConfig: jangConfig,
            status: status).nativeMTP == false)

        var forceOn = settings
        forceOn.mtp.mode = .forceOn
        #expect(forceOn.validationIssues(
            configData: configData,
            jangConfig: jangConfig,
            mtpStatus: status).contains {
                $0.severity == .error && $0.field == "mtp.mode"
            })
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

    @Test("native MTP activation supports Qwen3.5 MoE only with explicit tensor evidence and tuning")
    func nativeMTPActivationSupportsQwen35MoEWithTensorEvidence() async throws {
        let config = """
        {
          "model_type": "qwen3_5_moe",
          "text_config": {
            "model_type": "qwen3_5_moe_text",
            "mtp_num_hidden_layers": 1
          }
        }
        """.data(using: .utf8)!
        let status = MTPBundleStatus(
            bundleHasMTP: true,
            configuredLayers: 1,
            tensorCount: 42,
            visionTensorCount: 333,
            mode: .preservedEnabled,
            nativeMTPTuning: NativeMTPTuning(
                bestDepth: 3,
                validated: true,
                outputEquivalent: true,
                artifact: "docs/internal/release-gates/qwen-depth3/result.json"))

        let shouldLoad = try await NativeMTPActivation.withExplicitRequest(true) {
            try NativeMTPActivation.shouldLoadNativeMTPWeights(
                configData: config,
                baseModelType: "qwen3_5_moe",
                status: status)
        }

        #expect(shouldLoad)
    }

    @Test("native MTP activation requires usable tuning even when tensors exist")
    func nativeMTPActivationRequiresUsableTuningEvenWhenTensorsExist() async throws {
        let config = """
        {
          "model_type": "qwen3_5_moe",
          "text_config": {
            "model_type": "qwen3_5_moe_text",
            "mtp_num_hidden_layers": 1
          }
        }
        """.data(using: .utf8)!
        let status = MTPBundleStatus(
            bundleHasMTP: true,
            configuredLayers: 1,
            tensorCount: 42,
            visionTensorCount: 333,
            mode: .preservedEnabled)

        await #expect(throws: NativeMTPActivationError.self) {
            _ = try await NativeMTPActivation.withExplicitRequest(true) {
                try NativeMTPActivation.shouldLoadNativeMTPWeights(
                    configData: config,
                    baseModelType: "qwen3_5_moe",
                    status: status)
            }
        }
    }

    @Test("native MTP activation can be requested task-locally without process env")
    func nativeMTPActivationSupportsTaskLocalRequest() async throws {
        let config = """
        {
          "model_type": "qwen3_5_moe",
          "text_config": {
            "model_type": "qwen3_5_moe_text",
            "mtp_num_hidden_layers": 1
          }
        }
        """.data(using: .utf8)!
        let status = MTPBundleStatus(
            bundleHasMTP: true,
            configuredLayers: 1,
            tensorCount: 42,
            visionTensorCount: 333,
            mode: .preservedEnabled,
            nativeMTPTuning: NativeMTPTuning(
                bestDepth: 3,
                validated: true,
                outputEquivalent: true,
                artifact: "docs/internal/release-gates/qwen-depth3/result.json"))

        let active = try await NativeMTPActivation.withExplicitRequest(true) {
            try NativeMTPActivation.shouldLoadNativeMTPWeights(
                configData: config,
                baseModelType: "qwen3_5_moe",
                status: status)
        }
        #expect(active)
    }

    @Test("native MTP task-local false overrides poisoned process env")
    func nativeMTPTaskLocalFalseOverridesPoisonedEnv() async throws {
        setenv("VMLX_NATIVE_MTP", "1", 1)
        defer { unsetenv("VMLINUX_NATIVE_MTP"); unsetenv("VMLX_NATIVE_MTP") }

        let config = """
        {
          "model_type": "qwen3_5_moe",
          "text_config": {
            "model_type": "qwen3_5_moe_text",
            "mtp_num_hidden_layers": 1
          }
        }
        """.data(using: .utf8)!
        let status = MTPBundleStatus(
            bundleHasMTP: true,
            configuredLayers: 1,
            tensorCount: 42,
            visionTensorCount: 333,
            mode: .preservedEnabled)

        let inactive = try await NativeMTPActivation.withExplicitRequest(false) {
            try NativeMTPActivation.shouldLoadNativeMTPWeights(
                configData: config,
                baseModelType: "qwen3_5_moe",
                status: status)
        }
        #expect(!inactive)
    }

    @Test("native MTP env aliases keep canonical VMLX spelling live")
    func nativeMTPEnvAliasesKeepCanonicalVMLXSpellingLive() {
        setenv("VMLX_NATIVE_MTP", "1", 1)
        setenv("VMLX_NATIVE_MTP_HYBRID_VERIFY", "chunk_lazy_repair", 1)
        defer {
            unsetenv("VMLX_NATIVE_MTP")
            unsetenv("VMLINUX_NATIVE_MTP")
            unsetenv("VMLX_NATIVE_MTP_HYBRID_VERIFY")
            unsetenv("VMLINUX_NATIVE_MTP_HYBRID_VERIFY")
        }

        #expect(NativeMTPActivation.isExplicitlyRequested)
        #expect(NativeMTPVerifierStatePolicy.mode == .lazyRepair)

        unsetenv("VMLX_NATIVE_MTP")
        unsetenv("VMLX_NATIVE_MTP_HYBRID_VERIFY")
        setenv("VMLINUX_NATIVE_MTP", "1", 1)
        setenv("VMLINUX_NATIVE_MTP_HYBRID_VERIFY", "chunk_fast", 1)

        #expect(NativeMTPActivation.isExplicitlyRequested)
        #expect(NativeMTPVerifierStatePolicy.mode == .captureCommit)
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

    @Test("JANG config parses role-level MXTQ metadata")
    func jangConfigParsesRoleLevelMXTQMetadata() throws {
        let config = try JangLoader.parseConfig(from: [
            "quantization": [
                "method": "affine+mxtq",
                "group_size": 64,
                "bits_default": 4,
            ] as [String: Any],
            "mxtq_bits": [
                "attention": 8,
                "shared_expert": 8,
                "mamba_proj": 8,
                "routed_expert": 4,
                "embed_tokens": 8,
                "lm_head": 8,
            ] as [String: Any],
        ])

        #expect(config.quantization.blockSize == 64)
        #expect(config.mxtqBits["attention"] == 8)
        #expect(config.mxtqBits["shared_expert"] == 8)
        #expect(config.mxtqBits["mamba_proj"] == 8)
        #expect(config.mxtqBits["routed_expert"] == 4)
        #expect(config.mxtqBits["embed_tokens"] == 8)
        #expect(config.mxtqBits["lm_head"] == 8)
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
            configEvidence: ["text_config.mtp_num_hidden_layers=1"],
            nativeMTPTuning: NativeMTPTuning(
                bestDepth: 3,
                validated: true,
                outputEquivalent: true,
                artifact: "docs/internal/release-gates/qwen-depth3/result.json"))
        let configuration = ModelConfiguration(
            directory: root,
            mtpStatus: status)

        let resolved = configuration.resolved(modelDirectory: root, tokenizerDirectory: root)

        #expect(configuration.mtpStatus == status)
        #expect(resolved.mtpStatus == status)
        #expect(resolved.mtpStatus?.requiresAcceptRejectBeforeEnable == false)
        #expect(resolved.mtpStatus?.canAutoLaunchMTP == true)
    }

    @Test("native MTP auto decode policy is real tensor evidence gated")
    func nativeMTPAutoDecodePolicyIsRealTensorEvidenceGated() throws {
        let denseMXFP8Config = """
        {
          "model_type": "qwen3_vl",
          "text_config": { "model_type": "qwen3_5", "mtp_num_hidden_layers": 1 },
          "quantization": { "mode": "mxfp8", "bits": 8 }
        }
        """.data(using: .utf8)!
        let moeMXFP8Config = """
        {
          "model_type": "qwen3_5_moe",
          "text_config": { "model_type": "qwen3_5_moe_text", "mtp_num_hidden_layers": 1 },
          "quantization": { "mode": "mxfp8", "bits": 8 }
        }
        """.data(using: .utf8)!
        let preserved = MTPBundleStatus(
            bundleHasMTP: true,
            configuredLayers: 1,
            tensorCount: 31,
            visionTensorCount: 333,
            mode: .preservedEnabled,
            nativeMTPTuning: NativeMTPTuning(
                bestDepth: 3,
                validated: true,
                outputEquivalent: true,
                artifact: "docs/internal/release-gates/qwen-depth3/result.json"))
        let verified = MTPBundleStatus(
            bundleHasMTP: true,
            configuredLayers: 1,
            tensorCount: 31,
            visionTensorCount: 333,
            mode: .speculativeVerified,
            nativeMTPTuning: NativeMTPTuning(
                bestDepth: 3,
                validated: true,
                outputEquivalent: true,
                artifact: "docs/internal/release-gates/qwen-depth3/result.json"))
        let blockedTuning = MTPBundleStatus(
            bundleHasMTP: true,
            configuredLayers: 1,
            tensorCount: 31,
            visionTensorCount: 333,
            mode: .speculativeVerified,
            nativeMTPTuning: NativeMTPTuning(
                validated: false,
                outputEquivalent: false,
                blocked: true,
                reason: "forced diagnostic was not production-valid"))
        let noTuning = MTPBundleStatus(
            bundleHasMTP: true,
            configuredLayers: 1,
            tensorCount: 31,
            visionTensorCount: 333,
            mode: .speculativeVerified)
        let missingWeights = MTPBundleStatus(
            bundleHasMTP: false,
            configuredLayers: 1,
            tensorCount: 0,
            mode: .metadataOnlyMissingWeights)

        let denseAuto = NativeMTPAutoDecodePolicy.recommendation(
            configData: denseMXFP8Config,
            jangConfig: nil,
            status: preserved)
        #expect(denseAuto?.depth == 3)
        #expect(denseAuto?.verifierMode == "sequential_repair")
        #expect(denseAuto?.evidence.contains("tuning_file=vmlx_mtp_tuning.json") == true)
        #expect(NativeMTPAutoDecodePolicy.recommendation(
            configData: denseMXFP8Config,
            jangConfig: nil,
            status: noTuning) == nil)
        #expect(NativeMTPAutoDecodePolicy.recommendation(
            configData: denseMXFP8Config,
            jangConfig: nil,
            status: missingWeights,
            requireVerifiedRuntime: false) == nil)

        let denseReporting = NativeMTPAutoDecodePolicy.recommendation(
            configData: denseMXFP8Config,
            jangConfig: nil,
            status: preserved,
            requireVerifiedRuntime: false)
        #expect(denseReporting?.depth == 3)
        #expect(denseReporting?.verifierMode == "sequential_repair")

        let moeVerified = NativeMTPAutoDecodePolicy.recommendation(
            configData: moeMXFP8Config,
            jangConfig: nil,
            status: verified)
        #expect(moeVerified?.depth == 3)
        #expect(moeVerified?.verifierMode == "sequential_repair")

        let jang2k = NativeMTPAutoDecodePolicy.recommendation(
            configData: denseMXFP8Config,
            jangConfig: JangConfig(
                quantization: JangQuantization(
                    method: "jang",
                    profile: "JANG_2K",
                    targetBits: 2,
                    actualBits: 2,
                    bitWidthsUsed: [2, 3, 6, 8]),
                sourceModel: JangSourceModel(architecture: "qwen3_5_moe"),
                architecture: JangArchitecture(hasMoE: true)),
            status: blockedTuning)
        #expect(jang2k == nil)
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

    @Test("Qwen3.5 text SSM cache records accepted-prefix offsets")
    func qwen35TextSSMCacheRecordsAcceptedPrefixOffsets() throws {
        let source = try Self.source("Libraries/MLXLLM/Models/Qwen35.swift")

        #expect(source.contains("cache.offset += S"))
        #expect(source.contains("offset: baseOffset + prefixLength"))
        #expect(!source.contains("offset: baseOffset)"))
    }

    @Test("Qwen3.5 SSM accepted-prefix capture advances token by token")
    func qwen35SSMAcceptedPrefixCaptureAdvancesTokenByToken() throws {
        let textSource = try Self.source("Libraries/MLXLLM/Models/Qwen35.swift")
        let vlSource = try Self.source("Libraries/MLXVLM/Models/Qwen35.swift")

        for source in [textSource, vlSource] {
            #expect(source.contains("var recurrentState = initialState"))
            #expect(source.contains("let tokenRange = (prefixLength - 1) ..< prefixLength"))
            #expect(source.contains("state: recurrentState"))
            #expect(source.contains("recurrentState = prefixState"))
            #expect(source.contains("stepMask(mask, index: prefixLength - 1)"))
            #expect(!source.contains("prefixMask(mask, length: prefixLength)"))
        }
    }

    @Test("Qwen3.5 GDN verifier state has strict and fast modes")
    func qwen35GDNVerifierStateHasStrictAndFastModes() throws {
        let textGDN = try Self.source("Libraries/MLXLLM/Models/GatedDelta.swift")
        let vlGDN = try Self.source("Libraries/MLXVLM/Models/Qwen35.swift")

        for source in [textGDN, vlGDN] {
            #expect(source.contains("roundStateEachStep"))
            #expect(source.contains("_strict"))
            #expect(source.contains("_fast"))
            #expect(source.contains("state[i] = static_cast<float>(static_cast<InT>(state[i]));"))
            #expect(source.contains("roundStateEachStep ? newState.asType(q.dtype) : newState"))
        }

        let textQwen = try Self.source("Libraries/MLXLLM/Models/Qwen35.swift")
        let vlQwen = try Self.source("Libraries/MLXVLM/Models/Qwen35.swift")
        for source in [textQwen, vlQwen] {
            #expect(source.contains("NativeMTPVerifierStatePolicy.shouldRoundGDNStateEachVerifierStep"))
            #expect(source.contains("NativeMTPVerifierStatePolicy.shouldRecordAcceptedPrefixStates"))
        }
    }

    @Test("native MTP prefix recurrent snapshots materialize before cache commit")
    func nativeMTPPrefixRecurrentSnapshotsMaterializeBeforeCacheCommit() throws {
        let source = try Self.source("Libraries/MLXLMCommon/KVCache.swift")
        let record = try #require(source.range(
            of: "public func recordPrefixCommitState(length: Int, arrays: [MLXArray], offset: Int)"))
        let eval = try #require(source.range(of: "MLX.eval(snapshotArrays)"))
        let store = try #require(source.range(of: "prefixCommitStates[length] = PrefixCommitState"))

        #expect(record.lowerBound < eval.lowerBound)
        #expect(eval.lowerBound < store.lowerBound)
    }

    @Test("native MTP lazy repair skips capture until partial rejection")
    func nativeMTPLazyRepairSkipsCaptureUntilPartialRejection() throws {
        let runtime = try Self.source("Libraries/MLXLMCommon/SpecDec/MTPRuntime.swift")
        let iterator = try Self.source(
            "Libraries/MLXLMCommon/SpecDec/NativeMTPTokenIterator.swift")

        #expect(runtime.contains("case lazyRepair = \"lazy_repair\""))
        #expect(runtime.contains("\"chunk_lazy_repair\""))
        #expect(runtime.contains("mode != .lazyRepair"))
        #expect(iterator.contains("private static func requiresLazyChunkRepair"))
        #expect(iterator.contains("lazyChunkRepair && accepted < drafts.count"))
        #expect(iterator.contains("verifierMode = NativeMTPVerifierStatePolicy.mode == .lazyRepair"))
    }

    @Test("native MTP partial reject refreshes private draft cache")
    func nativeMTPPartialRejectRefreshesPrivateDraftCache() throws {
        let source = try Self.source(
            "Libraries/MLXLMCommon/SpecDec/NativeMTPTokenIterator.swift")

        #expect(source.contains("mtpCache = model.makeNativeMTPCache()"))
        #expect(source.contains("mtpCacheRefreshCount += 1"))
        #expect(!source.contains("trimPromptCache(mtpCache"))
    }

    @Test("native MTP greedy verifier batches target argmax rows")
    func nativeMTPGreedyVerifierBatchesTargetArgmaxRows() throws {
        let source = try Self.source(
            "Libraries/MLXLMCommon/SpecDec/NativeMTPTokenIterator.swift")

        #expect(source.contains("batchedGreedyTargetTokenIds"))
        #expect(source.contains("argMax(candidateLogits, axis: -1).asType(.int32)"))
        #expect(source.contains("processor == nil"))
        #expect(source.contains("let batch = batchedGreedyTargetTokenIds("))
    }

    @Test("native MTP chunk verifier env alias is applied in iterator")
    func nativeMTPChunkVerifierEnvAliasIsAppliedInIterator() throws {
        let source = try Self.source(
            "Libraries/MLXLMCommon/SpecDec/NativeMTPTokenIterator.swift")

        #expect(source.contains("private static func nativeMTPHybridVerifySetting"))
        #expect(source.contains("env[\"VMLX_NATIVE_MTP_HYBRID_VERIFY\"]"))
        #expect(source.contains("env[\"VMLINUX_NATIVE_MTP_HYBRID_VERIFY\"]"))
        #expect(source.contains("switch nativeMTPHybridVerifySetting()?"))
    }

    @Test("RunBench perf can use bundle generation defaults")
    func runBenchPerfCanUseBundleGenerationDefaults() throws {
        let source = try Self.source("RunBench/Bench.swift")

        #expect(source.contains("BENCH_PERF_USE_GENERATION_CONFIG"))
        #expect(source.contains("samplingSource"))
        #expect(source.contains("BENCH_PERF_SEED"))
        #expect(source.contains("randomSeed: perfSeed"))
        #expect(source.contains("GenerateParameters(generationConfig: context.configuration.generationDefaults"))
        #expect(source.contains("samplingSource=%@"))
        #expect(source.contains("BENCH_PERF_PHASE_SNAPSHOT"))
        #expect(source.contains("PERF_PHASE label=%@"))
    }

    @Test("Qwen native MTP phase diagnostics include draft sidecar phases")
    func qwenNativeMTPPhaseDiagnosticsIncludeDraftSidecarPhases() throws {
        let llmSource = try Self.source("Libraries/MLXLLM/Models/Qwen35.swift")
        let vlmSource = try Self.source("Libraries/MLXVLM/Models/Qwen35.swift")

        #expect(llmSource.contains("\"llm_mtp_block\""))
        #expect(llmSource.contains("\"llm_mtp_lm_head\""))
        #expect(vlmSource.contains("\"vlm_mtp_block\""))
        #expect(vlmSource.contains("\"vlm_mtp_lm_head\""))
    }

    @Test("native MTP verifier mode telemetry is explicit")
    func nativeMTPVerifierModeTelemetryIsExplicit() throws {
        let source = try Self.source(
            "Libraries/MLXLMCommon/SpecDec/NativeMTPTokenIterator.swift")

        #expect(source.contains("VMLX_NATIVE_MTP_HYBRID_VERIFY"))
        #expect(source.contains("verifierMode=%@"))
        #expect(source.contains("targetForwards=%d"))
        #expect(source.contains("verifyInputTokens=%d"))
        #expect(source.contains("repairForwards=%d"))
        #expect(source.contains("chunk_commit"))
        #expect(source.contains("sequential_repair"))
    }

    @Test("native MTP chunk verifier stays opt-in for Mamba cache")
    func nativeMTPChunkVerifierStaysOptInForMambaCache() throws {
        let source = try Self.source(
            "Libraries/MLXLMCommon/SpecDec/NativeMTPTokenIterator.swift")

        #expect(source.contains("\"chunk_commit\""))
        #expect(source.contains("\"chunk_replay\""))
        #expect(source.contains("private static func requiresChunkTokenReplayRepair"))
        #expect(source.contains("speculativeSampler: SpeculativeSamplingController"))
        #expect(source.contains("!speculativeSampler.isGreedy && cache.contains(where: { $0 is MambaCache })"))
        #expect(source.contains("case invalidDepth(Int)"))
        #expect(source.contains("throw NativeMTPRuntimeError.invalidDepth(requestedDepth)"))
        #expect(source.contains("case \"sequential\", \"sequential_repair\", \"repair\":"))
        #expect(source.contains("return cache.contains { $0 is MambaCache }"))
    }

    @Test("native MTP chunk-lazy env overrides stochastic Mamba fallback")
    func nativeMTPChunkLazyEnvOverridesStochasticMambaFallback() throws {
        let source = try Self.source(
            "Libraries/MLXLMCommon/SpecDec/NativeMTPTokenIterator.swift")

        let overrideSwitch = try #require(source.range(
            of: "switch nativeMTPHybridVerifySetting()?.lowercased()"))
        let stochasticMambaFallback = try #require(source.range(
            of: "if !speculativeSampler.isGreedy && cache.contains(where: { $0 is MambaCache })"))

        #expect(overrideSwitch.lowerBound < stochasticMambaFallback.lowerBound)
    }

    @Test("BatchEngine.generate rejects native MTP without an active MTP head")
    func batchEngineGenerateRejectsNativeMTPWithoutActiveHead() async throws {
        try await FocusedMLXTestSupport.withLock {
            let model = try Qwen35TextModel(Self.tinyQwen35Config(mtpLayers: 0))
            let context = Self.nativeMTPDispatchContext(model: model)
            let engine = BatchEngine(context: context, maxBatchSize: 2)
            var params = GenerateParameters(maxTokens: 4, temperature: 0)
            params.draftStrategy = .nativeMTP(depth: 3)

            let stream = await engine.generate(
                input: LMInput(tokens: MLXArray([3, 5, 7])),
                parameters: params)
            let info = await Self.collectInfo(from: stream)

            #expect(info.count == 1)
            #expect(info.cancelled == 1)
        }
    }

    @Test("BatchEngine.generate routes active native MTP through the exclusive lane")
    func batchEngineGenerateRunsActiveNativeMTP() async throws {
        try await FocusedMLXTestSupport.withLock {
            let model = FocusedNativeMTPProbeTarget()
            #expect(model.nativeMTPAvailable)
            let context = Self.nativeMTPDispatchContext(model: model)
            let engine = BatchEngine(context: context, maxBatchSize: 2)
            var params = GenerateParameters(maxTokens: 4, temperature: 0)
            params.draftStrategy = .nativeMTP(depth: 3)

            let stream = await engine.generate(
                input: LMInput(tokens: MLXArray([3, 5, 7])),
                parameters: params)
            let info = await Self.collectInfo(from: stream)

            #expect(info.count == 1)
            #expect(info.cancelled == 0)
        }
    }

    @Test("BatchEngine.submit rejects native MTP instead of silently batching AR")
    func batchEngineSubmitRejectsNativeMTP() async throws {
        try await FocusedMLXTestSupport.withLock {
            let model = FocusedNativeMTPProbeTarget()
            let context = Self.nativeMTPDispatchContext(model: model)
            let engine = BatchEngine(context: context, maxBatchSize: 2)
            var params = GenerateParameters(maxTokens: 4, temperature: 0)
            params.draftStrategy = .nativeMTP(depth: 3)

            let (_, stream) = await engine.submit(
                input: LMInput(tokens: MLXArray([3, 5, 7])),
                parameters: params)
            var cancelled = 0
            for await event in stream {
                if case .info(let info) = event, info.stopReason == .cancelled {
                    cancelled += 1
                }
            }

            #expect(cancelled == 1)
        }
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

    @Test("shape-walk quantization supports JANG_2K 3-bit group-128 projections")
    func shapeWalkQuantizationSupportsJang2KThreeBitGroup128() {
        let inferred = JangLoader.inferBitWidthAndGroupSize(
            packedDim: 192,
            numGroups: 16,
            knownGroupSize: 128,
            bitWidthsUsed: [2, 3, 6, 8],
            expectedInDim: 2048)

        #expect(inferred.bits == 3)
        #expect(inferred.groupSize == 128)
    }

    @Test("shape-walk quantization honors known group size for stock MLX affine embeddings")
    func shapeWalkQuantizationHonorsKnownGroupSizeForStockMLXEmbeddings() {
        let direct = JangLoader.inferBitWidthAndGroupSize(
            packedDim: 256,
            numGroups: 32,
            knownGroupSize: 64)

        #expect(direct.bits == 4)
        #expect(direct.groupSize == 64)

        let weights: [String: MLXArray] = [
            "language_model.model.embed_tokens.weight": MLXArray.zeros(
                [248_320, 256], dtype: .uint32),
            "language_model.model.embed_tokens.scales": MLXArray.zeros(
                [248_320, 32], dtype: .float32),
        ]
        let inferred = JangLoader.inferPerLayerQuantizationFromShapes(
            weights: weights,
            defaultBits: 4,
            defaultGroupSize: 64,
            defaultMode: .affine)

        #expect(inferred?.quantization?.bits == 4)
        #expect(inferred?.quantization?.groupSize == 64)
        #expect(inferred?.perLayerQuantization["language_model.model.embed_tokens"] == nil)
    }

    @Test("shape-walk quantization uses hidden size for JANG_2K routed expert inputs")
    func shapeWalkQuantizationUsesHiddenSizeForJang2KRoutedExpertInputs() {
        let base = "model.layers.0.mlp.switch_mlp.gate_proj"
        let weights: [String: MLXArray] = [
            "\(base).weight": MLXArray.zeros([1, 128], dtype: .uint32),
            "\(base).scales": MLXArray.zeros([1, 16], dtype: .float32),
        ]

        let inferred = JangLoader.inferPerLayerQuantization(
            weights: weights,
            jangConfig: JangConfig(
                quantization: JangQuantization(
                    blockSize: 32,
                    bitWidthsUsed: [2, 3, 6, 8])),
            hiddenSizeHint: 2048,
            validInDims: [512, 2048],
            declaredDefaultQuantization: BaseConfiguration.Quantization(
                groupSize: 32, bits: 8))

        if case .quantize(let override)? = inferred.perLayerQuantization[base] {
            #expect(override.bits == 2)
            #expect(override.groupSize == 128)
        } else {
            Issue.record("Expected routed expert input override to use hidden input dim")
        }
    }

    @Test("shape-walk honors role-level MXTQ metadata for dense JANGTQ roles")
    func shapeWalkHonorsRoleLevelMXTQMetadata() {
        let base = "backbone.layers.0.mixer.in_proj"
        let weights: [String: MLXArray] = [
            "\(base).weight": MLXArray.zeros([1, 128], dtype: .uint32),
            "\(base).scales": MLXArray.zeros([1, 8], dtype: .float32),
        ]

        let inferred = JangLoader.inferPerLayerQuantization(
            weights: weights,
            jangConfig: JangConfig(
                quantization: JangQuantization(blockSize: 64),
                mxtqBits: [
                    "mamba_proj": 8,
                    "routed_expert": 4,
                ]),
            declaredDefaultQuantization: BaseConfiguration.Quantization(
                groupSize: 64, bits: 4))

        if case .quantize(let override)? = inferred.perLayerQuantization[base] {
            #expect(override.bits == 8)
            #expect(override.groupSize == 64)
        } else {
            Issue.record("Expected mamba projection override from role-level mxtq_bits")
        }
    }

    @Test("shape-walk quantization uses value dim for Qwen3.6 linear attention output")
    func shapeWalkQuantizationUsesValueDimForQwen36LinearAttentionOutput() {
        let base = "model.language_model.layers.0.linear_attn.out_proj"
        let weights: [String: MLXArray] = [
            "\(base).weight": MLXArray.zeros([1, 256], dtype: .uint32),
            "\(base).scales": MLXArray.zeros([1, 32], dtype: .float32),
        ]

        let inferred = JangLoader.inferPerLayerQuantization(
            weights: weights,
            jangConfig: JangConfig(
                quantization: JangQuantization(
                    blockSize: 32,
                    bitWidthsUsed: [2, 3, 6, 8])),
            hiddenSizeHint: 2048,
            linearAttnValueDimHint: 4096,
            validInDims: [512, 2048, 4096, 8192],
            declaredDefaultQuantization: BaseConfiguration.Quantization(
                groupSize: 32, bits: 8))

        if case .quantize(let override)? = inferred.perLayerQuantization[base] {
            #expect(override.bits == 2)
            #expect(override.groupSize == 128)
        } else {
            Issue.record("Expected linear attention output override to use value dim")
        }
    }

    @Test("shape-walk quantization uses ZAYA CCA output width for o_proj")
    func shapeWalkQuantizationUsesZayaCCAOutputWidthForOProj() {
        let base = "model.layers.0.sub.o_proj"
        let weights: [String: MLXArray] = [
            "\(base).weight": MLXArray.zeros([2048, 256], dtype: .uint32),
            "\(base).scales": MLXArray.zeros([2048, 32], dtype: .float32),
        ]

        let inferred = JangLoader.inferPerLayerQuantization(
            weights: weights,
            jangConfig: JangConfig(
                quantization: JangQuantization(
                    blockSize: 32,
                    bitWidthsUsed: [4, 8])),
            hiddenSizeHint: 2048,
            validInDims: [2048],
            declaredDefaultQuantization: BaseConfiguration.Quantization(
                groupSize: 32, bits: 8))

        if case .quantize(let override)? = inferred.perLayerQuantization[base] {
            #expect(override.bits == 8)
            #expect(override.groupSize == 32)
        } else {
            Issue.record("Expected ZAYA CCA o_proj override to use 1024-wide attention input")
        }
    }

    @Test("JANG gate dequantization uses hidden size for 6-bit shared expert gate")
    func jangGateDequantizationUsesHiddenSizeForSixBitSharedExpertGate() {
        let base = "model.layers.0.mlp.shared_expert_gate"
        var weights: [String: MLXArray] = [
            "\(base).weight": MLXArray.zeros([1, 384], dtype: .uint32),
            "\(base).scales": MLXArray.zeros([1, 32], dtype: .float32),
            "\(base).biases": MLXArray.zeros([1, 32], dtype: .float32),
        ]

        JangLoader.dequantizeMoEGates(
            weights: &weights,
            groupSize: 128,
            bitWidthsUsed: [2, 3, 6, 8],
            hiddenSizeHint: 2048)

        #expect(weights["\(base).weight"]?.shape == [1, 2048])
        #expect(weights["\(base).scales"] == nil)
        #expect(weights["\(base).biases"] == nil)
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

    @Test("Qwen3.5 sanitize honors explicit norm convention metadata")
    func qwen35SanitizeHonorsExplicitNormConventionMetadata() async throws {
        try await FocusedMLXTestSupport.withLock {
            let model = try Qwen35TextModel(Self.tinyQwen35Config(mtpLayers: 1))
            let norm = MLXArray([Float](repeating: 0.5, count: 16))
            let conv = MLXArray.zeros([16, 4, 4], dtype: .float32)

            let plusOne = model.sanitize(weights: [
                "model.layers.0.input_layernorm.weight": norm,
                "model.layers.0.linear_attn.conv1d.weight": conv,
                "mtp.layers.0.input_layernorm.weight": norm,
            ], metadata: ["norm_convention": "qwen3_5_language_mlx_plus_one"])
            #expect(
                plusOne["model.layers.0.input_layernorm.weight"]?.asArray(Float.self)
                    == [Float](repeating: 1.5, count: 16))
            #expect(
                plusOne["mtp.layers.0.input_layernorm.weight"]?.asArray(Float.self)
                    == [Float](repeating: 1.5, count: 16))

            let explicitNative = model.sanitize(weights: [
                "model.layers.0.input_layernorm.weight": norm,
                "model.layers.0.linear_attn.conv1d.weight": conv,
                "mtp.layers.0.input_layernorm.weight": norm,
            ], metadata: ["norm_convention": "mlx"])
            #expect(
                explicitNative["model.layers.0.input_layernorm.weight"]?.asArray(Float.self)
                    == [Float](repeating: 0.5, count: 16))
            #expect(
                explicitNative["mtp.layers.0.input_layernorm.weight"]?.asArray(Float.self)
                    == [Float](repeating: 0.5, count: 16))
        }
    }

    @Test("Qwen3.5 sanitize can shift raw MTP norms without shifting MLX-ready backbone")
    func qwen35SanitizeShiftsRawMTPNormsIndependently() async throws {
        try await FocusedMLXTestSupport.withLock {
            let model = try Qwen35TextModel(Self.tinyQwen35Config(mtpLayers: 1))
            let backboneReady = MLXArray([Float](repeating: 1.0, count: 16))
            let mtpRaw = MLXArray([Float](repeating: 0.0, count: 16))

            let sanitized = model.sanitize(weights: [
                "model.layers.0.input_layernorm.weight": backboneReady,
                "mtp.layers.0.input_layernorm.weight": mtpRaw,
                "mtp.pre_fc_norm_hidden.weight": mtpRaw,
            ])

            #expect(
                sanitized["model.layers.0.input_layernorm.weight"]?.asArray(Float.self)
                    == [Float](repeating: 1.0, count: 16))
            #expect(
                sanitized["mtp.layers.0.input_layernorm.weight"]?.asArray(Float.self)
                    == [Float](repeating: 1.0, count: 16))
            #expect(
                sanitized["mtp.pre_fc_norm_hidden.weight"]?.asArray(Float.self)
                    == [Float](repeating: 1.0, count: 16))
        }
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
        if ProcessInfo.processInfo.environment["VMLX_MTP_REAL_BUNDLE_EXPECTS_VL"] == "1" {
            #expect(status.visionTensorCount > 0)
            #expect(status.bundleHasVision)
        }

        let root = URL(fileURLWithPath: path)
        let configData = try Data(contentsOf: root.appendingPathComponent("config.json"))
        let jangConfig = try? JangLoader.loadConfig(at: root)
        let settings = VMLXServerRuntimeSettings()
        let launch = settings.resolvedMTPLaunch(
            configData: configData,
            jangConfig: jangConfig,
            status: status)
        let loadConfiguration = settings.resolvedLoadConfiguration(
            base: .off,
            configData: configData,
            jangConfig: jangConfig,
            status: status)
        if ProcessInfo.processInfo.environment["VMLX_MTP_REAL_BUNDLE_EXPECTS_BLOCKED"] == "1" {
            #expect(launch.launchMode == .off)
            #expect(!loadConfiguration.nativeMTP)
            var forceOn = settings
            forceOn.mtp.mode = .forceOn
            #expect(forceOn.validationIssues(
                configData: configData,
                jangConfig: jangConfig,
                mtpStatus: status).contains {
                    $0.severity == .error && $0.field == "mtp.mode"
                })
        } else {
            #expect(status.canAutoLaunchMTP)
            #expect(launch.launchMode == .speculative)
            #expect(launch.recommendation?.depth == status.nativeMTPTuning?.usableBestDepth)
            #expect(launch.recommendation?.evidence.contains("tuning_file=vmlx_mtp_tuning.json") == true)
            #expect(loadConfiguration.nativeMTP)
        }
    }

    @Test("Qwen3.6 loaders propagate norm convention metadata")
    func qwen36LoadersPropagateNormConventionMetadata() throws {
        let loadSource = try Self.source("Libraries/MLXLMCommon/Load.swift")
        let vlmSource = try Self.source("Libraries/MLXVLM/Models/Qwen35.swift")

        #expect(loadSource.contains("loadJangConfigSanitizeMetadata"))
        #expect(loadSource.contains("\"norm_convention\""))
        #expect(vlmSource.contains("normConvention: Self.normConvention(metadata)"))
        #expect(vlmSource.contains("usesQwenPlusOneNormConvention"))
    }

    @Test("LLM and VLM factories carry MTP tuning status like generation config")
    func factoriesCarryMTPTuningStatusLikeGenerationConfig() throws {
        let files = [
            "Libraries/MLXLLM/LLMModelFactory.swift",
            "Libraries/MLXVLM/VLMModelFactory.swift",
        ]

        for file in files {
            let source = try Self.source(file)
            #expect(source.contains("MTPBundleInspector.inspect("), "\(file) must inspect MTP")
            #expect(
                source.contains("NativeMTPActivation.shouldLoadNativeMTPWeights"),
                "\(file) must resolve native-MTP activation before loading weights")
            #expect(
                source.contains("loadPreservedMTP: loadNativeMTP"),
                "\(file) must pass the resolved MTP decision to weight loading")
            #expect(
                source.contains("generationDefaults: generationConfig"),
                "\(file) must keep bundle generation_config defaults wired")
            #expect(
                source.contains("mtpStatus: mtpStatus"),
                "\(file) must carry MTP status into ModelConfiguration")
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

    private static func source(_ relativePath: String) throws -> String {
        let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath, isDirectory: true)
        let url = root.appendingPathComponent(relativePath)
        return try String(contentsOf: url, encoding: .utf8)
    }

    private static func tinyQwen35Config(mtpLayers: Int) throws -> Qwen35TextConfiguration {
        let data = """
        {
          "hidden_size": 16,
          "num_hidden_layers": 1,
          "intermediate_size": 32,
          "num_attention_heads": 2,
          "num_key_value_heads": 2,
          "linear_num_value_heads": 1,
          "linear_num_key_heads": 1,
          "linear_key_head_dim": 8,
          "linear_value_head_dim": 8,
          "linear_conv_kernel_dim": 4,
          "head_dim": 8,
          "vocab_size": 32,
          "tie_word_embeddings": false,
          "mtp_num_hidden_layers": \(mtpLayers)
        }
        """.data(using: .utf8)!
        return try JSONDecoder().decode(Qwen35TextConfiguration.self, from: data)
    }

    private static func nativeMTPDispatchContext(model: any LanguageModel) -> ModelContext {
        let tokenizer = FocusedMTPTokenizer()
        return ModelContext(
            configuration: ModelConfiguration(id: "focused-native-mtp"),
            model: model,
            processor: FocusedMTPProcessor(tokenizer: tokenizer),
            tokenizer: tokenizer)
    }

    private static func collectInfo(
        from stream: AsyncStream<Generation>
    ) async -> (count: Int, cancelled: Int) {
        var count = 0
        var cancelled = 0
        for await event in stream {
            if case .info(let info) = event {
                count += 1
                if info.stopReason == .cancelled {
                    cancelled += 1
                }
            }
        }
        return (count, cancelled)
    }
}

private struct FocusedMTPTokenizer: Tokenizer {
    let vocabularySize = 64
    let eosTokenId: Int? = 60
    let unknownTokenId: Int? = 61
    let bosToken: String? = nil
    let eosToken: String? = nil
    let unknownToken: String? = nil

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        [3, 5, 7]
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        tokenIds.map(String.init).joined(separator: " ")
    }

    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { String(id) }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        [3, 5, 7]
    }
}

private struct FocusedMTPProcessor: UserInputProcessor {
    let tokenizer: any Tokenizer

    func prepare(input: UserInput) async throws -> LMInput {
        LMInput(tokens: MLXArray([3, 5, 7]))
    }
}

private final class FocusedNativeMTPProbeTarget: Module, LanguageModel, NativeMTPModel,
    KVCacheDimensionProvider, @unchecked Sendable
{
    var kvHeads: [Int] { [1] }
    var nativeMTPAvailable: Bool { true }

    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        preconditionFailure("BatchEngine native MTP dispatch fell through to AR callAsFunction")
    }

    func makeNativeMTPCache() -> [KVCache] {
        newCache(parameters: nil)
    }

    func nativeBackboneForward(_ inputs: MLXArray, cache: [KVCache]?) -> NativeMTPForwardResult {
        NativeMTPForwardResult(
            logits: logits(for: inputs),
            hiddenStates: hiddenStates(for: inputs))
    }

    func nativeMTPForward(
        hiddenStates: MLXArray,
        nextTokenIds: MLXArray,
        cache: [KVCache]?
    ) -> NativeMTPForwardResult {
        NativeMTPForwardResult(
            logits: logits(for: nextTokenIds),
            hiddenStates: self.hiddenStates(for: nextTokenIds))
    }

    private func sequenceLength(_ inputs: MLXArray) -> Int {
        inputs.ndim >= 2 ? inputs.dim(1) : inputs.size
    }

    private func logits(for inputs: MLXArray) -> MLXArray {
        MLXArray.zeros([1, sequenceLength(inputs), 16])
    }

    private func hiddenStates(for inputs: MLXArray) -> MLXArray {
        MLXArray.zeros([1, sequenceLength(inputs), 4])
    }
}
