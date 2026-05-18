// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXLMCommon
import Testing

@Suite("VMLX server runtime settings")
struct VMLXServerRuntimeSettingsTests {
    @Test("defaults preserve engine and bundle sampling decisions")
    func defaultsPreserveEngineAndBundleSamplingDecisions() {
        let settings = VMLXServerRuntimeSettings()

        #expect(settings.network.host == "127.0.0.1")
        #expect(settings.concurrency.continuousBatching)
        #expect(settings.cache.prefix.enabled)
        #expect(settings.cache.pagedKV.enabled)
        #expect(settings.cache.blockDisk.enabled)
        #expect(settings.cache.legacyDisk.enabled == false)
        #expect(settings.cache.enableSSMReDerive)
        #expect(settings.cache.defaultMaxKVSize == nil)
        #expect(settings.cache.longPromptMultiplier == 2.0)
        #expect(settings.generation.temperature == nil)
        #expect(settings.generation.topP == nil)
        #expect(settings.generation.topK == nil)
        #expect(settings.generation.minP == nil)
        #expect(settings.generation.repetitionPenalty == nil)
        #expect(settings.mtp.keepDraftCacheSeparate)
        #expect(settings.mtp.acceptedTokensOnlyEnterBaseCache)
    }

    @Test("paged cache rejects legacy disk cache conflict")
    func pagedCacheRejectsLegacyDiskCacheConflict() {
        var settings = VMLXServerRuntimeSettings()
        settings.cache.pagedKV.enabled = true
        settings.cache.legacyDisk.enabled = true

        let issues = settings.validationIssues()
        #expect(issues.contains {
            $0.severity == .error && $0.field == "cache.legacyDisk.enabled"
        })
    }

    @Test("bundle generation config applies before server overrides")
    func bundleGenerationConfigAppliesBeforeServerOverrides() {
        var settings = VMLXServerRuntimeSettings()
        settings.generation.temperature = 0.2
        settings.generation.topK = 17

        let bundle = GenerationConfigFile(
            maxNewTokens: 99,
            temperature: 0.7,
            topP: 0.85,
            topK: 40,
            minP: 0.05,
            repetitionPenalty: 1.08,
            doSample: true)
        let fallback = GenerateParameters(
            maxTokens: 11,
            temperature: 0.6,
            topP: 1.0,
            topK: 0,
            minP: 0.0,
            repetitionPenalty: nil)

        let params = settings.resolvedGenerateParameters(
            generationConfig: bundle,
            fallback: fallback)

        #expect(params.maxTokens == 99)
        #expect(params.temperature == 0.2)
        #expect(params.topP == 0.85)
        #expect(params.topK == 17)
        #expect(params.minP == 0.05)
        #expect(params.repetitionPenalty == 1.08)
    }

    @Test("bundle top-k reaches speculative sampler probabilities")
    func bundleTopKReachesSpeculativeSamplerProbabilities() {
        FocusedMLXTestSupport.withLock {
            let settings = VMLXServerRuntimeSettings()
            let bundle = GenerationConfigFile(
                temperature: 1.0,
                topP: 1.0,
                topK: 2,
                minP: 0.0,
                doSample: true)
            let params = settings.resolvedGenerateParameters(generationConfig: bundle)
            let sampler = SpeculativeSamplingController(parameters: params)
            let logits =
                MLXArray([0.0 as Float, 4.0 as Float, 3.0 as Float, 1.0 as Float])[.newAxis, .ellipsis]

            let probabilities = sampler.probabilities(logits: logits)[0].asArray(Float.self)

            #expect(abs(probabilities[0]) < 1e-6)
            #expect(probabilities[1] > 0)
            #expect(probabilities[2] > 0)
            #expect(abs(probabilities[3]) < 1e-6)
            #expect(abs(probabilities.reduce(0, +) - 1) < 1e-5)
        }
    }

    @Test("temperature is applied before nucleus sampling")
    func temperatureIsAppliedBeforeNucleusSampling() {
        FocusedMLXTestSupport.withLock {
            let logits = MLXArray([0.0 as Float, -0.3 as Float, -0.8 as Float])[.newAxis, .ellipsis]
            let directSampler = TopPSampler(
                temperature: 0.5,
                topP: 0.55,
                randomSeed: 17)

            for _ in 0 ..< 64 {
                #expect(directSampler.sample(logits: logits).item(Int.self) == 0)
            }

            let speculativeSampler = SpeculativeSamplingController(
                parameters: GenerateParameters(
                    temperature: 0.5,
                    topP: 0.55,
                    topK: 0,
                    minP: 0.0,
                    randomSeed: 17))

            let probabilities = speculativeSampler.probabilities(logits: logits)[0].asArray(Float.self)

            #expect(abs(probabilities[0] - 1) < 1e-6)
            #expect(abs(probabilities[1]) < 1e-6)
            #expect(abs(probabilities[2]) < 1e-6)
        }
    }

    @Test("nil server sampling fields do not add fake guards")
    func nilServerSamplingFieldsDoNotAddFakeGuards() {
        let settings = VMLXServerRuntimeSettings()
        let bundle = GenerationConfigFile(doSample: false)
        let fallback = GenerateParameters(
            maxTokens: 32,
            temperature: 0.6,
            topP: 1.0,
            topK: 0,
            minP: 0.0,
            repetitionPenalty: nil)

        let params = settings.resolvedGenerateParameters(
            generationConfig: bundle,
            fallback: fallback)

        #expect(params.maxTokens == 32)
        #expect(params.temperature == 0)
        #expect(params.topP == 1.0)
        #expect(params.topK == 0)
        #expect(params.minP == 0.0)
        #expect(params.repetitionPenalty == nil)
    }

    @Test("VLM JANG load uses quantization container, not deprecated alias")
    func vlmJangLoadUsesQuantizationContainer() throws {
        let repoRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let source = try String(
            contentsOf: repoRoot.appending(path: "Libraries/MLXVLM/VLMModelFactory.swift"),
            encoding: .utf8)

        #expect(source.contains("baseConfig.quantizationContainer?.quantization"))
        #expect(!source.contains("baseConfig.quantization : nil"))
    }

    @Test("MTP auto helper requires bundle tuning file")
    func mtpAutoHelperRequiresBundleTuningFile() {
        let settings = VMLXServerRuntimeSettings()
        let missingTuning = MTPBundleStatus(
            bundleHasMTP: true,
            configuredLayers: 4,
            tensorCount: 31,
            mode: .preservedEnabled)
        let tuned = MTPBundleStatus(
            bundleHasMTP: true,
            configuredLayers: 4,
            tensorCount: 31,
            mode: .preservedEnabled,
            nativeMTPTuning: NativeMTPTuning(
                bestDepth: 3,
                validated: true,
                outputEquivalent: true,
                artifact: "docs/internal/release-gates/qwen-depth3/result.json"))

        #expect(settings.effectiveMTPLaunchMode(for: missingTuning) == .off)
        #expect(settings.effectiveMTPLaunchMode(for: tuned) == .speculative)
        #expect(settings.validationIssues(mtpStatus: tuned).isEmpty)
    }

    @Test("MTP force-on requires complete tensor evidence and tuning")
    func mtpForceOnRequiresCompleteTensorEvidenceAndTuning() {
        var settings = VMLXServerRuntimeSettings()
        settings.mtp.mode = .forceOn
        let tensorProven = MTPBundleStatus(
            bundleHasMTP: true,
            configuredLayers: 4,
            tensorCount: 31,
            mode: .preservedEnabled)
        let tuned = MTPBundleStatus(
            bundleHasMTP: true,
            configuredLayers: 4,
            tensorCount: 31,
            mode: .preservedEnabled,
            nativeMTPTuning: NativeMTPTuning(
                bestDepth: 3,
                validated: true,
                outputEquivalent: true,
                artifact: "docs/internal/release-gates/qwen-depth3/result.json"))
        let metadataOnly = MTPBundleStatus(
            bundleHasMTP: false,
            configuredLayers: 4,
            tensorCount: 0,
            mode: .metadataOnlyMissingWeights)

        #expect(settings.effectiveMTPLaunchMode(for: metadataOnly) == .blocked)
        #expect(settings.validationIssues(mtpStatus: metadataOnly).contains {
            $0.severity == .error && $0.field == "mtp.mode"
        })
        #expect(settings.effectiveMTPLaunchMode(for: tensorProven) == .blocked)
        #expect(settings.validationIssues(mtpStatus: tensorProven).contains {
            $0.severity == .error && $0.field == "mtp.mode"
        })
        #expect(settings.effectiveMTPLaunchMode(for: tuned) == .speculative)
        #expect(settings.validationIssues(mtpStatus: tuned).isEmpty)
    }

    @Test("MTP cache boundary settings are not optional")
    func mtpCacheBoundarySettingsAreNotOptional() {
        var settings = VMLXServerRuntimeSettings()
        settings.mtp.keepDraftCacheSeparate = false
        settings.mtp.acceptedTokensOnlyEnterBaseCache = false

        let fields = Set(settings.validationIssues().map(\.field))

        #expect(fields.contains("mtp.keepDraftCacheSeparate"))
        #expect(fields.contains("mtp.acceptedTokensOnlyEnterBaseCache"))
    }

    @Test("MTP launch resolution uses config policy and draft limit")
    func mtpLaunchResolutionUsesConfigPolicyAndDraftLimit() {
        let config = """
        {
          "model_type": "qwen3_5_moe",
          "text_config": { "model_type": "qwen3_5_moe_text", "mtp_num_hidden_layers": 1 },
          "quantization": { "mode": "mxfp8", "bits": 8 }
        }
        """.data(using: .utf8)!
        let verified = MTPBundleStatus(
            bundleHasMTP: true,
            configuredLayers: 1,
            tensorCount: 31,
            mode: .speculativeVerified,
            nativeMTPTuning: NativeMTPTuning(
                bestDepth: 3,
                validated: true,
                outputEquivalent: true,
                artifact: "docs/internal/release-gates/qwen-depth3/result.json"))
        var settings = VMLXServerRuntimeSettings()
        settings.mtp.draftTokenLimit = 2

        let launch = settings.resolvedMTPLaunch(
            configData: config,
            jangConfig: nil,
            status: verified)

        #expect(launch.launchMode == .speculative)
        #expect(launch.recommendation?.depth == 2)
        #expect(launch.recommendation?.verifierMode == "sequential_repair")
        #expect(launch.recommendation?.evidence.contains("server_draft_token_limit=2") == true)
        if case .nativeMTP(let depth)? = settings.resolvedMTPDraftStrategy(
            configData: config,
            jangConfig: nil,
            status: verified)
        {
            #expect(depth == 2)
        } else {
            Issue.record("Resolved MTP draft strategy did not carry the capped native depth")
        }
    }

    @Test("tensor-proven Qwen MTP auto-launch resolves D3 load and draft settings")
    func tensorProvenQwenMTPAutoLaunchResolvesD3LoadAndDraftSettings() {
        let config = """
        {
          "model_type": "qwen3_vl",
          "text_config": { "model_type": "qwen3_5_moe_text", "mtp_num_hidden_layers": 1 },
          "quantization": { "mode": "mxfp4", "bits": 4 }
        }
        """.data(using: .utf8)!
        let preserved = MTPBundleStatus(
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
        let settings = VMLXServerRuntimeSettings()

        let candidate = NativeMTPAutoDecodePolicy.recommendation(
            configData: config,
            jangConfig: nil,
            status: preserved,
            requireVerifiedRuntime: false)
        let launch = settings.resolvedMTPLaunch(
            configData: config,
            jangConfig: nil,
            status: preserved)
        let loadConfiguration = settings.resolvedLoadConfiguration(
            base: .off,
            configData: config,
            jangConfig: nil,
            status: preserved)

        #expect(candidate?.depth == 3)
        #expect(candidate?.verifierMode == "sequential_repair")
        #expect(settings.effectiveMTPLaunchMode(for: preserved) == .speculative)
        #expect(launch.launchMode == .speculative)
        #expect(loadConfiguration.nativeMTP)
        if case .nativeMTP(let depth)? = settings.resolvedMTPDraftStrategy(
            configData: config,
            jangConfig: nil,
            status: preserved)
        {
            #expect(depth == 3)
        } else {
            Issue.record("Tensor-proven Qwen MTP should resolve a native-MTP draft strategy")
        }
        #expect(settings.validationIssues(
            configData: config,
            jangConfig: nil,
            mtpStatus: preserved).isEmpty)
    }

    @Test("MTP launch resolution blocks unsupported verified profiles")
    func mtpLaunchResolutionBlocksUnsupportedVerifiedProfiles() {
        let config = """
        {
          "model_type": "qwen3_5_moe",
          "text_config": { "model_type": "qwen3_5_moe_text", "mtp_num_hidden_layers": 1 },
          "quantization": { "mode": "affine", "bits": 2 }
        }
        """.data(using: .utf8)!
        let verified = MTPBundleStatus(
            bundleHasMTP: true,
            configuredLayers: 1,
            tensorCount: 44,
            mode: .speculativeVerified)
        var settings = VMLXServerRuntimeSettings()
        settings.mtp.mode = .forceOn

        let launch = settings.resolvedMTPLaunch(
            configData: config,
            jangConfig: JangConfig(
                quantization: JangQuantization(
                    method: "jang",
                    profile: "JANG_2K",
                    targetBits: 2,
                    actualBits: 2,
                    bitWidthsUsed: [2, 3, 6, 8]),
                sourceModel: JangSourceModel(architecture: "qwen3_5_moe"),
                architecture: JangArchitecture(hasMoE: true)),
            status: verified)

        #expect(settings.effectiveMTPLaunchMode(for: verified) == .blocked)
        #expect(launch.launchMode == .blocked)
        #expect(launch.recommendation == nil)
        #expect(settings.validationIssues(
            configData: config,
            jangConfig: JangConfig(
                quantization: JangQuantization(
                    method: "jang",
                    profile: "JANG_2K",
                    targetBits: 2,
                    actualBits: 2,
                    bitWidthsUsed: [2, 3, 6, 8]),
                sourceModel: JangSourceModel(architecture: "qwen3_5_moe"),
                architecture: JangArchitecture(hasMoE: true)),
            mtpStatus: verified).contains {
                $0.severity == .error && $0.field == "mtp.mode"
            })
        if settings.resolvedMTPDraftStrategy(
            configData: config,
            jangConfig: nil,
            status: verified) != nil {
            Issue.record("Unsupported verified JANG_2K profile should not resolve a native-MTP draft strategy")
        }
    }

    @Test("invalid sampling and sleep values report issues instead of clamping")
    func invalidSamplingAndSleepValuesReportIssuesInsteadOfClamping() {
        var settings = VMLXServerRuntimeSettings()
        settings.power.lightSleepAfterSeconds = 30
        settings.power.deepSleepAfterSeconds = 10
        settings.generation.temperature = -1
        settings.generation.topP = 1.5
        settings.generation.repetitionPenalty = 0
        settings.mtp.draftTokenLimit = 0

        let fields = Set(settings.validationIssues().map(\.field))
        #expect(fields.contains("power.deepSleepAfterSeconds"))
        #expect(fields.contains("generation.temperature"))
        #expect(fields.contains("generation.topP"))
        #expect(fields.contains("generation.repetitionPenalty"))
        #expect(fields.contains("mtp.draftTokenLimit"))
    }

    @Test("nonpositive sleep timers report issues instead of clamping")
    func nonpositiveSleepTimersReportIssuesInsteadOfClamping() {
        var settings = VMLXServerRuntimeSettings()
        settings.power.lightSleepAfterSeconds = -1
        settings.power.deepSleepAfterSeconds = 0

        let fields = Set(settings.validationIssues().map(\.field))

        #expect(fields.contains("power.lightSleepAfterSeconds"))
        #expect(fields.contains("power.deepSleepAfterSeconds"))
    }

    @Test("request validation rejects multimodal force-off lanes")
    func requestValidationRejectsMultimodalForceOffLanes() throws {
        var settings = VMLXServerRuntimeSettings()
        settings.multimodal.vlmMode = .forceOff
        let snapshot = ModelRuntimeCapabilitySnapshot(
            configuration: ModelConfiguration(directory: URL(fileURLWithPath: "/tmp/private/omni")),
            capabilities: JangCapabilities(
                supportsText: true,
                supportsVision: true,
                supportsVideo: true,
                supportsAudio: true,
                family: "nemotron_h_omni",
                modality: "omni"))
        let request = ModelRuntimeCapabilityRequest(
            modalities: [.text, .vision, .video, .audio])

        let result = settings.validateRequest(request, capabilitySnapshot: snapshot)

        #expect(!result.allowed)
        #expect(result.issues.map(\.code) == [
            "server_modality_disabled",
            "server_modality_disabled",
            "server_modality_disabled",
        ])
        #expect(result.issues.map(\.modality) == [.vision, .video, .audio])
        #expect(result.issues.first?.redactedLogFields["field"] == "multimodal.vlmMode")
        let encoded = String(decoding: try JSONEncoder().encode(result), as: UTF8.self)
        #expect(!encoded.contains("/tmp/private"))
        #expect(!encoded.contains("omni"))
    }

    @Test("request validation rejects disabled audio and video lanes")
    func requestValidationRejectsDisabledAudioAndVideoLanes() {
        var settings = VMLXServerRuntimeSettings()
        settings.multimodal.enableVideo = false
        settings.multimodal.enableAudio = false
        let request = ModelRuntimeCapabilityRequest(
            modalities: [.text, .vision, .video, .audio])

        let result = settings.validateRequest(request)

        #expect(!result.allowed)
        #expect(result.issues.map(\.code) == [
            "server_modality_disabled",
            "server_modality_disabled",
        ])
        #expect(result.issues.map(\.modality) == [.video, .audio])
        #expect(result.issues.map { $0.redactedLogFields["field"] ?? "" } == [
            "multimodal.enableVideo",
            "multimodal.enableAudio",
        ])
    }

    @Test("request validation rejects native MTP when server mode is off")
    func requestValidationRejectsNativeMTPWhenServerModeIsOff() {
        var settings = VMLXServerRuntimeSettings()
        settings.mtp.mode = .off
        let request = ModelRuntimeCapabilityRequest(
            modalities: [.text, .nativeMTP])

        let result = settings.validateRequest(request)

        #expect(!result.allowed)
        #expect(result.issues.map(\.code) == ["server_modality_disabled"])
        #expect(result.issues.first?.modality == .nativeMTP)
        #expect(result.issues.first?.redactedLogFields["field"] == "mtp.mode")
    }

    @Test("invalid server numeric settings report issues instead of clamping")
    func invalidServerNumericSettingsReportIssuesInsteadOfClamping() {
        var settings = VMLXServerRuntimeSettings()
        settings.network.host = " "
        settings.network.port = 0
        settings.network.rateLimitRequestsPerMinute = -1
        settings.network.timeoutSeconds = 0
        settings.concurrency.maxConcurrentSequences = 0
        settings.concurrency.prefillBatchSize = -8
        settings.concurrency.prefillStepSize = 0
        settings.concurrency.completionBatchSize = -1
        settings.cache.prefix.memoryLimitMB = 0
        settings.cache.prefix.memoryPercent = 150
        settings.cache.prefix.ttlMinutes = -5

        let fields = Set(settings.validationIssues().map(\.field))
        #expect(fields.contains("network.host"))
        #expect(fields.contains("network.port"))
        #expect(fields.contains("network.rateLimitRequestsPerMinute"))
        #expect(fields.contains("network.timeoutSeconds"))
        #expect(fields.contains("concurrency.maxConcurrentSequences"))
        #expect(fields.contains("concurrency.prefillBatchSize"))
        #expect(fields.contains("concurrency.prefillStepSize"))
        #expect(fields.contains("concurrency.completionBatchSize"))
        #expect(fields.contains("cache.prefix.memoryLimitMB"))
        #expect(fields.contains("cache.prefix.memoryPercent"))
        #expect(fields.contains("cache.prefix.ttlMinutes"))
    }

    @Test("server cache settings build concrete coordinator config")
    func serverCacheSettingsBuildConcreteCoordinatorConfig() {
        var settings = VMLXServerRuntimeSettings()
        settings.cache.pagedKV.enabled = true
        settings.cache.pagedKV.blockSize = 128
        settings.cache.pagedKV.maxBlocks = 2048
        settings.cache.blockDisk.enabled = true
        settings.cache.blockDisk.maxSizeGB = 42
        settings.cache.blockDisk.directory = "/tmp/vmlx-block-l2"
        settings.cache.liveKVCodec = .turboQuant
        settings.cache.turboQuantKeyBits = 4
        settings.cache.turboQuantValueBits = 4
        settings.cache.defaultMaxKVSize = 8192
        settings.cache.longPromptMultiplier = 1.5
        settings.cache.enableSSMReDerive = true

        let config = settings.cacheCoordinatorConfig(
            modelKey: "test-model",
            ssmMaxEntries: 77)

        #expect(config.usePagedCache)
        #expect(config.enableDiskCache)
        #expect(config.pagedBlockSize == 128)
        #expect(config.maxCacheBlocks == 2048)
        #expect(config.diskCacheMaxGB == 42)
        #expect(config.diskCacheDir?.path == "/tmp/vmlx-block-l2")
        #expect(config.ssmMaxEntries == 77)
        #expect(config.enableSSMReDerive)
        #expect(config.modelKey == "test-model")
        if case .turboQuant(let keyBits, let valueBits) = config.defaultKVMode {
            #expect(keyBits == 4)
            #expect(valueBits == 4)
        } else {
            Issue.record("TurboQuant KV settings did not reach CacheCoordinatorConfig")
        }
        #expect(config.defaultMaxKVSize == 8192)
        #expect(config.longPromptMultiplier == 1.5)
    }

    @Test("media cache salt is required when cache reuse is enabled")
    func mediaCacheSaltIsRequiredWhenCacheReuseIsEnabled() {
        var settings = VMLXServerRuntimeSettings()
        settings.multimodal.requireMediaSaltForCache = false

        #expect(settings.validationIssues().contains {
            $0.severity == .error && $0.field == "multimodal.requireMediaSaltForCache"
        })

        settings.cache.prefix.enabled = false
        settings.cache.pagedKV.enabled = false
        settings.cache.blockDisk.enabled = false
        settings.cache.legacyDisk.enabled = false

        #expect(!settings.validationIssues().contains {
            $0.field == "multimodal.requireMediaSaltForCache"
        })
    }

    @Test("turboquant KV requires explicit bit widths")
    func turboQuantKVRequiresExplicitBitWidths() {
        var settings = VMLXServerRuntimeSettings()
        settings.cache.liveKVCodec = .turboQuant

        #expect(settings.validationIssues().contains {
            $0.severity == .error && $0.field == "cache.liveKVCodec"
        })
        if case .none = settings.cacheCoordinatorConfig().defaultKVMode {
            // Expected: do not silently choose hidden TQ bit widths.
        } else {
            Issue.record("TurboQuant KV mode should not be inferred without bit widths")
        }
    }

    @Test("parser overrides validate known aliases")
    func parserOverridesValidateKnownAliases() {
        var valid = VMLXServerRuntimeSettings()
        valid.tools.toolParserOverride = "qwen3_6"
        valid.tools.reasoningParserOverride = "qwen3_6"
        #expect(valid.validationIssues().isEmpty)

        valid.tools.toolParserOverride = "auto"
        valid.tools.reasoningParserOverride = "none"
        #expect(valid.validationIssues().isEmpty)

        var invalid = VMLXServerRuntimeSettings()
        invalid.tools.toolParserOverride = "not-a-tool-parser"
        invalid.tools.reasoningParserOverride = "not-a-reasoning-parser"
        let fields = Set(invalid.validationIssues().map(\.field))

        #expect(fields.contains("tools.toolParserOverride"))
        #expect(fields.contains("tools.reasoningParserOverride"))
    }
}

@Suite("Runtime MoE top-k override focused contracts")
struct RuntimeMoETopKOverrideFocusedTests {
    @Test("explicit MoE top-k override only lowers routed expert count")
    func overrideOnlyLowersRoutedExpertCount() {
        let lowered = RuntimeMoETopKOverride.resolve(
            currentTopK: 8,
            modelType: "hy_v3",
            field: "num_experts_per_tok",
            environment: ["VMLX_MOE_TOPK_OVERRIDE": "4"])
        #expect(lowered.effectiveTopK == 4)
        #expect(lowered.applied)

        let neverRaises = RuntimeMoETopKOverride.resolve(
            currentTopK: 1,
            modelType: "zaya",
            field: "moe_router_topk",
            environment: ["VMLX_MOE_TOPK_OVERRIDE": "4"])
        #expect(neverRaises.effectiveTopK == 1)
        #expect(!neverRaises.applied)
        #expect(neverRaises.reason == .requestedTopKAboveCurrent)
    }

    @Test("MoE top-k override scopes cache keys and ignores invalid values")
    func overrideScopesCacheKeys() {
        #expect(RuntimeMoETopKOverride.cacheScopedModelKey(
            "hy3",
            environment: ["VMLX_MOE_TOPK_OVERRIDE": "4"]) == "hy3|moeTopK=4")
        #expect(RuntimeMoETopKOverride.cacheScopedModelKey(
            "hy3",
            environment: ["VMLINUX_MOE_TOPK_OVERRIDE": "2"]) == "hy3|moeTopK=2")
        #expect(RuntimeMoETopKOverride.cacheScopedModelKey(
            "hy3",
            environment: ["VMLX_MOE_TOPK_OVERRIDE": "0"]) == "hy3")
    }
}
