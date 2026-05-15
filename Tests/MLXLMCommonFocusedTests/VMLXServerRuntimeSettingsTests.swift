// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

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

    @Test("MTP auto does not launch preserved-only artifacts")
    func mtpAutoDoesNotLaunchPreservedOnlyArtifacts() {
        let settings = VMLXServerRuntimeSettings()
        let status = MTPBundleStatus(
            bundleHasMTP: true,
            configuredLayers: 4,
            tensorCount: 31,
            mode: .preservedEnabled)

        #expect(settings.effectiveMTPLaunchMode(for: status) == .off)
        #expect(settings.validationIssues(mtpStatus: status).isEmpty)
    }

    @Test("MTP force-on requires verified accept reject runtime")
    func mtpForceOnRequiresVerifiedAcceptRejectRuntime() {
        var settings = VMLXServerRuntimeSettings()
        settings.mtp.mode = .forceOn
        let preservedOnly = MTPBundleStatus(
            bundleHasMTP: true,
            configuredLayers: 4,
            tensorCount: 31,
            mode: .preservedEnabled)
        let verified = MTPBundleStatus(
            bundleHasMTP: true,
            configuredLayers: 4,
            tensorCount: 31,
            mode: .speculativeVerified)

        #expect(settings.effectiveMTPLaunchMode(for: preservedOnly) == .blocked)
        #expect(settings.validationIssues(mtpStatus: preservedOnly).contains {
            $0.severity == .error && $0.field == "mtp.mode"
        })
        #expect(settings.effectiveMTPLaunchMode(for: verified) == .speculative)
        #expect(settings.validationIssues(mtpStatus: verified).isEmpty)
    }

    @Test("invalid sampling and sleep values report issues instead of clamping")
    func invalidSamplingAndSleepValuesReportIssuesInsteadOfClamping() {
        var settings = VMLXServerRuntimeSettings()
        settings.power.lightSleepAfterSeconds = 30
        settings.power.deepSleepAfterSeconds = 10
        settings.generation.temperature = -1
        settings.generation.topP = 1.5
        settings.generation.repetitionPenalty = 0

        let fields = Set(settings.validationIssues().map(\.field))
        #expect(fields.contains("power.deepSleepAfterSeconds"))
        #expect(fields.contains("generation.temperature"))
        #expect(fields.contains("generation.topP"))
        #expect(fields.contains("generation.repetitionPenalty"))
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
}
