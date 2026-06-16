// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLXLMCommon
import Testing

@Suite("VMLX memory safety settings")
struct VMLXMemorySafetySettingsTests {
    @Test("defaults resolve Safe Auto without sampler changes")
    func defaultsResolveSafeAutoWithoutSamplerChanges() {
        let settings = VMLXServerRuntimeSettings()
        let facts = LoadBundleFacts(
            totalSafetensorsBytes: 12 << 30,
            isRouted: false,
            physicalMemory: 64 << 30,
            modelType: "gemma4",
            weightFormat: "mxfp4")

        let plan = settings.resolvedMemorySafetyPlan(
            baseLoadConfiguration: .osaurusProduction,
            bundleFacts: facts,
            request: VMLXMemoryRequestEstimate(
                workingSetBytes: 20 << 30,
                promptTokens: 4096,
                maxNewTokens: 256))
        let generate = settings.resolvedGenerateParameters(
            generationConfig: GenerationConfigFile(
                temperature: 0.7,
                topP: 0.95,
                topK: 64,
                doSample: true),
            fallback: GenerateParameters(
                maxTokens: 128,
                temperature: 0.2,
                topP: 1.0,
                topK: 0,
                repetitionPenalty: nil))

        #expect(settings.memorySafety.mode == .safeAuto)
        #expect(settings.memorySafety.slider == 2)
        #expect(plan.allowed)
        #expect(plan.loadConfiguration.memoryLimit == .fraction(0.70))
        #expect(plan.loadConfiguration.maxResidentBytes == .absolute(128 << 20))
        #expect(plan.loadConfiguration.useMmapSafetensors)
        #expect(plan.loadConfiguration.jangPress == .disabled)
        #expect(plan.cache.prefix.enabled)
        #expect(plan.cache.pagedKV.enabled)
        #expect(plan.cache.blockDisk.enabled)
        #expect(!plan.cache.legacyDisk.enabled)
        #expect(plan.cache.defaultMaxKVSize == 8192)
        #expect(plan.cache.enableSSMReDerive)
        #expect(plan.concurrency.maxConcurrentSequences == 1)
        #expect(generate.temperature == 0.7)
        #expect(generate.topP == 0.95)
        #expect(generate.topK == 64)
        #expect(generate.repetitionPenalty == nil)
    }

    @Test("validation rejects invalid custom values")
    func validationRejectsInvalidCustomValues() {
        var settings = VMLXServerRuntimeSettings()
        settings.memorySafety.slider = 9
        settings.memorySafety.customPhysicalMemoryFraction = 1.2
        settings.memorySafety.customAllocatorCacheBytes = 0
        settings.memorySafety.customDefaultMaxKVSize = 0
        settings.memorySafety.customMaxConcurrentSequences = 0

        let fields = Set(settings.validationIssues().map(\.field))

        #expect(fields.contains("memorySafety.slider"))
        #expect(fields.contains("memorySafety.customPhysicalMemoryFraction"))
        #expect(fields.contains("memorySafety.customAllocatorCacheBytes"))
        #expect(fields.contains("memorySafety.customDefaultMaxKVSize"))
        #expect(fields.contains("memorySafety.customMaxConcurrentSequences"))
    }

    @Test("memory safety preserves explicit cache disable")
    func memorySafetyPreservesExplicitCacheDisable() {
        var settings = VMLXServerRuntimeSettings()
        settings.cache.prefix.enabled = false
        settings.cache.pagedKV.enabled = true
        settings.cache.blockDisk.enabled = true
        settings.cache.legacyDisk.enabled = true

        let plan = settings.resolvedMemorySafetyPlan(
            baseLoadConfiguration: .osaurusProduction,
            bundleFacts: LoadBundleFacts(
                totalSafetensorsBytes: 4 << 30,
                isRouted: false,
                physicalMemory: 64 << 30,
                modelType: "gemma4",
                weightFormat: "mxfp4"))
        let coordinator = VMLXServerRuntimeSettings(
            cache: plan.cache,
            memorySafety: settings.memorySafety
        )
        .cacheCoordinatorConfig(modelKey: "gemma4|cache=disabled")

        #expect(!plan.cache.prefix.enabled)
        #expect(!plan.cache.pagedKV.enabled)
        #expect(!plan.cache.blockDisk.enabled)
        #expect(!plan.cache.legacyDisk.enabled)
        #expect(!coordinator.usePagedCache)
        #expect(!coordinator.enableDiskCache)
    }

    @Test("strict mode returns typed refusal for unknown and over budget estimates")
    func strictModeReturnsTypedRefusalForUnknownAndOverBudgetEstimates() {
        var settings = VMLXServerRuntimeSettings()
        settings.memorySafety.mode = .strict
        settings.memorySafety.slider = 3
        let facts = LoadBundleFacts(
            totalSafetensorsBytes: 20 << 30,
            isRouted: true,
            physicalMemory: 24 << 30,
            modelType: "gemma4",
            weightFormat: "jang_4m",
            hasJangConfig: true)

        let unknown = settings.resolvedMemorySafetyPlan(
            baseLoadConfiguration: .osaurusProduction,
            bundleFacts: facts,
            request: VMLXMemoryRequestEstimate())
        let overBudget = settings.resolvedMemorySafetyPlan(
            baseLoadConfiguration: .osaurusProduction,
            bundleFacts: facts,
            request: VMLXMemoryRequestEstimate(workingSetBytes: 20 << 30))

        #expect(!unknown.allowed)
        #expect(unknown.blockingIssues.contains {
            $0.field == "memorySafety.requestEstimate"
                && $0.message.contains("requires a request working-set estimate")
        })
        #expect(!overBudget.allowed)
        #expect(overBudget.blockingIssues.contains {
            $0.field == "memorySafety.requestEstimate"
                && $0.message.contains("exceeds resolved memory budget")
        })
        #expect(overBudget.loadConfiguration.memoryLimit == .fraction(0.60))
        #expect(overBudget.cache.defaultMaxKVSize == 4096)
    }

    @Test("memory safety does not make MLXPress the universal slider")
    func memorySafetyDoesNotMakeMLXPressTheUniversalSlider() {
        var settings = VMLXServerRuntimeSettings()
        let denseGemma = LoadBundleFacts(
            totalSafetensorsBytes: 30 << 30,
            isRouted: false,
            physicalMemory: 24 << 30,
            modelType: "gemma4",
            weightFormat: "mxfp4")
        let routedJang = LoadBundleFacts(
            totalSafetensorsBytes: 30 << 30,
            isRouted: true,
            physicalMemory: 24 << 30,
            modelType: "mimo_v2",
            weightFormat: "jangtq",
            hasJangConfig: true,
            hasJangTQRuntime: true,
            numRoutedExperts: 64,
            topK: 8)

        let densePlan = settings.resolvedMemorySafetyPlan(
            baseLoadConfiguration: .osaurusProduction,
            bundleFacts: denseGemma)
        settings.memorySafety.allowExperimentalMLXPress = true
        let routedPlan = settings.resolvedMemorySafetyPlan(
            baseLoadConfiguration: .osaurusProduction,
            bundleFacts: routedJang)

        #expect(densePlan.loadConfiguration.jangPress == .disabled)
        #expect(densePlan.warnings.contains {
            $0.contains("MLXPress/JangPress was not selected")
        })
        #expect(routedPlan.loadConfiguration.jangPress == .auto(envFallback: true))
    }

    @Test("memory safety preserves hybrid SSM and engine selected cache topology")
    func memorySafetyPreservesHybridSSMAndEngineSelectedCacheTopology() {
        var settings = VMLXServerRuntimeSettings()
        settings.cache.liveKVCodec = .engineSelected
        settings.cache.enableSSMReDerive = true
        let facts = LoadBundleFacts(
            totalSafetensorsBytes: 18 << 30,
            isRouted: true,
            physicalMemory: 64 << 30,
            modelType: "qwen3_6",
            weightFormat: "mxfp4")

        let plan = settings.resolvedMemorySafetyPlan(
            baseLoadConfiguration: .default,
            bundleFacts: facts)
        let coordinator = VMLXServerRuntimeSettings(
            concurrency: plan.concurrency,
            cache: plan.cache,
            memorySafety: settings.memorySafety)
            .cacheCoordinatorConfig(
                modelKey: "qwen3.6|mtp=preserved|cache=engine_selected",
                ssmMaxEntries: 96)

        #expect(plan.cache.liveKVCodec == .engineSelected)
        #expect(plan.cache.enableSSMReDerive)
        #expect(coordinator.usePagedCache)
        #expect(coordinator.enableDiskCache)
        #expect(coordinator.enableSSMReDerive)
        #expect(coordinator.ssmMaxEntries == 96)
        if case .turboQuant(let keyBits, let valueBits) = coordinator.defaultKVMode {
            #expect(keyBits == 3)
            #expect(valueBits == 3)
        } else {
            Issue.record("Memory safety should preserve engine-selected TurboQuant KV policy.")
        }
    }
}
