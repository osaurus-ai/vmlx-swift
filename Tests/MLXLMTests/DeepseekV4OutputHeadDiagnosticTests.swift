// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing
@testable import MLXLLM

@Suite("DSV4 output-head diagnostic snapshot", .serialized)
struct DeepseekV4OutputHeadDiagnosticTests {
    @Test("default qmm with tracing off")
    func defaultQMMWithTracingOff() throws {
        let snapshot = Self.snapshot(
            cacheRequested: false,
            shadowRequested: false,
            sourceQuantized: true,
            sourceSupported: true,
            prepared: false,
            cacheIdentity: nil,
            logicalBytes: 0,
            configuredLMHeadMode: "qmm")

        #expect(snapshot.effectivePath == .qmm)
        #expect(snapshot.shadowRequested == false)
        #expect(snapshot.sourceQuantized)

        let encoded = try JSONEncoder().encode(snapshot)
        let object = try #require(
            JSONSerialization.jsonObject(with: encoded) as? [String: Any])
        #expect(object["effective_path"] as? String == "qmm")
        #expect(object["source_quantized"] as? Bool == true)
        #expect(object["configured_lm_head_mode"] as? String == "qmm")
    }

    @Test("qmm with a non-quantized source uses the exact path")
    func qmmWithNonQuantizedSourceUsesExact() {
        let snapshot = Self.snapshot(
            cacheRequested: false,
            shadowRequested: false,
            sourceQuantized: false,
            sourceSupported: false,
            prepared: false,
            cacheIdentity: nil,
            logicalBytes: 0,
            configuredLMHeadMode: "qmm")

        #expect(!snapshot.sourceQuantized)
        #expect(snapshot.effectivePath == .exact)
    }

    @Test("exact mode without the cache uses the exact path")
    func exactWithoutCache() {
        let snapshot = Self.snapshot(
            cacheRequested: false,
            shadowRequested: false,
            sourceQuantized: true,
            sourceSupported: true,
            prepared: false,
            cacheIdentity: nil,
            logicalBytes: 0,
            configuredLMHeadMode: "exact")

        #expect(snapshot.effectivePath == .exact)
        #expect(snapshot.cacheIdentity == nil)
    }

    @Test("exact mode with a prepared cache uses exactCached")
    func exactWithCache() {
        let snapshot = Self.snapshot(
            cacheRequested: true,
            shadowRequested: false,
            sourceQuantized: true,
            sourceSupported: true,
            prepared: true,
            cacheIdentity: 1,
            logicalBytes: 1_024,
            configuredLMHeadMode: "exact")

        #expect(snapshot.effectivePath == .exactCached)
        #expect(snapshot.cacheIdentity != nil)
        #expect(snapshot.logicalBytes == 1_024)
    }

    @Test("prepared shadow reports the exact baseline path")
    func preparedShadowUsesExactBaseline() {
        let snapshot = Self.snapshot(
            cacheRequested: true,
            shadowRequested: true,
            sourceQuantized: true,
            sourceSupported: true,
            prepared: true,
            cacheIdentity: 3,
            logicalBytes: 1_024,
            configuredLMHeadMode: "qmm")

        #expect(snapshot.effectivePath == .exact)
    }

    @Test("cache flag alone selects exactCached even with qmm configured")
    func cacheFlagAloneUsesExactCached() {
        let snapshot = Self.snapshot(
            cacheRequested: true,
            shadowRequested: false,
            sourceQuantized: true,
            sourceSupported: true,
            prepared: true,
            cacheIdentity: 2,
            logicalBytes: 1_024,
            configuredLMHeadMode: "qmm")

        #expect(snapshot.configuredLMHeadMode == "qmm")
        #expect(snapshot.effectivePath == .exactCached)
    }

    @Test("unsupported quantized qmm resolves to qmm")
    func unsupportedQuantizedQMMResolvesToQMM() {
        let snapshot = Self.snapshot(
            cacheRequested: true,
            shadowRequested: false,
            sourceQuantized: true,
            sourceSupported: false,
            prepared: false,
            cacheIdentity: nil,
            logicalBytes: 0,
            configuredLMHeadMode: "qmm")

        #expect(snapshot.effectivePath == .qmm)
    }

    @Test("unsupported quantized source cannot prepare the cache")
    func unsupportedQuantizedSourceCannotPrepareCache() throws {
        try MLXMetalTestLock.withLock {
            let flag = "VMLX_DSV4_CACHE_FP32_LM_HEAD"
            let prior = ProcessInfo.processInfo.environment[flag]
            setenv(flag, "1", 1)
            defer {
                if let prior {
                    setenv(flag, prior, 1)
                } else {
                    unsetenv(flag)
                }
            }

            let head = Self.makeHead(bits: 4)
            let state = DeepseekV4FP32LMHeadCacheState()
            try state.prepare(
                lmHead: head,
                expectedInputDimensions: 32,
                expectedOutputDimensions: 8)

            let snapshot = state.diagnosticSnapshot(
                modelType: "deepseek_v4",
                lmHead: head,
                expectedInputDimensions: 32,
                expectedOutputDimensions: 8)
            #expect(snapshot.sourceQuantized)
            #expect(!snapshot.sourceSupported)
            #expect(!snapshot.prepared)
            #expect(snapshot.cacheIdentity == nil)
            #expect(snapshot.logicalBytes == 0)
        }
    }

    @Test("snapshot follows the current head after module replacement")
    func snapshotFollowsCurrentHeadAfterModuleReplacement() async throws {
        try await MLXMetalTestLock.withLock {
            let flag = "VMLX_DSV4_CACHE_FP32_LM_HEAD"
            let prior = ProcessInfo.processInfo.environment[flag]
            setenv(flag, "1", 1)
            defer {
                if let prior {
                    setenv(flag, prior, 1)
                } else {
                    unsetenv(flag)
                }
            }

            var config = DeepseekV4ModelSmokeTests.tinyConfig()
            config.hiddenSize = 32
            config.vocabSize = 8
            let model = DeepseekV4Model(config)
            try model.update(
                modules: ModuleChildren.unflattened([("lm_head", Self.makeHead())]),
                verify: .none)
            try model.prepareForInferenceAfterLoad()
            #expect(model.modelContainerDiagnosticSnapshot().prepared)

            try model.updateModule(
                key: "lm_head",
                Linear(32, 8, bias: false)
            )
            let snapshot = model.modelContainerDiagnosticSnapshot()
            #expect(!snapshot.sourceQuantized)
            #expect(!snapshot.sourceSupported)
            #expect(!snapshot.prepared)
            #expect(snapshot.cacheIdentity == nil)
            #expect(snapshot.logicalBytes == 0)
            #expect(snapshot.effectivePath == .exact)
        }
    }

    @Test("ModelContainer reads both DSV4 model conformers")
    func bothDSV4ConformersExposeSnapshot() async throws {
        try await MLXMetalTestLock.withLock {
            let flag = "VMLX_DSV4_CACHE_FP32_LM_HEAD"
            let prior = ProcessInfo.processInfo.environment[flag]
            setenv(flag, "1", 1)
            defer {
                if let prior {
                    setenv(flag, prior, 1)
                } else {
                    unsetenv(flag)
                }
            }

            var config = DeepseekV4ModelSmokeTests.tinyConfig()
            config.hiddenSize = 32
            config.vocabSize = 8

            let affine = DeepseekV4Model(config)
            try affine.update(
                modules: ModuleChildren.unflattened([("lm_head", Self.makeHead())]),
                verify: .none)
            try affine.prepareForInferenceAfterLoad()
            let affineContainer = ModelContainer(
                context: ModelContext(
                    configuration: ModelConfiguration(id: "dsv4-affine-diagnostic"),
                    model: affine,
                    processor: TestInputProcessor(),
                    tokenizer: TestTokenizer(vocabularySize: 8)))
            let affineSnapshot = try #require(await affineContainer.diagnosticSnapshot())

            let jangTQ = DeepseekV4JANGTQModel(config)
            try jangTQ.update(
                modules: ModuleChildren.unflattened([("lm_head", Self.makeHead())]),
                verify: .none)
            try jangTQ.prepareForInferenceAfterLoad()
            let jangTQContainer = ModelContainer(
                context: ModelContext(
                    configuration: ModelConfiguration(id: "dsv4-jangtq-diagnostic"),
                    model: jangTQ,
                    processor: TestInputProcessor(),
                    tokenizer: TestTokenizer(vocabularySize: 8)))
            let jangTQSnapshot = try #require(await jangTQContainer.diagnosticSnapshot())

            for snapshot in [affineSnapshot, jangTQSnapshot] {
                #expect(snapshot.cacheRequested)
                #expect(snapshot.shadowRequested == false)
                #expect(snapshot.sourceQuantized)
                #expect(snapshot.sourceSupported)
                #expect(snapshot.prepared)
                #expect(snapshot.cacheIdentity != nil)
                #expect(snapshot.logicalBytes == 1_024)
                #expect(snapshot.effectivePath == .exactCached)
            }
            #expect(affineSnapshot.modelType == "deepseek_v4")
            #expect(jangTQSnapshot.modelType == "deepseek_v4_jangtq")
            #expect(await affineContainer.diagnosticSnapshot() == affineSnapshot)
        }
    }

    private static func snapshot(
        cacheRequested: Bool,
        shadowRequested: Bool,
        sourceQuantized: Bool,
        sourceSupported: Bool,
        prepared: Bool,
        cacheIdentity: UInt64?,
        logicalBytes: Int,
        configuredLMHeadMode: String
    ) -> ModelContainerDiagnosticSnapshot {
        ModelContainerDiagnosticSnapshot(
            modelType: "deepseek_v4",
            cacheRequested: cacheRequested,
            shadowRequested: shadowRequested,
            sourceQuantized: sourceQuantized,
            sourceSupported: sourceSupported,
            prepared: prepared,
            cacheIdentity: cacheIdentity,
            logicalBytes: logicalBytes,
            configuredLMHeadMode: configuredLMHeadMode)
    }

    private static func makeHead(bits: Int = 8) -> QuantizedLinear {
        let values = (0 ..< 32 * 8).map { Float(($0 % 23) - 11) / 16 }
        return QuantizedLinear(
            weight: MLXArray(values, [8, 32]).asType(.float16),
            bias: MLXArray((0 ..< 8).map { Float($0 - 3) / 32 }).asType(.float16),
            groupSize: 32,
            bits: bits,
            mode: .affine)
    }
}
