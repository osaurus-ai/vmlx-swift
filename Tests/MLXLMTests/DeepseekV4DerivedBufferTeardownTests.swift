// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing
@testable import MLXLLM

@Suite("DSV4 derived-buffer teardown", .serialized)
struct DeepseekV4DerivedBufferTeardownTests {
    private static let featureFlag = "VMLX_DSV4_CACHE_FP32_LM_HEAD"

    private static func withFeatureFlag<T>(
        _ value: String?,
        _ body: () throws -> T
    ) rethrows -> T {
        let prior = ProcessInfo.processInfo.environment[featureFlag]
        if let value {
            setenv(featureFlag, value, 1)
        } else {
            unsetenv(featureFlag)
        }
        defer {
            if let prior {
                setenv(featureFlag, prior, 1)
            } else {
                unsetenv(featureFlag)
            }
        }
        return try body()
    }

    private static func makeHead() -> QuantizedLinear {
        let values = (0 ..< 32 * 8).map { Float(($0 % 23) - 11) / 16 }
        return QuantizedLinear(
            weight: MLXArray(values, [8, 32]).asType(.float16),
            bias: MLXArray((0 ..< 8).map { Float($0 - 3) / 32 }).asType(.float16),
            groupSize: 32,
            bits: 8,
            mode: .affine)
    }

    private static func makeAffineModel() throws -> DeepseekV4Model {
        var config = DeepseekV4ModelSmokeTests.tinyConfig()
        config.hiddenSize = 32
        config.vocabSize = 8
        let model = DeepseekV4Model(config)
        try model.update(
            modules: ModuleChildren.unflattened([("lm_head", makeHead())]),
            verify: .none)
        return model
    }

    private static func makeJANGTQModel() throws -> DeepseekV4JANGTQModel {
        var config = DeepseekV4ModelSmokeTests.tinyConfig()
        config.hiddenSize = 32
        config.vocabSize = 8
        let model = DeepseekV4JANGTQModel(config)
        try model.update(
            modules: ModuleChildren.unflattened([("lm_head", makeHead())]),
            verify: .none)
        return model
    }

    private static func releaseEvents(
        _ events: [DeepseekV4FP32LMHeadCacheEvent]
    ) -> [DeepseekV4FP32LMHeadCacheEvent] {
        events.filter { event in
            if case .release = event { return true }
            return false
        }
    }

    private static func rawFP32Bytes(_ array: MLXArray) -> Data {
        let bytes = array.asData(access: .copy)
        #expect(bytes.dType == .float32)
        return bytes.data
    }

    @Test("both DSV4 conformers release once and reprepare with a new identity")
    func bothConformersReleaseAndReprepare() throws {
        try MLXMetalTestLock.withLock {
            try Self.withFeatureFlag("1") {
                let affine = try Self.makeAffineModel()
                try affine.prepareForInferenceAfterLoad()
                let affineFirstIdentity = try #require(affine.dsv4FP32LMHeadCacheIdentity)
                let affineAlias = try #require(affine.dsv4FP32LMHeadCacheWeight)
                MLX.eval(affineAlias)
                let affineAliasBytes = Self.rawFP32Bytes(affineAlias)
                #expect(affine.dsv4FP32LMHeadCacheLogicalBytes == 1_024)

                affine.releaseDerivedBuffersForTeardown()
                #expect(!affine.dsv4FP32LMHeadCachePrepared)
                #expect(affine.dsv4FP32LMHeadCacheIdentity == nil)
                #expect(affine.dsv4FP32LMHeadCacheLogicalBytes == 0)
                MLX.eval(affineAlias)
                #expect(Self.rawFP32Bytes(affineAlias) == affineAliasBytes)
                let affineAfterFirstRelease = affine.dsv4FP32LMHeadCacheEvents
                #expect(Self.releaseEvents(affineAfterFirstRelease).count == 1)

                affine.releaseDerivedBuffersForTeardown()
                #expect(affine.dsv4FP32LMHeadCacheEvents == affineAfterFirstRelease)

                try affine.prepareForInferenceAfterLoad()
                let affineSecondIdentity = try #require(affine.dsv4FP32LMHeadCacheIdentity)
                #expect(affineSecondIdentity != affineFirstIdentity)
                #expect(affine.dsv4FP32LMHeadCacheEvents.contains { event in
                    if case let .identityChanged(previous, current) = event {
                        return previous == affineFirstIdentity && current == affineSecondIdentity
                    }
                    return false
                })
                #expect(affine.dsv4FP32LMHeadCacheEvents.contains { event in
                    if case .logicalBytes(1_024) = event { return true }
                    return false
                })

                let jangTQ = try Self.makeJANGTQModel()
                try jangTQ.prepareForInferenceAfterLoad()
                let jangFirstIdentity = try #require(jangTQ.dsv4FP32LMHeadCacheIdentity)
                #expect(jangTQ.dsv4FP32LMHeadCacheLogicalBytes == 1_024)

                jangTQ.releaseDerivedBuffersForTeardown()
                #expect(!jangTQ.dsv4FP32LMHeadCachePrepared)
                #expect(jangTQ.dsv4FP32LMHeadCacheIdentity == nil)
                #expect(jangTQ.dsv4FP32LMHeadCacheLogicalBytes == 0)
                let jangAfterFirstRelease = jangTQ.dsv4FP32LMHeadCacheEvents
                #expect(Self.releaseEvents(jangAfterFirstRelease).count == 1)

                jangTQ.releaseDerivedBuffersForTeardown()
                #expect(jangTQ.dsv4FP32LMHeadCacheEvents == jangAfterFirstRelease)

                try jangTQ.prepareForInferenceAfterLoad()
                let jangSecondIdentity = try #require(jangTQ.dsv4FP32LMHeadCacheIdentity)
                #expect(jangSecondIdentity != jangFirstIdentity)
                #expect(jangTQ.dsv4FP32LMHeadCacheEvents.contains { event in
                    if case let .identityChanged(previous, current) = event {
                        return previous == jangFirstIdentity && current == jangSecondIdentity
                    }
                    return false
                })
            }
        }
    }

    private final class SerializationProbeModel:
        Module, LanguageModel, PostLoadModelPreparation, @unchecked Sendable
    {
        private let lock = NSLock()
        private var activeReleases = 0
        private var maximumConcurrentReleasesStorage = 0
        private var releaseCountStorage = 0

        var maximumConcurrentReleases: Int {
            lock.lock()
            defer { lock.unlock() }
            return maximumConcurrentReleasesStorage
        }

        var releaseCount: Int {
            lock.lock()
            defer { lock.unlock() }
            return releaseCountStorage
        }

        func prepare(
            _ input: LMInput, cache: [KVCache], windowSize: Int?
        ) throws -> PrepareResult {
            .tokens(input.text)
        }

        func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
            MLXArray.zeros([1, 1, 1])
        }

        func newCache(parameters: GenerateParameters?) -> [KVCache] {
            []
        }

        func prepareForInferenceAfterLoad() throws {}

        func releaseDerivedBuffersForTeardown() {
            lock.lock()
            activeReleases += 1
            releaseCountStorage += 1
            maximumConcurrentReleasesStorage = max(
                maximumConcurrentReleasesStorage, activeReleases)
            lock.unlock()

            Thread.sleep(forTimeInterval: 0.01)

            lock.lock()
            activeReleases -= 1
            lock.unlock()
        }
    }

    @Test("ModelContainer serializes derived-buffer teardown calls")
    func modelContainerSerializesTeardown() async {
        let model = SerializationProbeModel()
        let context = ModelContext(
            configuration: ModelConfiguration(id: "dsv4-teardown-serialization"),
            model: model,
            processor: TestInputProcessor(),
            tokenizer: TestTokenizer(vocabularySize: 8))
        let container = ModelContainer(context: context)

        await withTaskGroup(of: Void.self) { group in
            group.addTask {
                await container.releaseDerivedBuffersForTeardown()
            }
            group.addTask {
                await container.releaseDerivedBuffersForTeardown()
            }
            await group.waitForAll()
        }

        #expect(model.releaseCount == 2)
        #expect(model.maximumConcurrentReleases == 1)
    }
}
