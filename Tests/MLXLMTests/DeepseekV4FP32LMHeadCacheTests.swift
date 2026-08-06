// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXNN
import Testing
@testable import MLXLLM

@Suite("DSV4 FP32 LM-head cache", .serialized)
struct DeepseekV4FP32LMHeadCacheTests {
    private static let featureFlag = "VMLX_DSV4_CACHE_FP32_LM_HEAD"
    private static let shadowFlag = "VMLX_DSV4_CACHE_FP32_LM_HEAD_SHADOW"

    private final class ConcurrentPreparationBox: @unchecked Sendable {
        let state = DeepseekV4FP32LMHeadCacheState()
        let head: QuantizedLinear
        private let resultLock = NSLock()
        private(set) var successes = 0
        private(set) var failures = 0

        init(head: QuantizedLinear) {
            self.head = head
        }

        func recordSuccess() {
            resultLock.lock()
            successes += 1
            resultLock.unlock()
        }

        func recordFailure() {
            resultLock.lock()
            failures += 1
            resultLock.unlock()
        }
    }

    private final class CacheOwningModule: Module {
        @ModuleInfo(key: "lm_head") var lmHead: Linear
        let cache = DeepseekV4FP32LMHeadCacheState()

        init(lmHead: Linear) {
            self._lmHead.wrappedValue = lmHead
        }
    }

    private static func withFeatureFlag<T>(
        _ value: String?,
        _ body: () throws -> T
    ) rethrows -> T {
        try withEnvironmentVariable(featureFlag, value, body)
    }

    private static func withShadowFlag<T>(
        _ value: String?,
        _ body: () throws -> T
    ) rethrows -> T {
        try withEnvironmentVariable(shadowFlag, value, body)
    }

    private static func withEnvironmentVariable<T>(
        _ name: String,
        _ value: String?,
        _ body: () throws -> T
    ) rethrows -> T {
        let prior = ProcessInfo.processInfo.environment[name]
        if let value {
            setenv(name, value, 1)
        } else {
            unsetenv(name)
        }
        defer {
            if let prior {
                setenv(name, prior, 1)
            } else {
                unsetenv(name)
            }
        }
        return try body()
    }

    private static func makeHead(
        inputDimensions: Int = 32,
        outputDimensions: Int = 8,
        bits: Int = 8,
        outputBias: Bool = true
    ) -> QuantizedLinear {
        let values = (0 ..< inputDimensions * outputDimensions).map {
            Float(($0 % 23) - 11) / 16
        }
        let weight = MLXArray(values, [outputDimensions, inputDimensions])
            .asType(.float16)
        let bias =
            outputBias
            ? MLXArray((0 ..< outputDimensions).map { Float($0 - 3) / 32 })
                .asType(.float16)
            : nil
        return QuantizedLinear(
            weight: weight,
            bias: bias,
            groupSize: 32,
            bits: bits,
            mode: .affine)
    }

    private static func rawFP32Bytes(_ array: MLXArray) -> Data {
        let bytes = array.asData(access: .copy)
        #expect(bytes.dType == .float32)
        return bytes.data
    }

    @Test("correct flag builds one byte-identical cache and preserves output bias")
    func enabledCacheMatchesCurrentExpressionAndLogits() throws {
        try MLXMetalTestLock.withLock {
            try Self.withFeatureFlag("1") {
                let head = Self.makeHead()
                let state = DeepseekV4FP32LMHeadCacheState()
                try state.prepare(
                    lmHead: head,
                    expectedInputDimensions: 32,
                    expectedOutputDimensions: 8)

                #expect(state.prepared)
                #expect(state.constructionCount == 1)
                let cachedWeight = try #require(state.cachedWeight)

                let currentWeight = MLX.dequantized(
                    head.weight,
                    scales: head.scales,
                    biases: head.biases,
                    groupSize: head.groupSize,
                    bits: head.bits
                ).asType(.float32)
                MLX.eval(currentWeight, cachedWeight)
                #expect(Self.rawFP32Bytes(currentWeight) == Self.rawFP32Bytes(cachedWeight))

                let hiddenValues = (0 ..< 32).map { index -> Float in
                    switch index {
                    case 0: return 0
                    case 1: return -0.0
                    case 2: return Float.leastNormalMagnitude
                    default: return Float((index % 9) - 4) / 8
                    }
                }
                let hidden = MLXArray(hiddenValues, [1, 1, 32]).asType(.bfloat16)
                let current = DeepseekV4Math.lmHeadFp32(hidden, lmHead: head)
                let cached = DeepseekV4Math.lmHeadFp32WithCachedWeight(
                    hidden, lmHead: head, weight: cachedWeight)
                MLX.eval(current, cached)
                #expect(Self.rawFP32Bytes(current) == Self.rawFP32Bytes(cached))

                #expect(throws: (any Error).self) {
                    try state.prepare(
                        lmHead: head,
                        expectedInputDimensions: 32,
                        expectedOutputDimensions: 8)
                }
                #expect(state.constructionCount == 1)
            }
        }
    }

    @Test("cache holder does not change the reflected parameter inventory")
    func cacheHolderIsNotAParameter() throws {
        try MLXMetalTestLock.withLock {
            try Self.withFeatureFlag("1") {
                let owner = CacheOwningModule(lmHead: Self.makeHead())
                let before = owner.parameters().flattened().map(\.0).sorted()
                try owner.cache.prepare(
                    lmHead: owner.lmHead,
                    expectedInputDimensions: 32,
                    expectedOutputDimensions: 8)
                let after = owner.parameters().flattened().map(\.0).sorted()

                #expect(owner.cache.prepared)
                #expect(before == after)
                #expect(!after.contains { $0.localizedCaseInsensitiveContains("cache") })
            }
        }
    }

    @Test("off and unsupported heads keep the current path")
    func fallbackIsFailClosed() throws {
        try MLXMetalTestLock.withLock {
            try Self.withFeatureFlag("0") {
                let state = DeepseekV4FP32LMHeadCacheState()
                try state.prepare(
                    lmHead: Self.makeHead(),
                    expectedInputDimensions: 32,
                    expectedOutputDimensions: 8)
                #expect(!state.prepared)
                #expect(state.cachedWeight == nil)
                #expect(state.constructionCount == 0)
                #expect(state.metadata?.supported == true)
            }

            try Self.withFeatureFlag("1") {
                let state = DeepseekV4FP32LMHeadCacheState()
                try state.prepare(
                    lmHead: Self.makeHead(bits: 4),
                    expectedInputDimensions: 32,
                    expectedOutputDimensions: 8)
                #expect(!state.prepared)
                #expect(state.cachedWeight == nil)
                #expect(state.constructionCount == 0)
                #expect(state.metadata?.supported == false)
            }
        }
    }

    @Test("affine cache rejects a missing quantization-bias array")
    func missingQuantizationBiasesAreUnsupported() throws {
        try MLXMetalTestLock.withLock {
            try Self.withFeatureFlag("1") {
                let source = Self.makeHead()
                let withoutBiases = QuantizedLinear(
                    weight: source.weight,
                    bias: source.bias,
                    scales: source.scales,
                    biases: nil,
                    groupSize: source.groupSize,
                    bits: source.bits,
                    mode: source.mode)
                let state = DeepseekV4FP32LMHeadCacheState()
                try state.prepare(
                    lmHead: withoutBiases,
                    expectedInputDimensions: 32,
                    expectedOutputDimensions: 8)

                #expect(state.metadata?.supported == false)
                #expect(!state.prepared)
                #expect(state.constructionCount == 0)
            }
        }
    }

    @Test("wrapper rejects a loaded head that disagrees with its configuration")
    func configuredDimensionsMustMatchLoadedHead() throws {
        try MLXMetalTestLock.withLock {
            try Self.withFeatureFlag("1") {
                let config = DeepseekV4ModelSmokeTests.tinyConfig()
                let model = DeepseekV4Model(config)
                try model.update(
                    modules: ModuleChildren.unflattened([
                        ("lm_head", Self.makeHead(inputDimensions: 32, outputDimensions: 8))
                    ]),
                    verify: .none)
                try model.prepareForInferenceAfterLoad()

                #expect(model.dsv4FP32LMHeadCacheMetadata?.supported == false)
                #expect(!model.dsv4FP32LMHeadCachePrepared)
                #expect(model.dsv4FP32LMHeadCacheConstructionCount == 0)
            }
        }
    }

    @Test("concurrent preparation constructs the derived weight once")
    func concurrentPreparationConstructsOnce() {
        MLXMetalTestLock.withLock {
            Self.withFeatureFlag("1") {
                let box = ConcurrentPreparationBox(head: Self.makeHead())
                DispatchQueue.concurrentPerform(iterations: 8) { _ in
                    do {
                        try box.state.prepare(
                            lmHead: box.head,
                            expectedInputDimensions: 32,
                            expectedOutputDimensions: 8)
                        box.recordSuccess()
                    } catch {
                        box.recordFailure()
                    }
                }

                #expect(box.successes == 1)
                #expect(box.failures == 7)
                #expect(box.state.prepared)
                #expect(box.state.constructionCount == 1)
            }
        }
    }

    @Test("both DSV4 wrappers keep the cache outside parameter reflection")
    func wrappersPreserveParameterInventory() throws {
        try MLXMetalTestLock.withLock {
            try Self.withFeatureFlag("1") {
                var config = DeepseekV4ModelSmokeTests.tinyConfig()
                config.hiddenSize = 32
                config.vocabSize = 8

                let affine = DeepseekV4Model(config)
                try affine.update(
                    modules: ModuleChildren.unflattened([("lm_head", Self.makeHead())]),
                    verify: .none)
                let affineBefore = affine.parameters().flattened().map(\.0).sorted()
                try affine.prepareForInferenceAfterLoad()
                let affineAfter = affine.parameters().flattened().map(\.0).sorted()
                #expect(affine.dsv4FP32LMHeadCachePrepared)
                #expect(affineBefore == affineAfter)

                let jangTQ = DeepseekV4JANGTQModel(config)
                try jangTQ.update(
                    modules: ModuleChildren.unflattened([("lm_head", Self.makeHead())]),
                    verify: .none)
                let jangTQBefore = jangTQ.parameters().flattened().map(\.0).sorted()
                try jangTQ.prepareForInferenceAfterLoad()
                let jangTQAfter = jangTQ.parameters().flattened().map(\.0).sorted()
                #expect(jangTQ.dsv4FP32LMHeadCachePrepared)
                #expect(jangTQBefore == jangTQAfter)
            }
        }
    }

    @Test("parameter and head-module updates invalidate the cache")
    func modelMutationInvalidatesCache() throws {
        try MLXMetalTestLock.withLock {
            try Self.withFeatureFlag("1") {
                var config = DeepseekV4ModelSmokeTests.tinyConfig()
                config.hiddenSize = 32
                config.vocabSize = 8

                let parameterUpdateModel = DeepseekV4Model(config)
                let parameterUpdateHead = Self.makeHead()
                try parameterUpdateModel.update(
                    modules: ModuleChildren.unflattened([
                        ("lm_head", parameterUpdateHead)
                    ]),
                    verify: .none)
                try parameterUpdateModel.prepareForInferenceAfterLoad()
                #expect(parameterUpdateModel.dsv4FP32LMHeadCachePrepared)

                try parameterUpdateModel.update(
                    parameters: ModuleParameters.unflattened([
                        ("lm_head.scales", parameterUpdateHead.scales + 1)
                    ]),
                    verify: .none)
                #expect(!parameterUpdateModel.dsv4FP32LMHeadCachePrepared)

                let moduleUpdateModel = DeepseekV4JANGTQModel(config)
                try moduleUpdateModel.update(
                    modules: ModuleChildren.unflattened([
                        ("lm_head", Self.makeHead())
                    ]),
                    verify: .none)
                try moduleUpdateModel.prepareForInferenceAfterLoad()
                #expect(moduleUpdateModel.dsv4FP32LMHeadCachePrepared)

                try moduleUpdateModel.update(
                    modules: ModuleChildren.unflattened([
                        ("lm_head", Self.makeHead())
                    ]),
                    verify: .none)
                #expect(!moduleUpdateModel.dsv4FP32LMHeadCachePrepared)
            }
        }
    }

    @Test("shadow mode compares every raw word and returns baseline logits")
    func shadowModeReturnsBaselineAndCountsWords() throws {
        try MLXMetalTestLock.withLock {
            try Self.withFeatureFlag("1") {
                try Self.withShadowFlag("1") {
                    let head = Self.makeHead()
                    let state = DeepseekV4FP32LMHeadCacheState()
                    try state.prepare(
                        lmHead: head,
                        expectedInputDimensions: 32,
                        expectedOutputDimensions: 8)
                    let hidden = MLXArray((0 ..< 32).map(Float.init), [1, 1, 32])
                        .asType(.bfloat16)
                    let expected = DeepseekV4Math.lmHeadFp32(hidden, lmHead: head)
                    let first = state.logits(hidden, lmHead: head)
                    let second = state.logits(hidden, lmHead: head)
                    MLX.eval(expected, first, second)

                    #expect(Self.rawFP32Bytes(first) == Self.rawFP32Bytes(expected))
                    #expect(Self.rawFP32Bytes(second) == Self.rawFP32Bytes(expected))
                    #expect(state.shadowComparisonCount == 2)
                    #expect(state.shadowComparedWordCount == 16)
                    #expect(state.shadowMismatchCount == 0)
                }
            }
        }
    }

    @Test("raw shadow comparison distinguishes negative zero")
    func rawShadowComparisonUsesFP32Words() {
        MLXMetalTestLock.withLock {
            let baseline = MLXArray([Float(0), -Float(0)]).asType(.float32)
            let cached = MLXArray([Float(0), Float(0)]).asType(.float32)
            let comparison = deepseekV4CompareRawFP32(
                baseline: baseline, cached: cached)

            #expect(!comparison.isEqual)
            #expect(comparison.wordCount == 2)
            #expect(comparison.firstMismatchWordIndex == 1)
            #expect(comparison.baselineWord == 0x8000_0000)
            #expect(comparison.cachedWord == 0)
            #expect(comparison.structuralMismatch == nil)
        }
    }

    @Test("cache state releases with its model-local owner")
    func cacheStateHasNoExternalOwner() throws {
        try MLXMetalTestLock.withLock {
            try Self.withFeatureFlag("1") {
                weak var releasedState: DeepseekV4FP32LMHeadCacheState?
                do {
                    let state = DeepseekV4FP32LMHeadCacheState()
                    releasedState = state
                    try state.prepare(
                        lmHead: Self.makeHead(),
                        expectedInputDimensions: 32,
                        expectedOutputDimensions: 8)
                    #expect(state.prepared)
                }
                #expect(releasedState == nil)
            }
        }
    }
}
