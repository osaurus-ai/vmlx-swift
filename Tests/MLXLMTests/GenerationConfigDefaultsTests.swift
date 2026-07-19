// Copyright © 2026 osaurus.

import Foundation
import MLX
@testable import MLXLMCommon
import MLXNN
import XCTest

final class GenerationConfigDefaultsTests: XCTestCase {

    func testDecodesSamplingDefaults() throws {
        let json = """
        {
          "eos_token_id": [1, 128803],
          "max_new_tokens": 321,
          "temperature": 0.7,
          "top_p": 0.95,
          "top_k": 40,
          "min_p": 0.05,
          "repetition_penalty": 1.05,
          "do_sample": true,
          "suppress_tokens": [258883, 258882]
        }
        """

        let config = try JSONDecoder.json5().decode(
            GenerationConfigFile.self, from: Data(json.utf8))

        XCTAssertEqual(config.eosTokenIds?.values, [1, 128803])
        XCTAssertEqual(config.maxNewTokens, 321)
        XCTAssertEqual(config.temperature, 0.7)
        XCTAssertEqual(config.topP, 0.95)
        XCTAssertEqual(config.topK, 40)
        XCTAssertEqual(config.minP, 0.05)
        XCTAssertEqual(config.repetitionPenalty, 1.05)
        XCTAssertEqual(config.doSample, true)
        XCTAssertEqual(config.suppressTokens, [258883, 258882])
    }

    func testResolvedEOSTokenIdsFallsBackToNestedTextConfig() throws {
        let json = """
        {
          "model_type": "qwen3_5_moe",
          "text_config": {
            "model_type": "qwen3_5_moe_text",
            "eos_token_id": 248046
          }
        }
        """
        let data = Data(json.utf8)
        let baseConfig = try JSONDecoder.json5().decode(BaseConfiguration.self, from: data)

        let eos = ModelTokenConfigurationResolver.resolvedEOSTokenIds(
            baseConfig: baseConfig,
            configurationData: data,
            generationConfig: nil)

        XCTAssertEqual(eos, [248046])
    }

    func testResolvedEOSTokenIdsGenerationConfigOverridesNestedTextConfig() throws {
        let json = """
        {
          "model_type": "qwen3_5_moe",
          "text_config": {
            "model_type": "qwen3_5_moe_text",
            "eos_token_id": 248046
          }
        }
        """
        let data = Data(json.utf8)
        let baseConfig = try JSONDecoder.json5().decode(BaseConfiguration.self, from: data)
        let generationConfig = GenerationConfigFile(eosTokenIds: IntOrIntArray([2, 11]))

        let eos = ModelTokenConfigurationResolver.resolvedEOSTokenIds(
            baseConfig: baseConfig,
            configurationData: data,
            generationConfig: generationConfig)

        XCTAssertEqual(eos, [2, 11])
    }

    func testGenerateParametersApplyConfigDefaultsAndPreserveRuntimeControls() {
        let fallback = GenerateParameters(
            maxTokens: 64,
            maxKVSize: 2048,
            kvBits: 4,
            kvGroupSize: 32,
            quantizedKVStart: 128,
            kvMode: .turboQuant(keyBits: 3, valueBits: 3),
            enableCompiledDecode: true,
            compiledMaxCacheLength: 4096,
            accelerationMode: .metal,
            enableCompiledBatchDecode: true,
            compiledBatchBuckets: [1, 2],
            temperature: 0.2,
            topP: 1.0,
            topK: 0,
            minP: 0.0,
            repetitionPenalty: nil,
            repetitionContextSize: 32,
            presencePenalty: 0.1,
            presenceContextSize: 24,
            frequencyPenalty: 0.2,
            frequencyContextSize: 25,
            prefillStepSize: 256,
            extraStopStrings: ["END"])

        let config = GenerationConfigFile(
            maxNewTokens: 123,
            temperature: 0.8,
            topP: 0.9,
            topK: 50,
            minP: 0.02,
            repetitionPenalty: 1.1,
            doSample: true,
            suppressTokens: [258883, 258882])

        let params = GenerateParameters(generationConfig: config, fallback: fallback)

        XCTAssertEqual(params.maxTokens, 123)
        XCTAssertEqual(params.temperature, 0.8)
        XCTAssertEqual(params.topP, 0.9)
        XCTAssertEqual(params.topK, 50)
        XCTAssertEqual(params.minP, 0.02)
        XCTAssertEqual(params.repetitionPenalty, 1.1)
        XCTAssertEqual(params.suppressTokens, [258883, 258882])

        XCTAssertEqual(params.maxKVSize, 2048)
        XCTAssertEqual(params.kvBits, 4)
        XCTAssertEqual(params.kvGroupSize, 32)
        XCTAssertEqual(params.quantizedKVStart, 128)
        XCTAssertEqual(params.kvMode, .turboQuant(keyBits: 3, valueBits: 3))
        XCTAssertTrue(params.enableCompiledDecode)
        XCTAssertEqual(params.compiledMaxCacheLength, 4096)
        XCTAssertTrue(params.enableCompiledBatchDecode)
        XCTAssertEqual(params.compiledBatchBuckets, [1, 2])
        XCTAssertEqual(params.repetitionContextSize, 32)
        XCTAssertEqual(params.presencePenalty, 0.1)
        XCTAssertEqual(params.frequencyPenalty, 0.2)
        XCTAssertEqual(params.prefillStepSize, 256)
        XCTAssertEqual(params.extraStopStrings, ["END"])
    }

    func testSuppressTokensProcessorMasksConfiguredLogits() {
        let params = GenerateParameters(suppressTokens: [1, 3, 99])
        let processor = params.processor()

        var logits = MLXArray.zeros([1, 5], type: Float32.self)
        logits = processor?.process(logits: logits) ?? logits
        let values = logits.asArray(Float.self)

        XCTAssertEqual(values[0], 0)
        XCTAssertTrue(values[1].isInfinite && values[1] < 0)
        XCTAssertEqual(values[2], 0)
        XCTAssertTrue(values[3].isInfinite && values[3] < 0)
        XCTAssertEqual(values[4], 0)
    }

    func testDoSampleFalseForcesGreedyEvenWhenTemperaturePresent() {
        let config = GenerationConfigFile(
            maxNewTokens: 50,
            temperature: 0.9,
            topP: 0.95,
            doSample: false)

        let params = GenerateParameters(generationConfig: config)

        XCTAssertEqual(params.maxTokens, 50)
        XCTAssertEqual(params.temperature, 0)
        XCTAssertEqual(params.topP, 0.95)
        XCTAssertTrue(params.sampler() is ArgMaxSampler)
    }

    func testModelConfigurationCarriesResolvedGenerationDefaults() {
        let config = GenerationConfigFile(
            maxNewTokens: 77,
            temperature: 0.4,
            topP: 0.8)
        let modelConfig = ModelConfiguration(
            id: "org/model",
            generationDefaults: config)

        let resolved = modelConfig.resolved(
            modelDirectory: URL(filePath: "/tmp/model"),
            tokenizerDirectory: URL(filePath: "/tmp/tokenizer"))

        XCTAssertEqual(resolved.generationDefaults, config)
        let params = GenerateParameters(generationConfig: resolved.generationDefaults)
        XCTAssertEqual(params.maxTokens, 77)
        XCTAssertEqual(params.temperature, 0.4)
        XCTAssertEqual(params.topP, 0.8)
    }

    /// Pins `ModelContainer.defaultGenerateParameters(fallback:)` —
    /// the opt-in convenience that returns a `GenerateParameters`
    /// initialized from the bundle's stamped `generation_config.json`
    /// values, with caller-supplied fallback for fields the config did
    /// not specify. Closes the production gap where the factory-side
    /// `generationDefaults` storage was wired without any
    /// ModelContainer-side consumer.
    func testModelContainerDefaultGenerateParametersAppliesGenerationConfig() async {
        let config = GenerationConfigFile(
            maxNewTokens: 222,
            temperature: 0.6,
            topP: 0.85,
            topK: 32,
            minP: 0.04,
            repetitionPenalty: 1.07,
            doSample: true)
        let modelConfig = ModelConfiguration(
            id: "org/test-model",
            generationDefaults: config)

        // Build a minimal ModelContext. We only exercise `configuration`
        // here, so a placeholder model/processor/tokenizer is fine — the
        // accessor under test reads `context.configuration.generationDefaults`.
        let context = ModelContext(
            configuration: modelConfig,
            model: TestStubLanguageModel(),
            processor: TestStubUserInputProcessor(),
            tokenizer: TestStubTokenizer())
        let container = ModelContainer(context: context)

        // Default fallback (no fields supplied).
        let defaults = await container.defaultGenerateParameters()
        XCTAssertEqual(defaults.maxTokens, 222)
        XCTAssertEqual(defaults.temperature, 0.6)
        XCTAssertEqual(defaults.topP, 0.85)
        XCTAssertEqual(defaults.topK, 32)
        XCTAssertEqual(defaults.minP, 0.04)
        XCTAssertEqual(defaults.repetitionPenalty, 1.07)

        // Explicit fallback fields the config does not override survive.
        let withFallback = await container.defaultGenerateParameters(
            fallback: GenerateParameters(
                maxKVSize: 4096,
                kvBits: 4,
                kvGroupSize: 32,
                prefillStepSize: 512))
        XCTAssertEqual(withFallback.maxTokens, 222) // from config
        XCTAssertEqual(withFallback.maxKVSize, 4096) // from fallback
        XCTAssertEqual(withFallback.kvBits, 4)
        XCTAssertEqual(withFallback.prefillStepSize, 512)
    }

    /// Pins the contract that a container with NO generation_config.json
    /// returns a `GenerateParameters` equal to the supplied fallback —
    /// i.e. the convenience method has no observable side effect for
    /// bundles that don't ship sampling defaults.
    func testModelContainerDefaultGenerateParametersFallsBackWhenConfigAbsent() async {
        let modelConfig = ModelConfiguration(id: "org/no-config-model")
        let context = ModelContext(
            configuration: modelConfig,
            model: TestStubLanguageModel(),
            processor: TestStubUserInputProcessor(),
            tokenizer: TestStubTokenizer())
        let container = ModelContainer(context: context)

        let fallback = GenerateParameters(maxTokens: 99, temperature: 0.33)
        let result = await container.defaultGenerateParameters(fallback: fallback)
        XCTAssertEqual(result.maxTokens, 99)
        XCTAssertEqual(result.temperature, 0.33)
    }

    func testModelContainerCacheTopologySnapshotUsesLiveCacheTypes() async {
        let modelConfig = ModelConfiguration(id: "org/hybrid-model")
        let context = ModelContext(
            configuration: modelConfig,
            model: TestStubLanguageModel(cache: [
                KVCacheSimple(),
                ChunkedKVCache(chunkSize: 8),
                QuantizedKVCache(groupSize: 32, bits: 4),
                TurboQuantKVCache(keyBits: 4, valueBits: 3),
                CompilableKVCache(maxLength: 32),
                CompilableTurboQuantKVCache(keyBits: 4, valueBits: 3),
                MambaCache(),
                CompilableMambaCache(),
                ArraysCache(size: 2),
                ZayaCCACache(batchSize: 1, convChannels: 4, hiddenSize: 8),
                CompilableRotatingKVCache(maxSize: 16),
                TestHybridPoolCache(),
                CacheList(RotatingKVCache(maxSize: 16), MambaCache()),
            ], copyCacheOnNewCache: false),
            processor: TestStubUserInputProcessor(),
            tokenizer: TestStubTokenizer())
        let container = ModelContainer(context: context)

        let topology = await container.cacheTopologySnapshot()

        XCTAssertEqual(topology.layerCount, 13)
        // ZayaCCACache owns one ordinary attention-KV cache in addition to
        // its separately counted native CCA companion state.
        XCTAssertEqual(topology.kvLayerCount, 4)
        XCTAssertEqual(topology.chunkedKVLayerCount, 1)
        XCTAssertEqual(topology.quantizedKVLayerCount, 1)
        XCTAssertEqual(topology.turboQuantKVLayerCount, 2)
        XCTAssertEqual(topology.compilableKVLayerCount, 1)
        XCTAssertEqual(topology.compilableTurboQuantKVLayerCount, 1)
        XCTAssertEqual(topology.rotatingKVLayerCount, 2)
        XCTAssertEqual(topology.compilableRotatingKVLayerCount, 1)
        XCTAssertEqual(topology.rotatingWrapperLayerCount, 1)
        XCTAssertEqual(topology.hybridPoolLayerCount, 1)
        XCTAssertEqual(topology.mambaLayerCount, 3)
        XCTAssertEqual(topology.compilableMambaLayerCount, 1)
        XCTAssertEqual(topology.arraysLayerCount, 1)
        XCTAssertEqual(topology.zayaCCALayerCount, 1)
        XCTAssertEqual(topology.cacheListLayerCount, 1)
        XCTAssertTrue(topology.requiresSSMCompanionState)
        XCTAssertTrue(topology.requiresDiskBackedCoordinatorRestore)
        XCTAssertTrue(topology.topologyTags.contains("companion=ssm"))
        XCTAssertTrue(topology.topologyTags.contains("restore=disk-backed"))
        XCTAssertTrue(topology.topologyTags.contains("turboQuantKVLayers=2"))
        XCTAssertTrue(topology.topologyTags.contains("hybridPoolLayers=1"))
    }
}

// MARK: - Test stubs

private final class TestStubLanguageModel: Module, LanguageModel {
    let cache: [any KVCache]
    let copyCacheOnNewCache: Bool

    init(cache: [any KVCache] = [], copyCacheOnNewCache: Bool = true) {
        self.cache = cache
        self.copyCacheOnNewCache = copyCacheOnNewCache
    }

    var kvHeads: [Int] { [] }
    var vocabularySize: Int { 0 }
    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        .tokens(input.text)
    }
    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        MLXArray.zeros([1, 1, 1])
    }
    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        copyCacheOnNewCache ? cache.map { $0.copy() } : cache
    }
}

private final class TestHybridPoolCache: BaseKVCache, HybridPoolCache {
    let rotating = RotatingKVCache(maxSize: 16)
    let compressRatio = 4
    let slidingWindow = 16

    override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        rotating.update(keys: keys, values: values)
    }

    override func innerState() -> [MLXArray] {
        rotating.state
    }

    override func copy() -> KVCache {
        TestHybridPoolCache()
    }

    func hybridPool(branch: HybridPoolBranch) -> MLXArray? { nil }
    func setHybridPool(branch: HybridPoolBranch, value: MLXArray?) {}
    func hybridBuffers(branch: HybridPoolBranch) -> (kv: MLXArray?, gate: MLXArray?) {
        (nil, nil)
    }
    func setHybridBuffers(branch: HybridPoolBranch, kv: MLXArray?, gate: MLXArray?) {}
}

private struct TestStubUserInputProcessor: UserInputProcessor {
    func prepare(input: UserInput) async throws -> LMInput {
        LMInput(tokens: MLXArray([Int32(0)]))
    }
}

private struct TestStubTokenizer: Tokenizer {
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { nil }
    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }
}
