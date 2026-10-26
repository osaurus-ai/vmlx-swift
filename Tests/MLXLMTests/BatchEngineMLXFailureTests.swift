import Foundation
@testable import MLX
import MLXNN
@testable import MLXLMCommon
import Testing

@Suite(.serialized)
struct BatchEngineMLXFailureTests {
    private static let interactivity = "[METAL] Command buffer execution failed: Impacting Interactivity (0000000e:kIOGPUCommandBufferCallbackErrorImpactingInteractivity)."

    /// Finite, deliberately wrong KV is written before dispatching the error.
    /// This tests publication, not a NaN filter, GPU hang or real Metal fault.
    private final class CacheMutationModel: Module, LanguageModel, KVCacheDimensionProvider,
        @unchecked Sendable
    {
        let vocabularySize = 32
        var kvHeads: [Int] { [1] }
        var calls = 0
        let failAt: Int?
        init(failAt: Int? = nil) { self.failAt = failAt }
        func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
            .tokens(input.text)
        }
        func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
            calls += 1
            let failed = calls == failAt
            let shape = [inputs.dim(0), 1, inputs.dim(1), 1]
            let keys = failed
                ? MLXArray.full(shape, values: MLXArray(Float(-4096)))
                : inputs.asType(.float32).expandedDimensions(axis: 1).expandedDimensions(axis: 3)
            let values = keys + Float(100)
            let retained = cache?.first?.update(keys: keys, values: values).0 ?? keys
            if failed {
                reportMLXError(BatchEngineMLXFailureTests.interactivity)
                reportMLXError("controlled secondary cache-mutation diagnostic")
            }
            // A wrong restored prefix changes logits and generated tokens.
            // The healthy replay therefore checks more than stream completion.
            let target = abs(Int(retained.sum().item(Float.self))) % 30
            let row = -square(MLXArray(0 ..< vocabularySize).asType(.float32) - Float(target))
            return MLX.broadcast(row, to: [inputs.dim(0), inputs.dim(1), vocabularySize])
        }
    }

    private struct RunResult {
        var text = ""
        var tokens: [Int] = []
        var infos: [GenerateCompletionInfo] = []
    }

    private func run(
        _ engine: BatchEngine, solo: Bool, prompt: [Int32], parameters: GenerateParameters,
        tokenizer: TestTokenizer
    ) async -> RunResult {
        var result = RunResult()
        let input = LMInput(tokens: MLXArray(prompt))
        if solo {
            for await event in await engine.generate(input: input, parameters: parameters) {
                if case .chunk(let text) = event { result.text += text }
                if case .info(let info) = event { result.infos.append(info) }
            }
        } else {
            let (_, stream) = await engine.submit(input: input, parameters: parameters)
            for await event in stream {
                if case .token(let token) = event { result.tokens.append(token) }
                if case .info(let info) = event { result.infos.append(info) }
            }
            result.text = tokenizer.decode(tokenIds: result.tokens, skipSpecialTokens: true)
        }
        return result
    }

    private func engine(
        model: CacheMutationModel, tokenizer: TestTokenizer, solo: Bool,
        coordinator: CacheCoordinator? = nil
    ) -> BatchEngine {
        var config = ModelConfiguration(id: "controlled-metal-cache-fixture")
        config.eosTokenIds = [31]
        let processor = TestInputProcessor(tokenizer: tokenizer, configuration: config,
            messageGenerator: DefaultMessageGenerator())
        return BatchEngine(context: ModelContext(configuration: config, model: model,
            processor: processor, tokenizer: tokenizer), maxBatchSize: solo ? 1 : 2,
            cacheCoordinator: coordinator)
    }

    @Test(.timeLimit(.minutes(1)), arguments: [false, true], [1, 3])
    func controlledInteractivityFailureDoesNotPublishPoisonedKV(solo: Bool, failAt: Int) async throws {
        try await MLXMetalTestLock.withLock {
            let directory = FileManager.default.temporaryDirectory
                .appendingPathComponent("controlled-metal-cache-\(UUID())", isDirectory: true)
            defer { try? FileManager.default.removeItem(at: directory) }
            let coordinator = CacheCoordinator(config: .init(
                usePagedCache: false, enableDiskCache: true, diskCacheMaxGB: 0.01,
                diskCacheDir: directory, modelKey: UUID().uuidString,
                preserveStandardKVStorageDType: true))
            let tokenizer = TestTokenizer(vocabularySize: 32)
            let model = CacheMutationModel(failAt: failAt)
            let failedEngine = engine(model: model, tokenizer: tokenizer, solo: solo, coordinator: coordinator)
            let referenceEngine = engine(model: CacheMutationModel(), tokenizer: tokenizer, solo: solo)
            let prompt: [Int32] = [1, 2, 3, 4, 5, 6, 7, 8]
            let parameters = GenerateParameters(maxTokens: 4, temperature: 0)
            do {
                let failed = await run(failedEngine, solo: solo, prompt: prompt,
                    parameters: parameters, tokenizer: tokenizer)
                let info = try #require(failed.infos.count == 1 ? failed.infos.first : nil)
                let failure = try #require(info.generationFailure)
                #expect(info.stopReason == .cancelled)
                #expect(failure.stage == (failAt == 1 ? .preparation : .decoding))
                #expect(failure.cause.contains(Self.interactivity))
                #expect(!failure.cause.contains("secondary"))
                if failAt == 1 {
                    #expect(info.generationTokenCount == 0)
                } else {
                    #expect(info.generationTokenCount > 0 && info.generationTokenCount < 4)
                }
                #expect(coordinator.waitForPendingDiskWrites())
                Stream.gpu.synchronize()

                let disk = try #require(coordinator.diskCache)
                let entries = disk.quotaEntries(retiringInvalidRecords: false)
                // Solo may retain a clean prefill checkpoint. A failed answer
                // or partially mutated decode state is never a reusable leaf.
                for entry in entries {
                    #expect(entry.tokenCount <= prompt.count)
                    #expect(!entry.isPostAnswer && entry.companionKey == nil && entry.companionBytes == 0)
                    let arrays = try loadArrays(url: directory.appendingPathComponent(entry.hash + ".safetensors"))
                    let keys = try #require(arrays["kv_0_keys"])
                    let values = try #require(arrays["kv_0_values"])
                    let cleanCache = KVCacheSimple()
                    let cleanModel = CacheMutationModel()
                    let prefix = MLXArray(Array(prompt.prefix(entry.tokenCount))).expandedDimensions(axis: 0)
                    let cleanLogits = cleanModel(prefix, cache: [cleanCache])
                    eval(cleanLogits)
                    let reference = cleanCache.state
                    try #require(reference.count == 2)
                    #expect(keys.shape == reference[0].shape && keys.dtype == reference[0].dtype)
                    #expect(values.shape == reference[1].shape && values.dtype == reference[1].dtype)
                    #expect(keys.asData().data == reference[0].asData().data)
                    #expect(values.asData().data == reference[1].asData().data)
                    let restored = KVCacheSimple()
                    restored.state = [keys, values]
                    let suffix = MLXArray([Int32(9)]).expandedDimensions(axis: 0)
                    let replayLogits = cleanModel(suffix, cache: [restored])
                    let referenceLogits = cleanModel(suffix, cache: [cleanCache])
                    #expect(replayLogits.asData().data == referenceLogits.asData().data)
                }
                if !solo { #expect(entries.isEmpty, "Failed scheduled slots skip all cache publication") }

                // Same engine, same prompt: a valid prefill hit is allowed,
                // but the next request must match a clean independent model.
                let healthy = await run(failedEngine, solo: solo, prompt: prompt,
                    parameters: parameters, tokenizer: tokenizer)
                let reference = await run(referenceEngine, solo: solo, prompt: prompt,
                    parameters: parameters, tokenizer: tokenizer)
                let healthyInfo = try #require(healthy.infos.count == 1 ? healthy.infos.first : nil)
                #expect(healthyInfo.generationFailure == nil && healthyInfo.stopReason == .length)
                #expect(healthyInfo.generationTokenCount == 4)
                #expect(!healthy.text.isEmpty)
                #expect(healthy.text == reference.text && healthy.tokens == reference.tokens)
                #expect(reference.infos.first?.generationFailure == nil)
                await failedEngine.shutdown()
                await referenceEngine.shutdown()
                let active = await failedEngine.activeCount
                let pending = await failedEngine.pendingCount
                #expect(active == 0 && pending == 0)
            } catch {
                await failedEngine.shutdown()
                await referenceEngine.shutdown()
                throw error
            }
        }
    }

    private final class FailingModel: Module, LanguageModel, KVCacheDimensionProvider,
        @unchecked Sendable
    {
        let vocabularySize = 32
        var kvHeads: [Int] { [1] }
        var calls = 0
        let failAt: Int
        init(failAt: Int) { self.failAt = failAt }
        func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
            .tokens(input.text)
        }
        func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
            calls += 1
            let shape = [inputs.dim(0), inputs.dim(1), vocabularySize]
            if calls == failAt {
                let kernel = MLXFast.metalKernel(
                    name: "batchFailureProbe", inputNames: ["a"], outputNames: ["out"],
                    source: "this is not valid metal source;")
                return kernel([MLXArray.zeros([1])], grid: (1, 1, 1),
                    threadGroup: (1, 1, 1), outputShapes: [shape], outputDTypes: [.float32])[0]
            }
            return MLXArray.zeros(shape)
        }
    }

    @Test(arguments: [1, 2])
    func failedStepDoesNotEmitOrFinishSuccessfully(failAt: Int) async {
        let model = FailingModel(failAt: failAt)
        var config = ModelConfiguration(id: "batch-error-audit")
        config.eosTokenIds = [31]
        let tokenizer = TestTokenizer()
        let processor = TestInputProcessor(tokenizer: tokenizer, configuration: config,
            messageGenerator: DefaultMessageGenerator())
        let context = ModelContext(configuration: config, model: model,
            processor: processor, tokenizer: tokenizer)
        let engine = BatchEngine(context: context, maxBatchSize: 2)
        let (_, stream) = await withErrorHandler({ _ in }) {
            await engine.submit(input: LMInput(tokens: MLXArray([Int32(1), 2, 3])),
                parameters: GenerateParameters(maxTokens: 2, temperature: 0))
        }
        var tokens: [Int] = []
        var completion: GenerateCompletionInfo?
        for await event in stream {
            switch event {
            case .token(let token): tokens.append(token)
            case .info(let info): completion = info
            default: break
            }
        }
        #expect(tokens.count == failAt - 1)
        #expect(completion?.stopReason == .cancelled)
        #expect(completion?.generationFailure?.stage == (failAt == 1 ? .preparation : .decoding))

        // Reuse the same engine after failure. Request-local failure must not
        // poison the next chat/delegation or leave it queued forever.
        let (_, healthy) = await engine.submit(
            input: LMInput(tokens: MLXArray([Int32(4), 5, 6])),
            parameters: GenerateParameters(maxTokens: 2, temperature: 0))
        var healthyTokens = 0
        var healthyInfo: GenerateCompletionInfo?
        for await event in healthy {
            if case .token = event { healthyTokens += 1 }
            if case .info(let info) = event { healthyInfo = info }
        }
        #expect(healthyTokens == 2)
        #expect(healthyInfo?.generationFailure == nil)
        #expect(healthyInfo?.stopReason == .length)
    }
}
