import Foundation
import MLX
import MLXNN
@testable import MLXLMCommon
import Testing

@Suite(.serialized)
struct BatchEngineMLXFailureTests {
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
