// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLMCommon

@Suite("Ineligible DFlash 2 requests retain ordinary decoding", .serialized)
struct DFlash2ARFallbackBatchTests {
    @Test(
        arguments: [1, 2],
        [
            "media", "presence", "frequency", "repetition", "suppression", "min-p", "one-token",
            "width-one",
        ])
    func matchesExplicitAR(batchSize: Int, scenario: String) async throws {
        try await MLXMetalTestLock.withLock {
            let model = FallbackFixtureModel()
            let processor = TestInputProcessor(
                tokenizer: TestTokenizer(vocabularySize: 16),
                configuration: ModelConfiguration(id: "dflash-ineligible-fixture"),
                messageGenerator: DefaultMessageGenerator())
            let context = ModelContext(
                configuration: processor.configuration, model: model,
                processor: processor, tokenizer: processor.tokenizer)

            var parameters = GenerateParameters(maxTokens: 4, temperature: 0)
            switch scenario {
            case "presence": parameters.presencePenalty = 2
            case "frequency": parameters.frequencyPenalty = 2
            case "repetition": parameters.repetitionPenalty = 2
            case "suppression": parameters.suppressTokens = [6]
            case "min-p": parameters.minP = 0.2
            case "one-token": parameters.maxTokens = 1
            default: break
            }
            func input() -> LMInput {
                LMInput(
                    text: .init(tokens: MLXArray([Int32(1), 2])),
                    image: scenario == "media"
                        ? .init(pixels: MLXArray.ones([1, 3, 2, 2])) : nil)
            }
            let ar = BatchEngine(context: context, maxBatchSize: batchSize)
            let reference = await collect(await ar.generate(input: input(), parameters: parameters))
            await ar.shutdown()
            #expect(reference.info?.generationTokenCount == parameters.maxTokens)
            #expect(reference.info?.stopReason != .cancelled)
            #expect(!reference.text.isEmpty)
            let mediaPreparesBefore = model.mediaPrepares

            // No drafter exists here. An ineligible request must never try to load it.
            parameters.draftStrategy = .dflash2(
                drafterPath: URL(fileURLWithPath: "/nonexistent/dflash-fallback-fixture"),
                blockSize: scenario == "width-one" ? 1 : 8)
            #expect(
                DFlash2TokenIterator.unservableReason(
                    parameters, input: input(), model: model) != nil)
            let engine = BatchEngine(context: context, maxBatchSize: batchSize)
            let fallback = await collect(
                await engine.generate(input: input(), parameters: parameters))
            await engine.shutdown()

            #expect(fallback.info?.stopReason == reference.info?.stopReason)
            #expect(fallback.info?.generationTokenCount == reference.info?.generationTokenCount)
            #expect(fallback.text == reference.text)
            if scenario == "media" {
                #expect(
                    model.mediaPrepares == mediaPreparesBefore + 1,
                    "ordinary preparation must consume the media rather than dropping it")
            }
        }
    }

    private func collect(_ stream: AsyncStream<Generation>) async -> (
        text: String, info: GenerateCompletionInfo?
    ) {
        var text = ""
        var info: GenerateCompletionInfo?
        for await event in stream {
            switch event {
            case .chunk(let chunk): text += chunk
            case .info(let completion): info = completion
            default: break
            }
        }
        return (text, info)
    }
}

private final class FallbackFixtureModel: Module, LanguageModel,
    KVCacheDimensionProvider, @unchecked Sendable
{
    var kvHeads: [Int] { [1] }
    private(set) var mediaPrepares = 0
    private var preferred = 6

    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        preferred = 6
        if let image = input.image {
            mediaPrepares += 1
            // A pixel-dependent signal makes silently dropping media observable.
            preferred = image.pixels.sum().item(Float.self) > 0 ? 9 : 6
        }
        return .tokens(input.text)
    }

    func callAsFunction(_ input: MLXArray, cache: [KVCache]?) -> MLXArray {
        var values = [Float](repeating: 0, count: 16)
        values[6] = 1
        values[9] = preferred == 9 ? 2 : 0.8
        return MLX.broadcast(
            MLXArray(values).reshaped([1, 1, 16]),
            to: [input.dim(0), input.dim(1), 16])
    }
}
