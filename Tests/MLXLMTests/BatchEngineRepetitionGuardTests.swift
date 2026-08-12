// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXLMCommon
import MLXNN
@preconcurrency import VMLXTokenizers
import XCTest

/// The degenerate-repetition guard lives in two places because the streaming
/// pipeline exists in two places: `TextToolTokenLoopHandler` (solo
/// `Evaluate.generate`) and the inner `Task.detached` bridge in
/// `BatchEngine.generate`. The batch copy is not a refactor of the solo one —
/// it is a parallel implementation, so a guard added to one is simply absent
/// from the other.
///
/// That is not hypothetical. The guard shipped in the solo handler while every
/// host that generates through `BatchEngine` — which is the path the chat app
/// uses — kept running unguarded. The collapse it was written for (a model
/// repeating two sentences until the token cap, finishing with no stop reason
/// at all) was still reachable in the product after the fix was merged.
///
/// So this test drives `BatchEngine.generate` end to end rather than unit-testing
/// the detector: a model that emits one token forever, and a tokenizer that
/// decodes it to a long primitive sentence. Without the wiring the run reaches
/// `maxTokens`; with it, the guard cuts the turn early and classifies it.
final class BatchEngineRepetitionGuardTests: XCTestCase {

    /// Emits the same token id forever, so the visible text is a pure cycle.
    private final class OneTokenModel: Module, LanguageModel, @unchecked Sendable {
        let vocabularySize: Int
        let tokenID: Int

        init(vocabularySize: Int, tokenID: Int) {
            self.vocabularySize = vocabularySize
            self.tokenID = tokenID
            super.init()
        }

        private func peakedLogits(count: Int) -> MLXArray {
            var row = [Float](repeating: -30, count: vocabularySize)
            row[tokenID] = 30
            let flat = MLXArray(Array(repeating: row, count: count).flatMap { $0 })
            return flat.reshaped([1, count, vocabularySize])
        }

        func callAsFunction(
            _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
        ) -> LMOutput {
            LMOutput(logits: peakedLogits(count: input.tokens.dim(-1)))
        }

        func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
            peakedLogits(count: inputs.dim(-1))
        }

        func newCache(parameters: GenerateParameters?) -> [KVCache] {
            [KVCacheSimple()]
        }

        func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult
        {
            .tokens(input.text)
        }
    }

    /// Decodes every id to the same sentence, so the visible stream is a pure
    /// verbatim cycle. 43 characters and primitive, clearing the detector's
    /// minimum-unit and primitivity rules — it deliberately ignores short
    /// filler like `---` or `| | |`.
    private struct RepeatingTokenizer: MLXLMCommon.Tokenizer {
        static let unit = "The answer is AppleScript; I do not repeat. "

        let _eosTokenId = 1_000_001
        let _unknownTokenId = 1_000_002

        func encode(text: String, addSpecialTokens: Bool) -> [Int] { [1, 2, 3] }

        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
            String(repeating: Self.unit, count: tokenIds.count)
        }

        /// Must return nil for unknown strings. BatchEngine widens its stop set
        /// by resolving candidate stop-token spellings through this method, so a
        /// tokenizer that answers with a real id for everything registers the
        /// model's only token as EOS — the run then ends at zero tokens and the
        /// halt assertions below pass without generating anything.
        func convertTokenToId(_ token: String) -> Int? { token == "EOS" ? _eosTokenId : nil }
        func convertIdToToken(_ id: Int) -> String? {
            id == _eosTokenId ? "EOS" : Self.unit
        }

        var bosToken: String? = nil
        var eosToken: String? = nil
        var eosTokenId: Int? { _eosTokenId }
        var unknownToken: String? = nil
        var unknownTokenId: Int? { _unknownTokenId }

        func applyChatTemplate(
            messages: [[String: any Sendable]],
            tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] {
            encode(text: "", addSpecialTokens: false)
        }
    }

    private func makeEngine() -> BatchEngine {
        let vocabSize = 64
        let model = OneTokenModel(vocabularySize: vocabSize, tokenID: 7)
        MLX.eval(model)

        var modelConfig = ModelConfiguration(id: "test-repetition")
        modelConfig.eosTokenIds = []

        let tokenizer = RepeatingTokenizer()
        let processor = TestInputProcessor(
            tokenizer: tokenizer,
            configuration: modelConfig,
            messageGenerator: DefaultMessageGenerator()
        )
        nonisolated(unsafe) let context = ModelContext(
            configuration: modelConfig,
            model: model,
            processor: processor,
            tokenizer: tokenizer
        )
        return BatchEngine(context: context, maxBatchSize: 1)
    }

    /// The turn must end well before the cap, and must carry a stop reason —
    /// the reported collapse ran to the cap and recorded none.
    func testRepetitionHaltsTheBatchStreamAndClassifiesIt() async throws {
        let engine = makeEngine()
        let maxTokens = 400

        let stream = await engine.generate(
            input: LMInput(tokens: MLXArray(Int32(1) ..< Int32(4))),
            parameters: GenerateParameters(maxTokens: maxTokens, temperature: 0)
        )

        var text = ""
        var info: GenerateCompletionInfo?
        for await event in stream {
            switch event {
            case .chunk(let c): text += c
            case .info(let i): info = i
            case .reasoning, .toolCall, .toolCallProgress, .prefillProgress: break
            }
        }

        let completion = try XCTUnwrap(info, "no terminal .info event")
        XCTAssertEqual(
            completion.stopReason, .stop,
            "a repetition halt must be classified, not left as exhaustion")
        XCTAssertLessThan(
            completion.generationTokenCount, maxTokens,
            "the guard must cut the turn before the token cap; got \(completion.generationTokenCount)")
        XCTAssertTrue(
            text.contains(RepeatingTokenizer.unit),
            "text emitted before the halt must still reach the consumer")
    }

    /// `VMLX_REPETITION_STOP=0` must disable the batch copy too, or the two
    /// pipelines disagree about an escape hatch users are told is global.
    func testEnvironmentOptOutIsHonoredByTheSameConstruction() {
        XCTAssertFalse(
            RepetitionCycleDetector.fromEnvironment(["VMLX_REPETITION_STOP": "0"]).isEnabled)
        XCTAssertTrue(RepetitionCycleDetector.fromEnvironment([:]).isEnabled)
    }
}
