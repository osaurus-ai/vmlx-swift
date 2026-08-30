// Copyright © 2025 Apple Inc.

import MLX
import MLXLLM
import MLXLMCommon
import Testing

@Suite(.serialized)
struct SpeculativeDecodingTests {

    let processor: any UserInputProcessor
    let mainContext: ModelContext
    let draftContext: ModelContext

    init() {
        MLXRandom.seed(0x5EC_DEC0D)

        let processor = TestInputProcessor()
        let modelConfig = Gemma3TextConfiguration(
            modelType: "text",
            hiddenSize: 64, hiddenLayers: 8, intermediateSize: 64,
            attentionHeads: 4, headDim: 64,
            rmsNormEps: 0.00001, vocabularySize: 100, kvHeads: 4,
            ropeTheta: 1_000_000, ropeLocalBaseFreq: 10_000,
            ropeTraditional: false, queryPreAttnScalar: 256,
            slidingWindow: 512, slidingWindowPattern: 1,
            maxPositionEmbeddings: 32768
        )

        let mainModel = Gemma3TextModel(modelConfig)
        let mainContext = ModelContext(
            configuration: processor.configuration,
            model: mainModel,
            processor: processor,
            tokenizer: processor.tokenizer
        )

        let draftModel = Gemma3TextModel(modelConfig)
        let draftContext = ModelContext(
            configuration: processor.configuration,
            model: draftModel,
            processor: processor,
            tokenizer: processor.tokenizer
        )

        eval(mainModel, draftModel)
        self.processor = processor
        self.mainContext = mainContext
        self.draftContext = draftContext
    }

    @Test(arguments: [2, 8, 48], [false, true])
    func `Speculative decoding matches default token generation`(
        numDraftTokens: Int,
        withLogitProcessor: Bool
    ) async throws {
        let modelInput = LMInput(tokens: MLXArray([3, 5, 7, 11, 13, 17, 19, 23]))
        let parameters = GenerateParameters(
            maxTokens: 32,
            temperature: 0.0,  // Use greedy decoding for deterministic output
            repetitionPenalty: withLogitProcessor ? 1.5 : nil,
            presencePenalty: withLogitProcessor ? 0.5 : nil,
            frequencyPenalty: withLogitProcessor ? 0.2 : nil,
        )

        var normalTokens: [Int] = []
        for await generation in try generateTokens(
            input: modelInput, parameters: parameters, context: mainContext
        ) {
            if let token = generation.token { normalTokens.append(token) }
        }

        var speculativeTokens: [Int] = []
        for await generation in try generateTokens(
            input: modelInput, parameters: parameters, context: mainContext,
            draftModel: draftContext.model, numDraftTokens: numDraftTokens
        ) {
            if let token = generation.token { speculativeTokens.append(token) }
        }

        #expect(!normalTokens.isEmpty)
        #expect(!speculativeTokens.isEmpty)
        #expect(normalTokens == speculativeTokens)
    }
    /// Isolates WHICH penalty breaks speculative equivalence.
    ///
    /// `RepetitionContext` holds a `TokenRing`, whose `append` REASSIGNS its `MLXArray` buffer, so a
    /// struct copy is genuinely independent. `PresencePenaltyContext` and `FrequencyPenaltyContext`
    /// hold `GeneratedTokenCounts`, which is a `final class` mutated in place by `record`. The
    /// speculative verifier copies the processor per round and walks it across all `numDraft + 1`
    /// positions — including drafts it then REJECTS — so with a shared counts object those rejected
    /// tokens are counted into the real processor, and accepted ones are counted twice.
    @Test(arguments: ["repetition", "presence", "frequency"])
    func whichPenaltyBreaksSpeculativeEquivalence(which: String) async throws {
        let modelInput = LMInput(tokens: MLXArray([3, 5, 7, 11, 13, 17, 19, 23]))
        let parameters = GenerateParameters(
            maxTokens: 32,
            temperature: 0.0,
            repetitionPenalty: which == "repetition" ? 1.5 : nil,
            presencePenalty: which == "presence" ? 0.5 : nil,
            frequencyPenalty: which == "frequency" ? 0.2 : nil,
        )
        var normal: [Int] = []
        for await g in try generateTokens(
            input: modelInput, parameters: parameters, context: mainContext)
        {
            if let t = g.token { normal.append(t) }
        }
        var speculative: [Int] = []
        for await g in try generateTokens(
            input: modelInput, parameters: parameters, context: mainContext,
            draftModel: draftContext.model, numDraftTokens: 8)
        {
            if let t = g.token { speculative.append(t) }
        }
        #expect(normal == speculative, "\(which) penalty diverges under speculative decoding")
    }

}
