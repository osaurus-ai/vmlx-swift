import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLMCommon

/// Exercises real common generation with a deterministic tiny target, not a
/// second implementation of the grammar or decode loop. No real model loading.
final class StructuredOutputGenerationTests: XCTestCase {
    private static func parameters(schema: String = #"{"type":"boolean"}"#, max: Int = 16)
        -> GenerateParameters
    {
        var p = GenerateParameters(maxTokens: max, temperature: 0)
        p.jsonSchema = schema
        return p
    }

    func testActualIteratorStopsBeforeForwardingEOSAgain() throws {
        try MLXMetalTestLock.withLock {
            let model = SchemaGenerationTarget(plan: [1, 0])
            var iterator = try TokenIterator(
                input: LMInput(tokens: MLXArray([Int32(3)])),
                model: model, parameters: Self.parameters(), tokenizer: SchemaGenerationTokenizer(),
                stopTokenIDs: [0])
            XCTAssertEqual(iterator.next(), 1)
            XCTAssertEqual(iterator.next(), 0)
            XCTAssertEqual(
                model.forwarded, [[3], [1]], "EOS must not trigger another model forward")
            XCTAssertNil(iterator.generationFailure)
            XCTAssertEqual(iterator.structuredOutputComplete, true)
        }
    }

    func testAsyncCompletionAndPublicationGateUseActualIterator() async throws {
        try await MLXMetalTestLock.withLock { () async throws -> Void in
            let tokenizer = SchemaGenerationTokenizer()
            let model = SchemaGenerationTarget(plan: [1, 0])
            let base = try TokenIterator(
                input: LMInput(tokens: MLXArray([Int32(3)])),
                model: model, parameters: Self.parameters(), tokenizer: tokenizer, stopTokenIDs: [0]
            )
            let audit = SchemaPublicationAudit()
            let (stream, task) = generateTokenTask(
                promptTokenCount: 1,
                modelConfiguration: ModelConfiguration(id: "fixture/schema"), tokenizer: tokenizer,
                iterator: SchemaPublicationIterator(base: base, audit: audit))
            var ids: [Int] = []
            var terminal: GenerateCompletionInfo?
            for await event in stream {
                switch event {
                case .token(let id): ids.append(id)
                case .info(let info): terminal = info
                case .prefillProgress: break
                }
            }
            await task.value
            XCTAssertEqual(ids, [1])
            XCTAssertNil(try XCTUnwrap(terminal).generationFailure)
            XCTAssertEqual(terminal?.stopReason, .stop)
            XCTAssertEqual(audit.calls, [true], "Successful emitted JSON may publish its boundary")
            XCTAssertEqual(model.forwarded, [[3], [1]])
        }
    }

    func testLengthFailureDoesNotPublishGeneratedBoundary() async throws {
        try await MLXMetalTestLock.withLock { () async throws -> Void in
            let tokenizer = SchemaGenerationTokenizer()
            let model = SchemaGenerationTarget(plan: [6, 1, 7, 2, 8, 0])
            let params = Self.parameters(
                schema: #"{"type":"array","items":{"type":"boolean"},"minItems":2,"maxItems":2}"#,
                max: 1)
            let base = try TokenIterator(
                input: LMInput(tokens: MLXArray([Int32(3)])),
                model: model, parameters: params, tokenizer: tokenizer, stopTokenIDs: [0])
            let audit = SchemaPublicationAudit()
            let (stream, task) = generateTokenTask(
                promptTokenCount: 1,
                modelConfiguration: ModelConfiguration(id: "fixture/schema"), tokenizer: tokenizer,
                iterator: SchemaPublicationIterator(base: base, audit: audit))
            var ids: [Int] = []
            var terminal: GenerateCompletionInfo?
            for await event in stream {
                switch event {
                case .token(let id): ids.append(id)
                case .info(let info): terminal = info
                case .prefillProgress: break
                }
            }
            await task.value
            XCTAssertEqual(ids, [6])
            XCTAssertEqual(terminal?.stopReason, .length)
            XCTAssertEqual(terminal?.generationFailure?.stage, .decoding)
            XCTAssertEqual(
                audit.calls, [false], "Truncation must not publish a successful generated boundary")
        }
    }

    func testInitialConstraintConflictNeverEmitsInternalZero() async throws {
        try await MLXMetalTestLock.withLock { () async throws -> Void in
            let tokenizer = SchemaGenerationTokenizer()
            let model = SchemaGenerationTarget(plan: [1, 0])
            var params = Self.parameters()
            params.suppressTokens = Array(0 ..< 9)
            let base = try TokenIterator(
                input: LMInput(tokens: MLXArray([Int32(3)])),
                model: model, parameters: params, tokenizer: tokenizer, stopTokenIDs: [0])
            let audit = SchemaPublicationAudit()
            let (stream, task) = generateTokenTask(
                promptTokenCount: 1,
                modelConfiguration: ModelConfiguration(id: "fixture/schema"), tokenizer: tokenizer,
                iterator: SchemaPublicationIterator(base: base, audit: audit),
                includeStopToken: true)
            var ids: [Int] = []
            var terminal: GenerateCompletionInfo?
            for await event in stream {
                switch event {
                case .token(let id): ids.append(id)
                case .info(let info): terminal = info
                case .prefillProgress: break
                }
            }
            await task.value
            XCTAssertEqual(ids, [], "Neither sentinel0 nor a fabricated EOS may be emitted")
            XCTAssertEqual(terminal?.generationFailure?.stage, .decoding)
            XCTAssertEqual(audit.calls, [false])
            XCTAssertEqual(model.forwarded, [[3]])
        }
    }

    func testLengthWithEOSAlreadySampledStillFailsPublication() async throws {
        try await MLXMetalTestLock.withLock { () async throws -> Void in
            let tokenizer = SchemaGenerationTokenizer()
            let target = SchemaGenerationTarget(plan: [1, 0])
            let iterator = try TokenIterator(
                input: LMInput(tokens: MLXArray([Int32(3)])),
                model: target, parameters: Self.parameters(max: 1), tokenizer: tokenizer,
                stopTokenIDs: [0])
            let audit = SchemaPublicationAudit()
            let (stream, task) = generateTokenTask(
                promptTokenCount: 1,
                modelConfiguration: ModelConfiguration(id: "fixture/schema"), tokenizer: tokenizer,
                iterator: SchemaPublicationIterator(base: iterator, audit: audit))
            var ids: [Int] = []
            var terminal: GenerateCompletionInfo?
            for await event in stream {
                if case .token(let id) = event { ids.append(id) }
                if case .info(let info) = event { terminal = info }
            }
            await task.value
            XCTAssertEqual(ids, [1])
            XCTAssertEqual(target.forwarded, [[3], [1]], "The iterator really sampled EOS ahead")
            XCTAssertEqual(terminal?.stopReason, .length)
            XCTAssertEqual(terminal?.generationFailure?.stage, .decoding)
            XCTAssertEqual(audit.calls, [false])
        }
    }

    func testCancellationWithEOSLookaheadFailsPublication() async throws {
        try await MLXMetalTestLock.withLock { () async throws -> Void in
            let tokenizer = SchemaGenerationTokenizer()
            let target = SchemaGenerationTarget(plan: [1, 0])
            let iterator = try TokenIterator(
                input: LMInput(tokens: MLXArray([Int32(3)])),
                model: target, parameters: Self.parameters(), tokenizer: tokenizer,
                stopTokenIDs: [0])
            let audit = SchemaPublicationAudit()
            let (stream, task) = generateTokenTask(
                promptTokenCount: 1,
                modelConfiguration: ModelConfiguration(id: "fixture/schema"), tokenizer: tokenizer,
                iterator: SchemaPublicationIterator(
                    base: iterator, audit: audit, cancelOnNext: true))
            var ids: [Int] = []
            var terminal: GenerateCompletionInfo?
            for await event in stream {
                if case .token(let id) = event { ids.append(id) }
                if case .info(let info) = event { terminal = info }
            }
            await task.value
            XCTAssertEqual(ids, [])
            XCTAssertEqual(target.forwarded, [[3], [1]])
            XCTAssertEqual(terminal?.stopReason, .cancelled)
            XCTAssertEqual(terminal?.generationFailure?.stage, .decoding)
            XCTAssertEqual(audit.calls, [false])
        }
    }

    func testExplicitTextStopRejectedBeforeAnyForward() throws {
        try MLXMetalTestLock.withLock {
            let target = SchemaGenerationTarget(plan: [1, 0])
            let tokenizer = SchemaGenerationTokenizer()
            let context = ModelContext(
                configuration: ModelConfiguration(id: "fixture/schema"),
                model: target, processor: StandInUserInputProcessor(), tokenizer: tokenizer)
            var parameters = Self.parameters()
            parameters.extraStopStrings = ["true"]
            XCTAssertThrowsError(
                try generateTokensTask(
                    input: LMInput(tokens: MLXArray([Int32(3)])),
                    parameters: parameters, context: context)
            ) { error in
                XCTAssertEqual((error as? GenerationFailure)?.stage, .preparation)
            }
            XCTAssertTrue(target.forwarded.isEmpty)
        }
    }

    func testBatchEngineSchemaIsolationAndFailedSlotDoNotChangeOrdinaryRequest() async throws {
        try await MLXMetalTestLock.withLock { () async throws -> Void in
            let tokenizer = SchemaGenerationTokenizer()
            let context = ModelContext(
                configuration: ModelConfiguration(id: "fixture/schema-batch"),
                model: SchemaBatchTarget(), processor: StandInUserInputProcessor(),
                tokenizer: tokenizer)
            let engine = BatchEngine(context: context, maxBatchSize: 4)
            let (_, booleanStream) = await engine.submit(
                input: LMInput(tokens: MLXArray([Int32(3)])), parameters: Self.parameters())
            let (_, falseStream) = await engine.submit(
                input: LMInput(tokens: MLXArray([Int32(3)])),
                parameters: Self.parameters(schema: #"{"type":"boolean","enum":[false]}"#))
            var conflict = Self.parameters()
            conflict.suppressTokens = Array(0 ..< 9)
            let (_, failedStream) = await engine.submit(
                input: LMInput(tokens: MLXArray([Int32(3)])), parameters: conflict)
            let (_, ordinaryStream) = await engine.submit(
                input: LMInput(tokens: MLXArray([Int32(3)])),
                parameters: GenerateParameters(maxTokens: 16, temperature: 0))
            let streams = [booleanStream, falseStream, failedStream, ordinaryStream]
            let expected = [[1], [2], [], [1]]
            for (index, stream) in streams.enumerated() {
                var ids: [Int] = []
                var terminal: GenerateCompletionInfo?
                for await event in stream {
                    if case .token(let id) = event { ids.append(id) }
                    if case .info(let info) = event { terminal = info }
                }
                XCTAssertEqual(ids, expected[index], "Wrong tokens for slot \(index)")
                let info = try XCTUnwrap(terminal)
                if index == 2 {
                    XCTAssertEqual(info.generationFailure?.stage, .decoding)
                } else {
                    XCTAssertEqual(info.stopReason, .stop)
                    XCTAssertNil(info.generationFailure)
                }
            }
        }
    }

    func testDirectSpeculativeConstructorRejectsSchemaBeforeForward() throws {
        try MLXMetalTestLock.withLock {
            let target = SchemaGenerationTarget(plan: [1, 0])
            let draft = SchemaGenerationTarget(plan: [1, 0])
            XCTAssertThrowsError(
                try SpeculativeTokenIterator(
                    input: LMInput(tokens: MLXArray([Int32(3)])),
                    mainModel: target, draftModel: draft, parameters: Self.parameters(),
                    numDraftTokens: 2))
            XCTAssertTrue(target.forwarded.isEmpty)
            XCTAssertTrue(draft.forwarded.isEmpty)
        }
    }

    func testStructuredTextPreservesLiteralReasoningAndToolMarkers() async throws {
        try await MLXMetalTestLock.withLock { () async throws -> Void in
            let tokenizer = SchemaMarkerTokenizer()
            let target = SchemaGenerationTarget(plan: [1, 0], vocabularySize: 3)
            let context = ModelContext(
                configuration: ModelConfiguration(
                    id: "fixture/schema",
                    toolCallFormat: .json, reasoningParserName: "qwen3"),
                model: target, processor: StandInUserInputProcessor(), tokenizer: tokenizer)
            let stream = try generate(
                input: LMInput(tokens: MLXArray([Int32(2)])),
                parameters: Self.parameters(schema: #"{"type":"string"}"#), context: context)
            var text = ""
            var terminal: GenerateCompletionInfo?
            for await event in stream {
                switch event {
                case .chunk(let chunk): text += chunk
                case .info(let info): terminal = info
                case .prefillProgress: break
                case .reasoning, .toolCall, .toolCallProgress:
                    XCTFail("JSON string contents must not become protocol events")
                }
            }
            XCTAssertEqual(text, SchemaMarkerTokenizer.json)
            XCTAssertEqual(terminal?.stopReason, .stop)
            XCTAssertNil(try XCTUnwrap(terminal).generationFailure)
        }
    }

    func testContextGenerationRejectsActiveReasoningWithoutChangingPrompt() throws {
        try MLXMetalTestLock.withLock {
            let target = SchemaGenerationTarget(plan: [1, 0])
            let tokenizer = SchemaGenerationTokenizer(reasoningPrompt: true)
            let context = ModelContext(
                configuration: ModelConfiguration(
                    id: "fixture/schema", reasoningParserName: "qwen3"),
                model: target, processor: StandInUserInputProcessor(), tokenizer: tokenizer)
            XCTAssertThrowsError(
                try generateTokensTask(
                    input: LMInput(tokens: MLXArray([Int32(3)])),
                    parameters: Self.parameters(), context: context)
            ) { error in
                XCTAssertEqual((error as? GenerationFailure)?.stage, .preparation)
            }
            XCTAssertTrue(target.forwarded.isEmpty)
            XCTAssertEqual(tokenizer.decode(tokenIds: [3], skipSpecialTokens: false), "<think>\n")
        }
    }
}

private final class SchemaGenerationTarget: Module, LanguageModel {
    let plan: [Int]
    var forwarded: [[Int]] = []
    let vocabularySize: Int
    init(plan: [Int], vocabularySize: Int = 9) {
        self.plan = plan
        self.vocabularySize = vocabularySize
    }
    var kvHeads: [Int] { [] }
    func newCache(parameters: GenerateParameters?) -> [KVCache] { [] }
    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        .tokens(input.text)
    }
    func callAsFunction(_ input: MLXArray, cache: [KVCache]?) -> MLXArray {
        let next = plan[min(forwarded.count, plan.count - 1)]
        forwarded.append(input.asArray(Int.self))
        var values = Array(repeating: Float(-10), count: vocabularySize)
        values[next] = 10
        return MLXArray(values).reshaped(1, 1, vocabularySize)
    }
}

private struct SchemaGenerationTokenizer: MLXLMCommon.Tokenizer {
    var reasoningPrompt = false
    private let pieces = ["<eos>", "true", "false", "<prompt>", " ", "<tool>", "[", ",", "]"]
    var grammarTokenVocabulary: GrammarTokenVocabulary? {
        .init(vocabulary: pieces, vocabularyType: .raw, specialTokenIDs: [0, 3, 5])
    }
    var bosToken: String? { nil }
    var eosToken: String? { "<eos>" }
    var unknownToken: String? { nil }
    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [3] }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        if reasoningPrompt, tokenIds == [3] { return "<think>\n" }
        return tokenIds.map { pieces[$0] }.joined()
    }
    func convertTokenToId(_ token: String) -> Int? { pieces.firstIndex(of: token) }
    func convertIdToToken(_ id: Int) -> String? { pieces.indices.contains(id) ? pieces[id] : nil }
    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [3] }
}

/// Observes the real generation loop's publication flag and delegates to the
/// actual iterator. This proves the lifecycle gate, not physical SSD writes.
private final class SchemaPublicationAudit: @unchecked Sendable { var calls: [Bool] = [] }
private struct SchemaPublicationIterator: TokenIteratorProtocol {
    var base: TokenIterator
    let audit: SchemaPublicationAudit
    var cancelOnNext = false
    var maxTokens: Int? { base.maxTokens }
    var tokenCount: Int { base.tokenCount }
    var promptPrefillTime: TimeInterval { base.promptPrefillTime }
    var promptTokenIds: [Int] { base.promptTokenIds }
    var generationFailure: GenerationFailure? { base.generationFailure }
    var structuredOutputComplete: Bool? { base.structuredOutputComplete }
    mutating func next() -> Int? {
        let token = base.next()
        if cancelOnNext { withUnsafeCurrentTask { $0?.cancel() } }
        return token
    }
    mutating func storeCacheAfterGeneration(
        generatedTokenIds: [Int], includeGeneratedBoundary: Bool
    ) {
        audit.calls.append(includeGeneratedBoundary)
        base.storeCacheAfterGeneration(
            generatedTokenIds: generatedTokenIds, includeGeneratedBoundary: includeGeneratedBoundary
        )
    }
}

/// Markers are ordinary vocabulary bytes inside the JSON string, not special IDs.
private struct SchemaMarkerTokenizer: MLXLMCommon.Tokenizer {
    static let json = #""<think>literal</think><tool_call>literal</tool_call>""#
    private var pieces: [String] { ["<eos>", Self.json, "<prompt>"] }
    var grammarTokenVocabulary: GrammarTokenVocabulary? {
        .init(vocabulary: pieces, vocabularyType: .raw, specialTokenIDs: [0, 2])
    }
    var bosToken: String? { nil }
    var eosToken: String? { "<eos>" }
    var unknownToken: String? { nil }
    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [2] }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        tokenIds.map { pieces[$0] }.joined()
    }
    func convertTokenToId(_ token: String) -> Int? { pieces.firstIndex(of: token) }
    func convertIdToToken(_ id: Int) -> String? { pieces.indices.contains(id) ? pieces[id] : nil }
    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [2] }
}

/// Input-local deterministic logits support the scheduler's genuine B-wide
/// forwards without shared token-position state. Constraints alone select false.
private final class SchemaBatchTarget: Module, LanguageModel {
    var vocabularySize: Int { 9 }
    var kvHeads: [Int] { [] }
    func newCache(parameters: GenerateParameters?) -> [KVCache] { [] }
    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        .tokens(input.text)
    }
    func callAsFunction(_ input: MLXArray, cache: [KVCache]?) -> MLXArray {
        let batch = input.ndim == 1 ? 1 : input.dim(0)
        let sequence = input.ndim == 1 ? input.size : input.dim(1)
        let tokens = input.asArray(Int.self)
        var values: [Float] = []
        for row in 0 ..< batch {
            for position in 0 ..< sequence {
                let token = tokens[row * sequence + position]
                var logits = Array(repeating: Float(-10), count: vocabularySize)
                if token == 3 {
                    logits[1] = 10
                    logits[2] = 9
                } else {
                    logits[0] = 10
                }
                values.append(contentsOf: logits)
            }
        }
        return MLXArray(values).reshaped(batch, sequence, vocabularySize)
    }
}

extension StructuredOutputGenerationTests {
    func testAdditionalProcessorCannotReopenSchemaMaskOrHideEOS() throws {
        try MLXMetalTestLock.withLock {
            let model = SchemaGenerationTarget(plan: [1, 0])
            var iterator = try TokenIterator(
                input: LMInput(tokens: MLXArray([Int32(3)])), model: model,
                parameters: Self.parameters(), additionalProcessor: SchemaReopeningProcessor(),
                tokenizer: SchemaGenerationTokenizer(), stopTokenIDs: [0])
            XCTAssertEqual(iterator.next(), 1)
            XCTAssertEqual(iterator.next(), 0)
            XCTAssertEqual(model.forwarded, [[3], [1]])
            XCTAssertEqual(iterator.structuredOutputComplete, true)
            XCTAssertNil(iterator.generationFailure)
        }
    }

    func testAdditionalConstraintFailureStopsSchemaGeneration() throws {
        try MLXMetalTestLock.withLock {
            let model = SchemaGenerationTarget(plan: [1, 0])
            var iterator = try TokenIterator(
                input: LMInput(tokens: MLXArray([Int32(3)])), model: model,
                parameters: Self.parameters(), additionalProcessor: SchemaFailingProcessor(),
                tokenizer: SchemaGenerationTokenizer(), stopTokenIDs: [0])
            XCTAssertNil(iterator.next())
            XCTAssertNotNil(iterator.generationFailure)
            XCTAssertEqual(iterator.structuredOutputComplete, false)
            XCTAssertEqual(model.forwarded, [[3]])
        }
    }

    func testAdditionalOrdinaryProcessorDoesNotActivateStructuredOutput() throws {
        try MLXMetalTestLock.withLock {
            var p = GenerateParameters(maxTokens: 2, temperature: 0)
            p.jsonSchema = nil
            let iterator = try TokenIterator(
                input: LMInput(tokens: MLXArray([Int32(3)])),
                model: SchemaGenerationTarget(plan: [1, 0]), parameters: p,
                additionalProcessor: SchemaReopeningProcessor())
            XCTAssertNil(iterator.structuredOutputComplete)
        }
    }
}

/// Deliberately discards the incoming mask and strongly favors invalid bytes.
/// The native schema must apply after this processor and retain EOS semantics.
private struct SchemaReopeningProcessor: LogitProcessor {
    mutating func prompt(_ prompt: MLXArray) {}
    mutating func didSample(token: MLXArray) {}
    func process(logits: MLXArray) -> MLXArray {
        MLXArray([Float(20), 10, 9, 100, 0, 0, 0, 0, 0]).reshaped(logits.shape)
    }
}

private struct SchemaFailingProcessor: LogitProcessor, ConstraintFailureReporting {
    var constraintFailure: JSONSchemaGrammarError? { .runtime("fixture additional constraint failed") }
    var constraintIsComplete: Bool { false }
    mutating func prompt(_ prompt: MLXArray) {}
    mutating func didSample(token: MLXArray) {}
    func process(logits: MLXArray) -> MLXArray { logits }
}
