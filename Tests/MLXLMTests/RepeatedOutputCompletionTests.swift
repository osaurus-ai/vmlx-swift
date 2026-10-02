// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLMCommon

/// Repeated text is data, not an authoritative completion signal. Exercise both
/// the B=1 solo path and the scheduled text bridge, including real EOS and limits.
final class RepeatedOutputCompletionTests: XCTestCase {
    private static let unit = "The north gate remains open.\n"

    private final class FixtureModel: Module, LanguageModel, @unchecked Sendable {
        let vocabularySize = 64
        let repeatsBeforeEOS: Int?

        init(repeatsBeforeEOS: Int?) {
            self.repeatsBeforeEOS = repeatsBeforeEOS
            super.init()
        }

        func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
            let length = inputs.dim(-1)
            let offset = cache?.first?.offset ?? 0
            let rows = (0 ..< length).map { position in
                // Three prompt tokens: its final position predicts the first output.
                let outputPosition = max(0, offset + position + 1 - 3)
                let token = repeatsBeforeEOS.map { outputPosition >= $0 ? 8 : 7 } ?? 7
                var row = [Float](repeating: -30, count: vocabularySize)
                row[token] = 30
                return row
            }
            if let first = cache?.first {
                let keys = MLXArray.zeros([inputs.dim(0), 1, length, 1])
                _ = first.update(keys: keys, values: keys)
            }
            return MLXArray(rows.flatMap { $0 }).reshaped([1, length, vocabularySize])
        }

        func callAsFunction(
            _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
        ) -> LMOutput {
            LMOutput(logits: callAsFunction(input.tokens, cache: cache))
        }

        func newCache(parameters: GenerateParameters?) -> [KVCache] { [KVCacheSimple()] }
        func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
            .tokens(input.text)
        }
    }

    private struct FixtureTokenizer: MLXLMCommon.Tokenizer {
        func encode(text: String, addSpecialTokens: Bool) -> [Int] { [1, 2, 3] }
        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
            tokenIds.map { $0 == 7 ? RepeatedOutputCompletionTests.unit : "" }.joined()
        }
        // Unknown spellings must not register the fixture token as EOS.
        func convertTokenToId(_ token: String) -> Int? { token == "EOS" ? 8 : nil }
        func convertIdToToken(_ id: Int) -> String? { id == 8 ? "EOS" : nil }
        var bosToken: String? { nil }
        var eosToken: String? { "EOS" }
        var eosTokenId: Int? { 8 }
        var unknownToken: String? { nil }
        var unknownTokenId: Int? { nil }
        func applyChatTemplate(
            messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] { [1, 2, 3] }
    }

    private func makeEngine(batchSize: Int, repeatsBeforeEOS: Int? = nil) -> BatchEngine {
        let model = FixtureModel(repeatsBeforeEOS: repeatsBeforeEOS)
        MLX.eval(model)
        let configuration = ModelConfiguration(id: "repeated-output-fixture")
        let tokenizer = FixtureTokenizer()
        let processor = TestInputProcessor(
            tokenizer: tokenizer, configuration: configuration,
            messageGenerator: DefaultMessageGenerator())
        let context = ModelContext(
            configuration: configuration, model: model, processor: processor, tokenizer: tokenizer)
        return BatchEngine(context: context, maxBatchSize: batchSize)
    }

    private func collect(_ stream: AsyncStream<Generation>) async -> (
        String, [GenerateCompletionInfo]
    ) {
        var text = ""
        var infos = [GenerateCompletionInfo]()
        for await event in stream {
            switch event {
            case .chunk(let chunk): text += chunk
            case .info(let info):
                infos.append(info)
                print(
                    "fixture tokens=\(info.generationTokenCount) tok/s=\(info.tokensPerSecond) stop=\(info.stopReason)"
                )
            case .reasoning, .toolCall, .toolCallProgress, .prefillProgress: break
            }
        }
        return (text, infos)
    }

    private func assertLength(batchSize: Int) async throws {
        let engine = makeEngine(batchSize: batchSize)
        let stream = await engine.generate(
            input: LMInput(tokens: MLXArray([Int32(1), 2, 3])),
            parameters: GenerateParameters(maxTokens: 80, temperature: 0))
        let (text, infos) = await collect(stream)
        XCTAssertEqual(infos.count, 1)
        let info = try XCTUnwrap(infos.first)
        XCTAssertEqual(info.stopReason, .length)
        XCTAssertEqual(info.generationTokenCount, 80)
        XCTAssertEqual(text, String(repeating: Self.unit, count: 80))
        XCTAssertGreaterThan(info.tokensPerSecond, 0)
        await engine.shutdown()
    }

    private func assertEOS(batchSize: Int) async throws {
        let engine = makeEngine(batchSize: batchSize, repeatsBeforeEOS: 20)
        let stream = await engine.generate(
            input: LMInput(tokens: MLXArray([Int32(1), 2, 3])),
            parameters: GenerateParameters(maxTokens: 80, temperature: 0))
        let (text, infos) = await collect(stream)
        XCTAssertEqual(infos.count, 1)
        let info = try XCTUnwrap(infos.first)
        XCTAssertEqual(info.stopReason, .stop)
        XCTAssertEqual(info.generationTokenCount, 20)
        XCTAssertEqual(text, String(repeating: Self.unit, count: 20))
        XCTAssertGreaterThan(info.tokensPerSecond, 0)
        await engine.shutdown()
    }

    private func assertConfiguredStop(batchSize: Int) async throws {
        let engine = makeEngine(batchSize: batchSize)
        let stream = await engine.generate(
            input: LMInput(tokens: MLXArray([Int32(1), 2, 3])),
            parameters: GenerateParameters(
                maxTokens: 80, temperature: 0, extraStopStrings: ["remains"]))
        let (text, infos) = await collect(stream)
        XCTAssertEqual(infos.count, 1)
        XCTAssertEqual(infos.first?.stopReason, .stop)
        XCTAssertEqual(text, "The north gate ")
        await engine.shutdown()
    }

    private func assertCancellationAndFollowUp(batchSize: Int) async throws {
        let engine = makeEngine(batchSize: batchSize)
        let stream = await engine.generate(
            input: LMInput(tokens: MLXArray([Int32(1), 2, 3])),
            parameters: GenerateParameters(maxTokens: 10_000, temperature: 0))
        var cancelled = false
        var infos = [GenerateCompletionInfo]()
        for await event in stream {
            if case .chunk(let text) = event, !text.isEmpty, !cancelled {
                cancelled = true
                await engine.shutdown()
            }
            if case .info(let info) = event { infos.append(info) }
        }
        XCTAssertTrue(cancelled)
        XCTAssertEqual(infos.count, 1)
        XCTAssertEqual(infos.first?.stopReason, .cancelled)
        // A new engine can use the same fixture after the old generation drains.
        try await assertEOS(batchSize: batchSize)
    }

    func testSoloRepeatedTextReachesConfiguredLength() async throws {
        try await assertLength(batchSize: 1)
    }
    func testScheduledRepeatedTextReachesConfiguredLength() async throws {
        try await assertLength(batchSize: 2)
    }
    func testSoloRepeatedTextReachesRealEOS() async throws { try await assertEOS(batchSize: 1) }
    func testScheduledRepeatedTextReachesRealEOS() async throws {
        try await assertEOS(batchSize: 2)
    }
    func testSoloConfiguredStopStillTruncatesExactly() async throws {
        try await assertConfiguredStop(batchSize: 1)
    }
    func testScheduledConfiguredStopStillTruncatesExactly() async throws {
        try await assertConfiguredStop(batchSize: 2)
    }
    func testSoloCancellationDrains() async throws {
        try await assertCancellationAndFollowUp(batchSize: 1)
    }
    func testScheduledCancellationDrains() async throws {
        try await assertCancellationAndFollowUp(batchSize: 2)
    }
    private func route(
        _ raw: String, stopStrings: [String] = [], tools: [ToolSpec]? = nil,
        reasoning: Bool = false
    ) -> (text: String, reasoning: String, calls: [ToolCall], halted: Bool) {
        let pieces = raw.map(String.init)
        var handler = TextToolTokenLoopHandler(
            tokenizer: StopStringPostStopLeakTests.FixedPieceTokenizer(pieces: pieces),
            format: .json, tools: tools,
            reasoningParser: reasoning ? ReasoningParser() : nil,
            stopStringMatcher: StopStringMatcher(stopStrings: stopStrings))
        var visible = ""
        var thought = ""
        var calls = [ToolCall]()
        let emit: (sending Generation) -> AsyncStream<Generation>.Continuation.YieldResult = {
            event in
            switch event {
            case .chunk(let text): visible += text
            case .reasoning(let text): thought += text
            case .toolCall(let call): calls.append(call)
            default: break
            }
            return .enqueued(remaining: .max)
        }
        var halted = false
        for token in pieces.indices {
            if !handler.onToken(token, emit: emit) {
                halted = true
                break
            }
        }
        handler.onGenerationEnd(emit: emit)
        return (visible, thought, calls, halted)
    }

    func testRepeatedPlainQuotedFencedTableAndFillerDataAreUnmodified() {
        let fixtures = [
            String(repeating: Self.unit, count: 80),
            String(repeating: "> " + Self.unit, count: 80),
            "```text\n" + String(repeating: Self.unit, count: 80) + "```",
            String(repeating: "| north | open | unchanged |\n", count: 80),
            String(repeating: ".", count: 2048),
        ]
        for text in fixtures {
            for stops in [[], ["NEVER_MATCH"]] {
                let result = route(text, stopStrings: stops)
                XCTAssertFalse(result.halted)
                XCTAssertEqual(result.text, text)
                XCTAssertEqual(result.reasoning, "")
                XCTAssertTrue(result.calls.isEmpty)
            }
        }
    }

    func testRepeatedReasoningAndToolArgumentsRemainScoped() throws {
        let repeated = String(repeating: Self.unit, count: 30)
        let payload = try JSONSerialization.data(
            withJSONObject: [
                "name": "record", "arguments": ["text": repeated],
            ], options: [.sortedKeys])
        let parameters: [String: any Sendable] = [
            "type": "object", "properties": ["text": ["type": "string"]],
        ]
        let function: [String: any Sendable] = ["name": "record", "parameters": parameters]
        let tools: [ToolSpec] = [["type": "function", "function": function]]
        let raw =
            "<think>" + repeated + "</think>Visible answer. "
            + "<tool_call>" + String(decoding: payload, as: UTF8.self) + "</tool_call>"
        let result = route(raw, stopStrings: ["remains"], tools: tools, reasoning: true)
        XCTAssertFalse(result.halted)
        XCTAssertEqual(result.text, "Visible answer. ")
        XCTAssertEqual(result.reasoning, repeated)
        XCTAssertEqual(result.calls.count, 1)
        XCTAssertEqual(result.calls.first?.function.name, "record")
        XCTAssertEqual(result.calls.first?.function.arguments["text"], .string(repeated))
    }

}
