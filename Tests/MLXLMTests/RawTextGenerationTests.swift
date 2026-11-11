import Foundation
import Testing
@testable import MLXLMCommon

struct RawTextGenerationTests {
    private struct Pieces: Tokenizer {
        let pieces: [String]
        var bosToken: String? { nil }
        var eosToken: String? { nil }
        var unknownToken: String? { nil }
        func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
            tokenIds.map { pieces[$0] }.joined()
        }
        func convertTokenToId(_ token: String) -> Int? { pieces.firstIndex(of: token) }
        func convertIdToToken(_ id: Int) -> String? { pieces[id] }
        func applyChatTemplate(messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
                               additionalContext: [String: any Sendable]?) throws -> [Int] { [] }
    }

    private func run(_ pieces: [String], raw: Bool, stops: [String] = [], structured: Bool = false) -> (text: String, reasoning: String, stopped: Bool) {
        var handler = TextToolTokenLoopHandler(
            tokenizer: Pieces(pieces: pieces), format: .json,
            reasoningParser: ReasoningParser.forPrompt(stampName: "qwen3", promptTail: nil),
            stopStringMatcher: StopStringMatcher(stopStrings: stops), parseChatMarkup: !raw,
            structuredOutput: structured)
        var text = ""
        var reasoning = ""
        let emit: (sending Generation) -> AsyncStream<Generation>.Continuation.YieldResult = { event in
            switch event {
            case .chunk(let value): text += value
            case .reasoning(let value): reasoning += value
            default: break
            }
            return .enqueued(remaining: .max)
        }
        for token in pieces.indices {
            if !handler.onToken(token, emit: emit) { break }
        }
        handler.onGenerationEnd(emit: emit)
        return (text, reasoning, handler.stopSequenceHit)
    }

    @Test func rawTextDoesNotDisappearIntoAChatReasoningChannel() {
        let pieces = ["one, ", "two, ", "three."]
        let raw = run(pieces, raw: true)
        #expect(raw.text == pieces.joined())
        #expect(raw.reasoning.isEmpty)
        let chat = run(pieces, raw: false)
        #expect(chat.text.isEmpty)
        #expect(chat.reasoning == pieces.joined())
    }

    @Test func rawTextPreservesLiteralReasoningAndToolMarkup() {
        let pieces = ["<think>", "literal 日本語", "</think>\n", "<tool_call>",
                      "{\"name\":\"literal\",\"arguments\":{}}", "</tool_call>"]
        let output = run(pieces, raw: true)
        #expect(output.text == pieces.joined())
        #expect(output.reasoning.isEmpty)
        #expect(!output.stopped)
    }

    @Test func rawStopAcrossPiecesTruncatesWithoutLeakingTheHeldTail() {
        let pieces = ["one, two, ", "ST", "OP", ", three, four, five, six, seven, eight."]
        let output = run(pieces, raw: true, stops: ["STOP"])
        #expect(output.text == "one, two, ")
        #expect(output.stopped)
    }
    @Test func schemaModeStillPreservesMarkupAndUsesGrammarTermination() {
        let pieces = ["{\"literal\":\"<think>", "STOP", "</think><tool_call>\"}"]
        for raw in [false, true] {
            let output = run(pieces, raw: raw, stops: ["STOP"], structured: true)
            #expect(output.text == pieces.joined())
            #expect(output.reasoning.isEmpty)
            #expect(!output.stopped)
        }
    }

}
