// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import Testing

@testable import MLXLMCommon

/// The stop-string matcher holds back text that could be the start of a stop
/// string. When a tool call completes, that held text — which PRECEDES the
/// call in the model's output — must be emitted BEFORE the `.toolCall` event:
/// consumers deliberately suppress all text once a tool call lands (no-leak
/// guard for post-tool prose), so a tail emitted after the event is silently
/// dropped. Live signature on ornith-1.0-9b-mxfp8: the assistant's sentence
/// cut mid-word right before every tool call ("…removing the temporary
/// director").
struct ToolCallStopMatcherOrderingTests {

    struct FixedPieceTokenizer: MLXLMCommon.Tokenizer {
        let pieces: [String]
        var vocabularySize: Int { pieces.count }
        func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
            tokenIds.map { pieces.indices.contains($0) ? pieces[$0] : "" }.joined()
        }
        func convertTokenToId(_ token: String) -> Int? { pieces.firstIndex(of: token) }
        func convertIdToToken(_ id: Int) -> String? {
            pieces.indices.contains(id) ? pieces[id] : nil
        }
        var bosToken: String? = nil
        var eosToken: String? = nil
        var eosTokenId: Int? { nil }
        var unknownToken: String? = nil
        var unknownTokenId: Int? { nil }
        func applyChatTemplate(
            messages: [[String: any Sendable]],
            tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] { [] }
    }

    private static let runCommandTool: [String: any Sendable] = [
        "type": "function",
        "function": [
            "name": "run_command",
            "description": "Run a shell command.",
            "parameters": [
                "type": "object",
                "properties": ["command": ["type": "string"]],
                "required": ["command"],
            ] as [String: any Sendable],
        ] as [String: any Sendable],
    ]

    private enum Event: Equatable {
        case chunk(String)
        case toolCall(String)
    }

    private func run(pieces: [String], stops: [String]) -> [Event] {
        var handler = TextToolTokenLoopHandler(
            tokenizer: FixedPieceTokenizer(pieces: pieces),
            format: .xmlFunction,
            tools: [Self.runCommandTool],
            reasoningParser: nil,
            stopStringMatcher: StopStringMatcher(stopStrings: stops)
        )
        var events: [Event] = []
        let emit: (sending Generation) -> AsyncStream<Generation>.Continuation.YieldResult = {
            event in
            switch event {
            case .chunk(let text): events.append(.chunk(text))
            case .toolCall(let call): events.append(.toolCall(call.function.name))
            default: break
            }
            return .enqueued(remaining: .max)
        }
        for token in pieces.indices {
            if !handler.onToken(token, emit: emit) { break }
        }
        handler.onGenerationEnd(emit: emit)
        return events
    }

    @Test("stop-matcher-held prose is emitted before the tool call event")
    func heldTailPrecedesToolCall() {
        // "EN" is a strict prefix of the stop string "END", so the matcher
        // holds it back for disambiguation while the envelope streams in.
        // Pad past the detokenizer's 24-char holdback so everything flushes.
        let events = run(
            pieces: [
                "Removing the temporary directory now, hold on.",
                "EN",
                "<tool_call>\n<function=run_command>\n<parameter=command>\nls\n",
                "</parameter>\n</function>\n</tool_call>",
            ],
            stops: ["END"])

        guard let callIndex = events.firstIndex(of: .toolCall("run_command")) else {
            Issue.record("tool call was not parsed: \(events)")
            return
        }
        let textBeforeCall = events[..<callIndex].compactMap {
            if case .chunk(let t) = $0 { return t } else { return nil }
        }.joined()
        let textAfterCall = events[(callIndex + 1)...].compactMap {
            if case .chunk(let t) = $0 { return t } else { return nil }
        }.joined()

        #expect(
            textBeforeCall == "Removing the temporary directory now, hold on.EN",
            "held stop-prefix text must surface before the tool call")
        #expect(
            textAfterCall.isEmpty,
            "no text may trail the tool call — consumers drop it as post-tool prose")
    }

    @Test("no stop strings: prose still fully precedes the tool call")
    func noStopsBaseline() {
        let events = run(
            pieces: [
                "Removing the temporary directory now, hold on.",
                "<tool_call>\n<function=run_command>\n<parameter=command>\nls\n",
                "</parameter>\n</function>\n</tool_call>",
            ],
            stops: [])

        guard let callIndex = events.firstIndex(of: .toolCall("run_command")) else {
            Issue.record("tool call was not parsed: \(events)")
            return
        }
        let textBeforeCall = events[..<callIndex].compactMap {
            if case .chunk(let t) = $0 { return t } else { return nil }
        }.joined()
        #expect(textBeforeCall == "Removing the temporary directory now, hold on.")
    }
}
