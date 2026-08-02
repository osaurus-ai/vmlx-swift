// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Prefix reuse across a DSV4 AGENT LOOP, driven through the real encoder.
//
// The existing DSV4 boundary tests either exercise `canonicalChatCacheBoundaries`
// against hand-written mock tokenizers (which encode the boundary logic's own
// assumptions) or check the encoder's rail contract on a single transcript.
// Neither runs a growing multi-round tool loop through the real encoder and asks
// the question the cache actually asks:
//
//   does round N's published boundary still describe round N+1's prompt?
//
// `DeepseekV4RenderTokenizer` closes that gap: it renders with the production
// `renderOpenAIChat` and tokenizes one Unicode scalar per token, which is
// prefix-preserving by construction (text prefix <=> token prefix). Real BPE can
// merge across a split point, but that can only LOSE a boundary — never invent
// one — and `exactPrefixBoundary` re-validates by token equality regardless.

import Foundation
import MLXLMCommon
import Testing

@Suite("DSV4 agent-loop prefix reuse")
struct DeepseekV4AgentLoopBoundaryTests {

    /// Real encoder + prefix-preserving tokenization, so a boundary computed
    /// here means the same thing it means in the engine.
    private struct DeepseekV4RenderTokenizer: GenerationPromptControllableTokenizer {
        var bosToken: String? { nil }
        var eosToken: String? { nil }
        var unknownToken: String? { nil }

        func encode(text: String, addSpecialTokens: Bool) -> [Int] { scalars(text) }
        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
            String(String.UnicodeScalarView(tokenIds.compactMap(Unicode.Scalar.init)))
        }
        func convertTokenToId(_ token: String) -> Int? { nil }
        func convertIdToToken(_ id: Int) -> String? { nil }

        private func scalars(_ text: String) -> [Int] {
            text.unicodeScalars.map { Int($0.value) }
        }

        func applyChatTemplate(
            messages: [[String: any Sendable]],
            tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] {
            try applyChatTemplate(
                messages: messages, tools: tools, additionalContext: additionalContext,
                addGenerationPrompt: true)
        }

        func applyChatTemplate(
            messages: [[String: any Sendable]],
            tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?,
            addGenerationPrompt: Bool
        ) throws -> [Int] {
            scalars(
                try DeepseekV4ChatEncoder.renderOpenAIChat(
                    messages: messages,
                    tools: tools,
                    additionalContext: additionalContext,
                    addGenerationPrompt: addGenerationPrompt))
        }
    }

    private static func tools() -> [[String: any Sendable]] {
        [
            [
                "type": "function",
                "function": [
                    "name": "file_write",
                    "description": "Write a file",
                    "parameters": [
                        "type": "object",
                        "properties": [
                            "path": ["type": "string"] as [String: any Sendable],
                            "content": ["type": "string"] as [String: any Sendable],
                        ] as [String: any Sendable],
                        "required": ["path", "content"],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ] as [String: any Sendable],
            [
                "type": "function",
                "function": [
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": [
                        "type": "object",
                        "properties": [
                            "path": ["type": "string"] as [String: any Sendable]
                        ] as [String: any Sendable],
                        "required": ["path"],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ] as [String: any Sendable],
        ]
    }

    /// DSV4 renders reasoning ONLY in thinking mode, and `renderOpenAIChat`
    /// defaults to `.chat` unless `enable_thinking` (or a reasoning effort) is
    /// present. A boundary test that omits it silently exercises a transcript
    /// with no `<think>` rails at all — which is not the agent-loop shape.
    private static func thinkingContext(
        toolChoice: String? = nil
    ) -> [String: any Sendable] {
        var context: [String: any Sendable] = ["enable_thinking": true]
        if let toolChoice { context["tool_choice"] = toolChoice }
        return context
    }

    private static func assistantToolCall(
        reasoning: String, calls: [(id: String, name: String, arguments: String)]
    ) -> [String: any Sendable] {
        [
            "role": "assistant",
            "content": "",
            "reasoning_content": reasoning,
            "tool_calls": calls.map { call in
                [
                    "id": call.id,
                    "type": "function",
                    "function": [
                        "name": call.name,
                        "arguments": call.arguments,
                    ] as [String: any Sendable],
                ] as [String: any Sendable]
            } as [any Sendable],
        ]
    }

    /// The question the cache asks. Publish round N's boundaries, then check the
    /// LARGEST published boundary still describes round N+1's prompt token for
    /// token — that is exactly what a disk-L2 probe does before reusing a block.
    private func boundaries(
        _ messages: [[String: any Sendable]]
    ) throws -> (all: [Int], stable: [Int], prompt: [Int]) {
        let tokenizer = DeepseekV4RenderTokenizer()
        let prompt = try tokenizer.applyChatTemplate(
            messages: messages, tools: Self.tools(),
            additionalContext: Self.thinkingContext(), addGenerationPrompt: true)
        let derived = canonicalChatCacheBoundaries(
            tokenizer: tokenizer,
            messages: messages,
            tools: Self.tools(),
            additionalContext: Self.thinkingContext(),
            promptTokens: prompt)
        return (derived.all, derived.stable, prompt)
    }

    // MARK: - Round 1: single tool call, result comes back

    @Test("a single-tool round publishes a history boundary")
    func singleToolRoundPublishesHistoryBoundary() throws {
        let base: [[String: any Sendable]] = [
            ["role": "system", "content": "You are a coding agent."],
            ["role": "user", "content": "Build me a minesweeper game"],
            Self.assistantToolCall(
                reasoning: "write the file",
                calls: [(id: "call_1", name: "file_write", arguments: #"{"path":"/tmp/m.html","content":"<html/>"}"#)]),
            ["role": "tool", "tool_call_id": "call_1", "content": "{\"ok\":true}"],
        ]
        let round = try boundaries(base)
        #expect(
            round.all.count > round.stable.count,
            "single-tool round published no history boundary: all=\(round.all) stable=\(round.stable)")
    }

    // MARK: - Round 2: PARALLEL tool calls, two results come back

    @Test("a parallel-tool round publishes a history boundary")
    func parallelToolRoundPublishesHistoryBoundary() throws {
        // The agent-loop shape the single-drop continuation fallback cannot
        // reach: one assistant turn issues TWO calls, so TWO tool messages
        // arrive. `mergeToolMessages` folds both into ONE user message, so
        // dropping a single trailing message leaves a half-populated user turn
        // whose render is not a prefix of anything.
        let messages: [[String: any Sendable]] = [
            ["role": "system", "content": "You are a coding agent."],
            ["role": "user", "content": "Build me a minesweeper game"],
            Self.assistantToolCall(
                reasoning: "read both files first",
                calls: [
                    (id: "call_1", name: "read_file", arguments: #"{"path":"/tmp/a.html"}"#),
                    (id: "call_2", name: "read_file", arguments: #"{"path":"/tmp/b.html"}"#),
                ]),
            ["role": "tool", "tool_call_id": "call_1", "content": "{\"body\":\"A\"}"],
            ["role": "tool", "tool_call_id": "call_2", "content": "{\"body\":\"B\"}"],
        ]
        let round = try boundaries(messages)
        #expect(
            round.all.count > round.stable.count,
            "parallel-tool round published no history boundary: all=\(round.all) stable=\(round.stable)")
    }

    // MARK: - The reuse question across consecutive rounds

    @Test("round N's boundary still describes round N+1's prompt")
    func boundarySurvivesIntoTheNextRound() throws {
        var messages: [[String: any Sendable]] = [
            ["role": "system", "content": "You are a coding agent."],
            ["role": "user", "content": "Build me a minesweeper game"],
        ]

        // Three agent rounds, alternating single and parallel calls — the mix a
        // real loop produces.
        let rounds: [[(id: String, name: String, arguments: String)]] = [
            [(id: "c1", name: "file_write", arguments: #"{"path":"/tmp/m.html","content":"<html/>"}"#)],
            [
                (id: "c2", name: "read_file", arguments: #"{"path":"/tmp/a.html"}"#),
                (id: "c3", name: "read_file", arguments: #"{"path":"/tmp/b.html"}"#),
            ],
            [(id: "c4", name: "file_write", arguments: #"{"path":"/tmp/n.html","content":"<b/>"}"#)],
        ]

        var previous: (all: [Int], prompt: [Int])?
        for (index, calls) in rounds.enumerated() {
            messages.append(
                Self.assistantToolCall(reasoning: "step \(index)", calls: calls))
            for call in calls {
                messages.append([
                    "role": "tool", "tool_call_id": call.id,
                    "content": "{\"ok\":true,\"step\":\(index)}",
                ])
            }

            let round = try boundaries(messages)

            if let previous, let reusable = previous.all.max() {
                // Round N stored a block AT this boundary. Round N+1 can only
                // reuse it if its own prompt still starts with those tokens.
                #expect(
                    reusable <= round.prompt.count
                        && round.prompt.prefix(reusable)
                            .elementsEqual(previous.prompt.prefix(reusable)),
                    """
                    round \(index): the block stored at boundary \(reusable) is no longer \
                    a prefix — every round re-prefills the whole transcript.
                    """)
            }

            #expect(
                round.all.max() ?? 0 > (round.stable.max() ?? 0),
                "round \(index) published no history boundary: all=\(round.all) stable=\(round.stable)")

            previous = (round.all, round.prompt)
        }
    }

    // MARK: - The store contract: prompt + generated

    /// `BatchEngine` stores the post-answer block at `promptTokens +
    /// slot.generatedTokenIds` — NOT at a published boundary. That block is
    /// reusable next round only if the next prompt literally starts with those
    /// tokens, which requires the encoder's RE-RENDER of the assistant turn to
    /// reproduce the model's raw emission byte for byte.
    ///
    /// This derives the required emission from the encoder itself: the delta
    /// between the round's prompt (rail included) and the same transcript plus
    /// the assistant turn. Anything the model emits that differs from this
    /// delta silently costs the whole post-answer entry.
    private func requiredEmission(
        history: [[String: any Sendable]],
        assistant: [String: any Sendable]
    ) throws -> String {
        let prompt = try DeepseekV4ChatEncoder.renderOpenAIChat(
            messages: history, tools: Self.tools(),
            additionalContext: Self.thinkingContext(), addGenerationPrompt: true)
        let withAssistant = try DeepseekV4ChatEncoder.renderOpenAIChat(
            messages: history + [assistant], tools: Self.tools(),
            additionalContext: Self.thinkingContext(), addGenerationPrompt: false)
        #expect(
            withAssistant.hasPrefix(prompt),
            "the round's prompt is not a prefix of the same transcript plus its own answer")
        return String(withAssistant.dropFirst(prompt.count))
    }

    @Test("the post-answer store block is reusable by the next round")
    func postAnswerStoreBlockIsReusable() throws {
        let history: [[String: any Sendable]] = [
            ["role": "system", "content": "You are a coding agent."],
            ["role": "user", "content": "Build me a minesweeper game"],
        ]
        let assistant = Self.assistantToolCall(
            reasoning: "I will write the file.",
            calls: [(id: "c1", name: "file_write", arguments: #"{"path":"/tmp/m.html","content":"<html/>"}"#)])

        // What the model must emit, character for character, for the stored
        // `prompt + generated` block to survive into the next round.
        let emission = try requiredEmission(history: history, assistant: assistant)

        // The tool-call envelope is preceded by a BLANK LINE in both the
        // official Python encoder (`tc_content += '\n\n' + ...`) and this Swift
        // port. A model that emits the envelope without that blank line
        // produces a stored block that can never be reused — the divergence
        // lands mid-transcript, exactly the "stored N, re-warm asks N+2, MISS"
        // shape seen live. Pinned so the requirement is explicit and testable
        // against a real capture.
        #expect(
            emission.contains("\n\n<｜DSML｜tool_calls>"),
            "tool-call envelope is not preceded by the encoder's blank line: \(emission.debugDescription)")

        // And the next round, which carries that assistant turn plus its tool
        // result, must still start with prompt + emission.
        let next = history + [
            assistant,
            ["role": "tool", "tool_call_id": "c1", "content": "{\"ok\":true}"],
        ]
        let prompt = try DeepseekV4ChatEncoder.renderOpenAIChat(
            messages: history, tools: Self.tools(),
            additionalContext: Self.thinkingContext(), addGenerationPrompt: true)
        let nextPrompt = try DeepseekV4ChatEncoder.renderOpenAIChat(
            messages: next, tools: Self.tools(),
            additionalContext: Self.thinkingContext(), addGenerationPrompt: true)
        #expect(
            nextPrompt.hasPrefix(prompt + emission),
            "post-answer block (prompt + generated) is not a prefix of the next round's prompt")
    }

    // MARK: - tool_choice=required injects a message that later rounds drop

    @Test("a required-tool reminder does not strand the previous round's prefix")
    func requiredToolReminderKeepsPrefix() throws {
        // `encode` synthesizes a `latest_reminder` message after the last user
        // turn whenever `tool_choice=required`. It is never persisted into the
        // caller's transcript, so the NEXT round re-synthesizes it after the
        // NEW last user turn — and the position it occupied last round is now
        // occupied by the assistant turn instead.
        //
        // If that is true, every `tool_choice=required` round stores a block
        // that the following round can never reuse.
        let history: [[String: any Sendable]] = [
            ["role": "system", "content": "You are a coding agent."],
            ["role": "user", "content": "Build me a minesweeper game"],
        ]
        let assistant = Self.assistantToolCall(
            reasoning: "write it",
            calls: [(id: "c1", name: "file_write", arguments: #"{"path":"/tmp/m.html","content":"<html/>"}"#)])
        let next = history + [
            assistant,
            ["role": "tool", "tool_call_id": "c1", "content": "{\"ok\":true}"],
        ]

        let promptRequired = try DeepseekV4ChatEncoder.renderOpenAIChat(
            messages: history, tools: Self.tools(),
            additionalContext: Self.thinkingContext(toolChoice: "required"),
            addGenerationPrompt: true)
        let nextRequired = try DeepseekV4ChatEncoder.renderOpenAIChat(
            messages: next, tools: Self.tools(),
            additionalContext: Self.thinkingContext(toolChoice: "required"),
            addGenerationPrompt: true)

        // The reusable span is whatever the two rounds still share.
        let shared = zip(promptRequired, nextRequired).prefix { $0 == $1 }.count
        let plain = try DeepseekV4ChatEncoder.renderOpenAIChat(
            messages: history, tools: Self.tools(),
            additionalContext: Self.thinkingContext(), addGenerationPrompt: false)
        #expect(
            shared >= plain.count,
            """
            tool_choice=required strands the prefix at char \(shared), before the \
            history render (\(plain.count) chars): the injected reminder moves \
            between rounds, so no round can reuse the previous one.
            """)
    }
}
