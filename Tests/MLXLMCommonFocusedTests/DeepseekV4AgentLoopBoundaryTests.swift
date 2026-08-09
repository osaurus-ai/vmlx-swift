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

    // MARK: - The emit -> parse -> re-render round trip the boundary depends on

    /// Live DSV4 (2026-08-02, `Prefill 4128/6569`): the turn carrying a
    /// completed `file_write` and its 5311-char artifact published
    /// `all=[193, 2442]` — the history boundary GONE, where every non-tool turn
    /// published `all=[193, 2442, 2591]`. The fetch then fell back to an older
    /// 2592 store and cold-prefilled `remaining=3977`.
    ///
    /// The boundary is derived by re-rendering the transcript and demanding an
    /// exact token prefix, so it survives only if
    ///
    ///     model emission -> parser -> arguments JSON -> re-render
    ///
    /// is byte-exact. This drives that loop with a realistic HTML payload: the
    /// characters a real `file_write` carries — newlines, quotes, angle
    /// brackets, backslashes, unicode — are exactly the ones a JSON round trip
    /// can alter.
    @Test("a written-file tool call survives emit -> parse -> re-render")
    func fileWriteToolCallRoundTripsByteExact() throws {
        // KNOWN DEFECT, deliberately recorded rather than deleted or weakened.
        //
        // The re-render is NOT byte-stable once `rawArgumentsJSON` is lost:
        // the sorted-key dict path emits `content` before `path`, diverging at
        // char 56 from the model's own emission order. Fixing it means keeping
        // the raw member order across a serialized transcript, which changes
        // `ToolCall.Function`'s Codable surface — a public wire-shape change
        // that does not belong in a tests-only change.
        //
        // `withKnownIssue` keeps CI green while the assertion stays live: if
        // the ordering is ever fixed, this FAILS as an unexpected pass and
        // whoever fixed it gets told to delete this wrapper.
        try withKnownIssue("tool-call re-render loses the model's argument order") {
        let payload = """
            <!DOCTYPE html>
            <html lang="en">
            <head><meta charset="utf-8"><title>Minesweeper</title></head>
            <body>
            <div class="cell" data-x="0" data-y="0"></div>
            <script>
            const GRID = 10, MINES = 15;
            let s = "a\\nb";
            if (x < 3 && y > 1) { console.log('hi "there"'); }
            const re = /\\d+/g;
            </script>
            </body>
            </html>
            """
        // The MODEL's emission order, not a normalized one. DSV4 writes the
        // path first and the body second, which is the natural order for
        // `file_write` — and it is NOT alphabetical ("content" sorts before
        // "path"). Building this with `.sortedKeys` would silently pre-agree
        // with a sorted re-render and prove nothing.
        let encodedPayload = String(
            data: try JSONSerialization.data(withJSONObject: [payload], options: []),
            encoding: .utf8)!
            .dropFirst().dropLast()  // strip the array brackets, keep the JSON string
        let arguments = "{\"path\":\"/tmp/minesweeper.html\",\"content\":"
            + encodedPayload + "}"

        // 1. What the encoder puts in history for this call.
        let rendered = DeepseekV4ChatEncoder.renderToolCallInvoke(
            name: "file_write", arguments: arguments)

        // 2. Parse it back the way the runtime parses the model's emission.
        let envelope =
            "<｜DSML｜tool_calls>\n\(rendered)\n</｜DSML｜tool_calls>"
        let parser = DSMLToolCallParser()
        let parsed = parser.parseEOS(envelope, tools: Self.tools())
        try #require(
            parsed.count == 1,
            "parser did not recover the file_write call from its own rendering")

        // 3. Re-render the parsed call — this is what turn N+1 puts in history.
        //
        // `rawArgumentsJSON` carries the exact wire spelling, but it is
        // deliberately excluded from `Codable` ("preserves the public wire
        // shape"), so it does NOT survive a trip through a serialized message
        // history. Model the turn-N+1 path faithfully: rebuild the arguments
        // from the decoded values only, which is what a client that persisted
        // and reloaded the transcript can supply.
        let decoded = parsed[0].function.arguments
        let reRendered = DeepseekV4ChatEncoder.renderToolCallInvoke(
            name: parsed[0].function.name,
            params: decoded.mapValues { $0.anyValue })

        guard rendered == reRendered else {
            let common = zip(rendered, reRendered).prefix { $0 == $1 }.count
            Issue.record(
                """
                tool-call re-render is NOT byte-stable across the parse round trip \
                — every turn after this call loses its history boundary.
                diverges at char \(common)
                  first  : \(String(rendered.dropFirst(common).prefix(160)).debugDescription)
                  second : \(String(reRendered.dropFirst(common).prefix(160)).debugDescription)
                """)
            return
        }

        // And the recovered payload must be the original, not a mangled copy.
        #expect(
            decoded["content"]?.anyValue as? String == payload,
            "file payload did not survive the DSML round trip intact")
        }
    }

    // MARK: - Retroactive history rewrites that cost the whole prefix

    /// The encoder decides `effectiveDrop` from whether ANY message still
    /// carries tools (`encode`, "if any message still carries tools, do NOT
    /// strip"). `renderOpenAIChat` attaches the request's tool schemas to the
    /// system message, so that flag tracks THIS REQUEST's tool list.
    ///
    /// An agent loop that drops the tool list for a final summarizing turn
    /// therefore flips `drop_thinking` on for a transcript that was rendered
    /// with reasoning throughout — rewriting every earlier assistant turn and
    /// invalidating the entire cached prefix at the worst possible moment.
    @Test("dropping the tool list mid-conversation rewrites earlier turns")
    func droppingToolsRewritesHistory() throws {
        let messages: [[String: any Sendable]] = [
            ["role": "system", "content": "You are a coding agent."],
            ["role": "user", "content": "Build me a minesweeper game"],
            Self.assistantToolCall(
                reasoning: "I will write the file and then explain it.",
                calls: [(id: "c1", name: "file_write", arguments: #"{"path":"/tmp/m.html","content":"<html/>"}"#)]),
            ["role": "tool", "tool_call_id": "c1", "content": "{\"ok\":true}"],
            ["role": "assistant", "content": "Done.", "reasoning_content": "wrote it"],
            ["role": "user", "content": "Summarize what you did"],
        ]

        let withTools = try DeepseekV4ChatEncoder.renderOpenAIChat(
            messages: messages, tools: Self.tools(),
            additionalContext: Self.thinkingContext(), addGenerationPrompt: true)
        let withoutTools = try DeepseekV4ChatEncoder.renderOpenAIChat(
            messages: messages, tools: nil,
            additionalContext: Self.thinkingContext(), addGenerationPrompt: true)

        let shared = zip(withTools, withoutTools).prefix { $0 == $1 }.count
        // Characterization: measure how much of the prefix survives the flip.
        // The tool schemas themselves live in the system block, so some early
        // divergence is expected and correct; what matters for the cache is
        // whether ANY meaningful shared prefix remains.
        #expect(
            shared > 0,
            "dropping tools diverges the prompt from byte 0 — no prefix reuse is possible at all")
    }

    /// `reasoning_effort` is emitted ONLY at index 0 in thinking mode
    /// (`encoding_dsv4.py`: `if index == 0 and thinking_mode == "thinking"`).
    /// Changing it between turns therefore shifts byte 0 of the transcript and
    /// invalidates every cached boundary including the stable system prefix.
    @Test("changing reasoning effort shifts the transcript from byte zero")
    func reasoningEffortShiftsFromByteZero() throws {
        let messages: [[String: any Sendable]] = [
            ["role": "system", "content": "You are a coding agent."],
            ["role": "user", "content": "Build me a minesweeper game"],
        ]
        func render(_ effort: String) throws -> String {
            try DeepseekV4ChatEncoder.renderOpenAIChat(
                messages: messages, tools: Self.tools(),
                additionalContext: ["enable_thinking": true, "reasoning_effort": effort],
                addGenerationPrompt: true)
        }
        let low = try render("low")
        let high = try render("high")

        // Every effort rail (including the enforced low preface) injects at
        // position 0, so low vs high diverges immediately after the shared
        // BOS + "Reasoning Effort: " stem (39 characters).
        #expect(low != high, "reasoning effort had no effect on the prompt at all")
        let shared = zip(low, high).prefix { $0 == $1 }.count
        #expect(
            shared < 64,
            """
            expected the effort prompt to land at the very front (shared=\(shared)); \
            if it lands later this test is measuring the wrong thing
            """)
    }

    /// Toggling `enable_thinking` changes BOTH the terminal rail and whether
    /// every historical assistant turn renders its reasoning — a whole-history
    /// rewrite, not a tail change.
    @Test("toggling thinking mode rewrites the whole history")
    func togglingThinkingRewritesHistory() throws {
        let messages: [[String: any Sendable]] = [
            ["role": "system", "content": "You are a coding agent."],
            ["role": "user", "content": "Build me a minesweeper game"],
            ["role": "assistant", "content": "Done.", "reasoning_content": "a long private plan"],
            ["role": "user", "content": "Now explain it"],
        ]
        func render(_ thinking: Bool) throws -> String {
            try DeepseekV4ChatEncoder.renderOpenAIChat(
                messages: messages, tools: Self.tools(),
                additionalContext: ["enable_thinking": thinking],
                addGenerationPrompt: true)
        }
        let thinking = try render(true)
        let chat = try render(false)
        #expect(thinking != chat, "thinking toggle had no effect")

        // The reasoning text must not survive into the chat-mode render.
        #expect(
            !chat.contains("a long private plan"),
            "chat mode leaked reasoning_content into the prompt")
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
