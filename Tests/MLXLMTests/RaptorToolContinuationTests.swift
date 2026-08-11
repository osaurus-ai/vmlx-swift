import Foundation
import MLX
import MLXLMCommon
import Testing
@preconcurrency import VMLXTokenizers

@testable import MLXHuggingFace
@testable import MLXLLM

/// Raptor 1.0 16B (laguna) answers the first turn by calling a capability
/// tool, then produces **nothing** on the continuation turn — three times in a
/// row, so the agent loop's nudge-and-retry is exhausted and the user sees
/// "The model returned empty output after tool execution."
///
/// The app cannot tell "the model emitted EOS immediately" from "the model
/// emitted text that the serving stack dropped". This runs the same
/// conversation straight through the engine — no agent loop, no UI, no
/// history compaction — and prints every channel the stream produces, so the
/// two are distinguishable.
///
/// Enable with `RAPTOR_LIVE=1`. The tool result is the real Osaurus Mail skill
/// document (`~/.osaurus/PluginSpecs/central/plugins/osaurus.mail.json`)
/// wrapped in the envelope `capabilities_load` returns.
@Suite("Raptor tool continuation", .serialized)
struct RaptorToolContinuationTests {

    static let bundle = URL(fileURLWithPath: NSHomeDirectory())
        .appendingPathComponent("models/OsaurusAI/Raptor-1.0-16B-A3B-qat-JANG_4M")

    static var enabled: Bool {
        ProcessInfo.processInfo.environment["RAPTOR_LIVE"] == "1"
            && FileManager.default.fileExists(
                atPath: bundle.appendingPathComponent("config.json").path)
    }

    /// The `capabilities_load` result the app fed back, verbatim in shape:
    /// `{"ok":true,"result":{"text":"## Skill: Osaurus Mail…"}}`.
    static func mailToolResult() throws -> String {
        let path = ProcessInfo.processInfo.environment["RAPTOR_TOOL_RESULT"]
            ?? "/private/tmp/claude-501/-Users-eric-vmlx-swift/cf33ccb8-c141-4823-ac15-d7bb83cca83d/scratchpad/mail_tool_result.json"
        return try String(contentsOf: URL(fileURLWithPath: path), encoding: .utf8)
    }

    static func toolSpec(_ name: String, _ description: String) -> ToolSpec {
        [
            "type": "function",
            "function": [
                "name": name,
                "description": description,
                "parameters": [
                    "type": "object",
                    "properties": [String: any Sendable](),
                    "required": [String](),
                ] as [String: any Sendable],
            ] as [String: any Sendable],
        ]
    }

    static let capabilityTools: [ToolSpec] = [
        toolSpec(
            "capabilities_discover",
            "Search installed capabilities by keyword when the Enabled list does not already name one."),
        toolSpec(
            "capabilities_load",
            "Load capabilities into the current session by ID. IDs come from the Enabled capabilities list or from `capabilities_discover` results — do not invent IDs."),
    ]

    struct Collected {
        var content = ""
        var reasoning = ""
        var toolCalls: [String] = []
        var progress = 0
        var tokens = 0
        var infoLine = ""
    }

    static func collect(_ stream: AsyncStream<Generation>) async -> Collected {
        var out = Collected()
        for await event in stream {
            switch event {
            case .chunk(let text):
                out.content += text
                out.tokens += 1
            case .reasoning(let text):
                out.reasoning += text
            case .toolCall(let call):
                out.toolCalls.append(call.function.name)
            case .info(let info):
                let pp = info.promptTime > 0
                    ? Double(info.promptTokenCount) / info.promptTime : 0
                let tg = info.generateTime > 0
                    ? Double(info.generationTokenCount) / info.generateTime : 0
                out.infoLine =
                    "prompt=\(info.promptTokenCount) gen=\(info.generationTokenCount)"
                    + " stop=\(info.stopReason)"
                    + " pp=\(String(format: "%.1f", pp))/s"
                    + " tg=\(String(format: "%.1f", tg))/s"
            default:
                out.progress += 1
            }
        }
        return out
    }

    static func report(_ label: String, _ c: Collected) {
        print("[raptor] ── \(label) ──")
        print("[raptor] \(c.infoLine)")
        print("[raptor] content chars=\(c.content.count) reasoning chars=\(c.reasoning.count) toolCalls=\(c.toolCalls) other-events=\(c.progress)")
        if !c.reasoning.isEmpty {
            print("[raptor] reasoning: \(c.reasoning.prefix(600))")
        }
        print("[raptor] content: \(c.content.isEmpty ? "<EMPTY>" : String(c.content.prefix(600)))")
    }

    /// Run one prompt through the engine and report every channel.
    static func run(
        label: String, chat: [Chat.Message], tools: [ToolSpec]?, maxTokens: Int = 320
    ) async throws -> Collected {
        let context = try await MLXLMCommon.loadModel(
            from: bundle, using: #huggingFaceTokenizerLoader())
        let input = try await context.processor.prepare(
            input: UserInput(chat: chat, tools: tools))

        let tokens = input.text.tokens.reshaped(-1).asArray(Int.self)
        let rendered = context.tokenizer.decode(tokenIds: tokens, skipSpecialTokens: false)
        print("[raptor] \(label): prompt tokens=\(tokens.count)")
        print("[raptor] \(label): prompt tail ⟦\(rendered.suffix(220))⟧")

        let params = GenerateParameters(maxTokens: maxTokens, temperature: 0)

        // Raw pass first: the serving stack peels reasoning off, swallows tool
        // envelopes and applies stop strings, so "empty" there is ambiguous.
        // Decoding the token ids directly shows exactly what the model emitted.
        var iterator = try TokenIterator(
            input: input, model: context.model, parameters: params)
        var raw: [Int] = []
        for _ in 0 ..< maxTokens {
            guard let id = iterator.next() else { break }
            raw.append(id)
            if id == context.tokenizer.eosTokenId { break }
        }
        let rawText = context.tokenizer.decode(tokenIds: raw, skipSpecialTokens: false)
        print("[raptor] \(label): RAW \(raw.count) tokens ids=\(raw.suffix(8))")
        print("[raptor] \(label): RAW ⟦\(rawText)⟧")

        nonisolated(unsafe) let send = input
        nonisolated(unsafe) let ctx = context
        let stream = try MLXLMCommon.generate(input: send, parameters: params, context: ctx)
        let collected = await collect(stream)
        report(label, collected)
        return collected
    }

    static let userQuestion = "how many unread emails do I have in my mailbox"

    static func conversation(toolResult: String) throws -> [Chat.Message] {
        [
            .user(userQuestion),
            .assistant(
                "",
                toolCalls: [
                    ToolCall(
                        id: "call_0_capabilities_load",
                        function: .init(
                            name: "capabilities_load",
                            arguments: ["ids": ["plugin/osaurus.mail"]]))
                ]),
            .tool(toolResult, toolCallId: "call_0_capabilities_load"),
        ]
    }

    // MARK: - Control: does the model answer at all?

    @Test("control — plain question, no tools", .enabled(if: enabled))
    func plainAnswer() async throws {
        let c = try await Self.run(
            label: "no-tools",
            chat: [.user("Name the capital of France in one short sentence.")],
            tools: nil)
        #expect(!c.content.isEmpty, "the model produced no content on a plain question")
    }

    // MARK: - Control: continuation after a *small* tool result

    @Test("continuation after a small tool result", .enabled(if: enabled))
    func smallToolResult() async throws {
        let c = try await Self.run(
            label: "small-tool-result",
            chat: try Self.conversation(
                toolResult: #"{"ok":true,"result":{"text":"Mail tools loaded: list_mailboxes, list_messages."}}"#),
            tools: Self.capabilityTools)
        #expect(
            !c.content.isEmpty || !c.toolCalls.isEmpty,
            "empty continuation even on a small tool result")
    }

    // MARK: - The reported failure

    @Test("continuation after the real Mail capability load", .enabled(if: enabled))
    func largeToolResult() async throws {
        let payload = try Self.mailToolResult()
        print("[raptor] tool result chars=\(payload.count)")
        let c = try await Self.run(
            label: "mail-capability-load",
            chat: try Self.conversation(toolResult: payload),
            tools: Self.capabilityTools)
        #expect(
            !c.content.isEmpty || !c.toolCalls.isEmpty,
            "the model returned empty output after tool execution — reproduces the report")
    }
}
