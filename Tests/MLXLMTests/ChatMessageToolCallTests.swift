// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Tests for Chat.Message.toolCalls + Chat.Message.toolCallId and the
// Jinja-renderer dict shape DefaultMessageGenerator produces for
// them. Motivated by osaurus's multi-turn-tool-call bug with MiniMax,
// Llama 3.1/3.2, Qwen 2.5 Instruct, Mistral Large, and every other
// model whose chat template reads `message.tool_calls[i]`.

import Foundation
import Testing

@testable import MLXLMCommon
@testable import MLXVLM

private final class CapturingGemma4Tokenizer: MLXLMCommon.Tokenizer, @unchecked Sendable {
    var capturedMessages: [[String: any Sendable]] = []
    var capturedAdditionalContext: [String: any Sendable]?

    func encode(text _: String, addSpecialTokens _: Bool) -> [Int] {
        [1]
    }

    func decode(tokenIds: [Int], skipSpecialTokens _: Bool) -> String {
        tokenIds.map(String.init).joined(separator: " ")
    }

    func convertTokenToId(_: String) -> Int? {
        nil
    }

    func convertIdToToken(_ id: Int) -> String? {
        String(id)
    }

    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools _: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        capturedMessages = messages
        capturedAdditionalContext = additionalContext
        return [1, 2, 3]
    }
}

@Suite("Chat.Message tool-call plumbing")
struct ChatMessageToolCallTests {

    // MARK: - Chat.Message constructors

    @Test("assistant with toolCalls carries structured calls through")
    func assistantWithToolCalls() {
        let call = ToolCall(
            function: .init(
                name: "get_weather",
                arguments: ["location": .string("NYC")]
            )
        )
        let msg = Chat.Message.assistant("", toolCalls: [call])
        #expect(msg.role == .assistant)
        #expect(msg.content == "")
        #expect(msg.toolCalls?.count == 1)
        #expect(msg.toolCalls?.first?.function.name == "get_weather")
    }

    @Test("tool message carries toolCallId")
    func toolMessageCarriesId() {
        let msg = Chat.Message.tool("72°F", toolCallId: "call_abc")
        #expect(msg.role == .tool)
        #expect(msg.content == "72°F")
        #expect(msg.toolCallId == "call_abc")
    }

    @Test("tool message defaults to nil toolCallId")
    func toolMessageDefaultId() {
        let msg = Chat.Message.tool("result")
        #expect(msg.toolCallId == nil)
    }

    // MARK: - Dict emission

    @Test("plain user message emits only role + content")
    func plainUserDict() {
        let msg = Chat.Message.user("hi")
        let dict = defaultMessageDict(for: msg)
        #expect(dict["role"] as? String == "user")
        #expect(dict["content"] as? String == "hi")
        #expect(dict["reasoning_content"] == nil)
        #expect(dict["tool_calls"] == nil)
        #expect(dict["tool_call_id"] == nil)
    }

    @Test("assistant reasoning_content is emitted for thinking templates")
    func assistantReasoningContentDict() {
        let msg = Chat.Message(
            role: .assistant,
            content: "Final answer.",
            reasoningContent: "Prior reasoning."
        )
        let dict = defaultMessageDict(for: msg)
        #expect(dict["role"] as? String == "assistant")
        #expect(dict["content"] as? String == "Final answer.")
        #expect(dict["reasoning_content"] as? String == "Prior reasoning.")
    }

    @Test("assistant with tool call emits both flat and nested views")
    func assistantToolCallDualView() {
        let call = ToolCall(
            function: .init(
                name: "multiply",
                arguments: [
                    "a": .int(3),
                    "b": .int(4),
                ]
            )
        )
        let msg = Chat.Message.assistant("", toolCalls: [call])
        let dict = defaultMessageDict(for: msg)

        guard
            let calls = dict["tool_calls"] as? [[String: any Sendable]],
            let first = calls.first
        else {
            Issue.record("tool_calls missing or wrong shape")
            return
        }

        // Flat view — MiniMax / Llama 3.1 Groq templates.
        #expect(first["name"] as? String == "multiply")
        let flatArgs = first["arguments"] as? [String: any Sendable]
        #expect(flatArgs != nil)
        // Int comes through as Int (anyValue preserves type).
        #expect(flatArgs?["a"] as? Int == 3)
        #expect(flatArgs?["b"] as? Int == 4)

        // Nested view — OpenAI / HuggingFace canonical templates.
        let nested = first["function"] as? [String: any Sendable]
        #expect(nested?["name"] as? String == "multiply")
        let nestedArgs = nested?["arguments"] as? [String: any Sendable]
        #expect(nestedArgs?["a"] as? Int == 3)

        // Metadata fields OpenAI-compatible consumers look for.
        #expect(first["type"] as? String == "function")
        #expect((first["id"] as? String)?.hasPrefix("call_0_") == true)
    }

    @Test("tool reply emits tool_call_id")
    func toolReplyIdInDict() {
        let msg = Chat.Message.tool("72°F", toolCallId: "call_abc")
        let dict = defaultMessageDict(for: msg)
        #expect(dict["role"] as? String == "tool")
        #expect(dict["content"] as? String == "72°F")
        #expect(dict["tool_call_id"] as? String == "call_abc")
    }

    @Test("generator names tool replies from prior assistant tool call id")
    func generatorNamesToolRepliesFromPriorAssistantToolCallID() {
        let call = ToolCall(
            id: "call_line",
            function: .init(
                name: "line_count",
                arguments: ["text": .string("red\ngreen\nblue")]
            )
        )
        let messages: [Chat.Message] = [
            .assistant("", toolCalls: [call]),
            .tool("{\"lines\":3}", toolCallId: "call_line"),
        ]

        let rendered = DefaultMessageGenerator().generate(messages: messages)
        #expect(rendered.count == 2)
        #expect(rendered[1]["role"] as? String == "tool")
        #expect(rendered[1]["tool_call_id"] as? String == "call_line")
        #expect(rendered[1]["name"] as? String == "line_count")
    }

    @Test("Qwen2VL generator names tool replies while preserving content arrays")
    func qwen2VLGeneratorNamesToolRepliesFromPriorAssistantToolCallID() {
        let call = ToolCall(
            id: "call_line",
            function: .init(
                name: "line_count",
                arguments: ["text": .string("one\ntwo")]
            )
        )
        let messages: [Chat.Message] = [
            .assistant("", toolCalls: [call]),
            .tool("{\"lines\":2}", toolCallId: "call_line"),
        ]

        let rendered = Qwen2VLMessageGenerator().generate(messages: messages)
        #expect(rendered.count == 2)
        #expect(rendered[1]["role"] as? String == "tool")
        #expect(rendered[1]["tool_call_id"] as? String == "call_line")
        #expect(rendered[1]["name"] as? String == "line_count")
        #expect(rendered[1]["content"] as? [[String: String]] == [["type": "text", "text": "{\"lines\":2}"]])
    }

    @Test("tool reply without id omits tool_call_id field")
    func toolReplyNoIdOmitsKey() {
        let msg = Chat.Message.tool("legacy result")
        let dict = defaultMessageDict(for: msg)
        #expect(dict["tool_call_id"] == nil)
    }

    @Test("multiple tool calls all get emitted with distinct ids")
    func multipleToolCalls() {
        let calls = [
            ToolCall(function: .init(
                name: "get_weather",
                arguments: ["city": .string("NYC")]
            )),
            ToolCall(function: .init(
                name: "get_time",
                arguments: ["tz": .string("America/New_York")]
            )),
        ]
        let msg = Chat.Message.assistant("", toolCalls: calls)
        let dict = defaultMessageDict(for: msg)

        let emitted = dict["tool_calls"] as? [[String: any Sendable]]
        #expect(emitted?.count == 2)
        let ids = emitted?.compactMap { $0["id"] as? String } ?? []
        #expect(ids.count == 2)
        #expect(Set(ids).count == 2, "ids must be distinct per call")
        #expect(emitted?[0]["name"] as? String == "get_weather")
        #expect(emitted?[1]["name"] as? String == "get_time")
    }

    // MARK: - Generator integration

    @Test("DefaultMessageGenerator passes tool_calls through")
    func defaultGeneratorTransit() {
        let call = ToolCall(function: .init(
            name: "search", arguments: ["q": .string("swift")]))
        let msg = Chat.Message.assistant("", toolCalls: [call])
        let gen = DefaultMessageGenerator()
        let dict = gen.generate(message: msg)
        #expect(dict["tool_calls"] != nil)
    }

    @Test("NoSystemMessageGenerator drops system but preserves tool_calls")
    func noSystemGeneratorPreservesToolCalls() {
        let call = ToolCall(function: .init(
            name: "f", arguments: [:]))
        let messages: [Chat.Message] = [
            .system("ignored"),
            .assistant("", toolCalls: [call]),
        ]
        let out = NoSystemMessageGenerator().generate(messages: messages)
        #expect(out.count == 1)
        #expect(out.first?["tool_calls"] != nil)
    }

    @Test("Qwen2VL generator preserves tool history while rendering media content arrays")
    func qwen2VLGeneratorPreservesToolHistoryMetadata() {
        let call = ToolCall(function: .init(
            name: "line_count",
            arguments: ["text": .string("one\ntwo")]
        ))

        let assistant = Qwen2VLMessageGenerator().generate(
            message: .assistant("", toolCalls: [call])
        )
        #expect(assistant["role"] as? String == "assistant")
        #expect(assistant["tool_calls"] != nil)
        #expect(assistant["content"] as? [[String: String]] == [["type": "text", "text": ""]])

        let tool = Qwen2VLMessageGenerator().generate(
            message: .tool("2", toolCallId: "call_abc")
        )
        #expect(tool["role"] as? String == "tool")
        #expect(tool["tool_call_id"] as? String == "call_abc")
        #expect(tool["content"] as? [[String: String]] == [["type": "text", "text": "2"]])
    }

    @Test("Gemma4 processor preserves text-only system content as a scalar string")
    func gemma4ProcessorPreservesTextOnlySystemContent() async throws {
        let mlxTestLock = lockSerializedMLXTest()
        _ = mlxTestLock
        let tokenizer = CapturingGemma4Tokenizer()
        let processor = Gemma4Processor(
            Self.gemma4ProcessorConfiguration(),
            tokenizer: tokenizer
        )
        let revision = "REVISION-B-CURRENT-SETTINGS-MUST-WIN"

        _ = try await processor.prepare(input: UserInput(chat: [
            .system(revision),
            .user("Reply with exactly CURRENT SETTINGS RESTORED."),
        ]))

        #expect(tokenizer.capturedMessages.count == 2)
        #expect(tokenizer.capturedMessages[0]["role"] as? String == "system")
        #expect(tokenizer.capturedMessages[0]["content"] as? String == revision)
        #expect(tokenizer.capturedMessages[1]["role"] as? String == "user")
        #expect(
            tokenizer.capturedMessages[1]["content"] as? String
                == "Reply with exactly CURRENT SETTINGS RESTORED."
        )
    }

    @Test("Gemma4 required tool choice compacts closed prior tool protocol before latest user")
    func gemma4RequiredToolChoiceCompactsClosedToolHistory() async throws {
        let call = ToolCall(
            id: "call_lines",
            function: .init(
                name: "line_count",
                arguments: ["text": .string("red\ngreen\nblue")]
            )
        )

        let messages = Qwen2VLMessageGenerator().generate(from: UserInput(
            chat: [
                .user("Use line_count on this exact text: red\ngreen\nblue"),
                .assistant("", toolCalls: [call]),
                .tool(#"{"lines":3}"#, toolCallId: "call_lines"),
                .assistant("Three lines were counted."),
                .user("Now use line_count on this exact text: one\ntwo"),
            ],
            tools: [Self.lineCountToolSpec()],
            additionalContext: ["tool_choice": "required"]
        ))
        let compacted = Gemma4Processor.compactCompletedToolHistoryForRequiredChoice(messages)

        let priorMessages = Array(compacted.dropLast())
        #expect(priorMessages.allSatisfy { $0["tool_calls"] == nil })
        #expect(!priorMessages.contains { $0["role"] as? String == "tool" })
        #expect(!compacted.contains {
            Self.contentText($0["content"]) == "Use line_count on this exact text: red\ngreen\nblue"
        })
        #expect(compacted.contains {
            Self.contentText($0["content"]) == "Three lines were counted."
        })
        #expect(!compacted.contains {
            Self.contentText($0["content"]) == "How many lines were counted?"
        })
        #expect(Self.contentText(compacted.last?["content"]) == "Now use line_count on this exact text: one\ntwo")
    }

    @Test("Gemma4 required tool choice summarizes closed tool result when no later answer exists")
    func gemma4RequiredToolChoiceSummarizesUnansweredToolHistory() async throws {
        let call = ToolCall(
            id: "call_lines",
            function: .init(
                name: "line_count",
                arguments: ["text": .string("red\ngreen\nblue")]
            )
        )

        let messages = Qwen2VLMessageGenerator().generate(from: UserInput(
            chat: [
                .user("Use line_count on this exact text: red\ngreen\nblue"),
                .assistant("", toolCalls: [call]),
                .tool(#"{"lines":3}"#, toolCallId: "call_lines"),
                .user("Now use line_count on this exact text: one\ntwo"),
            ],
            tools: [Self.lineCountToolSpec()],
            additionalContext: ["tool_choice": "required"]
        ))
        let compacted = Gemma4Processor.compactCompletedToolHistoryForRequiredChoice(messages)

        let priorMessages = Array(compacted.dropLast())
        #expect(priorMessages.allSatisfy { $0["tool_calls"] == nil })
        #expect(!priorMessages.contains { $0["role"] as? String == "tool" })
        #expect(!compacted.contains {
            Self.contentText($0["content"]) == "Use line_count on this exact text: red\ngreen\nblue"
        })
        #expect(compacted.contains {
            Self.contentText($0["content"]) == #"Tool line_count returned {"lines":3}."#
        })
    }

    private static func gemma4ProcessorConfiguration() -> Gemma4ProcessorConfiguration {
        try! JSONDecoder().decode(Gemma4ProcessorConfiguration.self, from: Data(#"""
        {
          "processor_class": "Gemma4Processor",
          "patch_size": 16,
          "max_soft_tokens": 280,
          "pooling_kernel_size": 3,
          "image_seq_length": 280,
          "audio_seq_length": 750
        }
        """#.utf8))
    }

    private static func lineCountToolSpec() -> ToolSpec {
        [
            "type": "function",
            "function": [
                "name": "line_count",
                "parameters": [
                    "type": "object",
                    "properties": [
                        "text": ["type": "string"] as [String: any Sendable],
                    ] as [String: any Sendable],
                    "required": ["text"],
                ] as [String: any Sendable],
            ] as [String: any Sendable],
        ]
    }

    private static func contentText(_ content: Any?) -> String {
        if let string = content as? String {
            return string
        }
        if let parts = content as? [[String: String]] {
            return parts.compactMap { $0["text"] }.joined(separator: "\n")
        }
        if let parts = content as? [[String: any Sendable]] {
            return parts.compactMap { part in
                guard part["type"] as? String == "text" else { return nil }
                return part["text"] as? String
            }.joined(separator: "\n")
        }
        return ""
    }
}
