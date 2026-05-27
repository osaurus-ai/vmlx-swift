// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Tests for Chat.Message.toolCalls + Chat.Message.toolCallId and the
// Jinja-renderer dict shape DefaultMessageGenerator produces for
// them. Motivated by osaurus's multi-turn-tool-call bug with MiniMax,
// Llama 3.1/3.2, Qwen 2.5 Instruct, Mistral Large, and every other
// model whose chat template reads `message.tool_calls[i]`.

import Testing

@testable import MLXLMCommon
@testable import MLXVLM

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
}
