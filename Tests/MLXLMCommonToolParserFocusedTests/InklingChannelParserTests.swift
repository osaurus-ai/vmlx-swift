// Copyright © 2026 Jinho Jang. All rights reserved.
//
// Inkling-2 (`inkling2`, `inkling2_loop`) writes each turn as channel messages, not think tags:
//
//   <|message_model|><|content_thinking|>REASONING<|end_message|>
//   <|message_model|><|content_text|>ANSWER<|end_message|>
//   <|message_model|>NAME<|content_invoke_tool_json|>{"name":"NAME","args":{…}}<|end_message|>
//   <|content_model_end_sampling|>
//
// These tests pin the reasoning / content split, the tool-call extraction through the real
// stream router, and that no protocol marker reaches visible text, for whole-turn and
// one-character-at-a-time streaming.

import Foundation
import Testing

@testable import MLXLMCommon

@Suite("Inkling channel reasoning + tool calls")
struct InklingChannelParserTests {
    static let thinking =
        "<|message_model|><|content_thinking|>The user wants Paris weather. Call the tool.<|end_message|>"
    static let text =
        "<|message_model|><|content_text|>It is 18 °C and clear in Paris.<|end_message|>"
    static let toolCall =
        "<|message_model|>get_weather<|content_invoke_tool_json|>{\"name\":\"get_weather\",\"args\":{\"city\":\"Paris\",\"unit\":\"c\"}}<|end_message|>"
    static let eos = "<|content_model_end_sampling|>"
    static let markers = [
        "<|message_model|>", "<|content_thinking|>", "<|content_text|>", "<|end_message|>",
        "<|content_invoke_tool_json|>", "<|content_model_end_sampling|>",
    ]

    /// Run `output` through the reasoning parser + tool processor exactly as the generation loop
    /// routes it, in chunks of `chunk` characters.
    static func route(_ output: String, chunk: Int) -> (
        reasoning: String, content: String, calls: [ToolCall]
    ) {
        var parser = ReasoningParser.fromCapabilityName("inkling")!
        let processor = ToolCallProcessor(
            format: .inkling, tools: [["type": "function", "function": ["name": "get_weather"]]])
        var reasoning = ""
        var content = ""
        var calls: [ToolCall] = []
        func absorb(_ events: [Generation]) {
            for e in events {
                switch e {
                case .chunk(let c): content += c
                case .reasoning(let r): reasoning += r
                case .toolCall(let c): calls.append(c)
                default: break
                }
            }
        }
        func feedSegments(_ segments: [ReasoningSegment]) {
            for seg in segments {
                switch seg {
                case .content(let c):
                    absorb(routeGenerationText(c, channel: .content, through: processor))
                case .reasoning(let r):
                    absorb(routeGenerationText(r, channel: .reasoning, through: processor))
                }
            }
        }
        var i = output.startIndex
        while i < output.endIndex {
            let j = output.index(i, offsetBy: chunk, limitedBy: output.endIndex) ?? output.endIndex
            feedSegments(parser.feed(String(output[i ..< j])))
            i = j
        }
        feedSegments(parser.flush())
        absorb(flushGenerationText(channel: .content, through: processor))
        return (reasoning, content, calls)
    }

    @Test("model_type selects the Inkling parsers")
    func selection() {
        #expect(ToolCallFormat.infer(from: "inkling2_loop") == .inkling)
        #expect(ToolCallFormat.infer(from: "inkling2") == .inkling)
        #expect(reasoningStampFromModelType("inkling2_loop") == "inkling")
        #expect(ReasoningParser.fromCapabilityName("inkling") != nil)
        #expect(ToolCallFormat.inkling.parsesToolCallsFromReasoningChannel == false)
    }

    @Test("thinking then text", arguments: [1, 3, 7, 1000])
    func thinkingThenText(chunk: Int) {
        let r = Self.route(Self.thinking + Self.text + Self.eos, chunk: chunk)
        #expect(r.reasoning == "The user wants Paris weather. Call the tool.")
        #expect(r.content == "It is 18 °C and clear in Paris.")
        #expect(r.calls.isEmpty)
        for m in Self.markers { #expect(!r.reasoning.contains(m) && !r.content.contains(m)) }
    }

    @Test("thinking then a tool call", arguments: [1, 2, 5, 1000])
    func toolCallTurn(chunk: Int) {
        let r = Self.route(Self.thinking + Self.toolCall + Self.eos, chunk: chunk)
        #expect(r.reasoning == "The user wants Paris weather. Call the tool.")
        #expect(r.calls.count == 1)
        #expect(r.calls.first?.function.name == "get_weather")
        #expect(r.calls.first?.function.arguments["city"] == .string("Paris"))
        #expect(r.calls.first?.function.arguments["unit"] == .string("c"))
        // The header (tool name) and every marker are protocol, never visible text.
        #expect(!r.content.contains("get_weather"))
        for m in Self.markers { #expect(!r.content.contains(m)) }
    }

    @Test("two tool calls in one turn", arguments: [1, 1000])
    func twoCalls(chunk: Int) {
        let second =
            "<|message_model|>get_weather<|content_invoke_tool_json|>{\"name\":\"get_weather\",\"args\":{\"city\":\"Oslo\"}}<|end_message|>"
        let r = Self.route(Self.thinking + Self.toolCall + second + Self.eos, chunk: chunk)
        #expect(r.calls.map(\.function.name) == ["get_weather", "get_weather"])
        #expect(r.calls.last?.function.arguments["city"] == .string("Oslo"))
        #expect(r.content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
    }

    @Test("reasoning-only turn stays on the reasoning rail")
    func reasoningOnly() {
        let r = Self.route(Self.thinking + Self.eos, chunk: 4)
        #expect(r.reasoning == "The user wants Paris weather. Call the tool.")
        #expect(r.content.isEmpty)
    }

    @Test("tool-like text inside thinking is not executed")
    func toolInThinking() {
        let turn =
            "<|message_model|><|content_thinking|>I could emit <|content_invoke_tool_json|>{\"name\":\"x\",\"args\":{}} here.<|end_message|>"
            + Self.text + Self.eos
        let r = Self.route(turn, chunk: 3)
        #expect(r.calls.isEmpty)
        #expect(r.content == "It is 18 °C and clear in Paris.")
    }

    @Test("prose containing '<|' that is not a marker survives")
    func literalAngle() {
        let turn =
            "<|message_model|><|content_text|>Use a <| b in Haskell.<|end_message|>" + Self.eos
        let r = Self.route(turn, chunk: 1)
        #expect(r.content == "Use a <| b in Haskell.")
    }

    @Test("tool parser accepts arguments / parameters spellings and a JSON-string args")
    func argumentSpellings() {
        let p = InklingToolCallParser()
        let a = p.parse(
            content:
                "<|content_invoke_tool_json|>{\"name\":\"f\",\"arguments\":{\"k\":1}}<|end_message|>",
            tools: nil)
        #expect(a?.function.name == "f")
        #expect(a?.function.arguments["k"] == .int(1))
        let b = p.parse(content: "{\"name\":\"f\",\"args\":\"{\\\"k\\\":\\\"v\\\"}\"}", tools: nil)
        #expect(b?.function.arguments["k"] == .string("v"))
        #expect(p.parse(content: "{\"args\":{}}", tools: nil) == nil)
    }
}
