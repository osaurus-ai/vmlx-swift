// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import Testing

@testable import MLXLMCommon

/// Official Hunyuan v3 suffixes every protocol marker with `:opensource`
/// (`<think:opensource>`, `<tool_calls:opensource>`, `<arg_key:opensource>`,
/// …) where the preview conversion used bare spellings. These tests pin the
/// parser stack to the official wire format, byte-faithful to the shipped
/// `chat_template.jinja`.
struct Hy3OfficialMarkersTests {

    private func weatherTool() -> [String: any Sendable] {
        [
            "type": "function",
            "function": [
                "name": "get_weather",
                "description": "Get weather",
                "parameters": [
                    "type": "object",
                    "properties": [
                        "city": ["type": "string"],
                        "days": ["type": "integer"],
                    ],
                    "required": ["city"],
                ] as [String: any Sendable],
            ] as [String: any Sendable],
        ]
    }

    private static let officialEnvelope = """
        <tool_calls:opensource>
        <tool_call:opensource>get_weather<tool_sep:opensource>
        <arg_key:opensource>city</arg_key:opensource><arg_value:opensource>Tokyo</arg_value:opensource>
        <arg_key:opensource>days</arg_key:opensource><arg_value:opensource>3</arg_value:opensource>
        </tool_call:opensource>
        </tool_calls:opensource>
        """

    @Test("official :opensource envelope parses through the processor")
    func officialEnvelopeParses() {
        let processor = ToolCallProcessor(format: .hunyuan, tools: [weatherTool()])
        var visible = ""
        if let text = processor.processChunk("Checking the forecast now. " + Self.officialEnvelope) {
            visible += text
        }
        if let tail = processor.processEOS() {
            visible += tail
        }
        #expect(processor.toolCalls.count == 1)
        let call = processor.toolCalls.first
        #expect(call?.function.name == "get_weather")
        #expect(call?.function.arguments["city"] == .string("Tokyo"))
        #expect(call?.function.arguments["days"] == .int(3))
        #expect(visible == "Checking the forecast now. ")
    }

    @Test("official envelope split into small chunks, incl. mid-suffix splits")
    func officialEnvelopeChunked() {
        let processor = ToolCallProcessor(format: .hunyuan, tools: [weatherTool()])
        var visible = ""
        var buffer = Substring(Self.officialEnvelope)
        // 7-char chunks guarantee splits inside `:opensource` suffixes.
        while !buffer.isEmpty {
            let chunk = String(buffer.prefix(7))
            buffer = buffer.dropFirst(7)
            if let text = processor.processChunk(chunk) {
                visible += text
            }
        }
        if let tail = processor.processEOS() {
            visible += tail
        }
        #expect(processor.toolCalls.count == 1)
        #expect(processor.toolCalls.first?.function.arguments["city"] == .string("Tokyo"))
        #expect(visible.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
    }

    @Test("preview bare-marker envelope still parses (back-compat)")
    func previewEnvelopeStillParses() {
        let processor = ToolCallProcessor(format: .hunyuan, tools: [weatherTool()])
        _ = processor.processChunk(
            "<tool_calls>\n<tool_call>get_weather<tool_sep>\n"
                + "<arg_key>city</arg_key><arg_value>Paris</arg_value>\n"
                + "</tool_call>\n</tool_calls>")
        _ = processor.processEOS()
        #expect(processor.toolCalls.count == 1)
        #expect(processor.toolCalls.first?.function.arguments["city"] == .string("Paris"))
    }

    @Test("hy3 reasoning parser uses the :opensource think markers")
    func reasoningMarkers() {
        let parser = ReasoningParser.fromCapabilityName("hy_v3")
        #expect(parser?.startTag == "<think:opensource>")
        #expect(parser?.endTag == "</think:opensource>")
    }

    @Test("no_think prompt tail (closed empty think) starts the parser in content")
    func noThinkTailStartsInContent() {
        // The official template's no_think generation prompt ends with a
        // pre-CLOSED empty think pair; starting in reasoning would route the
        // whole answer into reasoning_content.
        let parser = ReasoningParser.forPrompt(
            stampName: "hy_v3",
            promptTail: "<｜hy_Assistant:opensource｜><think:opensource></think:opensource>")
        #expect(parser?.isInsideReasoning == false)
    }

    @Test("high-effort prompt tail (open think) starts the parser in reasoning")
    func openThinkTailStartsInReasoning() {
        let parser = ReasoningParser.forPrompt(
            stampName: "hy_v3",
            promptTail: "<｜hy_Assistant:opensource｜><think:opensource>")
        #expect(parser?.isInsideReasoning == true)
    }

    @Test("reasoning then content splits at the :opensource close marker")
    func reasoningContentSplit() {
        var parser = ReasoningParser(
            startTag: "<think:opensource>",
            endTag: "</think:opensource>",
            startInReasoning: true)
        var reasoning = ""
        var content = ""
        for segment in parser.feed("planning the answer</think:opensource>The answer is 4.") {
            switch segment {
            case .reasoning(let r): reasoning += r
            case .content(let c): content += c
            }
        }
        for segment in parser.flush() {
            switch segment {
            case .reasoning(let r): reasoning += r
            case .content(let c): content += c
            }
        }
        #expect(reasoning.contains("planning the answer"))
        #expect(content.contains("The answer is 4."))
        #expect(!content.contains("</think"))
    }
}
