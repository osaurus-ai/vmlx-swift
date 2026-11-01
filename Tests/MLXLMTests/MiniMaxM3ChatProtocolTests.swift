// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import Testing

@testable import MLXLMCommon

/// MiniMax-M3 / M3.1 chat protocol: the `<mm:think>` reasoning rail with M3.1's adaptive first token, and the
/// namespace-token tool-call envelope with tag-named arguments.
///
/// M3.1's template has no thinking switch and no generation prefix: the first assistant token is the decision,
/// `<mm:think>` (200059: reasoning follows, closed by `</mm:think>` 200060) or `</mm:think>` directly (no reasoning). Past
/// turns render as `<mm:think>reasoning</mm:think>answer` or `</mm:think>answer`. Tool elements are written behind the
/// separator token `]<]minimax[>[`; the detokenizer keeps special tokens, so it is part of the text stream.
@Suite("MiniMax-M3 chat protocol: adaptive reasoning and tool calls")
struct MiniMaxM3ChatProtocolTests {
    static let ns = MiniMaxM3ToolCallParser.namespaceToken

    /// Feeds `chunks` through the parser and returns the concatenated reasoning and content.
    static func run(_ parser: ReasoningParser, _ chunks: [String]) -> (reasoning: String, content: String) {
        var parser = parser
        var reasoning = "", content = ""
        func take(_ segments: [ReasoningSegment]) {
            for segment in segments {
                switch segment {
                case .reasoning(let text): reasoning += text
                case .content(let text): content += text
                }
            }
        }
        for chunk in chunks { take(parser.feed(chunk)) }
        take(parser.flush())
        return (reasoning, content)
    }

    /// Every split of `text` into two chunks, plus one chunk per character.
    static func splits(_ text: String) -> [[String]] {
        var out = (0 ... text.count).map { cut -> [String] in
            let index = text.index(text.startIndex, offsetBy: cut)
            return [String(text[..<index]), String(text[index...])]
        }
        out.append(text.map { String($0) })
        return out
    }

    static let tool: [String: any Sendable] = [
        "type": "function",
        "function": [
            "name": "run",
            "description": "Run a shell command",
            "parameters": [
                "type": "object",
                "properties": [
                    "cmd": ["type": "string"] as [String: any Sendable],
                    "timeout": ["type": "integer"] as [String: any Sendable],
                ] as [String: any Sendable],
                "required": ["cmd"],
            ] as [String: any Sendable],
        ] as [String: any Sendable],
    ]

    // MARK: - Reasoning

    @Test("minimax_m3 model types resolve to the <mm:think> parser, not the M2 <think> rail")
    func reasoningResolution() throws {
        for modelType in ["minimax_m3", "minimax_m3_vl"] {
            #expect(reasoningStampFromModelType(modelType) == "minimax_m3")
            let resolved = ParserResolution.reasoning(capabilities: nil, modelType: modelType)
            let parser = try #require(resolved.parser)
            #expect(parser.startTag == "<mm:think>" && parser.endTag == "</mm:think>")
            #expect(!parser.isInsideReasoning, "the adaptive generation prompt has no reasoning prefix")
        }
        // M2 keeps its own rail.
        #expect(reasoningStampFromModelType("minimax_m2") == "think_xml")
        #expect(ReasoningParser.fromCapabilityName("think_xml")?.startTag == "<think>")

        // A template that renders the `<mm:think>` envelope names the M3 parser even for an unknown model type, and
        // the factories record it under its own stamp.
        let template = "{{ ']~b]ai\\n' }}{% if m.reasoning_content %}<mm:think>{{ m.reasoning_content }}</mm:think>"
            + "{% else %}</mm:think>{% endif %}{{ m.content }}"
        let fromTemplate = ParserResolution.reasoning(
            capabilities: nil, modelType: "some_future_type", chatTemplate: template)
        #expect(fromTemplate.source == .chatTemplate)
        #expect(fromTemplate.parser?.startTag == "<mm:think>")
        #expect(ParserResolution.templateReasoningStampName(fromTemplate.parser) == "minimax_m3")
    }

    @Test("Think-first turn: <mm:think> opens reasoning, </mm:think> hands over to the answer, at every split")
    func thinkFirst() throws {
        let parser = try #require(ReasoningParser.fromCapabilityName("minimax_m3"))
        let text = "<mm:think>The tests need gen.h; check the build target.</mm:think>Run `make gen` first."
        for chunks in Self.splits(text) {
            let (reasoning, content) = Self.run(parser, chunks)
            #expect(reasoning == "The tests need gen.h; check the build target.", "\(chunks.count) chunks")
            #expect(content == "Run `make gen` first.", "\(chunks.count) chunks")
        }
    }

    @Test("Skip-first turn: a leading </mm:think> is an empty reasoning rail; nothing leaks, at every split")
    func skipFirst() throws {
        let parser = try #require(ReasoningParser.fromCapabilityName("minimax_m3"))
        let text = "</mm:think>You're welcome."
        for chunks in Self.splits(text) {
            let (reasoning, content) = Self.run(parser, chunks)
            #expect(reasoning.isEmpty, "\(chunks): \(reasoning)")
            #expect(content == "You're welcome.", "\(chunks): \(content)")
        }
        // Control: the M2 `<think>` parser the model type used to resolve to leaks the M3 marker into the answer.
        let m2 = try #require(ReasoningParser.fromCapabilityName("think_xml"))
        let leaked = Self.run(m2, [text])
        #expect(leaked.content.contains("</mm:think>") || leaked.reasoning.contains("You're welcome."))
    }

    @Test("Prompt-tail resolution across an adaptive multi-turn conversation")
    func promptTails() throws {
        // Generation prompts: the M3.1 adaptive tail after a think turn and after a skip turn, the first turn, and the
        // M3 disabled-thinking tail (`</mm:think>` already written): every one starts in content.
        let tails = [
            "]~b]user\nWhy does make test fail?[e~[\n]~b]ai\n",
            "]~b]ai\n<mm:think>Read the failure.</mm:think>gen.h is missing.[e~[\n]~b]user\nthanks[e~[\n]~b]ai\n",
            "]~b]ai\n</mm:think>You're welcome.[e~[\n]~b]user\nand now?[e~[\n]~b]ai\n",
            "]~b]ai\n</mm:think>",
        ]
        for tail in tails {
            let parser = try #require(ReasoningParser.forPrompt(stampName: "minimax_m3", promptTail: tail))
            #expect(!parser.isInsideReasoning, "\(tail.debugDescription)")
            #expect(parser.startTag == "<mm:think>")
            // Both decisions parse from that state.
            #expect(Self.run(parser, ["<mm:think>", "plan", "</mm:think>", "answer"]) == ("plan", "answer"))
            #expect(Self.run(parser, ["</mm:think>", "answer"]) == ("", "answer"))
        }
        // A prompt that ends inside an open rail starts in reasoning.
        let open = try #require(ReasoningParser.forPrompt(stampName: "minimax_m3", promptTail: "]~b]ai\n<mm:think>"))
        #expect(open.isInsideReasoning)
        #expect(Self.run(open, ["plan", "</mm:think>", "answer"]) == ("plan", "answer"))
    }

    // MARK: - Tool calls

    @Test("minimax_m3 resolves to its own tool format")
    func toolFormatResolution() {
        #expect(ToolCallFormat.infer(from: "minimax_m3") == .minimaxM3)
        #expect(ToolCallFormat.infer(from: "minimax_m3_vl") == .minimaxM3)
        #expect(ToolCallFormat.infer(from: "minimax_m2") == .minimaxM2)
        #expect(ToolCallFormat.fromCapabilityName("minimax_m3") == .minimaxM3)
        // A family-level `minimax` stamp on an M3 bundle names the M3 envelope; on an M2 bundle, M2's.
        #expect(ToolCallFormat.fromCapabilityName("minimax", modelType: "minimax_m3_vl") == .minimaxM3)
        #expect(ToolCallFormat.fromCapabilityName("minimax", modelType: "minimax_m2") == .minimaxM2)
        #expect(ParserResolution.toolCall(capabilities: nil, modelType: "minimax_m3_vl").format == .minimaxM3)
        #expect(ToolCallFormat.minimaxM3.hasTaggedToolMarkers)
    }

    @Test("Tag-named arguments: scalars by schema or JSON, nested objects, <item> arrays, several invokes")
    func parseValues() throws {
        let ns = Self.ns
        let envelope = """
            \(ns)<tool_call>
            \(ns)<invoke name="run">
            \(ns)<cmd>ls build</cmd>
            \(ns)<timeout>30\(ns)</timeout>
            \(ns)<env>\(ns)<CI>true\(ns)</CI>\(ns)<NAME>m3\(ns)</NAME>\(ns)</env>
            \(ns)<args>\(ns)<item>-l\(ns)</item>\(ns)<item>2\(ns)</item>\(ns)</args>
            \(ns)</invoke>
            \(ns)<invoke name="get_weather">
            \(ns)<location>San Francisco\(ns)</location>
            \(ns)<days>3\(ns)</days>
            \(ns)</invoke>
            </tool_call>
            """
        let calls = MiniMaxM3ToolCallParser().parseEOS(envelope, tools: [Self.tool])
        try #require(calls.count == 2)
        #expect(calls[0].function.name == "run")
        #expect(calls[0].function.arguments["cmd"] == .string("ls build"))
        #expect(calls[0].function.arguments["timeout"] == .int(30))
        #expect(calls[0].function.arguments["env"] == .object(["CI": .bool(true), "NAME": .string("m3")]))
        #expect(calls[0].function.arguments["args"] == .array([.string("-l"), .int(2)]))
        // Undeclared tool: values are JSON when they parse as JSON (vMLX Python contract).
        #expect(calls[1].function.name == "get_weather")
        #expect(calls[1].function.arguments["location"] == .string("San Francisco"))
        #expect(calls[1].function.arguments["days"] == .int(3))

        // A schema string stays a string even when it looks numeric.
        let numeric = MiniMaxM3ToolCallParser().parseEOS(
            "\(ns)<tool_call>\n\(ns)<invoke name=\"run\">\n\(ns)<cmd>007\(ns)</cmd>\n\(ns)</invoke>\n</tool_call>",
            tools: [Self.tool])
        #expect(numeric.first?.function.arguments["cmd"] == .string("007"))

        // The separator truncated by its final `[` before `<tool_call>` (seen in live M3 output).
        let truncated = String(ns.dropLast()) + "<tool_call>\n\(ns)<invoke name=\"file_info\">\n"
            + "\(ns)<path>panel/package.json\(ns)</path>\n\(ns)</invoke>\n</tool_call>"
        let processor = ToolCallProcessor(format: .minimaxM3, tools: [Self.tool])
        let visible = (processor.processChunk(truncated) ?? "") + (processor.processEOS() ?? "")
        #expect(visible.isEmpty, "\(visible)")
        #expect(processor.toolCalls.map(\.function.name) == ["file_info"])
        #expect(processor.toolCalls.first?.function.arguments["path"] == .string("panel/package.json"))
    }

    @Test("A think-first agentic turn streams reasoning, then one tool call; no marker reaches the answer")
    func agenticTurnStream() throws {
        let ns = Self.ns
        // Token-shaped chunks: the special tokens arrive whole.
        let tokens = [
            "<mm:think>", "Run", " the tests", " first.", "</mm:think>", ns, "<tool_call>", "\n", ns, "<invoke",
            " name=\"run\">", "\n", ns, "<cmd>", "make", " test", ns, "</cmd>", "\n", ns, "<timeout>", "60", ns,
            "</timeout>", "\n", ns, "</invoke>", "\n", "</tool_call>",
        ]
        for chunking in [tokens, Array(tokens.joined()).map { String($0) }] {
            var reasoningParser = try #require(
                ReasoningParser.forPrompt(stampName: "minimax_m3", promptTail: "]~b]user\nwhy?[e~[\n]~b]ai\n"))
            let processor = ToolCallProcessor(format: .minimaxM3, tools: [Self.tool])
            var reasoning = "", visible = ""
            for chunk in chunking {
                for segment in reasoningParser.feed(chunk) {
                    switch segment {
                    case .reasoning(let text): reasoning += text
                    case .content(let text): visible += processor.processChunk(text) ?? ""
                    }
                }
            }
            for segment in reasoningParser.flush() {
                if case .content(let text) = segment { visible += processor.processChunk(text) ?? "" }
            }
            visible += processor.processEOS() ?? ""
            #expect(reasoning == "Run the tests first.")
            #expect(visible.isEmpty, "leaked: \(visible.debugDescription)")
            #expect(processor.toolCalls.count == 1)
            #expect(processor.toolCalls.first?.function.name == "run")
            #expect(processor.toolCalls.first?.function.arguments["cmd"] == .string("make test"))
            #expect(processor.toolCalls.first?.function.arguments["timeout"] == .int(60))
        }

        // Control: the M2 format (what minimax_m3 used to resolve to) extracts nothing from the M3 envelope.
        let m2 = ToolCallProcessor(format: .minimaxM2, tools: [Self.tool])
        let m2Visible = (m2.processChunk(tokens.dropFirst(5).joined()) ?? "") + (m2.processEOS() ?? "")
        #expect(m2.toolCalls.isEmpty)
        #expect(m2Visible.contains(ns))
    }

    @Test("Prose with brackets and the word MiniMax stays visible and calls nothing")
    func proseIsNotCaptured() {
        let prose = "Use arr[0] and [docs](x]y). MiniMax and minimax[>] stay as written.]"
        for chunks in Self.splits(prose) {
            let processor = ToolCallProcessor(format: .minimaxM3, tools: [Self.tool])
            var visible = ""
            for chunk in chunks { visible += processor.processChunk(chunk) ?? "" }
            visible += processor.processEOS() ?? ""
            #expect(visible == prose, "\(chunks.count) chunks: \(visible)")
            #expect(processor.toolCalls.isEmpty)
        }
    }
}
