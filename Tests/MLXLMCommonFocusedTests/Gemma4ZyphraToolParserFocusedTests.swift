// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Testing
@testable import MLXLMCommon

@Suite("Gemma4 Zyphra tool parser focused contracts")
struct Gemma4ZyphraToolParserFocusedTests {
    @Test("Gemma4 parser accepts live Zyphra multiline tool-call envelope")
    func gemma4ParserAcceptsLiveZyphraMultilineToolCallEnvelope() throws {
        let output = """
        <zyphra_tool_call>
        <function=line_count
        <parameter=text
        >red
        green
        blue
        </parameter>
        </function>
        </zyphra_tool_call>
        """
        let call = try #require(
            ToolCallFormat.gemma4.createParser().parse(
                content: output,
                tools: [lineCountToolSpec()]
            )
        )

        #expect(call.function.name == "line_count")
        #expect(call.function.arguments["text"] == .string("red\ngreen\nblue"))
    }

    @Test("Gemma4 processor routes live Zyphra envelope to tool call without visible leak")
    func gemma4ProcessorRoutesLiveZyphraEnvelopeWithoutVisibleLeak() {
        let output = """
        <zyphra_tool_call>
        <function=line_count
        <parameter=text
        >red
        green
        blue
        </parameter>
        </function>
        </zyphra_tool_call>
        """
        let processor = ToolCallProcessor(format: .gemma4, tools: [lineCountToolSpec()])
        var visible = ""
        for ch in output {
            visible += processor.processChunk(String(ch)) ?? ""
        }
        visible += processor.processEOS() ?? ""

        #expect(visible.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        #expect(!visible.contains("zyphra_tool_call"))
        #expect(!visible.contains("<function="))
        #expect(processor.toolCalls.count == 1)
        #expect(processor.toolCalls.first?.function.name == "line_count")
        #expect(processor.toolCalls.first?.function.arguments["text"] == .string("red\ngreen\nblue"))
    }

    @Test("Gemma4 routes Zyphra tool calls inside reasoning wrapper without reasoning leak")
    func gemma4RoutesZyphraToolCallsInsideReasoningWrapperWithoutLeak() {
        let output = """
        thought<|channel>```
        <zyphra_tool_call>
        <function=line_count>
        <parameter=text>
        red
        green
        blue
        </parameter>
        </function>
        </zyphra_tool_call>
        ```\u{200B} <|im_end|>
        ```

        In response to the user's request, I have provided a tool call.
        """
        let processor = ToolCallProcessor(format: .gemma4, tools: [lineCountToolSpec()])
        let events = routeGenerationText(output, channel: .reasoning, through: processor)

        let visible = events.compactMap(\.chunk).joined()
        let reasoning = events.compactMap { event -> String? in
            if case .reasoning(let text) = event { return text }
            return nil
        }.joined()
        let calls = events.compactMap { event -> ToolCall? in
            if case .toolCall(let call) = event { return call }
            return nil
        }

        #expect(visible.isEmpty)
        #expect(reasoning.isEmpty)
        #expect(calls.count == 1)
        #expect(calls.first?.function.name == "line_count")
        #expect(calls.first?.function.arguments["text"] == .string("red\ngreen\nblue"))
    }

    @Test("Gemma4 parser still accepts native Gemma4 tool-call envelope")
    func gemma4ParserStillAcceptsNativeEnvelope() throws {
        let output = #"<|tool_call>call:line_count{text:<|"|>one\ntwo<|"|>}<tool_call|>"#
        let call = try #require(
            ToolCallFormat.gemma4.createParser().parse(
                content: output,
                tools: [lineCountToolSpec()]
            )
        )

        #expect(call.function.name == "line_count")
        #expect(call.function.arguments["text"] == .string("one\ntwo"))
    }

    @Test("Gemma4 parser recovers observed MXFP4 E4B schema drift without visible leak")
    func gemma4ParserRecoversObservedMXFP4E4BSchemaDrift() throws {
        let output = "line_ount{textc:<|\"|>red\ngreen\nblue<|\"|>}"
        let call = try #require(
            ToolCallFormat.gemma4.createParser().parse(
                content: output,
                tools: [lineCountToolSpec()]
            )
        )

        #expect(call.function.name == "line_count")
        #expect(call.function.arguments["text"] == .string("red\ngreen\nblue"))
    }

    @Test("Gemma4 processor recovers observed MXFP4 E2B repeat required-call drift")
    func gemma4ProcessorRecoversObservedMXFP4E2BRepeatRequiredCallDrift() {
        let output = #"line_ountc{text: "one\ntwo"}"#
        let processor = ToolCallProcessor(format: .gemma4, tools: [lineCountToolSpec()])
        var visible = ""
        for ch in output {
            visible += processor.processChunk(String(ch)) ?? ""
        }
        visible += processor.processEOS() ?? ""

        #expect(visible.isEmpty)
        #expect(processor.toolCalls.count == 1)
        #expect(processor.toolCalls.first?.function.name == "line_count")
        #expect(processor.toolCalls.first?.function.arguments["text"] == .string("one\ntwo"))
    }

    @Test("Gemma4 schema drift recovery does not invent undeclared tool calls")
    func gemma4SchemaDriftRecoveryDoesNotInventUndeclaredToolCalls() {
        let parser = ToolCallFormat.gemma4.createParser()

        #expect(parser.parse(content: "line_ount{textc:<|\"|>red<|\"|>}", tools: nil) == nil)
        #expect(
            parser.parse(
                content: "other_tool{textc:<|\"|>red<|\"|>}",
                tools: [lineCountToolSpec()]
            ) == nil
        )
    }

    private func lineCountToolSpec() -> [String: any Sendable] {
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
                    "additionalProperties": false,
                ] as [String: any Sendable],
            ] as [String: any Sendable],
        ]
    }
}
