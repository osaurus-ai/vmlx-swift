// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLXLMCommon
import Testing

@Suite("DSV4 DSML tool parser focused contracts")
struct DSMLToolCallParserFocusedTests {
    @Test("DSML parser extracts every invoke and preserves typed parameters")
    func parserExtractsEveryInvokeAndTypedParameters() {
        let dsml = DeepseekV4Tokens.dsml
        let output = """
            <\(dsml)tool_calls>
            <\(dsml)invoke name="get_weather">
            <\(dsml)parameter name="city" string="true">Paris</\(dsml)parameter>
            <\(dsml)parameter name="days" string="false">3</\(dsml)parameter>
            </\(dsml)invoke>
            <\(dsml)invoke name="set_alarm">
            <\(dsml)parameter name="enabled" string="false">true</\(dsml)parameter>
            <\(dsml)parameter name="tags" string="false">["morning","work"]</\(dsml)parameter>
            </\(dsml)invoke>
            </\(dsml)tool_calls>
            """

        let calls = DSMLToolCallParser().parseEOS(output, tools: nil)

        #expect(calls.count == 2)
        #expect(calls[0].function.name == "get_weather")
        #expect(calls[0].function.arguments["city"] == .string("Paris"))
        #expect(calls[0].function.arguments["days"] == .int(3))
        #expect(calls[1].function.name == "set_alarm")
        #expect(calls[1].function.arguments["enabled"] == .bool(true))
        #expect(
            calls[1].function.arguments["tags"]
                == .array([.string("morning"), .string("work")])
        )
    }

    @Test("DSML parser accepts DSV4 abbreviated invoke close observed in live decode")
    func parserAcceptsAbbreviatedInvokeClose() {
        let dsml = DeepseekV4Tokens.dsml
        let output = """
            <\(dsml)tool_cals>
            <\(dsml)invoke name="get_weather">
            <\(dsml)parameter name="location" string="true">Tokyo</\(dsml)parameter>
            </\(dsml)inv>
            </\(dsml)tool_cals>
            """

        let calls = DSMLToolCallParser().parseEOS(output, tools: nil)

        #expect(calls.count == 1)
        #expect(calls.first?.function.name == "get_weather")
        #expect(calls.first?.function.arguments["location"] == .string("Tokyo"))
    }

    @Test("DSML processor buffers observed misspelled outer block instead of leaking markup")
    func processorBuffersObservedMisspelledOuterBlock() {
        let dsml = DeepseekV4Tokens.dsml
        let output = """
            Let me take a look.
            <\(dsml)tool_cals>
            <\(dsml)invoke name="file_read">
            <\(dsml)parameter name="path" string="true">/Users/eric/Desktop/testmandel/mandelbrot.py</\(dsml)parameter>
            </\(dsml)inv>
            </\(dsml)tool_cals>
            """
        let processor = ToolCallProcessor(format: .dsml)
        var visible = ""
        for ch in output {
            visible += processor.processChunk(String(ch)) ?? ""
        }
        visible += processor.processEOS() ?? ""

        #expect(processor.toolCalls.count == 1)
        #expect(processor.toolCalls.first?.function.name == "file_read")
        #expect(
            processor.toolCalls.first?.function.arguments["path"]
                == .string("/Users/eric/Desktop/testmandel/mandelbrot.py")
        )
        #expect(!visible.contains("DSML"))
        #expect(!visible.contains("tool_cals"))
        #expect(!visible.contains("invoke name"))
    }

    @Test("DSML processor accepts live tool_ccalls/tool_cs alias without visible markup leak")
    func processorAcceptsLiveToolCCallsToolCSAlias() {
        let dsml = DeepseekV4Tokens.dsml
        let output = """
            <\(dsml)tool_ccalls>
            <\(dsml)invoke name="file_read">
            <\(dsml)parameter name="path" string="true">/Users/eric/Desktop/testmandel/mandelbrot.py</\(dsml)parameter>
            </\(dsml)inv>
            </\(dsml)tool_cs>
            """
        let processor = ToolCallProcessor(format: .dsml)
        var visible = ""
        for ch in output {
            visible += processor.processChunk(String(ch)) ?? ""
        }
        visible += processor.processEOS() ?? ""

        #expect(processor.toolCalls.count == 1)
        #expect(processor.toolCalls.first?.function.name == "file_read")
        #expect(
            processor.toolCalls.first?.function.arguments["path"]
                == .string("/Users/eric/Desktop/testmandel/mandelbrot.py")
        )
        #expect(visible.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        #expect(!visible.contains("DSML"))
        #expect(!visible.contains("tool_ccalls"))
        #expect(!visible.contains("tool_cs"))
        #expect(!visible.contains("invoke name"))
    }

    @Test("DSML processor treats live tool_cimport wrapper as protocol")
    func processorAcceptsLiveToolCImportWrapper() {
        let dsml = DeepseekV4Tokens.dsml
        let output = """
            <\(dsml)tool_cimport>
            <\(dsml)invoke name="file_read">
            <\(dsml)parameter name="path" string="true">mandelbrot.py</\(dsml)parameter>
            <\(dsml)parameter name="start_line" string="false">1</\(dsml)parameter>
            <\(dsml)parameter name="end_line" string="false">1</\(dsml)parameter>
            </\(dsml)inv>
            </\(dsml)tool_cimport>
            """
        let processor = ToolCallProcessor(format: .dsml)
        var visible = ""
        for ch in output {
            visible += processor.processChunk(String(ch)) ?? ""
        }
        visible += processor.processEOS() ?? ""

        #expect(processor.toolCalls.count == 1)
        #expect(processor.toolCalls.first?.function.name == "file_read")
        #expect(processor.toolCalls.first?.function.arguments["path"] == .string("mandelbrot.py"))
        #expect(processor.toolCalls.first?.function.arguments["start_line"] == .int(1))
        #expect(processor.toolCalls.first?.function.arguments["end_line"] == .int(1))
        #expect(visible.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        #expect(!visible.contains("DSML"))
        #expect(!visible.contains("tool_cimport"))
        #expect(!visible.contains("invoke name"))
    }

    @Test("DSV4 instruct prompt routes DSML output to tool calls without reasoning leakage")
    func instructPromptRoutesDSMLWithoutReasoningLeakage() {
        let prompt = DeepseekV4ChatEncoder().encode(
            messages: [.init(role: .user, content: "Weather in Paris?")],
            thinkingMode: .chat
        )
        #expect(prompt.hasSuffix(DeepseekV4Tokens.thinkEnd))

        var reasoningParser = ReasoningParser.forPrompt(
            stampName: "think_xml",
            promptTail: promptTail(prompt)
        )
        let toolProcessor = ToolCallProcessor(format: .dsml)
        let dsml = DeepseekV4Tokens.dsml
        let output = """
            <\(dsml)tool_calls>
            <\(dsml)invoke name="get_weather">
            <\(dsml)parameter name="city" string="true">Paris</\(dsml)parameter>
            </\(dsml)invoke>
            </\(dsml)tool_calls>
            """

        var reasoning = ""
        var visible = ""
        for ch in output {
            if var parser = reasoningParser {
                for segment in parser.feed(String(ch)) {
                    switch segment {
                    case .reasoning(let text):
                        reasoning += text
                    case .content(let text):
                        visible += toolProcessor.processChunk(text) ?? ""
                    }
                }
                reasoningParser = parser
            } else {
                visible += toolProcessor.processChunk(String(ch)) ?? ""
            }
        }
        if var parser = reasoningParser {
            for segment in parser.flush() {
                switch segment {
                case .reasoning(let text):
                    reasoning += text
                case .content(let text):
                    visible += toolProcessor.processChunk(text) ?? ""
                }
            }
            reasoningParser = parser
        }
        toolProcessor.processEOS()

        #expect(reasoning.isEmpty)
        #expect(visible.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        #expect(toolProcessor.toolCalls.count == 1)
        #expect(toolProcessor.toolCalls.first?.function.name == "get_weather")
        #expect(toolProcessor.toolCalls.first?.function.arguments["city"] == .string("Paris"))
    }

    @Test("DSV4 capability aliases route to DSML before generic DeepSeek")
    func capabilityAliasesPreferDSML() {
        for stamp in ["dsml", "deepseek_v4", "deepseek_v4_flash", "deepseekv4"] {
            #expect(ToolCallFormat.fromCapabilityName(stamp) == .dsml)
        }
        #expect(ToolCallFormat.fromCapabilityName("deepseek") == .glm4)
        #expect(ToolCallFormat.fromCapabilityName("deepseek_v3") == .glm4)
    }

    private func promptTail(_ prompt: String) -> String {
        let start =
            prompt.index(
                prompt.endIndex,
                offsetBy: -256,
                limitedBy: prompt.startIndex
            ) ?? prompt.startIndex
        return String(prompt[start...])
    }
}
