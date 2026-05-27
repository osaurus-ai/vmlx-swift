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

    @Test("DSML processor accepts live tool_crs alias after bare tool-name marker")
    func processorAcceptsLiveToolCRSAliasAfterBareToolNameMarker() {
        let dsml = DeepseekV4Tokens.dsml
        let output = """
            -line_count
            <\(dsml)tool_crs>
            <\(dsml)invoke name="line_count">
            <\(dsml)parameter name="text" string="true">alpha
            beta
            gamma</\(dsml)parameter>
            </\(dsml)inv>
            </\(dsml)tool_crs>
            """
        let processor = ToolCallProcessor(format: .dsml, tools: lineCountToolSchema())
        var visible = ""
        for ch in output {
            visible += processor.processChunk(String(ch)) ?? ""
        }
        visible += processor.processEOS() ?? ""

        #expect(processor.toolCalls.count == 1)
        #expect(processor.toolCalls.first?.function.name == "line_count")
        #expect(
            processor.toolCalls.first?.function.arguments["text"]
                == .string("alpha\nbeta\ngamma")
        )
        #expect(visible.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        #expect(!visible.contains("DSML"))
        #expect(!visible.contains("tool_crs"))
        #expect(!visible.contains("invoke name"))
        #expect(!visible.contains("line_count"))
    }

    @Test("DSV4 processor accepts live request_tool XML rail")
    func processorAcceptsLiveRequestToolXMLRail() {
        let output = #"_only:request_tool<invoke><target_name>line_count</target><params><string>one\ntwo</string></params></invoke>"#
        let processor = ToolCallProcessor(format: .dsml, tools: lineCountToolSchema())
        var visible = ""
        for ch in output {
            visible += processor.processChunk(String(ch)) ?? ""
        }
        visible += processor.processEOS() ?? ""

        #expect(processor.toolCalls.count == 1)
        #expect(processor.toolCalls.first?.function.name == "line_count")
        #expect(processor.toolCalls.first?.function.arguments["text"] == .string("one\ntwo"))
        #expect(visible.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        #expect(!visible.contains("request_tool"))
        #expect(!visible.contains("<invoke>"))
        #expect(!visible.contains("line_count"))
    }

    @Test("DSML processor accepts live bare-name JSON with malformed string escape")
    func processorAcceptsLiveBareNameJSONWithMalformedStringEscape() {
        let output = #"line_count{"text":"alpha\nbeta\gamma"}"#
        let processor = ToolCallProcessor(format: .dsml, tools: lineCountToolSchema())
        var visible = ""
        for ch in output {
            visible += processor.processChunk(String(ch)) ?? ""
        }
        visible += processor.processEOS() ?? ""

        #expect(processor.toolCalls.count == 1)
        #expect(processor.toolCalls.first?.function.name == "line_count")
        #expect(
            processor.toolCalls.first?.function.arguments["text"]
                == .string("alpha\nbeta\\gamma")
        )
        #expect(visible.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        #expect(!visible.contains("line_count"))
        #expect(!visible.contains(#"{"text":"#))
    }

    @Test("DSML processor suppresses incomplete protocol opener at EOS")
    func processorSuppressesIncompleteProtocolOpenerAtEOS() {
        let dsml = DeepseekV4Tokens.dsml
        let output = "\n\n<\(dsml)tool_c"
        let processor = ToolCallProcessor(format: .dsml, tools: fileReadToolSchema())
        var visible = ""
        for ch in output {
            visible += processor.processChunk(String(ch)) ?? ""
        }
        visible += processor.processEOS() ?? ""

        #expect(processor.toolCalls.isEmpty)
        #expect(visible.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        #expect(!visible.contains("DSML"))
        #expect(!visible.contains("tool_c"))
    }

    @Test("DSML processor routes Osaurus folder and git tools through live aliases")
    func processorRoutesOsaurusFolderAndGitToolsThroughLiveAliases() {
        let fixtures: [DSMLToolFixture] = [
            .init(
                name: "file_tree",
                parameters: [
                    .init(name: "path", value: ".", string: true, expected: .string(".")),
                    .init(name: "max_depth", value: "2", string: false, expected: .int(2)),
                ]
            ),
            .init(
                name: "file_read",
                parameters: [
                    .init(name: "path", value: "mandelbrot.py", string: true, expected: .string("mandelbrot.py")),
                    .init(name: "start_line", value: "38", string: false, expected: .int(38)),
                    .init(name: "end_line", value: "41", string: false, expected: .int(41)),
                ]
            ),
            .init(
                name: "file_write",
                parameters: [
                    .init(name: "path", value: "osaurus_probe.txt", string: true, expected: .string("osaurus_probe.txt")),
                    .init(name: "content", value: "alpha\nbeta", string: true, expected: .string("alpha\nbeta")),
                ]
            ),
            .init(
                name: "file_edit",
                parameters: [
                    .init(name: "path", value: "osaurus_probe.txt", string: true, expected: .string("osaurus_probe.txt")),
                    .init(name: "old_string", value: "alpha", string: true, expected: .string("alpha")),
                    .init(name: "new_string", value: "beta", string: true, expected: .string("beta")),
                ]
            ),
            .init(
                name: "file_search",
                parameters: [
                    .init(name: "pattern", value: "np.clip", string: true, expected: .string("np.clip")),
                    .init(name: "path", value: "mandelbrot.py", string: true, expected: .string("mandelbrot.py")),
                    .init(name: "max_results", value: "3", string: false, expected: .int(3)),
                ]
            ),
            .init(
                name: "shell_run",
                parameters: [
                    .init(name: "command", value: "printf ok", string: true, expected: .string("printf ok")),
                    .init(name: "timeout", value: "5", string: false, expected: .int(5)),
                ]
            ),
            .init(name: "git_status", parameters: []),
            .init(
                name: "git_diff",
                parameters: [
                    .init(name: "path", value: "mandelbrot.py", string: true, expected: .string("mandelbrot.py")),
                    .init(name: "staged", value: "false", string: false, expected: .bool(false)),
                ]
            ),
            .init(
                name: "git_commit",
                parameters: [
                    .init(name: "message", value: "probe commit", string: true, expected: .string("probe commit")),
                ]
            ),
        ]

        for fixture in fixtures {
            let processor = ToolCallProcessor(format: .dsml)
            var visible = ""
            for ch in liveAliasDSML(for: fixture) {
                visible += processor.processChunk(String(ch)) ?? ""
            }
            visible += processor.processEOS() ?? ""

            #expect(
                visible.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                "\(fixture.name) DSML leaked visible text: \(visible)"
            )
            #expect(!visible.contains("DSML"), "\(fixture.name) leaked DSML marker: \(visible)")
            #expect(!visible.contains("tool_ccalls"), "\(fixture.name) leaked start alias: \(visible)")
            #expect(!visible.contains("tool_cs"), "\(fixture.name) leaked end alias: \(visible)")
            #expect(processor.toolCalls.count == 1, "\(fixture.name) should emit one tool call")

            let call = processor.toolCalls.first
            #expect(call?.function.name == fixture.name)
            for parameter in fixture.parameters {
                assertArgument(
                    call?.function.arguments[parameter.name],
                    matches: parameter.expected,
                    tool: fixture.name,
                    parameter: parameter.name
                )
            }
        }
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

    @Test("DSML inline JSON fallback routes only schema-valid tool objects")
    func inlineJSONFallbackRoutesOnlySchemaValidToolObjects() {
        let output = """
            {"tool":"file_read","path":"mandelbrot.py","start_line":38,"end_line":41}
            """
        let processor = ToolCallProcessor(format: .dsml, tools: fileReadToolSchema())
        var visible = ""
        for ch in output {
            visible += processor.processChunk(String(ch)) ?? ""
        }
        visible += processor.processEOS() ?? ""

        #expect(visible.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        #expect(processor.toolCalls.count == 1)
        let call = processor.toolCalls.first
        #expect(call?.function.name == "file_read")
        #expect(call?.function.arguments["path"] == .string("mandelbrot.py"))
        #expect(call?.function.arguments["start_line"] == .int(38))
        #expect(call?.function.arguments["end_line"] == .int(41))
    }

    @Test("DSML inline JSON fallback quarantines known malformed tool-shaped answers")
    func inlineJSONFallbackQuarantinesKnownMalformedToolShapedAnswers() {
        let output = """
            {"tool":"file_read","r":"np.clip(esc * 4.0 - 1.0, 0.0, 1.0)","g":"np.clip(1.0 - np.abs(esc * 2.0 - 1.0), 0.0, 1.0)","b":"np.clip(1.0 - esc * 2.0, 0.0, 1.0)"}
            """
        let processor = ToolCallProcessor(format: .dsml, tools: fileReadToolSchema())
        var visible = ""
        for ch in output {
            visible += processor.processChunk(String(ch)) ?? ""
        }
        visible += processor.processEOS() ?? ""

        #expect(visible.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        #expect(!visible.contains("\"tool\":\"file_read\""))
        #expect(!visible.contains("np.clip"))
        #expect(processor.toolCalls.count == 1)
        let call = processor.toolCalls.first
        #expect(call?.function.name == "file_read")
        #expect(call?.function.arguments["path"] == nil)
        #expect(call?.function.arguments["_error"] == .string("invalid_tool_arguments"))
        #expect(call?.function.arguments["_field"] == .string("path"))
        #expect(call?.function.arguments["r"] == nil)
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

    private struct DSMLToolFixture {
        let name: String
        let parameters: [DSMLParameterFixture]
    }

    private struct DSMLParameterFixture {
        let name: String
        let value: String
        let string: Bool
        let expected: DSMLExpectedArgument
    }

    private enum DSMLExpectedArgument {
        case string(String)
        case int(Int)
        case bool(Bool)
    }

    private func liveAliasDSML(for fixture: DSMLToolFixture) -> String {
        let dsml = DeepseekV4Tokens.dsml
        var lines = [
            "<\(dsml)tool_ccalls>",
            "<\(dsml)invoke name=\"\(fixture.name)\">",
        ]
        lines += fixture.parameters.map { parameter in
            "<\(dsml)parameter name=\"\(parameter.name)\" string=\"\(parameter.string ? "true" : "false")\">\(parameter.value)</\(dsml)parameter>"
        }
        lines += [
            "</\(dsml)inv>",
            "</\(dsml)tool_cs>",
        ]
        return lines.joined(separator: "\n")
    }

    private func assertArgument(
        _ actual: (any Sendable)?,
        matches expected: DSMLExpectedArgument,
        tool: String,
        parameter: String
    ) {
        switch expected {
        case .string(let value):
            #expect(actual as? JSONValue == .string(value), "\(tool).\(parameter) mismatch")
        case .int(let value):
            #expect(actual as? JSONValue == .int(value), "\(tool).\(parameter) mismatch")
        case .bool(let value):
            #expect(actual as? JSONValue == .bool(value), "\(tool).\(parameter) mismatch")
        }
    }

    private func fileReadToolSchema() -> [[String: any Sendable]] {
        [
            [
                "type": "function",
                "function": [
                    "name": "file_read",
                    "parameters": [
                        "type": "object",
                        "properties": [
                            "path": ["type": "string"] as [String: any Sendable],
                            "start_line": ["type": "integer"] as [String: any Sendable],
                            "end_line": ["type": "integer"] as [String: any Sendable],
                        ] as [String: any Sendable],
                        "required": ["path"],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ] as [String: any Sendable],
        ]
    }

    private func lineCountToolSchema() -> [[String: any Sendable]] {
        [
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
            ] as [String: any Sendable],
        ]
    }

}
