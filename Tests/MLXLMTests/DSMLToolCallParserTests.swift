// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Tests for `DSMLToolCallParser` — covers the DSV4 format documented in
// `jang/research/DSV-FAMILY-RUNTIME-GUIDE.md` §24. Remember: the `｜`
// markers are curly quotes (U+FF5C), not ASCII pipes.

import Foundation
import MLXLMCommon
import Testing

@Suite("DSML Tool-Call Parser (DSV4)")
struct DSMLToolCallParserTests {
    // MARK: - Fixtures

    /// Canonical single-invoke DSV4 tool-call block.
    static let singleInvoke = """
        <\u{FF5C}DSML\u{FF5C}tool_calls>
        <\u{FF5C}DSML\u{FF5C}invoke name="get_weather">
        <\u{FF5C}DSML\u{FF5C}parameter name="location" string="true">San Francisco</\u{FF5C}DSML\u{FF5C}parameter>
        <\u{FF5C}DSML\u{FF5C}parameter name="units" string="true">celsius</\u{FF5C}DSML\u{FF5C}parameter>
        </\u{FF5C}DSML\u{FF5C}invoke>
        </\u{FF5C}DSML\u{FF5C}tool_calls>
        """

    /// Two invokes inside one outer block — tests multi-call semantics.
    static let multiInvoke = """
        <\u{FF5C}DSML\u{FF5C}tool_calls>
        <\u{FF5C}DSML\u{FF5C}invoke name="get_weather">
        <\u{FF5C}DSML\u{FF5C}parameter name="location" string="true">San Francisco</\u{FF5C}DSML\u{FF5C}parameter>
        </\u{FF5C}DSML\u{FF5C}invoke>
        <\u{FF5C}DSML\u{FF5C}invoke name="get_time">
        <\u{FF5C}DSML\u{FF5C}parameter name="timezone" string="true">America/Los_Angeles</\u{FF5C}DSML\u{FF5C}parameter>
        </\u{FF5C}DSML\u{FF5C}invoke>
        </\u{FF5C}DSML\u{FF5C}tool_calls>
        """

    /// Single invoke with `string="false"` parameters — JSON-decoded per §24.
    static let jsonParams = """
        <\u{FF5C}DSML\u{FF5C}tool_calls>
        <\u{FF5C}DSML\u{FF5C}invoke name="set_config">
        <\u{FF5C}DSML\u{FF5C}parameter name="enabled" string="false">true</\u{FF5C}DSML\u{FF5C}parameter>
        <\u{FF5C}DSML\u{FF5C}parameter name="retries" string="false">5</\u{FF5C}DSML\u{FF5C}parameter>
        <\u{FF5C}DSML\u{FF5C}parameter name="tags" string="false">["a","b","c"]</\u{FF5C}DSML\u{FF5C}parameter>
        <\u{FF5C}DSML\u{FF5C}parameter name="label" string="true">ready</\u{FF5C}DSML\u{FF5C}parameter>
        </\u{FF5C}DSML\u{FF5C}invoke>
        </\u{FF5C}DSML\u{FF5C}tool_calls>
        """

    // MARK: - Single invoke

    @Test("parse(content:) extracts a single invoke")
    func singleInvokeParse() throws {
        let parser = DSMLToolCallParser()
        let call = try #require(parser.parse(content: Self.singleInvoke, tools: nil))
        #expect(call.function.name == "get_weather")
        let args = call.function.arguments
        #expect(args["location"] == .string("San Francisco"))
        #expect(args["units"] == .string("celsius"))
    }

    @Test("parseEOS returns the invoke in a single-call block")
    func singleInvokeEOS() {
        let parser = DSMLToolCallParser()
        let calls = parser.parseEOS(Self.singleInvoke, tools: nil)
        #expect(calls.count == 1)
        #expect(calls.first?.function.name == "get_weather")
    }

    // MARK: - Multi-invoke

    @Test("parseEOS extracts every invoke in a multi-call block, in order")
    func multiInvokeEOS() throws {
        let parser = DSMLToolCallParser()
        let calls = parser.parseEOS(Self.multiInvoke, tools: nil)
        #expect(calls.count == 2)
        #expect(calls[0].function.name == "get_weather")
        #expect(calls[0].function.arguments["location"] == .string("San Francisco"))
        #expect(calls[1].function.name == "get_time")
        #expect(calls[1].function.arguments["timezone"] == .string("America/Los_Angeles"))
    }

    @Test("parse(content:) returns the FIRST invoke when multiple are present")
    func multiInvokeFirstOnly() throws {
        let parser = DSMLToolCallParser()
        let call = try #require(parser.parse(content: Self.multiInvoke, tools: nil))
        #expect(call.function.name == "get_weather")
    }

    // MARK: - JSON-typed parameters

    @Test("string=\"false\" params are JSON-decoded (bool/int/array), string=\"true\" stays raw")
    func jsonParamDecoding() throws {
        let parser = DSMLToolCallParser()
        let call = try #require(parser.parse(content: Self.jsonParams, tools: nil))
        let args = call.function.arguments
        #expect(args["enabled"] == .bool(true))
        #expect(args["retries"] == .int(5))
        #expect(
            args["tags"]
                == .array([.string("a"), .string("b"), .string("c")]))
        #expect(args["label"] == .string("ready"))
    }

    // MARK: - Robustness

    @Test("Malformed JSON in string=\"false\" param falls back to raw string")
    func malformedJSONFallback() throws {
        let body = """
            <\u{FF5C}DSML\u{FF5C}tool_calls>
            <\u{FF5C}DSML\u{FF5C}invoke name="dangerous">
            <\u{FF5C}DSML\u{FF5C}parameter name="payload" string="false">{not: valid json,,}</\u{FF5C}DSML\u{FF5C}parameter>
            </\u{FF5C}DSML\u{FF5C}invoke>
            </\u{FF5C}DSML\u{FF5C}tool_calls>
            """
        let parser = DSMLToolCallParser()
        let call = try #require(parser.parse(content: body, tools: nil))
        #expect(call.function.name == "dangerous")
        // Fallback preserves the raw string so the call isn't dropped.
        #expect(call.function.arguments["payload"] == .string("{not: valid json,,}"))
    }

    @Test("Input with no invoke block returns nil")
    func noInvokeReturnsNil() {
        let parser = DSMLToolCallParser()
        #expect(parser.parse(content: "just plain chat, no tool calls here.", tools: nil) == nil)
        #expect(parser.parseEOS("just plain chat, no tool calls here.", tools: nil).isEmpty)
    }

    @Test("Multi-line value with surrounding newlines is trimmed by exactly one")
    func multilineValueTrim() throws {
        // Matches the Python reference behaviour in `encoding_dsv4.py`
        // which injects a leading+trailing newline around multi-line
        // string values for readability. Inner newlines must be kept.
        let body = """
            <\u{FF5C}DSML\u{FF5C}tool_calls>
            <\u{FF5C}DSML\u{FF5C}invoke name="write">
            <\u{FF5C}DSML\u{FF5C}parameter name="body" string="true">
            line1
            line2
            </\u{FF5C}DSML\u{FF5C}parameter>
            </\u{FF5C}DSML\u{FF5C}invoke>
            </\u{FF5C}DSML\u{FF5C}tool_calls>
            """
        let parser = DSMLToolCallParser()
        let call = try #require(parser.parse(content: body, tools: nil))
        #expect(call.function.arguments["body"] == .string("line1\nline2"))
    }

    // MARK: - Format registration

    @Test("deepseek_v4 model_type auto-infers the DSML format")
    func inferFromModelType() {
        #expect(ToolCallFormat.infer(from: "deepseek_v4") == .dsml)
        #expect(ToolCallFormat.infer(from: "deepseek_v4_flash") == .dsml)
        #expect(ToolCallFormat.infer(from: "deepseek_v4_pro") == .dsml)
    }

    @Test("capabilityName aliases resolve to .dsml")
    func fromCapabilityNameAliases() {
        #expect(ToolCallFormat.fromCapabilityName("dsml") == .dsml)
        #expect(ToolCallFormat.fromCapabilityName("deepseek_v4") == .dsml)
        #expect(ToolCallFormat.fromCapabilityName("deepseekv4") == .dsml)
        #expect(ToolCallFormat.fromCapabilityName("DSML") == .dsml)
    }

    @Test("ToolCallFormat.dsml rawValue round-trips for serialization")
    func rawValueRoundtrip() {
        #expect(ToolCallFormat.dsml.rawValue == "dsml")
        #expect(ToolCallFormat(rawValue: "dsml") == .dsml)
    }

    @Test("ToolCallFormat.dsml.createParser() returns a DSMLToolCallParser")
    func createParserType() {
        let parser = ToolCallFormat.dsml.createParser()
        #expect(parser is DSMLToolCallParser)
    }

    @Test("Nemotron parser buffers DSML and recovers an explicit partial invoke at EOS")
    func nemotronParserAcceptsLivePartialDSML() {
        let tools: [[String: any Sendable]] = [[
            "type": "function",
            "function": [
                "name": "line_count",
                "parameters": [
                    "type": "object",
                    "properties": [
                        "path": ["type": "string"] as [String: any Sendable],
                    ] as [String: any Sendable],
                    "required": ["path"],
                ] as [String: any Sendable],
            ] as [String: any Sendable],
        ]]
        let output = """
            At the start of the response, output your reasoning and then tool calls.

            <\u{FF5C}DSML\u{FF5C}tool_calls>
            <\u{FF5C}DSML\u{FF5C}invoke name="line_count">
            <\u{FF5C}DSML\u{FF5C}parameter name="path" string="true">/Users/eric/Desktop/testmandel/mandelbrot.py
            </\u{FF5C}DSMLｺ.
            """

        let processor = ToolCallProcessor(format: .nemotron, tools: tools)
        var visible = ""
        for chunk in output.split(separator: "\n", omittingEmptySubsequences: false) {
            visible += processor.processChunk(String(chunk) + "\n") ?? ""
        }
        visible += processor.processEOS() ?? ""

        #expect(processor.toolCalls.count == 1)
        #expect(processor.toolCalls.first?.function.name == "line_count")
        #expect(
            processor.toolCalls.first?.function.arguments["path"]
                == .string("/Users/eric/Desktop/testmandel/mandelbrot.py"))
        #expect(!visible.contains("DSML"))
        #expect(!visible.contains("line_count"))
    }
}
