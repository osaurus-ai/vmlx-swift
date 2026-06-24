// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLXLMCommon
import Testing

@Suite("Mistral tool parser focused contracts")
struct MistralToolCallParserFocusedTests {
    private let parser = MistralToolCallParser()

    // MARK: V7/V11 — JSON-array format (Mistral-Small-3.1/3.2 2503/2506)

    @Test("parses the 2503 JSON-array format [TOOL_CALLS][{name,arguments}]")
    func parsesJSONArrayFormat() {
        let buffer = #"[TOOL_CALLS][{"name": "get_weather", "arguments": {"city": "Tokyo"}}]"#
        let calls = parser.parseEOS(buffer, tools: nil)
        #expect(calls.count == 1)
        #expect(calls.first?.function.name == "get_weather")
        #expect(calls.first?.function.arguments["city"] == .string("Tokyo"))
    }

    @Test("parses JSON-array buffer without the [TOOL_CALLS] prefix")
    func parsesJSONArrayWithoutPrefix() {
        let buffer = #"[{"name": "get_weather", "arguments": {"city": "Paris"}}]"#
        let calls = parser.parseEOS(buffer, tools: nil)
        #expect(calls.count == 1)
        #expect(calls.first?.function.name == "get_weather")
        #expect(calls.first?.function.arguments["city"] == .string("Paris"))
    }

    @Test("parses multiple calls in one JSON array")
    func parsesMultipleCallsInArray() {
        let buffer =
            #"[TOOL_CALLS][{"name": "get_weather", "arguments": {"city": "Tokyo"}}, {"name": "get_time", "arguments": {"tz": "Asia/Tokyo"}}]"#
        let calls = parser.parseEOS(buffer, tools: nil)
        #expect(calls.count == 2)
        #expect(calls[0].function.name == "get_weather")
        #expect(calls[1].function.name == "get_time")
        #expect(calls[1].function.arguments["tz"] == .string("Asia/Tokyo"))
    }

    @Test("parses a no-argument call")
    func parsesNoArgumentCall() {
        let buffer = #"[TOOL_CALLS][{"name": "get_current_time", "arguments": {}}]"#
        let calls = parser.parseEOS(buffer, tools: nil)
        #expect(calls.count == 1)
        #expect(calls.first?.function.name == "get_current_time")
        #expect(calls.first?.function.arguments.isEmpty == true)
    }

    @Test("preserves typed argument values from the JSON array")
    func preservesTypedArguments() {
        let buffer =
            #"[TOOL_CALLS][{"name": "set_alarm", "arguments": {"enabled": true, "count": 3, "tags": ["a", "b"]}}]"#
        let calls = parser.parseEOS(buffer, tools: nil)
        #expect(calls.count == 1)
        let args = calls.first?.function.arguments
        #expect(args?["enabled"] == .bool(true))
        #expect(args?["count"] == .int(3))
        #expect(args?["tags"] == .array([.string("a"), .string("b")]))
    }

    // MARK: V13 — name[ARGS]{json} format (Mistral-3 2512, Devstral 2)

    @Test("still parses the V13 [ARGS] format")
    func parsesArgsFormat() {
        let call = parser.parse(
            content: #"[TOOL_CALLS]get_weather[ARGS]{"location": "Tokyo"}"#, tools: nil)
        #expect(call?.function.name == "get_weather")
        #expect(call?.function.arguments["location"] == .string("Tokyo"))
    }

    @Test("parses the V11 [CALL_ID]+[ARGS] variant")
    func parsesCallIdArgsFormat() {
        let call = parser.parse(
            content: #"[TOOL_CALLS]get_weather[CALL_ID]abc123[ARGS]{"location": "Osaka"}"#, tools: nil)
        #expect(call?.function.name == "get_weather")
        #expect(call?.function.arguments["location"] == .string("Osaka"))
    }

    @Test("parses multiple [ARGS] segments via parseEOS")
    func parsesMultipleArgsSegments() {
        let buffer =
            #"[TOOL_CALLS]fn1[ARGS]{"a": 1}[TOOL_CALLS]fn2[ARGS]{"b": 2}"#
        let calls = parser.parseEOS(buffer, tools: nil)
        #expect(calls.count == 2)
        #expect(calls[0].function.name == "fn1")
        #expect(calls[1].function.name == "fn2")
    }

    // MARK: Robustness — malformed input must not crash or fabricate calls

    @Test("malformed JSON yields no calls, no crash")
    func malformedYieldsNothing() {
        #expect(parser.parseEOS("[TOOL_CALLS]not json at all", tools: nil).isEmpty)
        #expect(parser.parseEOS("[TOOL_CALLS][{not valid}]", tools: nil).isEmpty)
        #expect(parser.parseEOS("", tools: nil).isEmpty)
        #expect(parser.parse(content: "[TOOL_CALLS][]", tools: nil) == nil)
    }

    @Test("array element without a name is skipped")
    func elementWithoutNameSkipped() {
        let buffer = #"[TOOL_CALLS][{"arguments": {"x": 1}}, {"name": "ok", "arguments": {}}]"#
        let calls = parser.parseEOS(buffer, tools: nil)
        #expect(calls.count == 1)
        #expect(calls.first?.function.name == "ok")
    }

    @Test("string-encoded arguments are decoded")
    func stringEncodedArguments() {
        let buffer = #"[TOOL_CALLS][{"name": "f", "arguments": "{\"k\": \"v\"}"}]"#
        let calls = parser.parseEOS(buffer, tools: nil)
        #expect(calls.count == 1)
        #expect(calls.first?.function.arguments["k"] == .string("v"))
    }
}
