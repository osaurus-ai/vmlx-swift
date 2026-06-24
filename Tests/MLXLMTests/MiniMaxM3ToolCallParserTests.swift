// MiniMax-M3 tool-call parser + routing tests (no model load).
// Proves minimax_m3 routes to its OWN format (not minimax_m2's) and that the
// namespaced `<tool_call><invoke name><key>v</key></invoke></tool_call>` envelope
// parses — including the bare form when the detokenizer elides the namespace
// token. Also pins that minimax_m2 routing is unchanged (M2/M2.7 regression).

import Foundation
@testable import MLXLMCommon
import Testing

@Suite("MiniMax-M3 tool-call parser")
struct MiniMaxM3ToolCallParserTests {
    private static let ns = "]<]minimax[>["

    @Test("minimax_m3 routes to .minimaxM3; minimax_m2 stays .minimaxM2")
    func routing() {
        #expect(ToolCallFormat.infer(from: "minimax_m3") == .minimaxM3)
        #expect(ToolCallFormat.infer(from: "minimax_m3_vl") == .minimaxM3)
        #expect(ToolCallFormat.infer(from: "minimax_m2") == .minimaxM2)
        #expect(ToolCallFormat.fromCapabilityName("minimax_m3") == .minimaxM3)
        #expect(ToolCallFormat.fromCapabilityName("minimax") == .minimaxM2)
    }

    @Test("parses the namespaced <tool_call><invoke><key> envelope")
    func parsesNamespaced() {
        let ns = Self.ns
        let raw = "\(ns)<tool_call>\n\(ns)<invoke name=\"read_file\">"
            + "\(ns)<path>/tmp/main.swift\(ns)</path>"
            + "\(ns)<max_lines>50\(ns)</max_lines>"
            + "\(ns)</invoke>\n\(ns)</tool_call>"
        let call = ToolCallFormat.minimaxM3.createParser().parse(content: raw, tools: nil)
        #expect(call?.function.name == "read_file")
        #expect(call?.function.arguments["path"] == .string("/tmp/main.swift"))
        #expect(call?.function.arguments["max_lines"] != nil)
    }

    @Test("parses the bare envelope when the namespace token is elided")
    func parsesBare() {
        let raw = "<tool_call><invoke name=\"list_dir\"><path>/src</path></invoke></tool_call>"
        let call = ToolCallFormat.minimaxM3.createParser().parse(content: raw, tools: nil)
        #expect(call?.function.name == "list_dir")
        #expect(call?.function.arguments["path"] == .string("/src"))
    }

    @Test("parses nested object + array args (parity with the Python authority's example)")
    func parsesNestedArgs() {
        let raw = """
            <tool_call>
            <invoke name="get_weather">
            <location>San Francisco</location>
            <opts><unit>celsius</unit></opts>
            <days><item>mon</item><item>tue</item></days>
            </invoke>
            </tool_call>
            """
        let call = ToolCallFormat.minimaxM3.createParser().parse(content: raw, tools: nil)
        #expect(call?.function.name == "get_weather")
        #expect(call?.function.arguments["location"] == .string("San Francisco"))
        #expect(call?.function.arguments["opts"] == .object(["unit": .string("celsius")]))
        #expect(call?.function.arguments["days"] == .array([.string("mon"), .string("tue")]))
    }

    @Test("strips a leading <mm:think> reasoning block before the tool_call")
    func stripsReasoningBlock() {
        let raw = "<mm:think>I should call the tool.</mm:think>"
            + "<tool_call><invoke name=\"f\"><n>3</n></invoke></tool_call>"
        let call = ToolCallFormat.minimaxM3.createParser().parse(content: raw, tools: nil)
        #expect(call?.function.name == "f")
        #expect(call?.function.arguments["n"] == .int(3))   // scalar coerced via JSON
    }

    @Test("the M2 parser does NOT parse M3 args (confirms the gap was real)")
    func m2CannotParseM3Args() {
        let raw = "<tool_call><invoke name=\"f\"><k>v</k></invoke></tool_call>"
        let m2 = ToolCallFormat.minimaxM2.createParser().parse(content: raw, tools: nil)
        // M2 looks for <parameter name=>; M3's bare <k>v</k> yields no such arg.
        #expect(m2?.function.arguments["k"] == nil)
    }
}
