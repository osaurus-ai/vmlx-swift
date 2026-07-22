// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import Testing
@testable import MLXLMCommon

@Suite("Foundation JSON number bridge contracts")
struct JSONValueFoundationNumberFocusedTests {
    @Test("Foundation integer one stays an integer instead of becoming true")
    func foundationIntegerOneStaysInteger() {
        #expect(JSONValue.from(NSNumber(value: 1)) == .int(1))
        #expect(JSONValue.from(NSNumber(value: 0)) == .int(0))
        #expect(JSONValue.from(NSNumber(value: 23)) == .int(23))
        #expect(JSONValue.from(NSNumber(value: -1)) == .int(-1))
        #expect(JSONValue.from(NSNumber(value: Int.max)) == .int(Int.max))
        #expect(JSONValue.from(NSNumber(value: 1.0)) == .double(1.0))
        #expect(JSONValue.from(NSNumber(value: 1.5)) == .double(1.5))
        #expect(JSONValue.from(NSNumber(value: true)) == .bool(true))
        #expect(JSONValue.from(NSNumber(value: false)) == .bool(false))
    }

    @Test("Qwen XML nested target mark one survives tool parsing")
    func qwenXMLTargetMarkOneSurvivesToolParsing() throws {
        let output = """
            <tool_call><function=agent_action><parameter=target>{"mark":1}</parameter><parameter=verb>click</parameter></function></tool_call>
            """
        let call = try #require(
            ToolCallFormat.xmlFunction.createParser().parse(
                content: output,
                tools: [agentActionToolSpec()]
            )
        )

        #expect(call.function.name == "agent_action")
        #expect(call.function.arguments["target"] == .object(["mark": .int(1)]))
        #expect(call.function.arguments["verb"] == .string("click"))
    }

    private func agentActionToolSpec() -> [String: any Sendable] {
        [
            "type": "function",
            "function": [
                "name": "agent_action",
                "parameters": [
                    "type": "object",
                    "properties": [
                        "verb": ["type": "string"] as [String: any Sendable],
                        "target": [
                            "type": "object",
                            "properties": [
                                "mark": ["type": "integer"] as [String: any Sendable]
                            ] as [String: any Sendable],
                        ] as [String: any Sendable],
                    ] as [String: any Sendable],
                    "required": ["verb"],
                ] as [String: any Sendable],
            ] as [String: any Sendable],
        ]
    }
}
