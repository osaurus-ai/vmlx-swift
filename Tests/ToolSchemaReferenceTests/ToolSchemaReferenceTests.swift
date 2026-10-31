import Foundation
import MLXLMCommon
import Testing
import VMLXJinja

@Suite("Template tool inputs preserve JSON Schema")
struct ToolSchemaReferenceTests {
    @Test("serialization retains constraints, nulls and property names at every depth")
    func fullSchemaReachesTemplate() throws {
        let source = #"{"function":{"name":"read_record","parameters":{"additionalProperties":false,"properties":{"choice":{"enum":["a",null],"type":["string","null"]},"free":true,"nested":{"additionalProperties":true,"properties":{"type":{"type":"integer"}},"type":"object"}},"required":["choice"],"type":"object"}},"type":"function"}"#
        let raw = try JSONSerialization.jsonObject(with: Data(source.utf8))
        // Decode through Foundation's real boxed scalar/null bridge.
        let value = try Value(any: raw)
        let tools: [ToolSpec] = [[
            "type": "function",
            "function": [
                "name": "read_record",
                "parameters": [
                    "additionalProperties": false,
                    "properties": [
                        "choice": ["enum": ["a", NSNull()] as [any Sendable],
                            "type": ["string", "null"]] as [String: any Sendable],
                        "free": true,
                        "nested": ["additionalProperties": true,
                            "properties": ["type": ["type": "integer"]],
                            "type": "object"] as [String: any Sendable],
                    ] as [String: any Sendable],
                    "required": ["choice"], "type": "object",
                ] as [String: any Sendable],
            ] as [String: any Sendable],
        ]]
        let passed = try #require(normalizedToolsForChatTemplate(tools)?.first)
        let template = try Template("{{ tool | tojson }}")
        let expected = try template.render(["tool": value])
        #expect(try template.render(["tool": Value(any: passed)]) == expected)
        #expect(try template.render(["tool": Value(any: normalizedToolForChatTemplate(tools[0]))]) == expected)
        #expect(expected.contains(#""additionalProperties": false"#))
        #expect(expected.contains(#""enum": ["a", null]"#))
        #expect(expected.contains(#""type": ["string", "null"]"#))
    }
}
