import Foundation
import MLXLMCommon
import Testing

/// The GLM4 template writes a string argument between `<arg_value>` tags with
/// no padding. Raptor 0.7 appended "\n## v1.1\n- bug fixes\n" to a changelog;
/// trimming the value dropped the blank line and the final newline, so the
/// model undid and resent the identical call until its budget ran out.
@Suite("GLM4 string argument whitespace")
struct GLM4StringArgumentWhitespaceTests {
    private let content = "\n## v1.1\n- bug fixes\n"

    private var tools: [[String: any Sendable]] {
        let properties: [String: any Sendable] = [
            "path": ["type": "string"], "content": ["type": "string"], "options": ["type": "object"],
        ]
        let parameters: [String: any Sendable] = ["type": "object", "properties": properties]
        let function: [String: any Sendable] = ["name": "file_write", "parameters": parameters]
        return [["type": "function", "function": function]]
    }

    private var raw: String {
        "<tool_call>file_write<arg_key>path</arg_key><arg_value>CHANGELOG.md</arg_value>"
            + "<arg_key>content</arg_key><arg_value>\(content)</arg_value>"
            + "<arg_key>options</arg_key><arg_value> {\"n\": 2}\n</arg_value></tool_call>"
    }

    @Test func declaredStringKeepsItsEdgeWhitespace() throws {
        let call = try #require(GLM4ToolCallParser().parse(content: raw, tools: tools))
        #expect(call.function.arguments["path"] == .string("CHANGELOG.md"))
        #expect(call.function.arguments["content"] == .string(content))
        #expect(call.function.arguments["options"] == .object(["n": .int(2)]))
    }

    @Test func streamedCallKeepsEdgeWhitespaceAtEveryBoundary() {
        for split in 0...raw.count {
            let index = raw.index(raw.startIndex, offsetBy: split)
            let processor = ToolCallProcessor(format: .glm4, tools: tools)
            _ = processor.processChunk(String(raw[..<index]))
            _ = processor.processChunk(String(raw[index...]))
            #expect(processor.toolCalls.count == 1, "split \(split)")
            #expect(
                processor.toolCalls.first?.function.arguments["content"] == .string(content),
                "split \(split)")
        }
    }

    @Test func undeclaredValueIsStillTrimmedBeforeDecoding() throws {
        let call = try #require(GLM4ToolCallParser().parse(content: raw, tools: nil))
        #expect(call.function.arguments["content"] == .string("## v1.1\n- bug fixes"))
        #expect(call.function.arguments["options"] == .object(["n": .int(2)]))
    }
}
