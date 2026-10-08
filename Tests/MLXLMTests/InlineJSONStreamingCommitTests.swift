import Foundation
import Testing

@testable import MLXLMCommon

/// Inline JSON tool calls (no wrapper tags — Llama 3) are parsed while they
/// stream. The parsers' end-of-stream repair (append up to two missing `}`)
/// used to run on every chunk, so `{"name": "get_weather", "parameters": {`
/// committed `get_weather({})` and the rest of the object leaked into the
/// visible answer. A call must commit only once its object has closed.
@Suite("Inline JSON tool calls commit only when complete")
struct InlineJSONStreamingCommitTests {

    static func tool(_ name: String, _ properties: [String], required: [String] = []) -> [String: any Sendable] {
        [
            "type": "function",
            "function": [
                "name": name,
                "parameters": [
                    "type": "object",
                    "properties": Dictionary(uniqueKeysWithValues: properties.map {
                        ($0, ["type": "string"] as [String: any Sendable])
                    }) as [String: any Sendable],
                    "required": required,
                ] as [String: any Sendable],
            ] as [String: any Sendable],
        ]
    }

    static let weather = tool("get_weather", ["location", "unit"])
    static let shell = tool("run", ["code"], required: ["code"])

    private func stream(
        _ text: String, format: ToolCallFormat, tools: [[String: any Sendable]], chunk: Int
    ) -> (calls: [ToolCall], visible: String) {
        let processor = ToolCallProcessor(format: format, tools: tools)
        var visible = ""
        var index = text.startIndex
        while index < text.endIndex {
            let next = text.index(index, offsetBy: chunk, limitedBy: text.endIndex) ?? text.endIndex
            visible += processor.processChunk(String(text[index..<next])) ?? ""
            index = next
        }
        visible += processor.processEOS() ?? ""
        return (processor.toolCalls, visible)
    }

    @Test("optional-only arguments survive token-sized streaming", arguments: [1, 2, 3, 5, 8])
    func optionalArgumentsNotTruncated(chunk: Int) {
        let text = #"{"name": "get_weather", "parameters": {"location": "Paris", "unit": "celsius"}}"#
        let result = stream(text, format: .llama3, tools: [Self.weather], chunk: chunk)
        #expect(result.calls.count == 1)
        #expect(result.calls.first?.function.arguments["location"] == .string("Paris"))
        #expect(result.calls.first?.function.arguments["unit"] == .string("celsius"))
        #expect(result.visible.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty, "leaked: \(result.visible)")
    }

    @Test("braces inside string arguments do not end the object early", arguments: [1, 4])
    func bracesInsideStrings(chunk: Int) {
        let code = "if (a) { b() } else {"
        let json = try! JSONSerialization.data(withJSONObject: ["name": "run", "parameters": ["code": code]], options: [.sortedKeys])
        let text = String(decoding: json, as: UTF8.self)
        let result = stream(text, format: .llama3, tools: [Self.shell], chunk: chunk)
        #expect(result.calls.count == 1)
        #expect(result.calls.first?.function.arguments["code"] == .string(code))
        #expect(result.visible.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty, "leaked: \(result.visible)")
    }

    @Test("prose before the call stays visible and the call still parses")
    func leadingProse() {
        let text = #"Let me check. {"name": "get_weather", "parameters": {"location": "Oslo"}}"#
        let result = stream(text, format: .llama3, tools: [Self.weather], chunk: 1)
        #expect(result.calls.first?.function.arguments["location"] == .string("Oslo"))
        #expect(result.visible.trimmingCharacters(in: .whitespaces) == "Let me check.")
    }

    @Test("a truncated object is still repaired at end of stream")
    func truncatedAtEOSStillRepaired() {
        let text = #"{"name": "get_weather", "parameters": {"location": "Rome"}"#
        let result = stream(text, format: .llama3, tools: [Self.weather], chunk: 1)
        #expect(result.calls.first?.function.arguments["location"] == .string("Rome"))
    }

    @Test("non-tool JSON is still shown as text")
    func plainJSONIsText() {
        let text = #"{"answer": 42}"#
        let result = stream(text, format: .llama3, tools: [Self.weather], chunk: 1)
        #expect(result.calls.isEmpty)
        #expect(result.visible == text)
    }
}
