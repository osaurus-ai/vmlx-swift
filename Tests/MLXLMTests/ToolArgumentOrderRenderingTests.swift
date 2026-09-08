import Foundation
import Testing
import MLXLMCommon
import VMLXJinja
@testable import VMLXTokenizers

@Suite("Tool argument order reaches native template rendering")
struct ToolArgumentOrderRenderingTests {
    @Test func jsonEscapedNamesRetainTheirNativeOrder() throws {
        let call = ToolCall(function: .init(
            name: "write", arguments: ["zé": .string("007"), "a\"b": .int(2)],
            rawArgumentsJSON: #"{"z\u00e9":"007","a\"b":2}"#))
        #expect(call.function.argumentOrder == ["zé", "a\"b"])
        let input = defaultMessageDict(for: .assistant("", toolCalls: [call]))
        #expect(try keys(input, nested: true) == "zé=007;a\"b=2;")
    }

    private func message(order: [String]?) throws -> [String: any Sendable] {
        let call = ToolCall(function: .init(
            name: "write", arguments: ["path": .string("note.md"), "content": .string("007")],
            argumentOrder: order))
        let restored = try JSONDecoder().decode(ToolCall.self, from: JSONEncoder().encode(call))
        return defaultMessageDict(for: .assistant("", toolCalls: [restored]))
    }

    private func keys(_ message: [String: any Sendable], nested: Bool) throws -> String {
        let access = nested ? "m.tool_calls[0].function.arguments" : "m.tool_calls[0].arguments"
        return try Template("{% for key, value in " + access + ".items() %}{{ key }}={{ value }};{% endfor %}")
            .render(["m": try chatTemplateMessageValue(message)])
    }

    @Test func persistedOrderReachesBothNativeViews() throws {
        let input = try message(order: ["path", "content"])
        #expect(try keys(input, nested: true) == "path=note.md;content=007;")
        #expect(try keys(input, nested: false) == "path=note.md;content=007;")
        guard case .object(let rendered) = try chatTemplateMessageValue(input) else {
            Issue.record("Expected message object"); return
        }
        #expect(rendered["_vmlx_tool_argument_orders"] == nil)
    }

    @Test(arguments: [["path", "path"], ["missing", "content"], ["path"], []])
    func invalidOrderDoesNotDropOrInventArguments(_ order: [String]) throws {
        #expect(try keys(message(order: order), nested: true) == "content=007;path=note.md;")
    }

    @Test func absentMetadataPreservesExistingRendering() throws {
        let input = try message(order: nil)
        #expect(try chatTemplateMessageValue(input) == Value(any: input))
        #expect(try keys(input, nested: true) == "content=007;path=note.md;")
    }
}
