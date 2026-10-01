import Foundation
import MLXLMCommon
import MLXVLM
import Testing

@Suite("GLM native message history")
struct Glm5NextMessageHistoryTests {
    static func readCall() -> ToolCall {
        let arguments: [String: JSONValue] = ["path": .string("note.md"), "max_lines": .int(7)]
        return ToolCall(id: "call_read", function: .init(
            name: "read_file", arguments: arguments,
            rawArgumentsJSON: #"{"path":"note.md","max_lines":7}"#))
    }

    static func writeCall() -> ToolCall {
        let arguments: [String: JSONValue] = ["path": .string("note.md"), "content": .string("007")]
        return ToolCall(id: "call_write", function: .init(
            name: "write_file", arguments: arguments,
            rawArgumentsJSON: #"{"path":"note.md","content":"007"}"#))
    }

    static func history() -> [Chat.Message] {
        [
            .user("Read note.md, then replace its contents with the inspected identifier 007."),
            .init(role: .assistant, content: "I will inspect the note.",
                reasoningContent: "Inspect the existing file before changing it.", toolCalls: [readCall()]),
            .tool("The inspected identifier is 007.", toolCallId: "call_read"),
            .init(role: .assistant, content: "I will write the inspected identifier.",
                reasoningContent: "The read returned 007; use that exact string.", toolCalls: [writeCall()]),
            .tool("Written 007 to note.md.", toolCallId: "call_write"),
        ]
    }

    @Test func reasoningAndDependentToolMetadataSurvive() throws {
        let messages = Glm5NextMessageGenerator().generate(messages: Self.history())
        #expect(messages[1]["reasoning_content"] as? String == "Inspect the existing file before changing it.")
        #expect(messages[3]["reasoning_content"] as? String == "The read returned 007; use that exact string.")
        for (index, name, id) in [(1, "read_file", "call_read"), (3, "write_file", "call_write")] {
            let calls = try #require(messages[index]["tool_calls"] as? [[String: any Sendable]])
            #expect(calls.count == 1)
            let call = try #require(calls.first)
            #expect(call["id"] as? String == id)
            #expect(call["name"] as? String == name)
            let function = try #require(call["function"] as? [String: any Sendable])
            #expect(function["name"] as? String == name)
        }
        #expect(messages[2]["tool_call_id"] as? String == "call_read")
        #expect(messages[4]["tool_call_id"] as? String == "call_write")
        let function = try #require((messages[3]["tool_calls"] as? [[String: any Sendable]])?.first?["function"] as? [String: any Sendable])
        let arguments = try #require(function["arguments"] as? [String: any Sendable])
        #expect(arguments["content"] as? String == "007")
    }

    @Test func emittedArgumentOrderSurvivesPersistence() throws {
        let direct = Glm5NextMessageGenerator().generate(message: Self.history()[3])
        #expect(direct["_vmlx_tool_argument_orders"] as? [[String]] == [["path", "content"]])
        #expect(direct["_vmlx_raw_tool_arguments_json"] as? [String] == [#"{"path":"note.md","content":"007"}"#])
        let restored = try JSONDecoder().decode(ToolCall.self, from: JSONEncoder().encode(Self.writeCall()))
        #expect(restored.id == "call_write")
        #expect(restored.function.rawArgumentsJSON == nil)
        #expect(restored.function.argumentOrder == ["path", "content"])
        let mapped = Glm5NextMessageGenerator().generate(
            message: .assistant("", toolCalls: [restored]))
        #expect(mapped["_vmlx_tool_argument_orders"] as? [[String]] == [["path", "content"]])
    }

    @Test func mediaItemsPreserveHistoryFieldsAndMarkerOrder() throws {
        let message = Chat.Message(
            role: .assistant, content: "Inspect these attachments.",
            images: [.url(URL(fileURLWithPath: "/fixture/image.png"))],
            videos: [.url(URL(fileURLWithPath: "/fixture/video.mp4"))],
            reasoningContent: "Consider both attachments.", toolCalls: [Self.readCall()])
        let mapped = Glm5NextMessageGenerator().generate(message: message)
        let items = try #require(mapped["content"] as? [[String: String]])
        #expect(items == [["type": "text", "text": message.content], ["type": "image"], ["type": "video"]])
        #expect(mapped["reasoning_content"] as? String == message.reasoningContent)
        #expect((mapped["tool_calls"] as? [[String: any Sendable]])?.first?["id"] as? String == "call_read")
    }
}
