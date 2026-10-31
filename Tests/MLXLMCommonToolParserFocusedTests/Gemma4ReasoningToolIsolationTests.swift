import Foundation
import Testing

@testable import MLXLMCommon

@Suite("Gemma 4 reasoning does not execute quoted tool history")
struct Gemma4ReasoningToolIsolationTests {
    private let tool: [String: any Sendable] = [
        "type": "function",
        "function": [
            "name": "calculate",
            "parameters": [
                "type": "object",
                "properties": ["expression": ["type": "string"]],
                "required": ["expression"],
            ] as [String: any Sendable],
        ] as [String: any Sendable],
    ]

    private func collect(_ raw: String, size: Int, prefilled: Bool = false)
        -> (reasoning: String, content: String, calls: [ToolCall])
    {
        var parser = ReasoningParser.forPrompt(
            stampName: "harmony", promptTail: prefilled ? "<|channel>thought\n" : nil)!
        let processor = ToolCallProcessor(format: .gemma4, tools: [tool])
        var reasoning = ""
        var content = ""
        var calls: [ToolCall] = []
        func receive(_ events: [Generation]) {
            for event in events {
                switch event {
                case .reasoning(let value): reasoning += value
                case .chunk(let value): content += value
                case .toolCall(let call): calls.append(call)
                default: break
                }
            }
        }
        func route(_ segments: [ReasoningSegment]) {
            for segment in segments {
                switch segment {
                case .reasoning(let value):
                    receive(routeGenerationText(value, channel: .reasoning, through: processor))
                case .content(let value):
                    receive(routeGenerationText(value, channel: .content, through: processor))
                }
            }
        }
        var index = raw.startIndex
        while index < raw.endIndex {
            let end = raw.index(index, offsetBy: size, limitedBy: raw.endIndex) ?? raw.endIndex
            route(parser.feed(String(raw[index ..< end])))
            index = end
        }
        route(parser.flush())
        receive(flushGenerationText(channel: .content, through: processor))
        return (reasoning, content, calls)
    }

    @Test(arguments: [1, 2, 3, 4, 8, 16, 64, 4096])
    func quotedHistoryRemainsReasoning(size: Int) {
        // Reduced from real raw-token output. The old parser executes the
        // historical call; larger chunks also absorb the quoted result as args.
        let thought =
            "The previous tool call was `calculate{expression: \"29*17+5\"}`. "
            + "The response was `{\"ok\":true,\"result\":{\"expression\":\"29*17+5\","
            + "\"result\":\"498\"},\"tool\":\"calculate\"}`. No new calculation is needed."
        let result = collect(
            "<|channel>thought\n" + thought + "<channel|>The previous result was 498.", size: size)
        #expect(result.calls.isEmpty)
        #expect(result.reasoning == thought)
        #expect(result.content == "The previous result was 498.")
    }

    @Test(arguments: [1, 2, 3, 4, 8, 16, 64, 4096])
    func quotedNativeEnvelopeRemainsReasoning(size: Int) {
        let thought =
            "Earlier transport: `<|tool_call>call:calculate{expression:<|\"|>29*17+5<|\"|>}<tool_call|>`."
        let result = collect(
            thought + "<channel|>The previous result was 498.", size: size, prefilled: true)
        #expect(result.calls.isEmpty)
        #expect(result.reasoning == thought)
        #expect(result.content == "The previous result was 498.")
    }

    @Test(arguments: [1, 2, 3, 4, 8, 16, 64, 4096])
    func contentToolStillExecutesAfterQuotedHistory(size: Int) {
        let thought =
            "Previously `call:calculate{expression:<|\"|>29*17+5<|\"|>}` returned 498. Now calculate a new value."
        let result = collect(
            "<|channel>thought\n" + thought + "<channel|>"
                + "<|tool_call>call:calculate{expression:<|\"|>47*19<|\"|>}<tool_call|>", size: size
        )
        #expect(result.reasoning == thought)
        #expect(result.content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        #expect(result.calls.count == 1)
        #expect(result.calls.first?.function.name == "calculate")
        #expect(result.calls.first?.function.arguments["expression"] == .string("47*19"))
    }

    @Test(arguments: [1, 2, 3, 4, 8, 16, 64, 4096])
    func contentBareFallbackRemainsSupported(size: Int) {
        let result = collect("call:calculate{expression:<|\"|>47*19<|\"|>}", size: size)
        #expect(result.calls.count == 1)
        #expect(result.calls.first?.function.arguments["expression"] == .string("47*19"))
    }
}
