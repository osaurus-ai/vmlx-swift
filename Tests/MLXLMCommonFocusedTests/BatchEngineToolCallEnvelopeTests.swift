import Foundation
import Testing

@testable import MLXLMCommon

/// `finishSlot` stores the post-answer cache boundary for tool-enabled
/// requests unless the generation emitted a tool call. The decision replays
/// the stream bridge's parser pipeline over the generated text, so it must
/// agree with what the consumer saw for every family, including the inline
/// and bare fallbacks that carry no envelope tag.
@Suite("BatchEngine finished-generation tool-call detection")
struct BatchEngineToolCallEnvelopeTests {
    private static let weatherTool: [ToolSpec] = {
        let city: [String: any Sendable] = ["type": "string"]
        let parameters: [String: any Sendable] = [
            "type": "object",
            "properties": ["city": city] as [String: any Sendable],
            "required": ["city"] as [String],
        ]
        let function: [String: any Sendable] = [
            "name": "get_weather",
            "description": "Weather for a city",
            "parameters": parameters,
        ]
        return [["type": "function", "function": function]]
    }()

    private static let taggedFormats: [ToolCallFormat] = [
        .json, .glm4, .xmlFunction, .gemma4, .lfm2, .minimaxM2, .minicpm5, .hunyuan, .mistral,
        .step, .nemotron, .kimiK2,
    ]

    @Test("plain prose never counts as a tool call")
    func plainAnswerIsNotAToolCall() {
        for format in Self.taggedFormats {
            #expect(
                !BatchEngine.generatedTextEmitsToolCall(
                    text: "The weather in Berlin is 17°C with light rain.",
                    format: format, tools: Self.weatherTool, reasoningParser: nil),
                "\(format) must not see a tool call in plain prose")
        }
    }

    @Test("tagged envelopes are detected per family")
    func envelopeIsDetected() {
        let samples: [(ToolCallFormat, String)] = [
            (
                .json,
                "<tool_call>{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Berlin\"}}</tool_call>"
            ),
            (
                .glm4,
                "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Berlin</arg_value></tool_call>"
            ),
            (
                .xmlFunction,
                "<tool_call><function=get_weather><parameter=city>Berlin</parameter></function></tool_call>"
            ),
            (.gemma4, "<|tool_call>call:get_weather{city:<|\"|>Berlin<|\"|>}<tool_call|>"),
            (.lfm2, "<|tool_call_start|>[get_weather(city='Berlin')]<|tool_call_end|>"),
            (
                .minimaxM2,
                "<minimax:tool_call><invoke name=\"get_weather\"><parameter name=\"city\">Berlin</parameter></invoke></minimax:tool_call>"
            ),
            (
                .minicpm5,
                "<function name=\"get_weather\"><param name=\"city\"><![CDATA[Berlin]]></param></function>"
            ),
            (
                .hunyuan,
                "<tool_calls>\n<tool_call>get_weather<tool_sep>\n<arg_key>city</arg_key><arg_value>Berlin</arg_value>\n</tool_call>\n</tool_calls>"
            ),
        ]
        for (format, text) in samples {
            #expect(
                BatchEngine.generatedTextEmitsToolCall(
                    text: "Sure. " + text, format: format, tools: Self.weatherTool,
                    reasoningParser: nil),
                "\(format) must detect its envelope")
        }
    }

    @Test("bare and inline fallbacks without an envelope tag are detected")
    func bareFallbacksAreDetected() {
        let samples: [(ToolCallFormat, String)] = [
            (.gemma4, "call:get_weather{city:<|\"|>Berlin<|\"|>}"),
            (.lfm2, "[get_weather(city=\"Berlin\")]"),
            (.mistral, "[{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Berlin\"}}]"),
        ]
        for (format, text) in samples {
            #expect(
                BatchEngine.generatedTextEmitsToolCall(
                    text: text, format: format, tools: Self.weatherTool, reasoningParser: nil),
                "\(format) must detect a bare call, exactly as the stream bridge parses it")
        }
    }

    @Test("a call-shaped mention inside reasoning follows the format's reasoning-channel rule")
    func reasoningMentionFollowsFormatRule() {
        let parser = ReasoningParser(startTag: "<think>", endTag: "</think>")
        // MiniCPM5 never parses tool calls out of reasoning, so an example
        // inside <think> stays prose and the boundary is kept.
        let minicpm =
            "<think>I could emit <function name=\"get_weather\"><param name=\"city\"><![CDATA[Berlin]]></param></function> but the user just wants a chat.</think>It is 17°C in Berlin."
        #expect(
            !BatchEngine.generatedTextEmitsToolCall(
                text: minicpm, format: .minicpm5, tools: Self.weatherTool, reasoningParser: parser))
        // The JSON contract parses the reasoning channel too, so the stream
        // bridge would surface this as a `.toolCall`; the replay must agree.
        let json =
            "<think>Let me call <tool_call>{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Berlin\"}}</tool_call></think>Done."
        #expect(
            BatchEngine.generatedTextEmitsToolCall(
                text: json, format: .json, tools: Self.weatherTool, reasoningParser: parser))
    }

    @Test("a call after a closed reasoning block is detected")
    func callAfterReasoningIsDetected() {
        let text =
            "<think>Need the weather.</think><tool_call>{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Berlin\"}}</tool_call>"
        let parser = ReasoningParser(startTag: "<think>", endTag: "</think>")
        #expect(
            BatchEngine.generatedTextEmitsToolCall(
                text: text, format: .json, tools: Self.weatherTool, reasoningParser: parser))
    }

    @Test("empty generations never count")
    func emptyIsFalse() {
        #expect(
            !BatchEngine.generatedTextEmitsToolCall(
                text: "", format: .json, tools: Self.weatherTool, reasoningParser: nil))
    }
}
