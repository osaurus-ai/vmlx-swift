// Copyright © 2025 Apple Inc.

import Foundation
import Testing

@testable import MLXLMCommon

/// Mistral V7/V11 drops the leading `[TOOL_CALLS]` token on tool turns whose
/// history contains images, emitting the call array as plain text:
/// `[{"name": "get_weather", "arguments": {"city": "Tokyo", "unit": "celsius"}}]`
/// (live repro: mistral-small-3.1-24b, deterministic at temperature 0).
/// These tests cover the bare-JSON-array fallback that routes that shape back
/// into tool calls, and the guards that keep ordinary text byte-exact.
struct MistralBareJSONArrayFallbackTests {

    private let weatherTools: [[String: any Sendable]] = [
        [
            "type": "function",
            "function": ["name": "get_weather"] as [String: any Sendable],
        ]
    ]

    /// Drive the processor with chunks and return the visible text.
    private func drive(
        _ processor: ToolCallProcessor, chunks: [String]
    ) -> String {
        var visible = ""
        for chunk in chunks {
            if let text = processor.processChunk(chunk) {
                visible += text
            }
        }
        if let tail = processor.processEOS() {
            visible += tail
        }
        return visible
    }

    @Test("bare tool-call array streamed in small chunks parses, no visible text")
    func bareArraySmallChunks() throws {
        let processor = ToolCallProcessor(format: .mistral, tools: weatherTools)
        let visible = drive(
            processor,
            chunks: [
                "[", "{\"", "name", "\": \"get_", "weather\", ",
                "\"arguments\": {\"city\": \"Tokyo\", ", "\"unit\": \"celsius\"}}", "]",
            ])

        #expect(visible == "")
        #expect(processor.toolCalls.count == 1)
        let call = try #require(processor.toolCalls.first)
        #expect(call.function.name == "get_weather")
        #expect(call.function.arguments["city"] == .string("Tokyo"))
        #expect(call.function.arguments["unit"] == .string("celsius"))
    }

    @Test("bare tool-call array in a single chunk parses")
    func bareArraySingleChunk() throws {
        let processor = ToolCallProcessor(format: .mistral, tools: weatherTools)
        let visible = drive(
            processor,
            chunks: [
                #"[{"name": "get_weather", "arguments": {"city": "Tokyo", "unit": "celsius"}}]"#
            ])

        #expect(visible == "")
        #expect(processor.toolCalls.count == 1)
        #expect(processor.toolCalls.first?.function.name == "get_weather")
    }

    @Test("leading prose before the bare array stays visible")
    func leadingProseStaysVisible() throws {
        let processor = ToolCallProcessor(format: .mistral, tools: weatherTools)
        let visible = drive(
            processor,
            chunks: [
                "I'll check that for you. ",
                #"[{"name": "get_weather", "arguments": {"city": "Tokyo"}}]"#,
            ])

        #expect(visible == "I'll check that for you. ")
        #expect(processor.toolCalls.count == 1)
        #expect(processor.toolCalls.first?.function.name == "get_weather")
    }

    @Test("multi-call bare array records every call")
    func multiCallArray() throws {
        let processor = ToolCallProcessor(format: .mistral, tools: weatherTools)
        let visible = drive(
            processor,
            chunks: [
                #"[{"name": "get_weather", "arguments": {"city": "Tokyo"}}, "#,
                #"{"name": "get_weather", "arguments": {"city": "Paris"}}]"#,
            ])

        #expect(visible == "")
        #expect(processor.toolCalls.count == 2)
        #expect(processor.toolCalls[0].function.arguments["city"] == .string("Tokyo"))
        #expect(processor.toolCalls[1].function.arguments["city"] == .string("Paris"))
    }

    @Test("JSON example with a non-registered name passes through byte-exact")
    func nonToolJSONPassesThrough() {
        let processor = ToolCallProcessor(format: .mistral, tools: weatherTools)
        let text = #"Here is an example: [{"name": "John", "age": 30}] as requested."#
        let visible = drive(processor, chunks: [text])

        #expect(visible == text)
        #expect(processor.toolCalls.isEmpty)
    }

    @Test("registered-name prefix that keeps going passes through byte-exact")
    func unknownNameWithRegisteredPrefixPassesThrough() {
        let processor = ToolCallProcessor(format: .mistral, tools: weatherTools)
        let text = #"[{"name": "get_weather_hourly", "arguments": {}}]"#
        let visible = drive(processor, chunks: [text])

        #expect(visible == text)
        #expect(processor.toolCalls.isEmpty)
    }

    @Test("markdown links and bracketed prose stay byte-exact")
    func markdownStaysVisible() {
        let processor = ToolCallProcessor(format: .mistral, tools: weatherTools)
        let text = "See [the docs](https://example.com) for details. [Note] Also [1] and [2]."
        let visible = drive(processor, chunks: [text])

        #expect(visible == text)
        #expect(processor.toolCalls.isEmpty)
    }

    @Test("without registered tools the array passes through verbatim")
    func noToolsPassesThrough() {
        let processor = ToolCallProcessor(format: .mistral, tools: nil)
        let text = #"[{"name": "get_weather", "arguments": {"city": "Tokyo"}}]"#
        let visible = drive(processor, chunks: [text])

        #expect(visible == text)
        #expect(processor.toolCalls.isEmpty)
    }

    @Test("tagged [TOOL_CALLS] path still parses (regression)")
    func taggedPathUnchanged() throws {
        let processor = ToolCallProcessor(format: .mistral, tools: weatherTools)
        let visible = drive(
            processor,
            chunks: [
                "[TOOL", "_CALLS]",
                #"[{"name": "get_weather", "arguments": {"city": "Berlin"}}]"#,
            ])

        #expect(visible == "")
        #expect(processor.toolCalls.count == 1)
        #expect(processor.toolCalls.first?.function.arguments["city"] == .string("Berlin"))
    }

    @Test("bracket characters inside argument strings do not end the array early")
    func bracketsInsideStrings() throws {
        let processor = ToolCallProcessor(format: .mistral, tools: weatherTools)
        let visible = drive(
            processor,
            chunks: [
                #"[{"name": "get_weather", "arguments": {"city": "Tokyo ]{[ \" oddity"}}]"#
            ])

        #expect(visible == "")
        #expect(processor.toolCalls.count == 1)
        #expect(
            processor.toolCalls.first?.function.arguments["city"]
                == .string(#"Tokyo ]{[ " oddity"#))
    }

    @Test("EOS while the array is incomplete flushes the held text verbatim")
    func eosMidArrayFlushes() {
        let processor = ToolCallProcessor(format: .mistral, tools: weatherTools)
        let text = #"[{"name": "get_weather", "arguments": {"city":"#
        let visible = drive(processor, chunks: [text])

        #expect(visible == text)
        #expect(processor.toolCalls.isEmpty)
    }

    @Test("whitespace between shape tokens is tolerated")
    func whitespaceTolerated() throws {
        let processor = ToolCallProcessor(format: .mistral, tools: weatherTools)
        let visible = drive(
            processor,
            chunks: [
                "[ \n  { \"name\" : \"get_weather\", \"arguments\": {\"city\": \"Oslo\"} } ]"
            ])

        #expect(visible == "")
        #expect(processor.toolCalls.count == 1)
        #expect(processor.toolCalls.first?.function.arguments["city"] == .string("Oslo"))
    }
}
