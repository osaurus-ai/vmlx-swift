// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLXLMCommon
import Testing

@Suite("Gemma4 tool parser focused contracts")
struct Gemma4ToolCallParserFocusedTests {
    @Test("Gemma4 parser normalizes JSON-style quoted argument keys")
    func parserNormalizesJSONStyleQuotedArgumentKeys() throws {
        let parser = GemmaFunctionParser(
            startTag: "<|tool_call>", endTag: "<tool_call|>", escapeMarker: "<|\"|>")
        let content =
            #"<|tool_call>call:browser_navigate{"url":"https://www.amazon.com/gp/cssb","wait_until":"networkidle"}<tool_call|>"#

        let toolCall = try #require(parser.parse(content: content, tools: nil))

        #expect(toolCall.function.name == "browser_navigate")
        #expect(toolCall.function.arguments["url"] == .string("https://www.amazon.com/gp/cssb"))
        #expect(toolCall.function.arguments["wait_until"] == .string("networkidle"))
        #expect(toolCall.function.arguments[#""url""#] == nil)
        #expect(toolCall.function.arguments[#""wait_until""#] == nil)
    }

    @Test("Gemma4 processor keeps quoted keys schema-addressable")
    func processorKeepsQuotedKeysSchemaAddressable() throws {
        let processor = ToolCallProcessor(format: .gemma4)
        _ = processor.processChunk(
            #"<|tool_call>call:browser_navigate{"url":"https://www.amazon.com/gp/cssb","wait_until":"networkidle"}<tool_call|>"#
        )

        #expect(processor.toolCalls.count == 1)
        let toolCall = try #require(processor.toolCalls.first)
        #expect(toolCall.function.name == "browser_navigate")
        #expect(toolCall.function.arguments["url"] == .string("https://www.amazon.com/gp/cssb"))
        #expect(toolCall.function.arguments["wait_until"] == .string("networkidle"))
        #expect(toolCall.function.arguments[#""url""#] == nil)
    }
}
