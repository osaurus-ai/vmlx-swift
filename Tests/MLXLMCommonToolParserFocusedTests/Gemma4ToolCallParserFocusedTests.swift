// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLXLMCommon
import Testing
import VMLXJinja

private func renderGemma4(_ context: [String: any Sendable]) throws -> String {
    var values: [String: Value] = [:]
    for (key, value) in context {
        values[key] = try Value(any: value)
    }
    return try Template(ChatTemplateFallbacks.gemma4WithTools).render(values)
}

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

    @Test("Gemma4 tool template renders browser schema with nullable enum")
    func gemma4ToolTemplateRendersBrowserSchemaWithNullableEnum() throws {
        let tool: ToolSpec = [
            "type": "function",
            "function": [
                "name": "browser_navigate",
                "description": "Navigate a browser page.",
                "parameters": [
                    "type": "object",
                    "properties": [
                        "url": [
                            "type": "string",
                            "description": "URL to navigate to.",
                        ] as [String: any Sendable],
                        "wait_until": [
                            "type": ["string", "null"] as [any Sendable],
                            "description": "Optional load state.",
                            "enum": ["load", "domcontentloaded", "networkidle", NSNull()] as [any Sendable],
                        ] as [String: any Sendable],
                    ] as [String: any Sendable],
                    "required": ["url"],
                    "additionalProperties": false,
                ] as [String: any Sendable],
            ] as [String: any Sendable],
        ]

        let normalized = try #require(normalizedToolsForChatTemplate([tool])?.first)
        let function = try #require(normalized["function"] as? [String: any Sendable])
        let parameters = try #require(function["parameters"] as? [String: any Sendable])
        let properties = try #require(parameters["properties"] as? [String: any Sendable])
        let waitUntil = try #require(properties["wait_until"] as? [String: any Sendable])
        #expect(waitUntil["type"] as? String == "string")
        #expect(waitUntil["nullable"] as? Bool == true)
        let enumValues = try #require(waitUntil["enum"] as? [any Sendable])
        #expect(enumValues.count == 3)
        #expect(!enumValues.contains { $0 is NSNull })

        let rendered = try renderGemma4([
            "messages": [
                [
                    "role": "user",
                    "content": "Download https://www.youtube.com/watch?v=e5DF8CxhyAE",
                ] as [String: any Sendable],
            ] as [any Sendable],
            "tools": [normalized] as [any Sendable],
            "tool_choice": "required",
            "add_generation_prompt": true,
            "enable_thinking": false,
            "bos_token": "<bos>",
        ])

        #expect(rendered.contains("declaration:browser_navigate"))
        #expect(rendered.contains("wait_until"))
        #expect(rendered.contains(#"type:<|"|>string<|"|>"#))
        #expect(!rendered.contains("upper filter requires string"))
        #expect(!rendered.contains("Cannot convert value of type Optional<Any> to Jinja Value"))
    }
}
