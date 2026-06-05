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

    @Test("Gemma4 tool template renders malformed schema types without crashing")
    func gemma4ToolTemplateRendersMalformedSchemaTypesWithoutCrashing() throws {
        let tool: ToolSpec = [
            "type": "function",
            "function": [
                "name": "schema_probe",
                "description": "Probe schema normalization for Gemma4 templates.",
                "parameters": [
                    "type": ["object", "null"] as [any Sendable],
                    "properties": [
                        "query": [
                            "type": NSNull(),
                            "description": "Search or URL query.",
                        ] as [String: any Sendable],
                        "format": [
                            "type": ["string", "integer"] as [any Sendable],
                            "description": "Preferred output format.",
                        ] as [String: any Sendable],
                        "nested": [
                            "properties": [
                                "type": [
                                    "type": "string",
                                    "description": "A literal property named type.",
                                ] as [String: any Sendable],
                            ] as [String: any Sendable],
                        ] as [String: any Sendable],
                    ] as [String: any Sendable],
                    "required": ["query"],
                ] as [String: any Sendable],
            ] as [String: any Sendable],
        ]

        let normalized = try #require(normalizedToolsForChatTemplate([tool])?.first)
        let function = try #require(normalized["function"] as? [String: any Sendable])
        let parameters = try #require(function["parameters"] as? [String: any Sendable])
        #expect(parameters["type"] as? String == "object")
        #expect(parameters["nullable"] as? Bool == true)

        let properties = try #require(parameters["properties"] as? [String: any Sendable])
        let query = try #require(properties["query"] as? [String: any Sendable])
        let format = try #require(properties["format"] as? [String: any Sendable])
        let nested = try #require(properties["nested"] as? [String: any Sendable])
        let nestedProperties = try #require(nested["properties"] as? [String: any Sendable])
        let literalTypeProperty = try #require(
            nestedProperties["type"] as? [String: any Sendable])

        #expect(query["type"] as? String == "string")
        #expect(query["nullable"] as? Bool == true)
        #expect(format["type"] as? String == "string")
        #expect(nested["type"] as? String == "object")
        #expect(literalTypeProperty["type"] as? String == "string")

        let rendered = try renderGemma4([
            "messages": [
                [
                    "role": "user",
                    "content": "Can you download this youtube video?",
                ] as [String: any Sendable],
            ] as [any Sendable],
            "tools": [normalized] as [any Sendable],
            "tool_choice": "required",
            "add_generation_prompt": true,
            "enable_thinking": false,
            "bos_token": "<bos>",
        ])

        #expect(rendered.contains("declaration:schema_probe"))
        #expect(rendered.contains("query"))
        #expect(rendered.contains("nested"))
        #expect(!rendered.contains("upper filter requires string"))
        #expect(!rendered.contains("Cannot convert value of type Optional<Any> to Jinja Value"))
    }

    @Test("Gemma4 tool template drops boolean additionalProperties in nested schemas")
    func gemma4ToolTemplateDropsBooleanAdditionalPropertiesInNestedSchemas() throws {
        let tool: ToolSpec = [
            "type": "function",
            "function": [
                "name": "db_update",
                "description": "Update rows in a local table.",
                "parameters": [
                    "type": "object",
                    "properties": [
                        "where": [
                            "type": "object",
                            "description": "Filter object.",
                            "additionalProperties": true,
                        ] as [String: any Sendable],
                        "set": [
                            "type": "object",
                            "description": "Update values.",
                            "additionalProperties": false,
                        ] as [String: any Sendable],
                        "typed": [
                            "type": "object",
                            "additionalProperties": [
                                "type": "string"
                            ] as [String: any Sendable],
                        ] as [String: any Sendable],
                    ] as [String: any Sendable],
                    "required": ["where", "set"],
                    "additionalProperties": false,
                ] as [String: any Sendable],
            ] as [String: any Sendable],
        ]

        let normalized = try #require(normalizedToolsForChatTemplate([tool])?.first)
        let function = try #require(normalized["function"] as? [String: any Sendable])
        let parameters = try #require(function["parameters"] as? [String: any Sendable])
        let properties = try #require(parameters["properties"] as? [String: any Sendable])
        let whereSchema = try #require(properties["where"] as? [String: any Sendable])
        let setSchema = try #require(properties["set"] as? [String: any Sendable])
        let typedSchema = try #require(properties["typed"] as? [String: any Sendable])

        #expect(parameters["additionalProperties"] == nil)
        #expect(whereSchema["additionalProperties"] == nil)
        #expect(setSchema["additionalProperties"] == nil)
        #expect(typedSchema["additionalProperties"] is [String: any Sendable])

        let rendered = try renderGemma4([
            "messages": [
                [
                    "role": "user",
                    "content": "Update the row where id is 1.",
                ] as [String: any Sendable],
            ] as [any Sendable],
            "tools": [normalized] as [any Sendable],
            "tool_choice": "required",
            "add_generation_prompt": true,
            "enable_thinking": false,
            "bos_token": "<bos>",
        ])

        #expect(rendered.contains("declaration:db_update"))
        #expect(rendered.contains("where"))
        #expect(rendered.contains("set"))
        #expect(!rendered.contains("upper filter requires string"))
        #expect(!rendered.contains("Cannot convert value of type Optional<Any> to Jinja Value"))
    }
}
