// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import VMLXJinja
import MLXLMCommon
import XCTest

private extension Template {
    func renderLaguna(_ context: [String: Any]) throws -> String {
        var values: [String: Value] = [:]
        for (key, value) in context {
            values[key] = try Value(any: value)
        }
        return try render(values)
    }
}

final class LagunaChatTemplateFallbackTests: XCTestCase {
    func testLagunaMinimalThinkingOffUsesPoolsideTurns() throws {
        let template = try Template(ChatTemplateFallbacks.lagunaMinimal)
        let rendered = try template.renderLaguna([
            "messages": [
                ["role": "user", "content": "hi"],
            ],
            "add_generation_prompt": true,
            "enable_thinking": false,
        ])

        XCTAssertTrue(
            rendered.hasPrefix("〈|EOS|〉<system>\n\nYou are a helpful"),
            rendered.debugDescription)
        XCTAssertTrue(rendered.contains("<user>\nhi\n</user>\n"), rendered.debugDescription)
        XCTAssertTrue(rendered.hasSuffix("<assistant>\n</think>\n"), rendered.debugDescription)
        XCTAssertFalse(rendered.contains("<|im_start|>"))
    }

    func testLagunaMinimalThinkingOnOpensReasoning() throws {
        let template = try Template(ChatTemplateFallbacks.lagunaMinimal)
        let rendered = try template.renderLaguna([
            "messages": [
                ["role": "user", "content": "hi"],
            ],
            "add_generation_prompt": true,
            "enable_thinking": true,
        ])

        XCTAssertTrue(rendered.hasSuffix("<assistant>\n<think>\n"), rendered.debugDescription)
    }

    func testLagunaMinimalAssistantHistoryPreservesReasoningAndContent() throws {
        let template = try Template(ChatTemplateFallbacks.lagunaMinimal)
        let rendered = try template.renderLaguna([
            "messages": [
                ["role": "user", "content": "hi"],
                [
                    "role": "assistant",
                    "reasoning_content": "brief internal note",
                    "content": "Hello!",
                ],
                ["role": "user", "content": "again"],
            ],
            "add_generation_prompt": true,
            "enable_thinking": false,
        ])

        XCTAssertTrue(
            rendered.contains("<think>\nbrief internal note\n</think>\nHello!\n</assistant>\n"),
            rendered.debugDescription)
        XCTAssertTrue(rendered.contains("<user>\nagain\n</user>\n"), rendered.debugDescription)
        XCTAssertTrue(rendered.hasSuffix("<assistant>\n</think>\n"), rendered.debugDescription)
    }

    func testLagunaRequiredToolChoiceRendersFunctionCallOnlyContract() throws {
        let template = try Template(ChatTemplateFallbacks.lagunaMinimal)
        let rendered = try template.renderLaguna([
            "messages": [
                [
                    "role": "user",
                    "content": "Use the line_count tool on this exact text: red\ngreen\nblue",
                ]
            ],
            "tools": [
                [
                    "type": "function",
                    "function": [
                        "name": "line_count",
                        "description": "Count newline-separated text lines.",
                        "parameters": [
                            "type": "object",
                            "properties": ["text": ["type": "string"]],
                            "required": ["text"],
                        ],
                    ],
                ]
            ],
            "tool_choice": "required",
            "tool_choice_name": "line_count",
            "add_generation_prompt": true,
            "enable_thinking": false,
        ])

        XCTAssertTrue(rendered.contains("<available_tools>"), rendered.debugDescription)
        XCTAssertTrue(rendered.contains("\"name\":\"line_count\""), rendered.debugDescription)
        XCTAssertTrue(rendered.contains("<tool_call>function-name"), rendered.debugDescription)
        XCTAssertTrue(rendered.contains("<arg_key>argument-key</arg_key>"), rendered.debugDescription)
        XCTAssertTrue(rendered.contains("<arg_value>value-of-argument-key</arg_value>"), rendered.debugDescription)
        XCTAssertTrue(rendered.contains("The current assistant response MUST be a function call."), rendered.debugDescription)
        XCTAssertTrue(rendered.contains("Use the `line_count` function."), rendered.debugDescription)
        XCTAssertTrue(rendered.contains("Include every required argument exactly as requested"), rendered.debugDescription)
        XCTAssertFalse(rendered.contains("Reply with prose"), rendered.debugDescription)
        XCTAssertTrue(rendered.hasSuffix("<assistant>\n</think>\n"), rendered.debugDescription)
    }
}
