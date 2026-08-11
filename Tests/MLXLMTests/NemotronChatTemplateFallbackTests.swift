// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import VMLXJinja
import MLXLMCommon
import XCTest

/// Nemotron 3.5 Lightning ships its chat template as a standalone
/// `chat_template.jinja` and carries no `chat_template` in
/// `tokenizer_config.json`, so `NemotronMinimal` is what actually renders every
/// tool-enabled turn. Its wire format therefore has to match the bundle's
/// template byte for byte.
///
/// It did not. The generation prompt was written as
///
///     <|im_start|>assistant
///     {%- if enable_thinking %}
///     <think>
///
/// and each `{%-` strips the newline in front of it, so the rendered tail
/// collapsed to `<|im_start|>assistant<think>` where the bundle emits
/// `<|im_start|>assistant\n<think>\n`. The model never saw that spelling in
/// training, so it closed the block on its first token: reasoning was enabled,
/// the block was open, and the turn still produced zero thinking. Measured live
/// before the fix — 637 generated tokens, 0 characters of reasoning, on a
/// prompt that explicitly asked for step-by-step work.
///
/// These pin the exact separators, because the failure was invisible at the
/// "does it contain `<think>`" level of detail.
final class NemotronChatTemplateFallbackTests: XCTestCase {

    private func render(_ context: [String: Any]) throws -> String {
        let template = try Template(ChatTemplateFallbacks.nemotronMinimal)
        var values: [String: Value] = [:]
        for (key, value) in context {
            values[key] = try Value(any: value)
        }
        return try template.render(values)
    }

    private let oneUser: [[String: Any]] = [["role": "user", "content": "hi"]]

    // MARK: - The generation prompt (the reasoning bug)

    func testThinkingOnOpensTheBlockWithTheTrainedSpelling() throws {
        let rendered = try render([
            "messages": oneUser,
            "add_generation_prompt": true,
            "enable_thinking": true,
        ])
        XCTAssertTrue(
            rendered.hasSuffix("<|im_start|>assistant\n<think>\n"),
            rendered.debugDescription)
        // The exact regression: newlines eaten by whitespace control.
        XCTAssertFalse(rendered.contains("assistant<think>"), rendered.debugDescription)
    }

    func testThinkingOffPreClosesTheBlock() throws {
        let rendered = try render([
            "messages": oneUser,
            "add_generation_prompt": true,
            "enable_thinking": false,
        ])
        XCTAssertTrue(
            rendered.hasSuffix("<|im_start|>assistant\n<think></think>"),
            rendered.debugDescription)
        XCTAssertFalse(rendered.contains("assistant<think>"), rendered.debugDescription)
    }

    /// On and off must differ only in the block, never in the header.
    func testBothRailsShareTheSameAssistantHeader() throws {
        for thinking in [true, false] {
            let rendered = try render([
                "messages": oneUser,
                "add_generation_prompt": true,
                "enable_thinking": thinking,
            ])
            XCTAssertTrue(
                rendered.contains("<|im_start|>assistant\n<think>"),
                "thinking=\(thinking): \(rendered.debugDescription)")
        }
    }

    // MARK: - Message separators

    /// The bundle emits `<|im_start|>user\n{content}<|im_end|>\n`. The fallback
    /// used to put a newline *before* `<|im_end|>` and drop the one after it.
    func testUserMessageMatchesBundleSeparators() throws {
        let rendered = try render([
            "messages": oneUser,
            "add_generation_prompt": true,
            "enable_thinking": true,
        ])
        XCTAssertTrue(rendered.contains("<|im_start|>user\nhi<|im_end|>\n"), rendered.debugDescription)
        XCTAssertFalse(rendered.contains("hi\n<|im_end|>"), rendered.debugDescription)
    }

    func testAssistantMessageMatchesBundleSeparators() throws {
        let rendered = try render([
            "messages": [
                ["role": "user", "content": "hi"],
                ["role": "assistant", "content": "hello"],
                ["role": "user", "content": "again"],
            ],
            "add_generation_prompt": true,
            "enable_thinking": true,
        ])
        XCTAssertTrue(
            rendered.contains("<|im_start|>assistant\nhello<|im_end|>\n"),
            rendered.debugDescription)
    }

    func testToolResponseMatchesBundleSeparators() throws {
        let rendered = try render([
            "messages": [
                ["role": "user", "content": "hi"],
                ["role": "tool", "content": "42"],
            ],
            "add_generation_prompt": true,
            "enable_thinking": true,
        ])
        XCTAssertTrue(
            rendered.contains("<|im_start|>user\n<tool_response>\n42\n</tool_response>\n<|im_end|>\n"),
            rendered.debugDescription)
    }

    /// Two `<|im_start|>` markers must never end up glued to the previous
    /// turn's `<|im_end|>` — that was the same stripped-newline defect showing
    /// up between every pair of messages.
    func testNoGluedTurnBoundaries() throws {
        let rendered = try render([
            "messages": [
                ["role": "user", "content": "one"],
                ["role": "assistant", "content": "two"],
                ["role": "user", "content": "three"],
            ],
            "add_generation_prompt": true,
            "enable_thinking": true,
        ])
        XCTAssertFalse(rendered.contains("<|im_end|><|im_start|>"), rendered.debugDescription)
    }

    // MARK: - Tool calls still render

    func testAssistantToolCallRendersFunctionBlock() throws {
        let rendered = try render([
            "messages": [
                ["role": "user", "content": "weather?"],
                [
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        [
                            "function": [
                                "name": "get_weather",
                                "arguments": ["city": "Paris"],
                            ]
                        ]
                    ],
                ],
            ],
            "add_generation_prompt": true,
            "enable_thinking": true,
        ])
        XCTAssertTrue(
            rendered.contains("<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>"),
            rendered.debugDescription)
        XCTAssertTrue(rendered.contains("</tool_call><|im_end|>\n"), rendered.debugDescription)
    }
}
