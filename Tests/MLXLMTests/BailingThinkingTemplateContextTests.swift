// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

final class BailingThinkingTemplateContextTests: XCTestCase {
    func testEnableThinkingTruePrependsDetailedThinkingOn() {
        let messages: [Message] = [
            ["role": "system", "content": "You are concise."],
            ["role": "user", "content": "hello"],
        ]

        let out = BailingThinkingTemplateContext.apply(
            to: messages,
            modelType: "bailing_hybrid",
            additionalContext: ["enable_thinking": true]
        )

        XCTAssertEqual(out[0]["role"] as? String, "system")
        XCTAssertEqual(out[0]["content"] as? String, "detailed thinking on\n\nYou are concise.")
        XCTAssertEqual(out[1]["content"] as? String, "hello")
    }

    func testEnableThinkingFalseInsertsSystemMessageWhenMissing() {
        let messages: [Message] = [
            ["role": "user", "content": "hello"]
        ]

        let out = BailingThinkingTemplateContext.apply(
            to: messages,
            modelType: "bailing_moe_v2_5",
            additionalContext: ["enable_thinking": false]
        )

        XCTAssertEqual(out.count, 2)
        XCTAssertEqual(out[0]["role"] as? String, "system")
        XCTAssertEqual(out[0]["content"] as? String, "detailed thinking off")
        XCTAssertEqual(out[1]["role"] as? String, "user")
    }

    func testExistingDirectiveIsReplaced() {
        let messages: [Message] = [
            ["role": "system", "content": "detailed thinking on\n\nYou are concise."],
            ["role": "user", "content": "hello"],
        ]

        let out = BailingThinkingTemplateContext.apply(
            to: messages,
            modelType: "bailing_hybrid",
            additionalContext: ["enable_thinking": false]
        )

        XCTAssertEqual(out[0]["content"] as? String, "detailed thinking off\n\nYou are concise.")
    }

    func testNonBailingModelIsUnchanged() {
        let messages: [Message] = [
            ["role": "system", "content": "You are concise."],
            ["role": "user", "content": "hello"],
        ]

        let out = BailingThinkingTemplateContext.apply(
            to: messages,
            modelType: "laguna",
            additionalContext: ["enable_thinking": false]
        )

        XCTAssertEqual(out[0]["content"] as? String, "You are concise.")
        XCTAssertEqual(out.count, messages.count)
    }

    func testMissingEnableThinkingContextIsUnchanged() {
        let messages: [Message] = [
            ["role": "user", "content": "hello"]
        ]

        let out = BailingThinkingTemplateContext.apply(
            to: messages,
            modelType: "bailing_hybrid",
            additionalContext: nil
        )

        XCTAssertEqual(out.count, messages.count)
        XCTAssertEqual(out[0]["role"] as? String, "user")
    }

    func testRequiredToolChoiceDoesNotInsertBailingSystemDirective() {
        let messages: [Message] = [
            ["role": "user", "content": "count lines"]
        ]

        let out = BailingThinkingTemplateContext.apply(
            to: messages,
            modelType: "bailing_hybrid",
            additionalContext: ["tool_choice": "required"]
        )

        XCTAssertEqual(out.count, messages.count)
        XCTAssertEqual(out[0]["role"] as? String, "user")
        XCTAssertEqual(out[0]["content"] as? String, "count lines")
    }

    func testNamedRequiredToolChoiceDoesNotInjectBailingFunctionDirective() {
        let messages: [Message] = [
            ["role": "system", "content": "You are concise."],
            ["role": "user", "content": "count lines"]
        ]

        let out = BailingThinkingTemplateContext.apply(
            to: messages,
            modelType: "bailing_hybrid",
            additionalContext: [
                "tool_choice": "required",
                "tool_choice_name": "line_count",
            ]
        )

        XCTAssertEqual(out.count, messages.count)
        XCTAssertEqual(out[0]["role"] as? String, "system")
        XCTAssertEqual(out[0]["content"] as? String, "You are concise.")
    }

    func testThinkingDirectiveIgnoresRequiredToolChoicePromptCoercion() {
        let messages: [Message] = [
            ["role": "system", "content": "detailed thinking on\n\nYou are concise."],
            ["role": "user", "content": "count lines"],
        ]

        let out = BailingThinkingTemplateContext.apply(
            to: messages,
            modelType: "bailing_moe_v2_5",
            additionalContext: [
                "enable_thinking": false,
                "tool_choice": "required",
            ]
        )

        XCTAssertEqual(
            out[0]["content"] as? String,
            """
            detailed thinking off

            You are concise.
            """
        )
    }
}
