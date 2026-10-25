// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLXHuggingFace
import MLXLMCommon
import XCTest
@preconcurrency import VMLXTokenizers

/// Tokenizer-only integration: no weights or generation. Compare the runtime
/// bridge with the actual bundle template, not an app-authored expected prompt.
final class Gemma4NativeTemplateContractTests: XCTestCase {
    func testValidBundleTemplateOwnsRequiredNamedAndAutomaticToolTurns() async throws {
        let environment = ProcessInfo.processInfo.environment
        if environment["VMLX_CHAT_TEMPLATE_OVERRIDE"]?.isEmpty == false {
            throw XCTSkip("This native-template row requires no explicit template override.")
        }
        let path = environment["VMLX_GEMMA4_TEST_TOKENIZER"]
            ?? "/Users/eric/models/dealign.ai/Gemma-4-26B-A4B-it-JANG_4M-CRACK"
        let directory = URL(fileURLWithPath: path)
        guard FileManager.default.fileExists(atPath: directory.appendingPathComponent("tokenizer.json").path),
            FileManager.default.fileExists(atPath: directory.appendingPathComponent("chat_template.jinja").path)
        else { throw XCTSkip("Local Gemma4 bundle tokenizer/template unavailable; no download performed.") }

        let native = try await AutoTokenizer.from(modelFolder: directory)
        let loader = #huggingFaceTokenizerLoader()
        let bridge = try await loader.load(from: directory)
        let controllable = try XCTUnwrap(bridge as? any GenerationPromptControllableTokenizer)
        let messages: [[String: any Sendable]] = [
            ["role": "user", "content": "Earlier user text."],
            ["role": "assistant", "content": "Earlier assistant answer."],
            ["role": "user", "content": "Use the line_count tool on this exact text: red\ngreen\nblue"],
        ]
        let tools: [[String: any Sendable]] = [[
            "type": "function",
            "function": ["name": "line_count", "description": "Count lines.",
                "parameters": ["type": "object", "properties": ["text": ["type": "string"]],
                    "required": ["text"]] as [String: any Sendable]] as [String: any Sendable],
        ]]
        for choice in ["auto", "required"] {
            for named in [false, true] where choice == "required" || !named {
                var context: [String: any Sendable] = ["enable_thinking": false, "tool_choice": choice]
                if named { context["tool_choice_name"] = "line_count" }
                for addPrompt in [false, true] {
                    let expected = try native.applyChatTemplate(messages: messages, chatTemplate: nil,
                        addGenerationPrompt: addPrompt, truncation: false, maxLength: nil,
                        tools: normalizedToolsForChatTemplate(tools), additionalContext: context)
                    let actual = try controllable.applyChatTemplate(messages: messages, tools: tools,
                        additionalContext: context, addGenerationPrompt: addPrompt)
                    XCTAssertEqual(actual, expected,
                        "A valid bundle template must not be replaced when tools are offered (choice=\(choice), named=\(named), addPrompt=\(addPrompt)).")
                    let decoded = bridge.decode(tokenIds: actual, skipSpecialTokens: false)
                    XCTAssertTrue(decoded.contains("Earlier assistant answer."))
                    XCTAssertTrue(decoded.contains("Use the line_count tool on this exact text: red\ngreen\nblue"))
                    XCTAssertFalse(decoded.contains("Tool use is REQUIRED"))
                    XCTAssertFalse(decoded.contains("Required call shape for the current request"))
                }
            }
        }
    }
}
