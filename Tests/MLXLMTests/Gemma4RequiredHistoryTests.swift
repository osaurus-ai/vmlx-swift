// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import CoreImage
import Foundation
import MLX
import Testing

@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

/// Captures the real processor's template input. Emits one image placeholder
/// per image part so the real pixel preparation/soft-token expansion is tested.
private final class HistoryTokenizer: Tokenizer, @unchecked Sendable {
    var firstMessages: [Message]?
    var firstContext: [String: any Sendable]?
    var firstTools: [ToolSpec]?
    static let imageID = 258880

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        text.utf8.map { Int($0) + 100 }
    }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
    func convertTokenToId(_ token: String) -> Int? {
        token == "<|image|>" ? Self.imageID : nil
    }
    func convertIdToToken(_ id: Int) -> String? { nil }
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }

    func applyChatTemplate(
        messages: [Message], tools: [ToolSpec]?, additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        if firstMessages == nil {
            firstMessages = messages
            firstContext = additionalContext
            firstTools = tools
        }
        return messages.flatMap { message -> [Int] in
            var tokens = encode(text: message["role"] as? String ?? "", addSpecialTokens: false)
            if let content = message["content"] as? String {
                tokens += encode(text: content, addSpecialTokens: false)
            } else if let parts = message["content"] as? [[String: any Sendable]] {
                for part in parts {
                    if part["type"] as? String == "image" { tokens.append(Self.imageID) }
                    if let text = part["text"] as? String {
                        tokens += encode(text: text, addSpecialTokens: false)
                    }
                }
            }
            return tokens
        }
    }
}

@Suite("Gemma required tool history", .serialized)
struct Gemma4RequiredHistoryTests {
    private static func context(_ choice: String) -> [String: any Sendable] {
        var result: [String: any Sendable] = ["enable_thinking": false, "tool_choice": choice]
        if choice == "named" {
            result["tool_choice"] = "required"
            result["tool_choice_name"] = "record_visual"
        }
        return result
    }

    private static var tools: [ToolSpec] {
        [
            [
                "type": "function",
                "function": [
                    "name": "record_visual",
                    "parameters": [
                        "type": "object", "properties": ["description": ["type": "string"]],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ]
        ]
    }

    private static func json(_ value: Any) throws -> Data {
        try JSONSerialization.data(withJSONObject: value, options: [.sortedKeys])
    }

    @Test(
        arguments: ["auto", "required", "named"],
        [
            "text", "earlier-image", "new-image", "repeated-image", "tool-image", "pending-tool",
            "pending-tool-image", "unanswered-tool", "unanswered-tool-image",
        ])
    func vlmHistory(choice: String, scenario: String) async throws {
        let mlxTestLock = lockSerializedMLXTest()
        _ = mlxTestLock
        let red = UserInput.Image.ciImage(
            CIImage(color: CIColor(red: 1, green: 0, blue: 0)).cropped(
                to: CGRect(x: 0, y: 0, width: 4, height: 4)))
        let blue = UserInput.Image.ciImage(
            CIImage(color: CIColor(red: 0, green: 0, blue: 1)).cropped(
                to: CGRect(x: 0, y: 0, width: 4, height: 4)))
        let call = ToolCall(
            id: "call_visual",
            function: .init(
                name: "record_visual", arguments: ["description": .string("original observation")],
                rawArgumentsJSON: #"{"description":"original observation"}"#))
        let earlierImages =
            ["earlier-image", "new-image", "repeated-image"].contains(scenario) ? [red] : []
        let newImages = scenario == "new-image" ? [blue] : scenario == "repeated-image" ? [red] : []
        var chat: [Chat.Message] = [
            .system("CURRENT SETTINGS. Keep the original observations."),
            .user("Original task and image.", images: earlierImages),
            .init(
                role: .assistant, content: "", reasoningContent: "Original reasoning.",
                toolCalls: [call]),
            .init(
                role: .tool, content: "Exact tool result.",
                images: ["tool-image", "pending-tool-image", "unanswered-tool-image"].contains(
                    scenario) ? [blue] : [],
                toolCallId: "call_visual"),
            .assistant("The first task is complete."),
            .user("Use the earlier result for this next task.", images: newImages),
        ]
        if scenario.hasPrefix("pending-tool") { chat.removeLast(2) }
        if scenario.hasPrefix("unanswered-tool") { chat.remove(at: 4) }
        let input = UserInput(
            chat: chat, tools: Self.tools, additionalContext: Self.context(choice))
        let config = try JSONDecoder().decode(
            Gemma4ProcessorConfiguration.self,
            from: Data(
                #"""
                {"processor_class":"Gemma4Processor","patch_size":2,"max_soft_tokens":4,
                 "pooling_kernel_size":1,"image_seq_length":4,"audio_seq_length":4}
                """#.utf8))
        let tokenizer = HistoryTokenizer()
        let prepared = try await Gemma4Processor(config, tokenizer: tokenizer).prepare(input: input)
        let messages = try #require(tokenizer.firstMessages)
        #expect(messages.count == chat.count)
        #expect(messages.map { $0["role"] as? String } == chat.map { Optional($0.role.rawValue) })
        #expect(try Self.json(tokenizer.firstContext ?? [:]) == Self.json(Self.context(choice)))
        #expect(try Self.json(tokenizer.firstTools ?? []) == Self.json(Self.tools))
        // Compare each message, including tool ids, raw argument order, reasoning,
        // scalar system content and image parts. Never replace results by prose.
        for (index, message) in chat.enumerated() where index < messages.count {
            var expected = defaultMessageDict(for: message)
            if !message.images.isEmpty {
                expected["content"] =
                    message.images.map { _ in ["type": "image"] }
                    + [["type": "text", "text": message.content]]
            }
            if message.role == .tool { expected["name"] = "record_visual" }
            #expect(try Self.json(messages[index]) == Self.json(expected))
        }
        let imageCount = input.images.count
        #expect(
            prepared.text.tokenIds?.filter { $0 == HistoryTokenizer.imageID }.count == imageCount
                * 4)
        if imageCount == 0 {
            #expect(prepared.image == nil)
        } else {
            let image = try #require(prepared.image)
            #expect(image.frames?.count == imageCount)
            #expect(image.pixels.dim(0) == imageCount)
            #expect(image.pixels.dim(1) == 3)
            let pixels = image.pixels.asArray(Float.self)
            let stride = image.pixels.dim(2) * image.pixels.dim(3)
            let firstIsRed = !["tool-image", "pending-tool-image", "unanswered-tool-image"]
                .contains(scenario)
            // CoreImage's sRGB conversion returns 0.99999994 for a solid 1.0
            // channel on the proof host. Keep slot/count checks exact; allow
            // only float conversion noise for the pixel ordering checks.
            #expect(abs(pixels[0] - (firstIsRed ? 1 : 0)) < 0.000001)
            #expect(abs(pixels[2 * stride] - (firstIsRed ? 0 : 1)) < 0.000001)
            if imageCount == 2 {
                #expect(abs(pixels[3 * stride] - (scenario == "repeated-image" ? 1 : 0)) < 0.000001)
                #expect(abs(pixels[5 * stride] - (scenario == "repeated-image" ? 0 : 1)) < 0.000001)
            }
        }
    }

    @Test(arguments: ["auto", "required", "named"], ["gemma4_text", "gemma4_unified_text"])
    func textHistory(choice: String, modelType: String) throws {
        let mlxTestLock = lockSerializedMLXTest()
        _ = mlxTestLock
        let tokenizer = HistoryTokenizer()
        let messages: [Message] = [
            ["role": "system", "content": "Original settings."],
            ["role": "developer", "content": "Original constraints."],
            ["role": "user", "content": "Original task.", "opaque": ["retained": true]],
            [
                "role": "assistant", "content": "", "reasoning_content": "Prior reasoning.",
                "tool_calls": [
                    [
                        "id": "first", "type": "function",
                        "function": [
                            "name": "record_visual",
                            "arguments": ["description": "original"],
                        ] as [String: any Sendable],
                    ] as [String: any Sendable]
                ],
            ],
            [
                "role": "tool", "content": "Exact result.", "tool_call_id": "first",
                "name": "record_visual",
            ],
            ["role": "assistant", "content": "Original answer."],
            ["role": "user", "content": "Follow up using that result."],
        ]
        let processor = LLMUserInputProcessor(
            tokenizer: tokenizer,
            configuration: .init(id: "fixture/gemma"), modelType: modelType,
            messageGenerator: DefaultMessageGenerator(),
            defaultAdditionalContext: ["enable_thinking": true])
        _ = try processor.prepare(
            input: UserInput(
                messages: messages, tools: Self.tools,
                additionalContext: Self.context(choice)))
        #expect(try Self.json(tokenizer.firstMessages ?? []) == Self.json(messages))
        #expect(try Self.json(tokenizer.firstContext ?? [:]) == Self.json(Self.context(choice)))
        #expect(try Self.json(tokenizer.firstTools ?? []) == Self.json(Self.tools))
    }
}
