// Copyright © 2026 Osaurus AI. All rights reserved.

@testable import MLXLMCommon
import Testing

@Suite("Canonical chat cache boundaries")
struct CanonicalChatCacheBoundariesTests {
    private enum RequiredUserError: Error {
        case missingUser
    }

    private struct BoundaryTokenizer: GenerationPromptControllableTokenizer {
        let breakStablePrefix: Bool

        var bosToken: String? { nil }
        var eosToken: String? { nil }
        var unknownToken: String? { nil }

        func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
        func convertTokenToId(_ token: String) -> Int? { nil }
        func convertIdToToken(_ id: Int) -> String? { nil }

        func applyChatTemplate(
            messages: [[String: any Sendable]],
            tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] {
            try applyChatTemplate(
                messages: messages,
                tools: tools,
                additionalContext: additionalContext,
                addGenerationPrompt: true)
        }

        func applyChatTemplate(
            messages: [[String: any Sendable]],
            tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?,
            addGenerationPrompt: Bool
        ) throws -> [Int] {
            var result = [1]
            let leadingInstructions = messages.prefix {
                let role = $0["role"] as? String
                return role == "system" || role == "developer"
            }
            if breakStablePrefix, messages.count == leadingInstructions.count {
                result = [7]
            }
            for message in leadingInstructions {
                result.append((message["role"] as? String) == "system" ? 10 : 11)
            }
            if tools?.isEmpty == false {
                result.append(20)
            }
            if let thinking = additionalContext?["enable_thinking"] as? Bool {
                result.append(thinking ? 21 : 22)
            }
            for message in messages.dropFirst(leadingInstructions.count) {
                switch message["role"] as? String {
                case "user": result.append(30)
                case "assistant": result.append(40)
                default: result.append(50)
                }
            }
            if addGenerationPrompt {
                result.append(99)
            }
            return result
        }
    }

    private struct ContentBoundaryTokenizer: GenerationPromptControllableTokenizer {
        var bosToken: String? { nil }
        var eosToken: String? { nil }
        var unknownToken: String? { nil }

        func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
        func convertTokenToId(_ token: String) -> Int? { nil }
        func convertIdToToken(_ id: Int) -> String? { nil }

        func applyChatTemplate(
            messages: [[String: any Sendable]],
            tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] {
            try applyChatTemplate(
                messages: messages,
                tools: tools,
                additionalContext: additionalContext,
                addGenerationPrompt: true)
        }

        func applyChatTemplate(
            messages: [[String: any Sendable]],
            tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?,
            addGenerationPrompt: Bool
        ) throws -> [Int] {
            var result = [1]
            for message in messages {
                switch message["role"] as? String {
                case "system":
                    result.append(10)
                    result.append(contentsOf: contentTokens(message))
                case "developer":
                    result.append(11)
                    result.append(contentsOf: contentTokens(message))
                case "user":
                    result.append(30)
                    result.append(contentsOf: contentTokens(message))
                case "assistant":
                    result.append(40)
                    result.append(contentsOf: contentTokens(message))
                default:
                    result.append(50)
                    result.append(contentsOf: contentTokens(message))
                }
            }
            if tools?.isEmpty == false {
                result.append(20)
            }
            if addGenerationPrompt {
                result.append(99)
            }
            return result
        }

        private func contentTokens(_ message: [String: any Sendable]) -> [Int] {
            let content = message["content"] as? String ?? ""
            return content.unicodeScalars.map { 1_000 + Int($0.value) }
        }
    }

    private let tools: [[String: any Sendable]] = [["type": "function"]]

    /// Minimal Qwen 3.5-shaped tokenizer: system/tool-only renders are
    /// rejected, while a user query makes the same leading instruction rail
    /// renderable. The user content token is deliberately different for each
    /// probe so the stable boundary stops before user-controlled text.
    private struct RequiredUserTokenizer: GenerationPromptControllableTokenizer {
        var bosToken: String? { nil }
        var eosToken: String? { nil }
        var unknownToken: String? { nil }

        func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
        func convertTokenToId(_ token: String) -> Int? { nil }
        func convertIdToToken(_ id: Int) -> String? { nil }

        func applyChatTemplate(
            messages: [[String: any Sendable]],
            tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] {
            try applyChatTemplate(
                messages: messages,
                tools: tools,
                additionalContext: additionalContext,
                addGenerationPrompt: true)
        }

        func applyChatTemplate(
            messages: [[String: any Sendable]],
            tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?,
            addGenerationPrompt: Bool
        ) throws -> [Int] {
            guard let user = messages.first(where: { $0["role"] as? String == "user" }),
                  let content = user["content"] as? String
            else {
                throw RequiredUserError.missingUser
            }

            var result = [1]
            if messages.first?["role"] as? String == "system" {
                result.append(10)
            }
            if tools?.isEmpty == false {
                result.append(20)
            }
            result.append(30) // user-turn header
            switch content.first {
            case "0": result.append(60)
            case "z": result.append(61)
            default: result.append(62)
            }
            result.append(31) // user-turn close
            if addGenerationPrompt {
                result.append(99)
            }
            return result
        }
    }

    @Test("stable system and tool prefix is distinct from full history")
    func stableSystemToolPrefix() throws {
        let tokenizer = BoundaryTokenizer(breakStablePrefix: false)
        let messages: [[String: any Sendable]] = [
            ["role": "system", "content": "rules"],
            ["role": "user", "content": "first task"],
        ]
        let prompt = try tokenizer.applyChatTemplate(
            messages: messages, tools: tools, additionalContext: nil)
        let boundaries = canonicalChatCacheBoundaries(
            tokenizer: tokenizer,
            messages: messages,
            tools: tools,
            additionalContext: nil,
            promptTokens: prompt)

        #expect(prompt == [1, 10, 20, 30, 99])
        #expect(boundaries.stable == [3])
        #expect(boundaries.all == [3, 4])
    }

    @Test("tool-only stable prefix is reusable when the template proves it")
    func toolOnlyStablePrefix() throws {
        let tokenizer = BoundaryTokenizer(breakStablePrefix: false)
        let messages: [[String: any Sendable]] = [
            ["role": "user", "content": "first task"]
        ]
        let prompt = try tokenizer.applyChatTemplate(
            messages: messages, tools: tools, additionalContext: nil)
        let boundaries = canonicalChatCacheBoundaries(
            tokenizer: tokenizer,
            messages: messages,
            tools: tools,
            additionalContext: nil,
            promptTokens: prompt)

        #expect(prompt == [1, 20, 30, 99])
        #expect(boundaries.stable == [2])
        #expect(boundaries.all == [2, 3])
    }

    @Test("template rewrites fail closed instead of storing a false prefix")
    func nonPrefixStableRenderIsRejected() throws {
        let tokenizer = BoundaryTokenizer(breakStablePrefix: true)
        let messages: [[String: any Sendable]] = [
            ["role": "system", "content": "rules"],
            ["role": "user", "content": "first task"],
        ]
        let prompt = try tokenizer.applyChatTemplate(
            messages: messages, tools: tools, additionalContext: nil)
        let boundaries = canonicalChatCacheBoundaries(
            tokenizer: tokenizer,
            messages: messages,
            tools: tools,
            additionalContext: nil,
            promptTokens: prompt)

        #expect(boundaries.stable.isEmpty)
        #expect(boundaries.all == [4])
    }

    @Test("a bare BOS without system instructions or tools is not persisted")
    func noStableMaterialDoesNotCreateBoundary() throws {
        let tokenizer = BoundaryTokenizer(breakStablePrefix: false)
        let messages: [[String: any Sendable]] = [
            ["role": "user", "content": "first task"]
        ]
        let prompt = try tokenizer.applyChatTemplate(
            messages: messages, tools: nil, additionalContext: nil)
        let boundaries = canonicalChatCacheBoundaries(
            tokenizer: tokenizer,
            messages: messages,
            tools: nil,
            additionalContext: nil,
            promptTokens: prompt)

        #expect(boundaries.stable.isEmpty)
        #expect(boundaries.all == [2])
    }

    @Test("required-user templates derive a stable prefix without user content")
    func requiredUserTemplateUsesProbeDerivedStablePrefix() throws {
        let tokenizer = RequiredUserTokenizer()
        let messages: [[String: any Sendable]] = [
            ["role": "system", "content": "rules"],
            ["role": "user", "content": "actual request"],
        ]
        let prompt = try tokenizer.applyChatTemplate(
            messages: messages, tools: tools, additionalContext: nil)
        let boundaries = canonicalChatCacheBoundaries(
            tokenizer: tokenizer,
            messages: messages,
            tools: tools,
            additionalContext: nil,
            promptTokens: prompt)

        #expect(prompt == [1, 10, 20, 30, 62, 31, 99])
        #expect(boundaries.stable == [4])
        #expect(boundaries.all == [4, 6])
        #expect(Array(prompt.prefix(try #require(boundaries.stable.first)))
            == [1, 10, 20, 30])
    }

    @Test("static system hint preserves an earlier reusable boundary inside a mutable system message")
    func staticSystemHintAddsEarlierBoundaryInsideMutableSystemMessage() throws {
        let tokenizer = ContentBoundaryTokenizer()
        let staticPrefix = "STATIC PROMPT"
        let messages: [[String: any Sendable]] = [
            ["role": "system", "content": staticPrefix + "\nmutable-db-schema-v1"],
            ["role": "user", "content": "create the next table"],
        ]
        let prompt = try tokenizer.applyChatTemplate(
            messages: messages, tools: tools, additionalContext: nil)
        let withoutHint = canonicalChatCacheBoundaries(
            tokenizer: tokenizer,
            messages: messages,
            tools: tools,
            additionalContext: nil,
            promptTokens: prompt)
        let withHint = canonicalChatCacheBoundaries(
            tokenizer: tokenizer,
            messages: messages,
            tools: tools,
            additionalContext: nil,
            promptTokens: prompt,
            staticSystemPrefix: staticPrefix)

        let hintedBoundary = try #require(withHint.stable.first)
        let fullSystemBoundary = try #require(withoutHint.stable.first)
        #expect(hintedBoundary < fullSystemBoundary)
        #expect(withHint.stable.contains(fullSystemBoundary))
        #expect(withHint.all.contains(hintedBoundary))
        #expect(Array(prompt.prefix(hintedBoundary)).last == 1_010)
    }

    @Test("template configuration changes produce a different stable token prefix")
    func configurationChangeInvalidatesStablePrefix() throws {
        let tokenizer = BoundaryTokenizer(breakStablePrefix: false)
        let messages: [[String: any Sendable]] = [
            ["role": "system", "content": "rules"],
            ["role": "user", "content": "first task"],
        ]
        let enabledContext: [String: any Sendable] = ["enable_thinking": true]
        let disabledContext: [String: any Sendable] = ["enable_thinking": false]
        let enabledPrompt = try tokenizer.applyChatTemplate(
            messages: messages, tools: tools, additionalContext: enabledContext)
        let disabledPrompt = try tokenizer.applyChatTemplate(
            messages: messages, tools: tools, additionalContext: disabledContext)
        let enabled = canonicalChatCacheBoundaries(
            tokenizer: tokenizer,
            messages: messages,
            tools: tools,
            additionalContext: enabledContext,
            promptTokens: enabledPrompt)
        let disabled = canonicalChatCacheBoundaries(
            tokenizer: tokenizer,
            messages: messages,
            tools: tools,
            additionalContext: disabledContext,
            promptTokens: disabledPrompt)

        let enabledBoundary = try #require(enabled.stable.first)
        let disabledBoundary = try #require(disabled.stable.first)
        #expect(Array(enabledPrompt.prefix(enabledBoundary))
            != Array(disabledPrompt.prefix(disabledBoundary)))
    }
}
