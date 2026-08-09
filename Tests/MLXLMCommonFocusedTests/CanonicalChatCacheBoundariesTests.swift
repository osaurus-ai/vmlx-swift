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

    /// DSV4-shaped: the generation rail is appended only after a
    /// user/developer turn, so a transcript ending in an assistant turn (an
    /// agent-loop continuation) renders identically with and without the
    /// generation prompt.
    private struct ContinuationRailTokenizer: GenerationPromptControllableTokenizer {
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
                    // Tool schemas render inside the system turn, as they do in
                    // every real template — keeping them here is what makes an
                    // earlier message list a true token prefix of a later one.
                    if tools?.isEmpty == false { result.append(20) }
                case "user": result.append(30)
                case "assistant": result.append(40)
                default: result.append(50)
                }
            }
            // The rail exists only when an assistant turn is being OPENED.
            let lastRole = messages.last?["role"] as? String
            if addGenerationPrompt, lastRole != "assistant", lastRole != "tool" {
                result.append(99)
            }
            return result
        }
    }

    @Test("a trailing assistant continuation still publishes a history boundary")
    func trailingAssistantContinuationPublishesHistoryBoundary() {
        let tokenizer = ContinuationRailTokenizer()
        let messages: [[String: any Sendable]] = [
            ["role": "system", "content": "instructions"],
            ["role": "user", "content": "build it"],
            ["role": "assistant", "content": "working"],
        ]
        let promptTokens = try! tokenizer.applyChatTemplate(
            messages: messages, tools: tools, additionalContext: nil,
            addGenerationPrompt: true)

        let boundaries = canonicalChatCacheBoundaries(
            tokenizer: tokenizer,
            messages: messages,
            tools: tools,
            additionalContext: nil,
            promptTokens: promptTokens)

        // Without the continuation fallback the whole-list render equals the
        // prompt, the strict `<` rejects it, and `all` collapses to the stable
        // prefix only — which is what made every DSV4 agent-loop turn
        // cold-prefill the entire growing transcript.
        let history = boundaries.all.filter { !boundaries.stable.contains($0) }
        #expect(!history.isEmpty, "history boundary was dropped for a continuation turn")

        // Whatever it published must still be a real, strictly shorter,
        // exact token prefix of the prompt.
        for boundary in boundaries.all {
            #expect(boundary < promptTokens.count)
            let rendered = try! tokenizer.applyChatTemplate(
                messages: Array(messages.dropLast()), tools: tools,
                additionalContext: nil, addGenerationPrompt: false)
            if boundary == rendered.count {
                #expect(promptTokens.prefix(boundary).elementsEqual(rendered))
            }
        }
    }

    @Test("a transcript ending in a tool result still publishes a history boundary")
    func trailingToolResultPublishesHistoryBoundary() {
        // The post-turn re-warm after a TOOL call ends with the tool result,
        // not with an assistant turn. Live DSV4 rows showed those re-warms
        // missing in a fixed pattern — stored N, next fetch asks N+2, MISS:
        //   store 2084 -> fetch 2086 MISS
        //   store 2286 -> fetch 2288 MISS
        //   store 2756 -> fetch 2758 MISS
        // while the visible turn that followed restored the same entry fine.
        // If a format appends no rail for this shape either, the same strict
        // `<` drops the boundary, so the fallback must cover it.
        let tokenizer = ContinuationRailTokenizer()
        let messages: [[String: any Sendable]] = [
            ["role": "system", "content": "instructions"],
            ["role": "user", "content": "write a file"],
            ["role": "assistant", "content": "calling the tool"],
            ["role": "tool", "content": "{\"ok\":true}"],
        ]
        let promptTokens = try! tokenizer.applyChatTemplate(
            messages: messages, tools: tools, additionalContext: nil,
            addGenerationPrompt: true)

        let boundaries = canonicalChatCacheBoundaries(
            tokenizer: tokenizer,
            messages: messages,
            tools: tools,
            additionalContext: nil,
            promptTokens: promptTokens)

        let history = boundaries.all.filter { !boundaries.stable.contains($0) }
        #expect(!history.isEmpty, "history boundary was dropped for a tool-result turn")
        for boundary in boundaries.all { #expect(boundary < promptTokens.count) }
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

    /// Builds a long alternating transcript: one 600-scalar system message
    /// plus `turns` further 600-scalar messages. `ContentBoundaryTokenizer`
    /// emits one token per scalar, so each message is ~601 tokens — wide
    /// enough to clear the ladder's 512-token minimum gap.
    private func longTranscript(turns: Int) -> [[String: any Sendable]] {
        var messages: [[String: any Sendable]] = [
            ["role": "system", "content": String(repeating: "s", count: 600)]
        ]
        for index in 0..<turns {
            messages.append([
                "role": index.isMultiple(of: 2) ? "user" : "assistant",
                "content": String(repeating: index.isMultiple(of: 2) ? "u" : "a", count: 600)
                    + String(index),
            ])
        }
        return messages
    }

    /// Every published boundary must be reproducible as the exact token
    /// render of some message prefix. This is the invariant the ladder must
    /// not weaken: more rungs, but never a rung that isn't byte-identical.
    private func assertEveryBoundaryIsAnExactMessagePrefix(
        _ boundaries: CanonicalChatCacheBoundaries,
        tokenizer: some GenerationPromptControllableTokenizer,
        messages: [[String: any Sendable]],
        promptTokens: [Int]
    ) {
        for boundary in boundaries.all {
            #expect(boundary > 0)
            #expect(boundary < promptTokens.count)
            let matchesSomeMessagePrefix = (0...messages.count).contains { count in
                guard let rendered = try? tokenizer.applyChatTemplate(
                    messages: Array(messages.prefix(count)), tools: nil,
                    additionalContext: nil, addGenerationPrompt: false),
                    rendered.count == boundary
                else { return false }
                return promptTokens.prefix(boundary).elementsEqual(rendered)
            }
            #expect(
                matchesSomeMessagePrefix,
                "a published boundary is not an exact render of any message prefix")
        }
    }

    @Test("intermediate history rungs are published between the stable prefix and the newest turn")
    func historyLadderFillsTheGapAboveTheStablePrefix() throws {
        // Live DSV4 published all=[1114, 2643, 5434] for a 5436-token prompt:
        // a 2.8k-token span with no stored entry. Two sends in that session
        // fell back to the frozen 2643 stable prefix and re-prefilled 1948 and
        // 2044 tokens that were still valid prefixes.
        let tokenizer = ContentBoundaryTokenizer()
        let messages = longTranscript(turns: 10)
        let promptTokens = try tokenizer.applyChatTemplate(
            messages: messages, tools: nil, additionalContext: nil,
            addGenerationPrompt: true)

        let boundaries = canonicalChatCacheBoundaries(
            tokenizer: tokenizer,
            messages: messages,
            tools: nil,
            additionalContext: nil,
            promptTokens: promptTokens)

        let stableFloor = try #require(boundaries.stable.last)
        let history = boundaries.all.filter { !boundaries.stable.contains($0) }
        let historyTop = try #require(history.max())

        // The regression: a single history boundary leaves the whole span
        // between the stable floor and the newest turn uncovered.
        #expect(
            history.count > 1,
            "only one history boundary published; a mid-transcript divergence still falls back to the stable prefix")

        let rungs = history.filter { $0 != historyTop }.sorted()
        #expect(rungs.count <= 3, "ladder must stay bounded: extra rungs cost a disk store each")
        for (index, rung) in rungs.enumerated() {
            #expect(rung - stableFloor >= 512)
            let nextHigher = index + 1 < rungs.count ? rungs[index + 1] : historyTop
            #expect(nextHigher - rung >= 512, "rungs must be spaced to be worth their store")
        }

        assertEveryBoundaryIsAnExactMessagePrefix(
            boundaries, tokenizer: tokenizer, messages: messages,
            promptTokens: promptTokens)
    }

    @Test("a short transcript publishes no extra rungs")
    func historyLadderStaysOffWhenTheGapIsSmall() throws {
        // The rungs are only worth their disk store when they skip real
        // prefill; a two-turn chat must keep the old single-boundary shape.
        let tokenizer = ContentBoundaryTokenizer()
        let messages: [[String: any Sendable]] = [
            ["role": "system", "content": "instructions"],
            ["role": "user", "content": "hello"],
        ]
        let promptTokens = try tokenizer.applyChatTemplate(
            messages: messages, tools: nil, additionalContext: nil,
            addGenerationPrompt: true)

        let boundaries = canonicalChatCacheBoundaries(
            tokenizer: tokenizer,
            messages: messages,
            tools: nil,
            additionalContext: nil,
            promptTokens: promptTokens)

        let history = boundaries.all.filter { !boundaries.stable.contains($0) }
        #expect(history.count <= 1)
        assertEveryBoundaryIsAnExactMessagePrefix(
            boundaries, tokenizer: tokenizer, messages: messages,
            promptTokens: promptTokens)
    }
}
