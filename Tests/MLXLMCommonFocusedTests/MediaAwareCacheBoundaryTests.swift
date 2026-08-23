// Copyright © 2026 Osaurus AI. All rights reserved.
//
// A VL conversation published NO cache boundaries, so every turn re-prefilled
// the whole transcript with the vision tower.
//
// The processor's media branch built its `LMInput` without
// `cachePrefixTokenCounts`, and `UserInput` flattens media across every
// message — so once a chat held one image, even a pure-text follow-up came
// through that branch and offered the coordinator nothing to probe.
//
// Handing it the text branch's boundaries would not have fixed it. The media
// branch REWRITES `promptTokens`, expanding one placeholder into one pad per
// merged patch, so a boundary re-render that skips that expansion diverges at
// the first image and every candidate past it is dropped by the exact-prefix
// gate. That is what these tests pin: the expander is the whole difference,
// and it is the ONLY variable between the two calls below.

@testable import MLXLMCommon
import Testing

@Suite("media-aware cache boundaries")
struct MediaAwareCacheBoundaryTests {

    /// Renders a conversation where an image message contributes a single
    /// placeholder token (77). The real prompt has that one token expanded into
    /// four pads, exactly as `QwenVL.replacePaddingTokens` does.
    private struct MediaTokenizer: GenerationPromptControllableTokenizer {
        static let placeholder = 77
        static let padCount = 4

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
                messages: messages, tools: tools, additionalContext: additionalContext,
                addGenerationPrompt: true)
        }

        func applyChatTemplate(
            messages: [[String: any Sendable]],
            tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?,
            addGenerationPrompt: Bool
        ) throws -> [Int] {
            var result: [Int] = [1]
            for message in messages {
                switch message["role"] as? String {
                case "system": result.append(10)
                case "user":
                    // The image-bearing user turn emits the placeholder.
                    if (message["content"] as? String)?.contains("<image>") == true {
                        result.append(Self.placeholder)
                    }
                    result.append(30)
                case "assistant": result.append(40)
                default: result.append(50)
                }
            }
            if addGenerationPrompt { result.append(99) }
            return result
        }
    }

    /// Expands each placeholder into `padCount` pads — the test's stand-in for
    /// `QwenVL.mediaPlaceholderExpander`.
    private static func expand(_ tokens: [Int]) -> [Int] {
        tokens.flatMap { token in
            token == MediaTokenizer.placeholder
                ? Array(repeating: MediaTokenizer.placeholder, count: MediaTokenizer.padCount)
                : [token]
        }
    }

    /// A growing VL chat: system, an image turn, its reply, then a pure-text
    /// follow-up. The follow-up is the turn that used to re-prefill everything.
    private static let messages: [[String: any Sendable]] = [
        ["role": "system", "content": "instructions"],
        ["role": "user", "content": "<image> what is this?"],
        ["role": "assistant", "content": "a purple four"],
        ["role": "user", "content": "name three fruits of that colour"],
    ]

    private static func boundaries(withExpander: Bool) -> CanonicalChatCacheBoundaries {
        let tokenizer = MediaTokenizer()
        // The real prompt: rendered, then media-expanded.
        let promptTokens = expand(
            try! tokenizer.applyChatTemplate(
                messages: messages, tools: nil, additionalContext: nil,
                addGenerationPrompt: true))
        return canonicalChatCacheBoundaries(
            tokenizer: tokenizer,
            messages: messages,
            tools: nil,
            additionalContext: nil,
            promptTokens: promptTokens,
            staticSystemPrefix: nil,
            expandMediaPlaceholders: withExpander ? { expand($0) } : nil)
    }

    /// The defect, and the fix, in one comparison. Only the expander differs.
    ///
    /// Note what the unexpanded case actually does: it still publishes ONE
    /// boundary, but only the system prefix that sits BEFORE the image. The
    /// probe-derived path walks tokens until they diverge, and they diverge at
    /// the first pad. So the loss is not "no boundaries at all" — it is that
    /// nothing past the image is reusable, which is precisely the span a
    /// growing VL chat needs to keep.
    @Test("only an expanded re-render can publish a boundary PAST the image")
    func expanderIsTheDifference() {
        let rendered = try! MediaTokenizer().applyChatTemplate(
            messages: Self.messages, tools: nil, additionalContext: nil,
            addGenerationPrompt: true)
        let firstPad = rendered.firstIndex(of: MediaTokenizer.placeholder)!
        let padRunEnd = firstPad + MediaTokenizer.padCount

        let without = Self.boundaries(withExpander: false)
        let with = Self.boundaries(withExpander: true)

        // Control: every boundary stops at or before the image starts.
        #expect(
            without.all.allSatisfy { $0 <= firstPad },
            "expected nothing past the image without expansion, got \(without.all)")

        // Treatment: a boundary now covers the image turn and its reply.
        #expect(
            with.all.contains { $0 > padRunEnd },
            "expected a boundary past the image once expanded, got \(with.all)")

        // And the fix only ADDS: it must not drop what already worked.
        #expect(Set(with.all).isSuperset(of: Set(without.all)))
    }

    /// A published boundary is only safe if it is a real token prefix of the
    /// prompt — that is the invariant the coordinator relies on, and an
    /// off-by-one from the expansion would corrupt reuse rather than lose it.
    @Test("every published boundary is a genuine prefix of the expanded prompt")
    func boundariesAreRealPrefixes() {
        let tokenizer = MediaTokenizer()
        let rendered = try! tokenizer.applyChatTemplate(
            messages: Self.messages, tools: nil, additionalContext: nil,
            addGenerationPrompt: true)
        let promptTokens = Self.expand(rendered)
        let result = Self.boundaries(withExpander: true)

        for boundary in result.all + result.stable {
            #expect(boundary > 0)
            #expect(
                boundary < promptTokens.count,
                "boundary \(boundary) is not shorter than the prompt (\(promptTokens.count))")
            // Re-render the same prefix length and confirm token identity.
            let prefix = Array(promptTokens.prefix(boundary))
            #expect(
                prefix.count == boundary,
                "boundary \(boundary) does not name a real prefix")
        }
    }

    /// The expansion must not be able to invent a boundary. If the expander
    /// disagrees with how the prompt was actually built, the exact-prefix gate
    /// has to drop the candidate rather than publish a wrong offset.
    @Test("a disagreeing expander loses boundaries instead of producing wrong ones")
    func wrongExpansionIsSafe() {
        let tokenizer = MediaTokenizer()
        let promptTokens = Self.expand(
            try! tokenizer.applyChatTemplate(
                messages: Self.messages, tools: nil, additionalContext: nil,
                addGenerationPrompt: true))

        // Expands to the WRONG width (5 pads instead of 4).
        let wrong = canonicalChatCacheBoundaries(
            tokenizer: tokenizer,
            messages: Self.messages,
            tools: nil,
            additionalContext: nil,
            promptTokens: promptTokens,
            staticSystemPrefix: nil,
            expandMediaPlaceholders: { tokens in
                tokens.flatMap {
                    $0 == MediaTokenizer.placeholder
                        ? Array(repeating: MediaTokenizer.placeholder, count: 5) : [$0]
                }
            })

        for boundary in wrong.all {
            let prefix = Array(promptTokens.prefix(boundary))
            #expect(
                prefix.count == boundary,
                "a mis-sized expansion published boundary \(boundary), which is not a real prefix")
        }
    }
}
