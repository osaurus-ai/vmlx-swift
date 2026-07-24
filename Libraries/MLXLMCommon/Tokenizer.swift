// Copyright © 2024 Apple Inc.

import Foundation

/// A protocol for tokenizing text into token IDs and decoding token IDs into text.
public protocol Tokenizer: Sendable {
    func encode(text: String, addSpecialTokens: Bool) -> [Int]
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String
    func convertTokenToId(_ token: String) -> Int?
    func convertIdToToken(_ id: Int) -> String?

    var bosToken: String? { get }
    var eosToken: String? { get }
    var unknownToken: String? { get }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int]
}

/// Optional tokenizer capability for rendering the same chat template with
/// `add_generation_prompt` disabled.
///
/// Cache stores use this to capture canonical history boundaries before the
/// assistant generation rail. That lets a later full-history request reuse a
/// safe prefix instead of falsely keying KV state that includes model-specific
/// generation-control tokens not present in rendered history.
public protocol GenerationPromptControllableTokenizer: Tokenizer {
    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?,
        addGenerationPrompt: Bool
    ) throws -> [Int]
}

/// Exact chat-template prefix boundaries that can be persisted independently.
///
/// `all` includes the canonical no-generation-prompt history boundary used by a
/// growing conversation. `stable` is the subset rendered from only the leading
/// system/developer instructions plus the request's tool schemas. Because the
/// stable boundary excludes the current user turn, a different new chat can
/// reuse the shared system/tool prefill from L2 instead of warming from token
/// zero again.
///
/// Every returned boundary is proven by token equality to be an actual prefix
/// of the active prompt. Templates that conditionally reorder or rewrite that
/// material therefore fail closed and return no such boundary.
public struct CanonicalChatCacheBoundaries: Sendable, Equatable {
    public let all: [Int]
    public let stable: [Int]

    public init(all: [Int], stable: [Int]) {
        self.all = all
        self.stable = stable
    }
}

/// Derive safe cache boundaries from the exact active chat template.
public func canonicalChatCacheBoundaries(
    tokenizer: any Tokenizer,
    messages: [[String: any Sendable]],
    tools: [[String: any Sendable]]?,
    additionalContext: [String: any Sendable]?,
    promptTokens: [Int],
    staticSystemPrefix: String? = nil
) -> CanonicalChatCacheBoundaries {
    guard let controllable = tokenizer as? any GenerationPromptControllableTokenizer else {
        return CanonicalChatCacheBoundaries(all: [], stable: [])
    }

    func exactPrefixBoundary(
        messages boundaryMessages: [[String: any Sendable]]
    ) -> Int? {
        guard let tokens = try? controllable.applyChatTemplate(
            messages: boundaryMessages,
            tools: tools,
            additionalContext: additionalContext,
            addGenerationPrompt: false),
            !tokens.isEmpty,
            tokens.count < promptTokens.count,
            promptTokens.prefix(tokens.count).elementsEqual(tokens)
        else {
            return nil
        }
        return tokens.count
    }

    /// Some otherwise valid chat templates refuse to render instructions and
    /// tools without a user query (Qwen 3.5 / Ornith / Bonsai raise
    /// `No user query found in messages.`). Derive the stable boundary without
    /// assuming a template shape: append two user probes whose first content
    /// tokens differ, then keep only the token prefix shared by both probes and
    /// the real prompt. The first probe divergence proves that no user content
    /// is included; the real-prompt comparison proves the result is reusable by
    /// this request.
    func probeDerivedStableBoundary(
        messages stableMessages: [[String: any Sendable]]
    ) -> Int? {
        func renderProbe(_ content: String) -> [Int]? {
            var probeMessages = stableMessages
            probeMessages.append(["role": "user", "content": content])
            return try? controllable.applyChatTemplate(
                messages: probeMessages,
                tools: tools,
                additionalContext: additionalContext,
                addGenerationPrompt: false)
        }

        guard let probeA = renderProbe("0"),
              let probeB = renderProbe("z"),
              !probeA.isEmpty,
              !probeB.isEmpty
        else {
            return nil
        }

        let limit = min(probeA.count, probeB.count, promptTokens.count)
        var boundary = 0
        while boundary < limit,
              probeA[boundary] == probeB[boundary],
              probeA[boundary] == promptTokens[boundary]
        {
            boundary += 1
        }

        guard boundary > 0,
              boundary < probeA.count,
              boundary < probeB.count,
              boundary < promptTokens.count
        else {
            return nil
        }
        return boundary
    }

    /// Osaurus composes the reusable static prompt prefix and the mutable
    /// database/sandbox state as separate manifest sections, then renders them
    /// into one system message for model compatibility. A changed dynamic
    /// section therefore invalidates the tokenizer-derived "whole system"
    /// boundary even though the leading static bytes are still reusable.
    ///
    /// Derive that earlier boundary with the same LCP proof used for required
    /// user templates: replace the current system message with the static
    /// prefix plus two divergent tails, then keep only token positions shared
    /// by both probes and the real prompt. This makes the boundary independent
    /// of the mutable suffix and of BPE merges across the split point.
    func hintedStaticSystemBoundary(_ staticPrefix: String?) -> Int? {
        guard let staticPrefix,
              !staticPrefix.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
              let systemIndex = messages.firstIndex(where: { message in
                  guard let role = message["role"] as? String else { return false }
                  return role == "system" || role == "developer"
              }),
              let originalContent = messages[systemIndex]["content"] as? String,
              originalContent.hasPrefix(staticPrefix)
        else {
            return nil
        }

        func renderProbe(_ tail: String) -> [Int]? {
            var probeMessages = Array(messages.prefix(systemIndex + 1))
            var systemMessage = probeMessages[systemIndex]
            systemMessage["content"] = staticPrefix + tail
            probeMessages[systemIndex] = systemMessage
            return try? controllable.applyChatTemplate(
                messages: probeMessages,
                tools: tools,
                additionalContext: additionalContext,
                addGenerationPrompt: false)
        }

        guard let probeA = renderProbe("\n0"),
              let probeB = renderProbe("\nz"),
              !probeA.isEmpty,
              !probeB.isEmpty
        else {
            return nil
        }

        let limit = min(probeA.count, probeB.count, promptTokens.count)
        var boundary = 0
        while boundary < limit,
              probeA[boundary] == probeB[boundary],
              probeA[boundary] == promptTokens[boundary]
        {
            boundary += 1
        }

        guard boundary > 0,
              boundary < probeA.count,
              boundary < probeB.count,
              boundary < promptTokens.count
        else {
            return nil
        }
        return boundary
    }

    // Only the leading instruction rail is stable across unrelated chats.
    // Tool schemas remain present because they are a separate template input;
    // this also permits a tool-only stable prefix when a template emits one for
    // an empty message list. Exact-prefix validation below is the safety gate.
    let stableMessages = Array(messages.prefix { message in
        guard let role = message["role"] as? String else { return false }
        return role == "system" || role == "developer"
    })
    let hasStableMaterial = !stableMessages.isEmpty || tools?.isEmpty == false
    let stableBoundary = hasStableMaterial
        ? exactPrefixBoundary(messages: stableMessages)
            ?? probeDerivedStableBoundary(messages: stableMessages)
        : nil
    let stable = Array(
        Set(
            [stableBoundary, hintedStaticSystemBoundary(staticSystemPrefix)]
                .compactMap { $0 }
        )
    ).sorted()
    let history = exactPrefixBoundary(messages: messages).map { [$0] } ?? []
    let all = Array(Set(stable + history)).sorted()
    if ProcessInfo.processInfo.environment["VMLX_CACHE_FETCH_TRACE"] == "1" {
        FileHandle.standardError.write(Data(
            "[vmlx][cache/boundaries] prompt=\(promptTokens.count) stable=\(stable) all=\(all)\n".utf8
        ))
    }
    return CanonicalChatCacheBoundaries(all: all, stable: stable)
}

extension Tokenizer {
    public func encode(text: String) -> [Int] {
        encode(text: text, addSpecialTokens: true)
    }

    public func decode(tokenIds: [Int]) -> String {
        decode(tokenIds: tokenIds, skipSpecialTokens: false)
    }

    public var eosTokenId: Int? {
        guard let eosToken else { return nil }
        return convertTokenToId(eosToken)
    }

    public var unknownTokenId: Int? {
        guard let unknownToken else { return nil }
        return convertTokenToId(unknownToken)
    }

    public func applyChatTemplate(
        messages: [[String: any Sendable]]
    ) throws -> [Int] {
        try applyChatTemplate(messages: messages, tools: nil, additionalContext: nil)
    }

    public func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?
    ) throws -> [Int] {
        try applyChatTemplate(messages: messages, tools: tools, additionalContext: nil)
    }
}

public enum TokenizerError: LocalizedError {
    case missingChatTemplate

    public var errorDescription: String? {
        switch self {
        case .missingChatTemplate:
            "This tokenizer does not have a chat template."
        }
    }
}

public protocol StreamingDetokenizer: IteratorProtocol<String> {
    mutating func append(token: Int)
}

public struct NaiveStreamingDetokenizer: StreamingDetokenizer {
    let tokenizer: any Tokenizer
    static let trailingHoldbackCharacters = 24

    var segmentTokens = [Int]()
    var segment = ""

    public init(tokenizer: any Tokenizer) {
        self.tokenizer = tokenizer
    }

    public mutating func append(token: Int) {
        segmentTokens.append(token)
    }

    mutating func startNewSegment() {
        let lastToken = segmentTokens.last
        segmentTokens.removeAll()
        if let lastToken {
            segmentTokens.append(lastToken)
            segment = tokenizer.decode(tokenIds: segmentTokens)
        } else {
            segment = ""
        }
    }

    public mutating func next() -> String? {
        emitDecodedSegment(segmentTokens, holdBackTail: true)
    }

    public mutating func flush() -> String? {
        emitDecodedSegment(segmentTokens, holdBackTail: false)
    }

    private mutating func emitDecodedSegment(_ tokens: [Int], holdBackTail: Bool) -> String? {
        var newSegment = tokenizer.decode(tokenIds: tokens)
        if holdBackTail {
            guard newSegment.count > Self.trailingHoldbackCharacters else {
                return nil
            }
            let stableEnd = newSegment.index(
                newSegment.endIndex,
                offsetBy: -Self.trailingHoldbackCharacters)
            newSegment = String(newSegment[..<stableEnd])
        }

        // Decode can produce a SHORTER string than the previous segment
        // when the tokenizer's stateful reassembly reinterprets earlier
        // tokens — e.g. `cleanUpTokenizationSpaces` substitutions
        // (" 's" → "'s", " ." → "."), byte-level BPE completing a
        // multi-byte UTF-8 grapheme that previously rendered as one or
        // more `\u{fffd}` replacements, or two adjacent specials
        // collapsing to a shorter rendered marker. Passing a negative
        // length to `String.suffix(_:)` traps with
        //   "Can't take a suffix of negative length from a collection"
        // which surfaces as a Swift `_assertionFailure` on the
        // generate()-pipeline Task (reproduced via
        // `NaiveStreamingDetokenizerShrinkTests`). Reconcile our
        // baseline and yield nothing for this step — the detokenizer
        // remains usable for future `append(token:)` calls.
        guard newSegment.count >= segment.count else {
            self.segment = newSegment
            return nil
        }

        let new = newSegment.suffix(newSegment.count - segment.count)

        // if the new segment ends with REPLACEMENT CHARACTER this means
        // that the token didn't produce a complete unicode character
        if new.last == "\u{fffd}" {
            return nil
        }

        // Defer mid-grapheme-cluster emits so streaming output never
        // splits a multi-codepoint emoji (regional-indicator pairs for
        // flags, ZWJ sequences for compound emoji, base+variation-
        // selector pairs). Without this guard, e.g. `🇺🇸` (US flag =
        // U+1F1FA + U+1F1F8) streams as two separate broken-box
        // glyphs — confirmed user-visible 2026-04-24 with
        // MiniMax-M2.7-Small JANGTQ rendering an emitted flag as
        // `❓国旗` in osaurus.
        //
        // Inspect the LAST grapheme cluster of `new` rather than its
        // last scalar — Swift treats `🇺🇸` as one grapheme even when
        // the character has two regional-indicator scalars, so a raw
        // scalar check would defer the completed flag forever.
        // Triggers:
        //   • Last grapheme is a single unpaired regional indicator
        //     (count == 1 within range 0x1F1E6 - 0x1F1FF) → wait for
        //     the sibling that completes the flag.
        //   • Last scalar of last grapheme is ZWJ (U+200D) → the
        //     ZWJ-emoji chain is mid-build; wait for the next codepoint.
        //   • Trailing high surrogate (rare in Swift String, but
        //     harmless to defer if it ever appears).
        if let lastChar = new.last {
            let scalars = Array(lastChar.unicodeScalars)
            if let lastScalarValue = scalars.last?.value {
                let isUnpairedRegionalIndicator =
                    scalars.count == 1
                    && (0x1F1E6...0x1F1FF).contains(lastScalarValue)
                let endsWithZWJ = lastScalarValue == 0x200D
                let endsWithHighSurrogate =
                    (0xD800...0xDBFF).contains(lastScalarValue)
                if isUnpairedRegionalIndicator || endsWithZWJ
                    || endsWithHighSurrogate
                {
                    return nil
                }
            }
        }

        if !holdBackTail && new.hasSuffix("\n") {
            startNewSegment()
        } else {
            self.segment = newSegment
        }

        return String(new)
    }
}
