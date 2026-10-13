// Copyright © 2024 Apple Inc.

import Foundation

/// A lossless piece from a compatible plain ByteLevel tokenizer.
public enum ByteLevelDecodingPiece: Sendable {
    case bytes([UInt8])
    case literal(String)
    case ignored
}

/// Exact, ID-indexed vocabulary metadata for grammar compilation. These are
/// tokenizer model pieces, not strings produced by decoding individual tokens.
/// Providers must identify the decoder encoding from tokenizer configuration and
/// include added tokens at their real IDs. Unsupported decoders return nil.
public struct GrammarTokenVocabulary: Sendable {
    public let vocabulary: [String]
    public let vocabularyType: JSONSchemaVocabularyType
    public let specialTokenIDs: Set<Int>

    public init(
        vocabulary: [String], vocabularyType: JSONSchemaVocabularyType,
        specialTokenIDs: Set<Int>
    ) {
        self.vocabulary = vocabulary
        self.vocabularyType = vocabularyType
        self.specialTokenIDs = specialTokenIDs
    }
}

/// A protocol for tokenizing text into token IDs and decoding token IDs into text.
public protocol Tokenizer: Sendable {
    /// Optional exact grammar vocabulary. Nil means constrained generation is
    /// unsupported; callers must not infer bytes using decode([tokenID]).
    var grammarTokenVocabulary: GrammarTokenVocabulary? { get }

    /// Optional lossless ByteLevel decoding with cleanup disabled. Other
    /// tokenizers retain the generic streaming decoder.
    var incrementalByteLevelDecoder: (@Sendable (Int) -> ByteLevelDecodingPiece)? { get }

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
    /// A transcript ending in an assistant turn is a CONTINUATION, and formats
    /// that model that faithfully append no generation rail for it — DSV4's
    /// official encoder only emits `<｜Assistant｜>` after a user/developer/
    /// latest_reminder message. Rendering with and without the generation
    /// prompt then produces identical tokens, so `exactPrefixBoundary`'s strict
    /// `tokens.count < promptTokens.count` rejects it and the history boundary
    /// disappears entirely.
    ///
    /// That silently killed prefix reuse for exactly the agent-loop turns that
    /// need it most. Live DSV4 row: a plain turn published
    /// `all=[72, 1128, 1456]`, but the first turn carrying a completed tool call
    /// and its artifact published only `all=[72, 1128]` and cold-prefilled 5560
    /// tokens — and because entries are stored AT published boundaries, the next
    /// iteration had nothing to reuse either, so every round re-prefilled the
    /// whole growing transcript.
    ///
    /// Fall back to the transcript without that trailing assistant turn. It is
    /// strictly shorter, and it is still proven by the same exact-token-prefix
    /// check — this does not relax token identity or admit suffix matching.
    /// A trailing `assistant` turn is a continuation; a trailing `tool` turn is
    /// the tool result the model is about to continue from. Both are shapes a
    /// format may render without a generation rail, and both appear in a normal
    /// agent loop — the post-turn re-warm after a TOOL call ends with the tool
    /// result, which showed up live as a fixed miss pattern (stored N, the
    /// re-warm asks N+2, MISS) on 2084→2086, 2286→2288 and 2756→2758.
    func trailingContinuationBoundary() -> Int? {
        guard messages.count > 1, let last = messages.last else { return nil }
        let role = last["role"] as? String
        guard role == "assistant" || role == "tool" else { return nil }
        return exactPrefixBoundary(messages: Array(messages.dropLast()))
    }

    /// Prove that the newest history boundary survives the next assistant
    /// continuation, rather than only proving that a no-generation render is
    /// a prefix of the CURRENT prompt.
    ///
    /// Some templates rewrite the separator immediately before the assistant
    /// rail when an assistant/tool message is materialised. Raptor/Qwen-shaped
    /// tool history exposed the concrete failure: the current user prompt ended
    /// in `\n<role>assistant`, while the next structured tool-call render used
    /// `<role>assistant` at that same position. The old boundary included the
    /// newline, so an otherwise identical growing prompt missed its just-written
    /// SSD entry and fell back to the static system prefix.
    ///
    /// Render two divergent future assistant messages and retain only the LCP
    /// shared by both renders AND the active prompt. Divergent contents prove
    /// the boundary contains no assistant-controlled payload; comparison with
    /// the active prompt preserves the exact-token KV invariant. This is
    /// template-derived and model-family agnostic.
    func assistantContinuationStableBoundary() -> Int? {
        guard !messages.isEmpty else { return nil }

        func renderProbe(_ content: String) -> [Int]? {
            var probeMessages = messages
            probeMessages.append(["role": "assistant", "content": content])
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

    /// Intermediate turn-aligned rungs between the stable prefix and the
    /// newest history boundary.
    ///
    /// Only ONE history boundary used to be published, so the entire span
    /// between the frozen system/tool prefix and the newest turn held no
    /// stored entry. Because entries are stored AT published boundaries, any
    /// divergence in that span dropped reuse all the way back to the system
    /// prefix. Live DSV4 row: `all=[1114, 2643, 5434]`, and two sends in that
    /// same session fell back to 2643 and re-prefilled 1948 and 2044 tokens
    /// that were still byte-identical prefixes.
    ///
    /// Exactness is unchanged: a rung is admitted only by `exactPrefixBoundary`,
    /// which requires the rendered message prefix to equal the real prompt
    /// token-for-token. Rungs are cut at message boundaries so they re-render
    /// identically on later turns, and spaced by `minimumGap` so the extra
    /// disk stores each buy a worthwhile amount of skipped prefill.
    func historyLadder(below top: Int) -> [Int] {
        guard ProcessInfo.processInfo.environment["VMLX_CACHE_BOUNDARY_LADDER"] != "0"
        else { return [] }
        let maximumRungs = 3
        let minimumGap = 512
        let floor = stable.last ?? 0
        guard top - floor >= minimumGap * 2 else { return [] }

        var rungs: [Int] = []
        var nextHigher = top
        // Exponential backoff from the tail: recent turns are re-rendered most
        // often, so granularity is worth more there than deep in the history.
        for drop in [1, 2, 4, 8, 16] {
            if rungs.count == maximumRungs { break }
            guard messages.count - drop > stableMessages.count else { break }
            guard let rung = exactPrefixBoundary(
                messages: Array(messages.dropLast(drop))),
                rung - floor >= minimumGap,
                nextHigher - rung >= minimumGap
            else { continue }
            rungs.append(rung)
            nextHigher = rung
        }
        return rungs
    }

    /// The continuation probe proves a boundary only when it reaches past the previous message
    /// boundary, i.e. it contains the newest message. A probe the template rejected or rewrote
    /// diverges earlier and proves nothing: K2 Horizon's template requires a thinking field on every
    /// assistant message, swift-jinja renders its `raise_exception` text instead of throwing, and the
    /// probe shared exactly one token (BOS) with the prompt. That `1` outranked the exact
    /// no-generation boundary, so every K2 follow-up at effort low/medium (generation rail
    /// `<ifm|think_faster>…`, which history renders differently) missed its stored prompt and
    /// re-prefilled the whole conversation: TTFT 1.2 → 5.6 s over five 2.5k-token turns.
    func provenContinuationBoundary() -> Int? {
        guard let boundary = assistantContinuationStableBoundary() else { return nil }
        let previous = messages.count > 1
            ? exactPrefixBoundary(messages: Array(messages.dropLast())) : nil
        let floor = Swift.max(previous ?? 0, stable.last ?? 0, 1)
        return boundary > floor ? boundary : nil
    }

    let historyTop = provenContinuationBoundary()
        ?? exactPrefixBoundary(messages: messages)
        ?? trailingContinuationBoundary()
    let history = historyTop.map { [$0] + historyLadder(below: $0) } ?? []
    let all = Array(Set(stable + history)).sorted()
    if ProcessInfo.processInfo.environment["VMLX_CACHE_FETCH_TRACE"] == "1" {
        // Token counts alone cannot distinguish two prompts of equal length, so
        // a boundary that is stored and then immediately missed looks identical
        // in the trace to one that genuinely diverged. Emit a digest of the
        // leading tokens as well: comparing `head=` across a store and the
        // following re-warm shows directly whether the prompt changed in the
        // system region rather than leaving it to inference.
        //
        // Diagnostic only — gated behind the same trace flag, never consulted
        // by cache logic, and never used as a cache key.
        func digest(_ ids: ArraySlice<Int>) -> String {
            var hash: UInt64 = 0xcbf2_9ce4_8422_2325
            for id in ids {
                hash = (hash ^ UInt64(bitPattern: Int64(id))) &* 0x1000_0000_01b3
            }
            return String(hash & 0xffff_ffff, radix: 16)
        }
        let headDigest = digest(promptTokens.prefix(256))
        let stableDigest = stable.first.map { digest(promptTokens.prefix($0)) } ?? "-"
        // Only the *first* stable boundary was digested, but the boundary that
        // actually gets stored is the last one. A divergence between the two —
        // an injected reasoning preface, a reordered tool block, a date — moves
        // the stored key while leaving `head`/`stable0` identical, which reads
        // in the trace as a cache bug rather than a changed prompt.
        let stableDigests = stable
            .map { "\($0):\(digest(promptTokens.prefix($0)))" }
            .joined(separator: ",")
        // `head`/`stable0` both sit inside the system prefix, so when they match
        // across a store and a failing re-warm they only prove the system region
        // is stable — the first live capture showed exactly that, and the
        // divergence has to be further in. `hist` covers the conversation span
        // up to the history boundary, which is where the remaining `stored+2`
        // miss must originate.
        let historyDigest = all.last.map { digest(promptTokens.prefix($0)) } ?? "-"
        FileHandle.standardError.write(Data(
            ("[vmlx][cache/boundaries] prompt=\(promptTokens.count) stable=\(stable) all=\(all)"
                + " head=\(headDigest) stable0=\(stableDigest) hist=\(historyDigest)"
                + " stableDigests=[\(stableDigests)]\n").utf8
        ))
    }
    return CanonicalChatCacheBoundaries(all: all, stable: stable)
}

extension Tokenizer {
    public var grammarTokenVocabulary: GrammarTokenVocabulary? { nil }
    public var incrementalByteLevelDecoder: (@Sendable (Int) -> ByteLevelDecodingPiece)? { nil }

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
    private let byteDecoder: (@Sendable (Int) -> ByteLevelDecodingPiece)?
    private var byteState = IncrementalByteLevelState()

    public init(tokenizer: any Tokenizer) {
        self.tokenizer = tokenizer
        self.byteDecoder = tokenizer.incrementalByteLevelDecoder
    }

    public mutating func append(token: Int) {
        if let byteDecoder {
            byteState.append(byteDecoder(token))
            return
        }
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
        if byteDecoder != nil { return byteState.emit(holdBackTail: true) }
        return emitDecodedSegment(segmentTokens, holdBackTail: true)
    }

    public mutating func flush() -> String? {
        if byteDecoder != nil { return byteState.emit(holdBackTail: false) }
        return emitDecodedSegment(segmentTokens, holdBackTail: false)
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

/// Only the un-emitted grapheme tail and an incomplete UTF8 scalar remain live.
/// No newline compaction: it would discard the held tail. A grapheme can contain
/// arbitrarily many combining scalars, so the tail is not capped in bytes.
private struct IncrementalByteLevelState {
    private var pendingBytes: [UInt8] = []
    private var tail = ""

    mutating func append(_ piece: ByteLevelDecodingPiece) {
        switch piece {
        case .ignored: break
        case .literal(let text):
            tail += String(decoding: pendingBytes, as: UTF8.self)
            pendingBytes.removeAll(keepingCapacity: true)
            tail += text
        case .bytes(let bytes):
            pendingBytes += bytes
            let held = Self.incompleteSuffixLength(pendingBytes)
            let complete = pendingBytes.count - held
            if complete > 0 {
                tail += String(decoding: pendingBytes.prefix(complete), as: UTF8.self)
                pendingBytes = Array(pendingBytes.suffix(held))
            }
        }
    }

    mutating func emit(holdBackTail: Bool) -> String? {
        let visible = holdBackTail ? tail : tail + String(decoding: pendingBytes, as: UTF8.self)
        let boundary: String.Index
        if holdBackTail {
            guard visible.count > NaiveStreamingDetokenizer.trailingHoldbackCharacters else { return nil }
            boundary = visible.index(visible.endIndex, offsetBy: -NaiveStreamingDetokenizer.trailingHoldbackCharacters)
        } else {
            boundary = visible.endIndex
        }
        let output = String(visible[..<boundary])
        guard !output.isEmpty else { return nil }
        // Preserve the generic decoder's terminal deferral contract. In
        // particular, flush of an incomplete byte run must not seal a U+FFFD
        // that a subsequent append can still complete.
        if let last = output.last {
            if last == "\u{fffd}" { return nil }
            let scalars = Array(last.unicodeScalars)
            if let value = scalars.last?.value,
               (value == 0x200D || (scalars.count == 1 && (0x1F1E6...0x1F1FF).contains(value))
                || (0xD800...0xDBFF).contains(value)) { return nil }
        }
        tail = String(visible[boundary...])
        if !holdBackTail { pendingBytes.removeAll(keepingCapacity: true) }
        return output
    }

    private static func incompleteSuffixLength(_ bytes: [UInt8]) -> Int {
        guard !bytes.isEmpty else { return 0 }
        for start in max(0, bytes.count - 3)..<bytes.count {
            let lead = bytes[start]
            let required: Int
            switch lead {
            case 0xC2...0xDF: required = 2
            case 0xE0...0xEF: required = 3
            case 0xF0...0xF4: required = 4
            default: continue
            }
            let present = bytes.count - start
            guard present < required,
                  bytes[(start + 1)...].allSatisfy({ (0x80...0xBF).contains($0) }) else { continue }
            if present >= 2 {
                let second = bytes[start + 1]
                if lead == 0xE0 && second < 0xA0 { continue }
                if lead == 0xED && second > 0x9F { continue }
                if lead == 0xF0 && second < 0x90 { continue }
                if lead == 0xF4 && second > 0x8F { continue }
            }
            return present
        }
        return 0
    }
}
