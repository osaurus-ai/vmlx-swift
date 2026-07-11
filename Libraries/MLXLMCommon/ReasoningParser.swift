// Copyright © 2026 Osaurus AI. All rights reserved.

import Foundation

// MARK: - ReasoningSegment

/// A segment of model output classified as either visible content or hidden
/// chain-of-thought reasoning.
public enum ReasoningSegment: Sendable, Equatable {
    /// Visible content the user should see.
    case content(String)
    /// Reasoning the application may want to display in a separate UI affordance
    /// (think pane, foldable section, etc.) — *not* the visible answer.
    case reasoning(String)
}

// MARK: - ReasoningParser

/// Streaming-safe parser that splits a token-by-token model stream into
/// `.content(...)` and `.reasoning(...)` segments based on tag delimiters.
///
/// **Why this lives in vmlx-swift-lm rather than osaurus:**
/// Models like Qwen 3.5/3.6, DeepSeek-R1, and others mark reasoning blocks
/// with literal vocabulary tokens (e.g. `<think>` / `</think>`) that they
/// have **deliberately marked as `special: false`** in their tokenizer config,
/// so every consumer (osaurus, llm-tool, anything else built on
/// vmlx-swift-lm) sees them as plain text. Each consumer would otherwise
/// re-implement the same boundary tracking — and get edge cases wrong.
/// Centralising it here keeps streaming behaviour consistent and lets
/// consumers choose to either show, hide, or relabel reasoning.
///
/// Default tags match Qwen 3.5 / Qwen 3.6 / DeepSeek-R1 (`<think>...</think>`).
/// Override `startTag`/`endTag` for models that use different markers.
///
/// ## Streaming contract
///
/// Token streams arrive in fragments — a single tag may be split across
/// several `feed(...)` calls (e.g. `<thi`, `nk>`). The parser buffers
/// **only** the portion that could be a partial tag prefix; everything
/// else is emitted immediately as `.content` or `.reasoning`.
///
/// On end-of-sequence, call `flush()` once to drain any remaining buffered
/// text. Anything still buffered after a final `flush()` is emitted as
/// `.content` (we never lose tokens to the parser).
///
/// ## Example
///
/// ```swift
/// var parser = ReasoningParser()  // defaults to <think>/</think>
/// for chunk in stream {
///     for segment in parser.feed(chunk) {
///         switch segment {
///         case .content(let text):   appendToVisibleAnswer(text)
///         case .reasoning(let text): appendToThinkPane(text)
///         }
///     }
/// }
/// for segment in parser.flush() { ... }
/// ```
public struct ReasoningParser: Sendable {

    // MARK: Configuration

    /// The tag that starts a reasoning block. Default `<think>`.
    public let startTag: String

    /// The tag that ends a reasoning block. Default `</think>`.
    public let endTag: String

    /// When true, the drain loop strips stray markers regardless of
    /// state — a `</think>` while in content mode is consumed as a
    /// model artifact (state stays in content), and a `<think>` while
    /// in reasoning mode is consumed similarly. Required for the
    /// `<think>`/`</think>` family because models occasionally emit
    /// duplicate or unmatched markers in interleaved-thinking decode
    /// (verified 2026-04-25 on MiniMax-Small JANGTQ where `</think>`
    /// leaked into the user-visible chunk stream three times across
    /// one assistant turn).
    ///
    /// When false, the drain loop only looks for whichever tag
    /// matches the current mode; the other tag passes through as
    /// literal content. Required for the harmony channel format
    /// where stray-tag leaks are the documented intent (legacy
    /// Gemma-4 channel parser behaviour — A2/A3 tests).
    public let stripStrayTags: Bool

    // MARK: State

    /// Text not yet emitted because it might be a partial tag prefix.
    private var buffer: String = ""

    /// Whether we're currently inside a reasoning block.
    private var insideReasoning: Bool = false

    /// Harmony-specific channel state. Gemma 4 uses
    /// `<|channel>...<channel|>`, while GPT-OSS uses
    /// `<|channel|>analysis<|message|>...<|end|>` and
    /// `<|channel|>final<|message|>...<|return|>`. A single
    /// `insideReasoning` bit is not enough for GPT-OSS final channels
    /// because final payloads are visible content but still need their
    /// control tags stripped.
    private var insideHarmonyChannel: Bool = false
    private var harmonyChannelIsReasoning: Bool = false
    private var harmonyChannelEndTags: [String] = []
    private var harmonyChannelShouldStripName: Bool = false
    private var harmonyChannelNameBuffer: String = ""

    /// Read-only access to the current parser state — true when the
    /// stream is currently inside a `<think>…</think>` block (or the
    /// family-specific equivalent). Useful at end-of-stream to detect
    /// the "model emitted EOS while still inside reasoning" pathology
    /// (a.k.a. "trapped thinking") that some reasoning-trained models
    /// exhibit on validation-style prompts.
    public var isInsideReasoning: Bool {
        insideReasoning
    }

    // MARK: Init

    /// - Parameters:
    ///   - startTag: The tag that opens a reasoning block.
    ///   - endTag: The tag that closes a reasoning block.
    ///   - startInReasoning: Start the parser already inside a reasoning
    ///     block. Use this when the chat template prefills the opening
    ///     tag at the prompt tail (e.g. Qwen 3.6 emits `<think>\n` at
    ///     the end of the assistant prompt when `enable_thinking=true`,
    ///     which is the template default) — the model's first output
    ///     byte is already reasoning, so starting in `.content` mode
    ///     would leak the entire CoT into the visible answer until the
    ///     first `</think>` flips state.
    ///   - stripStrayTags: see property doc. Defaults to `true` (correct
    ///     for the `<think>`/`</think>` family). The harmony parser
    ///     factory in `fromCapabilityName(_:)` overrides this to `false`.
    public init(
        startTag: String = "<think>",
        endTag: String = "</think>",
        startInReasoning: Bool = false,
        stripStrayTags: Bool = true
    ) {
        self.startTag = startTag
        self.endTag = endTag
        self.insideReasoning = startInReasoning
        self.stripStrayTags = stripStrayTags
    }

    // MARK: Streaming API

    /// Feed an incoming token-stream chunk. Returns zero or more segments.
    public mutating func feed(_ chunk: String) -> [ReasoningSegment] {
        guard !chunk.isEmpty else { return [] }
        buffer.append(chunk)
        if isHarmonyChannelParser {
            return drainHarmonyChannel()
        }
        return drain()
    }

    /// Call once when the stream ends. Flushes any buffered partial text
    /// as `.content` (so we never silently drop tokens).
    public mutating func flush() -> [ReasoningSegment] {
        if isHarmonyChannelParser {
            var out = drainHarmonyChannel(allowPartialTagAtEnd: false)
            if !buffer.isEmpty {
                let text = insideHarmonyChannel ? buffer : stripHarmonyControlText(buffer)
                if insideHarmonyChannel {
                    appendHarmonyChannelPayload(text, into: &out)
                } else if !text.isEmpty {
                    out.append(.content(text))
                }
                buffer.removeAll(keepingCapacity: false)
            }
            insideHarmonyChannel = false
            harmonyChannelIsReasoning = false
            harmonyChannelEndTags = []
            insideReasoning = false
            return out
        }

        var out = drain(allowPartialTagAtEnd: false)
        if !buffer.isEmpty {
            // Anything left over after the final drain is plain text — emit
            // as content (or as reasoning if we never saw a closing tag).
            out.append(insideReasoning ? .reasoning(buffer) : .content(buffer))
            buffer.removeAll(keepingCapacity: false)
        }
        insideReasoning = false
        return out
    }

    // MARK: Internals

    private var isHarmonyChannelParser: Bool {
        startTag == "<|channel>" && endTag == "<channel|>" && !stripStrayTags
    }

    /// Harmony parser that accepts both:
    ///
    /// - Gemma 4: `<|channel>thought\n...<channel|>`.
    /// - GPT-OSS: `<|channel|>analysis<|message|>...<|end|>` and
    ///   `<|channel|>final<|message|>...<|return|>`.
    ///
    /// The Gemma path routes all channel payloads to reasoning and strips
    /// simple identifier channel headers such as `thought\n`.
    /// The GPT-OSS path strips control tokens and routes `final` to visible
    /// content while routing other channels to reasoning.
    private mutating func drainHarmonyChannel(allowPartialTagAtEnd: Bool = true)
        -> [ReasoningSegment]
    {
        let gemmaStart = "<|channel>"
        let gptStart = "<|channel|>"
        let gptMessage = "<|message|>"
        let gptStartControl = "<|start|>"
        let gptEndTags = ["<|end|>", "<|return|>"]
        let openerTags = [gemmaStart, gptStart]
        let holdTags = openerTags + [gptMessage, gptStartControl, endTag] + gptEndTags

        var out: [ReasoningSegment] = []

        while !buffer.isEmpty {
            if insideHarmonyChannel {
                let endTags = harmonyChannelEndTags.isEmpty ? [endTag] : harmonyChannelEndTags
                guard let (range, _) = firstRange(of: endTags, in: buffer) else {
                    emitSafeHarmonyChannelPrefix(
                        into: &out,
                        tags: holdTags,
                        allowPartialTagAtEnd: allowPartialTagAtEnd)
                    break
                }

                let before = String(buffer[..<range.lowerBound])
                appendHarmonyChannelText(
                    before,
                    into: &out,
                    final: true,
                    stripIdentifierOnlyAtEnd: false)
                buffer.removeSubrange(buffer.startIndex..<range.upperBound)
                insideHarmonyChannel = false
                harmonyChannelIsReasoning = false
                harmonyChannelEndTags = []
                harmonyChannelShouldStripName = false
                harmonyChannelNameBuffer.removeAll(keepingCapacity: false)
                insideReasoning = false
                continue
            }

            guard let (range, tag) = firstRange(of: openerTags, in: buffer) else {
                let strayControlTags = [gptMessage] + gptEndTags
                if let (controlRange, _) = firstRange(of: strayControlTags, in: buffer) {
                    let before = stripHarmonyControlText(String(buffer[..<controlRange.lowerBound]))
                    if !before.isEmpty {
                        out.append(.content(before))
                    }
                    buffer.removeSubrange(buffer.startIndex..<controlRange.upperBound)
                    continue
                }

                if allowPartialTagAtEnd,
                    buffer.contains(gptStartControl)
                        || holdTags.contains(where: { hasPartialSuffix(of: $0, in: buffer) })
                {
                    break
                }
                emitSafeHarmonyPrefix(
                    into: &out,
                    tags: holdTags,
                    allowPartialTagAtEnd: allowPartialTagAtEnd)
                break
            }

            if tag == gptStart {
                guard let messageRange = buffer.range(
                    of: gptMessage,
                    range: range.upperBound..<buffer.endIndex)
                else {
                    if let newlineRange = buffer.range(
                        of: "\n",
                        range: range.upperBound..<buffer.endIndex)
                    {
                        let channelName = String(buffer[range.upperBound..<newlineRange.lowerBound])
                            .trimmingCharacters(in: .whitespacesAndNewlines)
                            .lowercased()
                        if channelName == "thought" || channelName == "thinking" {
                            let before = stripHarmonyControlText(String(buffer[..<range.lowerBound]))
                            if !before.isEmpty {
                                out.append(.content(before))
                            }
                            buffer.removeSubrange(buffer.startIndex..<newlineRange.upperBound)
                            insideHarmonyChannel = true
                            harmonyChannelIsReasoning = true
                            harmonyChannelEndTags = [endTag]
                            harmonyChannelShouldStripName = false
                            harmonyChannelNameBuffer.removeAll(keepingCapacity: false)
                            insideReasoning = true
                            continue
                        }
                    }
                    let before = stripHarmonyControlText(String(buffer[..<range.lowerBound]))
                    if !before.isEmpty {
                        out.append(.content(before))
                    }
                    buffer = String(buffer[range.lowerBound...])
                    break
                }

                let before = stripHarmonyControlText(String(buffer[..<range.lowerBound]))
                if !before.isEmpty {
                    out.append(.content(before))
                }
                let channelName = String(buffer[range.upperBound..<messageRange.lowerBound])
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                    .lowercased()
                buffer.removeSubrange(buffer.startIndex..<messageRange.upperBound)
                insideHarmonyChannel = true
                harmonyChannelIsReasoning = channelName != "final"
                harmonyChannelEndTags = gptEndTags
                harmonyChannelShouldStripName = false
                harmonyChannelNameBuffer.removeAll(keepingCapacity: false)
                insideReasoning = harmonyChannelIsReasoning
                continue
            }

            let before = stripHarmonyControlText(String(buffer[..<range.lowerBound]))
            if !before.isEmpty {
                out.append(.content(before))
            }
            buffer.removeSubrange(buffer.startIndex..<range.upperBound)
            insideHarmonyChannel = true
            harmonyChannelIsReasoning = true
            harmonyChannelEndTags = [endTag]
            harmonyChannelShouldStripName = true
            harmonyChannelNameBuffer.removeAll(keepingCapacity: false)
            insideReasoning = true
        }

        return out
    }

    private mutating func appendHarmonyChannelText(
        _ text: String,
        into out: inout [ReasoningSegment],
        final: Bool = false,
        stripIdentifierOnlyAtEnd: Bool = true
    ) {
        guard !text.isEmpty || final else { return }

        if harmonyChannelShouldStripName {
            harmonyChannelNameBuffer += text
            if let newline = harmonyChannelNameBuffer.firstIndex(of: "\n") {
                let firstLine = String(harmonyChannelNameBuffer[..<newline])
                let remainderStart = harmonyChannelNameBuffer.index(after: newline)
                let remainder = String(harmonyChannelNameBuffer[remainderStart...])
                harmonyChannelNameBuffer.removeAll(keepingCapacity: false)
                harmonyChannelShouldStripName = false
                let emitted = isHarmonyIdentifierChannelName(firstLine)
                    ? remainder
                    : firstLine + "\n" + remainder
                appendHarmonyChannelPayload(emitted, into: &out)
                return
            }

            if final {
                let emitted = isHarmonyThoughtChannelName(harmonyChannelNameBuffer)
                    || (stripIdentifierOnlyAtEnd
                        && isHarmonyIdentifierChannelName(harmonyChannelNameBuffer))
                    ? ""
                    : harmonyChannelNameBuffer
                harmonyChannelNameBuffer.removeAll(keepingCapacity: false)
                harmonyChannelShouldStripName = false
                appendHarmonyChannelPayload(emitted, into: &out)
            }
            return
        }

        appendHarmonyChannelPayload(text, into: &out)
    }

    private func appendHarmonyChannelPayload(_ text: String, into out: inout [ReasoningSegment]) {
        guard !text.isEmpty else { return }
        guard !isHarmonyThoughtChannelName(text) else { return }
        out.append(harmonyChannelIsReasoning ? .reasoning(text) : .content(text))
    }

    private func isHarmonyIdentifierChannelName(_ text: String) -> Bool {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let first = trimmed.unicodeScalars.first else { return true }
        guard CharacterSet.letters.contains(first) || first == "_" else { return false }
        return trimmed.unicodeScalars.allSatisfy {
            CharacterSet.alphanumerics.contains($0) || $0 == "_" || $0 == "-"
        }
    }

    private func isHarmonyThoughtChannelName(_ text: String) -> Bool {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return trimmed == "thought" || trimmed == "thinking"
    }

    private mutating func emitSafeHarmonyPrefix(
        into out: inout [ReasoningSegment],
        tags: [String],
        allowPartialTagAtEnd: Bool
    ) {
        if allowPartialTagAtEnd {
            let safeTail = max(0, (tags.map(\.count).max() ?? 1) - 1)
            guard buffer.count > safeTail else { return }
            let splitAt = buffer.index(buffer.endIndex, offsetBy: -safeTail)
            let safe = String(buffer[..<splitAt])
            appendHarmonyFreeText(safe, into: &out)
            buffer = String(buffer[splitAt...])
        } else {
            appendHarmonyFreeText(buffer, into: &out)
            buffer.removeAll(keepingCapacity: false)
        }
    }

    private mutating func emitSafeHarmonyChannelPrefix(
        into out: inout [ReasoningSegment],
        tags: [String],
        allowPartialTagAtEnd: Bool
    ) {
        if allowPartialTagAtEnd {
            let safeTail = max(0, (tags.map(\.count).max() ?? 1) - 1)
            guard buffer.count > safeTail else { return }
            let splitAt = buffer.index(buffer.endIndex, offsetBy: -safeTail)
            let safe = String(buffer[..<splitAt])
            appendHarmonyChannelText(safe, into: &out)
            buffer = String(buffer[splitAt...])
        } else {
            appendHarmonyChannelText(buffer, into: &out, final: true)
            buffer.removeAll(keepingCapacity: false)
        }
    }

    private func appendHarmonyFreeText(_ text: String, into out: inout [ReasoningSegment]) {
        let cleaned = stripHarmonyControlText(text)
        guard !cleaned.isEmpty else { return }
        out.append(.content(cleaned))
    }

    private func firstRange(of tags: [String], in text: String) -> (Range<String.Index>, String)? {
        var best: (Range<String.Index>, String)?
        for tag in tags where !tag.isEmpty {
            guard let range = text.range(of: tag) else { continue }
            if let current = best {
                if range.lowerBound < current.0.lowerBound
                    || (range.lowerBound == current.0.lowerBound && tag.count > current.1.count)
                {
                    best = (range, tag)
                }
            } else {
                best = (range, tag)
            }
        }
        return best
    }

    private func hasPartialSuffix(of tag: String, in text: String) -> Bool {
        guard !tag.isEmpty else { return false }
        let maxLength = min(tag.count - 1, text.count)
        guard maxLength > 0 else { return false }
        for length in stride(from: maxLength, through: 1, by: -1) {
            let suffix = String(text.suffix(length))
            if tag.hasPrefix(suffix) {
                return true
            }
        }
        return false
    }

    private func stripHarmonyControlText(_ text: String) -> String {
        var cleaned = text
        for marker in [
            "<|start|>assistant",
            "<|start|>system",
            "<|start|>user",
            "<|start|>tool",
            "<|start|>",
            "<|channel|>",
            "<|message|>",
            "<|end|>",
            "<|return|>",
        ] {
            cleaned = cleaned.replacingOccurrences(of: marker, with: "")
        }
        return cleaned
    }

    /// Process the buffer, peeling off as many complete segments as possible.
    /// `allowPartialTagAtEnd` keeps a tail of up to `max(startTag, endTag).count - 1`
    /// characters in the buffer when streaming, so a tag split across
    /// chunks isn't mistakenly emitted as content.
    ///
    /// Tag handling is symmetric — the loop scans for whichever of
    /// `startTag` / `endTag` appears EARLIEST in the buffer and sets state
    /// explicitly based on which one was found (open → reasoning, close →
    /// content). This makes the parser robust to interleaved-thinking
    /// pathologies where the model emits a stray `</think>` while already
    /// in content mode (or a stray `<think>` while already in reasoning).
    /// In the legacy "lookFor only one tag based on current state" design
    /// those stray markers leaked into the visible stream verbatim
    /// (reproduced 2026-04-25 on a MiniMax-Small JANGTQ chat where the
    /// model emitted three `</think>` markers across one assistant turn).
    private mutating func drain(allowPartialTagAtEnd: Bool = true)
        -> [ReasoningSegment]
    {
        var out: [ReasoningSegment] = []

        while !buffer.isEmpty {
            // Tag-search dispatch: stripStrayTags=true scans for both
            // and resolves to whichever appears first; stripStrayTags=
            // false (legacy harmony) only scans for the tag matching
            // the current mode.
            let firstTagRange: Range<String.Index>?
            let firstTagIsOpener: Bool
            if stripStrayTags {
                let openRange = buffer.range(of: startTag)
                let closeRange = buffer.range(of: endTag)
                switch (openRange, closeRange) {
                case (let o?, let c?):
                    if o.lowerBound <= c.lowerBound {
                        firstTagRange = o
                        firstTagIsOpener = true
                    } else {
                        firstTagRange = c
                        firstTagIsOpener = false
                    }
                case (let o?, nil):
                    firstTagRange = o
                    firstTagIsOpener = true
                case (nil, let c?):
                    firstTagRange = c
                    firstTagIsOpener = false
                case (nil, nil):
                    firstTagRange = nil
                    firstTagIsOpener = false
                }
            } else {
                let lookFor = insideReasoning ? endTag : startTag
                firstTagRange = buffer.range(of: lookFor)
                firstTagIsOpener = !insideReasoning
            }
            if let range = firstTagRange {
                // Emit everything before the tag in the current mode.
                let before = String(buffer[..<range.lowerBound])
                if !before.isEmpty {
                    out.append(insideReasoning ? .reasoning(before) : .content(before))
                }
                // Consume the tag itself (never emit it). Set state
                // explicitly per tag identity — open tag → reasoning,
                // close tag → content. With stripStrayTags=true the
                // already-in-state branch is a no-op state-wise but
                // the tag is still consumed.
                buffer.removeSubrange(buffer.startIndex..<range.upperBound)
                insideReasoning = firstTagIsOpener
                continue
            }

            // No complete tag in the buffer. If we might still be assembling
            // a tag prefix at the end, hold back enough characters that a
            // future chunk can complete it.
            if allowPartialTagAtEnd {
                // `max(startTag, endTag).count - 1` — use `max(0, …)` so
                // an edge-case empty tag (e.g. a model-specific override
                // mis-configured at init) doesn't produce a negative
                // `safeTail`, which would make the `offsetBy: -safeTail`
                // move forward past `endIndex` and trap in the stdlib.
                let safeTail = max(0, max(startTag.count, endTag.count) - 1)
                if buffer.count > safeTail {
                    let splitAt = buffer.index(buffer.endIndex, offsetBy: -safeTail)
                    let safe = String(buffer[..<splitAt])
                    if !safe.isEmpty {
                        out.append(insideReasoning ? .reasoning(safe) : .content(safe))
                    }
                    buffer = String(buffer[splitAt...])
                }
            } else {
                // End-of-stream drain: emit everything, no holdback.
                if !buffer.isEmpty {
                    out.append(insideReasoning ? .reasoning(buffer) : .content(buffer))
                    buffer.removeAll(keepingCapacity: false)
                }
            }
            break
        }

        return out
    }
}

// MARK: - Capability-name resolution

extension ReasoningParser {
    /// Build a parser from a `JangCapabilities.reasoningParser` string.
    ///
    /// Accepts every name the JANG converter currently produces plus the
    /// canonical `think_xml` / `harmony` / `none` values. Unknown names
    /// → `nil` (caller should fall back to model-type heuristics or skip
    /// parsing).
    ///
    /// Returns parsers pre-configured for each family's wire format:
    ///
    /// - **`<think>` family** (Qwen 3.5 / 3.6, DeepSeek-R1, GLM 4.x,
    ///   Nemotron, MiniMax) — `<think>…</think>`. The Qwen 3.6 chat
    ///   template prefills `<think>\n` at the end of the assistant
    ///   prompt by default (`enable_thinking=true` branch), so the
    ///   model's first output byte is ALREADY inside a think block.
    ///   To route those pre-`</think>` bytes to `.reasoning` instead of
    ///   leaking them into `.chunk`, we return parsers with
    ///   `startInReasoning=true`. Callers that explicitly disabled
    ///   `enable_thinking` should construct a parser directly with
    ///   `ReasoningParser()` (default `startInReasoning=false`).
    ///
    /// - **`harmony` family** (Gemma-4) — `<|channel>thought\n…<channel|>`.
    ///   Gemma-4's chat template emits this envelope unconditionally
    ///   for the thinking channel (an empty block when
    ///   `enable_thinking=false`, a populated block otherwise). The
    ///   model emits the opening tag explicitly, so `startInReasoning`
    ///   stays false.
    ///
    /// - **`none`** (Mistral, LFM2, plain models) — returns `nil` so the
    ///   pipeline skips reasoning parsing entirely.
    public static func fromCapabilityName(_ name: String?) -> ReasoningParser? {
        guard let name, !name.isEmpty else { return nil }
        let n = name.lowercased()
        let normalized = normalizedReasoningAlias(n)
        let compact = compactReasoningAlias(n)

        if compact.hasPrefix("gemma4") || compact.hasPrefix("gptoss") {
            return ReasoningParser(
                startTag: "<|channel>",
                endTag: "<channel|>",
                startInReasoning: false,
                stripStrayTags: false)
        }

        if normalized.hasPrefix("glm4_moe")
            || normalized.hasPrefix("glm5")
            || normalized.hasPrefix("glm_5")
            || compact.hasPrefix("glm5")
            || normalized.hasPrefix("deepseek")
            || normalized.hasPrefix("laguna")
        {
            return ReasoningParser(startInReasoning: true)
        }

        if normalized.hasPrefix("mistral4")
            || normalized.hasPrefix("mistral_4")
            || normalized.hasPrefix("mistral_small_4")
            || normalized.hasPrefix("mistral_large_4")
        {
            return ReasoningParser(
                startTag: "[THINK]",
                endTag: "[/THINK]",
                startInReasoning: false)
        }

        if normalized.hasPrefix("qwen3_vl")
            || normalized.hasPrefix("qwen3_5_vl")
            || normalized.hasPrefix("qwen3_6_vl")
        {
            return ReasoningParser(startInReasoning: true)
        }

        if compact.hasPrefix("nemotron") {
            return ReasoningParser(startInReasoning: true)
        }

        if compact.hasPrefix("step3p5")
            || compact.hasPrefix("step3p7")
            || compact.hasPrefix("stepfun")
        {
            return ReasoningParser(startInReasoning: true)
        }

        if normalized.hasPrefix("bailing")
            || normalized == "ling"
            || normalized.hasPrefix("ling_")
        {
            return ReasoningParser(startInReasoning: true)
        }

        if compact.hasPrefix("hy3")
            || normalized == "hy_v3"
            || normalized.hasPrefix("hy_v3_")
            || compact.hasPrefix("hunyuan")
            || compact == "tencent"
        {
            // Official Hunyuan v3 suffixes every protocol marker with
            // `:opensource` (`<think:opensource>…</think:opensource>`); the
            // template pre-fills an OPEN think at the prompt tail in
            // high/low reasoning modes and a CLOSED empty pair in no_think —
            // `forPrompt(stampName:promptTail:)` resolves the start state
            // from that tail, which is why these exact tags matter.
            return ReasoningParser(
                startTag: "<think:opensource>",
                endTag: "</think:opensource>",
                startInReasoning: true)
        }

        switch n {
        case "think_xml", "qwen3", "qwen3_5", "qwen35", "qwen3_6", "qwen36",
            "deepseek_r1", "deepseek-r1", "deepseek", "glm", "glm4", "glm5",
            "nemotron", "nemotron_h", "minimax", "minimax_m2",
            "kimi", "kimi_k2", "kimik2",
            "laguna", "laguna_xs", "laguna_s",
            "zaya", "zaya1", "zaya2",
            "step", "stepfun", "step3p5", "step3p7", "step3_5", "step3_7":
            // Start inside the reasoning block — matches the Qwen 3.x
            // family's chat-template default (`enable_thinking=true`
            // prefills `<think>\n` at prompt tail).
            return ReasoningParser(startInReasoning: true)
        case "harmony", "harmony_channel", "gemma4_channel", "gemma4":
            // Gemma-4 harmony-channel envelope. The training template
            // emits `<|channel>thought\n…\n<channel|>` for CoT (see
            // chat_template.jinja line 238), but at inference the
            // model also emits other channel names — `<|channel>`
            // followed by a JSON action block then `<channel|>` for
            // ReAct-style tool hints, `<|channel>analysis…<channel|>`
            // etc. We latch on the bare `<|channel>` opener so ANY
            // Gemma-style channel routes to `.reasoning` and nothing in
            // the envelope leaks into `.chunk`. Simple identifier channel
            // headers after `<|channel>` (for example `thought\n`) are
            // stripped before the reasoning delta reaches osaurus.
            return ReasoningParser(
                startTag: "<|channel>",
                endTag: "<channel|>",
                startInReasoning: false,
                // Harmony format: stray-tag leaks treated as literal
                // content per the legacy A2/A3 contract. The bare
                // `<channel|>` close marker is rare enough mid-content
                // that we don't want to silently strip it.
                stripStrayTags: false)
        case "none", "off", "disabled", "mistral", "gemma":
            return nil
        default:
            return nil
        }
    }

    private static func normalizedReasoningAlias(_ value: String) -> String {
        value
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
            .replacingOccurrences(of: "-", with: "_")
            .replacingOccurrences(of: ".", with: "_")
    }

    private static func compactReasoningAlias(_ value: String) -> String {
        normalizedReasoningAlias(value)
            .replacingOccurrences(of: "_", with: "")
    }

    /// Build a parser that accounts for the actual prompt state.
    ///
    /// Some chat templates prefill the reasoning opener (e.g. Qwen 3.x
    /// default emits `<think>\n` at prompt tail so the model output
    /// begins ALREADY inside a think block) while other template
    /// branches fully open AND close it inside the prompt (e.g. Qwen
    /// 3.x with `enable_thinking=false` emits `<think>\n\n</think>\n\n`
    /// — the model's output is pure content).
    ///
    /// `fromCapabilityName` can only return a stamp-based default.
    /// This method takes the DECODED tail of the prompt and overrides
    /// `startInReasoning` based on which state the prompt ends in.
    ///
    /// - Parameters:
    ///   - stampName: the `reasoningParserName` capability stamp.
    ///   - promptTail: decoded tail of the prompt (enough bytes to
    ///     contain any relevant opener/closer tags). Typically the
    ///     last ~100 characters of the prompt suffice. Pass `nil` to
    ///     fall back to stamp defaults.
    /// - Returns: a parser, or nil if the stamp resolves to no parser.
    public static func forPrompt(
        stampName: String?,
        promptTail: String?
    ) -> ReasoningParser? {
        guard let base = fromCapabilityName(stampName) else { return nil }

        // No prompt hint → use stamp default (whatever insideReasoning
        // was baked into `base` by fromCapabilityName).
        guard let promptTail, !promptTail.isEmpty else { return base }

        // Detect the last tag at the prompt tail.
        let startTag = base.startTag
        let endTag = base.endTag
        let lastOpener = promptTail.range(of: startTag, options: .backwards)
        let lastCloser = promptTail.range(of: endTag, options: .backwards)

        let startInReasoning: Bool
        switch (lastOpener, lastCloser) {
        case (let o?, let c?):
            // Whichever tag appears LATER wins. If closer is after opener
            // (the full block closed in the prompt), we start in content.
            // If opener is after closer (the model already re-opened a
            // block), start in reasoning.
            startInReasoning = o.lowerBound > c.lowerBound
        case (.some, nil):
            // Opener with no closer → prompt ends inside a think block.
            startInReasoning = true
        case (nil, .some):
            // Closer with no opener → prompt ends in content.
            startInReasoning = false
        case (nil, nil):
            // Neither opener nor closer in the prompt tail. The stamps
            // that bake `startInReasoning=true` (think_xml / qwen family)
            // do so to match chat templates that PREFILL `<think>` at
            // the prompt tail. If the tail is missing that opener
            // entirely, the template didn't prefill — e.g. the model
            // is mis-stamped, or an upstream consumer built its own
            // prompt. Starting in reasoning in that case routes the
            // entire answer into `.reasoning` which osaurus renders in
            // the thinking block (reported 2026-04-24 for LFM2 bundles
            // with stale stamps). Safer default: start in content; the
            // parser still latches on `<think>` mid-stream if the model
            // emits one, so Qwen 3.6 interleaved thinking still works.
            startInReasoning = false
        }

        var parser = ReasoningParser(
            startTag: startTag,
            endTag: endTag,
            startInReasoning: startInReasoning,
            // Preserve the family's stray-tag policy from `base` —
            // think_xml family keeps `stripStrayTags: true`, harmony
            // keeps `false`. Without this carry-over, harmony lost
            // its A2/A3 contract whenever `forPrompt(...)` was used.
            stripStrayTags: base.stripStrayTags)
        if startInReasoning && parser.isHarmonyChannelParser {
            parser.insideHarmonyChannel = true
            parser.harmonyChannelIsReasoning = true
            parser.harmonyChannelEndTags = [endTag]
            parser.insideReasoning = true
            if let lastOpener {
                let promptChannelTail = String(promptTail[lastOpener.upperBound...])
                if !promptChannelTail.contains("\n")
                    || promptChannelTail
                        .split(separator: "\n", maxSplits: 1, omittingEmptySubsequences: false)
                        .first
                        .map({ parser.isHarmonyThoughtChannelName(String($0)) }) == true
                {
                    parser.harmonyChannelShouldStripName = true
                    parser.harmonyChannelNameBuffer =
                        promptChannelTail.contains("\n") ? "" : promptChannelTail
                }
            }
        }
        return parser
    }
}

// MARK: - Whole-string convenience

extension ReasoningParser {
    /// One-shot extraction for non-streaming callers — splits a complete
    /// model response into reasoning + visible content.
    ///
    /// - Parameters:
    ///   - text: The full model output.
    ///   - startTag: Override start tag (default `<think>`).
    ///   - endTag: Override end tag (default `</think>`).
    /// - Returns: `(reasoning: String, content: String)`. Empty strings
    ///   if the corresponding segment is absent.
    public static func split(
        _ text: String,
        startTag: String = "<think>",
        endTag: String = "</think>"
    ) -> (reasoning: String, content: String) {
        var parser = ReasoningParser(startTag: startTag, endTag: endTag)
        var segments = parser.feed(text)
        segments.append(contentsOf: parser.flush())
        var reasoning = ""
        var content = ""
        for s in segments {
            switch s {
            case .reasoning(let r): reasoning.append(r)
            case .content(let c): content.append(c)
            }
        }
        return (reasoning, content)
    }
}

// MARK: - model_type → reasoning stamp (factory helper)

/// Pick a reasoning-parser stamp for a given `model_type` when the
/// JANG `capabilities.reasoning_parser` hint is absent. EXPLICIT
/// ALLOWLIST — every model_type not listed here falls through to
/// `"none"` (no reasoning parsing).
///
/// Historical note: both LLMModelFactory and VLMModelFactory used a
/// reverse-allowlist that defaulted everything outside
/// `{gemma4, gemma, mistral}` to `"think_xml"`. That parser starts
/// with `startInReasoning: true` to match Qwen's `<think>`-prefilled
/// prompt tail, so any model_type that DOESN'T emit a think envelope
/// (LFM2, LLaMA, Phi, StarCoder2, Cohere, OpenELM, InternLM2,
/// GPT-OSS, NanoChat, …) had its entire answer routed to
/// `Generation.reasoning(_)` and osaurus rendered it all in the
/// thinking block. Reported by osaurus user 2026-04-24 on LFM2.
///
/// Tests: `ReasoningStampFromModelTypeTests` + per-family
/// regressions in `ReasoningParserTests`.
///
/// - Parameter modelType: The raw `model_type` value from
///   `config.json`. Case-insensitive; empty / nil → `"none"`.
/// - Returns: A capability-name stamp that
///   `ReasoningParser.fromCapabilityName(_:)` understands. Never
///   `nil`; callers pass the returned string through to the parser.
public func reasoningStampFromModelType(_ modelType: String?) -> String {
    guard let modelType, !modelType.isEmpty else { return "none" }
    let t = modelType.lowercased()
    let normalized = t
        .trimmingCharacters(in: .whitespacesAndNewlines)
        .replacingOccurrences(of: "-", with: "_")
        .replacingOccurrences(of: ".", with: "_")
    let compact = normalized.replacingOccurrences(of: "_", with: "")

    // Official Hunyuan v3 uses `:opensource`-suffixed think markers; the
    // dedicated stamp resolves to that parser (see `fromCapabilityName`).
    if compact.hasPrefix("hy3") || compact.hasPrefix("hyv3") || compact.hasPrefix("hunyuan") {
        return "hy_v3"
    }

    // Gemma-4 harmony channel envelope: `<|channel>thought\n…<channel|>`.
    // Distinct from `<think>` XML.
    if compact.hasPrefix("gemma4") {
        return "harmony"
    }

    // DiffusionGemma shares the Gemma-4 chat template family and emits the
    // same `<|channel>thought\n…<channel|>` reasoning envelope.
    if compact.hasPrefix("diffusiongemma") {
        return "harmony"
    }

    // GPT-OSS native Harmony envelope (`<|start|>`, `<|channel|>`,
    // `<|end|>`, `<|return|>`, `<|message|>`). Without this stamp the
    // markers leak into the user-visible chunk stream because the
    // default parser treats them as content. Documented in
    // docs/OSAURUS-PRODUCTION-HANDOFF-2026-05-04.md as the gpt_oss
    // Harmony marker leak.
    if compact.hasPrefix("gptoss") {
        return "harmony"
    }

    // ZAYA1-VL is a sibling multimodal architecture, not the text
    // `zaya` chat-template path. Its production template is selected via
    // the VL sidecar / JANG capability contract and does not open the text
    // ZAYA `<think>` rail by model_type alone.
    if compact.hasPrefix("zaya1vl") {
        return "none"
    }

    // Explicit allowlist of model families that emit `<think>` /
    // `</think>` in their native chat template. These all resolve
    // via `ReasoningParser.fromCapabilityName` to the think_xml
    // parser.
    //
    // Checked as prefix matches so minor-version variants (qwen3_6,
    // qwen3_next_moe, deepseek_v4, kimi_k25, etc.) flow through to
    // the same stamp without an explicit entry each.
    let thinkXmlPrefixes = [
        "qwen3",        // qwen3, qwen3_5, qwen3_6, qwen3_moe, qwen3_next
        "deepseek",     // deepseek_v3, deepseek_v4, deepseek_r1
        "glm4moe",      // glm4_moe, glm4_moe_lite
        "glm5",         // glm5 family
        "minimax",      // minimax, minimax_m2, minimax_m3
        "kimi",         // kimi_k2, kimi_k25
        "nemotronh",    // NemotronH / Cascade series
        "holo",         // Holo3 variants
        "bailing",      // bailing_hybrid / bailing_moe_v2_5 (Ling-2.6-flash) —
                        // chat template emits `<think>...</think>` envelope
                        // when system message contains "detailed thinking on";
                        // jang_config.capabilities.reasoning_parser is
                        // "deepseek_r1" but stamping via model_type prefix
                        // lets non-JANG bundles also resolve correctly.
        "ling",         // Product-name aliases for the same Bailing/Ling
                        // runtime should not bypass the think_xml parser.
        "laguna",       // Poolside Laguna — `laguna_glm_thinking_v5/chat_template.jinja`
                        // emits `<think>...</think>` when enable_thinking=true.
                        // Pre-registered here so that on the day the vmlx
                        // `laguna` model class lands, the reasoning stamp
                        // resolution doesn't need a follow-up edit and
                        // CoT output won't leak into `.chunk` events.
        "zaya",         // Zyphra ZAYA text models. Templates prefill
                        // `<think>` when `enable_thinking=true`; if a
                        // bundle lacks a JANG reasoning stamp, model_type
                        // fallback must still route pre-`</think>` bytes to
                        // `.reasoning` instead of `.chunk`.
        "mimo",         // MiMo-V2 templates use the same `<think>` envelope.
        "step3p5",      // StepFun Step 3.5 text runtime / parser family.
        "step3p7",      // Step 3.7 VLM wrapper uses Step 3.5 text template.
        "stepfun",
    ]
    if thinkXmlPrefixes.contains(where: compact.hasPrefix) {
        return "think_xml"
    }

    // Default: no reasoning envelope. Output flows as plain `.chunk`
    // events with zero `.reasoning` leakage. Covers LFM2, LLaMA,
    // Phi 3/MoE, StarCoder2, Cohere, OpenELM, InternLM2, NanoChat,
    // BitNet, Mistral 3/4, Gemma 2/3/3n, plus any new
    // model_type that lands in LLMModelFactory without an explicit
    // reasoning stamp.
    return "none"
}
