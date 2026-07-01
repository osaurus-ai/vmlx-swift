// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation

/// Streaming text-level stop-sequence matcher.
///
/// Sits at the tail of the generation pipeline, after `ReasoningParser`
/// (which emits `.reasoning` events) and `ToolCallProcessor` (which
/// emits `.toolCall`). Matches against the user-visible `.chunk` text
/// only — reasoning and tool-call bytes are scoped separately per the
/// `Generation` contract.
///
/// ## Algorithm
///
/// Keeps a rolling tail buffer of size `maxStopLen - 1` where
/// `maxStopLen` is the longest configured stop string. On each `feed`,
/// the new piece is appended, every stop string is checked, and the
/// longest safe prefix (text that CANNOT be the start of a stop string)
/// is returned for emission. Remaining tail is held until the next
/// `feed` or `flush`.
///
/// This is the same pattern as `NaiveStreamingDetokenizer` trim + hold,
/// and `ToolCallProcessor.partialMatch` — guarantees that no stop-string
/// byte is ever emitted downstream.
///
/// ## Semantics
///
/// - When any stop string is fully present in the buffer, `feed` returns
///   `.stopped(emit: prefix)` where `prefix` is the buffered text up to
///   the start of the matched stop string. The stop string itself is
///   truncated — the caller must not emit it.
/// - Otherwise `feed` returns `.streaming(emit: safePrefix)` where
///   `safePrefix` is every character that cannot be part of any stop
///   string given the current tail. The stored tail shrinks to at most
///   `maxStopLen - 1` characters.
/// - `flush()` on end-of-stream returns whatever is in the buffer (no
///   more tokens will arrive, so the tail is safe to emit).
///
/// ## Performance
///
/// `feed` is O(L · S) per call where L is the tail length (bounded by
/// `maxStopLen`) and S is the number of stop strings. For typical
/// OpenAI-compatible usage (≤4 stop strings, each ≤32 chars) this is
/// negligible relative to detokenization.
public struct StopStringMatcher: Sendable {

    public enum FeedResult: Sendable, Equatable {
        /// No stop match yet. `emit` is the safe prefix that can be
        /// flushed downstream; remaining tail stays in the matcher.
        case streaming(emit: String)
        /// A stop string matched. `emit` is text up to the match start;
        /// the caller should then halt generation.
        case stopped(emit: String)
    }

    /// Stop strings, in priority order. Duplicates and empty strings
    /// are filtered at init — an empty stop string would match
    /// immediately and is treated as "no stop string". If any stop
    /// strings remain, `maxStopLen` > 0.
    public let stopStrings: [String]

    /// Longest configured stop string length. 0 when the matcher is
    /// disabled (no valid stop strings).
    public let maxStopLen: Int

    /// Rolling buffer — every character in this buffer has not yet
    /// been emitted. Size is bounded by `maxStopLen - 1` plus whatever
    /// was just appended via `feed`.
    private var buffer: String = ""

    public var isEnabled: Bool { !stopStrings.isEmpty }

    /// Create a matcher. Empty / duplicate stop strings are stripped.
    /// If no valid stop strings remain, the matcher is disabled and
    /// `feed` is a pure pass-through.
    public init(stopStrings: [String]) {
        var seen = Set<String>()
        let cleaned = stopStrings.compactMap { s -> String? in
            guard !s.isEmpty, !seen.contains(s) else { return nil }
            seen.insert(s)
            return s
        }
        self.stopStrings = cleaned
        self.maxStopLen = cleaned.map(\.count).max() ?? 0
    }

    /// Feed a new piece of decoded, reasoning-stripped, tool-call-
    /// stripped user-visible text. Returns either a streaming delta
    /// with whatever is safe to emit, or a stopped result with the
    /// final pre-stop emit.
    public mutating func feed(_ piece: String) -> FeedResult {
        guard isEnabled else {
            return .streaming(emit: piece)
        }

        buffer.append(piece)

        // 1. Any complete stop string inside the buffer?
        //    Use the earliest match among all configured stops (iterate
        //    stopStrings in priority order; `range(of:)` gives the
        //    first-occurrence position for each, then pick the lowest).
        var earliestMatch: (range: Range<String.Index>, stop: String)? = nil
        for stop in stopStrings {
            if let r = buffer.range(of: stop) {
                if let current = earliestMatch {
                    if r.lowerBound < current.range.lowerBound {
                        earliestMatch = (r, stop)
                    }
                } else {
                    earliestMatch = (r, stop)
                }
            }
        }
        if let match = earliestMatch {
            let emit = String(buffer[..<match.range.lowerBound])
            buffer.removeAll(keepingCapacity: false)
            return .stopped(emit: emit)
        }

        // 2. No complete match. Emit everything that CANNOT be the
        //    start of a future stop string, hold the rest in the tail.
        //    Safe-prefix length = max(0, buffer.count - (maxStopLen - 1)).
        //    Rationale: any character in the last `maxStopLen - 1`
        //    positions could still be the first character of a stop
        //    string whose remaining bytes arrive in the next feed.
        let holdLen = maxStopLen - 1
        let bufCount = buffer.count
        if bufCount <= holdLen {
            // Whole buffer is potentially-a-prefix; emit nothing.
            return .streaming(emit: "")
        }
        let splitOffset = bufCount - holdLen
        let splitIdx = buffer.index(buffer.startIndex, offsetBy: splitOffset)
        let emit = String(buffer[..<splitIdx])
        buffer = String(buffer[splitIdx...])
        return .streaming(emit: emit)
    }

    /// End-of-stream flush. Returns any tail that was withheld while
    /// waiting for disambiguation. Called exactly once per stream.
    public mutating func flush() -> String {
        guard isEnabled else { return "" }
        let tail = buffer
        buffer.removeAll(keepingCapacity: false)
        return tail
    }
}

struct ResolvedStopSequences: Sendable, Equatable {
    var tokenIDs: Set<Int>
    var textStopStrings: [String]

    init(tokenIDs: Set<Int> = [], textStopStrings: [String] = []) {
        self.tokenIDs = tokenIDs
        self.textStopStrings = textStopStrings
    }
}

let commonEndTokenStrings: [String] = [
    "<|im_end|>",       // Qwen / Mistral 3 / NemotronH-Omni / many
    "<|endoftext|>",    // Qwen2/3, GPT-style
    "<|eot_id|>",       // Llama 3.x
    "<|end_of_text|>",  // Llama 3.x alt
    "<|end|>",          // Phi 3, Phi 4
    "<|end_of_turn|>",  // Gemma family
    "<end_of_turn>",    // Gemma 2/3 alt spelling
    "〈|EOS|〉",          // Laguna / Poolside
    "<｜end▁of▁sentence｜>",  // DeepSeek V4 wide-pipe spelling
    "<|EOT|>",          // Uppercase EOT variants in some shipped templates
]

func mergeStopStrings(_ explicit: [String], _ defaults: [String]) -> [String] {
    var seen = Set<String>()
    var merged: [String] = []
    for string in explicit + defaults {
        guard !string.isEmpty, seen.insert(string).inserted else { continue }
        merged.append(string)
    }
    return merged
}

func resolveStopSequences(
    modelConfiguration: ModelConfiguration,
    tokenizer: Tokenizer,
    includeUnknownToken: Bool = false
) -> ResolvedStopSequences {
    var tokenIDs = modelConfiguration.eosTokenIds
    if let tokenizerEOS = tokenizer.eosTokenId {
        tokenIDs.insert(tokenizerEOS)
    }
    if includeUnknownToken, let unknownID = tokenizer.unknownTokenId {
        tokenIDs.insert(unknownID)
    }

    var textStopStrings: [String] = []
    for token in modelConfiguration.extraEOSTokens {
        resolveStopTokenString(
            token,
            tokenizer: tokenizer,
            tokenIDs: &tokenIDs,
            textStopStrings: &textStopStrings,
            allowExactTextFallback: true)
    }
    // The hard-coded `commonEndTokenStrings` fallback is applied ONLY to models that did
    // NOT authoritatively declare their stop set in `generation_config.json`. When a model
    // ships that file with an `eos_token_id` (the HF-canonical, mlx-lm-honored generation-time
    // stop list), the declaration is complete and authoritative — augmenting it with hard-coded
    // end strings can only ADD stops the model deliberately omitted. That over-reach is exactly
    // what made gpt-oss (OpenAI harmony) halt at the `<|end|>` channel separator that its own
    // `generation_config.json` correctly excludes from eos. So: trust the declaration when present;
    // keep the blanket only as a safety net for under-declared conversions that ship no
    // `generation_config.json` eos at all (e.g. some Phi-3 builds whose only `<|end|>` signal is
    // the tokenizer eos). `generationDefaults` is nil for any caller that didn't load one, so this
    // is conservative by default — the blanket is skipped only on positive evidence of a declaration.
    let hasAuthoritativeEOS =
        !(modelConfiguration.generationDefaults?.eosTokenIds?.values ?? []).isEmpty
    if !hasAuthoritativeEOS {
        for token in commonEndTokenStrings {
            resolveStopTokenString(
                token,
                tokenizer: tokenizer,
                tokenIDs: &tokenIDs,
                textStopStrings: &textStopStrings,
                allowExactTextFallback: false)
        }
    }

    return ResolvedStopSequences(
        tokenIDs: tokenIDs,
        textStopStrings: mergeStopStrings([], textStopStrings))
}

private func resolveStopTokenString(
    _ token: String,
    tokenizer: Tokenizer,
    tokenIDs: inout Set<Int>,
    textStopStrings: inout [String],
    allowExactTextFallback: Bool
) {
    if let id = tokenizer.convertTokenToId(token) {
        if let unknownID = tokenizer.unknownTokenId, id == unknownID {
            textStopStrings.append(contentsOf: textFallbackStopStrings(
                for: token,
                allowExactTextFallback: allowExactTextFallback))
            return
        }
        tokenIDs.insert(id)
        return
    }
    textStopStrings.append(contentsOf: textFallbackStopStrings(
        for: token,
        allowExactTextFallback: allowExactTextFallback))
}

private func textFallbackStopStrings(
    for token: String,
    allowExactTextFallback: Bool
) -> [String] {
    switch token {
    case "<｜end▁of▁sentence｜>":
        return [
            "<｜end▁of▁sentence｜>",
            "<｜end▁of▁sentence>",
            "<｜end▁of▁sent",
        ]
    default:
        if allowExactTextFallback, token.hasPrefix("<") || token.hasPrefix("〈") {
            return [token]
        }
        return []
    }
}
