// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX

/// An explicit, opt-in ceiling on how many tokens a reasoning block may run
/// before its close token is required.
///
/// ## Why this is not the banned mechanism
///
/// `NoHiddenReasoningCloseBiasFocusedTests` bans an earlier mechanism that
/// capped thinking by biasing the close token AUTOMATICALLY, from inside the
/// decode path, with no caller opt-in and nothing surfaced. The objection was
/// that generation silently stopped being the model's own output.
///
/// This type keeps that objection satisfied:
///
/// - **Off unless asked.** `armIfNeeded` returns nil unless
///   `VMLX_REASONING_BUDGET` names a positive token count. No call site turns
///   it on by default, and there is no "automatic" variant.
/// - **It never biases.** There is no nudge toward closing and no probability
///   reshaping while the budget is unspent — logits pass through untouched.
///   At the ceiling it forces the close token exactly once and then disarms
///   permanently, so the model writes every other token itself.
/// - **The effect is visible.** A capped block closes, so the turn produces a
///   real answer instead of the "thinking didn't close" chip. The transition
///   is reported through ``didForceClose`` and traced under
///   `VMLX_REASONING_BUDGET_TRACE`.
///
/// It is the upper bound to `MinimumReasoningFloor`'s lower bound: the floor
/// can only DELAY a close, this can only REQUIRE one, and neither can make the
/// model prefer a close while it still has budget.
///
/// ## Why a budget is needed at all
///
/// Reasoning depth is otherwise bounded only by prose. DSV4's "Low" rail is a
/// preface asking for one or two sentences, and its own encoder comment
/// concedes that hard first prompts still reason at length on that rail.
/// Measured live on DSV4 Flash: a single spec-heavy prompt produced a
/// 38,494-character think block over 10,554 tokens, hit the output cap
/// (`stop=length`), and returned 227 characters of answer — nothing usable.
/// Sampling was not at fault; the run was verified at the bundle's own
/// `temperature 0.6 / top_p 0.95`.
///
/// ## Per-family close tokens
///
/// Families delimit reasoning differently, so the close token is resolved by
/// asking the tokenizer which candidate spelling round-trips rather than
/// assuming `</think>`. `convertTokenToId` returns the unk id for unknown
/// tokens, so id existence alone proves nothing — every candidate is
/// round-tripped through `convertIdToToken`.
public enum ReasoningBudget {

    public struct Armed: Sendable {
        public let closeTokenID: Int
        public let tokenCount: Int
        /// Empty when the prompt already opened reasoning (count immediately).
        /// Otherwise the ids whose appearance starts the count.
        public let startTokenIDs: [Int]
        /// All open-tag ids in this vocab, so the ceiling can ban reopening
        /// even when the prompt primed the block (startTokenIDs empty).
        public let openTokenIDs: [Int]
    }

    /// Close-tag spellings across the shipped families, most specific first.
    /// A family whose close tag is absent from a bundle's vocab simply never
    /// arms — the budget is skipped, not approximated with a wrong token.
    public static let closeTokenCandidates: [String] = [
        "</think>",  // DSV4, Qwen 3.5/3.6, Ornith, Raptor/Laguna (deepseek_r1)
        "</thinking>",
        "<|/think|>",
        "<|end_thought|>",
        "<|inner_suffix|>",  // Apertus 1.5 (apertus1p5)
        "<end_of_turn>",
    ]

    /// Open-tag spellings, paired with `closeTokenCandidates` by family.
    /// Needed because only some templates PRIME the open tag (DSV4 renders
    /// `…assistant<think>`); others leave the model to emit it as its first
    /// generated token (VibeThinker, several Qwen 3.x fine-tunes). A budget
    /// that only armed on a primed tail would silently skip that second group.
    public static let openTokenCandidates: [String] = [
        "<think>", "<thinking>", "<|think|>", "<|start_thought|>",
        "<|inner_prefix|>",  // Apertus 1.5: its template primes neither tag, so the model
                             // emits the open marker itself as its first generated token.
    ]

    /// Resolve the open-token ids that exist in this bundle's vocab, so a
    /// model that opens its own reasoning can still be bounded.
    public static func openTokenIDs(tokenizer: any Tokenizer) -> [Int] {
        openTokenCandidates.compactMap { candidate in
            guard let id = tokenizer.convertTokenToId(candidate),
                tokenizer.convertIdToToken(id) == candidate
            else { return nil }
            return id
        }
    }

    /// Token ceiling from `VMLX_REASONING_BUDGET`. Absent or non-positive
    /// means the budget is inert, which is the default everywhere.
    public static var configuredTokenCount: Int? {
        guard
            let raw = ProcessInfo.processInfo.environment["VMLX_REASONING_BUDGET"],
            let parsed = Int(raw.trimmingCharacters(in: .whitespaces)),
            parsed > 0
        else { return nil }
        return parsed
    }

    /// Resolve a close token id that genuinely exists in this bundle's vocab.
    public static func closeTokenID(tokenizer: any Tokenizer) -> Int? {
        for candidate in closeTokenCandidates {
            guard let id = tokenizer.convertTokenToId(candidate),
                tokenizer.convertIdToToken(id) == candidate
            else { continue }
            return id
        }
        return nil
    }

    /// Arm only when a budget was explicitly configured. The counter starts
    /// immediately when the prompt tail already opens a reasoning block, and
    /// otherwise waits until the model emits an open tag itself — so both
    /// template-primed and self-opening families are covered, and a turn that
    /// never reasons is never touched.
    public static func armIfNeeded(
        tokenizer: any Tokenizer,
        promptTail: String?
    ) -> Armed? {
        guard let budget = configuredTokenCount else { return nil }
        return arm(tokenizer: tokenizer, promptTail: promptTail, tokenCount: budget)
    }

    /// Arm with an explicitly requested token count — the per-request form of
    /// the env-armed `armIfNeeded`, for serving layers that must bound
    /// reasoning on SOME requests without a process-global setting. The
    /// canonical case: an OpenAI-compatible client sends a finite
    /// `max_tokens` and reads only `content` — a think block that spends the
    /// whole cap returns an empty answer to a client that cannot see
    /// `reasoning_content` at all. Same opt-in discipline as the env path:
    /// the caller names a positive count, nothing arms by default, and the
    /// primed-vs-self-opening resolution is identical.
    public static func arm(
        tokenizer: any Tokenizer,
        promptTail: String?,
        tokenCount: Int
    ) -> Armed? {
        guard tokenCount > 0 else { return nil }
        guard let closeID = closeTokenID(tokenizer: tokenizer) else { return nil }
        let primed = promptTail.map(promptTailOpensReasoning) ?? false
        let openIDs = primed ? [] : openTokenIDs(tokenizer: tokenizer)
        // Not primed and no open tag in vocab: nothing could ever start a
        // reasoning block, so stay inert rather than counting plain output.
        guard primed || !openIDs.isEmpty else { return nil }
        return Armed(
            closeTokenID: closeID, tokenCount: tokenCount, startTokenIDs: openIDs,
            openTokenIDs: openTokenIDs(tokenizer: tokenizer))
    }

    /// True when the rendered prompt ends with an opened, unclosed reasoning
    /// block. Mirrors the floor's `…<think>` test but covers the other
    /// families' open spellings too.
    ///
    /// Trailing whitespace is ignored before the suffix check: Qwen 3.x
    /// templates prime `<think>\n` — with the literal newline — so a strict
    /// `hasSuffix("<think>")` classified those prompts as un-primed, the
    /// counter waited for an open tag the model never emits, and the ceiling
    /// stayed silently inert on exactly the family that needed it (verified
    /// live on Qwen3.8: a 300-token cap produced 300 reasoning tokens and an
    /// empty answer with a budget requested).
    public static func promptTailOpensReasoning(_ tail: String) -> Bool {
        let openers = ["<think>", "<thinking>", "<|think|>", "<|start_thought|>"]
        var trimmed = Substring(tail)
        while let last = trimmed.last, last.isWhitespace || last.isNewline {
            trimmed = trimmed.dropLast()
        }
        return openers.contains { trimmed.hasSuffix($0) }
    }
}

/// Requires a reasoning close token once the configured budget is spent.
///
/// While `remaining > 0` this is a pass-through: logits are returned byte for
/// byte, so nothing about the model's own distribution changes. On the step
/// where the budget reaches zero it masks everything except the close token —
/// a requirement, not a bias — and then disarms for the rest of the
/// generation so the answer and any tool call are sampled normally.
public struct ReasoningBudgetProcessor: LogitProcessor {
    private let closeTokenID: Int
    /// Empty means the prompt already opened reasoning, so counting starts on
    /// the first sampled token. Otherwise counting waits for one of these.
    private let startTokenIDs: Set<Int>
    /// Every open-tag id, banned after the ceiling so the block cannot reopen.
    private let openTokenIDs: [Int]
    private var started: Bool
    private var remaining: Int
    private var spent = false
    private let negInf = MLXArray(-Float.infinity)
    private let trace = ProcessInfo.processInfo.environment[
        "VMLX_REASONING_BUDGET_TRACE"] == "1"

    /// True once the close token has been required. Lets callers report that
    /// the block was capped rather than closed by the model.
    public private(set) var didForceClose = false

    public init(
        closeTokenID: Int, tokenCount: Int, startTokenIDs: [Int] = [],
        openTokenIDs: [Int] = []
    ) {
        self.closeTokenID = closeTokenID
        self.startTokenIDs = Set(startTokenIDs)
        self.openTokenIDs = openTokenIDs.isEmpty ? startTokenIDs : openTokenIDs
        self.started = startTokenIDs.isEmpty
        self.remaining = max(0, tokenCount)
    }

    public mutating func prompt(_ prompt: MLXArray) {}

    public func process(logits: MLXArray) -> MLXArray {
        // Ceiling already enforced: the block was closed, so the only thing
        // still masked is REOPENING it. Without this the model simply emits a
        // fresh `<think>` and keeps reasoning, which is exactly what a live
        // A/B showed — think length did not track the budget at all until
        // reopening was banned.
        if spent {
            guard !openTokenIDs.isEmpty else { return logits }
            let vocabSize = logits.dim(-1)
            let valid = openTokenIDs.filter { $0 >= 0 && $0 < vocabSize }
            guard !valid.isEmpty else { return logits }
            // Same out-of-place rule as above: never write into a logits
            // buffer that the caller still owns.
            let ids = MLXArray(Int32(0) ..< Int32(vocabSize))
            var banned = ids .== MLXArray(Int32(valid[0]))
            for id in valid.dropFirst() {
                banned = banned .|| (ids .== MLXArray(Int32(id)))
            }
            return MLX.where(banned, negInf, logits)
        }
        // Not reasoning yet, or budget unspent: never touch the distribution.
        guard started, remaining <= 0 else { return logits }
        let vocabSize = logits.dim(-1)
        guard closeTokenID >= 0, closeTokenID < vocabSize else { return logits }
        if trace {
            let line = "[vmlx][reasoning-budget] requiring close id=\(closeTokenID)\n"
            FileHandle.standardError.write(Data(line.utf8))
        }
        // Build the mask OUT OF PLACE. `MLXArray` is a class and subscript
        // assignment aliases its buffer, so writing one column into a `full`
        // array is not a safe way to keep a single logit — a live A/B showed
        // the close never actually winning, which is why raising the budget
        // produced MORE thinking rather than less.
        let ids = MLXArray(Int32(0) ..< Int32(vocabSize))
        let keep = ids .== MLXArray(Int32(closeTokenID))
        return MLX.where(keep, logits, negInf)
    }

    public mutating func didSample(token: MLXArray) {
        if !started {
            let id = token.reshaped(-1)[0].item(Int.self)
            guard startTokenIDs.contains(id) else { return }
            started = true
            if trace {
                let line = "[vmlx][reasoning-budget] reasoning opened by model; counting\n"
                FileHandle.standardError.write(Data(line.utf8))
            }
            return
        }
        if remaining > 0 {
            remaining -= 1
            return
        }
        guard !spent else { return }
        spent = true
        didForceClose = true
        if trace {
            let line = "[vmlx][reasoning-budget] close required; budget disarmed\n"
            FileHandle.standardError.write(Data(line.utf8))
        }
    }
}
