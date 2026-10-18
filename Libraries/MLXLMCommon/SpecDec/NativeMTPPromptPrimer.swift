// Copyright © 2026 Osaurus AI. All rights reserved.

import Foundation
import MLX

/// Native-MTP prompt priming. The trunk already computes the MTP head's input — its hidden state — for every
/// prompt token during prefill. Without priming the head's private cache starts empty, so its attention never
/// sees the prompt and early drafts are context-starved: Allosaurus 0.2 (qwen4_exp) accepted ~45 % of depth-1
/// drafts live against 67 % teacher-forced on the same head (2026-10-17). Priming folds every (hᵢ, xᵢ₊₁) pair
/// the prefill produced into that cache, then the final (hₙ₋₁, first sampled token); the first draft then sits at
/// position n with the prompt as context (vMLX Python `native_mtp_prompt_priming`).
///
/// After a prefix-cache restore the prefill only computes the uncached tail, so only the tail is folded. The
/// head's attention sees relative positions only, so a tail-only history stays consistent with later rows.
/// The head only proposes: priming changes acceptance, never output. `VMLX_NATIVE_MTP_PROMPT_PRIMING=0`
/// disables it (A/B and diagnosis).
public final class NativeMTPPromptPrimer: @unchecked Sendable {
    /// Set by the native-MTP iterator around its prompt prefill; models whose prefill can expose the head
    /// input per chunk hand each chunk to it (`fold`).
    @TaskLocal public static var current: NativeMTPPromptPrimer?

    /// Mutable only so focused tests of head-cache commit mechanics can start from an empty head.
    nonisolated(unsafe) public static var enabled: Bool = {
        let raw = ProcessInfo.processInfo.environment["VMLX_NATIVE_MTP_PROMPT_PRIMING"]?
            .trimmingCharacters(in: .whitespaces).lowercased()
        return !["0", "false", "no", "off"].contains(raw ?? "")
    }()

    private let model: any NativeMTPModel
    private let cache: [KVCache]
    /// Hidden state of the last folded position, not yet paired with the token that follows it.
    private var pending: MLXArray?
    /// Head-cache rows written so far.
    public private(set) var foldedRows = 0
    /// Prompt positions handed over by the model.
    public private(set) var receivedPositions = 0

    public init(model: any NativeMTPModel, cache: [KVCache]) {
        self.model = model
        self.cache = cache
    }

    /// `hidden` [1, S, H] is the head input of the chunk's `tokens` [1, S] (in prompt order, chunks contiguous).
    /// The head rows are evaluated before returning, so a long prompt never holds more than one chunk's graph.
    public func fold(hidden: MLXArray, tokens: MLXArray) {
        let count = tokens.dim(-1)
        guard count > 0, hidden.ndim == 3, hidden.dim(1) == count else { return }
        let ids = tokens.reshaped(1, count)
        receivedPositions += count
        let rowHidden: MLXArray
        let rowTokens: MLXArray
        if let pending {
            rowHidden = count > 1 ? concatenated([pending, hidden[0..., 0 ..< (count - 1)]], axis: 1) : pending
            rowTokens = ids
        } else {
            guard count > 1 else {
                self.pending = hidden[0..., (count - 1)...]
                return
            }
            rowHidden = hidden[0..., 0 ..< (count - 1)]
            rowTokens = ids[0..., 1...]
        }
        model.nativeMTPPrime(hiddenStates: rowHidden, nextTokenIds: rowTokens, cache: cache)
        foldedRows += rowTokens.dim(1)
        pending = hidden[0..., (count - 1)...]
        // Materialize per chunk, as the prefill materializes the trunk cache. Left lazy, every chunk's head
        // forward stayed in one graph until `finish`: 52 GB peak footprint at a 32k-token prompt on Allosaurus
        // 0.2 (10.9 GB without the head) — fatal on a 48 GB Mac.
        MLX.eval(cache, pending!)
    }

    /// Pair the last prompt position with the first sampled token and materialize the head cache.
    public func finish(firstToken: MLXArray) {
        if let pending {
            model.nativeMTPPrime(
                hiddenStates: pending, nextTokenIds: firstToken.reshaped(1, 1), cache: cache)
            foldedRows += 1
            self.pending = nil
        }
        MLX.eval(cache)
    }
}

/// A native-MTP model whose head accepts a multi-row (hidden, next token) window — the prompt history — in one
/// forward, and whose prefill hands its chunks to `NativeMTPPromptPrimer.current`. Only such models are primed;
/// `nativeMTPForward`'s base contract is one recursive draft step.
public protocol NativeMTPPromptPrimingModel: NativeMTPModel {}

public extension NativeMTPModel {
    /// Write head-cache rows for (hidden, next token) pairs without drafting. The default runs the draft forward
    /// and leaves its logits unevaluated (lazy), so only the cache rows are computed.
    func nativeMTPPrime(hiddenStates: MLXArray, nextTokenIds: MLXArray, cache: [KVCache]) {
        _ = nativeMTPForward(hiddenStates: hiddenStates, nextTokenIds: nextTokenIds, cache: cache)
    }
}
