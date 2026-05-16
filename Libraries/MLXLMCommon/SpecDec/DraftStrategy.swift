// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation

/// Speculative-decoding strategy for a single generation request.
///
/// Exposed on ``GenerateParameters/draftStrategy``. `nil` on that field means
/// the classic non-speculative path runs (byte-identical with upstream
/// ml-explore/mlx-swift-lm). Setting a non-nil value opts the generation
/// into the speculative runtime in `Libraries/MLXLMCommon/SpecDec/`.
///
/// ## Strategies
///
/// - ``none`` — explicit opt-out (same behaviour as setting
///   `draftStrategy = nil`). Kept so a caller can write
///   `parameters.draftStrategy = .none` to make intent explicit.
///
/// - ``autoregressive(draftModel:numDraftTokens:)`` — the existing
///   ``SpeculativeTokenIterator`` path (port of ml-explore/mlx-swift-lm#173).
///   Classic: a smaller draft model runs *D* sequential forward passes per
///   round; main model verifies in one forward. Typical speedup 1.3-1.8×.
///
/// - ``dflash(drafterPath:blockSize:)`` — DFlash block-diffusion drafter
///   ([arXiv 2602.06036](https://arxiv.org/abs/2602.06036)). ONE drafter
///   forward produces a whole block of draft logits; verification still
///   runs over a linear trajectory. Drafter weights live at `drafterPath`
///   (a HuggingFace `z-lab/<model>-DFlash` snapshot directory).
///
/// - ``ddtree(drafterPath:branchingBudget:blockSize:)`` — DDTree best-first
///   heap tree ([arXiv 2604.12989](https://arxiv.org/abs/2604.12989)).
///   Strict superset of ``dflash`` — the linear DFlash path is the
///   degenerate `branchingBudget = 1` tree. Verifies the full tree in one
///   target forward via an ancestor-only attention mask. Typical speedup
///   4-7× over autoregressive on pure-attention models; smaller gains on
///   hybrid-SSM models until Phase 3 per-node recurrent forking lands.
///
/// - ``nativeMTP(depth:)`` — uses a native MTP sidecar already present in
///   the target model artifact. This never loads a separate draft model and
///   is valid only for model instances conforming to ``NativeMTPModel``.
///
/// ## Picking a block size / budget
///
/// - `blockSize` (DFlash, DDTree) — number of future positions the drafter
///   emits per round. 4-8 is typical. Larger values amortise the drafter
///   forward but reduce acceptance on tail tokens. Match the drafter
///   checkpoint's training value (see HF `config.json` under
///   `block_size` for z-lab drafters).
/// - `branchingBudget` (DDTree) — maximum tree nodes (excluding root).
///   Paper recommends 32-64 for greedy decoding; 16-24 for sampling.
public enum DraftStrategy: @unchecked Sendable {

    /// No speculative decoding. Same as `nil` on
    /// ``GenerateParameters/draftStrategy``.
    case none

    /// Classic autoregressive draft-model speculative decoding.
    ///
    /// - Parameters:
    ///   - draftModel: the draft ``LanguageModel``. Must share the target's
    ///     tokenizer.
    ///   - numDraftTokens: number of tokens the draft model proposes per
    ///     round (typical 2-4).
    case autoregressive(draftModel: any LanguageModel, numDraftTokens: Int)

    /// DFlash — block-diffusion drafter with linear verification.
    ///
    /// - Parameters:
    ///   - drafterPath: directory containing the drafter's safetensors +
    ///     `config.json` (e.g. a local HF snapshot of
    ///     `z-lab/Qwen3.5-27B-DFlash`).
    ///   - blockSize: number of positions the drafter emits per round. Must
    ///     match the drafter's training `block_size`.
    case dflash(drafterPath: URL, blockSize: Int)

    /// DDTree — DFlash drafter + best-first tree verification.
    ///
    /// - Parameters:
    ///   - drafterPath: same as ``dflash(drafterPath:blockSize:)``.
    ///   - branchingBudget: maximum number of tree nodes (excluding root).
    ///     Controls the branching factor — higher values increase the
    ///     target verify cost but raise acceptance length.
    ///   - blockSize: positions emitted per drafter forward. Usually equals
    ///     the drafter's training value.
    case ddtree(drafterPath: URL, branchingBudget: Int, blockSize: Int)

    /// Native model-owned MTP draft/verify.
    ///
    /// This path is intentionally explicit. The model must have been loaded
    /// with real MTP tensors and `NativeMTPActivation` must have allowed them
    /// through the loader; otherwise the iterator refuses to start.
    case nativeMTP(depth: Int)

    /// Discriminator usable as a stable dictionary key / log label.
    /// Does not expose associated values.
    public var kindName: String {
        switch self {
        case .none: return "none"
        case .autoregressive: return "autoregressive"
        case .dflash: return "dflash"
        case .ddtree: return "ddtree"
        case .nativeMTP: return "native_mtp"
        }
    }

    /// True if this strategy activates the SpecDec runtime in
    /// `Libraries/MLXLMCommon/SpecDec/`. False for `.none` and
    /// `.autoregressive` which use pre-existing code paths.
    public var usesBlockDiffusion: Bool {
        switch self {
        case .dflash, .ddtree: return true
        case .none, .autoregressive, .nativeMTP: return false
        }
    }

    public var usesNativeMTP: Bool {
        if case .nativeMTP = self { return true }
        return false
    }
}
