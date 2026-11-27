// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Exposes a target model's token embedding + LM head so the SpecDec
// runtime can:
//   1. Embed the drafter's block input (bonus + mask tokens) via the
//      SHARED target embedding (DFlash drafters do not ship their own).
//   2. Project the drafter's per-position hidden output through the
//      target's LM head to obtain draft logits.
//
// This mirrors the Python reference in z-lab/dflash/dflash.py
// `spec_generate` which does:
//     noise_embedding = target.model.embed_tokens(block_output_ids)
//     draft_logits = target.lm_head(self(...))
//
// Models that don't conform to this protocol cannot be used as DFlash
// targets — SpecDecRuntime throws
// `SpecDecError.targetDoesNotSupportHiddenStateCapture` on first use.

import Foundation
import MLX

/// Language-model extension exposing the target's token embedding and
/// LM head so the DFlash drafter can share both with its target.
///
/// `embed` and `projectToLogits` must be mutually consistent — when the
/// target's embedding is TIED to its LM head (word-embedding tying),
/// `projectToLogits` MUST apply the transposed embedding, matching what
/// the target's own forward does internally.
public protocol TokenEmbedderModel: LanguageModel {

    /// Embed token IDs.
    ///
    /// - Parameter tokenIds: `(B, L)` Int32 token IDs.
    /// - Returns: `(B, L, hidden)` post-embedding tensor, identical to
    ///   what the target's own forward uses as its first-layer input.
    func embed(_ tokenIds: MLXArray) -> MLXArray

    /// Project hidden states to vocabulary logits.
    ///
    /// - Parameter hidden: `(B, L, hidden)` post-final-norm activation.
    /// - Returns: `(B, L, vocab)` logits — byte-identical to what the
    ///   target's own forward produces from the same hidden.
    func projectToLogits(_ hidden: MLXArray) -> MLXArray

    /// Padded checkpoint vocabulary size when the target's `lm_head` and the drafter's codebooks
    /// are stored at a padded width (e.g. Inkling-2-Mini: 201024) while valid logits are sliced
    /// to the unpadded vocabulary (200058). `nil` when logits width equals `config.vocab_size`.
    var paddedVocabularySize: Int? { get }
}

extension TokenEmbedderModel {
    public var paddedVocabularySize: Int? { nil }
}
