// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Optional capability that target models opt into so the DFlash drafter
// can read per-layer hidden states during a target forward pass.
//
// The drafter reads target hidden states at `dflash_config.target_layer_ids`
// (5 specific blocks, spread across the target's depth) and projects them
// via its `fc` layer into the context K/V it attends to. Without this hook
// the drafter has no conditioning signal, and acceptance falls to zero.
//
// Byte-compatibility rule: a forward with empty `captureLayerIDs` must
// produce identical logits to the plain `callAsFunction(_:cache:)` path.
// Capture is a pure side-effect — no sampling changes, no mask changes,
// no new gradients.

import Foundation
import MLX

/// Language-model extension that exposes per-block hidden states during
/// a forward pass. Required by DFlash drafters.
///
/// Layer indexing is **0-based on the block output**: `captureLayerIDs
/// = {0, 5, 10, 15, 20}` captures `h` after blocks 0, 5, 10, 15, 20
/// respectively. Implementers must capture *after* the block and *before*
/// passing to the next block (or to the final norm). HF config's
/// `dflash_config.target_layer_ids` ARE 0-based indices into
/// `target.model.layers` (per z-lab/dflash `_patch_model`) — pass them
/// through directly, no shift needed.
///
/// Models that don't conform to this protocol cannot be used as DFlash
/// targets — `SpecDecRuntime` throws
/// ``SpecDecError/targetDoesNotSupportHiddenStateCapture`` on first
/// forward.
public protocol HiddenStateCaptureModel: LanguageModel {

    /// Forward pass with per-layer hidden-state capture.
    ///
    /// - Parameters:
    ///   - inputs: same shape as `callAsFunction(_:cache:)`.
    ///   - cache: same as `callAsFunction(_:cache:)`.
    ///   - captureLayerIDs: 0-based block indices to capture. Empty set
    ///     means "no capture" — the function's output must be identical
    ///     to `callAsFunction(_:cache:)` byte-for-byte in that case.
    /// - Returns: `(logits, captured)` where `captured[layerID]` is the
    ///   `(B, L, hidden)` tensor produced by block `layerID`.
    func callAsFunction(
        _ inputs: MLXArray,
        cache: [KVCache]?,
        captureLayerIDs: Set<Int>
    ) -> (logits: MLXArray, capturedHiddenStates: [Int: MLXArray])
}

extension HiddenStateCaptureModel {
    /// Default: calls the capturing forward with an empty set.
    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let (logits, _) = callAsFunction(
            inputs, cache: cache, captureLayerIDs: [])
        return logits
    }
}

/// Given a target model that conforms to ``HiddenStateCaptureModel``,
/// concatenate the captured hidden states at the given layer IDs along
/// the hidden dimension in the order `target_layer_ids` specifies.
///
/// Matches the Python reference `extract_context_feature(hidden_states,
/// target_layer_ids)`. HF `target_layer_ids` are already 0-based indices
/// into `target.model.layers`; pass them through unchanged.
///
/// - Returns: `(B, L, len(targetLayerIDs) * hidden)` tensor ready for
///   `DFlashDraftModel.fc`.
public func extractContextFeature(
    captured: [Int: MLXArray],
    targetLayerIDs: [Int]
) throws -> MLXArray {
    guard !targetLayerIDs.isEmpty else {
        throw SpecDecError.invalidRequest("DFlash target_layer_ids must be non-empty")
    }
    var tensors: [MLXArray] = []
    tensors.reserveCapacity(targetLayerIDs.count)
    for id in targetLayerIDs {
        guard let h = captured[id] else {
            throw SpecDecError.invalidRequest(
                "DFlash missing captured hidden state for target layer \(id)")
        }
        tensors.append(h)
    }
    return concatenated(tensors, axis: -1)
}
