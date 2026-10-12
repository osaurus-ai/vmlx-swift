// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXLLM
import MLXLMCommon

/// GLM-5.3-Flash as a DFlash 2 target (drafter: `dflash2/` in the JANGHT bundles, 5 sliding-window
/// qwen3 layers, block 8, taps [5, 14, 24, 33, 42]). Port of vMLX Python `glm_dflash2.GLMDFlashTarget`:
///
/// - taps: the completed outputs of the drafter's target layers, each the unweighted mean over the
///   4 mHC streams (``Glm5NextLanguageModel/forwardCapturing``);
/// - embedding and LM head shared with the drafter;
/// - rollback: KDA layers keep their verify inputs and replay only the accepted rows on a rejection
///   (``KDAAttention/commitVerifyStash``); MLA/DSA layers trim, the indexed cache dropping the
///   matching indexer rows. Under input capture no per-prefix state is recorded.
extension Glm5Next: HiddenStateCaptureModel, TokenEmbedderModel, DFlash2VerifyRollbackModel {

    public func callAsFunction(
        _ inputs: MLXArray, cache: [KVCache]?, captureLayerIDs: Set<Int>
    ) -> (logits: MLXArray, capturedHiddenStates: [Int: MLXArray]) {
        do {
            let (hidden, captured) = try languageModel.forwardCapturing(
                inputs, caches: cache, captureLayerIDs: captureLayerIDs)
            return (projectToLogits(hidden), captured)
        } catch {
            // Same non-throwing contract as the token-only forward above.
            FileHandle.standardError.write(Data("[glm5_next] capturing forward failed: \(error)\n".utf8))
            return (MLXArray.zeros([inputs.dim(0), inputs.dim(1), vocabularySize]), [:])
        }
    }

    /// KDA rollback uses the verify-input stash (`input_capture`), which the DFlash 2 iterator selects
    /// for every target conforming to ``DFlash2VerifyRollbackModel``. Per-prefix recording is not
    /// implemented: with `VMLX_DFLASH2_INPUT_CAPTURE=0` a rejection finds no recorded state and the
    /// iterator ends the turn instead of continuing from a wrong state.
    public var supportsCapturingPrefixCommitRecording: Bool { true }

    public func embed(_ tokenIds: MLXArray) -> MLXArray {
        languageModel.embedTokens(tokenIds)
    }

    public func projectToLogits(_ hidden: MLXArray) -> MLXArray {
        if let lmHead { return lmHead(hidden) }
        return languageModel.embedTokens.asLinear(hidden)
    }

    public func commitVerifiedBlock(cache: [KVCache], acceptedInputs: Int) -> Bool {
        for index in 0 ..< languageModel.numDecoderLayers
        where languageModel.layers[index].kind == .linearAttention {
            guard index < cache.count, let mamba = cache[index] as? MambaCache,
                let kda = languageModel.layers[index].linearAttention,
                kda.commitVerifyStash(cache: mamba, acceptedInputs: acceptedInputs)
            else { return false }
        }
        return true
    }
}

/// No lane matmul on GLM-5.3. With it installed (DFlash 2 installs it on the target for batched verify), GLM-5.3-Flash
/// JANGHT2.4 answers coherently below index_topk and degrades past it: 1.5k-token prompt fine, 3k misreads the
/// request, 6k / 12k emit garbage from the first token (RunBench default arm and the osaurus app, 2026-10-11);
/// `VMLX_LANE_QMM=0` restores coherent output at 6k. vMLX Python found and fixed the same defect (lane_qmm
/// installed on the glm5_next target corrupted its served DFlash2 lane, `54156d41a`). The target keeps MLX's
/// kernels; the drafter keeps its own lane install.
/// GLM-5.3's verify costs ~0.25 AR steps per added row (33 / 47 / 61 / 90 ms at 1 / 2 / 4 / 8 rows), so on prose
/// (per-position acceptance ~0.6) a 4-row verify pays and a 5-row one barely does. With (3, 4, 6) available the chooser
/// settles on 4 for prose. Alternating A/B, 4 rounds, per MHz, against (5, 8, 16): prose +28 / +13 / +15 / +7 %;
/// code, easy code and easy prose within noise (mean +2 / -1 / -0.5 %).
extension Glm5Next: DFlash2WidthLadderPolicy {
    public var dflash2WidthLadder: [Int] { [3, 4, 6, 8, 16] }
}

extension Glm5Next: LaneQMMTargetPolicy {
    public var laneQMMEnabled: Bool { false }
    public var laneQMMTiled: Bool { false }
    public var laneQMMUntiledMinRows: Int { Int.max }
}
