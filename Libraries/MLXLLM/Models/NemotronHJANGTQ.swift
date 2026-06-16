// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// JANGTQ (TurboQuant codebook) variant of NemotronH's routed-expert
// switch_mlp. Replaces the affine `SwitchLinear` fc1 / fc2 with
// `TurboQuantSwitchLinear` so the codebook + Hadamard-rotation Metal
// kernels run instead of `gather_qmm`. Surrounding plumbing
// (NemotronHMoE, NemotronHBlock, NemotronHBackbone, NemotronHModel)
// is reused unchanged via the `NemotronHJANGTQContext` propagation.
//
// Target bundles:
//   - Nemotron-3-Nano-Omni-30B-A3B-JANGTQ4  (omni multimodal)
//   - Nemotron-3-Nano-Omni-30B-A3B-JANGTQ2  (omni multimodal)
//   - Cascade-2 / Super JANGTQ_2L / JANGTQ_4L variants (text-only)
//
// The Nemotron MoE uses ReLU² (not SwiGLU) with only fc1 + fc2 (no
// gate_proj), so we can NOT reuse `TurboQuantSwitchGLU` (gate+up+down).
// Instead we wire two `TurboQuantSwitchLinear` instances and apply
// relu² between them — each linear handles its own Hadamard rotation
// and codebook gather, and the kernel internally chains the
// per-(token, expert) dispatches.
//
// Wire format on disk (post-sanitize via `NemotronHModel.sanitize`):
//   backbone.layers.{l}.mixer.switch_mlp.fc1.tq_packed  (n_exp, hidden_inter, packed_in)
//   backbone.layers.{l}.mixer.switch_mlp.fc1.tq_norms   (n_exp, hidden_inter)
//   backbone.layers.{l}.mixer.switch_mlp.fc2.tq_packed  (n_exp, hidden, packed_inter)
//   backbone.layers.{l}.mixer.switch_mlp.fc2.tq_norms   (n_exp, hidden)
// (`.tq_bits` metadata is stripped at sanitize time.)
//
// Sidecar:
//   jangtq_runtime.safetensors → JANGTQRuntimeCache (signs / codebook)
//   loaded by `loadWeights` before model.update() so the kernels have
//   everything on first forward.

import MLX
import MLXLMCommon
import MLXNN

// MARK: - JANGTQ-flavored switch MLP

/// Drop-in replacement for ``NemotronHSwitchMLP`` that uses
/// ``TurboQuantSwitchLinear`` for fc1 + fc2. ReLU² activation matches
/// the affine path (mirrors `jang_tools/nemotron_omni/parakeet.py`'s
/// nemotron MoE wiring).
internal final class NemotronHJANGTQSwitchMLP: Module, NemotronHSwitchMLPLayer {
    @ModuleInfo(key: "fc1") var fc1: TurboQuantSwitchLinear
    @ModuleInfo(key: "fc2") var fc2: TurboQuantSwitchLinear

    let inputDims: Int
    let hiddenDims: Int
    let numExperts: Int

    init(
        inputDims: Int, hiddenDims: Int, numExperts: Int,
        bits: Int = 2, mxtqSeed: Int = 42
    ) {
        self.inputDims = inputDims
        self.hiddenDims = hiddenDims
        self.numExperts = numExperts
        self._fc1.wrappedValue = TurboQuantSwitchLinear(
            inFeatures: inputDims, outFeatures: hiddenDims,
            numExperts: numExperts, bits: bits, seed: mxtqSeed)
        self._fc2.wrappedValue = TurboQuantSwitchLinear(
            inFeatures: hiddenDims, outFeatures: inputDims,
            numExperts: numExperts, bits: bits, seed: mxtqSeed)
        super.init()
    }

    /// Forward through two TurboQuant linears with ReLU² in between.
    ///
    /// Shape contract (must match ``NemotronHSwitchMLP``):
    ///   - `x`       : `(B, T, hidden)` — pre-MoE token activations
    ///   - `indices` : `(B, T, K)` — expert ids per token
    ///   - return    : `(B, T, K, hidden)` — per-(token, expert) outputs.
    ///                 The caller (`NemotronHMoE.callAsFunction`) does
    ///                 `(y * scores[.ellipsis, .newAxis]).sum(axis: -2)`
    ///                 to reduce over the K dim.
    ///
    /// We bypass `TurboQuantSwitchLinear.callAsFunction(_:_:)` and call
    /// the lower-level `JANGTQKernels` directly because the wrapper's
    /// K-broadcast contract is broken for the affine-shape input we get
    /// here: it passes `nRows = batch * K` to the per-row gather kernel
    /// but only supplies `batch` rows of `xRot`. Calling the kernels
    /// directly lets us EXPAND the input to per-(token, expert) layout
    /// up-front, which is what the kernel actually expects.
    func callAsFunction(_ x: MLXArray, _ indices: MLXArray) -> MLXArray {
        let signsIn = JANGTQRuntimeCache.shared.requiredSigns(
            inFeatures: inputDims, seed: fc1.mxtqSeed)
        let signsInter = JANGTQRuntimeCache.shared.requiredSigns(
            inFeatures: hiddenDims, seed: fc2.mxtqSeed)
        let cbIn = JANGTQRuntimeCache.shared.requiredCodebook(
            inFeatures: inputDims, bits: fc1.bits)
        let cbInter = JANGTQRuntimeCache.shared.requiredCodebook(
            inFeatures: hiddenDims, bits: fc2.bits)

        // x: (B, T, hidden). Rotate once per token for fc1; the gather
        // kernel reuses each rotated row across the K selected experts.
        let totalTokens = x.size / inputDims          // = B * T
        let kSlots = indices.dim(-1)                  // = K
        let xPerToken = x.reshaped([totalTokens, inputDims]) // (B*T, hidden)

        let idxFlat = indices.reshaped([-1]).asType(.uint32)

        // === fc1: (B*T*K, hidden) → (B*T*K, hidden_inter) ===
        // Hadamard rotate, then gather TQ matmul per row.
        let xRot1 = JANGTQKernels.hadamardRotate(
            xPerToken, signs: signsIn, dim: inputDims)
        var h = JANGTQKernels.gatherTQTopK(
            xRot: xRot1, packed: fc1.packed, norms: fc1.norms,
            codebook: cbIn, rhsIndices: idxFlat,
            batchTokens: totalTokens, K: kSlots,
            inFeatures: inputDims, outFeatures: hiddenDims, bits: fc1.bits)

        // ReLU² activation — Nemotron MoE squared ReLU, NOT SwiGLU.
        let relu = MLX.maximum(h, MLXArray(0, dtype: h.dtype))
        h = relu * relu                                 // (B*T*K, hidden_inter)

        // === fc2: (B*T*K, hidden_inter) → (B*T*K, hidden) ===
        let xRot2 = JANGTQKernels.hadamardRotate(
            h, signs: signsInter, dim: hiddenDims)
        let out = JANGTQKernels.gatherTQ(
            xRot: xRot2, packed: fc2.packed, norms: fc2.norms,
            codebook: cbInter, rhsIndices: idxFlat,
            nRows: totalTokens * kSlots,
            inFeatures: hiddenDims, outFeatures: inputDims, bits: fc2.bits)

        // Reshape to (B, T, K, hidden) — matches affine path return.
        var outShape = Array(indices.shape)             // (B, T, K)
        outShape.append(inputDims)                      // (B, T, K, hidden)
        return out.reshaped(outShape).asType(x.dtype)
    }

}

extension StreamingTurboQuantSwitchReLUSquaredMLP: NemotronHSwitchMLPLayer,
    NemotronHReducedSwitchMLPLayer {}

// MARK: - Public NemotronHJANGTQ helpers
//
// Construction is via `NemotronHModel(args, jangtq: NemotronHJANGTQContext)`.
// The factory (LLMModelFactory / VLMModelFactory) resolves the JANGTQ bits
// from `jang_config.json` (`bit_widths_used` min, or `mxtq_bits` override)
// and instantiates with the right context. Sanitize stacking lives in
// `NemotronHModel.sanitize` — gated on the presence of `experts.{e}.{up,
// down}_proj.tq_packed` keys, so it's a no-op for affine bundles.
//
// Bundle config conventions for omni JANGTQ (per
// jang/research/NEMOTRON-OMNI-RUNTIME-2026-04-28.md §13):
//   jang_config.json
//     weight_format: "mxtq"
//     quantization.bit_widths_used: [4, 8] (JANGTQ4) or [2, 4, 8] (JANGTQ2)
//     quantization.profile: "JANGTQ4" / "JANGTQ2"
//   jangtq_runtime.safetensors (signs.{N}.{seed} + codebook.{N}.{bits})
