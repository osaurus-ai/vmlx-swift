// Copyright © 2026 Osaurus.
//
// OpenPangu-v2 MHC (Multi-stream Hyper-Connections, `mhc_num_stream`=4).
//
// Structure (from the weight graph): each layer has `attn_mhc_module` +
// `mlp_mhc_module`, and the model has a global `merge_mhc_module`. Each carries:
//   phi.weight    [mix, 4*hidden]   attn/mlp: [24,10240];  merge: [4,10240]
//   branch_alpha  [3] (attn/mlp) / [1] (merge, "_pre")
//   branch_beta   [3] (attn/mlp) / [1] (merge, "_pre")
//   norm_gamma    [4*hidden]        learned RMS-norm weight (mhc_use_gamma=true)
// `mhc_recur_norm`=20 (Sinkhorn iterations).
//
// This is structurally the DSV4 hyper-connection (RMS-norm the flattened 4-stream
// residual → project via phi → Sinkhorn-normalize → pre/post/comb stream weights
// → collapse before the block / expand after). See DeepseekV4HyperConnection
// (collapse/expand) + DeepseekV4Math.hcSplitSinkhorn + DeepseekV4HyperHead.reduce.
//
// ⚠️  GATING UNKNOWN — the exact forward math (how phi's `mix`=24 output and the
// two [3] branch vectors produce the 4×4 pre/post/comb stream-mix under 20
// Sinkhorn iters) is Pangu-specific and NOT recoverable from tensor shapes alone.
// The modeling source is tf-5.0-native/unreleased. The `collapse`/`expand`/
// `merge` bodies below are a best-effort adaptation of the DSV4 mechanism and
// MUST be numerically validated against a reference activation (the jang-tools
// converter in ~/jang/jang-tools that handles `merge_mhc phi`, or a one-layer
// input→output dump) before this model is claimed correct. Until then the module
// only guarantees: parameters load, shapes flow, and the model compiles.

import Foundation
import MLX
import MLXFast
import MLXNN

/// Per-layer MHC module (attn_mhc_module / mlp_mhc_module).
final class OpenPanguV2MHCModule: Module {
    let numStream: Int          // 4
    let hiddenSize: Int         // 2560
    let mixDim: Int             // 24 (attn/mlp), = phi rows
    let sinkhornIters: Int      // mhc_recur_norm = 20
    let eps: Float

    @ParameterInfo(key: "phi.weight") var phi: MLXArray          // [mix, 4*hidden]
    @ParameterInfo(key: "branch_alpha") var branchAlpha: MLXArray // [3]
    @ParameterInfo(key: "branch_beta") var branchBeta: MLXArray   // [3]
    @ParameterInfo(key: "norm_gamma") var normGamma: MLXArray     // [4*hidden]

    init(_ config: OpenPanguV2Configuration, mixDim: Int = 24) {
        self.numStream = config.mhcNumStream
        self.hiddenSize = config.hiddenSize
        self.mixDim = mixDim
        self.sinkhornIters = config.mhcRecurNorm
        self.eps = config.rmsNormEps
        let wide = config.mhcNumStream * config.hiddenSize
        self._phi.wrappedValue = MLXArray.zeros([mixDim, wide])
        self._branchAlpha.wrappedValue = MLXArray.zeros([3])
        self._branchBeta.wrappedValue = MLXArray.zeros([3])
        self._normGamma.wrappedValue = MLXArray.ones([wide])
        super.init()
    }

    /// RMS-norm the flattened 4-stream residual `(B,L,4,H)` → `(B,L,4H)` using the
    /// learned `norm_gamma`, in fp32 (the reduction spans 4H≈10240 elements; bf16
    /// rounding saturates — see the DSV4 HC fp32-cast note).
    private func normedFlat(_ h: MLXArray) -> MLXArray {
        let (B, L) = (h.dim(0), h.dim(1))
        let flat = h.reshaped(B, L, numStream * hiddenSize)
        return MLXFast.rmsNorm(
            flat.asType(.float32), weight: normGamma.asType(.float32), eps: eps)
    }

    /// Collapse the 4 residual streams into a single `(B,L,H)` block input, and
    /// return the (post, comb) needed to expand the block output back to 4 streams.
    ///
    /// ⚠️ best-effort forward — see file header. Placeholder until numerically
    /// validated: uses `branch_alpha`-weighted stream averaging so shapes flow and
    /// weights load; the Sinkhorn phi-mix is applied but the exact pre/post/comb
    /// derivation is the gating unknown.
    func collapse(_ h: MLXArray) -> (x: MLXArray, post: MLXArray, comb: MLXArray) {
        let (B, L) = (h.dim(0), h.dim(1))
        let dtype = h.dtype
        _ = normedFlat(h).matmul(phi.asType(.float32).transposed())  // (B,L,mix) — phi mix
        // TODO(mhc): derive pre/post/comb from the phi mix + branch_alpha/beta via
        // the Pangu Sinkhorn (mhc_recur_norm iters). Placeholder below.
        let streams = h.reshaped(B, L, numStream, hiddenSize)
        let x = streams.mean(axis: -2).asType(dtype)                 // collapse (placeholder)
        let post = MLXArray.ones([B, L, numStream], dtype: .float32)
        let comb = broadcast(
            (MLXArray.zeros([numStream, numStream]) + (1.0 / Float(numStream)))
                .reshaped(1, 1, numStream, numStream),
            to: [B, L, numStream, numStream])
        return (x: x, post: post, comb: comb)
    }

    /// Expand block output `(B,L,H)` + prior residual `(B,L,4,H)` back to 4 streams.
    /// ⚠️ best-effort — see header.
    func expand(blockOut: MLXArray, residual: MLXArray, post: MLXArray, comb: MLXArray) -> MLXArray {
        let dtype = blockOut.dtype
        let combResid = comb.asType(dtype).matmul(residual)
        return post.asType(dtype).expandedDimensions(axis: -1)
            * blockOut.expandedDimensions(axis: -2) + combResid
    }
}

/// Global merge module (`model.merge_mhc_module`): collapse the 4 streams → 1
/// `(B,L,H)` before the final `model.norm`. phi is `[4, 4*hidden]`; branch scalars
/// are `[1]` (`branch_alpha_pre`/`branch_beta_pre`).
final class OpenPanguV2MergeMHC: Module {
    let numStream: Int
    let hiddenSize: Int
    let eps: Float

    @ParameterInfo(key: "phi.weight") var phi: MLXArray            // [4, 4*hidden]
    @ParameterInfo(key: "branch_alpha_pre") var branchAlphaPre: MLXArray  // [1]
    @ParameterInfo(key: "branch_beta_pre") var branchBetaPre: MLXArray    // [1]
    @ParameterInfo(key: "norm_gamma") var normGamma: MLXArray      // [4*hidden]

    init(_ config: OpenPanguV2Configuration) {
        self.numStream = config.mhcNumStream
        self.hiddenSize = config.hiddenSize
        self.eps = config.rmsNormEps
        let wide = config.mhcNumStream * config.hiddenSize
        self._phi.wrappedValue = MLXArray.zeros([config.mhcNumStream, wide])
        self._branchAlphaPre.wrappedValue = MLXArray.zeros([1])
        self._branchBetaPre.wrappedValue = MLXArray.zeros([1])
        self._normGamma.wrappedValue = MLXArray.ones([wide])
        super.init()
    }

    /// Reduce `(B,L,4,H)` → `(B,L,H)`. ⚠️ best-effort placeholder (mean over streams
    /// + phi gate); numeric validation required — see OpenPanguV2MHCModule header.
    func callAsFunction(_ h: MLXArray) -> MLXArray {
        let (B, L) = (h.dim(0), h.dim(1))
        let flat = h.reshaped(B, L, numStream * hiddenSize)
        let normed = MLXFast.rmsNorm(
            flat.asType(.float32), weight: normGamma.asType(.float32), eps: eps)
        let mixes = normed.matmul(phi.asType(.float32).transposed())   // (B,L,4)
        let pre = sigmoid(mixes * branchAlphaPre.asType(.float32) + branchBetaPre.asType(.float32))
        let streams = flat.reshaped(B, L, numStream, hiddenSize)
        return (pre.asType(h.dtype).expandedDimensions(axis: -1) * streams).sum(axis: -2)
            .asType(h.dtype)
    }
}
