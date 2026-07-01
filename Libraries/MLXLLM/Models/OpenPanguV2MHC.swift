// Copyright © 2026 Osaurus.
//
// OpenPangu-v2 MHC (Multi-stream Hyper-Connections, `mhc_num_stream`=4).
//
// Structure (from the weight graph): each layer has `attn_mhc_module` +
// `mlp_mhc_module`, and the model has a global `merge_mhc_module`. Each carries:
//   phi.weight    [mix, 4*hidden]   attn/mlp: [24,10240];  merge: [4,10240]
//   branch_alpha  [3] (attn/mlp) / branch_alpha_pre [1] (merge)
//   branch_beta   [3] (attn/mlp) / branch_beta_pre  [1] (merge)
//   norm_gamma    [4*hidden]        learned RMS-norm weight (mhc_use_gamma=true)
// `mhc_recur_norm`=20 (Sinkhorn iterations).
//
// This is the DeepSeek-V4 hyper-connection mechanism (arXiv 2409.19606). The
// forward is a DIRECT reuse of `DeepseekV4Math.hcSplitSinkhorn` +
// `DeepseekV4HyperConnection.collapse/expand` + `DeepseekV4HyperHead.reduce`,
// which were validated numerically against the jang-tools reference
// (`mlx_vlm/models/deepseek_v4/hyper_connection.py`). The mapping from
// openpangu's parameters onto that mechanism is SHAPE-FORCED:
//
//   • phi[mix,4H]  ≡ DSV4 `fn`   — identical shape `((2+hc)*hc, hc*hidden)`
//                                   = [24,10240] for hc=4,H=2560. `mix`=24
//                                   splits pre[0:4], post[4:8], comb[8:24]→(4,4).
//   • branch_alpha[3] ≡ DSV4 `scale[3]` — per-field (pre/post/comb) scalar scale.
//   • branch_beta[3]  ≡ per-field bias → expanded to DSV4 `base[24]` by
//                        broadcasting each of the 3 scalars across its field
//                        (β0×hc pre, β1×hc post, β2×hc² comb). DSV4 stores a
//                        full [24] base; openpangu factors it to 3 per-field
//                        scalars — the per-field broadcast is the reconstruction.
//   • norm_gamma[4H]  ≡ the RMSNorm weight (DSV4 passes ones; mhc_use_gamma=true).
//
// The one non-shape-forced inference is that branch_alpha is the multiplicative
// scale and branch_beta the additive bias (standard affine naming: α=scale,
// β=shift; the RMS weight is named separately as norm_gamma). This is validated
// END-TO-END by live short-context coherence on JANG_2L (port step (i)); until
// that passes, MHC is "mechanism-faithful" but not "proven". See
// OPENPANGU-V2-PORT-STATUS.md.

import Foundation
import MLX
import MLXNN

/// Per-layer MHC module (attn_mhc_module / mlp_mhc_module).
final class OpenPanguV2MHCModule: Module {
    let numStream: Int          // 4 = hcMult
    let hiddenSize: Int         // 2560
    let mixDim: Int             // 24 = (2+hc)*hc, = phi rows
    let sinkhornIters: Int      // mhc_recur_norm = 20
    let eps: Float

    /// phi is a (quantized, 2-bit in JANG_2L) linear projection `4H → mix`; as a
    /// `Linear` module the standard loader substitutes `QuantizedLinear` and loads
    /// its weight+scales+biases. (A raw MLXArray param can't receive the quant
    /// scales/biases.) Used as `mixes = phi(normed)` = `normed @ phi.weight.T`.
    @ModuleInfo(key: "phi") var phi: Linear
    @ParameterInfo(key: "branch_alpha") var branchAlpha: MLXArray // [3]
    @ParameterInfo(key: "branch_beta") var branchBeta: MLXArray   // [3]
    @ParameterInfo(key: "norm_gamma") var normGamma: MLXArray     // [4*hidden]

    init(_ config: OpenPanguV2Configuration) {
        self.numStream = config.mhcNumStream
        self.hiddenSize = config.hiddenSize
        self.mixDim = (2 + config.mhcNumStream) * config.mhcNumStream
        self.sinkhornIters = config.mhcRecurNorm
        self.eps = config.rmsNormEps
        let wide = config.mhcNumStream * config.hiddenSize
        self._phi.wrappedValue = Linear(wide, mixDim, bias: false)
        self._branchAlpha.wrappedValue = MLXArray.zeros([3])
        self._branchBeta.wrappedValue = MLXArray.zeros([3])
        self._normGamma.wrappedValue = MLXArray.ones([wide])
        super.init()
    }

    /// DSV4 `scale` = branch_alpha (per-field pre/post/comb scale), shape (3,).
    private var scaleParam: MLXArray { branchAlpha.asType(.float32) }

    /// DSV4 `base[mix]` reconstructed from the 3 per-field `branch_beta` scalars:
    /// β0 broadcast over the `hc` pre positions, β1 over `hc` post, β2 over `hc²`
    /// comb. Shape ((2+hc)*hc,) = (24,).
    private var baseParam: MLXArray {
        let mh = numStream
        let beta = branchBeta.asType(.float32)
        let onesPre = MLXArray.ones([mh]).asType(.float32)
        let onesComb = MLXArray.ones([mh * mh]).asType(.float32)
        return concatenated(
            [onesPre * beta[0], onesPre * beta[1], onesComb * beta[2]], axis: 0)
    }

    /// mixes = rms_norm(flatten(h), norm_gamma, eps) @ phi.T. fp32 throughout —
    /// the reduction spans 4H≈10240 elements and bf16 rounding saturates the
    /// rsqrt (see the DSV4 HC fp32-cast note).
    private func mixes(_ h: MLXArray) -> MLXArray {
        let (B, L) = (h.dim(0), h.dim(1))
        let dtype = h.dtype
        let flat = h.reshaped(B, L, numStream * hiddenSize)
        let normed = MLXFast.rmsNorm(
            flat.asType(.float32), weight: normGamma.asType(.float32), eps: eps)
        // phi is (quantized) — run the projection in the model dtype, then lift
        // the mix back to fp32 for the sinkhorn (the 2-bit phi dominates error,
        // so the fp16 projection is negligible next to it).
        return phi(normed.asType(dtype)).asType(.float32)  // (B,L,mix)
    }

    /// Collapse the 4 residual streams `(B,L,4,H)` → single block input `(B,L,H)`,
    /// returning the (post, comb) needed to expand the block output back.
    /// Mirrors `DeepseekV4HyperConnection.collapse` exactly.
    func collapse(_ h: MLXArray) -> (x: MLXArray, post: MLXArray, comb: MLXArray) {
        let dtype = h.dtype
        let (B, L) = (h.dim(0), h.dim(1))
        let (pre, post, comb) = DeepseekV4Math.hcSplitSinkhorn(
            mixes: mixes(h), scale: scaleParam, base: baseParam,
            hcMult: numStream, iters: sinkhornIters, eps: eps)
        // y = sum(pre[..., None] * h, axis=2)
        let y = (pre.asType(dtype).expandedDimensions(axis: -1) * h).sum(axis: -2)
        return (x: y, post: post, comb: comb)
    }

    /// Expand block output `(B,L,H)` + prior residual `(B,L,4,H)` back to 4
    /// streams. Mirrors `DeepseekV4HyperConnection.expand`:
    ///   y = post[..., None] * blockOut[..., None, :] + matmul(comb, residual)
    func expand(blockOut: MLXArray, residual: MLXArray, post: MLXArray, comb: MLXArray) -> MLXArray {
        let dtype = blockOut.dtype
        let combResid = comb.asType(dtype).matmul(residual)
        return post.asType(dtype).expandedDimensions(axis: -1)
            * blockOut.expandedDimensions(axis: -2) + combResid
    }
}

/// Global merge module (`model.merge_mhc_module`): collapse the 4 streams → 1
/// `(B,L,H)` before the final `model.norm`. phi is `[4, 4*hidden]`; branch scalars
/// are `[1]` (`branch_alpha_pre`/`branch_beta_pre`). Mirrors
/// `DeepseekV4HyperHead.reduce` (sigmoid gate, NO sum-to-1 normalization).
final class OpenPanguV2MergeMHC: Module {
    let numStream: Int
    let hiddenSize: Int
    let eps: Float

    /// merge phi is fp16 in JANG_2L (not in the quant dict) — a plain `Linear`
    /// (`4H → mhcNumStream`) loads its `phi.weight`; the loader leaves it unquantized.
    @ModuleInfo(key: "phi") var phi: Linear
    @ParameterInfo(key: "branch_alpha_pre") var branchAlphaPre: MLXArray  // [1]
    @ParameterInfo(key: "branch_beta_pre") var branchBetaPre: MLXArray    // [1]
    @ParameterInfo(key: "norm_gamma") var normGamma: MLXArray      // [4*hidden]

    init(_ config: OpenPanguV2Configuration) {
        self.numStream = config.mhcNumStream
        self.hiddenSize = config.hiddenSize
        self.eps = config.rmsNormEps
        let wide = config.mhcNumStream * config.hiddenSize
        self._phi.wrappedValue = Linear(wide, config.mhcNumStream, bias: false)
        self._branchAlphaPre.wrappedValue = MLXArray.zeros([1])
        self._branchBetaPre.wrappedValue = MLXArray.zeros([1])
        self._normGamma.wrappedValue = MLXArray.ones([wide])
        super.init()
    }

    /// Reduce `(B,L,4,H)` → `(B,L,H)`.
    ///   mixes = rms_norm(flatten(h), norm_gamma, eps) @ phi.T   # (B,L,4)
    ///   pre   = sigmoid(mixes * alpha_pre + beta_pre) + eps
    ///   y     = sum(pre[..., None] * h, axis=2)
    func callAsFunction(_ h: MLXArray) -> MLXArray {
        let dtype = h.dtype
        let (B, L) = (h.dim(0), h.dim(1))
        let flat = h.reshaped(B, L, numStream * hiddenSize)
        let normed = MLXFast.rmsNorm(
            flat.asType(.float32), weight: normGamma.asType(.float32), eps: eps)
        let mixes = phi(normed.asType(dtype)).asType(.float32)          // (B,L,4)
        let pre = sigmoid(
            mixes * branchAlphaPre.asType(.float32) + branchBetaPre.asType(.float32))
            + MLXArray(eps)
        return (pre.asType(dtype).expandedDimensions(axis: -1) * h).sum(axis: -2)
            .asType(dtype)
    }
}
