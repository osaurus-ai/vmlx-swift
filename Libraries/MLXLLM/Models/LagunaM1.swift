// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Laguna-M.1 — affine-quantized text decoder (model_type=laguna, the Mistral-
// lineage M.1 line; distinct from the poolside XS.2 handled by `Laguna.swift`).
//
// Probed from the JANG_2L/JANG_1L bundles (header-only, no model load):
//   - 70 layers, hidden 4096, heads 64, kv 8, head_dim 128, vocab 100352.
//   - ALL full attention (sliding_window=0) → plain `KVCacheSimple` ×70.
//   - SINGLE YaRN RoPE (full rotary, partial_rotary_factor=1.0): theta 5e5,
//     factor 64, original_max_position_embeddings 4096, attention_factor→mscale.
//   - SEPARATE q/k/v/o_proj (XS.2 fuses qkv); per-head q_norm/k_norm RMSNorm(128).
//   - Per-ELEMENT attention gate: out * softplus(g_proj(x)), g_proj → heads*head_dim.
//   - MoE: 256 experts top-16 + 1 shared, sigmoid + e_score_correction_bias top-k
//     routing (DeepSeek-V3 recipe), routed contribution × moe_routed_scaling_factor
//     (=1.0 on M.1), shared expert UNSCALED. Layers 0-2 dense, 3-69 sparse.
//   - Per-module AFFINE quant (mode=affine, gs 64; embed 6b / lm_head 8b / attn 8b /
//     mlp 6b). Load.swift quantizes only modules whose checkpoint carries `.scales`,
//     so q_norm/k_norm/gate/e_score_correction_bias stay fp16 automatically.
//
// Reuses `LagunaConfiguration` (shared with Laguna.swift). The factory routes the
// affine M.1 bundle here via `gateMode == .perElement`.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Attention (separate q/k/v + per-element softplus gate)

private final class LagunaM1Attention: Module {
    let nHeads: Int
    let nKVHeads: Int
    let headDim: Int
    let ropeDim: Int
    let scale: Float
    let gateMode: LagunaGateMode

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm
    @ModuleInfo(key: "g_proj") var gProj: Linear?
    let rope: RoPELayer

    init(_ cfg: LagunaConfiguration) {
        self.nHeads = cfg.numAttentionHeads
        self.nKVHeads = cfg.numKeyValueHeads
        self.headDim = cfg.headDim
        self.scale = pow(Float(headDim), -0.5)
        self.gateMode = cfg.gateMode

        let h = cfg.hiddenSize
        self._qProj.wrappedValue = Linear(h, nHeads * headDim, bias: cfg.attentionBias)
        self._kProj.wrappedValue = Linear(h, nKVHeads * headDim, bias: cfg.attentionBias)
        self._vProj.wrappedValue = Linear(h, nKVHeads * headDim, bias: cfg.attentionBias)
        self._oProj.wrappedValue = Linear(nHeads * headDim, h, bias: cfg.attentionBias)
        self._qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: cfg.rmsNormEps)
        self._kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: cfg.rmsNormEps)

        // Per-element gate: g_proj → heads*head_dim, element-wise softplus gate.
        if cfg.gateMode == .perElement {
            self._gProj.wrappedValue = Linear(h, nHeads * headDim, bias: false)
        } else if cfg.gateMode == .perHead {
            self._gProj.wrappedValue = Linear(h, nHeads, bias: false)
        }

        // M.1 rotary: config ships NO `rotary_dim` and `partial_rotary_factor=1.0`,
        // so this is FULL rotary over the entire head_dim (128) — matching
        // modeling_laguna.py (partial=1.0 → rotary_dim=cos.shape[-1]=128, q_pass
        // empty). `cfg.rotaryDim` decodes to 0 when absent → falls back to
        // `headDim` here. The `rotaryDim>0` branch only engages if a future
        // bundle explicitly declares a partial `rotary_dim`. (Earlier crash was
        // YarnRoPE's mscale path, fixed below by pinning mscale_all_dim=mscale.)
        self.ropeDim = (cfg.rotaryDim > 0 && cfg.rotaryDim <= headDim) ? cfg.rotaryDim : headDim
        let (theta, _) = cfg.ropeFor(layerType: "full_attention")
        var scalingCfg: [String: StringOrNumber]? = nil
        if let entry = cfg.ropeParameters["full_attention"] {
            let ropeType = entry["rope_type"].flatMap { v -> String? in
                if case .string(let s) = v { return s }
                return nil
            } ?? "default"
            if ropeType != "default" {
                var dict = entry
                if let af = dict["attention_factor"], dict["mscale"] == nil {
                    dict["mscale"] = af
                    dict["attention_factor"] = nil
                }
                // Do NOT pin mscale_all_dim. The JANG reference runtime (mlx_lm
                // YarnRoPE, proven coherent on this exact bundle) leaves
                // mscale_all_dim at its default 0, giving
                //   _mscale = yarnGetMscale(factor,1)/yarnGetMscale(factor,0) ≈ 1.416
                // for factor=64 — a constant q/k length-scale the model was trained
                // with (≈2x on attention logits). An earlier crash-dodge pinned
                // mscale_all_dim = mscale → _mscale = 1.0, which REMOVED that scaling
                // → ~2x-too-soft attention → coherent-start-then-word-salad. The
                // YarnRoPE in-place crash is now fixed functionally in RoPEUtils, so
                // the real _mscale applies. (mscale stays = attention_factor = 1.0.)
                scalingCfg = dict.compactMapValues { $0 }
            }
        }
        self.rope = initializeRope(
            dims: ropeDim,
            base: theta,
            traditional: false,
            scalingConfig: scalingCfg,
            maxPositionEmbeddings: cfg.maxPositionEmbeddings)
    }

    /// Partial rotary: rotate the first `ropeDim` channels of head_dim, pass the
    /// tail through unchanged. `ropeDim == headDim` → full rotary.
    private func applyPartialRope(_ t: MLXArray, offset: Int) -> MLXArray {
        if ropeDim >= headDim { return rope(t, offset: offset) }
        let rot = t[.ellipsis, ..<ropeDim]
        let pass = t[.ellipsis, ropeDim...]
        return MLX.concatenated([rope(rot, offset: offset), pass], axis: -1)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, T) = (x.dim(0), x.dim(1))
        var q = qProj(x).reshaped(B, T, nHeads, headDim)
        var k = kProj(x).reshaped(B, T, nKVHeads, headDim)
        let v = vProj(x).reshaped(B, T, nKVHeads, headDim).transposed(0, 2, 1, 3)
        // Per-head q/k norm AFTER projection, BEFORE rope (reference order).
        q = qNorm(q).transposed(0, 2, 1, 3)
        k = kNorm(k).transposed(0, 2, 1, 3)
        let off = cache?.offset ?? 0
        q = applyPartialRope(q, offset: off)
        k = applyPartialRope(k, offset: off)

        var out = attentionWithCacheUpdate(
            queries: q, keys: k, values: v, cache: cache, scale: scale, mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, T, nHeads * headDim)

        if let g = gProj {
            // softplus (unbounded) — NOT sigmoid: HF reference amplifies rather
            // than damps; sigmoid drives residual-stream blow-up over the depth.
            let gate = softplus(g(x).asType(.float32)).asType(out.dtype)
            if gateMode == .perElement {
                out = out * gate
            } else {
                let gated = out.reshaped(B, T, nHeads, headDim) * gate.expandedDimensions(axis: -1)
                out = gated.reshaped(B, T, nHeads * headDim)
            }
        }
        return oProj(out)
    }
}

// MARK: - MLP variants

private final class LagunaM1DenseMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    init(dimensions: Int, hiddenDimensions: Int) {
        self._gate.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        self._up.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        self._down.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

private final class LagunaM1MoE: Module, UnaryLayer {
    let topK: Int
    let routedScale: Float

    @ModuleInfo(key: "gate") var gate: Linear
    @ParameterInfo(key: "e_score_correction_bias") var eScoreCorrectionBias: MLXArray
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "shared_expert") var sharedExpert: LagunaM1DenseMLP

    init(_ cfg: LagunaConfiguration) {
        self.topK = cfg.numExpertsPerTok
        self.routedScale = cfg.moeRoutedScalingFactor
        self._gate.wrappedValue = Linear(cfg.hiddenSize, cfg.numExperts, bias: false)
        self._eScoreCorrectionBias.wrappedValue = MLXArray.zeros([cfg.numExperts])
        self._switchMLP.wrappedValue = SwitchGLU(
            inputDims: cfg.hiddenSize,
            hiddenDims: cfg.moeIntermediateSize,
            numExperts: cfg.numExperts)
        self._sharedExpert.wrappedValue = LagunaM1DenseMLP(
            dimensions: cfg.hiddenSize, hiddenDimensions: cfg.sharedExpertIntermediateSize)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Sigmoid + bias top-k (DeepSeek-V3): the bias picks WHICH experts, but
        // the gating weight is the UN-biased sigmoid score, renormalized over k.
        let logits = gate(x).asType(.float32)
        let scores = sigmoid(logits)
        let part = argPartition(-(scores + eScoreCorrectionBias), kth: topK - 1, axis: -1)
        let inds = part[.ellipsis, ..<topK]
        var weights = MLX.takeAlong(scores, inds, axis: -1)
        weights = weights / (weights.sum(axis: -1, keepDims: true) + MLXArray(1e-20, dtype: weights.dtype))

        let y = (switchMLP(x, inds) * weights[.ellipsis, .newAxis].asType(x.dtype)).sum(axis: -2)
        // Routed contribution scaled; shared expert UNSCALED (HF order).
        return (y * routedScale + sharedExpert(x)).asType(x.dtype)
    }
}

// MARK: - Decoder layer

private final class LagunaM1Layer: Module {
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "self_attn") var attention: LagunaM1Attention
    fileprivate let mlp: UnaryLayer

    init(_ cfg: LagunaConfiguration, layerIndex: Int) {
        self._inputLayerNorm.wrappedValue = RMSNorm(dimensions: cfg.hiddenSize, eps: cfg.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: cfg.hiddenSize, eps: cfg.rmsNormEps)
        self._attention.wrappedValue = LagunaM1Attention(cfg)
        if cfg.mlpLayerTypes[layerIndex] == "dense" {
            self.mlp = LagunaM1DenseMLP(
                dimensions: cfg.hiddenSize, hiddenDimensions: cfg.intermediateSize)
        } else {
            self.mlp = LagunaM1MoE(cfg)
        }
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let normed = inputLayerNorm(x)
        let attnOut = attention(normed, mask: mask, cache: cache)
        let h = x + attnOut
        let mlpOut = mlp(postAttentionLayerNorm(h))
        return h + mlpOut
    }
}

// MARK: - Model

public final class LagunaM1Model: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    fileprivate let layers: [LagunaM1Layer]
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    let cfg: LagunaConfiguration

    public init(_ cfg: LagunaConfiguration) {
        self.cfg = cfg
        self.vocabularySize = cfg.vocabularySize
        self.kvHeads = (0 ..< cfg.numHiddenLayers).map { _ in cfg.numKeyValueHeads }
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: cfg.vocabularySize, dimensions: cfg.hiddenSize)
        self.layers = (0 ..< cfg.numHiddenLayers).map { LagunaM1Layer(cfg, layerIndex: $0) }
        self._norm.wrappedValue = RMSNorm(dimensions: cfg.hiddenSize, eps: cfg.rmsNormEps)
        if !cfg.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(cfg.hiddenSize, cfg.vocabularySize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = embedTokens(inputs)
        let mask = createAttentionMask(h: h, cache: cache?.first)
        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }
        h = norm(h)
        if let lmHead {
            return lmHead(h)
        }
        return embedTokens.asLinear(h)
    }

    /// All-full-attention → plain `KVCacheSimple` ×N (no sliding/sparse lanes).
    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        (0 ..< cfg.numHiddenLayers).map { _ in KVCacheSimple() }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        out.reserveCapacity(weights.count)
        for (key, value) in weights {
            var k = key
            if k.hasPrefix("model.") {
                k = String(k.dropFirst("model.".count))
            }
            // Router bias ships nested under `experts`; the module binds it flat.
            k = k.replacingOccurrences(
                of: ".mlp.experts.e_score_correction_bias",
                with: ".mlp.e_score_correction_bias")
            if k.contains("self_attn.rotary_emb.inv_freq") { continue }
            if k.hasSuffix(".tq_bits") { continue }
            if cfg.tieWordEmbeddings && k == "lm_head.weight" { continue }
            out[k] = value
        }
        return out
    }
}

extension LagunaM1Model: LoRAModel {
    public var loraLayers: [Module] { layers }
}
