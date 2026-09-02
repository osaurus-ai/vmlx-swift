// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// JANGTQ (TurboQuant codebook) variant of DeepseekV4.
//
// Swaps the per-expert routed `SwitchGLU` for `TurboQuantSwitchGLU` so
// the codebook Metal kernels run instead of affine `gather_qmm`.
// Everything else — MLA attention (with sinks + inverse RoPE + grouped
// O), mHC HyperConnection, MoE gate (sqrtsoftplus + hash routing),
// HyperHead reduce, norms, embed, lm_head, shared experts — is
// identical to `DeepseekV4Model` and reuses the same forward logic.
//
// Target bundles (from research/DSV4-RUNTIME-ARCHITECTURE.md §5):
//   - DSV4-Flash JANGTQ2 (74 GB, smallest) — routed 2-bit MXTQ, non-
//     routed 8-bit affine g=32
//   - DSV4-Flash JANGTQ4 (173 GB) — routed 4-bit affine g=32, non-
//     routed 8-bit affine g=32
//
// 2026-05-04 (DSV4 SWA/CSA/HSA correctness pass):
// `TurboQuantSwitchGLU` now accepts a `swigluLimit:` parameter that
// activates the DSV4 limited-SwiGLU clamp inside the fused gate+up
// Metal kernel: `silu(min(gate, 10)) * clip(up, -10, 10)`. This
// matches the codex_dsv4_fixkit `runtime_dsv4_fixed.py` patch and
// the `jang_tools.dsv4.mlx_model._dsv4_swiglu` reference. Shared
// experts (dense path) keep applying the clamp via `DeepseekV4MLP`.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - JANGTQ MoE (routed via TurboQuantSwitchGLU)

final class DeepseekV4MoEJANGTQ: Module, UnaryLayer {
    let config: DeepseekV4Configuration
    let layerIdx: Int
    let topK: Int
    @ModuleInfo(key: "switch_mlp") var switchMLP: TurboQuantSwitchGLU
    var gate: DeepseekV4MoEGate
    @ModuleInfo(key: "shared_experts") var sharedExperts: DeepseekV4MLP

    init(config: DeepseekV4Configuration, layerIdx: Int, mxtqBits: Int, mxtqSeed: Int) {
        self.config = config
        self.layerIdx = layerIdx
        self.topK = config.numExpertsPerTok
        if JANGTQStreamingExperts.isEnabled {
            self._switchMLP.wrappedValue = StreamingTurboQuantSwitchGLU(
                inputDims: config.hiddenSize,
                hiddenDims: config.moeIntermediateSize,
                numExperts: config.nRoutedExperts,
                gateUpBits: mxtqBits,
                downBits: mxtqBits,
                seed: mxtqSeed,
                swigluLimit: config.swigluLimit,
                layerIdx: layerIdx)
        } else {
            self._switchMLP.wrappedValue = TurboQuantSwitchGLU(
                inputDims: config.hiddenSize,
                hiddenDims: config.moeIntermediateSize,
                numExperts: config.nRoutedExperts,
                bits: mxtqBits,
                seed: mxtqSeed,
                // DSV4-Flash always routes through limited-SwiGLU. Pulling
                // the magnitude from `config.swigluLimit` (10.0 by default)
                // matches the dense path applied by `DeepseekV4MLP` and the
                // Python `_dsv4_swiglu` helper. Without this clamp, deep MoE
                // stacks diverge numerically per the DSV4 reference impl.
                swigluLimit: config.swigluLimit)
        }
        self.gate = DeepseekV4MoEGate(config: config, layerIdx: layerIdx)
        self._sharedExperts.wrappedValue = DeepseekV4MLP(
            hiddenSize: config.hiddenSize,
            intermediateSize: config.moeIntermediateSize * config.nSharedExperts,
            swigluLimit: config.swigluLimit)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        callAsFunction(x, inputIds: nil)
    }

    // See `DeepseekV4MoE`: request-scoped, must not be parked on `self`.
    func callAsFunction(_ x: MLXArray, inputIds: MLXArray?) -> MLXArray {
        let (indices, scores) = gate(x, inputIds: gate.isHashLayer ? inputIds : nil)
        JangPressCanonicalExpertAdvisor.shared.observe(layer: layerIdx, indices: indices)
        var y: MLXArray
        if let streaming = switchMLP as? StreamingTurboQuantSwitchGLU {
            y = streaming.reduced(x, indices: indices, scores: scores)
        } else {
            y = switchMLP(x, indices)
            y = (y * scores.asType(y.dtype)[.ellipsis, .newAxis]).sum(axis: -2)
        }
        y = y + sharedExperts(x)
        return y
    }
}

// MARK: - JANGTQ decoder layer (same wrapping as DSV4, JANGTQ MoE swap)

final class DeepseekV4DecoderLayerJANGTQ: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: DeepseekV4Attention
    @ModuleInfo(key: "mlp") var mlp: DeepseekV4MoEJANGTQ
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "attn_hc") var attnHC: DeepseekV4HyperConnection
    @ModuleInfo(key: "ffn_hc") var ffnHC: DeepseekV4HyperConnection

    init(config: DeepseekV4Configuration, layerIdx: Int, mxtqBits: Int, mxtqSeed: Int) {
        self._selfAttn.wrappedValue = DeepseekV4Attention(config: config, layerIdx: layerIdx)
        self._mlp.wrappedValue = DeepseekV4MoEJANGTQ(
            config: config, layerIdx: layerIdx, mxtqBits: mxtqBits, mxtqSeed: mxtqSeed)
        self._inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._attnHC.wrappedValue = DeepseekV4HyperConnection(config: config)
        self._ffnHC.wrappedValue = DeepseekV4HyperConnection(config: config)
    }

    func callAsFunction(
        _ h: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?,
        inputIds: MLXArray?
    ) -> MLXArray {
        let residualA = h
        let (xA, postA, combA) = attnHC.collapse(h)
        let attnOut = selfAttn(inputLayerNorm(xA), mask: mask, cache: cache)
        let hA = attnHC.expand(blockOut: attnOut, residual: residualA, post: postA, comb: combA)
        let residualF = hA
        let (xF, postF, combF) = ffnHC.collapse(hA)
        let ffnOut = mlp(postAttentionLayerNorm(xF), inputIds: inputIds)
        return ffnHC.expand(blockOut: ffnOut, residual: residualF, post: postF, comb: combF)
    }
}

// MARK: - JANGTQ inner + outer model

public final class DeepseekV4ModelInnerJANGTQ: Module {
    let config: DeepseekV4Configuration
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    var layers: [DeepseekV4DecoderLayerJANGTQ]
    public let routedExpertBitsByLayer: [Int]
    @ModuleInfo(key: "hc_head") var hcHead: DeepseekV4HyperHead
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(config: DeepseekV4Configuration, mxtqBits: Int?, mxtqSeed: Int) {
        self.config = config
        let layerBits = (0..<config.numHiddenLayers).map {
            mxtqBits ?? config.routedExpertBits(forLayer: $0)
        }
        self.routedExpertBitsByLayer = layerBits
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        self.layers = (0..<config.numHiddenLayers).map {
            DeepseekV4DecoderLayerJANGTQ(
                config: config,
                layerIdx: $0,
                mxtqBits: layerBits[$0],
                mxtqSeed: mxtqSeed)
        }
        self._hcHead.wrappedValue = DeepseekV4HyperHead(config: config)
        self._norm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = embedTokens(inputs)
        h = h.expandedDimensions(axis: -2)
        h = repeated(h, count: config.hcMult, axis: -2)
        let hFlat2 = h.reshaped(h.dim(0), h.dim(1), -1)
        let mask = createAttentionMask(h: hFlat2, cache: cache?.first)
        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i], inputIds: inputs)
        }
        return norm(hcHead.reduce(h))
    }
}

public final class DeepseekV4JANGTQModel:
    Module, LLMModel, KVCacheDimensionProvider, LoRAModel
{
    public var kvHeads: [Int]
    let config: DeepseekV4Configuration
    public var model: DeepseekV4ModelInnerJANGTQ
    public var routedExpertBitsByLayer: [Int] { model.routedExpertBitsByLayer }
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    public init(_ config: DeepseekV4Configuration, mxtqBits: Int? = nil, mxtqSeed: Int = 42) {
        self.config = config
        self.kvHeads = Array(repeating: 1, count: config.numHiddenLayers)
        self.model = DeepseekV4ModelInnerJANGTQ(
            config: config, mxtqBits: mxtqBits, mxtqSeed: mxtqSeed)
        self._lmHead.wrappedValue = Linear(
            config.hiddenSize, config.vocabSize, bias: false)
    }

    /// 2026-05-04 (DSV4 SWA/CSA/HSA correctness pass — pure long-context):
    /// Identical to `DeepseekV4Model.newCache(parameters:)` —
    /// always allocates the hybrid `DeepseekV4Cache` for `cr>0` layers
    /// and a `RotatingKVCache(window=128)` for `cr=0` layers. The
    /// `DSV4_LONG_CTX` toggle is gone (was a regression vector).
    /// Caller-level `.turboQuant` does not switch cache topology because
    /// that would drop CSA/HSA; only explicit `DSV4_KV_MODE=tq` does.
    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        let env = ProcessInfo.processInfo.environment
        let envMode = env["DSV4_KV_MODE"]?.lowercased()
        let mode: String = envMode ?? "sliding"

        return (0..<config.numHiddenLayers).map { layerIdx in
            switch mode {
            case "full", "tq":
                return KVCacheSimple()
            default:
                let cr =
                    config.compressRatios.count > layerIdx
                    ? config.compressRatios[layerIdx]
                    : DeepseekV4Model.defaultCompressRatio(
                        layerIdx: layerIdx,
                        numLayers: config.numHiddenLayers)
                if cr > 0 {
                    return DeepseekV4Cache(
                        slidingWindow: config.slidingWindow,
                        compressRatio: cr)
                }
                return RotatingKVCache(
                    maxSize: config.slidingWindow, keep: 0)
            }
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        DeepseekV4Math.lmHeadFp32(model(inputs, cache: cache), lmHead: lmHead)
    }

    /// Reuse DeepseekV4Model's sanitize — the weight naming contract
    /// is identical for JANGTQ bundles. The TurboQuantSwitchLinear
    /// infrastructure handles the MXTQ codebook indirection at load
    /// time via its own key renaming in `TurboQuantSwitchLinear.keyMap`.
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        // Delegate to the affine variant's sanitize — same remap rules.
        let stub = DeepseekV4Model(config)
        return stub.sanitize(weights: weights)
    }

    public var loraLayers: [Module] {
        model.layers
    }
}
