//
//  DeepseekOCRLanguage.swift
//  mlx-swift-lm
//
//  DeepSeek-V2 MoE text decoder for the DeepSeek-OCR / Unlimited-OCR VLM.
//  Line-by-line port of mlx-vlm's deepseekocr/language.py.
//
//  Architecture (verified from config.json):
//    - STANDARD multi-head attention with RoPE (NOT MLA). In config use_mla=false,
//      q_lora_rank=null, kv_lora_rank=null, qk_nope_head_dim=0 ⇒ mlx-vlm sets
//      attn_type="LlamaAttention". So this implements the LlamaAttention branch:
//      head_dim = hidden_size / num_attention_heads = 1280 / 10 = 128.
//    - MoE: n_routed_experts=64, n_shared_experts=2, num_experts_per_tok=6,
//      moe_intermediate_size=896, first_k_dense_replace=1 (layer 0 dense, 1-11 MoE),
//      topk_method="greedy", scoring_func="softmax", routed_scaling_factor=1.0.
//
//  Weight-key layout after the top model wraps this as `language_model`:
//    language_model.model.embed_tokens
//    language_model.model.layers.N.{self_attn,mlp,input_layernorm,post_attention_layernorm}
//    language_model.model.norm
//    language_model.lm_head
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

private typealias TextConfiguration = DeepseekOCRConfiguration.TextConfiguration

// MARK: - Attention (LlamaAttention branch)

/// Standard multi-head attention with RoPE. Mirrors language.py's `LlamaAttention`
/// (the branch used when qk_nope_head_dim == 0).
private class DeepseekOCRLlamaAttention: Module {
    let nHeads: Int
    let nKVHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    let rope: RoPE

    init(_ config: TextConfiguration) {
        let dim = config.hiddenSize
        self.nHeads = config.numAttentionHeads
        self.nKVHeads = config.numKeyValueHeads
        // head_dim = hidden_size // num_attention_heads (NOT num_key_value_heads)
        self.headDim = config.hiddenSize / config.numAttentionHeads
        self.scale = pow(Float(headDim), -0.5)

        let attentionBias = config.attentionBias

        self._qProj.wrappedValue = Linear(dim, nHeads * headDim, bias: attentionBias)
        self._kProj.wrappedValue = Linear(dim, nKVHeads * headDim, bias: attentionBias)
        self._vProj.wrappedValue = Linear(dim, nKVHeads * headDim, bias: attentionBias)
        self._oProj.wrappedValue = Linear(nHeads * headDim, dim, bias: attentionBias)

        // rope_scaling is only consulted for the "linear" type in language.py.
        var ropeScale: Float = 1
        if let ropeScaling = config.ropeScaling,
            ropeScaling["type"] == .string("linear"),
            let factor = ropeScaling["factor"]?.asFloat()
        {
            ropeScale = 1 / factor
        }

        // rope_traditional defaults to false in mlx-vlm's TextConfig.
        self.rope = RoPE(
            dimensions: headDim, traditional: false, base: config.ropeTheta, scale: ropeScale)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = qProj(x)
        var keys = kProj(x)
        var values = vProj(x)

        queries = queries.reshaped(B, L, nHeads, -1).transposed(0, 2, 1, 3)
        keys = keys.reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)

        queries = applyRotaryPosition(rope, to: queries, cache: cache)
        keys = applyRotaryPosition(rope, to: keys, cache: cache)

        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return oProj(output)
    }
}

// MARK: - Dense MLP

/// Dense gated MLP. Used by layer 0 (first_k_dense_replace) and the shared experts.
private class DeepseekOCRMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(config: TextConfiguration, hiddenSize: Int? = nil, intermediateSize: Int? = nil) {
        let hidden = hiddenSize ?? config.hiddenSize
        let inter = intermediateSize ?? config.intermediateSize
        self._gateProj.wrappedValue = Linear(hidden, inter, bias: false)
        self._upProj.wrappedValue = Linear(hidden, inter, bias: false)
        self._downProj.wrappedValue = Linear(inter, hidden, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

// MARK: - MoE gate

/// Router gate. Mirrors language.py's `MoEGate` for scoring_func="softmax" and
/// topk_method="greedy" (the configuration used by DeepSeek-OCR).
private class DeepseekOCRMoEGate: Module {
    let scoringFunc: String
    let topK: Int
    let nRoutedExperts: Int
    let routedScalingFactor: Float
    let topkMethod: String
    let nGroup: Int
    let topkGroup: Int

    // `weight` is a bare parameter named "weight" (matches gate.weight in safetensors).
    var weight: MLXArray
    // Only allocated for topk_method == "noaux_tc"; present so the parameter
    // name resolves if such weights ever appear. DeepSeek-OCR uses "greedy".
    @ParameterInfo(key: "e_score_correction_bias") var eScoreCorrectionBias: MLXArray

    init(_ config: TextConfiguration) {
        self.scoringFunc = config.scoringFunc
        self.topK = config.numExpertsPerTok ?? 1
        self.nRoutedExperts = config.nRoutedExperts ?? 1
        self.routedScalingFactor = config.routedScalingFactor
        self.topkMethod = config.topkMethod
        self.nGroup = config.nGroup ?? 1
        self.topkGroup = config.topkGroup ?? 1
        self.weight = zeros([nRoutedExperts, config.hiddenSize])
        self._eScoreCorrectionBias.wrappedValue = zeros([nRoutedExperts])
    }

    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        // gates = x @ weight.T
        let gates = x.matmul(weight.T)

        let scores: MLXArray
        switch scoringFunc {
        case "softmax":
            scores = softmax(gates, axis: -1, precise: true)
        case "sigmoid":
            scores = sigmoid(gates)
        default:
            fatalError("Unknown scoring function: \(scoringFunc)")
        }

        // DeepSeek-OCR uses topk_method == "greedy".
        // inds = argpartition(scores, kth=-k)[..., -k:]; gather selected scores.
        // (argPartition places the k largest in the last k positions; the
        //  exact ordering within those k is irrelevant — the weighted sum is
        //  permutation-invariant.)
        precondition(topkMethod == "greedy", "DeepseekOCR gate only supports topk_method=greedy")
        let k = topK
        let total = scores.dim(-1)
        let inds = argPartition(scores, kth: total - k, axis: -1)[.ellipsis, (total - k)...]
        var scoresSelected = takeAlong(scores, inds, axis: -1)

        scoresSelected = scoresSelected * routedScalingFactor
        return (inds, scoresSelected)
    }
}

// MARK: - MoE

/// Sparse mixture of experts with optional shared experts. Mirrors `DeepseekV2MoE`.
private class DeepseekOCRMoE: Module, UnaryLayer {
    let numExpertsPerTok: Int
    let hasSharedExperts: Bool
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    var gate: DeepseekOCRMoEGate
    @ModuleInfo(key: "shared_experts") var sharedExperts: DeepseekOCRMLP?

    init(_ config: TextConfiguration) {
        self.numExpertsPerTok = config.numExpertsPerTok ?? 1

        self._switchMLP.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.moeIntermediateSize,
            numExperts: config.nRoutedExperts ?? 1
        )

        self.gate = DeepseekOCRMoEGate(config)

        if let sharedCount = config.nSharedExperts {
            self.hasSharedExperts = true
            let intermediateSize = config.moeIntermediateSize * sharedCount
            self._sharedExperts.wrappedValue = DeepseekOCRMLP(
                config: config, intermediateSize: intermediateSize)
        } else {
            self.hasSharedExperts = false
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (inds, scores) = gate(x)
        var y = switchMLP(x, inds)
        y = (y * scores[.ellipsis, .newAxis]).sum(axis: -2)
        if let shared = sharedExperts {
            y = y + shared(x)
        }
        return y
    }
}

// MARK: - Decoder layer

private class DeepseekOCRDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: DeepseekOCRLlamaAttention
    var mlp: UnaryLayer
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ config: TextConfiguration, layerIdx: Int) {
        self._selfAttn.wrappedValue = DeepseekOCRLlamaAttention(config)

        // MoE when: n_routed_experts present AND layer >= first_k_dense_replace
        // AND layer % moe_layer_freq == 0. Otherwise a dense MLP.
        if config.nRoutedExperts != nil,
            layerIdx >= config.firstKDenseReplace,
            layerIdx % config.moeLayerFreq == 0
        {
            self.mlp = DeepseekOCRMoE(config)
        } else {
            self.mlp = DeepseekOCRMLP(config: config)
        }

        self._inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let r = selfAttn(inputLayerNorm(x), mask: mask, cache: cache)
        let h = x + r
        let r2 = mlp(postAttentionLayerNorm(h))
        return h + r2
    }
}

// MARK: - Inner model

/// The `model` submodule: embeddings, decoder stack, final norm.
/// Produces weight keys `language_model.model.*` once wrapped as `language_model`.
class DeepseekOCRTextModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    fileprivate var layers: [DeepseekOCRDecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(_ config: TextConfiguration) {
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        self.layers = (0 ..< config.numHiddenLayers).map {
            DeepseekOCRDecoderLayer(config, layerIdx: $0)
        }
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(
        _ inputs: MLXArray,
        cache: [KVCache]?,
        inputsEmbeds: MLXArray? = nil
    ) -> MLXArray {
        // When inputsEmbeds is provided (image-feature injection), use it instead
        // of embedding the input ids. Mirrors language.py DeepseekV2Model.__call__.
        var h: MLXArray
        if let inputsEmbeds {
            h = inputsEmbeds
        } else {
            h = embedTokens(inputs)
        }

        let mask = createAttentionMask(h: h, cache: cache?.first)

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }

        return norm(h)
    }
}

// MARK: - Language model

/// Public DeepSeek-V2 MoE text decoder. The top VLM model wraps an instance of
/// this as `language_model`, yielding the `language_model.*` weight keys.
public class DeepseekOCRLanguageModel: Module, KVCacheDimensionProvider {
    let config: TextConfiguration

    /// Exposes `model.embedTokens` so callers can do
    /// `language_model.model.embed_tokens(input_ids)` for image-feature injection.
    @ModuleInfo(key: "model") var model: DeepseekOCRTextModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    public var kvHeads: [Int] {
        Array(repeating: config.numKeyValueHeads, count: config.numHiddenLayers)
    }

    public init(_ config: TextConfiguration) {
        self.config = config
        self._model.wrappedValue = DeepseekOCRTextModelInner(config)
        self._lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabSize, bias: false)
    }

    /// Forward pass returning logits. When `inputsEmbeds` is provided it is used
    /// instead of embedding `inputs` (image features are injected this way).
    public func callAsFunction(
        _ inputs: MLXArray,
        cache: [KVCache]? = nil,
        inputsEmbeds: MLXArray? = nil
    ) -> MLXArray {
        let out = model(inputs, cache: cache, inputsEmbeds: inputsEmbeds)
        return lmHead(out)
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        (0 ..< config.numHiddenLayers).map { _ in
            if let maxKVSize = parameters?.maxKVSize {
                return RotatingKVCache(maxSize: maxKVSize, keep: 4)
            } else {
                return KVCacheSimple()
            }
        }
    }

    /// Stacks per-expert MLP weights into the SwitchGLU `switch_mlp` layout.
    /// Mirrors language.py's `LanguageModel.sanitize`. The top VLM model should
    /// call this from its own `sanitize`. Keys are the `language_model.model.*`
    /// paths produced once this module is wrapped as `language_model`.
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var newWeights = weights
        let nExperts = config.nRoutedExperts ?? 1

        for l in 0 ..< config.numHiddenLayers {
            let prefix = "language_model.model.layers.\(l)"
            for (_, projName) in [("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")] {
                for key in ["weight", "scales", "biases"] {
                    let firstKey = "\(prefix).mlp.experts.0.\(projName).\(key)"
                    if weights[firstKey] != nil {
                        let toJoin = (0 ..< nExperts).map { e -> MLXArray in
                            let k = "\(prefix).mlp.experts.\(e).\(projName).\(key)"
                            return newWeights.removeValue(forKey: k) ?? weights[k]!
                        }
                        newWeights["\(prefix).mlp.switch_mlp.\(projName).\(key)"] = stacked(toJoin)
                    }
                }
            }
        }

        return newWeights
    }
}
