// Copyright 2025 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Native runtime + configuration contract for Tencent Hunyuan v1 dense
// (`model_type = hunyuan_v1_dense`, `architectures = ["HunYuanDenseV1ForCausalLM"]`).
//
// Runtime shape: a standard Llama-style dense GQA decoder with two deltas:
//   1. Per-head Q/K RMSNorm (`query_layernorm` / `key_layernorm`) applied
//      *after* RoPE (note: opposite order from Hunyuan v3 `hy_v3`).
//   2. `DynamicNTKAlphaRoPE`, which is a plain RoPE with the base rescaled by
//      `alpha ** (dims / (dims - 2))` (from `rope_scaling.alpha`). With explicit
//      `base'`, `mx.fast.rope(base: base')` is identical to the reference's
//      precomputed-`freqs` path, so no custom-frequency machinery is needed.
//
// Port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/hunyuan_v1_dense.py
// Resolves osaurus issue #358 ("Unsupported model type: hunyuan_v1_dense").

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Attention

private class HunyuanV1DenseAttention: Module {

    let args: HunyuanV1DenseConfiguration
    let scale: Float
    let useQKNorm: Bool

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    @ModuleInfo(key: "query_layernorm") var queryNorm: RMSNorm?
    @ModuleInfo(key: "key_layernorm") var keyNorm: RMSNorm?

    let rope: RoPE

    init(_ args: HunyuanV1DenseConfiguration) {
        self.args = args

        let dim = args.hiddenSize
        let heads = args.attentionHeads
        let kvHeads = args.kvHeads
        let headDim = args.resolvedHeadDimensions
        self.scale = pow(Float(headDim), -0.5)
        self.useQKNorm = args.useQKNorm

        self._wq.wrappedValue = Linear(dim, heads * headDim, bias: args.attentionBias)
        self._wk.wrappedValue = Linear(dim, kvHeads * headDim, bias: args.attentionBias)
        self._wv.wrappedValue = Linear(dim, kvHeads * headDim, bias: args.attentionBias)
        self._wo.wrappedValue = Linear(heads * headDim, dim, bias: args.attentionBias)

        if args.useQKNorm {
            self._queryNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)
            self._keyNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)
        }

        // DynamicNTKAlphaRoPE collapses to a plain RoPE with a rescaled base.
        self.rope = RoPE(
            dimensions: headDim,
            traditional: false,
            base: args.effectiveRopeBase,
            scale: 1)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = wq(x)
        var keys = wk(x)
        var values = wv(x)

        let headDim = args.resolvedHeadDimensions
        queries = queries.reshaped(B, L, -1, headDim).transposed(0, 2, 1, 3)
        keys = keys.reshaped(B, L, -1, headDim).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, -1, headDim).transposed(0, 2, 1, 3)

        // RoPE first, then per-head Q/K norm (matches the reference order).
        queries = applyRotaryPosition(rope, to: queries, cache: cache)
        keys = applyRotaryPosition(rope, to: keys, cache: cache)

        if useQKNorm, let queryNorm, let keyNorm {
            queries = queryNorm(queries)
            keys = keyNorm(keys)
        }

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

        return wo(output)
    }
}

// MARK: - MLP

private class HunyuanV1DenseMLP: Module, UnaryLayer {

    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear

    init(_ args: HunyuanV1DenseConfiguration) {
        self._gate.wrappedValue = Linear(args.hiddenSize, args.intermediateSize, bias: false)
        self._down.wrappedValue = Linear(args.intermediateSize, args.hiddenSize, bias: false)
        self._up.wrappedValue = Linear(args.hiddenSize, args.intermediateSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

// MARK: - Decoder block

private class HunyuanV1DenseTransformerBlock: Module {
    @ModuleInfo(key: "self_attn") var attention: HunyuanV1DenseAttention
    @ModuleInfo(key: "mlp") var mlp: HunyuanV1DenseMLP

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ args: HunyuanV1DenseConfiguration) {
        self._attention.wrappedValue = HunyuanV1DenseAttention(args)
        self._mlp.wrappedValue = HunyuanV1DenseMLP(args)
        self._inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        var r = attention(inputLayerNorm(x), mask: mask, cache: cache)
        let h = x + r
        r = mlp(postAttentionLayerNorm(h))
        return h + r
    }
}

// MARK: - Model

private class HunyuanV1DenseModelInner: Module {

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    fileprivate let layers: [HunyuanV1DenseTransformerBlock]
    let norm: RMSNorm

    init(_ args: HunyuanV1DenseConfiguration) {
        precondition(args.vocabularySize > 0)

        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)

        self.layers = (0 ..< args.hiddenLayers).map { _ in HunyuanV1DenseTransformerBlock(args) }
        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embedTokens(inputs)

        let mask = createAttentionMask(h: h, cache: cache?.first)

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }

        return norm(h)
    }
}

public class HunyuanV1DenseModel: Module, LLMModel, KVCacheDimensionProvider {

    public let vocabularySize: Int
    public let kvHeads: [Int]

    fileprivate let model: HunyuanV1DenseModelInner
    let configuration: HunyuanV1DenseConfiguration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ args: HunyuanV1DenseConfiguration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
        self.model = HunyuanV1DenseModelInner(args)
        if !args.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let out = model(inputs, cache: cache)
        if let lmHead {
            return lmHead(out)
        } else {
            return model.embedTokens.asLinear(out)
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = weights.filter {
            !$0.key.contains("self_attn.rotary_emb.inv_freq")
        }
        if configuration.tieWordEmbeddings {
            sanitized["lm_head.weight"] = nil
        }
        return sanitized
    }

    public func messageGenerator(tokenizer: any Tokenizer) -> any MessageGenerator {
        // some models allow the system role and some do not -- this is enforced
        // by the chat template (code).
        do {
            let probe = [["role": "system", "content": "test"]]
            _ = try tokenizer.applyChatTemplate(messages: probe)
            return DefaultMessageGenerator()
        } catch {
            return NoSystemMessageGenerator()
        }
    }
}

// MARK: - Configuration

public struct HunyuanV1DenseConfiguration: Codable, Sendable {

    var hiddenSize: Int
    var hiddenLayers: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var headDimensions: Int?
    var rmsNormEps: Float
    var vocabularySize: Int
    var kvHeads: Int
    var maxPositionEmbeddings: Int = 32768
    var ropeTheta: Float = 10_000
    var useQKNorm: Bool = true
    var ropeScalingAlpha: Float = 1.0
    var tieWordEmbeddings: Bool = false
    var attentionBias: Bool = false

    var resolvedHeadDimensions: Int {
        headDimensions ?? (hiddenSize / attentionHeads)
    }

    /// `DynamicNTKAlphaRoPE`: base rescaled by `alpha ** (dims / (dims - 2))`.
    var effectiveRopeBase: Float {
        let dims = Float(resolvedHeadDimensions)
        guard ropeScalingAlpha != 1.0, dims > 2 else { return ropeTheta }
        return ropeTheta * pow(ropeScalingAlpha, dims / (dims - 2))
    }

    private struct RopeScaling: Codable {
        let alpha: Double?
    }

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case headDimensions = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case kvHeads = "num_key_value_heads"
        case maxPositionEmbeddings = "max_position_embeddings"
        case ropeTheta = "rope_theta"
        case useQKNorm = "use_qk_norm"
        case ropeScaling = "rope_scaling"
        case tieWordEmbeddings = "tie_word_embeddings"
        case attentionBias = "attention_bias"
    }

    public init(from decoder: Swift.Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        headDimensions = try container.decodeIfPresent(Int.self, forKey: .headDimensions)
        rmsNormEps = try container.decode(Float.self, forKey: .rmsNormEps)
        vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)
        kvHeads = try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? attentionHeads
        if let maxPositionEmbeddings = try container.decodeIfPresent(
            Int.self, forKey: .maxPositionEmbeddings)
        {
            self.maxPositionEmbeddings = maxPositionEmbeddings
        }
        if let ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) {
            self.ropeTheta = ropeTheta
        }
        if let useQKNorm = try container.decodeIfPresent(Bool.self, forKey: .useQKNorm) {
            self.useQKNorm = useQKNorm
        }
        if let ropeScaling = try container.decodeIfPresent(
            RopeScaling.self, forKey: .ropeScaling), let alpha = ropeScaling.alpha
        {
            self.ropeScalingAlpha = Float(alpha)
        }
        if let tieWordEmbeddings = try container.decodeIfPresent(
            Bool.self, forKey: .tieWordEmbeddings)
        {
            self.tieWordEmbeddings = tieWordEmbeddings
        }
        if let attentionBias = try container.decodeIfPresent(Bool.self, forKey: .attentionBias) {
            self.attentionBias = attentionBias
        }
    }

    public func encode(to encoder: Swift.Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(hiddenSize, forKey: .hiddenSize)
        try container.encode(hiddenLayers, forKey: .hiddenLayers)
        try container.encode(intermediateSize, forKey: .intermediateSize)
        try container.encode(attentionHeads, forKey: .attentionHeads)
        try container.encodeIfPresent(headDimensions, forKey: .headDimensions)
        try container.encode(rmsNormEps, forKey: .rmsNormEps)
        try container.encode(vocabularySize, forKey: .vocabularySize)
        try container.encode(kvHeads, forKey: .kvHeads)
        try container.encode(maxPositionEmbeddings, forKey: .maxPositionEmbeddings)
        try container.encode(ropeTheta, forKey: .ropeTheta)
        try container.encode(useQKNorm, forKey: .useQKNorm)
        try container.encode(RopeScaling(alpha: Double(ropeScalingAlpha)), forKey: .ropeScaling)
        try container.encode(tieWordEmbeddings, forKey: .tieWordEmbeddings)
        try container.encode(attentionBias, forKey: .attentionBias)
    }
}

// MARK: - LoRA

extension HunyuanV1DenseModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
