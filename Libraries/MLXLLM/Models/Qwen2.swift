//
//  Qwen2.swift
//  LLM
//
//  Created by John Mai on 2024/3/3.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen2.py

class Qwen2Attention: Module {
    let args: Qwen2Configuration
    let scale: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    let rope: RoPE

    public init(_ args: Qwen2Configuration) {
        self.args = args

        let dim = args.hiddenSize
        let heads = args.attentionHeads
        let kvHeads = args.kvHeads

        let headDim = args.hiddenSize / heads
        self.scale = pow(Float(headDim), -0.5)

        _wq.wrappedValue = Linear(dim, heads * headDim, bias: true)
        _wk.wrappedValue = Linear(dim, kvHeads * headDim, bias: true)
        _wv.wrappedValue = Linear(dim, kvHeads * headDim, bias: true)
        _wo.wrappedValue = Linear(heads * headDim, dim, bias: false)

        self.rope = RoPE(
            dimensions: headDim, traditional: args.ropeTraditional, base: args.ropeTheta,
            scale: args.ropeScale)
    }

    public func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = wq(x)
        var keys = wk(x)
        var values = wv(x)

        // prepare the queries, keys and values for the attention computation
        queries = queries.reshaped(B, L, args.attentionHeads, -1).transposed(0, 2, 1, 3)
        keys = keys.reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)

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

        return wo(output)
    }
}

class Qwen2MLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear

    public init(dimensions: Int, hiddenDimensions: Int) {
        _gate.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        _down.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        _up.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

class Qwen2TransformerBlock: Module {
    @ModuleInfo(key: "self_attn") var attention: Qwen2Attention
    let mlp: Qwen2MLP

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    public init(_ args: Qwen2Configuration) {
        _attention.wrappedValue = Qwen2Attention(args)
        self.mlp = Qwen2MLP(dimensions: args.hiddenSize, hiddenDimensions: args.intermediateSize)
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    public func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        var r = attention(inputLayerNorm(x), mask: mask, cache: cache)
        let h = x + r
        r = mlp(postAttentionLayerNorm(h))
        let out = h + r
        return out
    }
}

public class Qwen2ModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    fileprivate let layers: [Qwen2TransformerBlock]
    let norm: RMSNorm

    public init(_ args: Qwen2Configuration) {
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)

        self.layers = (0 ..< args.hiddenLayers)
            .map { _ in
                Qwen2TransformerBlock(args)
            }
        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embedTokens(inputs)

        let mask = createAttentionMask(h: h, cache: cache?.first)

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }

        return norm(h)
    }
}

public class Qwen2Model: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: Qwen2ModelInner
    let configuration: Qwen2Configuration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ args: Qwen2Configuration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
        self.model = Qwen2ModelInner(args)

        if !args.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var out = model(inputs, cache: cache)
        if let lmHead {
            out = lmHead(out)
        } else {
            out = model.embedTokens.asLinear(out)
        }
        return out
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var weights = weights

        if configuration.tieWordEmbeddings {
            weights["lm_head.weight"] = nil
        }

        // Remove unused precomputed rotary freqs
        return weights.filter {
            !$0.key.contains("self_attn.rotary_emb.inv_freq")
        }
    }
}

public struct Qwen2Configuration: Codable, Sendable {
    var hiddenSize: Int
    var hiddenLayers: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var rmsNormEps: Float
    var vocabularySize: Int
    var kvHeads: Int
    var ropeTheta: Float = 1_000_000
    var ropeTraditional: Bool = false
    var ropeScaling: [String: StringOrNumber]? = nil
    var tieWordEmbeddings = false

    var ropeScale: Float {
        guard let ropeScaling, ropeScaling["type"] == .string("linear"),
            let factor = ropeScaling["factor"]?.asFloat()
        else {
            return 1
        }
        return 1 / factor
    }

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case kvHeads = "num_key_value_heads"
        case ropeTheta = "rope_theta"
        case ropeTraditional = "rope_traditional"
        case ropeScaling = "rope_scaling"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    public init(from decoder: Decoder) throws {
        // custom implementation to handle optional keys with required values
        let container: KeyedDecodingContainer<Qwen2Configuration.CodingKeys> =
            try decoder.container(
                keyedBy: Qwen2Configuration.CodingKeys.self)

        self.hiddenSize = try container.decode(
            Int.self, forKey: Qwen2Configuration.CodingKeys.hiddenSize)
        self.hiddenLayers = try container.decode(
            Int.self, forKey: Qwen2Configuration.CodingKeys.hiddenLayers)
        self.intermediateSize = try container.decode(
            Int.self, forKey: Qwen2Configuration.CodingKeys.intermediateSize)
        self.attentionHeads = try container.decode(
            Int.self, forKey: Qwen2Configuration.CodingKeys.attentionHeads)
        self.rmsNormEps = try container.decode(
            Float.self, forKey: Qwen2Configuration.CodingKeys.rmsNormEps)
        self.vocabularySize = try container.decode(
            Int.self, forKey: Qwen2Configuration.CodingKeys.vocabularySize)
        self.kvHeads = try container.decode(Int.self, forKey: Qwen2Configuration.CodingKeys.kvHeads)
        self.ropeTheta =
            try container.decodeIfPresent(
                Float.self, forKey: Qwen2Configuration.CodingKeys.ropeTheta)
            ?? 1_000_000
        self.ropeTraditional =
            try container.decodeIfPresent(
                Bool.self, forKey: Qwen2Configuration.CodingKeys.ropeTraditional) ?? false
        self.ropeScaling = try container.decodeIfPresent(
            [String: StringOrNumber].self, forKey: Qwen2Configuration.CodingKeys.ropeScaling)
        self.tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false

        try Self.validatePositive(hiddenSize, key: .hiddenSize, in: container)
        try Self.validatePositive(hiddenLayers, key: .hiddenLayers, in: container)
        try Self.validatePositive(intermediateSize, key: .intermediateSize, in: container)
        try Self.validatePositive(attentionHeads, key: .attentionHeads, in: container)
        try Self.validatePositive(rmsNormEps, key: .rmsNormEps, in: container)
        try Self.validatePositive(vocabularySize, key: .vocabularySize, in: container)
        try Self.validatePositive(kvHeads, key: .kvHeads, in: container)
        try Self.validatePositive(ropeTheta, key: .ropeTheta, in: container)
        guard hiddenSize % attentionHeads == 0 else {
            throw DecodingError.dataCorruptedError(
                forKey: .hiddenSize,
                in: container,
                debugDescription:
                    "Qwen2 hidden_size must be divisible by num_attention_heads.")
        }
        guard attentionHeads % kvHeads == 0 else {
            throw DecodingError.dataCorruptedError(
                forKey: .kvHeads,
                in: container,
                debugDescription:
                    "Qwen2 num_attention_heads must be divisible by num_key_value_heads.")
        }
        try Self.validateRopeScaling(ropeScaling, key: .ropeScaling, in: container)
    }

    private static func validatePositive<K: CodingKey>(
        _ value: Int, key: K, in container: KeyedDecodingContainer<K>
    ) throws {
        guard value > 0 else {
            throw DecodingError.dataCorruptedError(
                forKey: key,
                in: container,
                debugDescription: "\(key.stringValue) must be greater than zero.")
        }
    }

    private static func validatePositive<K: CodingKey>(
        _ value: Float, key: K, in container: KeyedDecodingContainer<K>
    ) throws {
        guard value.isFinite, value > 0 else {
            throw DecodingError.dataCorruptedError(
                forKey: key,
                in: container,
                debugDescription: "\(key.stringValue) must be finite and greater than zero.")
        }
    }

    private static func validateRopeScaling<K: CodingKey>(
        _ value: [String: StringOrNumber]?, key: K, in container: KeyedDecodingContainer<K>
    ) throws {
        guard let value else { return }
        if let type = value["type"], type != .string("linear") {
            return
        }
        guard let factor = value["factor"]?.asFloat(), factor.isFinite, factor > 0 else {
            throw DecodingError.dataCorruptedError(
                forKey: key,
                in: container,
                debugDescription: "Qwen2 rope_scaling.factor must be a positive float.")
        }
    }
}

// MARK: - LoRA

extension Qwen2Model: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
