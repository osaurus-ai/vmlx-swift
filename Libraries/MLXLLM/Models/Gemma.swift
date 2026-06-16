// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// Port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/gemma.py

// Specialized norm for Gemma
class GemmaRMSNorm: Module, UnaryLayer {
    let weight: MLXArray
    let eps: Float

    public init(dimensions: Int, eps: Float = 1e-5) {
        self.weight = MLXArray.ones([dimensions])
        self.eps = eps
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        return MLXFast.rmsNorm(x, weight: 1.0 + self.weight, eps: self.eps)
    }
}

class GemmaAttention: Module {
    let args: GemmaConfiguration
    let nHeads: Int
    let nKVHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    let rope: RoPE

    public init(_ args: GemmaConfiguration) {
        self.args = args

        let dim = args.hiddenSize
        self.nHeads = args.attentionHeads
        self.nKVHeads = args.kvHeads
        self.headDim = args.headDimensions
        self.scale = pow(Float(headDim), -0.5)

        self._wq.wrappedValue = Linear(dim, nHeads * headDim, bias: false)
        self._wk.wrappedValue = Linear(dim, nKVHeads * headDim, bias: false)
        self._wv.wrappedValue = Linear(dim, nKVHeads * headDim, bias: false)
        self._wo.wrappedValue = Linear(nHeads * headDim, dim, bias: false)

        self.rope = RoPE(
            dimensions: headDim, traditional: args.ropeTraditional, base: args.ropeTheta)
    }

    public func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = wq(x)
        var keys = wk(x)
        var values = wv(x)

        // Prepare the queries, keys and values for the attention computation
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

        return wo(output)
    }
}

class GemmaMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear

    public init(dimensions: Int, hiddenDimensions: Int) {
        self._gate.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        self._down.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        self._up.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(gelu(gate(x)) * up(x))
    }
}

class GemmaTransformerBlock: Module {
    @ModuleInfo(key: "self_attn") var attention: GemmaAttention
    let mlp: GemmaMLP

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: Gemma.RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: Gemma.RMSNorm

    public init(_ args: GemmaConfiguration) {
        self._attention.wrappedValue = GemmaAttention(args)
        self.mlp = GemmaMLP(dimensions: args.hiddenSize, hiddenDimensions: args.intermediateSize)
        self._inputLayerNorm.wrappedValue = Gemma.RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = Gemma.RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    public func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        var r = attention(inputLayerNorm(x), mask: mask, cache: cache)
        let h = x + r
        r = mlp(postAttentionLayerNorm(h))
        return h + r
    }
}

public class GemmaModelInner: Module {
    let args: GemmaConfiguration
    let vocabularySize: Int
    let numHiddenLayers: Int

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    fileprivate let layers: [GemmaTransformerBlock]
    fileprivate let norm: Gemma.RMSNorm

    public init(_ args: GemmaConfiguration) {
        self.args = args
        self.vocabularySize = args.vocabularySize
        self.numHiddenLayers = args.hiddenLayers

        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)

        self.layers = (0 ..< args.hiddenLayers)
            .map { _ in
                GemmaTransformerBlock(args)
            }
        self.norm = Gemma.RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embedTokens(inputs)
        h = h * pow(Float(args.hiddenSize), 0.5)

        let mask = createAttentionMask(h: h, cache: cache?.first)

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }

        return norm(h)
    }
}

public class GemmaModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    let modelType: String
    public let model: GemmaModelInner

    public init(_ args: GemmaConfiguration) {
        self.modelType = args.modelType
        self.vocabularySize = args.vocabularySize
        self.kvHeads = Array(repeating: args.kvHeads, count: args.hiddenLayers)
        self.model = GemmaModelInner(args)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let out = model(inputs, cache: cache)
        return model.embedTokens.asLinear(out)
    }

    public func messageGenerator(tokenizer: any Tokenizer) -> any MessageGenerator {
        NoSystemMessageGenerator()
    }
}

public struct GemmaConfiguration: Codable, Sendable {
    var modelType: String
    var hiddenSize: Int
    var hiddenLayers: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var headDimensions: Int
    var rmsNormEps: Float
    var vocabularySize: Int
    var kvHeads: Int
    private let _ropeTheta: Float?
    public var ropeTheta: Float { _ropeTheta ?? 10_000 }
    private let _ropeTraditional: Bool?
    public var ropeTraditional: Bool { _ropeTraditional ?? false }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case headDimensions = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case kvHeads = "num_key_value_heads"
        case _ropeTheta = "rope_theta"
        case _ropeTraditional = "rope_traditional"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        modelType = try container.decode(String.self, forKey: .modelType)
        hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        headDimensions = try container.decode(Int.self, forKey: .headDimensions)
        rmsNormEps = try container.decode(Float.self, forKey: .rmsNormEps)
        vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)
        kvHeads = try container.decode(Int.self, forKey: .kvHeads)
        _ropeTheta = try container.decodeIfPresent(Float.self, forKey: ._ropeTheta)
        _ropeTraditional = try container.decodeIfPresent(Bool.self, forKey: ._ropeTraditional)

        try validateDecodedFields(container: container)
    }

    private func validateDecodedFields(container: KeyedDecodingContainer<CodingKeys>) throws {
        try validatePositive(hiddenSize, key: .hiddenSize, in: container)
        try validatePositive(hiddenLayers, key: .hiddenLayers, in: container)
        try validatePositive(intermediateSize, key: .intermediateSize, in: container)
        try validatePositive(attentionHeads, key: .attentionHeads, in: container)
        try validatePositive(headDimensions, key: .headDimensions, in: container)
        try validatePositive(rmsNormEps, key: .rmsNormEps, in: container)
        try validatePositive(vocabularySize, key: .vocabularySize, in: container)
        try validatePositive(kvHeads, key: .kvHeads, in: container)
        if let _ropeTheta {
            try validatePositive(_ropeTheta, key: ._ropeTheta, in: container)
        }

        guard hiddenSize == attentionHeads * headDimensions else {
            throw DecodingError.dataCorruptedError(
                forKey: .hiddenSize,
                in: container,
                debugDescription: "Gemma hidden_size must equal num_attention_heads * head_dim.")
        }
        guard attentionHeads % kvHeads == 0 else {
            throw DecodingError.dataCorruptedError(
                forKey: .kvHeads,
                in: container,
                debugDescription: "Gemma num_attention_heads must be divisible by num_key_value_heads.")
        }
    }

    private func validatePositive(
        _ value: Int,
        key: CodingKeys,
        in container: KeyedDecodingContainer<CodingKeys>
    ) throws {
        guard value > 0 else {
            throw DecodingError.dataCorruptedError(
                forKey: key,
                in: container,
                debugDescription: "Gemma \(key.rawValue) must be > 0.")
        }
    }

    private func validatePositive(
        _ value: Float,
        key: CodingKeys,
        in container: KeyedDecodingContainer<CodingKeys>
    ) throws {
        guard value.isFinite, value > 0 else {
            throw DecodingError.dataCorruptedError(
                forKey: key,
                in: container,
                debugDescription: "Gemma \(key.rawValue) must be finite and > 0.")
        }
    }
}

// MARK: - LoRA

extension GemmaModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
