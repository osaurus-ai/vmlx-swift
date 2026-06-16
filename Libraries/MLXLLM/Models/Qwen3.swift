//
//  Qwen3.swift
//  LLM
//
//  Created by John Mai on 2025/4/28.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3.py

class Qwen3Attention: Module {
    let args: Qwen3Configuration
    let scale: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let rope: RoPE

    public init(_ args: Qwen3Configuration) {
        self.args = args

        let dim = args.hiddenSize
        let heads = args.attentionHeads
        let kvHeads = args.kvHeads

        let headDim = args.headDim
        self.scale = pow(Float(headDim), -0.5)

        _wq.wrappedValue = Linear(dim, heads * headDim, bias: false)
        _wk.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
        _wv.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
        _wo.wrappedValue = Linear(heads * headDim, dim, bias: false)

        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)

        self.rope = RoPE(
            dimensions: headDim, traditional: false, base: args.ropeTheta,
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
        queries = qNorm(queries.reshaped(B, L, args.attentionHeads, -1)).transposed(0, 2, 1, 3)
        keys = kNorm(keys.reshaped(B, L, args.kvHeads, -1)).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)

        // Apply RoPE positioning
        queries = applyRotaryPosition(rope, to: queries, cache: cache)
        keys = applyRotaryPosition(rope, to: keys, cache: cache)

        // Use the automatic attention router that handles both quantized and regular caches
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

class Qwen3MLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear

    public init(dimensions: Int, hiddenDimensions: Int) {
        _gate.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        _down.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        _up.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let g = silu(gate(x))
        let u = up(x)
        let product = g * u
        return down(product)
    }
}

class Qwen3TransformerBlock: Module {
    @ModuleInfo(key: "self_attn") var attention: Qwen3Attention
    let mlp: Qwen3MLP

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    public init(_ args: Qwen3Configuration) {
        _attention.wrappedValue = Qwen3Attention(args)
        self.mlp = Qwen3MLP(dimensions: args.hiddenSize, hiddenDimensions: args.intermediateSize)
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

public class Qwen3ModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    fileprivate let layers: [Qwen3TransformerBlock]
    let norm: RMSNorm

    public init(_ args: Qwen3Configuration) {
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)

        self.layers = (0 ..< args.hiddenLayers)
            .map { _ in
                Qwen3TransformerBlock(args)
            }
        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        let (h, _) = callAsFunctionCapturing(
            inputs, cache: cache, captureLayerIDs: [])
        return h
    }

    /// Forward with optional per-block hidden-state capture. When
    /// `captureLayerIDs` is empty this is byte-identical to
    /// ``callAsFunction(_:cache:)``.
    public func callAsFunctionCapturing(
        _ inputs: MLXArray,
        cache: [KVCache]? = nil,
        captureLayerIDs: Set<Int>
    ) -> (MLXArray, [Int: MLXArray]) {
        var h = embedTokens(inputs)

        let mask = createAttentionMask(h: h, cache: cache?.first)

        var captured: [Int: MLXArray] = [:]
        captured.reserveCapacity(captureLayerIDs.count)

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
            if captureLayerIDs.contains(i) {
                captured[i] = h
            }
        }

        return (norm(h), captured)
    }
}

public class Qwen3Model: Module, LLMModel, KVCacheDimensionProvider, HiddenStateCaptureModel, TokenEmbedderModel {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: Qwen3ModelInner
    let configuration: Qwen3Configuration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ args: Qwen3Configuration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
        self.model = Qwen3ModelInner(args)

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

    /// `HiddenStateCaptureModel` conformance — exposes per-block hidden
    /// states so DFlash drafters can read the target's layer outputs at
    /// specified indices. Empty `captureLayerIDs` yields byte-identical
    /// logits to the plain forward.
    public func callAsFunction(
        _ inputs: MLXArray,
        cache: [KVCache]?,
        captureLayerIDs: Set<Int>
    ) -> (logits: MLXArray, capturedHiddenStates: [Int: MLXArray]) {
        let (finalHidden, captured) = model.callAsFunctionCapturing(
            inputs, cache: cache, captureLayerIDs: captureLayerIDs)
        let logits: MLXArray
        if let lmHead {
            logits = lmHead(finalHidden)
        } else {
            logits = model.embedTokens.asLinear(finalHidden)
        }
        return (logits, captured)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var weights = weights

        if configuration.tieWordEmbeddings {
            weights["lm_head.weight"] = nil
        }

        return weights
    }

    // MARK: - TokenEmbedderModel

    /// Embed token IDs via the target's shared embedding table. Matches
    /// what `Qwen3ModelInner.callAsFunction` does at its first line.
    public func embed(_ tokenIds: MLXArray) -> MLXArray {
        model.embedTokens(tokenIds)
    }

    /// Project hidden states through the target's LM head. For tied
    /// word embeddings the head is the transposed embedding. Mirrors
    /// the projection path in `callAsFunction(_:cache:)` exactly.
    public func projectToLogits(_ hidden: MLXArray) -> MLXArray {
        if let lmHead {
            return lmHead(hidden)
        }
        return model.embedTokens.asLinear(hidden)
    }
}

public struct Qwen3Configuration: Codable, Sendable {
    var hiddenSize: Int
    var hiddenLayers: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var rmsNormEps: Float
    var vocabularySize: Int
    var kvHeads: Int
    var ropeTheta: Float = 1_000_000
    var headDim: Int
    var ropeScaling: [String: StringOrNumber]? = nil
    var tieWordEmbeddings = false
    var maxPositionEmbeddings: Int = 32768

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
        case headDim = "head_dim"
        case ropeScaling = "rope_scaling"
        case tieWordEmbeddings = "tie_word_embeddings"
        case maxPositionEmbeddings = "max_position_embeddings"
    }

    public init(from decoder: Decoder) throws {
        // custom implementation to handle optional keys with required values
        let container: KeyedDecodingContainer<Qwen3Configuration.CodingKeys> =
            try decoder.container(
                keyedBy: Qwen3Configuration.CodingKeys.self)

        self.hiddenSize = try container.decode(
            Int.self, forKey: Qwen3Configuration.CodingKeys.hiddenSize)
        self.hiddenLayers = try container.decode(
            Int.self, forKey: Qwen3Configuration.CodingKeys.hiddenLayers)
        self.intermediateSize = try container.decode(
            Int.self, forKey: Qwen3Configuration.CodingKeys.intermediateSize)
        self.attentionHeads = try container.decode(
            Int.self, forKey: Qwen3Configuration.CodingKeys.attentionHeads)
        self.rmsNormEps = try container.decode(
            Float.self, forKey: Qwen3Configuration.CodingKeys.rmsNormEps)
        self.vocabularySize = try container.decode(
            Int.self, forKey: Qwen3Configuration.CodingKeys.vocabularySize)
        self.kvHeads = try container.decode(Int.self, forKey: Qwen3Configuration.CodingKeys.kvHeads)
        self.ropeTheta =
            try container.decodeIfPresent(
                Float.self, forKey: Qwen3Configuration.CodingKeys.ropeTheta)
            ?? 1_000_000
        self.headDim = try container.decode(
            Int.self, forKey: Qwen3Configuration.CodingKeys.headDim)
        self.ropeScaling = try container.decodeIfPresent(
            [String: StringOrNumber].self, forKey: Qwen3Configuration.CodingKeys.ropeScaling)
        self.tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        self.maxPositionEmbeddings =
            try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 32768

        try Self.validatePositive(hiddenSize, key: .hiddenSize, in: container)
        try Self.validatePositive(hiddenLayers, key: .hiddenLayers, in: container)
        try Self.validatePositive(intermediateSize, key: .intermediateSize, in: container)
        try Self.validatePositive(attentionHeads, key: .attentionHeads, in: container)
        try Self.validatePositive(rmsNormEps, key: .rmsNormEps, in: container)
        try Self.validatePositive(vocabularySize, key: .vocabularySize, in: container)
        try Self.validatePositive(kvHeads, key: .kvHeads, in: container)
        try Self.validatePositive(ropeTheta, key: .ropeTheta, in: container)
        try Self.validatePositive(headDim, key: .headDim, in: container)
        try Self.validatePositive(maxPositionEmbeddings, key: .maxPositionEmbeddings, in: container)
        guard hiddenSize == attentionHeads * headDim else {
            throw DecodingError.dataCorruptedError(
                forKey: .hiddenSize,
                in: container,
                debugDescription:
                    "Qwen3 hidden_size must equal num_attention_heads * head_dim.")
        }
        guard attentionHeads % kvHeads == 0 else {
            throw DecodingError.dataCorruptedError(
                forKey: .kvHeads,
                in: container,
                debugDescription:
                    "Qwen3 num_attention_heads must be divisible by num_key_value_heads.")
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
                debugDescription: "Qwen3 rope_scaling.factor must be a positive float.")
        }
    }
}

// MARK: - LoRA

extension Qwen3Model: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
