//
//  Qwen3MoE.swift
//  LLM
//
//  Created by John Mai on 2025/4/30.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3_moe.py

class Qwen3MoEAttention: Module {
    let args: Qwen3MoEConfiguration
    let scale: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let rope: RoPE

    public init(_ args: Qwen3MoEConfiguration, layerIdx: Int) {
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

class Qwen3MoEMLP: Module, UnaryLayer {
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

class Qwen3MoESparseMoeBlock: Module, UnaryLayer {
    let layerIdx: Int
    let numExperts: Int
    let topK: Int
    let normTopkProb: Bool

    @ModuleInfo(key: "gate") var gate: Linear
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU

    init(_ args: Qwen3MoEConfiguration, layerIdx: Int) {
        self.layerIdx = layerIdx
        self.numExperts = args.numExperts
        self.topK = args.numExpertsPerToken
        self.normTopkProb = args.normTopkProb

        _gate.wrappedValue = Linear(args.hiddenSize, numExperts, bias: false)
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: args.hiddenSize, hiddenDims: args.moeIntermediateSize, numExperts: numExperts
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let gates = gate(x)
        let softGates = MLX.softmax(gates, axis: -1, precise: true)

        let k = topK
        let inds = MLX.argPartition(-gates, kth: k - 1, axis: -1)[.ellipsis, ..<k]
        JangPressCanonicalExpertAdvisor.shared.observe(layer: layerIdx, indices: inds)
        var scores = MLX.takeAlong(softGates, inds, axis: -1)

        if normTopkProb {
            scores = scores / MLX.sum(scores, axis: -1, keepDims: true)
        }

        let y = switchMLP(x, inds)
        return (y * scores[.ellipsis, .newAxis]).sum(axis: -2)
    }
}

class Qwen3MoeDecoderLayer: Module {
    let args: Qwen3MoEConfiguration
    let layerIdx: Int

    @ModuleInfo(key: "self_attn") var selfAttn: Qwen3MoEAttention
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    fileprivate let mlp: UnaryLayer

    init(_ args: Qwen3MoEConfiguration, layerIdx: Int) {
        self.args = args
        self.layerIdx = layerIdx

        _selfAttn.wrappedValue = Qwen3MoEAttention(args, layerIdx: layerIdx)
        _inputLayerNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)

        if !args.mlpOnlyLayers.contains(layerIdx),
            args.numExperts > 0, (layerIdx + 1) % args.decoderSparseStep == 0
        {
            self.mlp = Qwen3MoESparseMoeBlock(args, layerIdx: layerIdx)
        } else {
            self.mlp = Qwen3MoEMLP(
                dimensions: args.hiddenSize, hiddenDimensions: args.intermediateSize)
        }
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        var r = selfAttn(inputLayerNorm(x), mask: mask, cache: cache)
        let h = x + r
        r = mlp(postAttentionLayerNorm(h))
        let out = h + r
        return out
    }
}

public class Qwen3MoEModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    fileprivate let layers: [Qwen3MoeDecoderLayer]
    let norm: RMSNorm
    let args: Qwen3MoEConfiguration

    init(_ args: Qwen3MoEConfiguration) {
        self.args = args

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)

        self.layers = (0 ..< args.hiddenLayers)
            .map { i in
                Qwen3MoeDecoderLayer(args, layerIdx: i)
            }
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

public class Qwen3MoEModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: Qwen3MoEModelInner
    let configuration: Qwen3MoEConfiguration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ args: Qwen3MoEConfiguration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
        self.model = Qwen3MoEModelInner(args)

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
        var sanitizedWeights = weights

        if configuration.tieWordEmbeddings {
            sanitizedWeights["lm_head.weight"] = nil
        }

        if sanitizedWeights["model.layers.0.mlp.experts.0.up_proj.weight"] == nil {
            return sanitizedWeights
        }

        for l in 0 ..< configuration.hiddenLayers {
            let prefix = "model.layers.\(l)"
            for n in ["up_proj", "down_proj", "gate_proj"] {
                if sanitizedWeights["\(prefix).mlp.experts.0.\(n).weight"] != nil {
                    let toJoin = (0 ..< configuration.numExperts).map { e in
                        sanitizedWeights.removeValue(
                            forKey: "\(prefix).mlp.experts.\(e).\(n).weight")!
                    }
                    sanitizedWeights["\(prefix).mlp.switch_mlp.\(n).weight"] = MLX.stacked(toJoin)
                }
            }
        }

        return sanitizedWeights
    }
}

public struct Qwen3MoEConfiguration: Codable, Sendable {
    var modelType: String = "qwen3_moe"
    var hiddenSize: Int
    var hiddenLayers: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var numExperts: Int
    var numExpertsPerToken: Int
    var decoderSparseStep: Int
    var mlpOnlyLayers: [Int]
    var moeIntermediateSize: Int
    var rmsNormEps: Float
    var vocabularySize: Int
    var kvHeads: Int
    var headDim: Int
    var ropeTheta: Float = 1_000_000
    var tieWordEmbeddings: Bool = false
    var maxPositionEmbeddings: Int = 32768
    var normTopkProb: Bool = false
    var ropeScaling: [String: StringOrNumber]? = nil

    var ropeScale: Float {
        guard let ropeScaling, ropeScaling["type"] == .string("linear"),
            let factor = ropeScaling["factor"]?.asFloat()
        else {
            return 1
        }
        return 1 / factor
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case numExperts = "num_experts"
        case numExpertsPerToken = "num_experts_per_tok"
        case decoderSparseStep = "decoder_sparse_step"
        case mlpOnlyLayers = "mlp_only_layers"
        case moeIntermediateSize = "moe_intermediate_size"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case ropeTheta = "rope_theta"
        case tieWordEmbeddings = "tie_word_embeddings"
        case maxPositionEmbeddings = "max_position_embeddings"
        case normTopkProb = "norm_topk_prob"
        case ropeScaling = "rope_scaling"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        self.modelType =
            try container.decodeIfPresent(String.self, forKey: .modelType) ?? "qwen3_moe"
        self.hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        self.hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        self.intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        self.attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        self.numExperts = try container.decode(Int.self, forKey: .numExperts)
        self.numExpertsPerToken = RuntimeMoETopKOverride.effectiveTopK(
            currentTopK: try container.decode(Int.self, forKey: .numExpertsPerToken),
            modelType: modelType,
            field: CodingKeys.numExpertsPerToken.rawValue)
        self.decoderSparseStep = try container.decode(Int.self, forKey: .decoderSparseStep)
        self.mlpOnlyLayers = try container.decode([Int].self, forKey: .mlpOnlyLayers)
        self.moeIntermediateSize = try container.decode(Int.self, forKey: .moeIntermediateSize)
        self.rmsNormEps = try container.decode(Float.self, forKey: .rmsNormEps)
        self.vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)
        self.kvHeads = try container.decode(Int.self, forKey: .kvHeads)
        self.headDim = try container.decode(Int.self, forKey: .headDim)
        self.ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1_000_000
        self.tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        self.maxPositionEmbeddings =
            try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 32768
        self.normTopkProb = try container.decodeIfPresent(Bool.self, forKey: .normTopkProb) ?? false
        self.ropeScaling = try container.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeScaling)

        try Self.validatePositive(hiddenSize, key: .hiddenSize, in: container)
        try Self.validatePositive(hiddenLayers, key: .hiddenLayers, in: container)
        try Self.validatePositive(intermediateSize, key: .intermediateSize, in: container)
        try Self.validatePositive(attentionHeads, key: .attentionHeads, in: container)
        try Self.validatePositive(numExperts, key: .numExperts, in: container)
        try Self.validatePositive(numExpertsPerToken, key: .numExpertsPerToken, in: container)
        try Self.validatePositive(decoderSparseStep, key: .decoderSparseStep, in: container)
        try Self.validatePositive(moeIntermediateSize, key: .moeIntermediateSize, in: container)
        try Self.validatePositive(rmsNormEps, key: .rmsNormEps, in: container)
        try Self.validatePositive(vocabularySize, key: .vocabularySize, in: container)
        try Self.validatePositive(kvHeads, key: .kvHeads, in: container)
        try Self.validatePositive(headDim, key: .headDim, in: container)
        try Self.validatePositive(ropeTheta, key: .ropeTheta, in: container)
        try Self.validatePositive(maxPositionEmbeddings, key: .maxPositionEmbeddings, in: container)
        guard hiddenSize == attentionHeads * headDim else {
            throw DecodingError.dataCorruptedError(
                forKey: .hiddenSize,
                in: container,
                debugDescription:
                    "Qwen3MoE hidden_size must equal num_attention_heads * head_dim.")
        }
        guard attentionHeads % kvHeads == 0 else {
            throw DecodingError.dataCorruptedError(
                forKey: .kvHeads,
                in: container,
                debugDescription:
                    "Qwen3MoE num_attention_heads must be divisible by num_key_value_heads.")
        }
        guard numExpertsPerToken <= numExperts else {
            throw DecodingError.dataCorruptedError(
                forKey: .numExpertsPerToken,
                in: container,
                debugDescription:
                    "Qwen3MoE num_experts_per_tok must be less than or equal to num_experts.")
        }
        guard mlpOnlyLayers.allSatisfy({ $0 >= 0 && $0 < hiddenLayers }) else {
            throw DecodingError.dataCorruptedError(
                forKey: .mlpOnlyLayers,
                in: container,
                debugDescription:
                    "Qwen3MoE mlp_only_layers entries must be valid layer indexes.")
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
                debugDescription: "Qwen3MoE rope_scaling.factor must be a positive float.")
        }
    }
}

// MARK: - LoRA

extension Qwen3MoEModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
