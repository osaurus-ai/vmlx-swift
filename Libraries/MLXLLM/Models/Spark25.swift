// Copyright 2026 The XHToken team and the HuggingFace Inc. team.
// SPDX-License-Identifier: Apache-2.0
// Swift port of XHToken/Spark-X2.5-4B modeling_spark.py.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

public struct Spark25Configuration: Codable, Sendable {
    enum LayerType: String, Codable, Sendable {
        case full = "full_attention"
        case sliding = "sliding_attention"
    }

    struct Rotary: Codable, Sendable {
        let theta: Float
        let partialFactor: Float
        enum CodingKeys: String, CodingKey {
            case theta = "rope_theta"
            case partialFactor = "partial_rotary_factor"
        }
    }

    let hiddenSize: Int
    let hiddenLayers: Int
    let intermediateSize: Int
    let attentionHeads: Int
    let kvHeads: Int
    let headDim: Int
    let vocabularySize: Int
    let rmsNormEps: Float
    let hiddenAct: String
    let gateAct: String
    let headwiseGate: Bool
    let attentionBias: Bool
    let mlpBias: Bool
    let tieWordEmbeddings: Bool
    let slidingWindow: Int?
    let layerTypes: [LayerType]
    let ropeParameters: [String: Rotary]

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case vocabularySize = "vocab_size"
        case rmsNormEps = "rms_norm_eps"
        case hiddenAct = "hidden_act"
        case gateAct = "gate_attn_act_mode"
        case headwiseGate = "headwise_attn_output_gate"
        case attentionBias = "attention_bias"
        case mlpBias = "mlp_bias"
        case tieWordEmbeddings = "tie_word_embeddings"
        case slidingWindow = "sliding_window"
        case layerTypes = "layer_types"
        case ropeParameters = "rope_parameters"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        attentionHeads = try c.decode(Int.self, forKey: .attentionHeads)
        kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? attentionHeads
        headDim = try c.decode(Int.self, forKey: .headDim)
        vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        hiddenAct = try c.decodeIfPresent(String.self, forKey: .hiddenAct) ?? "gelu"
        gateAct = try c.decodeIfPresent(String.self, forKey: .gateAct) ?? "sigmoid"
        headwiseGate = try c.decodeIfPresent(Bool.self, forKey: .headwiseGate) ?? false
        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        mlpBias = try c.decodeIfPresent(Bool.self, forKey: .mlpBias) ?? false
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        slidingWindow = try c.decodeIfPresent(Int.self, forKey: .slidingWindow)
        layerTypes =
            try c.decodeIfPresent([LayerType].self, forKey: .layerTypes)
            ?? Array(repeating: .full, count: Swift.max(0, hiddenLayers))
        ropeParameters =
            try c.decodeIfPresent([String: Rotary].self, forKey: .ropeParameters) ?? [:]

        guard hiddenSize > 0, hiddenLayers > 0, intermediateSize > 0,
            vocabularySize > 0, attentionHeads > 0, kvHeads > 0, headDim > 0,
            attentionHeads % kvHeads == 0, layerTypes.count == hiddenLayers,
            rmsNormEps.isFinite, rmsNormEps > 0, hiddenAct == "gelu",
            ["sigmoid", "silu"].contains(gateAct),
            !layerTypes.contains(.sliding) || (slidingWindow ?? 0) > 0
        else {
            throw DecodingError.dataCorrupted(
                .init(
                    codingPath: decoder.codingPath,
                    debugDescription: "Invalid Spark2.5 geometry or activation contract"))
        }
        for type in layerTypes {
            let r = rotary(for: type)
            let dims = Float(headDim) * r.partialFactor
            guard r.theta.isFinite, r.theta > 0, dims.isFinite,
                dims >= 2, dims <= Float(headDim), dims.rounded() == dims, Int(dims) % 2 == 0
            else {
                throw DecodingError.dataCorrupted(
                    .init(
                        codingPath: decoder.codingPath,
                        debugDescription: "Invalid Spark2.5 rotary dimensions or base"))
            }
        }
    }

    func rotary(for type: LayerType) -> Rotary {
        ropeParameters[type.rawValue] ?? Rotary(theta: 10_000, partialFactor: 1)
    }
}

final class Spark25Attention: Module {
    @ModuleInfo(key: "q_k_v_proj") var qkv: Linear
    @ModuleInfo(key: "out_proj") var output: Linear
    @ModuleInfo(key: "g_proj") var gate: Linear?
    let config: Spark25Configuration
    let rope: RoPE
    let scale: Float

    init(_ config: Spark25Configuration, type: Spark25Configuration.LayerType) {
        self.config = config
        let q = config.attentionHeads * config.headDim
        let kv = config.kvHeads * config.headDim
        _qkv.wrappedValue = Linear(config.hiddenSize, q + 2 * kv, bias: config.attentionBias)
        _output.wrappedValue = Linear(q, config.hiddenSize, bias: config.attentionBias)
        if config.headwiseGate {
            _gate.wrappedValue = Linear(
                config.hiddenSize, config.attentionHeads, bias: config.attentionBias)
        }
        let rotary = config.rotary(for: type)
        rope = RoPE(
            dimensions: Int(Float(config.headDim) * rotary.partialFactor), traditional: false,
            base: rotary.theta)
        scale = pow(Float(config.headDim), -0.5)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (b, n) = (x.dim(0), x.dim(1))
        let qDim = config.attentionHeads * config.headDim
        let kvDim = config.kvHeads * config.headDim
        let fused = qkv(x)
        var q = fused[.ellipsis, ..<qDim].reshaped(b, n, config.attentionHeads, config.headDim)
            .transposed(0, 2, 1, 3)
        var k = fused[.ellipsis, qDim ..< (qDim + kvDim)].reshaped(
            b, n, config.kvHeads, config.headDim
        ).transposed(0, 2, 1, 3)
        let v = fused[.ellipsis, (qDim + kvDim)...].reshaped(b, n, config.kvHeads, config.headDim)
            .transposed(0, 2, 1, 3)
        q = applyRotaryPosition(rope, to: q, cache: cache)
        k = applyRotaryPosition(rope, to: k, cache: cache)
        var attended = attentionWithCacheUpdate(
            queries: q, keys: k, values: v, cache: cache, scale: scale, mask: mask)
        if let gate {
            let score = gate(x).reshaped(b, n, config.attentionHeads, 1).transposed(0, 2, 1, 3)
                .asType(.float32)
            let weight = config.gateAct == "sigmoid" ? sigmoid(score) : silu(score)
            attended = attended * weight.asType(attended.dtype)
        }
        return output(attended.transposed(0, 2, 1, 3).reshaped(b, n, qDim))
    }
}

final class Spark25MLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    init(_ c: Spark25Configuration) {
        _gate.wrappedValue = Linear(c.hiddenSize, c.intermediateSize, bias: c.mlpBias)
        _up.wrappedValue = Linear(c.hiddenSize, c.intermediateSize, bias: c.mlpBias)
        _down.wrappedValue = Linear(c.intermediateSize, c.hiddenSize, bias: c.mlpBias)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { down(gelu(gate(x)) * up(x)) }
}

final class Spark25DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: Spark25Attention
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postNorm: RMSNorm
    let mlp: Spark25MLP
    init(_ c: Spark25Configuration, type: Spark25Configuration.LayerType) {
        _attention.wrappedValue = Spark25Attention(c, type: type)
        _inputNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
        _postNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
        mlp = Spark25MLP(c)
    }
    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        // Residuals accumulate in fp32; projection input dtype comes from a
        // floating norm, never the packed uint32 weight of a quantized Linear.
        let h = x + attention(inputNorm(x).asType(inputNorm.weight.dtype), mask: mask, cache: cache)
        return h + mlp(postNorm(h).asType(postNorm.weight.dtype))
    }
}

final class Spark25ModelInner: Module {
    @ModuleInfo(key: "embedding") var embedding: Embedding
    let layers: [Spark25DecoderLayer]
    let norm: RMSNorm
    let config: Spark25Configuration
    init(_ c: Spark25Configuration) {
        config = c
        _embedding.wrappedValue = Embedding(
            embeddingCount: c.vocabularySize, dimensions: c.hiddenSize)
        layers = c.layerTypes.map { Spark25DecoderLayer(c, type: $0) }
        norm = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
    }
    func callAsFunction(_ tokens: MLXArray, cache: [KVCache]?) -> MLXArray {
        let embedded = embedding(tokens)
        var h = embedded.asType(.float32)
        // Compute masks before updating any layer's cache. Each type owns a
        // different cache extent; a full-layer offset cannot size an SWA mask.
        let fullIndex = config.layerTypes.firstIndex(of: .full)
        let slidingIndex = config.layerTypes.firstIndex(of: .sliding)
        let full = createAttentionMask(h: h, cache: fullIndex.flatMap { cache?[$0] })
        let sliding = createAttentionMask(
            h: h, cache: slidingIndex.flatMap { cache?[$0] }, windowSize: config.slidingWindow)
        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: config.layerTypes[i] == .full ? full : sliding, cache: cache?[i])
        }
        return norm(h).asType(embedded.dtype)
    }
}

public final class Spark25Model: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]
    let model: Spark25ModelInner
    let config: Spark25Configuration
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ config: Spark25Configuration) {
        self.config = config
        vocabularySize = config.vocabularySize
        kvHeads = Array(repeating: config.kvHeads, count: config.hiddenLayers)
        model = Spark25ModelInner(config)
        if !config.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
        }
    }
    public var loraLayers: [Module] { model.layers }
    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let h = model(inputs, cache: cache)
        return lmHead?(h) ?? model.embedding.asLinear(h)
    }
    public func newCache(parameters: GenerateParameters? = nil) -> [KVCache] {
        config.layerTypes.map { type in
            if type == .sliding, let window = config.slidingWindow {
                return RotatingKVCache(maxSize: window, keep: 0)
            }
            return KVCacheSimple()
        }
    }
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var result = weights
        if config.tieWordEmbeddings { result["lm_head.weight"] = nil }
        return result
    }
}
