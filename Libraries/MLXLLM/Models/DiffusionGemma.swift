// Copyright © 2026 Jinho Jang (eric@jangq.ai)
//
// DiffusionGemma (google/diffusiongemma-26B-A4B-it) — block-diffusion text
// generation. 30-layer Gemma4-style MoE transformer (128 experts, top-8,
// parallel dense MLP + routed experts) with two forward modes sharing one
// set of weights:
//
//   - encoder: causal forward over committed tokens (prompt, then each
//     finalized canvas); writes the KV cache. Uses per-layer encoder
//     scalars (`model.encoder.language_model.layers.N.layer_scalar`).
//   - decoder: bidirectional forward over a 256-token noisy canvas; reads
//     the encoder cache without mutating it, with self-conditioning soft
//     embeddings mixed into the canvas embedding.
//
// Generation is driven by BlockDiffusionTokenIterator (MLXLMCommon) — this
// model must NEVER be routed through autoregressive next-token decode, so
// `prepare` throws.
//
// Python references:
//   mlx_vlm/models/diffusion_gemma/language.py
//   transformers/models/diffusion_gemma/modeling_diffusion_gemma.py

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Configuration

public struct DiffusionGemmaTextConfiguration: Codable, Sendable {
    let modelType: String
    let hiddenSize: Int
    let numHiddenLayers: Int
    let numAttentionHeads: Int
    let numKeyValueHeads: Int
    let numGlobalKeyValueHeads: Int?
    let headDim: Int
    let globalHeadDim: Int
    let intermediateSize: Int
    let moeIntermediateSize: Int
    let numExperts: Int
    let topKExperts: Int
    let vocabSize: Int
    let rmsNormEps: Float
    let slidingWindow: Int
    let layerTypes: [String]
    let finalLogitSoftcapping: Float?
    let attentionBias: Bool
    let padTokenId: Int
    let ropeParameters: [String: [String: StringOrNumber]]

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case numGlobalKeyValueHeads = "num_global_key_value_heads"
        case headDim = "head_dim"
        case globalHeadDim = "global_head_dim"
        case intermediateSize = "intermediate_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case numExperts = "num_experts"
        case topKExperts = "top_k_experts"
        case vocabSize = "vocab_size"
        case rmsNormEps = "rms_norm_eps"
        case slidingWindow = "sliding_window"
        case layerTypes = "layer_types"
        case finalLogitSoftcapping = "final_logit_softcapping"
        case attentionBias = "attention_bias"
        case padTokenId = "pad_token_id"
        case ropeParameters = "rope_parameters"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        modelType =
            try container.decodeIfPresent(String.self, forKey: .modelType)
            ?? "diffusion_gemma_text"
        hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 2816
        numHiddenLayers =
            try container.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 30
        numAttentionHeads =
            try container.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 16
        numKeyValueHeads =
            try container.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? 8
        numGlobalKeyValueHeads =
            try container.decodeIfPresent(Int.self, forKey: .numGlobalKeyValueHeads)
        headDim = try container.decodeIfPresent(Int.self, forKey: .headDim) ?? 256
        globalHeadDim = try container.decodeIfPresent(Int.self, forKey: .globalHeadDim) ?? 512
        intermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 2112
        moeIntermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .moeIntermediateSize) ?? 704
        numExperts = try container.decodeIfPresent(Int.self, forKey: .numExperts) ?? 128
        topKExperts = try container.decodeIfPresent(Int.self, forKey: .topKExperts) ?? 8
        vocabSize = try container.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 262144
        rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        slidingWindow = try container.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 1024
        let decodedLayerTypes =
            try container.decodeIfPresent([String].self, forKey: .layerTypes) ?? []
        if decodedLayerTypes.isEmpty {
            // Reference default: 5×sliding + 1×full, repeated; last layer full.
            let pattern =
                Array(repeating: "sliding_attention", count: 5) + ["full_attention"]
            var generated = (0 ..< numHiddenLayers).map { pattern[$0 % pattern.count] }
            if generated.last != "full_attention" {
                generated[generated.count - 1] = "full_attention"
            }
            layerTypes = generated
        } else {
            layerTypes = decodedLayerTypes
        }
        finalLogitSoftcapping =
            try container.decodeIfPresent(Float.self, forKey: .finalLogitSoftcapping)
        attentionBias =
            try container.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        padTokenId = try container.decodeIfPresent(Int.self, forKey: .padTokenId) ?? 0
        ropeParameters =
            try container.decodeIfPresent(
                [String: [String: StringOrNumber]].self, forKey: .ropeParameters) ?? [:]
    }
}

public struct DiffusionGemmaConfiguration: Codable, Sendable {
    public let modelType: String
    public let canvasLength: Int
    public let eosTokenIds: [Int]
    public let imageTokenId: Int?
    let textConfig: DiffusionGemmaTextConfiguration

    /// Text hidden size, exposed for the MLXVLM vision wiring.
    public var textHiddenSize: Int { textConfig.hiddenSize }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case canvasLength = "canvas_length"
        case eosTokenIds = "eos_token_id"
        case imageTokenId = "image_token_id"
        case textConfig = "text_config"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        modelType =
            try container.decodeIfPresent(String.self, forKey: .modelType) ?? "diffusion_gemma"
        canvasLength = try container.decodeIfPresent(Int.self, forKey: .canvasLength) ?? 256
        if let list = try? container.decode([Int].self, forKey: .eosTokenIds) {
            eosTokenIds = list
        } else if let single = try? container.decode(Int.self, forKey: .eosTokenIds) {
            eosTokenIds = [single]
        } else {
            eosTokenIds = []
        }
        imageTokenId = try container.decodeIfPresent(Int.self, forKey: .imageTokenId)
        textConfig = try container.decode(
            DiffusionGemmaTextConfiguration.self, forKey: .textConfig)
    }
}

// MARK: - Attention

/// Which forward mode a DiffusionGemma layer runs in.
enum DiffusionGemmaForwardMode {
    /// Causal over committed tokens; writes `cache`.
    case encoder
    /// Bidirectional over the canvas; reads the encoder KV from `cache`
    /// without mutating it. Payload is the shared encoder offset used for
    /// canvas RoPE positions.
    case decoder(encoderOffset: Int)
}

class DiffusionGemmaAttention: Module {
    let nHeads: Int
    let nKVHeads: Int
    let headDim: Int
    let isSliding: Bool
    let slidingWindow: Int
    let eps: Float

    @ModuleInfo(key: "q_proj") var queryProj: Linear
    @ModuleInfo(key: "k_proj") var keyProj: Linear
    @ModuleInfo(key: "v_proj") var valueProj: Linear?
    @ModuleInfo(key: "o_proj") var outputProj: Linear
    @ModuleInfo(key: "q_norm") var queryNorm: Gemma4RMSNorm
    @ModuleInfo(key: "k_norm") var keyNorm: Gemma4RMSNorm
    // v_norm is RMSNormNoScale (no learnable weight, not in checkpoint)

    @ModuleInfo var rope: RoPELayer

    init(_ config: DiffusionGemmaTextConfiguration, layerIndex: Int) {
        let layerType =
            layerIndex < config.layerTypes.count
            ? config.layerTypes[layerIndex] : "sliding_attention"
        self.isSliding = layerType == "sliding_attention"
        self.slidingWindow = config.slidingWindow
        self.eps = config.rmsNormEps

        self.nHeads = config.numAttentionHeads
        if isSliding {
            self.nKVHeads = config.numKeyValueHeads
            self.headDim = config.headDim
        } else {
            self.nKVHeads = config.numGlobalKeyValueHeads ?? config.numKeyValueHeads
            self.headDim = config.globalHeadDim
        }

        self._queryProj.wrappedValue = Linear(
            config.hiddenSize, nHeads * headDim, bias: config.attentionBias)
        self._keyProj.wrappedValue = Linear(
            config.hiddenSize, nKVHeads * headDim, bias: config.attentionBias)
        // The checkpoint only carries v_proj for sliding layers; full
        // attention layers share K=V (value path applies RMSNormNoScale to
        // the raw key projection).
        if isSliding {
            self._valueProj.wrappedValue = Linear(
                config.hiddenSize, nKVHeads * headDim, bias: config.attentionBias)
        }
        self._outputProj.wrappedValue = Linear(
            nHeads * headDim, config.hiddenSize, bias: config.attentionBias)
        self._queryNorm.wrappedValue = Gemma4RMSNorm(
            dimensions: headDim, eps: config.rmsNormEps)
        self._keyNorm.wrappedValue = Gemma4RMSNorm(
            dimensions: headDim, eps: config.rmsNormEps)

        let layerKey = isSliding ? "sliding_attention" : "full_attention"
        let ropeParams = config.ropeParameters[layerKey] ?? [:]
        let ropeTheta =
            ropeParams["rope_theta"]?.asFloat() ?? (isSliding ? 10000.0 : 1_000_000.0)
        let partialRotaryFactor =
            ropeParams["partial_rotary_factor"]?.asFloat() ?? (isSliding ? 1.0 : 0.25)
        let ropeType: String = {
            if let typeValue = ropeParams["type"] ?? ropeParams["rope_type"],
                case .string(let s) = typeValue
            {
                return s
            }
            return "default"
        }()
        // ProportionalRoPE consumes the partial factor itself, so it gets the
        // full head dim (same convention as Gemma4Text).
        let ropeDims =
            ropeType == "proportional"
            ? headDim : max(1, Int(Float(headDim) * partialRotaryFactor))
        self.rope = initializeRope(
            dims: ropeDims, base: ropeTheta, traditional: false,
            scalingConfig: ropeParams.isEmpty ? nil : ropeParams,
            maxPositionEmbeddings: nil)

        super.init()
    }

    /// Read this layer's encoder KV in temporal order without mutating the
    /// cache. Sliding layers only expose the trailing `slidingWindow - 1`
    /// positions (the canvas occupies the final window slot).
    private func encoderKV(from cache: KVCache) -> (keys: MLXArray, values: MLXArray)? {
        let read: (keys: MLXArray, values: MLXArray)?
        switch cache {
        case let rotating as RotatingKVCache:
            read = rotating.temporallyOrderedKV()
        case let simple as KVCacheSimple:
            read = simple.readKV()
        default:
            let state = cache.state
            read = state.count == 2 ? (state[0], state[1]) : nil
        }
        guard var kv = read else { return nil }
        if isSliding {
            let window = max(slidingWindow - 1, 0)
            let length = kv.keys.dim(2)
            if window > 0, length > window {
                kv = (
                    kv.keys[.ellipsis, (length - window)..., 0...],
                    kv.values[.ellipsis, (length - window)..., 0...]
                )
            }
        }
        return kv
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?,
        mode: DiffusionGemmaForwardMode
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = queryProj(x).reshaped(B, L, nHeads, headDim)
        queries = queryNorm(queries).transposed(0, 2, 1, 3)

        var keys = keyProj(x).reshaped(B, L, nKVHeads, headDim)
        let values: MLXArray
        if let valueProj {
            values = rmsNormNoScale(
                valueProj(x).reshaped(B, L, nKVHeads, headDim), eps: eps)
        } else {
            values = rmsNormNoScale(keys, eps: eps)
        }
        keys = keyNorm(keys)

        var keysT = keys.transposed(0, 2, 1, 3)
        let valuesT = values.transposed(0, 2, 1, 3)

        let attentionKeys: MLXArray
        let attentionValues: MLXArray
        switch mode {
        case .encoder:
            let offset = cache?.offset ?? 0
            queries = rope(queries, offset: offset)
            keysT = rope(keysT, offset: offset)
            if let cache {
                (attentionKeys, attentionValues) = cache.update(keys: keysT, values: valuesT)
            } else {
                (attentionKeys, attentionValues) = (keysT, valuesT)
            }
        case .decoder(let encoderOffset):
            // Canvas q/k take absolute positions after the committed context;
            // cached encoder keys were roped at their own absolute positions
            // when written.
            queries = rope(queries, offset: encoderOffset)
            keysT = rope(keysT, offset: encoderOffset)
            if let cache, let encoder = encoderKV(from: cache) {
                attentionKeys = concatenated([encoder.keys, keysT], axis: 2)
                attentionValues = concatenated([encoder.values, valuesT], axis: 2)
            } else {
                (attentionKeys, attentionValues) = (keysT, valuesT)
            }
        }

        // Gemma4-family attention scale is 1.0 (queries are RMS-normed).
        let sdpa = MLXFast.scaledDotProductAttention(
            queries: queries, keys: attentionKeys, values: attentionValues,
            scale: 1.0, mask: mask)
        return outputProj(sdpa.transposed(0, 2, 1, 3).reshaped(B, L, -1))
    }
}

// MARK: - Router / Experts

class DiffusionGemmaRouter: Module {
    @ModuleInfo(key: "proj") var proj: Linear
    @ModuleInfo(key: "scale") var routerScale: MLXArray
    @ModuleInfo(key: "per_expert_scale") var perExpertScale: MLXArray
    // pre-norm is RMSNormNoScale folded into the scaled rmsNorm weight below

    let topK: Int
    let rootSize: Float
    let eps: Float

    init(_ config: DiffusionGemmaTextConfiguration) {
        self.topK = config.topKExperts
        self.rootSize = pow(Float(config.hiddenSize), -0.5)
        self.eps = config.rmsNormEps
        self._proj.wrappedValue = Linear(config.hiddenSize, config.numExperts, bias: false)
        self._routerScale.wrappedValue = MLXArray.ones([config.hiddenSize])
        self._perExpertScale.wrappedValue = MLXArray.ones([config.numExperts])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> (indices: MLXArray, weights: MLXArray) {
        // rms_norm(x, nil) * scale * hidden^-0.5 == rms_norm(x, scale * hidden^-0.5)
        let h = MLXFast.rmsNorm(x, weight: routerScale * rootSize, eps: eps)
        let scores = proj(h)
        let topKIndices = argPartition(-scores, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
        let topKLogits = takeAlong(scores, topKIndices, axis: -1)
        var topKWeights = softmax(topKLogits, axis: -1, precise: true)
        topKWeights = topKWeights * perExpertScale[topKIndices]
        return (indices: topKIndices, weights: topKWeights)
    }
}

class DiffusionGemmaExperts: Module {
    @ModuleInfo(key: "switch_glu") var switchGLU: SwitchGLU

    init(_ config: DiffusionGemmaTextConfiguration) {
        self._switchGLU.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.moeIntermediateSize,
            numExperts: config.numExperts,
            activation: { safeGeluApproximate($0) },
            bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, indices: MLXArray, weights: MLXArray) -> MLXArray {
        let (B, S, H) = (x.dim(0), x.dim(1), x.dim(2))
        let K = indices.dim(-1)
        let expertOut = switchGLU(x.reshaped(B * S, H), indices.reshaped(B * S, K))
        let weightsFlat = expandedDimensions(weights.reshaped(B * S, K), axis: -1)
        return (expertOut * weightsFlat).sum(axis: -2).reshaped(B, S, H)
    }
}

// MARK: - Decoder layer (parallel dense MLP + routed MoE, Gemma4 structure)

class DiffusionGemmaLayer: Module {
    let layerIndex: Int

    @ModuleInfo(key: "self_attn") var selfAttention: DiffusionGemmaAttention
    @ModuleInfo var mlp: Gemma4MLP
    @ModuleInfo var router: DiffusionGemmaRouter
    @ModuleInfo var experts: DiffusionGemmaExperts

    @ModuleInfo(key: "input_layernorm") var inputLayernorm: Gemma4RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayernorm: Gemma4RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayernorm: Gemma4RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayernorm: Gemma4RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm_2") var preFeedforwardLayernorm2: Gemma4RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm_1") var postFeedforwardLayernorm1:
        Gemma4RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm_2") var postFeedforwardLayernorm2:
        Gemma4RMSNorm

    @ModuleInfo(key: "layer_scalar") var layerScalar: MLXArray

    init(_ config: DiffusionGemmaTextConfiguration, layerIndex: Int) {
        self.layerIndex = layerIndex
        self._selfAttention.wrappedValue = DiffusionGemmaAttention(
            config, layerIndex: layerIndex)
        self.mlp = Gemma4MLP(
            dimensions: config.hiddenSize, hiddenDimensions: config.intermediateSize)
        self.router = DiffusionGemmaRouter(config)
        self.experts = DiffusionGemmaExperts(config)

        let norm = { Gemma4RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps) }
        self._inputLayernorm.wrappedValue = norm()
        self._postAttentionLayernorm.wrappedValue = norm()
        self._preFeedforwardLayernorm.wrappedValue = norm()
        self._postFeedforwardLayernorm.wrappedValue = norm()
        self._preFeedforwardLayernorm2.wrappedValue = norm()
        self._postFeedforwardLayernorm1.wrappedValue = norm()
        self._postFeedforwardLayernorm2.wrappedValue = norm()

        self._layerScalar.wrappedValue = MLXArray([Float(1.0)])

        super.init()
    }

    /// `scalarOverride` carries the encoder-side layer scalar; the decoder
    /// pass uses this layer's own `layer_scalar`.
    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?,
        mode: DiffusionGemmaForwardMode,
        scalarOverride: MLXArray? = nil
    ) -> MLXArray {
        var residual = x
        var h = inputLayernorm(x)
        h = selfAttention(h, mask: mask, cache: cache, mode: mode)
        h = postAttentionLayernorm(h)
        h = residual + h

        residual = h

        var h1 = preFeedforwardLayernorm(h)
        h1 = mlp(h1)
        h1 = postFeedforwardLayernorm1(h1)

        let (topKIndices, topKWeights) = router(h)
        var h2 = preFeedforwardLayernorm2(h)
        h2 = experts(h2, indices: topKIndices, weights: topKWeights)
        h2 = postFeedforwardLayernorm2(h2)

        h = postFeedforwardLayernorm(h1 + h2)
        h = residual + h
        return h * (scalarOverride ?? layerScalar)
    }
}

// MARK: - Self-conditioning

class DiffusionGemmaSelfConditioning: Module {
    @ModuleInfo(key: "pre_norm") var preNorm: Gemma4RMSNorm
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear
    // post-norm is RMSNormNoScale (no learnable weight)

    let eps: Float

    init(_ config: DiffusionGemmaTextConfiguration) {
        self.eps = config.rmsNormEps
        self._preNorm.wrappedValue = Gemma4RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._gateProj.wrappedValue = Linear(
            config.hiddenSize, config.intermediateSize, bias: false)
        self._upProj.wrappedValue = Linear(
            config.hiddenSize, config.intermediateSize, bias: false)
        self._downProj.wrappedValue = Linear(
            config.intermediateSize, config.hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ inputsEmbeds: MLXArray, signal: MLXArray) -> MLXArray {
        let normed = preNorm(signal)
        let projected = downProj(safeGeluApproximate(gateProj(normed)) * upProj(normed))
        return rmsNormNoScale(inputsEmbeds + projected, eps: eps)
    }
}

// MARK: - Encoder layer scalars (checkpoint: model.encoder.language_model.layers.N.layer_scalar)

class DiffusionGemmaEncoderScalarLayer: Module {
    @ModuleInfo(key: "layer_scalar") var layerScalar: MLXArray

    override init() {
        self._layerScalar.wrappedValue = MLXArray([Float(1.0)])
        super.init()
    }
}

class DiffusionGemmaEncoderScalarStack: Module {
    @ModuleInfo var layers: [DiffusionGemmaEncoderScalarLayer]

    init(count: Int) {
        self._layers.wrappedValue = (0 ..< count).map { _ in
            DiffusionGemmaEncoderScalarLayer()
        }
        super.init()
    }
}

class DiffusionGemmaEncoderModule: Module {
    @ModuleInfo(key: "language_model") var languageModel: DiffusionGemmaEncoderScalarStack

    /// Vision attachment points. The concrete tower/embedder classes live in
    /// MLXVLM (which depends on this module, not vice versa), so they are
    /// installed post-init as opaque `Module`s purely for weight loading;
    /// the compute path is injected as a closure on the top-level model.
    @ModuleInfo(key: "vision_tower") var visionTower: Module?
    @ModuleInfo(key: "embed_vision") var embedVision: Module?

    init(layerCount: Int) {
        self._languageModel.wrappedValue = DiffusionGemmaEncoderScalarStack(count: layerCount)
        super.init()
    }
}

// MARK: - Decoder stack

class DiffusionGemmaDecoderModule: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo var layers: [DiffusionGemmaLayer]
    @ModuleInfo var norm: Gemma4RMSNorm
    @ModuleInfo(key: "self_conditioning") var selfConditioning: DiffusionGemmaSelfConditioning

    let config: DiffusionGemmaTextConfiguration

    init(_ config: DiffusionGemmaTextConfiguration) {
        self.config = config
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        self._layers.wrappedValue = (0 ..< config.numHiddenLayers).map { i in
            DiffusionGemmaLayer(config, layerIndex: i)
        }
        self.norm = Gemma4RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._selfConditioning.wrappedValue = DiffusionGemmaSelfConditioning(config)
        super.init()
    }

    func embedScale(_ dtype: DType) -> MLXArray {
        MLXArray(sqrt(Float(config.hiddenSize)), dtype: dtype)
    }
}

class DiffusionGemmaInnerModel: Module {
    @ModuleInfo var decoder: DiffusionGemmaDecoderModule
    @ModuleInfo var encoder: DiffusionGemmaEncoderModule

    init(_ config: DiffusionGemmaTextConfiguration) {
        self._decoder.wrappedValue = DiffusionGemmaDecoderModule(config)
        self._encoder.wrappedValue = DiffusionGemmaEncoderModule(
            layerCount: config.numHiddenLayers)
        super.init()
    }
}

// MARK: - Top-level model

public class DiffusionGemmaModel: Module, LLMModel {

    @ModuleInfo var model: DiffusionGemmaInnerModel

    public let config: DiffusionGemmaConfiguration
    let textConfig: DiffusionGemmaTextConfiguration

    public var vocabularySize: Int { textConfig.vocabSize }

    public init(_ config: DiffusionGemmaConfiguration) {
        self.config = config
        self.textConfig = config.textConfig
        self._model.wrappedValue = DiffusionGemmaInnerModel(config.textConfig)
        super.init()
    }

    // MARK: AR-route guard

    /// DiffusionGemma cannot be driven by the autoregressive TokenIterator;
    /// `generate()` dispatches conforming models to
    /// `BlockDiffusionTokenIterator` instead. Throwing here makes any other
    /// route fail loudly instead of silently producing AR garbage.
    public func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws
        -> PrepareResult
    {
        throw BlockDiffusionModelError.requiresBlockDiffusionEngine(config.modelType)
    }

    // MARK: Cache topology

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        (0 ..< textConfig.numHiddenLayers).map { i in
            let layerType =
                i < textConfig.layerTypes.count
                ? textConfig.layerTypes[i] : "sliding_attention"
            if layerType == "full_attention" {
                if let maxKVSize = parameters?.maxKVSize {
                    return RotatingKVCache(maxSize: maxKVSize, keep: 4)
                }
                return KVCacheSimple()
            }
            return RotatingKVCache(maxSize: textConfig.slidingWindow, keep: 0)
        }
    }

    // MARK: Forward passes

    private func masks(
        h: MLXArray, cache: [KVCache]
    ) -> (
        global: MLXFast.ScaledDotProductAttentionMaskMode,
        sliding: MLXFast.ScaledDotProductAttentionMaskMode
    ) {
        let layerTypes = textConfig.layerTypes
        let globalIdx =
            layerTypes.firstIndex(of: "full_attention") ?? (textConfig.numHiddenLayers - 1)
        let slidingIdx = layerTypes.firstIndex(of: "sliding_attention") ?? 0
        let globalCache = globalIdx < cache.count ? cache[globalIdx] : nil
        let slidingCache = slidingIdx < cache.count ? cache[slidingIdx] : nil
        return (
            global: createAttentionMask(h: h, cache: globalCache),
            sliding: createAttentionMask(
                h: h, cache: slidingCache, windowSize: textConfig.slidingWindow)
        )
    }

    func encoderForwardInternal(_ tokens: MLXArray, cache: [KVCache]) {
        let tokens = tokens.ndim == 1 ? tokens.expandedDimensions(axis: 0) : tokens
        let h = embedPromptTokens(tokens)
        runEncoderLayers(h, cache: cache, visionBlockIds: nil)
    }

    /// Encoder forward over precomputed embeddings (multimodal prefill).
    ///
    /// `visionBlockIds` carries per-position contiguous image-block ids
    /// (−1 for text positions). Image positions attend bidirectionally
    /// within their block on top of the causal/sliding mask — the
    /// `use_bidirectional_attention == "vision"` contract from the
    /// reference encoder.
    func encoderForwardInternal(
        embeddings: MLXArray, cache: [KVCache], visionBlockIds: MLXArray?
    ) {
        let h = embeddings.ndim == 2 ? embeddings.expandedDimensions(axis: 0) : embeddings
        runEncoderLayers(h, cache: cache, visionBlockIds: visionBlockIds)
    }

    private func runEncoderLayers(
        _ embeddings: MLXArray, cache: [KVCache], visionBlockIds: MLXArray?
    ) {
        var h = embeddings
        let masks: (
            global: MLXFast.ScaledDotProductAttentionMaskMode,
            sliding: MLXFast.ScaledDotProductAttentionMaskMode
        )
        if let visionBlockIds {
            masks = visionOverlayMasks(h: h, cache: cache, visionBlockIds: visionBlockIds)
        } else {
            masks = self.masks(h: h, cache: cache)
        }

        let layerTypes = textConfig.layerTypes
        for (i, layer) in model.decoder.layers.enumerated() {
            let isGlobal = i < layerTypes.count && layerTypes[i] == "full_attention"
            h = layer(
                h,
                mask: isGlobal ? masks.global : masks.sliding,
                cache: i < cache.count ? cache[i] : nil,
                mode: .encoder,
                scalarOverride: model.encoder.languageModel.layers[i].layerScalar)
        }
        // Encoder hidden states are unused — only the KV cache side effect
        // matters, so the final norm / logits are skipped.
    }

    /// Causal (+ sliding-window) masks OR'd with bidirectional attention
    /// inside each contiguous image block. Only used for media prefill,
    /// which runs single-shot from an empty cache.
    private func visionOverlayMasks(
        h: MLXArray, cache: [KVCache], visionBlockIds: MLXArray?
    ) -> (
        global: MLXFast.ScaledDotProductAttentionMaskMode,
        sliding: MLXFast.ScaledDotProductAttentionMaskMode
    ) {
        let n = h.dim(1)
        let offset = cache.first?.offset ?? 0
        let positions = MLXArray(Int32(offset) ..< Int32(offset + n))
        let queryPositions = expandedDimensions(positions, axis: -1)
        let keyPositions = expandedDimensions(positions, axis: 0)
        let causal = greaterEqual(queryPositions, keyPositions)
        var overlay = MLXArray.zeros([n, n], dtype: .bool)
        if let blockIds = visionBlockIds {
            let ids = blockIds.reshaped(-1)
            let q = expandedDimensions(ids, axis: -1)
            let k = expandedDimensions(ids, axis: 0)
            overlay = logicalAnd(
                greaterEqual(q, MLXArray(Int32(0))), equal(q, k))
        }
        let globalMask = logicalOr(causal, overlay)
        let window = textConfig.slidingWindow
        let withinWindow = less(
            queryPositions, keyPositions + MLXArray(Int32(window)))
        let slidingMask = logicalOr(logicalAnd(causal, withinWindow), overlay)
        return (
            global: .array(globalMask),
            sliding: .array(slidingMask)
        )
    }

    // MARK: Vision attachment (installed by the VLM factory)

    /// Compute closure injected by MLXVLM: builds the full spliced prompt
    /// embeddings (text embeds + image features scattered over the image
    /// placeholder positions) for a media-bearing `LMInput`. `nil` (or a
    /// closure returning `nil`) means text-only operation.
    public var visionPromptEmbedder:
        ((LMInput, DiffusionGemmaModel) throws -> (embeddings: MLXArray, visionBlockIds: MLXArray)?)?

    /// Install the vision tower + multimodal embedder modules (for weight
    /// loading) and the prompt-embedding compute closure. Called by the
    /// VLM factory before weights load; text-only loads never call this.
    public func installVision(
        tower: Module,
        embedder: Module,
        promptEmbedder:
            @escaping (LMInput, DiffusionGemmaModel) throws
                -> (embeddings: MLXArray, visionBlockIds: MLXArray)?
    ) {
        model.encoder.visionTower = tower
        model.encoder.embedVision = embedder
        visionPromptEmbedder = promptEmbedder
    }

    /// Prompt token embedding with image placeholders mapped to pad before
    /// embedding — the reference encoder always does this, with or without
    /// pixels. Without it, history turns that re-render `<|image|>`
    /// placeholders (but carry no pixels) would inject the raw image-token
    /// embedding 280× and derail the model into immediate EOS.
    public func embedPromptTokens(_ tokens: MLXArray) -> MLXArray {
        var tokens = tokens.ndim == 1 ? tokens.expandedDimensions(axis: 0) : tokens
        if let imageToken = config.imageTokenId {
            tokens = MLX.where(
                MLX.equal(tokens, MLXArray(Int32(imageToken))),
                MLXArray(Int32(textConfig.padTokenId)),
                tokens)
        }
        let h = model.decoder.embedTokens(tokens)
        return h * model.decoder.embedScale(h.dtype)
    }

    public var visionTowerModule: Module? { model.encoder.visionTower }
    public var visionEmbedderModule: Module? { model.encoder.embedVision }
    public var imageTokenId: Int? { config.imageTokenId }

    func decoderForwardInternal(
        canvas: MLXArray, cache: [KVCache], selfConditioningLogits: MLXArray?
    ) -> MLXArray {
        let canvas = canvas.ndim == 1 ? canvas.expandedDimensions(axis: 0) : canvas
        let embeds = model.decoder.embedTokens(canvas) * model.decoder.embedScale(.bfloat16)

        let signal: MLXArray
        if let logits = selfConditioningLogits {
            // Soft embeddings from the previous step's logits.
            let probs = softmax(logits.asType(.float32), axis: -1, precise: true)
            // The tied embedding head ships quantized in many bundles (e.g.
            // Gemma 4 QAT q6). Its packed `.weight` is grouped along the hidden
            // axis, so the soft-embedding contraction over the vocab axis is not
            // a valid quantized matmul and collapses the signal to a 0-D scalar
            // (crashing the MoE router on denoising step >= 2). Dequantize the
            // embedding table for this `probs @ W` soft-embedding; dense bundles
            // keep using the plain weight.
            let embedTable: MLXArray
            if let qEmbed = model.decoder.embedTokens as? QuantizedEmbedding {
                embedTable = dequantized(
                    qEmbed.weight, scales: qEmbed.scales, biases: qEmbed.biases,
                    groupSize: qEmbed.groupSize, bits: qEmbed.bits, mode: qEmbed.mode)
            } else {
                embedTable = model.decoder.embedTokens.weight
            }
            let soft = probs.asType(embeds.dtype).matmul(embedTable.asType(embeds.dtype))
            signal = soft * model.decoder.embedScale(embeds.dtype)
        } else {
            signal = MLXArray.zeros(embeds.shape, dtype: embeds.dtype)
        }
        var h = model.decoder.selfConditioning(embeds, signal: signal)

        // B=1, no padding: the canvas attends bidirectionally to itself and
        // to the visible encoder slice each attention layer selects, so no
        // mask is needed.
        let encoderOffset = cache.first?.offset ?? 0
        for (i, layer) in model.decoder.layers.enumerated() {
            h = layer(
                h,
                mask: .none,
                cache: i < cache.count ? cache[i] : nil,
                mode: .decoder(encoderOffset: encoderOffset))
        }
        h = model.decoder.norm(h)

        var logits = model.decoder.embedTokens.asLinear(h)
        if let cap = textConfig.finalLogitSoftcapping, cap > 0 {
            logits = tanh(logits / cap) * cap
        }
        return logits
    }

    // MARK: Weight sanitization

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var processed = [String: MLXArray]()
        processed.reserveCapacity(weights.count)

        for (key, value) in weights {
            // Vision tower / multimodal embedder: kept (with the checkpoint's
            // ClippableLinear `.linear.` nesting stripped) when the VLM
            // factory installed vision modules; dropped for text-only loads.
            if key.hasPrefix("model.encoder.vision_tower.")
                || key.hasPrefix("model.encoder.embed_vision.")
            {
                if model.encoder.visionTower != nil {
                    processed[key.replacingOccurrences(of: ".linear.", with: ".")] = value
                }
                continue
            }
            if key.hasPrefix("model.encoder.language_model.") {
                // Only the per-layer encoder scalars are real encoder
                // weights; everything else is tied to the decoder.
                if key.hasSuffix(".layer_scalar") {
                    processed[key] = value
                }
                continue
            }
            if key == "lm_head.weight" || key.contains("rotary_emb") {
                continue
            }

            // Routed experts: the checkpoint ships a fused gate_up_proj and
            // a bare down_proj as stacked 3-D tensors; the module tree uses
            // SwitchGLU with separate gate/up projections. Output-axis
            // splitting is safe for quantized tensors (packing is along the
            // input axis).
            if key.hasSuffix(".experts.gate_up_proj.weight")
                || key.hasSuffix(".experts.gate_up_proj.scales")
                || key.hasSuffix(".experts.gate_up_proj.biases")
            {
                let outputDim = value.dim(-2)
                let half = outputDim / 2
                let prefixRange = key.range(of: ".experts.gate_up_proj.")!
                let base = String(key[..<prefixRange.lowerBound])
                let suffix = String(key[prefixRange.upperBound...])
                processed["\(base).experts.switch_glu.gate_proj.\(suffix)"] =
                    value[.ellipsis, ..<half, 0...]
                processed["\(base).experts.switch_glu.up_proj.\(suffix)"] =
                    value[.ellipsis, half..., 0...]
                continue
            }
            if let range = key.range(of: ".experts.down_proj.") {
                let base = String(key[..<range.lowerBound])
                let suffix = String(key[range.upperBound...])
                processed["\(base).experts.switch_glu.down_proj.\(suffix)"] = value
                continue
            }

            processed[key] = value
        }

        return processed
    }
}

// MARK: - BlockDiffusionModel conformance

extension DiffusionGemmaModel: BlockDiffusionModel {
    public var blockDiffusionDefaults: BlockDiffusionParameters {
        BlockDiffusionParameters(
            canvasLength: config.canvasLength,
            eosTokenIds: Set(config.eosTokenIds),
            padTokenId: textConfig.padTokenId)
    }

    public var diffusionVocabularySize: Int { textConfig.vocabSize }

    public func encoderForward(_ tokens: MLXArray, cache: [KVCache]) {
        encoderForwardInternal(tokens, cache: cache)
    }

    public func decoderForward(
        canvas: MLXArray, cache: [KVCache], selfConditioningLogits: MLXArray?
    ) -> MLXArray {
        decoderForwardInternal(
            canvas: canvas, cache: cache, selfConditioningLogits: selfConditioningLogits)
    }

    public func encoderPromptEmbeddings(
        for input: LMInput
    ) throws -> (embeddings: MLXArray, visionBlockIds: MLXArray)? {
        try visionPromptEmbedder?(input, self)
    }

    public func encoderForward(
        embeddings: MLXArray, cache: [KVCache], visionBlockIds: MLXArray?
    ) {
        encoderForwardInternal(
            embeddings: embeddings, cache: cache, visionBlockIds: visionBlockIds)
    }
}

extension DiffusionGemmaModel: LoRAModel {
    public var loraLayers: [Module] {
        model.decoder.layers
    }
}
