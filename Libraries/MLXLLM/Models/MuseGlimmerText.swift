//
//  MuseGlimmerText.swift
//  vmlx-swift
//
//  Text tower for Muse Glimmer 30B (`model_type: muse_glimmer`).
//  Architecture notes and the five Gemma divergences that matter are in
//  `Libraries/MLXVLM/Models/MuseGlimmer_ARCHITECTURE.md`.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Configuration

public struct MuseGlimmerTextConfiguration: Codable, Sendable {
    public let modelType: String
    public let hiddenSize: Int
    public let hiddenLayers: Int
    public let intermediateSize: Int
    public let attentionHeads: Int
    public let headDim: Int
    public let kvHeads: Int
    public let vocabularySize: Int
    public let rmsNormEps: Float
    /// Distinct (1e-8) from `rmsNormEps` — used by the two post-sublayer norms.
    public let postNormEps: Float
    public let slidingWindow: Int
    public let maxPositionEmbeddings: Int
    public let qkScaleFactor: Float
    public let outputMultiplier: Float
    public let finalLogitSoftcapping: Float
    /// `"sliding_attention"` / `"full_attention"`, one per layer.
    public let layerTypes: [String]
    /// Per-layer RoPE base; `0` means that layer runs without rotary (NoPE).
    public let layerRopeTheta: [Float]

    public func isSliding(_ layer: Int) -> Bool {
        layerTypes.indices.contains(layer)
            ? layerTypes[layer] == "sliding_attention" : true
    }

    /// `nil` when the layer is NoPE.
    public func ropeTheta(_ layer: Int) -> Float? {
        guard layerRopeTheta.indices.contains(layer) else { return nil }
        let theta = layerRopeTheta[layer]
        return theta > 0 ? theta : nil
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case headDim = "head_dim"
        case kvHeads = "num_key_value_heads"
        case vocabularySize = "vocab_size"
        case rmsNormEps = "rms_norm_eps"
        case postNormEps = "post_norm_eps"
        case slidingWindow = "sliding_window"
        case maxPositionEmbeddings = "max_position_embeddings"
        case qkScaleFactor = "qk_scale_factor"
        case outputMultiplier = "output_multiplier"
        case finalLogitSoftcapping = "final_logit_softcapping"
        case layerTypes = "layer_types"
        case layerRopeTheta = "layer_rope_theta"
    }

    /// Decode-only: kept out of `CodingKeys` so `encode(to:)` still synthesizes
    /// (every case there must map to a stored property).
    private enum RopeKeys: String, CodingKey {
        case ropeParameters = "rope_parameters"
        case ropeTheta = "rope_theta"
    }

    enum VLMCodingKeys: String, CodingKey {
        case textConfig = "text_config"
    }

    private struct RopeParameters: Codable {
        let ropeTheta: Float?
        enum CodingKeys: String, CodingKey { case ropeTheta = "rope_theta" }
    }

    public init(from decoder: Decoder) throws {
        // A converted VLM keeps the text fields nested under `text_config`.
        let outer = try decoder.container(keyedBy: VLMCodingKeys.self)
        let c =
            outer.contains(.textConfig)
            ? try outer.nestedContainer(keyedBy: CodingKeys.self, forKey: .textConfig)
            : try decoder.container(keyedBy: CodingKeys.self)

        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "muse_glimmer_text"
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 6656
        hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 52
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 19968
        attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 32
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? 128
        kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 2
        vocabularySize = try c.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 202048
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5
        postNormEps = try c.decodeIfPresent(Float.self, forKey: .postNormEps) ?? 1e-8
        slidingWindow = try c.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 2048
        maxPositionEmbeddings =
            try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 131_072
        qkScaleFactor = try c.decodeIfPresent(Float.self, forKey: .qkScaleFactor) ?? 3.87
        outputMultiplier =
            try c.decodeIfPresent(Float.self, forKey: .outputMultiplier) ?? 0.196_116_135_138_18
        finalLogitSoftcapping =
            try c.decodeIfPresent(Float.self, forKey: .finalLogitSoftcapping) ?? 20.0

        // Global theta feeds the per-layer default when the checkpoint omits
        // the explicit list. `rope_parameters` is the transformers-5 shape;
        // a flat `rope_theta` is accepted for hand-written configs.
        let ropeContainer =
            outer.contains(.textConfig)
            ? try outer.nestedContainer(keyedBy: RopeKeys.self, forKey: .textConfig)
            : try decoder.container(keyedBy: RopeKeys.self)
        let ropeParams = try ropeContainer.decodeIfPresent(
            RopeParameters.self, forKey: .ropeParameters)
        let flatTheta = try ropeContainer.decodeIfPresent(Float.self, forKey: .ropeTheta)
        let globalTheta = ropeParams?.ropeTheta ?? flatTheta ?? 500_000.0

        let layers = hiddenLayers
        if let types = try c.decodeIfPresent([String].self, forKey: .layerTypes) {
            layerTypes = types
        } else {
            // Reference default: full attention on every 4th layer counted
            // backward from the last, sliding everywhere else.
            layerTypes = (0 ..< layers).map {
                (layers - 1 - $0) % 4 == 0 ? "full_attention" : "sliding_attention"
            }
        }
        if let thetas = try c.decodeIfPresent([Float].self, forKey: .layerRopeTheta) {
            layerRopeTheta = thetas
        } else {
            layerRopeTheta = (0 ..< layers).map {
                (layers - 1 - $0) % 4 == 0 ? 0 : globalTheta
            }
        }
    }
}

// MARK: - Norms

/// `x * rsqrt(mean(x²) + eps) * (1 + weight)`.
///
/// Muse Glimmer stores these weights zero-centered, so the `1 +` is load-
/// bearing: dropping it leaves the model running and merely degrades the
/// output, which is the hardest class of port bug to spot.
class MuseGlimmerCenteredRMSNorm: Module, UnaryLayer {
    let weight: MLXArray
    let eps: Float

    init(dimensions: Int, eps: Float) {
        self.weight = MLXArray.zeros([dimensions])
        self.eps = eps
    }

    /// The checkpoint stores zero-centered gains, so the mathematical weight is
    /// `1 + w`. That `+1` is folded into the tensor once during `sanitize`
    /// rather than recomputed here: this runs 2x per layer across 52 layers, so
    /// doing it per token adds 104 tensor allocations and dispatches to every
    /// single decode step for a value that never changes after load.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: self.weight, eps: self.eps)
    }
}

/// Unit-gain RMSNorm over `head_dim`, applied to Q and K. Carries no learned
/// weight in the checkpoint, so it is deliberately *not* a `Module` — the ones
/// tensor is cached per (dim, dtype) to keep 32 heads × 52 layers from
/// reallocating it on every token.
enum MuseGlimmerMath {
    private static let onesLock = NSLock()
    nonisolated(unsafe) private static var onesCache: [String: MLXArray] = [:]

    static func unitGain(dim: Int, dtype: DType) -> MLXArray {
        let key = "\(dim)-\(dtype)"
        onesLock.lock()
        defer { onesLock.unlock() }
        if let cached = onesCache[key] { return cached }
        let ones = MLXArray.ones([dim]).asType(dtype)
        onesCache[key] = ones
        return ones
    }

    static func scalelessRMSNorm(_ x: MLXArray, eps: Float) -> MLXArray {
        MLXFast.rmsNorm(x, weight: unitGain(dim: x.dim(-1), dtype: x.dtype), eps: eps)
    }
}

// MARK: - Attention

private class MuseGlimmerAttention: Module {
    let nHeads: Int
    let nKVHeads: Int
    let headDim: Int
    let scale: Float
    let qkScaleFactor: Float
    let rmsNormEps: Float

    @ModuleInfo(key: "q_proj") var queryProj: Linear
    @ModuleInfo(key: "k_proj") var keyProj: Linear
    @ModuleInfo(key: "v_proj") var valueProj: Linear
    @ModuleInfo(key: "o_proj") var outputProj: Linear
    /// Attention output gate — distinct from `mlp.gate_proj`.
    @ModuleInfo(key: "gate_proj") var gateProj: Linear

    /// `nil` on NoPE layers.
    let rope: RoPELayer?

    init(_ config: MuseGlimmerTextConfiguration, layerIdx: Int) {
        let dim = config.hiddenSize
        self.nHeads = config.attentionHeads
        self.nKVHeads = config.kvHeads
        self.headDim = config.headDim
        // The standard scale; `qkScaleFactor` rides on top of it, on Q only.
        self.scale = pow(Float(config.headDim), -0.5)
        self.qkScaleFactor = config.qkScaleFactor
        self.rmsNormEps = config.rmsNormEps

        self._queryProj.wrappedValue = Linear(dim, nHeads * headDim, bias: false)
        self._keyProj.wrappedValue = Linear(dim, nKVHeads * headDim, bias: false)
        self._valueProj.wrappedValue = Linear(dim, nKVHeads * headDim, bias: false)
        self._outputProj.wrappedValue = Linear(nHeads * headDim, dim, bias: false)
        self._gateProj.wrappedValue = Linear(dim, nHeads * headDim, bias: false)

        if let theta = config.ropeTheta(layerIdx) {
            self.rope = initializeRope(
                dims: config.headDim, base: theta, traditional: false,
                scalingConfig: nil, maxPositionEmbeddings: config.maxPositionEmbeddings)
        } else {
            self.rope = nil
        }

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache? = nil
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = queryProj(x).reshaped(B, L, nHeads, headDim).transposed(0, 2, 1, 3)
        var keys = keyProj(x).reshaped(B, L, nKVHeads, headDim).transposed(0, 2, 1, 3)
        let values = valueProj(x).reshaped(B, L, nKVHeads, headDim).transposed(0, 2, 1, 3)

        // Scaleless QK-norm, then the asymmetric factor on Q only.
        queries = MuseGlimmerMath.scalelessRMSNorm(queries, eps: rmsNormEps) * qkScaleFactor
        keys = MuseGlimmerMath.scalelessRMSNorm(keys, eps: rmsNormEps)

        // NoPE layers skip rotary entirely.
        if let rope {
            queries = applyRotaryPosition(rope, to: queries, cache: cache)
            keys = applyRotaryPosition(rope, to: keys, cache: cache)
        }

        let attn = attentionWithCacheUpdate(
            queries: queries, keys: keys, values: values,
            cache: cache, scale: scale, mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, nHeads * headDim)

        // Gate reads the layer input, not the attention output.
        return outputProj(attn * sigmoid(gateProj(x)))
    }
}

// MARK: - MLP

private class MuseGlimmerMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(dimensions: Int, hiddenDimensions: Int) {
        self._gateProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        self._upProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        self._downProj.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

// MARK: - Decoder layer

private class MuseGlimmerDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttention: MuseGlimmerAttention
    @ModuleInfo var mlp: MuseGlimmerMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: MuseGlimmerCenteredRMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm:
        MuseGlimmerCenteredRMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayerNorm:
        MuseGlimmerCenteredRMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayerNorm:
        MuseGlimmerCenteredRMSNorm

    init(_ config: MuseGlimmerTextConfiguration, layerIdx: Int) {
        self._selfAttention.wrappedValue = MuseGlimmerAttention(config, layerIdx: layerIdx)
        self.mlp = MuseGlimmerMLP(
            dimensions: config.hiddenSize, hiddenDimensions: config.intermediateSize)
        self._inputLayerNorm.wrappedValue = MuseGlimmerCenteredRMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = MuseGlimmerCenteredRMSNorm(
            dimensions: config.hiddenSize, eps: config.postNormEps)
        self._preFeedforwardLayerNorm.wrappedValue = MuseGlimmerCenteredRMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postFeedforwardLayerNorm.wrappedValue = MuseGlimmerCenteredRMSNorm(
            dimensions: config.hiddenSize, eps: config.postNormEps)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache? = nil
    ) -> MLXArray {
        // Sandwich norms: the post-* norms sit between the sublayer output and
        // the residual add, not after it.
        let attn = postAttentionLayerNorm(
            selfAttention(inputLayerNorm(x), mask: mask, cache: cache))
        let h = x + attn
        let ffn = postFeedforwardLayerNorm(mlp(preFeedforwardLayerNorm(h)))
        return h + ffn
    }
}

// MARK: - Model

public class MuseGlimmerModelInner: Module {
    @ModuleInfo(key: "embed_tokens") public var embedTokens: Embedding
    fileprivate let layers: [MuseGlimmerDecoderLayer]
    fileprivate let norm: MuseGlimmerCenteredRMSNorm

    let config: MuseGlimmerTextConfiguration
    /// Representative layer index per attention kind, for mask construction.
    private let firstSliding: Int
    private let firstFull: Int

    init(_ config: MuseGlimmerTextConfiguration) {
        self.config = config
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        self.layers = (0 ..< config.hiddenLayers).map {
            MuseGlimmerDecoderLayer(config, layerIdx: $0)
        }
        self.norm = MuseGlimmerCenteredRMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self.firstSliding = (0 ..< config.hiddenLayers).first { config.isSliding($0) } ?? 0
        self.firstFull = (0 ..< config.hiddenLayers).first { !config.isSliding($0) } ?? 0
        super.init()
    }

    /// The checkpoint's embedding is a *normed* embedding: a scaleless RMSNorm
    /// applied on top of the raw table lookup (`MuseGlimmerTextNormedEmbedding`
    /// in the reference — kept out of the weight matrix on purpose, so it can
    /// never be folded in). Skipping it feeds layer 0 activations at the wrong
    /// magnitude and the model emits fluent-looking token soup across random
    /// scripts. This is the divergence in place of Gemma's sqrt(hidden) scale,
    /// not the absence of one.
    public func embed(_ inputs: MLXArray) -> MLXArray {
        MuseGlimmerMath.scalelessRMSNorm(embedTokens(inputs), eps: config.rmsNormEps)
    }

    public func callAsFunction(
        _ inputs: MLXArray?, inputEmbedding: MLXArray? = nil, cache: [KVCache?]? = nil
    ) -> MLXArray {
        var h: MLXArray
        if let inputEmbedding {
            // Callers pass already-normed embeddings (see `embed`); vision
            // features scattered into that stream stay raw, matching the
            // reference's masked_scatter over the normed lookup.
            h = inputEmbedding
        } else if let inputs {
            h = embed(inputs)
        } else {
            fatalError("MuseGlimmer: one of inputs or inputEmbedding must be provided")
        }

        let fullMask = createAttentionMask(h: h, cache: cache?[firstFull] ?? nil)
        let slidingMask = createAttentionMask(
            h: h, cache: cache?[firstSliding] ?? nil, windowSize: config.slidingWindow)

        for (i, layer) in layers.enumerated() {
            let mask = config.isSliding(i) ? slidingMask : fullMask
            h = layer(h, mask: mask, cache: cache?[i] ?? nil)
        }
        return norm(h)
    }
}

public class MuseGlimmerTextModel: Module, LLMModel, KVCacheDimensionProvider {
    @ModuleInfo public var model: MuseGlimmerModelInner
    @ModuleInfo(key: "lm_head") public var lmHead: Linear

    public let config: MuseGlimmerTextConfiguration
    public var vocabularySize: Int { config.vocabularySize }
    public var kvHeads: [Int] { Array(repeating: config.kvHeads, count: config.hiddenLayers) }

    public init(_ config: MuseGlimmerTextConfiguration) {
        self.config = config
        self.model = MuseGlimmerModelInner(config)
        self._lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        let out = model(inputs, cache: cache)
        return MuseGlimmerTextModel.applyLogitTail(lmHead(out), config: config)
    }

    /// `lm_head(h) * output_multiplier`, then Gemma-style tanh softcapping.
    public static func applyLogitTail(_ logits: MLXArray, config: MuseGlimmerTextConfiguration)
        -> MLXArray
    {
        var out = logits * config.outputMultiplier
        let cap = config.finalLogitSoftcapping
        if cap > 0 {
            out = tanh(out / cap) * cap
        }
        return out
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        // The released checkpoint nests the text tower under `language_model.`
        // and carries the vision tower alongside it; a text-only load keeps
        // just the former.
        var out: [String: MLXArray] = [:]
        for (key, value) in weights {
            if key.hasPrefix("language_model.") {
                out[String(key.dropFirst("language_model.".count))] = value
            } else if !key.hasPrefix("model.vision_") {
                out[key] = value
            }
        }
        // Fold the centered-norm `+1` in once. Every norm this model uses is
        // zero-centered, and the vision tower's LayerNorms are excluded above,
        // so the transform applies to exactly the text tower's RMSNorm gains.
        for (key, value) in out where Self.isCenteredNormWeight(key) {
            out[key] = value + 1
        }
        return out
    }

    /// Zero-centered RMSNorm gains in the text tower. Matched by suffix because
    /// the layer index varies; `qk_norm` carries no weight and the vision
    /// tower's `norm1`/`norm2` are ordinary LayerNorms that must not be shifted.
    ///
    /// Every one of the decoder layer's FOUR norms is centered. Missing any of
    /// them leaves those layers without their `+1` entirely, which shifts unit
    /// gain to zero gain — the model still runs and still emits fluent text, so
    /// `MuseGlimmerCenteredNormCoverage` pins the list against the checkpoint.
    public static func isCenteredNormWeight(_ key: String) -> Bool {
        guard key.hasSuffix(".weight") else { return false }
        return key.hasSuffix("input_layernorm.weight")
            || key.hasSuffix("post_attention_layernorm.weight")
            || key.hasSuffix("pre_feedforward_layernorm.weight")
            || key.hasSuffix("post_feedforward_layernorm.weight")
            // Dotted suffix so the VLM's `language_model.model.norm.weight`
            // matches too, while `...layernorm.weight` (no dot before `norm`)
            // does not double-match through this branch.
            || key.hasSuffix(".norm.weight")
            || key == "norm.weight"
    }

    public func newCache(parameters: GenerateParameters? = nil) -> [KVCache] {
        (0 ..< config.hiddenLayers).map { i in
            if config.isSliding(i) {
                // `step` is how many rows the cache allocates at a time, and
                // `updateInPlace` writes the whole incoming chunk into that
                // block — so a prefill chunk larger than `step` runs off the
                // end of the freshly allocated region and trips a precondition
                // inside the MLX scatter. Sizing the step to the window means
                // any chunk the prefill loop can produce (it is capped to the
                // window) always fits.
                return RotatingKVCache(
                    maxSize: config.slidingWindow, keep: 0, step: config.slidingWindow)
            }
            let cache = StandardKVCache()
            cache.step = 1024
            return cache
        }
    }
}

extension MuseGlimmerTextModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
