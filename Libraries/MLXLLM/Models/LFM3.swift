//
//  LFM3.swift
//  mlx-swift-lm
//
//  LiquidAI LFM3: the LFM2 hybrid (gated short convolutions + grouped-query
//  attention) with three attention changes and sandwich normalization.
//
//  References: the bundles' own `modeling_lfm3.py`, and the MLX port in
//  mlx-vlm `mlx_vlm/models/lfm3/language.py`.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Per-layer-type rotary settings (`rope_parameters.full_attention` /
/// `rope_parameters.sliding_attention`).
public struct LFM3RopeParameters: Sendable, Equatable {
    public let ropeTheta: Float
    public let partialRotaryFactor: Float

    public init(ropeTheta: Float, partialRotaryFactor: Float = 1.0) {
        self.ropeTheta = ropeTheta
        self.partialRotaryFactor = partialRotaryFactor
    }
}

public struct LFM3Configuration: Codable, Sendable {
    public let modelType: String
    public let vocabularySize: Int
    public let hiddenSize: Int
    public let hiddenLayers: Int
    public let attentionHeads: Int
    public let kvHeads: Int
    public let headDim: Int
    public let intermediateSize: Int
    public let normEps: Float
    public let convBias: Bool
    public let convLCache: Int
    /// One entry per layer: `conv`, `full_attention` or `sliding_attention`.
    public let layerTypes: [String]
    public let slidingWindow: Int
    public let useSandwichNorm: Bool
    public let tieWordEmbeddings: Bool
    public let maxPositionEmbeddings: Int?
    public let ropeParameters: [String: LFM3RopeParameters]

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case intermediateSize = "intermediate_size"
        case normEps = "norm_eps"
        case convBias = "conv_bias"
        case convLCache = "conv_L_cache"
        case layerTypes = "layer_types"
        case slidingWindow = "sliding_window"
        case useSandwichNorm = "use_sandwich_norm"
        case tieWordEmbeddings = "tie_word_embeddings"
        case maxPositionEmbeddings = "max_position_embeddings"
        case ropeParameters = "rope_parameters"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? "lfm3"
        vocabularySize = try container.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 128000
        hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        kvHeads = try container.decode(Int.self, forKey: .kvHeads)
        headDim =
            try container.decodeIfPresent(Int.self, forKey: .headDim) ?? hiddenSize / attentionHeads
        intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        normEps = try container.decodeIfPresent(Float.self, forKey: .normEps) ?? 1e-5
        convBias = try container.decodeIfPresent(Bool.self, forKey: .convBias) ?? false
        convLCache = try container.decodeIfPresent(Int.self, forKey: .convLCache) ?? 7
        slidingWindow = try container.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 256
        useSandwichNorm =
            try container.decodeIfPresent(Bool.self, forKey: .useSandwichNorm) ?? true
        tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true
        maxPositionEmbeddings = try container.decodeIfPresent(
            Int.self, forKey: .maxPositionEmbeddings)

        // LFM3 has no default layer pattern: the checkpoint must say which
        // layers convolve and which attend.
        layerTypes = try container.decode([String].self, forKey: .layerTypes)
        guard layerTypes.count == hiddenLayers else {
            throw DecodingError.dataCorruptedError(
                forKey: .layerTypes, in: container,
                debugDescription:
                    "layer_types has \(layerTypes.count) entries for \(hiddenLayers) layers")
        }
        let known: Set<String> = ["conv", "full_attention", "sliding_attention"]
        if let unknown = layerTypes.first(where: { !known.contains($0) }) {
            throw DecodingError.dataCorruptedError(
                forKey: .layerTypes, in: container,
                debugDescription: "unknown LFM3 layer type '\(unknown)'")
        }

        let raw =
            try container.decodeIfPresent(
                [String: [String: StringOrNumber]].self, forKey: .ropeParameters) ?? [:]
        var rope: [String: LFM3RopeParameters] = [
            "full_attention": LFM3RopeParameters(ropeTheta: 1e6, partialRotaryFactor: 0.5),
            "sliding_attention": LFM3RopeParameters(ropeTheta: 1e4),
        ]
        for (layerType, parameters) in raw {
            // Only plain rotary embeddings exist in released checkpoints. A
            // scaled variant decoded as plain would run, but every position
            // past the original context would be wrong, so refuse it here.
            if case .string(let ropeType) = parameters["rope_type"], ropeType != "default" {
                throw DecodingError.dataCorruptedError(
                    forKey: .ropeParameters, in: container,
                    debugDescription: "LFM3 rope_type '\(ropeType)' is not supported")
            }
            rope[layerType] = LFM3RopeParameters(
                ropeTheta: parameters["rope_theta"]?.asFloat() ?? rope[layerType]?.ropeTheta
                    ?? 1e6,
                partialRotaryFactor: parameters["partial_rotary_factor"]?.asFloat() ?? 1.0)
        }
        ropeParameters = rope
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(modelType, forKey: .modelType)
        try container.encode(vocabularySize, forKey: .vocabularySize)
        try container.encode(hiddenSize, forKey: .hiddenSize)
        try container.encode(hiddenLayers, forKey: .hiddenLayers)
        try container.encode(attentionHeads, forKey: .attentionHeads)
        try container.encode(kvHeads, forKey: .kvHeads)
        try container.encode(headDim, forKey: .headDim)
        try container.encode(intermediateSize, forKey: .intermediateSize)
        try container.encode(normEps, forKey: .normEps)
        try container.encode(convBias, forKey: .convBias)
        try container.encode(convLCache, forKey: .convLCache)
        try container.encode(layerTypes, forKey: .layerTypes)
        try container.encode(slidingWindow, forKey: .slidingWindow)
        try container.encode(useSandwichNorm, forKey: .useSandwichNorm)
        try container.encode(tieWordEmbeddings, forKey: .tieWordEmbeddings)
        try container.encodeIfPresent(maxPositionEmbeddings, forKey: .maxPositionEmbeddings)
        let rope = ropeParameters.mapValues { p -> [String: StringOrNumber] in
            [
                "rope_type": .string("default"), "rope_theta": .float(p.ropeTheta),
                "partial_rotary_factor": .float(p.partialRotaryFactor),
            ]
        }
        try container.encode(rope, forKey: .ropeParameters)
    }

    func rope(for layerType: String) -> LFM3RopeParameters {
        ropeParameters[layerType] ?? LFM3RopeParameters(ropeTheta: 1e6)
    }
}

/// LFM3 rotary embedding.
///
/// INVARIANT: with `partial_rotary_factor < 1` LFM3 rotates the FIRST
/// `angles` coordinates of EACH HEAD HALF — pairs `(i, i + headDim/2)` for
/// `i < angles` — and leaves the rest of both halves unrotated (the reference
/// zero-pads `inv_freq`). That is not MLX's partial RoPE: `RoPE(dimensions:
/// 64)` rotates a contiguous 64-wide prefix with pairs `(i, i + 32)`, which
/// produces fluent-looking but wrong attention on every full-attention layer.
///
/// The fast kernel is run over the whole head with explicit frequencies; the
/// unrotated pairs get frequency 1e30, an angle of at most ~3e-25 rad at the
/// longest context, which is the identity in every float format. (Infinity
/// would be exact in IEEE arithmetic, but Metal compiles with fast math,
/// which may assume no infinities.)
final class LFM3RoPE: Module, OffsetLayer, ArrayOffsetLayer {
    let dimensions: Int
    let base: Float?
    let _freqs: MLXArray?

    init(headDim: Int, parameters: LFM3RopeParameters) {
        self.dimensions = headDim
        let half = headDim / 2
        let angles = min(half, max(1, Int(Float(headDim) * parameters.partialRotaryFactor) / 2))
        if angles == half {
            self.base = parameters.ropeTheta
            self._freqs = nil
        } else {
            let freqs = (0 ..< half).map { i -> Float in
                i < angles
                    ? Float(pow(Double(parameters.ropeTheta), Double(2 * i) / Double(headDim)))
                    : 1e30
            }
            self.base = nil
            self._freqs = MLXArray(freqs)
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray, offset: Int = 0) -> MLXArray {
        MLXFast.RoPE(
            x, dimensions: dimensions, traditional: false, base: base, scale: 1.0,
            offset: offset, freqs: _freqs)
    }

    func callAsFunction(_ x: MLXArray, offset: MLXArray) -> MLXArray {
        MLXFast.RoPE(
            x, dimensions: dimensions, traditional: false, base: base, scale: 1.0,
            offset: offset, freqs: _freqs)
    }
}

final class LFM3Attention: Module {
    let attentionHeads: Int
    let kvHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm
    /// One sigmoid gate per head, applied to that head's attention output.
    @ModuleInfo(key: "gate_proj") var gateProj: Linear

    let rope: LFM3RoPE

    init(_ args: LFM3Configuration, layerType: String) {
        self.attentionHeads = args.attentionHeads
        self.kvHeads = args.kvHeads
        self.headDim = args.headDim
        self.scale = pow(Float(headDim), -0.5)

        _qProj.wrappedValue = Linear(args.hiddenSize, attentionHeads * headDim, bias: false)
        _kProj.wrappedValue = Linear(args.hiddenSize, kvHeads * headDim, bias: false)
        _vProj.wrappedValue = Linear(args.hiddenSize, kvHeads * headDim, bias: false)
        _oProj.wrappedValue = Linear(attentionHeads * headDim, args.hiddenSize, bias: false)
        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.normEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.normEps)
        _gateProj.wrappedValue = Linear(args.hiddenSize, attentionHeads, bias: false)

        self.rope = LFM3RoPE(headDim: headDim, parameters: args.rope(for: layerType))
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = qNorm(qProj(x).reshaped(B, L, attentionHeads, headDim))
            .transposed(0, 2, 1, 3)
        var keys = kNorm(kProj(x).reshaped(B, L, kvHeads, headDim)).transposed(0, 2, 1, 3)
        let values = vProj(x).reshaped(B, L, kvHeads, headDim).transposed(0, 2, 1, 3)

        queries = applyRotaryPosition(rope, to: queries, cache: cache)
        keys = applyRotaryPosition(rope, to: keys, cache: cache)

        let output = attentionWithCacheUpdate(
            queries: queries, keys: keys, values: values, cache: cache, scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)  // [B, L, heads, headDim]

        let gate = sigmoid(gateProj(x)).expandedDimensions(axis: -1)  // [B, L, heads, 1]
        return oProj((output * gate).reshaped(B, L, -1))
    }
}

final class LFM3ShortConv: Module {
    let hiddenSize: Int
    let lCache: Int

    @ModuleInfo(key: "conv") var conv: Conv1d
    @ModuleInfo(key: "in_proj") var inProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(_ args: LFM3Configuration) {
        self.hiddenSize = args.hiddenSize
        self.lCache = args.convLCache
        _conv.wrappedValue = Conv1d(
            inputChannels: args.hiddenSize, outputChannels: args.hiddenSize,
            kernelSize: lCache, groups: args.hiddenSize, bias: args.convBias)
        _inProj.wrappedValue = Linear(args.hiddenSize, 3 * args.hiddenSize, bias: args.convBias)
        _outProj.wrappedValue = Linear(args.hiddenSize, args.hiddenSize, bias: args.convBias)
    }

    func callAsFunction(_ x: MLXArray, cache: MambaCache?) -> MLXArray {
        let tokenCount = x.dim(1)
        let parts = inProj(x).split(parts: 3, axis: -1)
        guard parts.count == 3 else { return x }
        let (b, c, h) = (parts[0], parts[1], parts[2])
        var bx = b * h

        let state =
            cache?[0]
            ?? MLXArray.zeros([bx.dim(0), lCache - 1, hiddenSize], dtype: bx.dtype)
        bx = concatenated([state, bx], axis: -2)
        if let cache {
            cache[0] = bx[0..., (bx.dim(1) - (lCache - 1))..., 0...]
            // INVARIANT: conv layers advance the logical offset with the
            // window. The prefix-cache boundary guard compares every layer's
            // offset to the prompt boundary, and a conv layer left at 0
            // silently vetoes every paged and disk store for the model (the
            // LFM2 family shipped with exactly that).
            cache.offset += tokenCount
        }
        return outProj(c * conv(bx))
    }
}

final class LFM3MLP: Module, UnaryLayer {
    @ModuleInfo(key: "w1") var w1: Linear
    @ModuleInfo(key: "w2") var w2: Linear
    @ModuleInfo(key: "w3") var w3: Linear

    init(_ args: LFM3Configuration) {
        _w1.wrappedValue = Linear(args.hiddenSize, args.intermediateSize, bias: false)
        _w2.wrappedValue = Linear(args.intermediateSize, args.hiddenSize, bias: false)
        _w3.wrappedValue = Linear(args.hiddenSize, args.intermediateSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        w2(silu(w1(x)) * w3(x))
    }
}

final class LFM3DecoderLayer: Module {
    let layerType: String
    var isAttentionLayer: Bool { layerType != "conv" }
    let useSandwichNorm: Bool

    @ModuleInfo(key: "self_attn") var attention: LFM3Attention?
    @ModuleInfo(key: "conv") var conv: LFM3ShortConv?
    @ModuleInfo(key: "feed_forward") var feedForward: LFM3MLP
    @ModuleInfo(key: "operator_norm") var operatorNorm: RMSNorm
    @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm
    @ModuleInfo(key: "post_operator_norm") var postOperatorNorm: RMSNorm?
    @ModuleInfo(key: "post_ffn_norm") var postFFNNorm: RMSNorm?

    init(_ args: LFM3Configuration, layerIdx: Int) {
        self.layerType = args.layerTypes[layerIdx]
        self.useSandwichNorm = args.useSandwichNorm
        if layerType == "conv" {
            _conv.wrappedValue = LFM3ShortConv(args)
        } else {
            _attention.wrappedValue = LFM3Attention(args, layerType: layerType)
        }
        _feedForward.wrappedValue = LFM3MLP(args)
        _operatorNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.normEps)
        _ffnNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.normEps)
        if useSandwichNorm {
            _postOperatorNorm.wrappedValue = RMSNorm(
                dimensions: args.hiddenSize, eps: args.normEps)
            _postFFNNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.normEps)
        }
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        var r =
            isAttentionLayer
            ? attention!(operatorNorm(x), mask: mask, cache: cache)
            : conv!(operatorNorm(x), cache: cache as? MambaCache)
        if let postOperatorNorm { r = postOperatorNorm(r) }
        let h = x + r
        var f = feedForward(ffnNorm(h))
        if let postFFNNorm { f = postFFNNorm(f) }
        return h + f
    }
}

public final class LFM3ModelInner: Module {
    let args: LFM3Configuration
    let layers: [LFM3DecoderLayer]
    let firstFullAttention: Int?
    let firstSlidingAttention: Int?

    @ModuleInfo(key: "embed_tokens") public var embedTokens: Embedding
    @ModuleInfo(key: "embedding_norm") var embeddingNorm: RMSNorm

    public init(_ args: LFM3Configuration) {
        self.args = args
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)
        self.layers = (0 ..< args.hiddenLayers).map { LFM3DecoderLayer(args, layerIdx: $0) }
        self.firstFullAttention = args.layerTypes.firstIndex(of: "full_attention")
        self.firstSlidingAttention = args.layerTypes.firstIndex(of: "sliding_attention")
        _embeddingNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.normEps)
    }

    public func callAsFunction(
        _ inputs: MLXArray?, cache: [KVCache]? = nil, inputEmbeddings: MLXArray? = nil
    ) -> MLXArray {
        var h = inputEmbeddings ?? embedTokens(inputs!)
        let n = h.dim(1)

        // One mask per attention kind, built from that kind's first cache:
        // full layers see the whole prefix, sliding layers the last
        // `sliding_window` positions. Conv layers take no mask.
        let fullMask: MLXFast.ScaledDotProductAttentionMaskMode =
            firstFullAttention.map {
                makeAttentionMask(n: n, cache: cache?[$0], windowSize: nil)
            } ?? .none
        let slidingMask: MLXFast.ScaledDotProductAttentionMaskMode =
            firstSlidingAttention.map {
                makeAttentionMask(n: n, cache: cache?[$0], windowSize: args.slidingWindow)
            } ?? .none

        for (i, layer) in layers.enumerated() {
            let mask: MLXFast.ScaledDotProductAttentionMaskMode
            switch layer.layerType {
            case "full_attention": mask = fullMask
            case "sliding_attention": mask = slidingMask
            default: mask = .none
            }
            h = layer(h, mask: mask, cache: cache?[i])
        }
        return embeddingNorm(h)
    }
}

public final class LFM3Model: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]
    public let model: LFM3ModelInner
    let configuration: LFM3Configuration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ args: LFM3Configuration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = args.layerTypes.map { $0 == "conv" ? 0 : args.kvHeads }
        self.model = LFM3ModelInner(args)
        if !args.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        logits(inputs, cache: cache)
    }

    /// Logits from token ids, or from precomputed input embeddings (the LFM3-VL / Omni
    /// paths splice media features into the embeddings before the decoder runs).
    /// `lastPositionOnly` projects only the final position: a prefill chunk needs one row of
    /// logits, and the full `[L, vocab]` head would be the largest matmul of the chunk.
    public func logits(
        _ inputs: MLXArray?, cache: [KVCache]?, inputEmbeddings: MLXArray? = nil,
        lastPositionOnly: Bool = false
    ) -> MLXArray {
        var out = model(inputs, cache: cache, inputEmbeddings: inputEmbeddings)
        if lastPositionOnly { out = out[0..., (out.dim(1) - 1)..., 0...] }
        if let lmHead { return lmHead(out) }
        return model.embedTokens.asLinear(out)
    }

    /// Accepts both the transformers layout (`model.*`) and the MLX-VLM layout
    /// every released LFM3 MLX bundle uses (`language_model.model.*`), even
    /// the text-only ones. Vision / audio towers in a multimodal bundle loaded
    /// as text-only are dropped.
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var result: [String: MLXArray] = [:]
        for (key, value) in weights {
            var name = key
            if name.hasPrefix("language_model.") {
                name.removeFirst("language_model.".count)
            }
            guard name.hasPrefix("model.") || name.hasPrefix("lm_head.") else { continue }
            if configuration.tieWordEmbeddings, name.hasPrefix("lm_head.") { continue }
            var value = value
            // transformers stores the depthwise conv as [C, 1, K]; MLX wants [C, K, 1].
            if name.hasSuffix("conv.conv.weight"), value.ndim == 3, value.dim(1) == 1,
                value.dim(2) > 1
            {
                value = value.transposed(0, 2, 1)
            }
            result[name] = value
        }
        return result
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        configuration.layerTypes.map { layerType -> KVCache in
            switch layerType {
            case "conv": MambaCache()
            case "sliding_attention":
                RotatingKVCache(maxSize: configuration.slidingWindow, keep: 0)
            default: KVCacheSimple()
            }
        }
    }
}

extension LFM3Model: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
