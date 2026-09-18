// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXNN

// ModernBERT (Warner et al., 2024) as a sentence encoder: token embeddings with no position table,
// alternating sliding-window and global attention with a RoPE base per kind, a GeGLU MLP with exact
// GELU, and bias-free projections and norms. Ported from Hugging Face transformers' Apache-2.0
// `modeling_modernbert.py` (4.57.3).

// MARK: - Configuration

/// A ModernBERT `config.json`, read in transformers 4's keys or transformers 5's
/// (`rope_parameters`, `layer_types`), with every absent field at transformers 4.57.3's default. An
/// absent `local_attention` comes from `sliding_window` when that is given. Decoding throws for a
/// configuration the port would otherwise run as a different function, could not run at all, or
/// would crash on: an activation other than exact GELU, scaled RoPE, or a `rope_scaling` that
/// overrides the bases, a layer pattern that `global_attn_every_n_layers` does not imply, a
/// non-positive size, a hidden size the attention heads do not divide, or a conflicting
/// `sliding_window` and `local_attention`.
public struct ModernBertConfiguration: Decodable, Sendable {
    /// Size of the token embedding table.
    public let vocabSize: Int

    /// Width of the residual stream.
    public let hiddenSize: Int

    /// Number of transformer layers.
    public let numHiddenLayers: Int

    /// Number of attention heads each layer's hidden size is split across.
    public let numAttentionHeads: Int

    /// Width of the MLP's gated layer; `Wi` emits twice this, the activated half then the gate.
    public let intermediateSize: Int

    /// The longest input the checkpoint was trained on; longer input is truncated to it.
    public let maxPositionEmbeddings: Int

    /// Epsilon added inside LayerNorm for numerical stability.
    public let normEps: Float

    /// The full width of a local layer's window, both sides together; transformers 5 calls half of
    /// it `sliding_window`.
    public let localAttention: Int

    /// Every Nth layer (0-indexed) is global attention; the others are local (sliding-window).
    public let globalAttnEveryNLayers: Int

    /// RoPE base for global-attention layers.
    public let globalRopeTheta: Float

    /// RoPE base for local (sliding-window) attention layers.
    public let localRopeTheta: Float

    /// Width of each attention head: `hiddenSize` divided across `numAttentionHeads`.
    public var headDim: Int { hiddenSize / numAttentionHeads }

    /// Each side of a local layer's window: a query attends keys at most this far away.
    public var windowHalfWidth: Int { localAttention / 2 }

    func isGlobal(layer index: Int) -> Bool { index % globalAttnEveryNLayers == 0 }

    /// One entry of transformers 5's `rope_parameters`, which is keyed by layer type.
    struct RopeParameters: Decodable {
        let ropeTheta: Float?
        let ropeType: String?
        /// The older spelling of `rope_type`, which transformers still reads.
        let type: String?

        var kind: String { ropeType ?? type ?? "default" }

        enum CodingKeys: String, CodingKey {
            case ropeTheta = "rope_theta"
            case ropeType = "rope_type"
            case type
        }
    }

    enum CodingKeys: String, CodingKey {
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case intermediateSize = "intermediate_size"
        case maxPositionEmbeddings = "max_position_embeddings"
        case normEps = "norm_eps"
        case localAttention = "local_attention"
        case slidingWindow = "sliding_window"
        case globalAttnEveryNLayers = "global_attn_every_n_layers"
        case globalRopeTheta = "global_rope_theta"
        case localRopeTheta = "local_rope_theta"
        case hiddenActivation = "hidden_activation"
        case ropeScaling = "rope_scaling"
        case ropeParameters = "rope_parameters"
        case layerTypes = "layer_types"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        // Every default is transformers 4.57.3's ModernBertConfig, so a checkpoint that omits a
        // field still loads as the reference would read it.
        vocabSize = try container.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 50368
        hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 768
        numHiddenLayers = try container.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 22
        numAttentionHeads =
            try container.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 12
        intermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 1152
        maxPositionEmbeddings =
            try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 8192
        normEps = try container.decodeIfPresent(Float.self, forKey: .normEps) ?? 1e-5
        // transformers 5 accepts a `sliding_window` key and lets it override `local_attention`,
        // which 4.57.3 ignores, so where both are given they must agree.
        let explicitLocalAttention =
            try container.decodeIfPresent(Int.self, forKey: .localAttention)
        let slidingWindow = try container.decodeIfPresent(Int.self, forKey: .slidingWindow)
        if let explicitLocalAttention, let slidingWindow,
            slidingWindow != explicitLocalAttention / 2
        {
            throw DecodingError.dataCorruptedError(
                forKey: .slidingWindow, in: container,
                debugDescription:
                    "sliding_window \(slidingWindow) does not match local_attention \(explicitLocalAttention) / 2"
            )
        }
        if let explicitLocalAttention {
            localAttention = explicitLocalAttention
        } else if let slidingWindow {
            guard slidingWindow > 0 else {
                throw DecodingError.dataCorruptedError(
                    forKey: .slidingWindow, in: container,
                    debugDescription: "sliding_window must be positive, got \(slidingWindow)")
            }
            let (doubled, overflowed) = slidingWindow.multipliedReportingOverflow(by: 2)
            guard !overflowed else {
                throw DecodingError.dataCorruptedError(
                    forKey: .slidingWindow, in: container,
                    debugDescription: "sliding_window \(slidingWindow) overflows when doubled")
            }
            localAttention = doubled
        } else {
            localAttention = 128
        }
        globalAttnEveryNLayers =
            try container.decodeIfPresent(Int.self, forKey: .globalAttnEveryNLayers) ?? 3
        // A malformed configuration must throw here rather than crash later on a zero divisor or an
        // inverted range.
        guard vocabSize > 0 else {
            throw DecodingError.dataCorruptedError(
                forKey: .vocabSize, in: container,
                debugDescription: "vocab_size must be positive, got \(vocabSize)")
        }
        guard hiddenSize > 0 else {
            throw DecodingError.dataCorruptedError(
                forKey: .hiddenSize, in: container,
                debugDescription: "hidden_size must be positive, got \(hiddenSize)")
        }
        guard numHiddenLayers > 0 else {
            throw DecodingError.dataCorruptedError(
                forKey: .numHiddenLayers, in: container,
                debugDescription: "num_hidden_layers must be positive, got \(numHiddenLayers)")
        }
        guard numAttentionHeads > 0 else {
            throw DecodingError.dataCorruptedError(
                forKey: .numAttentionHeads, in: container,
                debugDescription: "num_attention_heads must be positive, got \(numAttentionHeads)")
        }
        guard hiddenSize % numAttentionHeads == 0 else {
            throw DecodingError.dataCorruptedError(
                forKey: .numAttentionHeads, in: container,
                debugDescription:
                    "hidden_size \(hiddenSize) is not divisible by num_attention_heads \(numAttentionHeads)"
            )
        }
        guard intermediateSize > 0 else {
            throw DecodingError.dataCorruptedError(
                forKey: .intermediateSize, in: container,
                debugDescription: "intermediate_size must be positive, got \(intermediateSize)")
        }
        guard maxPositionEmbeddings > 0 else {
            throw DecodingError.dataCorruptedError(
                forKey: .maxPositionEmbeddings, in: container,
                debugDescription:
                    "max_position_embeddings must be positive, got \(maxPositionEmbeddings)")
        }
        guard globalAttnEveryNLayers > 0 else {
            throw DecodingError.dataCorruptedError(
                forKey: .globalAttnEveryNLayers, in: container,
                debugDescription:
                    "global_attn_every_n_layers must be positive, got \(globalAttnEveryNLayers)")
        }
        guard localAttention > 0 else {
            throw DecodingError.dataCorruptedError(
                forKey: .localAttention, in: container,
                debugDescription: "local_attention must be positive, got \(localAttention)")
        }
        // transformers 5 writes the RoPE bases into `rope_parameters`, keyed by layer type, and no
        // longer saves the two keys 4.57.3 reads. Where both exist, `rope_parameters` wins, as it
        // does there.
        let rope =
            try container.decodeIfPresent([String: RopeParameters].self, forKey: .ropeParameters)
            ?? [:]
        let legacyGlobal = try container.decodeIfPresent(Float.self, forKey: .globalRopeTheta)
        let global = rope["full_attention"]?.ropeTheta ?? legacyGlobal ?? 160_000
        globalRopeTheta = global
        // In 4.57.3's keys an absent local base means the config default, and an explicit null
        // means "use the global base", which is what the reference's attention does with a None.
        if let local = rope["sliding_attention"]?.ropeTheta {
            localRopeTheta = local
        } else if container.contains(.localRopeTheta) {
            localRopeTheta =
                try container.decodeIfPresent(Float.self, forKey: .localRopeTheta) ?? global
        } else {
            localRopeTheta = 10_000
        }
        // The port computes exact GELU and unscaled RoPE, with the layer pattern that
        // global_attn_every_n_layers implies. A checkpoint asking for anything else would load and
        // compute a different function without a sign, so it is refused here.
        if let other = rope.first(where: { $0.value.kind != "default" }) {
            throw DecodingError.dataCorruptedError(
                forKey: .ropeParameters, in: container,
                debugDescription:
                    "ModernBERT implements rope_type \"default\" only, not \"\(other.value.kind)\" for \(other.key)"
            )
        }
        if let types = try container.decodeIfPresent([String].self, forKey: .layerTypes) {
            let every = globalAttnEveryNLayers
            let implied = (0 ..< numHiddenLayers).map {
                $0 % every == 0 ? "full_attention" : "sliding_attention"
            }
            guard types == implied else {
                throw DecodingError.dataCorruptedError(
                    forKey: .layerTypes, in: container,
                    debugDescription:
                        "layer_types disagrees with global_attn_every_n_layers = \(globalAttnEveryNLayers)"
                )
            }
        }
        // transformers' ACT2FN maps both spellings to the exact erf GELU; anything else would
        // compute a different function.
        let activation =
            try container.decodeIfPresent(String.self, forKey: .hiddenActivation) ?? "gelu"
        guard activation == "gelu" || activation == "gelu_python" else {
            throw DecodingError.dataCorruptedError(
                forKey: .hiddenActivation, in: container,
                debugDescription:
                    "ModernBERT implements hidden_activation \"gelu\" and \"gelu_python\" only, not \"\(activation)\""
            )
        }
        // rope_scaling may be absent, null, {}, or an explicit "default": all mean unscaled RoPE,
        // which is what the port computes. Anything else would compute a different function.
        if container.contains(.ropeScaling) {
            let isNull = try container.decodeNil(forKey: .ropeScaling)
            if !isNull {
                let scaling = try container.decode(RopeParameters.self, forKey: .ropeScaling)
                // transformers 5 merges a "default" rope_scaling's own rope_theta into both
                // rope_parameters entries, replacing both bases; 4.57.3 and the port ignore it.
                guard scaling.kind == "default", scaling.ropeTheta == nil else {
                    let reason =
                        scaling.ropeTheta != nil
                        ? "an overriding rope_theta" : "rope_type \"\(scaling.kind)\""
                    throw DecodingError.dataCorruptedError(
                        forKey: .ropeScaling, in: container,
                        debugDescription:
                            "ModernBERT does not implement rope_scaling with \(reason)")
                }
            }
        }
    }
}

// MARK: - Building blocks
//
// Internal rather than private, unlike Bert.swift's, so tests can drive each one alone. Every
// Linear and Embedding is an @ModuleInfo property: the loader's quantize(model:) pass can only
// replace a module declared that way.

final class ModernBertEmbedding: Module {
    @ModuleInfo(key: "tok_embeddings") var tokEmbeddings: Embedding
    @ModuleInfo var norm: LayerNorm

    init(_ config: ModernBertConfiguration) {
        _tokEmbeddings.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        _norm.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize, eps: config.normEps, bias: false)
    }

    func callAsFunction(_ inputIds: MLXArray) -> MLXArray {
        norm(tokEmbeddings(inputIds))
    }
}

final class ModernBertAttention: Module {
    let heads: Int
    let headDim: Int
    let scale: Float
    @ModuleInfo(key: "Wqkv") var wqkv: Linear
    @ModuleInfo(key: "Wo") var wo: Linear
    let rope: RoPE

    init(_ config: ModernBertConfiguration, layer index: Int) {
        heads = config.numAttentionHeads
        headDim = config.headDim
        scale = 1 / Float(config.headDim).squareRoot()
        _wqkv.wrappedValue = Linear(config.hiddenSize, 3 * config.hiddenSize, bias: false)
        _wo.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: false)
        rope = RoPE(
            dimensions: config.headDim, traditional: false,
            base: config.isGlobal(layer: index) ? config.globalRopeTheta : config.localRopeTheta)
    }

    /// `mask` is boolean, `true` where a query may attend a key, broadcastable to
    /// `(batch, heads, length, length)`; `nil` lets every query attend every key.
    func callAsFunction(_ x: MLXArray, mask: MLXArray?) -> MLXArray {
        let (batch, length) = (x.dim(0), x.dim(1))
        // (batch, length, 3, heads, headDim) -> (3, batch, heads, length, headDim)
        let qkv = wqkv(x).reshaped(batch, length, 3, heads, headDim).transposed(2, 0, 3, 1, 4)
        let attended = MLXFast.scaledDotProductAttention(
            queries: rope(qkv[0]), keys: rope(qkv[1]), values: qkv[2], scale: scale,
            mask: mask.map { .array($0) } ?? .none)
        return wo(attended.transposed(0, 2, 1, 3).reshaped(batch, length, -1))
    }
}

final class ModernBertMLP: Module, UnaryLayer {
    @ModuleInfo(key: "Wi") var wi: Linear
    @ModuleInfo(key: "Wo") var wo: Linear

    init(_ config: ModernBertConfiguration) {
        _wi.wrappedValue = Linear(config.hiddenSize, 2 * config.intermediateSize, bias: false)
        _wo.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
    }

    /// GeGLU as the reference computes it: `input, gate = Wi(x).chunk(2)`, then
    /// `Wo(gelu(input) * gate)`, with the exact erf GELU. MLX's `.precise` is the tanh
    /// approximation, which ModernBERT was not trained with.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let halves = split(wi(x), parts: 2, axis: -1)
        return wo(gelu(halves[0]) * halves[1])
    }
}

final class ModernBertLayer: Module {
    /// Absent on layer 0, where the reference uses an identity and the checkpoint has no tensor.
    @ModuleInfo(key: "attn_norm") var attnNorm: LayerNorm?
    let attn: ModernBertAttention
    @ModuleInfo(key: "mlp_norm") var mlpNorm: LayerNorm
    let mlp: ModernBertMLP
    let isGlobal: Bool

    init(_ config: ModernBertConfiguration, layer index: Int) {
        isGlobal = config.isGlobal(layer: index)
        _attnNorm.wrappedValue =
            index == 0
            ? nil : LayerNorm(dimensions: config.hiddenSize, eps: config.normEps, bias: false)
        attn = ModernBertAttention(config, layer: index)
        _mlpNorm.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize, eps: config.normEps, bias: false)
        mlp = ModernBertMLP(config)
    }

    func callAsFunction(_ x: MLXArray, globalMask: MLXArray?, localMask: MLXArray) -> MLXArray {
        let normed = attnNorm.map { $0(x) } ?? x
        let h = x + attn(normed, mask: isGlobal ? globalMask : localMask)
        return h + mlp(mlpNorm(h))
    }
}

// MARK: - Model

/// The ModernBERT encoder as an ``EmbeddingModel``.
///
/// It returns the final hidden states, after `final_norm`, and no pooled output: the model has no
/// pooler, and pooling happens in ``Pooling``.
///
/// Inputs are token ids shaped `(batch, length)`, or `(length)` for a single sequence, which is
/// then a batch of one. An attention mask has the same length, with one row per sequence or a
/// single row for all of them: `true` or non-zero where a token is attended, `false` or 0 at
/// padding. An additive mask is not supported: a 0/-inf mask would be read as its inverse.
/// `positionIds` and `tokenTypeIds` are ignored: positions are always `0 ..< length`, the
/// reference's default.
///
/// Input beyond ``ModernBertConfiguration/maxPositionEmbeddings`` is truncated to it, the mask
/// with it, and the hidden states then cover only that length. The mask passed in matches the
/// full input, but a mask the caller keeps for pooling must be clamped to the returned length.
/// This is not tokenizer truncation, which keeps the closing separator: a caller who wants parity
/// with transformers' `truncation=True` must truncate when tokenizing.
public final class ModernBertModel: Module, EmbeddingModel {
    /// The number of token ids the embedding table holds, the configuration's `vocab_size`.
    public let vocabularySize: Int

    /// The CLS strategy the loader falls back on when a checkpoint has no `1_Pooling/config.json`;
    /// never derived from `classifier_pooling`, which configures Hugging Face's classification head
    /// rather than sentence pooling, and which a converter can leave reading `mean` on a CLS model.
    public let poolingStrategy: Pooling.Strategy? = .cls

    let configuration: ModernBertConfiguration
    @ModuleInfo var embeddings: ModernBertEmbedding
    let layers: [ModernBertLayer]
    @ModuleInfo(key: "final_norm") var finalNorm: LayerNorm

    /// Creates the encoder `configuration` describes, with placeholder weights until the loader, or
    /// `update(parameters:verify:)`, sets them.
    public init(_ configuration: ModernBertConfiguration) {
        self.configuration = configuration
        vocabularySize = configuration.vocabSize
        _embeddings.wrappedValue = ModernBertEmbedding(configuration)
        layers = (0 ..< configuration.numHiddenLayers).map {
            ModernBertLayer(configuration, layer: $0)
        }
        _finalNorm.wrappedValue = LayerNorm(
            dimensions: configuration.hiddenSize, eps: configuration.normEps, bias: false)
    }

    /// Encodes `inputs` into the final hidden states, shaped
    /// `(batch, min(length, maxPositionEmbeddings), hiddenSize)`, returned as `hiddenStates` with a
    /// `nil` `pooledOutput`. `positionIds` and `tokenTypeIds` are ignored; ``ModernBertModel``
    /// describes the conventions for `inputs` and `attentionMask`.
    public func callAsFunction(
        _ inputs: MLXArray, positionIds: MLXArray? = nil, tokenTypeIds: MLXArray? = nil,
        attentionMask: MLXArray? = nil
    ) -> EmbeddingModelOutput {
        let states = layerHiddenStates(inputs, attentionMask: attentionMask)
        return EmbeddingModelOutput(
            hiddenStates: finalNorm(states[states.count - 1]), pooledOutput: nil)
    }

    /// Every stage's output: `numHiddenLayers + 1` arrays shaped
    /// `(batch, min(length, maxPositionEmbeddings), hiddenSize)`, the embedding output and then
    /// each layer's, the last taken before `final_norm`, as transformers'
    /// `output_hidden_states=True` returns them. `inputs` and `attentionMask` follow the
    /// conventions ``ModernBertModel`` describes.
    public func layerHiddenStates(_ inputs: MLXArray, attentionMask: MLXArray? = nil) -> [MLXArray]
    {
        var ids = inputs.ndim == 1 ? inputs.expandedDimensions(axis: 0) : inputs
        var keep = attentionMask.map { mask in
            let m = mask.ndim == 1 ? mask.expandedDimensions(axis: 0) : mask
            return m.dtype == .bool ? m : m .!= MLXArray(Int32(0))
        }
        if let keep {
            precondition(
                keep.ndim == 2,
                "attention mask must be (batch, length) or (length), got shape \(keep.shape)")
            precondition(
                keep.dim(0) == 1 || keep.dim(0) == ids.dim(0),
                "attention mask batch \(keep.dim(0)) is neither 1 nor the input batch \(ids.dim(0))"
            )
            precondition(
                keep.dim(1) == ids.dim(1),
                "attention mask length \(keep.dim(1)) does not match input length \(ids.dim(1))")
        }
        // The limit is read from the config: 8192 for most ModernBERTs, and 32768 for IBM's
        // granite-embedding-311m-multilingual-r2, though a published 8-bit conversion of it says
        // 8192. No tokenizer or loader in this library truncates an encoder's input, and RoPE past
        // the limit extrapolates silently. A caller that keeps its own full-length mask must clamp
        // it too.
        let limit = configuration.maxPositionEmbeddings
        if ids.dim(1) > limit {
            ids = ids[0..., ..<limit]
            keep = keep?[0..., ..<limit]
        }
        let (globalMask, localMask) = masks(keep, length: ids.dim(1))
        var h = embeddings(ids)
        var states = [h]
        for layer in layers {
            h = layer(h, globalMask: globalMask, localMask: localMask)
            states.append(h)
        }
        return states
    }

    /// Boolean masks, `true` = attend, built once per forward pass. The window band is `L × L`
    /// boolean, built by broadcast comparisons, so no `L × L` integer array is materialized. With
    /// a caller mask, the global mask is that mask as `(B, 1, 1, L)`, broadcast inside the kernel,
    /// and the local mask is it AND the band, `(B, 1, L, L)` boolean. Without one, global layers
    /// take no mask and local layers the band alone.
    ///
    /// Additive masks are fragile here. MLX's tiled attention kernels scale them by log2(e) in
    /// float32, so a bfloat16 or float32 fill at the dtype's most negative finite value overflows
    /// to -inf; float16 cannot even hold the customary -1e9, which is -inf there. A row the window
    /// masks entirely then divides 0/0 when the keys fill whole kernel tiles (mlx-embeddings
    /// issue #80). A boolean mask cannot reach that path: MLX's kernels keep a
    /// fully masked row finite. The value it takes depends on which kernel ran, so padded
    /// positions are not comparable across implementations.
    func masks(_ keep: MLXArray?, length: Int) -> (global: MLXArray?, local: MLXArray) {
        let positions = MLXArray.arange(length, dtype: .int32)
        let rows = positions.expandedDimensions(axis: 1)
        let cols = positions.expandedDimensions(axis: 0)
        let w = MLXArray(Int32(configuration.windowHalfWidth))
        let band = logicalAnd(rows .<= cols + w, cols .<= rows + w)
        guard let keep else { return (nil, band) }
        let global = keep.reshaped(keep.dim(0), 1, 1, length)
        return (global, logicalAnd(global, band))
    }

    /// Maps a checkpoint's names onto this model's. Hugging Face's `ModernBertFor*` exports nest
    /// the encoder under `model.` and keep their task heads at the top level: `head.` and
    /// `decoder.` for masked LM, `head.` and `classifier.` for classification, QA and multiple
    /// choice. The prefix is stripped and the heads dropped because the loader's
    /// `update(parameters:verify: [.all])` rejects keys the model does not define. Every other name
    /// passes through unchanged, so a bare `ModernBertModel` checkpoint is untouched and a stray
    /// tensor still fails the load by name.
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = [String: MLXArray]()
        for (key, value) in weights {
            let bare = key.hasPrefix("model.") ? String(key.dropFirst("model.".count)) : key
            let name = bare != key && weights[bare] != nil ? key : bare
            if ["head.", "decoder.", "classifier."].contains(where: { name.hasPrefix($0) }) {
                continue
            }
            sanitized[name] = value
        }
        return sanitized
    }
}
