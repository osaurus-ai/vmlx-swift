//
//  LFM2MoE.swift
//  mlx-swift-lm
//
//  Created by Sachin Desai on 10/08/25.
//
//  Port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/lfm2_moe.py
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

public struct LFM2MoEConfiguration: Codable, Sendable {
    public let modelType: String
    public let vocabularySize: Int
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let moeIntermediateSize: Int
    public let hiddenLayers: Int
    public let numExperts: Int
    public let numExpertsPerToken: Int
    public let normTopkProb: Bool
    public let attentionHeads: Int
    public let kvHeads: Int
    public let maxPositionEmbeddings: Int
    public let useExpertBias: Bool
    public let numDenseLayers: Int
    public let normEps: Float
    public let convBias: Bool
    public let convLCache: Int
    public let ropeTheta: Float
    public let routedScalingFactor: Float

    private let _fullAttnIdxs: [Int]?
    private let layerTypes: [String]?
    private let ropeParameters: [String: StringOrNumber]?

    public var fullAttnIdxs: [Int] {
        if let explicit = _fullAttnIdxs {
            return explicit
        }
        guard let layerTypes else { return [] }
        return layerTypes.enumerated().compactMap { index, type in
            type == "full_attention" ? index : nil
        }
    }

    public var headDimensions: Int { hiddenSize / attentionHeads }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case numExperts = "num_experts"
        case numExpertsPerToken = "num_experts_per_tok"
        case normTopkProb = "norm_topk_prob"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case maxPositionEmbeddings = "max_position_embeddings"
        case useExpertBias = "use_expert_bias"
        case numDenseLayers = "num_dense_layers"
        case normEps = "norm_eps"
        case convBias = "conv_bias"
        case convLCache = "conv_L_cache"
        case ropeTheta = "rope_theta"
        case routedScalingFactor = "routed_scaling_factor"
        case ropeParameters = "rope_parameters"
        case _fullAttnIdxs = "full_attn_idxs"
        case layerTypes = "layer_types"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        self.modelType = try container.decode(String.self, forKey: .modelType)
        self.vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)
        self.hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        self.intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        self.moeIntermediateSize = try container.decode(Int.self, forKey: .moeIntermediateSize)
        self.hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        self.numExperts = try container.decode(Int.self, forKey: .numExperts)
        self.numExpertsPerToken = try container.decode(Int.self, forKey: .numExpertsPerToken)
        self.normTopkProb = try container.decode(Bool.self, forKey: .normTopkProb)
        self.attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        self.kvHeads = try container.decode(Int.self, forKey: .kvHeads)
        self.maxPositionEmbeddings = try container.decode(Int.self, forKey: .maxPositionEmbeddings)
        self.useExpertBias = try container.decode(Bool.self, forKey: .useExpertBias)
        self.numDenseLayers = try container.decode(Int.self, forKey: .numDenseLayers)
        self.normEps = try container.decode(Float.self, forKey: .normEps)
        self.convBias = try container.decode(Bool.self, forKey: .convBias)
        self.convLCache = try container.decode(Int.self, forKey: .convLCache)
        self.ropeParameters = try container.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeParameters)

        let ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1000000.0
        self.ropeTheta = ropeParameters?["rope_theta"]?.asFloat() ?? ropeTheta
        self.routedScalingFactor =
            try container.decodeIfPresent(Float.self, forKey: .routedScalingFactor) ?? 1
        self._fullAttnIdxs = try container.decodeIfPresent([Int].self, forKey: ._fullAttnIdxs)
        self.layerTypes = try container.decodeIfPresent([String].self, forKey: .layerTypes)
    }
}

class LFM2MoEAttention: Module {
    let args: LFM2MoEConfiguration
    let scale: Float
    let headDim: Int

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    @ModuleInfo(key: "q_layernorm") var qLayerNorm: RMSNorm
    @ModuleInfo(key: "k_layernorm") var kLayerNorm: RMSNorm

    let rope: RoPE

    init(_ args: LFM2MoEConfiguration) {
        self.args = args
        self.headDim = args.headDimensions
        self.scale = pow(Float(headDim), -0.5)

        let dim = args.hiddenSize
        _qProj.wrappedValue = Linear(dim, args.attentionHeads * headDim, bias: false)
        _kProj.wrappedValue = Linear(dim, args.kvHeads * headDim, bias: false)
        _vProj.wrappedValue = Linear(dim, args.kvHeads * headDim, bias: false)
        _outProj.wrappedValue = Linear(args.attentionHeads * headDim, dim, bias: false)

        _qLayerNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.normEps)
        _kLayerNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.normEps)

        self.rope = RoPE(
            dimensions: headDim,
            traditional: false,
            base: args.ropeTheta
        )
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)

        var queries = qProj(x)
        var keys = kProj(x)
        var values = vProj(x)

        queries = qLayerNorm(queries.reshaped(B, L, args.attentionHeads, -1)).transposed(0, 2, 1, 3)
        keys = kLayerNorm(keys.reshaped(B, L, args.kvHeads, -1)).transposed(0, 2, 1, 3)
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

        return outProj(output)
    }
}

class LFM2MoEShortConv: Module {
    let args: LFM2MoEConfiguration
    let layerIdx: Int
    let lCache: Int
    let bias: Bool

    @ModuleInfo(key: "conv") var conv: Conv1d
    @ModuleInfo(key: "in_proj") var inProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(_ args: LFM2MoEConfiguration, layerIdx: Int) {
        self.args = args
        self.layerIdx = layerIdx
        self.lCache = args.convLCache
        self.bias = args.convBias

        _conv.wrappedValue = Conv1d(
            inputChannels: args.hiddenSize,
            outputChannels: args.hiddenSize,
            kernelSize: lCache,
            groups: args.hiddenSize,
            bias: bias
        )

        _inProj.wrappedValue = Linear(args.hiddenSize, 3 * args.hiddenSize, bias: bias)
        _outProj.wrappedValue = Linear(args.hiddenSize, args.hiddenSize, bias: bias)
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXArray?,
        cache: MambaCache?
    ) -> MLXArray {
        let tokenCount = x.dim(1)
        let BCx = inProj(x)
        let parts = BCx.split(parts: 3, axis: -1)
        // Bail on a failed split rather than trapping on the subscripts below.
        guard parts.count == 3 else { return x }
        let B = parts[0]
        let C = parts[1]
        let xComp = parts[2]
        var Bx = B * xComp

        if let mask {
            let expandedMask = mask[.ellipsis, .newAxis]
            let zeros = MLXArray.zeros(Bx.shape, dtype: Bx.dtype)
            Bx = MLX.where(expandedMask, Bx, zeros)
        }

        var state = cache?[0]
        if state == nil {
            state = MLXArray.zeros([Bx.dim(0), lCache - 1, args.hiddenSize], dtype: Bx.dtype)
        }

        Bx = concatenated([state!, Bx], axis: -2)
        if let cache {
            let start = Bx.dim(1) - (lCache - 1)
            cache[0] = Bx[0..., start..., 0...]
            // Keep the logical offset in step with the conv window — the
            // prefix-cache boundary guard vetoes stores when any layer's
            // offset disagrees with the prompt boundary.
            cache.offset += tokenCount
        }

        let convOut = conv(Bx)
        let y = C * convOut
        return outProj(y)
    }
}

class LFM2MoEMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(_ args: LFM2MoEConfiguration, intermediateSize: Int? = nil) {
        let hidden = args.hiddenSize
        let inter = intermediateSize ?? args.intermediateSize

        _gateProj.wrappedValue = Linear(hidden, inter, bias: false)
        _upProj.wrappedValue = Linear(hidden, inter, bias: false)
        _downProj.wrappedValue = Linear(inter, hidden, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

class Lfm2MoeSparseMoeBlock: Module, UnaryLayer {
    let args: LFM2MoEConfiguration
    let layerIdx: Int
    let numExperts: Int
    let topK: Int
    let normTopKProb: Bool
    let useExpertBias: Bool

    @ModuleInfo(key: "gate") var gate: Linear
    @ModuleInfo(key: "switch_mlp") var experts: Module
    /// Ordinary affine accessor retained for router diagnostics.
    var switchMLP: any SwitchGLULayer { experts as! any SwitchGLULayer }
    @ModuleInfo(key: "expert_bias") var expertBias: MLXArray?

    init(_ args: LFM2MoEConfiguration, layerIdx: Int,
         routedExperts: (Module & WeightedRoutedExpertLayer)? = nil) {
        self.args = args
        self.layerIdx = layerIdx
        self.numExperts = args.numExperts
        self.topK = args.numExpertsPerToken
        self.normTopKProb = args.normTopkProb
        self.useExpertBias = args.useExpertBias

        _gate.wrappedValue = Linear(args.hiddenSize, numExperts, bias: false)
        if let routedExperts {
            _experts.wrappedValue = routedExperts
        } else {
            _experts.wrappedValue = SwitchGLU(
                inputDims: args.hiddenSize,
                hiddenDims: args.moeIntermediateSize,
                numExperts: numExperts,
                bias: false
            )
        }

        if useExpertBias {
            _expertBias.wrappedValue = MLXArray.zeros([numExperts])
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Sigmoid routing, matching the HF Transformers Lfm2Moe reference
        // (and mlx-lm 0.32): expert_bias only picks the top-k experts; the
        // mixing weights are the unbiased sigmoid scores, renormalized and
        // scaled by routed_scaling_factor.
        //
        // The dtype path is part of the contract. The reference takes the
        // sigmoid in the activation dtype, adds the checkpoint's float32
        // expert_bias (promoting the selection scores to float32) and mixes
        // with the activation-dtype scores. The bias must stay float32: the
        // checkpoint biases differ by ~1e-2 while sigmoid scores of competing
        // experts can sit within 1e-5 of each other, so a bias rounded to
        // bfloat16 collapses those near-ties into exact ties that argPartition
        // breaks arbitrarily, routing the token to a different expert than
        // the reference (see `preservesCheckpointParameterDTypes`).
        let k = topK
        let probabilities = sigmoid(gate(x))
        let selection: MLXArray
        if useExpertBias, let expertBias {
            selection = probabilities.asType(.float32) + expertBias
        } else {
            selection = probabilities
        }
        // Preserve the reference's selected-slot order as well as its set.
        // Negating then taking the first k chooses the same experts, but a
        // different permutation changes low-precision score normalization
        // and weighted reduction.
        let indices = argPartition(selection, kth: -k, axis: -1)[.ellipsis, (-k)...]
        var scores = takeAlong(probabilities, indices, axis: -1)
        if normTopKProb {
            scores = scores / (scores.sum(axis: -1, keepDims: true) + 1e-6)
        }
        if args.routedScalingFactor != 1 {
            scores = scores * args.routedScalingFactor
        }

        scores = scores.asType(x.dtype)

        // SwitchGLU also conforms to the shared weighted protocol, but LFM's
        // ordinary branch must retain its native contribution rounding.
        guard let affine = experts as? any SwitchGLULayer else {
            return (experts as! any WeightedRoutedExpertLayer).callRouted(x, indices: indices, scores: scores)
        }
        let expertOutputs = affine(x, indices)
        let weighted = expertOutputs * scores[.ellipsis, .newAxis]
        return weighted.sum(axis: -2)
    }
}

class LFM2MoEDecoderLayer: Module {
    let isAttentionLayer: Bool
    let usesDenseFeedForward: Bool

    @ModuleInfo(key: "self_attn") var attention: LFM2MoEAttention?
    @ModuleInfo(key: "conv") var conv: LFM2MoEShortConv?
    @ModuleInfo(key: "feed_forward") var feedForward: Module & UnaryLayer
    @ModuleInfo(key: "operator_norm") var operatorNorm: RMSNorm
    @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm

    init(_ args: LFM2MoEConfiguration, layerIdx: Int,
         routedExperts: (Module & WeightedRoutedExpertLayer)? = nil) {
        self.isAttentionLayer = args.fullAttnIdxs.contains(layerIdx)
        self.usesDenseFeedForward = layerIdx < args.numDenseLayers

        if isAttentionLayer {
            _attention.wrappedValue = LFM2MoEAttention(args)
        } else {
            _conv.wrappedValue = LFM2MoEShortConv(args, layerIdx: layerIdx)
        }

        if usesDenseFeedForward {
            _feedForward.wrappedValue = LFM2MoEMLP(args)
        } else {
            _feedForward.wrappedValue = Lfm2MoeSparseMoeBlock(args, layerIdx: layerIdx, routedExperts: routedExperts)
        }

        _operatorNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.normEps)
        _ffnNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.normEps)
    }

    func callAsFunction(
        _ x: MLXArray,
        attentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
        ssmMask: MLXArray?,
        cache: KVCache?
    ) -> MLXArray {
        let residual = operatorNorm(x)
        let r: MLXArray
        if isAttentionLayer {
            r = attention!(residual, mask: attentionMask, cache: cache)
        } else {
            r = conv!(residual, mask: ssmMask, cache: cache as? MambaCache)
        }

        let h = x + r
        let out = feedForward(ffnNorm(h))
        return h + out
    }
}

public class LFM2MoEModelInner: Module {
    let args: LFM2MoEConfiguration
    let layers: [LFM2MoEDecoderLayer]
    let firstAttentionIndex: Int?
    let firstConvIndex: Int?

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "embedding_norm") var embeddingNorm: RMSNorm

    init(_ args: LFM2MoEConfiguration, routedExperts: [Int: Module & WeightedRoutedExpertLayer] = [:]) {
        self.args = args
        precondition(args.vocabularySize > 0)

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize,
            dimensions: args.hiddenSize
        )

        self.layers = (0 ..< args.hiddenLayers).map {
            LFM2MoEDecoderLayer(args, layerIdx: $0, routedExperts: routedExperts[$0])
        }
        self.firstAttentionIndex = args.fullAttnIdxs.first
        self.firstConvIndex = layers.firstIndex(where: { !$0.isAttentionLayer })

        _embeddingNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.normEps)
    }

    func callAsFunction(
        _ inputs: MLXArray,
        cache: [KVCache]? = nil,
        inputEmbeddings: MLXArray? = nil
    ) -> MLXArray {
        var hidden = inputEmbeddings ?? embedTokens(inputs)

        let attentionMask: MLXFast.ScaledDotProductAttentionMaskMode = {
            guard let index = firstAttentionIndex else { return .none }
            let layerCache = cache.flatMap { index < $0.count ? $0[index] : nil }
            // Causality is required even when the caller does not retain KV.
            return createAttentionMask(h: hidden, cache: layerCache)
        }()

        let ssmMask: MLXArray? = {
            guard let index = firstConvIndex,
                let cache,
                index < cache.count
            else { return nil }
            return createSSMMask(h: hidden, cache: cache[index] as? MambaCache)
        }()

        for (i, layer) in layers.enumerated() {
            hidden = layer(hidden, attentionMask: attentionMask, ssmMask: ssmMask, cache: cache?[i])
        }

        return embeddingNorm(hidden)
    }
}

public class LFM2MoEModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]
    let configuration: LFM2MoEConfiguration

    public let model: LFM2MoEModelInner
    private var customTensorNames: Set<String> = []

    public init(_ args: LFM2MoEConfiguration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0 ..< args.hiddenLayers).map { layerIdx in
            args.fullAttnIdxs.contains(layerIdx) ? args.kvHeads : 0
        }
        self.model = LFM2MoEModelInner(args)
    }

    init(_ args: LFM2MoEConfiguration,
         routedExperts: [Int: Module & WeightedRoutedExpertLayer], excludedTensorNames: Set<String>) throws {
        let sparseLayers = Set(args.numDenseLayers ..< args.hiddenLayers)
        guard Set(routedExperts.keys) == sparseLayers,
            JANGHModelPreparation.isCanonicalRoutedExclusionSet(
                excludedTensorNames, routedLayers: sparseLayers, feedForwardModule: "feed_forward")
        else { throw LFM2MoEJANGHError.invalid("incomplete LFM custom routed installation") }
        configuration = args
        vocabularySize = args.vocabularySize
        kvHeads = (0 ..< args.hiddenLayers).map { args.fullAttnIdxs.contains($0) ? args.kvHeads : 0 }
        model = LFM2MoEModelInner(args, routedExperts: routedExperts)
        customTensorNames = excludedTensorNames
    }

    /// LFM2 MoE checkpoints store every floating parameter in the activation
    /// dtype except `feed_forward.expert_bias`, which HF keeps as a float32
    /// buffer and mlx-lm loads unchanged. The loader's blanket bfloat16
    /// materialization would round that bias and change expert selection on
    /// near-tied tokens, so this model keeps the checkpoint dtypes as loaded.
    public var preservesCheckpointParameterDTypes: Bool { true }

    /// Old softmax routing, rounded expert bias and the negative-k slot order
    /// produced incompatible hidden states even with identical weights/token IDs.
    /// This revision names the actual model's corrected forward semantics;
    /// equivalent kernel optimizations must keep it stable.
    public var cacheComputationIdentity: String? {
        "lfm2-moe:sigmoid-f32-bias-positive-topk-v1"
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let out = model(inputs, cache: cache)
        return model.embedTokens.asLinear(out)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]

        for (name, param) in weights {
            var tensor = param
            if name.contains("conv.weight") {
                if tensor.shape.last! > tensor.shape[1] {
                    tensor = tensor.transposed(0, 2, 1)
                }
            }

            // Older LFM checkpoints use w1/w2/w3 module names. Rename the
            // module path, not only `.weight`: quantized checkpoints carry
            // sibling `.scales` and `.biases` tensors that must follow the
            // same destination module or Module.update reports stale
            // w1/w2/w3 children as unhandled.
            let replacements = [
                ".w1.": ".gate_proj.",
                ".w2.": ".down_proj.",
                ".w3.": ".up_proj.",
            ]
            var updatedName = name
            for (old, new) in replacements where updatedName.contains(old) {
                updatedName = updatedName.replacingOccurrences(of: old, with: new)
            }

            sanitized[updatedName] = tensor
        }

        for layerIdx in 0 ..< configuration.hiddenLayers {
            let prefix = "model.layers.\(layerIdx)"
            let expertPrefix = "\(prefix).feed_forward.experts"

            if sanitized["\(expertPrefix).0.gate_proj.weight"] == nil {
                continue
            }

            for name in ["gate_proj", "down_proj", "up_proj"] {
                for leaf in ["weight", "scales", "biases"] {
                    let keys = (0 ..< configuration.numExperts).map {
                        "\(expertPrefix).\($0).\(name).\(leaf)"
                    }
                    guard keys.allSatisfy({ sanitized[$0] != nil }) else {
                        continue
                    }
                    let stacked = keys.map { sanitized.removeValue(forKey: $0)! }
                    sanitized[
                        "\(prefix).feed_forward.switch_mlp.\(name).\(leaf)"
                    ] = MLX.stacked(stacked)
                }
            }
        }

        return sanitized
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        (0 ..< configuration.hiddenLayers).map { layerIdx in
            if configuration.fullAttnIdxs.contains(layerIdx) {
                KVCacheSimple()
            } else {
                MambaCache()
            }
        }
    }
}

extension LFM2MoEModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}

extension LFM2MoEModel: SafetensorsLoadKeyExcluding {
    public func excludeFromGenericSafetensorsLoad(key: String) -> Bool { customTensorNames.contains(key) }
    public var requiresExactTensorMmapBuffers: Bool { !customTensorNames.isEmpty }
}
