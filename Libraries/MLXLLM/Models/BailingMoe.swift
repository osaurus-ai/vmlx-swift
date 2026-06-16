//
//  BailingMoe.swift
//  LLM
//
//  Port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/bailing_moe.py
//  This architecture is used by the Ling-family models (e.g., Ling Mini).
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

public struct BailingMoeConfiguration: Codable, Sendable {
    var modelType: String
    var hiddenSize: Int
    var intermediateSize: Int
    var maxPositionEmbeddings: Int?
    var moeIntermediateSize: Int
    var numExperts: Int
    var numSharedExperts: Int
    var normTopkProb: Bool
    var attentionHeads: Int
    var numExpertsPerToken: Int
    var hiddenLayers: Int
    var kvHeads: Int
    var rmsNormEps: Float
    var ropeTheta: Float
    var vocabularySize: Int
    var firstKDenseReplace: Int

    // Optional features
    var ropeScaling: [String: StringOrNumber]? = nil
    var useBias: Bool = false
    var useQKVBias: Bool = false
    var useQKNorm: Bool = false
    var tieWordEmbeddings: Bool = false
    var partialRotaryFactor: Float = 1.0
    var moeRouterEnableExpertBias: Bool = false
    var routedScalingFactor: Float = 1.0
    var scoreFunction: String = "softmax"
    var nGroup: Int = 1
    var topkGroup: Int = 4
    var moeSharedExpertIntermediateSize: Int? = nil

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case maxPositionEmbeddings = "max_position_embeddings"
        case moeIntermediateSize = "moe_intermediate_size"
        case numExperts = "num_experts"
        case numSharedExperts = "num_shared_experts"
        case normTopkProb = "norm_topk_prob"
        case attentionHeads = "num_attention_heads"
        case numExpertsPerToken = "num_experts_per_tok"
        case hiddenLayers = "num_hidden_layers"
        case kvHeads = "num_key_value_heads"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case vocabularySize = "vocab_size"
        case firstKDenseReplace = "first_k_dense_replace"
        case ropeScaling = "rope_scaling"
        case useBias = "use_bias"
        case useQKVBias = "use_qkv_bias"
        case useQKNorm = "use_qk_norm"
        case tieWordEmbeddings = "tie_word_embeddings"
        case partialRotaryFactor = "partial_rotary_factor"
        case moeRouterEnableExpertBias = "moe_router_enable_expert_bias"
        case routedScalingFactor = "routed_scaling_factor"
        case scoreFunction = "score_function"
        case nGroup = "n_group"
        case topkGroup = "topk_group"
        case moeSharedExpertIntermediateSize = "moe_shared_expert_intermediate_size"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        modelType = try container.decode(String.self, forKey: .modelType)
        hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        maxPositionEmbeddings = try container.decodeIfPresent(
            Int.self, forKey: .maxPositionEmbeddings)
        moeIntermediateSize = try container.decode(Int.self, forKey: .moeIntermediateSize)
        numExperts = try container.decode(Int.self, forKey: .numExperts)
        numSharedExperts = try container.decode(Int.self, forKey: .numSharedExperts)
        normTopkProb = try container.decode(Bool.self, forKey: .normTopkProb)
        attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        numExpertsPerToken = try container.decode(Int.self, forKey: .numExpertsPerToken)
        hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        kvHeads = try container.decode(Int.self, forKey: .kvHeads)
        rmsNormEps = try container.decode(Float.self, forKey: .rmsNormEps)
        ropeTheta = try container.decode(Float.self, forKey: .ropeTheta)
        vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)
        firstKDenseReplace = try container.decode(Int.self, forKey: .firstKDenseReplace)
        ropeScaling = try container.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeScaling)
        useBias = try container.decodeIfPresent(Bool.self, forKey: .useBias) ?? false
        useQKVBias = try container.decodeIfPresent(Bool.self, forKey: .useQKVBias) ?? false
        useQKNorm = try container.decodeIfPresent(Bool.self, forKey: .useQKNorm) ?? false
        tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        partialRotaryFactor =
            try container.decodeIfPresent(Float.self, forKey: .partialRotaryFactor) ?? 1.0
        moeRouterEnableExpertBias =
            try container.decodeIfPresent(Bool.self, forKey: .moeRouterEnableExpertBias) ?? false
        routedScalingFactor =
            try container.decodeIfPresent(Float.self, forKey: .routedScalingFactor) ?? 1.0
        scoreFunction =
            try container.decodeIfPresent(String.self, forKey: .scoreFunction) ?? "softmax"
        nGroup = try container.decodeIfPresent(Int.self, forKey: .nGroup) ?? 1
        topkGroup = try container.decodeIfPresent(Int.self, forKey: .topkGroup) ?? nGroup
        moeSharedExpertIntermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .moeSharedExpertIntermediateSize)

        try validateDecodedFields(container: container)
    }

    private func validateDecodedFields(container: KeyedDecodingContainer<CodingKeys>) throws {
        try validatePositive(hiddenSize, key: .hiddenSize, in: container)
        try validatePositive(intermediateSize, key: .intermediateSize, in: container)
        try validatePositive(moeIntermediateSize, key: .moeIntermediateSize, in: container)
        try validatePositive(numExperts, key: .numExperts, in: container)
        try validateNonNegative(numSharedExperts, key: .numSharedExperts, in: container)
        try validatePositive(attentionHeads, key: .attentionHeads, in: container)
        try validatePositive(numExpertsPerToken, key: .numExpertsPerToken, in: container)
        try validatePositive(hiddenLayers, key: .hiddenLayers, in: container)
        try validatePositive(kvHeads, key: .kvHeads, in: container)
        try validatePositive(rmsNormEps, key: .rmsNormEps, in: container)
        try validatePositive(ropeTheta, key: .ropeTheta, in: container)
        try validatePositive(vocabularySize, key: .vocabularySize, in: container)
        try validateNonNegative(firstKDenseReplace, key: .firstKDenseReplace, in: container)
        try validatePositive(partialRotaryFactor, key: .partialRotaryFactor, in: container)
        try validatePositive(routedScalingFactor, key: .routedScalingFactor, in: container)
        try validatePositive(nGroup, key: .nGroup, in: container)
        try validatePositive(topkGroup, key: .topkGroup, in: container)

        if let maxPositionEmbeddings {
            try validatePositive(maxPositionEmbeddings, key: .maxPositionEmbeddings, in: container)
        }
        if let moeSharedExpertIntermediateSize {
            try validatePositive(
                moeSharedExpertIntermediateSize,
                key: .moeSharedExpertIntermediateSize,
                in: container)
        }

        if hiddenSize % attentionHeads != 0 {
            throw DecodingError.dataCorruptedError(
                forKey: .hiddenSize,
                in: container,
                debugDescription:
                    "BailingMoe hidden_size must be divisible by num_attention_heads."
            )
        }

        if attentionHeads % kvHeads != 0 {
            throw DecodingError.dataCorruptedError(
                forKey: .kvHeads,
                in: container,
                debugDescription:
                    "BailingMoe num_attention_heads must be divisible by num_key_value_heads."
            )
        }

        let headDim = hiddenSize / attentionHeads
        let ropeDim = Int(Float(headDim) * partialRotaryFactor)
        if ropeDim <= 0 || ropeDim > headDim {
            throw DecodingError.dataCorruptedError(
                forKey: .partialRotaryFactor,
                in: container,
                debugDescription:
                    "BailingMoe rotary dimension must be positive and no larger than head_dim."
            )
        }

        if numExpertsPerToken > numExperts {
            throw DecodingError.dataCorruptedError(
                forKey: .numExpertsPerToken,
                in: container,
                debugDescription: "BailingMoe num_experts_per_tok must be <= num_experts."
            )
        }

        if numExperts % nGroup != 0 {
            throw DecodingError.dataCorruptedError(
                forKey: .nGroup,
                in: container,
                debugDescription: "BailingMoe num_experts must be divisible by n_group."
            )
        }

        if topkGroup > nGroup {
            throw DecodingError.dataCorruptedError(
                forKey: .topkGroup,
                in: container,
                debugDescription: "BailingMoe topk_group must be > 0 and <= n_group."
            )
        }

        if firstKDenseReplace > hiddenLayers {
            throw DecodingError.dataCorruptedError(
                forKey: .firstKDenseReplace,
                in: container,
                debugDescription:
                    "BailingMoe first_k_dense_replace must be <= num_hidden_layers."
            )
        }

        if scoreFunction != "softmax" && scoreFunction != "sigmoid" {
            throw DecodingError.dataCorruptedError(
                forKey: .scoreFunction,
                in: container,
                debugDescription: "BailingMoe score_function must be softmax or sigmoid."
            )
        }

        if let factor = ropeScaling?["factor"]?.asFloat() {
            if !factor.isFinite || factor <= 0 {
                throw DecodingError.dataCorruptedError(
                    forKey: .ropeScaling,
                    in: container,
                    debugDescription: "BailingMoe rope_scaling.factor must be finite and > 0."
                )
            }
        }
    }

    private func validatePositive(
        _ value: Int,
        key: CodingKeys,
        in container: KeyedDecodingContainer<CodingKeys>
    ) throws {
        if value <= 0 {
            throw DecodingError.dataCorruptedError(
                forKey: key,
                in: container,
                debugDescription: "BailingMoe \(key.rawValue) must be > 0."
            )
        }
    }

    private func validateNonNegative(
        _ value: Int,
        key: CodingKeys,
        in container: KeyedDecodingContainer<CodingKeys>
    ) throws {
        if value < 0 {
            throw DecodingError.dataCorruptedError(
                forKey: key,
                in: container,
                debugDescription: "BailingMoe \(key.rawValue) must be >= 0."
            )
        }
    }

    private func validatePositive(
        _ value: Float,
        key: CodingKeys,
        in container: KeyedDecodingContainer<CodingKeys>
    ) throws {
        if !value.isFinite || value <= 0 {
            throw DecodingError.dataCorruptedError(
                forKey: key,
                in: container,
                debugDescription: "BailingMoe \(key.rawValue) must be finite and > 0."
            )
        }
    }
}

class BailingMoeAttention: Module {
    let args: BailingMoeConfiguration
    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let ropeDim: Int
    let scale: Float

    @ModuleInfo(key: "query_key_value") var qkv: Linear
    @ModuleInfo(key: "dense") var wo: Linear

    @ModuleInfo(key: "query_layernorm") var qNorm: RMSNorm?
    @ModuleInfo(key: "key_layernorm") var kNorm: RMSNorm?

    let rope: RoPELayer

    init(_ args: BailingMoeConfiguration) {
        self.args = args
        self.heads = args.attentionHeads
        self.kvHeads = args.kvHeads
        self.headDim = args.hiddenSize / heads
        self.ropeDim = Int(Float(headDim) * args.partialRotaryFactor)
        self.scale = pow(Float(headDim), -0.5)

        _qkv.wrappedValue = Linear(
            args.hiddenSize,
            (heads + 2 * kvHeads) * headDim,
            bias: args.useQKVBias
        )
        _wo.wrappedValue = Linear(heads * headDim, args.hiddenSize, bias: args.useBias)

        if args.useQKNorm {
            _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)
            _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)
        } else {
            _qNorm.wrappedValue = nil
            _kNorm.wrappedValue = nil
        }

        self.rope = initializeRope(
            dims: ropeDim, base: args.ropeTheta,
            traditional: false, scalingConfig: args.ropeScaling,
            maxPositionEmbeddings: nil
        )
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        let qSize = heads * headDim
        let kSize = kvHeads * headDim
        let qkvOut = qkv(x)
        let splits = split(qkvOut, indices: [qSize, qSize + kSize], axis: -1)
        var queries = splits[0]
        var keys = splits[1]
        var values = splits[2]

        // reshape to (B, L, H, Hd), apply optional per-head norms, then transpose to (B, H, L, Hd)
        queries = queries.reshaped(B, L, heads, -1)
        keys = keys.reshaped(B, L, kvHeads, -1)

        if let qNorm { queries = qNorm(queries) }
        if let kNorm { keys = kNorm(keys) }

        queries = queries.transposed(0, 2, 1, 3)
        keys = keys.transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, kvHeads, -1).transposed(0, 2, 1, 3)

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

class BailingMoeMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear

    init(_ args: BailingMoeConfiguration, hiddenDims: Int? = nil) {
        let inter = hiddenDims ?? args.intermediateSize
        _gate.wrappedValue = Linear(args.hiddenSize, inter, bias: args.useBias)
        _down.wrappedValue = Linear(inter, args.hiddenSize, bias: args.useBias)
        _up.wrappedValue = Linear(args.hiddenSize, inter, bias: args.useBias)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let g = silu(gate(x)); let u = up(x)
        // bfloat16 shares float32's exponent range — no overflow possible.
        // Load.swift convertToBFloat16 ensures all activations are bfloat16.
        let product = g * u
        return down(product)
    }
}

class BailingMoeGate: Module, UnaryLayer {
    let topK: Int
    let nGroup: Int
    let topkGroup: Int
    let numExperts: Int
    let routedScalingFactor: Float
    let normTopkProb: Bool
    let scoreFunction: String

    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "expert_bias") var expertBias: MLXArray

    init(_ args: BailingMoeConfiguration) {
        self.topK = args.numExpertsPerToken
        self.nGroup = args.nGroup
        self.topkGroup = args.topkGroup
        self.routedScalingFactor = args.routedScalingFactor
        self.normTopkProb = args.normTopkProb
        self.scoreFunction = args.scoreFunction
        self.numExperts = args.numExperts

        _gate.wrappedValue = Linear(args.hiddenSize, args.numExperts, bias: false)
        _expertBias.wrappedValue = zeros([args.numExperts])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // This returns a packed result not directly used; callers use groupSelect to get inds and scores.
        gate(x)
    }

    func groupSelect(_ x: MLXArray) -> (inds: MLXArray, scores: MLXArray) {
        let (bsz, seqLen, _) = (x.dim(0), x.dim(1), x.dim(2))

        let logits = gate(x)
        var scores = sigmoid(logits)
        let scoresForChoice = scores + expertBias
        let groupScores = scoresForChoice.reshaped(bsz, seqLen, self.nGroup, -1)

        let topKGroup = top(groupScores, k: 2, axis: -1).sum(axis: -1, keepDims: true)
        let droppedGroups = nGroup - topkGroup
        if droppedGroups > 0 {
            let groupIdx = argPartition(topKGroup, kth: droppedGroups - 1, axis: -2)[
                .ellipsis, ..<droppedGroups, 0...]
            scores = putAlong(
                groupScores, groupIdx, values: MLXArray(0.0, dtype: groupScores.dtype), axis: -2)
            scores = flattened(scores, start: -2, end: -1)
        } else {
            scores = flattened(groupScores, start: -2, end: -1)
        }

        let k = topK
        let inds = argPartition(-scores, kth: k - 1, axis: -1)[.ellipsis, ..<k]
        scores = takeAlong(scores, inds, axis: -1)
        if topK > 1, normTopkProb {
            let denominator = scores.sum(axis: -1, keepDims: true) + MLXArray(1e-20, dtype: scores.dtype)
            scores = scores / denominator
        }
        scores = scores * routedScalingFactor
        return (inds, scores.asType(logits.dtype))
    }
}

class BailingMoeSparseMoeBlock: Module, UnaryLayer {
    let args: BailingMoeConfiguration
    let layerIdx: Int
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "gate") var gate: BailingMoeGate
    @ModuleInfo(key: "shared_experts") var sharedExperts: BailingMoeMLP?

    init(_ args: BailingMoeConfiguration, layerIdx: Int) {
        self.args = args
        self.layerIdx = layerIdx
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: args.hiddenSize, hiddenDims: args.moeIntermediateSize,
            numExperts: args.numExperts,
            bias: args.useBias
        )
        _gate.wrappedValue = BailingMoeGate(args)

        if args.numSharedExperts > 0 {
            let sharedDim =
                (args.moeSharedExpertIntermediateSize ?? args.moeIntermediateSize)
                * args.numSharedExperts
            _sharedExperts.wrappedValue = BailingMoeMLP(args, hiddenDims: sharedDim)
        } else {
            _sharedExperts.wrappedValue = nil
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (inds, weights) = gate.groupSelect(x)
        JangPressCanonicalExpertAdvisor.shared.observe(layer: layerIdx, indices: inds)
        var out = switchMLP(x, inds)
        out = (out * weights[.ellipsis, .newAxis]).sum(axis: -2)
        if let shared = sharedExperts {
            out = out + shared(x)
        }
        return out
    }
}

class BailingMoeTransformerBlock: Module {
    let args: BailingMoeConfiguration
    let layerIdx: Int

    @ModuleInfo(key: "attention") var attention: BailingMoeAttention
    @ModuleInfo(key: "mlp") var mlp: Module & UnaryLayer
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ args: BailingMoeConfiguration, layerIdx: Int) {
        self.args = args
        self.layerIdx = layerIdx

        _attention.wrappedValue = BailingMoeAttention(args)
        _inputLayerNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)

        if args.numExperts > 0 && layerIdx >= args.firstKDenseReplace {
            _mlp.wrappedValue = BailingMoeSparseMoeBlock(args, layerIdx: layerIdx)
        } else {
            _mlp.wrappedValue = BailingMoeMLP(args)
        }
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let r = attention(inputLayerNorm(x), mask: mask, cache: cache)
        let h = x + r
        let r2 = mlp(postAttentionLayerNorm(h))
        return h + r2
    }
}

public class BailingMoeModelInner: Module {
    @ModuleInfo(key: "word_embeddings") var embedTokens: Embedding
    let layers: [BailingMoeTransformerBlock]
    let norm: RMSNorm

    init(_ args: BailingMoeConfiguration) {
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)
        self.layers = (0 ..< args.hiddenLayers).map {
            BailingMoeTransformerBlock(args, layerIdx: $0)
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

public class BailingMoeModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]
    public let model: BailingMoeModelInner
    let configuration: BailingMoeConfiguration
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ args: BailingMoeConfiguration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
        self.model = BailingMoeModelInner(args)
        if !args.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let out = model(inputs, cache: cache)
        if let lmHead {
            return lmHead(out)
        } else {
            return model.embedTokens.asLinear(out)
        }
    }
}

extension BailingMoeModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
