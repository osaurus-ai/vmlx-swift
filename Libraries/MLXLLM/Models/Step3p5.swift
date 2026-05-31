//
// Step3p5 / Step3p7 text runtime.
//
// Port of `mlx_lm.models.step3p5`. Step 3.7 VLM bundles wrap the text
// decoder under `text_config` and `model.language_model.*`; this file
// intentionally implements the text path only. Vision patch/runtime proof is
// separate.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

private let step3p5RouterLock = NSLock()
private nonisolated(unsafe) var step3p5RouterCache:
    [Step3p5RouterKey: ([MLXArray]) -> [MLXArray]] = [:]

private struct Step3p5RouterKey: Hashable {
    let experts: Int
    let topK: Int
}

private func step3p5Router(experts: Int, topK: Int) -> ([MLXArray]) -> [MLXArray] {
    let key = Step3p5RouterKey(experts: experts, topK: topK)
    step3p5RouterLock.lock()
    defer { step3p5RouterLock.unlock() }
    if let cached = step3p5RouterCache[key] { return cached }
    let body: ([MLXArray]) -> [MLXArray] = { args in
        let gates = args[0].asType(.float32)
        let routerBias = args[1].asType(.float32)
        let scores = sigmoid(gates)
        let corrected = scores + routerBias
        let indices = argPartition(-corrected, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
        var weights = MLX.takeAlong(scores, indices, axis: -1)
        weights = weights / (weights.sum(axis: -1, keepDims: true) + MLXArray(1e-20))
        return [indices, weights]
    }
    let compiled = HardwareInfo.isCompiledDecodeSupported ? compile(shapeless: true, body) : body
    step3p5RouterCache[key] = compiled
    return compiled
}

private func step3p5ClampedSwiGLU(gate: MLXArray, up: MLXArray, limit: Float) -> MLXArray {
    guard limit > 0 else { return silu(gate) * up }
    return clip(silu(gate), max: limit) * clip(up, min: -limit, max: limit)
}

public struct Step3p5Configuration: Decodable, Sendable {
    public var modelType: String
    public var hiddenSize: Int
    public var numHiddenLayers: Int
    public var vocabSize: Int
    public var numAttentionHeads: Int
    public var numAttentionGroups: Int
    public var headDim: Int
    public var intermediateSize: Int
    public var rmsNormEps: Float
    public var ropeTheta: [Float]
    public var ropeScaling: [String: StringOrNumber]?
    public var maxPositionEmbeddings: Int
    public var slidingWindow: Int
    public var layerTypes: [String]
    public var yarnOnlyTypes: [String]
    public var partialRotaryFactors: [Float]
    public var useHeadWiseAttnGate: Bool
    public var moeNumExperts: Int
    public var moeTopK: Int
    public var moeIntermediateSize: Int
    public var shareExpertDim: Int
    public var moeLayersEnum: String?
    public var moeRouterScalingFactor: Float
    public var normExpertWeight: Bool
    public var swigluLimits: [Float]
    public var swigluLimitsShared: [Float]
    public var tieWordEmbeddings: Bool
    public var slidingNumAttentionHeads: Int?
    public var slidingNumAttentionGroups: Int?

    enum CodingKeys: String, CodingKey {
        case textConfig = "text_config"
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case vocabSize = "vocab_size"
        case numAttentionHeads = "num_attention_heads"
        case numAttentionGroups = "num_attention_groups"
        case headDim = "head_dim"
        case intermediateSize = "intermediate_size"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
        case maxPositionEmbeddings = "max_position_embeddings"
        case slidingWindow = "sliding_window"
        case layerTypes = "layer_types"
        case yarnOnlyTypes = "yarn_only_types"
        case partialRotaryFactors = "partial_rotary_factors"
        case attentionOtherSetting = "attention_other_setting"
        case useHeadWiseAttnGate = "use_head_wise_attn_gate"
        case moeNumExperts = "moe_num_experts"
        case moeTopK = "moe_top_k"
        case moeIntermediateSize = "moe_intermediate_size"
        case shareExpertDim = "share_expert_dim"
        case moeLayersEnum = "moe_layers_enum"
        case moeRouterScalingFactor = "moe_router_scaling_factor"
        case normExpertWeight = "norm_expert_weight"
        case swigluLimits = "swiglu_limits"
        case swigluLimitsShared = "swiglu_limits_shared"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    enum AttentionOtherKeys: String, CodingKey {
        case numAttentionHeads = "num_attention_heads"
        case numAttentionGroups = "num_attention_groups"
    }

    public init(from decoder: Decoder) throws {
        let root = try decoder.container(keyedBy: CodingKeys.self)
        let c: KeyedDecodingContainer<CodingKeys>
        if root.contains(.textConfig) {
            c = try root.nestedContainer(keyedBy: CodingKeys.self, forKey: .textConfig)
        } else {
            c = root
        }

        self.modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "step3p5"
        self.hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 4096
        self.numHiddenLayers = try c.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 45
        self.vocabSize = try c.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 128_896
        self.numAttentionHeads = try c.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 64
        self.numAttentionGroups = try c.decodeIfPresent(Int.self, forKey: .numAttentionGroups) ?? 8
        self.headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? 128
        self.intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 12_288
        self.rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5
        if let theta = try? c.decode([Float].self, forKey: .ropeTheta) {
            self.ropeTheta = theta
        } else {
            let scalar = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10_000
            self.ropeTheta = Array(repeating: scalar, count: numHiddenLayers)
        }
        self.ropeScaling = try c.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeScaling)
        self.maxPositionEmbeddings =
            try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 262_144
        self.slidingWindow = try c.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 512
        let decodedLayerTypes = try c.decodeIfPresent([String].self, forKey: .layerTypes) ?? []
        self.layerTypes = decodedLayerTypes.isEmpty
            ? (0..<numHiddenLayers).map { $0 % 4 == 0 ? "full_attention" : "sliding_attention" }
            : decodedLayerTypes
        self.yarnOnlyTypes = try c.decodeIfPresent([String].self, forKey: .yarnOnlyTypes) ?? []
        let partials = try c.decodeIfPresent([Float].self, forKey: .partialRotaryFactors) ?? []
        self.partialRotaryFactors = partials.isEmpty
            ? self.layerTypes.map { $0 == "full_attention" ? 0.5 : 1.0 }
            : partials
        self.useHeadWiseAttnGate =
            try c.decodeIfPresent(Bool.self, forKey: .useHeadWiseAttnGate) ?? true
        self.moeNumExperts = try c.decodeIfPresent(Int.self, forKey: .moeNumExperts) ?? 288
        self.moeTopK = try c.decodeIfPresent(Int.self, forKey: .moeTopK) ?? 8
        self.moeIntermediateSize =
            try c.decodeIfPresent(Int.self, forKey: .moeIntermediateSize) ?? 1280
        self.shareExpertDim = try c.decodeIfPresent(Int.self, forKey: .shareExpertDim) ?? 1280
        self.moeLayersEnum = try c.decodeIfPresent(String.self, forKey: .moeLayersEnum)
        self.moeRouterScalingFactor =
            try c.decodeIfPresent(Float.self, forKey: .moeRouterScalingFactor) ?? 3.0
        self.normExpertWeight =
            try c.decodeIfPresent(Bool.self, forKey: .normExpertWeight) ?? true
        self.swigluLimits = try c.decodeIfPresent([Float].self, forKey: .swigluLimits)
            ?? Array(repeating: 0, count: numHiddenLayers)
        self.swigluLimitsShared =
            try c.decodeIfPresent([Float].self, forKey: .swigluLimitsShared)
            ?? Array(repeating: 0, count: numHiddenLayers)
        self.tieWordEmbeddings =
            try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false

        if c.contains(.attentionOtherSetting),
            let other = try? c.nestedContainer(
                keyedBy: AttentionOtherKeys.self, forKey: .attentionOtherSetting)
        {
            self.slidingNumAttentionHeads =
                try other.decodeIfPresent(Int.self, forKey: .numAttentionHeads)
            self.slidingNumAttentionGroups =
                try other.decodeIfPresent(Int.self, forKey: .numAttentionGroups)
        } else {
            self.slidingNumAttentionHeads = nil
            self.slidingNumAttentionGroups = nil
        }
    }

    func isMoELayer(_ index: Int) -> Bool {
        guard let moeLayersEnum, !moeLayersEnum.isEmpty else {
            return index >= 1
        }
        return Set(moeLayersEnum.split(separator: ",").compactMap { Int($0) }).contains(index)
    }
}

private protocol Step3p5MLPLayer: Module {
    func callAsFunction(_ x: MLXArray) -> MLXArray
}

private final class Step3p5DenseMLP: Module, Step3p5MLPLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear
    let limit: Float

    init(hidden: Int, intermediate: Int, limit: Float) {
        self.limit = limit
        self._gateProj.wrappedValue = Linear(hidden, intermediate, bias: false)
        self._upProj.wrappedValue = Linear(hidden, intermediate, bias: false)
        self._downProj.wrappedValue = Linear(intermediate, hidden, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(step3p5ClampedSwiGLU(gate: gateProj(x), up: upProj(x), limit: limit))
    }
}

private protocol Step3p5SwitchLayer: Module {
    func callAsFunction(_ x: MLXArray, _ indices: MLXArray) -> MLXArray
}

extension SwitchGLU: Step3p5SwitchLayer {}
extension TurboQuantSwitchGLU: Step3p5SwitchLayer {}

public struct Step3p5JANGTQContext: Sendable {
    public let gateUpBits: Int
    public let downBits: Int
    public let seed: Int

    public init(gateUpBits: Int, downBits: Int, seed: Int) {
        self.gateUpBits = gateUpBits
        self.downBits = downBits
        self.seed = seed
    }
}

private final class Step3p5MoE: Module, Step3p5MLPLayer {
    let cfg: Step3p5Configuration
    let layerIndex: Int
    @ModuleInfo(key: "gate") var gate: Linear
    @ParameterInfo(key: "router_bias") var routerBias: MLXArray
    @ModuleInfo(key: "switch_mlp") var switchMLP: Module
    @ModuleInfo(key: "share_expert") var shareExpert: Step3p5DenseMLP

    init(_ cfg: Step3p5Configuration, layerIndex: Int, jangtq: Step3p5JANGTQContext?) {
        self.cfg = cfg
        self.layerIndex = layerIndex
        self._gate.wrappedValue = Linear(cfg.hiddenSize, cfg.moeNumExperts, bias: false)
        self._routerBias.wrappedValue = MLXArray.zeros([cfg.moeNumExperts])
        let limit = layerIndex < cfg.swigluLimits.count ? cfg.swigluLimits[layerIndex] : 0
        if let jangtq {
            self._switchMLP.wrappedValue = TurboQuantSwitchGLU(
                inputDims: cfg.hiddenSize,
                hiddenDims: cfg.moeIntermediateSize,
                numExperts: cfg.moeNumExperts,
                gateUpBits: jangtq.gateUpBits,
                downBits: jangtq.downBits,
                seed: jangtq.seed,
                swigluLimit: limit)
        } else {
            self._switchMLP.wrappedValue = SwitchGLU(
                inputDims: cfg.hiddenSize,
                hiddenDims: cfg.moeIntermediateSize,
                numExperts: cfg.moeNumExperts,
                glue: { gate, up in
                    step3p5ClampedSwiGLU(gate: gate, up: up, limit: limit)
                })
        }
        let sharedLimit = layerIndex < cfg.swigluLimitsShared.count
            ? cfg.swigluLimitsShared[layerIndex] : 0
        self._shareExpert.wrappedValue = Step3p5DenseMLP(
            hidden: cfg.hiddenSize,
            intermediate: cfg.shareExpertDim,
            limit: sharedLimit)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let routed = step3p5Router(experts: cfg.moeNumExperts, topK: cfg.moeTopK)([
            gate(x), routerBias,
        ])
        let indices = routed[0]
        let weights = routed[1] * MLXArray(cfg.moeRouterScalingFactor)
        guard let switchLayer = switchMLP as? Step3p5SwitchLayer else {
            fatalError("Step3p5MoE.switch_mlp has unsupported type \(type(of: switchMLP))")
        }
        if let streaming = switchLayer as? StreamingTurboQuantSwitchGLU {
            return (streaming.reduced(x, indices: indices, scores: weights) + shareExpert(x))
                .asType(x.dtype)
        }
        let yK = switchLayer(x, indices)
        let y = (yK * weights.expandedDimensions(axis: -1).asType(yK.dtype)).sum(axis: -2)
        return (y + shareExpert(x)).asType(x.dtype)
    }
}

private final class Step3p5Attention: Module {
    let cfg: Step3p5Configuration
    let layerIndex: Int
    let layerType: String
    let isSliding: Bool
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let scale: Float
    let ropeDim: Int
    let rope: RoPELayer

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm
    @ModuleInfo(key: "g_proj") var gProj: Linear?

    init(_ cfg: Step3p5Configuration, layerIndex: Int) {
        self.cfg = cfg
        self.layerIndex = layerIndex
        self.layerType = cfg.layerTypes[layerIndex]
        self.isSliding = layerType == "sliding_attention"
        self.numHeads = isSliding
            ? (cfg.slidingNumAttentionHeads ?? cfg.numAttentionHeads)
            : cfg.numAttentionHeads
        self.numKVHeads = isSliding
            ? (cfg.slidingNumAttentionGroups ?? cfg.numAttentionGroups)
            : cfg.numAttentionGroups
        self.headDim = cfg.headDim
        self.scale = pow(Float(headDim), -0.5)
        let partial = layerIndex < cfg.partialRotaryFactors.count
            ? cfg.partialRotaryFactors[layerIndex] : 1.0
        self.ropeDim = Int(Float(headDim) * partial)
        let theta = layerIndex < cfg.ropeTheta.count ? cfg.ropeTheta[layerIndex] : cfg.ropeTheta[0]
        let scaling = cfg.yarnOnlyTypes.isEmpty || cfg.yarnOnlyTypes.contains(layerType)
            ? cfg.ropeScaling : nil
        self.rope = initializeRope(
            dims: ropeDim,
            base: theta,
            traditional: false,
            scalingConfig: scaling,
            maxPositionEmbeddings: cfg.maxPositionEmbeddings)
        self._qProj.wrappedValue = Linear(
            cfg.hiddenSize, numHeads * headDim, bias: false)
        self._kProj.wrappedValue = Linear(
            cfg.hiddenSize, numKVHeads * headDim, bias: false)
        self._vProj.wrappedValue = Linear(
            cfg.hiddenSize, numKVHeads * headDim, bias: false)
        self._oProj.wrappedValue = Linear(numHeads * headDim, cfg.hiddenSize, bias: false)
        self._qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: cfg.rmsNormEps)
        self._kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: cfg.rmsNormEps)
        if cfg.useHeadWiseAttnGate {
            self._gProj.wrappedValue = Linear(cfg.hiddenSize, numHeads, bias: false)
        }
    }

    private func applyPartialRope(_ x: MLXArray, offset: Int) -> MLXArray {
        if ropeDim == headDim { return rope(x, offset: offset) }
        let rotated = rope(x[.ellipsis, ..<ropeDim], offset: offset)
        return MLX.concatenated([rotated, x[.ellipsis, ropeDim...]], axis: -1)
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)
        var q = qProj(x).reshaped(B, L, numHeads, headDim).transposed(0, 2, 1, 3)
        var k = kProj(x).reshaped(B, L, numKVHeads, headDim).transposed(0, 2, 1, 3)
        let v = vProj(x).reshaped(B, L, numKVHeads, headDim).transposed(0, 2, 1, 3)
        q = qNorm(q)
        k = kNorm(k)
        q = applyPartialRope(q, offset: cache?.offset ?? 0)
        k = applyPartialRope(k, offset: cache?.offset ?? 0)
        var out = attentionWithCacheUpdate(
            queries: q, keys: k, values: v,
            cache: cache, scale: scale, mask: mask)
            .transposed(0, 2, 1, 3)
        if let gProj {
            out = out * sigmoid(gProj(x)).expandedDimensions(axis: -1)
        }
        return oProj(out.reshaped(B, L, numHeads * headDim))
    }
}

private final class Step3p5Layer: Module {
    @ModuleInfo(key: "input_layernorm") var inputLayernorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayernorm: RMSNorm
    @ModuleInfo(key: "self_attn") var attention: Step3p5Attention
    let mlp: Step3p5MLPLayer
    let isSliding: Bool

    init(_ cfg: Step3p5Configuration, layerIndex: Int, jangtq: Step3p5JANGTQContext?) {
        self._inputLayernorm.wrappedValue = RMSNorm(
            dimensions: cfg.hiddenSize, eps: cfg.rmsNormEps)
        self._postAttentionLayernorm.wrappedValue = RMSNorm(
            dimensions: cfg.hiddenSize, eps: cfg.rmsNormEps)
        let attn = Step3p5Attention(cfg, layerIndex: layerIndex)
        self.isSliding = attn.isSliding
        self._attention.wrappedValue = attn
        if cfg.isMoELayer(layerIndex) {
            self.mlp = Step3p5MoE(cfg, layerIndex: layerIndex, jangtq: jangtq)
        } else {
            let limit = layerIndex < cfg.swigluLimitsShared.count
                ? cfg.swigluLimitsShared[layerIndex] : 0
            self.mlp = Step3p5DenseMLP(
                hidden: cfg.hiddenSize,
                intermediate: cfg.intermediateSize,
                limit: limit)
        }
    }

    func callAsFunction(
        _ x: MLXArray,
        fullMask: MLXFast.ScaledDotProductAttentionMaskMode,
        slidingMask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let mask = isSliding ? slidingMask : fullMask
        let h = x + attention(inputLayernorm(x), mask: mask, cache: cache)
        return h + mlp(postAttentionLayernorm(h))
    }
}

public final class Step3p5Model: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]
    let cfg: Step3p5Configuration

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    fileprivate let layers: [Step3p5Layer]
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ cfg: Step3p5Configuration, jangtq: Step3p5JANGTQContext? = nil) {
        self.cfg = cfg
        self.vocabularySize = cfg.vocabSize
        self.kvHeads = (0..<cfg.numHiddenLayers).map { idx in
            cfg.layerTypes[idx] == "sliding_attention"
                ? (cfg.slidingNumAttentionGroups ?? cfg.numAttentionGroups)
                : cfg.numAttentionGroups
        }
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: cfg.vocabSize, dimensions: cfg.hiddenSize)
        self.layers = (0..<cfg.numHiddenLayers).map {
            Step3p5Layer(cfg, layerIndex: $0, jangtq: jangtq)
        }
        self._norm.wrappedValue = RMSNorm(dimensions: cfg.hiddenSize, eps: cfg.rmsNormEps)
        if !cfg.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(cfg.hiddenSize, cfg.vocabSize, bias: false)
        }
    }

    public var loraLayers: [Module] { layers }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = embedTokens(inputs)
        let cacheArr = cache ?? []
        let fullIndex = layers.firstIndex { !$0.isSliding } ?? 0
        let slidingIndex = layers.firstIndex { $0.isSliding }
        let fullMask = createAttentionMask(
            h: h, cache: cacheArr.isEmpty ? nil : cacheArr[fullIndex])
        let slidingMask: MLXFast.ScaledDotProductAttentionMaskMode
        if let slidingIndex, !cacheArr.isEmpty {
            slidingMask = createAttentionMask(
                h: h, cache: cacheArr[slidingIndex], windowSize: cfg.slidingWindow)
        } else {
            slidingMask = .none
        }
        for (idx, layer) in layers.enumerated() {
            h = layer(
                h,
                fullMask: fullMask,
                slidingMask: slidingMask,
                cache: cacheArr.isEmpty ? nil : cacheArr[idx])
        }
        let out = norm(h)
        if cfg.tieWordEmbeddings {
            return embedTokens.asLinear(out)
        }
        return lmHead!(out)
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        Self.makeCache(
            layerTypes: cfg.layerTypes,
            slidingWindow: cfg.slidingWindow,
            maxPositionEmbeddings: cfg.maxPositionEmbeddings,
            parameters: parameters)
    }

    static func makeCache(
        layerTypes: [String],
        slidingWindow: Int,
        maxPositionEmbeddings: Int,
        parameters: GenerateParameters?
    ) -> [KVCache] {
        let callerWantsTQ: Bool = {
            guard let p = parameters else { return false }
            if case .turboQuant = p.kvMode { return true }
            return false
        }()
        let fullKVSize = parameters?.maxKVSize ?? maxPositionEmbeddings
        return layerTypes.map { layerType in
            if layerType == "sliding_attention" {
                return RotatingKVCache(maxSize: slidingWindow)
            }
            if callerWantsTQ {
                return KVCacheSimple()
            }
            if parameters?.maxKVSize != nil {
                return RotatingKVCache(maxSize: fullKVSize, keep: 4)
            }
            return KVCacheSimple()
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        let remaps = [
            (".moe.gate_proj.", ".mlp.switch_mlp.gate_proj."),
            (".moe.up_proj.", ".mlp.switch_mlp.up_proj."),
            (".moe.down_proj.", ".mlp.switch_mlp.down_proj."),
            (".moe.gate.", ".mlp.gate."),
            (".moe.router_bias", ".mlp.router_bias"),
            (".share_expert.", ".mlp.share_expert."),
        ]
        let isVanilla = weights.keys.contains { key in
            remaps.contains { key.contains($0.0) && !key.contains($0.1) }
        }
        var out: [String: MLXArray] = [:]
        for (key, value) in weights {
            if key.contains(".mtp") { continue }
            if key.hasPrefix("model.vision_model.")
                || key.hasPrefix("model.vit_large_projector.")
                || key.hasPrefix("vision_model.")
                || key.hasPrefix("vit_large_projector.")
            {
                continue
            }
            var k = key
            if k.hasPrefix("model.language_model.") {
                k = "model." + String(k.dropFirst("model.language_model.".count))
            }
            if k.hasPrefix("model.") {
                k = String(k.dropFirst("model.".count))
            }
            if k.hasSuffix(".self_attn.k_proj.k_scale")
                || k.hasSuffix(".self_attn.v_proj.v_scale")
                || k.hasSuffix(".tq_bits")
            {
                continue
            }
            if k.contains("self_attn.rotary_emb.inv_freq") { continue }
            if cfg.tieWordEmbeddings && k == "lm_head.weight" { continue }
            let parts = k.split(separator: ".")
            if parts.count > 2, parts[0] == "layers", let layer = Int(parts[1]),
                layer >= cfg.numHiddenLayers
            {
                continue
            }
            for (src, dst) in remaps where k.contains(src) && !k.contains(dst) {
                k = k.replacingOccurrences(of: src, with: dst)
                break
            }
            if isVanilla && k.hasSuffix(".weight") && k.contains("norm") {
                out[k] = value + 1
            } else {
                out[k] = value
            }
        }
        return out
    }
}
