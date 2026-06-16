//
//  MiMoV2Flash.swift
//  LLM
//
//  Port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/mimo_v2_flash.py
//  Created by Ronald Mannak on 2025/1/8.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

private func attentionWithCacheUpdateAndSinks(
    queries: MLXArray,
    keys: MLXArray,
    values: MLXArray,
    cache: KVCache?,
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode = .none,
    sinks: MLXArray? = nil
) -> MLXArray {
    guard let cache else {
        return MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: mask,
            sinks: sinks
        )
    }

    if let quantizedKVCache = cache as? QuantizedKVCacheProtocol {
        if let sinks {
            if let quantizedCache = cache as? QuantizedKVCache {
                let floatCache = quantizedCache.toUnquantized()
                let (cachedKeys, cachedValues) = floatCache.update(keys: keys, values: values)
                let refreshed = floatCache.toQuantized(
                    groupSize: quantizedCache.groupSize, bits: quantizedCache.bits)
                quantizedCache.state = refreshed.state
                quantizedCache.metaState = refreshed.metaState
                return MLXFast.scaledDotProductAttention(
                    queries: queries,
                    keys: cachedKeys,
                    values: cachedValues,
                    scale: scale,
                    mask: mask,
                    sinks: sinks
                )
            }
            let (cachedKeys, cachedValues) = cache.update(keys: keys, values: values)
            return MLXFast.scaledDotProductAttention(
                queries: queries,
                keys: cachedKeys,
                values: cachedValues,
                scale: scale,
                mask: mask,
                sinks: sinks
            )
        }
        let (quantizedKeys, quantizedValues) = quantizedKVCache.updateQuantized(
            keys: keys, values: values)
        return quantizedScaledDotProductAttention(
            queries: queries,
            quantizedKeys: quantizedKeys,
            quantizedValues: quantizedValues,
            scale: scale,
            mask: mask,
            groupSize: quantizedKVCache.groupSize,
            bits: quantizedKVCache.bits,
            mode: quantizedKVCache.mode
        )
    } else {
        let (cachedKeys, cachedValues) = cache.update(keys: keys, values: values)
        return MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: cachedKeys,
            values: cachedValues,
            scale: scale,
            mask: mask,
            sinks: sinks
        )
    }
}

private func groupExpertSelect(
    gates: MLXArray,
    eScoreCorrectionBias: MLXArray,
    topK: Int,
    nGroup: Int,
    topkGroup: Int,
    routedScalingFactor: Float,
    normTopkProb: Bool
) -> (MLXArray, MLXArray) {
    var scores = sigmoid(gates)
    let originalScores = scores
    scores = scores + eScoreCorrectionBias

    if nGroup > 1 {
        scores = unflatten(scores, axis: -1, shape: [nGroup, -1])
        let groupScores = top(scores, k: 2, axis: -1).sum(axis: -1, keepDims: true)
        let k = nGroup - topkGroup
        let groupIdx = argPartition(groupScores, kth: k - 1, axis: -2)[.ellipsis, ..<k, 0...]
        scores = putAlong(
            scores,
            stopGradient(groupIdx),
            values: MLXArray(0.0, dtype: scores.dtype),
            axis: -2
        )
        scores = flattened(scores, start: -2, end: -1)
    }

    let k = topK
    let inds = argPartition(-scores, kth: k - 1, axis: -1)[.ellipsis, ..<k]
    scores = takeAlong(originalScores, inds, axis: -1)
    if topK > 1, normTopkProb {
        let denominator = scores.sum(axis: -1, keepDims: true)
        scores = scores / (denominator + MLXArray(1e-20, dtype: scores.dtype))
    }
    scores = scores * routedScalingFactor

    return (inds, scores)
}

class MiMoV2FlashAttention: Module {
    let args: MiMoV2FlashConfiguration
    let isSlidingWindow: Bool
    let hasSinks: Bool
    let scale: Float

    let numAttentionHeads: Int
    let numKeyValueHeads: Int
    let headDim: Int
    let vHeadDim: Int

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear
    @ParameterInfo(key: "attention_sink_bias") var attentionSinkBias: MLXArray

    let rope: RoPE

    init(_ args: MiMoV2FlashConfiguration, isSlidingWindow: Bool) {
        self.args = args
        self.isSlidingWindow = isSlidingWindow

        if isSlidingWindow {
            self.numAttentionHeads = args.swaAttentionHeads
            self.numKeyValueHeads = args.swaKvHeads
            self.hasSinks = args.addSwaAttentionSinkBias
            self.headDim = args.swaHeadDim
            self.vHeadDim = args.swaVHeadDim
        } else {
            self.numAttentionHeads = args.attentionHeads
            self.numKeyValueHeads = args.kvHeads
            self.hasSinks = args.addFullAttentionSinkBias
            self.headDim = args.headDim
            self.vHeadDim = args.vHeadDim
        }

        self.scale = pow(Float(headDim), -0.5)

        _wq.wrappedValue = Linear(
            args.hiddenSize, numAttentionHeads * headDim, bias: false)
        _wk.wrappedValue = Linear(
            args.hiddenSize, numKeyValueHeads * headDim, bias: false)
        _wv.wrappedValue = Linear(
            args.hiddenSize, numKeyValueHeads * vHeadDim, bias: false)
        _wo.wrappedValue = Linear(
            numAttentionHeads * vHeadDim, args.hiddenSize, bias: false)

        _attentionSinkBias.wrappedValue = MLXArray.ones([numAttentionHeads])

        let ropeTheta = isSlidingWindow ? args.swaRopeTheta : args.ropeTheta
        let rotaryDims = Int(Float(args.partialRotaryFactor) * Float(headDim))
        self.rope = RoPE(
            dimensions: rotaryDims,
            traditional: false,
            base: ropeTheta
        )
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        let queries = wq(x)
        let keys = wk(x)
        let values = wv(x) * MLXArray(args.attentionValueScale ?? 1.0, dtype: x.dtype)

        let localAttentionHeads = queries.dim(-1) / headDim
        let localKeyValueHeads = keys.dim(-1) / headDim
        let localValueHeads = values.dim(-1) / vHeadDim

        var q = queries.reshaped(B, L, localAttentionHeads, -1).transposed(0, 2, 1, 3)
        var k = keys.reshaped(B, L, localKeyValueHeads, -1).transposed(0, 2, 1, 3)
        let v = values.reshaped(B, L, localValueHeads, -1).transposed(0, 2, 1, 3)

        q = applyRotaryPosition(rope, to: q, cache: cache)
        k = applyRotaryPosition(rope, to: k, cache: cache)

        let sinks: MLXArray?
        if hasSinks {
            sinks = attentionSinkBias
        } else {
            sinks = nil
        }

        let output = attentionWithCacheUpdateAndSinks(
            queries: q,
            keys: k,
            values: v,
            cache: cache,
            scale: scale,
            mask: mask,
            sinks: sinks
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return wo(output)
    }

    override func updateMissing(
        parameter: String,
        verify: VerifyUpdate,
        path: [String],
        modulePath: [String]
    ) throws {
        if parameter == "attention_sink_bias", hasSinks {
            // Keep the default you already set in init (ones([numAttentionHeads]))
            return
        }
        try super.updateMissing(
            parameter: parameter, verify: verify, path: path, modulePath: modulePath)
    }
}

class MiMoV2FlashMLP: Module, UnaryLayer {
    let hiddenSize: Int
    let intermediateSize: Int

    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(_ config: MiMoV2FlashConfiguration, hiddenSize: Int? = nil, intermediateSize: Int? = nil) {
        self.hiddenSize = hiddenSize ?? config.hiddenSize
        self.intermediateSize = intermediateSize ?? config.intermediateSize

        _gateProj.wrappedValue = Linear(self.hiddenSize, self.intermediateSize, bias: false)
        _upProj.wrappedValue = Linear(self.hiddenSize, self.intermediateSize, bias: false)
        _downProj.wrappedValue = Linear(self.intermediateSize, self.hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

class MiMoV2FlashMoEGate: Module {
    let topK: Int
    let normTopkProb: Bool
    let nRoutedExperts: Int
    let routedScalingFactor: Float
    let nGroup: Int
    let topkGroup: Int

    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "e_score_correction_bias") var eScoreCorrectionBias: MLXArray

    init(_ config: MiMoV2FlashConfiguration) {
        let nRoutedExperts = config.nRoutedExperts


        self.topK = config.numExpertsPerTok
        self.normTopkProb = config.normTopkProb
        self.nRoutedExperts = nRoutedExperts
        self.routedScalingFactor = config.routedScalingFactor ?? 1.0
        self.nGroup = config.nGroup
        self.topkGroup = config.topkGroup

        _weight.wrappedValue = MLXArray.zeros([nRoutedExperts, config.hiddenSize])
        _eScoreCorrectionBias.wrappedValue = MLXArray.zeros([nRoutedExperts])

        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        return groupExpertSelect(
            gates: x.matmul(weight.T),
            eScoreCorrectionBias: eScoreCorrectionBias,
            topK: topK,
            nGroup: nGroup,
            topkGroup: topkGroup,
            routedScalingFactor: routedScalingFactor,
            normTopkProb: normTopkProb
        )
    }
}

class MiMoV2FlashMoE: Module, UnaryLayer {
    let layerIdx: Int
    let numExpertsPerTok: Int
    let gate: MiMoV2FlashMoEGate

    @ModuleInfo(key: "switch_mlp") var switchMLP: Module & SwitchGLULayer
    @ModuleInfo(key: "shared_experts") var sharedExperts: MiMoV2FlashMLP?

    init(_ config: MiMoV2FlashConfiguration, layerIdx: Int) {
        let nRoutedExperts = config.nRoutedExperts

        self.layerIdx = layerIdx
        self.numExpertsPerTok = config.numExpertsPerTok
        self.gate = MiMoV2FlashMoEGate(config)

        if config.usesTurboQuantRoutedExperts {
            let gate = config.routedExpertQuantization(layerIndex: layerIdx, projection: "gate_proj")
            let down = config.routedExpertQuantization(layerIndex: layerIdx, projection: "down_proj")
            _switchMLP.wrappedValue = TurboQuantSwitchGLU(
                inputDims: config.hiddenSize,
                hiddenDims: config.moeIntermediateSize,
                numExperts: nRoutedExperts,
                gateUpBits: gate.bits,
                downBits: down.bits,
                seed: config.mxtqSeed
            )
        } else {
            _switchMLP.wrappedValue = SwitchGLU(
                inputDims: config.hiddenSize,
                hiddenDims: config.moeIntermediateSize,
                numExperts: nRoutedExperts
            )
        }

        if let shared = config.nSharedExperts {
            let intermediateSize = config.moeIntermediateSize * shared
            _sharedExperts.wrappedValue = MiMoV2FlashMLP(
                config, intermediateSize: intermediateSize)
        }

        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (inds, scores) = gate(x)
        JangPressCanonicalExpertAdvisor.shared.observe(layer: layerIdx, indices: inds)
        var y = switchMLP(x, inds)
        y = (y * scores[.ellipsis, .newAxis]).sum(axis: -2).asType(y.dtype)
        if let sharedExperts {
            y = y + sharedExperts(x)
        }
        return y
    }
}

class MiMoV2FlashDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: MiMoV2FlashAttention
    @ModuleInfo(key: "mlp") var mlp: Module & UnaryLayer
    let isSlidingWindow: Bool

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ config: MiMoV2FlashConfiguration, layerIdx: Int, isMoe: Bool, isSlidingWindow: Bool) {
        self.isSlidingWindow = isSlidingWindow
        _selfAttn.wrappedValue = MiMoV2FlashAttention(config, isSlidingWindow: isSlidingWindow)
        _mlp.wrappedValue = isMoe
            ? MiMoV2FlashMoE(config, layerIdx: layerIdx)
            : MiMoV2FlashMLP(config)
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.layernormEpsilon)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.layernormEpsilon)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let residual = x + selfAttn(inputLayerNorm(x), mask: mask, cache: cache)
        return residual + mlp(postAttentionLayerNorm(residual))
    }
}

public class MiMoV2FlashModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    let layers: [MiMoV2FlashDecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    let swaIdx: Int
    let gaIdx: Int
    let slidingWindowSize: Int
    let hybridLayerPattern: [Int]

    init(_ config: MiMoV2FlashConfiguration) {
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)

        self.layers = (0 ..< config.hiddenLayers).map { index in
            MiMoV2FlashDecoderLayer(
                config,
                layerIdx: index,
                isMoe: config.moeLayerFreq[index] == 1,
                isSlidingWindow: config.hybridLayerPattern[index] == 1
            )
        }
        _norm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.layernormEpsilon)
        self.swaIdx = config.hybridLayerPattern.firstIndex(of: 1) ?? 0
        self.gaIdx = config.hybridLayerPattern.firstIndex(of: 0) ?? 0
        self.slidingWindowSize = config.slidingWindowSize
        self.hybridLayerPattern = config.hybridLayerPattern
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = embedTokens(inputs)

        let fullMask = createAttentionMask(h: h, cache: cache?[gaIdx])
        let swaMask = createAttentionMask(
            h: h, cache: cache?[swaIdx], windowSize: slidingWindowSize)

        for (i, layer) in layers.enumerated() {
            let mask = hybridLayerPattern[i] == 1 ? swaMask : fullMask
            h = layer(h, mask: mask, cache: cache?[i])
        }

        return norm(h)
    }
}

public class MiMoV2FlashModel: Module, LLMModel, KVCacheDimensionProvider {
    public let modelType: String
    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: MiMoV2FlashModelInner
    let configuration: MiMoV2FlashConfiguration

    @ModuleInfo(key: "lm_head") var lmHead: Linear

    public init(_ config: MiMoV2FlashConfiguration) {
        self.configuration = config
        self.modelType = config.modelType
        self.vocabularySize = config.vocabularySize
        self.kvHeads = config.hybridLayerPattern.map {
            $0 == 1 ? config.swaKvHeads : config.kvHeads
        }
        self.model = MiMoV2FlashModelInner(config)
        _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let out = model(inputs, cache: cache)
        return lmHead(out)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        func expertPrefix(layer: Int, expert: Int, proj: String) -> String {
            "model.layers.\(layer).mlp.experts.\(expert).\(proj)"
        }

        func isRoutedExpertScaleKey(_ key: String) -> Bool {
            key.contains(".mlp.experts.") && key.hasSuffix(".weight_scale_inv")
        }

        func hasCompleteExpertSet(layer: Int, proj: String, suffixes: [String]) -> Bool {
            for expert in 0 ..< configuration.nRoutedExperts {
                for suffix in suffixes {
                    let key = "\(expertPrefix(layer: layer, expert: expert, proj: proj)).\(suffix)"
                    if sanitizedWeights[key] == nil {
                        return false
                    }
                }
            }
            return true
        }

        func dequant(weight: MLXArray, scaleInv: MLXArray) -> MLXArray {
            // MiMo source weights are FP8 E4M3 bytes, not integer uint8 values.
            let decodedWeight = weight.dtype == .uint8
                ? MLX.fromFP8(weight, dtype: .float32)
                : weight
            let dtype = decodedWeight.dtype
            let bs = 128
            let (m, n) = (decodedWeight.shape[0], decodedWeight.shape[1])
            let padBottom = bs * scaleInv.dim(0) - m
            let padSide = bs * scaleInv.dim(1) - n

            var paddedWeight = padded(
                decodedWeight, widths: [.init((0, padBottom)), .init((0, padSide))])
            paddedWeight = paddedWeight.reshaped(
                [(m + padBottom) / bs, bs, (n + padSide) / bs, bs])
            let scaled = paddedWeight * scaleInv[0..., .newAxis, 0..., .newAxis]
            return scaled.reshaped([m + padBottom, n + padSide])[0 ..< m, 0 ..< n]
                .asType(dtype)
        }

        var newWeights: [String: MLXArray] = [:]
        for (key, value) in weights {
            if key.contains("weight_scale_inv") {
                if isRoutedExpertScaleKey(key) {
                    newWeights[key] = value
                    continue
                }
                let weightKey = key.replacingOccurrences(of: "_scale_inv", with: "")
                if let weight = weights[weightKey] {
                    newWeights[weightKey] = dequant(weight: weight, scaleInv: value)
                }
            } else if newWeights[key] == nil {
                newWeights[key] = value
            }
        }

        var sanitizedWeights = newWeights.isEmpty ? weights : newWeights

        for key in Array(sanitizedWeights.keys) where key.hasSuffix(".tq_bits") {
            sanitizedWeights.removeValue(forKey: key)
        }
        for key in Array(sanitizedWeights.keys)
            where key.hasPrefix("audio_encoder.")
                || key.hasPrefix("encoder.")
                || key.hasPrefix("speech_embeddings.")
                || key.hasPrefix("visual.")
        {
            sanitizedWeights.removeValue(forKey: key)
        }

        for layerIndex in 0 ..< configuration.hiddenLayers {
            let prefix = "model.layers.\(layerIndex).self_attn"
            let isSliding = configuration.hybridLayerPattern[layerIndex] == 1
            let qRows =
                (isSliding ? configuration.swaAttentionHeads : configuration.attentionHeads)
                * (isSliding ? configuration.swaHeadDim : configuration.headDim)
            let kRows =
                (isSliding ? configuration.swaKvHeads : configuration.kvHeads)
                * (isSliding ? configuration.swaHeadDim : configuration.headDim)

            for suffix in ["weight", "scales", "biases"] {
                let fusedKey = "\(prefix).qkv_proj.\(suffix)"
                guard let fused = sanitizedWeights.removeValue(forKey: fusedKey) else {
                    continue
                }
                let qkv = split(fused, indices: [qRows, qRows + kRows], axis: 0)
                sanitizedWeights["\(prefix).q_proj.\(suffix)"] = qkv[0]
                sanitizedWeights["\(prefix).k_proj.\(suffix)"] = qkv[1]
                sanitizedWeights["\(prefix).v_proj.\(suffix)"] = qkv[2]
            }
        }

        for layerIndex in 0 ..< configuration.hiddenLayers {
            let prefix = "model.layers.\(layerIndex)"
            for (_, projName) in [("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")] {
                let firstPrefix = expertPrefix(layer: layerIndex, expert: 0, proj: projName)
                let firstScaleInvKey = "\(firstPrefix).weight_scale_inv"
                if sanitizedWeights["\(firstPrefix).weight"] != nil,
                   sanitizedWeights[firstScaleInvKey] != nil {
                    let quant = configuration.routedExpertQuantization(
                        layerIndex: layerIndex, projection: projName)
                    var weightParts: [MLXArray] = []
                    var scaleParts: [MLXArray] = []
                    var biasParts: [MLXArray] = []
                    guard hasCompleteExpertSet(
                        layer: layerIndex, proj: projName,
                        suffixes: ["weight", "weight_scale_inv"])
                    else {
                        continue
                    }

                    for expert in 0 ..< configuration.nRoutedExperts {
                        let expertBase = expertPrefix(layer: layerIndex, expert: expert, proj: projName)
                        let weightKey = "\(expertBase).weight"
                        let scaleInvKey = "\(expertBase).weight_scale_inv"
                        let weight = sanitizedWeights.removeValue(forKey: weightKey)!
                        let scaleInv = sanitizedWeights.removeValue(forKey: scaleInvKey)!
                        let dense = dequant(weight: weight, scaleInv: scaleInv).asType(.float32)
                        let (qWeight, scales, biases) = MLX.quantized(
                            dense, groupSize: quant.groupSize, bits: quant.bits, mode: .affine)
                        guard let biases else { continue }
                        weightParts.append(qWeight)
                        scaleParts.append(scales)
                        biasParts.append(biases)
                    }
                    guard weightParts.count == configuration.nRoutedExperts,
                        scaleParts.count == configuration.nRoutedExperts,
                        biasParts.count == configuration.nRoutedExperts
                    else {
                        continue
                    }
                    sanitizedWeights["\(prefix).mlp.switch_mlp.\(projName).weight"] =
                        MLX.stacked(weightParts)
                    sanitizedWeights["\(prefix).mlp.switch_mlp.\(projName).scales"] =
                        MLX.stacked(scaleParts)
                    sanitizedWeights["\(prefix).mlp.switch_mlp.\(projName).biases"] =
                        MLX.stacked(biasParts)
                    continue
                }

                for key in ["weight", "scales", "biases"] {
                    let firstKey = "\(firstPrefix).\(key)"
                    if sanitizedWeights[firstKey] != nil {
                        guard hasCompleteExpertSet(
                            layer: layerIndex, proj: projName, suffixes: [key])
                        else {
                            continue
                        }
                        let toJoin = (0 ..< configuration.nRoutedExperts).map {
                            sanitizedWeights.removeValue(
                                forKey: "\(expertPrefix(layer: layerIndex, expert: $0, proj: projName)).\(key)")!
                        }
                        sanitizedWeights["\(prefix).mlp.switch_mlp.\(projName).\(key)"] =
                            MLX.stacked(toJoin)
                    }
                }
            }
        }

        return sanitizedWeights.filter { key, _ in
            !key.hasPrefix("model.mtp")
        }
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        return model.layers.map { layer in
            if layer.isSlidingWindow {
                return RotatingKVCache(maxSize: configuration.slidingWindowSize)
            } else {
                return KVCacheSimple()
            }
        }
    }
}

// MARK: - Configuration

struct MiMoRoutedExpertBits: Codable, Sendable {
    let projections: [String: Int]

    init(scalar: Int) {
        self.projections = [
            "gate_proj": scalar,
            "up_proj": scalar,
            "down_proj": scalar,
        ]
    }

    init(projections: [String: Int]) {
        self.projections = projections
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let scalar = try? container.decode(Int.self) {
            self.init(scalar: scalar)
        } else {
        if let projections = try? container.decode([String: Int].self) {
            self.init(projections: projections)
        } else {
            let nested = try container.decode([String: [String: Int]].self)
            if let routed = nested["routed_expert"] ?? nested["routed_experts"] {
                self.init(projections: routed)
            } else if let first = nested.values.first {
                self.init(projections: first)
            } else {
                self.init(projections: [:])
            }
        }
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(projections)
    }

    func bits(for projection: String) -> Int? {
        projections[projection]
            ?? projections[projection.replacingOccurrences(of: "_proj", with: "")]
    }
}

struct MiMoRoutedExpertBitPlan: Codable, Sendable {
    var defaultBits: MiMoRoutedExpertBits?
    var layerOverrides: [String: MiMoRoutedExpertBits]?

    enum CodingKeys: String, CodingKey {
        case defaultBits = "default"
        case layerOverrides = "layer_overrides"
    }
}

public struct MiMoV2FlashConfiguration: Codable, Sendable {
    var modelType: String = "mimo_v2_flash"
    var numExpertsPerTok: Int
    var hybridLayerPattern: [Int]
    var moeLayerFreq: [Int]
    var addSwaAttentionSinkBias: Bool
    var addFullAttentionSinkBias: Bool
    var slidingWindowSize: Int
    var vocabularySize: Int
    var hiddenSize: Int
    var intermediateSize: Int
    var moeIntermediateSize: Int
    var hiddenLayers: Int
    var attentionHeads: Int
    var kvHeads: Int
    var nSharedExperts: Int?
    var nRoutedExperts: Int
    var routedScalingFactor: Float?
    var topkMethod: String
    var scoringFunc: String
    var normTopkProb: Bool
    var nGroup: Int
    var topkGroup: Int
    var maxPositionEmbeddings: Int
    var layernormEpsilon: Float
    var ropeTheta: Float
    var swaRopeTheta: Float
    var swaAttentionHeads: Int
    var swaKvHeads: Int
    var headDim: Int
    var vHeadDim: Int
    var swaHeadDim: Int
    var swaVHeadDim: Int
    var partialRotaryFactor: Float
    var attentionValueScale: Float?
    var routedExpertBits: MiMoRoutedExpertBits?
    var mxtqBits: MiMoRoutedExpertBits?
    var routedExpertBitPlan: MiMoRoutedExpertBitPlan?
    var routedExpertGroupSize: Int?
    var weightFormat: String?
    var mxtqSeed: Int

    var usesTurboQuantRoutedExperts: Bool {
        weightFormat?.lowercased() == "mxtq"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.modelType = try container.decodeIfPresent(String.self, forKey: .modelType)
            ?? "mimo_v2_flash"
        self.numExpertsPerTok = try container.decode(Int.self, forKey: .numExpertsPerTok)
        self.hybridLayerPattern = try container.decode([Int].self, forKey: .hybridLayerPattern)
        self.moeLayerFreq = try container.decode([Int].self, forKey: .moeLayerFreq)
        self.addSwaAttentionSinkBias = try container.decode(
            Bool.self, forKey: .addSwaAttentionSinkBias)
        self.addFullAttentionSinkBias = try container.decode(
            Bool.self, forKey: .addFullAttentionSinkBias)
        self.slidingWindowSize = try container.decode(Int.self, forKey: .slidingWindowSize)
        self.vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)
        self.hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        self.intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        self.moeIntermediateSize = try container.decode(Int.self, forKey: .moeIntermediateSize)
        self.hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        self.attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        self.kvHeads = try container.decode(Int.self, forKey: .kvHeads)
        self.nSharedExperts = try container.decodeIfPresent(Int.self, forKey: .nSharedExperts)
        self.nRoutedExperts = try container.decode(Int.self, forKey: .nRoutedExperts)
        self.routedScalingFactor = try container.decodeIfPresent(
            Float.self, forKey: .routedScalingFactor)
        self.topkMethod = try container.decode(String.self, forKey: .topkMethod)
        self.scoringFunc = try container.decode(String.self, forKey: .scoringFunc)
        self.normTopkProb = try container.decode(Bool.self, forKey: .normTopkProb)
        self.nGroup = try container.decode(Int.self, forKey: .nGroup)
        self.topkGroup = try container.decode(Int.self, forKey: .topkGroup)
        self.maxPositionEmbeddings = try container.decode(Int.self, forKey: .maxPositionEmbeddings)
        self.layernormEpsilon = try container.decode(Float.self, forKey: .layernormEpsilon)
        self.ropeTheta = try container.decode(Float.self, forKey: .ropeTheta)
        self.swaRopeTheta = try container.decode(Float.self, forKey: .swaRopeTheta)
        self.swaAttentionHeads = try container.decode(Int.self, forKey: .swaAttentionHeads)
        self.swaKvHeads = try container.decode(Int.self, forKey: .swaKvHeads)
        self.headDim = try container.decode(Int.self, forKey: .headDim)
        self.vHeadDim = try container.decode(Int.self, forKey: .vHeadDim)
        self.swaHeadDim = try container.decode(Int.self, forKey: .swaHeadDim)
        self.swaVHeadDim = try container.decode(Int.self, forKey: .swaVHeadDim)
        self.partialRotaryFactor = try container.decode(Float.self, forKey: .partialRotaryFactor)
        self.attentionValueScale = try container.decodeIfPresent(
            Float.self, forKey: .attentionValueScale)
        self.routedExpertBits = try container.decodeIfPresent(
            MiMoRoutedExpertBits.self, forKey: .routedExpertBits)
        self.mxtqBits = try container.decodeIfPresent(MiMoRoutedExpertBits.self, forKey: .mxtqBits)
        self.routedExpertBitPlan = try container.decodeIfPresent(
            MiMoRoutedExpertBitPlan.self, forKey: .routedExpertBitPlan)
        self.routedExpertGroupSize = try container.decodeIfPresent(
            Int.self, forKey: .routedExpertGroupSize)
        self.weightFormat = try container.decodeIfPresent(String.self, forKey: .weightFormat)
        self.mxtqSeed = try container.decodeIfPresent(Int.self, forKey: .mxtqSeed) ?? 42

        try Self.validatePositive(numExpertsPerTok, key: .numExpertsPerTok, in: container)
        try Self.validatePositive(slidingWindowSize, key: .slidingWindowSize, in: container)
        try Self.validatePositive(vocabularySize, key: .vocabularySize, in: container)
        try Self.validatePositive(hiddenSize, key: .hiddenSize, in: container)
        try Self.validatePositive(intermediateSize, key: .intermediateSize, in: container)
        try Self.validatePositive(moeIntermediateSize, key: .moeIntermediateSize, in: container)
        try Self.validatePositive(hiddenLayers, key: .hiddenLayers, in: container)
        try Self.validatePositive(attentionHeads, key: .attentionHeads, in: container)
        try Self.validatePositive(kvHeads, key: .kvHeads, in: container)
        try Self.validatePositive(nRoutedExperts, key: .nRoutedExperts, in: container)
        if let nSharedExperts {
            try Self.validateNonNegative(nSharedExperts, key: .nSharedExperts, in: container)
        }
        if let routedScalingFactor {
            try Self.validatePositive(routedScalingFactor, key: .routedScalingFactor, in: container)
        }
        try Self.validatePositive(nGroup, key: .nGroup, in: container)
        try Self.validatePositive(topkGroup, key: .topkGroup, in: container)
        try Self.validatePositive(maxPositionEmbeddings, key: .maxPositionEmbeddings, in: container)
        try Self.validatePositive(layernormEpsilon, key: .layernormEpsilon, in: container)
        try Self.validatePositive(ropeTheta, key: .ropeTheta, in: container)
        try Self.validatePositive(swaRopeTheta, key: .swaRopeTheta, in: container)
        try Self.validatePositive(swaAttentionHeads, key: .swaAttentionHeads, in: container)
        try Self.validatePositive(swaKvHeads, key: .swaKvHeads, in: container)
        try Self.validatePositive(headDim, key: .headDim, in: container)
        try Self.validatePositive(vHeadDim, key: .vHeadDim, in: container)
        try Self.validatePositive(swaHeadDim, key: .swaHeadDim, in: container)
        try Self.validatePositive(swaVHeadDim, key: .swaVHeadDim, in: container)
        try Self.validatePositive(partialRotaryFactor, key: .partialRotaryFactor, in: container)
        if let attentionValueScale {
            try Self.validatePositive(attentionValueScale, key: .attentionValueScale, in: container)
        }
        if let routedExpertGroupSize {
            try Self.validatePositive(
                routedExpertGroupSize, key: .routedExpertGroupSize, in: container)
        }
        try Self.validateRoutedExpertBits(routedExpertBits, key: .routedExpertBits, in: container)
        try Self.validateRoutedExpertBits(mxtqBits, key: .mxtqBits, in: container)
        try Self.validateRoutedExpertBitPlan(routedExpertBitPlan, key: .routedExpertBitPlan, in: container)
        try Self.validateNonNegative(mxtqSeed, key: .mxtqSeed, in: container)

        guard !modelType.isEmpty else {
            throw DecodingError.dataCorruptedError(
                forKey: .modelType,
                in: container,
                debugDescription: "MiMoV2Flash model_type must not be empty.")
        }
        if let weightFormat, !weightFormat.isEmpty, weightFormat.lowercased() != "mxtq" {
            throw DecodingError.dataCorruptedError(
                forKey: .weightFormat,
                in: container,
                debugDescription: "MiMoV2Flash weight_format must be mxtq when present.")
        }
        if weightFormat?.lowercased() == "mxtq" {
            for layerIndex in 0 ..< hiddenLayers where moeLayerFreq[layerIndex] == 1 {
                let gate = routedExpertQuantization(layerIndex: layerIndex, projection: "gate_proj")
                let up = routedExpertQuantization(layerIndex: layerIndex, projection: "up_proj")
                guard gate.bits == up.bits else {
                    throw DecodingError.dataCorruptedError(
                        forKey: .mxtqBits,
                        in: container,
                        debugDescription:
                            "MiMoV2Flash JANGTQ gate_proj and up_proj bits must match.")
                }
            }
        }
        guard hybridLayerPattern.count == hiddenLayers, moeLayerFreq.count == hiddenLayers else {
            throw DecodingError.dataCorruptedError(
                forKey: .hybridLayerPattern,
                in: container,
                debugDescription:
                    "MiMoV2Flash hybrid_layer_pattern and moe_layer_freq must match num_hidden_layers.")
        }
        guard hybridLayerPattern.allSatisfy({ $0 == 0 || $0 == 1 }),
            moeLayerFreq.allSatisfy({ $0 == 0 || $0 == 1 })
        else {
            throw DecodingError.dataCorruptedError(
                forKey: .hybridLayerPattern,
                in: container,
                debugDescription:
                    "MiMoV2Flash hybrid_layer_pattern and moe_layer_freq values must be 0 or 1.")
        }
        guard topkMethod == "noaux_tc" else {
            throw DecodingError.dataCorruptedError(
                forKey: .topkMethod,
                in: container,
                debugDescription: "MiMoV2Flash topk_method must be noaux_tc.")
        }
        guard scoringFunc == "sigmoid" else {
            throw DecodingError.dataCorruptedError(
                forKey: .scoringFunc,
                in: container,
                debugDescription: "MiMoV2Flash scoring_func must be sigmoid.")
        }
        guard attentionHeads % kvHeads == 0, swaAttentionHeads % swaKvHeads == 0 else {
            throw DecodingError.dataCorruptedError(
                forKey: .kvHeads,
                in: container,
                debugDescription:
                    "MiMoV2Flash attention heads must be divisible by KV heads.")
        }
        guard nRoutedExperts % nGroup == 0 else {
            throw DecodingError.dataCorruptedError(
                forKey: .nGroup,
                in: container,
                debugDescription:
                    "MiMoV2Flash n_routed_experts must be divisible by n_group.")
        }
        guard topkGroup <= nGroup else {
            throw DecodingError.dataCorruptedError(
                forKey: .topkGroup,
                in: container,
                debugDescription:
                    "MiMoV2Flash topk_group must be less than or equal to n_group.")
        }
        guard numExpertsPerTok <= nRoutedExperts else {
            throw DecodingError.dataCorruptedError(
                forKey: .numExpertsPerTok,
                in: container,
                debugDescription:
                    "MiMoV2Flash num_experts_per_tok must be less than or equal to n_routed_experts.")
        }
        let rotaryDim = Int(Float(headDim) * partialRotaryFactor)
        let swaRotaryDim = Int(Float(swaHeadDim) * partialRotaryFactor)
        guard rotaryDim > 0, rotaryDim <= headDim, swaRotaryDim > 0,
            swaRotaryDim <= swaHeadDim
        else {
            throw DecodingError.dataCorruptedError(
                forKey: .partialRotaryFactor,
                in: container,
                debugDescription:
                    "MiMoV2Flash rotary dimensions must be positive and no larger than the head dimensions.")
        }
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case numExpertsPerTok = "num_experts_per_tok"
        case hybridLayerPattern = "hybrid_layer_pattern"
        case moeLayerFreq = "moe_layer_freq"
        case addSwaAttentionSinkBias = "add_swa_attention_sink_bias"
        case addFullAttentionSinkBias = "add_full_attention_sink_bias"
        case slidingWindowSize = "sliding_window_size"
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case nSharedExperts = "n_shared_experts"
        case nRoutedExperts = "n_routed_experts"
        case routedScalingFactor = "routed_scaling_factor"
        case topkMethod = "topk_method"
        case scoringFunc = "scoring_func"
        case normTopkProb = "norm_topk_prob"
        case nGroup = "n_group"
        case topkGroup = "topk_group"
        case maxPositionEmbeddings = "max_position_embeddings"
        case layernormEpsilon = "layernorm_epsilon"
        case ropeTheta = "rope_theta"
        case swaRopeTheta = "swa_rope_theta"
        case swaAttentionHeads = "swa_num_attention_heads"
        case swaKvHeads = "swa_num_key_value_heads"
        case headDim = "head_dim"
        case vHeadDim = "v_head_dim"
        case swaHeadDim = "swa_head_dim"
        case swaVHeadDim = "swa_v_head_dim"
        case partialRotaryFactor = "partial_rotary_factor"
        case attentionValueScale = "attention_value_scale"
        case routedExpertBits = "routed_expert_bits"
        case mxtqBits = "mxtq_bits"
        case routedExpertBitPlan = "routed_expert_bit_plan"
        case routedExpertGroupSize = "routed_expert_group_size"
        case weightFormat = "weight_format"
        case mxtqSeed = "mxtq_seed"
    }

    func routedExpertQuantization(layerIndex: Int, projection: String) -> (bits: Int, groupSize: Int) {
        let env = ProcessInfo.processInfo.environment
        let groupSize = Int(env["TP_MIMO_ROUTED_EXPERT_GROUP_SIZE"] ?? "")
            ?? routedExpertGroupSize
            ?? 128
        let layerEnv = env["TP_MIMO_ROUTED_EXPERT_BITS_LAYER_\(layerIndex)"]
        let envBits = Self.bits(from: layerEnv, projection: projection)
            ?? Self.bits(from: env["TP_MIMO_ROUTED_EXPERT_BITS"], projection: projection)
        let plannedBits = routedExpertBitPlan?.layerOverrides?["\(layerIndex)"]?.bits(for: projection)
            ?? routedExpertBitPlan?.defaultBits?.bits(for: projection)
        let configBits = usesTurboQuantRoutedExperts
            ? (mxtqBits?.bits(for: projection) ?? routedExpertBits?.bits(for: projection))
            : (routedExpertBits?.bits(for: projection) ?? mxtqBits?.bits(for: projection))
        let bits = envBits ?? plannedBits ?? configBits ?? 4
        return (bits, groupSize)
    }

    private static func bits(from raw: String?, projection: String) -> Int? {
        guard let raw = raw?.trimmingCharacters(in: .whitespacesAndNewlines), !raw.isEmpty else {
            return nil
        }
        if let scalar = Int(raw) {
            return scalar
        }
        for part in raw.split(separator: ",") {
            let pair = part.split(whereSeparator: { $0 == "=" || $0 == ":" })
            guard pair.count == 2 else { continue }
            let key = pair[0].trimmingCharacters(in: .whitespacesAndNewlines)
            let value = pair[1].trimmingCharacters(in: .whitespacesAndNewlines)
            if key == projection || key == projection.replacingOccurrences(of: "_proj", with: "") {
                return Int(value)
            }
        }
        return nil
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

    private static func validateNonNegative<K: CodingKey>(
        _ value: Int, key: K, in container: KeyedDecodingContainer<K>
    ) throws {
        guard value >= 0 else {
            throw DecodingError.dataCorruptedError(
                forKey: key,
                in: container,
                debugDescription: "\(key.stringValue) must be non-negative.")
        }
    }

    private static func validateRoutedExpertBits<K: CodingKey>(
        _ value: MiMoRoutedExpertBits?, key: K, in container: KeyedDecodingContainer<K>
    ) throws {
        guard let value else { return }
        for (projection, bits) in value.projections {
            guard ["gate_proj", "up_proj", "down_proj", "gate", "up", "down"].contains(projection),
                [2, 3, 4, 5, 6, 8].contains(bits)
            else {
                throw DecodingError.dataCorruptedError(
                    forKey: key,
                    in: container,
                    debugDescription:
                        "MiMoV2Flash routed expert bits must use supported projections and bit widths.")
            }
        }
    }

    private static func validateRoutedExpertBitPlan<K: CodingKey>(
        _ value: MiMoRoutedExpertBitPlan?, key: K, in container: KeyedDecodingContainer<K>
    ) throws {
        guard let value else { return }
        try validateRoutedExpertBits(value.defaultBits, key: key, in: container)
        for (_, bits) in value.layerOverrides ?? [:] {
            try validateRoutedExpertBits(bits, key: key, in: container)
        }
    }
}

// MARK: - LoRA

extension MiMoV2FlashModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
