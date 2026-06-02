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

private enum MiMoV2FlashTrace {
    private static func enabled(_ key: String) -> Bool {
        switch ProcessInfo.processInfo.environment[key]?.lowercased() {
        case "1", "true", "yes", "on":
            return true
        default:
            return false
        }
    }

    static var layerTraceEnabled: Bool {
        enabled("VMLINUX_MIMO_LAYER_TRACE") || enabled("TP_MIMO_LAYER_TRACE")
    }

    static var forceLayerEval: Bool {
        enabled("VMLINUX_MIMO_LAYER_EVAL") || enabled("TP_MIMO_LAYER_EVAL")
    }

    static var eagerLayerEval: Bool {
        enabled("VMLINUX_MIMO_EAGER_LAYERS") || enabled("TP_MIMO_EAGER_LAYERS")
    }

    static var clearCacheAfterLayerEval: Bool {
        enabled("VMLINUX_MIMO_LAYER_CLEAR_CACHE") || enabled("TP_MIMO_LAYER_CLEAR_CACHE")
    }

    static func event(_ message: @autoclosure () -> String) {
        guard layerTraceEnabled || forceLayerEval || eagerLayerEval else { return }
        let elapsed = ProcessInfo.processInfo.systemUptime
        let rank = ProcessInfo.processInfo.environment["MLX_RANK"] ?? "?"
        let line = String(
            format: "[MiMoV2FlashTrace rank=%@ t=%.3f] %@\n",
            rank, elapsed, message())
        FileHandle.standardError.write(Data(line.utf8))
    }

    static func eval(_ value: MLXArray, name: String) {
        guard forceLayerEval || eagerLayerEval else { return }
        event("\(name) eval begin shape=\(value.shape)")
        let start = ProcessInfo.processInfo.systemUptime
        MLX.eval(value)
        let ms = (ProcessInfo.processInfo.systemUptime - start) * 1000
        event(String(format: "\(name) eval end %.1fms", ms))
        if clearCacheAfterLayerEval {
            Memory.clearCache()
            event("\(name) cache clear")
        }
    }
}

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
        precondition(sinks == nil, "Quantized SDPA does not support attention sinks.")
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
        precondition(
            localKeyValueHeads == localValueHeads,
            "MiMoV2FlashAttention TP head mismatch: k heads \(localKeyValueHeads), v heads \(localValueHeads)")

        var q = queries.reshaped(B, L, localAttentionHeads, -1).transposed(0, 2, 1, 3)
        var k = keys.reshaped(B, L, localKeyValueHeads, -1).transposed(0, 2, 1, 3)
        let v = values.reshaped(B, L, localValueHeads, -1).transposed(0, 2, 1, 3)

        q = applyRotaryPosition(rope, to: q, cache: cache)
        k = applyRotaryPosition(rope, to: k, cache: cache)

        let sinks: MLXArray?
        if hasSinks {
            precondition(
                attentionSinkBias.dim(0) == localAttentionHeads,
                "MiMoV2FlashAttention TP sink mismatch: sink dim \(attentionSinkBias.dim(0)), q heads \(localAttentionHeads)")
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

    init(hiddenSize: Int, intermediateSize: Int, bias: Bool) {
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize

        _gateProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: bias)
        _upProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: bias)
        _downProj.wrappedValue = Linear(intermediateSize, hiddenSize, bias: bias)
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
        guard let nRoutedExperts = config.nRoutedExperts else {
            fatalError("MiMoV2FlashMoEGate requires nRoutedExperts.")
        }

        precondition(config.topkMethod == "noaux_tc", "Unsupported topk method.")

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

    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "shared_experts") var sharedExperts: MiMoV2FlashMLP?

    init(_ config: MiMoV2FlashConfiguration, layerIdx: Int) {
        guard let nRoutedExperts = config.nRoutedExperts else {
            fatalError("MiMoV2FlashMoE requires nRoutedExperts.")
        }

        self.layerIdx = layerIdx
        self.numExpertsPerTok = config.numExpertsPerTok
        self.gate = MiMoV2FlashMoEGate(config)

        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.moeIntermediateSize,
            numExperts: nRoutedExperts
        )

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
    let layerIdx: Int
    let isMoe: Bool
    let isSlidingWindow: Bool

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ config: MiMoV2FlashConfiguration, layerIdx: Int, isMoe: Bool, isSlidingWindow: Bool) {
        self.layerIdx = layerIdx
        self.isMoe = isMoe
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
        MiMoV2FlashTrace.event(
            "layer \(layerIdx) begin shape=\(x.shape) kind=\(isMoe ? "moe" : "mlp") window=\(isSlidingWindow ? "swa" : "full")")
        let residual = x + selfAttn(inputLayerNorm(x), mask: mask, cache: cache)
        MiMoV2FlashTrace.event("layer \(layerIdx) attention graph shape=\(residual.shape)")
        MiMoV2FlashTrace.eval(residual, name: "layer \(layerIdx) attention")
        let output = residual + mlp(postAttentionLayerNorm(residual))
        MiMoV2FlashTrace.event("layer \(layerIdx) mlp graph shape=\(output.shape)")
        MiMoV2FlashTrace.eval(output, name: "layer \(layerIdx) mlp")
        MiMoV2FlashTrace.event("layer \(layerIdx) end")
        return output
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
        MiMoV2FlashTrace.event("model inner begin inputs=\(inputs.shape)")
        var h = embedTokens(inputs)
        MiMoV2FlashTrace.event("embed graph shape=\(h.shape)")
        MiMoV2FlashTrace.eval(h, name: "embed")

        let fullMask = createAttentionMask(h: h, cache: cache?[gaIdx])
        let swaMask = createAttentionMask(
            h: h, cache: cache?[swaIdx], windowSize: slidingWindowSize)

        for (i, layer) in layers.enumerated() {
            let mask = hybridLayerPattern[i] == 1 ? swaMask : fullMask
            MiMoV2FlashTrace.event("model layer \(i) dispatch")
            h = layer(h, mask: mask, cache: cache?[i])
        }

        let output = norm(h)
        MiMoV2FlashTrace.event("norm graph shape=\(output.shape)")
        MiMoV2FlashTrace.eval(output, name: "norm")
        MiMoV2FlashTrace.event("model inner end")
        return output
    }
}

final class MiMoV2VisionPatchEmbed: Module, UnaryLayer {
    @ModuleInfo(key: "proj") var proj: Conv3d

    let patchSize: Int
    let temporalPatchSize: Int
    let inChannels: Int
    let hiddenSize: Int

    init(_ config: MiMoV2VisionConfiguration) {
        self.patchSize = config.patchSize
        self.temporalPatchSize = config.temporalPatchSize
        self.inChannels = config.inChannels
        self.hiddenSize = config.hiddenSize

        let kernel = IntOrTriple([temporalPatchSize, patchSize, patchSize])
        _proj.wrappedValue = Conv3d(
            inputChannels: inChannels,
            outputChannels: hiddenSize,
            kernelSize: kernel,
            stride: kernel,
            bias: false
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var states = x.reshaped(
            -1,
            inChannels,
            temporalPatchSize,
            patchSize,
            patchSize
        ).movedAxis(source: 1, destination: 4)
        states = proj(states)
        return states.reshaped(-1, hiddenSize)
    }
}

final class MiMoV2VisionAttention: Module {
    let numHeads: Int
    let numKeyValueHeads: Int
    let headDim: Int
    let qRows: Int
    let kRows: Int
    let vRows: Int
    let scale: Float
    let usesSinks: Bool

    @ModuleInfo(key: "qkv") var qkv: Linear
    @ModuleInfo(key: "proj") var proj: Linear
    @ParameterInfo(key: "sinks") var sinks: MLXArray

    init(_ config: MiMoV2VisionConfiguration, usesSinks: Bool) {
        self.numHeads = config.numHeads
        self.numKeyValueHeads = config.numKeyValueHeads
        self.headDim = config.headDim
        self.qRows = config.numHeads * headDim
        self.kRows = config.numKeyValueHeads * headDim
        self.vRows = config.numKeyValueHeads * headDim
        self.scale = pow(Float(headDim), -0.5)
        self.usesSinks = usesSinks

        _qkv.wrappedValue = Linear(config.hiddenSize, qRows + kRows + vRows, bias: true)
        _proj.wrappedValue = Linear(numHeads * headDim, config.hiddenSize, bias: true)
        _sinks.wrappedValue = MLXArray.zeros([numHeads])
    }

    func callAsFunction(_ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode = .none)
        -> MLXArray
    {
        let sequenceLength = x.dim(0)
        let qkvStates = split(qkv(x), indices: [qRows, qRows + kRows], axis: -1)
        let q = qkvStates[0].reshaped(1, sequenceLength, numHeads, headDim)
            .transposed(0, 2, 1, 3)
        var k = qkvStates[1].reshaped(1, sequenceLength, numKeyValueHeads, headDim)
            .transposed(0, 2, 1, 3)
        var v = qkvStates[2].reshaped(1, sequenceLength, numKeyValueHeads, headDim)
            .transposed(0, 2, 1, 3)

        if numHeads != numKeyValueHeads {
            let repeats = numHeads / numKeyValueHeads
            k = repeated(k, count: repeats, axis: 1)
            v = repeated(v, count: repeats, axis: 1)
        }

        return proj(
            MLXFast.scaledDotProductAttention(
                queries: q,
                keys: k,
                values: v,
                scale: scale,
                mask: mask,
                sinks: usesSinks ? sinks : nil
            )
            .transposed(0, 2, 1, 3)
            .reshaped(sequenceLength, -1)
        )
    }

    override func updateMissing(
        parameter: String,
        verify: VerifyUpdate,
        path: [String],
        modulePath: [String]
    ) throws {
        if parameter == "sinks", !usesSinks {
            return
        }
        try super.updateMissing(
            parameter: parameter, verify: verify, path: path, modulePath: modulePath)
    }
}

final class MiMoV2VisionBlock: Module, UnaryLayer {
    @ModuleInfo(key: "norm1") var norm1: RMSNorm
    @ModuleInfo(key: "attn") var attn: MiMoV2VisionAttention
    @ModuleInfo(key: "norm2") var norm2: RMSNorm
    @ModuleInfo(key: "mlp") var mlp: MiMoV2FlashMLP

    init(_ config: MiMoV2VisionConfiguration, layerIndex: Int) {
        let usesSinks = config.useSink && !config.fullAttentionBlockIndexes.contains(layerIndex)
        _norm1.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.layernormEpsilon)
        _attn.wrappedValue = MiMoV2VisionAttention(config, usesSinks: usesSinks)
        _norm2.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.layernormEpsilon)
        _mlp.wrappedValue = MiMoV2FlashMLP(
            hiddenSize: config.hiddenSize,
            intermediateSize: config.intermediateSize,
            bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let residual = x + attn(norm1(x))
        return residual + mlp(norm2(residual))
    }
}

final class MiMoV2VisionMerger: Module, UnaryLayer {
    let hiddenSize: Int
    @ModuleInfo(key: "ln_q") var layerNormQ: RMSNorm
    @ModuleInfo(key: "mlp") var mlp: (Linear, GELU, Linear)

    init(_ config: MiMoV2VisionConfiguration) {
        self.hiddenSize = config.hiddenSize * config.spatialMergeSize * config.spatialMergeSize
        _layerNormQ.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.layernormEpsilon)
        _mlp.wrappedValue = (
            Linear(hiddenSize, hiddenSize),
            GELU(),
            Linear(hiddenSize, config.outHiddenSize)
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let merged = layerNormQ(x).reshaped(-1, hiddenSize)
        return mlp.2(mlp.1(mlp.0(merged)))
    }
}

final class MiMoV2VisionModel: Module, UnaryLayer {
    @ModuleInfo(key: "patch_embed") var patchEmbed: MiMoV2VisionPatchEmbed
    @ModuleInfo(key: "blocks") var blocks: [MiMoV2VisionBlock]
    @ModuleInfo(key: "merger") var merger: MiMoV2VisionMerger

    let fullAttentionBlockIndexes: Set<Int>
    let windowAttentionTypes: [Int]

    init(_ config: MiMoV2VisionConfiguration) {
        _patchEmbed.wrappedValue = MiMoV2VisionPatchEmbed(config)
        _blocks.wrappedValue = (0 ..< config.depth).map { index in
            MiMoV2VisionBlock(config, layerIndex: index)
        }
        _merger.wrappedValue = MiMoV2VisionMerger(config)
        self.fullAttentionBlockIndexes = Set(config.fullAttentionBlockIndexes)
        self.windowAttentionTypes = config.windowAttentionTypes
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var states = patchEmbed(x)
        for block in blocks {
            states = block(states)
        }
        return merger(states)
    }
}

final class MiMoV2AudioAttention: Module {
    let numHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    init(_ config: MiMoV2AudioConfiguration) {
        self.numHeads = config.inputLocalAttentionHeads
        self.headDim = config.inputLocalHeadDim
        self.scale = pow(Float(headDim), -0.5)
        _qProj.wrappedValue = Linear(config.inputLocalDim, config.inputLocalDim, bias: true)
        _kProj.wrappedValue = Linear(config.inputLocalDim, config.inputLocalDim, bias: true)
        _vProj.wrappedValue = Linear(config.inputLocalDim, config.inputLocalDim, bias: true)
        _oProj.wrappedValue = Linear(config.inputLocalDim, config.inputLocalDim, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))
        let q = qProj(x).reshaped(B, L, numHeads, headDim).transposed(0, 2, 1, 3)
        let k = kProj(x).reshaped(B, L, numHeads, headDim).transposed(0, 2, 1, 3)
        let v = vProj(x).reshaped(B, L, numHeads, headDim).transposed(0, 2, 1, 3)
        return oProj(
            MLXFast.scaledDotProductAttention(
                queries: q,
                keys: k,
                values: v,
                scale: scale,
                mask: .none
            )
            .transposed(0, 2, 1, 3)
            .reshaped(B, L, -1)
        )
    }
}

final class MiMoV2AudioLayer: Module, UnaryLayer {
    @ModuleInfo(key: "self_attn") var selfAttn: MiMoV2AudioAttention
    @ModuleInfo(key: "mlp") var mlp: MiMoV2FlashMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ config: MiMoV2AudioConfiguration) {
        _selfAttn.wrappedValue = MiMoV2AudioAttention(config)
        _mlp.wrappedValue = MiMoV2FlashMLP(
            hiddenSize: config.inputLocalDim,
            intermediateSize: config.inputLocalIntermediateSize,
            bias: false)
        _inputLayerNorm.wrappedValue = RMSNorm(dimensions: config.inputLocalDim, eps: 1e-6)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: config.inputLocalDim, eps: 1e-6)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let residual = x + selfAttn(inputLayerNorm(x))
        return residual + mlp(postAttentionLayerNorm(residual))
    }
}

final class MiMoV2AudioTransformer: Module, UnaryLayer {
    @ModuleInfo(key: "layers") var layers: [MiMoV2AudioLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm?

    init(_ config: MiMoV2AudioConfiguration) {
        _layers.wrappedValue = (0 ..< config.inputLocalLayers).map { _ in
            MiMoV2AudioLayer(config)
        }
        if config.addPostNorm {
            _norm.wrappedValue = RMSNorm(dimensions: config.inputLocalDim, eps: 1e-6)
        } else {
            _norm.wrappedValue = nil
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var states = x
        for layer in layers {
            states = layer(states)
        }
        return norm?(states) ?? states
    }
}

final class MiMoV2AudioProjection: Module, UnaryLayer {
    @ModuleInfo(key: "mlp") var mlp: (Linear, GELU, Linear)

    init(_ config: MiMoV2AudioConfiguration, textHiddenSize: Int) {
        let intermediateSize = textHiddenSize * 4
        _mlp.wrappedValue = (
            Linear(config.outHiddenSize, intermediateSize, bias: false),
            GELU(),
            Linear(intermediateSize, config.outHiddenSize, bias: false)
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        mlp.2(mlp.1(mlp.0(x)))
    }
}

final class MiMoV2AudioEncoder: Module, UnaryLayer {
    @ModuleInfo(key: "input_local_transformer") var inputLocalTransformer: MiMoV2AudioTransformer
    @ModuleInfo(key: "projection") var projection: MiMoV2AudioProjection

    init(_ config: MiMoV2AudioConfiguration, textHiddenSize: Int) {
        _inputLocalTransformer.wrappedValue = MiMoV2AudioTransformer(config)
        _projection.wrappedValue = MiMoV2AudioProjection(config, textHiddenSize: textHiddenSize)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        projection(inputLocalTransformer(x))
    }
}

public class MiMoV2FlashModel: Module, LLMModel, KVCacheDimensionProvider {
    public let modelType: String
    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: MiMoV2FlashModelInner
    let configuration: MiMoV2FlashConfiguration

    @ModuleInfo(key: "visual") var visual: MiMoV2VisionModel?
    @ModuleInfo(key: "audio_encoder") var audioEncoder: MiMoV2AudioEncoder?
    @ModuleInfo(key: "speech_embeddings") var speechEmbeddings: [Embedding]?
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    public init(_ config: MiMoV2FlashConfiguration) {
        self.configuration = config
        self.modelType = config.modelType
        self.vocabularySize = config.vocabularySize
        self.kvHeads = config.hybridLayerPattern.map {
            $0 == 1 ? config.swaKvHeads : config.kvHeads
        }
        self.model = MiMoV2FlashModelInner(config)
        if let visionConfig = config.visionConfig {
            _visual.wrappedValue = MiMoV2VisionModel(visionConfig)
        } else {
            _visual.wrappedValue = nil
        }
        if let audioConfig = config.audioConfig {
            _audioEncoder.wrappedValue = MiMoV2AudioEncoder(
                audioConfig,
                textHiddenSize: config.hiddenSize)
            _speechEmbeddings.wrappedValue = (0 ..< audioConfig.audioChannels).map { _ in
                Embedding(
                    embeddingCount: audioConfig.speechVocabularySize,
                    dimensions: audioConfig.inputLocalDim)
            }
        } else {
            _audioEncoder.wrappedValue = nil
            _speechEmbeddings.wrappedValue = nil
        }
        _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let out = model(inputs, cache: cache)
        return lmHead(out)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        func dequant(weight: MLXArray, scaleInv: MLXArray) -> MLXArray {
            let bs = 128
            let (m, n) = (weight.shape[0], weight.shape[1])
            let padBottom = bs * scaleInv.dim(0) - m
            let padSide = bs * scaleInv.dim(1) - n

            var paddedWeight = padded(
                weight.asType(.float32), widths: [.init((0, padBottom)), .init((0, padSide))])
            paddedWeight = paddedWeight.reshaped(
                [(m + padBottom) / bs, bs, (n + padSide) / bs, bs])
            let scaled =
                paddedWeight * scaleInv.asType(.float32)[0..., .newAxis, 0..., .newAxis]
            return scaled.reshaped([m + padBottom, n + padSide])[0 ..< m, 0 ..< n]
                .asType(.bfloat16)
        }

        func sourceQKVLayout(
            key: String, weight: MLXArray, scaleInv: MLXArray
        ) -> (tp: Int, qPer: Int, kPer: Int, vPer: Int, rankRows: Int, rankScaleRows: Int)? {
            let prefix = "model.layers."
            let suffix = ".self_attn.qkv_proj.weight"
            guard key.hasPrefix(prefix), key.hasSuffix(suffix) else { return nil }
            let remainder = key.dropFirst(prefix.count)
            guard let dot = remainder.firstIndex(of: "."),
                let layer = Int(remainder[..<dot])
            else { return nil }

            let isSliding = configuration.hybridLayerPattern[layer] == 1
            let qRows =
                (isSliding ? configuration.swaAttentionHeads : configuration.attentionHeads)
                * (isSliding ? configuration.swaHeadDim : configuration.headDim)
            let kRows =
                (isSliding ? configuration.swaKvHeads : configuration.kvHeads)
                * (isSliding ? configuration.swaHeadDim : configuration.headDim)
            let vRows =
                (isSliding ? configuration.swaKvHeads : configuration.kvHeads)
                * (isSliding ? configuration.swaVHeadDim : configuration.vHeadDim)
            guard qRows + kRows + vRows == weight.shape[0] else { return nil }

            let tp = 4
            guard qRows % tp == 0, kRows % tp == 0, vRows % tp == 0 else { return nil }
            let qPer = qRows / tp
            let kPer = kRows / tp
            let vPer = vRows / tp
            let rankRows = qPer + kPer + vPer
            let rankScaleRows = (rankRows + 127) / 128
            guard rankRows * tp == weight.shape[0],
                rankScaleRows * tp == scaleInv.shape[0]
            else {
                return nil
            }
            return (tp, qPer, kPer, vPer, rankRows, rankScaleRows)
        }

        func dequantInterleavedSourceQKV(
            key: String, weight: MLXArray, scaleInv: MLXArray
        ) -> MLXArray? {
            guard let layout = sourceQKVLayout(key: key, weight: weight, scaleInv: scaleInv)
            else {
                return nil
            }

            var qParts: [MLXArray] = []
            var kParts: [MLXArray] = []
            var vParts: [MLXArray] = []
            for rank in 0 ..< layout.tp {
                let wStart = rank * layout.rankRows
                let sStart = rank * layout.rankScaleRows
                let rankWeight = weight[wStart ..< (wStart + layout.rankRows), 0...]
                let rankScale = scaleInv[sStart ..< (sStart + layout.rankScaleRows), 0...]
                let rankOut = dequant(weight: rankWeight, scaleInv: rankScale)
                let qkv = split(
                    rankOut,
                    indices: [layout.qPer, layout.qPer + layout.kPer],
                    axis: 0)
                qParts.append(qkv[0])
                kParts.append(qkv[1])
                vParts.append(qkv[2])
            }
            return concatenated(qParts + kParts + vParts, axis: 0)
        }

        var newWeights: [String: MLXArray] = [:]
        for (key, value) in weights {
            if key.contains("weight_scale_inv") {
                let weightKey = key.replacingOccurrences(of: "_scale_inv", with: "")
                if let weight = weights[weightKey] {
                    newWeights[weightKey] =
                        dequantInterleavedSourceQKV(key: weightKey, weight: weight, scaleInv: value)
                        ?? dequant(weight: weight, scaleInv: value)
                }
            } else if newWeights[key] == nil {
                newWeights[key] = value
            }
        }

        var sanitizedWeights = newWeights.isEmpty ? weights : newWeights

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
                for key in ["weight", "scales", "biases"] {
                    let firstKey = "\(prefix).mlp.experts.0.\(projName).\(key)"
                    if sanitizedWeights[firstKey] != nil {
                        let toJoin = (0 ..< (configuration.nRoutedExperts ?? 1)).map {
                            sanitizedWeights.removeValue(
                                forKey: "\(prefix).mlp.experts.\($0).\(projName).\(key)")!
                        }
                        sanitizedWeights["\(prefix).mlp.switch_mlp.\(projName).\(key)"] =
                            MLX.stacked(toJoin)
                    }
                }
            }
        }

        if let visionConfig = configuration.visionConfig,
            let patchWeight = sanitizedWeights["visual.patch_embed.proj.weight"],
            patchWeight.shape.count == 5,
            patchWeight.shape[1] == visionConfig.inChannels,
            patchWeight.shape[2] == visionConfig.temporalPatchSize,
            patchWeight.shape[3] == visionConfig.patchSize,
            patchWeight.shape[4] == visionConfig.patchSize
        {
            sanitizedWeights["visual.patch_embed.proj.weight"] =
                patchWeight.transposed(0, 2, 3, 4, 1)
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

public struct MiMoV2VisionConfiguration: Codable, Sendable {
    var depth: Int
    var hiddenSize: Int
    var intermediateSize: Int
    var numHeads: Int
    var numKeyValueHeads: Int
    var outHiddenSize: Int
    var headDim: Int
    var patchSize: Int
    var temporalPatchSize: Int
    var inChannels: Int
    var spatialMergeSize: Int
    var windowSize: Int
    var visualTokenWindowSize: Int
    var fullAttentionBlockIndexes: [Int]
    var windowAttentionTypes: [Int]
    var useSink: Bool
    var layernormEpsilon: Float = 1e-6

    enum CodingKeys: String, CodingKey {
        case depth
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHeads = "num_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case outHiddenSize = "out_hidden_size"
        case headDim = "head_dim"
        case patchSize = "patch_size"
        case temporalPatchSize = "temporal_patch_size"
        case inChannels = "in_channels"
        case spatialMergeSize = "spatial_merge_size"
        case windowSize = "window_size"
        case visualTokenWindowSize = "visual_token_window_size"
        case fullAttentionBlockIndexes = "fullatt_block_indexes"
        case windowAttentionTypes = "vit_window_attn_types"
        case useSink = "use_sink"
        case layernormEpsilon = "layer_norm_eps"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.depth = try container.decode(Int.self, forKey: .depth)
        self.hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        self.intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        self.numHeads = try container.decode(Int.self, forKey: .numHeads)
        self.numKeyValueHeads = try container.decodeIfPresent(Int.self, forKey: .numKeyValueHeads)
            ?? numHeads
        self.outHiddenSize = try container.decode(Int.self, forKey: .outHiddenSize)
        if let headDim = try container.decodeIfPresent(Int.self, forKey: .headDim) {
            self.headDim = headDim
        } else if hiddenSize == 1280, numHeads == 32, numKeyValueHeads == 8 {
            self.headDim = 64
        } else {
            self.headDim = hiddenSize / numHeads
        }
        self.patchSize = try container.decode(Int.self, forKey: .patchSize)
        self.temporalPatchSize = try container.decode(Int.self, forKey: .temporalPatchSize)
        self.inChannels = try container.decodeIfPresent(Int.self, forKey: .inChannels) ?? 3
        self.spatialMergeSize = try container.decode(Int.self, forKey: .spatialMergeSize)
        self.windowSize = try container.decodeIfPresent(Int.self, forKey: .windowSize) ?? 0
        self.visualTokenWindowSize =
            try container.decodeIfPresent(Int.self, forKey: .visualTokenWindowSize) ?? 0
        self.fullAttentionBlockIndexes =
            try container.decodeIfPresent([Int].self, forKey: .fullAttentionBlockIndexes) ?? []
        self.windowAttentionTypes =
            try container.decodeIfPresent([Int].self, forKey: .windowAttentionTypes) ?? []
        self.useSink = try container.decodeIfPresent(Bool.self, forKey: .useSink) ?? false
        self.layernormEpsilon =
            try container.decodeIfPresent(Float.self, forKey: .layernormEpsilon) ?? 1e-6
    }
}

public struct MiMoV2AudioConfiguration: Codable, Sendable {
    var audioChannels: Int
    var speechVocabularySize: Int
    var inputLocalLayers: Int
    var inputLocalDim: Int
    var inputLocalAttentionHeads: Int
    var inputLocalHeadDim: Int
    var inputLocalIntermediateSize: Int
    var projectionLayers: Int
    var outHiddenSize: Int
    var ropeTheta: Float
    var partialRotaryFactor: Float
    var addPostNorm: Bool

    enum CodingKeys: String, CodingKey {
        case audioChannels = "audio_channels"
        case speechVocabularySize = "speech_vocab_size"
        case inputLocalLayers = "input_local_layers"
        case inputLocalDim = "input_local_dim"
        case inputLocalAttentionHeads = "input_local_attn_heads"
        case inputLocalHeadDim = "input_local_head_dim"
        case inputLocalIntermediateSize = "input_local_intermediate_size"
        case projectionLayers = "projection_layers"
        case outHiddenSize = "out_hidden_size"
        case ropeTheta = "rope_theta"
        case partialRotaryFactor = "partial_rotary_factor"
        case addPostNorm = "add_post_norm"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.audioChannels = try container.decode(Int.self, forKey: .audioChannels)
        if let intValue = try? container.decode(Int.self, forKey: .speechVocabularySize) {
            self.speechVocabularySize = intValue
        } else {
            let stringValue = try container.decode(String.self, forKey: .speechVocabularySize)
            self.speechVocabularySize = Int(stringValue) ?? 1280
        }
        self.inputLocalLayers = try container.decode(Int.self, forKey: .inputLocalLayers)
        self.inputLocalDim = try container.decode(Int.self, forKey: .inputLocalDim)
        self.inputLocalAttentionHeads =
            try container.decode(Int.self, forKey: .inputLocalAttentionHeads)
        self.inputLocalHeadDim = try container.decode(Int.self, forKey: .inputLocalHeadDim)
        self.inputLocalIntermediateSize =
            try container.decode(Int.self, forKey: .inputLocalIntermediateSize)
        self.projectionLayers = try container.decodeIfPresent(Int.self, forKey: .projectionLayers)
            ?? 2
        self.outHiddenSize = try container.decode(Int.self, forKey: .outHiddenSize)
        self.ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 640000
        self.partialRotaryFactor =
            try container.decodeIfPresent(Float.self, forKey: .partialRotaryFactor) ?? 1.0
        self.addPostNorm = try container.decodeIfPresent(Bool.self, forKey: .addPostNorm) ?? true
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
    var nRoutedExperts: Int?
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
    var visionConfig: MiMoV2VisionConfiguration?
    var audioConfig: MiMoV2AudioConfiguration?

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
        case visionConfig = "vision_config"
        case audioConfig = "audio_config"
    }
}

// MARK: - LoRA

extension MiMoV2FlashModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
