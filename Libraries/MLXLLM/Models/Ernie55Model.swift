// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXLMCommon
import MLXNN

struct Ernie55Configuration: Decodable {
    struct YaRN: Decodable {
        let factor: Float
        let betaFast: Float
        let betaSlow: Float
        let originalMaxPositionEmbeddings: Int
    }

    let hiddenSize: Int
    let numHiddenLayers: Int
    let vocabSize: Int
    let firstKDenseReplace: Int
    let intermediateSize: Int
    let moeIntermediateSize: Int
    let moeLatentSize: Int
    let nRoutedExperts: Int
    let nSharedExperts: Int
    let numExpertsPerTok: Int
    let normTopkProb: Bool
    let activationSituBeta: Float
    let activationSituLinearBeta: Float
    let hcMult: Int
    let hcSinkhornIters: Int
    let hcEps: Float
    let rmsNormEps: Float
    let compressRatios: [Int]
    let compressRopeTheta: Float
    let ropeScaling: YaRN
    let ropeTheta: Float
    let slidingWindow: Int
    let numAttentionHeads: Int
    let headDim: Int
    let qLoraRank: Int
    let qkRopeHeadDim: Int
    let oGroups: Int
    let oLoraRank: Int
    let hybridMlaNumAttentionHeads: Int
    let hybridMlaQLoraRank: Int
    let hybridMlaKvLoraRank: Int
    let hybridMlaQkNopeHeadDim: Int
    let hybridMlaQkRopeHeadDim: Int
    let hybridMlaVHeadDim: Int
    let indexNHeads: Int
    let indexHeadDim: Int
    let indexTopk: Int
    let vhaRank: Int?
    let numNextnPredictLayers: Int
    let eosTokenId: Int?

    static func decode(_ data: Data) throws -> Self {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let config = try decoder.decode(Self.self, from: data)
        guard config.compressRatios.count == config.numHiddenLayers,
            config.compressRatios.allSatisfy({ $0 == 128 || $0 == -2 }),
            config.slidingWindow == 128, config.firstKDenseReplace > 0,
            config.firstKDenseReplace < config.numHiddenLayers, config.numNextnPredictLayers == 0,
            config.moeLatentSize.isMultiple(of: 128), config.moeIntermediateSize.isMultiple(of: 128),
            config.numExpertsPerTok > 0, config.numExpertsPerTok <= config.nRoutedExperts,
            config.oGroups > 0, config.numAttentionHeads.isMultiple(of: config.oGroups)
        else {
            throw JangLoaderError.invalidConfig("Unsupported ERNIE 5.5 Flash geometry")
        }
        return config
    }

    var caps: (gate: Float, up: Float) { (activationSituBeta, activationSituLinearBeta) }
    var hcaShape: Ernie55AttentionShape {
        .init(hidden: hiddenSize, heads: numAttentionHeads, headWidth: headDim, queryRank: qLoraRank,
              ropeWidth: qkRopeHeadDim, epsilon: rmsNormEps, theta: compressRopeTheta,
              virtualHeadRank: vhaRank ?? 16)
    }
    var mlaShape: Ernie55AttentionShape {
        .init(hidden: hiddenSize, heads: hybridMlaNumAttentionHeads,
              headWidth: hybridMlaQkNopeHeadDim + hybridMlaQkRopeHeadDim, queryRank: hybridMlaQLoraRank,
              ropeWidth: hybridMlaQkRopeHeadDim, epsilon: rmsNormEps, theta: ropeTheta,
              virtualHeadRank: vhaRank ?? 16)
    }
}

final class Ernie55MLP: Module {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    let caps: (gate: Float, up: Float)

    init(_ c: Ernie55Configuration, width: Int) {
        caps = c.caps
        _gate.wrappedValue = Linear(c.hiddenSize, width, bias: false)
        _up.wrappedValue = Linear(c.hiddenSize, width, bias: false)
        _down.wrappedValue = Linear(width, c.hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let g = DeepseekV41FP8Rows.apply(gate, x)
        let u = DeepseekV41FP8Rows.apply(up, x)
        return DeepseekV41FP8Rows.apply(down, Ernie55Math.situ(g, u, gateCap: caps.gate, upCap: caps.up))
    }
}

final class Ernie55Router: Module {
    @ParameterInfo var weight: MLXArray
    @ParameterInfo(key: "weight_1") var second: MLXArray
    @ParameterInfo(key: "routed_scaling_factor_param") var expertScale: MLXArray
    let routes: Int
    let normalize: Bool

    init(_ c: Ernie55Configuration) {
        routes = c.numExpertsPerTok
        normalize = c.normTopkProb
        _weight.wrappedValue = MLXArray.zeros([c.nRoutedExperts, c.hiddenSize])
        _second.wrappedValue = MLXArray.zeros([c.nRoutedExperts, c.hiddenSize])
        _expertScale.wrappedValue = MLXArray.ones([c.nRoutedExperts])
        super.init()
    }

    func callAsFunction(_ x: MLXArray, correction: MLXArray) -> (MLXArray, MLXArray) {
        let route = Ernie55Math.route(
            x, first: weight, second: second, correction: correction,
            expertScale: expertScale, count: routes, normalize: normalize)
        return (route.indices, route.weights)
    }
}

final class Ernie55MoE: Module {
    @ModuleInfo var gate: Ernie55Router
    @ModuleInfo(key: "fc1_latent_proj") var latentIn: Linear
    @ModuleInfo(key: "fc2_latent_proj") var latentOut: Linear
    @ModuleInfo(key: "latent_norm") var latentNorm: RMSNorm
    @ModuleInfo(key: "shared_experts") var shared: Ernie55MLP
    @ParameterInfo(key: "e_score_correction_bias") var correction: MLXArray
    let routed: JANGTRoutedExpertLayer
    private var compiledDecode: (@Sendable ([MLXArray]) -> [MLXArray])?

    func compileDecode() {
        let gate = self.gate
        let latentIn = self.latentIn
        let latentOut = self.latentOut
        let latentNorm = self.latentNorm
        let shared = self.shared
        let routed = self.routed
        compiledDecode = vmlxTrustedCompile(inputs: [gate, latentIn, latentOut, latentNorm, shared]) { inputs in
            let x = inputs[0]
            let (ids, scores) = gate(x, correction: inputs[1])
            let combined = Self.routedActivations(latentIn(x), ids: ids, scores: scores, routed: routed)
            return [shared(x) + latentOut(latentNorm(combined))]
        }
    }

    init(_ c: Ernie55Configuration, routed: JANGTRoutedExpertLayer) {
        self.routed = routed
        _gate.wrappedValue = Ernie55Router(c)
        _latentIn.wrappedValue = Linear(c.hiddenSize, c.moeLatentSize, bias: false)
        _latentOut.wrappedValue = Linear(c.moeLatentSize, c.hiddenSize, bias: false)
        _latentNorm.wrappedValue = RMSNorm(dimensions: c.moeLatentSize, eps: c.rmsNormEps)
        _shared.wrappedValue = Ernie55MLP(c, width: c.nSharedExperts * c.moeIntermediateSize)
        _correction.wrappedValue = MLXArray.zeros([c.nRoutedExperts])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        if let compiledDecode, x.dim(1) == 1, !CompiledDecodeTrace.isActive {
            return compiledDecode([x, correction])[0]
        }
        let (ids, scores) = gate(x, correction: correction)
        let combined = Self.routedActivations(
            DeepseekV41FP8Rows.apply(latentIn, x), ids: ids, scores: scores, routed: routed)
        return shared(x) + DeepseekV41FP8Rows.apply(latentOut, latentNorm(combined))
    }

    private static func routedActivations(
        _ latent: MLXArray, ids: MLXArray, scores: MLXArray, routed: JANGTRoutedExpertLayer
    ) -> MLXArray {
        if let prefill = routed.routedW4A8Prefill(
            latent, indices: ids, scores: scores, roundActivations: Ernie55Math.roundExpertActivations)
        {
            return prefill
        }
        return routed.routedW4A8Decode(
            latent, indices: ids, scores: scores, roundActivations: Ernie55Math.roundExpertActivations)
    }
}

final class Ernie55HyperConnection: Module {
    @ParameterInfo var fn: MLXArray
    @ParameterInfo var base: MLXArray
    @ParameterInfo var scale: MLXArray
    let iterations: Int
    let mixingEpsilon: Float
    let normEpsilon: Float

    init(_ c: Ernie55Configuration) {
        iterations = c.hcSinkhornIters
        mixingEpsilon = c.hcEps
        normEpsilon = c.rmsNormEps
        let width = (2 + c.hcMult) * c.hcMult
        _fn.wrappedValue = MLXArray.zeros([width, c.hcMult * c.hiddenSize])
        _base.wrappedValue = MLXArray.zeros([width])
        _scale.wrappedValue = MLXArray.ones([3])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
        Ernie55Math.collapse(
            x, projection: fn, scales: scale, bias: base,
            iterations: iterations, mixingEpsilon: mixingEpsilon, normEpsilon: normEpsilon)
    }
}

final class Ernie55HyperHead: Module {
    @ParameterInfo var fn: MLXArray
    @ParameterInfo var base: MLXArray
    @ParameterInfo var scale: MLXArray
    let normEpsilon: Float
    let mixingEpsilon: Float

    init(_ c: Ernie55Configuration) {
        normEpsilon = c.rmsNormEps
        mixingEpsilon = c.hcEps
        _fn.wrappedValue = MLXArray.zeros([c.hcMult, c.hcMult * c.hiddenSize])
        _base.wrappedValue = MLXArray.zeros([c.hcMult])
        _scale.wrappedValue = MLXArray.ones([1])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let flat = x.reshaped(x.dim(0), x.dim(1), -1).asType(.float32)
        let normalized = MLXFast.rmsNorm(flat, weight: MLXArray.ones([flat.dim(-1)]), eps: normEpsilon)
        let gates = sigmoid(matmul(normalized, fn.transposed()) * scale + base) + mixingEpsilon
        return (gates.expandedDimensions(axis: -1) * x.asType(.float32)).sum(axis: -2).asType(x.dtype)
    }
}

final class Ernie55DecoderLayer: Module {
    @ModuleInfo var hca: Ernie55HCAAttention?
    @ModuleInfo var mla: Ernie55MLAAttention?
    @ModuleInfo var dense: Ernie55MLP?
    @ModuleInfo var moe: Ernie55MoE?
    @ModuleInfo(key: "attn_hc") var attentionConnection: Ernie55HyperConnection
    @ModuleInfo(key: "ffn_hc") var feedForwardConnection: Ernie55HyperConnection
    @ModuleInfo(key: "attn_norm") var attentionNorm: RMSNorm
    @ModuleInfo(key: "ffn_norm") var feedForwardNorm: RMSNorm

    init(_ c: Ernie55Configuration, at index: Int, rope: DeepseekV4RoPE, routed: JANGTRoutedExpertLayer?) {
        let ratio = c.compressRatios[index]
        if ratio == -2 {
            _mla.wrappedValue = Ernie55MLAAttention(
                shape: c.mlaShape, latentWidth: c.hybridMlaKvLoraRank, valueWidth: c.hybridMlaVHeadDim,
                indexerHeads: c.indexNHeads, indexerDimension: c.indexHeadDim, topK: c.indexTopk,
                window: c.slidingWindow)
        } else {
            _hca.wrappedValue = Ernie55HCAAttention(
                shape: c.hcaShape, window: c.slidingWindow, ratio: ratio, outputGroups: c.oGroups,
                outputRank: c.oLoraRank, rope: rope)
        }
        if let routed {
            _moe.wrappedValue = Ernie55MoE(c, routed: routed)
        } else {
            _dense.wrappedValue = Ernie55MLP(c, width: c.intermediateSize)
        }
        _attentionConnection.wrappedValue = Ernie55HyperConnection(c)
        _feedForwardConnection.wrappedValue = Ernie55HyperConnection(c)
        _attentionNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
        _feedForwardNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
        super.init()
    }

    func prepare() {
        hca?.prepare()
        mla?.prepare()
    }

    func callAsFunction(
        _ x: MLXArray, cache: KVCache?, layerIndex: Int = 0, capture: Ernie55VerifyCapture? = nil
    ) -> MLXArray {
        let normalized: MLXArray
        let post: MLXArray
        let mix: MLXArray
        if let fused = Ernie55Kernels.batchedHCCollapseNorm(
            h: x, fn: attentionConnection.fn, scale: attentionConnection.scale, base: attentionConnection.base,
            normWeight: attentionNorm.weight, iterations: attentionConnection.iterations,
            mixingEpsilon: attentionConnection.mixingEpsilon, normEpsilon: attentionConnection.normEpsilon,
            rmsEpsilon: attentionNorm.eps)
        {
            (normalized, post, mix) = fused
        } else {
            let (input, p, m) = attentionConnection(x)
            normalized = attentionNorm(input)
            (post, mix) = (p, m)
        }
        let attended = hca?(normalized, cache: cache as? DeepseekV4Cache, layerIndex: layerIndex, capture: capture)
            ?? mla!(normalized, cache: cache as? CacheList, layerIndex: layerIndex, capture: capture)
        let residual = Ernie55Math.expand(attended, residual: x, post: post, mix: mix)
        let ffn: MLXArray
        let nextPost: MLXArray
        let nextMix: MLXArray
        if let fused = Ernie55Kernels.batchedHCCollapseNorm(
            h: residual, fn: feedForwardConnection.fn, scale: feedForwardConnection.scale,
            base: feedForwardConnection.base, normWeight: feedForwardNorm.weight,
            iterations: feedForwardConnection.iterations,
            mixingEpsilon: feedForwardConnection.mixingEpsilon,
            normEpsilon: feedForwardConnection.normEpsilon, rmsEpsilon: feedForwardNorm.eps)
        {
            (ffn, nextPost, nextMix) = fused
        } else {
            let (hidden, p, m) = feedForwardConnection(residual)
            ffn = feedForwardNorm(hidden)
            (nextPost, nextMix) = (p, m)
        }
        let output = moe?(ffn) ?? dense!(ffn)
        return Ernie55Math.expand(output, residual: residual, post: nextPost, mix: nextMix)
    }

    /// Chained fused mHC execution for 1...16 rows (AR decode & DSpark verify): folds the
    /// previous sublayer's expand directly into this sublayer's mix+Sinkhorn+collapse+RMSNorm.
    func callFused(
        prevStream: MLXArray, prevOut: MLXArray?, prevPost: MLXArray?, prevComb: MLXArray?,
        cache: KVCache?, layerIndex: Int = 0, capture: Ernie55VerifyCapture? = nil
    ) -> (stream: MLXArray, out: MLXArray, post: MLXArray, comb: MLXArray, expandedInput: MLXArray)? {
        guard let attnHC = Ernie55Kernels.fusedHCCollapseNorm(
            h: prevStream, prevOut: prevOut, prevPost: prevPost, prevComb: prevComb,
            fn: attentionConnection.fn, scale: attentionConnection.scale, base: attentionConnection.base,
            normWeight: attentionNorm.weight, iterations: attentionConnection.iterations,
            mixingEpsilon: attentionConnection.mixingEpsilon, normEpsilon: attentionConnection.normEpsilon,
            rmsEpsilon: attentionNorm.eps)
        else { return nil }
        let attended = hca?(attnHC.normed, cache: cache as? DeepseekV4Cache, layerIndex: layerIndex, capture: capture)
            ?? mla!(attnHC.normed, cache: cache as? CacheList, layerIndex: layerIndex, capture: capture)
        guard let ffnHC = Ernie55Kernels.fusedHCCollapseNorm(
            h: attnHC.stream, prevOut: attended, prevPost: attnHC.post, prevComb: attnHC.comb,
            fn: feedForwardConnection.fn, scale: feedForwardConnection.scale, base: feedForwardConnection.base,
            normWeight: feedForwardNorm.weight, iterations: feedForwardConnection.iterations,
            mixingEpsilon: feedForwardConnection.mixingEpsilon, normEpsilon: feedForwardConnection.normEpsilon,
            rmsEpsilon: feedForwardNorm.eps)
        else { return nil }
        let output = moe?(ffnHC.normed) ?? dense!(ffnHC.normed)
        return (ffnHC.stream, output, ffnHC.post, ffnHC.comb, attnHC.stream)
    }
}

final class Ernie55Backbone: Module {
    @ModuleInfo(key: "embed_tokens") var embedding: Embedding
    @ModuleInfo(key: "hc_head") var head: Ernie55HyperHead
    @ModuleInfo var norm: RMSNorm
    let layers: [Ernie55DecoderLayer]
    let streams: Int
    /// 0-based decoder layer indices whose post-layer mean-over-streams hidden is tapped for DSpark.
    var dsparkTapLayers: [Int] = []
    /// Sliding window capacity of the DSpark hidden tail stored on layer 0's cache.
    var dsparkTailCapacity: Int = 128

    init(_ c: Ernie55Configuration, routed: [Int: JANGTRoutedExpertLayer]) {
        streams = c.hcMult
        let rope = DeepseekV4RoPE(
            dim: c.qkRopeHeadDim, base: c.compressRopeTheta, factor: c.ropeScaling.factor,
            origMaxPos: c.ropeScaling.originalMaxPositionEmbeddings,
            betaFast: c.ropeScaling.betaFast, betaSlow: c.ropeScaling.betaSlow)
        layers = (0..<c.numHiddenLayers).map {
            Ernie55DecoderLayer(c, at: $0, rope: rope, routed: routed[$0])
        }
        _embedding.wrappedValue = Embedding(embeddingCount: c.vocabSize, dimensions: c.hiddenSize)
        _head.wrappedValue = Ernie55HyperHead(c)
        _norm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
        super.init()
    }

    func callAsFunction(_ tokens: MLXArray, cache: [KVCache]?) -> MLXArray {
        let holder = cache?.first as? DeepseekV4Cache
        let startPos = holder?.offset ?? 0
        let capture = holder?.ernieVerifyCapture
        capture?.begin(tokens: tokens.dim(1))
        let collectTaps = !dsparkTapLayers.isEmpty && holder != nil && tokens.dim(0) == 1
        var tapRows: [Int: MLXArray] = [:]
        if collectTaps { tapRows.reserveCapacity(dsparkTapLayers.count) }

        let embeds = embedding(tokens)
        var x = broadcast(embeds.expandedDimensions(axis: -2),
                          to: [embeds.dim(0), embeds.dim(1), streams, embeds.dim(-1)])
        if x.dim(0) == 1 && (1...16).contains(x.dim(1)) && Device.defaultDevice().deviceType == .gpu {
            var stream = contiguous(x)
            var pendingOut: MLXArray? = nil
            var pendingPost: MLXArray? = nil
            var pendingComb: MLXArray? = nil
            var usedFused = true
            for (index, layer) in layers.enumerated() {
                if let step = layer.callFused(
                    prevStream: stream, prevOut: pendingOut, prevPost: pendingPost, prevComb: pendingComb,
                    cache: cache?[index], layerIndex: index, capture: capture)
                {
                    if collectTaps && index > 0 && dsparkTapLayers.contains(index - 1) {
                        tapRows[index - 1] = mean(step.expandedInput.asType(.float32), axis: -2)
                            .asType(step.expandedInput.dtype)[0]
                    }
                    stream = step.stream
                    pendingOut = step.out
                    pendingPost = step.post
                    pendingComb = step.comb
                } else {
                    if let out = pendingOut, let post = pendingPost, let comb = pendingComb {
                        stream = Ernie55Math.expand(out, residual: stream, post: post, mix: comb)
                        if collectTaps && index > 0 && dsparkTapLayers.contains(index - 1) {
                            tapRows[index - 1] = mean(stream.asType(.float32), axis: -2)
                                .asType(stream.dtype)[0]
                        }
                    }
                    x = layer(stream, cache: cache?[index], layerIndex: index, capture: capture)
                    if collectTaps && dsparkTapLayers.contains(index) {
                        tapRows[index] = mean(x.asType(.float32), axis: -2).asType(x.dtype)[0]
                    }
                    for remaining in (index + 1)..<layers.count {
                        x = layers[remaining](x, cache: cache?[remaining], layerIndex: remaining, capture: capture)
                        if collectTaps && dsparkTapLayers.contains(remaining) {
                            tapRows[remaining] = mean(x.asType(.float32), axis: -2).asType(x.dtype)[0]
                        }
                    }
                    usedFused = false
                    break
                }
            }
            if usedFused, let out = pendingOut, let post = pendingPost, let comb = pendingComb {
                x = Ernie55Math.expand(out, residual: stream, post: post, mix: comb)
                let lastIndex = layers.count - 1
                if collectTaps && dsparkTapLayers.contains(lastIndex) {
                    tapRows[lastIndex] = mean(x.asType(.float32), axis: -2).asType(x.dtype)[0]
                }
            }
        } else {
            for (index, layer) in layers.enumerated() {
                x = layer(x, cache: cache?[index], layerIndex: index, capture: capture)
                if collectTaps && dsparkTapLayers.contains(index) {
                    tapRows[index] = mean(x.asType(.float32), axis: -2).asType(x.dtype)[0]
                }
            }
        }
        if collectTaps, let holder, tapRows.count == dsparkTapLayers.count {
            let joined = concatenated(dsparkTapLayers.map { tapRows[$0]! }, axis: -1)
            var tail = holder.auxiliaryHiddenTail ?? HybridPoolHiddenTail(capacity: dsparkTailCapacity)
            tail.record(rows: joined, startPosition: startPos)
            holder.auxiliaryHiddenTail = tail
        }
        return norm(head(x))
    }
}

final class Ernie55Model: Module, LLMModel, ModalityBearing, SafetensorsLoadKeyExcluding {
    let model: Ernie55Backbone
    @ModuleInfo(key: "lm_head") var lmHead: Linear
    @ParameterInfo(key: "multimax_ranges") var logitRanges: MLXArray
    @ParameterInfo(key: "multimax_ts") var logitSlopes: MLXArray

    let args: Ernie55Configuration
    let excludedTensorNames: Set<String>
    let dsparkAttachLock = NSLock()
    var dsparkHead: Ernie55DSparkHead?
    var vocabularySize: Int { args.vocabSize }
    var modalities: Set<ModelRuntimeRequestModality> { [.text] }
    var loraLayers: [Module] { model.layers }
    var preservesCheckpointParameterDTypes: Bool { true }
    var cacheStorageDTypeIdentity: String? { "ernie55-hca-fp32-pending-v1" }
    var cacheComputationIdentity: String? { "ernie55-hca-mla-dsa-v5" }
    var supportsWholeForwardCompilation: Bool { false }
    var maximumSupportedDecodeBatchSize: Int? { 1 }
    var recommendedPrefillStepSize: Int? { 2048 }
    var requiresResidentSafetensorsWeights: Bool { true }

    init(_ args: Ernie55Configuration, routed: [Int: JANGTRoutedExpertLayer], excluded: Set<String>) {
        self.args = args
        self.excludedTensorNames = excluded
        self.model = Ernie55Backbone(args, routed: routed)
        _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabSize, bias: false)
        _logitRanges.wrappedValue = MLXArray.zeros([4])
        _logitSlopes.wrappedValue = MLXArray.zeros([4])
        super.init()
    }

    func projectLogits(_ hidden: MLXArray, applyMultiMax: Bool = true) -> MLXArray {
        if let fused = Ernie55Kernels.lmHeadMultiMax(
            hidden, weight: lmHead.weight, ranges: logitRanges, slopes: logitSlopes,
            applyMultiMax: applyMultiMax)
        {
            return fused
        }
        let logits = matmul(hidden.asType(lmHead.weight.dtype), lmHead.weight.transposed()).asType(.float32)
        return applyMultiMax ? Ernie55Math.multiMax(logits, ranges: logitRanges, slopes: logitSlopes) : logits
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        projectLogits(model(inputs, cache: cache))
    }

    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        try withError { error -> PrepareResult in
            // Default GenerateParameters.prefillStepSize is 512; promote default/unspecified to
            // ERNIE's 2048-token chunk (exact multiple of HCA ratio 128) so 512-expert NAX tiles
            // stay well-populated without misaligning HCA compressor windows.
            let stepSize = (windowSize == nil || windowSize == 512) ? 2048 : max(windowSize!, 1)
            var flatTokens = input.text.tokens.reshaped([-1])
            let totalTokens = flatTokens.size
            guard totalTokens > 0 else { return .tokens(input.text) }
            while flatTokens.size > stepSize {
                try Task.checkCancellation()
                let step = min(stepSize, flatTokens.size - 1)
                let chunkTokens = flatTokens[..<step][.newAxis, 0...]
                // Intermediate prefill chunks discard logits: skip lm_head completely.
                _ = model(chunkTokens, cache: cache)
                MLX.eval(cache)
                try error.check()
                flatTokens = flatTokens[step...]
                PrefillProgressReporter.reportCompletedUnits(totalTokens - flatTokens.size)
            }
            // Final chunk: compute lm_head ONLY for the last token.
            let finalTokens = flatTokens[.newAxis, 0...]
            let hidden = model(finalTokens, cache: cache)
            let lastHidden = hidden[0..., (hidden.dim(1) - 1)..., 0...]
            let logits = projectLogits(lastHidden)
            MLX.eval(logits)
            try error.check()
            PrefillProgressReporter.reportCompletedUnits(totalTokens)
            if totalTokens > 512 { Memory.clearCache() }
            return .logits(LMOutput(logits: logits))
        }
    }

    /// Called after the loader has installed the final projection modules.
    func prepareDecode() {
        for layer in model.layers { layer.prepare() }
        guard HardwareInfo.isCompiledDecodeSupported else { return }
        for layer in model.layers { layer.moe?.compileDecode() }
    }

    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        args.compressRatios.map { ratio -> KVCache in
            if ratio == -2 { return CacheList(KVCacheSimple(), KVCacheSimple()) }
            return DeepseekV4Cache(slidingWindow: args.slidingWindow, compressRatio: ratio,
                                   poolQuantizationEnabled: false, compactLocalWindow: true, sharesLocalKeysAndValues: true)
        }
    }

    func excludeFromGenericSafetensorsLoad(key: String) -> Bool {
        excludedTensorNames.contains(key)
    }
}

/// ERNIE's native renderer writes an assistant tool-call turn back with its closing `<|ernie:im_end|>`
/// stop token, so tool-call turns can persist a post-answer SSD boundary for exact-prefix follow-up reuse.
extension Ernie55Model: ToolCallTurnStopTokenModel {
    public var toolCallTurnStopTokenIds: Set<Int> { [args.eosTokenId ?? 2] }
}
