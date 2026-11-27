// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN

// Native ERNIE 5.5 Flash Preview DSpark speculative drafter and target verify/commit hooks.
//
// Sidecar layout in `dspark/`:
// - `draft-manifest.json`: block_size (7), mask_token (1024), features (1-based target layer ids
//   [40, 41, 42] -> 0-based [39, 40, 41]), layers (3), expert_quantization (3-bit jangtq2 H32),
//   feature_quantization (4-bit affine g64 with input_scale).
// - `draft-head.safetensors`: feature_affine4.{weight,scales,biases,input_scale} (or FP8
//   feature_projection), feature_norm, mhc_contract.hc_head_{fn,base,scale}, norm,
//   markov_head.markov_w1/w2, confidence_head.
// - `draft-layer-{0..N-1}.safetensors`: 3 ERNIE HCA-window layers (no compressor, sliding window
//   128, non-causal over block + context KV) + 4-stream mHC + 512-expert top-10 JANGH3 W4A8 MoE.
//
// Adaptive (`DSparkDecoder`) chooses draft width per step across 1...ladderTop (2 * blockSize);
// `VMLX_DSPARK_WIDTH` pins width for diagnostics only, `VMLX_DSPARK_MARKOV_CANDIDATES` (default
// 64; 0 = full vocab) sets top-K Markov rescoring width.

// MARK: - Verify Capture

/// Per-layer cache inputs captured during one ERNIE verify forward for zero-refeed prefix commit.
final class Ernie55VerifyCapture {
    enum LayerRecord {
        case hca(offset: Int, kv: MLXArray, x: MLXArray)
        case mla(offset: Int, kv: MLXArray, kPos: MLXArray, indexKeys: MLXArray)

        var offset: Int {
            switch self {
            case .hca(let offset, _, _): return offset
            case .mla(let offset, _, _, _): return offset
            }
        }
    }

    private(set) var tokens = 0
    private(set) var layers: [Int: LayerRecord] = [:]
    private(set) var valid = true
    private var forwards = 0

    func begin(tokens: Int) {
        forwards += 1
        self.tokens = tokens
        layers.removeAll(keepingCapacity: true)
        if forwards > 1 { valid = false }
    }

    func record(layer: Int, _ record: LayerRecord) {
        if layers[layer] != nil { valid = false }
        layers[layer] = record
    }
}

// MARK: - Manifest & Config

struct Ernie55DSparkConfig: Sendable {
    let blockSize: Int
    let maskTokenId: Int
    let layers: Int
    /// 0-based target decoder layer indices whose post-layer mean-over-streams hidden is tapped.
    let targetLayerIndices: [Int]
    let experts: Int
    let expertBits: Int
    let expertRotation: String
    let featureBits: Int
    let featureGroupSize: Int
    let slidingWindow: Int
    let hiddenSize: Int
    let vocabularySize: Int

    enum LoadError: Error, CustomStringConvertible {
        case missingFile(URL)
        case missingTensor(String)
        case unexpected(String)

        var description: String {
            switch self {
            case .missingFile(let url): return "ERNIE DSpark sidecar file missing: \(url.path)"
            case .missingTensor(let key): return "ERNIE DSpark sidecar tensor missing: \(key)"
            case .unexpected(let detail): return "ERNIE DSpark sidecar invalid: \(detail)"
            }
        }
    }

    static func decode(directory: URL, target: Ernie55Configuration) throws -> Self {
        let url = directory.appendingPathComponent("draft-manifest.json")
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw LoadError.missingFile(url)
        }
        let data = try Data(contentsOf: url)
        guard let raw = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw LoadError.unexpected("draft-manifest.json is not a JSON object")
        }
        func reqInt(_ dict: [String: Any], _ key: String) throws -> Int {
            guard let v = (dict[key] as? NSNumber)?.intValue ?? (dict[key] as? Int) else {
                throw LoadError.unexpected("missing integer \(key) in draft-manifest.json")
            }
            return v
        }
        let blockSize = try reqInt(raw, "block_size")
        let maskTokenId = try reqInt(raw, "mask_token")
        let layers = try reqInt(raw, "layers")
        let experts = try reqInt(raw, "experts")
        guard let features = (raw["features"] as? [Any])?.compactMap({ ($0 as? NSNumber)?.intValue ?? ($0 as? Int) }),
            !features.isEmpty
        else {
            throw LoadError.unexpected("missing features array in draft-manifest.json")
        }
        // Manifest `features` are 1-based decoder layer ordinals (e.g. [40, 41, 42] -> 0-based [39, 40, 41]).
        let zeroBased = features.map { $0 - 1 }
        guard zeroBased.allSatisfy({ (0 ..< target.numHiddenLayers).contains($0) }) else {
            throw LoadError.unexpected("draft features \(features) out of range for \(target.numHiddenLayers) layers")
        }
        guard experts == target.nRoutedExperts else {
            throw LoadError.unexpected("draft experts \(experts) != target \(target.nRoutedExperts)")
        }
        let eq = raw["expert_quantization"] as? [String: Any] ?? [:]
        let expertBits = (eq["bits"] as? NSNumber)?.intValue ?? 3
        let expertRotation = eq["rotation"] as? String ?? "hadamard32"
        let fq = raw["feature_quantization"] as? [String: Any] ?? [:]
        let featureBits = (fq["bits"] as? NSNumber)?.intValue ?? 4
        let featureGroupSize = (fq["group_size"] as? NSNumber)?.intValue ?? 64
        return Self(
            blockSize: blockSize, maskTokenId: maskTokenId, layers: layers,
            targetLayerIndices: zeroBased, experts: experts,
            expertBits: expertBits, expertRotation: expertRotation,
            featureBits: featureBits, featureGroupSize: featureGroupSize,
            slidingWindow: target.slidingWindow, hiddenSize: target.hiddenSize,
            vocabularySize: target.vocabSize)
    }
}

// MARK: - Draft Attention (HCA without compressor, non-causal over block + context sliding window)

final class Ernie55DSparkAttention: Module {
    let shape: Ernie55AttentionShape
    let window: Int
    let outputGroups: Int
    let outputRank: Int

    @ModuleInfo(key: "wq_a") var queryDown: Linear
    @ModuleInfo(key: "wq_b") var queryUp: Linear
    @ModuleInfo(key: "wkv") var keyValue: Linear
    @ModuleInfo(key: "q_norm") var queryNorm: RMSNorm
    @ModuleInfo(key: "kv_norm") var keyValueNorm: RMSNorm
    @ModuleInfo(key: "wo_b") var output: Linear
    @ModuleInfo(key: "o_gate") var gate: Linear
    @ModuleInfo(key: "vha") var virtualHeads: Ernie55VirtualHeads
    @ParameterInfo(key: "o_group_weight") var groupedOutput: MLXArray
    @ParameterInfo(key: "o_group_scales") var groupedOutputScales: MLXArray
    @ParameterInfo(key: "attn_sink") var sinks: MLXArray
    private var _attnSinkBF16: MLXArray?
    private var _qbOGWeight: MLXArray?
    private var _qbOGScales: MLXArray?

    init(shape: Ernie55AttentionShape, window: Int, outputGroups: Int, outputRank: Int) {
        self.shape = shape
        self.window = window
        self.outputGroups = outputGroups
        self.outputRank = outputRank
        _queryDown.wrappedValue = Linear(shape.hidden, shape.queryRank, bias: false)
        _queryUp.wrappedValue = Linear(shape.queryRank, shape.heads * shape.headWidth, bias: false)
        _keyValue.wrappedValue = Linear(shape.hidden, shape.headWidth, bias: false)
        _queryNorm.wrappedValue = RMSNorm(dimensions: shape.queryRank, eps: shape.epsilon)
        _keyValueNorm.wrappedValue = RMSNorm(dimensions: shape.headWidth, eps: shape.epsilon)
        _output.wrappedValue = Linear(outputGroups * outputRank, shape.hidden, bias: false)
        _gate.wrappedValue = Linear(shape.queryRank, outputGroups * outputRank, bias: false)
        _virtualHeads.wrappedValue = Ernie55VirtualHeads(heads: shape.heads, rank: shape.virtualHeadRank)
        let groupIn = shape.heads * shape.headWidth / outputGroups
        _groupedOutput.wrappedValue = MLXArray.zeros([outputGroups, outputRank, groupIn / 4], dtype: .uint32)
        _groupedOutputScales.wrappedValue = MLXArray.zeros([outputGroups, outputRank, groupIn / 32], dtype: .uint8)
        _sinks.wrappedValue = MLXArray.zeros([shape.heads])
        super.init()
    }

    func prepare() {
        virtualHeads.prepare()
        let sinkBF16 = contiguous(sinks.asType(.bfloat16))
        eval(sinkBF16)
        _attnSinkBF16 = sinkBF16
        guard ProcessInfo.processInfo.environment["VMLX_ERNIE_FUSE_PROJ"] != "0" else { return }
        if _qbOGWeight == nil,
            let qb = queryUp as? QuantizedLinear, qb.mode == .mxfp8, qb.bias == nil,
            let og = gate as? QuantizedLinear, og.mode == .mxfp8, og.bias == nil,
            qb.weight.dim(1) == og.weight.dim(1)
        {
            let w = contiguous(concatenated([qb.weight, og.weight], axis: 0))
            let s = contiguous(concatenated([qb.scales, og.scales], axis: 0))
            eval(w, s)
            _qbOGWeight = w
            _qbOGScales = s
            Ernie55Kernels.releaseMXFP8(qb)
            Ernie55Kernels.releaseMXFP8(og)
        }
    }

    /// Committed context KV rows `[1, 1, C, headWidth]` rotated at positions `offset ..< offset + C`.
    func contextKV(_ context: MLXArray, rope: DeepseekV4RoPE, offset: Int) -> MLXArray {
        let count = context.dim(1)
        let phase = rope.cosSin(offset: offset, length: count)
        let raw = Ernie55Kernels.mxfp8Linear(keyValue, context)
        let normed = keyValueNorm(raw).expandedDimensions(axis: 1)
        return DeepseekV4Math.applyPartialRoPE(
            normed, cos: phase.cos, sin: phase.sin, ropeDim: shape.ropeWidth)
    }

    func callAsFunction(
        _ x: MLXArray, rope: DeepseekV4RoPE, contextKV: MLXArray?, contextStart: Int, offset: Int
    ) -> MLXArray {
        let (batch, length) = (x.dim(0), x.dim(1))
        let phase = rope.cosSin(offset: offset, length: length)
        let queryRank = queryNorm(Ernie55Kernels.mxfp8Linear(queryDown, x))
        let rawQUp: MLXArray
        let rawGate: MLXArray?
        if let w = _qbOGWeight, let s = _qbOGScales {
            let combined = Ernie55Kernels.mxfp8Rows(queryRank, weight: w, scales: s)
                ?? quantizedMM(queryRank, w, scales: s, biases: nil, transpose: true, groupSize: 32, bits: 8, mode: .mxfp8)
            let qWidth = shape.heads * shape.headWidth
            rawQUp = combined[.ellipsis, ..<qWidth]
            rawGate = combined[.ellipsis, qWidth...]
        } else {
            rawQUp = Ernie55Kernels.mxfp8Linear(queryUp, queryRank)
            rawGate = nil
        }
        let unrotated = rawQUp.reshaped(batch, length, shape.heads, shape.headWidth)
        let (q, ownKV) = prepareAttentionRows(
            unrotated, rawKV: Ernie55Kernels.mxfp8Linear(keyValue, x), phase: phase)
        let kv = contextKV.map { concatenated([$0, ownKV], axis: 2) } ?? ownKV

        var mask: MLXArray? = nil
        if let contextKV {
            let ctxLen = contextKV.dim(2)
            let lastQueryFloor = offset + length - 1 - window + 1
            if lastQueryFloor > contextStart {
                let queries = MLXArray(Int32(offset) ..< Int32(offset + length)).reshaped(length, 1)
                let keys = MLXArray(Int32(offset - ctxLen) ..< Int32(offset + length)).reshaped(1, ctxLen + length)
                mask = keys .>= (queries - Int32(window - 1))
            }
        }

        let typedSinks = (q.dtype == .bfloat16 ? _attnSinkBF16 : nil) ?? sinks.asType(q.dtype)
        let attended = MLXFast.scaledDotProductAttention(
            queries: q, keys: kv, values: kv, scale: 1 / sqrt(Float(shape.headWidth)),
            mask: mask, sinks: typedSinks)
        let groups: MLXArray
        if batch == 1, let fusedGroups = Ernie55Kernels.invRoPEVHAGroup(
            attended, cos: phase.cos, sin: phase.sin, vhaMix: virtualHeads.mixMatrix,
            ropeWidth: shape.ropeWidth)
        {
            groups = fusedGroups
        } else {
            let inverted = DeepseekV4Math.applyPartialRoPE(
                attended, cos: phase.cos, sin: -phase.sin, ropeDim: shape.ropeWidth)
            let mixed = virtualHeads(inverted.transposed(0, 2, 1, 3))
            groups = mixed.reshaped(batch, length, outputGroups, -1).transposed(0, 2, 1, 3)
        }
        let reducedGroups: MLXArray
        if groupedOutputScales.size > 0 {
            let g3 = groups.ndim == 3 ? groups : groups.squeezed(axis: 0)
            let qOut = Ernie55Kernels.mxfp8Rows(g3, weight: groupedOutput, scales: groupedOutputScales)
                ?? DeepseekV41FP8Rows.rows(g3, weight: groupedOutput, scales: groupedOutputScales)
                ?? quantizedMM(g3, groupedOutput, scales: groupedOutputScales, biases: nil,
                               transpose: true, groupSize: 32, bits: 8, mode: .mxfp8)
            reducedGroups = qOut.transposed(1, 0, 2).reshaped(batch, length, outputGroups * outputRank)
        } else {
            let g4 = groups.ndim == 4 ? groups : groups.expandedDimensions(axis: 0)
            reducedGroups = matmul(g4, groupedOutput.swappedAxes(-1, -2))
                .transposed(0, 2, 1, 3).reshaped(batch, length, outputGroups * outputRank)
        }
        let gateInput = rawGate ?? Ernie55Kernels.mxfp8Linear(gate, queryRank)
        let gated = reducedGroups * sigmoid(gateInput)
        return Ernie55Kernels.mxfp8Linear(output, gated)
    }

    private func prepareAttentionRows(
        _ query: MLXArray, rawKV: MLXArray, phase: (cos: MLXArray, sin: MLXArray)
    ) -> (MLXArray, MLXArray) {
        if let fused = Ernie55Kernels.prepareHCARows(
            query, rawKV: rawKV, keyWeight: keyValueNorm.weight,
            cos: phase.cos, sin: phase.sin, heads: shape.heads,
            ropeWidth: shape.ropeWidth, epsilon: shape.epsilon)
        {
            return fused
        }
        let normalized = MLXFast.rmsNorm(
            query, weight: MLXArray.ones([shape.headWidth], dtype: query.dtype), eps: shape.epsilon)
        return (
            DeepseekV4Math.applyPartialRoPE(
                normalized.transposed(0, 2, 1, 3), cos: phase.cos, sin: phase.sin, ropeDim: shape.ropeWidth),
            DeepseekV4Math.applyPartialRoPE(
                keyValueNorm(rawKV).expandedDimensions(axis: 1),
                cos: phase.cos, sin: phase.sin, ropeDim: shape.ropeWidth)
        )
    }
}

// MARK: - Draft Layer

final class Ernie55DSparkLayer: Module {
    @ModuleInfo var attn: Ernie55DSparkAttention
    @ModuleInfo var moe: Ernie55MoE
    @ModuleInfo(key: "attn_hc") var attentionConnection: Ernie55HyperConnection
    @ModuleInfo(key: "ffn_hc") var feedForwardConnection: Ernie55HyperConnection
    @ModuleInfo(key: "attn_norm") var attentionNorm: RMSNorm
    @ModuleInfo(key: "ffn_norm") var feedForwardNorm: RMSNorm

    init(_ c: Ernie55Configuration, routed: any Ernie55RoutedExperts) {
        _attn.wrappedValue = Ernie55DSparkAttention(
            shape: c.hcaShape, window: c.slidingWindow, outputGroups: c.oGroups,
            outputRank: c.oLoraRank)
        _moe.wrappedValue = Ernie55MoE(c, routed: routed)
        _attentionConnection.wrappedValue = Ernie55HyperConnection(c)
        _feedForwardConnection.wrappedValue = Ernie55HyperConnection(c)
        _attentionNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
        _feedForwardNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
        super.init()
    }

    func prepare() {
        attn.prepare()
        moe.prepare()
    }

    func callAsFunction(
        _ x: MLXArray, rope: DeepseekV4RoPE, contextKV: MLXArray?, contextStart: Int, offset: Int
    ) -> MLXArray {
        let (input, post, mix) = attentionConnection(x)
        let attended = attn(
            attentionNorm(input), rope: rope, contextKV: contextKV, contextStart: contextStart, offset: offset)
        let residual = Ernie55Math.expand(attended, residual: x, post: post, mix: mix)
        let (hidden, nextPost, nextMix) = feedForwardConnection(residual)
        let output = moe(feedForwardNorm(hidden))
        return Ernie55Math.expand(output, residual: residual, post: nextPost, mix: nextMix)
    }

    func callFused(
        prevStream: MLXArray, prevOut: MLXArray?, prevPost: MLXArray?, prevComb: MLXArray?,
        rope: DeepseekV4RoPE, contextKV: MLXArray?, contextStart: Int, offset: Int
    ) -> (stream: MLXArray, out: MLXArray, post: MLXArray, comb: MLXArray)? {
        guard let attnHC = Ernie55Kernels.fusedHCCollapseNorm(
            h: prevStream, prevOut: prevOut, prevPost: prevPost, prevComb: prevComb,
            fn: attentionConnection.fn, scale: attentionConnection.scale, base: attentionConnection.base,
            normWeight: attentionNorm.weight, iterations: attentionConnection.iterations,
            mixingEpsilon: attentionConnection.mixingEpsilon, normEpsilon: attentionConnection.normEpsilon,
            rmsEpsilon: attentionNorm.eps)
        else { return nil }
        let attended = attn(
            attnHC.normed, rope: rope, contextKV: contextKV, contextStart: contextStart, offset: offset)
        guard let ffnHC = Ernie55Kernels.fusedHCCollapseNorm(
            h: attnHC.stream, prevOut: attended, prevPost: attnHC.post, prevComb: attnHC.comb,
            fn: feedForwardConnection.fn, scale: feedForwardConnection.scale, base: feedForwardConnection.base,
            normWeight: feedForwardNorm.weight, iterations: feedForwardConnection.iterations,
            mixingEpsilon: feedForwardConnection.mixingEpsilon, normEpsilon: feedForwardConnection.normEpsilon,
            rmsEpsilon: feedForwardNorm.eps)
        else { return nil }
        let output = moe(ffnHC.normed)
        return (ffnHC.stream, output, ffnHC.post, ffnHC.comb)
    }
}

// MARK: - Feature Projection

enum Ernie55DSparkFeatureProjection {
    case affine(weight: MLXArray, scales: MLXArray, biases: MLXArray, inputScale: MLXArray, bits: Int, groupSize: Int)
    case mxfp8(weight: MLXArray, scales: MLXArray)

    func callAsFunction(_ features: MLXArray) -> MLXArray {
        switch self {
        case .affine(let w, let s, let b, let inputScale, let bits, let groupSize):
            let scaled = (features.asType(.float32) / inputScale).asType(features.dtype)
            return quantizedMM(
                scaled, w, scales: s, biases: b, transpose: true,
                groupSize: groupSize, bits: bits, mode: .affine)
        case .mxfp8(let w, let s):
            return Ernie55Kernels.mxfp8Rows(features, weight: w, scales: s)
                ?? DeepseekV41FP8Rows.rows(features, weight: w, scales: s)
                ?? quantizedMM(features, w, scales: s, biases: nil, transpose: true,
                               groupSize: 32, bits: 8, mode: .mxfp8)
        }
    }
}

// MARK: - Draft Context (per request)

struct Ernie55DSparkContext {
    var kv: [MLXArray]
    var start: Int
    var end: Int
    var count: Int { end - start }

    static func empty(at position: Int) -> Self { Self(kv: [], start: position, end: position) }

    mutating func append(_ rows: [MLXArray]) {
        guard let n = rows.first?.dim(2), n > 0 else { return }
        if kv.isEmpty {
            kv = rows
        } else {
            kv = zip(kv, rows).map { concatenated([$0, $1], axis: 2) }
        }
        end += n
    }

    mutating func keepLast(_ n: Int) {
        guard n > 0, count > n else { return }
        let drop = count - n
        kv = kv.map { $0[0..., 0..., drop..., 0...] }
        start += drop
    }
}

// MARK: - Head (weights shared across requests)

final class Ernie55DSparkHead: @unchecked Sendable {
    let config: Ernie55DSparkConfig
    let targetConfig: Ernie55Configuration
    let rope: DeepseekV4RoPE
    let layers: [Ernie55DSparkLayer]
    let featureProj: Ernie55DSparkFeatureProjection
    let featureNorm: MLXArray
    let hcHead: Ernie55HyperHead
    let normWeight: MLXArray
    /// `[vocab, rank]` BF16 (transposed once from checkpoint `[rank, vocab]`).
    let markovEmbed: MLXArray
    /// `[vocab, rank]` BF16.
    let markovHead: MLXArray
    let embed: (MLXArray) -> MLXArray
    let rawHead: (MLXArray) -> MLXArray
    let multiMax: (MLXArray) -> MLXArray
    let multiMaxRanges: MLXArray?
    let multiMaxSlopes: MLXArray?
    let candidates: Int

    init(
        config: Ernie55DSparkConfig, targetConfig: Ernie55Configuration,
        rope: DeepseekV4RoPE, layers: [Ernie55DSparkLayer],
        featureProj: Ernie55DSparkFeatureProjection,
        featureNorm: MLXArray, hcHead: Ernie55HyperHead, normWeight: MLXArray,
        markovEmbed: MLXArray, markovHead: MLXArray,
        embed: @escaping (MLXArray) -> MLXArray,
        rawHead: @escaping (MLXArray) -> MLXArray,
        multiMax: @escaping (MLXArray) -> MLXArray,
        multiMaxRanges: MLXArray? = nil,
        multiMaxSlopes: MLXArray? = nil,
        candidates: Int = Ernie55DSpark.markovCandidates
    ) {
        self.config = config
        self.targetConfig = targetConfig
        self.rope = rope
        self.layers = layers
        self.featureProj = featureProj
        self.featureNorm = featureNorm
        self.hcHead = hcHead
        self.normWeight = normWeight
        self.markovEmbed = markovEmbed
        self.markovHead = markovHead
        self.embed = embed
        self.rawHead = rawHead
        self.multiMax = multiMax
        self.multiMaxRanges = multiMaxRanges
        self.multiMaxSlopes = multiMaxSlopes
        self.candidates = candidates
    }

    /// Project committed target-layer taps `[1, C, 3 * hidden]` to per-layer context KV `[1, 1, C, 512]`.
    func contextRows(_ features: MLXArray, offset: Int) -> [MLXArray] {
        let projected = featureProj(features)
        let ctx = MLXFast.rmsNorm(projected, weight: featureNorm, eps: targetConfig.rmsNormEps)
        return layers.map { $0.attn.contextKV(ctx, rope: rope, offset: offset) }
    }

    /// Draft `count` tokens after `anchor` at `start` (the anchor's position) over `context` (ending at `start`).
    func propose(
        anchor: Int, start: Int, count: Int, context: Ernie55DSparkContext,
        noise: MLXArray? = nil, inverseTemperature: Float = 1
    ) -> MLXArray {
        let span = max(config.blockSize, count)
        var ids = [Int32](repeating: Int32(config.maskTokenId), count: span)
        ids[0] = Int32(anchor)
        let embeds = embed(MLXArray(ids).reshaped(1, span))
        var x = broadcast(
            embeds.expandedDimensions(axis: -2),
            to: [1, span, targetConfig.hcMult, targetConfig.hiddenSize])

        if (1...16).contains(span) && Device.defaultDevice().deviceType == .gpu {
            var stream = contiguous(x)
            var pendingOut: MLXArray? = nil
            var pendingPost: MLXArray? = nil
            var pendingComb: MLXArray? = nil
            var usedFused = true
            for (i, layer) in layers.enumerated() {
                let ctxKV = i < context.kv.count ? context.kv[i] : nil
                if let step = layer.callFused(
                    prevStream: stream, prevOut: pendingOut, prevPost: pendingPost, prevComb: pendingComb,
                    rope: rope, contextKV: ctxKV, contextStart: context.start, offset: start)
                {
                    stream = step.stream
                    pendingOut = step.out
                    pendingPost = step.post
                    pendingComb = step.comb
                } else {
                    if let out = pendingOut, let post = pendingPost, let comb = pendingComb {
                        stream = Ernie55Math.expand(out, residual: stream, post: post, mix: comb)
                    }
                    x = layer(stream, rope: rope, contextKV: ctxKV, contextStart: context.start, offset: start)
                    for rem in (i + 1) ..< layers.count {
                        let remKV = rem < context.kv.count ? context.kv[rem] : nil
                        x = layers[rem](x, rope: rope, contextKV: remKV, contextStart: context.start, offset: start)
                    }
                    usedFused = false
                    break
                }
            }
            if usedFused, let out = pendingOut, let post = pendingPost, let comb = pendingComb {
                x = Ernie55Math.expand(out, residual: stream, post: post, mix: comb)
            }
        } else {
            for (i, layer) in layers.enumerated() {
                let ctxKV = i < context.kv.count ? context.kv[i] : nil
                x = layer(x, rope: rope, contextKV: ctxKV, contextStart: context.start, offset: start)
            }
        }

        let collapsed = hcHead(x)[0..., ..<count, 0...]
        let normed = MLXFast.rmsNorm(collapsed, weight: normWeight, eps: targetConfig.rmsNormEps)
        let base = rawHead(normed)[0].asType(.float32)  // [count, vocab]

        var tokens: [MLXArray] = []
        tokens.reserveCapacity(count)
        let vocab = config.vocabularySize
        // For coupled sampled speculation (noise != nil), use a wider candidate set (or full vocab)
        // so Gumbel draws from the top-p nucleus are not clipped by a narrow greedy top-K.
        let effectiveCandidates = (noise != nil && candidates > 0) ? max(candidates, 256) : candidates
        // Fast path for greedy full-vocab Markov selection (single fused 512x256 Metal kernel,
        // matching vMLX Python `ernie_dspark_markov_argmax` with zero 201K-element intermediates).
        if noise == nil, let ranges = multiMaxRanges, let slopes = multiMaxSlopes {
            var previous = MLXArray(Int32(anchor))
            for i in 0 ..< count {
                let transition = markovEmbed[previous]
                let token: MLXArray
                if let fast = Ernie55Kernels.dsparkMarkovArgmax(
                    baseRow: base[i], markovHead: markovHead, transition: transition,
                    ranges: ranges, ts: slopes)
                {
                    token = fast
                } else {
                    let bias = matmul(markovHead, transition).asType(.float32)
                    token = argMax(multiMax(base[i] + bias)).asType(.int32)
                }
                tokens.append(token.reshaped(1))
                previous = token
            }
        } else if effectiveCandidates > 0 && effectiveCandidates < vocab {
            let k = effectiveCandidates
            let cand = argPartition(-base, kth: k - 1, axis: -1)[0..., ..<k]  // [count, K]
            let candBase = takeAlong(base, cand, axis: -1)  // [count, K]
            var previous = MLXArray(Int32(anchor))
            for i in 0 ..< count {
                let transition = markovEmbed[previous].asType(.float32)  // [rank]
                let bias = matmul(markovHead[cand[i]].asType(.float32), transition)  // [K]
                var score = multiMax(candBase[i] + bias)
                if let noise {
                    score = score * inverseTemperature + take(noise[i], cand[i], axis: 0)
                }
                let token = take(cand[i], argMax(score), axis: 0).asType(.int32)
                tokens.append(token.reshaped(1))
                previous = token
            }
        } else {
            var previous = MLXArray(Int32(anchor))
            for i in 0 ..< count {
                let transition = markovEmbed[previous]  // [rank] BF16
                let bias = matmul(markovHead, transition).asType(.float32)  // [vocab]
                var score = multiMax(base[i] + bias)
                if let noise {
                    score = score * inverseTemperature + noise[i]
                }
                let token = argMax(score).asType(.int32)
                tokens.append(token.reshaped(1))
                previous = token
            }
        }
        return concatenated(tokens, axis: 0)
    }
}

// MARK: - Loader

extension Ernie55DSparkHead {
    private static let oddCubicCodebookParams: [Int: (alpha: Double, beta: Double)] = [
        2: (0.8929999999999999, 0.05065),
        3: (0.47124999999999995, 0.011599999999999997),
    ]

    static func load(
        directory: URL, targetConfig: Ernie55Configuration,
        embed: @escaping (MLXArray) -> MLXArray,
        rawHead: @escaping (MLXArray) -> MLXArray,
        multiMax: @escaping (MLXArray) -> MLXArray,
        multiMaxRanges: MLXArray? = nil,
        multiMaxSlopes: MLXArray? = nil
    ) throws -> Ernie55DSparkHead {
        let cfg = try Ernie55DSparkConfig.decode(directory: directory, target: targetConfig)
        guard let cb = oddCubicCodebookParams[cfg.expertBits] else {
            throw Ernie55DSparkConfig.LoadError.unexpected("unsupported JANGH expert bits \(cfg.expertBits)")
        }
        // Baidu ERNIE 5.5 DSpark draft layers use plain RoPE (factor = 1.0, base = rope_theta = 8,000,000),
        // NOT the target HCA layers' YaRN-scaled compress_rope_theta (160,000, factor = 16).
        let rope = DeepseekV4RoPE(
            dim: targetConfig.qkRopeHeadDim, base: targetConfig.ropeTheta, factor: 1.0)

        let caps = targetConfig.caps
        var layers: [Ernie55DSparkLayer] = []
        layers.reserveCapacity(cfg.layers)
        for index in 0 ..< cfg.layers {
            let shardURL = directory.appendingPathComponent("draft-layer-\(index).safetensors")
            guard FileManager.default.fileExists(atPath: shardURL.path) else {
                throw Ernie55DSparkConfig.LoadError.missingFile(shardURL)
            }
            let raw = try loadArrays(url: shardURL)
            let prefix = "model.layers.\(index)."
            func req(_ suffix: String) throws -> MLXArray {
                guard let t = raw[prefix + suffix] else {
                    throw Ernie55DSparkConfig.LoadError.missingTensor(prefix + suffix)
                }
                return t
            }
            let routed = try JANGTRoutedExpertLayer(
                codebookBits: cfg.expertBits, rotation: cfg.expertRotation,
                alpha: cb.alpha, beta: cb.beta,
                gatePacked: try req("ffn.switch_mlp.gate_proj.tq2_packed"),
                gateScales: try req("ffn.switch_mlp.gate_proj.tq2_scales"),
                upPacked: try req("ffn.switch_mlp.up_proj.tq2_packed"),
                upScales: try req("ffn.switch_mlp.up_proj.tq2_scales"),
                downPacked: try req("ffn.switch_mlp.down_proj.tq2_packed"),
                downScales: try req("ffn.switch_mlp.down_proj.tq2_scales"),
                hiddenSize: targetConfig.moeLatentSize,
                intermediateSize: targetConfig.moeIntermediateSize,
                activationLimit: nil,
                activation: .ernie55(gateCap: caps.gate, upCap: caps.up))

            let layer = Ernie55DSparkLayer(targetConfig, routed: routed)
            let mapped = sanitizeDraftLayer(raw, prefix: prefix, targetConfig: targetConfig)
            quantize(model: layer) { path, module -> (groupSize: Int, bits: Int, mode: QuantizationMode)? in
                guard module is Linear, mapped[path + ".scales"] != nil else { return nil }
                return (32, 8, .mxfp8)
            }
            do {
                try layer.update(parameters: ModuleParameters.unflattened(mapped), verify: .all)
            } catch {
                throw Ernie55DSparkConfig.LoadError.unexpected("draft layer \(index): \(error)")
            }
            layer.prepare()
            eval(layer)
            layers.append(layer)
        }

        let headURL = directory.appendingPathComponent("draft-head.safetensors")
        guard FileManager.default.fileExists(atPath: headURL.path) else {
            throw Ernie55DSparkConfig.LoadError.missingFile(headURL)
        }
        let headRaw = try loadArrays(url: headURL)
        func reqHead(_ key: String) throws -> MLXArray {
            guard let t = headRaw[key] else { throw Ernie55DSparkConfig.LoadError.missingTensor(key) }
            return t
        }

        let featureProj: Ernie55DSparkFeatureProjection
        if let fw = headRaw["feature_affine4.weight"] {
            featureProj = .affine(
                weight: fw,
                scales: try reqHead("feature_affine4.scales"),
                biases: try reqHead("feature_affine4.biases"),
                inputScale: try reqHead("feature_affine4.input_scale").asType(.float32),
                bits: cfg.featureBits,
                groupSize: cfg.featureGroupSize)
        } else {
            let (w, s) = Ernie55Model.unpackFP8(
                try reqHead("feature_projection.weight"),
                scale: try reqHead("feature_projection.weight_scale_inv"),
                keepPacked: true)
            guard let s else {
                throw Ernie55DSparkConfig.LoadError.unexpected("feature_projection must be exact MXFP8")
            }
            featureProj = .mxfp8(weight: w, scales: s)
        }

        let hcHead = Ernie55HyperHead(targetConfig)
        let hcParams: [String: MLXArray] = [
            "fn": contiguous(try reqHead("mhc_contract.hc_head_fn").transposed().asType(.float32)),
            "base": try reqHead("mhc_contract.hc_head_base").asType(.float32),
            "scale": try reqHead("mhc_contract.hc_head_scale").reshaped(-1).asType(.float32),
        ]
        try hcHead.update(parameters: ModuleParameters.unflattened(hcParams), verify: .all)

        let featureNorm = try reqHead("feature_norm.weight")
        let normWeight = try reqHead("norm.weight")
        let markovEmbed = contiguous(try reqHead("markov_head.markov_w1.weight").transposed().asType(.bfloat16))
        let markovHead = contiguous(try reqHead("markov_head.markov_w2.weight").asType(.bfloat16))
        eval(hcHead, featureNorm, normWeight, markovEmbed, markovHead)
        Memory.clearCache()

        return Ernie55DSparkHead(
            config: cfg, targetConfig: targetConfig, rope: rope, layers: layers, featureProj: featureProj,
            featureNorm: featureNorm, hcHead: hcHead, normWeight: normWeight,
            markovEmbed: markovEmbed, markovHead: markovHead,
            embed: embed, rawHead: rawHead, multiMax: multiMax,
            multiMaxRanges: multiMaxRanges, multiMaxSlopes: multiMaxSlopes)
    }

    /// Loads an inline MXFP4 ERNIE 5.5 DSpark drafter (`model.mtp.{0..N-1}.*` + `model.mtp.{feature_*,
    /// mhc_contract.*, norm.*, markov_head.*}`) directly from the bundle's `model.safetensors.index.json`
    /// using uncached `ResidentSafetensorsReader` I/O on only the shards containing `model.mtp.*`.
    static func loadInlineMXFP4(
        directory: URL, targetConfig: Ernie55Configuration,
        embed: @escaping (MLXArray) -> MLXArray,
        rawHead: @escaping (MLXArray) -> MLXArray,
        multiMax: @escaping (MLXArray) -> MLXArray,
        multiMaxRanges: MLXArray? = nil,
        multiMaxSlopes: MLXArray? = nil
    ) throws -> Ernie55DSparkHead {
        let configURL = directory.appendingPathComponent("config.json")
        let configData = try Data(contentsOf: configURL)
        guard let root = try JSONSerialization.jsonObject(with: configData) as? [String: Any] else {
            throw Ernie55DSparkConfig.LoadError.unexpected("config.json is not a JSON object")
        }
        let indexURL = directory.appendingPathComponent("model.safetensors.index.json")
        let indexData = try Data(contentsOf: indexURL)
        guard let indexJSON = try JSONSerialization.jsonObject(with: indexData) as? [String: Any],
            let weightMap = indexJSON["weight_map"] as? [String: String]
        else {
            throw Ernie55DSparkConfig.LoadError.unexpected("invalid model.safetensors.index.json")
        }

        let blockSize = (root["dspark_block_size"] as? NSNumber)?.intValue ?? 7
        let maskTokenId = (root["dspark_noise_token_id"] as? NSNumber)?.intValue ?? 1024
        let rawFeatures = (root["dspark_target_layer_ids"] as? [Any])?.compactMap { ($0 as? NSNumber)?.intValue }
            ?? [targetConfig.numHiddenLayers - 3, targetConfig.numHiddenLayers - 2, targetConfig.numHiddenLayers - 1]
        let zeroBased = rawFeatures.map { $0 - 1 }
        guard !zeroBased.isEmpty, zeroBased.allSatisfy({ (0 ..< targetConfig.numHiddenLayers).contains($0) }) else {
            throw Ernie55DSparkConfig.LoadError.unexpected("invalid dspark_target_layer_ids \(rawFeatures)")
        }

        // Discover inline draft layer indices (e.g. model.mtp.0, model.mtp.1, model.mtp.2).
        var layerSet = Set<Int>()
        for key in weightMap.keys where key.hasPrefix("model.mtp.") {
            let parts = key.split(separator: ".")
            if parts.count > 3, let idx = Int(parts[2]) {
                layerSet.insert(idx)
            }
        }
        let layerIndices = layerSet.sorted()
        guard !layerIndices.isEmpty else {
            throw Ernie55DSparkConfig.LoadError.unexpected("no model.mtp.<N> layers in index")
        }

        let cfg = Ernie55DSparkConfig(
            blockSize: blockSize, maskTokenId: maskTokenId, layers: layerIndices.count,
            targetLayerIndices: zeroBased, experts: targetConfig.nRoutedExperts,
            expertBits: 4, expertRotation: "none",
            featureBits: 8, featureGroupSize: 32,
            slidingWindow: targetConfig.slidingWindow, hiddenSize: targetConfig.hiddenSize,
            vocabularySize: targetConfig.vocabSize)

        let routedLayers = try Ernie55MXFP4Preparation.loadStackedMXFP4Experts(
            directory: directory,
            weightMap: weightMap,
            layerPrefix: "model.mtp.",
            layerIndices: layerIndices,
            expertCount: targetConfig.nRoutedExperts,
            hiddenSize: targetConfig.moeLatentSize,
            intermediateSize: targetConfig.moeIntermediateSize)

        // Load only non-expert model.mtp.* keys from the shards that contain them.
        var shardWanted: [String: Set<String>] = [:]
        for (key, shard) in weightMap where key.hasPrefix("model.mtp.") && !key.contains(".block_sparse_moe.experts.") {
            shardWanted[shard, default: []].insert(key)
        }
        var shardAll: [String: Set<String>] = [:]
        for (key, shard) in weightMap where shardWanted[shard] != nil {
            shardAll[shard, default: []].insert(key)
        }
        var mtpRaw: [String: MLXArray] = [:]
        for shard in shardWanted.keys.sorted() {
            let excluded = (shardAll[shard] ?? []).subtracting(shardWanted[shard] ?? [])
            let url = directory.appendingPathComponent(shard)
            let (loaded, _) = try ResidentSafetensorsReader.loadWeights(url: url, excludingKeys: excluded)
            for (k, v) in loaded { mtpRaw[k] = v }
        }

        let rope = DeepseekV4RoPE(
            dim: targetConfig.qkRopeHeadDim, base: targetConfig.ropeTheta, factor: 1.0)
        var layers: [Ernie55DSparkLayer] = []
        layers.reserveCapacity(layerIndices.count)
        for index in layerIndices {
            guard let routed = routedLayers[index] else {
                throw Ernie55DSparkConfig.LoadError.unexpected("missing routed MXFP4 experts for model.mtp.\(index)")
            }
            let layer = Ernie55DSparkLayer(targetConfig, routed: routed)
            let mapped = sanitizeDraftLayer(mtpRaw, prefix: "model.mtp.\(index).", targetConfig: targetConfig)
            quantize(model: layer) { path, module -> (groupSize: Int, bits: Int, mode: QuantizationMode)? in
                guard module is Linear, mapped[path + ".scales"] != nil else { return nil }
                return (32, 8, .mxfp8)
            }
            do {
                try layer.update(parameters: ModuleParameters.unflattened(mapped), verify: .all)
            } catch {
                throw Ernie55DSparkConfig.LoadError.unexpected("inline MXFP4 draft layer \(index): \(error)")
            }
            layer.prepare()
            eval(layer)
            layers.append(layer)
        }

        func reqHead(_ suffix: String) throws -> MLXArray {
            let key = "model.mtp." + suffix
            guard let t = mtpRaw[key] else { throw Ernie55DSparkConfig.LoadError.missingTensor(key) }
            return t
        }
        let (fw, fs) = Ernie55Model.unpackFP8(
            try reqHead("feature_projection.weight"),
            scale: try reqHead("feature_projection.weight_scale_inv"),
            keepPacked: true)
        guard let fs else {
            throw Ernie55DSparkConfig.LoadError.unexpected("inline feature_projection must be exact MXFP8")
        }
        let featureProj = Ernie55DSparkFeatureProjection.mxfp8(weight: fw, scales: fs)

        let hcHead = Ernie55HyperHead(targetConfig)
        let hcParams: [String: MLXArray] = [
            "fn": contiguous(try reqHead("mhc_contract.hc_head_fn").transposed().asType(.float32)),
            "base": try reqHead("mhc_contract.hc_head_base").asType(.float32),
            "scale": try reqHead("mhc_contract.hc_head_scale").reshaped(-1).asType(.float32),
        ]
        try hcHead.update(parameters: ModuleParameters.unflattened(hcParams), verify: .all)

        let featureNorm = try reqHead("feature_norm.weight")
        let normWeight = try reqHead("norm.weight")
        let markovEmbed = contiguous(try reqHead("markov_head.markov_w1.weight").transposed().asType(.bfloat16))
        let markovHead = contiguous(try reqHead("markov_head.markov_w2.weight").asType(.bfloat16))
        eval(fw, fs, hcHead, featureNorm, normWeight, markovEmbed, markovHead)
        Memory.clearCache()

        return Ernie55DSparkHead(
            config: cfg, targetConfig: targetConfig, rope: rope, layers: layers, featureProj: featureProj,
            featureNorm: featureNorm, hcHead: hcHead, normWeight: normWeight,
            markovEmbed: markovEmbed, markovHead: markovHead,
            embed: embed, rawHead: rawHead, multiMax: multiMax,
            multiMaxRanges: multiMaxRanges, multiMaxSlopes: multiMaxSlopes)
    }

    private static func sanitizeDraftLayer(
        _ raw: [String: MLXArray], prefix: String, targetConfig: Ernie55Configuration
    ) -> [String: MLXArray] {
        let replacements = [
            ("input_layernorm.", "attn_norm."),
            ("post_attention_layernorm.", "ffn_norm."),
            ("block_sparse_moe.", "moe."),
            ("self_attention_hyper_connection.", "attn_hc."),
            ("mlp_hyper_connection.", "ffn_hc."),
            ("self_attn.gate_proj.", "attn.o_gate."),
            ("self_attn.core_attention.", "attn."),
            ("self_attn.", "attn."),
            (".linear_q_down_proj.", ".wq_a."),
            (".linear_q_up_proj.", ".wq_b."),
            (".linear_kv_proj.", ".wkv."),
            (".q_layernorm.", ".q_norm."),
            (".kv_layernorm.", ".kv_norm."),
            (".o_proj.", ".wo_b."),
            (".vha_postmix_U", ".vha.U"),
            (".vha_postmix_V", ".vha.V"),
            (".softmax_offset", ".attn_sink"),
            (".shared_experts.w1.", ".shared_experts.gate_proj."),
            (".shared_experts.w2.", ".shared_experts.down_proj."),
            (".shared_experts.w3.", ".shared_experts.up_proj."),
        ]
        var result: [String: MLXArray] = [:]
        for (name, tensor) in raw where !name.hasSuffix("_scale_inv") {
            guard name.hasPrefix(prefix) else { continue }
            var key = String(name.dropFirst(prefix.count))
            if key.hasPrefix("ffn.switch_mlp.") { continue }
            var value = tensor
            var fp8Scale: MLXArray?
            if let scale = raw[name + "_scale_inv"] {
                let (converted, scaleBytes) = Ernie55Model.unpackFP8(tensor, scale: scale, keepPacked: true)
                value = converted
                fp8Scale = scaleBytes
            }
            for (from, to) in replacements {
                key = key.replacingOccurrences(of: from, with: to)
            }
            if key.hasSuffix(".mapping_proj.weight") {
                key = key.replacingOccurrences(of: ".mapping_proj.weight", with: ".fn")
                value = contiguous(value.transposed().asType(.float32))
            } else if key.hasSuffix("_hc.bias") {
                key = String(key.dropLast(4)) + "base"
            } else if key.hasSuffix(".linear_o_group_proj") {
                key = String(key.dropLast("linear_o_group_proj".count)) + "o_group_weight"
                value = contiguous(value.reshaped(targetConfig.oGroups, targetConfig.oLoraRank, -1))
                if let s = fp8Scale {
                    fp8Scale = contiguous(s.reshaped(targetConfig.oGroups, targetConfig.oLoraRank, -1))
                } else {
                    fp8Scale = MLXArray.zeros([0], dtype: .uint8)
                }
            }
            result[key] = value
            if let fp8Scale {
                result[String(key.dropLast("weight".count)) + "scales"] = fp8Scale
            }
        }
        for kind in ["attn_hc", "ffn_hc"] {
            let keys = ["alpha_pre", "alpha_post", "alpha_res"].map { "\(kind).\($0)" }
            if keys.allSatisfy({ result[$0] != nil }) {
                result["\(kind).scale"] = concatenated(keys.map { result.removeValue(forKey: $0)!.reshaped(-1) })
            }
        }
        return result
    }
}

// MARK: - Runtime Helpers

enum Ernie55DSpark {
    static func tapHolder(_ cache: [KVCache]?) -> DeepseekV4Cache? {
        cache?.first as? DeepseekV4Cache
    }

    static func environmentInt(_ name: String) -> Int? {
        guard let raw = ProcessInfo.processInfo.environment[name]?.trimmingCharacters(in: .whitespaces),
            !raw.isEmpty, let value = Int(raw), value >= 1
        else { return nil }
        return value
    }

    /// `VMLX_DSPARK_MARKOV_CANDIDATES` (default 64; 0 = full vocabulary).
    static var markovCandidates: Int {
        guard let raw = ProcessInfo.processInfo.environment["VMLX_DSPARK_MARKOV_CANDIDATES"],
            let v = Int(raw)
        else { return 64 }
        return max(0, v)
    }

    /// Adaptive draft ladder top: twice the trained block (`VMLX_DSPARK_MAX_DRAFT` overrides for diagnostics).
    static func ladderTop(block: Int) -> Int {
        if let pinned = environmentInt("VMLX_DSPARK_MAX_DRAFT") {
            return pinned
        }
        return max(1, 2 * block)
    }

    /// `VMLX_DSPARK_WIDTH`: pins draft width for diagnostics only; nil lets Adaptive choose.
    static var pinnedWidth: Int? { environmentInt("VMLX_DSPARK_WIDTH") }
}

// MARK: - Session (per request)

final class Ernie55DSparkSession: DSparkDraftSession {
    let head: Ernie55DSparkHead
    weak var model: Ernie55Model?
    private(set) var context: Ernie55DSparkContext

    init(head: Ernie55DSparkHead) {
        self.head = head
        self.context = .empty(at: 0)
    }

    func sync(end: Int, tail: HybridPoolHiddenTail?) {
        let window = head.config.slidingWindow
        guard let tail, tail.endPosition == end, end > 0, !tail.isEmpty else {
            context = .empty(at: end)
            return
        }
        if context.end > end || context.end < tail.startPosition || (context.count == 0 && context.end != end) {
            context = .empty(at: max(tail.startPosition, end - window))
        }
        if context.end < end, let rows = tail.rows(from: context.end, to: end) {
            let fresh = head.contextRows(rows.expandedDimensions(axis: 0), offset: context.end)
            context.append(fresh)
            context.keepLast(window)
            asyncEval(context.kv)
        }
    }

    private func proposal(
        seed: Int, cache: [KVCache], count: Int, noise: MLXArray?, inverseTemperature: Float
    ) -> [Int] {
        let holder = Ernie55DSpark.tapHolder(cache)
        let end = holder?.offset ?? (cache.first?.offset ?? 0)
        sync(end: end, tail: holder?.auxiliaryHiddenTail)
        let tokens = head.propose(
            anchor: seed, start: end, count: count, context: context,
            noise: noise, inverseTemperature: inverseTemperature)
        eval(tokens)
        return tokens.asArray(Int32.self).map(Int.init)
    }

    func draft(seed: Int, cache: [KVCache]) -> [Int] {
        draft(seed: seed, cache: cache, count: head.config.blockSize)
    }

    func draft(seed: Int, cache: [KVCache], count: Int) -> [Int] {
        proposal(seed: seed, cache: cache, count: max(1, count), noise: nil, inverseTemperature: 1)
    }

    func draft(seed: Int, cache: [KVCache], noise: MLXArray, inverseTemperature: Float) -> [Int] {
        draft(seed: seed, cache: cache, count: head.config.blockSize,
              noise: noise, inverseTemperature: inverseTemperature)
    }

    func draft(seed: Int, cache: [KVCache], count: Int, noise: MLXArray, inverseTemperature: Float) -> [Int] {
        let n = max(1, count)
        return proposal(
            seed: seed, cache: cache, count: n, noise: noise[..<n],
            inverseTemperature: inverseTemperature)
    }

    var vocabularySize: Int? { head.config.vocabularySize }

    /// In-place rollback replays only the kept HCA/MLA rows, so zero-accept (keep = 1) needs no second forward.
    var commitsAnyPrefix: Bool { true }

    func beginVerifyCapture(cache: [KVCache]) -> Bool {
        guard model != nil, let holder = Ernie55DSpark.tapHolder(cache) else { return false }
        holder.ernieVerifyCapture = Ernie55VerifyCapture()
        return true
    }

    func commitVerifiedPrefix(
        cache: [KVCache], checkpoint: SpeculativeCacheListCheckpoint, keep: Int
    ) -> Bool {
        model?.dsparkCommitPrefix(cache: cache, checkpoint: checkpoint, keep: keep) ?? false
    }

    func endVerifyCapture(cache: [KVCache]) {
        Ernie55DSpark.tapHolder(cache)?.ernieVerifyCapture = nil
    }
}

// MARK: - Ernie55Model conformance

extension Ernie55Model: DSparkSpeculativeModel {
    private var attachedHead: Ernie55DSparkHead? {
        dsparkAttachLock.lock()
        defer { dsparkAttachLock.unlock() }
        return dsparkHead
    }

    public var dsparkBlockSize: Int? { attachedHead?.config.blockSize }

    public var dsparkLadderTop: Int? { attachedHead.map { Ernie55DSpark.ladderTop(block: $0.config.blockSize) } }

    public var dsparkPinnedWidth: Int? {
        guard attachedHead != nil, let width = Ernie55DSpark.pinnedWidth else { return nil }
        NSLog("[ERNIE-5.5 DSpark] VMLX_DSPARK_WIDTH=\(width) pins every step's width (diagnostic lever)")
        return width
    }

    public func attachDSparkDraft(directory: URL) throws -> Bool {
        if DSparkEnvironment.miniMaxDisabled() || DSparkEnvironment.disabled() { return false }
        dsparkAttachLock.lock()
        defer { dsparkAttachLock.unlock() }
        if dsparkHead != nil { return true }
        let isSidecar = VMLXServerRuntimeSettings.declaresErnie55DSparkDrafter(directory)
        let isInline = !isSidecar && VMLXServerRuntimeSettings.declaresErnie55InlineDSparkDrafter(directory)
        guard isSidecar || isInline else { return false }
        let start = Date()
        let head: Ernie55DSparkHead
        if isSidecar {
            head = try Ernie55DSparkHead.load(
                directory: directory, targetConfig: args,
                embed: { [unowned self] ids in self.model.embedding(ids) },
                rawHead: { [unowned self] h in self.projectLogits(h, applyMultiMax: false) },
                multiMax: { [unowned self] logits in
                    Ernie55Math.multiMax(logits, ranges: self.logitRanges, slopes: self.logitSlopes)
                },
                multiMaxRanges: self.logitRanges,
                multiMaxSlopes: self.logitSlopes)
        } else {
            head = try Ernie55DSparkHead.loadInlineMXFP4(
                directory: directory, targetConfig: args,
                embed: { [unowned self] ids in self.model.embedding(ids) },
                rawHead: { [unowned self] h in self.projectLogits(h, applyMultiMax: false) },
                multiMax: { [unowned self] logits in
                    Ernie55Math.multiMax(logits, ranges: self.logitRanges, slopes: self.logitSlopes)
                },
                multiMaxRanges: self.logitRanges,
                multiMaxSlopes: self.logitSlopes)
        }
        dsparkHead = head
        model.dsparkTailCapacity = head.config.slidingWindow
        model.dsparkTapLayers = head.config.targetLayerIndices
        NSLog(
            "[ERNIE-5.5 DSpark] drafter attached from \(directory.path): \(head.config.layers) layers, block "
                + "\(head.config.blockSize), taps \(head.config.targetLayerIndices), window "
                + "\(head.config.slidingWindow), draft ladder 1...\(Ernie55DSpark.ladderTop(block: head.config.blockSize)) "
                + "(Adaptive), " + String(format: "%.1f s", Date().timeIntervalSince(start)))
        return true
    }

    public func makeDSparkDraftSession() -> (any DSparkDraftSession)? {
        attachedHead.map { head in
            let session = Ernie55DSparkSession(head: head)
            session.model = self
            return session
        }
    }

    func dsparkCommitPrefix(
        cache: [KVCache], checkpoint: SpeculativeCacheListCheckpoint, keep: Int
    ) -> Bool {
        guard let holder = Ernie55DSpark.tapHolder(cache),
            let capture = holder.ernieVerifyCapture, capture.valid,
            keep >= 1, keep <= capture.tokens,
            cache.count == model.layers.count,
            capture.layers.count == model.layers.count
        else { return false }
        let records = (0 ..< model.layers.count).compactMap { capture.layers[$0] }
        guard records.count == model.layers.count else { return false }
        let start = records[0].offset
        guard records.allSatisfy({ $0.offset == start }) else { return false }
        let tail = holder.auxiliaryHiddenTail
        checkpoint.restore()
        guard holder.offset == start else {
            preconditionFailure("ERNIE DSpark commit: checkpoint offset \(holder.offset) != capture \(start)")
        }
        for (i, layer) in model.layers.enumerated() {
            switch records[i] {
            case .hca(let offset, let kv, let x):
                guard let hca = layer.hca, let v4 = cache[i] as? DeepseekV4Cache else { return false }
                hca.commitCapturedPrefix(offset: offset, kv: kv, x: x, keep: keep, cache: v4)
            case .mla(_, let kv, let kPos, let indexKeys):
                guard let mla = layer.mla, let list = cache[i] as? CacheList else { return false }
                mla.commitCapturedPrefix(kv: kv, kPos: kPos, indexKeys: indexKeys, keep: keep, cache: list)
            }
        }
        let end = start + keep
        if var tail, tail.endPosition >= end {
            tail.truncate(toEnd: end)
            holder.auxiliaryHiddenTail = tail.isEmpty ? nil : tail
        } else {
            holder.auxiliaryHiddenTail = tail
        }
        holder.ernieVerifyCapture = nil
        return true
    }

    public func dsparkCacheUnsupportedReason(_ cache: [KVCache]) -> String? {
        guard Ernie55DSpark.tapHolder(cache) != nil else {
            return "the cache has no ERNIE HCA holder layer"
        }
        guard SpeculativeCacheListCheckpoint.supports(cache) else {
            return "a cache layer cannot be checkpointed in place"
        }
        return nil
    }
}
