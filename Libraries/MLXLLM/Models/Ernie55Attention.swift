// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXLMCommon
import MLXNN

struct Ernie55AttentionShape {
    let hidden: Int
    let heads: Int
    let headWidth: Int
    let queryRank: Int
    let ropeWidth: Int
    let epsilon: Float
    let theta: Float
    let virtualHeadRank: Int
}

/// Low-rank postmix materializes I + VUᵀ in FP32 and applies it in
/// the attention value dtype. Keep it independent of the output gate.
final class Ernie55VirtualHeads: Module {
    @ParameterInfo(key: "U") var u: MLXArray
    @ParameterInfo(key: "V") var v: MLXArray
    let heads: Int
    private var _cachedMix: MLXArray?

    init(heads: Int, rank: Int) {
        self.heads = heads
        _u.wrappedValue = MLXArray.zeros([heads, rank])
        _v.wrappedValue = MLXArray.zeros([heads, rank])
        super.init()
    }

    func prepare() {
        let mix = (eye(heads) + matmul(v.asType(.float32), u.asType(.float32).transposed())).asType(.bfloat16)
        eval(mix)
        _cachedMix = mix
    }

    var mixMatrix: MLXArray {
        _cachedMix ?? (eye(heads) + matmul(v.asType(.float32), u.asType(.float32).transposed())).asType(.bfloat16)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        matmul(mixMatrix.asType(x.dtype), x)
    }
}

final class Ernie55HCAAttention: Module {
    let shape: Ernie55AttentionShape
    let window: Int
    let ratio: Int
    let outputGroups: Int
    let outputRank: Int
    let rope: DeepseekV4RoPE

    @ModuleInfo(key: "wq_a") var queryDown: Linear
    @ModuleInfo(key: "wq_b") var queryUp: Linear
    @ModuleInfo(key: "wkv") var keyValue: Linear
    @ModuleInfo(key: "q_norm") var queryNorm: RMSNorm
    @ModuleInfo(key: "kv_norm") var keyValueNorm: RMSNorm
    @ModuleInfo(key: "wo_b") var output: Linear
    @ModuleInfo(key: "o_gate") var gate: Linear
    @ModuleInfo(key: "vha") var virtualHeads: Ernie55VirtualHeads
    @ModuleInfo var compressor: Ernie55HCACompressor
    @ParameterInfo(key: "o_group_weight") var groupedOutput: MLXArray
    @ParameterInfo(key: "o_group_scales") var groupedOutputScales: MLXArray
    @ParameterInfo(key: "attn_sink") var sinks: MLXArray

    init(
        shape: Ernie55AttentionShape, window: Int, ratio: Int, outputGroups: Int,
        outputRank: Int, rope: DeepseekV4RoPE
    ) {
        precondition(shape.heads.isMultiple(of: outputGroups))
        self.shape = shape
        self.window = window
        self.ratio = ratio
        self.outputGroups = outputGroups
        self.outputRank = outputRank
        self.rope = rope
        _queryDown.wrappedValue = Linear(shape.hidden, shape.queryRank, bias: false)
        _queryUp.wrappedValue = Linear(shape.queryRank, shape.heads * shape.headWidth, bias: false)
        _keyValue.wrappedValue = Linear(shape.hidden, shape.headWidth, bias: false)
        _queryNorm.wrappedValue = RMSNorm(dimensions: shape.queryRank, eps: shape.epsilon)
        _keyValueNorm.wrappedValue = RMSNorm(dimensions: shape.headWidth, eps: shape.epsilon)
        _output.wrappedValue = Linear(outputGroups * outputRank, shape.hidden, bias: false)
        _gate.wrappedValue = Linear(shape.queryRank, outputGroups * outputRank, bias: false)
        _virtualHeads.wrappedValue = Ernie55VirtualHeads(heads: shape.heads, rank: shape.virtualHeadRank)
        _compressor.wrappedValue = Ernie55HCACompressor(
            hiddenSize: shape.hidden, headDimension: shape.headWidth, ratio: ratio, epsilon: shape.epsilon)
        let groupIn = shape.heads * shape.headWidth / outputGroups
        _groupedOutput.wrappedValue = MLXArray.zeros([outputGroups, outputRank, groupIn / 4], dtype: .uint32)
        _groupedOutputScales.wrappedValue = MLXArray.zeros([outputGroups, outputRank, groupIn / 32], dtype: .uint8)
        _sinks.wrappedValue = MLXArray.zeros([shape.heads])
        super.init()
    }

    func prepare() {
        virtualHeads.prepare()
    }

    func callAsFunction(
        _ x: MLXArray, cache: DeepseekV4Cache?, layerIndex: Int = 0, capture: Ernie55VerifyCapture? = nil
    ) -> MLXArray {
        let (batch, length) = (x.dim(0), x.dim(1))
        let offset = cache?.offset ?? 0
        let phase = rope.cosSin(offset: offset, length: length)
        let queryRank = queryNorm(DeepseekV41FP8Rows.apply(queryDown, x))
        let unrotated = DeepseekV41FP8Rows.apply(queryUp, queryRank)
            .reshaped(batch, length, shape.heads, shape.headWidth)
        let (q, currentKV) = prepareAttentionRows(
            unrotated, rawKV: DeepseekV41FP8Rows.apply(keyValue, x), phase: phase)
        if let capture {
            capture.record(layer: layerIndex, .hca(offset: offset, kv: currentKV, x: x))
        }
        let attended: MLXArray
        if length == 1 {
            // Single-token decode: every local window row and every completed pooled row is visible.
            var kv = cache?.updateSharedKV(currentKV) ?? currentKV
            let pooled = compressor(x, rope: rope, cache: cache, start: offset)
            if pooled.dim(1) > 0 {
                kv = concatenated([kv, pooled.expandedDimensions(axis: 1)], axis: 2)
            }
            attended = MLXFast.scaledDotProductAttention(
                queries: q, keys: kv, values: kv, scale: 1 / sqrt(Float(shape.headWidth)),
                mask: nil, sinks: sinks.asType(q.dtype))
        } else if length > 256 {
            // Prefill sub-blocking: MLX's head_dim=512 SDPA materializes the full QxK score matrix.
            // Slicing queries into 256-token blocks restricts each block to only its active 128-token
            // sliding window + completed pooled keys (~5x less attention work and peak memory).
            let localKV = cache?.updateSharedKV(currentKV) ?? currentKV
            let pooled = compressor(x, rope: rope, cache: cache, start: offset)
            let priorLocal = localKV.dim(2) - length
            let subSize = 256
            let scale: Float = 1 / sqrt(Float(shape.headWidth))
            let typedSinks = sinks.asType(q.dtype)
            var chunks: [MLXArray] = []
            chunks.reserveCapacity((length + subSize - 1) / subSize)
            var subStart = 0
            while subStart < length {
                let subEnd = min(length, subStart + subSize)
                let subLen = subEnd - subStart
                let qSub = q[0..., 0..., subStart ..< subEnd, 0...]
                let kLo = max(0, priorLocal + subStart - (window - 1))
                let kHi = priorLocal + subEnd
                var kvSub = localKV[0..., 0..., kLo ..< kHi, 0...]
                let qPos = MLXArray(Int32(offset + subStart) ..< Int32(offset + subEnd)).reshaped(subLen, 1)
                let kPos = MLXArray(Int32(offset - priorLocal + kLo) ..< Int32(offset - priorLocal + kHi))
                    .reshaped(1, kHi - kLo)
                var subMask = (kPos .<= qPos) & (kPos .> (qPos - Int32(window)))
                if pooled.dim(1) > 0 {
                    let maxPools = min(pooled.dim(1), (offset + subEnd) / ratio)
                    if maxPools > 0 {
                        let poolMask = Ernie55Math.hcaPoolMask(
                            start: offset + subStart, length: subLen, columns: maxPools, ratio: ratio)
                        subMask = concatenated([subMask, poolMask], axis: -1)
                        kvSub = concatenated(
                            [kvSub, pooled[0..., ..<maxPools, 0...].expandedDimensions(axis: 1)], axis: 2)
                    }
                }
                chunks.append(MLXFast.scaledDotProductAttention(
                    queries: qSub, keys: kvSub, values: kvSub, scale: scale,
                    mask: subMask, sinks: typedSinks))
                subStart = subEnd
            }
            attended = chunks.count == 1 ? chunks[0] : concatenated(chunks, axis: 2)
        } else {
            var mask: MLXArray?
            let localMode = makeAttentionMask(n: length, cache: cache, windowSize: window, returnArray: true)
            var kv = cache?.updateSharedKV(currentKV) ?? currentKV
            if case .array(let local) = localMode {
                mask = local[.ellipsis, (local.dim(-1) - kv.dim(2))...]
            }
            let pooled = compressor(x, rope: rope, cache: cache, start: offset)
            if pooled.dim(1) > 0 {
                let prefixMask = Ernie55Math.hcaPoolMask(
                    start: offset, length: length, columns: pooled.dim(1), ratio: ratio)
                mask = concatenated([
                    mask ?? MLXArray.ones([length, kv.dim(2)], dtype: .bool), prefixMask,
                ], axis: -1)
                kv = concatenated([kv, pooled.expandedDimensions(axis: 1)], axis: 2)
            }
            attended = MLXFast.scaledDotProductAttention(
                queries: q, keys: kv, values: kv, scale: 1 / sqrt(Float(shape.headWidth)),
                mask: mask, sinks: sinks.asType(q.dtype))
        }
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
            let qOut = DeepseekV41FP8Rows.rows(g3, weight: groupedOutput, scales: groupedOutputScales)
                ?? quantizedMM(g3, groupedOutput, scales: groupedOutputScales, biases: nil,
                               transpose: true, groupSize: 32, bits: 8, mode: .mxfp8)
            reducedGroups = qOut.transposed(1, 0, 2).reshaped(batch, length, outputGroups * outputRank)
        } else {
            let g4 = groups.ndim == 4 ? groups : groups.expandedDimensions(axis: 0)
            reducedGroups = matmul(g4, groupedOutput.swappedAxes(-1, -2))
                .transposed(0, 2, 1, 3).reshaped(batch, length, outputGroups * outputRank)
        }
        let gated = reducedGroups * sigmoid(DeepseekV41FP8Rows.apply(gate, queryRank))
        return DeepseekV41FP8Rows.apply(output, gated)
    }

    func commitCapturedPrefix(offset: Int, kv: MLXArray, x: MLXArray, keep: Int, cache: DeepseekV4Cache) {
        _ = cache.updateSharedKV(kv[0..., 0..., ..<keep, 0...])
        _ = compressor(x[0..., ..<keep, 0...], rope: rope, cache: cache, start: offset)
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
        return (DeepseekV4Math.applyPartialRoPE(
            normalized.transposed(0, 2, 1, 3), cos: phase.cos, sin: phase.sin, ropeDim: shape.ropeWidth),
            DeepseekV4Math.applyPartialRoPE(
                keyValueNorm(rawKV).expandedDimensions(axis: 1),
                cos: phase.cos, sin: phase.sin, ropeDim: shape.ropeWidth))
    }
}

/// ERNIE's sparse MLA index keeps its own rotated keys. Selection always
/// includes the recent window and ranks only the older causal positions.
final class Ernie55MLAIndexer: Module {
    let heads: Int
    let dimension: Int
    let topK: Int
    let window: Int
    let ropeWidth: Int
    let theta: Float
    @ModuleInfo(key: "wq_b") var query: Linear
    @ModuleInfo(key: "wk") var key: Linear
    @ModuleInfo(key: "k_norm") var norm: LayerNorm
    @ModuleInfo(key: "weights_proj") var weights: Linear

    init(shape: Ernie55AttentionShape, heads: Int, dimension: Int, topK: Int, window: Int) {
        self.heads = heads
        self.dimension = dimension
        self.topK = topK
        self.window = window
        self.ropeWidth = shape.ropeWidth
        self.theta = shape.theta
        _query.wrappedValue = Linear(shape.queryRank, heads * dimension, bias: false)
        _key.wrappedValue = Linear(shape.hidden, dimension, bias: false)
        _norm.wrappedValue = LayerNorm(dimensions: dimension, eps: 1e-6)
        _weights.wrappedValue = Linear(shape.hidden, heads, bias: false)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, queryRank: MLXArray, mask: MLXArray, cache: KVCache?, offset: Int
    ) -> (mask: MLXArray, indices: MLXArray?, currentKeys: MLXArray) {
        let (batch, length) = (x.dim(0), x.dim(1))
        let projected = norm(key(x)).expandedDimensions(axis: 1)
        let currentKeys = MLXFast.RoPE(
            projected, dimensions: ropeWidth, traditional: false, base: theta, scale: 1, offset: offset)
        var keys = currentKeys
        if let cache {
            (keys, _) = cache.update(
                keys: keys, values: MLXArray.zeros([batch, 1, length, 0], dtype: keys.dtype))
        }
        let columns = keys.dim(2)
        guard columns > topK + window else { return (mask, nil, currentKeys) }
        let projectedQ = DeepseekV41FP8Rows.apply(query, queryRank)
            .reshaped(batch, length, heads, dimension).transposed(0, 2, 1, 3)
        let q = MLXFast.RoPE(
            projectedQ, dimensions: ropeWidth, traditional: false, base: theta, scale: 1, offset: offset)
        let distances = MLXArray(Int32(0)..<Int32(columns)).expandedDimensions(axis: 0)
            - MLXArray(Int32(columns - length)..<Int32(columns)).expandedDimensions(axis: 1)
        let recent = (distances .<= 0) & (distances .> -window)
        let older = distances .<= -window
        let headWeights = weights(x).asType(.float32) * (1 / sqrt(Float(heads * dimension)))
        let dots = maximum(matmul(q, keys.swappedAxes(-1, -2)), 0).asType(.float32)
        let scores = (dots * headWeights.transposed(0, 2, 1).expandedDimensions(axis: -1))
            .sum(axis: 1, keepDims: true)
        let valid = older & mask
        let eligible = `where`(valid, scores, MLXArray(-Float.infinity))
        let selected = argPartition(-eligible, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
        let marked = putAlong(MLXArray.zeros(scores.shape, dtype: .bool), selected, values: MLXArray(true), axis: -1)
        let sparse = ((marked & older) | recent) & mask
        guard length == 1 else { return (sparse, nil, currentKeys) }
        let recentIndices = broadcast(
            MLXArray(Int32(columns - window)..<Int32(columns)).reshaped(1, 1, 1, window),
            to: [batch, 1, 1, window])
        return (sparse, concatenated([selected.asType(.int32), recentIndices], axis: -1), currentKeys)
    }
}

final class Ernie55MLAAttention: Module {
    let shape: Ernie55AttentionShape
    let rotary: RoPE
    let latentWidth: Int
    let valueWidth: Int
    let plainWidth: Int
    @ModuleInfo(key: "q_a_proj") var queryDown: Linear
    @ModuleInfo(key: "q_a_layernorm") var queryNorm: RMSNorm
    @ModuleInfo(key: "q_b_proj") var queryUp: Linear
    @ModuleInfo(key: "kv_a_proj_with_mqa") var keyValue: Linear
    @ModuleInfo(key: "kv_a_layernorm") var keyValueNorm: RMSNorm
    @ModuleInfo(key: "wo_b") var output: Linear
    @ModuleInfo(key: "o_gate") var gate: Linear
    @ModuleInfo(key: "vha") var virtualHeads: Ernie55VirtualHeads
    @ModuleInfo var indexer: Ernie55MLAIndexer
    @ParameterInfo(key: "embed_q_weight") var embedQuery: MLXArray
    @ParameterInfo(key: "unembed_out_weight") var unembedValue: MLXArray
    @ParameterInfo(key: "attn_sink") var sinks: MLXArray
    private var _embedQueryT: MLXArray?
    private var _unembedValueT: MLXArray?

    init(
        shape: Ernie55AttentionShape, latentWidth: Int, valueWidth: Int,
        indexerHeads: Int, indexerDimension: Int, topK: Int, window: Int
    ) {
        self.shape = shape
        self.latentWidth = latentWidth
        self.valueWidth = valueWidth
        self.plainWidth = shape.headWidth - shape.ropeWidth
        self.rotary = RoPE(dimensions: shape.ropeWidth, traditional: true, base: shape.theta)
        _queryDown.wrappedValue = Linear(shape.hidden, shape.queryRank, bias: false)
        _queryNorm.wrappedValue = RMSNorm(dimensions: shape.queryRank, eps: shape.epsilon)
        _queryUp.wrappedValue = Linear(shape.queryRank, shape.heads * shape.headWidth, bias: false)
        _keyValue.wrappedValue = Linear(shape.hidden, latentWidth + shape.ropeWidth, bias: false)
        _keyValueNorm.wrappedValue = RMSNorm(dimensions: latentWidth, eps: shape.epsilon)
        _output.wrappedValue = Linear(shape.heads * valueWidth, shape.hidden, bias: false)
        _gate.wrappedValue = Linear(shape.queryRank, shape.heads * valueWidth, bias: false)
        _virtualHeads.wrappedValue = Ernie55VirtualHeads(heads: shape.heads, rank: shape.virtualHeadRank)
        _indexer.wrappedValue = Ernie55MLAIndexer(
            shape: shape, heads: indexerHeads, dimension: indexerDimension, topK: topK, window: window)
        _embedQuery.wrappedValue = MLXArray.zeros([shape.heads, latentWidth, plainWidth])
        _unembedValue.wrappedValue = MLXArray.zeros([shape.heads, valueWidth, latentWidth])
        _sinks.wrappedValue = MLXArray.zeros([shape.heads])
        super.init()
    }

    func prepare() {
        virtualHeads.prepare()
        let eqT = contiguous(embedQuery.swappedAxes(-1, -2))
        let uvT = contiguous(unembedValue.swappedAxes(-1, -2))
        eval(eqT, uvT)
        _embedQueryT = eqT
        _unembedValueT = uvT
    }

    func callAsFunction(
        _ x: MLXArray, cache: CacheList?, layerIndex: Int = 0, capture: Ernie55VerifyCapture? = nil
    ) -> MLXArray {
        let (batch, length) = (x.dim(0), x.dim(1))
        let latentCache = cache?[0]
        let offset = latentCache?.offset ?? 0
        let latentQuery = queryNorm(DeepseekV41FP8Rows.apply(queryDown, x))
        let q = DeepseekV41FP8Rows.apply(queryUp, latentQuery)
            .reshaped(batch, length, shape.heads, shape.headWidth).transposed(0, 2, 1, 3)
        let qPlain = q[.ellipsis, ..<plainWidth]
        let qPos = rotary(q[.ellipsis, plainWidth...], offset: offset)
        let kvProjection = DeepseekV41FP8Rows.apply(keyValue, x)
        let currentKV = keyValueNorm(kvProjection[.ellipsis, ..<latentWidth]).expandedDimensions(axis: 1)
        let currentKPos = rotary(kvProjection[.ellipsis, latentWidth...].expandedDimensions(axis: 1), offset: offset)
        var kv = currentKV
        var kPos = currentKPos
        if let latentCache { (kv, kPos) = latentCache.update(keys: kv, values: kPos) }
        let causal = createCausalMask(n: length, offset: offset)
        let selection = indexer(x, queryRank: latentQuery, mask: causal, cache: cache?[1], offset: offset)
        if let capture {
            capture.record(
                layer: layerIndex,
                .mla(offset: offset, kv: currentKV, kPos: currentKPos, indexKeys: selection.currentKeys))
        }
        var mask = selection.mask
        if let indices = selection.indices {
            let takeRows = indices[0..., 0..., 0, 0..., .newAxis]
            kv = takeAlong(kv, takeRows, axis: 2)
            kPos = takeAlong(kPos, takeRows, axis: 2)
            mask = takeAlong(mask, indices, axis: -1)
        }
        let scale: Float = 1 / sqrt(Float(shape.headWidth))
        let positions = matmul(qPos * scale, kPos.swappedAxes(-1, -2))
        let bias = `where`(mask, positions, MLXArray(-Float.infinity).asType(positions.dtype))
        let eqT = _embedQueryT ?? embedQuery.swappedAxes(-1, -2)
        let uvT = _unembedValueT ?? unembedValue.swappedAxes(-1, -2)
        // Use absorbed MLA for 1-token decode and 2...16-row DSpark verify (same arithmetic, avoids expanding KV).
        let absorbed = length <= 16
        let queries = absorbed ? matmul(qPlain, eqT) : qPlain
        let keys = absorbed ? kv : matmul(kv, embedQuery)
        let values = absorbed ? kv : matmul(kv, uvT)
        let attention = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: scale, mask: bias,
            sinks: sinks.asType(queries.dtype))
        let out = absorbed ? matmul(attention, uvT) : attention
        let mixed = virtualHeads(out.transposed(0, 2, 1, 3)).reshaped(batch, length, -1)
        return DeepseekV41FP8Rows.apply(output, mixed * sigmoid(DeepseekV41FP8Rows.apply(gate, latentQuery)))
    }

    func commitCapturedPrefix(
        kv: MLXArray, kPos: MLXArray, indexKeys: MLXArray, keep: Int, cache: CacheList
    ) {
        _ = cache[0].update(
            keys: kv[0..., 0..., ..<keep, 0...], values: kPos[0..., 0..., ..<keep, 0...])
        _ = cache[1].update(
            keys: indexKeys[0..., 0..., ..<keep, 0...],
            values: MLXArray.zeros([indexKeys.dim(0), 1, keep, 0], dtype: indexKeys.dtype))
    }
}
