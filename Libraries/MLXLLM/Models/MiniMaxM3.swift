// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MiniMax-M3 / M3.1 text model (`minimax_m3_vl`, `minimax_m3`).
//
// The numerical contract is the vMLX Python runtime (`vmlx_engine/models/minimax_m3/minimax_m3.py`), which follows
// the upstream `MiniMaxM3VL` text stack:
//
// - GQA attention with per-head (1 + w) RMS norms on q / k and partial rotary (the first `rotary_dim` of each head).
// - MiniMax Sparse Attention (MSA) on the layers `sparse_attention_freq` marks: a Lightning indexer scores the cached
//   indexer keys, max-pools them per `index_block_size` keys, and every KV group keeps its own top
//   `index_topk_blocks` blocks (the query's own block always wins). Each group's query heads attend only its blocks.
//   M3 is sparse from layer 3; M3.1 is sparse on every layer.
// - SwiGLU-OAI MLPs: gate clipped above at `swiglu_limit`, up clipped to [-limit, limit],
//   gate * sigmoid(alpha * gate) * (up + 1).
// - MoE on the layers `moe_layer_freq` marks: sigmoid router, `e_score_correction_bias` for the choice only, top-k
//   weights normalized and scaled by `routed_scaling_factor`, plus one shared expert.
//
// The official M3.1 checkpoint loads as stored (MXFP8 linears, NVFP4 routed experts); see MiniMaxM3SourceFormat.swift.
// A JANGHT bundle (config.json `jangt` header) keeps its trunk in MXFP8 / affine per module and its routed experts as
// mapped JANGT trellis (or JANGH codebook) banks; see MiniMaxM3JANGHPreparation.swift.
// Attention over the selected blocks runs on block-sparse Metal kernels (decode rows, prefill chunks) or, as the
// reference, a dense additive mask per KV group; see MiniMaxM3MSA.swift.

/// `(1 + weight)` RMS norm. The sum is formed in the weight's dtype, as in the Python runtime.
final class MiniMaxM3RMSNorm: Module, UnaryLayer {
    let weight: MLXArray
    let eps: Float

    /// `1 + weight`, computed once per loaded weight instead of on every call (the same expression, so the same
    /// values bit for bit). Per decode step M3.1 runs 361 of these norms; building `1 + weight` each time added an
    /// AsType and an Add launch to each (722 of the step's ~4,960 kernel nodes). A module update replaces the
    /// weight's contents in place (same object), so `update(parameters:...)` drops the cache.
    /// `VMLX_M3_NORM_SCALE_CACHE=0` restores the per-call form (A/B lever).
    private var cachedScale: MLXArray?
    private let cacheLock = NSLock()
    static let scaleCacheEnabled = ProcessInfo.processInfo.environment["VMLX_M3_NORM_SCALE_CACHE"] != "0"

    init(dimensions: Int, eps: Float) {
        self.weight = MLXArray.zeros([dimensions])
        self.eps = eps
    }

    var scale: MLXArray {
        guard Self.scaleCacheEnabled else { return 1 + weight }
        return cacheLock.withLock {
            if let cachedScale { return cachedScale }
            let scale = 1 + weight
            eval(scale)
            cachedScale = scale
            return scale
        }
    }

    @discardableResult
    override func update(
        parameters: ModuleParameters, verify: Module.VerifyUpdate, path: [String] = [],
        modulePath: [String] = []
    ) throws -> Self {
        cacheLock.withLock { cachedScale = nil }
        return try super.update(parameters: parameters, verify: verify, path: path, modulePath: modulePath)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: scale, eps: eps)
    }
}

/// SwiGLU-OAI: `gate` is clipped from above, `up` to [-limit, limit]; returns gate * sigmoid(alpha * gate) * (up + 1).
func miniMaxM3SwiGLU(up: MLXArray, gate: MLXArray, alpha: Float, limit: Float) -> MLXArray {
    let gate = minimum(gate, limit)
    let up = clip(up, min: -limit, max: limit)
    return gate * sigmoid(alpha * gate) * (up + 1)
}

/// Lightning indexer: selects, per query and per KV group, the key blocks that group attends.
final class MiniMaxM3Indexer: Module {
    let heads: Int
    let headDim: Int
    let blockSize: Int
    let topKBlocks: Int
    let localBlocks: Int

    @ModuleInfo(key: "index_q_proj") var queryProjection: Linear
    @ModuleInfo(key: "index_k_proj") var keyProjection: Linear
    @ModuleInfo(key: "index_q_norm") var queryNorm: MiniMaxM3RMSNorm
    @ModuleInfo(key: "index_k_norm") var keyNorm: MiniMaxM3RMSNorm

    let rope: RoPE

    init(_ config: MiniMaxM3Configuration) {
        heads = config.sparseAttention.indexHeads
        headDim = config.sparseAttention.indexHeadDim
        blockSize = config.sparseAttention.blockSize
        topKBlocks = config.sparseAttention.topKBlocks
        localBlocks = config.sparseAttention.localBlocks
        _queryProjection.wrappedValue = Linear(config.hiddenSize, heads * headDim, bias: false)
        _keyProjection.wrappedValue = Linear(config.hiddenSize, headDim, bias: false)
        _queryNorm.wrappedValue = MiniMaxM3RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        _keyNorm.wrappedValue = MiniMaxM3RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        rope = RoPE(dimensions: config.rotaryDim, traditional: false, base: config.ropeTheta)
    }

    /// Index queries `[B, heads, Sq, d]` and keys `[B, 1, Sq, d]` of `x` at positions `offset...` (norm + RoPE).
    /// `linear` replaces the projection call (DSpark verify: a decode step's arithmetic per row).
    func project(
        _ x: MLXArray, offset: Int, linear: MiniMaxM3LinearCall = MiniMaxM3RowExact.plain
    ) -> (queries: MLXArray, keys: MLXArray) {
        let (B, queryLength) = (x.dim(0), x.dim(1))
        let queries = queryNorm(linear(queryProjection, x).reshaped(B, queryLength, heads, headDim))
            .transposed(0, 2, 1, 3)
        let keys = keyNorm(linear(keyProjection, x).reshaped(B, queryLength, 1, headDim)).transposed(0, 2, 1, 3)
        return (rope(queries, offset: offset), rope(keys, offset: offset))
    }

    /// The block selection for index queries at `offset...` over the index keys `[B, 1, Sk, d]` (nil while every
    /// block is visible: then only causality applies). See `MiniMaxM3MSA.select`.
    func select(queries: MLXArray, keys: MLXArray, offset: Int) -> MiniMaxM3MSASelection? {
        MiniMaxM3MSA.select(
            queries: queries, keys: keys, offset: offset, blockSize: blockSize, topK: topKBlocks,
            localBlocks: localBlocks)
    }

    /// Appends this step's indexer keys to `cache` (which must already hold this step's K/V) and selects the blocks
    /// every query row of every KV group attends.
    func callAsFunction(_ x: MLXArray, cache: MiniMaxM3SparseCache?, offset: Int) -> MiniMaxM3MSASelection? {
        let (queries, keys) = project(x, offset: offset)
        return select(queries: queries, keys: cache?.updateIndexKeys(keys) ?? keys, offset: offset)
    }

    /// The selection as an additive key bias `[B, G, Sq, Sk]` in float32: 0 where query `i` of group `g` attends key
    /// `j`, -inf elsewhere, causality included. nil while every block is visible.
    func keyBias(_ x: MLXArray, cache: MiniMaxM3SparseCache?, offset: Int) -> MLXArray? {
        self(x, cache: cache, offset: offset)?.denseBias()
    }
}

final class MiniMaxM3Attention: Module {
    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear
    @ModuleInfo(key: "q_norm") var qNorm: MiniMaxM3RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: MiniMaxM3RMSNorm
    @ModuleInfo(key: "indexer") var indexer: MiniMaxM3Indexer?

    let rope: RoPE
    /// The RoPE geometry of `rope` (and of the indexer's; both non-traditional, unscaled): the fused-decode kernels
    /// rebuild the angles from it.
    let rotaryDim: Int
    let ropeTheta: Float

    init(_ config: MiniMaxM3Configuration, layer: Int) {
        heads = config.attentionHeads
        kvHeads = config.kvHeads
        headDim = config.headDim
        scale = pow(Float(headDim), -0.5)
        rotaryDim = config.rotaryDim
        ropeTheta = config.ropeTheta
        _wq.wrappedValue = Linear(config.hiddenSize, heads * headDim, bias: false)
        _wk.wrappedValue = Linear(config.hiddenSize, kvHeads * headDim, bias: false)
        _wv.wrappedValue = Linear(config.hiddenSize, kvHeads * headDim, bias: false)
        _wo.wrappedValue = Linear(heads * headDim, config.hiddenSize, bias: false)
        _qNorm.wrappedValue = MiniMaxM3RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        _kNorm.wrappedValue = MiniMaxM3RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        _indexer.wrappedValue = config.isSparse(layer) ? MiniMaxM3Indexer(config) : nil
        rope = RoPE(dimensions: config.rotaryDim, traditional: false, base: config.ropeTheta)
    }

    /// q `[B, H, L, D]`, k / v `[B, KV, L, D]` of `x` at positions `offset...` (q / k norms + partial RoPE).
    /// `linear` replaces the projection call (DSpark verify: a decode step's arithmetic per row).
    func project(
        _ x: MLXArray, offset: Int, linear: MiniMaxM3LinearCall = MiniMaxM3RowExact.plain
    ) -> (queries: MLXArray, keys: MLXArray, values: MLXArray) {
        let (B, L) = (x.dim(0), x.dim(1))
        let queries = qNorm(linear(wq, x).reshaped(B, L, heads, headDim)).transposed(0, 2, 1, 3)
        let keys = kNorm(linear(wk, x).reshaped(B, L, kvHeads, headDim)).transposed(0, 2, 1, 3)
        let values = linear(wv, x).reshaped(B, L, kvHeads, headDim).transposed(0, 2, 1, 3)
        return (rope(queries, offset: offset), rope(keys, offset: offset), values)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))
        let offset = cache?.offset ?? 0
        var (queries, keys, values) = project(x, offset: offset)

        guard let indexer else {
            let output = attentionWithCacheUpdate(
                queries: queries, keys: keys, values: values, cache: cache, scale: scale, mask: mask)
            return wo(output.transposed(0, 2, 1, 3).reshaped(B, L, -1))
        }

        // K/V are appended before the indexer runs, so the indexer's key history has the same length as K/V.
        let sparseCache: MiniMaxM3SparseCache?
        if let cache {
            guard let typed = cache as? MiniMaxM3SparseCache else {
                preconditionFailure(
                    "MiniMax-M3 sparse attention layer needs a MiniMaxM3SparseCache (it keeps the indexer keys); "
                        + "got \(type(of: cache)). Create caches with MiniMaxM3Model.newCache(parameters:).")
            }
            sparseCache = typed
            (keys, values) = typed.update(keys: keys, values: values)
        } else {
            sparseCache = nil
        }

        let selection = indexer(x, cache: sparseCache, offset: offset)
        let output = attend(queries, keys, values, selection: selection, mask: mask)
        return wo(output.transposed(0, 2, 1, 3).reshaped(B, L, -1))
    }

    /// Attention output `[B, H, L, D]` (before o_proj) of the queries over keys / values, restricted to the MSA
    /// selection when there is one. `MiniMaxM3MSA.choosePath` picks the path (`VMLX_MINIMAX_MSA` forces one); the
    /// dense mask path is the reference.
    func attend(
        _ queries: MLXArray, _ keys: MLXArray, _ values: MLXArray, selection: MiniMaxM3MSASelection?,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray {
        guard let selection else {
            return MLXFast.scaledDotProductAttention(
                queries: queries, keys: keys, values: values, scale: scale, mask: mask)
        }
        switch MiniMaxM3MSA.choosePath(selection, queries: queries, keys: keys, mask: mask, kvHeads: kvHeads) {
        case .decode:
            return MiniMaxM3MSA.decodeKernelAttention(
                queries: queries, keys: keys, values: values, selection: selection, scale: scale)
        case .gather:
            return MiniMaxM3MSA.gatheredAttention(
                queries: queries, keys: keys, values: values, selection: selection, scale: scale)
        case .kernel:
            return MiniMaxM3MSA.kernelAttention(
                queries: queries, keys: keys, values: values, selection: selection, scale: scale)
        case .dense:
            // 0 and -inf are exact in every float dtype, so the cast to the activation dtype is lossless.
            var bias = selection.denseBias().asType(queries.dtype)
            if case .array(let extra) = mask {
                bias =
                    extra.dtype == .bool
                    ? MLX.where(extra, bias, -Float.infinity)
                    : bias + extra.asType(queries.dtype)
            }
            return groupedAttention(queries, keys, values, bias: bias)
        }
    }

    /// Attention where each KV group has its own key mask. Query heads are contiguous per KV head, so folding the
    /// groups into the batch axis lets one `[B * G, 1, L, Sk]` mask cover each group's query heads.
    private func groupedAttention(
        _ queries: MLXArray, _ keys: MLXArray, _ values: MLXArray, bias: MLXArray
    ) -> MLXArray {
        let (B, H, L, D) = (queries.dim(0), queries.dim(1), queries.dim(2), queries.dim(3))
        let groups = bias.dim(1)
        let keyLength = keys.dim(2)
        guard groups == kvHeads else {
            return MLXFast.scaledDotProductAttention(
                queries: queries, keys: keys, values: values, scale: scale,
                mask: .array(repeated(bias, count: H / groups, axis: 1)))
        }
        let output = MLXFast.scaledDotProductAttention(
            queries: queries.reshaped(B * groups, H / groups, L, D),
            keys: keys.reshaped(B * groups, 1, keyLength, D),
            values: values.reshaped(B * groups, 1, keyLength, values.dim(3)),
            scale: scale,
            mask: .array(bias.reshaped(B * groups, 1, L, keyLength)))
        return output.reshaped(B, H, L, -1)
    }
}

/// Dense MLP and the shared expert: `down(swiglu_oai(up(x), gate(x)))`.
final class MiniMaxM3MLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    let alpha: Float
    let limit: Float

    init(dimensions: Int, hiddenDimensions: Int, alpha: Float, limit: Float) {
        _gateProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        _upProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        _downProj.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        self.alpha = alpha
        self.limit = limit
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(miniMaxM3SwiGLU(up: upProj(x), gate: gateProj(x), alpha: alpha, limit: limit))
    }
}

/// Routed experts. Each projection is a `SwitchLinear` (float), a `QuantizedSwitchLinear` (installed by the generic
/// loader for quantized bundles), or a `MiniMaxM3NVFP4SwitchLinear` (the official checkpoint's NVFP4 experts).
final class MiniMaxM3SwitchGLU: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Module
    @ModuleInfo(key: "up_proj") var upProj: Module
    @ModuleInfo(key: "down_proj") var downProj: Module

    let alpha: Float
    let limit: Float

    init(dimensions: Int, hiddenDimensions: Int, experts: Int, alpha: Float, limit: Float) {
        _gateProj.wrappedValue = SwitchLinear(
            inputDims: dimensions, outputDims: hiddenDimensions, numExperts: experts, bias: false)
        _upProj.wrappedValue = SwitchLinear(
            inputDims: dimensions, outputDims: hiddenDimensions, numExperts: experts, bias: false)
        _downProj.wrappedValue = SwitchLinear(
            inputDims: hiddenDimensions, outputDims: dimensions, numExperts: experts, bias: false)
        self.alpha = alpha
        self.limit = limit
    }

    /// `x` `[..., D]`, `indices` `[..., k]` -> `[..., k, D]`.
    func callAsFunction(_ x: MLXArray, _ indices: MLXArray) -> MLXArray {
        var x = expandedDimensions(x, axes: [-2, -3])
        let sorted = indices.size >= 64
        var routed = indices
        var inverseOrder = MLXArray()
        if sorted {
            (x, routed, inverseOrder) = gatherSort(x: x, indices: indices)
        }
        let up = Self.project(upProj, x, routed, sorted: sorted)
        let gate = Self.project(gateProj, x, routed, sorted: sorted)
        var y = Self.project(
            downProj, miniMaxM3SwiGLU(up: up, gate: gate, alpha: alpha, limit: limit), routed,
            sorted: sorted)
        if sorted {
            y = scatterUnsort(x: y, invOrder: inverseOrder, shape: indices.shape)
        }
        return y.squeezed(axis: -2)
    }

    private static func project(
        _ projection: Module, _ x: MLXArray, _ indices: MLXArray, sorted: Bool
    ) -> MLXArray {
        if let projection = projection as? MiniMaxM3NVFP4SwitchLinear {
            return projection(x, indices, sortedIndices: sorted)
        }
        if let projection = projection as? SwitchLinear {
            return projection(x, indices, sortedIndices: sorted)
        }
        preconditionFailure("MiniMax-M3 routed expert projection has unexpected type \(type(of: projection))")
    }
}

final class MiniMaxM3SparseMoeBlock: Module, UnaryLayer {
    let topK: Int
    let routedScalingFactor: Float
    let normTopKProb: Bool

    @ModuleInfo(key: "gate") var gate: Linear
    @ParameterInfo(key: "e_score_correction_bias") var eScoreCorrectionBias: MLXArray
    /// A `MiniMaxM3SwitchGLU` (float, affine-quantized or official NVFP4 stacks) or, in a JANGHT bundle, a mapped
    /// bank layer (`WeightedRoutedExpertLayer`) that gathers, activates and router-weights the routed experts in one
    /// call with the model's SwiGLU-OAI.
    @ModuleInfo(key: "switch_mlp") var experts: Module
    @ModuleInfo(key: "shared_experts") var sharedExperts: MiniMaxM3MLP
    /// The fused MoE kernels' view of this block (`MiniMaxM3FusedMoE.layer`), built on first use. Stored on the block so
    /// it dies with the model: a process-wide table keyed by block identity kept every layer's routed banks alive after
    /// the model was released (a reload then held two models). Leading underscore: not a parameter.
    var _fusedMoELayer: MiniMaxM3FusedMoE.LayerBox?

    /// The stacked routed experts; nil when the layer runs mapped JANGHT banks.
    var switchMLP: MiniMaxM3SwitchGLU? { experts as? MiniMaxM3SwitchGLU }

    init(_ config: MiniMaxM3Configuration, routed: (any WeightedRoutedExpertLayer)? = nil) {
        topK = config.numExpertsPerTok
        routedScalingFactor = config.routedScalingFactor
        normTopKProb = config.normTopKProb
        _gate.wrappedValue = Linear(config.hiddenSize, config.numLocalExperts, bias: false)
        _eScoreCorrectionBias.wrappedValue = MLXArray.zeros([config.numLocalExperts])
        _experts.wrappedValue = routed ?? MiniMaxM3SwitchGLU(
            dimensions: config.hiddenSize, hiddenDimensions: config.intermediateSize,
            experts: config.numLocalExperts, alpha: config.swigluAlpha, limit: config.swigluLimit)
        _sharedExperts.wrappedValue = MiniMaxM3MLP(
            dimensions: config.hiddenSize, hiddenDimensions: config.sharedIntermediateSize,
            alpha: config.swigluAlpha, limit: config.swigluLimit)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (indices, scores) = route(x)
        let routed: MLXArray
        if let switchMLP {
            routed = (switchMLP(x, indices) * scores[.ellipsis, .newAxis]).sum(axis: -2)
        } else {
            // Mapped banks: the router-weighted sum in F32, returned in x's dtype (Python `JTMixedSwitchGLU.routed`).
            routed = (experts as! any WeightedRoutedExpertLayer).callRouted(x, indices: indices, scores: scores)
        }
        return combine(routed: routed, x: x)
    }

    /// Sigmoid router; the correction bias steers the choice but not the weights. `[..., topK]` expert indices and
    /// their normalized, scaled weights (float32).
    func route(_ x: MLXArray) -> (indices: MLXArray, scores: MLXArray) {
        let routerScores = sigmoid(gate(x).asType(.float32))
        let choice = routerScores + eScoreCorrectionBias
        let indices = argPartition(-choice, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
        var scores = takeAlong(routerScores, indices, axis: -1)
        if topK > 1 && normTopKProb {
            scores = scores / scores.sum(axis: -1, keepDims: true)
        }
        return (indices, scores * routedScalingFactor)
    }

    /// The routed rows plus the shared expert of `x`, in `x`'s dtype.
    func combine(routed: MLXArray, x: MLXArray) -> MLXArray {
        (routed + sharedExperts(x)).asType(x.dtype)
    }
}

final class MiniMaxM3DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: MiniMaxM3Attention
    @ModuleInfo(key: "mlp") var mlp: UnaryLayer
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: MiniMaxM3RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: MiniMaxM3RMSNorm
    /// This layer's index in the stack (the fused-decode seam passes it to the kernels).
    let index: Int

    init(_ config: MiniMaxM3Configuration, layer: Int, routed: (any WeightedRoutedExpertLayer)? = nil) {
        index = layer
        _selfAttn.wrappedValue = MiniMaxM3Attention(config, layer: layer)
        _mlp.wrappedValue =
            config.isMoE(layer)
            ? MiniMaxM3SparseMoeBlock(config, routed: routed)
            : MiniMaxM3MLP(
                dimensions: config.hiddenSize, hiddenDimensions: config.denseIntermediateSize,
                alpha: config.swigluAlpha, limit: config.swigluLimit)
        _inputLayerNorm.wrappedValue = MiniMaxM3RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = MiniMaxM3RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        mlpHalf(attentionHalf(x, mask: mask, cache: cache))
    }

    /// `x + attention(input_layernorm(x))`, appending this call's rows to `cache`. Fused decode
    /// (`MiniMaxM3FusedDecode`, off by default) takes 1...16-row single-sequence calls, block by block.
    func attentionHalf(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        MiniMaxM3FusedDecode.active(rows: x.dim(1), batch: x.dim(0))?
            .attentionBlock(layer: index, module: self, x: x, cache: cache, verify: false)
            ?? x + selfAttn(inputLayerNorm(x), mask: mask, cache: cache)
    }

    /// `h + mlp(post_attention_layernorm(h))` (the fused MLP block for 1...16-row calls when it is on).
    func mlpHalf(_ h: MLXArray) -> MLXArray {
        MiniMaxM3FusedDecode.active(rows: h.dim(1), batch: h.dim(0))?
            .mlpBlock(layer: index, module: self, h: h, verify: false)
            ?? h + mlp(postAttentionLayerNorm(h))
    }

    /// The MLP half of every piece of a layer-major prefill group (`MiniMaxM3ModelInner.prefillPieces`): each piece's
    /// rows are exactly `mlpHalf` of that piece, except that the JANGT routed experts of the pieces whose routed call
    /// takes the expert-sorted prefill kernel (`JANGTRoutedExpertLayer.usesDecodeKernels` false: >= 512 (row, slot)
    /// pairs, NAX) run in ONE call over all of those pieces' rows. That kernel's output rows depend only on their own
    /// row and its routing (rows sorted by expert, each row's tile products and its fp32 routing sum are per row), so the
    /// rows equal the per-piece calls bit for bit while every expert's weights are decoded once per group instead of once
    /// per piece. The router, the shared expert and the combine stay per piece: their matmuls' rows depend on the row
    /// count (split-K / tile choice), which is why the pieces themselves exist. `groupRouted` false = `mlpHalf` per piece.
    /// Tests: told (layer, pieces) for every grouped routed-expert call.
    @TaskLocal static var groupedRoutedObserver: (@Sendable (Int, Int) -> Void)?

    func mlpHalves(_ hs: [MLXArray], groupRouted: Bool, sortedAll: Bool = false) -> [MLXArray] {
        guard groupRouted || sortedAll, let block = mlp as? MiniMaxM3SparseMoeBlock,
            let bank = block.experts as? JANGTRoutedExpertLayer
        else { return hs.map(mlpHalf) }
        // `sortedAll` (`MiniMaxM3Model.prefillSortedAll`): every non-fused piece's routed experts on the sorted tiles,
        // whatever its row count; otherwise only the pieces whose own call takes them (>= 512 (row, slot) pairs).
        let grouped = hs.indices.filter { i in
            let h = hs[i]
            guard MiniMaxM3FusedDecode.active(rows: h.dim(1), batch: h.dim(0)) == nil else { return false }
            return sortedAll
                ? h.dim(0) * h.dim(1) * block.topK >= MiniMaxM3Model.prefillSortedMinPairs
                    && JANGTRoutedExpertLayer.sortedPrefillAvailable(dtype: h.dtype)
                : !JANGTRoutedExpertLayer.usesDecodeKernels(pairs: h.dim(0) * h.dim(1) * block.topK, dtype: h.dtype)
        }
        guard sortedAll ? !grouped.isEmpty : grouped.count > 1 else { return hs.map(mlpHalf) }
        func routedCall(_ x: MLXArray, _ indices: MLXArray, _ scores: MLXArray) -> MLXArray {
            sortedAll
                ? bank.callRoutedSorted(x, indices: indices, scores: scores)
                : bank.callRouted(x, indices: indices, scores: scores)
        }
        var out = [MLXArray?](repeating: nil, count: hs.count)
        var normed: [MLXArray] = []
        var routes: [(indices: MLXArray, scores: MLXArray)] = []
        for i in grouped {
            let xn = postAttentionLayerNorm(hs[i])
            normed.append(xn)
            routes.append(block.route(xn))
        }
        if groupRouted && grouped.count > 1 {
            Self.groupedRoutedObserver?(index, grouped.count)
            let routed = routedCall(
                concatenated(normed, axis: 1), concatenated(routes.map(\.indices), axis: 1),
                concatenated(routes.map(\.scores), axis: 1))
            var at = 0
            for (j, i) in grouped.enumerated() {
                let n = hs[i].dim(1)
                out[i] = hs[i] + block.combine(routed: routed[0..., at ..< (at + n), 0...], x: normed[j])
                at += n
            }
        } else {
            for (j, i) in grouped.enumerated() {
                out[i] = hs[i] + block.combine(
                    routed: routedCall(normed[j], routes[j].indices, routes[j].scores), x: normed[j])
            }
        }
        for i in hs.indices where out[i] == nil { out[i] = mlpHalf(hs[i]) }
        return out.map { $0! }
    }
}

public final class MiniMaxM3ModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    let layers: [MiniMaxM3DecoderLayer]
    @ModuleInfo(key: "norm") var norm: MiniMaxM3RMSNorm

    /// DSpark: the decoder layers whose output rows (residual stream, before the final norm) the attached drafter
    /// reads, in the drafter's order; empty while no drafter is attached. Every forward of a single sequence then
    /// records the last `dsparkTailCapacity` rows of them on the cache list's tap holder (`MiniMaxM3DSpark`).
    var dsparkTapLayers: [Int] = []
    var dsparkTailCapacity = 0

    init(_ config: MiniMaxM3Configuration, routed: [Int: any WeightedRoutedExpertLayer] = [:]) {
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        layers = (0 ..< config.hiddenLayers).map {
            MiniMaxM3DecoderLayer(config, layer: $0, routed: routed[$0])
        }
        _norm.wrappedValue = MiniMaxM3RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    /// `recordTaps` false: a prefill piece that ends before the drafter's window records no tap rows (they would leave
    /// the tail before the drafter reads it; `MiniMaxM3Model.prepare`). `inputEmbeddings`: the rows to run instead of
    /// `embedTokens(inputs)` (an image prefill piece, `MiniMaxM3ImageSplice`).
    func callAsFunction(
        _ inputs: MLXArray, cache: [KVCache]?, recordTaps tapsWanted: Bool = true, inputEmbeddings: MLXArray? = nil
    ) -> MLXArray {
        let holder = MiniMaxM3DSpark.tapHolder(cache)
        if let holder, let cache, MiniMaxM3DSpark.verifies(inputs, holder: holder) {
            return verifyForward(inputs, cache: cache, holder: holder)
        }
        let start = cache?.first?.offset ?? 0
        let recording = tapsWanted && !dsparkTapLayers.isEmpty && holder != nil && inputs.dim(0) == 1
        var taps: [Int: MLXArray] = [:]
        var h = inputEmbeddings ?? embedTokens(inputs)
        // The Python runtime's mask (mlx_lm create_attention_mask): symbolic causal for every multi-token call, at any
        // cache offset (MLX SDPA aligns a causal mask bottom-right, so the key length carries the offset); none for
        // one token. Dense layers use it as is; sparse layers fold causality into their block selection. The cache's
        // own mask is an explicit array once its offset is nonzero, and MSA routes any array mask (batch padding) to
        // the dense mask path: every prefill chunk after the first, and every multi-row verify call, would skip the
        // block-sparse kernels.
        let mask: MLXFast.ScaledDotProductAttentionMaskMode = h.dim(1) > 1 ? .causal : .none
        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
            if recording && dsparkTapLayers.contains(i) { taps[i] = h }
        }
        if recording, let holder { recordTaps(taps, start: start, holder: holder) }
        return norm(h)
    }

    /// `prefillPieces` with the row-wise work of the group's pieces in ONE call per op (`VMLX_M3_PREFILL_BATCH_ROWS`):
    /// the RMS norms, the q / k / index RoPE, the K / V / index-key appends (one append of all the group's rows; each
    /// piece's attention reads the lane up to its own end, the arrays its own append returns), the router's sigmoid /
    /// top-k / normalization, the SwiGLU of the shared expert and the dense MLP, the combine and the residual adds. Every
    /// matrix product whose rows depend on the row count (MLX picks split-K / tiles / the vector kernel by M: q / k / v /
    /// o, the indexer projections, the router, the shared expert and dense MLP projections) still runs per piece on the
    /// piece's own rows, as do the block selection and the attention (their key length and path are the piece's). The
    /// routed experts follow `mlpHalves`' rule (one call per group on the sorted tiles). Rows, lanes and logits equal
    /// piece by piece; a piece the fused decode path would take, or a layer without the sparse cache, falls back to
    /// `prefillPieces` (the whole group).
    func prefillPiecesBatched(
        _ pieces: [MLXArray], cache: [KVCache], recordTaps: [Bool], groupRouted: Bool, sortedAll: Bool,
        embeddings: [MLXArray?]? = nil
    ) {
        let rows = pieces.map { $0.dim(1) }
        guard pieces.count > 1, pieces.allSatisfy({ $0.ndim == 2 && $0.dim(0) == 1 }),
            cache.count == layers.count,
            rows.allSatisfy({ MiniMaxM3FusedDecode.active(rows: $0, batch: 1) == nil })
        else {
            prefillPieces(
                pieces, cache: cache, recordTaps: recordTaps, groupRouted: groupRouted, sortedAll: sortedAll,
                embeddings: embeddings)
            return
        }
        let holder = MiniMaxM3DSpark.tapHolder(cache)
        let recording = !dsparkTapLayers.isEmpty && holder != nil
        let start0 = cache[0].offset
        var bounds: [Range<Int>] = []  // rows of each piece inside the group
        var at = 0
        for n in rows {
            bounds.append(at ..< (at + n))
            at += n
        }
        let total = at
        func split(_ x: MLXArray) -> [MLXArray] { bounds.map { x[0..., $0, 0...] } }
        func perPiece(_ xs: [MLXArray], _ f: (MLXArray) -> MLXArray) -> MLXArray {
            concatenated(xs.map(f), axis: 1)
        }
        // A projection whose rows are invariant from `T` rows on, on this device (`MiniMaxM3RowInvariance`, probed
        // once per shape): the pieces of at least `T` rows in one call, the others per piece.
        let maxRows = max(total, MiniMaxM3RowInvariance.probeRows)
        func project(_ linear: Linear, _ xs: [MLXArray]) -> MLXArray {
            guard let t = MiniMaxM3RowInvariance.threshold(linear, maxRows: maxRows, dtype: xs[0].dtype) else {
                return perPiece(xs) { linear($0) }
            }
            let big = xs.indices.filter { xs[$0].dim(1) >= t }
            guard big.count > 1 else { return perPiece(xs) { linear($0) } }
            var out = [MLXArray?](repeating: nil, count: xs.count)
            let y = linear(concatenated(big.map { xs[$0] }, axis: 1))
            var at = 0
            for i in big {
                let n = xs[i].dim(1)
                out[i] = y[0..., at ..< (at + n), 0...]
                at += n
            }
            for i in xs.indices where out[i] == nil { out[i] = linear(xs[i]) }
            return concatenated(out.map { $0! }, axis: 1)
        }
        // MLP half: post-norm, router (matmul per piece or admitted), routed experts (`routedRows`), shared expert /
        // dense MLP projections (per piece or admitted), SwiGLU, combine and residual on all rows.
        func batchedMLP(_ layer: MiniMaxM3DecoderLayer, _ h: MLXArray, _ li: Int) -> MLXArray {
            let xn = layer.postAttentionLayerNorm(h)
            let xns = split(xn)
            if let block = layer.mlp as? MiniMaxM3SparseMoeBlock {
                // `MiniMaxM3SparseMoeBlock.route`, the router's matmul per piece, the rest on all rows.
                let routerScores = sigmoid(project(block.gate, xns).asType(.float32))
                let choice = routerScores + block.eScoreCorrectionBias
                let indices = argPartition(-choice, kth: block.topK - 1, axis: -1)[.ellipsis, ..<block.topK]
                var scores = takeAlong(routerScores, indices, axis: -1)
                if block.topK > 1 && block.normTopKProb {
                    scores = scores / scores.sum(axis: -1, keepDims: true)
                }
                scores = scores * block.routedScalingFactor
                let routed = routedRows(
                    block: block, xns: xns, indices: split(indices), scores: split(scores), groupRouted: groupRouted,
                    sortedAll: sortedAll, layer: li)
                let shared = block.sharedExperts
                let activated = miniMaxM3SwiGLU(
                    up: project(shared.upProj, xns), gate: project(shared.gateProj, xns),
                    alpha: shared.alpha, limit: shared.limit)
                let sharedOut = project(shared.downProj, split(activated))
                return h + (routed + sharedOut).asType(xn.dtype)
            } else if let dense = layer.mlp as? MiniMaxM3MLP {
                let activated = miniMaxM3SwiGLU(
                    up: project(dense.upProj, xns), gate: project(dense.gateProj, xns),
                    alpha: dense.alpha, limit: dense.limit)
                return h + project(dense.downProj, split(activated))
            } else {
                return h + perPiece(xns) { layer.mlp($0) }
            }
        }
        var h = concatenated(pieces.indices.map { embeddings?[$0] ?? embedTokens(pieces[$0]) }, axis: 1)
        var taps = [[Int: MLXArray]](repeating: [:], count: pieces.count)
        for (li, layer) in layers.enumerated() {
            let attn = layer.selfAttn
            guard let indexer = attn.indexer, let c = cache[li] as? MiniMaxM3SparseCache else {
                // A dense-attention layer (M3: layers 0-2): its attention half per piece, the MLP half below batched.
                let masks: [MLXFast.ScaledDotProductAttentionMaskMode] = rows.map { $0 > 1 ? .causal : .none }
                h = concatenated(
                    zip(split(h), masks).map { layer.attentionHalf($0.0, mask: $0.1, cache: cache[li]) }, axis: 1)
                h = batchedMLP(layer, h, li)
                if recording && dsparkTapLayers.contains(li) {
                    let hs = split(h)
                    for i in pieces.indices where recordTaps[i] { taps[i][li] = hs[i] }
                }
                continue
            }
            let offset = c.offset
            // Attention half: `MiniMaxM3Attention.callAsFunction`, the row-wise ops on all rows at once.
            let x = layer.inputLayerNorm(h)
            let xs = split(x)
            let queries = attn.rope(
                attn.qNorm(project(attn.wq, xs).reshaped(1, total, attn.heads, attn.headDim))
                    .transposed(0, 2, 1, 3), offset: offset)
            let keysNew = attn.rope(
                attn.kNorm(project(attn.wk, xs).reshaped(1, total, attn.kvHeads, attn.headDim))
                    .transposed(0, 2, 1, 3), offset: offset)
            let valuesNew = project(attn.wv, xs).reshaped(1, total, attn.kvHeads, attn.headDim)
                .transposed(0, 2, 1, 3)
            let (keysAll, valuesAll) = c.update(keys: keysNew, values: valuesNew)
            let indexQueries = indexer.rope(
                indexer.queryNorm(project(indexer.queryProjection, xs)
                    .reshaped(1, total, indexer.heads, indexer.headDim)).transposed(0, 2, 1, 3), offset: offset)
            let indexKeysNew = indexer.rope(
                indexer.keyNorm(project(indexer.keyProjection, xs).reshaped(1, total, 1, indexer.headDim))
                    .transposed(0, 2, 1, 3), offset: offset)
            let indexKeysAll = c.updateIndexKeys(indexKeysNew)
            var attended: [MLXArray] = []
            for (i, b) in bounds.enumerated() {
                let end = offset + b.upperBound
                let selection = indexer.select(
                    queries: indexQueries[0..., 0..., b, 0...], keys: indexKeysAll[.ellipsis, ..<end, 0...],
                    offset: offset + b.lowerBound)
                let output = attn.attend(
                    queries[0..., 0..., b, 0...], keysAll[.ellipsis, ..<end, 0...],
                    valuesAll[.ellipsis, ..<end, 0...], selection: selection,
                    mask: rows[i] > 1 ? .causal : .none)
                attended.append(output.transposed(0, 2, 1, 3).reshaped(1, rows[i], -1))
            }
            h = h + project(attn.wo, attended)
            h = batchedMLP(layer, h, li)
            if recording && dsparkTapLayers.contains(li) {
                let hs = split(h)
                for i in pieces.indices where recordTaps[i] { taps[i][li] = hs[i] }
            }
        }
        if recording, let holder {
            var position = start0
            for i in pieces.indices {
                if recordTaps[i] { self.recordTaps(taps[i], start: position, holder: holder) }
                position += rows[i]
            }
        }
    }

    /// The routed rows `[1, total, D]` of a batched group (`prefillPiecesBatched`): `mlpHalves`' rule per piece (the
    /// sorted tiles in one call per group, the stock `callRouted` for the rest).
    private func routedRows(
        block: MiniMaxM3SparseMoeBlock, xns: [MLXArray], indices: [MLXArray], scores: [MLXArray], groupRouted: Bool,
        sortedAll: Bool, layer: Int
    ) -> MLXArray {
        guard let bank = block.experts as? JANGTRoutedExpertLayer else {
            return concatenated(
                xns.indices.map { i in
                    if let switchMLP = block.switchMLP {
                        return (switchMLP(xns[i], indices[i]) * scores[i][.ellipsis, .newAxis]).sum(axis: -2)
                    }
                    return (block.experts as! any WeightedRoutedExpertLayer).callRouted(
                        xns[i], indices: indices[i], scores: scores[i])
                }, axis: 1)
        }
        let sorted = xns.indices.filter { i in
            sortedAll
                ? xns[i].dim(1) * block.topK >= MiniMaxM3Model.prefillSortedMinPairs
                    && JANGTRoutedExpertLayer.sortedPrefillAvailable(dtype: xns[i].dtype)
                : !JANGTRoutedExpertLayer.usesDecodeKernels(pairs: xns[i].dim(1) * block.topK, dtype: xns[i].dtype)
        }
        var out = [MLXArray?](repeating: nil, count: xns.count)
        let together = groupRouted && (sortedAll ? sorted.count > 1 : sorted.count > 1)
        if together {
            MiniMaxM3DecoderLayer.groupedRoutedObserver?(layer, sorted.count)
            let x = concatenated(sorted.map { xns[$0] }, axis: 1)
            let ind = concatenated(sorted.map { indices[$0] }, axis: 1)
            let sc = concatenated(sorted.map { scores[$0] }, axis: 1)
            let y = sortedAll
                ? bank.callRoutedSorted(x, indices: ind, scores: sc) : bank.callRouted(x, indices: ind, scores: sc)
            var at = 0
            for i in sorted {
                let n = xns[i].dim(1)
                out[i] = y[0..., at ..< (at + n), 0...]
                at += n
            }
        }
        for i in xns.indices where out[i] == nil {
            out[i] = sortedAll && sorted.contains(i)
                ? bank.callRoutedSorted(xns[i], indices: indices[i], scores: scores[i])
                : bank.callRouted(xns[i], indices: indices[i], scores: scores[i])
        }
        return concatenated(out.map { $0! }, axis: 1)
    }

    /// Prefill consecutive prompt pieces (token arrays `[1, L_i]`, in order, appended to `cache`) LAYER BY LAYER, cache
    /// rows only (no final norm, no head): every piece runs exactly the ops of its own `callAsFunction` (same matrix
    /// shapes, the same cache rows visible to its attention, the fused paths for small pieces as before), except that
    /// the routed experts of a MoE layer run once over the rows of the pieces that take the expert-sorted prefill kernel
    /// (`MiniMaxM3DecoderLayer.mlpHalves`). Rows, lanes and every later logit equal piece by piece, bit for bit.
    /// `recordTaps[i]`: piece `i` records its DSpark tap rows (in piece order, as piece-by-piece calls would).
    ///
    /// Why (vMLX Python `MiniMaxM3Model.forward_pieces`, 71aee382): the aligned prefill cuts an M3 prompt at every
    /// multiple of the step, at every turn cut and at the store boundary, so a many-message chat prefills in many small
    /// pieces; each piece's routed experts decode every expert it routes to once, whatever its row count (on the real
    /// M3.1 bundle a few-hundred-row piece cost ~23 % of a full 2,048-row chunk). The pieces must stay pieces (the K / V
    /// / index projections' and the router's rows depend on the row count), the routed experts need not.
    func prefillPieces(
        _ pieces: [MLXArray], cache: [KVCache], recordTaps: [Bool], groupRouted: Bool, sortedAll: Bool = false,
        embeddings: [MLXArray?]? = nil
    ) {
        precondition(pieces.count == recordTaps.count && !pieces.isEmpty && cache.count == layers.count)
        let holder = MiniMaxM3DSpark.tapHolder(cache)
        let recording = !dsparkTapLayers.isEmpty && holder != nil
        var starts: [Int] = []
        var position = cache.first?.offset ?? 0
        for piece in pieces {
            precondition(piece.ndim == 2 && piece.dim(0) == 1, "MiniMax-M3 prefill pieces are single-sequence [1, L]")
            starts.append(position)
            position += piece.dim(1)
        }
        var hs = pieces.indices.map { embeddings?[$0] ?? embedTokens(pieces[$0]) }
        let masks: [MLXFast.ScaledDotProductAttentionMaskMode] = hs.map { $0.dim(1) > 1 ? .causal : .none }
        var taps = [[Int: MLXArray]](repeating: [:], count: pieces.count)
        for (li, layer) in layers.enumerated() {
            for i in hs.indices {
                hs[i] = layer.attentionHalf(hs[i], mask: masks[i], cache: cache[li])
            }
            hs = layer.mlpHalves(hs, groupRouted: groupRouted, sortedAll: sortedAll)
            if recording && dsparkTapLayers.contains(li) {
                for i in hs.indices where recordTaps[i] { taps[i][li] = hs[i] }
            }
        }
        if recording, let holder {
            for i in hs.indices where recordTaps[i] { self.recordTaps(taps[i], start: starts[i], holder: holder) }
        }
    }
}

public final class MiniMaxM3Model: Module, LLMModel, KVCacheDimensionProvider, SafetensorsLoadKeyExcluding,
    AlignedPrefillModel, AlignedPrefillTurnMarkerResolving, ModalityBearing
{
    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: MiniMaxM3ModelInner
    let configuration: MiniMaxM3Configuration
    /// JANGHT: the routed bank tensors the mapped layers own (`...switch_mlp.<proj>.jt_*` / `tq2_*`), kept out of the
    /// generic safetensors load. Empty for every other bundle.
    let customTensorNames: Set<String>

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    /// The image path (`MiniMaxM3Vision.swift`): built when the config carries `vision_config` and the caller wants the
    /// vision lane; nil for a text-only build (its tensors are then dropped at load).
    @ModuleInfo(key: "vision") public var vision: MiniMaxM3VisionStack?

    /// What this instance carries: text, plus vision when the stack is built.
    public var modalities: Set<ModelRuntimeRequestModality> { vision == nil ? [.text] : [.text, .vision] }

    /// The attached DSpark drafter (`dspark/`), shared by every request; nil until `attachDSparkDraft`.
    var dsparkHead: MiniMaxM3DSparkHead?
    let dsparkAttachLock = NSLock()
    /// The chat template's message markers, resolved through the serving tokenizer (`alignedPrefillTurnMarkers`).
    private var resolvedTurnMarkers: AlignedPrefillTurnMarkers?
    private let turnMarkersLock = NSLock()

    public init(_ configuration: MiniMaxM3Configuration, vision: MiniMaxM3VisionConfiguration? = nil) {
        self.configuration = configuration
        vocabularySize = configuration.vocabularySize
        kvHeads = Array(repeating: configuration.kvHeads, count: configuration.hiddenLayers)
        customTensorNames = []
        model = MiniMaxM3ModelInner(configuration)
        if !configuration.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(
                configuration.hiddenSize, configuration.vocabularySize, bias: false)
        }
        if let vision { _vision.wrappedValue = MiniMaxM3VisionStack(vision) }
    }

    /// A JANGHT bundle: every MoE layer runs its admitted mapped bank layer (`routedExperts`, keyed by layer), and
    /// `excludedTensorNames` are exactly those banks' tensors.
    init(
        _ configuration: MiniMaxM3Configuration, routedExperts: [Int: any WeightedRoutedExpertLayer],
        excludedTensorNames: Set<String>, vision: MiniMaxM3VisionConfiguration? = nil
    ) throws {
        let moeLayers = Set((0 ..< configuration.hiddenLayers).filter(configuration.isMoE))
        guard Set(routedExperts.keys) == moeLayers,
            JANGHModelPreparation.isCanonicalRoutedExclusionSet(excludedTensorNames, routedLayers: moeLayers)
        else {
            throw MiniMaxM3JANGHPreparation.PreparationError.invalid(
                "MiniMax-M3 JANGHT: routed banks \(routedExperts.keys.sorted()) do not cover the MoE layers "
                    + "\(moeLayers.sorted())")
        }
        self.configuration = configuration
        vocabularySize = configuration.vocabularySize
        kvHeads = Array(repeating: configuration.kvHeads, count: configuration.hiddenLayers)
        customTensorNames = excludedTensorNames
        model = MiniMaxM3ModelInner(configuration, routed: routedExperts)
        if !configuration.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(
                configuration.hiddenSize, configuration.vocabularySize, bias: false)
        }
        if let vision { _vision.wrappedValue = MiniMaxM3VisionStack(vision) }
    }

    public func excludeFromGenericSafetensorsLoad(key: String) -> Bool { customTensorNames.contains(key) }
    public var requiresExactTensorMmapBuffers: Bool { !customTensorNames.isEmpty }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        // A DSpark verify forward (armed on the cache, several rows): every row on a decode step's arithmetic,
        // the output head included.
        let verifying = MiniMaxM3DSpark.tapHolder(cache).map { MiniMaxM3DSpark.verifies(inputs, holder: $0) } ?? false
        let out = model(inputs, cache: cache)
        let logits: MLXArray
        if verifying {
            logits = lmHead.map { MiniMaxM3RowExact.linear($0, out) }
                ?? MiniMaxM3RowExact.rows(out) { model.embedTokens.asLinear($0) }
        } else if let lmHead {
            logits = lmHead(out)
        } else {
            logits = model.embedTokens.asLinear(out)
        }
        AlignedPrefill.hooks.logitsObserver?(logits)
        return logits
    }

    // MARK: Prompt prefill

    /// Prompt prefill. With the request's aligned plan (`AlignedPrefillPrefix.swift`, installed on the caches by the
    /// engine) the prompt is cut at every ABSOLUTE multiple of the prefill step and at the plan's store boundary, so a
    /// cold prefill and a continuation after a cache hit at a chunk boundary (or at an exact-geometry store boundary)
    /// run the same matrices for every remaining row: the hit's state and logits equal the cold prefill's bit for bit.
    /// MSA turns a few ulps of chunk-shape rounding into a different block choice, so for M3 this is what makes a
    /// cached answer the uncached answer. Without a plan (`VMLX_M3_ALIGNED_PREFIX=0`, or no engine plan): chunks of
    /// `windowSize` from wherever the cache starts, as `LLMModel`'s default prepare (the previous behaviour).
    ///
    /// Non-final pieces evaluate the cache rows only (no output head); the last piece is returned for the engine's
    /// sampling forward.
    public func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        let shape = input.text.tokens.shape
        if shape.count >= 2 && shape[0] != 1 {
            fatalError("MiniMaxM3Model.prepare expects single-sequence input (batch=1), got shape \(shape)")
        }
        let tokens = input.text.tokens.reshaped([-1])
        let mask: MLXArray? = input.text.mask.map { $0.ndim >= 2 ? $0.reshaped([-1]) : $0 }
        let count = tokens.size
        let aligned = AlignedPrefill.layers(cache)
        let start = cache.first?.offset ?? 0
        let step = max(1, windowSize ?? 512)
        let plan = aligned.first?.alignedPrefillPlan
        let laneRowsBefore = aligned.flatMap(\.alignedPrefillLaneRows)

        var pieces: [Range<Int>]
        var followsPlan = false
        // The invariant a restore must leave: every layer and lane holds exactly the rows before this prefill, and
        // they plus this input are the whole prompt the plan was made for.
        if let plan, plan.step == step, start + count == plan.promptTokenCount,
            cache.allSatisfy({ $0.offset == start }), laneRowsBefore.allSatisfy({ $0 == start }),
            aligned.allSatisfy({ $0.alignedPrefillPlan == plan })
        {
            pieces = AlignedPrefillGeometry.pieces(
                start: start, end: start + count, step: plan.step,
                boundary: AlignedPrefill.hooks.controlOmitsBoundaryCut ? nil : plan.boundary,
                turnCuts: AlignedPrefill.hooks.controlOmitsTurnCuts ? [] : plan.turnCuts ?? [])
            followsPlan = true
        } else {
            if let plan {
                // Not a refusal: the prompt still prefills, in the previous relative chunks, and its rows carry no
                // aligned geometry (no later request reuses them as aligned).
                RuntimeDiagnostics.write(Data(
                    ("[vmlx][minimax-m3] aligned prefill plan {\(plan)} does not fit this prefill (cache at \(start), "
                        + "\(count) tokens, step \(step), lanes \(Array(Set(laneRowsBefore)).sorted())): relative "
                        + "chunking\n").utf8))
            }
            pieces = []
            var position = start
            while start + count - position > step {
                pieces.append(position ..< (position + step))
                position += step
            }
            if position < start + count || pieces.isEmpty { pieces.append(position ..< (start + count)) }
        }
        // An image request: the features of the images whose tokens this input holds, and no cut inside an image run
        // (a run meets its features in one forward; the request's turn cuts, chunk boundaries outside the runs and the
        // store boundary stay).
        var splice: MiniMaxM3ImageSplice?
        if let image = input.image, let vision {
            splice = try MiniMaxM3ImageSplice.make(
                tokens: tokens.asType(.int32).asArray(Int32.self).map(Int.init), start: start, image: image,
                vision: vision)
            if let splice { pieces = splice.unsplit(pieces) }
        }
        for layer in aligned {
            layer.alignedPrefillTag = followsPlan ? plan?.tag : nil
        }
        AlignedPrefill.hooks.prefillObserver?(AlignedPrefillPrefillRecord(
            start: start, end: start + count, pieces: pieces, plan: plan, aligned: followsPlan,
            laneRowsBefore: laneRowsBefore))

        // DSpark: a piece that ends before the drafter's window (the last `dsparkTailCapacity` prompt rows) records no tap
        // rows; the tail would drop them before the drafter reads them (vMLX 0f3eb8c0). `VMLX_DSPARK_PREFILL_ALL_TAPS=1`
        // taps every piece.
        let promptEnd = start + count
        let tapsFrom: Int? =
            model.dsparkTailCapacity > 0 && !Self.prefillAllTaps ? promptEnd - model.dsparkTailCapacity : nil
        func wantsTaps(_ piece: Range<Int>) -> Bool { tapsFrom.map { piece.upperBound > $0 } ?? true }
        // Pieces of one prefill chunk (turn cuts / the store boundary inside it) run as one layer-major group
        // (`MiniMaxM3ModelInner.prefillPieces`: every piece's rows exactly its own forward's, the routed experts once
        // per group). `VMLX_M3_PREFILL_GROUP=0` runs piece by piece; `VMLX_M3_PREFILL_GROUP_ROUTED=0` keeps the
        // layer-major order with one routed call per piece (A/B).
        let groups = Self.pieceGroups(
            Array(pieces.dropLast()), step: step, grouping: followsPlan && !cache.isEmpty && Self.prefillGroupingEnabled)
        // Every non-final piece of an aligned prefill on the sorted routed tiles (`prefillSortedAll`): one rule for the
        // grouped and the piece-by-piece forms, so they stay bit-identical.
        let sortedAll = followsPlan && !cache.isEmpty && Self.prefillSortedAll
        var reserved: Bool? = nil
        for group in groups {
            // A disconnected client cancels the producer; stop between chunks instead of encoding the whole prompt.
            try Task.checkCancellation()
            if reserved == nil, !cache.isEmpty {
                reserved = MiniMaxM3SparseCache.reservePrefill(cache, rows: promptEnd - group[0].lowerBound)
            }
            let inputs = group.map { piece -> MLXArray in
                let local = (piece.lowerBound - start) ..< (piece.upperBound - start)
                return tokens[local][.newAxis, 0...]
            }
            let embeddings = splice.map { splice in
                zip(group, inputs).map { splice.embeddings(for: $0, ids: $1, embed: { model.embedTokens($0) }) }
            }
            if group.count == 1 && !sortedAll {
                _ = model(
                    inputs[0], cache: cache.isEmpty ? nil : cache, recordTaps: wantsTaps(group[0]),
                    inputEmbeddings: embeddings?[0])
            } else if group.count > 1 && Self.prefillBatchRows {
                model.prefillPiecesBatched(
                    inputs, cache: cache, recordTaps: group.map(wantsTaps), groupRouted: Self.prefillGroupRouted,
                    sortedAll: sortedAll, embeddings: embeddings)
            } else {
                model.prefillPieces(
                    inputs, cache: cache, recordTaps: group.map(wantsTaps), groupRouted: Self.prefillGroupRouted,
                    sortedAll: sortedAll, embeddings: embeddings)
            }
            // `VMLX_M3_PREFILL_ASYNC=1` (A/B, default off): the next group's graph is built while this one runs.
            if Self.prefillAsync { asyncEval(cache) } else { MLX.eval(cache) }
            PrefillProgressReporter.reportCompletedUnits(group[group.count - 1].upperBound - start)
            // The buffer cache is kept through the prefill (each chunk's transients are reused, not re-allocated) and
            // released once at its end (vMLX 0967ec2f); `VMLX_M3_PREFILL_CLEAR_EACH_CHUNK=1` = the per-chunk release.
            if Self.prefillClearEachChunk { Memory.clearCache() }
        }
        if !groups.isEmpty, !Self.prefillClearEachChunk { Memory.clearCache() }
        guard let last = pieces.last else {
            return .tokens(LMInput.Text(tokens: tokens, mask: mask))
        }
        let local = (last.lowerBound - start) ..< (last.upperBound - start)
        if let splice, splice.touches(last) {
            // Image tokens in the final piece (not produced by the chat template, whose generation prompt follows the
            // last message): it runs here with its features, and its logits go to the sampler.
            let ids = tokens[local][.newAxis, 0...]
            let out = model(
                ids, cache: cache.isEmpty ? nil : cache,
                inputEmbeddings: splice.embeddings(for: last, ids: ids, embed: { model.embedTokens($0) }))
            let logits = lmHead.map { $0(out) } ?? model.embedTokens.asLinear(out)
            AlignedPrefill.hooks.logitsObserver?(logits)
            return .logits(LMOutput(logits: logits))
        }
        return .tokens(LMInput.Text(tokens: tokens[local], mask: mask.map { $0[local] }))
    }

    /// One cache per layer, following the attention type of each layer: sparse (MSA) layers keep keys, values and
    /// indexer keys; dense attention layers keep keys and values. MSA blocks are anchored to absolute positions, so
    /// no layer uses a rotating window (`maxKVSize` is not applied).
    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        (0 ..< configuration.hiddenLayers).map { layer in
            configuration.isSparse(layer) ? MiniMaxM3SparseCache() as KVCache : KVCacheSimple()
        }
    }

    /// The sparse layers' cache is not a promotable compiled cache, and the block selection depends on the host-side
    /// cache length.
    public var supportsWholeForwardCompilation: Bool { false }

    /// Every M3 prompt prefill is cut at absolute chunk boundaries and the store boundary (`prepare`), so prefix-cache
    /// hits are bit-identical to a cold prefill. `VMLX_M3_ALIGNED_PREFIX=0` turns it off (previous behaviour). With the
    /// template's message markers resolved from the serving tokenizer it also cuts at every non-assistant message end
    /// (v2 turn cuts; `VMLX_M3_STRUCTURAL_CUTS=0` keeps v1); the markers are part of the identity (cache salt).
    public var alignedPrefillSchema: String? {
        guard AlignedPrefillGeometry.isEnabled() else { return nil }
        // Pieces whose routed experts all run on the sorted tiles carry other numbers than the stock rule's.
        let sorted = Self.prefillSortedAll ? ":routed=sorted\(Self.prefillSortedMinPairs)" : ""
        if let markers = alignedPrefillTurnMarkers {
            return "minimax-m3:" + AlignedPrefillGeometry.turnsSchema + ":turns=" + markers.signature + sorted
        }
        return "minimax-m3:" + AlignedPrefillGeometry.schema + sorted
    }

    /// The M3 chat template's message markers (`]~b]` + role, `[e~[`; assistant role `ai\n`), resolved through the
    /// serving tokenizer by `ModelContext` (`resolveAlignedPrefillTurnMarkers`); nil without a resolving tokenizer or
    /// with `VMLX_M3_STRUCTURAL_CUTS=0`.
    public var alignedPrefillTurnMarkers: AlignedPrefillTurnMarkers? {
        guard AlignedPrefillGeometry.structuralCutsEnabled() else { return nil }
        return turnMarkersLock.withLock { resolvedTurnMarkers }
    }

    public func resolveAlignedPrefillTurnMarkers(tokenizer: any Tokenizer) {
        let markers = AlignedPrefillTurnMarkers.resolve(
            tokenizer: tokenizer, bosText: "]~b]", eosText: "[e~[", assistantRoleText: "ai\n")
        turnMarkersLock.withLock { resolvedTurnMarkers = markers }
    }

    /// Tests: set the markers directly (a stand-in tokenizer has no M3 vocabulary).
    func setAlignedPrefillTurnMarkers(_ markers: AlignedPrefillTurnMarkers?) {
        turnMarkersLock.withLock { resolvedTurnMarkers = markers }
    }

    /// The three-lane sparse cache has no batched (B > 1) form yet.
    public var maximumSupportedDecodeBatchSize: Int? { 1 }

    /// Measured on the real M3.1 JANGHT1.9 bundle (Mac Studio, `M3RealT5 stepab`, cold prefill, interleaved, 3 probes):
    /// 8,269 tokens 959 / 719 / 621 pp/s and 32,845 tokens 938 / 657 / 608 pp/s at steps 2048 / 1024 / 512 (smaller
    /// chunks run the routed experts at a quarter of the rows per expert). Osaurus applies it when neither the request
    /// nor a per-model tuning record picks a step; without it the library default 512 ran its 35k-token turns. It is
    /// also the aligned-prefill chunk C of vMLX Python's M3 runtime.
    public var recommendedPrefillStepSize: Int? { 2048 }

    // MARK: Prefill levers (each A/B switch restores the previous behaviour; every default is bit-identical to it)

    /// Tests: per-task overrides of the environment levers below (nil = the environment's).
    @TaskLocal static var prefillGroupingOverride: Bool?
    @TaskLocal static var prefillGroupRoutedOverride: Bool?
    @TaskLocal static var prefillClearEachChunkOverride: Bool?
    @TaskLocal static var prefillAllTapsOverride: Bool?

    private static func flag(_ name: String, default value: Bool) -> Bool {
        guard let raw = ProcessInfo.processInfo.environment[name]?.trimmingCharacters(in: .whitespaces).lowercased(),
            !raw.isEmpty
        else { return value }
        return !["0", "false", "off", "no"].contains(raw)
    }

    /// `VMLX_M3_PREFILL_GROUP` (default on): the pieces of one prefill chunk run as one layer-major group.
    static var prefillGroupingEnabled: Bool { prefillGroupingOverride ?? flag("VMLX_M3_PREFILL_GROUP", default: true) }
    /// `VMLX_M3_PREFILL_GROUP_ROUTED` (default on): inside a group, one routed-expert call for the sorted-kernel pieces.
    static var prefillGroupRouted: Bool {
        prefillGroupRoutedOverride ?? flag("VMLX_M3_PREFILL_GROUP_ROUTED", default: true)
    }
    /// `VMLX_M3_PREFILL_CLEAR_EACH_CHUNK` (default off): release MLX's buffer cache after every prefill chunk (before
    /// 2026-10-29) instead of once at the end of the prefill.
    static var prefillClearEachChunk: Bool {
        prefillClearEachChunkOverride ?? flag("VMLX_M3_PREFILL_CLEAR_EACH_CHUNK", default: false)
    }
    /// `VMLX_M3_PREFILL_SORTED` (default on; `=0` = the stock 128-row rule): the JANGT routed experts of every non-final piece of an
    /// aligned prefill with at least `prefillSortedMinPairs` (row, slot) pairs (18 rows at top-4, vMLX Python's rule) run
    /// on the expert-sorted prefill tiles; the stock rule sends pieces under 128 rows (512 pairs) to the decode kernels,
    /// where each small piece decoded nearly every expert on its own (on a many-message chat most pieces are small). A
    /// row's output on those tiles depends only on its own input, so the grouped and piece-by-piece forms stay
    /// bit-identical; the pieces of 18-127 rows get another kernel family's numbers, so the rule is part of the cache
    /// identity (`alignedPrefillSchema`).
    @TaskLocal static var prefillSortedAllOverride: Bool?
    static var prefillSortedAll: Bool { prefillSortedAllOverride ?? flag("VMLX_M3_PREFILL_SORTED", default: true) }
    /// The (row, slot) pairs from which a non-final piece takes the sorted tiles under `prefillSortedAll`: a number
    /// value of `VMLX_M3_PREFILL_SORTED` (e.g. 128), never below `sortedFloorPairs`; any other "on" value = the floor.
    @TaskLocal static var prefillSortedMinPairsOverride: Int?
    static var prefillSortedMinPairs: Int {
        let raw = ProcessInfo.processInfo.environment["VMLX_M3_PREFILL_SORTED"]?.trimmingCharacters(in: .whitespaces)
        let wanted = prefillSortedMinPairsOverride ?? raw.flatMap(Int.init) ?? sortedFloorPairs
        return max(sortedFloorPairs, wanted)
    }
    /// vMLX Python's `SORT_THRESHOLD` (72 pairs = 18 rows at top-4), and a floor: on the real M3.1 bundle the JANGT
    /// sorted NAX tiles fail to build their Metal kernel for a 1-row call ("no matching function for call to
    /// jt_gather_qmm_nax", `M3RealT5 rowdep`, Studio 2026-10-29), so smaller pieces always keep the decode kernels.
    static let sortedFloorPairs = 72
    /// `VMLX_M3_PREFILL_BATCH_ROWS` (default on; `=0` = `prefillPieces`): inside a group, the row-wise ops of all pieces
    /// in one call each (`MiniMaxM3ModelInner.prefillPiecesBatched`); the row-count-dependent matmuls stay per piece
    /// unless this device's probe admits them. Bit-identical. Studio, raw held-out stream 8,269 tokens: 625 -> 701 pp/s.
    @TaskLocal static var prefillBatchRowsOverride: Bool?
    static var prefillBatchRows: Bool {
        prefillBatchRowsOverride ?? flag("VMLX_M3_PREFILL_BATCH_ROWS", default: true)
    }
    /// `VMLX_M3_PREFILL_ASYNC` (default off, A/B): evaluate each prefill group asynchronously, so the CPU builds the next
    /// group's graph while the GPU runs this one.
    @TaskLocal static var prefillAsyncOverride: Bool?
    static var prefillAsync: Bool { prefillAsyncOverride ?? flag("VMLX_M3_PREFILL_ASYNC", default: false) }
    /// `VMLX_DSPARK_PREFILL_ALL_TAPS` (default off): record DSpark taps for every prefill piece, also those before the
    /// drafter's window.
    static var prefillAllTaps: Bool { prefillAllTapsOverride ?? flag("VMLX_DSPARK_PREFILL_ALL_TAPS", default: false) }

    /// `pieces` (absolute, consecutive) grouped by the prefill chunk they start in (`lowerBound / step`), or one group
    /// per piece when `grouping` is false.
    static func pieceGroups(_ pieces: [Range<Int>], step: Int, grouping: Bool) -> [[Range<Int>]] {
        guard grouping else { return pieces.map { [$0] } }
        var groups: [[Range<Int>]] = []
        let step = max(1, step)
        for piece in pieces {
            if let first = groups.last?.first, first.lowerBound / step == piece.lowerBound / step {
                groups[groups.count - 1].append(piece)
            } else {
                groups.append([piece])
            }
        }
        return groups
    }

    /// Keep the checkpoint's dtypes (F32 router and per-expert scales, BF16 norms and embeddings), as the Python
    /// runtime does; the model owns its activation casts.
    public var preservesCheckpointParameterDTypes: Bool { true }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = [String: MLXArray]()
        for (key, value) in weights {
            // MTP heads are not part of the model; the vision stack is, when it is built.
            if key.hasPrefix("mtp") { continue }
            if let (name, tensor) = MiniMaxM3VisionStack.sanitize(key: key, value: value) {
                if vision != nil { sanitized[name] = tensor }
                continue
            }
            var name = key
            if name.hasPrefix("language_model.model.") {
                name = "model." + name.dropFirst("language_model.model.".count)
            } else if name.hasPrefix("language_model.lm_head") {
                name = "lm_head" + name.dropFirst("language_model.lm_head".count)
            }
            name = name.replacingOccurrences(of: ".block_sparse_moe.", with: ".mlp.")
            // The checkpoint keeps the indexer projections flat on self_attn; this model nests them.
            name = name.replacingOccurrences(of: ".self_attn.index_", with: ".self_attn.indexer.index_")
            sanitized[name] = value
        }
        if configuration.tieWordEmbeddings {
            sanitized["lm_head.weight"] = nil
        }
        if MiniMaxM3SourceFormat.isSourceCheckpoint(sanitized) {
            sanitized = MiniMaxM3SourceFormat.install(into: self, weights: sanitized)
        }
        return sanitized
    }
}

extension MiniMaxM3Model: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}

// MARK: - Configuration

/// Reads `config.json` as shipped: the text fields from `text_config` (or the top level when there is none) and the
/// MSA fields from `text_config.sparse_attention_config`. A top-level `num_local_experts` wins over the text one.
public struct MiniMaxM3Configuration: Codable, Sendable {

    public struct SparseAttention: Codable, Sendable {
        var indexHeads: Int = 4
        var indexHeadDim: Int = 128
        var blockSize: Int = 128
        var topKBlocks: Int = 16
        var localBlocks: Int = 1
        /// Per layer, nonzero = MSA. Absent: layers 3 and up.
        var layerFrequency: [Int]?

        enum CodingKeys: String, CodingKey {
            case indexHeads = "sparse_num_index_heads"
            case indexHeadDim = "sparse_index_dim"
            case blockSize = "sparse_block_size"
            case topKBlocks = "sparse_topk_blocks"
            case localBlocks = "sparse_local_block"
            case layerFrequency = "sparse_attention_freq"
        }

        init() {}

        public init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            indexHeads = try container.decodeIfPresent(Int.self, forKey: .indexHeads) ?? indexHeads
            indexHeadDim = try container.decodeIfPresent(Int.self, forKey: .indexHeadDim) ?? indexHeadDim
            blockSize = try container.decodeIfPresent(Int.self, forKey: .blockSize) ?? blockSize
            topKBlocks = try container.decodeIfPresent(Int.self, forKey: .topKBlocks) ?? topKBlocks
            localBlocks = try container.decodeIfPresent(Int.self, forKey: .localBlocks) ?? localBlocks
            layerFrequency = try decodeLayerFlags(container, forKey: .layerFrequency)
        }
    }

    var modelType: String
    var hiddenSize: Int
    var hiddenLayers: Int
    var intermediateSize: Int
    var denseIntermediateSize: Int
    var sharedIntermediateSize: Int
    var attentionHeads: Int
    var kvHeads: Int
    var headDim: Int
    var rotaryDim: Int
    var ropeTheta: Float
    var rmsNormEps: Float
    var vocabularySize: Int
    var numLocalExperts: Int
    var numExpertsPerTok: Int
    var routedScalingFactor: Float
    var normTopKProb: Bool
    var swigluAlpha: Float
    var swigluLimit: Float
    /// Per layer, nonzero = MoE. Absent: layers 3 and up.
    var moeLayerFrequency: [Int]?
    var tieWordEmbeddings: Bool
    var sparseAttention: SparseAttention

    func isMoE(_ layer: Int) -> Bool {
        moeLayerFrequency.map { $0[layer] != 0 } ?? (layer >= 3)
    }

    func isSparse(_ layer: Int) -> Bool {
        sparseAttention.layerFrequency.map { $0[layer] != 0 } ?? (layer >= 3)
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case denseIntermediateSize = "dense_intermediate_size"
        case sharedIntermediateSize = "shared_intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case rotaryDim = "rotary_dim"
        case ropeTheta = "rope_theta"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case numLocalExperts = "num_local_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case routedScalingFactor = "routed_scaling_factor"
        case normTopKProb = "norm_topk_prob"
        case swigluAlpha = "swiglu_alpha"
        case swigluLimit = "swiglu_limit"
        case moeLayerFrequency = "moe_layer_freq"
        case tieWordEmbeddings = "tie_word_embeddings"
        case sparseAttention = "sparse_attention_config"
    }

    private enum WrapperKeys: String, CodingKey {
        case modelType = "model_type"
        case textConfig = "text_config"
        case numLocalExperts = "num_local_experts"
    }

    public init(from decoder: Decoder) throws {
        let wrapper = try decoder.container(keyedBy: WrapperKeys.self)
        let text =
            wrapper.contains(.textConfig)
            ? try wrapper.nestedContainer(keyedBy: CodingKeys.self, forKey: .textConfig)
            : try decoder.container(keyedBy: CodingKeys.self)

        modelType = try wrapper.decodeIfPresent(String.self, forKey: .modelType) ?? "minimax_m3_vl"
        hiddenSize = try text.decode(Int.self, forKey: .hiddenSize)
        hiddenLayers = try text.decode(Int.self, forKey: .hiddenLayers)
        intermediateSize = try text.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 3072
        denseIntermediateSize =
            try text.decodeIfPresent(Int.self, forKey: .denseIntermediateSize) ?? 12288
        sharedIntermediateSize =
            try text.decodeIfPresent(Int.self, forKey: .sharedIntermediateSize) ?? 3072
        attentionHeads = try text.decode(Int.self, forKey: .attentionHeads)
        kvHeads = try text.decode(Int.self, forKey: .kvHeads)
        headDim = try text.decodeIfPresent(Int.self, forKey: .headDim) ?? 128
        rotaryDim = try text.decodeIfPresent(Int.self, forKey: .rotaryDim) ?? 64
        ropeTheta = try text.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 5_000_000
        rmsNormEps = try text.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        vocabularySize = try text.decode(Int.self, forKey: .vocabularySize)
        numLocalExperts =
            try wrapper.decodeIfPresent(Int.self, forKey: .numLocalExperts)
            ?? text.decodeIfPresent(Int.self, forKey: .numLocalExperts) ?? 100
        numExpertsPerTok = RuntimeMoETopKOverride.effectiveTopK(
            currentTopK: try text.decodeIfPresent(Int.self, forKey: .numExpertsPerTok) ?? 4,
            modelType: modelType,
            field: CodingKeys.numExpertsPerTok.rawValue)
        routedScalingFactor =
            try text.decodeIfPresent(Float.self, forKey: .routedScalingFactor) ?? 2.0
        normTopKProb = try text.decodeIfPresent(Bool.self, forKey: .normTopKProb) ?? true
        swigluAlpha = try text.decodeIfPresent(Float.self, forKey: .swigluAlpha) ?? 1.702
        swigluLimit = try text.decodeIfPresent(Float.self, forKey: .swigluLimit) ?? 7.0
        moeLayerFrequency = try decodeLayerFlags(text, forKey: .moeLayerFrequency)
        tieWordEmbeddings = try text.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        sparseAttention =
            try text.decodeIfPresent(SparseAttention.self, forKey: .sparseAttention) ?? SparseAttention()

        for (key, flags) in [
            (CodingKeys.moeLayerFrequency, moeLayerFrequency),
            (CodingKeys.sparseAttention, sparseAttention.layerFrequency),
        ] {
            if let flags, flags.count != hiddenLayers {
                throw DecodingError.dataCorruptedError(
                    forKey: key, in: text,
                    debugDescription:
                        "\(key.rawValue) lists \(flags.count) layers; num_hidden_layers is \(hiddenLayers)")
            }
        }
    }
}

/// Per-layer flags may be written as 0 / 1 or as booleans.
private func decodeLayerFlags<K: CodingKey>(
    _ container: KeyedDecodingContainer<K>, forKey key: K
) throws -> [Int]? {
    guard container.contains(key), try !container.decodeNil(forKey: key) else { return nil }
    if let flags = try? container.decode([Int].self, forKey: key) {
        return flags
    }
    return try container.decode([Bool].self, forKey: key).map { $0 ? 1 : 0 }
}

/// Per-device row invariance of a projection (`Linear` / `QuantizedLinear`): the smallest row count `T` from which a
/// row's output does not depend on how many rows the call carries, up to `maxRows`, or nil. MLX picks split-K, tile
/// sizes and the vector kernel by the row count M and the device, so it is measured, not assumed: once per (module type,
/// shape, quantization, dtype, maxRows), on random inputs, op(x[..< m]) == op(x)[..< m] bit for bit for every m from
/// the candidate to the candidate + 64 and then every 32 rows to `maxRows` (MLX's choices change at tile multiples).
/// Candidates 33, 65, 129, 257, 513, 1025: the first that holds is admitted. (vMLX Python measured on MLX 0.32 / M5: q /
/// o / shared-down / dense-down from 33 rows, shared gate / up from 65, k / v / index-q from 513, index-k and the fp32
/// router only past ~1,500.) `VMLX_M3_ROW_INVARIANCE=0` admits nothing (every projection per piece).
enum MiniMaxM3RowInvariance {
    /// The row counts probed at least (a prefill step chunk).
    static let probeRows = 2048
    static let candidates = [33, 65, 129, 257, 513, 1025]
    private static let lock = NSLock()
    nonisolated(unsafe) private static var admitted: [String: Int?] = [:]
    /// Tests: the probe results so far.
    static var results: [String: Int?] { lock.withLock { admitted } }

    static func key(_ linear: Linear, maxRows: Int, dtype: DType) -> String {
        var key = "\(type(of: linear))|\(linear.weight.shape)|\(linear.weight.dtype)|\(dtype)|\(maxRows)"
        if let q = linear as? QuantizedLinear {
            key += "|\(q.mode)|\(q.bits)|\(q.groupSize)"
        }
        return key
    }

    /// CONTROL ONLY (tests that must fail): every projection admitted from this row count, unprobed.
    @TaskLocal static var controlForcedThreshold: Int?

    static func threshold(_ linear: Linear, maxRows: Int, dtype: DType) -> Int? {
        if let forced = controlForcedThreshold { return forced }
        guard ProcessInfo.processInfo.environment["VMLX_M3_ROW_INVARIANCE"] != "0" else { return nil }
        let k = key(linear, maxRows: maxRows, dtype: dtype)
        if let known = lock.withLock({ admitted[k] }) { return known }
        let result = probe(linear, maxRows: maxRows, dtype: dtype)
        lock.withLock { admitted[k] = result }
        RuntimeDiagnostics.write(Data(
            "[vmlx][minimax-m3] row invariance \(k): \(result.map { "from \($0) rows" } ?? "none")\n".utf8))
        return result
    }

    static func probe(_ linear: Linear, maxRows: Int, dtype: DType) -> Int? {
        let inputs = (linear as? QuantizedLinear).map { $0.scales.dim(-1) * $0.groupSize } ?? linear.weight.dim(-1)
        let x = (MLXRandom.normal([1, maxRows, inputs], key: MLXRandom.key(29)) * Float(0.5)).asType(dtype)
        let full = linear(x)
        eval(full)
        for candidate in candidates where candidate <= maxRows {
            var sizes = Array(candidate ... min(maxRows, candidate + 64))
            sizes += stride(from: candidate + 96, through: maxRows, by: 32)
            if !sizes.contains(maxRows) { sizes.append(maxRows) }
            var holds = true
            for m in sizes where holds {
                // Rows are independent inputs; the kernel a row count picks is what is probed.
                holds = arrayEqual(linear(x[0..., ..<m, 0...]), full[0..., ..<m, 0...]).item(Bool.self)
            }
            if holds { return candidate }
        }
        return nil
    }
}
