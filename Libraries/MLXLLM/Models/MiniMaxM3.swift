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

    init(dimensions: Int, eps: Float) {
        self.weight = MLXArray.zeros([dimensions])
        self.eps = eps
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: 1 + weight, eps: eps)
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
    func project(_ x: MLXArray, offset: Int) -> (queries: MLXArray, keys: MLXArray) {
        let (B, queryLength) = (x.dim(0), x.dim(1))
        let queries = queryNorm(queryProjection(x).reshaped(B, queryLength, heads, headDim))
            .transposed(0, 2, 1, 3)
        let keys = keyNorm(keyProjection(x).reshaped(B, queryLength, 1, headDim)).transposed(0, 2, 1, 3)
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

    init(_ config: MiniMaxM3Configuration, layer: Int) {
        heads = config.attentionHeads
        kvHeads = config.kvHeads
        headDim = config.headDim
        scale = pow(Float(headDim), -0.5)
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
    func project(_ x: MLXArray, offset: Int) -> (queries: MLXArray, keys: MLXArray, values: MLXArray) {
        let (B, L) = (x.dim(0), x.dim(1))
        let queries = qNorm(wq(x).reshaped(B, L, heads, headDim)).transposed(0, 2, 1, 3)
        let keys = kNorm(wk(x).reshaped(B, L, kvHeads, headDim)).transposed(0, 2, 1, 3)
        let values = wv(x).reshaped(B, L, kvHeads, headDim).transposed(0, 2, 1, 3)
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
        // Sigmoid router; the correction bias steers the choice but not the weights.
        let routerScores = sigmoid(gate(x).asType(.float32))
        let choice = routerScores + eScoreCorrectionBias
        let indices = argPartition(-choice, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
        var scores = takeAlong(routerScores, indices, axis: -1)
        if topK > 1 && normTopKProb {
            scores = scores / scores.sum(axis: -1, keepDims: true)
        }
        scores = scores * routedScalingFactor

        let routed: MLXArray
        if let switchMLP {
            routed = (switchMLP(x, indices) * scores[.ellipsis, .newAxis]).sum(axis: -2)
        } else {
            // Mapped banks: the router-weighted sum in F32, returned in x's dtype (Python `JTMixedSwitchGLU.routed`).
            routed = (experts as! any WeightedRoutedExpertLayer).callRouted(x, indices: indices, scores: scores)
        }
        return (routed + sharedExperts(x)).asType(x.dtype)
    }
}

final class MiniMaxM3DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: MiniMaxM3Attention
    @ModuleInfo(key: "mlp") var mlp: UnaryLayer
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: MiniMaxM3RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: MiniMaxM3RMSNorm

    init(_ config: MiniMaxM3Configuration, layer: Int, routed: (any WeightedRoutedExpertLayer)? = nil) {
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
        let h = x + selfAttn(inputLayerNorm(x), mask: mask, cache: cache)
        return h + mlp(postAttentionLayerNorm(h))
    }
}

public final class MiniMaxM3ModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    let layers: [MiniMaxM3DecoderLayer]
    @ModuleInfo(key: "norm") var norm: MiniMaxM3RMSNorm

    init(_ config: MiniMaxM3Configuration, routed: [Int: any WeightedRoutedExpertLayer] = [:]) {
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        layers = (0 ..< config.hiddenLayers).map {
            MiniMaxM3DecoderLayer(config, layer: $0, routed: routed[$0])
        }
        _norm.wrappedValue = MiniMaxM3RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = embedTokens(inputs)
        // The Python runtime's mask (mlx_lm create_attention_mask): symbolic causal for every multi-token call, at any
        // cache offset (MLX SDPA aligns a causal mask bottom-right, so the key length carries the offset); none for
        // one token. Dense layers use it as is; sparse layers fold causality into their block selection. The cache's
        // own mask is an explicit array once its offset is nonzero, and MSA routes any array mask (batch padding) to
        // the dense mask path: every prefill chunk after the first, and every multi-row verify call, would skip the
        // block-sparse kernels.
        let mask: MLXFast.ScaledDotProductAttentionMaskMode = h.dim(1) > 1 ? .causal : .none
        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }
        return norm(h)
    }
}

public final class MiniMaxM3Model: Module, LLMModel, KVCacheDimensionProvider, SafetensorsLoadKeyExcluding {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: MiniMaxM3ModelInner
    let configuration: MiniMaxM3Configuration
    /// JANGHT: the routed bank tensors the mapped layers own (`...switch_mlp.<proj>.jt_*` / `tq2_*`), kept out of the
    /// generic safetensors load. Empty for every other bundle.
    let customTensorNames: Set<String>

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ configuration: MiniMaxM3Configuration) {
        self.configuration = configuration
        vocabularySize = configuration.vocabularySize
        kvHeads = Array(repeating: configuration.kvHeads, count: configuration.hiddenLayers)
        customTensorNames = []
        model = MiniMaxM3ModelInner(configuration)
        if !configuration.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(
                configuration.hiddenSize, configuration.vocabularySize, bias: false)
        }
    }

    /// A JANGHT bundle: every MoE layer runs its admitted mapped bank layer (`routedExperts`, keyed by layer), and
    /// `excludedTensorNames` are exactly those banks' tensors.
    init(
        _ configuration: MiniMaxM3Configuration, routedExperts: [Int: any WeightedRoutedExpertLayer],
        excludedTensorNames: Set<String>
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
    }

    public func excludeFromGenericSafetensorsLoad(key: String) -> Bool { customTensorNames.contains(key) }
    public var requiresExactTensorMmapBuffers: Bool { !customTensorNames.isEmpty }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let out = model(inputs, cache: cache)
        if let lmHead {
            return lmHead(out)
        }
        return model.embedTokens.asLinear(out)
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

    /// The three-lane sparse cache has no batched (B > 1) form yet.
    public var maximumSupportedDecodeBatchSize: Int? { 1 }

    /// Keep the checkpoint's dtypes (F32 router and per-expert scales, BF16 norms and embeddings), as the Python
    /// runtime does; the model owns its activation casts.
    public var preservesCheckpointParameterDTypes: Bool { true }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = [String: MLXArray]()
        for (key, value) in weights {
            // MTP heads and the vision stack are not part of the text model.
            if key.hasPrefix("mtp") || key.hasPrefix("vision_tower.")
                || key.hasPrefix("multi_modal_projector.") || key.hasPrefix("patch_merge_mlp.")
            {
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
