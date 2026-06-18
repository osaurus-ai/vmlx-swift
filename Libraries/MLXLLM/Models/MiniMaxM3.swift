// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// MiniMax-M3 (minimax_m3 / minimax_m3_vl) text runtime — full MSA decode.
//
// Port of vllm-mlx `models/minimax_m3/minimax_m3.py`. M3 is a GQA MoE decoder
// (n_kv=4, head_dim=128) with two attention regimes:
//   * layers 0-2  — dense full causal attention (stock `KVCacheSimple`)
//   * layers 3-59 — MiniMax Sparse Attention (MSA): a Lightning Indexer picks the
//     top-k 128-token key blocks per query and the main branch attends only those
//     via an additive block mask. Indexer keys live in `MiniMaxM3SparseCache`
//     alongside K/V; selection is recomputed every step.
// Below topk*block (=2048) tokens every block is visible, so MSA reduces to full
// causal attention. Quant is standard affine (per-module from config.json); the
// Load path quantizes only modules whose checkpoint carries `.scales`, so the
// indexer projections, router gate, and all norms stay fp16 with no custom
// predicate.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - gpt_oss SwiGLU (clamped, alpha-scaled) — gate+up fused activation

/// gpt_oss swiglu: `(xLinear+1) * (clamp(xGlu) * sigmoid(alpha*clamp(xGlu)))`,
/// with xGlu clamped to `max=limit` and xLinear to `[-limit, limit]`. Matches
/// `mlx_lm.models.gpt_oss.swiglu`. Caller convention: `xLinear` = up, `xGlu` = gate.
private func minimaxM3Swiglu(
    xLinear: MLXArray, xGlu: MLXArray, alpha: Float = 1.702, limit: Float = 7.0
) -> MLXArray {
    let glu = clip(xGlu, max: MLXArray(limit, dtype: xGlu.dtype))
    let lin = clip(
        xLinear, min: MLXArray(-limit, dtype: xLinear.dtype),
        max: MLXArray(limit, dtype: xLinear.dtype))
    let outGlu = glu * sigmoid(alpha * glu)
    return outGlu * (lin + 1)
}

/// Dense / shared-expert MLP using the gpt_oss clamped swiglu.
private class SwiGLUOAIMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear
    let alpha: Float
    let limit: Float

    init(dimensions: Int, hiddenDimensions: Int, alpha: Float, limit: Float) {
        self._gateProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        self._upProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        self._downProj.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        self.alpha = alpha
        self.limit = limit
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(minimaxM3Swiglu(xLinear: upProj(x), xGlu: gateProj(x), alpha: alpha, limit: limit))
    }
}

// MARK: - Lightning Indexer (block selection)

/// Scores `idx_q` against cached `idx_k`, max-pools per `block`-token block,
/// selects top-k blocks per query, and returns an additive mask `[B, 1, Sq, Sk]`
/// (0 on allowed keys, -inf elsewhere) that composes with the causal mask.
/// Returns `nil` when every block is visible (short context) → caller uses the
/// plain causal mask (full attention).
private class MiniMaxM3Indexer: Module {
    let nh: Int
    let d: Int
    let block: Int
    let topk: Int
    let local: Int

    @ModuleInfo(key: "index_q_proj") var indexQProj: Linear
    @ModuleInfo(key: "index_k_proj") var indexKProj: Linear
    @ModuleInfo(key: "index_q_norm") var indexQNorm: GemmaRMSNorm
    @ModuleInfo(key: "index_k_norm") var indexKNorm: GemmaRMSNorm
    let rope: RoPE

    init(_ args: MiniMaxM3Configuration) {
        self.nh = args.indexNHeads
        self.d = args.indexHeadDim
        self.block = args.indexBlockSize
        self.topk = args.indexTopkBlocks
        self.local = args.indexLocalBlocks
        self._indexQProj.wrappedValue = Linear(args.hiddenSize, nh * d, bias: false)
        self._indexKProj.wrappedValue = Linear(args.hiddenSize, d, bias: false)
        self._indexQNorm.wrappedValue = GemmaRMSNorm(dimensions: d, eps: args.rmsNormEps)
        self._indexKNorm.wrappedValue = GemmaRMSNorm(dimensions: d, eps: args.rmsNormEps)
        self.rope = RoPE(dimensions: args.rotaryDim, traditional: false, base: args.ropeTheta)
    }

    func callAsFunction(_ x: MLXArray, cache: MiniMaxM3SparseCache, offset: Int) -> MLXArray? {
        let (B, Sq) = (x.dim(0), x.dim(1))
        var idxQ = indexQNorm(indexQProj(x).reshaped(B, Sq, nh, d)).transposed(0, 2, 1, 3)
        var idxK = indexKNorm(indexKProj(x).reshaped(B, Sq, 1, d)).transposed(0, 2, 1, 3)
        idxQ = rope(idxQ, offset: offset)
        idxK = rope(idxK, offset: offset)
        idxK = cache.updateIndex(idxK)  // [B, 1, Sk, d], lockstep with K/V
        let Sk = idxK.dim(2)

        // scores [B, nh, Sq, Sk] in fp32, masked causal
        var scores = matmul(idxQ.asType(.float32), idxK.asType(.float32).transposed(0, 1, 3, 2))
        let qPos = MLXArray(offset ..< (offset + Sq)).reshaped(1, 1, Sq, 1)
        let kPos = MLXArray(0 ..< Sk).reshaped(1, 1, 1, Sk)
        let negInf = MLXArray(-Float.infinity, dtype: .float32)
        scores = MLX.where(kPos .> qPos, negInf, scores)

        // pad Sk to a block multiple, max-pool per block, max over heads
        let nBlocks = (Sk + block - 1) / block
        let pad = nBlocks * block - Sk
        if pad > 0 {
            let padTensor = MLX.full([B, nh, Sq, pad], values: negInf, dtype: .float32)
            scores = concatenated([scores, padTensor], axis: -1)
        }
        scores = scores.reshaped(B, nh, Sq, nBlocks, block)
        var blockScores = scores.max(axis: -1).max(axis: 1)  // [B, Sq, nBlocks]

        // force each query's own (local) block(s) to always win
        let qBlock = (MLXArray(offset ..< (offset + Sq)) / MLXArray(Int32(block)))  // [Sq]
        if local > 0 {
            let loc = MLXArray(0 ..< local).reshaped(1, 1, local)
            var localIdx = MLX.maximum(qBlock.reshaped(1, Sq, 1) - loc, MLXArray(Int32(0)))
            localIdx = broadcast(localIdx, to: [B, Sq, local])
            blockScores = putAlong(
                blockScores, localIdx, values: MLXArray(Float.infinity, dtype: .float32), axis: -1)
        }

        let keep = min(topk, nBlocks)
        if keep >= nBlocks {
            return nil  // all blocks visible → only causal matters
        }

        // top-k blocks per query → [B,1,Sq,Sk] keep-bias (0 kept / -inf else)
        let topIdx = argPartition(-blockScores, kth: keep - 1, axis: -1)[.ellipsis, ..<keep]
        var blockKeep = MLX.full([B, Sq, nBlocks], values: negInf, dtype: .float32)
        blockKeep = putAlong(
            blockKeep, topIdx, values: MLXArray(Float(0), dtype: .float32), axis: -1)
        var keyBias = repeated(blockKeep, count: block, axis: -1)[0..., 0..., 0 ..< Sk]  // [B,Sq,Sk]
        keyBias = keyBias.reshaped(B, 1, Sq, Sk)
        keyBias = MLX.where(kPos .> qPos, negInf, keyBias)  // re-enforce causal
        return keyBias
    }
}

// MARK: - Attention (dense + MSA)

private class MiniMaxM3Attention: Module {
    let nHeads: Int
    let nKV: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: GemmaRMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: GemmaRMSNorm
    let rope: RoPE
    @ModuleInfo(key: "indexer") var indexer: MiniMaxM3Indexer?

    init(_ args: MiniMaxM3Configuration, layerIdx: Int) {
        self.nHeads = args.attentionHeads
        self.nKV = args.kvHeads
        self.headDim = args.headDim
        self.scale = pow(Float(headDim), -0.5)
        let d = args.hiddenSize
        self._qProj.wrappedValue = Linear(d, nHeads * headDim, bias: false)
        self._kProj.wrappedValue = Linear(d, nKV * headDim, bias: false)
        self._vProj.wrappedValue = Linear(d, nKV * headDim, bias: false)
        self._oProj.wrappedValue = Linear(nHeads * headDim, d, bias: false)
        self._qNorm.wrappedValue = GemmaRMSNorm(dimensions: headDim, eps: args.rmsNormEps)
        self._kNorm.wrappedValue = GemmaRMSNorm(dimensions: headDim, eps: args.rmsNormEps)
        self.rope = RoPE(dimensions: args.rotaryDim, traditional: false, base: args.ropeTheta)
        self._indexer.wrappedValue = args.isSparse(layerIdx) ? MiniMaxM3Indexer(args) : nil
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))
        let off = cache?.offset ?? 0

        var q = qNorm(qProj(x).reshaped(B, L, nHeads, headDim)).transposed(0, 2, 1, 3)
        var k = kNorm(kProj(x).reshaped(B, L, nKV, headDim)).transposed(0, 2, 1, 3)
        var v = vProj(x).reshaped(B, L, nKV, headDim).transposed(0, 2, 1, 3)
        q = rope(q, offset: off)
        k = rope(k, offset: off)

        // Append K/V FIRST (upstream M3 ordering) so the indexer sees the
        // completed post-append Sk. `off` (captured pre-append) stays the correct
        // RoPE position. Manual update→SDPA (not attentionWithCacheUpdate) because
        // the indexer must run between the append and the SDPA. M3 forces TQ-KV
        // off, so the cache is always KVCacheSimple / MiniMaxM3SparseCache.
        if let cache {
            (k, v) = cache.update(keys: k, values: v)
        }

        var attnMask = mask
        if let indexer, let sparseCache = cache as? MiniMaxM3SparseCache {
            if let blockBias = indexer(x, cache: sparseCache, offset: off) {
                // 0.0/-inf are exact in bf16; SDPA requires the mask to promote to
                // the bf16 compute dtype once the indexer fires past 2048 tokens.
                attnMask = .array(blockBias.asType(q.dtype))
            }
        }

        let o = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: attnMask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)
        return oProj(o)
    }
}

// MARK: - Sparse MoE block (routed + shared experts)

private class MiniMaxM3SparseMoeBlock: Module, UnaryLayer {
    let topK: Int
    let normTopkProb: Bool
    let routedScalingFactor: Float

    @ModuleInfo(key: "gate") var gate: Linear
    @ParameterInfo(key: "e_score_correction_bias") var eScoreCorrectionBias: MLXArray
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "shared_experts") var sharedExperts: SwiGLUOAIMLP

    init(_ args: MiniMaxM3Configuration) {
        self.topK = args.numExpertsPerTok
        self.normTopkProb = args.normTopkProb
        self.routedScalingFactor = args.routedScalingFactor

        self._gate.wrappedValue = Linear(args.hiddenSize, args.numLocalExperts, bias: false)
        self._eScoreCorrectionBias.wrappedValue = MLXArray.zeros([args.numLocalExperts])
        // Routed experts use the gpt_oss clamped swiglu via the `glue` override
        // (gate, up) -> swiglu(up, gate). The Load path swaps in
        // QuantizedSwitchLinear from the checkpoint's `.scales`.
        let alpha = args.swigluAlpha
        let limit = args.swigluLimit
        self._switchMLP.wrappedValue = SwitchGLU(
            inputDims: args.hiddenSize,
            hiddenDims: args.intermediateSize,
            numExperts: args.numLocalExperts,
            glue: { gateActivation, up in
                minimaxM3Swiglu(xLinear: up, xGlu: gateActivation, alpha: alpha, limit: limit)
            }
        )
        self._sharedExperts.wrappedValue = SwiGLUOAIMLP(
            dimensions: args.hiddenSize,
            hiddenDimensions: args.sharedIntermediateSize,
            alpha: alpha, limit: limit)
    }

    /// deepseek_v3 group_expert_select with n_group = topk_group = 1 (group logic
    /// collapses): sigmoid(gate)+bias → top-k → gather original sigmoid scores →
    /// normalize → scale by routedScalingFactor.
    private func route(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let gates = gate(x)
        let origScores = sigmoid(gates.asType(.float32))
        let scoresForChoice = origScores + eScoreCorrectionBias
        let inds = argPartition(-scoresForChoice, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
        var scores = takeAlong(origScores, inds, axis: -1)
        if topK > 1, normTopkProb {
            scores = scores / (scores.sum(axis: -1, keepDims: true) + MLXArray(Float(1e-20)))
        }
        scores = scores * routedScalingFactor
        return (inds, scores)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (inds, scores) = route(x)
        let routed = (switchMLP(x, inds) * scores[.ellipsis, .newAxis]).sum(axis: -2)
        // route() returns fp32 scores → promote the residual stream; cast back to
        // x's dtype so downstream matmuls stay bf16 (matches the Python fix).
        return (routed + sharedExperts(x)).asType(x.dtype)
    }
}

// MARK: - Decoder layer

private class MiniMaxM3DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: MiniMaxM3Attention
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: GemmaRMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: GemmaRMSNorm
    fileprivate let mlp: UnaryLayer

    init(_ args: MiniMaxM3Configuration, layerIdx: Int) {
        self._selfAttn.wrappedValue = MiniMaxM3Attention(args, layerIdx: layerIdx)
        self._inputLayerNorm.wrappedValue = GemmaRMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = GemmaRMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        if args.isMoe(layerIdx) {
            self.mlp = MiniMaxM3SparseMoeBlock(args)
        } else {
            self.mlp = SwiGLUOAIMLP(
                dimensions: args.hiddenSize, hiddenDimensions: args.denseIntermediateSize,
                alpha: args.swigluAlpha, limit: args.swigluLimit)
        }
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let h = x + selfAttn(inputLayerNorm(x), mask: mask, cache: cache)
        return h + mlp(postAttentionLayerNorm(h))
    }
}

// MARK: - Model

private class MiniMaxM3ModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    fileprivate let layers: [MiniMaxM3DecoderLayer]
    @ModuleInfo(key: "norm") var norm: GemmaRMSNorm

    init(_ args: MiniMaxM3Configuration) {
        precondition(args.vocabularySize > 0)
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)
        self.layers = (0 ..< args.hiddenLayers).map { MiniMaxM3DecoderLayer(args, layerIdx: $0) }
        self._norm.wrappedValue = GemmaRMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = embedTokens(inputs)
        // Dense layers (0-2) use this causal mask; sparse layers build their own
        // block mask but fall back to it below the 2048-token threshold.
        let mask = createAttentionMask(h: h, cache: cache?.first)
        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }
        return norm(h)
    }
}

public class MiniMaxM3Model: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    fileprivate let model: MiniMaxM3ModelInner
    let configuration: MiniMaxM3Configuration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ args: MiniMaxM3Configuration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
        self.model = MiniMaxM3ModelInner(args)
        if !args.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var out = model(inputs, cache: cache)
        if let lmHead {
            out = lmHead(out)
        } else {
            out = model.embedTokens.asLinear(out)
        }
        return out
    }

    /// Sparse layers (3-59) carry the 3-lane `MiniMaxM3SparseCache`; dense layers
    /// (0-2) use a stock `KVCacheSimple`. The cache list must stay heterogeneous —
    /// downcasting the sparse entry to a plain KVCache drops `idx_keys` and loops.
    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        (0 ..< configuration.hiddenLayers).map { li -> KVCache in
            configuration.isSparse(li)
                ? MiniMaxM3SparseCache(indexDim: configuration.indexHeadDim)
                : KVCacheSimple()
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        out.reserveCapacity(weights.count)
        for (rawKey, value) in weights {
            if rawKey.hasPrefix("mtp") { continue }
            // Text-only build: drop the VL vision stack + projector.
            if rawKey.hasPrefix("vision_tower.") || rawKey.hasPrefix("multi_modal_projector.")
                || rawKey.hasPrefix("patch_merge_mlp.")
            {
                continue
            }
            var key = rawKey
            if key.hasPrefix("language_model.model.") {
                key = "model." + key.dropFirst("language_model.model.".count)
            } else if key.hasPrefix("language_model.lm_head") {
                key = "lm_head" + key.dropFirst("language_model.lm_head".count)
            }
            key = key.replacingOccurrences(of: ".block_sparse_moe.", with: ".mlp.")
            // Indexer projections are flat on self_attn in the checkpoint; the
            // model nests them under the `indexer` submodule.
            key = key.replacingOccurrences(
                of: ".self_attn.index_", with: ".self_attn.indexer.index_")
            out[key] = value
        }
        if configuration.tieWordEmbeddings {
            out["lm_head.weight"] = nil
        }
        return out
    }
}

extension MiniMaxM3Model: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}

// MARK: - Configuration

public struct MiniMaxM3Configuration: Codable, Sendable {
    var modelType: String = "minimax_m3"
    var hiddenSize: Int = 6144
    var hiddenLayers: Int = 60
    var intermediateSize: Int = 3072
    var denseIntermediateSize: Int = 12288
    var sharedIntermediateSize: Int = 3072
    var attentionHeads: Int = 64
    var kvHeads: Int = 4
    var headDim: Int = 128
    var rotaryDim: Int = 64
    var ropeTheta: Float = 5_000_000
    var rmsNormEps: Float = 1e-6
    var vocabularySize: Int = 200064
    var numLocalExperts: Int = 100
    var numExpertsPerTok: Int = 4
    var nSharedExperts: Int = 1
    var routedScalingFactor: Float = 2.0
    var normTopkProb: Bool = true
    var swigluAlpha: Float = 1.702
    var swigluLimit: Float = 7.0
    var moeLayerFreq: [Int]? = nil
    // MSA indexer
    var indexNHeads: Int = 4
    var indexHeadDim: Int = 128
    var indexBlockSize: Int = 128
    var indexTopkBlocks: Int = 16
    var indexLocalBlocks: Int = 1
    var sparseAttentionFreq: [Int]? = nil
    var tieWordEmbeddings: Bool = false

    func isMoe(_ li: Int) -> Bool {
        if let freq = moeLayerFreq { return freq[li] != 0 }
        return li >= 3
    }

    func isSparse(_ li: Int) -> Bool {
        if let freq = sparseAttentionFreq { return freq[li] != 0 }
        return li >= 3
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
        case nSharedExperts = "n_shared_experts"
        case routedScalingFactor = "routed_scaling_factor"
        case normTopkProb = "norm_topk_prob"
        case swigluAlpha = "swiglu_alpha"
        case swigluLimit = "swiglu_limit"
        case moeLayerFreq = "moe_layer_freq"
        case tieWordEmbeddings = "tie_word_embeddings"
        case sparseAttentionConfig = "sparse_attention_config"
        // Indexer fields are decoded from the nested sparse_attention_config
        // (SparseKeys); listed here only so Encodable auto-synthesizes.
        case indexNHeads = "index_n_heads"
        case indexHeadDim = "index_head_dim"
        case indexBlockSize = "index_block_size"
        case indexTopkBlocks = "index_topk_blocks"
        case indexLocalBlocks = "index_local_blocks"
        case sparseAttentionFreq = "sparse_attention_freq"
    }

    enum SparseKeys: String, CodingKey {
        case sparseNumIndexHeads = "sparse_num_index_heads"
        case sparseIndexDim = "sparse_index_dim"
        case sparseBlockSize = "sparse_block_size"
        case sparseTopkBlocks = "sparse_topk_blocks"
        case sparseLocalBlock = "sparse_local_block"
        case sparseAttentionFreq = "sparse_attention_freq"
    }

    enum OuterKeys: String, CodingKey {
        case textConfig = "text_config"
        case modelType = "model_type"
        case numLocalExperts = "num_local_experts"
    }

    // Configs are only ever decoded by the factory; encode is required for the
    // `create<C: Codable, M>` constraint but never exercised. Emit the flat fields.
    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(modelType, forKey: .modelType)
        try c.encode(hiddenSize, forKey: .hiddenSize)
        try c.encode(hiddenLayers, forKey: .hiddenLayers)
        try c.encode(intermediateSize, forKey: .intermediateSize)
        try c.encode(denseIntermediateSize, forKey: .denseIntermediateSize)
        try c.encode(sharedIntermediateSize, forKey: .sharedIntermediateSize)
        try c.encode(attentionHeads, forKey: .attentionHeads)
        try c.encode(kvHeads, forKey: .kvHeads)
        try c.encode(headDim, forKey: .headDim)
        try c.encode(rotaryDim, forKey: .rotaryDim)
        try c.encode(ropeTheta, forKey: .ropeTheta)
        try c.encode(rmsNormEps, forKey: .rmsNormEps)
        try c.encode(vocabularySize, forKey: .vocabularySize)
        try c.encode(numLocalExperts, forKey: .numLocalExperts)
        try c.encode(numExpertsPerTok, forKey: .numExpertsPerTok)
        try c.encode(nSharedExperts, forKey: .nSharedExperts)
        try c.encode(routedScalingFactor, forKey: .routedScalingFactor)
        try c.encode(normTopkProb, forKey: .normTopkProb)
        try c.encode(swigluAlpha, forKey: .swigluAlpha)
        try c.encode(swigluLimit, forKey: .swigluLimit)
        try c.encodeIfPresent(moeLayerFreq, forKey: .moeLayerFreq)
        try c.encode(tieWordEmbeddings, forKey: .tieWordEmbeddings)
        try c.encode(indexNHeads, forKey: .indexNHeads)
        try c.encode(indexHeadDim, forKey: .indexHeadDim)
        try c.encode(indexBlockSize, forKey: .indexBlockSize)
        try c.encode(indexTopkBlocks, forKey: .indexTopkBlocks)
        try c.encode(indexLocalBlocks, forKey: .indexLocalBlocks)
        try c.encodeIfPresent(sparseAttentionFreq, forKey: .sparseAttentionFreq)
    }

    public init(from decoder: Decoder) throws {
        let outer = try decoder.container(keyedBy: OuterKeys.self)
        // The JANG bundle nests real fields under `text_config`; fall back to the
        // top level for plain (non-VL) configs.
        let c: KeyedDecodingContainer<CodingKeys>
        if outer.contains(.textConfig) {
            c = try outer.nestedContainer(keyedBy: CodingKeys.self, forKey: .textConfig)
        } else {
            c = try decoder.container(keyedBy: CodingKeys.self)
        }

        func i(_ k: CodingKeys, _ d: Int) -> Int {
            ((try? c.decodeIfPresent(Int.self, forKey: k)) ?? nil) ?? d
        }
        func f(_ k: CodingKeys, _ d: Float) -> Float {
            ((try? c.decodeIfPresent(Float.self, forKey: k)) ?? nil) ?? d
        }

        self.modelType =
            (try? outer.decodeIfPresent(String.self, forKey: .modelType))
            ?? (try? c.decodeIfPresent(String.self, forKey: .modelType)) ?? "minimax_m3"
        self.hiddenSize = i(.hiddenSize, 6144)
        self.hiddenLayers = i(.hiddenLayers, 60)
        self.intermediateSize = i(.intermediateSize, 3072)
        self.denseIntermediateSize = i(.denseIntermediateSize, 12288)
        self.sharedIntermediateSize = i(.sharedIntermediateSize, 3072)
        self.attentionHeads = i(.attentionHeads, 64)
        self.kvHeads = i(.kvHeads, 4)
        self.headDim = i(.headDim, 128)
        self.rotaryDim = i(.rotaryDim, 64)
        self.ropeTheta = f(.ropeTheta, 5_000_000)
        self.rmsNormEps = f(.rmsNormEps, 1e-6)
        self.vocabularySize = i(.vocabularySize, 200064)
        // num_local_experts can sit at the top level (REAP-pruned bundles).
        self.numLocalExperts =
            (try? outer.decodeIfPresent(Int.self, forKey: .numLocalExperts))
            ?? (try? c.decodeIfPresent(Int.self, forKey: .numLocalExperts)) ?? 100
        self.numExpertsPerTok = i(.numExpertsPerTok, 4)
        self.nSharedExperts = i(.nSharedExperts, 1)
        self.routedScalingFactor = f(.routedScalingFactor, 2.0)
        self.normTopkProb = (try? c.decodeIfPresent(Bool.self, forKey: .normTopkProb)) ?? true
        self.swigluAlpha = f(.swigluAlpha, 1.702)
        self.swigluLimit = f(.swigluLimit, 7.0)
        self.moeLayerFreq = try? c.decodeIfPresent([Int].self, forKey: .moeLayerFreq)
        self.tieWordEmbeddings =
            (try? c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings)) ?? false

        // sparse_attention_config (nested under text_config)
        if let sca = try? c.nestedContainer(keyedBy: SparseKeys.self, forKey: .sparseAttentionConfig)
        {
            self.indexNHeads =
                (try? sca.decodeIfPresent(Int.self, forKey: .sparseNumIndexHeads)) ?? 4
            self.indexHeadDim = (try? sca.decodeIfPresent(Int.self, forKey: .sparseIndexDim)) ?? 128
            self.indexBlockSize =
                (try? sca.decodeIfPresent(Int.self, forKey: .sparseBlockSize)) ?? 128
            self.indexTopkBlocks =
                (try? sca.decodeIfPresent(Int.self, forKey: .sparseTopkBlocks)) ?? 16
            self.indexLocalBlocks =
                (try? sca.decodeIfPresent(Int.self, forKey: .sparseLocalBlock)) ?? 1
            self.sparseAttentionFreq = try? sca.decodeIfPresent(
                [Int].self, forKey: .sparseAttentionFreq)
        }
    }
}
