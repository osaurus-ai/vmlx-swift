// Copyright © 2026 Osaurus.

// OpenPangu 2.0 Flash (`openpangu_v2` / OpenPanguV2ForCausalLM).
//
// Architecture (reverse-engineered from the weight graph — the modeling source
// is native transformers-5.0 and not public; see OPENPANGU-V2-PORT-STATUS.md):
//   • DeepSeek-style MLA (q/kv low-rank, qk_nope 128 + qk_rope 64, v 128) with
//     three extra causal depthwise convs (qa_conv / compresskv_conv / o_conv,
//     kernel 3, stateful) and 128 learned attention sinks per layer.
//   • Per-layer hybrid attention: the 16 `dsaLayers` are full-attention with a
//     lightning indexer (top-`indexTopk`); the `swaLayers` are sliding-window
//     (`slidingWindowList`).
//   • MHC = Hyper-Connections with `mhcNumStream` residual streams
//     (attn_mhc / mlp_mhc per layer + a global merge_mhc).
//   • Sandwich norm (input/post-attn/pre-mlp/post-mlp + block_post on 9 layers).
//   • DeepSeek-V3 MoE: 256 routed + 1 shared, first `firstKDenseReplace` dense,
//     biased top-k routing (`eScoreCorrectionBias`, `routedScalingFactor`).
//   • MTP depth `numNextnPredictLayers` (layers 46–48).

import Foundation
import MLX

public struct OpenPanguV2Configuration: Codable, Sendable {
    // Core dims
    public var vocabSize: Int = 151552
    public var hiddenSize: Int = 2560
    public var intermediateSize: Int = 9216
    public var moeIntermediateSize: Int = 1024
    public var numHiddenLayers: Int = 46
    public var numAttentionHeads: Int = 48
    public var numKeyValueHeads: Int = 48
    public var maxPositionEmbeddings: Int = 524288
    public var rmsNormEps: Float = 1e-5
    public var hiddenAct: String = "silu"
    public var tieWordEmbeddings: Bool = false

    // MLA
    public var qLoraRank: Int = 1024
    public var kvLoraRank: Int = 512
    public var qkNopeHeadDim: Int = 128
    public var qkRopeHeadDim: Int = 64
    public var vHeadDim: Int = 128
    public var ropeTheta: Float = 6_400_000
    public var ropeInterleave: Bool = false
    /// Learned attention sinks prepended to the KV (`param_sink_*`). 0 disables.
    public var paramSinkNumber: Int = 128

    // Hybrid attention: DSA (full + indexer) vs SWA (sliding window)
    public var dsaLayers: [Int] = []
    public var swaLayers: [Int] = []
    public var slidingWindow: Int = 512
    /// Per-SWA-layer window (falls back to `slidingWindow`). 512×30 then 2048×3.
    public var slidingWindowList: [Int] = []
    public var routerSlidingWindow: Int = 3

    // DSA lightning indexer (only on `dsaLayers`)
    public var indexHeadDim: Int = 128
    public var indexNHeads: Int = 24
    public var indexTopk: Int = 2048

    // MoE
    public var nRoutedExperts: Int = 256
    public var nSharedExperts: Int = 1
    public var numExpertsPerTok: Int = 8
    public var firstKDenseReplace: Int = 2
    public var normTopkProb: Bool = true
    public var routedScalingFactor: Float = 2.5
    public var routerEnableExpertBias: Bool = true

    // MHC (Hyper-Connections)
    public var useMhc: Bool = true
    public var mhcNumStream: Int = 4
    public var mhcRecurNorm: Int = 20
    public var mhcUseGamma: Bool = true
    public var useMome: Bool = true

    // Sandwich norm
    public var sandwichNorm: Bool = true
    public var blockPostLayernormIdx: [Int] = []

    // MTP
    public var numNextnPredictLayers: Int = 3

    // Derived helpers
    /// qk_head_dim = nope + rope (192).
    public var qkHeadDim: Int { qkNopeHeadDim + qkRopeHeadDim }
    /// Per-layer attention kind.
    public func isSlidingLayer(_ i: Int) -> Bool { swaLayers.contains(i) }
    public func isDSALayer(_ i: Int) -> Bool { !swaLayers.contains(i) }
    /// Effective sliding window for SWA layer `i` (index into the SWA sublist).
    public func slidingWindowFor(_ i: Int) -> Int {
        // sliding_window_list is indexed by position among SWA layers.
        guard !slidingWindowList.isEmpty else { return slidingWindow }
        let swaOrdinal = swaLayers.prefix(while: { $0 < i }).count
        return swaOrdinal < slidingWindowList.count
            ? slidingWindowList[swaOrdinal] : slidingWindowList.last!
    }
    public var isMoELayer: (Int) -> Bool { { $0 >= self.firstKDenseReplace } }

    enum CodingKeys: String, CodingKey {
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case hiddenAct = "hidden_act"
        case tieWordEmbeddings = "tie_word_embeddings"
        case qLoraRank = "q_lora_rank"
        case kvLoraRank = "kv_lora_rank"
        case qkNopeHeadDim = "qk_nope_head_dim"
        case qkRopeHeadDim = "qk_rope_head_dim"
        case vHeadDim = "v_head_dim"
        case ropeTheta = "rope_theta"
        case ropeInterleave = "rope_interleave"
        case paramSinkNumber = "param_sink_number"
        case dsaLayers = "dsa_layers"
        case swaLayers = "swa_layers"
        case slidingWindow = "sliding_window"
        case slidingWindowList = "sliding_window_list"
        case routerSlidingWindow = "router_sliding_window"
        case indexHeadDim = "index_head_dim"
        case indexNHeads = "index_n_heads"
        case indexTopk = "index_topk"
        case nRoutedExperts = "n_routed_experts"
        case nSharedExperts = "n_shared_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case firstKDenseReplace = "first_k_dense_replace"
        case normTopkProb = "norm_topk_prob"
        case routedScalingFactor = "routed_scaling_factor"
        case routerEnableExpertBias = "router_enable_expert_bias"
        case useMhc = "use_mhc"
        case mhcNumStream = "mhc_num_stream"
        case mhcRecurNorm = "mhc_recur_norm"
        case mhcUseGamma = "mhc_use_gamma"
        case useMome = "use_mome"
        case sandwichNorm = "sandwich_norm"
        case blockPostLayernormIdx = "block_post_layernorm_idx"
        case numNextnPredictLayers = "num_nextn_predict_layers"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        func d<T: Decodable>(_ k: CodingKeys, _ dflt: T) throws -> T {
            try c.decodeIfPresent(T.self, forKey: k) ?? dflt
        }
        vocabSize = try d(.vocabSize, vocabSize)
        hiddenSize = try d(.hiddenSize, hiddenSize)
        intermediateSize = try d(.intermediateSize, intermediateSize)
        moeIntermediateSize = try d(.moeIntermediateSize, moeIntermediateSize)
        numHiddenLayers = try d(.numHiddenLayers, numHiddenLayers)
        numAttentionHeads = try d(.numAttentionHeads, numAttentionHeads)
        numKeyValueHeads = try d(.numKeyValueHeads, numAttentionHeads)
        maxPositionEmbeddings = try d(.maxPositionEmbeddings, maxPositionEmbeddings)
        rmsNormEps = try d(.rmsNormEps, rmsNormEps)
        hiddenAct = try d(.hiddenAct, hiddenAct)
        tieWordEmbeddings = try d(.tieWordEmbeddings, tieWordEmbeddings)
        qLoraRank = try d(.qLoraRank, qLoraRank)
        kvLoraRank = try d(.kvLoraRank, kvLoraRank)
        qkNopeHeadDim = try d(.qkNopeHeadDim, qkNopeHeadDim)
        qkRopeHeadDim = try d(.qkRopeHeadDim, qkRopeHeadDim)
        vHeadDim = try d(.vHeadDim, vHeadDim)
        ropeTheta = try d(.ropeTheta, ropeTheta)
        ropeInterleave = try d(.ropeInterleave, ropeInterleave)
        paramSinkNumber = try d(.paramSinkNumber, paramSinkNumber)
        dsaLayers = try d(.dsaLayers, dsaLayers)
        swaLayers = try d(.swaLayers, swaLayers)
        slidingWindow = try d(.slidingWindow, slidingWindow)
        slidingWindowList = try d(.slidingWindowList, slidingWindowList)
        routerSlidingWindow = try d(.routerSlidingWindow, routerSlidingWindow)
        indexHeadDim = try d(.indexHeadDim, indexHeadDim)
        indexNHeads = try d(.indexNHeads, indexNHeads)
        indexTopk = try d(.indexTopk, indexTopk)
        nRoutedExperts = try d(.nRoutedExperts, nRoutedExperts)
        nSharedExperts = try d(.nSharedExperts, nSharedExperts)
        numExpertsPerTok = try d(.numExpertsPerTok, numExpertsPerTok)
        firstKDenseReplace = try d(.firstKDenseReplace, firstKDenseReplace)
        normTopkProb = try d(.normTopkProb, normTopkProb)
        routedScalingFactor = try d(.routedScalingFactor, routedScalingFactor)
        routerEnableExpertBias = try d(.routerEnableExpertBias, routerEnableExpertBias)
        useMhc = try d(.useMhc, useMhc)
        mhcNumStream = try d(.mhcNumStream, mhcNumStream)
        mhcRecurNorm = try d(.mhcRecurNorm, mhcRecurNorm)
        mhcUseGamma = try d(.mhcUseGamma, mhcUseGamma)
        useMome = try d(.useMome, useMome)
        sandwichNorm = try d(.sandwichNorm, sandwichNorm)
        blockPostLayernormIdx = try d(.blockPostLayernormIdx, blockPostLayernormIdx)
        numNextnPredictLayers = try d(.numNextnPredictLayers, numNextnPredictLayers)
    }
}
