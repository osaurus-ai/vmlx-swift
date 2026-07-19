// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// ZAYA1-8B port — single model class, three MoE backends.
//
// Architecture summary:
//   - 80 decoder layers, alternating: even = CCA-attention, odd = MoE
//   - Hidden 2048, 16 query heads, 2 KV heads, head_dim 128, cca_num_q_heads 8
//   - CCA attention: linear_q (→1024), linear_k (→256), val_proj1+val_proj2
//     (concat → 256), conv_qk(2 layers, kernel 2), o_proj (1024 → 2048)
//   - CCA state per attention layer (FLOAT32): conv_state[B,1280,2], prev_hs[B,2048]
//   - MoE: 16 experts top-1, router MLP (256 hidden), MOD skip route (17th logit)
//   - Tied embeddings, rope_theta=5_000_000, partial_rotary_factor=0.5

import Foundation
import MLX
import MLXLMCommon
import MLXNN

private let zayaDebugLayerStats =
    ProcessInfo.processInfo.environment["VMLX_ZAYA_LAYER_STATS"] == "1"
private let zayaDebugLayerLimit =
    Int(ProcessInfo.processInfo.environment["VMLX_ZAYA_LAYER_LIMIT"] ?? "6") ?? 6

private func zayaPrintLayerStats(_ label: String, _ h: MLXArray) {
    guard zayaDebugLayerStats else { return }
    let last = h[0, h.dim(1) - 1, 0...].asType(.float32)
    let l2 = sqrt((last * last).sum()).item(Float.self)
    let mean = last.mean().item(Float.self)
    let centered = last - mean
    let std = sqrt((centered * centered).mean()).item(Float.self)
    let first = (0..<min(4, last.size)).map { i in
        String(format: "%.6f", last[i].item(Float.self))
    }.joined(separator: ",")
    FileHandle.standardError.write(Data(
        String(format: "[ZAYA_STATS] %@ shape=%@ last_l2=%.4f last_mean=%.6f last_std=%.6f first4=[%@]\n",
            label, "\(h.shape)", l2, mean, std, first).utf8))
}

func zayaScaledL2Normalize(_ x: MLXArray, scale: Float) -> MLXArray {
    let xf = x.asType(.float32)
    let norm = sqrt((xf * xf).sum(axis: -1, keepDims: true) + 1e-6)
    return (xf * (scale / norm)).asType(x.dtype)
}

// MARK: - Norm utilities

/// ZAYA RMSNorm mirrors the Zyphra reference: compute variance in fp32,
/// multiply by the learned weight directly, then cast back to input dtype.
public final class ZayaRMSNorm: Module, UnaryLayer {
    public let weight: MLXArray
    let eps: Float

    public init(dimensions: Int, eps: Float = 1e-6) {
        self.weight = MLXArray.ones([dimensions])
        self.eps = eps
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let xf = x.asType(.float32)
        let variance = (xf * xf).mean(axis: -1, keepDims: true)
        return ((xf * rsqrt(variance + eps)) * weight.asType(.float32)).asType(x.dtype)
    }
}

// MARK: - Configuration

public struct ZayaTextConfiguration: Codable, Sendable {
    public var modelType: String = "zaya"
    public var hiddenSize: Int = 2048
    public var numHiddenLayers: Int = 80
    public var numAttentionHeads: Int = 16
    public var numKeyValueHeads: Int = 2
    public var numQueryGroups: Int = 2
    public var ccaNumQHeads: Int = 8
    public var kvChannels: Int = 128            // head_dim
    public var numExperts: Int = 16
    public var moeRouterTopk: Int = 1
    public var maxPositionEmbeddings: Int = 131_072
    public var ropeTheta: Float = 5_000_000
    public var partialRotaryFactor: Float = 0.5
    public var vocabSize: Int = 262_272
    public var normEpsilon: Float = 1e-6
    public var ffnHiddenSize: Int = 2048
    public var tieWordEmbeddings: Bool = true
    public var scaleResidualMerge: Bool = true
    public var residualInFP32: Bool = true

    // JANGTQ / quantization knobs (filled by factory after jang_config merge).
    public var weightFormat: String?
    public var mxtqBits: Int?
    public var mxtqGateUpBits: Int?
    public var mxtqDownBits: Int?
    public var mxtqSeed: Int?
    public var zayaExpertLayout: String?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case numQueryGroups = "num_query_groups"
        case ccaNumQHeads = "cca_num_q_heads"
        case kvChannels = "kv_channels"
        case numExperts = "num_experts"
        case moeRouterTopk = "moe_router_topk"
        case maxPositionEmbeddings = "max_position_embeddings"
        case ropeTheta = "rope_theta"
        case partialRotaryFactor = "partial_rotary_factor"
        case vocabSize = "vocab_size"
        case normEpsilon = "norm_epsilon"
        case ffnHiddenSize = "ffn_hidden_size"
        case tieWordEmbeddings = "tie_word_embeddings"
        case scaleResidualMerge = "scale_residual_merge"
        case residualInFP32 = "residual_in_fp32"
        case weightFormat = "weight_format"
        case mxtqBits = "mxtq_bits"
        case mxtqGateUpBits = "mxtq_gate_up_bits"
        case mxtqDownBits = "mxtq_down_bits"
        case mxtqSeed = "mxtq_seed"
        case zayaExpertLayout = "zaya_expert_layout"
    }

    public init() {}

    private struct RoutedExpertBitMetadata {
        let routedBits: Int?
        let gateUpBits: Int?
        let downBits: Int?
    }

    /// ZAYA stores `ffn_hidden_size` as the fused `linear_fc1` output
    /// width (gate + up). SwitchGLU wants the per-branch intermediate
    /// width, so the real 8B config's 4096 maps to 2048.
    var expertIntermediateSize: Int {
        if ffnHiddenSize == hiddenSize {
            return ffnHiddenSize
        }
        if ffnHiddenSize % 2 == 0 {
            return ffnHiddenSize / 2
        }
        return ffnHiddenSize
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        // Scalar fields with defaults — decode-if-present to keep tolerant
        // of partial configs (test fixtures, future variants).
        if let v = try c.decodeIfPresent(String.self, forKey: .modelType) { self.modelType = v }
        if let v = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) { self.hiddenSize = v }
        if let v = try c.decodeIfPresent(Int.self, forKey: .numHiddenLayers) { self.numHiddenLayers = v }
        if let v = try c.decodeIfPresent(Int.self, forKey: .numAttentionHeads) { self.numAttentionHeads = v }
        if let v = try c.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) { self.numKeyValueHeads = v }
        if let v = try c.decodeIfPresent(Int.self, forKey: .numQueryGroups) { self.numQueryGroups = v }
        if let v = try c.decodeIfPresent(Int.self, forKey: .ccaNumQHeads) { self.ccaNumQHeads = v }
        if let v = try c.decodeIfPresent(Int.self, forKey: .kvChannels) { self.kvChannels = v }
        if let v = try c.decodeIfPresent(Int.self, forKey: .numExperts) { self.numExperts = v }
        if let v = try c.decodeIfPresent(Int.self, forKey: .moeRouterTopk) { self.moeRouterTopk = v }
        if let v = try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) { self.maxPositionEmbeddings = v }
        if let v = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) { self.ropeTheta = v }
        if let v = try c.decodeIfPresent(Float.self, forKey: .partialRotaryFactor) { self.partialRotaryFactor = v }
        if let v = try c.decodeIfPresent(Int.self, forKey: .vocabSize) { self.vocabSize = v }
        if let v = try c.decodeIfPresent(Float.self, forKey: .normEpsilon) { self.normEpsilon = v }
        if let v = try c.decodeIfPresent(Int.self, forKey: .ffnHiddenSize) { self.ffnHiddenSize = v }
        if let v = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) { self.tieWordEmbeddings = v }
        if let v = try c.decodeIfPresent(Bool.self, forKey: .scaleResidualMerge) { self.scaleResidualMerge = v }
        if let v = try c.decodeIfPresent(Bool.self, forKey: .residualInFP32) { self.residualInFP32 = v }

        self.weightFormat = try c.decodeIfPresent(String.self, forKey: .weightFormat)
        let routed = try Self.decodeRoutedExpertBits(from: c)
        self.mxtqBits = routed.routedBits
        self.mxtqGateUpBits =
            try c.decodeIfPresent(Int.self, forKey: .mxtqGateUpBits)
            ?? routed.gateUpBits
        self.mxtqDownBits =
            try c.decodeIfPresent(Int.self, forKey: .mxtqDownBits)
            ?? routed.downBits
        self.mxtqSeed = try c.decodeIfPresent(Int.self, forKey: .mxtqSeed)
        self.zayaExpertLayout = try c.decodeIfPresent(String.self, forKey: .zayaExpertLayout)
    }

    private static func decodeRoutedExpertBits(
        from container: KeyedDecodingContainer<CodingKeys>
    ) throws -> RoutedExpertBitMetadata {
        if let flat = try? container.decodeIfPresent(Int.self, forKey: .mxtqBits) {
            return RoutedExpertBitMetadata(routedBits: flat, gateUpBits: flat, downBits: flat)
        }
        guard container.contains(.mxtqBits) else {
            return RoutedExpertBitMetadata(routedBits: nil, gateUpBits: nil, downBits: nil)
        }
        if let dict = try? container.decode([String: Int].self, forKey: .mxtqBits) {
            // `.values.first` is per-process-random on a multi-role dict → the
            // routed bit width (hence every MoE layer's dequant) would flip on
            // reload. Deterministic: named aliases, then the MINIMUM width
            // (routed experts are the most aggressively quantized JANG tier).
            let routed = dict["routed_expert"] ?? dict["routed"] ?? dict.values.min()
            return RoutedExpertBitMetadata(
                routedBits: routed,
                gateUpBits: routed,
                downBits: routed)
        }
        struct ProjectionBits: Decodable {
            let gateProj: Int?
            let upProj: Int?
            let downProj: Int?
            enum CodingKeys: String, CodingKey {
                case gateProj = "gate_proj"
                case upProj = "up_proj"
                case downProj = "down_proj"
            }
        }
        struct NestedBits: Decodable {
            let routedExpert: ProjectionBits?
            enum CodingKeys: String, CodingKey { case routedExpert = "routed_expert" }
        }
        let nested = try container.decode(NestedBits.self, forKey: .mxtqBits)
        guard let routedExpert = nested.routedExpert else {
            throw DecodingError.dataCorrupted(.init(
                codingPath: container.codingPath + [CodingKeys.mxtqBits],
                debugDescription:
                    "mxtq_bits dictionary must include routed_expert or integer role values"
            ))
        }
        let gate = routedExpert.gateProj
        let up = routedExpert.upProj
        if let gate, let up, gate != up {
            throw DecodingError.dataCorrupted(.init(
                codingPath: container.codingPath + [CodingKeys.mxtqBits],
                debugDescription:
                    "mxtq_bits.routed_expert gate_proj and up_proj must match for fused gate/up kernels"
            ))
        }
        let gateUp = gate ?? up
        let down = routedExpert.downProj
        return RoutedExpertBitMetadata(
            routedBits: gateUp ?? down,
            gateUpBits: gateUp,
            downBits: down)
    }
}

public struct ZayaConfiguration: Codable, Sendable {
    public var modelType: String = "zaya"
    public var textConfig: ZayaTextConfiguration = ZayaTextConfiguration()

    public init() {}

    public init(from decoder: Decoder) throws {
        // ZAYA configs ship flat — no text_config wrapper. Decode the same
        // payload into both modelType and textConfig.
        let single = try ZayaTextConfiguration(from: decoder)
        self.modelType = single.modelType
        self.textConfig = single
    }

    public func encode(to encoder: Encoder) throws {
        try textConfig.encode(to: encoder)
    }
}

/// Picks the MoE backend at module-init time.
public enum ZayaMoEContext: Sendable, Equatable {
    /// JANGTQ2 / JANGTQ4 / JANGTQ_K — codebook-quantized routed experts.
    case jangtq(gateUpBits: Int, downBits: Int, seed: Int)
    /// MXFP4 — affine-4 routed experts.
    case affine(bits: Int, groupSize: Int)
    /// Base BF16 — bf16 routed experts (stack-at-load from per-expert keys).
    case bf16
}

// MARK: - Residual scale block

/// Per-layer residual merge: out = (hidden_states_scale * h + hidden_states_bias)
///                              + (residual_scale * x + residual_bias)
/// Layer 0 has only the (hidden_states_*) pair on disk — sanitize fills the
/// missing `residual_*` slots with neutral defaults (1.0 / 0.0) so the module
/// has a uniform parameter set across layers (avoids MLXNN's mismatched-array
/// container error during quantize-traversal).
public final class ZayaResScale: Module {
    @ModuleInfo(key: "hidden_states_scale") var hiddenScale: MLXArray
    @ModuleInfo(key: "hidden_states_bias") var hiddenBias: MLXArray
    @ModuleInfo(key: "residual_scale") var residualScale: MLXArray
    @ModuleInfo(key: "residual_bias") var residualBias: MLXArray

    public override init() {
        self._hiddenScale.wrappedValue = MLXArray.ones([1])
        self._hiddenBias.wrappedValue = MLXArray.zeros([1])
        self._residualScale.wrappedValue = MLXArray.ones([1])
        self._residualBias.wrappedValue = MLXArray.zeros([1])
        super.init()
    }

    public func apply(
        residual: MLXArray?, hiddenStates: MLXArray
    ) -> (residual: MLXArray?, hiddenStates: MLXArray) {
        let h = (hiddenStates + hiddenBias.asType(hiddenStates.dtype))
            * hiddenScale.asType(hiddenStates.dtype)
        guard let residual else {
            return (nil, h)
        }
        let r = (residual + residualBias.asType(residual.dtype))
            * residualScale.asType(residual.dtype)
        return (r, h)
    }
}

// MARK: - CCA-attention QKV block

final class ZayaCCAQKV: Module {
    @ModuleInfo(key: "linear_q") var linearQ: Linear
    @ModuleInfo(key: "linear_k") var linearK: Linear
    @ModuleInfo(key: "val_proj1") var valProj1: Linear
    @ModuleInfo(key: "val_proj2") var valProj2: Linear
    /// Two causal Conv1d in series. Bundle ships `conv_qk.0.{weight,bias}`
    /// and `conv_qk.1.{weight,bias}` — a [Conv1d, Conv1d] array maps cleanly.
    /// The first kernel mixes channels (in_channels=1, out=1280), the second
    /// mixes within the head_dim window (in_channels=128, out=1280).
    @ModuleInfo(key: "conv_qk") var convQK: [Conv1d]
    @ModuleInfo(key: "temp") var temp: MLXArray

    let qDim: Int
    let kDim: Int
    let headDim: Int
    let convChannels: Int

    init(_ cfg: ZayaTextConfiguration) {
        let H = cfg.hiddenSize
        let qDim = cfg.ccaNumQHeads * cfg.kvChannels
        let kDim = cfg.numQueryGroups * cfg.kvChannels
        self.qDim = qDim
        self.kDim = kDim
        self.headDim = cfg.kvChannels
        self.convChannels = qDim + kDim

        self._linearQ.wrappedValue = Linear(H, qDim, bias: false)
        self._linearK.wrappedValue = Linear(H, kDim, bias: false)
        self._valProj1.wrappedValue = Linear(H, cfg.kvChannels, bias: false)
        self._valProj2.wrappedValue = Linear(H, cfg.kvChannels, bias: false)
        // conv_qk[0] is depthwise: groups=convChannels (per-channel convolution).
        // Bundle weight shape `[1280, 1, 2]` decodes as (out=1280, in/groups=1, k=2)
        // in PyTorch convention; MLX-Swift expects `[out, k, in/groups]` so the
        // weights are transposed in sanitize.
        //
        // conv_qk[1] is head-grouped: groups=convChannels/headDim (per "head").
        // Bundle weight `[1280, 128, 2]` → MLX shape `[1280, 2, 128]`.
        self._convQK.wrappedValue = [
            Conv1d(
                inputChannels: convChannels,
                outputChannels: convChannels,
                kernelSize: 2,
                groups: convChannels,
                bias: true),
            Conv1d(
                inputChannels: convChannels,
                outputChannels: convChannels,
                kernelSize: 2,
                groups: convChannels / cfg.kvChannels,
                bias: true),
        ]
        self._temp.wrappedValue = MLXArray.zeros([2])
        super.init()
    }
}

// MARK: - Sub-layer protocol

/// Each ZAYA decoder layer holds either a CCA attention block (even layers)
/// or an MoE block (odd layers). Both expose `forwardSubLayer(_:cache:)` so
/// the parent decoder can call them uniformly via a non-optional @ModuleInfo
/// — matching the pattern used by `BailingDecoderLayer.attention: any BailingAttention`.
/// Without this uniformity MLXNN's update path fails with `mismatchedContainers`
/// when the parameter tree contains different per-layer sub-layer keys.
protocol ZayaSubLayer: Module {
    func forwardSubLayer(
        _ x: MLXArray, cache: KVCache?, routerState: MLXArray?
    ) -> (output: MLXArray, routerState: MLXArray?)
}

// MARK: - CCA-attention layer

final class ZayaCCAAttention: Module, ZayaSubLayer {
    @ModuleInfo(key: "qkv") var qkv: ZayaCCAQKV
    @ModuleInfo(key: "o_proj") var oProj: Linear

    let qHeads: Int
    let kvHeads: Int
    let headDim: Int
    let qDim: Int
    let kDim: Int
    let convChannels: Int
    let ropeDim: Int
    let ropeTheta: Float
    let scale: Float
    let hiddenSize: Int
    let layerIndex: Int
    let rope: RoPE

    init(_ cfg: ZayaTextConfiguration, layerIndex: Int) {
        self.layerIndex = layerIndex
        self.qHeads = cfg.ccaNumQHeads
        self.kvHeads = cfg.numQueryGroups
        self.headDim = cfg.kvChannels
        self.qDim = qHeads * headDim
        self.kDim = kvHeads * headDim
        self.convChannels = qDim + kDim
        self.ropeDim = Int((Float(headDim) * cfg.partialRotaryFactor).rounded(.toNearestOrEven))
        self.ropeTheta = cfg.ropeTheta
        self.scale = 1.0 / Float(headDim).squareRoot()
        self.hiddenSize = cfg.hiddenSize

        self._qkv.wrappedValue = ZayaCCAQKV(cfg)
        self._oProj.wrappedValue = Linear(qDim, cfg.hiddenSize, bias: false)
        self.rope = RoPE(dimensions: ropeDim, traditional: false, base: ropeTheta)
        super.init()
    }

    /// Adapt the heterogeneous cache type passed by the decoder layer.
    func forwardSubLayer(
        _ x: MLXArray, cache: KVCache?, routerState: MLXArray?
    ) -> (output: MLXArray, routerState: MLXArray?) {
        (
            callAsFunction(x,
            cache: cache as? ZayaCCACache,
            batchCache: cache as? BatchZayaCCACache),
            nil
        )
    }

    /// Mirrors Zyphra's CCA reference:
    /// - build q/k input frames, plus q/k mean residuals, before any RoPE
    /// - run the two causal conv_qk kernels over q/k input frames
    /// - update conv_state with the last two q/k input frames, not conv output
    /// - use previous normalized hidden state for val_proj2
    /// - L2-normalize q/k per head, apply temp to keys, then apply RoPE
    /// - update ordinary KV cache and attention as a standard GQA block
    func callAsFunction(
        _ x: MLXArray,
        cache: ZayaCCACache?,
        batchCache: BatchZayaCCACache?
    ) -> MLXArray {
        let B = x.dim(0)
        let T = x.dim(1)

        // Q/K projections in feature form [B,T,C]. CCA conv state stores these
        // pre-conv input frames, exactly like ZayaDynamicCache.conv_states.
        let qSeq = qkv.linearQ(x)                                           // [B,T,1024]
        let kSeq = qkv.linearK(x)                                           // [B,T,256]
        let qkInput = concatenated([qSeq, kSeq], axis: -1)                  // [B,T,1280]
        let qkInputCT = qkInput.transposed(0, 2, 1)                         // [B,1280,T]
        if zayaDebugLayerStats, layerIndex < zayaDebugLayerLimit {
            zayaPrintLayerStats("CCA\(layerIndex) qSeq", qSeq)
            zayaPrintLayerStats("CCA\(layerIndex) kSeq", kSeq)
        }

        // Mean residuals from the raw q/k projections.
        let repeatN = qHeads / kvHeads
        let queryPre = qSeq.reshaped([B, T, qHeads, headDim])               // [B,T,8,128]
        let keyPre = kSeq.reshaped([B, T, kvHeads, headDim])                // [B,T,2,128]
        let keyPreRep = repeated(keyPre, count: repeatN, axis: 2)           // [B,T,8,128]
        let qkMeanQ = (queryPre + keyPreRep) / MLXArray(2.0, dtype: queryPre.dtype)
        let qkMeanK = qkMeanQ
            .reshaped([B, T, kvHeads, repeatN, headDim])
            .sum(axis: 3) / MLXArray(Float(repeatN), dtype: qkMeanQ.dtype) // [B,T,2,128]

        // Read prior CCA state (gathered for batched B>1).
        let priorConv: MLXArray
        let priorPrev: MLXArray
        if let bc = batchCache {
            let gathered = bc.gatherCCA()
            priorConv = gathered.conv
            priorPrev = gathered.prev
        } else if let c = cache {
            let state = c.readCCA()
            priorConv = state.conv
            priorPrev = state.prev
        } else {
            priorConv = MLXArray.zeros([B, convChannels, 2], dtype: .float32)
            priorPrev = MLXArray.zeros([B, hiddenSize], dtype: .float32)
        }
        let hasPriorCCA = (cache?.offset ?? batchCache?.offset ?? 0) > 0
        let qkAug = hasPriorCCA
            ? concatenated([priorConv.asType(qkInputCT.dtype), qkInputCT], axis: -1)
            : concatenated([
                MLXArray.zeros([B, convChannels, 2], dtype: qkInputCT.dtype),
                qkInputCT,
            ], axis: -1)                                               // [B,1280,T+2]

        // MLX Conv1d expects [N,L,C]. ZAYA's reference is an nn.Sequential
        // of two Conv1d layers with no activation between them.
        let c0in = qkAug.transposed(0, 2, 1)
        let c0out = qkv.convQK[0](c0in)
        let qkPostFeat = qkv.convQK[1](c0out)                               // [B,T,1280]
        if zayaDebugLayerStats, layerIndex < zayaDebugLayerLimit {
            zayaPrintLayerStats("CCA\(layerIndex) qkPost", qkPostFeat)
        }

        // New conv state = last two q/k input frames. For prefill T>2 this
        // crops to the final two tokens; for decode it rolls prior+current.
        let inputStateLen = qkAug.dim(-1)
        let newConv: MLXArray = {
            if inputStateLen >= 2 {
                return qkAug[0..., 0..., (inputStateLen - 2)..<inputStateLen].asType(.float32)
            }
            let pad = MLXArray.zeros([B, convChannels, 2 - inputStateLen], dtype: qkAug.dtype)
            return concatenated([pad, qkAug], axis: -1).asType(.float32)
        }()

        // Split conv output back into q/k and add the q/k mean residual.
        var qOut = qkPostFeat[0..., 0..., 0..<qDim]
            .reshaped([B, T, qHeads, headDim]) + qkMeanQ                 // [B,T,8,128]
        var kOut = qkPostFeat[0..., 0..., qDim..<(qDim + kDim)]
            .reshaped([B, T, kvHeads, headDim]) + qkMeanK                // [B,T,2,128]

        // Values: v1 uses current normalized stream; v2 uses a one-token
        // delayed stream seeded from prev_hs when the cache is already warm.
        let hsDelayed: MLXArray = {
            if hasPriorCCA {
                let first = priorPrev.asType(x.dtype).expandedDimensions(axis: 1) // [B,1,H]
                if T == 1 {
                    return first
                }
                return concatenated([first, x[0..., 0..<(T - 1), 0...]], axis: 1)
            }
            if T == 1 {
                return MLXArray.zeros([B, 1, hiddenSize], dtype: x.dtype)
            }
            return concatenated([
                MLXArray.zeros([B, 1, hiddenSize], dtype: x.dtype),
                x[0..., 0..<(T - 1), 0...],
            ], axis: 1)
        }()
        let v = concatenated([qkv.valProj1(x), qkv.valProj2(hsDelayed)], axis: -1)
            .reshaped([B, T, kvHeads, headDim])
            .transposed(0, 2, 1, 3)                                      // [B,2,T,128]

        // L2-normalize q/k per head, scale by sqrt(head_dim), apply per-KV
        // temperature to keys, then apply partial RoPE.
        let sqrtHead = Float(headDim).squareRoot()
        qOut = zayaScaledL2Normalize(qOut, scale: sqrtHead)
        kOut = zayaScaledL2Normalize(kOut, scale: sqrtHead)
        kOut = kOut * qkv.temp.asType(kOut.dtype).reshaped([1, 1, kvHeads, 1])
        if zayaDebugLayerStats, layerIndex < zayaDebugLayerLimit {
            zayaPrintLayerStats("CCA\(layerIndex) qNorm",
                qOut.reshaped([B, T, qDim]))
            zayaPrintLayerStats("CCA\(layerIndex) kNormTemp",
                kOut.reshaped([B, T, kDim]))
        }

        let qForRoPE = qOut.transposed(0, 2, 1, 3)
        let kForRoPE = kOut.transposed(0, 2, 1, 3)
        let qRot: MLXArray
        let kRot: MLXArray
        if let bc = batchCache {
            qRot = rope(qForRoPE, offset: bc.offsetArray)                 // [B,8,T,128]
            kRot = rope(kForRoPE, offset: bc.offsetArray)                 // [B,2,T,128]
        } else {
            let offset = cache?.offset ?? 0
            qRot = rope(qForRoPE, offset: offset)                         // [B,8,T,128]
            kRot = rope(kForRoPE, offset: offset)                         // [B,2,T,128]
        }

        // Build the mask before update so batched uneven-length slots use
        // pre-step offsets. BatchZayaCCACache.update then returns K/V padded
        // to exactly the same max(offset_i + T) length.
        let mask: MLXFast.ScaledDotProductAttentionMaskMode
        if let bc = batchCache {
            mask = bc.makeMask(n: T, windowSize: nil, returnArray: false)
        } else if let c = cache {
            mask = c.makeMask(n: T, windowSize: nil, returnArray: false)
        } else if T > 1 {
            mask = .causal
        } else {
            mask = .none
        }

        // KV cache update (per-slot in batched mode via BatchZayaCCACache).
        var (kFull, vFull) = (kRot, v)
        if let bc = batchCache {
            (kFull, vFull) = bc.update(keys: kRot, values: v)
        } else if let c = cache {
            (kFull, vFull) = c.update(keys: kRot, values: v)
        }

        // Repeat KV to match query heads.
        let kRep = repeated(kFull, count: repeatN, axis: 1)
        let vRep = repeated(vFull, count: repeatN, axis: 1)

        let attn = MLXFast.scaledDotProductAttention(
            queries: qRot, keys: kRep, values: vRep, scale: scale, mask: mask)
        let attnFlat = attn.transposed(0, 2, 1, 3).reshaped([B, T, qDim])
        let out = oProj(attnFlat)
        if zayaDebugLayerStats, layerIndex < zayaDebugLayerLimit {
            zayaPrintLayerStats("CCA\(layerIndex) attnFlat", attnFlat)
            zayaPrintLayerStats("CCA\(layerIndex) out", out)
        }

        // New prev_hs = last normalized hidden input, used by val_proj2 on
        // the next warm pass/decode token.
        let newPrev = x[0..., (T - 1)..<T, 0...]
                        .reshaped([B, hiddenSize])
                        .asType(.float32)

        if let bc = batchCache {
            bc.scatterCCA(conv: newConv, prev: newPrev)
        } else if let c = cache {
            c.writeCCA(conv: newConv, prev: newPrev)
        }
        return out
    }
}

// MARK: - MoE layer

public final class ZayaRouter: Module {
    @ModuleInfo(key: "rmsnorm_eda") var edaNorm: ZayaRMSNorm
    /// Bundle ships `router_mlp.{0,1,2}.{weight,bias}` after sanitize
    /// compresses the original Sequential indices `{0,2,4}` (with
    /// implicit ReLUs at 1, 3) to a 3-Linear tuple. This avoids the
    /// homogeneous-array bias reuse that can incorrectly apply a 256-wide
    /// bias to the final 17-logit projection.
    @ModuleInfo(key: "router_mlp") var routerMLP: (Linear, Linear, Linear)
    @ModuleInfo(key: "down_proj") var downProj: Linear
    @ModuleInfo(key: "balancing_biases") var balancingBiases: MLXArray
    /// Per-channel EDA carry scale applied to the previous MoE router state.
    /// Bundle ships this for 39/40 MoE layers — the very first MoE block
    /// (layer 1) has no previous MoE state and omits it. Sanitize fills the
    /// missing slot with a neutral 1.0 scale of shape [routerHidden] so the
    /// parameter tree is uniform across all MoE layers.
    @ModuleInfo(key: "router_states_scale") var routerStatesScale: MLXArray

    let numExperts: Int
    let routerHidden: Int

    public init(_ cfg: ZayaTextConfiguration) {
        self.numExperts = cfg.numExperts
        let H = cfg.hiddenSize
        let R = 256  // router hidden dim (canonical to ZAYA1; not in config)
        self.routerHidden = R
        self._edaNorm.wrappedValue = ZayaRMSNorm(dimensions: R, eps: cfg.normEpsilon)
        self._routerMLP.wrappedValue = (
            Linear(R, R, bias: true),
            Linear(R, R, bias: true),
            Linear(R, cfg.numExperts + 1, bias: true)
        )
        self._downProj.wrappedValue = Linear(H, R, bias: true)
        self._balancingBiases.wrappedValue = MLXArray.zeros([cfg.numExperts + 1])
        // Default neutral scale (router_hidden width) — sanitize replaces
        // with the bundle value when present.
        self._routerStatesScale.wrappedValue = MLXArray.ones([R])
        super.init()
    }

    /// Returns `(expertIdx, weight, activeMask, nextRouterState)`.
    ///
    /// ZAYA's router first projects the 2048-wide residual stream into the
    /// 256-wide router space, adds the previous MoE router state through EDA,
    /// then runs RMSNorm + a 3-layer MLP. The final logit is the MOD/skip
    /// route; when it wins, the expert contribution is zero and the layer
    /// proceeds through the residual path.
    public func route(
        _ x: MLXArray, previousRouterState: MLXArray?
    ) -> (idx: MLXArray, weights: MLXArray, activeMask: MLXArray, nextRouterState: MLXArray) {
        let projected = downProj(x)                                      // [B*T, 256]
        let routerState: MLXArray
        if let previousRouterState {
            routerState = projected
                + previousRouterState.asType(projected.dtype)
                    * routerStatesScale.asType(projected.dtype)
        } else {
            routerState = projected
        }

        let normed = edaNorm(routerState)
        let r0 = gelu(routerMLP.0(normed))
        let r1 = gelu(routerMLP.1(r0))
        let logits = routerMLP.2(r1)                                     // [B*T, E+1]

        let probs = softmax(logits, axis: -1)                            // [B*T, E+1]
        // Balancing biases affect choice only, not the gathered route prob.
        let biased = probs.asType(.float32) + balancingBiases.asType(.float32)
        let idxAll = argMax(biased, axis: -1)                             // [B*T]
        let weights = takeAlong(probs, idxAll.expandedDimensions(axis: -1), axis: -1)
                        .squeezed(axis: -1)                               // [B*T]

        let activeMask = (idxAll .< Int32(numExperts)).asType(probs.dtype)
        let idx = minimum(idxAll, MLXArray(Int32(numExperts - 1))).asType(.uint32)
        return (idx, weights, activeMask, routerState)
    }
}

/// Polymorphic switch primitive — JANGTQ uses TurboQuantSwitchGLU,
/// MXFP4/BF16 use the standard SwitchGLU.
public protocol ZayaSwitchPrimitive: Module {
    func callAsFunction(_ x: MLXArray, _ indices: MLXArray) -> MLXArray
}
extension SwitchGLU: ZayaSwitchPrimitive {}
extension TurboQuantSwitchGLU: ZayaSwitchPrimitive {}

public final class ZayaExperts: Module {
    @ModuleInfo(key: "switch_mlp") var switchMLP: ZayaSwitchPrimitive

    public init(_ cfg: ZayaTextConfiguration, context: ZayaMoEContext?, layerIdx: Int? = nil) {
        let H = cfg.hiddenSize
        let I = cfg.expertIntermediateSize
        let E = cfg.numExperts
        switch context {
        case .some(.jangtq(let gateUp, let down, let seed)):
            if JANGTQStreamingExperts.isEnabled, let layerIdx {
                self._switchMLP.wrappedValue = StreamingTurboQuantSwitchGLU(
                    inputDims: H, hiddenDims: I, numExperts: E,
                    gateUpBits: gateUp, downBits: down, seed: seed,
                    layerIdx: layerIdx)
            } else {
                self._switchMLP.wrappedValue = TurboQuantSwitchGLU(
                    inputDims: H, hiddenDims: I, numExperts: E,
                    gateUpBits: gateUp, downBits: down, seed: seed)
            }
        case .some(.affine), .some(.bf16), nil:
            self._switchMLP.wrappedValue = SwitchGLU(
                inputDims: H, hiddenDims: I, numExperts: E)
        }
        super.init()
    }

    public func callAsFunction(_ x: MLXArray, _ indices: MLXArray) -> MLXArray {
        switchMLP(x, indices)
    }
}

final class ZayaMoEBlock: Module, ZayaSubLayer {
    @ModuleInfo(key: "router") var router: ZayaRouter
    @ModuleInfo(key: "experts") var experts: ZayaExperts

    let hiddenSize: Int

    init(_ cfg: ZayaTextConfiguration, context: ZayaMoEContext?, layerIdx: Int? = nil) {
        self.hiddenSize = cfg.hiddenSize
        self._router.wrappedValue = ZayaRouter(cfg)
        self._experts.wrappedValue = ZayaExperts(cfg, context: context, layerIdx: layerIdx)
        super.init()
    }

    func forwardSubLayer(
        _ x: MLXArray, cache: KVCache?, routerState: MLXArray?
    ) -> (output: MLXArray, routerState: MLXArray?) {
        callAsFunction(x, previousRouterState: routerState)
    }

    /// Given normed input `nx` (the input_norm output of the layer),
    /// runs the router + experts and returns the additive contribution
    /// from the routed expert. The 17th MOD route is a learned skip path:
    /// the reference uses the normalized hidden state itself as the expert
    /// output for that route, then multiplies by the selected route prob.
    func callAsFunction(
        _ nx: MLXArray, previousRouterState: MLXArray?
    ) -> (output: MLXArray, routerState: MLXArray) {
        let B = nx.dim(0)
        let T = nx.dim(1)
        let xFlat = nx.reshaped([B * T, hiddenSize])

        let (idx, weights, activeMask, nextRouterState) = router.route(
            xFlat, previousRouterState: previousRouterState)
        // SwitchGLU / TurboQuantSwitchGLU expect (x, indices).
        let xIn = xFlat.reshaped([B, T, hiddenSize])
        let idx2D = idx.reshaped([B, T, 1])                       // [B,T,K=1]
        let expertOut = experts.switchMLP(xIn, idx2D)             // [B,T,K,H] or [B,T,H]
        let expertFlat: MLXArray
        if expertOut.ndim == 4 {
            // Sum across the K=1 axis.
            expertFlat = expertOut.sum(axis: 2).reshaped([B * T, hiddenSize])
        } else {
            expertFlat = expertOut.reshaped([B * T, hiddenSize])
        }
        let routeWeights = weights.reshaped([B * T, 1]).asType(expertFlat.dtype)
        let active = activeMask.reshaped([B * T, 1]).asType(expertFlat.dtype)
        let modPassThrough = xFlat.asType(expertFlat.dtype)
        let selectedOutput = expertFlat * active + modPassThrough * (1 - active)
        let weighted = selectedOutput * routeWeights
        return (weighted.reshaped([B, T, hiddenSize]), nextRouterState)
    }
}

// MARK: - Decoder layer

final class ZayaDecoderLayer: Module {
    @ModuleInfo(key: "input_norm") var inputNorm: ZayaRMSNorm
    @ModuleInfo(key: "res_scale") var resScale: ZayaResScale
    /// One non-optional sub-layer per Bailing's pattern. The bundle key is
    /// canonicalized to `sub` in sanitize (was `self_attn` for even layers
    /// and `zaya_block` for odd). MLXNN's update path needs uniform keys
    /// across array elements, so this is the structural fix that lets
    /// quantize traverse the layers array without mismatchedContainers.
    @ModuleInfo(key: "sub") var sub: any ZayaSubLayer

    let isAttention: Bool
    let layerIndex: Int
    let scaleResidualMerge: Bool
    let residualInFP32: Bool

    init(_ cfg: ZayaTextConfiguration, layerIdx: Int, context: ZayaMoEContext?) {
        self.isAttention = (layerIdx % 2 == 0)
        self.layerIndex = layerIdx
        self.scaleResidualMerge = cfg.scaleResidualMerge
        self.residualInFP32 = cfg.residualInFP32
        self._inputNorm.wrappedValue = ZayaRMSNorm(dimensions: cfg.hiddenSize, eps: cfg.normEpsilon)
        self._resScale.wrappedValue = ZayaResScale()
        if isAttention {
            self._sub.wrappedValue = ZayaCCAAttention(cfg, layerIndex: layerIdx)
        } else {
            self._sub.wrappedValue = ZayaMoEBlock(cfg, context: context, layerIdx: layerIdx)
        }
        super.init()
    }

    func callAsFunction(
        _ hiddenStates: MLXArray, residual priorResidual: MLXArray?,
        cache: KVCache?, routerState: MLXArray?
    ) -> (hiddenStates: MLXArray, residual: MLXArray, routerState: MLXArray?) {
        let scaled: (residual: MLXArray?, hiddenStates: MLXArray)
        if scaleResidualMerge {
            scaled = resScale.apply(residual: priorResidual, hiddenStates: hiddenStates)
        } else {
            scaled = (residual: priorResidual, hiddenStates: hiddenStates)
        }
        let residual: MLXArray
        if let r = scaled.residual {
            residual = scaled.hiddenStates + r
        } else if residualInFP32 {
            residual = scaled.hiddenStates.asType(.float32)
        } else {
            residual = scaled.hiddenStates
        }
        let nx = inputNorm(residual.asType(inputNorm.weight.dtype))
        if zayaDebugLayerStats, layerIndex < zayaDebugLayerLimit {
            zayaPrintLayerStats("NX layer\(layerIndex)", nx)
        }
        let (h, nextRouterState) = sub.forwardSubLayer(
            nx, cache: cache, routerState: routerState)
        return (h, residual, nextRouterState)
    }
}

// MARK: - Inner trunk

public final class ZayaModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    fileprivate let layers: [ZayaDecoderLayer]
    @ModuleInfo(key: "final_norm") var finalNorm: ZayaRMSNorm
    @ModuleInfo(key: "res_scale") var resScale: ZayaResScale

    let numHiddenLayers: Int
    let scaleResidualMerge: Bool
    let residualInFP32: Bool

    init(_ cfg: ZayaTextConfiguration, context: ZayaMoEContext?) {
        self.numHiddenLayers = cfg.numHiddenLayers
        self.scaleResidualMerge = cfg.scaleResidualMerge
        self.residualInFP32 = cfg.residualInFP32
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: cfg.vocabSize, dimensions: cfg.hiddenSize)
        self.layers = (0 ..< cfg.numHiddenLayers).map { l in
            ZayaDecoderLayer(cfg, layerIdx: l, context: context)
        }
        self._finalNorm.wrappedValue = ZayaRMSNorm(dimensions: cfg.hiddenSize, eps: cfg.normEpsilon)
        self._resScale.wrappedValue = ZayaResScale()
        super.init()
    }

    func callAsFunction(
        _ inputs: MLXArray,
        cache: [KVCache]?,
        inputEmbedding: MLXArray? = nil
    ) -> MLXArray {
        let embed = inputEmbedding ?? embedTokens(inputs)
        var h = embed
        zayaPrintLayerStats("HS 0 embed", h)
        var residual: MLXArray?
        var routerState: MLXArray?
        for (i, layer) in layers.enumerated() {
            let result = layer(
                h, residual: residual, cache: cache?[i], routerState: routerState)
            h = result.hiddenStates
            residual = result.residual
            if let next = result.routerState {
                routerState = next
            }
            if zayaDebugLayerStats, i + 1 <= zayaDebugLayerLimit {
                zayaPrintLayerStats("HS \(i + 1) layer\(i)", h)
            }
        }

        let scaled: (residual: MLXArray?, hiddenStates: MLXArray)
        if scaleResidualMerge {
            scaled = resScale.apply(residual: residual, hiddenStates: h)
        } else {
            scaled = (residual: residual, hiddenStates: h)
        }
        let finalResidual: MLXArray
        if let r = scaled.residual {
            finalResidual = scaled.hiddenStates + r
        } else if residualInFP32 {
            finalResidual = scaled.hiddenStates.asType(.float32)
        } else {
            finalResidual = scaled.hiddenStates
        }
        let out = finalNorm(finalResidual.asType(finalNorm.weight.dtype))
        zayaPrintLayerStats("HS final", out)
        return out
    }

    /// Fill array gaps in `layers` before delegating to MLXNN's standard
    /// update path. ZAYA's MoE layers have no affine-quantized linears —
    /// the routed experts are TurboQuant codebook and the router is fp16
    /// passthrough — so the load-time `quantize` updates dict only
    /// contains entries for even (CCA) layers. Without this fill, MLXNN's
    /// array recursion sees `.none` values at odd indices and fails with
    /// `mismatchedContainers`.
    public override func update(
        modules: ModuleChildren, verify: VerifyUpdate,
        path: [String] = [], modulePath: [String] = []
    ) throws -> Self {
        var modules = modules
        if let layersItem = modules["layers"], case .array(let arr) = layersItem {
            var filled = arr
            for i in filled.indices {
                if case .none = filled[i] {
                    filled[i] = .dictionary([:])
                }
            }
            modules["layers"] = .array(filled)
        }
        return try super.update(
            modules: modules, verify: verify, path: path, modulePath: modulePath)
    }
}

// MARK: - Top-level model

public final class ZayaModel: Module, LLMModel, KVCacheDimensionProvider {
    @ModuleInfo(key: "model") public var model: ZayaModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public let configuration: ZayaConfiguration
    public let context: ZayaMoEContext?
    public let kvHeads: [Int]
    public var vocabularySize: Int { configuration.textConfig.vocabSize }

    public init(_ configuration: ZayaConfiguration, moe context: ZayaMoEContext?) {
        self.configuration = configuration
        self.context = context
        let cfg = configuration.textConfig
        self._model.wrappedValue = ZayaModelInner(cfg, context: context)
        if !cfg.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(cfg.hiddenSize, cfg.vocabSize, bias: false)
        }
        self.kvHeads = (0 ..< cfg.numHiddenLayers).map { _ in cfg.numQueryGroups }
        super.init()
    }

    public func callAsFunction(
        _ inputs: MLXArray,
        cache: [KVCache]?,
        inputEmbedding: MLXArray? = nil
    ) -> MLXArray {
        let h = model(inputs, cache: cache, inputEmbedding: inputEmbedding)
        if let lmHead {
            return lmHead(h)
        }
        return model.embedTokens.asLinear(h)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        callAsFunction(inputs, cache: cache, inputEmbedding: nil)
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        let cfg = configuration.textConfig
        let convChannels = cfg.ccaNumQHeads * cfg.kvChannels + cfg.numQueryGroups * cfg.kvChannels
        return (0 ..< cfg.numHiddenLayers).map { l in
            if l % 2 == 0 {
                return ZayaCCACache(
                    batchSize: 1,
                    convChannels: convChannels,
                    hiddenSize: cfg.hiddenSize)
            } else {
                // Explicit no-op for MoE-only layers. Using KVCacheSimple here
                // made generic TQ conversion compress unused placeholders
                // while leaving the real KV inside ZayaCCACache untouched.
                return ZayaMoEPlaceholderCache()
            }
        }
    }

    /// Rewrite bundle keys to the module's hierarchy and stack BF16 per-expert
    /// weights when present. JANGTQ + MXFP4 already ship pre-stacked.
    public func sanitize(weights w: [String: MLXArray]) -> [String: MLXArray] {
        var weights = w

        // Tied embeddings — drop lm_head.weight so MLX doesn't try to bind it
        // to a non-existent module (lmHead is nil under tieWordEmbeddings=true).
        if configuration.textConfig.tieWordEmbeddings {
            weights["lm_head.weight"] = nil
            weights["lm_head.scales"] = nil
            weights["lm_head.biases"] = nil
        }

        // Strip per-tensor .tq_bits hints — metadata, not module parameters.
        for k in Array(weights.keys) where k.hasSuffix(".tq_bits") {
            weights[k] = nil
        }

        // Conv1d weights ship in PyTorch order `[out, in/groups, kernel]` but
        // MLX-Swift's Conv1d expects `[out, kernel, in/groups]`. Swap axes 1↔2
        // for any conv_qk weight (3-D tensor whose last dim isn't already
        // the kernel size). Mirrors the conv1d permute in Qwen35JANGTQ.sanitize.
        for k in Array(weights.keys) where k.contains(".conv_qk.") && k.hasSuffix(".weight") {
            guard let w = weights[k], w.ndim == 3 else { continue }
            // PyTorch shape [out, in/groups, kernel=2] → MLX [out, kernel, in/groups]
            if w.dim(-1) != 2 { continue }   // already in MLX order or unexpected layout
            weights[k] = w.movedAxis(source: 2, destination: 1)
        }

        // Canonicalize the per-layer sub-block key. The bundle ships
        // `model.layers.{L}.self_attn.*` for even (CCA) layers and
        // `model.layers.{L}.zaya_block.*` for odd (MoE) layers. The Swift
        // module declares a single `sub` field per layer (Bailing pattern)
        // so MLXNN's array-update path sees uniform keys across all 80
        // layers and doesn't fail with mismatchedContainers.
        for k in Array(weights.keys) {
            for needle in [".self_attn.", ".zaya_block."] {
                if let r = k.range(of: needle) {
                    let newKey = k.replacingCharacters(in: r, with: ".sub.")
                    weights[newKey] = weights[k]
                    weights[k] = nil
                    break
                }
            }
        }

        // Layer 0's res_scale on disk has only hidden_states_{scale,bias} —
        // fill in neutral residual_{scale,bias} (1.0 / 0.0) so the module
        // structure is uniform across all 80 layers (avoids the same
        // mismatchedContainers issue at the res_scale child level).
        for layer in 0 ..< configuration.textConfig.numHiddenLayers {
            let prefix = "model.layers.\(layer).res_scale"
            if weights["\(prefix).residual_scale"] == nil {
                weights["\(prefix).residual_scale"] = MLXArray.ones([1], dtype: .float32)
            }
            if weights["\(prefix).residual_bias"] == nil {
                weights["\(prefix).residual_bias"] = MLXArray.zeros([1], dtype: .float32)
            }
        }

        // Layer 1's router has no `router_states_scale` in the bundle
        // (39/40 MoE layers ship it; first MoE block omits it). Fill the
        // missing slot with a neutral [routerHidden]=256 1.0 vector so all
        // MoE routers have uniform parameter shapes.
        let routerHidden = 256  // canonical to ZAYA1
        for layer in stride(from: 1, to: configuration.textConfig.numHiddenLayers, by: 2) {
            let key = "model.layers.\(layer).sub.router.router_states_scale"
            if weights[key] == nil {
                weights[key] = MLXArray.ones([routerHidden], dtype: .float32)
            }
        }

        // Compress router_mlp.{0,2,4} -> router_mlp.{0,1,2}. The bundle
        // ships a Sequential(Linear, ReLU, Linear, ReLU, Linear); the Swift
        // module declares 3 Linears in a tuple with explicit activations.
        // Do this as a two-phase rewrite so `.4 -> .2` cannot overwrite the
        // source `.2` tensor before it is copied to `.1`.
        var routerDeletes: [String] = []
        var routerRewrites: [(String, MLXArray)] = []
        for k in Array(weights.keys) {
            guard k.contains(".router_mlp."), let value = weights[k] else { continue }
            let renames: [(String, String)] = [
                (".router_mlp.2.", ".router_mlp.1."),
                (".router_mlp.4.", ".router_mlp.2."),
            ]
            for (from, to) in renames where k.contains(from) {
                routerDeletes.append(k)
                routerRewrites.append((k.replacingOccurrences(of: from, with: to), value))
                break
            }
        }
        for k in routerDeletes {
            weights[k] = nil
        }
        for (k, value) in routerRewrites {
            weights[k] = value
        }

        // Source ZAYA has bias on router_mlp.0/2 but not on router_mlp.4
        // (after compression: 0/1 but not 2). Fill a zero final bias so the
        // Swift Linear preserves source math while keeping uniform keys.
        for layer in stride(from: 1, to: configuration.textConfig.numHiddenLayers, by: 2) {
            let key = "model.layers.\(layer).sub.router.router_mlp.2.bias"
            if weights[key] == nil {
                weights[key] = MLXArray.zeros(
                    [configuration.textConfig.numExperts + 1], dtype: .float32)
            }
        }

        // Per-MoE-layer layout sniff. Note: by this point sub-block keys
        // have already been canonicalized from `.zaya_block.*` to `.sub.*`.
        let cfg = configuration.textConfig
        let H = cfg.hiddenSize
        let E = cfg.numExperts
        for layer in stride(from: 1, to: cfg.numHiddenLayers, by: 2) {
            let prefix = "model.layers.\(layer).sub.experts"
            let stackedTQProbe = "\(prefix).switch_mlp.gate_proj.tq_packed"
            let stackedAffineProbe = "\(prefix).switch_mlp.gate_proj.weight"
            let perExpertProbe = "\(prefix).local_experts.0.linear_fc1.weight"

            if weights[stackedTQProbe] != nil { continue }
            if weights[stackedAffineProbe] != nil { continue }

            guard weights[perExpertProbe] != nil else { continue }

            // Stack per-expert BF16 weights.
            // linear_fc1 = [2H, H], split rows: [:H] = gate_proj, [H:] = up_proj.
            // linear_fc2 = [H, H] = down_proj.
            var gates: [MLXArray] = []
            var ups: [MLXArray] = []
            var downs: [MLXArray] = []
            gates.reserveCapacity(E); ups.reserveCapacity(E); downs.reserveCapacity(E)
            for e in 0 ..< E {
                let fc1Key = "\(prefix).local_experts.\(e).linear_fc1.weight"
                let fc2Key = "\(prefix).local_experts.\(e).linear_fc2.weight"
                guard let fc1 = weights.removeValue(forKey: fc1Key),
                      let fc2 = weights.removeValue(forKey: fc2Key) else { continue }
                gates.append(fc1[0..<H, 0...])
                ups.append(fc1[H..<(2 * H), 0...])
                downs.append(fc2)
            }
            weights["\(prefix).switch_mlp.gate_proj.weight"] = loadTimeMaterializedStacked(gates)
            weights["\(prefix).switch_mlp.up_proj.weight"] = loadTimeMaterializedStacked(ups)
            weights["\(prefix).switch_mlp.down_proj.weight"] = loadTimeMaterializedStacked(downs)
        }

        return weights
    }
}

extension ZayaModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers as [Module]
    }
}
