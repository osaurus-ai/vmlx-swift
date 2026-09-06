// Copyright © 2026 Apple Inc.

// BailingMoeV3 — the Ling 3.0 family (e.g. Ling-3.0-tiny).
//
// Same `model_type` string ("bailing_hybrid") as Ling 2.6, but a DIFFERENT
// architecture (`architectures = ["BailingMoeV3ForCausalLM"]`):
//   * Linear layers are KDA (Kimi Delta Attention): per-stream short causal
//     convolutions + a delta-rule recurrence with a PER-CHANNEL decay gate —
//     not the Lightning/GLA recurrence `BailingHybrid.swift` implements.
//   * Full-attention layers are MLA with interleaved RoPE and an optional
//     head-wise sigmoid output gate (`g_proj`).
//   * MoE is DeepSeek-V3-shaped: sigmoid scores, `expert_bias` added only for
//     ROUTING (original scores are what get normalized), group-limited top-k,
//     `routed_scaling_factor`, plus shared experts.
// Layer rule: `(idx + 1) % layer_group_size == 0` (and any trailing partial
// group) is MLA; everything else is KDA. Dense MLP below
// `first_k_dense_replace`, MoE from there on.
//
// Reference: `modeling_bailing_moe_v3.py` shipped inside the upstream bundle
// (inclusionAI/Ling-3.0-tiny) plus fla's `fused_recurrent_kda` for the exact
// recurrence (see KDADelta.swift).

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Configuration

public struct BailingMoeV3Configuration: Codable, Sendable {
    var hiddenSize: Int
    var numHiddenLayers: Int
    var intermediateSize: Int
    var numAttentionHeads: Int
    var numKeyValueHeads: Int
    var vocabSize: Int
    var rmsNormEps: Float
    var ropeTheta: Float
    var tieWordEmbeddings: Bool

    // Layer typing
    var layerGroupSize: Int
    var firstKDenseReplace: Int

    // KDA
    var headDim: Int
    var shortConvKernelSize: Int
    var noKdaLora: Bool
    var kdaSafeGate: Bool
    var kdaLowerBound: Float

    // MLA
    var qLoraRank: Int?
    var kvLoraRank: Int
    var qkRopeHeadDim: Int
    var qkNopeHeadDim: Int
    var vHeadDim: Int
    var ropeInterleave: Bool
    var useQkvBias: Bool
    var ropeScaling: [String: StringOrNumber]?
    var gatedAttentionProjGranularityType: String?

    // MoE
    var numExperts: Int?
    var numExpertsPerTok: Int
    var numSharedExperts: Int
    var moeIntermediateSize: Int
    var moeSharedExpertIntermediateSize: Int
    var nGroup: Int
    var topkGroup: Int
    var routedScalingFactor: Float
    var normTopkProb: Bool
    var scoreFunction: String

    var qkHeadDim: Int { qkNopeHeadDim + qkRopeHeadDim }

    /// Same rule as the reference `BailingMoeV3DecoderLayer`: the last layer
    /// of each complete group is full attention, and every layer past the
    /// last complete group is full attention too.
    func isFullAttentionLayer(_ idx: Int) -> Bool {
        (idx + 1) % layerGroupSize == 0
            || idx >= (numHiddenLayers / layerGroupSize) * layerGroupSize
    }

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case vocabSize = "vocab_size"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case tieWordEmbeddings = "tie_word_embeddings"
        case layerGroupSize = "layer_group_size"
        case firstKDenseReplace = "first_k_dense_replace"
        case headDim = "head_dim"
        case shortConvKernelSize = "short_conv_kernel_size"
        case noKdaLora = "no_kda_lora"
        case kdaSafeGate = "kda_safe_gate"
        case kdaLowerBound = "kda_lower_bound"
        case qLoraRank = "q_lora_rank"
        case kvLoraRank = "kv_lora_rank"
        case qkRopeHeadDim = "qk_rope_head_dim"
        case qkNopeHeadDim = "qk_nope_head_dim"
        case vHeadDim = "v_head_dim"
        case ropeInterleave = "rope_interleave"
        case useQkvBias = "use_qkv_bias"
        case ropeScaling = "rope_scaling"
        case gatedAttentionProjGranularityType = "gated_attention_proj_granularity_type"
        case numExperts = "num_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case numSharedExperts = "num_shared_experts"
        case moeIntermediateSize = "moe_intermediate_size"
        case moeSharedExpertIntermediateSize = "moe_shared_expert_intermediate_size"
        case nGroup = "n_group"
        case topkGroup = "topk_group"
        case routedScalingFactor = "routed_scaling_factor"
        case normTopkProb = "norm_topk_prob"
        case scoreFunction = "score_function"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        numHiddenLayers = try c.decode(Int.self, forKey: .numHiddenLayers)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        numAttentionHeads = try c.decode(Int.self, forKey: .numAttentionHeads)
        numKeyValueHeads =
            try c.decodeIfPresent(Int.self, forKey: .numKeyValueHeads)
            ?? numAttentionHeads
        vocabSize = try c.decode(Int.self, forKey: .vocabSize)
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10000
        tieWordEmbeddings =
            try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        layerGroupSize = try c.decodeIfPresent(Int.self, forKey: .layerGroupSize) ?? 4
        firstKDenseReplace =
            try c.decodeIfPresent(Int.self, forKey: .firstKDenseReplace) ?? 0
        headDim =
            try c.decodeIfPresent(Int.self, forKey: .headDim)
            ?? hiddenSize / numAttentionHeads
        shortConvKernelSize =
            try c.decodeIfPresent(Int.self, forKey: .shortConvKernelSize) ?? 4
        noKdaLora = try c.decodeIfPresent(Bool.self, forKey: .noKdaLora) ?? true
        kdaSafeGate = try c.decodeIfPresent(Bool.self, forKey: .kdaSafeGate) ?? true
        kdaLowerBound =
            try c.decodeIfPresent(Float.self, forKey: .kdaLowerBound) ?? -5
        qLoraRank = try c.decodeIfPresent(Int.self, forKey: .qLoraRank)
        kvLoraRank = try c.decodeIfPresent(Int.self, forKey: .kvLoraRank) ?? 512
        qkRopeHeadDim = try c.decodeIfPresent(Int.self, forKey: .qkRopeHeadDim) ?? 64
        qkNopeHeadDim = try c.decodeIfPresent(Int.self, forKey: .qkNopeHeadDim) ?? 128
        vHeadDim = try c.decodeIfPresent(Int.self, forKey: .vHeadDim) ?? 128
        ropeInterleave = try c.decodeIfPresent(Bool.self, forKey: .ropeInterleave) ?? true
        useQkvBias = try c.decodeIfPresent(Bool.self, forKey: .useQkvBias) ?? false
        ropeScaling = try c.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeScaling)
        gatedAttentionProjGranularityType = try c.decodeIfPresent(
            String.self, forKey: .gatedAttentionProjGranularityType)
        numExperts = try c.decodeIfPresent(Int.self, forKey: .numExperts)
        numExpertsPerTok =
            try c.decodeIfPresent(Int.self, forKey: .numExpertsPerTok) ?? 8
        numSharedExperts =
            try c.decodeIfPresent(Int.self, forKey: .numSharedExperts) ?? 0
        moeIntermediateSize =
            try c.decodeIfPresent(Int.self, forKey: .moeIntermediateSize) ?? 512
        moeSharedExpertIntermediateSize =
            try c.decodeIfPresent(Int.self, forKey: .moeSharedExpertIntermediateSize)
            ?? moeIntermediateSize
        nGroup = try c.decodeIfPresent(Int.self, forKey: .nGroup) ?? 1
        topkGroup = try c.decodeIfPresent(Int.self, forKey: .topkGroup) ?? 1
        routedScalingFactor =
            try c.decodeIfPresent(Float.self, forKey: .routedScalingFactor) ?? 1
        normTopkProb = try c.decodeIfPresent(Bool.self, forKey: .normTopkProb) ?? true
        scoreFunction =
            try c.decodeIfPresent(String.self, forKey: .scoreFunction) ?? "sigmoid"
    }
}

// MARK: - Norms

/// RMSNorm whose gate goes through SIGMOID — Ling 3.0's `o_norm` is fla's
/// `FusedRMSNormGated(activation='sigmoid')`, unlike Qwen3Next's silu gate.
public final class BailingV3RMSNormGated: Module {
    let weight: MLXArray
    let eps: Float

    public init(dimensions: Int, eps: Float = 1e-6) {
        self.weight = MLXArray.ones([dimensions])
        self.eps = eps
        super.init()
    }

    public func callAsFunction(_ x: MLXArray, gate: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: weight, eps: eps) * sigmoid(gate.asType(.float32))
            .asType(x.dtype)
    }
}

// MARK: - KDA linear attention

public final class BailingV3KDAAttention: Module {
    let numHeads: Int
    let headDim: Int
    let projectionSize: Int
    let convKernelSize: Int
    let safeGate: Bool
    let lowerBound: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear

    @ModuleInfo(key: "q_conv1d") var qConv: Conv1d
    @ModuleInfo(key: "k_conv1d") var kConv: Conv1d
    @ModuleInfo(key: "v_conv1d") var vConv: Conv1d

    @ParameterInfo(key: "A_log") var aLog: MLXArray
    @ParameterInfo(key: "dt_bias") var dtBias: MLXArray

    @ModuleInfo(key: "f_proj") var fProj: Linear?
    @ModuleInfo(key: "f_a_proj") var fAProj: Linear?
    @ModuleInfo(key: "f_b_proj") var fBProj: Linear?
    @ModuleInfo(key: "b_proj") var bProj: Linear
    @ModuleInfo(key: "g_proj") var gProj: Linear?
    @ModuleInfo(key: "g_a_proj") var gAProj: Linear?
    @ModuleInfo(key: "g_b_proj") var gBProj: Linear?

    @ModuleInfo(key: "o_norm") var oNorm: BailingV3RMSNormGated
    @ModuleInfo(key: "o_proj") var oProj: Linear

    /// Convenience for Ling 3.0 / BailingMoeV3, which is where this module started.
    convenience init(_ args: BailingMoeV3Configuration) {
        self.init(
            hiddenSize: args.hiddenSize, numHeads: args.numAttentionHeads,
            headDim: args.headDim, convKernelSize: args.shortConvKernelSize,
            safeGate: args.kdaSafeGate, lowerBound: args.kdaLowerBound,
            useLoRAGates: !args.noKdaLora, rmsNormEps: args.rmsNormEps)
    }

    /// The designated initialiser, in explicit dimensions rather than one family's configuration
    /// type — so a second family can build the same module without importing the first one's
    /// config. GLM-5.3 (`glm5_next`) uses it: its `linear_attn_config` names exactly these
    /// quantities (`num_heads`, `head_dim`, `short_conv_kernel_size`, `gate_lower_bound`) and its
    /// weights carry the same parameter names, down to `f_a_proj` / `g_b_proj` / `o_norm`.
    public init(
        hiddenSize: Int,
        numHeads: Int,
        headDim: Int,
        convKernelSize: Int,
        safeGate: Bool,
        lowerBound: Float,
        useLoRAGates: Bool,
        rmsNormEps: Float
    ) {
        self.numHeads = numHeads
        self.headDim = headDim
        self.projectionSize = numHeads * headDim
        self.convKernelSize = convKernelSize
        self.safeGate = safeGate
        self.lowerBound = lowerBound

        _qProj.wrappedValue = Linear(hiddenSize, projectionSize, bias: false)
        _kProj.wrappedValue = Linear(hiddenSize, projectionSize, bias: false)
        _vProj.wrappedValue = Linear(hiddenSize, projectionSize, bias: false)

        let makeConv = { (channels: Int, kernel: Int) -> Conv1d in
            Conv1d(
                inputChannels: channels,
                outputChannels: channels,
                kernelSize: kernel,
                stride: 1, padding: 0, dilation: 1,
                groups: channels, bias: false)
        }
        _qConv.wrappedValue = makeConv(projectionSize, convKernelSize)
        _kConv.wrappedValue = makeConv(projectionSize, convKernelSize)
        _vConv.wrappedValue = makeConv(projectionSize, convKernelSize)

        _aLog.wrappedValue = MLXArray.zeros([numHeads])
        _dtBias.wrappedValue = MLXArray.zeros([projectionSize])

        if useLoRAGates {
            _fAProj.wrappedValue = Linear(hiddenSize, headDim, bias: false)
            _fBProj.wrappedValue = Linear(headDim, projectionSize, bias: false)
            _gAProj.wrappedValue = Linear(hiddenSize, headDim, bias: false)
            _gBProj.wrappedValue = Linear(headDim, projectionSize, bias: false)
        } else {
            _fProj.wrappedValue = Linear(hiddenSize, projectionSize, bias: false)
            _gProj.wrappedValue = Linear(hiddenSize, projectionSize, bias: false)
        }
        _bProj.wrappedValue = Linear(hiddenSize, numHeads, bias: false)

        _oNorm.wrappedValue = BailingV3RMSNormGated(dimensions: headDim, eps: rmsNormEps)
        _oProj.wrappedValue = Linear(projectionSize, hiddenSize, bias: false)
        super.init()
    }

    public func callAsFunction(
        _ x: MLXArray, mask: MLXArray? = nil, cache: MambaCache? = nil
    ) -> MLXArray {
        let B = x.dim(0)
        let T = x.dim(1)
        let dtype = x.dtype
        let proj = projectionSize

        var mixed = concatenated([qProj(x), kProj(x), vProj(x)], axis: -1)
        if let mask {
            mixed = MLX.where(
                expandedDimensions(mask, axis: -1), mixed, MLXArray.zeros(like: mixed))
        }

        let convState: MLXArray
        if let s = cache?[0] {
            convState = s
        } else {
            convState = MLXArray.zeros(
                [B, convKernelSize - 1, 3 * proj], dtype: dtype)
        }
        let convInput = concatenated([convState, mixed], axis: 1)
        if let cache {
            cache[0] = convInput[0..., (1 - convKernelSize)..., 0...]
        }

        var q = silu(qConv(convInput[0..., 0..., 0 ..< proj]))
        var k = silu(kConv(convInput[0..., 0..., proj ..< (2 * proj)]))
        let v = silu(vConv(convInput[0..., 0..., (2 * proj) ..< (3 * proj)]))
            .reshaped(B, T, numHeads, headDim)

        q = q.reshaped(B, T, numHeads, headDim)
        k = k.reshaped(B, T, numHeads, headDim)

        // fla applies true per-head L2 norm (eps 1e-6 on the SUM of squares)
        // and scales q by Dk^-0.5. An unweighted rmsNorm is l2norm * sqrt(D)
        // — but its eps sits on the MEAN of squares, so the equivalent
        // stabiliser is 1e-6 / D (upstream mlx-lm `_normalize_kda_qk` uses
        // exactly that). A bare 1e-6 here was D× too large.
        let invScale = pow(Float(headDim), -0.5)
        let l2Eps = Float(1e-6) / Float(headDim)
        q = MLXArray(invScale * invScale, dtype: q.dtype)
            * MLXFast.rmsNorm(q, weight: MLXArray.mlxNone, eps: l2Eps)
        k = MLXArray(invScale, dtype: k.dtype)
            * MLXFast.rmsNorm(k, weight: MLXArray.mlxNone, eps: l2Eps)

        let fRaw: MLXArray
        if let fProj {
            fRaw = fProj(x).reshaped(B, T, numHeads, headDim)
        } else {
            fRaw = fBProj!(fAProj!(x)).reshaped(B, T, numHeads, headDim)
        }
        let beta = sigmoid(bProj(x).asType(.float32))

        let (y, newState) = kdaUpdate(
            q: q, k: k, v: v, fRaw: fRaw, beta: beta,
            aLog: aLog, dtBias: dtBias,
            safeGate: safeGate, lowerBound: lowerBound,
            state: cache?[1], mask: mask)
        if let cache {
            cache[1] = newState
            // The disk-L2 eligibility gate requires every layer's offset to
            // advance in step; a KDA layer stuck at 0 silently vetoed every
            // store for the whole model (zero kv_v2 entries, measured live).
            // Same contract as Qwen35's GatedDeltaNet (`cache.offset += S`).
            cache.offset += T
        }

        let gate: MLXArray
        if let gProj {
            gate = gProj(x).reshaped(B, T, numHeads, headDim)
        } else {
            gate = gBProj!(gAProj!(x)).reshaped(B, T, numHeads, headDim)
        }
        let out = oNorm(y.asType(dtype), gate: gate)
        return oProj(out.reshaped(B, T, -1))
    }
}

// MARK: - MLA full attention

final class BailingV3MLAAttention: Module {
    let numHeads: Int
    let qLoraRank: Int?
    let qkRopeHeadDim: Int
    let kvLoraRank: Int
    let vHeadDim: Int
    let qkNopeHeadDim: Int
    let qHeadDim: Int
    let gateGranularity: String?
    var scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear?
    @ModuleInfo(key: "q_a_proj") var qAProj: Linear?
    @ModuleInfo(key: "q_a_layernorm") var qALayerNorm: RMSNorm?
    @ModuleInfo(key: "q_b_proj") var qBProj: Linear?
    @ModuleInfo(key: "kv_a_proj_with_mqa") var kvAProjWithMqa: Linear
    @ModuleInfo(key: "kv_a_layernorm") var kvALayerNorm: RMSNorm
    @ModuleInfo(key: "kv_b_proj") var kvBProj: Linear
    @ModuleInfo(key: "g_proj") var gProj: Linear?
    @ModuleInfo(key: "dense") var dense: Linear

    let rope: RoPE

    init(_ args: BailingMoeV3Configuration) {
        self.numHeads = args.numAttentionHeads
        self.qLoraRank = args.qLoraRank
        self.qkRopeHeadDim = args.qkRopeHeadDim
        self.kvLoraRank = args.kvLoraRank
        self.vHeadDim = args.vHeadDim
        self.qkNopeHeadDim = args.qkNopeHeadDim
        self.qHeadDim = args.qkHeadDim
        self.gateGranularity = args.gatedAttentionProjGranularityType
        self.scale = pow(Float(qHeadDim), -0.5)

        if let q = args.qLoraRank {
            _qAProj.wrappedValue = Linear(args.hiddenSize, q, bias: args.useQkvBias)
            _qALayerNorm.wrappedValue = RMSNorm(dimensions: q, eps: args.rmsNormEps)
            _qBProj.wrappedValue = Linear(q, numHeads * qHeadDim, bias: false)
        } else {
            _qProj.wrappedValue = Linear(
                args.hiddenSize, numHeads * qHeadDim, bias: false)
        }
        _kvAProjWithMqa.wrappedValue = Linear(
            args.hiddenSize, kvLoraRank + qkRopeHeadDim, bias: args.useQkvBias)
        _kvALayerNorm.wrappedValue = RMSNorm(
            dimensions: kvLoraRank, eps: args.rmsNormEps)
        _kvBProj.wrappedValue = Linear(
            kvLoraRank, numHeads * (qkNopeHeadDim + vHeadDim), bias: false)
        switch gateGranularity {
        case "head_wise":
            _gProj.wrappedValue = Linear(args.hiddenSize, numHeads, bias: false)
        case "element_wise":
            _gProj.wrappedValue = Linear(
                args.hiddenSize, numHeads * vHeadDim, bias: false)
        default:
            break
        }
        _dense.wrappedValue = Linear(
            numHeads * vHeadDim, args.hiddenSize, bias: args.useQkvBias)

        // YaRN mscale correction, matching the reference (rope_scaling is
        // nil on Ling-3.0-tiny so this is a no-op there).
        if let ropeScaling = args.ropeScaling {
            let mScaleAllDim = ropeScaling["mscale_all_dim"]?.asFloat() ?? 0.0
            let scalingFactor = ropeScaling["factor"]?.asFloat() ?? 1.0
            if mScaleAllDim != 0, scalingFactor > 1 {
                let s = 0.1 * mScaleAllDim * log(scalingFactor) + 1.0
                self.scale = self.scale * s * s
            }
        }

        // rope_interleave = true → traditional (interleaved-pair) RoPE.
        self.rope = RoPE(
            dimensions: qkRopeHeadDim, traditional: args.ropeInterleave,
            base: args.ropeTheta)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)

        var q: MLXArray
        if qLoraRank == nil {
            q = qProj!(x)
        } else {
            q = qBProj!(qALayerNorm!(qAProj!(x)))
        }
        q = q.reshaped(B, L, numHeads, qHeadDim).transposed(0, 2, 1, 3)
        let qSplit = split(q, indices: [qkNopeHeadDim], axis: -1)
        let qNope = qSplit[0]
        var qPe = qSplit[1]

        var compressedKv = kvAProjWithMqa(x)
        let kvSplit = split(compressedKv, indices: [kvLoraRank], axis: -1)
        compressedKv = kvSplit[0]
        var kPe = kvSplit[1]
        kPe = kPe.reshaped(B, L, 1, qkRopeHeadDim).transposed(0, 2, 1, 3)

        var kv = kvBProj(kvALayerNorm(compressedKv))
        kv = kv.reshaped(B, L, numHeads, qkNopeHeadDim + vHeadDim)
            .transposed(0, 2, 1, 3)
        let kvPartSplit = split(kv, indices: [qkNopeHeadDim], axis: -1)
        let kNope = kvPartSplit[0]
        var values = kvPartSplit[1]

        qPe = applyRotaryPosition(rope, to: qPe, cache: cache)
        kPe = applyRotaryPosition(rope, to: kPe, cache: cache)
        kPe = repeated(kPe, count: numHeads, axis: 1)

        let queries = concatenated([qNope, qPe], axis: -1)
        var keys = concatenated([kNope, kPe], axis: -1)

        if let cache {
            (keys, values) = cache.update(keys: keys, values: values)
        }

        var output = mlaScaledDotProductAttention(
            queries: queries, keys: keys, values: values,
            scale: scale, mask: mask
        )
        .transposed(0, 2, 1, 3)  // [B, L, H, Dv]

        if let gProj {
            let gate = sigmoid(gProj(x).asType(.float32)).asType(output.dtype)
            if gateGranularity == "head_wise" {
                output = output * gate.reshaped(B, L, numHeads, 1)
            } else {
                output = output * gate.reshaped(B, L, numHeads, vHeadDim)
            }
        }
        return dense(output.reshaped(B, L, -1))
    }
}

// MARK: - MLP / MoE

final class BailingV3MLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(_ args: BailingMoeV3Configuration, intermediateSize: Int? = nil) {
        let inter = intermediateSize ?? args.intermediateSize
        _gateProj.wrappedValue = Linear(args.hiddenSize, inter, bias: false)
        _upProj.wrappedValue = Linear(args.hiddenSize, inter, bias: false)
        _downProj.wrappedValue = Linear(inter, args.hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

final class BailingV3MoEGate: Module {
    let topK: Int
    let nGroup: Int
    let topkGroup: Int
    let routedScalingFactor: Float
    let normTopkProb: Bool
    let scoreFunction: String

    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "expert_bias") var expertBias: MLXArray

    init(_ args: BailingMoeV3Configuration) {
        guard let numExperts = args.numExperts else {
            fatalError("BailingV3MoEGate requires num_experts")
        }
        self.topK = args.numExpertsPerTok
        self.nGroup = args.nGroup
        self.topkGroup = args.topkGroup
        self.routedScalingFactor = args.routedScalingFactor
        self.normTopkProb = args.normTopkProb
        self.scoreFunction = args.scoreFunction
        _weight.wrappedValue = MLXArray.zeros([numExperts, args.hiddenSize])
        _expertBias.wrappedValue = MLXArray.zeros([numExperts])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        // Router in float32, as `router_dtype = "fp32"` demands.
        let logits = x.asType(.float32).matmul(weight.asType(.float32).T)
        let scores =
            scoreFunction == "sigmoid"
            ? sigmoid(logits) : softmax(logits, axis: -1, precise: true)

        // expert_bias participates in SELECTION only; the returned weights
        // come from the un-biased scores (reference lines 398-404).
        var selection = scores + expertBias.asType(.float32)

        if nGroup > 1 {
            // Group score = sum of the top-2 experts in the group; keep the
            // best `topk_group` groups, drop the rest to -inf so a negative
            // expert_bias cannot resurrect a dropped group.
            var grouped = unflatten(selection, axis: -1, shape: [nGroup, -1])
            let groupScores = top(grouped, k: 2, axis: -1).sum(axis: -1, keepDims: true)
            let dropCount = nGroup - topkGroup
            let dropIdx = argPartition(groupScores, kth: dropCount - 1, axis: -2)[
                .ellipsis, ..<dropCount, 0...]
            grouped = putAlong(
                grouped, stopGradient(dropIdx),
                values: MLXArray(-Float.infinity, dtype: grouped.dtype), axis: -2)
            selection = flattened(grouped, start: -2, end: -1)
        }

        let inds = argPartition(-selection, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
        var selectedScores = takeAlong(scores, inds, axis: -1)

        if topK > 1, normTopkProb {
            selectedScores =
                selectedScores / (selectedScores.sum(axis: -1, keepDims: true) + 1e-20)
        }
        selectedScores = selectedScores * routedScalingFactor
        return (inds, selectedScores)
    }
}

final class BailingV3SparseMoE: Module, UnaryLayer {
    let gate: BailingV3MoEGate

    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "shared_experts") var sharedExperts: BailingV3MLP?

    init(_ args: BailingMoeV3Configuration) {
        guard let numExperts = args.numExperts else {
            fatalError("BailingV3SparseMoE requires num_experts")
        }
        self.gate = BailingV3MoEGate(args)
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: args.hiddenSize,
            hiddenDims: args.moeIntermediateSize,
            numExperts: numExperts)
        if args.numSharedExperts > 0 {
            _sharedExperts.wrappedValue = BailingV3MLP(
                args,
                intermediateSize: args.moeSharedExpertIntermediateSize
                    * args.numSharedExperts)
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (inds, scores) = gate(x)
        var y = Self.combineExpertOutputs(switchMLP(x, inds), scores: scores)
        if let sharedExperts {
            y = y + sharedExperts(x)
        }
        return y
    }

    /// Routed-expert combination in float32, cast back once — the reference
    /// (`modeling_bailing_moe_v3.py`, `.type(topk_weight.dtype).mul_(…).sum(…).type(new_x.dtype)`)
    /// and upstream mlx-lm both weight and sum in the scores' float32. The
    /// previous form rounded the scores to the expert dtype and summed the
    /// top-k products in bf16.
    static func combineExpertOutputs(_ y: MLXArray, scores: MLXArray) -> MLXArray {
        (y.asType(.float32) * scores[.ellipsis, .newAxis].asType(.float32))
            .sum(axis: -2)
            .asType(y.dtype)
    }
}

// MARK: - Decoder layer / model

/// Both layer kinds load from the same `attention` bundle key (the Python
/// class has a single `self.attention` attribute), so the decoder layer keeps
/// one existential property — the same trick as BailingHybrid.
protocol BailingV3Attention: Module {
    func callAsV3Attention(
        _ x: MLXArray,
        attnMask: MLXFast.ScaledDotProductAttentionMaskMode,
        ssmMask: MLXArray?,
        cache: KVCache?
    ) -> MLXArray
}

extension BailingV3MLAAttention: BailingV3Attention {
    func callAsV3Attention(
        _ x: MLXArray,
        attnMask: MLXFast.ScaledDotProductAttentionMaskMode,
        ssmMask: MLXArray?,
        cache: KVCache?
    ) -> MLXArray {
        callAsFunction(x, mask: attnMask, cache: cache)
    }
}

extension BailingV3KDAAttention: BailingV3Attention {
    func callAsV3Attention(
        _ x: MLXArray,
        attnMask: MLXFast.ScaledDotProductAttentionMaskMode,
        ssmMask: MLXArray?,
        cache: KVCache?
    ) -> MLXArray {
        // The recurrence is implicitly causal; only the SSM padding mask
        // applies here.
        callAsFunction(x, mask: ssmMask, cache: cache as? MambaCache)
    }
}

final class BailingV3DecoderLayer: Module {
    let isFullAttention: Bool

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    @ModuleInfo(key: "attention") var attention: any BailingV3Attention
    let mlp: UnaryLayer

    init(_ args: BailingMoeV3Configuration, layerIdx: Int) {
        self.isFullAttention = args.isFullAttentionLayer(layerIdx)
        if isFullAttention {
            _attention.wrappedValue = BailingV3MLAAttention(args)
        } else {
            _attention.wrappedValue = BailingV3KDAAttention(args)
        }
        if args.numExperts != nil, layerIdx >= args.firstKDenseReplace {
            self.mlp = BailingV3SparseMoE(args)
        } else {
            self.mlp = BailingV3MLP(args)
        }
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        attnMask: MLXFast.ScaledDotProductAttentionMaskMode,
        ssmMask: MLXArray?,
        cache: KVCache?
    ) -> MLXArray {
        let r = attention.callAsV3Attention(
            inputLayerNorm(x), attnMask: attnMask, ssmMask: ssmMask, cache: cache)
        let h = x + r
        let r2 = mlp(postAttentionLayerNorm(h))
        return h + r2
    }
}

public class BailingMoeV3LanguageModel: Module {
    let args: BailingMoeV3Configuration

    @ModuleInfo(key: "word_embeddings") var wordEmbeddings: Embedding
    var layers: [BailingV3DecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    let firstFullIdx: Int
    let firstLinearIdx: Int

    init(_ args: BailingMoeV3Configuration) {
        self.args = args
        self._wordEmbeddings.wrappedValue = Embedding(
            embeddingCount: args.vocabSize, dimensions: args.hiddenSize)
        self.layers = (0 ..< args.numHiddenLayers).map {
            BailingV3DecoderLayer(args, layerIdx: $0)
        }
        self._norm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        self.firstFullIdx =
            (0 ..< args.numHiddenLayers).first(where: { args.isFullAttentionLayer($0) })
            ?? 0
        self.firstLinearIdx =
            (0 ..< args.numHiddenLayers).first(where: { !args.isFullAttentionLayer($0) })
            ?? 0
        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = wordEmbeddings(inputs)

        let fullCache = cache?[firstFullIdx]
        let attnMask = createAttentionMask(h: h, cache: fullCache)
        let ssmMask = createSSMMask(h: h, cache: cache?[firstLinearIdx] as? MambaCache)

        for (i, layer) in layers.enumerated() {
            let layerCache = (cache != nil && i < cache!.count) ? cache![i] : nil
            h = layer(h, attnMask: attnMask, ssmMask: ssmMask, cache: layerCache)
        }
        return norm(h)
    }
}

public class BailingMoeV3Model: Module, LLMModel, KVCacheDimensionProvider {
    public var kvHeads: [Int]
    let args: BailingMoeV3Configuration

    @ModuleInfo(key: "model") var model: BailingMoeV3LanguageModel
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ args: BailingMoeV3Configuration) {
        self.args = args
        self._model.wrappedValue = BailingMoeV3LanguageModel(args)
        if !args.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(
                args.hiddenSize, args.vocabSize, bias: false)
        }
        // Only the full-attention (MLA) layers hold a KV cache.
        self.kvHeads = (0 ..< args.numHiddenLayers).map {
            args.isFullAttentionLayer($0) ? args.numAttentionHeads : 0
        }
        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let out = model(inputs, cache: cache)
        if let lmHead {
            return lmHead(out)
        }
        return model.wordEmbeddings.asLinear(out)
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        (0 ..< args.numHiddenLayers).map { i in
            if args.isFullAttentionLayer(i) {
                if let maxKVSize = parameters?.maxKVSize {
                    return RotatingKVCache(maxSize: maxKVSize, keep: 4)
                }
                return KVCacheSimple()
            }
            // Slot 0: concatenated q/k/v conv tail. Slot 1: recurrent state.
            return MambaCache()
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var out = weights

        if args.tieWordEmbeddings {
            out.removeValue(forKey: "lm_head.weight")
        }

        // MTP head weights (num_nextn_predict_layers > 0 bundles) are not
        // part of the standard forward.
        for key in out.keys.filter({ $0.contains("mtp.") }) {
            out.removeValue(forKey: key)
        }

        // Torch conv layout [C, 1, k] → MLX [C, k, 1].
        for (key, value) in out {
            if key.contains("conv1d.weight"), value.ndim == 3, value.dim(-1) != 1 {
                out[key] = value.movedAxis(source: 2, destination: 1)
            }
        }

        // Upstream (unstamped) bundles ship loose per-expert weights; JANG
        // bundles are already stacked under `switch_mlp`.
        if out["model.layers.1.mlp.experts.0.up_proj.weight"] != nil {
            for l in 0 ..< args.numHiddenLayers {
                let prefix = "model.layers.\(l).mlp"
                guard out["\(prefix).experts.0.up_proj.weight"] != nil else { continue }
                for n in ["gate_proj", "up_proj", "down_proj"] {
                    let joined = (0 ..< (args.numExperts ?? 0)).compactMap {
                        out.removeValue(forKey: "\(prefix).experts.\($0).\(n).weight")
                    }
                    if !joined.isEmpty {
                        out["\(prefix).switch_mlp.\(n).weight"] = MLX.stacked(joined)
                    }
                }
            }
        }
        return out
    }
}

extension BailingMoeV3Model: LoRAModel {
    public var loraLayers: [Module] { model.layers }
    public var loraDefaultKeys: [String] { ["attention.kv_a_proj_with_mqa"] }
}
