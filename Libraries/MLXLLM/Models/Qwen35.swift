//
//  Qwen35.swift
//  mlx-swift-lm
//
//  Created by John Mai on 2026/2/9.
//
//  Port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3_5.py
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Compiled shared expert gate: sigmoid(gate_output) * expert_output → 1 fused op.
private let compiledSigmoidGate: @Sendable (MLXArray, MLXArray) -> MLXArray = {
    let body: @Sendable (MLXArray, MLXArray) -> MLXArray = { (gateOutput: MLXArray, expertOutput: MLXArray) -> MLXArray in
        sigmoid(gateOutput) * expertOutput
    }
    return HardwareInfo.isCompiledDecodeSupported ? compile(shapeless: true, body) : body
}()


// MARK: - Configuration

private enum RopeParametersCodingKey: String, CodingKey {
    case ropeParameters = "rope_parameters"
}

public struct Qwen35TextConfiguration: Codable, Sendable {
    var modelType: String = ""
    var hiddenSize: Int = 4096
    var hiddenLayers: Int = 32
    var intermediateSize: Int = 14336
    var attentionHeads: Int = 32
    var kvHeads: Int = 8
    var linearNumValueHeads: Int = 64
    var linearNumKeyHeads: Int = 16
    var linearKeyHeadDim: Int = 192
    var linearValueHeadDim: Int = 128
    var linearConvKernelDim: Int = 4
    var rmsNormEps: Float = 1e-6
    var vocabularySize: Int = 151_936
    var ropeTheta: Float = 100000.0
    var partialRotaryFactor: Float = 0.25
    var maxPositionEmbeddings: Int = 131072
    var tieWordEmbeddings: Bool = false
    var attentionBias: Bool = false
    var headDim: Int?
    var ropeScaling: [String: StringOrNumber]?
    var fullAttentionInterval: Int = 4
    var mtpNumHiddenLayers: Int = 0

    // MoE fields
    var numExperts: Int = 0
    var numExpertsPerTok: Int = 0
    var decoderSparseStep: Int = 1
    var sharedExpertIntermediateSize: Int = 0
    var moeIntermediateSize: Int = 0
    var normTopkProb: Bool = true

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case linearNumValueHeads = "linear_num_value_heads"
        case linearNumKeyHeads = "linear_num_key_heads"
        case linearKeyHeadDim = "linear_key_head_dim"
        case linearValueHeadDim = "linear_value_head_dim"
        case linearConvKernelDim = "linear_conv_kernel_dim"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case ropeTheta = "rope_theta"
        case partialRotaryFactor = "partial_rotary_factor"
        case maxPositionEmbeddings = "max_position_embeddings"
        case tieWordEmbeddings = "tie_word_embeddings"
        case attentionBias = "attention_bias"
        case headDim = "head_dim"
        case ropeScaling = "rope_scaling"
        case fullAttentionInterval = "full_attention_interval"
        case mtpNumHiddenLayers = "mtp_num_hidden_layers"
        case numExperts = "num_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case decoderSparseStep = "decoder_sparse_step"
        case sharedExpertIntermediateSize = "shared_expert_intermediate_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case normTopkProb = "norm_topk_prob"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let defaultRopeParameters: [String: StringOrNumber] = [
            "type": .string("default"),
            "mrope_section": .ints([11, 11, 10]),
            "rope_theta": .float(100000.0),
            "partial_rotary_factor": .float(0.25),
        ]

        self.modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? ""
        self.hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 4096
        self.hiddenLayers = try container.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 32
        self.intermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 14336
        self.attentionHeads = try container.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 32
        self.kvHeads = try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 8
        self.linearNumValueHeads =
            try container.decodeIfPresent(Int.self, forKey: .linearNumValueHeads) ?? 64
        self.linearNumKeyHeads =
            try container.decodeIfPresent(Int.self, forKey: .linearNumKeyHeads) ?? 16
        self.linearKeyHeadDim =
            try container.decodeIfPresent(Int.self, forKey: .linearKeyHeadDim) ?? 192
        self.linearValueHeadDim =
            try container.decodeIfPresent(Int.self, forKey: .linearValueHeadDim) ?? 128
        self.linearConvKernelDim =
            try container.decodeIfPresent(Int.self, forKey: .linearConvKernelDim) ?? 4
        self.rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        self.vocabularySize =
            try container.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 151_936
        self.maxPositionEmbeddings =
            try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 131072
        self.tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        self.attentionBias =
            try container.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        self.headDim = try container.decodeIfPresent(Int.self, forKey: .headDim)
        self.fullAttentionInterval =
            try container.decodeIfPresent(Int.self, forKey: .fullAttentionInterval) ?? 4
        self.mtpNumHiddenLayers =
            try container.decodeIfPresent(Int.self, forKey: .mtpNumHiddenLayers) ?? 0

        // MoE fields
        self.numExperts = try container.decodeIfPresent(Int.self, forKey: .numExperts) ?? 0
        self.numExpertsPerTok = RuntimeMoETopKOverride.effectiveTopK(
            currentTopK: try container.decodeIfPresent(Int.self, forKey: .numExpertsPerTok) ?? 0,
            modelType: modelType,
            field: CodingKeys.numExpertsPerTok.rawValue)
        self.decoderSparseStep =
            try container.decodeIfPresent(Int.self, forKey: .decoderSparseStep) ?? 1
        self.sharedExpertIntermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .sharedExpertIntermediateSize) ?? 0
        self.moeIntermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .moeIntermediateSize) ?? 0
        self.normTopkProb = try container.decodeIfPresent(Bool.self, forKey: .normTopkProb) ?? true

        let ropeContainer = try decoder.container(keyedBy: RopeParametersCodingKey.self)
        let ropeParameters = try ropeContainer.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeParameters)

        if var ropeParameters {
            if ropeParameters["type"] == nil, let ropeType = ropeParameters["rope_type"] {
                ropeParameters["type"] = ropeType
            }
            self.ropeTheta = ropeParameters["rope_theta"]?.asFloat() ?? 100000.0
            self.partialRotaryFactor =
                ropeParameters["partial_rotary_factor"]?.asFloat() ?? 0.25
            self.ropeScaling = ropeParameters
        } else {
            self.ropeTheta =
                try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 100000.0
            self.partialRotaryFactor =
                try container.decodeIfPresent(Float.self, forKey: .partialRotaryFactor) ?? 0.25
            self.ropeScaling =
                try container.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeScaling)
                ?? defaultRopeParameters
        }

        if self.headDim == nil {
            self.headDim = self.hiddenSize / self.attentionHeads
        }
    }
}

// MARK: - GatedDeltaNet

final class Qwen35GatedDeltaNet: Module {
    let hiddenSize: Int
    let numVHeads: Int
    let numKHeads: Int
    let headKDim: Int
    let headVDim: Int
    let keyDim: Int
    let valueDim: Int
    let convKernelSize: Int
    let convDim: Int

    @ModuleInfo(key: "conv1d") var conv1d: Conv1d
    @ModuleInfo(key: "in_proj_qkv") var inProjQKV: Linear
    @ModuleInfo(key: "in_proj_z") var inProjZ: Linear
    @ModuleInfo(key: "in_proj_b") var inProjB: Linear
    @ModuleInfo(key: "in_proj_a") var inProjA: Linear

    @ParameterInfo(key: "dt_bias") var dtBias: MLXArray
    @ParameterInfo(key: "A_log") var aLog: MLXArray

    @ModuleInfo(key: "norm") var norm: Qwen3NextRMSNormGated
    @ModuleInfo(key: "out_proj") var outProj: Linear

    // Pre-computed scaled-norm weights — q uses (1/sqrt(headDim))^2, k uses (1/sqrt(headDim)).
    // Replaces `scalar * MLXFast.rmsNorm(x, weight: nil)` (2 ops) with one
    // `MLXFast.rmsNorm(x, weight: precomputed)` (1 op) by baking the scale into
    // the rms_norm weight vector. Computed lazily on first use because we need
    // the parameter dtype which isn't known at init.
    private var qScaleWeight: MLXArray?
    private var kScaleWeight: MLXArray?

    init(_ args: Qwen35TextConfiguration) {
        self.hiddenSize = args.hiddenSize
        self.numVHeads = args.linearNumValueHeads
        self.numKHeads = args.linearNumKeyHeads
        self.headKDim = args.linearKeyHeadDim
        self.headVDim = args.linearValueHeadDim
        self.keyDim = headKDim * numKHeads
        self.valueDim = headVDim * numVHeads
        self.convKernelSize = args.linearConvKernelDim
        self.convDim = keyDim * 2 + valueDim

        precondition(
            numVHeads % numKHeads == 0,
            "num_v_heads (\(numVHeads)) must be divisible by num_k_heads (\(numKHeads))"
        )

        _conv1d.wrappedValue = Conv1d(
            inputChannels: convDim,
            outputChannels: convDim,
            kernelSize: convKernelSize,
            stride: 1,
            padding: 0,
            dilation: 1,
            groups: convDim,
            bias: false
        )

        _inProjQKV.wrappedValue = Linear(hiddenSize, keyDim * 2 + valueDim, bias: false)
        _inProjZ.wrappedValue = Linear(hiddenSize, valueDim, bias: false)
        _inProjB.wrappedValue = Linear(hiddenSize, numVHeads, bias: false)
        _inProjA.wrappedValue = Linear(hiddenSize, numVHeads, bias: false)

        _dtBias.wrappedValue = MLXArray.ones([numVHeads])
        let a = MLXRandom.uniform(low: 0, high: 16, [numVHeads])
        _aLog.wrappedValue = log(a)

        _norm.wrappedValue = Qwen3NextRMSNormGated(dimensions: headVDim, eps: args.rmsNormEps)
        _outProj.wrappedValue = Linear(valueDim, hiddenSize, bias: false)

        super.init()
    }

    func callAsFunction(
        _ inputs: MLXArray,
        mask: MLXArray? = nil,
        cache: MambaCache? = nil,
        recordPrefixCommitStates: Bool = false
    ) -> MLXArray {
        let B = inputs.dim(0)
        let S = inputs.dim(1)

        var qkv = inProjQKV(inputs)
        let z = inProjZ(inputs).reshaped(B, S, numVHeads, headVDim)
        let b = inProjB(inputs)
        let a = inProjA(inputs)

        let convState: MLXArray
        if let cacheState = cache?[0] {
            convState = cacheState
        } else {
            convState = MLXArray.zeros([B, convKernelSize - 1, convDim], dtype: inputs.dtype)
        }

        if let mask {
            qkv = MLX.where(mask[.ellipsis, .newAxis], qkv, 0)
        }

        let convInput = concatenated([convState, qkv], axis: 1)
            .reshaped(B, convState.dim(1) + S, convDim)
        if let cache {
            let end = convInput.dim(1)
            let start = max(0, end - (convKernelSize - 1))
            cache[0] = convInput[0..., start ..< end, 0...]
        }

        let convOut = silu(conv1d(convInput))

        let convSplit = MLX.split(convOut, indices: [keyDim, 2 * keyDim], axis: -1)
        let q = convSplit[0].reshaped(B, S, numKHeads, headKDim)
        let k = convSplit[1].reshaped(B, S, numKHeads, headKDim)
        let v = convSplit[2].reshaped(B, S, numVHeads, headVDim)

        let initialState = cache?[1]
        var state = initialState
        // Fused scaled rms_norm: bake the per-head-dim scale into the rms_norm
        // weight vector so MLXFast.rmsNorm does (scale * normed) in one Metal
        // dispatch instead of two (rms_norm + scalar multiply). Saves ~2 ops
        // per linear layer per token (~60 ops/token for 30 linear layers).
        if qScaleWeight == nil || qScaleWeight!.dtype != q.dtype {
            let invScale = pow(Float(headKDim), -0.5)
            qScaleWeight = MLXArray.full(
                [headKDim], values: MLXArray(pow(invScale, 2), dtype: q.dtype), dtype: q.dtype)
            kScaleWeight = MLXArray.full(
                [headKDim], values: MLXArray(invScale, dtype: k.dtype), dtype: k.dtype)
        }
        let qNormed = MLXFast.rmsNorm(q, weight: qScaleWeight!, eps: 1e-6)
        let kNormed = MLXFast.rmsNorm(k, weight: kScaleWeight!, eps: 1e-6)

        var out: MLXArray

        (out, state) = gatedDeltaUpdate(
            q: qNormed,
            k: kNormed,
            v: v,
            a: a,
            b: b,
            aLog: aLog,
            dtBias: dtBias,
            state: state,
            mask: mask
        )
        let finalState = state!

        if let cache {
            if recordPrefixCommitStates, S > 1 {
                self.recordPrefixCommitStates(
                    cache: cache,
                    convInput: convInput,
                    q: qNormed,
                    k: kNormed,
                    v: v,
                    a: a,
                    b: b,
                    initialState: initialState,
                    finalState: finalState,
                    mask: mask,
                    baseOffset: cache.offset)
            }
            cache[1] = finalState
            cache.offset += S
        }

        out = norm(out, gate: z)
        return outProj(out.reshaped(B, S, -1))
    }

    private func recordPrefixCommitStates(
        cache: MambaCache,
        convInput: MLXArray,
        q: MLXArray,
        k: MLXArray,
        v: MLXArray,
        a: MLXArray,
        b: MLXArray,
        initialState: MLXArray?,
        finalState: MLXArray,
        mask: MLXArray?,
        baseOffset: Int
    ) {
        let sequenceLength = q.dim(1)
        guard sequenceLength > 0 else { return }

        for prefixLength in 1 ... sequenceLength {
            let convEnd = convInput.dim(1) - sequenceLength + prefixLength
            let convStart = max(0, convEnd - max(0, convKernelSize - 1))
            let convState = convInput[0..., convStart ..< convEnd, 0...]

            let recurrentState: MLXArray
            if prefixLength == sequenceLength {
                recurrentState = finalState
            } else {
                let (_, prefixState) = gatedDeltaUpdate(
                    q: q[0..., 0 ..< prefixLength, 0..., 0...],
                    k: k[0..., 0 ..< prefixLength, 0..., 0...],
                    v: v[0..., 0 ..< prefixLength, 0..., 0...],
                    a: a[0..., 0 ..< prefixLength, 0...],
                    b: b[0..., 0 ..< prefixLength, 0...],
                    aLog: aLog,
                    dtBias: dtBias,
                    state: initialState,
                    mask: prefixMask(mask, length: prefixLength))
                recurrentState = prefixState
            }

            cache.recordPrefixCommitState(
                length: prefixLength,
                arrays: [convState, recurrentState],
                offset: baseOffset + prefixLength)
        }
    }

    private func prefixMask(_ mask: MLXArray?, length: Int) -> MLXArray? {
        guard let mask else { return nil }
        if mask.ndim == 1 {
            return mask[0 ..< length]
        }
        if mask.ndim == 2 {
            return mask[0..., 0 ..< length]
        }
        return mask[0..., 0 ..< length, 0...]
    }
}

// MARK: - Attention

final class Qwen35Attention: Module {
    let attentionHeads: Int
    let kvHeads: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let rope: RoPELayer

    init(_ args: Qwen35TextConfiguration) {
        let headDim = args.headDim ?? (args.hiddenSize / args.attentionHeads)
        self.attentionHeads = args.attentionHeads
        self.kvHeads = args.kvHeads
        self.scale = pow(Float(headDim), -0.5)

        _qProj.wrappedValue = Linear(
            args.hiddenSize, args.attentionHeads * headDim * 2, bias: args.attentionBias)
        _kProj.wrappedValue = Linear(
            args.hiddenSize, args.kvHeads * headDim, bias: args.attentionBias)
        _vProj.wrappedValue = Linear(
            args.hiddenSize, args.kvHeads * headDim, bias: args.attentionBias)
        _oProj.wrappedValue = Linear(
            args.attentionHeads * headDim, args.hiddenSize, bias: args.attentionBias)

        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)

        let ropeDims = Int(Float(headDim) * args.partialRotaryFactor)
        self.rope = initializeRope(
            dims: max(1, ropeDims),
            base: args.ropeTheta,
            traditional: false,
            scalingConfig: args.ropeScaling,
            maxPositionEmbeddings: args.maxPositionEmbeddings
        )

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)

        let qProjOutput = qProj(x)
        let qSplit = qProjOutput.reshaped(B, L, attentionHeads, -1).split(parts: 2, axis: -1)
        var queries = qSplit[0]
        let gate = qSplit[1].reshaped(B, L, -1)

        var keys = kProj(x)
        var values = vProj(x)

        queries = qNorm(queries).transposed(0, 2, 1, 3)
        keys = kNorm(keys.reshaped(B, L, kvHeads, -1)).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, kvHeads, -1).transposed(0, 2, 1, 3)

        queries = applyRotaryPosition(rope, to: queries, cache: cache)
        keys = applyRotaryPosition(rope, to: keys, cache: cache)

        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return oProj(sigmoidMultiply(output, gate))
    }
}

// MARK: - SparseMoeBlock

final class Qwen35SparseMoeBlock: Module, UnaryLayer {
    let layerIdx: Int
    let normTopkProb: Bool
    let numExperts: Int
    let topK: Int

    @ModuleInfo(key: "gate") var gate: Linear
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU

    @ModuleInfo(key: "shared_expert") var sharedExpert: Qwen3NextMLP
    @ModuleInfo(key: "shared_expert_gate") var sharedExpertGate: Linear

    init(_ args: Qwen35TextConfiguration, layerIdx: Int) {
        self.layerIdx = layerIdx
        self.normTopkProb = args.normTopkProb
        self.numExperts = args.numExperts
        self.topK = args.numExpertsPerTok

        _gate.wrappedValue = Linear(args.hiddenSize, args.numExperts, bias: false)
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: args.hiddenSize,
            hiddenDims: args.moeIntermediateSize,
            numExperts: args.numExperts
        )

        _sharedExpert.wrappedValue = Qwen3NextMLP(
            dimensions: args.hiddenSize,
            hiddenDimensions: args.sharedExpertIntermediateSize
        )
        _sharedExpertGate.wrappedValue = Linear(args.hiddenSize, 1, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var gates = gate(x)
        gates = MLX.softmax(gates, axis: -1, precise: true)

        let k = topK
        let kth = gates.dim(-1) - k
        let inds = MLX.argPartition(gates, kth: kth, axis: -1)[.ellipsis, (kth)...]
        JangPressCanonicalExpertAdvisor.shared.observe(layer: layerIdx, indices: inds)
        var scores = MLX.takeAlong(gates, inds, axis: -1)
        if normTopkProb {
            scores = scores / scores.sum(axis: -1, keepDims: true)
        }

        let y = switchMLP(x, inds)
        let combined = (y * scores[.ellipsis, .newAxis]).sum(axis: -2)

        let sharedY = sharedExpert(x)
        let gatedSharedY = compiledSigmoidGate(sharedExpertGate(x), sharedY)

        return combined + gatedSharedY
    }
}

// MARK: - Decoder Layer

final class Qwen35DecoderLayer: Module {
    let isLinear: Bool

    @ModuleInfo(key: "self_attn") var selfAttn: Qwen35Attention?
    @ModuleInfo(key: "linear_attn") var linearAttn: Qwen35GatedDeltaNet?

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    @ModuleInfo(key: "mlp") var mlp: Module

    init(_ args: Qwen35TextConfiguration, layerIdx: Int) {
        self.isLinear = (layerIdx + 1) % args.fullAttentionInterval != 0

        if isLinear {
            _linearAttn.wrappedValue = Qwen35GatedDeltaNet(args)
        } else {
            _selfAttn.wrappedValue = Qwen35Attention(args)
        }

        if args.numExperts > 0 {
            _mlp.wrappedValue = Qwen35SparseMoeBlock(args, layerIdx: layerIdx)
        } else {
            _mlp.wrappedValue = Qwen3NextMLP(
                dimensions: args.hiddenSize,
                hiddenDimensions: args.intermediateSize
            )
        }

        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize,
            eps: args.rmsNormEps
        )
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize,
            eps: args.rmsNormEps
        )

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        attentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
        ssmMask: MLXArray?,
        cache: KVCache?,
        recordPrefixCommitStates: Bool = false
    ) -> MLXArray {
        let r: MLXArray
        if isLinear {
            r = linearAttn!(
                inputLayerNorm(x),
                mask: ssmMask,
                cache: cache as? MambaCache,
                recordPrefixCommitStates: recordPrefixCommitStates)
        } else {
            r = selfAttn!(inputLayerNorm(x), mask: attentionMask, cache: cache)
        }

        let h = x + r
        return h + (mlp as! UnaryLayer)(postAttentionLayerNorm(h))
    }
}

// MARK: - Native MTP

final class Qwen35MTPDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: Qwen35Attention
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "mlp") var mlp: Module

    init(_ args: Qwen35TextConfiguration) {
        _selfAttn.wrappedValue = Qwen35Attention(args)
        _inputLayerNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        if args.numExperts > 0 {
            _mlp.wrappedValue = Qwen35SparseMoeBlock(args, layerIdx: -1)
        } else {
            _mlp.wrappedValue = Qwen3NextMLP(
                dimensions: args.hiddenSize,
                hiddenDimensions: args.intermediateSize)
        }
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        attentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let h = x + selfAttn(inputLayerNorm(x), mask: attentionMask, cache: cache)
        return h + (mlp as! UnaryLayer)(postAttentionLayerNorm(h))
    }
}

final class Qwen35MTPModule: Module {
    @ModuleInfo(key: "pre_fc_norm_hidden") var preFCNormHidden: RMSNorm
    @ModuleInfo(key: "pre_fc_norm_embedding") var preFCNormEmbedding: RMSNorm
    @ModuleInfo(key: "fc") var fc: Linear
    @ModuleInfo(key: "layers") var layers: [Qwen35MTPDecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(_ args: Qwen35TextConfiguration) {
        _preFCNormHidden.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _preFCNormEmbedding.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _fc.wrappedValue = Linear(args.hiddenSize * 2, args.hiddenSize, bias: false)
        _layers.wrappedValue = (0 ..< max(0, args.mtpNumHiddenLayers)).map { _ in
            Qwen35MTPDecoderLayer(args)
        }
        _norm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        super.init()
    }

    func preNormHidden(
        hiddenStates: MLXArray,
        nextTokenIds: MLXArray,
        embedTokens: Embedding,
        cache: [KVCache]?
    ) -> MLXArray {
        let embeds = embedTokens(nextTokenIds)
        let fusedInput = concatenated([
            preFCNormEmbedding(embeds),
            preFCNormHidden(hiddenStates),
        ], axis: -1)
        var hiddenStates = fc(fusedInput)

        var cacheArray = cache
        if cacheArray == nil {
            cacheArray = (0 ..< layers.count).map { _ in KVCacheSimple() as KVCache }
        }
        let mask = createAttentionMask(h: hiddenStates, cache: cacheArray?.first)
        for (index, layer) in layers.enumerated() {
            hiddenStates = layer(hiddenStates, attentionMask: mask, cache: cacheArray?[index])
        }
        return hiddenStates
    }

    func callAsFunction(
        hiddenStates: MLXArray,
        nextTokenIds: MLXArray,
        embedTokens: Embedding,
        cache: [KVCache]?
    ) -> MLXArray {
        norm(preNormHidden(
            hiddenStates: hiddenStates,
            nextTokenIds: nextTokenIds,
            embedTokens: embedTokens,
            cache: cache))
    }

    func makeCache() -> [KVCache] {
        (0 ..< layers.count).map { _ in KVCacheSimple() as KVCache }
    }
}

// MARK: - Text Model

public class Qwen35TextModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    fileprivate let layers: [Qwen35DecoderLayer]
    let norm: RMSNorm

    let ssmIdx: Int
    let faIdx: Int

    init(_ args: Qwen35TextConfiguration) {
        precondition(args.vocabularySize > 0)

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize,
            dimensions: args.hiddenSize
        )

        self.layers = (0 ..< args.hiddenLayers).map { layerIdx in
            Qwen35DecoderLayer(args, layerIdx: layerIdx)
        }

        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)

        self.ssmIdx = 0
        self.faIdx = args.fullAttentionInterval - 1

        super.init()
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache?]? = nil) -> MLXArray {
        let (h, _) = callAsFunctionCapturing(
            inputs, cache: cache, captureLayerIDs: [])
        return h
    }

    func callAsFunctionPreNorm(
        _ inputs: MLXArray,
        cache: [KVCache?]? = nil,
        recordPrefixCommitStates: Bool = false
    ) -> MLXArray {
        var hiddenStates = embedTokens(inputs)

        var cacheArray = cache
        if cacheArray == nil {
            cacheArray = Array(repeating: nil as KVCache?, count: layers.count)
        }

        let faMask = createAttentionMask(h: hiddenStates, cache: cacheArray?[faIdx])
        let ssmMask = createSSMMask(h: hiddenStates, cache: cacheArray?[ssmIdx] as? MambaCache)

        for (index, layer) in layers.enumerated() {
            let mask = layer.isLinear ? ssmMask : nil
            let attnMask =
                layer.isLinear
                ? MLXFast.ScaledDotProductAttentionMaskMode.none : faMask
            hiddenStates = layer(
                hiddenStates,
                attentionMask: attnMask,
                ssmMask: mask,
                cache: cacheArray?[index],
                recordPrefixCommitStates: recordPrefixCommitStates)
        }

        return hiddenStates
    }

    /// Forward with optional per-block hidden-state capture. Mirrors
    /// `Qwen3ModelInner.callAsFunctionCapturing`. Empty `captureLayerIDs`
    /// produces output byte-identical to the plain forward.
    func callAsFunctionCapturing(
        _ inputs: MLXArray,
        cache: [KVCache?]? = nil,
        captureLayerIDs: Set<Int>
    ) -> (MLXArray, [Int: MLXArray]) {
        var hiddenStates = embedTokens(inputs)

        var cacheArray = cache
        if cacheArray == nil {
            cacheArray = Array(repeating: nil as KVCache?, count: layers.count)
        }

        let faMask = createAttentionMask(h: hiddenStates, cache: cacheArray?[faIdx])
        let ssmMask = createSSMMask(h: hiddenStates, cache: cacheArray?[ssmIdx] as? MambaCache)

        var captured: [Int: MLXArray] = [:]
        captured.reserveCapacity(captureLayerIDs.count)

        for (i, layer) in layers.enumerated() {
            let mask = layer.isLinear ? ssmMask : nil
            let attnMask =
                layer.isLinear
                ? MLXFast.ScaledDotProductAttentionMaskMode.none : faMask
            hiddenStates = layer(
                hiddenStates, attentionMask: attnMask, ssmMask: mask, cache: cacheArray?[i])
            if captureLayerIDs.contains(i) {
                captured[i] = hiddenStates
            }
        }

        return (norm(hiddenStates), captured)
    }
}

public class Qwen35TextModel: Module, LLMModel, KVCacheDimensionProvider, HiddenStateCaptureModel, TokenEmbedderModel, NativeMTPModel {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: Qwen35TextModelInner
    let configuration: Qwen35TextConfiguration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?
    @ModuleInfo(key: "mtp") var mtp: Qwen35MTPModule?

    public init(_ args: Qwen35TextConfiguration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
        self.model = Qwen35TextModelInner(args)

        if !args.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }
        if args.mtpNumHiddenLayers > 0 {
            _mtp.wrappedValue = Qwen35MTPModule(args)
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

    /// `HiddenStateCaptureModel` conformance — forwards to the inner
    /// capturing variant. Empty `captureLayerIDs` yields byte-identical
    /// logits to the plain forward. See Qwen35TextModelInner.callAsFunctionCapturing.
    public func callAsFunction(
        _ inputs: MLXArray,
        cache: [KVCache]?,
        captureLayerIDs: Set<Int>
    ) -> (logits: MLXArray, capturedHiddenStates: [Int: MLXArray]) {
        let (finalHidden, captured) = model.callAsFunctionCapturing(
            inputs, cache: cache, captureLayerIDs: captureLayerIDs)
        let logits: MLXArray
        if let lmHead {
            logits = lmHead(finalHidden)
        } else {
            logits = model.embedTokens.asLinear(finalHidden)
        }
        return (logits, captured)
    }

    // MARK: - TokenEmbedderModel

    public func embed(_ tokenIds: MLXArray) -> MLXArray {
        model.embedTokens(tokenIds)
    }

    public func projectToLogits(_ hidden: MLXArray) -> MLXArray {
        if let lmHead {
            return lmHead(hidden)
        }
        return model.embedTokens.asLinear(hidden)
    }

    // MARK: - NativeMTPModel

    public var nativeMTPAvailable: Bool { mtp != nil }

    public func makeNativeMTPCache() -> [KVCache] {
        mtp?.makeCache() ?? []
    }

    public func nativeBackboneForward(
        _ inputs: MLXArray,
        cache: [KVCache]?
    ) -> NativeMTPForwardResult {
        let cacheOpt: [KVCache?]? = cache?.map { $0 as KVCache? }
        let hidden = model.callAsFunctionPreNorm(inputs, cache: cacheOpt)
        return NativeMTPForwardResult(
            logits: projectToLogits(model.norm(hidden)),
            hiddenStates: hidden)
    }

    public func nativeBackboneMTPVerifyForward(
        _ inputs: MLXArray,
        cache: [KVCache]?
    ) -> NativeMTPForwardResult {
        let cacheOpt: [KVCache?]? = cache?.map { $0 as KVCache? }
        let hidden = model.callAsFunctionPreNorm(
            inputs,
            cache: cacheOpt,
            recordPrefixCommitStates: true)
        return NativeMTPForwardResult(
            logits: projectToLogits(model.norm(hidden)),
            hiddenStates: hidden)
    }

    public func nativeMTPForward(
        hiddenStates: MLXArray,
        nextTokenIds: MLXArray,
        cache: [KVCache]?
    ) -> NativeMTPForwardResult {
        guard let mtp else {
            fatalError("Qwen35 nativeMTPForward called without an MTP module")
        }
        let hidden = mtp.preNormHidden(
            hiddenStates: hiddenStates,
            nextTokenIds: nextTokenIds,
            embedTokens: model.embedTokens,
            cache: cache)
        return NativeMTPForwardResult(
            logits: projectToLogits(mtp.norm(hidden)),
            hiddenStates: hidden)
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        // 2026-05-01: honor `parameters.maxKVSize` for the attention slots.
        // Audit found Qwen3.5 / Qwen3.6 previously ignored maxKVSize on
        // `KVCacheSimple()`, leaving the CacheCoordinator's
        // `defaultMaxKVSize` contract a silent no-op — long-context
        // prompts could not be bounded. Mamba layers (`isLinear == true`)
        // ignore the bound by design (their hidden state is fixed-size).
        return model.layers.map { layer in
            if layer.isLinear {
                return MambaCache()
            }
            if let maxKVSize = parameters?.maxKVSize {
                return RotatingKVCache(maxSize: maxKVSize, keep: 4)
            }
            return KVCacheSimple()
        }
    }

    public func sanitize(weights: [String: MLXArray], metadata: [String: String]) -> [String:
        MLXArray]
    {
        sanitize(weights: weights, normConvention: Self.normConvention(metadata))
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        sanitize(weights: weights, normConvention: nil)
    }

    private func sanitize(weights: [String: MLXArray], normConvention: String?) -> [String:
        MLXArray]
    {
        let loadNativeMTP = configuration.mtpNumHiddenLayers > 0
        var weights = loadNativeMTP ? weights : weights.filter { !Self.isMTPWeightKey($0.key) }
        let hasUnsanitizedConv1d = weights.contains { key, value in
            key.contains("conv1d.weight") && value.dim(-1) != 1
        }
        let explicitNormConvention = normConvention != nil
        let shouldShiftNormWeights = Self.usesQwenPlusOneNormConvention(normConvention)
            || (!explicitNormConvention
                && (hasUnsanitizedConv1d || Self.baseNormWeightsNeedShift(weights)))
        let shouldShiftMTPNormWeights = loadNativeMTP
            && (shouldShiftNormWeights || Self.mtpNormWeightsNeedShift(weights))

        if configuration.tieWordEmbeddings {
            weights["lm_head.weight"] = nil
        }

        // All 5 norm types use the (1+weight) convention in Qwen3.5.
        // Matches Python mlx-lm and osa-jang reference implementations.
        let normKeysToShift = [
            ".input_layernorm.weight",
            ".post_attention_layernorm.weight",
            "model.norm.weight",
            ".q_norm.weight",
            ".k_norm.weight",
        ]

        for k in Array(weights.keys) {
            guard let v = weights[k] else { continue }
            if k.contains("conv1d.weight") && v.dim(-1) != 1 {
                weights[k] = v.movedAxis(source: 2, destination: 1)
                continue
            }
            let isMTPNorm = loadNativeMTP && Self.isMTPWeightKey(k) && k.hasSuffix(".weight")
                && (k.contains("norm") || k.contains("q_norm") || k.contains("k_norm"))
            let shouldShiftBaseNorm = shouldShiftNormWeights
                && normKeysToShift.contains(where: { k.hasSuffix($0) })
            let shouldShiftMTPNorm = isMTPNorm && shouldShiftMTPNormWeights
            if (shouldShiftBaseNorm || shouldShiftMTPNorm)
                && v.ndim == 1
            {
                weights[k] = v + MLXArray(1, dtype: v.dtype)
            }
        }

        return weights
    }

    private static func mtpNormWeightsNeedShift(_ weights: [String: MLXArray]) -> Bool {
        let probeSuffixes = [
            "mtp.layers.0.input_layernorm.weight",
            "mtp.pre_fc_norm_hidden.weight",
            "mtp.pre_fc_norm_embedding.weight",
        ]
        for suffix in probeSuffixes {
            for (key, value) in weights where value.ndim == 1 {
                guard Self.isMTPWeightKey(key), key.hasSuffix(suffix) else {
                    continue
                }
                return value.asType(.float32).mean().item(Float.self) < 0.5
            }
        }
        return false
    }

    private static func normConvention(_ metadata: [String: String]) -> String? {
        let value = metadata["norm_convention"] ?? metadata["runtime.norm_convention"]
        return value?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
    }

    private static func usesQwenPlusOneNormConvention(_ value: String?) -> Bool {
        value == "qwen3_5_language_mlx_plus_one"
            || value == "qwen35_language_mlx_plus_one"
            || value == "mlx_plus_one"
    }

    private static func isMTPWeightKey(_ key: String) -> Bool {
        key.hasPrefix("mtp.")
            || key.hasPrefix("model.mtp_layers.")
            || key.contains(".mtp.")
            || key.contains(".mtp_layers.")
    }

    private static func baseNormWeightsNeedShift(_ weights: [String: MLXArray]) -> Bool {
        let probeSuffixes = [
            ".input_layernorm.weight",
            ".post_attention_layernorm.weight",
        ]
        for (key, value) in weights where value.ndim == 1 {
            guard probeSuffixes.contains(where: { key.hasSuffix($0) }) else {
                continue
            }
            let mean = value.asType(.float32).mean().item(Float.self)
            if mean < 0.5 { return true }
            if mean > 0.5 { return false }
        }
        return false
    }
}

extension Qwen35TextModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}

// MARK: - Top-level Model

public class Qwen35Model: Module, LLMModel, KVCacheDimensionProvider, HiddenStateCaptureModel, TokenEmbedderModel, NativeMTPModel {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    @ModuleInfo(key: "language_model") var languageModel: Qwen35TextModel

    public init(_ args: Qwen35Configuration) {
        let textModel = Qwen35TextModel(args.textConfig)
        self.vocabularySize = textModel.vocabularySize
        self.kvHeads = textModel.kvHeads
        _languageModel.wrappedValue = textModel
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache)
    }

    /// `HiddenStateCaptureModel` conformance — delegates through the
    /// inner language model. Hybrid SSM decoder layers capture alongside
    /// attention layers; drafters that target Qwen 3.5 can read states
    /// from any of the `num_hidden_layers` blocks.
    public func callAsFunction(
        _ inputs: MLXArray,
        cache: [KVCache]?,
        captureLayerIDs: Set<Int>
    ) -> (logits: MLXArray, capturedHiddenStates: [Int: MLXArray]) {
        languageModel(inputs, cache: cache, captureLayerIDs: captureLayerIDs)
    }

    public func embed(_ tokenIds: MLXArray) -> MLXArray {
        languageModel.embed(tokenIds)
    }

    public func projectToLogits(_ hidden: MLXArray) -> MLXArray {
        languageModel.projectToLogits(hidden)
    }

    // MARK: - NativeMTPModel

    public var nativeMTPAvailable: Bool { languageModel.nativeMTPAvailable }

    public func makeNativeMTPCache() -> [KVCache] {
        languageModel.makeNativeMTPCache()
    }

    public func nativeBackboneForward(
        _ inputs: MLXArray,
        cache: [KVCache]?
    ) -> NativeMTPForwardResult {
        languageModel.nativeBackboneForward(inputs, cache: cache)
    }

    public func nativeBackboneMTPVerifyForward(
        _ inputs: MLXArray,
        cache: [KVCache]?
    ) -> NativeMTPForwardResult {
        languageModel.nativeBackboneMTPVerifyForward(inputs, cache: cache)
    }

    public func nativeMTPForward(
        hiddenStates: MLXArray,
        nextTokenIds: MLXArray,
        cache: [KVCache]?
    ) -> NativeMTPForwardResult {
        languageModel.nativeMTPForward(
            hiddenStates: hiddenStates,
            nextTokenIds: nextTokenIds,
            cache: cache)
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        languageModel.newCache(parameters: parameters)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = [String: MLXArray]()
        for (key, value) in weights {
            if key.hasPrefix("vision_tower") || key.hasPrefix("model.visual") {
                continue
            }

            var key = key
            if key.hasPrefix("model.language_model") {
                key = key.replacingOccurrences(
                    of: "model.language_model", with: "language_model.model")
            } else if !key.hasPrefix("language_model.") {
                key = "language_model." + key
            }
            sanitized[key] = value
        }

        return languageModel.sanitize(weights: sanitized)
    }
}

extension Qwen35Model: LoRAModel {
    public var loraLayers: [Module] {
        languageModel.model.layers
    }
}
