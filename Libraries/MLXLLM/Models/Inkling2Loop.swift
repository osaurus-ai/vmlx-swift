// Copyright © 2026 Jinho Jang. All rights reserved.
//
// Inkling-2-Mini (`model_type: inkling2_loop`, thinkingmachines/Inkling-2-Mini).
//
// Reference: transformers PR #51050 "Add Inkling2 Mini" (the only reference). Port source:
// jang_tools/inkling2/model.py, which is parity-tested against that PR (full logits relL2 1.4e-6).
// Every rounding point below follows that port unless a comment says otherwise.
//
// Architecture (24 layers, hidden 2304, 128 experts top-8 on every layer):
//   * no RoPE: a content-conditioned relative bias gathered at q_pos - k_pos from ABSOLUTE
//     positions, zero outside [0, rel_extent). Keys therefore stay in strict temporal order;
//     a ring buffer that rotates slot order would rotate the bias.
//   * 20 sliding layers (window 1024) + 4 full layers {5, 11, 17, 23}; per-TOKEN attention
//     sinks with a zero value; q scaled-L2-clip, k L2-clip, v unweighted RMSNorm;
//     output = o_proj(attn * sigmoid(gate_proj(h))). Full layers take separate q/k/v inputs
//     from a 3-query block attention residual, use dynamic k/v short convs and log-scale q and
//     the relative states past 8192 tokens.
//   * block attention residuals every 6 layers instead of a plain residual stream.
//   * 4 residual short convs per layer (k, v, post-attention, post-MLP), fp32.
//   * MoE: sigmoid router + selection bias, weights = selected scores / sum * 8, fp32
//     accumulation; activation softcap(g, 4) * sigmoid(g) * softcap(u, 25) (NOT SwiGLU).
//   * THE LOOP: layers [6, 18) run 3 times with the same weights. Passes 2..3 restart from the
//     pre-loop hidden + block states; every loop layer gets 4 messages (q, k, v, mlp) read by
//     `message_input` from the previous pass, mixed in by rank-64 gates. Layers 18..23 continue
//     from the last pass. Every pass owns its own KV + conv state for the loop layers.
//
// Cache layout (96 entries): `[0, 48)` hold one KV cache per (layer, pass) slot
// (`cacheSlot(layer:pass:)`): `KVCacheSimple` (full history) for the full-attention visits and
// `TemporalWindowKVCache` (logical position, only the last window + rollback rows held, temporal
// order for the relative bias) for the sliding ones; `[48, 96)` are single-slot `MambaCache`s
// holding that slot's four short-conv states packed as one fp32 `[B, K-1, 2*kvDim + 2*hidden]`
// array. The coordinator stores the windowed slots as model-owned disk state, so a stored prefix
// costs the window, not the history, per sliding visit. Attention takes the last
// `window - 1 + T` rows `update` returns and derives key positions from the logical offset.
// VMLX_INKLING_WINDOWED_KV=0 keeps every slot a full-history `KVCacheSimple` (A/B control arm).

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Configuration

public struct Inkling2LoopConfiguration: Codable, Sendable {
    var modelType: String = "inkling2_loop"
    var hiddenSize: Int = 2304
    var numHiddenLayers: Int = 24
    var vocabSize: Int = 201024
    var unpaddedVocabSize: Int? = 200058
    var numAttentionHeads: Int = 24
    var numKeyValueHeads: Int = 8
    var headDim: Int = 128
    var swaNumAttentionHeads: Int?
    var swaNumKeyValueHeads: Int?
    var swaHeadDim: Int?
    var rmsNormEps: Float = 1e-6
    var finalLogitSoftcapping: Float? = 30
    var dRel: Int = 16
    var relExtent: Int = 1024
    var slidingWindowSize: Int = 1024
    var localLayerIds: [Int]?
    var logScalingNFloor: Float = 8192
    var logScalingAlpha: Float = 0.1
    var qkL2ClipThreshold: Float = 22.627416997969522
    var relL2ClipThreshold: Float = 8.0
    var sconvKernelSize: Int = 4
    var nLayersPerAttnResBlock: Int = 6
    var nRoutedExperts: Int = 128
    var numExpertsPerTok: Int = 8
    var intermediateSize: Int = 1152
    var mlpGateSoftcap: Float = 4
    var mlpUpSoftcap: Float = 25
    var routeScale: Float = 8
    var loopStart: Int = 6
    var loopEnd: Int = 18
    var loopPasses: Int = 3
    var loopGateRank: Int = 64
    var eosTokenId: [Int]?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case vocabSize = "vocab_size"
        case unpaddedVocabSize = "unpadded_vocab_size"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case swaNumAttentionHeads = "swa_num_attention_heads"
        case swaNumKeyValueHeads = "swa_num_key_value_heads"
        case swaHeadDim = "swa_head_dim"
        case rmsNormEps = "rms_norm_eps"
        case finalLogitSoftcapping = "final_logit_softcapping"
        case dRel = "d_rel"
        case relExtent = "rel_extent"
        case slidingWindowSize = "sliding_window_size"
        case localLayerIds = "local_layer_ids"
        case logScalingNFloor = "log_scaling_n_floor"
        case logScalingAlpha = "log_scaling_alpha"
        case qkL2ClipThreshold = "qk_l2_clip_threshold"
        case relL2ClipThreshold = "rel_l2_clip_threshold"
        case sconvKernelSize = "sconv_kernel_size"
        case nLayersPerAttnResBlock = "n_layers_per_attn_res_block"
        case nRoutedExperts = "n_routed_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case intermediateSize = "intermediate_size"
        case mlpGateSoftcap = "mlp_gate_softcap"
        case mlpUpSoftcap = "mlp_up_softcap"
        case routeScale = "route_scale"
        case loopConfig = "loop_config"
        case textConfig = "text_config"
        case eosTokenId = "eos_token_id"
        case mtpConfig = "mtp_config"
        case visionConfig = "vision_config"
        case imageTokenId = "image_token_id"
    }

    struct MTPConfig: Codable {
        var hiddenSize: Int?
        var numHiddenLayers: Int?
        var numAttentionHeads: Int?
        var numKeyValueHeads: Int?
        var headDim: Int?
        var intermediateSize: Int?
        var rmsNormEps: Float?
        var ropeTheta: Float?
        var localLayerIds: [Int]?
        var slidingWindowSize: Int?
        var blockSize: Int?
        var markovRank: Int?
        var markovHeadType: String?
        var targetLayerIds: [Int]?
        enum CodingKeys: String, CodingKey {
            case hiddenSize = "hidden_size"
            case numHiddenLayers = "num_hidden_layers"
            case numAttentionHeads = "num_attention_heads"
            case numKeyValueHeads = "num_key_value_heads"
            case headDim = "head_dim"
            case intermediateSize = "intermediate_size"
            case rmsNormEps = "rms_norm_eps"
            case ropeTheta = "rope_theta"
            case localLayerIds = "local_layer_ids"
            case slidingWindowSize = "sliding_window_size"
            case blockSize = "block_size"
            case markovRank = "markov_rank"
            case markovHeadType = "markov_head_type"
            case targetLayerIds = "target_layer_ids"
        }
    }

    /// The native DSpark drafter (`mtp_config` + `mtp.safetensors`), when the bundle declares one
    /// with the `rnn_hidden` Markov head this runtime implements.
    var dspark: Inkling2DSparkConfig?
    /// The vision tower (`vision_config`), nil for a text-only bundle or construction
    /// (`VMLX_INKLING2_VL=0`).
    public var vision: Inkling2VisionConfiguration?

    struct LoopConfig: Codable {
        var coreStart: Int?
        var coreEnd: Int?
        var numPasses: Int?
        var gateRank: Int?
        var isVarloop: Bool?
        enum CodingKeys: String, CodingKey {
            case coreStart = "core_start"
            case coreEnd = "core_end"
            case numPasses = "num_passes"
            case gateRank = "gate_rank"
            case isVarloop = "is_varloop"
        }
    }

    public init() {}

    public init(from decoder: Decoder) throws {
        let outer = try decoder.container(keyedBy: CodingKeys.self)
        // The released config wraps the text model in `text_config` (multimodal wrapper,
        // top-level `model_type: inkling2_loop`); a bare text config is accepted too.
        let c =
            outer.contains(.textConfig)
            ? try outer.nestedContainer(keyedBy: CodingKeys.self, forKey: .textConfig) : outer
        func v<T: Decodable>(_ key: CodingKeys, _ fallback: T) throws -> T {
            try c.decodeIfPresent(T.self, forKey: key) ?? fallback
        }
        modelType = try outer.decodeIfPresent(String.self, forKey: .modelType) ?? "inkling2_loop"
        hiddenSize = try v(.hiddenSize, 2304)
        numHiddenLayers = try v(.numHiddenLayers, 24)
        vocabSize = try v(.vocabSize, 201024)
        unpaddedVocabSize = try c.decodeIfPresent(Int.self, forKey: .unpaddedVocabSize)
        numAttentionHeads = try v(.numAttentionHeads, 24)
        numKeyValueHeads = try v(.numKeyValueHeads, 8)
        headDim = try v(.headDim, 128)
        swaNumAttentionHeads = try c.decodeIfPresent(Int.self, forKey: .swaNumAttentionHeads)
        swaNumKeyValueHeads = try c.decodeIfPresent(Int.self, forKey: .swaNumKeyValueHeads)
        swaHeadDim = try c.decodeIfPresent(Int.self, forKey: .swaHeadDim)
        rmsNormEps = try v(.rmsNormEps, 1e-6)
        finalLogitSoftcapping = try c.decodeIfPresent(Float.self, forKey: .finalLogitSoftcapping)
        dRel = try v(.dRel, 16)
        relExtent = try v(.relExtent, 1024)
        slidingWindowSize = try v(.slidingWindowSize, 1024)
        localLayerIds = try c.decodeIfPresent([Int].self, forKey: .localLayerIds)
        // The transformers config default is 128000; the shipped config says 8192. Always read it.
        logScalingNFloor = try v(.logScalingNFloor, 128000)
        logScalingAlpha = try v(.logScalingAlpha, 0.1)
        qkL2ClipThreshold = try v(.qkL2ClipThreshold, 22.627416997969522)
        relL2ClipThreshold = try v(.relL2ClipThreshold, 8.0)
        sconvKernelSize = try v(.sconvKernelSize, 4)
        nLayersPerAttnResBlock = try v(.nLayersPerAttnResBlock, 6)
        nRoutedExperts = try v(.nRoutedExperts, 128)
        numExpertsPerTok = try v(.numExpertsPerTok, 8)
        intermediateSize = try v(.intermediateSize, 1152)
        mlpGateSoftcap = try v(.mlpGateSoftcap, 4)
        mlpUpSoftcap = try v(.mlpUpSoftcap, 25)
        routeScale = try v(.routeScale, 8)
        let loop = try c.decodeIfPresent(LoopConfig.self, forKey: .loopConfig)
        // transformers PR #51050 Inkling2LoopTextConfig defaults.
        loopStart = loop?.coreStart ?? 6
        loopEnd = loop?.coreEnd ?? 18
        loopPasses = max(loop?.numPasses ?? 3, 1)
        loopGateRank = loop?.gateRank ?? 64
        if let m = try? outer.decodeIfPresent(MTPConfig.self, forKey: .mtpConfig),
            (m.markovHeadType ?? "rnn_hidden") == "rnn_hidden"
        {
            var d = Inkling2DSparkConfig()
            d.hidden = m.hiddenSize ?? hiddenSize
            d.layers = m.numHiddenLayers ?? 3
            d.nq = m.numAttentionHeads ?? numAttentionHeads
            d.nkv = m.numKeyValueHeads ?? numKeyValueHeads
            d.hd = m.headDim ?? headDim
            d.inter = m.intermediateSize ?? 9216
            d.eps = m.rmsNormEps ?? rmsNormEps
            d.theta = m.ropeTheta ?? 1_000_000
            d.localLayers = m.localLayerIds ?? [0, 2]
            d.window = m.slidingWindowSize ?? 1024
            d.blockSize = m.blockSize ?? 8
            d.markovRank = m.markovRank ?? 256
            d.vocab = vocabSize
            d.unpaddedVocab = unpaddedVocabSize ?? vocabSize
            d.softcap = finalLogitSoftcapping ?? 30
            d.targetLayerIds = m.targetLayerIds ?? [5, 23, 41, 48]
            dspark = d
        }
        if Inkling2VisionConfiguration.enabled,
            var v = try? outer.decodeIfPresent(Inkling2VisionConfiguration.self, forKey: .visionConfig)
        {
            v.textHiddenSize = hiddenSize
            if let id = try? outer.decodeIfPresent(Int.self, forKey: .imageTokenId) { v.imageTokenId = id }
            vision = v
        }
        if let one = try? outer.decodeIfPresent(Int.self, forKey: .eosTokenId) {
            eosTokenId = [one]
        } else {
            eosTokenId = try? outer.decodeIfPresent([Int].self, forKey: .eosTokenId)
        }
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(modelType, forKey: .modelType)
        try c.encode(hiddenSize, forKey: .hiddenSize)
        try c.encode(numHiddenLayers, forKey: .numHiddenLayers)
        try c.encode(vocabSize, forKey: .vocabSize)
    }

    // MARK: derived topology

    var numLoopLayers: Int { loopEnd - loopStart }

    func isLoopLayer(_ i: Int) -> Bool { loopPasses > 1 && i >= loopStart && i < loopEnd }

    /// Physical layers + one extra copy of every loop layer per extra pass.
    var numCacheSlots: Int { numHiddenLayers + (loopPasses - 1) * numLoopLayers }

    /// Slot of `layer` in loop pass `pass` (0-based): pass 0 and non-loop layers use the physical
    /// slot; pass p >= 1 of a loop layer uses `layers + (p-1)*loopLayers + (layer - loopStart)`,
    /// i.e. the reference's `layer_idx + cache_offset`.
    func cacheSlot(layer: Int, pass: Int) -> Int {
        if pass == 0 || !isLoopLayer(layer) { return layer }
        return numHiddenLayers + (pass - 1) * numLoopLayers + layer - loopStart
    }

    func sourceLayer(ofSlot slot: Int) -> Int {
        slot < numHiddenLayers
            ? slot : loopStart + (slot - numHiddenLayers) % numLoopLayers
    }

    func isSliding(_ i: Int) -> Bool {
        if let localLayerIds { return localLayerIds.contains(i) }
        return (i + 1) % 6 != 0
    }

    func isBlockBoundary(_ i: Int) -> Bool { i % nLayersPerAttnResBlock == 0 }

    func heads(_ i: Int) -> (q: Int, kv: Int, dim: Int) {
        if isSliding(i) {
            return (
                swaNumAttentionHeads ?? numAttentionHeads, swaNumKeyValueHeads ?? numKeyValueHeads,
                swaHeadDim ?? headDim
            )
        }
        return (numAttentionHeads, numKeyValueHeads, headDim)
    }

    /// Width of one slot's packed conv state: k conv, v conv, post-attention, post-MLP.
    var convStateWidth: Int {
        let (_, kv, d) = heads(0)
        return 2 * kv * d + 2 * hiddenSize
    }
}

// MARK: - Numerics helpers (fp32, reference rounding points)

private let f32 = DType.float32

/// Runtime switch (diagnostics / parity A/B): run single-token steps through the general path.
public enum Inkling2Runtime {
    nonisolated(unsafe) public static var forceGeneralDecode = false
}

/// Fused Metal kernels for the small per-token blocks (`Inkling2Kernels`). `VMLX_INKLING2_KERNELS=0`
/// runs the op-by-op reference path instead (parity A/B).
let inklingUseKernels = ProcessInfo.processInfo.environment["VMLX_INKLING2_KERNELS"] != "0"

/// `Inkling2RMSNorm`: fp32 statistics, `(w_f32 * normed_f32).astype(x.dtype)`, no +1 shift.
/// `MLXFast.rmsNorm` on fp32 input with an fp32 weight is one rounding at the end, like the
/// reference.
final class Inkling2RMSNorm: Module {
    var weight: MLXArray
    let eps: Float

    init(_ dims: Int, eps: Float) {
        weight = MLXArray.ones([dims])
        self.eps = eps
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x.asType(f32), weight: weight.asType(f32), eps: eps).asType(x.dtype)
    }
}

/// `Inkling2UnweightedRMSNorm`: fp32 `x * rsqrt(mean(x^2) + eps)`, back to `x.dtype`.
@inline(__always)
private func unweightedRMS(_ x: MLXArray, ones: MLXArray, eps: Float = 1e-6) -> MLXArray {
    MLXFast.rmsNorm(x.asType(f32), weight: ones, eps: eps).asType(x.dtype)
}

/// Scale each head vector down to `threshold` when longer (optionally after a learned
/// per-head scale); fp32, back to `x.dtype`.
@inline(__always)
private func l2Clip(_ x: MLXArray, threshold: Float, scale: MLXArray? = nil) -> MLXArray {
    var xf = x.asType(f32)
    if let scale { xf = xf * scale.asType(f32) }
    let norm = MLX.sqrt(MLX.sum(xf * xf, axis: -1, keepDims: true))
    return (xf * MLX.minimum(threshold / (norm + 1e-6), 1)).asType(x.dtype)
}

@inline(__always)
private func softcap(_ x: MLXArray, _ cap: Float) -> MLXArray {
    MLX.tanh(x / cap) * cap
}

// MARK: - Short convolutions (residual, scaled, fp32; state = last K-1 PRE-conv inputs)

/// A layer's four conv states live in ONE packed fp32 `[B, K-1, P]` array; each conv reads its
/// columns `offset ..< offset + C` and returns its new `[B, K-1, C]` state.
private func convState(_ packed: MLXArray?, offset: Int, like x: MLXArray, kernel: Int) -> MLXArray
{
    packed?[0..., 0..., offset ..< (offset + x.dim(2))]
        ?? MLXArray.zeros([x.dim(0), kernel - 1, x.dim(2)], dtype: f32)
}

/// Static depthwise causal conv: `(conv(x) + x) / sqrt(K + 1)`. `weight` is `[K, C]` (fp32 after
/// sanitize), tap `K - 1` weighting the CURRENT token (torch conv1d with left padding K-1).
final class Inkling2ShortConv: Module {
    var weight: MLXArray
    let kernel: Int

    init(channels: Int, kernel: Int) {
        weight = MLXArray.zeros([kernel, channels])
        self.kernel = kernel
    }

    /// Returns `(output in x.dtype, new state [B, K-1, C] fp32)`.
    func callAsFunction(_ x: MLXArray, packed: MLXArray?, offset: Int) -> (MLXArray, MLXArray) {
        if inklingUseKernels, kernel == 4 {
            let st = packed ?? MLXArray.zeros([x.dim(0), 3, offset + x.dim(2)], dtype: f32)
            return Inkling2Kernels.conv(x, packed: st, offset: offset, weight: weight)
        }
        let K = kernel
        let pad = K - 1
        let T = x.dim(1)
        let xf = x.asType(f32)
        let xp = MLX.concatenated(
            [convState(packed, offset: offset, like: x, kernel: K), xf], axis: 1)
        let w = weight.asType(f32)
        var y = xp[0..., 0 ..< T, 0...] * w[0]
        for j in 1 ..< K {
            y = y + xp[0..., j ..< (j + T), 0...] * w[j]
        }
        let out = ((y + xf) / Float(K + 1).squareRoot()).asType(x.dtype)
        return (out, xp[0..., (xp.dim(1) - pad)..., 0...])
    }
}

extension Inkling2ShortConv {
    /// The conv fused with the layer's residual arithmetic (see `Inkling2Kernels.residualConv`).
    /// mode 0: fp32(conv); 1: fp32(res) + fp32(conv); 2: dtype(res + fp32(conv)).
    func residual(
        _ x: MLXArray, packed: MLXArray?, offset: Int, res: MLXArray, mode: Int, outDType: DType
    ) -> (MLXArray, MLXArray) {
        if inklingUseKernels, kernel == 4 {
            let (B, T, C) = (x.dim(0), x.dim(1), x.dim(2))
            let st = packed ?? MLXArray.zeros([B, 3, offset + C], dtype: f32)
            let outs = Inkling2Kernels.residualConv(
                [x, st, weight, res, MLXArray([Int32(offset), Int32(mode)])],
                grid: (C, B, 1), threadGroup: (min(256, C), 1, 1),
                outputShapes: [[B, T, C], [B, 3, C]], outputDTypes: [outDType, f32])
            return (outs[0], outs[1])
        }
        let (a, state) = self(x, packed: packed, offset: offset)
        switch mode {
        case 0: return (a.asType(f32), state)
        case 1: return (res.asType(f32) + a.asType(f32), state)
        default: return ((res.asType(f32) + a.asType(f32)).asType(outDType), state)
        }
    }
}

/// Full-layer k/v conv (`Inkling2DynamicShortConvolution`): taps = `static_weight` `[K, C]`
/// (index = BACKWARD distance: tap 0 = current token) + per-token `weight_proj(dyn)` shared by
/// the channels of each kv head. Residual, divided by `sqrt(2K + 1)`, fp32.
final class Inkling2DynamicShortConv: Module {
    @ParameterInfo(key: "static_weight") var staticWeight: MLXArray
    @ModuleInfo(key: "weight_proj") var weightProj: Linear
    let kernel: Int
    let heads: Int

    init(channels: Int, inputSize: Int, heads: Int, kernel: Int) {
        _staticWeight.wrappedValue = MLXArray.zeros([kernel, channels])
        _weightProj.wrappedValue = Linear(inputSize, heads * kernel, bias: false)
        self.kernel = kernel
        self.heads = heads
    }

    func callAsFunction(_ x: MLXArray, dynamicInput: MLXArray, packed: MLXArray?, offset: Int)
        -> (MLXArray, MLXArray)
    {
        let K = kernel
        let pad = K - 1
        let (B, T, C) = (x.dim(0), x.dim(1), x.dim(2))
        let nh = heads
        if inklingUseKernels, K == 4 {
            let st = packed ?? MLXArray.zeros([B, 3, offset + C], dtype: f32)
            return Inkling2Kernels.conv(
                x, packed: st, offset: offset, weight: staticWeight, dyn: weightProj(dynamicInput))
        }
        let dyn = weightProj(dynamicInput).asType(f32).reshaped([B, T, nh, 1, K])
        let st = staticWeight.asType(f32)
        let xf = x.asType(f32)
        let xp = MLX.concatenated(
            [convState(packed, offset: offset, like: x, kernel: K), xf], axis: 1)
        var y: MLXArray? = nil
        for lag in 0 ..< K {
            let xl = xp[0..., (pad - lag) ..< (pad - lag + T), 0...]  // token `lag` steps back
            let w = st[lag].reshaped([nh, C / nh]) + dyn[.ellipsis, lag]  // [B, T, nh, C/nh]
            let term = (xl.reshaped([B, T, nh, C / nh]) * w).reshaped([B, T, C])
            y = y.map { $0 + term } ?? term
        }
        let out = ((y! + xf) / Float(2 * K + 1).squareRoot()).asType(x.dtype)
        return (out, xp[0..., (xp.dim(1) - pad)..., 0...])
    }
}

// MARK: - Attention

final class Inkling2RelLogits: Module {
    var proj: MLXArray
    let extent: Int

    init(dRel: Int, extent: Int) {
        proj = MLXArray.zeros([dRel, extent])
        self.extent = extent
    }
}

final class Inkling2ScaledL2Clip: Module {
    var scale: MLXArray
    init(heads: Int, dim: Int) { scale = MLXArray.ones([heads, dim]) }
}

final class Inkling2Attention: Module {
    let layer: Int
    let isSliding: Bool
    let nq: Int
    let nkv: Int
    let hd: Int
    let dRel: Int
    let window: Int?
    let relExtent: Int
    let scale: Float
    let logFloor: Float
    let logAlpha: Float
    let qkClip: Float
    let relClip: Float
    let eps: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "r_proj") var rProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "sink_proj") var sinkProj: Linear
    @ModuleInfo(key: "k_sconv") var kSconv: Module
    @ModuleInfo(key: "v_sconv") var vSconv: Module
    @ModuleInfo(key: "rel_logits_proj") var relLogits: Inkling2RelLogits
    @ModuleInfo(key: "q_l2_clip") var qL2Clip: Inkling2ScaledL2Clip
    @ModuleInfo(key: "rel_l2_clip") var relL2Clip: Inkling2ScaledL2Clip

    init(_ args: Inkling2LoopConfiguration, layer: Int) {
        self.layer = layer
        isSliding = args.isSliding(layer)
        (nq, nkv, hd) = args.heads(layer)
        dRel = args.dRel
        window = isSliding ? args.slidingWindowSize : nil
        // Sliding layers band the bias to the window; full layers use rel_extent.
        relExtent = isSliding ? args.slidingWindowSize : args.relExtent
        scale = 1 / Float(hd).squareRoot()
        logFloor = args.logScalingNFloor
        logAlpha = args.logScalingAlpha
        qkClip = args.qkL2ClipThreshold
        relClip = args.relL2ClipThreshold
        eps = args.rmsNormEps
        let d = args.hiddenSize
        _qProj.wrappedValue = Linear(d, nq * hd, bias: false)
        _kProj.wrappedValue = Linear(d, nkv * hd, bias: false)
        _vProj.wrappedValue = Linear(d, nkv * hd, bias: false)
        _rProj.wrappedValue = Linear(d, nq * dRel, bias: false)
        _oProj.wrappedValue = Linear(nq * hd, d, bias: false)
        _gateProj.wrappedValue = Linear(d, nq * hd, bias: false)
        _sinkProj.wrappedValue = Linear(d, nq, bias: false)
        let K = args.sconvKernelSize
        if isSliding {
            _kSconv.wrappedValue = Inkling2ShortConv(channels: nkv * hd, kernel: K)
            _vSconv.wrappedValue = Inkling2ShortConv(channels: nkv * hd, kernel: K)
        } else {
            _kSconv.wrappedValue = Inkling2DynamicShortConv(
                channels: nkv * hd, inputSize: d, heads: nkv, kernel: K)
            _vSconv.wrappedValue = Inkling2DynamicShortConv(
                channels: nkv * hd, inputSize: d, heads: nkv, kernel: K)
        }
        _relLogits.wrappedValue = Inkling2RelLogits(dRel: dRel, extent: relExtent)
        _qL2Clip.wrappedValue = Inkling2ScaledL2Clip(heads: nq, dim: hd)
        _relL2Clip.wrappedValue = Inkling2ScaledL2Clip(heads: nq, dim: dRel)
        clipThresholds = MLXArray([
            args.qkL2ClipThreshold, args.relL2ClipThreshold, args.rmsNormEps,
        ])
    }

    /// `[q/k clip, rel clip, v-norm eps]` for `Inkling2Kernels.attnClips`.
    let clipThresholds: MLXArray

    private let fusedXBox = Inkling2DerivedBox()
    private let fusedXKVBox = Inkling2DerivedBox()

    /// One linear map for the x-fed projections, or nil when their storage differs (mixed
    /// quantization). Plain weights are concatenated; quantized ones only when every part shares
    /// bits / group size / mode, concatenating packed rows, scales and biases.
    func fusedProjection(includeKV: Bool) -> ((MLXArray) -> MLXArray)? {
        let parts: [Linear] =
            [qProj, rProj, gateProj, sinkProj] + (includeKV ? [kProj, vProj] : [])
        let box = includeKV ? fusedXKVBox : fusedXBox
        if let q0 = parts[0] as? QuantizedLinear {
            let qs = parts.compactMap { $0 as? QuantizedLinear }
            guard qs.count == parts.count,
                qs.allSatisfy({
                    $0.bits == q0.bits && $0.groupSize == q0.groupSize && $0.mode == q0.mode
                        && ($0.biases == nil) == (q0.biases == nil) && $0.bias == nil
                })
            else { return nil }
            if box.value == nil {
                box.value = MLX.concatenated(qs.map(\.weight), axis: 0)
                box.extra = MLX.concatenated(qs.map(\.scales), axis: 0)
                box.extra2 =
                    q0.biases == nil ? nil : MLX.concatenated(qs.map { $0.biases! }, axis: 0)
            }
            let (w, sc, bi) = (box.value!, box.extra!, box.extra2)
            let (gs, bits, mode) = (q0.groupSize, q0.bits, q0.mode)
            return { x in
                MLX.quantizedMM(
                    x, w, scales: sc, biases: bi, transpose: true, groupSize: gs, bits: bits,
                    mode: mode)
            }
        }
        guard parts.allSatisfy({ type(of: $0) == Linear.self && $0.bias == nil }) else {
            return nil
        }
        if box.value == nil {
            // [out, in] row-major like `Linear.weight`: MLX's GEMV is fastest on its transpose.
            box.value = MLX.concatenated(parts.map(\.weight), axis: 0)
        }
        let w = box.value!
        return { x in MLX.matmul(x, w.T) }
    }

    /// Projections, convs, clips. `x` feeds q / r / gate / sinks; `keyInput` / `valueInput` feed
    /// k / v (nil → `x`; full layers always pass them). Returns
    /// `[q, k, v, r, gate, sinks, newConvK, newConvV]` with q / k / v as `[B, H, T, D]`
    /// and r as `[B, T, nq, dRel]`. Pure: no cache access (compiled for decode).
    func project(
        _ x: MLXArray, keyInput: MLXArray?, valueInput: MLXArray?, packed: MLXArray?,
        ones: MLXArray
    ) -> [MLXArray] {
        let (B, T) = (x.dim(0), x.dim(1))
        let ki = keyInput ?? x
        let vi = valueInput ?? x
        var k: MLXArray
        var v: MLXArray
        let newK: MLXArray
        let newV: MLXArray
        let c = nkv * hd
        // Every projection that reads `x` in ONE GEMV ([q | r | gate | sink], plus [k | v] when
        // they read `x` too); outputs are column slices (rows are independent dot products, so
        // the values are those of the separate projections).
        let kvFromX = keyInput == nil && valueInput == nil
        var q: MLXArray
        var r: MLXArray
        let gate: MLXArray
        let sinks: MLXArray
        var kRaw: MLXArray
        var vRaw: MLXArray
        if inklingUseKernels, let fused = fusedProjection(includeKV: kvFromX) {
            let y = fused(x)
            var o = 0
            func take(_ n: Int) -> MLXArray {
                defer { o += n }
                return y[.ellipsis, o ..< (o + n)]
            }
            q = take(nq * hd)
            r = take(nq * dRel)
            gate = take(nq * hd)
            sinks = take(nq)
            kRaw = kvFromX ? take(c) : kProj(ki)
            vRaw = kvFromX ? take(c) : vProj(vi)
        } else {
            q = qProj(x)
            r = rProj(x)
            gate = gateProj(x)
            sinks = sinkProj(x)  // [B, T, nq]
            kRaw = kProj(ki)
            vRaw = vProj(vi)
        }
        if let conv = kSconv as? Inkling2ShortConv {
            (k, newK) = conv(kRaw, packed: packed, offset: 0)
            (v, newV) = (vSconv as! Inkling2ShortConv)(vRaw, packed: packed, offset: c)
        } else {
            (k, newK) = (kSconv as! Inkling2DynamicShortConv)(
                kRaw, dynamicInput: ki, packed: packed, offset: 0)
            (v, newV) = (vSconv as! Inkling2DynamicShortConv)(
                vRaw, dynamicInput: vi, packed: packed, offset: c)
        }

        if inklingUseKernels, hd <= 128, dRel <= 32 {
            let R = B * T
            let outs = Inkling2Kernels.attnClips(
                [
                    q.reshaped([R, nq * hd]), r.reshaped([R, nq * dRel]),
                    k.reshaped([R, nkv * hd]),
                    v.reshaped([R, nkv * hd]), qL2Clip.scale.asType(f32),
                    relL2Clip.scale.asType(f32), clipThresholds,
                ], grid: ((2 * nq + 2 * nkv) * 32, R, 1), threadGroup: (32, 1, 1),
                outputShapes: [[R, nq * hd], [R, nq * dRel], [R, nkv * hd], [R, nkv * hd]],
                outputDTypes: [q.dtype, r.dtype, k.dtype, v.dtype])
            return [
                outs[0].reshaped([B, T, nq, hd]).transposed(0, 2, 1, 3),
                outs[2].reshaped([B, T, nkv, hd]).transposed(0, 2, 1, 3),
                outs[3].reshaped([B, T, nkv, hd]).transposed(0, 2, 1, 3),
                outs[1].reshaped([B, T, nq, dRel]), gate, sinks, newK, newV, kRaw, vRaw,
            ]
        }
        q = l2Clip(q.reshaped([B, T, nq, hd]), threshold: qkClip, scale: qL2Clip.scale)
            .transposed(0, 2, 1, 3)
        r = l2Clip(r.reshaped([B, T, nq, dRel]), threshold: relClip, scale: relL2Clip.scale)
        k = l2Clip(k.reshaped([B, T, nkv, hd]), threshold: qkClip).transposed(0, 2, 1, 3)
        v = unweightedRMS(v.reshaped([B, T, nkv, hd]), ones: ones[0 ..< hd], eps: eps)
            .transposed(0, 2, 1, 3)
        return [q, k, v, r, gate, sinks, newK, newV, kRaw, vRaw]
    }

    /// Log scaling, cache update, relative bias and attention. q / k / v `[B, H, T, D]`,
    /// r `[B, T, nq, dRel]`, sinks `[B, T, nq]`. Returns `[B, T, nq * hd]`.
    func attend(
        q qIn: MLXArray, k: MLXArray, v: MLXArray, r rIn: MLXArray, sinks: MLXArray,
        kv: (any KVCache)?
    ) -> MLXArray {
        let (B, T) = (qIn.dim(0), qIn.dim(2))
        var q = qIn
        var r = rIn
        // Absolute positions; `seen` is read BEFORE the cache update.
        let seen = kv?.offset ?? 0
        if !isSliding, seen + T > Int(logFloor) {
            // Log scaling of q AND the relative states (after the clips, before the bias
            // gather). Exactly 1 for positions below n_floor, so it is skipped there.
            let qPos = MLXArray(Int32(seen) ..< Int32(seen + T)).asType(f32)
            let tau = (1 + logAlpha * MLX.log(MLX.maximum((qPos + 1) / logFloor, 1)))
                .reshaped([1, T, 1, 1])
            q = (q.asType(f32) * tau.transposed(0, 2, 1, 3)).asType(q.dtype)
            r = (r.asType(f32) * tau).asType(r.dtype)
        }

        var keys = k
        var values = v
        if let kv {
            (keys, values) = kv.update(keys: k, values: v)
        }
        var S = keys.dim(2)
        var kStart = seen + T - S
        if let window {
            // A query at p sees keys p-window+1 ... p, so a T-token call needs window-1+T keys.
            let visible = min(S, window - 1 + T)
            if visible < S {
                keys = keys[0..., 0..., (S - visible)..., 0...]
                values = values[0..., 0..., (S - visible)..., 0...]
                kStart += S - visible
                S = visible
            }
        }

        // Relative bias: rel[b, h, t, d] = (r @ proj)[b, t, h, d], gathered at d = q_pos - k_pos,
        // exactly zero outside 0 <= d < extent.
        let out: MLXArray
        if T == 1 {
            // Decode: keys sit at kStart ... seen, distance runs S-1 ... 0.
            var bias: MLXArray
            if inklingUseKernels, B == 1 {
                bias = Inkling2Kernels.decodeBias(
                    [
                        r.reshaped([nq, dRel]), relLogits.proj.asType(r.dtype),
                        MLXArray([Int32(S)]),
                    ],
                    grid: (S, nq, 1), threadGroup: (min(256, S), 1, 1), outputShapes: [[nq, S]],
                    outputDTypes: [r.dtype])[0].reshaped([1, nq, 1, S])
            } else {
                let rel = MLX.matmul(r, relLogits.proj.asType(r.dtype)).transposed(0, 2, 1, 3)
                let n = min(S, relExtent)
                let idx = MLXArray((0 ..< Int32(n)).reversed())  // n-1 ... 0
                bias = MLX.take(rel, idx, axis: -1)  // [B, H, 1, n]
                if n < S {
                    bias = MLX.concatenated(
                        [MLXArray.zeros([B, nq, 1, S - n], dtype: bias.dtype), bias], axis: -1)
                }
            }
            // A single query's per-token sink IS a per-head constant: the fused kernel's sinks.
            out = MLXFast.scaledDotProductAttention(
                queries: q, keys: keys, values: values, scale: scale,
                mask: .array(bias.asType(q.dtype)),
                sinks: sinks.reshaped([nq]).asType(q.dtype))
        } else {
            let qPos = MLXArray(Int32(seen) ..< Int32(seen + T))
            let kPos = MLXArray(Int32(kStart) ..< Int32(kStart + S))
            let d = qPos.expandedDimensions(axis: 1) - kPos.expandedDimensions(axis: 0)  // [T,S]
            let idx = MLX.clip(d, min: 0, max: relExtent - 1)
            let rel = MLX.matmul(r, relLogits.proj.asType(r.dtype)).transposed(0, 2, 1, 3)
            let gathered = MLX.takeAlong(
                rel, MLX.broadcast(idx.reshaped([1, 1, T, S]), to: [B, nq, T, S]), axis: -1)
            var valid = d .>= 0
            if let window { valid = valid .&& (d .< window) }
            let inBand = (d .>= 0) .&& (d .< relExtent)
            let bias = MLX.where(inBand, gathered, MLXArray(Float(0)).asType(gathered.dtype))
            let negInf = MLXArray(-Float.infinity).asType(q.dtype)
            let additive = MLX.where(valid, bias.asType(q.dtype), negInf)
            // Per-token sinks (zero value): one extra all-zero key whose additive mask entry is
            // the sink logit, so q·0·scale + sink = sink and its value contributes 0.
            let sinkCol = sinks.asType(q.dtype).transposed(0, 2, 1).expandedDimensions(axis: -1)
            let mask = MLX.concatenated([sinkCol, additive], axis: -1)  // [B, nq, T, S+1]
            let zk = MLXArray.zeros([B, nkv, 1, hd], dtype: keys.dtype)
            out = MLXFast.scaledDotProductAttention(
                queries: q, keys: MLX.concatenated([zk, keys], axis: 2),
                values: MLX.concatenated([zk.asType(values.dtype), values], axis: 2),
                scale: scale, mask: .array(mask))
        }
        return out.transposed(0, 2, 1, 3).reshaped([B, T, nq * hd])
    }

    /// `o_proj(attn * sigmoid(gate))`, gate in fp32.
    func finish(_ merged: MLXArray, gate: MLXArray) -> MLXArray {
        if inklingUseKernels {
            return oProj(
                Inkling2Kernels.sigmoidGate(
                    [merged, gate.reshaped(merged.shape)], grid: (merged.size, 1, 1),
                    threadGroup: (256, 1, 1), outputShapes: [merged.shape],
                    outputDTypes: [merged.dtype])[0])
        }
        return oProj((merged.asType(f32) * MLX.sigmoid(gate.asType(f32))).asType(merged.dtype))
    }
}

// MARK: - MoE

/// Routed experts kept in the bundle's own encoding, run by `gatherQuantizedMM`.
///
/// * NVFP4 (ModelOpt or MLX): `mode: .nvfp4` (packed e2m1, even element in the low nibble, e4m3
///   per-16 scales). `gather_qmm` has no per-expert global scale, so the per-expert factor
///   (`amax / outer_scale_denominator`, MLX `global_scale / 2688`, ModelOpt `weight_scale_2`) is
///   applied to the OUTPUT rows: `x @ (W_e * s_e)^T == s_e * (x @ W_e^T)` for a scalar `s_e`.
/// * MXFP8 / MXFP4: `mode: .mxfp8` / `.mxfp4` (ue8m0 per-32 scales), no extra factor.
/// * Affine (community MLX conversions, any bits / group size): `mode: .affine` with `biases`.
///
/// The format is read off the loaded tensors (scale dtype and the packed widths against the
/// input width), never off a config value that a converter may have left stale. Parameter names
/// avoid `weight` / `scales` so the generic loader never re-wraps this module as a
/// `QuantizedSwitchLinear`.
final class Inkling2NativeSwitchLinear: Module {
    var codes: MLXArray
    @ParameterInfo(key: "block_scales") var blockScales: MLXArray
    @ParameterInfo(key: "expert_scale") var expertScale: MLXArray
    /// Affine experts' per-group biases; an empty placeholder for FP8/FP4 experts (`update()` cannot
    /// set an array onto a nil parameter, so the slot always holds an array).
    @ParameterInfo(key: "affine_biases") var affineBiases: MLXArray
    let inputDims: Int

    init(inputDims: Int, outputDims: Int, experts: Int) {
        self.inputDims = inputDims
        codes = MLXArray.zeros([experts, outputDims, inputDims / 8], dtype: .uint32)
        _blockScales.wrappedValue = MLXArray.zeros(
            [experts, outputDims, inputDims / 16], dtype: .uint8)
        _expertScale.wrappedValue = MLXArray.ones([experts], dtype: f32)
        _affineBiases.wrappedValue = MLXArray.zeros([0], dtype: .bfloat16)
    }

    struct Format: Equatable {
        let mode: QuantizationMode
        let bits: Int
        let groupSize: Int
    }

    var format: Format { Self.format(codes: codes, scales: blockScales, inputDims: inputDims) }

    /// The quantization of a packed `[..., out, in * bits / 32]` uint32 matrix with per-group
    /// `scales`: a uint8 scale is an FP8 block scale (NVFP4 per 16 / MXFP4 per 32 with 4-bit codes,
    /// MXFP8 per 32 with 8-bit codes); a float scale is affine (bits and group from the widths).
    static func format(codes: MLXArray, scales: MLXArray, inputDims: Int) -> Format {
        let packed = codes.dim(-1)
        let groups = scales.dim(-1)
        let bits = packed * 32 / max(inputDims, 1)
        let group = inputDims / max(groups, 1)
        if scales.dtype == .uint8 {
            if bits == 8 { return Format(mode: .mxfp8, bits: 8, groupSize: 32) }
            return group == 16
                ? Format(mode: .nvfp4, bits: 4, groupSize: 16)
                : Format(mode: .mxfp4, bits: 4, groupSize: 32)
        }
        return Format(mode: .affine, bits: bits, groupSize: group)
    }

    var isNVFP4: Bool { format.mode == .nvfp4 }

    func callAsFunction(_ x: MLXArray, _ indices: MLXArray, sorted: Bool) -> MLXArray {
        let f = format
        if f.mode == .affine, x.dtype != blockScales.dtype {
            // MLX's affine gather has no kernel for fp32 activations with half-precision scales and
            // falls back to a far slower path (measured 699 vs 301 us for k = 8 on M5 Max): run it
            // in the scale dtype and hand the caller its own dtype back. FP8/FP4 experts keep fp32.
            return MLX.gatherQuantizedMM(
                x.asType(blockScales.dtype), codes, scales: blockScales,
                biases: affineBiases.size > 0 ? affineBiases : nil,
                rhsIndices: indices, transpose: true,
                groupSize: f.groupSize, bits: f.bits, mode: .affine, sortedIndices: sorted
            ).asType(x.dtype)
        }
        return MLX.gatherQuantizedMM(
            x, codes, scales: blockScales,
            biases: f.mode == .affine && affineBiases.size > 0 ? affineBiases : nil,
            rhsIndices: indices, transpose: true,
            groupSize: f.groupSize, bits: f.bits, mode: f.mode, sortedIndices: sorted)
    }
}

final class Inkling2SwitchMLP: Module {
    /// Fused gate/up exactly as the checkpoint stores it: rows ROW-INTERLEAVED (row 2i = gate_i,
    /// row 2i+1 = up_i). One gather produces both; the output's last axis is de-interleaved.
    @ModuleInfo(key: "gate_up_proj") var gateUp: Inkling2NativeSwitchLinear
    @ModuleInfo(key: "down_proj") var down: Inkling2NativeSwitchLinear
    let inter: Int
    let gateCap: Float
    let upCap: Float
    let caps: MLXArray

    init(_ args: Inkling2LoopConfiguration) {
        let (H, I, E) = (args.hiddenSize, args.intermediateSize, args.nRoutedExperts)
        _gateUp.wrappedValue = Inkling2NativeSwitchLinear(
            inputDims: H, outputDims: 2 * I, experts: E)
        _down.wrappedValue = Inkling2NativeSwitchLinear(inputDims: I, outputDims: H, experts: E)
        inter = I
        gateCap = args.mlpGateSoftcap
        upCap = args.mlpUpSoftcap
        caps = MLXArray([args.mlpGateSoftcap, args.mlpUpSoftcap])
    }

    /// `x` `[N, H]` (model dtype), `inds` `[N, k]`, route weights `w` `[N, k]` fp32 → `[N, H]` in the
    /// model dtype: `sum_k expert_k(x) * w_k`, fp32 accumulation.
    func callAsFunction(_ x: MLXArray, _ inds: MLXArray, _ w: MLXArray) -> MLXArray {
        let (N, k) = (inds.dim(0), inds.dim(1))
        // fp32 activations into the quantized matmul: the bf16 values are exact in fp32, the
        // kernel accumulates in fp32 either way, and the per-expert factor then multiplies an
        // unrounded product (the reference folds it into the weights before the matmul).
        var xs = x.asType(f32).reshaped([N, 1, 1, x.dim(-1)])
        var idx = inds
        var inverse: MLXArray? = nil
        let doSort = inds.size >= 64
        if doSort {
            let flat = inds.flattened()
            let order = MLX.argSort(flat)
            inverse = MLX.argSort(order)
            xs = xs.flattened(start: 0, end: -3)[order.floorDivide(k)]
            idx = flat[order]
        }
        let gu = gateUp(xs, idx, sorted: doSort)  // [..., 1, 2I]
        let gateScale = gateUp.isNVFP4 ? gateUp.expertScale : nil
        let downScale = down.isNVFP4 ? down.expertScale : nil
        let rows = idx.size
        var act: MLXArray
        if inklingUseKernels {
            let ones = MLXArray.ones([gateUp.expertScale.dim(0)], dtype: f32)
            act =
                Inkling2Kernels.moeActivation(
                    [gu.reshaped([rows, 2 * inter]), idx.flattened(), gateScale ?? ones, caps],
                    grid: (inter, rows, 1), threadGroup: (min(256, inter), 1, 1),
                    outputShapes: [[rows, inter]], outputDTypes: [f32])[0]
            act = act.reshaped(Array(gu.shape.dropLast()) + [inter])
        } else {
            var g2 = gu
            if let gateScale { g2 = g2 * gateScale[idx].reshaped(idx.shape + [1, 1]) }
            let pair = g2.reshaped(Array(g2.shape.dropLast()) + [inter, 2])
            let g = pair[.ellipsis, 0]
            let u = pair[.ellipsis, 1]
            // softcap(g, 4) * sigmoid(g) * softcap(u, 25) in fp32; the sigmoid sees the RAW gate.
            // The reference rounds the activation to the model dtype before the down projection.
            act = (softcap(g, gateCap) * MLX.sigmoid(g) * softcap(u, upCap))
                .asType(x.dtype).asType(f32)
        }
        var y = down(act, idx, sorted: doSort)  // [..., 1, H]
        if let inverse {
            y = y[inverse]
        }
        y = y.reshaped([N, k, y.dim(-1)])
        if inklingUseKernels {
            let ones = MLXArray.ones([down.expertScale.dim(0)], dtype: f32)
            return Inkling2Kernels.moeCombine(
                [y, inds, downScale ?? ones, w], grid: (y.dim(-1), N, 1),
                threadGroup: (256, 1, 1), outputShapes: [[N, y.dim(-1)]],
                outputDTypes: [x.dtype])[0]
        }
        if let downScale { y = y * downScale[inds].expandedDimensions(axis: -1) }
        // Reference: (expert_out * weight_fp32) index_add_ into an fp32 buffer.
        return MLX.sum(y * w.expandedDimensions(axis: -1), axis: -2).asType(x.dtype)
    }
}

final class Inkling2Router: Module {
    var weight: MLXArray
    @ParameterInfo(key: "e_score_correction_bias") var bias: MLXArray
    let topK: Int
    let routeScale: Float
    let meta: MLXArray
    let scaleArray: MLXArray

    init(_ args: Inkling2LoopConfiguration) {
        weight = MLXArray.zeros([args.nRoutedExperts, args.hiddenSize])
        _bias.wrappedValue = MLXArray.zeros([args.nRoutedExperts])
        topK = args.numExpertsPerTok
        routeScale = args.routeScale
        meta = MLXArray([Int32(args.numExpertsPerTok)])
        scaleArray = MLXArray([args.routeScale])
    }

    /// scores = sigmoid(x @ W^T); SELECTION by top-k of scores + bias; weights = the selected
    /// RAW scores / (sum + 1e-20) * route_scale. fp32.
    func callAsFunction(_ flat: MLXArray) -> (MLXArray, MLXArray) {
        let logits = MLX.matmul(flat.asType(f32), weight.asType(f32).T)
        if inklingUseKernels, weight.dim(0) <= 256, topK <= 16 {
            let rows = flat.dim(0)
            let tg = min(256, max(32, weight.dim(0)))
            let out = Inkling2Kernels.routerSelect(
                [logits, bias.asType(f32), meta, scaleArray],
                grid: (rows * tg, 1, 1), threadGroup: (tg, 1, 1),
                outputShapes: [[rows, topK], [rows, topK]], outputDTypes: [.uint32, f32])
            return (out[0], out[1])
        }
        let scores = MLX.sigmoid(logits)
        let sel = scores + bias.asType(f32)
        let inds = MLX.argPartition(-sel, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
        var w = MLX.takeAlong(scores, inds, axis: -1)
        w = w / (MLX.sum(w, axis: -1, keepDims: true) + 1e-20)
        return (inds, w * routeScale)
    }
}

final class Inkling2MoE: Module {
    @ModuleInfo(key: "gate") var gate: Inkling2Router
    /// The checkpoint's native NVFP4 / MXFP8 experts; nil for a JANGH bundle.
    @ModuleInfo(key: "switch_mlp") var switchMLP: Inkling2SwitchMLP?
    /// A JANGH bundle's mapped codebook banks for this layer (no parameters: the generic loader never sees the
    /// `tq2_*` tensors). Gate/up run with Inkling's softcapped activation fused into the epilogue.
    @ModuleInfo(key: "jangh_experts") var janghExperts: JANGHRoutedExpertLayer?
    /// `[mlp_gate_softcap, mlp_up_softcap]` fp32 (the batch-1 JANGH kernels' activation constants).
    let softcaps: MLXArray

    init(_ args: Inkling2LoopConfiguration, jangh: JANGHRoutedExpertLayer? = nil) {
        softcaps = MLXArray([args.mlpGateSoftcap, args.mlpUpSoftcap])
        _gate.wrappedValue = Inkling2Router(args)
        if let jangh {
            _janghExperts.wrappedValue = jangh
        } else {
            _switchMLP.wrappedValue = Inkling2SwitchMLP(args)
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (B, T, H) = (x.dim(0), x.dim(1), x.dim(2))
        let flat = x.reshaped([-1, H])
        let (inds, w) = gate(flat)
        if let janghExperts {
            // Reference: (expert_out * weight_fp32) summed in fp32, cast to the input dtype.
            return janghExperts.callRouted(flat, indices: inds.asType(.uint32), scores: w)
                .reshaped([B, T, H])
        }
        return switchMLP!(flat, inds, w).reshaped([B, T, H])
    }
}

// MARK: - Depth attention (block residuals, loop messages, message gates)

/// Block states travel as ONE fp32 array `[B, T, S, H]` (S grows at block boundaries).
func inklingAppendState(_ stack: MLXArray?, _ h: MLXArray) -> MLXArray {
    let e = h.asType(f32).expandedDimensions(axis: 2)
    return stack.map { MLX.concatenated([$0, e], axis: 2) } ?? e
}

/// `Inkling2BlockAttentionResidual`: fixed learned fp32 queries attend over the depth axis
/// (completed block states + the current partial block), keys = values = the unweighted-RMS
/// normalized states, scale H^-0.5. Per token; nothing goes in the KV cache.
final class Inkling2BlockAttentionResidual: Module {
    var query: MLXArray

    init(hidden: Int, queries: Int) {
        query = queries == 1 ? MLXArray.zeros([hidden]) : MLXArray.zeros([queries, hidden])
    }

    var numQueries: Int { query.ndim == 1 ? 1 : query.dim(0) }

    /// `stack` `[B, T, S, H]` fp32 (nil = no completed states), `partial` `[B, T, H]`.
    /// Returns `[B, T, nq, H]` fp32.
    func callAsFunction(_ stack: MLXArray?, _ partial: MLXArray, ones: MLXArray) -> MLXArray {
        let (B, T, H) = (partial.dim(0), partial.dim(1), partial.dim(2))
        if inklingUseKernels {
            let R = B * T
            return Inkling2Kernels.depth(
                stack: stack.map { $0.reshaped([R, $0.dim(2), H]) },
                partial: partial.reshaped([R, H]), query: query.asType(f32), rows: R
            ).reshaped([B, T, numQueries, H])
        }
        let sources = inklingAppendState(stack, partial)
        let stacked = MLXFast.rmsNorm(sources, weight: ones, eps: 1e-6)  // [B, T, S, H]
        let q = query.asType(f32).reshaped([-1, H])
        let scores = MLX.matmul(stacked, q.T) * (1 / Float(H).squareRoot())  // [B,T,S,nq]
        let p = MLX.softmax(scores, axis: -2, precise: true)
        return MLX.matmul(p.swappedAxes(-1, -2), stacked)
    }
}

extension Inkling2BlockAttentionResidual {
    /// The depth attention followed by each query's weighted RMSNorm (the layer's input /
    /// post-attention layernorm), rounded once to `dtype`: `[B, T, nq, H]`.
    func normed(
        _ stack: MLXArray?, _ partial: MLXArray, norms: [Inkling2RMSNorm], dtype: DType,
        ones: MLXArray, box: Inkling2DerivedBox
    ) -> MLXArray {
        let (B, T, H) = (partial.dim(0), partial.dim(1), partial.dim(2))
        if inklingUseKernels {
            if box.value == nil {
                box.value = MLX.stacked(norms.map { $0.weight.asType(f32) }, axis: 0)
                box.extra = MLXArray([norms[0].eps])
            }
            let R = B * T
            return Inkling2Kernels.depth(
                stack: stack.map { $0.reshaped([R, $0.dim(2), H]) },
                partial: partial.reshaped([R, H]), query: query.asType(f32), rows: R,
                normWeight: box.value, normEps: box.extra, outDType: dtype
            ).reshaped([B, T, numQueries, H])
        }
        let raw = self(stack, partial, ones: ones)
        return MLX.stacked(
            norms.enumerated().map { i, n in n(raw[.ellipsis, i, 0...]).asType(dtype) }, axis: -2)
    }
}

/// `Inkling2LoopMessageGate`: `(1 - tanh(forget)) * h + tanh(gain) * msg`, with
/// gate_in = [rms(h), rms(msg)] (fp32) -> down_proj (in its weight dtype) -> fp32 ->
/// forget / gain projections in fp32.
final class Inkling2MessageGate: Module {
    @ModuleInfo(key: "down_proj") var downProj: Linear
    @ModuleInfo(key: "forget_gate_proj") var forgetProj: Linear
    @ModuleInfo(key: "message_gate_proj") var gainProj: Linear
    let eps: Float
    let epsArray: MLXArray
    /// `[Wf; Wg]^T` fp32 `[rank, 2H]`, built on first use (one GEMV for both gates). Held in a
    /// plain box so module reflection does not treat it as a parameter.
    private let derived = Inkling2DerivedBox()

    init(hidden: Int, rank: Int, eps: Float) {
        _downProj.wrappedValue = Linear(2 * hidden, rank, bias: false)
        _forgetProj.wrappedValue = Linear(rank, hidden, bias: false)
        _gainProj.wrappedValue = Linear(rank, hidden, bias: false)
        self.eps = eps
        epsArray = MLXArray([eps])
    }

    /// Returns fp32 `[..., H]`.
    func callAsFunction(_ h: MLXArray, _ msg: MLXArray, ones: MLXArray) -> MLXArray {
        let wdt: DType = (downProj as? QuantizedLinear)?.scales.dtype ?? downProj.weight.dtype
        if inklingUseKernels {
            let H = h.dim(-1)
            let shape = h.shape
            let R = h.size / H
            let h2 = h.reshaped([R, H])
            let m2 = msg.reshaped([R, H])
            let gin = Inkling2Kernels.gateInput(
                [h2, m2, epsArray], grid: (R * 256, 1, 1), threadGroup: (256, 1, 1),
                outputShapes: [[R, 2 * H]], outputDTypes: [wdt])[0]
            if derived.value == nil {
                derived.value = MLX.contiguous(
                    MLX.concatenated(
                        [forgetProj.weight.asType(f32), gainProj.weight.asType(f32)], axis: 0
                    ).T)  // [rank, 2H]
            }
            let fg = MLX.matmul(downProj(gin).asType(f32), derived.value!)  // [R, 2H]
            return Inkling2Kernels.gateCombine(
                [fg, h2, m2], grid: (H, R, 1), threadGroup: (256, 1, 1), outputShapes: [[R, H]],
                outputDTypes: [f32])[0].reshaped(shape)
        }
        let hf = h.asType(f32)
        let mf = msg.asType(f32)
        let g0 = MLX.concatenated(
            [
                MLXFast.rmsNorm(hf, weight: ones, eps: eps),
                MLXFast.rmsNorm(mf, weight: ones, eps: eps),
            ], axis: -1)
        let g = downProj(g0.asType(wdt)).asType(f32)
        let forget = MLX.tanh(MLX.matmul(g, forgetProj.weight.asType(f32).T))
        let gain = MLX.tanh(MLX.matmul(g, gainProj.weight.asType(f32).T))
        return (1 - forget) * hf + gain * mf
    }
}

/// Load-derived arrays a module caches outside its parameter tree.
final class Inkling2DerivedBox: @unchecked Sendable {
    var value: MLXArray?
    var extra: MLXArray?
    var extra2: MLXArray?
}

// MARK: - Decoder layer

final class Inkling2DecoderLayer: Module {
    let index: Int
    let isSliding: Bool
    let isBlockBoundary: Bool
    let isLoopLayer: Bool
    let convKV: Int
    let hidden: Int

    @ModuleInfo(key: "attention_input") var attentionInput: Inkling2BlockAttentionResidual?
    @ModuleInfo(key: "mlp_input") var mlpInput: Inkling2BlockAttentionResidual
    @ModuleInfo(key: "self_attn") var selfAttn: Inkling2Attention
    @ModuleInfo(key: "mlp") var mlp: Inkling2MoE
    @ModuleInfo(key: "input_layernorm") var inputLN: Inkling2RMSNorm?
    @ModuleInfo(key: "q_input_layernorm") var qInputLN: Inkling2RMSNorm?
    @ModuleInfo(key: "k_input_layernorm") var kInputLN: Inkling2RMSNorm?
    @ModuleInfo(key: "v_input_layernorm") var vInputLN: Inkling2RMSNorm?
    @ModuleInfo(key: "post_attention_layernorm") var postAttnLN: Inkling2RMSNorm
    @ModuleInfo(key: "post_attention_convolution") var postAttnConv: Inkling2ShortConv
    @ModuleInfo(key: "post_mlp_convolution") var postMLPConv: Inkling2ShortConv
    @ModuleInfo(key: "q_message_gate") var qGate: Inkling2MessageGate?
    @ModuleInfo(key: "k_message_gate") var kGate: Inkling2MessageGate?
    @ModuleInfo(key: "v_message_gate") var vGate: Inkling2MessageGate?
    @ModuleInfo(key: "mlp_message_gate") var mlpGate: Inkling2MessageGate?
    /// Re-laid-out weights for `decodeStep` (Inkling2Decode.swift), built on first use.
    let decodeWeights = Inkling2DecodeWeights()
    /// Stacked fp32 norm weights for the fused depth-attention + layernorm kernels.
    private let attnNormBox = Inkling2DerivedBox()
    private let mlpNormBox = Inkling2DerivedBox()

    init(_ args: Inkling2LoopConfiguration, index: Int, jangh: JANGHRoutedExpertLayer? = nil) {
        self.index = index
        isSliding = args.isSliding(index)
        isBlockBoundary = args.isBlockBoundary(index)
        isLoopLayer = args.isLoopLayer(index)
        let (_, kv, d) = args.heads(index)
        convKV = kv * d
        let H = args.hiddenSize
        hidden = H
        if index > 0 {
            _attentionInput.wrappedValue = Inkling2BlockAttentionResidual(
                hidden: H, queries: isSliding ? 1 : 3)
        }
        _mlpInput.wrappedValue = Inkling2BlockAttentionResidual(hidden: H, queries: 1)
        _selfAttn.wrappedValue = Inkling2Attention(args, layer: index)
        _mlp.wrappedValue = Inkling2MoE(args, jangh: jangh)
        let eps = args.rmsNormEps
        if isSliding {
            _inputLN.wrappedValue = Inkling2RMSNorm(H, eps: eps)
        } else {
            _qInputLN.wrappedValue = Inkling2RMSNorm(H, eps: eps)
            _kInputLN.wrappedValue = Inkling2RMSNorm(H, eps: eps)
            _vInputLN.wrappedValue = Inkling2RMSNorm(H, eps: eps)
        }
        _postAttnLN.wrappedValue = Inkling2RMSNorm(H, eps: eps)
        _postAttnConv.wrappedValue = Inkling2ShortConv(channels: H, kernel: args.sconvKernelSize)
        _postMLPConv.wrappedValue = Inkling2ShortConv(channels: H, kernel: args.sconvKernelSize)
        if isLoopLayer {
            let r = args.loopGateRank
            _qGate.wrappedValue = Inkling2MessageGate(hidden: H, rank: r, eps: eps)
            _kGate.wrappedValue = Inkling2MessageGate(hidden: H, rank: r, eps: eps)
            _vGate.wrappedValue = Inkling2MessageGate(hidden: H, rank: r, eps: eps)
            _mlpGate.wrappedValue = Inkling2MessageGate(hidden: H, rank: r, eps: eps)
        }
    }

    /// Reference residual structure:
    ///   attn_in = attention_input(block_states, h)          (layer 0: h itself)
    ///   a = post_attention_convolution(o_proj(attn(norm(attn_in))) * ...)
    ///   block boundary: block_states += [h]; residual = a   else residual = h + a
    ///   m = post_mlp_convolution(mlp(post_attention_layernorm(mlp_input(block_states, residual))))
    ///   return residual + m
    /// `stack` = the completed block states `[B, T, S, H]` fp32. Returns the new hidden and the
    /// stack the caller continues with (this layer's input appended at a block boundary).
    /// `messages` `[B, T, 4, H]` (q, k, v, mlp) only in loop passes after the first.
    func callAsFunction(
        _ h: MLXArray, stack: MLXArray?, kv: (any KVCache)?, conv: MambaCache?,
        messages: MLXArray?, ones: MLXArray, recorder: Inkling2ConvRecorder? = nil
    ) -> (MLXArray, MLXArray?) {
        let dtype = h.dtype
        let T = h.dim(1)
        let c = convKV
        // Packed conv state [B, K-1, kvDim | kvDim | H | H] (fp32).
        let packed = conv?[0]

        let pre: [MLXArray]
        if isSliding, messages == nil, let attentionInput {
            // depth attention + input_layernorm fused (one kernel)
            let x = attentionInput.normed(
                stack, h, norms: [inputLN!], dtype: dtype, ones: ones, box: attnNormBox
            )[.ellipsis, 0, 0...]
            pre = selfAttn.project(x, keyInput: nil, valueInput: nil, packed: packed, ones: ones)
        } else if !isSliding, messages == nil, let attentionInput {
            let three = attentionInput.normed(
                stack, h, norms: [qInputLN!, kInputLN!, vInputLN!], dtype: dtype, ones: ones,
                box: attnNormBox)
            pre = selfAttn.project(
                three[.ellipsis, 0, 0...], keyInput: three[.ellipsis, 1, 0...],
                valueInput: three[.ellipsis, 2, 0...], packed: packed, ones: ones)
        } else if isSliding {
            let x = attentionInput.map { $0(stack, h, ones: ones)[.ellipsis, 0, 0...] } ?? h
            let ln = inputLN!
            if let messages {
                pre = selfAttn.project(
                    ln(qGate!(x, messages[.ellipsis, 0, 0...], ones: ones)).asType(dtype),
                    keyInput: ln(kGate!(x, messages[.ellipsis, 1, 0...], ones: ones)).asType(dtype),
                    valueInput: ln(vGate!(x, messages[.ellipsis, 2, 0...], ones: ones)).asType(
                        dtype),
                    packed: packed, ones: ones)
            } else {
                pre = selfAttn.project(
                    ln(x).asType(dtype), keyInput: nil, valueInput: nil, packed: packed, ones: ones)
            }
        } else {
            var qh = h
            var kh = h
            var vh = h
            if let attentionInput {
                let three = attentionInput(stack, h, ones: ones)
                qh = three[.ellipsis, 0, 0...]
                kh = three[.ellipsis, 1, 0...]
                vh = three[.ellipsis, 2, 0...]
            }
            if let messages {
                qh = qGate!(qh, messages[.ellipsis, 0, 0...], ones: ones)
                kh = kGate!(kh, messages[.ellipsis, 1, 0...], ones: ones)
                vh = vGate!(vh, messages[.ellipsis, 2, 0...], ones: ones)
            }
            pre = selfAttn.project(
                qInputLN!(qh).asType(dtype), keyInput: kInputLN!(kh).asType(dtype),
                valueInput: vInputLN!(vh).asType(dtype), packed: packed, ones: ones)
        }
        let merged = selfAttn.attend(
            q: pre[0], k: pre[1], v: pre[2], r: pre[3], sinks: pre[5], kv: kv)

        // residual = a (block boundary) or h + a, fp32, fused with the post-attention conv.
        let attnConvIn = selfAttn.finish(merged, gate: pre[4])
        let (residual, newAttn) = postAttnConv.residual(
            attnConvIn, packed: packed, offset: 2 * c, res: h,
            mode: isBlockBoundary ? 0 : 1, outDType: f32)
        let states = isBlockBoundary ? inklingAppendState(stack, h) : stack
        var m: MLXArray
        if let messages {
            m = mlpInput(states, residual, ones: ones)[.ellipsis, 0, 0...]
            m = mlpGate!(m, messages[.ellipsis, 3, 0...], ones: ones)
            m = postAttnLN(m).asType(dtype)
        } else {
            m =
                mlpInput.normed(
                    states, residual, norms: [postAttnLN], dtype: dtype, ones: ones, box: mlpNormBox
                )[.ellipsis, 0, 0...]
        }
        m = mlp(m)
        let (out, newMLP) = postMLPConv.residual(
            m, packed: packed, offset: 2 * c + hidden, res: residual, mode: 2, outDType: dtype)
        if let recorder, let conv {
            // Speculative verify: keep every row's PRE-conv inputs so the state can be rewound
            // to any accepted row (conv state = the last K-1 inputs).
            recorder.record(
                conv: conv,
                oldState: packed ?? MLXArray.zeros([h.dim(0), 3, 2 * c + 2 * hidden], dtype: f32),
                inputs: MLX.concatenated(
                    [
                        pre[8].asType(f32), pre[9].asType(f32), attnConvIn.asType(f32),
                        m.asType(f32),
                    ],
                    axis: -1))
        }
        if let conv {
            conv[0] = MLX.concatenated([pre[6], pre[7], newAttn, newMLP], axis: -1)
            conv.offset += T
        }
        return (out, states)
    }
}

// MARK: - Text model

/// `Inkling2LoopMessageAttentionResidual`: 4 learned queries (q, k, v, mlp) per loop layer attend
/// over the PREVIOUS pass's loop-block states + its final hidden. `query` `[loopLayers, 4, H]`.
final class Inkling2MessageInput: Module {
    var query: MLXArray

    init(loopLayers: Int, hidden: Int) {
        query = MLXArray.zeros([loopLayers, 4, hidden])
    }

    /// All loop layers' messages for one pass at once: the normalized stack is the same for every
    /// layer of the pass, only the queries differ. `prev` `[B, T, S, H]` fp32, `partial` `[B, T, H]`.
    /// Returns `[B, T, loopLayers * 4, H]` fp32.
    func callAsFunction(_ prev: MLXArray?, _ partial: MLXArray, ones: MLXArray) -> MLXArray {
        let stacked = MLXFast.rmsNorm(
            inklingAppendState(prev, partial), weight: ones, eps: 1e-6)  // [B, T, S, H]
        let H = stacked.dim(-1)
        let q = query.asType(f32).reshaped([-1, H])  // [L*4, H]
        let p = MLX.softmax(
            MLX.matmul(stacked, q.T) * (1 / Float(H).squareRoot()), axis: -2, precise: true)
        return MLX.matmul(p.swappedAxes(-1, -2), stacked)
    }
}

public final class Inkling2LoopModelInner: Module {
    let args: Inkling2LoopConfiguration
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "embed_norm") var embedNorm: Inkling2RMSNorm
    let layers: [Inkling2DecoderLayer]
    @ModuleInfo(key: "norm") var norm: Inkling2RMSNorm
    @ModuleInfo(key: "final_input") var finalInput: Inkling2BlockAttentionResidual
    @ModuleInfo(key: "message_input") var messageInput: Inkling2MessageInput?

    /// fp32 ones: the weight of every unweighted RMSNorm (largest width used is `hidden`).
    let ones: MLXArray

    init(_ args: Inkling2LoopConfiguration, routedExperts: [Int: JANGHRoutedExpertLayer] = [:]) {
        self.args = args
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabSize, dimensions: args.hiddenSize)
        _embedNorm.wrappedValue = Inkling2RMSNorm(args.hiddenSize, eps: args.rmsNormEps)
        layers = (0 ..< args.numHiddenLayers).map {
            Inkling2DecoderLayer(args, index: $0, jangh: routedExperts[$0])
        }
        _norm.wrappedValue = Inkling2RMSNorm(args.hiddenSize, eps: args.rmsNormEps)
        _finalInput.wrappedValue = Inkling2BlockAttentionResidual(
            hidden: args.hiddenSize, queries: 1)
        if args.loopPasses > 1 {
            _messageInput.wrappedValue = Inkling2MessageInput(
                loopLayers: args.numLoopLayers, hidden: args.hiddenSize)
        }
        ones = MLXArray.ones([args.hiddenSize], dtype: .float32)
    }

    /// Every layer can take the single-token decode path (plain-weight projections). Checked once.
    lazy var decodeSupported: Bool = layers.allSatisfy { $0.supportsDecodeStep() }

    /// Token embeddings INCLUDING embed_norm (reference `Inkling2NormedEmbedding`).
    func embed(_ ids: MLXArray) -> MLXArray { embedNorm(embedTokens(ids)) }

    func callAsFunction(_ ids: MLXArray, cache: [KVCache]?) -> MLXArray {
        forward(ids, cache: cache, capture: nil, recorder: nil)
    }

    /// `capture`: unrolled layer-visit ids whose layer INPUT (the attn-res partial accumulator,
    /// BF16) is collected; `numLayerVisits` (48) means the final normed hidden. In capture order.
    /// `recorder`: speculative verify bookkeeping (conv inputs per slot).
    func forward(
        _ ids: MLXArray, cache: [KVCache]?, capture: Inkling2Capture?,
        recorder: Inkling2ConvRecorder?, embeddings: MLXArray? = nil
    ) -> MLXArray {
        // `embeddings`: the prompt's input rows when they are not plain token embeddings (image
        // features spliced in); `ids` still gives the shape.
        var h = embeddings ?? embed(ids)
        let dtype = h.dtype
        let slots = args.numCacheSlots
        if let cache {
            precondition(
                cache.count == 2 * slots,
                "inkling2_loop: expected \(2 * slots) cache entries, got \(cache.count)")
        }
        func kv(_ layer: Int, _ pass: Int) -> (any KVCache)? {
            cache.map { $0[args.cacheSlot(layer: layer, pass: pass)] }
        }
        func conv(_ layer: Int, _ pass: Int) -> MambaCache? {
            cache.map { $0[slots + args.cacheSlot(layer: layer, pass: pass)] as! MambaCache }
        }

        var stack: MLXArray? = nil
        let fastDecode =
            inklingUseKernels && cache != nil && ids.dim(0) == 1 && ids.dim(1) == 1
            && !Inkling2Runtime.forceGeneralDecode && decodeSupported && recorder == nil
        // Short multi-row calls (speculative verify, small chunks) take the fused multi-row path.
        let multiRow =
            inklingUseKernels && cache != nil && ids.dim(0) == 1 && ids.dim(1) > 1
            && ids.dim(1) <= 16 && !Inkling2Runtime.forceGeneralDecode
            && decodeSupported
        let scalars = Inkling2StepScalars()
        func visit(_ layer: Int, _ pass: Int) -> Int {
            if layer < args.loopStart { return layer }
            if layer >= args.loopEnd {
                return layer + (args.loopPasses - 1) * args.numLoopLayers
            }
            return layer + pass * args.numLoopLayers
        }
        func run(_ ids: Range<Int>, pass: Int, messages: MLXArray?) {
            for (j, i) in ids.enumerated() {
                if let capture, capture.ids.contains(visit(i, pass)) {
                    capture.values[visit(i, pass)] = h
                }
                if multiRow {
                    (h, stack) = layers[i].multiRowStep(
                        h, stack: stack, kv: kv(i, pass)!, conv: conv(i, pass)!, messages: messages,
                        loopIndex: j, ones: ones, recorder: recorder)
                    continue
                }
                if let recorder {
                    let msgs = messages.map { $0[0..., 0..., (4 * j) ..< (4 * j + 4), 0...] }
                    (h, stack) = layers[i](
                        h, stack: stack, kv: kv(i, pass), conv: conv(i, pass), messages: msgs,
                        ones: ones, recorder: recorder)
                    continue
                }
                if fastDecode {
                    (h, stack) = layers[i].decodeStep(
                        h, stack: stack, kv: kv(i, pass)!, conv: conv(i, pass)!, messages: messages,
                        loopIndex: j, ones: ones, scalars: scalars)
                    continue
                }
                let msgs = messages.map { $0[0..., 0..., (4 * j) ..< (4 * j + 4), 0...] }
                (h, stack) = layers[i](
                    h, stack: stack, kv: kv(i, pass), conv: conv(i, pass), messages: msgs,
                    ones: ones)
            }
        }

        let s0 = args.loopStart
        let s1 = args.loopEnd
        if args.loopPasses <= 1 {
            run(0 ..< args.numHiddenLayers, pass: 0, messages: nil)
        } else {
            run(0 ..< s0, pass: 0, messages: nil)
            let preH = h
            let preStack = stack
            let nPre = preStack?.dim(2) ?? 0
            run(s0 ..< s1, pass: 0, messages: nil)
            for p in 1 ..< args.loopPasses {
                // Reference: previous_block_states = block_states[num_pre_loop_blocks + 1:]
                // (+1 skips the pre-loop output the pass appended at its first boundary),
                // previous_hidden = the pass's last hidden; restart from the pre-loop states.
                let all = stack!
                let prev: MLXArray? =
                    all.dim(2) > nPre + 1 ? all[0..., 0..., (nPre + 1)..., 0...] : nil
                let messages = messageInput!(prev, h, ones: ones)
                h = preH
                stack = preStack
                run(s0 ..< s1, pass: p, messages: messages)
            }
            run(s1 ..< args.numHiddenLayers, pass: 0, messages: nil)
        }
        h = finalInput(stack, h, ones: ones)[.ellipsis, 0, 0...]
        let out = norm(h).asType(dtype)
        if let capture {
            capture.values[args.numHiddenLayers + (args.loopPasses - 1) * args.numLoopLayers] = out
        }
        return out
    }
}

/// Layer-visit capture for the DSpark drafter (see `Inkling2LoopModelInner.forward`).
public final class Inkling2Capture: @unchecked Sendable {
    let ids: [Int]
    var values: [Int: MLXArray] = [:]
    init(ids: [Int]) { self.ids = ids }

    /// `[B * T, ids.count, H]` in capture order.
    func stacked() -> MLXArray {
        let parts = ids.map { values[$0]! }
        let s = MLX.stacked(parts, axis: 2)  // [B, T, n, H]
        return s.reshaped([-1, ids.count, s.dim(-1)])
    }
}

/// Speculative-verify bookkeeping: per conv cache, its state before the verify and every row's
/// PRE-conv inputs, so `rewind(keep:)` can rebuild the state after any accepted row.
public final class Inkling2ConvRecorder: @unchecked Sendable {
    private var entries: [(MambaCache, MLXArray, MLXArray)] = []

    func record(conv: MambaCache, oldState: MLXArray, inputs: MLXArray) {
        entries.append((conv, oldState, inputs))
    }

    /// Keep the first `keep` verify rows: state = last 3 of [old state, inputs[0 ..< keep]].
    func rewind(keep: Int) {
        for (conv, old, inputs) in entries {
            let T = inputs.dim(1)
            guard keep < T else { continue }
            let xp = MLX.concatenated([old, inputs], axis: 1)  // [B, 3 + T, P]
            conv[0] = xp[0..., keep ..< (keep + 3), 0...]
            conv.offset -= (T - keep)
        }
        entries.removeAll()
    }
}

// MARK: - Model

public final class Inkling2LoopModel: Module, LLMModel, ModalityBearing {
    public let vocabularySize: Int
    let args: Inkling2LoopConfiguration
    public let model: Inkling2LoopModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear
    @ModuleInfo(key: "mtp_drafter") var drafter: Inkling2DSparkDrafter?
    /// Image input (`Inkling2Vision.swift`); nil for a text-only bundle or construction.
    @ModuleInfo(key: "vision_tower") public var vision: Inkling2VisionModel?

    /// Lanes this instance carries: the vision lane exists only when the tower was built.
    public var modalities: Set<ModelRuntimeRequestModality> { vision == nil ? [.text] : [.text, .vision] }
    /// Set by `sanitize` when the bundle actually ships the drafter's tensors.
    public private(set) var hasDrafterWeights = false
    /// A dSpark verify in flight (set by the draft session around one target forward).
    var pendingVerify: Inkling2PendingVerify?

    /// Tensors the generic loader must skip: a JANGH bundle's routed codebook banks (mapped by the preparation).
    let customTensorNames: Set<String>

    public init(_ args: Inkling2LoopConfiguration) {
        self.args = args
        vocabularySize = args.vocabSize
        customTensorNames = []
        model = Inkling2LoopModelInner(args)
        if let d = args.dspark { _drafter.wrappedValue = Inkling2DSparkDrafter(d) }
        if let v = args.vision { _vision.wrappedValue = Inkling2VisionModel(v) }
        // `embed` and `unembed` are separate tensors in the checkpoint, never tied.
        _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabSize, bias: false)
    }

    /// A JANGH bundle: every layer's routed experts are the admitted mapped codebook banks (`routedExperts`, keyed by
    /// layer, all layers) and `excludedTensorNames` are exactly those banks' tensors.
    init(
        _ args: Inkling2LoopConfiguration, routedExperts: [Int: JANGHRoutedExpertLayer],
        excludedTensorNames: Set<String>
    ) throws {
        let layers = Set(0 ..< args.numHiddenLayers)
        guard Set(routedExperts.keys) == layers,
            JANGHModelPreparation.isCanonicalRoutedExclusionSet(excludedTensorNames, routedLayers: layers)
        else {
            throw Inkling2LoopJANGHPreparation.PreparationError.invalid(
                "Inkling-2 JANGH: routed banks \(routedExperts.keys.sorted()) do not cover layers 0..<\(args.numHiddenLayers)")
        }
        self.args = args
        vocabularySize = args.vocabSize
        customTensorNames = excludedTensorNames
        model = Inkling2LoopModelInner(args, routedExperts: routedExperts)
        if let d = args.dspark { _drafter.wrappedValue = Inkling2DSparkDrafter(d) }
        if let v = args.vision { _vision.wrappedValue = Inkling2VisionModel(v) }
        _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabSize, bias: false)
    }

    /// The per-slot cache state is path dependent and the forward is not batch aware.
    public var maximumSupportedDecodeBatchSize: Int? { 1 }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        let slots = args.numCacheSlots
        let windowed = ProcessInfo.processInfo.environment["VMLX_INKLING_WINDOWED_KV"] != "0"
        var cache =
            (0 ..< slots).map { slot -> KVCache in
                let layer = args.sourceLayer(ofSlot: slot)
                guard windowed, args.isSliding(layer) else { return KVCacheSimple() }
                let h = args.heads(layer)
                return TemporalWindowKVCache(window: args.slidingWindowSize, heads: h.kv, headDim: h.dim)
            }
            + (0 ..< slots).map { _ in MambaCache(slots: 1) as KVCache }
        if parameters?.draftStrategy?.usesDSpark == true, dsparkAvailable, let d = drafter {
            // dSpark: the drafter's context KV per draft layer + the last row's target features.
            cache += (0 ..< d.cfg.layers).map { _ in KVCacheSimple() as KVCache }
            cache.append(MambaCache(slots: 1))
        }
        return cache
    }

    func logits(_ h: MLXArray) -> MLXArray {
        var out = lmHead(h).asType(.float32)
        if let cap = args.finalLogitSoftcapping, cap > 0 {
            out = softcap(out, cap)
        }
        if let u = args.unpaddedVocabSize, u < out.dim(-1) {
            out = out[.ellipsis, ..<u]
        }
        return out
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        logits(trunk(inputs, cache: cache))
    }

    /// Chunked text prefill. Non-final chunks run the trunk only (no 200k-vocab logits); the
    /// cache is evaluated between chunks so each chunk's activations are released. The last
    /// chunk's logits are computed for its final row only.
    public func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws
        -> PrepareResult
    {
        let ids =
            input.text.tokens.ndim == 1
            ? input.text.tokens.expandedDimensions(axis: 0) : input.text.tokens
        // Image rows: the prompt's embeddings with the image features spliced in (nil for text, or
        // when every image is inside the restored cache prefix).
        let embeddings = try imageEmbeddings(ids: ids, image: input.image)
        let n = ids.dim(1)
        let step = max(windowSize ?? 512, 1)
        var offset = 0
        while n - offset > step {
            try Task.checkCancellation()
            _ = trunk(
                ids[0..., offset ..< (offset + step)], cache: cache,
                embeddings: embeddings?[0..., offset ..< (offset + step), 0...])
            MLX.eval(cache)
            offset += step
            PrefillProgressReporter.reportCompletedUnits(offset)
            MLX.Memory.clearCache()
        }
        let h = trunk(ids[0..., offset...], cache: cache, embeddings: embeddings?[0..., offset..., 0...])
        return .logits(LMOutput(logits: logits(h[0..., (h.dim(1) - 1)..., 0...])))
    }

    /// `[1, T, H]` input rows for a prompt carrying image placeholders: embed + embed_norm for every
    /// token, the image tiles' projected features at the placeholders (reference `masked_scatter`).
    /// The placeholders in `ids` are the trailing ones of the request's images; a leading image may
    /// be partly inside the restored prefix, in which case only its remaining tiles are used.
    func imageEmbeddings(ids: MLXArray, image: LMInput.ProcessedImage?) throws -> MLXArray? {
        guard let image, let vision, let frames = image.frames, !frames.isEmpty else { return nil }
        let tokens = ids.reshaped([-1]).asType(.int32).asArray(Int32.self).map(Int.init)
        let spans = try Inkling2VisionLayout.spans(
            tokens: tokens, imageTokenId: vision.config.imageTokenId,
            grids: frames.map { (rows: $0.h, cols: $0.w) })
        guard !spans.isEmpty else { return nil }
        var h = model.embed(ids)
        for span in spans {
            let count = span.rows * span.cols
            let tiles = image.pixels[span.tileStart ..< (span.tileStart + count)]
            var feats = vision.features(tiles: tiles, rows: span.rows, cols: span.cols)
            if span.skippedTiles > 0 { feats = feats[span.skippedTiles...] }
            h[0..., span.start ..< (span.start + span.length), 0...] = feats.asType(h.dtype)
                .expandedDimensions(axis: 0)
            MLX.eval(h)
        }
        return h
    }

    /// The checkpoint's fp32 tensors (attention-residual / message queries, router bias, the
    /// per-expert NVFP4 factor) are part of the numerical contract; every activation cast is
    /// explicit in this file, so the loader's blanket bf16 materialisation must not run.
    public var preservesCheckpointParameterDTypes: Bool { true }

    /// Bumped only when the forward's arithmetic for persisted KV / conv state changes.
    public var cacheComputationIdentity: String? { "inkling2_loop.v1" }

    // MARK: weights

    public func sanitize(weights rawWeights: [String: MLXArray]) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        // Community conversions: the mlx-vlm tree (`language_model.model.*`), quantized dense
        // layers, and routed experts in HF (`experts.gate_up_proj`, gate half then up half) or
        // mlx-vlm (`switch_mlp.{gate,up,down}_proj`) form. All of them are brought to this
        // model's tree here, so every later path (fused decode, verify, dSpark) sees one layout.
        let expected = expectedParameterShapes()
        let tree = Self.isFusedProjectionLayout(rawWeights)
            ? Self.convertingFusedProjectionLayout(
                rawWeights, expected: expected, loopCore: args.loopStart ..< args.loopEnd)
            : rawWeights
        let weights = Self.dequantizingDenseLayers(
            Self.normalizingLayoutKeys(tree), expected: expected)
        var communityExperts: [String: [String: MLXArray]] = [:]
        let qKey = "model.llm.middle_cycle_cross_pass_queries"
        let kvKey = "model.llm.middle_cycle_cross_pass_kv_queries"
        if let q = weights[qKey], let kvq = weights[kvKey] {
            // [2n, H] (q, mlp per layer) + [2n, H] (k, v per layer) -> [n, 4, H] (q, k, v, mlp).
            let H = q.dim(-1)
            let qq = q.reshaped([-1, 2, H])
            let kk = kvq.reshaped([-1, 2, H])
            out["model.message_input.query"] = MLX.concatenated(
                [qq[0..., ..<1, 0...], kk, qq[0..., 1..., 0...]], axis: 1)
        }
        let auxSuffixes = ["_weight_scale", "_weight_amax", "_weight_outer_scale_denominator"]
        for (key, value) in weights {
            if key == qKey || key == kvKey { continue }
            if auxSuffixes.contains(where: { key.hasSuffix($0) }) { continue }
            // Vision tower: every layout -> `vision_tower.*` of this tree (dropped for a text-only
            // construction). Audio: not part of this module tree.
            if let name = Self.visionKey(key) {
                if vision != nil { out[name] = value }
                continue
            }
            if key.hasPrefix("model.audio") { continue }
            if key.hasPrefix("mtp_drafter.")
                || !(key.hasPrefix("model.") || key.hasPrefix("lm_head"))
            {
                // Native drafter (`mtp.safetensors`: `layers.N.*`, `markov_head.*`, `mask_embed`,
                // `fuse_weight`, `norm.weight`) -> the `mtp_drafter` submodule, as stored (BF16).
                if drafter != nil {
                    let k = key.hasPrefix("mtp_drafter.") ? key : "mtp_drafter." + key
                    out[k] = value
                    hasDrafterWeights = true
                }
                continue
            }
            if key.hasSuffix("mlp.experts.w13_weight") || key.hasSuffix("mlp.experts.w2_weight") {
                let isW13 = key.hasSuffix("w13_weight")
                let base = Inkling2LoopModel.mapKey(
                    String(key.dropLast((isW13 ? "experts.w13_weight" : "experts.w2_weight").count))
                )
                let leg = isW13 ? "gate_up_proj" : "down_proj"
                let prefix = base + "switch_mlp.\(leg)."
                guard let scales = weights[key + "_scale"] else { continue }
                out[prefix + "codes"] = MLX.contiguous(value).view(dtype: .uint32)
                out[prefix + "block_scales"] = scales
                if let amax = weights[key + "_amax"] {
                    let den =
                        weights[key + "_outer_scale_denominator"]?.reshaped([-1])[0]
                        ?? MLXArray(Float(2688))
                    out[prefix + "expert_scale"] = amax.asType(.float32) / den.asType(.float32)
                } else {
                    out[prefix + "expert_scale"] = MLXArray.ones([value.dim(0)], dtype: .float32)
                }
                continue
            }
            var n = Inkling2LoopModel.mapKey(key)
            if let (base, leg, leaf) = Self.communityExpertKey(n) {
                communityExperts[base + "|" + leg, default: [:]][leaf] = value
                continue
            }
            var v = value
            if n.hasSuffix("sconv.weight_proj") { n += ".weight" }
            if n.hasSuffix("_sconv.weight") || n.hasSuffix("_convolution.weight") {
                // Static sconv: checkpoint (C, 1, K) / MLX conv (C, K, 1) -> [K, C].
                if v.ndim == 3 {
                    v = v.reshaped([v.dim(0), -1]).transposed(1, 0)
                }
            }
            // Small parameters the forward reads in fp32: store them fp32 once instead of
            // casting on every call (values are unchanged: bf16 -> fp32 is exact).
            if n.hasSuffix("mlp.gate.weight") || n.hasSuffix("forget_gate_proj.weight")
                || n.hasSuffix("message_gate_proj.weight") || n.hasSuffix("_sconv.weight")
                || n.hasSuffix("_convolution.weight") || n.hasSuffix("sconv.static_weight")
                || n.hasSuffix("layernorm.weight") || n.hasSuffix("embed_norm.weight")
                || n == "model.norm.weight" || n.hasSuffix("_l2_clip.scale")
            {
                v = v.asType(.float32)
            }
            if ProcessInfo.processInfo.environment["VMLX_INKLING2_F32"] == "1",
                v.dtype == .bfloat16 || v.dtype == .float16
            {
                // Parity diagnostics: run every floating tensor (hence every activation) in fp32.
                v = v.asType(.float32)
            }
            out[n] = v
        }
        for (k, v) in Self.canonicalExperts(communityExperts) { out[k] = v }
        return out
    }

    /// Parameter shapes of this (placeholder) module tree, keyed like `sanitize`'s output.
    private func expectedParameterShapes() -> [String: [Int]] {
        var shapes: [String: [Int]] = [:]
        for (k, v) in parameters().flattened() { shapes[k] = v.shape }
        return shapes
    }

    /// A vision-tower tensor of any bundle layout -> its `vision_tower.*` name in this tree, nil
    /// for any other tensor. Official `model.visual.backbone.*` + `model.visual.proj_{in,out}`;
    /// mlx-lm `model.vision_tower.*` (`merger.linear_{1,2}`); mlx-vlm `vision_tower.*`
    /// (`proj_{in,out}`); `vision_tower.backbone.*`.
    static func visionKey(_ key: String) -> String? {
        var k: Substring
        if key.hasPrefix("model.visual.") {
            k = key.dropFirst("model.visual.".count)
        } else if key.hasPrefix("model.vision_tower.") {
            k = key.dropFirst("model.vision_tower.".count)
        } else if key.hasPrefix("vision_tower.") {
            k = key.dropFirst("vision_tower.".count)
        } else if key.hasPrefix("visual.") {
            k = key.dropFirst("visual.".count)
        } else {
            return nil
        }
        if k.hasPrefix("backbone.") { k = k.dropFirst("backbone.".count) }
        var n = "." + String(k) + "."
        for (a, b) in [
            (".proj_in.", ".merger.linear_1."), (".proj_out.", ".merger.linear_2."),
            (".ln_pre.", ".pre_layernorm."), (".ln_1.", ".layernorm_before."), (".ln_2.", ".layernorm_after."),
            (".ls_1.gamma.", ".lambda_1."), (".ls_2.gamma.", ".lambda_2."),
            (".mlp.c_fc.", ".mlp.fc1."), (".mlp.c_proj.", ".mlp.fc2."),
            (".embeddings.position_embedding.weight.", ".embeddings.position_embedding."),
        ] {
            n = n.replacingOccurrences(of: a, with: b)
        }
        return "vision_tower" + n.dropLast()
    }

    /// The fused-projection layout: `model.layers.N.mlp.router_weight` with fused
    /// `attn.in_proj` / `loop_down` / `loop_gain_forget` tensors.
    static func isFusedProjectionLayout(_ weights: [String: MLXArray]) -> Bool {
        weights.keys.contains { $0.hasPrefix("model.layers.") && $0.hasSuffix(".mlp.router_weight") }
    }

    /// The fused-projection layout back to this model's tree. Its packing is a set of exact
    /// reshapes and row concatenations of the source tensors:
    /// * sliding layers: `attn.in_proj` = rows [q, k, v, r, gate, sink]; global layers:
    ///   `attn.q_proj` = [q, r, gate, sink], `attn.{k,v}_proj` = [k|v, {k,v}_sconv.weight_proj],
    ///   `attn.{k,v}_static` = the static taps. Quantized ones are dequantized first (quantization
    ///   is per row, so a row split of the dequantized matrix is exact); every row count comes from
    ///   this model's own parameter shapes.
    /// * `attn.{k,v}_conv`, `attn_conv`, `mlp_conv`: `[C, K]` taps (this tree holds `[K, C]`).
    /// * `loop_down` `[4, 2, H, r]` and `loop_gain_forget` `[4, r, 2H]` over (attn, k, v, mlp):
    ///   `down_weight` `[r, 2H]`, `gain_weight` / `forget_weight` `[H, r]` per message gate.
    /// * `loop_message.query` `[4, H]` per core layer, already in this model's (q, k, v, mlp) order.
    /// * experts `mlp.experts.{w13,w2}` (`w13` rows in source order, NVFP4 `global_scale` in MLX's
    ///   amax convention) -> the mlx-lm names, handled with the other community experts.
    static func convertingFusedProjectionLayout(
        _ weights: [String: MLXArray], expected: [String: [Int]], loopCore: Range<Int>
    ) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        var messageQueries: [Int: MLXArray] = [:]
        func rows(_ name: String) -> Int? { expected[name]?.first }
        func dense(_ base: String, inputs: Int) -> MLXArray? {
            guard let w = weights[base + ".weight"] else { return nil }
            guard let scales = weights[base + ".scales"] else { return w }
            let f = Inkling2NativeSwitchLinear.format(codes: w, scales: scales, inputDims: inputs)
            return MLX.dequantized(
                w, scales: scales, biases: f.mode == .affine ? weights[base + ".biases"] : nil,
                groupSize: f.groupSize, bits: f.bits, mode: f.mode,
                globalScale: f.mode == .nvfp4
                    ? weights[base + ".global_scale"]?.reshaped([1]).asType(.float32) : nil,
                dtype: scales.dtype == .uint8 ? .bfloat16 : scales.dtype)
        }
        func split(_ m: MLXArray, _ parts: [(String, Int?)]) {
            var start = 0
            for (name, count) in parts {
                guard let count else {
                    RuntimeDiagnostics.write(Data(
                        "[vmlx][inkling2/load] fused-projection split: no parameter \(name)\n".utf8))
                    return
                }
                out[name] = m[start ..< start + count]
                start += count
            }
            if start != m.dim(0) {
                RuntimeDiagnostics.write(Data(
                    "[vmlx][inkling2/load] fused-projection split covers \(start) of \(m.dim(0)) rows for \(parts.first?.0 ?? "?")\n".utf8))
            }
        }
        func taps(_ v: MLXArray, _ name: String) -> MLXArray {
            // `[C, K]` -> this tree's `[K, C]`.
            if let shape = expected[name], v.ndim == 2, v.shape != shape, v.shape.reversed() == shape {
                return v.transposed(1, 0)
            }
            return v
        }
        var consumed = Set<String>()
        for key in weights.keys where key.hasPrefix("model.layers.") {
            let parts = key.split(separator: ".", maxSplits: 3).map(String.init)
            guard parts.count == 4, let idx = Int(parts[2]) else { continue }
            let pre = "model.layers.\(idx)."
            let sa = pre + "self_attn."
            let rest = parts[3]
            let hidden = expected[sa + "q_proj.weight"]?.last ?? 0
            switch rest {
            case "attn.in_proj.weight":
                if let m = dense(pre + "attn.in_proj", inputs: hidden) {
                    split(m, ["q_proj", "k_proj", "v_proj", "r_proj", "gate_proj", "sink_proj"].map {
                        (sa + $0 + ".weight", rows(sa + $0 + ".weight"))
                    })
                }
            case "attn.q_proj.weight":
                if let m = dense(pre + "attn.q_proj", inputs: hidden) {
                    split(m, ["q_proj", "r_proj", "gate_proj", "sink_proj"].map {
                        (sa + $0 + ".weight", rows(sa + $0 + ".weight"))
                    })
                }
            case "attn.k_proj.weight", "attn.v_proj.weight":
                let n = rest.hasPrefix("attn.k") ? "k" : "v"
                if let m = dense(pre + "attn.\(n)_proj", inputs: hidden) {
                    split(m, [
                        (sa + "\(n)_proj.weight", rows(sa + "\(n)_proj.weight")),
                        (sa + "\(n)_sconv.weight_proj.weight", rows(sa + "\(n)_sconv.weight_proj.weight")),
                    ])
                }
            case "attn.o_proj.weight":
                let name = sa + "o_proj.weight"
                if let m = dense(pre + "attn.o_proj", inputs: expected[name]?.last ?? 0) { out[name] = m }
            case "attn.k_static", "attn.v_static":
                out[sa + (rest == "attn.k_static" ? "k" : "v") + "_sconv.static_weight"] = weights[key]
            case "attn.k_conv", "attn.v_conv":
                let name = sa + (rest == "attn.k_conv" ? "k" : "v") + "_sconv.weight"
                out[name] = taps(weights[key]!, name)
            case "attn_conv", "mlp_conv":
                let name = pre + (rest == "attn_conv" ? "post_attention" : "post_mlp") + "_convolution.weight"
                out[name] = taps(weights[key]!, name)
            case "attn.q_scale": out[sa + "q_l2_clip.scale"] = weights[key]
            case "attn.r_scale": out[sa + "rel_l2_clip.scale"] = weights[key]
            case "attn.rel_proj": out[sa + "rel_logits_proj.proj"] = weights[key]
            case "mlp.router_weight": out[pre + "mlp.gate.weight"] = weights[key]
            case "mlp.router_bias": out[pre + "mlp.gate.e_score_correction_bias"] = weights[key]
            case "attn_input.query", "mlp_input.query":
                let name = pre + (rest == "attn_input.query" ? "attention_input" : "mlp_input") + ".query"
                let v = weights[key]!
                out[name] = expected[name].map { v.reshaped($0) } ?? v
            case "loop_message.query":
                messageQueries[idx] = weights[key]
            case "loop_down":
                // [4, 2, H, r] -> per gate [r, 2H].
                let v = weights[key]!
                for (i, g) in ["q", "k", "v", "mlp"].enumerated() {
                    let d = v[i]  // [2, H, r]
                    out[pre + "\(g)_message_gate.down_proj.weight"] =
                        d.transposed(2, 0, 1).reshaped([d.dim(2), 2 * d.dim(1)])
                }
            case "loop_gain_forget":
                // [4, r, 2H] -> gain / forget `[H, r]`.
                let v = weights[key]!
                for (i, g) in ["q", "k", "v", "mlp"].enumerated() {
                    let h = v.dim(2) / 2
                    out[pre + "\(g)_message_gate.message_gate_proj.weight"] = v[i][0..., ..<h].T
                    out[pre + "\(g)_message_gate.forget_gate_proj.weight"] = v[i][0..., h...].T
                }
            default:
                if rest.hasPrefix("mlp.experts.w13.") {
                    out[pre + "mlp.experts.gate_up_proj." + rest.dropFirst("mlp.experts.w13.".count)] = weights[key]
                } else if rest.hasPrefix("mlp.experts.w2.") {
                    out[pre + "mlp.experts.down_proj." + rest.dropFirst("mlp.experts.w2.".count)] = weights[key]
                } else if rest.hasPrefix("attn.") && (rest.hasSuffix(".scales") || rest.hasSuffix(".biases")
                    || rest.hasSuffix(".global_scale"))
                {
                    // consumed with its matrix
                } else {
                    out[key] = weights[key]  // norms: attn_norm, mlp_norm, {q,k,v}_input_norm
                }
            }
            consumed.insert(key)
        }
        for (key, value) in weights where !consumed.contains(key) {
            switch key {
            case "model.embed.weight", "model.embed.scales", "model.embed.biases":
                if key == "model.embed.weight" {
                    let name = "model.embed_tokens.weight"
                    out[name] = dense("model.embed", inputs: expected[name]?.last ?? 0)
                }
            case "model.final_input.query":
                out[key] = expected[key].map { value.reshaped($0) } ?? value
            default:
                out[key] = value  // lm_head.*, model.embed_norm, model.norm, mtp / vision
            }
        }
        if !messageQueries.isEmpty {
            let ordered = loopCore.compactMap { messageQueries[$0] }
            if ordered.count == loopCore.count {
                out["model.message_input.query"] = MLX.stacked(ordered, axis: 0)
            }
        }
        return out
    }

    /// Key prefixes of other runtimes' trees -> this model's checkpoint-style prefixes.
    static func normalizingLayoutKeys(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        out.reserveCapacity(weights.count)
        for (key, value) in weights {
            var k = key
            if k.hasPrefix("language_model.model.") {
                k = "model." + k.dropFirst("language_model.model.".count)
            } else if k.hasPrefix("language_model.lm_head") {
                k = String(k.dropFirst("language_model.".count))
            }
            out[k] = value
        }
        return out
    }

    /// Dense layers (tree-path suffixes) the runtime runs quantized as `QuantizedLinear` /
    /// `QuantizedEmbedding`: the attention-input projections the fused decode / verify paths
    /// read as quantized GEMVs, the output projection, the embedding and the unembedding. Every
    /// other quantized dense layer (sink, dynamic-tap projections) is dequantized at load.
    static let runtimeQuantizedDenseLayers: [String] = [
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.r_proj",
        "self_attn.gate_proj", "self_attn.o_proj", "embed_tokens", "lm_head",
    ]

    /// Quantized dense layers (`X.weight` + `X.scales` [+ `X.biases`, `X.global_scale`]) are
    /// dequantized to the model dtype at load (unless listed in `runtimeQuantizedDenseLayers`):
    /// the fused decode path reads these weights directly, and the values are exactly the ones
    /// the bundle encodes. Bits and group size
    /// come from this model's own shape for the layer (never from a converter's config, which
    /// can be stale), the mode from the scale dtype. Routed experts and the vision tower are
    /// left as they are.
    static func dequantizingDenseLayers(
        _ weights: [String: MLXArray], expected: [String: [Int]]
    ) -> [String: MLXArray] {
        var out = weights
        for key in weights.keys where key.hasSuffix(".scales") {
            let base = String(key.dropLast(".scales".count))
            if base.contains(".experts.") || base.contains(".switch_mlp.")
                || base.hasPrefix("model.visual") || base.hasPrefix("model.vision_tower")
                || base.hasPrefix("vision_tower") || base.hasPrefix("model.audio")
            {
                continue
            }
            guard let w = weights[base + ".weight"], let scales = weights[key] else { continue }
            let tree = Inkling2LoopModel.mapKey(base + ".weight")
            let treeBase = String(tree.dropLast(".weight".count))
            if runtimeQuantizedDenseLayers.contains(where: { treeBase.hasSuffix($0) }) { continue }
            guard let shape = expected[tree], let inputs = shape.last, inputs > 0 else {
                RuntimeDiagnostics.write(Data(
                    "[vmlx][inkling2/load] quantized \(base) has no matching parameter; left as stored\n".utf8))
                continue
            }
            let f = Inkling2NativeSwitchLinear.format(codes: w, scales: scales, inputDims: inputs)
            let dtype: DType = scales.dtype == .uint8 ? .bfloat16 : scales.dtype
            let global = weights[base + ".global_scale"] ?? weights[base + ".weight_global_scale"]
            var v = MLX.dequantized(
                w, scales: scales, biases: f.mode == .affine ? weights[base + ".biases"] : nil,
                groupSize: f.groupSize, bits: f.bits, mode: f.mode,
                globalScale: f.mode == .nvfp4 ? global?.reshaped([1]).asType(.float32) : nil,
                dtype: dtype)
            if f.mode == .nvfp4, global == nil, let s2 = weights[base + ".weight_scale_2"] {
                v = (v.asType(.float32) * s2.asType(.float32).reshaped([1])).asType(dtype)
            }
            out[base + ".weight"] = v
            for aux in [".scales", ".biases", ".global_scale", ".weight_global_scale", ".weight_scale_2"] {
                out.removeValue(forKey: base + aux)
            }
        }
        return out
    }

    /// `(…mlp., leg, leaf)` for a community routed-expert tensor (tree-mapped key): HF
    /// `experts.{gate_up_proj,down_proj}` or mlx-vlm `switch_mlp.{gate_proj,up_proj,down_proj}`.
    static func communityExpertKey(_ key: String) -> (String, String, String)? {
        let leaves = ["weight", "scales", "biases", "global_scale", "weight_global_scale", "weight_scale_2"]
        for (infix, legs) in [
            (".mlp.experts.", ["gate_up_proj", "down_proj"]),
            (".mlp.switch_mlp.", ["gate_proj", "up_proj", "down_proj"]),
        ] {
            guard let r = key.range(of: infix) else { continue }
            let rest = key[r.upperBound...]
            for leg in legs where rest.hasPrefix(leg + ".") {
                let leaf = String(rest.dropFirst(leg.count + 1))
                guard leaves.contains(leaf) else { return nil }
                let style = infix == ".mlp.experts." ? "hf_" : "vlm_"
                return (String(key[..<r.lowerBound]) + ".mlp.", style + leg, leaf)
            }
        }
        return nil
    }

    /// Community expert legs -> this model's `switch_mlp.{gate_up_proj,down_proj}` parameters
    /// (`codes`, `block_scales`, `expert_scale`, `affine_biases`). gate/up rows end up
    /// ROW-INTERLEAVED like the official `w13` (row 2i = gate_i, row 2i+1 = up_i). The mlx-lm port
    /// (ml-explore/mlx-lm#2086, every `model.language_model.*` MLX bundle) renames `w13` to
    /// `gate_up_proj` and keeps its row order (it splits with `[0::2]` / `[1::2]`), so that tensor
    /// is used as stored; mlx-vlm splits gate and up into two tensors, re-interleaved here.
    /// (Transformers' own `gate_up_proj` is gate half then up half, but it is only ever saved
    /// unquantized, which this module does not load.) Row permutations are exact for any per-row
    /// quantization.
    static func canonicalExperts(_ parts: [String: [String: MLXArray]]) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        var layers = Set<String>()
        for key in parts.keys { layers.insert(String(key.split(separator: "|")[0])) }
        func leg(_ base: String, _ name: String) -> [String: MLXArray]? { parts[base + "|" + name] }
        func interleave(_ g: MLXArray, _ u: MLXArray) -> MLXArray {
            let (e, i) = (g.dim(0), g.dim(1))
            return MLX.contiguous(
                MLX.stacked([g, u], axis: 2).reshaped([e, 2 * i] + Array(g.shape.dropFirst(2))))
        }
        func factor(_ p: [String: MLXArray], experts: Int) -> MLXArray? {
            if let g = p["global_scale"] ?? p["weight_global_scale"] {
                return (g.asType(.float32) / Float(448 * 6)).reshaped([-1])
                    * MLXArray.ones([experts], dtype: .float32)
            }
            if let s2 = p["weight_scale_2"] {
                return s2.asType(.float32).reshaped([-1]) * MLXArray.ones([experts], dtype: .float32)
            }
            return nil
        }
        func emit(_ prefix: String, codes: MLXArray, scales: MLXArray, biases: MLXArray?,
                  factor f: MLXArray?) {
            let isFloatScale = scales.dtype != .uint8
            out[prefix + "codes"] = codes.dtype == .uint32 ? codes : MLX.contiguous(codes).view(dtype: .uint32)
            out[prefix + "block_scales"] = scales
            out[prefix + "expert_scale"] = f ?? MLXArray.ones([codes.dim(0)], dtype: .float32)
            if isFloatScale, let biases { out[prefix + "affine_biases"] = biases }
        }
        for base in layers {
            let target = base + "switch_mlp."
            if let gu = leg(base, "hf_gate_up_proj") {
                guard let w = gu["weight"], let s = gu["scales"] else {
                    Self.unsupportedExperts(base, "unquantized HF gate_up_proj"); continue
                }
                emit(target + "gate_up_proj.", codes: w, scales: s, biases: gu["biases"],
                     factor: factor(gu, experts: w.dim(0)))
            } else if let g = leg(base, "vlm_gate_proj"), let u = leg(base, "vlm_up_proj") {
                guard let gw = g["weight"], let gs = g["scales"], let uw = u["weight"], let us = u["scales"]
                else { Self.unsupportedExperts(base, "unquantized switch_mlp gate/up"); continue }
                let fg = factor(g, experts: gw.dim(0))
                let fu = factor(u, experts: uw.dim(0))
                if let fg, let fu, MLX.abs(fg - fu).max().item(Float.self) > 0 {
                    // One per-expert factor serves both halves of the fused gate/up gather.
                    Self.unsupportedExperts(base, "gate and up carry different per-expert scales")
                    continue
                }
                var biases: MLXArray? = nil
                if let gb = g["biases"], let ub = u["biases"] { biases = interleave(gb, ub) }
                emit(target + "gate_up_proj.", codes: interleave(gw, uw), scales: interleave(gs, us),
                     biases: biases, factor: fg ?? fu)
            }
            if let d = leg(base, "hf_down_proj") ?? leg(base, "vlm_down_proj") {
                guard let w = d["weight"], let s = d["scales"] else {
                    Self.unsupportedExperts(base, "unquantized down_proj"); continue
                }
                emit(target + "down_proj.", codes: w, scales: s, biases: d["biases"],
                     factor: factor(d, experts: w.dim(0)))
            }
        }
        return out
    }

    private static func unsupportedExperts(_ base: String, _ why: String) {
        RuntimeDiagnostics.write(Data(
            "[vmlx][inkling2/load] routed experts at \(base) not loaded: \(why)\n".utf8))
    }

    private static let renames: [(String, String)] = [
        (".controllers.attn.", ".q_message_gate."),
        (".controllers.k.", ".k_message_gate."),
        (".controllers.v.", ".v_message_gate."),
        (".controllers.mlp.", ".mlp_message_gate."),
        ("_message_gate.down_weight.", "_message_gate.down_proj.weight."),
        ("_message_gate.gain_weight.", "_message_gate.message_gate_proj.weight."),
        ("_message_gate.forget_weight.", "_message_gate.forget_gate_proj.weight."),
        (".attn.wq_du.", ".self_attn.q_proj."),
        (".attn.wk_dv.", ".self_attn.k_proj."),
        (".attn.wv_dv.", ".self_attn.v_proj."),
        (".attn.wr_du.", ".self_attn.r_proj."),
        (".attn.wo_ud.", ".self_attn.o_proj."),
        (".attn.gate_attn.", ".self_attn.gate_proj."),
        (".attn.attn_sink_proj.", ".self_attn.sink_proj."),
        (".attn_norm.", ".input_layernorm."),
        (".mlp_norm.", ".post_attention_layernorm."),
        (".q_input_norm.", ".q_input_layernorm."),
        (".k_input_norm.", ".k_input_layernorm."),
        (".v_input_norm.", ".v_input_layernorm."),
        (".attn_sconv.", ".post_attention_convolution."),
        (".mlp_sconv.", ".post_mlp_convolution."),
        ("sconv.static_w.", "sconv.static_weight."),
        (".mlp.gate.bias.", ".mlp.gate.e_score_correction_bias."),
        // mlx-vlm tree names (language_model.model.* bundles).
        (".mlp.gate_weight.", ".mlp.gate.weight."),
        (".mlp.e_score_correction_bias.", ".mlp.gate.e_score_correction_bias."),
        (".self_attn.q_scale.", ".self_attn.q_l2_clip.scale."),
        (".self_attn.rel_scale.", ".self_attn.rel_l2_clip.scale."),
        (".self_attn.rel_proj.", ".self_attn.rel_logits_proj.proj."),
        ("_sconv.conv.weight.", "_sconv.weight."),
        ("_convolution.conv.weight.", "_convolution.weight."),
        (".attn.", ".self_attn."),
    ]

    /// Checkpoint key -> module-tree key (jang_tools `map_key`); idempotent on tree names.
    static func mapKey(_ key: String) -> String {
        var n = key
        if n.hasPrefix("model.llm.") {
            n = "model." + n.dropFirst("model.llm.".count)
        } else if n.hasPrefix("model.language_model.") {
            n = "model." + n.dropFirst("model.language_model.".count)
        }
        if n.hasPrefix("model.attn_res.attn_queries."),
            let i = Int(n.dropFirst("model.attn_res.attn_queries.".count))
        {
            return "model.layers.\(i + 1).attention_input.query"
        }
        if n.hasPrefix("model.attn_res.mlp_queries."),
            let i = Int(n.dropFirst("model.attn_res.mlp_queries.".count))
        {
            return "model.layers.\(i).mlp_input.query"
        }
        if n == "model.attn_res.output_query" { return "model.final_input.query" }
        if n == "model.message_queries" { return "model.message_input.query" }
        n += "."
        for (a, b) in renames {
            n = n.replacingOccurrences(of: a, with: b)
        }
        n = n.replacingOccurrences(of: ".embed_tokens.embed_norm.", with: ".embed_norm.")
        n = n.replacingOccurrences(of: ".conv1d.weight.", with: ".weight.")
        n.removeLast()
        if n == "model.embed.weight" || n == "model.embed" {
            n = n.replacingOccurrences(of: "model.embed", with: "model.embed_tokens")
        } else if n == "model.unembed.weight" || n == "model.unembed" {
            n = n.replacingOccurrences(of: "model.unembed", with: "lm_head")
        } else if n.hasPrefix("model.lm_head") {
            n = String(n.dropFirst("model.".count))
        }
        return n
    }
}

extension Inkling2LoopModel {
    /// The unembedding as a dense `[V, H]` matrix: the weight itself, or an affine head dequantized (transient).
    func denseUnembedWeight() -> MLXArray {
        if let q = lmHead as? QuantizedLinear {
            return MLX.dequantized(
                q.weight, scales: q.scales, biases: q.biases, groupSize: q.groupSize, bits: q.bits, mode: q.mode)
        }
        return lmHead.weight
    }
}

extension Inkling2LoopModel: SafetensorsLoadKeyExcluding {
    public func excludeFromGenericSafetensorsLoad(key: String) -> Bool { customTensorNames.contains(key) }
    public var requiresExactTensorMmapBuffers: Bool { !customTensorNames.isEmpty }
}

extension Inkling2LoopModel: LoRAModel {
    public var loraLayers: [Module] { model.layers }
}

/// Inkling's template renders an assistant tool-call turn (`reasoning_content` + `<|message_model|>NAME
/// <|content_invoke_tool_json|>{"name":…,"args":{…}}<|end_message|><|content_model_end_sampling|>`) back to exactly
/// the tokens the model generated, so the tool-call turn stores a post-answer boundary and the request carrying the
/// tool result restores past it instead of re-prefilling the conversation.
extension Inkling2LoopModel: ToolCallTurnStopTokenModel {
    /// `<|content_model_end_sampling|>` closes every assistant turn in the template, tool calls included.
    public var toolCallTurnStopTokenIds: Set<Int> { Set(args.eosTokenId ?? [200006]) }
}
