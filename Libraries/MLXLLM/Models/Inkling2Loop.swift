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
// Cache layout (96 entries): `[0, 48)` are `KVCacheSimple`, one per (layer, pass) slot
// (`cacheSlot(layer:pass:)`); `[48, 96)` are single-slot `MambaCache`s holding that slot's four
// short-conv states packed as one fp32 `[B, K-1, 2*kvDim + 2*hidden]` array. Both are classes
// the cache coordinator already stores, trims, pages and serialises (the Qwen 3.5 hybrid
// layout), so prefix / paged / SSD caching needs no Inkling-specific cache code. Sliding layers
// keep their full KV history in the simple cache and attend to its last `window - 1 + T` rows.

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
    }

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

/// Static depthwise causal conv: `(conv(x) + x) / sqrt(K + 1)`. `weight` is `[K, C]`, tap
/// `K - 1` weighting the CURRENT token (torch conv1d with left padding K-1).
final class Inkling2ShortConv: Module {
    var weight: MLXArray
    let kernel: Int

    init(channels: Int, kernel: Int) {
        weight = MLXArray.zeros([kernel, channels])
        self.kernel = kernel
    }

    /// Returns `(output in x.dtype, new state [B, K-1, C] fp32)`.
    func callAsFunction(_ x: MLXArray, state: MLXArray?) -> (MLXArray, MLXArray) {
        let K = kernel
        let pad = K - 1
        let T = x.dim(1)
        let xf = x.asType(f32)
        let st = state ?? MLXArray.zeros([x.dim(0), pad, x.dim(2)], dtype: f32)
        let xp = MLX.concatenated([st, xf], axis: 1)
        let w = weight.asType(f32)
        var y = xp[0..., 0 ..< T, 0...] * w[0]
        for j in 1 ..< K {
            y = y + xp[0..., j ..< (j + T), 0...] * w[j]
        }
        let out = ((y + xf) / Float(K + 1).squareRoot()).asType(x.dtype)
        return (out, xp[0..., (xp.dim(1) - pad)..., 0...])
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

    func callAsFunction(_ x: MLXArray, dynamicInput: MLXArray, state: MLXArray?) -> (
        MLXArray, MLXArray
    ) {
        let K = kernel
        let pad = K - 1
        let (B, T, C) = (x.dim(0), x.dim(1), x.dim(2))
        let nh = heads
        let dyn = weightProj(dynamicInput).asType(f32).reshaped([B, T, nh, 1, K])
        let st = staticWeight.asType(f32)
        let xf = x.asType(f32)
        let s0 = state ?? MLXArray.zeros([B, pad, C], dtype: f32)
        let xp = MLX.concatenated([s0, xf], axis: 1)
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
    }

    /// `x` feeds q / r / gate / sinks; `keyInput` / `valueInput` feed k / v (nil → `x`; full
    /// layers always pass them). Conv states: `(k, v)` in, `(k, v)` out.
    func callAsFunction(
        _ x: MLXArray, keyInput: MLXArray?, valueInput: MLXArray?, kv: KVCacheSimple?,
        convK: MLXArray?, convV: MLXArray?, ones: MLXArray
    ) -> (MLXArray, MLXArray, MLXArray) {
        let (B, T) = (x.dim(0), x.dim(1))
        let ki = keyInput ?? x
        let vi = valueInput ?? x
        var k: MLXArray
        var v: MLXArray
        let newK: MLXArray
        let newV: MLXArray
        if let conv = kSconv as? Inkling2ShortConv {
            (k, newK) = conv(kProj(ki), state: convK)
            (v, newV) = (vSconv as! Inkling2ShortConv)(vProj(vi), state: convV)
        } else {
            (k, newK) = (kSconv as! Inkling2DynamicShortConv)(
                kProj(ki), dynamicInput: ki, state: convK)
            (v, newV) = (vSconv as! Inkling2DynamicShortConv)(
                vProj(vi), dynamicInput: vi, state: convV)
        }
        var q = qProj(x)
        var r = rProj(x)
        let gate = gateProj(x)
        let sinks = sinkProj(x)  // [B, T, nq]

        q = l2Clip(q.reshaped([B, T, nq, hd]), threshold: qkClip, scale: qL2Clip.scale)
        r = l2Clip(r.reshaped([B, T, nq, dRel]), threshold: relClip, scale: relL2Clip.scale)
        k = l2Clip(k.reshaped([B, T, nkv, hd]), threshold: qkClip).transposed(0, 2, 1, 3)
        v = unweightedRMS(v.reshaped([B, T, nkv, hd]), ones: ones[0 ..< hd], eps: eps)
            .transposed(0, 2, 1, 3)

        // Absolute positions; `seen` is read BEFORE the cache update.
        let seen = kv?.offset ?? 0
        if !isSliding, seen + T > Int(logFloor) {
            // Log scaling of q AND the relative states (after the clips, before the bias
            // gather). Exactly 1 for positions below n_floor, so it is skipped there.
            let qPos = MLXArray(Int32(seen) ..< Int32(seen + T)).asType(f32)
            let tau = (1 + logAlpha * MLX.log(MLX.maximum((qPos + 1) / logFloor, 1)))
                .reshaped([1, T, 1, 1])
            q = (q.asType(f32) * tau).asType(q.dtype)
            r = (r.asType(f32) * tau).asType(r.dtype)
        }
        q = q.transposed(0, 2, 1, 3)  // [B, nq, T, hd]

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
        let rel = MLX.matmul(r, relLogits.proj.asType(r.dtype)).transposed(0, 2, 1, 3)  // [B,H,T,E]
        let out: MLXArray
        if T == 1 {
            // Decode: keys sit at kStart ... seen, distance runs S-1 ... 0.
            let n = min(S, relExtent)
            let idx = MLXArray((0 ..< Int32(n)).reversed())  // n-1 ... 0
            var bias = MLX.take(rel, idx, axis: -1)  // [B, H, 1, n]
            if n < S {
                bias = MLX.concatenated(
                    [MLXArray.zeros([B, nq, 1, S - n], dtype: bias.dtype), bias], axis: -1)
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
        let merged = out.transposed(0, 2, 1, 3).reshaped([B, T, nq * hd])
        let gated = (merged.asType(f32) * MLX.sigmoid(gate.asType(f32))).asType(merged.dtype)
        return (oProj(gated), newK, newV)
    }
}

// MARK: - MoE

/// Routed experts kept in the source checkpoint's own encoding, run by `gatherQuantizedMM`.
///
/// * NVFP4 (ModelOpt): MLX `mode: .nvfp4` (packed e2m1, even element in the low nibble, e4m3
///   per-16 scales). This MLX has no per-expert `global_scale` on `gather_qmm`, so the
///   per-expert factor `amax / outer_scale_denominator` is applied to the OUTPUT rows instead:
///   `x @ (W_e * s_e)^T == s_e * (x @ W_e^T)` for a scalar `s_e`.
/// * MXFP8: `mode: .mxfp8` (e4m3 codes, ue8m0 per-32 scales), no extra factor.
///
/// Parameter names avoid `weight` / `scales` so the generic loader never re-wraps this module
/// as an affine `QuantizedSwitchLinear`.
final class Inkling2NativeSwitchLinear: Module {
    var codes: MLXArray
    @ParameterInfo(key: "block_scales") var blockScales: MLXArray
    @ParameterInfo(key: "expert_scale") var expertScale: MLXArray

    init(inputDims: Int, outputDims: Int, experts: Int) {
        codes = MLXArray.zeros([experts, outputDims, inputDims / 8], dtype: .uint32)
        _blockScales.wrappedValue = MLXArray.zeros(
            [experts, outputDims, inputDims / 16], dtype: .uint8)
        _expertScale.wrappedValue = MLXArray.ones([experts], dtype: f32)
    }

    var isNVFP4: Bool { codes.dim(-1) / blockScales.dim(-1) == 2 }

    func callAsFunction(_ x: MLXArray, _ indices: MLXArray, sorted: Bool) -> MLXArray {
        let nv = isNVFP4
        return MLX.gatherQuantizedMM(
            x, codes, scales: blockScales, biases: nil, rhsIndices: indices, transpose: true,
            groupSize: nv ? 16 : 32, bits: nv ? 4 : 8, mode: nv ? .nvfp4 : .mxfp8,
            sortedIndices: sorted)
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

    init(_ args: Inkling2LoopConfiguration) {
        let (H, I, E) = (args.hiddenSize, args.intermediateSize, args.nRoutedExperts)
        _gateUp.wrappedValue = Inkling2NativeSwitchLinear(
            inputDims: H, outputDims: 2 * I, experts: E)
        _down.wrappedValue = Inkling2NativeSwitchLinear(inputDims: I, outputDims: H, experts: E)
        inter = I
        gateCap = args.mlpGateSoftcap
        upCap = args.mlpUpSoftcap
    }

    /// `x` `[N, H]` (model dtype), `inds` `[N, k]` → `[N, k, H]` fp32 (unweighted expert outputs).
    func callAsFunction(_ x: MLXArray, _ inds: MLXArray) -> MLXArray {
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
        var gu = gateUp(xs, idx, sorted: doSort)  // [..., 1, 2I]
        if gateUp.isNVFP4 {
            gu = gu * gateUp.expertScale[idx].reshaped(idx.shape + [1, 1])
        }
        let pair = gu.reshaped(Array(gu.shape.dropLast()) + [inter, 2])
        let g = pair[.ellipsis, 0]
        let u = pair[.ellipsis, 1]
        // softcap(g, 4) * sigmoid(g) * softcap(u, 25) in fp32; the sigmoid sees the RAW gate.
        // The reference rounds the activation to the model dtype before the down projection.
        let act = (softcap(g, gateCap) * MLX.sigmoid(g) * softcap(u, upCap))
            .asType(x.dtype).asType(f32)
        var y = down(act, idx, sorted: doSort)  // [..., 1, H]
        if down.isNVFP4 {
            y = y * down.expertScale[idx].reshaped(idx.shape + [1, 1])
        }
        if let inverse {
            y = y[inverse].reshaped([N, k, 1, y.dim(-1)])
        }
        return y.squeezed(axis: -2)
    }
}

final class Inkling2Router: Module {
    var weight: MLXArray
    @ParameterInfo(key: "e_score_correction_bias") var bias: MLXArray
    let topK: Int
    let routeScale: Float

    init(_ args: Inkling2LoopConfiguration) {
        weight = MLXArray.zeros([args.nRoutedExperts, args.hiddenSize])
        _bias.wrappedValue = MLXArray.zeros([args.nRoutedExperts])
        topK = args.numExpertsPerTok
        routeScale = args.routeScale
    }

    /// scores = sigmoid(x @ W^T); SELECTION by top-k of scores + bias; weights = the selected
    /// RAW scores / (sum + 1e-20) * route_scale. fp32.
    func callAsFunction(_ flat: MLXArray) -> (MLXArray, MLXArray) {
        let logits = MLX.matmul(flat.asType(f32), weight.asType(f32).T)
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
    @ModuleInfo(key: "switch_mlp") var switchMLP: Inkling2SwitchMLP

    init(_ args: Inkling2LoopConfiguration) {
        _gate.wrappedValue = Inkling2Router(args)
        _switchMLP.wrappedValue = Inkling2SwitchMLP(args)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (B, T, H) = (x.dim(0), x.dim(1), x.dim(2))
        let flat = x.reshaped([-1, H])
        let (inds, w) = gate(flat)
        let y = switchMLP(flat, inds)  // [N, k, H] fp32
        // Reference: (expert_out * weight_fp32) index_add_ into an fp32 buffer.
        return MLX.sum(y * w.expandedDimensions(axis: -1), axis: -2).asType(x.dtype)
            .reshaped([B, T, H])
    }
}

// MARK: - Depth attention (block residuals, loop messages, message gates)

/// `Inkling2BlockAttentionResidual`: fixed learned fp32 queries attend over the depth axis
/// (completed block states + the current partial block), keys = values = the unweighted-RMS
/// normalized states, scale H^-0.5. Per token; nothing goes in the KV cache.
final class Inkling2BlockAttentionResidual: Module {
    var query: MLXArray

    init(hidden: Int, queries: Int) {
        query = queries == 1 ? MLXArray.zeros([hidden]) : MLXArray.zeros([queries, hidden])
    }

    /// Returns `[B, T, nq, H]` fp32.
    func callAsFunction(_ states: [MLXArray], _ partial: MLXArray, ones: MLXArray) -> MLXArray {
        let stacked = MLXFast.rmsNorm(
            MLX.stacked((states + [partial]).map { $0.asType(f32) }, axis: -2), weight: ones,
            eps: 1e-6)  // [B, T, S, H]
        let H = stacked.dim(-1)
        let q = query.asType(f32).reshaped([-1, H])
        let scores = MLX.matmul(stacked, q.T) * (1 / Float(H).squareRoot())  // [B,T,S,nq]
        let p = MLX.softmax(scores, axis: -2, precise: true)
        return MLX.matmul(p.swappedAxes(-1, -2), stacked)
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

    init(hidden: Int, rank: Int, eps: Float) {
        _downProj.wrappedValue = Linear(2 * hidden, rank, bias: false)
        _forgetProj.wrappedValue = Linear(rank, hidden, bias: false)
        _gainProj.wrappedValue = Linear(rank, hidden, bias: false)
        self.eps = eps
    }

    func callAsFunction(_ h: MLXArray, _ msg: MLXArray, ones: MLXArray) -> MLXArray {
        let hf = h.asType(f32)
        let mf = msg.asType(f32)
        let g0 = MLX.concatenated(
            [
                MLXFast.rmsNorm(hf, weight: ones, eps: eps),
                MLXFast.rmsNorm(mf, weight: ones, eps: eps),
            ], axis: -1)
        let wdt: DType = (downProj as? QuantizedLinear)?.scales.dtype ?? downProj.weight.dtype
        let g = downProj(g0.asType(wdt)).asType(f32)
        let forget = MLX.tanh(MLX.matmul(g, forgetProj.weight.asType(f32).T))
        let gain = MLX.tanh(MLX.matmul(g, gainProj.weight.asType(f32).T))
        return (1 - forget) * hf + gain * mf
    }
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

    init(_ args: Inkling2LoopConfiguration, index: Int) {
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
        _mlp.wrappedValue = Inkling2MoE(args)
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
    ///   a = post_attention_convolution(self_attn(norm(attn_in)))
    ///   block boundary: block_states += [h]; residual = a   else residual = h + a
    ///   m = post_mlp_convolution(mlp(post_attention_layernorm(mlp_input(block_states, residual))))
    ///   return residual + m
    /// The caller appends the layer INPUT to its own block_states at boundaries.
    /// `messages` `[B, T, 4, H]` (q, k, v, mlp) only in loop passes after the first.
    func callAsFunction(
        _ h: MLXArray, blockStates: [MLXArray], kv: KVCacheSimple?, conv: MambaCache?,
        messages: MLXArray?, ones: MLXArray
    ) -> MLXArray {
        let dtype = h.dtype
        // Packed conv state [B, K-1, kvDim | kvDim | H | H] (fp32).
        let packed = conv?[0]
        func part(_ lo: Int, _ width: Int) -> MLXArray? {
            packed.map { $0[0..., 0..., lo ..< (lo + width)] }
        }
        let c = convKV
        let mq = messages.map { $0[.ellipsis, 0, 0...] }
        let mk = messages.map { $0[.ellipsis, 1, 0...] }
        let mv = messages.map { $0[.ellipsis, 2, 0...] }
        let mm = messages.map { $0[.ellipsis, 3, 0...] }

        var attnOut: MLXArray
        var newK: MLXArray
        var newV: MLXArray
        if isSliding {
            let x =
                attentionInput.map { $0(blockStates, h, ones: ones)[.ellipsis, 0, 0...] } ?? h
            let ln = inputLN!
            if let mq, let mk, let mv {
                (attnOut, newK, newV) = selfAttn(
                    ln(qGate!(x, mq, ones: ones)).asType(dtype),
                    keyInput: ln(kGate!(x, mk, ones: ones)).asType(dtype),
                    valueInput: ln(vGate!(x, mv, ones: ones)).asType(dtype),
                    kv: kv, convK: part(0, c), convV: part(c, c), ones: ones)
            } else {
                (attnOut, newK, newV) = selfAttn(
                    ln(x).asType(dtype), keyInput: nil, valueInput: nil, kv: kv,
                    convK: part(0, c), convV: part(c, c), ones: ones)
            }
        } else {
            var qh = h
            var kh = h
            var vh = h
            if let attentionInput {
                let three = attentionInput(blockStates, h, ones: ones)
                qh = three[.ellipsis, 0, 0...]
                kh = three[.ellipsis, 1, 0...]
                vh = three[.ellipsis, 2, 0...]
            }
            if let mq, let mk, let mv {
                qh = qGate!(qh, mq, ones: ones)
                kh = kGate!(kh, mk, ones: ones)
                vh = vGate!(vh, mv, ones: ones)
            }
            (attnOut, newK, newV) = selfAttn(
                qInputLN!(qh).asType(dtype), keyInput: kInputLN!(kh).asType(dtype),
                valueInput: vInputLN!(vh).asType(dtype), kv: kv, convK: part(0, c),
                convV: part(c, c), ones: ones)
        }
        let (a, newAttn) = postAttnConv(attnOut, state: part(2 * c, hidden))

        var residual = a.asType(f32)
        var states = blockStates
        if isBlockBoundary {
            states.append(h)
        } else {
            residual = h.asType(f32) + residual
        }
        var m = mlpInput(states, residual, ones: ones)[.ellipsis, 0, 0...]
        if let mm { m = mlpGate!(m, mm, ones: ones) }
        m = mlp(postAttnLN(m).asType(dtype))
        let (mo, newMLP) = postMLPConv(m, state: part(2 * c + hidden, hidden))
        if let conv {
            conv[0] = MLX.concatenated([newK, newV, newAttn, newMLP], axis: -1)
            conv.offset += h.dim(1)
        }
        return (residual + mo.asType(f32)).asType(dtype)
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
    /// layer of the pass, only the queries differ. Returns `[B, T, loopLayers * 4, H]` fp32.
    func callAsFunction(_ states: [MLXArray], _ partial: MLXArray, ones: MLXArray) -> MLXArray {
        let stacked = MLXFast.rmsNorm(
            MLX.stacked((states + [partial]).map { $0.asType(f32) }, axis: -2), weight: ones,
            eps: 1e-6)  // [B, T, S, H]
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

    init(_ args: Inkling2LoopConfiguration) {
        self.args = args
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabSize, dimensions: args.hiddenSize)
        _embedNorm.wrappedValue = Inkling2RMSNorm(args.hiddenSize, eps: args.rmsNormEps)
        layers = (0 ..< args.numHiddenLayers).map { Inkling2DecoderLayer(args, index: $0) }
        _norm.wrappedValue = Inkling2RMSNorm(args.hiddenSize, eps: args.rmsNormEps)
        _finalInput.wrappedValue = Inkling2BlockAttentionResidual(
            hidden: args.hiddenSize, queries: 1)
        if args.loopPasses > 1 {
            _messageInput.wrappedValue = Inkling2MessageInput(
                loopLayers: args.numLoopLayers, hidden: args.hiddenSize)
        }
        ones = MLXArray.ones([args.hiddenSize], dtype: .float32)
    }

    /// Token embeddings INCLUDING embed_norm (reference `Inkling2NormedEmbedding`).
    func embed(_ ids: MLXArray) -> MLXArray { embedNorm(embedTokens(ids)) }

    func callAsFunction(_ ids: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = embed(ids)
        let dtype = h.dtype
        let slots = args.numCacheSlots
        if let cache {
            precondition(
                cache.count == 2 * slots,
                "inkling2_loop: expected \(2 * slots) cache entries, got \(cache.count)")
        }
        func kv(_ layer: Int, _ pass: Int) -> KVCacheSimple? {
            cache.map { $0[args.cacheSlot(layer: layer, pass: pass)] as! KVCacheSimple }
        }
        func conv(_ layer: Int, _ pass: Int) -> MambaCache? {
            cache.map { $0[slots + args.cacheSlot(layer: layer, pass: pass)] as! MambaCache }
        }

        var blockStates: [MLXArray] = []
        func run(_ ids: Range<Int>, pass: Int, messages: MLXArray?) {
            for (j, i) in ids.enumerated() {
                let layer = layers[i]
                let msgs = messages.map { $0[0..., 0..., (4 * j) ..< (4 * j + 4), 0...] }
                let next = layer(
                    h, blockStates: blockStates, kv: kv(i, pass), conv: conv(i, pass),
                    messages: msgs, ones: ones)
                if layer.isBlockBoundary { blockStates.append(h) }
                h = next
            }
        }

        let s0 = args.loopStart
        let s1 = args.loopEnd
        if args.loopPasses <= 1 {
            run(0 ..< args.numHiddenLayers, pass: 0, messages: nil)
        } else {
            run(0 ..< s0, pass: 0, messages: nil)
            let preH = h
            let preStates = blockStates
            let nPre = preStates.count
            run(s0 ..< s1, pass: 0, messages: nil)
            for p in 1 ..< args.loopPasses {
                // Reference: previous_block_states = block_states[num_pre_loop_blocks + 1:]
                // (+1 skips the pre-loop output the pass appended at its first boundary),
                // previous_hidden = the pass's last hidden; restart from the pre-loop states.
                let prev = Array(blockStates[(nPre + 1)...])
                let messages = messageInput!(prev, h, ones: ones)
                h = preH
                blockStates = preStates
                run(s0 ..< s1, pass: p, messages: messages)
            }
            run(s1 ..< args.numHiddenLayers, pass: 0, messages: nil)
        }
        h = finalInput(blockStates, h, ones: ones)[.ellipsis, 0, 0...]
        return norm(h).asType(dtype)
    }
}

// MARK: - Model

public final class Inkling2LoopModel: Module, LLMModel {
    public let vocabularySize: Int
    let args: Inkling2LoopConfiguration
    public let model: Inkling2LoopModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    public init(_ args: Inkling2LoopConfiguration) {
        self.args = args
        vocabularySize = args.vocabSize
        model = Inkling2LoopModelInner(args)
        // `embed` and `unembed` are separate tensors in the checkpoint, never tied.
        _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabSize, bias: false)
    }

    /// The per-slot cache state is path dependent and the forward is not batch aware.
    public var maximumSupportedDecodeBatchSize: Int? { 1 }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        let slots = args.numCacheSlots
        return (0 ..< slots).map { _ in KVCacheSimple() as KVCache }
            + (0 ..< slots).map { _ in MambaCache(slots: 1) as KVCache }
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
        logits(model(inputs, cache: cache))
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
        let n = ids.dim(1)
        let step = max(windowSize ?? 512, 1)
        var offset = 0
        while n - offset > step {
            try Task.checkCancellation()
            _ = model(ids[0..., offset ..< (offset + step)], cache: cache)
            MLX.eval(cache)
            offset += step
            PrefillProgressReporter.reportCompletedUnits(offset)
            MLX.Memory.clearCache()
        }
        let h = model(ids[0..., offset...], cache: cache)
        return .logits(LMOutput(logits: logits(h[0..., (h.dim(1) - 1)..., 0...])))
    }

    /// The checkpoint's fp32 tensors (attention-residual / message queries, router bias, the
    /// per-expert NVFP4 factor) are part of the numerical contract; every activation cast is
    /// explicit in this file, so the loader's blanket bf16 materialisation must not run.
    public var preservesCheckpointParameterDTypes: Bool { true }

    /// Bumped only when the forward's arithmetic for persisted KV / conv state changes.
    public var cacheComputationIdentity: String? { "inkling2_loop.v1" }

    // MARK: weights

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
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
            // Vision tower (text-only runtime), the native drafter (`mtp.safetensors`: keys
            // outside `model.`), audio: not part of this module tree.
            if key.hasPrefix("model.visual.") || key.hasPrefix("model.vision_tower.")
                || key.hasPrefix("vision_tower.") || key.hasPrefix("visual.")
                || key.hasPrefix("model.audio") || key.hasPrefix("mtp_drafter.")
                || !(key.hasPrefix("model.") || key.hasPrefix("lm_head"))
            {
                continue
            }
            if key.hasSuffix("mlp.experts.w13_weight") || key.hasSuffix("mlp.experts.w2_weight") {
                let isW13 = key.hasSuffix("w13_weight")
                let base = Inkling2LoopModel.mapKey(
                    String(key.dropLast((isW13 ? "experts.w13_weight" : "experts.w2_weight").count)))
                let leg = isW13 ? "gate_up_proj" : "down_proj"
                let prefix = base + "switch_mlp.\(leg)."
                guard let scales = weights[key + "_scale"] else { continue }
                out[prefix + "codes"] = MLX.contiguous(value).view(dtype: .uint32)
                out[prefix + "block_scales"] = scales
                if let amax = weights[key + "_amax"] {
                    let den = weights[key + "_outer_scale_denominator"]?.reshaped([-1])[0]
                        ?? MLXArray(Float(2688))
                    out[prefix + "expert_scale"] = amax.asType(.float32) / den.asType(.float32)
                } else {
                    out[prefix + "expert_scale"] = MLXArray.ones([value.dim(0)], dtype: .float32)
                }
                continue
            }
            var n = Inkling2LoopModel.mapKey(key)
            var v = value
            if n.hasSuffix("sconv.weight_proj") { n += ".weight" }
            if n.hasSuffix("_sconv.weight") || n.hasSuffix("_convolution.weight") {
                // Static sconv: checkpoint (C, 1, K) / MLX conv (C, K, 1) -> [K, C].
                if v.ndim == 3 {
                    v = v.reshaped([v.dim(0), -1]).transposed(1, 0)
                }
            }
            out[n] = v
        }
        return out
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

extension Inkling2LoopModel: LoRAModel {
    public var loraLayers: [Module] { model.layers }
}
