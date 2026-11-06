//
//  Ricursive.swift
//  LLM
//
//  Ripple-65B-Preview (ricursiveAI, `model_type: ricursive`). Port of the vMLX Python reference
//  (`vmlx_engine/models/ricursive/ricursive.py`, golden-checked against the release's BF16 logits), which follows
//  `modeling_ricursive.py` @ 240d5966:
//
//  - plain RMSNorm (`weight * x * rsqrt(mean(x^2) + eps)`, variance in float32, NO +1). eps = 1e-6 is not negligible:
//    residual variance falls to about 1e-5 on layers 4-10.
//  - per-head q/k RMSNorm before RoPE; v has no norm.
//  - four norms per block: input/post_attention around attention, pre_feedforward/post_feedforward around the FFN.
//  - value residual: v_proj reads lerp(scaled embedding, normed input, sigmoid(value_res_lambda)).
//  - gated attention: output * sigmoid(g_proj(normed input)), the sigmoid in float32.
//  - FFN: gate clamped above at 20, up clamped to +/-20, then silu(gate) * up.
//  - layer_types: sliding_attention (window 1024 = current token + 1023 before it, RoPE theta 1e4 in float32) and
//    full_attention (every 5th layer, NO RoPE).
//  - embed_scale: the product is taken in float32 and rounded back (a bf16 product rounds the scale first and differs
//    on 14% of elements). Untied lm_head.
//
//  Cache: full-attention layers use `KVCacheSimple`; sliding layers use `RicursiveSlidingKVCache` and attend to the
//  newest `sliding_window` keys through a slice plus an explicit window mask instead of a ring buffer. That keeps every
//  cache trimmable (speculative rollback past the window stays exact, which a 1024-row ring cannot do) while SSD entries
//  and restored caches hold only the window.
//

import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN

public struct RicursiveConfiguration: Codable, Sendable {
    var hiddenSize: Int
    var hiddenLayers: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var kvHeads: Int
    var headDim: Int
    var vocabularySize: Int
    var rmsNormEps: Float
    var ropeTheta: Float
    var slidingWindow: Int
    var layerTypes: [String]
    var useGatedAttention: Bool
    var gateFullPrecision: Bool
    var useValueResidual: Bool
    var ffnClampThreshold: Float?
    var embedScale: Float?
    var tieWordEmbeddings: Bool
    var maxPositionEmbeddings: Int

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case vocabularySize = "vocab_size"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case slidingWindow = "sliding_window"
        case layerTypes = "layer_types"
        case useGatedAttention = "use_gated_attention"
        case gateFullPrecision = "gate_full_precision"
        case useValueResidual = "use_valres"
        case ffnClampThreshold = "ffn_clamp_threshold"
        case embedScale = "embed_scale"
        case tieWordEmbeddings = "tie_word_embeddings"
        case maxPositionEmbeddings = "max_position_embeddings"
    }

    public enum ContractError: Error, CustomStringConvertible {
        case invalid(String)
        public var description: String {
            switch self { case .invalid(let m): return "ricursive: \(m)" }
        }
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        attentionHeads = try c.decode(Int.self, forKey: .attentionHeads)
        kvHeads = try c.decode(Int.self, forKey: .kvHeads)
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? hiddenSize / attentionHeads
        vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10_000
        slidingWindow = try c.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 1024
        let declared = try c.decodeIfPresent([String].self, forKey: .layerTypes) ?? []
        layerTypes =
            declared.isEmpty
            ? (0 ..< hiddenLayers).map { $0 % 5 == 4 ? "full_attention" : "sliding_attention" }
            : declared
        useGatedAttention = try c.decodeIfPresent(Bool.self, forKey: .useGatedAttention) ?? true
        gateFullPrecision = try c.decodeIfPresent(Bool.self, forKey: .gateFullPrecision) ?? true
        useValueResidual = try c.decodeIfPresent(Bool.self, forKey: .useValueResidual) ?? true
        ffnClampThreshold = try c.decodeIfPresent(Float.self, forKey: .ffnClampThreshold)
        embedScale = try c.decodeIfPresent(Float.self, forKey: .embedScale)
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        maxPositionEmbeddings =
            try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 2_097_152
        guard layerTypes.count == hiddenLayers else {
            throw ContractError.invalid("layer_types must name every layer")
        }
        guard Set(layerTypes).isSubset(of: ["sliding_attention", "full_attention"]) else {
            throw ContractError.invalid("unsupported layer type in \(Set(layerTypes).sorted())")
        }
        guard slidingWindow > 0 else { throw ContractError.invalid("sliding_window must be positive") }
    }
}

/// Plain RMSNorm, matching the reference exactly: normalise in float32, round to the input dtype, then multiply by the
/// weight (no +1). `MLXFast.rmsNorm` computes the same expression (`w * T(x * rsqrt(mean(x^2) + eps))`).
final class RicursiveRMSNorm: Module, UnaryLayer {
    let weight: MLXArray
    let eps: Float
    init(dimensions: Int, eps: Float) {
        weight = MLXArray.ones([dimensions])
        self.eps = eps
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: weight, eps: eps)
    }
}

/// Per-forward attention context shared by every layer of one kind: the mask and, for sliding layers, how many of the
/// oldest returned keys fall outside the window.
struct RicursiveAttentionPlan {
    /// Absolute position of the first query row.
    let offset: Int
    let queryLength: Int
    let window: Int
    private var slidingMasks: [Int: MLXArray] = [:]
    private var fullMask: MLXArray?

    init(offset: Int, queryLength: Int, window: Int) {
        self.offset = offset
        self.queryLength = queryLength
        self.window = window
    }

    /// Sliding window mask for `keyLength` keys whose last key sits at the last query's position. Query `i` (absolute
    /// `offset + i`) sees key `j` (absolute `first + j`) when `first + j <= offset + i` and
    /// `first + j > offset + i - window`.
    mutating func slidingMask(keyLength: Int) -> MLXArray {
        if let m = slidingMasks[keyLength] { return m }
        let first = offset + queryLength - keyLength
        let q = MLXArray(Int32(offset) ..< Int32(offset + queryLength))[0..., .newAxis]
        let k = MLXArray(Int32(first) ..< Int32(first + keyLength))[.newAxis]
        let m = (k .<= q) .&& (k .> (q - Int32(window)))
        slidingMasks[keyLength] = m
        return m
    }

    mutating func causalMask(keyLength: Int) -> MLXArray {
        if let fullMask, fullMask.dim(-1) == keyLength { return fullMask }
        let first = offset + queryLength - keyLength
        let q = MLXArray(Int32(offset) ..< Int32(offset + queryLength))[0..., .newAxis]
        let k = MLXArray(Int32(first) ..< Int32(first + keyLength))[.newAxis]
        let m = k .<= q
        fullMask = m
        return m
    }
}

final class RicursiveAttention: Module {
    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let scale: Float
    let window: Int
    let sliding: Bool
    let gateFullPrecision: Bool
    let rope: RoPE?

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear
    @ModuleInfo(key: "g_proj") var wg: Linear?
    @ModuleInfo(key: "q_norm") var qNorm: RicursiveRMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RicursiveRMSNorm
    @ParameterInfo(key: "value_res_lambda") var valueResLambda: MLXArray?

    init(_ c: RicursiveConfiguration, layer: Int) {
        heads = c.attentionHeads
        kvHeads = c.kvHeads
        headDim = c.headDim
        scale = pow(Float(c.headDim), -0.5)
        window = c.slidingWindow
        sliding = c.layerTypes[layer] == "sliding_attention"
        gateFullPrecision = c.gateFullPrecision
        _wq.wrappedValue = Linear(c.hiddenSize, heads * headDim, bias: false)
        _wk.wrappedValue = Linear(c.hiddenSize, kvHeads * headDim, bias: false)
        _wv.wrappedValue = Linear(c.hiddenSize, kvHeads * headDim, bias: false)
        _wo.wrappedValue = Linear(heads * headDim, c.hiddenSize, bias: false)
        if c.useGatedAttention {
            _wg.wrappedValue = Linear(c.hiddenSize, heads * headDim, bias: false)
        }
        _qNorm.wrappedValue = RicursiveRMSNorm(dimensions: headDim, eps: c.rmsNormEps)
        _kNorm.wrappedValue = RicursiveRMSNorm(dimensions: headDim, eps: c.rmsNormEps)
        if c.useValueResidual {
            _valueResLambda.wrappedValue = MLXArray.zeros([1])
        }
        rope = sliding ? RoPE(dimensions: headDim, traditional: false, base: c.ropeTheta) : nil
    }

    private func rotate(_ x: MLXArray, cache: KVCache?) -> MLXArray {
        guard let rope else { return x }
        // The reference applies RoPE in float32.
        return applyRotaryPosition(rope, to: x.asType(.float32), cache: cache).asType(x.dtype)
    }

    func callAsFunction(
        _ x: MLXArray, valueResidual: MLXArray?, cache: KVCache?, plan: inout RicursiveAttentionPlan
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))
        var vIn = x
        if let valueResidual, let valueResLambda {
            let blend = sigmoid(valueResLambda[0].asType(.float32)).asType(x.dtype)
            let r = valueResidual.asType(x.dtype)
            vIn = r + blend * (x - r)
        }
        var q = qNorm(wq(x).reshaped(B, L, heads, headDim)).transposed(0, 2, 1, 3)
        var k = kNorm(wk(x).reshaped(B, L, kvHeads, headDim)).transposed(0, 2, 1, 3)
        let v = wv(vIn).reshaped(B, L, kvHeads, headDim).transposed(0, 2, 1, 3)
        q = rotate(q, cache: cache)
        k = rotate(k, cache: cache)

        let out: MLXArray
        if let cache, !(cache is QuantizedKVCacheProtocol), !(cache is CompilableKVCache),
            cache.maxSize == nil, B == 1 || !(cache is BatchKVCache)
        {
            // Append-only cache: the update returns this sequence's whole history in temporal order.
            var (keys, values) = cache.update(keys: k, values: v)
            var keyLength = keys.dim(2)
            if sliding, keyLength > window - 1 + L {
                let skip = keyLength - (window - 1 + L)
                keys = keys[.ellipsis, skip..., 0...]
                values = values[.ellipsis, skip..., 0...]
                keyLength -= skip
            }
            let mask: MLXFast.ScaledDotProductAttentionMaskMode
            if L == 1 {
                mask = .none
            } else if sliding {
                mask = .array(plan.slidingMask(keyLength: keyLength))
            } else if keyLength == L {
                mask = .causal
            } else {
                mask = .array(plan.causalMask(keyLength: keyLength))
            }
            out = MLXFast.scaledDotProductAttention(
                queries: q, keys: keys, values: values, scale: scale, mask: mask)
        } else if let cache {
            // Quantized, bounded or multi-sequence batched cache: full keys with an explicit (window) mask.
            let mask: MLXFast.ScaledDotProductAttentionMaskMode
            if cache is BatchKVCache || cache is CompilableKVCache {
                // Fixed-size / padded buffers: the cache's own (offset-aware) mask covers the unused rows.
                mask = cache.makeMask(n: L, windowSize: sliding ? window : nil, returnArray: true)
            } else if sliding {
                mask = .array(createCausalMask(n: L, offset: cache.offset, windowSize: window))
            } else {
                mask = L == 1 ? .none : .array(createCausalMask(n: L, offset: cache.offset))
            }
            out = attentionWithCacheUpdate(
                queries: q, keys: k, values: v, cache: cache, scale: scale, mask: mask)
        } else {
            let mask: MLXFast.ScaledDotProductAttentionMaskMode =
                L == 1
                ? .none
                : (sliding
                    ? .array(createCausalMask(n: L, offset: 0, windowSize: window)) : .causal)
            out = MLXFast.scaledDotProductAttention(
                queries: q, keys: k, values: v, scale: scale, mask: mask)
        }
        var attended = out.transposed(0, 2, 1, 3).reshaped(B, L, -1)
        if let wg {
            var g = wg(x)
            if gateFullPrecision { g = g.asType(.float32) }
            attended = attended * sigmoid(g).asType(attended.dtype)
        }
        return wo(attended)
    }
}

final class RicursiveMLP: Module, UnaryLayer {
    let clamp: Float?
    @ModuleInfo(key: "gate_proj") var gate: Module
    @ModuleInfo(key: "up_proj") var up: Module
    @ModuleInfo(key: "down_proj") var down: Module

    init(_ c: RicursiveConfiguration) {
        clamp = c.ffnClampThreshold.flatMap { $0 > 0 ? $0 : nil }
        _gate.wrappedValue = Linear(c.hiddenSize, c.intermediateSize, bias: false)
        _up.wrappedValue = Linear(c.hiddenSize, c.intermediateSize, bias: false)
        _down.wrappedValue = Linear(c.intermediateSize, c.hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var g = (gate as! UnaryLayer)(x)
        var u = (up as! UnaryLayer)(x)
        if let clamp {
            g = minimum(g, MLXArray(clamp, dtype: g.dtype))
            u = clip(u, min: MLXArray(-clamp, dtype: u.dtype), max: MLXArray(clamp, dtype: u.dtype))
        }
        return (down as! UnaryLayer)(silu(g) * u)
    }
}

final class RicursiveDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: RicursiveAttention
    @ModuleInfo(key: "mlp") var mlp: RicursiveMLP
    @ModuleInfo(key: "input_layernorm") var inputNorm: RicursiveRMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionNorm: RicursiveRMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardNorm: RicursiveRMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardNorm: RicursiveRMSNorm

    init(_ c: RicursiveConfiguration, layer: Int) {
        _attention.wrappedValue = RicursiveAttention(c, layer: layer)
        _mlp.wrappedValue = RicursiveMLP(c)
        _inputNorm.wrappedValue = RicursiveRMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
        _postAttentionNorm.wrappedValue = RicursiveRMSNorm(
            dimensions: c.hiddenSize, eps: c.rmsNormEps)
        _preFeedforwardNorm.wrappedValue = RicursiveRMSNorm(
            dimensions: c.hiddenSize, eps: c.rmsNormEps)
        _postFeedforwardNorm.wrappedValue = RicursiveRMSNorm(
            dimensions: c.hiddenSize, eps: c.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray, valueResidual: MLXArray?, cache: KVCache?, plan: inout RicursiveAttentionPlan
    ) -> MLXArray {
        let a = attention(inputNorm(x), valueResidual: valueResidual, cache: cache, plan: &plan)
        let h = x + postAttentionNorm(a)
        return h + postFeedforwardNorm(mlp(preFeedforwardNorm(h)))
    }
}

public final class RicursiveModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    let layers: [RicursiveDecoderLayer]
    let norm: RicursiveRMSNorm
    let configuration: RicursiveConfiguration

    init(_ c: RicursiveConfiguration) {
        configuration = c
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: c.vocabularySize, dimensions: c.hiddenSize)
        layers = (0 ..< c.hiddenLayers).map { RicursiveDecoderLayer(c, layer: $0) }
        norm = RicursiveRMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
    }

    /// Scaled embedding: float32 product, rounded back to the embedding dtype.
    func embed(_ inputs: MLXArray) -> MLXArray {
        let h = embedTokens(inputs)
        guard let scale = configuration.embedScale else { return h }
        return (h.asType(.float32) * scale).asType(h.dtype)
    }

    /// `VMLX_RICURSIVE_TRACE=1` (diagnostic): log the first forward's cache types per distinct (rows, types) shape.
    static let traceEnabled = ProcessInfo.processInfo.environment["VMLX_RICURSIVE_TRACE"] == "1"
    nonisolated(unsafe) static var traced: Set<String> = []

    func callAsFunctionCapturing(
        _ inputs: MLXArray, cache: [KVCache]?, captureLayerIDs: Set<Int>
    ) -> (MLXArray, [Int: MLXArray]) {
        if Self.traceEnabled {
            let key = "rows=\(inputs.dim(-1)) caches=\(Set((cache ?? []).map { String(describing: type(of: $0)) }).sorted())"
            if Self.traced.insert(key).inserted {
                FileHandle.standardError.write(Data("[ricursive] forward \(key)\n".utf8))
            }
        }
        var h = embed(inputs)
        let offset = cache?.first?.offset ?? 0
        var slidingPlan = RicursiveAttentionPlan(
            offset: offset, queryLength: h.dim(1), window: configuration.slidingWindow)
        var fullPlan = slidingPlan
        let residual = configuration.useValueResidual ? h : nil
        var captured: [Int: MLXArray] = [:]
        captured.reserveCapacity(captureLayerIDs.count)
        for (i, layer) in layers.enumerated() {
            if layer.attention.sliding {
                h = layer(h, valueResidual: residual, cache: cache?[i], plan: &slidingPlan)
            } else {
                h = layer(h, valueResidual: residual, cache: cache?[i], plan: &fullPlan)
            }
            // DFlash taps decoder OUTPUTS, before the final norm.
            if captureLayerIDs.contains(i) { captured[i] = h }
        }
        return (norm(h), captured)
    }
}

public final class RicursiveModel: Module, LLMModel, KVCacheDimensionProvider,
    HiddenStateCaptureModel, TokenEmbedderModel
{
    public let vocabularySize: Int
    public let kvHeads: [Int]
    let configuration: RicursiveConfiguration
    /// Packed JANGH tensors owned by custom dense modules (JANGH2 bundles); excluded from the generic load.
    var customDenseTensorNames: Set<String> = []

    public let model: RicursiveModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ c: RicursiveConfiguration) {
        configuration = c
        vocabularySize = c.vocabularySize
        kvHeads = Array(repeating: c.kvHeads, count: c.hiddenLayers)
        model = RicursiveModelInner(c)
        if !c.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(c.hiddenSize, c.vocabularySize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        projectToLogits(model.callAsFunctionCapturing(inputs, cache: cache, captureLayerIDs: []).0)
    }

    public func callAsFunction(
        _ inputs: MLXArray, cache: [KVCache]?, captureLayerIDs: Set<Int>
    ) -> (logits: MLXArray, capturedHiddenStates: [Int: MLXArray]) {
        let (h, captured) = model.callAsFunctionCapturing(
            inputs, cache: cache, captureLayerIDs: captureLayerIDs)
        return (projectToLogits(h), captured)
    }

    /// Eager decode only. Inside a whole-forward compile trace the dense JANGH fast QMV is disabled
    /// (`CompiledDecodeTrace.isActive`), and the static compile buffers make every sliding layer read its whole buffer:
    /// measured on JANGH2 at 1,500 tokens, eager 1-row forward 61 ms vs 207 ms without the fast QMV (AR probe through the
    /// compiled path: ~130 ms/token).
    public var supportsWholeForwardCompilation: Bool { false }

    /// Full-attention layers read their whole history (`KVCacheSimple`); sliding layers read, persist and restore only
    /// their window (`RicursiveSlidingKVCache`).
    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        configuration.layerTypes.map { type in
            type == "sliding_attention"
                ? RicursiveSlidingKVCache(window: configuration.slidingWindow) as KVCache : KVCacheSimple()
        }
    }

    /// One sequence per decode: a restored sliding layer holds fewer rows than its position, which the batched
    /// padding and masks do not model.
    public var maximumSupportedDecodeBatchSize: Int? { 1 }

    /// The drafter shares this embedding, scaled exactly as the target's first layer reads it.
    public func embed(_ tokenIds: MLXArray) -> MLXArray {
        model.embed(tokenIds)
    }

    public func projectToLogits(_ hidden: MLXArray) -> MLXArray {
        if let lmHead { return lmHead(hidden) }
        return model.embedTokens.asLinear(hidden)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var w = weights.filter { !$0.key.contains("rotary_emb.inv_freq") }
        if configuration.tieWordEmbeddings { w["lm_head.weight"] = nil }
        return w
    }
}

extension RicursiveModel: SafetensorsLoadKeyExcluding {
    public func excludeFromGenericSafetensorsLoad(key: String) -> Bool {
        customDenseTensorNames.contains(key)
    }
    public var requiresExactTensorMmapBuffers: Bool { !customDenseTensorNames.isEmpty }
}

/// Lane matmul placement, measured 2026-11-06 on JANG_4D (ms per target forward at 1/2/4/8/16 rows, 1,500 tokens):
/// tiled lane 116/123/138/144/147, MLX only 92/97/115/195/156, untiled lane from 8 rows 92/97/115/165/170. Tiled modules
/// run the lane for every row count, so a tiled install would slow ordinary AR decode by ~25 % for the rest of the
/// process after the first DFlash2 request. Untiled keeps AR on MLX's qmv and moves 8+ row verify blocks to the lane.
extension RicursiveModel: LaneQMMTargetPolicy {
    public var laneQMMTiled: Bool { false }
    public var laneQMMUntiledMinRows: Int { 8 }
}

/// Candidate verify widths (AR is added by the chooser, which picks among them per cycle). Measured 2026-11-06, ms per
/// target forward at 1,500 tokens, 1/5/8/16/32 rows: JANGH2 64/174/156/164/259, JANG_4D 90/186/160/165/232. A narrow
/// width costs what 16 rows do and verifies less, so it can never win; 32 stays for highly predictable text.
extension RicursiveModel: DFlash2WidthLadderPolicy {
    public var dflash2WidthLadder: [Int] { [16, 32] }
    public var dflash2BoundsCostsByWiderRungs: Bool { true }
    public var dflash2ProbesWhileEvidenceIsThin: Bool { true }
}

/// The vendor template reproduces an unchanged assistant message to the exact generated tokens, tool calls included
/// (with the separator-stripping reasoning parser).
extension RicursiveModel: ToolCallTurnBoundaryModel {}

extension RicursiveModel: LoRAModel {
    public var loraLayers: [Module] { model.layers }
}
