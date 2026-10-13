//
//  K2Horizon.swift
//  IFM K2-Horizon dense (model_type "k2_horizon").
//
//  Port of jang_tools.k2_horizon.model. The Python reference was compared with vendor code;
//  Swift bundle parity requires separate runtime qualification.
//  Llama-shaped, with GROUPED RMSNorm: input_layernorm / post_attention_layernorm / final norm normalise each of
//  `layernorm_num_groups` contiguous slices (4 on the 7B) separately, then apply one full-width weight. Plain
//  RMSNorm over the whole hidden vector is wrong but still emits fluent text for a while — test with parity, not vibes.
//  GQA 32q/8kv x 128, RoPE theta 1e7 (traditional = false), no q/k norm, SwiGLU, untied lm_head (250,624 vocab).
//  mlp_layout "switch1" (JANGH bundles): the MLP is a ONE-expert routed stack `mlp.switch_mlp.*` so the routed
//  JANGH banks (K2HorizonJANGHPreparation) apply unchanged; every token is routed to expert 0 with weight 1.
//  "dense_jangh_down" keeps affine gate/up and explicitly affine down exceptions while mapping only declared JANGH downs.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

public struct K2HorizonConfiguration: Codable, Sendable {
    var modelType: String = "k2_horizon"
    var hiddenSize: Int
    var hiddenLayers: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var kvHeads: Int
    var headDim: Int
    var rmsNormEps: Float
    var vocabularySize: Int
    var layerNormGroups: Int = 1
    var ropeParameters: [String: StringOrNumber]? = nil
    var tieWordEmbeddings: Bool = false
    var mlpLayout: String = "dense"
    var numExperts: Int = 0
    var movaNumExperts: Int = 0
    var queryKeyNorm: Bool = false
    var attentionGate: String? = nil
    var attentionBias: Bool = false
    var ropeHeadDim: Int? = nil
    var flatRopeTheta: Float = 10_000_000
    var useSlidingWindow: Bool = false
    var slidingWindow: Int? = nil
    var hiddenActivation: String = "silu"

    enum ContractError: Error { case unsupported(String) }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case layerNormGroups = "layernorm_num_groups"
        case ropeParameters = "rope_parameters"
        case tieWordEmbeddings = "tie_word_embeddings"
        case mlpLayout = "mlp_layout"
        case numExperts = "num_experts"
        case movaNumExperts = "mova_num_experts"
        case queryKeyNorm = "query_key_norm"
        case attentionGate = "attention_gate_func"
        case attentionBias = "attention_bias"
        case ropeHeadDim = "rope_head_dim"
        case flatRopeTheta = "rope_theta"
        case useSlidingWindow = "use_sliding_window"
        case slidingWindow = "sliding_window"
        case hiddenActivation = "hidden_act"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "k2_horizon"
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        attentionHeads = try c.decode(Int.self, forKey: .attentionHeads)
        kvHeads = try c.decode(Int.self, forKey: .kvHeads)
        headDim = try c.decode(Int.self, forKey: .headDim)
        rmsNormEps = try c.decode(Float.self, forKey: .rmsNormEps)
        vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        layerNormGroups = try c.decodeIfPresent(Int.self, forKey: .layerNormGroups) ?? 1
        ropeParameters = try c.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeParameters)
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        mlpLayout = try c.decodeIfPresent(String.self, forKey: .mlpLayout) ?? "dense"
        numExperts = try c.decodeIfPresent(Int.self, forKey: .numExperts) ?? 0
        movaNumExperts = try c.decodeIfPresent(Int.self, forKey: .movaNumExperts) ?? 0
        queryKeyNorm = try c.decodeIfPresent(Bool.self, forKey: .queryKeyNorm) ?? false
        attentionGate = try c.decodeIfPresent(String.self, forKey: .attentionGate)
        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        ropeHeadDim = try c.decodeIfPresent(Int.self, forKey: .ropeHeadDim)
        flatRopeTheta = try c.decodeIfPresent(Float.self, forKey: .flatRopeTheta) ?? 10_000_000
        useSlidingWindow = try c.decodeIfPresent(Bool.self, forKey: .useSlidingWindow) ?? false
        slidingWindow = try c.decodeIfPresent(Int.self, forKey: .slidingWindow)
        hiddenActivation =
            c.contains(.hiddenActivation)
            ? try c.decode(String.self, forKey: .hiddenActivation) : "silu"
        // Vendor attention uses sliding_window independently of use_sliding_window.
        // This implementation admits only full attention and the native SiLU MLP.
        guard slidingWindow == nil, hiddenActivation == "silu" else {
            throw ContractError.unsupported("K2 requires full attention and silu activation")
        }
        guard modelType == "k2_horizon", numExperts == 0, movaNumExperts == 0,
            !queryKeyNorm, attentionGate == nil, !attentionBias, !useSlidingWindow,
            ropeHeadDim == nil || ropeHeadDim == headDim,
            ["dense", "switch1", "dense_jangh_down"].contains(mlpLayout)
        else {
            throw ContractError.unsupported("unsupported K2 attention, expert, or MLP layout")
        }
        guard hiddenSize > 0, hiddenLayers > 0, intermediateSize > 0, vocabularySize > 0,
            attentionHeads > 0, kvHeads > 0, attentionHeads.isMultiple(of: kvHeads),
            headDim > 0, headDim.isMultiple(of: 2), layerNormGroups > 0,
            hiddenSize.isMultiple(of: layerNormGroups), rmsNormEps.isFinite, rmsNormEps > 0,
            ropeTheta.isFinite, ropeTheta > 0
        else {
            throw ContractError.unsupported(
                "invalid K2 dimensions or normalization/rotary constants")
        }
        if let theta = ropeParameters?["rope_theta"] {
            switch theta {
            case .int, .float: break
            default: throw ContractError.unsupported("K2 rope_theta must be numeric")
            }
        }
        if let ropeType = ropeParameters?["rope_type"], ropeType != .string("default") {
            throw ContractError.unsupported("unsupported K2 rope_type")
        }
    }

    var ropeTheta: Float { ropeParameters?["rope_theta"]?.asFloat() ?? flatRopeTheta }
}

/// RMSNorm per contiguous group followed by the checkpoint full-width weight.
final class K2GroupedRMSNorm: Module, UnaryLayer {
    let weight: MLXArray
    let groups: Int, eps: Float
    /// The per-group unit weight `MLXFast.rmsNorm` requires. Built once per
    /// dtype and reused: creating it inside `callAsFunction` added a fill
    /// kernel to every norm (73 per decoded token on the 7B). Same values,
    /// same call, so the output is bit-identical. Held outside the Module
    /// property graph so it is never mistaken for a checkpoint parameter.
    private let unitWeights = UnitWeightCache()
    init(dimensions: Int, groups: Int, eps: Float) {
        weight = MLXArray.ones([dimensions])
        self.groups = groups
        self.eps = eps
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        if groups == 1 { return MLXFast.rmsNorm(x, weight: weight, eps: eps) }
        var shape = x.shape
        let d = shape.removeLast()
        let y = MLXFast.rmsNorm(
            x.reshaped(shape + [groups, d / groups]),
            weight: unitWeights.ones(count: d / groups, dtype: x.dtype), eps: eps)
        return y.reshaped(x.shape) * weight
    }
}

final class UnitWeightCache: @unchecked Sendable {
    private let lock = NSLock()
    private var arrays: [String: MLXArray] = [:]
    func ones(count: Int, dtype: DType) -> MLXArray {
        // `eval` is illegal inside a compile trace; a traced graph builds its
        // constant once anyway, so trace without touching the cache.
        if CompiledDecodeTrace.isActive { return MLXArray.ones([count], dtype: dtype) }
        let key = "\(count)|\(dtype)"
        lock.lock()
        defer { lock.unlock() }
        if let cached = arrays[key] { return cached }
        let created = MLXArray.ones([count], dtype: dtype)
        eval(created)
        arrays[key] = created
        return created
    }
}

/// Conditional LoRA for one attention or MLP block (IFM Uno), attached by
/// ``K2HorizonModel/installUnoAdapter(_:)``. A plain holder, not a Module: the adapter is never part
/// of the bundle's parameters. Projections that share an input share one Aᵀ matmul (q/k/v; gate/up),
/// and the row mask multiplies the rank-r hidden, not the full-width output.
final class K2UnoLoRA: @unchecked Sendable {
    let mask: UnoRowMask
    let rank: Int
    /// Aᵀ of the projections reading the block input, concatenated: (in, n·r).
    let aInput: MLXArray
    /// scale·Bᵀ per input projection, in `aInput` order.
    let bInput: [MLXArray]
    /// Aᵀ and scale·Bᵀ of the output projection (o_proj or down_proj).
    let aOutput: MLXArray
    let bOutput: MLXArray

    init(
        inputs: [UnoAdapter.Projection], output: UnoAdapter.Projection, rank: Int, mask: UnoRowMask
    ) {
        self.mask = mask
        self.rank = rank
        self.aInput = contiguous(concatenated(inputs.map(\.aT), axis: 1))
        self.bInput = inputs.map(\.bTScaled)
        self.aOutput = output.aT
        self.bOutput = output.bTScaled
        eval(aInput)
    }

    /// Deltas for the input projections while the mask is set (nil otherwise).
    func inputDeltas(_ x: MLXArray) -> [MLXArray]? {
        guard let rows = mask.rows else { return nil }
        let h = matmul(x, aInput) * rows
        return bInput.enumerated().map { i, b in
            matmul(h[.ellipsis, (i * rank) ..< ((i + 1) * rank)], b)
        }
    }

    func outputDelta(_ x: MLXArray) -> MLXArray? {
        guard let rows = mask.rows else { return nil }
        return matmul(matmul(x, aOutput) * rows, bOutput)
    }
}

final class K2Attention: Module {
    let heads: Int, kvHeads: Int, scale: Float
    var uno: K2UnoLoRA?
    let rope: RoPE
    /// RoPE parameters for the per-row-offset tree path (`RoPE`'s own are internal to MLXNN).
    let ropeDimensions: Int, ropeBase: Float
    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear
    init(_ c: K2HorizonConfiguration) {
        heads = c.attentionHeads
        kvHeads = c.kvHeads
        scale = pow(Float(c.headDim), -0.5)
        _wq.wrappedValue = Linear(c.hiddenSize, heads * c.headDim, bias: false)
        _wk.wrappedValue = Linear(c.hiddenSize, kvHeads * c.headDim, bias: false)
        _wv.wrappedValue = Linear(c.hiddenSize, kvHeads * c.headDim, bias: false)
        _wo.wrappedValue = Linear(heads * c.headDim, c.hiddenSize, bias: false)
        rope = RoPE(dimensions: c.headDim, traditional: false, base: c.ropeTheta)
        ropeDimensions = c.headDim
        ropeBase = c.ropeTheta
    }
    /// Key length from which 9-32-row blocks attend in vector-kernel chunks. Measured crossover
    /// (12-row forward, M5 Max, medians of 3 rounds): one fused call / chunks = 24.2 / 24.7 ms at
    /// 1,024 keys, 26.7 / 25.9 at 1,536, 29.2 / 27.2 at 2,048, 59.4 / 44.1 at 8,192.
    /// `VMLX_K2_CHUNKED_ATTN_MIN_KEYS` overrides it for A/B runs.
    static let chunkedBlockMinKeys =
        Int(ProcessInfo.processInfo.environment["VMLX_K2_CHUNKED_ATTN_MIN_KEYS"] ?? "") ?? 1536

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        if let scope = DFlash2TreeScope.current, let cache, x.dim(0) == 1 {
            return treeAttention(x, cache: cache, scope: scope)
        }
        let (B, L) = (x.dim(0), x.dim(1))
        var qp = wq(x), kp = wk(x), vp = wv(x)
        if let d = uno?.inputDeltas(x) {
            qp = qp + d[0]
            kp = kp + d[1]
            vp = vp + d[2]
        }
        var q = qp.reshaped(B, L, heads, -1).transposed(0, 2, 1, 3)
        var k = kp.reshaped(B, L, kvHeads, -1).transposed(0, 2, 1, 3)
        let v = vp.reshaped(B, L, kvHeads, -1).transposed(0, 2, 1, 3)
        q = applyRotaryPosition(rope, to: q, cache: cache)
        k = applyRotaryPosition(rope, to: k, cache: cache)
        let o: MLXArray
        if L >= 2, L <= 16, MultiRowDecodeAttention.active, let cache, type(of: cache) == KVCacheSimple.self,
            MultiRowDecodeAttention.maskAllowed(mask, queryLength: L, keyLength: cache.offset + L)
        {
            // Speculative draft/verify block: KV read once for all rows (see MultiRowDecodeAttention).
            let (keys, values) = cache.update(keys: k, values: v)
            if MultiRowDecodeAttention.eligible(queries: q, keys: keys, values: values) {
                o = MultiRowDecodeAttention.attend(queries: q, keys: keys, values: values, scale: scale)
            } else {
                o = MLXFast.scaledDotProductAttention(
                    queries: q, keys: keys, values: values, scale: scale, mask: mask)
            }
        } else if L > 8, L <= 32, let cache, !(cache is QuantizedKVCacheProtocol),
            cache.offset + L >= K2Attention.chunkedBlockMinKeys
        {
            // Draft/verify blocks over a long context: see `chunkedVectorAttention`.
            let (keys, values) = cache.update(keys: k, values: v)
            o = chunkedVectorAttention(queries: q, keys: keys, values: values, scale: scale, mask: mask)
                ?? MLXFast.scaledDotProductAttention(
                    queries: q, keys: keys, values: values, scale: scale, mask: mask)
        } else {
            o = attentionWithCacheUpdate(
                queries: q, keys: k, values: v, cache: cache, scale: scale, mask: mask)
        }
        let attended = o.transposed(0, 2, 1, 3).reshaped(B, L, -1)
        let out = wo(attended)
        if let d = uno?.outputDelta(attended) { return out + d }
        return out
    }
}

final class K2DenseMLP: Module, UnaryLayer {
    var uno: K2UnoLoRA?
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Module
    init(_ d: Int, _ h: Int, down: JANGHDenseLinear? = nil) {
        _gate.wrappedValue = Linear(d, h, bias: false)
        _up.wrappedValue = Linear(d, h, bias: false)
        _down.wrappedValue = down ?? Linear(h, d, bias: false)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        guard let d = uno?.inputDeltas(x) else { return (down as! UnaryLayer)(silu(gate(x)) * up(x)) }
        let h = silu(gate(x) + d[0]) * (up(x) + d[1])
        let out = (down as! UnaryLayer)(h)
        if let dd = uno?.outputDelta(h) { return out + dd }
        return out
    }
}

/// Dense MLP stored as a one-expert routed stack (JANGH bundles).
final class K2SwitchDenseMLP: Module, UnaryLayer {
    @ModuleInfo(key: "switch_mlp") var experts: Module
    init(_ c: K2HorizonConfiguration, routed: (Module & WeightedRoutedExpertLayer)?) {
        _experts.wrappedValue =
            routed
            ?? SwitchGLU(inputDims: c.hiddenSize, hiddenDims: c.intermediateSize, numExperts: 1)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let lead = Array(x.shape.dropLast())
        let idx = MLXArray.zeros(lead + [1], type: UInt32.self)
        let w = MLXArray.ones(lead + [1], dtype: x.dtype)
        return (experts as! any WeightedRoutedExpertLayer).callRouted(x, indices: idx, scores: w)
    }
}

final class K2DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: K2Attention
    @ModuleInfo(key: "input_layernorm") var inputNorm: K2GroupedRMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postNorm: K2GroupedRMSNorm
    let mlp: UnaryLayer
    init(
        _ c: K2HorizonConfiguration, routed: (Module & WeightedRoutedExpertLayer)?,
        down: JANGHDenseLinear?
    ) {
        _attention.wrappedValue = K2Attention(c)
        _inputNorm.wrappedValue = K2GroupedRMSNorm(
            dimensions: c.hiddenSize, groups: c.layerNormGroups, eps: c.rmsNormEps)
        _postNorm.wrappedValue = K2GroupedRMSNorm(
            dimensions: c.hiddenSize, groups: c.layerNormGroups, eps: c.rmsNormEps)
        mlp =
            c.mlpLayout == "switch1"
            ? K2SwitchDenseMLP(c, routed: routed)
            : K2DenseMLP(c.hiddenSize, c.intermediateSize, down: down)
    }
    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let h = x + attention(inputNorm(x), mask: mask, cache: cache)
        return h + mlp(postNorm(h))
    }
}

public final class K2HorizonModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    let layers: [K2DecoderLayer]
    let norm: K2GroupedRMSNorm
    init(
        _ c: K2HorizonConfiguration, routedFactory: K2HorizonModel.RoutedFactory?,
        denseDown: [Int: JANGHDenseLinear]
    ) throws {
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: c.vocabularySize, dimensions: c.hiddenSize)
        layers = try (0 ..< c.hiddenLayers).map {
            try K2DecoderLayer(c, routed: routedFactory?($0), down: denseDown[$0])
        }
        norm = K2GroupedRMSNorm(
            dimensions: c.hiddenSize, groups: c.layerNormGroups, eps: c.rmsNormEps)
    }
    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        callAsFunctionCapturing(inputs, cache: cache, captureLayerIDs: []).0
    }

    /// VMLX_K2_DFLASH_TAP_INPUT=1 (diagnostic): capture each tap layer's INPUT instead of its output.
    /// Read once, not per forward.
    static let tapInput = ProcessInfo.processInfo.environment["VMLX_K2_DFLASH_TAP_INPUT"] == "1"

    func callAsFunctionCapturing(
        _ inputs: MLXArray, cache: [KVCache]?, captureLayerIDs: Set<Int>
    ) -> (MLXArray, [Int: MLXArray]) {
        var h = embedTokens(inputs)
        let mask = createAttentionMask(h: h, cache: cache?.first)
        var captured: [Int: MLXArray] = [:]
        captured.reserveCapacity(captureLayerIDs.count)
        let tapInput = Self.tapInput
        for (i, layer) in layers.enumerated() {
            if tapInput, captureLayerIDs.contains(i) { captured[i] = h }
            h = layer(h, mask: mask, cache: cache?[i])
            // DFlash indexes decoder OUTPUTS before the final grouped norm.
            if !tapInput, captureLayerIDs.contains(i) { captured[i] = h }
        }
        return (norm(h), captured)
    }
}

public final class K2HorizonModel: Module, LLMModel, KVCacheDimensionProvider,
    HiddenStateCaptureModel, TokenEmbedderModel
{
    public typealias RoutedFactory = (Int) throws -> (Module & WeightedRoutedExpertLayer)
    public let vocabularySize: Int
    public let kvHeads: [Int]
    public var requiresSequentialDFlash2Verification: Bool { true }
    public var dflash2PrefillSeparatesFinalToken: Bool { false }
    let configuration: K2HorizonConfiguration
    let excludedSafetensorsKeys: Set<String>
    /// Directory of the Uno adapter currently attached (``installUnoAdapter(_:)``).
    var installedUnoAdapterPath: String?
    public let model: K2HorizonModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public convenience init(_ c: K2HorizonConfiguration) throws {
        try self.init(c, routedFactory: nil)
    }

    init(
        _ c: K2HorizonConfiguration, routedFactory: RoutedFactory?,
        denseDown: [Int: JANGHDenseLinear] = [:], excludedSafetensorsKeys: Set<String> = []
    ) throws {
        guard c.mlpLayout != "dense_jangh_down" || !denseDown.isEmpty else {
            throw K2HorizonConfiguration.ContractError.unsupported(
                "dense JANGH requires admitted banks")
        }
        configuration = c
        vocabularySize = c.vocabularySize
        kvHeads = Array(repeating: c.kvHeads, count: c.hiddenLayers)
        self.excludedSafetensorsKeys = excludedSafetensorsKeys
        model = try K2HorizonModelInner(c, routedFactory: routedFactory, denseDown: denseDown)
        if !c.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(c.hiddenSize, c.vocabularySize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let h = model(inputs, cache: cache)
        return projectToLogits(h)
    }

    public func callAsFunction(
        _ inputs: MLXArray, cache: [KVCache]?, captureLayerIDs: Set<Int>
    ) -> (logits: MLXArray, capturedHiddenStates: [Int: MLXArray]) {
        let (h, captured) = model.callAsFunctionCapturing(
            inputs, cache: cache, captureLayerIDs: captureLayerIDs)
        return (projectToLogits(h), captured)
    }

    public func embed(_ tokenIds: MLXArray) -> MLXArray {
        model.embedTokens(tokenIds)
    }

    public func projectToLogits(_ hidden: MLXArray) -> MLXArray {
        if let lmHead { return lmHead(hidden) }
        return model.embedTokens.asLinear(hidden)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var w = weights.filter { !$0.key.contains("rotary_emb.inv_freq") }
        if configuration.tieWordEmbeddings { w["lm_head.weight"] = nil }
        if configuration.mlpLayout == "switch1" {  // bf16 source loaded into the switch1 layout
            for k in Array(w.keys)
            where k.contains(".mlp.") && !k.contains(".switch_mlp.") && k.hasSuffix("_proj.weight")
            {
                w[k.replacingOccurrences(of: ".mlp.", with: ".mlp.switch_mlp.")] =
                    expandedDimensions(w.removeValue(forKey: k)!, axis: 0)
            }
        }
        return w
    }
}

extension K2HorizonModel: SafetensorsLoadKeyExcluding {
    public func excludeFromGenericSafetensorsLoad(key: String) -> Bool {
        excludedSafetensorsKeys.contains(key)
    }
    public var requiresExactTensorMmapBuffers: Bool { !excludedSafetensorsKeys.isEmpty }
}

extension K2HorizonModel: LoRAModel {
    public var loraLayers: [Module] { model.layers }
}

/// Lane matmul placement for K2 verify forwards, measured on Diplodocus preview 3 (release build,
/// 1,024-token context, full clock): a tiled lane 1-row forward costs 16.9 ms against MLX's qmv
/// 12.8 ms, which would slow every ordinary decode step after the first DFlash2 request; untiled
/// lane costs 20.8 ms at 4 rows (MLX 16.8) and 22.75 ms at 8 rows (MLX 26.2). So: untiled, lane
/// from 8 rows.
extension K2HorizonModel: LaneQMMTargetPolicy {
    public var laneQMMTiled: Bool { false }
    /// In-place scales: Diplodocus preview 3 Uno decode measured the same with and without the packed
    /// copy (47-58 vs 50-59 tok/s, 3 rounds, identical text) and the copy is 250 MB (2026-10-11).
    public var laneQMMPackedScales: Bool { false }
    public var laneQMMUntiledMinRows: Int { 8 }
}

extension K2HorizonModel: UnoSelfSpeculativeModel {
    /// K2 Uno trains against mask token id 250,624 (= the vocabulary size): noise ids are [1, vocab).
    public var unoNoiseTokenBound: Int { vocabularySize }

    /// Attach an IFM Uno adapter (conditional LoRA on q/k/v/o and gate/up/down of every layer). The
    /// adapter must cover all seven projections of every layer; a partial adapter is refused because
    /// the uncovered modules would draft from base weights without anyone noticing.
    public func installUnoAdapter(_ adapter: UnoAdapter) throws -> UnoRowMask {
        let layers = model.layers
        if let first = layers.first?.attention.uno, installedUnoAdapterPath == adapter.directory.path {
            return first.mask
        }
        let mask = UnoRowMask()
        var attention: [K2UnoLoRA] = []
        var mlps: [K2UnoLoRA] = []
        for (i, layer) in layers.enumerated() {
            guard layer.mlp is K2DenseMLP else {
                throw UnoAdapter.LoadError.invalidConfig("layer \(i) MLP is not dense (mlp_layout switch1)")
            }
            func take(_ group: String, _ names: [String]) throws -> [String: UnoAdapter.Projection] {
                var out: [String: UnoAdapter.Projection] = [:]
                for name in names {
                    let key = "model.layers.\(i).\(group).\(name)"
                    guard let p = adapter.projections[key] else {
                        throw UnoAdapter.LoadError.malformedTensor("missing \(key)")
                    }
                    out[name] = p
                }
                return out
            }
            let a = try take("self_attn", ["q_proj", "k_proj", "v_proj", "o_proj"])
            let m = try take("mlp", ["gate_proj", "up_proj", "down_proj"])
            attention.append(K2UnoLoRA(
                inputs: [a["q_proj"]!, a["k_proj"]!, a["v_proj"]!], output: a["o_proj"]!,
                rank: adapter.rank, mask: mask))
            mlps.append(K2UnoLoRA(
                inputs: [m["gate_proj"]!, m["up_proj"]!], output: m["down_proj"]!,
                rank: adapter.rank, mask: mask))
        }
        guard adapter.projections.count == 7 * layers.count else {
            throw UnoAdapter.LoadError.malformedTensor(
                "adapter has \(adapter.projections.count) projections, target expects \(7 * layers.count)")
        }
        for (i, layer) in layers.enumerated() {
            layer.attention.uno = attention[i]
            (layer.mlp as! K2DenseMLP).uno = mlps[i]
        }
        installedUnoAdapterPath = adapter.directory.path
        return mask
    }
}


// MARK: - DFlash 2 draft-tree verify (K2 Horizon: dense, every layer full attention + KVCacheSimple)

extension K2Attention {
    /// Rows sit at `offset + depth` (siblings share a position); each row attends to the committed prefix plus its own
    /// root path. RoPE runs once per tensor with the rows on the batch axis and a per-row offset array (same fused
    /// kernel as decode). Attention goes through `MultiRowDecodeAttention` with the tree's window bits when eligible.
    func treeAttention(_ x: MLXArray, cache: KVCache, scope: DFlash2TreeScope) -> MLXArray {
        let L = x.dim(1)
        let prefix = cache.offset
        var q = wq(x).reshaped(1, L, heads, -1).transposed(0, 2, 1, 3)
        var k = wk(x).reshaped(1, L, kvHeads, -1).transposed(0, 2, 1, 3)
        let v = wv(x).reshaped(1, L, kvHeads, -1).transposed(0, 2, 1, 3)
        let positions = MLXArray(scope.plan.depths.map { Int32(prefix + scope.positionDelta + $0) })
        func rotate(_ t: MLXArray) -> MLXArray {
            // [1, H, L, D] -> [L, H, 1, D]: one row per batch entry, each at its own offset.
            let rows = t.transposed(2, 1, 0, 3)
            let rotated = MLXFast.RoPE(
                rows, dimensions: ropeDimensions, traditional: false, base: ropeBase, scale: 1,
                offset: positions)
            return rotated.transposed(2, 1, 0, 3)
        }
        q = rotate(q)
        k = rotate(k)
        let (keys, values) = cache.update(keys: k, values: v)
        let o: MLXArray
        if L <= 16, MultiRowDecodeAttention.eligible(queries: q, keys: keys, values: values) {
            o = MultiRowDecodeAttention.attend(
                queries: q, keys: keys, values: values, scale: scale,
                windowBits: MultiRowDecodeAttention.windowBits(
                    scope.plan.paths.map { path in
                        var seen = [Bool](repeating: false, count: L)
                        for a in path { seen[a] = true }
                        return seen
                    }))
        } else {
            o = MLXFast.scaledDotProductAttention(
                queries: q, keys: keys, values: values, scale: scale,
                mask: .array(scope.attentionMask(prefix: prefix, dtype: q.dtype)))
        }
        return wo(o.transposed(0, 2, 1, 3).reshaped(1, L, -1))
    }
}

extension K2HorizonModel: DFlash2TreeVerifyModel {
    public func dflash2TreeForward(
        _ inputs: MLXArray, cache: [KVCache], captureLayerIDs: Set<Int>, scope: DFlash2TreeScope
    ) -> (MLXArray, [Int: MLXArray]) {
        DFlash2TreeScope.with(scope) {
            let (h, captured) = model.callAsFunctionCapturing(
                inputs, cache: cache, captureLayerIDs: captureLayerIDs)
            return (projectToLogits(h), captured)
        }
    }

    public func dflash2CommitTree(cache: [KVCache], scope: DFlash2TreeScope, keptRows: [Int]) -> Bool {
        for layer in cache {
            guard let kv = layer as? KVCacheSimple,
                kv.compactTreeWindow(window: scope.plan.rows, keptRows: keptRows)
            else { return false }
        }
        return true
    }

    public func dflash2SupportsTree(cache: [KVCache]) -> Bool {
        // Exact type: compile-path / quantized / rotating caches store rows differently.
        !cache.isEmpty && cache.allSatisfy { type(of: $0) == KVCacheSimple.self }
    }
}
