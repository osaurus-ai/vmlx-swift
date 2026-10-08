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

/// One quantized matmul for several projections that read the same input
/// (q/k/v, gate/up). K2 bundles stamp each role group at one scheme
/// (JANG_4M q/k/v 5-bit, gate/up 4-bit; JANGH7 8-bit / 6-bit), so the rows
/// concatenate into a single weight and one dispatch replaces two or three.
///
/// RAM: the members are repointed at row views of the fused buffers, so the
/// weights exist once — the originals are released, not kept for prefill.
/// Exactness: each output row is the same dot product over K in the same
/// order; `testFusedProjectionsAreBitIdentical` pins decode and prefill.
/// `VMLX_K2_FUSED_PROJECTIONS=0` is the same-binary control.
final class K2FusedProjection: @unchecked Sendable {
    nonisolated(unsafe) static var enabledOverride: Bool?
    /// Number of fused projections built in this process (tests, diagnostics).
    nonisolated(unsafe) static var builtCount = 0
    static var enabled: Bool {
        enabledOverride ?? (RuntimeEnvironment.value("VMLX_K2_FUSED_PROJECTIONS") != "0")
    }

    /// Mirrors MLXNN's internal `QuantizedRuntime.propagatePromotedOutput`.
    static let propagatePromotedOutput =
        ProcessInfo.processInfo.environment["VMLX_QUANTIZED_OUTPUT_PROMOTE"] == "1"

    let weight: MLXArray
    let scales: MLXArray
    let biases: MLXArray?
    let groupSize: Int
    let bits: Int
    let mode: QuantizationMode
    let splitIndices: [Int]

    private init(
        weight: MLXArray, scales: MLXArray, biases: MLXArray?, groupSize: Int, bits: Int,
        mode: QuantizationMode, splitIndices: [Int]
    ) {
        self.weight = weight
        self.scales = scales
        self.biases = biases
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode
        self.splitIndices = splitIndices
    }

    static func make(_ members: [Linear]) -> K2FusedProjection? {
        guard enabled, !CompiledDecodeTrace.isActive, members.count >= 2 else { return nil }
        let quantized = members.compactMap { $0 as? QuantizedLinear }
        guard quantized.count == members.count, let first = quantized.first,
            quantized.allSatisfy({
                $0.groupSize == first.groupSize && $0.bits == first.bits && $0.mode == first.mode
                    && $0.bias == nil && ($0.biases == nil) == (first.biases == nil)
                    && $0.weight.dtype == first.weight.dtype
                    && $0.scales.dtype == first.scales.dtype
                    && $0.biases?.dtype == first.biases?.dtype
            })
        else { return nil }
        let rows = quantized.map { $0.weight.dim(0) }
        let weight = concatenated(quantized.map(\.weight), axis: 0)
        let scales = concatenated(quantized.map(\.scales), axis: 0)
        let biases = first.biases == nil ? nil : concatenated(quantized.map { $0.biases! }, axis: 0)
        eval([weight, scales] + (biases.map { [$0] } ?? []))
        // Repoint every member at its row range of the fused buffers (views:
        // contiguous row slices share the buffer), dropping the originals.
        var start = 0
        var views: [MLXArray] = []
        for (member, count) in zip(quantized, rows) {
            let range = start ..< start + count
            var parameters: [String: MLXArray] = [
                "weight": weight[range], "scales": scales[range],
            ]
            if let biases { parameters["biases"] = biases[range] }
            views.append(contentsOf: parameters.values)
            member.update(parameters: ModuleParameters.unflattened(parameters))
            start += count
        }
        eval(views)
        var splits: [Int] = []
        var running = 0
        for count in rows.dropLast() {
            running += count
            splits.append(running)
        }
        builtCount += 1
        return K2FusedProjection(
            weight: weight, scales: scales, biases: biases, groupSize: first.groupSize,
            bits: first.bits, mode: first.mode, splitIndices: splits)
    }

    func callAsFunction(_ x: MLXArray) -> [MLXArray] {
        var y = quantizedMM(
            x, weight, scales: scales, biases: biases, transpose: true,
            groupSize: groupSize, bits: bits, mode: mode)
        // Same output-dtype pin as QuantizedLinear.callAsFunction.
        if !Self.propagatePromotedOutput, y.dtype != x.dtype,
            x.dtype == .bfloat16 || x.dtype == .float16
        {
            y = y.asType(x.dtype)
        }
        return split(y, indices: splitIndices, axis: -1)
    }
}

/// Lazily built once per module; held outside the Module property graph so
/// it is never reflected as parameters (the members keep the views).
final class K2FusionSlot: @unchecked Sendable {
    private var attempted = false
    private var fused: K2FusedProjection?
    func get(_ build: () -> K2FusedProjection?) -> K2FusedProjection? {
        if attempted { return fused }
        // Never build (eval) inside a compile trace; try again next call.
        guard !CompiledDecodeTrace.isActive else { return nil }
        attempted = true
        fused = build()
        return fused
    }
}

final class K2Attention: Module {
    let heads: Int, kvHeads: Int, scale: Float
    let rope: RoPE
    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear
    private let qkvFusion = K2FusionSlot()
    init(_ c: K2HorizonConfiguration) {
        heads = c.attentionHeads
        kvHeads = c.kvHeads
        scale = pow(Float(c.headDim), -0.5)
        _wq.wrappedValue = Linear(c.hiddenSize, heads * c.headDim, bias: false)
        _wk.wrappedValue = Linear(c.hiddenSize, kvHeads * c.headDim, bias: false)
        _wv.wrappedValue = Linear(c.hiddenSize, kvHeads * c.headDim, bias: false)
        _wo.wrappedValue = Linear(heads * c.headDim, c.hiddenSize, bias: false)
        rope = RoPE(dimensions: c.headDim, traditional: false, base: c.ropeTheta)
    }
    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))
        let projected: [MLXArray] =
            qkvFusion.get({ K2FusedProjection.make([wq, wk, wv]) })?(x) ?? [wq(x), wk(x), wv(x)]
        var q = projected[0].reshaped(B, L, heads, -1).transposed(0, 2, 1, 3)
        var k = projected[1].reshaped(B, L, kvHeads, -1).transposed(0, 2, 1, 3)
        let v = projected[2].reshaped(B, L, kvHeads, -1).transposed(0, 2, 1, 3)
        q = applyRotaryPosition(rope, to: q, cache: cache)
        k = applyRotaryPosition(rope, to: k, cache: cache)
        let o = attentionWithCacheUpdate(
            queries: q, keys: k, values: v, cache: cache, scale: scale, mask: mask)
        return wo(o.transposed(0, 2, 1, 3).reshaped(B, L, -1))
    }
}

final class K2DenseMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Module
    private let gateUpFusion = K2FusionSlot()
    init(_ d: Int, _ h: Int, down: JANGHDenseLinear? = nil) {
        _gate.wrappedValue = Linear(d, h, bias: false)
        _up.wrappedValue = Linear(d, h, bias: false)
        _down.wrappedValue = down ?? Linear(h, d, bias: false)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        if let parts = gateUpFusion.get({ K2FusedProjection.make([gate, up]) })?(x) {
            return (down as! UnaryLayer)(silu(parts[0]) * parts[1])
        }
        return (down as! UnaryLayer)(silu(gate(x)) * up(x))
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
        var h = embedTokens(inputs)
        let mask = createAttentionMask(h: h, cache: cache?.first)
        for (i, layer) in layers.enumerated() { h = layer(h, mask: mask, cache: cache?[i]) }
        return norm(h)
    }
}

public final class K2HorizonModel: Module, LLMModel, KVCacheDimensionProvider {
    public typealias RoutedFactory = (Int) throws -> (Module & WeightedRoutedExpertLayer)
    public let vocabularySize: Int
    public let kvHeads: [Int]
    let configuration: K2HorizonConfiguration
    let excludedSafetensorsKeys: Set<String>
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
        if let lmHead { return lmHead(h) }
        return model.embedTokens.asLinear(h)
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
