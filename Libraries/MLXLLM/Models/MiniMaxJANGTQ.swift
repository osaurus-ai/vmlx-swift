//
//  MiniMaxJANGTQ.swift
//  vMLXLLM
//
//  JANGTQ (TurboQuant codebook) variant of MiniMax — identical model
//  structure, but swaps `SwitchGLU` → `TurboQuantSwitchGLU` so the MoE
//  projections run the JANGTQ codebook Metal kernels instead of
//  `gather_qmm`. Attention / RMSNorm / RoPE / SDPA are unchanged — they
//  already call the same `mx.fast.*` C++ entry points Python uses.
//
//  Created by Jinho Jang (eric@jangq.ai).
//

import Foundation
import MLX
import MLXNN
import MLXLMCommon

// MARK: - Compiled router fast path

private struct MiniMaxJANGTQRouterKey: Hashable {
    let numExperts: Int
    let k: Int
}

private nonisolated(unsafe) var miniMaxJANGTQRouterCache:
    [MiniMaxJANGTQRouterKey: ([MLXArray]) -> [MLXArray]] = [:]
private let miniMaxJANGTQRouterLock = NSLock()

private func miniMaxJANGTQRouterCompileEnabled(environment: [String: String] = ProcessInfo.processInfo.environment) -> Bool {
    let raw = environment["VMLX_MINIMAX_ROUTER_COMPILE"]
        ?? environment["VMLINUX_MINIMAX_ROUTER_COMPILE"]
    switch raw?.lowercased() {
    case "1", "true", "on", "yes":
        return true
    default:
        return false
    }
}

private func miniMaxJANGTQRouter(numExperts: Int, k: Int) -> ([MLXArray]) -> [MLXArray] {
    let key = MiniMaxJANGTQRouterKey(numExperts: numExperts, k: k)
    miniMaxJANGTQRouterLock.lock()
    defer { miniMaxJANGTQRouterLock.unlock() }
    if let cached = miniMaxJANGTQRouterCache[key] { return cached }

    let topStart = numExperts - k
    let body: ([MLXArray]) -> [MLXArray] = { args in
        let gates = args[0]
        let bias = args[1]
        let originalScores = sigmoid(gates)
        let biasedScores = originalScores + bias
        let inds = argPartition(biasedScores, kth: topStart, axis: -1)[
            .ellipsis, topStart ..< numExperts]
        var scores = takeAlong(originalScores, inds, axis: -1)
        scores = scores
            / (scores.sum(axis: -1, keepDims: true) + MLXArray(1e-20, dtype: scores.dtype))
        return [inds, scores]
    }

    let router = (HardwareInfo.isCompiledDecodeSupported && miniMaxJANGTQRouterCompileEnabled())
        ? compile(shapeless: false, body)
        : body
    miniMaxJANGTQRouterCache[key] = router
    return router
}

private func miniMaxJANGTQHasCompleteExpertSet(
    _ weights: [String: MLXArray],
    prefix: String,
    originalProjection: String,
    suffix: String,
    expertCount: Int
) -> Bool {
    for expert in 0 ..< expertCount {
        if weights["\(prefix).experts.\(expert).\(originalProjection).\(suffix)"] == nil {
            return false
        }
    }
    return true
}

// MARK: - Attention (identical to MiniMax.swift)

private class MiniMaxJANGTQAttention: Module {
    let args: MiniMaxJANGTQConfiguration
    let scale: Float
    let numAttentionHeads: Int
    let numKeyValueHeads: Int
    let headDim: Int
    let qOutDim: Int
    let kvOutDim: Int

    @ModuleInfo(key: "qkv_proj") var wqkv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm?
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm?

    let rope: RoPE

    init(_ args: MiniMaxJANGTQConfiguration) {
        self.args = args
        self.numAttentionHeads = args.attentionHeads
        self.numKeyValueHeads = args.kvHeads
        self.headDim = args.headDim ?? (args.hiddenSize / args.attentionHeads)
        self.scale = pow(Float(headDim), -0.5)
        self.qOutDim = numAttentionHeads * headDim
        self.kvOutDim = numKeyValueHeads * headDim

        _wqkv.wrappedValue = Linear(args.hiddenSize, qOutDim + 2 * kvOutDim, bias: false)
        _wo.wrappedValue = Linear(qOutDim, args.hiddenSize, bias: false)

        if args.useQkNorm {
            _qNorm.wrappedValue = RMSNorm(
                dimensions: numAttentionHeads * headDim, eps: args.rmsNormEps)
            _kNorm.wrappedValue = RMSNorm(
                dimensions: numKeyValueHeads * headDim, eps: args.rmsNormEps)
        }

        self.rope = RoPE(
            dimensions: args.rotaryDim,
            traditional: false,
            base: args.ropeTheta
        )
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        let qkv = wqkv(x)
        var queries = qkv[.ellipsis, 0 ..< qOutDim]
        var keys = qkv[.ellipsis, qOutDim ..< (qOutDim + kvOutDim)]
        let values = qkv[.ellipsis, (qOutDim + kvOutDim) ..< (qOutDim + 2 * kvOutDim)]

        if let qNorm, let kNorm {
            queries = qNorm(queries)
            keys = kNorm(keys)
        }

        var q = queries.reshaped(B, L, numAttentionHeads, -1).transposed(0, 2, 1, 3)
        var k = keys.reshaped(B, L, numKeyValueHeads, -1).transposed(0, 2, 1, 3)
        let v = values.reshaped(B, L, numKeyValueHeads, -1).transposed(0, 2, 1, 3)

        q = applyRotaryPosition(rope, to: q, cache: cache)
        k = applyRotaryPosition(rope, to: k, cache: cache)

        let output = attentionWithCacheUpdate(
            queries: q, keys: k, values: v,
            cache: cache, scale: scale, mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return wo(output)
    }
}

// MARK: - MoE block (JANGTQ — swaps SwitchGLU for TurboQuantSwitchGLU)

private class MiniMaxJANGTQSparseMoeBlock: Module {
    let layerIdx: Int
    let numExpertsPerTok: Int

    @ModuleInfo(key: "gate") var gate: Linear
    @ModuleInfo(key: "switch_mlp") var switchMLP: TurboQuantSwitchGLU
    @ParameterInfo(key: "e_score_correction_bias") var eScoreCorrectionBias: MLXArray

    init(_ args: MiniMaxJANGTQConfiguration, layerIdx: Int) {
        self.layerIdx = layerIdx
        self.numExpertsPerTok = args.numExpertsPerTok

        _gate.wrappedValue = Linear(args.hiddenSize, args.numLocalExperts, bias: false)
        // Per-projection bits (JANGTQ_K) — fall back to uniform mxtqBits
        // when the config didn't ship per-projection overrides. Same
        // result as the pre-2026-05-04 uniform-bit constructor.
        let gateUpBits = args.mxtqGateUpBits ?? args.mxtqBits
        let downBits = args.mxtqDownBits ?? args.mxtqBits
        if JANGTQStreamingExperts.usesActiveExpertModule {
            _switchMLP.wrappedValue = StreamingTurboQuantSwitchGLU(
                inputDims: args.hiddenSize,
                hiddenDims: args.intermediateSize,
                numExperts: args.numLocalExperts,
                gateUpBits: gateUpBits,
                downBits: downBits,
                seed: args.mxtqSeed,
                layerIdx: layerIdx)
        } else {
            _switchMLP.wrappedValue = TurboQuantSwitchGLU(
                inputDims: args.hiddenSize,
                hiddenDims: args.intermediateSize,
                numExperts: args.numLocalExperts,
                gateUpBits: gateUpBits,
                downBits: downBits,
                seed: args.mxtqSeed
            )
        }
        _eScoreCorrectionBias.wrappedValue = MLXArray.zeros([args.numLocalExperts])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // CRITICAL: upcast x to fp32 before the gate Linear. Mirrors the
        // Python reference (`mlx_lm/models/minimax.py:178`):
        //     gates = self.gate(x.astype(mx.float32))
        //
        // With 154 experts (post-prune from 256), bf16 precision in the
        // gate matmul produces near-tied scores that cause argpartition
        // top-k to pick different experts on each run — giving
        // non-deterministic garbage output at T=0. fp32 stabilizes the
        // routing decision. (2026-05-02 fix; matches MiniMax.swift:309 affine path
        // which already does this correctly.)
        let gates = gate(x.asType(.float32))

        let routed = miniMaxJANGTQRouter(numExperts: gates.dim(-1), k: numExpertsPerTok)([
            gates, eScoreCorrectionBias,
        ])
        let inds = routed[0]
        JangPressCanonicalExpertAdvisor.shared.observe(layer: layerIdx, indices: inds)
        let scores = routed[1].asType(x.dtype)

        if let streaming = switchMLP as? StreamingTurboQuantSwitchGLU {
            return streaming.reduced(x, indices: inds, scores: scores)
        }
        let y = switchMLP(x, inds)
        return (y * scores[.ellipsis, .newAxis]).sum(axis: -2)
    }
}

// MARK: - Decoder layer

private class MiniMaxJANGTQDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: MiniMaxJANGTQAttention
    @ModuleInfo(key: "block_sparse_moe") var blockSparseMoe: MiniMaxJANGTQSparseMoeBlock
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ args: MiniMaxJANGTQConfiguration, layerIdx: Int) {
        _selfAttn.wrappedValue = MiniMaxJANGTQAttention(args)
        _blockSparseMoe.wrappedValue = MiniMaxJANGTQSparseMoeBlock(args, layerIdx: layerIdx)
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        var hidden = x + selfAttn(inputLayerNorm(x), mask: mask, cache: cache)
        hidden = hidden + blockSparseMoe(postAttentionLayerNorm(hidden))
        return hidden
    }
}

// MARK: - Inner model

public class MiniMaxJANGTQModelInner: Module {
    let args: MiniMaxJANGTQConfiguration

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    fileprivate let layers: [MiniMaxJANGTQDecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(_ args: MiniMaxJANGTQConfiguration) {
        self.args = args
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)
        self.layers = (0 ..< args.hiddenLayers).map { MiniMaxJANGTQDecoderLayer(args, layerIdx: $0) }
        _norm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = embedTokens(inputs)
        let mask = createAttentionMask(h: h, cache: cache?.first)
        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }
        return norm(h)
    }
}

// MARK: - Top-level model

public class MiniMaxJANGTQModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: MiniMaxJANGTQModelInner
    let configuration: MiniMaxJANGTQConfiguration
    let modelType: String

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ args: MiniMaxJANGTQConfiguration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = Array(repeating: args.kvHeads, count: args.hiddenLayers)
        self.modelType = args.modelType
        self.model = MiniMaxJANGTQModelInner(args)

        if !args.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let out = model(inputs, cache: cache)
        if let lmHead {
            return lmHead(out)
        }
        return model.embedTokens.asLinear(out)
    }

    /// Stacks per-expert JANGTQ tensors into the `switch_mlp` layout expected
    /// by `TurboQuantSwitchGLU`. Python writer uses `w1`/`w2`/`w3` tensor
    /// names (mirrors `MiniMax.swift` sanitize for the affine path). Also
    /// strips `.tq_bits` metadata tensors — they're not module parameters.
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = weights

        if configuration.tieWordEmbeddings {
            sanitized["lm_head.weight"] = nil
        }

        // Drop tq_bits metadata tensors anywhere in the tree.
        for key in sanitized.keys where key.hasSuffix(".tq_bits") {
            sanitized[key] = nil
        }

        // MiniMax keeps attention dense/affine while the MoE block is
        // TurboQuant. Fuse q/k/v at load time to match the optimized affine
        // path and avoid two extra decode matmul dispatches per layer.
        for layerIndex in 0 ..< configuration.hiddenLayers {
            let prefix = "model.layers.\(layerIndex).self_attn"
            let qKey = "\(prefix).q_proj"
            let kKey = "\(prefix).k_proj"
            let vKey = "\(prefix).v_proj"
            let fusedKey = "\(prefix).qkv_proj"

            guard let qW = sanitized["\(qKey).weight"],
                  let kW = sanitized["\(kKey).weight"],
                  let vW = sanitized["\(vKey).weight"]
            else { continue }

            let qPacked = qW.dim(qW.ndim - 1)
            let kPacked = kW.dim(kW.ndim - 1)
            let vPacked = vW.dim(vW.ndim - 1)
            if qPacked != kPacked || kPacked != vPacked {
                FileHandle.standardError.write(Data(
                    """
                    [MiniMaxJANGTQ sanitize] layer \(layerIndex) self_attn has \
                    mismatched bit widths across q/k/v projections \
                    (q packed_in=\(qPacked), k=\(kPacked), v=\(vPacked)). \
                    QKV fusion requires identical bit widths. \
                    Leaving source q/k/v keys intact so load verification fails.
                    """.utf8))
                continue
            }

            sanitized["\(fusedKey).weight"] = concatenated([qW, kW, vW], axis: 0)
            sanitized["\(qKey).weight"] = nil
            sanitized["\(kKey).weight"] = nil
            sanitized["\(vKey).weight"] = nil

            for suffix in ["scales", "biases"] {
                let qS = sanitized["\(qKey).\(suffix)"]
                let kS = sanitized["\(kKey).\(suffix)"]
                let vS = sanitized["\(vKey).\(suffix)"]
                guard let qS, let kS, let vS else { continue }
                sanitized["\(fusedKey).\(suffix)"] = concatenated([qS, kS, vS], axis: 0)
                sanitized["\(qKey).\(suffix)"] = nil
                sanitized["\(kKey).\(suffix)"] = nil
                sanitized["\(vKey).\(suffix)"] = nil
            }
        }

        let probe = "model.layers.0.block_sparse_moe.experts.0.w1.tq_packed"
        guard sanitized[probe] != nil else { return sanitized }

        let renames: [(String, String)] = [
            ("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")
        ]
        let residentExperts = JANGTQStreamingExperts.residentExpertsEnabled
        if residentExperts {
            JANGTQStreamingExperts.resetResidentTensors()
        }
        for layer in 0 ..< configuration.hiddenLayers {
            let prefix = "model.layers.\(layer).block_sparse_moe"
            for (orig, updated) in renames {
                for key in ["tq_packed", "tq_norms"] {
                    let first = "\(prefix).experts.0.\(orig).\(key)"
                    guard sanitized[first] != nil else { continue }
                    if JANGTQStreamingExperts.isEnabled {
                        for e in 0 ..< configuration.numLocalExperts {
                            sanitized["\(prefix).experts.\(e).\(orig).\(key)"] = nil
                        }
                        continue
                    }
                    if residentExperts {
                        guard miniMaxJANGTQHasCompleteExpertSet(
                            sanitized,
                            prefix: prefix,
                            originalProjection: orig,
                            suffix: key,
                            expertCount: configuration.numLocalExperts)
                        else {
                            FileHandle.standardError.write(Data(
                                """
                                [MiniMaxJANGTQ sanitize] layer \(layer) \(orig).\(key) \
                                has an incomplete resident expert tensor set. \
                                Leaving source expert keys intact so load verification fails.
                                """.utf8))
                            continue
                        }
                        for e in 0 ..< configuration.numLocalExperts {
                            let source = "\(prefix).experts.\(e).\(orig).\(key)"
                            guard let array = sanitized[source] else { continue }
                            sanitized[source] = nil
                            JANGTQStreamingExperts.registerResidentTensor(
                                layerIdx: layer,
                                expertIdx: e,
                                projectionName: updated,
                                suffixName: key,
                                array: array)
                        }
                        continue
                    }
                    let target = "\(prefix).switch_mlp.\(updated).\(key)"
                    if sanitized[target] != nil {
                        for e in 0 ..< configuration.numLocalExperts {
                            sanitized["\(prefix).experts.\(e).\(orig).\(key)"] = nil
                        }
                        continue
                    }
                    guard miniMaxJANGTQHasCompleteExpertSet(
                        sanitized,
                        prefix: prefix,
                        originalProjection: orig,
                        suffix: key,
                        expertCount: configuration.numLocalExperts)
                    else {
                        FileHandle.standardError.write(Data(
                            """
                            [MiniMaxJANGTQ sanitize] layer \(layer) \(orig).\(key) \
                            has an incomplete expert tensor set. \
                            Leaving source expert keys intact so load verification fails.
                            """.utf8))
                        continue
                    }
                    let stacked = (0 ..< configuration.numLocalExperts).map { e -> MLXArray in
                        let source = "\(prefix).experts.\(e).\(orig).\(key)"
                        let array = sanitized[source]!
                        sanitized[source] = nil
                        return array
                    }
                    sanitized[target] = loadTimeMachBackedStacked(stacked, label: target)
                        ?? loadTimeMaterializedStacked(stacked)
                }
            }
        }

        return sanitized
    }
}

// MARK: - Configuration

public struct MiniMaxJANGTQConfiguration: Codable, Sendable {
    public var modelType: String = "minimax_m2"
    public var hiddenSize: Int
    public var intermediateSize: Int
    public var attentionHeads: Int
    public var kvHeads: Int
    public var maxPositionEmbeddings: Int
    public var numExpertsPerTok: Int
    public var numLocalExperts: Int
    public var sharedIntermediateSize: Int
    public var hiddenLayers: Int
    public var rmsNormEps: Float
    public var ropeTheta: Float
    public var rotaryDim: Int
    public var vocabularySize: Int
    public var tieWordEmbeddings: Bool = false
    public var scoringFunc: String = "sigmoid"
    public var headDim: Int?
    public var useQkNorm: Bool = true

    // JANGTQ-specific
    public var weightFormat: String = "mxtq"
    public var mxtqBits: Int = 2
    public var mxtqSeed: Int = 42
    /// Per-projection bit widths (JANGTQ_K profile). When absent in
    /// config.json (uniform JANGTQ2 / JANGTQ4 bundles) both default to
    /// `mxtqBits` — bit-for-bit identical to the legacy uniform path.
    /// Surfaced separately so the model can construct the routed-MoE
    /// `TurboQuantSwitchGLU` with `gateUpBits != downBits` for
    /// MiniMax-M2.7-JANGTQ_K (gate=2 / up=2 / down=4). LLMModelFactory
    /// merges these from `jang_config.json:mxtq_bits.routed_expert`
    /// when the latter is a per-projection dict.
    public var mxtqGateUpBits: Int?
    public var mxtqDownBits: Int?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case maxPositionEmbeddings = "max_position_embeddings"
        case numExpertsPerTok = "num_experts_per_tok"
        case numLocalExperts = "num_local_experts"
        case sharedIntermediateSize = "shared_intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case rotaryDim = "rotary_dim"
        case vocabularySize = "vocab_size"
        case tieWordEmbeddings = "tie_word_embeddings"
        case scoringFunc = "scoring_func"
        case headDim = "head_dim"
        case useQkNorm = "use_qk_norm"
        case weightFormat = "weight_format"
        case mxtqBits = "mxtq_bits"
        case mxtqSeed = "mxtq_seed"
        case mxtqGateUpBits = "mxtq_gate_up_bits"
        case mxtqDownBits = "mxtq_down_bits"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? "minimax_m2"
        hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        kvHeads = try container.decode(Int.self, forKey: .kvHeads)
        maxPositionEmbeddings = try container.decode(Int.self, forKey: .maxPositionEmbeddings)
        numExpertsPerTok = RuntimeMoETopKOverride.effectiveTopK(
            currentTopK: try container.decode(Int.self, forKey: .numExpertsPerTok),
            modelType: modelType,
            field: CodingKeys.numExpertsPerTok.rawValue)
        numLocalExperts = try container.decode(Int.self, forKey: .numLocalExperts)
        sharedIntermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .sharedIntermediateSize) ?? 0
        hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        rmsNormEps = try container.decode(Float.self, forKey: .rmsNormEps)
        ropeTheta = try container.decode(Float.self, forKey: .ropeTheta)
        rotaryDim = try container.decode(Int.self, forKey: .rotaryDim)
        vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)
        tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        scoringFunc = try container.decodeIfPresent(String.self, forKey: .scoringFunc) ?? "sigmoid"
        headDim = try container.decodeIfPresent(Int.self, forKey: .headDim)
        useQkNorm = try container.decodeIfPresent(Bool.self, forKey: .useQkNorm) ?? true

        weightFormat =
            try container.decodeIfPresent(String.self, forKey: .weightFormat) ?? "mxtq"
        mxtqSeed = try container.decodeIfPresent(Int.self, forKey: .mxtqSeed) ?? 42
        mxtqGateUpBits = try container.decodeIfPresent(Int.self, forKey: .mxtqGateUpBits)
        mxtqDownBits = try container.decodeIfPresent(Int.self, forKey: .mxtqDownBits)

        if let routed = try container.decodeIfPresent(MxtqBitsSpec.self, forKey: .mxtqBits)?.routed {
            apply(routedBits: routed)
        } else if let routed = Self.peekQuantizationRoutedBits(decoder) {
            apply(routedBits: routed)
        } else {
            mxtqBits = 2
        }
    }

    private mutating func apply(routedBits: RoutedMxtqBits) {
        switch routedBits {
        case .uniform(let bits):
            mxtqBits = bits
        case .projected(let gateUp, let down):
            mxtqBits = gateUp ?? down ?? 2
            if mxtqGateUpBits == nil { mxtqGateUpBits = gateUp }
            if mxtqDownBits == nil { mxtqDownBits = down }
        }
    }
}

private enum MiniMaxQuantizationKey: String, CodingKey {
    case quantization
}

private struct MiniMaxQuantizationPeek: Decodable {
    let bits: Int?
    let routedExpertBits: RoutedMxtqBits?
    let mxtqBits: MxtqBitsSpec?

    enum CodingKeys: String, CodingKey {
        case bits
        case routedExpertBits = "routed_expert_bits"
        case mxtqBits = "mxtq_bits"
    }
}

private extension MiniMaxJANGTQConfiguration {
    static func peekQuantizationRoutedBits(_ decoder: Decoder) -> RoutedMxtqBits? {
        guard let outer = try? decoder.container(keyedBy: MiniMaxQuantizationKey.self),
              let q = try? outer.decodeIfPresent(
                MiniMaxQuantizationPeek.self, forKey: .quantization)
        else { return nil }
        if let routed = q.routedExpertBits {
            return routed
        }
        if let routed = q.mxtqBits?.routed {
            return routed
        }
        guard let bits = q.bits, bits == 2 || bits == 4 else { return nil }
        return .uniform(bits)
    }
}

private struct MxtqBitsSpec: Decodable {
    let routed: RoutedMxtqBits?

    enum CodingKeys: String, CodingKey {
        case routedExpert = "routed_expert"
        case routed
    }

    init(from decoder: Decoder) throws {
        if let flat = try? Int(from: decoder) {
            routed = .uniform(flat)
            return
        }
        if let container = try? decoder.container(keyedBy: CodingKeys.self),
            let nested =
                (try? container.decodeIfPresent(RoutedMxtqBits.self, forKey: .routedExpert))
                ?? (try? container.decodeIfPresent(RoutedMxtqBits.self, forKey: .routed))
        {
            routed = nested
            return
        }
        if let direct = try? RoutedMxtqBits(from: decoder), direct.isSpecified {
            routed = direct
            return
        }
        routed = nil
    }
}

private enum RoutedMxtqBits: Decodable {
    case uniform(Int)
    case projected(gateUp: Int?, down: Int?)

    enum CodingKeys: String, CodingKey {
        case gateProj = "gate_proj"
        case gateUpProj = "gate_up_proj"
        case upProj = "up_proj"
        case downProj = "down_proj"
    }

    init(from decoder: Decoder) throws {
        if let flat = try? Int(from: decoder) {
            self = .uniform(flat)
            return
        }
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let gate = try container.decodeIfPresent(Int.self, forKey: .gateProj)
        let gateUp = try container.decodeIfPresent(Int.self, forKey: .gateUpProj)
        let up = try container.decodeIfPresent(Int.self, forKey: .upProj)
        let down = try container.decodeIfPresent(Int.self, forKey: .downProj)
        self = .projected(gateUp: gateUp ?? gate ?? up, down: down)
    }

    var isSpecified: Bool {
        switch self {
        case .uniform:
            return true
        case .projected(let gateUp, let down):
            return gateUp != nil || down != nil
        }
    }
}

// MARK: - LoRA

extension MiniMaxJANGTQModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
