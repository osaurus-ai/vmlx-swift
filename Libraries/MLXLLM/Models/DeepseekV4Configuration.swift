// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// DeepSeek-V4 (DSV4-Flash / DSV4-Pro) configuration.
//
// Mirrors `ModelArgs` in the Python reference
// `jang-tools/jang_tools/dsv4_prune/mlx_model.py` and the fields
// documented in `jang/research/DSV4-RUNTIME-ARCHITECTURE.md` §1.
//
// DSV4 is architecturally distinct from DSV3 — carries mHC (manifold
// hyper-connections), MLA with head_dim=512 (no split nope/pe), grouped
// low-rank O, learned attention sinks, sqrtsoftplus routing, hash
// routing for first `numHashLayers`, per-layer compress_ratio (0/4/128),
// YaRN RoPE only on compress_ratio>0 layers, and swiglu_limit=10.

import Foundation
import MLXLMCommon

/// DeepSeek-V4 architecture + tokenizer + quant configuration.
/// Every field is decoded from `config.json` (via `CodingKeys`) and
/// has a sensible default matching DSV4-Flash (284B / 21B active).
public struct DeepseekV4Configuration: Codable, Sendable {
    // MARK: - Core transformer

    public var vocabSize: Int = 129_280
    public var hiddenSize: Int = 4096
    public var numHiddenLayers: Int = 43
    public var numAttentionHeads: Int = 64
    /// DSV4 uses a SINGLE latent KV head broadcast to all Q heads.
    public var numKeyValueHeads: Int = 1
    public var headDim: Int = 512
    /// Rotary applied only to last `qkRopeHeadDim` dims of the
    /// head-dim=512 vector; the first (headDim - qkRopeHeadDim) = 448
    /// dims are "no-position".
    public var qkRopeHeadDim: Int = 64
    public var qLoraRank: Int = 1024
    public var rmsNormEps: Float = 1e-6
    public var maxPositionEmbeddings: Int = 1_048_576

    // MARK: - MLA — grouped low-rank O

    /// `wo_a` splits head output into `oGroups` × `oLoraRank` via an
    /// einsum `bsgd,grd→bsgr`, then concatenates groups before `wo_b`.
    public var oGroups: Int = 8
    public var oLoraRank: Int = 1024

    // MARK: - MoE (Mixture of Experts)

    public var nRoutedExperts: Int = 256
    public var nSharedExperts: Int = 1
    public var numExpertsPerTok: Int = 6
    public var moeIntermediateSize: Int = 2048
    /// Hash routing bypasses topk for the first `numHashLayers` layers —
    /// a learned `tid2eid` table maps token id → expert id directly.
    public var numHashLayers: Int = 3
    public var scoringFunc: String = "sqrtsoftplus"
    public var normTopkProb: Bool = true
    public var routedScalingFactor: Float = 1.5
    /// Clamp for DSV4 SwiGLU: `silu(min(gate, lim)) * clip(up, ±lim)`.
    /// Set to 10.0 in DSV4-Flash; essential to prevent activation blow-up.
    public var swigluLimit: Float = 10.0
    /// Default MXTQ bit width for routed experts. DSV4 JANGTQ-K bundles
    /// can override individual layers through ``routedExpertLayerBits``.
    public var routedExpertDefaultBits: Int = 2
    /// Per-layer routed expert bit overrides from `routed_expert_bit_plan`.
    /// Example: DSV4-Flash-JANGTQ-K keeps most routed layers at 2-bit but
    /// preserves layers 23/25/28/34/36 at 4-bit.
    public var routedExpertLayerBits: [Int: Int] = [:]

    // MARK: - mHC (Manifold Hyper-Connections)

    /// Number of parallel residual-stream copies threaded through each
    /// decoder block (collapse → process → expand). Sinkhorn
    /// doubly-stochastic mixing matrix preserves residual norm.
    public var hcMult: Int = 4
    /// Sinkhorn iterations for `comb` row/col normalization.
    public var hcSinkhornIters: Int = 20
    public var hcEps: Float = 1e-6

    // MARK: - RoPE

    /// Rope theta for layers with `compress_ratio == 0` (no YaRN).
    public var ropeTheta: Float = 10000.0
    /// Rope theta for layers with `compress_ratio > 0` (with YaRN).
    public var compressRopeTheta: Float = 160000.0
    public var ropeScaling: [String: StringOrNumber]? = nil

    // MARK: - Sliding window + compressor

    public var slidingWindow: Int = 128
    /// Per-layer compress ratio ∈ {0, 4, 128}. Layers with >0 use the
    /// Compressor + (for ratio=4) Indexer path for global context.
    public var compressRatios: [Int] = []

    // MARK: - Indexer (sparse attention, only layers with ratio=4)

    public var indexNHeads: Int = 64
    public var indexHeadDim: Int = 128
    public var indexTopk: Int = 512

    // MARK: - Attention sink (learned per-head logit prepended pre-softmax)

    /// Whether the model ships a learned per-head `attn_sink` bias that
    /// is appended as a logit column before softmax (then dropped). DSV4
    /// ships it per layer; setting to false disables the contribution.
    public var useAttnSink: Bool = true

    // MARK: - CodingKeys

    enum CodingKeys: String, CodingKey {
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case qkRopeHeadDim = "qk_rope_head_dim"
        case qLoraRank = "q_lora_rank"
        case rmsNormEps = "rms_norm_eps"
        case maxPositionEmbeddings = "max_position_embeddings"
        case oGroups = "o_groups"
        case oLoraRank = "o_lora_rank"
        case nRoutedExperts = "n_routed_experts"
        case nSharedExperts = "n_shared_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case moeIntermediateSize = "moe_intermediate_size"
        case numHashLayers = "num_hash_layers"
        case scoringFunc = "scoring_func"
        case normTopkProb = "norm_topk_prob"
        case routedScalingFactor = "routed_scaling_factor"
        case swigluLimit = "swiglu_limit"
        case routedExpertBits = "routed_expert_bits"
        case mxtqBits = "mxtq_bits"
        case routedExpertBitPlan = "routed_expert_bit_plan"
        case quantization
        case hcMult = "hc_mult"
        case hcSinkhornIters = "hc_sinkhorn_iters"
        case hcEps = "hc_eps"
        case ropeTheta = "rope_theta"
        case compressRopeTheta = "compress_rope_theta"
        case ropeScaling = "rope_scaling"
        case slidingWindow = "sliding_window"
        case compressRatios = "compress_ratios"
        case indexNHeads = "index_n_heads"
        case indexHeadDim = "index_head_dim"
        case indexTopk = "index_topk"
        case useAttnSink = "use_attn_sink"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)

        func req<T: Decodable>(_ k: CodingKeys, _ fallback: T) -> T {
            (try? c.decode(T.self, forKey: k)) ?? fallback
        }

        self.vocabSize = req(.vocabSize, 129_280)
        self.hiddenSize = req(.hiddenSize, 4096)
        self.numHiddenLayers = req(.numHiddenLayers, 43)
        self.numAttentionHeads = req(.numAttentionHeads, 64)
        self.numKeyValueHeads = req(.numKeyValueHeads, 1)
        self.headDim = req(.headDim, 512)
        self.qkRopeHeadDim = req(.qkRopeHeadDim, 64)
        self.qLoraRank = req(.qLoraRank, 1024)
        self.rmsNormEps = req(.rmsNormEps, 1e-6)
        self.maxPositionEmbeddings = req(.maxPositionEmbeddings, 1_048_576)
        self.oGroups = req(.oGroups, 8)
        self.oLoraRank = req(.oLoraRank, 1024)
        self.nRoutedExperts = req(.nRoutedExperts, 256)
        self.nSharedExperts = req(.nSharedExperts, 1)
        let configuredNumExpertsPerTok: Int = req(.numExpertsPerTok, 6)
        self.numExpertsPerTok = RuntimeMoETopKOverride.effectiveTopK(
            currentTopK: configuredNumExpertsPerTok,
            modelType: "deepseek_v4",
            field: "num_experts_per_tok")
        self.moeIntermediateSize = req(.moeIntermediateSize, 2048)
        self.numHashLayers = req(.numHashLayers, 3)
        self.scoringFunc = req(.scoringFunc, "sqrtsoftplus")
        self.normTopkProb = req(.normTopkProb, true)
        self.routedScalingFactor = req(.routedScalingFactor, 1.5)
        self.swigluLimit = req(.swigluLimit, 10.0)
        self.routedExpertDefaultBits = Self.decodeRoutedExpertDefaultBits(from: c)
        self.routedExpertLayerBits = Self.decodeRoutedExpertLayerBits(from: c)
        self.hcMult = req(.hcMult, 4)
        self.hcSinkhornIters = req(.hcSinkhornIters, 20)
        self.hcEps = req(.hcEps, 1e-6)
        self.ropeTheta = req(.ropeTheta, 10000.0)
        self.compressRopeTheta = req(.compressRopeTheta, 160_000.0)
        self.ropeScaling = try? c.decode([String: StringOrNumber].self, forKey: .ropeScaling)
        self.slidingWindow = req(.slidingWindow, 128)
        self.compressRatios = req(.compressRatios, [])
        self.indexNHeads = req(.indexNHeads, 64)
        self.indexHeadDim = req(.indexHeadDim, 128)
        self.indexTopk = req(.indexTopk, 512)
        self.useAttnSink = req(.useAttnSink, true)

        try validateDecodedFields(container: c)
    }

    public init() {}

    private func validateDecodedFields(container c: KeyedDecodingContainer<CodingKeys>) throws {
        try Self.validatePositive(vocabSize, key: .vocabSize, in: c)
        try Self.validatePositive(hiddenSize, key: .hiddenSize, in: c)
        try Self.validatePositive(numHiddenLayers, key: .numHiddenLayers, in: c)
        try Self.validatePositive(numAttentionHeads, key: .numAttentionHeads, in: c)
        try Self.validatePositive(numKeyValueHeads, key: .numKeyValueHeads, in: c)
        try Self.validatePositive(headDim, key: .headDim, in: c)
        try Self.validatePositive(qkRopeHeadDim, key: .qkRopeHeadDim, in: c)
        try Self.validatePositive(qLoraRank, key: .qLoraRank, in: c)
        try Self.validatePositive(rmsNormEps, key: .rmsNormEps, in: c)
        try Self.validatePositive(maxPositionEmbeddings, key: .maxPositionEmbeddings, in: c)
        try Self.validatePositive(oGroups, key: .oGroups, in: c)
        try Self.validatePositive(oLoraRank, key: .oLoraRank, in: c)
        try Self.validatePositive(nRoutedExperts, key: .nRoutedExperts, in: c)
        try Self.validateNonNegative(nSharedExperts, key: .nSharedExperts, in: c)
        try Self.validatePositive(numExpertsPerTok, key: .numExpertsPerTok, in: c)
        try Self.validatePositive(moeIntermediateSize, key: .moeIntermediateSize, in: c)
        try Self.validateNonNegative(numHashLayers, key: .numHashLayers, in: c)
        try Self.validatePositive(routedScalingFactor, key: .routedScalingFactor, in: c)
        try Self.validatePositive(swigluLimit, key: .swigluLimit, in: c)
        try Self.validatePositive(hcMult, key: .hcMult, in: c)
        try Self.validatePositive(hcSinkhornIters, key: .hcSinkhornIters, in: c)
        try Self.validatePositive(hcEps, key: .hcEps, in: c)
        try Self.validatePositive(ropeTheta, key: .ropeTheta, in: c)
        try Self.validatePositive(compressRopeTheta, key: .compressRopeTheta, in: c)
        try Self.validatePositive(slidingWindow, key: .slidingWindow, in: c)
        try Self.validatePositive(indexNHeads, key: .indexNHeads, in: c)
        try Self.validatePositive(indexHeadDim, key: .indexHeadDim, in: c)
        try Self.validatePositive(indexTopk, key: .indexTopk, in: c)

        guard qkRopeHeadDim <= headDim else {
            throw DecodingError.dataCorruptedError(
                forKey: .qkRopeHeadDim,
                in: c,
                debugDescription: "DeepseekV4 qk_rope_head_dim must be in 1...head_dim.")
        }
        guard hiddenSize % numAttentionHeads == 0 else {
            throw DecodingError.dataCorruptedError(
                forKey: .hiddenSize,
                in: c,
                debugDescription:
                    "DeepseekV4 hidden_size must be divisible by num_attention_heads.")
        }
        guard numAttentionHeads % numKeyValueHeads == 0 else {
            throw DecodingError.dataCorruptedError(
                forKey: .numKeyValueHeads,
                in: c,
                debugDescription:
                    "DeepseekV4 num_attention_heads must be divisible by num_key_value_heads.")
        }
        guard numExpertsPerTok <= nRoutedExperts else {
            throw DecodingError.dataCorruptedError(
                forKey: .numExpertsPerTok,
                in: c,
                debugDescription:
                    "DeepseekV4 num_experts_per_tok must be in 1...n_routed_experts.")
        }
        guard numHashLayers <= numHiddenLayers else {
            throw DecodingError.dataCorruptedError(
                forKey: .numHashLayers,
                in: c,
                debugDescription:
                    "DeepseekV4 num_hash_layers must be in 0...num_hidden_layers.")
        }
        if !compressRatios.isEmpty {
            guard compressRatios.count == numHiddenLayers else {
                throw DecodingError.dataCorruptedError(
                    forKey: .compressRatios,
                    in: c,
                    debugDescription:
                        "DeepseekV4 compress_ratios count must match num_hidden_layers when provided.")
            }
            for ratio in compressRatios where ratio != 0 && ratio != 4 && ratio != 128 {
                throw DecodingError.dataCorruptedError(
                    forKey: .compressRatios,
                    in: c,
                    debugDescription:
                        "DeepseekV4 compress_ratios entries must be one of 0, 4, or 128.")
            }
        }
        try Self.validateRoutedExpertBits(routedExpertDefaultBits, key: .routedExpertBits, in: c)
        for (layer, bits) in routedExpertLayerBits {
            guard layer >= 0 && layer < numHiddenLayers else {
                throw DecodingError.dataCorruptedError(
                    forKey: .routedExpertBitPlan,
                    in: c,
                    debugDescription:
                        "DeepseekV4 routed expert bit plan layers must be in 0..<num_hidden_layers.")
            }
            try Self.validateRoutedExpertBits(bits, key: .routedExpertBitPlan, in: c)
        }
    }

    private static func validatePositive<K: CodingKey>(
        _ value: Int, key: K, in container: KeyedDecodingContainer<K>
    ) throws {
        guard value > 0 else {
            throw DecodingError.dataCorruptedError(
                forKey: key,
                in: container,
                debugDescription:
                    "DeepseekV4 config \(key.stringValue) must be greater than zero.")
        }
    }

    private static func validateNonNegative<K: CodingKey>(
        _ value: Int, key: K, in container: KeyedDecodingContainer<K>
    ) throws {
        guard value >= 0 else {
            throw DecodingError.dataCorruptedError(
                forKey: key,
                in: container,
                debugDescription: "DeepseekV4 config \(key.stringValue) must be nonnegative.")
        }
    }

    private static func validatePositive<K: CodingKey>(
        _ value: Float, key: K, in container: KeyedDecodingContainer<K>
    ) throws {
        guard value.isFinite && value > 0 else {
            throw DecodingError.dataCorruptedError(
                forKey: key,
                in: container,
                debugDescription:
                    "DeepseekV4 config \(key.stringValue) must be finite and greater than zero.")
        }
    }

    private static func validateRoutedExpertBits<K: CodingKey>(
        _ bits: Int, key: K, in container: KeyedDecodingContainer<K>
    ) throws {
        guard bits == 2 || bits == 4 else {
            throw DecodingError.dataCorruptedError(
                forKey: key,
                in: container,
                debugDescription: "DeepseekV4 routed expert bits must be 2 or 4.")
        }
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(vocabSize, forKey: .vocabSize)
        try c.encode(hiddenSize, forKey: .hiddenSize)
        try c.encode(numHiddenLayers, forKey: .numHiddenLayers)
        try c.encode(numAttentionHeads, forKey: .numAttentionHeads)
        try c.encode(numKeyValueHeads, forKey: .numKeyValueHeads)
        try c.encode(headDim, forKey: .headDim)
        try c.encode(qkRopeHeadDim, forKey: .qkRopeHeadDim)
        try c.encode(qLoraRank, forKey: .qLoraRank)
        try c.encode(rmsNormEps, forKey: .rmsNormEps)
        try c.encode(maxPositionEmbeddings, forKey: .maxPositionEmbeddings)
        try c.encode(oGroups, forKey: .oGroups)
        try c.encode(oLoraRank, forKey: .oLoraRank)
        try c.encode(nRoutedExperts, forKey: .nRoutedExperts)
        try c.encode(nSharedExperts, forKey: .nSharedExperts)
        try c.encode(numExpertsPerTok, forKey: .numExpertsPerTok)
        try c.encode(moeIntermediateSize, forKey: .moeIntermediateSize)
        try c.encode(numHashLayers, forKey: .numHashLayers)
        try c.encode(scoringFunc, forKey: .scoringFunc)
        try c.encode(normTopkProb, forKey: .normTopkProb)
        try c.encode(routedScalingFactor, forKey: .routedScalingFactor)
        try c.encode(swigluLimit, forKey: .swigluLimit)
        try c.encode(routedExpertDefaultBits, forKey: .routedExpertBits)
        if !routedExpertLayerBits.isEmpty {
            let layerBits = Dictionary(uniqueKeysWithValues: routedExpertLayerBits.map {
                (String($0.key), $0.value)
            })
            try c.encode(
                RoutedExpertBitPlan(
                    defaultBits: routedExpertDefaultBits,
                    routedLayerBits: layerBits),
                forKey: .routedExpertBitPlan)
        }
        try c.encode(hcMult, forKey: .hcMult)
        try c.encode(hcSinkhornIters, forKey: .hcSinkhornIters)
        try c.encode(hcEps, forKey: .hcEps)
        try c.encode(ropeTheta, forKey: .ropeTheta)
        try c.encode(compressRopeTheta, forKey: .compressRopeTheta)
        try c.encodeIfPresent(ropeScaling, forKey: .ropeScaling)
        try c.encode(slidingWindow, forKey: .slidingWindow)
        try c.encode(compressRatios, forKey: .compressRatios)
        try c.encode(indexNHeads, forKey: .indexNHeads)
        try c.encode(indexHeadDim, forKey: .indexHeadDim)
        try c.encode(indexTopk, forKey: .indexTopk)
        try c.encode(useAttnSink, forKey: .useAttnSink)
    }

    private struct MxtqBitsSpec: Decodable {
        let routedExpert: Int?

        enum CodingKeys: String, CodingKey {
            case routedExpert = "routed_expert"
        }

        init(from decoder: Decoder) throws {
            if let single = try? decoder.singleValueContainer().decode(Int.self) {
                self.routedExpert = single
                return
            }
            let c = try decoder.container(keyedBy: CodingKeys.self)
            self.routedExpert = try c.decodeIfPresent(Int.self, forKey: .routedExpert)
        }
    }

    private struct RoutedExpertBitPlan: Codable {
        let defaultBits: Int?
        let routedLayerBits: [String: Int]?

        init(defaultBits: Int?, routedLayerBits: [String: Int]?) {
            self.defaultBits = defaultBits
            self.routedLayerBits = routedLayerBits
        }

        enum CodingKeys: String, CodingKey {
            case defaultBits = "default_bits"
            case routedLayerBits = "routed_layer_bits"
        }
    }

    private struct QuantizationSpec: Decodable {
        let routedExpertBits: Int?
        let mxtqBits: MxtqBitsSpec?
        let routedExpertBitPlan: RoutedExpertBitPlan?
        let routedExperts: RoutedExperts?

        enum CodingKeys: String, CodingKey {
            case routedExpertBits = "routed_expert_bits"
            case mxtqBits = "mxtq_bits"
            case routedExpertBitPlan = "routed_expert_bit_plan"
            case routedExperts = "routed_experts"
        }

        struct RoutedExperts: Decodable {
            let bits: Int?
            let bitPlan: RoutedExpertBitPlan?

            enum CodingKeys: String, CodingKey {
                case bits
                case bitPlan = "bit_plan"
            }
        }
    }

    private static func decodeRoutedExpertDefaultBits(
        from c: KeyedDecodingContainer<CodingKeys>
    ) -> Int {
        if let topPlan = try? c.decode(RoutedExpertBitPlan.self, forKey: .routedExpertBitPlan),
           let bits = topPlan.defaultBits {
            return bits
        }
        if let bits = try? c.decode(Int.self, forKey: .routedExpertBits) {
            return bits
        }
        if let mxtq = try? c.decode(MxtqBitsSpec.self, forKey: .mxtqBits),
           let bits = mxtq.routedExpert {
            return bits
        }
        if let q = try? c.decode(QuantizationSpec.self, forKey: .quantization) {
            if let bits = q.routedExpertBitPlan?.defaultBits {
                return bits
            }
            if let bits = q.routedExperts?.bitPlan?.defaultBits {
                return bits
            }
            if let bits = q.routedExpertBits {
                return bits
            }
            if let bits = q.routedExperts?.bits {
                return bits
            }
            if let bits = q.mxtqBits?.routedExpert {
                return bits
            }
        }
        return 2
    }

    private static func decodeRoutedExpertLayerBits(
        from c: KeyedDecodingContainer<CodingKeys>
    ) -> [Int: Int] {
        func parse(_ raw: [String: Int]?) -> [Int: Int] {
            guard let raw else { return [:] }
            var result: [Int: Int] = [:]
            for (key, value) in raw {
                if let layer = Int(key) {
                    result[layer] = value
                }
            }
            return result
        }

        if let plan = try? c.decode(RoutedExpertBitPlan.self, forKey: .routedExpertBitPlan),
           !(plan.routedLayerBits?.isEmpty ?? true) {
            return parse(plan.routedLayerBits)
        }
        if let q = try? c.decode(QuantizationSpec.self, forKey: .quantization) {
            if !(q.routedExpertBitPlan?.routedLayerBits?.isEmpty ?? true) {
                return parse(q.routedExpertBitPlan?.routedLayerBits)
            }
            if !(q.routedExperts?.bitPlan?.routedLayerBits?.isEmpty ?? true) {
                return parse(q.routedExperts?.bitPlan?.routedLayerBits)
            }
        }
        return [:]
    }
}

extension DeepseekV4Configuration {
    /// True for layers that carry the compressor (and, at ratio=4, the
    /// indexer). The `Compressor` + `Indexer` modules attach only to
    /// layers with `compress_ratio > 0`.
    public func hasCompressor(layer: Int) -> Bool {
        guard layer < compressRatios.count else { return false }
        return compressRatios[layer] > 0
    }

    /// True for the first `numHashLayers` — these bypass softmax topk
    /// and route tokens to experts via the `tid2eid` hash table.
    public func isHashLayer(_ layer: Int) -> Bool {
        layer < numHashLayers
    }

    /// Per-layer rope theta. DSV4 uses a higher theta on compressor
    /// layers (with YaRN scaling), lower theta on plain attention.
    public func ropeTheta(forLayer layer: Int) -> Float {
        hasCompressor(layer: layer) ? compressRopeTheta : ropeTheta
    }

    /// Routed expert MXTQ bit width for a decoder layer.
    public func routedExpertBits(forLayer layer: Int) -> Int {
        routedExpertLayerBits[layer] ?? routedExpertDefaultBits
    }
}
