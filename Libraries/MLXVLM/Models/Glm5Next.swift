// Copyright © 2026 osaurus-eval contributors
// SPDX-License-Identifier: MIT
//
// GLM-5.3 (`glm5_next`) — configuration, construction plan, and checkpoint-key policy.
//
// WHAT THIS FILE IS, AND IS NOT
//     It is the part that can be verified today: the configuration decodes the real bundle's
//     `config.json`, the construction plan resolves through `ModelComponentMapping` like every other
//     converted family, and `sanitize` implements the checkpoint-key policy the bundle actually
//     needs. The decoder's forward pass is NOT here — see `Glm5NextDecoderUnavailable`. Writing one
//     against weights that are still downloading would be code no test could exercise.
//
// WHY SO LITTLE OF IT IS NEW
//     Read from the shipped bundle (1434 tensors over the completed shards), almost every mechanism
//     already exists in this repo:
//
//       * `attn_hc` / `ffn_hc` are hyper-connections with Sinkhorn mixing — `DeepseekV4HyperConnection`,
//         down to the `deepseek_v4_hc_split_sinkhorn` kernel. The bundle stores the SAME three
//         parameters under an `hc_` prefix, which is why `sanitize` strips it rather than a new
//         module being written.
//       * the attention names are DeepSeek MLA verbatim: `q_a_proj`, `q_b_proj`,
//         `kv_a_proj_with_mqa`, `kv_a_layernorm`, `kv_b_proj`.
//       * `A_log`, `dt_bias` and `q/k/v_conv1d` are the gated-delta linear attention Qwen3.5 uses.
//       * the MoE is DeepSeek's: `switch_mlp`, `shared_experts`, `e_score_correction_bias`,
//         `noaux_tc` routing over a sigmoid score.
//
//     The genuinely new piece is the indexer's key-pooling compression
//     (`index_kpool_compress_ape` / `_gate`), which has no counterpart here.

import Foundation
import MLX
import AVFoundation
import CoreImage
import CoreMedia
import MLXLLM
import MLXLMCommon
import MLXNN

// MARK: - Configuration

/// Which attention a decoder layer runs. GLM-5.3 interleaves them explicitly rather than by
/// interval — 34 linear to 11 sparse in the shipped 45-layer bundle — so the schedule is read as a
/// list and not derived from a stride the way `Qwen35`'s `full_attention_interval` is.
/// Gate for the GLM5-next sparse-index pooling/scoring dtype. Default runs
/// in the packed cache's native dtype (no per-step full-context fp32
/// copies); the env var restores the historical fp32 pipeline for A/B.
enum Glm5NextIndexerRuntime {
    nonisolated(unsafe) static var poolFP32: Bool = {
        ProcessInfo.processInfo.environment["VMLX_GLM5_INDEX_FP32"] == "1"
    }()
}

public enum Glm5NextLayerKind: String, Codable, Sendable {
    case linearAttention = "linear_attention"
    case deepseekSparseAttention = "deepseek_sparse_attention"
}

/// Whether a layer's MLP is dense or a routed mixture. `first_k_dense_replace` states the same thing
/// as a count; the list is authoritative and is what this reads.
public enum Glm5NextMLPKind: String, Codable, Sendable {
    case dense
    case sparse
}

/// The linear-attention stanza. Its quantities are exactly KDA's — which is what identifies the
/// mechanism: `num_heads`, `head_dim`, `short_conv_kernel_size` and a `gate_lower_bound` that is
/// Ling 3.0's `kda_lower_bound` under another name.
///
/// It also names the schedule a SECOND time, as `kda_layers` and `full_attn_layers`. Two independent
/// statements of the same fact are worth checking against each other rather than picking one.
public struct Glm5NextLinearAttentionConfiguration: Codable, Sendable {
    public let numHeads: Int
    public let headDim: Int
    public let shortConvKernelSize: Int
    public let gateLowerBound: Float
    public let kdaLayers: [Int]?
    public let fullAttnLayers: [Int]?

    enum CodingKeys: String, CodingKey {
        case numHeads = "num_heads"
        case headDim = "head_dim"
        case shortConvKernelSize = "short_conv_kernel_size"
        case gateLowerBound = "gate_lower_bound"
        case kdaLayers = "kda_layers"
        case fullAttnLayers = "full_attn_layers"
    }
}

public struct Glm5NextTextConfiguration: Codable, Sendable {
    public let modelType: String
    public let hiddenSize: Int
    public let numHiddenLayers: Int
    public let intermediateSize: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let vocabSize: Int
    public let rmsNormEps: Float
    public let maxPositionEmbeddings: Int

    // MLA. `qkRopeHeadDim` is 0 in the shipped bundle and `mlaUseNope` is true: this family runs
    // MLA with NO rotary split, which is not the DeepSeek default and is why both are read rather
    // than assumed.
    public let kvLoraRank: Int
    public let qLoraRank: Int
    public let qkNopeHeadDim: Int
    public let qkRopeHeadDim: Int
    public let vHeadDim: Int
    public let mlaUseNope: Bool

    // Mixture of experts.
    public let nRoutedExperts: Int
    public let nSharedExperts: Int
    public let numExpertsPerTok: Int
    public let moeIntermediateSize: Int
    public let firstKDenseReplace: Int
    public let scoringFunc: String
    public let topkMethod: String
    public let routedScalingFactor: Float
    public let normTopkProb: Bool
    /// Both are 1 in the shipped bundle, i.e. grouped routing is effectively off — but they are read
    /// rather than assumed, because a variant that turns it on would otherwise route wrongly while
    /// every shape still matched.
    public let nGroup: Int
    public let topkGroup: Int

    // Hyper-connections, shared with DeepSeek V4.
    public let mhc: Bool
    public let hcMult: Int
    public let hcSinkhornIters: Int
    public let hcEps: Float

    // Sparse indexer.
    public let indexHeadDim: Int
    public let indexNHeads: Int
    public let indexTopk: Int
    public let indexKpool: Int
    public let indexKpoolCompress: Bool
    public let indexKpoolAlwaysSelectTail: Bool

    public let numNextnPredictLayers: Int
    /// Declared on BOTH the text config and the top level in this bundle; the text one governs the
    /// head, and defaults false so a bundle omitting it still ships an `lm_head`.
    public let tieWordEmbeddings: Bool
    public let swigluLimit: Float?

    public let layerTypes: [Glm5NextLayerKind]
    public let mlpLayerTypes: [Glm5NextMLPKind]
    public let linearAttnConfig: Glm5NextLinearAttentionConfiguration

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case vocabSize = "vocab_size"
        case rmsNormEps = "rms_norm_eps"
        case maxPositionEmbeddings = "max_position_embeddings"
        case kvLoraRank = "kv_lora_rank"
        case qLoraRank = "q_lora_rank"
        case qkNopeHeadDim = "qk_nope_head_dim"
        case qkRopeHeadDim = "qk_rope_head_dim"
        case vHeadDim = "v_head_dim"
        case mlaUseNope = "mla_use_nope"
        case nRoutedExperts = "n_routed_experts"
        case nSharedExperts = "n_shared_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case moeIntermediateSize = "moe_intermediate_size"
        case firstKDenseReplace = "first_k_dense_replace"
        case scoringFunc = "scoring_func"
        case topkMethod = "topk_method"
        case routedScalingFactor = "routed_scaling_factor"
        case normTopkProb = "norm_topk_prob"
        case nGroup = "n_group"
        case topkGroup = "topk_group"
        case mhc
        case hcMult = "hc_mult"
        case hcSinkhornIters = "hc_sinkhorn_iters"
        case hcEps = "hc_eps"
        case indexHeadDim = "index_head_dim"
        case indexNHeads = "index_n_heads"
        case indexTopk = "index_topk"
        case indexKpool = "index_kpool"
        case indexKpoolCompress = "index_kpool_compress"
        case indexKpoolAlwaysSelectTail = "index_kpool_always_select_tail"
        case numNextnPredictLayers = "num_nextn_predict_layers"
        case tieWordEmbeddings = "tie_word_embeddings"
        case swigluLimit = "swiglu_limit"
        case layerTypes = "layer_types"
        case mlpLayerTypes = "mlp_layer_types"
        case linearAttnConfig = "linear_attn_config"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decode(String.self, forKey: .modelType)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        numHiddenLayers = try c.decode(Int.self, forKey: .numHiddenLayers)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        numAttentionHeads = try c.decode(Int.self, forKey: .numAttentionHeads)
        numKeyValueHeads = try c.decode(Int.self, forKey: .numKeyValueHeads)
        vocabSize = try c.decode(Int.self, forKey: .vocabSize)
        rmsNormEps = try c.decode(Float.self, forKey: .rmsNormEps)
        maxPositionEmbeddings = try c.decode(Int.self, forKey: .maxPositionEmbeddings)
        kvLoraRank = try c.decode(Int.self, forKey: .kvLoraRank)
        qLoraRank = try c.decode(Int.self, forKey: .qLoraRank)
        qkNopeHeadDim = try c.decode(Int.self, forKey: .qkNopeHeadDim)
        qkRopeHeadDim = try c.decode(Int.self, forKey: .qkRopeHeadDim)
        vHeadDim = try c.decode(Int.self, forKey: .vHeadDim)
        mlaUseNope = try c.decode(Bool.self, forKey: .mlaUseNope)
        nRoutedExperts = try c.decode(Int.self, forKey: .nRoutedExperts)
        nSharedExperts = try c.decode(Int.self, forKey: .nSharedExperts)
        numExpertsPerTok = try c.decode(Int.self, forKey: .numExpertsPerTok)
        moeIntermediateSize = try c.decode(Int.self, forKey: .moeIntermediateSize)
        firstKDenseReplace = try c.decode(Int.self, forKey: .firstKDenseReplace)
        scoringFunc = try c.decode(String.self, forKey: .scoringFunc)
        topkMethod = try c.decode(String.self, forKey: .topkMethod)
        routedScalingFactor = try c.decode(Float.self, forKey: .routedScalingFactor)
        normTopkProb = try c.decode(Bool.self, forKey: .normTopkProb)
        nGroup = try c.decodeIfPresent(Int.self, forKey: .nGroup) ?? 1
        topkGroup = try c.decodeIfPresent(Int.self, forKey: .topkGroup) ?? 1
        mhc = try c.decode(Bool.self, forKey: .mhc)
        hcMult = try c.decode(Int.self, forKey: .hcMult)
        hcSinkhornIters = try c.decode(Int.self, forKey: .hcSinkhornIters)
        hcEps = try c.decode(Float.self, forKey: .hcEps)
        indexHeadDim = try c.decode(Int.self, forKey: .indexHeadDim)
        indexNHeads = try c.decode(Int.self, forKey: .indexNHeads)
        indexTopk = try c.decode(Int.self, forKey: .indexTopk)
        indexKpool = try c.decode(Int.self, forKey: .indexKpool)
        indexKpoolCompress = try c.decode(Bool.self, forKey: .indexKpoolCompress)
        indexKpoolAlwaysSelectTail = try c.decodeIfPresent(
            Bool.self, forKey: .indexKpoolAlwaysSelectTail) ?? false
        // ABSENT MEANS ZERO. The same model is published with and without multi-token prediction,
        // and the version without simply omits the key rather than setting it to 0.
        numNextnPredictLayers = try c.decodeIfPresent(
            Int.self, forKey: .numNextnPredictLayers) ?? 0
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        swigluLimit = try c.decodeIfPresent(Float.self, forKey: .swigluLimit)
        layerTypes = try c.decode([Glm5NextLayerKind].self, forKey: .layerTypes)
        mlpLayerTypes = try c.decode([Glm5NextMLPKind].self, forKey: .mlpLayerTypes)
        linearAttnConfig = try c.decode(
            Glm5NextLinearAttentionConfiguration.self, forKey: .linearAttnConfig)
    }

    /// The layer schedule, validated. A `layer_types` that disagrees with `num_hidden_layers` is a
    /// malformed bundle, and catching it here beats an out-of-range subscript deep in a forward pass.
    public func validatedSchedule() throws -> [Glm5NextLayerKind] {
        guard layerTypes.count == numHiddenLayers else {
            throw Glm5NextConfigurationError.scheduleLengthMismatch(
                field: "layer_types", got: layerTypes.count, expected: numHiddenLayers)
        }
        guard mlpLayerTypes.count == numHiddenLayers else {
            throw Glm5NextConfigurationError.scheduleLengthMismatch(
                field: "mlp_layer_types", got: mlpLayerTypes.count, expected: numHiddenLayers)
        }
        // `linear_attn_config` states the same schedule as index lists. Where both are present they
        // must agree: a bundle where they disagree would build one attention and address another.
        if let kda = linearAttnConfig.kdaLayers {
            let declared = Set(layerTypes.enumerated().filter { $0.element == .linearAttention }
                .map(\.offset))
            guard Set(kda) == declared else {
                throw Glm5NextConfigurationError.scheduleDisagreement(
                    field: "kda_layers", listed: Set(kda).count, fromLayerTypes: declared.count)
            }
        }
        if let full = linearAttnConfig.fullAttnLayers {
            let declared = Set(layerTypes.enumerated()
                .filter { $0.element == .deepseekSparseAttention }.map(\.offset))
            guard Set(full) == declared else {
                throw Glm5NextConfigurationError.scheduleDisagreement(
                    field: "full_attn_layers", listed: Set(full).count,
                    fromLayerTypes: declared.count)
            }
        }
        return layerTypes
    }

    /// True when MLA carries no rotary split. Kept as a named property because two independent
    /// fields have to agree, and a bundle setting only one of them is a bundle to reject rather than
    /// to guess about.
    public var usesNoPositionalEncoding: Bool { mlaUseNope && qkRopeHeadDim == 0 }
}

public struct Glm5NextVisionConfiguration: Codable, Sendable {
    public let modelType: String
    public let depth: Int
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let outHiddenSize: Int
    public let numHeads: Int
    public let inChannels: Int
    public let imageSize: Int
    public let patchSize: Int
    public let spatialMergeSize: Int
    public let temporalPatchSize: Int
    public let rmsNormEps: Float
    public let projectionIntermediateSize: Int?
    public let swigluLimit: Float?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case depth
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case outHiddenSize = "out_hidden_size"
        case numHeads = "num_heads"
        case inChannels = "in_channels"
        case imageSize = "image_size"
        case patchSize = "patch_size"
        case spatialMergeSize = "spatial_merge_size"
        case temporalPatchSize = "temporal_patch_size"
        case rmsNormEps = "rms_norm_eps"
        case projectionIntermediateSize = "projection_intermediate_size"
        case swigluLimit = "swiglu_limit"
    }
}

public struct Glm5NextConfiguration: Codable, Sendable {
    public let modelType: String
    public let quantization: Glm5NextQuantization?
    public let textConfig: Glm5NextTextConfiguration
    public let visionConfig: Glm5NextVisionConfiguration?
    public let imageTokenId: Int?
    public let videoTokenId: Int?
    public let imageStartTokenId: Int?
    public let imageEndTokenId: Int?
    public let videoStartTokenId: Int?
    public let videoEndTokenId: Int?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case quantization
        case textConfig = "text_config"
        case visionConfig = "vision_config"
        case imageTokenId = "image_token_id"
        case videoTokenId = "video_token_id"
        case imageStartTokenId = "image_start_token_id"
        case imageEndTokenId = "image_end_token_id"
        case videoStartTokenId = "video_start_token_id"
        case videoEndTokenId = "video_end_token_id"
    }

    /// A vision tower is constructible only when the stanza AND the image token are both present.
    /// Either alone is a bundle that would build a tower nothing can address, or address a tower
    /// that was never built.
    public var canBuildVisionTower: Bool { visionConfig != nil && imageTokenId != nil }

    /// Video additionally needs its own token: the tower is shared, the addressing is not.
    public var canConsumeVideo: Bool { canBuildVisionTower && videoTokenId != nil }
}

public enum Glm5NextConfigurationError: Error, CustomStringConvertible {
    case scheduleLengthMismatch(field: String, got: Int, expected: Int)
    case scheduleDisagreement(field: String, listed: Int, fromLayerTypes: Int)

    public var description: String {
        switch self {
        case .scheduleLengthMismatch(let field, let got, let expected):
            return
                "glm5_next: `\(field)` lists \(got) layers but `num_hidden_layers` is \(expected); "
                + "the bundle's layer schedule is inconsistent"
        case .scheduleDisagreement(let field, let listed, let fromLayerTypes):
            return
                "glm5_next: `linear_attn_config.\(field)` names \(listed) layers but `layer_types` "
                + "implies \(fromLayerTypes); the bundle states its schedule twice and the two "
                + "statements disagree"
        }
    }
}

// MARK: - Checkpoint-key policy

/// The key rewrites this bundle needs, as a value so they can be tested without a model.
///
/// Kept separate from the model for the reason `Gemma4`'s shared policy is: the text-only route and
/// the VLM route must not be able to drift apart, and a policy that lives in one class's `sanitize`
/// inevitably does.
public enum Glm5NextCheckpointKeys {

    /// `attn_hc.hc_fn` → `attn_hc.fn`, and the same for `hc_scale` / `hc_base`.
    ///
    /// `DeepseekV4HyperConnection` declares its parameters as `fn`, `scale` and `base`; this bundle
    /// ships the identical tensors under an `hc_` prefix. Renaming here is what lets the existing
    /// module be reused rather than duplicated under new parameter names.
    public static func stripHyperConnectionPrefix(_ key: String) -> String {
        guard key.contains("_hc.hc_") else { return key }
        return key.replacingOccurrences(of: "_hc.hc_", with: "_hc.")
    }

    /// Weights for a tower that was not built. Dropping them is what makes a narrowed construction
    /// loadable at all: MLX refuses keys with no matching module.
    public static func isVisionKey(_ key: String) -> Bool {
        key.hasPrefix("model.visual.") || key.hasPrefix("visual.")
            || key.hasPrefix("model.vision_tower.") || key.hasPrefix("vision_tower.")
    }

    /// Tensors the bundle stores BARE that the corresponding module addresses as `.weight`.
    ///
    /// Neither half of this is visible from the configuration. The conv kernels ship as
    /// `…self_attn.q_conv1d` of shape [C, k] while a `Conv1d` wants `q_conv1d.weight` of shape
    /// [C, k, 1]; `o_norm` ships as a bare [128] vector while the gated RMSNorm declares `weight`.
    /// Both were found by comparing the module's parameter tree against the shipped tensor names —
    /// a config-only reading would have missed them, and the first version of that test missed them
    /// too by accepting a similar-looking key instead of the exact one.
    ///
    /// `needsChannelAxis` distinguishes the two: a depthwise conv gains a trailing 1, a norm vector
    /// does not.
    public static let bareTensorParameters: [(name: String, needsChannelAxis: Bool)] = [
        ("q_conv1d", true), ("k_conv1d", true), ("v_conv1d", true), ("o_norm", false),
    ]

    /// Bare tensors that are PARAMETERS, not module weights — so they keep their own names and must
    /// NOT be given a `.weight` suffix. The indexer's key-pool code and gate are declared with
    /// `@ParameterInfo`, which addresses them exactly as the checkpoint spells them.
    public static let bareParameterNames = [
        "index_kpool_compress_ape", "index_kpool_compress_gate",
    ]

    /// The `.weight` key the module expects, or nil when this is not one of those tensors.
    ///
    /// Matches a bare leaf as well as a dotted path, so it works on a full checkpoint key and on a
    /// leaf name alike — the earlier version required a leading dot and silently answered nil for
    /// every leaf, which made the test that uses it pass while proving nothing.
    public static func bareTensorWeightKey(_ key: String) -> String? {
        // A `@ParameterInfo` tensor is already addressed correctly; suffixing it would break it.
        for name in bareParameterNames where key == name || key.hasSuffix("." + name) { return nil }
        for (name, _) in bareTensorParameters where key == name || key.hasSuffix("." + name) {
            return key + ".weight"
        }
        return nil
    }

    private static func needsChannelAxis(_ key: String) -> Bool {
        for (name, axis) in bareTensorParameters where key == name || key.hasSuffix("." + name) {
            return axis
        }
        return false
    }

    /// Apply the policy. `keepVision` is the plan's answer, not a guess from the key set.
    public static func sanitize(
        _ weights: [String: MLXArray], keepVision: Bool
    ) -> [String: MLXArray] {
        var out = [String: MLXArray](minimumCapacity: weights.count)
        for (key, value) in weights {
            if !keepVision, isVisionKey(key) { continue }
            // Torch conv layout [O, I, …] -> MLX [O, …, I]. `patch_embed.proj` is 5-D and
            // `downsample` is 4-D; a bundle already converted is left alone, which is what the
            // trailing-dimension check decides.
            if key.hasSuffix("patch_embed.proj.weight") || key.hasSuffix("downsample.weight") {
                if value.ndim == 5, value.dim(1) != value.dim(4) || value.dim(1) < value.dim(4) {
                    out[key] = value.transposed(0, 2, 3, 4, 1)
                } else if value.ndim == 4, value.dim(1) > value.dim(3) {
                    out[key] = value.transposed(0, 2, 3, 1)
                } else {
                    out[key] = value
                }
                continue
            }
            if let weightKey = bareTensorWeightKey(key) {
                // [C, k] -> [C, k, 1] for a depthwise conv; a norm vector is left as it is.
                out[weightKey] =
                    (needsChannelAxis(key) && value.ndim == 2)
                    ? expandedDimensions(value, axis: -1) : value
                continue
            }
            out[stripHyperConnectionPrefix(key)] = value
        }
        return out
    }
}

// MARK: - Linear attention

/// GLM-5.3's linear-attention layer.
///
/// It IS KDA — Kimi Delta Attention, the same mechanism Ling 3.0 ships — so this builds
/// ``KDAAttention`` rather than carrying a second implementation. That was not the obvious guess:
/// the family is a Qwen-adjacent hybrid and Qwen3.5's `GatedDeltaNet` looked like the donor, but it
/// shares only `A_log` and `dt_bias`. The weights settle it — `f_a_proj [128,4096]`,
/// `f_b_proj [8192,128]`, the `g_*` pair, `b_proj [64,4096]`, `o_norm [128]` and three separate
/// depthwise `*_conv1d [8192,4]` are Bailing's parameter set exactly, name for name and shape for
/// shape.
///
/// Two knobs are named differently and are mapped here rather than at the call site, so the mapping
/// has one place to be wrong: `gate_lower_bound` is Ling's `kda_lower_bound`, and this family always
/// uses the low-rank f/g gates (the bundle has no unfactored `f_proj`).
public final class Glm5NextLinearAttention: Module {

    @ModuleInfo(key: "self_attn") public var attention: KDAAttention

    public init(_ config: Glm5NextTextConfiguration) {
        let linear = config.linearAttnConfig
        _attention.wrappedValue = KDAAttention(
            hiddenSize: config.hiddenSize,
            numHeads: linear.numHeads,
            headDim: linear.headDim,
            convKernelSize: linear.shortConvKernelSize,
            // Ling gates this behind `kda_safe_gate`; GLM-5.3 ships no such field and its
            // `gate_lower_bound` is set, which is the configuration that clamps.
            safeGate: true,
            lowerBound: linear.gateLowerBound,
            // No unfactored `f_proj`/`g_proj` exists in this bundle — only the low-rank pairs.
            useLoRAGates: true,
            rmsNormEps: config.rmsNormEps)
        super.init()
    }

    public func callAsFunction(
        _ x: MLXArray, mask: MLXArray? = nil, cache: MambaCache? = nil
    ) -> MLXArray {
        attention(x, mask: mask, cache: cache)
    }
}

// MARK: - Sparse attention

/// GLM-5.3's sparse-attention layer: DeepSeek MLA plus a learned key indexer.
///
/// The MLA half is classic V3 naming and needs no invention — `q_a_proj` → `q_a_layernorm` →
/// `q_b_proj`, `kv_a_proj_with_mqa` → `kv_a_layernorm` → `kv_b_proj`, then `o_proj` — the same shape
/// `GLM4MOELite` already builds for this vendor's earlier family. Note DeepSeek V4's attention is
/// NOT the donor despite the mechanism being closest to it: it spells the same projections
/// `wq_a`/`wq_b`/`wkv`, so reusing it would have meant fighting the names.
///
/// What is unusual here is the absence of RoPE. `qk_rope_head_dim` is 0 and `mla_use_nope` is true,
/// so the query and key head dimension is entirely the "nope" half and there is no rotary split to
/// apply. Both fields are checked together, because a bundle setting only one of them is a bundle to
/// reject rather than to guess about.
public final class Glm5NextSparseAttention: Module {

    public let numHeads: Int
    public let qkNopeHeadDim: Int
    public let vHeadDim: Int
    public let qHeadDim: Int
    public let scale: Float
    public let usesRoPE: Bool

    @ModuleInfo(key: "q_a_proj") public var qAProj: Linear
    @ModuleInfo(key: "q_a_layernorm") public var qALayerNorm: RMSNorm
    @ModuleInfo(key: "q_b_proj") public var qBProj: Linear
    @ModuleInfo(key: "kv_a_proj_with_mqa") public var kvAProjWithMqa: Linear
    @ModuleInfo(key: "kv_a_layernorm") public var kvALayerNorm: RMSNorm
    @ModuleInfo(key: "kv_b_proj") public var kvBProj: Linear
    @ModuleInfo(key: "o_proj") public var oProj: Linear
    @ModuleInfo(key: "indexer") public var indexer: Glm5NextIndexer

    public init(_ config: Glm5NextTextConfiguration) {
        self.numHeads = config.numAttentionHeads
        self.qkNopeHeadDim = config.qkNopeHeadDim
        self.vHeadDim = config.vHeadDim
        // With no rotary split the query/key head dim IS the nope half.
        self.qHeadDim = config.qkNopeHeadDim + config.qkRopeHeadDim
        self.scale = pow(Float(qHeadDim), -0.5)
        self.usesRoPE = !config.usesNoPositionalEncoding

        _qAProj.wrappedValue = Linear(config.hiddenSize, config.qLoraRank, bias: false)
        _qALayerNorm.wrappedValue = RMSNorm(
            dimensions: config.qLoraRank, eps: config.rmsNormEps)
        _qBProj.wrappedValue = Linear(config.qLoraRank, numHeads * qHeadDim, bias: false)

        // `_with_mqa` carries the compressed KV plus any rotary key channels. With
        // `qk_rope_head_dim == 0` there are none, so its width is the LoRA rank alone — which is
        // what the shipped [512, 4096] projection shows.
        _kvAProjWithMqa.wrappedValue = Linear(
            config.hiddenSize, config.kvLoraRank + config.qkRopeHeadDim, bias: false)
        _kvALayerNorm.wrappedValue = RMSNorm(
            dimensions: config.kvLoraRank, eps: config.rmsNormEps)
        _kvBProj.wrappedValue = Linear(
            config.kvLoraRank, numHeads * (config.qkNopeHeadDim + config.vHeadDim), bias: false)

        _oProj.wrappedValue = Linear(numHeads * config.vHeadDim, config.hiddenSize, bias: false)
        _indexer.wrappedValue = Glm5NextIndexer(config)
        super.init()
    }

    /// MLA over the keys the indexer selects.
    ///
    /// Below `index_topk` the selection is skipped, and that is an EQUIVALENCE rather than an
    /// approximation. With a sequence of `N <= index_topk`, the pool count is `ceil(N / kpool)`
    /// which is at most `index_topk / kpool = select_k`, so every pool is selected; the tokens in
    /// the incomplete trailing pool — the only ones pooling drops — are exactly what
    /// `index_kpool_always_select_tail` puts back. The resulting mask is the causal mask, so full
    /// attention computes the same thing more cheaply. `indexerAgreesWithFullAttentionBelowTopK`
    /// asserts that rather than leaving it as reasoning, and the equivalence is guarded on the tail
    /// flag, which is what makes it true.
    ///
    /// Above `index_topk` the selection runs for real.
    public func callAsFunction(
        _ x: MLXArray, mask: MLXArray? = nil, cache: KVCache? = nil
    ) throws -> MLXArray {
        let B = x.dim(0)
        let T = x.dim(1)
        let cached = cache?.offset ?? 0
        let indexed = cache as? Glm5NextIndexedKVCache
        let selecting = cached + T > indexer.topK
        if selecting, !indexer.alwaysSelectTail {
            throw Glm5NextDecoderUnavailable(
                detail:
                    "index_kpool_always_select_tail is off; tokens in the incomplete trailing pool "
                    + "would be unreachable and this file has no reference for that configuration")
        }

        let qResid = qALayerNorm(qAProj(x))
        let selected = try indexer(
            hidden: x, qResid: qResid, cache: indexed, selecting: selecting)

        var q = qBProj(qResid)
        q = q.reshaped(B, T, numHeads, qHeadDim).transposed(0, 2, 1, 3)

        // No rotary split, so `_with_mqa` carries the compressed KV alone.
        let compressed = kvALayerNorm(kvAProjWithMqa(x))
        var kv = kvBProj(compressed)
        kv = kv.reshaped(B, T, numHeads, qkNopeHeadDim + vHeadDim).transposed(0, 2, 1, 3)
        var keys = kv[.ellipsis, ..<qkNopeHeadDim]
        var values = kv[.ellipsis, qkNopeHeadDim...]

        if let cache {
            (keys, values) = cache.update(keys: keys, values: values)
        }

        // The sparse mask REPLACES the causal one rather than joining it: the indexer has already
        // applied causality (a pool is a candidate only if its last token is visible to the query),
        // so anding them again would be redundant, and passing the causal mask instead of the
        // sparse one would silently un-sparsify the layer.
        var effectiveMask = mask
        if let selected {
            let visible = indexer.maskFromIndices(selected, kvLength: keys.dim(2))
            effectiveMask = MLX.where(
                visible, MLXArray(Float(0)).asType(q.dtype),
                MLXArray(-Float.greatestFiniteMagnitude).asType(q.dtype))
        }
        let out = MLXFast.scaledDotProductAttention(
            queries: q, keys: keys, values: values, scale: scale, mask: effectiveMask)
        return oProj(out.transposed(0, 2, 1, 3).reshaped(B, T, numHeads * vHeadDim))
    }
}

/// The learned sparse-key indexer: which keys each query attends to, out of a 1M-token context.
///
/// `wq_b` and `weights_proj` are the two `DeepseekV4Indexer` also has. The rest is this family's
/// own: a single `wk` of [128, 4096] where `index_head_dim` is 128, so there is ONE key head rather
/// than `index_n_heads` of them — the shape leaves no other reading, though whether it behaves as
/// MQA in the forward pass is not something the shape can settle. It also carries a `k_norm` with
/// BOTH weight and bias, unusual here where most norms are weight-only, and the key-pool
/// compression.
///
/// KEY POOLING IS THE PART WITH NO PRECEDENT IN THIS REPO, AND ITS SEMANTICS ARE NOT ESTABLISHED
/// HERE. What follows is read off the checkpoint, not from a paper or a reference implementation:
///
///   * `index_kpool` is 4, `index_kpool_compress` is true, `index_kpool_always_select_tail` is true;
///   * `index_kpool_compress_gate` is [128, 4096] — a projection from the hidden size to the index
///     head dim, by shape;
///   * `index_kpool_compress_ape` is [4, 128] — one vector per pool slot, by shape.
///
/// Those SHAPES are facts. Calling `_ape` a positional code over the slots, or `_gate` a per-pool
/// gate, is a guess consistent with the names and nothing more. The declarations below are a
/// parameter SURFACE verified against the bundle; they are not a claim to know what the forward pass
/// computes. Anyone implementing it should read GLM's reference implementation first — this file
/// will let the weights load, and will not tell them what to do with them.
public final class Glm5NextIndexer: Module {

    public let numHeads: Int
    public let headDim: Int
    public let topK: Int
    public let poolSize: Int
    public let poolCompressed: Bool
    public let alwaysSelectTail: Bool

    @ModuleInfo(key: "wq_b") public var wqB: Linear
    @ModuleInfo(key: "wk") public var wk: Linear
    @ModuleInfo(key: "k_norm") public var kNorm: LayerNorm
    @ModuleInfo(key: "weights_proj") public var weightsProj: Linear

    /// Present only when `index_kpool_compress` is set.
    @ParameterInfo(key: "index_kpool_compress_ape") public var poolPositionalCode: MLXArray?
    @ParameterInfo(key: "index_kpool_compress_gate") public var poolGate: MLXArray?

    public convenience init(_ config: Glm5NextTextConfiguration) {
        self.init(
            numHeads: config.indexNHeads, headDim: config.indexHeadDim,
            topK: config.indexTopk, poolSize: config.indexKpool,
            poolCompressed: config.indexKpoolCompress,
            alwaysSelectTail: config.indexKpoolAlwaysSelectTail,
            hiddenSize: config.hiddenSize, qLoraRank: config.qLoraRank)
    }

    /// The designated initialiser, in explicit quantities rather than a configuration.
    ///
    /// The selection math is the part of this family with no precedent in the repo, so it has to be
    /// constructible at test sizes — with an `index_topk` small enough that selection actually
    /// discards something. Driving it through a whole `Glm5NextTextConfiguration` would mean a
    /// 45-layer JSON fixture to exercise four tensors.
    public init(
        numHeads: Int, headDim: Int, topK: Int, poolSize: Int, poolCompressed: Bool,
        alwaysSelectTail: Bool, hiddenSize: Int, qLoraRank: Int
    ) {
        self.numHeads = numHeads
        self.headDim = headDim
        self.topK = topK
        self.poolSize = poolSize
        self.poolCompressed = poolCompressed
        self.alwaysSelectTail = alwaysSelectTail

        // Queries come off the SAME low-rank trunk the attention uses, which is why this is `wq_b`
        // with no `wq_a`: it reads `q_a_proj`'s output, not the hidden state.
        _wqB.wrappedValue = Linear(qLoraRank, numHeads * headDim, bias: false)
        // One shared key head over the hidden state.
        _wk.wrappedValue = Linear(hiddenSize, headDim, bias: false)
        // Weight AND bias — the shipped bundle carries both.
        _kNorm.wrappedValue = LayerNorm(dimensions: headDim)
        // One scalar per index head.
        _weightsProj.wrappedValue = Linear(hiddenSize, numHeads, bias: false)

        if poolCompressed {
            _poolPositionalCode.wrappedValue = MLXArray.zeros([poolSize, headDim])
            _poolGate.wrappedValue = MLXArray.zeros([headDim, hiddenSize])
        }
        super.init()
    }

    /// Score this step's tokens, remember them, and return the indices each query may attend to.
    ///
    /// The cache append happens on EVERY call, including the ones where selection is skipped
    /// because the sequence still fits inside `index_topk`. Skipping it there would leave holes in
    /// the packed history, and the first selection past the threshold would score pools built from
    /// whatever happened to be adjacent.
    public func callAsFunction(
        hidden: MLXArray, qResid: MLXArray, cache: Glm5NextIndexedKVCache?, selecting: Bool
    ) throws -> MLXArray? {
        let B = hidden.dim(0)
        let S = hidden.dim(1)
        guard let gate = poolGate else {
            throw Glm5NextDecoderUnavailable(
                detail: "index_kpool_compress is off; this indexer has no compression gate")
        }
        let queryOffset = cache?.offset ?? 0
        let k = kNorm(wk(hidden))
        let gateScores = matmul(hidden, gate.transposed()).asType(k.dtype)
        let validChannel = MLXArray.ones([B, S, 1], dtype: k.dtype)
        let packedNew = concatenated([k, gateScores, validChannel], axis: -1)
        let packed = cache?.updateIndexer(packedNew) ?? packedNew

        guard selecting else { return nil }
        let q = wqB(qResid).reshaped(B, S, numHeads, headDim)
        return try selectTopK(
            packed: packed, queries: q, hidden: hidden, queryOffset: queryOffset)
    }
}

// MARK: - Indexer selection

/// The indexer's own state cache.
///
/// The indexer scores POOLS OF PAST KEYS, so at decode step `t` it needs the index key and the
/// compression gate score for every position seen so far — quantities derived from hidden states
/// that no longer exist by then. They have to be cached, and they have to be trimmed and copied in
/// lockstep with the KV they index into: a packed buffer one row shorter than the KV would shift
/// every selected index by one, silently, and attend to the wrong tokens.
/// The indexer's own state cache.
///
/// The indexer scores POOLS OF PAST KEYS, so at decode step `t` it needs the index key and the
/// compression gate score for every position seen so far — quantities derived from hidden states
/// that no longer exist by then. They have to be cached, and they have to be trimmed and copied in
/// lockstep with the KV they index into: a packed buffer one row shorter than the KV would shift
/// every selected index by one, silently, and attend to the wrong tokens.
///
/// Composed around `KVCacheSimple` rather than derived from it, because that class is `public` and
/// not `open` — it cannot be subclassed from this module. Composition also keeps the forwarding
/// explicit, which is what makes `trim` and `copy` obviously cover BOTH buffers.
public final class Glm5NextIndexedKVCache: KVCache {

    private let kv = KVCacheSimple()

    /// `(B, N, 2 * indexHeadDim + 1)` — the index key, the gate score, and a validity flag.
    ///
    /// The validity channel is always 1 for the producer in this file, which never pads. It is kept
    /// anyway because the reference's pooling reads it, and a pooler that cannot express "this slot
    /// is padding" would have to be rewritten rather than fed differently if padding ever arrives.
    public private(set) var indexerPacked: MLXArray?

    public init() {}

    /// Append this step's rows and return the whole history.
    public func updateIndexer(_ new: MLXArray) -> MLXArray {
        if let existing = indexerPacked {
            indexerPacked = concatenated([existing, new], axis: 1)
        } else {
            indexerPacked = new
        }
        return indexerPacked!
    }

    public var offset: Int { kv.offset }
    public var maxSize: Int? { kv.maxSize }

    public func innerState() -> [MLXArray] {
        kv.innerState() + [indexerPacked].compactMap { $0 }
    }

    public func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        kv.update(keys: keys, values: values)
    }

    public var state: [MLXArray] {
        get { kv.state + [indexerPacked].compactMap { $0 } }
        set {
            // The packed buffer is the ONLY optional trailing entry, so its presence is decided by
            // the count rather than by position — restoring it into the KV slots would corrupt both.
            let kvCount = kv.state.count
            let kvPortion = Array(newValue.prefix(kvCount))
            // Same empty guard as `copy()`: the inner setter traps on a count
            // that is not exactly 2, and an empty restore (fresh snapshot)
            // yields no KV arrays.
            if !kvPortion.isEmpty {
                kv.state = kvPortion
            }
            indexerPacked = newValue.count > kvCount ? newValue[kvCount] : nil
        }
    }

    public var metaState: [String] {
        get { kv.metaState }
        set { kv.metaState = newValue }
    }

    public var isTrimmable: Bool { kv.isTrimmable }

    @discardableResult
    public func trim(_ n: Int) -> Int {
        let trimmed = kv.trim(n)
        if trimmed > 0, let packed = indexerPacked {
            let keep = max(0, packed.dim(1) - trimmed)
            indexerPacked = keep > 0 ? packed[0..., ..<keep, 0...] : nil
        }
        return trimmed
    }

    public func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        kv.makeMask(n: n, windowSize: windowSize, returnArray: returnArray)
    }

    public func copy() -> any KVCache {
        let copy = Glm5NextIndexedKVCache()
        // The inner KVCacheSimple's state setter requires EXACTLY [keys,
        // values] and traps on any other count. A fresh cache (no tokens yet)
        // yields an EMPTY state, so restore it only when present — the same
        // guard `KVCacheSimple.copy()` and `DeepseekV4Compressor` already use.
        // Without it, snapshotting a fresh cache at the prompt boundary (the
        // very first generation) round-tripped `[]` through the setter and
        // trapped, crashing the app before any token was produced.
        let innerState = kv.state
        if !innerState.isEmpty {
            copy.kv.state = innerState.map(Self.owned)
        }
        copy.kv.metaState = kv.metaState
        copy.kv.offset = kv.offset
        copy.indexerPacked = indexerPacked.map(Self.owned)
        return copy
    }

    /// `MLXLMCommon.ownedStateCopy` is internal, so the idiom is repeated rather than imported.
    ///
    /// `* 1` and not `x[.ellipsis]`: a slice SHARES the source buffer, so a retained copy would keep
    /// the live cache multiply-referenced and block MLX's buffer donation on the per-token in-place
    /// update — the difference between a pointer bump and a full-capacity copy on every decode step.
    private static func owned(_ x: MLXArray) -> MLXArray { x * 1 }
}

extension Glm5NextIndexer {

    /// Rebuild the compressed pool candidates from the packed state.
    ///
    /// Returns the pooled keys `(B, P, D)`, the raw token indices each pool stands for `(P, K)`, and
    /// whether each pool is selectable `(B, P)`.
    ///
    /// The layout is `(P, K)` and not the reference's per-batch `(B, P, K)` because the guard below
    /// establishes that every sequence starts at the same slot, which makes the per-batch copies
    /// identical. The reference additionally masks individual entries of an INVALID pool to -1;
    /// that is subsumed here, because an invalid pool never becomes a candidate and its whole group
    /// is set to -1 by `selectedValid` during expansion. Returning one shared layout keeps the
    /// expansion a plain gather instead of a batched one.
    ///
    /// Two details in the reference are easy to lose and both change the result. Pools are laid out
    /// from the FIRST VALID token, not from slot 0, so that left-padding does not shift every pool
    /// boundary. And a pool is valid only if ALL `K` of its members are — an incomplete trailing
    /// pool is never selectable, which is precisely why `index_kpool_always_select_tail` exists to
    /// put those tokens back.
    public func pooledStates(
        packed: MLXArray
    ) throws -> (keys: MLXArray, indices: MLXArray, valid: MLXArray) {
        let B = packed.dim(0)
        let N = packed.dim(1)
        let keys = packed[.ellipsis, ..<headDim]
        let gate = packed[.ellipsis, headDim ..< (2 * headDim)]
        let valid = packed[.ellipsis, 2 * headDim] .!= MLXArray(Float(0))

        // Pool layout is shared across the batch only when every sequence starts at the same slot.
        // Without padding that is trivially true; with it, per-sequence gathers would be required,
        // and producing one shared layout anyway would quietly mis-pool the padded rows.
        let anyValid = valid.any(axis: -1)
        let firstKey = MLX.where(anyValid, valid.asType(.int32).argMax(axis: -1), MLXArray(Int32(N)))
        let firstValues = firstKey.asType(.int32).asArray(Int32.self)
        guard let start = firstValues.first, firstValues.allSatisfy({ $0 == start }) else {
            throw Glm5NextDecoderUnavailable(
                detail:
                    "the sequences in this batch begin at different offsets (padding); the indexer's "
                    + "pool layout is shared across the batch and cannot express that")
        }

        let poolCount = (N + poolSize - 1) / poolSize
        let offsets = MLXArray(Int32(0) ..< Int32(poolCount * poolSize))
            .reshaped(poolCount, poolSize)
        let poolIndices = offsets + MLXArray(start)
        let safe = clip(poolIndices, min: MLXArray(Int32(0)), max: MLXArray(Int32(N - 1)))
        let flatSafe = safe.reshaped(-1)

        let groupedKeys = take(keys, flatSafe, axis: 1)
            .reshaped(B, poolCount, poolSize, headDim)
        let groupedGate = take(gate, flatSafe, axis: 1)
            .reshaped(B, poolCount, poolSize, headDim)
        var groupedValid = take(valid, flatSafe, axis: 1)
            .reshaped(B, poolCount, poolSize)
        groupedValid = groupedValid .&& (poolIndices .< MLXArray(Int32(N)))

        let poolValid = groupedValid.all(axis: -1)

        guard let ape = poolPositionalCode else {
            throw Glm5NextDecoderUnavailable(
                detail: "index_kpool_compress is off; pooled selection needs the compression code")
        }
        // A LEARNED weighted average within each pool: softmax over the POOL-MEMBER axis, per
        // feature channel — not one scalar weight per member.
        //
        // The gathers above span the ENTIRE packed index history and this
        // runs every decode step, so fp32 casts here materialized two
        // full-context fp32 tensors per token (the MLA-cast tax class).
        // The softmax reduces over the tiny pool-member axis only, and the
        // pooled keys feed SELECTION scores, so native-dtype math changes
        // at most near-tie pool choices. `VMLX_GLM5_INDEX_FP32=1` restores
        // the fp32 pipeline for A/B.
        let poolMathDtype: DType =
            Glm5NextIndexerRuntime.poolFP32 ? .float32 : packed.dtype
        var logits = groupedGate.asType(poolMathDtype)
            + ape.asType(poolMathDtype)[.newAxis, .newAxis]
        logits = MLX.where(
            groupedValid.expandedDimensions(axis: -1), logits,
            MLXArray(-Float.greatestFiniteMagnitude).asType(poolMathDtype))
        var probabilities = softmax(logits, axis: 2)
        // A fully invalid pool softmaxes -inf against -inf and yields NaN; it is masked out of the
        // selection anyway, but a NaN here would poison the pooled key and, through it, the scores
        // of every VALID pool in the same matmul.
        probabilities = MLX.where(
            isNaN(probabilities), MLXArray(Float(0)).asType(poolMathDtype), probabilities)
        let pooled = (probabilities * groupedKeys.asType(poolMathDtype)).sum(axis: 2)

        return (pooled.asType(packed.dtype), poolIndices, poolValid)
    }

    /// Select the pools each query attends to, and expand them back into raw token indices.
    ///
    /// The result is `(B, S, width)` of Int32 where -1 means "nothing selected here"; `width` is
    /// `index_topk`, plus `index_kpool - 1` when the incomplete tail is always appended.
    public func selectTopK(
        packed: MLXArray, queries: MLXArray, hidden: MLXArray, queryOffset: Int
    ) throws -> MLXArray {
        let B = packed.dim(0)
        let N = packed.dim(1)
        let S = hidden.dim(1)

        let (poolKeys, poolIndices, poolValid) = try pooledStates(packed: packed)
        let poolCount = poolIndices.dim(0)

        // (B, S, H, D) @ (B, 1, D, P) -> (B, S, H, P)
        // Native-dtype matmul (fp32 accumulation happens inside the GEMM);
        // only the small score tensor is carried in fp32 for the weighting
        // math below. Casting poolKeys to fp32 copied the whole pool per
        // decode step. Gated with the pooling math above.
        var scores: MLXArray
        if Glm5NextIndexerRuntime.poolFP32 {
            scores = matmul(
                queries.asType(.float32),
                poolKeys.asType(.float32).swappedAxes(-1, -2).expandedDimensions(axis: 1))
        } else {
            scores = matmul(
                queries.asType(poolKeys.dtype),
                poolKeys.swappedAxes(-1, -2).expandedDimensions(axis: 1)
            ).asType(.float32)
        }
        scores = maximum(scores * (1.0 / sqrt(Float(headDim))), MLXArray(Float(0)))

        // One learned scalar per index head, then a weighted sum ACROSS heads.
        let weights =
            weightsProj(hidden).asType(.float32) * (1.0 / sqrt(Float(numHeads)))
        var indexScores = matmul(weights.expandedDimensions(axis: -2), scores)
            .squeezed(axis: -2)

        // A pool is a candidate only if its LAST token is visible to this query — which is what
        // makes the selection causal without materialising a per-token comparison.
        let kvPositions = MLXArray(Int32(0) ..< Int32(N))
        let queryPositions = MLXArray(Int32(queryOffset) ..< Int32(queryOffset + S))
        let causal = kvPositions[.newAxis, 0...] .<= queryPositions[0..., .newAxis]  // (S, N)
        let validKeys = packed[.ellipsis, 2 * headDim] .!= MLXArray(Float(0))  // (B, N)
        let visible = causal[.newAxis] .&& validKeys[0..., .newAxis, 0...]  // (B, S, N)

        // The final token of EVERY pool — the last column of the (P, K) layout, not the last pool.
        // `[0..., -1]` selects the last ROW; `[.ellipsis, k]` selects the right values but returns
        // them as (1, P) rather than (P,), and that spurious axis propagates through the `take`
        // below into every downstream shape. Reshaped explicitly rather than relying on the
        // subscript to drop it.
        let poolEnd = clip(
            poolIndices[.ellipsis, poolSize - 1].reshaped(-1),
            min: MLXArray(Int32(0)), max: MLXArray(Int32(N - 1)))
        let poolVisible = take(visible, poolEnd, axis: -1)  // (B, S, P)
        let candidates = poolVisible .&& poolValid.expandedDimensions(axis: 1)
        indexScores = MLX.where(
            candidates, indexScores, MLXArray(-Float.greatestFiniteMagnitude))

        let selectCount = min(topK / poolSize, poolCount)
        // `argPartition` gives the k largest without ordering them; order is irrelevant here
        // because the result is scattered into a boolean mask.
        let ordered = argSort(-indexScores, axis: -1)
        let selected = take(
            ordered, MLXArray(Int32(0) ..< Int32(selectCount)), axis: -1)  // (B, S, selectCount)

        let selectedValid = takeAlong(candidates, selected, axis: -1)
        // (P, K) gathered by (B, S, selectCount) -> (B, S, selectCount, K)
        let selectedIndices = take(poolIndices, selected, axis: 0)
        precondition(
            selectedIndices.dim(-1) == poolSize,
            "pool layout must be (P, K); a batch axis here would gather along the wrong dimension")
        var expanded = selectedIndices.reshaped(B, S, selectCount * poolSize)
        let keepExpanded = repeated(
            selectedValid.expandedDimensions(axis: -1), count: poolSize, axis: -1)
            .reshaped(B, S, selectCount * poolSize)
        expanded = MLX.where(keepExpanded, expanded, MLXArray(Int32(-1)))

        var width = topK
        if alwaysSelectTail, poolSize > 1 {
            expanded = concatenated(
                [expanded, tailIndices(visible: visible, validKeys: validKeys, kvLength: N)],
                axis: -1)
            width += poolSize - 1
        }

        if expanded.dim(-1) < width {
            let padding = MLXArray.full(
                [B, S, width - expanded.dim(-1)], values: MLXArray(Int32(-1)))
            expanded = concatenated([expanded, padding], axis: -1)
        }
        return expanded[.ellipsis, ..<width]
    }

    /// The tokens after the last COMPLETE pool, which pooling alone would drop.
    private func tailIndices(
        visible: MLXArray, validKeys: MLXArray, kvLength: Int
    ) -> MLXArray {
        let B = visible.dim(0)
        let S = visible.dim(1)
        let maxWidth = poolSize - 1

        let anyValid = validKeys.any(axis: -1)
        let firstKey = MLX.where(
            anyValid, validKeys.asType(.int32).argMax(axis: -1), MLXArray(Int32(kvLength)))
        let visibleCount = visible.asType(.int32).sum(axis: -1)  // (B, S)
        let tailCount = visibleCount % MLXArray(Int32(poolSize))
        let offsets = MLXArray(Int32(0) ..< Int32(maxWidth))

        let tailStart = firstKey.expandedDimensions(axis: -1) + visibleCount - tailCount
        var indices = tailStart.expandedDimensions(axis: -1) + offsets  // (B, S, maxWidth)

        let withinTail = offsets[.newAxis, .newAxis, 0...] .< tailCount.expandedDimensions(axis: -1)
        let withinCache = indices .< MLXArray(Int32(kvLength))
        let safe = clip(indices, min: MLXArray(Int32(0)), max: MLXArray(Int32(kvLength - 1)))
        let tailVisible = takeAlong(visible, safe, axis: -1)

        indices = MLX.where(
            withinTail .&& withinCache .&& tailVisible, indices, MLXArray(Int32(-1)))
        _ = B
        _ = S
        return indices
    }

    /// Turn selected indices into the boolean attention mask, `(B, 1, S, N)`.
    ///
    /// The scatter target carries ONE EXTRA COLUMN as a bin for the -1 entries. Clamping them to 0
    /// instead — the obvious move, and what the reference's clamp does before it multiplies by a
    /// validity flag the scatter here cannot express — would mark position 0 visible to every query,
    /// which is exactly the kind of error that produces fluent, confidently wrong text.
    public func maskFromIndices(_ indices: MLXArray, kvLength: Int) -> MLXArray {
        let B = indices.dim(0)
        let S = indices.dim(1)
        let valid = (indices .>= MLXArray(Int32(0))) .&& (indices .< MLXArray(Int32(kvLength)))
        let target = MLX.where(valid, indices, MLXArray(Int32(kvLength)))
        var counts = MLXArray.zeros([B, S, kvLength + 1], dtype: .int32)
        counts = putAlong(counts, target, values: MLXArray.ones(target.shape, dtype: .int32), axis: -1)
        return (counts[.ellipsis, ..<kvLength] .!= MLXArray(Int32(0))).expandedDimensions(axis: 1)
    }
}


// MARK: - Vision tower

/// One transformer block of GLM-5.3's vision tower.
///
/// Pre-norm with RMSNorm, a fused `qkv` carrying bias, per-head-dim q/k norms, and a BIASED SwiGLU
/// MLP. The biases are worth naming: this repo's language MLPs are overwhelmingly bias-free, and a
/// `Linear(bias: false)` here would silently fail to address `mlp.gate_proj.bias`.
public final class Glm5NextVisionBlock: Module {

    @ModuleInfo(key: "norm1") public var norm1: RMSNorm
    @ModuleInfo(key: "norm2") public var norm2: RMSNorm
    @ModuleInfo(key: "attn") public var attention: Glm5NextVisionAttention
    @ModuleInfo(key: "mlp") public var mlp: Glm5NextVisionMLP

    public init(_ config: Glm5NextVisionConfiguration) {
        _norm1.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _norm2.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _attention.wrappedValue = Glm5NextVisionAttention(config)
        _mlp.wrappedValue = Glm5NextVisionMLP(config)
        super.init()
    }

    /// Ordinary pre-norm residuals — the tower has no hyper-connections; those are the decoder's.
    public func callAsFunction(
        _ x: MLXArray, mask: MLXArray? = nil,
        positionEmbeddings: (cos: MLXArray, sin: MLXArray)? = nil
    ) -> MLXArray {
        var h = x + attention(norm1(x), mask: mask, positionEmbeddings: positionEmbeddings)
        h = h + mlp(norm2(h))
        return h
    }
}

/// The vision tower's 2-D rotary embedding, over (h, w) patch coordinates.
///
/// THIS WAS MISSING ENTIRELY, and nothing structural could have caught it: a rotary embedding has
/// no learned parameters, so a checkpoint-versus-parameter-tree comparison — which this file has,
/// in both directions — is blind to its absence. The tower loaded every shipped tensor, ran, and
/// produced features with no idea where anything was on the page. It described pictures the way you
/// would describe a bag of shuffled tiles.
///
/// `Glm5NextVisionRotaryEmbedding(head_dim // 2)` in the reference: each of the two axes contributes
/// `dim / 2` frequencies, so the concatenated `(h, w)` block is `dim` wide and doubling it for the
/// half-rotation gives exactly `head_dim`.
public final class Glm5NextVisionRotary {

    public let dimensions: Int
    private let inverseFrequencies: MLXArray

    public init(dimensions: Int, theta: Float = 10000.0) {
        self.dimensions = dimensions
        let exponents = MLXArray(stride(from: 0, to: dimensions, by: 2).map { Float($0) })
            / Float(dimensions)
        self.inverseFrequencies = 1.0 / MLX.pow(MLXArray(theta), exponents)
    }

    /// `positions` is `(N, 2)` of (h, w) indices; returns cos and sin of shape `(N, 2 * dimensions)`.
    public func callAsFunction(_ positions: MLXArray) -> (cos: MLXArray, sin: MLXArray) {
        let frequencies = positions.asType(.float32).expandedDimensions(axis: -1)
            * inverseFrequencies
        let flattened = frequencies.reshaped(positions.dim(0), -1)
        let embedding = concatenated([flattened, flattened], axis: -1)
        return (MLX.cos(embedding), MLX.sin(embedding))
    }

    /// The (h, w) index of every patch, in the BLOCK-MAJOR order the patchifier emits.
    ///
    /// Not raster order. `QwenVL.patchify` groups each `merge x merge` neighbourhood into
    /// consecutive sequence positions, and the reference's position ids reshape to
    /// `(h/m, m, w/m, m)` and transpose the middle axes to match. Numbering the patches row by row
    /// instead would give every one of them the wrong coordinate.
    public static func positionIds(grid: THW, mergeSize: Int) -> MLXArray {
        let h = grid.h, w = grid.w, m = mergeSize
        let rows = broadcast(
            MLXArray(Int32(0) ..< Int32(h)).reshaped(h, 1), to: [h, w])
        let columns = broadcast(
            MLXArray(Int32(0) ..< Int32(w)).reshaped(1, w), to: [h, w])
        func blockMajor(_ x: MLXArray) -> MLXArray {
            x.reshaped(h / m, m, w / m, m).transposed(0, 2, 1, 3).reshaped(-1)
        }
        let single = stacked([blockMajor(rows), blockMajor(columns)], axis: -1)
        return grid.t == 1 ? single : tiled(single, repetitions: [grid.t, 1])
    }
}

/// Fused-QKV self-attention with per-head-dim query and key norms.
public final class Glm5NextVisionAttention: Module {

    public let numHeads: Int
    public let headDim: Int
    public let scale: Float

    @ModuleInfo(key: "qkv") public var qkv: Linear
    @ModuleInfo(key: "q_norm") public var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") public var kNorm: RMSNorm
    @ModuleInfo(key: "proj") public var proj: Linear

    public init(_ config: Glm5NextVisionConfiguration) {
        self.numHeads = config.numHeads
        self.headDim = config.hiddenSize / config.numHeads
        self.scale = pow(Float(headDim), -0.5)
        _qkv.wrappedValue = Linear(config.hiddenSize, 3 * config.hiddenSize, bias: true)
        // On HEAD DIM (64), not hidden size — the shipped [64] vectors say so.
        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        _proj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: true)
        super.init()
    }

    public func callAsFunction(
        _ x: MLXArray, mask: MLXArray? = nil,
        positionEmbeddings: (cos: MLXArray, sin: MLXArray)? = nil
    ) -> MLXArray {
        let B = x.dim(0), T = x.dim(1)
        let fused = qkv(x).reshaped(B, T, 3, numHeads, headDim).transposed(2, 0, 3, 1, 4)
        // Norms are per HEAD DIM and applied AFTER the split, which is why they are [64].
        var q = qNorm(fused[0])
        var k = kNorm(fused[1])
        let v = fused[2]

        // Rotary AFTER the q/k norms, as in the reference — normalising a rotated vector is not the
        // same operation and the order is not a detail.
        if let positionEmbeddings {
            func rotateHalf(_ x: MLXArray) -> MLXArray {
                let half = x.dim(-1) / 2
                let first = x[.ellipsis, ..<half]
                let second = x[.ellipsis, half...]
                return concatenated([-second, first], axis: -1)
            }
            // cos/sin are (T, headDim); q/k are (B, heads, T, headDim).
            let cosine = positionEmbeddings.cos[.newAxis, .newAxis, 0..., 0...]
            let sine = positionEmbeddings.sin[.newAxis, .newAxis, 0..., 0...]
            let qF = q.asType(.float32), kF = k.asType(.float32)
            q = (qF * cosine + rotateHalf(qF) * sine).asType(q.dtype)
            k = (kF * cosine + rotateHalf(kF) * sine).asType(k.dtype)
        }
        let out = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: mask)
        return proj(out.transposed(0, 2, 1, 3).reshaped(B, T, numHeads * headDim))
    }
}

/// Biased SwiGLU.
public final class Glm5NextVisionMLP: Module {

    @ModuleInfo(key: "gate_proj") public var gate: Linear
    @ModuleInfo(key: "up_proj") public var up: Linear
    @ModuleInfo(key: "down_proj") public var down: Linear
    public let swigluLimit: Float?

    public init(_ config: Glm5NextVisionConfiguration) {
        self.swigluLimit = config.swigluLimit
        _gate.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: true)
        _up.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: true)
        _down.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: true)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(Glm5NextActivation.clampedSwiGLU(gate: gate(x), up: up(x), limit: swigluLimit))
    }
}

/// Projects merged patches into the language model's width.
///
/// Same five parts as `Glm4v`'s `PatchMerger`, which is the family this descends from: a projection,
/// a LayerNorm (weight AND bias, unlike the RMSNorms elsewhere in the tower), then a SwiGLU whose
/// intermediate is `projection_intermediate_size` rather than the tower's own.
public final class Glm5NextPatchMerger: Module {

    @ModuleInfo(key: "proj") public var proj: Linear
    @ModuleInfo(key: "post_projection_norm") public var postProjectionNorm: LayerNorm
    @ModuleInfo(key: "gate_proj") public var gate: Linear
    @ModuleInfo(key: "up_proj") public var up: Linear
    @ModuleInfo(key: "down_proj") public var down: Linear

    public let swigluLimit: Float?

    public init(_ config: Glm5NextVisionConfiguration) {
        let out = config.outHiddenSize
        let intermediate = config.projectionIntermediateSize ?? out
        self.swigluLimit = config.swigluLimit
        // `downsample` has already widened to `out`, so this maps out -> out.
        _proj.wrappedValue = Linear(out, out, bias: false)
        _postProjectionNorm.wrappedValue = LayerNorm(dimensions: out)
        _gate.wrappedValue = Linear(out, intermediate, bias: false)
        _up.wrappedValue = Linear(out, intermediate, bias: false)
        _down.wrappedValue = Linear(intermediate, out, bias: false)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // GELU after the norm — `act1` in the reference. It was missing entirely, so the merger
        // handed the language model a LINEAR projection of the tower's output where a non-linear
        // one was trained. Nothing about that fails: the shapes are right, the values are finite,
        // and the model still describes an image — just not the one in front of it.
        let h = geluApproximate(postProjectionNorm(proj(x)))
        return down(
            Glm5NextActivation.clampedSwiGLU(gate: gate(h), up: up(h), limit: swigluLimit))
    }
}

/// GLM-5.3's vision tower.
///
/// Structurally `Glm4v`'s, MINUS two modules it does not ship: there is no `embeddings` and no
/// `post_conv_layernorm` in the checkpoint, so declaring either would leave a parameter nothing can
/// supply. Verified against the tensor names rather than inherited from the donor.
public final class Glm5NextVisionTower: Module {

    public let spatialMergeSize: Int
    public let temporalPatchSize: Int
    /// Not a `@ModuleInfo`: it holds no parameters, so binding it as a module would add an empty
    /// node to the tree that the checkpoint comparison would then have to special-case.
    public private(set) var rotary: Glm5NextVisionRotary!

    @ModuleInfo(key: "patch_embed") public var patchEmbed: Glm5NextPatchEmbed
    @ModuleInfo(key: "blocks") public var blocks: [Glm5NextVisionBlock]
    @ModuleInfo(key: "post_layernorm") public var postLayerNorm: RMSNorm
    @ModuleInfo(key: "downsample") public var downsample: Conv2d
    @ModuleInfo(key: "merger") public var merger: Glm5NextPatchMerger

    public init(_ config: Glm5NextVisionConfiguration) {
        self.spatialMergeSize = config.spatialMergeSize
        self.temporalPatchSize = config.temporalPatchSize
        _patchEmbed.wrappedValue = Glm5NextPatchEmbed(config)
        _blocks.wrappedValue = (0 ..< config.depth).map { _ in Glm5NextVisionBlock(config) }
        _postLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        // Merges a spatialMergeSize × spatialMergeSize patch neighbourhood and widens to the
        // language model's hidden size in one convolution — the shipped [4096, 1024, 2, 2].
        _downsample.wrappedValue = Conv2d(
            inputChannels: config.hiddenSize, outputChannels: config.outHiddenSize,
            kernelSize: .init(config.spatialMergeSize), stride: .init(config.spatialMergeSize))
        _merger.wrappedValue = Glm5NextPatchMerger(config)
        // Parameter-free, so it appears in no checkpoint and in no parameter tree.
        self.rotary = Glm5NextVisionRotary(
            dimensions: (config.hiddenSize / config.numHeads) / 2)
        super.init()
    }

    /// Patches in, language-width embeddings out.
    ///
    /// `downsample` is a CONVOLUTION over the spatial grid, so the flat patch sequence has to be
    /// folded back to (H, W) before it and flattened after — the merge is spatial, not a reshape of
    /// adjacent sequence positions.
    public func callAsFunction(_ patches: MLXArray, grid: THW) throws -> MLXArray {
        // The patch embed emits a flat (N, C) sequence; the blocks want the (B, T, H) that every
        // attention in this file is written against. One batch axis for the whole stack, removed
        // before the spatial fold below — cheaper and clearer than teaching each block to guess.
        guard grid.h % spatialMergeSize == 0, grid.w % spatialMergeSize == 0 else {
            throw Glm5NextInputShapeError(
                got: [grid.h, grid.w],
                expected: "a grid divisible by spatial_merge_size \(spatialMergeSize)")
        }

        let positions = Glm5NextVisionRotary.positionIds(grid: grid, mergeSize: spatialMergeSize)
        let positionEmbeddings = rotary(positions)

        var h = patchEmbed(patches).expandedDimensions(axis: 0)
        for block in blocks { h = block(h, positionEmbeddings: positionEmbeddings) }
        h = postLayerNorm(h)
        h = h.squeezed(axis: 0)

        // THE SPATIAL FOLD IS OVER CONSECUTIVE SEQUENCE POSITIONS, NOT OVER AN (H, W) GRID.
        //
        // `QwenVL.patchify` emits patches BLOCK-MAJOR: each `merge x merge` neighbourhood already
        // occupies `merge^2` consecutive rows. So the merge is `view(-1, m, m, C)` and a conv whose
        // kernel covers the whole tile — which is what the reference does. Reshaping to
        // `(t, h, w, C)` and striding a conv across it, as this did, reads the sequence as if it
        // were raster-ordered and therefore convolves patches that are not neighbours at all. The
        // output has the right shape and the wrong content, which is why it survived every shape
        // test and showed up only as a model describing the wrong picture.
        let channels = h.dim(-1)
        h = h.reshaped(-1, spatialMergeSize, spatialMergeSize, channels)
        h = downsample(h)
        return merger(h.reshaped(-1, h.dim(-1)))
    }
}

/// Conv3d patchifier: `temporal_patch_size` frames by `patch_size` square.
public final class Glm5NextPatchEmbed: Module {

    @ModuleInfo(key: "proj") public var proj: Conv3d

    public let inChannels: Int
    public let temporalPatchSize: Int
    public let patchSize: Int
    public let embedDim: Int

    public init(_ config: Glm5NextVisionConfiguration) {
        self.inChannels = config.inChannels
        self.temporalPatchSize = config.temporalPatchSize
        self.patchSize = config.patchSize
        self.embedDim = config.hiddenSize
        _proj.wrappedValue = Conv3d(
            inputChannels: config.inChannels,
            outputChannels: config.hiddenSize,
            kernelSize: IntOrTriple((config.temporalPatchSize, config.patchSize, config.patchSize)),
            stride: IntOrTriple((config.temporalPatchSize, config.patchSize, config.patchSize)),
            bias: true)
        super.init()
    }

    /// Flat patches in, embeddings out. The conv is 3-D because a patch spans
    /// `temporal_patch_size` frames; a still image is fed as that many identical frames, which is
    /// what the processor's temporal padding produces.
    ///
    /// The processor hands over one FLAT row per patch — `C * T * patch * patch` values in the
    /// order the reference patchifier wrote them. The conv wants that row unfolded, and it wants
    /// channels LAST, as every MLX conv does. Both steps happen here rather than in the caller so
    /// that the tower's input contract stays "whatever the processor emits".
    public func callAsFunction(_ patches: MLXArray) -> MLXArray {
        let unfolded = patches
            .reshaped(-1, inChannels, temporalPatchSize, patchSize, patchSize)
            .movedAxis(source: 1, destination: 4)
        return proj(unfolded).reshaped(-1, embedDim)
    }
}

// MARK: - Feed-forward

/// The dense MLP of the first `first_k_dense_replace` layers.
///
/// Bias-free, unlike the vision tower's — the shipped tensors carry no `bias` entry, only the
/// quantized triple.
public final class Glm5NextDenseMLP: Module {

    @ModuleInfo(key: "gate_proj") public var gate: Linear
    @ModuleInfo(key: "up_proj") public var up: Linear
    @ModuleInfo(key: "down_proj") public var down: Linear

    public let swigluLimit: Float?

    public init(_ config: Glm5NextTextConfiguration) {
        self.swigluLimit = config.swigluLimit
        _gate.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _up.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _down.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(Glm5NextActivation.clampedSwiGLU(gate: gate(x), up: up(x), limit: swigluLimit))
    }
}

/// `swiglu_limit` is 10.0 in this bundle, so the activation is CLAMPED. One place, because the
/// dense MLP, the shared expert, the routed experts and the vision MLP all use it, and a clamp
/// applied in three of four would be wrong only on the tail — silently.
public enum Glm5NextActivation {
    /// `down(silu(clamp(gate, max: limit)) * clamp(up, -limit ... limit))`.
    ///
    /// Three things about the clamp, all of which this had wrong. It applies to the RAW
    /// projections, BEFORE the activation — clamping `silu(gate)` instead caps a different quantity
    /// at a different place. The GATE clamp is upper-bound ONLY (`min=None` in the reference), not
    /// symmetric, so it must not touch the negative tail SiLU depends on. And `up` is clamped too,
    /// symmetrically; leaving it unclamped, as this did, lets an outlier through the multiply that
    /// the reference would have bounded.
    public static func clampedSwiGLU(gate g: MLXArray, up u: MLXArray, limit: Float?) -> MLXArray {
        guard let limit else { return silu(g) * u }
        let gateClamped = minimum(g, MLXArray(limit))
        let upClamped = clip(u, min: -limit, max: limit)
        return silu(gateClamped) * upClamped
    }
}

/// The router. `noaux_tc` over a sigmoid score, with a learned per-expert correction bias.
///
/// The bias is applied to the scores used for CHOOSING, not to the scores used for weighting — that
/// is what "aux-loss-free" means here, and getting it the wrong way round changes which experts run
/// while leaving every shape intact. `DeepseekV3`'s `MoEGate` already states the same contract.
public final class Glm5NextMoEGate: Module {

    public let topK: Int
    public let normTopkProb: Bool
    public let routedScalingFactor: Float
    public let numGroups: Int
    public let topkGroup: Int

    public var weight: MLXArray

    public init(_ config: Glm5NextTextConfiguration) {
        self.topK = config.numExpertsPerTok
        self.normTopkProb = config.normTopkProb
        self.routedScalingFactor = config.routedScalingFactor
        self.numGroups = config.nGroup
        self.topkGroup = config.topkGroup
        self.weight = MLXArray.zeros([config.nRoutedExperts, config.hiddenSize])
        super.init()
    }

    /// Returns the chosen expert indices and their weights.
    ///
    /// `noaux_tc`: the correction bias shifts the scores used for CHOOSING, and the weights come
    /// from the UNSHIFTED scores. Applying it to both — the natural mistake — changes which experts
    /// run AND how much they count, while every shape stays valid.
    public func callAsFunction(
        _ x: MLXArray, correctionBias: MLXArray
    ) -> (indices: MLXArray, weights: MLXArray) {
        let logits = x.matmul(weight.T)
        let originalScores = sigmoid(logits.asType(.float32))
        let scoresForChoice = originalScores + correctionBias

        let indices = argPartition(-scoresForChoice, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
        var scores = takeAlong(originalScores, indices, axis: -1)
        if topK > 1, normTopkProb {
            let denominator = scores.sum(axis: -1, keepDims: true)
                + MLXArray(1e-20, dtype: scores.dtype)
            scores = scores / denominator
        }
        // Applied whether or not the weights were normalised — it scales the routed branch against
        // the always-on shared expert, which is a separate concern from normalisation.
        scores = scores * routedScalingFactor
        return (indices, scores.asType(x.dtype))
    }
}

/// The sparse MLP: 288 routed experts fused into a `SwitchGLU`, plus one always-on shared expert.
public final class Glm5NextMoE: Module {

    /// Sibling of `gate`, NOT a child of it — `mlp.e_score_correction_bias`, where DeepSeek V3 puts
    /// the same tensor at `mlp.gate.e_score_correction_bias`. Copying V3's nesting produced a
    /// parameter the checkpoint could not supply, which is what the shipped-tensor test reported.
    @ParameterInfo(key: "e_score_correction_bias") public var eScoreCorrectionBias: MLXArray

    @ModuleInfo(key: "gate") public var gate: Glm5NextMoEGate
    @ModuleInfo(key: "switch_mlp") public var switchMLP: SwitchGLU
    @ModuleInfo(key: "shared_experts") public var sharedExperts: Glm5NextSharedExpert

    public init(_ config: Glm5NextTextConfiguration) {
        _eScoreCorrectionBias.wrappedValue = MLXArray.zeros([config.nRoutedExperts])
        _gate.wrappedValue = Glm5NextMoEGate(config)
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.moeIntermediateSize,
            numExperts: config.nRoutedExperts,
            // `swiglu_limit` is 10.0 in this bundle, so the activation is CLAMPED. Plain silu would
            // be wrong on the tail, silently — nothing about the shapes would object.
            activation: { clip(silu($0), min: -config.swigluLimit!, max: config.swigluLimit!) })
        _sharedExperts.wrappedValue = Glm5NextSharedExpert(config)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (indices, weights) = gate(x, correctionBias: eScoreCorrectionBias)
        let routed = switchMLP(x, indices)
        // `switchMLP` returns [..., topK, hidden]; weight each expert's contribution and sum.
        let combined = (routed * expandedDimensions(weights, axis: -1)).sum(axis: -2)
        // The shared expert is ALWAYS on — it is not one of the routed `topK`.
        return combined + sharedExperts(x)
    }
}

/// The shared expert, at `moe_intermediate_size` — `n_shared_experts` is 1 in the shipped bundle,
/// and the checkpoint stores it as a single unfused MLP rather than a one-entry stack.
public final class Glm5NextSharedExpert: Module {

    @ModuleInfo(key: "gate_proj") public var gate: Linear
    @ModuleInfo(key: "up_proj") public var up: Linear
    @ModuleInfo(key: "down_proj") public var down: Linear

    public let swigluLimit: Float?

    public init(_ config: Glm5NextTextConfiguration) {
        self.swigluLimit = config.swigluLimit
        let width = config.moeIntermediateSize * config.nSharedExperts
        _gate.wrappedValue = Linear(config.hiddenSize, width, bias: false)
        _up.wrappedValue = Linear(config.hiddenSize, width, bias: false)
        _down.wrappedValue = Linear(width, config.hiddenSize, bias: false)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(Glm5NextActivation.clampedSwiGLU(gate: gate(x), up: up(x), limit: swigluLimit))
    }
}

// MARK: - Decoder layer

/// One decoder layer, of whichever kind the schedule names.
///
/// Both the attention and the MLP vary per layer, independently: layer 3 is the first sparse
/// ATTENTION layer while layers 0-2 are the dense MLPs, so the two schedules do not line up and are
/// read separately. Exactly one attention module and one MLP is non-nil.
///
/// `attn_hc` and `ffn_hc` are hyper-connections — `DeepseekV4HyperConnection`, the same mechanism and
/// the same three parameters, which is why `sanitize` strips the bundle's `hc_` prefix rather than a
/// second module being written.
public final class Glm5NextDecoderLayer: Module {

    public let kind: Glm5NextLayerKind
    public let mlpKind: Glm5NextMLPKind
    /// True for the trailing multi-token-prediction layer, which the bundle stores as one more
    /// entry in `model.layers` rather than under a head of its own.
    public let isMultiTokenPrediction: Bool

    @ModuleInfo(key: "input_layernorm") public var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") public var postAttentionLayerNorm: RMSNorm

    /// ONE property per checkpoint key, typed as `Module`.
    ///
    /// Two `@ModuleInfo` properties sharing a key — a `KDAAttention?` and a
    /// `Glm5NextSparseAttention?` both keyed `self_attn`, whichever is nil — does not give the tree
    /// a choice; it gives it a collision, and the nil one wins. Loading real weights failed with
    /// `incompatibleItems(path: ["self_attn"], item: "none")`, which no amount of building the
    /// module or comparing parameter NAMES had revealed: the names were right, the tree was not.
    @ModuleInfo(key: "self_attn") public var attention: Module
    @ModuleInfo(key: "mlp") public var feedForward: Module

    /// Typed views. The kind is fixed at construction, so these are casts, not searches.
    public var linearAttention: KDAAttention? { attention as? KDAAttention }
    public var sparseAttention: Glm5NextSparseAttention? { attention as? Glm5NextSparseAttention }
    public var denseMLP: Glm5NextDenseMLP? { feedForward as? Glm5NextDenseMLP }
    public var moe: Glm5NextMoE? { feedForward as? Glm5NextMoE }

    /// Present ONLY on the MTP layer. All four are optional so that a bundle built without
    /// multi-token prediction — the same model is published both ways — declares no parameter its
    /// checkpoint cannot supply.
    @ModuleInfo(key: "enorm") public var embeddingNorm: RMSNorm?
    @ModuleInfo(key: "hnorm") public var hiddenNorm: RMSNorm?
    @ModuleInfo(key: "eh_proj") public var ehProj: Linear?
    @ModuleInfo(key: "shared_head") public var sharedHead: Glm5NextSharedHead?

    /// Hyper-connections replace the plain residual. Present when `mhc` is set, which it is in the
    /// shipped bundle; a variant without them falls back to `x + block(norm(x))`.
    @ModuleInfo(key: "attn_hc") public var attentionHC: DeepseekV4HyperConnection?
    @ModuleInfo(key: "ffn_hc") public var ffnHC: DeepseekV4HyperConnection?

    public init(
        _ config: Glm5NextTextConfiguration, kind: Glm5NextLayerKind, mlpKind: Glm5NextMLPKind,
        isMultiTokenPrediction: Bool = false
    ) {
        self.kind = kind
        self.mlpKind = mlpKind
        self.isMultiTokenPrediction = isMultiTokenPrediction
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)

        switch kind {
        case .linearAttention:
            let linear = config.linearAttnConfig
            _attention.wrappedValue = KDAAttention(
                hiddenSize: config.hiddenSize, numHeads: linear.numHeads,
                headDim: linear.headDim, convKernelSize: linear.shortConvKernelSize,
                safeGate: true, lowerBound: linear.gateLowerBound,
                useLoRAGates: true, rmsNormEps: config.rmsNormEps)
        case .deepseekSparseAttention:
            _attention.wrappedValue = Glm5NextSparseAttention(config)
        }

        switch mlpKind {
        case .dense: _feedForward.wrappedValue = Glm5NextDenseMLP(config)
        case .sparse: _feedForward.wrappedValue = Glm5NextMoE(config)
        }

        // NOT on the MTP layer. The shipped layer 45 carries no `attn_hc` / `ffn_hc` at all — it
        // runs plain residuals while the backbone runs hyper-connections. Building them here
        // unconditionally declared six parameters the checkpoint cannot supply, which the
        // whole-bundle test named.
        if config.mhc, !isMultiTokenPrediction {
            // `hcEps` for the Sinkhorn arithmetic, `rmsNormEps` for the input norm — two DIFFERENT
            // config fields, as the reference has them. GLM ships 1e-6 and 1e-5; DeepSeek V4 ships
            // 1e-6 for both, which is why borrowing its module without this distinction went
            // unnoticed until the oracle sized the residual.
            _attentionHC.wrappedValue = DeepseekV4HyperConnection(
                hcMult: config.hcMult, sinkhornIterations: config.hcSinkhornIters,
                eps: config.hcEps, normEps: config.rmsNormEps, hiddenSize: config.hiddenSize)
            _ffnHC.wrappedValue = DeepseekV4HyperConnection(
                hcMult: config.hcMult, sinkhornIterations: config.hcSinkhornIters,
                eps: config.hcEps, normEps: config.rmsNormEps, hiddenSize: config.hiddenSize)
        }

        if isMultiTokenPrediction {
            _embeddingNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
            _hiddenNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
            // Fuses the NORMED embedding with the NORMED hidden state, so its input is twice the
            // hidden size — which the shipped [4096, 8192] projection confirms.
            _ehProj.wrappedValue = Linear(
                2 * config.hiddenSize, config.hiddenSize, bias: false)
            _sharedHead.wrappedValue = Glm5NextSharedHead(config)
        }
        super.init()
    }

    /// One layer: attention then feed-forward, each wrapped by its hyper-connection.
    ///
    /// The HC replaces the residual rather than sitting beside it — `collapse` produces the block's
    /// input plus the state `expand` needs to fold the output back, so the pair must bracket the
    /// block exactly. `DeepseekV4Attention` establishes the same order.
    /// One cache slot per layer, whatever kind this layer needs.
    ///
    /// `MambaCache` conforms to `KVCache`, so the model's `[KVCache]` carries both kinds and each
    /// layer casts to the one it uses — the arrangement `Qwen35` already uses for the same hybrid.
    public func callAsFunction(
        _ x: MLXArray, mask: MLXArray? = nil, cache: KVCache? = nil
    ) throws -> MLXArray {
        let linearCache = cache as? MambaCache
        // ---- attention ----
        var h: MLXArray
        if let attentionHC {
            let residual = x
            let (collapsed, post, comb) = attentionHC.collapse(x)
            let out = try runAttention(inputLayerNorm(collapsed), mask: mask, cache: cache,
                                       linearCache: linearCache)
            h = attentionHC.expand(blockOut: out, residual: residual, post: post, comb: comb)
        } else {
            h = x + (try runAttention(inputLayerNorm(x), mask: mask, cache: cache,
                                      linearCache: linearCache))
        }

        // ---- feed-forward ----
        if let ffnHC {
            let residual = h
            let (collapsed, post, comb) = ffnHC.collapse(h)
            let out = runFeedForward(postAttentionLayerNorm(collapsed))
            h = ffnHC.expand(blockOut: out, residual: residual, post: post, comb: comb)
        } else {
            h = h + runFeedForward(postAttentionLayerNorm(h))
        }
        return h
    }

    private func runAttention(
        _ x: MLXArray, mask: MLXArray?, cache: KVCache?, linearCache: MambaCache?
    ) throws -> MLXArray {
        // A linear layer must NOT be handed a KV cache and vice versa; the cast above is what
        // decides, and a nil here means the caller built the wrong cache kind for this layer.
        if let linearAttention {
            return linearAttention(x, mask: mask, cache: linearCache)
        }
        guard let sparseAttention else {
            throw Glm5NextDecoderUnavailable(detail: "layer has neither attention module")
        }
        return try sparseAttention(x, mask: mask, cache: cache)
    }

    private func runFeedForward(_ x: MLXArray) -> MLXArray {
        if let moe { return moe(x) }
        if let denseMLP { return denseMLP(x) }
        return x
    }
}

/// The MTP head's output norm. It is `shared_head` because the head SHARES the backbone's
/// `lm_head`; only the norm before it is its own.
public final class Glm5NextSharedHead: Module {

    @ModuleInfo(key: "norm") public var norm: RMSNorm

    public init(_ config: Glm5NextTextConfiguration) {
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }
}

// MARK: - LanguageModel conformance

extension Glm5Next: LanguageModel, VisionLanguageModelProtocol, VLMModel {

    public var vocabularySize: Int { config.textConfig.vocabSize }

    public var loraLayers: [Module] {
        // The DECODER layers only. The MTP layer is not part of ordinary generation, so adapting it
        // would train weights no ordinary forward consults.
        Array(languageModel.layers.prefix(languageModel.numDecoderLayers))
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        languageModel.makeCache(maxKVSize: parameters?.maxKVSize)
    }

    public func prepare(
        _ input: LMInput, cache: [KVCache], windowSize: Int?
    ) throws -> PrepareResult {
        let imagePixels = input.image?.pixels
        let videoPixels = input.video?.pixels
        guard imagePixels != nil || videoPixels != nil else {
            return .tokens(input.text)
        }

        guard let tower = visionTower else {
            // Media arrived for a model built WITHOUT the tower — a narrowed construction. Answering
            // from the text would look exactly like having seen it.
            throw Glm5NextDecoderUnavailable(
                detail:
                    "media was supplied but this instance was built without a vision tower; "
                    + "construct it requesting .vision")
        }

        // The tower runs ONCE, here, on the whole prompt — not per generated token.
        let dtype = tower.patchEmbed.proj.weight.dtype
        // ONE CALL PER GRID. A still image is a single grid, but a video is one grid PER TEMPORAL
        // PATCH, and the tower's spatial fold is written against a single (t, h, w). Taking
        // `frames.first` and handing it the whole concatenated pixel block — which is what this did
        // — reshaped three frames' rows into one frame's shape and trapped.
        func encode(_ pixels: MLXArray?, _ frames: [THW]?) throws -> MLXArray? {
            guard let pixels, let grids = frames, !grids.isEmpty else { return nil }
            var offset = 0
            var encoded = [MLXArray]()
            for grid in grids {
                let rows = grid.product
                guard offset + rows <= pixels.dim(0) else {
                    throw Glm5NextInputShapeError(
                        got: [pixels.dim(0), offset + rows],
                        expected: "enough patch rows for every grid; the processor's pixel block and "
                            + "its grid list disagree")
                }
                encoded.append(
                    try tower(pixels[offset ..< (offset + rows), 0...].asType(dtype), grid: grid))
                offset += rows
            }
            return encoded.count == 1 ? encoded[0] : concatenated(encoded, axis: 0)
        }
        let imageFeatures = try encode(imagePixels, input.image?.frames)
        let videoFeatures = try encode(videoPixels, input.video?.frames)

        let embeddings = languageModel.embedTokens(input.text.tokens)
        let spliced = try spliceMediaFeatures(
            inputIds: input.text.tokens.reshaped(-1), embeddings: embeddings,
            imageFeatures: imageFeatures, videoFeatures: videoFeatures)

        let hidden = try languageModel(
            input.text.tokens, mask: nil, caches: cache, inputEmbedding: spliced)
        let logits = lmHead.map { $0(hidden) } ?? languageModel.embedTokens.asLinear(hidden)
        return .logits(LMOutput(logits: logits))
    }

    public func callAsFunction(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        // The protocol's forward cannot throw, and this model's genuinely can — a sequence past
        // `index_topk`, or a malformed shape. Reporting through `LMOutput` is not possible either,
        // so the failure is surfaced as a zero-logit output ONLY after being written to stderr,
        // rather than being silently swallowed.
        do {
            let logits = try callAsFunction(
                input.tokens, mask: nil, caches: cache, inputEmbedding: nil)
            return LMOutput(logits: logits)
        } catch {
            FileHandle.standardError.write(
                Data("[glm5_next] forward failed: \(error)\n".utf8))
            return LMOutput(
                logits: MLXArray.zeros(
                    [input.tokens.dim(0), input.tokens.dim(1), vocabularySize]))
        }
    }
}

// MARK: - Input processing

/// Turns a `UserInput` into the tokens and pixels the model consumes.
///
/// NOT WIRED TO THE ROUTED FACTORY YET — see the note on `Glm5Next`. It exists so the image path has
/// one owner, and so the token-budget arithmetic is testable without a 96 GB model.
/// `processor_config.json` as the bundle actually ships it.
///
/// The file is NOT a flat image-processor config: it nests `image_processor` and `video_processor`,
/// and the two differ in ways that matter. `video_processor` carries its own `fps` (2) and a
/// `max_image_tokens` of 240,000 against the image side's 8,000 — a whole-clip budget rather than a
/// per-frame one. Decoding the top level as an image config fails outright on `image_mean`, which is
/// how this was found; decoding only the image half and using it for video would have succeeded and
/// quietly downscaled every frame.
public struct Glm5NextProcessorConfiguration: Codable, Sendable {
    public let imageProcessor: Glm5NextImageProcessorConfiguration
    public let videoProcessor: Glm5NextImageProcessorConfiguration?
    public let videoFPS: Double?

    enum CodingKeys: String, CodingKey {
        case imageProcessor = "image_processor"
        case videoProcessor = "video_processor"
    }
    private enum VideoKeys: String, CodingKey { case fps }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.imageProcessor = try c.decode(
            Glm5NextImageProcessorConfiguration.self, forKey: .imageProcessor)
        self.videoProcessor = try c.decodeIfPresent(
            Glm5NextImageProcessorConfiguration.self, forKey: .videoProcessor)
        // `fps` lives only on the video half and has no image counterpart, so it is read separately
        // rather than added to the shared shape as an always-nil field.
        if let video = try? c.nestedContainer(keyedBy: VideoKeys.self, forKey: .videoProcessor) {
            self.videoFPS = try video.decodeIfPresent(Double.self, forKey: .fps)
        } else {
            self.videoFPS = nil
        }
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(imageProcessor, forKey: .imageProcessor)
        try c.encodeIfPresent(videoProcessor, forKey: .videoProcessor)
    }
}

public final class Glm5NextProcessor: UserInputProcessor {

    private let config: Glm5NextImageProcessorConfiguration
    /// The VIDEO half of `processor_config.json`, which carries its own token budget. Defaults to
    /// the image half when a bundle ships only one, so a single-section config still works.
    private let videoConfig: Glm5NextImageProcessorConfiguration
    /// `video_processor.fps` — 2 in every shipped GLM-5.3 bundle, but read rather than assumed.
    private let videoFPS: Double
    private let tokenizer: any Tokenizer
    private let imageToken: Int
    private let imageStartToken: Int?
    private let imageEndToken: Int?
    private let videoToken: Int?
    /// `nonisolated(unsafe)`: a value type holding only `Int`s and `[Float]`, but `Glm5NextImageProcessor`
    /// is not declared `Sendable` and marking it so would be a claim about a type this file does not
    /// own. Constructed once in `init` and never mutated.
    private nonisolated(unsafe) let processor: Glm5NextImageProcessor

    /// The initialiser the processor REGISTRY calls, which hands over only the processor config and
    /// the tokenizer.
    ///
    /// The bundle declares `processor_class: "Glm5NextProcessor"`, and registering the model without
    /// registering this left the family loading its 95 GB of weights and then failing with
    /// "Unsupported processor type" — a table filled in on one side only.
    ///
    /// The media ids live in the MODEL config, which is not passed here, so they are resolved from
    /// the vocabulary instead. `convertTokenToId` rather than `encode(...).last`, because encode
    /// may prepend a BOS and would then hand back the wrong id with no sign anything was wrong.
    public convenience init(
        config: Glm5NextProcessorConfiguration, tokenizer: any Tokenizer
    ) throws {
        try self.init(
            imageConfig: config.imageProcessor,
            videoConfig: config.videoProcessor ?? config.imageProcessor,
            videoFPS: config.videoFPS ?? 2.0, tokenizer: tokenizer)
    }

    public convenience init(
        imageConfig: Glm5NextImageProcessorConfiguration,
        videoConfig: Glm5NextImageProcessorConfiguration,
        videoFPS: Double, tokenizer: any Tokenizer
    ) throws {
        guard let imageToken = tokenizer.convertTokenToId("<|image|>") else {
            throw Glm5NextDecoderUnavailable(
                detail: "this tokenizer has no <|image|> token, so image placeholders cannot be "
                    + "written; the bundle's tokenizer.json is not GLM-5.3's")
        }
        self.init(
            config: imageConfig, tokenizer: tokenizer, imageToken: imageToken,
            imageStartToken: tokenizer.convertTokenToId("<|begin_of_image|>"),
            imageEndToken: tokenizer.convertTokenToId("<|end_of_image|>"),
            videoConfig: videoConfig, videoFPS: videoFPS)
    }

    public init(
        config: Glm5NextImageProcessorConfiguration, tokenizer: any Tokenizer, imageToken: Int,
        imageStartToken: Int? = nil, imageEndToken: Int? = nil,
        videoConfig: Glm5NextImageProcessorConfiguration? = nil, videoFPS: Double = 2.0
    ) {
        self.config = config
        self.videoConfig = videoConfig ?? config
        self.videoFPS = videoFPS
        self.tokenizer = tokenizer
        self.imageToken = imageToken
        self.imageStartToken = imageStartToken
        self.imageEndToken = imageEndToken
        self.videoToken = tokenizer.convertTokenToId("<|video|>")
        self.processor = Glm5NextImageProcessor(config)
    }

    /// The patch tensor and grid for one image, plus how many placeholders it needs.
    ///
    /// Returned together on purpose: the placeholder count and the tower's output length have to
    /// agree, and deriving them in two places is how they come to disagree.
    public func preprocess(image: CIImage) throws -> (
        patches: MLXArray, grid: THW, tokenCount: Int
    ) {
        let extent = image.extent
        let (height, width) = try processor.targetSize(
            height: Int(extent.height), width: Int(extent.width))

        var processed = MediaProcessing.inSRGBToneCurveSpace(image)
        processed = MediaProcessing.resampleBicubic(processed, to: .init(width: width, height: height))
        processed = MediaProcessing.normalize(
            processed, mean: config.imageMean.asCGFloat3, std: config.imageStd.asCGFloat3)
        let array = MediaProcessing.asMLXArray(processed)

        // A still image is fed as `temporal_patch_size` identical frames, because the patch embed is
        // a 3-D convolution spanning that many.
        let frames = Array(repeating: array, count: config.temporalPatchSize)
        let (patches, grid) = try QwenVL.patchify(
            images: frames, mergeSize: config.mergeSize, patchSize: config.patchSize,
            temporalPatchSize: config.temporalPatchSize)
        return (patches, grid, processor.tokenCount(height: height, width: width))
    }

    /// Sample a video into ONE GRID PER TEMPORAL PATCH, the way GLM-5.3's own processor does.
    ///
    /// GLM-5.3 has no video pathway in the model: `Glm5NextProcessor.replace_video_token` in
    /// transformers rewrites the video placeholder into a RUN OF IMAGE BLOCKS, one per temporal
    /// patch, each followed by that frame's timestamp. So a video reaches the language model as a
    /// sequence of images, and the splice below never sees a video token at all.
    ///
    /// Its `replace_video_token` is byte-identical to GLM-4V's. `replace_frame_token_id` is NOT:
    /// GLM-4V writes a bare integer second, GLM-5.3 writes `"%.1f seconds"`. Copying GLM-4V's
    /// wholesale would build a prompt that differs from the reference in exactly one token run,
    /// which no shape check would ever catch.
    ///
    /// Sampling is 2 fps, its video processor's `fps` default; the frame ceiling is its
    /// `max_frames` (2048), and unlike GLM-4V there is no `max_duration` cap (it is 0).
    /// GLM-5.3's VIDEO canvas: the largest aligned frame size whose WHOLE-CLIP pixel budget fits.
    ///
    /// A port of `smart_resize` in transformers `video_processing_glm5_next.py`. The distinction
    /// from the image path is the one that matters here: the image processor sizes ONE picture
    /// against `max_image_tokens` (8,000 in the shipped bundle), while this sizes EVERY FRAME
    /// against a budget shared across the clip (240,000). Sizing frames with the image budget — as
    /// this file did first — is not merely a different number: it ignores the frame count entirely,
    /// so a two-second clip and a two-minute one would get identically sized frames and the long
    /// one would blow far past what the model was trained to receive.
    ///
    /// The search is the reference's: binary-search the content height, align both axes up to
    /// `factor`, and keep the largest candidate whose `frames x H x W` fits.
    static func videoCanvas(
        frameCount: Int, height: Int, width: Int,
        temporalFactor: Int, factor: Int, minTokens: Int, maxTokens: Int
    ) -> (height: Int, width: Int) {
        let pixelsPerToken = temporalFactor * factor * factor
        let minPixels = minTokens * pixelsPerToken
        let maxPixels = maxTokens * pixelsPerToken
        func align(_ value: Int) -> Int { Int((Double(value) / Double(factor)).rounded(.up)) * factor }

        let alignedFrames = max(
            temporalFactor,
            Int((Double(frameCount) / Double(temporalFactor)).rounded()) * temporalFactor)

        var alignedHeight = align(height)
        var alignedWidth = align(width)
        var budget = alignedFrames * alignedHeight * alignedWidth

        if budget < minPixels, frameCount > 0, height > 0, width > 0 {
            let scale = (Double(minPixels) / Double(frameCount * height * width)).squareRoot()
            alignedHeight = align(max(1, Int((Double(height) * scale).rounded(.up))))
            alignedWidth = align(max(1, Int((Double(width) * scale).rounded(.up))))
            budget = alignedFrames * alignedHeight * alignedWidth
        }
        guard budget > maxPixels else { return (alignedHeight, alignedWidth) }

        var low = 1
        var high = height
        var best = (height: factor, width: factor)
        while low <= high {
            let contentHeight = (low + high) / 2
            let contentWidth = max(1, Int((Double(width) * Double(contentHeight) / Double(height))
                .rounded(.down)))
            let candidateHeight = align(contentHeight)
            let candidateWidth = align(contentWidth)
            if alignedFrames * candidateHeight * candidateWidth <= maxPixels {
                best = (candidateHeight, candidateWidth)
                low = contentHeight + 1
            } else {
                high = contentHeight - 1
            }
        }
        return best
    }

    /// The text that follows each video frame's image block.
    ///
    /// GLM-5.3 writes `"2.5 seconds"`; GLM-4V writes the bare integer `"2"`. The two families share
    /// a byte-identical `replace_video_token` and differ ONLY here, which is precisely the kind of
    /// difference that survives every shape check and every smoke test — the prompt is well-formed
    /// either way, just not the one the weights were trained against.
    /// `Glm5NextVideoTimestampTests` pins both formats against each other.
    public static func frameTimestampMarkup(_ seconds: Double) -> String {
        String(format: "%.1f seconds", seconds)
    }

    private func preprocessVideo(
        _ video: UserInput.Video
    ) async throws -> (pixels: MLXArray, grids: [THW], seconds: [Double]) {
        // The canvas depends on how many frames there will BE, so it cannot be decided from the
        // first frame the way an image's can. Duration and natural size come from the asset, which
        // is metadata rather than decode, so this costs nothing and keeps sampling to one pass.
        let asset = video.asAVAssetForSizing()
        let duration = try await CMTimeGetSeconds(asset.load(.duration))
        let track = try await asset.loadTracks(withMediaType: .video).first
        let natural = try await track?.load(.naturalSize) ?? .zero
        let sampled = min(2048, max(1, Int((duration * videoFPS).rounded(.down))))
        let canvas = Self.videoCanvas(
            frameCount: sampled,
            height: Int(natural.height), width: Int(natural.width),
            temporalFactor: videoConfig.temporalPatchSize,
            factor: videoConfig.patchSize * videoConfig.mergeSize,
            minTokens: videoConfig.minImageTokens, maxTokens: videoConfig.maxImageTokens)
        let target = CGSize(width: canvas.width, height: canvas.height)

        let sequence = try await MediaProcessing.asProcessedSequence(
            video, targetFPS: { [videoFPS] _ in videoFPS }, maxFrames: 2048
        ) { frame in
            var processed = MediaProcessing.inSRGBToneCurveSpace(frame.frame)
            processed = MediaProcessing.resampleBicubic(processed, to: target)
            processed = MediaProcessing.normalize(
                processed, mean: self.videoConfig.imageMean.asCGFloat3,
                std: self.videoConfig.imageStd.asCGFloat3)
            return VideoFrame(frame: processed, timeStamp: frame.timeStamp)
        }
        guard !sequence.frames.isEmpty else {
            throw Glm5NextDecoderUnavailable(detail: "the video decoded to no frames at 2 fps")
        }

        // Each temporal patch is patchified on its OWN so every one yields a `t == 1` grid — the
        // per-frame form the prompt's image blocks correspond to. Patchifying the whole clip at once
        // would give a single `t == frames` grid and one enormous image block instead.
        let step = config.temporalPatchSize
        var chunks = [MLXArray]()
        var grids = [THW]()
        var seconds = [Double]()
        for start in stride(from: 0, to: sequence.frames.count, by: step) {
            let slice = Array(
                sequence.frames[start ..< min(start + step, sequence.frames.count)])
            let (patches, grid) = try QwenVL.patchify(
                images: slice, mergeSize: videoConfig.mergeSize,
                patchSize: videoConfig.patchSize,
                temporalPatchSize: videoConfig.temporalPatchSize)
            chunks.append(patches)
            grids.append(grid)
            // The reference takes `metadata.timestamps[::2]` — the first raw frame of each patch.
            seconds.append(
                CMTimeGetSeconds(sequence.timestamps[min(start, sequence.timestamps.count - 1)]))
        }
        return (concatenated(chunks), grids, seconds)
    }

    /// Replace one media placeholder with the run of tokens its grids actually need.
    ///
    /// The chat template renders a SINGLE marker per attachment — `<|begin_of_image|><|image|>`
    /// `<|end_of_image|>` — and the real token count depends on the grid, so the expansion has to
    /// happen after templating and before the model sees anything.
    /// Replace the single media TOKEN the template rendered with the run the grids need.
    ///
    /// Located by token ID, not by re-encoding the marker string. `encode` of a marker can prepend a
    /// BOS or split differently from how the template's own output tokenised, and then the needle is
    /// simply never found — which is exactly what happened, reported as "the chat template rendered
    /// no marker" when the template had rendered it perfectly well. One id cannot be mis-tokenised.
    ///
    /// The template's surrounding `<|begin_of_image|>` / `<|end_of_image|>` markers are LEFT ALONE:
    /// they are ordinary tokens with their own embeddings, not placeholders, and the reference
    /// replaces only the inner token too.
    private func expandPlaceholder(
        in tokens: [Int], token: Int, name: String, replacement: [Int]
    ) throws -> [Int] {
        guard let position = tokens.firstIndex(of: token) else {
            throw Glm5NextDecoderUnavailable(
                detail: "the chat template rendered no \(name) token (id \(token)), so the "
                    + "attachment has nowhere to go")
        }
        var out = Array(tokens[tokens.startIndex ..< position])
        out.append(contentsOf: replacement)
        out.append(contentsOf: tokens[(position + 1)...])
        return out
    }

    /// One video frame's block: the image markers, its placeholders, and its timestamp.
    private func frameBlock(tokenCount: Int, seconds: Double) -> [Int] {
        var block = [Int]()
        if let start = imageStartToken { block.append(start) }
        block.append(contentsOf: Array(repeating: imageToken, count: tokenCount))
        if let end = imageEndToken { block.append(end) }
        block.append(contentsOf: tokenizer.encode(text: Self.frameTimestampMarkup(seconds)))
        return block
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        // THROUGH THE CHAT TEMPLATE. This previously raw-encoded the prompt text and appended image
        // tokens after it, which produced a prompt with no role framing at all — the model was doing
        // free continuation, and answered a picture of a red 1 with "The digit is 3, and it is red.
        // The digit is 3, and it is orange." Fluent, confident, and nothing to do with the image.
        var promptTokens = try tokenizer.applyChatTemplate(
            messages: Glm5NextMessageGenerator().generate(from: input),
            tools: input.tools,
            additionalContext: input.additionalContext)

        if !input.videos.isEmpty {
            guard input.images.isEmpty else {
                throw Glm5NextDecoderUnavailable(
                    detail: "images and video in one request: their placeholder runs are positional "
                        + "and interleaving them correctly needs the reference's ordering")
            }
            guard input.videos.count == 1 else {
                throw Glm5NextDecoderUnavailable(
                    detail: "\(input.videos.count) videos supplied; only one per request is wired")
            }
            let (pixels, grids, seconds) = try await preprocessVideo(input.videos[0])
            let mergeArea = videoConfig.mergeSize * videoConfig.mergeSize
            var replacement = [Int]()
            for (grid, second) in zip(grids, seconds) {
                replacement.append(
                    contentsOf: frameBlock(
                        tokenCount: grid.product / mergeArea, seconds: second))
            }
            guard let videoToken else {
                throw Glm5NextDecoderUnavailable(
                    detail: "this bundle declares no video_token_id, so video cannot be placed")
            }
            promptTokens = try expandPlaceholder(
                in: promptTokens, token: videoToken, name: "<|video|>", replacement: replacement)
            return LMInput(
                text: .init(tokens: MLXArray(promptTokens.map { Int32($0) })[.newAxis, 0...]),
                image: .init(pixels: pixels, frames: grids),
                cacheScopeSalt: cacheScopeSalt(from: input.additionalContext))
        }

        guard !input.images.isEmpty else {
            return LMInput(
                text: .init(tokens: MLXArray(promptTokens.map { Int32($0) })[.newAxis, 0...]),
                cacheScopeSalt: cacheScopeSalt(from: input.additionalContext))
        }

        // One image at a time: the splice takes a single feature block per media kind, and
        // concatenating several would need their placeholder runs kept in order.
        guard input.images.count == 1 else {
            throw Glm5NextDecoderUnavailable(
                detail: "\(input.images.count) images supplied; only one per request is wired")
        }

        let (patches, grid, tokenCount) = try preprocess(image: try input.images[0].asCIImage())
        promptTokens = try expandPlaceholder(
            in: promptTokens, token: imageToken, name: "<|image|>",
            replacement: Array(repeating: imageToken, count: tokenCount))

        return LMInput(
            text: .init(tokens: MLXArray(promptTokens.map { Int32($0) })[.newAxis, 0...]),
            image: .init(pixels: patches, frames: [grid]),
            cacheScopeSalt: cacheScopeSalt(from: input.additionalContext))
    }
}

// MARK: - Image splicing

extension Glm5Next {

    /// Writes tower output into the token embeddings at the media placeholders.
    ///
    /// IMAGE AND VIDEO ARE SCATTERED SEPARATELY, each onto its own placeholder kind. Pooling
    /// `imageToken || videoToken` into one index list — which is what the shared
    /// `QwenVL.mergeInputIdsWithImageFeatures` does, and what this did at first — is correct only
    /// while the feature rows happen to appear in the same order as the placeholders. They do not:
    /// preparation concatenates image pixels before video pixels, while placeholders appear in
    /// CONVERSATION order. A turn whose earlier message carried a video and whose current one
    /// carries an image has video pads first, so a pooled scatter lays the image's rows onto the
    /// video's pads and the two blocks swap. `Qwen35` documents the same hazard.
    ///
    /// The START and END markers (154830/154831, 154832/154833) bracket each run but are ORDINARY
    /// tokens keeping their own embeddings, so they are not placeholders and must not be written.
    public func spliceMediaFeatures(
        inputIds: MLXArray, embeddings: MLXArray,
        imageFeatures: MLXArray? = nil, videoFeatures: MLXArray? = nil
    ) throws -> MLXArray {
        var result = embeddings.ndim == 2 ? embeddings[.newAxis, 0..., 0...] : embeddings
        let ids = inputIds.asArray(Int.self)

        // NOTE ON THE VIDEO BRANCH. GLM-5.3's processor rewrites a video into a run of IMAGE
        // blocks (see `preprocessVideo`), so a video's frames arrive here as `imageFeatures` and a
        // prompt built by this file never contains a video token. The branch is kept because the
        // token id is real and a prompt from elsewhere could carry one — but it is NOT the path a
        // video takes, and reading it as such was the mistake this comment exists to prevent.
        for (token, features, label) in [
            (config.imageTokenId, imageFeatures, "image"),
            (config.videoTokenId, videoFeatures, "video"),
        ] {
            guard let token, let features else { continue }
            let positions = ids.enumerated().filter { $0.element == token }.map(\.offset)
            let provided = features.ndim >= 2 ? features.dim(features.ndim - 2) : 0
            guard positions.count == provided else {
                throw Glm5NextInputShapeError(
                    got: [positions.count, provided],
                    expected:
                        "one \(label) feature per \(label) placeholder; the prompt reserved "
                        + "\(positions.count) and the tower produced \(provided)")
            }
            guard !positions.isEmpty else { continue }
            let rows = features.ndim == 2 ? features[.newAxis, 0..., 0...] : features
            result[0..., MLXArray(positions), 0...] = rows
        }
        return result
    }

    /// Kept for the single-kind case, which is every prompt that carries only images.
    public func spliceImageFeatures(
        inputIds: MLXArray, embeddings: MLXArray, imageFeatures: MLXArray
    ) throws -> MLXArray {
        try spliceMediaFeatures(
            inputIds: inputIds, embeddings: embeddings, imageFeatures: imageFeatures)
    }
}

// MARK: - Language model

/// The decoder stack: embeddings, the scheduled layers, and the final norm.
///
/// The MTP layer, when the bundle has one, is simply the LAST ENTRY of `layers` — that is how the
/// checkpoint stores it (`model.layers.45` alongside `model.layers.0…44`), so modelling it as a
/// separate head would mean rewriting keys for no gain. `num_nextn_predict_layers` decides whether
/// it exists at all: the same model is published with and without MTP, and the version without must
/// declare no parameter its checkpoint lacks.
public final class Glm5NextLanguageModel: Module {

    public let numDecoderLayers: Int
    public let numMTPLayers: Int
    public let usesHyperConnections: Bool
    public let hcMult: Int

    @ModuleInfo(key: "embed_tokens") public var embedTokens: Embedding
    @ModuleInfo(key: "layers") public var layers: [Glm5NextDecoderLayer]
    @ModuleInfo(key: "norm") public var norm: RMSNorm

    public init(_ config: Glm5NextTextConfiguration) throws {
        let schedule = try config.validatedSchedule()
        self.numDecoderLayers = config.numHiddenLayers
        self.numMTPLayers = max(0, config.numNextnPredictLayers)
        self.usesHyperConnections = config.mhc
        self.hcMult = config.hcMult

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.hiddenSize)

        var built = (0 ..< config.numHiddenLayers).map { index in
            Glm5NextDecoderLayer(
                config, kind: schedule[index], mlpKind: config.mlpLayerTypes[index])
        }
        // The MTP layer is NOT in `layer_types` — that list is exactly `num_hidden_layers` long, as
        // `validatedSchedule` enforces. Its kinds come from the weights instead: the shipped layer
        // 45 carries MLA plus an indexer and a routed MLP.
        for _ in 0 ..< numMTPLayers {
            built.append(
                Glm5NextDecoderLayer(
                    config, kind: .deepseekSparseAttention, mlpKind: .sparse,
                    isMultiTokenPrediction: true))
        }
        _layers.wrappedValue = built
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    /// One cache per DECODER layer, of the kind that layer needs.
    ///
    /// The MTP layer is excluded for the same reason it is excluded from the forward: it is driven
    /// by speculative decoding, not by ordinary generation, and a cache list longer than the stack
    /// would silently misalign every layer's slot.
    public func makeCache(maxKVSize: Int? = nil) -> [KVCache] {
        (0 ..< numDecoderLayers).map { index in
            if layers[index].kind == .linearAttention { return MambaCache() }
            if let maxKVSize { return RotatingKVCache(maxSize: maxKVSize, keep: 4) }
            // Sparse layers get the cache that ALSO remembers the indexer's packed rows. A plain
            // KVCacheSimple would still generate — the selection is skipped below index_topk — and
            // would then fail, or worse mis-select, at the moment the sequence crosses it.
            return Glm5NextIndexedKVCache()
        }
    }

    /// The MTP layer, if this bundle has one.
    public var multiTokenPredictionLayer: Glm5NextDecoderLayer? {
        numMTPLayers > 0 ? layers.last : nil
    }

    /// Runs the BACKBONE only — the MTP layer is deliberately excluded.
    ///
    /// It predicts the token after next and is driven by the speculative-decoding path, not by
    /// ordinary generation. Folding it into the stack would corrupt every ordinary forward, which is
    /// exactly the kind of mistake `layers.count` invites when the MTP layer lives inside `layers`.
    public func callAsFunction(
        _ inputs: MLXArray, mask: MLXArray? = nil, caches: [KVCache]? = nil,
        inputEmbedding: MLXArray? = nil
    ) throws -> MLXArray {
        var h = inputEmbedding ?? embedTokens(inputs)

        // WITH hyper-connections the residual stream is WIDER than the hidden size: (B, L, H)
        // becomes (B, L, hcMult, H), tiled, and every `collapse`/`expand` pair works on that shape.
        // `DeepseekV4Model` does the same. A stream left at (B, L, H) fails inside `collapse` with a
        // reshape error rather than anything that names the cause.
        if usesHyperConnections {
            h = repeated(h.expandedDimensions(axis: -2), count: hcMult, axis: -2)
        }

        for index in 0 ..< numDecoderLayers {
            h = try layers[index](h, mask: mask, cache: caches?[safe: index])
        }

        if usesHyperConnections {
            h = Self.reduceHyperConnectionStream(h)
        }
        return norm(h)
    }

    /// Collapses the hcMult copies back to one hidden state: an UNWEIGHTED MEAN.
    ///
    /// VERIFIED against the reference. `Glm5NextTextHyperHead` in transformers'
    /// `models/glm5_next/modeling_glm5_next.py` is one line — `hidden_streams.mean(dim=2)` — under
    /// the docstring "Unlike DeepSeek-V4, this is an unweighted mean". DeepSeek V4 reduces with a
    /// LEARNED head; GLM ships no such weights, which is what made a parameter-free reduce
    /// inferable from the checkpoint alone. But mean and sum are both parameter-free, and the
    /// checkpoint cannot distinguish them: this file previously summed, on the grounds that
    /// summation is the usual choice in the hyper-connection literature. Only the reference settles
    /// it. `Glm5NextHyperConnectionOracleTests` pins the whole bracketing against a transcription of
    /// that file, and would catch a silent return to the sum.
    ///
    /// Static because it holds no instance state, and because a free function is far easier to put
    /// under a differential oracle than a method reachable only through a constructed model.
    public static func reduceHyperConnectionStream(_ h: MLXArray) -> MLXArray {
        h.mean(axis: -2)
    }
}

extension Array {
    /// Bounds-safe lookup for the optional per-layer cache arrays: a caller may pass none, or a
    /// shorter list than there are layers, and neither should trap.
    fileprivate subscript(safe index: Int) -> Element? {
        indices.contains(index) ? self[index] : nil
    }
}

extension UserInput.Video {
    /// The asset, for METADATA only — duration and natural size, which the video canvas needs
    /// before any frame is decoded. Decoding still goes through `MediaProcessing`.
    fileprivate func asAVAssetForSizing() -> AVAsset {
        switch self {
        case .avAsset(let asset): return asset
        case .url(let url): return AVURLAsset(url: url)
        default: return AVURLAsset(url: URL(fileURLWithPath: "/dev/null"))
        }
    }
}

/// Emits the media items GLM-5.3's chat template looks for.
///
/// Its template dispatches on `item.type in ['image', 'image_url']` and `['video', 'video_url']`;
/// a message with no such item renders no marker at all, and the attachment is silently dropped.
public struct Glm5NextMessageGenerator: MessageGenerator {
    public init() {}

    public func generate(message: Chat.Message) -> MLXLMCommon.Message {
        [
            "role": message.role.rawValue,
            "content": [["type": "text", "text": message.content]]
                + message.images.map { _ in ["type": "image"] }
                + message.videos.map { _ in ["type": "video"] },
        ]
    }
}

// MARK: - Image processing

/// The bundle's `processor_config.json`, image half.
public struct Glm5NextImageProcessorConfiguration: Codable, Sendable {
    public let imageMean: [Float]
    public let imageStd: [Float]
    public let patchSize: Int
    public let mergeSize: Int
    public let temporalPatchSize: Int
    public let minImageTokens: Int
    public let maxImageTokens: Int
    public let patchExpandFactor: Int?

    enum CodingKeys: String, CodingKey {
        case imageMean = "image_mean"
        case imageStd = "image_std"
        case patchSize = "patch_size"
        case mergeSize = "merge_size"
        case temporalPatchSize = "temporal_patch_size"
        case minImageTokens = "min_image_tokens"
        case maxImageTokens = "max_image_tokens"
        case patchExpandFactor = "patch_expand_factor"
    }

    /// Pixels per token after merging: a token covers a `mergeSize × mergeSize` block of
    /// `patchSize × patchSize` patches, so 14 and 2 give 784.
    public var pixelsPerToken: Int { patchSize * mergeSize * patchSize * mergeSize }

    /// The resize grid must land on whole merged blocks.
    public var resizeFactor: Int { patchSize * mergeSize }

    /// THE CONVERSION THAT MATTERS. This bundle states its budget in TOKENS — 16 to 8000 — where
    /// `QwenVL.targetSize` wants PIXELS. Passing the token counts straight through would ask for an
    /// image of 8000 pixels total, roughly 89×89, and every image would be destroyed while
    /// everything downstream still ran.
    public var minPixels: Int { minImageTokens * pixelsPerToken }
    public var maxPixels: Int { maxImageTokens * pixelsPerToken }
}

/// Turns an image into the patch sequence the tower consumes.
public struct Glm5NextImageProcessor {

    public let config: Glm5NextImageProcessorConfiguration

    public init(_ config: Glm5NextImageProcessorConfiguration) {
        self.config = config
    }

    /// Resize target for an image, in pixels, honouring the token budget.
    public func targetSize(height: Int, width: Int) throws -> (height: Int, width: Int) {
        try QwenVL.targetSize(
            height: height, width: width,
            factor: config.resizeFactor,
            minPixels: config.minPixels, maxPixels: config.maxPixels)
    }

    /// How many image tokens a resized image becomes — what the prompt must reserve.
    ///
    /// Derived from the grid rather than counted after the fact, because the prompt has to be built
    /// BEFORE the tower runs: the placeholder count and the tower's output length must agree, and a
    /// mismatch is a shape error deep in the splice rather than anything that names the cause.
    public func tokenCount(height: Int, width: Int) -> Int {
        (height / config.resizeFactor) * (width / config.resizeFactor)
    }
}

// MARK: - Quantization

/// The bundle's per-module quantization, as declared in `config.json`.
///
/// It is MIXED: 495 modules at 8-bit/g64, 111 at 2-bit/g128, plus 3-bit and 6-bit stragglers. A
/// single global (bits, groupSize) — the usual shortcut — would misread most of the file, and
/// wrongly-dequantized weights produce plausible-looking garbage rather than an error.
public struct Glm5NextQuantization: Codable, Sendable {
    public let groupSize: Int
    public let bits: Int
    /// Keyed by module path, e.g. `model.layers.0.mlp.down_proj`.
    public let perModule: [String: Setting]

    public struct Setting: Codable, Sendable {
        public let groupSize: Int
        public let bits: Int
        enum CodingKeys: String, CodingKey {
            case groupSize = "group_size"
            case bits
        }
    }

    enum CodingKeys: String, CodingKey {
        case groupSize = "group_size"
        case bits
    }

    /// Any key, so the per-module entries can be walked without naming them.
    private struct ModuleKey: CodingKey {
        let stringValue: String
        var intValue: Int? { nil }
        init?(stringValue: String) { self.stringValue = stringValue }
        init?(intValue: Int) { nil }
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        groupSize = try c.decode(Int.self, forKey: .groupSize)
        bits = try c.decode(Int.self, forKey: .bits)

        // The object mixes two scalars with hundreds of per-module objects. Decoding the whole thing
        // as `[String: Setting?]` does NOT skip the scalars — it fails with a type mismatch on the
        // first one — so each key is tried individually and the ones that are not settings are
        // simply not settings.
        var modules: [String: Setting] = [:]
        let any = try decoder.container(keyedBy: ModuleKey.self)
        for key in any.allKeys where key.stringValue != "group_size" && key.stringValue != "bits" {
            if let setting = try? any.decode(Setting.self, forKey: key) {
                modules[key.stringValue] = setting
            }
        }
        perModule = modules
    }

    /// The setting for a module path, falling back to the file-wide default.
    public func setting(for path: String) -> (groupSize: Int, bits: Int) {
        if let s = perModule[path] { return (s.groupSize, s.bits) }
        return (groupSize, bits)
    }
}

// MARK: - Construction

/// Raised by the decoder entry points until the compute modules land.
///
/// Deliberately loud and specific. The alternative — leaving `glm5_next` unregistered — reports
/// `unsupportedModelType`, which says nothing about how far support has got.
/// A malformed input, reported rather than trapped.
public struct Glm5NextInputShapeError: Error, CustomStringConvertible {
    public let got: [Int]
    public let expected: String
    public var description: String {
        "glm5_next: expected \(expected), got shape \(got)"
    }
}

public struct Glm5NextDecoderUnavailable: Error, CustomStringConvertible {
    public let detail: String
    public var description: String {
        "glm5_next: configuration and construction are supported, but the decoder is not implemented "
            + "yet (\(detail)). The remaining work is the indexer's key-pooling compression "
            + "(index_kpool_compress_ape/_gate), which has no counterpart in this repo."
    }
}

/// GLM-5.3 (`glm5_next`).
///
/// Conforms to ``ModelComponentMapping`` from the outset, so that registering it later is one line
/// in the factory table — `createSelecting(Glm5NextConfiguration.self, Glm5Next.init(_:requesting:))`
/// — with the narrowing behaviour already decided and tested.
///
/// NOT registered yet, deliberately. `createSelecting` requires `LanguageModel`, whose forward pass
/// does not throw; conforming before the decoder exists would mean a `fatalError` on the one path a
/// caller actually reaches, which is the failure mode this repo has been removing (see the
/// force-unwrap in `PromptTailDecodeTests` that took the whole test process down). An
/// `unsupportedModelType` is a worse message but an honest one.
public class Glm5Next: Module, ModalityBearing, ModelComponentMapping {

    public let config: Glm5NextConfiguration
    public let plan: ResolvedConstructionPlan
    public let modalities: Set<ModelRuntimeRequestModality>

    @ModuleInfo(key: "model") public var languageModel: Glm5NextLanguageModel
    /// Separate from the embeddings: `tie_word_embeddings` is false in the shipped bundle, and a
    /// tied one would ship no `lm_head` at all.
    @ModuleInfo(key: "lm_head") public var lmHead: Linear?
    /// `visual`, not `model.visual` — the checkpoint puts the tower at the top level. Built only
    /// when the plan asks for it, which is what makes a narrowed construction loadable.
    @ModuleInfo(key: "visual") public var visionTower: Glm5NextVisionTower?

    /// The validated decoder schedule, resolved once at construction so a forward pass never has to
    /// re-derive it or risk disagreeing with the configuration.
    public let schedule: [Glm5NextLayerKind]

    public static func constructibleModalities(
        of config: Glm5NextConfiguration
    ) -> Set<ModelRuntimeRequestModality> {
        var out: Set<ModelRuntimeRequestModality> = [.text]
        if config.canBuildVisionTower { out.insert(.vision) }
        if config.canConsumeVideo { out.insert(.video) }
        return out
    }

    /// Which MODULES a request needs. Both media lanes are served by the one tower, so either is
    /// enough to require it — the distinction Qwen3.5 had to learn the hard way.
    public static func components(
        for requested: Set<ModelRuntimeRequestModality>, of config: Glm5NextConfiguration
    ) -> Set<ModelComponent> {
        var out: Set<ModelComponent> = []
        if requested.contains(.vision) || requested.contains(.video) { out.insert(.visionTower) }
        return out
    }

    /// What the built modules serve. `.text` comes from this family's core being a text LM; `.video`
    /// additionally needs the video token, which the tower knows nothing about.
    public static func servedModalities(
        by components: Set<ModelComponent>, of config: Glm5NextConfiguration
    ) -> Set<ModelRuntimeRequestModality> {
        var out: Set<ModelRuntimeRequestModality> = []
        if components.contains(.languageCore) { out.insert(.text) }
        if components.contains(.visionTower) {
            out.insert(.vision)
            if config.canConsumeVideo { out.insert(.video) }
        }
        return out
    }

    public convenience init(_ config: Glm5NextConfiguration) throws {
        try self.init(config, requesting: nil)
    }

    /// - Parameter requesting: the caller's subset, or nil for everything the configuration offers.
    public init(
        _ config: Glm5NextConfiguration, requesting: Set<ModelRuntimeRequestModality>?
    ) throws {
        let plan = try Self.resolveConstruction(config, requesting: requesting)
        self.config = config
        self.plan = plan
        self.schedule = try config.textConfig.validatedSchedule()
        self.modalities = plan.served
        _languageModel.wrappedValue = try Glm5NextLanguageModel(config.textConfig)
        if !config.textConfig.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(
                config.textConfig.hiddenSize, config.textConfig.vocabSize, bias: false)
        }
        if plan.builds(.visionTower), let vision = config.visionConfig {
            _visionTower.wrappedValue = Glm5NextVisionTower(vision)
        }
        super.init()
    }

    /// Applies the bundle's key policy, asking the PLAN whether the tower exists rather than
    /// inferring it from which keys happen to be present.
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        Glm5NextCheckpointKeys.sanitize(weights, keepVision: plan.builds(.visionTower))
    }

    /// The linear-attention layers, in decoder order. Built lazily by `buildLinearAttentionLayers`
    /// so that constructing the model — which the routed factory does to answer questions about
    /// modalities — does not allocate 34 layers' worth of projections.
    public private(set) var linearAttentionLayers: [Int: Glm5NextLinearAttention] = [:]

    /// Instantiates the linear-attention layers named by the schedule.
    ///
    /// Keyed by DECODER INDEX rather than stored as a dense array, because the schedule is sparse:
    /// layer 3 is sparse attention, so a dense array would need a hole and every consumer would have
    /// to remember which positions are real.
    public func buildLinearAttentionLayers() {
        guard linearAttentionLayers.isEmpty else { return }
        for index in layerIndices(of: .linearAttention) {
            linearAttentionLayers[index] = Glm5NextLinearAttention(config.textConfig)
        }
    }

    /// Layer indices running each attention kind, which is what a decoder needs to dispatch on.
    public func layerIndices(of kind: Glm5NextLayerKind) -> [Int] {
        schedule.enumerated().filter { $0.element == kind }.map(\.offset)
    }

    /// Text-only forward: token ids to logits.
    ///
    /// Vision is not wired in yet — the tower builds and its weights bind, but nothing splices image
    /// embeddings into the token stream, so passing pixels would silently ignore them. That is why
    /// this takes token ids and nothing else.
    public func callAsFunction(
        _ inputs: MLXArray, mask: MLXArray? = nil, caches: [KVCache]? = nil,
        inputEmbedding: MLXArray? = nil
    ) throws -> MLXArray {
        // A 1-D token array reaches MLX's `reshape` and dies there with a FATAL error, which takes
        // the process down rather than failing the call. Check the rank here, where it can be said
        // in terms of the caller's mistake.
        guard inputs.ndim == 2 else {
            throw Glm5NextInputShapeError(
                got: inputs.shape, expected: "[batch, sequence] token ids")
        }
        let hidden = try languageModel(
            inputs, mask: mask, caches: caches, inputEmbedding: inputEmbedding)
        if let lmHead { return lmHead(hidden) }
        return languageModel.embedTokens.asLinear(hidden)
    }
}

extension Array where Element == Float {
    /// `MediaProcessing.normalize` wants tuples; the config carries arrays.
    fileprivate var asCGFloat3: (CGFloat, CGFloat, CGFloat) {
        (
            CGFloat(self.count > 0 ? self[0] : 0),
            CGFloat(self.count > 1 ? self[1] : 0),
            CGFloat(self.count > 2 ? self[2] : 0)
        )
    }
}
