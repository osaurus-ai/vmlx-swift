// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// JANGTQ (TurboQuant codebook) variant of DeepseekV3. Swaps the
// per-expert routed `SwitchGLU` for `TurboQuantSwitchGLU` so the
// codebook Metal kernels run instead of `gather_qmm`. MLA attention,
// norms, RoPE, SDPA, `lm_head`, `embed_tokens`, shared experts, and
// dense layer 0 are untouched — they already call the same
// `mx.fast.*` / `mx.quantized_matmul` entry points Python uses.
//
// Target bundles:
//   - Kimi-K2.6-REAP-30-JANGTQ_1L  (model_type = "kimi_k25", 191 GB)
//   - Kimi-K2.6-REAP-50-JANGTQ_1L  (future, ~118 GB)
//   - DeepSeek-V3 JANGTQ bundles    (model_type = "deepseek_v3")
//
// Non-goals (per jang/research/KIMI-K2.6-IMPLEMENTATION.md §4):
//   - MLA L==1 absorb branch — current materialized-K/V path now uses
//     the decode-only fp32 SDPA cast, but it still skips the absorb
//     optimization. Adding absorb is a follow-up perf optimization.
//   - Vision tower / MoonViT — VL variants need `KimiVLM.swift` +
//     `KimiMoonViT.swift` layered on top of this text model.
//
// The dispatch from `LLMModelFactory` peeks `weight_format == "mxtq"`
// in config.json and routes to `DeepseekV3JANGTQModel` here; all
// other `deepseek_v3` / `kimi_k25` bundles continue to use the
// standard `DeepseekV3Model`.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

private func mlxPressStreamingMoETokenChunkSize() -> Int {
    let env = ProcessInfo.processInfo.environment
    let raw =
        env["MLXPRESS_STREAMING_MOE_TOKEN_CHUNK_SIZE"]
        ?? env["MLXPRESS_STREAMING_MOE_TOKEN_CHUNK"]
        ?? env["VMLX_JANGTQ_PREFILL_STEP"]
        ?? env["MLXPRESS_STREAMING_TOKEN_CHUNK_SIZE"]
        ?? env["MLXPRESS_STREAMING_TOKEN_CHUNK"]
        ?? "16"
    return max(1, Int(raw) ?? 16)
}

private func mlxPressStreamingMoESyncBetweenChunks() -> Bool {
    let env = ProcessInfo.processInfo.environment
    let raw =
        env["MLXPRESS_STREAMING_MOE_PREFILL_SYNC"]?
            .lowercased()
        ?? env["VMLX_JANGTQ_PREFILL_SYNC"]?
            .lowercased()
        ?? "1"
    return raw != "0" && raw != "false" && raw != "no" && raw != "off"
}

private func mlxPressStreamingEvalDecoderLayer() -> Bool {
    let env = ProcessInfo.processInfo.environment
    let raw =
        env["MLXPRESS_STREAMING_EVAL_DECODER_LAYER"]?
            .lowercased()
        ?? env["MLXPRESS_STREAMING_EVAL_EACH_LAYER"]?
            .lowercased()
        ?? "1"
    return raw != "0" && raw != "false" && raw != "no" && raw != "off"
}

// MARK: - Configuration

/// JANGTQ-flavored DeepseekV3 configuration. Inherits every field the
/// affine-side `DeepseekV3Configuration` accepts; adds three JANGTQ
/// knobs (`weight_format`, `mxtq_bits`, `mxtq_seed`).
///
/// We keep the affine config intact rather than subclassing so any
/// future field added to the affine `DeepseekV3Configuration` doesn't
/// silently change the JANGTQ wire format. The `asAffine()` helper
/// JSON round-trips us into the exact shape `DeepseekV3Attention` /
/// `DeepseekV3MLP` / `DeepseekV3MoE` / `MoEGate` expect — they all take
/// a `DeepseekV3Configuration`.
public struct DeepseekV3JANGTQConfiguration: Codable, Sendable {
    // Mirrors DeepseekV3Configuration exactly.
    public var modelType: String = "deepseek_v3"
    public var vocabSize: Int
    public var hiddenSize: Int
    public var intermediateSize: Int
    public var moeIntermediateSize: Int
    public var numHiddenLayers: Int
    public var numAttentionHeads: Int
    public var numKeyValueHeads: Int
    public var nSharedExperts: Int?
    public var nRoutedExperts: Int?
    public var routedScalingFactor: Float
    public var kvLoraRank: Int
    public var qLoraRank: Int
    public var qkRopeHeadDim: Int
    public var vHeadDim: Int
    public var qkNopeHeadDim: Int
    public var normTopkProb: Bool
    public var nGroup: Int?
    public var topkGroup: Int?
    public var numExpertsPerTok: Int?
    public var moeLayerFreq: Int
    public var firstKDenseReplace: Int
    public var maxPositionEmbeddings: Int
    public var rmsNormEps: Float
    public var ropeTheta: Float
    public var ropeScaling: [String: StringOrNumber]?
    public var attentionBias: Bool

    // JANGTQ-specific
    public var weightFormat: String = "mxtq"
    public var mxtqBits: Int = 2
    public var mxtqGateUpBits: Int?
    public var mxtqDownBits: Int?
    public var mxtqSeed: Int = 42

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case nSharedExperts = "n_shared_experts"
        case nRoutedExperts = "n_routed_experts"
        case routedScalingFactor = "routed_scaling_factor"
        case kvLoraRank = "kv_lora_rank"
        case qLoraRank = "q_lora_rank"
        case qkRopeHeadDim = "qk_rope_head_dim"
        case vHeadDim = "v_head_dim"
        case qkNopeHeadDim = "qk_nope_head_dim"
        case normTopkProb = "norm_topk_prob"
        case nGroup = "n_group"
        case topkGroup = "topk_group"
        case numExpertsPerTok = "num_experts_per_tok"
        case moeLayerFreq = "moe_layer_freq"
        case firstKDenseReplace = "first_k_dense_replace"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
        case attentionBias = "attention_bias"
        case weightFormat = "weight_format"
        case mxtqBits = "mxtq_bits"
        case mxtqGateUpBits = "mxtq_gate_up_bits"
        case mxtqDownBits = "mxtq_down_bits"
        case mxtqSeed = "mxtq_seed"
    }

    /// Project into the affine-side `DeepseekV3Configuration` via JSON
    /// round-trip. The internal DeepseekV3 modules all init from that
    /// type; we fabricate one by re-encoding without the JANGTQ fields.
    fileprivate func asAffine() throws -> DeepseekV3Configuration {
        let encoder = JSONEncoder()
        let decoder = JSONDecoder()
        let data = try encoder.encode(AffineProjection(self))
        return try decoder.decode(DeepseekV3Configuration.self, from: data)
    }
}

/// Re-encode the JANGTQ config without the JANGTQ-only fields so it
/// round-trips through the affine `DeepseekV3Configuration` decoder.
private struct AffineProjection: Encodable {
    let vocabSize: Int
    let hiddenSize: Int
    let intermediateSize: Int
    let moeIntermediateSize: Int
    let numHiddenLayers: Int
    let numAttentionHeads: Int
    let numKeyValueHeads: Int
    let nSharedExperts: Int?
    let nRoutedExperts: Int?
    let routedScalingFactor: Float
    let kvLoraRank: Int
    let qLoraRank: Int
    let qkRopeHeadDim: Int
    let vHeadDim: Int
    let qkNopeHeadDim: Int
    let normTopkProb: Bool
    let nGroup: Int?
    let topkGroup: Int?
    let numExpertsPerTok: Int?
    let moeLayerFreq: Int
    let firstKDenseReplace: Int
    let maxPositionEmbeddings: Int
    let rmsNormEps: Float
    let ropeTheta: Float
    let ropeScaling: [String: StringOrNumber]?
    let attentionBias: Bool

    enum CodingKeys: String, CodingKey {
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case nSharedExperts = "n_shared_experts"
        case nRoutedExperts = "n_routed_experts"
        case routedScalingFactor = "routed_scaling_factor"
        case kvLoraRank = "kv_lora_rank"
        case qLoraRank = "q_lora_rank"
        case qkRopeHeadDim = "qk_rope_head_dim"
        case vHeadDim = "v_head_dim"
        case qkNopeHeadDim = "qk_nope_head_dim"
        case normTopkProb = "norm_topk_prob"
        case nGroup = "n_group"
        case topkGroup = "topk_group"
        case numExpertsPerTok = "num_experts_per_tok"
        case moeLayerFreq = "moe_layer_freq"
        case firstKDenseReplace = "first_k_dense_replace"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
        case attentionBias = "attention_bias"
    }

    init(_ src: DeepseekV3JANGTQConfiguration) {
        self.vocabSize = src.vocabSize
        self.hiddenSize = src.hiddenSize
        self.intermediateSize = src.intermediateSize
        self.moeIntermediateSize = src.moeIntermediateSize
        self.numHiddenLayers = src.numHiddenLayers
        self.numAttentionHeads = src.numAttentionHeads
        self.numKeyValueHeads = src.numKeyValueHeads
        self.nSharedExperts = src.nSharedExperts
        self.nRoutedExperts = src.nRoutedExperts
        self.routedScalingFactor = src.routedScalingFactor
        self.kvLoraRank = src.kvLoraRank
        self.qLoraRank = src.qLoraRank
        self.qkRopeHeadDim = src.qkRopeHeadDim
        self.vHeadDim = src.vHeadDim
        self.qkNopeHeadDim = src.qkNopeHeadDim
        self.normTopkProb = src.normTopkProb
        self.nGroup = src.nGroup
        self.topkGroup = src.topkGroup
        self.numExpertsPerTok = src.numExpertsPerTok
        self.moeLayerFreq = src.moeLayerFreq
        self.firstKDenseReplace = src.firstKDenseReplace
        self.maxPositionEmbeddings = src.maxPositionEmbeddings
        self.rmsNormEps = src.rmsNormEps
        self.ropeTheta = src.ropeTheta
        self.ropeScaling = src.ropeScaling
        self.attentionBias = src.attentionBias
    }
}

// MARK: - JANGTQ MoE block

/// Mirrors `DeepseekV3MoE` but routes the per-expert projections
/// through `TurboQuantSwitchGLU`. Everything else — gate (full-
/// precision `MoEGate`), shared experts (standard `DeepseekV3MLP`,
/// loaded as affine 8-bit `QuantizedLinear` at hydration time),
/// activation (`clippedSilu`) — is identical to the affine path.
///
/// The shared expert carries the non-routed FFN component
/// DeepSeek-V3 / Kimi K2.6 always run. Per
/// `KIMI-K2.6-IMPLEMENTATION.md §1.3` shared experts are 8-bit affine
/// (not MXTQ), so the `DeepseekV3MLP` used here is correct.
final class DeepseekV3JANGTQMoE: Module, UnaryLayer {
    let config: DeepseekV3Configuration
    let layerIdx: Int
    let numExpertsPerTok: Int
    let streamingMoEChunkSize: Int
    let syncStreamingMoEChunks: Bool

    @ModuleInfo(key: "switch_mlp") var switchMLP: TurboQuantSwitchGLU
    var gate: MoEGate
    @ModuleInfo(key: "shared_experts") var sharedExperts: DeepseekV3MLP?

    init(config: DeepseekV3Configuration, jangtq: DeepseekV3JANGTQConfiguration, layerIdx: Int) {
        self.config = config
        self.layerIdx = layerIdx
        self.numExpertsPerTok = config.numExpertsPerTok ?? 1
        self.streamingMoEChunkSize = mlxPressStreamingMoETokenChunkSize()
        self.syncStreamingMoEChunks = mlxPressStreamingMoESyncBetweenChunks()

        if JANGTQStreamingExperts.isEnabled {
            _switchMLP.wrappedValue = StreamingTurboQuantSwitchGLU(
                inputDims: config.hiddenSize,
                hiddenDims: config.moeIntermediateSize,
                numExperts: config.nRoutedExperts ?? 1,
                gateUpBits: jangtq.mxtqGateUpBits ?? jangtq.mxtqBits,
                downBits: jangtq.mxtqDownBits ?? jangtq.mxtqBits,
                seed: jangtq.mxtqSeed,
                layerIdx: layerIdx)
        } else {
            _switchMLP.wrappedValue = TurboQuantSwitchGLU(
                inputDims: config.hiddenSize,
                hiddenDims: config.moeIntermediateSize,
                numExperts: config.nRoutedExperts ?? 1,
                gateUpBits: jangtq.mxtqGateUpBits ?? jangtq.mxtqBits,
                downBits: jangtq.mxtqDownBits ?? jangtq.mxtqBits,
                seed: jangtq.mxtqSeed
            )
        }

        self.gate = MoEGate(config: config)

        if let sharedExpertCount = config.nSharedExperts {
            let intermediateSize = config.moeIntermediateSize * sharedExpertCount
            self._sharedExperts.wrappedValue = DeepseekV3MLP(
                config: config, intermediateSize: intermediateSize)
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        if JANGTQStreamingExperts.isEnabled {
            let batchTokens = x.size / config.hiddenSize
            let chunkSize = max(1, streamingMoEChunkSize)
            if batchTokens > chunkSize {
                return callStreamingChunks(x, batchTokens: batchTokens, chunkSize: chunkSize)
            }
        }
        return callUnchunked(x)
    }

    private func callStreamingChunks(
        _ x: MLXArray,
        batchTokens: Int,
        chunkSize: Int
    ) -> MLXArray {
        let xFlat = x.reshaped([batchTokens, config.hiddenSize])
        var chunks: [MLXArray] = []
        chunks.reserveCapacity((batchTokens + chunkSize - 1) / chunkSize)

        var start = 0
        while start < batchTokens {
            let end = min(start + chunkSize, batchTokens)
            let xChunk = xFlat[start ..< end, 0...]
                .reshaped([1, end - start, config.hiddenSize])
            let yChunk = stopGradient(
                MLXPressStreamingProfile.time("kimi.moe.chunk") {
                    callUnchunked(xChunk).reshaped([end - start, config.hiddenSize])
                })
            if syncStreamingMoEChunks {
                MLXPressStreamingProfile.time("kimi.moe.chunk_eval") {
                    MLX.eval(yChunk)
                    MLX.Memory.clearCache()
                }
            }
            chunks.append(yChunk)
            start = end
        }

        return concatenated(chunks, axis: 0).reshaped(x.shape)
    }

    private func callUnchunked(_ x: MLXArray) -> MLXArray {
        let routeInput: MLXArray
        if JANGTQStreamingExperts.isEnabled
            && JANGTQStreamingExperts.materializeBeforeRouterReadback
        {
            routeInput = stopGradient(x)
            MLXPressStreamingProfile.time("kimi.router_input_eval") {
                MLX.eval(routeInput)
                MLX.Memory.clearCache()
            }
        } else {
            routeInput = x
        }

        let (indices, scores) = MLXPressStreamingProfile.time("kimi.router_gate") {
            gate(routeInput)
        }
        JangPressCanonicalExpertAdvisor.shared.observe(layer: layerIdx, indices: indices)
        var y: MLXArray
        if let streaming = switchMLP as? StreamingTurboQuantSwitchGLU {
            y = MLXPressStreamingProfile.time("kimi.streaming_reduce") {
                streaming.reduced(routeInput, indices: indices, scores: scores)
            }
        } else {
            y = switchMLP(routeInput, indices)
            y = (y * scores[.ellipsis, .newAxis]).sum(axis: -2)
        }

        if let shared = sharedExperts {
            let sharedOutput = MLXPressStreamingProfile.time("kimi.shared_experts_build") {
                shared(routeInput)
            }
            y = y + sharedOutput
        }
        return y
    }
}

// MARK: - Decoder layer

/// Identical structure to `DeepseekV3DecoderLayer` — just swaps the
/// MoE block for the JANGTQ variant.
final class DeepseekV3JANGTQDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: DeepseekV3Attention
    var mlp: UnaryLayer
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    let layerIdx: Int
    let evalStreamingLayerOutput: Bool

    init(
        config: DeepseekV3Configuration,
        jangtq: DeepseekV3JANGTQConfiguration,
        layerIdx: Int
    ) {
        self.layerIdx = layerIdx
        self.evalStreamingLayerOutput = mlxPressStreamingEvalDecoderLayer()
        self._selfAttn.wrappedValue = DeepseekV3Attention(config: config)

        if config.nRoutedExperts != nil,
            layerIdx >= config.firstKDenseReplace,
            layerIdx % config.moeLayerFreq == 0
        {
            self.mlp = DeepseekV3JANGTQMoE(config: config, jangtq: jangtq, layerIdx: layerIdx)
        } else {
            // Dense layer 0 (and any other non-MoE layer): standard
            // affine MLP. Per §1.3 these carry 8-bit QuantizedLinear
            // weights, not MXTQ. No JANGTQ swap needed.
            self.mlp = DeepseekV3MLP(config: config)
        }

        self._inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let r = MLXPressStreamingProfile.time("kimi.attention_build") {
            selfAttn(inputLayerNorm(x), mask: mask, cache: cache)
        }
        let h = x + r
        let r2 = MLXPressStreamingProfile.time("kimi.mlp_build") {
            mlp(postAttentionLayerNorm(h))
        }
        var out = h + r2
        if JANGTQStreamingExperts.isEnabled && evalStreamingLayerOutput {
            MLXPressStreamingProfile.time("kimi.layer_eval") {
                MLX.eval(out)
                MLX.Memory.clearCache()
            }
            out = stopGradient(out)
        }
        MLXPressMmapColdSweep.afterLayer(layerIdx, materialized: out)
        return out
    }
}

// MARK: - Inner model

public class DeepseekV3JANGTQModelInner: Module {
    var config: DeepseekV3Configuration
    var vocabSize: Int
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    var layers: [DeepseekV3JANGTQDecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(
        config: DeepseekV3Configuration,
        jangtq: DeepseekV3JANGTQConfiguration
    ) {
        self.config = config
        self.vocabSize = config.vocabSize
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        self.layers = (0 ..< config.numHiddenLayers).map {
            DeepseekV3JANGTQDecoderLayer(config: config, jangtq: jangtq, layerIdx: $0)
        }
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(_ x: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = embedTokens(x)
        let attentionMask = createAttentionMask(h: h, cache: cache?.first)
        for (i, layer) in layers.enumerated() {
            let layerCache = (cache != nil && i < cache!.count) ? cache![i] : nil
            h = layer(h, mask: attentionMask, cache: layerCache)
        }
        return norm(h)
    }
}

// MARK: - Top-level model

public class DeepseekV3JANGTQModel: Module, LLMModel, KVCacheDimensionProvider, LoRAModel {
    public var kvHeads: [Int] = []

    let jangtqConfig: DeepseekV3JANGTQConfiguration
    let affineConfig: DeepseekV3Configuration
    public var model: DeepseekV3JANGTQModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    public init(_ args: DeepseekV3JANGTQConfiguration) throws {
        self.jangtqConfig = args
        self.affineConfig = try args.asAffine()
        self.kvHeads = Array(
            repeating: args.numKeyValueHeads,
            count: args.numHiddenLayers)
        self.model = DeepseekV3JANGTQModelInner(
            config: self.affineConfig, jangtq: args)
        self._lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabSize, bias: false)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let out = model(inputs, cache: cache)
        return lmHead(out)
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        let count = affineConfig.numHiddenLayers
        if let maxKVSize = parameters?.maxKVSize {
            return (0 ..< count).map { _ in
                RotatingKVCache(maxSize: maxKVSize, keep: 4)
            }
        }
        return (0 ..< count).map { _ in KVCacheSimple() }
    }

    /// Sanitize for the JANGTQ wire format. Three jobs, mirroring
    /// both `DeepseekV3Model.sanitize` (affine-path stacking + final
    /// filter) and `Qwen35JANGTQTextModel.sanitize`
    /// (JANGTQ-specific `.tq_bits` strip + `.tq_packed`/`.tq_norms`
    /// stacking):
    ///
    /// 1. DeepseekV3's per-block FP8 dequant: resolve any
    ///    `weight_scale_inv` pairs into the full weight tensor.
    ///    Kimi K2.6 JANGTQ bundles don't carry FP8 scales (they
    ///    ship affine 8-bit on attention / shared / bookend
    ///    modules + MXTQ 2-bit on routed experts), so this is a
    ///    pass-through in practice. Kept for upstream DeepSeek-V3
    ///    FP8 `.safetensors` that might get JANGTQ-converted in a
    ///    way that preserves the scale tensors.
    ///
    /// 2. Drop `.tq_bits` metadata tensors anywhere in the tree —
    ///    these are per-tensor bit-width hints for the sidecar, not
    ///    module parameters. `QuantizedLinear.load_weights` would
    ///    otherwise reject the extra unexpected key.
    ///
    /// 3. Stack per-expert `experts.{E}.{gate,up,down}_proj.{tq_packed,tq_norms}`
    ///    tensors into the 3D layout `TurboQuantSwitchGLU` expects
    ///    under `mlp.switch_mlp.{gate,up,down}_proj.{tq_packed,tq_norms}`.
    ///    Per `jang_tools.load_jangtq._hydrate_jangtq_model` § "MoE
    ///    stacking". The non-JANGTQ `.weight`/`.scales`/`.biases`
    ///    triplets on attention / shared / bookend modules are still
    ///    stacked via the standard DeepseekV3 sanitize path above.
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        // --- Job 0: Kimi-K2.6 (kimi_k25) shape adapter.
        //
        // Kimi K2.6 ships as a multimodal-shaped bundle — its weights
        // live under `language_model.*` (LLM stack) plus
        // `vision_tower.*` and `mm_projector.*` (the multimodal
        // pieces). Plain DeepSeek V3 bundles use top-level paths.
        // The factory (`dispatchDeepseekV3Family`) flattens the
        // config (`text_config` unwrap) so the LLM-only path can run
        // Kimi K2.6 without a Kimi VLM class. Mirror that flattening
        // on the weight side: strip the `language_model.` prefix and
        // drop vision-tower / projector tensors. Idempotent — flat
        // bundles pass through unchanged.
        var newWeights: [String: MLXArray] = [:]
        for (key, value) in weights {
            if key.hasPrefix("vision_tower.")
                || key.hasPrefix("mm_projector.")
                || key.hasPrefix("multi_modal_projector.")
            {
                continue
            }
            var newKey = key
            if newKey.hasPrefix("language_model.") {
                newKey = String(newKey.dropFirst("language_model.".count))
            }
            newWeights[newKey] = value
        }

        // --- Job 1: FP8 weight_scale_inv dequant (affine pass-through
        // for JANGTQ bundles; matches `DeepseekV3Model.sanitize`).
        func dequant(weight: MLXArray, scaleInv: MLXArray) -> MLXArray {
            let bs = 128
            let (m, n) = (weight.shape[0], weight.shape[1])
            let padBottom = (bs - m % bs) % bs
            let padSide = (bs - n % bs) % bs
            var p = padded(weight, widths: [.init((0, padBottom)), .init((0, padSide))])
            p = p.reshaped([(m + padBottom) / bs, bs, (n + padSide) / bs, bs])
            let scaled = p * scaleInv[0..., .newAxis, 0..., .newAxis]
            return scaled.reshaped([m + padBottom, n + padSide])[0 ..< m, 0 ..< n]
        }
        let postShapeWeights = newWeights
        for (key, value) in postShapeWeights {
            if key.contains("weight_scale_inv") {
                let weightKey = key.replacingOccurrences(of: "_scale_inv", with: "")
                if let weight = postShapeWeights[weightKey] {
                    newWeights[weightKey] = dequant(weight: weight, scaleInv: value)
                }
            }
        }

        // --- Job 2: strip `.tq_bits` metadata tensors.
        for key in Array(newWeights.keys) where key.hasSuffix(".tq_bits") {
            newWeights[key] = nil
        }

        // --- Job 3a: stack JANGTQ routed-expert tensors.
        // Wire format uses `gate_proj` / `down_proj` / `up_proj` (matching
        // mlx-lm Python DeepseekV3). `mlx_lm.models.deepseek_v3.DeepseekV3MoE`
        // iterates over the proj-name list in that order — we do too.
        let tqRenames: [String] = ["gate_proj", "down_proj", "up_proj"]
        let tqProbe = "model.layers.\(affineConfig.firstKDenseReplace).mlp.experts.0.gate_proj.tq_packed"
        let hasTQExperts = newWeights[tqProbe] != nil
        if hasTQExperts {
            for layer in 0 ..< affineConfig.numHiddenLayers {
                let prefix = "model.layers.\(layer)"
                guard newWeights["\(prefix).mlp.experts.0.gate_proj.tq_packed"] != nil else {
                    // Layer doesn't have TQ experts (e.g. dense
                    // layer 0). Skip; affine stacking below handles
                    // any standard per-expert weights.
                    continue
                }
                for projName in tqRenames {
                    for kind in ["tq_packed", "tq_norms"] {
                        let first = "\(prefix).mlp.experts.0.\(projName).\(kind)"
                        guard newWeights[first] != nil else { continue }
                        if JANGTQStreamingExperts.isEnabled {
                            for e in 0 ..< (affineConfig.nRoutedExperts ?? 1) {
                                newWeights.removeValue(
                                    forKey: "\(prefix).mlp.experts.\(e).\(projName).\(kind)")
                            }
                            continue
                        }
                        let target = "\(prefix).mlp.switch_mlp.\(projName).\(kind)"
                        if newWeights[target] != nil {
                            for e in 0 ..< (affineConfig.nRoutedExperts ?? 1) {
                                newWeights.removeValue(
                                    forKey: "\(prefix).mlp.experts.\(e).\(projName).\(kind)")
                            }
                            continue
                        }
                        let stacked: [MLXArray] = (0 ..< (affineConfig.nRoutedExperts ?? 1)).map {
                            e in
                            newWeights.removeValue(
                                forKey: "\(prefix).mlp.experts.\(e).\(projName).\(kind)")!
                        }
                        newWeights[target] = MLX.stacked(stacked)
                    }
                }
            }
        }

        // --- Job 3b: stack any remaining affine-format per-expert
        // tensors. This covers the standard `.weight`/`.scales`/`.biases`
        // triplets that attention / shared / bookend modules carry in
        // JANGTQ bundles too (they're still affine 8-bit).
        for l in 0 ..< affineConfig.numHiddenLayers {
            let prefix = "model.layers.\(l)"
            for (_, projName) in [("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")] {
                for key in ["weight", "scales", "biases"] {
                    let firstKey = "\(prefix).mlp.experts.0.\(projName).\(key)"
                    if newWeights[firstKey] != nil {
                        let joined = (0 ..< (affineConfig.nRoutedExperts ?? 1)).map {
                            newWeights["\(prefix).mlp.experts.\($0).\(projName).\(key)"]!
                        }
                        newWeights["\(prefix).mlp.switch_mlp.\(projName).\(key)"] =
                            stacked(joined)
                    }
                }
            }
        }

        // Match DeepseekV3Model.sanitize's final filter — drop the
        // overflow layer 61 that DeepSeek's public weights always
        // ship with extra for later MTP speculative decoding, and
        // strip `rotary_emb.inv_freq` buffers (we recompute RoPE on
        // the fly).
        return newWeights.filter { key, _ in
            !key.starts(with: "model.layers.61") && !key.contains("rotary_emb.inv_freq")
        }
    }

    public var loraLayers: [Module] {
        model.layers
    }
}
