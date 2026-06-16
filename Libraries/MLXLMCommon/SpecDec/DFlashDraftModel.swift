// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Port target: z-lab/gpt-oss-20b-DFlash dflash.py (277 lines) and
//              z-lab/Qwen3.5-27B-DFlash dflash.py (187 lines).
//              Byte-compatible forward pass with the humanrouter/ddtree-mlx
//              Python MLX runtime.
//
// The DFlash drafter is a small Qwen3-style transformer (5-8 layers) whose
// attention reads K/V from BOTH its own `noise_embedding` input AND a
// stack of target-model hidden states at specific layer indices. The
// drafter does not ship its own embedding or LM head — it consumes the
// target's `embed_tokens(block)` at the input and the target's `lm_head`
// at the output.
//
// Phase 1 (iter 3-6) implementation. Not wired into `SpecDecRuntime`
// yet — that lands once byte-parity with the Python MLX reference is
// pinned on 10 fixed `(prompt, target_hidden)` pairs at temp 0.

import Foundation
import MLX
import MLXNN

// MARK: - Configuration

/// Drafter-specific section of the HF `config.json`.
public struct DFlashInnerConfig: Codable, Sendable {
    public let maskTokenId: Int
    public let targetLayerIds: [Int]

    enum CodingKeys: String, CodingKey {
        case maskTokenId = "mask_token_id"
        case targetLayerIds = "target_layer_ids"
    }
}

/// Full drafter configuration decoded from `z-lab/<model>-DFlash/config.json`.
///
/// Matches the shape of `transformers.Qwen3Config` with the DFlash
/// extensions (`dflash_config`, `block_size`). All fields map 1:1 with
/// the HF JSON keys to keep the loader a thin JSON decode.
public struct DFlashDrafterConfiguration: Codable, Sendable {

    // Core transformer dims
    public let hiddenSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let headDim: Int
    public let intermediateSize: Int

    // Norm / rope
    public let rmsNormEps: Float
    public let ropeTheta: Float
    public let maxPositionEmbeddings: Int

    // Qwen3-specific / architectural flags
    public let attentionBias: Bool
    public let hiddenAct: String

    // DFlash extensions
    public let blockSize: Int
    public let dflashConfig: DFlashInnerConfig

    // Informational
    public let modelType: String?
    public let dtype: String?

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case intermediateSize = "intermediate_size"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case maxPositionEmbeddings = "max_position_embeddings"
        case attentionBias = "attention_bias"
        case hiddenAct = "hidden_act"
        case blockSize = "block_size"
        case dflashConfig = "dflash_config"
        case modelType = "model_type"
        case dtype
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        self.numHiddenLayers = try c.decode(Int.self, forKey: .numHiddenLayers)
        self.numAttentionHeads = try c.decode(Int.self, forKey: .numAttentionHeads)
        self.numKeyValueHeads = try c.decodeIfPresent(Int.self, forKey: .numKeyValueHeads)
            ?? self.numAttentionHeads
        self.headDim = try c.decodeIfPresent(Int.self, forKey: .headDim)
            ?? (self.hiddenSize / self.numAttentionHeads)
        self.intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        self.rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        self.ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1_000_000.0
        self.maxPositionEmbeddings = try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings)
            ?? 131_072
        self.attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        self.hiddenAct = try c.decodeIfPresent(String.self, forKey: .hiddenAct) ?? "silu"
        self.blockSize = try c.decode(Int.self, forKey: .blockSize)
        self.dflashConfig = try c.decode(DFlashInnerConfig.self, forKey: .dflashConfig)
        guard !self.dflashConfig.targetLayerIds.isEmpty else {
            throw DecodingError.dataCorruptedError(
                forKey: .dflashConfig,
                in: c,
                debugDescription: "DFlash dflash_config.target_layer_ids must be non-empty")
        }
        guard self.dflashConfig.targetLayerIds.allSatisfy({ $0 >= 0 }) else {
            throw DecodingError.dataCorruptedError(
                forKey: .dflashConfig,
                in: c,
                debugDescription: "DFlash dflash_config.target_layer_ids must be nonnegative")
        }
        self.modelType = try c.decodeIfPresent(String.self, forKey: .modelType)
        self.dtype = try c.decodeIfPresent(String.self, forKey: .dtype)
    }
}

// MARK: - Attention with target-hidden K/V injection

/// Drafter attention that augments K and V with target-model hidden states.
///
/// Port of `Qwen3DFlashAttention`. Differs from plain `Qwen3Attention`:
/// - K and V are computed from BOTH `target_hidden` (context projection)
///   AND `hidden_states` (the drafter's own noise sequence).
/// - The target-hidden projection is NOT RoPE-rotated (it's treated as
///   already positioned by the target).
/// - Q is rotated normally over the drafter's `q_len`.
/// - `is_causal = False` in Python — masking is handled explicitly by the
///   `attentionMask` passed in from the runtime.
final class DFlashAttention: Module {
    let args: DFlashDrafterConfiguration
    let scale: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let rope: RoPE

    public init(_ args: DFlashDrafterConfiguration) {
        self.args = args
        let dim = args.hiddenSize
        let heads = args.numAttentionHeads
        let kvHeads = args.numKeyValueHeads
        let headDim = args.headDim
        self.scale = pow(Float(headDim), -0.5)

        _wq.wrappedValue = Linear(dim, heads * headDim, bias: args.attentionBias)
        _wk.wrappedValue = Linear(dim, kvHeads * headDim, bias: args.attentionBias)
        _wv.wrappedValue = Linear(dim, kvHeads * headDim, bias: args.attentionBias)
        _wo.wrappedValue = Linear(heads * headDim, dim, bias: args.attentionBias)

        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)

        self.rope = RoPE(
            dimensions: headDim, traditional: false, base: args.ropeTheta, scale: 1)
    }

    /// - Parameters:
    ///   - hiddenStates: `(B, q_len, hidden)` drafter input (noise embedding).
    ///   - targetHidden: `(B, ctx_len, hidden)` post-`fc`+`hidden_norm`
    ///     target context projection.
    ///   - positionIds: `(B, q_len)` Int32 absolute positions for the
    ///     drafter queries.
    ///   - mask: `(..., q_len, ctx_len + q_len)` additive attention mask.
    ///     `nil` → causal-over-concat default.
    public func callAsFunction(
        hiddenStates: MLXArray,
        targetHidden: MLXArray,
        positionIds: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray {
        let b = hiddenStates.dim(0)
        let qLen = hiddenStates.dim(1)
        let ctxLen = targetHidden.dim(1)
        let heads = args.numAttentionHeads
        let kvHeads = args.numKeyValueHeads
        let headDim = args.headDim

        // Q comes only from hidden_states (drafter's own sequence).
        var q = wq(hiddenStates)
            .reshaped(b, qLen, heads, headDim)
        q = qNorm(q).transposed(0, 2, 1, 3)

        // K/V come from concat(target_hidden, hidden_states) — this is the
        // "KV injection" of the DFlash paper.
        let kCtx = wk(targetHidden)
        let kNoise = wk(hiddenStates)
        let vCtx = wv(targetHidden)
        let vNoise = wv(hiddenStates)
        var k = concatenated([kCtx, kNoise], axis: 1)
            .reshaped(b, ctxLen + qLen, kvHeads, headDim)
        var v = concatenated([vCtx, vNoise], axis: 1)
            .reshaped(b, ctxLen + qLen, kvHeads, headDim)
        k = kNorm(k).transposed(0, 2, 1, 3)
        v = v.transposed(0, 2, 1, 3)

        // RoPE applies to q over its own positions; k uses its "full"
        // positions which include the ctx_len prefix. The Python reference
        // rotates both with the same (cos, sin) but using the last q_len
        // slice for q. We match by applying RoPE to q at `positionIds`
        // and to k at the concatenated range `[0..ctxLen+qLen)` —
        // equivalent because ctx_len positions are assumed continuous with
        // the q positions (the target provided them in order).
        q = rope(q, offset: positionIds.asArray(Int.self).first ?? 0)
        k = rope(k, offset: 0)

        let output = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(b, qLen, -1)

        return wo(output)
    }
}

// MARK: - MLP

/// SiLU-gated MLP, identical to `Qwen3MLP`.
final class DFlashMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear

    public init(dimensions: Int, hiddenDimensions: Int) {
        _gate.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        _down.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        _up.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

// MARK: - Decoder layer

final class DFlashDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: DFlashAttention
    let mlp: DFlashMLP

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    public init(_ args: DFlashDrafterConfiguration) {
        _attention.wrappedValue = DFlashAttention(args)
        self.mlp = DFlashMLP(
            dimensions: args.hiddenSize, hiddenDimensions: args.intermediateSize)
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    public func callAsFunction(
        hiddenStates: MLXArray,
        targetHidden: MLXArray,
        positionIds: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray {
        var h = hiddenStates
        let r1 = attention(
            hiddenStates: inputLayerNorm(h),
            targetHidden: targetHidden,
            positionIds: positionIds,
            mask: mask)
        h = h + r1
        let r2 = mlp(postAttentionLayerNorm(h))
        return h + r2
    }
}

// MARK: - Top-level drafter

/// DFlash block-diffusion drafter.
///
/// Consumes a shared-with-target embedding of the block input
/// (`[bonus, mask, mask, ...]`) and a projected-concat of target hidden
/// states from `dflash_config.target_layer_ids`, and produces per-position
/// drafter hidden states. The caller applies the target model's LM head
/// to these hidden states to obtain draft logits.
///
/// Phase 1 acceptance: forward pass byte-identical to humanrouter/ddtree-mlx
/// on 10 fixed `(noise_embedding, target_hidden, position_ids)` triples at
/// fp32. Current state (iter 2): architecture + weight load path; forward
/// untested end-to-end.
public final class DFlashDraftModel: Module, @unchecked Sendable {

    public let config: DFlashDrafterConfiguration

    fileprivate let layers: [DFlashDecoderLayer]
    let norm: RMSNorm

    /// Projection from `len(target_layer_ids) * hidden` → `hidden`. Applied
    /// once to the concatenated target context before the decoder stack.
    @ModuleInfo(key: "fc") var fc: Linear

    @ModuleInfo(key: "hidden_norm") var hiddenNorm: RMSNorm

    public init(_ config: DFlashDrafterConfiguration) {
        self.config = config
        self.layers = (0..<config.numHiddenLayers).map { _ in DFlashDecoderLayer(config) }
        self.norm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        let nTargetLayers = config.dflashConfig.targetLayerIds.count
        _fc.wrappedValue = Linear(
            nTargetLayers * config.hiddenSize, config.hiddenSize, bias: false)
        _hiddenNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    /// Forward pass.
    ///
    /// - Parameters:
    ///   - noiseEmbedding: `(B, block_size, hidden)` — result of
    ///     `target.embed_tokens([bonus, mask, mask, ..., mask])`. Shared
    ///     embedding with target.
    ///   - targetHidden: `(B, ctx_len, len(target_layer_ids) * hidden)` —
    ///     concatenation along the last dim of target hidden states at
    ///     `target_layer_ids`.
    ///   - positionIds: `(B, block_size)` absolute positions.
    ///   - attentionMask: `(1, 1, q_len, ctx_len + q_len)` additive mask
    ///     or `nil` for causal-over-concat default.
    /// - Returns: `(B, block_size, hidden)` post-LN drafter output ready
    ///   for the target's LM head.
    public func callAsFunction(
        noiseEmbedding: MLXArray,
        targetHidden: MLXArray,
        positionIds: MLXArray,
        attentionMask: MLXFast.ScaledDotProductAttentionMaskMode = .none
    ) -> MLXArray {
        var h = noiseEmbedding
        let projected = hiddenNorm(fc(targetHidden))
        for layer in layers {
            h = layer(
                hiddenStates: h,
                targetHidden: projected,
                positionIds: positionIds,
                mask: attentionMask)
        }
        return norm(h)
    }
}
