// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// DFlash 2 block-diffusion drafter.
//
// Port target: z-lab/dflash `dflash/model_mlx.py` (the reference MLX
// implementation shipped by the authors) — `DFlash2DraftModel`,
// `DFlash2DecoderLayer`, `GroupedDynamicCausalConv`, `CandidateSelector`.
// Structure and tensor names mirror that file 1:1 so the checkpoint's
// safetensors keys land without remapping and so a divergence is
// traceable to a single line.
//
// What DFlash 2 adds over DFlash 1 (`DFlashDraftModel.swift`):
//
//  1. **Two-tap grouped dynamic causal convolutions** around each
//     sublayer. `attention_conv` / `mlp_conv` generate a per-token
//     kernel from the token itself (`kernel_projection`) and add it to a
//     learned `base_kernel`. This is what keeps draft quality from
//     decaying toward the end of a block.
//  2. **A candidate-path selector.** Instead of committing to the
//     per-position argmax, the drafter keeps `selector_top_k`
//     candidates per position and traces ONE coherent path through them
//     with a low-rank bigram score
//     (`predecessor_codebook · hidden · successor_codebook`). The path
//     is what gets verified.
//  3. **A real drafter KV cache.** DFlash 1's Swift port recomputed the
//     target-context projection every round; DFlash 2 accumulates it in
//     the drafter's own (sliding) cache, so a round only pays for the
//     newly committed positions.
//
// The drafter ships neither an embedding nor an LM head — it consumes
// the target's `embed_tokens` at the input and the target's `lm_head` at
// the output, both reached through ``TokenEmbedderModel``. The target is
// passed per-call rather than stored, so the drafter's own parameter
// tree stays exactly the checkpoint's tensors.

import Foundation
import MLX
import MLXNN
import MLXRandom

// MARK: - Configuration

/// The `dflash_config` object inside a DFlash 2 `config.json`.
public struct DFlash2InnerConfig: Codable, Sendable {
    public let blockSize: Int
    public let maskTokenId: Int
    public let targetLayerIds: [Int]
    public let convKernelSize: Int
    public let convGroupSize: Int
    public let selectorRank: Int
    public let selectorTopK: Int
    public let inputEmbeddingScale: Float
    public let outputMultiplier: Float
    public let finalLogitSoftcapping: Float?

    enum CodingKeys: String, CodingKey {
        case blockSize = "block_size"
        case maskTokenId = "mask_token_id"
        case targetLayerIds = "target_layer_ids"
        case convKernelSize = "conv_kernel_size"
        case convGroupSize = "conv_group_size"
        case selectorRank = "selector_rank"
        case selectorTopK = "selector_top_k"
        case inputEmbeddingScale = "input_embedding_scale"
        case outputMultiplier = "output_multiplier"
        case finalLogitSoftcapping = "final_logit_softcapping"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.blockSize = try c.decodeIfPresent(Int.self, forKey: .blockSize) ?? 16
        self.maskTokenId = try c.decode(Int.self, forKey: .maskTokenId)
        self.targetLayerIds = try c.decode([Int].self, forKey: .targetLayerIds)
        self.convKernelSize = try c.decodeIfPresent(Int.self, forKey: .convKernelSize) ?? 0
        self.convGroupSize = try c.decodeIfPresent(Int.self, forKey: .convGroupSize) ?? 0
        self.selectorRank = try c.decodeIfPresent(Int.self, forKey: .selectorRank) ?? 0
        self.selectorTopK = try c.decodeIfPresent(Int.self, forKey: .selectorTopK) ?? 0
        self.inputEmbeddingScale =
            try c.decodeIfPresent(Float.self, forKey: .inputEmbeddingScale) ?? 1.0
        self.outputMultiplier = try c.decodeIfPresent(Float.self, forKey: .outputMultiplier) ?? 1.0
        self.finalLogitSoftcapping = try c.decodeIfPresent(
            Float.self, forKey: .finalLogitSoftcapping)
    }
}

/// Full DFlash 2 drafter configuration.
///
/// Mirrors `load_draft` in the reference `model_mlx.py`, including its
/// fallbacks: `block_size` may live either in `dflash_config` or at the
/// top level, `rope_theta` may live under `rope_parameters`, and
/// `layer_types` defaults to all-`full_attention`.
public struct DFlash2Configuration: Sendable {
    public let hiddenSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let headDim: Int
    public let intermediateSize: Int
    public let vocabSize: Int
    public let rmsNormEps: Float
    public let ropeTheta: Float
    public let maxPositionEmbeddings: Int
    public let layerTypes: [String]
    public let slidingWindow: Int?
    /// `nil` means "follow the layer's own sliding flag", matching the
    /// reference's `is_causal = self.is_sliding if config.is_causal is None`.
    /// DFlash 2 checkpoints ship an explicit `false` — the block is drafted
    /// bidirectionally, which is the whole point of block diffusion.
    public let isCausal: Bool?
    public let dflash: DFlash2InnerConfig

    public var blockSize: Int { dflash.blockSize }
    public var maskTokenId: Int { dflash.maskTokenId }
    public var targetLayerIds: [Int] { dflash.targetLayerIds }

    /// True when the checkpoint carries the DFlash 2 selector + conv
    /// tensors. A DFlash 1 checkpoint decoded through this type has both
    /// counts at zero and must not be run on this path.
    public var isDFlash2: Bool {
        dflash.selectorRank > 0 && dflash.selectorTopK > 0 && dflash.convKernelSize > 0
    }

    /// Decode from raw `config.json` contents.
    public init(json root: [String: Any]) throws {
        func int(_ key: String) throws -> Int {
            guard let v = root[key] as? Int else {
                throw DFlash2LoadError.missingConfigKey(key)
            }
            return v
        }
        guard let dflashRaw = root["dflash_config"] else {
            throw DFlash2LoadError.missingConfigKey("dflash_config")
        }
        let dflashData = try JSONSerialization.data(withJSONObject: dflashRaw)
        var inner = try JSONDecoder().decode(DFlash2InnerConfig.self, from: dflashData)

        self.hiddenSize = try int("hidden_size")
        self.numHiddenLayers = try int("num_hidden_layers")
        self.numAttentionHeads = try int("num_attention_heads")
        self.numKeyValueHeads = (root["num_key_value_heads"] as? Int) ?? self.numAttentionHeads
        self.headDim = (root["head_dim"] as? Int) ?? (self.hiddenSize / self.numAttentionHeads)
        self.intermediateSize = try int("intermediate_size")
        self.vocabSize = try int("vocab_size")
        self.rmsNormEps = ((root["rms_norm_eps"] as? NSNumber)?.floatValue) ?? 1e-6
        self.maxPositionEmbeddings = (root["max_position_embeddings"] as? Int) ?? 131_072

        // `rope_parameters` (transformers ≥5) or the legacy flat key.
        let ropeObject = (root["rope_parameters"] as? [String: Any])
            ?? (root["rope_scaling"] as? [String: Any])
        self.ropeTheta =
            ((root["rope_theta"] as? NSNumber)?.floatValue)
            ?? ((ropeObject?["rope_theta"] as? NSNumber)?.floatValue)
            ?? 10_000.0

        let declaredTypes = root["layer_types"] as? [String]
        let types = declaredTypes ?? Array(repeating: "full_attention", count: self.numHiddenLayers)
        guard types.count == self.numHiddenLayers else {
            throw DFlash2LoadError.layerTypeCountMismatch(
                types.count, self.numHiddenLayers)
        }
        let unknown = Set(types).subtracting(["full_attention", "sliding_attention"])
        guard unknown.isEmpty else {
            throw DFlash2LoadError.unsupportedLayerTypes(unknown.sorted())
        }
        self.layerTypes = types
        self.slidingWindow = root["sliding_window"] as? Int
        if types.contains("sliding_attention"), self.slidingWindow == nil {
            throw DFlash2LoadError.missingConfigKey("sliding_window")
        }
        self.isCausal = root["is_causal"] as? Bool

        // `block_size` at the top level wins only when `dflash_config`
        // omitted it — matching the reference's `dflash.get("block_size",
        // cfg.get("block_size", 16))` precedence.
        if (try? JSONSerialization.data(withJSONObject: dflashRaw)) != nil,
            (dflashRaw as? [String: Any])?["block_size"] == nil,
            let top = root["block_size"] as? Int
        {
            inner = inner.replacingBlockSize(top)
        }
        self.dflash = inner
    }
}

extension DFlash2InnerConfig {
    fileprivate func replacingBlockSize(_ newValue: Int) -> DFlash2InnerConfig {
        // Codable-only struct: rebuild through JSON so the single
        // memberwise path stays the decoder.
        var dict: [String: Any] = [
            "block_size": newValue,
            "mask_token_id": maskTokenId,
            "target_layer_ids": targetLayerIds,
            "conv_kernel_size": convKernelSize,
            "conv_group_size": convGroupSize,
            "selector_rank": selectorRank,
            "selector_top_k": selectorTopK,
            "input_embedding_scale": inputEmbeddingScale,
            "output_multiplier": outputMultiplier,
        ]
        if let finalLogitSoftcapping {
            dict["final_logit_softcapping"] = finalLogitSoftcapping
        }
        let data = try! JSONSerialization.data(withJSONObject: dict)
        return try! JSONDecoder().decode(DFlash2InnerConfig.self, from: data)
    }
}

// MARK: - Grouped dynamic causal convolution

/// Two-tap depthwise causal convolution whose kernel is partly learned
/// and partly generated from the token being convolved.
///
/// Port of `GroupedDynamicCausalConv`. `base_kernel` is
/// `(2, kernel_size, hidden)` — two independent kernels, one used by
/// `prepare` (before the sublayer) and one by `finish` (after it) —
/// and `kernel_projection` emits `2 * kernel_size * groups` values per
/// token, where `groups = hidden / conv_group_size`. Channels inside a
/// group share a dynamic tap; the base kernel is per-channel.
///
/// The convolution carries no state between calls. Each drafting round
/// re-runs the whole block from its anchor token, so position 0 always
/// legitimately sees zeros to its left — there is no cross-round conv
/// cache to maintain (and the reference keeps none).
final class GroupedDynamicCausalConv: Module {
    let kernelSize: Int
    let groupSize: Int

    @ParameterInfo(key: "base_kernel") var baseKernel: MLXArray
    @ModuleInfo(key: "kernel_projection") var kernelProjection: Linear

    init(hiddenSize: Int, kernelSize: Int, groupSize: Int) {
        self.kernelSize = kernelSize
        self.groupSize = groupSize
        let groups = hiddenSize / groupSize
        self._baseKernel.wrappedValue = MLXArray.zeros([2, kernelSize, hiddenSize])
        self._kernelProjection.wrappedValue = Linear(
            hiddenSize, 2 * kernelSize * groups, bias: false)
        super.init()
    }

    /// `(B, L, hidden)` → convolved input for the sublayer, plus the
    /// dynamic kernel that ``finish(_:dynamic:)`` will apply to the
    /// sublayer's output.
    func prepare(_ hidden: MLXArray) -> (MLXArray, MLXArray) {
        let groups = hidden.dim(-1) / groupSize
        let b = hidden.dim(0)
        let l = hidden.dim(1)
        let dynamic = kernelProjection(hidden)
            .reshaped(b, l, 2, kernelSize, groups)
        let convolved = Self.convolve(
            hidden: hidden,
            dynamic: dynamic[0..., 0..., 0, 0..., 0...],
            base: baseKernel[0],
            groupSize: groupSize)
        return (convolved, dynamic[0..., 0..., 1, 0..., 0...])
    }

    func finish(_ hidden: MLXArray, dynamic: MLXArray) -> MLXArray {
        Self.convolve(
            hidden: hidden, dynamic: dynamic, base: baseKernel[1], groupSize: groupSize)
    }

    /// `_grouped_dynamic_convolve`. `base` is `(kernel_size, hidden)`,
    /// `dynamic` is `(B, L, kernel_size, groups)`.
    static func convolve(
        hidden: MLXArray, dynamic: MLXArray, base: MLXArray, groupSize: Int
    ) -> MLXArray {
        let batch = hidden.dim(0)
        let length = hidden.dim(1)
        let hiddenSize = hidden.dim(2)
        let groups = hiddenSize / groupSize
        let taps = base.dim(0)

        let blocks = hidden.reshaped(batch, length, groups, groupSize)
        let dyn = dynamic.reshaped(batch, length, taps, groups, 1)

        var output = MLXArray.zeros(like: blocks)
        for offset in 0 ..< taps {
            // Causal shift: tap `offset` reads the token `offset` steps
            // back, with zeros before the start of the block.
            let values: MLXArray
            if offset == 0 {
                values = blocks
            } else {
                let shifted = blocks[0..., ..<(length - offset), 0..., 0...]
                let pad = MLXArray.zeros(
                    [batch, offset, groups, groupSize], dtype: blocks.dtype)
                values = concatenated([pad, shifted], axis: 1)
            }
            let kernel = base[offset].reshaped(1, 1, groups, groupSize).asType(hidden.dtype)
            output = output + kernel * values
            output = output + dyn[0..., 0..., offset, 0..., 0...] * values
        }
        return output.reshaped(batch, length, hiddenSize)
    }
}

// MARK: - Attention

/// Drafter attention. Queries come from the drafted block; keys/values
/// come from BOTH the cached target-context projection and the block
/// itself.
///
/// Port of `DFlashAttention.__call__`. The block's own K/V are appended
/// transiently and never enter the cache — only the target-context rows
/// are persisted, because those are the ones that stay valid after the
/// target verifies and the block is discarded.
final class DFlash2Attention: Module {
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let scale: Float
    let isSliding: Bool
    let slidingWindow: Int?
    let isCausal: Bool

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    init(_ config: DFlash2Configuration, layerIndex: Int) {
        let dim = config.hiddenSize
        self.numHeads = config.numAttentionHeads
        self.numKVHeads = config.numKeyValueHeads
        self.headDim = config.headDim
        self.scale = pow(Float(config.headDim), -0.5)
        self.isSliding = config.layerTypes[layerIndex] == "sliding_attention"
        self.slidingWindow = self.isSliding ? config.slidingWindow : nil
        self.isCausal = config.isCausal ?? self.isSliding

        self._wq.wrappedValue = Linear(dim, numHeads * headDim, bias: false)
        self._wk.wrappedValue = Linear(dim, numKVHeads * headDim, bias: false)
        self._wv.wrappedValue = Linear(dim, numKVHeads * headDim, bias: false)
        self._wo.wrappedValue = Linear(numHeads * headDim, dim, bias: false)
        self._qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        self._kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, context: MLXArray, rope: RoPE, cache: KVCache
    ) -> MLXArray {
        let b = x.dim(0)
        let l = x.dim(1)
        var xCtx = context
        var s = xCtx.dim(1)

        // A single context chunk longer than the window (long prompts)
        // is clipped from the front; the tokens dropped here are still
        // counted in the cache offset so RoPE positions stay absolute.
        if isSliding, let window = slidingWindow {
            let keepCtx = window - 1
            if s > keepCtx {
                let skip = s - keepCtx
                xCtx = xCtx[0..., skip..., 0...]
                s = xCtx.dim(1)
                cache.offsetForDFlash2 += skip
            }
        }

        var queries = wq(x).reshaped(b, l, numHeads, headDim)
        queries = qNorm(queries).transposed(0, 2, 1, 3)

        var ctxKeys = wk(xCtx).reshaped(b, s, numKVHeads, headDim)
        ctxKeys = kNorm(ctxKeys).transposed(0, 2, 1, 3)
        let ctxValues = wv(xCtx).reshaped(b, s, numKVHeads, headDim).transposed(0, 2, 1, 3)

        var propKeys = wk(x).reshaped(b, l, numKVHeads, headDim)
        propKeys = kNorm(propKeys).transposed(0, 2, 1, 3)
        let propValues = wv(x).reshaped(b, l, numKVHeads, headDim).transposed(0, 2, 1, 3)

        let offset = cache.offset
        queries = rope(queries, offset: offset + s)
        ctxKeys = rope(ctxKeys, offset: offset)
        propKeys = rope(propKeys, offset: offset + s)

        var (keys, values) = cache.update(keys: ctxKeys, values: ctxValues)
        // A wrapped ring buffer comes back in rotation order, which would
        // misalign the position-derived mask below. The rotating cache
        // exposes a temporally-ordered view for exactly this case.
        if let rotating = cache as? RotatingKVCache,
            let ordered = rotating.temporallyOrderedKV()
        {
            keys = ordered.keys
            values = ordered.values
        }
        let ctxLen = keys.dim(2)
        keys = concatenated([keys, propKeys], axis: 2)
        values = concatenated([values, propValues], axis: 2)

        let mask = Self.mask(
            queryLength: l, contextLength: ctxLen,
            isCausal: isCausal, slidingWindow: isSliding ? slidingWindow : nil)

        let output = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: scale, mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(b, l, -1)
        return wo(output)
    }

    /// Boolean keep-mask over `(L, ctxLen + L)`.
    ///
    /// Context columns follow the sliding window; block columns are open
    /// in both directions unless the layer is causal. That bidirectional
    /// block is the diffusion part — every position in the block sees
    /// every other one, which is what lets a single forward draft all of
    /// them at once.
    static func mask(
        queryLength l: Int, contextLength ctxLen: Int, isCausal: Bool, slidingWindow: Int?
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        if slidingWindow == nil && !isCausal {
            return .none
        }
        let query = (MLXArray(Int32(ctxLen)) + MLXArray(0 ..< Int32(l))).reshaped(l, 1)
        let key = MLXArray(0 ..< Int32(ctxLen + l)).reshaped(1, ctxLen + l)
        if let window = slidingWindow {
            let context = (key .< Int32(ctxLen)) .&& ((query - key) .< Int32(window))
            var block = key .>= Int32(ctxLen)
            if isCausal {
                block = block .&& (key .<= query)
            }
            return .array(context .|| block)
        }
        return .array(key .<= query)
    }
}

extension KVCache {
    /// Settable offset used by the DFlash 2 sliding-context clip.
    ///
    /// ``KVCache`` only publishes `offset` as get-only; every concrete
    /// cache in this package derives from ``BaseKVCache``, where it is a
    /// stored var. The clip has to advance it so RoPE positions stay
    /// absolute after front-truncating an over-long context chunk.
    var offsetForDFlash2: Int {
        get { offset }
        nonmutating set {
            if let base = self as? BaseKVCache {
                base.offset = newValue
            }
        }
    }
}

// MARK: - Decoder layer

final class DFlash2DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: DFlash2Attention
    let mlp: DFlash2MLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "attention_conv") var attentionConv: GroupedDynamicCausalConv
    @ModuleInfo(key: "mlp_conv") var mlpConv: GroupedDynamicCausalConv

    init(_ config: DFlash2Configuration, layerIndex: Int) {
        self._attention.wrappedValue = DFlash2Attention(config, layerIndex: layerIndex)
        self.mlp = DFlash2MLP(
            dimensions: config.hiddenSize, hiddenDimensions: config.intermediateSize)
        self._inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._attentionConv.wrappedValue = GroupedDynamicCausalConv(
            hiddenSize: config.hiddenSize,
            kernelSize: config.dflash.convKernelSize,
            groupSize: config.dflash.convGroupSize)
        self._mlpConv.wrappedValue = GroupedDynamicCausalConv(
            hiddenSize: config.hiddenSize,
            kernelSize: config.dflash.convKernelSize,
            groupSize: config.dflash.convGroupSize)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, context: MLXArray, rope: RoPE, cache: KVCache
    ) -> MLXArray {
        var residual = x
        var (h, kernel) = attentionConv.prepare(inputLayerNorm(x))
        h = residual
            + attentionConv.finish(
                attention(h, context: context, rope: rope, cache: cache), dynamic: kernel)

        residual = h
        let (m, mlpKernel) = mlpConv.prepare(postAttentionLayerNorm(h))
        return residual + mlpConv.finish(mlp(m), dynamic: mlpKernel)
    }
}

/// SiLU-gated MLP, identical to `Qwen3MLP`.
final class DFlash2MLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear

    init(dimensions: Int, hiddenDimensions: Int) {
        self._gate.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        self._down.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        self._up.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

// MARK: - Candidate selector

/// Result of one drafting round.
public struct DFlash2Proposal {
    /// `(B, L)` — the traced path, one token per block position.
    public let tokens: MLXArray
    /// `(B, L, top_k)` — the candidate set each path token was chosen from.
    /// Needed by the sampled accept path to recover q(x) for a draft token.
    public let candidates: MLXArray
    /// `(B, L, top_k)` draft probabilities over `candidates`, or `nil`
    /// under greedy decoding where no q is needed.
    public let probabilities: MLXArray?
}

/// Traces one coherent path through the per-position candidate sets.
///
/// Port of `CandidateSelector.select`. Per position the score of a
/// candidate is its own logit (`unary`) plus a low-rank bigram edge
/// against the token actually chosen at the previous position:
///
///     edge(prev, cand) = Σ  predecessor[prev] · hidden · successor[cand]
///
/// Because the edge depends on the previous *choice*, the trace is
/// sequential over the block — but each step is a `(K, rank)` reduction,
/// not a vocabulary-wide one, so the whole trace costs far less than the
/// single LM-head projection that produced the logits.
final class DFlash2CandidateSelector: Module {
    let topK: Int

    @ModuleInfo(key: "predecessor_codebook") var predecessorCodebook: Embedding
    @ModuleInfo(key: "successor_codebook") var successorCodebook: Embedding
    @ModuleInfo(key: "hidden_projection") var hiddenProjection: Linear

    init(_ config: DFlash2Configuration) {
        self.topK = config.dflash.selectorTopK
        self._predecessorCodebook.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.dflash.selectorRank)
        self._successorCodebook.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.dflash.selectorRank)
        self._hiddenProjection.wrappedValue = Linear(
            config.hiddenSize, config.dflash.selectorRank, bias: false)
        super.init()
    }

    /// - Parameters:
    ///   - hidden: `(B, L, hidden)` post-norm drafter output.
    ///   - logits: `(B, L, vocab)` draft logits for the same positions.
    ///   - anchorIds: `(B,)` the token immediately before the block —
    ///     the predecessor the first edge scores against.
    ///   - temperature: `0` traces the argmax path; `> 0` samples each
    ///     step from the candidate scores and returns the q distribution.
    func select(
        hidden: MLXArray, logits: MLXArray, anchorIds: MLXArray, temperature: Float
    ) -> DFlash2Proposal {
        let vocab = logits.dim(-1)
        let length = hidden.dim(1)
        // argPartition places the K largest in the trailing K slots.
        let candidates = argPartition(logits, kth: vocab - topK, axis: -1)[
            .ellipsis, (vocab - topK)...]
        let unary = takeAlong(logits, candidates, axis: -1)
        let projected = hiddenProjection(hidden)

        var predecessor = anchorIds
        var path: [MLXArray] = []
        var qRows: [MLXArray] = []
        path.reserveCapacity(length)

        for position in 0 ..< length {
            let prev = predecessorCodebook(predecessor).expandedDimensions(axis: 1)
            let h = projected[0..., position, 0...].expandedDimensions(axis: 1)
            let succ = successorCodebook(candidates[0..., position, 0...])
            let edges = (prev * h * succ).sum(axis: -1)
            let scores = unary[0..., position, 0...] + edges

            let selected: MLXArray
            if temperature > 0 {
                let q = DFlash2Sampling.probabilities(
                    scores.expandedDimensions(axis: 1), temperature: temperature)[0..., 0, 0...]
                selected = DFlash2Sampling.sample(probabilities: q)
                qRows.append(q)
            } else {
                selected = argMax(scores, axis: -1)
            }
            predecessor = takeAlong(
                candidates[0..., position, 0...], selected.expandedDimensions(axis: -1), axis: -1
            )[0..., 0]
            path.append(predecessor)
        }

        return DFlash2Proposal(
            tokens: stacked(path, axis: 1),
            candidates: candidates,
            probabilities: qRows.isEmpty ? nil : stacked(qRows, axis: 1))
    }
}

// MARK: - Draft model

/// DFlash 2 block-diffusion drafter.
public final class DFlash2DraftModel: Module, @unchecked Sendable {

    public let config: DFlash2Configuration

    fileprivate let layers: [DFlash2DecoderLayer]
    let norm: RMSNorm

    /// Projection from `len(target_layer_ids) * hidden` → `hidden`,
    /// applied once to the concatenated target context per round.
    @ModuleInfo(key: "fc") var fc: Linear
    @ModuleInfo(key: "hidden_norm") var hiddenNorm: RMSNorm
    @ModuleInfo(key: "candidate_selector") var candidateSelector: DFlash2CandidateSelector

    private let rope: RoPE

    public init(_ config: DFlash2Configuration) {
        self.config = config
        self.layers = (0 ..< config.numHiddenLayers).map {
            DFlash2DecoderLayer(config, layerIndex: $0)
        }
        self.norm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._fc.wrappedValue = Linear(
            config.targetLayerIds.count * config.hiddenSize, config.hiddenSize, bias: false)
        self._hiddenNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._candidateSelector.wrappedValue = DFlash2CandidateSelector(config)
        self.rope = RoPE(
            dimensions: config.headDim, traditional: false, base: config.ropeTheta, scale: 1)
        super.init()
    }

    /// Fresh per-generation drafter cache — one entry per layer, sliding
    /// where the layer is.
    public func makeCache() -> [KVCache] {
        config.layerTypes.map { type in
            if type == "sliding_attention", let window = config.slidingWindow {
                return RotatingKVCache(maxSize: window - 1, keep: 0)
            }
            return KVCacheSimple()
        }
    }

    /// Run the drafter backbone.
    ///
    /// - Parameters:
    ///   - inputs: `(B, block)` token IDs — `[anchor, mask, mask, …]`.
    ///   - targetHidden: `(B, ctx, len(target_layer_ids) * hidden)` target
    ///     hidden states for the newly committed positions.
    ///   - cache: the drafter's own cache from ``makeCache()``.
    ///   - embedder: the target, whose embedding the drafter shares.
    ///   - logitsStart: drop this many leading positions before the final
    ///     norm. `1` skips the anchor, whose token is already known.
    public func hiddenStates(
        inputs: MLXArray,
        targetHidden: MLXArray,
        cache: [KVCache],
        embedder: any TokenEmbedderModel,
        logitsStart: Int = 0
    ) -> MLXArray {
        var h = embedder.embed(inputs)
        if config.dflash.inputEmbeddingScale != 1.0 {
            h = h * config.dflash.inputEmbeddingScale
        }
        let context = hiddenNorm(fc(targetHidden.asType(h.dtype)))
        for (layer, c) in zip(layers, cache) {
            h = layer(h, context: context, rope: rope, cache: c)
        }
        if logitsStart > 0 {
            h = h[0..., logitsStart..., 0...]
        }
        return norm(h)
    }

    /// Project drafter hidden states through the target's LM head.
    public func computeLogits(
        _ hidden: MLXArray, embedder: any TokenEmbedderModel
    ) -> MLXArray {
        var logits = embedder.projectToLogits(hidden)
        if config.dflash.outputMultiplier != 1.0 {
            logits = logits * config.dflash.outputMultiplier
        }
        if let cap = config.dflash.finalLogitSoftcapping, cap > 0 {
            logits = tanh(logits / cap) * cap
        }
        return logits
    }

    /// Intermediate activations of the first layer plus every layer's
    /// output. Exists so a port divergence can be localised to a stage
    /// instead of inferred from the final tokens — see
    /// `DFlash2StageProbeTests`.
    public struct StageProbe {
        public let context: MLXArray
        public let layer0InputNorm: MLXArray
        public let layer0ConvInput: MLXArray
        public let layer0Kernel: MLXArray
        public let layer0Attention: MLXArray
        public let layer0AttentionConv: MLXArray
        public let layer0PostAttention: MLXArray
        public let layer0MLPInput: MLXArray
        public let perLayer: [MLXArray]
        public let final: MLXArray
    }

    /// Re-runs the backbone recording every intermediate. Diagnostic
    /// only — ``hiddenStates(inputs:targetHidden:cache:embedder:logitsStart:)``
    /// is the path decode uses, and this one duplicates its body rather
    /// than instrumenting it so the hot path carries no branches.
    public func probeStages(
        embedded: MLXArray, targetHidden: MLXArray, cache: [KVCache]
    ) -> StageProbe {
        let context = hiddenNorm(fc(targetHidden.asType(embedded.dtype)))
        var h = embedded
        var perLayer: [MLXArray] = []

        var l0InputNorm = h
        var l0ConvInput = h
        var l0Kernel = h
        var l0Attention = h
        var l0AttentionConv = h
        var l0PostAttention = h
        var l0MLPInput = h

        for (index, layer) in layers.enumerated() {
            if index == 0 {
                l0InputNorm = layer.inputLayerNorm(h)
                let (convIn, kernel) = layer.attentionConv.prepare(l0InputNorm)
                l0ConvInput = convIn
                l0Kernel = kernel
                l0Attention = layer.attention(
                    convIn, context: context, rope: rope, cache: cache[index])
                l0AttentionConv = layer.attentionConv.finish(l0Attention, dynamic: kernel)
                l0PostAttention = h + l0AttentionConv
                let (mlpIn, mlpKernel) = layer.mlpConv.prepare(
                    layer.postAttentionLayerNorm(l0PostAttention))
                l0MLPInput = mlpIn
                h = l0PostAttention + layer.mlpConv.finish(layer.mlp(mlpIn), dynamic: mlpKernel)
            } else {
                h = layer(h, context: context, rope: rope, cache: cache[index])
            }
            perLayer.append(h)
        }

        return StageProbe(
            context: context,
            layer0InputNorm: l0InputNorm,
            layer0ConvInput: l0ConvInput,
            layer0Kernel: l0Kernel,
            layer0Attention: l0Attention,
            layer0AttentionConv: l0AttentionConv,
            layer0PostAttention: l0PostAttention,
            layer0MLPInput: l0MLPInput,
            perLayer: perLayer,
            final: norm(h))
    }

    /// One drafting round: backbone forward, LM head, candidate-path trace.
    public func propose(
        inputs: MLXArray,
        targetHidden: MLXArray,
        cache: [KVCache],
        embedder: any TokenEmbedderModel,
        temperature: Float,
        logitsStart: Int = 1
    ) -> DFlash2Proposal {
        let hidden = hiddenStates(
            inputs: inputs, targetHidden: targetHidden, cache: cache,
            embedder: embedder, logitsStart: logitsStart)
        let logits = computeLogits(hidden, embedder: embedder)
        return candidateSelector.select(
            hidden: hidden, logits: logits, anchorIds: inputs[0..., 0],
            temperature: temperature)
    }
}
