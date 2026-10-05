// NemotronLabs VoiceChat — the EAR-TTS transformer and RVQ code generation.
//
// Port of `RVQEARTTSModel` in `mlx_vlm/models/nemotron_voicechat/tts.py`.
//
// The backbone is Gemma3-shaped (28 layers, 1152 hidden, q/k norms, sliding
// window 7500 with pattern 6) but is NOT the Gemma3 module already in
// MLXLLM: this checkpoint DELETES `embed_tokens` (vocab_size 1 — inputs are
// always embeddings, never token ids), so the layer stack is spelled out here
// against the bundle's own keys (`backbone.layers.N.*`, `backbone.norm`).
//
// 🚨 Norm convention: every norm here is `VoiceChatOffsetRMSNorm` (1 + weight).
// The nemotron_h STT backbone next door uses plain RMSNorm. One model, two
// conventions — the mistake that empties output while every structural check
// stays green.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MLXRandom

/// Gemma3-style attention with per-head q/k norms and alternating
/// global / sliding-window RoPE bases.
public class VoiceChatTTSAttention: Module {
    let nHeads: Int
    let nKVHeads: Int
    let headDim: Int
    let repeats: Int
    let attnScale: Float
    let rope: RoPE

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: VoiceChatOffsetRMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: VoiceChatOffsetRMSNorm

    public init(config: VoiceChatTTSConfiguration, isSliding: Bool) {
        let hidden = config.hiddenSize
        self.nHeads = config.numAttentionHeads
        self.nKVHeads = config.numKeyValueHeads
        self.headDim = config.headDim
        self.repeats = nHeads / nKVHeads
        self.attnScale = Float(
            1.0 / Foundation.sqrt(Double(config.queryPreAttnScalar ?? 256.0)))
        let eps = config.rmsNormEps ?? 1e-6
        self.rope = RoPE(
            dimensions: headDim, traditional: false,
            base: isSliding
                ? (config.ropeLocalBaseFreq ?? 10_000.0)
                : (config.ropeGlobalBaseFreq ?? 1_000_000.0))
        self._qProj.wrappedValue = Linear(hidden, nHeads * headDim, bias: false)
        self._kProj.wrappedValue = Linear(hidden, nKVHeads * headDim, bias: false)
        self._vProj.wrappedValue = Linear(hidden, nKVHeads * headDim, bias: false)
        self._oProj.wrappedValue = Linear(nHeads * headDim, hidden, bias: false)
        self._qNorm.wrappedValue = VoiceChatOffsetRMSNorm(dimensions: headDim, eps: eps)
        self._kNorm.wrappedValue = VoiceChatOffsetRMSNorm(dimensions: headDim, eps: eps)
    }

    public func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let B = x.dim(0), L = x.dim(1)
        var q = qProj(x).reshaped(B, L, nHeads, -1).transposed(0, 2, 1, 3)
        var k = kProj(x).reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)
        var v = vProj(x).reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)

        q = qNorm(q)
        k = kNorm(k)

        let offset = cache?.offset ?? 0
        q = rope(q, offset: offset)
        k = rope(k, offset: offset)
        if let cache {
            (k, v) = cache.update(keys: k, values: v)
        }

        let out = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: attnScale, mask: mask)
        return oProj(out.transposed(0, 2, 1, 3).reshaped(B, L, -1))
    }
}

public class VoiceChatTTSBlock: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: VoiceChatTTSAttention
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: VoiceChatOffsetRMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: VoiceChatOffsetRMSNorm
    @ModuleInfo(key: "mlp") var mlp: VoiceChatTTSMLP
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayerNorm: VoiceChatOffsetRMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayerNorm:
        VoiceChatOffsetRMSNorm

    public init(config: VoiceChatTTSConfiguration, isSliding: Bool) {
        let hidden = config.hiddenSize
        let eps = config.rmsNormEps ?? 1e-6
        self._selfAttn.wrappedValue = VoiceChatTTSAttention(config: config, isSliding: isSliding)
        self._inputLayerNorm.wrappedValue = VoiceChatOffsetRMSNorm(dimensions: hidden, eps: eps)
        self._postAttentionLayerNorm.wrappedValue = VoiceChatOffsetRMSNorm(
            dimensions: hidden, eps: eps)
        self._mlp.wrappedValue = VoiceChatTTSMLP(
            hiddenSize: hidden, intermediateSize: config.intermediateSize)
        self._preFeedforwardLayerNorm.wrappedValue = VoiceChatOffsetRMSNorm(
            dimensions: hidden, eps: eps)
        self._postFeedforwardLayerNorm.wrappedValue = VoiceChatOffsetRMSNorm(
            dimensions: hidden, eps: eps)
    }

    public func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        var x =
            x
            + postAttentionLayerNorm(selfAttn(inputLayerNorm(x), mask: mask, cache: cache))
        x = x + postFeedforwardLayerNorm(mlp(preFeedforwardLayerNorm(x)))
        return x
    }
}

/// The 28-layer speech transformer. Embeddings in, hidden out — this
/// checkpoint has no embedding table at all.
public class VoiceChatTTSBackbone: Module {
    public let config: VoiceChatTTSConfiguration
    public let slidingWindowPattern: Int

    @ModuleInfo(key: "layers") var layers: [VoiceChatTTSBlock]
    @ModuleInfo(key: "norm") var norm: VoiceChatOffsetRMSNorm

    /// `VOICECHAT_LAYER_DUMP=<dir>` writes each layer's single-step output for
    /// stage-by-stage comparison against the reference implementation.
    static let layerDumpDirectory = ProcessInfo.processInfo.environment["VOICECHAT_LAYER_DUMP"]

    /// Diagnostic arm: use a plain cache for the sliding layers too.
    static let useSimpleCacheEverywhere =
        ProcessInfo.processInfo.environment["VOICECHAT_SIMPLE_CACHE"] == "1"

    public init(_ config: VoiceChatTTSConfiguration) {
        self.config = config
        let pattern = config.slidingWindowPattern ?? 6
        self.slidingWindowPattern = pattern
        self._layers.wrappedValue = (0 ..< config.numHiddenLayers).map { idx in
            // Global every `pattern`-th layer (1-indexed), sliding otherwise.
            VoiceChatTTSBlock(config: config, isSliding: (idx + 1) % pattern != 0)
        }
        self._norm.wrappedValue = VoiceChatOffsetRMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps ?? 1e-6)
    }

    public func callAsFunction(_ inputsEmbeds: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = inputsEmbeds
        // Global layers see the whole history; sliding layers get a windowed
        // mask sized to the rotating cache. Masks are built from the matching
        // cache kind so an offset from a rotating buffer cannot be applied to
        // a full one.
        //
        // 🚨 A SINGLE-token step needs NO mask at all: the one new query may
        // attend to every cached key (the rotating buffer is itself the
        // window). Asking for a causal mask here is not a harmless extra — it
        // misaligns the query against a cache that already holds the history,
        // and the result is a hidden state that is close to right and
        // therefore renders as fluent-sounding babble instead of words. The
        // whole-sequence warmup path was exact while this step was not.
        let isSingleStep = h.dim(1) == 1
        let globalMask: MLXFast.ScaledDotProductAttentionMaskMode =
            isSingleStep
            ? .none
            : createAttentionMask(h: h, cache: cache.map { $0[slidingWindowPattern - 1] } ?? nil)
        let slidingMask: MLXFast.ScaledDotProductAttentionMaskMode =
            isSingleStep || slidingWindowPattern <= 1
            ? .none
            : createAttentionMask(
                h: h, cache: cache?.first, windowSize: config.slidingWindow)

        for (i, layer) in layers.enumerated() {
            let isGlobal = (i % slidingWindowPattern == slidingWindowPattern - 1)
            h = layer(h, mask: isGlobal ? globalMask : slidingMask, cache: cache?[i])
            if let dir = Self.layerDumpDirectory, isSingleStep {
                MLX.eval(h)
                let flat = h.asType(.float32).reshaped([-1]).asArray(Float.self)
                flat.withUnsafeBufferPointer {
                    try? Data(buffer: $0).write(
                        to: URL(fileURLWithPath: dir).appendingPathComponent(
                            String(format: "mine-layer-%02d.f32", i)))
                }
            }
        }
        return norm(h)
    }

    /// Per-layer caches: global layers keep everything, sliding layers keep a
    /// 7500-frame rotating window (`keep: 0` — no attention sink here).
    public func makeCache() -> [KVCache] {
        (0 ..< config.numHiddenLayers).map { idx in
            // Both kinds are the same cache here: see VoiceChatKVCache for
            // why, and for the measurement that forced it.
            VoiceChatConcatKVCache()
        }
    }
}

/// One TTS step's outputs.
public struct VoiceChatTTSStepOutput {
    public let codes: MLXArray
    public let hiddenStates: MLXArray
}

/// The RVQ EAR-TTS model: code embedding, text conditioning, gated fusion,
/// the transformer, and iterative mixture-of-Gaussians RVQ code generation.
public class VoiceChatRVQEARTTSModel: Module {
    public let config: VoiceChatTTSConfiguration

    @ModuleInfo(key: "backbone") public var backbone: VoiceChatTTSBackbone
    @ParameterInfo(key: "bos_emb") var bosEmb: MLXArray
    @ParameterInfo(key: "null_emb") var nullEmb: MLXArray
    @ModuleInfo(key: "embed_code") var embedCode: Linear
    @ModuleInfo(key: "embed_subword") public var embedSubword: VoiceChatCharAwareSubwordEncoder
    @ModuleInfo(key: "gated_fusion_audio_text") var gatedFusion: VoiceChatGatedFusion
    @ParameterInfo(key: "audio_prompt_projection_W") var audioPromptProjectionW: MLXArray
    @ModuleInfo(key: "mog_head") public var mogHead: VoiceChatMoGHead
    /// 🚨 RVQ codebook, read raw — must stay fp in every quantized bundle.
    @ParameterInfo(key: "rvq_embs") var rvqEmbs: MLXArray

    public init(_ config: VoiceChatTTSConfiguration) {
        self.config = config
        let hidden = config.hiddenSize
        self._backbone.wrappedValue = VoiceChatTTSBackbone(config)
        self._bosEmb.wrappedValue = MLXArray.zeros([hidden])
        self._nullEmb.wrappedValue = MLXArray.zeros([hidden])
        self._embedCode.wrappedValue = Linear(config.latentSize, hidden, bias: false)
        self._embedSubword.wrappedValue = VoiceChatCharAwareSubwordEncoder(
            config: config.characterEncoder
                ?? VoiceChatCharacterEncoderConfiguration(
                    hiddenSize: nil, intermediateSize: nil, numHiddenLayers: nil,
                    numAttentionHeads: nil, numKeyValueHeads: nil, headDim: nil,
                    rmsNormEps: nil, queryPreAttnScalar: nil, attnLogitSoftcapping: nil,
                    ropeBase: nil, charVocabSize: nil),
            outSize: hidden)
        self._gatedFusion.wrappedValue = VoiceChatGatedFusion(
            hiddenSize: hidden, numCodebooks: config.numQuantizers,
            eps: config.rmsNormEps ?? 1e-6)
        self._audioPromptProjectionW.wrappedValue = MLXArray.zeros([hidden, hidden])
        self._mogHead.wrappedValue = VoiceChatMoGHead(
            hiddenSize: hidden, outSize: config.latentSize,
            config: config.mogHead
                ?? VoiceChatMoGConfiguration(
                    intermediateSize: nil, lowRank: nil, minLogStd: nil,
                    numLayers: nil, numPredictions: nil, eps: nil))
        self._rvqEmbs.wrappedValue = MLXArray.zeros([
            config.numQuantizers, config.codebookSize, config.latentSize,
        ])
    }

    public func setVocabulary(_ vocabulary: [String: Int]) throws {
        try embedSubword.setVocabulary(vocabulary)
    }

    public func makeCache() -> [KVCache] { backbone.makeCache() }

    /// Depth-summed latent for a full RVQ code stack.
    ///
    /// An extra all-zero PAD row is appended to every codebook so the
    /// "not yet chosen" sentinel (index == codebookSize) contributes nothing —
    /// that is how partially-refined codes stay valid mid-iteration.
    public func depthSumEmbedding(_ code: MLXArray) -> MLXArray {
        let padding = MLXArray.zeros(
            [config.numQuantizers, 1, config.latentSize], dtype: rvqEmbs.dtype)
        let embeddings = MLX.concatenated([rvqEmbs, padding], axis: 1)
        var result = MLXArray.zeros(
            code.shape.dropLast() + [config.latentSize], dtype: embeddings.dtype)
        for idx in 0 ..< config.numQuantizers {
            result = result + embeddings[idx][code[.ellipsis, idx]]
        }
        return result
    }

    /// Text conditioning, doubled for classifier-free guidance (the second
    /// half is the NULL embedding — the unconditional branch).
    private func condition(
        subwordIds: MLXArray, subwordMask: MLXArray?, batchSize: Int, guidance: Bool
    ) -> MLXArray {
        var cond = embedSubword(subwordIds, mask: subwordMask)
        if guidance {
            // Built exactly as the reference does — double, then overwrite the
            // second half — rather than concatenating a broadcast view onto the
            // original. A broadcast is a stride-0 VIEW, and taking it straight
            // into the concatenated batch is where the unconditional row can
            // stop being independent of the conditional one. The guidance
            // subtraction (cond + s·(cond − uncond)) then amplifies whatever is
            // wrong there, which is why the conditional half matched at every
            // layer while the unconditional half collapsed.
            cond = MLX.concatenated([cond, cond], axis: 0)
            let nullShape = [batchSize, cond.dim(1), cond.dim(2)]
            let null = MLX.broadcast(nullEmb.asType(cond.dtype), to: nullShape)
            cond = MLX.concatenated([cond[0 ..< batchSize], null], axis: 0)
        }
        return cond
    }

    /// Prime the transformer with the speaker prompt so the first generated
    /// frame already sounds like the voice.
    ///
    /// Frames before the audio BOS carry the PROMPT LATENT (`Aria`) instead of
    /// code embeddings — that is where speaker identity enters, and it is why
    /// a custom voice is an 83 KiB latent rather than a retrain.
    public func warmup(
        code: MLXArray,
        subwordIds: MLXArray,
        subwordMask: MLXArray,
        audioMask: MLXArray,
        audioPromptLatent: MLXArray?,
        guidance: Bool = true,
        cache: [KVCache]? = nil
    ) -> (MLXArray, [KVCache]) {
        let shifted = MLX.concatenated(
            [MLXArray.zeros(like: code[0..., 0 ..< 1, 0...]), code[0..., ..<(code.dim(1) - 1), 0...]],
            axis: 1)
        var codeEmbed = embedCode(depthSumEmbedding(shifted))

        let previousAudio = MLX.concatenated(
            [
                MLXArray.zeros(like: audioMask[0..., 0 ..< 1]),
                audioMask[0..., ..<(audioMask.dim(1) - 1)],
            ], axis: 1)
        let bosMask = MLX.logicalAnd(audioMask, MLX.logicalNot(previousAudio))
        let preBOS = MLX.cumsum(bosMask.asType(.int32), axis: 1) .== MLXArray(Int32(0))

        // 🚨 The frozen prompt projection carries NO fan-in scaling, so it must
        // be normalised here or cloning is unusable.
        //
        // `audio_prompt_projection_W` is 1152x1152 with std 1.000149 — an
        // unscaled random normal, not a trained matrix (a trained one would
        // carry roughly 1/sqrt(1152) = 0.029). Applied raw it amplifies by
        // ~sqrt(hidden): measured against the checkpoint, a prompt frame comes
        // out at RMS 39.74 while the shipped `audio_prompt_latents.Aria` frames
        // it replaces sit at 1.69 — 23.5x hot. The backbone then gets prompt
        // conditioning far outside anything it saw in training, which is why
        // cloned voices changed timbre but stopped forming words.
        //
        // 1/sqrt(hidden) is the norm-preserving scale for a random projection
        // and lands prompt frames at RMS ~1.17, the same order as Aria. The
        // reference never reaches this branch (it always supplies the Aria
        // latent), so there is no upstream behaviour to match here — the scale
        // is justified by measurement, and `VMLX_VOICECHAT_PROMPT_SCALE`
        // overrides it for anyone re-deriving that.
        let promptScale =
            ProcessInfo.processInfo.environment["VMLX_VOICECHAT_PROMPT_SCALE"]
            .flatMap(Float.init) ?? (1.0 / Foundation.sqrt(Float(codeEmbed.dim(-1))))
        var projectedPrompt = MLX.matmul(codeEmbed, audioPromptProjectionW) * promptScale
        if let audioPromptLatent {
            projectedPrompt = audioPromptLatent.asType(codeEmbed.dtype)
        }
        codeEmbed = MLX.where(preBOS.expandedDimensions(axis: -1), projectedPrompt, codeEmbed)
        codeEmbed = codeEmbed + bosMask.expandedDimensions(axis: -1).asType(codeEmbed.dtype)
            * bosEmb.asType(codeEmbed.dtype)

        let batchSize = code.dim(0)
        if guidance {
            codeEmbed = MLX.concatenated([codeEmbed, codeEmbed], axis: 0)
        }
        let cond = condition(
            subwordIds: subwordIds, subwordMask: subwordMask, batchSize: batchSize,
            guidance: guidance)
        let inputs = gatedFusion(audio: codeEmbed, text: cond)
        let resolvedCache = cache ?? makeCache()
        let hidden = backbone(inputs, cache: resolvedCache)
        return (hidden, resolvedCache)
    }

    /// One decode step over a single frame of codes.
    public func step(
        code: MLXArray, subwordIds: MLXArray, subwordMask: MLXArray?, cache: [KVCache],
        guidance: Bool = true
    ) -> VoiceChatTTSStepOutput {
        var codeEmbed = embedCode(depthSumEmbedding(code))
        let batchSize = code.dim(0)
        if guidance {
            codeEmbed = MLX.concatenated([codeEmbed, codeEmbed], axis: 0)
        }
        let cond = condition(
            subwordIds: subwordIds, subwordMask: subwordMask, batchSize: batchSize,
            guidance: guidance)
        let inputs = gatedFusion(audio: codeEmbed, text: cond)
        if let dir = VoiceChatTTSBackbone.layerDumpDirectory {
            MLX.eval(inputs)
            let flat = inputs.asType(.float32).reshaped([-1]).asArray(Float.self)
            flat.withUnsafeBufferPointer {
                try? Data(buffer: $0).write(
                    to: URL(fileURLWithPath: dir).appendingPathComponent("mine-step-input.f32"))
            }
        }
        let hidden = backbone(inputs, cache: cache)
        return VoiceChatTTSStepOutput(
            codes: generateCodes(hidden), hiddenStates: hidden)
    }

    /// Nearest-neighbour RVQ encode of quantizers `[start, start+count)`.
    private func rvqEncodeStep(
        residual: MLXArray, code: MLXArray, start: Int, count: Int
    ) -> MLXArray {
        var residual = residual
        var pieces = (0 ..< config.numQuantizers).map { code[.ellipsis, $0] }
        for idx in start ..< (start + count) {
            let embedding = rvqEmbs[idx]
            let distances =
                MLX.sum(embedding * embedding, axis: -1)
                - 2.0 * MLX.matmul(residual, embedding.T)
            let selected = MLX.argMin(distances, axis: -1)
            residual = residual - embedding[selected]
            pieces[idx] = selected
        }
        return MLX.stacked(pieces, axis: -1)
    }

    /// Iteratively refine all 31 residual codebooks from the transformer's
    /// hidden state. `hiddenStates` must carry conditional and unconditional
    /// halves (classifier-free guidance).
    public func generateCodes(_ hiddenStates: MLXArray) -> MLXArray {
        // With guidance off the caller passes a single conditional batch; the
        // refinement below is written against a cond/uncond pair, so the same
        // hidden serves as both and the guidance term cancels to zero.
        let guided = hiddenStates.dim(0) % 2 == 0 && hiddenStates.dim(0) > 1
        let half = guided ? hiddenStates.dim(0) / 2 : hiddenStates.dim(0)
        let conditional = guided ? hiddenStates[0 ..< half] : hiddenStates
        let unconditional = guided ? hiddenStates[half...] : hiddenStates

        let iterations = config.numIterations ?? 8
        let exponent = Double(config.exponent ?? 3.0)
        var code = MLXArray.full(
            [conditional.dim(0), conditional.dim(1), config.numQuantizers],
            values: MLXArray(Int32(config.codebookSize)))

        // How many codebooks remain masked at each refinement rate.
        let masked: [Int] = (0 ..< iterations).map { i in
            let rate = Double(i) / Double(iterations)
            let value = Foundation.pow(1.0 - Foundation.pow(rate, exponent), 1.0 / exponent)
            return Int(ceil(value * Double(config.numQuantizers)))
        }
        let counts: [Int] = (0 ..< masked.count).map { i in
            masked[i] - (i + 1 < masked.count ? masked[i + 1] : 0)
        }

        var completed = 0
        for count in counts where count != 0 {
            let embedded = embedCode(depthSumEmbedding(code))
            // Guided: the head needs a cond/uncond pair to difference. Not
            // guided: a single batch, or the pair would double the batch while
            // `code` stays single and the RVQ re-encode below cannot stack.
            let mogInput =
                guided
                ? MLX.concatenated([embedded + conditional, embedded + unconditional], axis: 0)
                : embedded + conditional
            let (mu, logs) = mogHead.infer(
                mogInput,
                guidanceScale: guided ? (config.guidanceScale ?? 0.2) : 0,
                topP: config.topP ?? 0.95)
            let residual =
                mu + MLX.exp(logs) * MLXRandom.normal(mu.shape) * (config.noiseScale ?? 0.001)
            code = rvqEncodeStep(
                residual: residual, code: code, start: completed, count: count)
            completed += count
        }
        return code
    }
}
