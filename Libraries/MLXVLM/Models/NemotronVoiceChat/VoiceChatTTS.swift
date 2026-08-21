// NemotronLabs VoiceChat — EAR-TTS speech decoder.
//
// Port of `mlx_vlm/models/nemotron_voicechat/tts.py`. Module hierarchy mirrors
// the NeMo checkpoint exactly so the bundle's `tts_model.*` keys map 1:1 and
// the per-module quantization map in published quants applies unchanged.
//
// 🚨 TWO RMSNorm conventions live in ONE model: the nemotron_h backbone uses
// plain RMSNorm (applies `weight`), everything in THIS file uses
// `VoiceChatOffsetRMSNorm` (Gemma-style, `1 + weight`). Getting this wrong
// emptied text output while imatrix rel-err, a NaN scan, and ASR all read
// healthy — trap #1, already paid for once.
//
// 🚨 fp-protected tensors (the loader must NEVER quantize or re-dtype them):
//   rvq_embs                       RVQ codebook, read raw
//   audio_prompt_latents.*         speaker identity
//   mog_head.proj_mus.weight       read RAW and reshaped below
//   embed_subword.embed_tokens     its dtype is read to allocate a buffer

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MLXRandom

/// Gemma-style RMSNorm: the learned weight is an offset from one.
public class VoiceChatOffsetRMSNorm: Module, UnaryLayer {
    @ParameterInfo(key: "weight") public var weight: MLXArray
    public let eps: Float

    public init(dimensions: Int, eps: Float = 1e-6) {
        self._weight.wrappedValue = MLXArray.zeros([dimensions])
        self.eps = eps
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let dtype = x.dtype
        let x32 = x.asType(.float32)
        return MLXFast.rmsNorm(x32, weight: 1.0 + weight.asType(.float32), eps: eps)
            .asType(dtype)
    }
}

/// gate/up/down MLP with approximate GELU (matches `nn.gelu_approx`).
public class VoiceChatTTSMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    public init(hiddenSize: Int, intermediateSize: Int) {
        self._gateProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._upProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._downProj.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(geluApproximate(gateProj(x)) * upProj(x))
    }
}

/// pre_norm → mlp → post_norm residual block (MoG head stack element).
public class VoiceChatMLPLayer: Module, UnaryLayer {
    @ModuleInfo(key: "pre_norm") var preNorm: VoiceChatOffsetRMSNorm
    @ModuleInfo(key: "mlp") var mlp: VoiceChatTTSMLP
    @ModuleInfo(key: "post_norm") var postNorm: VoiceChatOffsetRMSNorm

    public init(hiddenSize: Int, intermediateSize: Int, eps: Float) {
        self._preNorm.wrappedValue = VoiceChatOffsetRMSNorm(dimensions: hiddenSize, eps: eps)
        self._mlp.wrappedValue = VoiceChatTTSMLP(
            hiddenSize: hiddenSize, intermediateSize: intermediateSize)
        self._postNorm.wrappedValue = VoiceChatOffsetRMSNorm(dimensions: hiddenSize, eps: eps)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        x + postNorm(mlp(preNorm(x)))
    }
}

/// Nucleus filter. Keeps the smallest set of tokens whose probability mass
/// reaches `topP`; everything else goes to -inf.
func voiceChatTopPLogits(_ logits: MLXArray, topP: Float) -> MLXArray {
    if topP >= 1.0 { return logits }
    let probs = MLX.softmax(logits.asType(.float32), axis: -1)
    let sorted = MLX.sorted(probs, axis: -1)  // ascending
    let cumulative = MLX.cumsum(sorted, axis: -1)
    let keepSorted = cumulative .> MLXArray(1.0 - topP)
    // Smallest KEPT probability per row; comparing raw probs against it
    // reproduces the kept set (ties may keep equal-probability extras, which
    // is sampling-equivalent).
    let minKept = MLX.min(MLX.where(keepSorted, sorted, MLXArray(Float(2))), axis: -1, keepDims: true)
    return MLX.where(probs .>= minKept, logits, MLXArray(-Float.infinity))
}

/// Mixture-of-Gaussians head for continuous RVQ refinement.
public class VoiceChatMoGHead: Module {
    public let outSize: Int
    public let lowRank: Int
    public let numPredictions: Int
    public let minLogStd: Float

    @ModuleInfo(key: "mlp_stack") var mlpStack: [Module]
    @ModuleInfo(key: "proj_logits") var projLogits: Linear
    /// 🚨 Read RAW below (`weight.reshaped`), never through the Linear call —
    /// must stay unquantized in every published bundle.
    @ModuleInfo(key: "proj_mus") var projMus: Linear
    @ModuleInfo(key: "proj_logs") var projLogs: Linear
    @ModuleInfo(key: "proj_else") var projElse: Linear
    @ParameterInfo(key: "low_mat") var lowMat: MLXArray

    public init(hiddenSize: Int, outSize: Int, config: VoiceChatMoGConfiguration) {
        let intermediate = config.intermediateSize ?? 4608
        let eps = config.eps ?? 1e-6
        let layers = config.numLayers ?? 3
        let predictions = config.numPredictions ?? 1024
        let rank = config.lowRank ?? 64
        self.outSize = outSize
        self.lowRank = rank
        self.numPredictions = predictions
        self.minLogStd = config.minLogStd ?? -4.0

        var stack: [Module] = (0 ..< layers).map { _ in
            VoiceChatMLPLayer(hiddenSize: hiddenSize, intermediateSize: intermediate, eps: eps)
        }
        stack.append(VoiceChatOffsetRMSNorm(dimensions: hiddenSize, eps: eps))
        self._mlpStack.wrappedValue = stack
        self._projLogits.wrappedValue = Linear(hiddenSize, predictions, bias: false)
        self._projMus.wrappedValue = Linear(hiddenSize, predictions * rank, bias: false)
        self._projLogs.wrappedValue = Linear(hiddenSize, 1, bias: false)
        self._projElse.wrappedValue = Linear(hiddenSize, outSize, bias: false)
        self._lowMat.wrappedValue = MLXArray.zeros([predictions, outSize, rank])
    }

    /// `VMLX_VOICECHAT_MOG_ARGMAX=1` replaces the Gumbel draw with an argmax
    /// over the mixture logits — a diagnostic arm for isolating sampling bugs
    /// from head bugs.
    static let deterministicComponent =
        ProcessInfo.processInfo.environment["VMLX_VOICECHAT_MOG_ARGMAX"] == "1"

    /// - Returns: (sampled latent residual mean, log-std)
    public func infer(
        _ input: MLXArray, guidanceScale: Float, topP: Float
    ) -> (MLXArray, MLXArray) {
        var x = input
        for layer in mlpStack {
            x = (layer as! UnaryLayer)(x)
        }

        if guidanceScale > 0 {
            precondition(x.dim(0) % 2 == 0, "classifier-free guidance requires an even batch")
            let half = x.dim(0) / 2
            let cond = x[0 ..< half]
            let uncond = x[half...]
            x = cond + guidanceScale * (cond - uncond)
        }

        let b = x.dim(0), t = x.dim(1)
        let logits = voiceChatTopPLogits(projLogits(x), topP: topP)
        let component: MLXArray
        if Self.deterministicComponent {
            // Diagnostic arm: pick the most likely mixture component instead of
            // sampling one. If speech becomes intelligible only here, the fault
            // is in the nucleus filter or the Gumbel draw, not in the head.
            component = MLX.argMax(logits.asType(.float32), axis: -1)
        } else {
            let uniform = MLXRandom.uniform(low: 0.0, high: 1.0, logits.shape)
            let gumbel = -MLX.log(-MLX.log(uniform + 1e-8) + 1e-8)
            component = MLX.argMax(
                MLX.log(MLX.softmax(logits.asType(.float32), axis: -1)) + gumbel, axis: -1)
        }

        let flatX = x.reshaped([-1, x.dim(-1)])
        let flatComponent = component.reshaped([-1])
        // proj_mus RAW: (predictions*rank, hidden) → (predictions, rank, hidden)
        let mus = projMus.weight.reshaped([numPredictions, lowRank, -1])[flatComponent]
        var mu = MLX.matmul(mus, flatX.expandedDimensions(axis: -1)).squeezed(axis: -1)
        let low = lowMat[flatComponent]
        mu = MLX.matmul(low, mu.expandedDimensions(axis: -1)).squeezed(axis: -1)
            .reshaped([b, t, outSize])
        let residual = projElse(x)
        let logs = MLX.maximum(projLogs(x), MLXArray(minLogStd))
        return (mu * MLX.exp(logs) + residual, logs)
    }
}

// MARK: - Character-aware subword conditioning

/// T5Gemma self-attention with logit soft-capping (char encoder only).
public class VoiceChatT5GemmaAttention: Module {
    let nHeads: Int
    let nKVHeads: Int
    let headDim: Int
    let repeats: Int
    let attnScale: Float
    let softcap: Float
    let rope: RoPE

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    public init(config: VoiceChatCharacterEncoderConfiguration) {
        let hidden = config.hiddenSize ?? 1152
        self.nHeads = config.numAttentionHeads ?? 16
        self.nKVHeads = config.numKeyValueHeads ?? 16
        self.headDim = config.headDim ?? 72
        self.repeats = nHeads / nKVHeads
        self.attnScale = Float(1.0 / Foundation.sqrt(Double(config.queryPreAttnScalar ?? 256.0)))
        self.softcap = config.attnLogitSoftcapping ?? 50.0
        self.rope = RoPE(
            dimensions: headDim, traditional: false, base: config.ropeBase ?? 10_000.0)
        self._qProj.wrappedValue = Linear(hidden, nHeads * headDim, bias: false)
        self._kProj.wrappedValue = Linear(hidden, nKVHeads * headDim, bias: false)
        self._vProj.wrappedValue = Linear(hidden, nKVHeads * headDim, bias: false)
        self._oProj.wrappedValue = Linear(nHeads * headDim, hidden, bias: false)
    }

    /// `mask`: (B, L) boolean validity, broadcast over heads/queries.
    public func callAsFunction(_ x: MLXArray, mask: MLXArray?) -> MLXArray {
        let b = x.dim(0), length = x.dim(1)
        var q = qProj(x).reshaped([b, length, nHeads, -1]).transposed(0, 2, 1, 3)
        var k = kProj(x).reshaped([b, length, nKVHeads, -1]).transposed(0, 2, 1, 3)
        var v = vProj(x).reshaped([b, length, nKVHeads, -1]).transposed(0, 2, 1, 3)
        q = rope(q)
        k = rope(k)
        if repeats > 1 {
            k = MLX.repeated(k, count: repeats, axis: 1)
            v = MLX.repeated(v, count: repeats, axis: 1)
        }
        var scores = MLX.matmul(q, k.transposed(0, 1, 3, 2)) * MLXArray(attnScale)
        scores = MLX.tanh(scores / softcap) * softcap
        if let mask {
            let m = mask.reshaped([b, 1, 1, length])
            scores = MLX.where(m, scores, MLXArray(Float(-1e30)))
        }
        let probs = MLX.softmax(scores.asType(.float32), axis: -1).asType(q.dtype)
        let out = MLX.matmul(probs, v).transposed(0, 2, 1, 3).reshaped([b, length, -1])
        return oProj(out)
    }
}

public class VoiceChatT5GemmaLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: VoiceChatT5GemmaAttention
    @ModuleInfo(key: "pre_self_attn_layernorm") var preSelfAttn: VoiceChatOffsetRMSNorm
    @ModuleInfo(key: "post_self_attn_layernorm") var postSelfAttn: VoiceChatOffsetRMSNorm
    @ModuleInfo(key: "mlp") var mlp: VoiceChatTTSMLP
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFF: VoiceChatOffsetRMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFF: VoiceChatOffsetRMSNorm

    public init(config: VoiceChatCharacterEncoderConfiguration) {
        let hidden = config.hiddenSize ?? 1152
        let eps = config.rmsNormEps ?? 1e-6
        self._selfAttn.wrappedValue = VoiceChatT5GemmaAttention(config: config)
        self._preSelfAttn.wrappedValue = VoiceChatOffsetRMSNorm(dimensions: hidden, eps: eps)
        self._postSelfAttn.wrappedValue = VoiceChatOffsetRMSNorm(dimensions: hidden, eps: eps)
        self._mlp.wrappedValue = VoiceChatTTSMLP(
            hiddenSize: hidden, intermediateSize: config.intermediateSize ?? 4608)
        self._preFF.wrappedValue = VoiceChatOffsetRMSNorm(dimensions: hidden, eps: eps)
        self._postFF.wrappedValue = VoiceChatOffsetRMSNorm(dimensions: hidden, eps: eps)
    }

    public func callAsFunction(_ x: MLXArray, mask: MLXArray?) -> MLXArray {
        var x = x + postSelfAttn(selfAttn(preSelfAttn(x), mask: mask))
        x = x + postFF(mlp(preFF(x)))
        return x
    }
}

public class VoiceChatT5GemmaEncoder: Module {
    @ModuleInfo(key: "layers") var layers: [VoiceChatT5GemmaLayer]
    @ModuleInfo(key: "norm") var norm: VoiceChatOffsetRMSNorm
    let inputScale: Float

    public init(config: VoiceChatCharacterEncoderConfiguration) {
        let hidden = config.hiddenSize ?? 1152
        self._layers.wrappedValue = (0 ..< (config.numHiddenLayers ?? 1)).map { _ in
            VoiceChatT5GemmaLayer(config: config)
        }
        self._norm.wrappedValue = VoiceChatOffsetRMSNorm(
            dimensions: hidden, eps: config.rmsNormEps ?? 1e-6)
        self.inputScale = Float(Foundation.sqrt(Double(hidden)))
    }

    public func callAsFunction(_ inputsEmbeds: MLXArray, mask: MLXArray?) -> MLXArray {
        var x = inputsEmbeds * MLXArray(inputScale).asType(inputsEmbeds.dtype)
        for layer in layers {
            x = layer(x, mask: mask)
        }
        return norm(x)
    }
}

/// Wrapper whose only purpose is the checkpoint's `backbone.encoder.` nesting.
public class VoiceChatT5GemmaBackbone: Module {
    @ModuleInfo(key: "encoder") var encoder: VoiceChatT5GemmaEncoder
    public init(config: VoiceChatCharacterEncoderConfiguration) {
        self._encoder.wrappedValue = VoiceChatT5GemmaEncoder(config: config)
    }
}

/// Adds a learned "is-continuation" flag embedding per subword.
public class VoiceChatSubwordFlagEmbedding: Module {
    @ParameterInfo(key: "pad_tensor") var padTensor: MLXArray
    @ParameterInfo(key: "is_continuation") var isContinuation: MLXArray
    @ModuleInfo(key: "cont_emb") var contEmb: Embedding

    public init(vocabSize: Int, hiddenSize: Int) {
        self._padTensor.wrappedValue = MLXArray(Int32(vocabSize))
        self._isContinuation.wrappedValue = MLXArray.zeros([vocabSize + 1], dtype: .int32)
        self._contEmb.wrappedValue = Embedding(embeddingCount: 2, dimensions: hiddenSize)
    }

    public func callAsFunction(_ embeds: MLXArray, tokenIds: MLXArray) -> MLXArray {
        let limit = isContinuation.dim(0) - 1
        let safe = MLX.where(tokenIds .>= MLXArray(Int32(limit)), padTensor, tokenIds)
        return embeds + contEmb(isContinuation[safe])
    }
}

/// Adds BOS/EOS/none flag embeddings per subword.
public class VoiceChatBOSEOSEmbedding: Module {
    @ParameterInfo(key: "pad_tensor") var padTensor: MLXArray
    @ParameterInfo(key: "special_flags") var specialFlags: MLXArray
    @ModuleInfo(key: "special_emb") var specialEmb: Embedding

    public init(vocabSize: Int, hiddenSize: Int) {
        // The source stores max(vocab.values()), not vocabulary length.
        self._padTensor.wrappedValue = MLXArray(Int32(vocabSize - 1))
        self._specialFlags.wrappedValue = MLXArray.zeros([vocabSize], dtype: .int32)
        self._specialEmb.wrappedValue = Embedding(embeddingCount: 3, dimensions: hiddenSize)
    }

    public func callAsFunction(_ embeds: MLXArray, tokenIds: MLXArray) -> MLXArray {
        let limit = specialFlags.dim(0)
        let safe = MLX.where(tokenIds .>= MLXArray(Int32(limit)), padTensor, tokenIds)
        return embeds + specialEmb(specialFlags[safe])
    }
}

/// Character-aware subword conditioning: each subword's characters run through
/// a tiny T5Gemma encoder, mean-pool, project, then add flag embeddings.
public class VoiceChatCharAwareSubwordEncoder: Module {
    @ModuleInfo(key: "backbone") var backbone: VoiceChatT5GemmaBackbone
    /// 🚨 dtype of this table is READ to allocate the output buffer — stays fp.
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "proj_embedding") var projEmbedding: Linear
    @ModuleInfo(key: "subword_flag_emb") var subwordFlagEmb: VoiceChatSubwordFlagEmbedding
    @ModuleInfo(key: "bos_eos_emb") var bosEosEmb: VoiceChatBOSEOSEmbedding

    public let charPaddingIdx: Int
    /// subword id → character ids; built from the tokenizer via `setVocabulary`.
    private var subwordToChars: [Int: [Int]]?

    public init(
        config: VoiceChatCharacterEncoderConfiguration, outSize: Int, vocabSize: Int = 131_072
    ) {
        let hidden = config.hiddenSize ?? 1152
        let charVocab = config.charVocabSize ?? 257
        self._backbone.wrappedValue = VoiceChatT5GemmaBackbone(config: config)
        self._embedTokens.wrappedValue = Embedding(embeddingCount: charVocab, dimensions: hidden)
        self._projEmbedding.wrappedValue = Linear(hidden, outSize, bias: false)
        self._subwordFlagEmb.wrappedValue = VoiceChatSubwordFlagEmbedding(
            vocabSize: vocabSize, hiddenSize: hidden)
        self._bosEosEmb.wrappedValue = VoiceChatBOSEOSEmbedding(
            vocabSize: vocabSize, hiddenSize: hidden)
        self.charPaddingIdx = charVocab - 1
    }

    /// Build the dense character vocabulary exactly as NeMo does: single-char
    /// tokens sorted by id become the char table. Must be called before TTS.
    public func setVocabulary(_ vocabulary: [String: Int]) throws {
        let single = vocabulary.filter { $0.key.count == 1 }
        let characters = single.sorted { $0.value < $1.value }.map(\.key)
        var charToId = [Character: Int]()
        for (idx, ch) in characters.enumerated() {
            charToId[Character(ch)] = idx
        }
        guard charToId.count + 1 == embedTokens.weight.dim(0) else {
            throw NSError(
                domain: "VoiceChatTTS", code: 1,
                userInfo: [
                    NSLocalizedDescriptionKey:
                        "tokenizer-derived character vocabulary has \(charToId.count + 1) entries, expected \(embedTokens.weight.dim(0))"
                ])
        }
        var map = [Int: [Int]]()
        for (token, idx) in vocabulary {
            map[idx] = token.compactMap { charToId[$0] }
        }
        self.subwordToChars = map
    }

    /// subword ids (B, T) [+ optional validity mask] → conditioning (B, T, out).
    public func callAsFunction(
        _ subwordIds: MLXArray, mask subwordMask: MLXArray? = nil
    ) -> MLXArray {
        guard let subwordToChars else {
            fatalError("setVocabulary(tokenizer vocab) must be called before TTS")
        }
        let B = subwordIds.dim(0), T = subwordIds.dim(1)
        let ids: [Int32] = subwordIds.reshaped([-1]).asArray(Int32.self)
        let valid: [Bool] =
            subwordMask.map { $0.reshaped([-1]).asArray(Bool.self) }
            ?? [Bool](repeating: true, count: ids.count)

        var sequences = [[Int]]()
        var flatPositions = [Int32]()
        for i in 0 ..< ids.count where valid[i] {
            sequences.append(subwordToChars[Int(ids[i])] ?? [])
            flatPositions.append(Int32(i))
        }

        let outWidth = projEmbedding.weight.dim(0)
        var out = MLXArray.zeros([B * T, outWidth], dtype: embedTokens.weight.dtype)
        let maxLength = sequences.map(\.count).max() ?? 0
        if !sequences.isEmpty && maxLength > 0 {
            let charRows: [[Int32]] = sequences.map { seq in
                seq.map(Int32.init) + [Int32](repeating: Int32(charPaddingIdx), count: maxLength - seq.count)
            }
            let charIds = MLXArray(charRows.flatMap { $0 }).reshaped([sequences.count, maxLength])
            let lengths = MLXArray(sequences.map { Int32($0.count) })
            let mask = MLXArray(0 ..< maxLength).reshaped([1, maxLength])
                .< lengths.reshaped([-1, 1])
            let hidden = backbone.encoder(embedTokens(charIds), mask: mask)
            let pooled =
                MLX.sum(hidden * mask.expandedDimensions(axis: -1).asType(hidden.dtype), axis: 1)
                / MLX.maximum(lengths.reshaped([-1, 1]), MLXArray(Int32(1))).asType(hidden.dtype)
            let projected = projEmbedding(pooled)
            out[MLXArray(flatPositions)] = projected.asType(out.dtype)
        }
        var result = out.reshaped([B, T, outWidth])
        result = subwordFlagEmb(result, tokenIds: subwordIds)
        return bosEosEmb(result, tokenIds: subwordIds)
    }
}

/// Gated fusion of code-embedding and text-conditioning streams.
public class VoiceChatGatedFusion: Module {
    public let numCodebooks: Int
    @ModuleInfo(key: "audio_proj") var audioProj: Linear
    @ModuleInfo(key: "text_proj") var textProj: Linear
    @ParameterInfo(key: "gate") var gate: MLXArray
    @ParameterInfo(key: "residual_scale") var residualScale: MLXArray
    @ModuleInfo(key: "final_norm") var finalNorm: VoiceChatOffsetRMSNorm

    public init(hiddenSize: Int, numCodebooks: Int, eps: Float) {
        self.numCodebooks = numCodebooks
        self._audioProj.wrappedValue = Linear(hiddenSize, hiddenSize)
        self._textProj.wrappedValue = Linear(hiddenSize, hiddenSize)
        self._gate.wrappedValue = MLXArray.zeros([hiddenSize])
        self._residualScale.wrappedValue = MLXArray(Float(0.5))
        self._finalNorm.wrappedValue = VoiceChatOffsetRMSNorm(dimensions: hiddenSize, eps: eps)
    }

    public func callAsFunction(audio: MLXArray, text: MLXArray) -> MLXArray {
        let a = audioProj(audio / Float(numCodebooks))
        let t = textProj(text)
        let g = MLX.sigmoid(gate).asType(a.dtype)
        let scale = MLX.sigmoid(residualScale).asType(a.dtype)
        return finalNorm(scale * (g * a + (1.0 - g) * t))
    }
}
