// Copyright © 2026 Jinho Jang (eric@jangq.ai)

import Accelerate
import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

// Native vMLX implementation of NVIDIA Nemotron-Labs-Audex. The family
// combines either a Nemotron-Dense or Nemotron-H decoder, a Qwen2-Audio
// (NV-Whisper) encoder, and a squared-ReLU projector.

public struct AudexConfiguration: Codable, Sendable {
    public struct AudioConfiguration: Codable, Sendable {
        let dModel: Int
        let attentionHeads: Int
        let ffnDim: Int
        let layers: Int
        let maxSourcePositions: Int
        let melBins: Int

        enum CodingKeys: String, CodingKey {
            case dModel = "d_model"
            case attentionHeads = "encoder_attention_heads"
            case ffnDim = "encoder_ffn_dim"
            case layers = "encoder_layers"
            case maxSourcePositions = "max_source_positions"
            case melBins = "num_mel_bins"
        }
    }

    let hiddenSize: Int
    let intermediateSize: Int
    let hiddenLayers: Int
    let attentionHeads: Int
    let kvHeads: Int
    let headDim: Int
    let vocabularySize: Int
    let maxPositionEmbeddings: Int
    let normEps: Float
    let ropeTheta: Float
    let tieWordEmbeddings: Bool
    let audio: AudioConfiguration
    let audioProjectorIntermediateSize: Int
    let audioProjectorNormEps: Float
    let soundTokenId: Int
    let soundStartTokenId: Int
    let soundEndTokenId: Int
    let soundTargetRate: Int
    let soundClipDuration: Float
    let soundEmbeddingSize: Int

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case vocabularySize = "vocab_size"
        case maxPositionEmbeddings = "max_position_embeddings"
        case normEps = "norm_eps"
        case ropeParameters = "rope_parameters"
        case tieWordEmbeddings = "tie_word_embeddings"
        case audio = "audio_config"
        case audioProjectorIntermediateSize = "audio_projector_intermediate_size"
        case audioProjectorNormEps = "audio_projector_norm_eps"
        case soundTokenId = "sound_token_id"
        case soundStartTokenId = "sound_start_token_id"
        case soundEndTokenId = "sound_end_token_id"
        case soundTargetRate = "sound_target_rate"
        case soundClipDuration = "sound_clip_duration"
        case soundEmbeddingSize = "sound_embedding_size"
    }

    enum RopeKeys: String, CodingKey { case theta = "rope_theta" }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 2048
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 9216
        hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 28
        attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 16
        kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 8
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? 128
        vocabularySize = try c.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 205_312
        maxPositionEmbeddings =
            try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 131_072
        normEps = try c.decodeIfPresent(Float.self, forKey: .normEps) ?? 1e-5
        if let rope = try? c.nestedContainer(keyedBy: RopeKeys.self, forKey: .ropeParameters) {
            ropeTheta = try rope.decodeIfPresent(Float.self, forKey: .theta) ?? 100_000_000
        } else {
            ropeTheta = 100_000_000
        }
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        audio = try c.decode(AudioConfiguration.self, forKey: .audio)
        audioProjectorIntermediateSize =
            try c.decodeIfPresent(Int.self, forKey: .audioProjectorIntermediateSize) ?? 4096
        audioProjectorNormEps =
            try c.decodeIfPresent(Float.self, forKey: .audioProjectorNormEps) ?? 1e-5
        soundTokenId = try c.decodeIfPresent(Int.self, forKey: .soundTokenId) ?? 29
        soundStartTokenId = try c.decodeIfPresent(Int.self, forKey: .soundStartTokenId) ?? 30
        soundEndTokenId = try c.decodeIfPresent(Int.self, forKey: .soundEndTokenId) ?? 31
        soundTargetRate = try c.decodeIfPresent(Int.self, forKey: .soundTargetRate) ?? 16_000
        soundClipDuration = try c.decodeIfPresent(Float.self, forKey: .soundClipDuration) ?? 30
        soundEmbeddingSize = try c.decodeIfPresent(Int.self, forKey: .soundEmbeddingSize) ?? 750
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(hiddenSize, forKey: .hiddenSize)
        try c.encode(intermediateSize, forKey: .intermediateSize)
        try c.encode(hiddenLayers, forKey: .hiddenLayers)
        try c.encode(attentionHeads, forKey: .attentionHeads)
        try c.encode(kvHeads, forKey: .kvHeads)
        try c.encode(headDim, forKey: .headDim)
        try c.encode(vocabularySize, forKey: .vocabularySize)
        try c.encode(maxPositionEmbeddings, forKey: .maxPositionEmbeddings)
        try c.encode(normEps, forKey: .normEps)
        var rope = c.nestedContainer(keyedBy: RopeKeys.self, forKey: .ropeParameters)
        try rope.encode(ropeTheta, forKey: .theta)
        try c.encode(tieWordEmbeddings, forKey: .tieWordEmbeddings)
        try c.encode(audio, forKey: .audio)
        try c.encode(audioProjectorIntermediateSize, forKey: .audioProjectorIntermediateSize)
        try c.encode(audioProjectorNormEps, forKey: .audioProjectorNormEps)
        try c.encode(soundTokenId, forKey: .soundTokenId)
        try c.encode(soundStartTokenId, forKey: .soundStartTokenId)
        try c.encode(soundEndTokenId, forKey: .soundEndTokenId)
        try c.encode(soundTargetRate, forKey: .soundTargetRate)
        try c.encode(soundClipDuration, forKey: .soundClipDuration)
        try c.encode(soundEmbeddingSize, forKey: .soundEmbeddingSize)
    }
}

/// Configuration for the Nemotron-H MoE/SSM Audex checkpoint. The language
/// fields decode through the existing Nemotron-H runtime while the audio
/// fields retain the same NV-Whisper/projector contract as the dense model.
public struct AudexHConfiguration: Codable, Sendable {
    let language: NemotronHConfiguration
    let audio: AudexConfiguration.AudioConfiguration
    let audioProjectorIntermediateSize: Int
    let audioProjectorNormEps: Float
    let soundTokenId: Int
    let soundStartTokenId: Int
    let soundEndTokenId: Int
    let soundTargetRate: Int
    let soundClipDuration: Float
    let soundEmbeddingSize: Int

    enum CodingKeys: String, CodingKey {
        case audio = "audio_config"
        case audioProjectorIntermediateSize = "audio_projector_intermediate_size"
        case audioProjectorNormEps = "audio_projector_norm_eps"
        case soundTokenId = "sound_token_id"
        case soundStartTokenId = "sound_start_token_id"
        case soundEndTokenId = "sound_end_token_id"
        case soundTargetRate = "sound_target_rate"
        case soundClipDuration = "sound_clip_duration"
        case soundEmbeddingSize = "sound_embedding_size"
    }

    public init(from decoder: Decoder) throws {
        language = try NemotronHConfiguration(from: decoder)
        let c = try decoder.container(keyedBy: CodingKeys.self)
        audio = try c.decode(AudexConfiguration.AudioConfiguration.self, forKey: .audio)
        audioProjectorIntermediateSize =
            try c.decodeIfPresent(Int.self, forKey: .audioProjectorIntermediateSize) ?? 4096
        audioProjectorNormEps =
            try c.decodeIfPresent(Float.self, forKey: .audioProjectorNormEps) ?? 1e-5
        soundTokenId = try c.decodeIfPresent(Int.self, forKey: .soundTokenId) ?? 29
        soundStartTokenId = try c.decodeIfPresent(Int.self, forKey: .soundStartTokenId) ?? 30
        soundEndTokenId = try c.decodeIfPresent(Int.self, forKey: .soundEndTokenId) ?? 31
        soundTargetRate = try c.decodeIfPresent(Int.self, forKey: .soundTargetRate) ?? 16_000
        soundClipDuration = try c.decodeIfPresent(Float.self, forKey: .soundClipDuration) ?? 30
        soundEmbeddingSize = try c.decodeIfPresent(Int.self, forKey: .soundEmbeddingSize) ?? 750
    }

    public func encode(to encoder: Encoder) throws {
        try language.encode(to: encoder)
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(audio, forKey: .audio)
        try c.encode(audioProjectorIntermediateSize, forKey: .audioProjectorIntermediateSize)
        try c.encode(audioProjectorNormEps, forKey: .audioProjectorNormEps)
        try c.encode(soundTokenId, forKey: .soundTokenId)
        try c.encode(soundStartTokenId, forKey: .soundStartTokenId)
        try c.encode(soundEndTokenId, forKey: .soundEndTokenId)
        try c.encode(soundTargetRate, forKey: .soundTargetRate)
        try c.encode(soundClipDuration, forKey: .soundClipDuration)
        try c.encode(soundEmbeddingSize, forKey: .soundEmbeddingSize)
    }
}

private func audexRelu2(_ x: MLXArray) -> MLXArray {
    let y = MLX.maximum(x, MLXArray(0, dtype: x.dtype))
    return y * y
}

private final class AudexAttention: Module {
    @ModuleInfo(key: "q_proj") var q: Linear
    @ModuleInfo(key: "k_proj") var k: Linear
    @ModuleInfo(key: "v_proj") var v: Linear
    @ModuleInfo(key: "o_proj") var output: Linear
    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let scale: Float
    let rope: RoPELayer

    init(_ config: AudexConfiguration) {
        heads = config.attentionHeads
        kvHeads = config.kvHeads
        headDim = config.headDim
        scale = pow(Float(headDim), -0.5)
        _q.wrappedValue = Linear(config.hiddenSize, heads * headDim, bias: false)
        _k.wrappedValue = Linear(config.hiddenSize, kvHeads * headDim, bias: false)
        _v.wrappedValue = Linear(config.hiddenSize, kvHeads * headDim, bias: false)
        _output.wrappedValue = Linear(heads * headDim, config.hiddenSize, bias: false)
        rope = initializeRope(
            dims: headDim, base: config.ropeTheta, traditional: false,
            scalingConfig: nil, maxPositionEmbeddings: config.maxPositionEmbeddings)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let b = x.dim(0)
        let t = x.dim(1)
        var queries = q(x).reshaped(b, t, heads, headDim).transposed(0, 2, 1, 3)
        var keys = k(x).reshaped(b, t, kvHeads, headDim).transposed(0, 2, 1, 3)
        let values = v(x).reshaped(b, t, kvHeads, headDim).transposed(0, 2, 1, 3)
        queries = applyRotaryPosition(rope, to: queries, cache: cache)
        keys = applyRotaryPosition(rope, to: keys, cache: cache)
        let attended = attentionWithCacheUpdate(
            queries: queries, keys: keys, values: values, cache: cache,
            scale: scale, mask: mask
        )
        .transposed(0, 2, 1, 3).reshaped(b, t, heads * headDim)
        return output(attended)
    }
}

private final class AudexMLP: Module {
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    init(_ config: AudexConfiguration) {
        _up.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _down.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { down(audexRelu2(up(x))) }
}

private final class AudexDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: AudexAttention
    @ModuleInfo(key: "mlp") var mlp: AudexMLP
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionNorm: RMSNorm

    init(_ config: AudexConfiguration) {
        _attention.wrappedValue = AudexAttention(config)
        _mlp.wrappedValue = AudexMLP(config)
        _inputNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.normEps)
        _postAttentionNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.normEps)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let h = x + attention(inputNorm(x), mask: mask, cache: cache)
        return h + mlp(postAttentionNorm(h))
    }
}

private final class AudexTextModel: Module {
    @ModuleInfo(key: "embed_tokens") var embeddings: Embedding
    let layers: [AudexDecoderLayer]
    let norm: RMSNorm

    init(_ config: AudexConfiguration) {
        _embeddings.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        layers = (0 ..< config.hiddenLayers).map { _ in AudexDecoderLayer(config) }
        norm = RMSNorm(dimensions: config.hiddenSize, eps: config.normEps)
    }

    func forward(_ x: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = x
        let mask = createAttentionMask(h: h, cache: cache?.first)
        for (index, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[index])
        }
        return norm(h)
    }
}

private final class AudexAudioAttention: Module {
    @ModuleInfo(key: "q_proj") var q: Linear
    @ModuleInfo(key: "k_proj") var k: Linear
    @ModuleInfo(key: "v_proj") var v: Linear
    @ModuleInfo(key: "out_proj") var output: Linear
    let heads: Int
    let headDim: Int
    let scale: Float

    init(_ config: AudexConfiguration.AudioConfiguration) {
        heads = config.attentionHeads
        headDim = config.dModel / heads
        scale = pow(Float(headDim), -0.5)
        _q.wrappedValue = Linear(config.dModel, config.dModel, bias: true)
        _k.wrappedValue = Linear(config.dModel, config.dModel, bias: false)
        _v.wrappedValue = Linear(config.dModel, config.dModel, bias: true)
        _output.wrappedValue = Linear(config.dModel, config.dModel, bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let b = x.dim(0)
        let t = x.dim(1)
        let d = x.dim(2)
        let queries = q(x).reshaped(b, t, heads, headDim).transposed(0, 2, 1, 3)
        let keys = k(x).reshaped(b, t, heads, headDim).transposed(0, 2, 1, 3)
        let values = v(x).reshaped(b, t, heads, headDim).transposed(0, 2, 1, 3)
        let h = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: scale, mask: .none
        )
        .transposed(0, 2, 1, 3).reshaped(b, t, d)
        return output(h)
    }
}

private final class AudexAudioEncoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: AudexAudioAttention
    @ModuleInfo(key: "self_attn_layer_norm") var attentionNorm: LayerNorm
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear
    @ModuleInfo(key: "final_layer_norm") var finalNorm: LayerNorm
    let gelu = GELU()

    init(_ config: AudexConfiguration.AudioConfiguration) {
        _attention.wrappedValue = AudexAudioAttention(config)
        _attentionNorm.wrappedValue = LayerNorm(dimensions: config.dModel)
        _fc1.wrappedValue = Linear(config.dModel, config.ffnDim, bias: true)
        _fc2.wrappedValue = Linear(config.ffnDim, config.dModel, bias: true)
        _finalNorm.wrappedValue = LayerNorm(dimensions: config.dModel)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = x + attention(attentionNorm(x))
        return h + fc2(gelu(fc1(finalNorm(h))))
    }
}

private final class AudexAudioEncoder: Module {
    @ModuleInfo(key: "conv1") var conv1: Conv1d
    @ModuleInfo(key: "conv2") var conv2: Conv1d
    @ModuleInfo(key: "embed_positions") var positions: Embedding
    let layers: [AudexAudioEncoderLayer]
    @ModuleInfo(key: "layer_norm") var norm: LayerNorm
    let gelu = GELU()

    init(_ config: AudexConfiguration.AudioConfiguration) {
        _conv1.wrappedValue = Conv1d(
            inputChannels: config.melBins, outputChannels: config.dModel,
            kernelSize: 3, padding: 1, bias: true)
        _conv2.wrappedValue = Conv1d(
            inputChannels: config.dModel, outputChannels: config.dModel,
            kernelSize: 3, stride: 2, padding: 1, bias: true)
        _positions.wrappedValue = Embedding(
            embeddingCount: config.maxSourcePositions, dimensions: config.dModel)
        layers = (0 ..< config.layers).map { _ in AudexAudioEncoderLayer(config) }
        _norm.wrappedValue = LayerNorm(dimensions: config.dModel)
    }

    func callAsFunction(_ features: MLXArray) -> MLXArray {
        // Whisper input [clips, mel, frames] -> MLX Conv1d [clips, frames, mel].
        var h = features.transposed(0, 2, 1)
        h = gelu(conv1(h))
        h = gelu(conv2(h))
        let ids = MLXArray(Array(0 ..< h.dim(1)))[.newAxis, 0...]
        h = h + positions(ids)
        for layer in layers { h = layer(h) }
        let evenLength = (h.dim(1) / 2) * 2
        h = h[0..., ..<evenLength, 0...]
            .reshaped(h.dim(0), evenLength / 2, 2, h.dim(2)).mean(axis: 2)
        return norm(h)
    }
}

private final class AudexAudioProjector: Module {
    let norm: RMSNorm
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear

    init(
        audio: AudexConfiguration.AudioConfiguration,
        intermediateSize: Int,
        outputSize: Int,
        normEps: Float
    ) {
        norm = RMSNorm(dimensions: audio.dModel, eps: normEps)
        _fc1.wrappedValue = Linear(
            audio.dModel, intermediateSize, bias: false)
        _fc2.wrappedValue = Linear(
            intermediateSize, outputSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { fc2(audexRelu2(fc1(norm(x)))) }
}

public final class Audex: Module, VLMModel, KVCacheDimensionProvider {
    @ModuleInfo(key: "model") private var textModel: AudexTextModel
    @ModuleInfo(key: "lm_head") private var lmHead: Linear
    @ModuleInfo(key: "audio_encoder") private var audioEncoder: AudexAudioEncoder
    @ModuleInfo(key: "audio_projector") private var audioProjector: AudexAudioProjector

    public let config: AudexConfiguration
    public let vocabularySize: Int
    public let kvHeads: [Int]
    public var loraLayers: [Module] { textModel.layers }

    public init(_ config: AudexConfiguration) {
        self.config = config
        vocabularySize = config.vocabularySize
        kvHeads = Array(repeating: config.kvHeads, count: config.hiddenLayers)
        _textModel.wrappedValue = AudexTextModel(config)
        _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
        _audioEncoder.wrappedValue = AudexAudioEncoder(config.audio)
        _audioProjector.wrappedValue = AudexAudioProjector(
            audio: config.audio,
            intermediateSize: config.audioProjectorIntermediateSize,
            outputSize: config.hiddenSize,
            normEps: config.audioProjectorNormEps)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        lmHead(textModel.forward(textModel.embeddings(inputs), cache: cache))
    }

    private func forward(embeddings: MLXArray, cache: [KVCache]?) -> MLXArray {
        lmHead(textModel.forward(embeddings, cache: cache))
    }

    public func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws
        -> PrepareResult
    {
        guard input.image == nil, input.video == nil else {
            throw VLMError.processing("Audex-2B supports audio and text, not image or video input.")
        }

        let tokenShape = input.text.tokens.shape
        guard tokenShape.count == 2, tokenShape[0] == 1 else {
            throw VLMError.processing(
                "Audex-2B currently requires batch size 1; got \(tokenShape).")
        }

        var embeddings = textModel.embeddings(input.text.tokens)
        if let audio = input.audio {
            let audioFeatures: MLXArray
            if let preEncoded = audio.preEncodedEmbedding {
                audioFeatures =
                    preEncoded.ndim == 2 ? preEncoded : preEncoded.reshaped(-1, preEncoded.dim(-1))
            } else {
                guard audio.sampleRate == config.soundTargetRate else {
                    throw VLMError.processing(
                        "Audex processor must provide \(config.soundTargetRate) Hz PCM; got \(audio.sampleRate) Hz."
                    )
                }
                let clips =
                    audio.waveform.ndim == 1
                    ? audio.waveform.reshaped(1, -1) : audio.waveform
                var melRows = [MLXArray]()
                for i in 0 ..< clips.dim(0) {
                    let pcm = clips[i].asType(.float32).asArray(Float.self)
                    melRows.append(audexWhisperFeatures(pcm))
                }
                let mel = melRows.count == 1 ? melRows[0] : MLX.concatenated(melRows, axis: 0)
                PrefillProgressReporter.reportCompletedUnits(1)
                audioFeatures = audioProjector(audioEncoder(mel))
                    .reshaped(-1, config.hiddenSize)
            }
            let mask = MLX.equal(input.text.tokens, MLXArray(Int32(config.soundTokenId)))
            let count = mask.reshaped([-1]).asArray(Int.self).reduce(0, +)
            guard count == audioFeatures.dim(0) else {
                throw VLMError.processing(
                    "Audex audio placeholder mismatch: prompt has \(count), encoder produced \(audioFeatures.dim(0))."
                )
            }
            let expandedMask = MLX.broadcast(
                mask.expandedDimensions(axis: -1), to: embeddings.shape)
            embeddings = try gemma4MaskedScatter(
                input: embeddings, mask: expandedMask,
                source: audioFeatures.asType(embeddings.dtype))
        }

        let logits = try chunkedPrefillEmbedding(
            inputEmbedding: embeddings, cache: cache,
            prefillStepSize: windowSize ?? 512
        ) { chunk in
            forward(embeddings: chunk, cache: cache)
        }
        return .logits(LMOutput(logits: logits))
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var result = [String: MLXArray]()
        for (key, value) in weights {
            if key.hasPrefix("audio_encoder.conv"), key.hasSuffix(".weight"), value.ndim == 3 {
                result[key] = value.transposed(0, 2, 1)
            } else {
                result[key] = value
            }
        }
        return result
    }
}

/// Audex-30B-A3B uses the production Nemotron-H hybrid Mamba/attention/MoE
/// decoder and the same native NV-Whisper audio frontend as Audex-2B.
public final class AudexH: Module, VLMModel, KVCacheDimensionProvider {
    @ModuleInfo(key: "language_model") private var languageModel: NemotronHModel
    @ModuleInfo(key: "audio_encoder") private var audioEncoder: AudexAudioEncoder
    @ModuleInfo(key: "audio_projector") private var audioProjector: AudexAudioProjector

    public let config: AudexHConfiguration
    public var vocabularySize: Int { languageModel.vocabularySize }
    public var kvHeads: [Int] { languageModel.kvHeads }
    public var loraLayers: [Module] { languageModel.loraLayers }

    public init(_ config: AudexHConfiguration) {
        self.config = config
        _languageModel.wrappedValue = NemotronHModel(config.language)
        _audioEncoder.wrappedValue = AudexAudioEncoder(config.audio)
        _audioProjector.wrappedValue = AudexAudioProjector(
            audio: config.audio,
            intermediateSize: config.audioProjectorIntermediateSize,
            outputSize: config.language.hiddenSize,
            normEps: config.audioProjectorNormEps)
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        languageModel.newCache(parameters: parameters)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache)
    }

    private func forward(embeddings: MLXArray, cache: [KVCache]?) -> MLXArray {
        languageModel(inputsEmbeds: embeddings, cache: cache)
    }

    public func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws
        -> PrepareResult
    {
        guard input.image == nil, input.video == nil else {
            throw VLMError.processing("Audex supports audio and text, not image or video input.")
        }

        let tokenShape = input.text.tokens.shape
        guard tokenShape.count == 2, tokenShape[0] == 1 else {
            throw VLMError.processing(
                "Audex currently requires batch size 1; got \(tokenShape).")
        }

        var embeddings = languageModel.embedTokens(input.text.tokens)
        if let audio = input.audio {
            let audioFeatures: MLXArray
            if let preEncoded = audio.preEncodedEmbedding {
                audioFeatures =
                    preEncoded.ndim == 2 ? preEncoded : preEncoded.reshaped(-1, preEncoded.dim(-1))
            } else {
                guard audio.sampleRate == config.soundTargetRate else {
                    throw VLMError.processing(
                        "Audex processor must provide \(config.soundTargetRate) Hz PCM; got \(audio.sampleRate) Hz."
                    )
                }
                let clips =
                    audio.waveform.ndim == 1
                    ? audio.waveform.reshaped(1, -1) : audio.waveform
                var melRows = [MLXArray]()
                for i in 0 ..< clips.dim(0) {
                    let pcm = clips[i].asType(.float32).asArray(Float.self)
                    melRows.append(audexWhisperFeatures(pcm))
                }
                let mel = melRows.count == 1 ? melRows[0] : MLX.concatenated(melRows, axis: 0)
                PrefillProgressReporter.reportCompletedUnits(1)
                audioFeatures = audioProjector(audioEncoder(mel))
                    .reshaped(-1, config.language.hiddenSize)
            }
            let mask = MLX.equal(input.text.tokens, MLXArray(Int32(config.soundTokenId)))
            let count = mask.reshaped([-1]).asArray(Int.self).reduce(0, +)
            guard count == audioFeatures.dim(0) else {
                throw VLMError.processing(
                    "Audex audio placeholder mismatch: prompt has \(count), encoder produced \(audioFeatures.dim(0))."
                )
            }
            let expandedMask = MLX.broadcast(
                mask.expandedDimensions(axis: -1), to: embeddings.shape)
            embeddings = try gemma4MaskedScatter(
                input: embeddings, mask: expandedMask,
                source: audioFeatures.asType(embeddings.dtype))
        }

        let logits = try chunkedPrefillEmbedding(
            inputEmbedding: embeddings, cache: cache,
            prefillStepSize: windowSize ?? 512
        ) { chunk in
            forward(embeddings: chunk, cache: cache)
        }
        return .logits(LMOutput(logits: logits))
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var languageWeights = [String: MLXArray]()
        var result = [String: MLXArray]()
        for (key, value) in weights {
            if key.hasPrefix("audio_encoder.") || key.hasPrefix("audio_projector.") {
                if key.hasPrefix("audio_encoder.conv"), key.hasSuffix(".weight"), value.ndim == 3 {
                    result[key] = value.transposed(0, 2, 1)
                } else {
                    result[key] = value
                }
            } else {
                languageWeights[key] = value
            }
        }
        for (key, value) in languageModel.sanitize(weights: languageWeights) {
            result["language_model.\(key)"] = value
        }
        return result
    }
}

// MARK: - Whisper feature extraction

private enum AudexMel {
    static let sampleRate = 16_000
    static let fft = 400
    static let hop = 160
    static let bins = 128
    static let maxSamples = 480_000
}

private func audexHzToMel(_ frequency: Float) -> Float {
    if frequency < 1000 { return 3 * frequency / 200 }
    return 15 + log(frequency / 1000) / (log(6.4) / 27)
}

private func audexMelToHz(_ mel: Float) -> Float {
    if mel < 15 { return 200 * mel / 3 }
    return 1000 * exp((log(6.4) / 27) * (mel - 15))
}

private let audexMelFilterbank: [[Float]] = {
    let frequencyBins = AudexMel.fft / 2 + 1
    let minMel = audexHzToMel(0)
    let maxMel = audexHzToMel(Float(AudexMel.sampleRate / 2))
    let corners = (0 ..< AudexMel.bins + 2).map { index in
        audexMelToHz(minMel + (maxMel - minMel) * Float(index) / Float(AudexMel.bins + 1))
    }
    let fftFrequencies = (0 ..< frequencyBins).map {
        Float($0) * Float(AudexMel.sampleRate) / Float(AudexMel.fft)
    }
    return (0 ..< frequencyBins).map { k in
        (0 ..< AudexMel.bins).map { m in
            let lower = (fftFrequencies[k] - corners[m]) / (corners[m + 1] - corners[m])
            let upper = (corners[m + 2] - fftFrequencies[k]) / (corners[m + 2] - corners[m + 1])
            let triangle = max(0, min(lower, upper))
            return triangle * 2 / (corners[m + 2] - corners[m])
        }
    }
}()

private let audexHann: [Float] = (0 ..< AudexMel.fft).map { index in
    0.5 - 0.5 * cos(2 * Float.pi * Float(index) / Float(AudexMel.fft))
}

/// Transformers WhisperFeatureExtractor parity: 30 s right padding, centered
/// periodic-Hann STFT, Slaney-normalized 128-bin power mel, then dynamic-range
/// clipping and affine log normalization. Returns [1, 128, 3000].
func audexWhisperFeatures(_ input: [Float]) -> MLXArray {
    var pcm = Array(input.prefix(AudexMel.maxSamples))
    if pcm.isEmpty { pcm = [0] }
    if pcm.count < AudexMel.maxSamples {
        pcm.append(contentsOf: repeatElement(0, count: AudexMel.maxSamples - pcm.count))
    }

    let pad = AudexMel.fft / 2
    var centered = [Float]()
    centered.reserveCapacity(pcm.count + pad * 2)
    // numpy reflect padding: x[200],...,x[1] then x then x[-2],...,x[-201].
    for i in stride(from: pad, through: 1, by: -1) { centered.append(pcm[i]) }
    centered.append(contentsOf: pcm)
    for i in 1 ... pad { centered.append(pcm[pcm.count - 1 - i]) }

    let frameCount = (centered.count - AudexMel.fft) / AudexMel.hop + 1  // 3001
    var framed = [Float](repeating: 0, count: frameCount * AudexMel.fft)
    for frame in 0 ..< frameCount {
        let source = frame * AudexMel.hop
        let destination = frame * AudexMel.fft
        for index in 0 ..< AudexMel.fft {
            framed[destination + index] = centered[source + index] * audexHann[index]
        }
    }

    // Use an actual length-400 real FFT. Zero-padding to a radix-2 FFT changes
    // Whisper's frequency grid and is not processor-compatible.
    let frames = MLXArray(framed).reshaped(frameCount, AudexMel.fft)
    let spectrum = MLXFFT.rfft(frames, n: AudexMel.fft, axis: -1, stream: .cpu)
    let magnitude = MLX.abs(spectrum)
    let power = magnitude * magnitude
    let filters = MLXArray(audexMelFilterbank.flatMap { $0 })
        .reshaped(AudexMel.fft / 2 + 1, AudexMel.bins)
    var logMel =
        MLX.log(MLX.maximum(power.matmul(filters), MLXArray(1e-10)))
        / Float(log(10.0))
    logMel = logMel[..<(frameCount - 1), 0...]  // HF drops the last STFT frame
    logMel = MLX.maximum(logMel, logMel.max() - 8)
    logMel = (logMel + 4) / 4
    return logMel.transposed(1, 0).expandedDimensions(axis: 0)
}

// MARK: - Input processor

public struct AudexProcessorConfiguration: Codable, Sendable {
    public let processorClass: String
    enum CodingKeys: String, CodingKey { case processorClass = "processor_class" }
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        processorClass =
            try c.decodeIfPresent(String.self, forKey: .processorClass)
            ?? "Qwen2AudioProcessor"
    }
}

public struct AudexProcessor: UserInputProcessor {
    private let tokenizer: any Tokenizer
    private static let sampleRate = 16_000
    private static let clipSamples = 480_000
    private static let tokensPerClip = 750
    private static let soundTokenId = 29

    public init(_: AudexProcessorConfiguration, tokenizer: any Tokenizer) {
        self.tokenizer = tokenizer
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        guard input.images.isEmpty, input.videos.isEmpty else {
            throw VLMError.processing("Audex accepts audio and text only.")
        }
        var clipRows = [[Float]]()
        func appendClips(_ audios: [UserInput.Audio]) throws -> Int {
            let initialCount = clipRows.count
            for audio in audios {
                let pcm = try Self.waveform(audio)
                let clipCount = max(
                    1, Int(ceil(Double(max(1, pcm.count)) / Double(Self.clipSamples))))
                for clipIndex in 0 ..< clipCount {
                    let start = clipIndex * Self.clipSamples
                    let end = min(start + Self.clipSamples, pcm.count)
                    var clip = start < end ? Array(pcm[start ..< end]) : [Float]()
                    if clip.count < Self.clipSamples {
                        clip.append(
                            contentsOf: repeatElement(
                                0, count: Self.clipSamples - clip.count))
                    }
                    clipRows.append(clip)
                }
            }
            return clipRows.count - initialCount
        }

        var messages = DefaultMessageGenerator().generate(from: input)
        var audioTokenCounts = [Int]()
        let media = "<so_start><so_embedding><so_end>\n"
        switch input.prompt {
        case .chat(let chat):
            for (index, message) in chat.enumerated() where !message.audios.isEmpty {
                let clipCount = try appendClips(message.audios)
                audioTokenCounts.append(clipCount * Self.tokensPerClip)
                guard messages.indices.contains(index) else {
                    throw VLMError.processing(
                        "Audex message generator changed the structured chat message count.")
                }
                messages[index]["content"] =
                    media + Self.contentText(messages[index]["content"])
            }
        case .text, .messages:
            let clipCount = try appendClips(input.audios)
            if clipCount > 0 {
                audioTokenCounts.append(clipCount * Self.tokensPerClip)
                if let index = messages.indices.reversed().first(where: {
                    (messages[$0]["role"] as? String) == "user"
                }) {
                    messages[index]["content"] =
                        media + Self.contentText(messages[index]["content"])
                } else {
                    messages.append(["role": "user", "content": media])
                }
            }
        }

        let renderedTokens = try tokenizer.applyChatTemplate(
            messages: messages, tools: input.tools, additionalContext: input.additionalContext)
        var tokens = [Int]()
        let totalAudioTokens = audioTokenCounts.reduce(0, +)
        tokens.reserveCapacity(
            renderedTokens.count + max(0, totalAudioTokens - audioTokenCounts.count))
        var expandedMarkerCount = 0
        for token in renderedTokens {
            if token == Self.soundTokenId, expandedMarkerCount < audioTokenCounts.count {
                tokens.append(
                    contentsOf: repeatElement(
                        Self.soundTokenId, count: audioTokenCounts[expandedMarkerCount]))
                expandedMarkerCount += 1
            } else {
                tokens.append(token)
            }
        }
        if expandedMarkerCount != audioTokenCounts.count {
            throw VLMError.processing(
                "Audex tokenizer preserved \(expandedMarkerCount)/\(audioTokenCounts.count) "
                    + "<so_embedding> markers as token ID \(Self.soundTokenId).")
        }
        let tokenArray = MLXArray(tokens).expandedDimensions(axis: 0)
        let audio: LMInput.ProcessedAudio? =
            clipRows.isEmpty
            ? nil
            : .init(
                waveform: MLXArray(clipRows.flatMap { $0 }).reshaped(
                    clipRows.count, Self.clipSamples),
                sampleRate: Self.sampleRate)
        return LMInput(
            text: .init(
                tokens: tokenArray, mask: ones(like: tokenArray).asType(.int8), tokenIds: tokens),
            audio: audio, mediaTokenIds: audio == nil ? nil : [Self.soundTokenId],
            cacheScopeSalt: cacheScopeSalt(from: input.additionalContext), toolSchemas: input.tools)
    }

    private static func waveform(_ audio: UserInput.Audio) throws -> [Float] {
        switch audio {
        case .url(let url):
            return try nemotronOmniLoadAudioFile(url, targetSampleRate: Double(sampleRate))
        case .samples(let pcm, let rate):
            return rate == sampleRate
                ? pcm : linearResamplePCM(pcm, fromRate: rate, toRate: sampleRate)
        case .array(let array, let rate):
            let pcm = array.reshaped([-1]).asType(.float32).asArray(Float.self)
            return rate == sampleRate
                ? pcm : linearResamplePCM(pcm, fromRate: rate, toRate: sampleRate)
        case .preEncoded(let pcm, let rate, _):
            return rate == sampleRate
                ? pcm : linearResamplePCM(pcm, fromRate: rate, toRate: sampleRate)
        }
    }

    private static func contentText(_ content: (any Sendable)?) -> String {
        if let text = content as? String { return text }
        if let parts = content as? [[String: any Sendable]] {
            return parts.compactMap { $0["text"] as? String }.joined()
        }
        return ""
    }
}
