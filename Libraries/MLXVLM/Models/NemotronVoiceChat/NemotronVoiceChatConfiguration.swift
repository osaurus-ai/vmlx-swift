// NemotronLabs VoiceChat 11B — configuration.
//
// Full-duplex speech model: listens (16 kHz), transcribes, answers in text, and
// synthesizes aligned speech (22.05 kHz) on a continuous timeline.
//
// Load target is the MLX-layout bundle (`mlx-community/NemotronLabs-VoiceChat-11B-*`),
// NOT the NVIDIA original. The original ships a **NeMo training config** — no
// `model_type`, no `architectures`, and one 44 GB fp32 safetensors — which no
// HF-config-driven loader can dispatch. The MLX bundle keeps the same
// `stt_model.*` / `tts_model.*` tensor names and only changes dtype/sharding,
// so this config maps 1:1 onto the published weights.
//
// Verified against the real bundle (2026-08-20): 1632 tensors, 11.095 B params,
// `stt_model` 997 tensors / 10.098 B, `tts_model` 635 tensors / 0.997 B.

import Foundation
import MLXLLM

/// Mel front-end. Matches `AudioToMelSpectrogramPreprocessor` in the NeMo config.
public struct VoiceChatPreprocessorConfiguration: Codable, Sendable {
    public let sampleRate: Int
    public let features: Int
    public let nFFT: Int
    public let windowSize: Double
    public let windowStride: Double

    enum CodingKeys: String, CodingKey {
        case sampleRate = "sample_rate"
        case features
        case nFFT = "n_fft"
        case windowSize = "window_size"
        case windowStride = "window_stride"
    }
}

/// Conformer perception encoder.
///
/// 🚨 The three streaming fields are what separate this from the offline
/// Parakeet encoder already in `NemotronHOmni/Parakeet.swift`:
///   * `convNormType == "layer_norm"` — Parakeet.swift uses `NemotronHBatchNorm1d`.
///   * `attContextStyle == "chunked_limited"` — Parakeet.swift is full-attention;
///     chunked requires carrying encoder context across chunk boundaries.
///   * `causalDownsampling` / `convContextSize == "causal"`.
/// Getting any of these wrong yields an encoder that is *correct offline* and
/// wrong on a live mic, which is the failure mode a non-streaming test cannot see.
public struct VoiceChatEncoderConfiguration: Codable, Sendable {
    public let dModel: Int
    public let nLayers: Int
    public let nHeads: Int
    public let featIn: Int
    public let ffExpansionFactor: Int
    public let convKernelSize: Int
    public let subsamplingFactor: Int
    public let subsamplingConvChannels: Int
    public let selfAttentionModel: String
    public let convNormType: String?
    public let attContextStyle: String?
    /// 🚨 A LIST OF PAIRS, not a flat array: the real bundle ships
    /// `[[70, 0]]` — `[left_context, right_context]` in encoder frames.
    /// Decoding this as `[Int]` fails on the shipped config.
    public let attContextSize: [[Int]]?
    public let causalDownsampling: Bool?
    public let convContextSize: String?
    public let useBias: Bool?
    public let untieBiases: Bool?
    public let xscaling: Bool?
    public let posEmbMaxLen: Int?

    enum CodingKeys: String, CodingKey {
        case dModel = "d_model"
        case nLayers = "n_layers"
        case nHeads = "n_heads"
        case featIn = "feat_in"
        case ffExpansionFactor = "ff_expansion_factor"
        case convKernelSize = "conv_kernel_size"
        case subsamplingFactor = "subsampling_factor"
        case subsamplingConvChannels = "subsampling_conv_channels"
        case selfAttentionModel = "self_attention_model"
        case convNormType = "conv_norm_type"
        case attContextStyle = "att_context_style"
        case attContextSize = "att_context_size"
        case causalDownsampling = "causal_downsampling"
        case convContextSize = "conv_context_size"
        case useBias = "use_bias"
        case untieBiases = "untie_biases"
        case xscaling
        case posEmbMaxLen = "pos_emb_max_len"
    }

    /// True when the encoder must run causally/chunked rather than offline.
    public var isStreaming: Bool {
        (attContextStyle ?? "regular") != "regular" || (causalDownsampling ?? false)
    }

    /// Left context in encoder frames (70 on the shipped bundle) — how much
    /// history each chunk may attend to. Sizes the state carried across chunks.
    public var leftContextFrames: Int? { attContextSize?.first?.first }

    /// Right context in encoder frames. **0 on the shipped bundle**, i.e. NO
    /// lookahead: the encoder never waits for future audio. That is precisely
    /// what makes low-latency duplex possible, so a non-zero value here would
    /// silently add algorithmic latency to every response.
    public var rightContextFrames: Int? {
        guard let p = attContextSize?.first, p.count > 1 else { return nil }
        return p[1]
    }

    /// Fully causal — no lookahead at all.
    public var isFullyCausal: Bool { rightContextFrames == 0 }
}

/// RNN-T prediction network.
public struct VoiceChatRNNTDecoderConfiguration: Codable, Sendable {
    public let predHidden: Int?
    public let predRNNLayers: Int?
    public let vocabSize: Int?
    /// `true` on the shipped bundle: the embedding table carries one extra row
    /// (1025 = vocab + blank) so the blank id can be fed as a start pad.
    public let blankAsPad: Bool?

    enum CodingKeys: String, CodingKey {
        case predHidden = "pred_hidden"
        case predRNNLayers = "pred_rnn_layers"
        case vocabSize = "vocab_size"
        case blankAsPad = "blank_as_pad"
    }
}

/// RNN-T joint network.
public struct VoiceChatRNNTJointConfiguration: Codable, Sendable {
    public let jointHidden: Int?
    public let numClasses: Int?
    public let encoderHidden: Int?
    public let predHidden: Int?
    public let activation: String?

    enum CodingKeys: String, CodingKey {
        case jointHidden = "joint_hidden"
        case numClasses = "num_classes"
        case encoderHidden = "encoder_hidden"
        case predHidden = "pred_hidden"
        case activation
    }
}

/// 🚨 The RNN-T decoder and joint live INSIDE `audio_config`, not at top level.
/// Easy to miss when writing these structs, and a silent nil if missed.
public struct VoiceChatAudioConfiguration: Codable, Sendable {
    public let preprocessor: VoiceChatPreprocessorConfiguration
    public let encoder: VoiceChatEncoderConfiguration
    public let decoder: VoiceChatRNNTDecoderConfiguration?
    public let joint: VoiceChatRNNTJointConfiguration?
    public let outputDim: Int
    public let maxSymbols: Int

    enum CodingKeys: String, CodingKey {
        case preprocessor, encoder, decoder, joint
        case outputDim = "output_dim"
        case maxSymbols = "max_symbols"
    }
}

/// Mixture-of-Gaussians refinement head (nested in `tts_config.mog_head`).
public struct VoiceChatMoGConfiguration: Codable, Sendable {
    public let intermediateSize: Int?
    public let lowRank: Int?
    public let minLogStd: Float?
    public let numLayers: Int?
    public let numPredictions: Int?
    public let eps: Float?

    enum CodingKeys: String, CodingKey {
        case intermediateSize = "intermediate_size"
        case lowRank = "low_rank"
        case minLogStd = "min_log_std"
        case numLayers = "num_layers"
        case numPredictions = "num_predictions"
        case eps
    }
}

/// Character-aware subword encoder (nested in `tts_config.character_encoder`).
/// Values the bundle omits fall back to the reference dataclass defaults —
/// notably `attn_logit_softcapping 50.0` and `char_vocab_size 257`, which are
/// NOT in the shipped JSON but are load-bearing for the encoder math.
public struct VoiceChatCharacterEncoderConfiguration: Codable, Sendable {
    public let hiddenSize: Int?
    public let intermediateSize: Int?
    public let numHiddenLayers: Int?
    public let numAttentionHeads: Int?
    public let numKeyValueHeads: Int?
    public let headDim: Int?
    public let rmsNormEps: Float?
    public let queryPreAttnScalar: Float?
    public let attnLogitSoftcapping: Float?
    public let ropeBase: Float?
    public let charVocabSize: Int?

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case queryPreAttnScalar = "query_pre_attn_scalar"
        case attnLogitSoftcapping = "attn_logit_softcapping"
        case ropeBase = "rope_base"
        case charVocabSize = "char_vocab_size"
    }
}

/// Speech-generation transformer + mixture-of-Gaussians head.
public struct VoiceChatTTSConfiguration: Codable, Sendable {
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let headDim: Int
    public let slidingWindow: Int
    public let latentSize: Int
    public let numQuantizers: Int
    public let codebookSize: Int
    public let numDelaySpeechTokens: Int
    public let disableEOSPrediction: Bool?
    public let useGatedFusionForTextAudio: Bool?
    public let guidanceScale: Float?
    public let topP: Float?
    public let noiseScale: Float?
    public let audioPromptDuration: Float?
    public let characterEncoder: VoiceChatCharacterEncoderConfiguration?
    public let mogHead: VoiceChatMoGConfiguration?
    /// Reference-dataclass defaults the shipped JSON omits entirely.
    public let rmsNormEps: Float?
    public let queryPreAttnScalar: Float?
    public let slidingWindowPattern: Int?
    public let ropeGlobalBaseFreq: Float?
    public let ropeLocalBaseFreq: Float?
    public let numIterations: Int?
    public let exponent: Float?

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case slidingWindow = "sliding_window"
        case latentSize = "latent_size"
        case numQuantizers = "num_quantizers"
        case codebookSize = "codebook_size"
        case numDelaySpeechTokens = "num_delay_speech_tokens"
        case disableEOSPrediction = "disable_eos_prediction"
        case useGatedFusionForTextAudio = "use_gated_fusion_for_text_audio"
        case guidanceScale = "guidance_scale"
        case topP = "top_p"
        case noiseScale = "noise_scale"
        case audioPromptDuration = "audio_prompt_duration"
        case characterEncoder = "character_encoder"
        case mogHead = "mog_head"
        case rmsNormEps = "rms_norm_eps"
        case queryPreAttnScalar = "query_pre_attn_scalar"
        case slidingWindowPattern = "sliding_window_pattern"
        case ropeGlobalBaseFreq = "rope_global_base_freq"
        case ropeLocalBaseFreq = "rope_local_base_freq"
        case numIterations = "num_iterations"
        case exponent
    }
}

/// Neural audio codec (ConvNeXt-style blocks + residual VQ), 22.05 kHz out.
public struct VoiceChatCodecConfiguration: Codable, Sendable {
    public let sampleRate: Int
    public let baseChannels: Int
    public let channelMultipliers: [Int]
    public let downsampleRates: [Int]
    public let blocksPerStage: Int
    public let blockKernelSize: Int
    public let latentDim: Int
    public let nFFT: Int
    public let hopLength: Int
    public let numQuantizers: Int
    public let codebookSize: Int

    enum CodingKeys: String, CodingKey {
        case sampleRate = "sample_rate"
        case baseChannels = "base_channels"
        case channelMultipliers = "channel_multipliers"
        case downsampleRates = "downsample_rates"
        case blocksPerStage = "blocks_per_stage"
        case blockKernelSize = "block_kernel_size"
        case latentDim = "latent_dim"
        case nFFT = "n_fft"
        case hopLength = "hop_length"
        case numQuantizers = "num_quantizers"
        case codebookSize = "codebook_size"
    }

    /// Real + imaginary rFFT bins the codec consumes/produces: 18 at n_fft 16.
    public var stftChannels: Int { (nFFT / 2 + 1) * 2 }

    /// Waveform samples per codec frame — hop × product of the stage strides.
    public var waveformToTokenRatio: Int {
        downsampleRates.reduce(hopLength, *)
    }
}

public struct NemotronVoiceChatConfiguration: Codable, Sendable {
    /// nemotron_h backbone — 56 layers, `hybrid_override_pattern`
    /// `M-M-M-MM-M-M-M*-M-M-M*-M-M-M-M*-M-M-M-M*-M-MM-M-M-M-M-M-`
    /// (M = Mamba2 ×27, - = MLP ×25, * = attention ×4). Cross-checked against
    /// the weights: 27 `mixer.A_log`, 25 `mixer.up_proj`, 4 `mixer.q_proj`.
    public let textConfig: NemotronHConfiguration
    public let audioConfig: VoiceChatAudioConfiguration
    public let ttsConfig: VoiceChatTTSConfiguration
    public let codecConfig: VoiceChatCodecConfiguration

    public let bosTokenId: Int
    public let eosTokenId: Int
    public let padTokenId: Int
    /// Emitted when the agent should be silent on the audio timeline — a duplex
    /// model is always producing *something*, including deliberate silence.
    public let silenceTokenId: Int
    /// RNN-T blank. Note it is 1024 == `rnnt_vocabulary.count`, i.e. one past
    /// the last real token, not a value inside the vocabulary.
    public let rnntBlankId: Int

    public let inputSampleRate: Int
    public let outputSampleRate: Int
    public let frameDuration: Double

    /// 🚨 Tool calls are a WEIGHTED PARALLEL CHANNEL with their own full
    /// vocab-sized head (`stt_model.function_head`, 0.587 B — the same size as
    /// `lm_head`). They are not text parsed after the fact. A runtime that
    /// reads only `lm_head` loses tool calling entirely and silently.
    public let functionChannelWeight: Float

    public let speaker: String?
    public let rnntVocabulary: [String]?

    enum CodingKeys: String, CodingKey {
        case textConfig = "text_config"
        case audioConfig = "audio_config"
        case ttsConfig = "tts_config"
        case codecConfig = "codec_config"
        case bosTokenId = "bos_token_id"
        case eosTokenId = "eos_token_id"
        case padTokenId = "pad_token_id"
        case silenceTokenId = "silence_token_id"
        case rnntBlankId = "rnnt_blank_id"
        case inputSampleRate = "input_sample_rate"
        case outputSampleRate = "output_sample_rate"
        case frameDuration = "frame_duration"
        case functionChannelWeight = "function_channel_weight"
        case speaker
        case rnntVocabulary = "rnnt_vocabulary"
    }

    /// Frames per second on the shared duplex timeline (12.5 at 0.08 s).
    public var framesPerSecond: Double { 1.0 / frameDuration }
}
