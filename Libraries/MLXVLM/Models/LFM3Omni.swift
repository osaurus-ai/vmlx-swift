//
//  LFM3Omni.swift
//  mlx-swift-lm
//
//  LiquidAI LFM3-Audio and LFM3-Omni, speech INPUT path: log-mel features, a
//  strided Conv2d downsampler, a windowed LFM2 trunk and a projector feed the
//  LFM3 decoder through `<|audio|>` placeholders. LFM3-Omni adds the LFM3-VL
//  vision tower. Text output only: the depth decoder and codec that produce
//  speech are not loaded (see `LFM3Omni.sanitize`).
//
//  References: the bundles' `modeling_lfm3_audio.py` / `modeling_lfm3_omni.py`,
//  and the MLX port in mlx-vlm `mlx_vlm/models/lfm3_audio/`, `lfm3_omni/`.
//

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

// MARK: - Configuration

/// The LFM2 trunk inside the speech encoder.
public struct LFM3SpeechTrunkConfiguration: Codable, Sendable {
    public let hiddenSize: Int
    public let feedForwardSize: Int
    public let attentionHeads: Int
    public let kvHeads: Int
    public let normEps: Float
    public let convLCache: Int
    public let convBias: Bool
    public let layerTypes: [String]
    public let ropeTheta: Float

    public var headDim: Int { hiddenSize / attentionHeads }

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case blockFFDim = "block_ff_dim"
        case autoAdjust = "block_auto_adjust_ff_dim"
        case multipleOf = "block_multiple_of"
        case multiplier = "block_ffn_dim_multiplier"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case hiddenLayers = "num_hidden_layers"
        case normEps = "norm_eps"
        case convLCache = "conv_L_cache"
        case convBias = "conv_bias"
        case layerTypes = "layer_types"
        case ropeParameters = "rope_parameters"
        case ropeTheta = "rope_theta"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        attentionHeads = try c.decode(Int.self, forKey: .attentionHeads)
        kvHeads = try c.decode(Int.self, forKey: .kvHeads)
        normEps = try c.decodeIfPresent(Float.self, forKey: .normEps) ?? 1e-5
        convLCache = try c.decodeIfPresent(Int.self, forKey: .convLCache) ?? 3
        convBias = try c.decodeIfPresent(Bool.self, forKey: .convBias) ?? false
        let layers = try c.decode(Int.self, forKey: .hiddenLayers)
        layerTypes = try c.decode([String].self, forKey: .layerTypes)
        guard layerTypes.count == layers else {
            throw DecodingError.dataCorruptedError(
                forKey: .layerTypes, in: c,
                debugDescription: "speech trunk layer_types has \(layerTypes.count) of \(layers)")
        }
        let rope = try c.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeParameters)
        ropeTheta =
            try rope?["rope_theta"]?.asFloat()
            ?? c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1e6
        // LFM2's feed-forward sizing: 2/3 of the nominal width, times the multiplier, rounded
        // up to `block_multiple_of` (2560 -> 1792 in the released encoder).
        var ff =
            try c.decodeIfPresent(Int.self, forKey: .blockFFDim)
            ?? c.decode(Int.self, forKey: .intermediateSize)
        if try c.decodeIfPresent(Bool.self, forKey: .autoAdjust) ?? true {
            let multipleOf = try c.decodeIfPresent(Int.self, forKey: .multipleOf) ?? 256
            ff = Int(Float(2 * ff) / 3)
            if let multiplier = try c.decodeIfPresent(Float.self, forKey: .multiplier) {
                ff = Int(multiplier * Float(ff))
            }
            ff = multipleOf * ((ff + multipleOf - 1) / multipleOf)
        }
        feedForwardSize = ff
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(hiddenSize, forKey: .hiddenSize)
        try c.encode(feedForwardSize, forKey: .blockFFDim)
        try c.encode(false, forKey: .autoAdjust)
        try c.encode(attentionHeads, forKey: .attentionHeads)
        try c.encode(kvHeads, forKey: .kvHeads)
        try c.encode(layerTypes.count, forKey: .hiddenLayers)
        try c.encode(normEps, forKey: .normEps)
        try c.encode(convLCache, forKey: .convLCache)
        try c.encode(convBias, forKey: .convBias)
        try c.encode(layerTypes, forKey: .layerTypes)
        try c.encode(ropeTheta, forKey: .ropeTheta)
    }
}

public struct LFM3SpeechEncoderConfiguration: Codable, Sendable {
    public let convChannels: Int
    public let hiddenSize: Int
    public let melBins: Int
    public let slidingWindow: Int
    public let trunk: LFM3SpeechTrunkConfiguration

    public init(
        convChannels: Int, hiddenSize: Int, melBins: Int, slidingWindow: Int,
        trunk: LFM3SpeechTrunkConfiguration
    ) {
        self.convChannels = convChannels
        self.hiddenSize = hiddenSize
        self.melBins = melBins
        self.slidingWindow = slidingWindow
        self.trunk = trunk
    }
}

/// Both released layouts: LFM3-Audio's training export (`lfm`, `encoder`, `depthformer`)
/// and LFM3-Omni's canonical one (`text_config`, `vision_config`, `audio_encoder_config`).
public struct LFM3OmniConfiguration: Codable, Sendable {
    public let modelType: String
    public let textConfiguration: LFM3Configuration
    public let speechEncoder: LFM3SpeechEncoderConfiguration
    /// Present for LFM3-Omni, absent for LFM3-Audio.
    public let vision: LFM3VLConfiguration?
    public let audioTokenId: Int
    public let audioOutTokenId: Int
    public let imageTokenId: Int
    public let videoTokenId: Int

    private enum Keys: String, CodingKey {
        case modelType = "model_type"
        case lfm, encoder
        case textConfig = "text_config"
        case visionConfig = "vision_config"
        case audioEncoderConfig = "audio_encoder_config"
        case audioTokenId = "audio_token_id"
        case audioOutTokenId = "audio_out_token_id"
        case imageTokenId = "image_token_id"
        case videoTokenId = "video_token_id"
    }

    private enum AudioEncoderKeys: String, CodingKey {
        case dimIn = "dim_in", dimConv = "dim_conv", dimOut = "dim_out", lfm
        case convChannels = "conv_channels", hiddenSize = "hidden_size"
        case melBins = "num_mel_bins", textConfig = "text_config"
        case slidingWindow = "sliding_window"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: Keys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "lfm3_audio"
        audioTokenId = try c.decodeIfPresent(Int.self, forKey: .audioTokenId) ?? 125031
        audioOutTokenId = try c.decodeIfPresent(Int.self, forKey: .audioOutTokenId) ?? 125032
        imageTokenId = try c.decodeIfPresent(Int.self, forKey: .imageTokenId) ?? 125030
        videoTokenId = try c.decodeIfPresent(Int.self, forKey: .videoTokenId) ?? 125029

        let e: KeyedDecodingContainer<AudioEncoderKeys>
        if c.contains(.lfm) {
            textConfiguration = try c.decode(LFM3Configuration.self, forKey: .lfm)
            e = try c.nestedContainer(keyedBy: AudioEncoderKeys.self, forKey: .encoder)
        } else {
            textConfiguration = try c.decode(LFM3Configuration.self, forKey: .textConfig)
            e = try c.nestedContainer(keyedBy: AudioEncoderKeys.self, forKey: .audioEncoderConfig)
        }
        speechEncoder = LFM3SpeechEncoderConfiguration(
            convChannels: try e.decodeIfPresent(Int.self, forKey: .dimConv)
                ?? e.decodeIfPresent(Int.self, forKey: .convChannels) ?? 256,
            hiddenSize: try e.decodeIfPresent(Int.self, forKey: .dimOut)
                ?? e.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 768,
            melBins: try e.decodeIfPresent(Int.self, forKey: .dimIn)
                ?? e.decodeIfPresent(Int.self, forKey: .melBins) ?? 128,
            slidingWindow: try e.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 32,
            trunk: try e.decodeIfPresent(LFM3SpeechTrunkConfiguration.self, forKey: .lfm)
                ?? e.decode(LFM3SpeechTrunkConfiguration.self, forKey: .textConfig))

        // Decided by model type, not by key presence: the quantized LFM3-Audio exports carry a
        // stray empty `vision_config` ({}), and a real Omni vision block must fail loudly.
        vision = modelType == "lfm3_omni" ? try LFM3VLConfiguration(from: decoder) : nil
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: Keys.self)
        try c.encode(modelType, forKey: .modelType)
        try c.encode(textConfiguration, forKey: .textConfig)
        var e = c.nestedContainer(keyedBy: AudioEncoderKeys.self, forKey: .audioEncoderConfig)
        try e.encode(speechEncoder.convChannels, forKey: .convChannels)
        try e.encode(speechEncoder.hiddenSize, forKey: .hiddenSize)
        try e.encode(speechEncoder.melBins, forKey: .melBins)
        try e.encode(speechEncoder.slidingWindow, forKey: .slidingWindow)
        try e.encode(speechEncoder.trunk, forKey: .textConfig)
        try c.encode(audioTokenId, forKey: .audioTokenId)
        try c.encode(audioOutTokenId, forKey: .audioOutTokenId)
        try c.encode(imageTokenId, forKey: .imageTokenId)
        try c.encode(videoTokenId, forKey: .videoTokenId)
        try vision?.encode(to: encoder)
    }
}

// MARK: - Features

/// The reference `Lfm3AudioFeatureExtractor`: 16 kHz mono, 25 ms periodic-Hann frames every
/// 10 ms with NO centering (the clip is right-padded to 400 + k*160 samples), power spectrum,
/// 128 Slaney mel bins, log10 floored at -12, then (x + 4) / 4.
enum LFM3SpeechFeatures {
    static let sampleRate = 16_000
    static let fft = 400
    static let hop = 160
    static let bins = 128
    static let logFloor: Float = -12

    static func frameCount(samples: Int) -> Int {
        1 + max(0, (samples - fft + hop - 1) / hop)
    }

    /// Backbone embeddings a clip of `frames` feature frames becomes (one per 8 frames).
    static func encoderTokens(frames: Int) -> Int { (frames + 7) / 8 }

    /// Slaney-scale, Slaney-normalized triangular filters, built in Double as the reference's
    /// `mel_filters(..., precise=True)`: `[fft/2 + 1, bins]`, row-major.
    static let filterbank: [Float] = {
        func hzToMel(_ f: Double) -> Double {
            f < 1000 ? 3 * f / 200 : 15 + log(f / 1000) / (log(6.4) / 27)
        }
        func melToHz(_ m: Double) -> Double {
            m < 15 ? 200 * m / 3 : 1000 * exp((log(6.4) / 27) * (m - 15))
        }
        let frequencyBins = fft / 2 + 1
        let maxMel = hzToMel(Double(sampleRate) / 2)
        let corners = (0 ..< bins + 2).map { melToHz(maxMel * Double($0) / Double(bins + 1)) }
        var weights = [Float](repeating: 0, count: frequencyBins * bins)
        for k in 0 ..< frequencyBins {
            let f = Double(k) * Double(sampleRate) / Double(fft)
            for m in 0 ..< bins {
                let lower = (f - corners[m]) / (corners[m + 1] - corners[m])
                let upper = (corners[m + 2] - f) / (corners[m + 2] - corners[m + 1])
                weights[k * bins + m] = Float(
                    max(0, min(lower, upper)) * 2 / (corners[m + 2] - corners[m]))
            }
        }
        return weights
    }()

    static let window: [Float] = (0 ..< fft).map {
        Float(0.5 - 0.5 * cos(2 * Double.pi * Double($0) / Double(fft)))
    }

    /// One clip -> `[bins, frames]` features.
    static func logMel(_ pcm: [Float]) -> MLXArray {
        let frames = frameCount(samples: pcm.count)
        var padded = pcm
        padded.append(contentsOf: repeatElement(0, count: fft + (frames - 1) * hop - pcm.count))
        var framed = [Float](repeating: 0, count: frames * fft)
        for t in 0 ..< frames {
            for i in 0 ..< fft { framed[t * fft + i] = padded[t * hop + i] * window[i] }
        }
        // A true length-400 DFT: zero-padding to a power of two moves the frequency grid.
        let spectrum = MLXFFT.rfft(MLXArray(framed, [frames, fft]), n: fft, axis: -1, stream: .cpu)
        let magnitude = abs(spectrum)
        let mel = matmul(magnitude * magnitude, MLXArray(filterbank, [fft / 2 + 1, bins]))
        let logMel = maximum(log(mel) / Float(log(10.0)), MLXArray(logFloor))
        return ((logMel + 4) / 4).transposed(1, 0)
    }
}

// MARK: - Speech encoder

enum LFM3Speech {
    private static func fp32Silu(_ x: MLXArray) -> MLXArray {
        // The reference evaluates activations of bf16 inputs in fp32 and rounds once.
        silu(x.asType(.float32)).asType(x.dtype)
    }

    /// 128 mel bins x T frames -> T/8 vectors: three stride-2 3x3 convolutions over the
    /// (mel, time) plane, then a linear map of each time step's (channel, mel) block.
    final class Downsampler: Module {
        @ModuleInfo(key: "conv_in") var convIn: Conv2d
        @ModuleInfo(key: "conv_mid") var convMid: Conv2d
        @ModuleInfo(key: "lin_mid") var linMid: Conv2d
        @ModuleInfo(key: "conv_end") var convEnd: Conv2d
        @ModuleInfo(key: "lin_end") var linEnd: Conv2d
        @ModuleInfo(key: "lin_out") var linOut: Linear

        init(_ config: LFM3SpeechEncoderConfiguration) {
            let c = config.convChannels
            _convIn.wrappedValue = Conv2d(
                inputChannels: 1, outputChannels: c, kernelSize: 3, stride: 2, padding: 1)
            _convMid.wrappedValue = Conv2d(
                inputChannels: c, outputChannels: c, kernelSize: 3, stride: 2, padding: 1, groups: c)
            _linMid.wrappedValue = Conv2d(inputChannels: c, outputChannels: c, kernelSize: 1)
            _convEnd.wrappedValue = Conv2d(
                inputChannels: c, outputChannels: c, kernelSize: 3, stride: 2, padding: 1, groups: c)
            _linEnd.wrappedValue = Conv2d(inputChannels: c, outputChannels: c, kernelSize: 1)
            _linOut.wrappedValue = Linear(config.melBins * c / 8, config.hiddenSize)
        }

        /// `features`: `[bins, T]` for one clip -> `[1, ceil(T / 8), hidden]`.
        func callAsFunction(_ features: MLXArray) -> MLXArray {
            // NHWC: H = mel bins, W = time.
            var x = features.expandedDimensions(axes: [0, -1]).asType(convIn.weight.dtype)
            x = fp32Silu(convIn(x))
            x = fp32Silu(linMid(convMid(x)))
            x = fp32Silu(linEnd(convEnd(x)))
            let (b, f, t, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
            // The reference flattens (channel, mel) channel-major.
            return linOut(x.transposed(0, 2, 3, 1).reshaped(b, t, c * f))
        }
    }

    final class Attention: Module {
        let heads: Int
        let kvHeads: Int
        let headDim: Int
        let scale: Float
        let rope: RoPE

        @ModuleInfo(key: "q_proj") var qProj: Linear
        @ModuleInfo(key: "k_proj") var kProj: Linear
        @ModuleInfo(key: "v_proj") var vProj: Linear
        @ModuleInfo(key: "out_proj") var outProj: Linear
        @ModuleInfo(key: "q_layernorm") var qNorm: RMSNorm
        @ModuleInfo(key: "k_layernorm") var kNorm: RMSNorm

        init(_ config: LFM3SpeechTrunkConfiguration) {
            heads = config.attentionHeads
            kvHeads = config.kvHeads
            headDim = config.headDim
            scale = pow(Float(headDim), -0.5)
            rope = RoPE(dimensions: headDim, traditional: false, base: config.ropeTheta)
            let d = config.hiddenSize
            _qProj.wrappedValue = Linear(d, heads * headDim, bias: false)
            _kProj.wrappedValue = Linear(d, kvHeads * headDim, bias: false)
            _vProj.wrappedValue = Linear(d, kvHeads * headDim, bias: false)
            _outProj.wrappedValue = Linear(heads * headDim, d, bias: false)
            _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.normEps)
            _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.normEps)
        }

        func callAsFunction(_ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode)
            -> MLXArray
        {
            let (b, l) = (x.dim(0), x.dim(1))
            let q = rope(qNorm(qProj(x).reshaped(b, l, heads, headDim)).transposed(0, 2, 1, 3))
            let k = rope(kNorm(kProj(x).reshaped(b, l, kvHeads, headDim)).transposed(0, 2, 1, 3))
            let v = vProj(x).reshaped(b, l, kvHeads, headDim).transposed(0, 2, 1, 3)
            let out = MLXFast.scaledDotProductAttention(
                queries: q, keys: k, values: v, scale: scale, mask: mask)
            return outProj(out.transposed(0, 2, 1, 3).reshaped(b, l, -1))
        }
    }

    final class ShortConv: Module {
        let hiddenSize: Int
        let lCache: Int

        @ModuleInfo(key: "conv") var conv: Conv1d
        @ModuleInfo(key: "in_proj") var inProj: Linear
        @ModuleInfo(key: "out_proj") var outProj: Linear

        init(_ config: LFM3SpeechTrunkConfiguration) {
            hiddenSize = config.hiddenSize
            lCache = config.convLCache
            _conv.wrappedValue = Conv1d(
                inputChannels: hiddenSize, outputChannels: hiddenSize, kernelSize: lCache,
                groups: hiddenSize, bias: config.convBias)
            _inProj.wrappedValue = Linear(hiddenSize, 3 * hiddenSize, bias: config.convBias)
            _outProj.wrappedValue = Linear(hiddenSize, hiddenSize, bias: config.convBias)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let parts = inProj(x).split(parts: 3, axis: -1)
            let bx = parts[0] * parts[2]
            // Causal: the window starts from zeros, as an uncached LFM2 conv does.
            let padded = concatenated(
                [MLXArray.zeros([bx.dim(0), lCache - 1, hiddenSize], dtype: bx.dtype), bx],
                axis: -2)
            return outProj(parts[1] * conv(padded))
        }
    }

    final class MLP: Module, UnaryLayer {
        @ModuleInfo(key: "w1") var w1: Linear
        @ModuleInfo(key: "w2") var w2: Linear
        @ModuleInfo(key: "w3") var w3: Linear

        init(_ config: LFM3SpeechTrunkConfiguration) {
            _w1.wrappedValue = Linear(config.hiddenSize, config.feedForwardSize, bias: false)
            _w2.wrappedValue = Linear(config.feedForwardSize, config.hiddenSize, bias: false)
            _w3.wrappedValue = Linear(config.hiddenSize, config.feedForwardSize, bias: false)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray { w2(silu(w1(x)) * w3(x)) }
    }

    final class Layer: Module {
        let isAttention: Bool

        @ModuleInfo(key: "self_attn") var attention: Attention?
        @ModuleInfo(key: "conv") var conv: ShortConv?
        @ModuleInfo(key: "feed_forward") var feedForward: MLP
        @ModuleInfo(key: "operator_norm") var operatorNorm: RMSNorm
        @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm

        init(_ config: LFM3SpeechTrunkConfiguration, index: Int) {
            isAttention = config.layerTypes[index] == "full_attention"
            if isAttention {
                _attention.wrappedValue = Attention(config)
            } else {
                _conv.wrappedValue = ShortConv(config)
            }
            _feedForward.wrappedValue = MLP(config)
            _operatorNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.normEps)
            _ffnNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.normEps)
        }

        func callAsFunction(_ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode)
            -> MLXArray
        {
            let h = operatorNorm(x)
            let x = x + (isAttention ? attention!(h, mask: mask) : conv!(h))
            return x + feedForward(ffnNorm(x))
        }
    }

    final class Trunk: Module {
        let layers: [Layer]
        @ModuleInfo(key: "embedding_norm") var embeddingNorm: RMSNorm

        init(_ config: LFM3SpeechTrunkConfiguration) {
            layers = (0 ..< config.layerTypes.count).map { Layer(config, index: $0) }
            _embeddingNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.normEps)
        }

        func callAsFunction(_ h: MLXArray, window: Int) -> MLXArray {
            // Causal, each frame seeing at most the last `window` frames (itself included).
            let mask = MLXFast.ScaledDotProductAttentionMaskMode.array(
                createCausalMask(n: h.dim(1), offset: 0, windowSize: window))
            var h = h
            for layer in layers { h = layer(h, mask: mask) }
            return embeddingNorm(h)
        }
    }

    final class Encoder: Module {
        let window: Int
        @ModuleInfo(key: "downsample") var downsample: Downsampler
        @ModuleInfo(key: "lfm") var lfm: Trunk

        init(_ config: LFM3SpeechEncoderConfiguration) {
            window = config.slidingWindow
            _downsample.wrappedValue = Downsampler(config)
            _lfm.wrappedValue = Trunk(config.trunk)
        }

        /// One clip's `[bins, T]` features -> `[ceil(T / 8), hidden]`.
        func callAsFunction(_ features: MLXArray) -> MLXArray {
            lfm(downsample(features), window: window)[0]
        }
    }

    final class Projector: Module {
        @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm
        @ModuleInfo(key: "linear_1") var linear1: Linear
        @ModuleInfo(key: "linear_2") var linear2: Linear

        init(inputSize: Int, outputSize: Int) {
            _layerNorm.wrappedValue = LayerNorm(dimensions: inputSize, eps: 1e-5)
            _linear1.wrappedValue = Linear(inputSize, outputSize)
            _linear2.wrappedValue = Linear(outputSize, outputSize)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let h = linear1(layerNorm(x))
            return linear2(gelu(h.asType(.float32)).asType(h.dtype))
        }
    }
}

// MARK: - Model

/// LFM3-Audio (`lfm3_audio`) and LFM3-Omni (`lfm3_omni`).
public final class LFM3Omni: Module, VLMModel, KVCacheDimensionProvider, ModalityBearing {
    @ModuleInfo(key: "language_model") var languageModel: LFM3Model
    @ModuleInfo(key: "audio_tower") var audioTower: LFM3Speech.Encoder
    @ModuleInfo(key: "audio_projector") var audioProjector: LFM3Speech.Projector
    @ModuleInfo(key: "vision_tower") var visionTower: LFM3VLVision.VisionModel?
    @ModuleInfo(key: "multi_modal_projector") var visionProjector: LFM3VLProjector?

    public let config: LFM3OmniConfiguration
    /// The text head never samples a media placeholder (the reference sets these to -inf).
    let placeholderBias: MLXArray

    public init(_ config: LFM3OmniConfiguration) {
        self.config = config
        _languageModel.wrappedValue = LFM3Model(config.textConfiguration)
        _audioTower.wrappedValue = LFM3Speech.Encoder(config.speechEncoder)
        _audioProjector.wrappedValue = LFM3Speech.Projector(
            inputSize: config.speechEncoder.hiddenSize,
            outputSize: config.textConfiguration.hiddenSize)
        if let vision = config.vision {
            _visionTower.wrappedValue = LFM3VLVision.VisionModel(vision.visionConfiguration)
            _visionProjector.wrappedValue = LFM3VLProjector(vision)
        }
        var placeholders = [config.audioTokenId, config.audioOutTokenId]
        if config.vision != nil { placeholders += [config.imageTokenId, config.videoTokenId] }
        var bias = [Float](repeating: 0, count: config.textConfiguration.vocabularySize)
        for id in placeholders where id < bias.count { bias[id] = -Float.infinity }
        placeholderBias = MLXArray(bias)
    }

    public var vocabularySize: Int { config.textConfiguration.vocabularySize }
    public var kvHeads: [Int] { languageModel.kvHeads }
    public var loraLayers: [Module] { languageModel.loraLayers }
    public var modalities: Set<ModelRuntimeRequestModality> {
        config.vision == nil ? [.text, .audio] : [.text, .audio, .vision, .video]
    }

    func speechFeatures(pcm: [Float]) -> MLXArray {
        audioProjector(audioTower(LFM3SpeechFeatures.logMel(pcm)))
    }

    func inputEmbeddings(for input: LMInput) throws -> MLXArray {
        let ids = input.text.tokens.reshaped(1, -1)
        var embeddings = languageModel.model.embedTokens(ids)
        let tokenIds = ids.reshaped(-1).asArray(Int.self)
        if let visionTower, let visionProjector {
            let media: [(MLXArray?, [THW]?, Int)] = [
                (input.image?.pixels, input.image?.frames, config.imageTokenId),
                (input.video?.pixels, input.video?.frames, config.videoTokenId),
            ]
            for case (let pixels?, let grids?, let tokenId) in media where !grids.isEmpty {
                embeddings = try LFM3VL.scatter(
                    visionProjector(try visionTower(pixels, grids: grids)), into: embeddings,
                    tokenIds: tokenIds, tokenId: tokenId)
            }
        }
        if let audio = input.audio {
            let pcm = audio.waveform.reshaped(-1).asType(.float32).asArray(Float.self)
            var clips: [[Float]] = []
            var start = 0
            for count in audio.clipSampleCounts ?? [pcm.count] {
                clips.append(Array(pcm[start ..< min(start + count, pcm.count)]))
                start += count
            }
            let features = concatenated(clips.map { speechFeatures(pcm: $0) }, axis: 0)
            embeddings = try LFM3VL.scatter(
                features, into: embeddings, tokenIds: tokenIds, tokenId: config.audioTokenId)
        }
        return embeddings
    }

    public func prepare(_ input: LMInput, cache: [any KVCache], windowSize: Int?) throws
        -> PrepareResult
    {
        let embeddings = try inputEmbeddings(for: input)
        let logits = try chunkedPrefillEmbedding(
            inputEmbedding: embeddings, cache: cache, prefillStepSize: windowSize ?? 512
        ) { chunk in
            languageModel.logits(nil, cache: cache, inputEmbeddings: chunk, lastPositionOnly: true)
        }
        return .logits(LMOutput(logits: logits + placeholderBias))
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache) + placeholderBias
    }

    public func newCache(parameters: GenerateParameters?) -> [any KVCache] {
        languageModel.newCache(parameters: parameters)
    }

    /// Accepts the released MLX layout and the training-export / transformers spellings.
    ///
    /// INVARIANT: only the speech INPUT path is built. `depth_decoder.*` and
    /// `codebook_embedding.*` (the 12-codebook speech generator) are dropped here, so these
    /// bundles answer in text; a model that should speak needs those modules and the codec.
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        let legacy: [(String, String)] = [
            ("lfm.model.", "language_model.model."),
            ("encoder.downsample.", "audio_tower.downsample."),
            ("encoder.lfm.lfm.", "audio_tower.lfm."),
            ("audio_encoder.", "audio_tower."),
            ("mlp.model.0.", "audio_projector.layer_norm."),
            ("mlp.model.1.", "audio_projector.linear_1."),
            ("mlp.model.3.", "audio_projector.linear_2."),
            ("model.language_model.", "language_model.model."),
            ("model.vision_tower.", "vision_tower."),
            ("model.multi_modal_projector.", "multi_modal_projector."),
        ]
        var result: [String: MLXArray] = [:]
        for (key, value) in weights {
            var name = key
            for (old, new) in legacy where name.hasPrefix(old) {
                name = new + name.dropFirst(old.count)
                break
            }
            if name.hasPrefix("lm_head.") { name = "language_model." + name }
            if name.hasPrefix("depth") || name.hasPrefix("codebook_embedding")
                || name.hasPrefix("mask_embedding")
            {
                continue
            }
            if config.textConfiguration.tieWordEmbeddings, name.hasPrefix("language_model.lm_head.") {
                continue
            }
            var value = value
            if name.hasSuffix("conv.conv.weight"), value.ndim == 3, value.dim(1) == 1,
                value.dim(2) > 1
            {
                value = value.transposed(0, 2, 1)
            } else if name.hasSuffix("spatial_conv.weight"), value.ndim == 4, value.dim(1) == 1,
                value.dim(3) > 1
            {
                value = value.transposed(0, 2, 3, 1)
            } else if name.hasPrefix("audio_tower.downsample."), value.ndim == 4,
                value.dim(2) == value.dim(3), value.dim(3) > 1 || value.dim(1) > 1
            {
                // torch Conv2d [O, I, k, k] -> MLX [O, k, k, I]. Released MLX bundles are
                // already [O, k, k, I] (3x3 kernels end in I = 1, 1x1 kernels have dims
                // 1 and 2 equal to 1 with I in dim 3), which this test leaves alone.
                value = value.transposed(0, 2, 3, 1)
            } else if name == "audio_tower.downsample.lin_out.weight", value.ndim == 3 {
                value = value[0..., 0..., 0]  // torch Conv1d [O, I, 1] -> Linear
            }
            result[name] = value
        }
        return result
    }
}

// MARK: - Processor

/// Speech (and for Omni, image / video) placeholders for LFM3-Audio and LFM3-Omni.
///
/// INVARIANT: each `<|audio|>` expands to ceil(frames / 8) audio tokens, where frames is the
/// clip's feature-frame count at 16 kHz; the decoder scatters one encoder row per token. The
/// templates differ (Audio leaves `<|audio|>` bare, Omni wraps it in vision_start/end) and only
/// the inner token is replaced, so both stay exactly as the reference builds them.
public struct LFM3OmniProcessor: UserInputProcessor {
    private let tokenizer: any Tokenizer
    /// Present for LFM3-Omni bundles (image / video settings).
    private let vision: LFM3VLProcessor?

    public init(tokenizer: any Tokenizer, vision: Qwen3VLProcessorConfiguration?) {
        self.tokenizer = tokenizer
        self.vision = vision.map { LFM3VLProcessor($0, tokenizer: tokenizer) }
    }

    static func waveform(_ audio: UserInput.Audio) throws -> [Float] {
        let rate = LFM3SpeechFeatures.sampleRate
        switch audio {
        case .url(let url):
            return try nemotronOmniLoadAudioFile(url, targetSampleRate: Double(rate))
        case .samples(let pcm, let sr), .preEncoded(let pcm, let sr, _):
            return sr == rate ? pcm : linearResamplePCM(pcm, fromRate: sr, toRate: rate)
        case .array(let array, let sr):
            let pcm = array.reshaped(-1).asType(.float32).asArray(Float.self)
            return sr == rate ? pcm : linearResamplePCM(pcm, fromRate: sr, toRate: rate)
        }
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        let messages = LFM3MessageGenerator().generate(from: input)
        let rendered = try tokenizer.applyChatTemplate(
            messages: messages, tools: input.tools, additionalContext: input.additionalContext)
        let boundaries = canonicalChatCacheBoundaries(
            tokenizer: tokenizer, messages: messages, tools: input.tools,
            additionalContext: input.additionalContext, promptTokens: rendered,
            staticSystemPrefix: input.cacheStableSystemPrefix)

        var media = LFM3VLProcessor.VisualMedia(
            tokens: rendered, boundaries: boundaries, image: nil, video: nil)
        if let vision, !(input.images.isEmpty && input.videos.isEmpty) {
            media = try await vision.expandVisualMedia(
                tokens: rendered, boundaries: boundaries, input: input)
        } else if !(input.images.isEmpty && input.videos.isEmpty) {
            throw VLMError.processing("LFM3-Audio does not read images or video")
        }

        var processedAudio: LMInput.ProcessedAudio?
        if !input.audios.isEmpty {
            guard let id = tokenizer.convertTokenToId("<|audio|>") else {
                throw VLMError.processing("LFM3 tokenizer has no <|audio|>")
            }
            let clips = try input.audios.map(Self.waveform)
            let replacements = clips.map { clip in
                Array(
                    repeating: id,
                    count: LFM3SpeechFeatures.encoderTokens(
                        frames: LFM3SpeechFeatures.frameCount(samples: clip.count)))
            }
            let all = try LFM3VLProcessor.expand(
                media.tokens, placeholder: id, replacements: replacements,
                boundaries: media.boundaries.all)
            let stable = try LFM3VLProcessor.expand(
                media.tokens, placeholder: id, replacements: replacements,
                boundaries: media.boundaries.stable)
            media.tokens = all.tokens
            media.boundaries = CanonicalChatCacheBoundaries(
                all: all.boundaries, stable: stable.boundaries)
            processedAudio = LMInput.ProcessedAudio(
                waveform: MLXArray(clips.flatMap { $0 }),
                sampleRate: LFM3SpeechFeatures.sampleRate,
                clipSampleCounts: clips.map(\.count))
        }

        let promptArray = MLXArray(media.tokens).expandedDimensions(axis: 0)
        return LMInput(
            text: .init(
                tokens: promptArray, mask: ones(like: promptArray).asType(.int8),
                tokenIds: media.tokens),
            image: media.image,
            video: media.video,
            audio: processedAudio,
            mediaTokenIds: QwenVL.mediaTokenIds(
                tokenizer: tokenizer, tokens: ["<|image|>", "<|video|>", "<|audio|>"]),
            cacheScopeSalt: cacheScopeSalt(from: input.additionalContext),
            cachePrefixTokenCounts: media.boundaries.all,
            cacheStablePrefixTokenCounts: media.boundaries.stable,
            toolSchemas: input.tools)
    }
}

/// Content items the LFM3 templates expand: media first (`image`, `video`, `audio`), then text.
public struct LFM3MessageGenerator: MessageGenerator {
    public init() {}

    public func generate(message: Chat.Message) -> MLXLMCommon.Message {
        var result = defaultMessageDict(for: message)
        guard !(message.images.isEmpty && message.videos.isEmpty && message.audios.isEmpty)
        else { return result }
        result["content"] =
            message.images.map { _ in ["type": "image"] }
            + message.videos.map { _ in ["type": "video"] }
            + message.audios.map { _ in ["type": "audio"] }
            + [["type": "text", "text": message.content]]
        return result
    }
}
