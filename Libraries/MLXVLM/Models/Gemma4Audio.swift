// Copyright © 2024-2026 Jinho Jang (eric@jangq.ai)
//
// Gemma 4 E-series (E2B/E4B) audio tower — Universal Speech Model (USM)
// style conformer encoder. Port source of truth:
// transformers/models/gemma4/{feature_extraction,configuration,modeling}_gemma4.py
//
// Pipeline (exact HF parity, all params verified against the
// OsaurusAI--gemma-4-E2B-it-qat-JANG_4M bundle's config.json /
// processor_config.json and the bundled-python transformers reference):
//
//   1. Waveform → log-mel frames `[T, 128]` (Gemma4AudioFeatureExtractor):
//      - 16 kHz mono PCM, truncated at 480 000 samples (30 s)
//      - semicausal padding: prepend frame_length/2 = 160 zeros
//      - frame_length 320 (20 ms), hop_length 160 (10 ms)
//      - per frame: take 320 samples (preemphasis = 0), multiply by a
//        PERIODIC Hann window w[n] = 0.5 − 0.5·cos(2πn/320)
//      - rFFT with fft_length 512 (frame right-zero-padded), MAGNITUDE
//        spectrum (not power)
//      - HTK mel filterbank: 128 filters, 0…8000 Hz, norm=None,
//        mel = 2595·log10(1 + f/700); fft bins linspace(0, 8000, 257)
//      - log(mel + mel_floor 1e-3); no per-bin mean/stddev in shipped configs
//   2. Subsampling: two Conv2d k=3 s=2 p=1 blocks over (time, mel) with
//      channels 1 → 128 → 32, each followed by bias-less LayerNorm over
//      channels + ReLU; flatten (mel/4=32) × 32ch = 1024 → input_proj_linear
//      (1024 → hidden_size 1024). Time reduction ×4: T → ⌈T/2⌉ → ⌈T/4⌉.
//   3. 12 conformer layers (hidden 1024, 8 heads, head_dim 128):
//      ff1 → (clip, RMSNorm) → chunked local self-attention → (clip,
//      RMSNorm) + residual → causal light conv (GLU + depthwise k=5)
//      → ff2 → (clip, RMSNorm out). Feed-forwards scale their branch by
//      residual_weight 0.5. RMSNorm eps 1e-6, weight used directly (no +1).
//      Attention: chunk 12, left context 13 (max_past = 12), right 0,
//      logit softcap 50 (tanh), invalid logit −1e9, relative position
//      bias from a sinusoidal table of context_size/2 + 1 = 13 positions,
//      q scaled by head_dim^-0.5/ln2 · softplus(per_dim_scale),
//      k scaled by ln(1+e)/ln2. The blocked validity mask is ANDed with a
//      sliding window of (attention_context_left − 1, attention_context_right)
//      per HF `create_bidirectional_mask(and_mask_function=...)`.
//      NOTE: the bundled-python transformers snapshot has a mask-dtype bug —
//      `create_bidirectional_mask` returns a float ADDITIVE mask (0 / −inf)
//      but `Gemma4AudioAttention.forward` applies it with
//      `masked_fill(mask.logical_not(), −1e9)`, which treats 0.0 as "drop"
//      and −inf as "keep", i.e. attends the COMPLEMENT of the window. This
//      port implements the intended (trained) window semantics instead;
//      proof: verbatim transcription of synthesized speech through the E2B
//      QAT bundle (see /tmp/gemma4-audio-proof and the PR notes), which is
//      impossible with the inverted mask.
//   4. output_proj (1024 → output_proj_dims 1536, bias).
//   5. `embed_audio` (Gemma4MultimodalEmbedder in Gemma4.swift) consumes the
//      1536-wide tower output: RMSNorm(no scale) → Linear(1536 → text hidden).
//
// Clipped linears (use_clipped_linears = true): every q/k/v/post and
// ffw/lconv linear is wrapped with scalar input_min/input_max/output_min/
// output_max tensors from the checkpoint; inference clamps the linear's
// input and output to those bounds (HF Gemma4ClippableLinear.forward).
//
// Soft-token count parity (HF Gemma4Processor.replace_audio_token):
// simulate the two k=3 s=2 p=1 convs on the frame mask —
// t_out = (t − 1)/2 + 1 applied twice, keeping every other frame — which
// for a contiguous valid prefix is exactly ⌈T/4⌉ tokens. 30 s of audio
// → 2999 mel frames → 750 tokens = audio_seq_length.
//
// Design: the mel extractor runs in `Gemma4Processor.prepare` (CPU/vDSP,
// no model weights needed) so the processor can expand `<|audio|>`
// placeholders to the exact post-subsampling token count. The conformer
// tower needs checkpoint weights, so it runs inside `Gemma4.prepare`:
// mel frames travel in `LMInput.ProcessedAudio.waveform` (shape
// `[N, T, 128]`, padded rows exactly 0 like the HF extractor's
// mask-zeroed output) with `preEncodedEmbedding == nil`; pre-encoded
// embeddings keep bypassing the tower.

import Accelerate
import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Configuration

/// `config.json` → `audio_config` for `model_type == "gemma4_audio"`
/// (E2B/E4B). Defaults mirror HF `Gemma4AudioConfig`.
public struct Gemma4AudioConfig: Codable, Sendable {
    public let modelType: String
    public let hiddenSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let subsamplingConvChannels: [Int]
    public let convKernelSize: Int
    public let residualWeight: Float
    public let attentionChunkSize: Int
    public let attentionContextLeft: Int
    public let attentionContextRight: Int
    public let attentionLogitCap: Float
    public let attentionInvalidLogitsValue: Float
    public let useClippedLinears: Bool
    public let rmsNormEps: Float
    public let gradientClipping: Float
    public let outputProjDims: Int

    /// The conformer tower is only instantiated for the E-series
    /// `gemma4_audio` config. The unified 12B bundles carry
    /// `model_type == "gemma4_unified_audio"` (encoder-free raw chunking)
    /// and must NOT build a tower.
    public var isConformerTower: Bool { modelType == "gemma4_audio" }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case subsamplingConvChannels = "subsampling_conv_channels"
        case convKernelSize = "conv_kernel_size"
        case residualWeight = "residual_weight"
        case attentionChunkSize = "attention_chunk_size"
        case attentionContextLeft = "attention_context_left"
        case attentionContextRight = "attention_context_right"
        case attentionLogitCap = "attention_logit_cap"
        case attentionInvalidLogitsValue = "attention_invalid_logits_value"
        case useClippedLinears = "use_clipped_linears"
        case rmsNormEps = "rms_norm_eps"
        case gradientClipping = "gradient_clipping"
        case outputProjDims = "output_proj_dims"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? ""
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1024
        numHiddenLayers = try c.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 12
        numAttentionHeads = try c.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 8
        subsamplingConvChannels =
            try c.decodeIfPresent([Int].self, forKey: .subsamplingConvChannels) ?? [128, 32]
        convKernelSize = try c.decodeIfPresent(Int.self, forKey: .convKernelSize) ?? 5
        residualWeight = try c.decodeIfPresent(Float.self, forKey: .residualWeight) ?? 0.5
        attentionChunkSize = try c.decodeIfPresent(Int.self, forKey: .attentionChunkSize) ?? 12
        attentionContextLeft = try c.decodeIfPresent(Int.self, forKey: .attentionContextLeft) ?? 13
        attentionContextRight =
            try c.decodeIfPresent(Int.self, forKey: .attentionContextRight) ?? 0
        attentionLogitCap = try c.decodeIfPresent(Float.self, forKey: .attentionLogitCap) ?? 50.0
        attentionInvalidLogitsValue =
            try c.decodeIfPresent(Float.self, forKey: .attentionInvalidLogitsValue) ?? -1.0e9
        useClippedLinears = try c.decodeIfPresent(Bool.self, forKey: .useClippedLinears) ?? true
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        gradientClipping = try c.decodeIfPresent(Float.self, forKey: .gradientClipping) ?? 1e10
        outputProjDims = try c.decodeIfPresent(Int.self, forKey: .outputProjDims) ?? 1536
    }
}

// MARK: - Mel feature extractor (Gemma4AudioFeatureExtractor parity)

/// Fixed extractor parameters from `processor_config.json` / HF defaults.
/// All shipped E-series bundles use exactly these values (verified against
/// OsaurusAI--gemma-4-E2B-it-qat-JANG_4M processor_config.json).
enum Gemma4AudioMel {
    static let sampleRate = 16_000
    static let melBins = 128
    static let frameLength = 320  // 20 ms
    static let hopLength = 160  // 10 ms
    static let fftLength = 512  // 2^ceil(log2(320)), fft_overdrive=false
    static let melFloor: Float = 1e-3
    static let minFrequency: Float = 0
    static let maxFrequency: Float = 8000
    /// Feature-extractor `max_length` — 30 s at 16 kHz, yielding 2999 mel
    /// frames → exactly `audio_seq_length` (750) soft tokens.
    static let maxSamples = 480_000
}

/// HF `Gemma4Processor.replace_audio_token` parity: number of `<|audio|>`
/// soft tokens for `melFrameCount` valid mel frames. Two simulated
/// k=3 s=2 p=1 convs keep frames at indices ≡ 0 (mod 4), i.e. ⌈T/4⌉
/// for a contiguous valid prefix. This must equal the conformer tower's
/// valid output length so `maskedScatter` sees a 1:1 match.
func gemma4AudioSoftTokenCount(melFrameCount: Int) -> Int {
    melFrameCount > 0 ? (melFrameCount + 3) / 4 : 0
}

/// HTK mel filterbank, transformers `mel_filter_bank(norm=None,
/// mel_scale="htk")` parity. Returns `[nBins][nMels]` (257 × 128).
private func gemma4HTKMelFilterbank(
    nBins: Int, nMels: Int, minFrequency: Float, maxFrequency: Float
) -> [[Float]] {
    func hzToMel(_ f: Float) -> Float { 2595.0 * log10(1.0 + f / 700.0) }
    func melToHz(_ m: Float) -> Float { 700.0 * (pow(10.0, m / 2595.0) - 1.0) }

    let melMin = hzToMel(minFrequency)
    let melMax = hzToMel(maxFrequency)
    // num_mel_filters + 2 corner frequencies
    var filterFreqs = [Float](repeating: 0, count: nMels + 2)
    for i in 0 ..< (nMels + 2) {
        let mel = melMin + (melMax - melMin) * Float(i) / Float(nMels + 1)
        filterFreqs[i] = melToHz(mel)
    }
    // fft_freqs = linspace(0, sampling_rate // 2, num_frequency_bins)
    var fftFreqs = [Float](repeating: 0, count: nBins)
    for k in 0 ..< nBins {
        fftFreqs[k] = maxFrequency * Float(k) / Float(nBins - 1)
    }

    var filters = [[Float]](repeating: [Float](repeating: 0, count: nMels), count: nBins)
    for k in 0 ..< nBins {
        for m in 0 ..< nMels {
            let down = -(filterFreqs[m] - fftFreqs[k]) / (filterFreqs[m + 1] - filterFreqs[m])
            let up = (filterFreqs[m + 2] - fftFreqs[k]) / (filterFreqs[m + 2] - filterFreqs[m + 1])
            filters[k][m] = max(0, min(down, up))
        }
    }
    return filters
}

private let gemma4MelFilterbank: [[Float]] = gemma4HTKMelFilterbank(
    nBins: Gemma4AudioMel.fftLength / 2 + 1,
    nMels: Gemma4AudioMel.melBins,
    minFrequency: Gemma4AudioMel.minFrequency,
    maxFrequency: Gemma4AudioMel.maxFrequency)

/// Periodic Hann window of `frameLength`: w[n] = 0.5 − 0.5·cos(2πn/N).
private let gemma4HannWindow: [Float] = (0 ..< Gemma4AudioMel.frameLength).map { n in
    0.5 - 0.5 * cos(2.0 * Float.pi * Float(n) / Float(Gemma4AudioMel.frameLength))
}

/// Extract Gemma4 log-mel features from 16 kHz mono PCM.
/// Returns `[T, 128]` Float32; `T = (L − 1)/hop + 1 − frame/hop` over the
/// semicausally padded waveform (≥ 1 frame requires ≥ 161 input samples;
/// shorter clips are zero-extended to the minimum analyzable length).
func gemma4ExtractMelFeatures(_ pcm: [Float]) -> MLXArray {
    let frameLength = Gemma4AudioMel.frameLength
    let hop = Gemma4AudioMel.hopLength
    let nFFT = Gemma4AudioMel.fftLength
    let nBins = nFFT / 2 + 1
    let nMels = Gemma4AudioMel.melBins

    var samples = pcm
    if samples.count > Gemma4AudioMel.maxSamples {
        samples = Array(samples.prefix(Gemma4AudioMel.maxSamples))
    }
    // The unfold window is frame_length + 1 samples; with the semicausal
    // left pad of frame_length/2 the first frame needs L ≥ 161. Zero-extend
    // ultra-short clips so they still produce one frame.
    let minSamples = frameLength + 1 - frameLength / 2
    if samples.count < minSamples {
        samples.append(contentsOf: [Float](repeating: 0, count: minSamples - samples.count))
    }

    // Semicausal time padding: prepend frame_length/2 zeros.
    var padded = [Float](repeating: 0, count: frameLength / 2)
    padded.append(contentsOf: samples)

    let windowSize = frameLength + 1
    let nFrames = (padded.count - windowSize) / hop + 1

    let log2n = vDSP_Length(log2(Double(nFFT)))
    guard let setup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2)) else {
        return MLXArray.zeros([0, nMels])
    }
    defer { vDSP_destroy_fftsetup(setup) }

    var mel = [Float](repeating: 0, count: nFrames * nMels)
    var frame = [Float](repeating: 0, count: nFFT)
    var realIn = [Float](repeating: 0, count: nFFT / 2)
    var imagIn = [Float](repeating: 0, count: nFFT / 2)
    var magnitude = [Float](repeating: 0, count: nBins)

    for f in 0 ..< nFrames {
        let start = f * hop
        // preemphasis == 0 → frame = first frame_length samples of the
        // (frame_length + 1)-sample unfold window, then Hann, then
        // right-zero-pad to fft_length (np.fft.rfft(n=512) semantics).
        for i in 0 ..< frameLength {
            frame[i] = padded[start + i] * gemma4HannWindow[i]
        }
        for i in frameLength ..< nFFT { frame[i] = 0 }

        frame.withUnsafeBytes { rawBuf in
            let ptr = rawBuf.baseAddress!.assumingMemoryBound(to: DSPComplex.self)
            realIn.withUnsafeMutableBufferPointer { rPtr in
                imagIn.withUnsafeMutableBufferPointer { iPtr in
                    var split = DSPSplitComplex(
                        realp: rPtr.baseAddress!, imagp: iPtr.baseAddress!)
                    vDSP_ctoz(ptr, 2, &split, 1, vDSP_Length(nFFT / 2))
                    vDSP_fft_zrip(setup, &split, 1, log2n, FFTDirection(FFT_FORWARD))
                    // vDSP zrip output is 2× numpy's rfft; scale by 0.5.
                    // Magnitude spectrum (NOT power) per the HF extractor.
                    let r0 = rPtr[0] * 0.5
                    magnitude[0] = abs(r0)
                    let rn = iPtr[0] * 0.5
                    magnitude[nFFT / 2] = abs(rn)
                    for k in 1 ..< (nFFT / 2) {
                        let re = rPtr[k] * 0.5
                        let im = iPtr[k] * 0.5
                        magnitude[k] = sqrt(re * re + im * im)
                    }
                }
            }
        }

        for m in 0 ..< nMels {
            var s: Float = 0
            for k in 0 ..< nBins {
                s += magnitude[k] * gemma4MelFilterbank[k][m]
            }
            mel[f * nMels + m] = log(s + Gemma4AudioMel.melFloor)
        }
    }

    return MLXArray(mel).reshaped(nFrames, nMels)
}

// MARK: - Tower building blocks

/// RMSNorm with scale, weight used directly (HF Gemma4RMSNorm, no +1).
private class AudioRMSNorm: Module, UnaryLayer {
    let weight: MLXArray
    let eps: Float
    init(dimensions: Int, eps: Float = 1e-6) {
        self.weight = MLXArray.ones([dimensions])
        self.eps = eps
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: weight, eps: eps)
    }
}

/// LayerNorm with elementwise affine weight but NO bias
/// (HF `nn.LayerNorm(..., elementwise_affine=True, bias=False)`).
private class AudioLayerNorm: Module, UnaryLayer {
    let weight: MLXArray
    let eps: Float
    init(dimensions: Int, eps: Float) {
        self.weight = MLXArray.ones([dimensions])
        self.eps = eps
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let xf = x.asType(.float32)
        let mu = xf.mean(axis: -1, keepDims: true)
        let centered = xf - mu
        let variance = (centered * centered).mean(axis: -1, keepDims: true)
        return (centered * rsqrt(variance + eps) * weight.asType(.float32)).asType(x.dtype)
    }
}

/// HF Gemma4ClippableLinear: clamp input and output to scalar bounds
/// shipped in the checkpoint (`input_min/input_max/output_min/output_max`).
private class Gemma4ClippedLinear: Module {
    @ModuleInfo(key: "linear") var linear: Linear
    @ParameterInfo(key: "input_min") var inputMin: MLXArray
    @ParameterInfo(key: "input_max") var inputMax: MLXArray
    @ParameterInfo(key: "output_min") var outputMin: MLXArray
    @ParameterInfo(key: "output_max") var outputMax: MLXArray
    let clipped: Bool

    init(_ inputDims: Int, _ outputDims: Int, clipped: Bool) {
        self.clipped = clipped
        _linear.wrappedValue = Linear(inputDims, outputDims, bias: false)
        _inputMin.wrappedValue = MLXArray(-Float.infinity)
        _inputMax.wrappedValue = MLXArray(Float.infinity)
        _outputMin.wrappedValue = MLXArray(-Float.infinity)
        _outputMax.wrappedValue = MLXArray(Float.infinity)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        if clipped {
            h = clip(h, min: inputMin.asType(h.dtype), max: inputMax.asType(h.dtype))
        }
        h = linear(h)
        if clipped {
            h = clip(h, min: outputMin.asType(h.dtype), max: outputMax.asType(h.dtype))
        }
        return h
    }
}

/// One subsampling block: Conv2d k=3 s=2 p=1 (no bias) → channel
/// LayerNorm (no bias) → ReLU. Input/output NHWC `[B, T, F, C]`.
private class Gemma4AudioSubSampleConvLayer: Module {
    @ModuleInfo(key: "conv") var conv: Conv2d
    @ModuleInfo(key: "norm") var norm: AudioLayerNorm

    init(inChannels: Int, outChannels: Int, eps: Float) {
        _conv.wrappedValue = Conv2d(
            inputChannels: inChannels, outputChannels: outChannels,
            kernelSize: 3, stride: 2, padding: 1, bias: false)
        _norm.wrappedValue = AudioLayerNorm(dimensions: outChannels, eps: eps)
        super.init()
    }

    /// `timeMask` is a float `[B, T]` validity mask multiplied in before
    /// the conv, mirroring HF's `hidden_states * mask[:, None, :, None]`.
    func callAsFunction(_ x: MLXArray, timeMask: MLXArray?) -> MLXArray {
        var h = x
        if let timeMask {
            h = h * timeMask.expandedDimensions(axes: [-1, -2]).asType(h.dtype)
        }
        h = conv(h.asType(conv.weight.dtype))
        return relu(norm(h))
    }
}

private class Gemma4AudioSubSampleConvProjection: Module {
    @ModuleInfo(key: "layer0") var layer0: Gemma4AudioSubSampleConvLayer
    @ModuleInfo(key: "layer1") var layer1: Gemma4AudioSubSampleConvLayer
    @ModuleInfo(key: "input_proj_linear") var inputProjLinear: Linear

    init(_ config: Gemma4AudioConfig, melBins: Int) {
        _layer0.wrappedValue = Gemma4AudioSubSampleConvLayer(
            inChannels: 1, outChannels: config.subsamplingConvChannels[0],
            eps: config.rmsNormEps)
        _layer1.wrappedValue = Gemma4AudioSubSampleConvLayer(
            inChannels: config.subsamplingConvChannels[0],
            outChannels: config.subsamplingConvChannels[1],
            eps: config.rmsNormEps)
        let projInputDim = (melBins / 4) * config.subsamplingConvChannels[1]
        _inputProjLinear.wrappedValue = Linear(projInputDim, config.hiddenSize, bias: false)
        super.init()
    }

    /// `features`: `[B, T, melBins]`; `timeMask`: float `[B, T]` or nil.
    /// Returns `[B, ceil(T/4), hidden]`.
    func callAsFunction(_ features: MLXArray, timeMask: MLXArray?) -> MLXArray {
        var h = features.expandedDimensions(axis: -1)  // NHWC [B, T, F, 1]
        h = layer0(h, timeMask: timeMask)
        let mask1 = timeMask.map { $0[0..., .stride(by: 2)] }
        h = layer1(h, timeMask: mask1)
        // NHWC [B, T4, F4, C] → [B, T4, F4*C]; identical element order to
        // HF's permute(0, 2, 3, 1).reshape(b, t, f*c) from NCHW.
        let (B, t) = (h.dim(0), h.dim(1))
        h = h.reshaped(B, t, -1)
        return inputProjLinear(h)
    }
}

/// HF Gemma4AudioFeedForward: clip → RMSNorm → 4× linear (clipped) →
/// SiLU → linear (clipped) → clip → RMSNorm → ×residual_weight + residual.
private class Gemma4AudioFeedForward: Module {
    @ModuleInfo(key: "ffw_layer_1") var ffw1: Gemma4ClippedLinear
    @ModuleInfo(key: "ffw_layer_2") var ffw2: Gemma4ClippedLinear
    @ModuleInfo(key: "pre_layer_norm") var preLayerNorm: AudioRMSNorm
    @ModuleInfo(key: "post_layer_norm") var postLayerNorm: AudioRMSNorm
    let residualWeight: Float
    let gradClip: Float

    init(_ config: Gemma4AudioConfig) {
        _ffw1.wrappedValue = Gemma4ClippedLinear(
            config.hiddenSize, config.hiddenSize * 4, clipped: config.useClippedLinears)
        _ffw2.wrappedValue = Gemma4ClippedLinear(
            config.hiddenSize * 4, config.hiddenSize, clipped: config.useClippedLinears)
        _preLayerNorm.wrappedValue = AudioRMSNorm(dimensions: config.hiddenSize)
        _postLayerNorm.wrappedValue = AudioRMSNorm(dimensions: config.hiddenSize)
        residualWeight = config.residualWeight
        gradClip = config.gradientClipping
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let residual = x
        var h = clip(x, min: MLXArray(-gradClip), max: MLXArray(gradClip))
        h = preLayerNorm(h)
        h = ffw1(h)
        h = silu(h)
        h = ffw2(h)
        h = clip(h, min: MLXArray(-gradClip), max: MLXArray(gradClip))
        h = postLayerNorm(h)
        return h * residualWeight + residual
    }
}

/// HF Gemma4AudioLightConv1d: RMSNorm → GLU(linear 2×) → causal
/// depthwise conv k=5 → clip → RMSNorm → SiLU → linear → + residual.
private class Gemma4AudioLightConv1d: Module {
    @ModuleInfo(key: "linear_start") var linearStart: Gemma4ClippedLinear
    @ModuleInfo(key: "linear_end") var linearEnd: Gemma4ClippedLinear
    @ModuleInfo(key: "depthwise_conv1d") var depthwiseConv: Conv1d
    @ModuleInfo(key: "pre_layer_norm") var preLayerNorm: AudioRMSNorm
    @ModuleInfo(key: "conv_norm") var convNorm: AudioRMSNorm
    let leftPad: Int
    let gradClip: Float

    init(_ config: Gemma4AudioConfig) {
        _linearStart.wrappedValue = Gemma4ClippedLinear(
            config.hiddenSize, config.hiddenSize * 2, clipped: config.useClippedLinears)
        _linearEnd.wrappedValue = Gemma4ClippedLinear(
            config.hiddenSize, config.hiddenSize, clipped: config.useClippedLinears)
        _depthwiseConv.wrappedValue = Conv1d(
            inputChannels: config.hiddenSize, outputChannels: config.hiddenSize,
            kernelSize: config.convKernelSize, groups: config.hiddenSize, bias: false)
        _preLayerNorm.wrappedValue = AudioRMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _convNorm.wrappedValue = AudioRMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        // Gemma4AudioCausalConv1d.left_pad = effective_kernel − stride.
        leftPad = config.convKernelSize - 1
        gradClip = config.gradientClipping
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let residual = x
        var h = preLayerNorm(x)
        h = linearStart(h)
        let half = h.dim(-1) / 2
        h = h[.ellipsis, ..<half] * sigmoid(h[.ellipsis, half...])  // GLU
        h = padded(h, widths: [[0, 0], [leftPad, 0], [0, 0]])
        h = depthwiseConv(h)
        h = clip(h, min: MLXArray(-gradClip), max: MLXArray(gradClip))
        h = convNorm(h)
        h = silu(h)
        h = linearEnd(h)
        return h + residual
    }
}

/// Chunked local attention with relative position bias (HF
/// Gemma4AudioAttention). Computation runs in float32 like the reference.
private class Gemma4AudioAttention: Module {
    @ModuleInfo(key: "q_proj") var qProj: Gemma4ClippedLinear
    @ModuleInfo(key: "k_proj") var kProj: Gemma4ClippedLinear
    @ModuleInfo(key: "v_proj") var vProj: Gemma4ClippedLinear
    @ModuleInfo(key: "post") var post: Gemma4ClippedLinear
    @ModuleInfo(key: "relative_k_proj") var relativeKProj: Linear
    @ParameterInfo(key: "per_dim_scale") var perDimScale: MLXArray

    let numHeads: Int
    let headDim: Int
    let chunkSize: Int
    let maxPast: Int
    let maxFuture: Int
    let contextSize: Int
    let qScale: Float
    let kScale: Float
    let softcap: Float
    let invalidLogit: Float

    init(_ config: Gemma4AudioConfig) {
        numHeads = config.numAttentionHeads
        headDim = config.hiddenSize / config.numAttentionHeads
        chunkSize = config.attentionChunkSize
        maxPast = config.attentionContextLeft - 1
        maxFuture = config.attentionContextRight
        contextSize = chunkSize + maxPast + maxFuture
        qScale = pow(Float(headDim), -0.5) / log(2.0)
        kScale = log(1.0 + exp(1.0)) / log(2.0)
        softcap = config.attentionLogitCap
        invalidLogit = config.attentionInvalidLogitsValue
        let h = config.hiddenSize
        _qProj.wrappedValue = Gemma4ClippedLinear(h, h, clipped: config.useClippedLinears)
        _kProj.wrappedValue = Gemma4ClippedLinear(h, h, clipped: config.useClippedLinears)
        _vProj.wrappedValue = Gemma4ClippedLinear(h, h, clipped: config.useClippedLinears)
        _post.wrappedValue = Gemma4ClippedLinear(h, h, clipped: config.useClippedLinears)
        _relativeKProj.wrappedValue = Linear(h, h, bias: false)
        _perDimScale.wrappedValue = MLXArray.zeros([headDim])
        super.init()
    }

    /// Overlapping context windows of `contextSize` per block, stride
    /// `chunkSize` (HF `_extract_block_context` unfold).
    private func extractBlockContext(_ x: MLXArray, numBlocks: Int) -> MLXArray {
        let p = padded(
            x, widths: [[0, 0], [maxPast, maxFuture + chunkSize - 1], [0, 0], [0, 0]])
        var idx = [Int32]()
        idx.reserveCapacity(numBlocks * contextSize)
        for b in 0 ..< numBlocks {
            for j in 0 ..< contextSize {
                idx.append(Int32(b * chunkSize + j))
            }
        }
        let indices = MLXArray(idx).reshaped(numBlocks, contextSize)
        return p.take(indices, axis: 1)  // [B, nb, ctx, H, D]
    }

    /// HF `_rel_shift` (Transformer-XL appendix B) for blocked attention.
    private func relShift(_ x: MLXArray) -> MLXArray {
        let (B, H, nb, bs, posLen) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3), x.dim(4))
        var y = padded(
            x, widths: [[0, 0], [0, 0], [0, 0], [0, 0], [0, contextSize + 1 - posLen]])
        y = y.reshaped(B, H, nb, bs * (contextSize + 1))
        y = y[.ellipsis, ..<(bs * contextSize)]
        return y.reshaped(B, H, nb, bs, contextSize)
    }

    /// `x`: `[B, T, hidden]`; `positionEmbeddings`: `[numPos, hidden]`
    /// float32; `blockedMask`: bool `[B, 1, nb, chunk, ctx]`.
    func callAsFunction(
        _ x: MLXArray, positionEmbeddings: MLXArray, blockedMask: MLXArray
    ) -> MLXArray {
        let (B, T) = (x.dim(0), x.dim(1))
        var q = qProj(x).asType(.float32).reshaped(B, T, numHeads, headDim)
        var k = kProj(x).asType(.float32).reshaped(B, T, numHeads, headDim)
        let v = vProj(x).asType(.float32).reshaped(B, T, numHeads, headDim)

        q = q * (softplus(perDimScale.asType(.float32)) * qScale)
        k = k * kScale

        let nb = (T + chunkSize - 1) / chunkSize
        let padQ = nb * chunkSize - T
        var qb = q
        if padQ > 0 {
            qb = padded(qb, widths: [[0, 0], [0, padQ], [0, 0], [0, 0]])
        }
        qb = qb.reshaped(B, nb, chunkSize, numHeads, headDim)
        let kb = extractBlockContext(k, numBlocks: nb)
        let vb = extractBlockContext(v, numBlocks: nb)

        var relK = relativeKProj(positionEmbeddings.asType(relativeKProj.weight.dtype))
        relK = relK.asType(.float32).reshaped(-1, numHeads, headDim)  // [P, H, D]

        let queries = qb.transposed(0, 3, 1, 2, 4)  // [B, H, nb, chunk, D]
        let matrixAC = matmul(queries, kb.transposed(0, 3, 1, 4, 2))  // [B,H,nb,chunk,ctx]

        let qFlat = queries.reshaped(B, numHeads, nb * chunkSize, headDim)
        var matrixBD = matmul(qFlat, relK.transposed(1, 2, 0))  // [B, H, Q, P]
        matrixBD = matrixBD.reshaped(B, numHeads, nb, chunkSize, -1)
        matrixBD = relShift(matrixBD)

        var attn = matrixAC + matrixBD
        attn = tanh(attn / softcap) * softcap
        attn = MLX.where(blockedMask, attn, MLXArray(invalidLogit))
        attn = softmax(attn, axis: -1, precise: true)

        var out = matmul(attn, vb.transposed(0, 3, 1, 2, 4))  // [B, H, nb, chunk, D]
        out = out.transposed(0, 2, 3, 1, 4).reshaped(B, nb * chunkSize, numHeads * headDim)
        out = out[0..., ..<T]
        return post(out.asType(post.linear.weight.dtype))
    }
}

private class Gemma4AudioLayer: Module {
    @ModuleInfo(key: "feed_forward1") var feedForward1: Gemma4AudioFeedForward
    @ModuleInfo(key: "feed_forward2") var feedForward2: Gemma4AudioFeedForward
    @ModuleInfo(key: "self_attn") var selfAttn: Gemma4AudioAttention
    @ModuleInfo(key: "lconv1d") var lconv1d: Gemma4AudioLightConv1d
    @ModuleInfo(key: "norm_pre_attn") var normPreAttn: AudioRMSNorm
    @ModuleInfo(key: "norm_post_attn") var normPostAttn: AudioRMSNorm
    @ModuleInfo(key: "norm_out") var normOut: AudioRMSNorm
    let gradClip: Float

    init(_ config: Gemma4AudioConfig) {
        _feedForward1.wrappedValue = Gemma4AudioFeedForward(config)
        _feedForward2.wrappedValue = Gemma4AudioFeedForward(config)
        _selfAttn.wrappedValue = Gemma4AudioAttention(config)
        _lconv1d.wrappedValue = Gemma4AudioLightConv1d(config)
        _normPreAttn.wrappedValue = AudioRMSNorm(dimensions: config.hiddenSize)
        _normPostAttn.wrappedValue = AudioRMSNorm(dimensions: config.hiddenSize)
        _normOut.wrappedValue = AudioRMSNorm(dimensions: config.hiddenSize)
        gradClip = config.gradientClipping
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, positionEmbeddings: MLXArray, blockedMask: MLXArray
    ) -> MLXArray {
        var h = feedForward1(x)
        let residual = h
        h = clip(h, min: MLXArray(-gradClip), max: MLXArray(gradClip))
        h = normPreAttn(h)
        h = selfAttn(h, positionEmbeddings: positionEmbeddings, blockedMask: blockedMask)
        h = clip(h, min: MLXArray(-gradClip), max: MLXArray(gradClip))
        h = normPostAttn(h)
        h = h + residual
        h = lconv1d(h)
        h = feedForward2(h)
        h = clip(h, min: MLXArray(-gradClip), max: MLXArray(gradClip))
        return normOut(h)
    }
}

// MARK: - Tower

/// Gemma 4 E-series conformer audio encoder (`audio_tower.*` weights).
/// Tensor keys mirror the checkpoint weight map exactly, e.g.
/// `audio_tower.layers.0.feed_forward1.ffw_layer_1.linear.weight`.
class Gemma4AudioTower: Module {
    @ModuleInfo(key: "subsample_conv_projection")
    fileprivate var subsampleConvProjection: Gemma4AudioSubSampleConvProjection
    @ModuleInfo(key: "layers") fileprivate var layers: [Gemma4AudioLayer]
    @ModuleInfo(key: "output_proj") var outputProj: Linear

    let config: Gemma4AudioConfig

    init(_ config: Gemma4AudioConfig, melBins: Int = Gemma4AudioMel.melBins) {
        self.config = config
        _subsampleConvProjection.wrappedValue = Gemma4AudioSubSampleConvProjection(
            config, melBins: melBins)
        _layers.wrappedValue = (0 ..< config.numHiddenLayers).map { _ in
            Gemma4AudioLayer(config)
        }
        _outputProj.wrappedValue = Linear(config.hiddenSize, config.outputProjDims, bias: true)
        super.init()
    }

    /// Sinusoidal relative positional table, `[contextSize/2 + 1, hidden]`,
    /// positions contextSize/2 … 0, layout `[sin…, cos…]`
    /// (HF Gemma4AudioRelPositionalEncoding).
    private func relPositionEmbeddings() -> MLXArray {
        let hidden = config.hiddenSize
        let contextSize =
            config.attentionChunkSize + config.attentionContextLeft - 1
            + config.attentionContextRight
        let numTimescales = hidden / 2
        let logIncrement = log(10000.0) / Double(max(numTimescales - 1, 1))
        let numPos = contextSize / 2 + 1
        var table = [Float](repeating: 0, count: numPos * hidden)
        for (row, position) in stride(from: contextSize / 2, through: 0, by: -1).enumerated() {
            for i in 0 ..< numTimescales {
                let invTimescale = exp(Double(i) * -logIncrement)
                let t = Double(position) * invTimescale
                table[row * hidden + i] = Float(sin(t))
                table[row * hidden + numTimescales + i] = Float(cos(t))
            }
        }
        return MLXArray(table).reshaped(numPos, hidden)
    }

    /// Blocked attention mask, bool `[B, 1, nb, chunk, ctx]`:
    /// kv must be a real, valid frame AND inside the sliding window
    /// 0 ≤ q − kv < attention_context_left − 1 (HF
    /// `create_bidirectional_mask` + `sliding_window_mask_function`
    /// + `_convert_4d_mask_to_blocked_5d`).
    private func blockedMask(
        seqLength: Int, numBlocks: Int, validLengths: [Int]
    ) -> MLXArray {
        let chunk = config.attentionChunkSize
        let leftWindow = config.attentionContextLeft - 1
        let rightWindow = config.attentionContextRight
        let ctx = chunk + leftWindow + rightWindow
        let B = validLengths.count
        var bools = [Bool](repeating: false, count: B * numBlocks * chunk * ctx)
        var offset = 0
        for b in 0 ..< B {
            let valid = min(validLengths[b], seqLength)
            for block in 0 ..< numBlocks {
                let kvStart = block * chunk - leftWindow
                for i in 0 ..< chunk {
                    let qPos = block * chunk + i
                    for j in 0 ..< ctx {
                        let kvPos = kvStart + j
                        if qPos < seqLength, kvPos >= 0, kvPos < valid {
                            let dist = qPos - kvPos
                            let inLeft = dist >= 0 && dist < leftWindow
                            let inRight = dist < 0 && -dist < rightWindow
                            bools[offset] = inLeft || inRight
                        }
                        offset += 1
                    }
                }
            }
        }
        return MLXArray(bools).reshaped(B, 1, numBlocks, chunk, ctx)
    }

    /// Encode mel features.
    ///
    /// - Parameters:
    ///   - melFeatures: `[B, T, melBins]` log-mel frames; padded rows
    ///     (beyond each item's valid length) must be exactly zero, the
    ///     same contract as the HF extractor's mask-zeroed output.
    ///   - validFrameCounts: per-item count of valid (prefix) mel frames;
    ///     nil ⇒ all frames valid.
    /// - Returns: `[B, ceil(T/4), output_proj_dims]`; item `b` has
    ///   `gemma4AudioSoftTokenCount(melFrameCount: validFrameCounts[b])`
    ///   valid output tokens.
    func callAsFunction(_ melFeatures: MLXArray, validFrameCounts: [Int]? = nil) -> MLXArray {
        let (B, T) = (melFeatures.dim(0), melFeatures.dim(1))
        let validLengths = validFrameCounts ?? Array(repeating: T, count: B)
        precondition(validLengths.count == B, "validFrameCounts must have one entry per item")

        // Float time mask for the subsampling convs (HF zeroes invalid
        // frames between conv layers). Skip when everything is valid.
        var timeMask: MLXArray? = nil
        if validLengths.contains(where: { $0 < T }) {
            var maskValues = [Float](repeating: 0, count: B * T)
            for b in 0 ..< B {
                for t in 0 ..< min(validLengths[b], T) { maskValues[b * T + t] = 1 }
            }
            timeMask = MLXArray(maskValues).reshaped(B, T)
        }

        var h = subsampleConvProjection(melFeatures, timeMask: timeMask)
        let t4 = h.dim(1)
        let nb = (t4 + config.attentionChunkSize - 1) / config.attentionChunkSize
        // Two stride-2 convs: valid length subsamples as ceil(v/4)
        // (= indices ≡ 0 mod 4), matching replace_audio_token.
        let validSubsampled = validLengths.map { gemma4AudioSoftTokenCount(melFrameCount: $0) }
        let mask = blockedMask(seqLength: t4, numBlocks: nb, validLengths: validSubsampled)
        let positionEmbeddings = relPositionEmbeddings()

        var intermediates: [String: MLXArray] = [:]
        let dumpDir = ProcessInfo.processInfo.environment["VMLX_GEMMA4_AUDIO_DUMP_DIR"]
        if dumpDir != nil { intermediates["subsample"] = h.asType(.float32) }
        for (i, layer) in layers.enumerated() {
            h = layer(h, positionEmbeddings: positionEmbeddings, blockedMask: mask)
            if dumpDir != nil { intermediates["layer\(i)"] = h.asType(.float32) }
        }
        if let dumpDir {
            let dir = URL(fileURLWithPath: dumpDir)
            try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            try? MLX.save(
                arrays: intermediates,
                url: dir.appendingPathComponent("gemma4-audio-tower-intermediates.safetensors"))
        }
        return outputProj(h)
    }
}
