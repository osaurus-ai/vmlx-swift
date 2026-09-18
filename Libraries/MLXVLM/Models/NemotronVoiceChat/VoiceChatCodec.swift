// NemotronLabs VoiceChat — the 22.05 kHz neural audio codec.
//
// Port of `mlx_audio/codec/models/nemotron_voicechat/codec.py`. ConvNeXt-1d
// encoder/decoder around a 31-stage probabilistic residual VQ, with an
// iSTFT vocoder head (n_fft 16, hop 4 — a very short window, so the overlap-add
// is cheap even at 22 kHz).
//
// Streaming discipline (this is what makes live speech-to-speech possible):
//   * every depthwise conv is CAUSAL (left pad K-1) and carries its K-1 tail
//     across chunks via `VoiceChatCausalConv1dCache`;
//   * the iSTFT carries 4 spectrogram frames on each side so consecutive
//     decoded chunks are sample-continuous instead of clicking at every
//     boundary — the periodic artifact no whole-file test can see.

import Foundation
import MLX
import MLXFFT
import MLXNN

/// Per-layer causal-conv and spectrogram overlap state for one stream.
public final class VoiceChatCausalConv1dCache {
    private var cache: [String: MLXArray] = [:]

    public init() {}

    /// Prepend this layer's carried tail to `states` and remember the new tail.
    /// `flush` drops the entry after use (end of utterance).
    public func update(
        _ states: MLXArray, layer: String, padding: Int, paddingValue: Float = 0.0,
        flush: Bool = false
    ) -> MLXArray {
        precondition(padding >= 0, "padding must be non-negative")
        if padding == 0 { return states }

        let previous: MLXArray
        if let existing = cache[layer], existing.dim(0) == states.dim(0),
            existing.dim(1) == padding, existing.dim(2) == states.dim(2)
        {
            previous = existing.asType(states.dtype)
        } else {
            previous = MLXArray.full(
                [states.dim(0), padding, states.dim(2)],
                values: MLXArray(paddingValue).asType(states.dtype))
        }

        let padded = MLX.concatenated([previous, states], axis: 1)
        cache[layer] = padded[0..., (padded.dim(1) - padding)..., 0...]
        if flush { cache.removeValue(forKey: layer) }
        return padded
    }

    public func clear() { cache.removeAll() }
}

/// LayerNorm over the channel axis of `(B, T, C)`, with weight AND bias
/// (the codec's blocks ship both; the conformer's `batch_norm` also does).
public class VoiceChatChannelLayerNorm: Module, UnaryLayer {
    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "bias") var bias: MLXArray
    let eps: Float

    public init(channels: Int, eps: Float = 1e-6) {
        self._weight.wrappedValue = MLXArray.ones([channels])
        self._bias.wrappedValue = MLXArray.zeros([channels])
        self.eps = eps
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let mean = MLX.mean(x, axis: -1, keepDims: true)
        let variance = MLX.variance(x, axis: -1, keepDims: true)
        return (x - mean) * MLX.rsqrt(variance + eps) * weight + bias
    }
}

/// ConvNeXt block: causal depthwise → channel LayerNorm → 4× pointwise MLP.
public class VoiceChatConvNeXtBlock1d: Module {
    public let kernelSize: Int
    public let layerId: Int

    @ModuleInfo(key: "dwconv") var dwconv: Conv1d
    @ModuleInfo(key: "norm") var norm: VoiceChatChannelLayerNorm
    @ModuleInfo(key: "pwconv1") var pwconv1: Conv1d
    @ModuleInfo(key: "pwconv2") var pwconv2: Conv1d

    public init(channels: Int, kernelSize: Int, layerId: Int) {
        self.kernelSize = kernelSize
        self.layerId = layerId
        self._dwconv.wrappedValue = Conv1d(
            inputChannels: channels, outputChannels: channels, kernelSize: kernelSize,
            groups: channels, bias: true)
        self._norm.wrappedValue = VoiceChatChannelLayerNorm(channels: channels)
        self._pwconv1.wrappedValue = Conv1d(
            inputChannels: channels, outputChannels: 4 * channels, kernelSize: 1, bias: true)
        self._pwconv2.wrappedValue = Conv1d(
            inputChannels: 4 * channels, outputChannels: channels, kernelSize: 1, bias: true)
    }

    public func callAsFunction(
        _ x: MLXArray, cache: VoiceChatCausalConv1dCache? = nil, flush: Bool = false
    ) -> MLXArray {
        let residual = x
        var h: MLXArray
        if let cache {
            h = cache.update(x, layer: "conv\(layerId)", padding: kernelSize - 1, flush: flush)
        } else {
            h = MLX.padded(
                x, widths: [.init((0, 0)), .init((kernelSize - 1, 0)), .init((0, 0))])
        }
        h = dwconv(h)
        h = norm(h)
        h = pwconv1(h)
        h = gelu(h)
        h = pwconv2(h)
        return residual + h
    }
}

/// Encoder: 1×1 in, then per stage {blocks…, strided downsample}.
public class VoiceChatCodecEncoder: Module {
    @ModuleInfo(key: "layers") var layers: [Module]

    public init(_ config: VoiceChatCodecConfiguration) {
        let channels = config.channelMultipliers.map { config.baseChannels * $0 }
        precondition(
            channels.count == config.downsampleRates.count,
            "channel_multipliers and downsample_rates must have equal lengths")

        var built: [Module] = [
            Conv1d(
                inputChannels: config.stftChannels, outputChannels: channels[0],
                kernelSize: 1, bias: false)
        ]
        var blockIndex = 0
        for (index, stageChannels) in channels.enumerated() {
            for _ in 0 ..< config.blocksPerStage {
                built.append(
                    VoiceChatConvNeXtBlock1d(
                        channels: stageChannels, kernelSize: config.blockKernelSize,
                        layerId: blockIndex))
                blockIndex += 1
            }
            let next = index + 1 < channels.count ? channels[index + 1] : config.latentDim
            let rate = config.downsampleRates[index]
            built.append(
                Conv1d(
                    inputChannels: stageChannels, outputChannels: next, kernelSize: rate,
                    stride: rate, bias: false))
        }
        self._layers.wrappedValue = built
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for layer in layers {
            if let block = layer as? VoiceChatConvNeXtBlock1d {
                h = block(h)
            } else {
                h = (layer as! UnaryLayer)(h)
            }
        }
        return h
    }
}

/// Decoder: mirrored — per stage {transposed upsample, blocks…}, then 1×1 out.
public class VoiceChatCodecDecoder: Module {
    @ModuleInfo(key: "layers") var layers: [Module]

    public init(_ config: VoiceChatCodecConfiguration) {
        let channels = config.channelMultipliers.map { config.baseChannels * $0 }
        var built: [Module] = []
        var sourceChannels = config.latentDim
        var blockIndex = 0
        for (stageChannels, rate) in zip(channels.reversed(), config.downsampleRates.reversed()) {
            built.append(
                ConvTransposed1d(
                    inputChannels: sourceChannels, outputChannels: stageChannels,
                    kernelSize: rate, stride: rate, bias: false))
            for _ in 0 ..< config.blocksPerStage {
                built.append(
                    VoiceChatConvNeXtBlock1d(
                        channels: stageChannels, kernelSize: config.blockKernelSize,
                        layerId: blockIndex))
                blockIndex += 1
            }
            sourceChannels = stageChannels
        }
        built.append(
            Conv1d(
                inputChannels: channels[0], outputChannels: config.stftChannels,
                kernelSize: 1, bias: false))
        self._layers.wrappedValue = built
    }

    public func callAsFunction(
        _ x: MLXArray, cache: VoiceChatCausalConv1dCache? = nil, flush: Bool = false
    ) -> MLXArray {
        var h = x
        for layer in layers {
            if let block = layer as? VoiceChatConvNeXtBlock1d {
                h = block(h, cache: cache, flush: flush)
            } else {
                h = (layer as! UnaryLayer)(h)
            }
        }
        return h
    }
}

/// Scalar per-quantizer variance (training-path parameter; kept so a strict
/// load consumes every checkpoint tensor rather than silently ignoring some).
public class VoiceChatScalarVariance: Module {
    @ParameterInfo(key: "variance") var variance: MLXArray
    public override init() {
        self._variance.wrappedValue = MLXArray(Float(1.0))
    }
}

/// 31-stage probabilistic residual vector quantizer.
public class VoiceChatPRVQ: Module {
    @ParameterInfo(key: "mus_list") var musList: [MLXArray]
    @ModuleInfo(key: "variance_list") var varianceList: [VoiceChatScalarVariance]

    public init(_ config: VoiceChatCodecConfiguration) {
        self._musList.wrappedValue = (0 ..< config.numQuantizers).map { _ in
            MLXArray.zeros([config.codebookSize, config.latentDim])
        }
        self._varianceList.wrappedValue = (0 ..< config.numQuantizers).map { _ in
            VoiceChatScalarVariance()
        }
    }

    /// (B, T, D) latents → (B, Q, T) code ids.
    public func encode(_ latents: MLXArray) -> MLXArray {
        var residual = latents
        var codes = [MLXArray]()
        for means in musList {
            let distances =
                MLX.sum(residual * residual, axis: -1, keepDims: true)
                - 2.0 * MLX.matmul(residual, means.T)
                + MLX.sum(means * means, axis: -1).reshaped([1, 1, -1])
            let indices = MLX.argMin(distances, axis: -1)
            codes.append(indices)
            residual = residual - means[indices]
        }
        return MLX.stacked(codes, axis: 1)
    }

    /// (B, Q, T) code ids → (B, T, D) latents.
    public func decode(_ codes: MLXArray) -> MLXArray {
        precondition(codes.ndim == 3, "codes must have shape (B, Q, T)")
        precondition(
            codes.dim(1) <= musList.count,
            "received \(codes.dim(1)) quantizers, maximum is \(musList.count)")
        var latents = MLXArray.zeros(
            [codes.dim(0), codes.dim(2), musList[0].dim(-1)], dtype: musList[0].dtype)
        for index in 0 ..< codes.dim(1) {
            latents = latents + musList[index][codes[0..., index, 0...]]
        }
        return latents
    }
}

/// Periodic Hann window (matches `mlx_audio.dsp.hanning(periodic: true)`).
func voiceChatHann(_ n: Int) -> MLXArray {
    let values = (0 ..< n).map { i in
        Float(0.5 - 0.5 * Foundation.cos(2.0 * Double.pi * Double(i) / Double(n)))
    }
    return MLXArray(values)
}

/// Overlap-add iSTFT with window-sum normalisation, `center: false`.
func voiceChatISTFT(
    real: MLXArray, imag: MLXArray, nFFT: Int, hopLength: Int, window: MLXArray
) -> MLXArray {
    // real/imag: (B, bins, frames)
    let B = real.dim(0), frames = real.dim(2)
    let spectrum = real.transposed(0, 2, 1).asType(.float32)
        + MLXArray(0, dtype: .float32).asType(.float32) * 0  // shape anchor
    _ = spectrum
    let complexSpec = real.transposed(0, 2, 1).asType(.complex64)
        + imag.transposed(0, 2, 1).asType(.complex64) * MLXArray(real: 0, imaginary: 1)
    // (B, frames, nFFT) time-domain frames
    let framesTime = MLXFFT.irfft(complexSpec, n: nFFT, axis: -1) * window.reshaped([1, 1, nFFT])

    let length = (frames - 1) * hopLength + nFFT
    var out = MLXArray.zeros([B, length], dtype: .float32)
    var windowSum = MLXArray.zeros([length], dtype: .float32)
    let windowSquared = window * window
    for f in 0 ..< frames {
        let start = f * hopLength
        out[0..., start ..< (start + nFFT)] =
            out[0..., start ..< (start + nFFT)] + framesTime[0..., f, 0...]
        windowSum[start ..< (start + nFFT)] =
            windowSum[start ..< (start + nFFT)] + windowSquared
    }
    return out / MLX.maximum(windowSum, MLXArray(Float(1e-8))).reshaped([1, length])
}

/// The embedded 22.05 kHz codec.
public class VoiceChatCodec: Module {
    public let config: VoiceChatCodecConfiguration

    @ModuleInfo(key: "encoder") public var encoder: VoiceChatCodecEncoder
    @ModuleInfo(key: "decoder") public var decoder: VoiceChatCodecDecoder
    @ModuleInfo(key: "prvq") public var prvq: VoiceChatPRVQ

    public init(_ config: VoiceChatCodecConfiguration) {
        self.config = config
        self._encoder.wrappedValue = VoiceChatCodecEncoder(config)
        self._decoder.wrappedValue = VoiceChatCodecDecoder(config)
        self._prvq.wrappedValue = VoiceChatPRVQ(config)
    }

    /// (B, samples) or (B, 1, samples) waveform → (B, T, D) latents.
    ///
    /// Needed for BOTH the silent speaker-prompt lead-in and, later, cloning a
    /// custom voice: making a voice means audio → latent, so the ENCODER ships
    /// and is used, not just the decoder.
    public func encodeLatents(_ waveform: MLXArray) -> MLXArray {
        var wave = waveform
        if wave.ndim == 3 {
            precondition(wave.dim(1) == 1, "only mono waveforms are supported")
            wave = wave[0..., 0, 0...]
        }
        precondition(wave.ndim == 2, "waveform must be (batch, samples)")
        precondition(wave.dim(-1) > 0, "waveform must contain at least one sample")

        // STFT with the same centring convention as the decoder's iSTFT.
        let nFFT = config.nFFT
        let hop = config.hopLength
        let padLeft = (nFFT - hop) / 2
        let padRight = nFFT - hop - padLeft
        var padded = MLX.padded(
            wave.asType(.float32), widths: [.init((0, 0)), .init((padLeft, padRight))])
        if padded.dim(-1) < nFFT {
            padded = MLX.padded(
                padded, widths: [.init((0, 0)), .init((0, nFFT - padded.dim(-1)))])
        }
        let window = voiceChatHann(nFFT)
        let frames = (padded.dim(-1) - nFFT) / hop + 1
        var slices = [MLXArray]()
        slices.reserveCapacity(frames)
        for f in 0 ..< frames {
            let start = f * hop
            slices.append(padded[0..., start ..< (start + nFFT)] * window.reshaped([1, nFFT]))
        }
        let framed = MLX.stacked(slices, axis: 1)  // (B, frames, nFFT)
        let spectrum = MLXFFT.rfft(framed, axis: -1)  // (B, frames, bins)
        let features = MLX.concatenated(
            [spectrum.realPart(), spectrum.imaginaryPart()], axis: -1)
        return encoder(features)
    }

    /// (B, samples) waveform → (B, Q, T) code ids.
    public func encode(_ waveform: MLXArray) -> MLXArray {
        prvq.encode(encodeLatents(waveform))
    }

    /// (B, Q, T) codes → (B, 1, samples) waveform at 22.05 kHz.
    public func decode(
        _ codes: MLXArray, cache: VoiceChatCausalConv1dCache? = nil, flush: Bool = false
    ) -> MLXArray {
        decodeLatents(prvq.decode(codes), cache: cache, flush: flush)
    }

    public func decodeLatents(
        _ latents: MLXArray, cache: VoiceChatCausalConv1dCache? = nil, flush: Bool = false
    ) -> MLXArray {
        let features = decoder(latents, cache: cache, flush: flush).transposed(0, 2, 1)
        let numBins = config.nFFT / 2 + 1
        let magnitudeLogits = features[0..., 0 ..< numBins, 0...]
        let phase = features[0..., numBins..., 0...]

        // Softplus-parameterised magnitude, capped at 100 (matches reference).
        let maxMagnitude: Float = 100.0
        let magnitude =
            maxMagnitude
            * MLX.exp(-softplus(-magnitudeLogits + MLXArray(Foundation.log(maxMagnitude))))
        var real = magnitude * MLX.cos(phase)
        var imag = magnitude * MLX.sin(phase)
        // DC and Nyquist bins of an rFFT are real-valued.
        imag = MLX.concatenated(
            [
                MLXArray.zeros(like: imag[0..., 0 ..< 1, 0...]),
                imag[0..., 1 ..< (imag.dim(1) - 1), 0...],
                MLXArray.zeros(like: imag[0..., (imag.dim(1) - 1)..., 0...]),
            ], axis: 1)

        // Streaming continuity: carry 4 spectrogram frames on each side so a
        // decoded chunk joins the previous one without a boundary click.
        let halfSpecPadding = Int(
            ceil(Double((config.nFFT - config.hopLength) / 2) / Double(config.hopLength)))
        if let cache {
            let specPadding = halfSpecPadding * 2
            real = cache.update(
                real.transposed(0, 2, 1), layer: "istft_real", padding: specPadding,
                flush: flush
            ).transposed(0, 2, 1)
            imag = cache.update(
                imag.transposed(0, 2, 1), layer: "istft_imag", padding: specPadding,
                flush: flush
            ).transposed(0, 2, 1)
            if flush {
                let tail = MLXArray.zeros(
                    [real.dim(0), real.dim(1), halfSpecPadding], dtype: real.dtype)
                real = MLX.concatenated([real, tail], axis: 2)
                imag = MLX.concatenated([imag, tail], axis: 2)
            }
        }

        let window = voiceChatHann(config.nFFT)
        var waveform = voiceChatISTFT(
            real: real, imag: imag, nFFT: config.nFFT, hopLength: config.hopLength,
            window: window)
        let padLeft = (config.nFFT - config.hopLength) / 2
        let padRight = config.nFFT - config.hopLength - padLeft
        waveform = waveform[0..., padLeft ..< (waveform.dim(-1) - padRight)]
        if cache != nil {
            let halfWavePadding = halfSpecPadding * config.hopLength
            waveform = waveform[0..., halfWavePadding ..< (waveform.dim(-1) - halfWavePadding)]
        }
        return waveform.expandedDimensions(axis: 1)
    }

    /// Map the bundle's codec tensors onto this tree. Conv weights arrive in
    /// PyTorch layout; decoder stage-start ConvTranspose1d needs a DIFFERENT
    /// transpose from every other 3-D weight, which is the trap this mirrors
    /// from the reference.
    public static func sanitized(
        _ weights: [String: MLXArray], config: VoiceChatCodecConfiguration
    ) -> [String: MLXArray] {
        var converted = [String: MLXArray]()
        let stageWidth = config.blocksPerStage + 1
        let stageRegion = stageWidth * config.downsampleRates.count

        for (rawKey, rawValue) in weights {
            let key = rawKey.replacingOccurrences(
                of: "prvq._variance_list.", with: "prvq.variance_list.")
            var value = rawValue
            let isWeight = key.hasSuffix(".weight") && value.ndim == 3
            if isWeight && key.hasPrefix("decoder.layers.") {
                let parts = key.components(separatedBy: ".")
                let layerIndex = Int(parts[2]) ?? -1
                if layerIndex < stageRegion, layerIndex % stageWidth == 0,
                    !key.contains("dwconv")
                {
                    value = value.transposed(1, 2, 0)  // ConvTranspose1d
                } else {
                    value = value.transposed(0, 2, 1)
                }
            } else if isWeight {
                value = value.transposed(0, 2, 1)
            }
            // `mus_list.N` / `variance_list.N` land as-is on the parameter arrays.
            converted[key] = value
        }
        return converted
    }
}
