// Apertus 1.5 audio tokenizer — the ENCODER half of WavTokenizer.
//
// Same design as the vision side (see `Apertus1p5VisionTokenizer.swift`): audio is quantised into
// ids from a 4096-entry codebook and fed as ORDINARY TOKENS at `audio_token_offset` 262344. The
// arithmetic closes — 262344 + 4096 = 266440, inside the 266,752 text vocabulary — so the language
// model is again untouched.
//
// WHAT IS DELIBERATELY NOT PORTED. The checkpoint carries 226 audio tensors; only ~63 are used
// here. `backbone.convnext` (108), `backbone.pos_net` (44) and `head.linear` [2402, 768] are the
// DECODER — that head is an ISTFT magnitude/phase output (2402 = 1201 x 2), i.e. the vocoder that
// turns tokens back into a waveform. A model consuming audio never runs it, exactly as the vision
// side ships no decoder. `quantizer.codebook.{cluster_size,embed_avg,inited}` are EMA training
// buffers, not inference state.
//
// The encoder is EnCodec/SEANet, read off the weights rather than assumed:
//
//     layer  0  conv 1 -> 32,   k=7            (input)
//     layer  1  residual block  32 -> 16 -> 32 (+ 1x1 shortcut)
//     layer  3  downsample      32 -> 64,  k=8   stride 4
//     layer  4  residual block  64
//     layer  6  downsample      64 -> 128, k=10  stride 5
//     layer  7  residual block  128
//     layer  9  downsample     128 -> 256, k=10  stride 5
//     layer 10  residual block  256
//     layer 12  downsample     256 -> 512, k=12  stride 6
//     layer 13  2-layer LSTM   (2048 = 4 gates x 512)
//     layer 15  conv 512 -> 512, k=7           (output)
//
// Even-numbered gaps are parameterless ELU activations. Total downsampling 4*5*5*6 = 600, which is
// WavTokenizer's documented hop — so 24 kHz audio yields 40 tokens per second.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Configuration

public struct Apertus1p5AudioTokenizerConfiguration: Codable, Sendable {
    public let codebookSize: Int
    public let codebookDim: Int
    private let _sampleRate: Int?
    /// WavTokenizer operates at 24 kHz; the bundle does not always spell it out.
    public var sampleRate: Int { _sampleRate ?? 24000 }

    enum CodingKeys: String, CodingKey {
        case codebookSize = "codebook_size"
        case codebookDim = "codebook_dim"
        case _sampleRate = "sample_rate"
    }
}

// MARK: - Building blocks

/// SEANet residual unit: ELU -> conv(k=3) -> ELU -> conv(k=1), plus a 1x1 shortcut.
///
/// The checkpoint numbers these as `block.1` and `block.3` because `block.0` and `block.2` are the
/// parameterless activations — so the indices are not contiguous and must not be renumbered.
private class AudioResidualUnit: Module {
    @ModuleInfo(key: "block") var block: [Module]
    @ModuleInfo(key: "shortcut") var shortcut: WeightNormConv1d

    init(channels: Int, compress: Int, kernelSize: Int) {
        let hidden = channels / compress
        // Slots 0 and 2 are the parameterless ELUs. They are NUMBERED in the checkpoint
        // (`block.1`, `block.3`), so renumbering the convs to 0/1 would leave every weight
        // unbound — placeholders keep the indices honest.
        self._block.wrappedValue = [
            Identity(),
            WeightNormConv1d(inC: channels, outC: hidden, k: kernelSize, stride: 1),
            Identity(),
            WeightNormConv1d(inC: hidden, outC: channels, k: 1, stride: 1),
        ]
        self._shortcut.wrappedValue = WeightNormConv1d(
            inC: channels, outC: channels, k: 1, stride: 1)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        guard block.count == 4,
            let c1 = block[1] as? WeightNormConv1d,
            let c2 = block[3] as? WeightNormConv1d
        else { return x }
        var h = c1(elu(x))
        h = c2(elu(h))
        return shortcut(x) + h
    }
}

/// A 1-D convolution whose checkpoint weight is stored under PyTorch weight-norm.
///
/// The weight lives as two tensors — `parametrizations.weight.original0` (g, per-output-channel
/// magnitude) and `original1` (v, direction) — and the effective weight is `g * v / ||v||`. Loading
/// `original1` alone runs perfectly well and is wrong by a per-channel scale, which is precisely the
/// kind of silent error that produces plausible-but-meaningless codebook ids. The reconstruction is
/// done once in `sanitize`, so at inference this is an ordinary Conv1d.
private class WeightNormConv1d: Module, UnaryLayer {
    @ModuleInfo(key: "conv") var conv: Conv1d
    let padding: Int

    init(inC: Int, outC: Int, k: Int, stride: Int) {
        // SEANet pads causally-ish: total padding k - stride, applied on the left. Doing it here
        // rather than via Conv1d's symmetric `padding` keeps the output length exact.
        self.padding = max(0, k - stride)
        self._conv.wrappedValue = Conv1d(
            inputChannels: inC, outputChannels: outC, kernelSize: k, stride: stride, padding: 0)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // NLC layout: pad the LENGTH axis on the left.
        let padded = padded(x, widths: [.init((0, 0)), .init((padding, 0)), .init((0, 0))])
        return conv(padded)
    }
}

/// SEANet's recurrent block: a 2-layer LSTM whose output is added back to its input.
///
/// The checkpoint stores it as torch does — `lstm.weight_ih_l0/l1`, `weight_hh_l0/l1`,
/// `bias_ih_l0/l1`, `bias_hh_l0/l1` — while MLX's `LSTM` uses `Wx`/`Wh`/`bias` per layer. The
/// remapping happens in `sanitize`; the gate order needs no adjustment, because MLX splits
/// i,f,g,o and so does torch.
private class SEANetLSTM: Module {
    @ModuleInfo(key: "lstm") var lstm: [LSTM]

    init(size: Int) {
        self._lstm.wrappedValue = [
            LSTM(inputSize: size, hiddenSize: size), LSTM(inputSize: size, hiddenSize: size),
        ]
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for layer in lstm { h = layer(h).0 }
        return h + x        // residual, as in SEANet
    }
}

// MARK: - Encoder

private class AudioEncoder: Module {
    @ModuleInfo(key: "layers") var layers: [Module]

    /// Indices match the checkpoint exactly, gaps and all — see the header table.
    init(hiddenSize: Int, compress: Int, kernelSize: Int) {
        var built: [Module] = []
        let ratios = [(4, 8), (5, 10), (5, 10), (6, 12)]   // (stride, kernel)
        var channels = 32
        built.append(WeightNormConv1d(inC: 1, outC: channels, k: kernelSize, stride: 1))  // 0
        for (stride, k) in ratios {
            built.append(AudioResidualUnit(channels: channels, compress: compress, kernelSize: 3))
            built.append(Identity())                                                       // ELU
            built.append(
                WeightNormConv1d(inC: channels, outC: channels * 2, k: k, stride: stride))
            channels *= 2
        }
        built.append(SEANetLSTM(size: hiddenSize))                                         // 13
        built.append(Identity())                                                           // 14
        built.append(
            WeightNormConv1d(inC: hiddenSize, outC: hiddenSize, k: kernelSize, stride: 1))  // 15
        self._layers.wrappedValue = built
    }
}

// MARK: - Tokenizer

public class Apertus1p5AudioTokenizer: Module {
    @ModuleInfo(key: "encoder") fileprivate var encoder: AudioEncoder
    @ModuleInfo(key: "quantizer") var quantizer: Quantizer

    /// Holds only the codebook itself; the EMA buffers in the checkpoint are training state.
    public class Quantizer: Module {
        @ModuleInfo(key: "codebook") var codebook: Codebook
        public class Codebook: Module {
            @ModuleInfo(key: "embed") var embed: MLXArray
            init(size: Int, dim: Int) { self._embed.wrappedValue = MLXArray.zeros([size, dim]) }
        }
        init(size: Int, dim: Int) { self._codebook.wrappedValue = Codebook(size: size, dim: dim) }
    }

    public let config: Apertus1p5AudioTokenizerConfiguration

    public init(_ config: Apertus1p5AudioTokenizerConfiguration) {
        self.config = config
        self._encoder.wrappedValue = AudioEncoder(
            hiddenSize: config.codebookDim, compress: 2, kernelSize: 7)
        self._quantizer.wrappedValue = Quantizer(
            size: config.codebookSize, dim: config.codebookDim)
    }

    /// Reconstruct weight-norm and reorder convolutions, exactly once at load.
    ///
    /// Also DROPS the decoder (`backbone.*`, `head.*`) and the quantiser's EMA buffers — see the
    /// header for why those exist but are not needed.
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        // Pair up g and v before touching anything, since both are needed to rebuild one weight.
        var g: [String: MLXArray] = [:], v: [String: MLXArray] = [:]
        var out: [String: MLXArray] = [:]
        for (k, value) in weights {
            if k.hasPrefix("backbone.") || k.hasPrefix("head.") { continue }
            if k.hasSuffix("cluster_size") || k.hasSuffix("inited") { continue }
            // EMA training buffers, and the PACKED copies of the codebook. Quantised bundles ship
            // the codebook twice -- plain float plus `.weight`/`.scales`/`.biases` -- and keeping
            // both makes the loader see a nested module where the tree has a leaf. The float copy
            // is full precision, and a codebook must not be quantised regardless: the encoder does
            // a nearest-neighbour search in it.
            if k.contains("embed_avg") { continue }
            if let r = k.range(of: ".parametrizations.weight.original0") {
                g[String(k[..<r.lowerBound])] = value
            } else if let r = k.range(of: ".parametrizations.weight.original1") {
                v[String(k[..<r.lowerBound])] = value
            } else {
                out[k] = value
            }
        }
        // torch LSTM -> MLX LSTM. `Wx`/`Wh` map straight across (same i,f,g,o gate order); torch
        // applies TWO biases and MLX one, so they are summed — dropping `bias_hh` would be a
        // silent, plausible-looking error.
        var lstmBias: [String: (ih: MLXArray?, hh: MLXArray?)] = [:]
        for (k, value) in out where k.contains(".lstm.") {
            guard let r = k.range(of: ".lstm.") else { continue }
            let base = String(k[..<r.upperBound])          // "...layers.13.lstm."
            let tail = String(k[r.upperBound...])          // "weight_ih_l0" etc
            guard let layerChar = tail.last, let layer = Int(String(layerChar)) else { continue }
            let target = "\(base)\(layer)"
            if tail.hasPrefix("weight_ih") { out[target + ".Wx"] = value }
            else if tail.hasPrefix("weight_hh") { out[target + ".Wh"] = value }
            else if tail.hasPrefix("bias_ih") {
                lstmBias[target, default: (nil, nil)].ih = value
            } else if tail.hasPrefix("bias_hh") {
                lstmBias[target, default: (nil, nil)].hh = value
            }
            out[k] = nil
        }
        for (target, b) in lstmBias {
            if let ih = b.ih, let hh = b.hh { out[target + ".bias"] = ih + hh }
        }

        for (base, direction) in v {
            guard let magnitude = g[base] else { continue }
            // w = g * v / ||v||, norm taken over every axis except the output channel — matching
            // torch's weight_norm(dim=0).
            let axes = Array(1 ..< direction.ndim)
            let norm = sqrt((direction * direction).sum(axes: axes, keepDims: true))
            var w = magnitude * direction / norm
            // torch Conv1d is [O, I, kW]; MLX is [O, kW, I].
            if w.ndim == 3 { w = w.transposed(0, 2, 1) }
            out["\(base).weight"] = w
        }
        return out
    }

    /// Encode mono PCM (shape [samples] or [1, samples]) to codebook ids, one per 600 samples.
    public func encode(_ pcm: MLXArray) -> MLXArray {
        var x = pcm.ndim == 1 ? pcm.reshaped(1, -1, 1) : pcm.reshaped(1, -1, 1)
        for layer in encoder.layers {
            switch layer {
            case let c as WeightNormConv1d: x = c(x)
            case let r as AudioResidualUnit: x = r(x)
            case let l as SEANetLSTM: x = l(x)
            default: x = elu(x)                        // the parameterless activation slots
            }
        }
        let dim = x.dim(2)
        let flat = x.reshaped(-1, dim)
        let book = quantizer.codebook.embed
        let bookSq = (book * book).sum(axis: 1)
        let cross = matmul(flat, book.transposed(1, 0))
        return argMin(bookSq.expandedDimensions(axis: 0) - 2 * cross, axis: 1)
    }
}
