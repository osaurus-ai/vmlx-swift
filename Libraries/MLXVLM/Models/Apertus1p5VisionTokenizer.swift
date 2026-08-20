// Apertus 1.5 vision tokenizer — the ENCODER half of a VQ-VAE.
//
// Apertus 1.5 is multimodal in a way that is structurally unlike every other VLM in this repo, and
// the difference is the whole reason this file is short on plumbing and long on convolutions.
//
// The usual design is an encoder plus a PROJECTOR: a vision transformer produces continuous
// embeddings which are linearly mapped into the language model's hidden space and spliced into the
// embedding sequence. Apertus 1.5 instead QUANTISES the image into discrete ids drawn from a
// 131,072-entry codebook, offsets them by `image_token_offset`, and feeds them as ORDINARY TOKENS.
// The arithmetic in the config confirms the intent exactly:
//
//     image_token_offset 131272 + codebook 131072 == 262344 == audio_token_offset
//
// inside a text vocabulary of 266,752. So the codebook lives *in the token embedding table*, and the
// language model needs no architectural change whatsoever — no projector, no cross-attention, no
// merge step. The entire port is: turn pixels into 256 integers.
//
// Only the ENCODER ships in the bundle (247 tensors, no decoder), which is right: a VLM never needs
// to turn tokens back into pixels.
//
// The architecture is taming-transformers' `Encoder` verbatim, confirmed against the weights rather
// than assumed:
//
//     conv_in 3->256
//     down.0  256 -> 256  x4 res blocks, downsample      256x256 -> 128x128
//     down.1  256 -> 256  x4 res blocks, downsample      128x128 ->  64x64
//     down.2  256 -> 512  x4 res blocks, downsample       64x64  ->  32x32   (nin_shortcut)
//     down.3  512 -> 512  x4 res blocks, downsample       32x32  ->  16x16
//     down.4  512 -> 1024 x4 res blocks, ATTENTION         16x16            (nin_shortcut)
//     mid     block_1, attn_1, block_2                     16x16
//     norm_out, conv_out 1024 -> 256                       16x16
//     quant_conv 256 -> 256 (1x1)                          16x16
//     quantize: nearest neighbour in a [131072, 256] codebook
//
// 16 x 16 = 256 latent positions, hence exactly 256 image tokens per image.

import CoreImage
import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Building blocks

/// taming-transformers uses swish (x * sigmoid(x)) throughout, NOT gelu. Getting this wrong changes
/// every activation subtly and shows up as slightly-wrong codebook ids rather than as a crash.
private func swish(_ x: MLXArray) -> MLXArray { x * sigmoid(x) }

/// `Normalize` in taming is `GroupNorm(num_groups=32, eps=1e-6, affine=True)`.
///
/// `pytorchCompatible: true` matters: MLX's default groups the channels differently, so without it
/// the normalisation is subtly wrong in a way that produces plausible-looking output.
private func normalize(_ channels: Int) -> GroupNorm {
    GroupNorm(groupCount: 32, dimensions: channels, eps: 1e-6, affine: true, pytorchCompatible: true)
}

/// Pre-activation residual block. `nin_shortcut` exists ONLY where the channel count changes
/// (stages 2 and 4 here), which is why it is optional rather than always present.
private class ResnetBlock: Module {
    @ModuleInfo(key: "norm1") var norm1: GroupNorm
    @ModuleInfo(key: "conv1") var conv1: Conv2d
    @ModuleInfo(key: "norm2") var norm2: GroupNorm
    @ModuleInfo(key: "conv2") var conv2: Conv2d
    @ModuleInfo(key: "nin_shortcut") var ninShortcut: Conv2d?

    init(inChannels: Int, outChannels: Int) {
        self._norm1.wrappedValue = normalize(inChannels)
        self._conv1.wrappedValue = Conv2d(
            inputChannels: inChannels, outputChannels: outChannels, kernelSize: 3, padding: 1)
        self._norm2.wrappedValue = normalize(outChannels)
        self._conv2.wrappedValue = Conv2d(
            inputChannels: outChannels, outputChannels: outChannels, kernelSize: 3, padding: 1)
        self._ninShortcut.wrappedValue =
            inChannels == outChannels
            ? nil
            : Conv2d(inputChannels: inChannels, outputChannels: outChannels, kernelSize: 1)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = conv1(swish(norm1(x)))
        h = conv2(swish(norm2(h)))
        return (ninShortcut.map { $0(x) } ?? x) + h
    }
}

/// Single-head spatial self-attention over the 16x16 grid, with 1x1 convolutions standing in for the
/// projections — taming's `AttnBlock`. Applied only where the resolution is in `attn_resolutions`.
private class AttnBlock: Module {
    let channels: Int
    @ModuleInfo(key: "norm") var norm: GroupNorm
    @ModuleInfo(key: "q") var q: Conv2d
    @ModuleInfo(key: "k") var k: Conv2d
    @ModuleInfo(key: "v") var v: Conv2d
    @ModuleInfo(key: "proj_out") var projOut: Conv2d

    init(channels: Int) {
        self.channels = channels
        self._norm.wrappedValue = normalize(channels)
        self._q.wrappedValue = Conv2d(inputChannels: channels, outputChannels: channels, kernelSize: 1)
        self._k.wrappedValue = Conv2d(inputChannels: channels, outputChannels: channels, kernelSize: 1)
        self._v.wrappedValue = Conv2d(inputChannels: channels, outputChannels: channels, kernelSize: 1)
        self._projOut.wrappedValue = Conv2d(
            inputChannels: channels, outputChannels: channels, kernelSize: 1)
    }

    /// `x` is NHWC (MLX's convolution layout).
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = norm(x)
        let (B, H, W) = (x.dim(0), x.dim(1), x.dim(2))
        // Flatten the grid to a sequence: [B, H*W, C]. Single head, scale 1/sqrt(C) — taming applies
        // the scale to the SCORES, which is what scaledDotProductAttention does too.
        let qh = q(h).reshaped(B, H * W, channels)
        let kh = k(h).reshaped(B, H * W, channels)
        let vh = v(h).reshaped(B, H * W, channels)
        let attended = MLXFast.scaledDotProductAttention(
            queries: qh.expandedDimensions(axis: 1),
            keys: kh.expandedDimensions(axis: 1),
            values: vh.expandedDimensions(axis: 1),
            scale: 1.0 / sqrt(Float(channels)),
            mask: .none
        ).squeezed(axis: 1)
        return x + projOut(attended.reshaped(B, H, W, channels))
    }
}

/// Stride-2 convolution with taming's ASYMMETRIC padding: it pads right and bottom by one and uses
/// no padding in the convolution itself. A symmetric `padding: 1` would shift the whole feature map
/// by half a pixel — silently, and only visible as wrong codebook ids.
private class Downsample: Module {
    @ModuleInfo(key: "conv") var conv: Conv2d

    init(channels: Int) {
        self._conv.wrappedValue = Conv2d(
            inputChannels: channels, outputChannels: channels, kernelSize: 3, stride: 2, padding: 0)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // NHWC: pad H and W on the high side only.
        let padded = padded(x, widths: [.init((0, 0)), .init((0, 1)), .init((0, 1)), .init((0, 0))])
        return conv(padded)
    }
}

/// One resolution stage: N residual blocks, optional attention, optional downsample. Modelled as a
/// `Module` with array children so the weight paths land as `down.<i>.block.<j>...`, matching the
/// checkpoint exactly.
private class DownStage: Module {
    @ModuleInfo(key: "block") var block: [ResnetBlock]
    @ModuleInfo(key: "attn") var attn: [AttnBlock]
    @ModuleInfo(key: "downsample") var downsample: Downsample?

    init(inChannels: Int, outChannels: Int, numBlocks: Int, useAttention: Bool, downsampleAfter: Bool)
    {
        var blocks: [ResnetBlock] = []
        var attns: [AttnBlock] = []
        var c = inChannels
        for _ in 0 ..< numBlocks {
            blocks.append(ResnetBlock(inChannels: c, outChannels: outChannels))
            c = outChannels
            if useAttention { attns.append(AttnBlock(channels: outChannels)) }
        }
        self._block.wrappedValue = blocks
        self._attn.wrappedValue = attns
        self._downsample.wrappedValue = downsampleAfter ? Downsample(channels: outChannels) : nil
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for (i, b) in block.enumerated() {
            h = b(h)
            if i < attn.count { h = attn[i](h) }
        }
        return downsample.map { $0(h) } ?? h
    }
}

private class MidBlock: Module {
    @ModuleInfo(key: "block_1") var block1: ResnetBlock
    @ModuleInfo(key: "attn_1") var attn1: AttnBlock
    @ModuleInfo(key: "block_2") var block2: ResnetBlock

    init(channels: Int) {
        self._block1.wrappedValue = ResnetBlock(inChannels: channels, outChannels: channels)
        self._attn1.wrappedValue = AttnBlock(channels: channels)
        self._block2.wrappedValue = ResnetBlock(inChannels: channels, outChannels: channels)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { block2(attn1(block1(x))) }
}

// MARK: - Encoder

private class VQEncoder: Module {
    @ModuleInfo(key: "conv_in") var convIn: Conv2d
    @ModuleInfo(key: "down") var down: [DownStage]
    @ModuleInfo(key: "mid") var mid: MidBlock
    @ModuleInfo(key: "norm_out") var normOut: GroupNorm
    @ModuleInfo(key: "conv_out") var convOut: Conv2d

    init(_ config: Apertus1p5VisionTokenizerConfiguration) {
        let base = config.baseChannels
        let mults = config.channelMultiplier
        self._convIn.wrappedValue = Conv2d(
            inputChannels: config.inChannels, outputChannels: base, kernelSize: 3, padding: 1)

        var stages: [DownStage] = []
        var resolution = config.resolution
        var inC = base
        for (i, mult) in mults.enumerated() {
            let outC = base * mult
            let isLast = i == mults.count - 1
            stages.append(
                DownStage(
                    inChannels: inC, outChannels: outC, numBlocks: config.numResBlocks,
                    useAttention: config.attnResolutions.contains(resolution),
                    downsampleAfter: !isLast))
            inC = outC
            if !isLast { resolution /= 2 }
        }
        self._down.wrappedValue = stages
        self._mid.wrappedValue = MidBlock(channels: inC)
        self._normOut.wrappedValue = normalize(inC)
        self._convOut.wrappedValue = Conv2d(
            inputChannels: inC, outputChannels: config.latentChannels, kernelSize: 3, padding: 1)
    }

    /// `x` is NHWC in [-1, 1]. Returns the pre-quantisation latent, NHWC.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = convIn(x)
        for stage in down { h = stage(h) }
        h = mid(h)
        return convOut(swish(normOut(h)))
    }
}

// MARK: - Tokenizer

public class Apertus1p5VisionTokenizer: Module {
    @ModuleInfo(key: "encoder") fileprivate var encoder: VQEncoder
    @ModuleInfo(key: "quant_conv") var quantConv: Conv2d
    @ModuleInfo(key: "quantize") var quantize: VQEmbedding

    /// The codebook. Held in its own `Module` purely so the weight path is
    /// `quantize.embedding.weight`, matching the checkpoint.
    public class VQEmbedding: Module {
        @ModuleInfo(key: "embedding") var embedding: Embedding
        init(codebookSize: Int, embedDim: Int) {
            self._embedding.wrappedValue = Embedding(
                embeddingCount: codebookSize, dimensions: embedDim)
        }
    }

    public init(_ config: Apertus1p5VisionTokenizerConfiguration) {
        self._encoder.wrappedValue = VQEncoder(config)
        self._quantConv.wrappedValue = Conv2d(
            inputChannels: config.latentChannels, outputChannels: config.embedDim, kernelSize: 1)
        self._quantize.wrappedValue = VQEmbedding(
            codebookSize: config.codebookSize, embedDim: config.embedDim)
    }

    /// Encode one preprocessed image (NHWC, [-1, 1]) to codebook ids in raster order.
    ///
    /// The quantiser is nearest-neighbour in L2, expanded to `|z|^2 - 2 z.e + |e|^2` so it is two
    /// reductions and one matmul rather than materialising a [positions, 131072, 256] difference
    /// tensor — which at 256 positions would be 34 GB.
    ///
    /// `|z|^2` is dropped because it is constant across the argmin, and its absence is deliberate
    /// rather than an oversight: keeping it would change nothing but the arithmetic cost.
    public func encode(_ pixels: MLXArray) -> MLXArray {
        let latent = quantConv(encoder(pixels))          // [B, 16, 16, embedDim]
        let dim = latent.dim(3)
        let flat = latent.reshaped(-1, dim)              // [B*256, embedDim]
        let codebook = quantize.embedding.weight         // [codebookSize, embedDim]
        let codebookSq = (codebook * codebook).sum(axis: 1)          // [codebookSize]
        let cross = matmul(flat, codebook.transposed(1, 0))          // [B*256, codebookSize]
        let distances = codebookSq.expandedDimensions(axis: 0) - 2 * cross
        return argMin(distances, axis: 1)                // [B*256]
    }

    /// Reorder torch's OIHW convolution weights to MLX's OHWI, and leave everything else alone.
    /// Bundles converted by MLX tooling are already OHWI, so the reorder is gated on shape rather
    /// than applied blindly — a double transpose is silent and catastrophic.

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var out = [String: MLXArray]()
        for (k, v) in weights {
            // Quantised bundles ship the codebook TWICE: the plain float tensor plus a packed
            // `.weight`/`.scales`/`.biases` triple. Keeping both makes the loader see a nested
            // module where the tree has a leaf parameter (`incompatibleItems`). The float copy is
            // full precision here, so the packed siblings are simply dropped -- and the codebook is
            // the one tensor that must NOT be quantised anyway, since the encoder does a
            // nearest-neighbour search in it.
            guard k.contains("conv") || k.hasSuffix(".q.weight") || k.hasSuffix(".k.weight")
                || k.hasSuffix(".v.weight") || k.hasSuffix(".proj_out.weight")
                || k.contains("nin_shortcut")
            else {
                out[k] = v
                continue
            }
            // torch: [O, I, kH, kW]. MLX: [O, kH, kW, I]. Already-MLX weights have their SMALLEST
            // trailing dims in positions 1-2 (the kernel), so test that rather than guessing.
            if v.ndim == 4, v.dim(1) > v.dim(2) || v.dim(1) > v.dim(3) {
                out[k] = v.transposed(0, 2, 3, 1)
            } else {
                out[k] = v
            }
        }
        return out
    }
}

// MARK: - Configuration

public struct Apertus1p5VisionTokenizerConfiguration: Codable, Sendable {
    public let codebookSize: Int
    public let embedDim: Int
    public let latentChannels: Int
    public let baseChannels: Int
    public let channelMultiplier: [Int]
    public let numResBlocks: Int
    public let attnResolutions: [Int]
    public let resolution: Int
    public let inChannels: Int

    enum CodingKeys: String, CodingKey {
        case codebookSize = "codebook_size"
        case embedDim = "embed_dim"
        case latentChannels = "latent_channels"
        case baseChannels = "base_channels"
        case channelMultiplier = "channel_multiplier"
        case numResBlocks = "num_res_blocks"
        case attnResolutions = "attn_resolutions"
        case resolution
        case inChannels = "in_channels"
    }
}
