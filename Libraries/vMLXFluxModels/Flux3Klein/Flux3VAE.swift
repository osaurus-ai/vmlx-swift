import Foundation
@preconcurrency import MLX
import MLXNN
import vMLXFluxKit

// AutoencoderKLFlux2 from mflux d587aac. NHWC inside the convolutional graph;
// GroupNorm reduces within channel groups over spatial positions in float32.
private final class Flux3VAEResnet {
    let norm1, norm2: MFluxGroupNorm
    let conv1, conv2: MFluxConv2D
    let shortcut: MFluxConv2D?
    init(store: MFluxStore, prefix: String) throws {
        norm1 = try store.groupNorm("vae", "\(prefix).norm1")
        norm2 = try store.groupNorm("vae", "\(prefix).norm2")
        conv1 = try store.conv2d("vae", "\(prefix).conv1", padding: 1)
        conv2 = try store.conv2d("vae", "\(prefix).conv2", padding: 1)
        shortcut = store.hasKey("vae", "\(prefix).conv_shortcut.weight") ? try store.conv2d("vae", "\(prefix).conv_shortcut") : nil
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = conv1(silu(norm1(x.asType(.float32)).asType(.bfloat16)))
        return (shortcut?(x) ?? x) + conv2(silu(norm2(h.asType(.float32)).asType(.bfloat16)))
    }
}

private final class Flux3VAEAttention {
    let norm: MFluxGroupNorm
    let q, k, v, out: MFluxLinear
    init(store: MFluxStore, prefix: String) throws {
        norm = try store.groupNorm("vae", "\(prefix).group_norm")
        // These eight attention projections are quantized in every shipped bundle.
        q = try store.linear("vae", "\(prefix).to_q", inputDimensions: 512, outputDimensions: 512, bias: true)
        k = try store.linear("vae", "\(prefix).to_k", inputDimensions: 512, outputDimensions: 512, bias: true)
        v = try store.linear("vae", "\(prefix).to_v", inputDimensions: 512, outputDimensions: 512, bias: true)
        out = try store.linear("vae", "\(prefix).to_out", inputDimensions: 512, outputDimensions: 512, bias: true)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let n = norm(x.asType(.float32)).asType(.bfloat16)
        let b = x.dim(0), h = x.dim(1), w = x.dim(2), c = x.dim(3)
        func split(_ value: MLXArray) -> MLXArray { value.reshaped([b, h * w, 1, c]).transposed(0, 2, 1, 3) }
        let attended = MLX.scaledDotProductAttention(queries: split(q(n)), keys: split(k(n)), values: split(v(n)),
            scale: 1 / Float(c).squareRoot(), mask: nil)
        return x + out(attended.transposed(0, 2, 1, 3).reshaped([b, h, w, c]))
    }
}

private final class Flux3VAEMid {
    let first, last: Flux3VAEResnet
    let attention: Flux3VAEAttention
    init(store: MFluxStore, prefix: String) throws {
        first = try Flux3VAEResnet(store: store, prefix: "\(prefix).resnets.0")
        attention = try Flux3VAEAttention(store: store, prefix: "\(prefix).attentions.0")
        last = try Flux3VAEResnet(store: store, prefix: "\(prefix).resnets.1")
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { last(attention(first(x))) }
}

final class Flux3VAE {
    private let encoderIn, encoderOut, decoderIn, decoderOut, quant, postQuant: MFluxConv2D
    private let encoderNorm, decoderNorm: MFluxGroupNorm
    private let encoderMid, decoderMid: Flux3VAEMid
    private let downBlocks, upBlocks: [[Flux3VAEResnet]]
    private let down, up: [MFluxConv2D]
    private let bnMean, bnStd: MLXArray

    init(store: MFluxStore) throws {
        encoderIn = try store.conv2d("vae", "encoder.conv_in", padding: 1)
        encoderOut = try store.conv2d("vae", "encoder.conv_out", padding: 1)
        decoderIn = try store.conv2d("vae", "decoder.conv_in", padding: 1)
        decoderOut = try store.conv2d("vae", "decoder.conv_out", padding: 1)
        encoderNorm = try store.groupNorm("vae", "encoder.conv_norm_out")
        decoderNorm = try store.groupNorm("vae", "decoder.conv_norm_out")
        encoderMid = try Flux3VAEMid(store: store, prefix: "encoder.mid_block")
        decoderMid = try Flux3VAEMid(store: store, prefix: "decoder.mid_block")
        quant = try store.conv2d("vae", "quant_conv")
        postQuant = try store.conv2d("vae", "post_quant_conv")
        downBlocks = try (0..<4).map { i in try (0..<2).map { j in
            try Flux3VAEResnet(store: store, prefix: "encoder.down_blocks.\(i).resnets.\(j)")
        } }
        upBlocks = try (0..<4).map { i in try (0..<3).map { j in
            try Flux3VAEResnet(store: store, prefix: "decoder.up_blocks.\(i).resnets.\(j)")
        } }
        down = try (0..<3).map { try store.conv2d("vae", "encoder.down_blocks.\($0).downsamplers.0.conv", stride: 2) }
        up = try (0..<3).map { try store.conv2d("vae", "decoder.up_blocks.\($0).upsamplers.0.conv", padding: 1) }
        bnMean = try store.tensor("vae", "bn.running_mean").reshaped([1, 128, 1, 1])
        let variance = try store.tensor("vae", "bn.running_var").reshaped([1, 128, 1, 1])
        bnStd = sqrt(variance + 1e-4)
    }

    /// NCHW [-1,1] float32 image -> BF16 normalized (1, H/16*W/16, 128) reference tokens.
    func encodeTokens(_ image: MLXArray) throws -> MLXArray {
        var x = encoderIn(image.transposed(0, 2, 3, 1))
        for (index, blocks) in downBlocks.enumerated() {
            try FluxInterruptScope.check()
            for block in blocks { x = block(x) }
            if index < down.count {
                // mflux pads only bottom/right, then uses a valid 3x3 stride-2 convolution.
                let right = MLXArray.zeros([x.dim(0), x.dim(1), 1, x.dim(3)], dtype: x.dtype)
                x = concatenated([x, right], axis: 2)
                let bottom = MLXArray.zeros([x.dim(0), 1, x.dim(2), x.dim(3)], dtype: x.dtype)
                x = down[index](concatenated([x, bottom], axis: 1))
            }
            eval(x)
        }
        x = encoderMid(x)
        x = quant(encoderOut(silu(encoderNorm(x.asType(.float32)).asType(.bfloat16))))
        let mean = x[0..., 0..., 0..., ..<32].transposed(0, 3, 1, 2)
        let patch = Self.patchify(mean)
        let normalized = (patch - bnMean.asType(patch.dtype)) / bnStd.asType(patch.dtype)
        let b = normalized.dim(0), h = normalized.dim(2), w = normalized.dim(3)
        return normalized.reshaped([b, 128, h * w]).transposed(0, 2, 1).asType(.bfloat16)
    }

    /// Float32 Euler tokens -> raw NCHW [-1,1] image (postprocessing is the pipeline's job).
    func decodeTokens(_ tokens: MLXArray, height: Int, width: Int, materializeStages: Bool = true) throws -> MLXArray {
        let h = height / 16, w = width / 16
        let packed = tokens.reshaped([tokens.dim(0), h, w, 128]).transposed(0, 3, 1, 2)
        var x = Self.unpatchify(packed * bnStd + bnMean).transposed(0, 2, 3, 1)
        x = decoderMid(decoderIn(postQuant(x)))
        for (index, blocks) in upBlocks.enumerated() {
            try FluxInterruptScope.check()
            for block in blocks { x = block(x) }
            if index < up.count { x = up[index](repeated(repeated(x, count: 2, axis: 1), count: 2, axis: 2)) }
            if materializeStages { eval(x) }
        }
        return decoderOut(silu(decoderNorm(x.asType(.float32)).asType(.bfloat16))).transposed(0, 3, 1, 2)
    }

    /// c' = c*4 + dy*2 + dx (not channel-last packing).
    static func patchify(_ x: MLXArray) -> MLXArray {
        let b = x.dim(0), c = x.dim(1), h = x.dim(2), w = x.dim(3)
        return x.reshaped([b, c, h / 2, 2, w / 2, 2]).transposed(0, 1, 3, 5, 2, 4).reshaped([b, c * 4, h / 2, w / 2])
    }
    static func unpatchify(_ x: MLXArray) -> MLXArray {
        let b = x.dim(0), c = x.dim(1), h = x.dim(2), w = x.dim(3)
        return x.reshaped([b, c / 4, 2, 2, h, w]).transposed(0, 1, 4, 2, 5, 3).reshaped([b, c / 4, h * 2, w * 2])
    }
}
