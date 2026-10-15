// SeedVR2's reference 512px tiles / 64px overlap, cosine-weighted CPU float32 assembly.
import Foundation
import MLX

public enum SeedVR2VAETiling {
    public static func encode(_ image: MLXArray, vae: SeedVR2VAE, checkpoint: () throws -> Void)
        throws -> MLXArray
    {
        let h = image.dim(-2)
        let w = image.dim(-1)
        if h <= 512 && w <= 512 { return try vae.encode(image, checkpoint: checkpoint) }
        return try assemble(
            height: h / 8, width: w / 8, channels: 16, tile: 64, overlap: 8,
            outputScale: 1, checkpoint: checkpoint
        ) { y, x, endY, endX in
            let encoded = try vae.encode(
                image[0..., 0..., (y * 8) ..< (endY * 8), (x * 8) ..< (endX * 8)],
                checkpoint: checkpoint)
            return encoded.squeezed(axis: 2)
        }.expandedDimensions(axis: 2)
    }

    public static func decode(_ latent: MLXArray, vae: SeedVR2VAE, checkpoint: () throws -> Void)
        throws -> MLXArray
    {
        let h = latent.dim(-2)
        let w = latent.dim(-1)
        if h <= 64 && w <= 64 {
            return try vae.decode(latent, checkpoint: checkpoint).squeezed(axis: 2)
        }
        return try assemble(
            height: h, width: w, channels: 3, tile: 64, overlap: 8,
            outputScale: 8, checkpoint: checkpoint
        ) { y, x, endY, endX in
            try vae.decode(latent[0..., 0..., 0..., y ..< endY, x ..< endX], checkpoint: checkpoint)
                .squeezed(axis: 2)
        }
    }

    static func assemble(
        height: Int, width: Int, channels: Int, tile: Int, overlap: Int,
        outputScale: Int, checkpoint: () throws -> Void,
        operation: (Int, Int, Int, Int) throws -> MLXArray
    ) throws -> MLXArray {
        let oh = height * outputScale
        let ow = width * outputScale
        let plane = oh * ow
        var output = [Float](repeating: 0, count: plane * channels)
        var count = [Float](repeating: 0, count: plane)
        let stride = tile - overlap
        let overlapPixels = overlap * outputScale
        let ramp = (0 ..< overlapPixels).map { i -> Float in
            let t = overlapPixels > 1 ? Float(i) / Float(overlapPixels - 1) : 0
            return 0.5 - 0.5 * cosf(t * Float.pi)
        }
        for y in Swift.stride(from: 0, to: height, by: stride) {
            let endY = min(y + tile, height)
            if y > 0 && endY - y <= overlap { continue }
            for x in Swift.stride(from: 0, to: width, by: stride) {
                let endX = min(x + tile, width)
                if x > 0 && endX - x <= overlap { continue }
                try checkpoint()
                let tensor = try operation(y, x, endY, endX)
                let th = tensor.dim(-2)
                let tw = tensor.dim(-1)
                let values = try tensor.asType(.float32).asArrayChecked(Float.self)
                let eh = min((endY - y) * outputScale, th)
                let ew = min((endX - x) * outputScale, tw)
                let ovh = min(overlapPixels, eh - 1)
                let ovw = min(overlapPixels, ew - 1)
                var wy = [Float](repeating: 1, count: eh)
                var wx = [Float](repeating: 1, count: ew)
                for i in 0 ..< ovh {
                    if y > 0 { wy[i] = ramp[i] }
                    if endY < height { wy[eh - ovh + i] = 1 - ramp[i] }
                }
                for i in 0 ..< ovw {
                    if x > 0 { wx[i] = ramp[i] }
                    if endX < width { wx[ew - ovw + i] = 1 - ramp[i] }
                }
                for yy in 0 ..< eh {
                    try checkpoint()
                    for xx in 0 ..< ew {
                        let weight = wy[yy] * wx[xx]
                        let dest = (y * outputScale + yy) * ow + x * outputScale + xx
                        count[dest] += weight
                        for c in 0 ..< channels {
                            output[c * plane + dest] += values[c * th * tw + yy * tw + xx] * weight
                        }
                    }
                }
            }
        }
        for c in 0 ..< channels {
            try checkpoint()
            for i in 0 ..< plane { output[c * plane + i] /= max(count[i], 0.000001) }
        }
        return MLXArray(output, [1, channels, oh, ow])
    }
}
