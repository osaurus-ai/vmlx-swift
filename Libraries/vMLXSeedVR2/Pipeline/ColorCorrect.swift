// SeedVR2 wavelet + LAB rank transfer, following mflux seedvr2_util.py.
// CPU float32 and stable rank ordering matter: tiny GPU LAB differences can
// reorder tied colors and assign a very different pixel from the reference.
import Foundation
import MLX

public enum SeedVR2ColorCorrect {
    public static func labTransfer(
        content: MLXArray, style: MLXArray, luminanceWeight: Float = 0.8,
        checkpoint: () throws -> Void = {}
    ) throws -> MLXArray {
        precondition(content.shape == style.shape && content.ndim == 4 && content.dim(1) == 3)
        let shape = content.shape
        let h = shape[2]
        let w = shape[3]
        let pixels = h * w
        let c = try content.asType(.float32).asArrayChecked(Float.self)
        let s = try style.asType(.float32).asArrayChecked(Float.self)
        var result = [Float](repeating: 0, count: c.count)
        for batch in 0 ..< shape[0] {
            let offset = batch * pixels * 3
            let cb = Array(c[offset ..< offset + pixels * 3])
            let sb = Array(s[offset ..< offset + pixels * 3])
            let (high, _) = try decompose(cb, h: h, w: w, checkpoint: checkpoint)
            let (_, low) = try decompose(sb, h: h, w: w, checkpoint: checkpoint)
            var clab = [[Float]](repeating: [], count: 3)
            var slab = clab
            for i in 0 ..< pixels {
                if i % 4096 == 0 { try checkpoint() }
                let crgb = (0 ..< 3).map { channel -> Float in
                    let v = min(max(high[channel * pixels + i] + low[channel * pixels + i], -1), 1)
                    return (v + 1) * 0.5
                }
                let srgb = (0 ..< 3).map { min(max((sb[$0 * pixels + i] + 1) * 0.5, 0), 1) }
                let cv = rgbToLab(crgb)
                let sv = rgbToLab(srgb)
                for channel in 0 ..< 3 {
                    clab[channel].append(cv[channel])
                    slab[channel].append(sv[channel])
                }
            }
            var matched: [[Float]] = []
            for channel in 0 ..< 3 {
                try checkpoint()
                matched.append(try histMatch(clab[channel], slab[channel], checkpoint: checkpoint))
            }
            for i in 0 ..< pixels {
                if i % 4096 == 0 { try checkpoint() }
                let l =
                    luminanceWeight < 1
                    ? luminanceWeight * clab[0][i] + (1 - luminanceWeight) * matched[0][i]
                    : clab[0][i]
                let rgb = labToRGB([l, matched[1][i], matched[2][i]])
                for channel in 0 ..< 3 {
                    result[offset + channel * pixels + i] = min(max(rgb[channel], 0), 1) * 2 - 1
                }
            }
        }
        return MLXArray(result, shape).asType(content.dtype)
    }

    static func histMatch(
        _ source: [Float], _ reference: [Float], checkpoint: () throws -> Void = {}
    ) throws -> [Float] {
        let indices = try stableIndices(source, checkpoint: checkpoint)
        let refIndices = try stableIndices(reference, checkpoint: checkpoint)
        var output = source
        for (rank, index) in indices.enumerated() {
            if rank % 4096 == 0 { try checkpoint() }
            output[index] = reference[refIndices[rank]]
        }
        return output
    }

    /// Bounded sort chunks + stable merges permit cancellation even during large
    /// (4K) color-transfer stages. Equal values retain the original pixel order.
    private static func stableIndices(_ values: [Float], checkpoint: () throws -> Void) throws
        -> [Int]
    {
        var indices = Array(values.indices)
        func before(_ a: Int, _ b: Int) -> Bool {
            values[a] == values[b] ? a < b : values[a] < values[b]
        }
        let chunk = 4096
        for start in stride(from: 0, to: indices.count, by: chunk) {
            try checkpoint()
            let end = min(start + chunk, indices.count)
            indices.replaceSubrange(start ..< end, with: indices[start ..< end].sorted(by: before))
        }
        var temporary = indices
        var size = chunk
        while size < indices.count {
            for start in stride(from: 0, to: indices.count, by: size * 2) {
                let mid = min(start + size, indices.count)
                let end = min(start + size * 2, indices.count)
                var left = start
                var right = mid
                for destination in start ..< end {
                    if destination % 4096 == 0 { try checkpoint() }
                    if left < mid && (right >= end || before(indices[left], indices[right])) {
                        temporary[destination] = indices[left]
                        left += 1
                    } else {
                        temporary[destination] = indices[right]
                        right += 1
                    }
                }
            }
            swap(&indices, &temporary)
            size *= 2
        }
        return indices
    }

    private static func decompose(_ image: [Float], h: Int, w: Int, checkpoint: () throws -> Void)
        throws -> ([Float], [Float])
    {
        var high = [Float](repeating: 0, count: image.count)
        var current = image
        let taps: [Float] = [0.0625, 0.125, 0.0625, 0.125, 0.25, 0.125, 0.0625, 0.125, 0.0625]
        for level in 0 ..< 5 {
            let radius = min(1 << level, max(1, min(h, w) / 8))
            var low = [Float](repeating: 0, count: image.count)
            for channel in 0 ..< 3 {
                for y in 0 ..< h {
                    try checkpoint()
                    for x in 0 ..< w {
                        var sum: Float = 0
                        for ky in 0 ..< 3 {
                            for kx in 0 ..< 3 {
                                let yy = min(max(y + (ky - 1) * radius, 0), h - 1)
                                let xx = min(max(x + (kx - 1) * radius, 0), w - 1)
                                sum += taps[ky * 3 + kx] * current[channel * h * w + yy * w + xx]
                            }
                        }
                        let i = channel * h * w + y * w + x
                        low[i] = sum
                        high[i] += current[i] - sum
                    }
                }
            }
            current = low
        }
        return (high, current)
    }

    private static func rgbToLab(_ rgb: [Float]) -> [Float] {
        let v = rgb.map { $0 > 0.04045 ? powf(($0 + 0.055) / 1.055, 2.4) : $0 / 12.92 }
        let x = (v[0] * 0.4124564 + v[1] * 0.3575761 + v[2] * 0.1804375) / 0.95047
        let y = v[0] * 0.2126729 + v[1] * 0.7151522 + v[2] * 0.0721750
        let z = (v[0] * 0.0193339 + v[1] * 0.1191920 + v[2] * 0.9503041) / 1.08883
        func f(_ t: Float) -> Float {
            t > Float(pow(6.0 / 29.0, 3)) ? cbrtf(t) : (Float(pow(29.0 / 3.0, 3)) * t + 16) / 116
        }
        let fx = f(x)
        let fy = f(y)
        let fz = f(z)
        return [116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)]
    }

    private static func labToRGB(_ lab: [Float]) -> [Float] {
        let fy = (lab[0] + 16) / 116
        let fx = lab[1] / 500 + fy
        let fz = fy - lab[2] / 200
        func inv(_ t: Float) -> Float {
            t > Float(6.0 / 29.0) ? powf(t, 3) : (116 * t - 16) / Float(pow(29.0 / 3.0, 3))
        }
        let x = inv(fx) * 0.95047
        let y = inv(fy)
        let z = inv(fz) * 1.08883
        let v: [Float] = [
            x * 3.2404542 - y * 1.5371385 - z * 0.4985314,
            -x * 0.9692660 + y * 1.8760108 + z * 0.0415560,
            x * 0.0556434 - y * 0.2040259 + z * 1.0572252,
        ]
        return v.map {
            $0 > 0.0031308 ? 1.055 * powf(max($0, 0), Float(1.0 / 2.4)) - 0.055 : 12.92 * $0
        }
    }
}
