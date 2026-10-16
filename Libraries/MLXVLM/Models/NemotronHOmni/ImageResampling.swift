import Foundation
import MLX

/// Float32 separable Keys cubic interpolation, align_corners=false and
/// antialias=true, as used by Nemotron's bundled image processor. Unlike PIL's
/// uint8 resampler this retains overshoot and fractional pixels between passes.
/// Reference: PyTorch aten/src/ATen/native/cpu/UpSampleKernel.cpp,
/// HelperInterpCubic and _compute_indices_min_size_weights_aa.
enum NemotronImageResampling {
    private static func coefficients(input: Int, output: Int) -> [(Int, [Float])] {
        let scale = Float(input) / Float(output)
        let filterScale = max(scale, 1)
        let support = 2 * filterScale
        func cubic(_ value: Float) -> Float {
            let x = abs(value)
            if x < 1 { return ((1.5 * x - 2.5) * x) * x + 1 }
            if x < 2 { return (((-0.5 * x + 2.5) * x - 4) * x) + 2 }
            return 0
        }
        return (0 ..< output).map { index in
            let center = scale * (Float(index) + 0.5)
            let first = max(0, Int(center - support + 0.5))
            let end = min(input, Int(center + support + 0.5))
            let weights = (first ..< end).map {
                cubic((Float($0) - center + 0.5) / filterScale)
            }
            let sum = weights.reduce(Float(0), +)
            return (first, weights.map { $0 / sum })
        }
    }

    /// Interleaved RGB, values in the source 0...255 range (no clipping).
    static func resize(
        _ pixels: [UInt8], width: Int, height: Int, targetWidth: Int, targetHeight: Int
    ) -> [Float] {
        precondition(width > 0 && height > 0 && targetWidth > 0 && targetHeight > 0)
        precondition(pixels.count == width * height * 3)
        var horizontal = pixels.map(Float.init)
        if targetWidth != width {
            let columns = coefficients(input: width, output: targetWidth)
            horizontal = [Float](repeating: 0, count: height * targetWidth * 3)
            for row in 0 ..< height {
                for column in 0 ..< targetWidth {
                    let (first, weights) = columns[column]
                    for channel in 0 ..< 3 {
                        var value: Float = 0
                        for (offset, weight) in weights.enumerated() {
                            value += Float(pixels[(row * width + first + offset) * 3 + channel]) * weight
                        }
                        horizontal[(row * targetWidth + column) * 3 + channel] = value
                    }
                }
            }
        }
        guard targetHeight != height else { return horizontal }
        let rows = coefficients(input: height, output: targetHeight)
        var output = [Float](repeating: 0, count: targetHeight * targetWidth * 3)
        for row in 0 ..< targetHeight {
            let (first, weights) = rows[row]
            for column in 0 ..< targetWidth {
                for channel in 0 ..< 3 {
                    var value: Float = 0
                    for (offset, weight) in weights.enumerated() {
                        value += horizontal[((first + offset) * targetWidth + column) * 3 + channel] * weight
                    }
                    output[(row * targetWidth + column) * 3 + channel] = value
                }
            }
        }
        return output
    }
}
