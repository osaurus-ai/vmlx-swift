import Foundation
import MLX
import vMLXFluxKit

#if canImport(AppKit)
    import AppKit
    import ImageIO
#endif

public enum SeedVR2Preprocess {
    public struct Image {
        public let tensor: MLXArray
        public let width: Int
        public let height: Int
    }

    public static func read(_ url: URL, scale: Int, checkpoint: () throws -> Void = {}) throws
        -> Image
    {
        #if canImport(AppKit)
            guard [2, 4].contains(scale) else {
                throw FluxError.invalidField(
                    field: "scale", value: String(scale), allowed: "2 or 4")
            }
            try checkpoint()
            guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
                let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil)
                    as? [CFString: Any],
                let rawWidth = properties[kCGImagePropertyPixelWidth] as? Int,
                let rawHeight = properties[kCGImagePropertyPixelHeight] as? Int,
                rawWidth > 0, rawHeight > 0
            else { throw FluxError.invalidRequest("Cannot read SeedVR2 source dimensions") }
            try SeedVR2BundleMetadata.validateUpscale(
                sourceWidth: rawWidth, sourceHeight: rawHeight, scale: scale, format: .png)
            guard
                let image = CGImageSourceCreateThumbnailAtIndex(
                    source, 0,
                    [
                        kCGImageSourceCreateThumbnailFromImageAlways: true,
                        kCGImageSourceCreateThumbnailWithTransform: true,
                        kCGImageSourceThumbnailMaxPixelSize: max(rawWidth, rawHeight),
                    ] as CFDictionary)
            else { throw FluxError.invalidRequest("Cannot decode SeedVR2 source image") }
            let sourceWidth = image.width
            let sourceHeight = image.height
            let width = sourceWidth * scale
            let height = sourceHeight * scale
            // Host admission belongs to Osaurus; arithmetic here must still be representable.
            let rep = NSBitmapImageRep(cgImage: image)
            var bytes = [UInt8](repeating: 0, count: sourceWidth * sourceHeight * 3)
            for y in 0 ..< sourceHeight {
                try checkpoint()
                for x in 0 ..< sourceWidth {
                    var pixel = [Int](repeating: 0, count: rep.samplesPerPixel)
                    rep.getPixel(&pixel, atX: x, y: y)
                    let hasAlpha = rep.hasAlpha
                    let alphaFirst = rep.bitmapFormat.contains(.alphaFirst)
                    let first = hasAlpha && alphaFirst ? 1 : 0
                    let colorChannels = rep.samplesPerPixel - (hasAlpha ? 1 : 0)
                    guard colorChannels == 1 || colorChannels == 3 else {
                        throw FluxError.invalidRequest(
                            "SeedVR2 source must decode to RGB or grayscale")
                    }
                    let maximum = Double((1 << rep.bitsPerSample) - 1)
                    let alpha =
                        hasAlpha
                        ? Double(pixel[alphaFirst ? 0 : rep.samplesPerPixel - 1]) / maximum : 1
                    let premultiplied =
                        hasAlpha && !rep.bitmapFormat.contains(.alphaNonpremultiplied)
                    for c in 0 ..< 3 {
                        var value = Double(pixel[first + (colorChannels == 1 ? 0 : c)]) / maximum
                        if premultiplied && alpha > 0 { value /= alpha }
                        bytes[(y * sourceWidth + x) * 3 + c] = UInt8(
                            clamping: Int((value * 255).rounded()))
                    }
                }
            }
            let resized = try resizeRGB(
                bytes, width: sourceWidth, height: sourceHeight,
                targetWidth: width, targetHeight: height, checkpoint: checkpoint)
            let paddedWidth = (width + 15) / 16 * 16
            let paddedHeight = (height + 15) / 16 * 16
            var values = [Float](repeating: -1, count: 3 * paddedHeight * paddedWidth)
            for y in 0 ..< height {
                try checkpoint()
                for x in 0 ..< width {
                    for c in 0 ..< 3 {
                        values[c * paddedWidth * paddedHeight + y * paddedWidth + x] =
                            Float(resized[(y * width + x) * 3 + c]) / 255 * 2 - 1
                    }
                }
            }
            return Image(
                tensor: MLXArray(values, [1, 3, paddedHeight, paddedWidth]), width: width,
                height: height)
        #else
            throw FluxError.notImplemented("SeedVR2 image decoding requires AppKit")
        #endif
    }

    /// Pillow's separable bicubic filter, including half-pixel centers, antialias
    /// support and 22-bit integer coefficients/rounding for 8-bit RGB images.
    static func resizeRGB(
        _ input: [UInt8], width: Int, height: Int, targetWidth: Int, targetHeight: Int,
        checkpoint: () throws -> Void = {}
    ) throws -> [UInt8] {
        func coefficients(_ source: Int, _ destination: Int) -> [(Int, [Int64])] {
            let scale = Double(source) / Double(destination)
            let filterScale = max(scale, 1)
            let support = 2 * filterScale
            return (0 ..< destination).map { i in
                let center = (Double(i) + 0.5) * scale
                let start = max(Int(center - support + 0.5), 0)
                let end = min(Int(center + support + 0.5), source)
                var weights = (start ..< end).map { j -> Double in
                    let x = abs((Double(j) - center + 0.5) / filterScale)
                    if x < 1 { return (1.5 * x - 2.5) * x * x + 1 }
                    if x < 2 { return ((-0.5 * x + 2.5) * x - 4) * x + 2 }
                    return 0
                }
                let total = weights.reduce(0, +)
                weights = weights.map { $0 / total }
                return (
                    start,
                    weights.map { Int64(($0 * Double(1 << 22)).rounded(.toNearestOrAwayFromZero)) }
                )
            }
        }
        let cx = coefficients(width, targetWidth)
        let cy = coefficients(height, targetHeight)
        var horizontal = [UInt8](repeating: 0, count: targetWidth * height * 3)
        for y in 0 ..< height {
            try checkpoint()
            for x in 0 ..< targetWidth {
                for c in 0 ..< 3 {
                    var sum: Int64 = 1 << 21
                    for (j, weight) in cx[x].1.enumerated() {
                        sum += Int64(input[(y * width + cx[x].0 + j) * 3 + c]) * weight
                    }
                    horizontal[(y * targetWidth + x) * 3 + c] = UInt8(clamping: sum >> 22)
                }
            }
        }
        var output = [UInt8](repeating: 0, count: targetWidth * targetHeight * 3)
        for y in 0 ..< targetHeight {
            try checkpoint()
            for x in 0 ..< targetWidth {
                for c in 0 ..< 3 {
                    var sum: Int64 = 1 << 21
                    for (j, weight) in cy[y].1.enumerated() {
                        sum += Int64(horizontal[((cy[y].0 + j) * targetWidth + x) * 3 + c]) * weight
                    }
                    output[(y * targetWidth + x) * 3 + c] = UInt8(clamping: sum >> 22)
                }
            }
        }
        return output
    }
}
