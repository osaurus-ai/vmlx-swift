import Foundation
@preconcurrency import MLX
#if canImport(AppKit)
import AppKit
import CoreImage
#endif

// MARK: - ImageIO
//
// Write an MLX tensor to a PNG on disk. Used by every model as the
// final step of `generate()` / `edit()` / `upscale()`. Isolated here so
// the VAE decode path has a single function to call at the end of
// sampling.
//
// Expected input shape: (B, C=3, H, W) float in [0, 1]. Values outside
// that range are clamped.

public enum ImageIO {
    public enum RGBNormalization: Sendable {
        case zeroToOne
        case minusOneToOne
        case openAIClip
    }

    public static func dimensions(of url: URL) throws -> (width: Int, height: Int) {
        #if canImport(AppKit)
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
              let width = properties[kCGImagePropertyPixelWidth] as? Int,
              let height = properties[kCGImagePropertyPixelHeight] as? Int
        else {
            throw FluxError.invalidRequest("failed to read image dimensions at \(url.path)")
        }
        return (width, height)
        #else
        throw FluxError.notImplemented("ImageIO.dimensions requires AppKit")
        #endif
    }

    public static func readRGBValues(
        _ url: URL,
        width: Int,
        height: Int,
        normalization: RGBNormalization
    ) throws -> [Float] {
        #if canImport(AppKit)
        guard width > 0, height > 0 else {
            throw FluxError.invalidRequest("image read dimensions must be positive")
        }
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
        else {
            throw FluxError.invalidRequest("failed to read image at \(url.path)")
        }

        let bytesPerRow = width * 4
        var rgba = [UInt8](repeating: 0, count: height * bytesPerRow)
        let colorSpace = image.colorSpace ?? CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue
            | CGBitmapInfo.byteOrder32Big.rawValue
        let rendered = rgba.withUnsafeMutableBytes { buffer -> Bool in
            guard let context = CGContext(
                data: buffer.baseAddress,
                width: width,
                height: height,
                bitsPerComponent: 8,
                bytesPerRow: bytesPerRow,
                space: colorSpace,
                bitmapInfo: bitmapInfo)
            else { return false }
            context.interpolationQuality = .high
            context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
            return true
        }
        guard rendered else {
            throw FluxError.invalidRequest("failed to render image at \(url.path)")
        }

        let plane = width * height
        var values = [Float](repeating: 0, count: plane * 3)
        for y in 0 ..< height {
            for x in 0 ..< width {
                let pixelOffset = y * bytesPerRow + x * 4
                let outputOffset = y * width + x
                let r = Float(rgba[pixelOffset]) / 255
                let g = Float(rgba[pixelOffset + 1]) / 255
                let b = Float(rgba[pixelOffset + 2]) / 255
                values[outputOffset] = normalize(r, channel: 0, normalization: normalization)
                values[plane + outputOffset] = normalize(g, channel: 1, normalization: normalization)
                values[plane * 2 + outputOffset] = normalize(b, channel: 2, normalization: normalization)
            }
        }
        return values
        #else
        throw FluxError.notImplemented("ImageIO.readRGBValues requires AppKit")
        #endif
    }

    public static func readRGBTensor(
        _ url: URL,
        width: Int,
        height: Int,
        normalization: RGBNormalization
    ) throws -> MLXArray {
        let values = try readRGBValues(url, width: width, height: height, normalization: normalization)
        return MLXArray(values, [1, 3, height, width]).asType(.float32)
    }

    private static func normalize(
        _ value: Float,
        channel: Int,
        normalization: RGBNormalization
    ) -> Float {
        switch normalization {
        case .zeroToOne:
            return value
        case .minusOneToOne:
            return value * 2 - 1
        case .openAIClip:
            let mean: [Float] = [0.48145466, 0.4578275, 0.40821073]
            let std: [Float] = [0.26862954, 0.26130258, 0.27577711]
            return (value - mean[channel]) / std[channel]
        }
    }

    /// Save an image tensor to `dir/<prefix>-<uuid>.png`.
    /// Returns the URL of the written file.
    @MainActor
    public static func writePNG(
        _ tensor: MLXArray,
        outputDir: URL,
        prefix: String = "vmlx"
    ) throws -> URL {
        #if canImport(AppKit)
        guard tensor.ndim == 4 || tensor.ndim == 3 else {
            throw FluxError.invalidRequest(
                "image tensor must be (B,C,H,W) or (C,H,W), got ndim=\(tensor.ndim)")
        }
        // Squeeze batch dim if present.
        let single: MLXArray
        if tensor.ndim == 4 {
            single = tensor[0]
        } else {
            single = tensor
        }
        let channels = single.dim(0)
        let height = single.dim(1)
        let width = single.dim(2)

        guard channels == 3 || channels == 1 else {
            throw FluxError.invalidRequest(
                "image tensor must have 1 or 3 channels, got \(channels)")
        }

        // Clamp to [0, 1], scale to [0, 255], cast to uint8.
        let clamped = clip(single, min: MLXArray(Float(0)), max: MLXArray(Float(1)))
        let scaled = clamped * MLXArray(Float(255))
        let rounded = MLX.round(scaled)
        let asUInt8 = rounded.asType(.uint8)

        // (C, H, W) → (H, W, C) for pixel buffer interpretation.
        let interleaved = asUInt8.transposed(1, 2, 0)
        let bytes = interleaved.asArray(UInt8.self)

        // Build an NSBitmapImageRep and serialize.
        let bitsPerPixel = channels == 3 ? 24 : 8
        let bytesPerRow = width * channels
        guard let rep = NSBitmapImageRep(
            bitmapDataPlanes: nil,
            pixelsWide: width,
            pixelsHigh: height,
            bitsPerSample: 8,
            samplesPerPixel: channels,
            hasAlpha: false,
            isPlanar: false,
            colorSpaceName: channels == 3 ? .calibratedRGB : .calibratedWhite,
            bytesPerRow: bytesPerRow,
            bitsPerPixel: bitsPerPixel
        ) else {
            throw FluxError.invalidRequest("failed to create NSBitmapImageRep")
        }
        // Copy pixel data into the rep.
        if let ptr = rep.bitmapData {
            bytes.withUnsafeBufferPointer { src in
                if let base = src.baseAddress {
                    ptr.update(from: base, count: bytes.count)
                }
            }
        }
        guard let pngData = rep.representation(using: .png, properties: [:]) else {
            throw FluxError.invalidRequest("failed to PNG-encode image")
        }

        try FileManager.default.createDirectory(
            at: outputDir, withIntermediateDirectories: true)
        let filename = "\(prefix)-\(UUID().uuidString).png"
        let url = outputDir.appendingPathComponent(filename)
        try pngData.write(to: url)
        return url
        #else
        throw FluxError.notImplemented("ImageIO.writePNG requires AppKit")
        #endif
    }
}
