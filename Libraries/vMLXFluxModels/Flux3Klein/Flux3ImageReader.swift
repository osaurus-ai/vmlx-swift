import Foundation
import CoreGraphics
import ImageIO
import vMLXFluxKit

/// ImageIO decodes pixels on the CPU; EXIF orientation and all resampling are
/// explicit host operations. No CoreImage context, Metal device or model is used.
enum Flux3ImageReader {
    static func dimensions(_ url: URL) throws -> Flux3ReferencePlan.Size {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
              let w = properties[kCGImagePropertyPixelWidth] as? Int,
              let h = properties[kCGImagePropertyPixelHeight] as? Int else {
            throw FluxError.invalidRequest("cannot read reference dimensions")
        }
        let orientation = (properties[kCGImagePropertyOrientation] as? NSNumber)?.intValue ?? 1
        return (5...8).contains(orientation) ? .init(h, w) : .init(w, h)
    }
    static func read(_ url: URL) throws -> Flux3Pixels {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw FluxError.invalidRequest("cannot read reference image at \(url.path)")
        }
        let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any]
        let orientation = (properties?[kCGImagePropertyOrientation] as? NSNumber)?.intValue ?? 1
        let w = image.width, h = image.height
        var rgb = [UInt8](repeating: 0, count: w * h * 3)
        // Preserve RGB8/RGBA8 source channels (including alpha-zero RGB) when
        // ImageIO exposes them directly, matching PIL convert("RGB").
        if image.bitsPerComponent == 8, image.colorSpace?.model == .rgb,
           image.bitmapInfo.intersection(.byteOrderMask).isEmpty || image.bitmapInfo.contains(.byteOrder32Big),
           let data = image.dataProvider?.data, CFDataGetLength(data) >= image.bytesPerRow * h,
           let bytes = CFDataGetBytePtr(data) {
            let n = image.bitsPerPixel / 8
            let alpha = image.alphaInfo
            if n == 3 || n == 4 && (alpha == .last || alpha == .first || alpha == .noneSkipFirst || alpha == .noneSkipLast) {
                let offset = alpha == .first || alpha == .noneSkipFirst ? 1 : 0
                for y in 0..<h { for x in 0..<w { for c in 0..<3 {
                    rgb[(y * w + x) * 3 + c] = bytes[y * image.bytesPerRow + x * n + offset + c]
                } } }
                return try Flux3Pixels(width: w, height: h, rgb: rgb).oriented(orientation)
            }
        }
        var rgba = [UInt8](repeating: 0, count: w * h * 4)
        let rendered = rgba.withUnsafeMutableBytes { storage -> Bool in
            guard let context = CGContext(data: storage.baseAddress, width: w, height: h, bitsPerComponent: 8,
                bytesPerRow: w * 4, space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue) else { return false }
            context.setBlendMode(.copy)
            context.interpolationQuality = .none
            context.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
            return true
        }
        guard rendered else { throw FluxError.invalidRequest("cannot decode reference RGB pixels") }
        for i in 0..<w * h { for c in 0..<3 {
            let a = Int(rgba[i * 4 + 3]), value = Int(rgba[i * 4 + c])
            rgb[i * 3 + c] = a == 0 ? 0 : UInt8(clamping: (value * 255 + a / 2) / a)
        } }
        return try Flux3Pixels(width: w, height: h, rgb: rgb).oriented(orientation)
    }
}
