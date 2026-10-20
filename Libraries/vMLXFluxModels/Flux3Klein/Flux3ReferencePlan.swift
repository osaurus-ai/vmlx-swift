import Foundation
import vMLXFluxKit

/// Pixel geometry from mflux d587aac flux3_reference_images.py. RGB/EXIF decoding
/// and Lanczos resampling belong to the image I/O layer, outside this pure plan.
public struct Flux3ReferencePlan: Equatable, Sendable {
    public struct Size: Equatable, Sendable {
        public let width, height: Int
        public init(_ width: Int, _ height: Int) { self.width = width; self.height = height }
    }
    public struct Box: Equatable, Sendable {
        public let left, top, right, bottom: Int
        public var size: Size { Size(right - left, bottom - top) }
    }
    public let resized: Size
    public let canvas: Size
    public let crop: Box?
    public let symmetricPadding: Bool

    public static func single(_ size: Size, limits: Flux3Bundle.Stamp.Limits) throws -> Self {
        guard size.width > 0, size.height > 0 else { throw FluxError.invalidRequest("empty reference image") }
        let w = Double(size.width), h = Double(size.height)
        var scale = min(1, sqrt(Double(limits.singleReferenceMaxPixels) / (w * h)), Double(limits.maxSide) / max(w, h))
        while true {
            let fit = Size(max(1, Int((w * scale).rounded(.toNearestOrEven))), max(1, Int((h * scale).rounded(.toNearestOrEven))))
            func side(_ value: Int) -> Int { min(limits.maxSide, max(limits.minSide, ((value + limits.sizeMultiple - 1) / limits.sizeMultiple) * limits.sizeMultiple)) }
            let canvas = Size(side(fit.width), side(fit.height))
            if canvas.width * canvas.height <= limits.singleReferenceMaxPixels {
                let left = (canvas.width - fit.width) / 2, top = (canvas.height - fit.height) / 2
                return Self(resized: fit, canvas: canvas,
                    crop: fit == canvas ? nil : Box(left: left, top: top, right: left + fit.width, bottom: top + fit.height),
                    symmetricPadding: min(fit.width, fit.height) <= limits.sizeMultiple)
            }
            scale *= 0.995
        }
    }

    public static func multiple(_ size: Size, limits: Flux3Bundle.Stamp.Limits) throws -> Self {
        guard size.width > 0, size.height > 0 else { throw FluxError.invalidRequest("empty reference image") }
        var w = size.width, h = size.height
        if Double(w) * Double(h) > Double(limits.multiReferenceMaxPixelsEach) {
            let scale = sqrt(Double(limits.multiReferenceMaxPixelsEach) / (Double(w) * Double(h)))
            w = Int(Double(w) * scale); h = Int(Double(h) * scale)
        }
        let cw = w / limits.sizeMultiple * limits.sizeMultiple, ch = h / limits.sizeMultiple * limits.sizeMultiple
        guard cw > 0, ch > 0 else { throw FluxError.invalidRequest("reference needs at least 16 pixels per side after resize") }
        let left = (w - cw) / 2, top = (h - ch) / 2
        return Self(resized: Size(w, h), canvas: Size(cw, ch),
            crop: Box(left: left, top: top, right: left + cw, bottom: top + ch), symmetricPadding: false)
    }

    /// Single references can return the original fitted (non-multiple-of-16) crop.
    /// The denoiser still samples on the padded, valid canvas.
    public func output(width: Int?, height: Int?, single: Bool) -> (canvas: Size, crop: Box?) {
        if single, let crop, (width == nil && height == nil || width == crop.size.width && height == crop.size.height) {
            return (canvas, crop)
        }
        return (Size(width ?? canvas.width, height ?? canvas.height), nil)
    }
}
