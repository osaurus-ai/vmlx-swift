import Foundation
@preconcurrency import MLX
#if canImport(AppKit)
import AppKit
import ImageIO
import UniformTypeIdentifiers
#endif

/// One in-flight preview per request. Tensor work is submitted from the sampling
/// lane; only the evaluated pixel read and PNG encoding run on the worker.
/// Slow encodes are dropped rather than building a queue of stale previews.
public final class ImagePreviewEmitter: @unchecked Sendable {
    private let interval: Int
    public let maxPixel: Int
    private let callback: @Sendable (Data, Int) -> Void
    private let lock = NSLock()
    private var busy = false
    private var closed = false
    private static let worker = DispatchQueue(label: "vmlx.image-preview", qos: .utility)

    public init?(interval: Int?, maxPixel: Int = 256, callback: @escaping @Sendable (Data, Int) -> Void) {
        guard let interval else { return nil }
        self.interval = interval
        self.maxPixel = maxPixel
        self.callback = callback
    }

    /// Called before the terminal event; no preview can be emitted afterward.
    public func close() {
        lock.lock(); closed = true; lock.unlock()
    }

    /// `makeImage` builds a small NCHW RGB graph using the model's actual VAE.
    /// It must not synchronously evaluate intermediate VAE stages.
    public func emit(step: Int, makeImage: () -> MLXArray?) {
        guard interval > 0, step % interval == 0, !PreviewPressure.shared.isPressured else { return }
        lock.lock()
        guard !busy, !closed else { lock.unlock(); return }
        busy = true
        lock.unlock()
        guard let image = makeImage() else { lock.lock(); busy = false; lock.unlock(); return }
        let h = image.dim(2), w = image.dim(3)
        let stride = max(1, Int(ceil(Double(max(h, w)) / Double(maxPixel))))
        let rows = MLXArray(Array(Swift.stride(from: 0, to: h, by: stride)).map(Int32.init))
        let cols = MLXArray(Array(Swift.stride(from: 0, to: w, by: stride)).map(Int32.init))
        let small = take(take(image[0], rows, axis: 1), cols, axis: 2)
        let pixels = MLX.round(clip(small, min: Float(0), max: Float(1)) * Float(255))
            .asType(.uint8).transposed(1, 2, 0)
        asyncEval(pixels)
        let snapshot = PreviewPixels(pixels, width: cols.size, height: rows.size)
        Self.worker.async { [self, snapshot] in
            defer { lock.lock(); busy = false; lock.unlock() }
            guard let bytes = try? snapshot.array.asArrayChecked(UInt8.self),
                  let png = Self.encode(bytes, width: snapshot.width, height: snapshot.height) else { return }
            lock.lock()
            if !closed { callback(png, step) }
            lock.unlock()
        }
    }

    /// Reduce a spatial latent before VAE decode, preserving aspect ratio.
    /// NCHW is the common decoder layout; Qwen 2.1 converts NHWC at its boundary.
    public func reducedLatent(_ latent: MLXArray, scale: Int) -> MLXArray {
        let h = latent.dim(2), w = latent.dim(3)
        let limit = max(1, maxPixel / scale)
        let stride = max(1, Int(ceil(Double(max(h, w)) / Double(limit))))
        let rows = MLXArray(Array(Swift.stride(from: 0, to: h, by: stride)).map(Int32.init))
        let cols = MLXArray(Array(Swift.stride(from: 0, to: w, by: stride)).map(Int32.init))
        return take(take(latent, rows, axis: 2), cols, axis: 3)
    }

    private static func encode(_ bytes: [UInt8], width: Int, height: Int) -> Data? {
        #if canImport(AppKit)
        guard let provider = CGDataProvider(data: Data(bytes) as CFData),
            let image = CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 24,
                bytesPerRow: width * 3, space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: [],
                provider: provider, decode: nil, shouldInterpolate: true, intent: .defaultIntent) else { return nil }
        var current = image
        while true {
            let data = NSMutableData()
            guard let destination = CGImageDestinationCreateWithData(data, UTType.png.identifier as CFString, 1, nil) else { return nil }
            CGImageDestinationAddImage(destination, current, nil)
            guard CGImageDestinationFinalize(destination) else { return nil }
            if data.length < 64 * 1024 { return data as Data }
            let w = max(1, current.width / 2), h = max(1, current.height / 2)
            guard let context = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8,
                bytesPerRow: w * 4, space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { return nil }
            context.interpolationQuality = .medium
            context.draw(current, in: CGRect(x: 0, y: 0, width: w, height: h))
            guard let resized = context.makeImage() else { return nil }
            current = resized
        }
        #else
        return nil
        #endif
    }
}

private final class PreviewPixels: @unchecked Sendable {
    let array: MLXArray
    let width: Int
    let height: Int
    init(_ array: MLXArray, width: Int, height: Int) { self.array = array; self.width = width; self.height = height }
}

private final class PreviewPressure: @unchecked Sendable {
    static let shared = PreviewPressure()
    private let lock = NSLock()
    private var pressured = false
    #if os(macOS)
    private let source: DispatchSourceMemoryPressure
    #endif
    var isPressured: Bool { lock.lock(); defer { lock.unlock() }; return pressured }
    private init() {
        #if os(macOS)
        source = DispatchSource.makeMemoryPressureSource(eventMask: [.normal, .warning, .critical], queue: .global(qos: .utility))
        source.setEventHandler { [weak self] in
            guard let self else { return }
            lock.lock(); pressured = !source.data.contains(.normal); lock.unlock()
        }
        source.resume()
        #endif
    }
}
