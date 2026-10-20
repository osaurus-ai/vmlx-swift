import Foundation
@preconcurrency import MLX
import vMLXFluxKit

public final class Flux3Klein: ImageGenerator, ImageEditor, @unchecked Sendable {
    public static let _register: Void = {
        ModelRegistry.register(Flux3Registration.entry { try await Flux3Klein(modelPath: $0, quantize: $1) })
    }()
    public let modelPath: URL
    public let quantize: Int?
    private let pipeline: Flux3Pipeline

    public init(modelPath: URL, quantize: Int?) async throws {
        let bundle = try Flux3Bundle(modelPath: modelPath)
        self.modelPath = modelPath
        // Packed bundle bits are authoritative; a caller's quantize hint must not
        // reinterpret or requantize an already-quantized checkpoint.
        self.quantize = bundle.stamp.quantization.bits
        pipeline = try await Flux3Pipeline(modelPath: modelPath, bundle: bundle)
    }

    /// Metadata-only estimate geometry, using the same EXIF-aware first-reference plan as editing.
    public static func editGeometry(modelPath: URL, sources: [URL], width: Int?, height: Int?) throws
        -> (width: Int, height: Int, sourcePixels: Int) {
        let bundle = try Flux3Bundle(modelPath: modelPath)
        try bundle.limits.validateEdit(width: nil, height: nil, steps: 1, guidance: 1, format: .png,
            sourceCount: sources.count, hasMask: false, strength: nil)
        let plans = try sources.map { source in
            let size = try Flux3ImageReader.dimensions(source)
            return try sources.count == 1 ? Flux3ReferencePlan.single(size, limits: bundle.stamp.limits)
                : Flux3ReferencePlan.multiple(size, limits: bundle.stamp.limits)
        }
        let output = plans[0].output(width: width, height: height, single: sources.count == 1)
        try bundle.limits.validateEdit(width: output.canvas.width, height: output.canvas.height, steps: 1,
            guidance: 1, format: .png, sourceCount: sources.count, hasMask: false, strength: nil)
        return (output.canvas.width, output.canvas.height, plans.reduce(0) { $0 + $1.canvas.width * $1.canvas.height })
    }

    public func generate(_ request: ImageGenRequest) -> AsyncThrowingStream<ImageGenEvent, Error> {
        do { try pipeline.bundle.limits.validate(request) }
        catch { return AsyncThrowingStream { $0.finish(throwing: error) } }
        if let negative = request.negativePrompt, !negative.isEmpty {
            print("[Flux3] negative_prompt ignored: distilled checkpoint has no unconditional branch")
        }
        let interrupt = request.interrupt ?? FluxInterrupt()
        let seed = request.seeds?.first ?? request.seed ?? UInt64.random(in: 0...UInt64(UInt32.max))
        return run(outputDir: request.outputDir, seed: seed, interrupt: interrupt,
            previewInterval: request.previewInterval, previewMaxPixel: request.previewMaxPixel) { pipeline, preview, progress in
            try pipeline.sample(prompt: request.prompt, width: request.width, height: request.height,
                steps: request.steps, seed: seed, references: nil, preview: preview, progress: progress)
        }
    }

    public func edit(_ request: ImageEditRequest) -> AsyncThrowingStream<ImageGenEvent, Error> {
        do {
            let limits = pipeline.bundle.limits
            try limits.validatePreview(interval: request.previewInterval, maxPixel: request.previewMaxPixel)
            try limits.validateSampling(sampler: request.sampler, scheduler: request.scheduler, shift: request.shift)
            // Validate the resolved reference canvas below. A replayed fitted crop
            // may itself have odd dimensions but samples on its padded 16x canvas.
            try limits.validateEdit(width: nil, height: nil, steps: request.steps, guidance: request.guidance,
                format: request.outputFormat, sourceCount: request.sourceImages.count,
                hasMask: request.mask != nil, strength: request.strength)
        } catch { return AsyncThrowingStream { $0.finish(throwing: error) } }
        let interrupt = request.interrupt ?? FluxInterrupt()
        let seed = request.seed ?? UInt64.random(in: 0...UInt64(UInt32.max))
        return run(outputDir: request.outputDir, seed: seed, interrupt: interrupt,
            previewInterval: request.previewInterval, previewMaxPixel: request.previewMaxPixel) { pipeline, preview, progress in
            let references = try pipeline.references(request.sourceImages, width: request.width, height: request.height)
            return try pipeline.sample(prompt: request.prompt, width: references.width, height: references.height,
                steps: request.steps, seed: seed, references: references, preview: preview, progress: progress)
        }
    }

    private func run(outputDir: URL, seed: UInt64, interrupt: FluxInterrupt, previewInterval: Int?, previewMaxPixel: Int,
        body: @Sendable @escaping (Flux3Pipeline, ImagePreviewEmitter?, (Int, Int, Double?) -> Void) throws -> (MLXArray, EffectiveImageParameters))
        -> AsyncThrowingStream<ImageGenEvent, Error> {
        AsyncThrowingStream { continuation in
            continuation.onTermination = { if case .cancelled = $0 { interrupt.cancel() } }
            Task { [self] in
                do {
                    try FluxInterruptScope.$current.withValue(interrupt) {
                        try interrupt.check()
                        continuation.yield(.step(step: 0, total: 0, etaSeconds: nil))
                        let preview = ImagePreviewEmitter(interval: previewInterval, maxPixel: previewMaxPixel) { data, step in
                            continuation.yield(.preview(pngData: data, step: step))
                        }
                        defer { preview?.close() }
                        let started = Date()
                        let (image, effective) = try withError { _ in
                            try body(pipeline, preview) { step, total, eta in
                                continuation.yield(.step(step: step, total: total, etaSeconds: eta))
                            }
                        }
                        try interrupt.check()
                        preview?.close()
                        let url = try ImageIO.writePNG(image, outputDir: outputDir, prefix: "flux3-image-klein")
                        if interrupt.isCancelled { try? FileManager.default.removeItem(at: url); throw CancellationError() }
                        let elapsed = Date().timeIntervalSince(started)
                        print("[Flux3] operation_seconds=\(elapsed) images_per_second=\(1 / elapsed) includes_model_load=false")
                        continuation.yield(.completed(url: url, seed: seed, effectiveParameters: effective))
                    }
                    continuation.finish()
                } catch is CancellationError { continuation.yield(.cancelled); continuation.finish() }
                catch { continuation.finish(throwing: error) }
            }
        }
    }
}
