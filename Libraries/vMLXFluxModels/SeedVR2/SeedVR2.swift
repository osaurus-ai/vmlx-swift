import Foundation
import vMLXFluxKit
import vMLXSeedVR2

/// Native one-step SeedVR2 3B restoration using a self-contained local export.
public final class SeedVR2: ImageUpscaler, @unchecked Sendable {
    public static let _register: Void = {
        ModelRegistry.register(
            ModelEntry(
                name: "seedvr2", displayName: "SeedVR2 Upscaler", kind: .imageUpscale,
                defaultSteps: 1, defaultGuidance: 1,
                limits: { path in
                    _ = try SeedVR2BundleMetadata(directory: path)
                    return FluxModelLimits(
                        dimensionMultiple: 2, minDimension: 2,
                        maxPixels: SeedVR2BundleMetadata.maximumOutputPixels,
                        minimumSteps: 1, maximumSteps: 1, fixedSteps: 1,
                        supportsNegativePrompt: false, supportsEditNegativePrompt: false,
                        supportsStrength: false, supportsMask: false, maxSourceImages: 1,
                        supportedFormats: [.png])
                },
                loader: { path, quant in try SeedVR2(modelPath: path, quantize: quant) }
            ))
    }()

    public let modelPath: URL
    public let quantize: Int?
    // A model instance owns mutable MLX modules. Concurrent callers queue instead
    // of evaluating or unloading the same module from different tasks.
    private let queue = DispatchQueue(label: "vmlx.seedvr2.producer")
    private let pipeline: SeedVR2Pipeline

    public init(modelPath: URL, quantize: Int?) throws {
        self.modelPath = modelPath
        let bundle = try SeedVR2Bundle(directory: modelPath)
        if let quantize, quantize != bundle.bits {
            throw FluxError.invalidRequest("SeedVR2 quantization must match the exported bundle")
        }
        self.quantize = bundle.bits
        pipeline = try SeedVR2Pipeline(bundle: bundle)
        _ = Self._register
    }

    public func upscale(_ request: UpscaleRequest) -> AsyncThrowingStream<ImageGenEvent, Error> {
        let interrupt = request.interrupt ?? FluxInterrupt()
        return AsyncThrowingStream { continuation in
            continuation.onTermination = { reason in
                if case .cancelled = reason { interrupt.cancel() }
            }
            queue.async { [self] in
                autoreleasepool {
                    do {
                        try interrupt.check()
                        guard request.outputFormat == .png else {
                            throw FluxError.invalidField(
                                field: "outputFormat", value: request.outputFormat.rawValue,
                                allowed: "png")
                        }
                        // This is a distilled one-step model, irrespective of a legacy UI step count.
                        let seed = request.seed ?? UInt64.random(in: 0 ... UInt64.max)
                        let (image, width, height) = try pipeline.upscale(
                            request, seed: seed, checkpoint: interrupt.check
                        ) {
                            continuation.yield(.step(step: 1, total: 1, etaSeconds: nil))
                        }
                        try interrupt.check()
                        let url = try ImageIO.writePNG(
                            image, outputDir: request.outputDir, prefix: "seedvr2", atomic: true)
                        do { try interrupt.check() } catch {
                            try? FileManager.default.removeItem(at: url)
                            throw error
                        }
                        continuation.yield(
                            .completed(
                                url: url, seed: seed,
                                effectiveParameters: .init(
                                    width: width, height: height, steps: 1, guidance: 1, seed: seed,
                                    sampler: "euler", scheduler: "seedvr2_euler",
                                    scale: request.scale)))
                        continuation.finish()
                    } catch is CancellationError {
                        continuation.yield(.cancelled)
                        continuation.finish()
                    } catch { continuation.finish(throwing: error) }
                }
            }
        }
    }
}
