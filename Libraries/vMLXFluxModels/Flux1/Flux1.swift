import Foundation
import vMLXFluxKit

// MARK: - Flux1
//
// Original FLUX.1 family: Schnell (4-step), Dev (20-step). Both use the
// dual-encoder (T5 + CLIP) transformer backbone with flow-matching
// sampling. Python source: `mflux.models.flux.variants.txt2img.flux.Flux1`.
//
// Schnell uses the native pipeline below. Other variants retain explicit
// unsupported errors until their pipelines are implemented.

public final class Flux1Schnell: ImageGenerator, @unchecked Sendable {
    public static let _register: Void = {
        ModelRegistry.register(ModelEntry(
            name: "flux1-schnell",
            displayName: "FLUX.1 Schnell",
            kind: .imageGen,
            defaultSteps: 4,
            defaultGuidance: 0.0,   // Schnell doesn't use CFG
            supportsLoRA: true,
            limits: { _ in NativeImageRequestContracts.fluxSchnell },
            loader: { path, quant in
                _ = Flux1Schnell._register  // ensure registered
                return try await Flux1Schnell(modelPath: path, quantize: quant)
            }
        ))
    }()

    public let modelPath: URL
    public let quantize: Int?
    private let pipeline: FluxSchnellPipeline

    public init(modelPath: URL, quantize: Int?) async throws {
        self.modelPath = modelPath
        self.quantize = quantize
        _ = Self._register
        guard FileManager.default.fileExists(atPath: modelPath.path) else {
            throw FluxError.weightsNotFound(modelPath)
        }
        self.pipeline = try await FluxSchnellPipeline(modelPath: modelPath)
    }

    public func generate(_ request: ImageGenRequest) -> AsyncThrowingStream<ImageGenEvent, Error> {
        do { try NativeImageRequestContracts.fluxSchnell.validate(request) }
        catch { return AsyncThrowingStream { $0.finish(throwing: error) } }
        return AsyncThrowingStream { continuation in
            Task { [weak self] in
                guard let self else { continuation.finish(); return }
                do {
                    let seed = request.seed ?? UInt64.random(in: 0...UInt64(UInt32.max))
                    let preview = ImagePreviewEmitter(interval: request.previewInterval, maxPixel: request.previewMaxPixel) { data, step in
                        continuation.yield(.preview(pngData: data, step: step))
                    }
                    defer { preview?.close() }
                    guard request.steps > 0 else {
                        throw FluxError.invalidRequest("Flux steps must be greater than zero")
                    }
                    let image = try self.pipeline.generate(
                        prompt: request.prompt,
                        width: request.width, height: request.height,
                        steps: request.steps, seed: seed, preview: preview
                    ) { step, total, eta in
                        continuation.yield(.step(step: step, total: total, etaSeconds: eta))
                    }
                    preview?.close()
                    try FluxInterruptScope.check()
                    let outURL = try ImageIO.writePNG(image, outputDir: request.outputDir, prefix: "flux1-schnell")
                    continuation.yield(.completed(url: outURL, seed: seed,
                        effectiveParameters: try NativeImageRequestContracts.parameters(url: outURL, steps: request.steps, guidance: 0, seed: seed, scheduler: "flux-flow", shift: FlowMatchEulerScheduler.computeShift(imageSeqLen: (request.width / 16) * (request.height / 16)))))
                    continuation.finish()
                } catch is CancellationError {
                    continuation.yield(.cancelled)
                    continuation.finish()
                } catch {
                    let message = String(describing: error)
                    let hfAuth = message.contains("401") || message.contains("403")
                    continuation.yield(.failed(message: message, hfAuth: hfAuth))
                    continuation.finish()
                }
            }
        }
    }
}

public final class Flux1Dev: ImageGenerator, @unchecked Sendable {
    public static let _register: Void = {
        ModelRegistry.register(ModelEntry(
            name: "flux1-dev",
            displayName: "FLUX.1 Dev",
            kind: .imageGen,
            defaultSteps: 20,
            defaultGuidance: 3.5,
            supportsLoRA: true,
            loader: { path, quant in
                _ = Flux1Dev._register
                return try Flux1Dev(modelPath: path, quantize: quant)
            }
        ))
    }()

    public let modelPath: URL
    public let quantize: Int?

    public init(modelPath: URL, quantize: Int?) throws {
        self.modelPath = modelPath
        self.quantize = quantize
        _ = Self._register
        guard FileManager.default.fileExists(atPath: modelPath.path) else {
            throw FluxError.weightsNotFound(modelPath)
        }
    }

    public func generate(_ request: ImageGenRequest) -> AsyncThrowingStream<ImageGenEvent, Error> {
        AsyncThrowingStream { continuation in
            continuation.finish(throwing: FluxError.notImplemented(
                "Flux1Dev.generate — port from mflux/models/flux/variants/txt2img/flux.py"))
        }
    }
}

// MARK: - Flux1Kontext (edit via text prompt, no mask required)

public final class Flux1Kontext: ImageEditor, @unchecked Sendable {
    public static let _register: Void = {
        ModelRegistry.register(ModelEntry(
            name: "flux1-kontext",
            displayName: "FLUX.1 Kontext",
            kind: .imageEdit,
            defaultSteps: 28,
            defaultGuidance: 2.5,
            loader: { path, quant in
                _ = Flux1Kontext._register
                return try Flux1Kontext(modelPath: path, quantize: quant)
            }
        ))
    }()

    public let modelPath: URL
    public let quantize: Int?

    public init(modelPath: URL, quantize: Int?) throws {
        self.modelPath = modelPath
        self.quantize = quantize
        _ = Self._register
        guard FileManager.default.fileExists(atPath: modelPath.path) else {
            throw FluxError.weightsNotFound(modelPath)
        }
    }

    public func edit(_ request: ImageEditRequest) -> AsyncThrowingStream<ImageGenEvent, Error> {
        AsyncThrowingStream { continuation in
            continuation.finish(throwing: FluxError.notImplemented(
                "Flux1Kontext.edit — port from mflux/models/flux/variants/kontext/flux_kontext.py"))
        }
    }
}

// MARK: - Flux1Fill (inpaint / outpaint via mask)

public final class Flux1Fill: ImageEditor, @unchecked Sendable {
    public static let _register: Void = {
        ModelRegistry.register(ModelEntry(
            name: "flux1-fill",
            displayName: "FLUX.1 Fill",
            kind: .imageEdit,
            defaultSteps: 28,
            defaultGuidance: 30.0,
            loader: { path, quant in
                _ = Flux1Fill._register
                return try Flux1Fill(modelPath: path, quantize: quant)
            }
        ))
    }()

    public let modelPath: URL
    public let quantize: Int?

    public init(modelPath: URL, quantize: Int?) throws {
        self.modelPath = modelPath
        self.quantize = quantize
        _ = Self._register
        guard FileManager.default.fileExists(atPath: modelPath.path) else {
            throw FluxError.weightsNotFound(modelPath)
        }
    }

    public func edit(_ request: ImageEditRequest) -> AsyncThrowingStream<ImageGenEvent, Error> {
        AsyncThrowingStream { continuation in
            continuation.finish(throwing: FluxError.notImplemented(
                "Flux1Fill.edit — port from mflux/models/flux/variants/fill/flux_fill.py"))
        }
    }
}
