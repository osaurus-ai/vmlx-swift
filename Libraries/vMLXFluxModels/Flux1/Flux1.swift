import Foundation
import vMLXFluxKit

// MARK: - Flux1
//
// Original FLUX.1 family: Schnell (4-step), Dev (20-step). Both use the
// dual-encoder (T5 + CLIP) transformer backbone with flow-matching
// sampling. Python source: `mflux.models.flux.variants.txt2img.flux.Flux1`.
//
// This is a SCAFFOLD — every method throws `notImplemented`. The real
// port needs:
//   1. FluxTransformer (DiT-style, ~12B params for Dev, ~2.3B for Schnell)
//   2. T5-XXL text encoder (~5B params)
//   3. CLIP-L text encoder
//   4. Autoencoder (VAE)
//   5. Flow-matching scheduler (Euler step, sigmas from 0→1)
//   6. Weight loader for mlx .safetensors layout
// All of that slots into the method bodies below without touching the
// rest of the package — the protocol contract is stable.

public final class Flux1Schnell: ImageGenerator, @unchecked Sendable {
    public static let _register: Void = {
        ModelRegistry.register(ModelEntry(
            name: "flux1-schnell",
            displayName: "FLUX.1 Schnell",
            kind: .imageGen,
            defaultSteps: 4,
            defaultGuidance: 0.0,   // Schnell doesn't use CFG
            supportsLoRA: true,
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
        AsyncThrowingStream { continuation in
            Task { [weak self] in
                guard let self else { continuation.finish(); return }
                do {
                    guard request.steps > 0 else {
                        throw FluxError.invalidRequest("Flux steps must be greater than zero")
                    }
                    let image = try self.pipeline.generate(
                        prompt: request.prompt,
                        width: request.width, height: request.height,
                        steps: request.steps, seed: request.seed
                    ) { step, total, eta in
                        continuation.yield(.step(step: step, total: total, etaSeconds: eta))
                    }
                    let outURL = try await MainActor.run {
                        try ImageIO.writePNG(image, outputDir: request.outputDir, prefix: "flux1-schnell")
                    }
                    continuation.yield(.completed(url: outURL, seed: request.seed ?? 0))
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
