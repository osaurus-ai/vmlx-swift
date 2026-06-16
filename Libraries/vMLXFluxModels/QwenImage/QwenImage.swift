import Foundation
import vMLXFluxKit

// Qwen-Image (gen) + Qwen-Image-Edit — Alibaba's image model family.
// Python source: `mflux.models.qwen.variants.{txt2img.qwen_image, edit.qwen_image_edit}`.

public final class QwenImage: ImageGenerator, @unchecked Sendable {
    public static let _register: Void = {
        ModelRegistry.register(ModelEntry(
            name: "qwen-image",
            displayName: "Qwen-Image",
            kind: .imageGen,
            defaultSteps: 30,
            defaultGuidance: 4.0,
            loader: { path, quant in
                _ = QwenImage._register
                return try await QwenImage(modelPath: path, quantize: quant)
            }
        ))
    }()

    public let modelPath: URL
    public let quantize: Int?
    private let pipeline: QwenImagePipeline

    public init(modelPath: URL, quantize: Int?) async throws {
        self.modelPath = modelPath
        self.quantize = quantize
        _ = Self._register
        guard FileManager.default.fileExists(atPath: modelPath.path) else {
            throw FluxError.weightsNotFound(modelPath)
        }
        self.pipeline = try await QwenImagePipeline(modelPath: modelPath)
    }

    public func generate(_ request: ImageGenRequest) -> AsyncThrowingStream<ImageGenEvent, Error> {
        AsyncThrowingStream { continuation in
            Task { [weak self] in
                guard let self else { continuation.finish(); return }
                do {
                    guard request.steps > 0 else {
                        throw FluxError.invalidRequest("Qwen steps must be greater than zero")
                    }
                    let image = try self.pipeline.generate(
                        prompt: request.prompt, negativePrompt: request.negativePrompt,
                        width: request.width, height: request.height, steps: request.steps,
                        guidance: request.guidance, seed: request.seed
                    ) { step, total, eta in
                        continuation.yield(.step(step: step, total: total, etaSeconds: eta))
                    }
                    let outURL = try await MainActor.run {
                        try ImageIO.writePNG(image, outputDir: request.outputDir, prefix: "qwen-image")
                    }
                    continuation.yield(.completed(url: outURL, seed: request.seed ?? 0))
                    continuation.finish()
                } catch {
                    let message = String(describing: error)
                    continuation.yield(.failed(message: message, hfAuth: message.contains("401") || message.contains("403")))
                    continuation.finish()
                }
            }
        }
    }
}

public final class QwenImageEdit: ImageEditor, @unchecked Sendable {
    public static let _register: Void = {
        ModelRegistry.register(ModelEntry(
            name: "qwen-image-edit",
            displayName: "Qwen-Image-Edit",
            kind: .imageEdit,
            defaultSteps: 30,
            defaultGuidance: 4.0,
            loader: { path, quant in
                _ = QwenImageEdit._register
                return try QwenImageEdit(modelPath: path, quantize: quant)
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
                "QwenImageEdit.edit — port from mflux/models/qwen/variants/edit/qwen_image_edit.py"))
        }
    }
}
