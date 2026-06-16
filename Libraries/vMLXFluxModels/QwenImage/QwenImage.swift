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
        try QwenImageEditBundleValidator.validate(modelPath)
    }

    public func edit(_ request: ImageEditRequest) -> AsyncThrowingStream<ImageGenEvent, Error> {
        AsyncThrowingStream { continuation in
            Task { [weak self] in
                guard let self else { continuation.finish(); return }
                do {
                    if request.mask != nil {
                        throw FluxError.notImplemented("QwenImageEdit masks are not wired yet")
                    }
                    let pipeline = try QwenImageEditPipeline(modelPath: self.modelPath)
                    let image = try await pipeline.edit(
                        prompt: request.prompt,
                        sourceImage: request.sourceImage,
                        width: request.width,
                        height: request.height,
                        steps: request.steps,
                        guidance: request.guidance,
                        seed: request.seed
                    ) { step, total, eta in
                        continuation.yield(.step(step: step, total: total, etaSeconds: eta))
                    }
                    let outURL = try await MainActor.run {
                        try ImageIO.writePNG(
                            image,
                            outputDir: request.outputDir,
                            prefix: "qwen-image-edit")
                    }
                    continuation.yield(.completed(url: outURL, seed: request.seed ?? 0))
                    continuation.finish()
                } catch {
                    let message = String(describing: error)
                    continuation.yield(.failed(
                        message: message,
                        hfAuth: message.contains("401") || message.contains("403")))
                    continuation.finish()
                }
            }
        }
    }
}

private enum QwenImageEditBundleValidator {
    private static let requiredFiles = [
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
    ]

    private static let requiredWeightsByComponent: [String: [String]] = [
        "text_encoder": [
            "encoder.embed_tokens.weight",
            "encoder.layers.0.self_attn.q_proj.weight",
            "encoder.norm.weight",
            "encoder.visual.patch_embed.proj.weight",
            "encoder.visual.blocks.0.attn.qkv.weight",
            "encoder.visual.blocks.31.attn.qkv.weight",
            "encoder.visual.merger.mlp_1.weight",
        ],
        "transformer": [
            "img_in.weight",
            "txt_in.weight",
            "time_text_embed.timestep_embedder.linear_1.weight",
            "transformer_blocks.0.attn.add_q_proj.weight",
            "transformer_blocks.59.img_ff.mlp_out.weight",
            "proj_out.weight",
        ],
        "vae": [
            "encoder.conv_in.conv3d.weight",
            "encoder.down_blocks.0.resnets.0.conv1.conv3d.weight",
            "quant_conv.conv3d.weight",
            "post_quant_conv.conv3d.weight",
            "decoder.conv_in.conv3d.weight",
            "decoder.conv_out.conv3d.weight",
        ],
    ]

    static func validate(_ modelPath: URL) throws {
        let fm = FileManager.default
        var reasons: [String] = []
        for relativePath in requiredFiles {
            let url = modelPath.appendingPathComponent(relativePath)
            if !fm.fileExists(atPath: url.path) {
                reasons.append("missing \(relativePath)")
            }
        }

        for component in requiredWeightsByComponent.keys.sorted() {
            let keys = try WeightLoader.indexedWeightKeys(in: modelPath, component: component)
            if keys.isEmpty {
                reasons.append("missing \(component) safetensors index")
                continue
            }
            for key in requiredWeightsByComponent[component, default: []] where !keys.contains(key) {
                reasons.append("missing \(component) weight \(key)")
            }
        }

        if !reasons.isEmpty {
            throw FluxError.localModelIncomplete(modelPath, reasons: reasons)
        }
    }
}
