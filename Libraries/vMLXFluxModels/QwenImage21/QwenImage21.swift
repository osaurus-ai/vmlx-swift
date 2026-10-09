import Foundation
@preconcurrency import MLX
import vMLXFluxKit

// Qwen-Image-2.1 — one bundle serves text-to-image AND reference editing
// (up to 10 ordered reference images). mflux source:
// `mflux.models.qwen21.variants.edit.qwen_image_21_edit.QwenImage21Edit`
// (the save path that keeps the Qwen3-VL vision tower). Canonical name:
// "qwen-image-2.1". Defaults follow mflux: 40 steps, guidance 1.0 (no CFG).
//
// Qwen-Image-2.1-Turbo ("qwen-image-2.1-turbo") is the same architecture with distilled weights and a
// fixed 8-node sigma grid in its model_index.json (`sample_sigmas`); the pipeline samples on that grid
// whenever a bundle defines one. Defaults: 8 steps, guidance 1.0.

public final class QwenImage21: ImageGenerator, ImageEditor, @unchecked Sendable {
    public static let _register: Void = {
        ModelRegistry.register(ModelEntry(
            name: "qwen-image-2.1",
            displayName: "Qwen-Image-2.1",
            kind: .imageGen,
            defaultSteps: 40,
            defaultGuidance: 1.0,
            loader: { path, quant in
                _ = QwenImage21._register
                return try await QwenImage21(modelPath: path, quantize: quant)
            }
        ))
        ModelRegistry.register(ModelEntry(
            name: "qwen-image-2.1-turbo",
            displayName: "Qwen-Image-2.1-Turbo",
            kind: .imageGen,
            defaultSteps: 8,
            defaultGuidance: 1.0,
            loader: { path, quant in
                _ = QwenImage21._register
                return try await QwenImage21(modelPath: path, quantize: quant, requiresFixedSchedule: true)
            }
        ))
    }()

    /// Output file prefix: the Turbo bundle (fixed sampling grid) is named as such.
    private var outputPrefix: String {
        pipeline.sampleSigmas == nil ? "qwen-image-2.1" : "qwen-image-2.1-turbo"
    }

    public let modelPath: URL
    public let quantize: Int?
    private let pipeline: QwenImage21Pipeline

    public init(modelPath: URL, quantize: Int?, requiresFixedSchedule: Bool = false) async throws {
        self.modelPath = modelPath
        self.quantize = quantize
        _ = Self._register
        guard FileManager.default.fileExists(atPath: modelPath.path) else {
            throw FluxError.weightsNotFound(modelPath)
        }
        // Validate scheduling metadata before allocating any model weights.
        let grid = try QwenImage21Schedule.sampleSigmas(modelPath: modelPath)
        if requiresFixedSchedule && grid == nil {
            throw FluxError.localModelIncomplete(modelPath, reasons: ["Turbo requires model_index.json sample_sigmas"])
        }
        let loaded = try WeightLoader.load(from: modelPath)
        try QwenImage21BundleValidator.validate(modelPath, loaded: loaded)
        self.pipeline = try await QwenImage21Pipeline(modelPath: modelPath, loaded: loaded, sampleSigmas: grid)
    }

    public func generate(_ request: ImageGenRequest) -> AsyncThrowingStream<ImageGenEvent, Error> {
        let seed = request.seed ?? UInt64.random(in: 0 ... UInt64(UInt32.max))
        return run(prefix: outputPrefix, outputDir: request.outputDir, seed: seed) { pipeline, progress in
            try pipeline.generate(
                prompt: request.prompt, negativePrompt: request.negativePrompt, references: [],
                width: request.width, height: request.height, steps: request.steps,
                guidance: request.guidance, seed: seed, progress: progress)
        }
    }

    public func edit(_ request: ImageEditRequest) -> AsyncThrowingStream<ImageGenEvent, Error> {
        if request.mask != nil {
            return AsyncThrowingStream { continuation in
                continuation.yield(.failed(message: "Qwen-Image-2.1 masks are not wired yet", hfAuth: false))
                continuation.finish()
            }
        }
        let seed = request.seed ?? UInt64.random(in: 0 ... UInt64(UInt32.max))
        return run(prefix: outputPrefix + "-edit", outputDir: request.outputDir, seed: seed) { pipeline, progress in
            try pipeline.generate(
                prompt: request.prompt, negativePrompt: nil, references: request.sourceImages,
                width: request.width, height: request.height, steps: request.steps,
                guidance: request.guidance, seed: seed, progress: progress)
        }
    }

    private func run(
        prefix: String, outputDir: URL, seed: UInt64,
        _ body: @escaping @Sendable (QwenImage21Pipeline, (Int, Int, Double?) -> Void) throws -> MLXArray
    ) -> AsyncThrowingStream<ImageGenEvent, Error> {
        AsyncThrowingStream { continuation in
            Task { [weak self] in
                guard let self else { continuation.finish(); return }
                do {
                    let image = try body(self.pipeline) { step, total, eta in
                        continuation.yield(.step(step: step, total: total, etaSeconds: eta))
                    }
                    let url = try ImageIO.writePNG(image, outputDir: outputDir, prefix: prefix)
                    continuation.yield(.completed(url: url, seed: seed))
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

enum QwenImage21BundleValidator {
    private static let requiredWeights: [String: [String]] = [
        "transformer": [
            "img_in.weight",
            "txt_in.in_layer.weight",
            "transformer_blocks.0.attn.to_q.weight",
            "transformer_blocks.31.img_mlp.out.weight",
            "proj_out.weight",
        ],
        "text_encoder": [
            "language_model.embed_tokens.weight",
            "language_model.layers.0.self_attn.q_proj.weight",
            "language_model.layers.35.mlp.down_proj.weight",
            "visual.patch_embed.proj.weight",
            "visual.blocks.26.attn.qkv.weight",
            "visual.merger.linear_fc2.weight",
        ],
        "vae": [
            "encoder.conv_in.weight",
            "quant_conv.weight",
            "post_quant_conv.weight",
            "decoder.conv_in.weight",
            "decoder.conv_out.weight",
        ],
    ]

    static func validate(_ modelPath: URL, loaded: LoadedWeights) throws {
        var reasons: [String] = []
        let fm = FileManager.default
        let hasTokenizer = ["processor/tokenizer.json", "tokenizer/tokenizer.json"].contains {
            fm.fileExists(atPath: modelPath.appendingPathComponent($0).path)
        }
        if !hasTokenizer { reasons.append("missing processor/tokenizer.json") }
        for file in ["transformer/config.json", "text_encoder/config.json", "vae/config.json"]
        where !fm.fileExists(atPath: modelPath.appendingPathComponent(file).path) {
            reasons.append("missing \(file)")
        }
        for component in requiredWeights.keys.sorted() {
            guard let weights = loaded.componentWeights[component], !weights.isEmpty else {
                reasons.append("missing \(component) component")
                continue
            }
            for key in requiredWeights[component, default: []] where weights[key] == nil {
                reasons.append("missing \(component) weight \(key)")
            }
        }
        if !reasons.isEmpty {
            throw FluxError.localModelIncomplete(modelPath, reasons: reasons)
        }
    }
}
