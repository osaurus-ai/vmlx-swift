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

public final class QwenImage21: ImageGenerator, ImageEditor, FluxStageModel, @unchecked Sendable {
    public static let _register: Void = {
        ModelRegistry.register(ModelEntry(
            name: "qwen-image-2.1",
            displayName: "Qwen-Image-2.1",
            kind: .imageGen,
            defaultSteps: 40,
            defaultGuidance: 1.0,
            supportsLoRA: true,
            limits: { try requestLimits(modelPath: $0) },
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
            supportsLoRA: true,
            limits: { try requestLimits(modelPath: $0, requiresFixedSchedule: true) },
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

    public var stageSpatialScale: Int { 16 }
    public var stageChannels: Int { pipeline.cfg.zDim }
    public func stageEncodePrompt(_ prompt: String, negativePrompt: String?) throws -> FluxStageConditioning {
        pipeline.encodeStagePrompt(prompt, negativePrompt: negativePrompt)
    }
    public func stageEncodeImage(_ image: URL, width: Int, height: Int) throws -> MLXArray {
        try pipeline.encodeStageImage(image, width: width, height: height)
    }
    public func stageDecode(_ latent: MLXArray) throws -> MLXArray { pipeline.decodeStageLatent(latent) }
    public func stageSample(_ conditioning: FluxStageConditioning, latent: MLXArray,
                            width: Int, height: Int, options: FluxStageSampling,
                            interrupt: FluxInterrupt?, progress: (Int, Int, Double?) -> Void)
        throws -> (MLXArray, EffectiveImageParameters) {
        let request = ImageGenRequest(prompt: "", width: width, height: height, steps: options.steps,
            guidance: options.guidance, seed: options.seed, outputDir: modelPath,
            sampler: options.sampler, scheduler: options.scheduler, shift: options.shift)
        try Self.limits(fixedSteps: pipeline.sampleSigmas?.count).validate(request)
        let prepared = try pipeline.stagePrepared(conditioning, width: width, height: height,
            steps: options.steps, guidance: options.guidance)
        var effective: EffectiveImageParameters?
        let result = try pipeline.sampleLatent(prepared, seed: options.seed, interrupt: interrupt,
            sampler: options.sampler, scheduler: options.scheduler, shift: options.shift,
            initialLatent: latent, denoise: options.denoise, resolved: { effective = $0 }, progress: progress)
        guard let effective else { throw FluxError.invalidRequest("sampling did not resolve effective parameters") }
        return (result, effective)
    }

    public let modelPath: URL
    public let quantize: Int?
    private let pipeline: QwenImage21Pipeline
    /// Diagnostic count of actual text-encoder executions (positive and CFG).
    public var promptEncodingCount: Int { pipeline.promptEncodingCount }

    /// No weights are loaded by this query. Host admission limits are separate.
    public static func requestLimits(modelPath: URL, requiresFixedSchedule: Bool = false) throws -> FluxModelLimits {
        let grid = try QwenImage21Schedule.sampleSigmas(modelPath: modelPath)
        if requiresFixedSchedule && grid == nil {
            throw FluxError.localModelIncomplete(modelPath, reasons: ["Turbo requires model_index.json sample_sigmas"])
        }
        return limits(fixedSteps: grid?.count)
    }

    private static func limits(fixedSteps: Int?) -> FluxModelLimits {
        FluxModelLimits(
            dimensionMultiple: 32, minDimension: 32, minimumSteps: 2, fixedSteps: fixedSteps,
            supportsNegativePrompt: true, supportsEditNegativePrompt: false,
            supportsStrength: false, supportsMask: false, maxSourceImages: 10, supportedFormats: [.png],
            maximumImages: 8, maximumParallelImages: 1,
            samplers: fixedSteps == nil ? ["euler", "heun"] : ["euler"],
            schedulers: fixedSteps == nil ? ["qwen-image-2.1", "linear"] : ["bundle-sigmas"],
            defaultSampler: "euler", defaultScheduler: fixedSteps == nil ? "qwen-image-2.1" : "bundle-sigmas",
            supportsShift: fixedSteps == nil)
    }

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
        do { try Self.limits(fixedSteps: pipeline.sampleSigmas?.count).validate(request) }
        catch { return AsyncThrowingStream { $0.finish(throwing: error) } }
        let interrupt = request.interrupt ?? FluxInterrupt()
        return AsyncThrowingStream { continuation in
            continuation.onTermination = { if case .cancelled = $0 { interrupt.cancel() } }
            Task { [self] in
                do {
                    try interrupt.check()
                    let plan = try ImageBatchPlan(request, maximumImages: 8, maximumParallelImages: 1)
                    let before = pipeline.promptEncodingCount
                    let prepared = try pipeline.prepare(
                        prompt: request.prompt, negativePrompt: request.negativePrompt, references: [],
                        width: request.width, height: request.height, steps: request.steps,
                        guidance: request.guidance)
                    let started = Date()
                    for (index, seed) in plan.seeds.enumerated() {
                        try interrupt.check()
                        let preview = ImagePreviewEmitter(interval: request.previewInterval, maxPixel: request.previewMaxPixel) { data, step in
                            continuation.yield(.preview(pngData: data, step: step))
                        }
                        defer { preview?.close() }
                        var effective: EffectiveImageParameters?
                        let image = try pipeline.sample(prepared, seed: seed, interrupt: interrupt,
                            sampler: request.sampler, scheduler: request.scheduler, shift: request.shift, preview: preview,
                            resolved: { effective = $0 }, progress: { step, total, eta in
                                let completed = index * total + step
                                let all = total * plan.seeds.count
                                let remaining = Date().timeIntervalSince(started) / Double(completed) * Double(all - completed)
                                continuation.yield(.step(step: completed, total: all, etaSeconds: remaining))
                            })
                        try interrupt.check()
                        preview?.close()
                        let url = try ImageIO.writePNG(image, outputDir: request.outputDir, prefix: outputPrefix)
                        if interrupt.isCancelled {
                            try FileManager.default.removeItem(at: url)
                            throw CancellationError()
                        }
                        if plan.seeds.count == 1 {
                            continuation.yield(.completed(url: url, seed: seed, effectiveParameters: effective))
                        } else {
                            continuation.yield(.imageCompleted(url: url, seed: seed, index: index, effectiveParameters: effective))
                        }
                    }
                    print("[FluxBatch] family=\(outputPrefix) images=\(plan.seeds.count) parallel=\(plan.batchSize) prompt_encodes=\(pipeline.promptEncodingCount - before)")
                    continuation.finish()
                } catch is CancellationError {
                    continuation.yield(.cancelled)
                    continuation.finish()
                } catch {
                    let message = String(describing: error)
                    continuation.yield(.failed(message: message,
                        hfAuth: message.contains("401") || message.contains("403")))
                    continuation.finish()
                }
            }
        }
    }

    public func edit(_ request: ImageEditRequest) -> AsyncThrowingStream<ImageGenEvent, Error> {
        do { try Self.limits(fixedSteps: pipeline.sampleSigmas?.count).validate(request) }
        catch { return AsyncThrowingStream { $0.finish(throwing: error) } }
        let seed = request.seed ?? UInt64.random(in: 0 ... UInt64(UInt32.max))
        let interrupt = request.interrupt ?? FluxInterrupt()
        return run(prefix: outputPrefix + "-edit", outputDir: request.outputDir, seed: seed, interrupt: interrupt, previewInterval: request.previewInterval, previewMaxPixel: request.previewMaxPixel) { pipeline, progress, resolved, preview in
            try pipeline.generate(
                prompt: request.prompt, negativePrompt: nil, references: request.sourceImages,
                width: request.width, height: request.height, steps: request.steps,
                guidance: request.guidance, seed: seed, sampler: request.sampler,
                scheduler: request.scheduler, shift: request.shift, interrupt: interrupt, preview: preview, resolved: resolved, progress: progress)
        }
    }

    private func run(
        prefix: String, outputDir: URL, seed: UInt64, interrupt: FluxInterrupt, previewInterval: Int? = nil, previewMaxPixel: Int = 256,
        _ body: @escaping @Sendable (QwenImage21Pipeline, (Int, Int, Double?) -> Void, (EffectiveImageParameters) -> Void, ImagePreviewEmitter?) throws -> MLXArray
    ) -> AsyncThrowingStream<ImageGenEvent, Error> {
        AsyncThrowingStream { continuation in
            continuation.onTermination = { if case .cancelled = $0 { interrupt.cancel() } }
            Task { [weak self] in
                guard let self else { continuation.finish(); return }
                do {
                    try interrupt.check()
                    let preview = ImagePreviewEmitter(interval: previewInterval, maxPixel: previewMaxPixel) { data, step in
                        continuation.yield(.preview(pngData: data, step: step))
                    }
                    defer { preview?.close() }
                    var parameters: EffectiveImageParameters?
                    let image = try body(self.pipeline, { step, total, eta in
                        continuation.yield(.step(step: step, total: total, etaSeconds: eta))
                    }, { parameters = $0 }, preview)
                    try interrupt.check()
                    preview?.close()
                    let url = try ImageIO.writePNG(image, outputDir: outputDir, prefix: prefix)
                    if interrupt.isCancelled {
                        try FileManager.default.removeItem(at: url)
                        throw CancellationError()
                    }
                    continuation.yield(.completed(url: url, seed: seed, effectiveParameters: parameters))
                    continuation.finish()
                } catch is CancellationError {
                    continuation.yield(.cancelled)
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
