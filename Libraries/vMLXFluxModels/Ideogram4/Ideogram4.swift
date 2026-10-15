import Foundation
@preconcurrency import MLX
import vMLXFluxKit

// Ideogram 4 — open-weights text-to-image (strong typography/text rendering).
// mflux source: `mflux.models.ideogram4.variants.txt2img.ideogram4.Ideogram4`.
// mflux-compatible weights: `ideogram-ai/ideogram-4-fp8` (fp8) or
// `ideogram-ai/ideogram-4-nf4` (4-bit). Canonical name: "ideogram".
//
// Arch (for the port): Qwen3 text encoder (reuse the Qwen LM pattern from
// QwenImageNative), a 34-layer DiT transformer (emb_dim 4608, 18 heads,
// intermediate 12288, in_channels 128, llm_features_dim 4096*13 = multi-layer
// Qwen3 hidden states, rope_theta 5e6, adaLN_dim 512), and a VAE.
//
// NOTE: ideogram-4 uses fp8 or bitsandbytes NF4 quantization for transformer
// linears — different paths than the MLX group-quant (weight/scales/biases)
// that flux/qwen/z-image use via MFluxStore. Typography has current fp8 live
// proof; strict object-icon prompts have current fp8 and NF4 live proof.

public final class Ideogram4: ImageGenerator, FluxAdapterPreparing, @unchecked Sendable {
    public static let _register: Void = {
        ModelRegistry.register(ModelEntry(
            name: "ideogram",
            displayName: "Ideogram 4",
            kind: .imageGen,
            defaultSteps: 20,
            defaultGuidance: 7,
            supportsLoRA: true,
            limits: { _ in NativeImageRequestContracts.ideogram },
            loader: { path, quant in
                _ = Ideogram4._register
                return try Ideogram4(modelPath: path, quantize: quant)
            }
        ))
    }()

    public let modelPath: URL
    public let quantize: Int?
    public let loadedWeights: LoadedWeights
    private let store: MFluxStore
    private var adapterPipeline: Ideogram4Pipeline?

    public func prepareAdapterLayers() async throws {
        if adapterPipeline == nil {
            adapterPipeline = try await Ideogram4Pipeline(modelPath: modelPath, loadedWeights: loadedWeights)
        }
    }

    public init(modelPath: URL, quantize: Int?) throws {
        self.modelPath = modelPath
        self.quantize = quantize
        _ = Self._register
        guard FileManager.default.fileExists(atPath: modelPath.path) else {
            throw FluxError.weightsNotFound(modelPath)
        }
        let loaded = try WeightLoader.load(from: modelPath)
        try Ideogram4BundleValidator.validate(modelPath, loaded: loaded)
        self.loadedWeights = loaded
        self.store = MFluxStore(loaded)
    }

    public func generate(_ request: ImageGenRequest) -> AsyncThrowingStream<ImageGenEvent, Error> {
        do { try NativeImageRequestContracts.ideogram.validate(request) }
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
                        throw FluxError.invalidRequest("Ideogram steps must be greater than zero")
                    }
                    let pipeline: Ideogram4Pipeline
                    if let adapterPipeline { pipeline = adapterPipeline }
                    else { pipeline = try await Ideogram4Pipeline(modelPath: self.modelPath, loadedWeights: self.loadedWeights) }
                    let image = try pipeline.generate(
                        prompt: request.prompt,
                        width: request.width,
                        height: request.height,
                        steps: request.steps,
                        guidance: request.guidance,
                        seed: seed, preview: preview
                    ) { step, total, eta in
                        continuation.yield(.step(step: step, total: total, etaSeconds: eta))
                    }
                    preview?.close()
                    let outURL = try ImageIO.writePNG(image, outputDir: request.outputDir, prefix: "ideogram")
                    continuation.yield(.completed(url: outURL, seed: seed,
                        effectiveParameters: try NativeImageRequestContracts.parameters(url: outURL, steps: request.steps, guidance: request.guidance, seed: seed, scheduler: "ideogram-logit-normal")))
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

enum Ideogram4BundleValidator {
    private static let requiredFiles = [
        "tokenizer/tokenizer.json",
    ]

    private static let requiredWeightsByComponent: [String: [String]] = [
        "text_encoder": [
            "language_model.embed_tokens.weight",
            "language_model.layers.0.self_attn.q_proj.weight",
            "language_model.layers.35.mlp.down_proj.weight",
            "language_model.norm.weight",
        ],
        "transformer": [
            "input_proj.weight",
            "llm_cond_proj.weight",
            "layers.0.attention.qkv.weight",
            "layers.33.feed_forward.w3.weight",
            "final_layer.linear.weight",
        ],
        "unconditional_transformer": [
            "input_proj.weight",
            "llm_cond_proj.weight",
            "layers.0.attention.qkv.weight",
            "layers.33.feed_forward.w3.weight",
            "final_layer.linear.weight",
        ],
        "vae": [
            "decoder.conv_in.weight",
            "decoder.conv_out.weight",
            "post_quant_conv.weight",
        ],
    ]

    private static let requiredQuantizedLinearsByComponent: [String: [String]] = [
        "transformer": [
            "input_proj",
            "llm_cond_proj",
            "layers.0.attention.qkv",
            "layers.33.feed_forward.w3",
            "final_layer.linear",
        ],
        "unconditional_transformer": [
            "input_proj",
            "llm_cond_proj",
            "layers.0.attention.qkv",
            "layers.33.feed_forward.w3",
            "final_layer.linear",
        ],
    ]

    static func validate(_ modelPath: URL, loaded: LoadedWeights) throws {
        let fm = FileManager.default
        var reasons: [String] = []
        for relativePath in requiredFiles {
            let url = modelPath.appendingPathComponent(relativePath)
            if !fm.fileExists(atPath: url.path) {
                reasons.append("missing \(relativePath)")
            }
        }

        // Resolve keys through MFluxStore so diffusers (`language_model.*`) and
        // mflux-saved (prefix-free) text-encoder layouts both validate.
        let store = MFluxStore(loaded)
        for component in requiredWeightsByComponent.keys.sorted() {
            guard let weights = loaded.componentWeights[component], !weights.isEmpty else {
                reasons.append("missing \(component) component")
                continue
            }
            for key in requiredWeightsByComponent[component, default: []] where !store.hasKey(component, key) {
                reasons.append("missing \(component) weight \(key)")
            }
            for prefix in requiredQuantizedLinearsByComponent[component, default: []] {
                if hasFp8Linear(prefix, in: weights) || hasNF4Linear(prefix, in: weights)
                    || hasGroupQuantLinear(prefix, in: weights)
                {
                    continue
                }
                reasons.append("missing \(component) quant metadata for \(prefix) (fp8 weight_scale, bitsandbytes NF4 absmax/quant_map/state, or MLX group scales)")
            }
        }

        if !reasons.isEmpty {
            throw FluxError.localModelIncomplete(modelPath, reasons: reasons)
        }
    }

    private static func hasFp8Linear(_ prefix: String, in weights: [String: MLXArray]) -> Bool {
        weights["\(prefix).weight_scale"] != nil
    }

    private static func hasNF4Linear(_ prefix: String, in weights: [String: MLXArray]) -> Bool {
        weights["\(prefix).weight.absmax"] != nil &&
            weights["\(prefix).weight.quant_map"] != nil &&
            weights["\(prefix).weight.quant_state.bitsandbytes__nf4"] != nil
    }

    /// mflux `-q 4/8` saves: packed uint32 `weight` + `scales` (+ `biases`).
    private static func hasGroupQuantLinear(_ prefix: String, in weights: [String: MLXArray]) -> Bool {
        weights["\(prefix).weight"] != nil && weights["\(prefix).scales"] != nil
    }
}
