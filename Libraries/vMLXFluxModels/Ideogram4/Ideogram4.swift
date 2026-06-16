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

public final class Ideogram4: ImageGenerator, @unchecked Sendable {
    public static let _register: Void = {
        ModelRegistry.register(ModelEntry(
            name: "ideogram",
            displayName: "Ideogram 4",
            kind: .imageGen,
            defaultSteps: 20,
            defaultGuidance: 7,
            supportsLoRA: true,
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
        AsyncThrowingStream { continuation in
            Task { [weak self] in
                guard let self else { continuation.finish(); return }
                do {
                    guard request.steps > 0 else {
                        throw FluxError.invalidRequest("Ideogram steps must be greater than zero")
                    }
                    let pipeline = try await Ideogram4Pipeline(
                        modelPath: self.modelPath,
                        loadedWeights: self.loadedWeights)
                    let image = try pipeline.generate(
                        prompt: request.prompt,
                        width: request.width,
                        height: request.height,
                        steps: request.steps,
                        guidance: request.guidance,
                        seed: request.seed
                    ) { step, total, eta in
                        continuation.yield(.step(step: step, total: total, etaSeconds: eta))
                    }
                    let outURL = try await MainActor.run {
                        try ImageIO.writePNG(image, outputDir: request.outputDir, prefix: "ideogram")
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

        for component in requiredWeightsByComponent.keys.sorted() {
            guard let weights = loaded.componentWeights[component], !weights.isEmpty else {
                reasons.append("missing \(component) component")
                continue
            }
            for key in requiredWeightsByComponent[component, default: []] where weights[key] == nil {
                reasons.append("missing \(component) weight \(key)")
            }
            for prefix in requiredQuantizedLinearsByComponent[component, default: []] {
                if hasFp8Linear(prefix, in: weights) || hasNF4Linear(prefix, in: weights) {
                    continue
                }
                reasons.append("missing \(component) quant metadata for \(prefix) (fp8 weight_scale or bitsandbytes NF4 absmax/quant_map/state)")
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
}
