import Foundation
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
// NOTE: ideogram-4 uses **fp8 quantization** (mflux fp8_linear) for the
// transformer — a DIFFERENT quant path than the MLX group-quant (weight/
// scales/biases) that flux/qwen/z-image use via MFluxStore. MFluxStore now has
// fp8 `weight_scale` Linear support; STATUS: scaffold — generate throws
// notImplemented until the Qwen3 encoder, DiT, unconditional transformer, and
// VAE execution path are ported.

public final class Ideogram4: ImageGenerator, @unchecked Sendable {
    public static let _register: Void = {
        ModelRegistry.register(ModelEntry(
            name: "ideogram",
            displayName: "Ideogram 4",
            kind: .imageGen,
            defaultSteps: 28,
            defaultGuidance: 3.5,
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
            continuation.finish(throwing: FluxError.notImplemented(
                "Ideogram4.generate — port from mflux/models/ideogram4 "
                    + "(Qwen3 encoder + 34-layer fp8 DiT + unconditional transformer + VAE)."))
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
            "input_proj.weight_scale",
            "llm_cond_proj.weight",
            "layers.0.attention.qkv.weight",
            "layers.33.feed_forward.w3.weight",
            "final_layer.linear.weight",
        ],
        "unconditional_transformer": [
            "input_proj.weight",
            "input_proj.weight_scale",
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
        }

        if !reasons.isEmpty {
            throw FluxError.localModelIncomplete(modelPath, reasons: reasons)
        }
    }
}
