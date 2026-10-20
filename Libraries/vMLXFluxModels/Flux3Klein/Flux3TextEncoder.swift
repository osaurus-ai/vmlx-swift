import Foundation
@preconcurrency import MLX
import MLXNN
import VMLXTokenizers
import vMLXFluxKit

/// Qwen3-VL-4B language layers only. The last feature tap is BEFORE final norm.
/// The language block is shared with QwenImage21; no vision tower is constructed.
final class Flux3TextEncoder {
    private let embed: MFluxEmbedding
    private let layers: [Q21TextLayer]
    private let tokenizer: any VMLXTokenizers.Tokenizer
    private let template: String
    private let padID: Int
    private let features: Flux3Bundle.Stamp.Features
    private let cfg: QwenImage21Config
    private let invFreq: MLXArray

    init(store: MFluxStore, modelPath: URL, bundle: Flux3Bundle) async throws {
        features = bundle.stamp.textFeatures
        let folder = modelPath.appendingPathComponent("text_encoder")
        let data = try Data(contentsOf: folder.appendingPathComponent("config.json"))
        let raw = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        guard let text = raw?["text_config"] as? [String: Any] ?? raw else {
            throw FluxError.invalidRequest("FLUX 3 missing language config")
        }
        func int(_ name: String) throws -> Int {
            guard let value = text[name] as? Int else { throw FluxError.invalidRequest("missing text_encoder \(name)") }
            return value
        }
        var config = QwenImage21Config()
        config.hidden = try int("hidden_size"); config.intermediate = try int("intermediate_size")
        config.textLayers = try int("num_hidden_layers"); config.textHeads = try int("num_attention_heads")
        config.kvHeads = try int("num_key_value_heads"); config.textHeadDim = try int("head_dim")
        guard config.textLayers == 32, config.hidden == 2560, config.textHeads == 32,
              config.kvHeads == 8, config.textHeadDim == 128,
              let theta = text["rope_theta"] as? NSNumber, let eps = text["rms_norm_eps"] as? NSNumber else {
            throw FluxError.invalidRequest("unsupported FLUX 3 Qwen3-VL language config")
        }
        config.ropeTheta = theta.floatValue; config.rmsEps = eps.floatValue
        cfg = config
        embed = try store.embedding("text_encoder", "language_model.embed_tokens", dimensions: config.hidden)
        layers = try (0..<config.textLayers).map { try Q21TextLayer(store: store, index: $0, cfg: config, normWeightInFloat32: true) }
        // mflux saves rotary_emb.inv_freq with the language module. Use that
        // buffer just as the reference does, rather than recomputing saved state.
        invFreq = try store.tensor("text_encoder", "language_model.rotary_emb.inv_freq").asType(.float32)
        tokenizer = try await AutoTokenizer.from(modelFolder: folder, strict: false)
        template = try String(contentsOf: folder.appendingPathComponent("chat_template.jinja"), encoding: .utf8)
        let tokenConfig = try JSONSerialization.jsonObject(with: Data(contentsOf: folder.appendingPathComponent("tokenizer_config.json"))) as? [String: Any]
        guard let padToken = tokenConfig?["pad_token"] as? String, let id = tokenizer.convertTokenToId(padToken) else {
            throw FluxError.invalidRequest("FLUX 3 tokenizer must declare a valid pad_token")
        }
        padID = id
    }

    func encode(_ prompt: String) throws -> (tokens: MLXArray, ids: [[Float]]) {
        let ids = try tokenizer.applyChatTemplate(messages: [["role": "user", "content": prompt]],
            chatTemplate: .literal(template), addGenerationPrompt: true, truncation: false, maxLength: nil, tools: nil)
        let plan = Flux3PromptPlan(tokens: ids, padID: padID, maxTokens: features.maxTokensMflux, padMultiple: features.padMultiple)
        let s = plan.ids.count
        guard s > 0 else { throw FluxError.invalidRequest("FLUX 3 template produced no tokens") }
        var hidden = embed(MLXArray(plan.ids).reshaped([1, s]))
        let angles = MLXArray(plan.positions).asType(.float32).reshaped([1, s, 1]) * invFreq.reshaped([1, 1, -1])
        let phases = concatenated([angles, angles], axis: -1)
        let cs = MLX.cos(phases).asType(hidden.dtype), sn = MLX.sin(phases).asType(hidden.dtype)
        // Padding masks keys, while padded query positions still produce feature rows.
        var mask = [Float](repeating: 0, count: s * s)
        for i in 0..<s {
            for j in 0..<s where j > i || plan.mask[j] == 0 { mask[i * s + j] = -.infinity }
        }
        let attentionMask = MLXArray(mask, [1, 1, s, s]).asType(hidden.dtype)
        var taps: [MLXArray] = []
        for (index, layer) in layers.enumerated() {
            try FluxInterruptScope.check()
            hidden = layer(hidden, cos: cs, sin: sn, mask: attentionMask)
            eval(hidden)
            if features.hiddenStateLayers.contains(index + 1) { taps.append(hidden) }
        }
        return (concatenated(taps, axis: -1).asType(.bfloat16), Flux3PromptPlan.textIDs(length: s))
    }
}
