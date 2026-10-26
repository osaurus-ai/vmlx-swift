import Foundation
import MLXLMCommon

/// DeepSeek-V4.1 JANGHT bundles: config.json carries a `jangt` header and a per-module `quantization` plan keyed by
/// CHECKPOINT path -- routed experts `layers.L.ffn.experts.{gate,up,down}_proj` as JANGT trellis (`mode: jangt`), the
/// release FP8 projections as MLX `mxfp8` (attention, indexer wq_b, shared experts, engram wkv), embed / head and the
/// vision / aligner Linears affine 8-bit g64, and the dSpark draft's `mtp.S.*` modules (loaded by the draft, not here).
///
/// Admission goes through the JANGH preparation (strict partition + mapped trellis banks); the routed banks become
/// `JANGTRoutedExpertLayer`s held by the V4.1 MoE. The ordinary plan the generic loader reads is rewritten from checkpoint
/// paths to this runtime's module paths (the same mapping as `DeepseekV4Model.sanitize`), so every mxfp8 / affine
/// module is quantized from its explicit entry instead of the shape walk's guess (mxfp8 has no biases and would otherwise
/// be taken as affine).
struct DeepseekV41JANGHPreparation {
    enum PreparationError: Error, CustomStringConvertible {
        case invalid(String)
        var description: String {
            switch self { case .invalid(let message): return message }
        }
    }

    let banks: JANGHModelPreparation
    let ordinaryConfiguration: Data
    let activationLimit: Float?

    static func loadIfDeclared(directory: URL, configurationData: Data) throws -> DeepseekV41JANGHPreparation? {
        guard var root = try? JSONSerialization.jsonObject(with: configurationData) as? [String: Any],
            root["model_type"] as? String == "deepseek_v41", let header = root["jangt"] as? [String: Any]
        else { return nil }
        // The dSpark draft's banks (`mtp.S.*`) are loaded by the draft; the backbone partition never sees them.
        for key in ["quantization", "quantization_config"] {
            guard var plan = root[key] as? [String: Any] else { continue }
            for name in plan.keys where name.hasPrefix("mtp.") { plan.removeValue(forKey: name) }
            root[key] = plan
        }
        let backbone = try JSONSerialization.data(withJSONObject: root, options: [.sortedKeys])
        // config.json is the only format owner of a JANGHT bundle; jang_config.json is chat / runtime metadata.
        guard JANGHModelPreparation.declaresCustomFormat(configuration: backbone, sidecar: nil) else { return nil }
        let text = root["text_config"] as? [String: Any] ?? root
        guard let hidden = text["hidden_size"] as? Int, let inter = text["moe_intermediate_size"] as? Int,
            let experts = text["n_routed_experts"] as? Int, let layers = text["num_hidden_layers"] as? Int,
            let topK = text["num_experts_per_tok"] as? Int
        else {
            throw PreparationError.invalid("deepseek_v41 JANGHT: incomplete text_config geometry")
        }
        let prepared = try JANGHModelPreparation(
            directory: directory, configuration: backbone, sidecar: nil,
            layout: .init(modelType: "deepseek_v41", hiddenSize: hidden, intermediateSize: inter,
                          expertCount: experts, sparseLayers: Set(0 ..< layers), routesPerToken: topK,
                          parentTemplate: "layers.{L}.ffn.experts"))
        guard var ordinary = try JSONSerialization.jsonObject(with: prepared.ordinaryConfiguration) as? [String: Any]
        else { throw PreparationError.invalid("deepseek_v41 JANGHT: ordinary configuration") }
        for key in ["quantization", "quantization_config"] {
            guard let plan = ordinary[key] as? [String: Any] else { continue }
            var mapped: [String: Any] = [:]
            for (name, value) in plan {
                let path = Self.modulePath(name)
                guard mapped[path] == nil else {
                    throw PreparationError.invalid("deepseek_v41 JANGHT: \(name) collides at \(path)")
                }
                mapped[path] = value
            }
            ordinary[key] = mapped
        }
        let limit = (header["swiglu_limit"] as? Double) ?? (text["swiglu_limit"] as? Double)
        return Self(banks: prepared,
                    ordinaryConfiguration: try JSONSerialization.data(withJSONObject: ordinary, options: [.sortedKeys]),
                    activationLimit: limit.flatMap { $0 > 0 ? Float($0) : nil })
    }

    /// Checkpoint module path -> this runtime's module path (DeepseekV4Model.sanitize). Plan scalars
    /// (`group_size`, `bits`, `mode`) and unknown names pass through unchanged.
    static func modulePath(_ name: String) -> String {
        if name == "embed" { return "model.embed_tokens" }
        if name == "head" { return "lm_head" }
        if name.hasPrefix("vision.") || name.hasPrefix("aligner.") { return "v41_vision." + name }
        let parts = name.split(separator: ".", omittingEmptySubsequences: false).map(String.init)
        guard parts.count >= 3, parts[0] == "layers", Int(parts[1]) != nil else { return name }
        let rest = Array(parts[2...])
        var mapped: [String]
        switch rest[0] {
        case "attn":
            mapped = rest.count > 1 && rest[1] == "indexer"
                ? ["self_attn", "v41_indexer"] + rest[2...] : ["self_attn"] + rest[1...]
        case "ffn" where rest.count > 1 && rest[1] == "shared_experts":
            let role = ["w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"]
            mapped = ["mlp", "shared_experts"] + rest[2...].map { role[$0] ?? $0 }
        case "ffn" where rest.count > 1 && rest[1] == "experts":
            mapped = ["mlp", "switch_mlp"] + rest[2...]
        case "ffn":
            mapped = ["mlp"] + rest[1...]
        default:
            mapped = rest
        }
        return "model.layers.\(parts[1])." + mapped.joined(separator: ".")
    }

    func construct(configurationData: Data, requesting: Set<ModelRuntimeRequestModality>?) throws -> DeepseekV4Model {
        var config = try LLMTypeRegistry.decodeDeepseekV4Configuration(data: configurationData)
        if let requesting, !requesting.contains(.vision) { config.vision = nil }
        let routed = try banks.makeRoutedExperts(activationLimit: activationLimit)
        return DeepseekV4Model(config, routedExperts: routed, excludedTensorNames: banks.excludedTensorNames)
    }
}
