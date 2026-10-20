// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLXLMCommon
import MLXNN

/// Mixed routed JANGH + affine Qwen3.5-MoE bundles (including Ornith 122B).
/// Only backbone expert banks are custom; vision, shared experts and the optional
/// native MTP head retain their ordinary per-module quantization plans.
struct Qwen35MoEJANGHPreparation {
    let banks: JANGHModelPreparation

    /// Declared by the explicit contract (`mode: jangtq2`), never by a folder name.
    static func loadIfDeclared(directory: URL, configurationData: Data) throws -> Self? {
        guard
            let root = try? JSONSerialization.jsonObject(with: configurationData) as? [String: Any],
            root["model_type"] as? String == "qwen3_5_moe",
            let text = root["text_config"] as? [String: Any],
            (text["num_experts"] as? Int ?? 0) > 0
        else { return nil }
        let sidecarURL = directory.appendingPathComponent("jang_config.json")
        let sidecar =
            FileManager.default.fileExists(atPath: sidecarURL.path)
            ? try Data(contentsOf: sidecarURL) : nil
        guard JANGHModelPreparation.declaresCustomFormat(configuration: configurationData, sidecar: sidecar)
        else { return nil }
        guard let hidden = text["hidden_size"] as? Int, let inter = text["moe_intermediate_size"] as? Int,
            let experts = text["num_experts"] as? Int, let layers = text["num_hidden_layers"] as? Int,
            let topK = text["num_experts_per_tok"] as? Int,
            hidden > 0, inter > 0, experts > 0, layers > 0, topK > 0, topK <= experts
        else { throw Qwen35JANGHError.invalid("qwen3_5_moe JANGH: missing MoE text_config dimensions") }
        // Qwen35MoE constructs a routed block at every decoder layer. Do not
        // admit a sparse schedule that the instantiated model does not implement.
        guard (text["decoder_sparse_step"] as? Int ?? 1) == 1,
            (text["mlp_only_layers"] as? [Int] ?? []).isEmpty
        else { throw Qwen35JANGHError.invalid("qwen3_5_moe JANGH: unsupported sparse-layer schedule") }
        let sparse = Set(0 ..< layers)
        return Self(
            banks: try JANGHModelPreparation(
                directory: directory, configuration: configurationData, sidecar: sidecar,
                layout: .init(
                    modelType: "qwen3_5_moe", hiddenSize: hidden, intermediateSize: inter,
                    expertCount: experts, sparseLayers: sparse, routesPerToken: topK,
                    modulePrefix: "language_model.")))
    }

    /// Replace the admitted routed banks before ordinary weight loading. The
    /// factory supplies its MTP-enabled or MTP-scrubbed configuration unchanged.
    func construct(configurationData: Data, requesting: Set<ModelRuntimeRequestModality>?) throws -> Qwen35MoE {
        let config = try JSONDecoder.json5().decode(Qwen35Configuration.self, from: configurationData)
        let model = try Qwen35MoE(config, requesting: requesting)
        let routed = try banks.makeRoutedExperts(activationLimit: nil)
        let modules = Set(model.leafModules().flattened().map(\.0))
            .union(model.children().flattened().map(\.0))
        var updates: [(String, Module)] = []
        for (layer, bank) in routed {
            let path = "language_model.model.layers.\(layer).mlp.switch_mlp"
            guard modules.contains(where: { $0 == path || $0.hasPrefix(path + ".") }) else {
                throw Qwen35JANGHError.invalid("qwen3_5_moe JANGH module \(path) is not in the model")
            }
            updates.append((path, bank))
        }
        model.update(modules: ModuleChildren.unflattened(updates))
        model.customDenseTensorNames = banks.excludedTensorNames
        return model
    }
}
