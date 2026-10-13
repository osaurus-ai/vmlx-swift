// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// MiniCPM-V 4.7 35B-A3B JANGH2: the Qwen3.5-MoE text model's routed experts
/// (`language_model.model.layers.L.mlp.switch_mlp.{gate,up,down}_proj.tq2_{packed,scales}`, 40 layers x 256 experts,
/// top-8; gate/up 2-bit, down 3-bit, hadamard32-rotated odd-cubic codebook) are JANGH banks. Everything else
/// (attention, GDN, shared expert, router, embeddings, head, vision) is ordinary affine and loads through the generic
/// path. The banks run on the same JANGH codebook kernels as Qwen3.8 Flash-Next / Allosaurus JANGH.
struct MiniCPMV47JANGHPreparation {
    let banks: JANGHModelPreparation

    /// Declared by the explicit contract (`mode: jangtq2`), never by a folder name.
    static func loadIfDeclared(directory: URL, configurationData: Data) throws -> Self? {
        guard
            let root = try? JSONSerialization.jsonObject(with: configurationData) as? [String: Any],
            root["model_type"] as? String == "minicpmv4_7",
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
        else { throw Qwen35JANGHError.invalid("minicpmv4_7 JANGH: missing MoE text_config dimensions") }
        let step = text["decoder_sparse_step"] as? Int ?? 1
        let sparse = Set((0 ..< layers).filter { step <= 1 || ($0 + 1) % step == 0 })
        return Self(
            banks: try JANGHModelPreparation(
                directory: directory, configuration: configurationData, sidecar: sidecar,
                layout: .init(
                    modelType: "minicpmv4_7", hiddenSize: hidden, intermediateSize: inter,
                    expertCount: experts, sparseLayers: sparse, routesPerToken: topK,
                    modulePrefix: "language_model.")))
    }

    /// Build the Qwen3.5-MoE with MiniCPM's vision tower (see `minicpmv4_7` in VLMModelFactory) from the ordinary
    /// configuration, then replace every routed `switch_mlp` with its JANGH bank BEFORE weights load. The vision
    /// weights are ordinary bf16 and load through the generic path.
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
                throw Qwen35JANGHError.invalid("minicpmv4_7 JANGH module \(path) is not in the model")
            }
            updates.append((path, bank))
        }
        model.update(modules: ModuleChildren.unflattened(updates))
        model.customDenseTensorNames = banks.excludedTensorNames
        return model
    }
}
