// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLXLMCommon

/// MiniMax-M3 / M3.1 JANGHT bundles (`minimax_m3_vl`; config.json carries a top-level `jangt` header).
///
/// The bundle contract is the one the vMLX Python runtime loads (`minimax_m3.Model` + `jangt.switch.install_jangt`):
/// - `quantization` is keyed by LIVE module path (tensor names are live paths too, no `language_model.` prefix):
///   `model.embed_tokens` / `lm_head` affine, every attention / indexer / dense-MLP / shared-expert linear either MLX
///   `mxfp8` or affine 4 / 6 / 8 (mixed per module), routers `model.layers.L.mlp.gate` unquantized F32;
/// - routed experts `model.layers.L.mlp.switch_mlp.{gate,up,down}_proj` are `{"mode": "jangt", "kbits": K}` trellis
///   banks (`jt_packed`, `jt_scale`, `jt_su`, `jt_ocol`, `jt_wout`, plus `jt_cs` on down), or JANGH codebook banks;
/// - the vision tower keeps its source names (dropped by `MiniMaxM3Model.sanitize`); `dspark/` is the drafter's.
///
/// config.json is the only format owner; jang_config.json (`format: jangtq2`, `format_name: JANGHT`) is runtime
/// metadata. Admission is the shared JANGH preparation (strict partition, mapped banks, no dense expert placeholders);
/// the ordinary per-module plan, with the routed banks marked unquantized, goes to the generic loader.
///
/// The routed activation is M3's SwiGLU-OAI from the text config (`swiglu_alpha`, `swiglu_limit`), never the
/// bundle's: `h = min(g, L) * sigmoid(alpha * min(g, L)) * (clip(u, -L, L) + 1)`.
struct MiniMaxM3JANGHPreparation {
    enum PreparationError: Error, CustomStringConvertible {
        case invalid(String)
        var description: String {
            switch self { case .invalid(let message): return message }
        }
    }

    static let modelTypes: Set<String> = ["minimax_m3_vl", "minimax_m3"]

    let configuration: MiniMaxM3Configuration
    let banks: JANGHModelPreparation

    var ordinaryConfiguration: Data { banks.ordinaryConfiguration }

    /// nil unless config.json is a MiniMax-M3 config with a `jangt` header and custom routed projections.
    static func loadIfDeclared(directory: URL, configurationData: Data) throws -> Self? {
        guard let root = try? JSONSerialization.jsonObject(with: configurationData) as? [String: Any],
            let modelType = root["model_type"] as? String, modelTypes.contains(modelType),
            root["jangt"] is [String: Any],
            JANGHModelPreparation.declaresCustomFormat(configuration: configurationData, sidecar: nil)
        else { return nil }
        let config = try JSONDecoder.json5().decode(MiniMaxM3Configuration.self, from: configurationData)
        let moeLayers = Set((0 ..< config.hiddenLayers).filter(config.isMoE))
        guard !moeLayers.isEmpty else {
            throw PreparationError.invalid("MiniMax-M3 JANGHT: the config declares no MoE layer")
        }
        let banks = try JANGHModelPreparation(
            directory: directory, configuration: configurationData, sidecar: nil,
            layout: .init(
                modelType: modelType, hiddenSize: config.hiddenSize, intermediateSize: config.intermediateSize,
                expertCount: config.numLocalExperts, sparseLayers: moeLayers,
                routesPerToken: config.numExpertsPerTok))
        // The ordinary plan must decode before any bank is mapped (JANGHModelPreparation contract).
        _ = try JSONDecoder.json5().decode(BaseConfiguration.self, from: banks.ordinaryConfiguration)
        return Self(configuration: config, banks: banks)
    }

    func construct(requesting: Set<ModelRuntimeRequestModality>?) throws -> MiniMaxM3Model {
        // Text model only, as the non-JANGHT dispatch: a vision lane is refused instead of served text-only.
        _ = try Set.resolveForConstruction(requested: requesting, constructible: [.text])
        let routed = try banks.makeRoutedExperts(
            activationLimit: configuration.swigluLimit > 0 ? configuration.swigluLimit : nil,
            activation: .oai(alpha: configuration.swigluAlpha))
        return try MiniMaxM3Model(
            configuration, routedExperts: routed, excludedTensorNames: banks.excludedTensorNames)
    }
}
