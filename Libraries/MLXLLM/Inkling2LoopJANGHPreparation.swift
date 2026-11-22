// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLXLMCommon

/// Inkling-2-Mini JANGH bundles (`inkling2_loop`).
///
/// Bundle contract (written by the JANG converter, the plan the Python reference loads):
/// - `quantization` keyed by LIVE module path, the only plan in config.json: `model.layers.L.self_attn.{q,k,v,o,gate,r}
///   _proj`, `model.embed_tokens` and `lm_head` affine 8-bit / group 64; every other tensor (router + selection bias,
///   sinks, depth / message queries and gates, short convs, clip scales, relative-bias projection, norms) as released;
/// - routed experts `model.layers.L.mlp.switch_mlp.{gate,up,down}_proj` are `{"mode": "jangtq2", "bits": 2|3,
///   "rotation": "hadamard32"}` codebook banks (`tq2_packed`, `tq2_scales`) on all 24 layers, gate and up with the
///   same width per layer, separate (not row-interleaved like the checkpoint's `w13`);
/// - `vision_tower.*` (the vision tower, BF16 as released) and `mtp_drafter.*` (the native dSpark drafter, BF16).
///
/// The routed activation is the model's (`mlp_gate_softcap`, `mlp_up_softcap` from the config), never the bundle's:
/// `h = softcap(g, 4) * sigmoid(g) * softcap(u, 25)`, fused into the JANGH gate/up epilogues.
struct Inkling2LoopJANGHPreparation {
    enum PreparationError: Error, CustomStringConvertible {
        case invalid(String)
        var description: String {
            switch self { case .invalid(let message): return message }
        }
    }

    let configuration: Inkling2LoopConfiguration
    let banks: JANGHModelPreparation

    var ordinaryConfiguration: Data { banks.ordinaryConfiguration }

    /// nil unless config.json is an `inkling2_loop` config with custom (JANGH) routed projections.
    static func loadIfDeclared(directory: URL, configurationData: Data) throws -> Self? {
        guard let root = try? JSONSerialization.jsonObject(with: configurationData) as? [String: Any],
            root["model_type"] as? String == "inkling2_loop",
            JANGHModelPreparation.declaresCustomFormat(configuration: configurationData, sidecar: nil)
        else { return nil }
        let config = try JSONDecoder.json5().decode(Inkling2LoopConfiguration.self, from: configurationData)
        let banks = try JANGHModelPreparation(
            directory: directory, configuration: configurationData, sidecar: nil,
            layout: .init(
                modelType: "inkling2_loop", hiddenSize: config.hiddenSize,
                intermediateSize: config.intermediateSize, expertCount: config.nRoutedExperts,
                sparseLayers: Set(0 ..< config.numHiddenLayers), routesPerToken: config.numExpertsPerTok))
        // The ordinary plan must decode before any bank is mapped (JANGHModelPreparation contract).
        _ = try JSONDecoder.json5().decode(BaseConfiguration.self, from: banks.ordinaryConfiguration)
        return Self(configuration: config, banks: banks)
    }

    func construct(requesting: Set<ModelRuntimeRequestModality>?) throws -> Inkling2LoopModel {
        // Lanes resolve exactly as the non-JANGH dispatch: the vision tower (BF16 as released, `vision_tower.*`) is
        // built only when the config declares one and the request includes the vision lane (or names none); only the
        // routed experts differ.
        var args = configuration
        let lanes = try Set.resolveForConstruction(
            requested: requesting, constructible: args.vision == nil ? [.text] : [.text, .vision])
        if !lanes.contains(.vision) { args.vision = nil }
        let routed = try banks.makeCodebookRoutedExperts(
            activation: .inkling(gateCap: args.mlpGateSoftcap, upCap: args.mlpUpSoftcap))
        return try Inkling2LoopModel(args, routedExperts: routed, excludedTensorNames: banks.excludedTensorNames)
    }
}
