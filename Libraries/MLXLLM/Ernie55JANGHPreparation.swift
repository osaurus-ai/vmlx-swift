// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLXLMCommon

struct Ernie55JANGHPreparation {
    let configuration: Ernie55Configuration
    let banks: JANGHModelPreparation

    let ordinaryConfiguration: Data

    static func loadIfDeclared(directory: URL, configurationData: Data) throws -> Self? {
        guard var root = try JSONSerialization.jsonObject(with: configurationData) as? [String: Any],
            root["model_type"] as? String == "ernie5_5_moe",
            JANGHModelPreparation.declaresCustomFormat(configuration: configurationData, sidecar: nil)
        else { return nil }
        let config = try Ernie55Configuration.decode(configurationData)
        if root["jangtq"] == nil {
            // The legacy ERNIE converter used the version-2 fixed codebooks but
            // recorded the per-bank plan in both config.json and jang_config.json.
            // Recover this one assembly schema only, before strict admission.
            let manifest = try Data(contentsOf: directory.appendingPathComponent("jang_config.json"))
            guard let export = try JSONSerialization.jsonObject(with: manifest) as? [String: Any],
                export["format"] as? String == "jangtq2",
                export["source"] as? String == "baidu/ERNIE-5.5-Flash-Preview",
                export["source_revision"] as? String == "ace1100dcde7bf4eb06add6ae11ce4f43a11ddfe",
                export["expert_fp8"] as? Bool == true,
                let sidecarPlan = export["expert_units"] as? NSDictionary,
                let plan = root["quantization"] as? NSDictionary, sidecarPlan == plan
            else { throw JangLoaderError.invalidConfig("ERNIE JANGH codebook declaration is missing") }
            root["jangtq"] = Self.legacyCodebooks()
        }
        let banks = try JANGHModelPreparation(
            directory: directory, configuration: try JSONSerialization.data(withJSONObject: root), sidecar: nil,
            layout: .init(
                modelType: "ernie5_5_moe", hiddenSize: config.moeLatentSize,
                intermediateSize: config.moeIntermediateSize, expertCount: config.nRoutedExperts,
                sparseLayers: Set(config.firstKDenseReplace..<config.numHiddenLayers),
                routesPerToken: config.numExpertsPerTok), allowsUnalignedResidentRegions: true)
        // The published plan describes only routed expert banks. Generic
        // linears use the projection loader's exact MXFP8 representation.
        guard var exported = try JSONSerialization.jsonObject(with: banks.ordinaryConfiguration) as? [String: Any],
            var ordinary = exported["quantization"] as? [String: Any]
        else { throw JangLoaderError.invalidConfig("Missing ERNIE ordinary quantization plan") }
        ordinary["group_size"] = 32
        ordinary["bits"] = 8
        ordinary["mode"] = "mxfp8"
        exported["quantization"] = ordinary
        exported["quantization_config"] = ordinary
        return Self(configuration: config, banks: banks,
                    ordinaryConfiguration: try JSONSerialization.data(withJSONObject: exported))
    }

    private static func legacyCodebooks() -> [String: Any] {
        let parameters = ["2": (0.8929999999999999, 0.05065),
                          "3": (0.47124999999999995, 0.011599999999999997)]
        let books = parameters.map { width, constants -> (String, [String: Any]) in
            let count = 1 << Int(width)!
            let (alpha, beta) = constants
            let levels = (0..<count).map { index -> Float in
                let u = Double(index) - Double(count - 1) / 2
                return Float(u * (alpha + beta * u * u))
            }
            return (width, ["alpha": alpha, "beta": beta, "levels": levels])
        }
        return ["version": 2, "packing": "lsb-bitstream", "scale_dtype": "float16",
                "codebook_family": "odd-cubic", "rotation": "hadamard32",
                "codebooks": Dictionary(uniqueKeysWithValues: books)]
    }

    func construct(requesting: Set<ModelRuntimeRequestModality>?) throws -> Ernie55Model {
        _ = try Set.resolveForConstruction(requested: requesting, constructible: [.text])
        let caps = configuration.caps
        let routed = try banks.makeRoutedExperts(
            activationLimit: nil, activation: .ernie55(gateCap: caps.gate, upCap: caps.up))
        let experts: [Int: any Ernie55RoutedExperts] = try routed.mapValues { layer -> any Ernie55RoutedExperts in
            guard let ernieBank = layer as? JANGTRoutedExpertLayer else {
                throw JangLoaderError.invalidConfig("ERNIE needs the SiTU routed expert banks")
            }
            return ernieBank
        }
        return Ernie55Model(configuration, routed: experts, excluded: banks.excludedTensorNames)
    }
}
