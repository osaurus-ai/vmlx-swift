import Foundation
import MLXLMCommon

struct K2HorizonJANGHPreparation {
    let configuration: K2HorizonConfiguration
    let routed: JANGHModelPreparation?
    let dense: JANGHDenseModelPreparation?
    var ordinaryConfiguration: Data {
        dense?.ordinaryConfiguration ?? routed!.ordinaryConfiguration
    }

    static func loadIfDeclared(directory: URL, configurationData: Data) throws -> Self? {
        guard
            let root = try JSONSerialization.jsonObject(with: configurationData) as? [String: Any],
            root["model_type"] as? String == "k2_horizon"
        else { return nil }
        let sidecarURL = directory.appendingPathComponent("jang_config.json")
        let sidecar =
            FileManager.default.fileExists(atPath: sidecarURL.path)
            ? try Data(contentsOf: sidecarURL) : nil
        guard
            JANGHModelPreparation.declaresCustomFormat(
                configuration: configurationData, sidecar: sidecar)
        else { return nil }
        let c = try JSONDecoder.json5().decode(K2HorizonConfiguration.self, from: configurationData)
        guard !NativeMTPActivation.isExplicitlyRequested else {
            throw K2HorizonConfiguration.ContractError.unsupported(
                "K2 does not implement native MTP")
        }
        let result: Self
        switch c.mlpLayout {
        case "switch1":
            // Routed execution switches to tiled prefill at 64 routed rows.
            // Both prefill backends require K divisible by 64: hiddenSize for
            // gate/up and intermediateSize for down. Decode alone accepts 32.
            guard c.hiddenSize.isMultiple(of: 64), c.intermediateSize.isMultiple(of: 64) else {
                throw K2HorizonConfiguration.ContractError.unsupported(
                    "K2 switch1 JANGH requires hidden_size and intermediate_size divisible by 64 for prefill"
                )
            }
            result = Self(
                configuration: c,
                routed: try JANGHModelPreparation(
                    directory: directory, configuration: configurationData, sidecar: sidecar,
                    layout: .init(
                        modelType: "k2_horizon", hiddenSize: c.hiddenSize,
                        intermediateSize: c.intermediateSize, expertCount: 1,
                        sparseLayers: Set(0 ..< c.hiddenLayers), routesPerToken: 1)), dense: nil)
        case "dense_jangh_down":
            result = Self(
                configuration: c, routed: nil,
                dense: try JANGHDenseModelPreparation(
                    directory: directory, configuration: configurationData, sidecar: sidecar,
                    hiddenSize: c.hiddenSize, intermediateSize: c.intermediateSize,
                    layerCount: c.hiddenLayers))
        default:
            throw K2HorizonConfiguration.ContractError.unsupported(
                "K2 JANGH requires switch1 or dense_jangh_down")
        }
        // Reject malformed ordinary quantization before any packed bank is mapped.
        _ = try JSONDecoder.json5().decode(
            BaseConfiguration.self, from: result.ordinaryConfiguration)
        return result
    }

    func construct(requesting: Set<ModelRuntimeRequestModality>?) throws -> K2HorizonModel {
        guard requesting == nil || requesting == [.text] else {
            throw K2HorizonConfiguration.ContractError.unsupported("K2 accepts text input only")
        }
        if let dense {
            return try K2HorizonModel(
                configuration, routedFactory: nil,
                denseDown: dense.makeProjections(fastDecode: true),
                excludedSafetensorsKeys: dense.excludedTensorNames)
        }
        let banks = try routed!.makeRoutedExperts(activationLimit: nil)
        return try K2HorizonModel(
            configuration,
            routedFactory: { layer in
                guard let bank = banks[layer] else {
                    throw K2HorizonConfiguration.ContractError.unsupported(
                        "missing admitted K2 routed layer")
                }
                return bank
            }, excludedSafetensorsKeys: routed!.excludedTensorNames)
    }
}
