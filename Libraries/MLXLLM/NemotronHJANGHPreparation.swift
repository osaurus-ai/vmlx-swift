import Foundation
import MLXLMCommon

/// Nemotron-H bundles whose routed experts are JANGH codebook banks (Nemotron
/// 3.5 Bolt JANGH2): gate-less relu2 `switch_mlp.{fc1,fc2}` stacks, each MoE
/// block with its own geometry (`block_configs`). Validates the architecture
/// and both weight formats before the model exists; the banks never pass
/// through the ordinary affine quantizer.
public struct NemotronHJANGHPreparation {
    public let configuration: NemotronHConfiguration
    public let banks: JANGHModelPreparation

    public static func loadIfDeclared(directory: URL, configurationData: Data) throws
        -> NemotronHJANGHPreparation?
    {
        guard let root = try? JSONSerialization.jsonObject(with: configurationData) as? [String: Any],
            root["model_type"] as? String == "nemotron_h"
        else { return nil }
        let sidecarURL = directory.appendingPathComponent("jang_config.json")
        let sidecar = FileManager.default.fileExists(atPath: sidecarURL.path)
            ? try Data(contentsOf: sidecarURL) : nil
        guard JANGHModelPreparation.declaresCustomFormat(
            configuration: configurationData, sidecar: sidecar)
        else { return nil }
        return try Self(directory: directory, configurationData: configurationData, sidecar: sidecar)
    }

    init(directory: URL, configurationData: Data, sidecar: Data?) throws {
        let config = try JSONDecoder.json5().decode(
            NemotronHConfiguration.self, from: configurationData)
        var layers: [Int: JANGHRelu2RoutedLayout.Geometry] = [:]
        for (index, kind) in config.hybridOverridePattern.enumerated() where kind == "E" {
            let geometry = config.moeGeometry(layer: index)
            layers[index] = .init(
                experts: geometry.experts, intermediate: geometry.intermediate,
                routes: geometry.topK)
        }
        banks = try JANGHModelPreparation(
            directory: directory, configuration: configurationData, sidecar: sidecar,
            relu2Layout: .init(
                modelType: "nemotron_h",
                expertInputWidth: config.moeLatentSize ?? config.hiddenSize,
                layers: layers,
                parentTemplate: "backbone.layers.{L}.mixer.switch_mlp"))
        configuration = config
    }

    /// The text model with every routed block bound to its mapped bank.
    /// Call only after the ordinary configuration decoded strictly.
    public func makeLanguageModel() throws -> NemotronHModel {
        let routed = try banks.makeRelu2RoutedExperts()
        let expected = Set(
            configuration.hybridOverridePattern.enumerated().filter { $0.element == "E" }.map(\.offset))
        guard Set(routed.keys) == expected else {
            throw NemotronHJANGHPreparationError.incompleteBanks
        }
        return NemotronHModel(
            configuration, routedLayers: routed,
            excludedSafetensorsKeys: banks.excludedTensorNames)
    }
}

public enum NemotronHJANGHPreparationError: Error, CustomStringConvertible {
    case incompleteBanks

    public var description: String {
        "Nemotron-H JANGH banks do not cover every MoE block"
    }
}
