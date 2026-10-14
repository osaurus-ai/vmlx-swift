import Foundation
import MLXLMCommon

/// LFM codebook banks are admitted from config plus exact shard headers, never a bundle name.
enum LFM2MoEJANGHError: Error { case invalid(String) }

struct LFM2MoEJANGHPreparation {
    let configuration: LFM2MoEConfiguration
    let banks: JANGHModelPreparation

    static func loadIfDeclared(directory: URL, configurationData: Data) throws -> Self? {
        guard let root = try? JSONSerialization.jsonObject(with: configurationData) as? [String: Any],
            root["model_type"] as? String == "lfm2_moe"
        else { return nil }
        let sidecarURL = directory.appendingPathComponent("jang_config.json")
        let sidecar = FileManager.default.fileExists(atPath: sidecarURL.path)
            ? try Data(contentsOf: sidecarURL) : nil
        guard JANGHModelPreparation.declaresCustomFormat(configuration: configurationData, sidecar: sidecar)
        else { return nil }
        let config = try JSONDecoder.json5().decode(LFM2MoEConfiguration.self, from: configurationData)
        guard config.hiddenLayers > 0, config.numDenseLayers >= 0,
            config.numDenseLayers < config.hiddenLayers, config.hiddenSize > 0,
            config.moeIntermediateSize > 0, config.numExperts > 0,
            config.numExpertsPerToken > 0, config.numExpertsPerToken <= config.numExperts
        else { throw LFM2MoEJANGHError.invalid("invalid LFM JANGH architecture dimensions") }
        let banks = try JANGHModelPreparation(
            directory: directory, configuration: configurationData, sidecar: sidecar,
            layout: .init(modelType: "lfm2_moe", hiddenSize: config.hiddenSize,
                intermediateSize: config.moeIntermediateSize, expertCount: config.numExperts,
                sparseLayers: Set(config.numDenseLayers ..< config.hiddenLayers),
                routesPerToken: config.numExpertsPerToken, feedForwardModule: "feed_forward"))
        // Fail unknown ordinary modes before mapped bank construction.
        _ = try JSONDecoder.json5().decode(BaseConfiguration.self, from: banks.ordinaryConfiguration)
        return Self(configuration: config, banks: banks)
    }

    static func validateRequestedModalities(_ requesting: Set<ModelRuntimeRequestModality>?) throws {
        // Pure text families have no optional tower to select. Empty requests,
        // tools and reasoning keep the same language core; media needs a VLM.
        if requesting?.isDisjoint(with: [.vision, .video, .audio]) == false {
            throw LFM2MoEJANGHError.invalid("LFM2-MoE accepts text input only")
        }
    }

    func construct(requesting: Set<ModelRuntimeRequestModality>?) throws -> LFM2MoEModel {
        try Self.validateRequestedModalities(requesting)
        return try LFM2MoEModel(configuration,
            routedExperts: banks.makeRoutedExperts(activationLimit: nil),
            excludedTensorNames: banks.excludedTensorNames)
    }
}
