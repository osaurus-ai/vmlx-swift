import Foundation
import MLXLMCommon

/// Validate final codebook banks independently of the RTN affine development bundle.
struct FlintJANGHPreparation {
    let configuration: FlintConfiguration
    let banks: JANGHModelPreparation

    static func loadIfDeclared(directory: URL, configurationData: Data) throws -> Self? {
        guard let root = try JSONSerialization.jsonObject(with: configurationData) as? [String: Any],
            root["model_type"] as? String == "flint" else { return nil }
        let url = directory.appendingPathComponent("jang_config.json")
        let sidecar = FileManager.default.fileExists(atPath: url.path) ? try Data(contentsOf: url) : nil
        guard JANGHModelPreparation.declaresCustomFormat(configuration: configurationData, sidecar: sidecar)
        else { return nil }
        let config = try JSONDecoder.json5().decode(FlintConfiguration.self, from: configurationData)
        try config.validate()
        let banks = try JANGHModelPreparation(directory: directory,
            configuration: configurationData, sidecar: sidecar,
            layout: .init(modelType: "flint", hiddenSize: config.hiddenSize,
                intermediateSize: config.moeIntermediateSize, expertCount: config.numExperts,
                sparseLayers: Set(0..<config.hiddenLayers).subtracting(config.mlpOnlyLayers),
                routesPerToken: config.topK))
        _ = try JSONDecoder.json5().decode(BaseConfiguration.self, from: banks.ordinaryConfiguration)
        return Self(configuration: config, banks: banks)
    }

    func construct(requesting: Set<ModelRuntimeRequestModality>?) throws -> FlintModel {
        guard requesting == nil || requesting == [.text] else { throw FlintError.invalidConfiguration }
        return try FlintModel(configuration, routed: banks.makeRoutedExperts(activationLimit: nil),
                              excluded: banks.excludedTensorNames)
    }
}
