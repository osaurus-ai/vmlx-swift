import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Ripple-65B-Preview JANGH2 (dense `ricursive`): every decoder MLP projection
/// (`model.layers.L.mlp.{gate,up,down}_proj`, 80 layers x 3 = 240) is a one-expert JANGH codebook bank
/// (`mode: jangtq2`, 2-bit, hadamard32 rotation, odd-cubic codebook, fp16 row scale with the unit gain folded in).
/// Attention, embeddings and the head are ordinary affine entries in the same config.json plan.
///
/// config.json is the single owner of the plan: the bundle's jang_config.json carries the header and chat/runtime
/// metadata only (its plan moved to `quantization_summary`).
struct RicursiveJANGHPreparation {
    let configuration: RicursiveConfiguration
    let dense: JANGHDenseModelPreparation

    var ordinaryConfiguration: Data { dense.ordinaryConfiguration }

    /// Declared by the explicit contract (`mode: jangtq2` / `jangtq` header), never by a folder name.
    static func loadIfDeclared(directory: URL, configurationData: Data) throws -> Self? {
        guard
            let root = try? JSONSerialization.jsonObject(with: configurationData) as? [String: Any],
            root["model_type"] as? String == "ricursive"
        else { return nil }
        let sidecarURL = directory.appendingPathComponent("jang_config.json")
        let sidecar =
            FileManager.default.fileExists(atPath: sidecarURL.path)
            ? try Data(contentsOf: sidecarURL) : nil
        guard
            JANGHModelPreparation.declaresCustomFormat(
                configuration: configurationData, sidecar: sidecar)
        else { return nil }
        let c = try JSONDecoder.json5().decode(RicursiveConfiguration.self, from: configurationData)
        let result = Self(
            configuration: c,
            dense: try JANGHDenseModelPreparation(
                qwen35Directory: directory, configuration: configurationData, sidecar: sidecar,
                hiddenSize: c.hiddenSize, intermediateSize: c.intermediateSize,
                layerCount: c.hiddenLayers))
        // Reject a malformed ordinary plan before any packed bank is used.
        _ = try JSONDecoder.json5().decode(BaseConfiguration.self, from: result.ordinaryConfiguration)
        return result
    }

    /// Build the ordinary model, then swap every dense JANGH projection in BEFORE weights load (custom modules are
    /// `false` in the ordinary plan, so they are plain Linear placeholders until replaced here).
    func construct(requesting: Set<ModelRuntimeRequestModality>?) throws -> RicursiveModel {
        guard requesting == nil || requesting == [.text] else {
            throw RicursiveConfiguration.ContractError.invalid("Ripple accepts text input only")
        }
        let model = RicursiveModel(configuration)
        // 1 row: the dense fast QMV (bitwise-admitted). 2+ rows (verify windows, prefill): the sorted / split-K tile,
        // ~flat from 2 to 16 rows on Ripple's shapes.
        let projections = try dense.makeProjectionsByPath()
        let leaves = Set(model.leafModules().flattened().map(\.0))
        var updates: [(String, Module)] = []
        for (path, module) in projections {
            guard leaves.contains(path) else {
                throw RicursiveConfiguration.ContractError.invalid(
                    "JANGH module \(path) is not a model leaf")
            }
            updates.append((path, module))
        }
        guard updates.count == 3 * configuration.hiddenLayers else {
            throw RicursiveConfiguration.ContractError.invalid(
                "JANGH2 must cover every MLP projection (\(updates.count) of \(3 * configuration.hiddenLayers))")
        }
        model.update(modules: ModuleChildren.unflattened(updates))
        model.customDenseTensorNames = dense.excludedTensorNames
        return model
    }
}
