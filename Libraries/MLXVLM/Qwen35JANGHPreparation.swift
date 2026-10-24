import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Qwen3.8-27B JANGH2 (dense `qwen3_5` + vision): every decoder MLP projection
/// (`language_model.model.layers.L.mlp.{gate,up,down}_proj`, 64 layers x 3 = 192) is a one-expert JANGH codebook
/// bank (`mode: jangtq2`, 2-bit, hadamard32 rotation, odd-cubic codebook); everything else (GDN, attention,
/// embeddings, head, vision) is ordinary affine and loads through the generic path.
///
/// Before 2026-10-06 the Swift loader failed at config decode ("Unsupported quantization mode 'jangtq2'") because
/// only routed (`switch_mlp`) and K2 down-only JANGH layouts were admitted. Port of vMLX Python
/// `jangh/dense.py` (TQLinear: one-expert bank, gather QMV < 4 rows, sorted QMM >= 4 rows).
enum Qwen35JANGHError: Error, CustomStringConvertible {
    case invalid(String)
    var description: String {
        if case .invalid(let m) = self { return m }
        return ""
    }
}

struct Qwen35JANGHPreparation {
    let dense: JANGHDenseModelPreparation

    /// Declared by the explicit contract (`mode: jangtq2` / `jangtq` header), never by a folder name.
    static func loadIfDeclared(directory: URL, configurationData: Data) throws -> Self? {
        guard
            let root = try? JSONSerialization.jsonObject(with: configurationData) as? [String: Any],
            root["model_type"] as? String == "qwen3_5"
        else { return nil }
        let sidecarURL = directory.appendingPathComponent("jang_config.json")
        let sidecarData =
            FileManager.default.fileExists(atPath: sidecarURL.path)
            ? try Data(contentsOf: sidecarURL) : nil
        guard
            JANGHModelPreparation.declaresCustomFormat(
                configuration: configurationData, sidecar: sidecarData)
        else { return nil }
        // Partition admission distinguishes the documented scalar summary from
        // a second header/plan; never discard a declared conflicting sidecar.
        guard let text = root["text_config"] as? [String: Any],
            let hidden = text["hidden_size"] as? Int, let inter = text["intermediate_size"] as? Int,
            let layers = text["num_hidden_layers"] as? Int
        else { throw Qwen35JANGHError.invalid("qwen3_5 JANGH: missing text_config dimensions") }
        return Self(
            dense: try JANGHDenseModelPreparation(
                qwen35Directory: directory, configuration: configurationData, sidecar: sidecarData,
                hiddenSize: hidden, intermediateSize: inter, layerCount: layers))
    }

    /// Construct the ordinary model (custom modules are `false` in the ordinary plan, so they stay plain
    /// Linear placeholders), then replace every dense JANGH projection BEFORE weights load.
    func construct(configurationData: Data, requesting: Set<ModelRuntimeRequestModality>?, decisionBackbone: Bool = false) throws
        -> Qwen35
    {
        let config = try JSONDecoder.json5().decode(
            Qwen35Configuration.self, from: configurationData)
        let model = try decisionBackbone ? Qwen35(decisionConfiguration: config) : Qwen35(config, requesting: requesting)
        let projections = try dense.makeProjectionsByPath()
        let leaves = Set(model.leafModules().flattened().map(\.0))
        var updates: [(String, Module)] = []
        for (path, module) in projections {
            guard leaves.contains(path) else {
                throw Qwen35JANGHError.invalid("qwen3_5 JANGH module \(path) is not a model leaf")
            }
            updates.append((path, module))
        }
        model.update(modules: ModuleChildren.unflattened(updates))
        model.customDenseTensorNames = dense.excludedTensorNames
        return model
    }
}
