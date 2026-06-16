import Foundation
@preconcurrency import MLX
import MLXNN
@preconcurrency import MLXLMCommon

// MARK: - WeightLoader
//
// Loads safetensors weight shards from a local model directory and
// applies vMLX's JANG-aware remapping if the model has a `jang_config.json`.
//
// Flow (JANG-less path):
//   1. Read `model.safetensors.index.json` to enumerate all shards.
//   2. Open each shard via `MLX.loadArrays(url:)`.
//   3. Merge into a single `[String: MLXArray]` dict keyed by weight name.
//
// Flow (JANG path):
//   1. Parse `jang_config.json` via `JangLoader` (reused from vmlx-swift-lm).
//   2. Enumerate shards as above.
//   3. Apply per-layer quantization metadata before use — the
//      `QuantizedLinear` modules in the model module tree check the
//      `jangConfig.quantization.bitWidthsUsed[layer_idx]` to pick
//      the right decode path.
//
// For now this file is the SCAFFOLD — weight loading returns a dict,
// per-layer JANG remapping plugs in when the first real model port
// (Flux1Schnell) has a concrete module tree to apply it to.

public struct LoadedWeights: Sendable {
    public let weights: [String: MLXArray]
    public let componentWeights: [String: [String: MLXArray]]
    public let jangConfig: MLXLMCommon.JangConfig?

    public init(
        weights: [String: MLXArray],
        componentWeights: [String: [String: MLXArray]] = [:],
        jangConfig: MLXLMCommon.JangConfig? = nil
    ) {
        self.weights = weights
        self.componentWeights = componentWeights
        self.jangConfig = jangConfig
    }
}

public enum WeightLoader {

    public static func indexedWeightKeys(in directory: URL, component: String) throws -> Set<String> {
        let componentURL = component == "root"
            ? directory
            : directory.appendingPathComponent(component, isDirectory: true)
        guard let indexURL = firstIndexURL(in: componentURL) else { return [] }
        let data = try Data(contentsOf: indexURL)
        guard let obj = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let weightMap = obj["weight_map"] as? [String: String]
        else {
            throw FluxError.invalidRequest("invalid safetensors index at \(indexURL.path)")
        }
        let fm = FileManager.default
        for shardName in Set(weightMap.values) {
            let shardURL = componentURL.appendingPathComponent(shardName)
            guard fm.fileExists(atPath: shardURL.path) else {
                throw FluxError.weightsNotFound(shardURL)
            }
        }
        return Set(weightMap.keys)
    }

    /// Load all safetensors shards from a model directory and return a
    /// merged dict. If the directory contains `jang_config.json`, the
    /// parsed config is returned alongside so the caller can apply
    /// per-layer quantization during module construction.
    public static func load(from directory: URL) throws -> LoadedWeights {
        // Detect JANG first so the caller gets the config regardless of
        // the shard layout.
        let jang = try JangBridge.detect(at: directory)

        let componentShardGroups = try enumerateShardGroups(in: directory)
        guard !componentShardGroups.isEmpty else {
            throw FluxError.weightsNotFound(directory)
        }

        var merged: [String: MLXArray] = [:]
        var byComponent: [String: [String: MLXArray]] = [:]
        for group in componentShardGroups {
            var componentWeights: [String: MLXArray] = [:]
            for shard in group.shards {
                let arrays = try MLX.loadArrays(url: shard)
                for (key, value) in arrays {
                    componentWeights[key] = value
                    let mergedKey = group.component == "root"
                        ? key
                        : "\(group.component).\(key)"
                    merged[mergedKey] = value
                }
            }
            byComponent[group.component] = componentWeights
        }

        return LoadedWeights(
            weights: merged,
            componentWeights: byComponent,
            jangConfig: jang.config)
    }

    private struct ShardGroup {
        let component: String
        let shards: [URL]
    }

    /// Enumerate .safetensors files in the directory and known
    /// Diffusers/MFlux component subdirectories. Prefers each
    /// `model.safetensors.index.json` manifest when present so shards
    /// are deterministic, falling back to sorted direct `.safetensors`.
    private static func enumerateShardGroups(in directory: URL) throws -> [ShardGroup] {
        var groups: [ShardGroup] = []
        let rootShards = try enumerateShards(in: directory)
        if !rootShards.isEmpty {
            groups.append(ShardGroup(component: "root", shards: rootShards))
        }
        for component in ["transformer", "unconditional_transformer", "text_encoder", "text_encoder_2", "vae"] {
            let componentURL = directory.appendingPathComponent(component, isDirectory: true)
            let shards = try enumerateShards(in: componentURL)
            if !shards.isEmpty {
                groups.append(ShardGroup(component: component, shards: shards))
            }
        }
        return groups
    }

    private static let indexCandidateNames = [
        "model.safetensors.index.json",
        "diffusion_pytorch_model.safetensors.index.json",
    ]

    private static func firstIndexURL(in directory: URL) -> URL? {
        let fm = FileManager.default
        return indexCandidateNames
            .map { directory.appendingPathComponent($0) }
            .first(where: { fm.fileExists(atPath: $0.path) })
    }

    private static func enumerateShards(in directory: URL) throws -> [URL] {
        let fm = FileManager.default
        var isDirectory: ObjCBool = false
        guard fm.fileExists(atPath: directory.path, isDirectory: &isDirectory),
              isDirectory.boolValue
        else { return [] }
        if let indexURL = firstIndexURL(in: directory) {
            // Parse the index to get the unique set of shards.
            let data = try Data(contentsOf: indexURL)
            if let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let weightMap = obj["weight_map"] as? [String: String] {
                let shardNames = Set(weightMap.values)
                return shardNames
                    .sorted()
                    .map { directory.appendingPathComponent($0) }
            }
        }
        // Fallback: glob all .safetensors in the directory.
        let entries = try fm.contentsOfDirectory(atPath: directory.path)
        let shardNames = entries
            .filter { $0.hasSuffix(".safetensors") }
            .sorted()
        return shardNames.map { directory.appendingPathComponent($0) }
    }
}
