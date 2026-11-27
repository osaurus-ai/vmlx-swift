// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXLMCommon

/// Loads native OCP MXFP4 ERNIE 5.5 Flash bundles (e.g. `dealignai/ERNIE-5.5-Flash-Preview-UNCENSORED-MXFP4`),
/// whose routed MoE experts are stored as unstacked `I8 [N, K/2]` + `F8_E8M0 [N, K/32]` tensors across
/// safetensors shards, while dense/attention/shared/latent weights use 128×128 `F8_E4M3` FP8.
///
/// Shards are streamed sequentially with uncached `ResidentSafetensorsReader` I/O and stacked layer-by-layer
/// into contiguous `[E, N, K/8]` `uint32` + `[E, N, K/32]` `uint8` banks so transient load RAM stays bounded.
struct Ernie55MXFP4Preparation {
    let directory: URL
    let configuration: Ernie55Configuration
    let ordinaryConfiguration: Data
    let excludedTensorNames: Set<String>
    let indexWeightMap: [String: String]

    static func loadIfDeclared(directory: URL, configurationData: Data) throws -> Self? {
        guard var root = try JSONSerialization.jsonObject(with: configurationData) as? [String: Any],
            root["model_type"] as? String == "ernie5_5_moe",
            !JANGHModelPreparation.declaresCustomFormat(configuration: configurationData, sidecar: nil),
            let qcfg = root["quantization_config"] as? [String: Any],
            (qcfg["expert_dtype"] as? String)?.lowercased() == "mxfp4"
        else { return nil }

        let config = try Ernie55Configuration.decode(configurationData)
        let indexURL = directory.appendingPathComponent("model.safetensors.index.json")
        guard FileManager.default.fileExists(atPath: indexURL.path) else {
            throw JangLoaderError.invalidConfig("ERNIE MXFP4 bundle requires model.safetensors.index.json")
        }
        let indexData = try Data(contentsOf: indexURL)
        guard let indexJSON = try JSONSerialization.jsonObject(with: indexData) as? [String: Any],
            let weightMap = indexJSON["weight_map"] as? [String: String]
        else {
            throw JangLoaderError.invalidConfig("Invalid model.safetensors.index.json in ERNIE MXFP4 bundle")
        }

        // Exclude all routed expert tensors (loaded and stacked directly below) and inline DSpark MTP
        // tensors (loaded on demand by Ernie55DSparkHead.loadInlineMXFP4).
        var excluded = Set<String>()
        excluded.reserveCapacity(weightMap.count)
        for key in weightMap.keys {
            if key.contains(".block_sparse_moe.experts.") || key.hasPrefix("model.mtp.") {
                excluded.insert(key)
            }
        }
        // Rewrite quantization metadata in-memory so BaseConfiguration decodes cleanly while
        // non-expert FP8 linears use Ernie55Model.fp8Quantization (MXFP8).
        let ordinary: [String: Any] = ["group_size": 32, "bits": 8, "mode": "mxfp8"]
        root["quantization"] = ordinary
        root["quantization_config"] = ordinary
        let ordinaryData = try JSONSerialization.data(withJSONObject: root)
        return Self(
            directory: directory,
            configuration: config,
            ordinaryConfiguration: ordinaryData,
            excludedTensorNames: excluded,
            indexWeightMap: weightMap)
    }

    func construct(requesting: Set<ModelRuntimeRequestModality>?) throws -> Ernie55Model {
        _ = try Set.resolveForConstruction(requested: requesting, constructible: [.text])
        let layers = try Self.loadStackedMXFP4Experts(
            directory: directory,
            weightMap: indexWeightMap,
            layerPrefix: "model.layers.",
            layerIndices: Array(configuration.firstKDenseReplace ..< configuration.numHiddenLayers),
            expertCount: configuration.nRoutedExperts,
            hiddenSize: configuration.moeLatentSize,
            intermediateSize: configuration.moeIntermediateSize)
        return Ernie55Model(configuration, routed: layers, excluded: excludedTensorNames)
    }

    /// Streams only the needed shards once (uncached) and stacks each layer's unstacked MXFP4
    /// `experts.{0..<E}.{w1,w2,w3}.{weight,weight_scale_inv}` into contiguous `[E, N, K/8]` `uint32`
    /// and `[E, N, K/32]` `uint8` banks.
    static func loadStackedMXFP4Experts(
        directory: URL,
        weightMap: [String: String],
        layerPrefix: String,
        layerIndices: [Int],
        expertCount: Int,
        hiddenSize: Int,
        intermediateSize: Int
    ) throws -> [Int: any Ernie55RoutedExperts] {
        // Group required expert keys by layer and shard.
        var layerShards: [Int: Set<String>] = [:]
        var shardWantedKeys: [String: Set<String>] = [:]
        for layer in layerIndices {
            let prefix = "\(layerPrefix)\(layer).block_sparse_moe.experts."
            var shards = Set<String>()
            for (key, shard) in weightMap where key.hasPrefix(prefix) {
                shards.insert(shard)
                shardWantedKeys[shard, default: []].insert(key)
            }
            guard !shards.isEmpty else {
                throw JangLoaderError.invalidConfig("Missing MXFP4 routed experts for \(prefix)")
            }
            layerShards[layer] = shards
        }

        // Precompute all keys per shard so ResidentSafetensorsReader skips non-expert keys in O(1).
        var shardAllKeys: [String: Set<String>] = [:]
        for (key, shard) in weightMap where shardWantedKeys[shard] != nil {
            shardAllKeys[shard, default: []].insert(key)
        }

        // Cache loaded shard tensors only while still needed by remaining layers.
        var shardRefCount: [String: Int] = [:]
        for shards in layerShards.values {
            for s in shards { shardRefCount[s, default: 0] += 1 }
        }
        var loadedShards: [String: [String: MLXArray]] = [:]

        func ensureShard(_ shard: String) throws -> [String: MLXArray] {
            if let existing = loadedShards[shard] { return existing }
            let wanted = shardWantedKeys[shard] ?? []
            let excluded = (shardAllKeys[shard] ?? []).subtracting(wanted)
            let url = directory.appendingPathComponent(shard)
            let (tensors, _) = try ResidentSafetensorsReader.loadWeights(url: url, excludingKeys: excluded)
            loadedShards[shard] = tensors
            return tensors
        }

        func releaseShard(_ shard: String) {
            guard let count = shardRefCount[shard] else { return }
            if count <= 1 {
                shardRefCount.removeValue(forKey: shard)
                loadedShards.removeValue(forKey: shard)
            } else {
                shardRefCount[shard] = count - 1
            }
        }

        var result: [Int: any Ernie55RoutedExperts] = [:]
        result.reserveCapacity(layerIndices.count)

        for (stepIdx, layer) in layerIndices.enumerated() {
            try Task.checkCancellation()
            let shards = Array(layerShards[layer] ?? []).sorted()
            var pool: [String: MLXArray] = [:]
            for shard in shards {
                let dict = try ensureShard(shard)
                let prefix = "\(layerPrefix)\(layer).block_sparse_moe.experts."
                for (k, v) in dict where k.hasPrefix(prefix) {
                    pool[k] = v
                }
            }

            func stackProj(_ proj: String) throws -> (MLXArray, MLXArray) {
                var ws: [MLXArray] = []
                var ss: [MLXArray] = []
                ws.reserveCapacity(expertCount)
                ss.reserveCapacity(expertCount)
                for e in 0 ..< expertCount {
                    let base = "\(layerPrefix)\(layer).block_sparse_moe.experts.\(e).\(proj)"
                    guard let wRaw = pool.removeValue(forKey: base + ".weight"),
                        let sRaw = pool.removeValue(forKey: base + ".weight_scale_inv")
                    else {
                        throw JangLoaderError.invalidConfig("Missing MXFP4 tensor \(base)")
                    }
                    // I8 [N, K/2] -> uint32 [N, K/8]; F8_E8M0 [N, K/32] -> uint8 [N, K/32].
                    let wU32 = contiguous(wRaw).view(dtype: .uint32)
                    let sU8 = sRaw.dtype == .uint8 ? sRaw : contiguous(sRaw).view(dtype: .uint8)
                    ws.append(wU32)
                    ss.append(sU8)
                }
                let stackedW = contiguous(MLX.stacked(ws, axis: 0))
                let stackedS = contiguous(MLX.stacked(ss, axis: 0))
                return (stackedW, stackedS)
            }

            let (gw, gs) = try stackProj("w1")
            let (uw, us) = try stackProj("w3")
            let (dw, ds) = try stackProj("w2")
            MLX.eval(gw, gs, uw, us, dw, ds)
            result[layer] = Ernie55MXFP4RoutedExpertLayer(
                gateWeight: gw, gateScales: gs,
                upWeight: uw, upScales: us,
                downWeight: dw, downScales: ds,
                hiddenSize: hiddenSize, intermediateSize: intermediateSize)

            for shard in shards { releaseShard(shard) }
            if (stepIdx + 1) % 4 == 0 {
                MLX.Memory.clearCache()
            }
        }
        MLX.Memory.clearCache()
        return result
    }
}
