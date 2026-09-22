// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXNN

/// Auxiliary weights are selected before loading shared shards. In particular,
/// loading the vision tower must not also map the text expert banks beside it.
enum MiMoV26AuxiliaryLoader {
    private struct Index: Decodable {
        let weight_map: [String: String]
    }

    private static func names(in file: URL) throws -> Set<String> {
        let handle = try FileHandle(forReadingFrom: file)
        defer { try? handle.close() }
        guard let prefix = try handle.read(upToCount: 8), prefix.count == 8 else {
            throw VLMError.processing("Truncated MiMo safetensors header")
        }
        let count = prefix.withUnsafeBytes { UInt64(littleEndian: $0.loadUnaligned(as: UInt64.self)) }
        guard count > 0, count <= 64 * 1024 * 1024,
            let bytes = try handle.read(upToCount: Int(count)), bytes.count == Int(count),
            let header = try JSONSerialization.jsonObject(with: bytes) as? [String: Any]
        else { throw VLMError.processing("Invalid MiMo safetensors header") }
        return Set(header.keys.filter { $0 != "__metadata__" })
    }

    static func selectedWeights(
        in directory: URL, matching include: (String) -> Bool
    ) throws -> [String: MLXArray] {
        let indexURL = directory.appendingPathComponent("model.safetensors.index.json")
        if FileManager.default.fileExists(atPath: indexURL.path) {
            let index = try JSONDecoder().decode(Index.self, from: Data(contentsOf: indexURL))
            let selected = index.weight_map.filter { include($0.key) }
            guard !selected.isEmpty else { throw VLMError.processing("MiMo auxiliary weights are missing") }
            var result: [String: MLXArray] = [:]
            for shard in Set(selected.values).sorted() {
                guard !shard.isEmpty, shard == URL(fileURLWithPath: shard).lastPathComponent,
                    shard.hasSuffix(".safetensors") else {
                    throw VLMError.processing("Invalid MiMo auxiliary shard name")
                }
                let file = directory.appendingPathComponent(shard)
                let wanted = Set(selected.filter { $0.value == shard }.keys)
                let all = try names(in: file)
                guard wanted.isSubset(of: all) else {
                    throw VLMError.processing("MiMo auxiliary index references missing tensors")
                }
                let (arrays, _) = try loadArraysAndMetadata(
                    url: file, excludingKeys: all.subtracting(wanted), exactTensorBuffers: true)
                guard Set(arrays.keys) == wanted else {
                    throw VLMError.processing("MiMo auxiliary shard does not match its index")
                }
                result.merge(arrays) { first, _ in first }
            }
            return result
        }
        let file = directory.appendingPathComponent("model.safetensors")
        let all = try names(in: file)
        let wanted = all.filter(include)
        guard !wanted.isEmpty else { throw VLMError.processing("MiMo auxiliary weights are missing") }
        return try loadArraysAndMetadata(
            url: file, excludingKeys: all.subtracting(wanted), exactTensorBuffers: true).0
    }

    static func vision(
        in directory: URL, configuration: MiMoV26VisionConfiguration
    ) throws -> MiMoV26VisionTower {
        let raw = try selectedWeights(in: directory) { $0.hasPrefix("visual.") }
        var weights: [String: MLXArray] = [:]
        for (source, tensor) in raw {
            let key = String(source.dropFirst("visual.".count))
            weights[key] = key == "patch_embed.proj.weight" && tensor.ndim == 5
                ? tensor.reshaped(tensor.dim(0), -1) : tensor
        }
        let tower = MiMoV26VisionTower(configuration)
        try tower.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        return tower
    }

    static func audioEncoder(
        in directory: URL, configuration: MiMoV26AudioConfiguration
    ) throws -> MiMoV26AudioEncoder {
        let raw = try selectedWeights(in: directory) {
            $0.hasPrefix("speech_embeddings.")
                || ($0.hasPrefix("audio_encoder.")
                    && !$0.hasPrefix("audio_encoder.input_local_transformer.embed_tokens."))
        }
        var weights: [String: MLXArray] = [:]
        for (source, tensor) in raw {
            let key = source.hasPrefix("audio_encoder.")
                ? String(source.dropFirst("audio_encoder.".count)) : source
            let mapped = key.replacingOccurrences(of: "projection.mlp.0.", with: "projection.fc1.")
                .replacingOccurrences(of: "projection.mlp.2.", with: "projection.fc2.")
            weights[mapped] = tensor
        }
        let encoder = MiMoV26AudioEncoder(configuration)
        try encoder.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        return encoder
    }

    static func audioTokenizer(
        in directory: URL, configuration: MiMoV26AudioTokenizerConfiguration
    ) throws -> MiMoV26AudioTokenizer {
        let raw = try selectedWeights(in: directory) {
            $0.hasPrefix("encoder.") && (!$0.hasPrefix("encoder.quantizer.")
                || $0.hasSuffix("._codebook.embed"))
        }
        var weights: [String: MLXArray] = [:]
        for (source, tensor) in raw {
            var key = String(source.dropFirst("encoder.".count))
            var value = tensor
            if key.hasPrefix("quantizer.") {
                let parts = key.split(separator: ".")
                guard parts.count == 6, parts[0] == "quantizer", parts[1] == "vq",
                    parts[2] == "layers", let index = Int(parts[3]),
                    index >= 0, index < configuration.num_quantizers,
                    parts[4] == "_codebook", parts[5] == "embed" else {
                    throw VLMError.processing("Invalid MiMo audio codebook key")
                }
                key = "codebooks.\(index).weight"
            } else if key == "conv1.weight" || key == "conv2.weight" {
                guard tensor.ndim == 3 else {
                    throw VLMError.processing("MiMo audio convolution weights must have rank three")
                }
                value = tensor.transposed(0, 2, 1)
            } else if key == "down_sample_layer.0.weight" {
                guard tensor.ndim == 3 else {
                    throw VLMError.processing("MiMo audio downsample weights must have rank three")
                }
                key = "down_sample_layer.conv.weight"
                value = tensor.transposed(0, 2, 1)
            }
            // Native input-audio tokenization uses FP32. Widen encoder BF16
            // weights, but never round the original FP32 RVQ codebooks.
            weights[key] = value.asType(.float32)
        }
        let tokenizer = try MiMoV26AudioTokenizer(configuration)
        try tokenizer.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        return tokenizer
    }
}
