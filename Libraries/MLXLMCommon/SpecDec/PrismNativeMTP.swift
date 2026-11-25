// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXNN

/// The optional Prism sidecar uses GGUF block stems and independent affine 4-bit/group-32
/// weights. The target's Hadamard embedding and output head remain shared.
package enum PrismNativeMTP {
    package static let evidence = "prism_mtp_sidecar=mtp/mtp-v2-head.safetensors"

    private static let prefix = "language_model.mtp."
    private static let projections = [
        "nextn.eh_proj": "fc",
        "attn_q": "layers.0.self_attn.q_proj",
        "attn_k": "layers.0.self_attn.k_proj",
        "attn_v": "layers.0.self_attn.v_proj",
        "attn_output": "layers.0.self_attn.o_proj",
        "ffn_gate": "layers.0.mlp.gate_proj",
        "ffn_up": "layers.0.mlp.up_proj",
        "ffn_down": "layers.0.mlp.down_proj",
    ]
    private static let norms = [
        "nextn.enorm": "pre_fc_norm_embedding",
        "nextn.hnorm": "pre_fc_norm_hidden",
        "nextn.shared_head_norm": "norm",
        "attn_norm": "layers.0.input_layernorm",
        "post_attention_norm": "layers.0.post_attention_layernorm",
        "attn_q_norm": "layers.0.self_attn.q_norm",
        "attn_k_norm": "layers.0.self_attn.k_norm",
    ]
    private static let mapping: [String: String] = Dictionary(uniqueKeysWithValues:
        projections.flatMap { source, target in
            ["weight", "scales", "biases"].map { ("\(source).\($0)", "\(prefix)\(target).\($0)") }
        } + norms.map { ("\($0.key).weight", "\(prefix)\($0.value).weight") })

    static func sidecar(at directory: URL) -> URL {
        directory.appendingPathComponent("mtp/mtp-v2-head.safetensors")
    }

    static func translatedNames(_ names: [String]) throws -> [String] {
        guard Set(names) == Set(mapping.keys) else {
            throw JangLoaderError.invalidConfig("incomplete or unsupported Prism MTP sidecar")
        }
        return mapping.values.sorted()
    }

    static func load(at directory: URL, model: LanguageModel) throws -> [String: MLXArray] {
        let (arrays, metadata) = try loadArraysAndMetadata(url: sidecar(at: directory))
        _ = try translatedNames(Array(arrays.keys))
        guard metadata["bits"] == "4", metadata["group_size"] == "32" else {
            throw JangLoaderError.invalidConfig("Prism MTP sidecar must declare affine 4-bit, group 32")
        }
        let modules = Dictionary(uniqueKeysWithValues: model.namedModules())
        for (source, target) in projections {
            guard let module = modules[prefix + target] as? Linear else {
                throw JangLoaderError.loadFailed("missing Prism MTP projection: \(target)")
            }
            let (rows, width) = module.shape
            guard width % 32 == 0,
                let weight = arrays["\(source).weight"], weight.dtype == .uint32,
                weight.shape == [rows, width / 8],
                let scales = arrays["\(source).scales"], scales.dtype == .float16,
                scales.shape == [rows, width / 32],
                let biases = arrays["\(source).biases"], biases.dtype == scales.dtype,
                biases.shape == scales.shape
            else {
                throw JangLoaderError.invalidConfig("invalid Prism MTP projection layout: \(source)")
            }
        }
        let expectedNorms = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        for (source, target) in norms {
            guard let expected = expectedNorms["\(prefix)\(target).weight"],
                let weight = arrays["\(source).weight"], weight.shape == expected.shape,
                weight.dtype == .float32 || weight.dtype == .float16
            else {
                throw JangLoaderError.invalidConfig("invalid Prism MTP norm layout: \(source)")
            }
        }
        return Dictionary(uniqueKeysWithValues: mapping.map { ($0.value, arrays[$0.key]!) })
    }

    static func quantization(
        extending existing: BaseConfiguration.PerLayerQuantization?
    ) -> BaseConfiguration.PerLayerQuantization {
        var combined: BaseConfiguration.PerLayerQuantization = existing ?? .init(
            quantization: .init(groupSize: 128, bits: 2), perLayerQuantization: [:])
        let draftQuantization = BaseConfiguration.Quantization(groupSize: 32, bits: 4)
        for target in projections.values {
            combined.perLayerQuantization[prefix + target] = .quantize(draftQuantization)
        }
        return combined
    }
}
