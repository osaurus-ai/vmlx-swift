// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXLMCommon

extension Ernie55Model {
    static let fp8Quantization = BaseConfiguration.Quantization(groupSize: 32, bits: 8, mode: .mxfp8)

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        let replacements = [
            (".input_layernorm.", ".attn_norm."),
            (".post_attention_layernorm.", ".ffn_norm."),
            (".block_sparse_moe.", ".moe."),
            (".mlp.", ".dense."),
            (".core_attention.", "."),
            (".linear_q_down_proj.", ".wq_a."),
            (".linear_q_up_proj.", ".wq_b."),
            (".linear_kv_proj.", ".wkv."),
            (".q_layernorm.", ".q_norm."),
            (".kv_layernorm.", ".kv_norm."),
            (".compressor.linear_wkv.", ".compressor.wkv."),
            (".compressor.linear_wgate.", ".compressor.wgate."),
            (".o_proj.", ".wo_b."),
            (".vha_postmix_U", ".vha.U"),
            (".vha_postmix_V", ".vha.V"),
            (".softmax_offset", ".attn_sink"),
            (".shared_experts.w1.", ".shared_experts.gate_proj."),
            (".shared_experts.w2.", ".shared_experts.down_proj."),
            (".shared_experts.w3.", ".shared_experts.up_proj."),
        ]
        var result: [String: MLXArray] = [:]
        for (name, raw) in weights where !name.hasSuffix("_scale_inv") {
            // Skip optional upstream vLLM/SGLang FlashIndexer sidecar weights and inline MTP/DSpark weights.
            if name.hasPrefix("model.mtp.")
                || name.contains(".indexer.flashindexer_bias")
                || name.contains(".indexer.linear_gate.")
            {
                continue
            }
            var key = name
            var value = raw
            let parts = name.split(separator: ".")
            let layerIndex = parts.count > 3 && parts[0] == "model" && parts[1] == "layers"
                ? Int(parts[2]) : nil
            var fp8Scale: MLXArray?
            if let scale = weights[name + "_scale_inv"] {
                // Only kv_b_proj splits along output heads into embed_q and unembed_out.
                // Every other FP8 projection (including linear_o_group_proj and MLA linears)
                // stays packed in exact MXFP8 (-1.75 GB RAM and -1.75 GB/tok bandwidth).
                let dense = name.hasSuffix(".kv_b_proj.weight")
                let (converted, scaleBytes) = Self.unpackFP8(raw, scale: scale, keepPacked: !dense)
                value = converted
                fp8Scale = scaleBytes
            }
            if let layerIndex, layerIndex < args.numHiddenLayers {
                let isMLA = args.compressRatios[layerIndex] == -2
                let attention = isMLA ? ".mla." : ".hca."
                key = key.replacingOccurrences(of: ".self_attn.gate_proj.", with: attention + "o_gate.")
                key = key.replacingOccurrences(of: ".self_attn.", with: attention)
                key = key.replacingOccurrences(of: ".self_attention_hyper_connection.", with: ".attn_hc.")
                key = key.replacingOccurrences(of: ".mlp_hyper_connection.", with: ".ffn_hc.")
                if key.hasSuffix(".mapping_proj.weight") {
                    key = key.replacingOccurrences(of: ".mapping_proj.weight", with: ".fn")
                    value = contiguous(value.transposed().asType(.float32))
                } else if key.hasSuffix("_hc.bias") {
                    key = String(key.dropLast(4)) + "base"
                }
                for (from, to) in replacements {
                    key = key.replacingOccurrences(of: from, with: to)
                }
                if key.hasSuffix(".linear_o_group_proj") {
                    key = String(key.dropLast("linear_o_group_proj".count)) + "o_group_weight"
                    value = contiguous(value.reshaped(args.oGroups, args.oLoraRank, -1))
                    if let s = fp8Scale {
                        fp8Scale = contiguous(s.reshaped(args.oGroups, args.oLoraRank, -1))
                    } else {
                        fp8Scale = MLXArray.zeros([0], dtype: .uint8)
                    }
                }
                if isMLA, key.hasSuffix(".kv_b_proj.weight") {
                    let plain = args.hybridMlaQkNopeHeadDim
                    let width = plain + args.hybridMlaVHeadDim
                    let split = value.reshaped(args.hybridMlaNumAttentionHeads, width, args.hybridMlaKvLoraRank)
                    let prefix = String(key.dropLast("kv_b_proj.weight".count))
                    result[prefix + "embed_q_weight"] = contiguous(split[0..., ..<plain, 0...].swappedAxes(-1, -2))
                    result[prefix + "unembed_out_weight"] = contiguous(split[0..., plain..., 0...])
                    continue
                }
            } else if key.hasPrefix("model.mhc_contract.hc_head_") {
                key = key.replacingOccurrences(of: "model.mhc_contract.hc_head_", with: "model.hc_head.")
                if key.hasSuffix(".fn") { value = contiguous(value.transposed().asType(.float32)) }
                if key.hasSuffix(".scale") { value = value.reshaped(-1) }
            } else if key.hasPrefix("model.lm_head.") {
                key = String(key.dropFirst("model.lm_head.".count))
            }
            // Keep lm_head.weight in its native BF16 (-1.65 GB RAM and -1.65 GB/tok bandwidth);
            // Ernie55Kernels.lmHeadMultiMax reads BF16 directly and accumulates in FP32.
            result[key] = value
            if let fp8Scale {
                result[String(key.dropLast("weight".count)) + "scales"] = fp8Scale
            }
        }
        for index in 0..<args.numHiddenLayers {
            for kind in ["attn_hc", "ffn_hc"] {
                let path = "model.layers.\(index).\(kind)."
                let keys = ["alpha_pre", "alpha_post", "alpha_res"].map { path + $0 }
                if keys.allSatisfy({ result[$0] != nil }) {
                    result[path + "scale"] = concatenated(keys.map { result.removeValue(forKey: $0)!.reshaped(-1) })
                }
            }
        }
        return result
    }

    /// Exact E4M3/E8M0 repack for the released 128×128 FP8 linears.
    /// Non-power-of-two FP8 scales retain the original values in BF16.
    static func unpackFP8(
        _ codes: MLXArray, scale: MLXArray, keepPacked: Bool
    ) -> (weight: MLXArray, scaleBytes: MLXArray?) {
        let rows = codes.dim(0)
        let columns = codes.size / rows
        let raw = scale.asType(.float32).view(dtype: .uint32)
        let exponent = bitwiseAnd(rightShift(raw, 23), 255)
        if keepPacked, columns.isMultiple(of: 32), codes.ndim == 2,
            all((bitwiseAnd(raw, UInt32(0x807f_ffff)) .== 0) & (exponent .> 0) & (exponent .< 255)).item(Bool.self)
        {
            let packed = contiguous(codes).view(dtype: .uint32)
            let scales = repeated(repeated(exponent.asType(.uint8), count: 128, axis: 0), count: 4, axis: 1)
            return (packed, contiguous(scales[..<rows, ..<(columns / 32)]))
        }
        let scales = repeated(repeated(scale, count: 128, axis: 0), count: 128, axis: 1)
        let dense = fromFP8(codes.reshaped(rows, columns), dtype: .float32) * scales[..<rows, ..<columns]
        return (dense.asType(.bfloat16).reshaped(codes.shape), nil)
    }
}
