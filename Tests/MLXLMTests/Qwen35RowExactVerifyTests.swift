// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Native-MTP target verification for Qwen 3.5 MoE (Ornith): every verify row's logits must be
// bitwise the logits of the single-token AR step at that position, or greedy MTP output drifts
// from greedy AR. Before the row-exact routes, an Ornith 122B 2-row verify differed at layer 0
// (8-bit in_proj_qkv / in_proj_z: MLX's multi-row qmv), then in the dense router and shared-expert
// gate, and every 3+-row verify differed in attention (rows x 16 query heads per KV head leaves
// MLX's vector SDPA kernel). Greedy MTP diverged from AR within ~50 tokens on code and prose.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXVLM

@Suite("Qwen3.5 MoE native-MTP verify is row-exact", .serialized)
struct Qwen35RowExactVerifyTests {

    /// Ornith-shaped in miniature: 8-bit affine trunk (the row-exact QMV admits K % 256 == 0),
    /// 16 query heads per KV head, GDN + full attention, routed + shared experts.
    private func model() throws -> Qwen35 {
        let config = try JSONDecoder.json5().decode(
            Qwen35Configuration.self,
            from: """
                {
                  "model_type": "qwen3_5_moe",
                  "text_config": {
                    "model_type": "qwen3_5_moe_text",
                    "num_experts": 8,
                    "num_experts_per_tok": 2,
                    "moe_intermediate_size": 256,
                    "shared_expert_intermediate_size": 256,
                    "hidden_size": 256,
                    "num_hidden_layers": 4,
                    "intermediate_size": 512,
                    "num_attention_heads": 32,
                    "num_key_value_heads": 2,
                    "linear_num_value_heads": 4,
                    "linear_num_key_heads": 2,
                    "linear_key_head_dim": 64,
                    "linear_value_head_dim": 64,
                    "linear_conv_kernel_dim": 4,
                    "head_dim": 64,
                    "full_attention_interval": 2,
                    "vocab_size": 512,
                    "tie_word_embeddings": false,
                    "rms_norm_eps": 1e-6,
                    "rope_parameters": {
                      "rope_type": "default",
                      "rope_theta": 100000.0,
                      "partial_rotary_factor": 0.25,
                      "mrope_section": [4, 2, 2]
                    }
                  },
                  "vocab_size": 512,
                  "image_token_id": 510,
                  "video_token_id": 509
                }
                """.data(using: .utf8)!)
        MLXRandom.seed(7)
        let model = Qwen35(config)
        quantize(model: model, groupSize: 64, bits: 8) { _, module in
            module is Linear || module is SwitchLinear
        }
        // BF16 activations, scales and norms like the shipped bundles (the row-exact routes are BF16).
        model.update(
            parameters: model.parameters().mapValues { $0.dtype == .float32 ? $0.asType(.bfloat16) : $0 })
        MLX.eval(model.parameters())
        return model
    }

    @Test("verify rows equal AR steps bitwise", arguments: [2, 3, 4])
    func verifyMatchesAutoregressive(width: Int) throws {
        let model = try model()
        let prompt = MLXArray((0 ..< 40).map { Int32(($0 * 37 + 11) % 500) }).reshaped(1, -1)
        let tokens = (0 ..< width).map { Int32(($0 * 53 + 3) % 500) }

        let base = model.newCache(parameters: GenerateParameters())
        MLX.eval(model.nativeBackboneForward(prompt, cache: base).logits)
        MLX.eval(base)

        let arCache = base.map { $0.copy() }
        var arRows: [MLXArray] = []
        for token in tokens {
            let out = model.nativeBackboneForward(
                MLXArray([token]).reshaped(1, 1), cache: arCache)
            let row = out.logits[0, -1].asType(.float32)
            MLX.eval(row)
            arRows.append(row)
        }

        let verifyCache = base.map { $0.copy() }
        let verified = NativeMTPVerifierStatePolicy.withVerifierMode("input_capture_staged") {
            model.nativeBackboneMTPVerifyForward(
                MLXArray(tokens).reshaped(1, width), cache: verifyCache)
        }
        let rows = verified.logits[0].asType(.float32)
        MLX.eval(rows)

        for (row, reference) in arRows.enumerated() {
            let diff = abs(rows[row] - reference).max().item(Float.self)
            #expect(diff == 0, "width \(width) row \(row): max |verify - AR| = \(diff)")
        }
    }
}
