// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// `LanguageModel.recommendedPrefillStepSize`: routed-MoE Qwen 3.5 backbones
// (Ornith, Qwen3.6-35B) recommend 2048-token prefill chunks; dense backbones
// and other architectures keep the library default (nil).

import Foundation
import MLXLMCommon
import Testing

@testable import MLXVLM

@Suite("Qwen3.5 recommended prefill step")
struct Qwen35RecommendedPrefillStepTests {

    private func configuration(moe: Bool) throws -> Qwen35Configuration {
        let moeFields =
            moe
            ? """
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 16,
            "shared_expert_intermediate_size": 16,
            """
            : ""
        return try JSONDecoder.json5().decode(
            Qwen35Configuration.self,
            from: """
                {
                  "model_type": "\(moe ? "qwen3_5_moe" : "qwen3_5")",
                  "text_config": {
                    "model_type": "\(moe ? "qwen3_5_moe_text" : "qwen3_5_text")",
                    \(moeFields)
                    "hidden_size": 32,
                    "num_hidden_layers": 4,
                    "intermediate_size": 64,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 2,
                    "linear_num_value_heads": 4,
                    "linear_num_key_heads": 2,
                    "linear_key_head_dim": 8,
                    "linear_value_head_dim": 8,
                    "linear_conv_kernel_dim": 2,
                    "head_dim": 8,
                    "full_attention_interval": 4,
                    "vocab_size": 100,
                    "tie_word_embeddings": false,
                    "rms_norm_eps": 1e-6,
                    "rope_parameters": {
                      "rope_type": "default",
                      "rope_theta": 100000.0,
                      "partial_rotary_factor": 0.25,
                      "mrope_section": [1, 1, 1]
                    }
                  },
                  "vocab_size": 100,
                  "image_token_id": 98,
                  "video_token_id": 97
                }
                """.data(using: .utf8)!)
    }

    @Test("routed-MoE backbone recommends 2048-token prefill chunks")
    func moeRecommends2048() throws {
        let config = try configuration(moe: true)
        #expect(config.textConfiguration.numExperts == 4)
        let model = Qwen35(config)
        #expect(model.recommendedPrefillStepSize == 2048)
    }

    @Test("dense backbone keeps the library default")
    func denseKeepsDefault() throws {
        let model = Qwen35(try configuration(moe: false))
        #expect(model.recommendedPrefillStepSize == nil)
    }
}
