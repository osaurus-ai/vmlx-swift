// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// `Nemotron-Labs-Audex-30B-A3B-*` declares `model_type = "nemotron_h_audex"`,
/// which no dispatch entry claimed, so selecting it in the app failed outright
/// with `Unsupported model type: nemotron_h_audex` — the text tower was
/// unreachable even though this runtime can drive it.
///
/// Its text tower is byte-for-byte the Nemotron-H shape Lightning uses: 52
/// layers, hidden 2688, 32 heads / 2 KV heads, intermediate 1856, n_groups 8,
/// ssm_state 128, 64 mamba heads at head_dim 64. Only the vocab differs (larger,
/// for the sound tokens) plus an `audio_encoder` / `audio_projector` tower that
/// this runtime has no module for.
@Suite("Nemotron Audex dispatch")
struct NemotronAudexDispatchTests {

    /// The real 30B-A3B Audex text-tower shape, with the audio keys present.
    private var audexConfig: Data {
        #"""
        {
          "architectures": [
            "NemotronHAudexForConditionalGeneration"
          ],
          "attention_bias": false,
          "attention_dropout": 0.0,
          "audio_encoder_hidden_size": 1280,
          "audio_model_type": "NV-Whisper",
          "audio_projector_activation": "relu2",
          "audio_projector_intermediate_size": 4096,
          "audio_projector_norm_eps": 1e-05,
          "bos_token_id": 1,
          "chunk_size": 128,
          "conv_kernel": 4,
          "dtype": "bfloat16",
          "eos_token_id": 11,
          "expand": 2,
          "head_dim": 128,
          "hidden_dropout": 0.0,
          "hidden_size": 2688,
          "hybrid_override_pattern": "MEMEM*EMEMEM",
          "initializer_range": 0.02,
          "intermediate_size": 1856,
          "layer_norm_epsilon": 1e-05,
          "mamba_head_dim": 64,
          "mamba_hidden_act": "silu",
          "mamba_num_heads": 64,
          "mamba_proj_bias": false,
          "max_position_embeddings": 262144,
          "mlp_bias": false,
          "mlp_hidden_act": "relu2",
          "model_type": "nemotron_h_audex",
          "moe_intermediate_size": 1856,
          "moe_shared_expert_intermediate_size": 3712,
          "n_group": 1,
          "n_groups": 8,
          "n_routed_experts": 128,
          "n_shared_experts": 1,
          "norm_eps": 1e-05,
          "norm_topk_prob": true,
          "num_attention_heads": 32,
          "num_experts_per_tok": 6,
          "num_hidden_layers": 12,
          "num_key_value_heads": 2,
          "num_logits_to_keep": 1,
          "pad_token_id": 0,
          "partial_rotary_factor": 1.0,
          "rescale_prenorm_residual": true,
          "residual_in_fp32": false,
          "rope_theta": 10000,
          "routed_scaling_factor": 2.5,
          "sliding_window": null,
          "sound_clip_duration": 30.0,
          "sound_embedding_size": 750,
          "sound_end_token": "<so_end>",
          "sound_end_token_id": 31,
          "sound_model_type": "hf:///nv-whisper",
          "sound_start_token": "<so_start>",
          "sound_start_token_id": 30,
          "sound_target_rate": 16000,
          "sound_token": "<so_embedding>",
          "sound_token_id": 29,
          "ssm_state_size": 128,
          "tie_word_embeddings": false,
          "time_step_floor": 0.0001,
          "time_step_max": 0.1,
          "time_step_min": 0.001,
          "topk_group": 1,
          "use_bias": false,
          "use_cache": true,
          "use_conv_bias": true,
          "use_mamba_kernels": true,
          "vocab_size": 205312
        }
        """#.data(using: .utf8)!
    }

    @Test("a nemotron_h_audex bundle resolves to a model instead of throwing")
    func audexDispatches() async throws {
        let model = try await LLMTypeRegistry.shared.createModel(configuration: audexConfig, modelType: "nemotron_h_audex")
        #expect(model is NemotronHModel, "expected the Nemotron-H text tower")
    }

    /// The audio tower has no module to bind to, so those weights must be
    /// dropped rather than failing `verify`. The prefixes are
    /// `audio_encoder.` / `audio_projector.` — NOT the `sound_*` spelling the
    /// scrub list already had, which is why they needed adding.
    @Test("the audio tower is dropped and the text tower survives")
    func audioWeightsAreScrubbed() async throws {
        let model = try await LLMTypeRegistry.shared.createModel(configuration: audexConfig, modelType: "nemotron_h_audex")
        let nemotron = try #require(model as? NemotronHModel)

        let weights: [String: MLXArray] = [
            "backbone.embeddings.weight": MLXArray.zeros([8, 4]),
            "lm_head.weight": MLXArray.zeros([8, 4]),
            "audio_encoder.layers.0.self_attn.q_proj.weight": MLXArray.zeros([4, 4]),
            "audio_encoder.conv1.weight": MLXArray.zeros([4, 4]),
            "audio_projector.linear_1.weight": MLXArray.zeros([4, 4]),
        ]
        let sanitized = nemotron.sanitize(weights: weights)

        #expect(sanitized.keys.contains { $0.hasPrefix("backbone.") })
        #expect(!sanitized.keys.contains { $0.hasPrefix("audio_encoder.") })
        #expect(!sanitized.keys.contains { $0.hasPrefix("audio_projector.") })
    }

    /// The plain `nemotron_h` route must keep working unchanged.
    @Test("nemotron_h still dispatches")
    func nemotronHStillDispatches() async throws {
        let config = #"""
            {
              "architectures": [
                "NemotronHForCausalLM"
              ],
              "attention_bias": false,
              "attention_dropout": 0.0,
              "bos_token_id": 1,
              "chunk_size": 128,
              "conv_kernel": 4,
              "dtype": "bfloat16",
              "eos_token_id": 11,
              "expand": 2,
              "head_dim": 128,
              "hidden_dropout": 0.0,
              "hidden_size": 2688,
              "hybrid_override_pattern": "MEMEM*EMEMEM",
              "initializer_range": 0.02,
              "intermediate_size": 1856,
              "layer_norm_epsilon": 1e-05,
              "mamba_head_dim": 64,
              "mamba_hidden_act": "silu",
              "mamba_num_heads": 64,
              "mamba_proj_bias": false,
              "max_position_embeddings": 262144,
              "mlp_bias": false,
              "mlp_hidden_act": "relu2",
              "model_type": "nemotron_h",
              "moe_intermediate_size": 1856,
              "moe_shared_expert_intermediate_size": 3712,
              "n_group": 1,
              "n_groups": 8,
              "n_routed_experts": 128,
              "n_shared_experts": 1,
              "norm_eps": 1e-05,
              "norm_topk_prob": true,
              "num_attention_heads": 32,
              "num_experts_per_tok": 6,
              "num_hidden_layers": 12,
              "num_key_value_heads": 2,
              "num_logits_to_keep": 1,
              "pad_token_id": 0,
              "partial_rotary_factor": 1.0,
              "rescale_prenorm_residual": true,
              "residual_in_fp32": false,
              "rope_theta": 10000,
              "routed_scaling_factor": 2.5,
              "sliding_window": null,
              "ssm_state_size": 128,
              "tie_word_embeddings": false,
              "time_step_floor": 0.0001,
              "time_step_max": 0.1,
              "time_step_min": 0.001,
              "topk_group": 1,
              "use_bias": false,
              "use_cache": true,
              "use_conv_bias": true,
              "use_mamba_kernels": true,
              "vocab_size": 131072
            }
            """#.data(using: .utf8)!
        let m = try await LLMTypeRegistry.shared.createModel(configuration: config, modelType: "nemotron_h")
        #expect(m is NemotronHModel)
    }

    /// The 2B Audex bundles are `nemotron_dense_audex` — a DENSE architecture
    /// with no hybrid pattern and no mamba parameters. It is deliberately NOT
    /// routed here; claiming it would load a mismatched model rather than fail
    /// honestly.
    @Test("the dense Audex variant is still refused")
    func denseAudexStaysUnsupported() async {
        let config = #"""
            {"model_type": "nemotron_dense_audex", "num_hidden_layers": 28, "hidden_size": 2048}
            """#.data(using: .utf8)!
        await #expect(throws: (any Error).self) {
            _ = try await LLMTypeRegistry.shared.createModel(
                configuration: config, modelType: "nemotron_dense_audex")
        }
    }
}
