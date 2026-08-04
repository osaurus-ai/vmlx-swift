// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import Testing

/// After a forward pass of N tokens, EVERY cache layer must report
/// `offset == N` — the short-conv `MambaCache` layers included.
///
/// The CacheCoordinator's boundary-offset store guard (vmlx#208) refuses
/// any prefix-cache store whose cache offsets disagree with the prompt
/// boundary. LFM2/LFM2.5's conv blocks updated their conv window but never
/// advanced `cache.offset`, so all 22 short-conv layers sat at 0 forever
/// and the guard refused every paged AND disk store for the whole family —
/// observed live as zero `kv_v2` entries after full multiturn sessions
/// while the KV-layer offsets tracked the boundary correctly.
@Suite("LFM2 cache offset invariant", .serialized)
struct LFM2CacheOffsetInvariantTests {

    private func denseModel() throws -> LFM2Model {
        let config = """
            {
              "model_type": "lfm2",
              "vocab_size": 16,
              "hidden_size": 8,
              "num_hidden_layers": 2,
              "num_attention_heads": 2,
              "num_key_value_heads": 1,
              "max_position_embeddings": 128,
              "norm_eps": 0.00001,
              "conv_bias": false,
              "conv_L_cache": 3,
              "block_ff_dim": 16,
              "block_auto_adjust_ff_dim": false,
              "layer_types": ["conv", "full_attention"],
              "rope_theta": 10000000.0
            }
            """
        let configuration = try JSONDecoder().decode(
            LFM2Configuration.self,
            from: Data(config.utf8))
        return LFM2Model(configuration)
    }

    private func moeModel() throws -> LFM2MoEModel {
        let config = """
            {
              "model_type": "lfm2_moe",
              "vocab_size": 16,
              "hidden_size": 4,
              "intermediate_size": 8,
              "moe_intermediate_size": 2,
              "num_hidden_layers": 2,
              "num_experts": 2,
              "num_experts_per_tok": 1,
              "norm_topk_prob": true,
              "num_attention_heads": 1,
              "num_key_value_heads": 1,
              "max_position_embeddings": 128,
              "use_expert_bias": false,
              "num_dense_layers": 1,
              "norm_eps": 0.00001,
              "conv_bias": false,
              "conv_L_cache": 3,
              "layer_types": ["conv", "conv"]
            }
            """
        let configuration = try JSONDecoder().decode(
            LFM2MoEConfiguration.self,
            from: Data(config.utf8))
        return LFM2MoEModel(configuration)
    }

    @Test("dense LFM2 advances every cache layer's offset, conv included")
    func denseOffsetsTrackTokens() throws {
        let model = try denseModel()
        let cache = model.newCache(parameters: nil)
        let prompt = MLXArray([1, 2, 3, 4, 5]).reshaped(1, 5)
        _ = model(prompt, cache: cache)
        for (i, layer) in cache.enumerated() {
            #expect(
                layer.offset == 5,
                "layer \(i) (\(type(of: layer))) offset \(layer.offset) != 5 after 5-token prefill")
        }
        // Decode step: one more token everywhere.
        let step = MLXArray([6]).reshaped(1, 1)
        _ = model(step, cache: cache)
        for (i, layer) in cache.enumerated() {
            #expect(
                layer.offset == 6,
                "layer \(i) (\(type(of: layer))) offset \(layer.offset) != 6 after decode step")
        }
    }

    @Test("LFM2 MoE advances every cache layer's offset, conv included")
    func moeOffsetsTrackTokens() throws {
        let model = try moeModel()
        let cache = model.newCache(parameters: nil)
        let prompt = MLXArray([1, 2, 3]).reshaped(1, 3)
        _ = model(prompt, cache: cache)
        for (i, layer) in cache.enumerated() {
            #expect(
                layer.offset == 3,
                "layer \(i) (\(type(of: layer))) offset \(layer.offset) != 3 after 3-token prefill")
        }
    }
}
