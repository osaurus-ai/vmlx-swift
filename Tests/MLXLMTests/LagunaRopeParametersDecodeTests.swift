// Copyright © 2026 osaurus.
//
// Regression coverage for `LagunaConfiguration`'s `rope_parameters` decode.
// Real Laguna bundles ship a mixed-shape object — a top-level scalar
// `original_max_position_embeddings` sits alongside the per-layer-type
// sub-dicts (`full_attention`, `sliding_attention`). Decoding strictly as
// `[String: [String: StringOrNumber]]` fails on the scalar; we filter to
// dict-valued entries instead so loading proceeds.

import Foundation
import MLX
@testable import MLXLLM
@testable import MLXLMCommon
import Testing

@Suite("Laguna rope_parameters — mixed-shape decode")
struct LagunaRopeParametersDecodeTests {

    private static let s21TinyConfig = #"""
    {
      "model_type": "laguna",
      "hidden_size": 8,
      "intermediate_size": 16,
      "num_hidden_layers": 2,
      "num_attention_heads": 2,
      "num_key_value_heads": 1,
      "num_attention_heads_per_layer": [2, 4],
      "head_dim": 4,
      "max_position_embeddings": 64,
      "vocab_size": 32,
      "rms_norm_eps": 1.0e-5,
      "tie_word_embeddings": true,
      "layer_types": ["full_attention", "sliding_attention"],
      "mlp_layer_types": ["dense", "sparse"],
      "sliding_window": 4,
      "moe_intermediate_size": 4,
      "shared_expert_intermediate_size": 4,
      "num_experts": 4,
      "num_experts_per_tok": 2,
      "gating": "per-head",
      "rope_parameters": {
        "full_attention": {
          "rope_theta": 500000.0,
          "rope_type": "default",
          "partial_rotary_factor": 0.5
        },
        "sliding_attention": {
          "rope_theta": 10000.0,
          "rope_type": "default",
          "partial_rotary_factor": 1.0
        }
      }
    }
    """#

    private func s21TinyModel() throws -> LagunaModel {
        let cfg = try JSONDecoder().decode(
            LagunaConfiguration.self, from: Data(Self.s21TinyConfig.utf8))
        return LagunaModel(cfg, jangtq: nil)
    }

    /// Minimum-viable Laguna config covering all required fields plus the
    /// problematic `rope_parameters` shape. Only the rope decode is
    /// asserted — the rest of the config is supplied so JSON decoding
    /// doesn't fail on missing keys.
    private static let configWithMixedRope = #"""
    {
      "model_type": "laguna",
      "hidden_size": 64,
      "intermediate_size": 128,
      "num_hidden_layers": 2,
      "num_attention_heads": 4,
      "num_key_value_heads": 2,
      "head_dim": 16,
      "max_position_embeddings": 4096,
      "vocab_size": 1024,
      "rms_norm_eps": 1.0e-5,
      "tie_word_embeddings": true,
      "layer_types": ["sliding_attention", "full_attention"],
      "moe_intermediate_size": 64,
      "num_experts_per_tok": 2,
      "num_local_experts": 4,
      "num_shared_experts": 1,
      "use_qk_norm": true,
      "rope_parameters": {
        "full_attention": {
          "rope_theta": 500000.0,
          "rope_type": "default"
        },
        "sliding_attention": {
          "rope_theta": 500000.0,
          "rope_type": "default"
        },
        "original_max_position_embeddings": 4096
      }
    }
    """#

    @Test("Mixed-shape rope_parameters decodes without throwing")
    func decodeMixedShape() throws {
        let data = Self.configWithMixedRope.data(using: .utf8)!
        // The bug: this decode used to throw `Type mismatch at
        // rope_parameters.original_max_position_embeddings`.
        let cfg = try JSONDecoder().decode(LagunaConfiguration.self, from: data)
        // The two per-layer-type sub-dicts must survive; the top-level
        // scalar is dropped (it's available via `maxPositionEmbeddings`).
        #expect(cfg.ropeParameters.keys.contains("full_attention"))
        #expect(cfg.ropeParameters.keys.contains("sliding_attention"))
        #expect(!cfg.ropeParameters.keys.contains("original_max_position_embeddings"))
    }

    /// Pure (no top-level scalar) shape must continue to decode — the
    /// permissive path mustn't regress the strict shape for older bundles.
    @Test("Pure-dict rope_parameters still decodes")
    func decodePureDictShape() throws {
        let json = #"""
        {
          "model_type": "laguna",
          "hidden_size": 64,
          "intermediate_size": 128,
          "num_hidden_layers": 1,
          "num_attention_heads": 4,
          "num_key_value_heads": 2,
          "head_dim": 16,
          "max_position_embeddings": 4096,
          "vocab_size": 1024,
          "rms_norm_eps": 1.0e-5,
          "tie_word_embeddings": true,
          "layer_types": ["full_attention"],
          "moe_intermediate_size": 64,
          "num_experts_per_tok": 2,
          "num_local_experts": 4,
          "num_shared_experts": 1,
          "use_qk_norm": true,
          "rope_parameters": {
            "full_attention": {
              "rope_theta": 500000.0,
              "rope_type": "default"
            }
          }
        }
        """#
        let cfg = try JSONDecoder().decode(LagunaConfiguration.self, from: json.data(using: .utf8)!)
        #expect(cfg.ropeParameters.keys.contains("full_attention"))
    }

    /// Missing rope_parameters entirely → empty dict, no crash.
    @Test("Absent rope_parameters → empty dict")
    func decodeAbsentRopeParameters() throws {
        let json = #"""
        {
          "model_type": "laguna",
          "hidden_size": 64,
          "intermediate_size": 128,
          "num_hidden_layers": 1,
          "num_attention_heads": 4,
          "num_key_value_heads": 2,
          "head_dim": 16,
          "max_position_embeddings": 4096,
          "vocab_size": 1024,
          "rms_norm_eps": 1.0e-5,
          "tie_word_embeddings": true,
          "layer_types": ["full_attention"],
          "moe_intermediate_size": 64,
          "num_experts_per_tok": 2,
          "num_local_experts": 4,
          "num_shared_experts": 1,
          "use_qk_norm": true
        }
        """#
        let cfg = try JSONDecoder().decode(LagunaConfiguration.self, from: json.data(using: .utf8)!)
        #expect(cfg.ropeParameters.isEmpty)
    }

    @Test("sanitize fuses Laguna attention q/k/v into qkv_proj")
    func sanitizeFusesAttentionQKV() throws {
        let cfg = try JSONDecoder().decode(
            LagunaConfiguration.self,
            from: #"""
            {
              "model_type": "laguna",
              "hidden_size": 4,
              "intermediate_size": 8,
              "num_hidden_layers": 1,
              "num_attention_heads": 2,
              "num_key_value_heads": 1,
              "head_dim": 2,
              "max_position_embeddings": 64,
              "vocab_size": 16,
              "rms_norm_eps": 1.0e-5,
              "tie_word_embeddings": true,
              "layer_types": ["full_attention"],
              "mlp_layer_types": ["dense"],
              "num_attention_heads_per_layer": [2],
              "moe_intermediate_size": 4,
              "shared_expert_intermediate_size": 4,
              "num_experts": 4,
              "num_experts_per_tok": 2
            }
            """#.data(using: .utf8)!)

        let model = LagunaModel(cfg, jangtq: nil)
        let prefix = "model.layers.0.self_attn"
        let weights: [String: MLXArray] = [
            "\(prefix).q_proj.weight": MLXArray.ones([4, 3]),
            "\(prefix).k_proj.weight": MLXArray.ones([2, 3]) * 2,
            "\(prefix).v_proj.weight": MLXArray.ones([2, 3]) * 3,
            "\(prefix).q_proj.scales": MLXArray.ones([4, 1]),
            "\(prefix).k_proj.scales": MLXArray.ones([2, 1]) * 2,
            "\(prefix).v_proj.scales": MLXArray.ones([2, 1]) * 3,
            "\(prefix).q_proj.biases": MLXArray.ones([4, 1]),
            "\(prefix).k_proj.biases": MLXArray.ones([2, 1]) * 2,
            "\(prefix).v_proj.biases": MLXArray.ones([2, 1]) * 3,
        ]

        let sanitized = model.sanitize(weights: weights)
        #expect(sanitized["layers.0.self_attn.qkv_proj.weight"]?.shape == [8, 3])
        #expect(sanitized["layers.0.self_attn.qkv_proj.scales"]?.shape == [8, 1])
        #expect(sanitized["layers.0.self_attn.qkv_proj.biases"]?.shape == [8, 1])
        #expect(sanitized["layers.0.self_attn.q_proj.weight"] == nil)
        #expect(sanitized["layers.0.self_attn.k_proj.weight"] == nil)
        #expect(sanitized["layers.0.self_attn.v_proj.weight"] == nil)
    }

    @Test("S-2.1 uses released switch_mlp module path")
    func s21UsesSwitchMLPPath() throws {
        let model = try s21TinyModel()
        let leaves = Dictionary(uniqueKeysWithValues: model.leafModules().flattened())

        #expect(leaves["layers.1.mlp.switch_mlp.gate_proj"] is SwitchLinear)
        #expect(leaves["layers.1.mlp.switch_mlp.up_proj"] is SwitchLinear)
        #expect(leaves["layers.1.mlp.switch_mlp.down_proj"] is SwitchLinear)
        #expect(leaves["layers.1.mlp.experts.gate_proj"] == nil)
    }

    @Test("sanitize preserves S-2.1 switch_mlp and normalizes legacy experts")
    func sanitizeNormalizesRoutedExpertPaths() throws {
        let model = try s21TinyModel()
        let current = model.sanitize(weights: [
            "model.layers.1.mlp.switch_mlp.gate_proj.weight": MLXArray.ones([4, 4, 2]),
            "model.layers.1.mlp.switch_mlp.up_proj.weight": MLXArray.ones([4, 4, 2]),
            "model.layers.1.mlp.switch_mlp.down_proj.weight": MLXArray.ones([4, 8, 1]),
        ])
        #expect(current["layers.1.mlp.switch_mlp.gate_proj.weight"] != nil)
        #expect(current["layers.1.mlp.switch_mlp.up_proj.weight"] != nil)
        #expect(current["layers.1.mlp.switch_mlp.down_proj.weight"] != nil)

        let legacy = model.sanitize(weights: [
            "model.layers.1.mlp.experts.gate_up_proj.weight": MLXArray.ones([4, 8, 2]),
            "model.layers.1.mlp.experts.down_proj.weight": MLXArray.ones([4, 8, 1]),
            "model.layers.1.mlp.experts.e_score_correction_bias": MLXArray.ones([4]),
        ])
        #expect(legacy["layers.1.mlp.switch_mlp.gate_proj.weight"]?.shape == [4, 4, 2])
        #expect(legacy["layers.1.mlp.switch_mlp.up_proj.weight"]?.shape == [4, 4, 2])
        #expect(legacy["layers.1.mlp.switch_mlp.down_proj.weight"] != nil)
        #expect(legacy["layers.1.mlp.e_score_correction_bias"] != nil)
    }

    @Test("S-2.1 no-cache prefill uses a banded SWA mask")
    func s21NoCacheUsesSlidingMaskPastWindow() throws {
        let model = try s21TinyModel()
        let masks = model.attentionMasks(
            h: MLXArray.zeros([1, 9, 8]), cache: nil)

        if case .causal = masks.full {
            // expected
        } else {
            Issue.record("full-attention prefill must use the causal fast path")
        }
        if case .array(let mask) = masks.sliding {
            #expect(mask.shape == [9, 9])
        } else {
            Issue.record("SWA prefill longer than the window must use a band mask")
        }
    }

    @Test("S-2.1 cache topology is full KV plus keep-zero rotating SWA")
    func s21CacheTopology() throws {
        let model = try s21TinyModel()
        for parameters in [
            nil,
            GenerateParameters(kvMode: .turboQuant(keyBits: 4, valueBits: 4)),
        ] {
            let cache = model.newCache(parameters: parameters)
            #expect(cache.count == 2)
            #expect(cache[0] is KVCacheSimple)
            let sliding = try #require(cache[1] as? RotatingKVCache)
            #expect(sliding.maxSize == 4)
            #expect(sliding.keep == 0)
        }
    }

    @Test("S-2.1 per-module affine widths retain 2/3/6/8-bit declarations")
    func s21PerModuleQuantizationWidths() throws {
        let json = #"""
        {
          "model_type": "laguna",
          "quantization": {
            "bits": 8,
            "group_size": 64,
            "model.embed_tokens": {"bits": 6},
            "model.layers.1.mlp.switch_mlp.gate_proj": {"bits": 2},
            "model.layers.1.mlp.switch_mlp.down_proj": {"bits": 3},
            "model.layers.1.self_attn.q_proj": {"bits": 8}
          }
        }
        """#
        let base = try JSONDecoder.json5().decode(
            BaseConfiguration.self, from: Data(json.utf8))
        let perLayer = try #require(base.quantizationContainer?.perLayerQuantization)

        #expect(perLayer.quantization(layer: "model.embed_tokens")?.bits == 6)
        #expect(perLayer.quantization(
            layer: "model.layers.1.mlp.switch_mlp.gate_proj")?.bits == 2)
        #expect(perLayer.quantization(
            layer: "model.layers.1.mlp.switch_mlp.down_proj")?.bits == 3)
        #expect(perLayer.quantization(
            layer: "model.layers.1.self_attn.q_proj")?.bits == 8)
        #expect(perLayer.quantization(
            layer: "model.layers.1.mlp.switch_mlp.up_proj")?.bits == 8)
    }

    @Test("Laguna thinking default is declarative and request-overridable")
    func lagunaThinkingDefaultComesFromBundleMetadata() throws {
        let caps = JangCapabilities(
            thinkInTemplate: true, supportsThinking: true, family: "laguna")
        let jangChat = JangChatConfig(
            templateKwargsDefaults: ChatTemplateKwargsDefaults(enableThinking: true))

        let jangFallback = try #require(llmDefaultAdditionalContext(
            modelType: "laguna", capabilities: caps,
            generationConfig: nil, chatConfig: jangChat))
        #expect(jangFallback["enable_thinking"] as? Bool == true)

        let generationWins = try #require(llmDefaultAdditionalContext(
            modelType: "laguna", capabilities: caps,
            generationConfig: GenerationConfigFile(
                defaultChatTemplateKwargs: ChatTemplateKwargsDefaults(
                    enableThinking: false)),
            chatConfig: jangChat))
        #expect(generationWins["enable_thinking"] as? Bool == false)

        var requestMerged = generationWins
        requestMerged["enable_thinking"] = true
        #expect(requestMerged["enable_thinking"] as? Bool == true)

        let nonThinking = try #require(llmDefaultAdditionalContext(
            modelType: "example", capabilities: JangCapabilities(
                supportsThinking: false),
            generationConfig: GenerationConfigFile(
                defaultChatTemplateKwargs: ChatTemplateKwargsDefaults(
                    enableThinking: true)),
            chatConfig: nil))
        #expect(nonThinking["enable_thinking"] as? Bool == false)
    }
}
