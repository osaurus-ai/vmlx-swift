import Foundation
import MLX
import MLXLLM
import Testing

@Suite("LFM2 MoE legacy quantized weight sanitization")
struct LFM2MoESanitizeTests {
    private func model() throws -> LFM2MoEModel {
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

    private func scalar(_ value: Float) -> MLXArray {
        MLXArray([value])
    }

    @Test("dense w1/w2/w3 quantization sidecars follow renamed modules")
    func denseQuantizationSidecarsAreRenamed() throws {
        try MLXMetalTestLock.withLock {
            let weights = [
                "model.layers.0.feed_forward.w1.weight": scalar(1),
                "model.layers.0.feed_forward.w1.scales": scalar(2),
                "model.layers.0.feed_forward.w1.biases": scalar(3),
                "model.layers.0.feed_forward.w2.weight": scalar(4),
                "model.layers.0.feed_forward.w2.scales": scalar(5),
                "model.layers.0.feed_forward.w3.weight": scalar(6),
                "model.layers.0.feed_forward.w3.scales": scalar(7),
            ]

            let sanitized = try model().sanitize(weights: weights)

            #expect(sanitized["model.layers.0.feed_forward.gate_proj.weight"] != nil)
            #expect(sanitized["model.layers.0.feed_forward.gate_proj.scales"] != nil)
            #expect(sanitized["model.layers.0.feed_forward.gate_proj.biases"] != nil)
            #expect(sanitized["model.layers.0.feed_forward.down_proj.weight"] != nil)
            #expect(sanitized["model.layers.0.feed_forward.down_proj.scales"] != nil)
            #expect(sanitized["model.layers.0.feed_forward.up_proj.weight"] != nil)
            #expect(sanitized["model.layers.0.feed_forward.up_proj.scales"] != nil)
            #expect(sanitized.keys.allSatisfy { !$0.contains(".w1.") })
            #expect(sanitized.keys.allSatisfy { !$0.contains(".w2.") })
            #expect(sanitized.keys.allSatisfy { !$0.contains(".w3.") })
        }
    }

    @Test("per-expert quantization sidecars stack with expert weights")
    func expertQuantizationSidecarsAreStacked() throws {
        try MLXMetalTestLock.withLock {
            var weights: [String: MLXArray] = [:]
            for expert in 0 ..< 2 {
                for (projection, base) in [("w1", 10), ("w2", 20), ("w3", 30)] {
                    let prefix =
                        "model.layers.1.feed_forward.experts.\(expert).\(projection)"
                    weights["\(prefix).weight"] = scalar(Float(base + expert))
                    weights["\(prefix).scales"] = scalar(Float(base + expert + 2))
                    weights["\(prefix).biases"] = scalar(Float(base + expert + 4))
                }
            }

            let sanitized = try model().sanitize(weights: weights)

            for (projection, base) in [
                ("gate_proj", 10),
                ("down_proj", 20),
                ("up_proj", 30),
            ] {
                for (leaf, offset) in [
                    ("weight", 0),
                    ("scales", 2),
                    ("biases", 4),
                ] {
                    let key =
                        "model.layers.1.feed_forward.switch_mlp.\(projection).\(leaf)"
                    #expect(sanitized[key]?.shape == [2, 1])
                    #expect(
                        sanitized[key]?.asArray(Float.self)
                            == [Float(base + offset), Float(base + offset + 1)])
                }
            }
            #expect(sanitized.keys.allSatisfy { !$0.contains(".experts.") })
        }
    }
}
