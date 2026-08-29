import Foundation
import MLX
import MLXNN
@testable import MLXVLM
import Testing

@Suite("Qwen3.5-VL MTP MoE weight sanitizer", .serialized)
struct Qwen35VLMMTPMoESanitizerTests {
    private struct SafetensorsIndex: Decodable {
        let weightMap: [String: String]

        enum CodingKeys: String, CodingKey {
            case weightMap = "weight_map"
        }
    }

    private static let ornith35Path = FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent("models/dealign.ai/Ornith-1.5-35B-A3B-UNCENSORED-JANG_4M")

    private func tinyConfiguration() throws -> Qwen35Configuration {
        try JSONDecoder.json5().decode(
            Qwen35Configuration.self,
            from: """
            {
              "model_type": "qwen3_5_moe",
              "text_config": {
                "model_type": "qwen3_5_moe_text",
                "hidden_size": 32,
                "num_hidden_layers": 1,
                "intermediate_size": 64,
                "moe_intermediate_size": 32,
                "shared_expert_intermediate_size": 32,
                "num_experts": 2,
                "num_experts_per_tok": 1,
                "mtp_num_hidden_layers": 1,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "linear_num_value_heads": 4,
                "linear_num_key_heads": 2,
                "linear_key_head_dim": 8,
                "linear_value_head_dim": 8,
                "linear_conv_kernel_dim": 2,
                "head_dim": 8,
                "full_attention_interval": 1,
                "vocab_size": 100,
                "rms_norm_eps": 1e-6,
                "rope_parameters": {
                  "rope_type": "default",
                  "rope_theta": 100000.0,
                  "partial_rotary_factor": 0.25,
                  "mrope_section": [1, 1, 1]
                }
              },
              "vision_config": {
                "model_type": "qwen3_vl",
                "depth": 1,
                "hidden_size": 16,
                "intermediate_size": 32,
                "out_hidden_size": 32,
                "num_heads": 4,
                "patch_size": 2,
                "spatial_merge_size": 2,
                "temporal_patch_size": 1,
                "num_position_embeddings": 32
              },
              "vocab_size": 100,
              "image_token_id": 98,
              "video_token_id": 97
            }
            """.data(using: .utf8)!)
    }

    @Test("per-expert affine MTP tensors become stacked switch tensors")
    func perExpertAffineMTPTensorsBecomeStackedSwitchTensors() throws {
        let model = Qwen35MoE(try tinyConfiguration())
        var weights: [String: MLXArray] = [:]
        for expert in 0 ..< 2 {
            for projection in ["gate_proj", "up_proj", "down_proj"] {
                let base = "mtp.layers.0.mlp.experts.\(expert).\(projection)"
                weights["\(base).weight"] = MLXArray.zeros([8, 4], dtype: .uint32)
                weights["\(base).scales"] = MLXArray.zeros([8, 1])
                weights["\(base).biases"] = MLXArray.zeros([8, 1])
            }
        }

        let sanitized = model.sanitize(weights: weights)

        for projection in ["gate_proj", "up_proj", "down_proj"] {
            for suffix in ["weight", "scales", "biases"] {
                let target =
                    "language_model.mtp.layers.0.mlp.switch_mlp.\(projection).\(suffix)"
                #expect(sanitized[target]?.shape.first == 2, "Missing stacked target \(target)")
            }
        }
        #expect(!sanitized.keys.contains { $0.contains(".mlp.experts.") })
    }

    @Test("incomplete per-expert MTP payload remains visible to strict update")
    func incompletePerExpertMTPPayloadFailsClosed() throws {
        let model = Qwen35MoE(try tinyConfiguration())
        let source = "mtp.layers.0.mlp.experts.0.gate_proj.weight"
        let sanitized = model.sanitize(weights: [
            source: MLXArray.zeros([8, 4], dtype: .uint32),
        ])

        #expect(sanitized["language_model.\(source)"] != nil)
        #expect(sanitized[
            "language_model.mtp.layers.0.mlp.switch_mlp.gate_proj.weight"
        ] == nil)
    }

    @Test("pre-stacked MTP switch tensors are idempotent")
    func preStackedMTPSwitchTensorsRemainUnchanged() throws {
        let model = Qwen35MoE(try tinyConfiguration())
        let source = "mtp.layers.0.mlp.switch_mlp.gate_proj.weight"
        let value = MLXArray.zeros([2, 8, 4], dtype: .uint32)
        let sanitized = model.sanitize(weights: [source: value])

        #expect(sanitized["language_model.\(source)"]?.shape == [2, 8, 4])
    }

    @Test("per-expert affine MTP tensors bind and execute a quantized draft forward")
    func perExpertAffineMTPBindsAndForwards() throws {
        let model = Qwen35MoE(try tinyConfiguration())
        quantize(
            model: model, groupSize: 32, bits: 4,
            filter: { path, _ in
                path.contains("language_model.mtp.layers.0.mlp.switch_mlp")
            })

        var checkpoint: [String: MLXArray] = [:]
        for expert in 0 ..< 2 {
            for projection in ["gate_proj", "up_proj"] {
                let (weight, scales, biases) = MLX.quantized(
                    MLXArray.ones([32, 32]), groupSize: 32, bits: 4)
                let base = "mtp.layers.0.mlp.experts.\(expert).\(projection)"
                checkpoint["\(base).weight"] = weight
                checkpoint["\(base).scales"] = scales
                checkpoint["\(base).biases"] = biases
            }
            let (weight, scales, biases) = MLX.quantized(
                MLXArray.ones([32, 32]), groupSize: 32, bits: 4)
            let base = "mtp.layers.0.mlp.experts.\(expert).down_proj"
            checkpoint["\(base).weight"] = weight
            checkpoint["\(base).scales"] = scales
            checkpoint["\(base).biases"] = biases
        }

        var fullParameters = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        for (key, value) in model.sanitize(weights: checkpoint) {
            fullParameters[key] = value
        }
        try model.update(
            parameters: ModuleParameters.unflattened(fullParameters),
            verify: .all)

        let result = model.nativeMTPForward(
            hiddenStates: MLXArray.zeros([1, 1, 32]),
            nextTokenIds: MLXArray([Int32(1)])[.newAxis, .ellipsis],
            cache: model.makeNativeMTPCache())
        eval(result.logits, result.hiddenStates)

        #expect(result.logits.shape == [1, 1, 100])
        #expect(result.hiddenStates.shape == [1, 1, 32])
        #expect(result.logits.asType(.float32).asArray(Float.self).allSatisfy { $0.isFinite })
    }

    @Test(
        "real Ornith-35B index exposes complete 256-expert affine MTP topology",
        .enabled(if: FileManager.default.fileExists(
            atPath: ornith35Path.appendingPathComponent("model.safetensors.index.json").path)))
    func realOrnith35IndexPinsMTPLoadShape() throws {
        let index = try JSONDecoder().decode(
            SafetensorsIndex.self,
            from: Data(contentsOf: Self.ornith35Path
                .appendingPathComponent("model.safetensors.index.json")))
        let expertKeys = index.weightMap.keys.filter {
            $0.hasPrefix("mtp.layers.0.mlp.experts.")
        }

        #expect(expertKeys.count == 256 * 3 * 3)
        for expert in 0 ..< 256 {
            for projection in ["gate_proj", "up_proj", "down_proj"] {
                for suffix in ["weight", "scales", "biases"] {
                    #expect(index.weightMap[
                        "mtp.layers.0.mlp.experts.\(expert).\(projection).\(suffix)"
                    ] == "model-mtp-of-00005.safetensors")
                }
            }
        }
    }
}
