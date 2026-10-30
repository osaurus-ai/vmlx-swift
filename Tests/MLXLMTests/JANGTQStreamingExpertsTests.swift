import Foundation
import MLX
@testable import MLXLLM
import Testing
@testable import MLXLMCommon

@Suite("JANGTQ streaming expert index", .serialized)
struct JANGTQStreamingExpertsTests {
    /// Streaming experts are retired as a production mechanism:
    /// `JANGTQStreamingExperts.explicitStreamingEnabled` is `false`, so a
    /// Qwen35 MoE JANGTQ bundle with large stacked experts, which used to
    /// auto-enable them, must no longer do so.
    @Test("Qwen35 MoE JANGTQ bundles no longer auto-enable retired streaming experts")
    func qwen35MoEJANGTQLargeStackedExpertsDoNotAutoEnableStreaming() throws {
        let bundle = try Self.makeBundle(name: "qwen35")
        defer { try? FileManager.default.removeItem(at: bundle) }

        try Self.writeQwen35Config(at: bundle, numExperts: 512)
        try Self.writeSafetensors(
            at: bundle.appendingPathComponent("model-00001-of-00001.safetensors"),
            tensors: Self.qwen35SwitchMLPTensors(layer: 0, experts: 512))

        try Self.withStreamingEnv(value: nil) {
            #expect(!JANGTQStreamingExperts.shouldAutoEnableQwen35MoE(modelDirectory: bundle))
        }
    }

    @Test("explicit streaming disable wins over Qwen35 auto-enable")
    func explicitStreamingDisableWinsOverQwen35AutoEnable() throws {
        let bundle = try Self.makeBundle(name: "qwen35-disabled")
        defer { try? FileManager.default.removeItem(at: bundle) }

        try Self.writeQwen35Config(at: bundle, numExperts: 512)
        try Self.writeSafetensors(
            at: bundle.appendingPathComponent("model-00001-of-00001.safetensors"),
            tensors: Self.qwen35SwitchMLPTensors(layer: 0, experts: 512))

        try Self.withStreamingEnv(value: "0") {
            #expect(!JANGTQStreamingExperts.shouldAutoEnableQwen35MoE(modelDirectory: bundle))
        }
    }

    @Test("prestacked switch_mlp tensors are streamable")
    func prestackedSwitchMLPTensorsAreStreamable() throws {
        let bundle = try Self.makeBundle(name: "prestacked")
        defer { try? FileManager.default.removeItem(at: bundle) }

        try Self.writeSafetensors(
            at: bundle.appendingPathComponent("model-00001-of-00001.safetensors"),
            tensors: [
                TensorSpec(
                    name: "model.layers.0.block_sparse_moe.switch_mlp.gate_proj.tq_packed",
                    dtype: "U32", shape: [2, 3, 1]),
                TensorSpec(
                    name: "model.layers.0.block_sparse_moe.switch_mlp.gate_proj.tq_norms",
                    dtype: "F16", shape: [2, 3]),
                TensorSpec(
                    name: "model.layers.0.block_sparse_moe.switch_mlp.up_proj.tq_packed",
                    dtype: "U32", shape: [2, 3, 1]),
                TensorSpec(
                    name: "model.layers.0.block_sparse_moe.switch_mlp.up_proj.tq_norms",
                    dtype: "F16", shape: [2, 3]),
                TensorSpec(
                    name: "model.layers.0.block_sparse_moe.switch_mlp.down_proj.tq_packed",
                    dtype: "U32", shape: [2, 2, 1]),
                TensorSpec(
                    name: "model.layers.0.block_sparse_moe.switch_mlp.down_proj.tq_norms",
                    dtype: "F16", shape: [2, 2]),
            ])

        #expect(JANGTQStreamingExperts.hasStreamableExperts(in: bundle))
    }

    @Test("Qwen35 auto-streaming sanitize removes per-expert tensors without stacking")
    func qwen35AutoStreamingSanitizeRemovesPerExpertTensors() throws {
        let configData = Data(
            """
            {
              "model_type": "qwen3_5_moe",
              "auto_streaming_experts": true,
              "text_config": {
                "model_type": "qwen3_5_moe_text",
                "vocab_size": 128,
                "hidden_size": 8,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "linear_num_value_heads": 1,
                "linear_num_key_heads": 1,
                "linear_key_head_dim": 4,
                "linear_value_head_dim": 4,
                "linear_conv_kernel_dim": 4,
                "intermediate_size": 16,
                "shared_expert_intermediate_size": 16,
                "moe_intermediate_size": 4,
                "num_experts": 2,
                "num_experts_per_tok": 1,
                "decoder_sparse_step": 1,
                "full_attention_interval": 1,
                "mxtq_bits": 2
              }
            }
            """.utf8)
        let config = try JSONDecoder().decode(Qwen35JANGTQConfiguration.self, from: configData)
        let model = Qwen35JANGTQTextModel(config.textConfig)
        let weights = Self.qwen35PerExpertWeights(layer: 0, experts: 2)

        let sanitized = model.sanitize(weights: weights)

        #expect(config.textConfig.autoStreamingExperts)
        #expect(sanitized.keys.allSatisfy { !$0.contains(".experts.") })
        #expect(sanitized["model.layers.0.mlp.switch_mlp.gate_proj.tq_packed"] == nil)
        #expect(sanitized["model.layers.0.mlp.switch_mlp.up_proj.tq_packed"] == nil)
        #expect(sanitized["model.layers.0.mlp.switch_mlp.down_proj.tq_packed"] == nil)
    }

    @Test("Zaya zaya_block switch_mlp tensors are streamable")
    func zayaSwitchMLPTensorsAreStreamable() throws {
        let bundle = try Self.makeBundle(name: "zaya")
        defer { try? FileManager.default.removeItem(at: bundle) }

        try Self.writeSafetensors(
            at: bundle.appendingPathComponent("model-00001-of-00001.safetensors"),
            tensors: [
                TensorSpec(
                    name: "model.layers.1.zaya_block.experts.switch_mlp.gate_proj.tq_packed",
                    dtype: "U32", shape: [2, 3, 1]),
                TensorSpec(
                    name: "model.layers.1.zaya_block.experts.switch_mlp.gate_proj.tq_norms",
                    dtype: "F16", shape: [2, 3]),
                TensorSpec(
                    name: "model.layers.1.zaya_block.experts.switch_mlp.up_proj.tq_packed",
                    dtype: "U32", shape: [2, 3, 1]),
                TensorSpec(
                    name: "model.layers.1.zaya_block.experts.switch_mlp.up_proj.tq_norms",
                    dtype: "F16", shape: [2, 3]),
                TensorSpec(
                    name: "model.layers.1.zaya_block.experts.switch_mlp.down_proj.tq_packed",
                    dtype: "U32", shape: [2, 2, 1]),
                TensorSpec(
                    name: "model.layers.1.zaya_block.experts.switch_mlp.down_proj.tq_norms",
                    dtype: "F16", shape: [2, 2]),
            ])

        #expect(JANGTQStreamingExperts.hasStreamableExperts(in: bundle))
    }

    @Test("legacy per-expert tensors remain streamable")
    func perExpertTensorsRemainStreamable() throws {
        let bundle = try Self.makeBundle(name: "per-expert")
        defer { try? FileManager.default.removeItem(at: bundle) }

        try Self.writeSafetensors(
            at: bundle.appendingPathComponent("model-00001-of-00001.safetensors"),
            tensors: [
                TensorSpec(
                    name: "model.layers.0.block_sparse_moe.experts.0.w1.tq_packed",
                    dtype: "U32", shape: [3, 1]),
                TensorSpec(
                    name: "model.layers.0.block_sparse_moe.experts.0.w1.tq_norms",
                    dtype: "F16", shape: [3]),
            ])

        #expect(JANGTQStreamingExperts.hasStreamableExperts(in: bundle))
    }

    private struct TensorSpec {
        var name: String
        var dtype: String
        var shape: [Int]
    }

    private static func makeBundle(name: String) throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("jangtq-streaming-\(name)-\(UUID().uuidString)",
                isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private static func writeQwen35Config(at bundle: URL, numExperts: Int) throws {
        let config: [String: Any] = [
            "model_type": "qwen3_5_moe",
            "text_config": [
                "model_type": "qwen3_5_moe_text",
                "num_experts": numExperts,
                "num_experts_per_tok": 10,
            ],
        ]
        let jangConfig: [String: Any] = [
            "format": "jangtq",
            "weight_format": "mxtq",
            "profile": "JANGTQ2",
        ]
        try JSONSerialization.data(withJSONObject: config, options: [.sortedKeys])
            .write(to: bundle.appendingPathComponent("config.json"))
        try JSONSerialization.data(withJSONObject: jangConfig, options: [.sortedKeys])
            .write(to: bundle.appendingPathComponent("jang_config.json"))
    }

    private static func qwen35SwitchMLPTensors(layer: Int, experts: Int) -> [TensorSpec] {
        [
            TensorSpec(
                name: "language_model.model.layers.\(layer).mlp.switch_mlp.gate_proj.tq_packed",
                dtype: "U32", shape: [experts, 3, 1]),
            TensorSpec(
                name: "language_model.model.layers.\(layer).mlp.switch_mlp.gate_proj.tq_norms",
                dtype: "F16", shape: [experts, 3]),
            TensorSpec(
                name: "language_model.model.layers.\(layer).mlp.switch_mlp.up_proj.tq_packed",
                dtype: "U32", shape: [experts, 3, 1]),
            TensorSpec(
                name: "language_model.model.layers.\(layer).mlp.switch_mlp.up_proj.tq_norms",
                dtype: "F16", shape: [experts, 3]),
            TensorSpec(
                name: "language_model.model.layers.\(layer).mlp.switch_mlp.down_proj.tq_packed",
                dtype: "U32", shape: [experts, 2, 1]),
            TensorSpec(
                name: "language_model.model.layers.\(layer).mlp.switch_mlp.down_proj.tq_norms",
                dtype: "F16", shape: [experts, 2]),
        ]
    }

    private static func qwen35PerExpertWeights(layer: Int, experts: Int) -> [String: MLXArray] {
        var weights: [String: MLXArray] = [:]
        for expert in 0 ..< experts {
            let prefix = "model.layers.\(layer).mlp.experts.\(expert)"
            weights["\(prefix).w1.tq_packed"] = MLXArray.zeros([4, 1], dtype: .uint32)
            weights["\(prefix).w1.tq_norms"] = MLXArray.zeros([4], dtype: .float16)
            weights["\(prefix).w2.tq_packed"] = MLXArray.zeros([8, 1], dtype: .uint32)
            weights["\(prefix).w2.tq_norms"] = MLXArray.zeros([8], dtype: .float16)
            weights["\(prefix).w3.tq_packed"] = MLXArray.zeros([4, 1], dtype: .uint32)
            weights["\(prefix).w3.tq_norms"] = MLXArray.zeros([4], dtype: .float16)
        }
        return weights
    }

    private static func withStreamingEnv(value: String?, body: () throws -> Void) throws {
        try withEnv("MLXPRESS_STREAMING_EXPERTS", value: value) {
            try withEnv("JANGPRESS_STREAMING_EXPERTS", value: value, body: body)
        }
    }

    private static func withEnv(_ key: String, value: String?, body: () throws -> Void) throws {
        let previous = getenv(key).map { String(cString: $0) }
        if let value {
            setenv(key, value, 1)
        } else {
            unsetenv(key)
        }
        defer {
            if let previous {
                setenv(key, previous, 1)
            } else {
                unsetenv(key)
            }
        }
        try body()
    }

    private static func writeSafetensors(at url: URL, tensors: [TensorSpec]) throws {
        var header: [String: Any] = [:]
        var offset = 0
        for tensor in tensors {
            let byteCount = tensor.shape.reduce(1, *) * byteWidth(dtype: tensor.dtype)
            header[tensor.name] = [
                "dtype": tensor.dtype,
                "shape": tensor.shape,
                "data_offsets": [offset, offset + byteCount],
            ]
            offset += byteCount
        }
        let headerJSON = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        var data = Data()
        var headerSize = UInt64(headerJSON.count)
        withUnsafeBytes(of: &headerSize) { data.append(contentsOf: $0) }
        data.append(headerJSON)
        data.append(Data(count: offset))
        try data.write(to: url)
    }

    private static func byteWidth(dtype: String) -> Int {
        switch dtype {
        case "F16", "BF16", "I16", "U16": return 2
        case "F32", "I32", "U32": return 4
        case "F64", "I64", "U64": return 8
        default: return 1
        }
    }
}
