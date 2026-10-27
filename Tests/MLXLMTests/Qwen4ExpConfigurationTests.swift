// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXVLM

@Suite("qwen4_exp configuration")
struct Qwen4ExpConfigurationTests {
    private func configData(dtype: String, routedBits: [[Int]]? = nil) -> Data {
        let quantization: String
        if let routedBits {
            var entries: [String] = ["\"group_size\":64", "\"bits\":8"]
            for (layer, bits) in routedBits.enumerated() {
                precondition(bits.count == 3)
                for (projection, bit) in zip(
                    ["gate_proj", "up_proj", "down_proj"], bits)
                {
                    entries.append(
                        "\"language_model.layers.\(layer).mlp.switch_mlp.\(projection)\":"
                            + "{\"group_size\":64,\"bits\":\(bit)}")
                }
            }
            quantization = "\"quantization\":{\(entries.joined(separator: ","))},"
        } else {
            quantization = ""
        }
        return Data("""
            {
              "model_type":"qwen4_exp",
              \(quantization)
              "text_config":{
                "model_type":"qwen4_exp_text","dtype":"\(dtype)",
                "mamba_ssm_dtype":"float32","mtp_num_hidden_layers":1,
                "hidden_size":64,"num_hidden_layers":2,"intermediate_size":64,
                "num_attention_heads":4,"num_key_value_heads":1,"head_dim":16,
                "linear_num_value_heads":4,"linear_num_key_heads":1,
                "linear_key_head_dim":16,"linear_value_head_dim":16,
                "linear_conv_kernel_dim":4,"vocab_size":128,
                "num_experts":8,"num_experts_per_tok":2,
                "moe_intermediate_size":16,"shared_expert_intermediate_size":16,
                "layer_types":["linear_attention","full_attention"],
                "hc_count":4,"hc_lowrank":8,"ple_layer_ids":[2],
                "ple_embed_dim":64,"ple_conv_kernel_size":4,
                "ngram_size":3,"heads_per_ngram":2,"ngram_vocab_size_base":101,
                "make_ngram_vocab_size_divisible_by":128,"seed":null,
                "split_ngram_parts":4,"indexer_n_heads":2,"indexer_kv_heads":1,
                "indexer_head_dim":8,"indexer_budget":32,"indexer_compress_ratio":4
              },
              "vision_config":{
                "model_type":"qwen3_vl","depth":2,"hidden_size":64,
                "intermediate_size":128,"out_hidden_size":64,"num_heads":4,
                "patch_size":14,"spatial_merge_size":2,"temporal_patch_size":2,
                "num_position_embeddings":64
              }
            }
            """.utf8)
    }

    @Test("decodes native text_config fields and null seed fallback")
    func decodesNativeConfig() throws {
        let config = try JSONDecoder().decode(
            Qwen4ExpConfiguration.self, from: configData(dtype: "bfloat16"))
        #expect(config.base.textConfiguration.modelType == "qwen4_exp_text")
        #expect(config.declaredComputeDType == .bfloat16)
        #expect(config.extras.mambaSSMDType == "float32")
        #expect(config.extras.mtpNumHiddenLayers == 1)
        #expect(config.extras.layerTypes == ["linear_attention", "full_attention"])
        #expect(config.extras.seed == 1234)
        #expect(config.extras.hcCount == 4)
        #expect(VLMTypeRegistry.supportedModelTypes.contains("qwen4_exp"))

        let model = Qwen4Exp(config)
        #expect(model.usesResidentNativeAffineRoutedExperts)
        #expect(model.requiresExactTensorMmapBuffers)
        #expect(!model.excludeFromGenericSafetensorsLoad(
            key: "language_model.layers.0.mlp.switch_mlp.gate_proj.weight"))
        #expect(model.excludeFromGenericSafetensorsLoad(
            key: "language_model.layers.1.ple.ngram_embedding.shards.0.weight"))
        #expect(!model.excludeFromGenericSafetensorsLoad(
            key: "language_model.layers.0.mlp.shared_expert.gate_proj.weight"))

        let sanitized = model.sanitize(weights: [
            "language_model.layers.0.mlp.switch_mlp.gate_proj.weight":
                MLXArray.zeros([2, 2], dtype: .uint32),
            "language_model.layers.0.mlp.switch_mlp.gate_proj.scales":
                MLXArray.ones([2, 2], dtype: .float16),
            "language_model.layers.0.mlp.switch_mlp.gate_proj.biases":
                MLXArray.zeros([2, 2], dtype: .float16),
            "language_model.layers.0.attn_hyper_connection.hc_norm.weight":
                MLXArray.ones([64], dtype: .bfloat16),
            "language_model.layers.0.linear_attn.A_log":
                MLXArray.ones([4], dtype: .float32),
        ])
        #expect(sanitized[
            "language_model.layers.0.mlp.switch_mlp.gate_proj.scales"]?.dtype == .float16)
        #expect(sanitized[
            "language_model.layers.0.mlp.switch_mlp.gate_proj.biases"]?.dtype == .float16)
        #expect(sanitized[
            "language_model.layers.0.attn_hyper_connection.hc_norm.weight"]?.dtype == .bfloat16)
        #expect(sanitized["language_model.layers.0.linear_attn.A_log"]?.dtype == .float32)

    }

    @Test("4M q4g64 verifier isolates the complete MoE block by decode row")
    func q4VerifierUsesDecodeEquivalentMoERows() throws {
        let config = try JSONDecoder().decode(
            Qwen4ExpConfiguration.self,
            from: configData(dtype: "bfloat16", routedBits: [[4, 4, 4], [4, 4, 4]]))
        #expect(config.hasUniformQ4G64TrunkRoutedExperts)
        var text = config.base.textConfiguration
        text.moeIntermediateSize = 64
        text.sharedExpertIntermediateSize = 64
        let block = Qwen35Language.SparseMoeBlock(
            text, layerIdx: 0,
            allowFusedGateUpCache: false, compileDecodeRegions: true,
            decodeEquivalentVerifierRows: config.hasUniformQ4G64TrunkRoutedExperts)
        quantize(model: block, groupSize: 64, bits: 4)

        let values = (0 ..< (4 * 64)).map { Float(($0 % 37) - 18) / 19 }
        let input = MLXArray(values).reshaped(1, 4, 64).asType(.bfloat16)
        #expect(block.requiresDecodeEquivalentRows(input))

        let batched = block(input)
        let rows = MLX.split(input, parts: 4, axis: 1)
        let independent = MLX.concatenated(rows.map { block($0) }, axis: 1)
        let error = max(
            abs(batched.asType(.float32) - independent.asType(.float32)))
        MLX.eval(error)
        #expect(error.item(Float.self) == 0)

        for bits in [2, 6] {
            let nonFourBit = Qwen35Language.SparseMoeBlock(
                text, layerIdx: 0,
                allowFusedGateUpCache: false, compileDecodeRegions: true,
                decodeEquivalentVerifierRows: true)
            quantize(model: nonFourBit, groupSize: 64, bits: bits)
            #expect(!nonFourBit.requiresDecodeEquivalentRows(input))
        }
    }

    @Test("mixed 4S and q2 2L routed layouts never enable 4M verifier rows")
    func mixedAndLowBitLayoutsDoNotEnableVerifierRows() throws {
        let mixed = try JSONDecoder().decode(
            Qwen4ExpConfiguration.self,
            from: configData(dtype: "bfloat16", routedBits: [[3, 3, 3], [3, 2, 4]]))
        let lowBit = try JSONDecoder().decode(
            Qwen4ExpConfiguration.self,
            from: configData(dtype: "bfloat16", routedBits: [[2, 2, 2], [2, 2, 2]]))
        #expect(!mixed.hasUniformQ4G64TrunkRoutedExperts)
        #expect(!lowBit.hasUniformQ4G64TrunkRoutedExperts)

        var text = mixed.base.textConfiguration
        text.moeIntermediateSize = 64
        text.sharedExpertIntermediateSize = 64
        let input = MLXArray.zeros([1, 4, 64], dtype: .bfloat16)
        let mixedBlock = Qwen35Language.SparseMoeBlock(
            text, layerIdx: 0, allowFusedGateUpCache: false,
            compileDecodeRegions: true,
            decodeEquivalentVerifierRows: mixed.hasUniformQ4G64TrunkRoutedExperts)
        quantize(model: mixedBlock, groupSize: 64, bits: 4)
        #expect(!mixedBlock.requiresDecodeEquivalentRows(input))
    }

    @Test("packed affine metadata preserves checkpoint storage dtype")
    func packedAffineMetadataPreservesStorageDType() throws {
        let config = try JSONDecoder().decode(
            Qwen4ExpConfiguration.self, from: configData(dtype: "float16"))
        let model = Qwen4Exp(config)
        let sanitized = model.sanitize(weights: [
            "language_model.embed_tokens.weight":
                MLXArray.zeros([2, 2], dtype: .uint32),
            "language_model.embed_tokens.scales":
                MLXArray.ones([2, 2], dtype: .bfloat16),
            "language_model.embed_tokens.biases":
                MLXArray.zeros([2, 2], dtype: .bfloat16),
            "lm_head.weight": MLXArray.zeros([2, 2], dtype: .uint32),
            "lm_head.scales": MLXArray.ones([2, 2], dtype: .bfloat16),
            "lm_head.biases": MLXArray.zeros([2, 2], dtype: .bfloat16),
            "language_model.layers.0.linear_attn.A_log":
                MLXArray.ones([4], dtype: .float32),
        ])
        #expect(sanitized["language_model.embed_tokens.scales"]?.dtype == .bfloat16)
        #expect(sanitized["language_model.embed_tokens.biases"]?.dtype == .bfloat16)
        #expect(sanitized["lm_head.scales"]?.dtype == .bfloat16)
        #expect(sanitized["lm_head.biases"]?.dtype == .bfloat16)
        #expect(sanitized["language_model.layers.0.linear_attn.A_log"]?.dtype == .float32)
    }

    private func floats(_ array: MLXArray?) -> [Float] {
        array?.asType(.float32).asArray(Float.self) ?? []
    }

    /// The layout an mlx-vlm-style converter writes (ddalcu/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit):
    /// the module tree's own names, with every value already through the reference's sanitize. On
    /// that bundle the shifted norms equal JANG_2L's `runtime_plus1_applied` values bit for bit, and
    /// the two MTP fusion norms equal JANG_2L's zero-centred values plus one.
    @Test("an mlx-vlm-converted bundle is renamed and its values are not transformed twice")
    func sanitizesConvertedLayout() throws {
        try MLXMetalTestLock.withLock {
            let config = try JSONDecoder().decode(
                Qwen4ExpConfiguration.self, from: configData(dtype: "bfloat16"))
            let model = Qwen4Exp(config)
            let pleConv = MLXArray(0 ..< 32).reshaped(8, 4, 1).asType(.bfloat16)
            let sanitized = model.sanitize(weights: [
                "language_model.model.layers.0.attn_hyper_connection.hc_norm.weight":
                    MLXArray([1.25, 0.5] as [Float]).asType(.bfloat16),
                "language_model.model.layers.1.self_attn.q_norm.weight":
                    MLXArray([1.125, 0.65625] as [Float]).asType(.bfloat16),
                "language_model.model.layers.1.ple.norm_key.weight":
                    MLXArray([0.828125] as [Float]).asType(.bfloat16),
                "language_model.model.layers.1.ple.conv1d.weight": pleConv,
                "language_model.model.layers.1.ple.ple_embedding.layer_multipliers":
                    MLXArray([3, 5, 7] as [Int32]),
                "language_model.model.layers.1.ple.ple_embedding.ngram_heads_offsets":
                    MLXArray([0, 101] as [Int32]),
                "language_model.model.layers.0.linear_attn.conv1d.weight":
                    MLXArray.zeros([8, 4, 1], dtype: .bfloat16),
                "language_model.model.embed_tokens.weight": MLXArray.zeros([2, 2], dtype: .uint32),
                "language_model.lm_head.weight": MLXArray.zeros([2, 2], dtype: .uint32),
                "language_model.mtp.fc_hidden.weight": MLXArray.zeros([2, 2], dtype: .uint32),
                "language_model.mtp.layers.0.self_attn.k_norm.weight":
                    MLXArray([3.3125] as [Float]).asType(.bfloat16),
                "language_model.mtp.pre_fc_norm_hidden.weight":
                    MLXArray([1.078125, 0.35546875] as [Float]).asType(.bfloat16),
                "language_model.mtp.pre_fc_norm_embedding.weight":
                    MLXArray([0.35546875] as [Float]).asType(.bfloat16),
                "model.visual.blocks.0.norm1.weight": MLXArray.ones([4], dtype: .bfloat16),
            ])
            #expect(Set(sanitized.keys) == [
                "language_model.layers.0.attn_hyper_connection.hc_norm.weight",
                "language_model.layers.1.self_attn.q_norm.weight",
                "language_model.layers.1.ple.norm_key.weight",
                "language_model.layers.1.ple.conv1d_weight",
                "language_model.layers.0.linear_attn.conv1d.weight",
                "language_model.embed_tokens.weight",
                "lm_head.weight",
                "mtp.fc_hidden.weight",
                "mtp.layers.0.self_attn.k_norm.weight",
                "mtp.pre_fc_norm_hidden.weight",
                "mtp.pre_fc_norm_embedding.weight",
                "visual.blocks.0.norm1.weight",
            ])
            // Already shifted: no second +1.
            #expect(floats(sanitized["language_model.layers.0.attn_hyper_connection.hc_norm.weight"])
                == [1.25, 0.5])
            #expect(floats(sanitized["language_model.layers.1.self_attn.q_norm.weight"]) == [1.125, 0.65625])
            #expect(floats(sanitized["language_model.layers.1.ple.norm_key.weight"]) == [0.828125])
            #expect(floats(sanitized["mtp.layers.0.self_attn.k_norm.weight"]) == [3.3125])
            // The MTP fusion norms are applied as `1 + weight`, so the folded one comes back out.
            #expect(floats(sanitized["mtp.pre_fc_norm_hidden.weight"]) == [0.078125, -0.64453125])
            #expect(floats(sanitized["mtp.pre_fc_norm_embedding.weight"]) == [-0.64453125])
            // [C, K, 1] (MLX conv layout) loses its trailing axis, values in place.
            #expect(sanitized["language_model.layers.1.ple.conv1d_weight"]?.shape == [8, 4])
            #expect(floats(sanitized["language_model.layers.1.ple.conv1d_weight"])
                == (0 ..< 32).map(Float.init))
            #expect(sanitized["language_model.layers.0.linear_attn.conv1d.weight"]?.shape == [8, 4, 1])
        }
    }

    /// The control: official Hugging Face names still get the reference's value transforms.
    @Test("official checkpoint names still get the +1 and the conv squeeze")
    func sanitizesOfficialLayout() throws {
        try MLXMetalTestLock.withLock {
            let config = try JSONDecoder().decode(
                Qwen4ExpConfiguration.self, from: configData(dtype: "bfloat16"))
            let model = Qwen4Exp(config)
            let pleConv = MLXArray(0 ..< 32).reshaped(8, 1, 4).asType(.bfloat16)
            let sanitized = model.sanitize(weights: [
                "model.language_model.layers.0.attn_hyper_connection.hc_norm.weight":
                    MLXArray([0.25, -0.5] as [Float]).asType(.bfloat16),
                "model.language_model.layers.1.ple.conv1d.weight": pleConv,
                "model.mtp.pre_fc_norm_hidden.weight": MLXArray([0.078125] as [Float]).asType(.bfloat16),
                "model.visual.blocks.0.norm1.weight": MLXArray.ones([4], dtype: .bfloat16),
                // PyTorch's depthwise Conv1d stores [C, 1, K]; MLX's Conv1d binds [C, K, 1].
                "model.language_model.layers.0.linear_attn.conv1d.weight":
                    MLXArray(0 ..< 32).reshaped(8, 1, 4).asType(.bfloat16),
            ])
            #expect(Set(sanitized.keys) == [
                "language_model.layers.0.attn_hyper_connection.hc_norm.weight",
                "language_model.layers.1.ple.conv1d_weight",
                "language_model.layers.0.linear_attn.conv1d.weight",
                "mtp.pre_fc_norm_hidden.weight",
                "visual.blocks.0.norm1.weight",
            ])
            // The GDN conv the model binds is MLX's [C, K, 1]; the sanitized official weight must match it.
            let bound = Dictionary(uniqueKeysWithValues: model.parameters().flattened())[
                "language_model.layers.0.linear_attn.conv1d.weight"]
            #expect(bound?.ndim == 3 && bound?.dim(2) == 1)
            // Moving a size-1 axis keeps the memory order, so the values stay 0..<32.
            #expect(sanitized["language_model.layers.0.linear_attn.conv1d.weight"]?.shape == [8, 4, 1])
            #expect(floats(sanitized["language_model.layers.0.linear_attn.conv1d.weight"])
                == (0 ..< 32).map(Float.init))
            #expect(floats(sanitized["language_model.layers.0.attn_hyper_connection.hc_norm.weight"])
                == [1.25, 0.5])
            #expect(floats(sanitized["mtp.pre_fc_norm_hidden.weight"]) == [0.078125])
            #expect(sanitized["language_model.layers.1.ple.conv1d_weight"]?.shape == [8, 4])
            #expect(floats(sanitized["language_model.layers.1.ple.conv1d_weight"])
                == (0 ..< 32).map(Float.init))
        }
    }
}
