// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import MLXLMCommon
import MLXRandom
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

    @Test("tiny GDN QSA staged prefixes preserve continuation without PLE")
    func stagedPrefixContinuationWithoutPLE() throws {
        try MLXMetalTestLock.withLock {
            var root = try #require(JSONSerialization.jsonObject(
                with: configData(dtype: "float32")) as? [String: Any])
            var text = try #require(root["text_config"] as? [String: Any])
            // This row qualifies GDN/QSA only. PLE requires a real table fixture
            // and remains a separate gate before production enablement.
            text["ple_layer_ids"] = [Int]()
            root["text_config"] = text
            let config = try JSONDecoder().decode(Qwen4ExpConfiguration.self,
                from: JSONSerialization.data(withJSONObject: root))
            #expect(config.base.textConfiguration.hiddenSize == 64)
            #expect(config.base.textConfiguration.hiddenLayers == 2)
            MLXRandom.seed(829)
            let model = try Qwen4Exp(config, requesting: [.text])
            let count = model.parameters().flattened().reduce(0) { $0 + $1.1.size }
            try #require(count < 2_000_000, "Never evaluate a production-size fixture")
            func compare(_ actual: MLXArray, _ expected: MLXArray, _ label: String) {
                #expect(actual.shape == expected.shape, "\(label)")
                guard actual.shape == expected.shape else { return }
                let error = abs(actual.asType(.float32) - expected.asType(.float32)).max().item(Float.self)
                print("QWEN-STAGED \(label) maxAbs=\(error) parameters=\(count)")
                #expect(error.isFinite && error <= 1e-5, "\(label) maxAbs=\(error)")
            }
            for accepted in 1...4 {
                let staged = model.newCache(parameters: nil)
                let reference = model.newCache(parameters: nil)
                let prefix = MLXArray([Int32(1), 2, 3]).reshaped(1, 3)
                for cache in [staged, reference] {
                    let out = model.nativeBackboneForward(prefix, cache: cache)
                    MLX.eval(out.logits, cache)
                }
                let recurrent = try #require(staged.first as? MambaCache)
                let before = recurrent.state.map { $0 * 1 }
                MLX.eval(before)
                let beforeOffset = recurrent.offset
                let block = MLXArray([Int32(4), 5, 6, 7]).reshaped(1, 4)
                let verified = NativeMTPVerifierStatePolicy.withVerifierMode("input_capture_staged") {
                    model.nativeBackboneMTPVerifyForward(block, cache: staged)
                }
                MLX.eval(verified.logits, staged)
                #expect(recurrent.offset == beforeOffset)
                #expect(recurrent.state.count == before.count)
                for (index, pair) in zip(recurrent.state, before).enumerated() {
                    compare(pair.0, pair.1, "uncommitted prefix=\(accepted) slot=\(index)")
                }
                for layer in staged where layer.isTrimmable { _ = layer.trim(4 - accepted) }
                try #require(model.commitStagedVerifiedBlock(
                    cache: staged, acceptedInputs: accepted, blockLength: 4))
                for token in 4..<(4 + accepted) {
                    let out = model.nativeBackboneForward(MLXArray([Int32(token)]).reshaped(1, 1), cache: reference)
                    MLX.eval(out.logits, reference)
                    let row = token - 4
                    compare(verified.logits[0..., row..<(row + 1), 0...], out.logits,
                        "verifier logits prefix=\(accepted) row=\(row)")
                }
                let referenceRecurrent = try #require(reference.first as? MambaCache)
                #expect(recurrent.offset == referenceRecurrent.offset)
                #expect(recurrent.state.count == referenceRecurrent.state.count)
                #expect(staged.map(\.offset) == reference.map(\.offset))
                for (index, pair) in zip(recurrent.state, referenceRecurrent.state).enumerated() {
                    compare(pair.0, pair.1, "committed prefix=\(accepted) slot=\(index)")
                }
                let next = MLXArray([Int32(11)]).reshaped(1, 1)
                let actual = model.nativeBackboneForward(next, cache: staged)
                let expected = model.nativeBackboneForward(next, cache: reference)
                compare(actual.logits, expected.logits, "continuation prefix=\(accepted)")
                let spread = (expected.logits.max() - expected.logits.min()).item(Float.self)
                #expect(spread.isFinite && spread > 1e-6, "Reject a constant-logit oracle")
            }
        }
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
}
