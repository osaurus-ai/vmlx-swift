import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXVLM

@Suite("Qwen3.5-VL gated-delta forward", .serialized)
struct Qwen35VLMGatedDeltaTests {
    private func tinyConfiguration() throws -> Qwen35Configuration {
        try JSONDecoder.json5().decode(
            Qwen35Configuration.self,
            from: """
                {
                  "model_type": "qwen3_5_moe",
                  "text_config": {
                    "model_type": "qwen3_5_moe_text",
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

    @Test("tiny VLM text path executes linear-attention gated-delta cache")
    func tinyVLMTextPathExecutesGatedDeltaCache() throws {
        try MLXMetalTestLock.withLock {
            let model = Qwen35(try tinyConfiguration())
            let cache = model.newCache(parameters: nil)
            let input = MLXArray([1, 2, 3])[.newAxis, .ellipsis]

            let logits = model(input, cache: cache)
            MLX.eval(logits)

            #expect(logits.shape == [1, 3, 100])
            #expect(cache.count == 4)
            #expect(cache.first is MambaCache)
            #expect(cache.first?.offset == 3)
            let mambaCache = try #require(cache.first as? MambaCache)
            #expect(mambaCache[1]?.dtype == .float32)
        }
    }

    @Test("batch=2 decode preserves one position offset per sequence")
    func batchDecodePreservesPerSequenceOffsets() throws {
        try MLXMetalTestLock.withLock {
            let model = Qwen35(try tinyConfiguration())
            let first = model.newCache(parameters: nil)
            let second = model.newCache(parameters: nil)

            MLX.eval(model(MLXArray([1, 2, 3])[.newAxis, .ellipsis], cache: first))
            MLX.eval(model(MLXArray([4, 5])[.newAxis, .ellipsis], cache: second))

            let batchCaches: [KVCache] = zip(first, second).map { lhs, rhs in
                if let lhs = lhs as? ArraysCache, let rhs = rhs as? ArraysCache {
                    return BatchArraysCache(slotCaches: [lhs, rhs])
                }
                return BatchKVCache(slotCaches: [lhs, rhs])
            }
            let decodeTokens = MLXArray([Int32(6), Int32(7)]).reshaped(2, 1)
            let logits = model(decodeTokens, cache: batchCaches)
            MLX.eval(logits)

            #expect(logits.shape == [2, 1, 100])
            let fullAttention = try #require(batchCaches.last as? BatchKVCache)
            #expect(fullAttention.offsetArray.asArray(Int32.self) == [4, 3])
        }
    }

    @Test("compiled decode policy is architecture-scoped and explicitly overridable")
    func compiledDecodePolicyIsArchitectureScoped() throws {
        var ornith = try tinyConfiguration().textConfiguration
        ornith.modelType = "qwen3_5_moe_text"
        ornith.hiddenSize = 2048
        ornith.hiddenLayers = 40
        ornith.fullAttentionInterval = 4
        ornith.numExperts = 256
        ornith.numExpertsPerTok = 8
        ornith.moeIntermediateSize = 512
        ornith.linearNumKeyHeads = 16
        ornith.linearNumValueHeads = 32
        ornith.linearKeyHeadDim = 128
        ornith.linearValueHeadDim = 128

        #expect(Qwen35Language.shouldCompileDecodeRegions(ornith, environment: [:]))
        #expect(
            !Qwen35Language.shouldCompileDecodeRegions(
                ornith,
                environment: ["VMLINUX_QWEN35_COMPILE_DECODE_REGIONS": "false"]))

        var neighbor = ornith
        neighbor.hiddenSize = 2560
        #expect(!Qwen35Language.shouldCompileDecodeRegions(neighbor, environment: [:]))
        #expect(
            Qwen35Language.shouldCompileDecodeRegions(
                neighbor,
                environment: ["VMLINUX_QWEN35_COMPILE_DECODE_REGIONS": "1"]))
    }

    @Test("compiled GDN tail matches eager SiLU and sigmoid gate contracts")
    func compiledGDNTailMatchesEagerContracts() throws {
        try MLXMetalTestLock.withLock {
            let heads = 4
            let headDimension = 8
            let valueDimension = heads * headDimension
            let hiddenDimension = 16
            let eps: Float = 1e-6
            let output = MLXRandom.uniform(
                low: -1, high: 1, [1, 1, heads, headDimension],
                key: MLXRandom.key(701)
            ).asType(.bfloat16)
            let gate = MLXRandom.uniform(
                low: -1, high: 1, [1, 1, heads, headDimension],
                key: MLXRandom.key(702)
            ).asType(.bfloat16)
            let normWeight = MLXRandom.uniform(
                low: 0.5, high: 1.5, [headDimension],
                key: MLXRandom.key(703)
            ).asType(.bfloat16)
            let sourceWeight = MLXRandom.uniform(
                low: -0.5, high: 0.5, [hiddenDimension, valueDimension],
                key: MLXRandom.key(704)
            ).asType(.bfloat16)
            let (weight, scales, biases) = MLX.quantized(
                sourceWeight, groupSize: 32, bits: 4, mode: .affine)
            let affineBiases = try #require(biases)

            for sigmoidGate in [false, true] {
                let compiled = try #require(
                    Qwen4ExpCompiledGDNInputs.callTail(
                        output: output, gate: gate, sigmoidGate: sigmoidGate,
                        normWeight: normWeight,
                        outWeight: weight, outScales: scales, outBiases: affineBiases,
                        eps: eps, groupSize: 32, bits: 4, mode: .affine))
                let normalized = MLXFast.rmsNorm(output, weight: normWeight, eps: eps)
                let activatedGate =
                    sigmoidGate
                    ? sigmoid(gate.asType(.float32))
                    : gate.asType(.float32) * sigmoid(gate.asType(.float32))
                let gated = (normalized.asType(.float32) * activatedGate)
                    .asType(output.dtype)
                    .reshaped(1, 1, valueDimension)
                let eager = quantizedMM(
                    gated, weight, scales: scales, biases: affineBiases,
                    transpose: true, groupSize: 32, bits: 4, mode: .affine)
                MLX.eval(compiled, eager)

                #expect(compiled.shape == eager.shape)
                let difference = abs(
                    compiled.asType(.float32) - eager.asType(.float32)
                ).max().item(Float.self)
                #expect(difference < 1e-3, "gate=\(sigmoidGate) max diff \(difference)")
            }
        }
    }
}
