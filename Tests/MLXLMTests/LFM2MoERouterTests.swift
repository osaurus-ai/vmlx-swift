import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM

@Suite("LFM2 MoE router matches the HF Lfm2Moe reference")
struct LFM2MoERouterTests {
    private func block(scaling: Float, experts: Int = 4, topK: Int = 2, useBias: Bool = true) throws -> Lfm2MoeSparseMoeBlock {
        let config = """
            {
              "model_type": "lfm2_moe",
              "vocab_size": 16,
              "hidden_size": 2,
              "intermediate_size": 4,
              "moe_intermediate_size": 3,
              "num_hidden_layers": 1,
              "num_experts": \(experts),
              "num_experts_per_tok": \(topK),
              "norm_topk_prob": true,
              "num_attention_heads": 1,
              "num_key_value_heads": 1,
              "max_position_embeddings": 128,
              "use_expert_bias": \(useBias),
              "num_dense_layers": 0,
              "norm_eps": 0.00001,
              "conv_bias": false,
              "conv_L_cache": 3,
              "routed_scaling_factor": \(scaling),
              "layer_types": ["conv"]
            }
            """
        let configuration = try JSONDecoder().decode(
            LFM2MoEConfiguration.self, from: Data(config.utf8))
        return Lfm2MoeSparseMoeBlock(configuration, layerIdx: 0)
    }

    @Test("bias selects experts, unbiased sigmoid scores weight them")
    func biasSelectsUnbiasedSigmoidWeights() throws {
        try MLXMetalTestLock.withLock {
            let moe = try block(scaling: 2.5)
            // x = [1, 0] makes the router logits the first column of the gate.
            let logits: [Float] = [2.0, 1.0, -1.0, -3.0]
            let gateWeight = MLXArray(logits.flatMap { [$0, 0] }, [4, 2])
            moe.update(parameters: ModuleParameters.unflattened([
                "gate.weight": gateWeight,
                // Expert 3 has the lowest sigmoid score but the bias lifts it
                // into the top two; expert 0 still wins on its own score.
                "expert_bias": MLXArray([0, 0, 0, 5] as [Float]),
            ]))
            let x = MLXArray([1, 0] as [Float], [1, 1, 2])

            let output = moe(x)

            let sig = logits.map { 1 / (1 + exp(-$0)) }
            let picked = [0, 3]
            let sum = sig[0] + sig[3] + 1e-6
            let weights = picked.map { sig[$0] / sum * 2.5 }
            let indices = MLXArray(picked.map { Int32($0) }, [1, 1, 2])
            let experts = moe.switchMLP(x, indices)
            let expected = (experts * MLXArray(weights, [1, 1, 2])[.ellipsis, .newAxis])
                .sum(axis: -2)

            #expect(allClose(output, expected, atol: 1e-5).item(Bool.self))
        }
    }

    @Test("BF16 mixing preserves reference top-k slot ordering")
    func bfloat16ReferenceSlotOrder() throws {
        try MLXMetalTestLock.withLock {
            let moe = try block(scaling: 1, experts: 8, topK: 4)
            let logits: [Float] = [2, 1, -1, -3, 0.5, 0.75, -0.5, 0.25]
            moe.update(parameters: ModuleParameters.unflattened([
                "gate.weight": MLXArray(logits.flatMap { [$0, 0] }, [8, 2]).asType(.bfloat16),
                "expert_bias": MLXArray([0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07] as [Float]),
                "switch_mlp.gate_proj.weight": sin(MLXArray(0..<48).asType(.float32) * 0.13)
                    .reshaped(8, 3, 2).asType(.bfloat16),
                "switch_mlp.up_proj.weight": cos(MLXArray(0..<48).asType(.float32) * 0.17)
                    .reshaped(8, 3, 2).asType(.bfloat16),
                "switch_mlp.down_proj.weight": sin(MLXArray(0..<48).asType(.float32) * 0.19)
                    .reshaped(8, 2, 3).asType(.bfloat16),
            ]))
            let x = sin(MLXArray(0..<128).asType(.float32) * 0.29)
                .reshaped(1, 64, 2).asType(.bfloat16)
            let probabilities = sigmoid(moe.gate(x))
            let selection = probabilities.asType(.float32) + moe.expertBias!
            let indices = argPartition(selection, kth: -4, axis: -1)[.ellipsis, (-4)...]
            let selected = takeAlong(probabilities, indices, axis: -1)
            let scores = (selected / (selected.sum(axis: -1, keepDims: true) + 1e-6))
                .asType(x.dtype)
            let expected = (moe.switchMLP(x, indices) * scores[.ellipsis, .newAxis])
                .sum(axis: -2)
            #expect(arrayEqual(moe(x), expected).item(Bool.self))
        }
    }

    @Test("unbiased BF16 routing uses the native tied-score partition")
    func bfloat16UnbiasedReferencePartition() throws {
        try MLXMetalTestLock.withLock {
            let moe = try block(scaling: 1, experts: 8, topK: 4, useBias: false)
            var parameters = Dictionary(uniqueKeysWithValues: moe.parameters().flattened().map { name, array in
                (name, sin(MLXArray(0..<array.size).asType(.float32) * 0.13)
                    .reshaped(array.shape).asType(.bfloat16))
            })
            let logits: [Float] = [2, 2, 2, 2, 1, 1, 1, 1]
            parameters["gate.weight"] = MLXArray(logits.flatMap { [$0, 0] }, [8, 2])
                .asType(.bfloat16)
            moe.update(parameters: ModuleParameters.unflattened(parameters))
            let x = MLXArray([1, 0] as [Float], [1, 1, 2]).asType(.bfloat16)
            let probabilities = sigmoid(moe.gate(x))
            let indices = argPartition(probabilities, kth: -4, axis: -1)[.ellipsis, (-4)...]
            let selected = takeAlong(probabilities, indices, axis: -1)
            let scores = (selected / (selected.sum(axis: -1, keepDims: true) + 1e-6))
                .asType(x.dtype)
            let expected = (moe.switchMLP(x, indices) * scores[.ellipsis, .newAxis])
                .sum(axis: -2)
            #expect(arrayEqual(moe(x), expected).item(Bool.self))
        }
    }

    @Test("routed_scaling_factor defaults to 1 when absent")
    func scalingDefaultsToOne() throws {
        let config = """
            {"model_type":"lfm2_moe","vocab_size":16,"hidden_size":2,"intermediate_size":4,
             "moe_intermediate_size":3,"num_hidden_layers":1,"num_experts":4,"num_experts_per_tok":2,
             "norm_topk_prob":true,"num_attention_heads":1,"num_key_value_heads":1,
             "max_position_embeddings":128,"use_expert_bias":true,"num_dense_layers":0,
             "norm_eps":0.00001,"conv_bias":false,"conv_L_cache":3,"layer_types":["conv"]}
            """
        let configuration = try JSONDecoder().decode(
            LFM2MoEConfiguration.self, from: Data(config.utf8))
        #expect(configuration.routedScalingFactor == 1)
    }
}
