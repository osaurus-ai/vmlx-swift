import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM

@Suite("LFM2 MoE router matches the HF Lfm2Moe reference")
struct LFM2MoERouterTests {
    private func block(scaling: Float) throws -> Lfm2MoeSparseMoeBlock {
        let config = """
            {
              "model_type": "lfm2_moe",
              "vocab_size": 16,
              "hidden_size": 2,
              "intermediate_size": 4,
              "moe_intermediate_size": 3,
              "num_hidden_layers": 1,
              "num_experts": 4,
              "num_experts_per_tok": 2,
              "norm_topk_prob": true,
              "num_attention_heads": 1,
              "num_key_value_heads": 1,
              "max_position_embeddings": 128,
              "use_expert_bias": true,
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
