import Foundation
import MLX
import MLXDistributedTP
import MLXLLM
import MLXLMCommon
import MLXNN

private let minimalMiMoV25Config = #"""
{
  "model_type": "mimo_v2",
  "num_experts_per_tok": 2,
  "hybrid_layer_pattern": [0, 1, 1, 0],
  "moe_layer_freq": [0, 1, 1, 1],
  "add_swa_attention_sink_bias": true,
  "add_full_attention_sink_bias": false,
  "sliding_window_size": 128,
  "vocab_size": 256,
  "hidden_size": 32,
  "intermediate_size": 64,
  "moe_intermediate_size": 16,
  "num_hidden_layers": 4,
  "num_attention_heads": 8,
  "num_key_value_heads": 4,
  "n_shared_experts": 1,
  "n_routed_experts": 4,
  "routed_scaling_factor": 1.0,
  "topk_method": "noaux_tc",
  "scoring_func": "sigmoid",
  "norm_topk_prob": true,
  "n_group": 1,
  "topk_group": 1,
  "max_position_embeddings": 4096,
  "layernorm_epsilon": 1e-6,
  "rope_theta": 10000000,
  "swa_rope_theta": 10000,
  "swa_num_attention_heads": 8,
  "swa_num_key_value_heads": 8,
  "head_dim": 4,
  "v_head_dim": 4,
  "swa_head_dim": 4,
  "swa_v_head_dim": 4,
  "partial_rotary_factor": 0.5,
  "attention_value_scale": 0.707
}
"""#

@main
struct MiMoTPPlanProbe {
    static func main() throws {
        let config = try JSONDecoder.json5().decode(
            MiMoV2FlashConfiguration.self,
            from: Data(minimalMiMoV25Config.utf8))
        let model = MiMoV2FlashModel(config)
        let group = Group.singleProcessTest(rank: 2, size: 4)

        let replaced = ShardingPlan.mimoV2.apply(to: model, group: group)
        print("MiMoTPPlanProbe replaced=\(replaced.count)")

        try require(replaced.contains("model.layers.0.self_attn.q_proj"), "missing full q_proj rewrite")
        try require(replaced.contains("model.layers.0.self_attn.o_proj"), "missing full o_proj rewrite")
        try require(
            replaced.contains("model.layers.1.mlp.switch_mlp.gate_proj"),
            "missing routed switch_mlp gate rewrite")
        try require(
            replaced.contains("model.layers.1.mlp.switch_mlp.down_proj"),
            "missing routed switch_mlp down rewrite")
        try require(
            replaced.contains("model.layers.1.mlp.shared_experts.gate_proj"),
            "missing shared_experts gate rewrite")
        try require(replaced.count >= 33, "expected attention + dense/shared/routed MLP rewrites")

        let leaves = Dictionary(uniqueKeysWithValues: model.leafModules().flattened())
        try requireShape(
            leaves["model.layers.0.self_attn.q_proj"] as? AllToShardedLinear,
            [8, 32],
            "full attention q_proj")
        try requireShape(
            leaves["model.layers.0.self_attn.k_proj"] as? AllToShardedLinear,
            [4, 32],
            "full attention k_proj")
        try requireShape(
            leaves["model.layers.0.self_attn.v_proj"] as? AllToShardedLinear,
            [4, 32],
            "full attention v_proj")
        try requireShape(
            leaves["model.layers.0.self_attn.o_proj"] as? ShardedToAllLinear,
            [32, 8],
            "full attention o_proj")

        try requireShape(
            leaves["model.layers.1.self_attn.q_proj"] as? AllToShardedLinear,
            [8, 32],
            "SWA attention q_proj")
        try requireShape(
            leaves["model.layers.1.self_attn.k_proj"] as? AllToShardedLinear,
            [8, 32],
            "SWA attention k_proj")
        try requireShape(
            leaves["model.layers.1.self_attn.v_proj"] as? AllToShardedLinear,
            [8, 32],
            "SWA attention v_proj")
        try requireShape(
            leaves["model.layers.1.self_attn.o_proj"] as? ShardedToAllLinear,
            [32, 8],
            "SWA attention o_proj")

        try requireShape(
            leaves["model.layers.1.mlp.switch_mlp.gate_proj"] as? AllToShardedSwitchLinear,
            [4, 4, 32],
            "routed switch_mlp gate")
        try requireShape(
            leaves["model.layers.1.mlp.switch_mlp.up_proj"] as? AllToShardedSwitchLinear,
            [4, 4, 32],
            "routed switch_mlp up")
        try requireShape(
            leaves["model.layers.1.mlp.switch_mlp.down_proj"] as? ShardedToAllSwitchLinear,
            [4, 32, 4],
            "routed switch_mlp down")
        try requireShape(
            leaves["model.layers.1.mlp.shared_experts.gate_proj"] as? AllToShardedLinear,
            [4, 32],
            "shared experts gate")
        try requireShape(
            leaves["model.layers.1.mlp.shared_experts.down_proj"] as? ShardedToAllLinear,
            [32, 4],
            "shared experts down")

        let parameters = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        try require(
            parameters["model.layers.1.self_attn.attention_sink_bias"]?.shape == [2],
            "SWA attention_sink_bias was not rank-local")

        print("MiMoTPPlanProbe PASS")
    }
}

private func require(_ condition: Bool, _ message: String) throws {
    if !condition {
        throw ProbeError.failure(message)
    }
}

private func requireShape(_ linear: Linear?, _ expected: [Int], _ label: String) throws {
    guard let linear else {
        throw ProbeError.failure("\(label) was not replaced with expected Linear type")
    }
    try require(linear.weight.shape == expected, "\(label) shape \(linear.weight.shape), expected \(expected)")
}

private func requireShape(_ linear: SwitchLinear?, _ expected: [Int], _ label: String) throws {
    guard let linear else {
        throw ProbeError.failure("\(label) was not replaced with expected SwitchLinear type")
    }
    try require(linear.weight.shape == expected, "\(label) shape \(linear.weight.shape), expected \(expected)")
}

private enum ProbeError: Error, CustomStringConvertible {
    case failure(String)

    var description: String {
        switch self {
        case .failure(let message):
            return message
        }
    }
}
