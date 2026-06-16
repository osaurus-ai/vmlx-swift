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
  "moe_intermediate_size": 128,
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

        let quantizeExperts = ProcessInfo.processInfo.environment["TP_QUANTIZE_EXPERTS"] == "1"
        if quantizeExperts {
            let packedInput = 32 / max(1, 32 / (Int(ProcessInfo.processInfo.environment["TP_MIMO_ROUTED_EXPERT_BITS"] ?? "") ?? 8))
            try requireQuantizedShape(
                leaves["model.layers.1.mlp.switch_mlp.gate_proj"] as? QuantizedAllToShardedSwitchLinear,
                weight: [4, 32, packedInput],
                scales: [4, 32, 1],
                biases: [4, 32, 1],
                "routed quantized switch_mlp gate")
            try requireQuantizedShape(
                leaves["model.layers.1.mlp.switch_mlp.up_proj"] as? QuantizedAllToShardedSwitchLinear,
                weight: [4, 32, packedInput],
                scales: [4, 32, 1],
                biases: [4, 32, 1],
                "routed quantized switch_mlp up")
            try requireQuantizedShape(
                leaves["model.layers.1.mlp.switch_mlp.down_proj"] as? QuantizedShardedToAllSwitchLinear,
                weight: [4, 32, packedInput],
                scales: [4, 32, 1],
                biases: [4, 32, 1],
                "routed quantized switch_mlp down")
        } else {
            try requireShape(
                leaves["model.layers.1.mlp.switch_mlp.gate_proj"] as? AllToShardedSwitchLinear,
                [4, 32, 32],
                "routed switch_mlp gate")
            try requireShape(
                leaves["model.layers.1.mlp.switch_mlp.up_proj"] as? AllToShardedSwitchLinear,
                [4, 32, 32],
                "routed switch_mlp up")
            try requireShape(
                leaves["model.layers.1.mlp.switch_mlp.down_proj"] as? ShardedToAllSwitchLinear,
                [4, 32, 32],
                "routed switch_mlp down")
        }
        try requireShape(
            leaves["model.layers.1.mlp.shared_experts.gate_proj"] as? AllToShardedLinear,
            [32, 32],
            "shared experts gate")
        try requireShape(
            leaves["model.layers.1.mlp.shared_experts.down_proj"] as? ShardedToAllLinear,
            [32, 32],
            "shared experts down")

        let parameters = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        try require(
            parameters["model.layers.1.self_attn.attention_sink_bias"]?.shape == [2],
            "SWA attention_sink_bias was not rank-local")

        try verifyMiMoSanitizeQuantizesFP8Experts()
        try verifyFP8DecodeRoundTrip()
        try verifyQuantizedSwitchTP()

        print("MiMoTPPlanProbe PASS")
    }

    private static func verifyFP8DecodeRoundTrip() throws {
        let values = MLXArray([Float32(1.0), Float32(-2.0), Float32(0.5), Float32(-0.25)], [2, 2])
        let decoded = MLX.fromFP8(MLX.toFP8(values), dtype: .float32)
        eval(decoded)
        let actual = decoded.asArray(Float.self)
        let expected: [Float] = [1.0, -2.0, 0.5, -0.25]
        for (index, pair) in zip(actual, expected).enumerated() {
            try require(abs(pair.0 - pair.1) <= 0.001, "FP8 round-trip mismatch at \(index): \(pair.0) vs \(pair.1)")
        }
        print("MiMoTPPlanProbe fp8_decode_roundtrip=PASS")
    }

    private static func verifyMiMoSanitizeQuantizesFP8Experts() throws {
        var configJSON = try JSONSerialization.jsonObject(with: Data(minimalMiMoV25Config.utf8)) as! [String: Any]
        configJSON["hidden_size"] = 128
        configJSON["intermediate_size"] = 128
        configJSON["moe_intermediate_size"] = 128
        configJSON["n_routed_experts"] = 2
        configJSON["num_experts_per_tok"] = 1
        let data = try JSONSerialization.data(withJSONObject: configJSON)
        let config = try JSONDecoder.json5().decode(MiMoV2FlashConfiguration.self, from: data)
        let model = MiMoV2FlashModel(config)

        var weights: [String: MLXArray] = [:]
        for expert in 0 ..< 2 {
            for proj in ["gate_proj", "up_proj", "down_proj"] {
                let base = "model.layers.1.mlp.experts.\(expert).\(proj)"
                weights["\(base).weight"] = MLXArray.ones([128, 128]).asType(.uint8)
                weights["\(base).weight_scale_inv"] = MLXArray.ones([1, 1])
            }
        }

        let sanitized = model.sanitize(weights: weights)
        let bits = Int(ProcessInfo.processInfo.environment["TP_MIMO_ROUTED_EXPERT_BITS"] ?? "") ?? 4
        let groupSize = Int(ProcessInfo.processInfo.environment["TP_MIMO_ROUTED_EXPERT_GROUP_SIZE"] ?? "") ?? 128
        let packedWidth = 128 / max(1, 32 / bits)
        let scaleGroups = max(1, 128 / groupSize)
        for proj in ["gate_proj", "up_proj", "down_proj"] {
            let base = "model.layers.1.mlp.switch_mlp.\(proj)"
            try require(
                sanitized["\(base).weight"]?.shape == [2, 128, packedWidth],
                "\(proj) quantized weight missing/wrong shape")
            try require(
                sanitized["\(base).scales"]?.shape == [2, 128, scaleGroups],
                "\(proj) scales shape \(String(describing: sanitized["\(base).scales"]?.shape)), expected [2, 128, \(scaleGroups)]")
            try require(
                sanitized["\(base).biases"]?.shape == [2, 128, scaleGroups],
                "\(proj) biases shape \(String(describing: sanitized["\(base).biases"]?.shape)), expected [2, 128, \(scaleGroups)]")
            try require(sanitized["model.layers.1.mlp.experts.0.\(proj).weight"] == nil, "\(proj) per-expert weight leaked")
            try require(sanitized["model.layers.1.mlp.experts.0.\(proj).weight_scale_inv"] == nil, "\(proj) scale_inv leaked")
        }
        print("MiMoTPPlanProbe fp8_expert_sanitize_quantized=PASS")
    }

    private static func verifyQuantizedSwitchTP() throws {
        let root = QuantizedSwitchProbeRoot()
        let plan = ShardingPlan(directives: [
            "gate": .allToSharded(segments: 1),
            "up": .allToSharded(segments: 1),
            "down": .shardedToAll(segments: 1),
        ])
        let group = Group.singleProcessTest(rank: 2, size: 4)
        let replaced = plan.apply(to: root, group: group)
        try require(replaced.contains("gate"), "missing quantized gate TP rewrite")
        try require(replaced.contains("up"), "missing quantized up TP rewrite")
        try require(replaced.contains("down"), "missing quantized down TP rewrite")

        let leaves = Dictionary(uniqueKeysWithValues: root.leafModules().flattened())
        guard let shardedGate = leaves["gate"] as? QuantizedAllToShardedSwitchLinear,
              let shardedUp = leaves["up"] as? QuantizedAllToShardedSwitchLinear,
              let shardedDown = leaves["down"] as? QuantizedShardedToAllSwitchLinear
        else {
            throw ProbeError.failure("MiMo TP plan did not preserve QuantizedSwitchLinear wrappers")
        }

        try requireQuantizedShape(shardedGate, weight: [4, 32, 4], scales: [4, 32, 1], biases: [4, 32, 1], "quantized gate")
        try requireQuantizedShape(shardedUp, weight: [4, 32, 4], scales: [4, 32, 1], biases: [4, 32, 1], "quantized up")
        try requireQuantizedShape(shardedDown, weight: [4, 32, 4], scales: [4, 32, 1], biases: [4, 32, 1], "quantized down")
        try require(shardedGate.groupSize == 32 && shardedGate.bits == 4 && shardedGate.mode == .affine, "quantized gate metadata changed")
        try require(shardedDown.groupSize == 32 && shardedDown.bits == 4 && shardedDown.mode == .affine, "quantized down metadata changed")
        print("MiMoTPPlanProbe quantized_switch_tp=PASS")
    }
}

private final class QuantizedSwitchProbeRoot: Module {
    @ModuleInfo(key: "gate") var gate: SwitchLinear
    @ModuleInfo(key: "up") var up: SwitchLinear
    @ModuleInfo(key: "down") var down: SwitchLinear

    override init() {
        _gate.wrappedValue = QuantizedSwitchLinear(
            SwitchLinear(inputDims: 32, outputDims: 128, numExperts: 4, bias: false),
            groupSize: 32,
            bits: 4,
            mode: .affine)
        _up.wrappedValue = QuantizedSwitchLinear(
            SwitchLinear(inputDims: 32, outputDims: 128, numExperts: 4, bias: false),
            groupSize: 32,
            bits: 4,
            mode: .affine)
        _down.wrappedValue = QuantizedSwitchLinear(
            SwitchLinear(inputDims: 128, outputDims: 32, numExperts: 4, bias: false),
            groupSize: 32,
            bits: 4,
            mode: .affine)
        super.init()
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

private func requireQuantizedShape(
    _ linear: QuantizedSwitchLinear?,
    weight: [Int],
    scales: [Int],
    biases: [Int],
    _ label: String
) throws {
    guard let linear else {
        throw ProbeError.failure("\(label) was not replaced with expected quantized SwitchLinear type")
    }
    try requireQuantizedShape(linear, weight: weight, scales: scales, biases: biases, label)
}

private func requireQuantizedShape(
    _ linear: QuantizedSwitchLinear,
    weight: [Int],
    scales: [Int],
    biases: [Int],
    _ label: String
) throws {
    try require(linear.weight.shape == weight, "\(label) weight shape \(linear.weight.shape), expected \(weight)")
    try require(linear.scales.shape == scales, "\(label) scales shape \(linear.scales.shape), expected \(scales)")
    try require(linear.biases?.shape == biases, "\(label) biases shape \(String(describing: linear.biases?.shape)), expected \(biases)")
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
