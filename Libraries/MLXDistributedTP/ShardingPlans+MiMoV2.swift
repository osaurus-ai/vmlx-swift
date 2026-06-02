import Foundation

extension ShardingPlan {
    /// MiMo-V2.5 text tensor-parallel plan.
    ///
    /// This is intentionally separate from the Llama plan because MiMo has:
    /// - asymmetric full/SWA KV heads
    /// - SWA-only `attention_sink_bias`
    /// - routed `SwitchGLU` expert stacks with 3D `[expert, out, in]` weights
    /// - optional shared experts in each MoE block
    public static let mimoV2 = ShardingPlan(
        directives: Self.mimoV2TextDirectives(),
        parameterDirectives: Self.mimoV2TextParameterDirectives())

    private static func mimoV2TextDirectives(hiddenLayers: Int = 48)
        -> [String: LinearShardingDirective]
    {
        var directives: [String: LinearShardingDirective] = [:]
        for layer in 0 ..< hiddenLayers {
            let prefix = "model.layers.\(layer)"

            // Attention projections. MiMo source qkv is sanitized into split
            // q/k/v modules before this plan runs.
            directives["\(prefix).self_attn.q_proj"] = .allToSharded(segments: 1)
            directives["\(prefix).self_attn.k_proj"] = .allToSharded(segments: 1)
            directives["\(prefix).self_attn.v_proj"] = .allToSharded(segments: 1)
            directives["\(prefix).self_attn.o_proj"] = .shardedToAll(segments: 1)

            // Dense layer-0 MLP and shared experts.
            directives["\(prefix).mlp.gate_proj"] = .allToSharded(segments: 1)
            directives["\(prefix).mlp.up_proj"] = .allToSharded(segments: 1)
            directives["\(prefix).mlp.down_proj"] = .shardedToAll(segments: 1)
            directives["\(prefix).mlp.shared_experts.gate_proj"] = .allToSharded(segments: 1)
            directives["\(prefix).mlp.shared_experts.up_proj"] = .allToSharded(segments: 1)
            directives["\(prefix).mlp.shared_experts.down_proj"] = .shardedToAll(segments: 1)

            // Routed SwitchGLU experts. The replacement classes shard every
            // expert's projection tensor, not just ordinary 2D Linear leaves.
            directives["\(prefix).mlp.switch_mlp.gate_proj"] = .allToSharded(segments: 1)
            directives["\(prefix).mlp.switch_mlp.up_proj"] = .allToSharded(segments: 1)
            directives["\(prefix).mlp.switch_mlp.down_proj"] = .shardedToAll(segments: 1)
        }
        return directives
    }

    private static func mimoV2TextParameterDirectives(hiddenLayers: Int = 48)
        -> [String: ParameterShardingDirective]
    {
        var directives: [String: ParameterShardingDirective] = [:]
        for layer in 0 ..< hiddenLayers {
            // SWA layers have one sink per query head. Full-attention layers
            // do not use this parameter, and missing sinks remain untouched.
            directives["model.layers.\(layer).self_attn.attention_sink_bias"] =
                .slice(axis: 0, segments: 1)
        }
        return directives
    }
}
