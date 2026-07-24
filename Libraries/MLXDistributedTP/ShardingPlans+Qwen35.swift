import Foundation

extension ShardingPlan {
    /// Qwen 3.5 / Qwen3-family first-pass tensor-parallel plan.
    ///
    /// This plan covers the projection names shared by Qwen3 dense,
    /// Qwen3 MoE, and Qwen 3.5 text models:
    ///
    /// - self-attention q/k/v are output-sharded
    /// - self-attention o is input-sharded and all-reduced
    /// - dense MLP and shared expert gate/up are output-sharded
    /// - dense MLP and shared expert down are input-sharded and all-reduced
    ///
    /// Qwen 3.5 GatedDelta / SSM companion layers are intentionally not
    /// sharded here. Routed SwitchGLU experts are also left replicated until
    /// the SwitchLinear sharding path is part of the same clean proof. Both
    /// require separate parity proof for recurrent state, prefix replay, and
    /// L2/companion-cache restore before they can be treated as
    /// tensor-parallel data-plane proof.
    public static let qwen35 = ShardingPlan(
        directives: [
            "self_attn.q_proj": .allToSharded(segments: 1),
            "self_attn.k_proj": .allToSharded(segments: 1),
            "self_attn.v_proj": .allToSharded(segments: 1),
            "self_attn.o_proj": .shardedToAll(segments: 1),

            "mlp.gate_proj": .allToSharded(segments: 1),
            "mlp.up_proj": .allToSharded(segments: 1),
            "mlp.down_proj": .shardedToAll(segments: 1),

            "shared_expert.gate_proj": .allToSharded(segments: 1),
            "shared_expert.up_proj": .allToSharded(segments: 1),
            "shared_expert.down_proj": .shardedToAll(segments: 1),
            "shared_experts.gate_proj": .allToSharded(segments: 1),
            "shared_experts.up_proj": .allToSharded(segments: 1),
            "shared_experts.down_proj": .shardedToAll(segments: 1),
        ])
}
