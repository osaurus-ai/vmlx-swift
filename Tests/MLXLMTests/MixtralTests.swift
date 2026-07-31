import Foundation
import MLX
import Testing

@testable import MLXLLM

/// Pins the two pieces of the Mixtral port that carry real risk: the top-k router selection
/// (`argPartition` + softmax over the gathered gates) and the expert-weight `sanitize` that folds
/// per-expert `block_sparse_moe.experts.N.{w1,w2,w3}` into the stacked `switch_mlp`. The sanitize
/// tests exercise all three paths — fuse, already-stacked no-op, and the truncated-bundle guard that
/// replaced a stacking force-unwrap with a load error.
@Suite("Mixtral MoE")
struct MixtralTests {

    private static func filled(_ a: Int, _ b: Int) -> MLXArray {
        MLXArray(Array(repeating: Float(1), count: a * b)).reshaped(a, b)
    }

    /// The exact selection the router runs: `argPartition(-gates, kth: k-1)[..<k]` then softmax over
    /// the gathered gates. Fixed gates → the two highest experts, softmax scores summing to 1 with the
    /// top gate dominating — and it must be repeatable call-to-call (cold-vs-warm determinism).
    @Test("top-k router selects the highest gates, deterministically")
    func topKRoutingSelectsHighestGates() {
        MLXMetalTestLock.withLock {
            let gates = MLXArray([Float(1.0), 3.0, 2.0, 0.5]).reshaped(1, 4)
            let k = 2

            func select() -> MLXArray {
                MLX.argPartition(-gates, kth: k - 1, axis: -1)[.ellipsis, ..<k]
            }
            let inds = select()
            let selected = Set(inds.asArray(Int32.self).map(Int.init))
            #expect(selected == [1, 2])  // gates 3.0 and 2.0

            // Same input, same output — the routing has no run-to-run drift.
            #expect(inds.asArray(Int32.self) == select().asArray(Int32.self))

            let scores = MLX.softmax(MLX.takeAlong(gates, inds, axis: -1), axis: -1, precise: true)
                .asArray(Float.self)
            #expect(abs(scores.reduce(0, +) - 1.0) < 1e-5)
            // softmax([3, 2]) → {0.7311, 0.2689}; the top expert dominates.
            #expect(abs(scores.max()! - 0.7310586) < 1e-4)
        }
    }

    /// `argPartition`'s tie-break is implementation-defined, but it must be REPEATABLE within a build —
    /// otherwise routing (and thus the whole forward) would drift on tied gates. This pins repeatability,
    /// not a particular tie winner.
    @Test("tied gates route repeatably")
    func tiedGatesRouteRepeatably() {
        MLXMetalTestLock.withLock {
            let tied = MLXArray([Float(2.0), 2.0, 1.0, 0.5]).reshaped(1, 4)
            func select() -> [Int32] {
                MLX.argPartition(-tied, kth: 1, axis: -1)[.ellipsis, ..<2].asArray(Int32.self)
            }
            #expect(select() == select())
        }
    }

    // Memberwise init (via @testable) rather than JSON: synthesized `Decodable` ignores the struct's
    // property defaults, so a trimmed JSON fixture would fail on the first absent key. The memberwise
    // init honors the defaults, so we only name the fields the test cares about.
    private func tinyModel() -> MixtralModel {
        let cfg = MixtralConfiguration(
            vocabularySize: 8, hiddenSize: 4, intermediateSize: 6,
            hiddenLayers: 1, attentionHeads: 2, kvHeads: 1,
            numLocalExperts: 2, numExpertsPerToken: 2)
        return MixtralModel(cfg)
    }

    /// The HF layout: separate `experts.{0,1}.{w1,w2,w3}.weight` → stacked `switch_mlp.{gate,down,up}_proj`
    /// with a leading expert axis. `w1→gate_proj`, `w2→down_proj`, `w3→up_proj`; the per-expert keys are
    /// consumed.
    @Test("sanitize stacks per-expert weights into switch_mlp")
    func sanitizeStacksPerExpertWeights() {
        let model = tinyModel()
        MLXMetalTestLock.withLock {
            let p = "model.layers.0.block_sparse_moe"
            let weights: [String: MLXArray] = [
                "\(p).experts.0.w1.weight": Self.filled(6, 4),
                "\(p).experts.1.w1.weight": Self.filled(6, 4),
                "\(p).experts.0.w2.weight": Self.filled(4, 6),
                "\(p).experts.1.w2.weight": Self.filled(4, 6),
                "\(p).experts.0.w3.weight": Self.filled(6, 4),
                "\(p).experts.1.w3.weight": Self.filled(6, 4),
            ]
            let out = model.sanitize(weights: weights)

            #expect(out["\(p).switch_mlp.gate_proj.weight"]?.shape == [2, 6, 4])  // w1
            #expect(out["\(p).switch_mlp.down_proj.weight"]?.shape == [2, 4, 6])  // w2
            #expect(out["\(p).switch_mlp.up_proj.weight"]?.shape == [2, 6, 4])    // w3
            #expect(out["\(p).experts.0.w1.weight"] == nil)  // consumed, not left behind
        }
    }

    /// A JANG-stacked bundle arrives already fused (no `experts.0.w1.weight`); sanitize must pass it
    /// through untouched rather than trying to re-stack.
    @Test("sanitize is a no-op for an already-stacked bundle")
    func sanitizeNoOpForStackedBundle() {
        let model = tinyModel()
        MLXMetalTestLock.withLock {
            let p = "model.layers.0.block_sparse_moe"
            let already: [String: MLXArray] = ["\(p).switch_mlp.gate_proj.weight": Self.filled(2, 6)]
            let out = model.sanitize(weights: already)
            #expect(out["\(p).switch_mlp.gate_proj.weight"] != nil)          // untouched
            #expect(!out.keys.contains { $0.contains("experts.") })          // nothing invented
        }
    }

    /// The guard that replaced a stacking force-unwrap: a truncated bundle carries expert 0 but is
    /// missing a later shard. Stacking blind would trap; instead the keys are left in place so the
    /// module bind reports a load error rather than crashing.
    @Test("sanitize leaves a truncated bundle for the module bind to reject")
    func sanitizeLeavesTruncatedBundle() {
        let model = tinyModel()
        MLXMetalTestLock.withLock {
            let p = "model.layers.0.block_sparse_moe"
            let truncated: [String: MLXArray] = ["\(p).experts.0.w1.weight": Self.filled(6, 4)]
            // expert 1 absent (num_local_experts = 2)
            let out = model.sanitize(weights: truncated)
            #expect(out["\(p).switch_mlp.gate_proj.weight"] == nil)  // NOT stacked
            #expect(out["\(p).experts.0.w1.weight"] != nil)          // left for the bind to flag
        }
    }
}
