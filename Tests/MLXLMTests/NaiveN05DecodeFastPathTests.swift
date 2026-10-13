import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MLXRandom
import XCTest
@testable import MLXLLM

/// One-token decode without the mask (sliding layers; sparse layers at or below index_topk; above it the masked path
/// runs) against the masked decode path, at Naive-N0.5's attention geometry (64 heads, 192/128 head dims, sliding
/// window 128 with sinks), below and past the window and a 64-key index_topk: bit-identical logits.
final class NaiveN05DecodeFastPathTests: XCTestCase {
    private func contract(topK: Int = 64) throws -> NaiveN05ArchitectureContract {
        let values: [String: Any] = [
            "model_type": "naive_n05_flash", "hidden_size": 256, "intermediate_size": 64,
            "num_hidden_layers": 2, "vocab_size": 64, "num_attention_heads": 64,
            "num_key_value_heads": 4, "head_dim": 192, "v_head_dim": 128,
            "swa_num_attention_heads": 64, "swa_num_key_value_heads": 8,
            "swa_head_dim": 192, "swa_v_head_dim": 128, "partial_rotary_factor": 0.334,
            "n_routed_experts": 4, "num_experts_per_tok": 2, "moe_intermediate_size": 32,
            "hybrid_layer_pattern": [0, 1], "moe_layer_freq": [0, 1],
            "index_n_heads": 4, "index_head_dim": 128, "index_top_k": topK,
            "sliding_window": 128,
        ]
        return try JSONDecoder().decode(NaiveN05ArchitectureContract.self,
            from: JSONSerialization.data(withJSONObject: values))
    }

    func testDecodeFastPathMatchesMaskedPath() throws {
        try check(topK: 64)
    }

    /// Past the sliding window but below a 4096-key index_topk: only the no-mask branch runs, on both layer kinds.
    func testNoMaskPastWindowIsExact() throws {
        try check(topK: 4096)
    }

    private func check(topK: Int) throws {
        try MLXMetalTestLock.withLock {
            MLXRandom.seed(77)
            let model = try NaiveN05FlashModel(contract(topK: topK))
            try model.update(parameters: ModuleParameters.unflattened(
                model.parameters().flattened().map { name, value in
                    (name, name.contains(".mlp.gate.") ? value : (value * 4).asType(.bfloat16))
                }), verify: [.all])
            let longPrompt = MLXRandom.randInt(low: 0, high: 64, [1, 200]).asType(.int32)
            let next = MLXRandom.randInt(low: 0, high: 64, [6]).asType(.int32)
            // 40 + 6 keys stay below the 64-key index_topk and the 128-key window: the no-mask branch alone.
            for (label, prompt) in [("below topk", longPrompt[0..., ..<40]), ("past window and topk", longPrompt)] {
            func run(fast: Bool) throws -> MLXArray {
                NaiveN05FlashMath.decodeWithoutMaskEnabled = fast
                defer { NaiveN05FlashMath.decodeWithoutMaskEnabled = true }
                let cache = model.newCache()
                _ = try model(prompt, cache: cache)
                var logits: [MLXArray] = []
                for i in 0 ..< 6 {
                    logits.append(try model(next[i ..< (i + 1)].reshaped(1, 1), cache: cache))
                }
                let out = concatenated(logits, axis: 1)
                eval(out)
                return out
            }
            let fast = try run(fast: true)
            let masked = try run(fast: false)
            let diff = abs(fast.asType(.float32) - masked.asType(.float32)).max().item(Float.self)
            let scale = abs(masked.asType(.float32)).max().item(Float.self)
            print("Naive decode fast path vs masked, topK \(topK), \(label): max |diff| \(diff) (max |logit| \(scale))")
            XCTAssertGreaterThan(scale, 0)
            XCTAssertEqual(diff, 0, label)
            }
        }
    }
}
