import Foundation
import Testing

@testable import MLXLMCommon

/// Ornith 1.5 122B (Qwen 3.5 MoE): row-exact verify makes greedy MTP equal AR, so Default and
/// On (Adaptive) launch native MTP at the measured cold-start depth. Keyed from model_type plus
/// topology; other Qwen 3.5 MoE shapes stay tuning-driven, and a bundle's explicit block still wins.
@Suite struct OrnithMTPFamilyColdStartTests {
    private func ornith(_ overrides: [String: Any] = [:]) -> [String: Any] {
        var text: [String: Any] = [
            "model_type": "qwen3_5_moe_text", "hidden_size": 3072, "num_hidden_layers": 48,
            "num_experts": 256, "num_experts_per_tok": 8, "mtp_num_hidden_layers": 1,
        ]
        text.merge(overrides) { $1 }
        return ["model_type": "qwen3_5_moe", "text_config": text]
    }

    @Test("Ornith 122B topology gets the measured cold-start depth")
    func ornith122BColdStart() throws {
        #expect(NativeMTPAutoDecodePolicy.measuredFamilyColdStartDepth(config: ornith(), fallback: nil)
            == NativeMTPAutoDecodePolicy.ornith122BColdStartDepth)
        let data = try JSONSerialization.data(withJSONObject: ornith())
        let status = MTPBundleStatus(bundleHasMTP: true, configuredLayers: 1,
            tensorCount: 10, mode: .preservedEnabled)
        #expect(NativeMTPAutoDecodePolicy.recommendation(
            configData: data, jangConfig: nil, status: status)?.depth
            == NativeMTPAutoDecodePolicy.ornith122BColdStartDepth)
    }

    @Test("other Qwen 3.5 MoE shapes and headless bundles stay tuning-driven")
    func otherShapesHaveNoColdStart() {
        for c in [
            ornith(["hidden_size": 2048, "num_hidden_layers": 40, "moe_intermediate_size": 512]),
            ornith(["mtp_num_hidden_layers": 0]),
            ornith(["num_experts": 128]),
            ornith(["model_type": "qwen3_5_text"]),
        ] {
            #expect(NativeMTPAutoDecodePolicy.measuredFamilyColdStartDepth(config: c, fallback: nil) == nil)
        }
    }

    @Test("an explicit bundle block still wins over the family depth")
    func explicitBlockWins() throws {
        let data = try JSONSerialization.data(withJSONObject: ornith())
        let tuning = NativeMTPTuning(
            bestDepth: 1, validated: true, outputEquivalent: true, blocked: true,
            manualBlocked: true, modelTypes: ["qwen3_5_moe"])
        let status = MTPBundleStatus(bundleHasMTP: true, configuredLayers: 1,
            tensorCount: 10, mode: .preservedEnabled, nativeMTPTuning: tuning)
        #expect(NativeMTPAutoDecodePolicy.recommendation(
            configData: data, jangConfig: nil, status: status) == nil)
    }
}
