import Foundation
import Testing

@testable import MLXLMCommon

/// Nemotron 3.5 latent-MoE native MTP (Super-VL) launches Adaptive from a
/// measured family cold start; other Nemotron-H shapes stay tuning-driven.
@Suite struct NemotronMTPFamilyColdStartTests {
    private func config(_ extra: [String: Any]) -> [String: Any] {
        var c: [String: Any] = ["model_type": "nemotron_h", "num_nextn_predict_layers": 1]
        c.merge(extra) { $1 }
        return c
    }

    @Test("latent-MoE [attention, moe] MTP head gets the measured cold start")
    func latentMoEMTPColdStart() {
        let c = config(["moe_latent_size": 1024, "mtp_layers_block_type": ["attention", "moe"]])
        #expect(NativeMTPAutoDecodePolicy.measuredFamilyColdStartDepth(config: c, fallback: nil) == 3)
    }

    @Test("other Nemotron-H shapes stay tuning-driven")
    func otherShapesHaveNoColdStart() {
        for c in [
            config(["mtp_layers_block_type": ["attention", "moe"]]),
            config(["moe_latent_size": 1024]),
            config(["moe_latent_size": 1024, "mtp_layers_block_type": ["attention"]]),
            ["model_type": "glm5_next", "moe_latent_size": 1024,
             "mtp_layers_block_type": ["attention", "moe"]] as [String: Any],
        ] {
            #expect(NativeMTPAutoDecodePolicy.measuredFamilyColdStartDepth(config: c, fallback: nil) == nil)
        }
    }

    /// Opt-in: VMLX_SUPER_BUNDLE=<Nemotron-3.5-Super-VL-120B-A12B-JANG_4M directory>.
    @Test("real Super-VL bundle auto-launches native MTP at the family cold start",
        .enabled(if: ProcessInfo.processInfo.environment["VMLX_SUPER_BUNDLE"] != nil))
    func realSuperBundle() throws {
        let root = URL(fileURLWithPath: ProcessInfo.processInfo.environment["VMLX_SUPER_BUNDLE"]!)
        let configData = try Data(contentsOf: root.appendingPathComponent("config.json"))
        let status = try MTPBundleInspector.inspect(modelDirectory: root)
        #expect(status.hasCompleteMTPArtifact)
        #expect(status.measuredFamilyAutoDepth == 3)
        let recommendation = NativeMTPAutoDecodePolicy.recommendation(
            configData: configData, jangConfig: nil, status: status)
        #expect(recommendation?.depth == 3)
        #expect(recommendation?.evidence.contains(
            "activation=nemotron_h_latent_moe_measured_family_cold_start") == true)
    }
}
