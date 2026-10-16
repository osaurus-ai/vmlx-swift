import Foundation
import Testing

@testable import MLXLMCommon

/// Physical Nemotron heads remain available, but Auto requires measured tuning.
@Suite struct NemotronMTPFamilyColdStartTests {
    private func config(_ extra: [String: Any]) -> [String: Any] {
        var c: [String: Any] = ["model_type": "nemotron_h", "num_nextn_predict_layers": 1]
        c.merge(extra) { $1 }
        return c
    }

    @Test("latent-MoE head presence alone does not enable Auto")
    func latentMoEMTPRequiresTuning() {
        let c = config(["moe_latent_size": 1024, "mtp_layers_block_type": ["attention", "moe"]])
        #expect(NativeMTPAutoDecodePolicy.measuredFamilyColdStartDepth(config: c, fallback: nil) == nil)
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

    @Test("Nemotron Auto requires a validated equivalent measured speedup")
    func measuredSpeedupGate() throws {
        let c = config(["moe_latent_size": 1024, "mtp_layers_block_type": ["attention", "moe"]])
        let data = try JSONSerialization.data(withJSONObject: c)
        for speed in [0.0, 35.0, 40.0, 45.0] {
            let tuning = NativeMTPTuning(
                bestDepth: 3, validated: true, outputEquivalent: true,
                baselineTokensPerSecond: 40, bestTokensPerSecond: speed,
                speedupVsBaseline: speed / 40, modelTypes: ["nemotron_h"])
            let status = MTPBundleStatus(bundleHasMTP: true, configuredLayers: 1,
                tensorCount: 10, mode: .speculativeVerified, nativeMTPTuning: tuning)
            let recommendation = NativeMTPAutoDecodePolicy.recommendation(
                configData: data, jangConfig: nil, status: status)
            #expect(recommendation?.depth == (speed > 40 ? 3 : nil))
        }
        let status = MTPBundleStatus(bundleHasMTP: true, configuredLayers: 1,
            tensorCount: 10, mode: .speculativeVerified)
        #expect(NativeMTPAutoDecodePolicy.recommendation(
            configData: data, jangConfig: nil, status: status) == nil)
        #expect(NativeMTPAutoDecodePolicy.manualRecommendation(
            depth: 3, configData: data, jangConfig: nil, status: status)?.depth == 3)
        #expect(NativeMTPAutoDecodePolicy.measuredFamilyColdStartDepth(
            config: ["model_type": "qwen4_exp"], fallback: nil) == 3)
    }

    /// Opt-in: VMLX_SUPER_BUNDLE=<Nemotron-3.5-Super-VL-120B-A12B-JANG_4M directory>.
    @Test("real Super-VL bundle keeps Auto off without measured speedup",
        .enabled(if: ProcessInfo.processInfo.environment["VMLX_SUPER_BUNDLE"] != nil))
    func realSuperBundle() throws {
        let root = URL(fileURLWithPath: ProcessInfo.processInfo.environment["VMLX_SUPER_BUNDLE"]!)
        let configData = try Data(contentsOf: root.appendingPathComponent("config.json"))
        let status = try MTPBundleInspector.inspect(modelDirectory: root)
        #expect(status.hasCompleteMTPArtifact)
        #expect(status.measuredFamilyAutoDepth == nil)
        let recommendation = NativeMTPAutoDecodePolicy.recommendation(
            configData: configData, jangConfig: nil, status: status)
        #expect(!status.hasUsableNativeMTPTuning)
        #expect(recommendation == nil)
        #expect(NativeMTPAutoDecodePolicy.manualRecommendation(
            depth: 3, configData: configData, jangConfig: nil, status: status)?.depth == 3)
    }
}
