import Foundation
import Testing

@testable import MLXLMCommon

/// Metadata-policy consistency only, not a complete-head or real-load proof.
@Suite struct NativeMTPManualSafetyParityTests {
    private func settings() -> VMLXServerRuntimeSettings {
        var settings = VMLXServerRuntimeSettings()
        settings.mtp.mode = .forceOn
        settings.mtp.explicitDepth = 3
        return settings
    }

    @Test func explicitSafetyBlockHasTheSameMeaningAtEveryGate() {
        for family in ["qwen3_5", "qwen4_exp"] {
            for manual in [false, true] {
                let status = MTPBundleStatus(
                    bundleHasMTP: true, configuredLayers: 1, tensorCount: 31,
                    mode: .preservedEnabled,
                    nativeMTPTuning: NativeMTPTuning(
                        bestDepth: 2, blocked: true, manualBlocked: manual,
                        reason: "fixture safety block"))
                let settings = settings()
                let resolved = settings.resolvedMTPLaunch(
                    configData: Data("{\"model_type\":\"\(family)\"}".utf8),
                    jangConfig: nil, status: status)
                #expect(resolved.launchMode == .blocked)
                #expect(settings.effectiveMTPLaunchMode(for: status) == resolved.launchMode)
                #expect(resolved.reason.contains("fixture safety block"))
                #expect(settings.validationIssues(mtpStatus: status).contains {
                    $0.field == "mtp.mode" && $0.message.contains("fixture safety block")
                })
            }
        }
    }

    @Test func missingMeasurementIsNotAnExplicitSafetyBlock() {
        for family in ["qwen3_5", "qwen4_exp"] {
            let status = MTPBundleStatus(
                bundleHasMTP: true, configuredLayers: 1, tensorCount: 31,
                mode: .preservedEnabled)
            let settings = settings()
            let resolved = settings.resolvedMTPLaunch(
                configData: Data("{\"model_type\":\"\(family)\"}".utf8),
                jangConfig: nil, status: status)
            #expect(resolved.launchMode == .speculative)
            #expect(resolved.recommendation?.depth == 3)
            #expect(settings.effectiveMTPLaunchMode(for: status) == resolved.launchMode)
        }
    }
}
