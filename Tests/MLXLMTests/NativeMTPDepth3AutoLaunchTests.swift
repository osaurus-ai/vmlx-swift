// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import Testing

@testable import MLXLMCommon

/// The ONLY two native-MTP models we ship are:
///   - `OsaurusAI/Qwen3.6-27B-MXFP8-MTP`      (model_type `qwen3_5`)
///   - `OsaurusAI/Qwen3.6-35B-A3B-MXFP8-MTP`  (model_type `qwen3_5_moe`)
///
/// Both must auto-launch native MTP at **depth 3** with no user action. The depth
/// is not guessed from the model name or the MTP layer count (both bundles ship a
/// SINGLE MTP head, `mtp_layers: 1` — depth 3 comes from iterating that head, not
/// from having three of them). It is read from the bundle-local, measured
/// `vmlx_mtp_tuning.json`.
///
/// That makes the tuning artifact load-bearing in a way nothing guards today:
/// `NativeMTPAutoDecodePolicy.recommendation` is FAIL-CLOSED and returns nil —
/// silently disabling speculative decode entirely, not merely lowering the depth —
/// if ANY of six gates misses. Re-publishing a bundle without the file, or with a
/// tuning row whose `speedup_vs_baseline` slips to 1.0, turns MTP off with no
/// error anywhere.
///
/// These fixtures are the REAL published tuning rows (verified against
/// huggingface.co/OsaurusAI/Qwen3.6-27B-MXFP8-MTP and .../Qwen3.6-35B-A3B-MXFP8-MTP).
@Suite("Both shipped native-MTP models auto-launch at depth 3")
struct NativeMTPDepth3AutoLaunchTests {

    /// The published `vmlx_mtp_tuning.json`, verbatim.
    private static let publishedTuningJSON = """
        {
          "native_mtp": {
            "best_depth": 3,
            "validated": true,
            "output_equivalent": true,
            "cache_mode": "off",
            "prompt_class": "deterministic_count_96_tokens",
            "measured_at": "2026-05-17",
            "artifact": "docs/internal/release-gates/20260517_qwen36_27b_mxfp8_depth_sweep_selector_probe/result.json",
            "baseline_tok_s": 15.795,
            "best_tok_s": 28.936,
            "speedup_vs_baseline": 1.832
          }
        }
        """

    private static func publishedTuning() throws -> NativeMTPTuning {
        let data = Data(publishedTuningJSON.utf8)
        let root = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        let native = try #require(root?["native_mtp"] as? [String: Any])
        let nativeData = try JSONSerialization.data(withJSONObject: native)
        return try JSONDecoder().decode(NativeMTPTuning.self, from: nativeData)
    }

    /// A bundle as it actually ships: MTP present, one head, tensors on disk, a
    /// runtime that can accept/reject, and the measured tuning row.
    private static func shippedStatus(tuning: NativeMTPTuning) -> MTPBundleStatus {
        MTPBundleStatus(
            bundleHasMTP: true,
            configuredLayers: 1,  // both bundles: mtp_layers = 1
            tensorCount: 23,
            mode: .preservedEnabled,
            nativeMTPTuning: tuning)
    }

    private static func config(modelType: String) -> Data {
        Data(
            """
            {"model_type": "\(modelType)", "quantization": {"bits": 8}}
            """.utf8)
    }

    // MARK: - The load-bearing contract

    @Test(
        "the two shipped MTP models auto-launch at depth 3",
        arguments: [
            ("qwen3_5", "Qwen3.6-27B-MXFP8-MTP"),
            ("qwen3_5_moe", "Qwen3.6-35B-A3B-MXFP8-MTP"),
        ])
    func bothShippedModelsAutoLaunchAtDepth3(modelType: String, label: String) throws {
        let tuning = try Self.publishedTuning()
        let status = Self.shippedStatus(tuning: tuning)

        let rec = try #require(
            NativeMTPAutoDecodePolicy.recommendation(
                configData: Self.config(modelType: modelType),
                jangConfig: nil,
                status: status,
                requireVerifiedRuntime: true),
            """
            \(label) (model_type=\(modelType)) produced NO native-MTP recommendation. \
            recommendation() is fail-closed: a nil here means speculative decode is \
            silently OFF for this model — not merely running at a lower depth. \
            Rejection reason: \
            \(NativeMTPAutoDecodePolicy.rejectionReason(
                configData: Self.config(modelType: modelType),
                jangConfig: nil,
                status: status,
                requireVerifiedRuntime: true) ?? "none reported")
            """)

        #expect(rec.depth == 3, "\(label) must run native MTP at depth 3, got \(rec.depth)")
        #expect(status.canAutoLaunchMTP, "\(label) must auto-launch without user action")
        #expect(status.speculativeDecodeEnabled)
    }

    /// End to end through the settings resolver — the value generation actually uses.
    @Test(
        "the resolved draft strategy is .nativeMTP(depth: 3)",
        arguments: ["qwen3_5", "qwen3_5_moe"])
    func resolvedDraftStrategyIsNativeMTPDepth3(modelType: String) throws {
        let status = Self.shippedStatus(tuning: try Self.publishedTuning())
        let settings = VMLXServerRuntimeSettings()

        let strategy = try #require(
            settings.resolvedMTPDraftStrategy(
                configData: Self.config(modelType: modelType),
                jangConfig: nil,
                status: status),
            "default settings must resolve a native-MTP strategy for \(modelType)")

        guard case .nativeMTP(let depth, _) = strategy else {
            Issue.record("expected .nativeMTP, got \(strategy)")
            return
        }
        #expect(depth == 3)
        #expect(strategy.usesNativeMTP)
        #expect(strategy.usesBlockDiffusion == false)
    }

    // MARK: - Why the artifact is load-bearing (the silent-off traps)

    /// The trap that would ship MTP silently disabled: a bundle republished
    /// without its tuning row. The tensors are all still there, so nothing else
    /// complains.
    @Test("a bundle with no tuning row gets NO recommendation (silent-off trap)")
    func missingTuningDisablesMTPEntirely() {
        let status = MTPBundleStatus(
            bundleHasMTP: true,
            configuredLayers: 1,
            tensorCount: 23,
            mode: .preservedEnabled,
            nativeMTPTuning: nil)

        #expect(status.hasCompleteMTPArtifact, "the tensors are present and complete…")
        #expect(status.canAutoLaunchMTP == false, "…yet MTP cannot auto-launch")
        #expect(status.requiresNativeMTPTuningBeforeAutoLaunch)
        #expect(
            NativeMTPAutoDecodePolicy.recommendation(
                configData: Self.config(modelType: "qwen3_5"),
                jangConfig: nil,
                status: status) == nil)
    }

    /// `usableBestDepth` also demands the measured row prove a real speedup. A
    /// re-measure that lands at parity turns MTP off — by design, but worth pinning
    /// so the behaviour is deliberate rather than a surprise.
    @Test("a tuning row without a real speedup does not enable MTP")
    func noSpeedupDisablesMTP() throws {
        let flat = """
            {"best_depth": 3, "validated": true, "output_equivalent": true,
             "baseline_tok_s": 20.0, "best_tok_s": 20.0, "speedup_vs_baseline": 1.0}
            """
        let tuning = try JSONDecoder().decode(NativeMTPTuning.self, from: Data(flat.utf8))
        #expect(tuning.bestDepth == 3)
        #expect(tuning.usableBestDepth == nil, "no speedup ⇒ not usable")
    }
}
