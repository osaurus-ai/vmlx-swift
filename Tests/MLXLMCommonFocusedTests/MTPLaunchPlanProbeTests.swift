// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Replays the exact osaurus load-plan decision chain against a real
// bundle and prints every gate — the probe that answers "why did the
// app load this model without native MTP" without rebuilding the app.
//
//   VMLX_MTP_PROBE_BUNDLE=$HOME/models/JANGQ-AI/Qwen3.8-27B-JANG_4D \
//   swift test --filter MTPLaunchPlanProbeTests

import Foundation
import Testing

@testable import MLXLMCommon

@Suite("MTP launch-plan probe")
struct MTPLaunchPlanProbeTests {

    @Test("print the resolved launch plan for a real bundle")
    func launchPlanProbe() throws {
        guard let path = ProcessInfo.processInfo.environment["VMLX_MTP_PROBE_BUNDLE"] else {
            return
        }
        let dir = URL(fileURLWithPath: path)
        let status = try MTPBundleInspector.inspect(modelDirectory: dir)
        print("[probe] statusLine=\(status.statusLine)")
        print("[probe] hasCompleteMTPArtifact=\(status.hasCompleteMTPArtifact)")
        print("[probe] canAutoLaunchMTP=\(status.canAutoLaunchMTP)")
        print("[probe] hasUsableNativeMTPTuning=\(status.hasUsableNativeMTPTuning)")
        print("[probe] tuning=\(String(describing: status.nativeMTPTuning?.usableBestDepth))")

        let configData = try? Data(
            contentsOf: dir.appendingPathComponent("config.json"))
        let settings = VMLXServerRuntimeSettings()
        let launch = settings.resolvedMTPLaunch(
            configData: configData, jangConfig: nil, status: status)
        print("[probe] launchMode=\(launch.launchMode) reason=\(launch.reason)")
        let strategy = settings.resolvedMTPDraftStrategy(
            configData: configData, jangConfig: nil, status: status)
        print("[probe] strategy=\(String(describing: strategy))")
        if launch.launchMode != .speculative {
            let reject = NativeMTPAutoDecodePolicy.rejectionReason(
                configData: configData, jangConfig: nil, status: status,
                requireVerifiedRuntime: true)
            print("[probe] rejection=\(String(describing: reject))")
        }
    }
}
