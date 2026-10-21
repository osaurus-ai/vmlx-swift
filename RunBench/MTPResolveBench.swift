// BENCH_MTPRESOLVE=1 — for every bundle in BENCH_MODELS (comma-separated paths), resolve speculation using the
// Osaurus bundle default unless BENCH_MTP_MODE is explicitly set: inspect the bundle,
// resolvedMTPLaunch / resolvedLoadConfiguration / resolvedMTPDraftStrategy, and print the decision + reason.
// Metadata only — no weights are loaded.

import Foundation
import MLXLMCommon

func runMTPResolveBench() throws {
    let env = ProcessInfo.processInfo.environment
    let paths = (env["BENCH_MODELS"] ?? "").split(separator: ",").map(String.init)
    var settings = VMLXServerRuntimeSettings()
    if let mode = env["BENCH_MTP_MODE"], let parsed = VMLXMTPServerMode(rawValue: mode) {
        settings.mtp.mode = parsed
    }
    for path in paths {
        let url = URL(fileURLWithPath: path)
        let configData = try? Data(contentsOf: url.appendingPathComponent("config.json"))
        let jang = try? JangLoader.loadConfig(at: url)
        let status = try? MTPBundleInspector.inspect(modelDirectory: url, jangConfig: jang)
        let launch = settings.resolvedMTPLaunch(
            configData: configData, jangConfig: jang, status: status)
        let load = settings.resolvedLoadConfiguration(
            configData: configData, jangConfig: jang, status: status, bundleDirectory: url)
        let strategy = settings.resolvedMTPDraftStrategy(
            configData: configData, jangConfig: jang, status: status, bundleDirectory: url)
        print("[MTPRESOLVE] \(url.lastPathComponent)")
        print(
            "  status: complete=\(status?.hasCompleteMTPArtifact ?? false) layers=\(status?.configuredLayers ?? -1) tensors=\(status?.tensorCount ?? -1) mode=\(status.map { String(describing: $0.mode) } ?? "nil") tuning=\(status?.nativeMTPTuning != nil) usableTuning=\(status?.hasUsableNativeMTPTuning ?? false) familyAuto=\(String(describing: status?.measuredFamilyAutoDepth)) blocked=\(status?.isExplicitlyBlocked ?? false) canAuto=\(status?.canAutoLaunchMTP ?? false)"
        )
        print(
            "  nativeLaunch=\(launch.launchMode) depth=\(launch.recommendation.map { String($0.depth) } ?? "-") loadNativeMTP=\(load.nativeMTP) strategy=\(strategy.map { String(describing: $0) } ?? "nil")"
        )
        print("  nativeReason: \(launch.reason)")
    }
}
