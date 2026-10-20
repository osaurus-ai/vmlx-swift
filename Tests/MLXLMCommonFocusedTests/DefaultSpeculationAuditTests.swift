import Foundation
import Testing

@testable import MLXLMCommon

/// Opt-in audit: VMLX_DEFAULT_AUDIT_BUNDLES=<comma-separated bundle dirs>. Prints what the Default and
/// On (Adaptive) MTP settings resolve to for each real bundle (the engine's own launch policy plus the
/// bundled DFlash 2 / Uno drafter detection the app's Default uses).
@Suite struct DefaultSpeculationAuditTests {
    @Test("Default speculation per bundle",
        .enabled(if: ProcessInfo.processInfo.environment["VMLX_DEFAULT_AUDIT_BUNDLES"] != nil))
    func audit() throws {
        let dirs = ProcessInfo.processInfo.environment["VMLX_DEFAULT_AUDIT_BUNDLES"]!
            .split(separator: ",").map { URL(fileURLWithPath: String($0)) }
        for dir in dirs {
            let config = try? Data(contentsOf: dir.appendingPathComponent("config.json"))
            let status = try? MTPBundleInspector.inspect(modelDirectory: dir)
            var settings = VMLXServerRuntimeSettings()
            let dflash = settings.bundledDFlash2Drafter(configData: config, bundleDirectory: dir) != nil
            let uno = settings.bundledUnoAdapter(configData: config, bundleDirectory: dir) != nil
            settings.mtp.mode = .familyDefault
            let def = settings.resolvedMTPLaunch(configData: config, jangConfig: nil, status: status)
            settings.mtp.mode = .auto
            let adaptive = settings.resolvedMTPLaunch(configData: config, jangConfig: nil, status: status)
            print("[AUDIT] \(dir.lastPathComponent) mtpHead=\(status?.bundleHasMTP ?? false) "
                + "familyDepth=\(status?.measuredFamilyAutoDepth.map(String.init) ?? "-") "
                + "tuningUsable=\(status?.hasUsableNativeMTPTuning ?? false) blocked=\(status?.isExplicitlyBlocked ?? false) "
                + "dflash2=\(dflash) uno=\(uno) default=\(def.launchMode) d=\(def.recommendation?.depth.description ?? "-") "
                + "adaptive=\(adaptive.launchMode) | \(status?.statusLine ?? "no status")")
        }
    }
}
