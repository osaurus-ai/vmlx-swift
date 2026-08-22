// Copyright © 2026 Osaurus AI. All rights reserved.
//
// Probe: what depth do the Qwen bundles ON THIS DISK actually launch at?
//
// Skips cleanly when ~/models is absent, so it is a local instrument rather
// than a CI gate.

import Foundation
import Testing

@testable import MLXLMCommon

@Suite("qwen shipped MTP depth probe")
struct QwenMTPShippedDepthProbeTests {
    @Test("report the resolved launch depth for every local Qwen MTP bundle")
    func reportShippedDepths() throws {
        let root = URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent("models")
        guard FileManager.default.fileExists(atPath: root.path) else { return }

        var found: [(String, String)] = []
        let fm = FileManager.default
        guard let e = fm.enumerator(at: root, includingPropertiesForKeys: nil) else { return }
        for case let url as URL in e {
            guard url.lastPathComponent == "config.json" else { continue }
            let dir = url.deletingLastPathComponent()
            if dir.path.contains("/Logs/") { continue }
            let name = dir.path.replacingOccurrences(of: root.path + "/", with: "")
            let configData = try? Data(contentsOf: dir.appendingPathComponent("config.json"))
            let jang = try? JangLoader.loadConfig(at: dir)
            let status = try? MTPBundleInspector.inspect(modelDirectory: dir, jangConfig: jang)
            guard let status, status.hasCompleteMTPArtifact else { continue }
            let tuning = status.nativeMTPTuning
            let rec = NativeMTPAutoDecodePolicy.recommendation(
                configData: configData, jangConfig: jang, status: status)
            let verdict =
                rec.map { "LAUNCHES depth=\($0.depth) verifier=\($0.verifierMode ?? "-")" }
                ?? "NO MTP (tuning decoded=\(tuning != nil), usableBestDepth=\(String(describing: tuning?.usableBestDepth)), bestDepth=\(String(describing: tuning?.bestDepth)))"
            found.append((name, verdict))
        }
        for (n, v) in found.sorted(by: { $0.0 < $1.0 }) {
            print("MTPPROBE \(n) -> \(v)")
        }
        print("MTPPROBE total=\(found.count)")
    }
}
