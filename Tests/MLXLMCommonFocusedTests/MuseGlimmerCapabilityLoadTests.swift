// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// The claim, against a real bundle: a multimodal Muse Glimmer loaded text-only builds no vision
// tower, and its language tower is unchanged.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MLXVLM
import Testing

@Suite("Muse Glimmer capability-driven load (real bundle)")
struct MuseGlimmerCapabilityLoadTests {

    /// The real converted bundle. Absent on a machine without it, so every test here skips rather
    /// than fails — a missing 26 GB model is not a defect in this code.
    static var bundleConfig: URL? {
        let root = URL(fileURLWithPath: NSHomeDirectory())
            .appendingPathComponent("Library/MLModels")
        for org in ["OsaurusAI", "cubiculum", "JANGQ-AI"] {
            let d = root.appendingPathComponent(org)
            guard let kids = try? FileManager.default.contentsOfDirectory(atPath: d.path) else { continue }
            for k in kids where k.hasPrefix("Muse-Glimmer") {
                let c = d.appendingPathComponent(k).appendingPathComponent("config.json")
                if FileManager.default.fileExists(atPath: c.path) { return c }
            }
        }
        return nil
    }

    private func config() throws -> MuseGlimmerConfiguration? {
        guard let url = Self.bundleConfig else { return nil }
        return try JSONDecoder.json5().decode(
            MuseGlimmerConfiguration.self, from: Data(contentsOf: url))
    }

    @Test("the real bundle declares text, image and video")
    func realBundleDeclares() throws {
        guard let cfg = try config() else { return }
        #expect(cfg.visionConfiguration != nil)
        #expect(MuseGlimmer.constructibleModalities(of: cfg) == [.text, .vision, .video])
    }

    /// The measurement that matters: same bundle, two loads, and the text-only one must not carry
    /// the tower. Parameter COUNT is the observable — it is what allocation follows from.
    @Test("text-only construction omits the vision tower entirely")
    func textOnlyOmitsTower() throws {
        guard let cfg = try config() else { return }

        let full = MuseGlimmer(cfg)                                   // everything declared
        let text = try MuseGlimmer(cfg, requesting: [.text])          // narrowed

        func params(_ m: Module) -> Int {
            m.parameters().flattened().reduce(0) { $0 + $1.1.size }
        }
        let pFull = params(full), pText = params(text)

        #expect(text.modalities == [.text])
        #expect(full.modalities == [.text, .vision, .video])
        #expect(pText < pFull, "text-only must allocate strictly less")

        let savedGB = Double(pFull - pText) * 2.0 / 1e9   // fp16 lower bound on the tower
        print("""
              full  : \(pFull) params
              text  : \(pText) params
              spared: \(pFull - pText) params (>= \(String(format: "%.2f", savedGB)) GB at fp16)
              """)
    }

    /// The language tower must be untouched by narrowing — otherwise this changes answers rather
    /// than packaging, and every banked Muse Glimmer row would be invalidated.
    @Test("the language tower is identical in both loads")
    func languageTowerUnchanged() throws {
        guard let cfg = try config() else { return }
        let full = MuseGlimmer(cfg)
        let text = try MuseGlimmer(cfg, requesting: [.text])

        func languageKeys(_ m: Module) -> [String] {
            m.parameters().flattened().map(\.0).filter { $0.hasPrefix("language_model") }.sorted()
        }
        func languageSize(_ m: Module) -> Int {
            m.parameters().flattened().filter { $0.0.hasPrefix("language_model") }
                .reduce(0) { $0 + $1.1.size }
        }
        #expect(languageKeys(full) == languageKeys(text), "the text tower's shape must not change")
        #expect(languageSize(full) == languageSize(text), "…nor its parameter count")
    }

    @Test("a text-only instance exposes no vision keys at all")
    func noVisionKeysWhenNarrowed() throws {
        guard let cfg = try config() else { return }
        let text = try MuseGlimmer(cfg, requesting: [.text])
        let visionish = text.parameters().flattened().map(\.0)
            .filter { $0.contains("vision") }
        #expect(visionish.isEmpty, "leftover vision parameters: \(visionish.prefix(3))")
    }
}
