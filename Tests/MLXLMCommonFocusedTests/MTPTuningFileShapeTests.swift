// Copyright © 2026 Osaurus AI. All rights reserved.
//
// `vmlx_mtp_tuning.json` is written in two shapes, and the reader understood
// one of them.
//
// PR #278's depth sweep writes the measurement NESTED under `native_mtp`.
// `stamp_qwen38_27b` writes the same keys FLAT at the top level. Only nested
// decoded, so every flat file was silently ignored — indistinguishable, to the
// runtime and to `rejectionReason`, from a bundle with no tuning file at all.
//
// Measured across ~/models on 2026-08-22 with the real inspector and the real
// policy: of 15 bundles carrying a complete MTP artifact, exactly ONE
// launched. Five Qwen3.8-27B bundles carried a flat `best_depth: 1` stamp the
// runtime could not see.
//
// The second half matters as much as the first: reading the flat file must NOT
// start launching unmeasured stamps. The launch gate is deliberately
// fail-closed — depth has to come from artifact-local measurement, never from
// a name or a hopeful default.

import Foundation
import Testing

@testable import MLXLMCommon

@Suite("MTP tuning file shape")
struct MTPTuningFileShapeTests {

    private func decode(_ json: String) throws -> NativeMTPTuning? {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("mtptune-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        try json.write(
            to: dir.appendingPathComponent(NativeMTPTuning.fileName),
            atomically: true,
            encoding: .utf8)
        return try MTPBundleInspector.nativeMTPTuningForTesting(from: dir)
    }

    /// The shape the sweep writes.
    @Test("nested native_mtp still decodes")
    func nestedShapeDecodes() throws {
        let t = try decode(#"""
        {"native_mtp": {"best_depth": 3, "validated": true, "output_equivalent": true,
         "blocked": false, "baseline_tok_s": 17.5, "best_tok_s": 25.1,
         "speedup_vs_baseline": 1.43}}
        """#)
        #expect(t?.bestDepth == 3)
        #expect(t?.usableBestDepth == 3)
    }

    /// The shape the stamper writes — silently invisible before this.
    @Test("flat top-level keys decode too")
    func flatShapeDecodes() throws {
        let t = try decode(#"""
        {"best_depth": 1, "blocked": false, "quantization_mode": "affine",
         "quantization_bits": 2, "model_types": ["qwen3_5"]}
        """#)
        #expect(t != nil, "a flat tuning file must not read as 'no tuning file'")
        #expect(t?.bestDepth == 1)
    }

    /// …and reading it must not make it launchable. This is the shipped state
    /// of five Qwen3.8-27B bundles: a `best_depth: 1` stamp whose own note says
    /// "Conservative UNMEASURED default … run a depth sweep".
    @Test("a flat UNMEASURED stamp still refuses to launch")
    func flatUnmeasuredStampStillRefusesToLaunch() throws {
        let t = try decode(#"""
        {"best_depth": 1, "blocked": false,
         "note": "Conservative UNMEASURED default: 1 draft/step."}
        """#)
        #expect(t?.bestDepth == 1)
        #expect(
            t?.usableBestDepth == nil,
            "an unmeasured stamp must never auto-launch; depth comes from measurement")
    }

    /// Each missing measurement field independently blocks the launch, so a
    /// partially-filled stamp cannot sneak through.
    ///
    /// Built by omitting one key at a time rather than by editing a JSON
    /// string — a first cut did string surgery, one pattern silently failed to
    /// match because of a newline, and the "drop" case still contained the key
    /// it claimed to drop. A fixture that does not actually remove the thing
    /// tests nothing.
    @Test func everyMeasurementFieldIsRequired() throws {
        let fields: [String: String] = [
            "validated": "true",
            "output_equivalent": "true",
            "baseline_tok_s": "20.0",
            "best_tok_s": "28.0",
            "speedup_vs_baseline": "1.4",
        ]

        func json(omitting omitted: String?) -> String {
            var pairs = ["\"best_depth\": 1", "\"blocked\": false"]
            for (k, v) in fields.sorted(by: { $0.key < $1.key }) where k != omitted {
                pairs.append("\"\(k)\": \(v)")
            }
            return "{" + pairs.joined(separator: ", ") + "}"
        }

        #expect(
            try decode(json(omitting: nil))?.usableBestDepth == 1,
            "a fully measured stamp must launch, or the omission cases below are vacuous")

        for key in fields.keys.sorted() {
            let body = json(omitting: key)
            #expect(!body.contains(key), "fixture failed to omit \(key)")
            #expect(
                try decode(body)?.usableBestDepth == nil,
                "omitting \(key) must block the launch")
        }
    }

    /// A speedup that is not actually a speedup must not launch. Shipping MTP
    /// that is slower than plain decode is the exact complaint that started
    /// the MTP work.
    @Test("a non-speedup never launches")
    func nonSpeedupNeverLaunches() throws {
        let t = try decode(#"""
        {"best_depth": 2, "blocked": false, "validated": true, "output_equivalent": true,
         "baseline_tok_s": 30.0, "best_tok_s": 29.0, "speedup_vs_baseline": 0.97}
        """#)
        #expect(t?.usableBestDepth == nil)
    }
}
