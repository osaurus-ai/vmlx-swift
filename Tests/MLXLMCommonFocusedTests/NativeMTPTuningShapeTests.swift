// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// The published `vmlx_mtp_tuning.json` nests its row under `native_mtp`
// (see NativeMTPDepth3AutoLaunchTests, which quotes the shipped Qwen3.6
// artifact verbatim). Every Qwen3.8-27B bundle ships the row FLAT instead.
//
// Because `NativeMTPTuningDocument.nativeMTP` is Optional, a flat file
// decoded "successfully" to nil. The bundle inspector then reported
// `tuning_file_missing` for a file sitting right there on disk, so
// `hasUsableNativeMTPTuning` was false, `canAutoLaunch` was false, and
// native MTP could NEVER launch on those bundles — no matter what the
// artifact said. The visible symptom was "Bundle does not have usable
// vmlx_mtp_tuning.json production tuning", which sends the diagnosis after
// the tuning VALUES when the problem is the file's SHAPE.

import Foundation
import Testing

@testable import MLXLMCommon

@Suite("Native MTP tuning file shape")
struct NativeMTPTuningShapeTests {

    private static let row = """
        "best_depth": 2, "validated": true, "output_equivalent": true,
        "baseline_tok_s": 38.82, "best_tok_s": 41.39, "speedup_vs_baseline": 1.066
        """

    private func write(_ json: String) throws -> URL {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("mtp-shape-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try Data(json.utf8).write(to: dir.appendingPathComponent("vmlx_mtp_tuning.json"))
        return dir
    }

    /// The published shape must keep working.
    @Test func nestedShapeLoads() throws {
        let dir = try write("{\"native_mtp\": {\(Self.row)}}")
        defer { try? FileManager.default.removeItem(at: dir) }
        let tuning = try #require(MTPBundleInspector.tuningForTesting(at: dir))
        #expect(tuning.usableBestDepth == 2)
    }

    /// The shape every Qwen3.8-27B bundle actually ships. Before this it
    /// silently produced nil and the bundle was reported as having no tuning.
    @Test func flatShapeAlsoLoads() throws {
        let dir = try write("{\(Self.row)}")
        defer { try? FileManager.default.removeItem(at: dir) }
        let tuning = try #require(
            MTPBundleInspector.tuningForTesting(at: dir),
            "a flat row must load; it was silently dropped before")
        #expect(tuning.usableBestDepth == 2)
    }

    /// No file at all is still no tuning — leniency must not invent one.
    @Test func absentFileYieldsNothing() throws {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("mtp-shape-empty-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(MTPBundleInspector.tuningForTesting(at: dir) == nil)
    }

    /// Garbage must not masquerade as a usable row.
    @Test func unparseableFileYieldsNothing() throws {
        let dir = try write("{ not json at all ")
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(MTPBundleInspector.tuningForTesting(at: dir) == nil)
    }

    /// A well-formed file carrying no depth is not a tuning row either —
    /// otherwise `{}` would decode to an all-defaults row and read as real.
    @Test func aRowWithoutADepthIsNotTuning() throws {
        let dir = try write("{\"validated\": true}")
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(MTPBundleInspector.tuningForTesting(at: dir) == nil)
    }
}
