// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// DFlash 2 must be INERT unless a user selected a drafter AND that
// drafter fits the bundle being served. Those are separate failures with
// the same blast radius: a drafter that silently attaches to the wrong
// model produces fluent nonsense (it borrows the target's LM head, so a
// vocabulary mismatch does not throw — it indexes a different token
// space).
//
// The synthetic-config version of this lives in
// DFlash2DrafterSelectionTests. This suite is the one that runs against
// REAL bundles on this machine, because a hand-written config can agree
// with the code by construction while a shipped bundle disagrees.
//
//   VMLX_DFLASH2_OTHER_TARGETS=/path/a,/path/b  swift test --filter DFlash2AppliesOnlyToItsTargetTests

import Foundation
import XCTest
@testable import MLXLMCommon

final class DFlash2AppliesOnlyToItsTargetTests: XCTestCase {

    private static var drafterURL: URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("models/Qwen3.8-27B-DFlash2")
    }

    /// Bundles that must NOT be served by the Qwen3.8 drafter. Defaults to
    /// the Ornith 1.5 line, which is the same `qwen3_5` hybrid family and
    /// therefore the most dangerous near-miss: it loads through the same
    /// model class, so only the config check stands between it and a
    /// mis-attached drafter.
    private static var otherTargets: [URL] {
        if let raw = ProcessInfo.processInfo.environment["VMLX_DFLASH2_OTHER_TARGETS"] {
            return raw.split(separator: ",").map { URL(fileURLWithPath: String($0)) }
        }
        let home = FileManager.default.homeDirectoryForCurrentUser
        return [
            "models/JANGQ-AI/Ornith-1.5-9B-JANG_4D",
            "models/JANGQ-AI/Ornith-1.5-9B-MXFP8",
            "models/JANGQ-AI/Ornith-1.5-35B-A3B-JANG_4M",
        ].map { home.appendingPathComponent($0) }
    }

    private func settings(drafterPath: String?) -> VMLXServerRuntimeSettings {
        var s = VMLXServerRuntimeSettings()
        s.mtp.dflash2DrafterPath = drafterPath
        return s
    }

    private func configData(_ bundle: URL) throws -> Data {
        try Data(contentsOf: bundle.appendingPathComponent("config.json"))
    }

    /// No drafter selected — the default — must never produce a DFlash 2
    /// strategy for ANY bundle.
    func testDefaultSettingsNeverSelectDFlash2() throws {
        let s = settings(drafterPath: nil)
        XCTAssertNil(s.mtp.dflash2DrafterPath, "a fresh settings object must not carry a drafter")
        for bundle in Self.otherTargets where FileManager.default.fileExists(atPath: bundle.path) {
            let data = try configData(bundle)
            XCTAssertNil(
                s.resolvedDFlash2Selection(configData: data),
                "\(bundle.lastPathComponent) selected a drafter with none configured")
            let strategy = s.resolvedMTPDraftStrategy(
                configData: data, jangConfig: nil, status: nil)
            if case .dflash2 = strategy {
                XCTFail("\(bundle.lastPathComponent) resolved to DFlash 2 with no drafter set")
            }
        }
    }

    /// A drafter IS selected, but the served bundle is a different model.
    /// It must be rejected with a readable reason, and the strategy must
    /// fall through to whatever that bundle would have used on its own.
    func testSelectedDrafterIsRejectedForOtherModels() throws {
        guard DFlash2Loader.looksLikeDFlash2Drafter(at: Self.drafterURL) else {
            throw XCTSkip("No DFlash 2 drafter at \(Self.drafterURL.path)")
        }
        let s = settings(drafterPath: Self.drafterURL.path)
        var checked = 0
        for bundle in Self.otherTargets where FileManager.default.fileExists(atPath: bundle.path) {
            checked += 1
            let data = try configData(bundle)
            XCTAssertNil(
                s.resolvedDFlash2Selection(configData: data),
                "\(bundle.lastPathComponent) accepted the Qwen3.8 drafter")
            let reason = s.dflash2RejectionReason(configData: data)
            XCTAssertNotNil(
                reason,
                "\(bundle.lastPathComponent) was rejected with no reason to show the user")
            print("[applies-only] \(bundle.lastPathComponent): \(reason ?? "-")")
            let strategy = s.resolvedMTPDraftStrategy(
                configData: data, jangConfig: nil, status: nil)
            if case .dflash2 = strategy {
                XCTFail("\(bundle.lastPathComponent) still resolved to DFlash 2")
            }
        }
        XCTAssertGreaterThan(checked, 0, "no comparison bundles present — test proved nothing")
    }

    /// The positive control: the drafter's OWN target must be accepted.
    /// Without this, the two tests above would pass on a drafter that
    /// matches nothing at all.
    func testDrafterIsAcceptedForItsOwnTarget() throws {
        guard DFlash2Loader.looksLikeDFlash2Drafter(at: Self.drafterURL) else {
            throw XCTSkip("No DFlash 2 drafter at \(Self.drafterURL.path)")
        }
        let target = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("models/JANGQ-AI/Qwen3.8-27B-JANG_4D")
        guard FileManager.default.fileExists(atPath: target.path) else {
            throw XCTSkip("No Qwen3.8-27B target bundle")
        }
        let s = settings(drafterPath: Self.drafterURL.path)
        let data = try configData(target)
        XCTAssertNil(
            s.dflash2RejectionReason(configData: data),
            "the drafter's own target was rejected")
        XCTAssertNotNil(
            s.resolvedDFlash2Selection(configData: data),
            "the drafter's own target did not select it")
    }
}
