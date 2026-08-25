// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// A bundle can declare its own speculative depth in jang_config:
//
//     runtime.mtp_num_speculative_tokens = 2
//     mtp.recommended_num_drafts          = 2
//     mtp.upstream_num_speculative_tokens = 2
//
// Nothing read any of them — grepping Libraries/ returned zero hits — so
// depth came only from `vmlx_mtp_tuning.json` and a bundle shipping a head
// plus a declared depth but no measured row got no depth at all.
//
// The declaration is the PUBLISHER's claim; the tuning row is a measurement
// on THIS machine. The measured row must therefore win wherever it is
// usable, and the two must stay distinguishable in the reason string.

import Foundation
import Testing

@testable import MLXLMCommon

@Suite("jang_config declared MTP depth")
struct JangDeclaredMTPDepthTests {

    private func jangJSON(runtimeDepth: Int?, mtpDepth: Int?) -> String {
        let runtimeExtra = runtimeDepth.map { ", \"mtp_num_speculative_tokens\": \($0)" } ?? ""
        let mtpBlock = mtpDepth.map { ", \"mtp\": {\"recommended_num_drafts\": \($0)}" } ?? ""
        return """
            {"runtime": {"bundle_has_mtp": true, "mtp_layers": 1,
             "mtp_mode": "preserved_enabled"\(runtimeExtra)}\(mtpBlock)}
            """
    }

    private func runtime(_ json: String) throws -> JangRuntime {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("jang-depth-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        try Data(json.utf8).write(to: dir.appendingPathComponent("jang_config.json"))
        return try JangLoader.loadConfig(at: dir).runtime
    }

    @Test func readsTheRuntimeSpelling() throws {
        let r = try runtime(jangJSON(runtimeDepth: 2, mtpDepth: nil))
        #expect(r.mtpDeclaredSpeculativeTokens == 2)
    }

    /// Qwen3.8-27B writes both; other publishers may write only one.
    @Test func fallsBackToTheMtpBlockSpelling() throws {
        let r = try runtime(jangJSON(runtimeDepth: nil, mtpDepth: 3))
        #expect(r.mtpDeclaredSpeculativeTokens == 3)
    }

    /// A bundle with a head but no declared depth must stay nil, not 0 —
    /// a zero would read as "declared, and it is none".
    @Test func noDeclarationStaysNil() throws {
        let r = try runtime(jangJSON(runtimeDepth: nil, mtpDepth: nil))
        #expect(r.mtpDeclaredSpeculativeTokens == nil)
    }

    /// Zero and negative declarations are not depths.
    @Test func nonPositiveDeclarationsAreIgnored() throws {
        #expect(try runtime(jangJSON(runtimeDepth: 0, mtpDepth: nil))
            .mtpDeclaredSpeculativeTokens == nil)
        #expect(try runtime(jangJSON(runtimeDepth: -1, mtpDepth: nil))
            .mtpDeclaredSpeculativeTokens == nil)
    }

    /// When both spellings are present and disagree, the runtime block is the
    /// more specific statement and wins. Pinned so the order is deliberate.
    @Test func runtimeSpellingWinsOverTheMtpBlock() throws {
        let r = try runtime(jangJSON(runtimeDepth: 2, mtpDepth: 3))
        #expect(r.mtpDeclaredSpeculativeTokens == 2)
    }
}
