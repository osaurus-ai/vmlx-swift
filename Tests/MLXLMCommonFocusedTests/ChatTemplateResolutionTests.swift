// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// A chat template that fails to resolve does not fail loudly: the runtime falls back to a generic
// format and the model goes on producing fluent, plausible text through the wrong prompt shape.
// Generation therefore cannot distinguish "template applied" from "template silently replaced",
// which is why this is asserted on the RESOLUTION rather than on any output.
//
// The concrete case: several VLMs ship their template ONLY as a sibling `chat_template.jinja` and
// leave `tokenizer_config.json` without one. GLM-4.5V is such a bundle.

import Foundation
import MLXLMCommon
import Testing

@Suite("Chat template resolution")
struct ChatTemplateResolutionTests {

    private func makeBundle(
        sidecar: String?, inline: String?, file: StaticString = #filePath
    ) throws -> URL {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("cts-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        if let sidecar {
            try sidecar.write(
                to: dir.appendingPathComponent("chat_template.jinja"), atomically: true,
                encoding: .utf8)
        }
        var config: [String: Any] = ["model_max_length": 4096]
        if let inline { config["chat_template"] = inline }
        try JSONSerialization.data(withJSONObject: config)
            .write(to: dir.appendingPathComponent("tokenizer_config.json"))
        return dir
    }

    @Test("a sidecar-only bundle resolves, and reports where it came from")
    func sidecarOnlyResolves() throws {
        let dir = try makeBundle(sidecar: "SIDECAR", inline: nil)
        let resolved = try #require(ChatTemplateResolver.resolve(modelDirectory: dir))
        #expect(resolved.text == "SIDECAR")
        #expect(resolved.source == .sidecar)
    }

    @Test("an inline-only bundle resolves from tokenizer_config")
    func inlineOnlyResolves() throws {
        let dir = try makeBundle(sidecar: nil, inline: "INLINE")
        let resolved = try #require(ChatTemplateResolver.resolve(modelDirectory: dir))
        #expect(resolved.text == "INLINE")
        #expect(resolved.source == .tokenizerConfig)
    }

    /// Precedence is observable, not incidental: a converter that rewrites the template writes the
    /// sidecar and often leaves a stale inline copy behind.
    @Test("the sidecar wins when both are present")
    func sidecarWins() throws {
        let dir = try makeBundle(sidecar: "SIDECAR", inline: "INLINE")
        let resolved = try #require(ChatTemplateResolver.resolve(modelDirectory: dir))
        #expect(resolved.text == "SIDECAR")
        #expect(resolved.source == .sidecar)
    }

    @Test("a bundle carrying neither resolves to nil rather than to something generic")
    func neitherResolvesToNil() throws {
        let dir = try makeBundle(sidecar: nil, inline: nil)
        #expect(ChatTemplateResolver.resolve(modelDirectory: dir) == nil)
    }

    /// The guard that matters on real bundles, stated without a per-family marker table: whatever a
    /// bundle ships as its sidecar is EXACTLY what resolution returns. A silent replacement — the
    /// failure this whole file exists for — breaks this for every bundle at once.
    @Test("on every local bundle, a shipped sidecar is returned verbatim")
    func localBundlesResolveToTheirOwnSidecar() throws {
        let root = URL(fileURLWithPath: NSHomeDirectory())
            .appendingPathComponent("Library/MLModels")
        let fm = FileManager.default
        guard let orgs = try? fm.contentsOfDirectory(atPath: root.path) else { return }

        var checkedSidecar = 0, checkedInline = 0, neither = 0
        for org in orgs.sorted() {
            let orgDir = root.appendingPathComponent(org)
            guard let kids = try? fm.contentsOfDirectory(atPath: orgDir.path) else { continue }
            for kid in kids.sorted() {
                let dir = orgDir.appendingPathComponent(kid)
                guard fm.fileExists(atPath: dir.appendingPathComponent("config.json").path)
                else { continue }
                let sidecarURL = dir.appendingPathComponent("chat_template.jinja")
                let resolved = ChatTemplateResolver.resolve(modelDirectory: dir)
                if let shipped = try? String(contentsOf: sidecarURL, encoding: .utf8) {
                    let r = try #require(resolved, "\(kid): ships a sidecar but resolved to nil")
                    #expect(r.source == .sidecar, "\(kid)")
                    #expect(r.text == shipped, "\(kid): resolved template is not the shipped one")
                    checkedSidecar += 1
                } else if resolved?.source == .tokenizerConfig {
                    checkedInline += 1
                } else {
                    neither += 1
                }
            }
        }
        // A bundle-gated assertion that never runs looks exactly like one that passed.
        print(
            "chat template resolution: \(checkedSidecar) sidecar, \(checkedInline) inline, "
                + "\(neither) neither")
        #expect(checkedSidecar + checkedInline > 0, "no local bundle carried a chat template")
    }
}
