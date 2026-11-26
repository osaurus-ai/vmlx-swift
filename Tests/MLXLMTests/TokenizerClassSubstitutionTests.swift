// Copyright 2025 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Iter 59 regression coverage: JangLoader.resolveTokenizerClassSubstitution
// rewrites tokenizer_config.json entries that swift-transformers 0.1.21
// does not know (notably "TokenizersBackend" used by mlx-community
// Qwen3.5-VL-9B) so loads succeed instead of throwing
// `unsupportedTokenizer`.
//
// Tests build a fake tokenizer directory in a tmp dir, invoke the
// substitution, and verify the returned shim dir has a rewritten
// tokenizer_config.json + symlinks to the other files. No real
// tokenizer load.

import Foundation
import XCTest

@testable import MLXLMCommon

final class TokenizerClassSubstitutionTests: XCTestCase {

    private var tmpRoot: URL!

    override func setUpWithError() throws {
        try super.setUpWithError()
        tmpRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent("TokenizerClassSub-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: tmpRoot, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        if let tmpRoot {
            try? FileManager.default.removeItem(at: tmpRoot)
        }
        try super.tearDownWithError()
    }

    /// Build a fake tokenizer directory with a specific tokenizer_class.
    @discardableResult
    private func writeFakeTokenizerDir(
        class cls: String,
        extraFiles: [String] = ["tokenizer.json", "chat_template.jinja"]
    ) throws -> URL {
        let dir = tmpRoot.appendingPathComponent("fake-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let cfg: [String: Any] = [
            "tokenizer_class": cls,
            "eos_token": "<|im_end|>",
            "model_max_length": 131072,
        ]
        let cfgData = try JSONSerialization.data(
            withJSONObject: cfg, options: [.prettyPrinted, .sortedKeys])
        try cfgData.write(to: dir.appendingPathComponent("tokenizer_config.json"))
        for name in extraFiles {
            let content = "(fake placeholder for \(name))"
            try content.data(using: .utf8)!
                .write(to: dir.appendingPathComponent(name))
        }
        return dir
    }

    private func readClass(from dir: URL) -> String? {
        let cfgURL = dir.appendingPathComponent("tokenizer_config.json")
        guard let data = try? Data(contentsOf: cfgURL),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return nil }
        return json["tokenizer_class"] as? String
    }

    // MARK: - Identity pass-through

    /// Already-supported classes return the input URL unchanged (no shim).
    func testSupportedClassReturnsInputUnchanged() throws {
        for knownClass in ["Qwen2Tokenizer", "LlamaTokenizer", "GemmaTokenizer"] {
            let dir = try writeFakeTokenizerDir(class: knownClass)
            let resolved = JangLoader.resolveTokenizerClassSubstitution(for: dir)
            XCTAssertEqual(
                resolved.standardizedFileURL.path,
                dir.standardizedFileURL.path,
                "Known class \(knownClass) must not produce a shim.")
        }
    }

    /// `Fast` suffix is stripped by swift-transformers; treat as supported.
    func testFastSuffixedClassesNotShimmed() throws {
        let dir = try writeFakeTokenizerDir(class: "Qwen2TokenizerFast")
        let resolved = JangLoader.resolveTokenizerClassSubstitution(for: dir)
        XCTAssertEqual(
            resolved.standardizedFileURL.path,
            dir.standardizedFileURL.path,
            "Fast-suffixed class must not trigger substitution — swift-transformers " +
            "strips `Fast` before knownTokenizers lookup.")
    }

    // MARK: - Default substitution table

    /// `TokenizersBackend` → `Qwen2Tokenizer`. This is the case that
    /// unblocks mlx-community/Qwen3.5-VL-9B-8bit.
    func testTokenizersBackendRewriteToQwen2Tokenizer() throws {
        let dir = try writeFakeTokenizerDir(class: "TokenizersBackend")
        let shim = JangLoader.resolveTokenizerClassSubstitution(for: dir)

        XCTAssertNotEqual(
            shim.standardizedFileURL.path,
            dir.standardizedFileURL.path,
            "Shim dir must be a different path from the input.")
        XCTAssertEqual(readClass(from: shim), "Qwen2Tokenizer",
            "TokenizersBackend must rewrite to Qwen2Tokenizer.")
    }

    func testTikTokenTokenizerRewriteToQwen2Tokenizer() throws {
        let dir = try writeFakeTokenizerDir(class: "TikTokenTokenizer")
        let shim = JangLoader.resolveTokenizerClassSubstitution(for: dir)

        XCTAssertNotEqual(
            shim.standardizedFileURL.path,
            dir.standardizedFileURL.path,
            "TikTokenTokenizer should produce a shim for swift-transformers.")
        XCTAssertEqual(readClass(from: shim), "Qwen2Tokenizer")
    }

    func testQwen3_5TokenizerRewriteToQwen2Tokenizer() throws {
        let dir = try writeFakeTokenizerDir(class: "Qwen3_5Tokenizer")
        let shim = JangLoader.resolveTokenizerClassSubstitution(for: dir)

        XCTAssertNotEqual(
            shim.standardizedFileURL.path,
            dir.standardizedFileURL.path,
            "Qwen3_5Tokenizer should produce a shim for swift-transformers.")
        XCTAssertEqual(readClass(from: shim), "Qwen2Tokenizer")
    }

    /// Every other file in the original dir must be reachable via the
    /// shim — they're symlinked, not copied.
    func testShimSymlinksOtherFiles() throws {
        let dir = try writeFakeTokenizerDir(
            class: "TokenizersBackend",
            extraFiles: ["tokenizer.json", "chat_template.jinja", "special_tokens_map.json"])
        let shim = JangLoader.resolveTokenizerClassSubstitution(for: dir)

        for expected in ["tokenizer.json", "chat_template.jinja", "special_tokens_map.json"] {
            let link = shim.appendingPathComponent(expected)
            XCTAssertTrue(FileManager.default.fileExists(atPath: link.path),
                "Shim must expose \(expected).")
            // Symlink (not a copy) — verify via destinationOfSymbolicLink.
            let dest = try FileManager.default.destinationOfSymbolicLink(atPath: link.path)
            XCTAssertFalse(dest.isEmpty,
                "\(expected) in shim must be a symlink, not a copy.")
        }
    }

    /// Unknown class with no mapping: pass through unchanged so the
    /// downstream loader surfaces its clear error instead of us
    /// silently picking the wrong substitute.
    func testUnknownClassWithNoMappingReturnsInputUnchanged() throws {
        let dir = try writeFakeTokenizerDir(class: "SomeNeverSeenTokenizer")
        let resolved = JangLoader.resolveTokenizerClassSubstitution(for: dir)
        XCTAssertEqual(
            resolved.standardizedFileURL.path,
            dir.standardizedFileURL.path,
            "Unmapped unknown class must NOT silently shim — the loader " +
            "should raise its own error so the operator knows what's wrong.")
    }

    // MARK: - Env-var override

    /// `VMLX_TOKENIZER_CLASS_OVERRIDE` forces a specific target even for
    /// already-supported classes.
    func testEnvOverrideForcesSubstitution() throws {
        let dir = try writeFakeTokenizerDir(class: "Qwen2Tokenizer")
        let shim = JangLoader.resolveTokenizerClassSubstitution(
            for: dir, overrideClass: "LlamaTokenizer")
        XCTAssertNotEqual(
            shim.standardizedFileURL.path,
            dir.standardizedFileURL.path,
            "Explicit override must create a shim even for already-supported classes.")
        XCTAssertEqual(readClass(from: shim), "LlamaTokenizer")
    }

    /// Missing tokenizer_config.json: return unchanged.
    func testMissingConfigReturnsInputUnchanged() throws {
        let dir = tmpRoot.appendingPathComponent("empty-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let resolved = JangLoader.resolveTokenizerClassSubstitution(for: dir)
        XCTAssertEqual(
            resolved.standardizedFileURL.path,
            dir.standardizedFileURL.path,
            "Dir with no tokenizer_config.json must return input — letting the " +
            "downstream loader surface the missing-file error.")
    }

    /// Corrupt tokenizer_config.json: return unchanged (don't crash).
    func testCorruptConfigReturnsInputUnchanged() throws {
        let dir = tmpRoot.appendingPathComponent("corrupt-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try "not json{".data(using: .utf8)!
            .write(to: dir.appendingPathComponent("tokenizer_config.json"))
        let resolved = JangLoader.resolveTokenizerClassSubstitution(for: dir)
        XCTAssertEqual(
            resolved.standardizedFileURL.path,
            dir.standardizedFileURL.path,
            "Corrupt config must not crash — return input and let loader error.")
    }
}
