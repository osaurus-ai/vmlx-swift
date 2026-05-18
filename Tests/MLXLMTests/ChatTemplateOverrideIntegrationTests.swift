// Copyright 2025 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Iter 51: integration test for the `VMLX_CHAT_TEMPLATE_OVERRIDE`
// escape hatch shipped in iter 50.
//
// Iter 50's unit coverage proved the minimal Gemma-4 template parses
// and renders at the Jinja layer. What was NOT directly tested was
// the end-to-end path: loading a real tokenizer via the
// `#huggingFaceTokenizerLoader()` macro-expanded bridge, setting the
// env var, calling `applyChatTemplate(messages:tools:additionalContext:)`,
// and verifying the bridge actually routes through the override.
//
// This test closes that gap. Uses the cached Qwen3-0.6B-8bit tokenizer
// (small, fast, deterministic) — we're testing the bridge's env-var
// plumbing, not anything Qwen-specific.

import Foundation
import XCTest
@preconcurrency import VMLXTokenizers

@testable import MLXHuggingFace
@testable import MLXLMCommon

final class ChatTemplateOverrideIntegrationTests: XCTestCase {

    private func tokenizerDirectory() -> URL? {
        // Well-known cached snapshot from the rest of the bench suite.
        // Resolved relative to the user's HF cache so it works for any
        // developer who has pulled Qwen3-0.6B; falls through to XCTSkip
        // on machines that haven't cached it. Override with
        // `VMLX_TEST_TOKENIZER_DIR` to point at a different tokenizer.
        let env = ProcessInfo.processInfo.environment
        if let override = env["VMLX_TEST_TOKENIZER_DIR"], !override.isEmpty {
            return URL(fileURLWithPath: override)
        }
        let hfCache = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/huggingface/hub")
        let repoDir = hfCache.appendingPathComponent(
            "models--mlx-community--Qwen3-0.6B-8bit/snapshots")
        // Walk snapshot subdirs — the hash is content-addressed so we
        // can't hardcode it.
        guard let snapshots = try? FileManager.default.contentsOfDirectory(
            at: repoDir, includingPropertiesForKeys: nil) else { return nil }
        for snap in snapshots {
            let tokJSON = snap.appendingPathComponent("tokenizer.json")
            if FileManager.default.fileExists(atPath: tokJSON.path) {
                return snap
            }
        }
        return nil
    }

    /// Write a minimal template to a temp file that produces a
    /// recognisable, unambiguous output. Returns the file URL.
    private func writeMarkerTemplate() throws -> URL {
        // The marker is verbose enough that the tokenizer-shipped
        // template CANNOT coincidentally produce it.
        let template = """
        {{- 'OVERRIDE_FIRED:' -}}
        {%- for m in messages -%}
            [{{ m['role'] }}={{ m['content'] }}]
        {%- endfor -%}
        """
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("iter51-override-\(UUID().uuidString).jinja")
        try template.data(using: .utf8)!.write(to: tmp)
        return tmp
    }

    // MARK: - Override honoured

    /// When `VMLX_CHAT_TEMPLATE_OVERRIDE` points to a valid template,
    /// the bridge must use it. Compares decoded tokens to the marker.
    func testOverrideFiresWhenEnvVarPointsAtValidTemplate() async throws {
        guard let dir = tokenizerDirectory() else {
            throw XCTSkip("Qwen3-0.6B tokenizer snapshot not cached.")
        }
        let templateURL = try writeMarkerTemplate()
        defer { try? FileManager.default.removeItem(at: templateURL) }

        setenv("VMLX_CHAT_TEMPLATE_OVERRIDE", templateURL.path, 1)
        defer { unsetenv("VMLX_CHAT_TEMPLATE_OVERRIDE") }

        let loader = #huggingFaceTokenizerLoader()
        let tokenizer = try await loader.load(from: dir)

        let messages: [[String: any Sendable]] = [
            ["role": "system", "content": "sysmsg"],
            ["role": "user", "content": "hello"],
        ]
        let tokenIds = try tokenizer.applyChatTemplate(
            messages: messages, tools: nil, additionalContext: nil)
        let decoded = tokenizer.decode(tokenIds: tokenIds, skipSpecialTokens: false)

        XCTAssertTrue(decoded.contains("OVERRIDE_FIRED:"),
            "Override template must be used when VMLX_CHAT_TEMPLATE_OVERRIDE is set. " +
            "Decoded: \(decoded)")
        XCTAssertTrue(decoded.contains("[system=sysmsg]"),
            "Override must receive the passed-in messages. Decoded: \(decoded)")
        XCTAssertTrue(decoded.contains("[user=hello]"))
    }

    // MARK: - Default unchanged

    /// With the env var UNSET, `applyChatTemplate` must fall through to
    /// the tokenizer's shipped template and NOT include the marker.
    /// This is the regression guard — catches an override path that
    /// fires unconditionally.
    func testDefaultPathUnchangedWhenEnvUnset() async throws {
        guard let dir = tokenizerDirectory() else {
            throw XCTSkip("Qwen3-0.6B tokenizer snapshot not cached.")
        }
        unsetenv("VMLX_CHAT_TEMPLATE_OVERRIDE")
        let loader = #huggingFaceTokenizerLoader()
        let tokenizer = try await loader.load(from: dir)

        let messages: [[String: any Sendable]] = [
            ["role": "user", "content": "hello"],
        ]
        let tokenIds = try tokenizer.applyChatTemplate(
            messages: messages, tools: nil, additionalContext: nil)
        let decoded = tokenizer.decode(tokenIds: tokenIds, skipSpecialTokens: false)

        XCTAssertFalse(decoded.contains("OVERRIDE_FIRED:"),
            "Marker must NOT appear when env var unset. Decoded: \(decoded)")
    }

    /// Set the env var but point at a non-existent file. The bridge
    /// must silently fall back to the shipped template — better to
    /// produce SOMETHING than crash because the operator mistyped a
    /// path. Pinned here so a "fail-fast on missing override" refactor
    /// would be an intentional decision, not accidental.
    func testMissingOverrideFileFallsBackToShippedTemplate() async throws {
        guard let dir = tokenizerDirectory() else {
            throw XCTSkip("Qwen3-0.6B tokenizer snapshot not cached.")
        }
        setenv("VMLX_CHAT_TEMPLATE_OVERRIDE",
               "/tmp/nonexistent-\(UUID().uuidString).jinja", 1)
        defer { unsetenv("VMLX_CHAT_TEMPLATE_OVERRIDE") }

        let loader = #huggingFaceTokenizerLoader()
        let tokenizer = try await loader.load(from: dir)

        let messages: [[String: any Sendable]] = [
            ["role": "user", "content": "hello"],
        ]
        // Shouldn't throw — bridge reads the file optionally and falls
        // back on nil.
        let tokenIds = try tokenizer.applyChatTemplate(
            messages: messages, tools: nil, additionalContext: nil)
        XCTAssertFalse(tokenIds.isEmpty,
            "Fallback to shipped template must produce a non-empty tokenization.")
    }

    /// Empty env var is the same as unset — treat as "no override".
    func testEmptyOverrideEnvTreatedAsUnset() async throws {
        guard let dir = tokenizerDirectory() else {
            throw XCTSkip("Qwen3-0.6B tokenizer snapshot not cached.")
        }
        setenv("VMLX_CHAT_TEMPLATE_OVERRIDE", "", 1)
        defer { unsetenv("VMLX_CHAT_TEMPLATE_OVERRIDE") }

        let loader = #huggingFaceTokenizerLoader()
        let tokenizer = try await loader.load(from: dir)

        let messages: [[String: any Sendable]] = [
            ["role": "user", "content": "hello"],
        ]
        let tokenIds = try tokenizer.applyChatTemplate(
            messages: messages, tools: nil, additionalContext: nil)
        let decoded = tokenizer.decode(tokenIds: tokenIds, skipSpecialTokens: false)
        XCTAssertFalse(decoded.contains("OVERRIDE_FIRED:"),
            "Empty env var must NOT trigger override. Decoded: \(decoded)")
    }

    // MARK: - Shipped Gemma4Minimal roundtrips via the bridge

    /// The template shipped in iter 50 must also work through the
    /// bridge path end-to-end (not just isolated VMLXJinja.Template).
    /// Catches packaging / whitespace / comment-stripping issues that
    /// might sneak in between disk and render.
    func testShippedGemma4MinimalTemplateRendersViaBridge() async throws {
        guard let dir = tokenizerDirectory() else {
            throw XCTSkip("Qwen3-0.6B tokenizer snapshot not cached.")
        }
        // Locate the shipped template file.
        var searchPath = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        var templateURL: URL? = nil
        for _ in 0..<6 {
            let candidate = searchPath.appendingPathComponent(
                "Libraries/MLXLMCommon/ChatTemplates/Gemma4Minimal.jinja")
            if FileManager.default.fileExists(atPath: candidate.path) {
                templateURL = candidate
                break
            }
            searchPath = searchPath.deletingLastPathComponent()
        }
        guard let templateURL else {
            throw XCTSkip("Gemma4Minimal.jinja not found relative to test source.")
        }

        setenv("VMLX_CHAT_TEMPLATE_OVERRIDE", templateURL.path, 1)
        defer { unsetenv("VMLX_CHAT_TEMPLATE_OVERRIDE") }

        let loader = #huggingFaceTokenizerLoader()
        let tokenizer = try await loader.load(from: dir)

        let messages: [[String: any Sendable]] = [
            ["role": "system", "content": "sys"],
            ["role": "user", "content": "hi"],
            ["role": "assistant", "content": "hello"],
            ["role": "user", "content": "again"],
        ]
        let tokenIds = try tokenizer.applyChatTemplate(
            messages: messages, tools: nil, additionalContext: nil)
        let decoded = tokenizer.decode(tokenIds: tokenIds, skipSpecialTokens: false)
        XCTAssertTrue(decoded.contains("<|turn>system\nsys<turn|>"),
            "Gemma4Minimal must emit the system-turn delimiter. Decoded: \(decoded)")
        XCTAssertTrue(decoded.contains("<|turn>user\nhi<turn|>"),
            "Gemma4Minimal must emit user-turn delimiters. Decoded: \(decoded)")
        XCTAssertTrue(decoded.contains("<|turn>model\nhello<turn|>"),
            "Gemma4Minimal maps assistant→model. Decoded: \(decoded)")
    }
}
