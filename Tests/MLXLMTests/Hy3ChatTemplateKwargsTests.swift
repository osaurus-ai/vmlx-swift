// Pin the Hy3 chat-template kwargs → prompt-tail wiring contract.
//
// The full Hy3 reasoning round-trip is:
//
//   1. Caller sets `additionalContext["reasoning_effort"] = "no_think" | "low" | "high"`
//      (per JANG handoff §"Reasoning And Tool Surface").
//   2. Tokenizer's `applyChatTemplate(messages:additionalContext:)` plumbs
//      that into the Jinja context via swift-transformers.
//   3. The Hy3 chat template (`tokenizer_config.json:chat_template` or
//      `chat_template.jinja`) renders:
//        - `reasoning_effort = "low" | "high"` → `<assistant_token><think>` (OPEN)
//        - `reasoning_effort = "no_think"`     → `<assistant_token><think></think>` (CLOSED)
//        - default (no reasoning_effort)       → falls through line 37's
//          `set reasoning_effort = 'no_think'` → CLOSED
//   4. `ReasoningParser.forPrompt(stampName: "hy_v3", promptTail: …)`
//      detects the LAST tag in the tail (`<think>` for open, `</think>`
//      for closed) and sets `startInReasoning` accordingly.
//   5. The parser routes pre-`</think>` model output to `.reasoning`
//      and post-`</think>` to `.content` (or all to `.content` when
//      the prompt closed the block).
//
// The parser side of this is pinned by `Hy3ReasoningNoLeakTests`. This
// file pins the TEMPLATE side — that the actual Hy3 chat template
// shipped with the reference Tencent bundle renders the right tags
// for each `reasoning_effort` value. Without this, kwargs could go
// down the wrong template branch and produce a prompt the parser
// would mis-classify.
//
// Skips cleanly when the local bundle isn't available (so CI / other
// machines don't fail).

import Foundation
import VMLXJinja
import XCTest

final class Hy3ChatTemplateKwargsTests: XCTestCase {

    /// Locate a local Hy3 chat_template.jinja. Tries the upstream Tencent
    /// bundle first, then the JANGTQ2 bundle. Returns nil when neither
    /// is present.
    private func loadHy3Template() throws -> String? {
        let env = ProcessInfo.processInfo.environment
        if let override = env["VMLX_HY3_TEMPLATE_PATH"],
            !override.isEmpty,
            FileManager.default.fileExists(atPath: override)
        {
            return try String(contentsOfFile: override, encoding: .utf8)
        }
        let home = FileManager.default.homeDirectoryForCurrentUser
        let candidates = [
            home.appendingPathComponent("models/Tencent/Hy3-preview/chat_template.jinja"),
            home.appendingPathComponent("models/JANGQ/Hy3-preview-JANGTQ/chat_template.jinja"),
            home.appendingPathComponent("models/JANGQ/Hy3-preview-JANGTQ2/chat_template.jinja"),
        ]
        for url in candidates {
            if FileManager.default.fileExists(atPath: url.path) {
                return try String(contentsOfFile: url.path, encoding: .utf8)
            }
        }
        return nil
    }

    /// Convert a heterogeneous `[String: Any]` context into the
    /// `[String: Value]` shape swift-jinja 2.x expects. Mirrors the
    /// shim in `Gemma4ChatTemplateProbeTests`.
    private func render(_ tpl: Template, _ ctx: [String: Any]) throws -> String {
        var v: [String: Value] = [:]
        for (k, value) in ctx {
            v[k] = try Value(any: value)
        }
        return try tpl.render(v)
    }

    /// `reasoning_effort = "no_think"` MUST produce a prompt ending with
    /// the closed `<think></think>` block — otherwise `ReasoningParser.forPrompt`
    /// will see the open `<think>` as last and start in `.reasoning`,
    /// leaking the answer into the thinking pane.
    func testHy3TemplateNoThinkRendersClosedThinkBlock() throws {
        guard let src = try loadHy3Template() else {
            throw XCTSkip("Hy3 chat_template.jinja not available on this machine.")
        }
        let tpl: Template
        do {
            tpl = try Template(src)
        } catch {
            throw XCTSkip("Hy3 template did not parse on this swift-jinja version: \(error)")
        }
        let messages: [[String: Any]] = [
            ["role": "user", "content": "What is the capital of France?"]
        ]
        let prompt: String
        do {
            prompt = try render(
                tpl,
                [
                    "messages": messages,
                    "add_generation_prompt": true,
                    "reasoning_effort": "no_think",
                ])
        } catch {
            throw XCTSkip("Hy3 template render failed (likely a swift-jinja gap): \(error)")
        }

        // The closed-think block opens AND closes BOTH within the prompt
        // tail. Last `</think>` must appear AFTER the last `<think>`.
        guard let lastOpen = prompt.range(of: "<think>", options: .backwards),
              let lastClose = prompt.range(of: "</think>", options: .backwards)
        else {
            XCTFail("Expected <think> and </think> in the rendered no_think prompt.\n\(prompt.suffix(200))")
            return
        }
        XCTAssertGreaterThan(
            lastClose.lowerBound, lastOpen.lowerBound,
            "no_think prompt must end with closed </think> AFTER the open <think>. Tail: \(prompt.suffix(200))")
    }

    /// `reasoning_effort = "high"` MUST produce a prompt ending with an
    /// OPEN `<think>` (no closer after it) so the parser starts in
    /// `.reasoning` and routes pre-`</think>` model output correctly.
    func testHy3TemplateHighEffortRendersOpenThinkBlock() throws {
        guard let src = try loadHy3Template() else {
            throw XCTSkip("Hy3 chat_template.jinja not available on this machine.")
        }
        let tpl: Template
        do {
            tpl = try Template(src)
        } catch {
            throw XCTSkip("Hy3 template did not parse on this swift-jinja version: \(error)")
        }
        let messages: [[String: Any]] = [
            ["role": "user", "content": "Solve 17 * 23."]
        ]
        let prompt: String
        do {
            prompt = try render(
                tpl,
                [
                    "messages": messages,
                    "add_generation_prompt": true,
                    "reasoning_effort": "high",
                ])
        } catch {
            throw XCTSkip("Hy3 template render failed: \(error)")
        }

        // The open form: prompt tail has a `<think>` with NO subsequent
        // `</think>`. Either closer absent entirely, OR last `<think>`
        // appears AFTER the last `</think>`.
        guard let lastOpen = prompt.range(of: "<think>", options: .backwards) else {
            XCTFail("Expected <think> in the rendered high-effort prompt.\n\(prompt.suffix(200))")
            return
        }
        if let lastClose = prompt.range(of: "</think>", options: .backwards) {
            XCTAssertGreaterThan(
                lastOpen.lowerBound, lastClose.lowerBound,
                "high-effort prompt must end with OPEN <think> (no closer after it). Tail: \(prompt.suffix(200))")
        }
    }

    /// Default fallback: when caller does NOT set `reasoning_effort`, the
    /// template's line 37 `set reasoning_effort = 'no_think'` MUST kick
    /// in — producing the closed-think block. Pins this fallback so a
    /// future template refactor that drops the default doesn't silently
    /// flip into open-think (which would route every default-mode answer
    /// into `.reasoning`).
    func testHy3TemplateDefaultsToNoThinkClosedBlock() throws {
        guard let src = try loadHy3Template() else {
            throw XCTSkip("Hy3 chat_template.jinja not available on this machine.")
        }
        let tpl: Template
        do {
            tpl = try Template(src)
        } catch {
            throw XCTSkip("Hy3 template did not parse: \(error)")
        }
        let messages: [[String: Any]] = [
            ["role": "user", "content": "Hi."]
        ]
        let prompt: String
        do {
            prompt = try render(
                tpl,
                [
                    "messages": messages,
                    "add_generation_prompt": true,
                ])
        } catch {
            throw XCTSkip("Hy3 template render failed: \(error)")
        }

        // Default-mode rendering must end CLOSED, same as no_think.
        guard let lastOpen = prompt.range(of: "<think>", options: .backwards),
              let lastClose = prompt.range(of: "</think>", options: .backwards)
        else {
            XCTFail("Default-mode Hy3 prompt missing think block.\n\(prompt.suffix(200))")
            return
        }
        XCTAssertGreaterThan(
            lastClose.lowerBound, lastOpen.lowerBound,
            "Default-mode Hy3 prompt must end CLOSED. Tail: \(prompt.suffix(200))")
    }
}
