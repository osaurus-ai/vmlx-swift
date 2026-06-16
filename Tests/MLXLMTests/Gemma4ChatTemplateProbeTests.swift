// Copyright 2025 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Iter 31 probe: pin down exactly which construct in the Gemma-4 chat template
// makes swift-jinja throw `syntax("Unexpected token: multiplicativeBinaryOperator")`.
//
// Motivation: Gemma-4 / Gemma-4n use a recursive Jinja template with tool-call
// formatting macros. Loading them through `#huggingFaceTokenizerLoader()` and
// then rendering a trivial chat fails with a parse error, which blocks
// `BENCH_BATCH_CHAT=1` real-model verification on any Gemma-4 family
// (sliding-window, Gemma-4-E2B, Gemma-4-E4B, Gemma-4-26B-A4B).
//
// These tests do NOT require any model weights. They exercise the template
// file directly via `VMLXJinja.Template`.

import Foundation
import XCTest

import MLXLMCommon
import VMLXJinja

// Convert a heterogeneous `[String: Any]` context into the
// `[String: Value]` shape swift-jinja 2.x expects. Used by the
// probe tests below which were authored against Jinja 1.x's
// `render(_: [String: Any])`. Production callers go through
// `Tokenizer.applyChatTemplate(messages:, …)` in swift-
// transformers which converts for us; only these standalone
// probe tests need the shim.
extension Template {
    fileprivate func renderAny(_ ctx: [String: Any]) throws -> String {
        var v: [String: Value] = [:]
        for (k, value) in ctx {
            v[k] = try Value(any: value)
        }
        return try self.render(v)
    }
}

final class Gemma4ChatTemplateProbeTests: XCTestCase {

    /// Full Gemma-4 chat template shipped with `mlx-community/gemma-4-e2b-it-4bit`.
    /// Loading from disk to keep the test file small. Skipped when the model
    /// isn't on this machine (so CI/other dev boxes don't fail).
    private func loadGemma4Template() throws -> String? {
        // Try the HF cache snapshots dir first (content-addressed),
        // then the well-known mlxstudio location. Env override
        // `VMLX_GEMMA4_TEMPLATE_PATH` lets a dev point this elsewhere
        // without editing sources.
        let env = ProcessInfo.processInfo.environment
        if let override = env["VMLX_GEMMA4_TEMPLATE_PATH"], !override.isEmpty,
           FileManager.default.fileExists(atPath: override) {
            return try String(contentsOfFile: override, encoding: .utf8)
        }
        let home = FileManager.default.homeDirectoryForCurrentUser
        // 1. HF hub cache — walk snapshots.
        let hfRepo = home.appendingPathComponent(
            ".cache/huggingface/hub/models--mlx-community--gemma-4-e2b-it-4bit/snapshots")
        if let snapshots = try? FileManager.default.contentsOfDirectory(
            at: hfRepo, includingPropertiesForKeys: nil)
        {
            for snap in snapshots {
                let tpl = snap.appendingPathComponent("chat_template.jinja")
                if FileManager.default.fileExists(atPath: tpl.path) {
                    return try String(contentsOfFile: tpl.path, encoding: .utf8)
                }
            }
        }
        // 2. mlxstudio well-known layout.
        let ms = home.appendingPathComponent(
            ".mlxstudio/models/MLXModels/mlx-community/gemma-4-e2b-it-4bit/chat_template.jinja")
        if FileManager.default.fileExists(atPath: ms.path) {
            return try String(contentsOfFile: ms.path, encoding: .utf8)
        }
        return nil
    }

    /// Parse the full Gemma-4 template and record what happens. Test always
    /// passes — it is a *probe* that documents the current parser state.
    /// When the library starts parsing this template, this probe switches
    /// from "parse fails" to "parse succeeds" and the engineer can remove
    /// the Gemma-4 template workaround.
    func testGemma4TemplateParseStatus() throws {
        guard let src = try loadGemma4Template() else {
            throw XCTSkip("Gemma-4 template not available on this machine.")
        }

        let thrown: Error?
        do {
            _ = try Template(src)
            thrown = nil
        } catch {
            thrown = error
        }

        if let e = thrown {
            print("[Gemma4TemplateProbe] parse FAILED: \(e)")
        } else {
            print("[Gemma4TemplateProbe] parse OK (length=\(src.count))")
        }
    }

    /// Minimal reproducer for the `multiplicativeBinaryOperator` failure.
    /// The parser chokes on this line from the real template:
    ///
    ///     {{- '<|"|>' + argument + '<|"|>' -}}
    ///
    /// Swift-jinja treats `|"` inside a plain-text literal the same way it
    /// treats the start of a filter argument — the `<|"|>` string literal
    /// (a Gemma-specific marker) is not the root cause. The real offender
    /// is **something inside one of the macro bodies** that the lexer
    /// mis-tokenises once the `|`-rich text has put it in an unexpected state.
    ///
    /// This reproducer peels off each macro until the parser succeeds; any
    /// surviving macro that still fails is the minimal culprit.
    func testGemma4TemplateMinimalReproducer() throws {
        // Distilled: just the failing macro with its argument filters.
        // Copied verbatim from `chat_template.jinja` lines 118-147 of the
        // Gemma-4-E2B template.
        let src = #"""
{%- macro format_argument(argument, escape_keys=True) -%}
    {%- if argument is string -%}
        {{- '<|"|>' + argument + '<|"|>' -}}
    {%- elif argument is boolean -%}
        {{- 'true' if argument else 'false' -}}
    {%- elif argument is mapping -%}
        {{- '{' -}}
        {%- set ns = namespace(found_first=false) -%}
        {%- for key, value in argument | dictsort -%}
            {%- if ns.found_first %},{% endif -%}
            {%- set ns.found_first = true -%}
            {%- if escape_keys -%}
                {{- '<|"|>' + key + '<|"|>' -}}
            {%- else -%}
                {{- key -}}
            {%- endif -%}
            :{{- format_argument(value, escape_keys=escape_keys) -}}
        {%- endfor -%}
        {{- '}' -}}
    {%- elif argument is sequence -%}
        {{- '[' -}}
        {%- for item in argument -%}
            {{- format_argument(item, escape_keys=escape_keys) -}}
            {%- if not loop.last %},{% endif -%}
        {%- endfor -%}
        {{- ']' -}}
    {%- else -%}
        {{- argument -}}
    {%- endif -%}
{%- endmacro -%}

{{ format_argument("hello") }}
"""#
        do {
            let t = try Template(src)
            // If parse succeeds, try rendering to be sure runtime semantics work.
            let out = try t.renderAny([:])
            print("[MinimalReproducer] parse+render OK: \(out)")
        } catch {
            print("[MinimalReproducer] parse/render FAILED: \(error)")
        }
    }

    /// Even narrower: parse just the single-line recursive call inside
    /// `format_argument`. The recursion expression itself is fine; the
    /// wrapping by `'<|"|>' + key + '<|"|>'` is the trigger.
    func testGemma4TemplateSimpleInfix() throws {
        // The ACTUAL line that crashes the parser.
        let src = #"""
{{- '<|"|>' + argument + '<|"|>' -}}
"""#
        do {
            _ = try Template(src)
            print("[SimpleInfix] parse OK")
        } catch {
            print("[SimpleInfix] FAILED: \(error)")
        }
    }

    /// Iter 31 finding: swift-jinja 1.3.0's parser TRAPS on certain
    /// truncated prefixes of this template (unbalanced macros, etc.),
    /// which makes line-level bisection impossible inside a single test
    /// process — the first "fails" sample kills the test runner with
    /// `Fatal error: String index is out of bounds`.
    ///
    /// The full template, however, throws a structured JinjaError:
    /// `syntax("Unexpected token: multiplicativeBinaryOperator")`.
    /// That originates from `Parser.parsePrimaryExpression` hitting a
    /// `*` / `/` / `%` token where an expression was expected — which
    /// means swift-jinja's lexer recognises some construct (likely a
    /// filter-chain or an if-expression) but its parser falls through to
    /// the default case on the next token.
    ///
    /// Former `testGemma4TemplateThrowsStructuredJinjaError` has flipped
    /// polarity — since vmlx-swift-lm now pulls swift-transformers 1.3.0
    /// (which transitively uses huggingface/swift-jinja 2.3.5+), the
    /// Gemma-4 template PARSES natively. This test now asserts the
    /// fixed state so a future upstream regression surfaces here.
    func testGemma4TemplateNowParsesNatively() throws {
        guard let src = try loadGemma4Template() else {
            throw XCTSkip("Gemma-4 template not available on this machine.")
        }
        do {
            _ = try Template(src)
        } catch {
            XCTFail("Gemma-4 template regressed in swift-jinja: \(error)")
        }
    }

    func testGemma4TemplateRendersNullableToolSchemaAfterNormalization() throws {
        guard let src = try loadGemma4Template() else {
            throw XCTSkip("Gemma-4 template not available on this machine.")
        }
        let template = try Template(src)
        let rawTools: [ToolSpec] = [[
            "type": "function",
            "function": [
                "name": "lookup",
                "description": "Looks up an optional label.",
                "parameters": [
                    "type": "object",
                    "properties": [
                        "label": [
                            "type": ["string", "null"],
                            "description": "Optional label.",
                        ] as [String: any Sendable],
                    ] as [String: any Sendable],
                    "required": ["label"],
                ] as [String: any Sendable],
            ] as [String: any Sendable],
        ]]

        let context: ([ToolSpec]) -> [String: Any] = { tools in
            [
                "bos_token": "<bos>",
                "messages": [
                    ["role": "user", "content": "look this up"] as [String: String],
                ],
                "tools": tools,
                "add_generation_prompt": true,
                "enable_thinking": false,
            ]
        }

        XCTAssertThrowsError(try template.renderAny(context(rawTools)))

        let normalizedTools = try XCTUnwrap(normalizedToolsForChatTemplate(rawTools))
        let rendered = try template.renderAny(context(normalizedTools))
        XCTAssertTrue(rendered.contains("nullable:true"))
        XCTAssertTrue(rendered.contains("type:<|\"|>STRING<|\"|>"))
    }

    /// Iter 52: the minimal template must emit image / video placeholder
    /// tokens when a caller passes multi-part content with `type: 'image'`
    /// / `type: 'video'`. Iter 50's version silently dropped everything
    /// except text — same bug class as iter 45's UserInput image drop.
    func testGemma4MinimalTemplateHandlesVLContent() throws {
        var searchPath = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        var found: URL? = nil
        for _ in 0..<6 {
            let candidate = searchPath
                .appendingPathComponent("Libraries/MLXLMCommon/ChatTemplates/Gemma4Minimal.jinja")
            if FileManager.default.fileExists(atPath: candidate.path) {
                found = candidate
                break
            }
            searchPath = searchPath.deletingLastPathComponent()
        }
        guard let templateURL = found else {
            throw XCTSkip("Gemma4Minimal.jinja not found relative to test source.")
        }
        let src = try String(contentsOf: templateURL, encoding: .utf8)
        let template = try Template(src)

        // Multi-part content: text + image + text + video.
        let rendered = try template.renderAny([
            "bos_token": "<bos>",
            "messages": [
                [
                    "role": "user",
                    "content": [
                        ["type": "text", "text": "look at this"] as [String: String],
                        ["type": "image"] as [String: String],
                        ["type": "text", "text": "and this video"] as [String: String],
                        ["type": "video"] as [String: String],
                    ] as [Any],
                ] as [String: Any],
            ] as [Any],
            "add_generation_prompt": true,
        ])
        XCTAssertTrue(rendered.contains("<|image|>"),
            "Image placeholder must survive through the minimal template. Got: \(rendered)")
        XCTAssertTrue(rendered.contains("<|video|>"),
            "Video placeholder must survive. Got: \(rendered)")
        XCTAssertTrue(rendered.contains("look at this"))
        XCTAssertTrue(rendered.contains("and this video"))
    }

    /// Iter 60: the Gemma4WithTools.jinja template adds tool-call + tool-response +
    /// thinking rendering on top of Gemma4Minimal. Pin that all four turn types
    /// render into Gemma-4's native `<|tool>` / `<|tool_call>` / `<|tool_response>`
    /// markers.
    func testGemma4WithToolsTemplateRenders() throws {
        var searchPath = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        var found: URL? = nil
        for _ in 0..<6 {
            let candidate = searchPath
                .appendingPathComponent("Libraries/MLXLMCommon/ChatTemplates/Gemma4WithTools.jinja")
            if FileManager.default.fileExists(atPath: candidate.path) {
                found = candidate; break
            }
            searchPath = searchPath.deletingLastPathComponent()
        }
        guard let templateURL = found else {
            throw XCTSkip("Gemma4WithTools.jinja not found.")
        }
        let src = try String(contentsOf: templateURL, encoding: .utf8)
        let template = try Template(src)

        let tools: [[String: Any]] = [
            [
                "type": "function",
                "function": [
                    "name": "get_weather",
                    "description": "fetch current weather",
                ] as [String: Any]
            ] as [String: Any]
        ]
        let messages: [[String: Any]] = [
            ["role": "system", "content": "You are a helpful assistant." ] as [String: Any],
            ["role": "user",   "content": "What's the weather in Paris?" ] as [String: Any],
            [
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    [
                        "function": [
                            "name": "get_weather",
                            "arguments": ["location": "Paris"] as [String: Any]
                        ] as [String: Any]
                    ] as [String: Any]
                ] as [Any]
            ] as [String: Any],
            [
                "role": "tool",
                "name": "get_weather",
                "content": "22°C, sunny"
            ] as [String: Any],
        ]

        let out = try template.renderAny([
            "bos_token": "<bos>",
            "messages": messages,
            "tools": tools,
            "add_generation_prompt": true,
        ])

        XCTAssertTrue(out.contains("<|tool>declaration:get_weather"),
            "Tool declaration must emit Gemma-4's <|tool> wrapper. Got: \(out)")
        XCTAssertTrue(out.contains("<|tool_call>call:get_weather{location:<|\"|>Paris<|\"|>}<tool_call|>"),
            "Assistant tool_call must render inline with Gemma's invoke syntax. Got: \(out)")
        XCTAssertTrue(out.contains("<|tool_response>response:get_weather{22°C, sunny}<tool_response|>"),
            "Tool response must render with Gemma's <|tool_response> wrapper. Got: \(out)")
        XCTAssertTrue(out.hasSuffix("<|turn>model\n"),
            "Generation prompt must end with open model turn.")
    }

    // MARK: - KV-cache prefix-stability contract (osaurus ↔ vmlx wiring)

    /// Locate `Libraries/MLXLMCommon/ChatTemplates/<name>.jinja` by walking up
    /// from this test source, regardless of CWD.
    private func loadBundledTemplate(_ name: String) throws -> Template {
        var searchPath = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        for _ in 0..<6 {
            let candidate = searchPath
                .appendingPathComponent("Libraries/MLXLMCommon/ChatTemplates/\(name).jinja")
            if FileManager.default.fileExists(atPath: candidate.path) {
                return try Template(try String(contentsOf: candidate, encoding: .utf8))
            }
            searchPath = searchPath.deletingLastPathComponent()
        }
        throw XCTSkip("\(name).jinja not found relative to test source.")
    }

    /// The system turn (`<|turn>system\n … <turn|>\n`) is the front of every
    /// Gemma-4 prompt and the segment the KV cache reuses across an agent
    /// loop's iterations. Returns everything up to and including the first
    /// `<turn|>\n`, or nil if no system turn was emitted.
    private func systemTurn(of rendered: String) -> String? {
        guard let closeRange = rendered.range(of: "<turn|>\n") else { return nil }
        return String(rendered[rendered.startIndex..<closeRange.upperBound])
    }

    /// Regression for the Gemma-4 "prefill resets to 0 after a tool call" bug
    /// (tpae's report; osaurus root cause removed in osaurus-ai/osaurus#1525).
    ///
    /// The agent loop renders the SAME conversation twice: once at the turn
    /// that produces a tool call, then again after appending the tool result.
    /// As long as the caller holds `tools` (and `tool_choice`) byte-stable —
    /// which #1525 now guarantees by deleting `finalizingPostToolChoice` and
    /// passing `resolvedToolChoice` unchanged every iteration — the rendered
    /// system turn (system content + `<|tool>declaration:` block) MUST be
    /// byte-identical and remain a literal prefix of the continuation prompt.
    /// That is exactly what lets the runtime extend the cached KV prefix
    /// instead of re-prefilling from token ~0.
    func testGemma4WithToolsSystemPrefixStableAcrossToolResultTurn() throws {
        let template = try loadBundledTemplate("Gemma4WithTools")
        let tools: [[String: Any]] = [[
            "type": "function",
            "function": [
                "name": "capabilities_discover",
                "description": "discover host capabilities",
            ] as [String: Any],
        ] as [String: Any]]

        let toolCallTurn: [[String: Any]] = [
            ["role": "system", "content": "You are a helpful assistant."] as [String: Any],
            ["role": "user", "content": "what can you do?"] as [String: Any],
        ]
        // Continuation: the assistant tool_call plus the tool result appended.
        let toolResultTurn: [[String: Any]] = toolCallTurn + [
            [
                "role": "assistant",
                "content": "",
                "tool_calls": [[
                    "function": [
                        "name": "capabilities_discover",
                        "arguments": [:] as [String: Any],
                    ] as [String: Any],
                ] as [String: Any]] as [Any],
            ] as [String: Any],
            ["role": "tool", "name": "capabilities_discover", "content": "{\"ok\":true}"] as [String: Any],
        ]

        let render: ([[String: Any]]) throws -> String = { messages in
            try template.renderAny([
                "bos_token": "<bos>",
                "messages": messages,
                "tools": tools,
                "add_generation_prompt": true,
            ])
        }

        let p1 = try render(toolCallTurn)
        let p2 = try render(toolResultTurn)

        let sys1 = try XCTUnwrap(systemTurn(of: p1), "tool-call turn emitted no system turn")
        let sys2 = try XCTUnwrap(systemTurn(of: p2), "tool-result turn emitted no system turn")

        XCTAssertTrue(sys1.contains("<|tool>declaration:capabilities_discover"),
            "system turn must carry the tool declaration. Got: \(sys1)")
        XCTAssertEqual(sys1, sys2,
            "system turn (system + tool declarations) must be byte-stable across the "
            + "tool-call → tool-result iterations so the KV prefix is reused.")
        XCTAssertTrue(p2.hasPrefix(sys1),
            "the continuation prompt must begin with the identical system turn — "
            + "otherwise the cache re-prefills from ~0.")
    }

    /// Locks in WHY the caller must never drop the tools list on a
    /// tool-result turn (the osaurus-side regression that #1525 removed):
    /// when `tools` disappears, the `<|tool>declaration:` block leaves the
    /// front of the system turn, so the continuation prompt no longer shares
    /// the cached prefix and the runtime re-prefills the whole context. This
    /// test asserts the busting mechanism directly, documenting the wiring
    /// contract for any future caller.
    func testGemma4WithToolsDroppingToolsBustsSystemPrefix() throws {
        let template = try loadBundledTemplate("Gemma4WithTools")
        let tools: [[String: Any]] = [[
            "type": "function",
            "function": [
                "name": "capabilities_discover",
                "description": "discover host capabilities",
            ] as [String: Any],
        ] as [String: Any]]
        let messages: [[String: Any]] = [
            ["role": "system", "content": "You are a helpful assistant."] as [String: Any],
            ["role": "user", "content": "what can you do?"] as [String: Any],
            [
                "role": "assistant",
                "content": "",
                "tool_calls": [[
                    "function": [
                        "name": "capabilities_discover",
                        "arguments": [:] as [String: Any],
                    ] as [String: Any],
                ] as [String: Any]] as [Any],
            ] as [String: Any],
            ["role": "tool", "name": "capabilities_discover", "content": "{\"ok\":true}"] as [String: Any],
        ]

        let withTools = try template.renderAny([
            "bos_token": "<bos>", "messages": messages, "tools": tools,
            "add_generation_prompt": true,
        ])
        let withoutTools = try template.renderAny([
            "bos_token": "<bos>", "messages": messages,
            "add_generation_prompt": true,
        ])

        let sysWith = try XCTUnwrap(systemTurn(of: withTools))
        XCTAssertTrue(sysWith.contains("<|tool>declaration:capabilities_discover"))
        // Dropping the tools strips the declaration block, so the with-tools
        // system turn is NO LONGER a prefix of the prompt — the cache busts.
        XCTAssertFalse(withoutTools.hasPrefix(sysWith),
            "dropping the tools list must (and here does) bust the cached system "
            + "prefix — this is the failure mode osaurus#1525 fixed by holding "
            + "tools byte-stable across iterations.")
    }

    /// Iter 52: the system-message branch must handle multi-part content
    /// too. iter 50's version assumed `content` was always a string —
    /// would crash on a system turn whose content comes through as a
    /// content-parts array.
    func testGemma4MinimalTemplateHandlesMultipartSystem() throws {
        var searchPath = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        var found: URL? = nil
        for _ in 0..<6 {
            let candidate = searchPath
                .appendingPathComponent("Libraries/MLXLMCommon/ChatTemplates/Gemma4Minimal.jinja")
            if FileManager.default.fileExists(atPath: candidate.path) {
                found = candidate
                break
            }
            searchPath = searchPath.deletingLastPathComponent()
        }
        guard let templateURL = found else {
            throw XCTSkip("Gemma4Minimal.jinja not found relative to test source.")
        }
        let src = try String(contentsOf: templateURL, encoding: .utf8)
        let template = try Template(src)
        let rendered = try template.renderAny([
            "bos_token": "<bos>",
            "messages": [
                [
                    "role": "system",
                    "content": [
                        ["type": "text", "text": "sys instructions"] as [String: String]
                    ] as [Any],
                ] as [String: Any],
                [
                    "role": "user",
                    "content": "hello",
                ] as [String: Any],
            ] as [Any],
            "add_generation_prompt": true,
        ])
        XCTAssertTrue(rendered.contains("<|turn>system\nsys instructions<turn|>"),
            "System turn multipart content must render. Got: \(rendered)")
    }

    /// Isolated probe: `{{- ',' if not loop.last -}}` is a Python-Jinja
    /// if-expression without else. Turns out swift-jinja 1.3.0 DOES
    /// support this form (test passes → parse OK). Kept as a regression
    /// guard in case a future swift-jinja pin loses that capability.
    func testJinjaIfExpressionWithoutElseIsSupported() throws {
        let src = "{{- ',' if not loop.last -}}"
        _ = try Template(src)
    }

    /// Isolated probe: `message.get('key')` — Python-style dict method
    /// invocation, used pervasively in the Gemma-4 template (lines 237,
    /// 261, 278, 283, 291, 292). Swift-jinja 1.3.0 supports generic
    /// `.identifier(args)` calls, so this path works — kept as a
    /// regression guard.
    func testJinjaDictGetMethodIsSupported() throws {
        let src = "{{ m.get('foo', 'fallback') }}"
        let t = try Template(src)
        let out = try t.renderAny(["m": ["foo": "bar"]])
        XCTAssertEqual(out, "bar")
    }

    /// Isolated probe: `is not string` — negated Jinja test. Gemma-4 uses
    /// this at line 288 (`elif tool_body is sequence and tool_body is not string`).
    func testJinjaIsNotTestIsSupported() throws {
        let src = "{% if x is not string %}Y{% else %}N{% endif %}"
        let t = try Template(src)
        let intOut = try t.renderAny(["x": 42])
        XCTAssertEqual(intOut, "Y", "`is not string` should report true for integers")
        let strOut = try t.renderAny(["x": "hello"])
        XCTAssertEqual(strOut, "N", "`is not string` should report false for strings")
    }

    /// Probe the SSM-style `namespace(...)` call that Gemma-4 uses to
    /// carry mutable state across loop iterations (line 125, 150, 175,
    /// 206, 246, 270). These are heavily chained with `.attr = value`
    /// assignments via `{%- set ns.x = ... -%}`.
    /// Iter 50 shipped a minimal Gemma-4-compatible template as a
    /// workaround for the upstream swift-jinja gap. Pin that the
    /// workaround template actually parses, renders a 2-turn chat
    /// correctly, and produces the expected turn delimiters.
    func testGemma4MinimalTemplateRenders() throws {
        let path = URL(fileURLWithPath:
            "Libraries/MLXLMCommon/ChatTemplates/Gemma4Minimal.jinja")
        // Find the file regardless of CWD — walk up from the test's
        // Resources dir until we hit a sibling `Libraries/` directory.
        var searchPath = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
        var found: URL? = nil
        for _ in 0..<6 {
            let candidate = searchPath
                .appendingPathComponent("Libraries/MLXLMCommon/ChatTemplates/Gemma4Minimal.jinja")
            if FileManager.default.fileExists(atPath: candidate.path) {
                found = candidate
                break
            }
            searchPath = searchPath.deletingLastPathComponent()
        }
        guard let templateURL = found else {
            throw XCTSkip("Gemma4Minimal.jinja not found relative to test source.")
        }
        _ = path  // silence unused-binding warning
        let src = try String(contentsOf: templateURL, encoding: .utf8)
        let template = try Template(src)
        let rendered = try template.renderAny([
            "bos_token": "<bos>",
            "messages": [
                ["role": "system", "content": "You are a helpful assistant."],
                ["role": "user",   "content": "Say hi."],
                ["role": "assistant", "content": "Hi!"],
                ["role": "user",   "content": "Say bye."],
            ] as [[String: String]],
            "add_generation_prompt": true,
        ])
        print("[Gemma4Minimal] rendered (\(rendered.count) chars):\n\(rendered)")
        // Core invariants — shouldn't hard-code every byte.
        XCTAssertTrue(rendered.hasPrefix("<bos>"),
            "Template must emit bos_token at the very start.")
        XCTAssertTrue(rendered.contains("<|turn>system\nYou are a helpful assistant.<turn|>\n"),
            "System-role turn framing must match Gemma-4's native delimiters.")
        XCTAssertTrue(rendered.contains("<|turn>user\nSay hi.<turn|>\n"))
        XCTAssertTrue(rendered.contains("<|turn>model\nHi!<turn|>\n"))
        XCTAssertTrue(rendered.contains("<|turn>user\nSay bye.<turn|>\n"))
        XCTAssertTrue(rendered.hasSuffix("<|turn>model\n"),
            "When add_generation_prompt is true, render ends with an open " +
            "model turn — this is where the model continues generating.")
    }

    func testDSV4MinimalTemplateHonorsThinkingToggle() throws {
        var searchPath = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
        var found: URL? = nil
        for _ in 0..<6 {
            let candidate = searchPath
                .appendingPathComponent("Libraries/MLXLMCommon/ChatTemplates/DSV4Minimal.jinja")
            if FileManager.default.fileExists(atPath: candidate.path) {
                found = candidate
                break
            }
            searchPath = searchPath.deletingLastPathComponent()
        }
        guard let templateURL = found else {
            throw XCTSkip("DSV4Minimal.jinja not found relative to test source.")
        }
        let src = try String(contentsOf: templateURL, encoding: .utf8)
        let template = try Template(src)
        let messages = [["role": "user", "content": "Remember sapphire-42."]]

        let chat = try template.renderAny([
            "messages": messages,
            "add_generation_prompt": true,
            "enable_thinking": false,
        ])
        XCTAssertTrue(
            chat.hasSuffix("<\u{FF5C}Assistant\u{FF5C}></think>"),
            "DSV4 chat mode must close the thinking block when enable_thinking=false.")

        let thinking = try template.renderAny([
            "messages": messages,
            "add_generation_prompt": true,
            "enable_thinking": true,
        ])
        XCTAssertTrue(
            thinking.hasSuffix("<\u{FF5C}Assistant\u{FF5C}><think>"),
            "DSV4 thinking mode must open the thinking block when enable_thinking=true.")
    }

    func testJinjaNamespaceAssignmentIsSupported() throws {
        let src = """
        {%- set ns = namespace(found_first=false) -%}
        {%- for i in [1,2,3] -%}
            {%- if ns.found_first %},{% endif -%}
            {{ i }}
            {%- set ns.found_first = true -%}
        {%- endfor -%}
        """
        let t = try Template(src)
        let out = try t.renderAny([:])
        XCTAssertEqual(out, "1,2,3",
            "namespace attribute assignment via `set ns.x = value` must work")
    }
}
