// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import Testing
import VMLXJinja
import VMLXTokenizers

@testable import MLXLMCommon

/// MiniMax-M3.1 thinking modes: the request -> template mapping (`MiniMaxM3ThinkingTemplateContext`), the M3.1
/// template with MiniMax-M3's own `thinking_mode` branches rendered by swift-jinja exactly as HF transformers renders
/// it (`Fixtures/MiniMaxM3/thinking-mode/renders.json`, `generate_thinking_mode_fixtures.py`), and the stream parser's
/// start state from the rendered prompt tail.
@Suite("MiniMax-M3.1 thinking_mode: adapter, template render, parser start state")
struct MiniMaxM3ThinkingModeTests {
    static let directory = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        .appendingPathComponent("Fixtures/MiniMaxM3/thinking-mode")

    static func fixture() throws -> [String: Any] {
        try JSONSerialization.jsonObject(with: Data(contentsOf: directory.appendingPathComponent("renders.json")))
            as! [String: Any]
    }

    /// Foundation JSON (no numbers or booleans in the fixture's messages / tools) to the Sendable shape a host passes.
    static func sendable(_ x: Any) -> any Sendable {
        if let s = x as? String { return s }
        if let a = x as? [Any] { return a.map { sendable($0) } }
        if let d = x as? [String: Any] { return d.mapValues { sendable($0) } as [String: any Sendable] }
        fatalError("unexpected fixture value \(x)")
    }

    /// The request context exactly as a host passes it (booleans stay Bool).
    static func request(_ raw: [String: Any]) -> [String: any Sendable] {
        var out: [String: any Sendable] = [:]
        for (k, v) in raw {
            if k == "enable_thinking" { out[k] = (v as! NSNumber).boolValue } else { out[k] = v as! String }
        }
        return out
    }

    /// swift-transformers' `applyChatTemplate` render: the repaired template with lstrip / trim blocks, messages and
    /// tools through `Value(any:)`, the additional context, and the tokenizer's special tokens.
    static func render(
        template: String, messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        addGenerationPrompt: Bool, context: [String: any Sendable]?, special: [String: String]
    ) throws -> String {
        let compiled = try Template(
            ChatTemplateRepair.repaired(template), with: .init(lstripBlocks: true, trimBlocks: true))
        var values: [String: Value] = [
            "messages": .array(try messages.map { try Value(any: $0) }),
            "add_generation_prompt": .boolean(addGenerationPrompt),
        ]
        if let tools { values["tools"] = .array(try tools.map { try Value(any: $0) }) }
        var extra = context ?? [:]
        if extra["thinking"] == nil, let enable = extra["enable_thinking"] { extra["thinking"] = enable }
        for (k, v) in extra { values[k] = try Value(any: v) }
        for (k, v) in special { values[k] = .string(v) }
        return try compiled.render(values)
    }

    static func firstDifference(_ a: String, _ b: String) -> String {
        let x = Array(a), y = Array(b)
        var i = 0
        while i < min(x.count, y.count), x[i] == y[i] { i += 1 }
        let lo = max(0, i - 60)
        return "at \(i) of \(x.count)/\(y.count): swift ...\(String(x[lo ..< min(x.count, i + 60)]).debugDescription) "
            + "hf ...\(String(y[lo ..< min(y.count, i + 60)]).debugDescription)"
    }

    @Test("Request -> template context: explicit thinking_mode wins, then the toggle, else adaptive (unset)")
    func adapterMapping() {
        let apply = { (ctx: [String: any Sendable]?) in
            MiniMaxM3ThinkingTemplateContext.apply(additionalContext: ctx, modelType: "minimax_m3_vl")
        }
        func mode(_ ctx: [String: any Sendable]?) -> String? { apply(ctx)?["thinking_mode"] as? String }
        #expect(apply(nil) == nil)
        #expect(mode(["enable_thinking": true]) == "enabled")
        #expect(mode(["enable_thinking": false]) == "disabled")
        #expect(mode(["thinking_mode": "enabled", "enable_thinking": false]) == "enabled")
        #expect(mode(["thinking_mode": " Disabled ", "enable_thinking": true]) == "disabled")
        #expect(mode(["thinking_mode": "adaptive", "enable_thinking": true]) == nil)
        #expect(mode(["thinking_mode": "instruct", "enable_thinking": true]) == "enabled")
        #expect(mode(["thinking_mode": "auto"]) == nil)
        // enable_thinking (and swift-transformers' `thinking` alias) never reach the M3 template.
        #expect(apply(["enable_thinking": true])?["enable_thinking"] == nil)
        #expect(apply(["enable_thinking": false, "thinking": false])?["thinking"] == nil)
        // reasoning_effort and unrelated keys pass through.
        let kept = apply(["reasoning_effort": "high", "enable_thinking": true, "custom": "x"])
        #expect(kept?["reasoning_effort"] as? String == "high" && kept?["custom"] as? String == "x")
        #expect(apply(["thinking_mode": "adaptive"]) == nil)
        // Other families are untouched; both M3 model types are covered.
        let other = MiniMaxM3ThinkingTemplateContext.apply(
            additionalContext: ["enable_thinking": true], modelType: "minimax_m2")
        #expect(other?["enable_thinking"] as? Bool == true && other?["thinking_mode"] == nil)
        #expect(MiniMaxM3ThinkingTemplateContext.applies(to: "minimax_m3"))
        #expect(MiniMaxM3ThinkingTemplateContext.applies(to: "MiniMax-M3-VL"))
        #expect(!MiniMaxM3ThinkingTemplateContext.applies(to: "minimax"))
        // The cache scope carries the mode (it changes the system line and the generation suffix).
        #expect(cacheScopeSalt(from: apply(["enable_thinking": true]))?.contains("thinking_mode=enabled") == true)
        #expect(cacheScopeSalt(from: apply(["enable_thinking": false]))?.contains("thinking_mode=disabled") == true)
        #expect(cacheScopeSalt(from: apply(["thinking_mode": "adaptive"])) == nil)
    }

    @Test("swift-jinja renders the thinking_mode template byte-identically to HF for every mode (render globals in macros)")
    func templateRendersLikeHF() throws {
        let fixture = try Self.fixture()
        let template = try String(contentsOf: Self.directory.appendingPathComponent("chat_template.jinja"), encoding: .utf8)
        let shipped = try String(
            contentsOf: Self.directory.appendingPathComponent("chat_template.shipped.jinja"), encoding: .utf8)
        // The template the bundles ship since 2026-10-28 (sha256 e8fa0fa5a433202a...).
        #expect((fixture["template_sha256"] as? String)?.hasPrefix("e8fa0fa5a433202a") == true)
        let special = (fixture["special_tokens"] as! [String: Any]).compactMapValues { $0 as? String }
        let tools = (fixture["tools"] as! [Any]).map { Self.sendable($0) as! [String: any Sendable] }
        let conversations = fixture["conversations"] as! [String: Any]
        let cases = fixture["cases"] as! [[String: Any]]
        try #require(cases.count == 120)
        var checked = 0, adaptiveEqualsShipped = 0
        for c in cases {
            let conv = c["conv"] as! String
            let messages = (conversations[conv] as! [Any]).map { Self.sendable($0) as! [String: any Sendable] }
            let gen = (c["add_generation_prompt"] as! NSNumber).boolValue
            let request = Self.request(c["request"] as! [String: Any])
            let context = MiniMaxM3ThinkingTemplateContext.apply(additionalContext: request, modelType: "minimax_m3_vl")
            // The adapter produces exactly the context the fixture rendered with.
            let expectedContext = c["template_context"] as! [String: Any]
            #expect((context?["thinking_mode"] as? String) == (expectedContext["thinking_mode"] as? String), "\(c["request"]!)")
            let rendered = try Self.render(
                template: template, messages: messages, tools: conv == "tool_loop" ? tools : nil,
                addGenerationPrompt: gen, context: context, special: special)
            let expected = c["expected"] as! String
            #expect(!expected.isEmpty)
            #expect(rendered == expected, "\(conv) gen \(gen) request \(c["request"]!): \(Self.firstDifference(rendered, expected))")
            let mode = context?["thinking_mode"] as? String
            if gen {
                switch mode {
                case "enabled"?: #expect(rendered.hasSuffix("]~b]ai\n<mm:think>"))
                case "disabled"?: #expect(rendered.hasSuffix("]~b]ai\n</mm:think>"))
                default: #expect(rendered.hasSuffix("]~b]ai\n"))
                }
            }
            if mode == nil {
                // Adaptive renders the shipped template byte for byte.
                let shippedRender = try Self.render(
                    template: shipped, messages: messages, tools: conv == "tool_loop" ? tools : nil,
                    addGenerationPrompt: gen, context: context, special: special)
                #expect(rendered == shippedRender, "\(conv) adaptive != shipped")
                #expect((c["equals_shipped"] as! NSNumber).boolValue)
                adaptiveEqualsShipped += 1
            }
            let line = mode.map { "Current thinking mode: \($0)." } ?? "Current thinking mode: adaptive."
            #expect(rendered.contains(line))
            checked += 1
        }
        #expect(checked == 120 && adaptiveEqualsShipped == 36)
    }

    @Test("Control: the template ignores enable_thinking (passed through unmapped it renders adaptive)")
    func controlUnmappedToggleRendersAdaptive() throws {
        let template = try String(contentsOf: Self.directory.appendingPathComponent("chat_template.jinja"), encoding: .utf8)
        let messages: [[String: any Sendable]] = [["role": "user", "content": "Hi"]]
        let special = ["bos_token": "]~b]", "eos_token": "[e~["]
        let unmapped = try Self.render(
            template: template, messages: messages, tools: nil, addGenerationPrompt: true,
            context: ["enable_thinking": true], special: special)
        let mapped = try Self.render(
            template: template, messages: messages, tools: nil, addGenerationPrompt: true,
            context: MiniMaxM3ThinkingTemplateContext.apply(
                additionalContext: ["enable_thinking": true], modelType: "minimax_m3_vl"), special: special)
        #expect(unmapped.hasSuffix("]~b]ai\n") && unmapped.contains("Current thinking mode: adaptive."))
        #expect(mapped.hasSuffix("<mm:think>") && unmapped != mapped)
    }

    @Test("Parser start state from the rendered prompt tail: inside reasoning after <mm:think>, content otherwise")
    func parserStartState() throws {
        let template = try String(contentsOf: Self.directory.appendingPathComponent("chat_template.jinja"), encoding: .utf8)
        let special = ["bos_token": "]~b]", "eos_token": "[e~["]
        // History with a closed reasoning block inside the tail window, so the tail is not marker-free.
        let messages: [[String: any Sendable]] = [
            ["role": "user", "content": "x"], ["role": "assistant", "content": "<mm:think>t</mm:think>answer"],
            ["role": "user", "content": "y"],
        ]
        func tail(_ toggle: Bool?) throws -> String {
            let ctx = MiniMaxM3ThinkingTemplateContext.apply(
                additionalContext: toggle.map { ["enable_thinking": $0] }, modelType: "minimax_m3")
            let r = try Self.render(
                template: template, messages: messages, tools: nil, addGenerationPrompt: true, context: ctx,
                special: special)
            return String(r.suffix(256))
        }
        func run(_ toggle: Bool?, _ stream: [String]) throws -> (String, String) {
            var parser = try #require(ReasoningParser.forPrompt(stampName: "minimax_m3", promptTail: try tail(toggle)))
            var reasoning = "", content = ""
            for piece in stream {
                for segment in parser.feed(piece) {
                    switch segment {
                    case .reasoning(let s): reasoning += s
                    case .content(let s): content += s
                    }
                }
            }
            for segment in parser.flush() {
                switch segment {
                case .reasoning(let s): reasoning += s
                case .content(let s): content += s
                }
            }
            return (reasoning, content)
        }
        // On: the prompt ends inside the rail; the model writes reasoning, closes it, answers.
        let on = try run(true, ["Let me ", "add.", "</mm:", "think>", "4"])
        #expect(on.0 == "Let me add." && on.1 == "4")
        // Off: the prompt closed the rail; everything is content.
        let off = try run(false, ["4", " is the answer"])
        #expect(off.0 == "" && off.1 == "4 is the answer")
        // Adaptive: either first token.
        let think = try run(nil, ["<mm:think>", "plan", "</mm:think>", "4"])
        #expect(think.0 == "plan" && think.1 == "4")
        let skip = try run(nil, ["</mm:think>", "4"])
        #expect(skip.0 == "" && skip.1 == "4")
        // Control: without the mapping, "On" leaves the prompt adaptive and the reasoning lands in content.
        var unmapped = try #require(ReasoningParser.forPrompt(
            stampName: "minimax_m3",
            promptTail: try Self.render(
                template: template, messages: messages, tools: nil, addGenerationPrompt: true,
                context: ["enable_thinking": true], special: special)))
        let leaked = (unmapped.feed("Let me add.") + unmapped.flush()).map { segment -> String in
            if case .content(let s) = segment { return s }
            return ""
        }.joined()
        #expect(leaked == "Let me add.", "control: \(leaked.debugDescription)")
    }
}
