// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// DeepSeek-V4.1 emits DSML tool calls with leading-space tag names
// (`<｜DSML｜ calls>` / `<｜DSML｜ invoke …>` / `<｜DSML｜ parameter …>`, bundle
// `encoding/encoding.py`). Live V4.1 runs returned `tool_calls: null` with the
// raw DSML in `content` because only the 0731 `tool_calls` dialect parsed.
// These tests pin the V4.1 dialect and that the 0731 dialect is unchanged.

import Foundation
import Testing

@testable import MLXLMCommon

@Suite("DeepSeek-V4.1 DSML tool calls")
struct DeepseekV41DSMLToolCallTests {
    static let d = "\u{FF5C}DSML\u{FF5C}"

    static var v41Envelope: String {
        "\n\n<\(d) calls>\n<\(d) invoke name=\"get_weather\">\n"
            + "<\(d) parameter name=\"location\" string=\"true\">Paris</\(d) parameter>\n"
            + "<\(d) parameter name=\"days\" string=\"false\">3</\(d) parameter>\n"
            + "</\(d) invoke>\n</\(d) calls>"
    }

    static var v4Envelope: String {
        "<\(d)tool_calls>\n<\(d)invoke name=\"get_weather\">\n"
            + "<\(d)parameter name=\"location\" string=\"true\">Paris</\(d)parameter>\n"
            + "</\(d)invoke>\n</\(d)tool_calls>"
    }

    @Test("model_type and capability stamps select the V4.1 dialect only for V4.1")
    func formatSelection() {
        #expect(ToolCallFormat.infer(from: "deepseek_v41") == .dsmlV41)
        #expect(ToolCallFormat.infer(from: "deepseek_v41_text") == .dsmlV41)
        #expect(ToolCallFormat.infer(from: "deepseek_v4") == .dsml)
        #expect(ToolCallFormat.fromCapabilityName("dsml", modelType: "deepseek_v41") == .dsmlV41)
        #expect(ToolCallFormat.fromCapabilityName("dsml", modelType: "deepseek_v4") == .dsml)
        #expect(ToolCallFormat.fromCapabilityName("dsml_v41") == .dsmlV41)
        #expect(ToolCallFormat.dsmlV41.parsesToolCallsFromReasoningChannel == false)
    }

    @Test("V4.1 parser extracts calls from the leading-space envelope")
    func parsesV41Envelope() throws {
        let parser = ToolCallFormat.dsmlV41.createParser()
        let calls = parser.parseEOS(Self.v41Envelope, tools: nil)
        #expect(calls.count == 1)
        let call = try #require(calls.first)
        #expect(call.function.name == "get_weather")
        #expect(call.function.arguments["location"] == .string("Paris"))
        #expect(call.function.arguments["days"] == .int(3))
    }

    @Test("streaming V4.1 output yields a tool call and no leaked DSML")
    func streamingV41() throws {
        let processor = ToolCallProcessor(format: .dsmlV41)
        var visible = ""
        let text = "Let me check." + Self.v41Envelope
        var i = text.startIndex
        while i < text.endIndex {
            let j = text.index(i, offsetBy: 5, limitedBy: text.endIndex) ?? text.endIndex
            visible += processor.processChunk(String(text[i..<j])) ?? ""
            i = j
        }
        visible += processor.processEOS() ?? ""
        #expect(!visible.contains("DSML"), "DSML leaked: \(visible)")
        #expect(visible.trimmingCharacters(in: .whitespacesAndNewlines) == "Let me check.")
        #expect(processor.toolCalls.count == 1)
        #expect(processor.toolCalls.first?.function.name == "get_weather")
    }

    @Test("each dialect accepts only its own tag names (0731 unchanged)")
    func dialectsAreDisjoint() {
        let v4 = ToolCallFormat.dsml.createParser()
        let v41 = ToolCallFormat.dsmlV41.createParser()
        #expect(v4.startTag == "<\(Self.d)tool_calls>")
        #expect(v4.parseEOS(Self.v4Envelope, tools: nil).count == 1)
        #expect(v4.parseEOS(Self.v41Envelope, tools: nil).isEmpty)
        #expect(v41.startTag == "<\(Self.d) calls>")
        #expect(v41.parseEOS(Self.v4Envelope, tools: nil).isEmpty)
    }

    @Test("encoder history renders V4.1 tags that the V4.1 parser round-trips")
    func encoderParserRoundTrip() throws {
        let block = DeepseekV4ChatEncoder.renderToolCallInvoke(
            name: "get_weather", arguments: #"{"location": "Paris", "days": 3}"#,
            family: .deepseekV41)
        let envelope = "<\(Self.d) calls>\n\(block)\n</\(Self.d) calls>"
        let calls = ToolCallFormat.dsmlV41.createParser().parseEOS(envelope, tools: nil)
        #expect(calls.count == 1)
        #expect(calls.first?.function.arguments["days"] == .int(3))
        // Double-encoded JSON arguments are tolerated on V4.1 (encoding.py).
        let doubled = DeepseekV4ChatEncoder.renderToolCallInvoke(
            name: "f", arguments: #""{\"a\": \"b\"}""#, family: .deepseekV41)
        #expect(doubled.contains("<\(Self.d) parameter name=\"a\" string=\"true\">b</\(Self.d) parameter>"))
    }

    @Test("V4.1 reasoning policy accepts the ladder and numeric budgets; 0731 unchanged")
    func reasoningPolicy() throws {
        #expect(DeepseekV4ReasoningPolicy.isDeepseekV41(modelType: "deepseek_v41"))
        #expect(!DeepseekV4ReasoningPolicy.isDeepseekV4(modelType: "deepseek_v41"))
        let ctx = try DeepseekV4ReasoningPolicy.normalizedAdditionalContext(
            ["reasoning_effort": "XHigh"], modelType: "deepseek_v41")
        #expect(ctx?["reasoning_effort"] as? String == "xhigh")
        #expect(ctx?["enable_thinking"] as? Bool == true)
        let num = try DeepseekV4ReasoningPolicy.normalizedAdditionalContext(
            ["reasoning_effort": "30"], modelType: "deepseek_v41")
        #expect(num?["reasoning_effort"] as? Int == 30)
        let off = try DeepseekV4ReasoningPolicy.normalizedAdditionalContext(
            ["reasoning_effort": "high", "enable_thinking": false], modelType: "deepseek_v41")
        #expect(off?["reasoning_effort"] == nil)
        #expect(throws: DeepseekV4ReasoningPolicyError.self) {
            _ = try DeepseekV4ReasoningPolicy.normalizedAdditionalContext(
                ["reasoning_effort": "turbo"], modelType: "deepseek_v41")
        }
        // 0731 still rejects the V4.1-only rungs.
        #expect(throws: DeepseekV4ReasoningPolicyError.self) {
            _ = try DeepseekV4ReasoningPolicy.normalizedAdditionalContext(
                ["reasoning_effort": "xhigh"], modelType: "deepseek_v4")
        }
        #expect(ReasoningCapabilityRegistry.policy(for: "deepseek_v41") == DeepseekV41EffortPolicy.self)
        #expect(ReasoningCapabilityRegistry.policy(for: "deepseek_v4") == DeepseekV4EffortPolicy.self)
    }

    @Test("V4.1 affine JANG bundles get uncapped resident pools like plain DSV4")
    func affineCaps() {
        func facts(_ type: String) -> LoadBundleFacts {
            LoadBundleFacts(
                totalSafetensorsBytes: 1, isRouted: true, physicalMemory: 1,
                modelType: type, hasJangConfig: true, numRoutedExperts: 384)
        }
        #expect(facts("deepseek_v41").isDeepseekV41AffineJANG)
        #expect(facts("deepseek_v41").requiresUncappedResidentPools)
        #expect(!facts("deepseek_v4").isDeepseekV41AffineJANG)
        #expect(facts("deepseek_v4").isPlainDeepseekV4AffineJANG)
    }
}
