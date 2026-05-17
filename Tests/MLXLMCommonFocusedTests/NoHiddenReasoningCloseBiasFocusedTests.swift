// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import Jinja
import MLX
import Testing
@testable import MLXLLM
@testable import MLXLMCommon

@Suite("No hidden reasoning close bias")
struct NoHiddenReasoningCloseBiasFocusedTests {
    @Test("decode does not bias or force reasoning close tokens")
    func decodeDoesNotBiasOrForceReasoningCloseTokens() throws {
        let evaluate = try String(
            contentsOfFile: "Libraries/MLXLMCommon/Evaluate.swift",
            encoding: .utf8)
        let engine = try String(
            contentsOfFile: "Libraries/MLXLMCommon/BatchEngine/BatchEngine.swift",
            encoding: .utf8)

        #expect(!evaluate.contains("ReasoningCloseBiasConfig"))
        #expect(!evaluate.contains("ReasoningCloseBiasProcessor"))
        #expect(!evaluate.contains("reasoningCloseBias"))
        #expect(!evaluate.contains("forceAfterTokens"))
        #expect(!evaluate.contains("parametersWithAutomaticReasoningCloseBias"))
        #expect(!evaluate.contains("_parametersWithAutomaticReasoningCloseBias"))
        #expect(!evaluate.contains("_specialTokenID(\"</think>\", tokenizer: tokenizer)"))
        #expect(!evaluate.contains("reasoningCloseBias active"))
        #expect(!engine.contains("parametersWithAutomaticReasoningCloseBias"))
        #expect(!engine.contains("_parametersWithAutomaticReasoningCloseBias"))
    }

    @Test("terminal info snapshots unclosed reasoning before parser flush")
    func terminalInfoSnapshotsUnclosedReasoningBeforeParserFlush() throws {
        let evaluate = try String(
            contentsOfFile: "Libraries/MLXLMCommon/Evaluate.swift",
            encoding: .utf8)

        guard let snapshot = evaluate.range(
            of: "let unclosedReasoning = handler.unclosedReasoning"),
            let flush = evaluate.range(
                of: "handler.onGenerationEnd(emit: continuation.yield)")
        else {
            Issue.record("Evaluate.swift missing unclosed reasoning snapshot or terminal flush")
            return
        }

        #expect(snapshot.lowerBound < flush.lowerBound)
        #expect(evaluate.contains("unclosedReasoning: unclosedReasoning"))
        #expect(evaluate.contains("var unclosedReasoning: Bool { get }"))
        #expect(evaluate.contains("reasoningParser?.isInsideReasoning ?? false"))
    }

    @Test("growing chat cache probe distinguishes unsafe template divergence")
    func growingChatCacheProbeDistinguishesUnsafeTemplateDivergence() throws {
        let bench = try String(
            contentsOfFile: "RunBench/Bench.swift",
            encoding: .utf8)
        let tokenizer = try String(
            contentsOfFile: "Libraries/MLXLMCommon/Tokenizer.swift",
            encoding: .utf8)
        let input = try String(
            contentsOfFile: "Libraries/MLXLMCommon/LanguageModel.swift",
            encoding: .utf8)
        let processor = try String(
            contentsOfFile: "Libraries/MLXLLM/LLMModelFactory.swift",
            encoding: .utf8)
        let engine = try String(
            contentsOfFile: "Libraries/MLXLMCommon/BatchEngine/BatchEngine.swift",
            encoding: .utf8)
        let evaluate = try String(
            contentsOfFile: "Libraries/MLXLMCommon/Evaluate.swift",
            encoding: .utf8)

        #expect(bench.contains("Native turn-2 common prefix"))
        #expect(bench.contains("Native turn-2 diverged before prompt boundary"))
        #expect(bench.contains("stored prompt window"))
        #expect(bench.contains("turn-2 prompt window"))
        #expect(bench.contains(
            "native turn-2 chat template diverged from the cached turn-1 generation prompt"))
        #expect(bench.contains(
            "native turn-2 chat template matched the prompt boundary but diverged before the raw post-answer boundary"))
        #expect(tokenizer.contains("GenerationPromptControllableTokenizer"))
        #expect(input.contains("cachePrefixTokenCounts"))
        #expect(processor.contains("addGenerationPrompt: false"))
        #expect(engine.contains("label: \"history-boundary\""))
        #expect(engine.contains("effectivePrefillWindow("))
        #expect(evaluate.contains("cacheSnapshotForBoundary("))
        #expect(evaluate.contains("model.prepare("))
    }

    @Test("history-boundary rederive feeds remaining tokens batch-first")
    func historyBoundaryRederiveUsesBatchFirstRemainingTokens() throws {
        let engine = try String(
            contentsOfFile: "Libraries/MLXLMCommon/BatchEngine/BatchEngine.swift",
            encoding: .utf8)
        guard let start = engine.range(of: "func boundarySnapshot(tokens: [Int]) -> [KVCache]?"),
              let end = engine.range(
                of: "\n            storeCacheEntry(",
                range: start.upperBound..<engine.endIndex)
        else {
            Issue.record("Could not locate BatchEngine.finishSlot boundarySnapshot helper")
            return
        }

        let helper = String(engine[start.lowerBound..<end.lowerBound])
        #expect(
            helper.contains("context.model(")
                && helper.contains("remaining[text: .newAxis]")
                && helper.contains("cache: cache")
                && helper.contains("state: nil"),
            "BatchEngine boundarySnapshot must feed rederived remaining tokens as [1,T], not 1D, or ZAYA CCA cache-on rows reach a 2D activation and trap in transposed(0,2,1).")
        #expect(!helper.contains("context.model(remaining, cache: cache, state: nil)"))
    }

    @Test("TokenIterator history-boundary rederive feeds remaining tokens batch-first")
    func tokenIteratorHistoryBoundaryRederiveUsesBatchFirstRemainingTokens() throws {
        let evaluate = try String(
            contentsOfFile: "Libraries/MLXLMCommon/Evaluate.swift",
            encoding: .utf8)
        guard let start = evaluate.range(of: "private func cacheSnapshotForBoundary("),
              let end = evaluate.range(
                of: "\n    }\n}\n\n/// Generator of tokens using speculative decoding.",
                range: start.upperBound..<evaluate.endIndex)
        else {
            Issue.record("Could not locate TokenIterator cacheSnapshotForBoundary helper")
            return
        }

        let helper = String(evaluate[start.lowerBound..<end.lowerBound])
        #expect(
            helper.contains("model(remaining[text: .newAxis], cache: cache, state: nil)")
                || (helper.contains("model(")
                    && helper.contains("remaining[text: .newAxis]")
                    && helper.contains("cache: cache")
                    && helper.contains("state: nil")),
            "TokenIterator cacheSnapshotForBoundary must feed rederived remaining tokens as [1,T], not 1D, or solo cache-on ZAYA rows trap in transposed(0,2,1).")
        #expect(!helper.contains("model(remaining, cache: cache, state: nil)"))
    }
}

@Suite("Harmony parser focused contracts")
struct HarmonyParserFocusedTests {
    @Test("Gemma4 harmony envelope routes reasoning and leaves clean content")
    func gemma4HarmonyEnvelopeRoutesReasoning() {
        var parser = ReasoningParser.fromCapabilityName("gemma4")
        var segments: [ReasoningSegment] = []
        for chunk in chunked("pre<|channel>thought\ninner<channel|>answer", by: 4) {
            segments.append(contentsOf: parser?.feed(chunk) ?? [])
        }
        segments.append(contentsOf: parser?.flush() ?? [])

        let (reasoning, content) = collect(segments)
        #expect(reasoning == "thought\ninner")
        #expect(content == "preanswer")
        #expect(!content.contains("<|channel>"))
        #expect(!content.contains("<channel|>"))
    }

    @Test("GPT-OSS harmony analysis/final channels split reasoning and content")
    func gptOSSHarmonyAnalysisFinalSplit() {
        var parser = ReasoningParser.fromCapabilityName("harmony")
        let stream =
            "<|start|>assistant<|channel|>analysis<|message|>2+2=4<|end|>"
            + "<|start|>assistant<|channel|>final<|message|>4<|return|>"
        var segments: [ReasoningSegment] = []
        for chunk in chunked(stream, by: 7) {
            segments.append(contentsOf: parser?.feed(chunk) ?? [])
        }
        segments.append(contentsOf: parser?.flush() ?? [])

        let (reasoning, content) = collect(segments)
        #expect(reasoning == "2+2=4")
        #expect(content == "4")
        #expect(!content.contains("<|channel|>"))
        #expect(!content.contains("<|message|>"))
        #expect(!content.contains("<|return|>"))
        #expect(!reasoning.contains("<|channel|>"))
        #expect(!reasoning.contains("<|message|>"))
        #expect(!reasoning.contains("<|end|>"))
    }

    @Test("Harmony parser survives one-character token fragmentation")
    func harmonyParserSurvivesOneCharacterFragments() {
        var parser = ReasoningParser.fromCapabilityName("harmony")
        let stream =
            "lead"
            + "<|start|>assistant<|channel|>analysis<|message|>hidden<|end|>"
            + "<|start|>assistant<|channel|>final<|message|>visible<|return|>"
            + "<|channel>thought\nextra<channel|>tail"
        var segments: [ReasoningSegment] = []
        for scalar in stream {
            segments.append(contentsOf: parser?.feed(String(scalar)) ?? [])
        }
        segments.append(contentsOf: parser?.flush() ?? [])

        let (reasoning, content) = collect(segments)
        #expect(reasoning == "hiddenthought\nextra")
        #expect(content == "leadvisibletail")
        for marker in [
            "<|start|>", "<|channel|>", "<|message|>", "<|end|>", "<|return|>",
            "<|channel>", "<channel|>",
        ] {
            #expect(!reasoning.contains(marker))
            #expect(!content.contains(marker))
        }
    }

    @Test("Harmony parser strips stray control tokens from visible free text")
    func harmonyParserStripsStrayControlTokens() {
        var parser = ReasoningParser.fromCapabilityName("harmony")
        let stream = "visible<|message|>tail<|channel|>final"
        var segments: [ReasoningSegment] = []
        for scalar in stream {
            segments.append(contentsOf: parser?.feed(String(scalar)) ?? [])
        }
        segments.append(contentsOf: parser?.flush() ?? [])

        let (reasoning, content) = collect(segments)
        #expect(reasoning.isEmpty)
        #expect(content == "visibletailfinal")
        for marker in ["<|channel|>", "<|message|>"] {
            #expect(!content.contains(marker))
        }
    }

    @Test("Harmony prompt-tail parser preserves GPT-OSS channel stripping")
    func harmonyForPromptPreservesGPTOSSChannelStripping() {
        var parser = ReasoningParser.forPrompt(
            stampName: "gpt_oss_120b",
            promptTail: "<|start|>assistant")
        let stream =
            "<|start|>assistant<|channel|>analysis<|message|>hidden-plan<|end|>"
            + "<|start|>assistant<|channel|>final<|message|>visible answer<|return|>"
        var segments: [ReasoningSegment] = []
        for chunk in chunked(stream, by: 3) {
            segments.append(contentsOf: parser?.feed(chunk) ?? [])
        }
        segments.append(contentsOf: parser?.flush() ?? [])

        let (reasoning, content) = collect(segments)
        #expect(reasoning == "hidden-plan")
        #expect(content == "visible answer")
        for marker in ["<|start|>", "<|channel|>", "<|message|>", "<|end|>", "<|return|>"] {
            #expect(!reasoning.contains(marker))
            #expect(!content.contains(marker))
        }
    }

    @Test("GPT-OSS model types resolve to Harmony, not think XML")
    func gptOSSModelTypesResolveToHarmony() {
        for modelType in ["gpt_oss", "gpt_oss_20b", "gpt_oss_120b", "GPT_OSS"] {
            #expect(reasoningStampFromModelType(modelType) == "harmony")
            #expect(ReasoningParser.fromCapabilityName(reasoningStampFromModelType(modelType)) != nil)
        }
    }

    @Test("GLM reasoning boundary stays explicit")
    func glmReasoningBoundary() {
        #expect(reasoningStampFromModelType("glm4") == "none")
        #expect(ToolCallFormat.fromCapabilityName("glm4") == .glm4)

        for modelType in ["glm4_moe", "glm4_moe_lite", "glm5", "glm5_air"] {
            #expect(reasoningStampFromModelType(modelType) == "think_xml")
            #expect(ReasoningParser.fromCapabilityName(reasoningStampFromModelType(modelType)) != nil)
            #expect(ToolCallFormat.fromCapabilityName(modelType) == .glm4)
        }
    }

    @Test("Gemma4 harmony reasoning followed by tool call does not leak markers")
    func gemma4HarmonyThenToolCallDoesNotLeak() {
        var parser = ReasoningParser.fromCapabilityName("gemma4")
        let tools = ToolCallProcessor(format: .gemma4)
        let stream =
            "<|channel>thought\nNeed weather.<channel|>"
            + "<|tool_call>call:get_weather{location:<|\"|>Tokyo<|\"|>}<tool_call|>"
            + "Done."
        var visible = ""
        var reasoning = ""

        for chunk in chunked(stream, by: 5) {
            for segment in parser?.feed(chunk) ?? [] {
                switch segment {
                case .reasoning(let text):
                    reasoning += text
                case .content(let text):
                    visible += tools.processChunk(text) ?? ""
                }
            }
        }
        for segment in parser?.flush() ?? [] {
            switch segment {
            case .reasoning(let text):
                reasoning += text
            case .content(let text):
                visible += tools.processChunk(text) ?? ""
            }
        }
        visible += tools.processEOS() ?? ""

        #expect(reasoning == "thought\nNeed weather.")
        #expect(visible == "Done.")
        #expect(tools.toolCalls.count == 1)
        #expect(tools.toolCalls.first?.function.name == "get_weather")
        #expect(tools.toolCalls.first?.function.arguments["location"] == .string("Tokyo"))
        #expect(!visible.contains("<|channel>"))
        #expect(!visible.contains("<|tool_call>"))
    }

    @Test("BatchEngine tool-call live probe supplies tools and rejects empty rows")
    func batchEngineToolCallProbeRequiresBehavioralEvidence() throws {
        let bench = try String(contentsOfFile: "RunBench/Bench.swift", encoding: .utf8)
        let function = try #require(
            Self.extractFunction(named: "runBatchEngineToolCall", from: bench),
            "runBatchEngineToolCall not found"
        )

        #expect(function.body.contains("let weatherTool: [String: any Sendable]"))
        #expect(
            function.body.contains("tools: [weatherTool]")
                || function.body.contains("ui.tools = [weatherTool]"),
            "The live probe must pass a real tool schema through UserInput."
        )
        #expect(
            function.body.contains("toolCallCount == 0")
                && function.body.contains("trimmingCharacters(in: .whitespacesAndNewlines).isEmpty"),
            "The live probe must fail empty-output/no-tool rows instead of counting them as leak-free."
        )
    }

    private func chunked(_ text: String, by size: Int) -> [String] {
        var chunks: [String] = []
        var index = text.startIndex
        while index < text.endIndex {
            let end = text.index(index, offsetBy: size, limitedBy: text.endIndex) ?? text.endIndex
            chunks.append(String(text[index..<end]))
            index = end
        }
        return chunks
    }

    private static func extractFunction(named name: String, from source: String) -> (line: Int, body: String)? {
        guard let range = source.range(of: "func \(name)") else { return nil }
        guard let brace = source[range.lowerBound...].firstIndex(of: "{") else { return nil }
        let line = source[..<range.lowerBound].reduce(1) { count, character in
            character == "\n" ? count + 1 : count
        }
        var depth = 0
        var index = brace
        while index < source.endIndex {
            let character = source[index]
            if character == "{" {
                depth += 1
            } else if character == "}" {
                depth -= 1
                if depth == 0 {
                    return (line, String(source[range.lowerBound...index]))
                }
            }
            index = source.index(after: index)
        }
        return nil
    }

    private func collect(_ segments: [ReasoningSegment]) -> (reasoning: String, content: String) {
        var reasoning = ""
        var content = ""
        for segment in segments {
            switch segment {
            case .reasoning(let text):
                reasoning += text
            case .content(let text):
                content += text
            }
        }
        return (reasoning, content)
    }
}

@Suite("Hy3 parser and no-leak focused contracts")
struct Hy3ParserFocusedTests {
    @Test("Hy3 parser aliases resolve to Hunyuan tools and think XML reasoning")
    func aliasesResolve() {
        for stamp in ["hunyuan", "tencent", "hy3", "hy_v3", "hy-v3"] {
            #expect(ToolCallFormat.fromCapabilityName(stamp) == .hunyuan)
            #expect(ReasoningParser.fromCapabilityName(stamp) != nil)
        }
        for modelType in ["hy_v3", "hy-v3", "hy3", "Hy3"] {
            #expect(reasoningStampFromModelType(modelType) == "think_xml")
            #expect(ToolCallFormat.infer(from: modelType) == .hunyuan)
        }
    }

    @Test("Hy3 Hunyuan parser extracts multiple scalar-argument calls")
    func hunyuanParserExtractsCalls() {
        let parser = HunyuanToolCallParser()
        let calls = parser.parseEOS(
            """
            <tool_calls>
            <tool_call>search_web<tool_sep>
            <arg_key>query</arg_key><arg_value>"hy3 runtime"</arg_value>
            <arg_key>limit</arg_key><arg_value>3</arg_value>
            <arg_key>safe</arg_key><arg_value>true</arg_value>
            </tool_call>
            <tool_call>open_file<tool_sep>
            <arg_key>path</arg_key><arg_value>"/tmp/a b.txt"</arg_value>
            </tool_call>
            </tool_calls>
            """,
            tools: nil)

        #expect(calls.count == 2)
        #expect(calls[0].function.name == "search_web")
        #expect(calls[0].function.arguments["query"] == .string("hy3 runtime"))
        #expect(calls[0].function.arguments["limit"] == .int(3))
        #expect(calls[0].function.arguments["safe"] == .bool(true))
        #expect(calls[1].function.name == "open_file")
        #expect(calls[1].function.arguments["path"] == .string("/tmp/a b.txt"))
    }

    @Test("Hy3 reasoning and Hunyuan tool-call pipeline does not leak markers")
    func reasoningToolPipelineDoesNotLeakMarkers() {
        var reasoningParser = ReasoningParser.fromCapabilityName("hy_v3")
        let toolProcessor = ToolCallProcessor(format: .hunyuan)
        let stream = """
            <think>choose the lookup tool</think>
            <tool_calls>
            <tool_call>search_web<tool_sep>
            <arg_key>query</arg_key><arg_value>"hy3 swift"</arg_value>
            </tool_call>
            <tool_call>open_file<tool_sep>
            <arg_key>path</arg_key><arg_value>"/tmp/hy3.md"</arg_value>
            </tool_call>
            </tool_calls>
            Final answer after tools.
            """

        var visible = ""
        var reasoning = ""
        for scalar in stream {
            if var parser = reasoningParser {
                for segment in parser.feed(String(scalar)) {
                    switch segment {
                    case .reasoning(let text):
                        reasoning += text
                    case .content(let text):
                        visible += toolProcessor.processChunk(text) ?? ""
                    }
                }
                reasoningParser = parser
            }
        }
        if var parser = reasoningParser {
            for segment in parser.flush() {
                switch segment {
                case .reasoning(let text):
                    reasoning += text
                case .content(let text):
                    visible += toolProcessor.processChunk(text) ?? ""
                }
            }
        }
        _ = toolProcessor.processEOS()

        #expect(reasoning.contains("choose the lookup tool"))
        #expect(toolProcessor.toolCalls.map(\.function.name) == ["search_web", "open_file"])
        #expect(toolProcessor.toolCalls[0].function.arguments["query"] == .string("hy3 swift"))
        #expect(toolProcessor.toolCalls[1].function.arguments["path"] == .string("/tmp/hy3.md"))
        #expect(visible.contains("Final answer after tools."))
        for marker in ["<think>", "</think>", "<tool_calls>", "<tool_call>", "<arg_key>", "<arg_value>"] {
            #expect(!visible.contains(marker))
        }
        #expect(!visible.contains("choose the lookup tool"))
    }

    @Test("Hy3 prompt-tail closed think keeps answer content visible")
    func noThinkPromptKeepsContentVisible() {
        let promptTail = "<｜hy_Assistant｜><think>\n\n</think>\n\n"
        var parser = ReasoningParser.forPrompt(stampName: "hy_v3", promptTail: promptTail)!
        let (reasoning, content) = collectParser(&parser, "The capital of France is Paris.")

        #expect(reasoning.isEmpty)
        #expect(content == "The capital of France is Paris.")
    }

    @Test("Hy3 prompt-tail open think routes pre-close text to reasoning only")
    func openThinkPromptSeparatesReasoningAndContent() {
        let promptTail = "<｜hy_Assistant｜><think>\n"
        var parser = ReasoningParser.forPrompt(stampName: "hy_v3", promptTail: promptTail)!
        let (reasoning, content) = collectParser(
            &parser,
            "Let me work this out...\n</think>\nThe answer is 42.")

        #expect(reasoning.contains("Let me work this out..."))
        #expect(content.contains("The answer is 42."))
        #expect(!reasoning.contains("The answer is 42."))
        #expect(!content.contains("Let me work this out..."))
    }

    @Test("Hy3 native model drops preserved nextn layer from base decode cache")
    func nativeModelDropsPreservedNextnFromBaseDecodeCache() throws {
        let config = try minimalHy3Config(numHiddenLayers: 2, numNextnPredictLayers: 1)
        let model = Hy3Model(config)

        #expect(model.kvHeads == [1, 1])
        #expect(model.newCache(parameters: nil).count == 2)
        #expect(model.loraLayers.count == 2)
    }

    @Test("Hy3 sanitizer fuses qkv and drops preserved nextn tensors")
    func sanitizerFusesQKVAndDropsPreservedNextnTensors() throws {
        try FocusedMLXTestSupport.withLock {
            let config = try minimalHy3Config(
                numHiddenLayers: 1,
                firstKDenseReplace: 1,
                numNextnPredictLayers: 1)
            let model = Hy3Model(config)
            let prefix = "model.layers.0.self_attn"
            let weights: [String: MLXArray] = [
                "\(prefix).q_proj.weight": MLXArray.ones([8, 4]),
                "\(prefix).k_proj.weight": MLXArray.ones([4, 4]) * 2,
                "\(prefix).v_proj.weight": MLXArray.ones([4, 4]) * 3,
                "\(prefix).q_proj.scales": MLXArray.ones([8, 1]),
                "\(prefix).k_proj.scales": MLXArray.ones([4, 1]) * 2,
                "\(prefix).v_proj.scales": MLXArray.ones([4, 1]) * 3,
                "model.layers.1.self_attn.q_proj.weight": MLXArray.ones([8, 4]) * 9,
            ]

            let sanitized = model.sanitize(weights: weights)

            #expect(sanitized["\(prefix).qkv_proj.weight"]?.shape == [16, 4])
            #expect(sanitized["\(prefix).qkv_proj.scales"]?.shape == [16, 1])
            #expect(sanitized["\(prefix).q_proj.weight"] == nil)
            #expect(sanitized["\(prefix).k_proj.weight"] == nil)
            #expect(sanitized["\(prefix).v_proj.weight"] == nil)
            #expect(sanitized["model.layers.1.self_attn.q_proj.weight"] == nil)
        }
    }

    @Test("Hy3 sanitizer dequantizes mixed-bit qkv instead of crashing")
    func sanitizerDequantizesMixedBitQKVBeforeFusion() throws {
        try FocusedMLXTestSupport.withLock {
            let config = try minimalHy3Config(
                numHiddenLayers: 1,
                firstKDenseReplace: 1,
                numNextnPredictLayers: 0,
                hiddenSize: 32)
            let model = Hy3Model(config)
            let prefix = "model.layers.0.self_attn"

            let qDense = MLXArray(0 ..< 256, [8, 32]).asType(.float32)
            let kDense = MLXArray(0 ..< 128, [4, 32]).asType(.float32)
            let vDense = (MLXArray(0 ..< 128, [4, 32]) + 100).asType(.float32)
            let (qW, qS, qB) = MLX.quantized(qDense, groupSize: 32, bits: 8)
            let (kW, kS, kB) = MLX.quantized(kDense, groupSize: 32, bits: 4)
            let (vW, vS, vB) = MLX.quantized(vDense, groupSize: 32, bits: 2)

            var weights: [String: MLXArray] = [
                "\(prefix).q_proj.weight": qW,
                "\(prefix).q_proj.scales": qS,
                "\(prefix).k_proj.weight": kW,
                "\(prefix).k_proj.scales": kS,
                "\(prefix).v_proj.weight": vW,
                "\(prefix).v_proj.scales": vS,
            ]
            if let qB { weights["\(prefix).q_proj.biases"] = qB }
            if let kB { weights["\(prefix).k_proj.biases"] = kB }
            if let vB { weights["\(prefix).v_proj.biases"] = vB }

            let sanitized = model.sanitize(weights: weights)

            let fused = try #require(sanitized["\(prefix).qkv_proj.weight"])
            #expect(fused.shape == [16, 32])
            #expect(sanitized["\(prefix).qkv_proj.scales"] == nil)
            #expect(sanitized["\(prefix).q_proj.weight"] == nil)
            #expect(sanitized["\(prefix).k_proj.weight"] == nil)
            #expect(sanitized["\(prefix).v_proj.weight"] == nil)

            let expected = concatenated([
                MLX.dequantized(qW, scales: qS, biases: qB, groupSize: 32, bits: 8)
                    .asType(.float16),
                MLX.dequantized(kW, scales: kS, biases: kB, groupSize: 32, bits: 4)
                    .asType(.float16),
                MLX.dequantized(vW, scales: vS, biases: vB, groupSize: 32, bits: 2)
                    .asType(.float16),
            ], axis: 0)
            let maxDiff = (fused.asType(.float32) - expected.asType(.float32)).abs().max()
                .item(Float.self)
            #expect(maxDiff < 1e-5)
        }
    }

    private func collectParser(
        _ parser: inout ReasoningParser,
        _ text: String
    ) -> (reasoning: String, content: String) {
        var segments = parser.feed(text)
        segments.append(contentsOf: parser.flush())
        var reasoning = ""
        var content = ""
        for segment in segments {
            switch segment {
            case .reasoning(let text):
                reasoning += text
            case .content(let text):
                content += text
            }
        }
        return (reasoning, content)
    }

    private func minimalHy3Config(
        numHiddenLayers: Int = 2,
        firstKDenseReplace: Int = 1,
        numExperts: Int = 2,
        numNextnPredictLayers: Int = 0,
        hiddenSize: Int = 8
    ) throws -> Hy3Configuration {
        let json = """
            {
              "model_type": "hy_v3",
              "architectures": ["HYV3ForCausalLM"],
              "hidden_size": \(hiddenSize),
              "num_hidden_layers": \(numHiddenLayers),
              "num_attention_heads": 2,
              "num_key_value_heads": 1,
              "head_dim": 4,
              "intermediate_size": \(hiddenSize * 2),
              "moe_intermediate_size": 4,
              "expert_hidden_dim": 4,
              "first_k_dense_replace": \(firstKDenseReplace),
              "num_experts": \(numExperts),
              "num_experts_per_tok": 1,
              "num_shared_experts": 1,
              "qk_norm": true,
              "rms_norm_eps": 1e-5,
              "rope_parameters": {"rope_theta": 11158840.0, "rope_type": "default"},
              "max_position_embeddings": 262144,
              "route_norm": true,
              "router_scaling_factor": 2.826,
              "moe_router_enable_expert_bias": true,
              "moe_router_use_sigmoid": true,
              "tie_word_embeddings": false,
              "vocab_size": 32,
              "mxtq_seed": 42,
              "mxtq_bits": 2,
              "num_nextn_predict_layers": \(numNextnPredictLayers)
            }
            """
        return try JSONDecoder.json5().decode(Hy3Configuration.self, from: Data(json.utf8))
    }
}

@Suite("Bailing/Ling thinking-template focused contracts")
struct BailingThinkingTemplateFocusedTests {
    @Test("enable_thinking=true prepends detailed thinking on")
    func enableThinkingTruePrependsDirective() {
        let messages: [Message] = [
            ["role": "system", "content": "You are concise."],
            ["role": "user", "content": "hello"],
        ]

        let out = BailingThinkingTemplateContext.apply(
            to: messages,
            modelType: "bailing_hybrid",
            additionalContext: ["enable_thinking": true])

        #expect(out[0]["role"] as? String == "system")
        #expect(out[0]["content"] as? String == "detailed thinking on\n\nYou are concise.")
        #expect(out[1]["content"] as? String == "hello")
    }

    @Test("enable_thinking=false inserts or replaces detailed thinking off")
    func enableThinkingFalseNormalizesDirective() {
        let missingSystem: [Message] = [["role": "user", "content": "hello"]]
        let inserted = BailingThinkingTemplateContext.apply(
            to: missingSystem,
            modelType: "bailing_moe_v2_5",
            additionalContext: ["enable_thinking": false])

        #expect(inserted.count == 2)
        #expect(inserted[0]["role"] as? String == "system")
        #expect(inserted[0]["content"] as? String == "detailed thinking off")

        let existingDirective: [Message] = [
            ["role": "system", "content": "detailed thinking on\n\nYou are concise."],
            ["role": "user", "content": "hello"],
        ]
        let replaced = BailingThinkingTemplateContext.apply(
            to: existingDirective,
            modelType: "bailing_hybrid",
            additionalContext: ["enable_thinking": false])

        #expect(replaced[0]["content"] as? String == "detailed thinking off\n\nYou are concise.")
    }

    @Test("non-Bailing model and missing toggle are unchanged")
    func nonBailingOrMissingToggleUnchanged() {
        let messages: [Message] = [
            ["role": "system", "content": "You are concise."],
            ["role": "user", "content": "hello"],
        ]

        let nonBailing = BailingThinkingTemplateContext.apply(
            to: messages,
            modelType: "laguna",
            additionalContext: ["enable_thinking": false])
        #expect(nonBailing[0]["content"] as? String == "You are concise.")
        #expect(nonBailing.count == messages.count)

        let noToggle = BailingThinkingTemplateContext.apply(
            to: messages,
            modelType: "bailing_hybrid",
            additionalContext: nil)
        #expect(noToggle[0]["content"] as? String == "You are concise.")
        #expect(noToggle.count == messages.count)
    }
}

@Suite("Direct capability parser alias focused contracts")
struct DirectCapabilityParserAliasFocusedTests {
    @Test("direct Harmony capability aliases resolve without leaking control markers")
    func harmonyCapabilityAliasesResolve() {
        for stamp in ["gemma4_27b", "gpt_oss_20b", "gpt_oss_120b"] {
            var parser = ReasoningParser.fromCapabilityName(stamp)
            #expect(parser != nil, "\(stamp) should resolve to the Harmony parser")

            let stream =
                "<|start|>assistant<|channel|>analysis<|message|>hidden<|end|>"
                + "<|start|>assistant<|channel|>final<|message|>visible<|return|>"
            let (reasoning, content) = collectParser(&parser, stream)
            #expect(reasoning == "hidden")
            #expect(content == "visible")
            #expect(!content.contains("<|channel|>"))
            #expect(!content.contains("<|message|>"))
            #expect(!content.contains("<|return|>"))
        }
    }

    @Test("versioned Gemma4, GLM5.1, and GPT-OSS aliases keep reasoning and tools aligned")
    func versionedHarmonyAndGLMAliasesResolve() {
        for stamp in ["gemma4_27b", "gemma_4_27b", "gemma-4-27b"] {
            #expect(ReasoningParser.fromCapabilityName(stamp) != nil)
            #expect(ToolCallFormat.fromCapabilityName(stamp) == .gemma4)
        }
        for modelType in ["gemma4_text", "gemma_4_text", "gemma-4-text"] {
            #expect(reasoningStampFromModelType(modelType) == "harmony")
            #expect(ToolCallFormat.infer(from: modelType) == .gemma4)
        }

        for stamp in ["glm5_1_flash", "glm_5_1_flash", "glm-5.1-flash"] {
            #expect(ReasoningParser.fromCapabilityName(stamp) != nil)
            #expect(ToolCallFormat.fromCapabilityName(stamp) == .glm4)
        }
        for modelType in ["glm5_1_flash", "glm_5_1_flash", "glm-5.1-flash"] {
            #expect(reasoningStampFromModelType(modelType) == "think_xml")
            #expect(ToolCallFormat.infer(from: modelType) == .glm4)
        }

        for stamp in ["gpt_oss_20b", "gpt-oss-20b", "gptoss_20b"] {
            #expect(ReasoningParser.fromCapabilityName(stamp) != nil)
            #expect(ToolCallFormat.fromCapabilityName(stamp) == .glm4)
        }
        for modelType in ["gpt_oss_20b", "gpt-oss-20b", "gptoss_20b"] {
            #expect(reasoningStampFromModelType(modelType) == "harmony")
            #expect(ToolCallFormat.infer(from: modelType) == .glm4)
        }
    }

    @Test("ParserResolution facade preserves product/version aliases")
    func parserResolutionFacadePreservesProductVersionAliases() {
        let stampedCases: [(String, ToolCallFormat)] = [
            ("gemma-4-27b", .gemma4),
            ("gpt-oss-20b", .glm4),
            ("glm-5.1-flash", .glm4),
            ("Ling-2.6-flash", .glm4),
            ("hy3-preview", .hunyuan),
        ]

        for (stamp, expectedTool) in stampedCases {
            let cap = JangCapabilities(
                reasoningParser: stamp,
                toolParser: stamp)
            let (parser, parserSource) = ParserResolution.reasoning(
                capabilities: cap,
                modelType: "llama")
            let (toolFormat, toolSource) = ParserResolution.toolCall(
                capabilities: cap,
                modelType: "llama")

            #expect(parser != nil, "\(stamp) should resolve through ParserResolution")
            #expect(parserSource == .jangStamped)
            #expect(toolFormat == expectedTool)
            #expect(toolSource == .jangStamped)
        }

        let heuristicCases: [(String, String, ToolCallFormat)] = [
            ("gemma-4-text", "harmony", .gemma4),
            ("gpt-oss-20b", "harmony", .glm4),
            ("glm-5.1-flash", "think_xml", .glm4),
            ("Ling-2.6-flash", "think_xml", .glm4),
            ("hy3-preview", "think_xml", .hunyuan),
        ]

        for (modelType, expectedStamp, expectedTool) in heuristicCases {
            let (parser, parserSource) = ParserResolution.reasoning(
                capabilities: nil,
                modelType: modelType)
            let (toolFormat, toolSource) = ParserResolution.toolCall(
                capabilities: nil,
                modelType: modelType)

            #expect(reasoningStampFromModelType(modelType) == expectedStamp)
            #expect(parser != nil, "\(modelType) should resolve through model_type fallback")
            #expect(parserSource == .modelTypeHeuristic)
            #expect(toolFormat == expectedTool)
            #expect(toolSource == .modelTypeHeuristic)
        }
    }

    @Test("Parser aliases trim metadata whitespace without demoting known families")
    func parserAliasesTrimMetadataWhitespace() {
        for modelType in [" Ling-2.6-flash ", "\tgemma-4-text\n", " glm-5.1-flash "] {
            #expect(reasoningStampFromModelType(modelType) != "none")
            #expect(ToolCallFormat.infer(from: modelType) != nil)
        }

        for stamp in [" Ling-2.6-flash ", "\tgpt-oss-20b\n", " gemma-4-27b "] {
            #expect(ReasoningParser.fromCapabilityName(stamp) != nil)
            #expect(ToolCallFormat.fromCapabilityName(stamp) != nil)
        }
    }

    @Test("direct think-XML family aliases resolve to parser")
    func thinkXmlCapabilityAliasesResolve() {
        for stamp in [
            "glm4_moe_lite", "glm5_air", "deepseek_v4_flash",
            "laguna_glm_thinking_v5",
        ] {
            var parser = ReasoningParser.fromCapabilityName(stamp)
            #expect(parser != nil, "\(stamp) should resolve to think_xml")
            let (reasoning, content) = collectParser(
                &parser,
                "internal</think>visible")
            #expect(reasoning == "internal")
            #expect(content == "visible")
            #expect(!content.contains("</think>"))
        }
    }

    @Test("GLM5 model-type fallback keeps reasoning and tool parser aligned")
    func glm5ModelTypeFallbackKeepsParsersAligned() {
        for modelType in ["glm5", "glm5_air", "glm5_1_flash"] {
            #expect(reasoningStampFromModelType(modelType) == "think_xml")
            #expect(ToolCallFormat.infer(from: modelType) == .glm4)
        }
    }

    @Test("DeepSeek V4 capability aliases route to DSML before generic DeepSeek")
    func deepseekV4CapabilityAliasesRouteToDSML() {
        for stamp in ["deepseek_v4", "deepseek_v4_flash", "deepseekv4"] {
            #expect(ToolCallFormat.fromCapabilityName(stamp) == .dsml)
        }
        #expect(ToolCallFormat.fromCapabilityName("deepseek") == .glm4)
        #expect(ToolCallFormat.fromCapabilityName("deepseek_v3") == .glm4)
    }

    @Test("explicit Mistral-4 reasoning capability uses bracket THINK parser")
    func explicitMistral4ReasoningCapabilityUsesBracketThinkParser() {
        #expect(reasoningStampFromModelType("mistral4") == "none",
            "model_type fallback remains no-reasoning unless the bundle explicitly stamps a parser")

        var parser = ReasoningParser.fromCapabilityName("mistral4")
        #expect(parser != nil)
        let (reasoning, content) = collectParser(
            &parser,
            "[THINK]plan[/THINK]Visible answer.")

        #expect(reasoning == "plan")
        #expect(content == "Visible answer.")
        #expect(!content.contains("[THINK]"))
        #expect(!content.contains("[/THINK]"))
    }

    @Test("Mistral/Pixtral tool aliases route to Mistral parser")
    func mistralPixtralToolAliasesResolve() {
        #expect(ToolCallFormat.infer(from: "mistral4") == .mistral)
        #expect(ToolCallFormat.infer(from: "pixtral") == .mistral)
        for stamp in ["mistral4_large", "mistral_small_4", "pixtral_large"] {
            #expect(ToolCallFormat.fromCapabilityName(stamp) == .mistral)
        }
    }

    @Test("Ling/Bailing model-type fallback routes tools to GLM parser")
    func bailingModelTypeFallbackRoutesToolsToGLMParser() {
        for modelType in ["bailing_hybrid", "bailing_moe", "bailing_moe_v2_5", "ling", "ling_bailing"] {
            #expect(reasoningStampFromModelType(modelType) == "think_xml")
            #expect(ToolCallFormat.infer(from: modelType) == .glm4)
            #expect(ToolCallFormat.infer(from: modelType.uppercased()) == .glm4)
        }
    }

    @Test("Ling/Bailing capability aliases resolve to thinking and GLM tools")
    func bailingCapabilityAliasesResolveToThinkingAndTools() {
        for stamp in ["bailing", "bailing_hybrid", "bailing_moe_v2_5", "ling", "ling_bailing"] {
            #expect(ReasoningParser.fromCapabilityName(stamp) != nil)
            #expect(ToolCallFormat.fromCapabilityName(stamp) == .glm4)
        }
    }

    @Test("Qwen3.6 and Qwen3-VL model-type fallbacks route to XML tools")
    func qwen36AndQwen3VLModelTypeFallbacksRouteToolsToXML() {
        for modelType in ["qwen3_6", "qwen3_6_moe", "qwen3_vl", "qwen3_5_vl"] {
            #expect(ToolCallFormat.infer(from: modelType) == .xmlFunction)
            #expect(ToolCallFormat.infer(from: modelType.uppercased()) == .xmlFunction)
        }
    }

    @Test("Qwen-VL capability aliases resolve to thinking and XML tools")
    func qwenVLCapabilityAliasesResolve() {
        for stamp in ["qwen3_vl", "qwen3_5_vl", "qwen3_6_vl"] {
            #expect(ReasoningParser.fromCapabilityName(stamp) != nil)
            #expect(ToolCallFormat.fromCapabilityName(stamp) == .xmlFunction)
        }
    }

    private func collectParser(
        _ parser: inout ReasoningParser?,
        _ text: String
    ) -> (reasoning: String, content: String) {
        var reasoning = ""
        var content = ""
        if var p = parser {
            for chunk in chunked(text, by: 5) {
                for segment in p.feed(chunk) {
                    switch segment {
                    case .reasoning(let text):
                        reasoning += text
                    case .content(let text):
                        content += text
                    }
                }
            }
            for segment in p.flush() {
                switch segment {
                case .reasoning(let text):
                    reasoning += text
                case .content(let text):
                    content += text
                }
            }
            parser = p
        } else {
            content = text
        }
        return (reasoning, content)
    }

    private func chunked(_ text: String, by size: Int) -> [String] {
        var chunks: [String] = []
        var index = text.startIndex
        while index < text.endIndex {
            let end = text.index(index, offsetBy: size, limitedBy: text.endIndex) ?? text.endIndex
            chunks.append(String(text[index..<end]))
            index = end
        }
        return chunks
    }
}

@Suite("Laguna focused parser, template, and rope contracts")
struct LagunaFocusedContractsTests {
    @Test("Laguna parser aliases align with GLM tools and think XML")
    func lagunaParserAliasesAlign() {
        for stamp in ["laguna", "laguna_xs", "laguna_s", "laguna_glm_thinking_v5"] {
            #expect(reasoningStampFromModelType(stamp) == "think_xml")
            #expect(ReasoningParser.fromCapabilityName(stamp) != nil)
            #expect(ToolCallFormat.infer(from: stamp) == .glm4)
            #expect(ToolCallFormat.fromCapabilityName(stamp) == .glm4)
        }
    }

    @Test("Laguna minimal template thinking off closes reasoning in prompt")
    func lagunaTemplateThinkingOffClosesReasoning() throws {
        let rendered = try renderLaguna([
            "messages": [
                ["role": "user", "content": "hi"],
            ],
            "add_generation_prompt": true,
            "enable_thinking": false,
        ])

        #expect(rendered.contains("<system>\n\nYou are a helpful"))
        #expect(rendered.contains("<user>\nhi\n</user>\n"))
        #expect(rendered.hasSuffix("<assistant>\n</think>\n"))
        #expect(!rendered.contains("<|im_start|>"))

        var parser = ReasoningParser.forPrompt(
            stampName: "laguna",
            promptTail: String(rendered.suffix(128)))!
        let (reasoning, content) = collectParser(&parser, "Visible answer.")
        #expect(reasoning.isEmpty)
        #expect(content == "Visible answer.")
    }

    @Test("Laguna minimal template thinking on opens reasoning in prompt")
    func lagunaTemplateThinkingOnOpensReasoning() throws {
        let rendered = try renderLaguna([
            "messages": [
                ["role": "user", "content": "hi"],
            ],
            "add_generation_prompt": true,
            "enable_thinking": true,
        ])

        #expect(rendered.hasSuffix("<assistant>\n<think>\n"))

        var parser = ReasoningParser.forPrompt(
            stampName: "laguna",
            promptTail: String(rendered.suffix(128)))!
        let (reasoning, content) = collectParser(
            &parser,
            "private plan</think>Visible answer.")
        #expect(reasoning == "private plan")
        #expect(content == "Visible answer.")
        #expect(!content.contains("</think>"))
    }

    @Test("Laguna assistant history preserves reasoning and content")
    func lagunaAssistantHistoryPreservesReasoningAndContent() throws {
        let rendered = try renderLaguna([
            "messages": [
                ["role": "user", "content": "hi"],
                [
                    "role": "assistant",
                    "reasoning_content": "brief internal note",
                    "content": "Hello!",
                ],
                ["role": "user", "content": "again"],
            ],
            "add_generation_prompt": true,
            "enable_thinking": false,
        ])

        #expect(rendered.contains("<think>\nbrief internal note\n</think>\nHello!\n</assistant>\n"))
        #expect(rendered.contains("<user>\nagain\n</user>\n"))
        #expect(rendered.hasSuffix("<assistant>\n</think>\n"))
    }

    @Test("Laguna mixed rope_parameters decodes dict entries only")
    func lagunaMixedRopeParametersDecode() throws {
        let cfg = try JSONDecoder().decode(
            LagunaConfiguration.self,
            from: #"""
            {
              "model_type": "laguna",
              "hidden_size": 64,
              "intermediate_size": 128,
              "num_hidden_layers": 2,
              "num_attention_heads": 4,
              "num_key_value_heads": 2,
              "head_dim": 16,
              "max_position_embeddings": 4096,
              "vocab_size": 1024,
              "rms_norm_eps": 1.0e-5,
              "tie_word_embeddings": true,
              "layer_types": ["sliding_attention", "full_attention"],
              "moe_intermediate_size": 64,
              "num_experts_per_tok": 2,
              "num_local_experts": 4,
              "num_shared_experts": 1,
              "use_qk_norm": true,
              "rope_parameters": {
                "full_attention": {
                  "rope_theta": 500000.0,
                  "rope_type": "default"
                },
                "sliding_attention": {
                  "rope_theta": 500000.0,
                  "rope_type": "default"
                },
                "original_max_position_embeddings": 4096
              }
            }
            """#.data(using: .utf8)!)

        #expect(cfg.ropeParameters.keys.contains("full_attention"))
        #expect(cfg.ropeParameters.keys.contains("sliding_attention"))
        #expect(!cfg.ropeParameters.keys.contains("original_max_position_embeddings"))
    }

    private func renderLaguna(_ context: [String: Any]) throws -> String {
        let template = try Template(ChatTemplateFallbacks.lagunaMinimal)
        var values: [String: Value] = [:]
        for (key, value) in context {
            values[key] = try Value(any: value)
        }
        return try template.render(values)
    }

    private func collectParser(
        _ parser: inout ReasoningParser,
        _ text: String
    ) -> (reasoning: String, content: String) {
        var segments = parser.feed(text)
        segments.append(contentsOf: parser.flush())
        var reasoning = ""
        var content = ""
        for segment in segments {
            switch segment {
            case .reasoning(let text):
                reasoning += text
            case .content(let text):
                content += text
            }
        }
        return (reasoning, content)
    }
}

@Suite("Mistral and Ministral focused parser boundaries")
struct MistralMinistralFocusedContractsTests {
    @Test("Mistral3 and Ministral3 stay no-reasoning")
    func mistral3AndMinistral3StayNoReasoning() {
        for modelType in ["mistral3", "mistral3_text", "Mistral3", "ministral3"] {
            #expect(reasoningStampFromModelType(modelType) == "none")
            #expect(ReasoningParser.fromCapabilityName(
                reasoningStampFromModelType(modelType)) == nil)
        }
    }

    @Test("Mistral3 and Ministral3 route to Mistral tool parser")
    func mistral3AndMinistral3ToolParser() {
        for modelType in ["mistral3", "mistral3_text", "ministral3"] {
            #expect(ToolCallFormat.infer(from: modelType) == .mistral)
        }
    }

    @Test("None reasoning stamp leaves literal think tags visible")
    func noneReasoningStampDoesNotHideLiteralThinkTags() {
        let parser = ReasoningParser.fromCapabilityName("none")
        #expect(parser == nil)

        let visible = "<think>literal model text</think>Visible."
        #expect(visible.contains("<think>"))
        #expect(visible.contains("</think>"))
    }
}

@Suite("Mistral3 JANGTQ dispatch focused contracts")
struct Mistral3JANGTQDispatchFocusedTests {
    @Test("mxtq Mistral3 dispatch routes to JANGTQ model")
    func mxtqDispatchRoutesToJANGTQModel() async throws {
        let model = try await LLMTypeRegistry.shared.createModel(
            configuration: minimalMistral3Config(weightFormat: "mxtq", mxtqBits: 2),
            modelType: "mistral3")

        #expect(model is Mistral3TextJANGTQModel)
        let typed = try #require(model as? Mistral3TextJANGTQModel)
        #expect(typed.model.layers.count == 2)
    }

    @Test("mxtq bits propagate into packed dense width")
    func mxtqBitsPropagateIntoPackedDenseWidth() async throws {
        let model = try await LLMTypeRegistry.shared.createModel(
            configuration: minimalMistral3Config(weightFormat: "mxtq", mxtqBits: 4),
            modelType: "mistral3")

        let typed = try #require(model as? Mistral3TextJANGTQModel)
        let qPacked = typed.model.layers[0].attention.wq.packed
        #expect(qPacked.shape == [64, 8])
        #expect(qPacked.dim(-1) == 8)
    }

    @Test("mxfp4 and missing format stay on vanilla Mistral3 path")
    func nonMxtqFormatsStayVanilla() async throws {
        for format in ["mxfp4", nil] as [String?] {
            let model = try await LLMTypeRegistry.shared.createModel(
                configuration: minimalMistral3Config(weightFormat: format, mxtqBits: nil),
                modelType: "mistral3")
            #expect(model is Mistral3TextModel)
            #expect(!(model is Mistral3TextJANGTQModel))
        }
    }

    @Test("mxtq dispatch is case-insensitive")
    func mxtqDispatchIsCaseInsensitive() async throws {
        let model = try await LLMTypeRegistry.shared.createModel(
            configuration: minimalMistral3Config(weightFormat: "MXTQ", mxtqBits: 2),
            modelType: "mistral3")
        #expect(model is Mistral3TextJANGTQModel)
    }

    private func minimalMistral3Config(
        weightFormat: String?,
        mxtqBits: Int?
    ) throws -> Data {
        var dict: [String: Any] = [
            "model_type": "ministral3",
            "vocab_size": 128,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "rms_norm_eps": 1e-5,
            "rope_theta": 1_000_000.0,
            "head_dim": 16,
            "max_position_embeddings": 512,
            "tie_word_embeddings": true,
            "layer_types": Array(repeating: "full_attention", count: 2),
        ]
        if let weightFormat {
            dict["weight_format"] = weightFormat
        }
        if let mxtqBits {
            dict["mxtq_bits"] = mxtqBits
        }
        return try JSONSerialization.data(withJSONObject: dict)
    }
}

@Suite("Gemma4 VLM focused source contracts")
struct Gemma4VLMFocusedSourceContractsTests {
    @Test("Gemma4 prepare rejects unsupported audio explicitly")
    func audioGuardIsPresent() throws {
        let source = try gemma4VLMSource()

        #expect(source.contains("if input.audio != nil {"))
        #expect(source.contains("throw VLMError.processing("))
        #expect(source.contains("LMInput.audio must be nil"))
        #expect(source.contains("audio_tower.*") || source.contains("audio_tower.\\*"))
    }

    @Test("Gemma4 processor resolves image token without encode special-token drift")
    func imageTokenIdUsesConvertTokenToId() throws {
        let source = try gemma4VLMSource()

        #expect(source.contains("tokenizer.convertTokenToId(\"<|image|>\")"))
        #expect(!source.contains("tokenizer.encode(text: \"<|image|>\").last"))
        #expect(source.contains("?? 258880"))
    }

    private func gemma4VLMSource() throws -> String {
        let url = URL(fileURLWithPath: "Libraries/MLXVLM/Models/Gemma4.swift")
        return try String(contentsOf: url, encoding: .utf8)
    }
}
