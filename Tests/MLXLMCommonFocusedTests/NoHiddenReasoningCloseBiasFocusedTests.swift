// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

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
