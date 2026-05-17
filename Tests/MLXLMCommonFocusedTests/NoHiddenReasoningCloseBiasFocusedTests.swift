// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Testing

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
