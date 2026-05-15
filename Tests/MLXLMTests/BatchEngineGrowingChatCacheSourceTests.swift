// Copyright © 2026 Osaurus AI. All rights reserved.

import Foundation
import Testing

@Suite("BatchEngine growing-chat cache source coverage")
struct BatchEngineGrowingChatCacheSourceTests {
    @Test("batch engine stores post-answer cache boundaries and keeps hybrid full-hit guard")
    func batchEngineStoresPostAnswerBoundaryForGrowingChat() throws {
        let source = try String(
            contentsOfFile: "Libraries/MLXLMCommon/BatchEngine/BatchEngine.swift",
            encoding: .utf8)
        let scheduler = try String(
            contentsOfFile: "Libraries/MLXLMCommon/BatchEngine/BatchScheduler.swift",
            encoding: .utf8)

        #expect(scheduler.contains("var generatedTokenIds: [Int] = []"))
        #expect(source.contains("slot.generatedTokenIds.append(tokenID)"))
        #expect(source.contains(#"label: "post-answer""#))
        #expect(source.contains("promptTokens + slot.generatedTokenIds"))
        #expect(source.contains("let unsafePartial = !remaining.isEmpty && hasMediaContent"))
        #expect(source.contains("let unsafeFullHit = remaining.isEmpty && hasPathDependentLayer"))
        #expect(source.contains("layer is MambaCache || layer is ArraysCache || layer is ZayaCCACache"))
        #expect(!source.contains("let unsafePartial = !remaining.isEmpty &&\n                        (hasMediaContent || hasSSMLayer)"))
    }

    @Test("token iterator mirrors post-answer cache boundary policy")
    func tokenIteratorStoresPostAnswerBoundaryForGrowingChat() throws {
        let source = try String(
            contentsOfFile: "Libraries/MLXLMCommon/Evaluate.swift",
            encoding: .utf8)

        #expect(source.contains("mutating func storeCacheAfterGeneration"))
        #expect(source.contains("generatedTokenIds.append(token)"))
        #expect(source.contains("let generatedBoundaryTokens = promptTokenIds + generatedTokenIds"))
        #expect(source.contains("includeGeneratedBoundary: stopReason == .stop && !handler.stopSequenceHit"))
        #expect(source.contains("let unsafePartial = !remainingTokens.isEmpty && hasMediaContent"))
        #expect(source.contains("let unsafeFullHit = remainingTokens.isEmpty && hasPathDependentLayer"))
        #expect(source.contains("layer is MambaCache || layer is ArraysCache || layer is ZayaCCACache"))
        #expect(!source.contains("let unsafePartial = !remainingTokens.isEmpty &&\n                        (hasMediaContent || hasSSMLayer)"))
    }

    @Test("token iterator materializes disk cache restores before prefill")
    func tokenIteratorMaterializesDiskRestoreBeforePrefill() throws {
        let source = try String(
            contentsOfFile: "Libraries/MLXLMCommon/Evaluate.swift",
            encoding: .utf8)

        #expect(source.contains("let diskRestored = restoreFromDiskArrays(diskArrays, into: &self.cache)"))
        #expect(source.contains("MLX.eval(self.cache)"))
        #expect(source.contains("Cache \\(detail.rawValue) hit: restored \\(diskRestored) tokens from disk"))
    }

    @Test("disk cache serializes MLX safetensors IO across model cache instances")
    func diskCacheSerializesMLXSafetensorsIOAcrossInstances() throws {
        let disk = try String(
            contentsOfFile: "Libraries/MLXLMCommon/Cache/DiskCache.swift",
            encoding: .utf8)
        let ssm = try String(
            contentsOfFile: "Libraries/MLXLMCommon/Cache/SSMCompanionDiskStore.swift",
            encoding: .utf8)

        #expect(disk.contains("enum MLXDiskCacheIOLock"))
        #expect(disk.contains("MLXDiskCacheIOLock.shared.lock()"))
        #expect(disk.contains("Stream.gpu.synchronize()"))
        #expect(disk.contains("try loadArraysAndMetadata(url: url)"))
        #expect(disk.contains("try save(arrays: arrays, metadata: [\"format\": \"mlx\"], url: url)"))
        #expect(ssm.contains("MLXDiskCacheIOLock.shared.lock()"))
        #expect(ssm.contains("Stream.gpu.synchronize()"))
        #expect(ssm.contains("loadArraysAndMetadata(url: safetensorsURL)"))
        #expect(ssm.contains("try save(arrays: arrays, metadata: [\"format\": \"mlx\"], url: safetensorsURL)"))
    }

    @Test("token iterator trims full cache hits before one-token seed prefill")
    func tokenIteratorTrimsFullCacheHitBeforeSeedPrefill() throws {
        let source = try String(
            contentsOfFile: "Libraries/MLXLMCommon/Evaluate.swift",
            encoding: .utf8)

        #expect(source.contains("let trimNeeded = cacheOffset - (promptLen - 1)"))
        #expect(source.contains("for layer in self.cache where layer.isTrimmable"))
        #expect(source.contains("_ = layer.trim(trimNeeded)"))
        #expect(source.contains("let lastToken = MLXArray([Int32(last)])"))
    }

    @Test("reasoning close-token forcing is not a decode feature")
    func reasoningCloseTokenForcingIsAbsent() throws {
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
        #expect(!evaluate.contains("token.item(Int.self) == config.tokenID"))
        #expect(!engine.contains("parametersWithAutomaticReasoningCloseBias"))
        #expect(!evaluate.contains("parametersWithAutomaticReasoningCloseBias"))
        #expect(!engine.contains("_parametersWithAutomaticReasoningCloseBias"))
        #expect(!evaluate.contains("_parametersWithAutomaticReasoningCloseBias"))
        #expect(!evaluate.contains("_specialTokenID(\"</think>\", tokenizer: tokenizer)"))
        #expect(!evaluate.contains("name.contains(\"minimax\") || modelTypeName.contains(\"minimax\")"))
        #expect(!evaluate.contains("reasoningCloseBias active"))
    }

    @Test("batch engine has env-gated reasoning prompt-tail diagnostics")
    func batchEngineHasReasoningPromptTailDiagnostics() throws {
        let engine = try String(
            contentsOfFile: "Libraries/MLXLMCommon/BatchEngine/BatchEngine.swift",
            encoding: .utf8)

        #expect(engine.contains("VMLINUX_REASONING_PROMPT_TAIL_LOG"))
        #expect(engine.contains("debugLogReasoningPromptTail"))
        #expect(engine.contains("path: \"BatchEngine.generate\""))
        #expect(engine.contains("path: \"BatchEngine.submit\""))
    }

    @Test("MiniMax stays off compiled decode until parity is proven")
    func minimaxCompiledDecodeIsDenied() throws {
        let evaluate = try String(
            contentsOfFile: "Libraries/MLXLMCommon/Evaluate.swift",
            encoding: .utf8)
        let engine = try String(
            contentsOfFile: "Libraries/MLXLMCommon/BatchEngine/BatchEngine.swift",
            encoding: .utf8)

        #expect(evaluate.contains("typeName.contains(\"minimax\")"))
        #expect(engine.contains("modelName.contains(\"minimax\")"))
        #expect(engine.contains("modelTypeName.contains(\"minimax\")"))
    }
}
