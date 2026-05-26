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
        #expect(scheduler.contains("var cachePromptTokenIds: [Int]"))
        #expect(scheduler.contains("var cachePromptUsesPostPrepareKey: Bool"))
        #expect(source.contains("slot.generatedTokenIds.append(tokenID)"))
        #expect(source.contains("slot.cachePromptTokenIds = effectivePromptTokens"))
        #expect(source.contains("let promptTokens = slot.cachePromptTokenIds"))
        #expect(source.contains(#"label: "post-answer""#))
        #expect(source.contains("promptTokens + slot.generatedTokenIds"))
        #expect(source.contains("slot.originalInput.cacheHitSuffixContainsMediaPlaceholder(remaining)"))
        #expect(source.contains("let unsafeFullHit = remaining.isEmpty && hasPathDependentLayer"))
        #expect(source.contains("!slot.originalInput.requiresPostPrepareCacheKey"))
        #expect(source.contains("layer is MambaCache || layer is ArraysCache || layer is ZayaCCACache"))
        #expect(source.contains("shouldSkipHistoryBoundaryRederiveAfterTrimMiss(promptCacheSnapshot)"))
        #expect(source.contains("Skipped history-boundary cache rederive after trim miss for slot"))
        #expect(!source.contains("let unsafePartial = !remaining.isEmpty &&\n                        (hasMediaContent || hasSSMLayer)"))
    }

    @Test("token iterator mirrors post-answer cache boundary policy")
    func tokenIteratorStoresPostAnswerBoundaryForGrowingChat() throws {
        let source = try String(
            contentsOfFile: "Libraries/MLXLMCommon/Evaluate.swift",
            encoding: .utf8)

        #expect(source.contains("mutating func storeCacheAfterGeneration"))
        #expect(source.contains("generatedTokenIds.append(token)"))
        #expect(source.contains("promptTokenIds = effectivePromptTokens"))
        #expect(source.contains("!input.requiresPostPrepareCacheKey"))
        #expect(source.contains("!originalInput.requiresPostPrepareCacheKey"))
        #expect(source.contains("let generatedBoundaryTokens = promptTokenIds + generatedTokenIds"))
        #expect(source.contains("includeGeneratedBoundary: stopReason == .stop && !handler.stopSequenceHit"))
        #expect(source.contains("input.cacheHitSuffixContainsMediaPlaceholder(remainingTokens)"))
        #expect(source.contains("let unsafeFullHit = remainingTokens.isEmpty && hasPathDependentLayer"))
        #expect(source.contains("layer is MambaCache || layer is ArraysCache || layer is ZayaCCACache"))
        #expect(source.contains("shouldSkipHistoryBoundaryRederiveAfterTrimMiss(promptSnapshot)"))
        #expect(source.contains("TokenIterator: skipped history-boundary cache rederive after trim miss"))
        #expect(!source.contains("let unsafePartial = !remainingTokens.isEmpty &&\n                        (hasMediaContent || hasSSMLayer)"))
    }

    @Test("native MTP iterator skips disk-backed history boundary rederive")
    func nativeMTPIteratorSkipsDiskBackedHistoryBoundaryReDerive() throws {
        let source = try String(
            contentsOfFile: "Libraries/MLXLMCommon/SpecDec/NativeMTPTokenIterator.swift",
            encoding: .utf8)

        #expect(source.contains("shouldSkipHistoryBoundaryRederiveAfterTrimMiss(promptSnapshot)"))
        #expect(source.contains("return nil"))
    }

    @Test("token iterator drains MLX around cache store before completion info")
    func tokenIteratorDrainsMLXAroundCacheStoreBeforeCompletionInfo() throws {
        let source = try String(
            contentsOfFile: "Libraries/MLXLMCommon/Evaluate.swift",
            encoding: .utf8)
        let taskRange = try #require(source.range(of: "private func generateLoopTask"))
        let task = String(source[taskRange.lowerBound...])

        let onEnd = try #require(task.range(of: "handler.onGenerationEnd(emit: continuation.yield)"))
        let store = try #require(task.range(of: "iterator.storeCacheAfterGeneration("))
        let preStoreSync = try #require(task.range(
            of: "Stream().synchronize()",
            range: onEnd.upperBound..<store.lowerBound))
        let postStoreSync = try #require(task.range(
            of: "Stream().synchronize()",
            range: store.upperBound..<task.endIndex))
        let advisorDrain = try #require(task.range(
            of: "MLXPressCanonicalExpertAdvisor.shared.waitUntilIdle()",
            range: postStoreSync.upperBound..<task.endIndex))
        let info = try #require(task.range(
            of: "handler.infoEvent(info)",
            range: advisorDrain.upperBound..<task.endIndex))
        let finish = try #require(task.range(
            of: "continuation.finish()",
            range: info.upperBound..<task.endIndex))

        #expect(onEnd.lowerBound < preStoreSync.lowerBound)
        #expect(preStoreSync.lowerBound < store.lowerBound)
        #expect(store.lowerBound < postStoreSync.lowerBound)
        #expect(postStoreSync.lowerBound < advisorDrain.lowerBound)
        #expect(advisorDrain.lowerBound < info.lowerBound)
        #expect(info.lowerBound < finish.lowerBound)
        #expect(task.range(
            of: "handler.infoEvent(info)",
            range: onEnd.upperBound..<store.lowerBound) == nil)
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

    @Test("history-boundary rederive skips disk-backed cache topologies after trim miss")
    func historyBoundaryRederiveSkipsDiskBackedTopologiesAfterTrimMiss() throws {
        let helpers = try String(
            contentsOfFile: "Libraries/MLXLMCommon/Cache/CacheHelpers.swift",
            encoding: .utf8)
        let evaluate = try String(
            contentsOfFile: "Libraries/MLXLMCommon/Evaluate.swift",
            encoding: .utf8)
        let batch = try String(
            contentsOfFile: "Libraries/MLXLMCommon/BatchEngine/BatchEngine.swift",
            encoding: .utf8)
        let nativeMTP = try String(
            contentsOfFile: "Libraries/MLXLMCommon/SpecDec/NativeMTPTokenIterator.swift",
            encoding: .utf8)

        #expect(helpers.contains("func shouldSkipHistoryBoundaryRederiveAfterTrimMiss"))
        #expect(helpers.contains("cacheRequiresDiskBackedCoordinatorRestore(cache)"))

        for source in [evaluate, batch, nativeMTP] {
            #expect(source.contains("shouldSkipHistoryBoundaryRederiveAfterTrimMiss("))
        }
        #expect(evaluate.contains("history-boundary cache rederive after trim miss"))
        #expect(batch.contains("history-boundary cache rederive after trim miss"))
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
        #expect(disk.contains("public enum MLXCacheIOLock"))
        #expect(disk.contains("withSerializedMLXCacheIO"))
        #expect(disk.contains("MLXDiskCacheIOLock.shared.lock()"))
        #expect(disk.contains("Stream.gpu.synchronize()"))
        #expect(disk.contains("try loadArraysAndMetadata(url: url)"))
        #expect(disk.contains("try save(arrays: arrays, metadata: [\"format\": \"mlx\"], url: url)"))
        #expect(ssm.contains("MLXDiskCacheIOLock.shared.lock()"))
        #expect(ssm.contains("Stream.gpu.synchronize()"))
        #expect(ssm.contains("loadArraysAndMetadata(url: safetensorsURL)"))
        #expect(ssm.contains("try save(arrays: arrays, metadata: [\"format\": \"mlx\"], url: safetensorsURL)"))
    }

    @Test("SSM companion cache serializes in-memory MLX materialization")
    func ssmCompanionCacheSerializesInMemoryMLXMaterialization() throws {
        let source = try String(
            contentsOfFile: "Libraries/MLXLMCommon/Cache/SSMStateCache.swift",
            encoding: .utf8)
        let evaluate = try String(
            contentsOfFile: "Libraries/MLXLMCommon/Evaluate.swift",
            encoding: .utf8)
        let store = try #require(source.range(of: "public func store("))
        let storeSource = String(source[store.lowerBound...])
        let promptTail = try #require(evaluate.range(of: "internal func _decodePromptTail("))
        let promptTailSource = String(evaluate[promptTail.lowerBound...])

        #expect(storeSource.contains("MLXCacheIOLock.withSerializedMLXCacheIO"))
        #expect(storeSource.contains("MLX.eval(materialized)"))
        #expect(storeSource.contains("Stream.gpu.synchronize()"))
        #expect(storeSource.contains("let disk: SSMCompanionDiskStore?"))
        #expect(promptTailSource.contains("input.text.tokenIds"))
        #expect(!promptTailSource.contains("tailArray.asArray"))

        let materialize = try #require(storeSource.range(of: "MLX.eval(materialized)"))
        let lruLock = try #require(storeSource.range(of: "lock.lock()"))
        let diskWrite = try #require(storeSource.range(of: "try? disk.store("))

        #expect(materialize.lowerBound < lruLock.lowerBound)
        #expect(lruLock.lowerBound < diskWrite.lowerBound)
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

    @Test("Laguna stays off compiled decode until parity is proven")
    func lagunaCompiledDecodeIsDenied() throws {
        let evaluate = try String(
            contentsOfFile: "Libraries/MLXLMCommon/Evaluate.swift",
            encoding: .utf8)
        let engine = try String(
            contentsOfFile: "Libraries/MLXLMCommon/BatchEngine/BatchEngine.swift",
            encoding: .utf8)

        #expect(evaluate.contains("typeName.contains(\"laguna\")"))
        #expect(engine.contains("modelName.contains(\"laguna\")"))
        #expect(engine.contains("modelTypeName.contains(\"laguna\")"))
    }
}
