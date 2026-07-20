// Copyright © 2026 Osaurus AI. All rights reserved.

import Foundation
@testable import MLXLMCommon
import Testing

@Suite("BatchEngine growing-chat cache source coverage")
struct BatchEngineGrowingChatCacheSourceTests {
    @Test("coordinator miss resets only populated caller-owned caches")
    func coordinatorMissResetRequiresPopulatedCache() {
        let empty = KVCacheSimple()
        #expect(!populatedCacheRequiresResetAfterCoordinatorMiss([empty]))

        let populated = KVCacheSimple()
        populated.offset = 37
        #expect(populatedCacheRequiresResetAfterCoordinatorMiss([empty, populated]))
    }

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
        #expect(scheduler.contains("let disablesGeneratedCacheBoundary: Bool"))
        #expect(scheduler.contains("request.input.toolSchemas?.isEmpty == false"))
        #expect(source.contains("!slot.disablesGeneratedCacheBoundary"))
        #expect(source.contains("slot.originalInput.cacheHitSuffixContainsMediaPlaceholder(remaining)"))
        #expect(source.contains("let requiresDiskBackedRestore ="))
        #expect(source.contains("cacheRequiresDiskBackedCoordinatorRestore(slot.cache)"))
        #expect(!source.contains("!requiresDiskBackedRestore &&\n                        !slot.originalInput.hasMediaContent"))
        let exactSnapshot = try #require(source.range(of: "exactBoundarySSMStatesFromSnapshotIfSufficient("))
        let rederive = try #require(source.range(of: "return reDeriveAndStoreSSMStatesForPromptBoundaries("))
        #expect(exactSnapshot.lowerBound < rederive.lowerBound)
        #expect(source.contains("var sharedPromptRederivedStates"))
        #expect(source.contains("reDeriveAndStoreSSMStatesAtPromptBoundaries("))
        #expect(source.contains("persistCapturedStatesToDisk: false"))
        #expect(source.contains("let unsafeFullHit ="))
        #expect(source.contains("remaining.isEmpty && requiresDiskBackedRestore"))
        #expect(source.contains("let seedBoundary = promptLen - 1"))
        #expect(source.contains("seedSSM, into: slot.cache, boundary: seedBoundary"))
        #expect(source.contains("boundary: seedBoundary"))
        #expect(!source.contains("disk-backed full cache hit: re-feeding last token can corrupt path-dependent or rotating state"))
        #expect(source.contains("!slot.originalInput.requiresPostPrepareCacheKey"))
        #expect(!source.contains("let hasPathDependentLayer = slot.cache.contains"))
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
        #expect(source.contains("&& !handler.emittedToolCall"))
        #expect(source.contains("input.cacheHitSuffixContainsMediaPlaceholder(remainingTokens)"))
        #expect(source.contains("let requiresDiskBackedRestore ="))
        #expect(source.contains("cacheRequiresDiskBackedCoordinatorRestore(self.cache)"))
        #expect(source.contains("let unsafeFullHit ="))
        #expect(source.contains("remainingTokens.isEmpty && requiresDiskBackedRestore"))
        #expect(!source.contains("!requiresDiskBackedRestore,\n                    !originalInput.hasMediaContent"))
        let exactSnapshot = try #require(source.range(of: "exactBoundarySSMStatesFromSnapshotIfSufficient("))
        let rederive = try #require(source.range(of: "return reDeriveAndStoreSSMStatesForPromptBoundaries("))
        #expect(exactSnapshot.lowerBound < rederive.lowerBound)
        #expect(source.contains("var sharedPromptRederivedStates"))
        #expect(source.contains("reDeriveAndStoreSSMStatesAtPromptBoundaries("))
        #expect(source.contains("persistCapturedStatesToDisk: false"))
        #expect(source.contains("let seedBoundary = promptLen - 1"))
        #expect(source.contains("seedSSM, into: self.cache, boundary: seedBoundary"))
        #expect(source.contains("boundary: seedBoundary"))
        #expect(!source.contains("disk-backed full cache hit: re-feeding last token can corrupt path-dependent or rotating state"))
        #expect(source.contains("cacheContainsPathDependentState(self.cache)"))
        #expect(source.contains(
            "Populated-cache coordinator miss: reset unverified cache for full prefill"))
        #expect(!source.contains("Populated-cache miss: full prefix matches cache"))
        #expect(!source.contains("Populated-cache miss: trimmed"))
        #expect(!source.contains("let hasPathDependentLayer = self.cache.contains"))
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
        #expect(source.contains(
            "a coordinator\n                // miss means this request's token/scope identity did not match"))
        #expect(source.contains("self.cache = model.newCache(parameters: effectiveParameters)"))
        #expect(source.contains("inputForPrepare = input"))
        #expect(source.contains("return nil"))
        #expect(!source.contains("!requiresDiskBackedRestore,\n                        !originalInput.hasMediaContent"))
        let exactSnapshot = try #require(source.range(of: "exactBoundarySSMStatesFromSnapshotIfSufficient("))
        let rederive = try #require(source.range(of: "return reDeriveAndStoreSSMStatesForPromptBoundaries("))
        #expect(exactSnapshot.lowerBound < rederive.lowerBound)
        #expect(source.contains("var sharedPromptRederivedStates"))
        #expect(source.contains("reDeriveAndStoreSSMStatesAtPromptBoundaries("))
        #expect(source.contains("persistCapturedStatesToDisk: false"))
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

        let serializedRestore = try #require(source.range(
            of: "let diskRestored = MLXCacheIOLock.withSerializedMLXCacheIO"))
        let restore = try #require(source.range(
            of: "let count = restoreFromDiskArrays(diskArrays, into: &self.cache)",
            range: serializedRestore.lowerBound..<source.endIndex))
        #expect(serializedRestore.lowerBound < restore.lowerBound)
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
        #expect(evaluate.contains("cacheRequiresDiskBackedCoordinatorRestore(self.cache)"))
        #expect(evaluate.contains("cacheHasStandaloneRotatingWindowState(self.cache)"))
        #expect(batch.contains("cacheHasStandaloneRotatingWindowState(slot.cache)"))
        #expect(evaluate.contains("path-dependent full cache hit missing seed-boundary SSM state"))
        #expect(batch.contains("path-dependent full cache hit missing seed-boundary SSM state"))
        #expect(batch.contains("cacheRequiresDiskBackedCoordinatorRestore(slot.cache)"))
        #expect(nativeMTP.contains("cacheRequiresDiskBackedCoordinatorRestore(self.cache)"))
        #expect(helpers.contains("func cacheHasStandaloneRotatingWindowState"))
        #expect(!evaluate.contains("skipped disk-backed fetch for active tool request"))
        #expect(!batch.contains("skipped disk-backed fetch for active tool request"))
        #expect(evaluate.contains("skipExactDiskBoundary: requiresDiskBackedRestore"))
        #expect(batch.contains("skipExactDiskBoundary: requiresDiskBackedRestore"))
        #expect(evaluate.contains(#"tokens: Array(promptTokenIds.dropLast())"#))
        #expect(batch.contains(#"tokens: Array(promptTokens.dropLast())"#))
        #expect(batch.contains(#"label: "disk-backed-safe-prompt-boundary""#))

        for source in [evaluate, batch, nativeMTP] {
            #expect(source.contains("shouldSkipHistoryBoundaryRederiveAfterTrimMiss("))
        }
        #expect(evaluate.contains("history-boundary cache rederive after trim miss"))
        #expect(batch.contains("history-boundary cache rederive after trim miss"))
        #expect(nativeMTP.contains("cacheContainsPathDependentState(self.cache)"))
        #expect(evaluate.contains("cacheCannotUsePagedCoordinatorRestore(self.cache)"))
        #expect(batch.contains("cacheCannotUsePagedCoordinatorRestore(cache)"))
        #expect(nativeMTP.contains("cacheCannotUsePagedCoordinatorRestore(self.cache)"))
        #expect(nativeMTP.contains("cacheHasArraysState"))
        #expect(nativeMTP.contains("boundary: matchedTokens"))
    }

    @Test("token iterator does not blanket-eval disk-backed cache snapshots before store")
    func tokenIteratorDoesNotBlanketEvalDiskBackedSnapshotsBeforeStore() throws {
        let source = try String(
            contentsOfFile: "Libraries/MLXLMCommon/Evaluate.swift",
            encoding: .utf8)
        let storeRange = try #require(source.range(of: "func store(\n            tokens: [Int],"))
        let store = String(source[storeRange.lowerBound...])
        let requiresRange = try #require(store.range(
            of: "let requiresDiskBackedRestore =\n                cacheRequiresDiskBackedCoordinatorRestore(snapshot)"))
        let evalRange = try #require(store.range(of: "if !requiresDiskBackedRestore {\n                MLX.eval(snapshot)\n            }"))
        let perLayerRange = try #require(store.range(of: "let perLayerData = requiresDiskBackedRestore"))

        #expect(requiresRange.lowerBound < evalRange.lowerBound)
        #expect(evalRange.upperBound < perLayerRange.lowerBound)
        #expect(!store.contains("let snapshot = cacheToStore.map { $0.copy() }\n            MLX.eval(snapshot)"))
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

    @Test("load-time stack materialization serializes with cache IO")
    func loadTimeStackMaterializationSerializesWithCacheIO() throws {
        let source = try String(
            contentsOfFile: "Libraries/MLXLMCommon/LoadTimeStacking.swift",
            encoding: .utf8)
        let helper = try #require(source.range(of: "public func loadTimeMaterializedStacked"))
        let helperSource = String(source[helper.lowerBound...])

        #expect(helperSource.contains("MLXCacheIOLock.withSerializedMLXCacheIO"))
        #expect(helperSource.contains("MLX.eval(result)"))
        #expect(helperSource.contains("Stream.gpu.synchronize()"))
        #expect(helperSource.contains("MLX.Memory.clearCache()"))
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

    @Test("MiMo V2 cache topology keeps TurboQuant limited to full-attention KV layers")
    func mimoV2CacheTopologyLimitsTurboQuantToFullAttentionKVLayers() throws {
        let model = try String(
            contentsOfFile: "Libraries/MLXLLM/Models/MiMoV2Flash.swift",
            encoding: .utf8)
        let kvCache = try String(
            contentsOfFile: "Libraries/MLXLMCommon/KVCache.swift",
            encoding: .utf8)

        #expect(model.contains("modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? \"mimo_v2_flash\""))
        #expect(model.contains("public func newCache(parameters: GenerateParameters?) -> [KVCache]"))
        #expect(model.contains("configuration.hybridLayerPattern.indices.map"))
        #expect(model.contains("configuration.isSlidingLayer(layerIndex)"))
        #expect(model.contains("RotatingKVCache(maxSize: configuration.slidingWindowSize)"))
        #expect(model.contains("KVCacheSimple()"))
        #expect(model.contains("if let quantizedKVCache = cache as? QuantizedKVCacheProtocol"))
        #expect(model.contains("precondition(sinks == nil, \"Quantized SDPA does not support attention sinks.\")"))
        #expect(kvCache.contains("Only converts `KVCacheSimple` layers. `RotatingKVCache`, `DeepseekV4Cache`,"))
        #expect(kvCache.contains("if let simpleCache = cache[i] as? KVCacheSimple"))
        #expect(kvCache.contains("TurboQuantKVCache.fromSimpleCache("))
        #expect(kvCache.contains("// RotatingKVCache, DeepseekV4Cache, MambaCache, CacheList,"))
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
