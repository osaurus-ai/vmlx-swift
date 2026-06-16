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

    @Test("SSM rederive is synchronous prompt-boundary path, not detached helper")
    func ssmRederiveUsesSynchronousPromptBoundaryPath() throws {
        let rederive = try String(
            contentsOfFile: "Libraries/MLXLMCommon/Cache/SSMReDerive.swift",
            encoding: .utf8)
        let evaluate = try String(
            contentsOfFile: "Libraries/MLXLMCommon/Evaluate.swift",
            encoding: .utf8)
        let engine = try String(
            contentsOfFile: "Libraries/MLXLMCommon/BatchEngine/BatchEngine.swift",
            encoding: .utf8)
        let nativeMTP = try String(
            contentsOfFile: "Libraries/MLXLMCommon/SpecDec/NativeMTPTokenIterator.swift",
            encoding: .utf8)

        #expect(rederive.contains("Swift parity: detached async re-derive is not active"))
        #expect(rederive.contains("reDeriveAndStoreSSMStatesForPromptBoundaries"))
        #expect(rederive.contains("captureCleanSSMStateInline"))
        #expect(rederive.contains("boundaryMode=prompt-and-paged-blocks"))
        #expect(!rederive.contains("hybridBlockDiskBoundary"))
        #expect(!rederive.contains("Task.detached"))
        #expect(evaluate.contains("reDeriveAndStoreSSMStatesForPromptBoundaries"))
        #expect(engine.contains("reDeriveAndStoreSSMStatesForPromptBoundaries"))
        #expect(!engine.contains("!requiresDiskBackedRestore &&\n                        !slot.originalInput.hasMediaContent"))
        #expect(!evaluate.contains("!requiresDiskBackedRestore &&\n                !originalInput.hasMediaContent"))
        #expect(!nativeMTP.contains("!requiresDiskBackedRestore &&\n                    !originalInput.hasMediaContent"))
    }

    @Test("hybrid full cache hits use seed-boundary SSM instead of unconditional rollback")
    func hybridFullCacheHitsUseSeedBoundarySSM() throws {
        let evaluate = try String(
            contentsOfFile: "Libraries/MLXLMCommon/Evaluate.swift",
            encoding: .utf8)
        let engine = try String(
            contentsOfFile: "Libraries/MLXLMCommon/BatchEngine/BatchEngine.swift",
            encoding: .utf8)
        let rederive = try String(
            contentsOfFile: "Libraries/MLXLMCommon/Cache/SSMReDerive.swift",
            encoding: .utf8)

        #expect(rederive.contains("boundaries.insert(promptTokenIds.count - 1)"))
        for source in [evaluate, engine] {
            #expect(source.contains("let seedBoundary = promptLen - 1"))
            #expect(source.contains("coordinator.ssmStateCache.fetch("))
            #expect(source.contains("boundary: seedBoundary"))
            #expect(source.contains("restoreSSMStates(seedSSM"))
            #expect(source.contains("missing seed-boundary SSM state"))
            #expect(!source.contains("path-dependent full disk hit: re-feeding last token would double-count recurrent state"))
            #expect(!source.contains("path-dependent full cache hit: re-feeding last token would double-count recurrent state"))
        }
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

    @Test("batch token trace is env-gated diagnostic only")
    func batchTokenTraceIsEnvGatedDiagnosticOnly() throws {
        let engine = try String(
            contentsOfFile: "Libraries/MLXLMCommon/BatchEngine/BatchEngine.swift",
            encoding: .utf8)
        let evaluate = try String(
            contentsOfFile: "Libraries/MLXLMCommon/Evaluate.swift",
            encoding: .utf8)

        #expect(engine.contains("VMLINUX_BATCH_TOKEN_TRACE"))
        #expect(engine.contains("VMLX_BATCH_TOKEN_TRACE"))
        #expect(engine.contains("VMLINUX_BATCH_TOKEN_TRACE_LIMIT"))
        #expect(engine.contains("VMLX_BATCH_TOKEN_TRACE_LIMIT"))
        #expect(engine.contains("VMLINUX_BATCH_TOKEN_TRACE_DECODE"))
        #expect(engine.contains("VMLX_BATCH_TOKEN_TRACE_DECODE"))
        #expect(engine.contains("debugBatchTokenTraceLine"))
        #expect(engine.contains("debugBatchTokenTraceDecodeEnabled"))
        #expect(engine.contains("if debugBatchTokenTraceDecodeEnabled()"))
        #expect(engine.contains("decodedPart = \"\""))
        #expect(engine.contains("token=\\(tokenID)\\(decodedPart)"))
        #expect(engine.contains("path: \"BatchEngine.generate.consume\""))
        #expect(engine.contains("path: \"BatchEngine.stepPrefill\""))
        #expect(engine.contains("path: \"BatchEngine.stepBatchDecode\""))
        #expect(engine.contains("path: \"BatchEngine.stepCompiledDecode\""))
        #expect(evaluate.contains("VMLINUX_BATCH_TOKEN_TRACE"))
        #expect(evaluate.contains("VMLX_BATCH_TOKEN_TRACE"))
        #expect(evaluate.contains("VMLINUX_BATCH_TOKEN_TRACE_DECODE"))
        #expect(evaluate.contains("VMLX_BATCH_TOKEN_TRACE_DECODE"))
        #expect(evaluate.contains("debugGenerateTokenTraceLine"))
        #expect(evaluate.contains("debugGenerateTokenTraceDecodeEnabled"))
        #expect(evaluate.contains("if debugGenerateTokenTraceDecodeEnabled()"))
        #expect(evaluate.contains("decodedPart = \"\""))
        #expect(evaluate.contains("token=\\(tokenID)\\(decodedPart)"))
        #expect(evaluate.contains("path: \"generateLoopTask.token\""))
        #expect(!engine.contains("ReasoningCloseBiasProcessor"))
        #expect(!engine.contains("parametersWithAutomaticReasoningCloseBias"))
        #expect(!evaluate.contains("ReasoningCloseBiasProcessor"))
        #expect(!evaluate.contains("parametersWithAutomaticReasoningCloseBias"))
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
