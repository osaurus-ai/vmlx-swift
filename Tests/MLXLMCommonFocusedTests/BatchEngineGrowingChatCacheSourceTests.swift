// Copyright © 2026 Osaurus AI. All rights reserved.

import Foundation
import MLX
import MLXLLM
@testable import MLXLMCommon
import Testing

@Suite("BatchEngine growing-chat cache source coverage")
struct BatchEngineGrowingChatCacheSourceTests {
    @Test("reusable-prefix warmup intent survives input copies and forbids hybrid stripping")
    func reusablePrefixWarmupIntentIsPreserved() throws {
        let source = try String(
            contentsOfFile: "Libraries/MLXLMCommon/LanguageModel.swift",
            encoding: .utf8)

        #expect(source.contains("public enum CachePromptIntent: Sendable, Equatable"))
        #expect(source.contains("case reusablePrefixWarmup"))
        #expect(source.contains("cachePromptIntent: CachePromptIntent = .generation"))
        #expect(source.components(
            separatedBy: "cachePromptIntent: cachePromptIntent").count - 1 == 3)
        #expect(source.contains(
            "guard cachePromptIntent != .reusablePrefixWarmup else { return false }"))
    }

    @Test("fresh required-tool restore policy survives input copies")
    func freshRequiredToolRestorePolicyIsPreserved() {
        let tokenArray = MLXArray([Int32(41), Int32(42), Int32(43)])
            .expandedDimensions(axis: 0)
        let input = LMInput(
            text: LMInput.Text(tokens: tokenArray),
            cacheRestorePolicy: .freshRequiredToolSelection)

        #expect(input.cacheRestorePolicy == .freshRequiredToolSelection)
        #expect(
            input.withToolSchemas(nil).cacheRestorePolicy
                == .freshRequiredToolSelection)
        #expect(
            input.withCacheRestorePolicy(.standard).cacheRestorePolicy
                == .standard)
    }

    @Test("all generation paths reject unsafe exact recurrent warmups and throwaway boundaries")
    func cacheWarmupPersistenceContractCoversAllGenerationPaths() throws {
        let evaluate = try String(
            contentsOfFile: "Libraries/MLXLMCommon/Evaluate.swift",
            encoding: .utf8)
        let batch = try String(
            contentsOfFile: "Libraries/MLXLMCommon/BatchEngine/BatchEngine.swift",
            encoding: .utf8)
        let mtp = try String(
            contentsOfFile: "Libraries/MLXLMCommon/SpecDec/NativeMTPTokenIterator.swift",
            encoding: .utf8)

        // The solo N-1 disk-seed boundary is published for reusable-prefix
        // warmups too (same contract as the batched path): a disk-backed
        // topology's warmup that skips the seed publishes nothing, and the
        // visible send re-prefills the identical prefix from scratch.
        #expect(!evaluate.contains(
            "input.cachePromptIntent != .reusablePrefixWarmup"))
        #expect(evaluate.contains(
            "Reusable-prefix warmups MUST publish this seed"))
        #expect(evaluate.contains(
            "originalInput.cachePromptIntent == .reusablePrefixWarmup"))
        #expect(batch.contains(
            "slot.originalInput.cachePromptIntent == .reusablePrefixWarmup"))
        #expect(mtp.contains(
            "originalInput.cachePromptIntent == .reusablePrefixWarmup"))
        // `ec1bc597` (#249) replaced the MTP iterator's inline
        // `originalInput.canCaptureHybridStripBoundary(...)` with the shared
        // `TokenIterator.hybridStripBoundaryIndex(...)`, which performs that same check
        // internally (`Evaluate.swift:1990`) along with the coordinator/hybrid/strip-index
        // guards. Assert the shared helper so all three paths are pinned to ONE
        // implementation instead of three drifting copies.
        #expect(mtp.contains("TokenIterator.hybridStripBoundaryIndex("))
        #expect(evaluate.contains("input.canCaptureHybridStripBoundary("))
        #expect(evaluate.contains("shouldPersistExactPromptBoundary("))
        #expect(batch.contains("shouldPersistExactPromptBoundary("))
        #expect(mtp.contains("shouldPersistExactPromptBoundary("))
        #expect(mtp.contains(
            "coordinator.requiresRecurrentSSMCompanion && boundary > 1"))

        // Solo and batched paths each gate both their N-1/generated stores;
        // native MTP has no N-1 writer and gates its generated store.
        #expect(evaluate.components(separatedBy: "!isReusablePrefixWarmup").count - 1 == 2)
        #expect(batch.components(separatedBy: "!isReusablePrefixWarmup").count - 1 == 2)
        #expect(mtp.components(separatedBy: "!isReusablePrefixWarmup").count - 1 == 1)
    }

    @Test("exact reusable-prefix warmups remain enabled only for non-recurrent cache topologies")
    func exactWarmupBoundaryRequiresRestorableTopology() {
        #expect(shouldPersistExactPromptBoundary(
            cachePromptIntent: .generation,
            requiresRecurrentSSMCompanion: true))
        #expect(shouldPersistExactPromptBoundary(
            cachePromptIntent: .reusablePrefixWarmup,
            requiresRecurrentSSMCompanion: false))
        #expect(!shouldPersistExactPromptBoundary(
            cachePromptIntent: .reusablePrefixWarmup,
            requiresRecurrentSSMCompanion: true))
    }

    @Test("non-recurrent reusable warmups never replay stable cache boundaries")
    func reusableWarmupStableBoundaryRederivePolicy() {
        #expect(!shouldForceStableBoundaryRederive(
            isStableBoundary: false,
            isReusablePrefixWarmup: false,
            requiresRecurrentSSMCompanion: false))
        #expect(shouldForceStableBoundaryRederive(
            isStableBoundary: true,
            isReusablePrefixWarmup: false,
            requiresRecurrentSSMCompanion: false))
        #expect(!shouldForceStableBoundaryRederive(
            isStableBoundary: true,
            isReusablePrefixWarmup: true,
            requiresRecurrentSSMCompanion: false))
        #expect(shouldForceStableBoundaryRederive(
            isStableBoundary: true,
            isReusablePrefixWarmup: true,
            requiresRecurrentSSMCompanion: true))
    }

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
        #expect(source.contains("shouldSkipHistoryBoundaryRederiveAfterTrimMiss(storageTopologySnapshot)"))
        #expect(source.contains("Skipped history-boundary cache rederive after trim miss for slot"))
        #expect(source.contains("cacheStablePrefixTokenCounts.contains(boundary)"))
        #expect(source.contains("forceRederive: shouldForceStableBoundaryRederive("))
        #expect(source.contains(#""stable-system-tool-boundary""#))
        #expect(source.contains("let storeBoundary = isStableBoundary"))
        #expect(source.contains(#""stable-system-tool-safe-seed""#))
        #expect(source.contains("coordinator.hasValidatedDiskEntry("))
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
        // The post-answer boundary key is aligned with what the cache
        // actually contains: the async decode pipeline forwards the consumed
        // stop token, so the aligned key may extend by that one drained
        // token instead of desynchronizing and losing the store to the
        // boundary-offset guard.
        #expect(source.contains("Self.generatedBoundaryTokensAligned("))
        #expect(source.contains("pendingDrainedTokenId:"))
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
        #expect(source.contains("shouldSkipHistoryBoundaryRederiveAfterTrimMiss(storageSnapshot)"))
        #expect(source.contains("TokenIterator: skipped history-boundary cache rederive after trim miss"))
        #expect(source.contains("cacheStablePrefixTokenCounts.contains(boundary)"))
        // The gate is now bound to a local before use rather than passed as an
        // `allowDiskBackedRederive:` argument; the predicate itself is unchanged.
        #expect(source.contains("shouldForceStableBoundaryRederive("))
        #expect(source.contains("let storeBoundary = isStableBoundary"))
        // `hasValidatedDiskEntry` → `hasDurableDiskEntry` was a deliberate fix, not a rename.
        // `hasValidated…` trusts only entries this process wrote, so after a restart — or on any
        // turn that restored from cache, where prefill never crosses the earlier boundaries — an
        // entry already on disk was rebuilt anyway. That rebuild replays the prefix through the
        // model and is cancellable, so a Stop mid-turn killed it as `rederive-failed …
        // CancellationError()`. Pin the durable form so the process-local check cannot return.
        #expect(source.contains("coordinator.hasDurableDiskEntry("))
        #expect(!source.contains("coordinator.hasValidatedDiskEntry("))
        #expect(!source.contains("let unsafePartial = !remainingTokens.isEmpty &&\n                        (hasMediaContent || hasSSMLayer)"))
    }

    @Test("native MTP iterator skips disk-backed history boundary rederive")
    func nativeMTPIteratorSkipsDiskBackedHistoryBoundaryReDerive() throws {
        let source = try String(
            contentsOfFile: "Libraries/MLXLMCommon/SpecDec/NativeMTPTokenIterator.swift",
            encoding: .utf8)

        #expect(source.contains("shouldSkipHistoryBoundaryRederiveAfterTrimMiss(promptSnapshot)"))
        #expect(source.contains("a coordinator"))
        #expect(source.contains("miss means this request's token/scope identity did not match"))
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
        #expect(source.contains("cacheStablePrefixTokenCounts.contains(boundary)"))
        #expect(source.contains(
            "allowDiskBackedRederive: shouldForceStableBoundaryRederive("))
        #expect(source.contains("coordinator.hasValidatedDiskEntry("))
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

    @Test("solo and batched DSV4 paths capture prompt-minus-one disk seeds during prefill")
    func generationPathsCaptureTypedDiskSeedDuringPrefill() throws {
        let source = try String(
            contentsOfFile: "Libraries/MLXLMCommon/Evaluate.swift",
            encoding: .utf8)
        let batch = try String(
            contentsOfFile: "Libraries/MLXLMCommon/BatchEngine/BatchEngine.swift",
            encoding: .utf8)
        let scheduler = try String(
            contentsOfFile: "Libraries/MLXLMCommon/BatchEngine/BatchScheduler.swift",
            encoding: .utf8)

        #expect(source.contains("static func diskSeedBoundaryIndex("))
        #expect(source.contains("cacheHasStandaloneRotatingWindowState(cache)"))
        #expect(source.contains("cacheRequiresPrefillCapturedDiskSeed(cache)"))
        #expect(source.contains("case diskSeed"))
        #expect(source.contains("diskSeedSnapshot = snapshot"))
        #expect(source.contains("split.head != nil"))
        #expect(source.contains("if diskSeedBoundary == seedTokens.count"))
        #expect(source.contains("seedSnapshot = capturedDiskSeed"))
        #expect(batch.contains("splitPrefillInputBeforeFinalToken("))
        #expect(batch.contains("cacheRequiresPrefillCapturedDiskSeed(slot.cache)"))
        #expect(!batch.contains(
            "slot.originalInput.cachePromptIntent != .reusablePrefixWarmup"))
        #expect(batch.contains(
            "excluding warmups here causes the"))
        #expect(batch.contains("remainingPromptUnits > 1"))
        #expect(batch.contains("let diskSeedSnapshot = makePromptBoundaryCacheSnapshot("))
        #expect(batch.contains(
            "storePrefillCapturedDiskSeed(diskSeedSnapshot, for: slot)"))
        #expect(batch.contains("slot.diskSeedSnapshot = nil"))
        #expect(batch.contains("let capturedDiskSeed = slot.diskSeedSnapshot"))
        #expect(batch.contains("capturedDiskSeed ?? boundarySnapshot("))
        #expect(batch.contains("slot.promptCacheSnapshot = makeRetainedExactPromptSnapshot("))
        #expect(batch.contains("let storageSnapshotTokenCount = promptCacheSnapshot == nil"))
        #expect(batch.contains("tokens.count <= storageSnapshotTokenCount"))
        #expect(batch.contains("shouldSkipHistoryBoundaryRederiveAfterTrimMiss(storageTopologySnapshot)"))
        #expect(batch.contains("liveSlot.diskSeedSnapshot = nil"))
        #expect(batch.contains("liveSlot.promptCacheSnapshot = nil"))
        #expect(source.contains("self.promptCacheSnapshot = makeRetainedExactPromptSnapshot("))
        #expect(source.contains("diskSeedSnapshot = nil"))
        #expect(scheduler.contains("var diskSeedSnapshot: [KVCache]?"))

        let prepare = try #require(source.range(of: "if let capture = prefillBoundaryCapture(of: input)"))
        let capture = try #require(source.range(of: "diskSeedSnapshot = snapshot"))
        let store = try #require(source.range(of: "seedSnapshot = capturedDiskSeed"))
        #expect(prepare.lowerBound < capture.lowerBound)
        #expect(capture.lowerBound < store.lowerBound)

        // Scope the one-snapshot assertion to the real coordinator initializer:
        // seed selection must precede the retained exact-snapshot decision.
        let seedSelection = try #require(source.range(
            of: "self.diskSeedBoundary = Self.diskSeedBoundaryIndex("))
        let retainedSnapshot = try #require(source.range(
            of: "self.promptCacheSnapshot = makeRetainedExactPromptSnapshot(",
            range: seedSelection.lowerBound..<source.endIndex))
        let compiledDecode = try #require(source.range(
            of: "if effectiveParameters.enableCompiledDecode",
            range: retainedSnapshot.lowerBound..<source.endIndex))
        #expect(seedSelection.lowerBound < retainedSnapshot.lowerBound)
        #expect(retainedSnapshot.lowerBound < compiledDecode.lowerBound)

        // A seed-only DSV4 store must still reach stable system/tool prefix
        // handling. The stable loop is a sibling of, not nested inside, the
        // optional exact-prompt store and consumes the topology snapshot.
        let topology = try #require(source.range(
            of: "let storageTopologySnapshot = promptCacheSnapshot ?? capturedDiskSeed"))
        let stableLoop = try #require(source.range(
            of: "for boundary in Set(cachePrefixTokenCounts).sorted()",
            range: topology.lowerBound..<source.endIndex))
        let generatedBoundary = try #require(source.range(
            of: "guard !usesCanonicalHybridBoundary",
            range: stableLoop.lowerBound..<source.endIndex))
        let storageBody = source[topology.lowerBound..<generatedBoundary.lowerBound]
        #expect(storageBody.contains("if let storageTopologySnapshot,"))
        #expect(storageBody.contains("storageSnapshot: storageTopologySnapshot"))
        #expect(topology.lowerBound < stableLoop.lowerBound)
    }

    @Test("batch warm DSV4 replay skips storage without bypassing terminal drain")
    func batchWarmDSV4ReplayPreservesTerminalDrain() throws {
        let source = try String(
            contentsOfFile: "Libraries/MLXLMCommon/BatchEngine/BatchEngine.swift",
            encoding: .utf8)
        let finishStart = try #require(source.range(of: "private func finishSlot("))
        let finishEnd = try #require(source.range(
            of: "\n}\n\n// BatchEngine uses the shared",
            range: finishStart.lowerBound..<source.endIndex))
        let finishBody = String(source[finishStart.lowerBound..<finishEnd.lowerBound])

        #expect(finishBody.contains(
            "if let storageTopologySnapshot = promptCacheSnapshot ?? capturedDiskSeed"))
        #expect(!finishBody.contains(
            "guard let storageTopologySnapshot = promptCacheSnapshot ?? capturedDiskSeed"))
        #expect(finishBody.contains(
            "skipped cache store because no new prompt-boundary snapshot was retained"))
        #expect(finishBody.components(separatedBy: "slot.continuation.finish()").count - 1 == 1)

        let skip = try #require(finishBody.range(of:
            "skipped cache store because no new prompt-boundary snapshot was retained"))
        let drain = try #require(finishBody.range(of: "Stream().synchronize()"))
        let finish = try #require(finishBody.range(of: "slot.continuation.finish()"))
        #expect(skip.lowerBound < drain.lowerBound)
        #expect(drain.lowerBound < finish.lowerBound)
    }

    @Test("only non-recurrent typed hybrid pools require prefill-captured disk seeds")
    func dsv4DiskSeedCapturePolicyIsTopologySpecific() {
        let dsv4 = DeepseekV4Cache(slidingWindow: 16, compressRatio: 4)
        #expect(cacheRequiresPrefillCapturedDiskSeed([dsv4]))
        #expect(!cacheRequiresPrefillCapturedDiskSeed([KVCacheSimple()]))
        #expect(!cacheRequiresPrefillCapturedDiskSeed([MambaCache()]))

        #expect(makeRetainedExactPromptSnapshot(from: [dsv4]) == nil)
        #expect(makeRetainedExactPromptSnapshot(from: [KVCacheSimple()]) != nil)
    }

    @Test("batch DSV4 persists its N-1 seed before decode and releases the duplicate")
    func dsv4DiskSeedDoesNotRemainResidentThroughDecode() throws {
        let source = try String(
            contentsOfFile: "Libraries/MLXLMCommon/BatchEngine/BatchEngine.swift",
            encoding: .utf8)

        #expect(source.contains("private func storePrefillCapturedDiskSeed("))
        #expect(source.contains(
            "storePrefillCapturedDiskSeed(diskSeedSnapshot, for: slot)"))
        #expect(source.contains(
            "label=disk-backed-safe-prompt-boundary-prefill"))
        let immediateStore = try #require(source.range(of:
            "storePrefillCapturedDiskSeed(diskSeedSnapshot, for: slot)"))
        let release = try #require(source.range(
            of: "slot.diskSeedSnapshot = nil",
            range: immediateStore.lowerBound..<source.endIndex))
        let tailPrefill = try #require(source.range(
            of: "return try context.model.prepare(\n                        split.tail",
            range: release.lowerBound..<source.endIndex))
        #expect(immediateStore.lowerBound < release.lowerBound)
        #expect(release.lowerBound < tailPrefill.lowerBound)
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
        #expect(evaluate.contains("let seedTokens = Array(promptTokenIds.dropLast())"))
        #expect(evaluate.contains("tokens: seedTokens"))
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

    @Test("batch text forwarding does not inherit cache-store actor isolation")
    func batchTextForwardingRunsOutsideEngineActor() throws {
        let engine = try String(
            contentsOfFile: "Libraries/MLXLMCommon/BatchEngine/BatchEngine.swift",
            encoding: .utf8)

        #expect(engine.contains("Task.detached {\n            var detokenizer"))
        #expect(engine.contains("continuation.yield(.info(finalInfo))"))
        #expect(engine.contains("await engineRef.recordTurboQuantDiagnostics("))
        let terminalYield = try #require(
            engine.range(of: "continuation.yield(.info(finalInfo))"))
        let actorHop = try #require(
            engine.range(of: "await engineRef.recordTurboQuantDiagnostics("))
        #expect(
            terminalYield.lowerBound < actorHop.lowerBound,
            "terminal info must be externally visible before diagnostics queue behind cache-store actor work"
        )
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
