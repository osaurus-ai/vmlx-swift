// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX

enum NativeMTPRuntimeError: Error, CustomStringConvertible {
    case modelDoesNotExposeNativeMTP
    case emptyPrompt
    case unsupportedSampling(String)
    case maxTokensTooSmall
    case verifierProducedNoTokens
    case verifierCacheCommitFailed
    case invalidDepth(Int)

    var description: String {
        switch self {
        case .modelDoesNotExposeNativeMTP:
            "native MTP requested but the loaded model has no active MTP head"
        case .emptyPrompt:
            "native MTP requires a non-empty prompt"
        case .unsupportedSampling(let detail):
            "native MTP sampling is unsupported for this request: \(detail)"
        case .maxTokensTooSmall:
            "native MTP requires maxTokens > 1; use the AR iterator for one-token probes"
        case .verifierProducedNoTokens:
            "native MTP verifier produced no token to emit"
        case .verifierCacheCommitFailed:
            "native MTP verifier could not commit accepted cache prefix"
        case .invalidDepth(let depth):
            "native MTP depth must be at least 1, got \(depth)"
        }
    }
}

private struct NativeMTPCacheCheckpoint {
    let cache: [KVCache]

    init(_ cache: [KVCache]) {
        self.cache = cache.map { $0.copy() }
    }

    func restore(into target: inout [KVCache]) {
        target = cache.map { $0.copy() }
    }
}

struct NativeMTPTokenIterator: TokenIteratorProtocol {
    let model: any NativeMTPModel
    var cache: [KVCache]
    var mtpCache: [KVCache]
    let cacheCoordinator: CacheCoordinator?
    var processor: LogitProcessor?
    let sampler: LogitSampler
    let speculativeSampler: SpeculativeSamplingController
    let maxTokens: Int?
    let depth: Int
    private var currentDepth: Int
    let verifierModeSetting: String?
    var promptTokenIds: [Int]
    let cachePrefixTokenCounts: [Int]
    let originalInput: LMInput
    let cacheInitParameters: GenerateParameters
    var promptCacheSnapshot: [KVCache]?
    let mediaSalt: String?

    var tokenCount = 0
    var promptPrefillTime: TimeInterval = 0

    private var pendingTokens: [Int] = []
    private var pendingIndex = 0
    private var nextMain: MLXArray?
    private var drafts: [MLXArray] = []
    private var draftProbabilities: [MLXArray] = []

    private(set) var verifyCalls = 0
    private(set) var acceptedByDepth: [Int: Int] = [:]
    private(set) var rejectedCount = 0
    private(set) var residualCorrectionCount = 0
    private(set) var bonusCount = 0
    private(set) var prefixCommitCount = 0
    private(set) var rollbackRepairCount = 0
    private(set) var mtpCacheRefreshCount = 0
    private(set) var chunkVerifierCount = 0
    private(set) var sequentialVerifierCount = 0
    private(set) var targetForwardCount = 0
    private(set) var verifyInputTokenCount = 0
    private(set) var repairForwardCount = 0
    private(set) var chunkReplayRepairCount = 0
    private(set) var seedMainForwardCount = 0
    private(set) var verifyMainForwardCount = 0
    private(set) var replayMainForwardCount = 0
    private(set) var mtpForwardCount = 0
    private(set) var autoregressiveFallbackTokenCount = 0
    private(set) var adaptiveDepthDownshiftCount = 0
    private(set) var seedMainForwardTime: TimeInterval = 0
    private(set) var verifyMainForwardTime: TimeInterval = 0
    private(set) var replayMainForwardTime: TimeInterval = 0
    private(set) var targetVerifyTime: TimeInterval = 0
    private(set) var mtpDraftTime: TimeInterval = 0
    private(set) var samplingTime: TimeInterval = 0
    private(set) var cacheCommitTime: TimeInterval = 0
    private(set) var materializeSyncTime: TimeInterval = 0
    private(set) var cacheSnapshotRestoreTime: TimeInterval = 0
    private(set) var acceptanceProbabilitySum = 0.0
    private(set) var acceptanceProbabilityCount = 0
    private var forceAutoregressiveFallback = false
    private var hybridSafetyWarmupComplete = false
    private var adaptiveWindow: [AdaptiveCycle] = []
    private var adaptiveFallbackReason: String?
    private let iteratorStartTime = Date.timeIntervalSinceReferenceDate

    private var usesHybridMambaCache: Bool {
        cache.contains { $0 is MambaCache }
    }

    private struct AdaptiveCycle {
        let depth: Int
        let accepted: Int
    }

    init(
        input: LMInput,
        model: any NativeMTPModel,
        cache: [KVCache]? = nil,
        parameters: GenerateParameters,
        depth requestedDepth: Int,
        cacheCoordinator: CacheCoordinator? = nil
    ) throws {
        guard model.nativeMTPAvailable else {
            throw NativeMTPRuntimeError.modelDoesNotExposeNativeMTP
        }
        if let maxTokens = parameters.maxTokens, maxTokens <= 1 {
            throw NativeMTPRuntimeError.maxTokensTooSmall
        }
        guard requestedDepth >= 1 else {
            throw NativeMTPRuntimeError.invalidDepth(requestedDepth)
        }
        guard input.text.tokens.size > 0 else {
            throw NativeMTPRuntimeError.emptyPrompt
        }
        if NativeMTPGDNReplayDiagnostics.enabled {
            NativeMTPGDNReplayDiagnostics.reset()
        }
        if NativeMTPPhaseDiagnostics.enabled {
            NativeMTPPhaseDiagnostics.reset()
        }

        var effectiveParameters = parameters
        if let coordinator = cacheCoordinator {
            let policy = coordinator.config.resolveKVPolicy(
                kvMode: parameters.kvMode,
                maxKVSize: parameters.maxKVSize,
                promptTokenCount: input.text.tokens.size)
            effectiveParameters.kvMode = policy.kvMode
            effectiveParameters.maxKVSize = policy.maxKVSize
        }
        guard effectiveParameters.canUseNativeMTP(for: input) else {
            throw NativeMTPRuntimeError.unsupportedSampling(
                "native MTP is enabled only for text-only greedy requests with temperature=0, top_p>=1, top_k=0, min_p=0, and no active penalties")
        }

        self.model = model
        self.cache = cache ?? model.newCache(parameters: effectiveParameters)
        self.mtpCache = model.makeNativeMTPCache()
        self.cacheCoordinator = cacheCoordinator
        self.processor = effectiveParameters.processor()
        self.sampler = effectiveParameters.sampler()
        self.speculativeSampler = SpeculativeSamplingController(parameters: effectiveParameters)
        self.maxTokens = effectiveParameters.maxTokens
        self.depth = requestedDepth
        self.currentDepth = requestedDepth
        self.verifierModeSetting = effectiveParameters.draftStrategy?.nativeMTPVerifierMode
        let promptTokenStart = Date.timeIntervalSinceReferenceDate
        let promptTokenIds = input.text.tokens.reshaped(-1).asArray(Int.self)
        let promptTokenElapsed = Date.timeIntervalSinceReferenceDate - promptTokenStart
        self.promptTokenIds = promptTokenIds
        self.cachePrefixTokenCounts = input.cachePrefixTokenCounts
        self.originalInput = input
        self.cacheInitParameters = effectiveParameters
        self.mediaSalt = computeCacheSalt(for: input, parameters: effectiveParameters)
        self.materializeSyncTime += promptTokenElapsed

        let requestsAffineKV: Bool = {
            if effectiveParameters.kvBits != nil { return true }
            if case .affine = effectiveParameters.kvMode { return true }
            return false
        }()
        if let coordinator = cacheCoordinator, requestsAffineKV {
            coordinator.setPagedIncompatible(true)
        }

        var inputForPrepare = input
        var cacheLookupTokenIds = promptTokenIds
        var cacheLookupUsesPostPrepareAlias = false
        if input.requiresPostPrepareCacheKey,
           let effectiveTokens = cacheCoordinator?.resolvePostPrepareCacheKeyAlias(
                rawTokens: promptTokenIds,
                mediaSalt: mediaSalt)
        {
            cacheLookupTokenIds = effectiveTokens
            cacheLookupUsesPostPrepareAlias = true
        }
        if let coordinator = cacheCoordinator,
           !cacheLookupTokenIds.isEmpty,
            (!input.requiresPostPrepareCacheKey || cacheLookupUsesPostPrepareAlias)
        {
            if !coordinator.isHybrid, cacheContainsPathDependentState(self.cache) {
                let topology = ModelCacheTopologySnapshot(cache: self.cache)
                coordinator.setHybrid(
                    true,
                    requiresRecurrentSSMCompanion:
                        topology.requiresRecurrentSSMCompanionState)
            }
            if !coordinator.isPagedIncompatible,
               cacheCannotUsePagedCoordinatorRestore(self.cache)
            {
                if cacheCanUsePagedWithRotatingCompanion(self.cache) {
                    coordinator.setPagedBoundaryCompanionRequired(true)
                } else {
                    coordinator.setPagedIncompatible(true)
                }
            }
            switch coordinator.fetch(
                tokens: cacheLookupTokenIds,
                mediaSalt: mediaSalt,
                preferredDiskBoundaries: originalInput.cacheStablePrefixTokenCounts
            ) {
            case .hit(
                let matchedTokens, let remainingTokens, _, let blocks,
                let ssmStates, let diskArrays):
                var restored = false
                if !blocks.isEmpty {
                    let restoredTokens = restoreLayerData(from: blocks, into: self.cache)
                    coordinator.release(blocks: blocks)
                    if restoredTokens > 0 {
                        if let ssm = ssmStates {
                            restoreSSMStates(
                                ssm, into: self.cache, boundary: matchedTokens)
                        }
                        restored = true
                    }
                }

                if let diskArrays, !restored {
                    let diskRestored = restoreFromDiskArrays(diskArrays, into: &self.cache)
                    if diskRestored > 0 {
                        let cacheHasArraysState = self.cache.contains {
                            String(describing: type(of: $0)).contains("Arrays")
                        }
                        if let ssm = ssmStates,
                           TQDiskSerializer.formatVersion(of: diskArrays) < 2
                            || cacheHasArraysState
                        {
                            restoreSSMStates(
                                ssm, into: self.cache, boundary: matchedTokens)
                        }
                        MLX.eval(self.cache)
                        restored = true
                    }
                }

                if restored {
                    if cacheLookupUsesPostPrepareAlias {
                        self.promptTokenIds = cacheLookupTokenIds
                    }
                    let requiresDiskBackedRestore =
                        cacheRequiresDiskBackedCoordinatorRestore(self.cache)
                    let unsafePartial =
                        input.cacheHitSuffixContainsMediaPlaceholder(remainingTokens)
                    let unsafeFullHit = remainingTokens.isEmpty && requiresDiskBackedRestore
                    if unsafePartial || unsafeFullHit {
                        self.cache = model.newCache(parameters: effectiveParameters)
                        inputForPrepare = input
                    } else if remainingTokens.isEmpty, let last = cacheLookupTokenIds.last {
                        let promptLen = cacheLookupTokenIds.count
                        let cacheOffset = self.cache.first?.offset ?? promptLen
                        let trimNeeded = cacheOffset - (promptLen - 1)
                        if trimNeeded < 0 {
                            self.cache = model.newCache(parameters: effectiveParameters)
                            inputForPrepare = input
                        } else {
                            if trimNeeded > 0 {
                                for layer in self.cache where layer.isTrimmable {
                                    _ = layer.trim(trimNeeded)
                                }
                            }
                            let lastToken = MLXArray([Int32(last)])
                                .expandedDimensions(axis: 0)
                            inputForPrepare = LMInput(text: LMInput.Text(tokens: lastToken))
                        }
                    } else {
                        let remainingArray = MLXArray(remainingTokens.map { Int32($0) })
                            .expandedDimensions(axis: 0)
                        inputForPrepare = LMInput(text: LMInput.Text(tokens: remainingArray))
                    }
                }
            case .miss:
                // Match TokenIterator's correctness invariant: a coordinator
                // miss means this request's token/scope identity did not match
                // any stored prefix. A caller-provided populated cache cannot
                // be safely reused from offsets alone (reasoning/tool/media
                // salts may have changed), so discard it before full prefill.
                if populatedCacheRequiresResetAfterCoordinatorMiss(self.cache) {
                    self.cache = model.newCache(parameters: effectiveParameters)
                    inputForPrepare = input
                }
            }
        }

        let start = Date.timeIntervalSinceReferenceDate
        let prepared = try model.prepare(
            inputForPrepare,
            cache: self.cache,
            windowSize: effectiveParameters.prefillStepSize)
        self.promptPrefillTime = Date.timeIntervalSinceReferenceDate - start
        self.promptCacheSnapshot = makePromptBoundaryCacheSnapshot(from: self.cache)

        let firstToken: MLXArray
        switch prepared {
        case .tokens(let tokens):
            processor?.prompt(input.text.tokens)
            let seedStart = Date.timeIntervalSinceReferenceDate
            let backbone = model.nativeBackboneForward(
                Self.sequenceInput(tokens.tokens),
                cache: self.cache)
            firstToken = Self.sampleLast(
                logits: backbone.logits,
                sampler: sampler,
                speculativeSampler: speculativeSampler,
                processor: &processor)
                .token
            let syncStart = Date.timeIntervalSinceReferenceDate
            MLX.eval(firstToken)
            self.materializeSyncTime += Date.timeIntervalSinceReferenceDate - syncStart
            self.seedMainForwardTime += Date.timeIntervalSinceReferenceDate - seedStart
            self.seedMainForwardCount += 1
        case .logits(let output):
            if let effectivePromptTokens = output.effectivePromptTokens,
               !effectivePromptTokens.isEmpty
            {
                self.promptTokenIds = effectivePromptTokens
                if originalInput.requiresPostPrepareCacheKey {
                    cacheCoordinator?.recordPostPrepareCacheKeyAlias(
                        rawTokens: originalInput.text.tokens.reshaped(-1).asArray(Int.self),
                        effectiveTokens: effectivePromptTokens,
                        mediaSalt: mediaSalt)
                }
                let promptTokens = MLXArray(effectivePromptTokens.map { Int32($0) })
                    .expandedDimensions(axis: 0)
                processor?.prompt(promptTokens)
            } else {
                processor?.prompt(input.text.tokens)
            }
            firstToken = Self.sampleLast(
                logits: output.logits,
                sampler: sampler,
                speculativeSampler: speculativeSampler,
                processor: &processor)
                .token
            let syncStart = Date.timeIntervalSinceReferenceDate
            MLX.eval(firstToken)
            self.materializeSyncTime += Date.timeIntervalSinceReferenceDate - syncStart
        }

        let firstID = recordMaterializeSync {
            firstToken.item(Int.self)
        }
        pendingTokens.append(firstID)

        let bridgeStart = Date.timeIntervalSinceReferenceDate
        let bridge = model.nativeBackboneForward(Self.tokenInput(firstToken), cache: self.cache)
        let secondToken = Self.sampleLast(
            logits: bridge.logits,
            sampler: sampler,
            speculativeSampler: speculativeSampler,
            processor: &processor)
            .token
        let secondSyncStart = Date.timeIntervalSinceReferenceDate
        MLX.eval(secondToken)
        self.materializeSyncTime += Date.timeIntervalSinceReferenceDate - secondSyncStart
        self.seedMainForwardTime += Date.timeIntervalSinceReferenceDate - bridgeStart
        self.seedMainForwardCount += 1

        nextMain = secondToken
        pendingTokens.append(recordMaterializeSync { secondToken.item(Int.self) })
        let draftStart = Date.timeIntervalSinceReferenceDate
        let draftBatch = Self.makeDrafts(
            model: model,
            hidden: Self.lastHidden(bridge.hiddenStates),
            nextToken: secondToken,
            mtpCache: mtpCache,
            depth: self.currentDepth,
            sampler: sampler,
            speculativeSampler: speculativeSampler,
            processor: processor)
        drafts = draftBatch.tokens
        draftProbabilities = draftBatch.probabilities
        mtpForwardCount += draftBatch.forwardCount
        materializeSyncTime += draftBatch.materializeSyncTime
        self.mtpDraftTime += Date.timeIntervalSinceReferenceDate - draftStart
    }

    mutating func next() -> Int? {
        if let maxTokens, tokenCount >= maxTokens {
            return nil
        }

        if pendingIndex >= pendingTokens.count {
            pendingTokens.removeAll(keepingCapacity: true)
            pendingIndex = 0
            do {
                if forceAutoregressiveFallback {
                    try generateAutoregressiveToken()
                } else {
                    try verifyCycle()
                }
            } catch {
                return nil
            }
        }

        guard pendingIndex < pendingTokens.count else { return nil }
        let token = pendingTokens[pendingIndex]
        pendingIndex += 1
        tokenCount += 1
        return token
    }

    mutating func storeCacheAfterGeneration(
        generatedTokenIds: [Int],
        includeGeneratedBoundary: Bool
    ) {
        if let coordinator = cacheCoordinator,
           !promptTokenIds.isEmpty,
           let promptCacheSnapshot
        {
            let sharedPromptStripBoundary: Int? = {
                guard ProcessInfo.processInfo.environment["VMLX_HYBRID_STRIPPED_STORE"] != "0",
                    coordinator.isHybrid,
                    !originalInput.hasMediaContent,
                    let turnStartToken = coordinator.genPromptSuffixTokens.first,
                    let stripAt = promptTokenIds.lastIndex(of: turnStartToken),
                    stripAt > 0,
                    stripAt < promptTokenIds.count - 1
                else { return nil }
                return stripAt
            }()
            var sharedPromptRederivedStates: [Int: [MLXArray]]?
            let sharedPromptAdditionalBoundaries = Array(Set(
                cachePrefixTokenCounts + [sharedPromptStripBoundary].compactMap { $0 }
            ))

            func store(tokens: [Int], snapshot: [KVCache], label _: String) {
                guard !tokens.isEmpty else { return }
                // Same guard as the other store paths: saving a cache materialises
                // it several times over at the memory high-water mark, and a
                // prefix-cache entry is only ever a speed-up for a later request —
                // it must never be able to take the host down. Re-checked per call
                // because this helper runs once per boundary. (MTP was the fourth
                // store path; the first three were guarded and this one was missed.)
                guard CacheStoreBudget.canStore(snapshot) else { return }
                let cacheSnapshot = snapshot.map { $0.copy() }
                MLX.eval(cacheSnapshot)
                let requiresDiskBackedRestore =
                    cacheRequiresDiskBackedCoordinatorRestore(cacheSnapshot)
                let perLayerData = requiresDiskBackedRestore
                    ? []
                    : extractLayerData(from: cacheSnapshot)
                let ssmCapture: [MLXArray]? = {
                    guard coordinator.isHybrid else { return nil }
                    if let exact = exactBoundarySSMStatesFromSnapshotIfSufficient(
                        coordinator: coordinator,
                        snapshot: cacheSnapshot,
                        tokenCount: tokens.count)
                    {
                        return exact
                    }
                    guard coordinator.config.enableSSMReDerive,
                        !originalInput.hasMediaContent
                    else {
                        return extractSSMStates(from: cacheSnapshot)
                    }
                    let isPromptPrefix = tokens.count <= promptTokenIds.count
                        && tokens.elementsEqual(promptTokenIds.prefix(tokens.count))
                    if isPromptPrefix {
                        if sharedPromptRederivedStates == nil {
                            sharedPromptRederivedStates =
                                reDeriveAndStoreSSMStatesAtPromptBoundaries(
                                    coordinator: coordinator,
                                    model: model,
                                    promptTokenIds: promptTokenIds,
                                    mediaSalt: mediaSalt,
                                    additionalBoundaries: sharedPromptAdditionalBoundaries,
                                    persistCapturedStatesToDisk: false,
                                    prefillStepSize: cacheInitParameters.prefillStepSize)
                        }
                        if let shared = sharedPromptRederivedStates?[tokens.count] {
                            return shared
                        }
                    }
                    return reDeriveAndStoreSSMStatesForPromptBoundaries(
                        coordinator: coordinator,
                        model: model,
                        promptTokenIds: tokens,
                        mediaSalt: mediaSalt,
                        persistCapturedStatesToDisk: false,
                        prefillStepSize: cacheInitParameters.prefillStepSize)
                }()
                let diskStoreCache = makeDiskStoreCache(
                    fromPromptBoundary: cacheSnapshot,
                    parameters: cacheInitParameters)
                coordinator.storeAfterGeneration(
                    promptTokens: tokens,
                    perLayerData: perLayerData,
                    ssmStates: ssmCapture,
                    cache: diskStoreCache,
                    mediaSalt: mediaSalt)
            }

            store(
                tokens: promptTokenIds,
                snapshot: promptCacheSnapshot,
                label: "prompt-boundary")

            if !originalInput.requiresPostPrepareCacheKey {
                for boundary in Set(cachePrefixTokenCounts).sorted()
                where boundary > 0 && boundary < promptTokenIds.count {
                    let boundaryTokens = Array(promptTokenIds.prefix(boundary))
                    let isStableBoundary = originalInput
                        .cacheStablePrefixTokenCounts.contains(boundary)
                    if isStableBoundary,
                       coordinator.hasValidatedDiskEntry(
                        tokens: boundaryTokens,
                        mediaSalt: mediaSalt)
                    {
                        continue
                    }
                    if let boundarySnapshot = cacheSnapshotForBoundary(
                        tokens: boundaryTokens,
                        promptSnapshot: promptCacheSnapshot,
                        allowDiskBackedRederive: isStableBoundary)
                    {
                        store(
                            tokens: boundaryTokens,
                            snapshot: boundarySnapshot,
                            label: "history-boundary")
                    }
                }

                // Gen-suffix-stripped cross-turn boundary (hybrid SSM) — same as
                // the solo TokenIterator path. The prompt boundary ends in the
                // chat template's generation-prompt suffix, which the NEXT chat
                // turn replaces with the assistant reply + following user turn, so
                // the full-prompt key never matches as a prefix. The stripped
                // boundary (everything before the final turn-start token) DOES,
                // so store it to enable growing-turn reuse under MTP. Clean SSM
                // comes from `store`'s re-derive (enableSSMReDerive).
                if ProcessInfo.processInfo.environment["VMLX_HYBRID_STRIPPED_STORE"] != "0",
                   coordinator.isHybrid,
                   !originalInput.hasMediaContent,
                   let stripAt = sharedPromptStripBoundary
                {
                    // NOTE: intentionally NOT gated on
                    // `!cachePrefixTokenCounts.contains(stripAt)` — see the solo
                    // TokenIterator path. For hybrid caches the history-boundary
                    // loop can't store this boundary (no allowDiskBackedRederive),
                    // and `stripAt` routinely coincides with a prefix-count entry.
                    let strippedTokens = Array(promptTokenIds.prefix(stripAt))
                    if let strippedSnapshot = cacheSnapshotForBoundary(
                        tokens: strippedTokens,
                        promptSnapshot: promptCacheSnapshot,
                        allowDiskBackedRederive: true)
                    {
                        store(
                            tokens: strippedTokens,
                            snapshot: strippedSnapshot,
                            label: "gen-suffix-stripped")
                    }
                }
            }

            if includeGeneratedBoundary,
               !generatedTokenIds.isEmpty,
               !cache.isEmpty
            {
                let postAnswerTokens = promptTokenIds + generatedTokenIds
                let postAnswerSnapshot = cache.map { $0.copy() }
                let offsets = postAnswerSnapshot.map(\.offset)
                if let offset = offsets.first,
                   offsets.allSatisfy({ $0 == offset })
                {
                    if offset == postAnswerTokens.count {
                        store(
                            tokens: postAnswerTokens,
                            snapshot: postAnswerSnapshot,
                            label: "post-answer")
                    } else if offset > postAnswerTokens.count {
                        let trimCount = offset - postAnswerTokens.count
                        if canTrimPromptCache(postAnswerSnapshot),
                           trimPromptCache(postAnswerSnapshot, numTokens: trimCount) == trimCount
                        {
                            MLX.eval(postAnswerSnapshot)
                            store(
                                tokens: postAnswerTokens,
                                snapshot: postAnswerSnapshot,
                                label: "post-answer")
                        }
                    }
                }
            }
        }

        let accepted = acceptedByDepth
            .sorted { $0.key < $1.key }
            .map { "\($0.key):\($0.value)" }
            .joined(separator: ",")
        let speculativeOutputTokens = Swift.max(
            generatedTokenIds.count - autoregressiveFallbackTokenCount,
            0)
        let avgCommitted = verifyCalls > 0
            ? Double(speculativeOutputTokens) / Double(verifyCalls)
            : 0
        let avgAcceptP = acceptanceProbabilityCount > 0
            ? acceptanceProbabilitySum / Double(acceptanceProbabilityCount)
            : 0
        let gdnReplay = NativeMTPGDNReplayDiagnostics.snapshot()
        let phaseSummary = NativeMTPPhaseDiagnostics.summary()
        let iteratorWallTime = Date.timeIntervalSinceReferenceDate - iteratorStartTime
        let adaptiveFallback = adaptiveFallbackReason ?? "none"
        let verifierMode: String
        if chunkVerifierCount > 0 && sequentialVerifierCount > 0 {
            verifierMode = "mixed"
        } else if chunkVerifierCount > 0 {
            verifierMode = chunkReplayRepairCount > 0
                ? "chunk_repair"
                : NativeMTPVerifierStatePolicy.mode(for: verifierModeSetting) == .lazyRepair
                ? "chunk_lazy_repair"
                : "chunk_commit"
        } else {
            verifierMode = "sequential_repair"
        }
        let line = String(
            format:
                "[NativeMTP] depth=%d activeDepth=%d verifyCalls=%d outputTokens=%d arFallbackTokens=%d acceptedByDepth=%@ bonus=%d rejected=%d residualCorrection=%d prefixCommit=%d rollbackRepair=%d mtpCacheRefresh=%d targetForwards=%d verifyInputTokens=%d repairForwards=%d seedMainForwards=%d verifyMainForwards=%d replayMainForwards=%d mtpForwards=%d avgCommittedPerVerify=%.2f avgAcceptP=%.3f adaptiveDownshifts=%d adaptiveFallback=%@ targetVerifySec=%.3f seedMainSec=%.3f verifyMainSec=%.3f replayMainSec=%.3f mtpDraftSec=%.3f samplingSec=%.3f cacheCommitSec=%.3f materializeSyncSec=%.3f cacheStateSec=%.3f iteratorWallSec=%.3f gdnReplayCalls=%d gdnReplayStates=%d gdnReplaySec=%.3f phaseDiag=%@ samplingMode=%@ verifierMode=%@ cacheMode=private-mtp+verifier-prefix-commit\n",
            depth,
            currentDepth,
            verifyCalls,
            generatedTokenIds.count,
            autoregressiveFallbackTokenCount,
            accepted.isEmpty ? "none" : accepted,
            bonusCount,
            rejectedCount,
            residualCorrectionCount,
            prefixCommitCount,
            rollbackRepairCount,
            mtpCacheRefreshCount,
            targetForwardCount,
            verifyInputTokenCount,
            repairForwardCount,
            seedMainForwardCount,
            verifyMainForwardCount,
            replayMainForwardCount,
            mtpForwardCount,
            avgCommitted,
            avgAcceptP,
            adaptiveDepthDownshiftCount,
            adaptiveFallback,
            targetVerifyTime,
            seedMainForwardTime,
            verifyMainForwardTime,
            replayMainForwardTime,
            mtpDraftTime,
            samplingTime,
            cacheCommitTime,
            materializeSyncTime,
            cacheSnapshotRestoreTime,
            iteratorWallTime,
            gdnReplay.calls,
            gdnReplay.prefixStates,
            gdnReplay.seconds,
            phaseSummary,
            speculativeSampler.isGreedy ? "greedy" : "exact-pq",
            verifierMode)
        FileHandle.standardError.write(Data(line.utf8))
    }

    @inline(__always)
    private mutating func recordMaterializeSync<T>(_ body: () -> T) -> T {
        let start = Date.timeIntervalSinceReferenceDate
        let result = body()
        materializeSyncTime += Date.timeIntervalSinceReferenceDate - start
        return result
    }

    @inline(__always)
    private mutating func recordCacheSnapshotRestore<T>(_ body: () -> T) -> T {
        let start = Date.timeIntervalSinceReferenceDate
        let result = body()
        cacheSnapshotRestoreTime += Date.timeIntervalSinceReferenceDate - start
        return result
    }

    private func cacheSnapshotForBoundary(
        tokens: [Int],
        promptSnapshot: [KVCache],
        allowDiskBackedRederive: Bool = false
    ) -> [KVCache]? {
        guard !tokens.isEmpty, tokens.count < promptTokenIds.count else {
            return nil
        }
        let trimCount = promptTokenIds.count - tokens.count
        let trimmed = promptSnapshot.map { $0.copy() }
        if canTrimPromptCache(trimmed),
           trimPromptCache(trimmed, numTokens: trimCount) == trimCount
        {
            MLX.eval(trimmed)
            return trimmed
        }

        // `allowDiskBackedRederive` bypasses the disk-backed skip-guard for the
        // cross-turn gen-suffix-stripped boundary (the one the next turn reuses).
        // Path-dependent hybrid SSM caches aren't trimmable, so without this the
        // stripped boundary is never stored and growing hybrid turns can't reuse
        // prefill. Mirrors the solo TokenIterator path in Evaluate.swift.
        if !allowDiskBackedRederive,
           shouldSkipHistoryBoundaryRederiveAfterTrimMiss(promptSnapshot) {
            if Self.traceEnabled {
                let line =
                    "[NativeMTPTrace] skipped history-boundary cache rederive after trim miss for disk-backed cache topology\n"
                FileHandle.standardError.write(Data(line.utf8))
            }
            return nil
        }

        if String(describing: Swift.type(of: model)).contains("Gemma3n") {
            return nil
        }

        do {
            let boundaryTokens = MLXArray(tokens.map { Int32($0) })
                .reshaped(1, tokens.count)
            let boundaryInput = LMInput(
                text: LMInput.Text(tokens: boundaryTokens),
                image: originalInput.image,
                video: originalInput.video,
                audio: originalInput.audio,
                mediaTokenIds: originalInput.mediaTokenIds,
                cacheScopeSalt: originalInput.cacheScopeSalt)
            let boundaryCache = model.newCache(parameters: cacheInitParameters)
            switch try model.prepare(
                boundaryInput,
                cache: boundaryCache,
                windowSize: cacheInitParameters.prefillStepSize)
            {
            case .tokens(let remaining):
                _ = model.nativeBackboneForward(
                    Self.sequenceInput(remaining.tokens),
                    cache: boundaryCache)
            case .logits:
                break
            }
            MLX.eval(boundaryCache)
            return boundaryCache
        } catch {
            return nil
        }
    }

    private mutating func verifyCycle() throws {
        guard let primary = nextMain, !drafts.isEmpty else {
            throw NativeMTPRuntimeError.verifierProducedNoTokens
        }

        let shouldUseHybridSafetyWarmup = usesHybridMambaCache
            && speculativeSampler.isGreedy
            && processor == nil
            && !hybridSafetyWarmupComplete

        if shouldUseHybridSafetyWarmup || Self.requiresSequentialVerifierRepair(
            cache,
            speculativeSampler: speculativeSampler,
            verifierMode: verifierModeSetting)
        {
            try verifyCycleSequential(primary: primary)
            return
        }

        let requested = [primary] + drafts
        let requestedInputIds = recordMaterializeSync {
            requested.map { Int32($0.item(Int.self)) }
        }
        let input = MLXArray(requestedInputIds).reshaped(1, requested.count)
        let replayChunkCommit = Self.requiresChunkTokenReplayRepair(
            cache,
            verifierMode: verifierModeSetting)
        let lazyChunkRepair = Self.requiresLazyChunkRepair(
            cache,
            verifierMode: verifierModeSetting)
        let canCommitVerifierCache = Self.canCommitVerifierCache(cache)
        let requiresSequentialRepair = Self.requiresSequentialVerifierRepair(
            cache,
            speculativeSampler: speculativeSampler,
            verifierMode: verifierModeSetting)
        let checkpointStart = Date.timeIntervalSinceReferenceDate
        let needsBatchedVerifierRecovery = speculativeSampler.isGreedy && processor == nil
        let checkpoint =
            (canCommitVerifierCache && !requiresSequentialRepair && !replayChunkCommit
                && !lazyChunkRepair && !needsBatchedVerifierRecovery)
            ? nil
            : NativeMTPCacheCheckpoint(cache)
        cacheSnapshotRestoreTime += Date.timeIntervalSinceReferenceDate - checkpointStart
        let verifyStart = Date.timeIntervalSinceReferenceDate
        let verifier = NativeMTPVerifierStatePolicy.withVerifierMode(verifierModeSetting) {
            model.nativeBackboneMTPVerifyForward(input, cache: cache)
        }
        let verifyElapsed = Date.timeIntervalSinceReferenceDate - verifyStart
        targetVerifyTime += verifyElapsed
        verifyMainForwardTime += verifyElapsed
        targetForwardCount += 1
        verifyMainForwardCount += 1
        verifyInputTokenCount += requested.count
        chunkVerifierCount += 1

        let sampleStart = Date.timeIntervalSinceReferenceDate
        guard let verifyDecision = Self.verifyDrafts(
            logits: verifier.logits,
            drafts: drafts,
            draftProbabilities: draftProbabilities,
            sampler: sampler,
            speculativeSampler: speculativeSampler,
            processor: processor)
        else {
            if let checkpoint {
                let restoreStart = Date.timeIntervalSinceReferenceDate
                checkpoint.restore(into: &cache)
                cacheSnapshotRestoreTime += Date.timeIntervalSinceReferenceDate - restoreStart
            }
            try verifyCycleSequential(primary: primary)
            return
        }
        materializeSyncTime += verifyDecision.materializeSyncTime
        samplingTime += Date.timeIntervalSinceReferenceDate - sampleStart

        var accepted = verifyDecision.accepted
        var nextVerifiedToken = verifyDecision.nextToken
        var repairedHiddenForNextMTP: MLXArray? = nil
        let shouldReplayAcceptedPrefix = replayChunkCommit
            || (lazyChunkRepair && accepted < drafts.count)
        if shouldReplayAcceptedPrefix {
            guard let checkpoint else {
                throw NativeMTPRuntimeError.verifierCacheCommitFailed
            }

            func restoreCheckpoint() {
                let restoreStart = Date.timeIntervalSinceReferenceDate
                checkpoint.restore(into: &cache)
                cacheSnapshotRestoreTime += Date.timeIntervalSinceReferenceDate - restoreStart
            }

            func replayPrefix(count: Int) -> NativeMTPForwardResult {
                let acceptedInputIds = recordMaterializeSync {
                    requested.prefix(count).map { Int32($0.item(Int.self)) }
                }
                let acceptedInput = MLXArray(acceptedInputIds).reshaped(1, count)
                let replayStart = Date.timeIntervalSinceReferenceDate
                let repaired = model.nativeBackboneForward(acceptedInput, cache: cache)
                MLX.eval(repaired.logits, repaired.hiddenStates)
                replayMainForwardTime += Date.timeIntervalSinceReferenceDate - replayStart
                repairForwardCount += 1
                replayMainForwardCount += 1
                return repaired
            }

            restoreCheckpoint()
            var repaired = replayPrefix(count: accepted + 1)

            if speculativeSampler.isGreedy, processor == nil {
                guard let audited = Self.batchedGreedyTargetTokenIds(
                    logits: repaired.logits,
                    count: accepted + 1)
                else {
                    restoreCheckpoint()
                    try verifyCycleSequential(primary: primary)
                    return
                }
                materializeSyncTime += audited.materializeSyncTime

                var auditedAccepted = 0
                while auditedAccepted < accepted {
                    let draftID = recordMaterializeSync {
                        requested[auditedAccepted + 1].item(Int.self)
                    }
                    guard audited.tokenIds[auditedAccepted] == draftID else { break }
                    auditedAccepted += 1
                }

                if auditedAccepted != accepted {
                    accepted = auditedAccepted
                    restoreCheckpoint()
                    repaired = replayPrefix(count: accepted + 1)
                }

                guard let verified = Self.batchedGreedyTargetTokenIds(
                    logits: repaired.logits,
                    count: accepted + 1)
                else {
                    restoreCheckpoint()
                    try verifyCycleSequential(primary: primary)
                    return
                }
                materializeSyncTime += verified.materializeSyncTime
                nextVerifiedToken = verified.tokens[accepted]
            }

            let repairedHidden =
                repaired.hiddenStates[0..., accepted ..< (accepted + 1), 0...]
            repairedHiddenForNextMTP = repairedHidden
            recordMaterializeSync {
                MLX.eval(nextVerifiedToken, repairedHidden)
            }
            rollbackRepairCount += 1
            if replayChunkCommit {
                chunkReplayRepairCount += 1
            }
        }
        if !speculativeSampler.isGreedy {
            acceptanceProbabilitySum += verifyDecision.acceptanceProbabilitySum
            acceptanceProbabilityCount += verifyDecision.acceptanceProbabilityCount
        }

        verifyCalls += 1
        acceptedByDepth[accepted, default: 0] += 1
        recordAdaptiveCycle(accepted: accepted)
        if Self.traceEnabled {
            let requestedIDs = recordMaterializeSync { requested.map { $0.item(Int.self) } }
            let currentDrafts = drafts
            let draftIDs = recordMaterializeSync { currentDrafts.map { $0.item(Int.self) } }
            let nextID = recordMaterializeSync { nextVerifiedToken.item(Int.self) }
            let line =
                "[NativeMTPTrace] call=\(verifyCalls) emitted=\(tokenCount) requested=\(requestedIDs) drafts=\(draftIDs) target=\(verifyDecision.targetTokenIds) accepted=\(accepted) next=\(nextID)\n"
            FileHandle.standardError.write(Data(line.utf8))
        }

        for token in drafts.prefix(accepted) {
            processor?.didSample(token: token)
            pendingTokens.append(recordMaterializeSync { token.item(Int.self) })
        }

        if requiresSequentialRepair && accepted > 0 {
            guard let checkpoint else {
                throw NativeMTPRuntimeError.verifierCacheCommitFailed
            }
            let restoreStart = Date.timeIntervalSinceReferenceDate
            checkpoint.restore(into: &cache)
            cacheSnapshotRestoreTime += Date.timeIntervalSinceReferenceDate - restoreStart

            let acceptedInputIds = recordMaterializeSync {
                requested.prefix(accepted + 1).map { Int32($0.item(Int.self)) }
            }
            let acceptedInput = MLXArray(acceptedInputIds).reshaped(1, accepted + 1)
            let replayStart = Date.timeIntervalSinceReferenceDate
            let repaired = model.nativeBackboneForward(acceptedInput, cache: cache)
            MLX.eval(repaired.logits, repaired.hiddenStates)
            replayMainForwardTime += Date.timeIntervalSinceReferenceDate - replayStart
            repairForwardCount += 1
            replayMainForwardCount += 1

            var repairedProcessor = processor
            nextVerifiedToken = Self.sampleLast(
                logits: repaired.logits,
                sampler: sampler,
                speculativeSampler: speculativeSampler,
                processor: &repairedProcessor)
                .token
            recordMaterializeSync {
                MLX.eval(nextVerifiedToken)
            }
            repairedHiddenForNextMTP =
                repaired.hiddenStates[0..., accepted ..< (accepted + 1), 0...]
            rollbackRepairCount += 1
        }

        let committedInputCount = accepted + 1
        let commitStart = Date.timeIntervalSinceReferenceDate
        let committedCache = repairedHiddenForNextMTP != nil
            ? true
            : canCommitVerifierCache
            ? Self.commitVerifierCache(
                &cache,
                committedInputCount: committedInputCount,
                totalInputCount: requested.count)
            : false
        cacheCommitTime += Date.timeIntervalSinceReferenceDate - commitStart
        if committedCache {
            prefixCommitCount += 1
        }

        let nextToken: MLXArray
        let hiddenForNextMTP: MLXArray
        if accepted == drafts.count {
            bonusCount += 1
            let bonus = nextVerifiedToken
            processor?.didSample(token: bonus)
            pendingTokens.append(recordMaterializeSync { bonus.item(Int.self) })
            nextToken = bonus
            hiddenForNextMTP = repairedHiddenForNextMTP
                ?? verifier.hiddenStates[0..., drafts.count ..< (drafts.count + 1), 0...]
        } else {
            rejectedCount += 1
            if !speculativeSampler.isGreedy {
                residualCorrectionCount += 1
            }

            let correction = nextVerifiedToken
            processor?.didSample(token: correction)
            pendingTokens.append(recordMaterializeSync { correction.item(Int.self) })
            nextToken = correction

            if let repairedHiddenForNextMTP {
                hiddenForNextMTP = repairedHiddenForNextMTP
            } else if committedCache {
                hiddenForNextMTP =
                    verifier.hiddenStates[0..., accepted ..< (accepted + 1), 0...]
            } else {
                rollbackRepairCount += 1
                guard let checkpoint else {
                    throw NativeMTPRuntimeError.verifierCacheCommitFailed
                }
                let restoreStart = Date.timeIntervalSinceReferenceDate
                checkpoint.restore(into: &cache)
                cacheSnapshotRestoreTime += Date.timeIntervalSinceReferenceDate - restoreStart

                let acceptedInputIds = recordMaterializeSync {
                    requested.prefix(accepted + 1).map { Int32($0.item(Int.self)) }
                }
                let acceptedInput = MLXArray(acceptedInputIds).reshaped(1, accepted + 1)
                let replayStart = Date.timeIntervalSinceReferenceDate
                let repaired = model.nativeBackboneForward(acceptedInput, cache: cache)
                MLX.eval(repaired.logits, repaired.hiddenStates)
                replayMainForwardTime += Date.timeIntervalSinceReferenceDate - replayStart
                repairForwardCount += 1
                replayMainForwardCount += 1
                hiddenForNextMTP =
                    repaired.hiddenStates[0..., accepted ..< (accepted + 1), 0...]
            }

            mtpCache = model.makeNativeMTPCache()
            mtpCacheRefreshCount += 1
        }

        guard !pendingTokens.isEmpty else {
            throw NativeMTPRuntimeError.verifierProducedNoTokens
        }

        nextMain = nextToken
        if forceAutoregressiveFallback {
            drafts.removeAll(keepingCapacity: true)
            draftProbabilities.removeAll(keepingCapacity: true)
            return
        }
        let draftStart = Date.timeIntervalSinceReferenceDate
        let draftBatch = Self.makeDrafts(
            model: model,
            hidden: hiddenForNextMTP,
            nextToken: nextToken,
            mtpCache: mtpCache,
            depth: currentDepth,
            sampler: sampler,
            speculativeSampler: speculativeSampler,
            processor: processor)
        drafts = draftBatch.tokens
        draftProbabilities = draftBatch.probabilities
        mtpForwardCount += draftBatch.forwardCount
        materializeSyncTime += draftBatch.materializeSyncTime
        mtpDraftTime += Date.timeIntervalSinceReferenceDate - draftStart
    }

    private mutating func recordAdaptiveCycle(accepted: Int) {
        adaptiveWindow.append(AdaptiveCycle(depth: currentDepth, accepted: accepted))
        if adaptiveWindow.count > Self.adaptiveWindowSize {
            adaptiveWindow.removeFirst(adaptiveWindow.count - Self.adaptiveWindowSize)
        }

        if usesHybridMambaCache,
           speculativeSampler.isGreedy,
           processor == nil,
           !hybridSafetyWarmupComplete,
           verifyCalls >= Self.hybridWarmupCycleCount
        {
            let acceptedTokens = acceptedByDepth.reduce(0) { partial, item in
                partial + item.key * item.value
            }
            let averageAccepted = Double(acceptedTokens) / Double(Swift.max(verifyCalls, 1))
            if averageAccepted >= Self.hybridWarmupMinimumAverageAccepted {
                hybridSafetyWarmupComplete = true
            } else {
                enableAutoregressiveFallback(
                    reason: String(
                        format: "hybrid_warmup_avg_accept=%.2f",
                        averageAccepted))
                return
            }
        }

        guard adaptiveWindow.count >= Self.adaptiveWindowSize,
              !forceAutoregressiveFallback
        else { return }

        let activeSamples = adaptiveWindow.filter { $0.depth == currentDepth }
        guard activeSamples.count >= Self.adaptiveMinimumSamplesPerDepth else { return }

        let acceptedTokens = activeSamples.reduce(0) { $0 + $1.accepted }
        let possibleDraftTokens = activeSamples.reduce(0) { $0 + $1.depth }
        guard possibleDraftTokens > 0 else { return }

        let acceptanceRatio = Double(acceptedTokens) / Double(possibleDraftTokens)
        if currentDepth >= 3, acceptanceRatio < Self.depthThreeMinimumAcceptanceRatio {
            currentDepth = 2
            adaptiveDepthDownshiftCount += 1
            adaptiveWindow.removeAll(keepingCapacity: true)
            mtpCache = model.makeNativeMTPCache()
            mtpCacheRefreshCount += 1
            return
        }

        if currentDepth <= 2, acceptanceRatio < Self.depthTwoMinimumAcceptanceRatio {
            enableAutoregressiveFallback(
                reason: String(
                    format: "adaptive_accept_ratio=%.2f_depth=%d",
                    acceptanceRatio,
                    currentDepth))
        }
    }

    private mutating func enableAutoregressiveFallback(reason: String) {
        forceAutoregressiveFallback = true
        adaptiveFallbackReason = reason
        drafts.removeAll(keepingCapacity: true)
        draftProbabilities.removeAll(keepingCapacity: true)
        mtpCache = model.makeNativeMTPCache()
        mtpCacheRefreshCount += 1
    }

    private mutating func generateAutoregressiveToken() throws {
        guard let primary = nextMain else {
            throw NativeMTPRuntimeError.verifierProducedNoTokens
        }

        let verifyStart = Date.timeIntervalSinceReferenceDate
        let output = model.nativeBackboneForward(Self.tokenInput(primary), cache: cache)
        MLX.eval(output.logits, output.hiddenStates)
        let elapsed = Date.timeIntervalSinceReferenceDate - verifyStart
        targetVerifyTime += elapsed
        verifyMainForwardTime += elapsed
        targetForwardCount += 1
        verifyMainForwardCount += 1
        verifyInputTokenCount += 1

        let sampleStart = Date.timeIntervalSinceReferenceDate
        let sample = Self.sampleLast(
            logits: output.logits,
            sampler: sampler,
            speculativeSampler: speculativeSampler,
            processor: &processor)
        recordMaterializeSync {
            MLX.eval(sample.token)
        }
        samplingTime += Date.timeIntervalSinceReferenceDate - sampleStart

        let tokenID = recordMaterializeSync { sample.token.item(Int.self) }
        pendingTokens.append(tokenID)
        autoregressiveFallbackTokenCount += 1
        nextMain = sample.token
    }

    private mutating func verifyCycleSequential(primary: MLXArray) throws {
        let requested = [primary] + drafts
        var accepted = 0
        var currentInput = primary
        var nextToken: MLXArray?
        var hiddenForNextMTP: MLXArray?
        var targetTokenIds: [Int] = []
        targetTokenIds.reserveCapacity(drafts.count + 1)

        verifyCalls += 1
        sequentialVerifierCount += 1

        for index in 0 ... drafts.count {
            let verifyStart = Date.timeIntervalSinceReferenceDate
            let verifier = model.nativeBackboneForward(Self.tokenInput(currentInput), cache: cache)
            MLX.eval(verifier.logits, verifier.hiddenStates)
            let verifyElapsed = Date.timeIntervalSinceReferenceDate - verifyStart
            targetVerifyTime += verifyElapsed
            verifyMainForwardTime += verifyElapsed
            targetForwardCount += 1
            verifyMainForwardCount += 1
            verifyInputTokenCount += 1

            hiddenForNextMTP = Self.lastHidden(verifier.hiddenStates)

            if speculativeSampler.isGreedy {
                let sampleStart = Date.timeIntervalSinceReferenceDate
                let sample = Self.sampleLast(
                    logits: verifier.logits,
                    sampler: sampler,
                    speculativeSampler: speculativeSampler,
                    processor: &processor)
                recordMaterializeSync {
                    MLX.eval(sample.token)
                }
                samplingTime += Date.timeIntervalSinceReferenceDate - sampleStart

                let targetID = recordMaterializeSync { sample.token.item(Int.self) }
                targetTokenIds.append(targetID)

                let currentDraft = index < drafts.count ? drafts[index] : nil
                if let currentDraft,
                   targetID == recordMaterializeSync({ currentDraft.item(Int.self) })
                {
                    accepted += 1
                    pendingTokens.append(targetID)
                    currentInput = currentDraft
                    continue
                }

                nextToken = sample.token
                pendingTokens.append(targetID)
                if index == drafts.count {
                    bonusCount += 1
                } else {
                    rejectedCount += 1
                    mtpCache = model.makeNativeMTPCache()
                    mtpCacheRefreshCount += 1
                }
                break
            }

            let sampleStart = Date.timeIntervalSinceReferenceDate
            let probabilities = Self.processedProbabilities(
                logits: verifier.logits[0..., -1, 0...],
                speculativeSampler: speculativeSampler,
                processor: &processor)
            recordMaterializeSync {
                MLX.eval(probabilities)
            }
            samplingTime += Date.timeIntervalSinceReferenceDate - sampleStart

            if index < drafts.count {
                let decision = speculativeSampler.acceptOrCorrect(
                    draftToken: drafts[index],
                    targetProbabilities: probabilities,
                    draftProbabilities: draftProbabilities[index])
                acceptanceProbabilitySum += Double(decision.acceptanceProbability)
                acceptanceProbabilityCount += 1

                if decision.accepted {
                    accepted += 1
                    let acceptedDraft = drafts[index]
                    processor?.didSample(token: acceptedDraft)
                    pendingTokens.append(recordMaterializeSync { acceptedDraft.item(Int.self) })
                    currentInput = acceptedDraft
                    continue
                }

                guard let correction = decision.correction else {
                    preconditionFailure("rejected speculative token must return a residual correction")
                }
                recordMaterializeSync {
                    MLX.eval(correction)
                }
                processor?.didSample(token: correction)
                nextToken = correction
                pendingTokens.append(recordMaterializeSync { correction.item(Int.self) })
                rejectedCount += 1
                residualCorrectionCount += 1
                mtpCache = model.makeNativeMTPCache()
                mtpCacheRefreshCount += 1
                break
            }

            let bonus = speculativeSampler.sampleFromTarget(probabilities: probabilities)
            recordMaterializeSync {
                MLX.eval(bonus)
            }
            processor?.didSample(token: bonus)
            nextToken = bonus
            pendingTokens.append(recordMaterializeSync { bonus.item(Int.self) })
            bonusCount += 1
            break
        }

        acceptedByDepth[accepted, default: 0] += 1
        recordAdaptiveCycle(accepted: accepted)
        prefixCommitCount += 1

        guard let nextToken, let hiddenForNextMTP else {
            throw NativeMTPRuntimeError.verifierProducedNoTokens
        }

        if Self.traceEnabled {
            let requestedIDs = recordMaterializeSync { requested.map { $0.item(Int.self) } }
            let currentDrafts = drafts
            let draftIDs = recordMaterializeSync { currentDrafts.map { $0.item(Int.self) } }
            let nextID = recordMaterializeSync { nextToken.item(Int.self) }
            let line =
                "[NativeMTPTrace] call=\(verifyCalls) emitted=\(tokenCount) requested=\(requestedIDs) drafts=\(draftIDs) target=\(targetTokenIds) accepted=\(accepted) next=\(nextID) sequential=1\n"
            FileHandle.standardError.write(Data(line.utf8))
        }

        nextMain = nextToken
        if forceAutoregressiveFallback {
            drafts.removeAll(keepingCapacity: true)
            draftProbabilities.removeAll(keepingCapacity: true)
            return
        }
        let draftStart = Date.timeIntervalSinceReferenceDate
        let draftBatch = Self.makeDrafts(
            model: model,
            hidden: hiddenForNextMTP,
            nextToken: nextToken,
            mtpCache: mtpCache,
            depth: currentDepth,
            sampler: sampler,
            speculativeSampler: speculativeSampler,
            processor: processor)
        drafts = draftBatch.tokens
        draftProbabilities = draftBatch.probabilities
        mtpForwardCount += draftBatch.forwardCount
        materializeSyncTime += draftBatch.materializeSyncTime
        mtpDraftTime += Date.timeIntervalSinceReferenceDate - draftStart
    }

    private struct VerifyDecision {
        let accepted: Int
        let nextToken: MLXArray
        let targetTokenIds: [Int]
        let acceptanceProbabilitySum: Double
        let acceptanceProbabilityCount: Int
        let materializeSyncTime: TimeInterval
    }

    private struct DraftBatch {
        let tokens: [MLXArray]
        let probabilities: [MLXArray]
        let forwardCount: Int
        let materializeSyncTime: TimeInterval
    }

    private static func verifyDrafts(
        logits: MLXArray,
        drafts: [MLXArray],
        draftProbabilities: [MLXArray],
        sampler: LogitSampler,
        speculativeSampler: SpeculativeSamplingController,
        processor: LogitProcessor?
    ) -> VerifyDecision? {
        if speculativeSampler.isGreedy {
            var materializeSyncTime: TimeInterval = 0
            let sampled: [MLXArray]
            let sampledIDs: [Int]
            if processor == nil {
                guard let batch = batchedGreedyTargetTokenIds(
                    logits: logits,
                    count: drafts.count + 1)
                else {
                    return nil
                }
                sampled = batch.tokens
                sampledIDs = batch.tokenIds
                materializeSyncTime += batch.materializeSyncTime
            } else {
                var tokenRows: [MLXArray] = []
                tokenRows.reserveCapacity(drafts.count + 1)
                var tokenIDs: [Int] = []
                tokenIDs.reserveCapacity(drafts.count + 1)
                var verifyProcessor = processor
                for index in 0 ... drafts.count {
                    let sample = sampleRow(
                        logits: logits[0..., index, 0...],
                        sampler: sampler,
                        speculativeSampler: speculativeSampler,
                        processor: &verifyProcessor)
                    let syncStart = Date.timeIntervalSinceReferenceDate
                    MLX.eval(sample.token)
                    tokenRows.append(sample.token)
                    tokenIDs.append(sample.token.item(Int.self))
                    materializeSyncTime += Date.timeIntervalSinceReferenceDate - syncStart
                }
                sampled = tokenRows
                sampledIDs = tokenIDs
            }

            var accepted = 0
            while accepted < drafts.count {
                let targetID = sampledIDs[accepted]
                let syncStart = Date.timeIntervalSinceReferenceDate
                let draftID = drafts[accepted].item(Int.self)
                materializeSyncTime += Date.timeIntervalSinceReferenceDate - syncStart
                guard targetID == draftID else { break }
                accepted += 1
            }

            if ProcessInfo.processInfo.environment["VMLX_NATIVE_MTP_FORCE_REJECT_ALL"] == "1" {
                return VerifyDecision(
                    accepted: 0,
                    nextToken: sampled[0],
                    targetTokenIds: sampledIDs,
                    acceptanceProbabilitySum: 0,
                    acceptanceProbabilityCount: 0,
                    materializeSyncTime: materializeSyncTime)
            }

            return VerifyDecision(
                accepted: accepted,
                nextToken: sampled[accepted],
                targetTokenIds: sampledIDs,
                acceptanceProbabilitySum: 0,
                acceptanceProbabilityCount: 0,
                materializeSyncTime: materializeSyncTime)
        }

        var materializeSyncTime: TimeInterval = 0
        var targetProbabilities: [MLXArray] = []
        targetProbabilities.reserveCapacity(drafts.count + 1)
        var verifyProcessor = processor
        for index in 0 ... drafts.count {
            let probabilities = processedProbabilities(
                logits: logits[0..., index, 0...],
                speculativeSampler: speculativeSampler,
                processor: &verifyProcessor)
            let syncStart = Date.timeIntervalSinceReferenceDate
            MLX.eval(probabilities)
            materializeSyncTime += Date.timeIntervalSinceReferenceDate - syncStart
            targetProbabilities.append(probabilities)
            if index < drafts.count {
                verifyProcessor?.didSample(token: drafts[index])
            }
        }

        var accepted = 0
        var probabilitySum = 0.0
        var probabilityCount = 0
        while accepted < drafts.count {
            let decision = speculativeSampler.acceptOrCorrect(
                draftToken: drafts[accepted],
                targetProbabilities: targetProbabilities[accepted],
                draftProbabilities: draftProbabilities[accepted])
            probabilitySum += Double(decision.acceptanceProbability)
            probabilityCount += 1

            if decision.accepted {
                accepted += 1
                continue
            }

            guard let correction = decision.correction else {
                preconditionFailure("rejected speculative token must return a residual correction")
            }
            let syncStart = Date.timeIntervalSinceReferenceDate
            MLX.eval(correction)
            materializeSyncTime += Date.timeIntervalSinceReferenceDate - syncStart
            return VerifyDecision(
                accepted: accepted,
                nextToken: correction,
                targetTokenIds: [],
                acceptanceProbabilitySum: probabilitySum,
                acceptanceProbabilityCount: probabilityCount,
                materializeSyncTime: materializeSyncTime)
        }

        let bonus = speculativeSampler.sampleFromTarget(probabilities: targetProbabilities[drafts.count])
        let syncStart = Date.timeIntervalSinceReferenceDate
        MLX.eval(bonus)
        materializeSyncTime += Date.timeIntervalSinceReferenceDate - syncStart
        return VerifyDecision(
            accepted: accepted,
            nextToken: bonus,
            targetTokenIds: [],
            acceptanceProbabilitySum: probabilitySum,
            acceptanceProbabilityCount: probabilityCount,
            materializeSyncTime: materializeSyncTime)
    }

    private static var traceEnabled: Bool {
        ProcessInfo.processInfo.environment["VMLX_NATIVE_MTP_TRACE"] == "1"
    }

    private static let adaptiveWindowSize = 12
    private static let adaptiveMinimumSamplesPerDepth = 6
    private static let depthThreeMinimumAcceptanceRatio = 0.85
    private static let depthTwoMinimumAcceptanceRatio = 0.75
    private static let hybridWarmupCycleCount = 16
    private static let hybridWarmupMinimumAverageAccepted = 2.75

    private static func nativeMTPHybridVerifySetting(_ verifierMode: String? = nil) -> String? {
        let env = ProcessInfo.processInfo.environment
        return verifierMode
            ?? env["VMLX_NATIVE_MTP_HYBRID_VERIFY"]
            ?? env["VMLINUX_NATIVE_MTP_HYBRID_VERIFY"]
    }

    private static func makeDrafts(
        model: any NativeMTPModel,
        hidden: MLXArray,
        nextToken: MLXArray,
        mtpCache: [KVCache],
        depth: Int,
        sampler: LogitSampler,
        speculativeSampler: SpeculativeSamplingController,
        processor: LogitProcessor?
    ) -> DraftBatch {
        var tokens: [MLXArray] = []
        tokens.reserveCapacity(depth)
        var probabilities: [MLXArray] = []
        probabilities.reserveCapacity(speculativeSampler.isGreedy ? 0 : depth)
        var forwardCount = 0
        var materializeSyncTime: TimeInterval = 0

        var hidden = hidden
        var token = nextToken
        var draftProcessor = processor
        for _ in 0 ..< depth {
            let out = model.nativeMTPForward(
                hiddenStates: hidden,
                nextTokenIds: tokenInput(token),
                cache: mtpCache)
            forwardCount += 1
            let draft = sampleLast(
                logits: out.logits,
                sampler: sampler,
                speculativeSampler: speculativeSampler,
                processor: &draftProcessor)
            let syncStart = Date.timeIntervalSinceReferenceDate
            MLX.eval(draft.token, out.hiddenStates)
            materializeSyncTime += Date.timeIntervalSinceReferenceDate - syncStart
            tokens.append(draft.token)
            if !speculativeSampler.isGreedy {
                probabilities.append(draft.probabilities)
            }
            hidden = lastHidden(out.hiddenStates)
            token = draft.token
        }

        return DraftBatch(
            tokens: tokens,
            probabilities: probabilities,
            forwardCount: forwardCount,
            materializeSyncTime: materializeSyncTime)
    }

    private static func canCommitVerifierCache(_ cache: [KVCache]) -> Bool {
        if ProcessInfo.processInfo.environment["VMLX_NATIVE_MTP_FORCE_ROLLBACK_REPAIR"] == "1" {
            return false
        }
        return cache.allSatisfy { layer in
            layer.isTrimmable || layer is MambaCache
        }
    }

    private static func requiresSequentialVerifierRepair(
        _ cache: [KVCache],
        speculativeSampler: SpeculativeSamplingController,
        verifierMode: String? = nil
    ) -> Bool {
        if ProcessInfo.processInfo.environment["VMLX_NATIVE_MTP_FORCE_SEQUENTIAL_REPAIR"] == "1" {
            return true
        }
        switch nativeMTPHybridVerifySetting(verifierMode)?.lowercased() {
        case "chunk", "chunk_commit", "capture_commit", "fast", "chunk_replay", "chunk_repair",
            "chunk_step_repair", "chunk_lazy_repair", "lazy_repair", "lazy", "fast_lazy":
            return false
        case "sequential", "sequential_repair", "repair":
            return true
        default:
            break
        }
        if !speculativeSampler.isGreedy && cache.contains(where: { $0 is MambaCache }) {
            return true
        }
        return false
    }

    private static func requiresChunkTokenReplayRepair(
        _ cache: [KVCache],
        verifierMode: String? = nil
    ) -> Bool {
        switch nativeMTPHybridVerifySetting(verifierMode)?.lowercased() {
        case "chunk_replay", "chunk_repair", "chunk_step_repair":
            return cache.contains { $0 is MambaCache }
        default:
            return false
        }
    }

    private static func requiresLazyChunkRepair(
        _ cache: [KVCache],
        verifierMode: String? = nil
    ) -> Bool {
        NativeMTPVerifierStatePolicy.mode(for: verifierMode) == .lazyRepair
            && cache.contains { $0 is MambaCache }
    }

    private static func commitVerifierCache(
        _ cache: inout [KVCache],
        committedInputCount: Int,
        totalInputCount: Int
    ) -> Bool {
        let rejectedInputCount = Swift.max(0, totalInputCount - committedInputCount)
        if rejectedInputCount == 0 {
            clearRecordedPrefixes(cache)
            return true
        }

        for layer in cache where !layer.isTrimmable {
            guard let mamba = layer as? MambaCache,
                mamba.commitRecordedPrefix(length: committedInputCount)
            else {
                clearRecordedPrefixes(cache)
                return false
            }
        }

        for layer in cache where layer.isTrimmable {
            _ = layer.trim(rejectedInputCount)
        }
        clearRecordedPrefixes(cache)
        return true
    }

    private static func clearRecordedPrefixes(_ cache: [KVCache]) {
        for layer in cache {
            (layer as? MambaCache)?.clearRecordedPrefixes()
        }
    }

    private static func lastHidden(_ hidden: MLXArray) -> MLXArray {
        let last = hidden.dim(1) - 1
        return hidden[0..., last ..< (last + 1), 0...]
    }

    private static func sampleLast(
        logits: MLXArray,
        sampler: LogitSampler,
        speculativeSampler: SpeculativeSamplingController,
        processor: inout LogitProcessor?
    ) -> SpeculativeSamplingController.Sample {
        sampleRow(
            logits: logits[0..., -1, 0...],
            sampler: sampler,
            speculativeSampler: speculativeSampler,
            processor: &processor)
    }

    private static func sampleRow(
        logits: MLXArray,
        sampler: LogitSampler,
        speculativeSampler: SpeculativeSamplingController,
        processor: inout LogitProcessor?
    ) -> SpeculativeSamplingController.Sample {
        var logits = logits
        if var local = processor {
            logits = local.process(logits: logits)
            let sample = sampleProcessedRow(
                logits: logits,
                sampler: sampler,
                speculativeSampler: speculativeSampler)
            local.didSample(token: sample.token)
            processor = local
            return sample
        }
        return sampleProcessedRow(
            logits: logits,
            sampler: sampler,
            speculativeSampler: speculativeSampler)
    }

    private static func sampleProcessedRow(
        logits: MLXArray,
        sampler: LogitSampler,
        speculativeSampler: SpeculativeSamplingController
    ) -> SpeculativeSamplingController.Sample {
        if speculativeSampler.isGreedy {
            let token = sampler.sample(logits: logits)
            return SpeculativeSamplingController.Sample(
                token: token,
                probabilities: MLXArray.zeros([0]))
        }
        return speculativeSampler.sample(logits: logits)
    }

    static func greedyTargetTokenIdsForTesting(
        logits: MLXArray,
        count: Int
    ) -> [Int]? {
        batchedGreedyTargetTokenIds(logits: logits, count: count)?.tokenIds
    }

    private static func batchedGreedyTargetTokenIds(
        logits: MLXArray,
        count: Int
    ) -> (tokens: [MLXArray], tokenIds: [Int], materializeSyncTime: TimeInterval)? {
        guard count > 0,
              logits.ndim >= 3,
              logits.shape.count >= 2,
              logits.shape[1] >= count
        else {
            return nil
        }

        let candidateLogits = logits[0..., 0 ..< count, 0...]
        let tokenBatch = argMax(candidateLogits, axis: -1).asType(.int32)
        let syncStart = Date.timeIntervalSinceReferenceDate
        MLX.eval(tokenBatch)
        guard tokenBatch.size == count else {
            return nil
        }
        let tokenIds = tokenBatch.reshaped(-1).asArray(Int32.self).map { Int($0) }
        guard tokenIds.count == count else {
            return nil
        }
        let materializeSyncTime = Date.timeIntervalSinceReferenceDate - syncStart
        let tokens = tokenIds.map { MLXArray([Int32($0)]) }
        return (
            tokens: tokens,
            tokenIds: tokenIds,
            materializeSyncTime: materializeSyncTime)
    }

    private static func processedProbabilities(
        logits: MLXArray,
        speculativeSampler: SpeculativeSamplingController,
        processor: inout LogitProcessor?
    ) -> MLXArray {
        var logits = logits
        if let local = processor {
            logits = local.process(logits: logits)
        }
        return speculativeSampler.probabilities(logits: logits)
    }

    private static func tokenInput(_ token: MLXArray) -> MLXArray {
        if token.ndim == 2 { return token }
        return token.reshaped(1, 1)
    }

    private static func sequenceInput(_ tokens: MLXArray) -> MLXArray {
        if tokens.ndim == 2 { return tokens }
        return tokens[.newAxis, 0...]
    }
}
