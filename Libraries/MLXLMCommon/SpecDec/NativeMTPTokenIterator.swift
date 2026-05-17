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
    let promptTokenIds: [Int]
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
    private let iteratorStartTime = Date.timeIntervalSinceReferenceDate

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

        self.model = model
        self.cache = cache ?? model.newCache(parameters: effectiveParameters)
        self.mtpCache = model.makeNativeMTPCache()
        self.cacheCoordinator = cacheCoordinator
        self.processor = effectiveParameters.processor()
        self.sampler = effectiveParameters.sampler()
        self.speculativeSampler = SpeculativeSamplingController(parameters: effectiveParameters)
        self.maxTokens = effectiveParameters.maxTokens
        self.depth = Swift.max(1, requestedDepth)
        let promptTokenStart = Date.timeIntervalSinceReferenceDate
        let promptTokenIds = input.text.tokens.reshaped(-1).asArray(Int.self)
        let promptTokenElapsed = Date.timeIntervalSinceReferenceDate - promptTokenStart
        self.promptTokenIds = promptTokenIds
        self.cachePrefixTokenCounts = input.cachePrefixTokenCounts
        self.originalInput = input
        self.cacheInitParameters = effectiveParameters
        self.mediaSalt = computeCacheSalt(for: input, parameters: effectiveParameters)
        self.materializeSyncTime += promptTokenElapsed

        if let coordinator = cacheCoordinator,
           effectiveParameters.kvBits != nil || effectiveParameters.kvMode != .none
        {
            coordinator.setPagedIncompatible(true)
        }

        var inputForPrepare = input
        if let coordinator = cacheCoordinator, !promptTokenIds.isEmpty {
            if !coordinator.isHybrid, cacheContainsPathDependentState(self.cache) {
                coordinator.setHybrid(true)
            }
            if !coordinator.isPagedIncompatible,
               cacheRequiresDiskBackedCoordinatorRestore(self.cache)
            {
                coordinator.setPagedIncompatible(true)
            }
            switch coordinator.fetch(tokens: promptTokenIds, mediaSalt: mediaSalt) {
            case .hit(_, let remainingTokens, _, let blocks, let ssmStates, let diskArrays):
                var restored = false
                if !blocks.isEmpty {
                    let restoredTokens = restoreLayerData(from: blocks, into: self.cache)
                    coordinator.release(blocks: blocks)
                    if restoredTokens > 0 {
                        if let ssm = ssmStates {
                            restoreSSMStates(ssm, into: self.cache)
                        }
                        restored = true
                    }
                }

                if let diskArrays, !restored {
                    let diskRestored = restoreFromDiskArrays(diskArrays, into: &self.cache)
                    if diskRestored > 0 {
                        if let ssm = ssmStates,
                           TQDiskSerializer.formatVersion(of: diskArrays) < 2
                        {
                            restoreSSMStates(ssm, into: self.cache)
                        }
                        MLX.eval(self.cache)
                        restored = true
                    }
                }

                if restored {
                    let hasPathDependentLayer = self.cache.contains { layer in
                        layer is MambaCache || layer is ArraysCache || layer is ZayaCCACache
                    }
                    let unsafePartial =
                        input.cacheHitSuffixContainsMediaPlaceholder(remainingTokens)
                    let unsafeFullHit = remainingTokens.isEmpty && hasPathDependentLayer
                    if unsafePartial || unsafeFullHit {
                        self.cache = model.newCache(parameters: effectiveParameters)
                        inputForPrepare = input
                    } else if remainingTokens.isEmpty, let last = promptTokenIds.last {
                        let promptLen = promptTokenIds.count
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
                break
            }
        }

        let start = Date.timeIntervalSinceReferenceDate
        processor?.prompt(input.text.tokens)
        let prepared = try model.prepare(
            inputForPrepare,
            cache: self.cache,
            windowSize: effectiveParameters.prefillStepSize)
        self.promptPrefillTime = Date.timeIntervalSinceReferenceDate - start
        self.promptCacheSnapshot = makePromptBoundaryCacheSnapshot(from: self.cache)

        let firstToken: MLXArray
        switch prepared {
        case .tokens(let tokens):
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
            depth: self.depth,
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
                try verifyCycle()
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
            func store(tokens: [Int], snapshot: [KVCache], label _: String) {
                guard !tokens.isEmpty else { return }
                let cacheSnapshot = snapshot.map { $0.copy() }
                MLX.eval(cacheSnapshot)
                let requiresDiskBackedRestore =
                    cacheRequiresDiskBackedCoordinatorRestore(cacheSnapshot)
                let perLayerData = requiresDiskBackedRestore
                    ? []
                    : extractLayerData(from: cacheSnapshot)
                let ssmCapture: [MLXArray]? = coordinator.isHybrid &&
                    coordinator.config.enableSSMReDerive &&
                    !requiresDiskBackedRestore
                    ? reDeriveAndStoreSSMStatesForPromptBoundaries(
                        coordinator: coordinator,
                        model: model,
                        promptTokenIds: tokens,
                        mediaSalt: mediaSalt,
                        prefillStepSize: cacheInitParameters.prefillStepSize)
                    : (coordinator.isHybrid ? extractSSMStates(from: cacheSnapshot) : nil)
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

            for boundary in Set(cachePrefixTokenCounts).sorted()
            where boundary > 0 && boundary < promptTokenIds.count {
                let boundaryTokens = Array(promptTokenIds.prefix(boundary))
                if let boundarySnapshot = cacheSnapshotForBoundary(
                    tokens: boundaryTokens,
                    promptSnapshot: promptCacheSnapshot)
                {
                    store(
                        tokens: boundaryTokens,
                        snapshot: boundarySnapshot,
                        label: "history-boundary")
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
        let avgCommitted = verifyCalls > 0
            ? Double(generatedTokenIds.count) / Double(verifyCalls)
            : 0
        let avgAcceptP = acceptanceProbabilityCount > 0
            ? acceptanceProbabilitySum / Double(acceptanceProbabilityCount)
            : 0
        let gdnReplay = NativeMTPGDNReplayDiagnostics.snapshot()
        let phaseSummary = NativeMTPPhaseDiagnostics.summary()
        let iteratorWallTime = Date.timeIntervalSinceReferenceDate - iteratorStartTime
        let verifierMode: String
        if chunkVerifierCount > 0 && sequentialVerifierCount > 0 {
            verifierMode = "mixed"
        } else if chunkVerifierCount > 0 {
            verifierMode = NativeMTPVerifierStatePolicy.mode == .lazyRepair
                ? "chunk_lazy_repair"
                : "chunk_commit"
        } else {
            verifierMode = "sequential_repair"
        }
        let line = String(
            format:
                "[NativeMTP] depth=%d verifyCalls=%d outputTokens=%d acceptedByDepth=%@ bonus=%d rejected=%d residualCorrection=%d prefixCommit=%d rollbackRepair=%d mtpCacheRefresh=%d targetForwards=%d verifyInputTokens=%d repairForwards=%d seedMainForwards=%d verifyMainForwards=%d replayMainForwards=%d mtpForwards=%d avgCommittedPerVerify=%.2f avgAcceptP=%.3f targetVerifySec=%.3f seedMainSec=%.3f verifyMainSec=%.3f replayMainSec=%.3f mtpDraftSec=%.3f samplingSec=%.3f cacheCommitSec=%.3f materializeSyncSec=%.3f cacheStateSec=%.3f iteratorWallSec=%.3f gdnReplayCalls=%d gdnReplayStates=%d gdnReplaySec=%.3f phaseDiag=%@ samplingMode=%@ verifierMode=%@ cacheMode=private-mtp+verifier-prefix-commit\n",
            depth,
            verifyCalls,
            generatedTokenIds.count,
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
        promptSnapshot: [KVCache]
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

        if Self.requiresSequentialVerifierRepair(cache, speculativeSampler: speculativeSampler) {
            try verifyCycleSequential(primary: primary)
            return
        }

        let requested = [primary] + drafts
        let requestedInputIds = recordMaterializeSync {
            requested.map { Int32($0.item(Int.self)) }
        }
        let input = MLXArray(requestedInputIds).reshaped(1, requested.count)
        let replayChunkCommit = Self.requiresChunkTokenReplayRepair(cache)
        let lazyChunkRepair = Self.requiresLazyChunkRepair(cache)
        let canCommitVerifierCache = Self.canCommitVerifierCache(cache)
        let requiresSequentialRepair = Self.requiresSequentialVerifierRepair(
            cache, speculativeSampler: speculativeSampler)
        let checkpointStart = Date.timeIntervalSinceReferenceDate
        let checkpoint =
            (canCommitVerifierCache && !requiresSequentialRepair && !replayChunkCommit
                && !lazyChunkRepair)
            ? nil
            : NativeMTPCacheCheckpoint(cache)
        cacheSnapshotRestoreTime += Date.timeIntervalSinceReferenceDate - checkpointStart
        let verifyStart = Date.timeIntervalSinceReferenceDate
        let verifier = model.nativeBackboneMTPVerifyForward(input, cache: cache)
        MLX.eval(verifier.logits, verifier.hiddenStates)
        let verifyElapsed = Date.timeIntervalSinceReferenceDate - verifyStart
        targetVerifyTime += verifyElapsed
        verifyMainForwardTime += verifyElapsed
        targetForwardCount += 1
        verifyMainForwardCount += 1
        verifyInputTokenCount += requested.count
        chunkVerifierCount += 1

        let sampleStart = Date.timeIntervalSinceReferenceDate
        let verifyDecision = Self.verifyDrafts(
            logits: verifier.logits,
            drafts: drafts,
            draftProbabilities: draftProbabilities,
            sampler: sampler,
            speculativeSampler: speculativeSampler,
            processor: processor)
        materializeSyncTime += verifyDecision.materializeSyncTime
        samplingTime += Date.timeIntervalSinceReferenceDate - sampleStart

        let accepted = verifyDecision.accepted
        var nextVerifiedToken = verifyDecision.nextToken
        var repairedHiddenForNextMTP: MLXArray? = nil
        let shouldReplayAcceptedPrefix = replayChunkCommit
            || (lazyChunkRepair && accepted < drafts.count)
        if shouldReplayAcceptedPrefix {
            guard let checkpoint else {
                throw NativeMTPRuntimeError.verifierCacheCommitFailed
            }
            let restoreStart = Date.timeIntervalSinceReferenceDate
            checkpoint.restore(into: &cache)
            cacheSnapshotRestoreTime += Date.timeIntervalSinceReferenceDate - restoreStart
            let committedInputs = Array(requested.prefix(accepted + 1))
            var repaired: NativeMTPForwardResult?
            for token in committedInputs {
                let replayStart = Date.timeIntervalSinceReferenceDate
                let step = model.nativeBackboneForward(Self.tokenInput(token), cache: cache)
                MLX.eval(step.logits, step.hiddenStates)
                replayMainForwardTime += Date.timeIntervalSinceReferenceDate - replayStart
                repaired = step
                repairForwardCount += 1
                replayMainForwardCount += 1
            }
            if let repaired {
                repairedHiddenForNextMTP = Self.lastHidden(repaired.hiddenStates)
                if speculativeSampler.isGreedy, processor == nil {
                    nextVerifiedToken = sampler.sample(logits: repaired.logits[0..., -1, 0...])
                    recordMaterializeSync {
                        MLX.eval(nextVerifiedToken)
                    }
                }
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
        let draftStart = Date.timeIntervalSinceReferenceDate
        let draftBatch = Self.makeDrafts(
            model: model,
            hidden: hiddenForNextMTP,
            nextToken: nextToken,
            mtpCache: mtpCache,
            depth: depth,
            sampler: sampler,
            speculativeSampler: speculativeSampler,
            processor: processor)
        drafts = draftBatch.tokens
        draftProbabilities = draftBatch.probabilities
        mtpForwardCount += draftBatch.forwardCount
        materializeSyncTime += draftBatch.materializeSyncTime
        mtpDraftTime += Date.timeIntervalSinceReferenceDate - draftStart
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
        let draftStart = Date.timeIntervalSinceReferenceDate
        let draftBatch = Self.makeDrafts(
            model: model,
            hidden: hiddenForNextMTP,
            nextToken: nextToken,
            mtpCache: mtpCache,
            depth: depth,
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
    ) -> VerifyDecision {
        if speculativeSampler.isGreedy {
            var materializeSyncTime: TimeInterval = 0
            let sampled: [MLXArray]
            let sampledIDs: [Int]
            if processor == nil {
                let batch = batchedGreedyTargetTokenIds(
                    logits: logits,
                    count: drafts.count + 1)
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

    private static func nativeMTPHybridVerifySetting() -> String? {
        let env = ProcessInfo.processInfo.environment
        return env["VMLX_NATIVE_MTP_HYBRID_VERIFY"]
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
        speculativeSampler: SpeculativeSamplingController
    ) -> Bool {
        if ProcessInfo.processInfo.environment["VMLX_NATIVE_MTP_FORCE_SEQUENTIAL_REPAIR"] == "1" {
            return true
        }
        if !speculativeSampler.isGreedy && cache.contains(where: { $0 is MambaCache }) {
            return true
        }
        switch nativeMTPHybridVerifySetting()?.lowercased() {
        case "chunk", "chunk_commit", "capture_commit", "fast", "chunk_replay", "chunk_repair",
            "chunk_step_repair", "chunk_lazy_repair", "lazy_repair", "lazy", "fast_lazy":
            return false
        case "sequential", "sequential_repair", "repair":
            return true
        default:
            break
        }
        return cache.contains { $0 is MambaCache }
    }

    private static func requiresChunkTokenReplayRepair(_ cache: [KVCache]) -> Bool {
        switch nativeMTPHybridVerifySetting()?.lowercased() {
        case "chunk_replay", "chunk_repair", "chunk_step_repair":
            return cache.contains { $0 is MambaCache }
        default:
            return false
        }
    }

    private static func requiresLazyChunkRepair(_ cache: [KVCache]) -> Bool {
        NativeMTPVerifierStatePolicy.mode == .lazyRepair
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

    private static func batchedGreedyTargetTokenIds(
        logits: MLXArray,
        count: Int
    ) -> (tokens: [MLXArray], tokenIds: [Int], materializeSyncTime: TimeInterval) {
        let candidateLogits = logits[0..., 0 ..< count, 0...]
        let tokenBatch = argMax(candidateLogits, axis: -1).asType(.int32)
        let syncStart = Date.timeIntervalSinceReferenceDate
        MLX.eval(tokenBatch)
        let tokenIds = tokenBatch.reshaped(-1).asArray(Int32.self).map { Int($0) }
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
