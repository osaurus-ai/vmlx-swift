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
    var processor: LogitProcessor?
    let sampler: LogitSampler
    let speculativeSampler: SpeculativeSamplingController
    let maxTokens: Int?
    let depth: Int
    let promptTokenIds: [Int]

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
    private(set) var targetVerifyTime: TimeInterval = 0
    private(set) var mtpDraftTime: TimeInterval = 0
    private(set) var samplingTime: TimeInterval = 0
    private(set) var cacheCommitTime: TimeInterval = 0
    private(set) var acceptanceProbabilitySum = 0.0
    private(set) var acceptanceProbabilityCount = 0

    init(
        input: LMInput,
        model: any NativeMTPModel,
        cache: [KVCache]? = nil,
        parameters: GenerateParameters,
        depth requestedDepth: Int
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

        self.model = model
        self.cache = cache ?? model.newCache(parameters: parameters)
        self.mtpCache = model.makeNativeMTPCache()
        self.processor = parameters.processor()
        self.sampler = parameters.sampler()
        self.speculativeSampler = SpeculativeSamplingController(parameters: parameters)
        self.maxTokens = parameters.maxTokens
        self.depth = Swift.max(1, requestedDepth)
        self.promptTokenIds = input.text.tokens.reshaped(-1).asArray(Int.self)

        let start = Date.timeIntervalSinceReferenceDate
        processor?.prompt(input.text.tokens)
        let prepared = try model.prepare(input, cache: self.cache, windowSize: parameters.prefillStepSize)
        self.promptPrefillTime = Date.timeIntervalSinceReferenceDate - start

        let firstToken: MLXArray
        switch prepared {
        case .tokens(let tokens):
            let backbone = model.nativeBackboneForward(
                Self.sequenceInput(tokens.tokens),
                cache: self.cache)
            firstToken = Self.sampleLast(
                logits: backbone.logits,
                sampler: sampler,
                speculativeSampler: speculativeSampler,
                processor: &processor)
                .token
        case .logits(let output):
            firstToken = Self.sampleLast(
                logits: output.logits,
                sampler: sampler,
                speculativeSampler: speculativeSampler,
                processor: &processor)
                .token
        }
        MLX.eval(firstToken)

        let firstID = firstToken.item(Int.self)
        pendingTokens.append(firstID)

        let bridge = model.nativeBackboneForward(Self.tokenInput(firstToken), cache: self.cache)
        let secondToken = Self.sampleLast(
            logits: bridge.logits,
            sampler: sampler,
            speculativeSampler: speculativeSampler,
            processor: &processor)
            .token
        MLX.eval(secondToken)

        nextMain = secondToken
        pendingTokens.append(secondToken.item(Int.self))
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
        includeGeneratedBoundary _: Bool
    ) {
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
        let line = String(
            format:
                "[NativeMTP] depth=%d verifyCalls=%d outputTokens=%d acceptedByDepth=%@ bonus=%d rejected=%d residualCorrection=%d prefixCommit=%d rollbackRepair=%d mtpCacheRefresh=%d avgCommittedPerVerify=%.2f avgAcceptP=%.3f targetVerifySec=%.3f mtpDraftSec=%.3f samplingSec=%.3f cacheCommitSec=%.3f samplingMode=%@ cacheMode=private-mtp+verifier-prefix-commit\n",
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
            avgCommitted,
            avgAcceptP,
            targetVerifyTime,
            mtpDraftTime,
            samplingTime,
            cacheCommitTime,
            speculativeSampler.isGreedy ? "greedy" : "exact-pq")
        FileHandle.standardError.write(Data(line.utf8))
    }

    private mutating func verifyCycle() throws {
        guard let primary = nextMain, !drafts.isEmpty else {
            throw NativeMTPRuntimeError.verifierProducedNoTokens
        }

        let requested = [primary] + drafts
        let input = MLXArray(requested.map { Int32($0.item(Int.self)) }).reshaped(1, requested.count)
        let canCommitVerifierCache = Self.canCommitVerifierCache(cache)
        let checkpoint = canCommitVerifierCache ? nil : NativeMTPCacheCheckpoint(cache)
        let verifyStart = Date.timeIntervalSinceReferenceDate
        let verifier = model.nativeBackboneMTPVerifyForward(input, cache: cache)
        MLX.eval(verifier.logits, verifier.hiddenStates)
        targetVerifyTime += Date.timeIntervalSinceReferenceDate - verifyStart

        let sampleStart = Date.timeIntervalSinceReferenceDate
        let verifyDecision = Self.verifyDrafts(
            logits: verifier.logits,
            drafts: drafts,
            draftProbabilities: draftProbabilities,
            sampler: sampler,
            speculativeSampler: speculativeSampler,
            processor: processor)
        samplingTime += Date.timeIntervalSinceReferenceDate - sampleStart

        let accepted = verifyDecision.accepted
        let nextVerifiedToken = verifyDecision.nextToken
        if !speculativeSampler.isGreedy {
            acceptanceProbabilitySum += verifyDecision.acceptanceProbabilitySum
            acceptanceProbabilityCount += verifyDecision.acceptanceProbabilityCount
        }

        verifyCalls += 1
        acceptedByDepth[accepted, default: 0] += 1

        for token in drafts.prefix(accepted) {
            processor?.didSample(token: token)
            pendingTokens.append(token.item(Int.self))
        }

        let committedInputCount = accepted + 1
        let commitStart = Date.timeIntervalSinceReferenceDate
        let committedCache = Self.commitVerifierCache(
            &cache,
            committedInputCount: committedInputCount,
            totalInputCount: requested.count)
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
            pendingTokens.append(bonus.item(Int.self))
            nextToken = bonus
            hiddenForNextMTP = verifier.hiddenStates[0..., drafts.count ..< (drafts.count + 1), 0...]
        } else {
            rejectedCount += 1
            if !speculativeSampler.isGreedy {
                residualCorrectionCount += 1
            }

            let correction = nextVerifiedToken
            processor?.didSample(token: correction)
            pendingTokens.append(correction.item(Int.self))
            nextToken = correction

            if committedCache {
                hiddenForNextMTP =
                    verifier.hiddenStates[0..., accepted ..< (accepted + 1), 0...]
            } else {
                rollbackRepairCount += 1
                guard let checkpoint else {
                    throw NativeMTPRuntimeError.verifierCacheCommitFailed
                }
                checkpoint.restore(into: &cache)

                let acceptedInput = MLXArray(
                    requested.prefix(accepted + 1).map { Int32($0.item(Int.self)) }
                ).reshaped(1, accepted + 1)
                let repaired = model.nativeBackboneForward(acceptedInput, cache: cache)
                MLX.eval(repaired.logits, repaired.hiddenStates)
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
        mtpDraftTime += Date.timeIntervalSinceReferenceDate - draftStart
    }

    private struct VerifyDecision {
        let accepted: Int
        let nextToken: MLXArray
        let acceptanceProbabilitySum: Double
        let acceptanceProbabilityCount: Int
    }

    private struct DraftBatch {
        let tokens: [MLXArray]
        let probabilities: [MLXArray]
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
            var sampled: [MLXArray] = []
            sampled.reserveCapacity(drafts.count + 1)
            var sampledIDs: [Int] = []
            sampledIDs.reserveCapacity(drafts.count + 1)
            var verifyProcessor = processor
            for index in 0 ... drafts.count {
                let sample = sampleRow(
                    logits: logits[0..., index, 0...],
                    sampler: sampler,
                    speculativeSampler: speculativeSampler,
                    processor: &verifyProcessor)
                MLX.eval(sample.token)
                sampled.append(sample.token)
                sampledIDs.append(sample.token.item(Int.self))
            }

            var accepted = 0
            while accepted < drafts.count {
                let targetID = sampledIDs[accepted]
                let draftID = drafts[accepted].item(Int.self)
                guard targetID == draftID else { break }
                accepted += 1
            }

            return VerifyDecision(
                accepted: accepted,
                nextToken: sampled[accepted],
                acceptanceProbabilitySum: 0,
                acceptanceProbabilityCount: 0)
        }

        var targetProbabilities: [MLXArray] = []
        targetProbabilities.reserveCapacity(drafts.count + 1)
        var verifyProcessor = processor
        for index in 0 ... drafts.count {
            let probabilities = processedProbabilities(
                logits: logits[0..., index, 0...],
                speculativeSampler: speculativeSampler,
                processor: &verifyProcessor)
            MLX.eval(probabilities)
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
            MLX.eval(correction)
            return VerifyDecision(
                accepted: accepted,
                nextToken: correction,
                acceptanceProbabilitySum: probabilitySum,
                acceptanceProbabilityCount: probabilityCount)
        }

        let bonus = speculativeSampler.sampleFromTarget(probabilities: targetProbabilities[drafts.count])
        MLX.eval(bonus)
        return VerifyDecision(
            accepted: accepted,
            nextToken: bonus,
            acceptanceProbabilitySum: probabilitySum,
            acceptanceProbabilityCount: probabilityCount)
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

        var hidden = hidden
        var token = nextToken
        var draftProcessor = processor
        for _ in 0 ..< depth {
            let out = model.nativeMTPForward(
                hiddenStates: hidden,
                nextTokenIds: tokenInput(token),
                cache: mtpCache)
            let draft = sampleLast(
                logits: out.logits,
                sampler: sampler,
                speculativeSampler: speculativeSampler,
                processor: &draftProcessor)
            MLX.eval(draft.token, out.hiddenStates)
            tokens.append(draft.token)
            if !speculativeSampler.isGreedy {
                probabilities.append(draft.probabilities)
            }
            hidden = lastHidden(out.hiddenStates)
            token = draft.token
        }

        return DraftBatch(tokens: tokens, probabilities: probabilities)
    }

    private static func canCommitVerifierCache(_ cache: [KVCache]) -> Bool {
        cache.allSatisfy { layer in
            layer.isTrimmable || layer is MambaCache
        }
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
            return SpeculativeSamplingController.Sample(token: token, probabilities: MLXArray.zeros([0]))
        }
        return speculativeSampler.sample(logits: logits)
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
