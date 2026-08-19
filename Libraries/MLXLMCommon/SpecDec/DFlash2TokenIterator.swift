// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// DFlash 2 speculative decode loop.
//
// Port target: `_stream_generate` in z-lab/dflash `dflash/model_mlx.py`,
// adapted to this package's ``TokenIteratorProtocol`` and to hybrid
// targets whose recurrent layers cannot be trimmed.
//
// Shape of one cycle:
//
//   1. Build the block `[anchor, mask, mask, …]` (block_size entries).
//   2. ONE drafter forward emits candidates for every masked position;
//      the selector traces a coherent path through them. That path is
//      `block_size - 1` draft tokens.
//   3. ONE target forward over `[anchor] + drafts` verifies all of them
//      and, as a side effect, produces the hidden states the drafter
//      will condition on next round.
//   4. Accept the longest agreeing prefix, append the target's own token
//      at the divergence, roll the target cache back to the accepted
//      length.
//
// Rollback on a hybrid target is the part the reference solves
// differently. `model_mlx.py` monkey-patches GatedDeltaNet to capture its
// inputs and recomputes the recurrent state over the accepted prefix.
// This package already had to solve the same problem for native MTP, and
// the solution there is better: the recurrent layers RECORD their state
// at each step of the verify forward, so committing the accepted prefix
// is a dictionary lookup rather than a recomputation. That machinery is
// reused verbatim here — see `MambaCache.commitRecordedPrefix`.

import Foundation
import MLX
import MLXRandom

/// Target-model requirements for DFlash 2. A model that satisfies these
/// three protocols can drive the block-diffusion path.
public typealias DFlash2Target = HiddenStateCaptureModel & TokenEmbedderModel

public enum DFlash2RuntimeError: Error, LocalizedError {
    case emptyPrompt
    case maxTokensTooSmall
    case unsupportedSampling(String)
    case targetLacksPrefixCommitRecording
    case drafterTargetMismatch(String)
    case blockSizeTooSmall(Int)

    public var errorDescription: String? {
        switch self {
        case .emptyPrompt:
            "DFlash 2 requires a non-empty prompt"
        case .maxTokensTooSmall:
            "DFlash 2 requires maxTokens > 1; use the plain iterator for one-token probes"
        case .unsupportedSampling(let detail):
            "DFlash 2 cannot serve this request: \(detail)"
        case .targetLacksPrefixCommitRecording:
            "DFlash 2 needs recurrent prefix-state recording to roll back a hybrid target, and this model does not provide it"
        case .drafterTargetMismatch(let detail):
            "DFlash 2 drafter does not match this model: \(detail)"
        case .blockSizeTooSmall(let value):
            "DFlash 2 block size must be at least 2, got \(value)"
        }
    }
}

/// Per-generation counters, mirrored into the generation info so a host
/// can show acceptance without a debug build.
public struct DFlash2GenerationStats: Sendable, Equatable {
    public var blockSize: Int = 0
    public var verifyCalls: Int = 0
    public var draftedTokens: Int = 0
    public var acceptedTokens: Int = 0
    public var emittedTokens: Int = 0
    public var seededContextRows: Int = 0
    public var autoregressiveFallbackTokens: Int = 0
    public var draftSeconds: Double = 0
    public var verifySeconds: Double = 0
    public var commitSeconds: Double = 0

    /// Mean number of tokens committed per target forward. 1.0 means the
    /// drafter contributed nothing; the ceiling is `blockSize`.
    public var acceptanceLength: Double {
        verifyCalls > 0 ? Double(emittedTokens) / Double(verifyCalls) : 0
    }
}

struct DFlash2TokenIterator: TokenIteratorProtocol {

    private let target: any DFlash2Target
    private let drafter: DFlash2DraftModel
    private let captureLayerIDs: Set<Int>
    private let orderedLayerIDs: [Int]

    private var cache: [KVCache]
    private var draftCache: [KVCache]
    private let cacheCoordinator: CacheCoordinator?

    private let sampler: LogitSampler
    private let temperature: Float
    private let topP: Float
    private let topK: Int
    private let isGreedy: Bool
    private let blockSize: Int
    private let maskTokenID: Int

    let maxTokens: Int?
    var tokenCount = 0
    var promptPrefillTime: TimeInterval = 0
    private(set) var promptTokenIds: [Int]
    private let cachePrefixTokenCounts: [Int]
    private let originalInput: LMInput
    private let cacheInitParameters: GenerateParameters
    private let mediaSalt: String?
    private var promptCacheSnapshot: [KVCache]?

    /// Index of the generation-suffix-stripped boundary in
    /// ``promptTokenIds`` — the last turn-start token, i.e. the end of the
    /// prompt with its trailing generation prompt removed. `nil` when it
    /// does not apply (dense target, no turn-start token, tiers disabled).
    ///
    /// For a hybrid target this is the ONLY reusable cross-turn
    /// checkpoint: the next turn's prompt replaces the generation suffix,
    /// so it never contains this turn's full prompt, and hybrid state is
    /// path-dependent, so the full-prompt cache cannot be trimmed back.
    /// Storing only the full prompt therefore stores a boundary no later
    /// turn can match — measured as turn 2 re-prefilling all 14 of 14
    /// tokens in `DFlash2PrefixCacheReuseTests` before this existed.
    private var hybridStripBoundary: Int?

    /// Cache state at ``hybridStripBoundary``, captured DURING prefill as
    /// it passes the boundary — one cache copy instead of a second full
    /// prefill after the answer. Released once stored.
    private var hybridStripSnapshot: [KVCache]?

    /// Target hidden states for the positions the drafter has not yet
    /// consumed — `(1, rows, len(target_layer_ids) * hidden)`. After
    /// prefill this is the tail of the prompt; afterwards it is exactly
    /// the tokens committed by the previous cycle.
    private var contextHidden: MLXArray

    private var pendingTokens: [Int] = []
    private var pendingIndex = 0
    private var lastToken: Int
    private var stats = DFlash2GenerationStats()

    private static let traceEnabled =
        ProcessInfo.processInfo.environment["VMLX_DFLASH2_TRACE"] == "1"

    /// Why this request cannot be served by DFlash 2, or `nil` when it
    /// can.
    ///
    /// Callers are expected to check this BEFORE constructing the
    /// iterator and to fall through to ordinary decoding when it returns
    /// a reason. Failing the request instead would mean a user who
    /// selected a drafter gets an error on a prompt that would otherwise
    /// have worked — the drafter is an optimisation, and an optimisation
    /// that cannot apply should get out of the way.
    ///
    /// What genuinely cannot be served: anything that rewrites logits
    /// per step. A block is drafted before any of those could have seen
    /// the tokens inside it, so accepting under them would emit a
    /// distribution the caller did not ask for.
    static func unservableReason(_ parameters: GenerateParameters) -> String? {
        // A penalty of exactly 1.0 (repetition) or 0 (presence/frequency)
        // is the identity. Bundles ship those as explicit defaults —
        // Qwen3.8 stamps `repetition_penalty: 1.0` — so treating "set" as
        // "active" would disable DFlash 2 for the very models it targets.
        if let repetition = parameters.repetitionPenalty, repetition != 1.0 {
            return "repetition_penalty \(repetition) needs per-step logits"
        }
        if let presence = parameters.presencePenalty, presence != 0 {
            return "presence_penalty \(presence) needs per-step logits"
        }
        if let frequency = parameters.frequencyPenalty, frequency != 0 {
            return "frequency_penalty \(frequency) needs per-step logits"
        }
        if !parameters.suppressTokens.isEmpty || !parameters.initialSuppressTokens.isEmpty
            || parameters.initialSuppressCount > 0
        {
            return "token suppression needs per-step logits"
        }
        if parameters.reasoningBudgetTokens != nil {
            return "a reasoning budget needs per-step logits"
        }
        if parameters.minP > 0 {
            return "min_p is not part of the DFlash 2 acceptance distribution"
        }
        if let maxTokens = parameters.maxTokens, maxTokens <= 1 {
            return "maxTokens \(maxTokens) is too small for a drafted block"
        }
        return nil
    }

    // MARK: - Init

    init(
        input: LMInput,
        target: any DFlash2Target,
        drafter: DFlash2DraftModel,
        blockSize requestedBlockSize: Int?,
        cache: [KVCache]? = nil,
        parameters: GenerateParameters,
        cacheCoordinator: CacheCoordinator? = nil
    ) throws {
        guard input.text.tokens.size > 0 else {
            throw DFlash2RuntimeError.emptyPrompt
        }
        if let maxTokens = parameters.maxTokens, maxTokens <= 1 {
            throw DFlash2RuntimeError.maxTokensTooSmall
        }
        if let reason = Self.unservableReason(parameters) {
            throw DFlash2RuntimeError.unsupportedSampling(reason)
        }

        let config = drafter.config
        let effectiveBlockSize = requestedBlockSize ?? config.blockSize
        guard effectiveBlockSize >= 2 else {
            throw DFlash2RuntimeError.blockSizeTooSmall(effectiveBlockSize)
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

        self.target = target
        self.drafter = drafter
        self.orderedLayerIDs = config.targetLayerIds
        self.captureLayerIDs = Set(config.targetLayerIds)
        self.cache = cache ?? target.newCache(parameters: effectiveParameters)
        self.draftCache = drafter.makeCache()
        self.cacheCoordinator = cacheCoordinator
        self.sampler = effectiveParameters.sampler()
        self.temperature = effectiveParameters.temperature
        self.topP = effectiveParameters.topP
        self.topK = effectiveParameters.topK
        self.isGreedy = effectiveParameters.temperature <= 0
        self.blockSize = effectiveBlockSize
        self.maskTokenID = config.maskTokenId
        self.maxTokens = effectiveParameters.maxTokens
        self.promptTokenIds = input.text.tokens.reshaped(-1).asArray(Int.self)
        self.cachePrefixTokenCounts = input.cachePrefixTokenCounts
        self.originalInput = input
        self.cacheInitParameters = effectiveParameters
        self.mediaSalt = computeCacheSalt(for: input, parameters: effectiveParameters)
        self.contextHidden = MLXArray.zeros([1, 0, 1])
        self.lastToken = 0

        // A hybrid target whose recurrent layers cannot record their
        // per-step state has no way back from a rejected block. Refuse up
        // front rather than corrupting the cache mid-turn.
        let hasUntrimmableState = self.cache.contains { !$0.isTrimmable }
        if hasUntrimmableState, !target.supportsCapturingPrefixCommitRecording {
            throw DFlash2RuntimeError.targetLacksPrefixCommitRecording
        }
        if hasUntrimmableState, self.cache.contains(where: { !($0 is MambaCache) && !$0.isTrimmable })
        {
            throw DFlash2RuntimeError.targetLacksPrefixCommitRecording
        }

        self.stats.blockSize = effectiveBlockSize

        // MARK: prefix reuse

        var tokensToPrefill = self.promptTokenIds
        if let coordinator = cacheCoordinator, !tokensToPrefill.isEmpty,
            !input.requiresPostPrepareCacheKey
        {
            if !coordinator.isHybrid, cacheContainsPathDependentState(self.cache) {
                let topology = ModelCacheTopologySnapshot(cache: self.cache)
                coordinator.setHybrid(
                    true,
                    requiresRecurrentSSMCompanion: topology.requiresRecurrentSSMCompanionState)
            }
            if !coordinator.isPagedIncompatible, cacheCannotUsePagedCoordinatorRestore(self.cache) {
                if cacheCanUsePagedWithRotatingCompanion(self.cache) {
                    coordinator.setPagedBoundaryCompanionRequired(true)
                } else {
                    coordinator.setPagedIncompatible(true)
                }
            }
            let result = coordinator.fetch(
                tokens: tokensToPrefill,
                mediaSalt: mediaSalt,
                preferredDiskBoundaries: input.cacheStablePrefixTokenCounts)
            if case .hit(
                let matchedTokens, let remainingTokens, _, let blocks, let ssmStates,
                let diskArrays) = result
            {
                var restored = false
                if !blocks.isEmpty {
                    let restoredTokens = restoreLayerData(from: blocks, into: self.cache)
                    coordinator.release(blocks: blocks)
                    if restoredTokens > 0 {
                        if let ssm = ssmStates {
                            restoreSSMStates(ssm, into: self.cache, boundary: matchedTokens)
                        }
                        restored = true
                    }
                }
                if let diskArrays, !restored {
                    if restoreFromDiskArrays(diskArrays, into: &self.cache) > 0 {
                        if let ssm = ssmStates {
                            restoreSSMStates(ssm, into: self.cache, boundary: matchedTokens)
                        }
                        MLX.eval(self.cache)
                        restored = true
                    }
                }
                if restored, Self.traceEnabled {
                    let offsets = Set(self.cache.map(\.offset)).sorted()
                    let kvLens = self.cache.compactMap { ($0 as? KVCacheSimple)?.state.first?.dim(2) }
                    FileHandle.standardError.write(Data(
                        "[DFlash2 restore] matched=\(matchedTokens) remaining=\(remainingTokens.count) offsets=\(offsets) kvLens=\(Set(kvLens).sorted())\n"
                            .utf8))
                }
                if restored {
                    if input.cacheHitSuffixContainsMediaPlaceholder(remainingTokens) {
                        self.cache = target.newCache(parameters: effectiveParameters)
                    } else if remainingTokens.isEmpty, let last = tokensToPrefill.last {
                        // A full hit still has to re-run the final token so
                        // the drafter gets at least one row of target hidden
                        // state to condition on; without it there is nothing
                        // to seed `contextHidden` with.
                        let cacheOffset = self.cache.first?.offset ?? tokensToPrefill.count
                        let trimNeeded = cacheOffset - (tokensToPrefill.count - 1)
                        if trimNeeded < 0 {
                            self.cache = target.newCache(parameters: effectiveParameters)
                        } else {
                            if trimNeeded > 0 {
                                for layer in self.cache where layer.isTrimmable {
                                    _ = layer.trim(trimNeeded)
                                }
                            }
                            tokensToPrefill = [last]
                        }
                    } else {
                        tokensToPrefill = remainingTokens
                    }
                }
            } else if populatedCacheRequiresResetAfterCoordinatorMiss(self.cache) {
                self.cache = target.newCache(parameters: effectiveParameters)
            }
        }

        // MARK: prefill with capture

        let prefillStart = Date.timeIntervalSinceReferenceDate
        // The drafter's context window is bounded by its own sliding
        // cache, so retaining more prompt hidden than that is pure waste.
        let hiddenLimit: Int? =
            config.layerTypes.allSatisfy { $0 == "sliding_attention" }
            ? (config.slidingWindow.map { $0 - 1 })
            : nil
        // Same boundary rule as TokenIterator, so both iterators agree on
        // what the canonical cross-turn checkpoint is.
        self.hybridStripBoundary = TokenIterator.hybridStripBoundaryIndex(
            coordinator: cacheCoordinator,
            promptTokenIds: self.promptTokenIds,
            input: input)
        // The boundary index is absolute; prefill only sees the suffix a
        // cache hit left over. A boundary inside the restored prefix needs
        // no capture — a stored boundary at least that long already exists
        // and is the one the next turn will match.
        let restoredCount = self.promptTokenIds.count - tokensToPrefill.count
        let captureAt: Int? = self.hybridStripBoundary.flatMap { stripAt in
            stripAt > restoredCount ? stripAt - restoredCount : nil
        }

        let prefill = try Self.prefill(
            tokens: tokensToPrefill,
            target: target,
            cache: &self.cache,
            captureLayerIDs: self.captureLayerIDs,
            orderedLayerIDs: self.orderedLayerIDs,
            hiddenLimit: hiddenLimit,
            stepSize: effectiveParameters.prefillStepSize,
            captureBoundaryAt: captureAt)
        self.hybridStripSnapshot = prefill.boundarySnapshot
        // Vocabulary agreement, checked against the first real logits row.
        // The drafter borrows the target's LM head, so a mismatch here
        // would not throw — it would silently index a different token
        // space and emit fluent nonsense.
        let targetVocab = prefill.lastLogits.dim(-1)
        guard targetVocab == config.vocabSize else {
            throw DFlash2RuntimeError.drafterTargetMismatch(
                "drafter vocab_size \(config.vocabSize) != model vocabulary \(targetVocab)")
        }
        self.contextHidden = prefill.hidden
        self.stats.seededContextRows = prefill.hidden.dim(1)

        // The drafter's cache starts counting where the retained hidden
        // window starts, so its RoPE positions line up with the target's.
        let hiddenOffset = self.cache.first.map { $0.offset - prefill.hidden.dim(1) } ?? 0
        for c in self.draftCache {
            c.offsetForDFlash2 = Swift.max(0, hiddenOffset)
        }

        self.promptCacheSnapshot = self.cache.map { $0.copy() }

        let first = self.sampler.sample(logits: prefill.lastLogits)
        MLX.eval(first)
        let firstToken = first.reshaped(-1)[0].item(Int.self)
        self.lastToken = firstToken
        self.pendingTokens = [firstToken]
        self.pendingIndex = 0
        self.stats.emittedTokens = 1
        self.promptPrefillTime = Date.timeIntervalSinceReferenceDate - prefillStart
    }

    // MARK: - Prefill

    private struct PrefillResult {
        let lastLogits: MLXArray
        let hidden: MLXArray
        /// Cache state exactly at `captureBoundaryAt` tokens, or `nil`
        /// when no capture was requested or the boundary was never crossed.
        let boundarySnapshot: [KVCache]?
    }

    /// Chunked prefill that also collects the target hidden states the
    /// drafter conditions on.
    ///
    /// `hiddenLimit` keeps only the trailing rows, so a 100k prompt does
    /// not materialise 100k × 5 × hidden of context the drafter's sliding
    /// window would immediately discard.
    private static func prefill(
        tokens: [Int],
        target: any DFlash2Target,
        cache: inout [KVCache],
        captureLayerIDs: Set<Int>,
        orderedLayerIDs: [Int],
        hiddenLimit: Int?,
        stepSize: Int,
        captureBoundaryAt: Int? = nil
    ) throws -> PrefillResult {
        guard !tokens.isEmpty else {
            throw DFlash2RuntimeError.emptyPrompt
        }
        let step = Swift.max(1, stepSize)
        var logits: MLXArray?
        var retained: MLXArray?
        var boundarySnapshot: [KVCache]?
        var start = 0
        while start < tokens.count {
            let remaining = tokens.count - start
            // Leave the final token to its own chunk so the last logits
            // row is the one the first sample comes from, matching the
            // reference's `min(step_size, remaining - 1)`.
            var take = remaining == 1 ? 1 : Swift.min(step, remaining - 1)
            // Force a chunk edge at the capture boundary so the snapshot
            // is the state after EXACTLY that many tokens — a snapshot
            // taken mid-chunk would describe a prefix no key ever matches.
            if let captureAt = captureBoundaryAt, start < captureAt, start + take > captureAt {
                take = captureAt - start
            }
            let end = start + take
            let chunk = MLXArray(tokens[start ..< end].map { Int32($0) })
                .reshaped(1, take)
            let (chunkLogits, captured) = target.callAsFunction(
                chunk, cache: cache, captureLayerIDs: captureLayerIDs,
                recordPrefixCommitStates: false)
            logits = chunkLogits
            let chunkHidden = extractContextFeature(
                captured: captured, targetLayerIDs: orderedLayerIDs)
            if let limit = hiddenLimit {
                let combined =
                    retained.map { concatenated([$0, chunkHidden], axis: 1) } ?? chunkHidden
                let rows = combined.dim(1)
                retained =
                    rows > limit ? combined[0..., (rows - limit)..., 0...] : combined
            } else {
                retained = retained.map { concatenated([$0, chunkHidden], axis: 1) } ?? chunkHidden
            }
            if end == captureBoundaryAt {
                MLX.eval(cache)
                boundarySnapshot = cache.map { $0.copy() }
            }
            if end < tokens.count {
                MLX.eval(cache)
                if let retained { MLX.eval(retained) }
                Memory.clearCache()
            }
            start = end
        }
        guard let logits, let retained else {
            throw DFlash2RuntimeError.emptyPrompt
        }
        let lastLogits = logits[0..., (logits.dim(1) - 1)..., 0...]
        MLX.eval(lastLogits, retained)
        return PrefillResult(
            lastLogits: lastLogits, hidden: retained, boundarySnapshot: boundarySnapshot)
    }

    // MARK: - Iteration

    mutating func next() -> Int? {
        if let maxTokens, tokenCount >= maxTokens {
            if Self.traceEnabled, stats.verifyCalls > 0 {
                let line = String(
                    format: "[DFlash2 stats] cycles=%d accLen=%.2f draft=%.2fs verify=%.2fs commit=%.2fs arFallback=%d\n",
                    stats.verifyCalls, stats.acceptanceLength, stats.draftSeconds,
                    stats.verifySeconds, stats.commitSeconds,
                    stats.autoregressiveFallbackTokens)
                FileHandle.standardError.write(Data(line.utf8))
            }
            return nil
        }
        if pendingIndex >= pendingTokens.count {
            guard runCycle() else { return nil }
        }
        guard pendingIndex < pendingTokens.count else { return nil }
        let token = pendingTokens[pendingIndex]
        pendingIndex += 1
        tokenCount += 1
        if tokenCount % 256 == 0 {
            Memory.clearCache()
        }
        return token
    }

    /// One draft → verify → accept cycle. Returns false when nothing more
    /// can be produced.
    private mutating func runCycle() -> Bool {
        pendingTokens.removeAll(keepingCapacity: true)
        pendingIndex = 0

        let budget = maxTokens.map { $0 - tokenCount } ?? Int.max
        guard budget > 0 else { return false }

        // The block spends one position on the anchor, so a block of size
        // `bs` yields at most `bs` new tokens (bs-1 drafts + 1 bonus).
        let bs = Swift.min(blockSize, budget + 1)
        if bs <= 1 {
            return runAutoregressiveStep()
        }

        // MARK: draft

        let draftStart = Date.timeIntervalSinceReferenceDate
        var blockIds = [Int32(lastToken)]
        blockIds.append(contentsOf: Array(repeating: Int32(maskTokenID), count: bs - 1))
        let block = MLXArray(blockIds).reshaped(1, bs)

        let proposal = drafter.propose(
            inputs: block,
            targetHidden: contextHidden,
            cache: draftCache,
            embedder: target,
            temperature: isGreedy ? 0 : temperature,
            logitsStart: 1)
        // Dispatch, don't sync: the verify graph below consumes the draft
        // tokens ON-GRAPH, so the GPU runs the drafter while the CPU is
        // still building the verify forward. The reference does the same
        // with `mx.async_eval(draft_tokens)`, and oMLX's Lightning MTP PR
        // credits collapsing to one host sync per cycle for a large share
        // of its speedup — this loop previously paid three.
        asyncEval(proposal.tokens)
        stats.draftSeconds += Date.timeIntervalSinceReferenceDate - draftStart

        // The sliding clip inside drafter attention can advance the cache
        // offset past the committed token count. Pull it back so the next
        // round's RoPE offsets stay absolute.
        let expectedDraftOffset = promptTokenIds.count + stats.emittedTokens - 1
        if let head = draftCache.first, head.offset > expectedDraftOffset {
            let excess = head.offset - expectedDraftOffset
            for c in draftCache {
                if c.isTrimmable {
                    _ = c.trim(excess)
                } else {
                    c.offsetForDFlash2 = Swift.max(0, c.offset - excess)
                }
            }
        }

        let draftTokens = proposal.tokens
        stats.draftedTokens += draftTokens.dim(1)

        // MARK: verify

        // Built on-graph from the drafter's output — no host read between
        // draft and verify. The old shape (asArray the drafts, rebuild an
        // MLXArray from host ints) forced a full pipeline drain per cycle
        // exactly where the reference pipelines.
        let verifyInput = concatenated(
            [MLXArray([Int32(lastToken)]).reshaped(1, 1), draftTokens], axis: 1)

        let hasRecurrentState = cache.contains { !$0.isTrimmable }
        // Input-capture is the fast rollback: recurrent layers stash
        // REFERENCES to this forward's inputs and a rejection replays only
        // the accepted rows, one kernel per layer. The per-prefix recording
        // fallback (capture_commit) re-runs the scan once per prefix WITH a
        // graph flush per record — measured at 48 layers × 7 prefixes = 336
        // launches + 336 flushes per cycle on Qwen3.8, most of a verify
        // that cost 5.1× a decode step.
        let usesInputCapture = hasRecurrentState && target is DFlash2VerifyRollbackModel
            && ProcessInfo.processInfo.environment["VMLX_DFLASH2_INPUT_CAPTURE"] != "0"
        let verifierMode = usesInputCapture
            ? "input_capture"
            : (hasRecurrentState ? "capture_commit" : "input_capture")
        let verifyStart = Date.timeIntervalSinceReferenceDate
        let (logits, captured) = NativeMTPVerifierStatePolicy.withVerifierMode(verifierMode) {
            target.callAsFunction(
                verifyInput, cache: cache, captureLayerIDs: captureLayerIDs,
                recordPrefixCommitStates: hasRecurrentState && !usesInputCapture)
        }
        let newHidden = extractContextFeature(
            captured: captured, targetLayerIDs: orderedLayerIDs)
        // Greedy target ids join the same flush as the hidden states, so
        // the argmax never costs its own graph submission.
        let greedyTargetIds = isGreedy ? argMax(logits, axis: -1) : nil
        if let greedyTargetIds {
            asyncEval(greedyTargetIds, newHidden)
        } else {
            asyncEval(logits, newHidden)
        }
        stats.verifySeconds += Date.timeIntervalSinceReferenceDate - verifyStart
        stats.verifyCalls += 1

        // MARK: accept — the cycle's single host sync happens on the first
        // asArray below, after both forwards are already in flight.

        let draftIDs = draftTokens.reshaped(-1).asArray(Int32.self).map(Int.init)
        let acceptance: DFlash2Sampling.Acceptance
        if isGreedy {
            let targetIDs = greedyTargetIds!.reshaped(-1).asArray(Int32.self).map(Int.init)
            acceptance = DFlash2Sampling.acceptGreedy(
                draftTokens: draftIDs, targetTokens: targetIDs)
        } else {
            let targetProbs = DFlash2Sampling.probabilities(
                logits, temperature: temperature, topP: topP, topK: topK)
            guard let draftProbs = proposal.probabilities else {
                // Sampled request with no q: cannot run the accept test.
                return runAutoregressiveStep()
            }
            acceptance = DFlash2Sampling.acceptSampled(
                draftTokens: draftTokens,
                targetProbabilities: targetProbs,
                draftProbabilities: draftProbs,
                draftIndices: proposal.candidates)
        }

        let accepted = acceptance.accepted
        stats.acceptedTokens += accepted

        // MARK: commit

        let commitStart = Date.timeIntervalSinceReferenceDate
        let committedInputs = accepted + 1
        let rejected = verifyInput.dim(1) - committedInputs
        if rejected > 0 {
            let committed: Bool
            if usesInputCapture, let rollback = target as? DFlash2VerifyRollbackModel {
                for layer in cache where layer.isTrimmable {
                    _ = layer.trim(rejected)
                }
                committed = rollback.commitVerifiedBlock(
                    cache: cache, acceptedInputs: committedInputs)
            } else {
                committed = Self.commit(
                    cache: cache, committedInputs: committedInputs, rejected: rejected)
            }
            guard committed
            else {
                // The recurrent layers had no recorded state for the
                // accepted length, so the cache can no longer be trusted
                // to describe the emitted tokens. Ending the turn here is
                // the only safe move — but a truncated answer is
                // indistinguishable from a short one, so say so loudly
                // rather than letting it look like a normal completion.
                stats.commitSeconds += Date.timeIntervalSinceReferenceDate - commitStart
                FileHandle.standardError.write(
                    Data(
                        ("[DFlash2] ABORTED at cycle \(stats.verifyCalls): no recorded "
                            + "recurrent state for the accepted prefix (\(committedInputs) of "
                            + "\(verifyInput.dim(1))). The turn is TRUNCATED at "
                            + "\(stats.emittedTokens) tokens.\n").utf8))
                return false
            }
        } else {
            for layer in cache { (layer as? MambaCache)?.clearVerifyInputStash() }
            Self.clearRecordedPrefixes(cache)
        }
        stats.commitSeconds += Date.timeIntervalSinceReferenceDate - commitStart

        contextHidden = newHidden[0..., ..<committedInputs, 0...]

        var emitted = draftIDs.prefix(accepted).map { $0 }
        emitted.append(acceptance.bonus)
        if emitted.count > budget {
            emitted = Array(emitted.prefix(budget))
        }
        guard let last = emitted.last else { return false }
        lastToken = last
        pendingTokens = emitted
        stats.emittedTokens += emitted.count

        if Self.traceEnabled {
            let line =
                "[DFlash2] cycle=\(stats.verifyCalls) bs=\(bs) drafted=\(draftIDs.count) accepted=\(accepted) emitted=\(emitted.count) ids=\(emitted) draftIds=\(draftIDs) ctxRows=\(contextHidden.dim(1)) accLen=\(String(format: "%.2f", stats.acceptanceLength))\n"
            FileHandle.standardError.write(Data(line.utf8))
        }
        return true
    }

    /// Single-token target step. Used when the remaining budget cannot
    /// fill a block and as the safety valve when acceptance cannot run.
    private mutating func runAutoregressiveStep() -> Bool {
        let input = MLXArray([Int32(lastToken)]).reshaped(1, 1)
        let (logits, captured) = target.callAsFunction(
            input, cache: cache, captureLayerIDs: captureLayerIDs,
            recordPrefixCommitStates: false)
        let hidden = extractContextFeature(captured: captured, targetLayerIDs: orderedLayerIDs)
        let sampled = sampler.sample(logits: logits[0..., -1, 0...])
        MLX.eval(sampled, hidden)
        let token = sampled.reshaped(-1)[0].item(Int.self)
        contextHidden = hidden
        lastToken = token
        pendingTokens = [token]
        pendingIndex = 0
        stats.emittedTokens += 1
        stats.autoregressiveFallbackTokens += 1
        return true
    }

    /// Roll the target cache back to the accepted prefix.
    ///
    /// Trimmable layers drop the rejected suffix directly. Recurrent
    /// layers restore the state they recorded at that exact step during
    /// the verify forward — no replay, no recomputation.
    private static func commit(
        cache: [KVCache], committedInputs: Int, rejected: Int
    ) -> Bool {
        for layer in cache where !layer.isTrimmable {
            guard let mamba = layer as? MambaCache,
                mamba.commitRecordedPrefix(length: committedInputs)
            else {
                clearRecordedPrefixes(cache)
                return false
            }
        }
        for layer in cache where layer.isTrimmable {
            _ = layer.trim(rejected)
        }
        return true
    }

    private static func clearRecordedPrefixes(_ cache: [KVCache]) {
        for layer in cache {
            (layer as? MambaCache)?.clearRecordedPrefixes()
        }
    }

    // MARK: - Stats + cache store

    var dflash2Stats: DFlash2GenerationStats { stats }

    mutating func storeCacheAfterGeneration(
        generatedTokenIds: [Int],
        includeGeneratedBoundary: Bool
    ) {
        guard let coordinator = cacheCoordinator, !promptTokenIds.isEmpty else { return }
        defer {
            promptCacheSnapshot = nil
            hybridStripSnapshot = nil
        }

        // The generation-suffix-stripped boundary is the canonical
        // cross-turn checkpoint, and when it exists it is the ONLY store:
        // the next templated turn replaces the generation suffix, so the
        // full prompt is a boundary it can never contain, and storing both
        // would make the wider entry shadow the reusable one on
        // longest-prefix match. Same policy as TokenIterator's
        // `usesCanonicalHybridBoundary`.
        if let stripAt = hybridStripBoundary {
            guard let snapshot = hybridStripSnapshot,
                snapshot.allSatisfy({ $0.offset == stripAt })
            else {
                // Boundary sat inside a restored prefix — an entry at
                // least that long is already stored and is the one the
                // next turn matches. Nothing to add.
                return
            }
            coordinator.storeAfterGeneration(
                promptTokens: Array(promptTokenIds.prefix(stripAt)),
                perLayerData: extractLayerData(from: snapshot),
                ssmStates: extractSSMStates(from: snapshot),
                cache: snapshot,
                mediaSalt: mediaSalt)
            return
        }

        // No strip boundary (dense target, or tiers off): the post-prefill
        // full-prompt snapshot is reusable directly. It is captured before
        // any speculation touched the cache, so it cannot carry a
        // rolled-back recurrent state.
        guard let snapshot = promptCacheSnapshot,
            snapshot.allSatisfy({ $0.offset == promptTokenIds.count })
        else { return }
        coordinator.storeAfterGeneration(
            promptTokens: promptTokenIds,
            perLayerData: extractLayerData(from: snapshot),
            ssmStates: extractSSMStates(from: snapshot),
            cache: snapshot,
            mediaSalt: mediaSalt)
    }
}
