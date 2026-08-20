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

    // MARK: verify prefetch
    //
    // A cycle's forwards take ~50-140ms of GPU; emitting its 3-7 accepted
    // tokens then costs the HOST several ms each in detokenize/stream/
    // stats, during which the GPU has nothing to do. Submitting the NEXT
    // cycle's draft+verify the moment this cycle's decision lands overlaps
    // that gap — same forwards, zero extra compute. The Python engine
    // measured the identical change at +65% on this model.
    //
    // Only safe under the STAGED verify: a staged forward leaves committed
    // recurrent state and offsets untouched, so an in-flight verify that
    // is never accepted has advanced ONLY the trimmable attention caches,
    // and abandoning it is a trim. Under eager input-capture the recurrent
    // state has already moved and there is no zero-accept rollback.
    private struct InFlightVerify {
        let blockSize: Int
        let proposal: DFlash2Proposal
        let draftTokens: MLXArray
        let verifyRows: Int
        let logits: MLXArray
        let greedyTargetIds: MLXArray?
        let newHidden: MLXArray
        let stagedCycle: Bool
        let usesInputCapture: Bool
        let hasRecurrentState: Bool
        let verifySeconds: Double
    }

    private var inFlight: InFlightVerify?

    /// Set when a drafter forward degenerates mid-turn (an MLX error in
    /// its layer stack degraded to a husk). The rest of the turn decodes
    /// autoregressively — a drafter can only cost acceptance, never
    /// correctness, so its failure must never end the turn.
    private var drafterDisabled = false

    /// OPT-IN (`VMLX_DFLASH2_VERIFY_PREFETCH=1`), and measured as a
    /// non-win on this runtime: through the real `generate()` path on
    /// Qwen3.8-27B-JANG_4D it is neutral-to-negative (think ON 25.95 vs
    /// 25.87 tok/s, think OFF 20.02 vs 21.22). The Python engine measured
    /// the same change at +65%, but it was closing a 10-18 ms/cycle host
    /// gap that this loop does not have: draft and verify are already
    /// dispatched with `asyncEval` and the cycle pays exactly one host
    /// sync, so there is little idle GPU to fill and the speculative
    /// block is pure cost whenever it is abandoned.
    private var prefetchEnabled: Bool {
        useStagedVerify
            && ProcessInfo.processInfo.environment["VMLX_DFLASH2_VERIFY_PREFETCH"] == "1"
    }

    // MARK: compiled verify (input_capture_staged)

    /// Whether full-size blocks run the staged verify (attention caches
    /// promoted to Compilable buffers; GDN commits from staging slots).
    private var useStagedVerify = false
    /// The first staged cycle runs EAGERLY to allocate the staging slots;
    /// `compile()` needs those objects to exist before the trace.
    private var stagedVerifyWarm = false
    /// Compiled S = 1+block verify forwards, keyed by block size. Each is
    /// a fixed shape; the adaptive controller below moves between a small
    /// set of sizes, so at most a handful of traces are ever built. Tail
    /// blocks and the AR fallback keep the eager path.
    private var compiledVerify: [Int: @Sendable ([MLXArray]) -> [MLXArray]] = [:]
    private var stagedWarmedSizes: Set<Int> = []

    // MARK: adaptive block size
    //
    // The right block size is a property of the CONTENT, not the model: on
    // Qwen3.8-27B-JANG_4D, code accepts 70-83% of drafts and wants a wide
    // block (b15: 7.27 accepted/cycle, 38.4 tok/s), while free prose
    // accepts 37-44% and is fastest narrow (b4: ~26 tok/s; b15 gives 22).
    // A fixed default cannot serve both — measured at 28.5 tok/s for code
    // at the old fixed b8, versus 38.4 at its own optimum.
    //
    // So track the acceptance RATE (accepted / drafted, which is
    // block-size independent) and widen or narrow within the ladder.
    /// Two entries, not a fine ladder: the measured surface is bimodal.
    /// Narrow blocks win on prose (b4 beats b8 in every measured cell),
    /// wide blocks win on code (b15: 43.3 tok/s vs b8's 33.3), and b8 —
    /// the drafter config's own default — was the WORST of the three in
    /// 6 of 6 cells. Probing costs real cycles, so probe only the two
    /// sizes that can actually win.
    private static let blockLadder = [4, 15]
    /// On unless the caller pinned a block size explicitly (a pinned size
    /// is a measurement request — honour it) or the kill switch is set.
    private let adaptiveBlockSizeEnabled: Bool
    private var adaptiveBlockSize: Int?
    /// Per-ladder-size observed throughput: emitted tokens and the verify
    /// seconds that produced them. Throughput is measured, not predicted —
    /// an acceptance-RATE rule looks principled and is wrong, because the
    /// rate falls mechanically as the block widens (more drafts to get
    /// right), so rate thresholds equilibrate mid-ladder instead of at the
    /// optimum. Measured on JANG_4D: a rate-threshold controller settled
    /// on b12 and produced 34.5 tok/s where fixed b15 produced 41.9.
    private var adaptiveTokens: [Int: Int] = [:]
    private var adaptiveSeconds: [Int: Double] = [:]
    private var adaptiveCyclesAtSize = 0
    private var adaptiveSettled = false

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
        // A caller-pinned block size is a measurement request — honour it
        // exactly. Only the bundle default adapts.
        // OPT-IN, not default. The probe measurably finds the right size
        // (see DFLASH2.md's block-size table), but switching size mid-turn
        // still interacts badly with the compiled staged verify: the
        // staging slots and the trace are both shaped by the block length,
        // and a switch can crash the process (SIGSEGV in
        // DFlash2LosslessSmokeTests) and perturb greedy output. Until that
        // is closed, callers get a fixed size and the tuning table.
        self.adaptiveBlockSizeEnabled =
            requestedBlockSize == nil
            && ProcessInfo.processInfo.environment["VMLX_DFLASH2_ADAPTIVE_BLOCK"] == "1"
        self.adaptiveBlockSize =
            self.adaptiveBlockSizeEnabled ? Self.recallLearnedBlockSize() : nil
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

        if Self.traceEnabled {
            let promptCount = promptTokenIds.count
            let prefillCount = tokensToPrefill.count
            let offsets: [Int] = self.cache.prefix(3).map { $0.offset }
            let line = "[DFlash2Init] prompt=\(promptCount) toPrefill=\(prefillCount)"
                + " restored=\(restoredCount) offsets=\(offsets)\n"
            FileHandle.standardError.write(Data(line.utf8))
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
        if Self.traceEnabled {
            let logitsShape = prefill.lastLogits.shape
            let hiddenShape = prefill.hidden.shape
            let line = "[DFlash2Init] lastLogits=\(logitsShape) hidden=\(hiddenShape)\n"
            FileHandle.standardError.write(Data(line.utf8))
        }
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
        if Self.traceEnabled {
            let firstShape = first.shape
            let firstDtype = first.dtype
            let samplerKind = String(describing: type(of: self.sampler))
            let line = "[DFlash2Init] first=\(firstShape) dtype=\(firstDtype) sampler=\(samplerKind)\n"
            FileHandle.standardError.write(Data(line.utf8))
        }
        let firstToken = first.reshaped(-1)[0].item(Int.self)
        self.lastToken = firstToken
        self.pendingTokens = [firstToken]
        self.pendingIndex = 0
        self.stats.emittedTokens = 1
        self.promptPrefillTime = Date.timeIntervalSinceReferenceDate - prefillStart

        // MARK: compiled verify promotion
        //
        // The verify forward is a FIXED shape (1 anchor + block drafts), so
        // it can be compiled exactly like plain decode — killing the
        // per-cycle CPU graph rebuild. Attention caches become static
        // Compilable buffers (graph-visible offset, trim = offset rewind);
        // recurrent layers run `input_capture_staged` and are committed
        // host-side from their staging slots. Promotion happens AFTER the
        // snapshots above so stored prefix-cache entries stay plain.
        // OPT-IN until the teardown crash is closed. The path is proven
        // token-identical and materially faster (see DFLASH2.md), but the
        // compiled traces capture the cache array and the iterator is a
        // STRUCT — copies retain traces independently, so clearing one
        // copy's table does not release them, and the process segfaults at
        // exit. Reproduce: run DFlash2LosslessSmokeTests with this ON
        // (tests pass, process exits with signal 11) versus OFF (clean).
        // Staged verify is independent of compiling it: staging is what
        // makes the commit lazy and an in-flight verify abandonable, so it
        // is what verify prefetch needs. Compiling the traced forward is a
        // separate opt-in below.
        if target is DFlash2StagedVerifyRollbackModel,
            ProcessInfo.processInfo.environment["VMLX_DFLASH2_STAGED_VERIFY"] != "0"
        {
            self.useStagedVerify = true
        }
        if useStagedVerify, HardwareInfo.isCompiledDecodeSupported,
            ProcessInfo.processInfo.environment["VMLX_DFLASH2_COMPILED_VERIFY"] == "1"
        {
            let promptOffset = self.cache.map(\.offset).max() ?? self.promptTokenIds.count
            let bufferLength = promptOffset + (self.maxTokens ?? 4096)
                + effectiveBlockSize + 8
            self.cache = self.cache.map { layer in
                if !(layer is CompilableKVCache), layer is KVCacheSimple {
                    return CompilableKVCache(from: layer, maxLength: bufferLength)
                }
                return layer
            }
            MLX.eval(self.cache)
            self.useStagedVerify = true
        }
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
            // Release the compiled traces at end of generation. They
            // capture the cache array, and letting them outlive the run
            // segfaults at process teardown (measured: clean with
            // VMLX_DFLASH2_COMPILED_VERIFY=0, SIGSEGV after the last test
            // with it on) — the same capture-cycle class as the compiled
            // decode trampoline that once pinned unloaded weights.
            abandonInFlightVerify()
            releaseCompiledVerify()
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

        // A prefetched verify is already on the GPU — consume it, provided
        // it still fits the remaining budget.
        if let prefetched = inFlight, prefetched.blockSize <= budget + 1 {
            inFlight = nil
            return completeVerify(prefetched)
        }
        // Anything else in flight was speculated against a budget or block
        // size we can no longer use; roll its rows back before proceeding.
        abandonInFlightVerify()

        // The block spends one position on the anchor, so a block of size
        // `bs` yields at most `bs` new tokens (bs-1 drafts + 1 bonus).
        let bs = Swift.min(adaptiveBlockSize ?? blockSize, budget + 1)
        if bs <= 1 || drafterDisabled {
            return runAutoregressiveStep()
        }
        guard let flight = issueVerify(bs: bs) else {
            // The drafter forward produced a degenerate result — an MLX
            // error inside its layer stack degraded to a scalar husk
            // (#123). A drafter can only ever cost acceptance, never
            // correctness, so the failure must never take the turn (or
            // the host) down: disable speculation for the REST of this
            // turn and decode autoregressively. Observed live 2026-08-20
            // after a fully-rejected block once the context passed the
            // drafter's 2048-token sliding window.
            drafterDisabled = true
            FileHandle.standardError.write(Data(
                ("[DFlash2] drafter forward degenerated at cycle "
                    + "\(stats.verifyCalls) (ctx=\(promptTokenIds.count + stats.emittedTokens)); "
                    + "speculation disabled for the rest of the turn, "
                    + "continuing autoregressively.\n").utf8))
            return runAutoregressiveStep()
        }
        return completeVerify(flight)
    }

    /// Build one cycle's draft + verify graphs and DISPATCH them, without
    /// reading anything back. Returns the handles the accept phase needs.
    ///
    /// Split out from the accept phase so the next cycle's forwards can be
    /// submitted while the host is still detokenizing and streaming this
    /// cycle's tokens — see `prefetchNextVerify`.
    private mutating func issueVerify(bs: Int) -> InFlightVerify? {

        // MARK: draft

        let draftStart = Date.timeIntervalSinceReferenceDate
        var blockIds = [Int32(lastToken)]
        blockIds.append(contentsOf: Array(repeating: Int32(maskTokenID), count: bs - 1))
        let block = MLXArray(blockIds).reshaped(1, bs)

        // The drafter runs inside its own error scope: a real MLX error in
        // its layer stack (observed live after a fully-rejected block past
        // the drafter's sliding window) must surface as a caught error and
        // an AR fallback, never as a degraded husk that dies on the next
        // host-side shape read.
        let proposal: DFlash2Proposal
        do {
            proposal = try withError {
                drafter.propose(
                    inputs: block,
                    targetHidden: contextHidden,
                    cache: draftCache,
                    embedder: target,
                    temperature: isGreedy ? 0 : temperature,
                    logitsStart: 1)
            }
        } catch {
            FileHandle.standardError.write(Data(
                "[DFlash2] drafter forward error: \(error)\n".utf8))
            return nil
        }
        // Host-side metadata check, no sync: a degraded drafter forward
        // (#123 husk) surfaces here as a token tensor with the wrong rank
        // or an empty draft row. The caller falls back to AR.
        guard proposal.tokens.ndim == 2, proposal.tokens.dim(1) == bs - 1 else {
            return nil
        }
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
        // Full-size blocks take the compiled staged verify; tail blocks
        // (shrunken bs near the budget) keep the eager path — compiling is
        // only worth it for the shape that runs hundreds of times.
        let stagedCycle = useStagedVerify && bs == (adaptiveBlockSize ?? blockSize)
        let verifyStart = Date.timeIntervalSinceReferenceDate
        var logits: MLXArray
        var greedyTargetIds: MLXArray? = nil
        let newHidden: MLXArray
        if stagedCycle {
            if !stagedWarmedSizes.contains(bs) {
                // Eager warm-up under the staged mode: allocates the
                // fixed staging slots the compile trace will track.
                let (l, captured) = NativeMTPVerifierStatePolicy.withVerifierMode(
                    "input_capture_staged")
                {
                    target.callAsFunction(
                        verifyInput, cache: cache, captureLayerIDs: captureLayerIDs,
                        recordPrefixCommitStates: false)
                }
                logits = l
                newHidden = extractContextFeature(
                    captured: captured, targetLayerIDs: orderedLayerIDs)
                stagedWarmedSizes.insert(bs)
            } else {
                if compiledVerify[bs] == nil { buildCompiledVerify(blockLength: bs) }
                let outs = compiledVerify[bs]!([verifyInput])
                logits = outs[0]
                if isGreedy { greedyTargetIds = outs[0] }
                // Captured hiddens come back in `orderedLayerIDs` order —
                // the same order extractContextFeature concatenates.
                newHidden = concatenated(Array(outs[1...]), axis: -1)
            }
        } else {
            let verifierMode = usesInputCapture
                ? "input_capture"
                : (hasRecurrentState ? "capture_commit" : "input_capture")
            let (l, captured) = NativeMTPVerifierStatePolicy.withVerifierMode(verifierMode) {
                target.callAsFunction(
                    verifyInput, cache: cache, captureLayerIDs: captureLayerIDs,
                    recordPrefixCommitStates: hasRecurrentState && !usesInputCapture)
            }
            logits = l
            newHidden = extractContextFeature(
                captured: captured, targetLayerIDs: orderedLayerIDs)
        }
        // Greedy target ids join the same flush as the hidden states, so
        // the argmax never costs its own graph submission.
        if isGreedy, greedyTargetIds == nil {
            greedyTargetIds = argMax(logits, axis: -1)
        }
        if let greedyTargetIds {
            asyncEval(greedyTargetIds, newHidden)
        } else {
            asyncEval(logits, newHidden)
        }
        let cycleVerifySeconds = Date.timeIntervalSinceReferenceDate - verifyStart
        stats.verifySeconds += cycleVerifySeconds
        stats.verifyCalls += 1

        return InFlightVerify(
            blockSize: bs,
            proposal: proposal,
            draftTokens: draftTokens,
            verifyRows: verifyInput.dim(1),
            logits: logits,
            greedyTargetIds: greedyTargetIds,
            newHidden: newHidden,
            stagedCycle: stagedCycle,
            usesInputCapture: usesInputCapture,
            hasRecurrentState: hasRecurrentState,
            verifySeconds: cycleVerifySeconds)
    }

    /// Read back one dispatched cycle, accept its longest agreeing prefix,
    /// commit the cache, and queue the emitted tokens.
    private mutating func completeVerify(_ flight: InFlightVerify) -> Bool {
        let bs = flight.blockSize
        let proposal = flight.proposal
        let draftTokens = flight.draftTokens
        let logits = flight.logits
        let greedyTargetIds = flight.greedyTargetIds
        let newHidden = flight.newHidden
        let stagedCycle = flight.stagedCycle
        let usesInputCapture = flight.usesInputCapture
        let cycleVerifySeconds = flight.verifySeconds
        let budget = maxTokens.map { $0 - tokenCount } ?? Int.max

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
        let rejected = flight.verifyRows - committedInputs
        if stagedCycle {
            // Staged verify left the recurrent state and offsets untouched;
            // EVERY cycle commits from the staging slots (a full accept
            // adopts the staged final state, no replay). Attention caches
            // advanced in-graph and just rewind their offset.
            if rejected > 0 {
                for layer in cache where layer.isTrimmable {
                    _ = layer.trim(rejected)
                }
            }
            let committed = (target as? DFlash2StagedVerifyRollbackModel)?
                .commitStagedVerifiedBlock(
                    cache: cache, acceptedInputs: committedInputs,
                    blockLength: flight.verifyRows) ?? false
            guard committed else {
                stats.commitSeconds += Date.timeIntervalSinceReferenceDate - commitStart
                FileHandle.standardError.write(
                    Data(
                        ("[DFlash2] ABORTED at cycle \(stats.verifyCalls): staged verify "
                            + "commit failed for \(committedInputs) of "
                            + "\(flight.verifyRows) rows. The turn is TRUNCATED at "
                            + "\(stats.emittedTokens) tokens.\n").utf8))
                return false
            }
        } else if rejected > 0 {
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
                            + "\(flight.verifyRows)). The turn is TRUNCATED at "
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
        if adaptiveBlockSizeEnabled, stagedCycle {
            updateAdaptiveBlockSize(
                emitted: emitted.count, verifySeconds: cycleVerifySeconds)
        }
        // The cache, contextHidden and lastToken now describe exactly the
        // committed prefix, so the next cycle can be built and dispatched
        // right here — it runs on the GPU while the caller detokenizes and
        // streams the tokens queued above.
        prefetchNextVerify()

        if Self.traceEnabled {
            let line =
                "[DFlash2] cycle=\(stats.verifyCalls) bs=\(bs) drafted=\(draftIDs.count) accepted=\(accepted) emitted=\(emitted.count) ids=\(emitted) draftIds=\(draftIDs) ctxRows=\(contextHidden.dim(1)) accLen=\(String(format: "%.2f", stats.acceptanceLength))\n"
            FileHandle.standardError.write(Data(line.utf8))
        }
        return true
    }

    /// Dispatch the next cycle's forwards, if one is worth having.
    private mutating func prefetchNextVerify() {
        guard prefetchEnabled, inFlight == nil else { return }
        // Budget left AFTER the tokens queued by the cycle that just
        // completed — they have not been consumed by `next()` yet.
        let remaining = maxTokens.map { $0 - tokenCount - pendingTokens.count } ?? Int.max
        guard remaining > 1 else { return }
        let bs = Swift.min(adaptiveBlockSize ?? blockSize, remaining + 1)
        guard bs > 1 else { return }
        inFlight = issueVerify(bs: bs)
    }

    /// Undo a dispatched-but-unaccepted verify. Under the staged verify it
    /// advanced only the trimmable attention caches, so the rollback is a
    /// trim plus dropping the staging slots; the recurrent state and its
    /// offset were never touched. Must run before ANY path that persists
    /// or reuses the cache, or unverified draft positions leak into a
    /// stored prefix.
    private mutating func abandonInFlightVerify() {
        guard let flight = inFlight else { return }
        inFlight = nil
        for layer in cache where layer.isTrimmable {
            _ = layer.trim(flight.verifyRows)
        }
        for layer in cache { (layer as? MambaCache)?.clearVerifyStaging() }
        stagedWarmedSizes.removeAll(keepingCapacity: true)
        // The drafter cache also advanced for the abandoned block.
        let expectedDraftOffset = promptTokenIds.count + stats.emittedTokens - 1
        for c in draftCache where c.offset > expectedDraftOffset {
            let excess = c.offset - expectedDraftOffset
            if c.isTrimmable {
                _ = c.trim(excess)
            } else {
                c.offsetForDFlash2 = Swift.max(0, c.offset - excess)
            }
        }
        if Self.traceEnabled {
            FileHandle.standardError.write(Data(
                "[DFlash2] abandoned in-flight verify (\(flight.verifyRows) rows)\n".utf8))
        }
    }

    /// Compile the fixed-shape staged verify forward. Built AFTER the
    /// eager warm-up cycle so the staging slots exist as persistent
    /// objects the trace's state tracking can capture.
    private mutating func buildCompiledVerify(blockLength: Int) {
        let capturedTarget = target
        let cacheRef = cache
        let capIDs = captureLayerIDs
        let ids = orderedLayerIDs
        let greedy = isGreedy
        self.compiledVerify[blockLength] = compile(
            inputs: cacheRef, outputs: cacheRef
        ) { (args: [MLXArray]) -> [MLXArray] in
            CompiledDecodeTrace.withActive {
                NativeMTPVerifierStatePolicy.withVerifierMode("input_capture_staged") {
                    let (logits, captured) = capturedTarget.callAsFunction(
                        args[0], cache: cacheRef, captureLayerIDs: capIDs,
                        recordPrefixCommitStates: false)
                    var outs: [MLXArray] = [greedy ? argMax(logits, axis: -1) : logits]
                    for id in ids { outs.append(captured[id]!) }
                    return outs
                }
            }
        }
        if Self.traceEnabled {
            FileHandle.standardError.write(Data(
                "[DFlash2] compiled verify built (S=\(blockLength))\n".utf8))
        }
    }

    /// Move the block size along the ladder from the observed acceptance
    /// RATE (accepted / drafted — independent of the block size that
    /// produced it, so it stays comparable across a change).
    ///
    /// Thresholds come from the measured matrix on Qwen3.8-27B-JANG_4D:
    /// code sits at 0.70-0.83 and is fastest at b15; prose sits at
    /// 0.37-0.44 and is fastest at b4. The band between them is flat
    /// enough that hysteresis matters more than the exact cut points —
    /// hence the dwell requirement before any move.
    private mutating func updateAdaptiveBlockSize(
        emitted: Int, verifySeconds: Double
    ) {
        let current = adaptiveBlockSize ?? blockSize
        adaptiveTokens[current, default: 0] += emitted
        adaptiveSeconds[current, default: 0] += verifySeconds
        adaptiveCyclesAtSize += 1
        guard !adaptiveSettled, adaptiveCyclesAtSize >= Self.adaptiveProbeCycles else { return }

        // Probe every ladder size once, then settle on the measured best
        // and stay there. Verify seconds (not wall time) is the comparison
        // base: it is the term block size actually moves, and it excludes
        // the drafter and host work that would add noise without signal.
        adaptiveCyclesAtSize = 0
        if let unprobed = Self.blockLadder.first(where: { adaptiveTokens[$0] == nil }) {
            changeBlockSize(from: current, to: unprobed, settled: false)
            return
        }
        let best = Self.blockLadder.max { a, b in
            throughput(of: a) < throughput(of: b)
        }
        adaptiveSettled = true
        if let best {
            Self.rememberLearnedBlockSize(best)
            if best != current {
                changeBlockSize(from: current, to: best, settled: true)
            }
        }
    }

    private static let adaptiveProbeCycles = 4

    /// Last size the probe settled on, remembered process-wide so the
    /// second and later turns of a session start where the first one
    /// finished instead of paying the probe again. Purely a starting
    /// hint — every turn still re-probes and can move away from it.
    private nonisolated(unsafe) static var learnedBlockSize: Int?
    private static let learnedBlockLock = NSLock()

    private static func rememberLearnedBlockSize(_ size: Int) {
        learnedBlockLock.lock()
        defer { learnedBlockLock.unlock() }
        learnedBlockSize = size
    }

    private static func recallLearnedBlockSize() -> Int? {
        learnedBlockLock.lock()
        defer { learnedBlockLock.unlock() }
        return learnedBlockSize
    }

    private func throughput(of blockSize: Int) -> Double {
        let seconds = adaptiveSeconds[blockSize] ?? 0
        guard seconds > 0 else { return 0 }
        return Double(adaptiveTokens[blockSize] ?? 0) / seconds
    }

    /// Switch the verify block length. The staging slots and every
    /// compiled trace are shaped by that length, so both must be dropped:
    /// a trace built for one row count cannot be replayed against slots
    /// holding another (it crashes the process, not just the numbers).
    /// The next cycle re-warms eagerly and rebuilds one trace.
    /// Drop every compiled verify trace and the staging slots they were
    /// built against. Safe at any point — the next staged cycle re-warms.
    mutating func releaseCompiledVerify() {
        guard !compiledVerify.isEmpty || !stagedWarmedSizes.isEmpty else { return }
        compiledVerify.removeAll(keepingCapacity: false)
        stagedWarmedSizes.removeAll(keepingCapacity: false)
        for layer in cache { (layer as? MambaCache)?.clearVerifyStaging() }
    }

    private mutating func changeBlockSize(from: Int, to: Int, settled: Bool) {
        adaptiveBlockSize = to
        for layer in cache { (layer as? MambaCache)?.clearVerifyStaging() }
        compiledVerify.removeAll(keepingCapacity: true)
        stagedWarmedSizes.removeAll(keepingCapacity: true)
        traceBlockChange(from: from, to: to, settled: settled)
    }

    private func traceBlockChange(from: Int, to: Int, settled: Bool) {
        guard Self.traceEnabled else { return }
        let rates = Self.blockLadder
            .map { String(format: "b%d=%.1f", $0, throughput(of: $0)) }
            .joined(separator: " ")
        FileHandle.standardError.write(Data(
            "[DFlash2] block \(from) -> \(to)\(settled ? " (settled)" : "") tok/verify-s: \(rates)\n"
                .utf8))
    }

    /// Single-token target step. Used when the remaining budget cannot
    /// fill a block and as the safety valve when acceptance cannot run.
    private mutating func runAutoregressiveStep() -> Bool {
        // A plain step reads and advances the same cache an in-flight
        // verify has already speculatively extended.
        abandonInFlightVerify()
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
        // Never persist a cache that still carries unverified draft rows.
        abandonInFlightVerify()
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
