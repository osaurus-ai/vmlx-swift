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

/// Process-wide memo of the hybrid safety-warmup verdict, keyed by model
/// instance. The 16-cycle probe otherwise re-runs on every request even
/// though its outcome is a property of the model, not the prompt — census
/// showed 84% of requests burning the probe just to land in AR fallback
/// again. Only the *warmup* verdict is memoized; adaptive acceptance-ratio
/// fallbacks stay per-request because they are content-dependent.
enum NativeMTPHybridWarmupMemo {
    private struct Entry {
        weak var model: AnyObject?
        let passed: Bool
    }

    private static let lock = NSLock()
    // `nonisolated(unsafe)` because the invariant is MANUAL, not static: every read and write below
    // is bracketed by `lock`. Swift 6 strict concurrency cannot see an NSLock discipline, so without
    // the annotation this is an error rather than the correctly-synchronised code it is.
    nonisolated(unsafe) private static var entries: [ObjectIdentifier: Entry] = [:]

    static func verdict(for model: AnyObject) -> Bool? {
        lock.lock()
        defer { lock.unlock() }
        let key = ObjectIdentifier(model)
        guard let entry = entries[key] else { return nil }
        // A deallocated model can hand its address to a new instance; the
        // weak reference detects that and drops the stale verdict rather
        // than skipping a safety probe the new model never ran.
        guard entry.model === model else {
            entries[key] = nil
            return nil
        }
        return entry.passed
    }

    static func record(_ passed: Bool, for model: AnyObject) {
        lock.lock()
        defer { lock.unlock() }
        entries[ObjectIdentifier(model)] = Entry(model: model, passed: passed)
    }
}

/// Structured per-generation diagnostics for native-MTP speculative decoding.
///
/// Carries the headline engagement/acceptance counters of the `[NativeMTP]`
/// summary line the iterator writes to stderr at the end of every generation.
/// Each field documents the `key=` entry it is assigned from; the assignment
/// reads the same source values, at the same lifecycle point, as the
/// `String(format:)` that renders the line, but representation can differ
/// where a field's doc says so (rounding, `nil` vs `none`, dense histogram).
/// The struct is a subset: the stderr line remains the complete diagnostic
/// surface and additionally carries repair/commit counters
/// (`residualCorrection=`, `prefixCommit=`, `rollbackRepair=`,
/// `mtpCacheRefresh=`), forward-pass counters, per-phase timing fields,
/// GDN-replay diagnostics, `phaseDiag=`, and `samplingMode=`, none of which
/// appear here. Hosts that only need these headline counters can read them
/// from ``GenerateCompletionInfo/nativeMTPStats`` instead of parsing stderr;
/// that property is `nil` for any generation that did not run the native-MTP
/// iterator.
public struct NativeMTPGenerationStats: Sendable, Equatable {
    /// Draft depth actually in effect at the start of generation (`depth=`).
    ///
    /// This is the REQUESTED depth after the policy cap, not the raw request:
    /// the iterator clamps to `VMLX_MTP_DEPTH_CAP` (default `2` — the D2
    /// ceiling; depths past 2 only ever won on a deterministic counting
    /// prompt, and Nemotron measured D3 at 0.48x on real prose). A host that
    /// asks for `.nativeMTP(depth: 3)` therefore sees `depth == 2` here —
    /// which is the point of surfacing it, since that gap is otherwise
    /// invisible without reading stderr.
    public let depth: Int

    /// Draft depth in effect at end of generation, after any adaptive
    /// downshifts (`activeDepth=`).
    public let activeDepth: Int

    /// Number of verify cycles executed (`verifyCalls=`).
    public let verifyCalls: Int

    /// The `generatedTokenIds.count` the generate loop passed to
    /// `storeCacheAfterGeneration` — identical to the stderr `outputTokens=`
    /// value. Because a generation-ending stop token is counted by the loop
    /// but never appended to `generatedTokenIds`, this can be one less than
    /// ``GenerateCompletionInfo/generationTokenCount`` on stop-token paths
    /// (`outputTokens=`).
    public let outputTokens: Int

    /// Tokens produced by the autoregressive fallback path
    /// (`arFallbackTokens=`).
    public let arFallbackTokens: Int

    /// Histogram of verify cycles by accepted draft count:
    /// `acceptedByDepth[n]` is the number of cycles that accepted exactly `n`
    /// draft tokens. Dense representation of the same counts the stderr line
    /// prints sparsely as non-zero `n:count` pairs — and as the literal
    /// `none` when the histogram is empty, which here is `[]`
    /// (`acceptedByDepth=`).
    public let acceptedByDepth: [Int]

    /// Bonus tokens sampled from the target after full acceptance (`bonus=`).
    public let bonusTokens: Int

    /// Rejected draft tokens (`rejected=`).
    public let rejectedTokens: Int

    /// Average committed tokens per verify cycle over the speculative output.
    /// Stores the same source double the stderr line prints, at full
    /// precision — the line renders it rounded to two decimals via `%.2f`
    /// (`avgCommittedPerVerify=`).
    public let avgCommittedPerVerify: Double

    /// Average draft acceptance probability; 0 under greedy sampling. Stores
    /// the same source double the stderr line prints, at full precision — the
    /// line renders it rounded to three decimals via `%.3f` (`avgAcceptP=`).
    public let avgAcceptProbability: Double

    /// Number of adaptive depth downshifts (`adaptiveDownshifts=`).
    public let adaptiveDownshifts: Int

    /// Why the adaptive controller fell back to autoregressive decode, or
    /// `nil` when it did not. `nil` here corresponds exactly to the stderr
    /// line printing `adaptiveFallback=none`; a non-`nil` reason is printed
    /// verbatim (`adaptiveFallback=`).
    public let adaptiveFallbackReason: String?

    /// Verifier mode used for this generation (`verifierMode=`).
    public let verifierMode: String

    /// Cache handling mode (`cacheMode=`).
    public let cacheMode: String

    public init(
        depth: Int,
        activeDepth: Int,
        verifyCalls: Int,
        outputTokens: Int,
        arFallbackTokens: Int,
        acceptedByDepth: [Int],
        bonusTokens: Int,
        rejectedTokens: Int,
        avgCommittedPerVerify: Double,
        avgAcceptProbability: Double,
        adaptiveDownshifts: Int,
        adaptiveFallbackReason: String?,
        verifierMode: String,
        cacheMode: String
    ) {
        self.depth = depth
        self.activeDepth = activeDepth
        self.verifyCalls = verifyCalls
        self.outputTokens = outputTokens
        self.arFallbackTokens = arFallbackTokens
        self.acceptedByDepth = acceptedByDepth
        self.bonusTokens = bonusTokens
        self.rejectedTokens = rejectedTokens
        self.avgCommittedPerVerify = avgCommittedPerVerify
        self.avgAcceptProbability = avgAcceptProbability
        self.adaptiveDownshifts = adaptiveDownshifts
        self.adaptiveFallbackReason = adaptiveFallbackReason
        self.verifierMode = verifierMode
        self.cacheMode = cacheMode
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
    /// True when this iterator warm-started from restored cache rows. The
    /// aligned head cache starts cold in that state, so early acceptance
    /// windows under-read — the adaptive controller widens its first
    /// judgment window 4× (the Python engine's restore-aware gate).
    private var restoredPrefixStart: Bool = false
    private var adaptiveDepthPromotionCount = 0
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

    // MARK: - Aligned head cache
    //
    // The MTP head drafts best when its KV holds exactly the pairs it was
    // trained on: (backbone_hidden_i, token_{i+1}) for every CONFIRMED
    // token, in order. This loop previously gave it neither. On accept it
    // retained a cache with a HOLE at every bonus token — the bonus never
    // passed through the head, so the next draft conditioned on a history
    // missing a token. On reject it threw the cache away entirely and
    // re-drafted from a single fused pair. Both starve the head of context.
    //
    // Aligned mode commits every token confirmed this cycle through the
    // head in the SAME forward that drafts the next one: `makeDrafts`
    // already samples from the LAST position, so a multi-token commit is
    // free — no extra head call. Drafts from deeper levels are appended
    // speculatively, so they are trimmed before each commit; one of them
    // may carry a rejected token, and they are built from the head's own
    // post-norm hidden rather than the backbone's.
    //
    // Ported from the Python engine, which measured acceptance 74.8% ->
    // 92.0% and 22.1 -> 36.1 tok/s on Qwen3.8-27B with no kernel work.
    // `VMLX_MTP_ALIGNED_HEAD_CACHE=0` restores the old behaviour.
    static let alignedHeadCacheEnabled: Bool = {
        let raw = ProcessInfo.processInfo.environment["VMLX_MTP_ALIGNED_HEAD_CACHE"]?
            .trimmingCharacters(in: .whitespaces).lowercased()
        return !["0", "false", "no", "off"].contains(raw ?? "")
    }()

    /// `VMLX_MTP_PHASE_TIMERS=1` adds a forced eval of the verify logits so
    /// the `targetVerifySec` span splits into host graph-build vs GPU wait
    /// (`verifyGpuWaitSec`). Attribution-only: the extra sync point changes
    /// pipelining, so totals from a timed run overstate the untimed cost.
    static let phaseTimersEnabled =
        ProcessInfo.processInfo.environment["VMLX_MTP_PHASE_TIMERS"] == "1"

    /// Head-cache rows appended by deeper draft levels last cycle. Trimmed
    /// before the next commit so an unverified draft can never persist.
    private var headChainPairs = 0

    /// Static so callers can trim without holding a mutating borrow on
    /// `self` across the surrounding expression (the caches are reference
    /// types, so the rows really are dropped).
    private static func trimHeadChain(_ cache: [KVCache], rows: Int) {
        guard rows > 0 else { return }
        for layer in cache where layer.isTrimmable {
            _ = layer.trim(rows)
        }
    }
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
    private(set) var stagedVerifierCommitCount = 0
    private(set) var targetVerifyTime: TimeInterval = 0
    private(set) var verifyGpuWaitTime: TimeInterval = 0

    // MARK: compiled staged verify
    //
    // The staged verify forward is a fixed shape (1 primary + depth
    // drafts), so it compiles exactly like plain decode: attention caches
    // become Compilable static buffers with a graph-visible offset, GDN
    // layers write their fixed staging slots in place (the staged mode was
    // designed for this — committed state is untouched inside the trace),
    // and the host-side staged commit runs outside the trace unchanged.
    // OPT-IN (`VMLX_MTP_COMPILED_VERIFY=1`): this iterator is a struct and
    // compiled traces capture the cache array, so copies retain traces —
    // the same teardown hazard DFlash2's compiled verify documents.
    /// Compiled verify forwards keyed by row count (depth+1). The adaptive
    /// controller moves between at most three depths, so at most three
    /// traces are ever built.
    private var compiledVerify: [Int: @Sendable ([MLXArray]) -> [MLXArray]] = [:]
    /// Row counts whose first staged cycle already ran eagerly. compile()
    /// needs the staging slots to exist as persistent objects before the
    /// trace, so the first cycle at each shape warms them.
    private var compiledVerifyWarmedSizes: Set<Int> = []
    private var compiledVerifyEnabled = false
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
    private(set) var nativeMTPStats: NativeMTPGenerationStats?
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
                "native MTP is enabled only for text-only requests with no active penalties, no suppress/reasoning-budget processors, and an unbounded KV window; sampled requests run the exact-pq accept path")
        }

        self.model = model
        self.cache = cache ?? model.newCache(parameters: effectiveParameters)
        self.mtpCache = model.makeNativeMTPCache()
        self.cacheCoordinator = cacheCoordinator
        self.processor = effectiveParameters.processor()
        self.sampler = effectiveParameters.sampler()
        self.speculativeSampler = SpeculativeSamplingController(parameters: effectiveParameters)
        self.maxTokens = effectiveParameters.maxTokens
        // Depth policy: D3 is the ceiling. The old D2 cap was calibrated on
        // the lazy-repair/sequential verifier, where every rejection cost a
        // checkpoint restore + full replay forward (Nemotron D3 0.48x, the
        // Python depth-3 prose figure 14% slower than AR). Under the staged
        // verifier those numbers no longer describe this code: 2026-08-19 on
        // Qwen3.8-27B-JANG_4D, D3 measured 28.4 tok/s vs D2 27.5 vs plain
        // 16.5, byte-identical output, 3.13 committed tokens per verify.
        // A default cap below what a measured tuning artifact requests is a
        // silent clamp on the host's explicit choice — the same bug this
        // comment already records for the cap=1 era. The adaptive controller
        // still downshifts unprofitable depth at runtime; benchmarks can
        // override via VMLX_MTP_DEPTH_CAP.
        let depthCap =
            ProcessInfo.processInfo.environment["VMLX_MTP_DEPTH_CAP"]
            .flatMap(Int.init).map { Swift.max($0, 1) } ?? 3
        self.depth = Swift.min(requestedDepth, depthCap)
        self.currentDepth = Swift.min(requestedDepth, depthCap)
        self.verifierModeSetting = effectiveParameters.draftStrategy?.nativeMTPVerifierMode
        let promptTokenStart = Date.timeIntervalSinceReferenceDate
        let promptTokenIds = input.text.tokens.reshaped(-1).asArray(Int.self)
        let promptTokenElapsed = Date.timeIntervalSinceReferenceDate - promptTokenStart
        self.promptTokenIds = promptTokenIds
        self.cachePrefixTokenCounts = input.cachePrefixTokenCounts
        self.restoredPrefixStart = input.cachePrefixTokenCounts.contains { $0 > 0 }
        self.originalInput = input
        self.cacheInitParameters = effectiveParameters
        self.mediaSalt = computeCacheSalt(for: input, parameters: effectiveParameters)
        self.materializeSyncTime += promptTokenElapsed

        if usesHybridMambaCache,
            let remembered = NativeMTPHybridWarmupMemo.verdict(for: model)
        {
            if remembered {
                hybridSafetyWarmupComplete = true
            } else {
                forceAutoregressiveFallback = true
                adaptiveFallbackReason = "hybrid_warmup_memo"
            }
        }

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
            let requiresDiskBackedRestore =
                cacheRequiresDiskBackedCoordinatorRestore(self.cache)
            let result: CacheFetchResult
            if requiresDiskBackedRestore,
               input.cacheRestorePolicy == .freshRequiredToolSelection
            {
                result = .miss
            } else {
                result = coordinator.fetch(
                    tokens: cacheLookupTokenIds,
                    mediaSalt: mediaSalt,
                    preferredDiskBoundaries: originalInput.cacheStablePrefixTokenCounts
                )
            }
            switch result {
            case .hit(
                let matchedTokens, let remainingTokens, _, let blocks,
                let ssmStates, let diskArrays):
                var restored = false
                var retainedDiskRestore = false
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
                            retainedDiskRestore = diskArrays != nil
                        }
                    } else {
                        let remainingArray = MLXArray(remainingTokens.map { Int32($0) })
                            .expandedDimensions(axis: 0)
                        inputForPrepare = LMInput(text: LMInput.Text(tokens: remainingArray))
                        retainedDiskRestore = diskArrays != nil
                    }
                    if retainedDiskRestore {
                        coordinator.touchStableDiskCheckpointsAfterRetainedRestore(
                            requestTokens: cacheLookupTokenIds,
                            matchedTokenCount: matchedTokens,
                            preferredDiskBoundaries: originalInput
                                .cacheStablePrefixTokenCounts,
                            skipExactDiskBoundary: false,
                            mediaSalt: mediaSalt)
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

        // MARK: compiled verify promotion — after prefill and the boundary
        // snapshot, so stored prefix-cache entries stay plain.
        if speculativeSampler.isGreedy, self.processor == nil,
            usesHybridMambaCache,
            model is DFlash2StagedVerifyRollbackModel,
            HardwareInfo.isCompiledDecodeSupported,
            ProcessInfo.processInfo.environment["VMLX_MTP_COMPILED_VERIFY"] == "1"
        {
            let promptOffset = self.cache.map(\.offset).max() ?? promptTokenIds.count
            let bufferLength = promptOffset + (self.maxTokens ?? 4096) + self.depth + 8
            self.cache = self.cache.map { layer in
                if !(layer is CompilableKVCache), layer is KVCacheSimple {
                    return CompilableKVCache(from: layer, maxLength: bufferLength)
                }
                return layer
            }
            MLX.eval(self.cache)
            self.compiledVerifyEnabled = true
        }
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
        // Compiled traces capture the cache array; they must not outlive
        // the generation they were built for.
        releaseCompiledVerify()
        if let coordinator = cacheCoordinator,
           !promptTokenIds.isEmpty,
           let promptCacheSnapshot
        {
            // Shared with the solo and batched engines — see the note there.
            let sharedPromptStripBoundary = TokenIterator.hybridStripBoundaryIndex(
                coordinator: coordinator,
                promptTokenIds: promptTokenIds,
                input: originalInput)
            let isReusablePrefixWarmup =
                originalInput.cachePromptIntent == .reusablePrefixWarmup
            let shouldPersistExactWarmupPrompt = shouldPersistExactPromptBoundary(
                cachePromptIntent: originalInput.cachePromptIntent,
                requiresRecurrentSSMCompanion:
                    coordinator.requiresRecurrentSSMCompanion)
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

            if shouldPersistExactWarmupPrompt {
                store(
                    tokens: promptTokenIds,
                    snapshot: promptCacheSnapshot,
                    label: "prompt-boundary")
            }

            if !originalInput.requiresPostPrepareCacheKey {
                for boundary in Set(cachePrefixTokenCounts).sorted()
                where boundary > 0 && boundary < promptTokenIds.count {
                    let isStableBoundary = originalInput
                        .cacheStablePrefixTokenCounts.contains(boundary)
                    let storeBoundary = isStableBoundary
                        && coordinator.requiresRecurrentSSMCompanion && boundary > 1
                        ? boundary - 1
                        : boundary
                    let boundaryTokens = Array(promptTokenIds.prefix(storeBoundary))
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
                        allowDiskBackedRederive: shouldForceStableBoundaryRederive(
                            isStableBoundary: isStableBoundary,
                            isReusablePrefixWarmup: isReusablePrefixWarmup,
                            requiresRecurrentSSMCompanion:
                                coordinator.requiresRecurrentSSMCompanion))
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

            if !isReusablePrefixWarmup,
               includeGeneratedBoundary,
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
            verifierMode = stagedVerifierCommitCount > 0
                ? "input_capture_staged"
                : chunkReplayRepairCount > 0
                ? "chunk_repair"
                : NativeMTPVerifierStatePolicy.mode(for: verifierModeSetting) == .lazyRepair
                ? "chunk_lazy_repair"
                : "chunk_commit"
        } else {
            verifierMode = "sequential_repair"
        }
        // Structured snapshot of the headline counters of the stderr summary
        // line below: each stats field is assigned here from the same source
        // value the corresponding `key=` entry prints, at this same point.
        // Representation differs where documented (the line rounds
        // `avgCommittedPerVerify`/`avgAcceptP`, prints a nil fallback reason
        // as `none`, and renders the histogram sparsely); the line's
        // repair/forward/timing/diagnostic keys have no struct counterpart.
        // `generateLoopTask` reads this after `storeCacheAfterGeneration`
        // returns and carries it on `GenerateCompletionInfo.nativeMTPStats`.
        let acceptedHistogram: [Int] = {
            guard let maxAccepted = acceptedByDepth.keys.max() else { return [] }
            return (0...maxAccepted).map { acceptedByDepth[$0] ?? 0 }
        }()
        nativeMTPStats = NativeMTPGenerationStats(
            depth: depth,
            activeDepth: currentDepth,
            verifyCalls: verifyCalls,
            outputTokens: generatedTokenIds.count,
            arFallbackTokens: autoregressiveFallbackTokenCount,
            acceptedByDepth: acceptedHistogram,
            bonusTokens: bonusCount,
            rejectedTokens: rejectedCount,
            avgCommittedPerVerify: avgCommitted,
            avgAcceptProbability: avgAcceptP,
            adaptiveDownshifts: adaptiveDepthDownshiftCount,
            adaptiveFallbackReason: adaptiveFallbackReason,
            verifierMode: verifierMode,
            cacheMode: "private-mtp+verifier-prefix-commit")
        let line = String(
            format:
                "[NativeMTP] depth=%d activeDepth=%d verifyCalls=%d outputTokens=%d arFallbackTokens=%d acceptedByDepth=%@ bonus=%d rejected=%d residualCorrection=%d prefixCommit=%d rollbackRepair=%d mtpCacheRefresh=%d targetForwards=%d verifyInputTokens=%d repairForwards=%d seedMainForwards=%d verifyMainForwards=%d replayMainForwards=%d mtpForwards=%d avgCommittedPerVerify=%.2f avgAcceptP=%.3f adaptiveDownshifts=%d adaptiveFallback=%@ targetVerifySec=%.3f verifyGpuWaitSec=%.3f seedMainSec=%.3f verifyMainSec=%.3f replayMainSec=%.3f mtpDraftSec=%.3f samplingSec=%.3f cacheCommitSec=%.3f materializeSyncSec=%.3f cacheStateSec=%.3f iteratorWallSec=%.3f gdnReplayCalls=%d gdnReplayStates=%d gdnReplaySec=%.3f phaseDiag=%@ samplingMode=%@ verifierMode=%@ cacheMode=private-mtp+verifier-prefix-commit\n",
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
            verifyGpuWaitTime,
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

    /// Compile the fixed-shape staged verify forward. Built AFTER the
    /// eager warm-up cycle at this row count, so the staging slots exist
    /// as persistent objects the trace's state tracking can capture.
    private mutating func buildCompiledVerify(rows: Int) {
        let capturedModel = model
        let cacheRef = cache
        compiledVerify[rows] = compile(
            inputs: cacheRef, outputs: cacheRef
        ) { (args: [MLXArray]) -> [MLXArray] in
            // This body runs ONLY while MLX records a trace; each execution
            // is one (re)trace. A healthy compiled verify traces once per
            // (rows) shape — a line per cycle means captured state is being
            // rebound somewhere and every replay silently degrades to a
            // rebuild.
            if ProcessInfo.processInfo.environment["VMLX_MTP_PHASE_TIMERS"] == "1" {
                FileHandle.standardError.write(
                    Data("[NativeMTP] compiled verify TRACE build rows=\(rows)\n".utf8))
            }
            return CompiledDecodeTrace.withActive {
                NativeMTPVerifierStatePolicy.withVerifierMode(
                    NativeMTPVerifierStatePolicy.Mode.inputCaptureStaged.rawValue
                ) {
                    let out = capturedModel.nativeBackboneMTPVerifyForward(
                        args[0], cache: cacheRef)
                    return [out.logits, out.hiddenStates]
                }
            }
        }
    }

    /// Drop the compiled verify traces. Mirrors DFlash2's release: traces
    /// capture the cache array, so they must not outlive the generation.
    mutating func releaseCompiledVerify() {
        compiledVerify.removeAll(keepingCapacity: false)
        compiledVerifyWarmedSizes.removeAll(keepingCapacity: false)
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

        // Staged verify (the DFlash 2 rollback machinery): GDN layers write
        // verify inputs + final state into fixed cache staging slots and
        // leave committed state and offset untouched; after acceptance the
        // host commits exactly the accepted rows. No checkpoint restore, no
        // full-model replay forward on partial accepts — the lazy-repair tax
        // that grows with depth (17 replay forwards over 32 cycles at D3).
        // The mode string is passed task-locally around the verify forward
        // ONLY; it must never leak into prefill/seed/sequential forwards.
        let explicitHybridMode = Self.nativeMTPHybridVerifySetting(verifierModeSetting)
        let stagedCapable = usesHybridMambaCache
            && speculativeSampler.isGreedy
            && processor == nil
            && model is DFlash2StagedVerifyRollbackModel
        let stagedVerify = stagedCapable
            && (explicitHybridMode == nil
                || NativeMTPVerifierStatePolicy.mode(for: explicitHybridMode)
                    == .inputCaptureStaged)

        // An EXPLICIT verifier mode (per-request, tuning artifact, or env) is
        // the operator's decision — warmup exists to pick a safety mode
        // automatically when nobody chose one. A measured tuning artifact
        // ships `verifier_mode` validated in that mode, so forcing 16
        // sequential-priced cycles in front of it only re-creates the
        // "MTP is slower" overhead the artifact already paid to rule out.
        // Staged verify skips warmup entirely: an unaccepted row can never
        // reach committed state, so the corruption class warmup guards
        // against is structurally gone, and the adaptive controller still
        // bails to AR when acceptance is hopeless.
        let shouldUseHybridSafetyWarmup = usesHybridMambaCache
            && speculativeSampler.isGreedy
            && processor == nil
            && !hybridSafetyWarmupComplete
            && explicitHybridMode == nil
            && !stagedVerify

        if !stagedVerify,
            shouldUseHybridSafetyWarmup || Self.requiresSequentialVerifierRepair(
                cache,
                speculativeSampler: speculativeSampler,
                verifierMode: verifierModeSetting)
        {
            try verifyCycleSequential(primary: primary)
            return
        }

        let requested = [primary] + drafts
        // ONE batched materialization for every id this cycle needs. The old
        // per-element `.item()` map cost one full pipeline drain per token,
        // and the replay/audit/pending paths below each re-materialized the
        // same ids again — 6-9 drains per verify cycle, which the sustained-D1
        // measurement showed dominating the whole cycle cost (5.4s of
        // materialize sync against 3.1s of actual forwards over 154 cycles).
        let requestedInputIds = recordMaterializeSync {
            stacked(requested.map { $0.reshaped(-1) }).asArray(Int32.self)
        }
        let input = MLXArray(requestedInputIds).reshaped(1, requested.count)
        let replayChunkCommit = !stagedVerify
            && Self.requiresChunkTokenReplayRepair(
                cache,
                verifierMode: verifierModeSetting)
        let lazyChunkRepair = !stagedVerify
            && Self.requiresLazyChunkRepair(
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
        let forwardVerifierMode = stagedVerify
            ? NativeMTPVerifierStatePolicy.Mode.inputCaptureStaged.rawValue
            : verifierModeSetting
        let verifier: NativeMTPForwardResult
        if stagedVerify, compiledVerifyEnabled,
            compiledVerifyWarmedSizes.contains(requested.count)
        {
            if compiledVerify[requested.count] == nil {
                buildCompiledVerify(rows: requested.count)
            }
            let outs = compiledVerify[requested.count]!([input])
            verifier = NativeMTPForwardResult(logits: outs[0], hiddenStates: outs[1])
            // Dispatch the replayed graph NOW instead of letting it drain in
            // one lump at the acceptance readback: measured 21.2-23.8 tok/s
            // without this dispatch vs 26.4-27.6 with a (blocking) eval here
            // — the submit is what mattered, so use the non-blocking form.
            asyncEval(verifier.logits, verifier.hiddenStates)
        } else {
            verifier = NativeMTPVerifierStatePolicy.withVerifierMode(forwardVerifierMode) {
                model.nativeBackboneMTPVerifyForward(input, cache: cache)
            }
            // Same submit-now rule the compiled branch measured (21.2-23.8
            // tok/s draining in one lump vs 26.4+ dispatching here): without
            // this, the whole verify graph sits unscheduled until the
            // acceptance readback inside `verifyDrafts` forces it, and the
            // GPU idles through all the host-side branch setup in between.
            asyncEval(verifier.logits, verifier.hiddenStates)
            if stagedVerify, compiledVerifyEnabled {
                compiledVerifyWarmedSizes.insert(requested.count)
            }
        }
        let verifyElapsed = Date.timeIntervalSinceReferenceDate - verifyStart
        targetVerifyTime += verifyElapsed
        verifyMainForwardTime += verifyElapsed
        if Self.phaseTimersEnabled {
            let gpuStart = Date.timeIntervalSinceReferenceDate
            MLX.eval(verifier.logits)
            verifyGpuWaitTime += Date.timeIntervalSinceReferenceDate - gpuStart
        }
        targetForwardCount += 1
        verifyMainForwardCount += 1
        verifyInputTokenCount += requested.count
        chunkVerifierCount += 1

        let sampleStart = Date.timeIntervalSinceReferenceDate
        guard let verifyDecision = Self.verifyDrafts(
            logits: verifier.logits,
            drafts: drafts,
            draftTokenIds: requestedInputIds.dropFirst().map(Int.init),
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
            if stagedVerify {
                for layer in cache { (layer as? MambaCache)?.clearVerifyStaging() }
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
                // Host ids already materialized once at cycle start.
                let acceptedInputIds = Array(requestedInputIds.prefix(count))
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
                    let draftID = Int(requestedInputIds[auditedAccepted + 1])
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

        for (index, token) in drafts.prefix(accepted).enumerated() {
            processor?.didSample(token: token)
            pendingTokens.append(Int(requestedInputIds[1 + index]))
        }

        if requiresSequentialRepair && accepted > 0 {
            guard let checkpoint else {
                throw NativeMTPRuntimeError.verifierCacheCommitFailed
            }
            let restoreStart = Date.timeIntervalSinceReferenceDate
            checkpoint.restore(into: &cache)
            cacheSnapshotRestoreTime += Date.timeIntervalSinceReferenceDate - restoreStart

            let acceptedInputIds = Array(requestedInputIds.prefix(accepted + 1))
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
        let committedCache: Bool
        if stagedVerify {
            // Staged verify left recurrent state and offsets untouched;
            // EVERY cycle commits from the staging slots (a full accept
            // adopts the staged final state — no replay forward, ever).
            // Attention caches advanced in-graph and just rewind.
            let rejectedRows = requested.count - committedInputCount
            if rejectedRows > 0 {
                for layer in cache where layer.isTrimmable {
                    _ = layer.trim(rejectedRows)
                }
            }
            guard let stagedModel = model as? DFlash2StagedVerifyRollbackModel,
                stagedModel.commitStagedVerifiedBlock(
                    cache: cache,
                    acceptedInputs: committedInputCount,
                    blockLength: requested.count)
            else {
                throw NativeMTPRuntimeError.verifierCacheCommitFailed
            }
            committedCache = true
            stagedVerifierCommitCount += 1
        } else {
            committedCache = repairedHiddenForNextMTP != nil
                ? true
                : canCommitVerifierCache
                ? Self.commitVerifierCache(
                    &cache,
                    committedInputCount: committedInputCount,
                    totalInputCount: requested.count)
                : false
        }
        cacheCommitTime += Date.timeIntervalSinceReferenceDate - commitStart
        if committedCache {
            prefixCommitCount += 1
        }

        let nextToken: MLXArray
        let hiddenForNextMTP: MLXArray
        // Aligned mode feeds the head EVERY token confirmed this cycle, so
        // it needs the confirmed tokens as a sequence rather than just the
        // final one. Built below in each branch and consumed by makeDrafts.
        var alignedCommitTokens: MLXArray?
        var alignedCommitHidden: MLXArray?
        if accepted == drafts.count {
            bonusCount += 1
            let bonus = nextVerifiedToken
            processor?.didSample(token: bonus)
            // verifyDrafts already materialized every sampled id in ONE
            // batched readback; calling .item() here drains the pipeline a
            // second time for a value we are holding. At ~10ms a drain on a
            // 27B that was a quarter of the cycle.
            pendingTokens.append(verifyDecision.targetTokenIds[drafts.count])
            nextToken = bonus
            hiddenForNextMTP = repairedHiddenForNextMTP
                ?? verifier.hiddenStates[0..., drafts.count ..< (drafts.count + 1), 0...]
            // Full accept: commit (h0,d1) … (h_{k-1},dk), (hk,bonus). The
            // bonus pair is the one the old retained cache always dropped.
            if Self.alignedHeadCacheEnabled, repairedHiddenForNextMTP == nil {
                Self.trimHeadChain(mtpCache, rows: headChainPairs)
                headChainPairs = 0
                // Copy out of `self` first: recordMaterializeSync is
                // mutating, so a closure reading self.drafts overlaps it.
                // Draft ids were materialized once at cycle start
                // (requestedInputIds) and the bonus id came back with the
                // batched target sample — no extra drain needed here.
                let ids = requestedInputIds.dropFirst().map { $0 }
                    + [Int32(verifyDecision.targetTokenIds[drafts.count])]
                alignedCommitTokens = MLXArray(ids).reshaped(1, ids.count)
                alignedCommitHidden =
                    verifier.hiddenStates[0..., 0 ..< ids.count, 0...]
            }
        } else {
            rejectedCount += 1
            if !speculativeSampler.isGreedy {
                residualCorrectionCount += 1
            }

            let correction = nextVerifiedToken
            processor?.didSample(token: correction)
            pendingTokens.append(verifyDecision.targetTokenIds[accepted])
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

            // Partial accept: the confirmed prefix is (h0,d1) … (h_{a-1},da)
            // followed by (ha, correction). Trim the speculative chain — one
            // of those rows carries the REJECTED draft — then commit the
            // confirmed pairs with backbone hiddens. Recreating the cache
            // here is what the old path did, and it is exactly the context
            // loss that held acceptance down.
            if Self.alignedHeadCacheEnabled, repairedHiddenForNextMTP == nil, committedCache {
                Self.trimHeadChain(mtpCache, rows: headChainPairs)
                headChainPairs = 0
                let ids = Array(requestedInputIds.dropFirst().prefix(accepted))
                    + [Int32(verifyDecision.targetTokenIds[accepted])]
                alignedCommitTokens = MLXArray(ids).reshaped(1, ids.count)
                alignedCommitHidden =
                    verifier.hiddenStates[0..., 0 ..< ids.count, 0...]
            } else {
                mtpCache = model.makeNativeMTPCache()
                mtpCacheRefreshCount += 1
            }
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
            hidden: alignedCommitHidden ?? hiddenForNextMTP,
            nextToken: alignedCommitTokens ?? nextToken,
            mtpCache: mtpCache,
            depth: currentDepth,
            sampler: sampler,
            speculativeSampler: speculativeSampler,
            processor: processor)
        drafts = draftBatch.tokens
        draftProbabilities = draftBatch.probabilities
        // Levels beyond the first append speculative rows to the head
        // cache; record how many so the next cycle trims them before
        // committing confirmed pairs over the top.
        headChainPairs = Self.alignedHeadCacheEnabled
            ? Swift.max(0, draftBatch.tokens.count - 1) : 0
        mtpForwardCount += draftBatch.forwardCount
        materializeSyncTime += draftBatch.materializeSyncTime
        mtpDraftTime += Date.timeIntervalSinceReferenceDate - draftStart
    }

    /// Measurement-only escape hatch: `VMLX_NATIVE_MTP_DISABLE_ADAPTIVE=1`
    /// keeps the adaptive controller from bailing or downshifting, so a bench
    /// can measure SUSTAINED speculation cost at a fixed depth. The shipped
    /// artifacts' floors are calibrated from exactly these runs; without the
    /// hatch, every hybrid leg measured "12 cycles of trying, then AR" and the
    /// steady-state number did not exist.
    private static let adaptiveDisabledForMeasurement =
        ProcessInfo.processInfo.environment["VMLX_NATIVE_MTP_DISABLE_ADAPTIVE"] == "1"

    private mutating func recordAdaptiveCycle(accepted: Int) {
        if Self.adaptiveDisabledForMeasurement {
            // The hatch must not leave the safety warmup permanently
            // incomplete: that silently priced EVERY cycle as a sequential
            // per-token verify and fabricated "MTP got slower with the
            // aligned cache" (2026-08-19). The hatch's whole point is
            // steady-state cost, so complete warmup by cycle count alone.
            if !hybridSafetyWarmupComplete, verifyCalls >= Self.hybridWarmupCycleCount {
                hybridSafetyWarmupComplete = true
            }
            return
        }
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
            let warmupFloor =
                Self.hybridWarmupMinimumAverageAcceptedPerDraft * Double(currentDepth)
            if averageAccepted >= warmupFloor {
                hybridSafetyWarmupComplete = true
                NativeMTPHybridWarmupMemo.record(true, for: model)
            } else {
                // The memo's contract is "a property of the model, not the
                // prompt" — but this floor is an acceptance criterion, and
                // acceptance is content. Memoize the failure only when it is
                // catastrophic (below even the depth-1 floor), which is the
                // broken/mismatched-head case the memo exists for. A miss
                // above that line means THIS prompt drafted badly (prose
                // warms up around 1.0, code above 3.0 on the same model);
                // fall back for this request and let the next one re-probe,
                // or a prose first turn locks every later code turn out of
                // MTP for the whole residency.
                if Self.warmupFailureIsModelProperty(averageAccepted: averageAccepted) {
                    NativeMTPHybridWarmupMemo.record(false, for: model)
                }
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
        // Staged verify changes the economics the legacy floors were tuned
        // for: a rejection no longer costs a checkpoint restore + full
        // replay forward, just an attention trim + fused staged commit.
        // Break-even per-draft acceptance from the 2026-08-19 staged cycle
        // costs (d1 82ms, d2 92ms, d3 110ms vs 59ms plain step on
        // Qwen3.8-27B): d1 0.39, d2 0.44, d3 0.52 — the floors below keep
        // a safety margin above break-even. Measured d3 ran 0.72 at
        // equal-or-better wall speed than d2; the legacy 0.85 floor would
        // have downshifted it.
        let staged = stagedVerifierCommitCount > 0
        let depthThreeFloor = staged ? 0.60 : Self.depthThreeMinimumAcceptanceRatio
        let depthTwoFloor = staged ? 0.50 : Self.depthTwoMinimumAcceptanceRatio
        let depthOneFloor = staged ? 0.40 : Self.depthOneMinimumAcceptanceRatio
        // A restored prefix arrives with a cold aligned-head cache; its first
        // windows under-read acceptance and a demote here sticks (see re-arm
        // below, but avoid the churn entirely): give restored sessions 4×
        // the sample budget before any demote judgment.
        let inRestoredGraceWindow =
            restoredPrefixStart && verifyCalls < 4 * Self.adaptiveWindowSize
        if currentDepth >= 3, acceptanceRatio < depthThreeFloor, !inRestoredGraceWindow {
            currentDepth = 2
            adaptiveDepthDownshiftCount += 1
            adaptiveWindow.removeAll(keepingCapacity: true)
            mtpCache = model.makeNativeMTPCache()
            mtpCacheRefreshCount += 1
            return
        }

        if currentDepth == 2, acceptanceRatio < depthTwoFloor, !inRestoredGraceWindow {
            // A failing D2 downshifts to D1 first — D1's breakeven is far
            // lower, so "D2 doesn't pay" is not evidence that speculation
            // itself doesn't.
            currentDepth = 1
            adaptiveDepthDownshiftCount += 1
            adaptiveWindow.removeAll(keepingCapacity: true)
            mtpCache = model.makeNativeMTPCache()
            mtpCacheRefreshCount += 1
            return
        }

        if currentDepth == 1, acceptanceRatio < depthOneFloor, !inRestoredGraceWindow {
            enableAutoregressiveFallback(
                reason: String(
                    format: "adaptive_accept_ratio=%.2f_depth=%d",
                    acceptanceRatio,
                    currentDepth))
            return
        }

        // Re-arm. The controller could historically only demote, so a single
        // cold window — a restored prefix arriving with a cold aligned-head
        // cache — pinned the whole session below its configured depth. When a
        // full window at the CURRENT depth clears the floor of the depth
        // above with margin, promote one level; the window reset re-prices
        // the new depth on fresh samples.
        if currentDepth < depth {
            let nextFloor: Double
            switch currentDepth {
            case 1: nextFloor = depthTwoFloor
            default: nextFloor = depthThreeFloor
            }
            if acceptanceRatio >= nextFloor + Self.adaptivePromotionMargin {
                currentDepth += 1
                adaptiveDepthPromotionCount += 1
                adaptiveWindow.removeAll(keepingCapacity: true)
            }
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
        draftTokenIds: [Int],
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
                // Draft ids were materialized once at cycle start — no
                // per-draft pipeline drain here.
                guard sampledIDs[accepted] == draftTokenIds[accepted] else { break }
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
    /// Extra acceptance a lower depth must show over the higher depth's floor
    /// before the controller re-arms upward — enough hysteresis that a
    /// borderline window doesn't flap between depths.
    private static let adaptivePromotionMargin = 0.10
    private static let adaptiveMinimumSamplesPerDepth = 6
    private static let depthThreeMinimumAcceptanceRatio = 0.85
    private static let depthTwoMinimumAcceptanceRatio = 0.75
    /// D1 breakeven is far below D2's. A depth-1 cycle costs one 2-position
    /// verify forward (≈ one decode forward on a bandwidth-bound dense
    /// backbone) plus a 1-layer MTP head forward, and emits 1 + accept
    /// tokens — so speculation pays for itself well below 0.75 acceptance.
    /// The flat ≤2 floor of 0.75 was measured killing a profitable D1 on
    /// Qwen3.6-27B prose: accept ratio 0.67 (avgCommittedPerVerify 1.67)
    /// tripped `adaptive_accept_ratio` at exactly window size 12, before the
    /// 16-cycle hybrid warmup could ever complete, so every hybrid run paid
    /// 12 sequential-priced cycles and then fell back to AR — 29.9 tok/s vs
    /// 34.0 AR, the precise shape of the "MTP is slower" reports.
    private static let depthOneMinimumAcceptanceRatio = 0.5
    private static let hybridWarmupCycleCount = 16
    /// Per-DRAFT warmup floor. The old absolute floor (2.75 average accepted
    /// drafts per cycle) is unreachable below depth 3 — depth 1 caps at 1.0
    /// and depth 2 at 2.0 — so hybrids running the shipped D1/D2 policy could
    /// never complete warmup, never reach the chunk verifier, and (worse)
    /// recorded a permanent negative `NativeMTPHybridWarmupMemo` for the
    /// model. 0.55 × depth keeps the same strictness the 2.75 value expressed
    /// at its native depth (2.75/3 ≈ 0.92 was Nemotron-D3-era calibration,
    /// deliberately relaxed here to the D1-breakeven scale).
    private static let hybridWarmupMinimumAverageAcceptedPerDraft = 0.55

    /// Whether a warmup miss is bad enough to blame the MODEL rather than the
    /// prompt — i.e. safe to memoize in `NativeMTPHybridWarmupMemo`. Below the
    /// depth-1 floor no depth could ever clear warmup, which only a broken or
    /// mismatched MTP head produces; anything above is content variance and
    /// must stay per-request.
    static func warmupFailureIsModelProperty(averageAccepted: Double) -> Bool {
        averageAccepted < hybridWarmupMinimumAverageAcceptedPerDraft
    }

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
            // Dispatch, do not drain: the verify forward consumes these
            // on-graph and the NEXT cycle reads every draft id in one
            // batched readback. A per-level MLX.eval here stalled the whole
            // pipeline once per draft for a value nobody reads yet.
            let syncStart = Date.timeIntervalSinceReferenceDate
            asyncEval(draft.token, out.hiddenStates)
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
