// Copyright © 2026 Jinho Jang (eric@jangq.ai)
//
// Block-diffusion token iterator — drives a BlockDiffusionModel through the
// reference generation algorithm:
//
//   outer loop (per canvas):
//     1. encoder forward over uncommitted tokens (prompt on prefill, the
//        previous finalized canvas afterwards) → KV cache append
//     2. random canvas init, self-conditioning reset
//     3. inner denoising loop (maxDenoisingSteps..1):
//        decoder forward → temperature-scheduled logits → categorical
//        denoiser canvas → entropy-bound accept → renoise rejected →
//        stable+confident early stop → logits become next step's
//        self-conditioning signal
//     4. finalize argmax canvas, truncate after EOS, emit tokens
//
// Tokens stream through the standard generateTask pipeline, so reasoning
// parsing, tool-call parsing, and stop strings behave exactly as they do for
// autoregressive models. Prompt prefix caching (paged + disk tiers) goes
// through the same CacheCoordinator hooks the AR iterators use.
//
// NOTE: `MLX.eval` below is MLX's graph-materialization API (forces lazy
// tensor computation); it does not execute code and is unrelated to
// JavaScript/Python eval().
//
// Python reference: transformers
// src/transformers/models/diffusion_gemma/generation_diffusion_gemma.py

import Foundation
import MLX
import os

public struct BlockDiffusionTokenIterator: TokenIteratorProtocol {
    private static let logger = Logger(
        subsystem: "vmlx", category: "BlockDiffusionTokenIterator")

    let model: any BlockDiffusionModel
    var cache: [KVCache]
    let options: BlockDiffusionParameters
    let cacheCoordinator: CacheCoordinator?
    let mediaSalt: String?

    public let maxTokens: Int?
    public var tokenCount = 0
    public var promptPrefillTime: TimeInterval = 0
    public var promptTokenIds: [Int]
    /// Conversation-stable prompt boundaries from the processor — prefixes a
    /// FUTURE request will actually contain. Chat templates append
    /// generation-control tokens (e.g. the Gemma thought stub
    /// `<|channel>thought\n<channel|>`) to the active turn but omit them when
    /// the turn is re-rendered as history, so the full prompt boundary can
    /// never extension-match; these boundaries can.
    let cachePrefixTokenCounts: [Int]

    var promptCacheSnapshot: [KVCache]?
    private let cacheInitParameters: GenerateParameters

    /// Per-request PRNG key for the canvas init, renoise, and denoiser draws.
    /// Split-and-advanced on every draw so generation is deterministic for a
    /// given `randomSeed`. `nil` when the request carries no seed — the draws
    /// then fall back to the global RNG (prior behaviour). Without this the
    /// three `MLX.randInt`/`MLX.categorical` calls below used the global RNG
    /// unconditionally, so block-diffusion output was non-reproducible even
    /// with a seed set.
    private var randomKey: MLXArray?

    private var pendingTokens: [Int] = []
    private var pendingIndex = 0
    private var finished = false
    private var canvasesEmitted = 0
    private let maxNewCanvases: Int
    /// Finalized canvas awaiting its encoder append (run lazily at the start
    /// of the next cycle so the final canvas is never encoded needlessly).
    private var pendingEncoderCanvas: [Int32]?
    /// Number of generated tokens already committed to the encoder cache
    /// (whole prior canvases plus any end-of-turn tail commit).
    private var committedGeneratedTokens = 0
    private var stopper: StableConfidentStopper
    private var statsReported = false

    // Instrumentation
    private(set) var denoisingForwardCount = 0
    private(set) var encoderForwardCount = 0
    private var denoiseTime: TimeInterval = 0
    private var decoderForwardTime: TimeInterval = 0
    private var samplerTime: TimeInterval = 0
    private var prefixCacheRestoredTokens = 0
    private let iteratorStartTime = Date.timeIntervalSinceReferenceDate

    public init(
        input: LMInput,
        model: any BlockDiffusionModel,
        cache: [KVCache]? = nil,
        parameters: GenerateParameters,
        options: BlockDiffusionParameters,
        cacheCoordinator: CacheCoordinator? = nil,
        prefillProgressHandler: (@Sendable (PrefillProgress) -> Void)? = nil
    ) throws {
        let promptTokenIds = input.text.tokens.reshaped(-1).asArray(Int.self)
        guard !promptTokenIds.isEmpty else {
            throw BlockDiffusionModelError.emptyPrompt
        }
        let totalPromptTokens = promptTokenIds.count
        // The block-diffusion encoder prefill does not flow through the
        // autoregressive `prepare(...)` path that emits `.prefillProgress`
        // frames, so we emit them here directly. Without this the UI counter
        // sits frozen at `0/N` for the entire (potentially many-second) 26B
        // encoder prefill, which reads as "stuck". Frames are clamped to the
        // prompt length and reported monotonically by the caller's gate.
        prefillProgressHandler?(
            PrefillProgress(
                stage: .prefill, completedUnitCount: 0,
                totalUnitCount: totalPromptTokens, detail: "diffusion"))

        self.model = model
        self.options = options
        self.cache = cache ?? model.newCache(parameters: parameters)
        self.cacheCoordinator = cacheCoordinator
        self.promptTokenIds = promptTokenIds
        self.cachePrefixTokenCounts = input.cachePrefixTokenCounts
        self.cacheInitParameters = parameters
        self.randomKey = parameters.randomSeed.map { MLX.key($0) }
        self.mediaSalt = computeCacheSalt(for: input, parameters: parameters)
        self.stopper = StableConfidentStopper(
            stabilityThreshold: options.stabilityThreshold,
            confidenceThreshold: options.confidenceThreshold)

        let requestedTokens = parameters.maxTokens ?? options.maxNewTokens
        self.maxTokens = requestedTokens
        self.maxNewCanvases = Swift.max(
            1, (requestedTokens + options.canvasLength - 1) / options.canvasLength)

        // ---- Prompt prefix cache (paged + disk tiers) -------------------
        // Diffusion is simpler than AR here: a full prefix hit needs no
        // trim-and-replay because the decoder reads the cache directly —
        // the prompt-boundary state is exactly the state the canvas loop
        // wants.
        //
        // Rotating sliding-window layers cannot round-trip through paged KV
        // blocks (ring/rotation metadata is disk-serialized via LayerKind),
        // so the coordinator must skip the paged tier and serve hits from
        // the disk tier — same contract as the AR iterators.
        if let coordinator = cacheCoordinator,
            !coordinator.isPagedIncompatible,
            cacheCannotUsePagedCoordinatorRestore(self.cache)
        {
            if cacheCanUsePagedWithRotatingCompanion(self.cache) {
                coordinator.setPagedBoundaryCompanionRequired(true)
            } else {
                coordinator.setPagedIncompatible(true)
            }
        }
        var tokensToEncode = promptTokenIds
        if let coordinator = cacheCoordinator,
            !input.requiresPostPrepareCacheKey,
            !input.hasMediaContent,
            // Only consult the prefix cache when starting from an empty
            // cache — a live multi-turn cache (ChatSession) already holds
            // prior turns and must not be overwritten.
            self.cache.allSatisfy({ $0.offset == 0 })
        {
            switch coordinator.fetch(tokens: promptTokenIds, mediaSalt: mediaSalt) {
            case .hit(let matchedTokens, let remainingTokens, _, let blocks, _, let diskArrays):
                var restored = false
                if !blocks.isEmpty {
                    let restoredTokens = restoreLayerData(from: blocks, into: self.cache)
                    coordinator.release(blocks: blocks)
                    restored = restoredTokens > 0
                }
                if !restored, let diskArrays {
                    restored = restoreFromDiskArrays(diskArrays, into: &self.cache) > 0
                    if restored {
                        MLX.eval(self.cache)
                    }
                }
                // Validate the restore: every layer must sit exactly at the
                // matched boundary, otherwise rebuild from scratch.
                let offsets = self.cache.map(\.offset)
                if restored,
                    let first = offsets.first,
                    offsets.allSatisfy({ $0 == first }),
                    first == matchedTokens,
                    matchedTokens + remainingTokens.count == promptTokenIds.count
                {
                    tokensToEncode = remainingTokens
                    prefixCacheRestoredTokens = matchedTokens
                    if matchedTokens > 0 {
                        prefillProgressHandler?(
                            PrefillProgress(
                                stage: .cacheRestore,
                                completedUnitCount: matchedTokens,
                                totalUnitCount: totalPromptTokens,
                                detail: "diffusion"))
                    }
                } else if restored {
                    self.cache = model.newCache(parameters: parameters)
                    tokensToEncode = promptTokenIds
                }
            case .miss:
                break
            }
        }

        // ---- Encoder prefill ---------------------------------------------
        let prefillStart = Date.timeIntervalSinceReferenceDate

        // Multimodal prompts prefill single-shot from spliced embeddings:
        // image blocks attend bidirectionally inside the prompt, which a
        // chunk boundary through an image span would break (mirrors the
        // reference encoder's chunked-prefill policy). The prefix cache is
        // already skipped for media above.
        if input.hasMediaContent,
            let spliced = try model.encoderPromptEmbeddings(for: input)
        {
            model.encoderForward(
                embeddings: spliced.embeddings,
                cache: self.cache,
                visionBlockIds: spliced.visionBlockIds)
            encoderForwardCount += 1
            MLX.eval(self.cache)
            self.promptPrefillTime = Date.timeIntervalSinceReferenceDate - prefillStart
            // Media prefill is single-shot (one bidirectional encoder pass),
            // so there is no incremental progress to report — jump to complete.
            prefillProgressHandler?(
                PrefillProgress(
                    stage: .complete, completedUnitCount: totalPromptTokens,
                    totalUnitCount: totalPromptTokens, detail: "diffusion"))
            if cacheCoordinator != nil {
                self.promptCacheSnapshot = makePromptBoundaryCacheSnapshot(from: self.cache)
            }
            return
        }

        let stepSize = Swift.max(parameters.prefillStepSize, 1)
        var remaining = tokensToEncode[...]
        var encodedTokens = prefixCacheRestoredTokens
        while !remaining.isEmpty {
            let chunk = Array(remaining.prefix(stepSize))
            remaining = remaining.dropFirst(stepSize)
            let chunkTokens = MLXArray(chunk.map { Int32($0) }).expandedDimensions(axis: 0)
            model.encoderForward(chunkTokens, cache: self.cache)
            encoderForwardCount += 1
            MLX.eval(self.cache)
            encodedTokens += chunk.count
            prefillProgressHandler?(
                PrefillProgress(
                    stage: remaining.isEmpty ? .complete : .prefill,
                    completedUnitCount: encodedTokens,
                    totalUnitCount: totalPromptTokens, detail: "diffusion"))
            if !remaining.isEmpty {
                Memory.clearCache()
            }
        }
        self.promptPrefillTime = Date.timeIntervalSinceReferenceDate - prefillStart
        // Emit a terminal complete frame even when there was nothing to encode
        // (full prefix-cache hit: tokensToEncode empty) so the counter lands at N/N.
        if tokensToEncode.isEmpty {
            prefillProgressHandler?(
                PrefillProgress(
                    stage: .complete, completedUnitCount: totalPromptTokens,
                    totalUnitCount: totalPromptTokens, detail: "diffusion"))
        }

        if cacheCoordinator != nil {
            self.promptCacheSnapshot = makePromptBoundaryCacheSnapshot(from: self.cache)
        }
    }

    public mutating func next() -> Int? {
        if let maxTokens, tokenCount >= maxTokens {
            commitEmittedTailToCache()
            reportStatsOnce()
            return nil
        }

        while pendingIndex >= pendingTokens.count {
            if finished || canvasesEmitted >= maxNewCanvases {
                commitEmittedTailToCache()
                reportStatsOnce()
                return nil
            }
            runCanvasCycle()
        }

        let token = pendingTokens[pendingIndex]
        pendingIndex += 1
        tokenCount += 1
        return token
    }

    /// End-of-generation cache contract: the encoder cache must hold the
    /// prompt plus exactly the emitted reply (minus a trailing EOS, matching
    /// the AR iterator where the final sampled token is never fed back).
    /// Canvas cycles only commit FULL prior canvases, so the final canvas —
    /// and any EOS/maxTokens truncation — leaves a tail to encode here.
    /// Without this, multi-turn sessions reusing the live cache would lose
    /// the end of the assistant's reply.
    private mutating func commitEmittedTailToCache() {
        let emittedCount = Swift.min(tokenCount, pendingTokens.count)
        var kept = Array(pendingTokens.prefix(emittedCount))
        if let last = kept.last, options.eosTokenIds.contains(last) {
            kept.removeLast()
        }
        guard kept.count > committedGeneratedTokens else {
            pendingEncoderCanvas = nil
            return
        }
        let delta = kept[committedGeneratedTokens...].map { Int32($0) }
        let tokens = MLXArray(delta).expandedDimensions(axis: 0)
        model.encoderForward(tokens, cache: cache)
        encoderForwardCount += 1
        MLX.eval(cache)
        committedGeneratedTokens = kept.count
        pendingEncoderCanvas = nil
    }

    // MARK: - Canvas cycle

    /// Split the request PRNG key, storing the advanced half and returning a
    /// fresh subkey for one draw. `nil` (no seed) → global RNG, unchanged.
    private mutating func nextRandomKey() -> MLXArray? {
        guard let k = randomKey else { return nil }
        let (advanced, use) = MLX.split(key: k)
        randomKey = advanced
        return use
    }

    private mutating func runCanvasCycle() {
        // 1. Commit the previous finalized canvas to the encoder cache.
        if let previous = pendingEncoderCanvas {
            let tokens = MLXArray(previous).expandedDimensions(axis: 0)
            model.encoderForward(tokens, cache: cache)
            encoderForwardCount += 1
            MLX.eval(cache)
            committedGeneratedTokens += previous.count
            pendingEncoderCanvas = nil
        }

        let cycleStart = Date.timeIntervalSinceReferenceDate
        let canvasLength = options.canvasLength
        let vocabSize = model.diffusionVocabularySize

        // 2. Random canvas, reset self-conditioning and stopping state.
        let initKey = nextRandomKey()
        var canvas = MLX.randInt(0 ..< Int32(vocabSize), [1, canvasLength], key: initKey)
        var selfConditioning: MLXArray? = nil
        var argmaxIds = [Int32](repeating: 0, count: canvasLength)
        stopper.reset()

        // 3. Denoising loop (reverse diffusion: curStep counts down).
        for curStep in stride(from: options.maxDenoisingSteps, through: 1, by: -1) {
            let forwardStart = Date.timeIntervalSinceReferenceDate
            let logits = model.decoderForward(
                canvas: canvas, cache: cache, selfConditioningLogits: selfConditioning)
            // Materializing here splits decoder-forward time from the
            // sampler pipeline in the stats line; both stay on-GPU.
            MLX.eval(logits)
            decoderForwardTime += Date.timeIntervalSinceReferenceDate - forwardStart
            let samplerStart = Date.timeIntervalSinceReferenceDate

            let temperature = blockDiffusionTemperature(
                curStep: curStep, maxSteps: options.maxDenoisingSteps,
                tMin: options.tMin, tMax: options.tMax)
            let processed = logits.asType(.float32) / temperature

            // No-empty-response guard. Block-diffusion collapses terse prompts (e.g. boolq's
            // "Respond with one of: yes, no.") to an all-EOS first canvas → empty output, even though
            // the model answers correctly whenever it emits anything. On the FIRST canvas, forbid EOS
            // at position 0 so at least one content token is produced; later canvases end normally, so
            // a genuinely complete reply can still stop. Mask only the SAMPLED copy — self-conditioning
            // below keeps the UNMASKED `processed`, so the denoising dynamics stay natural and the
            // position-0 content token is the model's real runner-up. (-1e9, not -inf, keeps entropy finite.)
            var sampled = processed
            if canvasesEmitted == 0 {
                for eos in options.eosTokenIds {
                    sampled[0, 0, eos] = MLXArray(Float(-1e9))
                }
            }

            let denoiserKey = nextRandomKey()
            let denoiserCanvas = MLX.categorical(sampled, key: denoiserKey).asType(.int32)
            let argmaxCanvas = argMax(sampled, axis: -1).asType(.int32)
            let entropy = canvasTokenEntropy(processedLogits: sampled)
            let acceptMask = entropyBoundAcceptMask(
                tokenEntropy: entropy, entropyBound: options.entropyBound)

            // Accepted positions adopt the denoiser tokens; rejected
            // positions are renoised with fresh random tokens.
            let freshKey = nextRandomKey()
            let fresh = MLX.randInt(0 ..< Int32(vocabSize), [1, canvasLength], key: freshKey)
            canvas = MLX.which(acceptMask, denoiserCanvas, fresh)

            denoisingForwardCount += 1

            // Host sync once per step for the stopping criteria.
            let meanEntropy = entropy.mean()
            MLX.eval(canvas, argmaxCanvas, meanEntropy)
            argmaxIds = argmaxCanvas[0].asArray(Int32.self)
            samplerTime += Date.timeIntervalSinceReferenceDate - samplerStart
            if stopper.shouldStop(
                argmaxCanvas: argmaxIds, meanEntropy: meanEntropy.item(Float.self))
            {
                break
            }

            // 5. Logits self-condition the next step.
            selfConditioning = processed
        }
        denoiseTime += Date.timeIntervalSinceReferenceDate - cycleStart

        // 4. Finalize: argmax canvas becomes the committed block; cut the
        // emitted stream after the first EOS.
        var emitted = argmaxIds.map(Int.init)
        if let eosIndex = emitted.firstIndex(where: { options.eosTokenIds.contains($0) }) {
            emitted = Array(emitted.prefix(through: eosIndex))
            finished = true
        } else {
            // Only an unfinished sequence needs the canvas in the encoder
            // cache for the next block.
            pendingEncoderCanvas = argmaxIds
        }
        pendingTokens.append(contentsOf: emitted)
        canvasesEmitted += 1
    }

    // MARK: - Prefix cache store (paged + disk/SSD tiers)

    public mutating func storeCacheAfterGeneration(
        generatedTokenIds: [Int],
        includeGeneratedBoundary: Bool
    ) {
        guard let coordinator = cacheCoordinator,
            !promptTokenIds.isEmpty,
            let promptCacheSnapshot
        else {
            reportStatsOnce()
            return
        }

        // Same guard as the other two store paths: saving the cache duplicates it
        // several times over at the memory high-water mark, and a prefix-cache entry
        // is only ever a speed-up for a later request — it must never be able to take
        // the host down. If the copies won't fit, don't make them.
        guard CacheStoreBudget.canStore(promptCacheSnapshot) else {
            let gib = Double(CacheStoreBudget.cacheBytes(promptCacheSnapshot)) / 1_073_741_824
            Self.logger.info(
                """
                prefix-cache: skipping store of a \(String(format: "%.1f", gib), privacy: .public) GiB \
                KV cache — the copies it requires do not fit in memory. Generation was unaffected; \
                only the cache entry was dropped.
                """
            )
            reportStatsOnce()
            return
        }

        let cacheSnapshot = promptCacheSnapshot.map { $0.copy() }
        let requiresDiskBackedRestore =
            cacheRequiresDiskBackedCoordinatorRestore(cacheSnapshot)
        if !requiresDiskBackedRestore {
            MLX.eval(cacheSnapshot)
        }
        // Rotating layers require typed LayerKind persistence. The coordinator
        // may also derive paged full-attention KV when this exact mixed cache
        // was admitted with a rotating boundary companion, matching AR.
        let perLayerData =
            requiresDiskBackedRestore ? [] : extractLayerData(from: cacheSnapshot)
        let diskStoreCache = makeDiskStoreCache(
            fromPromptBoundary: cacheSnapshot,
            parameters: cacheInitParameters)
        coordinator.storeAfterGeneration(
            promptTokens: promptTokenIds,
            perLayerData: perLayerData,
            ssmStates: nil,
            cache: diskStoreCache,
            mediaSalt: mediaSalt)

        // History boundaries: store the conversation-stable prefixes so the
        // NEXT request (which re-renders this turn without the generation
        // suffix) can extension-hit. Rotating caches are only trimmable
        // before their window wraps; when trimming is unavailable the
        // boundary store is skipped gracefully.
        for boundary in Set(cachePrefixTokenCounts).sorted()
        where boundary > 0 && boundary < promptTokenIds.count {
            // Re-check per boundary rather than relying on the check above: each
            // boundary copies the *untrimmed* cache again, and the stores before it
            // have already raised the memory floor. One store fitting does not mean
            // N more do.
            guard CacheStoreBudget.canStore(promptCacheSnapshot) else { break }
            let trimCount = promptTokenIds.count - boundary
            let boundarySnapshot = promptCacheSnapshot.map { $0.copy() }
            guard canTrimPromptCache(boundarySnapshot),
                trimPromptCache(boundarySnapshot, numTokens: trimCount) == trimCount
            else { continue }
            MLX.eval(boundarySnapshot)
            let boundaryTokens = Array(promptTokenIds.prefix(boundary))
            let boundaryPerLayer =
                cacheRequiresDiskBackedCoordinatorRestore(boundarySnapshot)
                ? [] : extractLayerData(from: boundarySnapshot)
            let boundaryDiskCache = makeDiskStoreCache(
                fromPromptBoundary: boundarySnapshot,
                parameters: cacheInitParameters)
            coordinator.storeAfterGeneration(
                promptTokens: boundaryTokens,
                perLayerData: boundaryPerLayer,
                ssmStates: nil,
                cache: boundaryDiskCache,
                mediaSalt: mediaSalt)
        }

        reportStatsOnce()
    }

    // MARK: - Instrumentation

    private mutating func reportStatsOnce() {
        guard !statsReported else { return }
        statsReported = true

        let emittedTokens = tokenCount
        let tokensPerForward =
            denoisingForwardCount > 0
            ? Double(emittedTokens) / Double(denoisingForwardCount) : 0
        let stepsPerCanvas =
            canvasesEmitted > 0
            ? Double(denoisingForwardCount) / Double(canvasesEmitted) : 0
        let wall = Date.timeIntervalSinceReferenceDate - iteratorStartTime
        let line = String(
            format:
                "[BlockDiffusion] canvases=%d denoisingForwards=%d avgStepsPerCanvas=%.1f "
                + "emittedTokens=%d tokensPerForward=%.2f encoderForwards=%d "
                + "prefixCacheRestoredTokens=%d prefillSec=%.3f denoiseSec=%.3f "
                + "decoderFwdSec=%.3f samplerSec=%.3f wallSec=%.3f "
                + "cacheMode=simple+rotating(window) maxDenoisingSteps=%d entropyBound=%.3f\n",
            canvasesEmitted,
            denoisingForwardCount,
            stepsPerCanvas,
            emittedTokens,
            tokensPerForward,
            encoderForwardCount,
            prefixCacheRestoredTokens,
            promptPrefillTime,
            denoiseTime,
            decoderForwardTime,
            samplerTime,
            wall,
            options.maxDenoisingSteps,
            options.entropyBound)
        FileHandle.standardError.write(Data(line.utf8))
    }
}
