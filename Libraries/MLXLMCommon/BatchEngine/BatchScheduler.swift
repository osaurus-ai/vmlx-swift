// Copyright 2025 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX

// MARK: - Slot Phase

/// The generation phase of an active slot in the batch engine.
enum SlotPhase {
    /// Still processing the prompt in chunks. The slot is not yet part of the
    /// batched decode step.
    case prefill

    /// Prompt fully processed, generating tokens one at a time as part of
    /// the batched decode step.
    case decode
}

// MARK: - Active Slot

/// An active generation slot managed by ``BatchEngine``.
///
/// Each slot represents one in-flight request. It owns its KV cache, sampler,
/// processor, and token history. Slots transition from `.prefill` → `.decode`
/// and eventually become finished when they hit a stop token or max token limit.
struct BatchSlot {
    /// Unique identifier matching the original request.
    let id: BatchRequestID

    /// Continuation for streaming tokens back to the caller.
    let continuation: AsyncStream<BatchGeneration>.Continuation

    /// The request's full ``GenerateParameters``, preserved so per-slot
    /// lifecycle hooks (quantization, compile path, etc.) can consult the
    /// original configuration. Added by Stage 0 of the batch-engine blockers
    /// work — ``BatchQuantize`` reads `kvMode`/`kvBits`/`quantizedKVStart`
    /// from here.
    let parameters: GenerateParameters

    /// Per-request sampler (temperature, topP, etc. from the request's `GenerateParameters`).
    let sampler: LogitSampler

    /// Per-request logit processor (repetition penalty, etc.).
    var processor: LogitProcessor?

    /// Set of token IDs that signal end of generation.
    let stopTokenIDs: Set<Int>

    /// Maximum tokens to generate for this request. `nil` = unlimited.
    let maxTokens: Int?

    /// Per-layer KV caches for this sequence (B=1 each).
    ///
    /// `var` (not `let`) so `BatchQuantize.maybeCompress` can swap `KVCacheSimple`
    /// layers for `TurboQuantKVCache` after the compression threshold is reached.
    /// The KVCache protocol is a reference type — a swap replaces the layer handle,
    /// per-slot state before the swap is migrated by `TurboQuantKVCache.fromSimpleCache`.
    var cache: [KVCache]

    /// Clean prompt-boundary cache snapshot captured after prefill and before
    /// any generated token is fed back into the model. Cache stores use this,
    /// not the live decode cache, because coordinator keys are prompt-only.
    var promptCacheSnapshot: [KVCache]?

    /// Token IDs that describe the cache snapshot at the prompt boundary.
    ///
    /// Usually this is the same as `originalInput.text.tokens`. Some models
    /// perform source-compatible prompt pruning inside `prepare` after media
    /// embeddings are available; those models return `LMOutput.effectivePromptTokens`,
    /// and cache stores must use that post-pruned key.
    var cachePromptTokenIds: [Int]

    /// Whether `cachePromptTokenIds` came from model-side post-prepare pruning.
    ///
    /// When true, input-side prefix counts are in the pre-pruned coordinate
    /// space and are not safe to reuse for history-boundary cache entries.
    var cachePromptUsesPostPrepareKey: Bool

    /// The original full input, preserved for VLM `prepare()` which needs image data.
    /// Also used for seeding the logit processor with the full prompt tokens.
    let originalInput: LMInput

    /// Remaining prompt tokens to process during prefill.
    var pendingTokens: MLXArray

    /// The next token to feed into the model (last sampled token during decode).
    var nextToken: MLXArray?

    /// True when `nextToken` has already been surfaced to the consumer.
    ///
    /// Prefill routes the first sampled token itself (EOS check + yield) so
    /// TTFT isn't delayed; the pipelined B=1 decode path must therefore skip
    /// routing on its first iteration and only start routing from the second
    /// input token. The sync decode path ignores this flag (it routes the
    /// freshly sampled token, never the input).
    var nextTokenAlreadyRouted: Bool = false

    /// Current generation phase.
    var phase: SlotPhase = .prefill

    /// Number of tokens generated so far (decode phase only).
    var generatedTokenCount: Int = 0

    /// Token IDs emitted to the caller for this request.
    ///
    /// CacheCoordinator stores a prompt-boundary entry after prefill, but a
    /// multi-turn chat's next prompt starts with the previous prompt plus the
    /// assistant answer. Recording visible generated tokens lets finishSlot
    /// persist that post-answer boundary too, so growing-chat turns can reuse
    /// the previous conversation prefix instead of missing every turn.
    var generatedTokenIds: [Int] = []

    /// True when this request can emit structured tool calls.
    ///
    /// BatchEngine parses tool calls outside the actor that stores cache
    /// entries, so finishSlot cannot observe the final `.toolCall` event
    /// directly. Treat tool-enabled requests conservatively: store prompt and
    /// history boundaries, but not generated/post-answer boundaries that could
    /// make a warm required-tool replay resume after the tool envelope.
    let disablesGeneratedCacheBoundary: Bool

    /// Number of prompt tokens (for completion info).
    let promptTokenCount: Int

    /// Timestamp when prefill started (for metrics).
    let prefillStartTime: Date

    /// Timestamp when decode started (for metrics).
    var decodeStartTime: Date?

    /// Whether this slot has finished generating.
    var isFinished: Bool = false

    /// Prefill step size for this request.
    let prefillStepSize: Int

    /// Stable fingerprint of any request-scope or media content in the input.
    /// `nil` for ordinary text-only requests. Computed once at slot
    /// construction and passed to the cache coordinator on both fetch
    /// (prefill) and store (finishSlot) so reasoning-mode and VLM multi-turn
    /// traffic can cache-hit without colliding with other modes/media.
    let mediaSalt: String?

    /// Compiled forward closure captured when this slot's cache is promoted
    /// to `CompilableKVCache` layers. Stage 1B.3 sets this for the solo
    /// slot in a `maxBatchSize == 1` engine when
    /// `parameters.enableCompiledBatchDecode` is on AND the cache family
    /// is `.simple`. Nil means "route this slot's decode through the
    /// uncompiled `BatchKVCache` path".
    ///
    /// When non-nil, `stepBatchDecode` feeds this slot's next token through
    /// the closure directly instead of wrapping it in a `BatchKVCache`.
    /// The closure mutates `slot.cache` (now `CompilableKVCache` layers) in
    /// place via the `_updateInternal` discipline; no wrapper is constructed
    /// per step.
    ///
    /// Stage 1B.4 will generalise to `maxBatchSize > 1` via a per-bucket
    /// `BucketHandle` holding shared `[B, H, maxLen, D]` buffers.
    var compiledForward: (@Sendable ([MLXArray]) -> [MLXArray])?

    // MARK: - Sampling

    /// Sample a token from logits, applying processor and sampler.
    ///
    /// Returns the sampled token as an `MLXArray` scalar.
    mutating func sampleToken(from logits: MLXArray) -> MLXArray {
        var logits = logits
        if var proc = processor {
            logits = proc.process(logits: logits)
            let token = sampler.sample(logits: logits)
            proc.didSample(token: token)
            self.processor = proc
            return token
        }
        return sampler.sample(logits: logits)
    }
}

// MARK: - Slot Construction

extension BatchSlot {
    /// Create a slot from a pending request.
    ///
    /// - Parameters:
    ///   - request: The pending request to activate.
    ///   - cache: Per-layer KV caches allocated by the model's `newCache()`.
    ///   - stopTokenIDs: Set of EOS token IDs for this model.
    init(from request: BatchPendingRequest, cache: [KVCache], stopTokenIDs: Set<Int>) {
        self.id = request.id
        self.continuation = request.continuation
        self.parameters = request.parameters
        self.sampler = request.parameters.sampler()
        self.processor = request.parameters.processor()
        self.stopTokenIDs = stopTokenIDs
        self.maxTokens = request.parameters.maxTokens
        self.cache = cache
        self.promptCacheSnapshot = nil
        self.cachePromptTokenIds = request.input.text.tokens.reshaped(-1).asArray(Int.self)
        self.cachePromptUsesPostPrepareKey = false
        self.originalInput = request.input
        self.pendingTokens = request.input.text.tokens
        self.nextToken = nil
        self.generatedTokenCount = 0
        self.generatedTokenIds = []
        self.disablesGeneratedCacheBoundary =
            request.input.toolSchemas?.isEmpty == false
        self.promptTokenCount = request.input.text.tokens.size
        self.prefillStartTime = Date()
        self.prefillStepSize = request.parameters.prefillStepSize
        self.mediaSalt = computeCacheSalt(for: request.input, parameters: request.parameters)
        self.compiledForward = nil
    }
}
