// Copyright 2025 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXNN
import os

/// Errors thrown by mutable ``BatchEngine`` configuration APIs.
public enum BatchEngineConfigurationError: Error, LocalizedError, Sendable {
    case invalidMaxBatchSize(Int)
    case engineShutdown

    public var errorDescription: String? {
        switch self {
        case .invalidMaxBatchSize(let value):
            return "BatchEngine maxBatchSize must be greater than zero, got \(value)"
        case .engineShutdown:
            return "BatchEngine is shut down and cannot be reconfigured"
        }
    }
}

private func cancelledBatchStream(
    promptTokenCount: Int
) -> (id: BatchRequestID, stream: AsyncStream<BatchGeneration>) {
    let id = BatchRequestID()
    let (stream, continuation) = AsyncStream<BatchGeneration>.makeStream()
    continuation.yield(.info(GenerateCompletionInfo(
        promptTokenCount: promptTokenCount,
        generationTokenCount: 0,
        promptTime: 0,
        generationTime: 0,
        stopReason: .cancelled
    )))
    continuation.finish()
    return (id, stream)
}

private func cancelledGenerationStream(
    promptTokenCount: Int
) -> AsyncStream<Generation> {
    let (stream, continuation) = AsyncStream<Generation>.makeStream()
    continuation.yield(.info(GenerateCompletionInfo(
        promptTokenCount: promptTokenCount,
        generationTokenCount: 0,
        promptTime: 0,
        generationTime: 0,
        stopReason: .cancelled
    )))
    continuation.finish()
    return stream
}

private final class PrefillProgressAccumulator: @unchecked Sendable {
    private let continuation: AsyncStream<BatchGeneration>.Continuation
    private let completedBeforePrefill: Int
    private let totalPromptUnits: Int
    private let lock = NSLock()
    private var lastReportedCompleted: Int

    init(
        continuation: AsyncStream<BatchGeneration>.Continuation,
        completedBeforePrefill: Int,
        totalPromptUnits: Int
    ) {
        self.continuation = continuation
        self.completedBeforePrefill = completedBeforePrefill
        self.totalPromptUnits = totalPromptUnits
        self.lastReportedCompleted = completedBeforePrefill
    }

    func report(completedInPrepare: Int) {
        let completed = min(
            totalPromptUnits,
            completedBeforePrefill + max(0, completedInPrepare))
        lock.lock()
        guard completed > lastReportedCompleted else {
            lock.unlock()
            return
        }
        lastReportedCompleted = completed
        lock.unlock()
        continuation.yield(.prefillProgress(PrefillProgress(
            stage: .prefill,
            completedUnitCount: completed,
            totalUnitCount: totalPromptUnits,
            detail: "chunk")))
    }
}

private func debugLogReasoningPromptTail(
    modelName: String,
    promptTail: String?,
    path: String
) {
    guard ProcessInfo.processInfo.environment["VMLINUX_REASONING_PROMPT_TAIL_LOG"] == "1"
    else { return }
    let escaped = (promptTail ?? "<nil>")
        .replacingOccurrences(of: "\\", with: "\\\\")
        .replacingOccurrences(of: "\n", with: "\\n")
        .replacingOccurrences(of: "\r", with: "\\r")
    let line = "[vmlx] reasoning promptTail path=\(path) model=\(modelName) tail=\(escaped)\n"
    if let data = line.data(using: .utf8) {
        FileHandle.standardError.write(data)
    }
}

private func debugDumpReasoningPrompt(
    input: LMInput,
    tokenizer: any Tokenizer,
    modelName: String,
    path: String
) {
    let env = ProcessInfo.processInfo.environment
    guard let dir = env["VMLINUX_REASONING_PROMPT_DUMP_DIR"],
          !dir.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    else { return }

    guard let tokenIds = input.text.tokenIds, !tokenIds.isEmpty else { return }
    let rendered = tokenizer.decode(tokenIds: tokenIds, skipSpecialTokens: false)

    let safeModel = modelName
        .map { ch in ch.isLetter || ch.isNumber || ch == "-" || ch == "_" ? ch : "_" }
        .reduce(into: "") { $0.append($1) }
    let timestamp = Int(Date().timeIntervalSince1970 * 1000)
    let pid = ProcessInfo.processInfo.processIdentifier
    let url = URL(fileURLWithPath: dir, isDirectory: true)
        .appendingPathComponent("prompt-\(timestamp)-\(pid)-\(path)-\(safeModel).txt")
    let body = """
    path=\(path)
    model=\(modelName)
    promptTokens=\(tokenIds.count)

    \(rendered)
    """
    do {
        try FileManager.default.createDirectory(
            at: URL(fileURLWithPath: dir, isDirectory: true),
            withIntermediateDirectories: true)
        try body.write(to: url, atomically: true, encoding: .utf8)
        let line = "[vmlx] reasoning promptDump path=\(path) model=\(modelName) file=\(url.path)\n"
        if let data = line.data(using: .utf8) {
            FileHandle.standardError.write(data)
        }
    } catch {
        let line = "[vmlx] reasoning promptDump failed path=\(path) model=\(modelName) error=\(error.localizedDescription)\n"
        if let data = line.data(using: .utf8) {
            FileHandle.standardError.write(data)
        }
    }
}

private final class BatchStreamTerminationState: @unchecked Sendable {
    private let lock = NSLock()
    private var completed = false

    func markCompleted() {
        lock.lock()
        completed = true
        lock.unlock()
    }

    func shouldCancelOnTermination() -> Bool {
        lock.lock()
        let shouldCancel = !completed
        lock.unlock()
        return shouldCancel
    }
}

// MARK: - BatchEngine

/// Continuous batching inference engine for mlx-swift-lm.
///
/// `BatchEngine` processes multiple generation requests simultaneously by batching
/// their decode steps through a single model forward pass. This provides significantly
/// higher throughput than serial single-sequence generation when serving multiple
/// concurrent requests.
///
/// ## Architecture
///
/// The engine follows the continuous batching pattern used by production inference
/// servers (vLLM, TGI):
///
/// 1. **Request submission** — Callers submit requests via ``submit(input:parameters:)``
///    and receive an `AsyncStream<BatchGeneration>` that yields tokens as they are generated.
///
/// 2. **Scheduling loop** — A background task runs the engine loop:
///    - Admits pending requests from the wait queue into active slots
///    - Processes prefill chunks for newly admitted requests (one chunk per iteration)
///    - Batches all decode-phase slots into a single `[B, 1]` forward pass
///    - Samples tokens independently per sequence using each request's own parameters
///    - Detects completion (EOS, max tokens) and cleans up finished slots
///
/// 3. **Cache management** — Each sequence owns its own `[KVCache]` array (B=1).
///    During batched decode, per-layer ``BatchKVCache`` wrappers present these as
///    a single `[B, H, L, D]` cache to the model.
///
/// ## Usage
///
/// ```swift
/// // Load model normally
/// let modelContext = try await ModelFactory.shared.load(...)
///
/// // Create engine — uses existing GenerateParameters per-request
/// let engine = BatchEngine(context: modelContext, maxBatchSize: 8)
///
/// // Submit requests (from different async contexts, e.g., HTTP handlers)
/// let stream = await engine.submit(input: lmInput, parameters: generateParams)
/// for await event in stream {
///     switch event {
///     case .token(let id):
///         // Feed to NaiveStreamingDetokenizer
///         detokenizer.append(token: id)
///     case .info(let completionInfo):
///         print(completionInfo.summary())
///     }
/// }
/// ```
///
/// ## Thread Safety
///
/// `BatchEngine` is an `actor` — all state is automatically isolated. The model
/// is only accessed from the engine's scheduling loop, ensuring single-threaded
/// model access without explicit locking.
///
/// ## Compatibility
///
/// - All input parameters come from the existing ``GenerateParameters`` struct.
///   No new configuration types are forced on callers.
/// - The engine uses the model's `callAsFunction` and `newCache` methods directly.
///   No model code changes are required.
/// - Existing single-sequence ``TokenIterator`` and ``generate()`` APIs are unaffected.
///
/// ## Extensibility
///
/// The slot cache type is `[KVCache]` (protocol-typed). Future cache implementations
/// (TurboQuant, paged caches, hybrid SSM) can be used as slot caches without changing
/// the engine core.
public actor BatchEngine {

    // MARK: - Configuration

    /// Maximum number of sequences decoded simultaneously in one batch.
    /// Additional requests are queued until a slot opens.
    public private(set) var maxBatchSize: Int

    /// Number of iterations between GPU memory cache purges.
    /// Matches the 256-token interval used by ``TokenIterator``.
    public let memoryPurgeInterval: Int

    // MARK: - State

    /// The loaded model context (model, tokenizer, config, processor).
    private let context: ModelContext

    /// Optional cache coordinator for multi-tier KV caching.
    /// When present, the engine will attempt to fetch cached state before prefill
    /// and store cache state after generation completes.
    private let cacheCoordinator: CacheCoordinator?

    /// Logger for cache-related diagnostics.
    private static let logger = Logger(subsystem: "vmlx", category: "BatchEngine")

    /// Set of token IDs that signal end of generation for this model.
    private let stopTokenIDs: Set<Int>

    /// Default decoded-text stop strings for special tokens that the
    /// tokenizer cannot resolve to IDs.
    private let defaultStopStrings: [String]

    /// Requests waiting to be admitted into active slots.
    private var waitQueue: [BatchPendingRequest] = []

    /// Active generation slots (max `maxBatchSize`).
    private var activeSlots: [BatchSlot] = []

    /// High-water mark for concurrently admitted slots. This is exposed for
    /// release gates because polling `activeCount` from outside the actor can
    /// miss short-lived overlap while a model forward monopolizes the executor.
    private var activeCountHighWatermark: Int = 0

    /// Decode iterations split because admitted slots had incompatible live
    /// cache/codec signatures. Mixed plain/TurboQuant KV slots may be active
    /// at the same time, but the scheduler must not force incompatible cache
    /// representations into one model forward.
    private var decodeCompatibilitySplitCount: Int = 0

    /// Number of slot cache arrays that actually crossed from plain KV into
    /// TurboQuant KV. Exposed only for release gates; kvMode alone is not
    /// proof that the live codec activated.
    private var turboQuantCompressionCount: Int = 0

    /// Most recent real before/after layer topology for a successful
    /// TurboQuant transition. This remains available after its request slot
    /// completes so host diagnostics can prove which layers converted.
    private var lastTurboQuantCacheTransition: TurboQuantCacheTransitionSnapshot?

    /// Background scheduling loop task handle.
    private var loopTask: Task<Void, Never>?

    /// Direct single-request generation task for `generate(...)` when the
    /// engine is configured as B=1 and no queued/active batch work exists.
    ///
    /// This routes Osaurus's default single-stream path through the same
    /// `TokenIterator` loop as `ModelContainer.generate(...)`, while keeping
    /// `submit(...)` and maxBatchSize > 1 on the continuous-batching scheduler.
    private var soloFastPathTask: Task<Void, Never>?
    private var soloFastPathID: UUID?
    private var soloFastPathHadMedia = false

    /// Terminal lifecycle flag. Once shutdown begins, stale engine handles
    /// reject future submissions instead of restarting GPU work.
    public private(set) var isShutdown: Bool = false

    /// Total decode steps since last memory purge.
    private var stepsSinceMemoryPurge: Int = 0

    /// Decode steps since the actor last yielded for control-plane work.
    /// Keep B=1 hot-path yields sparse for throughput, but do not let a long
    /// decode starve `cancel`, `shutdown`, or runtime configuration updates.
    private var stepsSinceControlPlaneYield: Int = 0

    /// Maximum B=1 decode steps before yielding back to the actor executor.
    private let controlPlaneYieldInterval: Int = 8

    /// Hy3/Hunyuan, Laguna, and MiniMax currently decode coherently on the uncompiled
    /// path but diverge on the single-slot compiled trace. Keep compile opt-in
    /// from silently taking those unsafe routes until each model path has a
    /// dedicated compiled-vs-uncompiled parity test.
    private var compiledDecodeDeniedForModel: Bool {
        if context.configuration.toolCallFormat == .hunyuan {
            return true
        }
        let modelName = context.configuration.name.lowercased()
        let modelTypeName = String(describing: type(of: context.model)).lowercased()
        return modelName.contains("hy3") || modelName.contains("hy_v3") || modelName.contains("hy-v3")
            || modelTypeName.contains("hy3") || modelTypeName.contains("hunyuan")
            || modelName.contains("laguna") || modelTypeName.contains("laguna")
            || modelName.contains("minimax") || modelTypeName.contains("minimax")
    }

    /// Initial admission window for B>1 engines. The scheduler runs prefill on
    /// the actor today, so once a long prefill starts a just-behind `submit`
    /// cannot enqueue until that prefill returns. Give immediately-following
    /// callers a short deterministic window to form the first batch without
    /// adding latency to single-stream B=1 engines.
    private let initialAdmissionCoalescingNanos: UInt64

    // MARK: - Initialization

    /// Create a new continuous batching engine.
    ///
    /// - Parameters:
    ///   - context: The loaded model context from ``ModelFactory``.
    ///   - maxBatchSize: Maximum concurrent sequences. Defaults to 8.
    ///     Higher values increase throughput but use more memory.
    ///   - memoryPurgeInterval: Steps between GPU memory cache purges. Defaults to 256.
    ///   - cacheCoordinator: Optional multi-tier cache coordinator. When provided,
    ///     the engine will attempt cache lookups before prefill and store cache state
    ///     after generation completes. Defaults to nil.
    public init(
        context: ModelContext,
        maxBatchSize: Int = 8,
        memoryPurgeInterval: Int = 256,
        cacheCoordinator: CacheCoordinator? = nil
    ) {
        precondition(maxBatchSize > 0, "BatchEngine maxBatchSize must be greater than zero")
        self.context = context
        self.maxBatchSize = maxBatchSize
        self.memoryPurgeInterval = memoryPurgeInterval
        self.cacheCoordinator = cacheCoordinator
        self.initialAdmissionCoalescingNanos = maxBatchSize > 1 ? 25_000_000 : 0

        let resolvedStops = resolveStopSequences(
            modelConfiguration: context.configuration,
            tokenizer: context.tokenizer,
            includeUnknownToken: true)
        self.stopTokenIDs = resolvedStops.tokenIDs
        self.defaultStopStrings = resolvedStops.textStopStrings
    }

    // MARK: - Public API

    /// Change the active-slot admission limit for future scheduling ticks.
    ///
    /// If the limit is increased, queued requests are admitted immediately up
    /// to the new capacity. If the limit is decreased below the current active
    /// slot count, no active request is cancelled; the engine simply stops
    /// admitting new work until active slots fall below the new limit.
    ///
    /// - Parameter newMaxBatchSize: New maximum number of active slots. Must be
    ///   greater than zero.
    public func updateMaxBatchSize(_ newMaxBatchSize: Int) throws {
        guard newMaxBatchSize > 0 else {
            throw BatchEngineConfigurationError.invalidMaxBatchSize(newMaxBatchSize)
        }
        guard !isShutdown else {
            throw BatchEngineConfigurationError.engineShutdown
        }
        guard newMaxBatchSize != maxBatchSize else { return }

        let old = maxBatchSize
        maxBatchSize = newMaxBatchSize
        Self.logger.info(
            "Updated maxBatchSize from \(old, privacy: .public) to \(newMaxBatchSize, privacy: .public)"
        )

        if newMaxBatchSize > old && soloFastPathTask == nil {
            admitPendingRequests()
            if !activeSlots.isEmpty {
                ensureLoopRunning()
            }
        }
    }

    /// Submit a generation request, returning raw token events.
    ///
    /// This is the low-level API. For text output, use ``generate(input:parameters:)``
    /// which handles detokenization automatically.
    ///
    /// - Parameters:
    ///   - input: Prepared model input (from `UserInputProcessor.prepare()`).
    ///   - parameters: Generation parameters for this request.
    /// - Returns: A tuple of `(requestID, stream)`. The stream yields token IDs
    ///   and completion info. Use the ID with ``cancel(_:)`` to stop early.
    @discardableResult
    public func submit(
        input: consuming sending LMInput,
        parameters: GenerateParameters
    ) -> (id: BatchRequestID, stream: AsyncStream<BatchGeneration>) {
        guard !isShutdown else {
            return cancelledBatchStream(promptTokenCount: input.text.tokens.size)
        }

        do {
            _ = try AccelerationRuntime.resolveTextDecode(parameters.accelerationMode)
        } catch {
            Self.logger.error(
                "Rejected acceleration request: \(error.localizedDescription, privacy: .public)"
            )
            return cancelledBatchStream(promptTokenCount: input.text.tokens.size)
        }

        if parameters.draftStrategy?.usesNativeMTP == true {
            Self.logger.error(
                "Rejected BatchEngine.submit native MTP request: raw batched native-MTP scheduling is not implemented; use BatchEngine.generate or Evaluate.generate for the exclusive native-MTP path."
            )
            return cancelledBatchStream(promptTokenCount: input.text.tokens.size)
        }

        let (stream, continuation) = AsyncStream<BatchGeneration>.makeStream()
        let promptTail = _decodePromptTail(
            input: input, tokenizer: context.tokenizer, tokens: 64)
        debugLogReasoningPromptTail(
            modelName: context.configuration.name,
            promptTail: promptTail,
            path: "BatchEngine.submit")
        debugDumpReasoningPrompt(
            input: input,
            tokenizer: context.tokenizer,
            modelName: context.configuration.name,
            path: "BatchEngine.submit")
        let request = BatchPendingRequest(
            input: input,
            parameters: parameters,
            continuation: continuation
        )
        waitQueue.append(request)
        if soloFastPathTask == nil {
            ensureLoopRunning()
        }
        return (request.id, stream)
    }

    /// Generate text from prepared input — drop-in replacement for `ModelContainer.generate()`.
    ///
    /// Returns the same `AsyncStream<Generation>` type as the existing single-sequence
    /// API, with `.chunk(String)` for decoded text and `.info(GenerateCompletionInfo)`
    /// for completion metrics. Handles detokenization internally.
    ///
    /// ## Example
    /// ```swift
    /// let engine = BatchEngine(context: modelContext)
    /// let input = try await modelContext.processor.prepare(input: userInput)
    /// let stream = await engine.generate(input: input, parameters: params)
    /// for await generation in stream {
    ///     switch generation {
    ///     case .chunk(let text): print(text, terminator: "")
    ///     case .reasoning: break    // route to a think-pane if you render CoT
    ///     case .info(let info): print("\n\(info.summary())")
    ///     case .toolCall: break
    ///     }
    /// }
    /// ```
    ///
    /// - Parameters:
    ///   - input: Prepared model input.
    ///   - parameters: Generation parameters for this request.
    /// - Returns: An `AsyncStream<Generation>` yielding text chunks and completion info.
    public func generate(
        input: consuming sending LMInput,
        parameters: GenerateParameters
    ) -> AsyncStream<Generation> {
        guard !isShutdown else {
            return cancelledGenerationStream(promptTokenCount: input.text.tokens.size)
        }

        do {
            _ = try AccelerationRuntime.resolveTextDecode(parameters.accelerationMode)
        } catch {
            Self.logger.error(
                "Rejected acceleration request: \(error.localizedDescription, privacy: .public)"
            )
            return cancelledGenerationStream(promptTokenCount: input.text.tokens.size)
        }

        // Block-diffusion speculative decoding dispatch. When
        // parameters.draftStrategy is .dflash or .ddtree AND the
        // target model conforms to HiddenStateCaptureModel +
        // TokenEmbedderModel, route through SpecDecStream. Zero API
        // churn for callers using .none / nil / .autoregressive — they
        // fall through to the batched-decode path below.
        if let strategy = parameters.draftStrategy,
            strategy.usesBlockDiffusion,
            let stream = SpecDecStream.streamViaStrategy(
                strategy: strategy,
                inputIds: input.text.tokens,
                context: context,
                maxNewTokens: parameters.maxTokens ?? 256,
                stopTokenIDs: [],
                temperature: parameters.temperature,
                toolSchemas: input.toolSchemas)
        {
            return stream
        }

        let tokenizer = context.tokenizer
        // Snapshot format + reasoning stamp + stop strings from the
        // configuration so the background task doesn't need to reach
        // back into the actor.
        let toolCallFormat = context.configuration.toolCallFormat ?? .json
        let toolSchemas = input.toolSchemas
        let reasoningParserName = context.configuration.reasoningParserName
        let extraStopStrings = mergeStopStrings(parameters.extraStopStrings, defaultStopStrings)

        // Decode the tail of the prompt for `ReasoningParser.forPrompt`
        // auto-detection. This tells the parser whether the prompt
        // ended inside a think/harmony block (e.g. Qwen 3.x default
        // `enable_thinking=true` → prompt ends `<think>\n` so the
        // model's first output byte is already reasoning) or after
        // a closed block (enable_thinking=false → prompt ends
        // `</think>\n\n` so the model starts in content).
        //
        // Tail of ~64 tokens is plenty for any realistic opener/closer
        // pair — the longest we handle is Gemma-4's `<|channel>thought\n`
        // (18 chars, ≤ 8 tokens). Using tokens not characters because
        // we have the tokenizer on hand.
        let promptTail = _decodePromptTail(
            input: input, tokenizer: tokenizer, tokens: 64)
        debugLogReasoningPromptTail(
            modelName: context.configuration.name,
            promptTail: promptTail,
            path: "BatchEngine.generate")
        debugDumpReasoningPrompt(
            input: input,
            tokenizer: tokenizer,
            modelName: context.configuration.name,
            path: "BatchEngine.generate")
        if parameters.draftStrategy?.usesNativeMTP == true {
            guard canStartExclusiveSoloPath else {
                Self.logger.error(
                    "Rejected BatchEngine.generate native MTP request: native MTP is an exclusive solo path until batched/paged native-MTP scheduling lands."
                )
                return cancelledGenerationStream(promptTokenCount: input.text.tokens.size)
            }
            return startSoloFastPath(
                input: input,
                parameters: parameters,
                promptTail: promptTail)
        }
        // Block-diffusion models (e.g. diffusion_gemma) generate whole
        // canvases via denoising and cannot share batched decode slots.
        // They run as an exclusive solo path; the batched path would fail
        // loudly anyway via the model's throwing prepare() guard.
        if context.model is any BlockDiffusionModel {
            guard canStartExclusiveSoloPath else {
                Self.logger.error(
                    "Rejected BatchEngine.generate block-diffusion request: block diffusion is an exclusive solo path until batched canvas scheduling lands."
                )
                return cancelledGenerationStream(promptTokenCount: input.text.tokens.size)
            }
            return startSoloFastPath(
                input: input,
                parameters: parameters,
                promptTail: promptTail)
        }
        if canStartSoloFastPath {
            return startSoloFastPath(
                input: input,
                parameters: parameters,
                promptTail: promptTail)
        }

        let promptTokenCount = input.text.tokens.size
        let (requestId, tokenStream) = submit(input: input, parameters: parameters)

        // Mirror the canonical `Evaluate.generateLoopTask` pattern: pair
        // `AsyncStream.makeStream()` with an unstructured `Task {}` that
        // owns the continuation. `if let` (not `while let`) — calling
        // `NaiveStreamingDetokenizer.next()` in a loop produces empty
        // strings forever and melts throughput under a real HF tokenizer.
        //
        // The inner pipeline matches `TextToolTokenLoopHandler` in
        // `Evaluate.swift` byte-for-byte: each decoded chunk runs through
        // an optional `ReasoningParser` first (peels off `<think>…</think>`
        // into `.reasoning` events), then through `ToolCallProcessor`
        // which extracts authoritative `.toolCall(ToolCall)` events,
        // then (if `extraStopStrings` set) through a `StopStringMatcher`
        // which halts upstream generation on substring match.
        let (outStream, continuation) = AsyncStream<Generation>.makeStream()
        let engineRef = self
        let terminationState = BatchStreamTerminationState()

        // Reap the slot when the consumer stops iterating (cancellation,
        // explicit break, or task drop). Without this, an orphan slot
        // keeps stepping inside the engine's scheduling loop, holding
        // Metal command buffers + pipelines alive. A subsequent request
        // that triggers a cache-restore path can collide with the
        // orphan slot's pipelines mid-encode and trigger
        // `Device::clear_library` →
        // `notifyExternalReferencesNonZeroOnDealloc`.
        //
        // Only run this on early consumer termination. Scheduling the
        // cancellation task after a normal completion can keep the
        // engine/model context alive until process teardown; JANGTQ
        // models with compiled helper state can then race MLX's global
        // compiler-cache finalizer on exit.
        //
        // Reported 2026-04-27 by osaurus integrator with the smoking-gun
        // diagnosis pointing at this exact missing handler.
        continuation.onTermination = {
            @Sendable [requestId, engineRef, terminationState] _ in
            guard terminationState.shouldCancelOnTermination() else { return }
            Task {
                await engineRef.cancel(requestId)
            }
        }

        Task {
            var detokenizer = NaiveStreamingDetokenizer(tokenizer: tokenizer)
            let activeToolSchemas = toolSchemas?.isEmpty == false ? toolSchemas : nil
            let toolCallProcessor: ToolCallProcessor? = {
                if let activeToolSchemas {
                    return ToolCallProcessor(format: toolCallFormat, tools: activeToolSchemas)
                }
                // No tools offered: still strip tool-call control markers for
                // tagged formats so a model that emits tool-call syntax anyway
                // (e.g. a hallucinated call under thinking) cannot leak literal
                // `<|tool_call>`/`call:` markers into visible text. Strip-only
                // mode discards the fabricated call since none was requested.
                if toolCallFormat.hasTaggedToolMarkers {
                    return ToolCallProcessor(
                        format: toolCallFormat, tools: nil, stripOnly: true)
                }
                return nil
            }()
            var reasoningParser = ReasoningParser.forPrompt(
                stampName: reasoningParserName,
                promptTail: promptTail)
            var stopMatcher = StopStringMatcher(stopStrings: extraStopStrings)
            var stopMatched = false

            func emitChunkThroughStop(_ text: String) {
                guard stopMatcher.isEnabled else {
                    continuation.yield(.chunk(text))
                    return
                }
                switch stopMatcher.feed(text) {
                case .streaming(let out):
                    if !out.isEmpty { continuation.yield(.chunk(out)) }
                case .stopped(let out):
                    if !out.isEmpty { continuation.yield(.chunk(out)) }
                    stopMatched = true
                }
            }

            func emitRouted(_ event: Generation) {
                switch event {
                case .chunk(let text):
                    emitChunkThroughStop(text)
                case .reasoning:
                    continuation.yield(event)
                case .prefillProgress:
                    continuation.yield(event)
                case .toolCall:
                    // Drain the stop-string matcher BEFORE the call event goes
                    // out, exactly as the solo path does (`Evaluate.swift`).
                    // Text still held there for disambiguation precedes the call
                    // in the model's output, but consumers stop forwarding text
                    // the moment a tool call lands (deliberate no-leak
                    // suppression of post-tool prose) — so a tail emitted after
                    // this event is silently dropped, cutting the visible answer
                    // mid-word. Without this, batch serving reproduced the same
                    // truncation the solo path already fixed.
                    if stopMatcher.isEnabled && !stopMatched {
                        let tail = stopMatcher.flush()
                        if !tail.isEmpty { continuation.yield(.chunk(tail)) }
                    }
                    continuation.yield(event)
                case .toolCallProgress:
                    continuation.yield(event)
                case .info:
                    continuation.yield(event)
                }
            }

            func pump(_ raw: String) {
                if stopMatched { return }
                let pieces: [String]
                if var parser = reasoningParser {
                    var kept: [String] = []
                    for segment in parser.feed(raw) {
                        switch segment {
                        case .content(let c):
                            kept.append(c)
                        case .reasoning(let r):
                            for event in routeGenerationText(
                                r,
                                channel: .reasoning,
                                through: toolCallProcessor
                            ) {
                                emitRouted(event)
                                if stopMatched { return }
                            }
                        }
                    }
                    reasoningParser = parser
                    pieces = kept
                } else {
                    pieces = [raw]
                }
                for piece in pieces {
                    for event in routeGenerationText(
                        piece,
                        channel: .content,
                        through: toolCallProcessor
                    ) {
                        emitRouted(event)
                        if stopMatched { return }
                    }
                }
            }

            // Reasoning state at end-of-stream, captured inside `flush()` in the
            // only window where it is truthful: after the detokenizer's held-back
            // tail has been pumped through the parser, and before
            // `ReasoningParser.flush()` clears the flag unconditionally.
            //
            // The detokenizer withholds a 24-character tail, so a close marker can
            // still be unparsed when the loop ends. `</think:opensource>` is 20
            // characters and fits entirely inside it — which is why Hunyuan v3
            // reported "ended inside <think>" on streams that closed cleanly and
            // split reasoning from content correctly. A short `</think>` clears the
            // holdback, so most families never surfaced this.
            var terminalInsideReasoning: Bool? = nil

            func flush() {
                // A matched stop string is the semantic end of the
                // response: everything still held in the detokenizer /
                // reasoning / tool buffers is chronologically AFTER the
                // match (a few tokens decode before the halt lands) and
                // must be discarded, not flushed — the matcher's buffer
                // was cleared at match time, so re-feeding would leak
                // post-stop text past the truncation point.
                if stopMatched { return }
                if let text = detokenizer.flush() {
                    pump(text)
                }
                terminalInsideReasoning = reasoningParser?.isInsideReasoning ?? false
                if var parser = reasoningParser {
                    for segment in parser.flush() {
                        switch segment {
                        case .content(let c):
                            for event in routeGenerationText(
                                c,
                                channel: .content,
                                through: toolCallProcessor
                            ) {
                                emitRouted(event)
                            }
                        case .reasoning(let r):
                            for event in routeGenerationText(
                                r,
                                channel: .reasoning,
                                through: toolCallProcessor
                            ) {
                                emitRouted(event)
                            }
                        }
                    }
                    reasoningParser = parser
                }
                // Route the tool processor's end-of-stream remainder
                // through the stop matcher (via emitRouted), not around
                // it: a stop-string occurrence held in the tool buffer at
                // EOS must still truncate, and feeding it keeps the
                // matcher's tail ordered AFTER this text so the drain
                // below cannot reorder characters.
                for event in flushGenerationText(
                    channel: reasoningParser?.isInsideReasoning == true ? .reasoning : .content,
                    through: toolCallProcessor
                ) {
                    emitRouted(event)
                }

                // Drain the stop-string matcher's held tail — no more
                // tokens are coming, whatever is held is safe to emit.
                // Skipped when stopMatched: the matcher already returned
                // its tail (pre-match prefix) at stop time.
                if stopMatcher.isEnabled && !stopMatched {
                    let tail = stopMatcher.flush()
                    if !tail.isEmpty { continuation.yield(.chunk(tail)) }
                }

            }

            var sawTerminalInfo = false
            var generatedTokenCount = 0
            let streamStartedAt = Date()

            for await event in tokenStream {
                switch event {
                case .prefillProgress(let progress):
                    continuation.yield(.prefillProgress(progress))
                case .token(let id):
                    generatedTokenCount += 1
                    detokenizer.append(token: id)
                    if let text = detokenizer.next() {
                        pump(text)
                    }
                    if stopMatched {
                        // Tell the BatchEngine actor to halt this slot
                        // on its next scheduling tick. The actor's
                        // `cancel(id:)` flips `isFinished` and emits
                        // its own `.info`; we transform that info's
                        // stopReason from `.cancelled` to `.stop`
                        // below when it arrives.
                        await engineRef.cancel(requestId)
                    }
                case .info(let info):
                    sawTerminalInfo = true
                    // Snapshot reasoning state BEFORE flush — `flush()`
                    // resets `insideReasoning` to false as part of
                    // draining the buffer. The pre-flush value is what
                    // the consumer wants: "was the LAST CONSUMED TOKEN
                    // inside a reasoning block?" If yes, the model
                    // ended without ever emitting `</think>`.
                    flush()
                    let unclosed =
                        terminalInsideReasoning ?? (reasoningParser?.isInsideReasoning ?? false)
                    detokenizer.startNewSegment()
                    // Detect "trapped thinking": stream ended while the
                    // reasoning parser was still inside a `<think>…</think>`
                    // block (no close tag ever observed). Surface it on
                    // the .info event so consumers can implement a UI
                    // fallback (mirror last sentence of .reasoning to
                    // .chunk, show "answer trapped in thinking" banner,
                    // etc.) without instrumenting the parser themselves.
                    let finalStop: GenerateStopReason
                    if stopMatched {
                        finalStop = .stop
                    } else {
                        finalStop = info.stopReason
                    }
                    let finalInfo = GenerateCompletionInfo(
                        promptTokenCount: info.promptTokenCount,
                        generationTokenCount: info.generationTokenCount,
                        promptTime: info.promptTime,
                        generationTime: info.generateTime,
                        stopReason: finalStop,
                        turboQuantCompressions: info.turboQuantCompressions,
                        turboQuantCacheTransition: info.turboQuantCacheTransition,
                        unclosedReasoning: unclosed)
                    if info.turboQuantCompressions > 0 {
                        turboQuantCompressionCount += info.turboQuantCompressions
                    }
                    if let transition = info.turboQuantCacheTransition {
                        lastTurboQuantCacheTransition = transition
                    }
                    terminationState.markCompleted()
                    continuation.yield(.info(finalInfo))
                }
            }
            if !sawTerminalInfo {
                // Defensive contract repair: every public `generate` stream must
                // terminate with completion info. Underlying token streams should
                // normally emit `.info` themselves, but if a lower layer closes
                // early we still need to flush held reasoning/tool-call text and
                // surface whether the model ended inside `<think>`.
                flush()
                let unclosed =
                    terminalInsideReasoning ?? (reasoningParser?.isInsideReasoning ?? false)
                detokenizer.startNewSegment()
                let elapsed = Date().timeIntervalSince(streamStartedAt)
                let finalInfo = GenerateCompletionInfo(
                    promptTokenCount: promptTokenCount,
                    generationTokenCount: generatedTokenCount,
                    promptTime: 0,
                    generationTime: elapsed,
                    stopReason: .cancelled,
                    unclosedReasoning: unclosed)
                continuation.yield(.info(finalInfo))
            }
            terminationState.markCompleted()
            continuation.finish()
        }
        return outStream
    }

    private var canStartSoloFastPath: Bool {
        maxBatchSize == 1 &&
            waitQueue.isEmpty &&
            activeSlots.isEmpty &&
            loopTask == nil &&
            soloFastPathTask == nil &&
            !isShutdown
    }

    private var canStartExclusiveSoloPath: Bool {
        waitQueue.isEmpty &&
            activeSlots.isEmpty &&
            loopTask == nil &&
            soloFastPathTask == nil &&
            !isShutdown
    }

    private func shouldSkipDiskBackedToolPromptSeedBoundary(for slot: BatchSlot) -> Bool {
        shouldSkipDiskBackedToolPromptSeedBoundary(
            toolSchemas: slot.originalInput.toolSchemas,
            disablesGeneratedCacheBoundary: slot.disablesGeneratedCacheBoundary)
    }

    private func shouldSkipDiskBackedToolPromptSeedBoundary(
        toolSchemas: [ToolSpec]?,
        disablesGeneratedCacheBoundary: Bool
    ) -> Bool {
        guard disablesGeneratedCacheBoundary || toolSchemas?.isEmpty == false else {
            return false
        }
        let modelName = context.configuration.name.lowercased()
        if modelName.contains("gemma-4") && modelName.contains("mxfp4") {
            return true
        }
        return false
    }

    /// Clamps prefill-progress frames so the reported completed count never
    /// decreases within a single request. A progress counter must be monotonic;
    /// partial / diverging-prefix cache restores can otherwise emit a frame
    /// whose `completedBeforePrefill` boundary is below an already-shown value,
    /// which the UI would render as the %% briefly ticking backward.
    private final class PrefillMonotonicGate: @unchecked Sendable {
        private let lock = NSLock()
        private var maxCompleted = 0
        func clamp(_ progress: PrefillProgress) -> PrefillProgress {
            lock.lock()
            let bumped = Swift.max(progress.completedUnitCount, maxCompleted)
            maxCompleted = bumped
            lock.unlock()
            if bumped == progress.completedUnitCount { return progress }
            return PrefillProgress(
                stage: progress.stage,
                completedUnitCount: bumped,
                totalUnitCount: progress.totalUnitCount,
                detail: progress.detail)
        }
    }

    private func startSoloFastPath(
        input: consuming sending LMInput,
        parameters: GenerateParameters,
        promptTail: String?
    ) -> AsyncStream<Generation> {
        let promptTokenCount = input.text.tokens.size
        let hasMediaContent = input.hasMediaContent
        let toolSchemas = input.toolSchemas
        let skipDiskBackedToolPromptSeedBoundary = shouldSkipDiskBackedToolPromptSeedBoundary(
            toolSchemas: toolSchemas,
            disablesGeneratedCacheBoundary: false)
        let fastPathID = UUID()
        var soloParameters = parameters
        soloParameters.extraStopStrings = mergeStopStrings(
            soloParameters.extraStopStrings,
            defaultStopStrings)
        if soloParameters.enableCompiledBatchDecode && !compiledDecodeDeniedForModel && !soloParameters.enableCompiledDecode {
            soloParameters.enableCompiledDecode = true
        }
        if let coordinator = cacheCoordinator {
            let (effMode, effMax) = coordinator.config.resolveKVPolicy(
                kvMode: soloParameters.kvMode,
                maxKVSize: soloParameters.maxKVSize,
                promptTokenCount: promptTokenCount
            )
            soloParameters.kvMode = effMode
            soloParameters.maxKVSize = effMax
        }
        context.jangPressRuntime.recordPromptTokenActivity(
            input.text.tokens.reshaped(-1).asArray(Int.self))

        let (outStream, continuation) = AsyncStream<Generation>.makeStream()
        // Monotonic gate: a prompt-processing counter must never tick backward.
        // Partial / diverging-prefix cache restores (e.g. reasoning-strip
        // templates that shorten history) can compute a `completedBeforePrefill`
        // boundary below an already-emitted frame; clamp every emitted frame's
        // completed count to a running max so the UI %% only ever advances.
        let prefillGate = PrefillMonotonicGate()
        continuation.yield(.prefillProgress(prefillGate.clamp(PrefillProgress(
            stage: .queued,
            completedUnitCount: 0,
            totalUnitCount: promptTokenCount,
            detail: "solo"))))

        let sourceStream: AsyncStream<Generation>
        let generationTask: Task<Void, Never>
        do {
            if let diffusionModel = context.model as? any BlockDiffusionModel {
                let options = diffusionModel.blockDiffusionDefaults
                    .resolving(generationConfig: context.configuration.generationDefaults)
                    .overriding(parameters: soloParameters)
                // Defer iterator construction (and therefore the encoder
                // prefill) into the streaming task — like the AR path below —
                // so `.prefillProgress` frames emitted by the block-diffusion
                // encoder loop reach the consumer LIVE instead of bursting
                // after the (already-prefilled) iterator is returned. The
                // prefill itself is unchanged; only frame timing changes.
                let promptTokenIdsForTail = input.text.tokens.reshaped(-1).asArray(Int.self)
                let deferredParameters = soloParameters
                let deferredOptions = options
                let deferredInputs = SendableBox(
                    (input, diffusionModel, cacheCoordinator))
                let deferredContinuation = continuation
                let makeIterator: @Sendable () throws -> any TokenIteratorProtocol = {
                    let (deferredInput, deferredModel, deferredCoordinator) =
                        deferredInputs.consume()
                    return try BlockDiffusionTokenIterator(
                        input: deferredInput,
                        model: deferredModel,
                        cache: nil,
                        parameters: deferredParameters,
                        options: deferredOptions,
                        cacheCoordinator: deferredCoordinator,
                        prefillProgressHandler: { progress in
                            deferredContinuation.yield(
                                .prefillProgress(prefillGate.clamp(progress)))
                        })
                }
                (sourceStream, generationTask) = generateTaskDeferred(
                    promptTokenCount: promptTokenCount,
                    modelConfiguration: context.configuration,
                    tokenizer: context.tokenizer,
                    promptTokenIds: promptTokenIdsForTail,
                    makeIterator: makeIterator,
                    extraStopStrings: soloParameters.extraStopStrings,
                    promptTail: promptTail,
                    toolSchemas: toolSchemas)
            } else if let strategy = soloParameters.draftStrategy,
                case .nativeMTP(depth: let depth, verifierMode: _) = strategy,
                soloParameters.canUseNativeMTP(for: input)
            {
                guard let nativeModel = context.model as? any NativeMTPModel else {
                    throw NativeMTPRuntimeError.modelDoesNotExposeNativeMTP
                }
                let iterator = try NativeMTPTokenIterator(
                    input: input,
                    model: nativeModel,
                    cache: nil,
                    parameters: soloParameters,
                    depth: depth,
                    cacheCoordinator: cacheCoordinator)
                (sourceStream, generationTask) = generateTask(
                    promptTokenCount: promptTokenCount,
                    modelConfiguration: context.configuration,
                    tokenizer: context.tokenizer,
                    iterator: iterator,
                    extraStopStrings: soloParameters.extraStopStrings,
                    promptTail: promptTail,
                    toolSchemas: toolSchemas)
            } else {
                // Defer TokenIterator construction (and therefore the prompt
                // prefill) into the streaming task so the consumer can observe
                // `.prefillProgress` frames live as prefill proceeds, rather
                // than as one burst after the (already-prefilled) iterator is
                // returned. The prefill itself is unchanged — `prepare` is
                // still invoked exactly once with the full input — so model
                // output and token/s are bit-identical; only the timing of
                // when progress frames reach the consumer changes.
                let promptTokenIdsForTail = input.text.tokens.reshaped(-1).asArray(Int.self)
                let deferredParameters = soloParameters
                let deferredSkipSeedBoundary = skipDiskBackedToolPromptSeedBoundary
                let deferredInputs = SendableBox(
                    (input, context.model, cacheCoordinator))
                let deferredContinuation = continuation
                let makeIterator: @Sendable () throws -> any TokenIteratorProtocol = {
                    let (deferredInput, deferredModel, deferredCoordinator) =
                        deferredInputs.consume()
                    return try TokenIterator(
                        input: deferredInput,
                        model: deferredModel,
                        cache: nil,
                        parameters: deferredParameters,
                        cacheCoordinator: deferredCoordinator,
                        skipDiskBackedToolPromptSeedBoundary: deferredSkipSeedBoundary,
                        prefillProgressHandler: { progress in
                            deferredContinuation.yield(.prefillProgress(prefillGate.clamp(progress)))
                        })
                }
                (sourceStream, generationTask) = generateTaskDeferred(
                    promptTokenCount: promptTokenCount,
                    modelConfiguration: context.configuration,
                    tokenizer: context.tokenizer,
                    promptTokenIds: promptTokenIdsForTail,
                    makeIterator: makeIterator,
                    extraStopStrings: soloParameters.extraStopStrings,
                    promptTail: promptTail,
                    toolSchemas: toolSchemas)
            }
        } catch {
            Self.logger.error(
                "Solo fast path setup failed: \(error.localizedDescription, privacy: .public)"
            )
            continuation.yield(.info(GenerateCompletionInfo(
                promptTokenCount: promptTokenCount,
                generationTokenCount: 0,
                promptTime: 0,
                generationTime: 0,
                stopReason: .cancelled
            )))
            continuation.finish()
            return outStream
        }

        soloFastPathID = fastPathID
        soloFastPathTask = generationTask
        soloFastPathHadMedia = hasMediaContent

        continuation.onTermination = { @Sendable _ in
            generationTask.cancel()
        }
        Task {
            for await generation in sourceStream {
                if case .info(let info) = generation, info.turboQuantCompressions > 0 {
                    turboQuantCompressionCount += info.turboQuantCompressions
                }
                if case .info(let info) = generation,
                   let transition = info.turboQuantCacheTransition
                {
                    lastTurboQuantCacheTransition = transition
                }
                continuation.yield(generation)
            }
            self.finishSoloFastPath(id: fastPathID)
            continuation.finish()
        }

        return outStream
    }

    private func finishSoloFastPath(id: UUID) {
        guard soloFastPathID == id else { return }
        Stream().synchronize()
        let shouldPurgeMediaWorkingSet = soloFastPathHadMedia
        soloFastPathID = nil
        soloFastPathTask = nil
        soloFastPathHadMedia = false
        if shouldPurgeMediaWorkingSet {
            Memory.clearCache()
            stepsSinceMemoryPurge = 0
        }
        if !isShutdown && !waitQueue.isEmpty {
            ensureLoopRunning()
        }
    }

    /// Cancel the active direct B=1 generation and wait until its producer
    /// has stopped touching the shared MLX/Metal stream.
    ///
    /// A serving layer that wraps ``generate(input:parameters:)`` in another
    /// `AsyncStream` cannot safely infer this drain point from its own
    /// consumer termination: dropping the wrapper must cancel the underlying
    /// solo task, but releasing the next-request gate before that task exits
    /// can overlap the cancelled prefill with the next prompt's cache restore.
    /// This method provides the missing cancellation-and-drain handshake.
    /// It is idempotent and is a no-op when no solo generation is active.
    public func cancelActiveSoloGenerationAndWait() async {
        guard let task = soloFastPathTask else { return }
        let id = soloFastPathID
        task.cancel()
        await task.value

        // The bridge that forwards the producer stream normally clears solo
        // ownership. It may not have reached that actor hop yet when
        // `task.value` resumes, so clear the same id here as well. The helper
        // is idempotent; the bridge's later call becomes a no-op.
        if let id {
            finishSoloFastPath(id: id)
        }
    }

    /// Cancel a specific request by ID.
    ///
    /// If the request is still in the wait queue, it is removed immediately.
    /// If it is actively generating, it is marked as finished and its stream
    /// is closed with a `.cancelled` stop reason.
    ///
    /// - Parameter id: The request ID returned by ``submit(input:parameters:)``.
    public func cancel(_ id: BatchRequestID) {
        // Check wait queue first
        if let idx = waitQueue.firstIndex(where: { $0.id == id }) {
            let request = waitQueue.remove(at: idx)
            request.continuation.yield(.info(GenerateCompletionInfo(
                promptTokenCount: request.input.text.tokens.size,
                generationTokenCount: 0,
                promptTime: 0,
                generationTime: 0,
                stopReason: .cancelled
            )))
            request.continuation.finish()
            return
        }

        // Check active slots
        if let idx = activeSlots.firstIndex(where: { $0.id == id }) {
            var slot = activeSlots[idx]
            finishSlot(slot, reason: .cancelled)
            slot.isFinished = true
            activeSlots[idx] = slot
        }
    }

    /// Shut down the engine, finishing all active streams.
    ///
    /// Pending requests receive a `.info` with `.cancelled` stop reason.
    /// Active slots are allowed to complete their current step before finishing.
    ///
    /// Returns only after the engine's producer tasks have actually exited
    /// and their queued GPU work has drained. This is the invariant serving
    /// layers rely on for teardown ordering: "shutdown returned" must mean
    /// "no producer owned by this engine will touch the shared Metal command
    /// queue again". Cancelling alone is not enough — a producer mid-prefill
    /// observes cancellation only at its next chunk/token boundary, and a
    /// model unload that frees weights while that producer is still encoding
    /// segfaults inside the Metal encoder (osaurus cold-load disconnect
    /// crash: disconnect → unload raced the dropped request's prefill).
    public func shutdown() async {
        guard !isShutdown else { return }
        isShutdown = true

        let drainLoopTask = loopTask
        let drainSoloTask = soloFastPathTask
        let shouldPurgeSoloMediaWorkingSet = soloFastPathHadMedia
        loopTask?.cancel()
        loopTask = nil
        soloFastPathTask?.cancel()
        soloFastPathTask = nil
        soloFastPathID = nil
        soloFastPathHadMedia = false

        // Finish all pending requests
        for request in waitQueue {
            request.continuation.yield(.info(GenerateCompletionInfo(
                promptTokenCount: request.input.text.tokens.size,
                generationTokenCount: 0,
                promptTime: 0,
                generationTime: 0,
                stopReason: .cancelled
            )))
            request.continuation.finish()
        }
        waitQueue.removeAll()

        // Finish all active slots
        for slot in activeSlots {
            finishSlot(slot, reason: .cancelled)
        }
        activeSlots.removeAll()

        // Await the cancelled producers before returning. Both tasks observe
        // cancellation at their next boundary (chunked prefill, per-token
        // decode) and end with their own GPU drains, so this is bounded.
        // The producers never call back into this actor, so awaiting them
        // here cannot deadlock; concurrent calls that interleave during the
        // suspension see `isShutdown == true` and fail closed.
        await drainSoloTask?.value
        await drainLoopTask?.value

        // Final fence: producers submit via `asyncEval`, so their last
        // command buffers may still be in flight when the tasks return.
        Stream().synchronize()
        if shouldPurgeSoloMediaWorkingSet {
            Memory.clearCache()
            stepsSinceMemoryPurge = 0
        }
    }

    /// The number of requests currently waiting in the queue.
    public var pendingCount: Int { waitQueue.count }

    /// The number of sequences currently being generated.
    public var activeCount: Int { activeSlots.count + (soloFastPathTask == nil ? 0 : 1) }

    /// Maximum active-slot count observed since engine creation.
    public var activeCountHighWatermarkForDiagnostics: Int { activeCountHighWatermark }

    /// Number of decode compatibility splits observed since engine creation.
    public var decodeCompatibilitySplitCountForDiagnostics: Int {
        decodeCompatibilitySplitCount
    }

    /// Number of successful KVCacheSimple -> TurboQuantKVCache transitions.
    public var turboQuantCompressionCountForDiagnostics: Int {
        turboQuantCompressionCount
    }

    /// Exact before/after cache classes from the most recent successful live
    /// TurboQuant transition, or `nil` when no transition has occurred.
    public var lastTurboQuantCacheTransitionForDiagnostics: TurboQuantCacheTransitionSnapshot? {
        lastTurboQuantCacheTransition
    }

    /// Whether the engine is currently running (has active or pending work).
    public var isRunning: Bool { loopTask != nil || soloFastPathTask != nil }

    var isSoloFastPathActiveForTesting: Bool { soloFastPathTask != nil }

    /// Whether the engine still accepts new generation requests.
    public var isAcceptingRequests: Bool { !isShutdown }

    // MARK: - Scheduling Loop

    /// Start the background scheduling loop if not already running.
    private func ensureLoopRunning() {
        guard loopTask == nil else { return }
        loopTask = Task {
            // Give immediately-following submits a bounded coalescing window
            // before the scheduler enters a potentially long prefill. A plain
            // `Task.yield()` is not deterministic enough: the scheduler can
            // still re-win the actor and monopolize it with the first request's
            // prefill before the second submit appends to `waitQueue`. Keep
            // this disabled for B=1 so single-stream TTFT is unchanged.
            if self.initialAdmissionCoalescingNanos > 0 {
                try? await Task.sleep(nanoseconds: self.initialAdmissionCoalescingNanos)
            }
            await self.schedulingLoop()
        }
    }

    /// Main scheduling loop. Runs until all work is complete.
    private func schedulingLoop() async {
        while !Task.isCancelled {
            // Exit when no work remains
            if waitQueue.isEmpty && activeSlots.isEmpty {
                break
            }

            // 1. Admit new requests from wait queue
            admitPendingRequests()

            // 2. Run one scheduling step
            step()

            // 3. Remove finished slots. Media requests retain their prepared
            //    image/video arrays in `originalInput` for the lifetime of the
            //    slot. Purge only after removing those final strong references;
            //    clearing inside `finishSlot` is too early and lets successive
            //    short media turns accumulate the allocator's released working
            //    set until the generic 256-step purge.
            let finishedMediaSlot = activeSlots.contains {
                $0.isFinished && $0.originalInput.hasMediaContent
            }
            activeSlots.removeAll { $0.isFinished }
            if finishedMediaSlot {
                Memory.clearCache()
                stepsSinceMemoryPurge = 0
            }

            // 4. Periodic memory cleanup
            stepsSinceMemoryPurge += 1
            if stepsSinceMemoryPurge >= memoryPurgeInterval {
                Memory.clearCache()
                stepsSinceMemoryPurge = 0
            }

            // 5. Yield to allow submit/cancel/shutdown/configuration calls
            //    and stream consumers to run. Yielding every token on the B=1
            //    hot path costs measurable throughput, but never yielding lets
            //    a long decode monopolize the actor until max_tokens. Keep the
            //    fairness yield sparse while yielding immediately for queued
            //    admissions or multi-slot fan-out.
            stepsSinceControlPlaneYield += 1
            let shouldYieldForControlPlane =
                stepsSinceControlPlaneYield >= controlPlaneYieldInterval
            if shouldYieldForControlPlane {
                stepsSinceControlPlaneYield = 0
            }
            if !waitQueue.isEmpty || activeSlots.count > 1 || shouldYieldForControlPlane {
                await Task.yield()
            }
        }

        loopTask = nil
    }

    // MARK: - Admission

    /// Move requests from the wait queue into active slots up to `maxBatchSize`.
    private func admitPendingRequests() {
        while activeSlots.count < maxBatchSize && !waitQueue.isEmpty {
            var request = waitQueue.removeFirst()
            context.jangPressRuntime.recordPromptTokenActivity(
                request.input.text.tokens.reshaped(-1).asArray(Int.self))

            // LONG-CTX (2026-04-21): apply the coordinator's KV-sizing
            // defaults before we allocate the slot's cache.
            //
            // Osaurus 0.17.0 removed its per-request `maxKVSize` UI knob
            // with the comment "KV cache sizing is owned end-to-end by
            // vmlx-swift-lm's CacheCoordinator". The coordinator honors
            // that contract here: when `GenerateParameters.kvMode` is
            // `.none` or `maxKVSize` is nil, the coordinator's
            // `defaultKVMode` / `defaultMaxKVSize` fill the gap. Requests
            // that did set their own values are untouched.
            //
            // The default `maxKVSize` is only applied to prompts that
            // exceed `longPromptMultiplier × defaultMaxKVSize` — short
            // chat turns never take a rotating-window hit from a global
            // cap they didn't opt into.
            if let coordinator = cacheCoordinator {
                let promptCount = request.input.text.tokens.size
                let (effMode, effMax) = coordinator.config.resolveKVPolicy(
                    kvMode: request.parameters.kvMode,
                    maxKVSize: request.parameters.maxKVSize,
                    promptTokenCount: promptCount
                )
                if effMode != request.parameters.kvMode {
                    request.parameters.kvMode = effMode
                    Self.logger.info(
                        "Slot \(request.id.description, privacy: .public): applied coordinator defaultKVMode"
                    )
                }
                if effMax != request.parameters.maxKVSize {
                    request.parameters.maxKVSize = effMax
                    Self.logger.info(
                        "Slot \(request.id.description, privacy: .public): applied coordinator defaultMaxKVSize=\(effMax ?? -1) for \(promptCount)-token prompt"
                    )
                }
            }

            // Stage 0: warn if the request asks for a KV-quant mode not yet
            // supported under batched decode (affine / legacy kvBits).
            // TurboQuant is supported and takes effect in `stepPrefill`'s
            // post-prefill compression hook. See BatchQuantize.swift.
            BatchQuantize.wrapNewCacheIfNeeded(
                slotID: request.id,
                parameters: request.parameters
            )

            let cache = context.model.newCache(parameters: request.parameters)
            let hasHybridPool = cache.contains { $0 is HybridPoolCache }

            // DSV4's cache is a composite local-window + compressor/indexer
            // pool. Keep it serialized even when the engine was constructed
            // with maxBatchSize > 1; the transient BatchKVCache wrapper only
            // models ordinary per-token KV and cannot batch the pool branches.
            if hasHybridPool && !activeSlots.isEmpty {
                waitQueue.insert(request, at: 0)
                Self.logger.info(
                    "Slot \(request.id.description, privacy: .public): deferred hybrid-pool request until active DSV4 slot drains"
                )
                break
            }

            // Iter 57: auto-detect hybrid models at admission so SSM
            // companion states round-trip through the coordinator.
            // Without this the caller has to remember to
            // `coordinator.setHybrid(true)` for Qwen3.6-MoE / Nemotron
            // Cascade / other Mamba-attn hybrids — every forgotten call
            // silently skips SSM-state store on finish, which breaks
            // cross-turn cache reuse for hybrid chat. The check is
            // idempotent; non-hybrid models never flip the flag because
            // `CacheFamily.classify` only returns `.heterogeneous` or
            // `.mamba` when a Mamba/SSM layer is present.
            if let coordinator = cacheCoordinator, !coordinator.isHybrid {
                let family = CacheFamily.classify(cache)
                if family == .heterogeneous || family == .mamba || family == .zayaCCA {
                    // Second-line check: at least one layer actually is
                    // a path-dependent cache (Mamba/Arrays SSM or ZAYA
                    // CCA-attention with conv_state+prev_hs) before
                    // flipping the flag. Keeps `.heterogeneous` models
                    // that mix attention + rotating (Gemma-4) from being
                    // misflagged.
                    if cacheContainsPathDependentState(cache) {
                        let topology = ModelCacheTopologySnapshot(cache: cache)
                        coordinator.setHybrid(
                            true,
                            requiresRecurrentSSMCompanion:
                                topology.requiresRecurrentSSMCompanionState)
                        Self.logger.info(
                            "Coordinator flipped to isHybrid=true on first hybrid slot admission"
                        )
                    }
                }
            }

            // 2026-05-04 (DSV4 SWA/CSA/HSA correctness pass) and
            // 2026-05-06 (Gemma4 SWA cache-hit fix):
            // Detect cache topologies the paged tier cannot represent at
            // admission so the coordinator routes prefix reuse through the
            // disk serializer instead.
            //
            // `PagedCacheManager` stores token-sliceable full-attention KV.
            // Mixed Gemma-style caches are eligible only with an exact-leaf
            // rotating ring companion; pool/CCA/affine and other unsupported
            // topologies continue through the typed disk serializer.
            if let coordinator = cacheCoordinator, !coordinator.isPagedIncompatible {
                if cacheCannotUsePagedCoordinatorRestore(cache) {
                    if cacheCanUsePagedWithRotatingCompanion(cache) {
                        coordinator.setPagedBoundaryCompanionRequired(true)
                        Self.logger.info(
                            "Coordinator enabled paged KV with rotating boundary companion on first mixed-cache slot admission"
                        )
                    } else {
                        coordinator.setPagedIncompatible(true)
                        Self.logger.info(
                            "Coordinator flipped to isPagedIncompatible=true on first paged-incompatible slot admission"
                        )
                    }
                }
            }

            let slot = BatchSlot(from: request, cache: cache, stopTokenIDs: stopTokenIDs)
            slot.continuation.yield(.prefillProgress(PrefillProgress(
                stage: .queued,
                completedUnitCount: 0,
                totalUnitCount: slot.promptTokenCount,
                detail: "admitted")))
            activeSlots.append(slot)
            activeCountHighWatermark = max(activeCountHighWatermark, activeSlots.count)
        }
    }

    /// DSV4's HybridPoolCache has mutable compressor/indexer pools that must
    /// be built in one forward for a prompt segment. Chunked prefill is still
    /// correct for ordinary KV and hybrid SSM models; restrict the override to
    /// the hybrid-pool cache family.
    private func effectivePrefillWindow(
        requested: Int,
        input: LMInput,
        cache: [KVCache]
    ) -> Int {
        guard cache.contains(where: { $0 is HybridPoolCache }) else {
            return requested
        }
        return Swift.max(requested, input.text.tokens.size)
    }

    // MARK: - Step Logic

    /// Run one scheduling step: prefill pending slots, then batch-decode active slots.
    private func step() {
        // Phase 1: Process one prefill chunk per slot that's still prefilling.
        // Prefill is done sequentially per slot (each chunk is large, batching
        // prefill chunks of different lengths wastes compute on padding).
        for i in activeSlots.indices where activeSlots[i].phase == .prefill {
            stepPrefill(slotIndex: i)
        }

        // Phase 2: Batch-decode all slots that are in decode phase.
        // Pick slots that are (a) in decode phase AND (b) not already
        // finished. The `!isFinished` check catches the edge case where
        // `stepPrefill` sampled an EOS as the very first decode token —
        // it sets `phase = .decode` before the EOS check, calls
        // `finishSlot`, sets `isFinished = true`, and leaves `nextToken`
        // nil (the non-EOS branch is where `nextToken` gets assigned).
        // Without this guard, `stepBatchDecode` force-unwraps that nil
        // `nextToken` at the `stacked(...)` call and crashes. The
        // `activeSlots.removeAll { $0.isFinished }` sweep runs AFTER
        // this phase, so finished slots remain visible here within the
        // same scheduling iteration.
        let decodeIndices = activeSlots.indices.filter {
            activeSlots[$0].phase == .decode && !activeSlots[$0].isFinished
        }
        if !decodeIndices.isEmpty {
            stepBatchDecode(slotIndices: decodeIndices)
        }
    }

    // MARK: - Prefill

    /// Run the full prefill for a slot using the model's `prepare()` method.
    ///
    /// This delegates to `model.prepare()` which handles:
    /// - **LLM models**: Chunked prefill of the prompt in `prefillStepSize` chunks
    /// - **VLM models**: Vision tower processing, `maskedScatter` of image embeddings,
    ///   and full prompt processing including multimodal fusion
    ///
    /// After prefill, samples the first decode token and transitions the slot to `.decode`.
    private func stepPrefill(slotIndex: Int) {
        var slot = activeSlots[slotIndex]
        let totalPromptUnits = max(0, slot.promptTokenCount)
        slot.continuation.yield(.prefillProgress(PrefillProgress(
            stage: .cacheLookup,
            completedUnitCount: 0,
            totalUnitCount: totalPromptUnits,
            detail: cacheCoordinator == nil ? "disabled" : "checking")))

        // Check multi-tier cache for a prefix match before running full prefill.
        // On cache hit, restore KV state and only prefill remaining tokens.
        //
        // VLM inputs (image/video) are now supported via `slot.mediaSalt`,
        // which mixes a pixel fingerprint into the cache-coordinator key so
        // "same text + same image" hits while "same text + different image"
        // misses. RotatingKVCache is still skipped because its sliding-window
        // semantics are incompatible with partial restore.
        var inputForPrepare = slot.originalInput
        // SLIDING-1: legacy `!hasRotatingCache` guard removed — v2 schema
        // round-trips ring buffer + 5-tuple metaState via `.rotating`
        // LayerKind. Sliding-window models (Gemma3/Gemma4 SWA, Mistral4
        // with maxKVSize, MiMoV2Flash, BaichuanM1, Qwen3.5-VL inherited)
        // now hit paged + L2 disk on the same path as standard KV.
        if let coordinator = cacheCoordinator {
            let rawTokenIds = slot.originalInput.text.tokens.asArray(Int.self)
            var tokenIds = rawTokenIds
            var usesPostPrepareAlias = false
            if slot.originalInput.requiresPostPrepareCacheKey {
                if let effectiveTokens = coordinator.resolvePostPrepareCacheKeyAlias(
                    rawTokens: rawTokenIds,
                    mediaSalt: slot.mediaSalt)
                {
                    tokenIds = effectiveTokens
                    usesPostPrepareAlias = true
                    Self.logger.info(
                        "Slot \(slot.id.description, privacy: .public): resolved post-prepare cache-key alias for \(rawTokenIds.count) raw tokens -> \(effectiveTokens.count) effective tokens"
                    )
                } else {
                    Self.logger.info(
                        "Slot \(slot.id.description, privacy: .public): skipped pre-prepare cache fetch because this input requires model-derived effective prompt tokens"
                    )
                }
            }
            guard !slot.originalInput.requiresPostPrepareCacheKey || usesPostPrepareAlias else {
                activeSlots[slotIndex] = slot
                return stepPrefillAfterCacheLookup(slotIndex: slotIndex, inputForPrepare: inputForPrepare)
            }
            let requiresDiskBackedRestore = cacheRequiresDiskBackedCoordinatorRestore(slot.cache)
            // Scope the fetch result to this coordinator branch. Disk-backed
            // topologies must all attempt restore; safety is decided by the
            // topology-aware restore path below, not a model-name denylist.
            do {
                let result = coordinator.fetch(
                    tokens: tokenIds,
                    mediaSalt: slot.mediaSalt,
                    skipExactDiskBoundary: requiresDiskBackedRestore,
                    preferredDiskBoundaries: slot.originalInput
                        .cacheStablePrefixTokenCounts)
                if case .hit(
                    let matchedTokens, let remaining, let detail, let blocks,
                    let ssmStates, let diskArrays) = result
                {
                    var restored = false
                    if !blocks.isEmpty {
                        let restoredTokens = restoreLayerData(from: blocks, into: slot.cache)
                        coordinator.release(blocks: blocks)
                        if restoredTokens > 0 {
                            if let ssm = ssmStates {
                                restoreSSMStates(
                                    ssm, into: slot.cache, boundary: matchedTokens)
                            }
                            restored = true
                            slot.continuation.yield(.prefillProgress(PrefillProgress(
                                stage: .cacheRestore,
                                completedUnitCount: min(restoredTokens, totalPromptUnits),
                                totalUnitCount: totalPromptUnits,
                                detail: detail.rawValue)))
                            Self.logger.info(
                                "Cache \(detail.rawValue) hit for slot \(slot.id): restored \(restoredTokens) tokens, prefilling \(remaining.count) remaining"
                            )
                        }
                    }

                    // Disk cache restore (blocks are empty, arrays are present)
                    if let diskArrays, !restored {
                        // Under `MLXCacheIOLock`: the restore's Metal evals
                        // must not interleave with other cache-adjacent GPU
                        // submitters (serving layers tokenize the next
                        // request's input under this lock). See the
                        // TokenIterator disk-restore path for the abort this
                        // prevents.
                        let diskRestored = MLXCacheIOLock.withSerializedMLXCacheIO {
                            () -> Int in
                            let count = restoreFromDiskArrays(diskArrays, into: &slot.cache)
                            if count > 0 {
                                // The v2 disk format has NO LayerKind for the
                                // GatedDeltaNet linear-attention (ArraysCache)
                                // state used by qwen3.5/ornith, so
                                // restoreFromDiskArrays serializes those layers
                                // as `.skip` and leaves them at their initial
                                // (empty) value. For such caches the companion
                                // SSM sidecar is the ONLY carrier of that
                                // recurrent state and MUST be applied even at
                                // fmtV>=2 — otherwise a cross-turn restore feeds
                                // the model an empty GatedDeltaNet state and the
                                // reused turn diverges from the cache-off ground
                                // truth. mamba/zayaCCA DO round-trip in v2, so
                                // this extra apply is gated on the cache actually
                                // holding an Arrays state (and is a same-value
                                // no-op otherwise). Mirrors the solo TokenIterator
                                // disk-restore path in Evaluate.swift.
                                let cacheHasArraysState = slot.cache.contains {
                                    String(describing: type(of: $0)).contains("Arrays")
                                }
                                if let ssm = ssmStates,
                                   TQDiskSerializer.formatVersion(of: diskArrays) < 2
                                    || cacheHasArraysState
                                {
                                    restoreSSMStates(
                                        ssm, into: slot.cache, boundary: matchedTokens)
                                }
                                // The materializing eval below is what
                                // actually submits the restore compute;
                                // keep it inside the lock so the whole
                                // restore is serialized, not just its
                                // graph construction.
                                MLX.eval(slot.cache)
                            }
                            return count
                        }
                        if diskRestored > 0 {
                            // 2026-04-27 fix: materialize restored cache state
                            // in its own command buffer BEFORE prefill builds
                            // its forward graph. Disk restore produces lazy
                            // MLXArrays (asType conversions, TQ component
                            // deserialization, mamba state copies). Without
                            // an explicit eval here, the next prefill forward
                            // builds a single command buffer containing both
                            // the cache materialization AND the model's
                            // custom kernel dispatches — combined allocation
                            // pressure can trigger `mlx::core::metal::Device::
                            // clear_library` mid-encode, evicting a kernel
                            // pipeline that's still referenced by the
                            // in-flight buffer →
                            // `notifyExternalReferencesNonZeroOnDealloc`
                            // assertion (osaurus repro 2026-04-27 on Qwen-3.6
                            // 35B A3B MXFP4 with warm disk-tier KV cache).
                            // Eager eval forces the cache state into GPU
                            // memory in a SEPARATE command buffer that
                            // commits before prefill encoding starts.
                            // (The eval itself runs above, inside the
                            // MLXCacheIOLock region.)
                            restored = true
                            slot.continuation.yield(.prefillProgress(PrefillProgress(
                                stage: .cacheRestore,
                                completedUnitCount: min(diskRestored, totalPromptUnits),
                                totalUnitCount: totalPromptUnits,
                                detail: detail.rawValue)))
                            Self.logger.info(
                                "Cache \(detail.rawValue) hit for slot \(slot.id): restored \(diskRestored) tokens from disk, prefilling \(remaining.count) remaining"
                            )
                        }
                    }

                    if restored {
                        if usesPostPrepareAlias {
                            slot.cachePromptTokenIds = tokenIds
                            slot.cachePromptUsesPostPrepareKey = true
                        }
                        // Two classes of partial-restore that must roll back to
                        // full prefill rather than feed "remaining" tokens into
                        // model.prepare — correctness over speed in both cases:
                        //
                        // 1. Media content: model-side media splice code aligns
                        //    placeholder token spans against image/video/audio
                        //    embedding tensors. Splitting that region across a
                        //    cache boundary can make the splice path crash or
                        //    attach the wrong media state.
                        //
                        // 2. Exact full hits on hybrid SSM: the restored SSM
                        //    state already includes the last token's recurrence
                        //    contribution. The remaining.isEmpty path has to
                        //    re-feed the last token to seed logits, which would
                        //    double-count that recurrence. Partial disk hits are
                        //    different: a complete state at boundary N plus
                        //    prefill over [N...M] is the intended Markov resume
                        //    path for MambaCache, ArraysCache, and ZayaCCACache.
                        // Full disk hit on hybrid-SSM is ALSO unsafe: the
                        // restored SSM state already includes the last
                        // token's recurrence contribution, so the
                        // remaining.isEmpty branch's "trim KV by 1 and
                        // re-feed last token" recipe double-counts the
                        // last token's SSM update. Result: logits sample
                        // EOS first, decode emits zero tokens (StabilityBench
                        // S2 reproducer on Qwen3.6-35B-A3B-JANGTQ4 2026-05-01).
                        // Same SSM-state path-dependence rationale as the
                        // remaining.nonEmpty case below.
                        let unsafePartial =
                            slot.originalInput.cacheHitSuffixContainsMediaPlaceholder(remaining)
                        // Only standalone rotating / sliding-window caches (Gemma,
                        // Mistral SWA) are proven to restore exactly and take the
                        // standard trim+re-feed fast path on a full hit. Keep the
                        // conservative full-prefill rollback for every other
                        // disk-backed topology — path-dependent recurrent, TurboQuant/
                        // Quantized, HybridPool — whose exact-restore is unverified.
                        let unsafeFullHit =
                            remaining.isEmpty && requiresDiskBackedRestore
                            && !cacheHasStandaloneRotatingWindowState(slot.cache)
                        if unsafePartial {
                            let slotIDStr = slot.id.description
                            Self.logger.info(
                                "Slot \(slotIDStr, privacy: .public): cache hit — rolling back to full prefill (media placeholder tokens remain in cache-hit suffix)"
                            )
                            slot.cache = context.model.newCache(parameters: slot.parameters)
                            inputForPrepare = slot.originalInput
                        } else if unsafeFullHit {
                            let promptLen = tokenIds.count
                            let seedBoundary = promptLen - 1
                            if seedBoundary > 0,
                               let last = tokenIds.last,
                               let seedSSM = coordinator.ssmStateCache.fetch(
                                tokens: tokenIds,
                                boundary: seedBoundary,
                                mediaSalt: slot.mediaSalt)
                            {
                                let cacheOffset = slot.cache.first?.offset ?? promptLen
                                let trimNeeded = cacheOffset - seedBoundary
                                if trimNeeded > 0 {
                                    for layer in slot.cache where layer.isTrimmable {
                                        _ = layer.trim(trimNeeded)
                                    }
                                    MLX.eval(slot.cache)
                                }
                                restoreSSMStates(
                                    seedSSM, into: slot.cache, boundary: seedBoundary)
                                MLX.eval(slot.cache)
                                let lastToken = MLXArray([Int32(last)])
                                    .expandedDimensions(axis: 0)
                                inputForPrepare = LMInput(
                                    text: LMInput.Text(tokens: lastToken),
                                    image: nil, video: nil)
                            } else {
                                let slotIDStr = slot.id.description
                                Self.logger.info(
                                    "Slot \(slotIDStr, privacy: .public): cache hit — rolling back to full prefill (path-dependent full cache hit missing seed-boundary SSM state)"
                                )
                                slot.cache = context.model.newCache(parameters: slot.parameters)
                                inputForPrepare = slot.originalInput
                            }
                        } else if remaining.isEmpty, let last = tokenIds.last {
                            // Full cache hit — feed last token to seed decode.
                            // Tensor must be 2D `[1, 1]`: the Qwen3_5 VLM
                            // `Qwen35Language.LanguageModel` reads
                            // `inputs.dim(1)` during position-id compute and
                            // crashes MLX with `SmallVector out of range`
                            // (array.cpp:335) on a 1D input. All other
                            // model forwards either broadcast 2D already
                            // or tolerate the extra leading axis — matches
                            // the sibling `Evaluate.swift:825` fix.
                            //
                            // Trim cache offset back to (promptLen - 1) before
                            // re-feeding the last token. Disk-tier hits restore
                            // KV for `promptLen + previousDecodeLen` entries
                            // (storage runs at finishSlot AFTER decode), so
                            // without trimming the model would re-feed the
                            // last prompt token at position `promptLen +
                            // previousDecodeLen` — RoPE then rotates by the
                            // wrong angle and the resulting logits typically
                            // sample EOS first-token, yielding 0 generated
                            // tokens (BENCH_BATCH_DISK_RESTORE 2026-04-24).
                            // Trim is a no-op for paged-tier hits because
                            // their `remaining.isEmpty == true` branch is
                            // only reached when the matched count already
                            // equals promptLen and offset already equals
                            // promptLen.
                            let promptLen = tokenIds.count
                            let cacheOffset = slot.cache.first?.offset ?? promptLen
                            let trimNeeded = cacheOffset - (promptLen - 1)
                            if trimNeeded > 0 {
                                for layer in slot.cache where layer.isTrimmable {
                                    _ = layer.trim(trimNeeded)
                                }
                                // 2026-05-01: force materialization of trim mutations
                                // before the prefill seed-forward consumes the cache.
                                // Trim is lazy; without this MLX call, trim's pending
                                // state changes get folded into the SAME command
                                // buffer that dispatches the JANGTQ kernels for the
                                // seed forward. The buffer's allocation pressure
                                // mid-encode can trigger Metal's library-cache
                                // eviction while the kernel pipeline is still
                                // referenced by the in-flight buffer →
                                // `notifyExternalReferencesNonZeroOnDealloc` crash
                                // inside `Device::clear_library`. Reproducer: 2nd
                                // request whose prompt is FULLY in disk-tier cache
                                // (so this remaining.isEmpty branch fires, trim
                                // runs, and a one-token forward immediately follows).
                                //
                                // Sibling to the disk-restore materialization at line
                                // 778 — that closes the `remaining.nonEmpty` paths;
                                // this one closes the `remaining.isEmpty + trim`
                                // path the prior fix missed.
                                MLX.eval(slot.cache)
                            }
                            let lastToken = MLXArray([Int32(last)])
                                .expandedDimensions(axis: 0)
                            inputForPrepare = LMInput(
                                text: LMInput.Text(tokens: lastToken),
                                image: nil, video: nil)
                        } else if remaining.isEmpty {
                            // Defensive fallback: no last token → roll back.
                            slot.cache = context.model.newCache(parameters: slot.parameters)
                            inputForPrepare = slot.originalInput
                            Self.logger.error(
                                "Slot \(slot.id.description, privacy: .public): cache .hit returned empty tokenIds — rolling back to full prefill"
                            )
                        } else {
                            // Remaining tokens path — same 2D shape contract.
                            let remainingArray = MLXArray(remaining.map { Int32($0) })
                                .expandedDimensions(axis: 0)
                            inputForPrepare = LMInput(
                                text: LMInput.Text(tokens: remainingArray),
                                image: nil, video: nil)
                        }
                    }
                }
            }
        }

        stepPrefillAfterCacheLookup(slotIndex: slotIndex, inputForPrepare: inputForPrepare, slot: slot)
    }

    private func stepPrefillAfterCacheLookup(
        slotIndex: Int,
        inputForPrepare: LMInput,
        slot initialSlot: BatchSlot? = nil
    ) {
        var slot = initialSlot ?? activeSlots[slotIndex]

        let totalPromptUnits = max(0, slot.promptTokenCount)
        let remainingPromptUnits = max(0, inputForPrepare.text.tokens.size)
        slot.continuation.yield(.prefillProgress(PrefillProgress(
            stage: .prefill,
            completedUnitCount: max(0, totalPromptUnits - remainingPromptUnits),
            totalUnitCount: totalPromptUnits,
            detail: "running")))

        // Prefill: either full input (cache miss) or remaining tokens (cache hit).
        let prepareResult: PrepareResult
        do {
            let completedBeforePrefill = max(0, totalPromptUnits - remainingPromptUnits)
            let progressAccumulator = PrefillProgressAccumulator(
                continuation: slot.continuation,
                completedBeforePrefill: completedBeforePrefill,
                totalPromptUnits: totalPromptUnits)
            prepareResult = try PrefillProgressReporter.withHandler({
                progressAccumulator.report(completedInPrepare: $0)
            }) {
                try context.model.prepare(
                    inputForPrepare,
                    cache: slot.cache,
                    windowSize: effectivePrefillWindow(
                        requested: slot.prefillStepSize,
                        input: inputForPrepare,
                        cache: slot.cache))
            }
        } catch {
            // Prefill failed (e.g., invalid input) — finish with cancellation
            finishSlot(slot, reason: .cancelled)
            slot.isFinished = true
            activeSlots[slotIndex] = slot
            return
        }

        slot.continuation.yield(.prefillProgress(PrefillProgress(
            stage: .complete,
            completedUnitCount: totalPromptUnits,
            totalUnitCount: totalPromptUnits,
            detail: "decode_ready")))

        // Extract the first generated token from the prepare result
        let firstToken: MLXArray
        switch prepareResult {
        case .tokens(let remainingText):
            // Seed the processor with the full prompt tokens.
            let promptTokens = slot.originalInput.text.tokens
            slot.processor?.prompt(promptTokens)

            // LLM path: prepare() consumed all but the last chunk, returned remaining tokens.
            // Run the last chunk through the model to get logits for the first decode token.
            let result = context.model(
                remainingText[text: .newAxis], cache: slot.cache, state: nil)
            MLX.eval(slot.cache)
            let logits = result.logits[0 ..< 1, -1, 0...]
            firstToken = slot.sampleToken(from: logits)

        case .logits(let result):
            if let effectivePromptTokens = result.effectivePromptTokens,
               !effectivePromptTokens.isEmpty
            {
                slot.cachePromptTokenIds = effectivePromptTokens
                slot.cachePromptUsesPostPrepareKey = true
                if slot.originalInput.requiresPostPrepareCacheKey {
                    cacheCoordinator?.recordPostPrepareCacheKeyAlias(
                        rawTokens: slot.originalInput.text.tokens.reshaped(-1).asArray(Int.self),
                        effectiveTokens: effectivePromptTokens,
                        mediaSalt: slot.mediaSalt)
                }
                let promptTokens = MLXArray(effectivePromptTokens.map { Int32($0) })
                    .expandedDimensions(axis: 0)
                slot.processor?.prompt(promptTokens)
            } else {
                let promptTokens = slot.originalInput.text.tokens
                slot.processor?.prompt(promptTokens)
            }
            // VLM path: prepare() already ran the full prompt and returned logits directly.
            let logits = result.logits[0 ..< 1, -1, 0...]
            firstToken = slot.sampleToken(from: logits)
        }

        // Capture the cache exactly at the prompt boundary. The first sampled
        // token has not been fed back into the model yet, so this snapshot is
        // safe for paged and L2 disk storage under the prompt-token key.
        slot.promptCacheSnapshot = makePromptBoundaryCacheSnapshot(from: slot.cache)

        let tokenID = firstToken.item(Int.self)

        slot.phase = .decode
        slot.decodeStartTime = Date()
        slot.pendingTokens = MLXArray([Int32]()) // clear

        // Check EOS on first generated token before yielding
        if stopTokenIDs.contains(tokenID) {
            finishSlot(slot, reason: .stop)
            slot.isFinished = true
        } else {
            slot.continuation.yield(.token(tokenID))
            slot.generatedTokenCount += 1
            slot.generatedTokenIds.append(tokenID)
            slot.nextToken = firstToken

            if let maxTokens = slot.maxTokens, slot.generatedTokenCount >= maxTokens {
                finishSlot(slot, reason: .length)
                slot.isFinished = true
            }
        }

        if !slot.isFinished {
            // Hybrid-SSM cross-turn cache seed: after prefill completes for
            // a hybrid-SSM slot, snapshot the SSM companion state keyed by
            // the prompt length and store it into the coordinator's
            // ``SSMStateCache``. This runs after the first token has been
            // yielded so TTFT does not pay the prompt-boundary bookkeeping.
            if let coordinator = cacheCoordinator, coordinator.isHybrid {
                // ZayaCCACache's `conv_state` + `prev_hs` are path-dependent
                // and round-trip through extractSSMStates / restoreSSMStates
                // (see CacheHelpers.swift:293-300). Include it here so the
                // post-prefill snapshot fires for ZAYA1 slots — without this
                // gate the snapshot path was only firing for Mamba/Arrays
                // hybrids and ZAYA's CCA state would never reach the
                // SSMStateCache for cross-turn restore.
                let hasSSM = slot.cache.contains {
                    $0 is MambaCache || $0 is ArraysCache || $0 is ZayaCCACache
                }
                if hasSSM {
                    let promptTokens = slot.cachePromptTokenIds
                    let ssmStates = extractSSMStates(from: slot.cache)
                    if !ssmStates.isEmpty {
                        coordinator.ssmStateCache.store(
                            ssmStates: ssmStates,
                            tokens: promptTokens,
                            boundary: promptTokens.count,
                            mediaSalt: slot.mediaSalt,
                            persistToDisk: false
                        )
                        Self.logger.debug(
                            "Slot \(slot.id.description, privacy: .public): stored SSM seed at boundary=\(promptTokens.count) (\(ssmStates.count) state arrays)"
                        )
                    }
                }
            }

            // Stage 0: KV-quant compression hook. For requests with
            // `kvMode: .turboQuant(...)`, this swaps `KVCacheSimple` layers for
            // `TurboQuantKVCache` once the first KV layer's offset exceeds the
            // TQ minimum threshold. Running after `yield(.token)` keeps TQ's
            // one-time encode/decode cost out of first-token latency while
            // preserving the compressed path for sustained decode.
            maybeCompressSlotCache(&slot)

            // Stage 1B.3: compile-decode promotion hook.
            self.maybePromoteToCompiledDecode(slot: &slot)
        }

        activeSlots[slotIndex] = slot
    }

    // MARK: - Compiled Decode Step (Stage 1B.3)

    /// Run a single decode step through a compiled forward closure for the
    /// `maxBatchSize == 1` path.
    ///
    /// The closure was captured in ``maybePromoteToCompiledDecode`` after
    /// prefill. It expects `[tokens]` as input and returns `[logits]` —
    /// both single-element arrays. `tokens` shape is `[1]` (one token for
    /// one sequence), `logits` shape is `[1, 1, V]`.
    ///
    /// Everything after the forward call (sampling, EOS checking, yield,
    /// per-step quantization hook) matches `stepBatchDecode`'s sampling
    /// loop. Duplicating rather than refactoring for now — the compiled
    /// path will grow its own concerns in Stage 1B.4 (liveness masks,
    /// multi-row routing) and merging logic prematurely would tangle
    /// both.
    private func stepCompiledDecode(
        slotIndex: Int,
        forward: @Sendable ([MLXArray]) -> [MLXArray]
    ) {
        var slot = activeSlots[slotIndex]
        guard let nextToken = slot.nextToken else {
            Self.logger.error(
                "Slot \(slot.id.description, privacy: .public): stepCompiledDecode called without nextToken"
            )
            return
        }

        // Run the compiled forward pass. Closure captures the slot's
        // CompilableKVCache layers as its state; mutating them via
        // `_updateInternal` is how the trace advances.
        let result = forward([nextToken])
        guard result.count == 1 else {
            Self.logger.error(
                "Slot \(slot.id.description, privacy: .public): compiled forward returned \(result.count) outputs, expected 1"
            )
            return
        }

        // result[0] shape: [1, 1, V]. Force materialisation so we can
        // read the sampled token ID below.
        MLX.eval(result[0])

        // Extract as [1, V] for the processor/sampler contract.
        let logits = result[0][0 ..< 1, 0, 0...]
        let token = slot.sampleToken(from: logits)
        let tokenID = token.item(Int.self)

        // Stage 0: per-step KV-quant hook. For compile+TQ this is a no-op
        // because compile requires `.simple` family (TQ compression would
        // have already run during prefill promotion or be blocked). Kept
        // for symmetry with `stepBatchDecode` so any future compile+quant
        // mode finds the hook wired in.
        maybeCompressSlotCache(&slot)

        // Stop conditions (same rules as uncompiled path).
        if stopTokenIDs.contains(tokenID) {
            finishSlot(slot, reason: .stop)
            slot.isFinished = true
        } else {
            slot.continuation.yield(.token(tokenID))
            slot.generatedTokenCount += 1
            slot.generatedTokenIds.append(tokenID)
            slot.nextToken = token

            if let maxTokens = slot.maxTokens, slot.generatedTokenCount >= maxTokens {
                finishSlot(slot, reason: .length)
                slot.isFinished = true
            }
        }

        activeSlots[slotIndex] = slot
    }

    // MARK: - Multi-Batch Compile Promotion (Stage 1B.4 scaffold)

    /// Stage 1B.4 hook — multi-batch compile promotion.
    ///
    /// **Status (2026-05-02):** intentional no-op. The full
    /// implementation (per-bucket `BucketHandle`, shared `[B, H,
    /// maxLen, D]` cache buffers, slot↔row lifecycle, liveness-mask
    /// plumbing through Compilable cache classes, multi-bucket
    /// fallback ladder) is deferred to its own iteration. Half-shipping
    /// would risk regressing the verified Stage 1B.3 single-slot path.
    /// See `STAGE-1B4-DESIGN-2026-05-02.md` for the architecture.
    ///
    /// What the full implementation will do here:
    ///   1. Look up an existing `BucketHandle` for `slot`'s cache family
    ///      and `compiledMaxCacheLength`, or build a new one.
    ///   2. If the bucket has a free row, assign it to this slot and
    ///      view the slot's cache as a row of the bucket's shared buffer.
    ///   3. Build the bucket's compiled forward closure on first admit.
    ///   4. Store the bucket reference on the slot so `stepBatchDecode`
    ///      can route through the compiled trace.
    ///
    /// Falls back to the uncompiled `stepBatchDecode` path silently —
    /// no error, just no compile speedup. That's the production
    /// behaviour today for `maxBatchSize > 1` deployments and is what
    /// callers expect.
    private func maybePromoteToBucket(slot: inout BatchSlot) {
        // Intentional no-op. See STAGE-1B4-DESIGN-2026-05-02.md.
        _ = slot
        return
    }

    // MARK: - Compile-Decode Promotion (Stage 1B.3)

    /// Promote a slot's cache to `CompilableKVCache` layers and build a
    /// compiled forward closure when all preconditions hold.
    ///
    /// Called from `stepPrefill` after `BatchQuantize.maybeCompress` runs
    /// (so TurboQuant-compressed slots are correctly excluded — their
    /// family is `.turboQuant`, not `.simple`).
    ///
    /// Preconditions (all must hold for promotion):
    ///  - `slot.parameters.enableCompiledBatchDecode == true`
    ///  - `self.maxBatchSize == 1` — Stage 1B.3 scope. `maxBatchSize > 1`
    ///    routes to `maybePromoteToBucket(slot:)` (Stage 1B.4 scaffold;
    ///    currently a no-op until full per-bucket cache + lifecycle
    ///    lands — see `STAGE-1B4-DESIGN-2026-05-02.md`).
    ///  - `HardwareInfo.isCompiledDecodeSupported` — dodges MLX#3329 on
    ///    affected macOS Tahoe Metal driver builds.
    ///  - `CacheFamily.classify(slot.cache) == .simple` — compile is only
    ///    wired for KVCacheSimple layers today.
    ///  - Every layer is an actual `KVCacheSimple` (not already
    ///    `CompilableKVCache`) so the `CompilableKVCache(from:)` conversion
    ///    has valid state to copy.
    ///
    /// When all hold, every layer is swapped for
    /// `CompilableKVCache(from: originalLayer, maxLength: compiledMaxCacheLength)`
    /// and the compiled forward closure is built via
    /// ``BatchCompile/compileForward(model:cacheRef:)``. `stepBatchDecode`
    /// then routes this slot's decode tokens through the closure.
    private func maybePromoteToCompiledDecode(slot: inout BatchSlot) {
        guard slot.parameters.enableCompiledBatchDecode else { return }
        guard !compiledDecodeDeniedForModel else { return }
        // Stage 1B.3 scope: single-slot path. Multi-slot promotion is
        // routed through `maybePromoteToBucket(slot:)` once Stage 1B.4
        // wires up `BucketHandle`. Today that helper is a no-op so
        // multi-slot deployments stay on the uncompiled `stepBatchDecode`
        // path. See STAGE-1B4-DESIGN-2026-05-02.md.
        if self.maxBatchSize > 1 {
            self.maybePromoteToBucket(slot: &slot)
            return
        }
        guard HardwareInfo.isCompiledDecodeSupported else { return }

        let family = CacheFamily.classify(slot.cache)
        let slotIDString = slot.id.description

        switch family {
        case .simple:
            // Stage 1B.3 path. Promote KVCacheSimple layers to
            // CompilableKVCache(from:) then build the compiled forward.
            // Skip if layers are already CompilableKVCache (e.g., restored
            // via cache coordinator — not yet implemented but harmless
            // guard).
            guard slot.cache.allSatisfy({ $0 is KVCacheSimple }) else { return }

            let maxLen = slot.parameters.compiledMaxCacheLength ?? 4096
            let promoted: [KVCache] = slot.cache.map { layer in
                CompilableKVCache(from: layer, maxLength: maxLen) as KVCache
            }
            MLX.eval(promoted)
            slot.cache = promoted
            slot.compiledForward = BatchCompile.compileForward(
                model: context.model, cacheRef: promoted)

            Self.logger.debug(
                "Slot \(slotIDString, privacy: .public): promoted to compiled decode via .simple family (maxLen=\(maxLen))"
            )

        case .turboQuant:
            // Stage 2 SHIPPED (iter 21). Root cause of the long-
            // investigated drift was `applyRotaryPosition` falling
            // through to the Int `cache.offset` for TurboQuant layers
            // instead of the MLXArray offset counter. Fixed in
            // `RoPEApplication.swift`. Multi-step compiled-vs-uncompiled
            // drift dropped from 6-13% to FP precision (~5e-7).
            //
            // All slots must be in compressed phase for compile to
            // engage — short-prompt slots still in fill phase run the
            // uncompiled path (next per-step maybeCompress hook will
            // compress them when threshold crosses).
            let allCompressed = slot.cache.allSatisfy { layer in
                (layer as? TurboQuantKVCache)?.phase == .compressed
            }
            guard allCompressed else { return }

            let promoted: [KVCache] = slot.cache.map { layer in
                CompilableTurboQuantKVCache(from: layer as! TurboQuantKVCache) as KVCache
            }
            MLX.eval(promoted)
            slot.cache = promoted
            slot.compiledForward = BatchCompile.compileForward(
                model: context.model, cacheRef: promoted)

            Self.logger.debug(
                "Slot \(slotIDString, privacy: .public): promoted to compiled decode via .turboQuant family"
            )

        case .rotating:
            // Stage 3 (iter 12 built, iter 13 wired). Sliding-window
            // models — Gemma3 / Gemma4 SWA layers / Mistral4 with
            // maxKVSize / MiMoV2Flash / BaichuanM1 / Qwen3.5-VL inherited —
            // promote each RotatingKVCache layer to
            // CompilableRotatingKVCache and build the compiled forward.
            //
            // Stage 3 verified drift:
            //   - Linear single-step: bit-identical (4.6e-7)
            //   - Growth-boundary 10 steps: ~8% (from 30% pre-fix)
            //   - Wrap-around 20 steps: ~3% (below 5% bar — from 68% pre-fix)
            guard slot.cache.allSatisfy({ $0 is RotatingKVCache && !($0 is CompilableRotatingKVCache) }) else {
                return
            }

            let promoted: [KVCache] = slot.cache.map { layer in
                CompilableRotatingKVCache(from: layer as! RotatingKVCache) as KVCache
            }
            MLX.eval(promoted)
            slot.cache = promoted
            slot.compiledForward = BatchCompile.compileForward(
                model: context.model, cacheRef: promoted)

            Self.logger.debug(
                "Slot \(slotIDString, privacy: .public): promoted to compiled decode via .rotating family"
            )

        case .cacheList:
            // Stage 5 (iter 22 wiring). Composite cache for FalconH1 /
            // BaichuanM1. Promote each CacheList layer to
            // CompilableCacheList; the composite's sub-caches get
            // promoted individually (KVCacheSimple → CompilableKVCache,
            // RotatingKVCache → CompilableRotatingKVCache, etc).
            //
            // Fall back to uncompiled if any sub-cache can't be promoted
            // (CompilableCacheList.allSubCachesCompileReady == false).
            let promoted: [KVCache] = slot.cache.map { layer in
                if let list = layer as? CacheList, !(layer is CompilableCacheList) {
                    return CompilableCacheList(from: list) as KVCache
                }
                return layer
            }
            let allReady = promoted.allSatisfy {
                ($0 as? CompilableCacheList)?.allSubCachesCompileReady ?? false
            }
            guard allReady else {
                Self.logger.debug(
                    "Slot \(slotIDString, privacy: .public): .cacheList compile skipped — not all sub-caches compile-ready"
                )
                return
            }
            MLX.eval(promoted)
            slot.cache = promoted
            slot.compiledForward = BatchCompile.compileForward(
                model: context.model, cacheRef: promoted)
            Self.logger.debug(
                "Slot \(slotIDString, privacy: .public): promoted to compiled decode via .cacheList family"
            )

        case .mamba, .zayaCCA, .heterogeneous:
            // Stage 4 pending (hybrid trace grouping is its own spec).
            //
            // Gemma3/Gemma4 hit this branch via `.heterogeneous` because
            // their cache mixes KVCacheSimple (full_attention) +
            // RotatingKVCache (sliding_attention). Decode runs through
            // the existing uncompiled BatchKVCache path.
            //
            // ZAYA1 (`.zayaCCA`) is also intentionally uncompiled in v1 —
            // CCA conv_qk + state writeback would need its own compilable
            // variant before joining the trace cache. Future Stage 6 work.
            Self.logger.debug(
                "Slot \(slotIDString, privacy: .public): compile skipped — family=\(family.description) (stage pending or heterogeneous)"
            )
            return
        }
    }

    // MARK: - Batched Decode

    /// Run one batched decode step across all decode-phase slots.
    ///
    /// Constructs `[B, 1]` input from each slot's next token, builds per-layer
    /// ``BatchKVCache`` wrappers, runs one model forward pass, then samples
    /// independently per sequence.
    private func stepBatchDecode(slotIndices: [Int]) {
        if slotIndices.count > 1 {
            let grouped = decodeCompatibilityGroups(slotIndices: slotIndices)
            if grouped.count > 1 {
                decodeCompatibilitySplitCount += 1
                Self.logger.debug(
                    "Splitting decode into \(grouped.count, privacy: .public) cache-compatible groups"
                )
                for group in grouped {
                    stepBatchDecode(slotIndices: group)
                }
                return
            }
        }

        // Stage 1B.3: single-slot compiled decode path. When this slot was
        // promoted to a compiled-forward during `stepPrefill`, route through
        // the compiled closure instead of constructing per-step BatchKVCache
        // wrappers. This path only engages at `maxBatchSize == 1` (the
        // promotion gate), so `slotIndices.count` is strictly 1 here.
        if slotIndices.count == 1,
            let forward = activeSlots[slotIndices[0]].compiledForward
        {
            stepCompiledDecode(slotIndex: slotIndices[0], forward: forward)
            return
        }

        // Defensive filter: drop any slot whose `nextToken` is nil
        // instead of force-unwrapping. The caller already filters on
        // `phase == .decode && !isFinished`, so this path SHOULD never
        // surface a nil — but a future regression (new stepPrefill
        // branch that transitions to .decode without setting
        // nextToken, cancel race, etc.) would crash the whole engine
        // instead of dropping one slot. Log when it happens so the
        // invariant violation is observable, not silent.
        let liveIndices = slotIndices.compactMap { idx -> (Int, MLXArray)? in
            if let tok = self.activeSlots[idx].nextToken {
                return (idx, tok)
            }
            Self.logger.error(
                "Slot \(self.activeSlots[idx].id.description, privacy: .public): nil nextToken in stepBatchDecode — dropping from batch"
            )
            return nil
        }
        guard !liveIndices.isEmpty else { return }
        let slotIndices = liveIndices.map { $0.0 }
        let tokenArrays = liveIndices.map { $0.1 }
        let B = slotIndices.count

        // Build batched input: [B, 1]
        let batchTokens = stacked(tokenArrays).reshaped(B, 1)

        // Per-layer batched cache wrappers. For B > 1 we need the
        // Batch wrappers to split/pad/stack per-slot caches across the
        // batch dim. For B == 1 the wrappers are pure overhead:
        // BatchKVCache allocates an offsetArray and adds a Swift
        // dispatch per update() call on every layer on every token.
        // On a hybrid-SSM 35B-A3B MoE decode with 48 plus layers that
        // is meaningful. Direct-pass at B == 1 recovers the overhead.
        let numLayers = activeSlots[slotIndices[0]].cache.count
        var layerCaches = [KVCache]()
        var batchArraysCaches = [BatchArraysCache]()  // track for splitBack
        var batchCacheLists = [BatchCacheList]()       // track for splitBack
        layerCaches.reserveCapacity(numLayers)

        if B == 1 {
            // Direct pass-through — no per-token wrapper allocation.
            layerCaches.append(contentsOf: activeSlots[slotIndices[0]].cache)
        } else {
            for layer in 0 ..< numLayers {
                let slotCachesForLayer = slotIndices.map { activeSlots[$0].cache[layer] }
                let representative = slotCachesForLayer[0]

                if let _ = representative as? CacheList {
                    let cacheLists = slotCachesForLayer.map { $0 as! CacheList }
                    let batchCL = BatchCacheList(slotCacheLists: cacheLists)
                    layerCaches.append(batchCL)
                    batchCacheLists.append(batchCL)
                } else if let _ = representative as? ArraysCache {
                    let arraysCaches = slotCachesForLayer.map { $0 as! ArraysCache }
                    let batchAC = BatchArraysCache(slotCaches: arraysCaches)
                    layerCaches.append(batchAC)
                    batchArraysCaches.append(batchAC)
                } else if let _ = representative as? ZayaCCACache {
                    // ZAYA CCA-attention layers — gather/scatter conv_state +
                    // prev_hs alongside the standard KV split/pad/stack.
                    let zayaCaches = slotCachesForLayer.map { $0 as! ZayaCCACache }
                    layerCaches.append(BatchZayaCCACache(slotCaches: zayaCaches))
                } else {
                    layerCaches.append(BatchKVCache(slotCaches: slotCachesForLayer))
                }
            }
        }

        // Run batched forward pass
        let result = context.model(
            LMInput.Text(tokens: batchTokens),
            cache: layerCaches,
            state: nil
        )
        // result.logits shape: [B, 1, vocabSize]

        // Async-eval the logits so GPU work kicks off while we do the
        // Swift-side bookkeeping below. We MUST still materialize
        // `tokenID` via `.item(Int.self)` for the EOS check (forces a
        // sync point), but by that time the forward has already been
        // in flight — saving the serialized `eval` → wait → sample
        // path that cost ~15% decode tok/s on hybrid-SSM 35B-A3B. This
        // mirrors `TokenIterator.next()`'s `asyncEval(token)` pattern.
        asyncEval(result.logits)

        // Split SSM states back to per-sequence caches
        for batchAC in batchArraysCaches {
            batchAC.splitBack()
        }
        for batchCL in batchCacheLists {
            batchCL.splitBack()
        }

        // Sample per sequence (lazy MLXArrays), then asyncEval the
        // whole batch of sampled tokens so the GPU sampling work
        // runs concurrently with the Swift-side bookkeeping below.
        // Mirrors `TokenIterator.next()`'s `asyncEval(token)` idiom
        // which is what gave the non-batch path its +15% edge on
        // 35B-A3B models.
        var sampledTokens: [MLXArray] = []
        sampledTokens.reserveCapacity(slotIndices.count)
        for (batchIdx, slotIdx) in slotIndices.enumerated() {
            let logits = result.logits[batchIdx ..< batchIdx + 1, 0, 0...]
            var slot = activeSlots[slotIdx]
            let token = slot.sampleToken(from: logits)
            sampledTokens.append(token)
            activeSlots[slotIdx] = slot
        }
        asyncEval(sampledTokens)

        // Sample per sequence and route results
        for (batchIdx, slotIdx) in slotIndices.enumerated() {
            var slot = activeSlots[slotIdx]
            let token = sampledTokens[batchIdx]
            // `.item(Int.self)` forces eval of the sampled-token op.
            // GPU is already running (kicked off by asyncEval above
            // of both the logits and the sampled tokens) — this wait
            // is much shorter than a synchronous eval + sample chain.
            let tokenID = token.item(Int.self)

            // Stage 0: per-step KV-quant compression hook. For slots with
            // short prompts that were below the TQ minimum threshold at
            // prefill end, this catches the threshold crossing during decode.
            // Slots already in TurboQuant phase short-circuit via the internal
            // `cache.contains(where: { $0 is TurboQuantKVCache })` guard, so
            // this is a cheap no-op once compressed.
            maybeCompressSlotCache(&slot)

            // Check stop conditions BEFORE yielding — don't emit EOS tokens to callers.
            // This matches TokenIterator behavior where the stop token is never surfaced.
            if stopTokenIDs.contains(tokenID) {
                finishSlot(slot, reason: .stop)
                slot.isFinished = true
            } else {
                slot.continuation.yield(.token(tokenID))
                slot.generatedTokenCount += 1
                slot.generatedTokenIds.append(tokenID)
                slot.nextToken = token

                if let maxTokens = slot.maxTokens, slot.generatedTokenCount >= maxTokens {
                    finishSlot(slot, reason: .length)
                    slot.isFinished = true
                }
            }

            activeSlots[slotIdx] = slot
        }
    }

    /// Preserve admission-level concurrency while only batching decode slots
    /// whose cache topology and requested live KV codec are compatible.
    ///
    /// Homogeneous plain/plain and TurboQuant/TurboQuant groups still batch
    /// normally. Incompatible groups step independently in the same scheduler
    /// iteration; this is correctness routing, not a sampling or model-behavior
    /// guard.
    private func decodeCompatibilityGroups(slotIndices: [Int]) -> [[Int]] {
        var orderedKeys: [String] = []
        var groups: [String: [Int]] = [:]
        for index in slotIndices {
            let key = decodeCompatibilityKey(for: activeSlots[index])
            if groups[key] == nil {
                orderedKeys.append(key)
                groups[key] = []
            }
            groups[key]?.append(index)
        }
        return orderedKeys.compactMap { groups[$0] }
    }

    private func decodeCompatibilityKey(for slot: BatchSlot) -> String {
        let kvModeKey: String
        switch slot.parameters.kvMode {
        case .none:
            kvModeKey = "kv:none"
        case .affine(let bits, let groupSize):
            kvModeKey = "kv:affine:\(bits):\(groupSize)"
        case .turboQuant(let keyBits, let valueBits):
            kvModeKey = "kv:tq:\(keyBits):\(valueBits)"
        }

        let cacheKey = slot.cache.map { layer -> String in
            if let tq = layer as? TurboQuantKVCache {
                return "TurboQuantKVCache:\(tq.keyBits):\(tq.valueBits):\(tq.phase)"
            }
            if let zaya = layer as? ZayaCCACache,
               let tq = zaya.turboQuantKVCache
            {
                return "ZayaCCACache:TQ:\(tq.keyBits):\(tq.valueBits):\(tq.phase)"
            }
            return String(reflecting: type(of: layer))
        }.joined(separator: "|")

        return kvModeKey + ";" + cacheKey
    }

    private func maybeCompressSlotCache(_ slot: inout BatchSlot) {
        let hadTQ = ModelCacheTopologySnapshot(cache: slot.cache).turboQuantKVLayerCount > 0
        let before = hadTQ ? nil : ModelCacheTopologySnapshot(cache: slot.cache)
        BatchQuantize.maybeCompress(
            cache: &slot.cache,
            parameters: slot.parameters
        )
        let hasTQ = ModelCacheTopologySnapshot(cache: slot.cache).turboQuantKVLayerCount > 0
        if !hadTQ, hasTQ, let before {
            turboQuantCompressionCount += 1
            lastTurboQuantCacheTransition = TurboQuantCacheTransitionSnapshot(
                before: before,
                after: ModelCacheTopologySnapshot(cache: slot.cache)
            )
        }
    }

    // MARK: - Completion

    /// Finish a slot by yielding completion info and closing its stream.
    ///
    /// When a cache coordinator is present and the slot completed normally
    /// (not cancelled), stores prompt and safe post-answer boundaries for
    /// future cache reuse.
    private func finishSlot(_ slot: BatchSlot, reason: GenerateStopReason) {
        let now = Date()
        let prefillTime = (slot.decodeStartTime ?? now).timeIntervalSince(slot.prefillStartTime)
        let decodeTime = slot.decodeStartTime.map { now.timeIntervalSince($0) } ?? 0
        let completionInfo = GenerateCompletionInfo(
            promptTokenCount: slot.promptTokenCount,
            generationTokenCount: slot.generatedTokenCount,
            promptTime: prefillTime,
            generationTime: decodeTime,
            stopReason: reason
        )

        // Surface completion before the cache store. The store may include a
        // synchronous hybrid-SSM prompt-boundary re-derive; running it before
        // `.info` makes hosts look frozen at end-of-stream. Keep the work
        // serialized here rather than detached because prior async re-derive
        // paths raced Metal command encoders on shared model state.
        slot.continuation.yield(.info(completionInfo))

        // Store cache state for completed (non-cancelled) generations.
        //
        // SLIDING-1 (2026-04-15): the legacy `!hasRotatingCache` guard
        // was removed once the v2 `TQDiskSerializer` learned to round-trip
        // ring buffer + 5-tuple metaState via `.rotating` LayerKind. The
        // `mediaSalt` is passed through so the stored key matches the key
        // the next fetch will look for (VL multi-turn cache hits).
        if reason != .cancelled, let coordinator = cacheCoordinator {
            let promptTokens = slot.cachePromptTokenIds
            let hasHybridPool = slot.cache.contains { $0 is HybridPoolCache }
            guard let promptCacheSnapshot = slot.promptCacheSnapshot
                ?? (hasHybridPool ? nil : makePromptBoundaryCacheSnapshot(from: slot.cache))
            else {
                Self.logger.error(
                    "Slot \(slot.id.description, privacy: .public): skipped cache store because no prompt-boundary snapshot exists for hybrid-pool cache"
                )
                slot.continuation.finish()
                return
            }

            func cacheCovers(_ tokenCount: Int, cache: [KVCache]) -> Bool {
                cache.map(\.offset).max() ?? 0 >= tokenCount
            }

            let sharedPromptStripBoundary: Int? = {
                guard ProcessInfo.processInfo.environment["VMLX_HYBRID_STRIPPED_STORE"] != "0",
                    coordinator.isHybrid,
                    let turnStartToken = coordinator.genPromptSuffixTokens.first,
                    let stripAt = promptTokens.lastIndex(of: turnStartToken),
                    stripAt > 0,
                    stripAt < promptTokens.count - 1,
                    slot.originalInput.canCaptureHybridStripBoundary(
                        promptTokenIds: promptTokens,
                        boundary: stripAt)
                else { return nil }
                return stripAt
            }()
            var sharedPromptRederivedStates: [Int: [MLXArray]]?
            let sharedPromptAdditionalBoundaries = Array(Set(
                slot.originalInput.cachePrefixTokenCounts
                    + [sharedPromptStripBoundary].compactMap { $0 }
            ))
            // A path-dependent hybrid chat prompt has one canonical reusable
            // boundary: the prompt with its trailing generation scaffold
            // removed.  The exact prompt boundary is deliberately rejected on
            // restore for these topologies, while the generated/post-answer
            // tokens are not guaranteed to match the template's historical
            // assistant rendering.  Persisting all three used to serialize
            // several almost-identical full snapshots at every turn (hundreds
            // of MB each for Bonsai) even though only this stripped boundary is
            // required by the next turn.  Dense/rotating, media-unsafe, raw,
            // and non-chat prompts have no proven strip boundary and keep the
            // existing prompt/post-answer policy.
            let usesCanonicalHybridBoundary =
                coordinator.isHybrid && sharedPromptStripBoundary != nil

            func storeCacheEntry(tokens: [Int], snapshot: [KVCache], label: String) {
                guard !tokens.isEmpty else { return }
                // Serialising the cache materialises it again (host `Data` for the
                // disk write, plus the disk-store cache) while the snapshot and the
                // live cache are both still resident. A prefix-cache entry only ever
                // speeds up some later request — it must never be able to take the
                // host down, so if the copies won't fit, don't make them.
                guard CacheStoreBudget.canStore(snapshot) else {
                    let gib = Double(CacheStoreBudget.cacheBytes(snapshot)) / 1_073_741_824
                    Self.logger.info(
                        """
                        prefix-cache: skipping \(label, privacy: .public) store of a \
                        \(String(format: "%.1f", gib), privacy: .public) GiB KV cache — the copies it \
                        requires do not fit in memory. Generation was unaffected; only the cache \
                        entry was dropped.
                        """
                    )
                    return
                }
                let requiresDiskBackedRestore =
                    cacheRequiresDiskBackedCoordinatorRestore(snapshot)
                let perLayerData = requiresDiskBackedRestore
                    ? []
                    : extractLayerData(from: snapshot)
                let ssmStates: [MLXArray]? = {
                    guard coordinator.isHybrid else { return nil }
                    if let exact = exactBoundarySSMStatesFromSnapshotIfSufficient(
                        coordinator: coordinator,
                        snapshot: snapshot,
                        tokenCount: tokens.count)
                    {
                        return exact
                    }
                    if coordinator.config.enableSSMReDerive &&
                        !slot.originalInput.hasMediaContent
                    {
                        let isPromptPrefix = tokens.count <= promptTokens.count
                            && tokens.elementsEqual(promptTokens.prefix(tokens.count))
                        if isPromptPrefix {
                            if sharedPromptRederivedStates == nil {
                                sharedPromptRederivedStates =
                                    reDeriveAndStoreSSMStatesAtPromptBoundaries(
                                        coordinator: coordinator,
                                        model: context.model,
                                        promptTokenIds: promptTokens,
                                        mediaSalt: slot.mediaSalt,
                                        additionalBoundaries: sharedPromptAdditionalBoundaries,
                                        persistCapturedStatesToDisk: false,
                                        prefillStepSize: slot.parameters.prefillStepSize)
                            }
                            if let shared = sharedPromptRederivedStates?[tokens.count] {
                                return shared
                            }
                        }
                        return reDeriveAndStoreSSMStatesForPromptBoundaries(
                            coordinator: coordinator,
                            model: context.model,
                            promptTokenIds: tokens,
                            mediaSalt: slot.mediaSalt,
                            persistCapturedStatesToDisk: false,
                            prefillStepSize: slot.parameters.prefillStepSize)
                    }
                    return extractSSMStates(from: snapshot)
                }()
                let diskKVMode = snapshot.contains(where: { $0 is ZayaCCACache })
                    ? selectivePromptBoundaryDiskKVMode(
                        cache: snapshot,
                        requested: slot.parameters.kvMode)
                    : slot.parameters.kvMode
                let diskStoreCache = makeDiskStoreCache(
                    fromPromptBoundary: snapshot,
                    kvBits: slot.parameters.kvBits,
                    kvGroupSize: slot.parameters.kvGroupSize,
                    quantizedKVStart: slot.parameters.quantizedKVStart,
                    kvMode: diskKVMode)
                coordinator.storeAfterGeneration(
                    promptTokens: tokens,
                    perLayerData: perLayerData,
                    ssmStates: ssmStates,
                    cache: diskStoreCache,
                    mediaSalt: slot.mediaSalt
                )
                if ProcessInfo.processInfo.environment["VMLX_CACHE_FETCH_TRACE"] == "1" {
                    FileHandle.standardError.write(Data(
                        "[vmlx][cache/store] label=\(label) count=\(tokens.count)\n".utf8))
                }
                Self.logger.debug(
                    "Stored \(label, privacy: .public) cache entry for slot \(slot.id.description, privacy: .public): \(tokens.count) tokens"
                )
            }

            func boundarySnapshot(tokens: [Int], forceRederive: Bool = false) -> [KVCache]? {
                guard !tokens.isEmpty, tokens.count < promptTokens.count else {
                    return nil
                }
                let trimCount = promptTokens.count - tokens.count
                let trimmed = promptCacheSnapshot.map { $0.copy() }
                if canTrimPromptCache(trimmed),
                   trimPromptCache(trimmed, numTokens: trimCount) == trimCount
                {
                    MLX.eval(trimmed)
                    return trimmed
                }

                // `forceRederive` bypasses the disk-backed skip-guard for the
                // cross-turn gen-suffix-stripped boundary — the ONE boundary the
                // next chat turn actually reuses. The guard was added to dodge a
                // Metal command-encoder race between the just-finished decode and
                // this re-derive; that race is now closed by the evalLock around
                // encode+commit (vmlx e0e2eb6e), and `finishSlot` runs
                // synchronously in the scheduling loop, so re-entering
                // `model.prepare` here no longer races live decode. Path-dependent
                // hybrid SSM caches aren't trimmable, so without this the stripped
                // boundary would never be stored and growing hybrid turns could
                // never reuse prefill.
                if !forceRederive,
                   shouldSkipHistoryBoundaryRederiveAfterTrimMiss(promptCacheSnapshot) {
                    Self.logger.debug(
                        "Skipped history-boundary cache rederive after trim miss for slot \(slot.id.description, privacy: .public): disk-backed cache topology"
                    )
                    return nil
                }

                if String(describing: Swift.type(of: context.model)).contains("Gemma3n") {
                    Self.logger.debug(
                        "Skipped Gemma3n history-boundary cache rederive for slot \(slot.id.description, privacy: .public) after trim miss"
                    )
                    return nil
                }

                do {
                    let boundaryTokens = MLXArray(tokens.map { Int32($0) })
                        .reshaped(1, tokens.count)
                    let boundaryInput = LMInput(
                        text: LMInput.Text(tokens: boundaryTokens),
                        image: slot.originalInput.image,
                        video: slot.originalInput.video,
                        audio: slot.originalInput.audio,
                        mediaTokenIds: slot.originalInput.mediaTokenIds,
                        cacheScopeSalt: slot.originalInput.cacheScopeSalt)
                    let cache = context.model.newCache(parameters: slot.parameters)
                    switch try context.model.prepare(
                        boundaryInput,
                        cache: cache,
                        windowSize: effectivePrefillWindow(
                            requested: slot.prefillStepSize,
                            input: boundaryInput,
                            cache: cache))
                    {
                    case .tokens(let remaining):
                        // Match the main prefill path's batch-first shape.
                        // ZAYA CCA reads B/T from activation rank and traps
                        // on a 1D token tensor during coordinator-only
                        // history-boundary cache rederive.
                        _ = context.model(
                            remaining[text: .newAxis],
                            cache: cache,
                            state: nil)
                    case .logits:
                        break
                    }
                    MLX.eval(cache)
                    return cache
                } catch {
                    if ProcessInfo.processInfo.environment["VMLX_SSM_STORE_TRACE"] != nil {
                        FileHandle.standardError.write(Data(
                            "[vmlx][ssm-strip-store] boundarySnapshot rederive THREW: \(String(describing: error))\n".utf8))
                    }
                    Self.logger.debug(
                        "Skipped history-boundary cache rederive for slot \(slot.id.description, privacy: .public): \(String(describing: error), privacy: .public)"
                    )
                    return nil
                }
            }

            if !usesCanonicalHybridBoundary {
                storeCacheEntry(
                    tokens: promptTokens,
                    snapshot: promptCacheSnapshot,
                    label: "prompt-boundary")
            }

            if !slot.cachePromptUsesPostPrepareKey {
                let requiresDiskBackedRestore =
                    cacheRequiresDiskBackedCoordinatorRestore(promptCacheSnapshot)
                if !usesCanonicalHybridBoundary,
                   requiresDiskBackedRestore,
                   !shouldSkipDiskBackedToolPromptSeedBoundary(for: slot),
                   promptTokens.count > 1,
                   let snapshot = boundarySnapshot(
                    tokens: Array(promptTokens.dropLast()),
                    // Direct full-KV + rotating-SWA stacks are fully typed on
                    // disk but stop being trimmable once the ring wraps. Their
                    // exact N-1 seed is still safe to rebuild synchronously;
                    // persisting it lets a fresh process re-feed only the final
                    // prompt token instead of cold-prefilling the whole chat.
                    forceRederive: cacheCanUsePagedWithRotatingCompanion(
                        promptCacheSnapshot))
                {
                    storeCacheEntry(
                        tokens: Array(promptTokens.dropLast()),
                        snapshot: snapshot,
                        label: "disk-backed-safe-prompt-boundary")
                } else if requiresDiskBackedRestore,
                          shouldSkipDiskBackedToolPromptSeedBoundary(for: slot)
                {
                    Self.logger.debug(
                        "Skipped disk-backed tool prompt seed boundary for \(self.context.configuration.name, privacy: .public): required-tool restore is not proven safe for this topology"
                    )
                }
                for boundary in Set(slot.originalInput.cachePrefixTokenCounts).sorted()
                where boundary > 0 && boundary < promptTokens.count {
                    let isStableBoundary = slot.originalInput
                        .cacheStablePrefixTokenCounts.contains(boundary)
                    if usesCanonicalHybridBoundary, !isStableBoundary {
                        continue
                    }
                    // Exact disk restores are deliberately rejected for
                    // path-dependent hybrid caches: GDN/Mamba/CCA needs an
                    // N-1 recurrent seed before the final token is re-fed.
                    // Persist the processor-proven stable prefix one token
                    // short so a brand-new chat can warm from SSD immediately
                    // instead of doing one full cold prefill just to create
                    // that seed for the *next* chat.
                    let storeBoundary = isStableBoundary
                        && requiresDiskBackedRestore && boundary > 1
                        ? boundary - 1
                        : boundary
                    let boundaryTokens = Array(promptTokens.prefix(storeBoundary))
                    if isStableBoundary,
                       coordinator.hasValidatedDiskEntry(
                        tokens: boundaryTokens,
                        mediaSalt: slot.mediaSalt)
                    {
                        Self.logger.debug(
                            "Skipped already-validated stable system/tool cache boundary for slot \(slot.id.description, privacy: .public): \(boundary, privacy: .public) tokens"
                        )
                        continue
                    }
                    if let snapshot = boundarySnapshot(
                        tokens: boundaryTokens,
                        forceRederive: isStableBoundary)
                    {
                        storeCacheEntry(
                            tokens: boundaryTokens,
                            snapshot: snapshot,
                            label: isStableBoundary
                                ? (storeBoundary == boundary
                                    ? "stable-system-tool-boundary"
                                    : "stable-system-tool-safe-seed")
                                : "history-boundary")
                    }
                }

                // Gen-suffix-stripped cross-turn boundary (hybrid SSM).
                //
                // The prompt boundary stored above ends in the chat template's
                // generation-prompt suffix (`<|im_start|>assistant\n`, …). The
                // NEXT chat turn replaces that suffix with the assistant reply +
                // the following user turn, so the full-prompt key can never match
                // as a prefix — which is why growing hybrid turns never reused
                // prefill and recomputed the whole context every turn. The
                // boundary the next turn DOES contain as an exact prefix is this
                // prompt stripped back to the end of the last real (user) message,
                // i.e. everything before the final turn-start token. Store it so
                // hybrid multi-turn chat reuses prior prefill.
                //
                // Correctness: KV comes from the prompt-boundary trim/re-derive;
                // clean SSM/GatedDeltaNet state at the stripped position comes from
                // `storeCacheEntry`'s re-derive (enableSSMReDerive), NOT from the
                // live post-generation state (which is ahead by the gen suffix).
                // The store only fires when the prompt's tail actually is the
                // template's gen-prompt suffix, so non-chat / tool-scaffold prompts
                // that don't match simply skip it (no reuse, still correct). Proven
                // cache-ON == cache-OFF (byte-identical, temp=0, fresh disk cache)
                // on qwen-agentworld-35b-a3b MXFP8 (GatedDeltaNet MoE) and
                // nemotron-omni-nano (Mamba-2); inert on dense gemma-4-e2b.
                // NOTE: intentionally NOT gated on
                // `!cachePrefixTokenCounts.contains(stripAt)`. For hybrid caches
                // the history-boundary path can't store this boundary without a
                // forced re-derive, and `stripAt` routinely coincides with a
                // `cachePrefixTokenCounts` entry — gating on it silently disables
                // the store entirely (the re-derive here is the only writer).
                if ProcessInfo.processInfo.environment["VMLX_HYBRID_STRIPPED_STORE"] != "0",
                   coordinator.isHybrid,
                   let stripAt = sharedPromptStripBoundary
                {
                    let strippedTokens = Array(promptTokens.prefix(stripAt))
                    if coordinator.hasValidatedDiskEntry(
                        tokens: strippedTokens,
                        mediaSalt: slot.mediaSalt)
                    {
                        Self.logger.debug(
                            "Skipped already-validated gen-suffix-stripped cache boundary for slot \(slot.id.description, privacy: .public): \(stripAt, privacy: .public) tokens"
                        )
                    } else if let snapshot = boundarySnapshot(
                        tokens: strippedTokens, forceRederive: true)
                    {
                        storeCacheEntry(
                            tokens: strippedTokens,
                            snapshot: snapshot,
                            label: "gen-suffix-stripped")
                    }
                }
            } else if !slot.originalInput.cachePrefixTokenCounts.isEmpty {
                Self.logger.debug(
                    "Skipped history-boundary cache entries for slot \(slot.id.description, privacy: .public): input prefix counts are pre-pruned but cache key is post-prepare"
                )
            }

            // A normal EOS stop means the last visible assistant token has
            // already been fed back into the cache before EOS was sampled.
            // Length/cancel stops can end immediately after sampling a token,
            // so the live cache may be one token behind the visible text.
            // Store the growing-chat boundary only when the cache offset proves
            // it covers prompt + generated tokens exactly enough to resume.
            let generatedBoundaryTokens = promptTokens + slot.generatedTokenIds
            if !usesCanonicalHybridBoundary,
               reason == .stop,
               !slot.disablesGeneratedCacheBoundary,
               !containsUnprovenZayaTurboQuantDiskState(slot.cache),
               !slot.generatedTokenIds.isEmpty,
               cacheCovers(generatedBoundaryTokens.count, cache: slot.cache)
            {
                storeCacheEntry(
                    tokens: generatedBoundaryTokens,
                    snapshot: slot.cache,
                    label: "post-answer")
            } else if !slot.generatedTokenIds.isEmpty {
                Self.logger.debug(
                    "Skipped post-answer cache entry for slot \(slot.id.description, privacy: .public): reason=\(String(describing: reason), privacy: .public) generated=\(slot.generatedTokenIds.count) cacheOffset=\((slot.cache.map(\.offset).max() ?? 0), privacy: .public)"
                )
            }
        }

        // Drain the GPU before signaling end-of-stream. Everything above —
        // the decode tail and the end-of-turn cache store (`MLX.eval` on
        // trimmed/boundary snapshots, hybrid-SSM re-derive forward passes) —
        // only SUBMITS work; MLX completes it asynchronously on its stream
        // thread. The host releases the process-wide GPU gate (osaurus'
        // MetalGate) when this stream finishes, so if the next exclusive
        // producer (image generation via a second MLX graph, the embedder, a
        // model load) starts while this cache-store eval is still in flight,
        // the two race on the shared Metal command buffer and crash
        // (EXC_BAD_ACCESS in `tryCoalescingPreviousComputeCommandEncoder`, or
        // `addCompletedHandler: provided after commit call`). Synchronizing on
        // THIS thread — which owns the command buffers — drains them in order
        // with no foreign-commit hazard, so "stream finished" provably means
        // "GPU idle." Mirrors the solo-fast-path drain in `finishSoloFastPath`.
        Stream().synchronize()

        slot.continuation.finish()

        // Long-context pressure relief: the global memoryPurgeInterval (256
        // decode steps) is too coarse for long requests where a single slot
        // can allocate several GB of activations before releasing the pool
        // back to the allocator. Without this, long-context traffic
        // degraded subsequent requests by holding onto the pool — manifesting
        // as decode-speed cratering on the next request submitted.
        //
        // Trigger a targeted purge when the just-finished slot had a
        // non-trivially-long prompt. 4096 tokens is the threshold: short
        // chat requests skip the extra C call (~100us) while long-context
        // or document-QA requests reclaim the pool at request boundaries.
        let longContextPurgeThreshold = 4096
        if slot.promptTokenCount >= longContextPurgeThreshold {
            Memory.clearCache()
            // Reset the global counter too so we don't double-purge on the
            // next scheduling tick.
            stepsSinceMemoryPurge = 0
        }
    }
}

// BatchEngine uses the shared `_decodePromptTail` helper from Evaluate.swift
// (same module, internal visibility) for `ReasoningParser.forPrompt`
// auto-detection of prompt-end state.
