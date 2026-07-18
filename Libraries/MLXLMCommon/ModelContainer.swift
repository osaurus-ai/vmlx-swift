// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN
import os

/// Source-backed cache topology reported by a loaded model.
///
/// This is intentionally derived from `LanguageModel.newCache(parameters:)`
/// rather than model ids or config strings, so host apps can make cache reuse
/// decisions from the same cache classes the runtime will actually allocate.
public struct ModelCacheTopologySnapshot: Codable, Sendable, Equatable {
    public var layerCount: Int
    public var kvLayerCount: Int
    public var chunkedKVLayerCount: Int
    public var quantizedKVLayerCount: Int
    public var turboQuantKVLayerCount: Int
    public var compilableKVLayerCount: Int
    public var compilableTurboQuantKVLayerCount: Int
    public var rotatingKVLayerCount: Int
    public var compilableRotatingKVLayerCount: Int
    public var rotatingWrapperLayerCount: Int
    public var hybridPoolLayerCount: Int
    public var mambaLayerCount: Int
    public var compilableMambaLayerCount: Int
    public var arraysLayerCount: Int
    public var zayaCCALayerCount: Int
    public var cacheListLayerCount: Int
    /// MiniMax-M3 MSA layers carry a `MiniMaxM3SparseCache` (KV + an idx_keys
    /// lane). Counted separately so reuse tiers preserve the composite (never a
    /// plain KV copy that drops idx_keys) and the coordinator marks it
    /// paged-incompatible — dynamic block selection has no page format. This is
    /// NOT an SSM companion (no conv/recurrent state); see
    /// `requiresMiniMaxM3SparseState`.
    public var miniMaxM3SparseLayerCount: Int

    public init(
        layerCount: Int = 0,
        kvLayerCount: Int = 0,
        chunkedKVLayerCount: Int = 0,
        quantizedKVLayerCount: Int = 0,
        turboQuantKVLayerCount: Int = 0,
        compilableKVLayerCount: Int = 0,
        compilableTurboQuantKVLayerCount: Int = 0,
        rotatingKVLayerCount: Int = 0,
        compilableRotatingKVLayerCount: Int = 0,
        rotatingWrapperLayerCount: Int = 0,
        hybridPoolLayerCount: Int = 0,
        mambaLayerCount: Int = 0,
        compilableMambaLayerCount: Int = 0,
        arraysLayerCount: Int = 0,
        zayaCCALayerCount: Int = 0,
        cacheListLayerCount: Int = 0,
        miniMaxM3SparseLayerCount: Int = 0
    ) {
        self.layerCount = layerCount
        self.kvLayerCount = kvLayerCount
        self.chunkedKVLayerCount = chunkedKVLayerCount
        self.quantizedKVLayerCount = quantizedKVLayerCount
        self.turboQuantKVLayerCount = turboQuantKVLayerCount
        self.compilableKVLayerCount = compilableKVLayerCount
        self.compilableTurboQuantKVLayerCount = compilableTurboQuantKVLayerCount
        self.rotatingKVLayerCount = rotatingKVLayerCount
        self.compilableRotatingKVLayerCount = compilableRotatingKVLayerCount
        self.rotatingWrapperLayerCount = rotatingWrapperLayerCount
        self.hybridPoolLayerCount = hybridPoolLayerCount
        self.mambaLayerCount = mambaLayerCount
        self.compilableMambaLayerCount = compilableMambaLayerCount
        self.arraysLayerCount = arraysLayerCount
        self.zayaCCALayerCount = zayaCCALayerCount
        self.cacheListLayerCount = cacheListLayerCount
        self.miniMaxM3SparseLayerCount = miniMaxM3SparseLayerCount
    }

    public init(cache: [any KVCache]) {
        self.init()
        layerCount = cache.count
        for layer in cache {
            record(layer)
        }
    }

    public var requiresSSMCompanionState: Bool {
        mambaLayerCount > 0 || arraysLayerCount > 0 || zayaCCALayerCount > 0
    }

    public var requiresZayaCCACompanionState: Bool {
        zayaCCALayerCount > 0
    }

    /// True when any layer is a MiniMax-M3 sparse (MSA) cache. The reuse tiers
    /// must keep that cache first-class (preserve the idx_keys lane) and the
    /// coordinator must run paged-incompatible for it.
    public var requiresMiniMaxM3SparseState: Bool {
        miniMaxM3SparseLayerCount > 0
    }

    public var requiresRecurrentSSMCompanionState: Bool {
        mambaLayerCount > 0 || arraysLayerCount > 0
    }

    public var requiresDiskBackedCoordinatorRestore: Bool {
        requiresSSMCompanionState
            || requiresMiniMaxM3SparseState
            || rotatingKVLayerCount > 0
            || compilableRotatingKVLayerCount > 0
            || rotatingWrapperLayerCount > 0
            || hybridPoolLayerCount > 0
            || quantizedKVLayerCount > 0
            || turboQuantKVLayerCount > 0
            || compilableTurboQuantKVLayerCount > 0
    }

    public var topologyTags: [String] {
        var tags: [String] = ["layers=\(layerCount)"]
        if kvLayerCount > 0 { tags.append("kvLayers=\(kvLayerCount)") }
        if chunkedKVLayerCount > 0 { tags.append("chunkedKVLayers=\(chunkedKVLayerCount)") }
        if quantizedKVLayerCount > 0 { tags.append("quantizedKVLayers=\(quantizedKVLayerCount)") }
        if turboQuantKVLayerCount > 0 { tags.append("turboQuantKVLayers=\(turboQuantKVLayerCount)") }
        if compilableKVLayerCount > 0 { tags.append("compilableKVLayers=\(compilableKVLayerCount)") }
        if compilableTurboQuantKVLayerCount > 0 {
            tags.append("compilableTurboQuantKVLayers=\(compilableTurboQuantKVLayerCount)")
        }
        if rotatingKVLayerCount > 0 { tags.append("rotatingLayers=\(rotatingKVLayerCount)") }
        if compilableRotatingKVLayerCount > 0 {
            tags.append("compilableRotatingLayers=\(compilableRotatingKVLayerCount)")
        }
        if rotatingWrapperLayerCount > 0 { tags.append("rotatingWrapperLayers=\(rotatingWrapperLayerCount)") }
        if hybridPoolLayerCount > 0 { tags.append("hybridPoolLayers=\(hybridPoolLayerCount)") }
        if mambaLayerCount > 0 { tags.append("mambaLayers=\(mambaLayerCount)") }
        if compilableMambaLayerCount > 0 { tags.append("compilableMambaLayers=\(compilableMambaLayerCount)") }
        if arraysLayerCount > 0 { tags.append("arraysLayers=\(arraysLayerCount)") }
        if zayaCCALayerCount > 0 { tags.append("zayaCCALayers=\(zayaCCALayerCount)") }
        if miniMaxM3SparseLayerCount > 0 { tags.append("minimaxM3SparseLayers=\(miniMaxM3SparseLayerCount)") }
        if cacheListLayerCount > 0 { tags.append("cacheListLayers=\(cacheListLayerCount)") }
        if requiresRecurrentSSMCompanionState { tags.append("companion=ssm") }
        if requiresZayaCCACompanionState { tags.append("companion=zaya-cca") }
        if requiresDiskBackedCoordinatorRestore { tags.append("restore=disk-backed") }
        return tags
    }

    private mutating func record(_ cache: any KVCache) {
        switch cache {
        case let list as CacheList:
            cacheListLayerCount += 1
            for index in 0..<list.count {
                record(list[index])
            }
        case is CompilableTurboQuantKVCache:
            compilableTurboQuantKVLayerCount += 1
            turboQuantKVLayerCount += 1
        case is TurboQuantKVCache:
            turboQuantKVLayerCount += 1
        case is QuantizedKVCache:
            quantizedKVLayerCount += 1
        case is CompilableRotatingKVCache:
            compilableRotatingKVLayerCount += 1
            rotatingKVLayerCount += 1
        case is HybridPoolCache:
            hybridPoolLayerCount += 1
            rotatingWrapperLayerCount += 1
        case is RotatingKVCacheWrapper:
            rotatingWrapperLayerCount += 1
        case is ZayaCCACache:
            zayaCCALayerCount += 1
        case is MiniMaxM3SparseCache:
            // KV + idx_keys composite. Distinct bucket so it is NOT miscounted as
            // a plain KV layer (which would let a reuse path drop the idx_keys
            // lane and loop). Drives requiresMiniMaxM3SparseState / disk-backed
            // restore / paged-incompatible.
            miniMaxM3SparseLayerCount += 1
        case is CompilableMambaCache:
            compilableMambaLayerCount += 1
            mambaLayerCount += 1
        case is MambaCache:
            mambaLayerCount += 1
        case is ArraysCache:
            arraysLayerCount += 1
        case is RotatingKVCache:
            rotatingKVLayerCount += 1
        case is CompilableKVCache:
            compilableKVLayerCount += 1
            kvLayerCount += 1
        case is ChunkedKVCache:
            chunkedKVLayerCount += 1
            kvLayerCount += 1
        case is KVCacheSimple:
            kvLayerCount += 1
        default:
            kvLayerCount += 1
        }
    }
}

/// Container for models that guarantees single threaded access.
///
/// Wrap models used by e.g. the UI in a ModelContainer. Callers can access
/// the model and/or tokenizer (any values from the ``ModelContext``):
///
/// ```swift
/// let messages = [["role": "user", "content": prompt]]
/// let promptTokens = try await modelContainer.perform { context in
///     try context.tokenizer.applyChatTemplate(messages: messages)
/// }
/// ```
///
/// or:
///
/// ```swift
/// let userInput: UserInput
/// let result = await modelContainer.perform { context in
///     let input = try await context.processor.prepare(input: userInput)
///     return generate(
///         input: input, parameters: generateParameters, context: context
///     ) { tokens in
///     ...
///     }
/// }
/// ```
public final class ModelContainer: Sendable {
    private let context: SerialAccessContainer<ModelContext>

    // MARK: - Multi-tier KV Cache

    /// Locked storage for the optional cache coordinator.
    private let _cacheCoordinator = OSAllocatedUnfairLock<CacheCoordinator?>(initialState: nil)

    /// Optional cache coordinator for multi-tier KV caching.
    /// Enable via ``enableCaching(config:)`` after model loading.
    public var cacheCoordinator: CacheCoordinator? {
        _cacheCoordinator.withLock { $0 }
    }

    /// Enable multi-tier KV caching with the given configuration.
    /// Call after model loading. Safe to call multiple times (replaces previous coordinator).
    /// Auto-detects hybrid models and sets modelKey from configuration if not provided.
    public func enableCaching(config: CacheCoordinatorConfig = CacheCoordinatorConfig()) {
        var config = config
        // Auto-set modelKey from model configuration if not provided
        if config.modelKey == nil {
            // Will be set asynchronously after first access — for now use a placeholder
            // that prevents cross-model poisoning within the same process.
            config.modelKey = "\(ObjectIdentifier(self))"
        }
        if let modelKey = config.modelKey {
            config.modelKey = RuntimeMoETopKOverride.cacheScopedModelKey(modelKey)
        }
        let coordinator = CacheCoordinator(config: config)
        _cacheCoordinator.withLock { $0 = coordinator }
    }

    /// Enable caching with auto-detection of hybrid models.
    /// Call after model loading. Inspects the model's cache types to detect SSM layers.
    public func enableCachingAsync(config baseConfig: CacheCoordinatorConfig = CacheCoordinatorConfig()) async {
        var config = baseConfig
        let modelConfig = await context.read { $0.configuration }
        if config.modelKey == nil {
            config.modelKey = modelConfig.name
        }
        if let modelKey = config.modelKey {
            config.modelKey = RuntimeMoETopKOverride.cacheScopedModelKey(modelKey)
        }

        let topology = await cacheTopologySnapshot()
        let isHybrid = topology.requiresSSMCompanionState

        // 2026-05-05 (Ling-2.6-flash multi-turn fix): hybrid models with
        // ArraysCache (Linear-Attn / GLA recurrence) live or die by the
        // SSMStateCache fetch in CacheCoordinator.fetch — and that fetch
        // is gated INSIDE the paged-hit branch. With the default
        // `pagedBlockSize=64`, short chat prompts (≤ 64 tokens, common
        // for Bailing/Ling chat templates which render to ~30 tokens)
        // store ZERO paged blocks → coordinator misses → SSM state never
        // restored → the live cache passed across ChatSession turns goes
        // stale → incoherent Turn 2 output (or SIGKILL on full re-prefill
        // since recurrentGLA can't handle L>~30 reliably).
        //
        // Lower the paged block size for hybrid models so even short
        // chat turns store at least one block, enabling the SSM-state
        // restoration path to fire on Turn 2+. 16 tokens covers system-
        // only prefixes and short user messages while keeping hash-chain
        // cost negligible.
        if isHybrid {
            config.pagedBlockSize = 16
        }

        let coordinator = CacheCoordinator(config: config)
        coordinator.setHybrid(isHybrid)
        coordinator.setGenPromptSuffixTokens(await computeGenPromptSuffixTokens())

        _cacheCoordinator.withLock { $0 = coordinator }
    }

    /// Compute the chat template's generation-prompt suffix — the tokens
    /// `add_generation_prompt=true` appends after the last message — by diffing
    /// a dummy chat rendered with vs. without the generation prompt. The store
    /// path uses this to persist an extra cache boundary stripped back to the
    /// user turn, so the NEXT chat turn (which replaces this suffix with the
    /// assistant reply) still matches a stored prefix and reuses it. Returns
    /// `[]` when unavailable/implausible (store then skips the stripped
    /// boundary; safe).
    private func computeGenPromptSuffixTokens() async -> [Int] {
        await context.read { ctx in
            guard let gp = ctx.tokenizer as? GenerationPromptControllableTokenizer
            else { return [] }
            let dummy: [[String: any Sendable]] = [["role": "user", "content": "x"]]
            guard
                let withGen = try? gp.applyChatTemplate(
                    messages: dummy, tools: nil, additionalContext: nil,
                    addGenerationPrompt: true),
                let withoutGen = try? gp.applyChatTemplate(
                    messages: dummy, tools: nil, additionalContext: nil,
                    addGenerationPrompt: false)
            else { return [] }
            var common = 0
            let maxCommon = min(withGen.count, withoutGen.count)
            while common < maxCommon, withGen[common] == withoutGen[common] {
                common += 1
            }
            let suffix = Array(withGen[common...])
            guard (1...64).contains(suffix.count) else { return [] }
            return suffix
        }
    }

    /// Inspect the loaded model's real cache topology without relying on
    /// model names or bundle path heuristics.
    ///
    /// Hosts use this to namespace disk-cache records and to decide whether
    /// prefix reuse needs companion recurrent/CCA state. The snapshot is
    /// derived from `model.newCache(parameters:)`, which is the same factory
    /// BatchEngine uses to create live request caches.
    public func cacheTopologySnapshot(
        parameters: GenerateParameters? = nil
    ) async -> ModelCacheTopologySnapshot {
        await context.read { ctx in
            ModelCacheTopologySnapshot(cache: ctx.model.newCache(parameters: parameters))
        }
    }

    /// Disable caching and release all cached state.
    public func disableCaching() {
        _cacheCoordinator.withLock { coordinator in
            coordinator?.releaseVolatile()
            coordinator = nil
        }
    }

    // MARK: - MLXPress runtime

    /// Locked storage for the optional MLXPress runtime. Settable
    /// from `loadModelContainer(from:using:loadConfiguration:)` so the
    /// runtime stays alive for the model's lifetime; callers (osaurus
    /// settings panel, JANG Studio inspector) poll status from
    /// anywhere via ``mlxPressStatus()``.
    private let _jangPressRuntime = OSAllocatedUnfairLock<JangPressRuntime>(
        initialState: .none)

    /// Read the current MLXPress runtime. `.none` when MLXPress was
    /// not activated for this load (e.g. `LoadConfiguration.jangPress
    /// == .disabled` or auto-threshold not met).
    public var jangPressRuntime: JangPressRuntime {
        _jangPressRuntime.withLock { $0 }
    }

    /// Replace the MLXPress runtime. Called once at load time by
    /// `loadModelContainer(from:using:loadConfiguration:)`. Safe to
    /// call multiple times — each call replaces; ARC drops the prior
    /// runtime and its tiers release.
    public func setJangPressRuntime(_ runtime: JangPressRuntime) {
        _jangPressRuntime.withLock { $0 = runtime }
    }

    /// Snapshot the current MLXPress status. Returns
    /// `MLXPressStatus.disabled` when the runtime is `.none`.
    /// Cheap enough to call on a polling timer (no heavy work; reads
    /// cached counters under a single lock).
    public func jangPressStatus() -> JangPressStatus {
        jangPressRuntime.status()
    }

    /// Public MLXPress spelling for the routed cold-tier runtime.
    public var mlxPressRuntime: MLXPressRuntime {
        jangPressRuntime
    }

    /// Public MLXPress spelling for replacing the routed cold-tier runtime.
    public func setMLXPressRuntime(_ runtime: MLXPressRuntime) {
        setJangPressRuntime(runtime)
    }

    /// Public MLXPress spelling for status polling.
    public func mlxPressStatus() -> MLXPressStatus {
        jangPressStatus()
    }

    public var configuration: ModelConfiguration {
        get async {
            await context.read { $0.configuration }
        }
    }

    /// Build `GenerateParameters` from the bundle's `generation_config.json`
    /// defaults (when present), falling back to the supplied parameters for
    /// every field the config did not specify.
    ///
    /// Opt-in convenience: `generate()` and `streamGenerate()` do NOT
    /// consume this automatically — the runtime keeps the explicit
    /// caller-controlled `GenerateParameters` contract. Callers that want
    /// the bundle's stamped defaults (max_new_tokens, temperature, top_p,
    /// top_k, min_p, repetition_penalty, do_sample) should pass the result
    /// of this method into `generate(...)`.
    ///
    /// Merge policy mirrors `GenerateParameters.init(generationConfig:fallback:)`
    /// at `Libraries/MLXLMCommon/Evaluate.swift:274`:
    /// fields present in `generationDefaults` override the matching field
    /// in `fallback`; fields absent use `fallback`; `do_sample == false`
    /// in the config forces `temperature = 0` even if a temperature is
    /// also specified.
    public func defaultGenerateParameters(
        fallback: GenerateParameters = GenerateParameters()
    ) async -> GenerateParameters {
        let config = await context.read { $0.configuration.generationDefaults }
        return GenerateParameters(generationConfig: config, fallback: fallback)
    }

    public var processor: UserInputProcessor {
        get async {
            await context.read { $0.processor }
        }
    }

    public var tokenizer: Tokenizer {
        get async {
            await context.read { $0.tokenizer }
        }
    }

    /// Whether this model supports vision/image input (is a VLM).
    public var isVLM: Bool {
        get async {
            await context.read { $0.isVLM }
        }
    }

    public init(context: consuming ModelContext) {
        self.context = .init(context)
    }

    /// Perform an action on the model and/or tokenizer. Callers _must_ eval any `MLXArray` before returning as
    /// `MLXArray` is not `Sendable`.
    @available(*, deprecated, message: "prefer perform(_:) that uses a ModelContext")
    public func perform<R: Sendable>(
        _ action: @Sendable (any LanguageModel, Tokenizer) throws -> sending R
    )
        async rethrows
        -> sending R
    {
        try await context.read {
            try action($0.model, $0.tokenizer)
        }
    }

    /// Perform an action on the model and/or tokenizer with additional context values.
    /// Callers _must_ eval any `MLXArray` before returning as
    /// `MLXArray` is not `Sendable`.
    @available(*, deprecated, message: "prefer perform(values:_:) that uses a ModelContext")
    public func perform<V: Sendable, R: Sendable>(
        values: V, _ action: @Sendable (any LanguageModel, Tokenizer, V) throws -> sending R
    ) async rethrows -> sending R {
        try await context.read {
            try action($0.model, $0.tokenizer, values)
        }
    }

    /// Perform an action on the ``ModelContext``. Callers _must_ eval any `MLXArray` before returning as
    /// `MLXArray` is not `Sendable`.
    ///
    /// - Note: The closure receives `ModelContext` which is not `Sendable`. This is intentional -
    ///   the closure runs within the actor's isolation, ensuring thread-safe access to the model.
    /// - Note: The `sending` keyword indicates the return value is transferred (not shared) across
    ///   isolation boundaries, allowing non-Sendable types to be safely returned.
    public func perform<R: Sendable>(
        _ action: @Sendable (ModelContext) async throws -> sending R
    ) async rethrows -> sending R {
        try await context.read {
            try await action($0)
        }
    }

    /// Perform an action on the ``ModelContext`` with additional context values.
    /// Callers _must_ eval any `MLXArray` before returning as
    /// `MLXArray` is not `Sendable`.
    public func perform<V: Sendable, R: Sendable>(
        values: V, _ action: @Sendable (ModelContext, V) async throws -> R
    ) async rethrows -> sending R {
        try await context.read {
            try await action($0, values)
        }
    }

    /// Perform an action on the ``ModelContext`` with additional (non `Sendable`) context values.
    /// Callers _must_ eval any `MLXArray` before returning as
    /// `MLXArray` is not `Sendable`.
    public func perform<V, R: Sendable>(
        nonSendable values: consuming V, _ action: @Sendable (ModelContext, V) async throws -> R
    ) async rethrows -> sending R {
        let values = SendableBox(values)
        return try await context.read {
            try await action($0, values.consume())
        }
    }

    /// Update the owned `ModelContext`.
    /// - Parameter action: update action
    public func update(_ action: @Sendable (inout ModelContext) -> Void) async {
        await context.update {
            action(&$0)
        }
    }

    // MARK: - Thread-safe convenience methods

    /// The resolved local model directory for the loaded container.
    public var modelDirectory: URL {
        get async throws {
            try (await configuration).modelDirectory
        }
    }

    /// The resolved local tokenizer directory for the loaded container.
    public var tokenizerDirectory: URL {
        get async throws {
            try (await configuration).tokenizerDirectory
        }
    }

    /// Prepare user input for generation.
    ///
    /// This method safely prepares input within the actor's isolation,
    /// avoiding the need for closure-based `perform` calls.
    ///
    /// - Parameter input: The user input to prepare
    /// - Returns: Prepared language model input (transferred via `sending`)
    /// - Note: The `sending` keyword indicates the return value is transferred (not shared),
    ///   allowing non-Sendable types like `LMInput` to safely cross isolation boundaries.
    public func prepare(input: consuming sending UserInput) async throws -> sending LMInput {
        let toolSchemas = input.tools
        let processor = await self.processor
        let prepared = try await processor.prepare(input: input)
        return prepared.withToolSchemas(toolSchemas)
    }

    /// Generate tokens from prepared input, returning an AsyncStream.
    ///
    /// This method provides a thread-safe way to generate tokens without
    /// needing to use closure-based `perform` calls.
    ///
    /// Example:
    /// ```swift
    /// let input = try await modelContainer.prepare(input: userInput)
    /// let stream = try modelContainer.generate(input: input, parameters: parameters)
    /// for await generation in stream {
    ///     switch generation {
    ///     case .chunk(let text): print(text)
    ///     case .reasoning: break  // optional: route to think-pane
    ///     case .info(let info): print(info.tokensPerSecond)
    ///     case .toolCall(let call): handleToolCall(call)
    ///     }
    /// }
    /// ```
    ///
    /// - Parameters:
    ///   - input: Prepared language model input (transferred via `sending`)
    ///   - parameters: Generation parameters
    ///   - wiredMemoryTicket: Optional wired memory ticket for policy-based coordination
    /// - Returns: An AsyncStream of generation events
    /// - Note: The `sending` parameter indicates the input is transferred (not shared),
    ///   allowing non-Sendable types like `LMInput` to safely cross isolation boundaries.
    public func generate(
        input: consuming sending LMInput,
        parameters: GenerateParameters,
        wiredMemoryTicket: WiredMemoryTicket? = nil
    ) async throws -> AsyncStream<Generation> {
        jangPressRuntime.recordPromptTokenActivity(
            input.text.tokens.reshaped(-1).asArray(Int.self))

        let input = SendableBox(input)
        let coordinator = self.cacheCoordinator

        // Note: this is only visiting the model exclusively
        // for the pre-fill time.  Beyond that there is no
        // shared mutable state.
        //
        // This means that there may be concurrent access to the
        // model weights themselves (but they are already evaluated).

        return try await context.read { context in
            try MLXLMCommon.generate(
                input: input.consume(),
                parameters: parameters,
                context: context,
                wiredMemoryTicket: wiredMemoryTicket,
                cacheCoordinator: coordinator
            )
        }
    }

    /// Decode token IDs to a string.
    ///
    /// - Parameter tokenIds: Array of token IDs
    /// - Returns: Decoded string
    public func decode(tokenIds: [Int]) async -> String {
        let tokenizer = await self.tokenizer
        return tokenizer.decode(tokenIds: tokenIds)
    }

    @available(*, deprecated, renamed: "decode(tokenIds:)")
    public func decode(tokens: [Int]) async -> String {
        await decode(tokenIds: tokens)
    }

    /// Encode a string to token IDs.
    ///
    /// - Parameter text: Text to encode
    /// - Returns: Array of token IDs
    public func encode(_ text: String) async -> [Int] {
        let tokenizer = await self.tokenizer
        return tokenizer.encode(text: text)
    }

    /// Apply chat template to messages and return token IDs.
    ///
    /// - Parameter messages: Array of message dictionaries with "role" and "content" keys
    /// - Returns: Array of token IDs
    @available(*, deprecated, message: "Use applyChatTemplate directly on tokenizer")
    public func applyChatTemplate(messages: [[String: String]]) async throws -> [Int] {
        let tokenizer = await self.tokenizer
        return try tokenizer.applyChatTemplate(messages: messages)
    }
}
