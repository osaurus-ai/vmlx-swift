// Copyright © 2024 Apple Inc.

import Foundation
import MLX
#if canImport(Cmlx)
import Cmlx
#endif

#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

/// File patterns required to resolve a tokenizer without downloading model weights.
package let tokenizerDownloadPatterns = ["*.json", "*.jinja"]
package let modelDownloadPatterns = ["*.safetensors"] + tokenizerDownloadPatterns

public enum ModelFactoryError: LocalizedError {
    case unsupportedModelType(String)
    case unsupportedProcessorType(String)
    case configurationFileError(String, String, Error)
    case configurationDecodingError(String, String, DecodingError)
    case noModelFactoryAvailable

    public var errorDescription: String? {
        switch self {
        case .unsupportedModelType(let type):
            return "Unsupported model type: \(type)"
        case .unsupportedProcessorType(let type):
            return "Unsupported processor type: \(type)"
        case .configurationFileError(let file, let modelName, let error):
            return "Error reading '\(file)' for model '\(modelName)': \(error.localizedDescription)"
        case .noModelFactoryAvailable:
            return "No model factory available via ModelFactoryRegistry"
        case .configurationDecodingError(let file, let modelName, let decodingError):
            let errorDetail = extractDecodingErrorDetail(decodingError)
            return "Failed to parse \(file) for model '\(modelName)': \(errorDetail)"
        }
    }

    private func extractDecodingErrorDetail(_ error: DecodingError) -> String {
        switch error {
        case .keyNotFound(let key, let context):
            let path = (context.codingPath + [key]).map { $0.stringValue }.joined(separator: ".")
            return "Missing field '\(path)'"
        case .typeMismatch(_, let context):
            let path = context.codingPath.map { $0.stringValue }.joined(separator: ".")
            return "Type mismatch at '\(path)'"
        case .valueNotFound(_, let context):
            let path = context.codingPath.map { $0.stringValue }.joined(separator: ".")
            return "Missing value at '\(path)'"
        case .dataCorrupted(let context):
            if context.codingPath.isEmpty {
                return "Invalid JSON"
            } else {
                let path = context.codingPath.map { $0.stringValue }.joined(separator: ".")
                return "Invalid data at '\(path)'"
            }
        @unknown default:
            return error.localizedDescription
        }
    }
}

/// Context of types that work together to provide a ``LanguageModel``.
///
/// A ``ModelContext`` is created by ``ModelFactory/load(from:configuration:progressHandler:)``.
/// This contains the following:
///
/// - ``ModelConfiguration`` -- identifier for the model
/// - ``LanguageModel`` -- the model itself, see ``generate(input:cache:parameters:context:)``
/// - ``UserInputProcessor`` -- can convert ``UserInput`` into ``LMInput``
/// - `Tokenizer` -- the tokenizer used by ``UserInputProcessor``
///
/// See also ``ModelFactory/loadContainer(from:configuration:progressHandler:)`` and
/// ``ModelContainer``.
public struct ModelContext: @unchecked Sendable {
    public var configuration: ModelConfiguration
    public var model: any LanguageModel
    public var processor: any UserInputProcessor
    public var tokenizer: Tokenizer
    public var jangPressRuntime: JangPressRuntime

    /// Whether this model supports vision/image input (is a VLM).
    public var isVLM: Bool { model is VisionLanguageModelProtocol }

    /// Public MLXPress spelling for the routed cold-tier runtime.
    public var mlxPressRuntime: MLXPressRuntime {
        get { jangPressRuntime }
        set { jangPressRuntime = newValue }
    }

    public init(
        configuration: ModelConfiguration, model: any LanguageModel,
        processor: any UserInputProcessor, tokenizer: any Tokenizer,
        jangPressRuntime: JangPressRuntime = .none
    ) {
        self.configuration = configuration
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer
        self.jangPressRuntime = jangPressRuntime
    }
}

/// Protocol for code that can load models.
///
/// ## See Also
/// - ``loadModel(from:id:progressHandler:)``
/// - ``loadModel(from:)-ModelContext``
/// - ``loadModelContainer(from:id:progressHandler:)``
/// - ``loadModelContainer(from:)-ModelContainer``
public protocol ModelFactory: Sendable {

    var modelRegistry: AbstractModelRegistry { get }

    func _load(
        configuration: ResolvedModelConfiguration,
        tokenizerLoader: any TokenizerLoader
    ) async throws -> ModelContext

}

extension ModelFactory {

    /// Resolve a model identifier, e.g. "mlx-community/Llama-3.2-3B-Instruct-4bit", into
    /// a ``ModelConfiguration``.
    ///
    /// This will either create a new (mostly unconfigured) ``ModelConfiguration`` or
    /// return a registered instance that matches the id.
    ///
    /// - Note: If the id doesn't exists in the configuration, this will return a new instance of it.
    /// If you want to check if the configuration in model registry, you should use ``contains(id:)``.
    public func configuration(id: String) -> ModelConfiguration {
        modelRegistry.configuration(id: id)
    }

    /// Returns true if ``modelRegistry`` contains a model with the id. Otherwise, false.
    public func contains(id: String) -> Bool {
        modelRegistry.contains(id: id)
    }

}

extension ModelFactory {

    /// Load a model from a ``Downloader`` and ``ModelConfiguration``,
    /// producing a ``ModelContext``.
    ///
    /// This resolves the configuration (downloading remote sources via the downloader)
    /// and then loads the model from local files.
    ///
    /// ## See Also
    /// - ``loadModel(from:configuration:useLatest:progressHandler:)``
    /// - ``loadModelContainer(from:configuration:useLatest:progressHandler:)``
    public func load(
        from downloader: any Downloader,
        using tokenizerLoader: any TokenizerLoader,
        configuration: ModelConfiguration,
        useLatest: Bool = false,
        progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
    ) async throws -> sending ModelContext {
        let resolved = try await resolve(
            configuration: configuration, from: downloader,
            useLatest: useLatest, progressHandler: progressHandler)
        return try await _load(configuration: resolved, tokenizerLoader: tokenizerLoader)
    }

    /// Load a model from a ``Downloader`` and ``ModelConfiguration``,
    /// producing a ``ModelContainer``.
    public func loadContainer(
        from downloader: any Downloader,
        using tokenizerLoader: any TokenizerLoader,
        configuration: ModelConfiguration,
        useLatest: Bool = false,
        progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
    ) async throws -> ModelContainer {
        let resolved = try await resolve(
            configuration: configuration, from: downloader,
            useLatest: useLatest, progressHandler: progressHandler)
        let context = try await _load(configuration: resolved, tokenizerLoader: tokenizerLoader)
        return ModelContainer(context: context)
    }

    /// Load a model from a local directory, producing a ``ModelContext``.
    ///
    /// No downloader is needed — the model and tokenizer are loaded from
    /// the given directory.
    public func load(
        from directory: URL,
        using tokenizerLoader: any TokenizerLoader
    ) async throws -> sending ModelContext {
        try await _load(
            configuration: .init(directory: directory), tokenizerLoader: tokenizerLoader)
    }

    /// Load a model from a local directory, producing a ``ModelContainer``.
    public func loadContainer(
        from directory: URL,
        using tokenizerLoader: any TokenizerLoader
    ) async throws -> ModelContainer {
        let context = try await _load(
            configuration: .init(directory: directory), tokenizerLoader: tokenizerLoader)
        return ModelContainer(context: context)
    }

}

/// Resolve a ``ModelConfiguration`` into a ``ResolvedModelConfiguration`` by
/// downloading remote sources via a ``Downloader``.
///
/// This handles the `.id` vs `.directory` switch for the model source and
/// resolves ``TokenizerSource`` for the tokenizer.
public func resolve(
    configuration: ModelConfiguration,
    from downloader: any Downloader,
    useLatest: Bool,
    progressHandler: @Sendable @escaping (Progress) -> Void
) async throws -> ResolvedModelConfiguration {
    let modelDirectory: URL
    switch configuration.id {
    case .id(let id, let revision):
        modelDirectory = try await downloader.download(
            id: id, revision: revision,
            matching: modelDownloadPatterns,
            useLatest: useLatest,
            progressHandler: progressHandler)
    case .directory(let directory):
        modelDirectory = directory
    }

    let tokenizerDirectory: URL
    switch configuration.tokenizerSource {
    case .id(let id, let revision):
        tokenizerDirectory = try await downloader.download(
            id: id, revision: revision,
            matching: tokenizerDownloadPatterns,
            useLatest: useLatest,
            progressHandler: { _ in })
    case .directory(let directory):
        tokenizerDirectory = directory
    case nil:
        tokenizerDirectory = modelDirectory
    }

    return configuration.resolved(
        modelDirectory: modelDirectory,
        tokenizerDirectory: tokenizerDirectory)
}

/// Load a model given a ``ModelConfiguration``, downloading via a ``Downloader``.
///
/// Returns a ``ModelContext`` holding the model and tokenizer without
/// an `actor` providing an isolation context.
///
/// - Parameters:
///   - downloader: the ``Downloader`` to use for fetching remote resources
///   - tokenizerLoader: the ``TokenizerLoader`` to use for loading the tokenizer
///   - configuration: a ``ModelConfiguration``
///   - useLatest: when true, always checks the provider for the latest version
///   - progressHandler: optional callback for progress
/// - Returns: a ``ModelContext``
public func loadModel(
    from downloader: any Downloader,
    using tokenizerLoader: any TokenizerLoader,
    configuration: ModelConfiguration,
    useLatest: Bool = false,
    progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
) async throws -> sending ModelContext {
    try await load {
        try await $0.load(
            from: downloader, using: tokenizerLoader, configuration: configuration,
            useLatest: useLatest, progressHandler: progressHandler)
    }
}

/// Load a model given a ``ModelConfiguration``, downloading via a ``Downloader``.
///
/// Returns a ``ModelContainer`` holding a ``ModelContext``
/// inside an actor providing isolation control for the values.
///
/// - Parameters:
///   - downloader: the ``Downloader`` to use for fetching remote resources
///   - tokenizerLoader: the ``TokenizerLoader`` to use for loading the tokenizer
///   - configuration: a ``ModelConfiguration``
///   - useLatest: when true, always checks the provider for the latest version
///   - progressHandler: optional callback for progress
/// - Returns: a ``ModelContainer``
public func loadModelContainer(
    from downloader: any Downloader,
    using tokenizerLoader: any TokenizerLoader,
    configuration: ModelConfiguration,
    useLatest: Bool = false,
    progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
) async throws -> sending ModelContainer {
    try await load {
        try await $0.loadContainer(
            from: downloader, using: tokenizerLoader, configuration: configuration,
            useLatest: useLatest, progressHandler: progressHandler)
    }
}

/// Load a model given a model identifier, downloading via a ``Downloader``.
///
/// Returns a ``ModelContext`` holding the model and tokenizer without
/// an `actor` providing an isolation context.
///
/// - Parameters:
///   - downloader: the ``Downloader`` to use for fetching remote resources
///   - tokenizerLoader: the ``TokenizerLoader`` to use for loading the tokenizer
///   - id: model identifier, e.g "mlx-community/Qwen3-4B-4bit"
///   - revision: revision to download (defaults to "main")
///   - useLatest: when true, always checks the provider for the latest version
///   - progressHandler: optional callback for progress
/// - Returns: a ``ModelContext``
public func loadModel(
    from downloader: any Downloader,
    using tokenizerLoader: any TokenizerLoader,
    id: String,
    revision: String = "main",
    useLatest: Bool = false,
    progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
) async throws -> sending ModelContext {
    try await load {
        try await $0.load(
            from: downloader, using: tokenizerLoader,
            configuration: .init(id: id, revision: revision),
            useLatest: useLatest, progressHandler: progressHandler)
    }
}

/// Load a model given a model identifier, downloading via a ``Downloader``.
///
/// Returns a ``ModelContainer`` holding a ``ModelContext``
/// inside an actor providing isolation control for the values.
///
/// - Parameters:
///   - downloader: the ``Downloader`` to use for fetching remote resources
///   - tokenizerLoader: the ``TokenizerLoader`` to use for loading the tokenizer
///   - id: model identifier, e.g "mlx-community/Qwen3-4B-4bit"
///   - revision: revision to download (defaults to "main")
///   - useLatest: when true, always checks the provider for the latest version
///   - progressHandler: optional callback for progress
/// - Returns: a ``ModelContainer``
public func loadModelContainer(
    from downloader: any Downloader,
    using tokenizerLoader: any TokenizerLoader,
    id: String,
    revision: String = "main",
    useLatest: Bool = false,
    progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
) async throws -> sending ModelContainer {
    try await load {
        try await $0.loadContainer(
            from: downloader, using: tokenizerLoader,
            configuration: .init(id: id, revision: revision),
            useLatest: useLatest, progressHandler: progressHandler)
    }
}

/// Load a model from a local directory of configuration and weights.
///
/// Returns a ``ModelContext`` holding the model and tokenizer without
/// an `actor` providing an isolation context.
///
/// - Parameters:
///   - directory: directory of configuration and weights
///   - tokenizerLoader: the ``TokenizerLoader`` to use for loading the tokenizer
/// - Returns: a ``ModelContext``
public func loadModel(
    from directory: URL,
    using tokenizerLoader: any TokenizerLoader
) async throws -> sending ModelContext {
    try await load {
        try await $0.load(from: directory, using: tokenizerLoader)
    }
}

/// Load a model from a local directory AND activate the JangPress
/// cold-weight tier (axis E) per `jangPress`.
///
/// Returns the `ModelContext` paired with a `JangPressRuntime` that
/// owns the JangPress tiers (mmap / mach / embed / controller). The
/// runtime stays valid as long as the caller holds it; on drop the
/// controller's deinit cancels its memory-pressure listener and
/// quiesce timer.
///
/// When `jangPress.enabled == false` (the default — `.disabled`),
/// the runtime is `.none` and behavior is identical to the plain
/// `loadModel(from:using:)`.
///
/// **Caller contract**: hold the returned `JangPressRuntime` alive
/// for the model session. Pair it with the `ModelContext` in the
/// caller's session struct. To shut JangPress down before model
/// unload, call `JangPressActivation.deactivate(runtime)`.
///
/// - Parameters:
///   - directory: directory of configuration and weights
///   - tokenizerLoader: the ``TokenizerLoader`` to use for loading the tokenizer
///   - jangPress: optional JangPress activation options
/// - Returns: tuple of (`ModelContext`, `JangPressRuntime`)
public func loadModel(
    from directory: URL,
    using tokenizerLoader: any TokenizerLoader,
    jangPress: JangPressLoadOptions
) async throws -> (ModelContext, JangPressRuntime) {
    var context = try await load {
        try await $0.load(from: directory, using: tokenizerLoader)
    }
    let runtime = JangPressActivation.activate(
        bundleURL: directory, options: jangPress)
    context.jangPressRuntime = runtime
    return (context, runtime)
}

/// Load a model from a local directory of configuration and weights.
///
/// Returns a ``ModelContainer`` holding a ``ModelContext``
/// inside an actor providing isolation control for the values.
///
/// - Parameters:
///   - directory: directory of configuration and weights
///   - tokenizerLoader: the ``TokenizerLoader`` to use for loading the tokenizer
/// - Returns: a ``ModelContainer``
public func loadModelContainer(
    from directory: URL,
    using tokenizerLoader: any TokenizerLoader
) async throws -> sending ModelContainer {
    try await load {
        try await $0.loadContainer(from: directory, using: tokenizerLoader)
    }
}

/// Load a model from a local directory using a typed ``LoadConfiguration``,
/// returning a ``ModelContainer`` (the actor-isolated wrapper that
/// osaurus uses).
///
/// This is the **recommended entry point for production hosts**. The
/// returned `ModelContainer` already has the resulting
/// ``JangPressRuntime`` stashed via `setJangPressRuntime` so callers
/// can poll ``ModelContainer/jangPressStatus()`` from anywhere
/// (settings panel, status bar, debug console).
///
/// Resolution order matches the `ModelContext`-returning sibling:
///
/// 1. ``LoadBundleFacts/inspect(bundleURL:)`` walks the bundle.
/// 2. ``LoadConfiguration/jangPress`` resolves to a concrete
///    ``JangPressLoadOptions``.
/// 3. ``LoadConfiguration/maxResidentBytes`` is applied as
///    `MLX.Memory.cacheLimit` for the duration of the load call.
/// 4. ``LoadConfiguration/memoryLimit`` is applied as
///    `MLX.Memory.memoryLimit` (clamped to
///    `MLX.GPU.maxRecommendedWorkingSetBytes()` so the 847a8c7
///    crash condition is impossible by construction).
/// 5. The model is loaded normally and wrapped in `ModelContainer`.
/// 6. JangPress tiers are activated per the resolved options.
/// 7. The runtime is stashed on the container.
///
/// - Parameters:
///   - directory: directory of configuration and weights
///   - tokenizerLoader: the ``TokenizerLoader`` to use
///   - loadConfiguration: typed JangPress + resident-cap policy.
///     Defaults to ``LoadConfiguration/default`` (JangPress disabled,
///     70%-of-physical-RAM cache/memory caps, mmap safetensors on).
/// - Returns: a `ModelContainer` whose `jangPressRuntime` reflects
///   the activation result.
public func loadModelContainer(
    from directory: URL,
    using tokenizerLoader: any TokenizerLoader,
    loadConfiguration: LoadConfiguration
) async throws -> ModelContainer {
    let (context, runtime) = try await loadModel(
        from: directory,
        using: tokenizerLoader,
        loadConfiguration: loadConfiguration)
    let container = ModelContainer(context: context)
    container.setJangPressRuntime(runtime)
    return container
}

/// Load a model from a local directory using a typed ``LoadConfiguration``
/// — the recommended entry point for hosts (osaurus, JANG Studio) that
/// want to wire user-facing toggles to JangPress and resident-cap
/// behavior without touching env vars.
///
/// Resolution order at entry:
///
/// 1. ``LoadBundleFacts/inspect(bundleURL:)`` walks the bundle for
///    safetensors byte total + routed-MoE detection.
/// 2. ``LoadConfiguration/jangPress`` is resolved to a concrete
///    ``JangPressLoadOptions`` via ``JangPressPolicy/resolve(facts:)``.
///    Precedence: explicit value > env (`JANGPRESS=...`) > `.auto`
///    threshold (routed AND bundle > 0.5 × physical RAM).
/// 3. ``LoadConfiguration/maxResidentBytes`` is applied as
///    `MLX.Memory.cacheLimit` for the duration of the load call,
///    restored on return.
/// 4. The model is loaded normally.
/// 5. JangPress tiers are activated per the resolved options (or
///    skipped when the resolved options are ``JangPressLoadOptions/disabled``).
///
/// **Caller contract**: hold the returned `JangPressRuntime` alive for
/// the model session. Pair it with the `ModelContext` in the caller's
/// session struct. To shut JangPress down before model unload, call
/// `JangPressActivation.deactivate(runtime)`.
///
/// - Parameters:
///   - directory: directory of configuration and weights
///   - tokenizerLoader: the ``TokenizerLoader`` to use for loading the tokenizer
///   - loadConfiguration: typed JangPress + resident-cap policy.
///     Defaults to ``LoadConfiguration/default`` (JangPress disabled,
///     70%-of-physical-RAM cache/memory caps, mmap safetensors on).
/// - Returns: tuple of (`ModelContext`, `JangPressRuntime`)
public func loadModel(
    from directory: URL,
    using tokenizerLoader: any TokenizerLoader,
    loadConfiguration: LoadConfiguration
) async throws -> (ModelContext, JangPressRuntime) {
    // 1. Inspect bundle once.
    let facts = LoadBundleFacts.inspect(bundleURL: directory)

    // 2. Resolve JangPress policy → concrete options.
    let resolvedOptions = loadConfiguration.jangPress.resolve(facts: facts)

    // 3. Apply resident cap (allocator pool) for the duration of load.
    //    Skipped when `.unlimited` so existing iter-25 in-loader cap
    //    (1 GB during sanitize) is not overridden.
    let priorCap: Int?
    if let cap = loadConfiguration.maxResidentBytes
        .applyAsCacheLimitInt(physicalMemory: facts.physicalMemory)
    {
        priorCap = MLX.Memory.cacheLimit
        MLX.Memory.cacheLimit = cap
    } else {
        priorCap = nil
    }
    defer {
        if let priorCap {
            MLX.Memory.cacheLimit = priorCap
        }
    }

    // 3b. Apply MLX memory budget cap. This bounds the total MLX
    //     allocation pool — calls to malloc wait on scheduled tasks if
    //     the cap is hit. Persists for the process lifetime (no defer
    //     restore) since the cap is a sensible default, not a load-only
    //     override. Last-writer-wins across multiple loads — a future
    //     refinement could compose via WiredMemoryManager tickets.
    //
    //     SAFETY: every cap is clamped to recommendedMaxWorkingSetBytes
    //     so we never trip Apple's "limit larger than max working set
    //     size" rejection (the original 847a8c7 crash condition). See
    //     docs/WIRED-LIMIT-INVESTIGATION-2026-05-03.md.
    if let rawCap = loadConfiguration.memoryLimit
        .applyAsCacheLimitInt(physicalMemory: facts.physicalMemory)
    {
        let workingSetCap = MLX.GPU.maxRecommendedWorkingSetBytes() ?? Int.max
        let safeCap = max(1 * 1024 * 1024 * 1024,  // never below 1 GB
            min(rawCap, workingSetCap))
        MLX.Memory.memoryLimit = safeCap
    }

    // 4. Optional disk-layout preparation. The permanent prestacked
    //    safetensors overlay is now explicit opt-in via
    //    MLXPRESS_PRESTACK=1 / JANGPRESS_PRESTACK=1. MLXPress's normal
    //    low-RAM route is compression-first canonical mmap residency:
    //    load routed tensors as regular model weights through the C++
    //    whole-shard mmap loader, then use cold-page advice and OS
    //    compression/reclaim instead of a large stacked cache overlay.
    //    Active-expert pread streaming is an explicit fallback/diagnostic
    //    path only. Alignment overlays remain separately gated by
    //    MLXPRESS_ALIGN_* / JANGPRESS_ALIGN_*.
    let loadDirectory = try JangPressPrestacker.prepareBundleIfNeeded(
        originalURL: directory,
        enabled: loadConfiguration.useMmapSafetensors)

    // 5. Load the model normally. Patched osaurus mlx-swift pins honor
    //    MLX_SAFETENSORS_MMAP=1 inside loadArraysAndMetadata(url:),
    //    making the canonical weight storage mmap-backed instead of a
    //    pread() copy. Older pins ignore the env knob.
    // Tensor-buffer mmap is a diagnostics/future-offset-kernel knob, not a
    // production default. Once the C++ loader began honoring it, MiniMax-Small
    // proved that wrapping every tensor span as its own Metal buffer increases
    // load time without lowering Activity Monitor footprint; the real blocker
    // is model sanitize replacing routed mmap tensors with full stacked banks.
    let tensorBuffersRaw = ProcessInfo.processInfo.environment["MLXPRESS_MMAP_TENSOR_BUFFERS"]
        ?? ProcessInfo.processInfo.environment["JANGPRESS_MMAP_TENSOR_BUFFERS"]
        ?? ""
    let useTensorMmapBuffers = loadConfiguration.useMmapSafetensors
        && resolvedOptions.enabled
        && resolvedOptions.backend == .mmap
        && ["1", "true", "on", "yes"].contains(tensorBuffersRaw.lowercased())
    var context = try await withMmapSafetensorsEnv(
        enabled: loadConfiguration.useMmapSafetensors,
        tensorBuffers: useTensorMmapBuffers,
        startColdPercent: useTensorMmapBuffers ? resolvedOptions.compressPct : nil
    ) {
        try await NativeMTPActivation.withExplicitRequest(loadConfiguration.nativeMTP) {
            try await load {
                try await $0.load(from: loadDirectory, using: tokenizerLoader)
            }
        }
    }
    _ = adviseCanonicalMmapRoutedExpertsIfAvailable(
        options: resolvedOptions,
        mmapEnabled: loadConfiguration.useMmapSafetensors)
    JangPressCanonicalExpertAdvisor.shared.configure(
        options: resolvedOptions,
        mmapEnabled: loadConfiguration.useMmapSafetensors,
        numRoutedExperts: facts.numRoutedExperts,
        topK: facts.topK)

    // 5. Activate JangPress per resolved options. `.disabled` short-
    //    circuits inside `JangPressActivation.activate` and returns
    //    `.none` so the caller still gets a uniform tuple shape.
    let runtime = JangPressActivation.activate(
        bundleURL: loadDirectory, options: resolvedOptions)
    if JangPressPrestacker.cleanupEphemeralPrestackDirectory(loadDirectory) {
        context.configuration.id = .directory(directory)
    }
    context.jangPressRuntime = runtime
    return (context, runtime)
}

private func load<R>(loader: (ModelFactory) async throws -> sending R) async throws -> sending R {
    let factories = ModelFactoryRegistry.shared.modelFactories()
    let traceFactoryFallbacks: Bool = {
        let raw = ProcessInfo.processInfo.environment["VMLINUX_MODEL_FACTORY_TRACE"]?
            .lowercased()
        return raw == "1" || raw == "true" || raw == "on" || raw == "yes"
    }()
    // Track all failures across factories. When multiple factories fail, the
    // most informative error wins: a real load/decode/weight failure is
    // strictly more useful than `unsupportedModelType` (which just means
    // "wrong factory"). Without this preference rule, a route like the
    // mistral3 LLM-side vision_config gate (which exists to defer to the
    // VLM factory) would mask the actual VLM-side failure that the user
    // needs to see.
    var realError: Error?
    var unsupportedError: Error?
    for factory in factories {
        do {
            let model = try await loader(factory)
            return model
        } catch let error as ModelFactoryError {
            if traceFactoryFallbacks {
                print("[ModelFactory] \(type(of: factory)) failed: \(error)")
            }
            switch error {
            case .unsupportedModelType, .unsupportedProcessorType:
                if unsupportedError == nil { unsupportedError = error }
            default:
                // configurationFileError / configurationDecodingError /
                // anything else — these are real failures.
                if realError == nil { realError = error }
            }
        } catch {
            if traceFactoryFallbacks {
                print("[ModelFactory] \(type(of: factory)) failed: \(error)")
            }
            if realError == nil { realError = error }
        }
    }

    if let realError {
        throw realError
    } else if let unsupportedError {
        throw unsupportedError
    } else {
        throw ModelFactoryError.noModelFactoryAvailable
    }
}

private func withMmapSafetensorsEnv<R>(
    enabled: Bool,
    tensorBuffers: Bool,
    startColdPercent: Int?,
    _ body: () async throws -> R
) async throws -> R {
#if canImport(Darwin) || canImport(Glibc)
    let mmapKey = "MLX_SAFETENSORS_MMAP"
    let vmlxMmapKey = "VMLX_MMAP_SAFETENSORS"
    let tensorKey = "MLX_SAFETENSORS_MMAP_TENSOR_BUFFERS"
    let startColdKey = "MLX_SAFETENSORS_MMAP_START_COLD"
    let coldPctKey = "MLX_SAFETENSORS_MMAP_COLD_PCT"
    let priorMmap = getenv(mmapKey).map { String(cString: $0) }
    let priorVmlxMmap = getenv(vmlxMmapKey).map { String(cString: $0) }
    let priorTensor = getenv(tensorKey).map { String(cString: $0) }
    let priorStartCold = getenv(startColdKey).map { String(cString: $0) }
    let priorColdPct = getenv(coldPctKey).map { String(cString: $0) }
    if enabled {
        setenv(mmapKey, "1", 1)
        setenv(vmlxMmapKey, "1", 1)
        if tensorBuffers {
            setenv(tensorKey, "1", 1)
        }
        if let startColdPercent, startColdPercent > 0 {
            setenv(startColdKey, "1", 1)
            setenv(
                coldPctKey,
                String(max(0, min(100, startColdPercent))),
                1)
        }
    } else {
        unsetenv(mmapKey)
        unsetenv(vmlxMmapKey)
        unsetenv(tensorKey)
        unsetenv(startColdKey)
        unsetenv(coldPctKey)
    }
    defer {
        if let priorMmap {
            setenv(mmapKey, priorMmap, 1)
        } else {
            unsetenv(mmapKey)
        }
        if let priorVmlxMmap {
            setenv(vmlxMmapKey, priorVmlxMmap, 1)
        } else {
            unsetenv(vmlxMmapKey)
        }
        if let priorTensor {
            setenv(tensorKey, priorTensor, 1)
        } else {
            unsetenv(tensorKey)
        }
        if let priorStartCold {
            setenv(startColdKey, priorStartCold, 1)
        } else {
            unsetenv(startColdKey)
        }
        if let priorColdPct {
            setenv(coldPctKey, priorColdPct, 1)
        } else {
            unsetenv(coldPctKey)
        }
    }
#endif
    return try await body()
}

@discardableResult
private func adviseCanonicalMmapRoutedExpertsIfAvailable(
    options: JangPressLoadOptions,
    mmapEnabled: Bool
) -> Int64? {
    guard mmapEnabled,
          options.enabled,
          options.backend == .mmap,
          options.compressPct > 0
    else { return nil }

#if canImport(Cmlx)
    guard let adviseRouted = lookupSafetensorsMmapAdviseRouted() else {
        return nil
    }

    let advised = adviseRouted(
        0,  // 0 = cold advice; C++ defaults to MADV_DONTNEED and can be env-selected.
        Int32(max(0, min(100, options.compressPct))))

    let env = ProcessInfo.processInfo.environment
    if advised > 0,
       (env["MLXPRESS_DEBUG"] == "1"
        || env["JANGPRESS_DEBUG"] == "1"
        || env["MLX_SAFETENSORS_MMAP_DEBUG"] == "1")
    {
        FileHandle.standardError.write(Data(
            "[MLXPress] advised \(advised) canonical mmap routed bytes cold (pct=\(options.compressPct))\n".utf8))
    }
    return advised
#else
    return nil
#endif
}

#if canImport(Cmlx)
private typealias SafetensorsMmapAdviseRoutedFn = @convention(c) (
    Int32,
    Int32
) -> Int64

private func lookupSafetensorsMmapAdviseRouted() -> SafetensorsMmapAdviseRoutedFn? {
    guard let handle = dlopen(nil, RTLD_NOW),
          let symbol = dlsym(handle, "mlx_safetensors_mmap_advise_routed")
    else {
        return nil
    }
    return unsafeBitCast(symbol, to: SafetensorsMmapAdviseRoutedFn.self)
}
#endif

/// Protocol for types that can provide ModelFactory instances.
///
/// Not used directly.
///
/// This is used internally to provide dynamic lookup of a trampoline -- this lets
/// API in MLXLMCommon use code present in MLXLLM:
///
/// ```swift
/// public class TrampolineModelFactory: NSObject, ModelFactoryTrampoline {
///     public static func modelFactory() -> (any MLXLMCommon.ModelFactory)? {
///         LLMModelFactory.shared
///     }
/// }
/// ```
///
/// That is looked up dynamically with:
///
/// ```swift
/// {
///     (NSClassFromString("MLXVLM.TrampolineModelFactory") as? ModelFactoryTrampoline.Type)?
///         .modelFactory()
/// }
/// ```
///
/// ## See Also
/// - ``ModelFactoryRegistry``
public protocol ModelFactoryTrampoline {
    static func modelFactory() -> ModelFactory?
}

/// Registry of ``ModelFactory`` trampolines.
///
/// This allows ``loadModel(from:id:progressHandler:)`` to use any ``ModelFactory`` instances
/// available but be defined in the `LLMCommon` layer.  This is not typically used directly -- it is
/// called via ``loadModel(from:id:progressHandler:)``:
///
/// ```swift
/// let model = try await loadModel(id: "mlx-community/Qwen3-4B-4bit")
/// ```
///
/// ## See Also
/// - ``loadModel(from:id:progressHandler:)``
/// - ``loadModel(from:)-ModelContext``
/// - ``loadModelContainer(from:id:progressHandler:)``
/// - ``loadModelContainer(from:)-ModelContainer``
final public class ModelFactoryRegistry: @unchecked Sendable {
    public static let shared = ModelFactoryRegistry()

    private let lock = NSLock()
    private var trampolines: [() -> ModelFactory?]

    private init() {
        self.trampolines = [
            {
                (NSClassFromString("MLXVLM.TrampolineModelFactory") as? ModelFactoryTrampoline.Type)?
                    .modelFactory()
            },
            {
                (NSClassFromString("MLXLLM.TrampolineModelFactory") as? ModelFactoryTrampoline.Type)?
                    .modelFactory()
            },
        ]
    }

    public func addTrampoline(_ trampoline: @escaping () -> ModelFactory?) {
        lock.withLock {
            trampolines.append(trampoline)
        }
    }

    public func modelFactories() -> [ModelFactory] {
        lock.withLock {
            trampolines.compactMap { $0() }
        }
    }
}
