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
    // Shared across context copies; released with the loaded context, including
    // cancelled/failed load cleanup. Never keyed by a model name or directory.
    var residencyLease: ModelResidency.Lease?

    /// Whether this model supports vision/image input (is a VLM).
    public var isVLM: Bool { model is VisionLanguageModelProtocol }

    public init(
        configuration: ModelConfiguration, model: any LanguageModel,
        processor: any UserInputProcessor, tokenizer: any Tokenizer
    ) {
        self.configuration = configuration
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer
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

    /// Load a model from a local directory with caller-supplied
    /// `ModelConfiguration` overrides, producing a ``ModelContext``.
    ///
    /// The model source is forced to `directory`, but fields such as
    /// `toolCallFormat`, `reasoningParserName`, `generationDefaults`, and
    /// `mtpStatus` are preserved. This lets hosts combine local-directory
    /// loading with server-panel parser overrides without bypassing the
    /// factory registry.
    public func load(
        from directory: URL,
        using tokenizerLoader: any TokenizerLoader,
        configuration: ModelConfiguration
    ) async throws -> sending ModelContext {
        let tokenizerDirectory: URL
        switch configuration.tokenizerSource {
        case .directory(let directory):
            tokenizerDirectory = directory
        case .id:
            throw ModelFactoryError.unsupportedModelType(
                "local directory load with remote tokenizerSource requires a Downloader")
        case nil:
            tokenizerDirectory = directory
        }
        var localConfiguration = configuration
        localConfiguration.id = .directory(directory)
        return try await _load(
            configuration: localConfiguration.resolved(
                modelDirectory: directory,
                tokenizerDirectory: tokenizerDirectory),
            tokenizerLoader: tokenizerLoader)
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

    /// Load a model from a local directory with caller-supplied
    /// `ModelConfiguration` overrides, producing a ``ModelContainer``.
    public func loadContainer(
        from directory: URL,
        using tokenizerLoader: any TokenizerLoader,
        configuration: ModelConfiguration
    ) async throws -> ModelContainer {
        let context = try await load(
            from: directory,
            using: tokenizerLoader,
            configuration: configuration)
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

/// Load a model with explicit memory, residency and mmap settings.
/// The container owns the context and its residency lease until unload.
public func loadModelContainer(
    from directory: URL,
    using tokenizerLoader: any TokenizerLoader,
    loadConfiguration: LoadConfiguration
) async throws -> ModelContainer {
    let context = try await loadModel(
        from: directory,
        using: tokenizerLoader,
        loadConfiguration: loadConfiguration)
    let container = ModelContainer(context: context)
    return container
}

/// Load a local-directory model with both caller-supplied
/// ``ModelConfiguration`` overrides and a typed ``LoadConfiguration``.
///
/// Hosts use this when server-panel parser/tool overrides must coexist with
/// mmap safetensors, memory caps, and native-MTP activation.
public func loadModelContainer(
    from directory: URL,
    using tokenizerLoader: any TokenizerLoader,
    configuration: ModelConfiguration,
    loadConfiguration: LoadConfiguration
) async throws -> ModelContainer {
    let context = try await loadModel(
        from: directory,
        using: tokenizerLoader,
        configuration: configuration,
        loadConfiguration: loadConfiguration)
    let container = ModelContainer(context: context)
    return container
}

/// Load a context with typed allocation, residency, parser and model settings.
/// Safetensors remain canonical; no parallel cold-weight tier is created.
public func loadModel(
    from directory: URL,
    using tokenizerLoader: any TokenizerLoader,
    loadConfiguration: LoadConfiguration
) async throws -> ModelContext {
    try await loadModel(
        from: directory,
        using: tokenizerLoader,
        configuration: .init(directory: directory),
        loadConfiguration: loadConfiguration)
}

/// Load a model from a local directory using caller-supplied
/// ``ModelConfiguration`` overrides plus a typed ``LoadConfiguration``.
public func loadModel(
    from directory: URL,
    using tokenizerLoader: any TokenizerLoader,
    configuration: ModelConfiguration,
    loadConfiguration: LoadConfiguration
) async throws -> ModelContext {
    // 1. Inspect bundle once.
    var loadTiming = ModelLoadTiming(scope: "runtime")
    let facts = LoadBundleFacts.inspect(bundleURL: directory)
    loadTiming.mark("bundle_inspection")
    let memoryLimit = loadConfiguration.resolvedSchedulingMemoryLimit(
        facts: facts, recommendedWorkingSetBytes: MLX.GPU.maxRecommendedWorkingSetBytes())
    let nativeResidentPoolCeiling =
        applyResidentPoolProcessMemoryLimitsIfNeeded(
            facts: facts,
            allocatorCacheLimit: loadConfiguration.maxResidentBytes
                .applyAsCacheLimitInt(physicalMemory: facts.physicalMemory))
    if loadConfiguration.memoryLimit != memoryLimit {
        LoadDiagnostics.write(
            Data(
                (facts.customRoutedFormat == .janghV2
                    ? "[Load] Automatic JANGH scheduling budget includes mapped weights plus scratch headroom; explicit limits and physical-memory admission remain separate\n"
                    : "[Load] Resident affine bundle uses RAM admission instead of the decode-throttling MLX memory limit\n")
                    .utf8))
    }

    // 3. Apply resident cap (allocator pool) for the duration of load.
    //    Skipped when `.unlimited` so existing iter-25 in-loader cap
    //    (1 GB during sanitize) is not overridden.
    let priorCap: Int?
    if nativeResidentPoolCeiling != nil {
        // This policy needs the native freed-buffer reuse pool for decode. Keep the
        // restored ceiling after load instead of putting a stale model-specific
        // cap back in the defer below.
        priorCap = nil
    } else if let cap = loadConfiguration.maxResidentBytes
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

    // 3b. Apply the process-wide MLX allocator scheduling/reuse budget.
    // This fork drains reusable buffers at the GC threshold; it is not a
    // physical-footprint cap. Respect explicit limits and the device ceiling.
    if nativeResidentPoolCeiling == nil,
        let rawCap =
            memoryLimit
            .applyAsCacheLimitInt(physicalMemory: facts.physicalMemory)
    {
        let workingSetCap = MLX.GPU.maxRecommendedWorkingSetBytes() ?? Int.max
        let safeCap = max(
            1 * 1024 * 1024 * 1024,  // never below 1 GB
            min(rawCap, workingSetCap))
        MLX.Memory.memoryLimit = safeCap
    }

    // 3c. Apply the selected model-residency budget. Unlike the allocator
    // reuse budget above, wiring makes pages non-reclaimable. The policy must
    // leave an OS reserve and respect the reported Metal ceiling. SSD-only
    // n-gram tables are excluded from weight accounting.
    let modelBytes = Int(clamping: facts.gpuResidentWeightBytes)
    let workingSet = MLX.GPU.maxRecommendedWorkingSetBytes() ?? Int.max
    let physical = Int(clamping: facts.physicalMemory)
    // Keep other contexts' residency intact. This snapshot includes existing
    // process wiring; the report describes headroom before this incoming load.
    let wireBudget = SystemWireBudget.snapshot()
    let residency = ModelResidency.shared.acquire(.init(
        weightBytes: modelBytes, enabled: loadConfiguration.keepModelInMemory,
        physicalMemory: physical, workingSet: workingSet,
        systemWireLimit: wireBudget.userWireLimitBytes))
    let residencyTarget = residency.target > 0 ? residency.target : nil
    let residencyApplied = residency.applied
    let wireReport = ModelWireBudgetReport(
        modelBytes: modelBytes, budget: wireBudget, residencyTargetBytes: residencyTarget)
    ModelWireBudgetReport.recordLastLoad(wireReport)
    LoadDiagnostics.write(Data(
        "[Load] Model residency requested=\(loadConfiguration.keepModelInMemory) target_bytes=\(residencyTarget ?? 0) weight_bytes=\(modelBytes) working_set_bytes=\(workingSet) applied=\(residencyApplied)\n".utf8))
    LoadDiagnostics.write(Data("\(wireReport.logLine)\n".utf8))
    loadTiming.mark("memory_and_residency")

    // Load original packed bytes into owned allocations; alignment repair and
    // mapped/prestacked overlays are no longer part of the production factory.
    let loadDirectory = directory
    loadTiming.mark("bundle_preparation")

    // Owned tensor storage is the single model-loading path. No process-wide
    // mmap environment mutation around an async factory operation.
    var context = try await {
        try await DeepseekV4ActivationQAT.withExplicitRequest(
            loadConfiguration.deepseekV4ActivationQAT
        ) {
            try await NativeMTPActivation.withExplicitRequest(loadConfiguration.nativeMTP) {
                try await NativeMTPActivation.$manualDepthOverride.withValue(
                    loadConfiguration.nativeMTPManualDepth
                ) {
                    try await load {
                        try await $0.load(
                            from: loadDirectory,
                            using: tokenizerLoader,
                            configuration: configuration)
                    }
                }
            }
        }
    }()
    loadTiming.mark("factory_load")

    context.residencyLease = residency.lease
    loadTiming.mark("runtime_activation")
    return context
}

/// Apply the existing resident-pool policy to MLX's process-wide ceilings.
///
/// An `.unlimited` resolved cap produces no integer assignment. Without an
/// explicit reset, a preceding capped model leaves its limit active for the
/// next model. Both the loader and policy resolver must use the same bundle
/// property: selecting this application step by a separate architecture check
/// left GLM's advertised unlimited policy running under the previous 70% cap.
/// Pre-load RAM admission remains the host's gate; this restores the native
/// MLX ceilings only for bundles already selected by that policy.
@discardableResult
func applyResidentPoolProcessMemoryLimitsIfNeeded(
    facts: LoadBundleFacts,
    allocatorCacheLimit: Int? = nil,
    recommendedWorkingSetBytes: Int? = MLX.GPU.maxRecommendedWorkingSetBytes()
) -> Int? {
    guard facts.requiresUncappedResidentPools else { return nil }

    let physical = Int(clamping: facts.physicalMemory)
    let physical95 = (physical / 20) * 19 + ((physical % 20) * 19) / 20
    let recommended150: Int = {
        guard let recommendedWorkingSetBytes else { return Int.max }
        let half = recommendedWorkingSetBytes / 2
        guard recommendedWorkingSetBytes <= Int.max - half else { return Int.max }
        return recommendedWorkingSetBytes + half
    }()
    let nativeCeiling = max(1 << 30, min(physical95, recommended150))
    let previousMemoryLimit = MLX.Memory.memoryLimit
    let previousCacheLimit = MLX.Memory.cacheLimit
    MLX.Memory.memoryLimit = nativeCeiling
    MLX.Memory.cacheLimit = min(nativeCeiling, max(0, allocatorCacheLimit ?? nativeCeiling))
    if ProcessInfo.processInfo.environment["VMLX_CACHE_FETCH_TRACE"] != nil {
        LoadDiagnostics.write(Data(
            ("[Load] resident-pool limits model_type=\(facts.modelType ?? "unknown") "
                + "previous_memory=\(previousMemoryLimit) previous_cache=\(previousCacheLimit) "
                    + "native_ceiling=\(nativeCeiling) allocator_ceiling=\(MLX.Memory.cacheLimit)\n")
                    .utf8))
    }
    return nativeCeiling
}

private func load<R>(loader: (ModelFactory) async throws -> sending R) async throws -> sending R {
    let factories = ModelFactoryRegistry.shared.modelFactories()
    let traceFactoryFallbacks: Bool = {
        let raw = RuntimeEnvironment.value("VMLX_MODEL_FACTORY_TRACE")?
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
