import Foundation
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
@preconcurrency import VMLXTokenizers

#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

public typealias MLXPressGenerateParameters = GenerateParameters
public typealias MLXPressGeneration = Generation
public typealias MLXPressCompletionInfo = GenerateCompletionInfo

public let MLXPressDefaultPrefillStepSize = 64

public func MLXPressDefaultGenerateParameters(
    maxTokens: Int? = 256,
    temperature: Float = 0,
    topP: Float = 1
) -> GenerateParameters {
    var parameters = GenerateParameters(
        maxTokens: maxTokens,
        temperature: temperature,
        topP: topP,
        prefillStepSize: MLXPressDefaultPrefillStepSize)
    parameters.draftStrategy = DraftStrategy.none
    return parameters
}

public func MLXPressBaseDecodeParameters(
    _ parameters: GenerateParameters
) -> GenerateParameters {
    var normalized = parameters
    normalized.draftStrategy = DraftStrategy.none
    return normalized
}

public enum MLXPressCompressionPolicy: Sendable, Equatable {
    case disabled
    case enabled(coldFraction: Double)
    case auto(envFallback: Bool)

    func upstream(bundleFacts facts: MLXPressBundleFacts) -> JangPressPolicy {
        switch self {
        case .disabled:
            return .disabled
        case .enabled(let coldFraction):
            return .enabled(coldFraction: coldFraction)
        case .auto(let envFallback):
            if envFallback, let envCompression = parseMLXPressEnvCompression() {
                return envCompression
            }
            return facts.isRouted ? .enabled(coldFraction: 0.70) : .disabled
        }
    }
}

public enum MLXPressResidentCap: Sendable, Equatable {
    case unlimited
    case fraction(Double)
    case absolute(UInt64)

    public static let `default`: MLXPressResidentCap = .fraction(0.70)
    public static let decodeCacheDefault: MLXPressResidentCap = .absolute(128 * 1024 * 1024)

    var upstream: ResidentCap {
        switch self {
        case .unlimited:
            return .unlimited
        case .fraction(let value):
            return .fraction(value)
        case .absolute(let bytes):
            return .absolute(bytes)
        }
    }
}

public enum MLXPressPrestackPolicy: Sendable, Equatable {
    case disabled
    case ephemeralTemporaryOverlay

    var usesEphemeralOverlay: Bool {
        switch self {
        case .disabled:
            return false
        case .ephemeralTemporaryOverlay:
            return true
        }
    }
}

public struct MLXPressCacheConfiguration: Sendable, Equatable {
    public var enabled: Bool
    public var usePagedCache: Bool
    public var enableDiskCache: Bool
    public var maxCacheBlocks: Int
    public var diskCacheMaxGB: Float
    public var diskCacheDir: URL?
    public var defaultKVMode: KVQuantizationMode
    public var defaultMaxKVSize: Int?
    public var longPromptMultiplier: Double

    public static let disabled = MLXPressCacheConfiguration(enabled: false)

    public init(
        enabled: Bool = true,
        usePagedCache: Bool = true,
        enableDiskCache: Bool = true,
        maxCacheBlocks: Int = 2_000,
        diskCacheMaxGB: Float = 10.0,
        diskCacheDir: URL? = nil,
        defaultKVMode: KVQuantizationMode = .turboQuant(keyBits: 3, valueBits: 3),
        defaultMaxKVSize: Int? = 8_192,
        longPromptMultiplier: Double = 2.0
    ) {
        self.enabled = enabled
        self.usePagedCache = usePagedCache
        self.enableDiskCache = enableDiskCache
        self.maxCacheBlocks = maxCacheBlocks
        self.diskCacheMaxGB = diskCacheMaxGB
        self.diskCacheDir = diskCacheDir
        self.defaultKVMode = defaultKVMode
        self.defaultMaxKVSize = defaultMaxKVSize
        self.longPromptMultiplier = longPromptMultiplier
    }

    func upstream(modelKey: String? = nil) -> CacheCoordinatorConfig {
        CacheCoordinatorConfig(
            usePagedCache: usePagedCache,
            enableDiskCache: enableDiskCache,
            maxCacheBlocks: maxCacheBlocks,
            diskCacheMaxGB: diskCacheMaxGB,
            diskCacheDir: diskCacheDir,
            modelKey: modelKey,
            defaultKVMode: defaultKVMode,
            defaultMaxKVSize: defaultMaxKVSize,
            longPromptMultiplier: longPromptMultiplier)
    }
}

public struct MLXPressLoadConfiguration: Sendable, Equatable {
    public var compression: MLXPressCompressionPolicy
    public var allocatorCacheLimit: MLXPressResidentCap
    public var memoryLimit: MLXPressResidentCap
    public var useMmapSafetensors: Bool
    public var enableRouterAdvice: Bool
    public var disableDecodeFusedGateUp: Bool
    public var enableActiveExpertStreaming: Bool
    public var prestack: MLXPressPrestackPolicy
    public var cache: MLXPressCacheConfiguration

    /// MLXPress base default: basic single-request runtime, mmap loader on, and
    /// cold-weight policy auto-enabled for routed bundles. The multi-tier
    /// cache coordinator is attached at load time with paged cache, disk L2,
    /// TurboQuant KV defaulting, and the long-prompt KV cap enabled. Multi-token
    /// prediction and speculative decoding are forced off in base generation so
    /// validation measures normal autoregressive decode: one accepted token per
    /// step.
    /// Decode-time gate/up fusion is disabled because it materializes extra
    /// routed-expert Metal buffers and defeats the Activity Monitor memory
    /// target on affine MoE bundles.
    /// Active-expert streaming is an explicit fallback/diagnostic path. The
    /// default MLXPress method is compression-first: load canonical weights via
    /// mmap-backed safetensors, advise inactive routed expert pages cold, and
    /// let macOS reclaim/compress those pages while keeping the model logically
    /// resident. Streaming only the active slice can keep Activity Monitor low,
    /// and its Darwin reader now uses a size-aware `pread` policy: OS-cached
    /// for routed bundles that fit the RAM warming budget, `F_NOCACHE` for
    /// Kimi-scale oversized bundles. Repeated SSD scatter reads still defeat the
    /// core MLXPress premise and must not be the default success path.
    /// Ephemeral prestack is an explicit transitional resident-compute option:
    /// it builds a temporary stacked JANGTQ overlay under `/tmp`, maps that
    /// overlay through the mmap loader, and removes the overlay after load once
    /// mappings are established, with a process-exit fallback. It is useful for
    /// MiniMax-class unstacked bundles while the no-overlay resident kernel path
    /// is still being built, but it is not the default and never writes a
    /// permanent prestack cache.
    public static let `default` = MLXPressLoadConfiguration()

    /// Strict plain MLX load/decode path.
    public static let plain = MLXPressLoadConfiguration(
        compression: .disabled,
        allocatorCacheLimit: .unlimited,
        memoryLimit: .unlimited,
        useMmapSafetensors: false,
        enableRouterAdvice: false,
        disableDecodeFusedGateUp: false,
        enableActiveExpertStreaming: false,
        prestack: .disabled,
        cache: .disabled)

    public init(
        compression: MLXPressCompressionPolicy = .auto(envFallback: true),
        allocatorCacheLimit: MLXPressResidentCap = .decodeCacheDefault,
        memoryLimit: MLXPressResidentCap = .default,
        useMmapSafetensors: Bool = true,
        enableRouterAdvice: Bool = false,
        disableDecodeFusedGateUp: Bool = true,
        enableActiveExpertStreaming: Bool = false,
        prestack: MLXPressPrestackPolicy = .disabled,
        cache: MLXPressCacheConfiguration = MLXPressCacheConfiguration()
    ) {
        self.compression = compression
        self.allocatorCacheLimit = allocatorCacheLimit
        self.memoryLimit = memoryLimit
        self.useMmapSafetensors = useMmapSafetensors
        self.enableRouterAdvice = enableRouterAdvice
        self.disableDecodeFusedGateUp = disableDecodeFusedGateUp
        self.enableActiveExpertStreaming = enableActiveExpertStreaming
        self.prestack = prestack
        self.cache = cache
    }

    func upstream(bundleFacts facts: MLXPressBundleFacts) -> LoadConfiguration {
        LoadConfiguration(
            jangPress: compression.upstream(bundleFacts: facts),
            maxResidentBytes: allocatorCacheLimit.upstream,
            memoryLimit: memoryLimit.upstream,
            useMmapSafetensors: useMmapSafetensors)
    }
}

public struct MLXPressStatus: Sendable, Equatable {
    public let enabled: Bool
    public let coldFraction: Double?
    public let backend: String
    public let tilesUnderManagement: Int
    public let totalRoutedBytes: UInt64

    init(_ upstream: JangPressStatus) {
        self.enabled = upstream.enabled
        self.coldFraction = upstream.coldFraction
        self.backend = upstream.backend.rawValue
        self.tilesUnderManagement = upstream.tilesUnderManagement
        self.totalRoutedBytes = upstream.totalRoutedBytes
    }
}

public struct MLXPressCacheStatus: Sendable, Equatable {
    public let enabled: Bool
    public let pagedCacheEnabled: Bool
    public let diskCacheEnabled: Bool
    public let hybrid: Bool
    public let pagedIncompatible: Bool
    public let defaultKVMode: String
    public let defaultMaxKVSize: Int?

    public init(coordinator: CacheCoordinator?) {
        guard let coordinator else {
            self.enabled = false
            self.pagedCacheEnabled = false
            self.diskCacheEnabled = false
            self.hybrid = false
            self.pagedIncompatible = false
            self.defaultKVMode = "none"
            self.defaultMaxKVSize = nil
            return
        }
        self.enabled = true
        self.pagedCacheEnabled = coordinator.pagedCache != nil
        self.diskCacheEnabled = coordinator.diskCache != nil
        self.hybrid = coordinator.isHybrid
        self.pagedIncompatible = coordinator.isPagedIncompatible
        self.defaultKVMode = MLXPressDescribeKVMode(coordinator.config.defaultKVMode)
        self.defaultMaxKVSize = coordinator.config.defaultMaxKVSize
    }
}

public struct MLXPressTokenizerLoader: TokenizerLoader {
    private let upstream: any TokenizerLoader

    public init() {
        self.upstream = #huggingFaceTokenizerLoader()
    }

    public func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
        try await upstream.load(from: directory)
    }
}

public final class MLXPressSession: Sendable {
    public let modelDirectory: URL
    public let bundleFacts: MLXPressBundleFacts
    public let container: ModelContainer
    private let allocatorCacheLimit: MLXPressResidentCap
    private let disableDecodeFusedGateUp: Bool

    private init(
        modelDirectory: URL,
        bundleFacts: MLXPressBundleFacts,
        container: ModelContainer,
        allocatorCacheLimit: MLXPressResidentCap,
        disableDecodeFusedGateUp: Bool
    ) {
        self.modelDirectory = modelDirectory
        self.bundleFacts = bundleFacts
        self.container = container
        self.allocatorCacheLimit = allocatorCacheLimit
        self.disableDecodeFusedGateUp = disableDecodeFusedGateUp
    }

    public static func load(
        from modelDirectory: URL,
        configuration: MLXPressLoadConfiguration = .default,
        tokenizerLoader: any TokenizerLoader = MLXPressTokenizerLoader()
    ) async throws -> MLXPressSession {
        let facts = MLXPressBundleFacts.inspect(at: modelDirectory)
        applyDecodeMemoryDefaults(configuration: configuration)
        let streamExperts = shouldUseActiveExpertStreaming(
            facts: facts,
            configuration: configuration)
        let useEphemeralPrestack = shouldUseEphemeralPrestack(
            facts: facts,
            configuration: configuration,
            activeExpertStreaming: streamExperts)
        let container = try await withEphemeralPrestackEnv(enabled: useEphemeralPrestack) {
            try await withActiveExpertStreamingEnv(
                enabled: streamExperts,
                modelDirectory: modelDirectory
            ) {
                try await withRouterAdviceEnv(configuration.enableRouterAdvice) {
                    try await loadModelContainer(
                        from: modelDirectory,
                        using: tokenizerLoader,
                        loadConfiguration: configuration.upstream(bundleFacts: facts))
                }
            }
        }
        if configuration.cache.enabled {
            await container.enableCachingAsync(
                config: configuration.cache.upstream(
                    modelKey: cacheModelKey(modelDirectory: modelDirectory, facts: facts)))
        }
        MLX.Memory.clearCache()
        return MLXPressSession(
            modelDirectory: modelDirectory,
            bundleFacts: facts,
            container: container,
            allocatorCacheLimit: configuration.allocatorCacheLimit,
            disableDecodeFusedGateUp: configuration.disableDecodeFusedGateUp)
    }

    public func status() -> MLXPressStatus {
        MLXPressStatus(container.jangPressStatus())
    }

    public func cacheStatus() -> MLXPressCacheStatus {
        MLXPressCacheStatus(coordinator: container.cacheCoordinator)
    }

    public func memorySnapshot() -> MLXPressMemorySnapshot {
        MLXPressMemorySnapshot.current()
    }

    public func encode(_ text: String) async -> [Int] {
        await container.encode(text)
    }

    public func decode(tokenIds: [Int]) async -> String {
        await container.decode(tokenIds: tokenIds)
    }

    public func generate(
        prompt: String,
        parameters: GenerateParameters = MLXPressDefaultGenerateParameters()
    ) async throws -> AsyncStream<Generation> {
        try await generate(
            input: UserInput(prompt: prompt),
            parameters: parameters)
    }

    public func generate(
        input: consuming sending UserInput,
        parameters: GenerateParameters = MLXPressDefaultGenerateParameters()
    ) async throws -> AsyncStream<Generation> {
        applyDecodeMemoryDefaults(
            allocatorCacheLimit: allocatorCacheLimit,
            disableDecodeFusedGateUp: disableDecodeFusedGateUp)
        let baseParameters = MLXPressBaseDecodeParameters(parameters)
        let prepared = try await container.prepare(input: input)
        return try await container.generate(input: prepared, parameters: baseParameters)
    }

    public func streamText(
        prompt: String,
        parameters: GenerateParameters = MLXPressDefaultGenerateParameters()
    ) async throws -> AsyncStream<String> {
        let source = try await generate(prompt: prompt, parameters: parameters)
        return AsyncStream { continuation in
            Task {
                for await event in source {
                    if case .chunk(let text) = event, !text.isEmpty {
                        continuation.yield(text)
                    }
                }
                continuation.finish()
            }
        }
    }
}

private func applyDecodeMemoryDefaults(configuration: MLXPressLoadConfiguration) {
    applyDecodeMemoryDefaults(
        allocatorCacheLimit: configuration.allocatorCacheLimit,
        disableDecodeFusedGateUp: configuration.disableDecodeFusedGateUp)
}

private func cacheModelKey(modelDirectory: URL, facts: MLXPressBundleFacts) -> String {
    let standardized = modelDirectory.standardizedFileURL
    return [
        "mlxpress",
        standardized.path,
        facts.format.rawValue,
        facts.modelType ?? "unknown",
        String(facts.totalSafetensorsBytes),
        safetensorsSignature(modelDirectory: standardized),
    ].joined(separator: "|")
}

private func safetensorsSignature(modelDirectory: URL) -> String {
    let fm = FileManager.default
    let files = (try? fm.contentsOfDirectory(
        at: modelDirectory,
        includingPropertiesForKeys: [.fileSizeKey, .contentModificationDateKey],
        options: [.skipsHiddenFiles])) ?? []
    return files
        .filter { $0.pathExtension == "safetensors" }
        .sorted { $0.lastPathComponent < $1.lastPathComponent }
        .map { file in
            let values = try? file.resourceValues(forKeys: [
                .fileSizeKey, .contentModificationDateKey,
            ])
            let size = values?.fileSize ?? -1
            let modified = values?.contentModificationDate?.timeIntervalSince1970 ?? 0
            return "\(file.lastPathComponent):\(size):\(String(format: "%.0f", modified))"
        }
        .joined(separator: ";")
}

private func applyDecodeMemoryDefaults(
    allocatorCacheLimit: MLXPressResidentCap,
    disableDecodeFusedGateUp: Bool
) {
    let cap = decodeCacheLimitOverride() ?? allocatorCacheLimit
    if let cacheLimit = cap.upstream.applyAsCacheLimitInt(
        physicalMemory: ProcessInfo.processInfo.physicalMemory)
    {
        MLX.Memory.cacheLimit = cacheLimit
    }
    guard disableDecodeFusedGateUp,
        getenvString("MLXPRESS_ALLOW_FUSED_GATE_UP") != "1"
    else { return }
    setenv("BENCH_NO_FUSED_GATE_UP", "1", 1)
}

private func decodeCacheLimitOverride() -> MLXPressResidentCap? {
    for key in ["MLXPRESS_DECODE_CACHE_LIMIT_MB", "JANGPRESS_DECODE_CACHE_LIMIT_MB"] {
        guard let raw = getenvString(key)?.trimmingCharacters(in: .whitespacesAndNewlines),
            !raw.isEmpty
        else { continue }
        let value = raw.lowercased()
        if value == "off" || value == "unlimited" || value == "none" {
            return .unlimited
        }
        guard let mb = Double(value), mb >= 0 else { continue }
        if mb == 0 {
            return .unlimited
        }
        let bytes = mb * 1024.0 * 1024.0
        return .absolute(UInt64(bytes.rounded()))
    }
    return nil
}

private func withRouterAdviceEnv<R>(
    _ enabled: Bool,
    operation: () async throws -> R
) async throws -> R {
    guard enabled else {
        return try await operation()
    }

    let keys = ["MLXPRESS_ROUTER_ADVICE", "JANGPRESS_ROUTER_ADVICE"]
    let previous = Dictionary(uniqueKeysWithValues: keys.map {
        ($0, getenvString($0))
    })
    for key in keys {
        setenv(key, "1", 1)
    }
    defer {
        for key in keys {
            if let value = previous[key] ?? nil {
                setenv(key, value, 1)
            } else {
                unsetenv(key)
            }
        }
    }
    return try await operation()
}

private func shouldUseActiveExpertStreaming(
    facts: MLXPressBundleFacts,
    configuration: MLXPressLoadConfiguration
) -> Bool {
    configuration.enableActiveExpertStreaming
        && facts.format == .jangTQ
        && facts.isRouted
        && JANGTQStreamingExperts.hasStreamableExperts(in: facts.directory)
}

private func shouldUseEphemeralPrestack(
    facts: MLXPressBundleFacts,
    configuration: MLXPressLoadConfiguration,
    activeExpertStreaming: Bool
) -> Bool {
    configuration.prestack.usesEphemeralOverlay
        && !activeExpertStreaming
        && configuration.useMmapSafetensors
        && facts.format == .jangTQ
        && facts.isRouted
}

private func withEphemeralPrestackEnv<R>(
    enabled: Bool,
    operation: () async throws -> R
) async throws -> R {
    guard enabled else {
        return try await operation()
    }

    let keys = [
        "MLXPRESS_PRESTACK",
        "JANGPRESS_PRESTACK",
        "MLXPRESS_PRESTACK_EPHEMERAL",
        "JANGPRESS_PRESTACK_EPHEMERAL",
        "MLXPRESS_PRESTACK_STRICT",
        "JANGPRESS_PRESTACK_STRICT",
    ]
    let previous = Dictionary(uniqueKeysWithValues: keys.map {
        ($0, getenvString($0))
    })
    setenv("MLXPRESS_PRESTACK", "1", 1)
    setenv("JANGPRESS_PRESTACK", "1", 1)
    setenv("MLXPRESS_PRESTACK_EPHEMERAL", "1", 1)
    setenv("JANGPRESS_PRESTACK_EPHEMERAL", "1", 1)
    setenv("MLXPRESS_PRESTACK_STRICT", "1", 1)
    setenv("JANGPRESS_PRESTACK_STRICT", "1", 1)
    FileHandle.standardError.write(Data(
        "[MLXPress] ephemeral prestack enabled; temporary routed overlay will be removed after mmap load\n".utf8))
    defer {
        for key in keys {
            if let value = previous[key] ?? nil {
                setenv(key, value, 1)
            } else {
                unsetenv(key)
            }
        }
    }
    return try await operation()
}

private func withActiveExpertStreamingEnv<R>(
    enabled: Bool,
    modelDirectory: URL,
    operation: () async throws -> R
) async throws -> R {
    guard enabled else {
        return try await operation()
    }

    let keys = [
        "MLXPRESS_STREAMING_EXPERTS",
        "MLXPRESS_MODEL_DIR",
        "MLXPRESS_STREAMING_EVAL_EACH_LAYER",
        "MLXPRESS_PRESTACK",
        "JANGPRESS_PRESTACK",
    ]
    let previous = Dictionary(uniqueKeysWithValues: keys.map {
        ($0, getenvString($0))
    })
    setenv("MLXPRESS_STREAMING_EXPERTS", "1", 1)
    setenv("MLXPRESS_MODEL_DIR", modelDirectory.resolvingSymlinksInPath().path, 1)
    if getenv("MLXPRESS_STREAMING_EVAL_EACH_LAYER") == nil {
        setenv("MLXPRESS_STREAMING_EVAL_EACH_LAYER", "1", 1)
    }
    setenv("MLXPRESS_PRESTACK", "0", 1)
    setenv("JANGPRESS_PRESTACK", "0", 1)
    FileHandle.standardError.write(Data(
        "[MLXPress] active-expert streaming enabled model=\(modelDirectory.lastPathComponent)\n".utf8))
    defer {
        for key in keys {
            if let value = previous[key] ?? nil {
                setenv(key, value, 1)
            } else {
                unsetenv(key)
            }
        }
    }
    return try await operation()
}

private func getenvString(_ key: String) -> String? {
    guard let raw = getenv(key) else { return nil }
    return String(cString: raw)
}

public func MLXPressDescribeKVMode(_ mode: KVQuantizationMode) -> String {
    switch mode {
    case .none:
        return "none"
    case .affine(let bits, let groupSize):
        return "affine(\(bits),group=\(groupSize))"
    case .turboQuant(let keyBits, let valueBits):
        return "turboQuant(k=\(keyBits),v=\(valueBits))"
    }
}

public func MLXPressClearMemoryCache() {
    MLX.Memory.clearCache()
}

private func parseMLXPressEnvCompression() -> JangPressPolicy? {
    guard let raw = getenvString("MLXPRESS") ?? getenvString("JANGPRESS") else {
        return nil
    }
    let value = raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
    if value == "0" || value == "off" || value == "false" || value == "no" {
        return .disabled
    }
    if let pct = Int(value), (0...95).contains(pct) {
        return .enabled(coldFraction: Double(pct) / 100.0)
    }
    return nil
}
