// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation

/// Stable engine-side contract for an Osaurus server/session settings panel.
///
/// The panel should treat these values as engine inputs, not as UI-only state.
/// `nil` means "do not override the bundle/engine default"; it must not be
/// converted into hidden sampling clamps or family-specific recovery behavior.
public struct VMLXServerRuntimeSettings: Codable, Sendable, Equatable {
    public static let contractVersion = 1

    public var network: VMLXServerNetworkSettings
    public var concurrency: VMLXServerConcurrencySettings
    public var cache: VMLXServerCacheSettings
    public var power: VMLXServerPowerSettings
    public var generation: VMLXServerGenerationDefaults
    public var tools: VMLXServerToolSettings
    public var multimodal: VMLXServerMultimodalSettings
    public var mtp: VMLXServerMTPSettings

    public init(
        network: VMLXServerNetworkSettings = .init(),
        concurrency: VMLXServerConcurrencySettings = .init(),
        cache: VMLXServerCacheSettings = .init(),
        power: VMLXServerPowerSettings = .init(),
        generation: VMLXServerGenerationDefaults = .init(),
        tools: VMLXServerToolSettings = .init(),
        multimodal: VMLXServerMultimodalSettings = .init(),
        mtp: VMLXServerMTPSettings = .init()
    ) {
        self.network = network
        self.concurrency = concurrency
        self.cache = cache
        self.power = power
        self.generation = generation
        self.tools = tools
        self.multimodal = multimodal
        self.mtp = mtp
    }

    public func validationIssues(
        mtpStatus: MTPBundleStatus? = nil
    ) -> [VMLXServerSettingsIssue] {
        var issues: [VMLXServerSettingsIssue] = []

        if network.host.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            issues.append(.error(
                field: "network.host",
                message: "Server host cannot be empty."))
        }
        if let port = network.port,
           !(1...65_535).contains(port) {
            issues.append(.error(
                field: "network.port",
                message: "Server port must be between 1 and 65535."))
        }
        if let rateLimit = network.rateLimitRequestsPerMinute,
           rateLimit <= 0 {
            issues.append(.error(
                field: "network.rateLimitRequestsPerMinute",
                message: "Rate limit must be positive. Use nil to disable rate limiting."))
        }
        if let timeout = network.timeoutSeconds,
           timeout <= 0 {
            issues.append(.error(
                field: "network.timeoutSeconds",
                message: "Timeout must be positive. Use nil for no timeout."))
        }
        if let maxConcurrent = concurrency.maxConcurrentSequences,
           maxConcurrent <= 0 {
            issues.append(.error(
                field: "concurrency.maxConcurrentSequences",
                message: "Max concurrent sequences must be positive."))
        }
        if let prefillBatchSize = concurrency.prefillBatchSize,
           prefillBatchSize <= 0 {
            issues.append(.error(
                field: "concurrency.prefillBatchSize",
                message: "Prefill batch size must be positive."))
        }
        if let prefillStepSize = concurrency.prefillStepSize,
           prefillStepSize <= 0 {
            issues.append(.error(
                field: "concurrency.prefillStepSize",
                message: "Prefill step size must be positive."))
        }
        if let completionBatchSize = concurrency.completionBatchSize,
           completionBatchSize <= 0 {
            issues.append(.error(
                field: "concurrency.completionBatchSize",
                message: "Completion batch size must be positive."))
        }
        if cache.pagedKV.enabled && cache.legacyDisk.enabled {
            issues.append(.error(
                field: "cache.legacyDisk.enabled",
                message: "Legacy disk cache cannot run at the same time as paged KV cache. Use block disk L2 for paged cache persistence."))
        }
        if !concurrency.continuousBatching &&
            (cache.prefix.enabled || cache.pagedKV.enabled || cache.blockDisk.enabled) {
            issues.append(.warning(
                field: "concurrency.continuousBatching",
                message: "Continuous batching is off, so prefix/paged/block-disk cache reuse will be limited or disabled."))
        }
        if let light = power.lightSleepAfterSeconds,
           light <= 0 {
            issues.append(.error(
                field: "power.lightSleepAfterSeconds",
                message: "Light sleep must be positive. Use nil to disable light sleep."))
        }
        if let deep = power.deepSleepAfterSeconds,
           deep <= 0 {
            issues.append(.error(
                field: "power.deepSleepAfterSeconds",
                message: "Deep sleep must be positive. Use nil to disable deep sleep."))
        }
        if let light = power.lightSleepAfterSeconds,
           let deep = power.deepSleepAfterSeconds,
           light > 0,
           deep > 0,
           deep <= light {
            issues.append(.error(
                field: "power.deepSleepAfterSeconds",
                message: "Deep sleep must be later than light sleep."))
        }
        if let streamInterval = generation.streamInterval,
           streamInterval < 1 {
            issues.append(.error(
                field: "generation.streamInterval",
                message: "Stream interval must be at least 1."))
        }
        if let temperature = generation.temperature,
           temperature < 0 {
            issues.append(.error(
                field: "generation.temperature",
                message: "Temperature cannot be negative."))
        }
        if let topP = generation.topP,
           !(0...1).contains(topP) {
            issues.append(.error(
                field: "generation.topP",
                message: "Top-P must be between 0 and 1."))
        }
        if let minP = generation.minP,
           !(0...1).contains(minP) {
            issues.append(.error(
                field: "generation.minP",
                message: "Min-P must be between 0 and 1."))
        }
        if let topK = generation.topK,
           topK < 0 {
            issues.append(.error(
                field: "generation.topK",
                message: "Top-K cannot be negative. Use nil for model default or 0 for disabled when supported."))
        }
        if let repetitionPenalty = generation.repetitionPenalty,
           repetitionPenalty <= 0 {
            issues.append(.error(
                field: "generation.repetitionPenalty",
                message: "Repetition penalty must be positive."))
        }
        if let memoryLimit = cache.prefix.memoryLimitMB, memoryLimit <= 0 {
            issues.append(.error(
                field: "cache.prefix.memoryLimitMB",
                message: "Prefix cache memory limit must be positive."))
        }
        if let memoryPercent = cache.prefix.memoryPercent,
           memoryPercent <= 0 || memoryPercent > 100 {
            issues.append(.error(
                field: "cache.prefix.memoryPercent",
                message: "Prefix cache memory percent must be greater than 0 and at most 100."))
        }
        if let ttl = cache.prefix.ttlMinutes, ttl <= 0 {
            issues.append(.error(
                field: "cache.prefix.ttlMinutes",
                message: "Prefix cache TTL must be positive. Use nil for no expiration."))
        }
        if let blockSize = cache.pagedKV.blockSize, blockSize <= 0 {
            issues.append(.error(
                field: "cache.pagedKV.blockSize",
                message: "Paged KV block size must be positive."))
        }
        if let maxBlocks = cache.pagedKV.maxBlocks, maxBlocks <= 0 {
            issues.append(.error(
                field: "cache.pagedKV.maxBlocks",
                message: "Paged KV max blocks must be positive."))
        }
        if let maxSize = cache.blockDisk.maxSizeGB, maxSize <= 0 {
            issues.append(.error(
                field: "cache.blockDisk.maxSizeGB",
                message: "Block disk L2 cache size must be positive."))
        }
        if let maxSize = cache.legacyDisk.maxSizeGB, maxSize <= 0 {
            issues.append(.error(
                field: "cache.legacyDisk.maxSizeGB",
                message: "Legacy disk cache size must be positive."))
        }
        if cache.liveKVCodec == .turboQuant {
            if cache.turboQuantKeyBits == nil || cache.turboQuantValueBits == nil {
                issues.append(.error(
                    field: "cache.liveKVCodec",
                    message: "TurboQuant KV requires explicit key and value bit widths."))
            }
        }
        if !multimodal.requireMediaSaltForCache
            && (cache.prefix.enabled
                || cache.pagedKV.enabled
                || cache.blockDisk.enabled
                || cache.legacyDisk.enabled) {
            issues.append(.error(
                field: "multimodal.requireMediaSaltForCache",
                message: "Media salt is required when any prompt or KV cache reuse tier is enabled."))
        }
        if let keyBits = cache.turboQuantKeyBits, !(2...8).contains(keyBits) {
            issues.append(.error(
                field: "cache.turboQuantKeyBits",
                message: "TurboQuant key bits must be between 2 and 8."))
        }
        if let valueBits = cache.turboQuantValueBits, !(2...8).contains(valueBits) {
            issues.append(.error(
                field: "cache.turboQuantValueBits",
                message: "TurboQuant value bits must be between 2 and 8."))
        }
        if let override = Self.nonEmptyOverride(tools.toolParserOverride),
           !Self.isNoopParserOverride(override),
           ToolCallFormat.fromCapabilityName(override) == nil {
            issues.append(.error(
                field: "tools.toolParserOverride",
                message: "Tool parser override is not a known parser alias."))
        }
        if let override = Self.nonEmptyOverride(tools.reasoningParserOverride),
           !Self.isNoopParserOverride(override),
           ReasoningParser.fromCapabilityName(override) == nil {
            issues.append(.error(
                field: "tools.reasoningParserOverride",
                message: "Reasoning parser override is not a known parser alias."))
        }
        if let draftTokenLimit = mtp.draftTokenLimit, draftTokenLimit <= 0 {
            issues.append(.error(
                field: "mtp.draftTokenLimit",
                message: "MTP draft token limit must be positive."))
        }
        if !mtp.keepDraftCacheSeparate {
            issues.append(.error(
                field: "mtp.keepDraftCacheSeparate",
                message: "Native MTP draft cache must stay separate from the verifier/base cache."))
        }
        if !mtp.acceptedTokensOnlyEnterBaseCache {
            issues.append(.error(
                field: "mtp.acceptedTokensOnlyEnterBaseCache",
                message: "Native MTP may commit only accepted verifier tokens to the base cache."))
        }
        if let defaultMaxKVSize = cache.defaultMaxKVSize, defaultMaxKVSize <= 0 {
            issues.append(.error(
                field: "cache.defaultMaxKVSize",
                message: "Default max KV size must be positive."))
        }
        if cache.longPromptMultiplier <= 0 {
            issues.append(.error(
                field: "cache.longPromptMultiplier",
                message: "Long-prompt multiplier must be positive."))
        }
        if mtp.mode == .forceOn {
            if let status = mtpStatus {
                if !status.canAutoLaunchMTP {
                    issues.append(.error(
                        field: "mtp.mode",
                        message: "MTP cannot be forced on until the bundle has complete tensor evidence and usable vmlx_mtp_tuning.json metadata for a supported native-MTP runtime."))
                }
            } else {
                issues.append(.warning(
                    field: "mtp.mode",
                    message: "MTP force-on was requested without a bundle status snapshot."))
            }
        }

        return issues
    }

    /// Validate settings with the same evidence used to resolve native-MTP
    /// launch policy.
    ///
    /// ``validationIssues(mtpStatus:)`` intentionally stays status-only for
    /// cheap UI checks. Osaurus should use this overload before launching a
    /// session when it has `config.json` bytes and optional `jang_config.json`
    /// metadata, because a bundle can have complete MTP tensors while a
    /// specific profile, such as Qwen3.6 JANG_2K, is still blocked by missing
    /// or unusable bundle-local tuning.
    public func validationIssues(
        configData: Data?,
        jangConfig: JangConfig?,
        mtpStatus: MTPBundleStatus?
    ) -> [VMLXServerSettingsIssue] {
        var issues = validationIssues(mtpStatus: mtpStatus)
        guard mtp.mode == .forceOn,
              !issues.contains(where: {
                  $0.severity == .error && $0.field == "mtp.mode"
              })
        else {
            return issues
        }

        let launch = resolvedMTPLaunch(
            configData: configData,
            jangConfig: jangConfig,
            status: mtpStatus)
        if launch.launchMode == .blocked {
            issues.append(.error(
                field: "mtp.mode",
                message: "MTP force-on is blocked for this bundle profile: \(launch.reason)"))
        }
        return issues
    }

    public func effectiveMTPLaunchMode(
        for status: MTPBundleStatus?
    ) -> VMLXMTPLaunchMode {
        switch mtp.mode {
        case .off:
            return .off
        case .auto:
            return (status?.canAutoLaunchMTP == true) ? .speculative : .off
        case .forceOn:
            return (status?.canAutoLaunchMTP == true) ? .speculative : .blocked
        }
    }

    /// Resolve a production MTP launch decision from the full model evidence.
    ///
    /// ``effectiveMTPLaunchMode(for:)`` is a status-only helper. Osaurus should
    /// use this method when it has the bundle's `config.json` bytes and optional
    /// JANG metadata so profile-specific depth blocks such as Qwen3.6 JANG_2K do
    /// not accidentally inherit a generic complete-tensor status.
    public func resolvedMTPLaunch(
        configData: Data?,
        jangConfig: JangConfig?,
        status: MTPBundleStatus?
    ) -> VMLXResolvedMTPLaunch {
        guard mtp.mode != .off else {
            return .init(launchMode: .off, recommendation: nil, reason: "MTP disabled by server settings.")
        }
        if let limit = mtp.draftTokenLimit, limit <= 0 {
            return .init(
                launchMode: .blocked,
                recommendation: nil,
                reason: "MTP draft token limit must be positive.")
        }

        guard let recommendation = NativeMTPAutoDecodePolicy.recommendation(
            configData: configData,
            jangConfig: jangConfig,
            status: status,
            requireVerifiedRuntime: true)
        else {
            let mode = mtp.mode == .forceOn ? VMLXMTPLaunchMode.blocked : .off
            return .init(
                launchMode: mode,
                recommendation: nil,
                reason: "No supported tensor-proven native-MTP recommendation for this bundle.")
        }

        let resolvedRecommendation: NativeMTPAutoDecodeRecommendation
        if let limit = mtp.draftTokenLimit, limit < recommendation.depth {
            resolvedRecommendation = NativeMTPAutoDecodeRecommendation(
                depth: limit,
                verifierMode: recommendation.verifierMode,
                reason: "\(recommendation.reason) Server draft-token limit capped depth from \(recommendation.depth) to \(limit).",
                evidence: recommendation.evidence + ["server_draft_token_limit=\(limit)"])
        } else {
            resolvedRecommendation = recommendation
        }

        return .init(
            launchMode: .speculative,
            recommendation: resolvedRecommendation,
            reason: resolvedRecommendation.reason)
    }

    /// Convenience bridge for request construction.
    ///
    /// Returns `nil` unless ``resolvedMTPLaunch(configData:jangConfig:status:)``
    /// returns `.speculative` with a concrete depth. This keeps native MTP out
    /// of raw batched paths unless the caller explicitly applies the returned
    /// strategy to an exclusive generate request.
    public func resolvedMTPDraftStrategy(
        configData: Data?,
        jangConfig: JangConfig?,
        status: MTPBundleStatus?
    ) -> DraftStrategy? {
        let launch = resolvedMTPLaunch(
            configData: configData,
            jangConfig: jangConfig,
            status: status)
        guard launch.launchMode == .speculative,
              let depth = launch.recommendation?.depth
        else {
            return nil
        }
        return .nativeMTP(depth: depth)
    }

    /// Resolve the load-time switch that preserves native-MTP sidecar weights.
    ///
    /// Native MTP needs two coordinated decisions: the model must be loaded with
    /// MTP tensors present, and generation must use ``DraftStrategy/nativeMTP``.
    /// Hosts should call this alongside ``resolvedMTPDraftStrategy`` so a real
    /// Qwen MTP bundle starts with the sidecar loaded, while non-MTP CRACK or
    /// metadata-only bundles keep the loader's MTP scrub path.
    public func resolvedLoadConfiguration(
        base: LoadConfiguration = .default,
        configData: Data?,
        jangConfig: JangConfig?,
        status: MTPBundleStatus?
    ) -> LoadConfiguration {
        var resolved = base
        resolved.nativeMTP = resolvedMTPLaunch(
            configData: configData,
            jangConfig: jangConfig,
            status: status).launchMode == .speculative
        return resolved
    }

    /// Resolve decode parameters for a server request.
    ///
    /// Merge order is intentionally narrow and auditable:
    /// 1. Start from the caller fallback.
    /// 2. Apply the bundle's `generation_config.json` values.
    /// 3. Apply only explicit server/UI overrides.
    ///
    /// Nil fields in ``generation`` mean "leave the bundle/default decision
    /// alone"; they never install hidden repetition penalties, temperature
    /// floors, top-p/top-k clamps, or family-specific rescue settings.
    public func resolvedGenerateParameters(
        generationConfig: GenerationConfigFile?,
        fallback: GenerateParameters = GenerateParameters()
    ) -> GenerateParameters {
        var resolved = GenerateParameters(
            generationConfig: generationConfig,
            fallback: fallback)
        if let maxTokens = generation.maxTokens {
            resolved.maxTokens = maxTokens
        }
        if let temperature = generation.temperature {
            resolved.temperature = Float(temperature)
        }
        if let topP = generation.topP {
            resolved.topP = Float(topP)
        }
        if let topK = generation.topK {
            resolved.topK = topK
        }
        if let minP = generation.minP {
            resolved.minP = Float(minP)
        }
        if let repetitionPenalty = generation.repetitionPenalty {
            resolved.repetitionPenalty = Float(repetitionPenalty)
        }
        return resolved
    }

    /// Validate a request shape against server policy and, when supplied, the
    /// model's capability snapshot.
    ///
    /// This is the request-time companion to ``validationIssues``. It prevents
    /// Osaurus plugin or gateway routes from bypassing server toggles such as
    /// VLM force-off, video/audio off, or native-MTP off, while reusing the same
    /// redacted capability-error JSON shape.
    public func validateRequest(
        _ request: ModelRuntimeCapabilityRequest,
        capabilitySnapshot: ModelRuntimeCapabilitySnapshot? = nil,
        unknownPolicy: ModelRuntimeCapabilityValidationPolicy = .rejectUnknown
    ) -> ModelRuntimeCapabilityValidationResult {
        var issues =
            capabilitySnapshot?
            .validate(request: request, unknownPolicy: unknownPolicy)
            .issues ?? []

        if multimodal.vlmMode == .forceOff {
            appendDisabled(
                [.vision, .video, .audio],
                requestedBy: request,
                field: "multimodal.vlmMode",
                to: &issues)
        } else {
            if !multimodal.enableVideo {
                appendDisabled(
                    [.video],
                    requestedBy: request,
                    field: "multimodal.enableVideo",
                    to: &issues)
            }
            if !multimodal.enableAudio {
                appendDisabled(
                    [.audio],
                    requestedBy: request,
                    field: "multimodal.enableAudio",
                    to: &issues)
            }
        }

        if mtp.mode == .off {
            appendDisabled(
                [.nativeMTP],
                requestedBy: request,
                field: "mtp.mode",
                to: &issues)
        }

        return ModelRuntimeCapabilityValidationResult(
            requestedModalities: request.sortedModalities,
            issues: issues)
    }

    /// Build the concrete cache coordinator configuration used by
    /// `BatchEngine`.
    ///
    /// This is the server-panel bridge for prefix/paged/L2-disk/SSM/TurboQuant
    /// KV settings. It performs no hidden quality rescue: `engine_selected` and
    /// `native` map to plain coordinator KV defaults, and TurboQuant KV is only
    /// selected when the caller explicitly supplies both bit widths.
    public func cacheCoordinatorConfig(
        modelKey: String? = nil,
        diskCacheDirectory: URL? = nil,
        ssmMaxEntries: Int = 50
    ) -> CacheCoordinatorConfig {
        let reuseEnabled = cache.prefix.enabled
        let diskEnabled: Bool
        let diskMaxSizeGB: Double?
        let diskDirectory: String?
        if reuseEnabled, cache.pagedKV.enabled {
            diskEnabled = cache.blockDisk.enabled
            diskMaxSizeGB = cache.blockDisk.maxSizeGB
            diskDirectory = cache.blockDisk.directory
        } else if reuseEnabled {
            diskEnabled = cache.legacyDisk.enabled
            diskMaxSizeGB = cache.legacyDisk.maxSizeGB
            diskDirectory = cache.legacyDisk.directory
        } else {
            diskEnabled = false
            diskMaxSizeGB = nil
            diskDirectory = nil
        }
        let diskDir = diskCacheDirectory
            ?? VMLXServerRuntimeSettings.resolvedDirectory(diskDirectory)
        let diskMaxGB = Float(diskMaxSizeGB ?? 10.0)

        return CacheCoordinatorConfig(
            usePagedCache: reuseEnabled && cache.pagedKV.enabled,
            enableDiskCache: diskEnabled,
            pagedBlockSize: cache.pagedKV.blockSize ?? 64,
            maxCacheBlocks: cache.pagedKV.maxBlocks ?? 1000,
            diskCacheMaxGB: diskMaxGB,
            diskCacheDir: diskDir,
            ssmMaxEntries: ssmMaxEntries,
            enableSSMReDerive: cache.enableSSMReDerive,
            modelKey: modelKey,
            defaultKVMode: cache.defaultKVMode,
            defaultMaxKVSize: cache.defaultMaxKVSize,
            longPromptMultiplier: cache.longPromptMultiplier)
    }

    private static func resolvedDirectory(_ path: String?) -> URL? {
        guard let path, !path.isEmpty else { return nil }
        if path == "~" {
            return FileManager.default.homeDirectoryForCurrentUser
        }
        if path.hasPrefix("~/") {
            let suffix = String(path.dropFirst(2))
            return FileManager.default.homeDirectoryForCurrentUser
                .appendingPathComponent(suffix)
        }
        return URL(fileURLWithPath: path)
    }

    private static func nonEmptyOverride(_ value: String?) -> String? {
        let trimmed = value?.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let trimmed, !trimmed.isEmpty else { return nil }
        return trimmed
    }

    private static func isNoopParserOverride(_ value: String) -> Bool {
        switch value.lowercased().replacingOccurrences(of: "-", with: "_") {
        case "auto", "none", "off", "disabled":
            true
        default:
            false
        }
    }

    private func appendDisabled(
        _ disabled: [ModelRuntimeRequestModality],
        requestedBy request: ModelRuntimeCapabilityRequest,
        field: String,
        to issues: inout [ModelRuntimeCapabilityIssue]
    ) {
        for modality in disabled where request.modalities.contains(modality) {
            issues.append(.disabledByServerSettings(modality: modality, field: field))
        }
    }
}

public struct VMLXServerNetworkSettings: Codable, Sendable, Equatable {
    public var host: String
    public var port: Int?
    public var apiKey: String?
    public var servedModelName: String?
    public var rateLimitRequestsPerMinute: Int?
    public var timeoutSeconds: Int?
    public var logLevel: VMLXServerLogLevel
    public var corsOrigins: [String]

    public init(
        host: String = "127.0.0.1",
        port: Int? = nil,
        apiKey: String? = nil,
        servedModelName: String? = nil,
        rateLimitRequestsPerMinute: Int? = nil,
        timeoutSeconds: Int? = nil,
        logLevel: VMLXServerLogLevel = .info,
        corsOrigins: [String] = ["*"]
    ) {
        self.host = host
        self.port = port
        self.apiKey = apiKey
        self.servedModelName = servedModelName
        self.rateLimitRequestsPerMinute = rateLimitRequestsPerMinute
        self.timeoutSeconds = timeoutSeconds
        self.logLevel = logLevel
        self.corsOrigins = corsOrigins
    }
}

public enum VMLXServerLogLevel: String, Codable, Sendable, Equatable, CaseIterable {
    case trace
    case debug
    case info
    case warning
    case error
}

public struct VMLXServerConcurrencySettings: Codable, Sendable, Equatable {
    public var maxConcurrentSequences: Int?
    public var prefillBatchSize: Int?
    public var prefillStepSize: Int?
    public var completionBatchSize: Int?
    public var continuousBatching: Bool
    public var smeltMode: VMLXServerSmeltMode

    public init(
        maxConcurrentSequences: Int? = nil,
        prefillBatchSize: Int? = nil,
        prefillStepSize: Int? = nil,
        completionBatchSize: Int? = nil,
        continuousBatching: Bool = true,
        smeltMode: VMLXServerSmeltMode = .engineSelected
    ) {
        self.maxConcurrentSequences = maxConcurrentSequences
        self.prefillBatchSize = prefillBatchSize
        self.prefillStepSize = prefillStepSize
        self.completionBatchSize = completionBatchSize
        self.continuousBatching = continuousBatching
        self.smeltMode = smeltMode
    }
}

public enum VMLXServerSmeltMode: String, Codable, Sendable, Equatable, CaseIterable {
    case engineSelected = "engine_selected"
    case disabled
    case flashMoE = "flash_moe"
    case ssdStreaming = "ssd_streaming"
}

public struct VMLXServerCacheSettings: Codable, Sendable, Equatable {
    public var prefix: VMLXPrefixCacheSettings
    public var pagedKV: VMLXPagedKVCacheSettings
    public var liveKVCodec: VMLXKVCacheCodec
    public var turboQuantKeyBits: Int?
    public var turboQuantValueBits: Int?
    public var defaultMaxKVSize: Int?
    public var longPromptMultiplier: Double
    public var storedKVCodec: VMLXStoredKVCacheCodec
    public var legacyDisk: VMLXDiskCacheSettings
    public var blockDisk: VMLXBlockDiskCacheSettings
    public var enableSSMReDerive: Bool

    public init(
        prefix: VMLXPrefixCacheSettings = .init(),
        pagedKV: VMLXPagedKVCacheSettings = .init(),
        liveKVCodec: VMLXKVCacheCodec = .engineSelected,
        turboQuantKeyBits: Int? = nil,
        turboQuantValueBits: Int? = nil,
        defaultMaxKVSize: Int? = nil,
        longPromptMultiplier: Double = 2.0,
        storedKVCodec: VMLXStoredKVCacheCodec = .auto,
        legacyDisk: VMLXDiskCacheSettings = .init(),
        blockDisk: VMLXBlockDiskCacheSettings = .init(),
        enableSSMReDerive: Bool = true
    ) {
        self.prefix = prefix
        self.pagedKV = pagedKV
        self.liveKVCodec = liveKVCodec
        self.turboQuantKeyBits = turboQuantKeyBits
        self.turboQuantValueBits = turboQuantValueBits
        self.defaultMaxKVSize = defaultMaxKVSize
        self.longPromptMultiplier = longPromptMultiplier
        self.storedKVCodec = storedKVCodec
        self.legacyDisk = legacyDisk
        self.blockDisk = blockDisk
        self.enableSSMReDerive = enableSSMReDerive
    }

    public var defaultKVMode: KVQuantizationMode {
        guard liveKVCodec == .turboQuant,
              let keyBits = turboQuantKeyBits,
              let valueBits = turboQuantValueBits
        else {
            return .none
        }
        return .turboQuant(keyBits: keyBits, valueBits: valueBits)
    }
}

public struct VMLXPrefixCacheSettings: Codable, Sendable, Equatable {
    public var enabled: Bool
    public var legacyEntryCountCache: Bool
    public var memoryLimitMB: Int?
    public var memoryPercent: Double?
    public var ttlMinutes: Int?

    public init(
        enabled: Bool = true,
        legacyEntryCountCache: Bool = false,
        memoryLimitMB: Int? = nil,
        memoryPercent: Double? = 15,
        ttlMinutes: Int? = nil
    ) {
        self.enabled = enabled
        self.legacyEntryCountCache = legacyEntryCountCache
        self.memoryLimitMB = memoryLimitMB
        self.memoryPercent = memoryPercent
        self.ttlMinutes = ttlMinutes
    }
}

public struct VMLXPagedKVCacheSettings: Codable, Sendable, Equatable {
    public var enabled: Bool
    public var blockSize: Int?
    public var maxBlocks: Int?

    public init(enabled: Bool = true, blockSize: Int? = nil, maxBlocks: Int? = nil) {
        self.enabled = enabled
        self.blockSize = blockSize
        self.maxBlocks = maxBlocks
    }
}

public enum VMLXKVCacheCodec: String, Codable, Sendable, Equatable, CaseIterable {
    case engineSelected = "engine_selected"
    case native
    case turboQuant = "turboquant"
    case none
}

public enum VMLXStoredKVCacheCodec: String, Codable, Sendable, Equatable, CaseIterable {
    case auto
    case native
    case turboQuant = "turboquant"
    case disabled
}

public struct VMLXDiskCacheSettings: Codable, Sendable, Equatable {
    public var enabled: Bool
    public var maxSizeGB: Double?
    public var directory: String?

    public init(
        enabled: Bool = false,
        maxSizeGB: Double? = nil,
        directory: String? = "~/.cache/vmlx-engine/prompt-cache"
    ) {
        self.enabled = enabled
        self.maxSizeGB = maxSizeGB
        self.directory = directory
    }
}

public struct VMLXBlockDiskCacheSettings: Codable, Sendable, Equatable {
    public var enabled: Bool
    public var maxSizeGB: Double?
    public var directory: String?

    public init(enabled: Bool = true, maxSizeGB: Double? = nil, directory: String? = nil) {
        self.enabled = enabled
        self.maxSizeGB = maxSizeGB
        self.directory = directory
    }
}

public struct VMLXServerPowerSettings: Codable, Sendable, Equatable {
    public var autoSleepEnabled: Bool
    public var lightSleepAfterSeconds: Int?
    public var deepSleepAfterSeconds: Int?
    public var wakeOnRequest: Bool
    public var jitLoad: Bool

    public init(
        autoSleepEnabled: Bool = false,
        lightSleepAfterSeconds: Int? = nil,
        deepSleepAfterSeconds: Int? = nil,
        wakeOnRequest: Bool = true,
        jitLoad: Bool = true
    ) {
        self.autoSleepEnabled = autoSleepEnabled
        self.lightSleepAfterSeconds = lightSleepAfterSeconds
        self.deepSleepAfterSeconds = deepSleepAfterSeconds
        self.wakeOnRequest = wakeOnRequest
        self.jitLoad = jitLoad
    }
}

public struct VMLXServerGenerationDefaults: Codable, Sendable, Equatable {
    public var streamInterval: Int?
    public var maxTokens: Int?
    public var temperature: Double?
    public var topP: Double?
    public var topK: Int?
    public var minP: Double?
    public var repetitionPenalty: Double?

    public init(
        streamInterval: Int? = 1,
        maxTokens: Int? = nil,
        temperature: Double? = nil,
        topP: Double? = nil,
        topK: Int? = nil,
        minP: Double? = nil,
        repetitionPenalty: Double? = nil
    ) {
        self.streamInterval = streamInterval
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.minP = minP
        self.repetitionPenalty = repetitionPenalty
    }
}

public struct VMLXServerToolSettings: Codable, Sendable, Equatable {
    public var mcpConfigFile: String?
    public var enableAutoToolChoice: Bool
    public var toolParserOverride: String?
    public var reasoningParserOverride: String?
    public var customChatTemplate: String?

    public init(
        mcpConfigFile: String? = nil,
        enableAutoToolChoice: Bool = false,
        toolParserOverride: String? = nil,
        reasoningParserOverride: String? = nil,
        customChatTemplate: String? = nil
    ) {
        self.mcpConfigFile = mcpConfigFile
        self.enableAutoToolChoice = enableAutoToolChoice
        self.toolParserOverride = toolParserOverride
        self.reasoningParserOverride = reasoningParserOverride
        self.customChatTemplate = customChatTemplate
    }
}

public struct VMLXServerMultimodalSettings: Codable, Sendable, Equatable {
    public var vlmMode: VMLXVLMServerMode
    public var requireMediaSaltForCache: Bool
    public var enableVideo: Bool
    public var enableAudio: Bool

    public init(
        vlmMode: VMLXVLMServerMode = .auto,
        requireMediaSaltForCache: Bool = true,
        enableVideo: Bool = true,
        enableAudio: Bool = true
    ) {
        self.vlmMode = vlmMode
        self.requireMediaSaltForCache = requireMediaSaltForCache
        self.enableVideo = enableVideo
        self.enableAudio = enableAudio
    }
}

public enum VMLXVLMServerMode: String, Codable, Sendable, Equatable, CaseIterable {
    case auto
    case forceOff = "force_off"
    case forceOn = "force_on"
}

public struct VMLXServerMTPSettings: Codable, Sendable, Equatable {
    public var mode: VMLXMTPServerMode
    public var draftTokenLimit: Int?
    public var keepDraftCacheSeparate: Bool
    public var acceptedTokensOnlyEnterBaseCache: Bool

    public init(
        mode: VMLXMTPServerMode = .off,
        draftTokenLimit: Int? = nil,
        keepDraftCacheSeparate: Bool = true,
        acceptedTokensOnlyEnterBaseCache: Bool = true
    ) {
        self.mode = mode
        self.draftTokenLimit = draftTokenLimit
        self.keepDraftCacheSeparate = keepDraftCacheSeparate
        self.acceptedTokensOnlyEnterBaseCache = acceptedTokensOnlyEnterBaseCache
    }
}

public enum VMLXMTPServerMode: String, Codable, Sendable, Equatable, CaseIterable {
    case auto
    case off
    case forceOn = "force_on"
}

public enum VMLXMTPLaunchMode: String, Codable, Sendable, Equatable, CaseIterable {
    case off
    case speculative
    case blocked
}

public struct VMLXResolvedMTPLaunch: Codable, Sendable, Equatable {
    public var launchMode: VMLXMTPLaunchMode
    public var recommendation: NativeMTPAutoDecodeRecommendation?
    public var reason: String

    public init(
        launchMode: VMLXMTPLaunchMode,
        recommendation: NativeMTPAutoDecodeRecommendation?,
        reason: String
    ) {
        self.launchMode = launchMode
        self.recommendation = recommendation
        self.reason = reason
    }
}

public struct VMLXServerSettingsIssue: Codable, Sendable, Equatable {
    public enum Severity: String, Codable, Sendable, Equatable {
        case warning
        case error
    }

    public var severity: Severity
    public var field: String
    public var message: String

    public init(severity: Severity, field: String, message: String) {
        self.severity = severity
        self.field = field
        self.message = message
    }

    public static func warning(field: String, message: String) -> VMLXServerSettingsIssue {
        .init(severity: .warning, field: field, message: message)
    }

    public static func error(field: String, message: String) -> VMLXServerSettingsIssue {
        .init(severity: .error, field: field, message: message)
    }
}
