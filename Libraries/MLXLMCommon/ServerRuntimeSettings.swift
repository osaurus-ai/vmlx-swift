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
    public var memorySafety: VMLXMemorySafetySettings
    /// Optional so existing server-runtime.json files decode unchanged.
    /// `effectivePerformance` resolves nil to defaults.
    public var performance: VMLXServerPerformanceSettings?

    public var effectivePerformance: VMLXServerPerformanceSettings {
        performance ?? .init()
    }

    public init(
        network: VMLXServerNetworkSettings = .init(),
        concurrency: VMLXServerConcurrencySettings = .init(),
        cache: VMLXServerCacheSettings = .init(),
        power: VMLXServerPowerSettings = .init(),
        generation: VMLXServerGenerationDefaults = .init(),
        tools: VMLXServerToolSettings = .init(),
        multimodal: VMLXServerMultimodalSettings = .init(),
        mtp: VMLXServerMTPSettings = .init(),
        memorySafety: VMLXMemorySafetySettings = .init(),
        performance: VMLXServerPerformanceSettings? = nil
    ) {
        self.network = network
        self.concurrency = concurrency
        self.cache = cache
        self.power = power
        self.generation = generation
        self.tools = tools
        self.multimodal = multimodal
        self.mtp = mtp
        self.memorySafety = memorySafety
        self.performance = performance
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
        if let diffusionSteps = generation.diffusionMaxDenoisingSteps {
            if diffusionSteps < 1 {
                issues.append(.error(
                    field: "generation.diffusionMaxDenoisingSteps",
                    message: "Diffusion denoising steps must be at least 1. Use nil for the bundle default."))
            } else if diffusionSteps < 12 {
                issues.append(.warning(
                    field: "generation.diffusionMaxDenoisingSteps",
                    message: "Diffusion budgets below 12 steps measurably break coherency on diffusiongemma-26B-A4B (8 steps produces word-salad spans)."))
            }
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
        issues.append(contentsOf: memorySafety.validationIssues())

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
            let reason = NativeMTPAutoDecodePolicy.rejectionReason(
                configData: configData,
                jangConfig: jangConfig,
                status: status,
                requireVerifiedRuntime: true)
                ?? "No supported tensor-proven native-MTP recommendation for this bundle."
            return .init(
                launchMode: mode,
                recommendation: nil,
                reason: reason)
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
        return .nativeMTP(depth: depth, verifierMode: launch.recommendation?.verifierMode)
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

    /// Resolve the single memory-safety slider into concrete engine knobs.
    ///
    /// This helper is intentionally policy-only. It does not inspect Activity
    /// Monitor free memory, does not alter generation defaults, and does not
    /// hide runtime bugs with sampler/template changes. Strict modes may return
    /// typed blocking issues before load/request execution; non-strict modes
    /// return warnings so hosts can show the risk while still allowing an
    /// explicit user launch.
    public func resolvedMemorySafetyPlan(
        baseLoadConfiguration: LoadConfiguration = .default,
        bundleFacts: LoadBundleFacts? = nil,
        host: MemoryStatus? = nil,
        request: VMLXMemoryRequestEstimate? = nil
    ) -> VMLXResolvedMemorySafetyPlan {
        let profile = memorySafety.profile
        let physicalMemory = host?.physicalMemory
            ?? bundleFacts?.physicalMemory
            ?? ProcessInfo.processInfo.physicalMemory
        let requestedFraction = memorySafety.customPhysicalMemoryFraction
            ?? profile.loadFraction
        let loadCap = ResidentCap.fraction(requestedFraction)
        let allocatorCap = memorySafety.customAllocatorCacheBytes.map(ResidentCap.absolute)
            ?? profile.allocatorCap

        var loadConfiguration = baseLoadConfiguration
        loadConfiguration.memoryLimit = loadCap
        loadConfiguration.maxResidentBytes = allocatorCap
        loadConfiguration.useMmapSafetensors = true
        loadConfiguration.jangPress = resolvedMemorySafetyJangPress(
            base: baseLoadConfiguration.jangPress,
            facts: bundleFacts)


        var resolvedConcurrency = concurrency
        resolvedConcurrency.maxConcurrentSequences =
            memorySafety.customMaxConcurrentSequences
            ?? resolvedConcurrency.maxConcurrentSequences
            ?? profile.maxConcurrentSequences

        var resolvedCache = cache
        if resolvedCache.prefix.enabled {
            resolvedCache.blockDisk.enabled = true
            resolvedCache.legacyDisk.enabled = false
            if resolvedCache.prefix.memoryLimitMB == nil {
                resolvedCache.prefix.memoryLimitMB = profile.prefixMemoryLimitMB
            }
            if resolvedCache.prefix.memoryPercent == nil {
                resolvedCache.prefix.memoryPercent = profile.prefixMemoryPercent
            }
        } else {
            resolvedCache.pagedKV.enabled = false
            resolvedCache.blockDisk.enabled = false
            resolvedCache.legacyDisk.enabled = false
        }
        resolvedCache.defaultMaxKVSize =
            memorySafety.customDefaultMaxKVSize
            ?? resolvedCache.defaultMaxKVSize
            ?? profile.defaultMaxKVSize

        var warnings = profile.warnings
        // Near-RAM-scale packs must materialize, not mmap. File-backed
        // weights are "reclaimable under pressure" — for a bundle whose
        // weights approach physical memory (Hy3-JANG_2K: 94 GB on 128 GB),
        // macOS cannot keep the full mapping hot next to KV, apps, and the
        // file cache itself, so decode refaults experts from SSD on every
        // step: the model answers a 10-token probe in ~a minute, times out
        // on real turns, and never shows up in RAM — reported live as "it
        // attempted to load and then nothing happened". Anonymous
        // (materialized) pages are not reclaimed that way, and a pack this
        // size fits the working set (87.6 GiB weights vs a ~96 GB budget).
        // The load fraction is raised to cover the weights plus headroom;
        // the load entry clamps it to the GPU working set, so an impossible
        // limit cannot be constructed here.
        if profile.allowsNearRAMScaleMaterialization,
            memorySafety.customPhysicalMemoryFraction == nil,
            let facts = bundleFacts,
            facts.totalSafetensorsBytes > 0,
            physicalMemory > 0,
            Double(facts.totalSafetensorsBytes) > 0.55 * Double(physicalMemory),
            // Only when the weights actually fit. A pack larger than ~86%
            // of RAM (e.g. 30 GiB on a 24 GiB host) cannot be made resident
            // at all — mmap streaming is its only viable mode, and
            // materializing it would push the host into swap/jetsam.
            Double(facts.totalSafetensorsBytes) <= 0.86 * Double(physicalMemory)
        {
            loadConfiguration.useMmapSafetensors = false
            let needFraction = min(
                0.92,
                Double(facts.totalSafetensorsBytes) / Double(physicalMemory) + 0.06)
            if needFraction > requestedFraction {
                loadConfiguration.memoryLimit = .fraction(needFraction)
            }
            warnings.append(
                "Weights (\(facts.totalSafetensorsBytes / 1_073_741_824) GiB) approach physical memory; loading materialized instead of mmap so pages stay resident."
            )
        }
        var blockingIssues: [VMLXServerSettingsIssue] = []

        if case .diagnosticDangerous = memorySafety.mode {
            warnings.append("Diagnostic memory mode uses caller-supplied limits and may exceed the host working set.")
        }
        if !memorySafety.allowExperimentalMLXPress,
           baseLoadConfiguration.jangPress != .disabled {
            warnings.append("MLXPress/JangPress was not selected by the memory slider. It remains disabled unless the host enables a proven routed-bundle lane explicitly.")
        }

        let resolvedBudgetBytes = loadCap.resolve(physicalMemory: physicalMemory)
        if let estimate = request {
            if estimate.workingSetBytes == nil,
               memorySafety.failClosedWhenEstimateUnknown || profile.failClosedWhenEstimateUnknown {
                blockingIssues.append(.error(
                    field: "memorySafety.requestEstimate",
                    message: "Strict memory safety requires a request working-set estimate before launch."))
            }
            if let budget = resolvedBudgetBytes,
               let estimateBytes = estimate.workingSetBytes,
               estimateBytes > budget {
                var message = "Estimated request working set \(estimateBytes) bytes exceeds resolved memory budget \(budget) bytes."
                if cache.liveKVCodec != .turboQuant {
                    message += " To fit without truncating context, enable TurboQuant KV cache (3-bit, ~5\u{00D7} smaller KV) in the cache settings, or reduce the context length / lower the memory-safety slider."
                } else {
                    message += " TurboQuant KV is already enabled; reduce the context length or lower the memory-safety slider to fit."
                }
                if profile.blocksOverBudget {
                    blockingIssues.append(.error(
                        field: "memorySafety.requestEstimate",
                        message: message))
                } else {
                    warnings.append(message)
                }
            }
        }

        let displaySummary = "mode=\(memorySafety.mode.rawValue) slider=\(memorySafety.slider) load_cap=\(requestedFraction) allocator_cap=\(allocatorCap.displayValue) max_concurrent=\(resolvedConcurrency.maxConcurrentSequences ?? 0) kv_cap=\(resolvedCache.defaultMaxKVSize ?? 0)"

        return VMLXResolvedMemorySafetyPlan(
            loadConfiguration: loadConfiguration,
            cache: resolvedCache,
            concurrency: resolvedConcurrency,
            resolvedPhysicalMemoryBytes: physicalMemory,
            resolvedLoadBudgetBytes: resolvedBudgetBytes,
            warnings: warnings,
            blockingIssues: blockingIssues,
            displaySummary: displaySummary)
    }

    private func resolvedMemorySafetyJangPress(
        base: JangPressPolicy,
        facts: LoadBundleFacts?
    ) -> JangPressPolicy {
        guard memorySafety.allowExperimentalMLXPress else {
            return .disabled
        }
        guard let facts,
              facts.isRouted,
              facts.hasJangConfig || facts.hasJangTQRuntime
        else {
            return .disabled
        }
        return base
    }

    /// Apply explicit server-panel parser overrides to a model configuration.
    ///
    /// Factories still own normal auto-detection from `jang_config.json` and
    /// `config.json.model_type`. This helper only mutates fields when the host
    /// has persisted a concrete override; "auto" keeps the factory result, and
    /// "none"/"off"/"disabled" disables reasoning parsing by stamping `none`.
    public func resolvedModelConfiguration(
        base: ModelConfiguration
    ) -> ModelConfiguration {
        var resolved = base
        if let override = Self.nonEmptyOverride(tools.toolParserOverride),
           !Self.isNoopParserOverride(override),
           let format = ToolCallFormat.fromCapabilityName(override) {
            resolved.toolCallFormat = format
        }
        if let override = Self.nonEmptyOverride(tools.reasoningParserOverride) {
            let normalized = override.lowercased().replacingOccurrences(of: "-", with: "_")
            switch normalized {
            case "auto":
                break
            case "none", "off", "disabled":
                resolved.reasoningParserName = "none"
            default:
                if ReasoningParser.fromCapabilityName(override) != nil {
                    resolved.reasoningParserName = override
                }
            }
        }
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
    /// KV settings. It performs no hidden quality rescue: `engine_selected`
    /// chooses the engine's production KV codec, `native` and `none` preserve
    /// float/native cache behavior, and explicit TurboQuant still requires
    /// caller-supplied bit widths.
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
        } else if reuseEnabled, cache.blockDisk.enabled {
            diskEnabled = true
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
        switch liveKVCodec {
        case .engineSelected:
            return .turboQuant()
        case .native, .none:
            return .none
        case .turboQuant:
            guard let keyBits = turboQuantKeyBits,
                  let valueBits = turboQuantValueBits
            else {
                return .none
            }
            return .turboQuant(keyBits: keyBits, valueBits: valueBits)
        }
    }
}

/// Decode-throughput settings the host wires to its server settings panel.
///
/// Optional on ``VMLXServerRuntimeSettings`` so pre-existing
/// `server-runtime.json` files decode unchanged (nil = all defaults).
public struct VMLXServerPerformanceSettings: Codable, Sendable, Equatable {
    /// Load-time codec for tied LM heads that ship UNQUANTIZED in an
    /// otherwise-quantized bundle (Gemma4 QAT ships the 262k-vocab tied
    /// embedding fp16; `asLinear` then streams the full fp16 table per
    /// decoded token — ~1.07 GB/token on E2B). llama.cpp ships the
    /// equivalent GGUF output head quantized (Q6_K-class), which is the
    /// documented baseline this engine is compared against.
    ///
    /// `fp16Passthrough` (default) changes nothing. Quantized codecs are
    /// applied only when the bundle itself is quantized AND the head has
    /// no quantization sidecars of its own — a pre-quantized head always
    /// loads as shipped.
    public var tiedHeadCodec: VMLXTiedHeadCodec
    /// Experimental MLX compiled decode. Measured +25% decode on Gemma4
    /// E2B QAT (132.5 -> 165.3 tok/s, M5 Max, 2026-06-12). Off by
    /// default pending the PR #1173 model-switch corruption root cause;
    /// hosts surface this as an explicit experimental toggle.
    public var compiledDecode: Bool

    public init(
        tiedHeadCodec: VMLXTiedHeadCodec = .fp16Passthrough,
        compiledDecode: Bool = false
    ) {
        self.tiedHeadCodec = tiedHeadCodec
        self.compiledDecode = compiledDecode
    }
}

public enum VMLXTiedHeadCodec: String, Codable, Sendable, Equatable, CaseIterable {
    /// Load the head exactly as shipped (default).
    case fp16Passthrough = "fp16_passthrough"
    /// Quantize an unquantized tied head at load: 8-bit affine, gs=64.
    case q8
    /// 6-bit affine, gs=64 — bandwidth/quality point closest to the
    /// llama.cpp Q6_K output head used by the documented GGUF baselines.
    case q6
    /// 4-bit affine, gs=64 — fastest, largest numeric delta.
    case q4

    public var quantization: (bits: Int, groupSize: Int)? {
        switch self {
        case .fp16Passthrough: return nil
        case .q8: return (8, 64)
        case .q6: return (6, 64)
        case .q4: return (4, 64)
        }
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

    public init(enabled: Bool = false, blockSize: Int? = nil, maxBlocks: Int? = nil) {
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

    /// Block-diffusion speed/quality budget (denoising steps per canvas)
    /// applied to ``BlockDiffusionModel`` generation via
    /// `GenerateParameters.diffusionMaxDenoisingSteps`. `nil` keeps the
    /// bundle's `generation_config.json` value. Measured on
    /// diffusiongemma-26B-A4B MXFP4 (M5 Max): 48 ≈ 37 tok/s,
    /// 16 ≈ 74 tok/s still coherent, 8 breaks coherency. Ignored by
    /// autoregressive models.
    public var diffusionMaxDenoisingSteps: Int?

    public init(
        streamInterval: Int? = 1,
        maxTokens: Int? = nil,
        temperature: Double? = nil,
        topP: Double? = nil,
        topK: Int? = nil,
        minP: Double? = nil,
        repetitionPenalty: Double? = nil,
        diffusionMaxDenoisingSteps: Int? = nil
    ) {
        self.streamInterval = streamInterval
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.minP = minP
        self.repetitionPenalty = repetitionPenalty
        self.diffusionMaxDenoisingSteps = diffusionMaxDenoisingSteps
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
        mode: VMLXMTPServerMode = .auto,
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

public enum VMLXMemorySafetyMode: String, Codable, Sendable, Equatable, CaseIterable {
    case performance
    case balanced
    case safeAuto = "safe_auto"
    case strict
    case diagnosticDangerous = "diagnostic_dangerous"
}

public struct VMLXMemorySafetySettings: Codable, Sendable, Equatable {
    public var mode: VMLXMemorySafetyMode
    public var slider: Int
    public var allowExperimentalMLXPress: Bool
    public var failClosedWhenEstimateUnknown: Bool
    public var customPhysicalMemoryFraction: Double?
    public var customAllocatorCacheBytes: UInt64?
    public var customDefaultMaxKVSize: Int?
    public var customMaxConcurrentSequences: Int?

    public init(
        mode: VMLXMemorySafetyMode = .safeAuto,
        slider: Int = 2,
        allowExperimentalMLXPress: Bool = false,
        failClosedWhenEstimateUnknown: Bool = false,
        customPhysicalMemoryFraction: Double? = nil,
        customAllocatorCacheBytes: UInt64? = nil,
        customDefaultMaxKVSize: Int? = nil,
        customMaxConcurrentSequences: Int? = nil
    ) {
        self.mode = mode
        self.slider = slider
        self.allowExperimentalMLXPress = allowExperimentalMLXPress
        self.failClosedWhenEstimateUnknown = failClosedWhenEstimateUnknown
        self.customPhysicalMemoryFraction = customPhysicalMemoryFraction
        self.customAllocatorCacheBytes = customAllocatorCacheBytes
        self.customDefaultMaxKVSize = customDefaultMaxKVSize
        self.customMaxConcurrentSequences = customMaxConcurrentSequences
    }

    public func validationIssues() -> [VMLXServerSettingsIssue] {
        var issues: [VMLXServerSettingsIssue] = []
        if !(0...4).contains(slider) {
            issues.append(.error(
                field: "memorySafety.slider",
                message: "Memory safety slider must be between 0 and 4."))
        }
        if let fraction = customPhysicalMemoryFraction,
           fraction <= 0 || fraction > 1 {
            issues.append(.error(
                field: "memorySafety.customPhysicalMemoryFraction",
                message: "Custom physical-memory fraction must be greater than 0 and at most 1."))
        }
        if let bytes = customAllocatorCacheBytes, bytes == 0 {
            issues.append(.error(
                field: "memorySafety.customAllocatorCacheBytes",
                message: "Custom allocator cache bytes must be positive."))
        }
        if let kv = customDefaultMaxKVSize, kv <= 0 {
            issues.append(.error(
                field: "memorySafety.customDefaultMaxKVSize",
                message: "Custom max KV size must be positive."))
        }
        if let maxConcurrent = customMaxConcurrentSequences, maxConcurrent <= 0 {
            issues.append(.error(
                field: "memorySafety.customMaxConcurrentSequences",
                message: "Custom max concurrent sequences must be positive."))
        }
        return issues
    }

    fileprivate var profile: VMLXMemorySafetyProfile {
        switch mode {
        case .performance:
            return .init(
                loadFraction: customPhysicalMemoryFraction ?? 0.90,
                allowsNearRAMScaleMaterialization: true,
                allocatorCap: customAllocatorCacheBytes.map(ResidentCap.absolute) ?? .unlimited,
                maxConcurrentSequences: customMaxConcurrentSequences ?? 2,
                prefixMemoryLimitMB: nil,
                prefixMemoryPercent: 20,
                defaultMaxKVSize: customDefaultMaxKVSize ?? 131072,
                failClosedWhenEstimateUnknown: false,
                blocksOverBudget: false,
                warnings: ["Performance memory mode may allow macOS compression or swap before refusing a request."])
        case .balanced:
            return .init(
                loadFraction: customPhysicalMemoryFraction ?? 0.75,
                allowsNearRAMScaleMaterialization: true,
                allocatorCap: customAllocatorCacheBytes.map(ResidentCap.absolute) ?? .absolute(1 << 30),
                maxConcurrentSequences: customMaxConcurrentSequences ?? 2,
                prefixMemoryLimitMB: 512,
                prefixMemoryPercent: 15,
                defaultMaxKVSize: customDefaultMaxKVSize ?? 65536,
                failClosedWhenEstimateUnknown: false,
                blocksOverBudget: false,
                warnings: [])
        case .safeAuto:
            return .init(
                loadFraction: customPhysicalMemoryFraction ?? 0.70,
                allowsNearRAMScaleMaterialization: true,
                allocatorCap: customAllocatorCacheBytes.map(ResidentCap.absolute) ?? .absolute(128 << 20),
                maxConcurrentSequences: customMaxConcurrentSequences ?? 1,
                prefixMemoryLimitMB: 128,
                prefixMemoryPercent: 15,
                defaultMaxKVSize: customDefaultMaxKVSize ?? 65536,
                failClosedWhenEstimateUnknown: false,
                blocksOverBudget: false,
                warnings: [])
        case .strict:
            return .init(
                loadFraction: customPhysicalMemoryFraction ?? 0.60,
                allocatorCap: customAllocatorCacheBytes.map(ResidentCap.absolute) ?? .absolute(128 << 20),
                maxConcurrentSequences: customMaxConcurrentSequences ?? 1,
                prefixMemoryLimitMB: 128,
                prefixMemoryPercent: 10,
                defaultMaxKVSize: customDefaultMaxKVSize ?? 16384,
                failClosedWhenEstimateUnknown: true,
                blocksOverBudget: true,
                warnings: [])
        case .diagnosticDangerous:
            return .init(
                loadFraction: customPhysicalMemoryFraction ?? 1.0,
                allocatorCap: customAllocatorCacheBytes.map(ResidentCap.absolute) ?? .unlimited,
                maxConcurrentSequences: customMaxConcurrentSequences ?? 1,
                prefixMemoryLimitMB: nil,
                prefixMemoryPercent: nil,
                defaultMaxKVSize: customDefaultMaxKVSize,
                failClosedWhenEstimateUnknown: failClosedWhenEstimateUnknown,
                blocksOverBudget: false,
                warnings: [])
        }
    }
}

fileprivate struct VMLXMemorySafetyProfile: Sendable, Equatable {
    var loadFraction: Double
    /// Whether this profile may switch a near-RAM-scale bundle from mmap to
    /// a materialized load (and raise the load fraction to cover it). The
    /// strict profile keeps its explicit budget authoritative, and the
    /// diagnostic mode uses caller-supplied limits verbatim.
    var allowsNearRAMScaleMaterialization: Bool = false
    var allocatorCap: ResidentCap
    var maxConcurrentSequences: Int
    var prefixMemoryLimitMB: Int?
    var prefixMemoryPercent: Double?
    var defaultMaxKVSize: Int?
    var failClosedWhenEstimateUnknown: Bool
    var blocksOverBudget: Bool
    var warnings: [String]
}

public struct VMLXMemoryRequestEstimate: Sendable, Equatable {
    public var workingSetBytes: UInt64?
    public var promptTokens: Int?
    public var maxNewTokens: Int?

    public init(
        workingSetBytes: UInt64? = nil,
        promptTokens: Int? = nil,
        maxNewTokens: Int? = nil
    ) {
        self.workingSetBytes = workingSetBytes
        self.promptTokens = promptTokens
        self.maxNewTokens = maxNewTokens
    }
}

public struct VMLXResolvedMemorySafetyPlan: Sendable, Equatable {
    public var loadConfiguration: LoadConfiguration
    public var cache: VMLXServerCacheSettings
    public var concurrency: VMLXServerConcurrencySettings
    public var resolvedPhysicalMemoryBytes: UInt64
    public var resolvedLoadBudgetBytes: UInt64?
    public var warnings: [String]
    public var blockingIssues: [VMLXServerSettingsIssue]
    public var displaySummary: String

    public init(
        loadConfiguration: LoadConfiguration,
        cache: VMLXServerCacheSettings,
        concurrency: VMLXServerConcurrencySettings,
        resolvedPhysicalMemoryBytes: UInt64,
        resolvedLoadBudgetBytes: UInt64?,
        warnings: [String],
        blockingIssues: [VMLXServerSettingsIssue],
        displaySummary: String
    ) {
        self.loadConfiguration = loadConfiguration
        self.cache = cache
        self.concurrency = concurrency
        self.resolvedPhysicalMemoryBytes = resolvedPhysicalMemoryBytes
        self.resolvedLoadBudgetBytes = resolvedLoadBudgetBytes
        self.warnings = warnings
        self.blockingIssues = blockingIssues
        self.displaySummary = displaySummary
    }

    public var allowed: Bool {
        blockingIssues.isEmpty
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

private extension ResidentCap {
    var displayValue: String {
        switch self {
        case .unlimited:
            return "unlimited"
        case .fraction(let value):
            return "fraction(\(value))"
        case .absolute(let bytes):
            return "absolute(\(bytes))"
        }
    }
}
