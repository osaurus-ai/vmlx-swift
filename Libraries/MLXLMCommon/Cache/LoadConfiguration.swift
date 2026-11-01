// Copyright © 2026 Jinho Jang. All rights reserved.
// Typed canonical loading, allocator and residency configuration.

import Foundation
import MLX

public enum ResidentCap: Sendable, Equatable {
    case unlimited
    case fraction(Double)
    case absolute(UInt64)

    /// Generic default; model-specific admission may resolve a different cap.
    public static let `default`: ResidentCap = .fraction(0.70)

    /// Resolve to a concrete byte cap given physical RAM. Returns
    /// `nil` for `.unlimited`.
    public func resolve(physicalMemory: UInt64) -> UInt64? {
        switch self {
        case .unlimited:
            return nil
        case .fraction(let f):
            let clamped = max(0.0, min(1.0, f))
            return UInt64(Double(physicalMemory) * clamped)
        case .absolute(let bytes):
            return bytes
        }
    }
}

/// Typed canonical model loading, memory limits and optional model features.
public struct LoadConfiguration: Sendable, Equatable {
    /// Default-denied original-file repair. Does not replace the existing opt-in.
    public var alignmentRepairAuthorization: AlignmentRepairAuthorization = .disabled

    /// Hard cap on resident weight bytes (deprecated semantics — use
    /// ``memoryLimit`` instead). Kept for backwards-source-compat.
    /// In the current implementation ``maxResidentBytes`` controls
    /// `MLX.Memory.cacheLimit` only (allocator pool reuse cap), while
    /// ``memoryLimit`` controls the total MLX allocation budget.
    public var maxResidentBytes: ResidentCap

    /// Cap on the total MLX allocation budget. Maps directly onto
    /// `MLX.Memory.memoryLimit`. Calls to malloc will wait on scheduled
    /// tasks if the limit is exceeded — this is how we prevent the
    /// JANGTQ stack-materialization spike from OOM-killing the process
    /// (Ralph iter-14 / Python `_apply_wired_limit_safe_default`
    /// equivalent on Swift).
    ///
    /// Resolved at load entry to `min(fraction × physicalMemory,
    /// recommendedMaxWorkingSetBytes)` so the 847a8c7 crash condition
    /// ("limit larger than max working set size") is impossible by
    /// construction. The clamp matches MLX-swift's
    /// `WiredSumPolicy.clamp(...)` semantics.
    ///
    /// Default `.fraction(0.70)` — Python's tested-good production
    /// value, capped to whichever is smaller of physical RAM and the
    /// GPU's recommended max working set.
    public var memoryLimit: ResidentCap {
        didSet { memoryLimitWasExplicit = true }
    }
    /// Explicit initializer values and subsequent assignments are never auto-raised.
    public private(set) var memoryLimitWasExplicit: Bool

    /// Legacy source-compatibility field. Model weights always use owned storage;
    /// true no longer enables file-backed mappings.
    public var useMmapSafetensors: Bool

    /// Load preserved native-MTP tensors and instantiate model-owned MTP
    /// modules when the bundle has real tensor evidence and the family is
    /// supported by ``NativeMTPActivation``.
    ///
    /// This is intentionally a load-time switch, separate from per-request
    /// ``GenerateParameters/draftStrategy``. A request can only use native MTP
    /// if the model was loaded with the sidecar weights present; conversely,
    /// loading the sidecar does not force speculative decode for every request.
    public var nativeMTP: Bool

    /// Request GPU residency while loaded, subject to the device ceiling and
    /// physical-memory reserve. Separate from the host's idle-unload timer.
    public var keepModelInMemory: Bool = true

    /// Explicit user-selected MTP depth (1–3) carried into the load so the
    /// activation gate can accept a tensor-complete head without a measured
    /// tuning artifact. Nil for Auto/Off — Auto stays tuning-gated.
    public var nativeMTPManualDepth: Int? = nil

    /// Load DSV4-0731 with the checkpoint's activation-QAT simulation graph.
    ///
    /// `nil` preserves the headless compatibility fallback
    /// (`VMLX_DSV4_OFFICIAL_ACTIVATION_QAT=1`). Osaurus always supplies an
    /// explicit value from its visible Decode Performance setting, whose
    /// product default is `false` because the current exact-bundle Release
    /// rows show a substantial throughput cost and no completed quality gate
    /// yet establishes activation-QAT as the better user-facing default.
    /// This is a load-time model-graph choice; changing it requires unloading
    /// and reloading the model.
    public var deepseekV4ActivationQAT: Bool?

    /// Canonical owned-weight loading with the existing allocator and memory policies.
    public static let `default` = LoadConfiguration()
    public static let osaurusProduction = LoadConfiguration.default

    /// Explicit diagnostic preset without memory caps or mmap loading.
    public static let off = LoadConfiguration(
        maxResidentBytes: .unlimited,
        memoryLimit: .unlimited,
        useMmapSafetensors: false,
        nativeMTP: false,
        deepseekV4ActivationQAT: false)

    public init(
        maxResidentBytes: ResidentCap = .default,
        memoryLimit: ResidentCap? = nil,
        useMmapSafetensors: Bool = false,
        nativeMTP: Bool = false,
        deepseekV4ActivationQAT: Bool? = nil
    ) {
        self.maxResidentBytes = maxResidentBytes
        self.memoryLimit = memoryLimit ?? .default
        self.memoryLimitWasExplicit = memoryLimit != nil
        self.useMmapSafetensors = useMmapSafetensors
        self.nativeMTP = nativeMTP
        self.deepseekV4ActivationQAT = deepseekV4ActivationQAT
    }

    /// A scheduling budget, not a physical-footprint limit. Every family owns
    /// its packed weights; an automatic allowance below them would repeatedly
    /// drain GPU work and reusable buffers. Explicit limits and oversized or
    /// unknown bundles remain admission decisions, never silently raised.
    func resolvedSchedulingMemoryLimit(
        facts: LoadBundleFacts, recommendedWorkingSetBytes: Int?
    ) -> ResidentCap {
        let ordinary = facts.resolveMLXMemoryLimit(requested: memoryLimit)
        guard !memoryLimitWasExplicit, facts.gpuResidentWeightBytes > 0,
            let recommendedWorkingSetBytes, recommendedWorkingSetBytes > 0,
            let base = ordinary.resolve(physicalMemory: facts.physicalMemory)
        else { return ordinary }
        let headroom: UInt64 = 4 << 30
        let desired = facts.gpuResidentWeightBytes.addingReportingOverflow(headroom)
        // Keep at least 15% of physical RAM outside this scheduling allowance.
        let physical85 = (facts.physicalMemory / 20) * 17
            + ((facts.physicalMemory % 20) * 17) / 20
        let ceiling = min(physical85, UInt64(recommendedWorkingSetBytes))
        // A model that fits the ceiling must not fall back below its own
        // weight size merely because the preferred 4 GiB scratch allowance
        // does not fit. That fallback drains the allocator's reusable buffers
        // on every step (Allosaurus on 48 GiB). A ceiling below the weights
        // remains an admission concern; do not silently exceed it here.
        guard !desired.overflow, facts.gpuResidentWeightBytes < ceiling else {
            return ordinary
        }
        let target = min(desired.partialValue, ceiling)
        guard target > base else { return ordinary }
        return .absolute(target)
    }
}

/// Per-load DSV4 activation-QAT request carried across the async model-factory
/// boundary without process-global mutable state.
public enum DeepseekV4ActivationQAT {
    @TaskLocal public static var explicitRequestOverride: Bool?

    /// Legacy/headless opt-in. Absence is deliberately OFF.
    public static var environmentDefault: Bool {
        let raw = ProcessInfo.processInfo.environment[
            "VMLX_DSV4_OFFICIAL_ACTIVATION_QAT"]?.lowercased()
        return raw == "1" || raw == "true" || raw == "on" || raw == "yes"
    }

    public static var enabledForCurrentLoad: Bool {
        explicitRequestOverride ?? environmentDefault
    }

    public static func withExplicitRequest<R>(
        _ enabled: Bool?,
        operation: () async throws -> R
    ) async rethrows -> R {
        try await $explicitRequestOverride.withValue(enabled) {
            try await operation()
        }
    }
}

// MARK: - Resolution

/// Snapshot of bundle facts used for canonical loading and memory admission —
/// pulled once at load entry so the resolver doesn't re-walk the
/// directory or re-parse `config.json`.
public struct LoadBundleFacts: Sendable, Equatable {
    /// Metadata classification only, not tensor geometry or model-runtime proof.
    public enum CustomRoutedFormat: Sendable, Equatable {
        case none
        case janghV2
        case invalidJANGHDeclaration
    }
    public private(set) var customRoutedFormat: CustomRoutedFormat = .none

    /// Converted MiMo V2.6 packed banks are owned GPU inputs by default.
    public private(set) var isMiMoV26MixedQuantized: Bool = false
    /// Top-level or nested `model_type` from config metadata, when available.
    public var modelType: String?

    /// Top-level or JANG-sidecar `weight_format`, lowercased when available.
    public var weightFormat: String?

    /// Embedded `jang_config.format`, lowercased when available. Newer JANG
    /// bundles carry this contract inside `config.json` instead of a sidecar.
    public var jangFormat: String?

    /// Top-level or nested text-model compute dtype declared by the bundle.
    public var declaredComputeDType: String?

    /// True when the bundle has a `jang_config.json` sidecar.
    public var hasJangConfig: Bool

    /// True when the bundle has a `jangtq_runtime.safetensors` sidecar.
    public var hasJangTQRuntime: Bool

    /// Declared routed-expert storage layout, normalized to lowercase.
    public var routedExpertLayout: String?

    /// True only when a plain affine DSV4 bundle declares the converter's
    /// pre-stacked layout and its weight index proves the complete canonical
    /// 43-layer routed bank with no split-expert tensors.
    public private(set) var hasPrestackedAffineRoutedExperts: Bool

    /// Sum of `*.safetensors` byte sizes in the bundle. `0` when
    /// inspection failed (treat as unknown — `.auto` falls through to
    /// disabled in that case).
    public var totalSafetensorsBytes: UInt64

    /// Bytes of tables that are read from SSD on demand and never become GPU-resident weights: the Qwen4Exp
    /// (Flash-Next / Allosaurus) per-layer n-gram embedding shards (`*.ple.ngram_embedding.*`, 18-54 GB). They are
    /// served by `pread` rows through the page cache, not MLX arrays.
    public var ssdResidentTableBytes: UInt64 = 0

    /// The weights that must be resident for decode: the bundle minus SSD-resident tables. Wiring, the
    /// materialize-instead-of-mmap rule and the JANGH scheduling allowance size against this, not the whole
    /// bundle (Allosaurus: 32.5 GB of weights in an 88 GB bundle).
    public var gpuResidentWeightBytes: UInt64 {
        totalSafetensorsBytes >= ssdResidentTableBytes ? totalSafetensorsBytes - ssdResidentTableBytes : 0
    }

    /// True iff `config.json` declares any of:
    /// `num_local_experts`, `num_experts`, `moe_intermediate_size`,
    /// `n_routed_experts`, or contains a non-empty `experts` mapping.
    /// `false` when inspection failed (treat as dense).
    public var isRouted: Bool

    /// Best-effort routed-expert count sniffed from `config.json`.
    /// Describes the routed expert topology, including each layer's
    /// set. `nil` when the bundle does not expose the shape.
    public var numRoutedExperts: Int?

    /// Best-effort top-k routed experts per token sniffed from
    /// `config.json`. `nil` falls back to the advisor's conservative
    /// default.
    public var topK: Int?

    /// Physical memory snapshot captured at the same moment, so cap
    /// math stays internally consistent with the bundle inspection.
    public var physicalMemory: UInt64

    public init(
        totalSafetensorsBytes: UInt64,
        isRouted: Bool,
        physicalMemory: UInt64,
        modelType: String? = nil,
        weightFormat: String? = nil,
        jangFormat: String? = nil,
        declaredComputeDType: String? = nil,
        hasJangConfig: Bool = false,
        hasJangTQRuntime: Bool = false,
        routedExpertLayout: String? = nil,
        numRoutedExperts: Int? = nil,
        topK: Int? = nil
    ) {
        self.modelType = modelType
        self.weightFormat = weightFormat?.lowercased()
        self.jangFormat = jangFormat?.lowercased()
        self.declaredComputeDType = declaredComputeDType?.lowercased()
        self.hasJangConfig = hasJangConfig
        self.hasJangTQRuntime = hasJangTQRuntime
        self.routedExpertLayout = routedExpertLayout?.lowercased()
        self.hasPrestackedAffineRoutedExperts = false
        self.totalSafetensorsBytes = totalSafetensorsBytes
        self.isRouted = isRouted
        self.physicalMemory = physicalMemory
        self.numRoutedExperts = numRoutedExperts
        self.topK = topK
    }

    /// Inspect the bundle directory at `url`. All probes are best-effort —
    /// failures degrade silently to safe defaults (zero bytes, dense).
    public static func inspect(bundleURL: URL) -> LoadBundleFacts {
        // A bundle reached through a symlinked directory (models kept on another volume and
        // linked into the models folder) must report the same facts as its target.
        // `contentsOfDirectory(at:)` throws ENOTDIR on a URL naming a symlink to a directory, so
        // the unresolved URL listed nothing: zero weight bytes, so no Metal residency budget, so
        // the whole model stayed reclaimable. DeepSeek-V4.1 JANG_2K (197 GiB) on a 256 GiB Mac
        // then had 69-80 GiB of its weights compressed right after load, and the first command
        // buffer that had to bring them back hit the GPU timeout.
        let url = bundleURL.resolvingSymlinksInPath()
        let fm = FileManager.default
        let physical = HostPhysicalMemory.bytes

        // Walk top-level for *.safetensors. Do not recurse: the loader
        // bundles never nest weights under subdirs.
        var totalBytes: UInt64 = 0
        if let entries = try? fm.contentsOfDirectory(
            at: url,
            includingPropertiesForKeys: [.fileSizeKey],
            options: [.skipsHiddenFiles])
        {
            for entry in entries where entry.pathExtension == "safetensors" {
                // Through the link: a Hugging Face cache stores every shard as a symlink into
                // `blobs/`, and a link reports its own few dozen bytes, not the weights'.
                if let values = try? entry.resolvingSymlinksInPath().resourceValues(forKeys: [.fileSizeKey]),
                    let size = values.fileSize
                {
                    totalBytes &+= UInt64(size)
                }
            }
        }

        // Probe config.json for routed markers.
        var routed = false
        var numRoutedExperts: Int?
        var topK: Int?
        var numHiddenLayers: Int?
        var modelType: String?
        var weightFormat: String?
        var jangFormat: String?
        var declaredComputeDType: String?
        var routedExpertLayout: String?
        var isMiMoV26MixedQuantized = false
        let rawConfiguration = try? Data(contentsOf: url.appendingPathComponent("config.json"))
        let rawSidecar = try? Data(contentsOf: url.appendingPathComponent("jang_config.json"))
        var customRoutedFormat: CustomRoutedFormat = .none
        if let rawConfiguration,
           JANGHModelPreparation.declaresCustomFormat(configuration: rawConfiguration, sidecar: rawSidecar) {
            customRoutedFormat = (try? JANGHConfigurationPartition(
                configuration: rawConfiguration, sidecar: rawSidecar)) == nil
                ? .invalidJANGHDeclaration : .janghV2
        }
        if let data = rawConfiguration,
            let json = try? JSONSerialization.jsonObject(with: data)
                as? [String: Any]
        {
            isMiMoV26MixedQuantized = MiMoV26BundleContract.matches(data)
            let routedKeys = [
                "num_local_experts",
                "num_experts",
                "moe_intermediate_size",
                "n_routed_experts",
            ]
            let routedExpertKeys = [
                "n_routed_experts",
                "num_local_experts",
                "num_experts",
            ]
            let topKKeys = [
                "num_experts_per_tok",
                "top_k_experts",
                "experts_per_token",
            ]
            func firstPositiveInt(
                in object: [String: Any],
                keys: [String]
            ) -> Int? {
                for key in keys {
                    if let n = object[key] as? Int, n > 0 {
                        return n
                    }
                    if let n = object[key] as? NSNumber, n.intValue > 0 {
                        return n.intValue
                    }
                }
                return nil
            }

            numRoutedExperts = firstPositiveInt(
                in: json, keys: routedExpertKeys)
            topK = firstPositiveInt(in: json, keys: topKKeys)
            numHiddenLayers = firstPositiveInt(
                in: json, keys: ["num_hidden_layers", "n_layers"])
            modelType = json["model_type"] as? String
            weightFormat = (json["weight_format"] as? String)?.lowercased()
            declaredComputeDType = ((json["dtype"] as? String)
                ?? (json["torch_dtype"] as? String))?.lowercased()
            if let embeddedJang = json["jang_config"] as? [String: Any] {
                jangFormat = (embeddedJang["format"] as? String)?.lowercased()
            }
            routedExpertLayout = (json["routed_expert_layout"] as? String)?.lowercased()
            for key in routedKeys {
                if let n = json[key] as? Int, n > 1 {
                    routed = true
                    break
                }
            }
            // Some VLMs nest the LM stanza under `text_config` /
            // `language_config`; check those too. Even when the
            // top-level wrapper already proved "routed", nested
            // values may be the only source for `num_experts_per_tok`.
            for sub in ["text_config", "language_config"] {
                if let nested = json[sub] as? [String: Any] {
                    if numRoutedExperts == nil {
                        numRoutedExperts = firstPositiveInt(
                            in: nested, keys: routedExpertKeys)
                    }
                    if topK == nil {
                        topK = firstPositiveInt(in: nested, keys: topKKeys)
                    }
                    if modelType == nil {
                        modelType = nested["model_type"] as? String
                    }
                    if weightFormat == nil {
                        weightFormat = (nested["weight_format"] as? String)?.lowercased()
                    }
                    if declaredComputeDType == nil {
                        declaredComputeDType = ((nested["dtype"] as? String)
                            ?? (nested["torch_dtype"] as? String))?.lowercased()
                    }
                    if numHiddenLayers == nil {
                        numHiddenLayers = firstPositiveInt(
                            in: nested, keys: ["num_hidden_layers", "n_layers"])
                    }
                    if routedExpertLayout == nil {
                        routedExpertLayout =
                            (nested["routed_expert_layout"] as? String)?.lowercased()
                    }
                    if !routed {
                        for key in routedKeys {
                            if let n = nested[key] as? Int, n > 1 {
                                routed = true
                                break
                            }
                        }
                    }
                }
            }
            if (numRoutedExperts ?? 0) > 1 {
                routed = true
            }
        }
        let jangConfigURL = url.appendingPathComponent("jang_config.json")
        let hasJangConfig = fm.fileExists(atPath: jangConfigURL.path)
        if hasJangConfig,
            let data = rawSidecar,
            let json = try? JSONSerialization.jsonObject(with: data)
                as? [String: Any]
        {
            if modelType == nil {
                modelType = json["model_type"] as? String
            }
            if weightFormat == nil {
                weightFormat = (json["weight_format"] as? String)?.lowercased()
            }
            if jangFormat == nil {
                jangFormat = (json["format"] as? String)?.lowercased()
            }
            if routedExpertLayout == nil {
                routedExpertLayout =
                    (json["routed_expert_layout"] as? String)?.lowercased()
            }
        }
        let hasJangTQRuntime = fm.fileExists(
            atPath: url.appendingPathComponent("jangtq_runtime.safetensors").path)
        let hasPrestackedAffineRoutedExperts =
            routedExpertLayout == "prestacked_affine"
            && numHiddenLayers == Self.deepseekV4AffineLayerCount
            && numRoutedExperts == 256
            && Self.hasCompleteDeepseekV4AffinePrestackedIndex(bundleURL: url)

        var facts = LoadBundleFacts(
            totalSafetensorsBytes: totalBytes,
            isRouted: routed,
            physicalMemory: physical,
            modelType: modelType,
            weightFormat: weightFormat,
            jangFormat: jangFormat,
            declaredComputeDType: declaredComputeDType,
            hasJangConfig: hasJangConfig,
            hasJangTQRuntime: hasJangTQRuntime,
            routedExpertLayout: routedExpertLayout,
            numRoutedExperts: numRoutedExperts,
            topK: topK)
        facts.hasPrestackedAffineRoutedExperts = hasPrestackedAffineRoutedExperts
        facts.isMiMoV26MixedQuantized = isMiMoV26MixedQuantized
        facts.customRoutedFormat = customRoutedFormat
        facts.ssdResidentTableBytes = Self.ngramTableBytes(bundleURL: url)
        return facts
    }

    /// Sum of `*.ple.ngram_embedding.*` tensor payloads, read from safetensors headers only (no weights).
    /// Unreadable headers and invalid spans are ignored, conservatively retaining those bytes as weights.
    static func ngramTableBytes(bundleURL url: URL) -> UInt64 {
        guard let entries = try? FileManager.default.contentsOfDirectory(
            at: url, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles])
        else { return 0 }
        var total: UInt64 = 0
        for entry in entries where entry.pathExtension == "safetensors" {
            guard let handle = try? FileHandle(forReadingFrom: entry) else { continue }
            defer { try? handle.close() }
            guard let lengthData = try? handle.read(upToCount: 8), lengthData.count == 8 else { continue }
            let length = lengthData.enumerated().reduce(UInt64(0)) { $0 | UInt64($1.element) << UInt64(8 * $1.offset) }
            guard length > 0, length <= 64 << 20, let headerData = try? handle.read(upToCount: Int(length)),
                headerData.count == Int(length),
                let header = try? JSONSerialization.jsonObject(with: headerData) as? [String: Any]
            else { continue }
            // A header describes offsets relative to the payload, not the file.
            // Never subtract untrusted signed offsets: Int.min...Int.max can
            // trap before the actual weight loader gets to reject the shard.
            guard let fileSize = try? handle.seekToEnd(), fileSize >= 8 + length else { continue }
            let payloadBytes = fileSize - 8 - length
            for (name, value) in header where name.contains(".ple.ngram_embedding.") {
                guard let offsets = (value as? [String: Any])?["data_offsets"] as? [Int],
                    offsets.count == 2, offsets[0] >= 0, offsets[1] >= offsets[0],
                    UInt64(offsets[1]) <= payloadBytes
                else { continue }
                let bytes = UInt64(offsets[1]) - UInt64(offsets[0])
                let sum = total.addingReportingOverflow(bytes)
                // If accounting cannot be represented, subtract nothing from
                // the resident estimate rather than wrap to a smaller value.
                guard !sum.overflow else { return 0 }
                total = sum.partialValue
            }
        }
        return total
    }

    private static let deepseekV4AffineLayerCount = 43
    private static let deepseekV4AffineProjections = [
        "gate_proj", "down_proj", "up_proj",
    ]
    private static let deepseekV4AffineQuantSuffixes = [
        "weight", "scales", "biases",
    ]

    private static let expectedDeepseekV4AffinePrestackedKeys: Set<String> = {
        var keys = Set<String>()
        for layer in 0 ..< deepseekV4AffineLayerCount {
            for projection in deepseekV4AffineProjections {
                for suffix in deepseekV4AffineQuantSuffixes {
                    keys.insert(
                        "layers.\(layer).mlp.switch_mlp.\(projection).\(suffix)")
                }
            }
        }
        return keys
    }()

    private static func hasCompleteDeepseekV4AffinePrestackedIndex(
        bundleURL: URL
    ) -> Bool {
        let indexURL = bundleURL.appendingPathComponent(
            "model.safetensors.index.json")
        guard
            let data = try? Data(contentsOf: indexURL),
            let json = try? JSONSerialization.jsonObject(with: data)
                as? [String: Any],
            let weightMap = json["weight_map"] as? [String: Any],
            weightMap.values.allSatisfy({ $0 is String })
        else {
            return false
        }

        let indexedKeys = Set(weightMap.keys)
        guard !indexedKeys.contains(where: isDeepseekV4SplitExpertKey) else {
            return false
        }

        let canonicalPrestackedKeys = indexedKeys.filter(
            isDeepseekV4CanonicalPrestackedKey)
        return canonicalPrestackedKeys.count
            == expectedDeepseekV4AffinePrestackedKeys.count
            && expectedDeepseekV4AffinePrestackedKeys.isSubset(of: indexedKeys)
    }

    private static func isDeepseekV4SplitExpertKey(_ key: String) -> Bool {
        key.hasPrefix("layers.") && key.contains(".ffn.experts.")
    }

    private static func isDeepseekV4CanonicalPrestackedKey(
        _ key: String
    ) -> Bool {
        let parts = key.split(separator: ".", omittingEmptySubsequences: false)
        guard parts.count == 6,
            parts[0] == "layers",
            Int(parts[1]) != nil,
            parts[2] == "mlp",
            parts[3] == "switch_mlp",
            deepseekV4AffineProjections.contains(String(parts[4])),
            deepseekV4AffineQuantSuffixes.contains(String(parts[5]))
        else {
            return false
        }
        return true
    }

    public var isPlainDeepseekV4AffineJANG: Bool {
        let type = modelType?.lowercased() ?? ""
        let format = weightFormat?.lowercased() ?? ""
        let declaresAffine = format.isEmpty || format == "affine" || format == "jang"
        return type == "deepseek_v4"
            && hasJangConfig
            && !hasJangTQRuntime
            && declaresAffine
            && isRouted
            && (numRoutedExperts ?? 0) >= 128
    }

    /// DeepSeek-V4.1 affine JANG (prestacked routed experts, disk-resident
    /// engram). Resident like plain DSV4 (the model requests it), so the MLX
    /// memory/allocator caps must not throttle decode either.
    public var isDeepseekV41AffineJANG: Bool {
        let type = modelType?.lowercased() ?? ""
        let format = weightFormat?.lowercased() ?? ""
        let declaresAffine = format.isEmpty || format == "affine" || format == "jang"
        return (type == "deepseek_v41" || type == "deepseek_v41_text")
            && hasJangConfig
            && !hasJangTQRuntime
            && declaresAffine
            && isRouted
            && (numRoutedExperts ?? 0) >= 128
    }

    /// Qwen3.8 Flash Next JANG keeps its large PLE/ngram table in a
    /// model-owned exact-row SSD reader. Match the bundle's explicit family,
    /// JANG format, and BF16 compute declaration; model names and directory
    /// names are deliberately irrelevant.
    public var isQwen4ExpBF16JANG: Bool {
        let type = modelType?.lowercased() ?? ""
        let format = jangFormat?.lowercased() ?? ""
        let dtype = declaredComputeDType?.lowercased() ?? ""
        return (type == "qwen4_exp" || type == "qwen4_exp_text")
            && format.hasPrefix("jang")
            && !hasJangTQRuntime
            && ["bfloat16", "bf16"].contains(dtype)
            && isRouted
    }

    /// GLM-5.3 (`glm5_next`) needs the same resident treatment DSV4 does, and for the same reason:
    /// a routed affine bank far larger than the page cache will hold.
    ///
    /// Measured on a 95 GB bundle / 128 GB machine. Mapped, it sits at 60-74 GB file-backed with
    /// free memory pinned near zero and the kernel evicting and re-reading continuously — `sys`
    /// runs ~37x `user`, i.e. the model spends its life faulting rather than computing.
    ///
    /// This is NOT "mmap is bad": mmap is what lets a model LARGER than RAM run at all. It is that
    /// a bundle which fits resident should be resident, because file-backed pages are evictable and
    /// macOS drops them eagerly, while anonymous pages are not. 95 GB of 128 GB fits.
    public var isGlm5NextAffineJANG: Bool {
        let type = modelType?.lowercased() ?? ""
        let format = weightFormat?.lowercased() ?? ""
        let declaresAffine = format.isEmpty || format == "affine" || format == "jang"
        return (type == "glm5_next" || type == "glm5_next_text")
            && customRoutedFormat == .none
            // A format label alone cannot validate custom banks, but it must
            // never opt a declared custom bundle into an affine resident policy.
            && jangFormat?.lowercased() != "jangtq2"
            && !hasJangTQRuntime
            && declaresAffine
            && isRouted
            && (numRoutedExperts ?? 0) >= 128
    }

    /// Plain split-expert DSV4 currently requires a resident compute bank.
    /// Qwen4Exp must retain the caller's mmap choice: its 61+ GiB packed bank
    /// is valid mixed-dtype MLX input, while its PLE/ngram tensors already stay
    /// in the model-owned exact-row SSD reader.
    public var requiresResidentSafetensors: Bool {
        (isPlainDeepseekV4AffineJANG && !hasPrestackedAffineRoutedExperts)
            || isGlm5NextAffineJANG
            || isMiMoV26MixedQuantized
    }

    public func resolveMmapSafetensors(requested: Bool) -> Bool {
        false // File-backed model-weight ownership is retired.
    }

    /// A resident affine DSV4 load cannot use `MLX.Memory.memoryLimit` as a
    /// RAM-safety mechanism. The allocator cap does not constrain Activity
    /// Monitor physical footprint for this bundle, but it serializes decode
    /// and collapses throughput. Hosts must perform their normal pre-load RAM
    /// admission check instead of silently throttling a successfully admitted
    /// model.
    /// Families whose resident bank is larger than the default cap, so a cap would force MLX to
    /// destroy and rebuild decode intermediates every token.
    ///
    /// Measured on GLM-5.3: the bundle is 95 GB and the default cap is 0.70 x 128 GB = 89.6 GB, so
    /// the model does not fit under its own allowance. A decode profile is then dominated by
    /// `MetalAllocator::malloc` and `IOGPUResourceCreate` — buffer churn through IOKit — rather
    /// than by any matmul, and generation runs at 0.38 tok/s.
    ///
    /// One property for both caps deliberately: they are the same decision about the same bundle,
    /// and a family added to one list and not the other is exactly how this reached production.
    public var requiresUncappedResidentPools: Bool {
        isPlainDeepseekV4AffineJANG || isGlm5NextAffineJANG || isDeepseekV41AffineJANG || isMiniMaxM3Routed
    }

    /// MiniMax-M3 / M3.1 (any routed format: JANGHT banks, affine, the MXFP8 source). Same mechanism as the families
    /// above, reached through context instead of weight size: Safe Auto's scheduling allowance is the resident
    /// weights + 4 GiB, and the sparse-attention KV grows 138,240 B per token, so from ~26k tokens of context (M3.1
    /// JANGHT1.9 + its drafter on a 128 GB Mac: 100.9 GB weights, 104.5 GB limit) every MLX eval waits for the queue
    /// to drain. Measured in the Osaurus GUI proof (m5-max, 2026-10-29): a 35k-token turn prefilled at 284 pp/s and
    /// decoded at 5.9 tok/s (AR 150 ms / token) against 36 tok/s at 33k on the same runtime without the limit.
    public var isMiniMaxM3Routed: Bool {
        let type = modelType?.lowercased() ?? ""
        return (type == "minimax_m3" || type == "minimax_m3_vl") && isRouted
    }

    public func resolveMLXMemoryLimit(requested: ResidentCap) -> ResidentCap {
        requiresUncappedResidentPools ? .unlimited : requested
    }

    /// Plain affine DSV4 must also keep MLX's freed-buffer reuse pool
    /// uncapped. `maxResidentBytes` controls `MLX.Memory.cacheLimit`, not the
    /// model's physical footprint; a small profile cap forces the large routed
    /// decode intermediates to be destroyed and rebuilt every token. That is
    /// the same throughput-collapse mechanism as the total MLX memory limit
    /// above, so RAM admission remains the owning safety gate for both caps.
    public func resolveMLXAllocatorCacheLimit(requested: ResidentCap) -> ResidentCap {
        requiresUncappedResidentPools ? .unlimited : requested
    }
}

extension ResidentCap {
    /// Apply the cap to MLX's allocator cache for the duration of a
    /// scoped block (typically one load call). Returns the prior limit
    /// so the caller can restore it via `defer`. `nil` return means
    /// `.unlimited` — caller should leave the limit untouched.
    ///
    /// Note: `MLX.Memory.cacheLimit` is the allocator's internal pool,
    /// not a hard RSS ceiling. Capping it forces freed intermediates
    /// to release back to the OS instead of accumulating in the pool —
    /// the same mechanism iter 25 uses with its hard-coded 1 GB.
    public func applyAsCacheLimitInt(physicalMemory: UInt64) -> Int? {
        guard let bytes = self.resolve(physicalMemory: physicalMemory) else {
            return nil
        }
        // Clamp to Int.max — `MLX.Memory.cacheLimit` is `Int`. On
        // 64-bit platforms this is 9.2 EB so the clamp is a formality.
        return Int(min(UInt64(Int.max), bytes))
    }
}

// MARK: - MemoryStatus

/// Snapshot of MLX memory caps + current process RSS. Designed for UI
/// display (osaurus settings panel, JANG Studio inspector). All fields
/// are plain values; cheap enough to poll on a timer.
///
/// Pull via ``MemoryStatus/snapshot()`` — bypasses any container plumbing
/// because all four MLX limits are process-global.
public struct MemoryStatus: Sendable, Equatable {
    /// Total MLX allocation budget. Calls to malloc wait if exceeded.
    /// Set at load entry to `min(LoadConfiguration.memoryLimit fraction
    /// × physicalMemory, recommendedMaxWorkingSetBytes)`.
    public let memoryLimit: Int

    /// Allocator pool reuse cap. Freed intermediates above this size
    /// release back to the OS instead of accumulating in the cache.
    public let cacheLimit: Int

    /// MTLDevice's recommended max working set size — the upper bound
    /// for any wired-limit setting (exceeding it triggers the 847a8c7
    /// crash). nil when Metal is unavailable.
    public let recommendedWorkingSetBytes: Int?

    /// System physical RAM as reported by `sysctl(HW_MEMSIZE)`.
    public let physicalMemory: UInt64

    /// Current process resident set size sampled from
    /// `mach_task_basic_info`. Reflects ALL process memory (MLX +
    /// frameworks + stack + heap), not just MLX-allocated tensors.
    public let currentRSS: UInt64

    /// Capture all current MLX caps + a fresh RSS sample.
    public static func snapshot() -> MemoryStatus {
        MemoryStatus(
            memoryLimit: MLX.Memory.memoryLimit,
            cacheLimit: MLX.Memory.cacheLimit,
            recommendedWorkingSetBytes: MLX.GPU.maxRecommendedWorkingSetBytes(),
            physicalMemory: HostPhysicalMemory.bytes,
            currentRSS: residentSetSizeBytes())
    }
}

#if canImport(Darwin)
import Darwin

/// Sample the current process RSS via mach_task_basic_info. Returns 0
/// on failure (rare — the syscall is well-behaved on macOS / iOS).
private func residentSetSizeBytes() -> UInt64 {
    var info = mach_task_basic_info()
    var count = mach_msg_type_number_t(
        MemoryLayout<mach_task_basic_info>.size / MemoryLayout<integer_t>.size)
    let kr = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_,
                task_flavor_t(MACH_TASK_BASIC_INFO),
                $0, &count)
        }
    }
    guard kr == KERN_SUCCESS else { return 0 }
    return UInt64(info.resident_size)
}
#else
private func residentSetSizeBytes() -> UInt64 { 0 }
#endif
