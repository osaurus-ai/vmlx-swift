// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX

/// Runtime mode stamped by a model bundle or inferred from its files.
///
/// `preserved_enabled` means the converted artifact carries MTP metadata and
/// tensors, not that speculative MTP decode is live. The engine may only launch
/// MTP automatically when the mode proves a verified accept/reject runtime.
public enum MTPRuntimeMode: String, Codable, Sendable, Equatable {
    case none
    case preservedDisabled = "preserved_disabled"
    case preservedEnabled = "preserved_enabled"
    case metadataOnlyMissingWeights = "metadata_only_missing_weights"
    case enabled
    case speculativeVerified = "speculative_verified"
    case unknown

    public init(rawMode: String?) {
        let normalized =
            (rawMode ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
            .replacingOccurrences(of: "-", with: "_")
            .replacingOccurrences(of: " ", with: "_")

        switch normalized {
        case "", "none", "off", "false":
            self = .none
        case "disabled", "preserved_disabled", "preserved_off":
            self = .preservedDisabled
        case "preserved", "preserved_enabled", "runtime_unwired", "available":
            self = .preservedEnabled
        case "metadata_only", "metadata_only_missing_weights", "config_only":
            self = .metadataOnlyMissingWeights
        case "enabled", "speculative_enabled", "accept_reject", "accept_reject_enabled":
            self = .enabled
        case "speculative_verified", "verified", "verified_accept_reject":
            self = .speculativeVerified
        default:
            self = .unknown
        }
    }

    public var hasSpeculativeAcceptReject: Bool {
        switch self {
        case .enabled, .speculativeVerified:
            true
        case .none, .preservedDisabled, .preservedEnabled, .metadataOnlyMissingWeights,
            .unknown:
            false
        }
    }
}

/// No-load MTP status derived from `config.json`, `jang_config.json`, and tensor
/// names. This type is safe to expose through Osaurus capability/status APIs.
public struct MTPBundleStatus: Codable, Sendable, Equatable {
    public let bundleHasMTP: Bool
    public let configuredLayers: Int
    public let tensorCount: Int
    public let visionTensorCount: Int
    public let mode: MTPRuntimeMode
    public let tensorSamples: [String]
    public let visionTensorSamples: [String]
    public let configEvidence: [String]

    public init(
        bundleHasMTP: Bool = false,
        configuredLayers: Int = 0,
        tensorCount: Int = 0,
        visionTensorCount: Int = 0,
        mode: MTPRuntimeMode = .none,
        tensorSamples: [String] = [],
        visionTensorSamples: [String] = [],
        configEvidence: [String] = []
    ) {
        self.bundleHasMTP = bundleHasMTP
        self.configuredLayers = configuredLayers
        self.tensorCount = tensorCount
        self.visionTensorCount = visionTensorCount
        self.mode = mode
        self.tensorSamples = tensorSamples
        self.visionTensorSamples = visionTensorSamples
        self.configEvidence = configEvidence
    }

    public var hasCompleteMTPArtifact: Bool {
        bundleHasMTP && configuredLayers > 0 && tensorCount > 0
    }

    public var speculativeDecodeEnabled: Bool {
        hasCompleteMTPArtifact && mode.hasSpeculativeAcceptReject
    }

    public var canAutoLaunchMTP: Bool {
        speculativeDecodeEnabled
    }

    public var requiresAcceptRejectBeforeEnable: Bool {
        hasCompleteMTPArtifact && !mode.hasSpeculativeAcceptReject
    }

    public var bundleHasVision: Bool {
        visionTensorCount > 0
    }

    public var statusLine: String {
        let base = "mtp: \(mode.rawValue), layers=\(configuredLayers), tensors=\(tensorCount)"
        if speculativeDecodeEnabled {
            return "\(base), speculative=on"
        }
        if requiresAcceptRejectBeforeEnable {
            return "\(base), speculative=off (accept/reject required)"
        }
        if configuredLayers > 0 && tensorCount == 0 {
            return "\(base), speculative=off (metadata only; MTP weights missing)"
        }
        return "\(base), speculative=off"
    }
}

/// Output from a native-MTP capable target forward.
///
/// `hiddenStates` is the pre-final-norm hidden state at the same positions as
/// `logits`. Qwen MTP heads fuse this hidden state with the next sampled token,
/// so final-norm activations are not a valid substitute.
public struct NativeMTPForwardResult {
    public let logits: MLXArray
    public let hiddenStates: MLXArray

    public init(logits: MLXArray, hiddenStates: MLXArray) {
        self.logits = logits
        self.hiddenStates = hiddenStates
    }
}

/// Capability exposed only by loaded models that have a real native MTP head.
public protocol NativeMTPModel: LanguageModel {
    /// True only when the concrete model instance has an instantiated MTP module
    /// whose weights were allowed through the loader.
    var nativeMTPAvailable: Bool { get }

    /// Private per-request cache for the MTP head. It is never prefix/paged/L2
    /// cache state.
    func makeNativeMTPCache() -> [KVCache]

    /// Target/backbone forward that returns logits plus pre-final-norm hidden.
    func nativeBackboneForward(_ inputs: MLXArray, cache: [KVCache]?) -> NativeMTPForwardResult

    /// Target/backbone verifier forward for native MTP.
    ///
    /// Implementations with non-trimmable recurrent state can record
    /// prefix-commit snapshots while still returning logits for the full
    /// verifier sequence. The iterator decides the accepted prefix after
    /// sampling and then commits those recorded states.
    func nativeBackboneMTPVerifyForward(
        _ inputs: MLXArray,
        cache: [KVCache]?
    ) -> NativeMTPForwardResult

    /// One recursive MTP draft step. `nextTokenIds` is the sampled token at the
    /// position after `hiddenStates`.
    func nativeMTPForward(
        hiddenStates: MLXArray,
        nextTokenIds: MLXArray,
        cache: [KVCache]?
    ) -> NativeMTPForwardResult
}

public extension NativeMTPModel {
    func nativeBackboneMTPVerifyForward(
        _ inputs: MLXArray,
        cache: [KVCache]?
    ) -> NativeMTPForwardResult {
        nativeBackboneForward(inputs, cache: cache)
    }
}

public enum NativeMTPActivationError: Error, CustomStringConvertible {
    case requestedButMissingArtifact(MTPBundleStatus?)
    case requestedForUnsupportedModel([String])
    case invalidConfigData

    public var description: String {
        switch self {
        case .requestedButMissingArtifact(let status):
            return "native MTP was requested but this bundle does not have complete MTP tensor evidence: \(status?.statusLine ?? "no status")"
        case .requestedForUnsupportedModel(let types):
            return "native MTP was requested for unsupported model type(s): \(types.joined(separator: ", "))"
        case .invalidConfigData:
            return "native MTP config rewrite failed"
        }
    }
}

/// Fail-closed native-MTP activation policy.
///
/// The runtime never enables native MTP from a path or marketing name. The first
/// Swift implementation is deliberately Qwen3.6/Qwen3.5 only and requires:
/// `VMLINUX_NATIVE_MTP=1`, supported config model type, and real MTP tensor keys.
public enum NativeMTPActivation {
    public static var isExplicitlyRequested: Bool {
        let raw = ProcessInfo.processInfo.environment["VMLINUX_NATIVE_MTP"]?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased() ?? "0"
        return ["1", "true", "yes", "on"].contains(raw)
    }

    public static func shouldLoadNativeMTPWeights(
        configData: Data,
        baseModelType: String,
        status: MTPBundleStatus?
    ) throws -> Bool {
        guard isExplicitlyRequested else { return false }
        let modelTypes = modelTypes(in: configData, fallback: baseModelType)
        guard modelTypes.contains(where: isSupportedQwenMTPModelType) else {
            throw NativeMTPActivationError.requestedForUnsupportedModel(modelTypes)
        }
        guard status?.hasCompleteMTPArtifact == true else {
            throw NativeMTPActivationError.requestedButMissingArtifact(status)
        }
        return true
    }

    public static func scrubInactiveMTPConfig(_ configData: Data) throws -> Data {
        guard var object = try JSONSerialization.jsonObject(with: configData) as? [String: Any]
        else {
            throw NativeMTPActivationError.invalidConfigData
        }
        scrubMTPKeys(in: &object)
        if var textConfig = object["text_config"] as? [String: Any] {
            scrubMTPKeys(in: &textConfig)
            object["text_config"] = textConfig
        }
        return try JSONSerialization.data(withJSONObject: object)
    }

    private static func scrubMTPKeys(in object: inout [String: Any]) {
        if object["mtp_num_hidden_layers"] != nil {
            object["mtp_num_hidden_layers"] = 0
        }
    }

    private static func modelTypes(in configData: Data, fallback: String) -> [String] {
        var result = Set<String>()
        if !fallback.isEmpty { result.insert(fallback) }
        if let object = try? JSONSerialization.jsonObject(with: configData) as? [String: Any] {
            if let top = object["model_type"] as? String { result.insert(top) }
            if let textConfig = object["text_config"] as? [String: Any],
                let text = textConfig["model_type"] as? String
            {
                result.insert(text)
            }
        }
        return Array(result).sorted()
    }

    private static func isSupportedQwenMTPModelType(_ value: String) -> Bool {
        let normalized = value.lowercased().replacingOccurrences(of: "-", with: "_")
        return normalized == "qwen3_5"
            || normalized == "qwen3_5_text"
            || normalized == "qwen3_5_vl"
            || normalized == "qwen3_vl"
    }
}

/// Contract object for future MTP decode wiring. It intentionally contains no
/// MLX arrays because draft cache/state must stay separate from accepted base KV.
public struct MTPDraftStateContract: Codable, Sendable, Equatable {
    public let mode: MTPRuntimeMode
    public let draftTokenLimit: Int
    public let cacheIsSeparateFromBase: Bool
    public let acceptedTokensOnlyEnterBaseCache: Bool

    public init(
        mode: MTPRuntimeMode,
        draftTokenLimit: Int,
        cacheIsSeparateFromBase: Bool = true,
        acceptedTokensOnlyEnterBaseCache: Bool = true
    ) {
        self.mode = mode
        self.draftTokenLimit = draftTokenLimit
        self.cacheIsSeparateFromBase = cacheIsSeparateFromBase
        self.acceptedTokensOnlyEnterBaseCache = acceptedTokensOnlyEnterBaseCache
    }

    public static func inactive(mode: MTPRuntimeMode = .none) -> MTPDraftStateContract {
        MTPDraftStateContract(mode: mode, draftTokenLimit: 0)
    }
}

/// Cache commit policy for a future MTP verifier round.
public enum MTPBackboneCacheCommitPolicy: String, Codable, Sendable, Equatable {
    /// Only tokens verified and accepted by the backbone model may enter base KV,
    /// paged cache, disk L2, SSM companion state, or media-scoped cache state.
    case acceptedVerifierTokensOnly = "accepted_verifier_tokens_only"
}

/// Correct ways to preserve backbone cache state after a partially accepted MTP
/// verify round. All-or-nothing draft acceptance is not a valid D2/D3 semantic.
public enum MTPPartialAcceptCommitStrategy: String, Codable, Sendable, Equatable {
    /// Capture intermediate verifier KV/recurrent states and install the state
    /// matching the accepted draft prefix. This is the intended speed path.
    case captureCommit = "capture_commit"

    /// Roll back to the primary verifier state, then re-forward the accepted
    /// draft prefix through the target model. This is correctness-first and
    /// measurably slower when partial rejections occur.
    case rollbackRepair = "rollback_repair"
}

/// Required telemetry surface for any future native-MTP speed claim.
public struct MTPSpeedBenchRequirements: Codable, Sendable, Equatable {
    public let requiresARBaseline: Bool
    public let requiresMTPDepth: Bool
    public let requiresVerifyCalls: Bool
    public let requiresAcceptedDraftedByDepth: Bool
    public let requiresCommittedTokensPerVerify: Bool
    public let requiresBonusTokenCount: Bool
    public let requiresCorrectionCount: Bool
    public let requiresPhaseTiming: Bool
    public let requiresCacheMode: Bool
    public let requiresVerifyKernelMode: Bool
    public let requiresDraftHeadMode: Bool
    public let requiresOutputTailReview: Bool

    public init(
        requiresARBaseline: Bool = true,
        requiresMTPDepth: Bool = true,
        requiresVerifyCalls: Bool = true,
        requiresAcceptedDraftedByDepth: Bool = true,
        requiresCommittedTokensPerVerify: Bool = true,
        requiresBonusTokenCount: Bool = true,
        requiresCorrectionCount: Bool = true,
        requiresPhaseTiming: Bool = true,
        requiresCacheMode: Bool = true,
        requiresVerifyKernelMode: Bool = true,
        requiresDraftHeadMode: Bool = true,
        requiresOutputTailReview: Bool = true
    ) {
        self.requiresARBaseline = requiresARBaseline
        self.requiresMTPDepth = requiresMTPDepth
        self.requiresVerifyCalls = requiresVerifyCalls
        self.requiresAcceptedDraftedByDepth = requiresAcceptedDraftedByDepth
        self.requiresCommittedTokensPerVerify = requiresCommittedTokensPerVerify
        self.requiresBonusTokenCount = requiresBonusTokenCount
        self.requiresCorrectionCount = requiresCorrectionCount
        self.requiresPhaseTiming = requiresPhaseTiming
        self.requiresCacheMode = requiresCacheMode
        self.requiresVerifyKernelMode = requiresVerifyKernelMode
        self.requiresDraftHeadMode = requiresDraftHeadMode
        self.requiresOutputTailReview = requiresOutputTailReview
    }

    public static let nativeMTP = MTPSpeedBenchRequirements()
}

/// Depth-aware runtime contract for recursive MTP draft/verify.
///
/// This is a correctness contract, not an implementation switch. It exists so
/// status/UI/server wiring can distinguish a one-token logits-only experiment
/// from the real D2/D3 path needed for useful MTP speedups.
public struct MTPRecursiveDraftContract: Codable, Sendable, Equatable {
    public let depth: Int
    public let draftStepReturnsHiddenState: Bool
    public let verifierIncludesPrimaryPosition: Bool
    public let backboneCacheCommitPolicy: MTPBackboneCacheCommitPolicy
    public let draftCacheIsPrivate: Bool
    public let minAcceptedDraftTokensPerVerify: Int
    public let maxAcceptedDraftTokensPerVerify: Int
    public let requiresVariablePrefixCommit: Bool
    public let requiresCompiledVerifyHotPath: Bool
    public let requiresSmallMVerifyTuning: Bool
    public let partialAcceptCommitStrategy: MTPPartialAcceptCommitStrategy
    public let speedBenchRequirements: MTPSpeedBenchRequirements

    public init(
        depth: Int,
        draftStepReturnsHiddenState: Bool = true,
        verifierIncludesPrimaryPosition: Bool = true,
        backboneCacheCommitPolicy: MTPBackboneCacheCommitPolicy = .acceptedVerifierTokensOnly,
        draftCacheIsPrivate: Bool = true,
        minAcceptedDraftTokensPerVerify: Int = 0,
        maxAcceptedDraftTokensPerVerify: Int? = nil,
        requiresVariablePrefixCommit: Bool = true,
        requiresCompiledVerifyHotPath: Bool = true,
        requiresSmallMVerifyTuning: Bool = true,
        partialAcceptCommitStrategy: MTPPartialAcceptCommitStrategy = .captureCommit,
        speedBenchRequirements: MTPSpeedBenchRequirements = .nativeMTP
    ) {
        self.depth = max(0, depth)
        self.draftStepReturnsHiddenState = draftStepReturnsHiddenState
        self.verifierIncludesPrimaryPosition = verifierIncludesPrimaryPosition
        self.backboneCacheCommitPolicy = backboneCacheCommitPolicy
        self.draftCacheIsPrivate = draftCacheIsPrivate
        self.minAcceptedDraftTokensPerVerify = max(0, minAcceptedDraftTokensPerVerify)
        self.maxAcceptedDraftTokensPerVerify = max(
            0, min(maxAcceptedDraftTokensPerVerify ?? max(0, depth), max(0, depth)))
        self.requiresVariablePrefixCommit = requiresVariablePrefixCommit
        self.requiresCompiledVerifyHotPath = requiresCompiledVerifyHotPath
        self.requiresSmallMVerifyTuning = requiresSmallMVerifyTuning
        self.partialAcceptCommitStrategy = partialAcceptCommitStrategy
        self.speedBenchRequirements = speedBenchRequirements
    }

    /// Target verifier positions per round: primary position plus recursive
    /// draft positions. D3 verifies `[primary, d1, d2, d3]`.
    public var verifierPositionsPerCycle: Int {
        (verifierIncludesPrimaryPosition ? 1 : 0) + depth
    }

    /// Maximum emitted tokens per successful verify cycle: accepted drafts plus
    /// the target bonus token from the verifier.
    public var maxCommittedTokensPerVerify: Int {
        maxAcceptedDraftTokensPerVerify + (verifierIncludesPrimaryPosition ? 1 : 0)
    }

    public func fullAcceptanceVerifyCycles(forOutputTokens outputTokens: Int) -> Int {
        guard outputTokens > 0, maxCommittedTokensPerVerify > 0 else { return 0 }
        return Int(ceil(Double(outputTokens) / Double(maxCommittedTokensPerVerify)))
    }

    public static let mtplxDepth3 = MTPRecursiveDraftContract(depth: 3)
}

/// No-load MTP/VL inspector. It reads JSON metadata and safetensors headers only;
/// it never materializes tensors or changes the active generation path.
public enum MTPBundleInspector {
    public static func inspect(
        modelDirectory: URL,
        jangConfig suppliedJangConfig: JangConfig? = nil
    ) throws -> MTPBundleStatus {
        let config = try loadJSONObjectIfExists(modelDirectory.appendingPathComponent("config.json"))
        let jangConfig = suppliedJangConfig ?? (try? JangLoader.loadConfig(at: modelDirectory))
        let (configuredLayers, evidence, mtpLayerPrefixes) = configuredMTPLayers(
            config: config,
            jangConfig: jangConfig)
        let tensorNames = try loadTensorNames(from: modelDirectory)
        let mtpNames = tensorNames.filter { isMTPName($0, mtpLayerPrefixes: mtpLayerPrefixes) }
        let visionNames = tensorNames.filter(isVisionName)

        let runtimeMode = jangConfig?.runtime.mtpMode ?? .none
        let runtimeBundleHasMTP = jangConfig?.runtime.bundleHasMTP ?? false
        let metadataClaimsMTP = runtimeBundleHasMTP || runtimeMode != .none || configuredLayers > 0
        let bundleHasMTP = !mtpNames.isEmpty
        var statusEvidence = evidence
        if runtimeBundleHasMTP {
            statusEvidence.append("jang_config.runtime.bundle_has_mtp=true")
        }

        let mode: MTPRuntimeMode
        if !bundleHasMTP && metadataClaimsMTP {
            mode = .metadataOnlyMissingWeights
        } else if runtimeMode != .none {
            mode = runtimeMode
        } else if bundleHasMTP && configuredLayers > 0 {
            mode = .preservedEnabled
        } else {
            mode = .none
        }

        return MTPBundleStatus(
            bundleHasMTP: bundleHasMTP,
            configuredLayers: configuredLayers,
            tensorCount: mtpNames.count,
            visionTensorCount: visionNames.count,
            mode: mode,
            tensorSamples: Array(mtpNames.sorted().prefix(8)),
            visionTensorSamples: Array(visionNames.sorted().prefix(8)),
            configEvidence: Array(Set(statusEvidence)).sorted())
    }

    private static func configuredMTPLayers(
        config: [String: Any]?,
        jangConfig: JangConfig?
    ) -> (Int, [String], [String]) {
        var layers = 0
        var evidence: [String] = []
        var layerPrefixes: [String] = []

        func note(_ value: Int?, _ key: String) {
            guard let value, value > 0 else { return }
            layers = max(layers, value)
            evidence.append("\(key)=\(value)")
        }

        if let config {
            note(intValue(config["num_nextn_predict_layers"]), "num_nextn_predict_layers")
            note(intValue(config["mtp_num_hidden_layers"]), "mtp_num_hidden_layers")

            if let textConfig = config["text_config"] as? [String: Any] {
                note(
                    intValue(textConfig["num_nextn_predict_layers"]),
                    "text_config.num_nextn_predict_layers")
                note(
                    intValue(textConfig["mtp_num_hidden_layers"]),
                    "text_config.mtp_num_hidden_layers")
            }

            let topBaseLayers = intValue(config["num_hidden_layers"])
            let textConfig = config["text_config"] as? [String: Any]
            let textBaseLayers = intValue(textConfig?["num_hidden_layers"])
            for baseLayer in [topBaseLayers, textBaseLayers].compactMap({ $0 }) where baseLayer > 0 {
                layerPrefixes.append("model.layers.\(baseLayer).")
                layerPrefixes.append("language_model.model.layers.\(baseLayer).")
            }
        }

        if let runtimeLayers = jangConfig?.runtime.mtpLayers, runtimeLayers > 0 {
            layers = max(layers, runtimeLayers)
            evidence.append("jang_config.runtime.mtp_layers=\(runtimeLayers)")
        }

        return (layers, Array(Set(evidence)).sorted(), Array(Set(layerPrefixes)).sorted())
    }

    private static func loadTensorNames(from directory: URL) throws -> [String] {
        let indexURL = directory.appendingPathComponent("model.safetensors.index.json")
        if let index = try loadJSONObjectIfExists(indexURL),
            let weightMap = index["weight_map"] as? [String: Any]
        {
            return Array(weightMap.keys)
        }

        let contents =
            (try? FileManager.default.contentsOfDirectory(
                at: directory,
                includingPropertiesForKeys: [.isRegularFileKey],
                options: [.skipsHiddenFiles])) ?? []
        let safetensors = contents.filter { $0.pathExtension == "safetensors" }

        var names: [String] = []
        for file in safetensors {
            names.append(contentsOf: try safetensorsHeaderNames(file))
        }
        return names
    }

    private static func safetensorsHeaderNames(_ url: URL) throws -> [String] {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }

        guard let lengthData = try handle.read(upToCount: 8), lengthData.count == 8 else {
            return []
        }
        var headerLength: UInt64 = 0
        for (index, byte) in lengthData.enumerated() {
            headerLength |= UInt64(byte) << UInt64(index * 8)
        }
        guard headerLength > 0, headerLength <= 64 * 1024 * 1024 else {
            return []
        }
        guard let headerData = try handle.read(upToCount: Int(headerLength)),
            headerData.count == Int(headerLength),
            let header = try JSONSerialization.jsonObject(with: headerData) as? [String: Any]
        else {
            return []
        }
        return header.keys.filter { $0 != "__metadata__" }
    }

    private static func isMTPName(_ name: String, mtpLayerPrefixes: [String]) -> Bool {
        let lower = name.lowercased()
        if lower.hasPrefix("mtp.") || lower.hasPrefix("model.mtp_layers.") {
            return true
        }
        if lower.contains(".mtp.") || lower.contains(".mtp_layers.") {
            return true
        }
        if lower.contains("nextn") || lower.contains("next_n") {
            return true
        }
        return mtpLayerPrefixes.contains { name.hasPrefix($0) }
    }

    private static func isVisionName(_ name: String) -> Bool {
        let lower = name.lowercased()
        return lower.hasPrefix("vision_tower.")
            || lower.contains(".vision_tower.")
            || lower.hasPrefix("visual.")
            || lower.contains(".visual.")
            || lower.hasPrefix("vision_model.")
            || lower.contains(".vision_model.")
            || lower.hasPrefix("multi_modal_projector.")
            || lower.hasPrefix("mm_projector.")
            || lower.hasPrefix("image_newline")
    }

    private static func loadJSONObjectIfExists(_ url: URL) throws -> [String: Any]? {
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        let data = try Data(contentsOf: url)
        return try JSONSerialization.jsonObject(with: data) as? [String: Any]
    }

    private static func intValue(_ value: Any?) -> Int? {
        if let value = value as? Int { return value }
        if let value = value as? Double { return Int(value) }
        if let value = value as? Float { return Int(value) }
        if let value = value as? NSNumber { return value.intValue }
        if let value = value as? String { return Int(value) }
        return nil
    }
}
