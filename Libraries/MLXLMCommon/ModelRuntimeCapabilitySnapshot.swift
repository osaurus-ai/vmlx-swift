// Copyright © 2024 Apple Inc.

import Foundation

/// Tristate capability flag for host/UI status surfaces.
///
/// `unknown` is intentional: a parser or model-type heuristic is not the same
/// thing as a production claim that a model is trained for a modality.
public enum ModelRuntimeCapabilitySupport: String, Codable, Sendable, Equatable {
    case supported
    case unsupported
    case unknown

    static func from(_ value: Bool?) -> Self? {
        guard let value else { return nil }
        return value ? .supported : .unsupported
    }
}

/// Coarse no-load bundle format classification for status and UI routing.
public enum ModelRuntimeBundleFormat: String, Codable, Sendable, Equatable {
    case mlx
    case jang
    case jangtq
    case mxfp
    case unknown
}

/// Source-backed trace explaining how a local model bundle was classified.
///
/// This is intentionally metadata-only. It reads JSON files, the optional
/// JANGTQ runtime sidecar header, and the existing MTP bundle status. It does
/// not load model weights or claim that generation is production-ready.
public struct ModelRuntimeDetectionSnapshot: Codable, Sendable, Equatable {
    public let bundleFormat: ModelRuntimeBundleFormat
    public let evidence: [String]

    public let configModelType: String?
    public let textConfigModelType: String?
    public let dispatchModelType: String?
    public let hasTextConfig: Bool
    public let hasVisionConfig: Bool
    public let hasPreprocessorConfig: Bool

    public let configWeightFormat: String?
    public let textConfigWeightFormat: String?
    public let jangWeightFormat: String?
    public let effectiveWeightFormat: String?
    public let hasJangConfig: Bool
    public let jangFormat: String?
    public let jangProfile: String?
    public let jangQuantizationMethod: String?

    public let quantizationBits: Int?
    public let quantizationMode: String?
    public let textConfigQuantizationMode: String?
    public let mxtqBits: Int?
    public let mxtqBitsSource: String?
    public let hasJANGTQSidecar: Bool
    public let sidecarCodebookBits: Int?

    public let nativeMTPMode: MTPRuntimeMode?
    public let nativeMTPConfiguredLayers: Int?
    public let nativeMTPTensorCount: Int?
    public let visionTensorCount: Int?
    public let nativeMTPCanAutoLaunch: Bool?

    enum CodingKeys: String, CodingKey {
        case bundleFormat = "bundle_format"
        case evidence
        case configModelType = "config_model_type"
        case textConfigModelType = "text_config_model_type"
        case dispatchModelType = "dispatch_model_type"
        case hasTextConfig = "has_text_config"
        case hasVisionConfig = "has_vision_config"
        case hasPreprocessorConfig = "has_preprocessor_config"
        case configWeightFormat = "config_weight_format"
        case textConfigWeightFormat = "text_config_weight_format"
        case jangWeightFormat = "jang_weight_format"
        case effectiveWeightFormat = "effective_weight_format"
        case hasJangConfig = "has_jang_config"
        case jangFormat = "jang_format"
        case jangProfile = "jang_profile"
        case jangQuantizationMethod = "jang_quantization_method"
        case quantizationBits = "quantization_bits"
        case quantizationMode = "quantization_mode"
        case textConfigQuantizationMode = "text_config_quantization_mode"
        case mxtqBits = "mxtq_bits"
        case mxtqBitsSource = "mxtq_bits_source"
        case hasJANGTQSidecar = "has_jangtq_sidecar"
        case sidecarCodebookBits = "sidecar_codebook_bits"
        case nativeMTPMode = "native_mtp_mode"
        case nativeMTPConfiguredLayers = "native_mtp_configured_layers"
        case nativeMTPTensorCount = "native_mtp_tensor_count"
        case visionTensorCount = "vision_tensor_count"
        case nativeMTPCanAutoLaunch = "native_mtp_can_auto_launch"
    }

    public init(
        modelDirectory: URL,
        mtpStatus suppliedMTPStatus: MTPBundleStatus? = nil
    ) throws {
        let config = try Self.loadJSONObjectIfExists(
            modelDirectory.appendingPathComponent("config.json"))
        let jang = try Self.loadJSONObjectIfExists(
            modelDirectory.appendingPathComponent("jang_config.json"))
        let textConfig = config?["text_config"] as? [String: Any]
        let quantization = config?["quantization"] as? [String: Any]
        let textConfigQuantization = textConfig?["quantization"] as? [String: Any]
        let jangQuantization = jang?["quantization"] as? [String: Any]
        let sidecarURL = modelDirectory.appendingPathComponent("jangtq_runtime.safetensors")
        let hasSidecar = FileManager.default.fileExists(atPath: sidecarURL.path)
        let sidecarBits = hasSidecar ? JANGTQRuntimeCache.sniffCodebookBits(at: sidecarURL) : nil
        let mtpStatus =
            suppliedMTPStatus
            ?? (try? MTPBundleInspector.inspect(
                modelDirectory: modelDirectory))

        let configWeightFormat = Self.lowerString(config?["weight_format"])
        let textConfigWeightFormat = Self.lowerString(textConfig?["weight_format"])
        let jangWeightFormat = Self.lowerString(jang?["weight_format"])
        let effectiveWeightFormat = jangWeightFormat ?? configWeightFormat ?? textConfigWeightFormat
        let configModelType = Self.string(config?["model_type"])
        let textConfigModelType = Self.string(textConfig?["model_type"])
        let hasVisionConfig =
            Self.bool(config?["vision_config"])
            || (config?["vision_config"] as? [String: Any]) != nil
        let hasPreprocessorConfig =
            FileManager.default.fileExists(
                atPath: modelDirectory.appendingPathComponent("preprocessor_config.json").path)
            || FileManager.default.fileExists(
                atPath: modelDirectory.appendingPathComponent("processor_config.json").path)
            || FileManager.default.fileExists(
                atPath: modelDirectory.appendingPathComponent("video_preprocessor_config.json").path
            )

        let configMxtq = Self.resolveMXTQBits(
            config?["mxtq_bits"],
            preferredSources: ["config.mxtq_bits"])
        let jangMxtq = Self.resolveMXTQBits(
            jang?["mxtq_bits"],
            preferredSources: ["jang_config.mxtq_bits"])
        let quantizationBits = Self.intValue(quantization?["bits"])
        let quantizationMode = Self.lowerString(quantization?["mode"])
        let textConfigQuantizationMode = Self.lowerString(textConfigQuantization?["mode"])
        let jangProfile = Self.string(jangQuantization?["profile"])
        let profileBits = jangtqBitsFromProfile(jangProfile)
        let resolvedBits: (Int?, String?) = {
            if let configMxtq { return (configMxtq.value, configMxtq.source) }
            if let jangMxtq { return (jangMxtq.value, jangMxtq.source) }
            if let sidecarBits { return (sidecarBits, "jangtq_runtime.safetensors.codebook") }
            if let profileBits { return (profileBits, "jang_config.quantization.profile") }
            return (nil, nil)
        }()

        var evidence: [String] = []
        if config != nil { evidence.append("config.json") }
        if jang != nil { evidence.append("jang_config.json") }
        if let configModelType { evidence.append("config.model_type=\(configModelType)") }
        if let textConfigModelType {
            evidence.append("text_config.model_type=\(textConfigModelType)")
        }
        if let configWeightFormat { evidence.append("config.weight_format=\(configWeightFormat)") }
        if let textConfigWeightFormat {
            evidence.append("text_config.weight_format=\(textConfigWeightFormat)")
        }
        if let jangWeightFormat { evidence.append("jang_config.weight_format=\(jangWeightFormat)") }
        if let quantizationMode { evidence.append("config.quantization.mode=\(quantizationMode)") }
        if let textConfigQuantizationMode {
            evidence.append("text_config.quantization.mode=\(textConfigQuantizationMode)")
        }
        if hasSidecar { evidence.append("jangtq_runtime.safetensors") }
        if let sidecarBits { evidence.append("sidecar.codebook_bits=\(sidecarBits)") }
        if let profileBits { evidence.append("jang_config.profile_bits=\(profileBits)") }
        if let mode = mtpStatus?.mode { evidence.append("native_mtp.mode=\(mode.rawValue)") }
        if mtpStatus?.bundleHasVision == true { evidence.append("native_mtp.vision_tensors=true") }
        if hasPreprocessorConfig { evidence.append("preprocessor_config=true") }

        let format: ModelRuntimeBundleFormat
        if effectiveWeightFormat == "mxtq" || resolvedBits.0 != nil || hasSidecar {
            format = .jangtq
        } else if Self.isMXFP(effectiveWeightFormat)
            || Self.isMXFP(quantizationMode)
            || Self.isMXFP(textConfigQuantizationMode)
        {
            format = .mxfp
        } else if jang != nil {
            format = .jang
        } else if config != nil {
            format = .mlx
        } else {
            format = .unknown
        }

        self.bundleFormat = format
        self.evidence = Array(Set(evidence)).sorted()
        self.configModelType = configModelType
        self.textConfigModelType = textConfigModelType
        self.dispatchModelType = textConfigModelType ?? configModelType
        self.hasTextConfig = textConfig != nil
        self.hasVisionConfig = hasVisionConfig || (mtpStatus?.bundleHasVision == true)
        self.hasPreprocessorConfig = hasPreprocessorConfig
        self.configWeightFormat = configWeightFormat
        self.textConfigWeightFormat = textConfigWeightFormat
        self.jangWeightFormat = jangWeightFormat
        self.effectiveWeightFormat = effectiveWeightFormat
        self.hasJangConfig = jang != nil
        self.jangFormat = Self.string(jang?["format"])
        self.jangProfile = jangProfile
        self.jangQuantizationMethod = Self.string(jangQuantization?["method"])
        self.quantizationBits = quantizationBits
        self.quantizationMode = quantizationMode
        self.textConfigQuantizationMode = textConfigQuantizationMode
        self.mxtqBits = resolvedBits.0
        self.mxtqBitsSource = resolvedBits.1
        self.hasJANGTQSidecar = hasSidecar
        self.sidecarCodebookBits = sidecarBits
        self.nativeMTPMode = mtpStatus?.mode
        self.nativeMTPConfiguredLayers = mtpStatus?.configuredLayers
        self.nativeMTPTensorCount = mtpStatus?.tensorCount
        self.visionTensorCount = mtpStatus?.visionTensorCount
        self.nativeMTPCanAutoLaunch = mtpStatus?.canAutoLaunchMTP
    }

    private static func resolveMXTQBits(
        _ value: Any?,
        preferredSources: [String]
    ) -> (value: Int, source: String)? {
        if let int = intValue(value) {
            return (int, preferredSources[0])
        }
        guard let dict = value as? [String: Any] else { return nil }
        if let routed = intValue(dict["routed_expert"]) {
            return (routed, "\(preferredSources[0]).routed_expert")
        }
        if let routed = dict["routed_expert"] as? [String: Any] {
            for key in ["gate_proj", "up_proj", "down_proj"] {
                if let bits = intValue(routed[key]) {
                    return (bits, "\(preferredSources[0]).routed_expert.\(key)")
                }
            }
        }
        for key in ["gate_proj", "up_proj", "down_proj", "q_proj", "k_proj", "v_proj", "o_proj"] {
            if let bits = intValue(dict[key]) {
                return (bits, "\(preferredSources[0]).\(key)")
            }
        }
        return nil
    }

    private static func loadJSONObjectIfExists(_ url: URL) throws -> [String: Any]? {
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        let data = try Data(contentsOf: url)
        return try JSONSerialization.jsonObject(with: data) as? [String: Any]
    }

    private static func string(_ value: Any?) -> String? {
        guard let value = value as? String, !value.isEmpty else { return nil }
        return value
    }

    private static func lowerString(_ value: Any?) -> String? {
        string(value)?.lowercased()
    }

    private static func bool(_ value: Any?) -> Bool {
        if let value = value as? Bool { return value }
        return value != nil
    }

    private static func isMXFP(_ value: String?) -> Bool {
        guard let value else { return false }
        return value.contains("mxfp") || value.contains("mx-fp")
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

/// Codable capability snapshot for Osaurus-style model status endpoints.
///
/// This is a no-load data surface. It combines the resolved model configuration,
/// JANG capability stamps, bundle `generation_config.json` defaults, and native
/// MTP status into a single JSON shape without changing decode behavior.
public struct ModelRuntimeCapabilitySnapshot: Codable, Sendable, Equatable {
    public let modelName: String
    public let modelType: String?
    public let family: String?
    public let modality: String?
    public let cacheType: String?

    public let supportsText: ModelRuntimeCapabilitySupport
    public let supportsVision: ModelRuntimeCapabilitySupport
    public let supportsVideo: ModelRuntimeCapabilitySupport
    public let supportsAudio: ModelRuntimeCapabilitySupport
    public let supportsTools: ModelRuntimeCapabilitySupport
    public let supportsReasoning: ModelRuntimeCapabilitySupport
    public let supportsNativeMTP: ModelRuntimeCapabilitySupport

    public let thinkInTemplate: Bool?
    public let toolParser: String?
    public let reasoningParser: String?
    public let generationDefaults: GenerationConfigFile?
    public let nativeMTP: MTPBundleStatusSnapshot?
    public let detection: ModelRuntimeDetectionSnapshot?

    public init(
        configuration: ModelConfiguration,
        capabilities: JangCapabilities? = nil,
        modelType: String? = nil,
        detection: ModelRuntimeDetectionSnapshot? = nil
    ) {
        self.modelName = configuration.name
        self.modelType = modelType
        self.family = capabilities?.family
        self.modality = capabilities?.modality
        self.cacheType = capabilities?.cacheType
        self.supportsText = Self.textSupport(capabilities)
        self.supportsVision = Self.visionSupport(
            capabilities: capabilities,
            mtpStatus: configuration.mtpStatus)
        self.supportsVideo = Self.videoSupport(capabilities)
        self.supportsAudio = Self.audioSupport(capabilities, modelType: modelType)
        self.supportsTools = Self.toolSupport(
            capabilities: capabilities,
            toolCallFormat: configuration.toolCallFormat)
        self.supportsReasoning = Self.reasoningSupport(
            capabilities: capabilities,
            reasoningParserName: configuration.reasoningParserName)
        self.supportsNativeMTP =
            configuration.mtpStatus?.canAutoLaunchMTP == true ? .supported : .unsupported
        self.thinkInTemplate = capabilities?.thinkInTemplate
        self.toolParser = capabilities?.toolParser ?? configuration.toolCallFormat?.rawValue
        self.reasoningParser = capabilities?.reasoningParser ?? configuration.reasoningParserName
        self.generationDefaults = configuration.generationDefaults
        self.nativeMTP = configuration.mtpStatus?.snapshot
        self.detection =
            detection
            ?? (try? configuration.modelDirectory).flatMap {
                try? ModelRuntimeDetectionSnapshot(
                    modelDirectory: $0,
                    mtpStatus: configuration.mtpStatus)
            }
    }

    public init(
        resolvedConfiguration: ResolvedModelConfiguration,
        capabilities: JangCapabilities? = nil,
        modelType: String? = nil,
        detection: ModelRuntimeDetectionSnapshot? = nil
    ) {
        self.modelName = resolvedConfiguration.name
        self.modelType = modelType
        self.family = capabilities?.family
        self.modality = capabilities?.modality
        self.cacheType = capabilities?.cacheType
        self.supportsText = Self.textSupport(capabilities)
        self.supportsVision = Self.visionSupport(
            capabilities: capabilities,
            mtpStatus: resolvedConfiguration.mtpStatus)
        self.supportsVideo = Self.videoSupport(capabilities)
        self.supportsAudio = Self.audioSupport(capabilities, modelType: modelType)
        self.supportsTools = Self.toolSupport(
            capabilities: capabilities,
            toolCallFormat: resolvedConfiguration.toolCallFormat)
        self.supportsReasoning = Self.reasoningSupport(
            capabilities: capabilities,
            reasoningParserName: resolvedConfiguration.reasoningParserName)
        self.supportsNativeMTP =
            resolvedConfiguration.mtpStatus?.canAutoLaunchMTP == true ? .supported : .unsupported
        self.thinkInTemplate = capabilities?.thinkInTemplate
        self.toolParser = capabilities?.toolParser ?? resolvedConfiguration.toolCallFormat?.rawValue
        self.reasoningParser =
            capabilities?.reasoningParser
            ?? resolvedConfiguration.reasoningParserName
        self.generationDefaults = resolvedConfiguration.generationDefaults
        self.nativeMTP = resolvedConfiguration.mtpStatus?.snapshot
        self.detection =
            detection
            ?? (try? ModelRuntimeDetectionSnapshot(
                modelDirectory: resolvedConfiguration.modelDirectory,
                mtpStatus: resolvedConfiguration.mtpStatus))
    }

    enum CodingKeys: String, CodingKey {
        case modelName = "model_name"
        case modelType = "model_type"
        case family
        case modality
        case cacheType = "cache_type"
        case supportsText = "supports_text"
        case supportsVision = "supports_vision"
        case supportsVideo = "supports_video"
        case supportsAudio = "supports_audio"
        case supportsTools = "supports_tools"
        case supportsReasoning = "supports_reasoning"
        case supportsNativeMTP = "supports_native_mtp"
        case thinkInTemplate = "think_in_template"
        case toolParser = "tool_parser"
        case reasoningParser = "reasoning_parser"
        case generationDefaults = "generation_defaults"
        case nativeMTP = "native_mtp"
        case detection
    }

    public func validate(
        request: ModelRuntimeCapabilityRequest,
        unknownPolicy: ModelRuntimeCapabilityValidationPolicy = .rejectUnknown
    ) -> ModelRuntimeCapabilityValidationResult {
        var issues: [ModelRuntimeCapabilityIssue] = []
        for modality in request.sortedModalities {
            let support = support(for: modality)
            switch support {
            case .supported:
                continue
            case .unsupported:
                issues.append(.unsupported(modality: modality, support: support))
            case .unknown:
                if unknownPolicy == .rejectUnknown {
                    issues.append(.unknown(modality: modality, support: support))
                }
            }
        }
        return ModelRuntimeCapabilityValidationResult(
            requestedModalities: request.sortedModalities,
            issues: issues)
    }

    private func support(
        for modality: ModelRuntimeRequestModality
    ) -> ModelRuntimeCapabilitySupport {
        switch modality {
        case .text:
            supportsText
        case .vision:
            supportsVision
        case .video:
            supportsVideo
        case .audio:
            supportsAudio
        case .tools:
            supportsTools
        case .reasoning:
            supportsReasoning
        case .nativeMTP:
            supportsNativeMTP
        }
    }
}

public enum ModelRuntimeRequestModality: String, Codable, Sendable, Equatable, CaseIterable {
    case text
    case vision
    case video
    case audio
    case tools
    case reasoning
    case nativeMTP = "native_mtp"
}

public struct ModelRuntimeCapabilityRequest: Codable, Sendable, Equatable {
    public let modalities: Set<ModelRuntimeRequestModality>

    public init(modalities: Set<ModelRuntimeRequestModality>) {
        self.modalities = modalities
    }

    public init(
        input: UserInput,
        usesReasoning: Bool = false,
        usesNativeMTP: Bool = false
    ) {
        var modalities: Set<ModelRuntimeRequestModality> = [.text]
        if !input.images.isEmpty {
            modalities.insert(.vision)
        }
        if !input.videos.isEmpty {
            modalities.insert(.video)
        }
        if !input.audios.isEmpty {
            modalities.insert(.audio)
        }
        if input.tools?.isEmpty == false {
            modalities.insert(.tools)
        }
        if usesReasoning {
            modalities.insert(.reasoning)
        }
        if usesNativeMTP {
            modalities.insert(.nativeMTP)
        }
        self.modalities = modalities
    }

    public var sortedModalities: [ModelRuntimeRequestModality] {
        modalities.sorted { lhs, rhs in
            Self.sortIndex(lhs) < Self.sortIndex(rhs)
        }
    }

    private static func sortIndex(_ modality: ModelRuntimeRequestModality) -> Int {
        switch modality {
        case .text:
            0
        case .vision:
            1
        case .video:
            2
        case .audio:
            3
        case .tools:
            4
        case .reasoning:
            5
        case .nativeMTP:
            6
        }
    }
}

public enum ModelRuntimeCapabilityValidationPolicy: String, Codable, Sendable, Equatable {
    case rejectUnknown = "reject_unknown"
    case allowUnknown = "allow_unknown"
}

public struct ModelRuntimeCapabilityIssue: Codable, Sendable, Equatable {
    public let code: String
    public let modality: ModelRuntimeRequestModality
    public let support: ModelRuntimeCapabilitySupport
    public let message: String
    public let redactedLogFields: [String: String]

    public init(
        code: String,
        modality: ModelRuntimeRequestModality,
        support: ModelRuntimeCapabilitySupport,
        message: String,
        redactedLogFields: [String: String]
    ) {
        self.code = code
        self.modality = modality
        self.support = support
        self.message = message
        self.redactedLogFields = redactedLogFields
    }

    public static func unsupported(
        modality: ModelRuntimeRequestModality,
        support: ModelRuntimeCapabilitySupport
    ) -> Self {
        Self(
            code: "unsupported_modality",
            modality: modality,
            support: support,
            message: "Model capability snapshot reports \(modality.rawValue) as unsupported.",
            redactedLogFields: [
                "code": "unsupported_modality",
                "modality": modality.rawValue,
                "support": support.rawValue,
            ])
    }

    public static func unknown(
        modality: ModelRuntimeRequestModality,
        support: ModelRuntimeCapabilitySupport
    ) -> Self {
        Self(
            code: "unknown_modality_support",
            modality: modality,
            support: support,
            message: "Model capability snapshot does not prove \(modality.rawValue) support.",
            redactedLogFields: [
                "code": "unknown_modality_support",
                "modality": modality.rawValue,
                "support": support.rawValue,
            ])
    }

    public static func disabledByServerSettings(
        modality: ModelRuntimeRequestModality,
        field: String
    ) -> Self {
        Self(
            code: "server_modality_disabled",
            modality: modality,
            support: .unsupported,
            message: "Server settings disable \(modality.rawValue) for this request.",
            redactedLogFields: [
                "code": "server_modality_disabled",
                "field": field,
                "modality": modality.rawValue,
                "support": ModelRuntimeCapabilitySupport.unsupported.rawValue,
            ])
    }

    enum CodingKeys: String, CodingKey {
        case code
        case modality
        case support
        case message
        case redactedLogFields = "redacted_log_fields"
    }
}

public struct ModelRuntimeCapabilityValidationResult: Codable, Sendable, Equatable {
    public let allowed: Bool
    public let requestedModalities: [ModelRuntimeRequestModality]
    public let issues: [ModelRuntimeCapabilityIssue]

    public init(
        requestedModalities: [ModelRuntimeRequestModality],
        issues: [ModelRuntimeCapabilityIssue]
    ) {
        self.allowed = issues.isEmpty
        self.requestedModalities = requestedModalities
        self.issues = issues
    }

    enum CodingKeys: String, CodingKey {
        case allowed
        case requestedModalities = "requested_modalities"
        case issues
    }
}

extension ModelRuntimeCapabilitySnapshot {
    fileprivate static func textSupport(_ capabilities: JangCapabilities?)
        -> ModelRuntimeCapabilitySupport
    {
        ModelRuntimeCapabilitySupport.from(capabilities?.supportsText) ?? .supported
    }

    fileprivate static func visionSupport(
        capabilities: JangCapabilities?,
        mtpStatus: MTPBundleStatus?
    ) -> ModelRuntimeCapabilitySupport {
        if let explicit = ModelRuntimeCapabilitySupport.from(capabilities?.supportsVision) {
            return explicit
        }
        if mtpStatus?.bundleHasVision == true {
            return .supported
        }
        return supportFromModality(
            capabilities?.modality,
            supportedTokens: ["vision", "image", "images", "vl", "vlm", "multimodal", "omni"],
            unsupportedTokens: ["text"])
    }

    fileprivate static func videoSupport(_ capabilities: JangCapabilities?)
        -> ModelRuntimeCapabilitySupport
    {
        if let explicit = ModelRuntimeCapabilitySupport.from(capabilities?.supportsVideo) {
            return explicit
        }
        return supportFromModality(
            capabilities?.modality,
            supportedTokens: ["video", "videos", "omni"],
            unsupportedTokens: ["text", "audio"])
    }

    fileprivate static func audioSupport(
        _ capabilities: JangCapabilities?,
        modelType: String?
    )
        -> ModelRuntimeCapabilitySupport
    {
        if let explicit = ModelRuntimeCapabilitySupport.from(capabilities?.supportsAudio) {
            return explicit
        }
        if let modelType = modelType?.lowercased(),
            ["nemotron_dense_audex", "nemotron_h_audex"].contains(modelType)
        {
            return .supported
        }
        return supportFromModality(
            capabilities?.modality,
            supportedTokens: ["audio", "speech", "voice", "omni"],
            unsupportedTokens: ["text", "vision", "image", "images", "vl", "vlm", "video"])
    }

    fileprivate static func toolSupport(
        capabilities: JangCapabilities?,
        toolCallFormat: ToolCallFormat?
    ) -> ModelRuntimeCapabilitySupport {
        if let explicit = ModelRuntimeCapabilitySupport.from(capabilities?.supportsTools) {
            return explicit
        }
        if capabilities?.toolParser != nil || toolCallFormat != nil {
            return .unknown
        }
        return .unknown
    }

    fileprivate static func reasoningSupport(
        capabilities: JangCapabilities?,
        reasoningParserName: String?
    ) -> ModelRuntimeCapabilitySupport {
        if let explicit = ModelRuntimeCapabilitySupport.from(capabilities?.supportsThinking) {
            return explicit
        }
        let stamp = capabilities?.reasoningParser ?? reasoningParserName
        guard let stamp else { return .unknown }
        return ReasoningParser.fromCapabilityName(stamp) == nil ? .unsupported : .supported
    }

    fileprivate static func supportFromModality(
        _ modality: String?,
        supportedTokens: Set<String>,
        unsupportedTokens: Set<String>
    ) -> ModelRuntimeCapabilitySupport {
        let tokens = modalityTokens(modality)
        if !tokens.isDisjoint(with: supportedTokens) {
            return .supported
        }
        if !tokens.isDisjoint(with: unsupportedTokens) {
            return .unsupported
        }
        return .unknown
    }

    fileprivate static func modalityTokens(_ modality: String?) -> Set<String> {
        guard let modality else { return [] }
        return Set(
            modality.lowercased().split { !$0.isLetter && !$0.isNumber }.map(String.init))
    }
}
