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

    public init(
        configuration: ModelConfiguration,
        capabilities: JangCapabilities? = nil,
        modelType: String? = nil
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
        self.supportsAudio = Self.audioSupport(capabilities)
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
    }

    public init(
        resolvedConfiguration: ResolvedModelConfiguration,
        capabilities: JangCapabilities? = nil,
        modelType: String? = nil
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
        self.supportsAudio = Self.audioSupport(capabilities)
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
        self.reasoningParser = capabilities?.reasoningParser
            ?? resolvedConfiguration.reasoningParserName
        self.generationDefaults = resolvedConfiguration.generationDefaults
        self.nativeMTP = resolvedConfiguration.mtpStatus?.snapshot
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

private extension ModelRuntimeCapabilitySnapshot {
    static func textSupport(_ capabilities: JangCapabilities?) -> ModelRuntimeCapabilitySupport {
        ModelRuntimeCapabilitySupport.from(capabilities?.supportsText) ?? .supported
    }

    static func visionSupport(
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

    static func videoSupport(_ capabilities: JangCapabilities?) -> ModelRuntimeCapabilitySupport {
        if let explicit = ModelRuntimeCapabilitySupport.from(capabilities?.supportsVideo) {
            return explicit
        }
        return supportFromModality(
            capabilities?.modality,
            supportedTokens: ["video", "videos", "omni"],
            unsupportedTokens: ["text", "audio"])
    }

    static func audioSupport(_ capabilities: JangCapabilities?) -> ModelRuntimeCapabilitySupport {
        if let explicit = ModelRuntimeCapabilitySupport.from(capabilities?.supportsAudio) {
            return explicit
        }
        return supportFromModality(
            capabilities?.modality,
            supportedTokens: ["audio", "speech", "voice", "omni"],
            unsupportedTokens: ["text", "vision", "image", "images", "vl", "vlm", "video"])
    }

    static func toolSupport(
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

    static func reasoningSupport(
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

    static func supportFromModality(
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

    static func modalityTokens(_ modality: String?) -> Set<String> {
        guard let modality else { return [] }
        return Set(
            modality.lowercased().split { !$0.isLetter && !$0.isNumber }.map(String.init))
    }
}
