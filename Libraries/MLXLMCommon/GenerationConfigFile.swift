// Copyright © 2024 Apple Inc.

import Foundation

/// Declarative defaults passed to a tokenizer chat template when a request
/// does not provide an explicit value. Hugging Face bundles currently use
/// this object primarily for `enable_thinking`; keeping it typed prevents an
/// engine from silently dropping a model-authored reasoning default.
public struct ChatTemplateKwargsDefaults: Codable, Equatable, Sendable {
    public var enableThinking: Bool?

    public init(enableThinking: Bool? = nil) {
        self.enableThinking = enableThinking
    }

    enum CodingKeys: String, CodingKey {
        case enableThinking = "enable_thinking"
    }
}

/// JSON wrapper for `generation_config.json` file.
///
/// This file can override values from `config.json`, particularly `eos_token_id`.
/// Following mlx-lm Python behavior, if `generation_config.json` exists and contains
/// `eos_token_id`, it takes precedence over the value in `config.json`.
public struct GenerationConfigFile: Codable, Equatable, Sendable {
    public var eosTokenIds: IntOrIntArray?
    public var maxNewTokens: Int?
    public var temperature: Float?
    public var topP: Float?
    public var topK: Int?
    public var minP: Float?
    public var repetitionPenalty: Float?
    public var doSample: Bool?
    public var suppressTokens: [Int]?
    public var defaultChatTemplateKwargs: ChatTemplateKwargsDefaults?

    // Block-diffusion fields (DiffusionGemma). HF serializes the sampler
    // config as a nested object with a `_cls_name` discriminator; only the
    // payload values are decoded here.
    public var maxDenoisingSteps: Int?
    public var tMin: Float?
    public var tMax: Float?
    public var stabilityThreshold: Int?
    public var confidenceThreshold: Float?
    public var padTokenId: Int?
    public var samplerConfig: SamplerConfig?

    public struct SamplerConfig: Codable, Equatable, Sendable {
        public var entropyBound: Float?

        public init(entropyBound: Float? = nil) {
            self.entropyBound = entropyBound
        }

        enum CodingKeys: String, CodingKey {
            case entropyBound = "entropy_bound"
        }
    }

    public init(
        eosTokenIds: IntOrIntArray? = nil,
        maxNewTokens: Int? = nil,
        temperature: Float? = nil,
        topP: Float? = nil,
        topK: Int? = nil,
        minP: Float? = nil,
        repetitionPenalty: Float? = nil,
        doSample: Bool? = nil,
        suppressTokens: [Int]? = nil,
        defaultChatTemplateKwargs: ChatTemplateKwargsDefaults? = nil,
        maxDenoisingSteps: Int? = nil,
        tMin: Float? = nil,
        tMax: Float? = nil,
        stabilityThreshold: Int? = nil,
        confidenceThreshold: Float? = nil,
        padTokenId: Int? = nil,
        samplerConfig: SamplerConfig? = nil
    ) {
        self.eosTokenIds = eosTokenIds
        self.maxNewTokens = maxNewTokens
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.minP = minP
        self.repetitionPenalty = repetitionPenalty
        self.doSample = doSample
        self.suppressTokens = suppressTokens
        self.defaultChatTemplateKwargs = defaultChatTemplateKwargs
        self.maxDenoisingSteps = maxDenoisingSteps
        self.tMin = tMin
        self.tMax = tMax
        self.stabilityThreshold = stabilityThreshold
        self.confidenceThreshold = confidenceThreshold
        self.padTokenId = padTokenId
        self.samplerConfig = samplerConfig
    }

    enum CodingKeys: String, CodingKey {
        case eosTokenIds = "eos_token_id"
        case maxNewTokens = "max_new_tokens"
        case temperature
        case topP = "top_p"
        case topK = "top_k"
        case minP = "min_p"
        case repetitionPenalty = "repetition_penalty"
        case doSample = "do_sample"
        case suppressTokens = "suppress_tokens"
        case defaultChatTemplateKwargs = "default_chat_template_kwargs"
        case maxDenoisingSteps = "max_denoising_steps"
        case tMin = "t_min"
        case tMax = "t_max"
        case stabilityThreshold = "stability_threshold"
        case confidenceThreshold = "confidence_threshold"
        case padTokenId = "pad_token_id"
        case samplerConfig = "sampler_config"
    }
}

public enum ModelTokenConfigurationResolver {
    public static func resolvedEOSTokenIds(
        baseConfig: BaseConfiguration,
        configurationData: Data,
        generationConfig: GenerationConfigFile?
    ) -> Set<Int> {
        var eosTokenIds = Set(baseConfig.eosTokenIds?.values ?? [])
        if eosTokenIds.isEmpty,
           let textConfigEosTokenIds = Self.textConfigEOSTokenIds(
                configurationData: configurationData)
        {
            eosTokenIds = Set(textConfigEosTokenIds)
        }
        if let generationEosTokenIds = generationConfig?.eosTokenIds?.values {
            eosTokenIds = Set(generationEosTokenIds)
        }
        return eosTokenIds
    }

    private struct TextConfigTokens: Codable {
        let eosTokenIds: IntOrIntArray?

        enum CodingKeys: String, CodingKey {
            case eosTokenIds = "eos_token_id"
        }
    }

    private struct TextConfigWrapper: Codable {
        let textConfig: TextConfigTokens?

        enum CodingKeys: String, CodingKey {
            case textConfig = "text_config"
        }
    }

    private static func textConfigEOSTokenIds(configurationData: Data) -> [Int]? {
        guard let wrapper = try? JSONDecoder.json5().decode(
            TextConfigWrapper.self, from: configurationData)
        else {
            return nil
        }
        return wrapper.textConfig?.eosTokenIds?.values
    }
}
