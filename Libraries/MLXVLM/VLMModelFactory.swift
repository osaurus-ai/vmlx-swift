// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXLMCommon

public enum VLMError: LocalizedError, Equatable {
    case imageRequired
    case maskRequired
    case singleImageAllowed
    case singleVideoAllowed
    case singleMediaTypeAllowed
    case imageProcessingFailure(String)
    case processing(String)
    case noVideoTrackFound
    case videoNotDecodable

    public var errorDescription: String? {
        switch self {
        case .imageRequired:
            return String(localized: "An image is required for this operation.")
        case .maskRequired:
            return String(localized: "An image mask is required for this operation.")
        case .singleImageAllowed:
            return String(localized: "Only a single image is allowed for this operation.")
        case .singleVideoAllowed:
            return String(localized: "Only a single video is allowed for this operation.")
        case .singleMediaTypeAllowed:
            return String(
                localized:
                    "Only a single media type (image or video) is allowed for this operation.")
        case .imageProcessingFailure(let details):
            return String(localized: "Failed to process the image: \(details)")
        case .processing(let details):
            return String(localized: "Processing error: \(details)")
        case .noVideoTrackFound:
            return String(localized: "Video file has no video tracks.")
        case .videoNotDecodable:
            return String(localized: "Video file not decodable.")
        }
    }
}

public struct BaseProcessorConfiguration: Codable, Sendable {
    public let processorClass: String

    enum CodingKeys: String, CodingKey {
        case processorClass = "processor_class"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        // Optional: bundles like Nemotron-Omni use `image_processor_type` and
        // have no `processor_class`. Default to empty so the model_type
        // override path can pick a processor; this is a no-op for bundles
        // that ship a real `processor_class`.
        self.processorClass = try c.decodeIfPresent(String.self, forKey: .processorClass) ?? ""
    }

    public init(processorClass: String) {
        self.processorClass = processorClass
    }
}

/// Creates a function that loads a configuration file and instantiates a model with the proper configuration
private func create<C: Codable, M>(
    _ configurationType: C.Type, _ modelInit: @escaping (C) -> M
) -> (Data) throws -> M {
    { data in
        let configuration = try JSONDecoder.json5().decode(C.self, from: data)
        return modelInit(configuration)
    }
}

private func chatTemplateText(modelDirectory: URL) -> String? {
    let templateURL = modelDirectory.appending(component: "chat_template.jinja")
    if let template = try? String(contentsOf: templateURL, encoding: .utf8) {
        return template
    }
    let tokenizerConfigURL = modelDirectory.appending(component: "tokenizer_config.json")
    guard
        let data = try? Data(contentsOf: tokenizerConfigURL),
        let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
        let template = object["chat_template"] as? String
    else {
        return nil
    }
    return template
}

private func create<C: Codable, P>(
    _ configurationType: C.Type,
    _ processorInit:
        @escaping (
            C,
            any Tokenizer
        ) -> P
) -> (Data, any Tokenizer) throws -> P {
    { data, tokenizer in
        let configuration = try JSONDecoder.json5().decode(C.self, from: data)
        return processorInit(configuration, tokenizer)
    }
}

/// Registry of model type, e.g 'llama', to functions that can instantiate the model from configuration.
///
/// Typically called via ``LLMModelFactory/load(from:configuration:progressHandler:)``.
public enum VLMTypeRegistry {

    /// The set of model type strings supported by the VLM factory.
    /// Use this to check if a model_type from config.json is a known VLM architecture.
    public static let supportedModelTypes: Set<String> = Set(_creators.keys)

    /// Shared instance with default model types.
    public static let shared: ModelTypeRegistry = .init(creators: _creators)

    nonisolated(unsafe) private static let _creators: [String: (Data) throws -> any LanguageModel] = [
        "paligemma": create(PaliGemmaConfiguration.self, PaliGemma.init),
        // Block-diffusion Gemma: the MLXLLM engine with the Gemma4 vision
        // tower installed. Generation runs via BlockDiffusionTokenIterator;
        // the throwing prepare() guard keeps AR routes loud.
        "diffusion_gemma": create(
            DiffusionGemmaVLMConfiguration.self, makeDiffusionGemmaVLM),
        "qwen2_vl": create(Qwen2VLConfiguration.self, Qwen2VL.init),
        "qwen2_5_vl": create(Qwen25VLConfiguration.self, Qwen25VL.init),
        "qwen3_vl": create(Qwen3VLConfiguration.self, Qwen3VL.init),
        "qwen3_5": create(Qwen35Configuration.self, Qwen35.init),
        "qwen3_5_moe": create(Qwen35Configuration.self, Qwen35MoE.init),
        "idefics3": create(Idefics3Configuration.self, Idefics3.init),
        "gemma3": create(Gemma3Configuration.self, Gemma3.init),
        "smolvlm": create(SmolVLM2Configuration.self, SmolVLM2.init),
        // TODO: see if we can make it work with fastvlm rather than llava_qwen2
        "fastvlm": create(FastVLMConfiguration.self, FastVLM.init),
        "llava_qwen2": create(FastVLMConfiguration.self, FastVLM.init),
        "pixtral": create(PixtralConfiguration.self, PixtralVLM.init),
        "mistral3": dispatchMistral3VLM,
        // Mistral 3.5 VLM bundles can expose `ministral3` at the OUTER
        // level (not just as the inner text_config.model_type). Register
        // both keys to the same dispatch.
        "ministral3": dispatchMistral3VLM,
        "lfm2_vl": create(LFM2VLConfiguration.self, LFM2VL.init),
        "lfm2-vl": create(LFM2VLConfiguration.self, LFM2VL.init),
        "glm_ocr": create(GlmOcrConfiguration.self, GlmOcr.init),
        // DeepSeek-OCR / Unlimited-OCR (top model_type "deepseek_vl_v2").
        "deepseek_vl_v2": create(DeepseekOCRConfiguration.self, DeepseekOCR.init),
        "deepseekocr": create(DeepseekOCRConfiguration.self, DeepseekOCR.init),
        "gemma4": create(Gemma4Configuration.self, Gemma4.init),
        "gemma4_unified": create(Gemma4Configuration.self, Gemma4.init),
        "nemotron_h_omni": create(NemotronHOmniConfiguration.self, NemotronHOmni.init),
        "NemotronH_Nano_Omni_Reasoning_V3":
            create(NemotronHOmniConfiguration.self, NemotronHOmni.init),
        "zaya1_vl": dispatchZaya1VL,
    ]

    private static func dispatchZaya1VL(data: Data) throws -> any LanguageModel {
        let configuration = try JSONDecoder.json5().decode(Zaya1VLConfiguration.self, from: data)
        return try Zaya1VL(configuration)
    }

    /// Shared dispatch for Mistral 3 / 3.5 VLM bundles, registered under
    /// both `mistral3` (canonical outer model_type) and `ministral3`
    /// (occasional outer spelling on Mistral 3.5 LLM-with-vision bundles).
    ///
    ///   1. `weight_format == "mxtq"` → `Mistral3VLMJANGTQ` (JANGTQ inner LM,
    ///      vanilla Pixtral vision tower per `mxtq_bits.vision_tower=
    ///      passthrough_fp16`).
    ///   2. `text_config.model_type == "mistral4"` → `Mistral4VLM` (Mistral 3
    ///      VLM wrapping a Mistral 4 text decoder).
    ///   3. Otherwise → vanilla `Mistral3VLM`.
    static func dispatchMistral3VLM(data: Data) throws -> any LanguageModel {
        struct WFCheck: Codable {
            let weightFormat: String?
            enum CodingKeys: String, CodingKey { case weightFormat = "weight_format" }
        }
        if let wf = try? JSONDecoder.json5().decode(WFCheck.self, from: data),
            wf.weightFormat?.lowercased() == "mxtq"
        {
            let config = try JSONDecoder.json5().decode(
                Mistral3VLMConfiguration.self, from: data)
            return Mistral3VLMJANGTQ(
                config,
                bits: config.mxtqBits ?? 2,
                seed: config.mxtqSeed ?? 42
            )
        }
        struct TextCheck: Codable {
            let textConfig: TextType?
            struct TextType: Codable {
                let modelType: String?
                enum CodingKeys: String, CodingKey { case modelType = "model_type" }
            }
            enum CodingKeys: String, CodingKey { case textConfig = "text_config" }
        }
        if let check = try? JSONDecoder.json5().decode(TextCheck.self, from: data),
            check.textConfig?.modelType == "mistral4"
        {
            let config = try JSONDecoder.json5().decode(Mistral4VLMConfiguration.self, from: data)
            return Mistral4VLM(config)
        }
        let config = try JSONDecoder.json5().decode(Mistral3VLMConfiguration.self, from: data)
        return Mistral3VLM(config)
    }
}

public enum VLMProcessorTypeRegistry {

    /// Shared instance with default processor types.
    public static let shared: ProcessorTypeRegistry = .init(creators: [
        "PaliGemmaProcessor": create(
            PaliGemmaProcessorConfiguration.self, PaliGemmaProcessor.init),
        "Qwen2VLProcessor": create(
            Qwen2VLProcessorConfiguration.self, Qwen2VLProcessor.init),
        "Qwen2_5_VLProcessor": create(
            Qwen25VLProcessorConfiguration.self, Qwen25VLProcessor.init),
        "Qwen3VLProcessor": create(
            Qwen3VLProcessorConfiguration.self, Qwen3VLProcessor.init),
        "Idefics3Processor": create(
            Idefics3ProcessorConfiguration.self, Idefics3Processor.init),
        "Gemma3Processor": create(
            Gemma3ProcessorConfiguration.self, Gemma3Processor.init),
        "SmolVLMProcessor": create(
            SmolVLMProcessorConfiguration.self, SmolVLMProcessor.init),
        "FastVLMProcessor": create(
            FastVLMProcessorConfiguration.self, FastVLMProcessor.init),
        "PixtralProcessor": create(
            PixtralProcessorConfiguration.self, PixtralProcessor.init),
        "Mistral3Processor": create(
            Mistral3VLMProcessorConfiguration.self, Mistral3VLMProcessor.init),
        "Lfm2VlProcessor": create(
            LFM2VLProcessorConfiguration.self, LFM2VLProcessor.init),
        "Glm46VProcessor": create(
            GlmOcrProcessorConfiguration.self, GlmOcrProcessor.init),
        // DeepSeek-OCR ships `processor_class: "DeepseekVLV2Processor"`. The
        // processor reads only `imageTokenIndex` from the model config, and
        // every DeepseekOCRConfiguration field is decodeIfPresent-defaulted, so
        // it decodes cleanly from processor_config.json too.
        "DeepseekVLV2Processor": create(
            DeepseekOCRConfiguration.self, DeepseekOCRProcessor.init),
        "Gemma4Processor": create(
            Gemma4ProcessorConfiguration.self, Gemma4Processor.init),
        "Gemma4UnifiedProcessor": create(
            Gemma4ProcessorConfiguration.self, Gemma4Processor.init),
        "NemotronHOmniProcessor": create(
            NemotronHOmniProcessorConfiguration.self, NemotronHOmniProcessor.init),
        "Zaya1VLProcessor": create(
            Qwen25VLProcessorConfiguration.self, Zaya1VLProcessor.init),
    ])
}

/// Registry of models and any overrides that go with them, e.g. prompt augmentation.
/// If asked for an unknown configuration this will use the model/tokenizer as-is.
///
/// The python tokenizers have a very rich set of implementations and configuration. The
/// swift-tokenizers code handles a good chunk of that and this is a place to augment that
/// implementation, if needed.
public class VLMRegistry: AbstractModelRegistry, @unchecked Sendable {

    /// Shared instance with default model configurations.
    public static let shared: VLMRegistry = .init(modelConfigurations: all())

    static public let paligemma3bMix448_8bit = ModelConfiguration(
        id: "mlx-community/paligemma-3b-mix-448-8bit",
        defaultPrompt: "Describe the image in English"
    )

    static public let qwen2VL2BInstruct4Bit = ModelConfiguration(
        id: "mlx-community/Qwen2-VL-2B-Instruct-4bit",
        defaultPrompt: "Describe the image in English"
    )

    static public let qwen2_5VL3BInstruct4Bit = ModelConfiguration(
        id: "mlx-community/Qwen2.5-VL-3B-Instruct-4bit",
        defaultPrompt: "Describe the image in English"
    )

    static public let qwen3VL4BInstruct4Bit = ModelConfiguration(
        id: "lmstudio-community/Qwen3-VL-4B-Instruct-MLX-4bit",
        defaultPrompt: "Describe the image in English"
    )

    static public let qwen3VL4BInstruct8Bit = ModelConfiguration(
        id: "mlx-community/Qwen3-VL-4B-Instruct-8bit",
        defaultPrompt: "Write a haiku about Swift programming"
    )

    static public let smolvlminstruct4bit = ModelConfiguration(
        id: "mlx-community/SmolVLM-Instruct-4bit",
        defaultPrompt: "Describe the image in English"
    )

    static public let lfm2_5_vl_1_6B_4bit = ModelConfiguration(
        id: "mlx-community/LFM2.5-VL-1.6B-4bit",
        defaultPrompt: ""
    )

    static public let lfm2_vl_1_6B_4bit = ModelConfiguration(
        id: "mlx-community/LFM2-VL-1.6B-4bit",
        defaultPrompt: ""
    )

    static public let mistral3_3B_Instruct_4bit = ModelConfiguration(
        id: "mlx-community/Ministral-3-3B-Instruct-2512-4bit",
        defaultPrompt: ""
    )

    static public let gemma3_4B_qat_4bit = ModelConfiguration(
        id: "mlx-community/gemma-3-4b-it-qat-4bit",
        defaultPrompt: "Describe the image in English",
        extraEOSTokens: ["<end_of_turn>"]
    )

    static public let gemma3_12B_qat_4bit = ModelConfiguration(
        id: "mlx-community/gemma-3-12b-it-qat-4bit",
        defaultPrompt: "Describe the image in English",
        extraEOSTokens: ["<end_of_turn>"]
    )

    static public let gemma3_27B_qat_4bit = ModelConfiguration(
        id: "mlx-community/gemma-3-27b-it-qat-4bit",
        defaultPrompt: "Describe the image in English",
        extraEOSTokens: ["<end_of_turn>"]
    )

    static public let smolvlm = ModelConfiguration(
        id: "HuggingFaceTB/SmolVLM2-500M-Video-Instruct-mlx",
        defaultPrompt:
            "What is the main action or notable event happening in this segment? Describe it in one brief sentence."
    )

    static public let fastvlm = ModelConfiguration(
        id: "mlx-community/FastVLM-0.5B-bf16",
        defaultPrompt: "Describe this image in detail."
    )

    static public let qwen3_5_27B_4bit = ModelConfiguration(
        id: "mlx-community/Qwen3.5-27B-4bit",
        defaultPrompt: "Describe the image in English"
    )

    static public let qwen3_5_35B_A3B_4bit = ModelConfiguration(
        id: "mlx-community/Qwen3.5-35B-A3B-4bit",
        defaultPrompt: "Describe the image in English"
    )

    static public func all() -> [ModelConfiguration] {
        [
            paligemma3bMix448_8bit,
            qwen2VL2BInstruct4Bit,
            qwen2_5VL3BInstruct4Bit,
            qwen3VL4BInstruct4Bit,
            qwen3VL4BInstruct8Bit,
            smolvlminstruct4bit,
            gemma3_4B_qat_4bit,
            gemma3_12B_qat_4bit,
            gemma3_27B_qat_4bit,
            smolvlm,
            fastvlm,
        ]
    }

}

@available(*, deprecated, renamed: "VLMRegistry", message: "Please use VLMRegistry directly.")
public typealias ModelRegistry = VLMRegistry

/// Factory for creating new LLMs.
///
/// Callers can use the `shared` instance or create a new instance if custom configuration
/// is required.
///
/// ```swift
/// let modelContainer = try await VLMModelFactory.shared.loadContainer(
///     configuration: VLMRegistry.paligemma3bMix4488bit)
/// ```
public final class VLMModelFactory: ModelFactory {

    public init(
        typeRegistry: ModelTypeRegistry, processorRegistry: ProcessorTypeRegistry,
        modelRegistry: AbstractModelRegistry
    ) {
        self.typeRegistry = typeRegistry
        self.processorRegistry = processorRegistry
        self.modelRegistry = modelRegistry
    }

    /// Shared instance with default behavior.
    public static let shared = VLMModelFactory(
        typeRegistry: VLMTypeRegistry.shared, processorRegistry: VLMProcessorTypeRegistry.shared,
        modelRegistry: VLMRegistry.shared)

    /// registry of model type, e.g. configuration value `paligemma` -> configuration and init methods
    public let typeRegistry: ModelTypeRegistry

    /// registry of input processor type, e.g. configuration value `PaliGemmaProcessor` -> configuration and init methods
    public let processorRegistry: ProcessorTypeRegistry

    /// registry of model id to configuration, e.g. `mlx-community/paligemma-3b-mix-448-8bit`
    public let modelRegistry: AbstractModelRegistry

    public func _load(
        configuration: ResolvedModelConfiguration,
        tokenizerLoader: any TokenizerLoader
    ) async throws -> sending ModelContext {
        let modelDirectory = configuration.modelDirectory

        // Load config.json once and decode for both base config and model-specific config
        let configurationURL = modelDirectory.appending(component: "config.json")
        let configData: Data
        do {
            configData = try Data(contentsOf: configurationURL)
        } catch {
            throw ModelFactoryError.configurationFileError(
                configurationURL.lastPathComponent, configuration.name, error)
        }
        let baseConfig: BaseConfiguration
        do {
            baseConfig = try JSONDecoder.json5().decode(BaseConfiguration.self, from: configData)
        } catch let error as DecodingError {
            throw ModelFactoryError.configurationDecodingError(
                configurationURL.lastPathComponent, configuration.name, error)
        }

        // Detect Nemotron-Omni bundles by presence of config_omni.json. The
        // bundle's config.json reports `model_type: nemotron_h` (LLM only),
        // so we override the dispatch model_type here so the VLM factory
        // routes to NemotronHOmni instead of failing or hitting the LLM path.
        var dispatchModelType = baseConfig.modelType
        let configOmniURL = modelDirectory.appending(component: "config_omni.json")
        if FileManager.default.fileExists(atPath: configOmniURL.path) {
            dispatchModelType = "NemotronH_Nano_Omni_Reasoning_V3"
        }

        // 2026-04-29: merge jang_config.json fields into the config.json
        // dictionary BEFORE decoding so omni `NemotronHOmniConfiguration`
        // sees `weight_format` + `mxtq_bits` + `mxtq_seed` and can opt
        // its inner `NemotronHModel` into the JANGTQ codebook MoE path.
        // Mirrors what `LLMModelFactory` does for non-omni JANGTQ
        // bundles (Cascade-2 / DSV3 / Qwen3.5 MoE). Idempotent: bundles
        // without `jang_config.json` skip this entirely.
        var mergedConfigData = configData
        let jangConfigURL = modelDirectory.appending(component: "jang_config.json")
        if FileManager.default.fileExists(atPath: jangConfigURL.path),
            let jangData = try? Data(contentsOf: jangConfigURL),
            let jangJSON = try? JSONSerialization.jsonObject(with: jangData)
                as? [String: Any],
            var configDict = try? JSONSerialization.jsonObject(with: configData)
                as? [String: Any]
        {
            let jangFormat = (jangJSON["format"] as? String)?.lowercased()
            let jangWeightFormat = (jangJSON["weight_format"] as? String)?.lowercased()
            let hasJANGTQSidecar = FileManager.default.fileExists(
                atPath: modelDirectory.appending(component: "jangtq_runtime.safetensors").path)
            let shouldMergeJANGTQMetadata =
                hasJANGTQSidecar
                || jangFormat?.contains("jangtq") == true
                || jangWeightFormat == "mxtq"
                || jangWeightFormat?.contains("jangtq") == true

            if shouldMergeJANGTQMetadata {
            // weight_format lives at top-level in jang_config.json
            if let wf = jangJSON["weight_format"] as? String {
                configDict["weight_format"] = wf
            }
            let bitsMaps: [[String: Any]] = [
                configDict["mxtq_bits"] as? [String: Any],
                jangJSON["mxtq_bits"] as? [String: Any],
            ].compactMap { $0 }
            func intValue(_ value: Any?) -> Int? {
                if let value = value as? Int {
                    return value
                }
                if let value = value as? NSNumber {
                    return value.intValue
                }
                return nil
            }
            for bitsMap in bitsMaps {
                let routedAny = bitsMap["routed_expert"]
                if let routedInt = intValue(routedAny) {
                    configDict["mxtq_bits"] = routedInt
                    break
                } else if let routedDict = routedAny as? [String: Any] {
                    let gate = intValue(routedDict["gate_proj"])
                    let up = intValue(routedDict["up_proj"])
                    let down = intValue(routedDict["down_proj"])
                    if let g = gate, let u = up, g == u {
                        configDict["mxtq_bits"] = g
                        configDict["mxtq_gate_up_bits"] = g
                        if let d = down {
                            configDict["mxtq_down_bits"] = d
                        }
                        break
                    } else if let g = gate, let u = up {
                        throw ModelFactoryError.configurationFileError(
                            jangConfigURL.lastPathComponent,
                            configuration.name,
                            NSError(
                                domain: "VLMModelFactory",
                                code: 4,
                                userInfo: [
                                    NSLocalizedDescriptionKey:
                                        "mxtq_bits.routed_expert gate_proj (\(g)) and up_proj (\(u)) differ; fused gate/up JANGTQ kernels require matching widths"
                                ]))
                    } else if let g = gate ?? up {
                        configDict["mxtq_bits"] = g
                        configDict["mxtq_gate_up_bits"] = g
                        if let d = down {
                            configDict["mxtq_down_bits"] = d
                        }
                        break
                    }
                }
            }
            // mxtq_bits comes from the converter's bit_widths_used list
            // (lowest bit width) or an explicit `mxtq_bits` key if present.
            if let qDict = jangJSON["quantization"] as? [String: Any] {
                if configDict["mxtq_bits"] == nil,
                    let bits = qDict["bit_widths_used"] as? [Int],
                    let minBits = bits.min()
                {
                    configDict["mxtq_bits"] = minBits
                }
                // Profile fallback: JANGTQ4 → 4 bits, JANGTQ2 → 2 bits.
                if configDict["mxtq_bits"] == nil,
                    let profile = qDict["profile"] as? String
                {
                    if profile.contains("4") { configDict["mxtq_bits"] = 4 }
                    else if profile.contains("2") { configDict["mxtq_bits"] = 2 }
                }
                if configDict["mxtq_seed"] == nil,
                    let seed = qDict["mxtq_seed"] as? Int
                {
                    configDict["mxtq_seed"] = seed
                }
            }
            if configDict["mxtq_seed"] == nil,
                let seed = jangJSON["mxtq_seed"] as? Int
            {
                configDict["mxtq_seed"] = seed
            }
            if var textConfig = configDict["text_config"] as? [String: Any] {
                for key in [
                    "weight_format",
                    "mxtq_bits",
                    "mxtq_gate_up_bits",
                    "mxtq_down_bits",
                    "mxtq_seed",
                ] {
                    if textConfig[key] == nil, let value = configDict[key] {
                        textConfig[key] = value
                    }
                }
                configDict["text_config"] = textConfig
            }
            }
            if let merged = try? JSONSerialization.data(withJSONObject: configDict) {
                mergedConfigData = merged
            }
        }

        let earlyJangConfig: JangConfig?
        if JangLoader.isJangModel(at: modelDirectory) {
            earlyJangConfig = try JangLoader.loadConfig(at: modelDirectory)
        } else {
            earlyJangConfig = nil
        }
        let earlyMTPStatus = try? MTPBundleInspector.inspect(
            modelDirectory: modelDirectory,
            jangConfig: earlyJangConfig)
        let loadNativeMTP = try NativeMTPActivation.shouldLoadNativeMTPWeights(
            configData: mergedConfigData,
            baseModelType: dispatchModelType,
            status: earlyMTPStatus)
        if !loadNativeMTP {
            mergedConfigData = try NativeMTPActivation.scrubInactiveMTPConfig(mergedConfigData)
        }

        let model: LanguageModel
        do {
            model = try await typeRegistry.createModel(
                configuration: mergedConfigData, modelType: dispatchModelType)
        } catch let error as DecodingError {
            throw ModelFactoryError.configurationDecodingError(
                configurationURL.lastPathComponent, configuration.name, error)
        }

        let generationConfigURL = modelDirectory.appending(component: "generation_config.json")
        let generationConfig =
            if let generationData = try? Data(contentsOf: generationConfigURL) {
                try? JSONDecoder.json5().decode(GenerationConfigFile.self, from: generationData)
            } else {
                nil as GenerationConfigFile?
            }
        // Load EOS token IDs from config.json, nested text_config, with optional
        // override from generation_config.json.
        var eosTokenIds = ModelTokenConfigurationResolver.resolvedEOSTokenIds(
            baseConfig: baseConfig,
            configurationData: mergedConfigData,
            generationConfig: generationConfig)
        if baseConfig.modelType == "deepseek_v4" {
            eosTokenIds.formUnion([1, 128803, 128804])
        }

        var mutableConfiguration = configuration
        mutableConfiguration.eosTokenIds = eosTokenIds

        // Detect JANG model BEFORE tool-format selection so the `capabilities.tool_parser`
        // stamp is authoritative for JANG bundles. Standard MLX models skip this and
        // jangConfig stays nil, in which case we fall through to the model_type heuristic.
        let jangConfig = earlyJangConfig
        let mtpStatus = earlyMTPStatus

        // Tool-format resolution priority (same rationale as LLMModelFactory):
        //   1. Caller-supplied `configuration.toolCallFormat` (explicit override).
        //   2. JANG `capabilities.tool_parser` stamp — authoritative when set.
        //   3. `ToolCallFormat.infer(from: modelType)` heuristic — last resort.
        if mutableConfiguration.toolCallFormat == nil {
            // Same priority ladder as LLMModelFactory — DSV4 VLM
            // bundles will stamp `chat.tool_calling.parser = "dsml"`.
            let chatStamped = ToolCallFormat.fromCapabilityName(
                jangConfig?.chat?.toolCalling?.parser)
            let jangStamped = ToolCallFormat.fromCapabilityName(
                jangConfig?.capabilities?.toolParser)
            mutableConfiguration.toolCallFormat =
                chatStamped
                ?? jangStamped
                ?? ToolCallFormat.infer(from: baseConfig.modelType)
        }

        // Reasoning-parser stamp (same precedence as LLMModelFactory).
        // VL models that emit `<think>` follow the same Qwen / DeepSeek
        // conventions as their text-only counterparts.
        //
        // See the LLMModelFactory twin for the full writeup on why
        // this is an explicit allowlist rather than a reverse-allowlist
        // default — shared helper in MLXLMCommon.
        if mutableConfiguration.reasoningParserName == nil {
            if ParserResolution.shouldIgnoreReasoningStamp(
                capabilities: jangConfig?.capabilities,
                modelType: baseConfig.modelType)
            {
                let resolvedReasoning = ParserResolution.reasoning(
                    capabilities: jangConfig?.capabilities,
                    modelType: baseConfig.modelType,
                    chatTemplate: chatTemplateText(modelDirectory: modelDirectory))
                mutableConfiguration.reasoningParserName =
                    resolvedReasoning.parser == nil
                    ? "none"
                    : (resolvedReasoning.source == .chatTemplate
                        ? "qwen3"
                        : reasoningStampFromModelType(baseConfig.modelType))
            } else if let stamp = jangConfig?.capabilities?.reasoningParser {
                mutableConfiguration.reasoningParserName = stamp
            } else {
                let resolvedReasoning = ParserResolution.reasoning(
                    capabilities: jangConfig?.capabilities,
                    modelType: baseConfig.modelType,
                    chatTemplate: chatTemplateText(modelDirectory: modelDirectory))
                mutableConfiguration.reasoningParserName =
                    resolvedReasoning.parser == nil
                    ? "none"
                    : (resolvedReasoning.source == .chatTemplate
                        ? "qwen3"
                        : reasoningStampFromModelType(baseConfig.modelType))
            }
        }

        // Load tokenizer from model directory (or alternate tokenizer repo),
        // processor config, and weights in parallel using async let.
        // Note: loadProcessorConfig does synchronous I/O but is marked async to enable
        // parallel scheduling. This may briefly block a cooperative thread pool thread,
        // but the config file is small and model loading is not a high-concurrency path.
        //
        // JANG VL bundles (e.g. Qwen3.5-VL-*-JANG_*) may also ship
        // weights-only. `resolveTokenizerDirectory` falls back to the cached
        // source model's tokenizer when that happens; otherwise returns the
        // original directory unchanged.
        let jangResolvedDir = JangLoader.resolveTokenizerDirectory(
            for: configuration.tokenizerDirectory)
        let templateResolvedDir = JangLoader.resolveChatTemplateSidecarSubstitution(
            for: jangResolvedDir)
        let tokenizerDirectory = JangLoader.resolveTokenizerClassSubstitution(
            for: templateResolvedDir)
        async let tokenizerTask = tokenizerLoader.load(from: tokenizerDirectory)
        async let processorConfigTask = loadProcessorConfig(from: modelDirectory)

        try loadWeights(
            modelDirectory: modelDirectory, model: model,
            // 2026-04-28: pass top-level `quantization` through too, matching
            // the LLM factory. Omni / VL bundles whose `config.json` carries
            // `quantization.group_size` need that prior to land at
            // `inferPerLayerQuantization` so the shape walk uses the right
            // gs. Without it, MXFP4 omni fell back to `jangConfig.blockSize`'s
            // default 64 instead of the actual gs=32, mis-inferring layers
            // and causing mid-prefill rmsNorm shape traps.
            quantization: jangConfig != nil ? baseConfig.quantizationContainer?.quantization : nil,
            perLayerQuantization: baseConfig.perLayerQuantization,
            jangConfig: jangConfig,
            loadPreservedMTP: loadNativeMTP)

        let tokenizer = try await tokenizerTask
        let processorConfigData: Data
        let baseProcessorConfig: BaseProcessorConfiguration
        do {
            (processorConfigData, baseProcessorConfig) = try await processorConfigTask
        } catch let error as ProcessorConfigError {
            if let decodingError = error.underlying as? DecodingError {
                throw ModelFactoryError.configurationDecodingError(
                    error.filename, configuration.name, decodingError)
            }
            throw ModelFactoryError.configurationFileError(
                error.filename, configuration.name, error.underlying)
        }

        // Override processor type based on model type for models that need special handling
        // Mistral3 models ship with "PixtralProcessor" in their config but need Mistral3Processor
        // to handle spatial merging correctly. Nemotron-Omni bundles use a custom
        // image_processor_type that doesn't map to processor_class — force the
        // NemotronHOmniProcessor when we've detected the omni bundle.
        let processorTypeOverrides: [String: String] = [
            "mistral3": "Mistral3Processor",
            // Mistral 3.5 VLM bundles can carry the outer model_type
            // `ministral3` (the inner text decoder spelling promoted to
            // the outer level). Their preprocessor_config.json still
            // ships `processor_class: "PixtralProcessor"`, which loses
            // Mistral3's spatial-merge handling. Force the spatial-merge
            // processor here for both spellings — same dispatch as
            // VLMTypeRegistry.dispatchMistral3VLM.
            "ministral3": "Mistral3Processor",
            "NemotronH_Nano_Omni_Reasoning_V3": "NemotronHOmniProcessor",
            "nemotron_h_omni": "NemotronHOmniProcessor",
        ]
        let processorType =
            processorTypeOverrides[dispatchModelType] ?? baseProcessorConfig.processorClass

        let baseProcessor = try await processorRegistry.createModel(
            configuration: processorConfigData,
            processorType: processorType, tokenizer: tokenizer)
        let defaultAdditionalContext = VLMDefaultContextUserInputProcessor.defaultContext(
            capabilities: jangConfig?.capabilities)
        let processor: any UserInputProcessor =
            if defaultAdditionalContext != nil {
                VLMDefaultContextUserInputProcessor(
                    base: baseProcessor,
                    defaultAdditionalContext: defaultAdditionalContext)
            } else {
                baseProcessor
            }

        // Build a ModelConfiguration for the ModelContext. When the JANG
        // fallback resolved to a different directory than the caller
        // requested, surface that in the `tokenizerSource` so any re-load
        // via this config uses the same tokenizer.
        let tokenizerSource: TokenizerSource? =
            tokenizerDirectory == modelDirectory
            ? nil
            : .directory(tokenizerDirectory)
        let modelConfig = ModelConfiguration(
            directory: modelDirectory,
            tokenizerSource: tokenizerSource,
            defaultPrompt: configuration.defaultPrompt,
            extraEOSTokens: mutableConfiguration.extraEOSTokens,
            eosTokenIds: mutableConfiguration.eosTokenIds,
            toolCallFormat: mutableConfiguration.toolCallFormat,
            reasoningParserName: mutableConfiguration.reasoningParserName,
            generationDefaults: generationConfig,
            mtpStatus: mtpStatus)

        return .init(
            configuration: modelConfig, model: model, processor: processor,
            tokenizer: tokenizer)
    }

}

/// Error wrapper that includes the filename for better error messages.
private struct ProcessorConfigError: Error {
    let filename: String
    let underlying: Error
}

/// Loads processor configuration, preferring preprocessor_config.json over processor_config.json.
/// Marked async to enable parallel scheduling via async let, though the underlying I/O is synchronous.
/// Throws ProcessorConfigError wrapping any underlying error with the filename.
private func loadProcessorConfig(from modelDirectory: URL) async throws -> (
    Data, BaseProcessorConfiguration
) {
    let processorConfigURL = modelDirectory.appending(component: "processor_config.json")
    let preprocessorConfigURL = modelDirectory.appending(component: "preprocessor_config.json")
    let videoPreprocessorConfigURL = modelDirectory.appending(
        component: "video_preprocessor_config.json")
    let url =
        FileManager.default.fileExists(atPath: preprocessorConfigURL.path)
        ? preprocessorConfigURL
        : processorConfigURL
    do {
        var data = try Data(contentsOf: url)
        if FileManager.default.fileExists(atPath: videoPreprocessorConfigURL.path) {
            do {
                guard
                    var processorObject = try JSONSerialization.jsonObject(with: data)
                        as? [String: Any]
                else {
                    throw DecodingError.dataCorrupted(
                        .init(
                            codingPath: [],
                            debugDescription: "\(url.lastPathComponent) is not a JSON object"))
                }
                let videoData = try Data(contentsOf: videoPreprocessorConfigURL)
                guard
                    let videoObject = try JSONSerialization.jsonObject(with: videoData)
                        as? [String: Any]
                else {
                    throw DecodingError.dataCorrupted(
                        .init(
                            codingPath: [],
                            debugDescription:
                                "\(videoPreprocessorConfigURL.lastPathComponent) is not a JSON object"
                        ))
                }
                processorObject["video_preprocessor_config"] = videoObject
                data = try JSONSerialization.data(withJSONObject: processorObject)
            } catch {
                throw ProcessorConfigError(
                    filename: videoPreprocessorConfigURL.lastPathComponent,
                    underlying: error)
            }
        }
        let config = try JSONDecoder.json5().decode(BaseProcessorConfiguration.self, from: data)
        return (data, config)
    } catch {
        if let error = error as? ProcessorConfigError {
            throw error
        }
        throw ProcessorConfigError(filename: url.lastPathComponent, underlying: error)
    }
}

/// Decorates a VLM processor with safe default chat-template context while
/// preserving explicit caller overrides.
struct VLMDefaultContextUserInputProcessor: UserInputProcessor {
    let base: any UserInputProcessor
    let defaultAdditionalContext: [String: any Sendable]?

    init(base: any UserInputProcessor, defaultAdditionalContext: [String: any Sendable]?) {
        self.base = base
        self.defaultAdditionalContext = defaultAdditionalContext
    }

    static func defaultContext(capabilities: JangCapabilities?) -> [String: any Sendable]? {
        var context: [String: any Sendable] = [:]

        if capabilities?.supportsThinking == false {
            context["enable_thinking"] = false
        }

        return context.isEmpty ? nil : context
    }

    func prepare(input: UserInput) async throws -> LMInput {
        guard let defaultAdditionalContext, !defaultAdditionalContext.isEmpty else {
            return try await base.prepare(input: input)
        }

        var merged = defaultAdditionalContext
        for (key, value) in input.additionalContext ?? [:] {
            merged[key] = value
        }

        let rewritten = UserInput(
            prompt: input.prompt,
            images: input.images,
            videos: input.videos,
            audios: input.audios,
            processing: input.processing,
            tools: input.tools,
            additionalContext: merged)
        return try await base.prepare(input: rewritten)
    }
}

public class TrampolineModelFactory: NSObject, ModelFactoryTrampoline {
    public static func modelFactory() -> (any MLXLMCommon.ModelFactory)? {
        VLMModelFactory.shared
    }
}
