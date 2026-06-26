//
//  DeepseekOCRConfiguration.swift
//  mlx-swift-lm
//
//  Configuration for DeepSeek-OCR / Unlimited-OCR (DeepseekOCRForCausalLM,
//  top model_type "deepseek_vl_v2"). Port of
//  https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models/deepseekocr (config.py)
//
//  Covers deepseek-ai/DeepSeek-OCR and baidu/Unlimited-OCR (identical arch).
//

import Foundation
import MLXLMCommon

/// Top-level configuration for the DeepSeek-OCR family.
public struct DeepseekOCRConfiguration: Codable, Sendable {

    /// DeepSeek-V2 MoE text decoder config. Standard attention path
    /// (use_mla=false / qk_nope_head_dim=0 ⇒ Llama-style attention).
    public struct TextConfiguration: Codable, Sendable {
        public var modelType: String
        public var vocabSize: Int
        public var hiddenSize: Int
        public var intermediateSize: Int
        public var moeIntermediateSize: Int
        public var numHiddenLayers: Int
        public var numAttentionHeads: Int
        public var numKeyValueHeads: Int
        public var nSharedExperts: Int?
        public var nRoutedExperts: Int?
        public var routedScalingFactor: Float
        public var numExpertsPerTok: Int?
        public var moeLayerFreq: Int
        public var firstKDenseReplace: Int
        public var maxPositionEmbeddings: Int
        public var rmsNormEps: Float
        public var ropeTheta: Float
        public var ropeScaling: [String: StringOrNumber]?
        public var attentionBias: Bool
        public var scoringFunc: String
        public var topkMethod: String
        public var nGroup: Int?
        public var topkGroup: Int?
        // Standard-attention head dim derived as hiddenSize / numAttentionHeads.

        public var headDim: Int { hiddenSize / numAttentionHeads }

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case vocabSize = "vocab_size"
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case moeIntermediateSize = "moe_intermediate_size"
            case numHiddenLayers = "num_hidden_layers"
            case numAttentionHeads = "num_attention_heads"
            case numKeyValueHeads = "num_key_value_heads"
            case nSharedExperts = "n_shared_experts"
            case nRoutedExperts = "n_routed_experts"
            case routedScalingFactor = "routed_scaling_factor"
            case numExpertsPerTok = "num_experts_per_tok"
            case moeLayerFreq = "moe_layer_freq"
            case firstKDenseReplace = "first_k_dense_replace"
            case maxPositionEmbeddings = "max_position_embeddings"
            case rmsNormEps = "rms_norm_eps"
            case ropeTheta = "rope_theta"
            case ropeScaling = "rope_scaling"
            case attentionBias = "attention_bias"
            case scoringFunc = "scoring_func"
            case topkMethod = "topk_method"
            case nGroup = "n_group"
            case topkGroup = "topk_group"
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "deepseek_v2"
            vocabSize = try c.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 129280
            hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1280
            intermediateSize =
                try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 6848
            moeIntermediateSize =
                try c.decodeIfPresent(Int.self, forKey: .moeIntermediateSize) ?? 896
            numHiddenLayers =
                try c.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 12
            numAttentionHeads =
                try c.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 10
            numKeyValueHeads =
                try c.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? numAttentionHeads
            nSharedExperts = try c.decodeIfPresent(Int.self, forKey: .nSharedExperts) ?? 2
            nRoutedExperts = try c.decodeIfPresent(Int.self, forKey: .nRoutedExperts) ?? 64
            routedScalingFactor =
                try c.decodeIfPresent(Float.self, forKey: .routedScalingFactor) ?? 1.0
            numExpertsPerTok = try c.decodeIfPresent(Int.self, forKey: .numExpertsPerTok) ?? 6
            moeLayerFreq = try c.decodeIfPresent(Int.self, forKey: .moeLayerFreq) ?? 1
            firstKDenseReplace =
                try c.decodeIfPresent(Int.self, forKey: .firstKDenseReplace) ?? 1
            maxPositionEmbeddings =
                try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 8192
            rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
            ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10000.0
            ropeScaling = try c.decodeIfPresent(
                [String: StringOrNumber].self, forKey: .ropeScaling)
            attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
            scoringFunc = try c.decodeIfPresent(String.self, forKey: .scoringFunc) ?? "softmax"
            topkMethod = try c.decodeIfPresent(String.self, forKey: .topkMethod) ?? "greedy"
            nGroup = try c.decodeIfPresent(Int.self, forKey: .nGroup) ?? 1
            topkGroup = try c.decodeIfPresent(Int.self, forKey: .topkGroup) ?? 1
        }
    }

    /// CLIP-L/14 vision tower (the `vision_model` branch of the DeepEncoder).
    public struct VisionConfiguration: Codable, Sendable {
        public var layers: Int
        public var width: Int
        public var heads: Int
        public var imageSize: Int
        public var patchSize: Int
        public var numChannels: Int
        public var layerNormEps: Float
        public var mlpRatio: Float

        public init(from decoder: Decoder) throws {
            // The HF config nests CLIP geometry under vision_config.width["clip-l-14-224"];
            // mlx-vlm flattens to these fields. Decode defensively with CLIP-L defaults.
            let c = try decoder.container(keyedBy: AnyKey.self)
            func i(_ k: String, _ d: Int) -> Int {
                (try? c.decode(Int.self, forKey: AnyKey(k))) ?? d
            }
            func f(_ k: String, _ d: Float) -> Float {
                (try? c.decode(Float.self, forKey: AnyKey(k))) ?? d
            }
            layers = i("layers", 24)
            width = i("width", 1024)
            heads = i("num_attention_heads", 16)
            imageSize = i("image_size", 224)
            patchSize = i("patch_size", 14)
            numChannels = i("num_channels", 3)
            layerNormEps = f("layer_norm_eps", 1e-6)
            mlpRatio = f("mlp_ratio", 3.7362)
        }

        // NOTE: compile-fix — factory's create<C: Codable> requires Encodable; this
        // struct decodes via dynamic AnyKey so encode cannot be synthesized. Never
        // called at runtime (factory only decodes); provided to satisfy conformance.
        public func encode(to encoder: Encoder) throws {
            var c = encoder.container(keyedBy: AnyKey.self)
            try c.encode(layers, forKey: AnyKey("layers"))
            try c.encode(width, forKey: AnyKey("width"))
            try c.encode(heads, forKey: AnyKey("num_attention_heads"))
            try c.encode(imageSize, forKey: AnyKey("image_size"))
            try c.encode(patchSize, forKey: AnyKey("patch_size"))
            try c.encode(numChannels, forKey: AnyKey("num_channels"))
            try c.encode(layerNormEps, forKey: AnyKey("layer_norm_eps"))
            try c.encode(mlpRatio, forKey: AnyKey("mlp_ratio"))
        }
    }

    /// SAM-ViT-B encoder (the `sam_model` branch of the DeepEncoder).
    /// Geometry is fixed by SAMViTConfig() in mlx-vlm (not read from config.json).
    public struct SAMViTConfiguration: Sendable {
        public var imageSize: Int = 1024
        public var width: Int = 768
        public var layers: Int = 12
        public var heads: Int = 12
        public var patchSize: Int = 16
        public var windowSize: Int = 14
        public var promptEmbedDim: Int = 256
        public var globalAttnIndexes: [Int] = [2, 5, 8, 11]
        public var downsampleChannels: [Int] = [512, 1024]
        public init() {}
    }

    /// MLP projector (vision feature dim → text hidden).
    public struct ProjectorConfiguration: Codable, Sendable {
        public var projectorType: String
        public var inputDim: Int
        public var nEmbed: Int
        public var depth: Int
        public var mlpRatio: Int
        public var downsampleRatio: Int

        enum CodingKeys: String, CodingKey {
            case projectorType = "projector_type"
            case inputDim = "input_dim"
            case nEmbed = "n_embed"
            case depth
            case mlpRatio = "mlp_ratio"
            case downsampleRatio = "downsample_ratio"
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            projectorType =
                try c.decodeIfPresent(String.self, forKey: .projectorType) ?? "linear"
            inputDim = try c.decodeIfPresent(Int.self, forKey: .inputDim) ?? 2048
            nEmbed = try c.decodeIfPresent(Int.self, forKey: .nEmbed) ?? 1280
            depth = try c.decodeIfPresent(Int.self, forKey: .depth) ?? 2
            mlpRatio = try c.decodeIfPresent(Int.self, forKey: .mlpRatio) ?? 1
            downsampleRatio =
                try c.decodeIfPresent(Int.self, forKey: .downsampleRatio) ?? 2
        }
    }

    public var textConfiguration: TextConfiguration
    public var visionConfiguration: VisionConfiguration
    public var projectorConfiguration: ProjectorConfiguration
    public var samConfiguration = SAMViTConfiguration()
    public var modelType: String
    public var imageTokenIndex: Int
    public var tileTag: String
    public var globalViewPos: String
    public var numImageTokens: Int
    public var quantization: BaseConfiguration.Quantization?

    enum CodingKeys: String, CodingKey {
        case textConfiguration = "language_config"
        case visionConfiguration = "vision_config"
        case projectorConfiguration = "projector_config"
        case modelType = "model_type"
        case imageTokenIndex = "image_token_index"
        case imageTokenId = "image_token_id"
        case tileTag = "tile_tag"
        case globalViewPos = "global_view_pos"
        case numImageTokens = "num_image_tokens"
        case quantization
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        // The same struct is decoded from BOTH config.json (full model config,
        // carries language_config/vision_config/projector_config) AND
        // processor_config.json (carries only image_token + tiling fields). The
        // sub-configs are therefore decodeIfPresent with all-defaulted fallbacks
        // (every sub-config field is itself decodeIfPresent-defaulted), so the
        // processor path doesn't trap on the missing `language_config` key.
        textConfiguration =
            try c.decodeIfPresent(TextConfiguration.self, forKey: .textConfiguration)
            ?? TextConfiguration(from: EmptyDecoder())
        visionConfiguration =
            try c.decodeIfPresent(VisionConfiguration.self, forKey: .visionConfiguration)
            ?? VisionConfiguration(from: EmptyDecoder())
        projectorConfiguration =
            try c.decodeIfPresent(ProjectorConfiguration.self, forKey: .projectorConfiguration)
            ?? ProjectorConfiguration(from: EmptyDecoder())
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "deepseek_vl_v2"
        // image token index: prefer image_token_index, fall back to image_token_id,
        // then the mlx-vlm default 128815.
        imageTokenIndex =
            try (c.decodeIfPresent(Int.self, forKey: .imageTokenIndex)
                ?? c.decodeIfPresent(Int.self, forKey: .imageTokenId))
            ?? 128815
        tileTag = try c.decodeIfPresent(String.self, forKey: .tileTag) ?? "2D"
        globalViewPos = try c.decodeIfPresent(String.self, forKey: .globalViewPos) ?? "head"
        numImageTokens = try c.decodeIfPresent(Int.self, forKey: .numImageTokens) ?? 576
        quantization = try c.decodeIfPresent(
            BaseConfiguration.Quantization.self, forKey: .quantization)
    }

    // NOTE: compile-fix — factory's create<C: Codable> requires Encodable. Encode
    // cannot be synthesized here because CodingKeys has an extra `imageTokenId` key
    // with no matching property. Never called at runtime (factory only decodes).
    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(textConfiguration, forKey: .textConfiguration)
        try c.encode(visionConfiguration, forKey: .visionConfiguration)
        try c.encode(projectorConfiguration, forKey: .projectorConfiguration)
        try c.encode(modelType, forKey: .modelType)
        try c.encode(imageTokenIndex, forKey: .imageTokenIndex)
        try c.encode(tileTag, forKey: .tileTag)
        try c.encode(globalViewPos, forKey: .globalViewPos)
        try c.encode(numImageTokens, forKey: .numImageTokens)
        try c.encodeIfPresent(quantization, forKey: .quantization)
    }
}

/// Minimal `Decoder` that exposes an empty keyed container, so a sub-config
/// whose fields are all `decodeIfPresent`-defaulted can be constructed with
/// every default (used when decoding from processor_config.json, which omits
/// language_config / vision_config / projector_config).
private struct EmptyDecoder: Decoder {
    var codingPath: [CodingKey] { [] }
    var userInfo: [CodingUserInfoKey: Any] { [:] }

    func container<Key: CodingKey>(keyedBy type: Key.Type)
        -> KeyedDecodingContainer<Key>
    {
        KeyedDecodingContainer(EmptyKeyed<Key>())
    }
    func unkeyedContainer() throws -> UnkeyedDecodingContainer {
        throw DecodingError.dataCorrupted(
            .init(codingPath: [], debugDescription: "EmptyDecoder has no unkeyed container"))
    }
    func singleValueContainer() throws -> SingleValueDecodingContainer {
        throw DecodingError.dataCorrupted(
            .init(codingPath: [], debugDescription: "EmptyDecoder has no single-value container"))
    }

    private struct EmptyKeyed<K: CodingKey>: KeyedDecodingContainerProtocol {
        typealias Key = K
        var codingPath: [CodingKey] { [] }
        var allKeys: [K] { [] }
        func contains(_ key: K) -> Bool { false }
        func decodeNil(forKey key: K) throws -> Bool { true }
        func decode<T: Decodable>(_ type: T.Type, forKey key: K) throws -> T {
            throw DecodingError.keyNotFound(
                key, .init(codingPath: [], debugDescription: "empty"))
        }
        func nestedContainer<NK: CodingKey>(keyedBy type: NK.Type, forKey key: K) throws
            -> KeyedDecodingContainer<NK>
        {
            KeyedDecodingContainer(EmptyKeyed<NK>())
        }
        func nestedUnkeyedContainer(forKey key: K) throws -> UnkeyedDecodingContainer {
            throw DecodingError.keyNotFound(
                key, .init(codingPath: [], debugDescription: "empty"))
        }
        func superDecoder() throws -> Decoder { EmptyDecoder() }
        func superDecoder(forKey key: K) throws -> Decoder { EmptyDecoder() }
    }
}

/// Helper coding key for flat-keyed sub-dicts (CLIP geometry).
private struct AnyKey: CodingKey {
    var stringValue: String
    var intValue: Int? { nil }
    init(_ s: String) { stringValue = s }
    init?(stringValue: String) { self.stringValue = stringValue }
    init?(intValue: Int) { nil }
}
