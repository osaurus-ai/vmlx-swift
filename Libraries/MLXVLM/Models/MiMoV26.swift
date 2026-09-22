// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

struct MiMoV26Configuration: Decodable {
    let text: MiMoV2FlashConfiguration
    let vision: MiMoV26VisionConfiguration?
    let audio: MiMoV26AudioConfiguration?
    let imageToken: Int
    let videoToken: Int
    let audioToken: Int

    enum CodingKeys: String, CodingKey {
        case vision = "vision_config", audio = "audio_config"
        case imageToken = "image_token_id", videoToken = "video_token_id", audioToken = "audio_token_id"
    }

    init(from decoder: Decoder) throws {
        text = try MiMoV2FlashConfiguration(from: decoder)
        let c = try decoder.container(keyedBy: CodingKeys.self)
        vision = try c.decodeIfPresent(MiMoV26VisionConfiguration.self, forKey: .vision)
        audio = try c.decodeIfPresent(MiMoV26AudioConfiguration.self, forKey: .audio)
        imageToken = try c.decode(Int.self, forKey: .imageToken)
        videoToken = try c.decode(Int.self, forKey: .videoToken)
        audioToken = try c.decode(Int.self, forKey: .audioToken)
    }
}

/// The generic load owns only the text parameters. Auxiliary towers select
/// their own checkpoint leaves on first media use, retaining native dtypes.
final class MiMoV26: Module, VLMModel, KVCacheDimensionProvider, SafetensorsLoadKeyExcluding,
    ModalityBearing
{
    @ModuleInfo(key: "language_model") var languageModel: MiMoV26TextModel
    @ModuleInfo(key: "visual") private var visionTower: MiMoV26VisionTower?
    @ModuleInfo(key: "audio_encoder") private var audioEncoder: MiMoV26AudioEncoder?
    @ModuleInfo(key: "audio_tokenizer") private var audioTokenizer: MiMoV26AudioTokenizer?
    private var directory: URL?
    private var audioFeatures: MiMoV26AudioFeatures.Configuration?
    let configuration: MiMoV26Configuration
    let modalities: Set<ModelRuntimeRequestModality>

    var vocabularySize: Int { languageModel.vocabularySize }
    var kvHeads: [Int] { languageModel.kvHeads }
    var loraLayers: [Module] { languageModel.loraLayers }
    var preservesCheckpointParameterDTypes: Bool { true }
    var supportsWholeForwardCompilation: Bool { languageModel.supportsWholeForwardCompilation }
    var requiresExactTensorMmapBuffers: Bool { true }

    init(_ configuration: MiMoV26Configuration, requesting: Set<ModelRuntimeRequestModality>? = nil) throws {
        var supported: Set<ModelRuntimeRequestModality> = [.text]
        if configuration.vision != nil { supported.formUnion([.vision, .video]) }
        if configuration.audio != nil { supported.insert(.audio) }
        let requested = requesting ?? supported
        guard requested.isSubset(of: supported) else {
            throw VLMError.processing("Requested MiMo modalities are absent from the configuration")
        }
        let textModel = try MiMoV26TextModel(configuration.text)
        guard configuration.vision == nil || configuration.vision?.outputSize == textModel.hiddenSize,
            configuration.audio == nil || configuration.audio?.outputSize == textModel.hiddenSize else {
            throw VLMError.processing("MiMo auxiliary output width does not match the text model")
        }
        // Validate before initializing Module's wrapped state. This also avoids
        // Swift 6.3's optimized ownership failure on a throwing post-super path.
        self.configuration = configuration
        modalities = requested.union([.text])
        _languageModel.wrappedValue = textModel
        super.init()
    }

    func configure(modelDirectory: URL) throws {
        try languageModel.configure(modelDirectory: modelDirectory)
        directory = modelDirectory
    }

    func excludeFromGenericSafetensorsLoad(key: String) -> Bool {
        languageModel.excludeFromGenericSafetensorsLoad(key: key)
    }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        Dictionary(uniqueKeysWithValues: languageModel.sanitize(weights: weights).map {
            ("language_model." + $0.key, $0.value)
        })
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache)
    }

    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        languageModel.newCache(parameters: parameters)
    }

    private func vision() throws -> MiMoV26VisionTower {
        if let visionTower { return visionTower }
        guard let directory, let config = configuration.vision else {
            throw VLMError.processing("MiMo vision requires its configured model directory")
        }
        let tower = try MiMoV26AuxiliaryLoader.vision(in: directory, configuration: config)
        self.visionTower = tower
        return tower
    }

    private func audio() throws -> (MiMoV26AudioTokenizer, MiMoV26AudioEncoder, MiMoV26AudioFeatures.Configuration) {
        if let audioTokenizer, let audioEncoder, let audioFeatures {
            return (audioTokenizer, audioEncoder, audioFeatures)
        }
        guard let directory, let config = configuration.audio else {
            throw VLMError.processing("MiMo input audio requires its configured model directory")
        }
        let sidecar = directory.appendingPathComponent("audio_tokenizer")
        let data = try Data(contentsOf: sidecar.appendingPathComponent("config.json"))
        let tokenizerConfig = try JSONDecoder().decode(MiMoV26AudioTokenizerConfiguration.self, from: data)
        let features = try JSONDecoder().decode(MiMoV26AudioFeatures.Configuration.self, from: data)
        let tokenizer = try MiMoV26AuxiliaryLoader.audioTokenizer(in: sidecar, configuration: tokenizerConfig)
        let encoder = try MiMoV26AuxiliaryLoader.audioEncoder(in: directory, configuration: config)
        self.audioTokenizer = tokenizer
        self.audioEncoder = encoder
        self.audioFeatures = features
        return (tokenizer, encoder, features)
    }

    /// Separate image/video batches are scattered to their own token positions.
    /// This preserves interleaved media order because each native vision item
    /// attends only within its own temporal frames.
    func inputEmbeddings(_ input: LMInput) throws -> MLXArray {
        let tokens = input.text.tokens.reshaped(-1)
        let ids = input.text.tokenIds ?? tokens.asArray(Int.self)
        guard tokens.size == ids.count else { throw VLMError.processing("MiMo prompt token identity mismatch") }
        var embeddings = languageModel.embed(tokens)
        func positions(_ token: Int) -> [Int] { ids.indices.filter { ids[$0] == token } }
        func scatter(_ values: MLXArray, at positions: [Int]) throws {
            guard values.ndim == 2, values.dim(0) == positions.count,
                values.dim(1) == languageModel.hiddenSize else {
                throw VLMError.processing("MiMo media embedding count or width does not match its placeholders")
            }
            embeddings[MLXArray(positions)] = values.asType(embeddings.dtype)
        }
        for (pixels, frames, token, modality) in [
            (input.image?.pixels, input.image?.frames, configuration.imageToken, ModelRuntimeRequestModality.vision),
            (input.video?.pixels, input.video?.frames, configuration.videoToken, ModelRuntimeRequestModality.video),
        ] {
            let slots = positions(token)
            guard let pixels else {
                guard slots.isEmpty else { throw VLMError.processing("MiMo visual placeholders have no payload") }
                continue
            }
            guard modalities.contains(modality), let frames, !slots.isEmpty else {
                throw VLMError.processing("MiMo visual payload requires a requested modality, grid, and placeholders")
            }
            try Task.checkCancellation()
            let features = try vision()(pixels, grid: frames)
            try scatter(features, at: slots)
        }
        let audioSlots = positions(configuration.audioToken)
        if let payload = input.audio {
            guard modalities.contains(.audio), !audioSlots.isEmpty else {
                throw VLMError.processing("MiMo audio payload requires the audio modality and placeholders")
            }
            if let features = payload.preEncodedEmbedding {
                try scatter(features, at: audioSlots)
            } else {
                let (tokenizer, encoder, features) = try audio()
                guard payload.sampleRate == features.sampleRate else {
                    throw VLMError.processing("MiMo prepared audio has the wrong sampling rate")
                }
                let samples = payload.waveform.reshaped(-1).asArray(Float.self)
                let lengths = payload.clipSampleCounts ?? [samples.count]
                guard lengths.allSatisfy({ $0 > 0 }), lengths.reduce(0, +) == samples.count else {
                    throw VLMError.processing("MiMo audio clip boundaries do not cover its waveform")
                }
                var offset = 0
                var outputs: [MLXArray] = []
                for length in lengths {
                    try Task.checkCancellation()
                    let mel = try MiMoV26AudioFeatures.logMel(Array(samples[offset..<(offset + length)]), configuration: features)
                    let codes = try tokenizer.tokenize([mel], segmentSize: encoder.configuration.segmentSize)
                    outputs.append(try encoder(codes[0]))
                    offset += length
                }
                try scatter(concatenated(outputs), at: audioSlots)
            }
        } else if !audioSlots.isEmpty {
            throw VLMError.processing("MiMo audio placeholders have no payload")
        }
        return embeddings.expandedDimensions(axis: 0)
    }

    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        guard input.hasMediaContent else {
            return try languageModel.prepare(input, cache: cache, windowSize: windowSize)
        }
        try Task.checkCancellation()
        let embeddings = try inputEmbeddings(input)
        guard embeddings.dim(1) > 0 else { throw VLMError.processing("MiMo prompt is empty") }
        let logits = try chunkedPrefillEmbedding(
            inputEmbedding: embeddings, cache: cache, prefillStepSize: windowSize ?? 512
        ) { languageModel(embeddings: $0, cache: cache) }
        return .logits(LMOutput(logits: logits))
    }
}
