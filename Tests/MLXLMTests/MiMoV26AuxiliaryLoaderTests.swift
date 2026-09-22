// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXNN
import MLXLMCommon
@testable import MLXLLM
@testable import MLXVLM
import Testing

@Suite("MiMo V2.6 auxiliary checkpoint loading", .serialized)
struct MiMoV26AuxiliaryLoaderTests {
    private static let fixtures = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        .appendingPathComponent("Fixtures/MiMoV26")

    private func temporaryDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }

    private func tensors(_ name: String) throws -> [String: MLXArray] {
        try MLX.loadArrays(url: Self.fixtures.appendingPathComponent("\(name)-reference.safetensors"))
    }

    private func configuration<T: Decodable>(_ name: String, as type: T.Type) throws -> T {
        try JSONDecoder().decode(type, from: Data(contentsOf: Self.fixtures.appendingPathComponent("\(name)-config.json")))
    }

    @Test("Multimodal wrapper preserves mixed text quantization and lazy auxiliary state")
    func wrappedCheckpoint() throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        var object = try #require(JSONSerialization.jsonObject(
            with: MiMoV26RuntimeTests.configuration(moe: true)) as? [String: Any])
        object["image_token_id"] = 120
        object["video_token_id"] = 121
        object["audio_token_id"] = 122
        let data = try JSONSerialization.data(withJSONObject: object)
        try data.write(to: directory.appendingPathComponent("config.json"))
        try Data("""
            {"version":2,"weight_format":"mixed_affine_mxfp4","profile":"JANG_2L",
             "quantization":{"method":"mixed","modes":["affine","mxfp4"]}}
            """.utf8).write(to: directory.appendingPathComponent("jang_config.json"))
        let original = try MiMoV26RuntimeTests.model(moe: true)
        try original.update(parameters: ModuleParameters.unflattened(original.parameters().flattened().map {
            ($0.0, $0.0.contains(".mlp.gate.") ? $0.1 + Float(0.001337) : $0.1.asType(.bfloat16))
        }), verify: .all)
        MLXNN.quantize(model: original, filter: { path, module in
            guard module is Quantizable else { return nil }
            if path.hasSuffix("switch_mlp.gate_proj") { return (32, 4, .mxfp4) }
            if path.hasSuffix("switch_mlp.up_proj") { return (64, 2, .affine) }
            if path.hasSuffix("self_attn.o_proj") { return (32, 8, .affine) }
            return (64, 8, .affine)
        })
        let weights = Dictionary(uniqueKeysWithValues: original.parameters().flattened())
        try MLX.save(arrays: weights, url: directory.appendingPathComponent("model.safetensors"))
        let model = try MiMoV26(JSONDecoder().decode(MiMoV26Configuration.self, from: data))
        try model.configure(modelDirectory: directory)
        let base = try JSONDecoder().decode(BaseConfiguration.self, from: data)
        try loadWeights(modelDirectory: directory, model: model,
                        perLayerQuantization: base.perLayerQuantization,
                        jangConfig: try JangLoader.loadConfig(at: directory))
        let after = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        #expect(Set(after.keys) == Set(weights.keys.map { "language_model." + $0 }))
        for (key, value) in weights {
            let actual = try #require(after["language_model." + key])
            #expect(value.dtype == actual.dtype)
            #expect(arrayEqual(value, actual).item(Bool.self))
        }
        let tokens = MLXArray([1, 2, 3]).reshaped(1, 3)
        let result = model(tokens, cache: model.newCache(parameters: nil))
        let expected = original(tokens, cache: original.newCache(parameters: nil))
        #expect(arrayEqual(result, expected).item(Bool.self))
        #expect(model.modalities == [.text])
        #expect(throws: (any Error).self) {
            try MiMoV26(JSONDecoder().decode(MiMoV26Configuration.self, from: data), requesting: [.audio])
        }
    }

    @Test("Vision selects indexed shards, excludes co-located text, and preserves native patch layout")
    func visionCheckpoint() throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let values = try tensors("vision")
        let config = try configuration("vision", as: MiMoV26VisionConfiguration.self)
        var weights: [String: MLXArray] = [:]
        for (name, value) in values where name.hasPrefix("weight.") {
            let key = String(name.dropFirst(7))
            weights["visual." + key] = key == "patch_embed.proj.weight"
                ? value.reshaped(value.dim(0), 3, 2, 2, 2) : value
        }
        weights["model.unused.weight"] = MLXArray([Float(123)])
        let shard = "model-00001-of-00002.safetensors"
        try MLX.save(arrays: weights, url: directory.appendingPathComponent(shard))
        var index = weights.mapValues { _ in shard }
        // An unselected shard must never be opened, even when missing.
        index["model.other.weight"] = "model-00002-of-00002.safetensors"
        try JSONSerialization.data(withJSONObject: ["weight_map": index])
            .write(to: directory.appendingPathComponent("model.safetensors.index.json"))
        let loaded = try MiMoV26AuxiliaryLoader.vision(in: directory, configuration: config)
        let output = try loaded(#require(values["pixels"]), grid: [THW(1, 4, 6), THW(2, 2, 4)])
        #expect(allClose(output, try #require(values["expected"]), rtol: 2e-4, atol: 2e-5).item(Bool.self))

        var object = try #require(JSONSerialization.jsonObject(with: MiMoV26RuntimeTests.configuration()) as? [String: Any])
        object["hidden_size"] = 16
        object["vision_config"] = try JSONSerialization.jsonObject(with: Data(contentsOf: Self.fixtures.appendingPathComponent("vision-config.json")))
        object["image_token_id"] = 120
        object["video_token_id"] = 121
        object["audio_token_id"] = 122
        let wrapped = try MiMoV26(JSONDecoder().decode(MiMoV26Configuration.self,
            from: JSONSerialization.data(withJSONObject: object)))
        try wrapped.configure(modelDirectory: directory)
        #expect(wrapped.parameters().flattened().allSatisfy { $0.0.hasPrefix("language_model.") })
        let ids = [1, 121, 121, 2, 120, 120, 120, 120, 120, 120, 3, 121, 121]
        let input = LMInput(text: .init(tokens: MLXArray(ids), tokenIds: ids),
            image: .init(pixels: try #require(values["pixels"])[..<24], frames: [THW(1, 4, 6)]),
            video: .init(pixels: try #require(values["pixels"])[24...], frames: [THW(2, 2, 4)]))
        let embeddings = try wrapped.inputEmbeddings(input)[0]
        let expected = try #require(values["expected"])
        #expect(allClose(embeddings[MLXArray([4, 5, 6, 7, 8, 9])], expected[..<6], rtol: 2e-4, atol: 2e-5).item(Bool.self))
        #expect(allClose(embeddings[MLXArray([1, 2, 11, 12])], expected[6...], rtol: 2e-4, atol: 2e-5).item(Bool.self))
        #expect(arrayEqual(embeddings[MLXArray([0, 3, 10])], wrapped.languageModel.embed(MLXArray([1, 2, 3]))).item(Bool.self))
        let fullCache = wrapped.newCache(parameters: nil), splitCache = wrapped.newCache(parameters: nil)
        let full = wrapped.languageModel(embeddings: embeddings[.newAxis], cache: fullCache)
        eval(full, fullCache)
        guard case .logits(let prepared) = try wrapped.prepare(input, cache: splitCache, windowSize: 4) else {
            Issue.record("Media prepare must return the actual final logits")
            return
        }
        MiMoV26RuntimeTests.expectChunkParity(prepared.logits[0, -1], full[0, -1], rtol: 2e-4, atol: 2e-5)
    }

    @Test("Media embedding prefill keeps exact cache positions across a single-token tail")
    func mediaChunking() throws {
        var object = try #require(JSONSerialization.jsonObject(with: MiMoV26RuntimeTests.configuration()) as? [String: Any])
        object["hidden_size"] = 16
        object["audio_config"] = try JSONSerialization.jsonObject(with: Data(contentsOf: Self.fixtures.appendingPathComponent("audio-encoder-config.json")))
        object["image_token_id"] = 120
        object["video_token_id"] = 121
        object["audio_token_id"] = 122
        let model = try MiMoV26(JSONDecoder().decode(MiMoV26Configuration.self,
            from: JSONSerialization.data(withJSONObject: object)))
        let ids = [1, 122, 2, 3, 4, 5, 6, 122, 7, 8, 9, 10, 11]
        let features = MLXArray((0..<32).map { Float($0) * 0.03 }).reshaped(2, 16)
        let input = LMInput(text: .init(tokens: MLXArray(ids), tokenIds: ids),
            audio: .init(waveform: MLXArray([Float(0)]), sampleRate: 24000, preEncodedEmbedding: features))
        let embeds = try model.inputEmbeddings(input)
        let expected = model.languageModel(embeddings: embeds, cache: model.newCache(parameters: nil))
        eval(expected)
        for step in [1, 2, 3, 4, 5, 7] {
            let cache = model.newCache(parameters: nil)
            guard case .logits(let result) = try model.prepare(input, cache: cache, windowSize: step) else {
                Issue.record("Media prefill returned no logits"); return
            }
            MiMoV26RuntimeTests.expectChunkParity(result.logits[0, -1], expected[0, -1])
            #expect(cache.allSatisfy { $0.offset == ids.count })
        }
        // Real chat declares stable system/history boundaries before media.
        // Exercise TokenIterator's boundary capture, not only model.prepare.
        for boundaries in [[1], [5], [10], [1, 5, 10]] {
            let stableInput = LMInput(text: input.text, audio: input.audio,
                mediaTokenIds: [122], cacheStablePrefixTokenCounts: boundaries)
            var iterator = try TokenIterator(input: stableInput, model: model,
                parameters: GenerateParameters(maxTokens: 1, temperature: 0))
            #expect(iterator.next() == argMax(expected[0, -1]).item(Int.self))
        }
    }

    @Test("Audio encoder maps native projection keys without rounding its checkpoint")
    func encoderCheckpoint() throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let values = try tensors("audio-encoder")
        let config = try configuration("audio-encoder", as: MiMoV26AudioConfiguration.self)
        var weights: [String: MLXArray] = [:]
        for (name, value) in values where name.hasPrefix("weight.") {
            let key = String(name.dropFirst(7))
                .replacingOccurrences(of: "projection.fc1.", with: "projection.mlp.0.")
                .replacingOccurrences(of: "projection.fc2.", with: "projection.mlp.2.")
            weights[key.hasPrefix("speech_embeddings.") ? key : "audio_encoder." + key] = value
        }
        weights["audio_encoder.input_local_transformer.embed_tokens.weight"] = MLXArray([Float(999)])
        try MLX.save(arrays: weights, url: directory.appendingPathComponent("model.safetensors"))
        let loaded = try MiMoV26AuxiliaryLoader.audioEncoder(in: directory, configuration: config)
        let output = try loaded(#require(values["codes"]))
        #expect(output.dtype == .bfloat16)
        #expect(arrayEqual(output, try #require(values["expected"])).item(Bool.self))
    }

    @Test("Audio tokenizer maps native convolutions and retains FP32 codebooks")
    func tokenizerCheckpoint() throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let values = try tensors("audio-tokenizer")
        let config = try configuration("audio-tokenizer", as: MiMoV26AudioTokenizerConfiguration.self)
        var weights: [String: MLXArray] = [:]
        for (name, value) in values where name.hasPrefix("weight.") {
            var key = String(name.dropFirst(7))
            var tensor = value
            if key.hasPrefix("codebooks.") {
                let index = key.split(separator: ".")[1]
                key = "quantizer.vq.layers.\(index)._codebook.embed"
            } else if key == "conv1.weight" || key == "conv2.weight" {
                tensor = value.transposed(0, 2, 1)
            } else if key == "down_sample_layer.conv.weight" {
                key = "down_sample_layer.0.weight"
                tensor = value.transposed(0, 2, 1)
            }
            weights["encoder." + key] = tensor
        }
        weights["decoder.unused.weight"] = MLXArray([Float(999)])
        weights["encoder.quantizer.vq.layers.0._codebook.cluster_size"] = MLXArray([Float(1)])
        try MLX.save(arrays: weights, url: directory.appendingPathComponent("model.safetensors"))
        let loaded = try MiMoV26AuxiliaryLoader.audioTokenizer(in: directory, configuration: config)
        for (key, value) in loaded.parameters().flattened() {
            #expect(value.dtype == .float32)
            #expect(arrayEqual(value, try #require(values["weight." + key]).asType(.float32)).item(Bool.self))
        }
        let codes = try loaded.tokenize([#require(values["mel0"]), #require(values["mel1"])], segmentSize: 6000)
        for i in 0..<2 {
            #expect(arrayEqual(codes[i], try #require(values["codes\(i)"])).item(Bool.self))
        }
        weights["encoder.conv1.weight"] = MLXArray([Float(1)])
        try MLX.save(arrays: weights, url: directory.appendingPathComponent("model.safetensors"))
        #expect(throws: (any Error).self) {
            try MiMoV26AuxiliaryLoader.audioTokenizer(in: directory, configuration: config)
        }
    }

    @Test("Auxiliary index rejects path traversal and missing selected tensors")
    func invalidIndex() throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let indexURL = directory.appendingPathComponent("model.safetensors.index.json")
        try MLX.save(arrays: ["other": MLXArray([1])], url: directory.appendingPathComponent("model.safetensors"))
        for shard in ["../model.safetensors", "model.safetensors"] {
            try JSONSerialization.data(withJSONObject: ["weight_map": ["visual.missing": shard]])
                .write(to: indexURL)
            #expect(throws: (any Error).self) {
                try MiMoV26AuxiliaryLoader.selectedWeights(in: directory) { $0.hasPrefix("visual.") }
            }
        }
    }
}
