// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXNN
import MLXLMCommon
@testable import MLXVLM
import Testing

@Suite("MiMo V2.6 Accelerate audio frontend", .serialized)
struct MiMoV26AudioFeaturesTests {
    @Test("Audio cache identity includes clip boundaries")
    func clipCacheIdentity() {
        let pcm = MLXArray([Float(1), 2, 3, 4])
        func input(_ counts: [Int]?) -> LMInput {
            LMInput(text: .init(tokens: MLXArray([1, 2])),
                    audio: .init(waveform: pcm, sampleRate: 24000, clipSampleCounts: counts))
        }
        #expect(computeMediaSalt(for: input([2, 2])) == computeMediaSalt(for: input([2, 2])))
        #expect(computeMediaSalt(for: input([2, 2])) != computeMediaSalt(for: input([1, 3])))
        #expect(computeMediaSalt(for: input([2, 2])) != computeMediaSalt(for: input(nil)))
    }

    @Test("Audio tokenizer preserves batched padding, skip layer, and residual codebook indices")
    func tokenizerReference() throws {
        let directory = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .appendingPathComponent("Fixtures/MiMoV26")
        let tensors = try MLX.loadArrays(url: directory.appendingPathComponent("audio-tokenizer-reference.safetensors"))
        let config = try JSONDecoder().decode(MiMoV26AudioTokenizerConfiguration.self,
            from: Data(contentsOf: directory.appendingPathComponent("audio-tokenizer-config.json")))
        let model = try MiMoV26AudioTokenizer(config)
        let weights = Dictionary(uniqueKeysWithValues: tensors.compactMap { key, value in
            key.hasPrefix("weight.") ? (String(key.dropFirst(7)), value) : nil
        })
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        let mels = try [#require(tensors["mel0"]), #require(tensors["mel1"])]
        let features = try Device.withDefaultDevice(.cpu) { try model.features(mels) }
        let priorDevice = Device.defaultDevice()
        let codes = try model.tokenize(mels, segmentSize: 6000)
        #expect(Device.defaultDevice() == priorDevice)
        #expect(throws: (any Error).self) { try model.tokenize(mels, segmentSize: 0) }
        #expect(Device.defaultDevice() == priorDevice)
        for index in 0..<2 {
            #expect(allClose(features[index], try #require(tensors["features\(index)"]), rtol: 1e-5, atol: 1e-5).item(Bool.self))
            #expect(arrayEqual(codes[index], try #require(tensors["codes\(index)"])).item(Bool.self))
        }
    }

    @Test("BF16 speech embedding and local transformer preserve clip grouping")
    func encoderReference() throws {
        let directory = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .appendingPathComponent("Fixtures/MiMoV26")
        let tensors = try MLX.loadArrays(url: directory.appendingPathComponent("audio-encoder-reference.safetensors"))
        let config = try JSONDecoder().decode(MiMoV26AudioConfiguration.self,
            from: Data(contentsOf: directory.appendingPathComponent("audio-encoder-config.json")))
        let model = MiMoV26AudioEncoder(config)
        let weights = Dictionary(uniqueKeysWithValues: tensors.compactMap { key, value in
            key.hasPrefix("weight.") ? (String(key.dropFirst(7)), value) : nil
        })
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        let codes = try #require(tensors["codes"])
        let output = try model(codes)
        #expect(output.dtype == .bfloat16)
        #expect(allClose(output, try #require(tensors["expected"]), rtol: 0, atol: 0).item(Bool.self))
        let first = try model(codes[..<4])
        let tail = try model(codes[4...])
        #expect(arrayEqual(output, concatenated([first, tail], axis: 0)).item(Bool.self))
    }

    @Test("Sinc-Hann resampling and magnitude HTK mel match the reference")
    func referenceParity() throws {
        let directory = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .appendingPathComponent("Fixtures/MiMoV26")
        let arrays = try MLX.loadArrays(url: directory.appendingPathComponent("audio-features-reference.safetensors"))
        let config = try JSONDecoder().decode(MiMoV26AudioFeatures.Configuration.self,
            from: Data(contentsOf: directory.appendingPathComponent("audio-features-config.json")))
        let raw = try #require(arrays["raw"]).asArray(Float.self)
        let samples = try MiMoV26AudioFeatures.resample(raw, from: 22050, to: 24000)
        let expected = try #require(arrays["resampled"])
        #expect(samples.count == 2400)
        #expect(allClose(MLXArray(samples), expected, rtol: 1e-6, atol: 1e-7).item(Bool.self))
        let mel = try MiMoV26AudioFeatures.logMel(samples, configuration: config)
        #expect(mel.shape == [11, 128])
        #expect(allClose(mel, try #require(arrays["mel"]), rtol: 1e-6, atol: 1e-6).item(Bool.self))
        #expect(try MiMoV26AudioFeatures.resample(raw, from: 22050, to: 22050) == raw)
    }
}
