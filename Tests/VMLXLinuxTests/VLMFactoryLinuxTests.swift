// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXVLM

/// How `VLMModelFactory` loads a VLM, on every platform. Each test writes a tiny Idefics3 bundle
/// with random weights. On Apple platforms the tests describe the factory as it always was. Where
/// Apple's media frameworks are absent, the model loads and its processor refuses `UserInput`. On
/// macOS this suite needs the metallib staged next to the test binary.
@Suite(.serialized) struct VLMFactoryLinuxTests {

    private struct FixtureTokenizer: Tokenizer {
        var bosToken: String? { nil }
        var eosToken: String? { nil }
        var unknownToken: String? { nil }
        func encode(text: String, addSpecialTokens: Bool) -> [Int] {
            text.utf8.map { 4 + Int($0) % 100 }
        }
        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
            tokenIds.map(String.init).joined(separator: " ")
        }
        func convertTokenToId(_ token: String) -> Int? { nil }
        func convertIdToToken(_ id: Int) -> String? { nil }
        func applyChatTemplate(
            messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] {
            messages.flatMap {
                encode(text: $0["content"] as? String ?? "", addSpecialTokens: false)
            }
        }
    }

    private struct FixtureTokenizerLoader: TokenizerLoader {
        func load(from directory: URL) async throws -> any Tokenizer { FixtureTokenizer() }
    }

    /// A processor that a caller registers. Its input is recognizable: one token, 7.
    private struct Marker: UserInputProcessor {
        func prepare(input: UserInput) async throws -> LMInput {
            LMInput(tokens: MLXArray([Int32(7)]))
        }
    }

    /// 16 patches of 16 x 16 pixels, merged 2 x 2, give 4 image tokens of id 120.
    private static var configuration: [String: Any] {
        [
            "model_type": "idefics3", "vocab_size": 128, "scale_factor": 2, "image_token_id": 120,
            "text_config": [
                "model_type": "llama", "hidden_size": 64, "num_hidden_layers": 2,
                "intermediate_size": 128, "num_attention_heads": 4, "num_key_value_heads": 2,
                "rms_norm_eps": 1e-5, "vocab_size": 128, "rope_theta": 10000,
            ],
            "vision_config": [
                "model_type": "idefics3_vision", "hidden_size": 64, "num_hidden_layers": 2,
                "intermediate_size": 128, "num_attention_heads": 2, "patch_size": 16,
                "image_size": 64,
            ],
        ]
    }

    private static func write(_ object: Any, to url: URL) throws {
        try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys]).write(to: url)
    }

    /// Writes the bundle to a new temporary directory: the model's own parameters, random (key 0),
    /// optionally quantized to 4 bits in groups of 64, and the two configuration files.
    private static func bundle(
        quantized: Bool = false, processorClass: String = "Idefics3Processor"
    ) throws -> URL {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            "vlm-factory-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        var configuration = Self.configuration
        if quantized { configuration["quantization"] = ["group_size": 64, "bits": 4] }
        try Device.withDefaultDevice(.cpu) {
            let model = Idefics3(
                try JSONDecoder().decode(
                    Idefics3Configuration.self,
                    from: JSONSerialization.data(withJSONObject: Self.configuration)))
            let parameters = model.parameters().flattened()
            let keys = MLXRandom.split(key: MLXRandom.key(0), into: parameters.count)
            let random = zip(parameters, keys).map { parameter, key in
                (parameter.0, MLXRandom.normal(parameter.1.shape, scale: 0.02, key: key))
            }
            _ = model.update(parameters: ModuleParameters.unflattened(random))
            if quantized { quantize(model: model, groupSize: 64, bits: 4) }
            try MLX.save(
                arrays: Dictionary(uniqueKeysWithValues: model.parameters().flattened()),
                url: directory.appendingPathComponent("model.safetensors"))
        }
        try write(configuration, to: directory.appendingPathComponent("config.json"))
        try write(
            [
                "processor_class": processorClass, "image_mean": [0.5, 0.5, 0.5],
                "image_std": [0.5, 0.5, 0.5], "size": ["longest_edge": 64],
            ], to: directory.appendingPathComponent("preprocessor_config.json"))
        return directory
    }

    private static func load(
        _ directory: URL, factory: VLMModelFactory = .shared
    ) async throws -> ModelContext {
        try await factory.load(from: directory, using: FixtureTokenizerLoader())
    }

    /// The prepared-input path: a caller that builds `LMInput` itself runs the model, quantized
    /// bundles included.
    @Test(arguments: [false, true]) func loadsAndRunsFromAPreparedInput(quantized: Bool)
        async throws
    {
        let directory = try Self.bundle(quantized: quantized)
        defer { try? FileManager.default.removeItem(at: directory) }
        let context = try await Self.load(directory)
        let model = try #require(context.model as? Idefics3)
        #expect(
            model.leafModules().flattened().contains { $0.1 is QuantizedLinear } == quantized)
        let input = LMInput(
            text: .init(
                tokens: MLXArray([1, 2, 120, 120, 120, 120, 3]).expandedDimensions(axis: 0)),
            image: .init(
                pixels: MLXRandom.uniform(-1 ..< 1, [1, 64, 64, 3], key: MLXRandom.key(1))))
        let result = try model.prepare(
            input, cache: model.newCache(parameters: nil), windowSize: nil)
        guard case .logits(let output) = result else {
            Issue.record("prepare returned tokens, not logits")
            return
        }
        #expect(output.logits.shape == [1, 7, 128])
        let finite = output.logits.asType(.float32).asArray(Float.self).allSatisfy { $0.isFinite }
        #expect(finite)
    }

    @Test func processorForTheBundle() async throws {
        let directory = try Self.bundle()
        defer { try? FileManager.default.removeItem(at: directory) }
        let context = try await Self.load(directory)
        #if canImport(CoreImage) && canImport(AVFoundation) && canImport(Accelerate)
            #expect(context.processor is Idefics3Processor)
        #else
            #expect(
                PreparedInputRequiredProcessor.message("Idefics3Processor")
                    == "Idefics3Processor needs Apple's media frameworks (CoreImage, AVFoundation, "
                    + "Accelerate), which this platform lacks. Register a processor for "
                    + "\"Idefics3Processor\", or prepare the LMInput yourself.")
            let refusal = VLMError.processing(
                PreparedInputRequiredProcessor.message("Idefics3Processor"))
            await #expect(throws: refusal) {
                try await context.processor.prepare(input: UserInput(prompt: "hello"))
            }
            await #expect(throws: refusal) {
                try await context.processor.prepare(
                    input: UserInput(prompt: "hello", images: [.array(.zeros([64, 64, 3]))]))
            }
        #endif
    }

    @Test func unknownProcessorType() async throws {
        let directory = try Self.bundle(processorClass: "NoSuchProcessor")
        defer { try? FileManager.default.removeItem(at: directory) }
        #if canImport(CoreImage) && canImport(AVFoundation) && canImport(Accelerate)
            let error = await #expect(throws: ModelFactoryError.self) {
                _ = try await Self.load(directory)
            }
            guard case .unsupportedProcessorType(let type)? = error else {
                Issue.record("expected unsupportedProcessorType, got \(String(describing: error))")
                return
            }
            #expect(type == "NoSuchProcessor")
        #else
            let context = try await Self.load(directory)
            await #expect(
                throws: VLMError.processing(
                    PreparedInputRequiredProcessor.message("NoSuchProcessor"))
            ) {
                try await context.processor.prepare(input: UserInput(prompt: "hello"))
            }
        #endif
    }

    /// A processor the caller registers wins on every platform.
    @Test func registeredProcessorIsUsed() async throws {
        let directory = try Self.bundle()
        defer { try? FileManager.default.removeItem(at: directory) }
        let factory = VLMModelFactory(
            typeRegistry: VLMTypeRegistry.shared,
            processorRegistry: ProcessorTypeRegistry(creators: [
                "Idefics3Processor": { _, _ in Marker() }
            ]),
            modelRegistry: VLMRegistry.shared)
        let context = try await Self.load(directory, factory: factory)
        let prepared = try await context.processor.prepare(input: UserInput(prompt: "hello"))
        #expect(prepared.text.tokens.asArray(Int32.self) == [7])
    }

    #if !canImport(Accelerate)
        /// MiMo V2.6 computes its log-mel features with vDSP: without Accelerate they raise an
        /// error, never silence. Its resampler is plain Swift and runs.
        @Test func mimoAudioFeaturesNeedAccelerate() throws {
            let configuration = try JSONDecoder().decode(
                MiMoV26AudioFeatures.Configuration.self,
                from: Data(
                    #"{"sampling_rate": 16000, "nfft": 64, "hop_length": 16, "window_size": 64, "n_mels": 8, "fmin": 0}"#
                        .utf8))
            let samples = (0 ..< 480).map { Float(sin(Double($0) * 0.05)) }
            #expect(
                throws: VLMError.processing(
                    "MiMo V2.6 audio features need Accelerate, which this platform lacks.")
            ) {
                _ = try MiMoV26AudioFeatures.logMel(samples, configuration: configuration)
            }
            #expect(try MiMoV26AudioFeatures.resample(samples, from: 16000, to: 16000) == samples)
            let resampled = try MiMoV26AudioFeatures.resample(samples, from: 48000, to: 16000)
            #expect(resampled.count == 160)
            let finite = resampled.allSatisfy { $0.isFinite }
            #expect(finite)
        }

        /// Nemotron-H Omni encodes raw audio with vDSP. Without Accelerate it raises an error for
        /// raw audio, and an audio embedding that is already encoded runs.
        @Test func nemotronAudioNeedsAPreEncodedEmbedding() throws {
            let model = NemotronHOmni(
                try JSONDecoder().decode(
                    NemotronHOmniConfiguration.self,
                    from: Data(
                        #"""
                        {"vocab_size": 64, "hidden_size": 64, "num_attention_heads": 4,
                         "num_key_value_heads": 2, "mamba_num_heads": 4, "mamba_head_dim": 16,
                         "ssm_state_size": 16, "conv_kernel": 4, "n_groups": 2,
                         "intermediate_size": 128, "moe_intermediate_size": 64,
                         "moe_shared_expert_intermediate_size": 64, "n_routed_experts": 4,
                         "num_experts_per_tok": 2, "hybrid_override_pattern": "*-"}
                        """#.utf8)))
            // Two audio placeholders (token 27, the configuration's sound token).
            let tokens = MLXArray([1, 27, 27, 2]).expandedDimensions(axis: 0)
            let waveform = MLXArray.zeros([1600])
            let raw = LMInput(text: .init(tokens: tokens), audio: .init(waveform: waveform))
            #expect(
                throws: VLMError.processing(
                    "Nemotron-H Omni audio input needs Accelerate, which this platform lacks.")
            ) {
                _ = try model.prepare(raw, cache: model.newCache(parameters: nil), windowSize: nil)
            }
            let encoded = LMInput(
                text: .init(tokens: tokens),
                audio: .init(
                    waveform: waveform,
                    preEncodedEmbedding: MLXRandom.normal([2, 64], key: MLXRandom.key(2))))
            let result = try model.prepare(
                encoded, cache: model.newCache(parameters: nil), windowSize: nil)
            guard case .logits(let output) = result else {
                Issue.record("prepare returned tokens, not logits")
                return
            }
            #expect(output.logits.shape == [1, 4, 64])
            let finite = output.logits.asType(.float32).asArray(Float.self)
                .allSatisfy { $0.isFinite }
            #expect(finite)
        }
    #endif
}
