// Copyright © 2026 Jinho Jang (eric@jangq.ai)

import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXVLM

private struct AudexTestTokenizer: Tokenizer {
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
    func convertTokenToId(_ token: String) -> Int? { token == "<so_embedding>" ? 29 : nil }
    func convertIdToToken(_ id: Int) -> String? { id == 29 ? "<so_embedding>" : nil }

    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        let text = messages.compactMap { $0["content"] as? String }.joined()
        let count = text.components(separatedBy: "<so_embedding>").count - 1
        return [1] + Array(repeating: 29, count: count) + [2]
    }
}

private struct AudexHistoryTestTokenizer: Tokenizer {
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
    func convertTokenToId(_ token: String) -> Int? { token == "<so_embedding>" ? 29 : nil }
    func convertIdToToken(_ id: Int) -> String? { id == 29 ? "<so_embedding>" : nil }

    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        messages.enumerated().flatMap { index, message in
            let content = message["content"] as? String ?? ""
            let markerCount = content.components(separatedBy: "<so_embedding>").count - 1
            return [100 + index] + Array(repeating: 29, count: markerCount) + [200 + index]
        }
    }
}

@Suite("Nemotron-Labs Audex")
struct AudexTests {
    @Test("Nemotron-H Audex configuration and registry entry decode")
    func nemotronHAudexConfigurationAndRegistry() throws {
        let json = Data(
            #"""
            {
              "model_type": "nemotron_h_audex",
              "vocab_size": 205312,
              "hidden_size": 2688,
              "num_hidden_layers": 52,
              "num_attention_heads": 32,
              "num_key_value_heads": 2,
              "mamba_num_heads": 64,
              "mamba_head_dim": 64,
              "ssm_state_size": 128,
              "conv_kernel": 4,
              "n_groups": 8,
              "intermediate_size": 21504,
              "moe_intermediate_size": 1856,
              "moe_shared_expert_intermediate_size": 3712,
              "n_routed_experts": 128,
              "n_shared_experts": 1,
              "num_experts_per_tok": 6,
              "hybrid_override_pattern": "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME",
              "audio_config": {
                "d_model": 1280,
                "encoder_attention_heads": 20,
                "encoder_ffn_dim": 5120,
                "encoder_layers": 32,
                "max_source_positions": 1500,
                "num_mel_bins": 128
              },
              "audio_projector_intermediate_size": 4096,
              "audio_projector_norm_eps": 0.00001,
              "sound_token_id": 29,
              "sound_start_token_id": 30,
              "sound_end_token_id": 31,
              "sound_target_rate": 16000,
              "sound_clip_duration": 30.0,
              "sound_embedding_size": 750
            }
            """#.utf8)

        let config = try JSONDecoder().decode(AudexHConfiguration.self, from: json)
        #expect(config.language.modelType == "nemotron_h_audex")
        #expect(config.language.hybridOverridePattern.count == 52)
        #expect(config.language.nRoutedExperts == 128)
        #expect(config.language.numExpertsPerTok == 6)
        #expect(config.soundTargetRate == 16_000)
        #expect(config.soundEmbeddingSize == 750)
        #expect(VLMTypeRegistry.supportedModelTypes.contains("nemotron_h_audex"))
    }

    @Test("Whisper features match the official processor on a deterministic waveform")
    func whisperFeatureParity() {
        let sampleRate = 16_000
        let pcm = (0 ..< sampleRate * 2).map { index -> Float in
            let t = Float(index) / Float(sampleRate)
            return 0.35 * sin(2 * .pi * 440 * t) + 0.1 * sin(2 * .pi * 1000 * t)
        }
        let features = audexWhisperFeatures(pcm)
        MLX.eval(features)
        #expect(features.shape == [1, 128, 3000])
        #expect(abs(features.min().item(Float.self) - (-0.5920956)) < 2e-4)
        #expect(abs(features.max().item(Float.self) - 1.4079044) < 2e-4)
        #expect(abs(features.mean().item(Float.self) - (-0.5820658)) < 2e-4)
        #expect(abs(features[0, 20, 100].item(Float.self) - 1.2099414) < 2e-4)
    }

    @Test("A 30-second clip expands to exactly 750 audio placeholders")
    func processorPlaceholderCount() async throws {
        let configJSON = Data(#"{"processor_class":"Qwen2AudioProcessor"}"#.utf8)
        let config = try JSONDecoder().decode(AudexProcessorConfiguration.self, from: configJSON)
        let processor = AudexProcessor(config, tokenizer: AudexTestTokenizer())
        let input = UserInput(
            prompt: "Transcribe the speech.",
            audios: [.samples([Float](repeating: 0, count: 16_000), sampleRate: 16_000)])
        let prepared = try await processor.prepare(input: input)
        let ids = prepared.text.tokenIds ?? []
        #expect(ids.filter { $0 == 29 }.count == 750)
        #expect(prepared.audio?.waveform.shape == [1, 480_000])
        #expect(prepared.mediaTokenIds == [29])
    }

    @Test("Audio markers stay attached to their structured history messages")
    func processorPreservesAudioMessageAssociation() async throws {
        let configJSON = Data(#"{"processor_class":"Qwen2AudioProcessor"}"#.utf8)
        let config = try JSONDecoder().decode(AudexProcessorConfiguration.self, from: configJSON)
        let processor = AudexProcessor(config, tokenizer: AudexHistoryTestTokenizer())
        let oneSecond = [Float](repeating: 0, count: 16_000)
        let input = UserInput(chat: [
            .user("Transcribe the first clip.", audios: [.samples(oneSecond, sampleRate: 16_000)]),
            .assistant("Prior transcript."),
            .user("Transcribe the second clip.", audios: [.samples(oneSecond, sampleRate: 16_000)]),
        ])

        let prepared = try await processor.prepare(input: input)
        let ids = prepared.text.tokenIds ?? []
        let firstMessageEnd = try #require(ids.firstIndex(of: 200))
        let secondMessageStart = try #require(ids.firstIndex(of: 102))
        let secondMessageEnd = try #require(ids.firstIndex(of: 202))
        let firstMarkers = ids[1 ..< firstMessageEnd].filter { $0 == 29 }
        let secondMarkers = ids[(secondMessageStart + 1) ..< secondMessageEnd].filter { $0 == 29 }

        #expect(firstMarkers.count == 750)
        #expect(secondMarkers.count == 750)
        #expect(ids.filter { $0 == 29 }.count == 1_500)
        #expect(prepared.audio?.waveform.shape == [2, 480_000])
    }
}
