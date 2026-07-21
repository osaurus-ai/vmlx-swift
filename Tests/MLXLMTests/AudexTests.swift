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

@Suite("Nemotron-Labs Audex-2B")
struct AudexTests {
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
}
