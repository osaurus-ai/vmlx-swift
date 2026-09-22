// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import CoreImage
import CoreMedia
import Foundation
import MLX
import MLXLMCommon
@testable import MLXVLM
import Testing

private struct MiMoMediaTestTokenizer: Tokenizer {
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
    func encode(text: String, addSpecialTokens: Bool) -> [Int] { text.utf8.map { Int($0) + 2000 } }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { nil }
    func applyChatTemplate(messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
                           additionalContext: [String: any Sendable]?) throws -> [Int] {
        var tokens: [Int] = []
        for message in messages {
            if let text = message["content"] as? String {
                tokens += encode(text: text, addSpecialTokens: false)
            } else if let parts = message["content"] as? [[String: String]] {
                for part in parts {
                    switch part["type"] {
                    case "image": tokens += [151652, 151655, 151653]
                    case "video": tokens += [151652, 151656, 151653]
                    case "audio": tokens += [151673, 151669, 151674]
                    case "text": tokens += encode(text: part["text"] ?? "", addSpecialTokens: false)
                    default: break
                    }
                }
            }
        }
        return tokens
    }
}

@Suite("MiMo V2.6 native media prompt preparation", .serialized)
struct MiMoV26ProcessorTests {
    private func processor() throws -> MiMoV26Processor {
        let file = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .appendingPathComponent("Fixtures/MiMoV26/processor-config.json")
        return try MiMoV26Processor(JSONDecoder().decode(MiMoV26ProcessorConfiguration.self,
            from: Data(contentsOf: file)), tokenizer: MiMoMediaTestTokenizer())
    }

    @Test("Expansion preserves interleaved image, video, audio, and text; rejects cardinality mismatches")
    func expansion() throws {
        let triples = [[10, 11, 12], [10, 21, 12], [30, 31, 32]]
        let tokens = [1, 10, 21, 12, 2, 30, 31, 32, 3, 10, 11, 12, 4, 10, 21, 12]
        let replacements = [[[10, 11, 11, 12]], [[40, 41], [42, 43]], [[30, 31, 31, 32]]]
        #expect(try MiMoV26Processor.expand(tokens, triples: triples, replacements: replacements)
            == [1, 40, 41, 2, 30, 31, 31, 32, 3, 10, 11, 11, 12, 4, 42, 43])
        #expect(throws: (any Error).self) {
            try MiMoV26Processor.expand(Array(tokens.dropLast(3)), triples: triples, replacements: replacements)
        }
        #expect(throws: (any Error).self) {
            try MiMoV26Processor.expand(tokens + [30, 31, 32], triples: triples, replacements: replacements)
        }
    }

    @Test("Structured messages retain media order and reasoning/tool history")
    func messageOrder() throws {
        let parts: [Chat.ContentPart] = [.text("before"), .video, .text("between"), .image, .audio]
        let message = Chat.Message(role: .assistant, content: "beforebetween", reasoningContent: "reasoning",
            toolCalls: [.init(id: "call-1", function: .init(name: "lookup", arguments: [:]))], contentParts: parts)
        let result = MiMoV26MessageGenerator().generate(message: message)
        let actual = try #require(result["content"] as? [[String: String]])
        #expect(actual.map { $0["type"] } == ["text", "video", "text", "image", "audio"])
        #expect(result["reasoning_content"] as? String == "reasoning")
        #expect(result["tool_calls"] != nil)
    }

    @Test("Real pixel and PCM preparation expands deterministic cache tokens with clip boundaries")
    func preparedMedia() async throws {
        let processor = try processor()
        let red = CIImage(color: CIColor(red: 1, green: 0, blue: 0)).cropped(to: CGRect(x: 0, y: 0, width: 64, height: 128))
        let frames = [UserInput.VideoFrame(frame: red, timeStamp: CMTime(seconds: 1, preferredTimescale: 100)),
                      UserInput.VideoFrame(frame: red, timeStamp: CMTime(seconds: 2, preferredTimescale: 100))]
        let samples = Array(repeating: Float(0.1), count: 3840)
        let message = Chat.Message(role: .user, content: "describe", images: [.ciImage(red)], videos: [.frames(frames)],
            audios: [.samples(samples, sampleRate: 24000), .samples(Array(samples.prefix(2000)), sampleRate: 24000)],
            contentParts: [.video, .text("describe"), .audio, .image, .audio])
        let result = try await processor.prepare(input: UserInput(chat: [message], additionalContext: ["enable_thinking": false]))
        let ids = try #require(result.text.tokenIds)
        #expect(ids.first == 151670)
        let image = try #require(result.image), video = try #require(result.video), audio = try #require(result.audio)
        let imageGrid = try #require(image.frames?.first), videoGrid = try #require(video.frames?.first)
        #expect(ids.filter { $0 == 151655 }.count == imageGrid.product / 4)
        #expect(ids.filter { $0 == 151656 }.count == videoGrid.product / 4)
        #expect(ids.filter { $0 == 151669 }.count == 3) // ceil(17/16) + ceil(9/16)
        #expect(audio.clipSampleCounts == [3840, 2000])
        #expect(audio.sampleRate == 24000 && audio.waveform.size == 5840)
        #expect(result.cacheScopeSalt == "reasoning=off")
        #expect(result.mediaTokenIds == [151655, 151656, 151669])
        #expect(ids.firstIndex(of: 151669)! < ids.firstIndex(of: 151655)!)
        #expect(ids.lastIndex(of: 151669)! > ids.lastIndex(of: 151655)!)
        #expect(computeMediaSalt(for: result) != nil)
        let again = try await processor.prepare(input: UserInput(chat: [message], additionalContext: ["enable_thinking": false]))
        #expect(computeMediaSalt(for: result) == computeMediaSalt(for: again))
        #expect(ids == again.text.tokenIds)
    }
}
