// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXHuggingFace
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM
@preconcurrency import VMLXTokenizers

/// LFM3-Audio / LFM3-Omni speech input: feature geometry, encoder sizing, and (opt-in) parity
/// with a released bundle.
@Suite("LFM3-Omni", .serialized)
struct LFM3OmniTests {

    @Test func frameAndTokenCountsFollowTheReference() {
        // No centering: 400 samples is one frame, every further 160 (or part) adds one.
        #expect(LFM3SpeechFeatures.frameCount(samples: 100) == 1)
        #expect(LFM3SpeechFeatures.frameCount(samples: 400) == 1)
        #expect(LFM3SpeechFeatures.frameCount(samples: 401) == 2)
        #expect(LFM3SpeechFeatures.frameCount(samples: 75_983) == 474)
        #expect(LFM3SpeechFeatures.encoderTokens(frames: 474) == 60)
        #expect(LFM3SpeechFeatures.encoderTokens(frames: 8) == 1)
        #expect(LFM3SpeechFeatures.encoderTokens(frames: 9) == 2)
    }

    @Test func trunkFeedForwardUsesLFM2Sizing() throws {
        let json = #"""
            {"hidden_size": 768, "intermediate_size": 2560, "num_attention_heads": 8,
             "num_key_value_heads": 4, "num_hidden_layers": 2, "block_auto_adjust_ff_dim": true,
             "block_multiple_of": 256, "block_ffn_dim_multiplier": 1.0,
             "layer_types": ["conv", "full_attention"],
             "rope_parameters": {"rope_theta": 1000000.0, "rope_type": "default"}}
            """#
        let trunk = try JSONDecoder().decode(
            LFM3SpeechTrunkConfiguration.self, from: Data(json.utf8))
        #expect(trunk.feedForwardSize == 1792, "2560 -> 1706 -> rounded up to 1792")
        #expect(trunk.headDim == 96)
        #expect(trunk.ropeTheta == 1e6)
    }

    @Test func placeholdersExpandPerClip() throws {
        // Two clips: 1.0 s (101 frames -> 13 tokens) and 0.3 s (28 frames -> 4 tokens).
        let counts = [16_000, 4_800].map {
            LFM3SpeechFeatures.encoderTokens(frames: LFM3SpeechFeatures.frameCount(samples: $0))
        }
        #expect(counts == [13, 4])
        let out = try LFM3VLProcessor.expand(
            [1, 7, 2, 7, 3], placeholder: 7,
            replacements: counts.map { Array(repeating: 7, count: $0) })
        #expect(out.tokens.count == 3 + 13 + 4)
    }

    // MARK: - Released bundle parity (opt-in)

    /// `LFM3_AUDIO_BUNDLE` (an LFM3-Audio or LFM3-Omni bundle), `LFM3_AUDIO_REF` (JSON +
    /// `.safetensors` from the reference script) and `LFM3_AUDIO_WAV` (16 kHz mono).
    @Test func releasedBundleMatchesReference() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let bundle = env["LFM3_AUDIO_BUNDLE"], let refPath = env["LFM3_AUDIO_REF"],
            let wavPath = env["LFM3_AUDIO_WAV"]
        else { return }
        let fp32 = env["LFM3_FP32"] == "1"
        let ref = try JSONSerialization.jsonObject(
            with: Data(contentsOf: URL(fileURLWithPath: refPath))) as! [String: Any]
        let cases = try #require(ref["cases"] as? [String: [String: Any]])
        let tensors = try MLX.loadArrays(url: URL(fileURLWithPath: refPath + ".safetensors"))
        let wav = URL(fileURLWithPath: wavPath)

        let context = try await VLMModelFactory.shared.load(
            from: URL(fileURLWithPath: bundle), using: #huggingFaceTokenizerLoader())
        let model = try #require(context.model as? LFM3Omni)
        if fp32 {
            let upcast = model.parameters().flattened().map { key, value in
                (key, value.dtype == .bfloat16 ? value.asType(.float32) : value)
            }
            try model.update(parameters: ModuleParameters.unflattened(upcast), verify: .none)
            eval(model)
        }
        let stops = Set(context.configuration.eosTokenIds).union(
            context.configuration.extraEOSTokens.compactMap {
                context.tokenizer.convertTokenToId($0)
            })
        #expect(stops.isSuperset(of: [124900, 125012, 125015]), "stops \(stops.sorted())")

        // Front end and encoder on the same PCM.
        let pcm = try LFM3OmniProcessor.waveform(.url(wav))
        let features = LFM3SpeechFeatures.logMel(pcm)
        let refFeatures = try #require(tensors["features"])
        #expect(features.shape == refFeatures.shape)
        let featureDiff = abs(features - refFeatures).max().item(Float.self)
        let speech = model.speechFeatures(pcm: pcm).asType(.float32)
        let refSpeech = try #require(tensors["speech"])
        let speechDiff = abs(speech - refSpeech).max().item(Float.self)
        let speechScale = abs(refSpeech).max().item(Float.self)
        print("[LFM3-Omni] log-mel max |diff| \(featureDiff); speech max |diff| \(speechDiff) (scale \(speechScale))")
        // float32 FFT here vs float64 in the reference: measured 0.00126 max on the
        // released test clip (near-silent bins, where log10 is steepest).
        #expect(featureDiff < 5e-3, "log-mel diff \(featureDiff)")
        #expect(speechDiff <= (fp32 ? 1e-3 : 0.08) * max(1, speechScale))

        for (name, c) in cases.sorted(by: { $0.key < $1.key }) {
            let question = name == "asr" ? "" : "What number does the speaker say is their favorite? Answer in text."
            var chat: [Chat.Message] = []
            if name == "asr" { chat.append(.system("Perform ASR in English.")) }
            chat.append(.user(question, audios: [.url(wav)]))
            let prepared = try await context.processor.prepare(input: UserInput(chat: chat))
            let ids = prepared.text.tokens.reshaped(-1).asArray(Int.self)
            let refIds = try #require(c["ids"] as? [Int])
            #expect(ids == refIds, "\(name): processor ids differ (\(ids.count) vs \(refIds.count))")

            let refGreedy = try #require(c["greedy"] as? [Int])
            var iterator = try TokenIterator(
                input: prepared, model: model,
                parameters: GenerateParameters(maxTokens: refGreedy.count, temperature: 0))
            var greedy: [Int] = []
            while greedy.count < refGreedy.count, let token = iterator.next() {
                greedy.append(token)
                if stops.contains(token) { break }
            }
            let prefix = zip(greedy, refGreedy).prefix { $0 == $1 }.count
            let text = context.tokenizer.decode(tokenIds: greedy)
            print("[LFM3-Omni] \(name): greedy prefix \(prefix)/\(refGreedy.count): \(text)")
            #expect(text.contains("42"), "\(name): answer must carry the spoken number")
            #expect(prefix >= min(refGreedy.count, 8), "\(name): greedy diverged at \(prefix)")
        }
    }
}
