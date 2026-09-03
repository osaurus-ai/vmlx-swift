import Foundation
import MLXLLM
import MLXVLM
import XCTest

/// Decodes the REAL `config.json` from the downloaded MLX bundle rather than a
/// hand-written fixture. A fixture only proves the struct matches what I typed;
/// this proves it matches what NVIDIA/mlx-community actually shipped.
///
/// Skips (rather than fails) when the bundle is absent, so CI without the 21 GB
/// download stays green.
public class NemotronVoiceChatConfigTests: XCTestCase {

    private static let bundlePath =
        ("~/models/nemotron-voicechat-src/VoiceChat-11B-mlx-bf16" as NSString)
        .expandingTildeInPath

    private func loadRealConfig() throws -> NemotronVoiceChatConfiguration? {
        let url = URL(fileURLWithPath: Self.bundlePath).appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(NemotronVoiceChatConfiguration.self, from: data)
    }

    func testDecodesRealBundleConfig() throws {
        guard let c = try loadRealConfig() else {
            throw XCTSkip("VoiceChat MLX bundle not present at \(Self.bundlePath)")
        }

        // ── backbone ───────────────────────────────────────────────────────
        XCTAssertEqual(c.textConfig.hiddenSize, 4480)
        XCTAssertEqual(c.textConfig.numHiddenLayers, 56)
        XCTAssertEqual(c.textConfig.vocabSize, 131072)

        // The pattern string must agree with the weights: 27 Mamba / 25 MLP /
        // 4 attention. If a future revision changes the pattern, this catches it
        // before a load produces a silently wrong layer stack.
        let pattern = Array(c.textConfig.hybridOverridePattern)
        XCTAssertEqual(pattern.count, 56, "pattern length must equal layer count")
        XCTAssertEqual(pattern.filter { $0 == "M" }.count, 27, "Mamba2 layers")
        XCTAssertEqual(pattern.filter { $0 == "-" }.count, 25, "MLP layers")
        XCTAssertEqual(pattern.filter { $0 == "*" }.count, 4, "attention layers")

        // ── audio in ───────────────────────────────────────────────────────
        XCTAssertEqual(c.inputSampleRate, 16000)
        XCTAssertEqual(c.audioConfig.preprocessor.sampleRate, 16000)
        XCTAssertEqual(c.audioConfig.preprocessor.features, 128)
        XCTAssertEqual(c.audioConfig.outputDim, c.textConfig.hiddenSize,
                       "perception output must match LLM hidden size")

        // ── audio out ──────────────────────────────────────────────────────
        XCTAssertEqual(c.outputSampleRate, 22050)
        XCTAssertEqual(c.codecConfig.sampleRate, c.outputSampleRate)
        XCTAssertEqual(c.frameDuration, 0.08, accuracy: 1e-9)
        XCTAssertEqual(c.framesPerSecond, 12.5, accuracy: 1e-9)

        // ── the trap: RNN-T decoder/joint are INSIDE audio_config ───────────
        XCTAssertNotNil(c.audioConfig.decoder,
                        "rnnt decoder must decode from audio_config, not top level")
        XCTAssertNotNil(c.audioConfig.joint,
                        "rnnt joint must decode from audio_config, not top level")

        // Blank is one PAST the vocabulary, not a member of it.
        if let vocab = c.rnntVocabulary {
            XCTAssertEqual(vocab.count, 1024)
            XCTAssertEqual(c.rnntBlankId, vocab.count,
                           "blank id is one past the last real token")
        }

        // ── tool calling is a first-class channel ──────────────────────────
        XCTAssertEqual(c.functionChannelWeight, 2.0, accuracy: 1e-6,
                       "tool calls are a WEIGHTED parallel channel with their own head")

        // ── tts / codec agreement ──────────────────────────────────────────
        XCTAssertEqual(c.ttsConfig.numQuantizers, c.codecConfig.numQuantizers,
                       "tts and codec must agree on RVQ depth")
        XCTAssertEqual(c.ttsConfig.codebookSize, c.codecConfig.codebookSize,
                       "tts and codec must agree on codebook size")
        XCTAssertEqual(c.ttsConfig.latentSize, c.codecConfig.latentDim,
                       "tts latent must match codec latent")

        XCTAssertEqual(c.speaker, "Aria")
    }

    /// The encoder is configured for STREAMING. If a future bundle drops these,
    /// the offline Parakeet path would silently be "correct" while being wrong
    /// on a live mic — the exact class of bug a non-streaming test cannot see.
    func testEncoderIsStreamingConfigured() throws {
        guard let c = try loadRealConfig() else {
            throw XCTSkip("VoiceChat MLX bundle not present")
        }
        let e = c.audioConfig.encoder
        XCTAssertEqual(e.dModel, 1024)
        XCTAssertEqual(e.nLayers, 24)
        XCTAssertEqual(e.nHeads, 8)
        XCTAssertEqual(e.selfAttentionModel, "rel_pos")
        XCTAssertEqual(e.subsamplingFactor, 8)

        XCTAssertTrue(e.isStreaming,
                      "VoiceChat's encoder must be streaming-configured")
        XCTAssertEqual(e.convNormType, "layer_norm",
                       "Parakeet.swift uses BatchNorm1d — this needs a LayerNorm variant")
        XCTAssertEqual(e.attContextStyle, "chunked_limited",
                       "full attention would break causal streaming")
        XCTAssertEqual(e.causalDownsampling, true)
        XCTAssertEqual(e.convContextSize, "causal")

        // att_context_size is [[70, 0]] — a list of [left, right] PAIRS.
        XCTAssertEqual(e.leftContextFrames, 70)
        XCTAssertEqual(e.rightContextFrames, 0)
        XCTAssertTrue(e.isFullyCausal,
                      "right context must be 0 — any lookahead adds algorithmic "
                      + "latency to every duplex response")
    }

    /// Guards the two facts a runtime is most likely to get wrong silently.
    func testDuplexChannelInvariants() throws {
        guard let c = try loadRealConfig() else {
            throw XCTSkip("VoiceChat MLX bundle not present")
        }
        // Silence is an emitted token: a duplex model is always producing
        // something on the audio timeline, including deliberate silence.
        XCTAssertEqual(c.silenceTokenId, 11)
        XCTAssertNotEqual(c.silenceTokenId, c.padTokenId,
                          "silence is emitted output, pad is absence — not the same")
        // Encoder frame rate must match the duplex timeline frame duration.
        XCTAssertEqual(c.framesPerSecond, 12.5, accuracy: 1e-9)
    }
}
