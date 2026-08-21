import AVFoundation
import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLMCommon
@testable import MLXVLM

/// **The harness that decides whether the voice actually says words.**
///
/// RMS, dynamic range, zero-crossing rate and spectral centroid all pass for
/// structured noise — the first batch of "voice samples" scored well on every
/// one of them and was babble. Listening is not available to an automated run,
/// so the only honest test is a CLOSED LOOP:
///
///   the model says something in its TEXT channel
///     → the speech decoder renders it to audio
///       → that audio goes back through the model's OWN RNN-T ASR
///         → the transcript must contain what the text channel said.
///
/// A vocoder emitting babble transcribes to nothing (or to unrelated tokens),
/// which no energy statistic can detect. The same loop doubles as the
/// regression gate for every future change to the speech path.
public class VoiceChatIntelligibilityTests: XCTestCase {

    private static var enabled: Bool {
        ProcessInfo.processInfo.environment["VOICECHAT_QUANT_PROOF"] == "1"
    }

    private static var bundlePath: String {
        ProcessInfo.processInfo.environment["VOICECHAT_QUANT_DIR"]
            ?? FileManager.default.homeDirectoryForCurrentUser
                .appendingPathComponent("models/OsaurusAI/NemotronLabs-VoiceChat-11B-JANG_4")
                .path
    }

    private static var fixtureURL: URL {
        if let override = ProcessInfo.processInfo.environment["VOICECHAT_FIXTURE"] {
            return URL(fileURLWithPath: override)
        }
        return FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(
            "models/nemotron-voicechat-src/NVIDIA-NemotronLabs-VoiceChat-11B/turn_taking.wav")
    }

    /// Optional: transcribe an EXTERNAL wav (e.g. the Python reference's
    /// output) with the same ASR, so "my speech is babble" can be separated
    /// from "my yardstick is broken".
    private static var externalAudio: String? {
        ProcessInfo.processInfo.environment["VOICECHAT_EXTERNAL_AUDIO"]
    }

    private static var proofDir: URL {
        URL(
            fileURLWithPath: ProcessInfo.processInfo.environment["VOICECHAT_PROOF_DIR"]
                ?? NSTemporaryDirectory())
    }

    private func readMono(_ url: URL, targetRate: Double, seconds: Double, offset: Double = 0)
        throws -> [Float]
    {
        let file = try AVAudioFile(forReading: url)
        let format = file.processingFormat
        guard
            let buffer = AVAudioPCMBuffer(
                pcmFormat: format, frameCapacity: AVAudioFrameCount(file.length))
        else { return [] }
        try file.read(into: buffer)
        guard let channels = buffer.floatChannelData else { return [] }
        let source = Array(
            UnsafeBufferPointer(start: channels[0], count: Int(buffer.frameLength)))
        var out = [Float]()
        for i in 0 ..< Int(seconds * targetRate) {
            let position = offset * format.sampleRate
                + Double(i) * format.sampleRate / targetRate
            let index = Int(position)
            guard index + 1 < source.count else { break }
            let fraction = Float(position - Double(index))
            out.append(source[index] * (1 - fraction) + source[index + 1] * fraction)
        }
        return out
    }

    private func resample(_ samples: [Float], from: Int, to: Int) -> [Float] {
        guard from != to, !samples.isEmpty else { return samples }
        let ratio = Double(from) / Double(to)
        let count = Int(Double(samples.count) / ratio)
        var out = [Float]()
        out.reserveCapacity(count)
        for i in 0 ..< count {
            let position = Double(i) * ratio
            let index = Int(position)
            guard index + 1 < samples.count else { break }
            let fraction = Float(position - Double(index))
            out.append(samples[index] * (1 - fraction) + samples[index + 1] * fraction)
        }
        return out
    }

    private func writeWAV(_ samples: [Float], sampleRate: Int, to url: URL) throws {
        try? FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32, sampleRate: Double(sampleRate), channels: 1,
            interleaved: false)!
        let file = try AVAudioFile(forWriting: url, settings: format.settings)
        guard
            let buffer = AVAudioPCMBuffer(
                pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count))
        else { return }
        buffer.frameLength = AVAudioFrameCount(samples.count)
        for (i, v) in samples.enumerated() { buffer.floatChannelData![0][i] = v }
        try file.write(from: buffer)
    }

    /// Transcribe arbitrary audio with the model's own RNN-T branch.
    /// Returns the decoded token pieces joined into text.
    static func transcribe(
        _ samples: [Float], model: NemotronVoiceChatModel,
        config: NemotronVoiceChatConfiguration
    ) -> (text: String, tokenCount: Int) {
        guard samples.count > 800 else { return ("", 0) }
        let mel = voiceChatLogMelSpectrogram(samples, config: config.audioConfig.preprocessor)
        let (_, encoded) = model.sttModel.perception(mel)
        var state = VoiceChatRNNTDecodeState()
        let tokens = model.sttModel.transcribe(encoded: encoded, state: &state)
        let vocabulary = config.rnntVocabulary ?? []
        let pieces = tokens.compactMap { id -> String? in
            guard id >= 0, id < vocabulary.count else { return nil }
            return vocabulary[id]
        }
        let text = pieces.joined()
            .replacingOccurrences(of: "\u{2581}", with: " ")
            .trimmingCharacters(in: .whitespaces)
        return (text, tokens.count)
    }

    /// Fraction of `expected`'s letters that appear, in order, in `actual`.
    /// Robust to ASR slips in a way exact equality is not.
    static func inOrderLetterOverlap(expected: String, actual: String) -> Double {
        let a = Array(expected.lowercased().filter { $0.isLetter })
        let b = Array(actual.lowercased().filter { $0.isLetter })
        guard !a.isEmpty else { return 0 }
        var i = 0
        for ch in b where i < a.count {
            if ch == a[i] { i += 1 }
        }
        return Double(i) / Double(a.count)
    }

    func testGeneratedSpeechTranscribesBackToWhatTheAgentSaid() throws {
        guard Self.enabled else { throw XCTSkip("set VOICECHAT_QUANT_PROOF=1") }
        let bundle = URL(fileURLWithPath: Self.bundlePath)
        guard VoiceChatLoader.looksLikeVoiceChatBundle(at: bundle) else {
            throw XCTSkip("no VoiceChat bundle")
        }
        let (model, config) = try VoiceChatLoader.load(from: bundle)

        let data = try Data(contentsOf: bundle.appendingPathComponent("tokenizer.json"))
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        let vocab = try XCTUnwrap((json?["model"] as? [String: Any])?["vocab"] as? [String: Int])
        var idToToken = [Int: String]()
        for (token, id) in vocab { idToToken[id] = token }
        try model.setVocabulary(vocab)

        // Sanity floor: the ASR branch must read the FIXTURE's own speech.
        // Without this, a transcript of "" for the generated audio could mean
        // either broken speech or a broken ASR, and the test would not say
        // which.
        let fixtureUser = try readMono(
            Self.fixtureURL, targetRate: Double(config.inputSampleRate), seconds: 6.0)
        let control = Self.transcribe(fixtureUser, model: model, config: config)
        print("[intelligibility] ASR control on the fixture: \"\(control.text.prefix(120))\"")
        XCTAssertGreaterThan(
            control.text.filter { $0.isLetter }.count, 8,
            "the ASR branch cannot even transcribe real recorded speech — fix that before "
                + "judging generated audio")

        if let external = Self.externalAudio {
            let ext = try readMono(
                URL(fileURLWithPath: external), targetRate: Double(config.inputSampleRate),
                seconds: 12.0)
            let heardExternal = Self.transcribe(ext, model: model, config: config)
            print(
                "[intelligibility] ASR of EXTERNAL audio (\(external)): "
                    + "\"\(heardExternal.text.prefix(160))\" (\(heardExternal.tokenCount) tokens)")
        }

        // The turn under test.
        let user = try readMono(
            Self.fixtureURL, targetRate: Double(config.inputSampleRate), seconds: 6.0)
        let mel = voiceChatLogMelSpectrogram(user, config: config.audioConfig.preprocessor)
        let (projected, encoded) = model.sttModel.perception(mel)
        let result = model.generateTurn(audioEmbeds: projected, asrEmbeds: encoded)

        let special: Set<Int> = [
            config.padTokenId, config.silenceTokenId, config.bosTokenId, config.eosTokenId,
        ]
        let saidText = result.textTokens.filter { !special.contains($0) }
            .compactMap { idToToken[$0] }
            .joined()
            .replacingOccurrences(of: "\u{2581}", with: " ")
            .replacingOccurrences(of: "Ġ", with: " ")
            .trimmingCharacters(in: .whitespaces)

        let generated = result.audio.asType(.float32).asArray(Float.self)
        try writeWAV(
            generated, sampleRate: result.sampleRate,
            to: Self.proofDir.appendingPathComponent("intelligibility-generated.wav"))

        // Back through the model's own ears, at its input rate.
        let heard = Self.transcribe(
            resample(generated, from: result.sampleRate, to: config.inputSampleRate),
            model: model, config: config)
        let overlap = Self.inOrderLetterOverlap(expected: saidText, actual: heard.text)

        print("[intelligibility] agent text channel : \"\(saidText.prefix(120))\"")
        print(
            "[intelligibility] ASR of its own speech: \"\(heard.text.prefix(120))\" "
                + "(\(heard.tokenCount) tokens)")
        print(String(format: "[intelligibility] in-order letter overlap: %.0f%%", overlap * 100))

        XCTAssertFalse(
            saidText.isEmpty, "the agent said nothing this turn — nothing to render or check")
        XCTAssertGreaterThan(
            heard.tokenCount, 0,
            "the generated audio transcribes to NOTHING — it is not speech, whatever its RMS "
                + "and zero-crossing rate say")
        XCTAssertGreaterThan(
            overlap, 0.5,
            "the generated speech does not say what the agent said "
                + "(text: \"\(saidText.prefix(60))\" vs heard: \"\(heard.text.prefix(60))\")")
    }
}
