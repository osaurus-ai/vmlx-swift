import AVFoundation
import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

/// Audio-to-audio proof on the SHIPPED quants: load the real bundle, feed the
/// NVIDIA fixture's user channel as 16 kHz mono, run the duplex turn loop, and
/// measure the rendered 22.05 kHz speech.
///
/// Per the live-test protocol, "a wav was produced" is NOT evidence — silence
/// and noise both produce wavs. Each leg asserts RMS, dynamic range, and
/// zero-crossing rate in a speech-plausible band, and writes the WAV to
/// `VOICECHAT_PROOF_DIR` so it can be listened to.
///
/// Runs only when `VOICECHAT_QUANT_PROOF=1` — an 11 B load per leg is far too
/// heavy for the default suite.
public class VoiceChatQuantAudioProofTests: XCTestCase {

    private static var enabled: Bool {
        ProcessInfo.processInfo.environment["VOICECHAT_QUANT_PROOF"] == "1"
    }

    private static var quantPaths: [(name: String, path: String)] {
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        if let override = ProcessInfo.processInfo.environment["VOICECHAT_QUANT_DIR"] {
            return [(URL(fileURLWithPath: override).lastPathComponent, override)]
        }
        return [
            ("MXFP8", "\(home)/models/JANGQ-AI/NemotronLabs-VoiceChat-11B-MXFP8"),
            ("JANG_4", "\(home)/models/OsaurusAI/NemotronLabs-VoiceChat-11B-JANG_4"),
            ("JANG_2", "\(home)/models/OsaurusAI/NemotronLabs-VoiceChat-11B-JANG_2"),
        ]
    }

    private static var fixtureURL: URL {
        let home = FileManager.default.homeDirectoryForCurrentUser
        if let override = ProcessInfo.processInfo.environment["VOICECHAT_FIXTURE"] {
            return URL(fileURLWithPath: override)
        }
        return home.appendingPathComponent(
            "models/nemotron-voicechat-src/NVIDIA-NemotronLabs-VoiceChat-11B/turn_taking.wav")
    }

    private static var proofDir: URL {
        URL(
            fileURLWithPath: ProcessInfo.processInfo.environment["VOICECHAT_PROOF_DIR"]
                ?? NSTemporaryDirectory())
    }

    /// Left channel (the user) of the stereo fixture, resampled to 16 kHz mono.
    private func loadUserChannel(seconds: Double, offset: Double = 0) throws -> [Float] {
        let file = try AVAudioFile(forReading: Self.fixtureURL)
        let format = file.processingFormat
        guard
            let buffer = AVAudioPCMBuffer(
                pcmFormat: format, frameCapacity: AVAudioFrameCount(file.length))
        else { throw XCTSkip("could not allocate fixture buffer") }
        try file.read(into: buffer)

        guard let channels = buffer.floatChannelData else {
            throw XCTSkip("fixture has no float samples")
        }
        let frames = Int(buffer.frameLength)
        let left = Array(UnsafeBufferPointer(start: channels[0], count: frames))

        // The fixtures are 24 kHz; the model listens at 16 kHz. Linear resample
        // — the encoder is robust to it and this keeps the proof dependency-free.
        let sourceRate = format.sampleRate
        let targetRate = 16000.0
        let wanted = Int(seconds * targetRate)
        let startSample = offset * sourceRate
        var out = [Float]()
        out.reserveCapacity(wanted)
        for i in 0 ..< wanted {
            let position = startSample + Double(i) * sourceRate / targetRate
            let index = Int(position)
            guard index + 1 < left.count else { break }
            let frac = Float(position - Double(index))
            out.append(left[index] * (1 - frac) + left[index + 1] * frac)
        }
        return out
    }

    private func writeWAV(_ samples: [Float], sampleRate: Int, to url: URL) throws {
        let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32, sampleRate: Double(sampleRate),
            channels: 1, interleaved: false)!
        let file = try AVAudioFile(forWriting: url, settings: format.settings)
        guard
            let buffer = AVAudioPCMBuffer(
                pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count))
        else { return }
        buffer.frameLength = AVAudioFrameCount(samples.count)
        for (i, value) in samples.enumerated() {
            buffer.floatChannelData![0][i] = value
        }
        try file.write(from: buffer)
    }

    /// Speech-plausibility measurements — the protocol's §"for audio, a wav is
    /// not evidence" rule.
    private struct AudioStats {
        let rms: Float
        let peak: Float
        let zeroCrossingRate: Float
        let dynamicRange: Float
    }

    private func measure(_ samples: [Float]) -> AudioStats {
        var sumSquares: Float = 0
        var peak: Float = 0
        var crossings = 0
        var minimum = Float.greatestFiniteMagnitude
        var maximum = -Float.greatestFiniteMagnitude
        for (i, value) in samples.enumerated() {
            sumSquares += value * value
            peak = Swift.max(peak, abs(value))
            minimum = Swift.min(minimum, value)
            maximum = Swift.max(maximum, value)
            if i > 0, (samples[i - 1] < 0) != (value < 0) { crossings += 1 }
        }
        let count = Float(Swift.max(samples.count, 1))
        return AudioStats(
            rms: (sumSquares / count).squareRoot(),
            peak: peak,
            zeroCrossingRate: Float(crossings) / count,
            dynamicRange: maximum - minimum)
    }

    func testEveryShippedQuantRendersSpeechFromRealAudio() throws {
        guard Self.enabled else {
            throw XCTSkip("set VOICECHAT_QUANT_PROOF=1 to run the per-quant audio proof")
        }
        guard FileManager.default.fileExists(atPath: Self.fixtureURL.path) else {
            throw XCTSkip("fixture missing at \(Self.fixtureURL.path)")
        }

        let seconds = Double(
            ProcessInfo.processInfo.environment["VOICECHAT_PROOF_SECONDS"] ?? "3.0") ?? 3.0
        let user = try loadUserChannel(seconds: seconds)
        XCTAssertGreaterThan(user.count, 1000, "fixture produced too few input samples")

        var summaries = [String]()
        for (name, path) in Self.quantPaths {
            guard VoiceChatLoader.looksLikeVoiceChatBundle(at: URL(fileURLWithPath: path))
            else {
                XCTFail("\(name): no VoiceChat bundle at \(path)")
                continue
            }

            let loadStart = Date()
            let (model, config) = try VoiceChatLoader.load(from: URL(fileURLWithPath: path))
            let loadSeconds = Date().timeIntervalSince(loadStart)

            // Tokenizer vocabulary for the character-aware TTS conditioning.
            let vocabURL = URL(fileURLWithPath: path).appendingPathComponent("tokenizer.json")
            if let data = try? Data(contentsOf: vocabURL),
                let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                let modelBlock = json["model"] as? [String: Any],
                let vocab = modelBlock["vocab"] as? [String: Int]
            {
                try model.setVocabulary(vocab)
            } else {
                XCTFail("\(name): could not read tokenizer vocabulary for TTS conditioning")
                continue
            }

            let mel = voiceChatLogMelSpectrogram(user, config: config.audioConfig.preprocessor)
            let (projected, encoded) = model.sttModel.perception(mel)
            MLX.eval(projected, encoded)

            let turnStart = Date()
            let result = model.generateTurn(audioEmbeds: projected, asrEmbeds: encoded)
            let turnSeconds = Date().timeIntervalSince(turnStart)

            let samples = result.audio.asType(.float32).asArray(Float.self)
            let stats = measure(samples)
            let outURL = Self.proofDir.appendingPathComponent(
                "voicechat-\(name.lowercased()).wav")
            try writeWAV(samples, sampleRate: result.sampleRate, to: outURL)

            let transcript = model.transcribeUser(result)
            let toolCalls = result.functionTokens.filter { $0 != config.padTokenId }.count

            summaries.append(
                String(
                    format:
                        "%@: load %.1fs · turn %.1fs · %d frames · %d samples @%d Hz · "
                        + "rms %.4f peak %.3f zcr %.4f range %.3f · asr %d tok · fn %d · %@",
                    name, loadSeconds, turnSeconds, result.audioFrames, samples.count,
                    result.sampleRate, stats.rms, stats.peak, stats.zeroCrossingRate,
                    stats.dynamicRange, transcript.count, toolCalls, outURL.lastPathComponent))

            // A wav is not evidence; these are.
            XCTAssertGreaterThan(
                samples.count, result.audioFrames * 100,
                "\(name): rendered far fewer samples than frames imply")
            XCTAssertTrue(
                samples.allSatisfy { $0.isFinite }, "\(name): rendered audio contains NaN/Inf")
            XCTAssertGreaterThan(stats.rms, 1e-4, "\(name): output is silence")
            XCTAssertLessThan(stats.peak, 10.0, "\(name): output is clipping/exploding")
            XCTAssertGreaterThan(
                stats.dynamicRange, 1e-3, "\(name): output is constant, not a waveform")
            // Speech at 22.05 kHz sits well under a 0.5 crossing rate; white
            // noise approaches it.
            XCTAssertLessThan(
                stats.zeroCrossingRate, 0.45,
                "\(name): zero-crossing rate is noise-like, not speech-like")
        }

        for line in summaries { print("[voicechat-proof] \(line)") }
        XCTAssertEqual(
            summaries.count, Self.quantPaths.count, "not every quant produced a proof leg")
    }

    /// 🚨 The check that separates "renders audio" from "renders THIS audio".
    ///
    /// A model that ignored its input would still pass every measurement in
    /// the test above — RMS, dynamic range, zero-crossing rate all stay
    /// speech-like for a fixed canned response. This is the word-list lesson
    /// from the VLM work: score OPPOSITE inputs and require them to differ.
    /// Two different windows of the fixture must produce materially different
    /// speech, transcripts, or both.
    func testOutputDependsOnTheInputAudio() throws {
        guard Self.enabled else {
            throw XCTSkip("set VOICECHAT_QUANT_PROOF=1 to run the per-quant audio proof")
        }
        guard FileManager.default.fileExists(atPath: Self.fixtureURL.path) else {
            throw XCTSkip("fixture missing at \(Self.fixtureURL.path)")
        }
        guard let (name, path) = Self.quantPaths.first(where: {
            VoiceChatLoader.looksLikeVoiceChatBundle(at: URL(fileURLWithPath: $0.path))
        }) else { throw XCTSkip("no VoiceChat bundle present") }

        let (model, config) = try VoiceChatLoader.load(from: URL(fileURLWithPath: path))
        let vocabURL = URL(fileURLWithPath: path).appendingPathComponent("tokenizer.json")
        let data = try Data(contentsOf: vocabURL)
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        let vocab = (json?["model"] as? [String: Any])?["vocab"] as? [String: Int]
        try model.setVocabulary(try XCTUnwrap(vocab))

        func turn(offsetSeconds: Double) throws -> (audio: [Float], text: [Int], asr: [Int]) {
            let samples = try loadUserChannel(seconds: 2.0, offset: offsetSeconds)
            let mel = voiceChatLogMelSpectrogram(
                samples, config: config.audioConfig.preprocessor)
            let (projected, encoded) = model.sttModel.perception(mel)
            let result = model.generateTurn(audioEmbeds: projected, asrEmbeds: encoded)
            return (
                result.audio.asType(.float32).asArray(Float.self),
                result.textTokens,
                model.transcribeUser(result)
            )
        }

        // Two windows several seconds apart in the same conversation.
        let early = try turn(offsetSeconds: 1.0)
        let late = try turn(offsetSeconds: 18.0)

        XCTAssertEqual(early.audio.count, late.audio.count, "same duration in, same out")
        let differing = zip(early.audio, late.audio).filter { abs($0 - $1) > 1e-4 }.count
        let fraction = Float(differing) / Float(Swift.max(early.audio.count, 1))
        print(
            String(
                format:
                    "[voicechat-proof] %@ input-dependence: %.1f%% of samples differ · "
                    + "text %d vs %d tok · asr %d vs %d tok",
                name, fraction * 100, early.text.count, late.text.count,
                early.asr.count, late.asr.count))

        XCTAssertGreaterThan(
            fraction, 0.10,
            "different input audio produced near-identical output — the model is not listening")
    }
}
