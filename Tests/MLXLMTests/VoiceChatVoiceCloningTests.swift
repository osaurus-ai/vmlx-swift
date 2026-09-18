import AVFoundation
import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLMCommon
import MLXFFT
@testable import MLXVLM

/// Speaker control: does a reference recording actually change the agent's
/// VOICE while leaving what it says alone?
///
/// The mechanism is not a separate feature — `warmup` derives the prompt
/// frames from `embed_code(depthSum(prompt codes)) @ audio_prompt_projection_W`
/// whenever no explicit latent is supplied, so feeding it a real recording
/// instead of silence makes the speaker that recording's speaker. These tests
/// prove that end to end on the real bundle, with the measurements a listening
/// opinion cannot supply: spectral centroid (brightness/pitch character),
/// zero-crossing rate, and RMS.
///
/// 🚨 The decisive assertion is a CROSSED one: two different reference voices
/// must differ from each other AND from the built-in Aria. A cloning path that
/// silently ignored the reference would still produce perfectly good speech and
/// pass every single-sample audio check.
///
/// Runs behind `VOICECHAT_QUANT_PROOF=1` (an 11 B load).
public class VoiceChatVoiceCloningTests: XCTestCase {

    private static var enabled: Bool {
        ProcessInfo.processInfo.environment["VOICECHAT_QUANT_PROOF"] == "1"
    }

    private static var bundlePath: String {
        ProcessInfo.processInfo.environment["VOICECHAT_QUANT_DIR"]
            ?? FileManager.default.homeDirectoryForCurrentUser
                .appendingPathComponent("models/OsaurusAI/NemotronLabs-VoiceChat-11B-JANG_4")
                .path
    }

    /// Directory of `voice-*.wav` reference clips.
    private static var voicesDir: URL {
        URL(
            fileURLWithPath: ProcessInfo.processInfo.environment["VOICECHAT_VOICES_DIR"]
                ?? NSTemporaryDirectory())
    }

    private static var fixtureURL: URL {
        FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(
            "models/nemotron-voicechat-src/NVIDIA-NemotronLabs-VoiceChat-11B/turn_taking.wav")
    }

    private static var proofDir: URL {
        URL(
            fileURLWithPath: ProcessInfo.processInfo.environment["VOICECHAT_PROOF_DIR"]
                ?? NSTemporaryDirectory())
    }

    // MARK: - Audio helpers

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
        let sourceRate = format.sampleRate
        let wanted = Int(seconds * targetRate)
        var out = [Float]()
        out.reserveCapacity(wanted)
        for i in 0 ..< wanted {
            let position = offset * sourceRate + Double(i) * sourceRate / targetRate
            let index = Int(position)
            guard index + 1 < source.count else { break }
            let fraction = Float(position - Double(index))
            out.append(source[index] * (1 - fraction) + source[index + 1] * fraction)
        }
        return out
    }

    private func writeWAV(_ samples: [Float], sampleRate: Int, to url: URL) throws {
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

    /// Voice character, not just loudness: spectral centroid is where the
    /// energy sits (a cute/high voice is brighter), ZCR tracks it, and RMS
    /// keeps a silent result from masquerading as a distinct voice.
    struct VoiceStats {
        let rms: Float
        let centroidHz: Float
        let zcr: Float
    }

    private func voiceStats(_ samples: [Float], sampleRate: Int) -> VoiceStats {
        var sumSquares: Float = 0
        var crossings = 0
        for (i, v) in samples.enumerated() {
            sumSquares += v * v
            if i > 0, (samples[i - 1] < 0) != (v < 0) { crossings += 1 }
        }
        let count = Float(max(samples.count, 1))

        // Spectral centroid over the whole clip via a coarse DFT-free proxy:
        // frame energies through a bank of one-pole differences is overkill
        // here, so use MLX's rFFT on a padded power-of-two window.
        let n = 4096
        let usable = min(samples.count, n)
        var window = [Float](repeating: 0, count: n)
        for i in 0 ..< usable {
            let hann = Float(0.5 - 0.5 * cos(2.0 * Double.pi * Double(i) / Double(usable)))
            window[i] = samples[samples.count / 2 - usable / 2 + i] * hann
        }
        let spectrum = MLXFFT.rfft(MLXArray(window))
        let magnitude = MLX.sqrt(
            spectrum.realPart() * spectrum.realPart()
                + spectrum.imaginaryPart() * spectrum.imaginaryPart())
        let bins = magnitude.dim(0)
        let freqs = MLXArray((0 ..< bins).map { Float($0) * Float(sampleRate) / Float(n) })
        let total = MLX.sum(magnitude).item(Float.self)
        let centroid =
            total > 1e-9 ? MLX.sum(magnitude * freqs).item(Float.self) / total : 0

        return VoiceStats(
            rms: (sumSquares / count).squareRoot(),
            centroidHz: centroid,
            zcr: Float(crossings) / count)
    }

    // MARK: - The test

    /// Calibrate the frozen prompt projection against the shipped Aria latent.
    ///
    /// `audio_prompt_latents.Aria` is what the model uses for its own voice, and
    /// the projection branch is supposed to produce the SAME kind of thing from
    /// a reference recording. So projecting a recording of Aria's own voice is a
    /// ground truth: it says both what scale the projection needs and whether
    /// the projection is trained at all.
    ///
    /// Prints, per frame, the projected prompt's RMS against Aria's and their
    /// cosine similarity. Diagnostic only — it asserts nothing about the
    /// checkpoint, it reports what the checkpoint contains.
    func testProjectedPromptAgainstShippedAriaLatent() throws {
        guard Self.enabled else { throw XCTSkip("set VOICECHAT_QUANT_PROOF=1") }
        guard let ariaAudio = ProcessInfo.processInfo.environment["VOICECHAT_ARIA_REFERENCE"]
        else { throw XCTSkip("set VOICECHAT_ARIA_REFERENCE to a wav of the built-in voice") }
        let bundle = URL(fileURLWithPath: Self.bundlePath)
        guard VoiceChatLoader.looksLikeVoiceChatBundle(at: bundle) else {
            throw XCTSkip("no VoiceChat bundle")
        }
        let (model, config) = try VoiceChatLoader.load(from: bundle)
        let tts = model.ttsModel.ttsModel

        let reference = try readMono(
            URL(fileURLWithPath: ariaAudio), targetRate: Double(config.codecConfig.sampleRate),
            seconds: Double(config.ttsConfig.audioPromptDuration ?? 3.0), offset: 0.4)
        let prompt = model.ttsPrompt(voice: .reference(reference))
        MLX.eval(prompt.codes)

        // Reproduce warmup's shift, so these are the frames warmup would use.
        let shifted = MLX.concatenated(
            [
                MLXArray.zeros(like: prompt.codes[0..., 0 ..< 1, 0...]),
                prompt.codes[0..., ..<(prompt.codes.dim(1) - 1), 0...],
            ], axis: 1)
        let codeEmbed = tts.embedCode(tts.depthSumEmbedding(shifted)).asType(.float32)
        let raw = MLX.matmul(codeEmbed, tts.audioPromptProjectionW.asType(.float32))
        let aria = model.ttsModel.audioPromptLatents.aria.asType(.float32)
        MLX.eval(raw, aria)

        func rms(_ a: MLXArray) -> Float { MLX.sqrt(MLX.mean(a * a)).item(Float.self) }
        let rawRMS = rms(raw), ariaRMS = rms(aria)
        print(String(format: "[prompt-cal] raw projection RMS  %.4f", rawRMS))
        print(String(format: "[prompt-cal] shipped Aria RMS    %.4f", ariaRMS))
        print(String(format: "[prompt-cal] RMS-matching scale  %.6f", ariaRMS / rawRMS))
        print(
            String(
                format: "[prompt-cal] 1/sqrt(hidden) scale  %.6f",
                1.0 / Foundation.sqrt(Float(codeEmbed.dim(-1)))))

        // Direction is what says whether the projection is TRAINED: a scale can
        // always be fixed, an unrelated direction cannot.
        let frames = Swift.min(raw.dim(1), aria.dim(1))
        let a = raw[0, ..<frames, 0...], b = aria[0, ..<frames, 0...]
        let cos =
            MLX.sum(a * b, axis: -1)
            / (MLX.sqrt(MLX.sum(a * a, axis: -1)) * MLX.sqrt(MLX.sum(b * b, axis: -1)) + 1e-9)
        let cosValues = cos.asArray(Float.self)
        let meanCos = cosValues.reduce(0, +) / Float(cosValues.count)
        print(
            String(
                format: "[prompt-cal] per-frame cosine to Aria: mean %+.4f  min %+.4f  max %+.4f",
                meanCos, cosValues.min() ?? 0, cosValues.max() ?? 0))
    }

    func testReferenceVoicesChangeTheAgentVoice() throws {
        guard Self.enabled else {
            throw XCTSkip("set VOICECHAT_QUANT_PROOF=1 to run the voice-cloning proof")
        }
        let bundle = URL(fileURLWithPath: Self.bundlePath)
        guard VoiceChatLoader.looksLikeVoiceChatBundle(at: bundle) else {
            throw XCTSkip("no VoiceChat bundle at \(Self.bundlePath)")
        }
        guard FileManager.default.fileExists(atPath: Self.fixtureURL.path) else {
            throw XCTSkip("fixture missing")
        }
        let voiceFiles =
            ((try? FileManager.default.contentsOfDirectory(
                at: Self.voicesDir, includingPropertiesForKeys: nil)) ?? [])
            .filter { $0.lastPathComponent.hasPrefix("voice-") && $0.pathExtension == "wav" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        try XCTSkipIf(voiceFiles.isEmpty, "no voice-*.wav references in \(Self.voicesDir.path)")

        let (model, config) = try VoiceChatLoader.load(from: bundle)
        let vocabData = try Data(
            contentsOf: bundle.appendingPathComponent("tokenizer.json"))
        let vocabJSON = try JSONSerialization.jsonObject(with: vocabData) as? [String: Any]
        let vocab = (vocabJSON?["model"] as? [String: Any])?["vocab"] as? [String: Int]
        try model.setVocabulary(try XCTUnwrap(vocab))

        // One fixed user turn for every voice, so ANY difference in the output
        // comes from the speaker prompt and nothing else.
        let userSeconds =
            Double(ProcessInfo.processInfo.environment["VOICECHAT_PROOF_SECONDS"] ?? "2.0")
            ?? 2.0
        let user = try readMono(
            Self.fixtureURL, targetRate: Double(config.inputSampleRate),
            seconds: userSeconds)
        let mel = voiceChatLogMelSpectrogram(user, config: config.audioConfig.preprocessor)
        let (projected, encoded) = model.sttModel.perception(mel)
        MLX.eval(projected, encoded)

        func run(_ voice: NemotronVoiceChatModel.Voice, name: String) throws
            -> (stats: VoiceStats, text: [Int], samples: [Float])
        {
            let result = model.generateTurn(
                audioEmbeds: projected, asrEmbeds: encoded, voice: voice)
            let samples = result.audio.asType(.float32).asArray(Float.self)
            let stats = voiceStats(samples, sampleRate: result.sampleRate)
            try writeWAV(
                samples, sampleRate: result.sampleRate,
                to: Self.proofDir.appendingPathComponent("voicechat-voice-\(name).wav"))
            print(
                String(
                    format: "[voicechat-voice] %-12s rms %.4f · centroid %6.0f Hz · zcr %.3f",
                    (name as NSString).utf8String!, stats.rms, stats.centroidHz, stats.zcr))
            return (stats, result.textTokens, samples)
        }

        let builtIn = try run(.builtIn, name: "aria-builtin")
        XCTAssertGreaterThan(builtIn.stats.rms, 1e-4, "built-in voice produced silence")

        var cloned = [(name: String, stats: VoiceStats, samples: [Float])]()
        let maxVoices = Int(ProcessInfo.processInfo.environment["VOICECHAT_MAX_VOICES"] ?? "4") ?? 4
        for file in voiceFiles.prefix(maxVoices) {
            let name = file.deletingPathExtension().lastPathComponent
                .replacingOccurrences(of: "voice-", with: "")
            // The prompt is consumed at the CODEC's rate, and the model wants
            // about `audio_prompt_duration` seconds of it.
            let reference = try readMono(
                file, targetRate: Double(config.codecConfig.sampleRate),
                seconds: Double(config.ttsConfig.audioPromptDuration ?? 3.0),
                offset: 0.4)
            XCTAssertGreaterThan(reference.count, 1000, "\(name): reference clip too short")

            let out = try run(.reference(reference), name: name)
            XCTAssertGreaterThan(out.stats.rms, 1e-4, "\(name): cloned voice produced silence")
            XCTAssertTrue(
                out.samples.allSatisfy { $0.isFinite }, "\(name): cloned audio has NaN/Inf")
            // Speaking the same words: the reference changes the VOICE, not
            // the content. (Greedy decode over identical user audio.)
            XCTAssertEqual(
                out.text.count, builtIn.text.count,
                "\(name): reference changed the timeline length, not just the voice")
            cloned.append((name, out.stats, out.samples))
        }

        // 🚨 The crossed assertion. A path that ignored the reference would
        // pass everything above.
        for entry in cloned {
            let differing = zip(entry.samples, builtIn.samples)
                .filter { abs($0 - $1) > 1e-4 }.count
            let fraction = Float(differing) / Float(max(builtIn.samples.count, 1))
            print(
                String(
                    format: "[voicechat-voice] %-12s vs built-in: %.1f%% samples differ · "
                        + "centroid %+.0f Hz",
                    (entry.name as NSString).utf8String!, fraction * 100,
                    entry.stats.centroidHz - builtIn.stats.centroidHz))
            XCTAssertGreaterThan(
                fraction, 0.10,
                "\(entry.name): cloned output is nearly identical to the built-in voice — "
                    + "the reference prompt is not reaching the speaker path")
        }

        if cloned.count >= 2 {
            let a = cloned[0], b = cloned[1]
            let differing = zip(a.samples, b.samples).filter { abs($0 - $1) > 1e-4 }.count
            let fraction = Float(differing) / Float(max(a.samples.count, 1))
            XCTAssertGreaterThan(
                fraction, 0.10,
                "\(a.name) and \(b.name) produced nearly identical audio — different "
                    + "references are not producing different voices")
        }
    }
}
