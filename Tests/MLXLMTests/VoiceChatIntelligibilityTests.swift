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

    /// 🚨 Start 14 s into the fixture, not at 0.
    ///
    /// The first six seconds of `turn_taking.wav` contain one word — the ASR
    /// control transcribes them as "Hi" — so a turn built from offset 0 gives
    /// the agent almost nothing to answer and it replies with a bare greeting.
    /// That made the gate measure a degenerate turn: it read 21% overlap on a
    /// perfectly healthy runtime simply because there was nothing to say. At
    /// 14 s the user asks a real question and the agent answers in a full
    /// sentence.
    private static var fixtureOffset: Double {
        Double(ProcessInfo.processInfo.environment["VOICECHAT_FIXTURE_OFFSET"] ?? "") ?? 14.0
    }

    /// How much user audio the turn gets. This also bounds how long the agent
    /// can SPEAK, because the duplex loop runs one frame of speech per frame of
    /// user audio — so too short a window truncates the reply mid-sentence and
    /// the gate then scores the un-rendered tail as missing speech.
    private static var fixtureSeconds: Double {
        Double(ProcessInfo.processInfo.environment["VOICECHAT_FIXTURE_SECONDS"] ?? "") ?? 10.0
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

    /// Longest common subsequence of letters, as a fraction of `expected`.
    ///
    /// 🚨 This is an LCS and not a single forward walk, because the walk
    /// produces FALSE FAILURES. It only allowed skips in `actual`, so one
    /// missing letter early stalled the pointer for the rest of the string:
    /// a run whose speech was plainly correct — "That sounds delicious. I'd be
    /// happy to help you make some peanut butter cookies", 32 tokens — scored
    /// 11% purely because the ASR dropped a leading "Oh,". The same flaw scored
    /// a correct clip at 30% over the single letter difference between
    /// "favourite" and "favorite".
    ///
    /// An LCS allows skips on BOTH sides, so a dropped or altered word costs
    /// only that word. Everything downstream trusts this number, so it must not
    /// invent failures.
    static func inOrderLetterOverlap(expected: String, actual: String) -> Double {
        let a = Array(expected.lowercased().filter { $0.isLetter })
        let b = Array(actual.lowercased().filter { $0.isLetter })
        guard !a.isEmpty, !b.isEmpty else { return 0 }
        var previous = [Int](repeating: 0, count: b.count + 1)
        var current = previous
        for i in 1 ... a.count {
            current[0] = 0
            for j in 1 ... b.count {
                current[j] =
                    a[i - 1] == b[j - 1]
                    ? previous[j - 1] + 1
                    : Swift.max(previous[j], current[j - 1])
            }
            swap(&previous, &current)
        }
        return Double(previous[b.count]) / Double(a.count)
    }

    /// The gate's own yardstick, pinned. Runs without a bundle.
    ///
    /// Both cases below were observed scoring as failures on speech that was
    /// plainly correct, which is the worst thing this harness can do: it sends
    /// you hunting a runtime bug that does not exist.
    func testOverlapMetricDoesNotInventFailures() {
        let dropped = Self.inOrderLetterOverlap(
            expected: "Oh, that sounds delicious! I'd be happy to help you make some peanut butter cookies.",
            actual: "That sounds delicious. I'd be happy to help you make some peanut butter cookies")
        XCTAssertGreaterThan(
            dropped, 0.9,
            "a dropped leading word must cost only that word — the forward-walk "
                + "version scored this exact pair at 11%")

        let spelling = Self.inOrderLetterOverlap(
            expected: "You're my favourite friend in the whole world.",
            actual: "You're my favorite friend in the whole world")
        XCTAssertGreaterThan(
            spelling, 0.9, "one letter of spelling drift must not fail a clip (scored 30%)")

        // It still has to reject the thing it exists to catch.
        XCTAssertLessThan(
            Self.inOrderLetterOverlap(
                expected: "Oh, that sounds delicious! I'd be happy to help you make one.",
                actual: ""),
            0.01, "silence must score zero")
        XCTAssertLessThan(
            Self.inOrderLetterOverlap(
                expected: "Bye bye! Come back and play with me soon.",
                actual: "quartz vex hjkl zwq"),
            0.5, "unrelated words must not pass")
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
            Self.fixtureURL, targetRate: Double(config.inputSampleRate), seconds: Self.fixtureSeconds,
            offset: Self.fixtureOffset)
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
            Self.fixtureURL, targetRate: Double(config.inputSampleRate), seconds: Self.fixtureSeconds,
            offset: Self.fixtureOffset)
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

        // The frame-by-frame schedule the model's own text channel produces.
        // Anything that drives the speech tower directly has to reproduce this
        // shape, so print it rather than guessing at a token rate.
        let timeline = result.textTokens.map { id -> String in
            if id == config.padTokenId { return "." }
            if id == config.silenceTokenId { return "_" }
            if id == config.eosTokenId { return "|" }
            return (idToToken[id] ?? "?")
                .replacingOccurrences(of: "\u{2581}", with: "-")
                .replacingOccurrences(of: "Ġ", with: "-")
        }
        print("[schedule] frames=\(result.textTokens.count) " + timeline.joined(separator: " "))

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

    /// What is the text channel actually deciding, frame by frame?
    ///
    /// A turn that emits only PAD looks identical whether the backbone is
    /// broken (garbage or flat logits) or merely degraded (confidently choosing
    /// silence, which is by far the most common target in duplex training).
    /// This prints the top candidates and their probabilities so the two can be
    /// told apart instead of guessed at.
    func testTextChannelLogitDiagnostic() throws {
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

        let user = try readMono(
            Self.fixtureURL, targetRate: Double(config.inputSampleRate),
            seconds: Self.fixtureSeconds, offset: Self.fixtureOffset)
        let mel = voiceChatLogMelSpectrogram(user, config: config.audioConfig.preprocessor)
        let (audioEmbeds, _) = model.sttModel.perception(mel)

        let padId = config.padTokenId
        let cache = model.sttModel.makeCache()
        var previousText = padId
        var previousFunction = padId
        var padWins = 0
        var frames = 0

        for time in 0 ..< audioEmbeds.dim(1) {
            let fused =
                model.sttModel.embed(MLXArray([Int32(previousText)]).reshaped([1, 1]))
                + audioEmbeds[0..., time ..< (time + 1), 0...]
                + config.functionChannelWeight
                * model.sttModel.embed(MLXArray([Int32(previousFunction)]).reshaped([1, 1]))
            let output = model.sttModel(inputsEmbeds: fused, cache: cache)
            let logits = output.textLogits[0..., -1].asType(.float32)
            MLX.eval(logits)

            let probs = MLX.softmax(logits, axis: -1)
            let top = MLX.argSort(logits, axis: -1)[0..., (-5)...].asArray(Int32.self).reversed()
            let probValues = probs.asArray(Float.self)
            let best = Int(top.first ?? 0)
            previousText = best
            previousFunction = output.functionLogits[0..., -1].argMax().item(Int.self)
            frames += 1
            if best == padId { padWins += 1 }

            if time % 20 == 0 || (best != padId && frames < 200) {
                let candidates = top.prefix(5).map { id -> String in
                    let name = Int(id) == padId ? "<PAD>" : (idToToken[Int(id)] ?? "?\(id)")
                    return String(format: "%@ %.3f", name, probValues[Int(id)])
                }
                let allValues = logits.asArray(Float.self)
                print(
                    String(
                        format: "[logits] f%03d max %+.2f mean %+.2f finite=%@ | ",
                        time, allValues.max() ?? 0,
                        allValues.reduce(0, +) / Float(allValues.count),
                        allValues.allSatisfy { $0.isFinite } ? "yes" : "NO")
                        + candidates.joined(separator: "  "))
            }
            if time > 60 { break }
        }
        print("[logits] PAD won \(padWins)/\(frames) frames")
    }

    /// Render chosen sentences with the speech tower and prove each one says
    /// what it was asked to say.
    ///
    /// `VOICECHAT_TTS_SCRIPT` points at JSON: `[{"name": …, "text": …,
    /// "ids": [...]}, …]` — ids come from the bundle's own tokenizer, so this
    /// test does not have to reimplement tokenization to be trustworthy.
    func testSynthesizeChosenLines() throws {
        guard Self.enabled else { throw XCTSkip("set VOICECHAT_QUANT_PROOF=1") }
        guard let script = ProcessInfo.processInfo.environment["VOICECHAT_TTS_SCRIPT"] else {
            throw XCTSkip("set VOICECHAT_TTS_SCRIPT")
        }
        let bundle = URL(fileURLWithPath: Self.bundlePath)
        guard VoiceChatLoader.looksLikeVoiceChatBundle(at: bundle) else {
            throw XCTSkip("no VoiceChat bundle")
        }
        let (model, config) = try VoiceChatLoader.load(from: bundle)
        let data = try Data(contentsOf: bundle.appendingPathComponent("tokenizer.json"))
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        let vocab = try XCTUnwrap((json?["model"] as? [String: Any])?["vocab"] as? [String: Int])
        try model.setVocabulary(vocab)

        struct Line: Decodable {
            let name: String
            let text: String
            let ids: [Int]
        }
        let lines = try JSONDecoder().decode(
            [Line].self, from: try Data(contentsOf: URL(fileURLWithPath: script)))
        // 1 = the model's own burst pacing (see `frameSchedule`). Spreading
        // tokens out garbles everything after the first word or two.
        let framesPerToken =
            Int(ProcessInfo.processInfo.environment["VOICECHAT_TTS_FPT"] ?? "1") ?? 1

        var weak = [String]()
        for line in lines {
            let schedule = model.frameSchedule(
                subwordIds: line.ids, framesPerToken: framesPerToken)
            let out = model.synthesize(frameTokens: schedule)
            let samples = out.audio.asType(.float32).asArray(Float.self)
            try writeWAV(
                samples, sampleRate: out.sampleRate,
                to: Self.proofDir.appendingPathComponent("say-\(line.name).wav"))
            let heard = Self.transcribe(
                resample(samples, from: out.sampleRate, to: config.inputSampleRate),
                model: model, config: config)
            let overlap = Self.inOrderLetterOverlap(expected: line.text, actual: heard.text)
            print(
                String(
                    format: "[say] %-14s %5.2fs asked \"%@\" → heard \"%@\" (%d tok) %.0f%%",
                    (line.name as NSString).utf8String!,
                    Double(samples.count) / Double(out.sampleRate),
                    line.text, heard.text, heard.tokenCount, overlap * 100))
            if overlap <= 0.5 { weak.append("\(line.name) (\(Int(overlap * 100))%)") }
        }
        XCTAssertTrue(
            weak.isEmpty,
            "these lines did not come back as the words they were given: "
                + weak.joined(separator: ", "))
    }

    /// Batch gate: every wav in `VOICECHAT_TRANSCRIBE_DIR` must contain WORDS,
    /// read back by the model's own ASR, in one model load.
    ///
    /// 🚨 This exists because a delivered folder of "voice samples" once turned
    /// out to be babble. Every clip in it passed RMS, dynamic-range and
    /// zero-crossing checks — those statistics are satisfied by structured
    /// noise, so they certify nothing about speech. Post-processing (pitch
    /// shifting for character voices) is exactly the kind of step that can
    /// destroy words while leaving the statistics intact, so nothing ships
    /// from a sample folder until it has been through here.
    ///
    /// Set `VOICECHAT_TRANSCRIBE_EXPECT` to the sentence the clips should say
    /// to also gate in-order letter overlap against it.
    func testEveryClipInDirectoryContainsWords() throws {
        guard Self.enabled else { throw XCTSkip("set VOICECHAT_QUANT_PROOF=1") }
        guard let dir = ProcessInfo.processInfo.environment["VOICECHAT_TRANSCRIBE_DIR"] else {
            throw XCTSkip("set VOICECHAT_TRANSCRIBE_DIR")
        }
        let bundle = URL(fileURLWithPath: Self.bundlePath)
        guard VoiceChatLoader.looksLikeVoiceChatBundle(at: bundle) else {
            throw XCTSkip("no VoiceChat bundle")
        }
        let (model, config) = try VoiceChatLoader.load(from: bundle)

        let files = try FileManager.default
            .contentsOfDirectory(at: URL(fileURLWithPath: dir), includingPropertiesForKeys: nil)
            .filter { $0.pathExtension.lowercased() == "wav" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        XCTAssertFalse(files.isEmpty, "no wav files in \(dir)")

        let expected = ProcessInfo.processInfo.environment["VOICECHAT_TRANSCRIBE_EXPECT"]
        var silent = [String]()
        var garbled = [String]()
        for file in files {
            let samples = try readMono(
                file, targetRate: Double(config.inputSampleRate), seconds: 30.0)
            let heard = Self.transcribe(samples, model: model, config: config)
            var line = "[clip] \(file.lastPathComponent): "
                + "\"\(heard.text.prefix(100))\" (\(heard.tokenCount) tokens)"
            if let expected {
                let overlap = Self.inOrderLetterOverlap(expected: expected, actual: heard.text)
                line += String(format: " overlap %.0f%%", overlap * 100)
                if overlap <= 0.5 { garbled.append(file.lastPathComponent) }
            }
            print(line)
            if heard.tokenCount == 0 { silent.append(file.lastPathComponent) }
        }
        XCTAssertTrue(
            silent.isEmpty,
            "these clips transcribe to NOTHING — they are not speech, whatever their energy "
                + "statistics say: \(silent.joined(separator: ", "))")
        XCTAssertTrue(
            garbled.isEmpty,
            "these clips do not say the expected sentence: \(garbled.joined(separator: ", "))")
    }
}
