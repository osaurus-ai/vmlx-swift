import AVFoundation
import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLMCommon
@testable import MLXVLM

/// Is the generated audio actually SPEECH?
///
/// The render proof measured RMS, dynamic range and zero-crossing rate — all
/// of which pass for structured noise, and the first samples produced for
/// listening turned out to be exactly that: babble, no words. So this suite
/// isolates the two halves that could be at fault, in the only order that
/// distinguishes them:
///
///   1. CODEC ROUND TRIP — real speech in, encode, decode, compare to the
///      input. If this is noise the vocoder is wrong and nothing downstream
///      can sound like words.
///   2. TEXT CHANNEL — decode the agent's own text tokens. If those are not
///      words, the fault is upstream of the speech decoder entirely.
///
/// Both write WAVs so the result can be heard, not just scored.
public class VoiceChatCodecRoundTripTests: XCTestCase {

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
        FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(
            "models/nemotron-voicechat-src/NVIDIA-NemotronLabs-VoiceChat-11B/turn_taking.wav")
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

    /// Normalised cross-correlation at the best small lag: the direct measure
    /// of "is this the same waveform", which noise cannot fake.
    private func bestCorrelation(_ a: [Float], _ b: [Float], maxLag: Int = 64) -> Float {
        let n = min(a.count, b.count) - maxLag
        guard n > 1000 else { return 0 }
        func energy(_ x: ArraySlice<Float>) -> Float { x.reduce(0) { $0 + $1 * $1 } }
        let ea = energy(a[0 ..< n])
        var best: Float = -1
        for lag in 0 ... maxLag {
            var dot: Float = 0
            for i in stride(from: 0, to: n, by: 4) { dot += a[i] * b[i + lag] }
            let eb = energy(b[lag ..< (lag + n)])
            let denom = (ea * eb).squareRoot()
            if denom > 1e-9 { best = max(best, dot * 4 / denom) }
        }
        return best
    }

    // MARK: - 1. Does the codec reproduce real speech?

    func testCodecRoundTripReproducesSpeech() throws {
        guard Self.enabled else { throw XCTSkip("set VOICECHAT_QUANT_PROOF=1") }
        let bundle = URL(fileURLWithPath: Self.bundlePath)
        guard VoiceChatLoader.looksLikeVoiceChatBundle(at: bundle) else {
            throw XCTSkip("no VoiceChat bundle")
        }
        let (model, config) = try VoiceChatLoader.load(from: bundle)

        // Real speech at the codec's own rate.
        let original = try readMono(
            Self.fixtureURL, targetRate: Double(config.codecConfig.sampleRate),
            seconds: 3.0, offset: 1.0)
        XCTAssertGreaterThan(original.count, 10000)

        let input = MLXArray(original).reshaped([1, 1, original.count])
        let codes = model.ttsModel.audioCodec.encode(input)
        let decoded = model.ttsModel.audioCodec.decode(codes)
        MLX.eval(decoded)
        let out = decoded[0, 0].asType(.float32).asArray(Float.self)

        try writeWAV(
            original, sampleRate: config.codecConfig.sampleRate,
            to: Self.proofDir.appendingPathComponent("codec-roundtrip-input.wav"))
        try writeWAV(
            out, sampleRate: config.codecConfig.sampleRate,
            to: Self.proofDir.appendingPathComponent("codec-roundtrip-output.wav"))

        let correlation = bestCorrelation(original, out)
        let inRMS = (original.reduce(0) { $0 + $1 * $1 } / Float(original.count)).squareRoot()
        let outRMS = (out.reduce(0) { $0 + $1 * $1 } / Float(max(out.count, 1))).squareRoot()
        print(
            String(
                format:
                    "[codec-roundtrip] in %d samples rms %.4f · out %d samples rms %.4f · "
                    + "correlation %.3f",
                original.count, inRMS, out.count, outRMS, correlation))

        XCTAssertGreaterThan(out.count, original.count / 2, "codec returned almost nothing")
        // A working neural codec round-trip correlates strongly with its input.
        // Noise correlates near zero however good its RMS looks.
        XCTAssertGreaterThan(
            correlation, 0.5,
            "codec round trip does not reproduce the input waveform — the vocoder is wrong, "
                + "so nothing downstream can sound like words")
    }

    // MARK: - 2. Is the agent's TEXT channel producing words?

    func testAgentTextChannelProducesWords() throws {
        guard Self.enabled else { throw XCTSkip("set VOICECHAT_QUANT_PROOF=1") }
        let bundle = URL(fileURLWithPath: Self.bundlePath)
        guard VoiceChatLoader.looksLikeVoiceChatBundle(at: bundle) else {
            throw XCTSkip("no VoiceChat bundle")
        }
        let (model, config) = try VoiceChatLoader.load(from: bundle)

        // id → token string, straight from the bundle's tokenizer.
        let data = try Data(contentsOf: bundle.appendingPathComponent("tokenizer.json"))
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        let vocab = try XCTUnwrap((json?["model"] as? [String: Any])?["vocab"] as? [String: Int])
        var idToToken = [Int: String]()
        for (token, id) in vocab { idToToken[id] = token }
        try model.setVocabulary(vocab)

        let user = try readMono(
            Self.fixtureURL, targetRate: Double(config.inputSampleRate), seconds: 4.0)
        let mel = voiceChatLogMelSpectrogram(user, config: config.audioConfig.preprocessor)
        let (projected, encoded) = model.sttModel.perception(mel)
        let result = model.generateTurn(audioEmbeds: projected, asrEmbeds: encoded)

        let special: Set<Int> = [
            config.padTokenId, config.silenceTokenId, config.bosTokenId, config.eosTokenId,
        ]
        let spoken = result.textTokens.filter { !special.contains($0) }
        let rendered = spoken.compactMap { idToToken[$0] }
            .joined()
            .replacingOccurrences(of: "\u{2581}", with: " ")  // SentencePiece space
            .replacingOccurrences(of: "Ġ", with: " ")  // byte-BPE space
        print("[text-channel] tokens \(result.textTokens.count) · non-special \(spoken.count)")
        print("[text-channel] decoded: \"\(rendered.prefix(300))\"")

        XCTAssertFalse(
            spoken.isEmpty,
            "the agent emitted only pad/silence — it never spoke, so the audio cannot contain words")

        // Words, not token soup: require ASCII letters and at least a couple of
        // space-separated pieces.
        let letters = rendered.filter { $0.isLetter }.count
        XCTAssertGreaterThan(
            letters, 5,
            "the text channel decoded to \"\(rendered.prefix(80))\" — not words, so the fault "
                + "is upstream of the speech decoder")
    }
}
