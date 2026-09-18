import AVFoundation
import Foundation
import MLX
import XCTest

@testable import MLXVLM

/// Dumps the exact 16 kHz samples and the mel features this runtime computes,
/// so they can be compared against the Python reference front-end
/// (`mlx_audio.stt.models.nemotron_asr.audio.log_mel_spectrogram`) on
/// byte-identical input.
///
/// A mel mismatch is invisible downstream: the encoder still produces
/// plausible features, the language model still answers, and only accuracy
/// (ASR words, speech quality) quietly degrades — which is exactly the class
/// of bug that shipped babble while every energy statistic looked healthy.
public class VoiceChatMelDumpTests: XCTestCase {

    func testDumpSamplesAndMel() throws {
        guard ProcessInfo.processInfo.environment["VOICECHAT_MEL_DUMP"] == "1" else {
            throw XCTSkip("set VOICECHAT_MEL_DUMP=1")
        }
        let out = URL(
            fileURLWithPath: ProcessInfo.processInfo.environment["VOICECHAT_PROOF_DIR"]
                ?? NSTemporaryDirectory())
        try? FileManager.default.createDirectory(
            at: out, withIntermediateDirectories: true)

        let fixture = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(
            "models/nemotron-voicechat-src/NVIDIA-NemotronLabs-VoiceChat-11B/turn_taking.wav")
        let file = try AVAudioFile(forReading: fixture)
        let format = file.processingFormat
        let buffer = try XCTUnwrap(
            AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(file.length)))
        try file.read(into: buffer)
        let channels = try XCTUnwrap(buffer.floatChannelData)
        let source = Array(
            UnsafeBufferPointer(start: channels[0], count: Int(buffer.frameLength)))

        var samples = [Float]()
        for i in 0 ..< Int(6.0 * 16000) {
            let position = Double(i) * format.sampleRate / 16000.0
            let index = Int(position)
            guard index + 1 < source.count else { break }
            let fraction = Float(position - Double(index))
            samples.append(source[index] * (1 - fraction) + source[index + 1] * fraction)
        }

        let config = VoiceChatPreprocessorConfiguration(
            sampleRate: 16000, features: 128, nFFT: 512, windowSize: 0.025,
            windowStride: 0.01)
        let mel = voiceChatLogMelSpectrogram(samples, config: config)
        MLX.eval(mel)
        let flat = mel.reshaped([-1]).asType(.float32).asArray(Float.self)

        samples.withUnsafeBufferPointer {
            try? Data(buffer: $0).write(to: out.appendingPathComponent("mel-samples.f32"))
        }
        flat.withUnsafeBufferPointer {
            try? Data(buffer: $0).write(to: out.appendingPathComponent("mel-swift.f32"))
        }
        print(
            "[mel-dump] samples \(samples.count) · mel shape \(mel.shape) · wrote to \(out.path)")
    }
}

extension VoiceChatPreprocessorConfiguration {
    /// Test-only constructor; the shipped type decodes from the bundle.
    init(sampleRate: Int, features: Int, nFFT: Int, windowSize: Double, windowStride: Double) {
        let json = """
            {"sample_rate": \(sampleRate), "features": \(features), "n_fft": \(nFFT),
             "window_size": \(windowSize), "window_stride": \(windowStride)}
            """
        self = try! JSONDecoder().decode(
            VoiceChatPreprocessorConfiguration.self, from: Data(json.utf8))
    }
}
