import AVFoundation
import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLMCommon
@testable import MLXVLM

/// Diagnostic for the speaker-prompt path: does a reference recording actually
/// reach the prompt frames, and does a DIFFERENT recording produce DIFFERENT
/// frames?
///
/// This exists because the first cloning run showed three references producing
/// byte-identical audio to the built-in voice while a fourth changed it — a
/// result that cannot be explained by "the model rendered similar speech", so
/// it had to be measured at the prompt itself rather than at the output.
public class VoiceChatPromptPathDiagnosticTests: XCTestCase {

    private static var enabled: Bool {
        ProcessInfo.processInfo.environment["VOICECHAT_QUANT_PROOF"] == "1"
    }

    private static var bundlePath: String {
        ProcessInfo.processInfo.environment["VOICECHAT_QUANT_DIR"]
            ?? FileManager.default.homeDirectoryForCurrentUser
                .appendingPathComponent("models/OsaurusAI/NemotronLabs-VoiceChat-11B-JANG_4")
                .path
    }

    private static var voicesDir: URL {
        URL(
            fileURLWithPath: ProcessInfo.processInfo.environment["VOICECHAT_VOICES_DIR"]
                ?? NSTemporaryDirectory())
    }

    private func readMono(_ url: URL, targetRate: Double, seconds: Double, offset: Double)
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
        var out = [Float]()
        for i in 0 ..< Int(seconds * targetRate) {
            let position = offset * sourceRate + Double(i) * sourceRate / targetRate
            let index = Int(position)
            guard index + 1 < source.count else { break }
            let fraction = Float(position - Double(index))
            out.append(source[index] * (1 - fraction) + source[index + 1] * fraction)
        }
        return out
    }

    func testReferenceAudioReachesThePromptCodes() throws {
        guard Self.enabled else { throw XCTSkip("set VOICECHAT_QUANT_PROOF=1") }
        let bundle = URL(fileURLWithPath: Self.bundlePath)
        guard VoiceChatLoader.looksLikeVoiceChatBundle(at: bundle) else {
            throw XCTSkip("no VoiceChat bundle")
        }
        let voiceFiles =
            ((try? FileManager.default.contentsOfDirectory(
                at: Self.voicesDir, includingPropertiesForKeys: nil)) ?? [])
            .filter { $0.lastPathComponent.hasPrefix("voice-") && $0.pathExtension == "wav" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        try XCTSkipIf(voiceFiles.isEmpty, "no references")

        let (model, config) = try VoiceChatLoader.load(from: bundle)

        // Silence baseline — what the built-in path encodes.
        let silentPrompt = model.ttsPrompt(voice: .builtIn)
        MLX.eval(silentPrompt.codes)
        let silentCodes = silentPrompt.codes.asType(.int32).asArray(Int32.self)

        var signatures = [String: [Int32]]()
        for file in voiceFiles {
            let name = file.deletingPathExtension().lastPathComponent
            let reference = try readMono(
                file, targetRate: Double(config.codecConfig.sampleRate),
                seconds: Double(config.ttsConfig.audioPromptDuration ?? 3.0), offset: 0.4)
            let prompt = model.ttsPrompt(voice: .reference(reference))
            MLX.eval(prompt.codes)
            let codes = prompt.codes.asType(.int32).asArray(Int32.self)
            let uniqueCount = Set(codes).count
            let differingFromSilence = zip(codes, silentCodes).filter { $0 != $1 }.count
            let inputRMS =
                (reference.reduce(Float(0)) { $0 + $1 * $1 } / Float(max(reference.count, 1)))
                .squareRoot()
            print(
                String(
                    format:
                        "[prompt-path] %-18s inRMS %.4f · codes %d · unique %d · differ-from-silence %d (%.0f%%) · latent %@",
                    (name as NSString).utf8String!, inputRMS, codes.count, uniqueCount,
                    differingFromSilence,
                    100.0 * Double(differingFromSilence) / Double(max(codes.count, 1)),
                    prompt.latent == nil ? "derived" : "built-in"))
            signatures[name] = codes

            XCTAssertNil(prompt.latent, "\(name): a reference must derive its own prompt latent")
            XCTAssertGreaterThan(
                differingFromSilence, 0,
                "\(name): reference audio encoded to the SAME codes as silence — the prompt "
                    + "never carries the speaker")
        }

        // Different references must produce different prompts.
        let names = signatures.keys.sorted()
        for i in 0 ..< names.count {
            for j in (i + 1) ..< names.count {
                let a = signatures[names[i]]!, b = signatures[names[j]]!
                let differing = zip(a, b).filter { $0 != $1 }.count
                XCTAssertGreaterThan(
                    differing, 0,
                    "\(names[i]) and \(names[j]) produced identical prompt codes")
            }
        }
    }
}
