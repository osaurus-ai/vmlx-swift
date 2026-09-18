import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLMCommon
@testable import MLXVLM

/// Stage-by-stage parity for the speech decoder against the Python reference.
///
/// The intelligibility loop proved the generated audio is not speech while the
/// text channel, the mel front-end, the character conditioning and the codec
/// round trip are all exact — so the fault is inside this stack. These tests
/// compare the two places a divergence can hide: the state the warmup writes
/// (which carries the speaker) and the hidden state of the first decode step
/// (which carries everything else).
public class VoiceChatTTSParityTests: XCTestCase {

    private static var enabled: Bool {
        ProcessInfo.processInfo.environment["VOICECHAT_QUANT_PROOF"] == "1"
    }

    private static var bundlePath: String {
        ProcessInfo.processInfo.environment["VOICECHAT_QUANT_DIR"]
            ?? FileManager.default.homeDirectoryForCurrentUser
                .appendingPathComponent("models/OsaurusAI/NemotronLabs-VoiceChat-11B-JANG_4")
                .path
    }

    private static var dumpDir: URL? {
        ProcessInfo.processInfo.environment["VOICECHAT_REF_DUMP"].map {
            URL(fileURLWithPath: $0)
        }
    }

    private func loadFloats(_ url: URL) throws -> [Float] {
        let data = try Data(contentsOf: url)
        return data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }

    private func stats(_ values: [Float]) -> (mean: Float, std: Float) {
        guard !values.isEmpty else { return (0, 0) }
        let mean = values.reduce(0, +) / Float(values.count)
        let variance =
            values.reduce(Float(0)) { $0 + ($1 - mean) * ($1 - mean) } / Float(values.count)
        return (mean, variance.squareRoot())
    }

    private func cosine(_ a: [Float], _ b: [Float]) -> Double {
        let n = min(a.count, b.count)
        var dot = 0.0, na = 0.0, nb = 0.0
        for i in 0 ..< n {
            dot += Double(a[i]) * Double(b[i])
            na += Double(a[i]) * Double(a[i])
            nb += Double(b[i]) * Double(b[i])
        }
        return (na > 0 && nb > 0) ? dot / (na.squareRoot() * nb.squareRoot()) : 0
    }

    func testWarmupAndStepMatchTheReference() throws {
        guard Self.enabled else { throw XCTSkip("set VOICECHAT_QUANT_PROOF=1") }
        guard let dump = Self.dumpDir else { throw XCTSkip("set VOICECHAT_REF_DUMP") }
        let bundle = URL(fileURLWithPath: Self.bundlePath)
        let (model, config) = try VoiceChatLoader.load(from: bundle)
        let data = try Data(contentsOf: bundle.appendingPathComponent("tokenizer.json"))
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        let vocab = try XCTUnwrap((json?["model"] as? [String: Any])?["vocab"] as? [String: Int])
        try model.setVocabulary(vocab)

        // Same prompt construction as the reference session.
        let prompt = model.ttsPrompt()
        MLX.eval(prompt.codes)
        print(
            "[tts-parity] prompt codes \(prompt.codes.shape) · subwords \(prompt.subwords.shape)")

        let (warmHidden, cache) = model.ttsModel.ttsModel.warmup(
            code: prompt.codes,
            subwordIds: prompt.subwords,
            subwordMask: prompt.subwordMask,
            audioMask: prompt.audioMask,
            audioPromptLatent: prompt.latent,
            guidance: true)
        MLX.eval(warmHidden)
        let mineWarm = warmHidden.asType(.float32).reshaped([-1]).asArray(Float.self)
        let refWarm = try loadFloats(dump.appendingPathComponent("ref-tts-warm.f32"))
        let mw = stats(mineWarm), rw = stats(refWarm)
        print(
            String(
                format: "[tts-parity] warm mine mean %.6f std %.6f · ref mean %.6f std %.6f "
                    + "· cosine %.4f",
                mw.mean, mw.std, rw.mean, rw.std, cosine(mineWarm, refWarm)))
        let halfW = min(mineWarm.count, refWarm.count) / 2
        print(
            String(
                format: "[tts-parity] warm halves: cond %.5f · uncond %.5f",
                cosine(Array(mineWarm[0 ..< halfW]), Array(refWarm[0 ..< halfW])),
                cosine(Array(mineWarm[halfW...]), Array(refWarm[halfW...]))))
        XCTAssertEqual(
            mineWarm.count, refWarm.count, "warmup hidden shape differs from the reference")
        XCTAssertGreaterThan(
            cosine(mineWarm, refWarm), 0.99,
            "the TTS warmup diverges from the reference — the speaker/prompt state the whole "
                + "turn is conditioned on is already wrong")

        print(
            "[tts-parity] cache offsets after warmup: "
                + "rotating[0]=\(cache[0].offset) global[\(model.ttsModel.ttsModel.backbone.slidingWindowPattern - 1)]="
                + "\(cache[model.ttsModel.ttsModel.backbone.slidingWindowPattern - 1].offset)")

        // Are the two guidance rows in the cache actually distinct?
        for layerIndex in [0, 1] {
            let stored = cache[layerIndex].state
            if let k = stored.first, k.dim(0) == 2 {
                MLX.eval(k)
                let row0 = MLX.sum(k[0].asType(.float32) * k[0].asType(.float32)).item(Float.self)
                    .squareRoot()
                let row1 = MLX.sum(k[1].asType(.float32) * k[1].asType(.float32)).item(Float.self)
                    .squareRoot()
                let diff = MLX.abs(k[0].asType(.float32) - k[1].asType(.float32)).max()
                    .item(Float.self)
                print(
                    String(
                        format: "[tts-parity] cache[%d] keys: cond|n|=%.2f uncond|n|=%.2f maxdiff=%.4f",
                        layerIndex, row0, row1, diff))
            }
        }

        // First decode step, identical inputs.
        let prev = prompt.codes[0..., (prompt.codes.dim(1) - 1)..., 0...]
        let current = MLXArray([Int32(5501)]).reshaped([1, 1])
        let step = model.ttsModel.ttsModel.step(
            code: prev, subwordIds: current,
            subwordMask: MLXArray.ones([1, 1], dtype: .bool), cache: cache, guidance: true)
        MLX.eval(step.hiddenStates, step.codes)
        let mineStep = step.hiddenStates.asType(.float32).reshaped([-1]).asArray(Float.self)
        let refStep = try loadFloats(dump.appendingPathComponent("ref-tts-step.f32"))
        let ms = stats(mineStep), rs = stats(refStep)
        print(
            String(
                format: "[tts-parity] step mine mean %.6f std %.6f · ref mean %.6f std %.6f "
                    + "· cosine %.4f",
                ms.mean, ms.std, rs.mean, rs.std, cosine(mineStep, refStep)))
        print(
            "[tts-parity] step codes mine "
                + "\(step.codes.asType(.int32).reshaped([-1]).asArray(Int32.self).prefix(12))")
        _ = config
        XCTAssertGreaterThan(
            cosine(mineStep, refStep), 0.99,
            "the first TTS decode step diverges from the reference")
    }
}
