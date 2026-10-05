import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLMCommon
@testable import MLXVLM

/// Parity for the character-aware TEXT CONDITIONING the speech decoder runs on.
///
/// This is the one stage where a silent mistake produces exactly the symptom
/// observed: perfect text on the language channel, babble on the speech
/// channel. The conditioning is built from each subword's CHARACTERS through a
/// vocabulary the runtime derives itself, so a different derivation yields
/// different — but perfectly well-formed — vectors, and every structural check
/// still passes.
///
/// Compares against a dump from the Python reference
/// (`mlx_vlm.models.nemotron_voicechat`) on identical token ids.
public class VoiceChatConditioningParityTests: XCTestCase {

    private static var enabled: Bool {
        ProcessInfo.processInfo.environment["VOICECHAT_QUANT_PROOF"] == "1"
    }

    private static var bundlePath: String {
        ProcessInfo.processInfo.environment["VOICECHAT_QUANT_DIR"]
            ?? FileManager.default.homeDirectoryForCurrentUser
                .appendingPathComponent("models/OsaurusAI/NemotronLabs-VoiceChat-11B-JANG_4")
                .path
    }

    func testCharacterConditioningMatchesTheReference() throws {
        guard Self.enabled else { throw XCTSkip("set VOICECHAT_QUANT_PROOF=1") }
        guard let dumpPath = ProcessInfo.processInfo.environment["VOICECHAT_REF_COND"],
            let idsRaw = ProcessInfo.processInfo.environment["VOICECHAT_REF_IDS"]
        else { throw XCTSkip("set VOICECHAT_REF_COND and VOICECHAT_REF_IDS") }

        let bundle = URL(fileURLWithPath: Self.bundlePath)
        let (model, _) = try VoiceChatLoader.load(from: bundle)
        let data = try Data(contentsOf: bundle.appendingPathComponent("tokenizer.json"))
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        let vocab = try XCTUnwrap((json?["model"] as? [String: Any])?["vocab"] as? [String: Int])
        try model.setVocabulary(vocab)

        let ids = idsRaw.split(separator: ",").compactMap { Int32($0) }
        let tokens = MLXArray(ids).reshaped([1, ids.count])
        let cond = model.ttsModel.ttsModel.embedSubword(tokens)
        MLX.eval(cond)
        let mine = cond.asType(.float32).reshaped([-1]).asArray(Float.self)

        let refData = try Data(contentsOf: URL(fileURLWithPath: dumpPath))
        let reference: [Float] = refData.withUnsafeBytes {
            Array($0.bindMemory(to: Float.self))
        }

        XCTAssertEqual(
            mine.count, reference.count,
            "conditioning shape differs from the reference (\(mine.count) vs \(reference.count))")

        let n = min(mine.count, reference.count)
        var dot: Double = 0, na: Double = 0, nb: Double = 0, maxDiff: Float = 0
        for i in 0 ..< n {
            dot += Double(mine[i]) * Double(reference[i])
            na += Double(mine[i]) * Double(mine[i])
            nb += Double(reference[i]) * Double(reference[i])
            maxDiff = max(maxDiff, abs(mine[i] - reference[i]))
        }
        let cosine = (na > 0 && nb > 0) ? dot / (na.squareRoot() * nb.squareRoot()) : 0
        let mineMean = mine.reduce(0, +) / Float(max(mine.count, 1))
        let refMean = reference.reduce(0, +) / Float(max(reference.count, 1))
        print(
            String(
                format:
                    "[cond-parity] mine mean %.6f · ref mean %.6f · max abs diff %.4f · cosine %.4f",
                mineMean, refMean, maxDiff, cosine))

        XCTAssertGreaterThan(
            cosine, 0.99,
            "character conditioning diverges from the reference — the speech decoder is being "
                + "told something different from what the text says, which is exactly how the "
                + "text channel stays perfect while the audio is babble")
    }
}
