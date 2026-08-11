import Foundation
import MLX
import Testing

@testable import MLXLLM

/// `sanitize` folds the centered-norm `+1` into the weights once at load, so
/// the per-token forward can skip it. That trade only holds if the fold covers
/// *every* centered norm: a missed one silently loses its `+1`, turning unit
/// gain into zero gain for those layers. The model keeps running and keeps
/// producing fluent text, which is exactly why this needs pinning — the first
/// version of the fold missed two of the four per-layer norms.
@Suite("Muse Glimmer centered norm coverage")
struct MuseGlimmerCenteredNormCoverageTests {

    static let bundle = URL(fileURLWithPath: NSHomeDirectory())
        .appendingPathComponent("models/JANGQ-AI/Muse-Glimmer-30B-JANG_4M")

    static var enabled: Bool {
        FileManager.default.fileExists(
            atPath: bundle.appendingPathComponent("model.safetensors.index.json").path)
    }

    @Test("every centered norm in the checkpoint is folded", .enabled(if: enabled))
    func coverage() throws {
        struct Index: Decodable { let weightMap: [String: String]
            enum CodingKeys: String, CodingKey { case weightMap = "weight_map" } }
        let data = try Data(
            contentsOf: Self.bundle.appendingPathComponent("model.safetensors.index.json"))
        let keys = try JSONDecoder().decode(Index.self, from: data).weightMap.keys

        // Text-tower norm weights, as the loader sees them after the
        // `language_model.` prefix is stripped.
        let textNorms = keys
            .filter { $0.hasPrefix("language_model.") && $0.contains("norm") && $0.hasSuffix(".weight") }
            .map { String($0.dropFirst("language_model.".count)) }
            .filter { !$0.contains("qk_norm") }
        #expect(!textNorms.isEmpty, "no text-tower norms found in the index")

        let missed = textNorms.filter { !MuseGlimmerTextModel.isCenteredNormWeight($0) }
        print("[norms] text-tower norm weights: \(textNorms.count), not folded: \(missed.count)")
        for m in Set(missed.map { key -> String in
            key.split(separator: ".").map { Int($0) == nil ? String($0) : "N" }.joined(separator: ".")
        }) {
            print("[norms]   MISSED \(m)")
        }
        #expect(missed.isEmpty,
            "these centered norms are not folded and will run without their +1: \(Set(missed.prefix(5)))")

        // The VLM loads with the `language_model.` prefix intact and uses its
        // own sanitize, so the same key list has to match that form too — the
        // first fold missed the final norm there and left the whole text tower
        // at zero gain on the vision path only.
        let vlmForm = keys.filter {
            $0.hasPrefix("language_model.") && $0.contains("norm") && $0.hasSuffix(".weight")
                && !$0.contains("qk_norm")
        }
        let vlmMissed = vlmForm.filter { !MuseGlimmerTextModel.isCenteredNormWeight($0) }
        print("[norms] VLM-form norms: \(vlmForm.count), not folded: \(vlmMissed.count)")
        #expect(vlmMissed.isEmpty,
            "VLM-form keys not folded: \(Set(vlmMissed.prefix(5)))")

        // And nothing outside the text tower may be folded.
        let visionNorms = keys.filter { $0.hasPrefix("model.vision_tower.") && $0.hasSuffix("norm.weight") }
        for v in visionNorms {
            #expect(!MuseGlimmerTextModel.isCenteredNormWeight(v),
                "vision LayerNorm \(v) must not be shifted")
        }
    }
}
