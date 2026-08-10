import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXVLM

/// `patch_embedding.weight` is `(1536, 1176)` and 1176 = 3 channels x 2 temporal
/// x 14 x 14. Nothing in the checkpoint states the order of those factors, and
/// the two candidates — `[channel][temporal][h][w]` (what `QwenVL.patchify`
/// emits, and what this port assumes) and `[temporal][channel][h][w]` — produce
/// identical shapes. Feeding the wrong one scrambles colour across the temporal
/// slots while leaving greyscale untouched, which is hard to spot.
///
/// The weights themselves settle it. A still image duplicates the frame across
/// the temporal axis, so the two temporal slots see identical input and their
/// learned weights co-adapt. Under the true layout, temporal pairs for a given
/// channel are far more alike than unrelated slices; under the false one, the
/// pairing lines up slices that never shared an input.
@Suite("Muse Glimmer patch layout")
struct MuseGlimmerPatchLayoutTests {

    static let bundle = URL(fileURLWithPath: NSHomeDirectory())
        .appendingPathComponent("models/JANGQ-AI/Muse-Glimmer-30B-JANG_6M")

    static var enabled: Bool {
        ProcessInfo.processInfo.environment["MUSE_VL_LIVE"] == "1"
            && FileManager.default.fileExists(
                atPath: bundle.appendingPathComponent("config.json").path)
    }

    @Test("the weights identify [temporal][channel] as the patch vector layout",
        .enabled(if: enabled))
    func layoutMatchesWeights() throws {
        var weight: MLXArray?
        for f in try FileManager.default.contentsOfDirectory(
            at: Self.bundle, includingPropertiesForKeys: nil)
        where f.pathExtension == "safetensors" {
            guard let arrays = try? MLX.loadArrays(url: f) else { continue }
            for (k, v) in arrays where k.hasSuffix("patch_embedder.patch_embedding.weight") {
                weight = v.asType(.float32)
            }
        }
        let w = try #require(weight)
        let (out, flat) = (w.dim(0), w.dim(1))
        let spatial = 14 * 14
        #expect(flat == 3 * 2 * spatial, "unexpected patch vector width \(flat)")

        func cosine(_ a: MLXArray, _ b: MLXArray) -> Float {
            let dot = (a * b).sum()
            let na = MLX.sqrt(a.square().sum())
            let nb = MLX.sqrt(b.square().sum())
            let v = dot / (na * nb + 1e-9)
            v.eval()
            return v.item(Float.self)
        }

        // Hypothesis A — [channel][temporal][h][w]: temporal partners sit
        // side by side inside each channel block.
        let a = w.reshaped(out, 3, 2, spatial)
        var aScore: Float = 0
        for c in 0 ..< 3 {
            aScore += cosine(a[0..., c, 0, 0...].flattened(), a[0..., c, 1, 0...].flattened())
        }
        aScore /= 3

        // Hypothesis B — [temporal][channel][h][w]: temporal partners are a
        // whole channel-block apart.
        let b = w.reshaped(out, 2, 3, spatial)
        var bScore: Float = 0
        for c in 0 ..< 3 {
            bScore += cosine(b[0..., 0, c, 0...].flattened(), b[0..., 1, c, 0...].flattened())
        }
        bScore /= 3

        // Baseline: slices that are neither temporal partners nor the same
        // channel. Whatever correlation exists here is not evidence of layout.
        var baseline: Float = 0
        var pairs = 0
        for c in 0 ..< 3 {
            for d in 0 ..< 3 where d != c {
                baseline += cosine(a[0..., c, 0, 0...].flattened(), a[0..., d, 1, 0...].flattened())
                pairs += 1
            }
        }
        baseline /= Float(pairs)

        print("[layout] A [channel][temporal] temporal-pair cosine = \(aScore)")
        print("[layout] B [temporal][channel] temporal-pair cosine = \(bScore)")
        print("[layout] unrelated-slice baseline                   = \(baseline)")

        // B pairs the co-adapted slices, so `temporalMajor` reorders
        // patchify's output before the projection. If this ever flips, that
        // reorder is wrong and colour will be interleaved across the temporal
        // slots again — silently, because greyscale is unaffected.
        #expect(bScore > aScore,
            "the weights no longer prefer [temporal][channel] (B=\(bScore) A=\(aScore)) — revisit temporalMajor")
        #expect(bScore > baseline,
            "the temporal pairing under B (\(bScore)) is no better than unrelated slices (\(baseline)), so this evidence does not support the reorder")
    }
}
