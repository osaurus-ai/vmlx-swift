import Cmlx
import Foundation
import MLX
import Testing

@testable import MLXLMCommon

/// Regression coverage for `NormConventionResolver`, the deterministic replacement for the old
/// per-process-random `(1 + weight)` RMSNorm shift heuristic. The key invariant these tests pin:
/// an UNRECOGNIZED `norm_convention` declaration must NOT silently disable the shift — it defers to
/// the order-independent vote, so a converter typo can't strand a raw bundle's norms unshifted.
struct NormConventionResolverTests {
    // Swift Testing does not always launch inside the XCTest resource bundle.
    private static let metalFixture: Void = {
        if let path = ProcessInfo.processInfo.environment["VMLX_TEST_METALLIB_PATH"] {
            precondition(FileManager.default.isReadableFile(atPath: path))
            precondition(mlx_metal_set_metallib_path(path) == 0)
        }
    }()

    init() { _ = Self.metalFixture }

    private let probes = [".input_layernorm.weight", ".post_attention_layernorm.weight"]

    /// Raw storage: norms are the deviation-from-1, so mean ≈ 0.
    private func rawWeights() -> [String: MLXArray] {
        [
            "model.layers.0.input_layernorm.weight": MLXArray(converting: [0.01, -0.02, 0.03, 0.0]),
            "model.layers.0.post_attention_layernorm.weight": MLXArray(
                converting: [0.0, 0.04, -0.01, 0.02]),
            "model.layers.1.input_layernorm.weight": MLXArray(converting: [0.02, 0.0, -0.03, 0.01]),
        ]
    }

    /// Already-shifted storage: norms are the applied weights, so mean ≈ 1.
    private func shiftedWeights() -> [String: MLXArray] {
        [
            "model.layers.0.input_layernorm.weight": MLXArray(converting: [1.01, 0.98, 1.03, 1.0]),
            "model.layers.0.post_attention_layernorm.weight": MLXArray(
                converting: [1.0, 1.04, 0.99, 1.02]),
            "model.layers.1.input_layernorm.weight": MLXArray(converting: [0.97, 1.0, 1.03, 1.01]),
        ]
    }

    @Test func recognizedPlusOneMarkerForcesShift() {
        // A recognized marker is authoritative even against already-shifted-looking weights.
        #expect(
            NormConventionResolver.shouldApplyPlusOneShift(
                metadataConvention: "mlx_plus_one", configConvention: nil, declaredConvention: nil,
                weights: shiftedWeights(), probeSuffixes: probes) == true)
    }

    @Test func rawQwenAWQDeclarationOverridesShiftedLookingMean() {
        // AWQ preserves raw-deviation storage but changes its magnitude:
        // (1 + 0.3) / 0.5 - 1 = 1.6. A mean threshold mistakes this for
        // an already-shifted tensor and produces the wrong normalization.
        let rawFolded = (MLXArray(converting: [0.3, 0.4]) + 1) / 0.5 - 1
        let weights = ["model.layers.0.post_attention_layernorm.weight": rawFolded]
        #expect(NormConventionResolver.weightsAppearUnshifted(weights, probeSuffixes: probes) == false)
        #expect(NormConventionResolver.shouldApplyPlusOneShift(
            metadataConvention: "qwen3_5_raw_weight", configConvention: nil,
            declaredConvention: nil, weights: weights, probeSuffixes: probes))
        #expect(NormConventionResolver.shouldApplyPlusOneShift(
            metadataConvention: nil, configConvention: "qwen3_5_raw_weight",
            declaredConvention: nil, weights: weights, probeSuffixes: probes))
    }

    @Test func noDeclarationVotesOnWeights() {
        #expect(
            NormConventionResolver.shouldApplyPlusOneShift(
                metadataConvention: nil, configConvention: nil, declaredConvention: nil,
                weights: rawWeights(), probeSuffixes: probes) == true)  // raw → shift
        #expect(
            NormConventionResolver.shouldApplyPlusOneShift(
                metadataConvention: nil, configConvention: nil, declaredConvention: nil,
                weights: shiftedWeights(), probeSuffixes: probes) == false)  // shifted → no-shift
    }

    /// THE REGRESSION: an unrecognized declaration must fall back to the vote, not silently no-shift.
    /// Before the fix, a raw bundle carrying e.g. `norm_convention: "rms_norm"` returned false and
    /// loaded with unshifted norms → garbage output.
    @Test func unrecognizedDeclarationFallsBackToVote() {
        #expect(
            NormConventionResolver.shouldApplyPlusOneShift(
                metadataConvention: "rms_norm", configConvention: nil, declaredConvention: nil,
                weights: rawWeights(), probeSuffixes: probes) == true)  // raw → shift (was: false)
        #expect(
            NormConventionResolver.shouldApplyPlusOneShift(
                metadataConvention: "qwen3_5_language_mlx_plus_ONE_typo", configConvention: nil,
                declaredConvention: nil,
                weights: rawWeights(), probeSuffixes: probes) == true)  // typo'd marker → vote → shift
        #expect(
            NormConventionResolver.shouldApplyPlusOneShift(
                metadataConvention: "rms_norm", configConvention: nil, declaredConvention: nil,
                weights: shiftedWeights(), probeSuffixes: probes) == false)  // shifted → no-shift
    }

    /// A recognized marker at a lower precedence still wins when the higher one is unrecognized.
    @Test func recognizedMarkerWinsThroughUnrecognizedHigherPrecedence() {
        #expect(
            NormConventionResolver.shouldApplyPlusOneShift(
                metadataConvention: "garbage", configConvention: "mlx_plus_one",
                declaredConvention: nil,
                weights: shiftedWeights(), probeSuffixes: probes) == true)
    }

    @Test func noProbeFallsBackToProvidedDefault() {
        #expect(
            NormConventionResolver.shouldApplyPlusOneShift(
                metadataConvention: nil, configConvention: nil, declaredConvention: nil,
                weights: ["unrelated.weight": MLXArray(converting: [1.0, 2.0])],
                probeSuffixes: probes, fallbackWhenNoProbe: { true }) == true)
    }
}
