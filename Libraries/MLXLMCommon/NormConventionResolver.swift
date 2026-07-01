// Architecture-agnostic resolution of the RMSNorm "(1 + weight)" shift convention.
//
// Several architectures (qwen3.5, qwen3-next, gemma-family, …) store RMSNorm weights as the
// deviation from 1 and apply `(1 + weight)` at load. Whether a given bundle needs that shift is
// authoritative information that should be *declared* — by the bundle (safetensors `metadata` or
// `config.json` `norm_convention`) or by the architecture itself — never guessed from the weights.
//
// This type centralizes that resolution so every model shares one correct implementation instead of
// hand-rolling its own. The hand-rolled version this replaces (in qwen3.5) returned on the FIRST
// probe norm in (per-process randomized) Swift `Dictionary` iteration order, which silently
// degenerated ~7.5% of loads when a few raw norms legitimately had mean > 0.5. New `(1 + weight)`
// architectures should adopt `shouldApplyPlusOneShift(...)` rather than re-implementing this.

import Foundation
import MLX

public enum NormConventionResolver {

    /// Accepted markers for the "(1 + weight)" RMSNorm convention.
    public static func usesPlusOne(_ value: String?) -> Bool {
        switch value?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
        case "qwen3_5_language_mlx_plus_one", "qwen35_language_mlx_plus_one", "mlx_plus_one":
            return true
        default:
            return false
        }
    }

    /// Order-independent detection of whether 1-D RMSNorm weights look UNSHIFTED (raw, mean ≈ 0 →
    /// need `+1`) vs already shifted (mean ≈ 1). Decides by MAJORITY VOTE over all probe norms, so it
    /// never depends on `Dictionary` iteration order. Returns `nil` when no probe norm is found.
    public static func weightsAppearUnshifted(
        _ weights: [String: MLXArray],
        probeSuffixes: [String],
        excluding isExcluded: (String) -> Bool = { _ in false }
    ) -> Bool? {
        var below = 0, above = 0
        for (key, value) in weights where value.ndim == 1 {
            guard !isExcluded(key),
                probeSuffixes.contains(where: { key.hasSuffix($0) }) else { continue }
            let mean = value.asType(.float32).mean().item(Float.self)
            if mean < 0.5 { below += 1 } else if mean > 0.5 { above += 1 }
        }
        if below + above == 0 { return nil }
        return below >= above
    }

    /// Resolve whether to apply the `(1 + weight)` shift, with deterministic precedence:
    /// per-bundle declaration → architecture declaration → order-independent vote.
    ///
    /// A DECLARATION is authoritative — it states the storage state outright and short-circuits the
    /// vote. A per-bundle declaration (safetensors metadata → `config.json`) wins first; then an
    /// architecture-level declaration (`declaredConvention`). Crucially, an architecture may declare
    /// only when ALL its bundles share one storage state. An architecture that ships the SAME class
    /// both raw (norm mean ≈ 0, deviation-from-1 → needs +1) AND already-shifted (mean ≈ 1 → must NOT
    /// be shifted again) — e.g. qwen3.5, where JangQ stores raw and MXFP4 stores pre-shifted — can
    /// make no truthful class-level claim, so it must declare `nil` and let the per-bundle config or
    /// the vote decide. The order-independent majority vote is the per-bundle measurement used when
    /// nothing is declared; it discriminates the cleanly bimodal signal (≈0 vs ≈1) regardless of
    /// `Dictionary` iteration order.
    public static func shouldApplyPlusOneShift(
        metadataConvention: String?,
        configConvention: String?,
        declaredConvention: String?,
        weights: [String: MLXArray],
        probeSuffixes: [String],
        excluding isExcluded: (String) -> Bool = { _ in false },
        fallbackWhenNoProbe: () -> Bool = { false }
    ) -> Bool {
        // Any declaration is authoritative and overrides the vote: per-bundle metadata/config first
        // (states THIS bundle's storage state), then an architecture-level declaration (valid only
        // for architectures with a uniform storage state across all their bundles).
        if let explicit = metadataConvention ?? configConvention ?? declaredConvention {
            return usesPlusOne(explicit)
        }
        // Nothing declared (the qwen3.5 reality: one class, mixed storage across bundles). Measure
        // THIS bundle, order-independently.
        return weightsAppearUnshifted(weights, probeSuffixes: probeSuffixes, excluding: isExcluded)
            ?? fallbackWhenNoProbe()
    }
}
