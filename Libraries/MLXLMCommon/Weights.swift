// Copyright © 2025 Apple Inc.

import Foundation
import MLX
import os

private let weightsLogger = Logger(subsystem: "vmlx", category: "Weights")

/// Helpers for normalizing a checkpoint's weight keys before they are bound to modules.
public enum Weights {

    /// Strips the VLM-style `language_model.` prefix that some conversions emit onto text-only bodies.
    ///
    /// The prefix is a **converter artifact, not a checkpoint convention**: a conversion that wraps a
    /// text-only model in the VLM key layout leaves the body — or, in the wild, just the output head —
    /// under `language_model.*`, while the text model binds its modules at the top level. Loading such
    /// a bundle otherwise fails with `Unhandled keys [language_model]`. Being tolerant here is
    /// deliberate: refusing to load a bundle someone already downloaded is worse than absorbing the
    /// artifact.
    ///
    /// Both spellings are handled: `model.language_model.foo` → `model.foo`, and
    /// `language_model.foo` → `foo`.
    ///
    /// - Parameters:
    ///   - weights: the checkpoint keys, as loaded.
    ///   - only: when non-`nil`, only keys whose **stripped** form begins with one of these strings are
    ///     rewritten; every other key passes through untouched. Use it when a checkpoint is known to
    ///     misplace exactly one module (e.g. `only: ["lm_head."]`) so the strip cannot silently absorb
    ///     an unrelated, genuine `language_model.*` key. `nil` strips wherever the prefix appears.
    /// - Returns: the weights with the prefix removed.
    ///
    /// ## Collisions are resolved deterministically
    ///
    /// A mixed-provenance re-bake can carry **both** spellings of a key — a converter writes the
    /// prefixed head while something upstream leaves the unprefixed one behind. Both then want the same
    /// destination. Resolving that by last-write-wins would make the bind depend on `Dictionary`'s
    /// iteration order, which is **seeded per process**: the model would silently load a different
    /// tensor from run to run. That is the worst failure shape available — it loads fine, generates
    /// fine-ish, and is wrong some fraction of the time.
    ///
    /// So: an unprefixed key already sitting at the destination always wins, and any remaining tie is
    /// broken by lexicographically smallest source key. The choice is arbitrary but *fixed*; the point
    /// is that it never varies between runs. Losing keys are logged.
    public static func stripLanguageModelPrefix(
        _ weights: [String: MLXArray], only: [String]? = nil
    ) -> [String: MLXArray] {
        // Pass 1: map every key to the destination it wants. Grouping first (rather than writing as we
        // go) is what makes collisions visible instead of order-dependent.
        var sourcesByDestination = [String: [String]](minimumCapacity: weights.count)
        for key in weights.keys {
            sourcesByDestination[strippedKey(key, only: only) ?? key, default: []].append(key)
        }

        // Pass 2: bind, resolving any contested destination by a rule that does not depend on
        // iteration order.
        var result = [String: MLXArray](minimumCapacity: sourcesByDestination.count)
        for (destination, sources) in sourcesByDestination {
            guard sources.count > 1 else {
                result[destination] = weights[sources[0]]
                continue
            }
            let winner = sources.contains(destination) ? destination : (sources.min() ?? destination)
            weightsLogger.warning(
                """
                checkpoint carries \(sources.count, privacy: .public) spellings of \
                \(destination, privacy: .public) (\(sources.sorted().joined(separator: ", "), privacy: .public)); \
                binding \(winner, privacy: .public) and dropping the rest. This bundle is malformed — \
                the weights should be re-converted.
                """)
            result[destination] = weights[winner]
        }
        return result
    }

    /// The destination `key` would be rewritten to, or `nil` if it carries no prefix (or `only`
    /// excludes it) and should pass through untouched.
    private static func strippedKey(_ key: String, only: [String]?) -> String? {
        let stripped: String
        if key.hasPrefix("model.language_model.") {
            stripped = "model." + key.dropFirst("model.language_model.".count)
        } else if key.hasPrefix("language_model.") {
            stripped = String(key.dropFirst("language_model.".count))
        } else {
            return nil
        }
        guard let only else { return stripped }
        return only.contains(where: stripped.hasPrefix) ? stripped : nil
    }
}
