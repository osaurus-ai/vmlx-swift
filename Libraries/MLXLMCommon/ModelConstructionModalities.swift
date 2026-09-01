// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation

/// Construction-time counterpart to `ModelRuntimeCapabilitySnapshot.validate(request:)`.
///
/// Two different questions are being asked about the same bundle, and they are deliberately not
/// merged:
///
/// - `ModelRuntimeCapabilitySnapshot` answers *"is this model trained for a modality"*. It merges
///   an explicit stamp, a structural probe, and coarse `modality` tokens, and it is honest about
///   the gap between them — hence `.unknown`. It gates a generation request.
/// - This file answers *"which towers can this configuration instantiate, and which of those does
///   the caller want built"*. It is structural, local to the config, and has no third state: a
///   config either carries a vision section or it does not.
///
/// The two can legitimately disagree — a bundle may declare `supportsVision` while shipping a
/// config with no vision section, which is a broken conversion rather than a policy question — so
/// keeping them separate preserves that signal instead of averaging it away. They share the
/// `ModelRuntimeRequestModality` vocabulary so no caller has to translate between them.
///
/// Nothing here implies `.text`. A model whose only lanes are audio-in and audio-out is unusual but
/// not incoherent, and a default that quietly adds text would misdescribe it.
/// A consequence worth stating, because it changes what bundles are worth SHIPPING.
///
/// Once a multimodal bundle can be constructed text-only, a text-only DERIVATIVE of that same
/// checkpoint has almost no reason to exist. It saves disk only for someone who never once wants
/// the vision lane; anyone who alternates now stores two copies of the same text tower to avoid
/// allocating a tower they could simply not build. The saving is small either way — on the 30B
/// Muse Glimmer the vision half is 1.92B of 29.8B parameters, about 6%.
///
/// The distinction that survives is between a derivative and a RELEASE. A natively text-only
/// model in the same family is not the multimodal one with the tower removed — it is separately
/// trained, often at a different parameter count, and its weights are its own. Those still need
/// the text-only path, and always will.
///
/// Which of the two the dual registrations actually serve is an empirical question, and at the
/// time of writing the local answer is stark: across 63 bundles in the nine families registered
/// in BOTH factories, every single one carries a `vision_config` AND a processor config. Not one
/// is a text-only release. So on this machine those LLM-side entries are never reached by the
/// routed loader — `loadModelContainer` tries the VLM factory first and it succeeds every time —
/// and they serve only callers who reach for `LLMModelFactory.shared.load` directly to express
/// "text only". That intent is exactly what the `requesting:` parameter here makes explicit,
/// which is a better place for it than a second registration.
///
public struct UnconstructibleModalities: Error, CustomStringConvertible {
    public let requested: Set<ModelRuntimeRequestModality>
    public let constructible: Set<ModelRuntimeRequestModality>

    public init(
        requested: Set<ModelRuntimeRequestModality>,
        constructible: Set<ModelRuntimeRequestModality>
    ) {
        self.requested = requested
        self.constructible = constructible
    }

    /// The part of the request this configuration cannot satisfy.
    public var excess: Set<ModelRuntimeRequestModality> {
        requested.subtracting(constructible)
    }

    public var description: String {
        "this configuration cannot build \(excess.modalityDescription) "
            + "(requested \(requested.modalityDescription); constructible \(constructible.modalityDescription))"
    }

}

/// An empty selection would build a model with no lanes at all — loadable, and useless.
public struct EmptyModalitySelection: Error, CustomStringConvertible {
    public let source: String

    public init(_ source: String) {
        self.source = source
    }

    public var description: String {
        "\(source) named no modalities; a model with no lanes cannot be constructed"
    }
}

extension Set where Element == ModelRuntimeRequestModality {
    /// Stable, readable rendering in the canonical modality order.
    public var modalityDescription: String {
        isEmpty
            ? "nothing"
            : ModelRuntimeCapabilityRequest(modalities: self)
                .sortedModalities.map(\.rawValue).joined(separator: ", ")
    }

    /// Narrow what a configuration can build down to what the caller asked for.
    ///
    /// - Parameters:
    ///   - requested: the caller's subset, or `nil` for "everything this configuration offers".
    ///   - constructible: what the configuration structurally permits.
    /// - Returns: exactly the lanes to instantiate.
    /// - Throws: `UnconstructibleModalities` if the request exceeds what can be built — refusing
    ///   here rather than yielding a model that fails later inside `prepare()`, where the cause is
    ///   no longer visible; `EmptyModalitySelection` if either side names nothing.
    public static func resolveForConstruction(
        requested: Set<ModelRuntimeRequestModality>?,
        constructible: Set<ModelRuntimeRequestModality>
    ) throws -> Set<ModelRuntimeRequestModality> {
        guard !constructible.isEmpty else {
            throw EmptyModalitySelection("this configuration")
        }
        guard let requested else { return constructible }
        guard !requested.isEmpty else {
            throw EmptyModalitySelection("the request")
        }
        guard requested.isSubset(of: constructible) else {
            throw UnconstructibleModalities(requested: requested, constructible: constructible)
        }
        return requested
    }
}

/// A model that reports which lanes it actually carries, as opposed to which its bundle advertises.
public protocol ModalityBearing {
    /// What this instance was built with — never wider than its configuration allowed.
    var modalities: Set<ModelRuntimeRequestModality> { get }
}
