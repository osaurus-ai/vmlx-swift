// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation

/// A physical module a construction may or may not build.
///
/// Separate from ``ModelRuntimeRequestModality`` on purpose. A request names LANES the caller wants
/// to use; this names MODULES that have to exist for those lanes to work, and the two do not
/// correspond one-to-one:
///
///   * `.video` and `.vision` are different lanes served by the SAME tower;
///   * `.audio` maps to different modules depending on the configuration — a conformer tower for
///     Gemma 4's encoder path, a projection alone for its encoder-free unified path;
///   * the language core is required by every request and is named by no lane at all.
///
/// Conflating the two is what let `requesting: [.video]` be accepted while nothing was built: the
/// check asked "is `.video` constructible?" and the construction asked "does the set contain
/// `.vision`?", and both answers were right about different questions.
public enum ModelComponent: String, Sendable, Hashable, CaseIterable {
    /// The language model itself. Every construction builds it; no request can omit it.
    case languageCore

    /// The shared image/video tower. Serves `.vision` and `.video` alike.
    case visionTower

    /// An audio encoder tower (Gemma 4's conformer path).
    case audioTower

    /// The projection from audio features into the text embedding space. Present on both audio
    /// paths — with a tower for the conformer variant, alone for the encoder-free unified one.
    case audioProjection
}

/// What a construction request resolved to.
///
/// Two fields because the two questions have different answers, and callers need both: `requested`
/// is what the caller asked for and is what the instance reports; `components` is what gets built.
public struct ResolvedConstructionPlan: Equatable, Sendable {
    /// The lanes the caller asked for, after validation. Literal — never widened to include a lane
    /// the caller did not name, because a video request is not an image request even though both
    /// need the same tower.
    public let requested: Set<ModelRuntimeRequestModality>

    /// The modules to build. Always contains ``ModelComponent/languageCore``.
    public let components: Set<ModelComponent>

    /// What the built instance can serve. This — not ``requested`` — is what a model reports as its
    /// `modalities`, because a caller asking "what can this do?" wants capability, not a receipt.
    public let served: Set<ModelRuntimeRequestModality>

    public init(
        requested: Set<ModelRuntimeRequestModality>, components: Set<ModelComponent>,
        served: Set<ModelRuntimeRequestModality>
    ) {
        self.requested = requested
        self.components = components
        self.served = served
    }

    public func builds(_ component: ModelComponent) -> Bool { components.contains(component) }
}

/// How a family maps requested lanes onto modules.
///
/// A family declares this instead of hardcoding `modalities.contains(.vision)` at each construction
/// site, which is what made the dependency invisible and therefore wrong.
public protocol ModelComponentMapping {
    /// The lanes this configuration can serve at all.
    static func constructibleModalities(of configuration: Configuration)
        -> Set<ModelRuntimeRequestModality>

    /// The modules needed to serve `requested`. Must include `.languageCore`.
    static func components(
        for requested: Set<ModelRuntimeRequestModality>, of configuration: Configuration
    ) -> Set<ModelComponent>

    /// What the BUILT instance can actually serve, given the modules it has.
    ///
    /// Deliberately has no default implementation, and in particular no "…and always text". Which
    /// lanes a set of modules enables is a property of the family: a text LM core serves `.text`,
    /// but a speech-to-speech model's core serves audio in and audio out and no text at all.
    /// A blanket rule would describe such a model wrongly, and `modalities` is what a caller reads
    /// to decide what it may send.
    static func servedModalities(
        by components: Set<ModelComponent>, of configuration: Configuration
    ) -> Set<ModelRuntimeRequestModality>

    associatedtype Configuration
}

extension ModelComponentMapping {
    /// Validate a request against what the configuration offers, and resolve it to a plan.
    ///
    /// The only public route to a plan: a caller cannot assemble one with an empty or unsupported
    /// selection, because this is where both are refused.
    public static func resolveConstruction(
        _ configuration: Configuration, requesting: Set<ModelRuntimeRequestModality>?
    ) throws -> ResolvedConstructionPlan {
        let constructible = constructibleModalities(of: configuration)
        let resolved = try Set.resolveForConstruction(
            requested: requesting, constructible: constructible)
        var built = components(for: resolved, of: configuration)
        built.insert(.languageCore)     // never optional, never omitted by a mapping's mistake
        return ResolvedConstructionPlan(
            requested: resolved, components: built,
            served: servedModalities(by: built, of: configuration))
    }
}
