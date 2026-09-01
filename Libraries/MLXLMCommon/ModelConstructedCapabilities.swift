// Copyright © 2026 osaurus-eval contributors
// SPDX-License-Identifier: MIT
//
// The other half of `ModelRuntimeCapabilitySnapshot`.
//
// That type answers "what does this bundle DECLARE?" — read from `JangCapabilities` before
// anything is loaded, and tri-state, because a bundle can simply be silent about a capability.
// This one answers "what does this INSTANCE actually have?", and is deliberately shaped to match
// it field for field so the two read as a designed pair rather than two unrelated mechanisms.
//
// It is NOT tri-state, and that difference is the point: `.unknown` exists in the snapshot because
// a declaration can be missing. A constructed model has no such doubt — the component is either
// there or it is not.

import Foundation

/// Whether a constructed model carries a capability. The two-state counterpart to
/// ``ModelRuntimeCapabilitySupport``, which needs a third case only because a *declaration* can be
/// absent.
public enum ModelRuntimeCapabilityPresence: String, Codable, Sendable, Equatable {
    case present
    case absent

    public init(_ isPresent: Bool) { self = isPresent ? .present : .absent }
    public var isPresent: Bool { self == .present }
}

/// A module that provides runtime capability once it has been successfully constructed.
///
/// The declaration is on the INSTANCE, not the type, and it reports only what the component itself
/// supplies. A vision tower says `.vision`; it does not say `.video`, because whether frames can be
/// fed depends on the MODEL's configuration (video token ids), not on the encoder. The model unions
/// what its components provide and then adds whatever its own configuration enables on top.
///
/// The reason to register rather than compute: a component that failed to build cannot register, so
/// the report describes what EXISTS rather than what was planned.
public protocol ModelCapabilityProviding {
    /// What this component supplies on its own, now that it exists.
    var providedModalities: Set<ModelRuntimeRequestModality> { get }
}

/// What a constructed model can actually do.
///
/// Mirrors ``ModelRuntimeCapabilitySnapshot``'s field-per-lane shape so a caller can hold both and
/// compare them directly: the snapshot is the claim, this is the delivery. A lane that is
/// `.supported` in the snapshot but `.absent` here is not a bug — it is the normal result of a
/// caller narrowing construction with `requesting:`.
public struct ModelRuntimeConstructedCapabilities: Codable, Sendable, Equatable {
    public let text: ModelRuntimeCapabilityPresence
    public let vision: ModelRuntimeCapabilityPresence
    public let video: ModelRuntimeCapabilityPresence
    public let audio: ModelRuntimeCapabilityPresence
    public let tools: ModelRuntimeCapabilityPresence
    public let reasoning: ModelRuntimeCapabilityPresence
    public let nativeMTP: ModelRuntimeCapabilityPresence

    /// The same information as a set, which is what most call sites want.
    public let modalities: Set<ModelRuntimeRequestModality>

    public init(_ modalities: Set<ModelRuntimeRequestModality>) {
        self.modalities = modalities
        self.text = .init(modalities.contains(.text))
        self.vision = .init(modalities.contains(.vision))
        self.video = .init(modalities.contains(.video))
        self.audio = .init(modalities.contains(.audio))
        self.tools = .init(modalities.contains(.tools))
        self.reasoning = .init(modalities.contains(.reasoning))
        self.nativeMTP = .init(modalities.contains(.nativeMTP))
    }
}

extension ModalityBearing {
    /// This instance's capabilities in the same shape the bundle's declaration uses.
    public var constructedCapabilities: ModelRuntimeConstructedCapabilities {
        ModelRuntimeConstructedCapabilities(modalities)
    }
}

extension Set where Element == ModelRuntimeRequestModality {
    /// Union of what a set of constructed components provides. `nil` components — the ones that
    /// were not built — contribute nothing, which is the whole reason this is collected rather
    /// than computed from the plan.
    public static func provided(by components: [(any ModelCapabilityProviding)?])
        -> Set<ModelRuntimeRequestModality>
    {
        components.compactMap { $0 }.reduce(into: []) { $0.formUnion($1.providedModalities) }
    }
}
