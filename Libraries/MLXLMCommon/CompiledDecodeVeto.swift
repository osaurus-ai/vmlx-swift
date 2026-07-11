// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation

/// Per-instance veto over compiled decode.
///
/// The engines (BatchEngine single-slot promotion, TokenIterator direct
/// compile) keep a name-based denylist for model families whose forward pass
/// is known to diverge under the compiled trace. That list is necessarily
/// coarse: it keys off type/bundle names, so it cannot distinguish two packs
/// of the same family where only one uses an untraceable path (Hy3 preview
/// TurboQuant streaming experts vs the official affine `SwitchGLU` packs).
///
/// A model that conforms decides for itself, per instance, from its own
/// configuration. When a model conforms, its answer is authoritative and the
/// name-based fallback is not consulted.
public protocol CompiledDecodeVetoing {
    /// `true` when this instance's forward pass must not be traced by
    /// `compile()` (e.g. it streams weights on the CPU mid-forward or
    /// otherwise executes work the tracer cannot record).
    var vetoesCompiledDecode: Bool { get }
}
