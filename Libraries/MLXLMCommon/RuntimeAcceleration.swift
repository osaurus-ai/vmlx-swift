// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation

/// Runtime accelerator requested by a caller or by `VMLX_ACCELERATOR`.
///
/// This is a selection contract, not an implementation shortcut. The current
/// MLX backend exposes CPU/GPU devices, while public Neural Engine execution is
/// reached through validated Core ML subgraphs. `ane-coreml` therefore fails
/// closed unless the call site supplies a manifest for a tested Core ML island.
public enum AccelerationMode: Sendable, Equatable, CustomStringConvertible {
    /// Current MLX + Metal runtime.
    case metal

    /// Use a validated accelerator island when one is available; otherwise
    /// keep the Metal path.
    case auto

    /// Require a validated Core ML island. If none exists for the target
    /// runtime surface, the request must fail rather than silently falling
    /// back to Metal.
    case aneCoreML

    /// A caller or environment value that did not match the public flag
    /// contract. Stored explicitly so generation can fail closed.
    case invalid(String)

    public static let environmentVariable = "VMLX_ACCELERATOR"

    /// The pre-rename spelling, still honoured. Kept as its own symbol rather than left as the
    /// value of `environmentVariable`: a consumer reading that symbol to build a settings UI or
    /// documentation would otherwise publish the legacy name outward.
    public static let legacyEnvironmentVariable = "VMLINUX_ACCELERATOR"

    public init(flagValue rawValue: String) {
        let normalized = rawValue.trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        switch normalized {
        case "", "metal", "gpu", "mlx", "default":
            self = .metal
        case "auto":
            self = .auto
        case "ane-coreml", "ane_coreml", "coreml-ane", "coreml_ane":
            self = .aneCoreML
        default:
            self = .invalid(rawValue)
        }
    }

    public var flagValue: String {
        switch self {
        case .metal: "metal"
        case .auto: "auto"
        case .aneCoreML: "ane-coreml"
        case .invalid(let raw): raw
        }
    }

    public var description: String { flagValue }
}

/// Runtime surface being accelerated.
public enum AccelerationTarget: String, Sendable, Equatable {
    /// Autoregressive text decode and prefill owned by MLX KV/cache state.
    case textDecode = "text-decode"

    /// Bounded media encoder island, e.g. RADIO vision or Parakeet audio.
    case mediaEncoder = "media-encoder"
}

/// Effective backend selected for a runtime surface.
public enum AccelerationDecision: Sendable, Equatable {
    case metal(reason: String)
    case coreMLANE(manifestID: String)
}

public enum AccelerationError: Error, LocalizedError, Sendable, Equatable {
    case invalidMode(String)
    case coreMLUnavailable
    case noValidatedCoreMLIsland(mode: AccelerationMode, target: AccelerationTarget)

    public var errorDescription: String? {
        switch self {
        case .invalidMode(let value):
            return
                "Invalid \(AccelerationMode.environmentVariable)=\(value). Supported values: metal, auto, ane-coreml."
        case .coreMLUnavailable:
            return "Core ML is unavailable on this platform; ANE acceleration cannot be selected."
        case .noValidatedCoreMLIsland(let mode, let target):
            return
                "\(mode.flagValue) was requested for \(target.rawValue), but no validated Core ML island is registered for that runtime surface."
        }
    }
}

public enum AccelerationRuntime {
    /// Reads the current process environment using the public flag name.
    public static func requestedMode(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> AccelerationMode {
        // Both, in that order. This read went through the symbol alone, so flipping the
        // constant without widening it here would have silently dropped every existing
        // VMLINUX_ACCELERATOR user.
        guard
            let raw = environment[AccelerationMode.environmentVariable]
                ?? environment[AccelerationMode.legacyEnvironmentVariable]
        else {
            return .metal
        }
        return AccelerationMode(flagValue: raw)
    }

    /// Resolves a requested mode to an effective runtime backend.
    ///
    /// `validatedCoreMLIslandID` is intentionally explicit. Callers must pass a
    /// manifest identifier only after parity and benchmark gates have passed for
    /// the active model family, preprocessing path, cache semantics, and device.
    public static func resolve(
        _ mode: AccelerationMode,
        target: AccelerationTarget,
        validatedCoreMLIslandID: String? = nil
    ) throws -> AccelerationDecision {
        switch mode {
        case .metal:
            return .metal(reason: "explicit-metal")
        case .auto:
            guard let manifestID = validatedCoreMLIslandID else {
                return .metal(reason: "no-validated-coreml-island")
            }
            try requireCoreML()
            return .coreMLANE(manifestID: manifestID)
        case .aneCoreML:
            guard let manifestID = validatedCoreMLIslandID else {
                throw AccelerationError.noValidatedCoreMLIsland(
                    mode: mode, target: target)
            }
            try requireCoreML()
            return .coreMLANE(manifestID: manifestID)
        case .invalid(let raw):
            throw AccelerationError.invalidMode(raw)
        }
    }

    /// Current text generation path is still MLX/Metal. Keep this helper at
    /// call sites so future text islands have a single place to attach a
    /// manifest after validation.
    public static func resolveTextDecode(
        _ mode: AccelerationMode
    ) throws -> AccelerationDecision {
        try resolve(mode, target: .textDecode)
    }

    private static func requireCoreML() throws {
        #if canImport(CoreML)
        return
        #else
        throw AccelerationError.coreMLUnavailable
        #endif
    }
}
