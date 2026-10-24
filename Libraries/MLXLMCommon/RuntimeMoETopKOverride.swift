// Copyright © 2026 osaurus.

import Foundation

/// Opt-in runtime override for routed-MoE token top-k.
///
/// This helper intentionally only lowers an already-configured routed-expert
/// top-k. It never raises low-topology models such as ZAYA top-1, and it does
/// not apply to sampler `top_k`, group routing, or speculative decoding.
public enum RuntimeMoETopKOverride {
    public static let environmentVariable = "VMLX_MOE_TOPK_OVERRIDE"
    public static let legacyEnvironmentVariable = "VMLINUX_MOE_TOPK_OVERRIDE"

    public enum Reason: Equatable, Sendable {
        case unset
        case lowered
        case invalidCurrentTopK
        case invalidRequestedTopK
        case requestedTopKAlreadySatisfied
        case requestedTopKAboveCurrent
    }

    public struct Decision: Equatable, Sendable {
        public let modelType: String
        public let field: String
        public let originalTopK: Int
        public let requestedTopK: Int?
        public let effectiveTopK: Int
        public let reason: Reason

        public var applied: Bool { reason == .lowered }
    }

    public static func effectiveTopK(
        currentTopK: Int,
        modelType: String,
        field: String,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Int {
        let decision = resolve(
            currentTopK: currentTopK,
            modelType: modelType,
            field: field,
            environment: environment)
        if decision.applied && environment == ProcessInfo.processInfo.environment {
            emitDiagnostic(decision)
        }
        return decision.effectiveTopK
    }

    public static func resolve(
        currentTopK: Int,
        modelType: String,
        field: String,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Decision {
        guard currentTopK > 0 else {
            return Decision(
                modelType: modelType,
                field: field,
                originalTopK: currentTopK,
                requestedTopK: nil,
                effectiveTopK: currentTopK,
                reason: .invalidCurrentTopK)
        }

        let raw = environment[environmentVariable] ?? environment[legacyEnvironmentVariable]
        guard let raw, !raw.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return Decision(
                modelType: modelType,
                field: field,
                originalTopK: currentTopK,
                requestedTopK: nil,
                effectiveTopK: currentTopK,
                reason: .unset)
        }

        guard let requested = Int(raw.trimmingCharacters(in: .whitespacesAndNewlines)),
              requested > 0
        else {
            return Decision(
                modelType: modelType,
                field: field,
                originalTopK: currentTopK,
                requestedTopK: nil,
                effectiveTopK: currentTopK,
                reason: .invalidRequestedTopK)
        }

        if requested < currentTopK {
            return Decision(
                modelType: modelType,
                field: field,
                originalTopK: currentTopK,
                requestedTopK: requested,
                effectiveTopK: requested,
                reason: .lowered)
        }

        let reason: Reason =
            requested == currentTopK ? .requestedTopKAlreadySatisfied : .requestedTopKAboveCurrent
        return Decision(
            modelType: modelType,
            field: field,
            originalTopK: currentTopK,
            requestedTopK: requested,
            effectiveTopK: currentTopK,
            reason: reason)
    }

    /// Cache-key discriminator for opt-in routing changes.
    ///
    /// The actual top-k lowering happens at model-config decode time, while
    /// cache coordinators are keyed later from ``ModelConfiguration.name``.
    /// Mix the requested value into cache model keys whenever a syntactically
    /// valid override is present so L2/paged cache entries created under
    /// different routing regimes cannot be reused across process restarts.
    public static func cacheKeyComponent(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> String? {
        let raw = environment[environmentVariable] ?? environment[legacyEnvironmentVariable]
        guard let raw,
              let requested = Int(raw.trimmingCharacters(in: .whitespacesAndNewlines)),
              requested > 0
        else {
            return nil
        }
        return "moeTopK=\(requested)"
    }

    public static func cacheScopedModelKey(
        _ modelKey: String,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> String {
        guard let component = cacheKeyComponent(environment: environment) else {
            return modelKey
        }
        return "\(modelKey)|\(component)"
    }

    private static func emitDiagnostic(_ decision: Decision) {
        let line =
            "MoE top-k override: \(decision.modelType) \(decision.field) "
            + "\(decision.originalTopK) -> \(decision.effectiveTopK)\n"
        if let data = line.data(using: .utf8) {
            RuntimeDiagnostics.write(data)
        }
    }
}
