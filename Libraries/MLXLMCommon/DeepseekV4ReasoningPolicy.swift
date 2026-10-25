// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation

/// Runtime policy for DSV4 public reasoning controls.
///
/// Public reasoning controls are passed through to the chat template after
/// spelling/case normalization. DSV4-0731 accepts exactly the canonical
/// `low`, `high`, and `max` effort names; unsupported values fail at the
/// processor boundary instead of silently selecting another rail.
public enum DeepseekV4ReasoningPolicy {
    /// Deprecated compatibility key. `reasoning_effort=max` now passes through
    /// without an opt-in environment variable.
    public static let rawMaxEnvironmentKey = "VMLX_DSV4_RAW_MAX"
    /// The pre-rename spelling, still honoured. See `RuntimeEnvironment`.
    public static let legacyRawMaxEnvironmentKey = "VMLINUX_DSV4_RAW_MAX"

    /// Deprecated compatibility key. Public request/template controls must win;
    /// process environment must not silently force direct-answer rails.
    public static let forceDirectRailEnvironmentKey = "VMLX_DSV4_FORCE_DIRECT_RAIL"
    /// The pre-rename spelling, still honoured. See `RuntimeEnvironment`.
    public static let legacyForceDirectRailEnvironmentKey = "VMLINUX_DSV4_FORCE_DIRECT_RAIL"

    public static func isDeepseekV4(modelType: String?) -> Bool {
        guard let modelType else { return false }
        let normalized =
            modelType
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
            .replacingOccurrences(of: "-", with: "_")
        return normalized == "deepseek_v4"
            || normalized.hasPrefix("deepseek_v4_")
            || normalized == "deepseekv4"
            || normalized.hasPrefix("deepseekv4_")
    }

    /// DeepSeek-V4.1 (`deepseek_v41`, text decoder `deepseek_v41_text`). Kept
    /// separate from ``isDeepseekV4(modelType:)`` so the 0731 policy is unchanged.
    public static func isDeepseekV41(modelType: String?) -> Bool {
        DeepseekV4ChatFamily(modelType: modelType) == .deepseekV41
    }

    /// DeepSeek-V4.1 effort vocabulary: the host ladder (weakest → strongest)
    /// mapped onto the numeric 1-100 budget by the V4.1 encoder.
    public static let deepseekV41Efforts = ["minimal", "low", "medium", "high", "xhigh", "max"]

    /// Normalize a V4.1 effort: a ladder name (`none` ≡ `minimal`) or a 1-100
    /// number (Int, Double or numeric string; out-of-range values are clamped
    /// by the encoder). Returns nil for anything else.
    public static func normalizedDeepseekV41ReasoningEffort(
        _ value: (any Sendable)?
    ) -> (any Sendable)? {
        if let i = value as? Int { return i }
        if let d = value as? Double, d.isFinite { return Int(d.rounded()) }
        guard let effort = normalizedString(value) else { return nil }
        if let i = Int(effort) { return i }
        if effort == "none" { return "minimal" }
        return deepseekV41Efforts.contains(effort) ? effort : nil
    }

    public static func rawMaxEnabled(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Bool {
        truthy(environment[rawMaxEnvironmentKey] ?? environment[legacyRawMaxEnvironmentKey])
    }

    @available(
        *, deprecated, message: "Do not use process env to override explicit reasoning controls."
    )
    public static func forceDirectRailEnabled(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Bool {
        truthy(environment[forceDirectRailEnvironmentKey]
            ?? environment[legacyForceDirectRailEnvironmentKey])
    }

    public static func normalizedReasoningEffort(
        _ value: (any Sendable)?,
        rawMaxEnabled: Bool = rawMaxEnabled()
    ) -> String? {
        guard let effort = normalizedString(value) else { return nil }
        switch effort {
        case "low", "high", "max":
            return effort
        default:
            return nil
        }
    }

    @available(
        *, deprecated,
        message:
            "DSV4 reasoning_effort accepts only low/high/max; use enable_thinking=false for the direct rail."
    )
    public static func isDirectRailEffort(_ value: (any Sendable)?) -> Bool {
        guard let effort = normalizedString(value) else { return false }
        switch effort {
        case "instruct", "none", "no_think", "nothink", "off", "false":
            return true
        default:
            return false
        }
    }

    public static func normalizedAdditionalContext(
        _ context: [String: any Sendable]?,
        modelType: String?,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> [String: any Sendable]? {
        if isDeepseekV41(modelType: modelType) {
            // DeepSeek-V4.1: same explicit-mode rule as 0731, with the V4.1
            // numeric/ladder effort vocabulary.
            var normalized = context ?? [:]
            if let rawEffort = normalized["reasoning_effort"] {
                guard let canonical = normalizedDeepseekV41ReasoningEffort(rawEffort) else {
                    throw DeepseekV4ReasoningPolicyError.invalidReasoningEffort(
                        String(describing: rawEffort))
                }
                if normalized["enable_thinking"] as? Bool == false {
                    normalized.removeValue(forKey: "reasoning_effort")
                } else {
                    normalized["enable_thinking"] = true
                    normalized["reasoning_effort"] = canonical
                }
            }
            return normalized.isEmpty ? nil : normalized
        }
        guard isDeepseekV4(modelType: modelType) else { return context }

        var normalized = context ?? [:]

        let rawEffort = normalized["reasoning_effort"]
        let effort: String?
        if let rawEffort {
            guard let canonical = normalizedReasoningEffort(rawEffort) else {
                throw DeepseekV4ReasoningPolicyError.invalidReasoningEffort(
                    String(describing: rawEffort)
                )
            }
            effort = canonical
        } else {
            effort = nil
        }

        // The explicit mode switch owns chat-vs-thinking selection. A valid
        // effort may refine thinking mode, but must never turn a caller's
        // explicit `false` back on.
        if normalized["enable_thinking"] as? Bool == false {
            normalized.removeValue(forKey: "reasoning_effort")
        } else if let effort {
            normalized["enable_thinking"] = true
            normalized["reasoning_effort"] = effort
        }

        return normalized.isEmpty ? nil : normalized
    }

    private static func normalizedString(_ value: (any Sendable)?) -> String? {
        if let value = value as? String {
            let normalized =
                value
                .trimmingCharacters(in: .whitespacesAndNewlines)
                .lowercased()
            return normalized.isEmpty ? nil : normalized
        }
        return nil
    }

    private static func truthy(_ value: String?) -> Bool {
        guard let value else { return false }
        switch value.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
        case "1", "true", "yes", "on":
            return true
        default:
            return false
        }
    }
}

public enum DeepseekV4ReasoningPolicyError: Error, LocalizedError, Sendable, Equatable {
    case invalidReasoningEffort(String)

    public var errorDescription: String? {
        switch self {
        case .invalidReasoningEffort(let value):
            return
                "Invalid DSV4 reasoning_effort `\(value)`; expected one of: low, high, max."
        }
    }
}
