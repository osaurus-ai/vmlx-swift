// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation

/// Runtime policy for DSV4 public reasoning controls.
///
/// Public reasoning controls are passed through to the chat template after
/// spelling/case normalization. The runtime must not silently downgrade
/// `reasoning_effort=max` or alias low/medium to high; any model behavior at
/// those rails is a real behavior to measure at the API boundary.
public enum DeepseekV4ReasoningPolicy {
    /// Deprecated compatibility key. `reasoning_effort=max` now passes through
    /// without an opt-in environment variable.
    public static let rawMaxEnvironmentKey = "VMLINUX_DSV4_RAW_MAX"

    /// Deprecated compatibility key. Public request/template controls must win;
    /// process environment must not silently force direct-answer rails.
    public static let forceDirectRailEnvironmentKey = "VMLINUX_DSV4_FORCE_DIRECT_RAIL"

    public static func isDeepseekV4(modelType: String?) -> Bool {
        guard let modelType else { return false }
        let normalized = modelType
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
            .replacingOccurrences(of: "-", with: "_")
        return normalized == "deepseek_v4"
            || normalized.hasPrefix("deepseek_v4_")
            || normalized == "deepseekv4"
            || normalized.hasPrefix("deepseekv4_")
    }

    public static func rawMaxEnabled(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Bool {
        truthy(environment[rawMaxEnvironmentKey])
    }

    @available(*, deprecated, message: "Do not use process env to override explicit reasoning controls.")
    public static func forceDirectRailEnabled(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Bool {
        truthy(environment[forceDirectRailEnvironmentKey])
    }

    public static func normalizedReasoningEffort(
        _ value: (any Sendable)?,
        rawMaxEnabled: Bool = rawMaxEnabled()
    ) -> String? {
        guard let effort = normalizedString(value) else { return nil }
        switch effort {
        case "max", "maximum":
            return "max"
        case "low", "medium", "high":
            return effort
        default:
            return nil
        }
    }

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
    ) -> [String: any Sendable]? {
        guard isDeepseekV4(modelType: modelType) else { return context }

        var normalized = context ?? [:]

        let rawEffort = normalized["reasoning_effort"]
        if let effort = normalizedReasoningEffort(rawEffort) {
            normalized["enable_thinking"] = true
            normalized["reasoning_effort"] = effort
        } else if isDirectRailEffort(rawEffort) {
            normalized["enable_thinking"] = false
            normalized.removeValue(forKey: "reasoning_effort")
        } else if normalized["enable_thinking"] as? Bool == false {
            normalized.removeValue(forKey: "reasoning_effort")
        }

        return normalized.isEmpty ? nil : normalized
    }

    private static func normalizedString(_ value: (any Sendable)?) -> String? {
        if let value = value as? String {
            let normalized = value
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
