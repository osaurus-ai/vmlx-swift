// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation

/// Runtime environment variables, under the project's `VMLX_` prefix.
///
/// The prefix is not new: 117 variables already use it, against 37 spelled `VMLINUX_` — the
/// Linux kernel image, a name this project has no claim to, being a macOS / Apple-silicon MLX
/// runtime with no Linux exposure outside the containerization frameworks. Someone reading the
/// environment surface is entitled to conclude the project has Linux heritage, and it has none.
///
/// The migration was already underway one variable at a time: twelve are read as
/// `env["VMLX_X"] ?? env["VMLINUX_X"]` inline, and `RuntimeMoETopKOverride` carries a
/// `legacyEnvironmentVariable` beside its `environmentVariable`. This finishes it for the rest.
///
/// It is an accessor rather than twenty-five more inline `??` pairs because the fallback is one
/// rule, and a rule copied to thirty-seven call sites is where the thirty-eighth gets it wrong.
/// Call sites pass the FULL new name, not a bare suffix, so `grep VMLX_ACCELERATOR` still finds
/// them — the legacy spelling is derived rather than written out.
///
/// Both spellings are honoured; `VMLX_` wins where both are set.
public enum RuntimeEnvironment {
    public static let prefix = "VMLX_"

    /// The pre-rename prefix, honoured so existing scripts and launch configs keep working.
    public static let legacyPrefix = "VMLINUX_"

    /// The legacy spelling of a `VMLX_` name — `nil` if `name` is not one.
    public static func legacyName(of name: String) -> String? {
        guard name.hasPrefix(prefix) else { return nil }
        return legacyPrefix + name.dropFirst(prefix.count)
    }

    /// The value of a runtime variable, preferring the `VMLX_` spelling.
    ///
    /// - Parameter name: the FULL current name, e.g. `"VMLX_ACCELERATOR"`. A name without the
    ///   prefix is looked up verbatim with no fallback, which is what a caller passing some
    ///   unrelated variable would want.
    public static func value(
        _ name: String,
        in environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> String? {
        if let current = environment[name] { return current }
        guard let legacy = legacyName(of: name) else { return nil }
        return environment[legacy]
    }

    /// A boolean runtime flag; absent or unparseable means `defaultValue`.
    ///
    /// The accepted spellings are the ones the call sites already accepted individually, so no
    /// variable changes meaning: `1` / `true` / `yes` / `on` are true.
    public static func flag(
        _ name: String,
        default defaultValue: Bool = false,
        in environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Bool {
        guard
            let raw = value(name, in: environment)?
                .trimmingCharacters(in: .whitespacesAndNewlines)
                .lowercased(),
            !raw.isEmpty
        else { return defaultValue }
        return ["1", "true", "yes", "on"].contains(raw)
    }
}
