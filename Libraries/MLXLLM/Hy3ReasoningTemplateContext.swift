// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation

/// Template-context adapter for official Hunyuan v3 (`model_type = hy_v3`).
///
/// The official chat template keys thinking on a `reasoning_effort` variable
/// with the closed set `no_think` / `low` / `high` — it IGNORES the
/// `enable_thinking` boolean every other family understands, and it
/// `raise_exception`s on any other effort value (so an OpenAI-style
/// `reasoning_effort: "medium"` would fail the whole render). This adapter
/// translates the request surface into the template's contract:
///
/// - explicit `reasoning_effort` wins: `no_think`/`low`/`high` pass through;
///   `none`/`off` → `no_think`, `minimal`/`medium` → `low`, `max`/anything
///   else non-empty → `high`.
/// - otherwise `enable_thinking` maps `true` → `high`, `false` → `no_think`.
/// - with neither present, nothing is injected and the template's own
///   default (`no_think`) applies.
public enum Hy3ReasoningTemplateContext {
    public static func applies(to modelType: String?) -> Bool {
        guard let t = modelType?.lowercased() else { return false }
        return t == "hy_v3" || t.hasPrefix("hy_v3_") || t.hasPrefix("hy3")
            || t.hasPrefix("hunyuan")
    }

    public static func apply(
        additionalContext: [String: any Sendable]?,
        modelType: String?
    ) -> [String: any Sendable]? {
        guard applies(to: modelType) else { return additionalContext }
        var context = additionalContext ?? [:]

        let effort: String?
        if let requested = (context["reasoning_effort"] as? String)?.lowercased(),
            !requested.isEmpty
        {
            switch requested {
            case "no_think", "low", "high": effort = requested
            case "none", "off": effort = "no_think"
            case "minimal", "medium": effort = "low"
            default: effort = "high"
            }
        } else if let enable = context["enable_thinking"] as? Bool {
            effort = enable ? "high" : "no_think"
        } else {
            effort = nil
        }

        if let effort {
            context["reasoning_effort"] = effort
        } else {
            // Never forward an unvalidated value the template would raise on.
            context.removeValue(forKey: "reasoning_effort")
        }
        return context.isEmpty ? nil : context
    }
}
