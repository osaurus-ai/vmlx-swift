// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation

/// Template-context adapter for MiniMax-M3 / M3.1 (`model_type = minimax_m3`, `minimax_m3_vl`).
///
/// The M3 templates branch on **`thinking_mode`** and ignore `enable_thinking`:
///
/// - unset / null / unknown / `adaptive`: the adaptive system line and no generation prefix; the model's first token
///   decides (`<mm:think>` 200059 thinks, `</mm:think>` 200060 answers directly). M3.1's shipped template renders this
///   byte for byte;
/// - `enabled`: "Current thinking mode: enabled. ..." and the generation prompt ends with `<mm:think>`;
/// - `disabled`: "Current thinking mode: disabled. ..." and the generation prompt ends with `</mm:think>`.
///
/// `reasoning_effort` is read by the template as is (`<effort>…</effort>` in the system message) and passed through.
///
/// The generic request surface (the app's thinking toggle, OpenAI `enable_thinking`) maps the way vMLX Python's
/// `_normalize_minimax_m3_thinking_mode` maps it: On -> `enabled`, Off -> `disabled`, omitted -> adaptive. An explicit
/// `thinking_mode` in M3's own vocabulary wins over the toggle. `enable_thinking` is then dropped (the template never
/// reads it) and adaptive is expressed by leaving `thinking_mode` unset, so an adaptive request renders exactly the
/// shipped prompt and shares its cache entries. The stream parser needs no flag: the generation loop resolves its start
/// state from the rendered prompt tail (`ReasoningParser.forPrompt`), inside reasoning after `<mm:think>`, in content
/// after `</mm:think>` or with no marker.
public enum MiniMaxM3ThinkingTemplateContext {
    /// The template's own vocabulary.
    public static let modes: Set<String> = ["enabled", "disabled", "adaptive"]
    public static let templateKey = "thinking_mode"

    public static func applies(to modelType: String?) -> Bool {
        guard let t = modelType?.lowercased().replacingOccurrences(of: "-", with: "_") else { return false }
        return t == "minimax_m3" || t.hasPrefix("minimax_m3_")
    }

    /// The template's mode for a context: nil = adaptive.
    public static func resolvedMode(_ context: [String: any Sendable]?) -> String? {
        if let raw = context?[templateKey] as? String {
            let mode = raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            if modes.contains(mode) { return mode == "adaptive" ? nil : mode }
        }
        switch context?["enable_thinking"] as? Bool {
        case true?: return "enabled"
        case false?: return "disabled"
        case nil: return nil
        }
    }

    public static func apply(
        additionalContext: [String: any Sendable]?,
        modelType: String?
    ) -> [String: any Sendable]? {
        guard applies(to: modelType) else { return additionalContext }
        var context = additionalContext ?? [:]
        let mode = resolvedMode(additionalContext)
        context.removeValue(forKey: "enable_thinking")
        context.removeValue(forKey: "thinking")
        if let mode {
            context[templateKey] = mode
        } else {
            context.removeValue(forKey: templateKey)
        }
        return context.isEmpty ? nil : context
    }
}
