// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLXLMCommon

/// Template-context adapter for Muse Glimmer (`model_type = muse_glimmer`, `muse_glimmer_text`).
///
/// The chat template keys reasoning on a `reasoning_strength` variable:
///
/// ```jinja
/// {%- set rs = reasoning_strength if reasoning_strength is defined and reasoning_strength else 'high' -%}
/// ```
///
/// That key appears nowhere else in this library, and no caller sends it — every request therefore
/// renders at the template's own fallback of `high`, whatever effort was asked for, and nothing
/// reports that the request was dropped. The bundle states the contract plainly in `jang_config.json`
/// (`chat.reasoning_control = "reasoning_strength"`, `reasoning_values = [low, medium, high, xhigh]`,
/// `reasoning_default = "high"`), so this is a translation gap rather than an unknown.
///
/// The same failure already has a name here: `ReasoningContract`'s fallback table records that an
/// undeclared gpt-oss "silently ran at medium while claiming level 1". This is that, one family over.
/// It survives a green suite because the tests pass `reasoning_strength` explicitly and so prove the
/// template honours it, while no production path ever sets it.
///
/// This adapter translates the request surface into the template's contract, mirroring
/// ``Hy3ReasoningTemplateContext``:
///
/// - explicit `reasoning_strength` wins, when it names one of the declared values.
/// - otherwise `reasoning_effort` — the key callers actually send — is mapped into this family's
///   vocabulary.
/// - otherwise `enable_thinking` is honoured as far as the model allows (see below).
/// - with none of them present nothing is injected, and the template's own `high` applies. That is
///   deliberate: injecting the default would restate a value already in force and make an inert
///   change look like a behavioural one.
public enum MuseGlimmerReasoningTemplateContext {

    /// The template variable this family reads. Named once, here, so a caller that wants to know
    /// which key carries the control has somewhere to read it from rather than a string literal
    /// buried in a translation.
    public static let templateKey = "reasoning_strength"

    /// Declared in `jang_config.json` as `chat.reasoning_values`, weakest → strongest.
    public static let strengths = ["low", "medium", "high", "xhigh"]

    public static func applies(to modelType: String?) -> Bool {
        guard let t = modelType?.lowercased() else { return false }
        return t == "muse_glimmer" || t.hasPrefix("muse_glimmer_")
    }

    public static func apply(
        additionalContext: [String: any Sendable]?,
        modelType: String?
    ) -> [String: any Sendable]? {
        guard applies(to: modelType) else { return additionalContext }
        var context = additionalContext ?? [:]

        let strength: String?
        if let explicit = (context[templateKey] as? String)?.lowercased(),
            strengths.contains(explicit)
        {
            strength = explicit
        } else if let requested = (context["reasoning_effort"] as? String)?.lowercased(),
            !requested.isEmpty
        {
            strength = normalize(requested)
        } else if let enable = context["enable_thinking"] as? Bool {
            // Muse Glimmer cannot stop reasoning — there is no value in its vocabulary that turns the
            // rail off, and `jang_config.json` declares no off mode. A caller asking to disable it is
            // asking for as little reasoning as possible, so honour that as the WEAKEST strength
            // rather than either ignoring the request or pretending it was satisfied. `true` injects
            // nothing: the template already defaults to reasoning, and restating that would be inert.
            strength = enable ? nil : strengths.first
        } else {
            strength = nil
        }

        if let strength {
            context[templateKey] = strength
        }
        return context.isEmpty ? nil : context
    }

    /// Map a generic effort word onto this family's vocabulary.
    ///
    /// Unrecognised values resolve to the declared default rather than being forwarded: the template
    /// does not validate, so an unknown string would silently render as itself in the system prompt
    /// (`Reasoning strength: banana.`) instead of failing loudly the way Hunyuan 3's `raise_exception`
    /// does. Landing on the default is the behaviour a caller gets today.
    static func normalize(_ raw: String) -> String {
        switch raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
        case "low", "medium", "high", "xhigh": return raw.lowercased()
        case "minimal", "none", "off", "no_think": return "low"
        case "max", "maximum", "highest": return "xhigh"
        default: return "high"
        }
    }
}
