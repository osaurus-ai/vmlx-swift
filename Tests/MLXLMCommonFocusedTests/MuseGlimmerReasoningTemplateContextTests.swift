// Copyright © 2026 Osaurus AI. All rights reserved.

import Foundation
@testable import MLXLLM
import Testing
import VMLXJinja

/// Muse Glimmer's reasoning control has to actually reach the template.
///
/// The template reads `reasoning_strength`; callers send `reasoning_effort`. Nothing translated
/// between them, so every request rendered at the template's own fallback of `high` no matter what
/// was asked, and nothing reported that the request had been dropped.
///
/// The first test is the bug. The rest pin the translation and, just as importantly, the cases where
/// this adapter must do NOTHING — an adapter that fires on the wrong family, or injects a default
/// that was already in force, trades a silent wrong answer for a silent behaviour change.
@Suite("Muse Glimmer reasoning strength reaches the template")
struct MuseGlimmerReasoningTemplateContextTests {

    private func apply(_ ctx: [String: any Sendable], type: String = "muse_glimmer")
        -> [String: any Sendable]
    {
        MuseGlimmerReasoningTemplateContext.apply(additionalContext: ctx, modelType: type) ?? [:]
    }
    private func strength(_ ctx: [String: any Sendable], type: String = "muse_glimmer") -> String? {
        apply(ctx, type: type)["reasoning_strength"] as? String
    }

    // MARK: The bug

    @Test("a requested effort is translated into the key the template reads")
    func effortBecomesStrength() {
        #expect(strength(["reasoning_effort": "low"]) == "low",
                "reasoning_effort did not reach the template as reasoning_strength — this is the bug")
        #expect(strength(["reasoning_effort": "xhigh"]) == "xhigh")
    }

    @Test("every declared value survives the round trip")
    func vocabularyRoundTrips() {
        for v in MuseGlimmerReasoningTemplateContext.strengths {
            #expect(strength(["reasoning_effort": v]) == v, "\(v) did not survive translation")
        }
    }

    @Test("generic effort words map onto this family's vocabulary")
    func aliasesMap() {
        #expect(strength(["reasoning_effort": "minimal"]) == "low")
        #expect(strength(["reasoning_effort": "none"]) == "low")
        #expect(strength(["reasoning_effort": "max"]) == "xhigh")
        #expect(strength(["reasoning_effort": "MEDIUM"]) == "medium", "matching must be case-insensitive")
    }

    /// The template does not validate, so an unknown value would render as itself —
    /// `Reasoning strength: banana.` — rather than failing the way Hunyuan 3's `raise_exception` does.
    @Test("an unrecognised value lands on the default instead of rendering verbatim")
    func unknownFallsBackToDefault() {
        let s = strength(["reasoning_effort": "banana"])
        #expect(s == "high")
        #expect(MuseGlimmerReasoningTemplateContext.strengths.contains(s ?? ""),
                "forwarded a value outside the declared vocabulary, which the template would print raw")
    }

    // MARK: Precedence

    @Test("an explicit reasoning_strength wins over a generic effort")
    func explicitWins() {
        #expect(strength(["reasoning_strength": "medium", "reasoning_effort": "low"]) == "medium")
    }

    /// Glimmer has no off value and declares no off mode, so "stop thinking" can only be honoured as
    /// far as the model allows. The weakest strength is that; silently leaving it at `high` is not.
    @Test("asking to disable reasoning yields the weakest strength this family has")
    func disableBecomesWeakest() {
        #expect(strength(["enable_thinking": false]) == "low")
    }

    // MARK: Where it must do nothing

    @Test("enable_thinking = true injects nothing, because the template already reasons")
    func enableTrueInjectsNothing() {
        #expect(strength(["enable_thinking": true]) == nil,
                "injected a default that was already in force, turning an inert call into a change")
    }

    @Test("an empty request injects nothing and leaves the template's own default")
    func silentStaysSilent() {
        #expect(MuseGlimmerReasoningTemplateContext.apply(additionalContext: nil,
                                                          modelType: "muse_glimmer") == nil)
        #expect(strength([:]) == nil)
    }

    /// The control. Without it, an adapter that fired on everything would pass every test above.
    @Test("other families are untouched")
    func otherFamiliesUntouched() {
        for other in ["hy_v3", "bailing_moe", "gpt_oss", "deepseek_v4", "llama", nil] as [String?] {
            let out = MuseGlimmerReasoningTemplateContext.apply(
                additionalContext: ["reasoning_effort": "low"], modelType: other) ?? [:]
            #expect(out["reasoning_strength"] == nil,
                    "wrote reasoning_strength for \(other ?? "nil"), which does not read that key")
            #expect(out["reasoning_effort"] as? String == "low",
                    "modified another family's reasoning_effort")
        }
    }

    @Test("the text tower is the same family")
    func textTowerMatches() {
        #expect(MuseGlimmerReasoningTemplateContext.applies(to: "muse_glimmer_text"))
        #expect(MuseGlimmerReasoningTemplateContext.applies(to: "muse_glimmer"))
        #expect(!MuseGlimmerReasoningTemplateContext.applies(to: "muse"))
    }

    // MARK: The end-to-end proof

    /// Render the template's own reasoning line, with and without this adapter.
    ///
    /// Everything above tests the translation in isolation, which cannot show the thing that made
    /// this bug survive: that a request looks accepted and changes nothing. This does. The fixture is
    /// the real line from Muse Glimmer's `chat_template.jinja` —
    ///
    /// ```jinja
    /// {%- set rs = reasoning_strength if reasoning_strength is defined and reasoning_strength else 'high' -%}
    /// ```
    ///
    /// — so the "before" case is not a reconstruction of the defect, it is the defect: a caller asks
    /// for `low`, the template never sees a key it recognises, and renders `high`.
    @Test("the request changes the rendered prompt, which is what it never did before")
    func requestReachesTheRenderedPrompt() throws {
        let line = """
            {%- set rs = reasoning_strength if reasoning_strength is defined and reasoning_strength else 'high' -%}
            Reasoning strength: {{ rs }}.
            """
        func render(_ ctx: [String: any Sendable]) throws -> String {
            var values: [String: Value] = [:]
            for (k, v) in ctx { values[k] = try Value(any: v) }
            return try Template(line).render(values)
        }

        let asked: [String: any Sendable] = ["reasoning_effort": "low"]

        let before = try render(asked)
        #expect(before == "Reasoning strength: high.",
                "fixture is wrong: without translation the template must fall back to high")

        let after = try render(apply(asked))
        #expect(after == "Reasoning strength: low.",
                "the adapter ran but the rendered prompt still does not carry the requested strength")
        #expect(before != after, "the request must change the prompt, or nothing was fixed")
    }
}
