// Copyright © 2026 Osaurus AI. All rights reserved.

import Foundation
@testable import MLXLMCommon
import Testing

@Suite("Reasoning effort: a model can be asked what levels it has")
struct ReasoningEffortPolicyTests {

    // MARK: The shape of each family

    @Test("Muse Glimmer is think-only with four efforts, and reads reasoning_strength")
    func museGlimmer() {
        let c = MuseGlimmerReasoningPolicy.capability
        #expect(c.levels == [1, 2, 3, 4])
        #expect(c.isThinkingOnly)
        #expect(!c.canDisableReasoning)
        #expect(c.defaultLevel == 3)              // "high"
        #expect(c.effort(for: 3) == "high")
        // The bug this fixes: the template reads reasoning_strength, and nothing populated it, so
        // every request ran at the template's own default no matter what was asked.
        #expect(c.templateKey == "reasoning_strength")
    }

    @Test("gpt-oss has three efforts, defaults to medium, and can be switched off")
    func gptOss() {
        let c = GptOssReasoningPolicy.capability
        #expect(c.levels == [0, 1, 2, 3])
        #expect(c.canDisableReasoning)
        // Not level 1. A published gpt-oss figure is almost certainly at medium, so a caller asking
        // for "the lowest thinking level" and a caller asking for "this model's default" must be
        // able to tell they are different requests.
        #expect(c.defaultLevel == 2)
        #expect(c.effort(for: 2) == "medium")
        #expect(c.templateKey == "reasoning_effort")
    }

    @Test("DeepSeek V4 keeps low/high/max and disables via enable_thinking, not an effort value")
    func deepseekV4() {
        let c = DeepseekV4EffortPolicy.capability
        #expect(c.levels == [0, 1, 2, 3])
        #expect(c.effort(for: 3) == "max")
        #expect(c.defaultLevel == 1)              // "low"
        #expect(c.offEffort == nil)               // off is enable_thinking = false
        // Delegates to the existing policy, so the accepted set cannot drift from it.
        #expect(DeepseekV4EffortPolicy.normalize("MAX") == "max")
        #expect(DeepseekV4EffortPolicy.normalize("medium") == nil)
    }

    @Test("Hunyuan 3 expresses off as an effort value and DEFAULTS to not reasoning")
    func hunyuan3() {
        let c = Hy3EffortPolicy.capability
        #expect(c.levels == [0, 1, 2])
        #expect(c.offEffort == "no_think")
        // Its template injects nothing when neither key is present and falls back to no_think, so the
        // model's own default is level 0 — a default that is not a thinking level at all.
        #expect(c.defaultLevel == 0)
        #expect(Hy3EffortPolicy.normalize("medium") == "low")   // documented alias
        #expect(Hy3EffortPolicy.normalize("off") == "no_think")
    }

    @Test("Bailing declares a toggle whose key is a boolean the adapter interprets, not a template var")
    func bailing() {
        let c = BailingReasoningPolicy.capability
        #expect(c.levels == [0, 1])
        #expect(c.templateKey == "enable_thinking")
        #expect(c.efforts.isEmpty)
        // Level 1 sets only the boolean — there is no effort name to send, and inventing one would
        // put a string into a slot the adapter reads as Bool.
        let on = c.applying(level: 1)
        #expect(on?["enable_thinking"] as? Bool == true)
        #expect(on?.count == 1)
        // Without this policy the family falls through to the template scan, finds no
        // `enable_thinking` (its template has none), and is reported as having no control at all —
        // while BailingThinkingTemplateContext stands ready to act on exactly that key.
        #expect(ReasoningCapabilityRegistry.capability(modelType: "bailing_moe").levels == [0, 1])
    }

    // MARK: The fallbacks — where most models are

    @Test("a model that declares nothing is still one of three definite shapes")
    func fallbackShapes() {
        #expect(ReasoningCapability.nonThinking.levels == [0])
        #expect(ReasoningCapability.thinkingOnly.levels == [1])
        #expect(ReasoningCapability.thinkingToggle.levels == [0, 1])
        // The distinction that matters to a caller: asking a think-only model to stop, or a
        // non-thinking model to think, both return something other than what was requested.
        #expect(ReasoningCapability.thinkingOnly.isThinkingOnly)
        #expect(!ReasoningCapability.thinkingToggle.isThinkingOnly)
    }

    @Test("supports_thinking = false wins over everything inferable")
    func nonThinkingDeclared() {
        let c = ReasoningCapabilityRegistry.capability(
            modelType: "some_new_arch", supportsThinking: false, templateMentionsToggle: true)
        #expect(c.levels == [0])
    }

    @Test("a template that reads enable_thinking implies an on/off model")
    func toggleInferred() {
        let c = ReasoningCapabilityRegistry.capability(
            modelType: "some_new_arch", templateMentionsToggle: true)
        #expect(c.levels == [0, 1])
    }

    @Test("a model that thinks but exposes no toggle is think-only, not a toggle")
    func thinkOnlyInferred() {
        let c = ReasoningCapabilityRegistry.capability(
            modelType: "some_new_arch", supportsThinking: true, templateMentionsToggle: false)
        #expect(c.levels == [1])
    }

    /// Knowing nothing reports NO control, not a toggle.
    ///
    /// This test used to assert `[0, 1]`, reasoning that a new capability API should not narrow what
    /// callers could already do — a caller could always pass `enable_thinking`, so an unrecognised
    /// model should keep saying yes. The resolver's step 4 says the opposite, and it is right: a
    /// toggle is a claim that reasoning can be switched FROM THE PROMPT, and for a model with no
    /// policy, no declared efforts and no toggle in its template there is nothing to switch. The
    /// claim would be false however permissively it was meant.
    ///
    /// The disagreement matters because `forModel(at:)` — the entry point every real caller uses —
    /// funnels a bundle whose artefacts betray nothing into this same call with everything defaulted.
    /// Under `[0, 1]` such a bundle advertises a toggle, a caller asking for level 1 gets
    /// `enable_thinking = true` that the template ignores, the model does not reason, and the run is
    /// recorded as a thinking run. Under `[0]` the caller is told there is no control, which is what
    /// the artefacts actually support.
    ///
    /// Kept as a named contract rather than deleted: the two readings are both defensible in the
    /// abstract, and the next person to find this call permissive-looking should see why it is not.
    @Test("knowing nothing reports no control, rather than a toggle it cannot honour")
    func unknownReportsNoControl() {
        #expect(ReasoningCapabilityRegistry.capability(modelType: nil).levels == [0])
        #expect(ReasoningCapabilityRegistry.capability(modelType: "brand_new").levels == [0])
        // The part of the original intent that still holds: infer no effort ladder from nothing.
        #expect(ReasoningCapabilityRegistry.capability(modelType: "brand_new").efforts.isEmpty)
    }

    // MARK: Resolution order

    @Test("a family policy beats a bundle declaration, which beats inference")
    func resolutionOrder() {
        // A policy exists → the bundle's (wrong) claim of two efforts is ignored, because the policy
        // also knows the template key and the bundle does not.
        let policied = ReasoningCapabilityRegistry.capability(
            modelType: "muse_glimmer", declaredEfforts: ["a", "b"], declaredDefault: "a")
        #expect(policied.levels == [1, 2, 3, 4])
        #expect(policied.templateKey == "reasoning_strength")

        // No policy → the model's own declaration is trusted over anything inferable.
        let declared = ReasoningCapabilityRegistry.capability(
            modelType: "unknown_arch", declaredEfforts: ["low", "mid", "high"],
            declaredDefault: "mid", templateMentionsToggle: true)
        #expect(declared.levels == [0, 1, 2, 3])
        #expect(declared.defaultLevel == 2)
        #expect(declared.effort(for: 3) == "high")
    }

    @Test("a declared default the family does not list is a declaration bug, not a level")
    func bogusDefaultFallsBack() {
        let c = ReasoningCapabilityRegistry.capability(
            modelType: "unknown_arch", declaredEfforts: ["low", "high"], declaredDefault: "enormous")
        #expect(c.defaultLevel == 1)
        #expect(c.levels == [0, 1, 2])
    }

    @Test("levels are contiguous and ascending for every family")
    func levelsAreWellFormed() {
        for p in ReasoningCapabilityRegistry.policies {
            let c = p.capability
            #expect(c.levels == Array(c.levels.min()!...c.levels.max()!), "\(p) has a gap")
            #expect(c.levels.contains(c.defaultLevel), "\(p) defaults outside its own levels")
            #expect(c.effort(for: 0) == nil, "\(p) named level 0 as an effort")
            #expect(c.effort(for: c.maximumLevel + 1) == nil, "\(p) answered past its top level")
        }
    }

    @Test("every family's predicate is disjoint from the others")
    func predicatesDoNotOverlap() {
        let types = ["gpt_oss", "muse_glimmer", "muse_glimmer_text", "deepseek_v4", "hy_v3",
                     "hunyuan_a13b", "bailing_moe", "bailing_hybrid", "bailing_moe_v2_5"]
        for t in types {
            let matches = ReasoningCapabilityRegistry.policies.filter { $0.applies(to: t) }
            #expect(matches.count == 1, "\(t) matched \(matches.count) policies")
        }
    }
}

/// Reading a capability off a bundle on disk.
///
/// Fixtures are WRITTEN by the test rather than committed, so it runs anywhere and does not depend on
/// a model collection. Each one isolates a decision the resolver has to make, and several encode a bug
/// that shipped: the template found in only one of its two locations, and a reasoning family detected
/// by only one of its several markers.
@Suite("Reasoning capability, resolved from a bundle")
struct ReasoningCapabilityBundleTests {

    private func bundle(_ files: [String: String]) throws -> URL {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("reasoning-fixture-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        for (name, body) in files {
            try body.write(to: dir.appendingPathComponent(name), atomically: true, encoding: .utf8)
        }
        return dir
    }

    @Test("the template is found in EITHER location")
    func templateInEitherPlace() throws {
        // Across real bundles the chat template is split almost evenly between a standalone
        // `chat_template.jinja` and a `chat_template` field inside `tokenizer_config.json`. Reading
        // only the first made half a collection look like it had no template — which reads as "no
        // toggle expressible", which reads as "non-thinking".
        let asFile = try bundle([
            "config.json": #"{"model_type":"unknown_family"}"#,
            "chat_template.jinja": "{% if enable_thinking %}…{% endif %}",
        ])
        let asField = try bundle([
            "config.json": #"{"model_type":"unknown_family"}"#,
            "tokenizer_config.json": #"{"chat_template":"{% if enable_thinking %}…{% endif %}"}"#,
        ])
        #expect(ReasoningCapability.forModel(at: asFile).levels == [0, 1])
        #expect(ReasoningCapability.forModel(at: asField).levels == [0, 1],
                "a template in tokenizer_config.json must count as a template")
    }

    @Test("a thinking marker is never reported as non-thinking", arguments: [
        "<think>reasoning</think>", "[THINK]reasoning[/THINK]", "<|channel|>analysis<|message|>",
    ])
    func anyMarkerMeansThinking(marker: String) throws {
        // Families disagree on the spelling. `Ministral-3-3B-Instruct-2512` ships `[THINK]` and
        // nothing else, and a single-marker check reported it as having no thinking channel at all.
        let dir = try bundle([
            "config.json": #"{"model_type":"unknown_family"}"#,
            "chat_template.jinja": "{{ '\(marker)' }}",
        ])
        let c = ReasoningCapability.forModel(at: dir)
        #expect(c.levels == [1], "\(marker) should read as think-only, got \(c.levels)")
        #expect(!c.canDisableReasoning)
    }

    @Test("no marker and no toggle is genuinely non-thinking")
    func noMarkerNoToggle() throws {
        // The control for the test above: if any content counted as a marker, that test would pass
        // for the wrong reason.
        let dir = try bundle([
            "config.json": #"{"model_type":"unknown_family"}"#,
            "chat_template.jinja": "{{ messages[0]['content'] }}",
        ])
        #expect(ReasoningCapability.forModel(at: dir).levels == [0])
    }

    @Test("an effort vocabulary is read under any of its three spellings", arguments: [
        #"{"chat":{"reasoning":{"reasoning_effort_levels":["a","b","c"]}}}"#,
        #"{"chat":{"reasoning":{},"reasoning_values":["a","b","c"]}}"#,
        #"{"chat":{"reasoning":{"modes":["a","b","c"]}}}"#,
    ])
    func effortSpellings(jang: String) throws {
        // Muse Glimmer declares four levels under `modes`, and reading only the canonical spelling
        // reported it as declaring none.
        let dir = try bundle(["config.json": #"{"model_type":"unknown_family"}"#, "jang_config.json": jang])
        let c = ReasoningCapability.forModel(at: dir)
        #expect(c.efforts == ["a", "b", "c"], "spelling not recognised: \(jang)")
    }

    @Test("supports_thinking = false beats every inference")
    func explicitlyNonThinking() throws {
        let dir = try bundle([
            "config.json": #"{"model_type":"unknown_family","capabilities":{"supports_thinking":false}}"#,
            "chat_template.jinja": "<think>{% if enable_thinking %}…{% endif %}",
        ])
        #expect(ReasoningCapability.forModel(at: dir).levels == [0])
    }

    @Test("a missing bundle resolves rather than crashing")
    func absentBundle() {
        let c = ReasoningCapability.forModel(at: URL(fileURLWithPath: "/nonexistent/model"))
        #expect(c.levels.contains(c.defaultLevel))
    }
}

/// A guess must be distinguishable from a declaration.
///
/// The fallbacks are the part of this design most likely to rot: a model that declares nothing still
/// resolves to one of three definite shapes, and if that guess is wrong the caller has no way to
/// notice — the same failure mode as the bug the capability surface exists to prevent. So the
/// provenance travels with the value instead of being flattened away, and these pin which rung
/// produces which.
@Suite("Where a capability came from is visible to the caller")
struct ReasoningCapabilitySourceTests {

    @Test("a registered family reports policy")
    func familyIsPolicy() {
        #expect(ReasoningCapabilityRegistry.capability(modelType: "muse_glimmer").source == .policy)
        #expect(ReasoningCapabilityRegistry.capability(modelType: "gpt_oss").source == .policy)
    }

    @Test("the model's own words report declared")
    func modelWordsAreDeclared() {
        let efforts = ReasoningCapabilityRegistry.capability(
            modelType: "brand_new", declaredEfforts: ["a", "b", "c"], declaredDefault: "b")
        #expect(efforts.source == .declared)
        #expect(ReasoningCapabilityRegistry.capability(
            modelType: "brand_new", supportsThinking: false).source == .declared)
        #expect(ReasoningCapabilityRegistry.capability(
            modelType: "brand_new", supportsThinking: true).source == .declared)
    }

    @Test("shapes we deduce report inferred")
    func deductionsAreInferred() {
        #expect(ReasoningCapabilityRegistry.capability(
            modelType: "brand_new", templateMentionsToggle: true).source == .inferred)
        #expect(ReasoningCapabilityRegistry.capability(
            modelType: "brand_new", reasoningSignal: true).source == .inferred)
        #expect(ReasoningCapabilityRegistry.capability(modelType: "brand_new").source == .inferred)
    }

    /// Relabelling must not quietly change the shape it relabels.
    @Test("provenance does not alter levels")
    func relabellingPreservesShape() {
        let declared = ReasoningCapabilityRegistry.capability(
            modelType: "brand_new", supportsThinking: true)
        #expect(declared.levels == ReasoningCapability.thinkingOnly.levels)
        #expect(declared.defaultLevel == ReasoningCapability.thinkingOnly.defaultLevel)
    }
}
