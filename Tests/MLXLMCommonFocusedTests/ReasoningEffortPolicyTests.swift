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

    // MARK: Selecting a level when off is an effort value

    @Test("where off is an effort value, a thinking level sends only its effort")
    func offAsEffortSendsNoBoolean() {
        // Hunyuan 3's template reads `reasoning_effort` and ignores `enable_thinking`, and the Osaurus
        // app sends a declared effort contract's kwargs and nothing else. A boolean beside the effort
        // is a kwarg nobody declared.
        let hy3 = Hy3EffortPolicy.capability
        #expect(sent(hy3, level: 0) == ["reasoning_effort": "string:no_think"])
        #expect(sent(hy3, level: 1) == ["reasoning_effort": "string:low"])
        #expect(sent(hy3, level: 2) == ["reasoning_effort": "string:high"])

        let declared = ReasoningCapability(
            efforts: ["low", "high"], defaultEffort: "off", supportsDisabling: true,
            offEffort: "off", templateKey: "reasoning_effort", source: .declared)
        #expect(sent(declared, level: 0) == ["reasoning_effort": "string:off"])
        #expect(sent(declared, level: 2) == ["reasoning_effort": "string:high"])
    }

    @Test("where off is enable_thinking, a thinking level still asserts it")
    func offAsBooleanStillAsserted() {
        // The control for the test above: only the effort spelling of off loses the boolean.
        let gptOss = GptOssReasoningPolicy.capability
        #expect(sent(gptOss, level: 0) == ["enable_thinking": "bool:false"])
        #expect(sent(gptOss, level: 1)
                == ["reasoning_effort": "string:low", "enable_thinking": "bool:true"])
        // An off effort with no key to carry it is spelled as the boolean at level 0, so the
        // thinking levels keep saying the boolean too.
        let keyless = ReasoningCapability(
            efforts: ["low", "high"], defaultEffort: nil, supportsDisabling: true,
            offEffort: "off", templateKey: nil, source: .declared)
        #expect(sent(keyless, level: 0) == ["enable_thinking": "bool:false"])
        #expect(sent(keyless, level: 1) == ["enable_thinking": "bool:true"])
    }

    @Test("no family policy declares per-level kwargs")
    func policiesHaveNoLevelKwargs() {
        for p in ReasoningCapabilityRegistry.policies {
            #expect(p.capability.levelKwargs.isEmpty, "\(p) declared per-level kwargs")
        }
    }
}

/// What `applying(level:)` sends, flattened to literals (`"bool:true"`, `"string:low"`), so a test can
/// state the whole context it expects — every key, and nothing besides — as one dictionary literal.
/// `nil` when the level is not on the scale.
fileprivate func sent(
    _ capability: ReasoningCapability, level: Int, to context: [String: any Sendable] = [:]
) -> [String: String]? {
    capability.applying(level: level, to: context)?.mapValues { value -> String in
        if let s = value as? String { return "string:\(s)" }
        if let b = value as? Bool { return "bool:\(b)" }
        return "other:\(value)"
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

    @Test("bundle efforts coexist with independent think/no-think modes")
    func declaredEffortsWithToggleModes() throws {
        let dir = try bundle([
            "config.json": #"{"model_type":"spark2_5"}"#,
            "jang_config.json": #"{"chat":{"reasoning":{"supported":true,"modes":["think","no_think"],"efforts":["low","high"],"default_effort":"low"}}}"#,
            "chat_template.jinja": "{{ enable_thinking }} {{ reasoning_effort }}",
        ])
        defer { try? FileManager.default.removeItem(at: dir) }
        let c = ReasoningCapability.forModel(at: dir)
        #expect(c.levels == [0, 1, 2])
        #expect(c.defaultLevel == 1)
        #expect(c.effort(for: 1) == "low")
        #expect(c.effort(for: 2) == "high")
        #expect(c.applying(level: 0)?["enable_thinking"] as? Bool == false)
        #expect(c.applying(level: 2)?["reasoning_effort"] as? String == "high")
        #expect(c.source == .declared)
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

    /// A JANGH2 bundle with no `reasoning` block at all, only the flat spelling inside `chat`.
    ///
    /// `unknown_family` is deliberate. `GLM-5.3-Flash-JANGH2` is the real bundle, and its family
    /// policy happens to declare the same `[low, high, max]` with the same `max` default, so a test
    /// built on `glm5_next` would pass whether or not the declaration was read.
    private func chatCapability(
        chat: String, extra: String = "",
        config: String = #"{"model_type":"unknown_family"}"#, template: String? = nil
    ) throws -> ReasoningCapability {
        var files = [
            "config.json": config,
            "jang_config.json": #"{\#(extra)"chat":{\#(chat)}}"#,
        ]
        if let template { files["chat_template.jinja"] = template }
        return ReasoningCapability.forModel(at: try bundle(files))
    }

    @Test("the flat chat.reasoning_efforts spelling is read, with its default")
    func flatEffortSpelling() throws {
        let c = try chatCapability(
            chat: #""reasoning_efforts":["low","high","max"],"reasoning_effort_default":"max""#)
        #expect(c.efforts == ["low", "high", "max"])
        #expect(c.source == .declared)
        // `max` is the TOP of the scale. A reader that took the list and dropped the default would
        // land on the weakest level, `low`, so this is what tells the two apart.
        #expect(c.effort(for: c.defaultLevel) == "max")
    }

    /// The bundle says the model supports thinking, which is the common case and the one that
    /// matters: `supports_thinking: true` is what turns an ordinary declared scale into one with an
    /// "off" level in front of it. A flat scale lists the efforts the model has, and Osaurus
    /// offers Off only where the list itself names it.
    private static let supportsThinking =
        #"{"model_type":"unknown_family","capabilities":{"supports_thinking":true}}"#

    @Test("a flat scale does not invent an off level")
    func flatSpellingDoesNotInventOff() throws {
        let c = try chatCapability(
            chat: #""reasoning_efforts":["low","high","max"],"reasoning_effort_default":"max""#,
            config: Self.supportsThinking)
        #expect(c.efforts == ["low", "high", "max"])
        #expect(c.levels == [1, 2, 3], "levels \(c.levels) — the list names no off")
        #expect(!c.canDisableReasoning)
        #expect(c.defaultLevel == 3)
    }

    @Test("an off entry in the flat list is the off effort, not a thinking level")
    func flatSpellingOffEntry() throws {
        let c = try chatCapability(
            chat: #""reasoning_efforts":["off","low","high"],"reasoning_effort_default":"off""#,
            config: Self.supportsThinking, template: "{{ reasoning_effort|default('low') }}")
        #expect(c.efforts == ["low", "high"])
        #expect(c.levels == [0, 1, 2])
        #expect(c.defaultLevel == 0, "the declared default is off")
        #expect(c.offEffort == "off")
        // Off is spelled as the effort VALUE the bundle declared, not as `enable_thinking`.
        #expect(c.applying(level: 0)?["reasoning_effort"] as? String == "off")
        #expect(c.applying(level: 2)?["reasoning_effort"] as? String == "high")
    }

    @Test("a flat list with one thinking effort and explicit off preserves both modes")
    func flatSpellingSingleEffortWithOff() throws {
        let c = try chatCapability(
            chat: #""reasoning_efforts":["off","high"],"reasoning_effort_default":"off""#,
            config: Self.supportsThinking, template: "{{ reasoning_effort|default('off') }}")
        #expect(c.levels == [0, 1])
        #expect(c.defaultLevel == 0)
        #expect(c.applying(level: 0)?["reasoning_effort"] as? String == "off")
        #expect(c.applying(level: 1)?["reasoning_effort"] as? String == "high")
        #expect(c.applying(level: 1)?["enable_thinking"] == nil)
    }

    @Test("the flat list is read trimmed and lowercased, default included")
    func flatSpellingIsNormalised() throws {
        let c = try chatCapability(
            chat: #""reasoning_efforts":[" Low","HIGH ","max"],"reasoning_effort_default":" MAX ""#)
        #expect(c.efforts == ["low", "high", "max"])
        #expect(c.effort(for: c.defaultLevel) == "max")
    }

    @Test("a flat list with an empty, repeated or doubly-off entry declares nothing",
          arguments: [#"["low",""]"#, #"["low","LOW","high"]"#, #"["off","none","low","high"]"#])
    func malformedFlatListDeclaresNothing(list: String) throws {
        // Osaurus reads each of these as "no effort control" rather than repairing the list, since
        // a repaired list is a vocabulary the bundle never wrote.
        let c = try chatCapability(chat: #""reasoning_efforts":\#(list)"#)
        #expect(c.efforts.isEmpty, "\(list) resolved to \(c.efforts)")
    }

    @Test("reasoning_effort_default wins over its alias default_reasoning_effort")
    func flatDefaultAliasLosesToTheStampsKey() throws {
        let list = #""reasoning_efforts":["low","high","max"]"#
        let alias = try chatCapability(chat: list + #","default_reasoning_effort":"high""#)
        #expect(alias.effort(for: alias.defaultLevel) == "high")
        // Both present: `reasoning_effort_default` is the stamp's own key, the other a tolerated
        // alias, so the alias must not override it.
        let both = try chatCapability(
            chat: list + #","reasoning_effort_default":"max","default_reasoning_effort":"low""#)
        #expect(both.effort(for: both.defaultLevel) == "max")
    }

    @Test("a reasoning block outranks the flat spelling, as it does in the app")
    func blockOutranksFlatSpelling() throws {
        // Osaurus reads the flat list only for a bundle with NO `reasoning` block. Where both
        // exist the block is the bundle's whole statement about efforts, so the flat list and its
        // default must not leak into it.
        let flat = #","reasoning_efforts":["x","y","z"],"reasoning_effort_default":"z""#
        let nested = try chatCapability(
            chat: #""reasoning":{"reasoning_effort_levels":["a","b","c"],"default_effort":"b"}"#
                + flat)
        #expect(nested.efforts == ["a", "b", "c"])
        #expect(nested.effort(for: nested.defaultLevel) == "b")

        let top = try chatCapability(
            chat: String(flat.dropFirst()),
            extra: #""reasoning":{"supported_reasoning_efforts":["a","b","c"],"#
                + #""default_reasoning_effort":"b"},"#)
        #expect(top.efforts == ["a", "b", "c"])
        #expect(top.effort(for: top.defaultLevel) == "b")
    }

    @Test("a nested block's reasoning_effort_supported:false withholds its later spellings")
    func nestedFlagWithholdsLaterSpellings() throws {
        // The raw reader this replaced checked the flag in whichever block it found, nested or
        // top-level, before reading `supported_reasoning_efforts`. The nested read must too.
        let block = #""supported_reasoning_efforts":["a","b","c"],"default_reasoning_effort":"b""#
        let denied = try chatCapability(
            chat: #""reasoning":{"supported":true,"reasoning_effort_supported":false,"# + block + "}")
        #expect(denied.efforts.isEmpty)
        let allowed = try chatCapability(chat: #""reasoning":{"supported":true,"# + block + "}")
        #expect(allowed.efforts == ["a", "b", "c"])
        #expect(allowed.effort(for: allowed.defaultLevel) == "b")
    }

    @Test("a block that declares no efforts still outranks the flat spelling")
    func keylessBlockOutranksFlatSpelling() throws {
        // The app's reading of a block WITHOUT a vocabulary is "no effort control", not "ask the
        // siblings" (Qwen3.6 stamps are keyless because the template ignores the kwarg), and a
        // top-level `reasoning_effort_supported: false` says it outright.
        let flat = #","reasoning_efforts":["x","y","z"],"reasoning_effort_default":"z""#
        let keyless = try chatCapability(chat: #""reasoning":{"supported":true}"# + flat)
        #expect(keyless.efforts.isEmpty)
        let denied = try chatCapability(
            chat: String(flat.dropFirst()),
            extra: #""reasoning":{"supported":true,"reasoning_effort_supported":false},"#)
        #expect(denied.efforts.isEmpty)
    }

    @Test("chat.reasoning_default still names the default, as before the loader owned the read")
    func siblingReasoningDefault() throws {
        // The default that goes with `reasoning_values`: a SIBLING of `chat.reasoning`, in the same
        // place as the vocabulary it belongs to. Reading the vocabulary and not this left such a
        // bundle at its weakest level.
        let c = try chatCapability(
            chat: #""reasoning_values":["a","b","c"],"reasoning_default":"c""#)
        #expect(c.efforts == ["a", "b", "c"])
        #expect(c.effort(for: c.defaultLevel) == "c")
    }

    @Test("a block's own default outranks chat.reasoning_default")
    func blockDefaultOutranksSiblingDefault() throws {
        let effort = try chatCapability(chat: #"""
            "reasoning":{"reasoning_effort_levels":["a","b","c"],"default_effort":"a"},
            "reasoning_default":"c"
            """#)
        #expect(effort.effort(for: effort.defaultLevel) == "a")
        let mode = try chatCapability(chat: #"""
            "reasoning":{"modes":["a","b","c"],"default_mode":"b"},"reasoning_default":"c"
            """#)
        #expect(mode.effort(for: mode.defaultLevel) == "b")
    }

    @Test("the flat default belongs to the flat list and to nothing else")
    func flatDefaultDoesNotLeak() throws {
        // `reasoning_effort_default` is the flat list's default. With no flat list to go with it
        // it names nothing, and with a vocabulary from `reasoning_values` it must not become
        // that vocabulary's default either.
        let alone = try chatCapability(
            chat: #""reasoning_values":["p","q","r"],"reasoning_effort_default":"r""#)
        #expect(alone.efforts == ["p", "q", "r"])
        #expect(alone.effort(for: alone.defaultLevel) == "p")
        // `q` is in both lists, so a flat default that leaked would land on a real level here.
        let outranked = try chatCapability(chat: #"""
            "reasoning_values":["p","q","r"],
            "reasoning_efforts":["x","q","z"],"reasoning_effort_default":"q"
            """#)
        #expect(outranked.efforts == ["p", "q", "r"])
        #expect(outranked.effort(for: outranked.defaultLevel) == "p")
    }

    @Test("a bundle that already read reasoning_values keeps reading it over the flat spelling")
    func valuesOutrankFlatSpelling() throws {
        // The flat spelling fills a gap; it does not change what an existing bundle meant.
        let c = try chatCapability(
            chat: #""reasoning_values":["p","q","r"],"reasoning_efforts":["x","y","z"]"#)
        #expect(c.efforts == ["p", "q", "r"])
    }

    @Test("a flat list that is not an effort scale is not one")
    func flatSpellingNeedsAScale() throws {
        // One entry is not a vocabulary (the resolver's own threshold), and a non-array is a
        // malformed stamp rather than something to coerce.
        #expect(try chatCapability(chat: #""reasoning_efforts":["high"]"#).efforts.isEmpty)
        #expect(try chatCapability(chat: #""reasoning_efforts":"low,high,max""#).efforts.isEmpty)
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

/// The declarations the Osaurus app reads (`DeclaredReasoningEffort`) beyond a plain effort list: a
/// reasoning block that names each mode's kwargs, `capabilities.reasoning_modes`, and the flat
/// `reasoning_efforts` list in either of its placements. Each real bundle's fixture is its own
/// `jang_config.json`, cut to the keys that matter.
extension ReasoningCapabilityBundleTests {

    private func jangCapability(
        _ jang: String, config: String = #"{"model_type":"unknown_family"}"#, template: String? = nil
    ) throws -> ReasoningCapability {
        var files = ["config.json": config, "jang_config.json": jang]
        if let template { files["chat_template.jinja"] = template }
        return ReasoningCapability.forModel(at: try bundle(files))
    }

    private func jangCapability(
        _ jang: [String: Any], config: String = #"{"model_type":"unknown_family"}"#,
        template: String? = nil
    ) throws -> ReasoningCapability {
        let data = try JSONSerialization.data(withJSONObject: jang, options: [.sortedKeys])
        return try jangCapability(String(decoding: data, as: UTF8.self), config: config, template: template)
    }

    // MARK: Two modes, off spelled as an effort (Diplodocus)

    /// `OsaurusAI/Diplodocus-preview-JANG_6M`: its `chat.reasoning` minus the prose `note`, and the
    /// flat list it also stamps under `capabilities`. The template reads
    /// `reasoning_effort | default('high')`, renders a closed empty think block for `low`, and never
    /// reads `enable_thinking`. `k2_horizon` has no family policy.
    private static func diplodocusBlock() -> [String: Any] {
        [
            "supported": true, "parser": "k2_horizon", "default_enabled": true,
            "default_mode": "think", "modes": ["think", "no_think"],
            "template_flag": "reasoning_effort",
            "mode_kwargs": [
                "think": ["reasoning_effort": "high"], "no_think": ["reasoning_effort": "low"],
            ] as [String: Any],
            "supported_reasoning_efforts": ["low", "high"],
            "reasoning_effort_transport": "chat_template_kwarg",
        ]
    }

    private func diplodocus(
        _ block: [String: Any] = ReasoningCapabilityBundleTests.diplodocusBlock()
    ) throws -> ReasoningCapability {
        try jangCapability(
            [
                "chat": ["reasoning": block] as [String: Any],
                "capabilities": [
                    "reasoning_efforts": ["low", "high"], "supports_reasoning_toggle": true,
                    "supports_thinking": true,
                ] as [String: Any],
            ],
            config: #"{"model_type":"k2_horizon"}"#,
            template: "{%- set effort = reasoning_effort | default('high') -%}")
    }

    @Test("a think/no_think block whose off is an effort is off plus one level, not two levels")
    func twoModeBlockIsOffAndOn() throws {
        // Reading `supported_reasoning_efforts` alone would make `low` a THINKING level, and this
        // template renders `low` as off.
        let c = try diplodocus()
        #expect(c.levels == [0, 1])
        #expect(c.defaultLevel == 1)
        #expect(c.efforts == ["high"])
        #expect(c.offEffort == "low")
        #expect(c.templateKey == "reasoning_effort")
        #expect(c.source == .declared)
        #expect(c.levelKwargs.isEmpty)
        #expect(sent(c, level: 0) == ["reasoning_effort": "string:low"])
        #expect(sent(c, level: 1) == ["reasoning_effort": "string:high"])
        #expect(sent(c, level: 2) == nil)
    }

    @Test("a two-mode block defaulting to no_think defaults to off")
    func twoModeDefaultOff() throws {
        var block = Self.diplodocusBlock()
        block["default_mode"] = "no_think"
        let c = try diplodocus(block)
        #expect(c.levels == [0, 1])
        #expect(c.defaultLevel == 0)
    }

    /// Each argument breaks exactly one of the conditions under which the Osaurus app reads a block
    /// as two modes, and gives what the block then means as a plain list.
    @Test("a two-mode block that breaks any one condition is a plain effort list", arguments: [
        "template_flag", "transport", "modes-set", "modes-count", "think-extra-kwarg",
        "no_think-extra-kwarg", "effort-not-a-string", "on-off-the-ladder", "same-effort",
        "supported-set", "default-mode",
    ])
    func twoModeConditions(broken: String) throws {
        var block = Self.diplodocusBlock()
        let expectedEfforts: [String]
        let expectedLevels: [Int]
        switch broken {
        case "template_flag":
            block["template_flag"] = "enable_thinking"
            (expectedEfforts, expectedLevels) = (["low", "high"], [0, 1, 2])
        case "transport":
            // Not a plain list either: the block's own effort travels some other way.
            block["reasoning_effort_transport"] = "request_body"
            (expectedEfforts, expectedLevels) = ([], [1])
        case "modes-set":
            block["modes"] = ["think", "chat"]
            (expectedEfforts, expectedLevels) = (["low", "high"], [0, 1, 2])
        case "modes-count":
            block["modes"] = ["think", "no_think", "think"]
            (expectedEfforts, expectedLevels) = (["low", "high"], [0, 1, 2])
        case "think-extra-kwarg":
            block["mode_kwargs"] = [
                "think": ["reasoning_effort": "high", "enable_thinking": true] as [String: Any],
                "no_think": ["reasoning_effort": "low"],
            ] as [String: Any]
            (expectedEfforts, expectedLevels) = (["low", "high"], [0, 1, 2])
        case "no_think-extra-kwarg":
            block["mode_kwargs"] = [
                "think": ["reasoning_effort": "high"],
                "no_think": ["reasoning_effort": "low", "enable_thinking": false] as [String: Any],
            ] as [String: Any]
            (expectedEfforts, expectedLevels) = (["low", "high"], [0, 1, 2])
        case "effort-not-a-string":
            block["mode_kwargs"] = [
                "think": ["reasoning_effort": true], "no_think": ["reasoning_effort": "low"],
            ] as [String: Any]
            (expectedEfforts, expectedLevels) = (["low", "high"], [0, 1, 2])
        case "on-off-the-ladder":
            block["mode_kwargs"] = [
                "think": ["reasoning_effort": "turbo"], "no_think": ["reasoning_effort": "low"],
            ] as [String: Any]
            block["supported_reasoning_efforts"] = ["low", "turbo"]
            (expectedEfforts, expectedLevels) = (["low", "turbo"], [0, 1, 2])
        case "same-effort":
            block["mode_kwargs"] = [
                "think": ["reasoning_effort": "high"], "no_think": ["reasoning_effort": "high"],
            ] as [String: Any]
            block["supported_reasoning_efforts"] = ["high"]
            // One entry is below the resolver's threshold for a scale; the template has no toggle.
            (expectedEfforts, expectedLevels) = ([], [1])
        case "supported-set":
            block["supported_reasoning_efforts"] = ["low", "medium", "high"]
            (expectedEfforts, expectedLevels) = (["low", "medium", "high"], [0, 1, 2, 3])
        case "default-mode":
            block["default_mode"] = "auto"
            (expectedEfforts, expectedLevels) = (["low", "high"], [0, 1, 2])
        default:
            Issue.record("unknown case \(broken)")
            return
        }
        let c = try diplodocus(block)
        #expect(c.efforts == expectedEfforts, "\(broken)")
        #expect(c.levels == expectedLevels, "\(broken)")
        #expect(c.offEffort == nil, "\(broken)")
    }

    // MARK: A nested block's effort list, and its transport

    @Test("a nested block reads supported_reasoning_efforts and default_reasoning_effort")
    func nestedSupportedEfforts() throws {
        let c = try jangCapability(#"""
            {"chat":{"reasoning":{"supported":true,
              "supported_reasoning_efforts":["low","medium","xhigh"],
              "default_reasoning_effort":"xhigh",
              "reasoning_effort_transport":"chat_template_kwarg"}}}
            """#)
        #expect(c.efforts == ["low", "medium", "xhigh"])
        #expect(c.levels == [0, 1, 2, 3])
        #expect(c.defaultLevel == 3)
        #expect(c.source == .declared)
    }

    @Test("a block whose effort travels some other way declares no effort scale", arguments: [
        #"{"chat":{"reasoning":{"supported_reasoning_efforts":["low","medium","xhigh"],"reasoning_effort_transport":"request_body"}}}"#,
        #"{"chat":{"reasoning":{"reasoning_effort_levels":["low","high","max"],"reasoning_effort_transport":"api_field"}}}"#,
        #"{"reasoning":{"supported_reasoning_efforts":["low","medium","xhigh"],"reasoning_effort_transport":"python_encoder"}}"#,
    ])
    func blockTransportRejected(jang: String) throws {
        #expect(try jangCapability(jang).efforts.isEmpty, "\(jang)")
    }

    @Test("a block's transport is compared trimmed and lowercased")
    func blockTransportNormalised() throws {
        let c = try jangCapability(#"""
            {"chat":{"reasoning":{"supported_reasoning_efforts":["low","medium","xhigh"],
              "reasoning_effort_transport":" Chat_Template_Kwarg "}}}
            """#)
        #expect(c.efforts == ["low", "medium", "xhigh"])
    }

    // MARK: capabilities.reasoning_modes (Nemotron 3.5)

    /// `OsaurusAI/Nemotron-3.5-Bolt-74B-A9B-24GB-JANGH2` (and 3.5 Super): no reasoning block, and a
    /// mode table whose kwargs are not efforts at all. `nemotron_h` has no family policy.
    private static let nemotronModes =
        #""reasoning_modes":{"on":{"enable_thinking":true},"medium":{"enable_thinking":true,"medium_effort":true},"off":{"enable_thinking":false}}"#

    private func nemotron(
        modes: String = nemotronModes, thinkingDefault: String = "true", chatExtra: String = ""
    ) throws -> ReasoningCapability {
        try jangCapability(
            #"{"chat":{"thinking":{"supported":true,"template_flag":"enable_thinking","default":\#(thinkingDefault)}\#(chatExtra)},"#
                + #""capabilities":{\#(modes),"supports_thinking":true}}"#,
            config: #"{"model_type":"nemotron_h"}"#,
            template: "{%- if enable_thinking -%}<think>{%- else -%}<think></think>{%- endif -%}")
    }

    @Test("reasoning_modes are levels whose kwargs are sent exactly as declared")
    func reasoningModes() throws {
        let c = try nemotron()
        #expect(c.levels == [0, 1, 2])
        #expect(c.defaultLevel == 2)
        #expect(c.efforts == ["medium", "high"])
        #expect(c.source == .declared)
        #expect(c.offEffort == nil)
        #expect(c.templateKey == nil)
        #expect(c.levelKwargs == [
            0: ["enable_thinking": .bool(false)],
            1: ["enable_thinking": .bool(true), "medium_effort": .bool(true)],
            2: ["enable_thinking": .bool(true)],
        ])
        #expect(sent(c, level: 0) == ["enable_thinking": "bool:false"])
        #expect(sent(c, level: 1) == ["enable_thinking": "bool:true", "medium_effort": "bool:true"])
        #expect(sent(c, level: 2) == ["enable_thinking": "bool:true"])
        #expect(sent(c, level: 3) == nil)
        // The caller's own context survives; only the declared kwargs are added.
        #expect(sent(c, level: 1, to: ["tool_choice": "required"])
                == ["tool_choice": "string:required", "enable_thinking": "bool:true",
                    "medium_effort": "bool:true"])
    }

    @Test("mode names land on the effort ladder, ascending, whatever their order or case")
    func reasoningModesLadder() throws {
        let c = try nemotron(modes: #"""
            "reasoning_modes":{"XHigh":{"reasoning_effort":"xhigh"},"minimal":{"reasoning_effort":"minimal"}," On ":{"reasoning_effort":"high"}}
            """#)
        #expect(c.efforts == ["minimal", "high", "xhigh"])
        #expect(c.levels == [1, 2, 3])
        #expect(c.defaultLevel == 2, "on is high, and high is the default")
        #expect(sent(c, level: 3) == ["reasoning_effort": "string:xhigh"])
        #expect(sent(c, level: 0) == nil, "no off mode, no level 0")
    }

    @Test("chat.thinking.default false makes the off mode the default")
    func reasoningModesDefaultOff() throws {
        #expect(try nemotron(thinkingDefault: "false").defaultLevel == 0)
        // No off mode to default to: no declared default, so the weakest thinking level.
        let noOff = try nemotron(
            modes: #""reasoning_modes":{"on":{"enable_thinking":true},"medium":{"medium_effort":true}}"#,
            thinkingDefault: "false")
        #expect(noOff.levels == [1, 2])
        #expect(noOff.defaultLevel == 1)
        // A prose default is not `false`.
        #expect(try nemotron(thinkingDefault: #""on (high)""#).defaultLevel == 2)
    }

    @Test("without a high mode the default is the strongest declared")
    func reasoningModesDefaultStrongest() throws {
        let c = try nemotron(modes: #"""
            "reasoning_modes":{"low":{"reasoning_effort":"low"},"medium":{"reasoning_effort":"medium"},"off":{"reasoning_effort":"none"}}
            """#)
        #expect(c.levels == [0, 1, 2])
        #expect(c.defaultLevel == 2)
    }

    /// Any one flaw voids the whole table, because a partly understood contract must not be half
    /// applied, and resolution moves on to the flat lists (`chat.reasoning_efforts` here).
    @Test("a reasoning_modes table with any flaw declares nothing", arguments: [
        #""reasoning_modes":{"on":{"enable_thinking":true},"turbo":{"enable_thinking":true}}"#,
        #""reasoning_modes":{"on":{"enable_thinking":true},"none":{"enable_thinking":false}}"#,
        #""reasoning_modes":{"on":{"enable_thinking":1}}"#,
        // A bad value beside good modes: dropping just that mode would leave a half table.
        #""reasoning_modes":{"on":{"enable_thinking":true},"medium":{"enable_thinking":true,"medium_effort":1}}"#,
        #""reasoning_modes":{"on":{"enable_thinking":null}}"#,
        #""reasoning_modes":{"on":{"enable_thinking":{"x":true}}}"#,
        #""reasoning_modes":{"on":{}}"#,
        #""reasoning_modes":{"on":true}"#,
        #""reasoning_modes":{"on":{"enable_thinking":true},"high":{"enable_thinking":true}}"#,
        #""reasoning_modes":{"off":{"enable_thinking":false}}"#,
        #""reasoning_modes":{}"#,
    ])
    func reasoningModesVoided(modes: String) throws {
        let c = try nemotron(
            modes: modes, chatExtra: #","reasoning_efforts":["low","high","max"]"#)
        #expect(c.levelKwargs.isEmpty, "\(modes)")
        #expect(c.efforts == ["low", "high", "max"], "\(modes)")
    }

    @Test("reasoning_modes outrank both flat lists")
    func reasoningModesOutrankFlatLists() throws {
        let c = try jangCapability(
            #"{"chat":{"reasoning_efforts":["low","high","max"]},"#
                + #""capabilities":{\#(Self.nemotronModes),"reasoning_efforts":["low","high"]}}"#)
        #expect(c.efforts == ["medium", "high"])
        #expect(c.levelKwargs.count == 3)
    }

    @Test("any reasoning block outranks reasoning_modes")
    func blockOutranksReasoningModes() throws {
        let c = try jangCapability(
            #"{"chat":{"reasoning":{"supported":true}},"capabilities":{\#(Self.nemotronModes)}}"#)
        #expect(c.levelKwargs.isEmpty)
        #expect(c.efforts.isEmpty)
    }

    @Test("reasoning_values keeps meaning what it meant over reasoning_modes")
    func valuesOutrankReasoningModes() throws {
        let c = try jangCapability(
            #"{"chat":{"reasoning_values":["p","q","r"]},"capabilities":{\#(Self.nemotronModes)}}"#)
        #expect(c.efforts == ["p", "q", "r"])
        #expect(c.levelKwargs.isEmpty)
    }

    @Test("a family policy still outranks every declaration")
    func policyOutranksDeclaredContracts() throws {
        let hy3 = try jangCapability(
            ["chat": ["reasoning": Self.diplodocusBlock()] as [String: Any]],
            config: #"{"model_type":"hy_v3"}"#)
        #expect(hy3.source == .policy)
        #expect(hy3.levels == [0, 1, 2])
        #expect(hy3.offEffort == "no_think")
        let gptOss = try jangCapability(
            #"{"capabilities":{\#(Self.nemotronModes)}}"#, config: #"{"model_type":"gpt_oss"}"#)
        #expect(gptOss.source == .policy)
        #expect(gptOss.levels == [0, 1, 2, 3])
        #expect(gptOss.levelKwargs.isEmpty)
    }

    // MARK: The flat list, in either placement

    @Test("a flat list under capabilities is read when chat has none")
    func flatCapabilitiesPlacement() throws {
        let c = try jangCapability(
            #"{"capabilities":{"reasoning_efforts":["low","high","max"],"reasoning_effort_default":"max"}}"#)
        #expect(c.efforts == ["low", "high", "max"])
        #expect(c.levels == [1, 2, 3])
        #expect(c.defaultLevel == 3)
        #expect(c.source == .declared)
        let alias = try jangCapability(
            #"{"capabilities":{"reasoning_efforts":["low","high","max"],"default_reasoning_effort":"high"}}"#)
        #expect(alias.defaultLevel == 2)
    }

    @Test("chat's flat list outranks capabilities'")
    func flatChatBeforeCapabilities() throws {
        let c = try jangCapability(
            #"{"chat":{"reasoning_efforts":["low","high"]},"#
                + #""capabilities":{"reasoning_efforts":["low","medium","xhigh"]}}"#)
        #expect(c.efforts == ["low", "high"])
    }

    @Test("the first placement with a list decides, even when it declares nothing usable", arguments: [
        #""reasoning_efforts":["low","LOW"]"#,
        #""reasoning_efforts":[]"#,
        #""reasoning_efforts":["low","high"],"reasoning_effort_transport":"request_body""#,
    ])
    func firstFlatPlacementDecides(chat: String) throws {
        let c = try jangCapability(
            #"{"chat":{\#(chat)},"capabilities":{"reasoning_efforts":["low","medium","xhigh"]}}"#)
        #expect(c.efforts.isEmpty, "\(chat) fell through to capabilities: \(c.efforts)")
    }

    @Test("a chat entry that is not a list is no list, so capabilities' is read")
    func flatChatNonListIsAbsent() throws {
        let c = try jangCapability(
            #"{"chat":{"reasoning_efforts":"low,high"},"#
                + #""capabilities":{"reasoning_efforts":["low","medium","xhigh"]}}"#)
        #expect(c.efforts == ["low", "medium", "xhigh"])
    }

    @Test("a flat list's transport is compared trimmed and lowercased, and only when declared")
    func flatTransport() throws {
        let normalised = try jangCapability(
            #"{"chat":{"reasoning_efforts":["low","high","max"],"reasoning_effort_transport":" CHAT_TEMPLATE_KWARG "}}"#)
        #expect(normalised.efforts == ["low", "high", "max"])
        let rejected = try jangCapability(
            #"{"capabilities":{"reasoning_efforts":["low","high","max"],"reasoning_effort_transport":"api_field"}}"#)
        #expect(rejected.efforts.isEmpty)
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
