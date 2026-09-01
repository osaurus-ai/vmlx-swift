import Foundation

// A model family can already be TOLD its reasoning effort. It cannot be ASKED what efforts it has.
//
// `DeepseekV4ReasoningPolicy` and `Hy3ReasoningTemplateContext` both normalise an effort string for
// their family, each with a private vocabulary and its own aliases, and neither can answer "what are
// your levels", "what is your default", or "can you stop reasoning at all". So every consumer
// re-derives that from the bundle — and gets it wrong, because the bundle is only half the evidence:
// the other half is here, in the family-specific code that knows which template key to populate.
//
// This file adds the missing question. It is deliberately additive: the existing policies keep their
// entry points and gain a declaration, and the two families that had no policy at all get one.

/// What one model family knows about its own reasoning controls.
public protocol ReasoningEffortPolicy: Sendable {
    /// Whether this policy governs the given `model_type`.
    static func applies(to modelType: String?) -> Bool

    /// The chat-template variable this family reads — `reasoning_effort` for most, but not all:
    /// Muse Glimmer's template reads `reasoning_strength`, which is why nothing was populating it.
    static var templateKey: String { get }

    /// Thinking efforts, ordered weakest to strongest, in the family's own vocabulary.
    /// Excludes any value that MEANS "off"; that is ``offEffort``.
    static var efforts: [String] { get }

    /// What the template falls back to when the key is absent. Knowing this is the difference
    /// between "we ran at level 1" and "we ran at whatever the template chose".
    static var defaultEffort: String { get }

    /// Whether reasoning can be turned off at all. Separate from ``offEffort`` because families
    /// disagree about HOW: DeepSeek V4 uses `enable_thinking = false` (its own deprecation note says
    /// so), while Hunyuan 3 expresses off as the effort value `no_think`.
    static var supportsDisabling: Bool { get }

    /// The effort value meaning "do not reason", when the family expresses off that way. `nil` when
    /// off is reached by another mechanism, or not at all.
    static var offEffort: String? { get }

    /// Map a caller's string onto this family's vocabulary, or `nil` if it has no meaning here.
    /// Map a raw effort string onto this family's vocabulary, or `nil` when it names nothing.
    ///
    /// RULE FOR CONFORMANCES: where the family already has a normaliser, this must DELEGATE to it
    /// rather than restate the accepted set. `DeepseekV4EffortPolicy` and `Hy3EffortPolicy` do, and
    /// the reason generalises — a second copy of a vocabulary drifts from the first the moment either
    /// changes, and the copy that drifts is the one no caller is testing. A family with no existing
    /// normaliser gets the default implementation below, which is derived from `efforts` and
    /// `offEffort` and so cannot drift from them either.
    static func normalize(_ raw: String) -> String?
}

extension ReasoningEffortPolicy {
    /// Case-insensitive exact match against the declared vocabulary. Families with aliases override.
    public static func normalize(_ raw: String) -> String? {
        let v = raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if let offEffort, v == offEffort { return offEffort }
        return efforts.first { $0 == v }
    }

    public static var capability: ReasoningCapability {
        ReasoningCapability(efforts: efforts, defaultEffort: defaultEffort,
                            supportsDisabling: supportsDisabling, offEffort: offEffort,
                            templateKey: templateKey)
    }
}

// MARK: - The querying surface

/// A model's reasoning controls as one integer scale: `0` = not reasoning, `1…N` = increasing effort.
///
/// The scale is the useful shape because it is what a caller has to render or record. The whole space
/// falls out of which levels are present, with no separate flags:
///
/// | levels | meaning |
/// |---|---|
/// | `[0]` | a non-thinking model |
/// | `[1]` | think-only, one effort |
/// | `[0, 1]` | thinking on/off |
/// | `[1, 2, 3, 4]` | think-only with four efforts |
/// | `[0, 1, 2, 3]` | on/off plus three efforts |
///
/// "Can this model stop thinking?" is `levels.contains(0)`, rather than a string comparison against
/// `["no_think", "none", "off", "false", "instruct", …]` maintained separately in each family.
///
/// WHAT COUNTS AS THINKING. A thinking model is one that exposes a thinking CHANNEL — a chain-of-
/// thought trace something can separate from the answer. What it does internally is not the question
/// and is not observable anyway. So:
///
/// - no level selection, no on/off, **no trace** → `[0]`, non-thinking, whatever it does internally;
/// - no level selection, no on/off, **but a trace** → `[1]`, thinking, just not controllable;
/// - an on/off control → `[0, 1]`; an effort vocabulary → `[0…N]` or `[1…N]`.
///
/// `VibeThinker-3B` is the first case: its template is plain ChatML with no delimiter and no toggle,
/// so the name promises reasoning the interface does not expose. Reporting `[1]` would offer a caller
/// a channel it cannot read.
///
/// Trace detection is a heuristic over the markers models actually ship — a `<think>` delimiter in the
/// template, a `reasoning_content` field in the tokenizer config, a declared reasoning parser. A model
/// with a novel delimiter reads as non-thinking until its marker is added here, which is a gap worth
/// knowing about rather than a claim to trust blindly.
public struct ReasoningCapability: Sendable, Equatable {
    /// Contiguous and ascending. Contains `0` exactly when reasoning can be disabled.
    public let levels: [Int]
    /// The level the model uses when asked for nothing in particular.
    public let defaultLevel: Int
    /// Effort names for levels `1…N`, in order. Empty when the family has no effort vocabulary.
    public let efforts: [String]
    /// The effort value meaning off, when the family expresses off that way.
    public let offEffort: String?
    /// The chat-template variable to populate, when there is one.
    public let templateKey: String?
    /// Where this capability came from. See ``Source``.
    public let source: Source

    /// How much the rest of this value is worth trusting.
    ///
    /// The fallbacks are the part most likely to rot: a model that declares nothing still resolves to
    /// one of three definite shapes, and if that guess is wrong the caller has no way to notice —
    /// which is the same failure mode as the bug this type exists to prevent. A caller must be able to
    /// tell a declaration from a guess, so the distinction is carried here rather than being flattened
    /// away. Render a control confidently for ``policy`` and ``declared``; treat ``inferred`` as a
    /// best effort that may be silently wrong.
    public enum Source: String, Sendable, Equatable, CaseIterable {
        /// A registered family policy — the strongest, because it also knows the template key.
        case policy
        /// The model's own declaration, from its bundle.
        case declared
        /// Our guess, from artefacts that merely suggest a shape.
        case inferred
    }

    public init(efforts: [String], defaultEffort: String?, supportsDisabling: Bool,
                offEffort: String? = nil, templateKey: String? = nil,
                source: Source = .policy) {
        self.source = source
        self.efforts = efforts
        self.offEffort = offEffort
        self.templateKey = templateKey
        let thinkingLevels = efforts.isEmpty ? [1] : Array(1...efforts.count)
        self.levels = supportsDisabling ? [0] + thinkingLevels : thinkingLevels
        // A family may DEFAULT to not reasoning — Hunyuan 3's template falls back to `no_think` when
        // nothing is injected — so the default can legitimately be level 0.
        if let defaultEffort, let offEffort, defaultEffort == offEffort, supportsDisabling {
            self.defaultLevel = 0
        } else if let defaultEffort, let i = efforts.firstIndex(of: defaultEffort) {
            self.defaultLevel = i + 1
        } else {
            // A default the family does not list is a declaration bug, not a level: fall back to the
            // weakest thinking level rather than inventing an index for it.
            self.defaultLevel = thinkingLevels.first ?? 0
        }
    }

    private init(levels: [Int], defaultLevel: Int, source: Source = .inferred) {
        self.levels = levels
        self.defaultLevel = defaultLevel
        self.efforts = []
        self.offEffort = nil
        self.templateKey = nil
        self.source = source
    }

    /// The same shape, relabelled as coming from the model's own declaration.
    ///
    /// The three constants below are shared instances and default to `inferred`, which is right for
    /// the rungs that deduce a shape. A rung reached BECAUSE the model said so needs the same levels
    /// with different provenance, and this makes that one word rather than a fourth constant.
    func declaring() -> ReasoningCapability {
        ReasoningCapability(levels: levels, defaultLevel: defaultLevel, source: .declared)
    }

    /// The effort string for a level, or `nil` for level 0 and for out-of-range levels.
    public func effort(for level: Int) -> String? {
        guard level > 0, level <= efforts.count else { return nil }
        return efforts[level - 1]
    }

    public var canDisableReasoning: Bool { levels.contains(0) }
    public var maximumLevel: Int { levels.max() ?? 0 }
    /// True when the model cannot be stopped from reasoning — the case a caller must not label as
    /// "thinking off" merely because it asked for it.
    public var isThinkingOnly: Bool { !levels.contains(0) }

    // MARK: Fallbacks for models that declare nothing
    //
    // Most models are here. They have no policy, no `chat.reasoning` block, and no effort vocabulary
    // — but they are still one of three things, and a caller still has to know which.

    /// Never reasons. `[0]`
    public static let nonThinking = ReasoningCapability(levels: [0], defaultLevel: 0)
    /// Always reasons, with no control over how much. `[1]`
    public static let thinkingOnly = ReasoningCapability(levels: [1], defaultLevel: 1)
    /// Reasoning can be switched on and off, with no effort scale. `[0, 1]`
    public static let thinkingToggle = ReasoningCapability(levels: [0, 1], defaultLevel: 1)
}

// MARK: - The families

/// gpt-oss / harmony. The template emits `"Reasoning: " + reasoning_effort` and defaults to `medium`
/// when the key is absent — so an undeclared gpt-oss runs at medium while a caller believes it asked
/// for the lowest level. Reasoning is disabled through `enable_thinking`, not through an effort value.
public enum GptOssReasoningPolicy: ReasoningEffortPolicy {
    public static func applies(to modelType: String?) -> Bool { modelType?.lowercased() == "gpt_oss" }
    public static let templateKey = "reasoning_effort"
    public static let efforts = ["low", "medium", "high"]
    public static let defaultEffort = "medium"
    public static let supportsDisabling = true
    public static let offEffort: String? = nil
}

/// Muse Glimmer. Its template reads `reasoning_strength`, NOT `reasoning_effort`:
///
///     {%- set rs = reasoning_strength if reasoning_strength is defined and reasoning_strength else 'high' -%}
///     {{- 'Reasoning strength: ' + rs + '.' -}}
///
/// Nothing in this library populated that key, so every request ran at `high` regardless of what was
/// asked. The model also cannot be stopped from reasoning — it declares no off mode — which is why
/// its levels start at 1.
public enum MuseGlimmerReasoningPolicy: ReasoningEffortPolicy {
    public static func applies(to modelType: String?) -> Bool {
        modelType?.lowercased().hasPrefix("muse_glimmer") ?? false
    }
    public static let templateKey = "reasoning_strength"
    /// Delegated, per the rule on `normalize` below and at the maintainer's suggestion on #305: the
    /// vocabulary lives in `MuseGlimmerReasoningTemplateContext`, which is the code that ENFORCES it
    /// when building the prompt. Restating the four words here would be a second copy that drifts the
    /// moment either changes, and the copy that drifts is the one no caller is testing.
    public static var efforts: [String] { MuseGlimmerReasoningTemplateContext.levels }
    public static let defaultEffort = "high"
    public static let supportsDisabling = false
    public static let offEffort: String? = nil

    /// Delegates rather than restating the aliases — the rule this protocol states for conformances
    /// whose family already has a normaliser, applied to the third such family.
    public static func normalize(_ raw: String) -> String? {
        MuseGlimmerReasoningTemplateContext.normalized(raw)
    }
}

/// Bailing. The odd one out, and the reason ``ReasoningEffortPolicy/templateKey`` is named for what it
/// SETS rather than how it travels: this family has no reasoning variable in its template at all.
/// `BailingThinkingTemplateContext` reads `enable_thinking` and injects a system directive —
/// "detailed thinking on" / "detailed thinking off" — into the messages instead.
///
/// So the key is `enable_thinking`, there is no effort vocabulary, and the delivery mechanism is the
/// adapter's business. A capability describes WHAT to set; how it reaches the model is not its
/// concern. Without this declaration the family would fall through to the template scan, find no
/// `enable_thinking` (there is none to find), and be reported as having no control — while an adapter
/// stands ready to act on exactly that key.
public enum BailingReasoningPolicy: ReasoningEffortPolicy {
    public static func applies(to modelType: String?) -> Bool {
        modelType?.lowercased().hasPrefix("bailing") ?? false
    }
    public static let templateKey = "enable_thinking"
    public static let efforts: [String] = []       // on/off only, no effort scale
    public static let defaultEffort = ""
    public static let supportsDisabling = true
    public static let offEffort: String? = nil
}

/// DeepSeek V4. `low/high/max` per `DeepseekV4ReasoningPolicy.normalizedReasoningEffort`; the direct
/// (non-reasoning) rail is `enable_thinking = false`, per that type's own deprecation note.
public enum DeepseekV4EffortPolicy: ReasoningEffortPolicy {
    public static func applies(to modelType: String?) -> Bool {
        DeepseekV4ReasoningPolicy.isDeepseekV4(modelType: modelType)
    }
    public static let templateKey = "reasoning_effort"
    public static let efforts = ["low", "high", "max"]
    public static let defaultEffort = "low"
    public static let supportsDisabling = true
    public static let offEffort: String? = nil
    public static func normalize(_ raw: String) -> String? {
        DeepseekV4ReasoningPolicy.normalizedReasoningEffort(raw)
    }
}

/// Hunyuan 3. Expresses off as an effort VALUE (`no_think`) rather than through `enable_thinking`,
/// and accepts aliases — `none|off → no_think`, `minimal|medium → low`.
public enum Hy3EffortPolicy: ReasoningEffortPolicy {
    /// The predicate lives here rather than in `Hy3ReasoningTemplateContext`, alongside
    /// `DeepseekV4ReasoningPolicy.isDeepseekV4`, so both families answer "is this you?" from the same
    /// module. `Hy3ReasoningTemplateContext.applies(to:)` delegates here — one definition, not two
    /// that can drift.
    public static func applies(to modelType: String?) -> Bool {
        guard let t = modelType?.lowercased() else { return false }
        return t == "hy_v3" || t.hasPrefix("hy_v3_") || t.hasPrefix("hy3") || t.hasPrefix("hunyuan")
    }
    public static let templateKey = "reasoning_effort"
    public static let efforts = ["low", "high"]
    /// The template injects nothing when neither key is present, and its OWN default is `no_think` —
    /// so this family defaults to not reasoning, which is level 0.
    public static let defaultEffort = "no_think"
    public static let supportsDisabling = true
    public static let offEffort: String? = "no_think"
    public static func normalize(_ raw: String) -> String? {
        switch raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
        case "no_think", "low", "high": return raw.lowercased()
        case "none", "off": return "no_think"
        case "minimal", "medium": return "low"
        default: return nil
        }
    }
}

// MARK: - Resolution

public enum ReasoningCapabilityRegistry {
    /// Families with a declaration of their own. Order is irrelevant: `applies(to:)` is disjoint.
    public static let policies: [any ReasoningEffortPolicy.Type] = [
        GptOssReasoningPolicy.self,
        MuseGlimmerReasoningPolicy.self,
        BailingReasoningPolicy.self,
        DeepseekV4EffortPolicy.self,
        Hy3EffortPolicy.self,
    ]

    public static func policy(for modelType: String?) -> (any ReasoningEffortPolicy.Type)? {
        policies.first { $0.applies(to: modelType) }
    }

    /// What this model can do, from the strongest evidence available.
    ///
    /// Order matters and is the point: a family policy knows the template key and the real
    /// vocabulary; a bundle declaration is the model author speaking about their own model; the
    /// remaining signals only distinguish the three shapes a model with no effort scale can take.
    ///
    /// - Parameters:
    ///   - modelType: `config.json`'s `model_type`.
    ///   - declaredEfforts: effort names from the bundle, if it declares any.
    ///   - declaredDefault: the bundle's default effort, if any.
    ///   - supportsThinking: `capabilities.supports_thinking`, when the bundle states it.
    ///   - templateMentionsToggle: whether the chat template reads `enable_thinking`.
    ///   - reasoningSignal: evidence the model reasons even though nothing declares a control — a
    ///     `<think>` delimiter or a `reasoning_content` field. Only consulted when no control is
    ///     expressible at all, to choose between think-only and non-thinking.
    public static func capability(
        modelType: String?,
        declaredEfforts: [String] = [],
        declaredDefault: String? = nil,
        supportsThinking: Bool? = nil,
        templateMentionsToggle: Bool = false,
        reasoningSignal: Bool = false
    ) -> ReasoningCapability {
        // 1. A family policy: the only source that also knows which template key to populate.
        if let p = policy(for: modelType) { return p.capability }

        // 2. The model's own declaration. Trusted over anything we could infer, and the reason a
        //    model that starts shipping a contract immediately supersedes our guess about it.
        if declaredEfforts.count > 1 {
            return ReasoningCapability(efforts: declaredEfforts, defaultEffort: declaredDefault,
                                       supportsDisabling: supportsThinking != false,
                                       source: .declared)
        }

        // 3. No effort scale. Still one of three distinct things, and a caller must not be left to
        //    assume: a non-thinking model asked to think, and a think-only model asked to stop, both
        //    silently return something the caller did not request.
        // `supports_thinking` is the model's own word, so it stays a DECLARATION even though it
        // carries no vocabulary. The template scan and the marker sweep below are ours, and are
        // marked `inferred` so a caller can tell a stated shape from a deduced one.
        if supportsThinking == false { return .nonThinking.declaring() }
        if templateMentionsToggle { return .thinkingToggle }
        if supportsThinking == true { return .thinkingOnly.declaring() }

        // 4. No control is expressible — the template does not read `enable_thinking`, so reasoning
        //    cannot be switched from the prompt whatever we claim. What remains is whether the model
        //    reasons at all, and `reasoningSignal` answers that from the artefacts that betray it: a
        //    `<think>` delimiter in the template, or a `reasoning_content` field in the tokenizer
        //    config. Reporting a TOGGLE here would be the worst of the three, because it advertises a
        //    control that does not exist — on 41 of 108 local bundles, spanning a judge model that
        //    never reasons and a "Thinking" build that always does.
        return reasoningSignal ? .thinkingOnly : .nonThinking
    }
}

// MARK: - Reading it off a bundle

extension ReasoningCapability {

    /// What this model on disk can do, from its own files.
    ///
    /// Three sources, because model authors have not converged on one:
    ///   - `jang_config.json` → `chat.reasoning`, the closest thing to a declaration;
    ///   - `config.json` → `capabilities.supports_thinking`, a blunt yes/no some bundles ship;
    ///   - `chat_template.jinja`, which is the ground truth about what the prompt can express.
    ///
    /// The template is consulted LAST but is not the weakest evidence — it is the only source that
    /// cannot lie, since it is the code that actually runs. It is used to settle the shape when
    /// nothing else declares an effort scale.
    public static func forModel(at directory: URL) -> ReasoningCapability {
        let cfg = json(at: directory.appendingPathComponent("config.json"))
        let modelType = (cfg?["model_type"] as? String)
            ?? ((cfg?["text_config"] as? [String: Any])?["model_type"] as? String)

        let jang = json(at: directory.appendingPathComponent("jang_config.json"))
        let chat = jang?["chat"] as? [String: Any]
        let reasoning = chat?["reasoning"] as? [String: Any]

        // Efforts, in the order the field names actually appear in the wild. `reasoning_effort_levels`
        // is the canonical spelling; `reasoning_values` is a SIBLING of `chat.reasoning` that Muse
        // Glimmer uses; and some bundles put the effort vocabulary in `modes`, which is only
        // distinguishable from real on/off modes by the absence of an off-mode name. Reading just the
        // first spelling is how a model declaring four levels was read as declaring none.
        var efforts = reasoning?["reasoning_effort_levels"] as? [String] ?? []
        if efforts.count < 2, let values = chat?["reasoning_values"] as? [String] { efforts = values }
        if efforts.count < 2, let modes = reasoning?["modes"] as? [String],
           !modes.contains(where: { offModeNames.contains($0.lowercased()) }) {
            efforts = modes
        }

        let declaredDefault = (reasoning?["default_effort"] as? String)
            ?? (reasoning?["default_mode"] as? String)
            ?? (chat?["reasoning_default"] as? String)

        // `supported: false` is a statement about reasoning; `supports_thinking` is the same statement
        // spelled differently in another file. Either one saying no is decisive.
        var supportsThinking: Bool?
        if let r = reasoning?["supported"] as? Bool { supportsThinking = r }
        if let caps = cfg?["capabilities"] as? [String: Any],
           let s = caps["supports_thinking"] as? Bool {
            supportsThinking = (supportsThinking ?? true) && s
        }

        // THE TEMPLATE LIVES IN ONE OF TWO PLACES, and reading only the first is how half a
        // collection reads as having no template at all. Hugging Face bundles ship it either as a
        // standalone `chat_template.jinja` or as a `chat_template` field inside
        // `tokenizer_config.json`; across the models surveyed for this the split was almost exactly
        // even. A bundle whose template we cannot find reads as "no toggle expressible", which then
        // reads as "non-thinking" — a confident answer derived from having looked in one place.
        let tokenizerCfg = (try? String(contentsOf: directory.appendingPathComponent("tokenizer_config.json"),
                                        encoding: .utf8)) ?? ""
        let template = (try? String(contentsOf: directory.appendingPathComponent("chat_template.jinja"),
                                    encoding: .utf8)) ?? tokenizerCfg
        let mentionsToggle = template.contains("enable_thinking")

        // Does it reason at all, when nothing says so? The model betrays itself in its own artefacts,
        // but families do not agree on how, so one marker is not enough. `<think>` is the common
        // spelling; Mistral's newer instructs use `[THINK]`; harmony emits a `<|channel|>` header; and
        // a `reasoning_content` field or a declared parser says so outright.
        //
        // `Ministral-3-3B-Instruct-2512` is why this is a LIST. It ships `[THINK]` and nothing else —
        // no `enable_thinking`, no `<think>` — so a single-marker check reported a reasoning model as
        // having no thinking channel at all. The list is extensible by construction and still a
        // heuristic: a family with a novel delimiter reads as non-thinking until its marker is added.
        let reasoningMarkers = ["<think>", "[THINK]", "<|channel|>", "reasoning_content"]
        let artefacts = template + tokenizerCfg
        let signal = reasoningMarkers.contains { artefacts.contains($0) }
            || (cfg?["capabilities"] as? [String: Any])?["reasoning_parser"] != nil

        return ReasoningCapabilityRegistry.capability(
            modelType: modelType, declaredEfforts: efforts, declaredDefault: declaredDefault,
            supportsThinking: supportsThinking, templateMentionsToggle: mentionsToggle,
            reasoningSignal: signal)
    }

    /// Names that mean "do not reason" across the families we have seen.
    static let offModeNames: Set<String> = [
        "no_think", "nothink", "non-thinking", "none", "off", "false", "instruct", "chat", "disabled",
    ]

    private static func json(at url: URL) -> [String: Any]? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        return (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
    }
}

// MARK: - Selecting a level

extension ReasoningCapability {

    /// Populate a chat-template context so this model reasons at `level`.
    ///
    /// Level 0 has TWO spellings and the difference is not cosmetic: most families disable reasoning
    /// with `enable_thinking = false`, while Hunyuan 3 ignores that boolean entirely and takes the
    /// effort VALUE `no_think` — passing the wrong one leaves the model reasoning while the caller
    /// believes it asked otherwise. Returns nil when the level is not on this model's scale, so a
    /// caller cannot quietly request something it cannot have.
    public func applying(level: Int, to context: [String: any Sendable] = [:]) -> [String: any Sendable]? {
        guard levels.contains(level) else { return nil }
        var out = context
        if level == 0 {
            if let offEffort, let templateKey { out[templateKey] = offEffort }
            else { out["enable_thinking"] = false }
            return out
        }
        if let templateKey, let name = effort(for: level) { out[templateKey] = name }
        // Only assert the boolean where it means something. A think-only model has no off state, so
        // saying `enable_thinking = true` at it is noise a template may or may not understand.
        if canDisableReasoning { out["enable_thinking"] = true }
        return out
    }

    /// The level a caller means by "thinking on".
    ///
    /// The model's declared default when that is a thinking level, else the weakest one. A family may
    /// legitimately default to NOT reasoning — Hunyuan 3 does — and "thinking at the default" cannot
    /// then mean level 0 without contradicting itself.
    ///
    /// NOTE this is a different quantity from ``defaultLevel``, which is the model's default OPERATING
    /// mode and may be 0. Under the 0…N paradigm ``defaultLevel`` is the one that matters — it is what
    /// a UI preselects — and this is mostly a PRESENTATION hint: which stop on the slider to mark as
    /// the model's own, once the user has moved off 0. It also serves the older shape of control, a
    /// think/no-think switch beside a separate effort selector, where the two are genuinely distinct
    /// settings. Do not treat it as a second default to reconcile with the first.
    public var defaultThinkingLevel: Int {
        defaultLevel >= 1 ? defaultLevel : (levels.first { $0 >= 1 } ?? 1)
    }
}
