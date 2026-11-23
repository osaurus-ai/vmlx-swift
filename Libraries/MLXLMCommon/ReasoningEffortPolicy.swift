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
    /// The exact chat-template kwargs for each level, for a model that declares them that way
    /// (`capabilities.reasoning_modes`: Nemotron 3.5's `medium` is `enable_thinking` AND
    /// `medium_effort`, which no single ``templateKey`` can express). When non-empty it is the
    /// whole contract: ``applying(level:to:)`` sends `levelKwargs[level]` and nothing else, and
    /// ``efforts`` only names the levels. Empty for every other capability.
    public let levelKwargs: [Int: [String: TemplateValue]]

    /// A declared chat-template kwarg value. Booleans and strings are the only kinds a mode
    /// declaration may carry; anything else voids the declaration rather than being coerced.
    public enum TemplateValue: Sendable, Equatable {
        case bool(Bool)
        case string(String)

        /// The value as a template context holds it.
        var contextValue: any Sendable {
            switch self {
            case .bool(let value): return value
            case .string(let value): return value
            }
        }
    }

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
        self.levelKwargs = [:]
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

    private init(levels: [Int], defaultLevel: Int, efforts: [String] = [],
                 levelKwargs: [Int: [String: TemplateValue]] = [:], source: Source = .inferred) {
        self.levels = levels
        self.defaultLevel = defaultLevel
        self.efforts = efforts
        self.offEffort = nil
        self.templateKey = nil
        self.levelKwargs = levelKwargs
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

/// DeepSeek-V4.1. Numeric 1-100 budget contract; the host ladder below maps
/// onto it in the V4.1 encoder (`minimal`=25 … `max`=100). Absent effort renders
/// low (50) by vmlx convention (the bundle reference default is high=75).
public enum DeepseekV41EffortPolicy: ReasoningEffortPolicy {
    public static func applies(to modelType: String?) -> Bool {
        DeepseekV4ReasoningPolicy.isDeepseekV41(modelType: modelType)
    }
    public static let templateKey = "reasoning_effort"
    public static let efforts = DeepseekV4ReasoningPolicy.deepseekV41Efforts
    public static let defaultEffort = "low"
    public static let supportsDisabling = true
    public static let offEffort: String? = nil
    public static func normalize(_ raw: String) -> String? {
        DeepseekV4ReasoningPolicy.normalizedDeepseekV41ReasoningEffort(raw).map {
            String(describing: $0)
        }
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

/// GLM-5.x (`glm5`, `glm5_next`). The control is a SYSTEM LINE, not a boolean, and the template
/// states the whole contract on its first two lines:
///
///     {%- set effective_reasoning_effort = reasoning_effort
///           if reasoning_effort is defined and reasoning_effort in ['low', 'high']
///           else 'max' -%}
///     {%- if effective_reasoning_effort is not none -%}
///     <|system|>Reasoning Effort: {{ effective_reasoning_effort | capitalize }}{%- endif -%}
///
/// So: `low` and `high` are honoured verbatim, ANYTHING else — including an absent key — resolves
/// to `max`, and there is no off state. The generation prompt then ends `<|assistant|><think>`
/// unconditionally, which is why the model reasons whatever you ask of it.
///
/// WITHOUT THIS POLICY the family fell through to the "no effort scale" branch, because a bundle's
/// effort vocabulary is only ever read from `jang_config.json` and neither `zai-org/GLM-5.3-Flash`
/// nor its first JANG conversion shipped one — the declaration lived in the template, where
/// nothing looked. (OsaurusAI's JANGH2 build does declare it, as the flat `chat.reasoning_efforts`
/// `[low, high, max]` with default `max`, and this policy agrees with that declaration on the
/// efforts, on the default and, since a flat list names no off, on there being none.)
/// `ReasoningCapability.forModel(at:)` therefore reported `levels=[1] efforts=[] templateKey=nil`,
/// `applying(level:)` returned an EMPTY context for the only level it admitted, no `reasoning_effort`
/// was ever populated, and every request — including one for the LEAST reasoning — took the
/// template's `else 'max'` branch. A control the model genuinely has was invisible, and the failure
/// was silent in both directions: nothing errored, and the rows recorded a level nobody ran at.
public enum Glm5EffortPolicy: ReasoningEffortPolicy {
    /// `glm5`, `glm5_next`, `glm_5_*` — but NOT `glm4*`, whose templates carry `enable_thinking`
    /// instead and are governed by the toggle path.
    public static func applies(to modelType: String?) -> Bool {
        guard let t = modelType?.lowercased() else { return false }
        let compact = t.replacingOccurrences(of: "_", with: "")
            .replacingOccurrences(of: "-", with: "")
            .replacingOccurrences(of: ".", with: "")
        return compact.hasPrefix("glm5")
    }
    public static let templateKey = "reasoning_effort"
    /// Weakest to strongest. `max` is in the list because the template reaches it by FALLING
    /// THROUGH: passing the literal string `max` is not in `['low', 'high']`, so the else-branch
    /// fires and the system line reads `Reasoning Effort: Max`. Sending it explicitly is therefore
    /// both correct and, unlike sending nothing, self-documenting in the recorded context.
    public static let efforts = ["low", "high", "max"]
    /// The template's OWN fallback when the key is absent — which is what every run got before this
    /// policy existed.
    public static let defaultEffort = "max"
    /// No off state: the template emits a Reasoning Effort line unconditionally and prefills
    /// `<think>` at the generation prompt. A caller asking for level 0 gets the nearest thing the
    /// model can do, and the harness records that it was substituted.
    public static let supportsDisabling = false
    public static let offEffort: String? = nil
    // No `normalize` override: the default is derived from `efforts` and cannot drift from it, and
    // this family has no aliases worth inventing. `medium` in particular is NOT mapped — low/high/max
    // has no middle rung, and guessing one would put a run at an effort nobody chose.
}

// MARK: - Resolution

public enum ReasoningCapabilityRegistry {
    /// Families with a declaration of their own. Order is irrelevant: `applies(to:)` is disjoint.
    public static let policies: [any ReasoningEffortPolicy.Type] = [
        GptOssReasoningPolicy.self,
        MuseGlimmerReasoningPolicy.self,
        BailingReasoningPolicy.self,
        DeepseekV4EffortPolicy.self,
        DeepseekV41EffortPolicy.self,
        Hy3EffortPolicy.self,
        Glm5EffortPolicy.self,
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
    ///   - declaredSupportsDisabling: whether the bundle's own scale includes an off. `nil` leaves
    ///     it to `supportsThinking`, as the spellings that never say. A scale that LISTS its
    ///     efforts (the flat `chat.reasoning_efforts`) says it outright: `false` when no entry is
    ///     an off, `true` when one is.
    ///   - declaredOffEffort: the effort VALUE that means off, for a scale that names one.
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
        reasoningSignal: Bool = false,
        declaredTemplateKey: String? = nil,
        declaredSupportsDisabling: Bool? = nil,
        declaredOffEffort: String? = nil
    ) -> ReasoningCapability {
        // 1. A family policy: the only source that also knows which template key to populate.
        if let p = policy(for: modelType) { return p.capability }

        // 2. The model's own declaration. Trusted over anything we could infer, and the reason a
        //    model that starts shipping a contract immediately supersedes our guess about it.
        if declaredEfforts.count > 1 || (declaredEfforts.count == 1 && declaredOffEffort != nil) {
            return ReasoningCapability(
                efforts: declaredEfforts, defaultEffort: declaredDefault,
                supportsDisabling: (declaredSupportsDisabling ?? true) && supportsThinking != false,
                offEffort: declaredOffEffort, templateKey: declaredTemplateKey,
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
    ///   - the JANG config, from `jang_config.json` or embedded in `config.json` and read through
    ///     `JangLoader`: a `chat.reasoning` or top-level `reasoning` block (a two-mode contract
    ///     first, then its effort list), or for a bundle without one the `chat` siblings that stand
    ///     in for it (`reasoning_values` and its `reasoning_default`), then
    ///     `capabilities.reasoning_modes`, then OsaurusAI's flat `reasoning_efforts` under `chat`
    ///     and then under `capabilities`. The closest thing to a declaration;
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

        // ONE READER, NOT TWO. The config lives in two places: a bundle may ship `jang_config.json`
        // beside its weights, or embed the same block inside `config.json`. `JangLoader.loadConfig`
        // already resolves both (file first, then embedded), parses the nested `chat.reasoning` and
        // the top-level `reasoning` shapes, and hands back a typed value. Digging the raw JSON here
        // made a second reader of exactly the same declaration, so every spelling and every flag had
        // to be taught to both, and the two could disagree about one bundle.
        //
        // The loader THROWS for a bundle with no JANG config, which is a precondition and not a
        // defect: `configNotFound` distinguishes "no JANG config" from "a config declaring nothing",
        // and callers branch on `isJangModel` to keep that distinction. This function must answer for
        // non-JANG bundles too, so it asks the loader only where the loader applies and falls back to
        // its own evidence otherwise. That keeps the two CONTRACTS separate while removing the
        // duplicated PARSING, which is where the drift was.
        let jangConfig: JangConfig? = JangLoader.isJangModel(at: directory)
            ? try? JangLoader.loadConfig(at: directory)
            : nil
        let chatCfg = jangConfig?.chat
        let reasoningCfg = chatCfg?.reasoning

        // Two declarations are COMPLETE contracts rather than effort lists, and are resolved here
        // in full: a reasoning block naming each mode's kwargs (`nativeThinkingModes`), and
        // `capabilities.reasoning_modes` (`reasoningModes`). Neither fits the resolver's "an effort
        // list of two or more" rung, and neither may be read as one.
        var contract: ReasoningCapability?
        var efforts: [String] = []
        var declaredDefault: String?
        var declaredSupportsDisabling: Bool?
        var declaredOffEffort: String?

        if let reasoningCfg {
            // A block is the bundle's whole statement about efforts: one without a vocabulary means
            // "no effort control", and a top-level `reasoning_effort_supported: false` says so
            // outright, so the siblings below must not contradict either. The Osaurus app reads
            // the two-mode contract FIRST, because the same block also lists its two effort values
            // (`supported_reasoning_efforts: [low, high]`), and read as a plain list that would make
            // `low` a thinking level when the template renders it as off.
            contract = nativeThinkingModes(reasoningCfg)

            // A block whose effort travels some other way than as a template kwarg has no effort
            // control we can exercise, whichever spelling its vocabulary uses.
            if contract == nil, transportIsChatTemplateKwarg(reasoningCfg.reasoningEffortTransport) {
                // Effort vocabulary, in the order the spellings appear in the wild. The typed path
                // carries `reasoning_effort_levels` and `supported_reasoning_efforts` in the same
                // field; `reasoning_values` is a SIBLING of `chat.reasoning` that Muse Glimmer uses;
                // and some bundles put the vocabulary in `modes`, distinguishable from real on/off
                // modes only by the absence of an off-mode name.
                efforts = (reasoningCfg.reasoningEffortLevels ?? []).compactMap { $0 }
                if efforts.count < 2, let values = chatCfg?.reasoningValues { efforts = values }
                if efforts.count < 2, let modes = reasoningCfg.modes,
                   !modes.contains(where: { offModeNames.contains($0.lowercased()) }) {
                    efforts = modes
                }
                // A block's own default first, then the sibling `chat.reasoning_default` that goes
                // with `reasoning_values`: the order the config was read in before the loader owned
                // the read.
                declaredDefault = reasoningCfg.defaultEffort ?? reasoningCfg.defaultMode
                    ?? chatCfg?.reasoningDefault
            }
        } else {
            if let values = chatCfg?.reasoningValues { efforts = values }
            declaredDefault = chatCfg?.reasoningDefault

            // With NO reasoning block, the Osaurus app looks in three more places, in this order:
            // `capabilities.reasoning_modes`, then the flat `reasoning_efforts` under `chat`, then
            // under `capabilities`. They come after `reasoning_values`, which the app does not read,
            // so a bundle that already read `reasoning_values` keeps meaning what it meant. A
            // `reasoning_modes` table with any flaw declares nothing, and resolution moves on.
            //
            // A flat list is the model's whole scale, so it also says whether there is an off: only
            // an entry that IS one (see `flatEffortScale`) gives the model a level 0. The resolver's
            // usual "reasoning can be switched off unless `supports_thinking` says no" does not
            // apply, or `[low, high, max]` would grow an off level. Conversely, `[off, high]`
            // is a complete scale: the template takes the named effort for each of its two modes.
            // The default travels with the declaration that supplied the vocabulary.
            if efforts.count < 2 {
                contract = reasoningModes(jangConfig?.capabilities?.reasoningModes,
                                          thinkingDefault: chatCfg?.thinking?.defaultEnabled)
                if contract == nil,
                   let flat = firstFlatEffortScale(chat: chatCfg, capabilities: jangConfig?.capabilities) {
                    efforts = flat.efforts
                    declaredDefault = flat.defaultEffort
                    declaredSupportsDisabling = flat.offEffort != nil
                    declaredOffEffort = flat.offEffort
                }
            }
        }

        // A family policy still outranks every declaration, as it does inside `capability(...)`.
        if let contract {
            return ReasoningCapabilityRegistry.policy(for: modelType)?.capability ?? contract
        }

        // `supported: false` is a statement about reasoning; `supports_thinking` is the same statement
        // spelled differently in another file. Either one saying no is decisive.
        var supportsThinking: Bool?
        if let r = reasoningCfg?.supported { supportsThinking = r }
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

        // WHICH KWARG carries the effort. A bundle declaring
        // `reasoning_effort_transport: "chat_template_kwarg"` names the transport and not the key,
        // and a capability with efforts but NO key is visible and inert: `applying(level:)` writes
        // the effort only when a key exists, so the request goes out without it and the template
        // picks its own default — the same end state as declaring nothing.
        //
        // Take the name from the template, which states it outright, rather than assuming the
        // common spelling. `reasoning_effort` is what most families read, but not all: Muse Glimmer
        // reads `reasoning_strength`, so a guess would be indistinguishable from a correct read
        // right up until it silently wasn't. A family policy still outranks this; it is consulted
        // first, inside `capability(...)`.
        let inferredKey = templateEffortKey(in: template)

        return ReasoningCapabilityRegistry.capability(
            modelType: modelType, declaredEfforts: efforts, declaredDefault: declaredDefault,
            supportsThinking: supportsThinking, templateMentionsToggle: mentionsToggle,
            reasoningSignal: signal, declaredTemplateKey: inferredKey,
            declaredSupportsDisabling: declaredSupportsDisabling,
            declaredOffEffort: declaredOffEffort)
    }

    /// A flat `chat.reasoning_efforts` list as Osaurus reads it (`DeclaredReasoningEffort`): each
    /// entry trimmed and lowercased, the list refused whole if any entry is empty or repeated, and
    /// an entry that names "off" taken out of the efforts and kept as the off effort.
    ///
    /// Refused rather than repaired because a repaired list is a vocabulary the bundle never wrote.
    /// So is a list with two off names (the app would have two ways to say the same rail). The
    /// default must be one of the entries, off included; anything else is no declared default.
    static func flatEffortScale(
        _ raw: [String], default rawDefault: String?
    ) -> (efforts: [String], offEffort: String?, defaultEffort: String?)? {
        let entries = raw.map { $0.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() }
        guard entries.allSatisfy({ !$0.isEmpty }), Set(entries).count == entries.count else {
            return nil
        }
        let offs = entries.filter { offModeNames.contains($0) }
        guard offs.count <= 1 else { return nil }
        let declared = rawDefault?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return (entries.filter { !offModeNames.contains($0) }, offs.first,
                declared.flatMap { entries.contains($0) ? $0 : nil })
    }

    /// The flat `reasoning_efforts` list from the first placement that HAS one, `chat` and then
    /// `capabilities`, as the Osaurus app looks. That first list decides even when it declares
    /// nothing usable (malformed, or its effort travels some other way than as a template
    /// kwarg): the bundle made its statement there, and a second list elsewhere is not a fallback
    /// for it. A value that is not a list of strings is no list, and the next placement is read.
    static func firstFlatEffortScale(
        chat: JangChatConfig?, capabilities: JangCapabilities?
    ) -> (efforts: [String], offEffort: String?, defaultEffort: String?)? {
        let placements: [(list: [String]?, defaultEffort: String?, transport: String?)] = [
            (chat?.reasoningEfforts, chat?.reasoningEffortDefault, chat?.reasoningEffortTransport),
            (capabilities?.reasoningEfforts, capabilities?.reasoningEffortDefault,
             capabilities?.reasoningEffortTransport),
        ]
        guard let first = placements.first(where: { $0.list != nil }), let list = first.list,
              transportIsChatTemplateKwarg(first.transport)
        else { return nil }
        return flatEffortScale(list, default: first.defaultEffort)
    }

    /// Whether a declared `reasoning_effort_transport` is one a chat template can receive. Any
    /// value other than `chat_template_kwarg` (trimmed, lowercased) names a delivery this library
    /// cannot make, so the effort control is treated as absent rather than sent the wrong way. An
    /// absent key keeps the kwarg convention, which is how most bundles that declare efforts
    /// expect them.
    static func transportIsChatTemplateKwarg(_ declared: String?) -> Bool {
        guard let declared else { return true }
        return declared.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            == "chat_template_kwarg"
    }

    /// The effort names the Osaurus app knows, weakest to strongest, across the conventions it meets
    /// (OpenAI's o-series and GPT-5.x, Codex's `ultra`, the local families). A declared mode is
    /// placed on this ladder by name, which is what orders its levels.
    static let effortLadder = ["minimal", "low", "medium", "high", "xhigh", "max", "ultra"]

    /// A reasoning block that selects thinking with an effort VALUE for each of two modes, as
    /// `OsaurusAI/Diplodocus-preview-JANG_6M` stamps it:
    ///
    ///     "template_flag": "reasoning_effort", "modes": ["think", "no_think"],
    ///     "mode_kwargs": {"think": {"reasoning_effort": "high"},
    ///                     "no_think": {"reasoning_effort": "low"}},
    ///     "supported_reasoning_efforts": ["low", "high"], "default_mode": "think"
    ///
    /// That is off plus one thinking level, with off spelled `reasoning_effort = low`: its template
    /// renders `low` as a closed, empty think block and never reads `enable_thinking`. The same
    /// block's effort list, read alone, would be a two-level scale with `low` as a thinking level,
    /// which is why this is checked first. Every condition is the Osaurus app's
    /// (`DeclaredReasoningEffort.nativeThinkingModes`), and failing any one leaves the block to be
    /// read as a plain list: a bare `[low, high]` does not by itself mean off and on.
    static func nativeThinkingModes(_ reasoning: JangChatReasoning) -> ReasoningCapability? {
        guard reasoning.templateFlag == "reasoning_effort",
              transportIsChatTemplateKwarg(reasoning.reasoningEffortTransport),
              let modes = reasoning.modes, Set(modes) == ["think", "no_think"], modes.count == 2,
              let declared = reasoning.reasoningEffortLevels, !declared.contains(nil),
              let on = reasoning.modeKwargs?["think"], on.count == 1,
              let off = reasoning.modeKwargs?["no_think"], off.count == 1,
              case .string(let onEffort)? = on["reasoning_effort"], effortLadder.contains(onEffort),
              case .string(let offEffort)? = off["reasoning_effort"], offEffort != onEffort,
              Set(declared.compactMap { $0 }) == [onEffort, offEffort],
              let defaultMode = reasoning.defaultMode, modes.contains(defaultMode)
        else { return nil }
        return ReasoningCapability(
            efforts: [onEffort], defaultEffort: defaultMode == "think" ? onEffort : offEffort,
            supportsDisabling: true, offEffort: offEffort, templateKey: "reasoning_effort",
            source: .declared)
    }

    /// `capabilities.reasoning_modes`: `{mode: {kwarg: bool | string}}`, as Nemotron 3.5 stamps it
    /// (`on` = `enable_thinking`, `medium` = that plus `medium_effort`, `off` = `enable_thinking:
    /// false`). Each mode is a level whose kwargs are sent verbatim (``levelKwargs``).
    ///
    /// Mode names are placed on ``effortLadder``: `on` is `high`, `off` is level 0, and any other
    /// name must be on the ladder itself; thinking levels ascend by ladder rank. A name off the
    /// ladder, a mode with no kwargs, two names for one level, or no thinking level at all voids
    /// the whole table (as does a kwarg value that is not a boolean or string, which the loader
    /// refuses): a partly understood contract must not be half applied.
    ///
    /// The default is off when `chat.thinking.default` is `false` and an off mode exists, else
    /// `high` if declared, else the strongest. `false` with no off mode leaves no declared default,
    /// which is the weakest thinking level here as everywhere else.
    static func reasoningModes(
        _ table: [String: [String: TemplateValue]]?, thinkingDefault: Bool?
    ) -> ReasoningCapability? {
        guard let table, !table.isEmpty else { return nil }
        var off: [String: TemplateValue]?
        var thinking: [String: [String: TemplateValue]] = [:]
        for (name, kwargs) in table {
            guard !kwargs.isEmpty else { return nil }
            let normalized = name.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            if normalized == "off" {
                guard off == nil else { return nil }
                off = kwargs
                continue
            }
            let effort = normalized == "on" ? "high" : normalized
            guard effortLadder.contains(effort), thinking[effort] == nil else { return nil }
            thinking[effort] = kwargs
        }
        let efforts = effortLadder.filter { thinking[$0] != nil }
        guard !efforts.isEmpty else { return nil }

        var levelKwargs: [Int: [String: TemplateValue]] = [:]
        if let off { levelKwargs[0] = off }
        for (index, effort) in efforts.enumerated() { levelKwargs[index + 1] = thinking[effort] }
        let defaultLevel: Int
        if thinkingDefault == false {
            defaultLevel = off == nil ? 1 : 0
        } else {
            defaultLevel = (efforts.firstIndex(of: "high") ?? efforts.count - 1) + 1
        }
        return ReasoningCapability(
            levels: levelKwargs.keys.sorted(), defaultLevel: defaultLevel, efforts: efforts,
            levelKwargs: levelKwargs, source: .declared)
    }

    /// Which chat-template variable carries the reasoning effort, read from the template itself.
    ///
    /// A bundle declaring `reasoning_effort_transport: "chat_template_kwarg"` names the transport and
    /// not the key, and a capability with efforts but no key is visible and INERT — `applying(level:)`
    /// writes the effort only when a key exists.
    ///
    /// Assuming `reasoning_effort` would be right for most families and silently wrong for the one
    /// that already differs, so the name is taken as evidence rather than as a default. Ordered most
    /// specific first: a template mentioning both should be read as using the distinctive one.
    ///
    /// EXTRACTED FOR TESTABILITY. Inline, this could not be falsified against the local zoo: the only
    /// `reasoning_strength` bundle also has a family policy, which is consulted first, so a test that
    /// went through `forModel(at:)` passed whether this read the template or hardcoded the common
    /// spelling — a saturated comparison that asserted nothing.
    public static func templateEffortKey(in template: String) -> String? {
        ["reasoning_strength", "reasoning_effort"].first { template.contains($0) }
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
    ///
    /// A model that declares each level's kwargs (``levelKwargs``) gets exactly those, added to
    /// `context`, and nothing this function would otherwise have inferred.
    public func applying(level: Int, to context: [String: any Sendable] = [:]) -> [String: any Sendable]? {
        guard levels.contains(level) else { return nil }
        var out = context
        if !levelKwargs.isEmpty {
            guard let kwargs = levelKwargs[level] else { return nil }
            for (key, value) in kwargs { out[key] = value.contextValue }
            return out
        }
        // Off as an effort VALUE means the template takes its whole instruction from that key.
        let offIsAnEffort = offEffort != nil && templateKey != nil
        if level == 0 {
            if let offEffort, let templateKey { out[templateKey] = offEffort }
            else { out["enable_thinking"] = false }
            return out
        }
        if let templateKey, let name = effort(for: level) { out[templateKey] = name }
        // Only assert the boolean where it means something. A think-only model has no off state, so
        // saying `enable_thinking = true` at it is noise a template may or may not understand. Nor
        // where off is an effort value: such a model was never told `enable_thinking = false`, its
        // contract is the effort key alone (Hunyuan 3's template ignores the boolean, Diplodocus's
        // never reads it), and the Osaurus app sends that contract's kwargs and nothing besides.
        if canDisableReasoning && !offIsAnEffort { out["enable_thinking"] = true }
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
