// Copyright © 2024-2026 Jinho Jang (eric@jangq.ai)
// JANG format support for mlx-swift-lm

import Foundation
import MLX

// MARK: - Config File Names

/// Primary JANG config file name.
public let jangConfigFileName = "jang_config.json"

/// Legacy config file names to search for (fallback only).
public let jangConfigFileNames = [
    "jang_config.json",
    "jjqf_config.json",
    "jang_cfg.json",
    "mxq_config.json",
]

// MARK: - JANG Config Structs

/// Quantization settings from jang_config.json `quantization` block.
public struct JangQuantization: Sendable, Equatable {
    public let method: String
    public let profile: String
    public let targetBits: Float
    public let actualBits: Float
    public let blockSize: Int
    public let bitWidthsUsed: [Int]
    public let quantizationScheme: String
    public let quantizationBackend: String

    public init(
        method: String = "jang-importance",
        profile: String = "JANG_2S",
        targetBits: Float = 2.5,
        actualBits: Float = 2.85,
        blockSize: Int = 64,
        // 2026-05-01: default to empty so `isAuthoritativeJang` checks
        // (`!bitWidthsUsed.isEmpty`) correctly distinguish bundles that
        // actually have JANG quant metadata vs bundles that just default-
        // construct this struct (e.g. mxfp4 bundles whose `jang_config.json`
        // has only a `mxfp4` field, not a `quantization` field). With the
        // old default `[2, 4, 6]`, mxfp4 bundles were misclassified as
        // authoritative-JANG, ignoring config.json's quantization.bits and
        // landing on `defaultBits = bitWidthsUsed.min() = 2` which produced
        // the (bits=2, gs=64) shape-walk interpretation that doubled the
        // embed_tokens output dim (24576 instead of hiddenSize=12288) and
        // crashed the next RMSNorm — observed on
        // Mistral-Medium-3.5-128B-mxfp4. Real JANG bundles always populate
        // bit_widths_used explicitly, so this only changes behavior for
        // non-JANG bundles.
        bitWidthsUsed: [Int] = [],
        quantizationScheme: String = "asymmetric",
        quantizationBackend: String = "mx.quantize"
    ) {
        self.method = method
        self.profile = profile
        self.targetBits = targetBits
        self.actualBits = actualBits
        self.blockSize = blockSize
        self.bitWidthsUsed = bitWidthsUsed
        self.quantizationScheme = quantizationScheme
        self.quantizationBackend = quantizationBackend
    }
}

/// Source model info from jang_config.json `source_model` block.
public struct JangSourceModel: Sendable, Equatable {
    public let name: String
    public let org: String
    public let architecture: String
    public let dtype: String
    public let parameters: String

    public init(
        name: String = "",
        org: String = "",
        architecture: String = "",
        dtype: String = "bfloat16",
        parameters: String = "0"
    ) {
        self.name = name
        self.org = org
        self.architecture = architecture
        self.dtype = dtype
        self.parameters = parameters
    }

    public var parameterCount: Int { Int(parameters) ?? 0 }

    /// HuggingFace canonical repo id, e.g. `MiniMaxAI/MiniMax-M2.7`. Empty if
    /// either `org` or `name` is missing.
    public var huggingFaceRepoID: String {
        guard !org.isEmpty, !name.isEmpty else { return "" }
        return "\(org)/\(name)"
    }
}

/// Architecture info from jang_config.json `architecture` block.
public struct JangArchitecture: Sendable, Equatable {
    public let type: String
    public let attention: String
    public let hasVision: Bool
    public let hasSSM: Bool
    public let hasMoE: Bool

    public init(
        type: String = "transformer",
        attention: String = "gqa",
        hasVision: Bool = false,
        hasSSM: Bool = false,
        hasMoE: Bool = false
    ) {
        self.type = type
        self.attention = attention
        self.hasVision = hasVision
        self.hasSSM = hasSSM
        self.hasMoE = hasMoE
    }
}

/// Runtime info from jang_config.json `runtime` block.
public struct JangRuntime: Sendable, Equatable {
    public let totalWeightBytes: Int
    public let totalWeightGB: Float
    public let bundleHasMTP: Bool
    public let mtpLayers: Int
    public let mtpMode: MTPRuntimeMode

    public init(
        totalWeightBytes: Int = 0,
        totalWeightGB: Float = 0,
        bundleHasMTP: Bool = false,
        mtpLayers: Int = 0,
        mtpMode: MTPRuntimeMode = .none
    ) {
        self.totalWeightBytes = totalWeightBytes
        self.totalWeightGB = totalWeightGB
        self.bundleHasMTP = bundleHasMTP
        self.mtpLayers = mtpLayers
        self.mtpMode = mtpMode
    }
}

/// Capability hints stamped into `jang_config.json` by the JANG converter.
///
/// Allows downstream consumers (osaurus, llm-tool, etc.) to pick the right
/// reasoning / tool-call parser without hard-coding per-model branching.
/// All fields are optional — missing values mean "unknown, fall back to
/// model-type heuristics."
///
/// Field naming is intentionally lenient: aliases produced by the JANG
/// converter (e.g. `tool_parser: "qwen"` instead of vmlx's canonical
/// `"xml_function"`) are normalized at consumption time by
/// `ToolCallFormat.fromCapabilityName(_:)` and
/// `ReasoningParser.fromCapabilityName(_:)`.
public struct JangCapabilities: Sendable {
    /// Reasoning-tag style. Known values: `qwen3`, `deepseek_r1`,
    /// `think_xml` (all → `<think>...</think>`); `gemma4` / `harmony`
    /// (Harmony channel envelopes); explicit `mistral4` capability stamps
    /// (`[THINK]...[/THINK]`); `none` / legacy `mistral` / legacy `gemma`
    /// (no reasoning parser). `nil` means unknown.
    public let reasoningParser: String?

    /// Tool-call format. Known values: `qwen`, `qwen3_coder` → `xml_function`;
    /// `minimax` → `minimax_m2`; `glm47`, `deepseek` → `glm4`; `deepseek_v4`
    /// → `dsml`; `gemma4` → `gemma4`; `hy3*` / `hunyuan*` → `hunyuan`;
    /// `nemotron` → `nemotron`; plus any canonical `ToolCallFormat`
    /// rawValue. `nil` means unknown.
    public let toolParser: String?

    /// Whether the model's chat template natively gates `<think>` blocks
    /// behind an `enable_thinking` flag. Consumers may flip this flag to
    /// suppress / require reasoning per request.
    public let thinkInTemplate: Bool?

    /// Whether the model is trained to emit tool calls.
    public let supportsTools: Bool?

    /// Whether the model is trained to emit reasoning blocks.
    public let supportsThinking: Bool?

    /// Explicit text lane support. `nil` means older bundles did not stamp it.
    public let supportsText: Bool?

    /// Explicit still-image / vision lane support. `nil` means older bundles
    /// should fall back to coarse `modality` and model-class evidence.
    public let supportsVision: Bool?

    /// Explicit video lane support. This is intentionally separate from
    /// `supportsVision` because many VLMs accept images but not videos.
    public let supportsVideo: Bool?

    /// Explicit audio lane support. Only Omni-style bundles should stamp this
    /// true; image/video support must not imply audio.
    public let supportsAudio: Bool?

    /// Family bucket for UI/registry grouping (e.g. `qwen3_5`, `gemma4`).
    public let family: String?

    /// `text` or `vision`. Hint for UI affordances; vmlx detects vision
    /// support from the model class itself.
    public let modality: String?

    /// `kv`, `hybrid`, or `mla`. Hint for cache/memory budgeting. vmlx
    /// engine selects the actual cache type from the model class — `mla`
    /// is currently a forward-looking hint (vmlx falls back to standard
    /// KV for MLA models).
    public let cacheType: String?

    /// Speculative-decoding strategy the JANG bundle ships alongside
    /// this target. Known values: `dflash`, `ddtree`, `autoregressive`,
    /// `none`. `nil` means the bundle does not ship a compatible
    /// drafter. Maps to ``DraftStrategy`` via
    /// ``ParserResolution/draftStrategy(capabilities:modelDirectory:)``.
    public let draftStrategy: String?

    /// Path to the drafter checkpoint, RELATIVE to `jang_config.json`.
    /// Typical value: `"drafter/"` (i.e. a subdirectory next to the
    /// target weights). `nil` when `draftStrategy` is absent or `none`.
    public let drafterPath: String?

    /// Branching budget for ``DraftStrategy/ddtree(drafterPath:branchingBudget:blockSize:)``.
    /// Paper recommends 32-64 for greedy, 16-24 for sampling. `nil`
    /// when `draftStrategy != "ddtree"`.
    public let branchingBudget: Int?

    /// Block size the drafter was trained with — must match
    /// `config.json["block_size"]` inside the drafter snapshot. When
    /// present, callers use this to satisfy
    /// ``DraftStrategy/dflash(drafterPath:blockSize:)`` etc.
    public let blockSize: Int?

    public init(
        reasoningParser: String? = nil,
        toolParser: String? = nil,
        thinkInTemplate: Bool? = nil,
        supportsTools: Bool? = nil,
        supportsThinking: Bool? = nil,
        supportsText: Bool? = nil,
        supportsVision: Bool? = nil,
        supportsVideo: Bool? = nil,
        supportsAudio: Bool? = nil,
        family: String? = nil,
        modality: String? = nil,
        cacheType: String? = nil,
        draftStrategy: String? = nil,
        drafterPath: String? = nil,
        branchingBudget: Int? = nil,
        blockSize: Int? = nil
    ) {
        self.reasoningParser = reasoningParser
        self.toolParser = toolParser
        self.thinkInTemplate = thinkInTemplate
        self.supportsTools = supportsTools
        self.supportsThinking = supportsThinking
        self.supportsText = supportsText
        self.supportsVision = supportsVision
        self.supportsVideo = supportsVideo
        self.supportsAudio = supportsAudio
        self.family = family
        self.modality = modality
        self.cacheType = cacheType
        self.draftStrategy = draftStrategy
        self.drafterPath = drafterPath
        self.branchingBudget = branchingBudget
        self.blockSize = blockSize
    }

    /// Source of a parser resolution — used for telemetry and so callers
    /// can log `detection_source=jang_stamped` when the JANG capabilities
    /// stamp wins, vs `detection_source=model_type_heuristic` when the
    /// loader had to fall back.
    public enum ResolutionSource: String, Sendable {
        /// Resolved from `jang_config.json["capabilities"]`.
        case jangStamped = "jang_stamped"
        /// Resolved from `config.json["model_type"]` heuristic (no stamp,
        /// or stamp value was unrecognised).
        case modelTypeHeuristic = "model_type_heuristic"
        /// Resolved from the actual chat template when a legacy stamp is
        /// ambiguous or contradicted by the template protocol.
        case chatTemplate = "chat_template"
        /// Neither stamp nor heuristic resolved a parser.
        case none = "none"
    }
}

/// Convenience facade for resolving parsers with explicit precedence.
///
/// Precedence (per vmlx-swift-lm production contract — matches the
/// Tier-1/Tier-2 split osaurus's engine uses):
/// 1. **JANG stamp wins** when present and value resolves.
/// 2. Otherwise fall back to `model_type` heuristic
///    (`ToolCallFormat.infer(from:)`).
/// 3. Otherwise `nil` (caller can render raw).
///
/// Designed so consumers can call this once and log a single
/// `detection_source=` value for diagnostics.
public enum ParserResolution {

    /// Resolve a `ReasoningParser` for a model.
    ///
    /// - Parameters:
    ///   - capabilities: the `JangCapabilities` block from `jang_config.json`
    ///     (pass `nil` for non-JANG models).
    ///   - modelType: the `model_type` field from `config.json` — used as
    ///     a heuristic fallback when no stamp is present.
    /// - Returns: a parser instance and the source it came from. The
    ///   parser is `nil` for models that don't emit reasoning (legacy
    ///   Mistral/Gemma, Llama, Phi, etc.) — callers should skip parsing and
    ///   stream raw.
    public static func reasoning(
        capabilities: JangCapabilities?,
        modelType: String?,
        chatTemplate: String? = nil
    ) -> (parser: ReasoningParser?, source: JangCapabilities.ResolutionSource) {
        if shouldIgnoreReasoningStamp(capabilities: capabilities, modelType: modelType) {
            if declaresLFM25ThinkingTemplate(modelType: modelType, chatTemplate: chatTemplate) {
                return (
                    ReasoningParser(startInReasoning: false),
                    .chatTemplate
                )
            }
            let stamp = reasoningStampFromModelType(modelType)
            return (
                stamp == "none" ? nil : ReasoningParser.fromCapabilityName(stamp),
                modelType?.isEmpty == false ? .modelTypeHeuristic : .none
            )
        }

        if let cap = capabilities, cap.reasoningParser != nil {
            // Stamped — honour exactly. `nil` is a valid stamp meaning
            // "this model emits no reasoning".
            return (
                ReasoningParser.fromCapabilityName(cap.reasoningParser),
                .jangStamped
            )
        }
        if declaresLFM25ThinkingTemplate(modelType: modelType, chatTemplate: chatTemplate) {
            return (
                ReasoningParser.fromCapabilityName("qwen3"),
                .chatTemplate
            )
        }
        // Template-native `<think>...</think>` envelope. This is the PRECISE
        // reasoning signal: a reasoning-tuned bundle (e.g. VibeThinker, a
        // qwen2-based thinker) carries the literal tag pair in its chat
        // template, while its non-reasoning siblings (Qwen2.5-Instruct, also
        // model_type `qwen2`) do not. The model_type heuristic below cannot
        // separate them, so resolving here keeps `qwen2` OUT of the model_type
        // allowlist (which would over-trigger and route plain answers into the
        // think pane — the original reverse-allowlist bug). Explicit JANG
        // stamps are already honoured above; Gemma's `<|think|>` mode marker
        // has no `</think>` close form, so the harmony path is unaffected.
        if templateDeclaresThinkEnvelope(chatTemplate) {
            return (ReasoningParser.fromCapabilityName("qwen3"), .chatTemplate)
        }
        // Heuristic: delegate to the canonical factory helper so this
        // stays byte-identical with `LLMModelFactory` / `VLMModelFactory`.
        // Historical note: this function previously carried its own
        // reverse-allowlist default that returned a live `ReasoningParser()`
        // for every non-{gemma,mistral} model_type, which drove the LFM2
        // "entire answer routed to .reasoning" bug. Never reintroduce a
        // local default here; `reasoningStampFromModelType` is the sole
        // source of truth.
        let stamp = reasoningStampFromModelType(modelType)
        if stamp == "none" {
            return (nil, modelType?.isEmpty == false ? .modelTypeHeuristic : .none)
        }
        return (
            ReasoningParser.fromCapabilityName(stamp),
            .modelTypeHeuristic
        )
    }

    /// True when the chat template carries a literal `<think>...</think>`
    /// reasoning envelope. This is the precise per-bundle reasoning signal that
    /// distinguishes a reasoning-tuned model from a non-reasoning sibling that
    /// shares its `model_type` (e.g. VibeThinker vs Qwen2.5-Instruct, both
    /// `qwen2`). Mirrors osaurus `LocalReasoningCapability.analyze`.
    static func templateDeclaresThinkEnvelope(_ chatTemplate: String?) -> Bool {
        guard let chatTemplate else { return false }
        return chatTemplate.contains("<think>") && chatTemplate.contains("</think>")
    }

    public static func shouldIgnoreReasoningStamp(
        capabilities: JangCapabilities?,
        modelType: String?
    ) -> Bool {
        guard let capabilities,
              let reasoningParser = capabilities.reasoningParser,
              ReasoningParser.fromCapabilityName(reasoningParser) != nil,
              capabilities.thinkInTemplate == false
        else { return false }

        let family = capabilities.family?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
            .replacingOccurrences(of: "-", with: "_")
            .replacingOccurrences(of: ".", with: "_") ?? ""
        let type = modelType?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
            .replacingOccurrences(of: "-", with: "_")
            .replacingOccurrences(of: ".", with: "_") ?? ""
        let compactType = type.replacingOccurrences(of: "_", with: "")
        let toolParser = capabilities.toolParser?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased() ?? ""

        // LFM2/LFM2.5 JANG bundles are Pythonic-tool models. Some early
        // converted bundles stamped `reasoning_parser=qwen3` while also
        // stamping `think_in_template=false`; trusting that routes normal
        // assistant output into `.reasoning` and prevents tool extraction.
        // Keep the tool parser stamp, but demote the impossible reasoning
        // stamp back to the model-type/template resolver.
        return family.hasPrefix("lfm2")
            || family.contains("lfm")
            || type.hasPrefix("lfm2")
            || compactType.hasPrefix("lfm25")
            || toolParser == "lfm2"
    }

    private static func declaresLFM25ThinkingTemplate(
        modelType: String?,
        chatTemplate: String?
    ) -> Bool {
        guard let modelType, let chatTemplate else { return false }
        let normalized = modelType
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
            .replacingOccurrences(of: "-", with: "_")
            .replacingOccurrences(of: ".", with: "_")
        let compact = normalized.replacingOccurrences(of: "_", with: "")
        guard compact == "lfm2moe" || compact.hasPrefix("lfm25") else {
            return false
        }
        return chatTemplate.contains("<think>")
            && chatTemplate.contains("</think>")
            && chatTemplate.contains("<|tool_call_start|>")
            && chatTemplate.contains("<|tool_call_end|>")
    }

    /// Resolve a `ToolCallFormat` for a model.
    ///
    /// - Parameters:
    ///   - capabilities: stamped capabilities, or `nil`.
    ///   - modelType: `model_type` from `config.json` for heuristic fallback.
    public static func toolCall(
        capabilities: JangCapabilities?,
        modelType: String?,
        chatTemplate: String? = nil
    ) -> (format: ToolCallFormat?, source: JangCapabilities.ResolutionSource) {
        if let cap = capabilities,
            let stamped = ToolCallFormat.fromCapabilityName(cap.toolParser)
        {
            if let templateFormat = templateDeclaredToolCallFormat(chatTemplate),
                shouldPreferTemplateToolCallFormat(
                    templateFormat,
                    stamped: stamped,
                    capabilities: cap,
                    modelType: modelType)
            {
                return (templateFormat, .chatTemplate)
            }
            return (stamped, .jangStamped)
        }
        if let modelType, let inferred = ToolCallFormat.infer(from: modelType) {
            return (inferred, .modelTypeHeuristic)
        }
        // Non-JANG bundle whose `model_type` the heuristic doesn't recognise.
        // The canonical example is plain Qwen3: `infer` deliberately leaves
        // `qwen3` / `qwen3_moe` nil because `qwen3_moe` is shared by BOTH the
        // instruct line (Hermes `<tool_call>{"name":…,"arguments":…}` → .json)
        // and Qwen3-Coder (`<tool_call><function=…><parameter=…>` → .xmlFunction),
        // so model_type alone cannot disambiguate them. Read the model's OWN
        // chat template — the ground truth for what it emits — to recover the
        // format instead of guessing or returning "tools unsupported".
        if let templateFormat = templateDeclaredToolCallFormat(chatTemplate) {
            return (templateFormat, .chatTemplate)
        }
        return (nil, .none)
    }

    private static func templateDeclaredToolCallFormat(_ chatTemplate: String?) -> ToolCallFormat? {
        guard let chatTemplate else { return nil }
        let lower = chatTemplate.lowercased()
        // XML-function envelope: `<tool_call><function=name><parameter=key>…`
        // (Qwen3-Coder, Qwen3.5/3.6, Nemotron-style). Checked BEFORE the bare-JSON
        // rule because a coder template still contains `<tool_call>` but carries
        // `<function=>`/`<parameter=>` bodies rather than a `{"name":…,"arguments":…}`
        // object. Verified against real Qwen3-Coder-30B-A3B `chat_template`.
        if lower.contains("<function=") && lower.contains("<parameter=") {
            return .xmlFunction
        }
        // Hermes bare-JSON envelope: `<tool_call>\n{"name":…,"arguments":…}\n</tool_call>`.
        // Verified against real Qwen3-4B (qwen3) and Qwen3-30B-A3B (qwen3_moe)
        // instruct `chat_template`. The `<arg_key>` exclusion keeps GLM/Ling-style
        // `<arg_key>`/`<arg_value>` templates from matching here.
        if lower.contains("<tool_call>")
            && lower.contains("\"name\"")
            && lower.contains("\"arguments\"")
            && !lower.contains("<arg_key>")
        {
            return .json
        }
        return nil
    }

    private static func shouldPreferTemplateToolCallFormat(
        _ templateFormat: ToolCallFormat,
        stamped: ToolCallFormat,
        capabilities: JangCapabilities,
        modelType: String?
    ) -> Bool {
        guard templateFormat != stamped else { return false }
        let family = capabilities.family?.lowercased() ?? ""
        let type = modelType?.lowercased() ?? ""
        let stamp = capabilities.toolParser?.lowercased() ?? ""
        return stamped == .glm4
            && templateFormat == .json
            && stamp == "deepseek"
            && (family.contains("bailing") || family.contains("ling")
                || type.contains("bailing") || type.contains("ling"))
    }

    /// Resolve a ``DraftStrategy`` from JANG capability stamp.
    ///
    /// Maps `capabilities.draft_strategy` + `capabilities.drafter_path`
    /// + `capabilities.branching_budget` + `capabilities.block_size` into
    /// a concrete `DraftStrategy` enum. The drafter path is resolved
    /// relative to `modelDirectory` (the snapshot root containing
    /// `jang_config.json`) — JANG bundles ship drafters co-located.
    ///
    /// Returns `nil` when:
    /// - `capabilities` is nil.
    /// - `draftStrategy` is nil, `"none"`, or unrecognised.
    /// - `drafterPath` is nil (strategy requires one but bundle
    ///   doesn't ship it).
    /// - `blockSize` is nil (required for both `.dflash` + `.ddtree`).
    ///
    /// - Parameters:
    ///   - capabilities: the `JangCapabilities` block from
    ///     `jang_config.json`.
    ///   - modelDirectory: the snapshot root. `capabilities.drafter_path`
    ///     is appended to this.
    public static func draftStrategy(
        capabilities: JangCapabilities?,
        modelDirectory: URL
    ) -> (strategy: DraftStrategy?, source: JangCapabilities.ResolutionSource) {
        guard let cap = capabilities,
            let name = cap.draftStrategy?.lowercased(),
            name != "none",
            let relativePath = cap.drafterPath,
            let blockSize = cap.blockSize
        else {
            return (nil, .none)
        }
        let drafterURL = modelDirectory
            .appendingPathComponent(relativePath, isDirectory: true)
            .resolvingSymlinksInPath()
        switch name {
        case "dflash":
            return (
                .dflash(drafterPath: drafterURL, blockSize: blockSize),
                .jangStamped
            )
        case "ddtree":
            let budget = cap.branchingBudget ?? 32
            return (
                .ddtree(
                    drafterPath: drafterURL,
                    branchingBudget: budget,
                    blockSize: blockSize),
                .jangStamped
            )
        default:
            return (nil, .none)
        }
    }
}

/// Parsed JANG model configuration from jang_config.json.
/// Reasoning-mode hint block from `jang_config.json -> chat.reasoning`.
///
/// Per `jang/research/DSV-FAMILY-RUNTIME-GUIDE.md` §23 + §25, DSV4
/// bundles ship explicit reasoning-mode metadata:
///
///   - `modes`: which modes the model supports (e.g. `["chat", "thinking"]`)
///   - `default_mode`: which mode to use if the caller doesn't pick one
///   - `thinking_start` / `thinking_end`: the envelope tags the
///     runtime should watch for (e.g. `<think>` / `</think>`)
///   - `reasoning_effort_levels`: allowed `reasoning_effort` knob
///     values (e.g. `["max", "high", nil]`)
///   - `drop_earlier_reasoning`: whether multi-turn chat should
///     strip earlier assistant reasoning before re-encoding
///
/// DSV4 is the first family that splits reasoning into a `"chat"`
/// mode (prompt ends with a CLOSED `</think>` empty block — parser
/// must start with `startInReasoning: false`) and a `"thinking"`
/// mode (prompt ends with an OPEN `<think>` — parser starts inside
/// reasoning). `ReasoningParser.forPrompt(stampName:promptTail:)`
/// already handles tail detection, but consumers need this struct
/// to know the default mode + allowed options.
public struct JangChatReasoning: Sendable, Equatable {
    public let supported: Bool?
    public let modes: [String]?
    public let defaultMode: String?
    public let thinkingStart: String?
    public let thinkingEnd: String?
    public let reasoningEffortLevels: [String?]?
    public let dropEarlierReasoning: Bool?

    public init(
        supported: Bool? = nil,
        modes: [String]? = nil,
        defaultMode: String? = nil,
        thinkingStart: String? = nil,
        thinkingEnd: String? = nil,
        reasoningEffortLevels: [String?]? = nil,
        dropEarlierReasoning: Bool? = nil
    ) {
        self.supported = supported
        self.modes = modes
        self.defaultMode = defaultMode
        self.thinkingStart = thinkingStart
        self.thinkingEnd = thinkingEnd
        self.reasoningEffortLevels = reasoningEffortLevels
        self.dropEarlierReasoning = dropEarlierReasoning
    }
}

/// Tool-calling hint block from `jang_config.json -> chat.tool_calling`.
/// DSV4 stamps `parser = "dsml"` + the DSML markup token; other
/// families may stamp parser names like `"xml_function"` or
/// `"kimi_k2"` that round-trip through
/// `ToolCallFormat.fromCapabilityName`.
public struct JangChatToolCalling: Sendable, Equatable {
    public let supported: Bool?
    public let parser: String?
    public let dsmlToken: String?
    public let toolCallsBlock: String?
    public let invokeBlock: String?
    public let parameterBlock: String?
    public let toolOutputTag: String?

    public init(
        supported: Bool? = nil,
        parser: String? = nil,
        dsmlToken: String? = nil,
        toolCallsBlock: String? = nil,
        invokeBlock: String? = nil,
        parameterBlock: String? = nil,
        toolOutputTag: String? = nil
    ) {
        self.supported = supported
        self.parser = parser
        self.dsmlToken = dsmlToken
        self.toolCallsBlock = toolCallsBlock
        self.invokeBlock = invokeBlock
        self.parameterBlock = parameterBlock
        self.toolOutputTag = toolOutputTag
    }
}

/// Sampling defaults from `jang_config.json -> chat.sampling_defaults`.
/// Consumers (BatchEngine / Evaluate) may apply these when the
/// caller doesn't pass explicit sampler params. DSV4-Flash recommends
/// `temperature=0.6, top_p=0.95, max_new_tokens=300`.
public struct JangChatSamplingDefaults: Sendable, Equatable {
    public let temperature: Float?
    public let topP: Float?
    public let maxNewTokens: Int?

    public init(
        temperature: Float? = nil, topP: Float? = nil, maxNewTokens: Int? = nil
    ) {
        self.temperature = temperature
        self.topP = topP
        self.maxNewTokens = maxNewTokens
    }
}

/// Top-level `jang_config.json -> chat` block. Aggregates reasoning
/// + tool-calling + sampling hints the runtime applies when
/// building prompts and configuring generation. Populated only
/// when the bundle carries the new DSV4-era schema; older bundles
/// fall back to `capabilities` + model_type heuristics.
public struct JangChatConfig: Sendable, Equatable {
    public let encoder: String?
    public let hasTokenizerChatTemplate: Bool?
    public let bosToken: String?
    public let bosTokenId: Int?
    public let eosToken: String?
    public let eosTokenId: Int?
    public let roleTokens: [String: String]?
    public let reasoning: JangChatReasoning?
    public let toolCalling: JangChatToolCalling?
    public let samplingDefaults: JangChatSamplingDefaults?

    public init(
        encoder: String? = nil,
        hasTokenizerChatTemplate: Bool? = nil,
        bosToken: String? = nil,
        bosTokenId: Int? = nil,
        eosToken: String? = nil,
        eosTokenId: Int? = nil,
        roleTokens: [String: String]? = nil,
        reasoning: JangChatReasoning? = nil,
        toolCalling: JangChatToolCalling? = nil,
        samplingDefaults: JangChatSamplingDefaults? = nil
    ) {
        self.encoder = encoder
        self.hasTokenizerChatTemplate = hasTokenizerChatTemplate
        self.bosToken = bosToken
        self.bosTokenId = bosTokenId
        self.eosToken = eosToken
        self.eosTokenId = eosTokenId
        self.roleTokens = roleTokens
        self.reasoning = reasoning
        self.toolCalling = toolCalling
        self.samplingDefaults = samplingDefaults
    }
}

public struct JangRoutedExpertBitPlan: Sendable, Equatable {
    public let defaultBits: [String: Int]
    public let layerOverrides: [Int: [String: Int]]

    public init(
        defaultBits: [String: Int] = [:],
        layerOverrides: [Int: [String: Int]] = [:]
    ) {
        self.defaultBits = defaultBits
        self.layerOverrides = layerOverrides
    }

    public func bits(layerIndex: Int, projection: String) -> Int? {
        layerOverrides[layerIndex]?[projection]
            ?? defaultBits[projection]
            ?? defaultBits[projection.replacingOccurrences(of: "_proj", with: "")]
    }

    public var allBitWidths: [Int] {
        Array(Set(defaultBits.values + layerOverrides.values.flatMap(\.values))).sorted()
    }
}

public struct JangConfig: Sendable {
    public let format: String
    public let formatVersion: String
    public var isV2: Bool { formatVersion.hasPrefix("2") }
    public let quantization: JangQuantization
    public let mxtqBits: [String: Int]
    public let routedExpertBitPlan: JangRoutedExpertBitPlan?
    public let sourceModel: JangSourceModel
    public let architecture: JangArchitecture
    public let runtime: JangRuntime

    /// Optional capability stamp from the JANG converter. `nil` for
    /// pre-stamp models — consumers should fall back to model-type
    /// heuristics.
    public let capabilities: JangCapabilities?

    /// Top-level `model_family` hint (new in DSV4-era jang_config —
    /// e.g. `"deepseek_v4"`, `"kimi_k26"`). Complements
    /// `capabilities.family` which is a UI / registry grouping;
    /// `modelFamily` is used by runtime chat-encoder dispatch.
    public let modelFamily: String?

    /// Optional `chat.*` block — present on DSV4-era bundles with
    /// explicit reasoning modes + tool-parser stamps + sampling
    /// defaults. `nil` on older bundles; consumers fall back to
    /// `capabilities` + model_type heuristics.
    public let chat: JangChatConfig?

    public init(
        format: String = "jang",
        formatVersion: String = "2.0",
        quantization: JangQuantization = JangQuantization(),
        mxtqBits: [String: Int] = [:],
        routedExpertBitPlan: JangRoutedExpertBitPlan? = nil,
        sourceModel: JangSourceModel = JangSourceModel(),
        architecture: JangArchitecture = JangArchitecture(),
        runtime: JangRuntime = JangRuntime(),
        capabilities: JangCapabilities? = nil,
        modelFamily: String? = nil,
        chat: JangChatConfig? = nil
    ) {
        self.format = format
        self.formatVersion = formatVersion
        self.quantization = quantization
        self.mxtqBits = mxtqBits
        self.routedExpertBitPlan = routedExpertBitPlan
        self.sourceModel = sourceModel
        self.architecture = architecture
        self.runtime = runtime
        self.capabilities = capabilities
        self.modelFamily = modelFamily
        self.chat = chat
    }
}

// MARK: - JANG Loader

/// JANG model loader — detects, parses config, and infers per-layer quantization.
public struct JangLoader: Sendable {

    /// Check if a model directory contains a JANG model.
    public static func isJangModel(at path: URL) -> Bool {
        findConfigPath(at: path) != nil
    }

    /// Find the JANG config file in a model directory.
    public static func findConfigPath(at modelPath: URL) -> URL? {
        for name in jangConfigFileNames {
            let configURL = modelPath.appendingPathComponent(name)
            if FileManager.default.fileExists(atPath: configURL.path) {
                return configURL
            }
        }
        // .jangspec bundles built before the Plan 6 builder update only place
        // jang_config.json under target/. Fall back to the bundle layout so
        // those still load without rebuilding the bundle.
        for name in jangConfigFileNames {
            let configURL = modelPath.appendingPathComponent("target")
                .appendingPathComponent(name)
            if FileManager.default.fileExists(atPath: configURL.path) {
                return configURL
            }
        }
        return nil
    }

    /// Resolve the directory that holds tokenizer files for a given model.
    ///
    /// The HuggingFace tokenizer loader (`AutoTokenizer.from(modelFolder:)`)
    /// expects `tokenizer.json` and/or `tokenizer_config.json` (plus optionally
    /// `chat_template.jinja`) in the directory it is pointed at. Most JANG /
    /// JANGTQ bundles ship **weights-only** — the snapshot directory contains
    /// `model.safetensors`, `config.json`, `jang_config.json` (and sometimes
    /// `jangtq_runtime.safetensors`) but no tokenizer files. Users are
    /// expected to re-use the tokenizer from the source model declared in
    /// `jang_config.json["source_model"]`.
    ///
    /// This helper implements that fallback for local-directory loads:
    ///
    /// 1. If `modelDirectory` itself has `tokenizer_config.json` or
    ///    `tokenizer.json` → return it unchanged (standard path).
    /// 2. Else if `modelDirectory` has `jang_config.json` with a populated
    ///    `source_model.org` + `source_model.name` → look up the HuggingFace
    ///    cache directory for that repo (`~/.cache/huggingface/hub/models--<org>--<name>`)
    ///    and return the first snapshot that has tokenizer files.
    /// 3. Else → return `modelDirectory` unchanged. The tokenizer loader will
    ///    surface its own error, which is the same behaviour as before this
    ///    helper existed.
    ///
    /// The fallback path **does not** perform network downloads. It only
    /// finds a tokenizer that has already been cached by `Downloader`. If the
    /// source model isn't cached, the returned URL still won't have
    /// tokenizer files and the loader will fail with a clear "no tokenizer"
    /// error — which is the signal for callers to `.download(id:)` the source
    /// repo first.
    ///
    /// - Parameters:
    ///   - modelDirectory: Directory of the model being loaded.
    ///   - huggingFaceCacheRoot: Override for the HF cache root. Defaults to
    ///     `~/.cache/huggingface/hub`. Exposed for unit tests.
    ///   - fileManager: File-manager used for probe. Exposed for unit tests.
    /// - Returns: A directory that should be passed to the tokenizer loader.
    public static func resolveTokenizerDirectory(
        for modelDirectory: URL,
        huggingFaceCacheRoot: URL? = nil,
        fileManager: FileManager = .default
    ) -> URL {
        if hasTokenizerFiles(at: modelDirectory, fileManager: fileManager),
           !shouldPreferSourceTokenizer(
                for: modelDirectory, fileManager: fileManager)
        {
            return modelDirectory
        }
        guard isJangModel(at: modelDirectory) else { return modelDirectory }

        // Read source_model from jang_config.json. Any parse failure or
        // missing org/name → caller gets the default (unchanged) path.
        let config: JangConfig
        do {
            config = try loadConfig(at: modelDirectory)
        } catch {
            return modelDirectory
        }
        let repo = config.sourceModel.huggingFaceRepoID
        guard !repo.isEmpty else { return modelDirectory }

        let cacheRoot = huggingFaceCacheRoot ?? defaultHuggingFaceCacheRoot()
        let cacheDirName = "models--\(config.sourceModel.org)--\(config.sourceModel.name)"
        let snapshotsRoot = cacheRoot
            .appendingPathComponent(cacheDirName)
            .appendingPathComponent("snapshots")

        guard let entries = try? fileManager.contentsOfDirectory(
            at: snapshotsRoot,
            includingPropertiesForKeys: nil
        ) else {
            return modelDirectory
        }

        // First snapshot directory that actually has tokenizer files wins.
        // HuggingFace snapshots are immutable per revision, so any of them
        // with the files is equally good; the presence check is what matters.
        for snapshot in entries where hasTokenizerFiles(at: snapshot, fileManager: fileManager) {
            return snapshot
        }
        return modelDirectory
    }

    /// Some VLM bundles carry the production multimodal chat template in a
    /// sibling `chat_template.json` file while `tokenizer_config.json` contains
    /// a text-only fallback template. The HuggingFace tokenizer loader only
    /// reads `tokenizer_config.json`, so those bundles silently lose image
    /// placeholders unless we materialize a tokenizer shim whose
    /// `chat_template` field points at the sidecar template.
    ///
    /// ZAYA1-VL JANG bundles have an additional real metadata contract:
    /// `jang_config.json` stamps `family = zaya1_vl`,
    /// `tool_parser = zaya_xml`, `think_in_template = false`, and
    /// `supports_tools = true`, while older
    /// tokenizer configs may still carry a plain `user:` / `assistant:`
    /// template that ignores image placeholders and tools. For those bundles,
    /// materialize the native ZAYA1-VL vision/tool template even if no sidecar
    /// file exists, while preserving `think_in_template=false`.
    ///
    /// This is intentionally data-driven, not family-name driven:
    ///
    /// - If `chat_template.json` exists, it must contain a string
    ///   `chat_template` with a vision placeholder marker.
    /// - Or `jang_config.json` must prove the ZAYA1-VL tool-aware contract.
    /// - The current tokenizer config must not already contain the same
    ///   production markers.
    ///
    /// If any condition is not met, returns `directory` unchanged.
    public static func resolveChatTemplateSidecarSubstitution(
        for directory: URL,
        fileManager: FileManager = .default
    ) -> URL {
        let configURL = directory.appendingPathComponent("tokenizer_config.json")
        guard fileManager.fileExists(atPath: configURL.path),
              let configData = try? Data(contentsOf: configURL),
              var configJSON = try? JSONSerialization.jsonObject(with: configData)
                as? [String: Any]
        else {
            return directory
        }

        let currentTemplate = configJSON["chat_template"] as? String
        let zayaToolAware = shouldUseZayaToolAwareTemplate(for: directory)
        let zayaVLToolAware = shouldUseZayaVLToolAwareTemplate(for: directory)
        let lfm2ToolAware = shouldUseLFM2ToolAwareTemplate(for: directory)
        if let currentTemplate,
           zayaToolAware,
           templateAlreadyMatchesZayaToolAware(currentTemplate),
           (!zayaVLToolAware || isVisionChatTemplate(currentTemplate))
        {
            return directory
        }
        if let currentTemplate,
           lfm2ToolAware,
           templateAlreadyMatchesLFM2ToolAware(currentTemplate)
        {
            return directory
        }

        let sidecarURL = directory.appendingPathComponent("chat_template.json")
        let sidecarTemplate: String? = {
            guard fileManager.fileExists(atPath: sidecarURL.path),
                  let sidecarData = try? Data(contentsOf: sidecarURL),
                  let sidecarJSON = try? JSONSerialization.jsonObject(with: sidecarData)
                    as? [String: Any],
                  let template = sidecarJSON["chat_template"] as? String,
                  isVisionChatTemplate(template)
            else {
                return nil
            }
            return template
        }()

        // Modern-HuggingFace fallback: current `transformers` ships the chat
        // template as a standalone `chat_template.jinja` text file and leaves
        // `tokenizer_config.json` without an inline `chat_template`. swift-
        // transformers only reads the inline field, so such a model is prompted
        // with no turn structure — an instruct model then emits EOS/pad
        // immediately and the response detokenizes to empty (observed on
        // LFM2.5 mxfp4/mxfp8 and VibeThinker-3B mxfp4/mxfp8/jang). When there is
        // no usable inline template and no more-specific substitution above
        // applies, inject the model's own `.jinja` template verbatim.
        let genericJinjaTemplate: String? = {
            if let currentTemplate, !currentTemplate.isEmpty { return nil }
            let jinjaURL = directory.appendingPathComponent("chat_template.jinja")
            guard fileManager.fileExists(atPath: jinjaURL.path),
                  let text = try? String(contentsOf: jinjaURL, encoding: .utf8),
                  !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            else {
                return nil
            }
            return text
        }()

        // Modern Mistral / Pixtral (and other non-Qwen/Gemma) VLM bundles ship
        // their real chat template ONLY in a sibling `chat_template.json` — the
        // Mistral-family `[SYSTEM_PROMPT]` / `[INST]` / `[IMG]` format — with no
        // inline `tokenizer_config.json` template and no `chat_template.jinja`.
        // The vision-sidecar path above only fires for templates
        // `isVisionChatTemplate` recognizes (Qwen/Gemma `<|vision_start|>` etc.,
        // NOT Mistral's `[IMG]`), so that template is skipped, swift-transformers
        // sees no template, and the model is prompted with a ChatML default that
        // leaks `<|im_start|>` markers (observed on mlx-community Mistral-Small
        // 3.1/3.2). When there is no usable inline template, no recognized-vision
        // sidecar substitution, and no `.jinja`, inject the model's own
        // `chat_template.json` template verbatim. Gated on an empty inline
        // template and a nil vision-sidecar, so bundles that already resolve a
        // template (inline, `.jinja`, or recognized-vision sidecar) are untouched.
        let genericJsonTemplate: String? = {
            if let currentTemplate, !currentTemplate.isEmpty { return nil }
            if sidecarTemplate != nil { return nil }
            guard fileManager.fileExists(atPath: sidecarURL.path),
                  let data = try? Data(contentsOf: sidecarURL),
                  let json = try? JSONSerialization.jsonObject(with: data)
                    as? [String: Any],
                  let template = json["chat_template"] as? String,
                  !template.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            else {
                return nil
            }
            return template
        }()

        // Mistral 3.x packs whose template (inline OR chat_template.json) is the
        // vision/text `[INST]` format with NO tool support (e.g. mlx-community
        // Mistral-Small-3.1/3.2 ship only the HF vision template). Mistral does
        // tools natively via the tekken `[TOOL_CALLS]name[ARGS]{}` format, but
        // the shipped template never renders `[AVAILABLE_TOOLS]`, so tools go
        // ungrounded. Swap in the complete Mistral template, which preserves the
        // native `[INST]`/`[IMG]` surface and adds tools + conditional reasoning.
        // Data-driven (`[INST]` present, `[AVAILABLE_TOOLS]` absent), not by name.
        let mistralCompleteTemplate: String? = {
            let existing: String? = {
                if let currentTemplate, !currentTemplate.isEmpty { return currentTemplate }
                return sidecarTemplate ?? genericJsonTemplate
            }()
            // Require BOTH Mistral-family markers: `[INST]` alone is also used
            // by Llama-2-family templates (which use `<<SYS>>`, not
            // `[SYSTEM_PROMPT]`), so gate on `[SYSTEM_PROMPT]` too to avoid
            // mis-applying the Mistral template to a non-Mistral `[INST]` model.
            guard let existing,
                  existing.contains("[INST]"),
                  existing.contains("[SYSTEM_PROMPT]"),
                  !existing.contains("[AVAILABLE_TOOLS]")
            else {
                return nil
            }
            return ChatTemplateFallbacks.mistral3CompleteMinimal
        }()

        guard zayaToolAware || lfm2ToolAware || sidecarTemplate != nil
            || genericJinjaTemplate != nil || genericJsonTemplate != nil
            || mistralCompleteTemplate != nil
        else {
            return directory
        }
        if !zayaToolAware,
           !lfm2ToolAware,
           mistralCompleteTemplate == nil,
           let currentTemplate,
           isVisionChatTemplate(currentTemplate)
        {
            return directory
        }

        let effectiveTemplate: String
        if zayaToolAware {
            effectiveTemplate = ChatTemplateFallbacks.zayaVLVisionToolMinimal
        } else if lfm2ToolAware {
            effectiveTemplate = ChatTemplateFallbacks.lfm2ToolMinimal
        } else if let mistralCompleteTemplate {
            effectiveTemplate = mistralCompleteTemplate
        } else if let sidecarTemplate {
            effectiveTemplate = sidecarTemplate
        } else if let genericJinjaTemplate {
            effectiveTemplate = genericJinjaTemplate
        } else {
            effectiveTemplate = genericJsonTemplate!
        }
        configJSON["chat_template"] = effectiveTemplate

        let shimDir = fileManager.temporaryDirectory.appendingPathComponent(
            "vmlx-chat-template-shim-\(UUID().uuidString)")
        do {
            try fileManager.createDirectory(
                at: shimDir, withIntermediateDirectories: true)
            let rewritten = try JSONSerialization.data(
                withJSONObject: configJSON, options: [.prettyPrinted, .sortedKeys])
            try rewritten.write(to: shimDir.appendingPathComponent("tokenizer_config.json"))
            try effectiveTemplate.write(
                to: shimDir.appendingPathComponent("chat_template.jinja"),
                atomically: true,
                encoding: .utf8)
            let rewrittenSidecar = try JSONSerialization.data(
                withJSONObject: ["chat_template": effectiveTemplate],
                options: [.prettyPrinted, .sortedKeys])
            try rewrittenSidecar.write(to: shimDir.appendingPathComponent("chat_template.json"))

            let entries = (try? fileManager.contentsOfDirectory(
                at: directory, includingPropertiesForKeys: nil)) ?? []
            let rewrittenTemplateFiles: Set<String> = [
                "tokenizer_config.json",
                "chat_template.jinja",
                "chat_template.json",
            ]
            for entry in entries where !rewrittenTemplateFiles.contains(entry.lastPathComponent) {
                let dest = shimDir.appendingPathComponent(entry.lastPathComponent)
                let real = (try? fileManager.destinationOfSymbolicLink(atPath: entry.path))
                    .flatMap { relative in
                        URL(fileURLWithPath: relative, relativeTo: entry.deletingLastPathComponent())
                            .standardizedFileURL
                    } ?? entry
                try? fileManager.createSymbolicLink(at: dest, withDestinationURL: real)
            }
            return shimDir
        } catch {
            return directory
        }
    }

    private static func isVisionChatTemplate(_ template: String) -> Bool {
        template.contains("<|vision_start|>")
            || template.contains("<|image_pad|>")
            || template.contains("<|video_pad|>")
            || template.contains("<|image|>")
            || template.contains("<image>")
    }

    private static func templateAlreadyMatchesZayaVLToolAware(_ template: String) -> Bool {
        isVisionChatTemplate(template)
            && template.contains("zyphra_tool_call")
            && template.contains("required_tool_choice")
            && template.contains("tool_choice")
            && template.contains("The current assistant response MUST be a tool call.")
    }

    private static func templateAlreadyMatchesZayaToolAware(_ template: String) -> Bool {
        template.contains("zyphra_tool_call")
            && template.contains("required_tool_choice")
            && template.contains("tool_choice")
            && template.contains("The current assistant response MUST be a tool call.")
    }

    private static func templateAlreadyMatchesLFM2ToolAware(_ template: String) -> Bool {
        template.contains("<|tool_call_start|>")
            && template.contains("<|tool_call_end|>")
            && template.contains("tool_choice")
    }

    private static func shouldUseZayaToolAwareTemplate(for directory: URL) -> Bool {
        guard let config = try? loadConfig(at: directory) else {
            return false
        }

        let family = config.capabilities?.family?.lowercased() ?? ""
        let parser = config.capabilities?.toolParser?.lowercased() ?? ""
        let supportsTools = config.capabilities?.supportsTools
        let isZayaText = family == "zaya"
            || family == "zaya1"
            || family.hasPrefix("zaya1_")
            || family.hasPrefix("zaya1-")
        return isZayaText
            && ["zaya", "zaya_xml", "zyphra", "zyphra_xml"].contains(parser)
            && config.capabilities?.thinkInTemplate == false
            && supportsTools != false
    }

    private static func shouldUseZayaVLToolAwareTemplate(for directory: URL) -> Bool {
        guard let config = try? loadConfig(at: directory) else {
            return false
        }

        let family = config.capabilities?.family?.lowercased() ?? ""
        let parser = config.capabilities?.toolParser?.lowercased() ?? ""
        let supportsTools = config.capabilities?.supportsTools
        return family.contains("zaya1_vl")
            && ["zaya", "zaya_xml", "zyphra", "zyphra_xml"].contains(parser)
            && config.capabilities?.thinkInTemplate == false
            && supportsTools != false
    }

    private static func shouldUseLFM2ToolAwareTemplate(for directory: URL) -> Bool {
        guard let config = try? loadConfig(at: directory) else {
            return false
        }

        let family = config.capabilities?.family?.lowercased() ?? ""
        let parser = config.capabilities?.toolParser?.lowercased() ?? ""
        let supportsTools = config.capabilities?.supportsTools
        return supportsTools != false
            && ["lfm2", "lfm2_moe", "lfm2.5", "lfm2_5", "lfm25"].contains(family)
            && ["lfm2", "lfm2_moe", "lfm2_5", "lfm25"].contains(parser)
            && config.capabilities?.thinkInTemplate == false
    }

    /// Check whether a directory already has the files that the HuggingFace
    /// tokenizer loader needs. Used by `resolveTokenizerDirectory(for:)`.
    public static func hasTokenizerFiles(
        at directory: URL,
        fileManager: FileManager = .default
    ) -> Bool {
        for name in ["tokenizer.json", "tokenizer_config.json"] {
            let url = directory.appendingPathComponent(name)
            if fileManager.fileExists(atPath: url.path) { return true }
        }
        return false
    }

    private static func hasTokenizerJson(
        at directory: URL,
        fileManager: FileManager = .default
    ) -> Bool {
        fileManager.fileExists(
            atPath: directory.appendingPathComponent("tokenizer.json").path)
    }

    private static func shouldPreferSourceTokenizer(
        for directory: URL,
        fileManager: FileManager = .default
    ) -> Bool {
        guard !hasTokenizerJson(at: directory, fileManager: fileManager) else {
            return false
        }
        let configURL = directory.appendingPathComponent("tokenizer_config.json")
        guard fileManager.fileExists(atPath: configURL.path),
              let data = try? Data(contentsOf: configURL),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let tokenizerClass = json["tokenizer_class"] as? String
        else {
            return false
        }
        let trimmed = tokenizerClass.replacingOccurrences(of: "Fast", with: "")
        return tokenizerClass == "TikTokenTokenizer"
            || trimmed == "TikTokenTokenizer"
    }

    // MARK: - tokenizer_class substitution

    /// swift-transformers 0.1.21's `knownTokenizers` doesn't include
    /// `TokenizersBackend` (used by some mlx-community snapshots like
    /// `mlx-community/Qwen3.5-VL-9B-8bit`) — loads throw
    /// `TokenizerError.unsupportedTokenizer("TokenizersBackend")`. This
    /// set lists all classes we know swift-transformers accepts. Callers
    /// that need different substitutions can override via env var
    /// `VMLX_TOKENIZER_CLASS_OVERRIDE=<target>`.
    public static let knownSupportedTokenizerClasses: Set<String> = [
        "CodeGenTokenizer", "CodeLlamaTokenizer", "FalconTokenizer",
        "GemmaTokenizer", "GPT2Tokenizer", "LlamaTokenizer", "T5Tokenizer",
        "WhisperTokenizer", "CohereTokenizer", "Qwen2Tokenizer",
        "PreTrainedTokenizer",
    ]

    /// Substitution map: when `tokenizer_class` is a key in this map
    /// and no env override is set, rewrite to the value. Tuned from
    /// real-world snapshots: `TokenizersBackend` on Qwen-family VL
    /// models is functionally `Qwen2Tokenizer`.
    public static let defaultTokenizerClassSubstitutions: [String: String] = [
        "TokenizersBackend": "Qwen2Tokenizer",
        // Kimi K2.x/K2.5/K2.6 bundles may ship a tiktoken.model plus a
        // generated tokenizer.json. swift-transformers does not register
        // TikTokenTokenizer as a class name, but the generated tokenizer
        // is a standard byte-level BPE tokenizer.
        "TikTokenTokenizer": "Qwen2Tokenizer",
    ]

    /// Like `resolveTokenizerDirectory(for:)` but also fixes
    /// `tokenizer_class` in `tokenizer_config.json` to an entry that
    /// swift-transformers 0.1.21 knows. If the class is already known,
    /// returns the input directory unchanged. If unknown and no
    /// substitute is available, returns unchanged (let the loader
    /// surface the clear error).
    ///
    /// When a substitution is required, writes a shim directory into
    /// `<tmp>/vmlx-tokenizer-shim-<uuid>/` containing the rewritten
    /// `tokenizer_config.json` plus symlinks to every other tokenizer
    /// file (tokenizer.json, chat_template.jinja, etc.). The caller
    /// should clean up the shim dir when done, but since they live in
    /// the OS temp dir the OS sweeps them eventually.
    ///
    /// Order of operations for a full load:
    ///
    /// 1. Caller has a model directory (maybe JANG, maybe not).
    /// 2. `resolveTokenizerDirectory(for:)` redirects weights-only JANG
    ///    bundles to their source-model snapshot.
    /// 3. `resolveTokenizerClassSubstitution(for:)` (this function)
    ///    rewrites `tokenizer_class` if it's unsupported.
    /// 4. The returned URL is passed to
    ///    `AutoTokenizer.from(modelFolder:)`.
    public static func resolveTokenizerClassSubstitution(
        for directory: URL,
        overrideClass: String? = nil,
        fileManager: FileManager = .default
    ) -> URL {
        let configURL = directory.appendingPathComponent("tokenizer_config.json")
        guard fileManager.fileExists(atPath: configURL.path) else {
            return directory  // nothing to rewrite; downstream loader errors
        }
        guard let data = try? Data(contentsOf: configURL),
              var json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            return directory
        }

        let currentClass = json["tokenizer_class"] as? String ?? ""
        let trimmedCurrent = currentClass.replacingOccurrences(of: "Fast", with: "")

        // Decide the target class.
        let target: String
        let envOverride = overrideClass
            ?? ProcessInfo.processInfo.environment["VMLX_TOKENIZER_CLASS_OVERRIDE"]
        if let envOverride, !envOverride.isEmpty {
            target = envOverride
        } else if knownSupportedTokenizerClasses.contains(trimmedCurrent) {
            return directory  // already supported
        } else if let mapped = defaultTokenizerClassSubstitutions[currentClass]
                            ?? defaultTokenizerClassSubstitutions[trimmedCurrent] {
            target = mapped
        } else {
            return directory  // unknown class, no known substitute
        }

        // If nothing to change, skip.
        if target == currentClass { return directory }

        json["tokenizer_class"] = target

        // Write to a shim dir next to the original.
        let shimDir = fileManager.temporaryDirectory.appendingPathComponent(
            "vmlx-tokenizer-shim-\(UUID().uuidString)")
        do {
            try fileManager.createDirectory(
                at: shimDir, withIntermediateDirectories: true)
            let rewritten = try JSONSerialization.data(
                withJSONObject: json, options: [.prettyPrinted, .sortedKeys])
            try rewritten.write(
                to: shimDir.appendingPathComponent("tokenizer_config.json"))
            // Symlink all OTHER files — tokenizer.json especially is often
            // large and we don't want to duplicate it.
            let entries = (try? fileManager.contentsOfDirectory(
                at: directory, includingPropertiesForKeys: nil)) ?? []
            for entry in entries where entry.lastPathComponent != "tokenizer_config.json" {
                let dest = shimDir.appendingPathComponent(entry.lastPathComponent)
                // Some tokenizer caches already contain symlinks — follow them
                // so our shim links to the actual file, not another link.
                let real = (try? fileManager.destinationOfSymbolicLink(atPath: entry.path))
                    .flatMap { relative in
                        URL(fileURLWithPath: relative, relativeTo: entry.deletingLastPathComponent())
                            .standardizedFileURL
                    } ?? entry
                try? fileManager.createSymbolicLink(at: dest, withDestinationURL: real)
            }
            return shimDir
        } catch {
            return directory
        }
    }

    /// Default HuggingFace hub cache root. Honours `HF_HOME` and `HF_HUB_CACHE`
    /// environment variables, otherwise falls back to `~/.cache/huggingface/hub`
    /// — matching the Python `huggingface_hub` resolution order.
    public static func defaultHuggingFaceCacheRoot() -> URL {
        let env = ProcessInfo.processInfo.environment
        if let hubCache = env["HF_HUB_CACHE"], !hubCache.isEmpty {
            return URL(fileURLWithPath: hubCache)
        }
        if let hfHome = env["HF_HOME"], !hfHome.isEmpty {
            return URL(fileURLWithPath: hfHome).appendingPathComponent("hub")
        }
        return FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/huggingface/hub")
    }

    /// Load and parse the JANG config from a model directory.
    public static func loadConfig(at modelPath: URL) throws -> JangConfig {
        guard let configURL = findConfigPath(at: modelPath) else {
            throw JangLoaderError.configNotFound(modelPath.path)
        }

        let data = try Data(contentsOf: configURL)
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw JangLoaderError.invalidConfig("Failed to parse JSON")
        }

        return try parseConfig(from: json)
    }

    /// Parse a JangConfig from a raw JSON dictionary.
    public static func parseConfig(from json: [String: Any]) throws -> JangConfig {
        let format = json["format"] as? String ?? "jang"
        let formatVersion = json["format_version"] as? String ?? "2.0"

        let mxtqBits = parseMXTQBits(json["mxtq_bits"])
        let routedExpertBits = parseMXTQBits(json["routed_expert_bits"])
        let routedExpertBitPlan = parseRoutedExpertBitPlan(
            json["routed_expert_bit_plan"],
            fallbackDefault: routedExpertBits.isEmpty ? mxtqBits : routedExpertBits)
        let topLevelRoutedGroupSize = json["routed_expert_group_size"] as? Int
        let topLevelBitWidths = Array(Set(
            Array(mxtqBits.values)
                + Array(routedExpertBits.values)
                + (routedExpertBitPlan?.allBitWidths ?? []))).sorted()

        let quantization: JangQuantization
        if let qDict = json["quantization"] as? [String: Any] {
            quantization = JangQuantization(
                method: qDict["method"] as? String ?? "jang-importance",
                profile: qDict["profile"] as? String ?? "JANG_2S",
                targetBits: floatValue(qDict["target_bits"]) ?? 2.5,
                actualBits: floatValue(qDict["actual_bits"]) ?? 2.5,
                blockSize: (qDict["block_size"] as? Int) ?? (qDict["group_size"] as? Int)
                    ?? topLevelRoutedGroupSize ?? 64,
                bitWidthsUsed: Array(Set(
                    (qDict["bit_widths_used"] as? [Int] ?? []) + topLevelBitWidths
                )).sorted(),
                quantizationScheme: qDict["quantization_scheme"] as? String ?? "asymmetric",
                quantizationBackend: qDict["quantization_backend"] as? String ?? "mx.quantize"
            )
        } else {
            quantization = JangQuantization(
                blockSize: topLevelRoutedGroupSize ?? 64,
                bitWidthsUsed: topLevelBitWidths)
        }

        let sourceModel = parseSourceModel(json["source_model"])

        let architecture: JangArchitecture
        if let aDict = json["architecture"] as? [String: Any] {
            architecture = JangArchitecture(
                type: aDict["type"] as? String ?? "transformer",
                attention: aDict["attention"] as? String ?? "gqa",
                hasVision: aDict["has_vision"] as? Bool ?? false,
                hasSSM: aDict["has_ssm"] as? Bool ?? false,
                hasMoE: aDict["has_moe"] as? Bool ?? false
            )
        } else {
            architecture = JangArchitecture()
        }

        let runtime: JangRuntime
        if let rDict = json["runtime"] as? [String: Any] {
            runtime = JangRuntime(
                totalWeightBytes: rDict["total_weight_bytes"] as? Int ?? 0,
                totalWeightGB: floatValue(rDict["total_weight_gb"]) ?? 0,
                bundleHasMTP: rDict["bundle_has_mtp"] as? Bool ?? false,
                mtpLayers: rDict["mtp_layers"] as? Int ?? 0,
                mtpMode: MTPRuntimeMode(rawMode: rDict["mtp_mode"] as? String)
            )
        } else {
            runtime = JangRuntime()
        }

        let capabilities: JangCapabilities?
        if let cDict = json["capabilities"] as? [String: Any] {
            capabilities = JangCapabilities(
                reasoningParser: cDict["reasoning_parser"] as? String,
                toolParser: cDict["tool_parser"] as? String,
                thinkInTemplate: cDict["think_in_template"] as? Bool,
                supportsTools: cDict["supports_tools"] as? Bool,
                supportsThinking: cDict["supports_thinking"] as? Bool,
                supportsText: cDict["supports_text"] as? Bool,
                supportsVision: cDict["supports_vision"] as? Bool,
                supportsVideo: cDict["supports_video"] as? Bool,
                supportsAudio: cDict["supports_audio"] as? Bool,
                family: cDict["family"] as? String,
                modality: cDict["modality"] as? String,
                cacheType: cDict["cache_type"] as? String,
                draftStrategy: cDict["draft_strategy"] as? String,
                drafterPath: cDict["drafter_path"] as? String,
                branchingBudget: cDict["branching_budget"] as? Int,
                blockSize: cDict["block_size"] as? Int
            )
        } else {
            capabilities = nil
        }

        // Top-level `model_family` hint (DSV4-era). Fallback to
        // `capabilities.family` for older bundles that carry family
        // under the capabilities block.
        let modelFamily =
            (json["model_family"] as? String) ?? capabilities?.family

        // New `chat` block — see JangChatConfig doc. Only present
        // on DSV4-era bundles; older bundles return nil here and
        // the runtime falls back to `capabilities` + model_type
        // heuristics. Parsed defensively (every field optional) so
        // partial adoption doesn't break loaders.
        let chat: JangChatConfig?
        if let chDict = json["chat"] as? [String: Any] {
            // reasoning subblock
            let reasoning: JangChatReasoning?
            if let rDict = chDict["reasoning"] as? [String: Any] {
                reasoning = JangChatReasoning(
                    supported: rDict["supported"] as? Bool,
                    modes: rDict["modes"] as? [String],
                    defaultMode: rDict["default_mode"] as? String,
                    thinkingStart: rDict["thinking_start"] as? String,
                    thinkingEnd: rDict["thinking_end"] as? String,
                    reasoningEffortLevels: parseEffortLevels(
                        rDict["reasoning_effort_levels"]),
                    dropEarlierReasoning: rDict["drop_earlier_reasoning"] as? Bool
                )
            } else { reasoning = nil }

            // tool_calling subblock
            let toolCalling: JangChatToolCalling?
            if let tDict = chDict["tool_calling"] as? [String: Any] {
                toolCalling = JangChatToolCalling(
                    supported: tDict["supported"] as? Bool,
                    parser: tDict["parser"] as? String,
                    dsmlToken: tDict["dsml_token"] as? String,
                    toolCallsBlock: tDict["tool_calls_block"] as? String,
                    invokeBlock: tDict["invoke_block"] as? String,
                    parameterBlock: tDict["parameter_block"] as? String,
                    toolOutputTag: tDict["tool_output_tag"] as? String
                )
            } else { toolCalling = nil }

            // sampling_defaults subblock
            let sampling: JangChatSamplingDefaults?
            if let sDict = chDict["sampling_defaults"] as? [String: Any] {
                sampling = JangChatSamplingDefaults(
                    temperature: floatValue(sDict["temperature"]),
                    topP: floatValue(sDict["top_p"]),
                    maxNewTokens: sDict["max_new_tokens"] as? Int
                )
            } else { sampling = nil }

            chat = JangChatConfig(
                encoder: chDict["encoder"] as? String,
                hasTokenizerChatTemplate:
                    chDict["has_tokenizer_chat_template"] as? Bool,
                bosToken: chDict["bos_token"] as? String,
                bosTokenId: chDict["bos_token_id"] as? Int,
                eosToken: chDict["eos_token"] as? String,
                eosTokenId: chDict["eos_token_id"] as? Int,
                roleTokens: chDict["role_tokens"] as? [String: String],
                reasoning: reasoning,
                toolCalling: toolCalling,
                samplingDefaults: sampling
            )
        } else {
            chat = nil
        }

        return JangConfig(
            format: format,
            formatVersion: formatVersion,
            quantization: quantization,
            mxtqBits: mxtqBits,
            routedExpertBitPlan: routedExpertBitPlan,
            sourceModel: sourceModel,
            architecture: architecture,
            runtime: runtime,
            capabilities: capabilities,
            modelFamily: modelFamily,
            chat: chat
        )
    }

    private static func parseMXTQBits(_ value: Any?) -> [String: Int] {
        if let bits = value as? Int {
            return ["routed_expert": bits]
        }
        guard let dict = value as? [String: Any] else { return [:] }
        var out: [String: Int] = [:]
        for (role, raw) in dict {
            if let bits = raw as? Int {
                out[role] = bits
            } else if let nested = raw as? [String: Any] {
                for (projection, nestedRaw) in nested {
                    if let bits = nestedRaw as? Int {
                        out["\(role).\(projection)"] = bits
                    }
                }
            }
        }
        return out
    }

    private static func parseRoutedExpertBitPlan(
        _ value: Any?,
        fallbackDefault: [String: Int]
    ) -> JangRoutedExpertBitPlan? {
        guard let dict = value as? [String: Any] else {
            return fallbackDefault.isEmpty
                ? nil
                : JangRoutedExpertBitPlan(defaultBits: normalizedProjectionBits(fallbackDefault))
        }

        let defaultBits = normalizedProjectionBits(
            parseMXTQBits(dict["default"]).isEmpty
                ? fallbackDefault
                : parseMXTQBits(dict["default"]))

        var layerOverrides: [Int: [String: Int]] = [:]
        if let overrides = dict["layer_overrides"] as? [String: Any] {
            for (layer, rawPlan) in overrides {
                guard let layerIndex = Int(layer) else { continue }
                let bits = normalizedProjectionBits(parseMXTQBits(rawPlan))
                if !bits.isEmpty {
                    layerOverrides[layerIndex] = bits
                }
            }
        } else if let overrides = dict["layerOverrides"] as? [String: Any] {
            for (layer, rawPlan) in overrides {
                guard let layerIndex = Int(layer) else { continue }
                let bits = normalizedProjectionBits(parseMXTQBits(rawPlan))
                if !bits.isEmpty {
                    layerOverrides[layerIndex] = bits
                }
            }
        }

        guard !defaultBits.isEmpty || !layerOverrides.isEmpty else { return nil }
        return JangRoutedExpertBitPlan(
            defaultBits: defaultBits,
            layerOverrides: layerOverrides)
    }

    private static func normalizedProjectionBits(_ bits: [String: Int]) -> [String: Int] {
        var out = bits
        for (key, value) in bits {
            let normalized = key
                .replacingOccurrences(of: "routed_expert.", with: "")
                .replacingOccurrences(of: "routed_experts.", with: "")
            if out[normalized] == nil {
                out[normalized] = value
            }
            if normalized == "gate" && out["gate_proj"] == nil {
                out["gate_proj"] = value
            } else if normalized == "up" && out["up_proj"] == nil {
                out["up_proj"] = value
            } else if normalized == "down" && out["down_proj"] == nil {
                out["down_proj"] = value
            }
        }
        return out
    }

    private static func parseSourceModel(_ raw: Any?) -> JangSourceModel {
        if let smDict = raw as? [String: Any] {
            let params: String
            if let s = smDict["parameters"] as? String {
                params = s
            } else if let n = smDict["parameters"] as? Int {
                params = String(n)
            } else {
                params = "0"
            }
            return JangSourceModel(
                name: smDict["name"] as? String ?? "",
                org: smDict["org"] as? String ?? "",
                architecture: smDict["architecture"] as? String ?? "",
                dtype: smDict["dtype"] as? String ?? "bfloat16",
                parameters: params
            )
        }
        guard let repo = raw as? String else { return JangSourceModel() }
        let trimmed = repo.trimmingCharacters(in: .whitespacesAndNewlines)
        let parts = trimmed.split(separator: "/", maxSplits: 1, omittingEmptySubsequences: true)
        if parts.count == 2 {
            return JangSourceModel(name: String(parts[1]), org: String(parts[0]))
        }
        return JangSourceModel(name: trimmed)
    }

    /// `reasoning_effort_levels` may contain `null` entries — the
    /// converter encodes "no effort override" as JSON null. Map them
    /// to Swift `nil` while preserving strings like `"max"` / `"high"`.
    private static func parseEffortLevels(_ raw: Any?) -> [String?]? {
        guard let arr = raw as? [Any] else { return nil }
        return arr.map { item in
            if item is NSNull { return nil }
            return item as? String
        }
    }

    // MARK: - Per-Layer Bit Width Inference

    /// Infer per-layer quantization from loaded JANG weights.
    ///
    /// JANG v2 stores different tensors at different bit widths. The bit width is
    /// inferred from tensor shapes: `actual_bits = (weight.shape[-1] * 32) / (scales.shape[-1] * group_size)`
    ///
    /// Returns a `BaseConfiguration.PerLayerQuantization` that the existing
    /// `loadWeights()` quantization path can use directly.
    /// Universal shape-based inference. Walks every `.scales` key in
    /// the bundle's weights, derives the actual `(bits, group_size)`
    /// from the `(weight, scales)` shape pair, and returns a per-layer
    /// quantization map. Works for any quantized bundle — JANG,
    /// JANGTQ-native, or stock MLX-quantized — because the math
    /// `weight.shape[-1] * 32 == bits * in_dim` and `scales.shape[-1] *
    /// group_size == in_dim` is the same regardless of how the bundle
    /// was produced.
    ///
    /// 2026-04-25: added because bundle `config.json` files can drift
    /// out of sync with the actual safetensors (e.g., a re-stamped
    /// `bits: 8` block while the routed-MoE codebook is still bits=2,
    /// or a converter bug emits the wrong override). Trusting the
    /// shape always gives a correct dequant; trusting config.json
    /// produces silent corruption (wrong dequant constants → garbage
    /// activations) or hard fatal errors (codebook miss).
    ///
    /// Resolution priority for the SHARED default (`bits`, `gs`):
    ///
    ///   1. Caller-supplied `defaultBits` / `defaultGroupSize`
    ///      (typically from config.json's top-level `quantization`).
    ///   2. The MOST FREQUENT (bits, gs) pair across all walked layers.
    ///   3. Hard-coded `(4, 64)` fallback.
    ///
    /// Per-layer entries are emitted only for layers whose
    /// shape-inferred quant differs from the chosen default. Layers
    /// whose shapes don't yield a valid `(bits, gs)` (e.g., MXTQ
    /// codebook entries that don't carry `.scales`) are skipped — they
    /// were never going to be quantized via this path anyway.
    public static func inferPerLayerQuantizationFromShapes(
        weights: [String: MLXArray],
        defaultBits: Int? = nil,
        defaultGroupSize: Int? = nil,
        defaultMode: QuantizationMode = .affine,
        bitWidthsHint: [Int] = []
    ) -> BaseConfiguration.PerLayerQuantization? {
        // Find every base path that has a `.scales` companion.
        var quantizedLayers = Set<String>()
        for key in weights.keys where key.hasSuffix(".scales") {
            quantizedLayers.insert(String(key.dropLast(".scales".count)))
        }
        guard !quantizedLayers.isEmpty else { return nil }

        // Walk shapes. The `bitWidthsHint` (if present) constrains the
        // ambiguous fallback search. If the caller didn't pass one,
        // prefer high-bit candidates first since the converter classify
        // rule puts attention/embed/lm_head/shared at the highest
        // available bits — matches "(8,32) first" pref order from the
        // jang_tools runtime fix design.
        let hintToUse: [Int] =
            bitWidthsHint.isEmpty ? [8, 6, 5, 4, 3, 2] : bitWidthsHint

        var inferred = [String: (bits: Int, groupSize: Int, mode: QuantizationMode)]()
        for basePath in quantizedLayers {
            guard let weightArray = weights[basePath + ".weight"],
                let scalesArray = weights[basePath + ".scales"]
            else { continue }
            let (bits, gs) = inferBitWidthAndGroupSize(
                weight: weightArray, scales: scalesArray,
                knownGroupSize: defaultGroupSize,
                bitWidthsUsed: hintToUse)
            let mode = weights[basePath + ".biases"] == nil ? defaultMode : .affine
            inferred[basePath] = (bits, gs, mode)
        }
        guard !inferred.isEmpty else { return nil }

        // Pick the shared default. Caller's hint wins when present;
        // otherwise we use the most frequent (bits, gs) pair.
        let chosenDefault: (bits: Int, groupSize: Int)
        if let b = defaultBits, let gs = defaultGroupSize {
            chosenDefault = (b, gs)
        } else {
            var counts = [String: (count: Int, bits: Int, gs: Int)]()
            for (_, t) in inferred {
                let k = "\(t.bits)/\(t.groupSize)"
                let prev = counts[k] ?? (0, t.bits, t.groupSize)
                counts[k] = (prev.count + 1, prev.bits, prev.gs)
            }
            if let top = counts.values.max(by: { $0.count < $1.count }) {
                chosenDefault = (top.bits, top.gs)
            } else {
                chosenDefault = (4, 64)
            }
        }

        var perLayer = [String: BaseConfiguration.QuantizationOption]()
        for (path, t) in inferred {
            if t.bits != chosenDefault.bits
                || t.groupSize != chosenDefault.groupSize
                || t.mode != defaultMode
            {
                perLayer[path] = .quantize(
                    BaseConfiguration.Quantization(
                        groupSize: t.groupSize, bits: t.bits, mode: t.mode))
            }
        }
        return BaseConfiguration.PerLayerQuantization(
            quantization: BaseConfiguration.Quantization(
                groupSize: chosenDefault.groupSize, bits: chosenDefault.bits, mode: defaultMode),
            perLayerQuantization: perLayer
        )
    }

    public static func inferPerLayerQuantization(
        weights: [String: MLXArray],
        jangConfig: JangConfig,
        hiddenSizeHint: Int? = nil,
        hiddenSizePerLayerInputHint: Int? = nil,
        linearAttnValueDimHint: Int? = nil,
        expertIntermediateSizeHint: Int? = nil,
        validInDims: Set<Int> = [],
        attentionOutputDimHints: Set<Int> = [],
        declaredDefaultQuantization: BaseConfiguration.Quantization? = nil,
        declaredPerLayerQuantization: BaseConfiguration.PerLayerQuantization? = nil
    ) -> BaseConfiguration.PerLayerQuantization {
        // JANGTQ bundles use two independent bit namespaces:
        //   - `mxtq_bits` / `routed_expert_bits` for tq_packed routed experts.
        //   - config.json::quantization for ordinary affine dense/router weights.
        // If config.json already decoded a top-level affine quantization block,
        // use that as the declared fallback for shape comparison. The actual
        // per-leaf load entries below remain shape-authoritative.
        let groupSize = declaredDefaultQuantization?.groupSize
            ?? jangConfig.quantization.blockSize
        var perLayer = [String: BaseConfiguration.QuantizationOption]()

        let defaultBits = declaredDefaultQuantization?.bits
            ?? (jangConfig.quantization.bitWidthsUsed.min() ?? 4)
        let defaultMode = declaredDefaultQuantization?.mode ?? .affine
        let bitWidthsUsed = Array(Set(
            jangConfig.quantization.bitWidthsUsed + [defaultBits]
        )).sorted()

        // Group weight keys by their base path (strip .weight/.scales/.biases suffix)
        var quantizedLayers = Set<String>()
        for key in weights.keys {
            if key.hasSuffix(".scales") {
                let basePath = String(key.dropLast(".scales".count))
                quantizedLayers.insert(basePath)
            }
        }

        func declaredMXTQRoleBits(for basePath: String) -> Int? {
            let roles = jangConfig.mxtqBits
            let layerIndex = layerIndexFromQuantizedBasePath(basePath)
            if basePath.contains(".switch_mlp.")
                || basePath.contains(".switch_glu.")
            {
                if basePath.hasSuffix(".gate_proj") {
                    return layerIndex.flatMap {
                        jangConfig.routedExpertBitPlan?.bits(
                            layerIndex: $0, projection: "gate_proj")
                    }
                        ?? roles["routed_expert.gate_proj"] ?? roles["gate_proj"]
                        ?? roles["routed_expert"]
                }
                if basePath.hasSuffix(".up_proj") {
                    return layerIndex.flatMap {
                        jangConfig.routedExpertBitPlan?.bits(
                            layerIndex: $0, projection: "up_proj")
                    }
                        ?? roles["routed_expert.up_proj"] ?? roles["up_proj"]
                        ?? roles["routed_expert"]
                }
                if basePath.hasSuffix(".down_proj") {
                    return layerIndex.flatMap {
                        jangConfig.routedExpertBitPlan?.bits(
                            layerIndex: $0, projection: "down_proj")
                    }
                        ?? roles["routed_expert.down_proj"] ?? roles["down_proj"]
                        ?? roles["routed_expert"]
                }
            }

            guard !roles.isEmpty else { return nil }

            if basePath.hasSuffix("lm_head") {
                return roles["lm_head"] ?? roles["embed_lm_head"]
            }
            if basePath.hasSuffix("embed_tokens")
                || basePath.hasSuffix("embeddings")
                || basePath.hasSuffix("embed")
            {
                return roles["embed_tokens"] ?? roles["embed_lm_head"]
            }
            if basePath.hasSuffix(".self_attn.q_proj")
                || basePath.hasSuffix(".self_attn.k_proj")
                || basePath.hasSuffix(".self_attn.v_proj")
                || basePath.hasSuffix(".self_attn.o_proj")
                || basePath.hasSuffix(".attn.q_proj")
                || basePath.hasSuffix(".attn.k_proj")
                || basePath.hasSuffix(".attn.v_proj")
                || basePath.hasSuffix(".attn.o_proj")
                || basePath.hasSuffix(".mixer.q_proj")
                || basePath.hasSuffix(".mixer.k_proj")
                || basePath.hasSuffix(".mixer.v_proj")
                || basePath.hasSuffix(".mixer.o_proj")
            {
                return roles["attention"]
            }
            if basePath.hasSuffix(".mixer.in_proj")
                || basePath.hasSuffix(".mixer.out_proj")
                || basePath.hasSuffix(".linear_attn.in_proj_qkv")
                || basePath.hasSuffix(".linear_attn.in_proj_z")
                || basePath.hasSuffix(".linear_attn.in_proj_a")
                || basePath.hasSuffix(".linear_attn.in_proj_b")
                || basePath.hasSuffix(".linear_attn.out_proj")
            {
                return roles["mamba_proj"] ?? roles["mamba_projection"] ?? roles["linear_attn"]
            }
            if basePath.contains(".shared_experts.")
                || basePath.contains(".shared_expert.")
            {
                return roles["shared_expert"]
            }
            if basePath.contains(".switch_mlp.")
                || basePath.contains(".switch_glu.")
            {
                if basePath.hasSuffix(".gate_proj") {
                    return roles["routed_expert.gate_proj"] ?? roles["routed_expert"]
                }
                if basePath.hasSuffix(".up_proj") {
                    return roles["routed_expert.up_proj"] ?? roles["routed_expert"]
                }
                if basePath.hasSuffix(".down_proj") {
                    return roles["routed_expert.down_proj"] ?? roles["routed_expert"]
                }
            }
            return nil
        }

        func declaredQuantization(for basePath: String) -> BaseConfiguration.Quantization? {
            func variants(_ key: String) -> [String] {
                var out = [key]
                if key.contains(".attn.") || key.hasSuffix(".attn") {
                    out.append(key.replacingOccurrences(of: ".attn.", with: ".self_attn."))
                    if key.hasSuffix(".attn") {
                        out.append(String(key.dropLast(".attn".count)) + ".self_attn")
                    }
                }
                if key.hasPrefix("language_model.model.") {
                    out.append(String(key.dropFirst("language_model.".count)))
                } else if key.hasPrefix("language_model.") {
                    out.append(String(key.dropFirst("language_model.".count)))
                } else {
                    out.append("model.\(key)")
                    out.append("language_model.\(key)")
                    out.append("language_model.model.\(key)")
                }
                return Array(Set(out))
            }

            if let declaredPerLayerQuantization {
                for key in variants(basePath) {
                    if let declared = declaredPerLayerQuantization.perLayerQuantization[key] {
                        switch declared {
                        case .quantize(let quantization):
                            return quantization
                        case .skip:
                            return nil
                        }
                    }
                }
            }
            if let roleBits = declaredMXTQRoleBits(for: basePath) {
                return BaseConfiguration.Quantization(
                    groupSize: groupSize, bits: roleBits, mode: defaultMode)
            }
            return declaredPerLayerQuantization?.quantization
                ?? declaredDefaultQuantization
        }

        func layerIndexFromQuantizedBasePath(_ basePath: String) -> Int? {
            guard let range = basePath.range(
                of: #"(?:^|\.)(?:model\.)?layers\.(\d+)\."#,
                options: .regularExpression)
            else {
                return nil
            }
            let match = String(basePath[range])
            guard let digitsRange = match.range(
                of: #"\d+"#,
                options: .regularExpression)
            else {
                return nil
            }
            return Int(match[digitsRange])
        }

        var disagreementCount = 0
        var sampleDeclared: (Int, Int)? = nil
        var sampleInferred: (Int, Int)? = nil

        // Shape truth wins for every leaf. Emit an override even when the
        // inferred pair matches the declared default so downstream lookup never
        // falls through to stale top-level metadata for a path variant.
        for basePath in quantizedLayers.sorted() {
            guard let weightArray = weights[basePath + ".weight"],
                let scalesArray = weights[basePath + ".scales"]
            else {
                continue
            }

            let packedDim = weightArray.shape.last ?? 0
            let numGroups = scalesArray.shape.last ?? 1
            let (bits, inferredGroupSize): (Int, Int)

            let isHiddenAnchor =
                basePath.hasSuffix("embed_tokens")
                || basePath.hasSuffix("embed")
                || basePath.hasSuffix("lm_head")
            let isHiddenInputProjection =
                basePath.hasSuffix(".linear_attn.in_proj_qkv")
                || basePath.hasSuffix(".linear_attn.in_proj_z")
                || basePath.hasSuffix(".linear_attn.in_proj_a")
                || basePath.hasSuffix(".linear_attn.in_proj_b")
                || basePath.hasSuffix(".self_attn.q_proj")
                || basePath.hasSuffix(".self_attn.k_proj")
                || basePath.hasSuffix(".self_attn.v_proj")
                || basePath.hasSuffix(".attn.q_proj")
                || basePath.hasSuffix(".attn.k_proj")
                || basePath.hasSuffix(".attn.v_proj")
                || basePath.hasSuffix(".mlp.gate_proj")
                || basePath.hasSuffix(".mlp.up_proj")
                || basePath.hasSuffix(".switch_mlp.gate_proj")
                || basePath.hasSuffix(".switch_mlp.up_proj")
                || basePath.hasSuffix(".switch_glu.gate_proj")
                || basePath.hasSuffix(".switch_glu.up_proj")
                || basePath.hasSuffix(".shared_expert.gate_proj")
                || basePath.hasSuffix(".shared_expert.up_proj")
            let isMTPFusionFC =
                basePath.hasSuffix(".mtp.fc")
                || basePath.hasSuffix("mtp.fc")
            let isLinearAttnOutputProjection =
                basePath.hasSuffix(".linear_attn.out_proj")
            let isAttentionOutputProjection =
                basePath.hasSuffix(".self_attn.o_proj")
                || basePath.hasSuffix(".attn.o_proj")
                || basePath.hasSuffix(".mixer.o_proj")
            let isZayaCCAOutputProjection =
                basePath.hasSuffix(".sub.o_proj")
            let isPerLayerProjection =
                basePath.hasSuffix(".per_layer_projection")
            let isPerLayerModelProjection =
                basePath.hasSuffix(".per_layer_model_projection")
            let isExpertDownProjection =
                basePath.hasSuffix("switch_mlp.down_proj")
                || basePath.hasSuffix("switch_glu.down_proj")
                || basePath.hasSuffix("shared_expert.down_proj")

            func inferFromUniqueValidInDim() -> (bits: Int, groupSize: Int)? {
                guard !validInDims.isEmpty else { return nil }
                let preferred: [(Int, Int)] = [
                    (8, 32), (8, 64), (8, 128),
                    (4, 32), (4, 64), (4, 128),
                    (2, 32), (2, 64), (2, 128),
                    (3, 32), (3, 64), (3, 128),
                    (5, 32), (5, 64), (5, 128),
                    (6, 32), (6, 64), (6, 128),
                ]
                var matches: [(bits: Int, groupSize: Int, inDim: Int)] = []
                for (candidateBits, candidateGroupSize) in preferred {
                    guard candidateBits > 0, (packedDim * 32) % candidateBits == 0 else {
                        continue
                    }
                    let inputDim = (packedDim * 32) / candidateBits
                    guard validInDims.contains(inputDim), inputDim % numGroups == 0 else {
                        continue
                    }
                    let impliedGroupSize = inputDim / numGroups
                    if impliedGroupSize == candidateGroupSize {
                        matches.append((candidateBits, candidateGroupSize, inputDim))
                    }
                }
                let uniqueInputDims = Set(matches.map(\.inDim))
                guard uniqueInputDims.count == 1, let first = matches.first else {
                    return nil
                }
                return (first.bits, first.groupSize)
            }

            // Trust the bundle's EXPLICIT declared per-module quant when it unpacks the
            // real `.weight`/`.scales` to an inputDim that is a known model dimension.
            // On ambiguous packed widths (512 → bits ∈ {2,4,8}) the shape walk can pick
            // a different, WRONG bit width: Laguna-M.1 ships 4-bit dense MLP, the walk
            // re-stamped 8-bit, and the 4-bit weights dequantized to a degenerate (empty)
            // output that collapsed the forward. A self-consistent declaration whose
            // inputDim ∈ validInDims is the author's verified intent. A genuinely stale
            // config (claims 8-bit for 4-bit-packed) unpacks to a NON-model inputDim,
            // fails this check, and still falls through to the shape walk.
            if let dq = declaredQuantization(for: basePath),
               dq.bits > 0, numGroups > 0, (packedDim * 32) % dq.bits == 0,
               case let declaredInputDim = (packedDim * 32) / dq.bits,
               declaredInputDim % numGroups == 0,
               declaredInputDim / numGroups == dq.groupSize,
               validInDims.contains(declaredInputDim)
            {
                (bits, inferredGroupSize) = (dq.bits, dq.groupSize)
            } else if (isHiddenAnchor || isHiddenInputProjection),
               let hiddenSize = hiddenSizeHint, hiddenSize > 0
            {
                (bits, inferredGroupSize) = inferBitWidthAndGroupSize(
                    packedDim: packedDim,
                    numGroups: numGroups,
                    knownGroupSize: groupSize,
                    bitWidthsUsed: bitWidthsUsed,
                    expectedInDim: hiddenSize)
            } else if isMTPFusionFC,
                      let hiddenSize = hiddenSizeHint, hiddenSize > 0
            {
                (bits, inferredGroupSize) = inferBitWidthAndGroupSize(
                    packedDim: packedDim,
                    numGroups: numGroups,
                    knownGroupSize: groupSize,
                    bitWidthsUsed: bitWidthsUsed,
                    expectedInDim: hiddenSize * 2)
            } else if isLinearAttnOutputProjection,
                      let valueDim = linearAttnValueDimHint, valueDim > 0
            {
                (bits, inferredGroupSize) = inferBitWidthAndGroupSize(
                    packedDim: packedDim,
                    numGroups: numGroups,
                    knownGroupSize: groupSize,
                    bitWidthsUsed: bitWidthsUsed,
                    expectedInDim: valueDim)
            } else if isAttentionOutputProjection,
                      !attentionOutputDimHints.isEmpty
            {
                let candidates = attentionOutputDimHints.sorted()
                var picked: (bits: Int, groupSize: Int)? = nil
                for dim in candidates where dim > 0 {
                    let inferred = inferBitWidthAndGroupSize(
                        packedDim: packedDim,
                        numGroups: numGroups,
                        knownGroupSize: groupSize,
                        bitWidthsUsed: bitWidthsUsed,
                        expectedInDim: dim)
                    let inputDim = (packedDim * 32) / inferred.bits
                    if inputDim == dim {
                        picked = inferred
                        break
                    }
                }
                if let picked {
                    (bits, inferredGroupSize) = picked
                } else {
                    (bits, inferredGroupSize) = inferBitWidthAndGroupSize(
                        weight: weightArray,
                        scales: scalesArray,
                        knownGroupSize: groupSize,
                        bitWidthsUsed: bitWidthsUsed)
                }
            } else if isZayaCCAOutputProjection,
                      let hiddenSize = hiddenSizeHint, hiddenSize > 1
            {
                // ZAYA text sanitizes the CCA attention block under `sub`.
                // Its `o_proj` consumes the 8-head CCA output (1024 for the
                // 2048-wide 8B artifacts), not the full hidden width. Shape
                // ambiguity otherwise maps `[2048,256]` to 4-bit/64 and makes
                // quantized_matmul expect a 2048-wide input at runtime.
                let ccaOutputDim = hiddenSize / 2
                (bits, inferredGroupSize) = inferBitWidthAndGroupSize(
                    packedDim: packedDim,
                    numGroups: numGroups,
                    knownGroupSize: groupSize,
                    bitWidthsUsed: bitWidthsUsed,
                    expectedInDim: ccaOutputDim)
            } else if isPerLayerProjection,
                      let perLayerInputDim = hiddenSizePerLayerInputHint,
                      perLayerInputDim > 0
            {
                (bits, inferredGroupSize) = inferBitWidthAndGroupSize(
                    packedDim: packedDim,
                    numGroups: numGroups,
                    knownGroupSize: groupSize,
                    bitWidthsUsed: bitWidthsUsed,
                    expectedInDim: perLayerInputDim)
            } else if isPerLayerModelProjection,
                      let hiddenSize = hiddenSizeHint, hiddenSize > 0
            {
                (bits, inferredGroupSize) = inferBitWidthAndGroupSize(
                    packedDim: packedDim,
                    numGroups: numGroups,
                    knownGroupSize: groupSize,
                    bitWidthsUsed: bitWidthsUsed,
                    expectedInDim: hiddenSize)
            } else if isExpertDownProjection,
                      let expertIntermediateSize = expertIntermediateSizeHint,
                      expertIntermediateSize > 0
            {
                (bits, inferredGroupSize) = inferBitWidthAndGroupSize(
                    packedDim: packedDim,
                    numGroups: numGroups,
                    knownGroupSize: groupSize,
                    bitWidthsUsed: bitWidthsUsed,
                    expectedInDim: expertIntermediateSize)
            } else if let picked = inferFromUniqueValidInDim() {
                (bits, inferredGroupSize) = picked
            } else if isExpertDownProjection && !validInDims.isEmpty {
                let preferred: [(Int, Int)] = [
                    (8, 32), (8, 64), (8, 128),
                    (4, 32), (4, 64), (4, 128),
                    (2, 32), (2, 64), (2, 128),
                    (3, 32), (3, 64), (3, 128),
                    (5, 32), (5, 64), (5, 128),
                    (6, 32), (6, 64), (6, 128),
                ]
                var picked: (Int, Int)? = nil
                for (candidateBits, candidateGroupSize) in preferred {
                    guard (packedDim * 32) % candidateBits == 0 else { continue }
                    let inputDim = (packedDim * 32) / candidateBits
                    guard validInDims.contains(inputDim), inputDim % numGroups == 0 else {
                        continue
                    }
                    let impliedGroupSize = inputDim / numGroups
                    if impliedGroupSize == candidateGroupSize {
                        picked = (candidateBits, candidateGroupSize)
                        break
                    }
                }
                if let picked {
                    (bits, inferredGroupSize) = picked
                } else {
                    (bits, inferredGroupSize) = inferBitWidthAndGroupSize(
                        weight: weightArray,
                        scales: scalesArray,
                        knownGroupSize: groupSize,
                        bitWidthsUsed: bitWidthsUsed)
                }
            } else {
                (bits, inferredGroupSize) = inferBitWidthAndGroupSize(
                    weight: weightArray,
                    scales: scalesArray,
                    knownGroupSize: groupSize,
                    bitWidthsUsed: bitWidthsUsed)
            }

            let mode = weights[basePath + ".biases"] == nil ? defaultMode : .affine
            let declaredForLayer = declaredQuantization(for: basePath)
            let declaredMatches =
                declaredForLayer?.bits == bits
                && declaredForLayer?.groupSize == inferredGroupSize
                && (declaredForLayer?.mode ?? defaultMode) == mode
            let layerHasMetadataDrift =
                declaredForLayer == nil
                || !declaredMatches

            if layerHasMetadataDrift {
                disagreementCount += 1
                if sampleDeclared == nil {
                    sampleDeclared = (
                        declaredForLayer?.bits ?? defaultBits,
                        declaredForLayer?.groupSize ?? groupSize
                    )
                    sampleInferred = (bits, inferredGroupSize)
                }
            }

            perLayer[basePath] = .quantize(
                BaseConfiguration.Quantization(
                    groupSize: inferredGroupSize,
                    bits: bits,
                    mode: mode))
        }

        if disagreementCount > 0,
           let declared = sampleDeclared,
           let inferred = sampleInferred
        {
            let plural = disagreementCount == 1 ? "" : "s"
            let line = (
                "[JangLoader] config-metadata mismatch patched in-memory: "
                    + "declared (bits=\(declared.0), gs=\(declared.1)) "
                    + "-> shape-inferred (bits=\(inferred.0), gs=\(inferred.1)), "
                    + "\(disagreementCount) per-layer override\(plural) applied.\n"
            )
            FileHandle.standardError.write(Data(line.utf8))
        }

        return BaseConfiguration.PerLayerQuantization(
            quantization: BaseConfiguration.Quantization(
                groupSize: groupSize, bits: defaultBits, mode: defaultMode),
            perLayerQuantization: perLayer
        )
    }

    /// Infer bit width from weight and scales tensor shapes using a fixed group size.
    public static func inferBitWidth(
        weight: MLXArray, scales: MLXArray, groupSize: Int
    ) -> Int {
        inferBitWidthAndGroupSize(weight: weight, scales: scales, knownGroupSize: groupSize).bits
    }

    /// Infer BOTH bit width and group size from weight and scales tensor shapes.
    ///
    /// A JANG quantized tensor has:
    ///   weight.shape[-1] = (in_dim * bits) / 32   (packed into uint32)
    ///   scales.shape[-1] = in_dim / groupSize     (one scale per group per row)
    ///
    /// From these two equations:
    ///   in_dim = scales.shape[-1] * groupSize
    ///   bits   = weight.shape[-1] * 32 / in_dim
    ///
    /// With knownGroupSize this is a direct calculation. Without it, the answer
    /// is not unique from shapes alone — multiple (bits, groupSize) pairs can
    /// produce the same packed shape. In that case we require the provided
    /// `bitWidthsUsed` from the JANG config to disambiguate, preferring
    /// higher bits first (JANG CRITICAL tier uses the highest bits).
    public static func inferBitWidthAndGroupSize(
        weight: MLXArray, scales: MLXArray, knownGroupSize: Int? = nil,
        bitWidthsUsed: [Int] = []
    ) -> (bits: Int, groupSize: Int) {
        inferBitWidthAndGroupSize(
            packedDim: weight.shape.last ?? 0,
            numGroups: scales.shape.last ?? 1,
            knownGroupSize: knownGroupSize,
            bitWidthsUsed: bitWidthsUsed)
    }

    public static func inferBitWidthAndGroupSize(
        packedDim: Int, numGroups: Int,
        knownGroupSize: Int? = nil,
        bitWidthsUsed: [Int] = []
    ) -> (bits: Int, groupSize: Int) {
        guard packedDim > 0 && numGroups > 0 else { return (4, knownGroupSize ?? 64) }

        // Affine quantized_matmul only supports bit widths {2,3,4,5,6,8}. JANGTQ
        // `mxtq_bits` can carry non-affine sentinels (e.g. `norms_router: 16` =
        // fp16 passthrough) that leak into `bitWidthsUsed` via topLevelBitWidths.
        // 16 is never a valid affine packing width; if selected it yields a
        // QuantizedLinear whose expanded in-dim is half the real one, crashing
        // quantized_matmul (MiniMax fused qkv_proj: (8,64) truth vs (16,32)
        // picked because gs=32 is the wrong per-layer group size). Filter to the
        // physically valid set so an impossible width can never be returned.
        let supportedAffineBits: [Int] = [2, 3, 4, 5, 6, 8]
        let affineBitWidths = bitWidthsUsed.filter(supportedAffineBits.contains)

        if let knownGroupSize, knownGroupSize > 0 {
            let inputDim = numGroups * knownGroupSize
            let packedBits = packedDim * 32
            if inputDim > 0, packedBits % inputDim == 0 {
                let bits = packedBits / inputDim
                let validBits = affineBitWidths.isEmpty ? supportedAffineBits : affineBitWidths
                if bits > 0, validBits.contains(bits) {
                    return (bits, knownGroupSize)
                }
            }
        }

        let preferred: [(Int, Int)] = [
            (8, 32), (8, 64), (8, 128),
            (4, 32), (4, 64), (4, 128),
            (2, 32), (2, 64), (2, 128),
            (3, 32), (3, 64), (3, 128),
            (5, 32), (5, 64), (5, 128),
            (6, 32), (6, 64), (6, 128),
        ]
        for (bits, groupSize) in preferred {
            guard (packedDim * 32) % bits == 0 else { continue }
            let inputDim = (packedDim * 32) / bits
            guard inputDim > 0, inputDim % numGroups == 0 else { continue }
            if inputDim / numGroups == groupSize {
                return (bits, groupSize)
            }
        }

        let candidates = affineBitWidths.isEmpty
            ? supportedAffineBits.sorted(by: >)
            : affineBitWidths.sorted(by: >)
        for bits in candidates {
            guard bits > 0, (packedDim * 32) % bits == 0 else { continue }
            let inputDim = (packedDim * 32) / bits
            guard inputDim > 0, inputDim % numGroups == 0 else { continue }
            return (bits, inputDim / numGroups)
        }

        return (4, knownGroupSize ?? 64)
    }

    public static func inferBitWidthAndGroupSize(
        packedDim: Int, numGroups: Int,
        knownGroupSize: Int? = nil,
        bitWidthsUsed: [Int] = [],
        expectedInDim: Int
    ) -> (bits: Int, groupSize: Int) {
        guard packedDim > 0 && numGroups > 0 && expectedInDim > 0 else {
            return inferBitWidthAndGroupSize(
                packedDim: packedDim,
                numGroups: numGroups,
                knownGroupSize: knownGroupSize,
                bitWidthsUsed: bitWidthsUsed)
        }

        let preferred: [(Int, Int)] = [
            (8, 32), (8, 64), (8, 128),
            (4, 32), (4, 64), (4, 128),
            (2, 32), (2, 64), (2, 128),
            (3, 32), (3, 64), (3, 128),
            (5, 32), (5, 64), (5, 128),
            (6, 32), (6, 64), (6, 128),
        ]
        for (bits, groupSize) in preferred {
            guard (packedDim * 32) % bits == 0 else { continue }
            let inputDim = (packedDim * 32) / bits
            guard inputDim == expectedInDim, inputDim % numGroups == 0 else {
                continue
            }
            if inputDim / numGroups == groupSize {
                return (bits, groupSize)
            }
        }

        return inferBitWidthAndGroupSize(
            packedDim: packedDim,
            numGroups: numGroups,
            knownGroupSize: knownGroupSize,
            bitWidthsUsed: bitWidthsUsed)
    }

    // MARK: - MoE Gate Dequantization

    /// Dequantize MoE gate/router weights from quantized uint32 to float.
    ///
    /// JANG quantizes MoE gate weights at CRITICAL tier (highest available bits)
    /// for routing precision, but the model expects them as plain float Linear
    /// (not QuantizedLinear). This function detects gate weights that have
    /// .scales/.biases companions and dequantizes them in-place.
    ///
    /// Gate patterns matched:
    /// - `.gate.weight` (not `.gate_proj.weight`) — Nemotron, MiniMax
    /// - `.mlp.gate.weight` — Qwen3.5 MoE, general MoE
    /// - `.mixer.gate.weight` — Nemotron-H
    /// - `.router.proj.weight` — Gemma4 (already handled separately)
    public static func dequantizeMoEGates(
        weights: inout [String: MLXArray],
        groupSize: Int,
        bitWidthsUsed: [Int] = [],
        hiddenSizeHint: Int? = nil
    ) {
        // Find gate weight keys that have .scales companion (meaning they're quantized)
        var gateBasePaths = Set<String>()

        for key in weights.keys {
            // Match gate patterns but NOT gate_proj (which is an expert MLP weight)
            if key.hasSuffix(".gate.scales") && !key.contains("gate_proj") && !key.contains("gate_up") {
                let basePath = String(key.dropLast(".scales".count))
                gateBasePaths.insert(basePath)
            }
            // Also match shared_expert_gate (Qwen3.5 MoE)
            if key.hasSuffix(".shared_expert_gate.scales") {
                let basePath = String(key.dropLast(".scales".count))
                gateBasePaths.insert(basePath)
            }
        }

        for basePath in gateBasePaths {
            guard let gateWeight = weights[basePath + ".weight"],
                let gateScales = weights[basePath + ".scales"]
            else { continue }

            let gateBiases = weights[basePath + ".biases"]

            let packedDim = gateWeight.shape.last ?? 0
            let numGroups = gateScales.shape.last ?? 1

            let inferred = hiddenSizeHint.flatMap { hiddenSize -> (bits: Int, groupSize: Int)? in
                guard hiddenSize > 0 else { return nil }
                return inferBitWidthAndGroupSize(
                    packedDim: packedDim,
                    numGroups: numGroups,
                    knownGroupSize: groupSize,
                    bitWidthsUsed: bitWidthsUsed,
                    expectedInDim: hiddenSize)
            } ?? inferBitWidthAndGroupSize(
                packedDim: packedDim,
                numGroups: numGroups,
                knownGroupSize: groupSize,
                bitWidthsUsed: bitWidthsUsed)

            // Dequantize to float32 for routing precision
            let dequantized = MLX.dequantized(
                gateWeight, scales: gateScales, biases: gateBiases,
                groupSize: inferred.groupSize, bits: inferred.bits)

            // Replace quantized gate with float version, remove scales/biases
            weights[basePath + ".weight"] = dequantized.asType(.float32)
            weights.removeValue(forKey: basePath + ".scales")
            weights.removeValue(forKey: basePath + ".biases")
        }
    }

    // MARK: - V1 Format Support

    /// Check if a model directory contains v1 format JANG weights.
    public static func hasV1Weights(at modelPath: URL) -> Bool {
        guard
            let files = try? FileManager.default.contentsOfDirectory(
                at: modelPath, includingPropertiesForKeys: nil)
        else { return false }
        return files.contains {
            $0.pathExtension == "safetensors" && $0.lastPathComponent.contains(".jang.")
        }
    }

    /// Load JANG v1 format weights (legacy uint8 → uint32 repacking).
    public static func loadV1Weights(at modelPath: URL) throws -> [String: MLXArray] {
        let fm = FileManager.default
        let files =
            try fm.contentsOfDirectory(at: modelPath, includingPropertiesForKeys: nil)
            .filter {
                $0.pathExtension == "safetensors" && $0.lastPathComponent.contains(".jang.")
            }

        guard !files.isEmpty else {
            throw JangLoaderError.loadFailed(
                "No .jang.safetensors files found at \(modelPath.path)")
        }

        var allWeights: [String: MLXArray] = [:]
        for file in files {
            let (weights, _) = try loadArraysAndMetadata(url: file)
            for (key, array) in weights {
                if array.dtype == .uint8 {
                    allWeights[key] = repackUint8ToUint32(array)
                } else {
                    allWeights[key] = array
                }
            }
        }
        return allWeights
    }

    /// Repack a uint8 array to uint32 by packing groups of 4 bytes (little-endian).
    private static func repackUint8ToUint32(_ array: MLXArray) -> MLXArray {
        let shape = array.shape
        let lastDim = shape.last ?? 0
        guard lastDim % 4 == 0 else { return array.asType(.uint32) }

        var newShape = shape
        newShape[newShape.count - 1] = lastDim / 4
        newShape.append(4)

        let reshaped = array.reshaped(newShape)
        let b0 = reshaped[0..., 0].asType(.uint32)
        let b1 = reshaped[0..., 1].asType(.uint32) << 8
        let b2 = reshaped[0..., 2].asType(.uint32) << 16
        let b3 = reshaped[0..., 3].asType(.uint32) << 24
        return b0 | b1 | b2 | b3
    }

    // MARK: - Helpers

    private static func floatValue(_ value: Any?) -> Float? {
        if let d = value as? Double { return Float(d) }
        if let f = value as? Float { return f }
        if let i = value as? Int { return Float(i) }
        return nil
    }
}

// MARK: - Errors

public enum JangLoaderError: Error, LocalizedError, Sendable {
    case configNotFound(String)
    case invalidConfig(String)
    case unsupportedVersion(String)
    case loadFailed(String)

    public var errorDescription: String? {
        switch self {
        case .configNotFound(let path): return "JANG config not found at: \(path)"
        case .invalidConfig(let msg): return "Invalid JANG config: \(msg)"
        case .unsupportedVersion(let ver): return "Unsupported JANG version: \(ver)"
        case .loadFailed(let msg): return "JANG load failed: \(msg)"
        }
    }
}
